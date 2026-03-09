
"""
mesh.py (updated)

Keeps the existing 1D pipeline:
- build 1D meshes from branchingData CSV via branch.py
- parse 1D *.vtp outputs and generate 1D checkpoint files (as before)
- returns terminal point coordinates (outlet_coords) from the 1D branched mesh boundary

Replaces ONLY the 3D meshing + tagging:
- reads an existing 3D volumetric mesh (VTU) for the tissue/bioreactor
- converts VTU -> XDMF (for FEniCSx)
- creates 2D facet MeshTags for:
    * outer wall facets
    * arterial concave patch (seeded near coords_inlet)
    * venous concave patch (seeded near coords_outlet)
    * inlet terminal faces (one tag per inlet branch terminal point)
    * outlet terminal faces (one tag per outlet branch terminal point)
- supports either duplicating/translating a single VTU into 4 wells, or using 4 VTUs.

Outputs (per well):
- bioreactor_well{i}.xdmf
- mesh_tags_well{i}.xdmf   (facet meshtags, fdim=2)

NOTE: This script assumes your VTU is a tetrahedral volume mesh.
"""

from __future__ import annotations

import os, subprocess, shutil
import re
import glob
import logging
from pathlib import Path
from send2trash import send2trash
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import gmsh
import meshio
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree

from mpi4py import MPI
import adios4dolfinx
import dolfinx as dfx
from dolfinx.io import XDMFFile
from dolfinx import mesh as dmesh
from basix.ufl import element

# ---------------------------------------------------------------------
# Local imports (branch.py)
# ---------------------------------------------------------------------
try:
    from geometry import branch
except ImportError:
    import branch

logger = logging.getLogger("mesh")
logger.setLevel(logging.INFO)

current_dir = Path("/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry")


def _copy_artifact(src: Path, dst: Path) -> None:
    if dst.exists():
        send2trash(dst)
    if src.is_dir():
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _write_well_suffixed_bp_outputs(n_wells: int) -> None:
    """
    Duplicate core BP outputs with a `_well{i}` suffix so run_all.py can copy
    well-specific files to organoid_i folders.
    """
    if n_wells < 1:
        return

    base_names = [
        "tagged_branches_inlet.bp",
        "tagged_branches_outlet.bp",
        "pressure_checkpoint_inlet.bp",
        "pressure_checkpoint_outlet.bp",
        "velocity_checkpoint_inlet.bp",
        "velocity_checkpoint_outlet.bp",
    ]
    optional_names = [
        "flow_checkpoint_inlet.bp",
        "flow_checkpoint_outlet.bp",
        "area_checkpoint_inlet.bp",
        "area_checkpoint_outlet.bp",
    ]

    for i in range(1, n_wells + 1):
        suffix = f"_well{i}.bp"
        for name in base_names + optional_names:
            src = current_dir / name
            if not src.exists():
                if name in base_names:
                    raise FileNotFoundError(f"Missing required BP output for well suffixing: {src}")
                continue
            dst = current_dir / name.replace(".bp", suffix)
            _copy_artifact(src, dst)


def _infer_well_count(default_n: int, *series: Optional[Sequence[str]]) -> int:
    lengths = [len(s) for s in series if s is not None]
    if not lengths:
        return default_n
    uniq = sorted(set(lengths))
    if len(uniq) != 1:
        raise ValueError(f"Inconsistent per-well input lengths: {uniq}")
    return int(uniq[0])


def _organoid_series_from_base(base: str, n_wells: int, require_all: bool = False) -> List[str]:
    """
    Expand .../organoid_1/... to .../organoid_i/... when available.
    If `require_all` is True, missing organoid_i paths raise.
    """
    base_p = Path(base)
    base_s = str(base_p)
    out: List[str] = []
    has_organoid_token = "organoid_1" in base_s

    for i in range(1, n_wells + 1):
        if has_organoid_token:
            cand = Path(base_s.replace("organoid_1", f"organoid_{i}"))
        else:
            cand = base_p
        if cand.exists():
            out.append(str(cand))
        else:
            if require_all:
                raise FileNotFoundError(f"Missing per-well path for organoid_{i}: {cand}")
            out.append(str(base_p))
    return out

# ---------------------------------------------------------------------
# 1D pipeline (kept from your original script)
# ---------------------------------------------------------------------

def geo_to_mesh_gmsh(geo_file: str, msh_file: str) -> None:
    import gmsh
    gmsh.initialize()
    gmsh.model.add("branched")
    gmsh.open(str(current_dir / geo_file))
    gmsh.model.mesh.generate(1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(current_dir / msh_file))
    gmsh.finalize()


def convert_mesh(msh_file: str, xdmf_file: str) -> None:
    """Extract 1D 'line' cells from msh and write XDMF."""
    msh = meshio.read(current_dir / msh_file)
    line_cells = [cell for cell in msh.cells if cell.type == "line"]
    if not line_cells:
        raise RuntimeError("No line cells found in the mesh.")
    line_cells = [("line", line_cells[0].data)]
    line_cell_data = {}
    if "gmsh:physical" in msh.cell_data_dict and "line" in msh.cell_data_dict["gmsh:physical"]:
        line_cell_data = {"gmsh:physical": [msh.cell_data_dict["gmsh:physical"]["line"]]}
    line_mesh = meshio.Mesh(points=msh.points, cells=line_cells, cell_data=line_cell_data if line_cell_data else None)
    line_mesh.write(current_dir / xdmf_file)


def xdmf_to_dolfinx(xdmf_file: str) -> dfx.mesh.Mesh:
    with XDMFFile(MPI.COMM_WORLD, current_dir / xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    # ensure basic connectivities
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    return mesh


def branch_mesh_tagging(mesh: dfx.mesh.Mesh, inlet: np.ndarray) -> Tuple[np.ndarray, dfx.mesh.MeshTags]:
    """Returns outlet coordinates on 1D boundary (excluding inlet point)."""
    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    boundary_facets_indices = dfx.mesh.exterior_facet_indices(mesh.topology)

    def near_inlet(x):
        return np.isclose(x[0], inlet[0]) & np.isclose(x[1], inlet[1]) & np.isclose(x[2], inlet[2])

    inlet_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, near_inlet)
    outlet_facets = np.setdiff1d(boundary_facets_indices, inlet_facets)

    facet_indices = np.concatenate([inlet_facets, outlet_facets])
    facet_markers = np.concatenate(
        [np.full(len(inlet_facets), 1, dtype=np.int32), np.full(len(outlet_facets), 2, dtype=np.int32)]
    )
    facet_tag = dfx.mesh.meshtags(mesh, fdim, facet_indices.astype(np.int32), facet_markers.astype(np.int32))
    outlet_coords = mesh.geometry.x[facet_tag.indices[facet_tag.values == 2]]
    return outlet_coords, facet_tag


def load_vtp(directory: str):
    """Kept as in your original: reads time series VTPs and returns arrays."""
    def extract_arrays(data_obj):
        return {
            data_obj.GetArray(i).GetName(): vtk_to_numpy(data_obj.GetArray(i))
            for i in range(data_obj.GetNumberOfArrays())
        }

    vtp_files = sorted(glob.glob(os.path.join(directory, "*.vtp")), key=lambda x: int(re.findall(r"\d+", x)[-1]))
    if not vtp_files:
        vtp_files = sorted(
            glob.glob(os.path.join(directory + os.sep + "timeseries", "*.vtp")),
            key=lambda x: int(re.findall(r"\d+", x)[-1]),
        )

    points_per_section = 20
    centerlineVel, centerlineFlow, pressure = [], [], []
    centerlineCoords, area = None, None

    for file in vtp_files:
        if file.endswith(("centerlines.vtp", "1d_model.vtp", "total.vtp")):
            continue
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file)
        reader.Update()
        polydata = reader.GetOutput()

        points = polydata.GetPoints()
        num_points = points.GetNumberOfPoints()
        coords = np.array([points.GetPoint(i) for i in range(num_points)])

        point_data = extract_arrays(polydata.GetPointData())

        area_ = point_data["Area"]
        flowrate_1d = point_data["Flowrate"]
        pressure_1d = point_data["Pressure_mmHg"]

        velocity_1d = flowrate_1d / area_
        centerlineVel.append(velocity_1d[::points_per_section])
        centerlineFlow.append(flowrate_1d[::points_per_section])
        pressure.append(pressure_1d[::points_per_section])

        centerlineCoords = coords.reshape(-1, points_per_section, 3).mean(axis=1)
        area = area_[::points_per_section]

    return centerlineVel, centerlineFlow, pressure, centerlineCoords, area


def generate_1d_files(xdmf_file: str, output_dir: str, file_prefix: str = "", inlet_coords: np.ndarray = None):
    
    # Load the converted XDMF mesh
    with XDMFFile(MPI.COMM_WORLD, current_dir/xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    P1 = dfx.fem.functionspace(mesh, ("CG", 1))  # Continuous Lagrange, degree 1
    P1_vec = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    P1vec = dfx.fem.functionspace(mesh, P1_vec)  # Vector space for velocity
    dof_coords = P1.tabulate_dof_coordinates()
    num_dofs = dof_coords.shape[0]

    velocity_fn = dfx.fem.Function(P1vec)
    pressure_fn = dfx.fem.Function(P1)
    flow_fn = dfx.fem.Function(P1)
    area_fn = dfx.fem.Function(P1)

    centerlineVel, centerlineFlow, pressure, centerlineCoords, area = load_vtp(output_dir)
    Nt = len(centerlineFlow)  # Number of timesteps

    fdim = mesh.topology.dim - 1

    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    outlet_coords, facet_tag = branch_mesh_tagging(mesh, inlet=inlet_coords)

    # If file exists, remove it
    if (current_dir / f"tagged_branches{file_prefix}.bp").exists():
        send2trash(current_dir / f"tagged_branches{file_prefix}.bp")

    adios4dolfinx.write_mesh(Path(str(current_dir / f"tagged_branches{file_prefix}.bp")), mesh, engine="BP4")
    adios4dolfinx.write_meshtags(Path(str(current_dir / f"tagged_branches{file_prefix}.bp")), mesh, facet_tag, engine="BP4")

    # Allocate u_val as a 2D array for all timesteps and dofs
    u_val = np.zeros((Nt, num_dofs))

    # Create connectivity between the mesh elements and their facets
    mesh.topology.create_connectivity(mesh.topology.dim,
                                       mesh.topology.dim - 1)
    
    vtx_vel = dfx.io.VTXWriter(
        MPI.COMM_WORLD, current_dir/f"velocity{file_prefix}.bp", [velocity_fn], engine="BP4"
    )
    
    vtx_pres = dfx.io.VTXWriter(
        MPI.COMM_WORLD, current_dir/f"pressure{file_prefix}.bp", [pressure_fn], engine="BP4"
    )
    vtx_flow = dfx.io.VTXWriter(
        MPI.COMM_WORLD, current_dir/f"flow{file_prefix}.bp", [flow_fn], engine="BP4"
    )
    vtx_area = dfx.io.VTXWriter(
        MPI.COMM_WORLD, current_dir/f"area{file_prefix}.bp", [area_fn], engine="BP4"
    )
    vtx_vel.write(0.0)
    vtx_pres.write(0.0)
    vtx_flow.write(0.0)
    vtx_area.write(0.0)
    # adios4dolfinx.write_mesh(Path("velocity.bp"), mesh, engine="BP4")
    # adios4dolfinx.write_mesh(Path("pressure.bp"), mesh, engine="BP4")
    # Commenting out nearest neighbor search
    tree = None  # KDTree for nearest neighbor search
    dt = 1

    # if checkpoint files exist, remove them
    if (current_dir / f"velocity_checkpoint{file_prefix}.bp").exists():
        send2trash(current_dir / f"velocity_checkpoint{file_prefix}.bp")
    if (current_dir / f"pressure_checkpoint{file_prefix}.bp").exists():
        send2trash(current_dir / f"pressure_checkpoint{file_prefix}.bp")
    if (current_dir / f"flow_checkpoint{file_prefix}.bp").exists():
        send2trash(current_dir / f"flow_checkpoint{file_prefix}.bp")
    if (current_dir / f"area_checkpoint{file_prefix}.bp").exists():
        send2trash(current_dir / f"area_checkpoint{file_prefix}.bp")

    for i in range(Nt):
        time = i * dt
        points = centerlineCoords  # Get the points for the current timestep
        scalar_velocity = centerlineVel[time]  # Get the velocity for the current timestep
        scalar_flow = centerlineFlow[time]  # Get the flow for the current timestep
        scalar_pressure = pressure[time]  # Get the pressure for the current timestep
        scalar_area = area  # Area is constant across timesteps

        if tree is None:
            tree = cKDTree(points)

        # Nearest neighbor interpolation
        distances, indices = tree.query(dof_coords)
        interpolated_velocity = scalar_velocity[indices]
        interpolated_pressure = scalar_pressure[indices]
        interpolated_flow = scalar_flow[indices]
        interpolated_area = scalar_area[indices]

        # Compute tangents using the next connected point
        tangents = np.zeros((len(indices), 3))
        for j, idx in enumerate(indices):
            # Simple forward difference: pick next point if possible, else previous
            if idx < len(points) - 1:
                delta = points[idx + 1] - points[idx]
            else:
                delta = points[idx] - points[idx - 1]
            tangent = delta / np.linalg.norm(delta)
            tangents[j] = tangent

        # Assign to velocity function
        velocity_fn.x.array[:] = (interpolated_velocity[:, np.newaxis] * tangents).flatten()
        pressure_fn.x.array[:] = interpolated_pressure
        flow_fn.x.array[:] = interpolated_flow
        area_fn.x.array[:] = interpolated_area

        adios4dolfinx.write_function(
            Path(current_dir / f"velocity_checkpoint{file_prefix}.bp"),
            u = velocity_fn,
            time=float(time),
            name="f")
        adios4dolfinx.write_function(
            Path(current_dir / f"pressure_checkpoint{file_prefix}.bp"),
            u = pressure_fn,
            time=float(time),
            name="f")
        adios4dolfinx.write_function(
            Path(current_dir / f"flow_checkpoint{file_prefix}.bp"),
            u = flow_fn,
            time=float(time),
            name="f")
        adios4dolfinx.write_function(
            Path(current_dir / f"area_checkpoint{file_prefix}.bp"),
            u = area_fn,
            time=float(time),
            name="f")

        # --- Write data to file with time stamp ---
        vtx_vel.write(float(time))
        vtx_pres.write(float(time))
        vtx_flow.write(float(time))
        vtx_area.write(float(time))

    vtx_vel.close()
    vtx_pres.close()
    vtx_flow.close()
    vtx_area.close()

    return outlet_coords

def import_branched_mesh(
    branching_data_file: str,
    output_1d: str,
    geo_file: str,
    msh_file: str,
    xdmf_file: str,
    fileprefix: str,
    coords: np.ndarray,
) -> np.ndarray:
    df = pd.read_csv(branching_data_file)
    branch.write_geo_from_branching_data(df, geo_file=geo_file)
    geo_to_mesh_gmsh(geo_file=geo_file, msh_file=msh_file)
    convert_mesh(msh_file=msh_file, xdmf_file=xdmf_file)
    _ = xdmf_to_dolfinx(xdmf_file=xdmf_file)
    outlet_coords = generate_1d_files(xdmf_file=xdmf_file, output_dir=output_1d, file_prefix=fileprefix, inlet_coords=coords)
    return outlet_coords

# ---------------------------------------------------------------------
# 3D replacement: VTU -> XDMF and facet meshtagging
# ---------------------------------------------------------------------

def vtu_to_xdmf(vtu_file: str, xdmf_file: str) -> None:
    """Convert VTU (tet volume) to XDMF for dolfinx."""
    m = meshio.read(vtu_file)
    tetra_cells = [c for c in m.cells if c.type in ("tetra", "tetra10")]
    if not tetra_cells:
        raise RuntimeError(f"No tetra cells found in VTU: {vtu_file}")
    # Prefer linear tetra for dolfinx import; keep connectivity as-is
    tet_mesh = meshio.Mesh(points=m.points, cells=[("tetra", tetra_cells[0].data.astype(np.int64))])
    meshio.write(current_dir / xdmf_file, tet_mesh)

import subprocess
from pathlib import Path
import meshio

import numpy as np
import meshio
import pyvista as pv
from pathlib import Path

def vtu_volume_to_surface_stl(vtu_in: str, stl_out: str) -> str:
    m = meshio.read(vtu_in)

    tet = None
    for c in m.cells:
        if c.type in ("tetra", "tetra10"):
            tet = c.data[:, :4] if c.type == "tetra10" else c.data
            break
    if tet is None:
        raise RuntimeError("No tetra cells found in VTU.")

    pts = m.points.astype(np.float64)
    nc = tet.shape[0]
    cells = np.hstack([np.full((nc, 1), 4, dtype=np.int64), tet.astype(np.int64)]).ravel()
    celltypes = np.full(nc, pv.CellType.TETRA, dtype=np.uint8)
    ug = pv.UnstructuredGrid(cells, celltypes, pts)

    surf = ug.extract_surface().triangulate()
    surf = surf.clean(tolerance=1e-8)
    try:
        surf = surf.fill_holes(1e9)  # helps ensure watertight
    except Exception:
        pass

    Path(stl_out).parent.mkdir(parents=True, exist_ok=True)
    surf.save(stl_out)
    return stl_out


def refine_vtu_with_gmsh(vtu_in: str, vtu_out: str, *, n_refine: int = 1, gmsh_exe: str = "gmsh"):
    """
    Refine an existing tetrahedral VTU using Gmsh, fully in Python.
    - Converts VTU -> MSH2
    - Runs gmsh -refine n_refine times
    - Converts refined MSH -> VTU (or you can write XDMF instead)
    """
    vtu_in = str(vtu_in)
    vtu_out = str(vtu_out)
    tmp_dir = Path(vtu_out).parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    msh0 = tmp_dir / "_tmp_in.msh"
    msh1 = tmp_dir / "_tmp_refined.msh"

    # 1) VTU -> MSH (MSH2 ASCII for compatibility)
    m = meshio.read(vtu_in)
    tet_cells = None
    for c in m.cells:
        if c.type == "tetra":
            tet_cells = c.data
            break
        if c.type == "tetra10":
            # gmsh can read 2nd order too, but refinement is simpler on linear tets;
            # downgrade if needed
            tet_cells = c.data[:, :4]
            break
    if tet_cells is None:
        raise RuntimeError(f"No tetra cells found in {vtu_in}")

    msh_mesh = meshio.Mesh(points=m.points, cells=[("tetra", tet_cells.astype(int))])
    meshio.write(msh0, msh_mesh, file_format="gmsh22")

    # 2) Refine with gmsh (repeat if requested)
    # gmsh -refine reads an existing mesh and refines it
    current = msh0
    for k in range(n_refine):
        cmd = [gmsh_exe, "-refine", str(current), "-format", "msh2", "-o", str(msh1)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"Gmsh refine failed:\nCMD: {' '.join(cmd)}\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}\n")
        current = msh1

    # 3) Refined MSH -> VTU
    mref = meshio.read(current)
    tet_cells = [c for c in mref.cells if c.type == "tetra"]
    if not tet_cells:
        raise RuntimeError("Refined mesh contains no tetra cells")

    out = meshio.Mesh(points=mref.points, cells=tet_cells)
    meshio.write(vtu_out, out, file_format="vtu")

    # Optional: cleanup
    try:
        msh0.unlink(missing_ok=True)
        msh1.unlink(missing_ok=True)
    except Exception:
        pass

import numpy as np
from dolfinx import mesh as dmesh

def refine_large_cells(msh, hmax: float, nsteps: int = 1):
    """
    Refine only cells whose estimated size exceeds hmax by splitting all edges
    belonging to those cells (dolfinx 0.10 requires edge-based refinement).
    """
    tdim = msh.topology.dim
    fdim = tdim - 1

    for _ in range(nsteps):
        # Ensure required connectivities exist
        msh.topology.create_entities(1)                 # create edges
        msh.topology.create_connectivity(tdim, 0)       # cell -> vertices
        msh.topology.create_connectivity(tdim, 1)       # cell -> edges

        c2v = msh.topology.connectivity(tdim, 0)
        c2e = msh.topology.connectivity(tdim, 1)
        X = msh.geometry.x

        ncells = msh.topology.index_map(tdim).size_local
        cell_marked = np.zeros(ncells, dtype=np.int8)

        # Mark large cells
        for c in range(ncells):
            vs = c2v.links(c)
            pts = X[vs]
            dmax = 0.0
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    d = np.linalg.norm(pts[i] - pts[j])
                    if d > dmax:
                        dmax = d
            if dmax > hmax:
                cell_marked[c] = 1

        if not cell_marked.any():
            break

        # Collect edges of marked cells
        edge_ids = []
        for c in np.where(cell_marked == 1)[0]:
            edge_ids.extend(c2e.links(c))

        # Unique edges (local)
        edges = np.unique(np.array(edge_ids, dtype=np.int32))

        # Refine by splitting these edges
        msh, _, _ = dmesh.refine(msh, edges)

    # Rebuild connectivities often used later (facet tagging etc.)
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(tdim, fdim)
    return msh

def refine_mesh_uniform(mesh: dfx.mesh.Mesh, n_refine: int) -> dfx.mesh.Mesh:
    tdim = mesh.topology.dim
    # For 3D tetra mesh: edges are dim=1
    mesh.topology.create_entities(1)
    mesh.topology.create_connectivity(1, 0)      # edge -> vertex
    mesh.topology.create_connectivity(tdim, 1)   # cell -> edge (needed by refine)

    for _ in range(n_refine):
        mesh, _, _ = dmesh.refine(mesh)
        # After refinement, rebuild the connectivities that downstream tagging/writing needs
        mesh.topology.create_entities(1)
        mesh.topology.create_connectivity(1, 0)
        mesh.topology.create_connectivity(mesh.topology.dim, 1)

    # And also make sure facets exist (you tag facets later)
    fdim = mesh.topology.dim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(mesh.topology.dim, fdim)

    return mesh

# def read_3d_mesh_from_vtu(vtu_file: str, xdmf_name: str, n_refine: int = 0) -> dfx.mesh.Mesh:
#     vtu_to_xdmf(vtu_file, xdmf_name)
#     with XDMFFile(MPI.COMM_WORLD, current_dir / xdmf_name, "r") as xdmf:
#         mesh = xdmf.read_mesh(name="Grid")

#     if n_refine > 0:
#         mesh = refine_mesh_uniform(mesh, n_refine)

#     # ensure facet entities exist for tagging/writing
#     tdim = mesh.topology.dim
#     fdim = tdim - 1
#     mesh.topology.create_entities(fdim)
#     mesh.topology.create_connectivity(fdim, 0)
#     mesh.topology.create_connectivity(tdim, fdim)
#     return mesh

import numpy as np
from dolfinx import mesh as dmesh

def refine_long_edges(msh, hmax: float, nsteps: int = 1):
    """
    Refine by splitting only edges longer than hmax.
    This avoids shrinking regions that are already fine.
    """
    for _ in range(nsteps):
        msh.topology.create_entities(1)            # edges
        msh.topology.create_connectivity(1, 0)     # edge -> vertices
        e2v = msh.topology.connectivity(1, 0)
        X = msh.geometry.x

        nedges = msh.topology.index_map(1).size_local
        long_edges = np.zeros(nedges, dtype=np.int8)

        for e in range(nedges):
            vs = e2v.links(e)
            p0, p1 = X[vs[0]], X[vs[1]]
            if np.linalg.norm(p1 - p0) > hmax:
                long_edges[e] = 1

        if not long_edges.any():
            break

        edges = np.where(long_edges == 1)[0].astype(np.int32)
        msh, _, _ = dmesh.refine(msh, edges)

    # rebuild common connectivities
    tdim = msh.topology.dim
    fdim = tdim - 1
    msh.topology.create_entities(fdim)
    msh.topology.create_connectivity(fdim, 0)
    msh.topology.create_connectivity(tdim, fdim)
    return msh

def read_3d_mesh_from_vtu(vtu_file: str, xdmf_name: str,
                          n_refine: int = 0,
                          refine_hmax: float = 0.01,
                          refine_steps: int = 0) -> dfx.mesh.Mesh:
    vtu_to_xdmf(vtu_file, xdmf_name)
    with XDMFFile(MPI.COMM_WORLD, current_dir / xdmf_name, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # Optional: selective refinement FIRST (targets only coarse elements)
    if refine_hmax is not None:
        mesh = refine_long_edges(mesh, hmax=refine_hmax, nsteps=refine_steps)

    # Optional: your existing uniform refinement
    if n_refine > 0:
        mesh = refine_mesh_uniform(mesh, n_refine)

    # ensure facet entities exist for tagging/writing
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(tdim, fdim)
    return mesh


def _boundary_facets_and_geometry(mesh: dfx.mesh.Mesh):
    """Compute boundary facet list, centers, normals, adjacency."""
    tdim = mesh.topology.dim
    fdim = tdim - 1
    gdim = mesh.geometry.dim
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)

    all_bdry_facets = dmesh.locate_entities_boundary(mesh, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
    if all_bdry_facets.size == 0:
        raise RuntimeError("No boundary facets found on 3D mesh.")

    f_to_v = mesh.topology.connectivity(fdim, 0)
    X = mesh.geometry.x

    nfac = len(all_bdry_facets)
    centers = np.zeros((nfac, gdim), dtype=float)
    normals = np.zeros((nfac, gdim), dtype=float)

    for i, f in enumerate(all_bdry_facets):
        vs = f_to_v.links(int(f))
        pts = X[vs]
        centers[i] = pts.mean(axis=0)
        v0, v1, v2 = pts[0], pts[1], pts[2]
        n = np.cross(v1 - v0, v2 - v0)
        nn = np.linalg.norm(n)
        if nn > 0:
            n /= nn
        normals[i] = n

    # adjacency (shared vertices)
    v_to_f: Dict[int, List[int]] = {}
    for f in all_bdry_facets:
        vs = f_to_v.links(int(f))
        for v in vs:
            v_to_f.setdefault(int(v), []).append(int(f))

    adj: Dict[int, set] = {int(f): set() for f in all_bdry_facets}
    for f in all_bdry_facets:
        vs = f_to_v.links(int(f))
        neigh = set()
        for v in vs:
            neigh.update(v_to_f[int(v)])
        neigh.discard(int(f))
        adj[int(f)] = neigh

    facet_to_idx = {int(f): i for i, f in enumerate(all_bdry_facets)}
    return all_bdry_facets.astype(np.int32), centers, normals, adj, facet_to_idx


def tag_surface_from_seed(mesh: dfx.mesh.Mesh, x0: np.ndarray, theta_deg: float, marker: int) -> dfx.mesh.MeshTags:
    """Your original region-growing patch tagger (auto seed normal)."""
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)

    all_bdry_facets, centers, normals, adj, facet_to_idx = _boundary_facets_and_geometry(mesh)

    x0 = np.asarray(x0, float)
    d2 = np.sum((centers - x0) ** 2, axis=1)
    seed_i = int(np.argmin(d2))
    seed_f = int(all_bdry_facets[seed_i])

    n0 = normals[seed_i].copy()
    nn = np.linalg.norm(n0)
    if nn == 0:
        raise RuntimeError("Seed facet has zero normal.")
    n0 /= nn

    cos_thr = float(np.cos(np.deg2rad(theta_deg)))

    selected = set()
    visited = set()
    stack = [seed_f]
    while stack:
        f = stack.pop()
        if f in visited:
            continue
        visited.add(f)
        i = facet_to_idx[f]
        nf = normals[i]
        if abs(np.dot(nf, n0)) >= cos_thr:
            selected.add(f)
            for nb in adj[f]:
                if nb not in visited:
                    stack.append(nb)

    if not selected:
        raise RuntimeError("tag_surface_from_seed selected no facets.")

    idx = np.array(sorted(selected), dtype=np.int32)
    vals = np.full_like(idx, marker, dtype=np.int32)
    return dmesh.meshtags(mesh, fdim, idx, vals)


def tag_terminal_plane_patch(
    mesh: dfx.mesh.Mesh,
    x0: np.ndarray,
    marker: int,
    theta_deg: float = 15.0,
    plane_tol: Optional[float] = None,
) -> dfx.mesh.MeshTags:
    """
    Tag a connected terminal face near x0, constrained to a single plane:
    - choose nearest boundary facet to x0
    - seed plane: (x - x_seed) · n0 = 0
    - region-grow over facets with normals aligned to n0 and centers close to seed plane.
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1

    all_bdry_facets, centers, normals, adj, facet_to_idx = _boundary_facets_and_geometry(mesh)

    x0 = np.asarray(x0, float)
    d2 = np.sum((centers - x0) ** 2, axis=1)
    seed_i = int(np.argmin(d2))
    seed_f = int(all_bdry_facets[seed_i])

    n0 = normals[seed_i].copy()
    nn = np.linalg.norm(n0)
    if nn == 0:
        raise RuntimeError("Terminal seed facet has zero normal.")
    n0 /= nn
    x_seed = centers[seed_i].copy()

    # default plane tolerance: 2% of local bbox diagonal
    if plane_tol is None:
        bbox = mesh.geometry.x.max(axis=0) - mesh.geometry.x.min(axis=0)
        plane_tol = 0.002 * float(np.linalg.norm(bbox))

    cos_thr = float(np.cos(np.deg2rad(theta_deg)))

    selected = set()
    visited = set()
    stack = [seed_f]
    while stack:
        f = stack.pop()
        if f in visited:
            continue
        visited.add(f)
        i = facet_to_idx[f]
        nf = normals[i]
        if abs(np.dot(nf, n0)) < cos_thr:
            continue
        # coplanarity check using facet center
        if abs(np.dot((centers[i] - x_seed), n0)) > plane_tol:
            continue
        selected.add(f)
        for nb in adj[f]:
            if nb not in visited:
                stack.append(nb)

    if not selected:
        raise RuntimeError(f"Terminal patch empty for x0={x0}. Try increasing plane_tol/theta_deg.")

    idx = np.array(sorted(selected), dtype=np.int32)
    vals = np.full_like(idx, marker, dtype=np.int32)
    return dmesh.meshtags(mesh, fdim, idx, vals)


def tag_outer_wall_facets(mesh: dfx.mesh.Mesh, marker: int, outer_tol: Optional[float] = None) -> Dict[int, int]:
    """Tag outer box boundary facets based on bbox planes."""
    tdim = mesh.topology.dim
    fdim = tdim - 1
    all_bdry_facets, centers, normals, adj, facet_to_idx = _boundary_facets_and_geometry(mesh)

    X = mesh.geometry.x
    mins = X.min(axis=0); maxs = X.max(axis=0)
    diag = float(np.linalg.norm(maxs - mins))
    if outer_tol is None:
        outer_tol = 1e-3 * diag

    facet_to_marker: Dict[int, int] = {}
    for f, c in zip(all_bdry_facets, centers):
        if (
            abs(c[0] - mins[0]) < outer_tol or abs(c[0] - maxs[0]) < outer_tol or
            abs(c[1] - mins[1]) < outer_tol or abs(c[1] - maxs[1]) < outer_tol or
            abs(c[2] - mins[2]) < outer_tol or abs(c[2] - maxs[2]) < outer_tol
        ):
            facet_to_marker[int(f)] = int(marker)
    return facet_to_marker

def _get_ftetwild_path():
    project_root = current_dir.parent.parent
    candidates = [
        os.environ.get("FTETWILD_PATH"),
        shutil.which("FloatTetwild_bin"),
        str(project_root / "clones" / "fTetWild" / "build" / "FloatTetwild_bin"),
        str(project_root / "clones" / "fTetWild" / "build" / "FloatTetwild_bin.exe"),
    ]

    for cand in candidates:
        if not cand:
            continue
        p = Path(cand).expanduser()
        if p.is_file():
            return str(p)

    raise RuntimeError(
        "ERROR: cannot find fTetWild. Looked for FTETWILD_PATH, PATH, and "
        f"{project_root / 'clones' / 'fTetWild' / 'build' / 'FloatTetwild_bin'}."
    )

def geo_to_mesh_gmsh(geo_file, msh_file, mesh_size=0.001):
    gmsh.initialize()
    gmsh.open(str(current_dir / geo_file))

    # Generate 1D mesh (use .generate(2) for 2D, .generate(3) for 3D)
    gmsh.model.mesh.generate(1)

    # Save mesh in MSH version 2 format
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(current_dir / msh_file))

    gmsh.finalize()    

def stl_to_mesh_gmsh(stl_file, msh_file,char_len_min=0.001, char_len_max=0.01): # let final char_len_max be 0.01

    gmsh.initialize()
    gmsh.model.add("bioreactor")
    gmsh.merge(stl_file)

    gmsh.model.mesh.removeDuplicateNodes()
    
    # Classify surfaces to reconstruct the volume
    angle = 30  # angle threshold in degrees for feature edges
    force_param = True
    include_boundary = True
    curve_angle = 180
    gmsh.model.mesh.classifySurfaces(angle * (3.14159265359 / 180), include_boundary, force_param, curve_angle * (3.14159265359 / 180))
    
    # Create geometry from mesh
    gmsh.model.mesh.createGeometry()
    gmsh.model.mesh.createTopology()

    # Create volume from surface
    s = gmsh.model.getEntities(2)  # Get all surfaces
    gmsh.model.geo.addSurfaceLoop([surf[1] for surf in s], 1)
    vol = gmsh.model.geo.addVolume([1], 1)
    gmsh.model.geo.synchronize()

    # Define physical group (optional but needed for DOLFINx)
    gmsh.model.addPhysicalGroup(3, [1], 1)  # Volume
    gmsh.model.setPhysicalName(3, 1, "volume")

    p = gmsh.model.addPhysicalGroup(3, [vol], 9)
    gmsh.model.setPhysicalName(3, p, "Wall")

    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len_max)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_len_min)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(str(current_dir / msh_file))
    logger.info(f"Created mesh {current_dir/msh_file} from {stl_file}")
    gmsh.finalize()

def stl_to_mesh_ftet(
    stl_file,
    msh_file,
    edge_length: float = 0.005,
    eps: float = 0.0005,
    max_threads: Optional[int] = None,
):
    """
    Mesh STL with fTetWild.

    For better surface preservation, reduce both:
    - edge_length (--lr): smaller target tetra edge length ratio
    - eps (--epsr): smaller geometric perturbation tolerance ratio
    """
    if edge_length <= 0:
        raise ValueError("edge_length must be > 0")
    if eps <= 0:
        raise ValueError("eps must be > 0")
    if eps >= edge_length:
        logger.warning(
            "eps (%g) >= edge_length (%g). This may over-smooth geometry.",
            eps,
            edge_length,
        )
    if max_threads is not None and max_threads <= 0:
        raise ValueError("max_threads must be > 0 when provided")

    float_tetwild_path = _get_ftetwild_path()
    out_path = current_dir / msh_file
    cmd = [
        float_tetwild_path,
        "-i",
        str(stl_file),
        "-o",
        str(out_path),
        "--lr",
        str(edge_length),
        "--epsr",
        str(eps),
    ]
    if max_threads is not None:
        cmd.extend(["--max-threads", str(int(max_threads))])
    logger.info("Running fTetWild: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

def mesh_to_xdmf(msh_file, xdmf_file):
    # Convert .msh to .vtu
    mesh = meshio.read(current_dir/msh_file)
    meshio.write(current_dir/xdmf_file, mesh)

def xdmf_to_dolfinx(xdmf_file):
    with XDMFFile(MPI.COMM_WORLD, current_dir/xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # Create connectivity between the mesh elements and their facets
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

def remove_seed_points(terminal_pts, seed_pt, tol=1e-8):
    """
    Remove the single row in terminal_pts that is closest to seed_pt.
    Returns (filtered_pts, removed_index, removed_point).
    """
    terminal_pts = np.asarray(terminal_pts, dtype=float).reshape(-1, 3)
    seed_pt = np.asarray(seed_pt, dtype=float).reshape(3,)

    if terminal_pts.shape[0] == 0:
        return terminal_pts, None, None

    d2 = np.sum((terminal_pts - seed_pt) ** 2, axis=1)
    idx = int(np.argmin(d2))
    removed = terminal_pts[idx].copy()
    filtered = np.delete(terminal_pts, idx, axis=0)
    return filtered, idx, removed

def build_facet_tags_for_mesh(
    mesh: dfx.mesh.Mesh,
    inlet_terminal_pts: np.ndarray,
    outlet_terminal_pts: np.ndarray,
    coords_inlet: np.ndarray,
    coords_outlet: np.ndarray,
    *,
    wall_marker: int = 1,
    arterial_concave_marker: int = 31,
    venous_concave_marker: int = 32,
    inlet_base_marker: int = 100,
    outlet_base_marker: int = 200,
    theta_concave_deg: float = 60.0,
    theta_terminal_deg: float = 30.0,
    plane_tol: Optional[float] = None,
    outer_tol: Optional[float] = None,
) -> dfx.mesh.MeshTags:
    """
    Builds a SINGLE facet MeshTags (fdim=2) with priority:
        wall < concave patches < terminal faces
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(tdim, fdim)

    facet_to_marker: Dict[int, int] = {}

    # 1) outer wall (lowest priority)
    facet_to_marker.update(tag_outer_wall_facets(mesh, marker=wall_marker, outer_tol=outer_tol))

    # 2) concave patches (overwrite wall)
    shifted_inlet = coords_inlet[:] + np.array([0, 0.1, 0.0])  # shift slightly to ensure seed is inside
    shifted_outlet = coords_outlet[:] + np.array([0, 0.1, 0.0])
    conc_art = tag_surface_from_seed(mesh, shifted_inlet, theta_deg=theta_concave_deg, marker=arterial_concave_marker)
    conc_ven = tag_surface_from_seed(mesh, shifted_outlet, theta_deg=theta_concave_deg, marker=venous_concave_marker)
    for f, v in zip(conc_art.indices, conc_art.values):
        facet_to_marker[int(f)] = int(v)
    for f, v in zip(conc_ven.indices, conc_ven.values):
        facet_to_marker[int(f)] = int(v)

    # 3) terminal faces (overwrite everything)
    # Keep all inlet/outlet terminal points; do not drop seed-corresponding terminals.
    # inlet terminals
    for k, pt in enumerate(np.asarray(inlet_terminal_pts, float)):
        tag = inlet_base_marker + k
        tpatch = tag_terminal_plane_patch(mesh, pt, marker=tag, theta_deg=theta_terminal_deg, plane_tol=plane_tol)
        for f, v in zip(tpatch.indices, tpatch.values):
            facet_to_marker[int(f)] = int(v)

    # outlet terminals
    for k, pt in enumerate(np.asarray(outlet_terminal_pts, float)):
        tag = outlet_base_marker + k
        tpatch = tag_terminal_plane_patch(mesh, pt, marker=tag, theta_deg=theta_terminal_deg, plane_tol=plane_tol)
        for f, v in zip(tpatch.indices, tpatch.values):
            facet_to_marker[int(f)] = int(v)

    facet_indices = np.array(sorted(facet_to_marker.keys()), dtype=np.int32)
    facet_values = np.array([facet_to_marker[i] for i in facet_indices], dtype=np.int32)
    facet_tags = dmesh.meshtags(mesh, fdim, facet_indices, facet_values)
    return facet_tags


def write_mesh_and_tags(mesh: dfx.mesh.Mesh, facet_tags: dfx.mesh.MeshTags, mesh_xdmf: str, tags_xdmf: str) -> None:
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(tdim, fdim)

    with XDMFFile(mesh.comm, str(current_dir / mesh_xdmf), "w") as xdmf:
        xdmf.write_mesh(mesh)

    with XDMFFile(mesh.comm, str(current_dir / tags_xdmf), "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)


def translate_mesh_inplace(mesh: dfx.mesh.Mesh, dy: float) -> None:
    mesh.geometry.x[:, 1] += float(dy)


# ---------------------------------------------------------------------
# High-level driver (keeps old 1D pipeline, swaps 3D)
# ---------------------------------------------------------------------

class Files:
    def __init__(
        self,
        tissue_vtu: Optional[str],
        tissue_stl: Optional[str],
        output_1d_inlet: str,
        output_1d_outlet: str,
        branching_data_inlet: str,
        branching_data_outlet: str,
        coords_inlet: np.ndarray,
        coords_outlet: np.ndarray,
        output_1d_inlet_list: Optional[Sequence[str]] = None,
        output_1d_outlet_list: Optional[Sequence[str]] = None,
        branching_data_inlet_list: Optional[Sequence[str]] = None,
        branching_data_outlet_list: Optional[Sequence[str]] = None,
        # 1D filenames
        geo_file_inlet: str = "branched_network_inlet.geo",
        msh_file_inlet: str = "branched_network_inlet.msh",
        xdmf_file_inlet: str = "branched_network_inlet.xdmf",
        geo_file_outlet: str = "branched_network_outlet.geo",
        msh_file_outlet: str = "branched_network_outlet.msh",
        xdmf_file_outlet: str = "branched_network_outlet.xdmf",
        # 3D input options
        tissue_vtu_list: Optional[Sequence[str]] = None,
        ftet_max_threads: Optional[int] = None,
        multiwell: bool = True,
        same_tissue_for_all_wells: bool = True,
        same_meshtags_for_all_wells: bool = False,
        well_spacing_y: float = 0.6,
        # tagging params
        wall_marker: int = 1,
        arterial_concave_marker: int = 31,
        venous_concave_marker: int = 32,
        inlet_base_marker: int = 100,
        outlet_base_marker: int = 200,
        theta_concave_deg: float = 70.0,
        theta_terminal_deg: float = 2.5,
        plane_tol: Optional[float] = None,
        outer_tol: Optional[float] = None,
    ):
        self.coords_inlet = np.asarray(coords_inlet, float)
        self.coords_outlet = np.asarray(coords_outlet, float)

        default_wells = 4 if multiwell else 1
        if tissue_vtu_list is not None:
            default_wells = len(list(tissue_vtu_list))
        n_wells = _infer_well_count(
            default_wells,
            output_1d_inlet_list,
            output_1d_outlet_list,
            branching_data_inlet_list,
            branching_data_outlet_list,
        )

        if branching_data_inlet_list is None:
            branching_in_list = _organoid_series_from_base(branching_data_inlet, n_wells, require_all=(n_wells > 1))
        else:
            branching_in_list = [str(p) for p in branching_data_inlet_list]
        if branching_data_outlet_list is None:
            branching_out_list = _organoid_series_from_base(branching_data_outlet, n_wells, require_all=(n_wells > 1))
        else:
            branching_out_list = [str(p) for p in branching_data_outlet_list]

        if output_1d_inlet_list is None:
            output_in_list = _organoid_series_from_base(output_1d_inlet, n_wells, require_all=False)
        else:
            output_in_list = [str(p) for p in output_1d_inlet_list]
        if output_1d_outlet_list is None:
            output_out_list = _organoid_series_from_base(output_1d_outlet, n_wells, require_all=False)
        else:
            output_out_list = [str(p) for p in output_1d_outlet_list]

        # If output folders for organoid_i are not present in the provided base
        # path (common in coupled/_tmp workflows), fall back to per-organoid
        # source folders adjacent to branchingData_*.csv in trial/prepped dirs.
        for i in range(n_wells):
            if not Path(output_in_list[i]).exists():
                cand = Path(branching_in_list[i]).parent / "0D_Input_Files" / "inlet"
                if cand.exists():
                    output_in_list[i] = str(cand)
            if not Path(output_out_list[i]).exists():
                cand = Path(branching_out_list[i]).parent / "0D_Input_Files" / "outlet"
                if cand.exists():
                    output_out_list[i] = str(cand)

        if not (len(branching_in_list) == len(branching_out_list) == len(output_in_list) == len(output_out_list) == n_wells):
            raise ValueError("Per-well input lists must all have the same length.")

        def _seed_from_branching_csv(csv_path: str, fallback: np.ndarray) -> np.ndarray:
            cols_candidates = [
                ("proximalCoordsX", "proximalCoordsY", "proximalCoordsZ"),
                ("x_prox", "y_prox", "z_prox"),
                ("x0", "y0", "z0"),
            ]
            try:
                df_seed = pd.read_csv(csv_path)
                for cols in cols_candidates:
                    if all(c in df_seed.columns for c in cols) and len(df_seed) > 0:
                        return np.asarray(
                            [
                                float(df_seed.iloc[0][cols[0]]),
                                float(df_seed.iloc[0][cols[1]]),
                                float(df_seed.iloc[0][cols[2]]),
                            ],
                            dtype=float,
                        )
            except Exception:
                pass
            return np.asarray(fallback, dtype=float).copy()

        # --------------------------
        # 1) Build 1D per well using each organoid's already-translated inputs
        # --------------------------
        inlet_terminal_pts_by_well: List[np.ndarray] = []
        outlet_terminal_pts_by_well: List[np.ndarray] = []
        coords_inlet_by_well: List[np.ndarray] = []
        coords_outlet_by_well: List[np.ndarray] = []

        for i in range(1, n_wells + 1):
            c_in = _seed_from_branching_csv(branching_in_list[i - 1], self.coords_inlet)
            c_out = _seed_from_branching_csv(branching_out_list[i - 1], self.coords_outlet)
            coords_inlet_by_well.append(c_in)
            coords_outlet_by_well.append(c_out)

            if n_wells > 1:
                suffix_in = f"_inlet_well{i}"
                suffix_out = f"_outlet_well{i}"
                geo_in = str(Path(geo_file_inlet).with_name(f"{Path(geo_file_inlet).stem}_well{i}{Path(geo_file_inlet).suffix}"))
                msh_in = str(Path(msh_file_inlet).with_name(f"{Path(msh_file_inlet).stem}_well{i}{Path(msh_file_inlet).suffix}"))
                xdmf_in = str(Path(xdmf_file_inlet).with_name(f"{Path(xdmf_file_inlet).stem}_well{i}{Path(xdmf_file_inlet).suffix}"))
                geo_out = str(Path(geo_file_outlet).with_name(f"{Path(geo_file_outlet).stem}_well{i}{Path(geo_file_outlet).suffix}"))
                msh_out = str(Path(msh_file_outlet).with_name(f"{Path(msh_file_outlet).stem}_well{i}{Path(msh_file_outlet).suffix}"))
                xdmf_out = str(Path(xdmf_file_outlet).with_name(f"{Path(xdmf_file_outlet).stem}_well{i}{Path(xdmf_file_outlet).suffix}"))
            else:
                suffix_in, suffix_out = "_inlet", "_outlet"
                geo_in, msh_in, xdmf_in = geo_file_inlet, msh_file_inlet, xdmf_file_inlet
                geo_out, msh_out, xdmf_out = geo_file_outlet, msh_file_outlet, xdmf_file_outlet

            if not Path(branching_in_list[i - 1]).exists():
                raise FileNotFoundError(f"Missing branching_data_inlet for well{i}: {branching_in_list[i - 1]}")
            if not Path(branching_out_list[i - 1]).exists():
                raise FileNotFoundError(f"Missing branching_data_outlet for well{i}: {branching_out_list[i - 1]}")
            if not Path(output_in_list[i - 1]).exists():
                raise FileNotFoundError(f"Missing output_1d_inlet folder for well{i}: {output_in_list[i - 1]}")
            if not Path(output_out_list[i - 1]).exists():
                raise FileNotFoundError(f"Missing output_1d_outlet folder for well{i}: {output_out_list[i - 1]}")

            inlet_terminal_pts = import_branched_mesh(
                branching_data_file=branching_in_list[i - 1],
                output_1d=output_in_list[i - 1],
                geo_file=geo_in,
                msh_file=msh_in,
                xdmf_file=xdmf_in,
                fileprefix=suffix_in,
                coords=c_in,
            )
            outlet_terminal_pts = import_branched_mesh(
                branching_data_file=branching_out_list[i - 1],
                output_1d=output_out_list[i - 1],
                geo_file=geo_out,
                msh_file=msh_out,
                xdmf_file=xdmf_out,
                fileprefix=suffix_out,
                coords=c_out,
            )
            inlet_terminal_pts_by_well.append(inlet_terminal_pts)
            outlet_terminal_pts_by_well.append(outlet_terminal_pts)

        # --------------------------
        # 2) Load/duplicate 3D and tag facets
        # --------------------------
        if tissue_vtu_list is not None:
            vtu_list = list(tissue_vtu_list)
            if len(vtu_list) != n_wells:
                raise ValueError(f"tissue_vtu_list must have {n_wells} VTU files.")
            for i in range(1, n_wells + 1):
                mesh_i = read_3d_mesh_from_vtu(vtu_list[i-1], xdmf_name=f"_tmp_bioreactor_well{i}.xdmf")
                facet_tags_i = build_facet_tags_for_mesh(
                    mesh_i,
                    inlet_terminal_pts=inlet_terminal_pts_by_well[i - 1],
                    outlet_terminal_pts=outlet_terminal_pts_by_well[i - 1],
                    coords_inlet=coords_inlet_by_well[i - 1],
                    coords_outlet=coords_outlet_by_well[i - 1],
                    wall_marker=wall_marker,
                    arterial_concave_marker=arterial_concave_marker,
                    venous_concave_marker=venous_concave_marker,
                    inlet_base_marker=inlet_base_marker,
                    outlet_base_marker=outlet_base_marker,
                    theta_concave_deg=theta_concave_deg,
                    theta_terminal_deg=theta_terminal_deg,
                    plane_tol=plane_tol,
                    outer_tol=outer_tol,
                )
                write_mesh_and_tags(mesh_i, facet_tags_i, f"bioreactor_well{i}.xdmf", f"mesh_tags_well{i}.xdmf")
            return

        if tissue_vtu is None and tissue_stl is None:
            raise ValueError("Provide tissue_vtu or tissue_vtu_list.")

        if not multiwell:
            mesh = read_3d_mesh_from_vtu(tissue_vtu, xdmf_name="_tmp_bioreactor.xdmf")
            facet_tags = build_facet_tags_for_mesh(
                mesh,
                inlet_terminal_pts=inlet_terminal_pts_by_well[0],
                outlet_terminal_pts=outlet_terminal_pts_by_well[0],
                coords_inlet=coords_inlet_by_well[0],
                coords_outlet=coords_outlet_by_well[0],
                wall_marker=wall_marker,
                arterial_concave_marker=arterial_concave_marker,
                venous_concave_marker=venous_concave_marker,
                inlet_base_marker=inlet_base_marker,
                outlet_base_marker=outlet_base_marker,
                theta_concave_deg=theta_concave_deg,
                theta_terminal_deg=theta_terminal_deg,
                plane_tol=plane_tol,
                outer_tol=outer_tol,
            )
            write_mesh_and_tags(mesh, facet_tags, "bioreactor.xdmf", "mesh_tags.xdmf")
            return

        # # multiwell with single VTU
        # refine_vtu_with_gmsh(
        #     vtu_in=tissue_vtu,
        #     vtu_out="tissue_domain_volume_refined.vtu",
        #     n_refine=0  # 1 is usually plenty (DOFs blow up fast!)
        # )

        mesh = read_3d_mesh_from_vtu(tissue_vtu, xdmf_name="_tmp_bioreactor_base.xdmf")
        # stl_to_mesh_ftet(
        #     tissue_stl,
        #     msh_file="bioreactor_base.msh",
        #     max_threads=ftet_max_threads,
        # )
        # mesh_to_xdmf(msh_file="bioreactor_base.msh", xdmf_file="bioreactor_base.xdmf")
        # xdmf_to_dolfinx(xdmf_file="bioreactor_base.xdmf")
        # with XDMFFile(MPI.COMM_WORLD, current_dir / "bioreactor_base.xdmf", "r") as xdmf:
        #     mesh = xdmf.read_mesh(name="Grid")

        if same_tissue_for_all_wells:
            # Translate shared VTU and either reuse well1 tags or retag per well.
            X0 = mesh.geometry.x.copy()
            facet_tags_well1 = build_facet_tags_for_mesh(
                mesh,
                inlet_terminal_pts=inlet_terminal_pts_by_well[0],
                outlet_terminal_pts=outlet_terminal_pts_by_well[0],
                coords_inlet=coords_inlet_by_well[0],
                coords_outlet=coords_outlet_by_well[0],
                wall_marker=wall_marker,
                arterial_concave_marker=arterial_concave_marker,
                venous_concave_marker=venous_concave_marker,
                inlet_base_marker=inlet_base_marker,
                outlet_base_marker=outlet_base_marker,
                theta_concave_deg=theta_concave_deg,
                theta_terminal_deg=theta_terminal_deg,
                plane_tol=plane_tol,
                outer_tol=outer_tol,
            )
            for i in range(1, n_wells + 1):
                dy = well_spacing_y * (i - 1)
                mesh.geometry.x[:] = X0 + np.array([0.0, dy, 0.0], dtype=X0.dtype)
                if same_meshtags_for_all_wells:
                    facet_tags = facet_tags_well1
                else:
                    facet_tags = build_facet_tags_for_mesh(
                        mesh,
                        inlet_terminal_pts=inlet_terminal_pts_by_well[i - 1],
                        outlet_terminal_pts=outlet_terminal_pts_by_well[i - 1],
                        coords_inlet=coords_inlet_by_well[i - 1],
                        coords_outlet=coords_outlet_by_well[i - 1],
                        wall_marker=wall_marker,
                        arterial_concave_marker=arterial_concave_marker,
                        venous_concave_marker=venous_concave_marker,
                        inlet_base_marker=inlet_base_marker,
                        outlet_base_marker=outlet_base_marker,
                        theta_concave_deg=theta_concave_deg,
                        theta_terminal_deg=theta_terminal_deg,
                        plane_tol=plane_tol,
                        outer_tol=outer_tol,
                    )
                write_mesh_and_tags(mesh, facet_tags, f"bioreactor_well{i}.xdmf", f"mesh_tags_well{i}.xdmf")

            # Restore base coords (optional)
            mesh.geometry.x[:] = X0
        else:
            # If you *aren't* using the same tissue for all wells, you need per-well VTUs (tissue_vtu_list)
            raise ValueError(f"same_tissue_for_all_wells=False requires tissue_vtu_list with {n_wells} meshes.")

        return

if __name__ == "__main__":
    # Example usage (update paths)
    Files(
        tissue_stl=None,
        tissue_vtu = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022626/Run15_10branches/tissue_domain_volume.vtu",
        output_1d_inlet="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022626/Run15_10branches/0D_Input_Files/inlet",
        output_1d_outlet="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022626/Run15_10branches/0D_Input_Files/outlet",
        branching_data_inlet="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022626/Run15_10branches/branchingData_0.csv",
        branching_data_outlet="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022626/Run15_10branches/branchingData_1.csv",
        coords_inlet=[-.175, 0.9, .55],
        coords_outlet=[.175, 0.9, .55],
        ftet_max_threads=24,
        multiwell=True,
        same_tissue_for_all_wells=True,
        well_spacing_y=0.6,
        inlet_base_marker=100,
        outlet_base_marker=200,
    )
