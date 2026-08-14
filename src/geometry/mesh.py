
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

import csv
import os, subprocess, shutil
import importlib
import re
import glob
import logging
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from send2trash import send2trash
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

from mpi4py import MPI
import adios4dolfinx
import dolfinx as dfx
from dolfinx.io import XDMFFile
from dolfinx import mesh as dmesh
from basix.ufl import element


class _LazyModule:
    """Import heavyweight, geometry-only dependencies on first use."""

    def __init__(self, module_name: str):
        self.module_name = module_name
        self.module = None

    def _load(self):
        if self.module is None:
            self.module = importlib.import_module(self.module_name)
        return self.module

    def __getattr__(self, name: str):
        return getattr(self._load(), name)


# Checkpoint regeneration does not use these large meshing/visualization
# packages. Loading them lazily keeps the Savio checkpoint path lightweight.
gmsh = _LazyModule("gmsh")
meshio = _LazyModule("meshio")
vtk = _LazyModule("vtk")
pv = _LazyModule("pyvista")


def vtk_to_numpy(*args, **kwargs):
    converter = importlib.import_module("vtk.util.numpy_support").vtk_to_numpy
    return converter(*args, **kwargs)

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


def _remove_generated_path(path: Path) -> None:
    """
    Remove generated files/directories before rewriting them.

    On HPC scratch filesystems, send2trash can fail when the volume-level
    .Trash-$UID directory is not writable. Fall back to direct removal so
    generated ADIOS .bp outputs can be refreshed in scratch.
    """
    path = Path(path)
    if not path.exists():
        return
    try:
        send2trash(path)
        return
    except Exception as exc:
        logger.warning("send2trash failed for %s (%s); removing directly", path, exc)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _read_branching_dataframe(csv_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _copy_artifact(src: Path, dst: Path) -> None:
    if dst.exists():
        _remove_generated_path(dst)
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


def _normalize(v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=float).reshape(-1)
    if vv.size != 3:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    n = float(np.linalg.norm(vv))
    if n <= 0.0:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vv / n


def _terminal_row_mask(df: pd.DataFrame) -> np.ndarray:
    child_cols = [c for c in df.columns if re.match(r"(?i)^child\d+$", str(c))]
    if not child_cols:
        return np.ones(len(df), dtype=bool)
    child_df = df[child_cols]
    empty = child_df.isna() | child_df.astype(str).apply(lambda s: s.str.strip() == "")
    return empty.all(axis=1).to_numpy(dtype=bool)


def terminal_interface_metadata_from_branching(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract terminal distal coordinates, axial normals and terminal areas.
    Columns expected from your branchingData CSV:
      distalCoordsX/Y/Z, W1/W2/W3, Radius (or Area).
    """
    df = _read_branching_dataframe(csv_path)
    if len(df) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0,))

    mask = _terminal_row_mask(df)
    coords = df.loc[mask, ["distalCoordsX", "distalCoordsY", "distalCoordsZ"]].to_numpy(dtype=float)

    if all(c in df.columns for c in ("W1", "W2", "W3")):
        normals = df.loc[mask, ["W1", "W2", "W3"]].to_numpy(dtype=float)
    else:
        p0 = df.loc[mask, ["proximalCoordsX", "proximalCoordsY", "proximalCoordsZ"]].to_numpy(dtype=float)
        normals = coords - p0
    normals = np.vstack([_normalize(v) for v in normals]) if len(normals) else np.zeros((0, 3))

    if "Area" in df.columns:
        areas = df.loc[mask, "Area"].to_numpy(dtype=float)
    elif "Radius" in df.columns:
        r = df.loc[mask, "Radius"].to_numpy(dtype=float)
        areas = np.pi * np.maximum(r, 0.0) ** 2
    else:
        areas = np.zeros((coords.shape[0],), dtype=float)

    return coords, normals, areas


def _drop_seed_terminal(
    coords: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    seed: np.ndarray,
    tol: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove at most one terminal: the nearest one if it lies within `tol`."""
    c = np.asarray(coords, dtype=float).reshape(-1, 3)
    n = np.asarray(normals, dtype=float).reshape(-1, 3)
    a = np.asarray(areas, dtype=float).reshape(-1)
    if len(c) == 0:
        return c, n, a
    s = np.asarray(seed, dtype=float).reshape(3,)
    d = np.linalg.norm(c - s[None, :], axis=1)
    i = int(np.argmin(d))
    if float(d[i]) > float(tol):
        return c, n, a
    keep = np.ones(len(c), dtype=bool)
    keep[i] = False
    return c[keep], n[keep], a[keep]


def _drop_nearest_seed_terminal(
    coords: np.ndarray,
    normals: np.ndarray,
    areas: np.ndarray,
    seed: np.ndarray,
    max_dist: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Drop exactly one terminal: the one nearest to `seed`, if close enough.
    This is more robust than exact coordinate matching after translations/rounding.
    """
    c = np.asarray(coords, dtype=float).reshape(-1, 3)
    n = np.asarray(normals, dtype=float).reshape(-1, 3)
    a = np.asarray(areas, dtype=float).reshape(-1)
    if len(c) == 0:
        return c, n, a
    s = np.asarray(seed, dtype=float).reshape(3,)
    d = np.linalg.norm(c - s[None, :], axis=1)
    i = int(np.argmin(d))
    if float(d[i]) > float(max_dist):
        return c, n, a
    keep = np.ones(len(c), dtype=bool)
    keep[i] = False
    return c[keep], n[keep], a[keep]

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


def _parse_branch_segment_name(name: str) -> Tuple[int, int]:
    raw = str(name).strip()
    m = re.search(r"branch[_\-\s]*(\d+)[_\-\s]*seg[_\-\s]*(\d+)", raw, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))

    parts = raw.split("_")
    if len(parts) >= 2:
        try:
            branch = int(parts[0].replace("branch", "").strip(" _-"))
            segment = int(parts[1].replace("seg", "").strip(" _-"))
            return branch, segment
        except Exception as exc:
            raise ValueError(f"Failed to parse branch/segment from {name!r}") from exc

    raise ValueError(f"Unrecognized 1D vessel name format: {name!r}")


def load_csv_timeseries(
    directory: str,
    n_axial_samples: int = 101,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the same centerline-style arrays previously recovered from VTP files,
    but directly from output.csv + geom.csv.

    Returns
    -------
    centerlineVel, centerlineFlow, pressure, centerlineCoords, area, times, tangents
    """
    base = Path(directory)
    geom_path = base / "geom.csv"
    output_path = base / "output.csv"
    if not geom_path.exists():
        raise FileNotFoundError(f"Missing geom.csv for direct checkpoint generation: {geom_path}")
    if not output_path.exists():
        raise FileNotFoundError(f"Missing output.csv for direct checkpoint generation: {output_path}")

    geom_data = np.genfromtxt(geom_path, delimiter=",", dtype=float)
    if geom_data.ndim == 1:
        geom_data = geom_data.reshape(1, -1)
    if geom_data.size == 0 or geom_data.shape[1] < 8:
        raise ValueError(f"Expected geom.csv with at least 8 columns, got shape {geom_data.shape} from {geom_path}")

    rows_by_key: Dict[Tuple[int, int], Dict[float, Tuple[float, float, float, float]]] = {}
    with output_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            branch, segment = _parse_branch_segment_name(row["name"])
            time = float(row["time"])
            flow_in = float(row["flow_in"])
            flow_out = float(row["flow_out"])
            pressure_in = float(row["pressure_in"])
            pressure_out = float(row["pressure_out"])
            rows_by_key.setdefault((branch, segment), {})[time] = (
                flow_in,
                flow_out,
                pressure_in,
                pressure_out,
            )

    if not rows_by_key:
        raise ValueError(f"No 1D timeseries rows were found in {output_path}")

    ordered_keys = sorted(rows_by_key.keys())
    ref_times = np.asarray(sorted(rows_by_key[ordered_keys[0]].keys()), dtype=float)
    if ref_times.size == 0:
        raise ValueError(f"No timesteps found in {output_path}")

    ordered_profiles: Dict[Tuple[int, int], List[Tuple[float, float, float, float]]] = {}
    for key in ordered_keys:
        time_map = rows_by_key[key]
        key_times = np.asarray(sorted(time_map.keys()), dtype=float)
        if key_times.size != ref_times.size or not np.allclose(key_times, ref_times, rtol=1e-12, atol=1e-12):
            raise ValueError(
                f"Inconsistent timestep grids in {output_path}; "
                f"{key} has {key_times.size} times, expected {ref_times.size}"
            )
        ordered_profiles[key] = [time_map[t] for t in key_times]

    s = np.linspace(0.0, 1.0, int(max(n_axial_samples, 2)), dtype=float)
    pressure_by_time: List[List[np.ndarray]] = [[] for _ in range(ref_times.size)]
    flow_by_time: List[List[np.ndarray]] = [[] for _ in range(ref_times.size)]
    centerline_coords_parts: List[np.ndarray] = []
    tangents_parts: List[np.ndarray] = []
    area_parts: List[np.ndarray] = []

    for branch, segment in ordered_keys:
        if branch < 0 or branch >= len(geom_data):
            raise IndexError(
                f"Branch index {branch} referenced in {output_path} is out of range for geom.csv with "
                f"{len(geom_data)} rows"
            )
        row = geom_data[branch]
        start = np.asarray(row[0:3], dtype=float)
        end = np.asarray(row[3:6], dtype=float)
        radius = float(row[7])
        area_value = float(np.pi * max(radius, 0.0) ** 2)
        tangent = _normalize(end - start)

        centerline_coords_parts.append(start[None, :] + s[:, None] * (end - start)[None, :])
        tangents_parts.append(np.tile(tangent[None, :], (s.size, 1)))
        area_parts.append(np.full(s.size, area_value, dtype=float))

        for tidx, (flow_in, flow_out, pressure_in, pressure_out) in enumerate(ordered_profiles[(branch, segment)]):
            flow_by_time[tidx].append(np.linspace(flow_in, flow_out, s.size, dtype=float))
            pressure_by_time[tidx].append(
                np.linspace(pressure_in / 1333.22, pressure_out / 1333.22, s.size, dtype=float)
            )

    centerline_coords = np.vstack(centerline_coords_parts)
    tangents = np.vstack(tangents_parts)
    area = np.concatenate(area_parts)
    centerline_flow = [np.concatenate(chunks) for chunks in flow_by_time]
    pressure = [np.concatenate(chunks) for chunks in pressure_by_time]
    centerline_vel = [flow / np.maximum(area, 1e-30) for flow in centerline_flow]
    return centerline_vel, centerline_flow, pressure, centerline_coords, area, ref_times, tangents


def _write_tagged_branch_bp(
    mesh: dfx.mesh.Mesh,
    facet_tag: dfx.mesh.MeshTags,
    file_prefix: str,
    rewrite: bool,
) -> None:
    path = current_dir / f"tagged_branches{file_prefix}.bp"
    if path.exists() and not rewrite:
        return
    if path.exists():
        _remove_generated_path(path)
    adios4dolfinx.write_mesh(Path(str(path)), mesh, engine="BP4")
    adios4dolfinx.write_meshtags(Path(str(path)), mesh, facet_tag, engine="BP4")


def generate_1d_files_from_csv(
    xdmf_file: str,
    output_dir: str,
    file_prefix: str = "",
    inlet_coords: np.ndarray = None,
    n_axial_samples: int = 101,
    rewrite_branch_tags: bool = False,
):
    """
    Faster checkpoint-generation path for coupled iterations.

    Reads output.csv + geom.csv directly and writes the BP checkpoint files used
    by Darcy/transport, bypassing plot_0d_results_to_3d.py and all VTP/PNG
    intermediate outputs.
    """
    with XDMFFile(MPI.COMM_WORLD, current_dir / xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    P1 = dfx.fem.functionspace(mesh, ("CG", 1))
    P1_vec = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    P1vec = dfx.fem.functionspace(mesh, P1_vec)
    dof_coords = P1.tabulate_dof_coordinates()

    velocity_fn = dfx.fem.Function(P1vec)
    pressure_fn = dfx.fem.Function(P1)
    flow_fn = dfx.fem.Function(P1)
    area_fn = dfx.fem.Function(P1)

    centerlineVel, centerlineFlow, pressure, centerlineCoords, area, times, tangents = load_csv_timeseries(
        output_dir, n_axial_samples=n_axial_samples
    )

    fdim = mesh.topology.dim - 1
    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    outlet_coords, facet_tag = branch_mesh_tagging(mesh, inlet=inlet_coords)
    _write_tagged_branch_bp(mesh, facet_tag, file_prefix, rewrite=rewrite_branch_tags)

    tree = cKDTree(centerlineCoords)
    _, indices = tree.query(dof_coords)
    indices = np.asarray(indices, dtype=int)
    sampled_tangents = tangents[indices]

    if (current_dir / f"velocity_checkpoint{file_prefix}.bp").exists():
        _remove_generated_path(current_dir / f"velocity_checkpoint{file_prefix}.bp")
    if (current_dir / f"pressure_checkpoint{file_prefix}.bp").exists():
        _remove_generated_path(current_dir / f"pressure_checkpoint{file_prefix}.bp")
    if (current_dir / f"flow_checkpoint{file_prefix}.bp").exists():
        _remove_generated_path(current_dir / f"flow_checkpoint{file_prefix}.bp")
    if (current_dir / f"area_checkpoint{file_prefix}.bp").exists():
        _remove_generated_path(current_dir / f"area_checkpoint{file_prefix}.bp")

    interpolated_area = area[indices]
    for tidx, time_value in enumerate(times):
        interpolated_velocity = centerlineVel[tidx][indices]
        interpolated_pressure = pressure[tidx][indices]
        interpolated_flow = centerlineFlow[tidx][indices]

        velocity_fn.x.array[:] = (interpolated_velocity[:, np.newaxis] * sampled_tangents).flatten()
        pressure_fn.x.array[:] = interpolated_pressure
        flow_fn.x.array[:] = interpolated_flow
        area_fn.x.array[:] = interpolated_area

        adios4dolfinx.write_function(
            Path(current_dir / f"velocity_checkpoint{file_prefix}.bp"),
            u=velocity_fn,
            time=float(time_value),
            name="f",
        )
        adios4dolfinx.write_function(
            Path(current_dir / f"pressure_checkpoint{file_prefix}.bp"),
            u=pressure_fn,
            time=float(time_value),
            name="f",
        )
        adios4dolfinx.write_function(
            Path(current_dir / f"flow_checkpoint{file_prefix}.bp"),
            u=flow_fn,
            time=float(time_value),
            name="f",
        )
        adios4dolfinx.write_function(
            Path(current_dir / f"area_checkpoint{file_prefix}.bp"),
            u=area_fn,
            time=float(time_value),
            name="f",
        )

    return outlet_coords


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
        _remove_generated_path(current_dir / f"tagged_branches{file_prefix}.bp")

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
        _remove_generated_path(current_dir / f"velocity_checkpoint{file_prefix}.bp")
    if (current_dir / f"pressure_checkpoint{file_prefix}.bp").exists():
        _remove_generated_path(current_dir / f"pressure_checkpoint{file_prefix}.bp")
    if (current_dir / f"flow_checkpoint{file_prefix}.bp").exists():
        _remove_generated_path(current_dir / f"flow_checkpoint{file_prefix}.bp")
    if (current_dir / f"area_checkpoint{file_prefix}.bp").exists():
        _remove_generated_path(current_dir / f"area_checkpoint{file_prefix}.bp")

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
    df = _read_branching_dataframe(branching_data_file)
    if len(df) == 0:
        logger.info("[1d] branching file %s has no rows; skipping 1D branch generation", branching_data_file)
        return np.zeros((0, 3), dtype=float)
    branch.write_geo_from_branching_data(df, geo_file=geo_file)
    geo_to_mesh_gmsh(geo_file=geo_file, msh_file=msh_file)
    convert_mesh(msh_file=msh_file, xdmf_file=xdmf_file)
    _ = xdmf_to_dolfinx(xdmf_file=xdmf_file)
    output_dir = Path(output_1d)
    geom_csv = output_dir / "geom.csv"
    output_csv = output_dir / "output.csv"
    if geom_csv.exists() and output_csv.exists():
        outlet_coords = generate_1d_files_from_csv(
            xdmf_file=xdmf_file,
            output_dir=output_1d,
            file_prefix=fileprefix,
            inlet_coords=coords,
        )
    else:
        outlet_coords = generate_1d_files(
            xdmf_file=xdmf_file,
            output_dir=output_1d,
            file_prefix=fileprefix,
            inlet_coords=coords,
        )
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


def read_3d_mesh_from_stl_with_embedded_terminals(
    stl_file: str,
    msh_file: str,
    terminal_centers: np.ndarray,
    terminal_normals: np.ndarray,
    terminal_areas: np.ndarray,
    terminal_markers: np.ndarray,
    *,
    wall_marker: int = 1,
    char_len_min: float = 0.001,
    char_len_max: float = 0.01,
    disk_npts: int = 24,
    terminal_refine_radius_factor: float = 2.5,
    terminal_refine_h_factor: float = 0.35,
) -> Tuple[dfx.mesh.Mesh, dfx.mesh.MeshTags]:
    stl_to_mesh_gmsh_with_embedded_disks(
        stl_file=stl_file,
        msh_file=msh_file,
        disk_centers=terminal_centers,
        disk_normals=terminal_normals,
        disk_areas=terminal_areas,
        disk_markers=terminal_markers,
        wall_marker=wall_marker,
        char_len_min=char_len_min,
        char_len_max=char_len_max,
        disk_npts=disk_npts,
        terminal_refine_radius_factor=terminal_refine_radius_factor,
        terminal_refine_h_factor=terminal_refine_h_factor,
    )
    mesh, facet_tags = read_3d_mesh_and_tags_from_msh(msh_file)
    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_entities(fdim)
    mesh.topology.create_connectivity(fdim, 0)
    mesh.topology.create_connectivity(tdim, fdim)
    _print_terminal_marker_coverage(
        facet_tags,
        list(zip(np.asarray(terminal_markers, dtype=int).tolist(), np.asarray(terminal_centers, dtype=float))),
        context="stl-embedded",
    )
    return mesh, facet_tags


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
    z_align_tol = 1.0

    n0 = normals[seed_i].copy()
    nn = np.linalg.norm(n0)
    if nn == 0:
        raise RuntimeError("Seed facet has zero normal.")
    n0 /= nn

    if abs(float(n0[2])) >= z_align_tol:
        order = np.argsort(d2)
        replacement = None
        for idx in order:
            cand_n = normals[int(idx)].copy()
            cand_nn = np.linalg.norm(cand_n)
            if cand_nn == 0:
                continue
            cand_n /= cand_nn
            if abs(float(cand_n[2])) < z_align_tol:
                replacement = int(idx)
                break
        if replacement is None:
            raise RuntimeError("Could not find a non-z-aligned seed facet for concave tagging.")
        seed_i = replacement
        seed_f = int(all_bdry_facets[seed_i])
        n0 = normals[seed_i].copy()
        n0 /= np.linalg.norm(n0)

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
        nf_norm = np.linalg.norm(nf)
        if nf_norm == 0:
            continue
        nf = nf / nf_norm
        if abs(float(nf[2])) >= z_align_tol:
            continue
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
    excluded_facets: Optional[Iterable[int]] = None,
    normal_hint: Optional[np.ndarray] = None,
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
    excluded = {int(f) for f in excluded_facets} if excluded_facets is not None else set()

    x0 = np.asarray(x0, float)
    d2 = np.sum((centers - x0) ** 2, axis=1)
    normal_hint_arr = None
    if normal_hint is not None:
        nh = np.asarray(normal_hint, dtype=float).reshape(-1)
        if nh.size == 3:
            nrm = np.linalg.norm(nh)
            if nrm > 0:
                normal_hint_arr = nh / nrm

    if plane_tol is None:
        bbox = mesh.geometry.x.max(axis=0) - mesh.geometry.x.min(axis=0)
        plane_tol = 0.002 * float(np.linalg.norm(bbox))

    cos_thr = float(np.cos(np.deg2rad(theta_deg)))

    if normal_hint_arr is not None:
        align = np.abs(normals @ normal_hint_arr)
        cand = np.where(align >= cos_thr)[0]
        if cand.size > 0:
            plane_band = np.abs((centers[cand] - x0[None, :]) @ normal_hint_arr) <= float(max(plane_tol, 1e-12))
            if np.any(plane_band):
                cand = cand[plane_band]
        if cand.size > 0:
            order = cand[np.argsort(d2[cand])]
        else:
            order = np.argsort(d2)
    else:
        order = np.argsort(d2)

    seed_i = None
    seed_f = None
    for idx in order:
        cand_f = int(all_bdry_facets[int(idx)])
        if cand_f in excluded:
            continue
        seed_i = int(idx)
        seed_f = cand_f
        break
    if seed_i is None or seed_f is None:
        raise RuntimeError(f"No available terminal seed facet for x0={x0}.")

    if normal_hint_arr is not None:
        n0 = normal_hint_arr
        x_seed = x0.copy()
    else:
        n0 = normals[seed_i].copy()
        nn = np.linalg.norm(n0)
        if nn == 0:
            raise RuntimeError("Terminal seed facet has zero normal.")
        n0 /= nn
        x_seed = centers[seed_i].copy()

    selected = set()
    visited = set()
    stack = [seed_f]
    while stack:
        f = stack.pop()
        if f in visited:
            continue
        visited.add(f)
        if f in excluded:
            continue
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
        raise RuntimeError(
            f"Terminal patch empty for x0={x0}. Try increasing plane_tol/theta_deg or reduce overlap with neighboring terminals."
        )

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


def _disk_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = _normalize(n)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = np.cross(n, ref)
    u = _normalize(u)
    v = np.cross(n, u)
    v = _normalize(v)
    return u, v


def _build_stl_inside_tester(stl_file: str, tol: float = 1e-9):
    """
    Build a robust point-in-volume tester from the STL surface using VTK.
    Returns (inside_fn, cleanup_fn).
    """
    reader = vtk.vtkSTLReader()
    reader.SetFileName(str(stl_file))
    reader.Update()
    surface = reader.GetOutput()

    enclosed = vtk.vtkSelectEnclosedPoints()
    enclosed.SetTolerance(float(tol))
    enclosed.Initialize(surface)

    def _inside(point: np.ndarray) -> bool:
        p = np.asarray(point, dtype=float).reshape(3,)
        return bool(enclosed.IsInsideSurface(float(p[0]), float(p[1]), float(p[2])))

    def _cleanup() -> None:
        try:
            enclosed.Complete()
        except Exception:
            pass

    return _inside, _cleanup


def _trim_disk_to_volume(
    center: np.ndarray,
    normal: np.ndarray,
    radius: float,
    inside_fn,
    *,
    n_angles: int,
) -> np.ndarray:
    """
    Keep the exact disk center/orientation/radius and trim only the portion that
    lies outside the tissue volume by sampling interior points in the disk plane.
    This works even when the disk center is outside the tissue volume.
    """
    c0 = np.asarray(center, dtype=float).reshape(3,)
    n = _normalize(normal)
    r = float(radius)
    u, v = _disk_basis_from_normal(n)
    nang = int(max(32, n_angles))
    nr = 18
    r_levels = np.linspace(0.0, r, nr + 1)

    def _sample_disk(center_shifted: np.ndarray):
        pts3: List[np.ndarray] = []
        pts2: List[np.ndarray] = []
        for rho in r_levels:
            for j in range(nang):
                th = 2.0 * np.pi * (j / float(nang))
                d = np.cos(th) * u + np.sin(th) * v
                p = center_shifted + rho * d
                if inside_fn(p):
                    pts3.append(p)
                    pts2.append(np.array([rho * np.cos(th), rho * np.sin(th)], dtype=float))
        return pts3, pts2

    # Near the outer wall, some terminal centers sit just outside the tissue.
    # Retry a few small offsets along +/- normal and keep the richest in-volume cut.
    offsets = [0.0, 0.15 * r, -0.15 * r, 0.30 * r, -0.30 * r, 0.50 * r, -0.50 * r]
    best_center = c0
    best_pts2: List[np.ndarray] = []
    for s in offsets:
        c = c0 + float(s) * n
        _pts3, _pts2 = _sample_disk(c)
        if len(_pts2) > len(best_pts2):
            best_center = c
            best_pts2 = _pts2

    if len(best_pts2) < 3:
        return np.zeros((0, 3), dtype=float)

    xy = np.asarray(best_pts2, dtype=float)
    try:
        hull = ConvexHull(xy)
        hull_xy = xy[hull.vertices]
    except Exception:
        ctr = xy.mean(axis=0)
        ang = np.arctan2(xy[:, 1] - ctr[1], xy[:, 0] - ctr[0])
        order = np.argsort(ang)
        hull_xy = xy[order]

    poly = np.zeros((len(hull_xy), 3), dtype=float)
    for i, (x, y) in enumerate(hull_xy):
        poly[i] = best_center + x * u + y * v
    return poly


def stl_to_mesh_gmsh_with_embedded_disks(
    stl_file: str,
    msh_file: str,
    disk_centers: np.ndarray,
    disk_normals: np.ndarray,
    disk_areas: np.ndarray,
    disk_markers: np.ndarray,
    *,
    wall_marker: int = 1,
    char_len_min: float = 0.001,
    char_len_max: float = 0.01,
    disk_npts: int = 24,
    terminal_refine_radius_factor: float = 2.5,
    terminal_refine_h_factor: float = 0.35,
    enable_optimization: bool = False,
) -> None:
    """
    Build tetra mesh from STL and embed terminal disks as INTERNAL surfaces.
    Each disk gets its own 2D physical marker: disk_base_marker + k.
    """
    centers = np.asarray(disk_centers, dtype=float).reshape(-1, 3)
    normals = np.asarray(disk_normals, dtype=float).reshape(-1, 3)
    areas = np.asarray(disk_areas, dtype=float).reshape(-1)
    markers = np.asarray(disk_markers, dtype=np.int32).reshape(-1)
    if len(centers) != len(normals) or len(centers) != len(areas) or len(centers) != len(markers):
        raise ValueError("disk_centers/disk_normals/disk_areas/disk_markers must have identical length.")

    inside_fn, inside_cleanup = _build_stl_inside_tester(stl_file)

    gmsh.initialize()
    gmsh.model.add("bioreactor_with_embedded_terminals")
    gmsh.merge(str(stl_file))
    gmsh.model.mesh.removeDuplicateNodes()

    angle = 30
    force_param = True
    include_boundary = True
    curve_angle = 180
    gmsh.model.mesh.classifySurfaces(
        angle * (np.pi / 180.0),
        include_boundary,
        force_param,
        curve_angle * (np.pi / 180.0),
    )
    gmsh.model.mesh.createGeometry()
    gmsh.model.mesh.createTopology()

    boundary_surfaces = [s[1] for s in gmsh.model.getEntities(2)]
    gmsh.model.geo.addSurfaceLoop(boundary_surfaces, 1)
    vol = gmsh.model.geo.addVolume([1], 1)
    gmsh.model.geo.synchronize()

    disk_surfaces: List[int] = []
    terminal_refine_specs: List[Tuple[np.ndarray, float]] = []
    try:
        for k, (c, n, a, marker) in enumerate(zip(centers, normals, areas, markers)):
            if not np.isfinite(a) or float(a) <= 0.0:
                continue
            r = float(np.sqrt(float(a) / np.pi))
            if r <= 0.0:
                continue
            poly_pts = _trim_disk_to_volume(
                center=c,
                normal=n,
                radius=r,
                inside_fn=inside_fn,
                n_angles=disk_npts,
            )
            if len(poly_pts) < 3:
                print(
                    "[3d-tags] skipping embedded disk "
                    f"k={int(k)} marker={int(marker)} center="
                    f"{np.array2string(np.asarray(c, dtype=float), precision=6, separator=', ')} "
                    "reason=no sufficient in-volume intersection",
                    flush=True,
                )
                continue
            h_local = float(max(0.1 * float(char_len_min), min(float(char_len_max), 0.33 * r)))
            pts = []
            for x in poly_pts:
                pts.append(gmsh.model.geo.addPoint(float(x[0]), float(x[1]), float(x[2]), h_local))
            lines = []
            for j in range(len(pts)):
                lines.append(gmsh.model.geo.addLine(pts[j], pts[(j + 1) % len(pts)]))
            loop = gmsh.model.geo.addCurveLoop(lines)
            s = gmsh.model.geo.addPlaneSurface([loop])
            disk_surfaces.append(s)
            terminal_refine_specs.append((np.asarray(c, dtype=float).copy(), float(r)))
            gmsh.model.geo.synchronize()
            gmsh.model.addPhysicalGroup(2, [s], int(marker))
            gmsh.model.setPhysicalName(2, int(marker), f"terminal_disk_{k}")
    finally:
        inside_cleanup()

    gmsh.model.geo.synchronize()
    if disk_surfaces:
        gmsh.model.mesh.embed(2, disk_surfaces, 3, vol)

    gmsh.model.addPhysicalGroup(3, [vol], 1)
    gmsh.model.setPhysicalName(3, 1, "volume")
    bnd_after = gmsh.model.getBoundary([(3, int(vol))], oriented=False, recursive=False)
    boundary_surfaces_after = [e[1] for e in bnd_after if e[0] == 2]
    disk_surface_set = set(int(s) for s in disk_surfaces)
    wall_surfaces = [int(s) for s in boundary_surfaces_after if int(s) not in disk_surface_set]
    if wall_surfaces:
        gmsh.model.addPhysicalGroup(2, wall_surfaces, int(wall_marker))
        gmsh.model.setPhysicalName(2, int(wall_marker), "outer_wall")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", float(char_len_max))
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", float(char_len_min))
    # Local volumetric refinement around terminal interfaces.
    if terminal_refine_specs:
        fields = []
        rr_fac = float(max(1.0, terminal_refine_radius_factor))
        hh_fac = float(max(0.05, terminal_refine_h_factor))
        for c, r in terminal_refine_specs:
            f = gmsh.model.mesh.field.add("Ball")
            gmsh.model.mesh.field.setNumber(f, "XCenter", float(c[0]))
            gmsh.model.mesh.field.setNumber(f, "YCenter", float(c[1]))
            gmsh.model.mesh.field.setNumber(f, "ZCenter", float(c[2]))
            gmsh.model.mesh.field.setNumber(f, "Radius", float(max(rr_fac * r, char_len_min)))
            h_in = max(0.1 * float(char_len_min), min(float(char_len_max), hh_fac * float(r)))
            gmsh.model.mesh.field.setNumber(f, "VIn", float(h_in))
            gmsh.model.mesh.field.setNumber(f, "VOut", float(char_len_max))
            fields.append(f)
        if fields:
            fmin = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", fields)
            gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)
    # Embedded surfaces can trigger crashes in the optimizer on some builds;
    # keep optimization off by default for robustness.
    gmsh.option.setNumber("Mesh.Optimize", 1 if enable_optimization else 0)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1 if enable_optimization else 0)
    gmsh.option.setNumber("Mesh.Smoothing", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(3)
    gmsh.write(str(current_dir / msh_file))
    logger.info("Created embedded-disk mesh %s from %s", current_dir / msh_file, stl_file)
    gmsh.finalize()


def read_3d_mesh_and_tags_from_msh(msh_file: str) -> Tuple[dfx.mesh.Mesh, dfx.mesh.MeshTags]:
    try:
        gmshio = importlib.import_module("dolfinx.io.gmshio")
    except ImportError:
        gmshio = importlib.import_module("dolfinx.io.gmsh")
    out = gmshio.read_from_msh(str(current_dir / msh_file), MPI.COMM_WORLD, 0, gdim=3)
    if not isinstance(out, tuple):
        raise RuntimeError(f"Unexpected gmshio.read_from_msh return type for {msh_file}: {type(out)!r}")
    if len(out) < 3:
        raise RuntimeError(
            f"Unexpected gmshio.read_from_msh return length for {msh_file}: "
            f"expected at least 3 entries, got {len(out)}"
        )
    mesh = out[0]
    facet_tags = out[2]
    if facet_tags is None:
        raise RuntimeError(f"No facet tags read from {msh_file}")
    return mesh, facet_tags

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


def _print_terminal_marker_coverage(
    facet_tags: dfx.mesh.MeshTags,
    marker_points: Sequence[Tuple[int, np.ndarray]],
    *,
    context: str,
) -> None:
    values = np.asarray(facet_tags.values, dtype=np.int32)
    present = 0
    missing: List[Tuple[int, np.ndarray]] = []

    for marker, point in marker_points:
        count = int(np.count_nonzero(values == int(marker)))
        if count > 0:
            present += 1
        else:
            missing.append((int(marker), np.asarray(point, dtype=float).reshape(3,)))

    print(f"[3d-tags] {context}: present={present}/{len(marker_points)} terminal markers", flush=True)
    for marker, point in missing:
        print(
            f"[3d-tags] {context} missing marker={marker} at point="
            f"{np.array2string(point, precision=6, separator=', ')}",
            flush=True,
        )

def build_facet_tags_for_mesh(
    mesh: dfx.mesh.Mesh,
    inlet_terminal_pts: np.ndarray,
    outlet_terminal_pts: np.ndarray,
    inlet_terminal_normals: Optional[np.ndarray],
    outlet_terminal_normals: Optional[np.ndarray],
    coords_inlet: np.ndarray,
    coords_outlet: np.ndarray,
    *,
    wall_marker: int = 1,
    arterial_concave_marker: int = 31,
    venous_concave_marker: int = 32,
    inlet_base_marker: int = 1000,
    outlet_base_marker: int = 2000,
    theta_concave_deg: float = 60.0,
    theta_concave_inlet_deg: Optional[float] = None,
    theta_concave_outlet_deg: Optional[float] = None,
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
    shifted_inlet = coords_inlet[:] + np.array([0, 0.0, 0.1])  # shift slightly to ensure seed is inside
    shifted_outlet = coords_outlet[:] + np.array([0, 0.0, 0.1])
    if theta_concave_inlet_deg is None:
        theta_concave_inlet_deg = theta_concave_deg
    if theta_concave_outlet_deg is None:
        theta_concave_outlet_deg = theta_concave_deg
    conc_art = tag_surface_from_seed(
        mesh, shifted_inlet, theta_deg=theta_concave_inlet_deg, marker=arterial_concave_marker
    )
    conc_ven = tag_surface_from_seed(
        mesh, shifted_outlet, theta_deg=theta_concave_outlet_deg, marker=venous_concave_marker
    )
    for f, v in zip(conc_art.indices, conc_art.values):
        facet_to_marker[int(f)] = int(v)
    for f, v in zip(conc_ven.indices, conc_ven.values):
        facet_to_marker[int(f)] = int(v)

    # 3) terminal faces (overwrite everything)
    # Keep all inlet/outlet terminal points; do not drop seed-corresponding terminals.
    used_terminal_facets: set[int] = set()
    # inlet terminals
    for k, pt in enumerate(np.asarray(inlet_terminal_pts, float)):
        tag = inlet_base_marker + k
        normal_hint = None
        if inlet_terminal_normals is not None and k < len(inlet_terminal_normals):
            normal_hint = np.asarray(inlet_terminal_normals[k], dtype=float)
        tpatch = tag_terminal_plane_patch(
            mesh,
            pt,
            marker=tag,
            theta_deg=theta_terminal_deg,
            plane_tol=plane_tol,
            excluded_facets=used_terminal_facets,
            normal_hint=normal_hint,
        )
        for f, v in zip(tpatch.indices, tpatch.values):
            facet_to_marker[int(f)] = int(v)
            used_terminal_facets.add(int(f))

    # outlet terminals
    for k, pt in enumerate(np.asarray(outlet_terminal_pts, float)):
        tag = outlet_base_marker + k
        normal_hint = None
        if outlet_terminal_normals is not None and k < len(outlet_terminal_normals):
            normal_hint = np.asarray(outlet_terminal_normals[k], dtype=float)
        tpatch = tag_terminal_plane_patch(
            mesh,
            pt,
            marker=tag,
            theta_deg=theta_terminal_deg,
            plane_tol=plane_tol,
            excluded_facets=used_terminal_facets,
            normal_hint=normal_hint,
        )
        for f, v in zip(tpatch.indices, tpatch.values):
            facet_to_marker[int(f)] = int(v)
            used_terminal_facets.add(int(f))

    facet_indices = np.array(sorted(facet_to_marker.keys()), dtype=np.int32)
    facet_values = np.array([facet_to_marker[i] for i in facet_indices], dtype=np.int32)
    facet_tags = dmesh.meshtags(mesh, fdim, facet_indices, facet_values)
    marker_points: List[Tuple[int, np.ndarray]] = []
    for k, pt in enumerate(np.asarray(inlet_terminal_pts, float)):
        marker_points.append((int(inlet_base_marker + k), np.asarray(pt, dtype=float)))
    for k, pt in enumerate(np.asarray(outlet_terminal_pts, float)):
        marker_points.append((int(outlet_base_marker + k), np.asarray(pt, dtype=float)))
    _print_terminal_marker_coverage(facet_tags, marker_points, context="vtu-boundary")
    return facet_tags


def overlay_concave_markers(
    mesh: dfx.mesh.Mesh,
    facet_tags: dfx.mesh.MeshTags,
    *,
    coords_inlet: np.ndarray,
    coords_outlet: np.ndarray,
    arterial_concave_marker: int,
    venous_concave_marker: int,
    theta_concave_deg: float,
    theta_concave_inlet_deg: Optional[float] = None,
    theta_concave_outlet_deg: Optional[float] = None,
    terminal_base_marker: int,
) -> dfx.mesh.MeshTags:
    """
    Overlay concave wall markers on an existing facet tag set while preserving
    embedded terminal markers (>= outlet_base_marker).
    """
    tdim = mesh.topology.dim
    fdim = tdim - 1
    facet_to_marker: Dict[int, int] = {
        int(f): int(v) for f, v in zip(facet_tags.indices, facet_tags.values)
    }

    shifted_inlet = np.asarray(coords_inlet, dtype=float) + np.array([0.0, 0.1, 0.0], dtype=float)
    shifted_outlet = np.asarray(coords_outlet, dtype=float) + np.array([0.0, 0.1, 0.0], dtype=float)
    if theta_concave_inlet_deg is None:
        theta_concave_inlet_deg = theta_concave_deg
    if theta_concave_outlet_deg is None:
        theta_concave_outlet_deg = theta_concave_deg
    conc_art = tag_surface_from_seed(
        mesh, shifted_inlet, theta_deg=theta_concave_inlet_deg, marker=arterial_concave_marker
    )
    conc_ven = tag_surface_from_seed(
        mesh, shifted_outlet, theta_deg=theta_concave_outlet_deg, marker=venous_concave_marker
    )

    for f, v in zip(conc_art.indices, conc_art.values):
        old = facet_to_marker.get(int(f), 0)
        if old < int(terminal_base_marker):
            facet_to_marker[int(f)] = int(v)
    for f, v in zip(conc_ven.indices, conc_ven.values):
        old = facet_to_marker.get(int(f), 0)
        if old < int(terminal_base_marker):
            facet_to_marker[int(f)] = int(v)

    idx = np.array(sorted(facet_to_marker.keys()), dtype=np.int32)
    vals = np.array([facet_to_marker[i] for i in idx], dtype=np.int32)
    return dmesh.meshtags(mesh, fdim, idx, vals)


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
        gmsh_char_len_min: float = 0.001,
        gmsh_char_len_max: float = 0.008,
        gmsh_disk_npts: int = 12,
        embedded_disk_radius_scale: float = 1.0,
        terminal_refine_radius_factor: float = 2.5,
        terminal_refine_h_factor: float = 0.35,
        multiwell: bool = True,
        same_tissue_for_all_wells: bool = True,
        same_meshtags_for_all_wells: bool = False,
        well_spacing_y: float = 0.6,
        disable_1d_terminals: bool = False,
        # tagging params
        wall_marker: int = 1,
        arterial_concave_marker: int = 31,
        venous_concave_marker: int = 32,
        inlet_base_marker: int = 1000,
        outlet_base_marker: int = 2000,
        theta_concave_deg: float = 60.0,
        theta_concave_inlet_deg: Optional[float] = None,
        theta_concave_outlet_deg: Optional[float] = None,
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
        inlet_terminal_normals_by_well: List[np.ndarray] = []
        inlet_terminal_areas_by_well: List[np.ndarray] = []
        outlet_terminal_pts_by_well: List[np.ndarray] = []
        outlet_terminal_normals_by_well: List[np.ndarray] = []
        outlet_terminal_areas_by_well: List[np.ndarray] = []
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

            if not disable_1d_terminals:
                if not Path(branching_in_list[i - 1]).exists():
                    raise FileNotFoundError(f"Missing branching_data_inlet for well{i}: {branching_in_list[i - 1]}")
                if not Path(branching_out_list[i - 1]).exists():
                    raise FileNotFoundError(f"Missing branching_data_outlet for well{i}: {branching_out_list[i - 1]}")
                if not Path(output_in_list[i - 1]).exists():
                    raise FileNotFoundError(f"Missing output_1d_inlet folder for well{i}: {output_in_list[i - 1]}")
                if not Path(output_out_list[i - 1]).exists():
                    raise FileNotFoundError(f"Missing output_1d_outlet folder for well{i}: {output_out_list[i - 1]}")

                in_meta_coords, in_meta_normals, in_meta_areas = terminal_interface_metadata_from_branching(
                    branching_in_list[i - 1]
                )
                out_meta_coords, out_meta_normals, out_meta_areas = terminal_interface_metadata_from_branching(
                    branching_out_list[i - 1]
                )
            else:
                in_meta_coords = np.zeros((0, 3), dtype=float)
                in_meta_normals = np.zeros((0, 3), dtype=float)
                in_meta_areas = np.zeros((0,), dtype=float)
                out_meta_coords = np.zeros((0, 3), dtype=float)
                out_meta_normals = np.zeros((0, 3), dtype=float)
                out_meta_areas = np.zeros((0,), dtype=float)
            # Keep all terminal metadata points. Do not drop points near the
            # inlet/outlet seeds; the terminal tagging stage now expects the
            # full metadata list to be preserved.
            # in_meta_coords, in_meta_normals, in_meta_areas = _drop_seed_terminal(
            #     in_meta_coords, in_meta_normals, in_meta_areas, self.coords_inlet
            # )
            # out_meta_coords, out_meta_normals, out_meta_areas = _drop_seed_terminal(
            #     out_meta_coords, out_meta_normals, out_meta_areas, self.coords_outlet
            # )
            # in_meta_coords, in_meta_normals, in_meta_areas = _drop_nearest_seed_terminal(
            #     in_meta_coords, in_meta_normals, in_meta_areas, c_in
            # )
            # out_meta_coords, out_meta_normals, out_meta_areas = _drop_nearest_seed_terminal(
            #     out_meta_coords, out_meta_normals, out_meta_areas, c_out
            # )
            # in_meta_coords, in_meta_normals, in_meta_areas = _drop_nearest_seed_terminal(
            #     in_meta_coords, in_meta_normals, in_meta_areas, self.coords_inlet
            # )
            # out_meta_coords, out_meta_normals, out_meta_areas = _drop_nearest_seed_terminal(
            #     out_meta_coords, out_meta_normals, out_meta_areas, self.coords_outlet
            # )

            if disable_1d_terminals:
                inlet_terminal_pts = np.zeros((0, 3), dtype=float)
                outlet_terminal_pts = np.zeros((0, 3), dtype=float)
            else:
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
            if len(in_meta_coords) > 0:
                inlet_terminal_pts_by_well.append(in_meta_coords)
                if len(in_meta_normals) == len(in_meta_coords):
                    inlet_terminal_normals_by_well.append(in_meta_normals)
                else:
                    inlet_terminal_normals_by_well.append(np.tile(np.array([1.0, 0.0, 0.0]), (len(in_meta_coords), 1)))
                if len(in_meta_areas) == len(in_meta_coords):
                    inlet_terminal_areas_by_well.append(in_meta_areas)
                else:
                    inlet_terminal_areas_by_well.append(np.full((len(in_meta_coords),), 1e-6, dtype=float))
            else:
                n_fallback_in = np.tile(_normalize(c_in - c_out), (len(inlet_terminal_pts), 1))
                a_fallback_in = np.full((len(inlet_terminal_pts),), 1e-6, dtype=float)
                inlet_terminal_pts_by_well.append(inlet_terminal_pts)
                inlet_terminal_normals_by_well.append(n_fallback_in)
                inlet_terminal_areas_by_well.append(a_fallback_in)
            if len(out_meta_coords) > 0:
                outlet_terminal_pts_by_well.append(out_meta_coords)
                if len(out_meta_normals) == len(out_meta_coords):
                    outlet_terminal_normals_by_well.append(out_meta_normals)
                else:
                    outlet_terminal_normals_by_well.append(np.tile(np.array([1.0, 0.0, 0.0]), (len(out_meta_coords), 1)))
                if len(out_meta_areas) == len(out_meta_coords):
                    outlet_terminal_areas_by_well.append(out_meta_areas)
                else:
                    outlet_terminal_areas_by_well.append(np.full((len(out_meta_coords),), 1e-6, dtype=float))
            else:
                n_fallback = np.tile(_normalize(c_out - c_in), (len(outlet_terminal_pts), 1))
                a_fallback = np.full((len(outlet_terminal_pts),), 1e-6, dtype=float)
                # Fallback if metadata unavailable: approximate normals from seed axis.
                outlet_terminal_pts_by_well.append(outlet_terminal_pts)
                outlet_terminal_normals_by_well.append(n_fallback)
                outlet_terminal_areas_by_well.append(a_fallback)
            logger.info(
                "[terminals] well=%d inlet=%d outlet=%d",
                i,
                len(inlet_terminal_pts_by_well[-1]),
                len(outlet_terminal_pts_by_well[-1]),
            )

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
                    inlet_terminal_normals=inlet_terminal_normals_by_well[i - 1],
                    outlet_terminal_normals=outlet_terminal_normals_by_well[i - 1],
                    coords_inlet=coords_inlet_by_well[i - 1],
                    coords_outlet=coords_outlet_by_well[i - 1],
                    wall_marker=wall_marker,
                    arterial_concave_marker=arterial_concave_marker,
                    venous_concave_marker=venous_concave_marker,
                    inlet_base_marker=inlet_base_marker,
                    outlet_base_marker=outlet_base_marker,
                    theta_concave_deg=theta_concave_deg,
                    theta_concave_inlet_deg=theta_concave_inlet_deg,
                    theta_concave_outlet_deg=theta_concave_outlet_deg,
                    theta_terminal_deg=theta_terminal_deg,
                    plane_tol=plane_tol,
                    outer_tol=outer_tol,
                )
                write_mesh_and_tags(mesh_i, facet_tags_i, f"bioreactor_well{i}.xdmf", f"mesh_tags_well{i}.xdmf")
            return

        if tissue_vtu is None and tissue_stl is None:
            raise ValueError("Provide tissue_vtu, tissue_stl, or tissue_vtu_list.")

        # STL path: create embedded inlet/outlet terminal disk surfaces as physical groups.
        if tissue_stl is not None:
            if not multiwell:
                terminal_centers = np.vstack([inlet_terminal_pts_by_well[0], outlet_terminal_pts_by_well[0]])
                terminal_normals = np.vstack([inlet_terminal_normals_by_well[0], outlet_terminal_normals_by_well[0]])
                terminal_areas = np.concatenate([inlet_terminal_areas_by_well[0], outlet_terminal_areas_by_well[0]])
                terminal_markers = np.concatenate(
                    [
                        inlet_base_marker + np.arange(len(inlet_terminal_pts_by_well[0]), dtype=np.int32),
                        outlet_base_marker + np.arange(len(outlet_terminal_pts_by_well[0]), dtype=np.int32),
                    ]
                )
                mesh, facet_tags = read_3d_mesh_from_stl_with_embedded_terminals(
                    stl_file=tissue_stl,
                    msh_file="_tmp_bioreactor_embedded.msh",
                    terminal_centers=terminal_centers,
                    terminal_normals=terminal_normals,
                    terminal_areas=terminal_areas,
                    terminal_markers=terminal_markers,
                    wall_marker=wall_marker,
                    char_len_min=gmsh_char_len_min,
                    char_len_max=gmsh_char_len_max,
                    disk_npts=gmsh_disk_npts,
                    terminal_refine_radius_factor=terminal_refine_radius_factor,
                    terminal_refine_h_factor=terminal_refine_h_factor,
                )
                facet_tags = overlay_concave_markers(
                    mesh,
                    facet_tags,
                    coords_inlet=coords_inlet_by_well[0],
                    coords_outlet=coords_outlet_by_well[0],
                    arterial_concave_marker=arterial_concave_marker,
                    venous_concave_marker=venous_concave_marker,
                    theta_concave_deg=theta_concave_deg,
                    theta_concave_inlet_deg=theta_concave_inlet_deg,
                    theta_concave_outlet_deg=theta_concave_outlet_deg,
                    terminal_base_marker=inlet_base_marker,
                )
                write_mesh_and_tags(mesh, facet_tags, "bioreactor.xdmf", "mesh_tags.xdmf")
                return

            if same_tissue_for_all_wells:
                terminal_centers = np.vstack([inlet_terminal_pts_by_well[0], outlet_terminal_pts_by_well[0]])
                terminal_normals = np.vstack([inlet_terminal_normals_by_well[0], outlet_terminal_normals_by_well[0]])
                terminal_areas = np.concatenate([inlet_terminal_areas_by_well[0], outlet_terminal_areas_by_well[0]])
                terminal_markers = np.concatenate(
                    [
                        inlet_base_marker + np.arange(len(inlet_terminal_pts_by_well[0]), dtype=np.int32),
                        outlet_base_marker + np.arange(len(outlet_terminal_pts_by_well[0]), dtype=np.int32),
                    ]
                )
                mesh, facet_tags = read_3d_mesh_from_stl_with_embedded_terminals(
                    stl_file=tissue_stl,
                    msh_file="_tmp_bioreactor_embedded_base.msh",
                    terminal_centers=terminal_centers,
                    terminal_normals=terminal_normals,
                    terminal_areas=terminal_areas,
                    terminal_markers=terminal_markers,
                    wall_marker=wall_marker,
                    char_len_min=gmsh_char_len_min,
                    char_len_max=gmsh_char_len_max,
                    disk_npts=gmsh_disk_npts,
                    terminal_refine_radius_factor=terminal_refine_radius_factor,
                    terminal_refine_h_factor=terminal_refine_h_factor,
                )
                facet_tags = overlay_concave_markers(
                    mesh,
                    facet_tags,
                    coords_inlet=coords_inlet_by_well[0],
                    coords_outlet=coords_outlet_by_well[0],
                    arterial_concave_marker=arterial_concave_marker,
                    venous_concave_marker=venous_concave_marker,
                    theta_concave_deg=theta_concave_deg,
                    theta_concave_inlet_deg=theta_concave_inlet_deg,
                    theta_concave_outlet_deg=theta_concave_outlet_deg,
                    terminal_base_marker=inlet_base_marker,
                )
                X0 = mesh.geometry.x.copy()
                for i in range(1, n_wells + 1):
                    dy = well_spacing_y * (i - 1)
                    mesh.geometry.x[:] = X0 + np.array([0.0, dy, 0.0], dtype=X0.dtype)
                    write_mesh_and_tags(mesh, facet_tags, f"bioreactor_well{i}.xdmf", f"mesh_tags_well{i}.xdmf")
                mesh.geometry.x[:] = X0
                return
            raise ValueError("For STL workflow, same_tissue_for_all_wells=False is not yet supported.")

        if not multiwell:
            mesh = read_3d_mesh_from_vtu(tissue_vtu, xdmf_name="_tmp_bioreactor.xdmf")
            facet_tags = build_facet_tags_for_mesh(
                mesh,
                inlet_terminal_pts=inlet_terminal_pts_by_well[0],
                outlet_terminal_pts=outlet_terminal_pts_by_well[0],
                inlet_terminal_normals=inlet_terminal_normals_by_well[0],
                outlet_terminal_normals=outlet_terminal_normals_by_well[0],
                coords_inlet=coords_inlet_by_well[0],
                coords_outlet=coords_outlet_by_well[0],
                wall_marker=wall_marker,
                arterial_concave_marker=arterial_concave_marker,
                venous_concave_marker=venous_concave_marker,
                inlet_base_marker=inlet_base_marker,
                outlet_base_marker=outlet_base_marker,
                theta_concave_deg=theta_concave_deg,
                theta_concave_inlet_deg=theta_concave_inlet_deg,
                theta_concave_outlet_deg=theta_concave_outlet_deg,
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
                inlet_terminal_normals=inlet_terminal_normals_by_well[0],
                outlet_terminal_normals=outlet_terminal_normals_by_well[0],
                coords_inlet=coords_inlet_by_well[0],
                coords_outlet=coords_outlet_by_well[0],
                wall_marker=wall_marker,
                arterial_concave_marker=arterial_concave_marker,
                venous_concave_marker=venous_concave_marker,
                inlet_base_marker=inlet_base_marker,
                outlet_base_marker=outlet_base_marker,
                theta_concave_deg=theta_concave_deg,
                theta_concave_inlet_deg=theta_concave_inlet_deg,
                theta_concave_outlet_deg=theta_concave_outlet_deg,
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
                        inlet_terminal_normals=inlet_terminal_normals_by_well[i - 1],
                        outlet_terminal_normals=outlet_terminal_normals_by_well[i - 1],
                        coords_inlet=coords_inlet_by_well[i - 1],
                        coords_outlet=coords_outlet_by_well[i - 1],
                        wall_marker=wall_marker,
                        arterial_concave_marker=arterial_concave_marker,
                        venous_concave_marker=venous_concave_marker,
                        inlet_base_marker=inlet_base_marker,
                        outlet_base_marker=outlet_base_marker,
                        theta_concave_deg=theta_concave_deg,
                        theta_concave_inlet_deg=theta_concave_inlet_deg,
                        theta_concave_outlet_deg=theta_concave_outlet_deg,
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
        tissue_stl= "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/files/stl/organoid-growth-domains/original/organoid-1.stl",
        tissue_vtu = None,
        output_1d_inlet="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/030426/Run1_10branches/0D_Input_Files/inlet",
        output_1d_outlet="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/030426/Run1_10branches/0D_Input_Files/outlet",
        branching_data_inlet="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/030426/Run1_10branches/branchingData_0.csv",
        branching_data_outlet="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/030426/Run1_10branches/branchingData_1.csv",
        coords_inlet=[-.18, 0.9, .55],
        coords_outlet=[.18, 0.9, .55],
        ftet_max_threads=24,
        multiwell=True,
        same_tissue_for_all_wells=True,
        well_spacing_y=0.6,
        inlet_base_marker=100,
        outlet_base_marker=200,
    )
