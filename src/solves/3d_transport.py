import argparse
import json
import os
from pathlib import Path
import re
import hashlib
import glob
import xml.etree.ElementTree as ET
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import adios4dolfinx
import dolfinx as dfx
import numpy as np
import ufl
import vtk
from basix.ufl import element
from dolfinx import fem, io
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, create_vector
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree
from vtk.util.numpy_support import vtk_to_numpy


################################################################################
# Mesh + checkpoint import
################################################################################


def _mesh_cache_root() -> Path:
    for key in ("SLURM_TMPDIR", "TMPDIR", "TEMP", "TMP"):
        val = os.environ.get(key, "").strip()
        if val:
            return Path(val).expanduser().resolve()
    return Path("/tmp") / os.environ.get("USER", "unknown")


def import_3d_mesh_with_facets(mesh_file: str, facet_file: str):
    """
    Read 3D perfusion mesh and facet meshtags.

    mesh_file: XDMF with the volume mesh (e.g. bioreactor.xdmf).
    facet_file: XDMF with boundary facet tags (e.g. mesh_tags_well1.xdmf),
                with marker 1 = inlet surface, 2 = outlet surface.
    """
    comm = MPI.COMM_WORLD

    def _read_xdmf(local_comm):
        with io.XDMFFile(local_comm, mesh_file, "r") as xdmf:
            try:
                # mesh.py writes the volume as Grid by default.
                m_loc = xdmf.read_mesh(name="Grid")
            except RuntimeError:
                # Fallback for files written with a different default grid name.
                m_loc = xdmf.read_mesh()

        with io.XDMFFile(local_comm, facet_file, "r") as xdmf:
            m_loc.topology.create_connectivity(m_loc.topology.dim, m_loc.topology.dim - 1)
            m_loc.topology.create_connectivity(m_loc.topology.dim - 1, m_loc.topology.dim)
            m_loc.topology.create_connectivity(m_loc.topology.dim - 1, 0)
            facet_tags_loc = xdmf.read_meshtags(m_loc, name="mesh_tags")
        return m_loc, facet_tags_loc

    use_bp_cache = os.environ.get("TRANSPORT_USE_BP_MESH_CACHE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if comm.size == 1 or not use_bp_cache:
        return _read_xdmf(comm)

    # Optional fallback for clusters where direct parallel XDMF/HDF5 reads are unstable.
    mesh_path = Path(mesh_file).expanduser().resolve()
    facet_path = Path(facet_file).expanduser().resolve()
    key = f"{mesh_path}|{facet_path}|{mesh_path.stat().st_mtime_ns}|{facet_path.stat().st_mtime_ns}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    cache_dir = _mesh_cache_root() / "darcy_mesh_cache"
    if comm.rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
    comm.barrier()
    cache_bp = cache_dir / f"mesh_facets_cache_{digest}.bp"

    if comm.rank == 0 and not cache_bp.exists():
        print(f"[mesh] Building BP cache for 3D mesh/facets: {cache_bp}", flush=True)
        m0, tags0 = _read_xdmf(MPI.COMM_SELF)
        adios4dolfinx.write_mesh(filename=cache_bp, mesh=m0)
        adios4dolfinx.write_meshtags(filename=cache_bp, mesh=m0, meshtags=tags0, meshtag_name="mesh_tags")
    comm.barrier()

    if comm.rank == 0:
        print(f"[mesh] Reading 3D mesh from BP cache: {cache_bp}", flush=True)
    m = adios4dolfinx.read_mesh(filename=cache_bp, comm=comm)
    if comm.rank == 0:
        print("[mesh] BP mesh read complete", flush=True)

    # Read facet tags from XDMF on the distributed mesh to avoid potential
    # multi-rank ADIOS meshtag read hangs.
    with io.XDMFFile(comm, facet_file, "r") as xdmf:
        m.topology.create_connectivity(m.topology.dim, m.topology.dim - 1)
        m.topology.create_connectivity(m.topology.dim - 1, m.topology.dim)
        m.topology.create_connectivity(m.topology.dim - 1, 0)
        facet_tags = xdmf.read_meshtags(m, name="mesh_tags")
    if comm.rank == 0:
        print("[mesh] XDMF facet tags read complete", flush=True)
    return m, facet_tags


def import_mesh(bp_file: str):
    mesh = adios4dolfinx.read_mesh(filename=Path(bp_file), comm=MPI.COMM_WORLD)
    tags = adios4dolfinx.read_meshtags(
        filename=Path(bp_file), mesh=mesh, meshtag_name="mesh_tags"
    )
    return mesh, tags


def import_flow_data(mesh_obj, bp_file: str):
    ts = adios4dolfinx.read_timestamps(
        bp_file, comm=MPI.COMM_WORLD, function_name="f"
    )
    V = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    q_src = fem.Function(V)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, q_src, name="f", time=t)
        q_src.x.scatter_forward()
        out.append(q_src.copy())
    return out


def _remove_xdmf_with_sidecar(xdmf_path: str | Path) -> None:
    """
    Remove an XDMF file and its conventional HDF5 sidecar before writing.

    DOLFINx writes time-series XDMF metadata and HDF5 datasets as a pair. If a
    previous run is interrupted or rerun with different time steps, stale HDF5
    datasets can remain while the XDMF points at a different set of paths. Some
    ParaView builds crash instead of reporting those missing datasets.
    """
    path = Path(xdmf_path)
    for candidate in (path, path.with_suffix(".h5")):
        try:
            if candidate.exists():
                candidate.unlink()
        except IsADirectoryError:
            pass


def _vtk_output_path(path: str | Path) -> Path:
    """Use a ParaView-friendly PVD path next to a requested output path."""
    p = Path(path)
    return p if p.suffix.lower() == ".pvd" else p.with_suffix(".pvd")


def _final_xdmf_output_path(path: str | Path) -> Path:
    """Avoid confusing a final-only XDMF with the live time-series path."""
    p = Path(path)
    stem = p.stem
    if stem.endswith("_final"):
        return p.with_suffix(".xdmf")
    return p.with_name(f"{stem}_final.xdmf")


def _remove_vtk_series(pvd_path: str | Path) -> None:
    """
    Remove a PVD file and the VTU files usually created beside it.

    DOLFINx/VTK writes a PVD index plus one VTU file per saved timestep. Stale
    VTU files are less dangerous than stale XDMF/HDF5 data, but removing them
    keeps reruns deterministic and avoids ParaView reading old snapshots.
    """
    path = Path(pvd_path)
    candidates = [path]
    candidates.extend(path.parent.glob(f"{path.stem}*.vtu"))
    candidates.extend(path.parent.glob(f"{path.stem}*.pvtu"))
    candidates.extend(path.parent.glob(f"{path.stem}_*.vtu"))
    candidates.extend(path.parent.glob(f"{path.stem}_*.pvtu"))
    for candidate in candidates:
        try:
            if candidate.exists():
                candidate.unlink()
        except IsADirectoryError:
            pass


################################################################################
# Transport solver
################################################################################


class TransportSolver:
    # Magliaro et al., Scientific Reports 2019, converted to cgs.
    # Paper units:
    #   D_org = 1.07e-9 m^2/s, D_M = 1.0e-9 m^2/s, D_water = 3.0e-9 m^2/s
    #   C_in = 0.2 mol/m^3, C_crit = 0.04 mol/m^3, k_m = 0.201 mol/m^3
    #   rho = 2.52e14 cells/m^3, OCR = 2.75e-17 mol/(cell s)
    # cgs conversions used here:
    #   m^2/s -> cm^2/s: multiply by 1e4
    #   mol/m^3 -> mol/cm^3: divide by 1e6
    #   cells/m^3 -> cells/cm^3: divide by 1e6
    PAPER_D_ORG_CM2_PER_S = 1.07e-5
    PAPER_D_GEL_CM2_PER_S = 1.0e-5
    PAPER_D_WATER_CM2_PER_S = 3.0e-5
    PAPER_C_IN_MOL_PER_CM3 = 2.0e-7
    PAPER_C_CRIT_MOL_PER_CM3 = 4.0e-8
    PAPER_KM_MOL_PER_CM3 = 2.01e-7
    PAPER_RHO_CELLS_PER_CM3 = 2.52e8
    PAPER_SINGLE_CELL_OCR_MOL_PER_CELL_S = 2.75e-17
    PAPER_VMAX_MOL_PER_CM3_S = (
        PAPER_RHO_CELLS_PER_CM3 * PAPER_SINGLE_CELL_OCR_MOL_PER_CELL_S
    )

    def __init__(
        self,
        bioreactor_domain,
        facet_file,
        mesh_inlet_file,
        mesh_outlet_file,
        flow_inlet_file,
        flow_outlet_file,
        vel_file,
        interface_bc_file="",
        concave_exchange_mode="velocity",
        skip_1d=False,
        T=10.0,
        dt=1.0,
        D_value=None,
        c_in_value=None,
        out_file="transport_c.xdmf",
        diffusion_region_path="",
        D_organoid=None,
        D_background=None,
        D_water=None,
        enable_mm_consumption=True,
        mm_km=None,
        mm_vmax=None,
        critical_oxygen_concentration=None,
        critical_output_file="",
        critical_summary_file="",
        output_format="vtk",
    ):
        # 3D tissue mesh + facet tags from the same geometry pipeline as Darcy.
        self.mesh, self.facet_tags = import_3d_mesh_with_facets(
            bioreactor_domain, facet_file
        )

        # Darcy velocity field projected to nodal P1 values.
        self.velocity, self.velocity_times, self._velocity_series = self._load_velocity(vel_file)

        self.skip_1d = bool(skip_1d)
        self.inlet_mesh = None
        self.inlet_tags = None
        self.outlet_mesh = None
        self.outlet_tags = None
        self.q_inlet_fun = None
        self.q_outlet_fun = None

        self.arterial_concave_marker = 31
        self.venous_concave_marker = 32
        self.inlet_base_marker = 1000
        self.outlet_base_marker = 2000
        self.concave_exchange_mode = str(concave_exchange_mode).strip().lower()
        if self.concave_exchange_mode not in {"velocity", "uniform"}:
            raise ValueError(
                "concave_exchange_mode must be 'velocity' or 'uniform', "
                f"got {concave_exchange_mode!r}"
            )
        self.concave_exchange = []
        self.inlet_exchange = []
        self.outlet_exchange = []
        self.x_inlet = np.zeros((0, 3), dtype=float)
        self.x_outlet = np.zeros((0, 3), dtype=float)
        self.q_inlet_vals = np.zeros((0,), dtype=float)
        self.q_outlet_vals = np.zeros((0,), dtype=float)
        self._velocity_point_coords = None
        self._velocity_local_point_ids = None

        if not self.skip_1d:
            # 1D inlet/outlet terminal meshes + checkpointed terminal flows.
            self.inlet_mesh, self.inlet_tags = import_mesh(mesh_inlet_file)
            self.outlet_mesh, self.outlet_tags = import_mesh(mesh_outlet_file)

            inlet_hist = import_flow_data(self.inlet_mesh, flow_inlet_file)
            outlet_hist = import_flow_data(self.outlet_mesh, flow_outlet_file)
            if not inlet_hist:
                raise RuntimeError(f"No inlet flow checkpoint data found in {flow_inlet_file}")
            if not outlet_hist:
                raise RuntimeError(f"No outlet flow checkpoint data found in {flow_outlet_file}")
            self.q_inlet_fun = inlet_hist[-1]
            self.q_outlet_fun = outlet_hist[-1]

            self._extract_terminal_data()
            self._build_terminal_exchange_data()

        if self.concave_exchange_mode == "uniform":
            self._build_concave_exchange_data(interface_bc_file)
        else:
            self._report_concave_velocity_fluxes()

        self.T = float(T)
        self.dt = float(dt)
        self.D_value = float(
            self.PAPER_D_ORG_CM2_PER_S if D_value is None else D_value
        )
        self.c_in_value = float(
            self.PAPER_C_IN_MOL_PER_CM3 if c_in_value is None else c_in_value
        )
        self.out_file = str(out_file)
        self.interface_bc_file = str(interface_bc_file)
        self.diffusion_region_path = str(diffusion_region_path).strip()
        self.D_organoid = float(
            self.D_value if D_organoid is None and D_value is not None else
            self.PAPER_D_ORG_CM2_PER_S if D_organoid is None else D_organoid
        )
        self.D_background = float(
            self.PAPER_D_GEL_CM2_PER_S if D_background is None else D_background
        )
        self.D_water = float(
            self.PAPER_D_WATER_CM2_PER_S if D_water is None else D_water
        )
        self.enable_mm_consumption = bool(enable_mm_consumption)
        self.mm_km = float(self.PAPER_KM_MOL_PER_CM3 if mm_km is None else mm_km)
        self.mm_vmax = float(
            self.PAPER_VMAX_MOL_PER_CM3_S if mm_vmax is None else mm_vmax
        )
        self.critical_oxygen_concentration = float(
            self.PAPER_C_CRIT_MOL_PER_CM3
            if critical_oxygen_concentration is None
            else critical_oxygen_concentration
        )
        self.critical_output_file = str(critical_output_file).strip()
        self.critical_summary_file = str(critical_summary_file).strip()
        self.output_format = str(output_format or "vtk").strip().lower()
        if self.output_format not in {"vtk", "xdmf", "both"}:
            raise ValueError(
                "--output-format must be one of 'vtk', 'xdmf', or 'both'; "
                f"got {self.output_format!r}."
            )

    ###########################################################################
    # Organoid mask + cgs oxygen constants
    ###########################################################################
    @staticmethod
    def _assign_dg0_cell_values(fun, cell_values):
        V0 = fun.function_space
        mesh = V0.mesh
        tdim = mesh.topology.dim
        nloc = mesh.topology.index_map(tdim).size_local
        values = np.asarray(cell_values, dtype=fun.x.array.dtype).reshape(-1)
        if values.size != nloc:
            raise ValueError(
                f"Expected one DG0 value for each owned cell ({nloc}); got {values.size}."
            )
        for c, value in enumerate(values):
            dofs = V0.dofmap.cell_dofs(c)
            if len(dofs) != 1:
                raise ValueError(
                    "Expected scalar DG0 space to have exactly one dof per cell; "
                    f"cell {c} has {len(dofs)}."
                )
            fun.x.array[int(dofs[0])] = value
        fun.x.scatter_forward()

    @staticmethod
    def _cell_geometry_points(mesh_obj, cell: int, gdim: int | None = None):
        if gdim is None:
            gdim = mesh_obj.geometry.dim
        gdofs = mesh_obj.geometry.dofmap[int(cell)]
        return mesh_obj.geometry.x[np.asarray(gdofs, dtype=np.int32), : int(gdim)]

    @staticmethod
    def _resolve_region_stls(path_str: str):
        raw = str(path_str).strip()
        if not raw:
            return []
        path = Path(raw).expanduser()
        if path.is_dir():
            return sorted(path.glob("*.stl"))
        if path.is_file():
            return [path]
        return [Path(p) for p in sorted(glob.glob(str(path))) if Path(p).suffix.lower() == ".stl"]

    @staticmethod
    def _build_stl_tester(stl_file: Path, tol: float = 1e-9):
        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(stl_file))
        reader.Update()
        surface = reader.GetOutput()
        if surface is None or surface.GetNumberOfPoints() == 0:
            raise ValueError(f"Failed to read STL surface from {stl_file}")

        enclosed = vtk.vtkSelectEnclosedPoints()
        enclosed.SetTolerance(float(tol))
        enclosed.Initialize(surface)
        distance = vtk.vtkImplicitPolyDataDistance()
        distance.SetInput(surface)

        def inside(point: np.ndarray) -> bool:
            p = np.asarray(point, dtype=float).reshape(3,)
            return bool(enclosed.IsInsideSurface(float(p[0]), float(p[1]), float(p[2])))

        def signed_distance(point: np.ndarray) -> float:
            p = np.asarray(point, dtype=float).reshape(3,)
            return float(distance.EvaluateFunction((float(p[0]), float(p[1]), float(p[2]))))

        def cleanup() -> None:
            try:
                enclosed.Complete()
            except Exception:
                pass

        return inside, signed_distance, cleanup

    @staticmethod
    def _smooth_transition_alpha(signed_distance: float, width: float) -> float:
        if width <= 0.0:
            return 1.0 if signed_distance <= 0.0 else 0.0
        if signed_distance <= -width:
            return 1.0
        if signed_distance >= width:
            return 0.0
        t = (signed_distance + width) / (2.0 * width)
        smooth = 3.0 * t * t - 2.0 * t * t * t
        return float(1.0 - smooth)

    def _build_organoid_indicator_and_diffusion(self, W):
        mesh = self.mesh
        chi = fem.Function(W)
        D_field = fem.Function(W)

        stl_files = self._resolve_region_stls(self.diffusion_region_path)
        nloc = mesh.topology.index_map(mesh.topology.dim).size_local
        alpha_local = np.zeros(nloc, dtype=np.float64)

        if not stl_files:
            # Without an STL mask, treat the whole domain as organoid tissue so
            # the consumption model is active and the legacy uniform-D behavior
            # remains close to the old solver.
            alpha_local[:] = 1.0
            mode = "uniform_organoid_no_stl"
        else:
            mode = "stl_organoid_mask"
            testers = []
            try:
                testers = [self._build_stl_tester(Path(p)) for p in stl_files]
                for cell in range(nloc):
                    pts = self._cell_geometry_points(mesh, int(cell), mesh.geometry.dim)
                    midpoint = np.mean(pts, axis=0)
                    signed_dist = np.inf
                    for inside_fn, dist_fn, _cleanup in testers:
                        d_mid = dist_fn(midpoint)
                        if inside_fn(midpoint):
                            signed_dist = min(signed_dist, d_mid)
                            continue
                        d_vertices = [dist_fn(p) for p in pts]
                        vertex_inside = [inside_fn(p) for p in pts]
                        if any(vertex_inside):
                            signed_dist = min(signed_dist, min(d_vertices))
                        else:
                            signed_dist = min(
                                signed_dist,
                                min(abs(d_mid), *(abs(dv) for dv in d_vertices)),
                            )
                    alpha_local[cell] = self._smooth_transition_alpha(
                        float(signed_dist),
                        0.0,
                    )
            finally:
                for _inside, _dist, cleanup in testers:
                    cleanup()

        D_local = (
            float(self.D_background)
            + alpha_local * (float(self.D_organoid) - float(self.D_background))
        )
        self._assign_dg0_cell_values(chi, alpha_local)
        self._assign_dg0_cell_values(D_field, D_local)

        local_org_vol = fem.assemble_scalar(fem.form(chi * ufl.dx(mesh)))
        org_vol = mesh.comm.allreduce(float(local_org_vol), op=MPI.SUM)
        if mesh.comm.rank == 0:
            print(
                "[transport] oxygen constants in cgs: "
                f"D_org={self.D_organoid:.6e} cm^2/s, "
                f"D_background={self.D_background:.6e} cm^2/s, "
                f"D_water={self.D_water:.6e} cm^2/s, "
                f"C_in={self.c_in_value:.6e} mol/cm^3, "
                f"Ccrit={self.critical_oxygen_concentration:.6e} mol/cm^3, "
                f"km={self.mm_km:.6e} mol/cm^3, "
                f"Vmax={self.mm_vmax:.6e} mol/(cm^3 s)",
                flush=True,
            )
            print(
                f"[transport] oxygen region mode={mode}, stl_count={len(stl_files)}, "
                f"organoid_volume={org_vol:.6e} cm^3",
                flush=True,
            )

        self.transport_metadata = {
            "units": "cgs",
            "concentration_units": "mol/cm^3",
            "diffusion_units": "cm^2/s",
            "time_units": "s",
            "region_mode": mode,
            "diffusion_region_path": str(self.diffusion_region_path),
            "resolved_stl_files": [str(p) for p in stl_files],
            "D_organoid_cm2_per_s": float(self.D_organoid),
            "D_background_cm2_per_s": float(self.D_background),
            "D_water_cm2_per_s": float(self.D_water),
            "c_in_mol_per_cm3": float(self.c_in_value),
            "critical_oxygen_mol_per_cm3": float(self.critical_oxygen_concentration),
            "michaelis_menten_km_mol_per_cm3": float(self.mm_km),
            "michaelis_menten_vmax_mol_per_cm3_s": float(self.mm_vmax),
            "michaelis_menten_enabled": bool(self.enable_mm_consumption),
            "organoid_volume_cm3": float(org_vol),
        }
        return chi, D_field

    ###########################################################################
    # Velocity import
    ###########################################################################
    def _read_vtu_velocity(self, vtu_file: str):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(vtu_file))
        reader.Update()

        grid = reader.GetOutput()
        velocity = grid.GetPointData().GetArray("f")
        if velocity is None and grid.GetPointData().GetNumberOfArrays() > 0:
            velocity = grid.GetPointData().GetArray(0)
        if velocity is None:
            raise RuntimeError(
                f"Could not find point-data array 'f' in velocity file: {vtu_file}"
            )

        velocity_np = vtk_to_numpy(velocity)
        V = fem.functionspace(
            self.mesh,
            element(
                "Lagrange",
                self.mesh.basix_cell(),
                1,
                shape=(self.mesh.geometry.dim,),
            ),
        )
        u = fem.Function(V)
        flat = np.asarray(velocity_np, dtype=np.float64).reshape(-1)
        if u.x.array.size != flat.size:
            raise ValueError(
                "Velocity field size does not match the transport mesh. "
                f"Expected {u.x.array.size} values, found {flat.size}."
            )
        u.x.array[:] = flat
        u.x.scatter_forward()
        return u

    def _read_vtu_velocity_point_data(self, vtu_file: str):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(vtu_file))
        reader.Update()

        grid = reader.GetOutput()
        points = grid.GetPoints()
        if points is None:
            raise RuntimeError(f"Could not read mesh points from velocity file: {vtu_file}")
        velocity = grid.GetPointData().GetArray("f")
        if velocity is None and grid.GetPointData().GetNumberOfArrays() > 0:
            velocity = grid.GetPointData().GetArray(0)
        if velocity is None:
            raise RuntimeError(
                f"Could not find point-data array 'f' in velocity file: {vtu_file}"
            )
        coords = np.asarray(vtk_to_numpy(points.GetData()), dtype=np.float64)
        values = np.asarray(vtk_to_numpy(velocity), dtype=np.float64)
        return coords, values

    def _velocity_function_from_flat(self, flat_values: np.ndarray):
        V = fem.functionspace(
            self.mesh,
            element(
                "Lagrange",
                self.mesh.basix_cell(),
                1,
                shape=(self.mesh.geometry.dim,),
            ),
        )
        u = fem.Function(V)
        flat = np.asarray(flat_values, dtype=np.float64).reshape(-1)
        if u.x.array.size != flat.size:
            raise ValueError(
                "Velocity field size does not match the transport mesh. "
                f"Expected {u.x.array.size} values, found {flat.size}."
            )
        u.x.array[:] = flat
        u.x.scatter_forward()
        return u

    def _velocity_function_from_point_data(self, coords: np.ndarray, values: np.ndarray):
        V = fem.functionspace(
            self.mesh,
            element(
                "Lagrange",
                self.mesh.basix_cell(),
                1,
                shape=(self.mesh.geometry.dim,),
            ),
        )
        u = fem.Function(V)

        coords = np.asarray(coords, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if coords.ndim != 2 or values.ndim != 2:
            raise ValueError("Velocity point data must be 2D arrays")
        if coords.shape[0] != values.shape[0]:
            raise ValueError(
                "Velocity coordinate/value arrays are inconsistent: "
                f"{coords.shape[0]} coords vs {values.shape[0]} values."
            )
        if values.shape[1] != self.mesh.geometry.dim:
            raise ValueError(
                "Velocity value dimension does not match the transport mesh. "
                f"Expected {self.mesh.geometry.dim}, found {values.shape[1]}."
            )

        local_coords = np.asarray(self.mesh.geometry.x, dtype=np.float64)
        if local_coords.shape[0] * self.mesh.geometry.dim == u.x.array.size and local_coords.shape[0] == values.shape[0]:
            # Serial fast path when the VTU point ordering already matches the mesh ordering.
            self._velocity_point_coords = coords
            self._velocity_local_point_ids = None
            u.x.array[:] = values.reshape(-1)
            u.x.scatter_forward()
            return u

        tree = cKDTree(coords)
        _, nn = tree.query(local_coords)
        self._velocity_point_coords = coords
        self._velocity_local_point_ids = np.asarray(nn, dtype=int)
        mapped = values[np.asarray(nn, dtype=int)]
        u.x.array[:] = mapped.reshape(-1)
        u.x.scatter_forward()
        return u

    def _map_velocity_point_values_to_local_flat(self, coords: np.ndarray, values: np.ndarray) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        if coords.ndim != 2 or values.ndim != 2:
            raise ValueError("Velocity point data must be 2D arrays")
        if coords.shape[0] != values.shape[0]:
            raise ValueError(
                "Velocity coordinate/value arrays are inconsistent: "
                f"{coords.shape[0]} coords vs {values.shape[0]} values."
            )
        if values.shape[1] != self.mesh.geometry.dim:
            raise ValueError(
                "Velocity value dimension does not match the transport mesh. "
                f"Expected {self.mesh.geometry.dim}, found {values.shape[1]}."
            )

        V = fem.functionspace(
            self.mesh,
            element(
                "Lagrange",
                self.mesh.basix_cell(),
                1,
                shape=(self.mesh.geometry.dim,),
            ),
        )
        expected_size = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
        local_coords = np.asarray(self.mesh.geometry.x, dtype=np.float64)

        if local_coords.shape[0] * self.mesh.geometry.dim == expected_size and local_coords.shape[0] == values.shape[0]:
            self._velocity_point_coords = coords
            self._velocity_local_point_ids = None
            return values.reshape(-1).copy()

        tree = cKDTree(coords)
        _, nn = tree.query(local_coords)
        self._velocity_point_coords = coords
        self._velocity_local_point_ids = np.asarray(nn, dtype=int)
        mapped = values[self._velocity_local_point_ids]
        return mapped.reshape(-1).copy()

    def _xdmf_reader(self):
        for name in ("vtkXdmf3ReaderT", "vtkXdmf3Reader", "vtkXdmfReader"):
            cls = getattr(vtk, name, None)
            if cls is not None:
                return cls()
        raise RuntimeError(
            "VTK in this environment does not provide an XDMF reader "
            "(tried vtkXdmf3ReaderT, vtkXdmf3Reader, vtkXdmfReader)."
        )

    def _convert_xdmf_velocity_to_vtu(self, xdmf_file: str) -> Path:
        xdmf_path = Path(xdmf_file).expanduser().resolve()
        if not xdmf_path.exists():
            raise FileNotFoundError(f"Missing XDMF velocity file: {xdmf_path}")

        out_dir = xdmf_path.parent
        cached_vtu = out_dir / "u_p0_000000.vtu"
        data_sidecar = xdmf_path.with_suffix(".h5")
        newest_src_mtime = max(
            p.stat().st_mtime
            for p in (xdmf_path, data_sidecar)
            if p.exists()
        )
        if cached_vtu.exists() and cached_vtu.stat().st_mtime >= newest_src_mtime:
            if self.mesh.comm.rank == 0:
                print(f"[transport] Reusing cached velocity VTU: {cached_vtu}", flush=True)
            return cached_vtu

        if self.mesh.comm.rank == 0:
            print(f"[transport] Converting XDMF velocity to VTU: {xdmf_path} -> {cached_vtu}", flush=True)
            reader = self._xdmf_reader()
            reader.SetFileName(str(xdmf_path))
            reader.UpdateInformation()

            n_steps = 0
            if hasattr(reader, "GetNumberOfTimeSteps"):
                try:
                    n_steps = int(reader.GetNumberOfTimeSteps())
                except Exception:
                    n_steps = 0
            if n_steps > 0 and hasattr(reader, "SetTimeStep"):
                reader.SetTimeStep(n_steps - 1)
            elif n_steps > 0 and hasattr(reader, "UpdateTimeStep"):
                try:
                    time_steps = [reader.GetTimeStep(i) for i in range(n_steps)]
                    reader.UpdateTimeStep(float(time_steps[-1]))
                except Exception:
                    pass

            reader.Update()
            grid = reader.GetOutput()
            if grid is None:
                raise RuntimeError(f"Failed to read XDMF velocity file: {xdmf_path}")

            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(str(cached_vtu))
            writer.SetInputData(grid)
            ok = writer.Write()
            if ok != 1:
                raise RuntimeError(f"Failed to write converted VTU velocity file: {cached_vtu}")

            pvd_path = out_dir / "u_p0_000000.pvd"
            if pvd_path.exists():
                try:
                    pvd_path.unlink()
                except Exception:
                    pass
        self.mesh.comm.barrier()
        return cached_vtu

    def _extract_xdmf_velocity_times(self, xdmf_file: str) -> list[float]:
        root = ET.parse(xdmf_file).getroot()
        grids = root.findall(".//Grid[@GridType='Collection']/Grid")
        if not grids:
            grids = root.findall(".//Domain/Grid[@GridType='Collection']/Grid")
        times: list[float] = []
        for grid in grids:
            time_node = grid.find("Time")
            if time_node is None:
                continue
            raw = time_node.attrib.get("Value", None)
            if raw is None:
                continue
            try:
                times.append(float(raw))
            except Exception:
                continue
        return times

    def _convert_xdmf_velocity_to_vtu_series(self, xdmf_file: str):
        xdmf_path = Path(xdmf_file).expanduser().resolve()
        data_sidecar = xdmf_path.with_suffix(".h5")
        out_dir = xdmf_path.parent
        time_values = self._extract_xdmf_velocity_times(str(xdmf_path))
        if not time_values:
            # Fall back to a single converted snapshot if there is no explicit temporal collection.
            single = self._convert_xdmf_velocity_to_vtu(str(xdmf_path))
            return np.asarray([0.0], dtype=np.float64), [single]

        newest_src_mtime = max(
            p.stat().st_mtime for p in (xdmf_path, data_sidecar) if p.exists()
        )
        vtu_paths = [out_dir / f"u_p0_{i:06d}.vtu" for i in range(len(time_values))]

        series_times = None
        series_paths = None
        if self.mesh.comm.rank == 0:
            up_to_date = all(p.exists() and p.stat().st_mtime >= newest_src_mtime for p in vtu_paths)
            if up_to_date:
                series_times = np.asarray(time_values, dtype=np.float64)
                series_paths = [str(p) for p in vtu_paths]
                print(f"[transport] Reusing cached velocity VTU series in: {out_dir}", flush=True)

        if self.mesh.comm.rank == 0 and series_times is None:
            reader = self._xdmf_reader()
            reader.SetFileName(str(xdmf_path))
            reader.UpdateInformation()
            print(f"[transport] Converting XDMF velocity series to VTU snapshots in: {out_dir}", flush=True)
            for i, tval in enumerate(time_values):
                if hasattr(reader, "SetTimeStep"):
                    reader.SetTimeStep(i)
                    reader.Update()
                elif hasattr(reader, "UpdateTimeStep"):
                    try:
                        reader.UpdateTimeStep(float(tval))
                    except Exception:
                        reader.Update()
                else:
                    reader.Update()

                grid = reader.GetOutput()
                if grid is None:
                    raise RuntimeError(f"Failed to read XDMF velocity file: {xdmf_path}")
                writer = vtk.vtkXMLUnstructuredGridWriter()
                writer.SetFileName(str(vtu_paths[i]))
                writer.SetInputData(grid)
                ok = writer.Write()
                if ok != 1:
                    raise RuntimeError(f"Failed to write converted VTU velocity file: {vtu_paths[i]}")

            series_times = np.asarray(time_values, dtype=np.float64)
            series_paths = [str(p) for p in vtu_paths]

        series_times = self.mesh.comm.bcast(series_times, root=0)
        series_paths = self.mesh.comm.bcast(series_paths, root=0)
        if series_times is None or series_paths is None:
            raise RuntimeError(f"Failed to convert/load XDMF velocity time series: {xdmf_path}")
        return np.asarray(series_times, dtype=np.float64), [Path(p) for p in series_paths]

    def _load_velocity(self, velocity_file: str):
        velocity_path = Path(velocity_file).expanduser().resolve()
        suffix = velocity_path.suffix.lower()
        if suffix == ".xdmf":
            times, vtu_paths = self._convert_xdmf_velocity_to_vtu_series(str(velocity_path))
            point_series = [self._read_vtu_velocity_point_data(str(p)) for p in vtu_paths]
            coords0 = np.asarray(point_series[0][0], dtype=np.float64)
            vals0 = np.asarray(point_series[0][1], dtype=np.float64)
            series_arrays = np.stack(
                [self._map_velocity_point_values_to_local_flat(coords0, np.asarray(vals, dtype=np.float64)) for _, vals in point_series],
                axis=0,
            )
            u = self._velocity_function_from_flat(series_arrays[-1])
            return u, times, series_arrays
        u = self._read_vtu_velocity(str(velocity_path))
        return u, np.asarray([0.0], dtype=np.float64), None

    def _velocity_at_time(self, t: float) -> np.ndarray | None:
        if self._velocity_series is None or len(self.velocity_times) <= 1:
            return None
        times = np.asarray(self.velocity_times, dtype=float)
        vals = np.asarray(self._velocity_series, dtype=np.float64)
        if vals.shape[0] != len(times):
            raise ValueError("Velocity time-series lengths are inconsistent")

        period = float(times[-1])
        if period > 0.0:
            t_eval = float(t) % period
            if np.isclose(t_eval, 0.0) and float(t) > 0.0:
                t_eval = period
        else:
            t_eval = float(t)

        if t_eval <= times[0]:
            return vals[0].copy()
        if t_eval >= times[-1]:
            return vals[-1].copy()

        j = int(np.searchsorted(times, t_eval, side="right") - 1)
        j = max(0, min(j, len(times) - 2))
        t0, t1 = float(times[j]), float(times[j + 1])
        if t1 == t0:
            return vals[j].copy()
        alpha = (t_eval - t0) / (t1 - t0)
        return (1.0 - alpha) * vals[j] + alpha * vals[j + 1]

    def _update_velocity_for_time(self, t: float) -> bool:
        flat = self._velocity_at_time(t)
        if flat is None:
            return False
        arr = np.asarray(flat, dtype=np.float64)
        arr = arr.reshape(-1)
        if arr.size != self.velocity.x.array.size:
            raise ValueError(
                "Mapped velocity field size does not match the local transport vector. "
                f"Expected {self.velocity.x.array.size} values, found {arr.size}."
            )
        self.velocity.x.array[:] = arr
        self.velocity.x.scatter_forward()
        return True

    def _inlet_concentration(self, t: float) -> float:
        """
        Time-dependent inlet concentration.

        The oxygen supply is kept on for the full simulated duration. This
        replaces the old finite pulse behavior used for passive tracer tests.
        """
        return self.c_in_value

    ###########################################################################
    # Terminal data from 1D checkpoints
    ###########################################################################
    def _extract_terminal_data(self):
        """
        Extract terminal coordinates from the 1D inlet/outlet meshes (marker 2)
        and sample the checkpointed terminal flow values there.
        """

        def extract_terminal_coords(mesh_obj, tags):
            marker = 2
            idx = np.where(tags.values == marker)[0]
            if len(idx) == 0:
                return np.zeros((0, mesh_obj.geometry.dim))
            entity_ids = tags.indices[idx]
            return mesh_obj.geometry.x[entity_ids]

        def sample_field_at_points(func, pts):
            if len(pts) == 0:
                return np.zeros(0)
            dof_coords = func.function_space.tabulate_dof_coordinates()
            tree = cKDTree(dof_coords)
            _, nn = tree.query(pts)
            return np.asarray(func.x.array[np.asarray(nn, dtype=int)], dtype=float)

        self.x_inlet = extract_terminal_coords(self.inlet_mesh, self.inlet_tags)
        self.x_outlet = extract_terminal_coords(self.outlet_mesh, self.outlet_tags)
        self.q_inlet_vals = sample_field_at_points(self.q_inlet_fun, self.x_inlet)
        self.q_outlet_vals = sample_field_at_points(self.q_outlet_fun, self.x_outlet)

        if self.mesh.comm.rank == 0:
            print(f"Found {len(self.x_inlet)} inlet terminals")
            print(f"Found {len(self.x_outlet)} outlet terminals")

    ###########################################################################
    # Darcy-style marker utilities
    ###########################################################################
    def _get_terminal_marker_sets(self):
        vals = np.unique(self.facet_tags.values)
        inlet_marks = np.sort(
            vals[(vals >= self.inlet_base_marker) & (vals < self.outlet_base_marker)]
        )
        outlet_marks = np.sort(
            vals[
                (vals >= self.outlet_base_marker)
                & (vals < self.outlet_base_marker + 1000)
            ]
        )
        return inlet_marks.astype(int), outlet_marks.astype(int)

    def _marker_centroids(self, markers):
        mesh = self.mesh
        tdim = mesh.topology.dim
        fdim = tdim - 1

        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(fdim, 0)
        f_to_v = mesh.topology.connectivity(fdim, 0)
        X = mesh.geometry.x

        centroids = np.zeros((len(markers), 3), dtype=np.float64)
        for i, marker in enumerate(markers):
            facets = self.facet_tags.find(int(marker))
            if facets.size == 0:
                centroids[i, :] = np.nan
                continue
            c = np.zeros((facets.size, 3), dtype=np.float64)
            for j, facet in enumerate(facets):
                verts = f_to_v.links(int(facet))
                c[j, :] = X[verts].mean(axis=0)
            centroids[i, :] = c.mean(axis=0)
        return centroids

    def _assign_markers_to_terminals(self, marker_centroids, terminal_coords):
        if marker_centroids.size == 0 or terminal_coords.size == 0:
            return np.zeros((0,), dtype=int)
        tree = cKDTree(terminal_coords)
        _, idx = tree.query(marker_centroids)
        return idx.astype(int)

    def _get_terminal_marker_data(self):
        inlet_marks, outlet_marks = self._get_terminal_marker_sets()
        inlet_c = self._marker_centroids(inlet_marks)
        outlet_c = self._marker_centroids(outlet_marks)
        idx_in = self._assign_markers_to_terminals(inlet_c, self.x_inlet)
        idx_out = self._assign_markers_to_terminals(outlet_c, self.x_outlet)
        return inlet_marks, outlet_marks, inlet_c, outlet_c, idx_in, idx_out

    def _split_marker_facets(self, marker: int):
        mesh = self.mesh
        tdim = mesh.topology.dim
        fdim = tdim - 1

        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(fdim, tdim)
        f_to_c = mesh.topology.connectivity(fdim, tdim)

        facets = self.facet_tags.find(int(marker))
        if facets.size == 0:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

        exterior = []
        interior = []
        for facet in facets:
            ncells = len(f_to_c.links(int(facet)))
            if ncells <= 1:
                exterior.append(int(facet))
            else:
                interior.append(int(facet))
        return np.asarray(exterior, dtype=np.int32), np.asarray(interior, dtype=np.int32)

    def _global_scalar(self, form_expr):
        local = float(fem.assemble_scalar(fem.form(form_expr)))
        return float(self.mesh.comm.allreduce(local, op=MPI.SUM))

    ###########################################################################
    # Facet exchange model
    ###########################################################################
    def _build_terminal_exchange_data(self):
        """
        Build per-marker surface flux densities g = |Q| / A for the Darcy terminal
        interfaces. Inlet markers inject c_in(t); outlet markers remove local c.

        Using surface fluxes on the tagged terminal facets is more realistic than
        Dirac sources/sinks because the exchange is distributed over the actual
        3D interface area used by the Darcy solve.
        """
        mesh = self.mesh
        domain = mesh.ufl_domain()
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=self.facet_tags)
        dS = ufl.Measure("dS", domain=domain, subdomain_data=self.facet_tags)

        inlet_marks, outlet_marks, _, _, idx_in, idx_out = self._get_terminal_marker_data()

        def build_data(markers, idx_map, flows, role):
            out = []
            for j, marker in enumerate(markers):
                if j >= len(idx_map):
                    continue

                ext_facets, int_facets = self._split_marker_facets(int(marker))
                area_ext = (
                    self._global_scalar(one * ds(int(marker))) if ext_facets.size > 0 else 0.0
                )
                area_int = (
                    self._global_scalar(one * dS(int(marker))) if int_facets.size > 0 else 0.0
                )
                area_total = area_ext + area_int

                q_raw = float(flows[idx_map[j]]) if len(flows) else 0.0
                q_mag = abs(q_raw)
                g = q_mag / area_total if area_total > 0.0 else 0.0

                out.append(
                    {
                        "marker": int(marker),
                        "role": role,
                        "flow_raw": q_raw,
                        "flow_mag": q_mag,
                        "area_exterior": area_ext,
                        "area_interior": area_int,
                        "area_total": area_total,
                        "flux_density": g,
                    }
                )
            return out

        self.inlet_exchange = build_data(inlet_marks, idx_in, self.q_inlet_vals, "inlet")
        self.outlet_exchange = build_data(
            outlet_marks, idx_out, self.q_outlet_vals, "outlet"
        )

        if mesh.comm.rank == 0:
            print("Configured inlet terminal exchange:")
            for data in self.inlet_exchange:
                print(
                    f"  marker {data['marker']}: |Q|={data['flow_mag']:.6e}, "
                    f"A={data['area_total']:.6e}, g={data['flux_density']:.6e}"
                )
            print("Configured outlet terminal exchange:")
            for data in self.outlet_exchange:
                print(
                    f"  marker {data['marker']}: |Q|={data['flow_mag']:.6e}, "
                    f"A={data['area_total']:.6e}, g={data['flux_density']:.6e}"
                )

    def _report_concave_velocity_fluxes(self):
        """
        Report the heterogeneous Darcy wall-normal flux implied by the velocity
        field on the concave markers.

        Positive u·n is outflow from the tissue; negative u·n is inflow.
        """
        mesh = self.mesh
        domain = mesh.ufl_domain()
        ds = ufl.Measure("ds", domain=domain, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(domain)
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        qn = ufl.dot(self.velocity, n)
        q_in = 0.5 * (-qn + abs(qn))
        q_out = 0.5 * (qn + abs(qn))

        entries = [
            ("arterial_concave", int(self.arterial_concave_marker)),
            ("venous_concave", int(self.venous_concave_marker)),
        ]

        if mesh.comm.rank == 0:
            print("Configured concave-wall exchange from local Darcy velocity:")
        for label, marker in entries:
            area = self._global_scalar(one * ds(marker))
            q_in_tot = self._global_scalar(q_in * ds(marker))
            q_out_tot = self._global_scalar(q_out * ds(marker))
            if mesh.comm.rank == 0:
                print(
                    f"  {label} (marker {marker}): "
                    f"A={area:.6e}, Qin={q_in_tot:.6e}, Qout={q_out_tot:.6e}"
                )

    def _build_concave_exchange_data(self, interface_bc_file):
        """
        Load concave-wall inflow/outflow totals from Darcy's interface_bc.json
        and convert them to uniform surface flux densities on markers 31 and 32.

        Darcy stores outward-normal fluid fluxes:
          - negative => inflow into the tissue
          - positive => outflow from the tissue
        """
        self.concave_exchange = []
        if not interface_bc_file:
            return

        comm = self.mesh.comm
        iface_path = Path(interface_bc_file)
        if comm.rank == 0:
            if not iface_path.exists():
                raise FileNotFoundError(f"Missing interface BC file: {iface_path}")
            with open(iface_path, "r") as fp:
                iface_data = json.load(fp)
        else:
            iface_data = None
        iface_data = comm.bcast(iface_data, root=0)

        mesh = self.mesh
        domain = mesh.ufl_domain()
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=self.facet_tags)

        entries = [
            (
                "arterial_concave",
                int(self.arterial_concave_marker),
                float(iface_data.get("q_artery_leak", 0.0)),
            ),
            (
                "venous_concave",
                int(self.venous_concave_marker),
                float(iface_data.get("q_venous_leak", 0.0)),
            ),
        ]

        for label, marker, q_raw in entries:
            area = self._global_scalar(one * ds(marker))
            flux_density = abs(q_raw) / area if area > 0.0 else 0.0
            if q_raw < 0.0:
                mode = "source"
            elif q_raw > 0.0:
                mode = "sink"
            else:
                mode = "none"
            self.concave_exchange.append(
                {
                    "label": label,
                    "marker": marker,
                    "flow_raw": q_raw,
                    "flow_mag": abs(q_raw),
                    "area_total": area,
                    "flux_density": flux_density,
                    "mode": mode,
                }
            )

        if mesh.comm.rank == 0:
            print(f"Loaded concave-wall exchange from {iface_path}")
            for data in self.concave_exchange:
                print(
                    f"  {data['label']} (marker {data['marker']}): "
                    f"Q={data['flow_raw']:.6e}, A={data['area_total']:.6e}, "
                    f"g={data['flux_density']:.6e}, mode={data['mode']}"
                )

    def _build_terminal_exchange_forms(self, c, w, c_in, ds_tagged, dS_tagged):
        """
        Assemble facet exchange terms.

        Exterior terminal markers use ds(marker). Interior embedded terminal
        markers use dS(marker). For DG concentration spaces, the interior terms
        are written symmetrically using facet averages so the source/sink is not
        biased to one arbitrary side of the embedded surface.
        """
        mesh = self.mesh

        a_exchange = 0
        L_exchange = 0

        for data in self.inlet_exchange:
            marker = int(data["marker"])
            g = fem.Constant(mesh, dfx.default_scalar_type(data["flux_density"]))

            if data["area_exterior"] > 0.0:
                L_exchange += g * c_in * w * ds_tagged(marker)
            if data["area_interior"] > 0.0:
                L_exchange += g * c_in * ufl.avg(w) * dS_tagged(marker)

        for data in self.outlet_exchange:
            marker = int(data["marker"])
            g = fem.Constant(mesh, dfx.default_scalar_type(data["flux_density"]))

            if data["area_exterior"] > 0.0:
                a_exchange += g * c * w * ds_tagged(marker)
            if data["area_interior"] > 0.0:
                a_exchange += g * ufl.avg(c * w) * dS_tagged(marker)

        return a_exchange, L_exchange

    def _build_concave_exchange_forms(self, c, w, c_in, ds_tagged, n):
        """
        Add concave-wall exchange.

        In velocity mode, the local Darcy wall-normal flux u·n is used directly:
          - u·n < 0: inflow into tissue, inject c_in(t)
          - u·n > 0: outflow from tissue, remove local concentration c

        In uniform mode, Darcy's interface_bc.json leak totals are smeared
        uniformly over each concave marker.
        """
        a_exchange = 0
        L_exchange = 0

        if self.concave_exchange_mode == "velocity":
            qn = ufl.dot(self.velocity, n)
            q_in = 0.5 * (-qn + abs(qn))
            q_out = 0.5 * (qn + abs(qn))
            for marker in (
                int(self.arterial_concave_marker),
                int(self.venous_concave_marker),
            ):
                L_exchange += q_in * c_in * w * ds_tagged(marker)
                a_exchange += q_out * c * w * ds_tagged(marker)
            return a_exchange, L_exchange

        mesh = self.mesh
        for data in self.concave_exchange:
            marker = int(data["marker"])
            g = fem.Constant(mesh, dfx.default_scalar_type(data["flux_density"]))
            if data["mode"] == "source":
                L_exchange += g * c_in * w * ds_tagged(marker)
            elif data["mode"] == "sink":
                a_exchange += g * c * w * ds_tagged(marker)

        return a_exchange, L_exchange

    def _compute_exchange_rates(self, c_h, c_in, ds_tagged, dS_tagged):
        """
        Return species exchange rates implied by the terminal source/sink model.

        Rates are reported in the same units as concentration times volumetric
        flow, i.e. "amount per unit time".
        """
        mesh = self.mesh

        inlet_rates = []
        for data in self.inlet_exchange:
            marker = int(data["marker"])
            g = fem.Constant(mesh, dfx.default_scalar_type(data["flux_density"]))
            rate = 0.0
            if data["area_exterior"] > 0.0:
                rate += self._global_scalar(g * c_in * ds_tagged(marker))
            if data["area_interior"] > 0.0:
                rate += self._global_scalar(g * c_in * dS_tagged(marker))
            inlet_rates.append(rate)

        outlet_rates = []
        for data in self.outlet_exchange:
            marker = int(data["marker"])
            g = fem.Constant(mesh, dfx.default_scalar_type(data["flux_density"]))
            rate = 0.0
            if data["area_exterior"] > 0.0:
                rate += self._global_scalar(g * c_h * ds_tagged(marker))
            if data["area_interior"] > 0.0:
                rate += self._global_scalar(g * ufl.avg(c_h) * dS_tagged(marker))
            outlet_rates.append(rate)

        return (
            np.asarray(inlet_rates, dtype=float),
            np.asarray(outlet_rates, dtype=float),
        )

    def _compute_concave_exchange_rates(self, c_h, c_in, ds_tagged, n):
        inflow_rates = []
        outflow_rates = []

        if self.concave_exchange_mode == "velocity":
            qn = ufl.dot(self.velocity, n)
            q_in = 0.5 * (-qn + abs(qn))
            q_out = 0.5 * (qn + abs(qn))
            for marker in (
                int(self.arterial_concave_marker),
                int(self.venous_concave_marker),
            ):
                inflow_rates.append(self._global_scalar(q_in * c_in * ds_tagged(marker)))
                outflow_rates.append(self._global_scalar(q_out * c_h * ds_tagged(marker)))
            return (
                np.asarray(inflow_rates, dtype=float),
                np.asarray(outflow_rates, dtype=float),
            )

        mesh = self.mesh
        for data in self.concave_exchange:
            marker = int(data["marker"])
            g = fem.Constant(mesh, dfx.default_scalar_type(data["flux_density"]))
            if data["mode"] == "source":
                rate = self._global_scalar(g * c_in * ds_tagged(marker))
                inflow_rates.append(rate)
            elif data["mode"] == "sink":
                rate = self._global_scalar(g * c_h * ds_tagged(marker))
                outflow_rates.append(rate)

        return (
            np.asarray(inflow_rates, dtype=float),
            np.asarray(outflow_rates, dtype=float),
        )

    def _update_mm_consumption_rate(self, rate_fun, c_old, organoid_indicator):
        """
        Semi-implicit Michaelis-Menten sink coefficient.

        The reaction is Vmax * chi * C / (km + C). To keep the transport solve
        linear, the denominator is lagged with the previous concentration:
        beta^n = Vmax * chi / (km + max(C^n, 0)), sink ~= beta^n C^{n+1}.
        """
        if not self.enable_mm_consumption:
            rate_fun.x.array[:] = 0.0
            rate_fun.x.scatter_forward()
            return

        c_prev = np.maximum(np.asarray(c_old.x.array, dtype=float), 0.0)
        chi = np.asarray(organoid_indicator.x.array, dtype=float)
        denom = float(self.mm_km) + c_prev
        denom = np.maximum(denom, np.finfo(float).tiny)
        rate_fun.x.array[:] = float(self.mm_vmax) * chi / denom
        rate_fun.x.scatter_forward()

    def _compute_consumption_rate(self, c_h, organoid_indicator, dx):
        if not self.enable_mm_consumption:
            return 0.0
        km = fem.Constant(self.mesh, dfx.default_scalar_type(self.mm_km))
        vmax = fem.Constant(self.mesh, dfx.default_scalar_type(self.mm_vmax))
        c_pos = ufl.max_value(c_h, 0.0)
        rate_expr = organoid_indicator * vmax * c_pos / (km + c_pos)
        return self._global_scalar(rate_expr * dx)

    def _critical_oxygen_stats(self, c_h, organoid_indicator, critical_mask, dx):
        crit = float(self.critical_oxygen_concentration)
        c_vals = np.asarray(c_h.x.array, dtype=float)
        below = (c_vals < crit).astype(np.float64)
        critical_mask.x.array[:] = below
        critical_mask.x.scatter_forward()

        one = fem.Constant(self.mesh, dfx.default_scalar_type(1.0))
        below_vol = self._global_scalar(critical_mask * dx)
        total_vol = self._global_scalar(one * dx)
        organoid_vol = self._global_scalar(organoid_indicator * dx)
        organoid_below_vol = self._global_scalar(critical_mask * organoid_indicator * dx)

        local_min = float(np.min(c_vals)) if c_vals.size else float("inf")
        local_max = float(np.max(c_vals)) if c_vals.size else float("-inf")
        c_min = float(self.mesh.comm.allreduce(local_min, op=MPI.MIN))
        c_max = float(self.mesh.comm.allreduce(local_max, op=MPI.MAX))
        if not np.isfinite(c_min):
            c_min = 0.0
        if not np.isfinite(c_max):
            c_max = 0.0

        return {
            "critical_oxygen_mol_per_cm3": crit,
            "min_concentration_mol_per_cm3": float(c_min),
            "max_concentration_mol_per_cm3": float(c_max),
            "below_critical_volume_cm3": float(below_vol),
            "domain_volume_cm3": float(total_vol),
            "below_critical_volume_fraction": float(below_vol / total_vol)
            if total_vol > 0.0
            else 0.0,
            "organoid_below_critical_volume_cm3": float(organoid_below_vol),
            "organoid_volume_cm3": float(organoid_vol),
            "organoid_below_critical_volume_fraction": float(organoid_below_vol / organoid_vol)
            if organoid_vol > 0.0
            else 0.0,
        }

    ###########################################################################
    # Solve advection-diffusion
    ###########################################################################
    def run(self):
        mesh = self.mesh
        domain = mesh.ufl_domain()

        DG = element("DG", mesh.basix_cell(), 0)
        P1 = element("Lagrange", mesh.basix_cell(), 1)
        W = fem.functionspace(mesh, DG)

        c_in = fem.Constant(mesh, dfx.default_scalar_type(self.c_in_value))
        delta_t = fem.Constant(mesh, dfx.default_scalar_type(self.dt))
        organoid_indicator, D = self._build_organoid_indicator_and_diffusion(W)
        mm_rate = fem.Function(W)

        c = ufl.TrialFunction(W)
        w = ufl.TestFunction(W)

        c_old = fem.Function(W)
        c_h = fem.Function(W)
        critical_mask = fem.Function(W)
        c_out = fem.Function(fem.functionspace(mesh, P1))
        critical_out = fem.Function(fem.functionspace(mesh, P1))
        c_old.x.array[:] = 0.0
        self._update_mm_consumption_rate(mm_rate, c_old, organoid_indicator)

        dx = ufl.Measure("dx", domain=domain)
        ds_tagged = ufl.Measure("ds", domain=domain, subdomain_data=self.facet_tags)
        dS_tagged = ufl.Measure("dS", domain=domain, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(domain)

        a_time = (c * w / delta_t) * dx
        a_advect = -ufl.dot(c * self.velocity, ufl.grad(w)) * dx
        a_diffuse = ufl.dot(D * ufl.grad(c), ufl.grad(w)) * dx
        a_consumption = mm_rate * c * w * dx
        a_exchange, L_exchange = self._build_terminal_exchange_forms(
            c, w, c_in, ds_tagged, dS_tagged
        )
        a_concave, L_concave = self._build_concave_exchange_forms(
            c, w, c_in, ds_tagged, n
        )

        a = a_time + a_advect + a_diffuse + a_consumption + a_exchange + a_concave
        L = (c_old / delta_t) * w * dx + L_exchange + L_concave

        # Upwind advection flux on all interior facets.
        un = (ufl.dot(self.velocity, n) + abs(ufl.dot(self.velocity, n))) / 2.0
        a += ufl.jump(w) * (un("+") * c("+") - un("-") * c("-")) * dS_tagged

        # SIPG diffusion for DG concentration.
        h = ufl.CellDiameter(domain)
        alpha = fem.Constant(mesh, dfx.default_scalar_type(10.0))
        a += D("+") * alpha("+") / h("+") * ufl.dot(
            ufl.jump(w, n), ufl.jump(c, n)
        ) * dS_tagged
        a -= D("+") * ufl.dot(ufl.avg(ufl.grad(w)), ufl.jump(c, n)) * dS_tagged
        a -= D("+") * ufl.dot(ufl.avg(ufl.grad(c)), ufl.jump(w, n)) * dS_tagged

        a_form = fem.form(a)
        L_form = fem.form(L)

        A = assemble_matrix(a_form)
        A.assemble()
        b = create_vector(L_form)

        solver = PETSc.KSP().create(mesh.comm)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")

        requested_out_path = Path(self.out_file)
        requested_critical_path = Path(
            self.critical_output_file
            if self.critical_output_file
            else str(requested_out_path.with_name(f"{requested_out_path.stem}_below_critical.xdmf"))
        )
        write_vtk = self.output_format in {"vtk", "both"}
        write_xdmf = self.output_format in {"xdmf", "both"}
        vtk_output_path = _vtk_output_path(requested_out_path)
        vtk_critical_output_path = _vtk_output_path(requested_critical_path)
        xdmf_final_output_path = _final_xdmf_output_path(requested_out_path)
        xdmf_critical_final_output_path = _final_xdmf_output_path(requested_critical_path)

        if mesh.comm.rank == 0:
            # If the requested path was previously used for a temporal XDMF
            # run, remove it even when the new default output is VTK. Otherwise
            # a stale crashing .xdmf/.h5 pair can remain next to the good .pvd.
            if requested_out_path.suffix.lower() == ".xdmf":
                _remove_xdmf_with_sidecar(requested_out_path)
            if requested_critical_path.suffix.lower() == ".xdmf":
                _remove_xdmf_with_sidecar(requested_critical_path)
            if write_vtk:
                _remove_vtk_series(vtk_output_path)
                _remove_vtk_series(vtk_critical_output_path)
            if write_xdmf:
                _remove_xdmf_with_sidecar(xdmf_final_output_path)
                _remove_xdmf_with_sidecar(xdmf_critical_final_output_path)
        mesh.comm.barrier()

        c_out.name = "oxygen_concentration"
        critical_out.name = "below_critical_oxygen"

        vtk_out = None
        vtk_critical_out = None
        if write_vtk:
            vtk_out = io.VTKFile(mesh.comm, str(vtk_output_path), "w")
            vtk_critical_out = io.VTKFile(mesh.comm, str(vtk_critical_output_path), "w")
            if mesh.comm.rank == 0:
                print(
                    f"[transport] Writing ParaView time series: {vtk_output_path} "
                    f"and {vtk_critical_output_path}",
                    flush=True,
                )
        if write_xdmf and mesh.comm.rank == 0:
            print(
                "[transport] XDMF output is written as final-only snapshots to avoid "
                "ParaView/HDF5 time-series crashes: "
                f"{xdmf_final_output_path} and {xdmf_critical_final_output_path}",
                flush=True,
            )

        t = 0.0
        nt = int(np.round(self.T / self.dt))
        injected_cumulative = 0.0
        removed_cumulative = 0.0
        consumed_cumulative = 0.0
        critical_rows = []

        for step in range(nt):
            t += self.dt
            c_in.value = dfx.default_scalar_type(self._inlet_concentration(t))
            self._update_mm_consumption_rate(mm_rate, c_old, organoid_indicator)

            if self._update_velocity_for_time(t):
                A = assemble_matrix(a_form)
                A.assemble()
                solver.setOperators(A)
            elif self.enable_mm_consumption:
                A = assemble_matrix(a_form)
                A.assemble()
                solver.setOperators(A)

            b.zeroEntries()
            assemble_vector(b, L_form)
            b.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )

            solver.solve(b, c_h.x.petsc_vec)
            c_h.x.scatter_forward()
            c_old.x.array[:] = c_h.x.array

            c_out.interpolate(c_h)
            critical_stats = self._critical_oxygen_stats(
                c_h,
                organoid_indicator,
                critical_mask,
                dx,
            )
            critical_out.interpolate(critical_mask)
            if vtk_out is not None:
                vtk_out.write_function(c_out, t)
            if vtk_critical_out is not None:
                vtk_critical_out.write_function(critical_out, t)

            inlet_rates, outlet_rates = self._compute_exchange_rates(
                c_h, c_in, ds_tagged, dS_tagged
            )
            concave_in_rates, concave_out_rates = self._compute_concave_exchange_rates(
                c_h, c_in, ds_tagged, n
            )
            consumption_rate = self._compute_consumption_rate(c_h, organoid_indicator, dx)
            inlet_total = float(np.sum(inlet_rates) + np.sum(concave_in_rates))
            outlet_total = float(np.sum(outlet_rates) + np.sum(concave_out_rates))
            terminal_in_total = float(np.sum(inlet_rates))
            terminal_out_total = float(np.sum(outlet_rates))
            concave_in_total = float(np.sum(concave_in_rates))
            concave_out_total = float(np.sum(concave_out_rates))
            injected_cumulative += inlet_total * self.dt
            removed_cumulative += outlet_total * self.dt
            consumed_cumulative += consumption_rate * self.dt
            row = {
                "step": int(step + 1),
                "time_s": float(t),
                "inlet_concentration_mol_per_cm3": float(c_in.value),
                "injection_rate_mol_per_s": float(inlet_total),
                "outflow_rate_mol_per_s": float(outlet_total),
                "consumption_rate_mol_per_s": float(consumption_rate),
                "injected_cumulative_mol": float(injected_cumulative),
                "removed_cumulative_mol": float(removed_cumulative),
                "consumed_cumulative_mol": float(consumed_cumulative),
                **critical_stats,
            }
            critical_rows.append(row)

            if mesh.comm.rank == 0:
                print(
                    f"Step {step + 1}/{nt}   t={t:.3f}   c_in={float(c_in.value):.6e}   "
                    f"inj_rate={inlet_total:.6e}   out_rate={outlet_total:.6e}   "
                    f"consume_rate={consumption_rate:.6e}   "
                    f"term_in={terminal_in_total:.6e}   term_out={terminal_out_total:.6e}   "
                    f"wall_in={concave_in_total:.6e}   wall_out={concave_out_total:.6e}   "
                    f"inj_cum={injected_cumulative:.6e}   out_cum={removed_cumulative:.6e}   "
                    f"cons_cum={consumed_cumulative:.6e}   "
                    f"org_below_crit={critical_stats['organoid_below_critical_volume_fraction']:.3e}"
                )

        if vtk_out is not None:
            vtk_out.close()
        if vtk_critical_out is not None:
            vtk_critical_out.close()

        if write_xdmf:
            # Write only the final state to XDMF/HDF5. Temporal XDMF from DOLFINx
            # can leave the XML index ahead of the HDF5 sidecar while a run is
            # active, and some ParaView builds crash on those incomplete pairs.
            xdmf_out = io.XDMFFile(mesh.comm, str(xdmf_final_output_path), "w")
            xdmf_out.write_mesh(mesh)
            xdmf_out.write_function(c_out, float(t))
            xdmf_out.close()

            xdmf_critical_out = io.XDMFFile(mesh.comm, str(xdmf_critical_final_output_path), "w")
            xdmf_critical_out.write_mesh(mesh)
            xdmf_critical_out.write_function(critical_out, float(t))
            xdmf_critical_out.close()

        summary_path = (
            Path(self.critical_summary_file)
            if self.critical_summary_file
            else Path(self.out_file).with_name(f"{Path(self.out_file).stem}_oxygen_summary.json")
        )
        if mesh.comm.rank == 0:
            payload = {
                "metadata": dict(getattr(self, "transport_metadata", {})),
                "final": critical_rows[-1] if critical_rows else {},
                "time_series": critical_rows,
                "output_format": self.output_format,
                "oxygen_pvd": str(vtk_output_path) if write_vtk else "",
                "critical_mask_pvd": str(vtk_critical_output_path) if write_vtk else "",
                "oxygen_final_xdmf": str(xdmf_final_output_path) if write_xdmf else "",
                "critical_mask_final_xdmf": str(xdmf_critical_final_output_path) if write_xdmf else "",
            }
            summary_path.write_text(json.dumps(payload, indent=2))


################################################################################
# CLI
################################################################################

def _resolve_organoid_path(path_str: str, organoid_id: int) -> str:
    """
    Resolve one path for a specific organoid.

    Supports either a format token such as `{organoid}` or an existing
    directory token like `organoid_3`.
    """
    s = str(path_str)
    if "{organoid}" in s:
        return s.format(organoid=int(organoid_id))
    return re.sub(r"organoid_\d+", f"organoid_{int(organoid_id)}", s)


def _resolve_output_path(path_str: str, organoid_id: int, batch_mode: bool) -> str:
    """
    Resolve one output path for a specific organoid.

    In batch mode, avoid output collisions when the user provides only a plain
    filename by appending `_organoid_<id>` before the suffix.
    """
    s = str(path_str)
    if not batch_mode:
        return s
    if "{organoid}" in s or re.search(r"organoid_\d+", s):
        return _resolve_organoid_path(s, organoid_id)
    p = Path(s)
    if p.suffix:
        return str(p.with_name(f"{p.stem}_organoid_{int(organoid_id)}{p.suffix}"))
    return f"{s}_organoid_{int(organoid_id)}"


def _resolve_region_path(path_str: str, organoid_id: int) -> str:
    """
    Resolve an organoid-specific STL path.

    Mirrors the Darcy permeability-region convention: a directory maps to
    organoid-<id>.stl when present, while explicit paths may use either
    "{organoid}", "X", or an existing organoid_<n> token.
    """
    raw = str(path_str).strip()
    if not raw:
        return ""
    if "{organoid}" in raw:
        return raw.format(organoid=int(organoid_id))
    if "X" in raw:
        return raw.replace("X", str(int(organoid_id)))
    p = Path(raw).expanduser()
    if p.is_dir():
        candidate = p / f"organoid-{int(organoid_id)}.stl"
        if candidate.exists():
            return str(candidate)
    return _resolve_organoid_path(raw, organoid_id)


def _build_solver_from_args(args, organoid_id: int | None, batch_mode: bool) -> TransportSolver:
    if organoid_id is None:
        return TransportSolver(
            bioreactor_domain=args.bioreactor_domain,
            facet_file=args.facet_file,
            mesh_inlet_file=args.mesh_inlet_file,
            mesh_outlet_file=args.mesh_outlet_file,
            flow_inlet_file=args.flow_inlet_file,
            flow_outlet_file=args.flow_outlet_file,
            vel_file=args.vel_file,
            interface_bc_file=args.interface_bc_file,
            concave_exchange_mode=args.concave_exchange_mode,
            skip_1d=args.skip_1d,
            T=args.T,
            dt=args.dt,
            D_value=args.D_value,
            c_in_value=args.c_in_value,
            out_file=args.out_file,
            diffusion_region_path=args.diffusion_region_path,
            D_organoid=args.D_organoid,
            D_background=args.D_background,
            D_water=args.D_water,
            enable_mm_consumption=not args.no_mm_consumption,
            mm_km=args.mm_km,
            mm_vmax=args.mm_vmax,
            critical_oxygen_concentration=args.critical_oxygen_concentration,
            critical_output_file=args.critical_output_file,
            critical_summary_file=args.critical_summary_file,
            output_format=args.output_format,
        )

    return TransportSolver(
        bioreactor_domain=_resolve_organoid_path(args.bioreactor_domain, organoid_id),
        facet_file=_resolve_organoid_path(args.facet_file, organoid_id),
        mesh_inlet_file=_resolve_organoid_path(args.mesh_inlet_file, organoid_id),
        mesh_outlet_file=_resolve_organoid_path(args.mesh_outlet_file, organoid_id),
        flow_inlet_file=_resolve_organoid_path(args.flow_inlet_file, organoid_id),
        flow_outlet_file=_resolve_organoid_path(args.flow_outlet_file, organoid_id),
        vel_file=_resolve_organoid_path(args.vel_file, organoid_id),
        interface_bc_file=_resolve_organoid_path(args.interface_bc_file, organoid_id),
        concave_exchange_mode=args.concave_exchange_mode,
        skip_1d=args.skip_1d,
        T=args.T,
        dt=args.dt,
        D_value=args.D_value,
        c_in_value=args.c_in_value,
        out_file=_resolve_output_path(args.out_file, organoid_id, batch_mode),
        diffusion_region_path=_resolve_region_path(args.diffusion_region_path, organoid_id)
        if args.diffusion_region_path
        else "",
        D_organoid=args.D_organoid,
        D_background=args.D_background,
        D_water=args.D_water,
        enable_mm_consumption=not args.no_mm_consumption,
        mm_km=args.mm_km,
        mm_vmax=args.mm_vmax,
        critical_oxygen_concentration=args.critical_oxygen_concentration,
        critical_output_file=_resolve_output_path(args.critical_output_file, organoid_id, batch_mode)
        if args.critical_output_file
        else "",
        critical_summary_file=_resolve_output_path(args.critical_summary_file, organoid_id, batch_mode)
        if args.critical_summary_file
        else "",
        output_format=args.output_format,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="3D advection-diffusion in the Darcy tissue domain with terminal facet exchange."
    )
    ap.add_argument(
        "--bioreactor-domain",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/coupling-output/run_20/organoid_3/geometry/bioreactor.xdmf",
        help="3D tissue volume mesh XDMF.",
    )
    ap.add_argument(
        "--facet-file",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/coupling-output/run_20/organoid_3/geometry/mesh_tags.xdmf",
        help="Facet tag XDMF written by src/geometry/mesh.py.",
    )
    ap.add_argument(
        "--mesh-inlet-file",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/coupling-output/run_20/organoid_3/geometry/tagged_branches_inlet.bp",
        help="1D inlet terminal mesh BP file.",
    )
    ap.add_argument(
        "--mesh-outlet-file",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/coupling-output/run_20/organoid_3/geometry/tagged_branches_outlet.bp",
        help="1D outlet terminal mesh BP file.",
    )
    ap.add_argument(
        "--flow-inlet-file",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/coupling-output/run_20/organoid_3/geometry/flow_checkpoint_inlet.bp",
        help="1D inlet flow checkpoint BP file.",
    )
    ap.add_argument(
        "--flow-outlet-file",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/coupling-output/run_20/organoid_3/geometry/flow_checkpoint_outlet.bp",
        help="1D outlet flow checkpoint BP file.",
    )
    ap.add_argument(
        "--vel-file",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/coupling-output/run_20/organoid_3/out_darcy/u_p0_000000.vtu",
        help="Darcy velocity file. Accepts legacy VTU or batched transient u.xdmf; XDMF is converted and cached as u_p0_000000.vtu.",
    )
    ap.add_argument(
        "--interface-bc-file",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/coupling-output/run_20/organoid_3/out_darcy/interface_bc.json",
        help="Darcy interface_bc.json containing q_artery_leak and q_venous_leak. Used only in uniform concave mode.",
    )
    ap.add_argument(
        "--concave-exchange-mode",
        choices=["velocity", "uniform"],
        default="velocity",
        help="Use local Darcy u·n on the concave walls ('velocity') or uniform flux densities from interface_bc.json ('uniform').",
    )
    ap.add_argument(
        "--skip-1d",
        action="store_true",
        help="Disable synthetic terminal exchange and run transport using only concave-wall exchange.",
    )
    ap.add_argument("--T", type=float, default=10000.0)
    ap.add_argument("--dt", type=float, default=100.0)
    ap.add_argument(
        "--D-value",
        type=float,
        default=None,
        help=(
            "Legacy uniform diffusion value in cm^2/s. The oxygen model now "
            "uses --D-organoid/--D-background instead; omit for paper cgs defaults."
        ),
    )
    ap.add_argument(
        "--c-in-value",
        type=float,
        default=None,
        help="Inlet oxygen concentration in mol/cm^3. Paper value: 2.0e-7.",
    )
    ap.add_argument(
        "--diffusion-region-path",
        default="",
        help=(
            "STL file, directory, glob, or path with an organoid_<n> token. "
            "The STL marks the consuming organoid region."
        ),
    )
    ap.add_argument(
        "--D-organoid",
        type=float,
        default=None,
        help="Oxygen diffusion in organoid tissue in cm^2/s. Paper value: 1.07e-5.",
    )
    ap.add_argument(
        "--D-background",
        type=float,
        default=None,
        help="Oxygen diffusion outside the organoid/STL mask in cm^2/s. Paper Matrigel value: 1.0e-5.",
    )
    ap.add_argument(
        "--D-water",
        type=float,
        default=None,
        help="Oxygen diffusion in water/medium in cm^2/s, recorded in metadata. Paper value: 3.0e-5.",
    )
    ap.add_argument(
        "--no-mm-consumption",
        action="store_true",
        help="Disable Michaelis-Menten oxygen consumption.",
    )
    ap.add_argument(
        "--mm-km",
        type=float,
        default=None,
        help="Michaelis-Menten k_m in mol/cm^3. Paper value: 2.01e-7.",
    )
    ap.add_argument(
        "--mm-vmax",
        type=float,
        default=None,
        help=(
            "Volumetric maximal oxygen consumption in mol/(cm^3 s). "
            "Paper-derived value: 2.52e8 cells/cm^3 * 2.75e-17 mol/(cell s) = 6.93e-9."
        ),
    )
    ap.add_argument(
        "--critical-oxygen-concentration",
        type=float,
        default=None,
        help="Critical oxygen concentration in mol/cm^3. Paper value: 4.0e-8.",
    )
    ap.add_argument(
        "--critical-output-file",
        default="",
        help="Optional XDMF output path for the below-critical oxygen mask.",
    )
    ap.add_argument(
        "--critical-summary-file",
        default="",
        help="Optional JSON path for critical-oxygen volume/time-series summary.",
    )
    ap.add_argument(
        "--output-format",
        choices=["vtk", "xdmf", "both"],
        default="vtk",
        help=(
            "Visualization format. 'vtk' writes ParaView-safe .pvd/.vtu time "
            "series. 'xdmf' writes final-only XDMF snapshots. 'both' writes both."
        ),
    )
    ap.add_argument("--out-file", default="transport_c_no_vasc.xdmf")
    ap.add_argument(
        "--organoid-ids",
        type=int,
        nargs="*",
        default=[],
        help=(
            "Optional batch organoid IDs, e.g. --organoid-ids 1 2 3 4. "
            "Input paths are rewritten by replacing either '{organoid}' or any "
            "existing 'organoid_<n>' token."
        ),
    )
    args = ap.parse_args()

    organoid_ids = [int(i) for i in args.organoid_ids] if args.organoid_ids else [None]
    batch_mode = len(organoid_ids) > 1

    for organoid_id in organoid_ids:
        if organoid_id is not None and MPI.COMM_WORLD.rank == 0:
            print(
                "\n"
                + "=" * 80
                + f"\nRunning transport for organoid_{int(organoid_id)}\n"
                + "=" * 80,
                flush=True,
            )
        solver = _build_solver_from_args(args, organoid_id, batch_mode)
        solver.run()
