import argparse
import csv
import hashlib
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import ufl
import numpy   as np
import dolfinx as dfx
import json
import pandas  as pd

from ufl               import dot, grad
from mpi4py            import MPI
from pathlib           import Path
from petsc4py          import PETSc
from basix.ufl         import element
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from scipy.spatial import cKDTree
import adios4dolfinx
from dolfinx.io import VTKFile

current_dir = Path("/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/solves")


def _make_linear_problem(a, L, bcs=None, petsc_options=None, petsc_options_prefix=None):
    kwargs = {}
    if bcs is not None:
        kwargs["bcs"] = bcs
    if petsc_options is not None:
        kwargs["petsc_options"] = petsc_options
    if petsc_options_prefix is not None:
        kwargs["petsc_options_prefix"] = petsc_options_prefix
    try:
        return LinearProblem(a, L, **kwargs)
    except TypeError as exc:
        if "petsc_options_prefix" not in str(exc):
            raise
        kwargs.pop("petsc_options_prefix", None)
        return LinearProblem(a, L, **kwargs)


def _mesh_cache_root() -> Path:
    for key in ("SLURM_TMPDIR", "TMPDIR", "TEMP", "TMP"):
        val = os.environ.get(key, "").strip()
        if val:
            return Path(val).expanduser().resolve()
    return Path("/tmp") / os.environ.get("USER", "unknown")

# -----------------------------------------------------
# Mesh importer
# -----------------------------------------------------
def import_3d_mesh(xdmf_file: str):
    with io.XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        m = xdmf.read_mesh(name="Grid")
        m.topology.create_connectivity(m.topology.dim, m.topology.dim-1)
        tags = xdmf.read_meshtags(m, name="mesh_tags")
    return m, tags

def import_mesh(bp_file: str, comm=MPI.COMM_WORLD):
    mesh = adios4dolfinx.read_mesh(filename=Path(bp_file), comm=comm)
    tags = adios4dolfinx.read_meshtags(filename=Path(bp_file), mesh=mesh, meshtag_name="mesh_tags")
    return mesh, tags

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

    if comm.size == 1:
        return _read_xdmf(comm)

    # For MPI runs, avoid direct parallel XDMF/HDF5 reads by caching to ADIOS BP.
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

    # Mesh and facet tags must come from the same distributed representation.
    # Reading the BP mesh and then remapping XDMF meshtags can silently drop
    # terminal markers on some MPI partitions.
    try:
        m.topology.create_connectivity(m.topology.dim, m.topology.dim - 1)
        m.topology.create_connectivity(m.topology.dim - 1, m.topology.dim)
        m.topology.create_connectivity(m.topology.dim - 1, 0)
        facet_tags = adios4dolfinx.read_meshtags(
            filename=cache_bp,
            mesh=m,
            meshtag_name="mesh_tags",
        )
        if comm.rank == 0:
            print("[mesh] BP facet tags read complete", flush=True)
    except Exception as exc:
        if comm.rank == 0:
            print(
                "[mesh] BP facet tag read failed; falling back to XDMF facet tags:",
                repr(exc),
                flush=True,
            )
        with io.XDMFFile(comm, facet_file, "r") as xdmf:
            m.topology.create_connectivity(m.topology.dim, m.topology.dim - 1)
            m.topology.create_connectivity(m.topology.dim - 1, m.topology.dim)
            m.topology.create_connectivity(m.topology.dim - 1, 0)
            facet_tags = xdmf.read_meshtags(m, name="mesh_tags")
        if comm.rank == 0:
            print("[mesh] XDMF facet tags read complete", flush=True)
    return m, facet_tags
# -----------------------------------------------------
# Pressure ADIOS importer
# -----------------------------------------------------
def import_pressure_data(mesh_obj, bp_file: str, comm=MPI.COMM_WORLD):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=comm,
                                       function_name="f")
    print("Timestamps found for pressure data:", ts)
    P1 = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    p_src = fem.Function(P1)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, p_src, name="f", time=t)
        p_src.x.scatter_forward()
        p_src.x.array[:] *= 1333.22  # Convert to dyne/cm²
        out.append(p_src.copy())
    return out   # list of Functions

# -----------------------------------------------------
# Flow ADIOS importer (scalar P1)
# -----------------------------------------------------
def import_flow_data(mesh_obj, bp_file: str, comm=MPI.COMM_WORLD):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=comm,
                                       function_name="f")
    print("Timestamps found for flow data:", ts)
    P1 = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    q_src = fem.Function(P1)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, q_src,
                                    name="f", time=t)
        q_src.x.scatter_forward()
        out.append(q_src.copy())
    return out   # list of Functions

def import_area_data(mesh_obj, bp_file: str, comm=MPI.COMM_WORLD):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=comm,
                                       function_name="f")
    print("Timestamps found for area data:", ts)
    P1 = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    a_src = fem.Function(P1)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, a_src,
                                    name="f", time=t)
        a_src.x.scatter_forward()
        out.append(a_src.copy())
    return out   # list of Functions


def select_checkpoint_entry(history, checkpoint_time_index: int, label: str):
    if not history:
        raise ValueError(f"No checkpoint entries found for {label}")

    n_hist = len(history)
    idx = int(checkpoint_time_index)
    if idx < 0:
        idx += n_hist
    if idx < 0 or idx >= n_hist:
        raise IndexError(
            f"checkpoint_time_index={checkpoint_time_index} is out of range for {label} "
            f"(available entries: {n_hist})"
        )
    return history[idx]


# ===================================================================
#                PERFUSION SOLVER WITH FULL COUPLING
# ===================================================================
from scipy.spatial import cKDTree

class PerfusionSolver:
    def __init__(self, bioreactor_domain, facet_file,
                 mesh_inlet_file, mesh_outlet_file,
                 pres_inlet_file, pres_outlet_file,
                 flow_inlet_file, flow_outlet_file,
                 inlet_coord, outlet_coord,
                 area_inlet_file=None, area_outlet_file=None,
                 concave_bc_mode: str = "dirichlet",
                 concave_pressure_profile: str = "constant",
                 concave_pressure_profile_axis: str = "y",
                 arterial_pressure_profile_low=None,
                 arterial_pressure_profile_high=None,
                 venous_pressure_profile_low=None,
                 venous_pressure_profile_high=None,
                 lp_arterial: float = 0.0,
                 lp_venous: float = 0.0,
                 skip_1d: bool = False,
                 fallback_inlet_pressure: float = 0.0,
                 fallback_outlet_pressure: float = 0.0,
                 checkpoint_time_index: int = -1,
                 inlet_flux_correction: bool = False,
                 inlet_flux_corr_max_iter: int = 5,
                 inlet_flux_corr_relax: float = 0.5,
                 inlet_flux_corr_tol: float = 0.05,
                 branching_in_file=None, branching_out_file=None,
                 inlet_output_csv=None, outlet_output_csv=None,
                 concave_inlet_pressure=None,
                 concave_outlet_pressure=None):

        # 3D perfusion domain + boundary facets
        if MPI.COMM_WORLD.rank == 0:
            print("About to open XDMF mesh", flush=True)
        MPI.COMM_WORLD.barrier()
        self.mesh, self.facet_tags = import_3d_mesh_with_facets(
            bioreactor_domain, facet_file
        )
        if self.mesh.comm.rank == 0:
            print("[mesh] 3D mesh/facet import complete", flush=True)

        # Marker conventions from mesh_tags_well*.xdmf
        self.arterial_concave_marker = 31
        self.venous_concave_marker = 32
        self.inlet_base_marker = 1000
        self.outlet_base_marker = 2000
        self.skip_1d = bool(skip_1d)
        self.fallback_inlet_pressure = float(fallback_inlet_pressure)
        self.fallback_outlet_pressure = float(fallback_outlet_pressure)
        self.checkpoint_time_index = int(checkpoint_time_index)

        # 1D tree terminal territory meshes/checkpoints.
        # Read only on rank 0 (COMM_SELF), then broadcast sampled arrays with
        # typed MPI collectives (avoids object-pickle bcast hangs on some setups).
        comm = self.mesh.comm
        if comm.rank == 0:
            if self.skip_1d:
                print("[1d] Rank 0 skipping inlet/outlet 1D trees; using fallback concave pressures", flush=True)
                x_inlet = np.zeros((0, 3), dtype=np.float64)
                x_outlet = np.zeros((0, 3), dtype=np.float64)
                p_inlet = np.zeros((0,), dtype=np.float64)
                q_inlet = np.zeros((0,), dtype=np.float64)
                a_inlet = np.zeros((0,), dtype=np.float64)
                p_outlet = np.zeros((0,), dtype=np.float64)
                q_outlet = np.zeros((0,), dtype=np.float64)
                a_outlet = np.zeros((0,), dtype=np.float64)
                p_in_BC = float(self.fallback_inlet_pressure)
                p_out_BC = float(self.fallback_outlet_pressure)
                n_in = 0
                n_out = 0
            else:
                print("[1d] Rank 0 reading inlet/outlet 1D meshes", flush=True)
                self.inlet_mesh, self.inlet_tags = import_mesh(mesh_inlet_file, comm=MPI.COMM_SELF)
                self.outlet_mesh, self.outlet_tags = import_mesh(mesh_outlet_file, comm=MPI.COMM_SELF)

                print("[1d] Rank 0 reading pressure checkpoints", flush=True)
                p_in_hist = import_pressure_data(self.inlet_mesh, pres_inlet_file, comm=MPI.COMM_SELF)
                p_out_hist = import_pressure_data(self.outlet_mesh, pres_outlet_file, comm=MPI.COMM_SELF)
                self.p_A_k = select_checkpoint_entry(
                    p_in_hist, self.checkpoint_time_index, "inlet pressure checkpoints"
                )
                self.p_V_k = select_checkpoint_entry(
                    p_out_hist, self.checkpoint_time_index, "outlet pressure checkpoints"
                )

                print("[1d] Rank 0 reading flow checkpoints", flush=True)
                q_in_hist = import_flow_data(self.inlet_mesh, flow_inlet_file, comm=MPI.COMM_SELF)
                q_out_hist = import_flow_data(self.outlet_mesh, flow_outlet_file, comm=MPI.COMM_SELF)
                self.q_A_k = select_checkpoint_entry(
                    q_in_hist, self.checkpoint_time_index, "inlet flow checkpoints"
                )
                self.q_V_k = select_checkpoint_entry(
                    q_out_hist, self.checkpoint_time_index, "outlet flow checkpoints"
                )

                self.a_A_k = None
                self.a_V_k = None
                if area_inlet_file is not None and Path(area_inlet_file).exists():
                    print("[1d] Rank 0 reading inlet area checkpoint", flush=True)
                    a_in_hist = import_area_data(self.inlet_mesh, area_inlet_file, comm=MPI.COMM_SELF)
                    self.a_A_k = select_checkpoint_entry(
                        a_in_hist, self.checkpoint_time_index, "inlet area checkpoints"
                    )
                if area_outlet_file is not None and Path(area_outlet_file).exists():
                    print("[1d] Rank 0 reading outlet area checkpoint", flush=True)
                    a_out_hist = import_area_data(self.outlet_mesh, area_outlet_file, comm=MPI.COMM_SELF)
                    self.a_V_k = select_checkpoint_entry(
                        a_out_hist, self.checkpoint_time_index, "outlet area checkpoints"
                    )

                print("[1d] Rank 0 sampling terminal data", flush=True)
                self._extract_terminal_data()
                p_in_BC = float(self._sample_pressure_at_point(self.p_A_k, np.asarray(inlet_coord, dtype=float)))
                p_out_BC = float(self._sample_pressure_at_point(self.p_V_k, np.asarray(outlet_coord, dtype=float)))

                x_inlet = np.ascontiguousarray(np.asarray(self.x_inlet, dtype=np.float64))
                x_outlet = np.ascontiguousarray(np.asarray(self.x_outlet, dtype=np.float64))
                p_inlet = np.ascontiguousarray(np.asarray(self.p_inlet, dtype=np.float64))
                q_inlet = np.ascontiguousarray(np.asarray(self.q_inlet, dtype=np.float64))
                a_inlet = np.ascontiguousarray(np.asarray(self.a_inlet, dtype=np.float64))
                p_outlet = np.ascontiguousarray(np.asarray(self.p_outlet, dtype=np.float64))
                q_outlet = np.ascontiguousarray(np.asarray(self.q_outlet, dtype=np.float64))
                a_outlet = np.ascontiguousarray(np.asarray(self.a_outlet, dtype=np.float64))
                n_in = int(x_inlet.shape[0])
                n_out = int(x_outlet.shape[0])
                print("[1d] Rank 0 broadcasting sampled 1D data", flush=True)
        else:
            x_inlet = x_outlet = None
            p_inlet = q_inlet = a_inlet = None
            p_outlet = q_outlet = a_outlet = None
            p_in_BC = None
            p_out_BC = None
            n_in = 0
            n_out = 0

        n_in = comm.bcast(n_in, root=0)
        n_out = comm.bcast(n_out, root=0)
        if comm.rank != 0:
            x_inlet = np.empty((n_in, 3), dtype=np.float64)
            x_outlet = np.empty((n_out, 3), dtype=np.float64)
            p_inlet = np.empty(n_in, dtype=np.float64)
            q_inlet = np.empty(n_in, dtype=np.float64)
            a_inlet = np.empty(n_in, dtype=np.float64)
            p_outlet = np.empty(n_out, dtype=np.float64)
            q_outlet = np.empty(n_out, dtype=np.float64)
            a_outlet = np.empty(n_out, dtype=np.float64)

        comm.Bcast(x_inlet, root=0)
        comm.Bcast(x_outlet, root=0)
        comm.Bcast(p_inlet, root=0)
        comm.Bcast(q_inlet, root=0)
        comm.Bcast(a_inlet, root=0)
        comm.Bcast(p_outlet, root=0)
        comm.Bcast(q_outlet, root=0)
        comm.Bcast(a_outlet, root=0)
        p_in_BC = comm.bcast(p_in_BC, root=0)
        p_out_BC = comm.bcast(p_out_BC, root=0)

        if comm.rank == 0:
            print("[1d] 1D sampled data broadcast complete", flush=True)
        self.x_inlet = x_inlet
        self.x_outlet = x_outlet
        self.p_inlet = p_inlet
        self.q_inlet = q_inlet
        self.a_inlet = a_inlet
        self.p_outlet = p_outlet
        self.q_outlet = q_outlet
        self.a_outlet = a_outlet

        # --- branching data for inlet/outlet trees (optional but recommended) ---
        self.branch_coords_in  = None
        self.branch_ids_in     = None
        self.branch_normals_in = None
        self.branch_coords_out = None
        self.branch_ids_out    = None
        self.branch_normals_out = None

        if branching_in_file is not None:
            (
                self.branch_coords_in,
                self.branch_ids_in,
                self.branch_normals_in,
            ) = self._load_terminal_branch_metadata(branching_in_file)

        if branching_out_file is not None:
            (
                self.branch_coords_out,
                self.branch_ids_out,
                self.branch_normals_out,
            ) = self._load_terminal_branch_metadata(branching_out_file)

        self.p_inlet_parent = np.asarray(self.p_inlet, dtype=float).copy()
        self.p_outlet_parent = np.asarray(self.p_outlet, dtype=float).copy()
        self.p_inlet_parent_source = "terminal_pressure_fallback"
        self.p_outlet_parent_source = "terminal_pressure_fallback"
        if inlet_output_csv is not None and self.branch_ids_in is not None and len(self.branch_ids_in):
            parent = self._load_terminal_parent_pressures_from_0d_csv(
                inlet_output_csv,
                self.branch_ids_in,
                checkpoint_time_index=self.checkpoint_time_index,
            )
            if parent is not None:
                self.p_inlet_parent = parent
                self.p_inlet_parent_source = "0d_output_csv_pressure_in"
        if outlet_output_csv is not None and self.branch_ids_out is not None and len(self.branch_ids_out):
            parent = self._load_terminal_parent_pressures_from_0d_csv(
                outlet_output_csv,
                self.branch_ids_out,
                checkpoint_time_index=self.checkpoint_time_index,
            )
            if parent is not None:
                self.p_outlet_parent = parent
                self.p_outlet_parent_source = "0d_output_csv_pressure_in"

        # store user-input coordinates for channel inlet/outlet (3D)
        self.inlet_coord  = np.asarray(inlet_coord,  dtype=float)
        self.outlet_coord = np.asarray(outlet_coord, dtype=float)
        self.concave_bc_mode = str(concave_bc_mode).strip().lower()
        if self.concave_bc_mode not in {"dirichlet", "robin"}:
            raise ValueError(f"concave_bc_mode must be 'dirichlet' or 'robin', got {concave_bc_mode!r}")
        self.lp_arterial = float(lp_arterial)
        self.lp_venous = float(lp_venous)
        self.inlet_flux_correction = bool(inlet_flux_correction)
        self.inlet_flux_corr_max_iter = int(max(0, inlet_flux_corr_max_iter))
        self.inlet_flux_corr_relax = float(np.clip(inlet_flux_corr_relax, 0.0, 1.0))
        self.inlet_flux_corr_tol = float(max(0.0, inlet_flux_corr_tol))
        self._inlet_flux_scale_by_marker = {}

        inlet_pressure_source = "fallback_skip_1d" if self.skip_1d else "sampled_1d_seed_coordinate"
        outlet_pressure_source = "fallback_skip_1d" if self.skip_1d else "sampled_1d_seed_coordinate"
        if concave_inlet_pressure is not None:
            p_in_BC = float(concave_inlet_pressure)
            inlet_pressure_source = "explicit_concave_pressure_override"
        if concave_outlet_pressure is not None:
            p_out_BC = float(concave_outlet_pressure)
            outlet_pressure_source = "explicit_concave_pressure_override"

        # BC pressures sampled/overridden on rank 0 and made available on all ranks.
        self.p_in_BC = float(p_in_BC)
        self.p_out_BC = float(p_out_BC)
        self.concave_inlet_pressure_source = inlet_pressure_source
        self.concave_outlet_pressure_source = outlet_pressure_source
        self.concave_pressure_profile = str(concave_pressure_profile).strip().lower()
        if self.concave_pressure_profile not in {"constant", "linear"}:
            raise ValueError(
                "concave_pressure_profile must be 'constant' or 'linear', "
                f"got {concave_pressure_profile!r}"
            )
        self.concave_pressure_profile_axis = self._resolve_axis_index(
            concave_pressure_profile_axis,
            self.mesh.geometry.dim,
        )
        self.concave_pressure_profile_axis_name = "xyz"[self.concave_pressure_profile_axis]
        self.concave_art_pressure_low = (
            float(self.p_in_BC)
            if arterial_pressure_profile_low is None
            else float(arterial_pressure_profile_low)
        )
        self.concave_art_pressure_high = (
            float(self.p_in_BC)
            if arterial_pressure_profile_high is None
            else float(arterial_pressure_profile_high)
        )
        self.concave_ven_pressure_low = (
            float(self.p_out_BC)
            if venous_pressure_profile_low is None
            else float(venous_pressure_profile_low)
        )
        self.concave_ven_pressure_high = (
            float(self.p_out_BC)
            if venous_pressure_profile_high is None
            else float(venous_pressure_profile_high)
        )
        self._concave_marker_axis_bounds = {}

        if self.mesh.comm.rank == 0:
            print(f"Dirichlet BC pressures: p_in = {self.p_in_BC:.3e}, "
                  f"p_out = {self.p_out_BC:.3e}")
            if self.concave_pressure_profile == "linear":
                print(
                    "Concave pressure profile: "
                    f"axis={self.concave_pressure_profile_axis_name}, "
                    f"arterial=({self.concave_art_pressure_low:.3e}, {self.concave_art_pressure_high:.3e}), "
                    f"venous=({self.concave_ven_pressure_low:.3e}, {self.concave_ven_pressure_high:.3e})"
                )

    @staticmethod
    def _resolve_axis_index(axis, gdim):
        if isinstance(axis, str):
            token = axis.strip().lower()
            mapping = {"x": 0, "y": 1, "z": 2}
            if token not in mapping:
                raise ValueError(
                    f"Unsupported concave pressure profile axis {axis!r}; expected one of x/y/z"
                )
            idx = mapping[token]
        else:
            idx = int(axis)
        if idx < 0 or idx >= int(gdim):
            raise ValueError(
                f"Concave pressure profile axis index {idx} is invalid for geometry dimension {gdim}"
            )
        return idx

    def _marker_axis_interval(self, marker):
        marker = int(marker)
        cached = self._concave_marker_axis_bounds.get(marker, None)
        if cached is not None:
            return cached

        mesh = self.mesh
        facet_tags = self.facet_tags
        fdim = mesh.topology.dim - 1
        axis = int(self.concave_pressure_profile_axis)

        facets = facet_tags.find(marker)
        if facets.size == 0:
            found_local = 0
            local_min = np.inf
            local_max = -np.inf
        else:
            axis_coords = []
            for facet in facets:
                pts = self._entity_geometry_points(mesh, fdim, int(facet))
                axis_coords.extend(float(v) for v in pts[:, axis])
            coords = np.asarray(axis_coords, dtype=np.float64)
            found_local = 1 if coords.size > 0 else 0
            local_min = float(np.min(coords)) if coords.size > 0 else np.inf
            local_max = float(np.max(coords)) if coords.size > 0 else -np.inf

        found_any = mesh.comm.allreduce(found_local, op=MPI.SUM)
        if found_any <= 0:
            self._concave_marker_axis_bounds[marker] = None
            return None

        bounds = (
            float(mesh.comm.allreduce(local_min, op=MPI.MIN)),
            float(mesh.comm.allreduce(local_max, op=MPI.MAX)),
        )
        self._concave_marker_axis_bounds[marker] = bounds
        return bounds

    def _evaluate_linear_profile_on_coords(self, coords, marker, pressure_low, pressure_high, fallback_pressure):
        pts = np.asarray(coords, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        vals = np.full(pts.shape[0], float(fallback_pressure), dtype=np.float64)
        bounds = self._marker_axis_interval(marker)
        if bounds is None:
            return vals
        x_min, x_max = bounds
        denom = float(x_max - x_min)
        if abs(denom) <= 1e-14:
            vals.fill(0.5 * (float(pressure_low) + float(pressure_high)))
            return vals
        xi = np.clip((pts[:, int(self.concave_pressure_profile_axis)] - x_min) / denom, 0.0, 1.0)
        vals[:] = float(pressure_low) + (float(pressure_high) - float(pressure_low)) * xi
        return vals

    def _build_concave_boundary_function(self, W, marker, pressure_low, pressure_high, fallback_pressure):
        f = fem.Function(W)
        if self.concave_pressure_profile == "linear":
            coords = W.tabulate_dof_coordinates()
            values = self._evaluate_linear_profile_on_coords(
                coords,
                marker,
                pressure_low,
                pressure_high,
                fallback_pressure,
            )
            f.x.array[:] = values
        else:
            f.x.array[:] = float(fallback_pressure)
        f.x.scatter_forward()
        return f

    def _build_concave_boundary_expr(self, marker, pressure_low, pressure_high, fallback_pressure):
        mesh = self.mesh
        if self.concave_pressure_profile != "linear":
            return fem.Constant(mesh, dfx.default_scalar_type(float(fallback_pressure)))

        bounds = self._marker_axis_interval(marker)
        if bounds is None:
            return fem.Constant(mesh, dfx.default_scalar_type(float(fallback_pressure)))
        x_min, x_max = bounds
        denom = float(x_max - x_min)
        if abs(denom) <= 1e-14:
            mean_pressure = 0.5 * (float(pressure_low) + float(pressure_high))
            return fem.Constant(mesh, dfx.default_scalar_type(mean_pressure))

        x = ufl.SpatialCoordinate(mesh)
        xi = (x[int(self.concave_pressure_profile_axis)] - x_min) / denom
        return float(pressure_low) + (float(pressure_high) - float(pressure_low)) * xi

    def _sample_pressure_at_point(self, func, x_pt):
        """
        Given a P1 pressure function 'func' on some mesh and a 3D coordinate x_pt,
        return the value at the nearest DOF.
        """
        dof_coords = func.function_space.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)
        _, nn = tree.query(x_pt.reshape(1, -1))
        return float(func.x.array[int(nn[0])])
    def _build_wall_dirichlet_bcs(self, W):
        """
        Build Dirichlet BCs on the *concave* arterial/venous surfaces using facet tags
        read from the facet_file (now mesh_tags_well1.xdmf).

        Expected markers (from mesh generation script):
          - 31: arterial concave surface  -> use self.p_in_BC
          - 32: venous  concave surface  -> use self.p_out_BC
        """
        mesh = self.mesh
        facet_tags = self.facet_tags
        fdim = mesh.topology.dim - 1

        ART_CONCAVE = getattr(self, "arterial_concave_marker", 31)
        VEN_CONCAVE = getattr(self, "venous_concave_marker", 32)

        # Ensure facet entities/connectivities exist
        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(fdim, 0)
        mesh.topology.create_connectivity(mesh.topology.dim, fdim)

        art_facets = facet_tags.find(ART_CONCAVE)
        ven_facets = facet_tags.find(VEN_CONCAVE)

        if mesh.comm.rank == 0:
            print("Number of arterial concave facets tagged:", art_facets.size)
            print("Number of venous  concave facets tagged:", ven_facets.size)

        art_dofs = fem.locate_dofs_topological(W, fdim, art_facets)
        ven_dofs = fem.locate_dofs_topological(W, fdim, ven_facets)

        p_art = self._build_concave_boundary_function(
            W,
            ART_CONCAVE,
            self.concave_art_pressure_low,
            self.concave_art_pressure_high,
            self.p_in_BC,
        )
        p_ven = self._build_concave_boundary_function(
            W,
            VEN_CONCAVE,
            self.concave_ven_pressure_low,
            self.concave_ven_pressure_high,
            self.p_out_BC,
        )

        bc_art = fem.dirichletbc(p_art, art_dofs, W)
        bc_ven = fem.dirichletbc(p_ven, ven_dofs, W)

        return [bc_art, bc_ven]

    def _build_concave_robin_terms(self, p, v):
        """
        Robin transfer on concave faces:
          (-K grad p · n) = Lp * (p_media - p)
        Weak form contribution:
          a += ∫ Lp * p * v ds
          L += ∫ Lp * p_media * v ds
        """
        mesh = self.mesh
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        a_r = 0
        L_r = 0

        if self.lp_arterial > 0.0:
            lp_a = fem.Constant(mesh, dfx.default_scalar_type(self.lp_arterial))
            p_a = self._build_concave_boundary_expr(
                self.arterial_concave_marker,
                self.concave_art_pressure_low,
                self.concave_art_pressure_high,
                self.p_in_BC,
            )
            a_r += lp_a * p * v * ds(int(self.arterial_concave_marker))
            L_r += lp_a * p_a * v * ds(int(self.arterial_concave_marker))

        if self.lp_venous > 0.0:
            lp_v = fem.Constant(mesh, dfx.default_scalar_type(self.lp_venous))
            p_v = self._build_concave_boundary_expr(
                self.venous_concave_marker,
                self.concave_ven_pressure_low,
                self.concave_ven_pressure_high,
                self.p_out_BC,
            )
            a_r += lp_v * p * v * ds(int(self.venous_concave_marker))
            L_r += lp_v * p_v * v * ds(int(self.venous_concave_marker))

        return a_r, L_r


    def _load_terminal_branch_metadata(self, csv_path):
        """
        Read a branchingData_*.csv and return:
          - coords: (N_term, 3) array of distal coordinates
          - ids:    (N_term,) array of branch IDs (or 0..N_term-1 if not present)
          - normals:(N_term, 3) array of terminal axial directions (W basis when available)

        Assumptions (tweak if your headers differ):
          * child columns contain 'child' in their name
          * distal coordinates in one of:
              (x_dist, y_dist, z_dist) or
              (xDist,  yDist,  zDist)  or
              (x, y, z) as a fallback
          * branch ID column is one of:
              branchID, branch_id, id, ID
        """
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return (
                np.zeros((0, 3)),
                np.zeros((0,), dtype=int),
                np.zeros((0, 3)),
            )

        # 1) find child columns and define terminal rows (no children)
        child_cols = [c for c in df.columns if "child" in c.lower()]
        if child_cols:
            # terminal = all child columns are NaN or empty string
            child_df = df[child_cols]
            mask_terminal = child_df.isna() | (child_df == "")
            mask_terminal = mask_terminal.all(axis=1)
        else:
            # if no child columns found, treat all as terminal
            mask_terminal = np.ones(len(df), dtype=bool)

        # 2) find distal coordinate columns
        candidate_sets = [
            ("x_dist", "y_dist", "z_dist"),
            ("xDist",  "yDist",  "zDist"),
            ("xd",     "yd",     "zd"),
            ("x",      "y",      "z"),
            ("distalCoordsX", "distalCoordsY", "distalCoordsZ")
        ]
        for cols in candidate_sets:
            if all(c in df.columns for c in cols):
                xcol, ycol, zcol = cols
                break
        else:
            raise ValueError(
                f"Could not find distal coordinate columns in {csv_path}. "
                f"Expected something like x_dist/y_dist/z_dist or x/y/z."
            )

        coords = df.loc[mask_terminal, [xcol, ycol, zcol]].to_numpy(dtype=float)

        if all(c in df.columns for c in ("W1", "W2", "W3")):
            normals = df.loc[mask_terminal, ["W1", "W2", "W3"]].to_numpy(dtype=float)
        elif all(c in df.columns for c in ("proximalCoordsX", "proximalCoordsY", "proximalCoordsZ")):
            pcols = ["proximalCoordsX", "proximalCoordsY", "proximalCoordsZ"]
            prox = df.loc[mask_terminal, pcols].to_numpy(dtype=float)
            normals = coords - prox
        else:
            normals = np.zeros_like(coords)

        if len(normals):
            nn = np.linalg.norm(normals, axis=1, keepdims=True)
            nn[nn <= 0.0] = 1.0
            normals = normals / nn

        # 3) branch IDs (optional but nice to keep)
        branch_id_col = None
        for cand in ["Index", "index", "branchID", "branch_id", "id", "ID"]:
            if cand in df.columns:
                branch_id_col = cand
                break

        terminal_rows = np.flatnonzero(np.asarray(mask_terminal, dtype=bool))
        ids = terminal_rows.astype(int)
        if branch_id_col is not None:
            raw_ids = pd.to_numeric(df.loc[mask_terminal, branch_id_col], errors="coerce").to_numpy()
            if raw_ids.size == ids.size and np.all(np.isfinite(raw_ids)):
                ids = raw_ids.astype(int)

        return coords, ids, normals

    @staticmethod
    def _parse_0d_branch_segment_name(name: str):
        import re

        text = str(name).strip()
        patterns = [
            r"branch\s+(-?\d+)_seg(-?\d+)",
            r"branch_(-?\d+)_seg_?(-?\d+)",
            r".*branch\s*[_ ](-?\d+).*seg\s*[_ ]?(-?\d+)",
        ]
        for pat in patterns:
            m = re.match(pat, text)
            if m:
                return int(m.group(1)), int(m.group(2))
        return None

    def _load_terminal_parent_pressures_from_0d_csv(
        self,
        csv_path,
        branch_ids,
        checkpoint_time_index: int = -1,
    ):
        path = Path(csv_path).expanduser()
        if not path.exists():
            return None

        rows_by_branch: dict[int, list[tuple[float, float]]] = {}
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    parsed = self._parse_0d_branch_segment_name(row.get("name", ""))
                    if parsed is None:
                        continue
                    branch, segment = parsed
                    if int(segment) != 0:
                        continue
                    try:
                        time = float(row.get("time", 0.0))
                        pressure = float(row["pressure_in"])
                    except Exception:
                        continue
                    rows_by_branch.setdefault(int(branch), []).append((time, pressure))
        except Exception:
            return None

        if not rows_by_branch:
            return None

        out = []
        for raw_id in np.asarray(branch_ids):
            try:
                branch_id = int(raw_id)
            except Exception:
                return None
            rows = sorted(rows_by_branch.get(branch_id, []), key=lambda item: item[0])
            if not rows:
                return None
            idx = int(checkpoint_time_index)
            if idx < 0:
                idx = len(rows) + idx
            idx = min(max(idx, 0), len(rows) - 1)
            out.append(float(rows[idx][1]))
        return np.asarray(out, dtype=float)

    def _load_terminal_branch_coords(self, csv_path):
        coords, ids, _normals = self._load_terminal_branch_metadata(csv_path)
        return coords, ids

    # ========================================================
    # Step A: Extract terminal coordinates and values
    # ========================================================
    def _extract_terminal_data(self):
        """
        Extract terminal coordinates (marker==2) directly from inlet/outlet meshtags.
        Then sample the nearest pressure and velocity values.
        """

        # ------------------------------------------------------
        # 1) Extract terminal coordinates directly from meshtags
        # ------------------------------------------------------
        def extract_terminal_coords(mesh_obj, tags):
            marker = 2
            idx = np.where(tags.values == marker)[0]
            if len(idx) == 0:
                return np.zeros((0, mesh_obj.geometry.dim))

            entity_ids = tags.indices[idx]

            # Directly pick the vertex coordinates (ADIOS tags apply to vertices)
            return mesh_obj.geometry.x[entity_ids]

        # inlet terminals
        self.x_inlet = extract_terminal_coords(self.inlet_mesh, self.inlet_tags)
        # outlet terminals
        self.x_outlet = extract_terminal_coords(self.outlet_mesh, self.outlet_tags)

        # ------------------------------------------------------
        # 2) Map each terminal coord to nearest DOF in pressure/flow fields
        # ------------------------------------------------------
        def sample_field_at_points(func, pts):
            if len(pts) == 0:
                return np.zeros(0)

            bs = int(getattr(func.function_space.dofmap, "bs", 1))
            dof_coords = func.function_space.tabulate_dof_coordinates()
            tree = cKDTree(dof_coords)
            _, nn = tree.query(pts)
            nn = nn.astype(int)
            arr = func.x.array
            if bs > 1 and arr.size == dof_coords.shape[0] * bs:
                return arr.reshape(dof_coords.shape[0], bs)[nn]
            return arr[nn]

        # inlet pressures/flows
        self.p_inlet = sample_field_at_points(self.p_A_k, self.x_inlet)
        self.q_inlet = sample_field_at_points(self.q_A_k, self.x_inlet)
        self.a_inlet = (
            sample_field_at_points(self.a_A_k, self.x_inlet)
            if self.a_A_k is not None else np.zeros(len(self.x_inlet))
        )

        # outlet pressures/flows
        self.p_outlet = sample_field_at_points(self.p_V_k, self.x_outlet)
        self.q_outlet = sample_field_at_points(self.q_V_k, self.x_outlet)
        self.a_outlet = (
            sample_field_at_points(self.a_V_k, self.x_outlet)
            if self.a_V_k is not None else np.zeros(len(self.x_outlet))
        )

        print(f"Found {len(self.x_inlet)} inlet terminals")
        print(f"Found {len(self.x_outlet)} outlet terminals")

    def _build_reordering_indices(self, coords_terminals, branch_coords):
        """
        For each distal coordinate in branch_coords, find the nearest
        Darcy terminal coordinate in coords_terminals.

        Returns an integer index array idx such that
            coords_terminals[idx[i], :] ~ branch_coords[i, :]
        """
        if coords_terminals is None or len(coords_terminals) == 0:
            return None
        if branch_coords is None or len(branch_coords) == 0:
            return None

        tree = cKDTree(coords_terminals)
        _, idx = tree.query(branch_coords)
        return idx.astype(int)


    # ========================================================
    # Terminal facet marker utilities (from mesh_tags_well*.xdmf)
    # ========================================================
    def _get_terminal_marker_sets(self):
        """
        Return sorted unique terminal facet markers for inlet/outlet based on the
        conventions used in mesh.py:
          inlet terminals  : [inlet_base_marker,  inlet_base_marker+1,  ...]
          outlet terminals : [outlet_base_marker, outlet_base_marker+1, ...]
        """
        local_vals = np.unique(np.asarray(self.facet_tags.values, dtype=np.int32))
        gathered = self.mesh.comm.allgather(local_vals.tolist())
        merged = sorted({int(v) for vals in gathered for v in vals})
        vals = np.asarray(merged, dtype=np.int32)
        inlet_base  = getattr(self, "inlet_base_marker", 1)
        outlet_base = getattr(self, "outlet_base_marker", 100)

        inlet_marks  = np.sort(vals[(vals >= inlet_base)  & (vals < outlet_base)])
        outlet_marks = np.sort(vals[(vals >= outlet_base) & (vals < outlet_base + 1000)])
        return inlet_marks.astype(int), outlet_marks.astype(int)

    def _marker_centroids(self, markers):
        """
        Compute one representative coordinate for each marker: the mean of boundary
        facet centroids in that marker set.
        """
        mesh = self.mesh
        tdim = mesh.topology.dim
        fdim = tdim - 1

        centroids = np.zeros((len(markers), 3), dtype=np.float64)
        for i, m in enumerate(markers):
            facets = self.facet_tags.find(int(m))
            local_sum = np.zeros(3, dtype=np.float64)
            local_count = np.array([0], dtype=np.int64)
            for f in facets:
                local_sum += self._entity_geometry_points(mesh, fdim, int(f)).mean(axis=0)
                local_count[0] += 1
            global_sum = np.zeros_like(local_sum)
            global_count = np.zeros_like(local_count)
            self.mesh.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
            self.mesh.comm.Allreduce(local_count, global_count, op=MPI.SUM)
            if int(global_count[0]) == 0:
                centroids[i, :] = np.nan
            else:
                centroids[i, :] = global_sum / float(global_count[0])
        return centroids

    def _marker_global_facet_counts(self, marker: int) -> tuple[int, int]:
        ext_facets, int_facets = self._split_marker_facets(int(marker))
        n_ext = int(self.mesh.comm.allreduce(int(ext_facets.size), op=MPI.SUM))
        n_int = int(self.mesh.comm.allreduce(int(int_facets.size), op=MPI.SUM))
        return n_ext, n_int

    def _assign_markers_to_terminals(self, marker_centroids, terminal_coords):
        """
        Match each marker centroid to the nearest terminal coordinate (KD-tree).
        Returns idx such that terminal_coords[idx[i]] corresponds to marker i.
        """
        if marker_centroids.size == 0 or terminal_coords.size == 0:
            return np.zeros((0,), dtype=int)
        tree = cKDTree(terminal_coords)
        _, idx = tree.query(marker_centroids)
        return idx.astype(int)

    def _get_terminal_marker_data(self):
        """
        Return inlet/outlet terminal markers, their centroids, and nearest 1D-terminal
        index mappings. This is shared by the terminal Neumann and Dirichlet builders.
        """
        inlet_marks, outlet_marks = self._get_terminal_marker_sets()
        inlet_c = self._marker_centroids(inlet_marks)
        outlet_c = self._marker_centroids(outlet_marks)

        idx_in = self._assign_markers_to_terminals(inlet_c, self.x_inlet)
        idx_out = self._assign_markers_to_terminals(outlet_c, self.x_outlet)

        self.inlet_terminal_markers = inlet_marks
        self.outlet_terminal_markers = outlet_marks
        self.inlet_terminal_centroids = inlet_c
        self.outlet_terminal_centroids = outlet_c
        # Marker-order -> 1D-terminal-order maps used by BC/diagnostics alignment.
        self.inlet_marker_to_1d_idx = idx_in
        self.outlet_marker_to_1d_idx = idx_out

        return inlet_marks, outlet_marks, inlet_c, outlet_c, idx_in, idx_out

    def _build_terminal_flux_form(self, v):
        """
        Build a Neumann-type RHS term that imposes inlet terminal flow rates
        *facet-wise* on the boundary facets tagged as terminal inlet faces.

        For each terminal marker m:
          g_m = Q_m / Area(m)
          RHS += ∫ (-g_m) * v ds(m)

        Notes
        -----
        - Inlet terminal flow is sampled from flow_checkpoint_inlet.bp
          and mapped to 3D terminal markers.
        - Outlet terminals are handled separately as Dirichlet pressure BCs
          from the venous tree.
        """
        mesh = self.mesh
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))

        inlet_marks, _, _, _, idx_in, _ = self._get_terminal_marker_data()
        self.inlet_marker_area_3d = np.zeros(len(inlet_marks), dtype=float)
        self.inlet_marker_area_1d = np.zeros(len(inlet_marks), dtype=float)
        self.inlet_marker_q_target = np.zeros(len(inlet_marks), dtype=float)

        L_flux = 0
        # Inlet terminals
        for j, m in enumerate(inlet_marks):
            if j >= len(idx_in):
                continue
            Q = float(self.q_inlet[idx_in[j]]) if len(self.q_inlet) else 0.0
            print("Flow target for marker", m, "is", Q)
            scale = float(self._inlet_flux_scale_by_marker.get(int(m), 1.0))
            Q *= scale
            A_face_local = float(fem.assemble_scalar(fem.form(one * ds(int(m)))))
            A_face = float(mesh.comm.allreduce(A_face_local, op=MPI.SUM))
            A_ckpt = float(self.a_inlet[idx_in[j]]) if len(self.a_inlet) else 0.0
            self.inlet_marker_area_3d[j] = A_face
            self.inlet_marker_area_1d[j] = A_ckpt
            self.inlet_marker_q_target[j] = Q
            # Enforce total terminal flow on the actual 3D marker area.
            g = Q / A_face if A_face > 0.0 else 0.0
            gc = fem.Constant(mesh, dfx.default_scalar_type(g))
            # For u = -K grad(p), natural BC term is K grad(p).n = -u.n.
            # To impose target u.n = g, RHS contribution must be -g * v.
            L_flux += -gc * v * ds(int(m))

        return L_flux

    def _get_inlet_marker_targets(self):
        inlet_marks, _, _, _, idx_in, _ = self._get_terminal_marker_data()
        q_target = np.zeros(len(inlet_marks), dtype=float)
        for j, _m in enumerate(inlet_marks):
            if j < len(idx_in) and len(self.q_inlet):
                q_target[j] = float(self.q_inlet[idx_in[j]])
        return inlet_marks.astype(int), q_target

    def _measure_inlet_marker_flux_from_pressure(self, p_h):
        inlet_marks = getattr(self, "inlet_terminal_markers", np.array([], dtype=int))
        if len(inlet_marks) == 0:
            return np.zeros(0, dtype=float)
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(self.mesh)
        out = np.zeros(len(inlet_marks), dtype=float)
        for j, m in enumerate(inlet_marks):
            expr = -ufl.dot(self.K_tensor * ufl.grad(p_h), n) * ds(int(m))
            local_flux = float(fem.assemble_scalar(fem.form(expr)))
            out[j] = float(self.mesh.comm.allreduce(local_flux, op=MPI.SUM))
        return out

    # ========================================================
    # Step B: Build q_src density approximation for Darcy
    # ========================================================
    def _build_q_src(self, W):
        """Return a zero volumetric source field (terminal flows are now imposed facet-wise)."""
        q_src = fem.Function(W)
        q_src.x.array[:] = 0.0
        q_src.x.scatter_forward()
        return q_src


    # ========================================================
    # Step C: Dirichlet BCs at terminal pressures
    # ========================================================
    def _build_terminal_bcs(self, W):
        """
        Build Dirichlet pressure BCs on both inlet and outlet terminal facets
        using pressures sampled from the arterial/venous trees.
        """
        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        bcs = []
        inlet_marks, outlet_marks, _, _, idx_in, idx_out = self._get_terminal_marker_data()

        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(mesh.topology.dim, fdim)

        n_in = 0
        for j, m in enumerate(inlet_marks):
            if j >= len(idx_in) or len(self.p_inlet) == 0:
                continue
            facets = self.facet_tags.find(int(m))
            if facets.size == 0:
                continue
            dofs = fem.locate_dofs_topological(W, fdim, facets)
            if dofs.size == 0:
                continue
            pval = float(self.p_inlet[idx_in[j]])
            bc_val = fem.Constant(mesh, dfx.default_scalar_type(pval))
            bcs.append(fem.dirichletbc(bc_val, dofs, W))
            n_in += 1

        n_out = 0
        for j, m in enumerate(outlet_marks):
            if j >= len(idx_out) or len(self.p_outlet) == 0:
                continue
            facets = self.facet_tags.find(int(m))
            if facets.size == 0:
                continue
            dofs = fem.locate_dofs_topological(W, fdim, facets)
            if dofs.size == 0:
                continue
            pval = float(self.p_outlet[idx_out[j]])
            bc_val = fem.Constant(mesh, dfx.default_scalar_type(pval))
            bcs.append(fem.dirichletbc(bc_val, dofs, W))
            n_out += 1

        if mesh.comm.rank == 0:
            print("Number of arterial terminal pressure BCs:", n_in)
            print("Number of venous terminal pressure BCs:", n_out)
        return bcs

    def _build_pressure_penalty_fields(self, W, radius=0.2, gamma=10.0):
        """
        Build beta(x) and beta_p0(x)=beta(x)*p_target(x) in P1 space.
        beta is nonzero only near terminal points.

        radius: support radius in the same units as your mesh coordinates
        gamma:  dimensionless strength multiplier
        """
        mesh = self.mesh
        K = 1e-5/1.0  # kappa/mu (use your actual values / pass in)
        # beta should scale like K/h^2. We'll estimate h from the mesh.
        h = dfx.cpp.mesh.h(mesh._cpp_object, mesh.topology.dim, np.arange(mesh.topology.index_map(mesh.topology.dim).size_local, dtype=np.int32))
        h_mean = float(np.mean(h)) if len(h) else 1.0
        beta0 = gamma * K / (h_mean**2)

        beta = fem.Function(W)
        beta_p0 = fem.Function(W)
        beta.x.array[:] = 0.0
        beta_p0.x.array[:] = 0.0

        X = W.tabulate_dof_coordinates()
        tree = cKDTree(X)

        def add_terminals(pts, pvals):
            if len(pts) == 0:
                return

            for x, p0 in zip(pts, pvals):
                # DOFs within radius
                idxs = tree.query_ball_point(x, r=radius)
                if len(idxs) == 0:
                    # fallback: nearest
                    _, nn = tree.query(x)
                    idxs = [int(nn)]

                idxs = np.array(idxs, dtype=int)
                dists = np.linalg.norm(X[idxs] - x, axis=1)

                # smooth weights (Gaussian)
                w = np.exp(-(dists / radius)**2)
                w /= np.sum(w)

                beta.x.array[idxs] += beta0 * w
                beta_p0.x.array[idxs] += beta0 * p0 * w

        # Apply to both inlet and outlet terminals if desired
        # add_terminals(self.x_inlet,  self.p_inlet)
        add_terminals(self.x_outlet, self.p_outlet)

        beta.x.scatter_forward()
        beta_p0.x.scatter_forward()
        return beta, beta_p0

    def _build_point_pressure_penalty(self, W, beta_value=1e6, use_inlet=True, use_outlet=True):
        """
        Pointwise weak pressure enforcement:
        adds beta at nearest DOF(s) to each terminal point.
        Returns:
        beta (Function in W): zeros except at selected DOFs
        rhs_beta (Function in W): beta * p_target at those DOFs
        """
        mesh = self.mesh
        beta = fem.Function(W)
        rhs_beta = fem.Function(W)
        beta.x.array[:] = 0.0
        rhs_beta.x.array[:] = 0.0

        dof_coords = W.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        def add(pts, pvals):
            if len(pts) == 0:
                return
            _, dofs = tree.query(pts)
            dofs = np.atleast_1d(dofs).astype(int)
            for d, p0 in zip(dofs, pvals):
                beta.x.array[d] += beta_value
                rhs_beta.x.array[d] += beta_value * float(p0)

        if use_inlet:
            add(self.x_inlet, self.p_inlet)
        if use_outlet:
            add(self.x_outlet, self.p_outlet)

        beta.x.scatter_forward()
        rhs_beta.x.scatter_forward()
        return beta, rhs_beta

    def _apply_pressure_gauge_postsolve(self, p_h):
        """
        Shift p_h by a constant so that its outlet pressures match the 1D model
        in some sense (single point or average).

        This does NOT change u_h, since u_h depends on grad(p_h).
        """

        mesh = self.mesh
        Wp = p_h.function_space
        dof_coords = Wp.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        # Sample 3D pressure at outlet terminals
        if len(self.x_outlet) == 0:
            if mesh.comm.rank == 0:
                print("No outlet terminals, skipping post-gauge.")
            return
        
        x_gauge = self.x_outlet[0:1]
        p_gauge = self.p_outlet[0]
        _, dofs = tree.query(x_gauge)
        d = int(dofs[0])

        _, nn = tree.query(self.x_outlet)
        nn = nn.astype(int)
        p_3d_out = p_h.x.array[nn]       # 3D Darcy pressures at outlet locations
        p_1d_out = self.p_outlet         # from your 1D model

        # Option A: match average outlet pressure
        offset = float(np.mean(p_1d_out - p_3d_out))

        # Option B (alternative): match one particular outlet, e.g. the first
        # offset = float(p_1d_out[0] - p_3d_out[0])

        # Apply shift
        p_h.x.array[:] += offset
        # p_h.x.array[d] = p_gauge # use if setting one terminal exactly
        p_h.x.scatter_forward()

        if mesh.comm.rank == 0:
            print(f"Applied post-gauge offset Δp = {offset:.6g}")

    
    # ========================================================
    # Step C-alt: Penalty-like enforcement of outlet pressures
    # ========================================================
    def _build_outlet_penalty_terms(self, W, penalty_factor=1e6):
        """
        Build two P1 fields:
          - penalty(x): alpha(x) nonzero near outlet DOFs
          - rhs_penalty(x): alpha(x) * p_outlet

        so that we can add
            ∫ penalty * p * v dx    to the bilinear form
            ∫ rhs_penalty * v dx    to the RHS

        This approximates a Dirichlet condition at the outlet
        in a 'smeared' volumetric sense, similar to how q_src
        is imposed.
        """
        mesh = self.mesh

        penalty = fem.Function(W)
        rhs_penalty = fem.Function(W)

        penalty.x.array[:] = 0.0
        rhs_penalty.x.array[:] = 0.0

        # DOF coordinates / KD-tree on P1 space
        dof_coords = W.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        if len(self.x_outlet) > 0:
            _, dofs = tree.query(self.x_outlet)
            dofs = np.atleast_1d(dofs).astype(int)

            for d, pval in zip(dofs, self.p_outlet):
                penalty.x.array[d]     += penalty_factor
                rhs_penalty.x.array[d] += penalty_factor * pval

        return penalty, rhs_penalty
    
    def build_beta_fields(self, W, term_pts, p_targets, K, r, gamma=1.0):
        """
        W: P1 space
        term_pts: (N,3) terminal coordinates
        p_targets: (N,) target pressures (e.g., from 0D/NS)
        K: scalar permeability/mu (kappa_over_mu)
        r: smear radius
        gamma: strength factor
        """
        mesh = W.mesh
        beta0 = gamma * float(K) / (r**2)

        beta = fem.Function(W)
        beta_p0 = fem.Function(W)
        beta.x.array[:] = 0.0
        beta_p0.x.array[:] = 0.0

        X = W.tabulate_dof_coordinates()
        tree = cKDTree(X)

        for x, p0 in zip(term_pts, p_targets):
            idxs = tree.query_ball_point(x, r=r)
            if len(idxs) == 0:
                # fallback: nearest dof
                _, nn = tree.query(x)
                idxs = [int(nn)]

            idxs = np.asarray(idxs, dtype=int)
            d = np.linalg.norm(X[idxs] - x, axis=1)
            w = np.exp(-(d / r)**2)
            w /= np.sum(w)

            beta.x.array[idxs] += beta0 * w
            beta_p0.x.array[idxs] += beta0 * float(p0) * w

        beta.x.scatter_forward()
        beta_p0.x.scatter_forward()
        return beta, beta_p0

    
    # ========================================================
    # Step D: Compute interface BC to pass back to ROM solver
    # ========================================================

    def _compute_q_from_div_u(self, u_h):
        """
        Compute f_h ≈ div(u_h) (in weak/L2-projected sense) and then
        q = cell_volume * f_h at inlet/outlet terminal coordinates.
        """

        mesh = self.mesh

        # Scalar space for divergence (P1)
        Q_el = element("Lagrange", mesh.basix_cell(), 1)
        Q = fem.functionspace(mesh, Q_el)

        f = ufl.TrialFunction(Q)
        v = ufl.TestFunction(Q)

        # L2 projection of div(u): (f, v) = -(u, grad v)
        a = ufl.inner(f, v) * ufl.dx
        L = -ufl.inner(u_h, ufl.grad(v)) * ufl.dx

        problem = _make_linear_problem(
            a,
            L,
            petsc_options_prefix="darcy_div_projection_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        f_h = problem.solve()   # f_h ~ div(u_h)

        # Global average cell volume (same as you used in _build_q_src)
        tdim = mesh.topology.dim
        ncells = mesh.topology.index_map(tdim).size_local
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        total_vol = fem.assemble_scalar(fem.form(one * ufl.dx))
        cell_vol = total_vol / max(ncells, 1)

        # Sample f_h at terminal coordinates and convert to q = cell_vol * f
        dof_coords = Q.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        def sample_q(pts):
            if len(pts) == 0:
                empty = np.zeros(0, dtype=float)
                return empty, empty
            _, nn = tree.query(pts)
            nn = nn.astype(int)
            f_vals = f_h.x.array[nn]
            return cell_vol * f_vals, f_vals  # q, div(u)

        q_inlet,  divu_inlet  = sample_q(self.x_inlet)
        q_outlet, divu_outlet = sample_q(self.x_outlet)

        return f_h, q_inlet, q_outlet, divu_inlet, divu_outlet, cell_vol


    def _compute_interface_bc(self, p_h, u_h, q_src, Q_art_leak, Q_ven_leak):
        mesh = self.mesh
        gdim = mesh.geometry.dim


        # -------------------------------------------------------------
        # Facet-wise sampling at terminal markers (preferred)
        # -------------------------------------------------------------
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))

        inlet_marks  = getattr(self, "inlet_terminal_markers",  np.array([], dtype=int))
        outlet_marks = getattr(self, "outlet_terminal_markers", np.array([], dtype=int))

        def facet_average_scalar(f):
            if len(f) == 0:
                return np.zeros(0)
            out = []
            for m in f:
                A = fem.assemble_scalar(fem.form(one * ds(int(m))))
                if A == 0:
                    out.append(0.0)
                else:
                    out.append(float(fem.assemble_scalar(fem.form(p_h * ds(int(m))))) / float(A))
            return np.array(out, dtype=float)

        def facet_average_vector(f):
            if len(f) == 0:
                return np.zeros((0, gdim))
            out = []
            for m in f:
                A = fem.assemble_scalar(fem.form(one * ds(int(m))))
                if A == 0:
                    out.append(np.zeros(gdim))
                else:
                    # assemble each component
                    comps = []
                    for k in range(gdim):
                        comps.append(float(fem.assemble_scalar(fem.form(u_h[k] * ds(int(m))))) / float(A))
                    out.append(np.array(comps))
            return np.vstack(out) if len(out) else np.zeros((0, gdim))

        def facet_flux(f):
            if len(f) == 0:
                return np.zeros(0)
            out = []
            for m in f:
                out.append(float(fem.assemble_scalar(fem.form(ufl.dot(u_h, ufl.FacetNormal(mesh)) * ds(int(m))))))
            return np.array(out, dtype=float)

        def facet_flux_from_pressure(f):
            """
            Compute flux from the primal Darcy field directly:
              u = -K grad(p),  Q = ∫ u·n ds

            For pressure-based solvers this is the consistent quantity to use for
            interface coupling. Mixed solvers delete self.K_tensor before calling
            this routine so they fall back to the primary H(div) velocity field.
            """
            if len(f) == 0:
                return np.zeros(0)
            if not hasattr(self, "K_tensor"):
                return facet_flux(f)
            n = ufl.FacetNormal(mesh)
            out = []
            for m in f:
                expr = -ufl.dot(self.K_tensor * ufl.grad(p_h), n) * ds(int(m))
                out.append(float(fem.assemble_scalar(fem.form(expr))))
            return np.array(out, dtype=float)

        # These replace point-wise pressure/velocity/flow samples at terminals
        p_inlet_raw  = facet_average_scalar(inlet_marks)
        p_outlet_raw = facet_average_scalar(outlet_marks)

        u_inlet_raw  = facet_average_vector(inlet_marks)
        u_outlet_raw = facet_average_vector(outlet_marks)

        # Keep interface_bc fluxes on one consistent representation.
        # For primal/CG solvers this should come from the solved pressure field
        # directly rather than from the postprocessed RT projection.
        q_inlet_raw  = facet_flux_from_pressure(inlet_marks)
        q_outlet_raw = facet_flux_from_pressure(outlet_marks)

        # Representative coords: marker centroids (computed earlier in _build_terminal_flux_form)
        coords_in = getattr(self, "inlet_terminal_centroids",  np.zeros((len(inlet_marks), 3)))
        coords_out = getattr(self, "outlet_terminal_centroids", np.zeros((len(outlet_marks), 3)))

        # Keep div(u) and cell_volume outputs for backward compatibility
        # (computed using previous point-sampling approach)
        # (point-sampling code removed: now using facet-wise sampling above)
        # ----- q = cell_vol * div(u) at interface points (Darcy order) -----
        f_h, q_inlet_divu_raw, q_outlet_divu_raw, divu_inlet_raw, divu_outlet_raw, cell_vol = \
            self._compute_q_from_div_u(u_h)

        # Optional global checks
        f_form = fem.form(f_h * ufl.dx)
        Q_from_divu = float(fem.assemble_scalar(f_form))
        qsrc_form = fem.form(q_src * ufl.dx)
        Q_from_qsrc = float(fem.assemble_scalar(qsrc_form))
        print("Total ∫ div(u) dx  =", Q_from_divu)
        print("Total ∫ q_src dx   =", Q_from_qsrc)

        # ------------------------------------------------------------------
        # REORDER everything to match branchingData_0 / branchingData_1 order
        # ------------------------------------------------------------------
        # Build target vectors in marker order first, then apply optional
        # branchingData reordering so targets and measured values stay aligned.
        def _safe_reindex(vals, idx):
            arr = np.asarray(vals, dtype=float)
            ids = np.asarray(idx, dtype=int)
            if ids.size == 0:
                return arr
            if np.any(ids < 0) or np.any(ids >= len(arr)):
                return arr
            return arr[ids]

        idx_in_marker = np.asarray(getattr(self, "inlet_marker_to_1d_idx", np.array([], dtype=int)), dtype=int)
        idx_out_marker = np.asarray(getattr(self, "outlet_marker_to_1d_idx", np.array([], dtype=int)), dtype=int)

        p_inlet_target_marker = _safe_reindex(self.p_inlet, idx_in_marker)
        p_outlet_target_marker = _safe_reindex(self.p_outlet, idx_out_marker)
        q_inlet_target_marker = _safe_reindex(self.q_inlet, idx_in_marker)

        # defaults: marker-order output (in case branchingData not provided)
        idx_in_branch = idx_out_branch = None

        if self.branch_coords_in is not None and len(self.branch_coords_in) > 0:
            idx_in_branch = self._build_reordering_indices(coords_in, self.branch_coords_in)
        if self.branch_coords_out is not None and len(self.branch_coords_out) > 0:
            idx_out_branch = self._build_reordering_indices(coords_out, self.branch_coords_out)

        if idx_in_branch is not None:
            p_inlet   = p_inlet_raw[idx_in_branch]
            u_inlet   = u_inlet_raw[idx_in_branch, :]
            q_inlet   = q_inlet_raw[idx_in_branch]
            divu_in   = divu_inlet_raw[idx_in_branch]
            coords_in = self.x_inlet[idx_in_branch, :]
            ids_in    = self.branch_ids_in
        else:
            p_inlet   = p_inlet_raw
            u_inlet   = u_inlet_raw
            q_inlet   = q_inlet_raw
            divu_in   = divu_inlet_raw
            coords_in = coords_in
            ids_in    = None

        if idx_out_branch is not None:
            p_outlet   = p_outlet_raw[idx_out_branch]
            u_outlet   = u_outlet_raw[idx_out_branch, :]
            q_outlet   = q_outlet_raw[idx_out_branch]
            divu_out   = divu_outlet_raw[idx_out_branch]
            coords_out = self.x_outlet[idx_out_branch, :]
            ids_out    = self.branch_ids_out
        else:
            p_outlet   = p_outlet_raw
            u_outlet   = u_outlet_raw
            q_outlet   = q_outlet_raw
            divu_out   = divu_outlet_raw
            coords_out = coords_out
            ids_out    = None

        # Means (in the final, reordered indexing)
        p_inlet_mean  = float(np.mean(p_inlet))  if len(p_inlet)  > 0 else float("nan")
        p_outlet_mean = float(np.mean(p_outlet)) if len(p_outlet) > 0 else float("nan")

        # ------------------------------------------------------------------
        # Build JSON dictionary in *branchingData* order
        # ------------------------------------------------------------------
        interface_bc = {
            # pressure
            "p_inlet_nodes":   p_inlet.tolist(),
            "p_outlet_nodes":  p_outlet.tolist(),
            "p_inlet_mean":    p_inlet_mean,
            "p_outlet_mean":   p_outlet_mean,

            # velocity at terminals
            "u_inlet_nodes":   u_inlet.tolist(),
            "u_outlet_nodes":  u_outlet.tolist(),

            # q = cell_vol * div(u) at terminals
            "q_inlet":         q_inlet.tolist(),
            "q_outlet":        q_outlet.tolist(),
            # Inward-positive convention (often easier to compare to 1D inflow targets)
            "q_inlet_inward":  (-q_inlet).tolist(),
            "divu_inlet":      divu_in.tolist(),
            "divu_outlet":     divu_out.tolist(),
            "cell_volume":     cell_vol,

            # total fluxes across inlet / outlet facet markers
            "q_artery_leak":   float(Q_art_leak),
            "q_venous_leak":   float(Q_ven_leak),
            "p_concave_inlet_bc": float(self.p_in_BC),
            "p_concave_outlet_bc": float(self.p_out_BC),
            "p_concave_inlet_bc_source": self.concave_inlet_pressure_source,
            "p_concave_outlet_bc_source": self.concave_outlet_pressure_source,
            "concave_pressure_profile": self.concave_pressure_profile,
            "concave_pressure_profile_axis": self.concave_pressure_profile_axis_name,
            "p_concave_inlet_bc_low": float(self.concave_art_pressure_low),
            "p_concave_inlet_bc_high": float(self.concave_art_pressure_high),
            "p_concave_outlet_bc_low": float(self.concave_ven_pressure_low),
            "p_concave_outlet_bc_high": float(self.concave_ven_pressure_high),
            "skip_1d": bool(self.skip_1d),

            # coords for mapping back to 1D (now in branchingData order if provided)
            "coords_inlet":    coords_in.tolist(),
            "coords_outlet":   coords_out.tolist(),
        }

        # Also store imposed inlet terminal flow targets (mapped to output order)
        # so mismatches can be diagnosed without ambiguity.
        if idx_in_branch is not None:
            p_inlet_target = p_inlet_target_marker[idx_in_branch]
        else:
            p_inlet_target = p_inlet_target_marker
        interface_bc["p_inlet_target"] = p_inlet_target.tolist()

        if idx_out_branch is not None:
            p_outlet_target = p_outlet_target_marker[idx_out_branch]
        else:
            p_outlet_target = p_outlet_target_marker
        interface_bc["p_outlet_target"] = p_outlet_target.tolist()

        if idx_in_branch is not None:
            q_inlet_target = q_inlet_target_marker[idx_in_branch]
        else:
            q_inlet_target = q_inlet_target_marker
        interface_bc["q_inlet_target"] = q_inlet_target.tolist()
        if len(q_inlet_target) == len(q_inlet):
            interface_bc["q_inlet_residual"] = (q_inlet - q_inlet_target).tolist()
        if hasattr(self, "inlet_marker_area_3d"):
            a3d = np.asarray(self.inlet_marker_area_3d, dtype=float)
            interface_bc["inlet_area_3d_marker"] = a3d.tolist()
        if hasattr(self, "inlet_marker_area_1d"):
            a1d = np.asarray(self.inlet_marker_area_1d, dtype=float)
            interface_bc["inlet_area_1d_checkpoint"] = a1d.tolist()
            if "inlet_area_3d_marker" in interface_bc and len(a1d) == len(interface_bc["inlet_area_3d_marker"]):
                a3d = np.asarray(interface_bc["inlet_area_3d_marker"], dtype=float)
                ratio = a3d / np.maximum(a1d, 1e-20)
                interface_bc["inlet_area_ratio_3d_over_1d"] = ratio.tolist()

        # optionally store branch IDs for sanity checking
        if ids_in is not None:
            interface_bc["branch_ids_inlet"] = ids_in.tolist()
        if ids_out is not None:
            interface_bc["branch_ids_outlet"] = ids_out.tolist()

        return interface_bc

    # -----------------------------------------------------------------------------
    # Permeability field K(x) in DG0 (same as your fixed version)
    # -----------------------------------------------------------------------------
    @staticmethod
    def _assign_dg0_cell_values(fun, cell_values):
        """Assign one scalar DG0 value per owned cell using the cell-to-dof map."""
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
        """Return physical coordinate dofs for a cell using the geometry dofmap."""
        if gdim is None:
            gdim = mesh_obj.geometry.dim
        gdofs = mesh_obj.geometry.dofmap[int(cell)]
        return mesh_obj.geometry.x[np.asarray(gdofs, dtype=np.int32), : int(gdim)]

    @staticmethod
    def _entity_geometry_points(mesh_obj, dim: int, entity: int, gdim: int | None = None):
        """Return physical coordinate dofs for a mesh entity using geometry mapping."""
        if gdim is None:
            gdim = mesh_obj.geometry.dim
        mesh_obj.topology.create_entities(int(dim))
        mesh_obj.topology.create_connectivity(int(dim), mesh_obj.topology.dim)
        gdofs = dfx.mesh.entities_to_geometry(
            mesh_obj,
            int(dim),
            np.asarray([int(entity)], dtype=np.int32),
            False,
        )[0]
        return mesh_obj.geometry.x[np.asarray(gdofs, dtype=np.int32), : int(gdim)]

    def make_K_DG0(self, mesh, K_base, K_high, centers, radius, inner_frac=0.5):
        """
        Build a DG0 permeability field with smooth transition from K_high near
        'centers' to K_base outside 'radius'.

        inner_frac : fraction of 'radius' that is pure high-K core.
                    e.g. 0.5 => core radius = 0.5 * radius, smooth ramp from
                    0.5*radius to radius.
        """
        V0 = fem.functionspace(mesh, ("DG", 0))
        Kfun = fem.Function(V0)

        # local cells only
        tdim = mesh.topology.dim
        nloc = mesh.topology.index_map(tdim).size_local
        cell_ids = np.arange(nloc, dtype=np.int32)

        # cell midpoints
        mids = np.zeros((nloc, mesh.geometry.dim), dtype=np.float64)
        for c in cell_ids:
            mids[c, :] = np.mean(self._cell_geometry_points(mesh, int(c)), axis=0)

        # base field
        alpha = np.zeros(nloc, dtype=np.float64)  # blending: 0 -> K_base, 1 -> K_high

        r_inner = inner_frac * radius
        r_outer = radius
        dr = r_outer - r_inner
        if dr <= 0:
            raise ValueError("inner_frac * radius must be < radius (inner_frac < 1).")

        centers = np.asarray(centers, dtype=np.float64)
        if centers.ndim == 1:
            centers = centers[None, :]

        for c0 in centers:
            d = np.linalg.norm(mids - c0[None, :], axis=1)

            # per-center blending
            alpha_c = np.zeros_like(d)

            # core region: fully high-K
            mask_inner = d <= r_inner
            alpha_c[mask_inner] = 1.0

            # transition shell: cosine ramp from 1 -> 0
            mask_shell = (d > r_inner) & (d < r_outer)
            s = (d[mask_shell] - r_inner) / dr  # in (0,1)
            alpha_c[mask_shell] = 0.5 * (1.0 + np.cos(np.pi * s))  # smooth C^1

            # combine contributions from multiple centers (take max)
            alpha = np.maximum(alpha, alpha_c)

        # final K field
        K_local = K_base + alpha * (K_high - K_base)

        self._assign_dg0_cell_values(Kfun, K_local)
        return Kfun

    from dolfinx import fem, mesh as dmesh
    import ufl
    import numpy as np
    from scipy.spatial import cKDTree

    def make_K_near_wall_DG0(
        self,
        mesh,
        facet_tags,
        wall_markers,
        K_base,
        K_wall,
        d_inner,
        d_outer,
    ):
        """
        Build a DG0 permeability field K(x) with higher permeability in a layer
        of thickness ~d_outer next to a set of boundary facets.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            Your 3D Darcy mesh.
        facet_tags : dolfinx.mesh.meshtags
            Facet meshtags for this mesh.
        wall_markers : int or sequence of int
            Markers of the boundary facets where you want the high-K layer
            (e.g. [INLET_MARK, OUTLET_MARK]).
        K_base : float
            Permeability in the bulk (away from wall).
        K_wall : float
            Permeability at the wall (d <= d_inner).
        d_inner : float
            Inner thickness for full K_wall (distance from wall where K = K_wall).
        d_outer : float
            Outer thickness where K smoothly transitions back to K_base.
            For d >= d_outer, K = K_base.
        """

        # Make sure wall_markers is an array
        wall_markers = np.atleast_1d(np.array(wall_markers, dtype=np.int32))

        V0 = fem.functionspace(mesh, ("DG", 0))
        Kfun = fem.Function(V0)

        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim
        fdim = tdim - 1

        mask = np.isin(facet_tags.values, wall_markers)
        wall_facets = facet_tags.indices[mask]

        wall_centers = np.zeros((len(wall_facets), gdim), dtype=np.float64)
        for i, f in enumerate(wall_facets):
            wall_centers[i, :] = self._entity_geometry_points(mesh, fdim, int(f), gdim).mean(axis=0)

        # Build KD-tree for distance queries
        tree = cKDTree(wall_centers)

        # --- 2) Cell midpoints ---
        nloc = mesh.topology.index_map(tdim).size_local
        cell_ids = np.arange(nloc, dtype=np.int32)

        mids = np.zeros((nloc, gdim), dtype=np.float64)
        for c in cell_ids:
            mids[c, :] = self._cell_geometry_points(mesh, int(c), gdim).mean(axis=0)

        # --- 3) Distance of each cell midpoint to nearest wall facet ---
        d, _ = tree.query(mids)  # d.shape = (nloc,)

        # --- 4) Smooth blending based on distance ---
        alpha = np.zeros_like(d)  # 0 -> K_base, 1 -> K_wall

        # region very close to wall: pure K_wall
        mask_inner = d <= d_inner
        alpha[mask_inner] = 1.0

        # transition layer: cosine ramp from 1 -> 0 between d_inner and d_outer
        mask_shell = (d > d_inner) & (d < d_outer)
        s = (d[mask_shell] - d_inner) / (d_outer - d_inner)  # in (0,1)
        alpha[mask_shell] = 0.5 * (1.0 + np.cos(np.pi * s))

        # outside layer: alpha = 0 already (K_base)

        # --- 5) Final K field ---
        K_local = alpha * (K_wall - K_base)

        # Assign to DG0 function
        self._assign_dg0_cell_values(Kfun, K_local)
        return Kfun

    # ========================================================
    # Solve Darcy
    # ========================================================
    def setup(self):

        mesh = self.mesh
        facet_tags = self.facet_tags
        P_el = element("Lagrange", mesh.basix_cell(), 1)
        W = fem.functionspace(mesh, P_el)

        # Darcy coefficients
        # kappa = 1e-5
        # mu    = 1.0
        # kappa_over_mu = fem.Constant(mesh, dfx.default_scalar_type(kappa/mu))
        # phi = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        # conductivity
    

        # weak form
        v = ufl.TestFunction(W)
        p = ufl.TrialFunction(W)

        # ---------------------------
        # K(x)
        # ---------------------------
        K_base = 1e-7  # bulk gel permeability
        K_wall = 1e-7    # higher permeability near wall
        K_high = 1.0e-8    # high-K blobs (e.g. organoids)
        d_inner = 0.01     # 2% of domain-size thickness of full high-K
        d_outer = 0.025    # smooth transition back to K_base by 5%
        r = 0.1
        centers = np.array([
            [0,  0.900,  0.4359],
            [0,  0.300,  0.4359],
            [0, -0.300,  0.4359],
            [0, -0.900, 0.4359],
        ], dtype=np.float64)
        INLET_MARK = getattr(self, "arterial_concave_marker", 32)
        OUTLET_MARK = getattr(self, "venous_concave_marker", 31)

        K_wall_blobs = self.make_K_near_wall_DG0(
            mesh,
            facet_tags,          # your Meshtags object
            wall_markers=[INLET_MARK, OUTLET_MARK],
            K_base=K_base,
            K_wall=K_wall,
            d_inner=d_inner,
            d_outer=d_outer,
        )
        Kfun = K_wall_blobs  # start with wall blobs, then combine with organoid blobs if desired
        K_blobs      = self.make_K_DG0(mesh, K_base, K_high, centers, r, inner_frac=0.5)

        # Combine: higher of the two at each cell
        Kfun = fem.Function(K_wall_blobs.function_space)
        Kfun.x.array[:] = K_wall_blobs.x.array[:] + K_blobs.x.array[:] 
        Kfun.x.scatter_forward()

        # Kfun is DG0 scalar, representing "magnitude" of permeability
        K_par_scale  = 1.0  # e.g. 5× more permeable along x than transverse
        K_perp_scale = 1.0

        K_par  = K_par_scale  * Kfun
        K_perp = K_perp_scale * Kfun

        I = ufl.Identity(mesh.geometry.dim)
        e = ufl.as_vector((1.0, 0.0, 0.0))  # unit vector in x-direction
        K_tensor = K_perp * I + (K_par - K_perp) * ufl.outer(e, e)
        self.K_tensor = K_tensor
        a = ufl.inner((K_tensor) * ufl.grad(p), ufl.grad(v)) * ufl.dx
        if self.concave_bc_mode == "robin":
            a_robin, _ = self._build_concave_robin_terms(p, v)
            a += a_robin

        # a = kappa_over_mu * dot(grad(p), grad(v)) * ufl.dx
        # a = ufl.inner(Kfun * ufl.grad(p), ufl.grad(v)) * ufl.dx

        # RHS from terminal flows
        q_src = self._build_q_src(W)
        dx = ufl.dx(self.mesh)
        vol = fem.assemble_scalar(fem.form(fem.Constant(self.mesh, 1.0) * dx))
        Q_tot = fem.assemble_scalar(fem.form(q_src * dx))

        if self.mesh.comm.rank == 0:
            print("Total ∫ q_src dx before correction =", Q_tot)

        # # subtract the mean to enforce ∫ q_src dx = 0
        # q_src_mean = Q_tot / vol
        # q_src.x.array[:] -= q_src_mean

        # Q_tot_new = fem.assemble_scalar(fem.form(q_src * dx))
        # if self.mesh.comm.rank == 0:
        #     print("Total ∫ q_src dx after correction  =", Q_tot_new)

        # # q_src.x.array[:] *= 1e-3   # sign convention
        # if mesh.comm.rank == 0:
        #     print("q_src  min/max:",  q_src.x.array.min(),  q_src.x.array.max())

        # L = q_src * v * ufl.dx + self._build_terminal_flux_form(v)
        # if self.concave_bc_mode == "robin":
        #     _, L_robin = self._build_concave_robin_terms(p, v)
        #     L += L_robin

        L = q_src * v * ufl.dx 
        # Dirichlet BCs:
        # - Terminal pressures (arterial + venous) from 1D trees.
        # - Concave inlet/outlet only in dirichlet mode.
        bcs = self._build_terminal_bcs(W)
        if self.concave_bc_mode == "dirichlet":
            bcs = self._build_wall_dirichlet_bcs(W) + bcs

        problem = _make_linear_problem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="darcy_pressure_solve_primary_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            },
        )

        p_h = problem.solve()


        # beta, rhs_beta = self._build_point_pressure_penalty(W, beta_value=1e3)

        # a = a + beta * p * v * ufl.dx
        # L = L + rhs_beta * v * ufl.dx

        # mesh size
        # X = W.tabulate_dof_coordinates()
        # tree = cKDTree(X)
        # dists, _ = tree.query(self.x_outlet, k=10)  # distances to 10 nearest dofs
        # h = np.median(dists[:, -1])               # k-th nearest distance
        # print("Estimated mesh h near outlets:", h)
        # r = 3.0 * h  # smear radius


        # r = 0.005  # fixed radius for testing
        # gamma = 1.0
        # K = 1e-7
        # beta0 = gamma * K / (r*r)

        # beta_outlet, beta_p0_outlet = self.build_beta_fields(W, self.x_outlet, self.p_outlet, K, r, gamma)
        # beta_inlet,  beta_p0_inlet  = self.build_beta_fields(W, self.x_inlet,  self.p_inlet,  K, r, gamma)
        # a = ufl.inner(Kfun * ufl.grad(p), ufl.grad(v)) * ufl.dx + beta_outlet * p * v * ufl.dx # + beta_inlet * p * v * ufl.dx
        # L = q_src * v * ufl.dx + beta_p0_outlet * v * ufl.dx # + beta_p0_inlet * v * ufl.dx

        # BCs from terminal pressures
        # bcs = self._build_terminal_bcs(W)
        a_form = fem.form(a)
        L_form = fem.form(L)
        problem = _make_linear_problem(
            a,
            L,
            bcs=bcs,
            petsc_options_prefix="darcy_pressure_solve_secondary_",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            },
        )

        p_h = problem.solve()


        # petsc_options = {
        #     "ksp_error_if_not_converged": True,
        #     "ksp_type": "preonly",
        #     "pc_type": "lu",
        #     "pc_factor_mat_solver_type": "mumps",
        #     "ksp_monitor": None,
        # }

        # ksp = PETSc.KSP().create(MPI.COMM_WORLD)
        # ksp.setOptionsPrefix("singular_direct")
        # opts = PETSc.Options()
        # opts.prefixPush(ksp.getOptionsPrefix())
        # for key, value in petsc_options.items():
        #     opts[key] = value
        # ksp.setFromOptions()
        # for key, value in petsc_options.items():
        #     del opts[key]
        # opts.prefixPop()

        # A = fem.petsc.assemble_matrix(a_form, bcs=bcs)
        # A.assemble()
        # b = fem.petsc.assemble_vector(L_form)
        # b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        # ksp.setOperators(A)

        # # build constant nullspace
        # nullspace = PETSc.NullSpace().create(constant=True, comm=MPI.COMM_WORLD)
        # # assert nullspace.test(A)
        # A.setNullSpace(nullspace)


        # ksp = PETSc.KSP().create(self.mesh.comm)
        # ksp.setOperators(A)
        # ksp.setType("preonly")
        # ksp.getPC().setType("lu")
        # ksp.getPC().setFactorSolverType("mumps")

        # p_h = fem.Function(W)
        # ksp.solve(b, p_h.x.petsc_vec)
        # p_h.x.scatter_forward()

        # p_h = problem.solve()

        # # Post-process gauge using 1D outlet pressures
        # self._apply_pressure_gauge_postsolve(p_h)

        # Project Darcy flux to an H(div)-conforming space so normal fluxes are
        # single-valued across facets.
        V = fem.functionspace(mesh, element("RT", mesh.basix_cell(), 1))
        projector = Projector(V)
        # Use the same anisotropic conductivity tensor as in the weak form
        # so postprocessed fluxes match the solved PDE.
        u_h = projector(-(K_tensor * grad(p_h)))

         # --- total leak flux across inlet / outlet facet markers ---
        # facet_tags: 1 = inlet surface, 2 = outlet surface
        n  = ufl.FacetNormal(mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)

        INLET_MARK  = getattr(self, "arterial_concave_marker", 31)  # arterial side (concave)
        OUTLET_MARK = getattr(self, "venous_concave_marker", 32)  # venous side (concave)

        # Flux = ∫_Γ (u · n) dS
        Q_art_leak = fem.assemble_scalar(
            fem.form(-ufl.dot(K_tensor * ufl.grad(p_h), n) * ds(INLET_MARK))
        )
        Q_ven_leak = fem.assemble_scalar(
            fem.form(-ufl.dot(K_tensor * ufl.grad(p_h), n) * ds(OUTLET_MARK))
        )

        # Make sure we have global (MPI-reduced) values
        Q_art_leak = mesh.comm.allreduce(Q_art_leak, op=MPI.SUM)
        Q_ven_leak = mesh.comm.allreduce(Q_ven_leak, op=MPI.SUM)

        # Write outputs
        out_dir = current_dir/"out_darcy"
        out_dir.mkdir(exist_ok=True)

        with io.XDMFFile(mesh.comm, out_dir/"p.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(p_h)

        with io.XDMFFile(mesh.comm, out_dir/"u.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(u_h)

        vtkfile = VTKFile(MPI.COMM_WORLD,  out_dir/"u.vtu", "w")
        P1 = fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1,
                                shape=(mesh.geometry.dim,)))
        u_P1= fem.Function(P1)
        u_P1.interpolate(u_h)
        vtkfile.write_function(u_P1)

        vtkfile_p = VTKFile(MPI.COMM_WORLD,  out_dir/"p.vtu", "w")
        vtkfile_p.write_function(p_h)

        # --- compute interface BCs for the 0D solver ---
        interface_bc = self._compute_interface_bc(p_h, u_h, q_src, Q_art_leak, Q_ven_leak)

        # write to JSON for a separate script
        if mesh.comm.rank == 0:
            with open(out_dir / "interface_bc.json", "w") as fp:
                json.dump(interface_bc, fp, indent=2)

        print("Darcy perfusion solve complete.")

        # Return interface_bc so a caller in the same Python process can use it
        return interface_bc

# ===================================================================
# Projector class
# ===================================================================
class Projector:
    def __init__(self, V):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        a = ufl.inner(u,v)*ufl.dx
        self.V = V
        self.u = fem.Function(V)
        self.a_cpp = fem.form(a)

    def __call__(self, f):
        v = ufl.TestFunction(self.V)
        L = ufl.inner(f,v)*ufl.dx

        A = fem.petsc.assemble_matrix(self.a_cpp, bcs=[])
        A.assemble()
        b = fem.petsc.assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES,
                      mode=PETSc.ScatterMode.REVERSE)

        solver = PETSc.KSP().create(MPI.COMM_WORLD)
        solver.setOperators(A)
        solver.setType("preonly")
        solver.getPC().setType("lu")
        solver.getPC().setFactorSolverType("mumps")

        solver.solve(b, self.u.x.petsc_vec)

        return self.u


# ===================================================================
# RUN
# ===================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run Darcy perfusion solve on prepared mesh/checkpoint files.")
    ap.add_argument("--bioreactor-domain", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/bioreactor_well1.xdmf")
    ap.add_argument("--facet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/mesh_tags_well1.xdmf")
    ap.add_argument("--mesh-inlet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/tagged_branches_inlet.bp")
    ap.add_argument("--mesh-outlet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/tagged_branches_outlet.bp")
    ap.add_argument("--pres-inlet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/pressure_checkpoint_inlet.bp")
    ap.add_argument("--pres-outlet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/pressure_checkpoint_outlet.bp")
    ap.add_argument("--flow-inlet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/flow_checkpoint_inlet.bp")
    ap.add_argument("--flow-outlet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/flow_checkpoint_outlet.bp")
    ap.add_argument("--area-inlet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/area_checkpoint_inlet.bp")
    ap.add_argument("--area-outlet-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/area_checkpoint_outlet.bp")
    ap.add_argument("--branching-in-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022526/Run8_10branches/branchingData_0.csv")
    ap.add_argument("--branching-out-file", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022526/Run8_10branches/branchingData_1.csv")
    ap.add_argument("--inlet-output-csv", default="", help="0D inlet-tree output.csv used to read terminal parent pressures.")
    ap.add_argument("--outlet-output-csv", default="", help="0D outlet-tree output.csv used to read terminal parent pressures.")
    ap.add_argument("--skip-1d", action="store_true", help="Skip inlet/outlet 1D tree loading and use fallback concave pressures.")
    ap.add_argument("--fallback-inlet-pressure", type=float, default=0.0)
    ap.add_argument("--fallback-outlet-pressure", type=float, default=0.0)
    ap.add_argument(
        "--concave-inlet-pressure",
        type=float,
        default=None,
        help="Optional explicit arterial concave-face pressure. Keeps 1D terminal loading enabled.",
    )
    ap.add_argument(
        "--concave-outlet-pressure",
        type=float,
        default=None,
        help="Optional explicit venous concave-face pressure. Keeps 1D terminal loading enabled.",
    )
    ap.add_argument("--checkpoint-time-index", type=int, default=-1,
                    help="Checkpoint time index to read from 1D pressure/flow/area histories (-1 = last).")
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-0.175, 0.9, 0.55])
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.175, 0.9, 0.55])
    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet",
                    help="Boundary model on concave arterial/venous faces.")
    ap.add_argument("--concave-pressure-profile", choices=["constant", "linear"], default="constant",
                    help="Pressure profile on concave arterial/venous faces.")
    ap.add_argument("--concave-pressure-profile-axis", choices=["x", "y", "z"], default="y",
                    help="Axis used for linear concave pressure profiles.")
    ap.add_argument("--arterial-pressure-profile-low", type=float, default=None,
                    help="Arterial concave pressure at the low-coordinate end of the profile axis.")
    ap.add_argument("--arterial-pressure-profile-high", type=float, default=None,
                    help="Arterial concave pressure at the high-coordinate end of the profile axis.")
    ap.add_argument("--venous-pressure-profile-low", type=float, default=None,
                    help="Venous concave pressure at the low-coordinate end of the profile axis.")
    ap.add_argument("--venous-pressure-profile-high", type=float, default=None,
                    help="Venous concave pressure at the high-coordinate end of the profile axis.")
    ap.add_argument("--lp-arterial", type=float, default=0.0,
                    help="Robin transfer coefficient on arterial concave face (used when --concave-bc-mode robin).")
    ap.add_argument("--lp-venous", type=float, default=0.0,
                    help="Robin transfer coefficient on venous concave face (used when --concave-bc-mode robin).")
    ap.add_argument("--inlet-flux-correction", action="store_true", default=False,
                    help="Enable iterative per-terminal inlet flux correction loop.")
    ap.add_argument("--inlet-flux-corr-max-iter", type=int, default=5,
                    help="Maximum correction iterations when --inlet-flux-correction is enabled.")
    ap.add_argument("--inlet-flux-corr-relax", type=float, default=0.5,
                    help="Relaxation factor in [0,1] for inlet flux correction updates.")
    ap.add_argument("--inlet-flux-corr-tol", type=float, default=0.05,
                    help="Stop when max relative inlet flow error <= tol.")
    ap.add_argument("--out-dir", default="", help="Optional output directory for out_darcy")
    args = ap.parse_args()

    if args.out_dir:
        current_dir = Path(args.out_dir).expanduser().resolve()

    solver = PerfusionSolver(
        args.bioreactor_domain,
        args.facet_file,
        args.mesh_inlet_file,
        args.mesh_outlet_file,
        args.pres_inlet_file,
        args.pres_outlet_file,
        args.flow_inlet_file,
        args.flow_outlet_file,
        np.array(args.coords_inlet, dtype=float),
        np.array(args.coords_outlet, dtype=float),
        area_inlet_file=args.area_inlet_file if args.area_inlet_file else None,
        area_outlet_file=args.area_outlet_file if args.area_outlet_file else None,
        concave_bc_mode=args.concave_bc_mode,
        concave_pressure_profile=args.concave_pressure_profile,
        concave_pressure_profile_axis=args.concave_pressure_profile_axis,
        arterial_pressure_profile_low=args.arterial_pressure_profile_low,
        arterial_pressure_profile_high=args.arterial_pressure_profile_high,
        venous_pressure_profile_low=args.venous_pressure_profile_low,
        venous_pressure_profile_high=args.venous_pressure_profile_high,
        lp_arterial=args.lp_arterial,
        lp_venous=args.lp_venous,
        skip_1d=args.skip_1d,
        fallback_inlet_pressure=args.fallback_inlet_pressure,
        fallback_outlet_pressure=args.fallback_outlet_pressure,
        checkpoint_time_index=args.checkpoint_time_index,
        inlet_flux_correction=args.inlet_flux_correction,
        inlet_flux_corr_max_iter=args.inlet_flux_corr_max_iter,
        inlet_flux_corr_relax=args.inlet_flux_corr_relax,
        inlet_flux_corr_tol=args.inlet_flux_corr_tol,
        branching_in_file=args.branching_in_file if args.branching_in_file else None,
        branching_out_file=args.branching_out_file if args.branching_out_file else None,
        inlet_output_csv=args.inlet_output_csv if args.inlet_output_csv else None,
        outlet_output_csv=args.outlet_output_csv if args.outlet_output_csv else None,
        concave_inlet_pressure=args.concave_inlet_pressure,
        concave_outlet_pressure=args.concave_outlet_pressure,
    )
    solver.setup()
 
