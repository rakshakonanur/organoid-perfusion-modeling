#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import adios4dolfinx
import dolfinx as dfx
import numpy as np
import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, io
from dolfinx.fem.petsc import apply_lifting, assemble_matrix, assemble_vector, set_bc
from mpi4py import MPI
from petsc4py import PETSc
from scipy.spatial import cKDTree

import darcy as darcy_cg
import darcy_mixed
from darcy import import_3d_mesh_with_facets, import_mesh
from darcy_mixed import PerfusionSolver as MixedPerfusionSolver

current_dir = Path(
    "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/solves"
)


def read_pressure_entry(mesh_obj, bp_file: str, entry_index: int, comm=MPI.COMM_WORLD):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=comm, function_name="f")
    if len(ts) == 0:
        raise ValueError(f"No pressure checkpoint entries found in {bp_file}")
    idx = int(entry_index)
    if idx < 0:
        idx += len(ts)
    if idx < 0 or idx >= len(ts):
        raise IndexError(f"entry_index={entry_index} out of range for {bp_file} ({len(ts)} entries)")
    P1 = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    p_src = fem.Function(P1)
    adios4dolfinx.read_function(bp_file, p_src, name="f", time=ts[idx])
    p_src.x.scatter_forward()
    p_src.x.array[:] *= 1333.22
    return p_src


def resolve_entry_index(entry_index: int, n_entries: int) -> int:
    idx = int(entry_index)
    if idx < 0:
        idx += int(n_entries)
    if idx < 0 or idx >= int(n_entries):
        raise IndexError(f"entry_index={entry_index} out of range ({n_entries} entries)")
    return idx


def read_scalar_entry(mesh_obj, bp_file: str, entry_index: int, label: str, comm=MPI.COMM_WORLD):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=comm, function_name="f")
    if len(ts) == 0:
        raise ValueError(f"No {label} checkpoint entries found in {bp_file}")
    idx = int(entry_index)
    if idx < 0:
        idx += len(ts)
    if idx < 0 or idx >= len(ts):
        raise IndexError(f"entry_index={entry_index} out of range for {bp_file} ({len(ts)} entries)")
    P1 = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    f_src = fem.Function(P1)
    adios4dolfinx.read_function(bp_file, f_src, name="f", time=ts[idx])
    f_src.x.scatter_forward()
    return f_src


def read_selected_scalar_entries(
    mesh_obj,
    bp_file: str,
    entry_indices: Sequence[int],
    label: str,
    *,
    pressure_scale: float = 1.0,
    comm=MPI.COMM_WORLD,
) -> Dict[int, fem.Function]:
    ts = adios4dolfinx.read_timestamps(bp_file, comm=comm, function_name="f")
    if len(ts) == 0:
        raise ValueError(f"No {label} checkpoint entries found in {bp_file}")
    V = fem.functionspace(mesh_obj, element("Lagrange", mesh_obj.basix_cell(), 1))
    f_src = fem.Function(V)
    out: Dict[int, fem.Function] = {}
    for entry_index in entry_indices:
        idx = resolve_entry_index(int(entry_index), len(ts))
        adios4dolfinx.read_function(bp_file, f_src, name="f", time=ts[idx])
        f_src.x.scatter_forward()
        if pressure_scale != 1.0:
            f_src.x.array[:] *= float(pressure_scale)
        out[int(entry_index)] = f_src.copy()
    return out


def resolve_selected_indices(indices: Sequence[int]) -> List[int]:
    out = [int(i) for i in indices]
    if not out:
        raise ValueError("At least one checkpoint index must be provided")
    return out


def extract_terminal_coords(mesh_obj, tags):
    marker = 2
    idx = np.where(tags.values == marker)[0]
    if len(idx) == 0:
        return np.zeros((0, mesh_obj.geometry.dim), dtype=np.float64)
    entity_ids = tags.indices[idx]
    return np.asarray(mesh_obj.geometry.x[entity_ids], dtype=np.float64)


def nearest_dof_indices(V, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0,), dtype=int)
    dof_coords = V.tabulate_dof_coordinates()
    tree = cKDTree(dof_coords)
    _, nn = tree.query(np.asarray(pts, dtype=float))
    return np.asarray(nn, dtype=int)


def nearest_single_dof(V, pt: np.ndarray) -> int:
    dof_coords = V.tabulate_dof_coordinates()
    tree = cKDTree(dof_coords)
    _, nn = tree.query(np.asarray(pt, dtype=float).reshape(1, -1))
    return int(nn[0])


def sample_scalar_at_dofs(func: fem.Function, dofs: np.ndarray) -> np.ndarray:
    if dofs.size == 0:
        return np.zeros((0,), dtype=float)
    return np.asarray(func.x.array[np.asarray(dofs, dtype=int)], dtype=float)


def read_pressure_for_time(output_csv: Path, vessel_name: str, time_value: float) -> Optional[float]:
    best_diff = float("inf")
    best_val: Optional[float] = None
    with output_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("name", "")).strip() != vessel_name:
                continue
            try:
                t = float(row.get("time", 0.0))
            except Exception:
                continue
            diff = abs(t - float(time_value))
            if diff > best_diff:
                continue
            raw = row.get("pressure_out", None)
            if raw in (None, ""):
                continue
            try:
                best_val = float(raw)
                best_diff = diff
            except Exception:
                continue
    return best_val


class CachedProjector:
    def __init__(self, V):
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        self.V = V
        self.u = fem.Function(V)
        self.a_form = fem.form(ufl.inner(u, v) * ufl.dx)
        self.A = assemble_matrix(self.a_form, bcs=[])
        self.A.assemble()
        self.ksp = PETSc.KSP().create(V.mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType("mumps")

    def __call__(self, f):
        v = ufl.TestFunction(self.V)
        L = ufl.inner(f, v) * ufl.dx
        b = assemble_vector(fem.form(L))
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        self.ksp.solve(b, self.u.x.petsc_vec)
        self.u.x.scatter_forward()
        return self.u


class TimeSeriesPerfusionSolver(MixedPerfusionSolver):
    def __init__(
        self,
        bioreactor_domain,
        facet_file,
        mesh_inlet_file,
        mesh_outlet_file,
        pres_inlet_file,
        pres_outlet_file,
        flow_inlet_file,
        flow_outlet_file,
        inlet_coord,
        outlet_coord,
        checkpoint_indices: Sequence[int],
        checkpoint_times: Sequence[float],
        area_inlet_file=None,
        area_outlet_file=None,
        concave_bc_mode: str = "dirichlet",
        lp_arterial: float = 0.0,
        lp_venous: float = 0.0,
        skip_1d: bool = False,
        fallback_inlet_pressure: float = 0.0,
        fallback_outlet_pressure: float = 0.0,
        inlet_flux_correction: bool = False,
        inlet_flux_corr_max_iter: int = 5,
        inlet_flux_corr_relax: float = 0.5,
        inlet_flux_corr_tol: float = 0.05,
        perm_region_path: str = "",
        perm_low: float = 1.0e-8,
        perm_high: float = 1.0e-6,
        perm_transition_width: float = 0.01,
        branching_in_file=None,
        branching_out_file=None,
        leak_pressure_output_csv: Optional[Path] = None,
        organoid_index: int = 0,
    ):
        self.perm_region_path = str(perm_region_path).strip()
        self.perm_low = float(perm_low)
        self.perm_high = float(perm_high)
        self.perm_transition_width = float(perm_transition_width)

        if MPI.COMM_WORLD.rank == 0:
            print("About to open XDMF mesh", flush=True)
        MPI.COMM_WORLD.barrier()
        self.mesh, self.facet_tags = import_3d_mesh_with_facets(bioreactor_domain, facet_file)
        if self.mesh.comm.rank == 0:
            print("[mesh] 3D mesh/facet import complete", flush=True)

        self.arterial_concave_marker = 31
        self.venous_concave_marker = 32
        self.inlet_base_marker = 1000
        self.outlet_base_marker = 2000
        self.skip_1d = bool(skip_1d)
        self.fallback_inlet_pressure = float(fallback_inlet_pressure)
        self.fallback_outlet_pressure = float(fallback_outlet_pressure)
        self.checkpoint_indices = resolve_selected_indices(checkpoint_indices)
        self.checkpoint_time_index = int(self.checkpoint_indices[0])
        self.checkpoint_times = np.asarray(checkpoint_times, dtype=float)
        if self.checkpoint_times.size not in {0, len(self.checkpoint_indices)}:
            raise ValueError("checkpoint_times must be empty or match checkpoint_indices in length")
        if self.checkpoint_times.size == 0:
            self.checkpoint_times = np.asarray(self.checkpoint_indices, dtype=float)
        self.leak_pressure_output_csv = Path(leak_pressure_output_csv).expanduser().resolve() if leak_pressure_output_csv else None
        self.organoid_index = int(organoid_index)

        comm = self.mesh.comm
        self._timeseries_samples: Dict[int, dict] = {}

        if comm.rank == 0:
            if self.skip_1d:
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

                x_inlet = np.ascontiguousarray(extract_terminal_coords(self.inlet_mesh, self.inlet_tags), dtype=np.float64)
                x_outlet = np.ascontiguousarray(extract_terminal_coords(self.outlet_mesh, self.outlet_tags), dtype=np.float64)
                n_in = int(x_inlet.shape[0])
                n_out = int(x_outlet.shape[0])

                P1_in = fem.functionspace(self.inlet_mesh, element("Lagrange", self.inlet_mesh.basix_cell(), 1))
                P1_out = fem.functionspace(self.outlet_mesh, element("Lagrange", self.outlet_mesh.basix_cell(), 1))
                dofs_in = nearest_dof_indices(P1_in, x_inlet)
                dofs_out = nearest_dof_indices(P1_out, x_outlet)
                bc_in_dof = nearest_single_dof(P1_in, np.asarray(inlet_coord, dtype=float))
                bc_out_dof = nearest_single_dof(P1_out, np.asarray(outlet_coord, dtype=float))

                print("[1d] Rank 0 reading only selected checkpoint entries", flush=True)
                p_A_series = read_selected_scalar_entries(
                    self.inlet_mesh,
                    pres_inlet_file,
                    self.checkpoint_indices,
                    "inlet pressure",
                    pressure_scale=1333.22,
                    comm=MPI.COMM_SELF,
                )
                p_V_series = read_selected_scalar_entries(
                    self.outlet_mesh,
                    pres_outlet_file,
                    self.checkpoint_indices,
                    "outlet pressure",
                    pressure_scale=1333.22,
                    comm=MPI.COMM_SELF,
                )
                q_A_series = read_selected_scalar_entries(
                    self.inlet_mesh,
                    flow_inlet_file,
                    self.checkpoint_indices,
                    "inlet flow",
                    comm=MPI.COMM_SELF,
                )
                q_V_series = read_selected_scalar_entries(
                    self.outlet_mesh,
                    flow_outlet_file,
                    self.checkpoint_indices,
                    "outlet flow",
                    comm=MPI.COMM_SELF,
                )
                a_A_series = None
                a_V_series = None
                if area_inlet_file is not None and Path(area_inlet_file).exists():
                    a_A_series = read_selected_scalar_entries(
                        self.inlet_mesh,
                        area_inlet_file,
                        self.checkpoint_indices,
                        "inlet area",
                        comm=MPI.COMM_SELF,
                    )
                if area_outlet_file is not None and Path(area_outlet_file).exists():
                    a_V_series = read_selected_scalar_entries(
                        self.outlet_mesh,
                        area_outlet_file,
                        self.checkpoint_indices,
                        "outlet area",
                        comm=MPI.COMM_SELF,
                    )

                for hist_idx in self.checkpoint_indices:
                    p_A_k = p_A_series[int(hist_idx)]
                    p_V_k = p_V_series[int(hist_idx)]
                    q_A_k = q_A_series[int(hist_idx)]
                    q_V_k = q_V_series[int(hist_idx)]
                    a_A_k = a_A_series[int(hist_idx)] if a_A_series is not None else None
                    a_V_k = a_V_series[int(hist_idx)] if a_V_series is not None else None
                    self._timeseries_samples[int(hist_idx)] = {
                        "p_inlet": sample_scalar_at_dofs(p_A_k, dofs_in),
                        "q_inlet": sample_scalar_at_dofs(q_A_k, dofs_in),
                        "a_inlet": sample_scalar_at_dofs(a_A_k, dofs_in) if a_A_k is not None else np.zeros(n_in, dtype=float),
                        "p_outlet": sample_scalar_at_dofs(p_V_k, dofs_out),
                        "q_outlet": sample_scalar_at_dofs(q_V_k, dofs_out),
                        "a_outlet": sample_scalar_at_dofs(a_V_k, dofs_out) if a_V_k is not None else np.zeros(n_out, dtype=float),
                        "p_in_BC": float(p_A_k.x.array[bc_in_dof]),
                        "p_out_BC": float(p_V_k.x.array[bc_out_dof]),
                    }

                first = self._timeseries_samples[int(self.checkpoint_indices[0])]
                p_inlet = np.ascontiguousarray(first["p_inlet"], dtype=np.float64)
                q_inlet = np.ascontiguousarray(first["q_inlet"], dtype=np.float64)
                a_inlet = np.ascontiguousarray(first["a_inlet"], dtype=np.float64)
                p_outlet = np.ascontiguousarray(first["p_outlet"], dtype=np.float64)
                q_outlet = np.ascontiguousarray(first["q_outlet"], dtype=np.float64)
                a_outlet = np.ascontiguousarray(first["a_outlet"], dtype=np.float64)
                p_in_BC = float(first["p_in_BC"])
                p_out_BC = float(first["p_out_BC"])
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

        self.x_inlet = x_inlet
        self.x_outlet = x_outlet
        self.p_inlet = p_inlet
        self.q_inlet = q_inlet
        self.a_inlet = a_inlet
        self.p_outlet = p_outlet
        self.q_outlet = q_outlet
        self.a_outlet = a_outlet

        self.branch_coords_in = None
        self.branch_ids_in = None
        self.branch_normals_in = None
        self.branch_coords_out = None
        self.branch_ids_out = None
        self.branch_normals_out = None

        if branching_in_file is not None:
            self.branch_coords_in, self.branch_ids_in, self.branch_normals_in = self._load_terminal_branch_metadata(branching_in_file)
        if branching_out_file is not None:
            self.branch_coords_out, self.branch_ids_out, self.branch_normals_out = self._load_terminal_branch_metadata(branching_out_file)

        self.inlet_coord = np.asarray(inlet_coord, dtype=float)
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
        self.p_in_BC = float(p_in_BC)
        self.p_out_BC = float(p_out_BC)
        self._terminal_counts = (n_in, n_out)

        if self.mesh.comm.rank == 0:
            print(f"Dirichlet BC pressures: p_in = {self.p_in_BC:.3e}, p_out = {self.p_out_BC:.3e}")

    def set_checkpoint_index(self, checkpoint_index: int, time_value: float) -> None:
        comm = self.mesh.comm
        n_in, n_out = self._terminal_counts
        self.checkpoint_time_index = int(checkpoint_index)
        if comm.rank == 0:
            if self.skip_1d:
                p_inlet = np.zeros(n_in, dtype=np.float64)
                q_inlet = np.zeros(n_in, dtype=np.float64)
                a_inlet = np.zeros(n_in, dtype=np.float64)
                p_outlet = np.zeros(n_out, dtype=np.float64)
                q_outlet = np.zeros(n_out, dtype=np.float64)
                a_outlet = np.zeros(n_out, dtype=np.float64)
                p_in_BC = float(self.fallback_inlet_pressure)
                p_out_BC = float(self.fallback_outlet_pressure)
                if self.leak_pressure_output_csv is not None and self.organoid_index > 0:
                    p_art = read_pressure_for_time(
                        self.leak_pressure_output_csv,
                        f"leak_art_{self.organoid_index}",
                        float(time_value),
                    )
                    p_ven = read_pressure_for_time(
                        self.leak_pressure_output_csv,
                        f"leak_ven_{self.organoid_index}",
                        float(time_value),
                    )
                    if p_art is not None:
                        p_in_BC = float(p_art)
                    if p_ven is not None:
                        p_out_BC = float(p_ven)
            else:
                sample = self._timeseries_samples.get(int(checkpoint_index))
                if sample is None:
                    raise KeyError(f"Checkpoint index {checkpoint_index} was not preloaded")
                p_inlet = np.ascontiguousarray(sample["p_inlet"], dtype=np.float64)
                q_inlet = np.ascontiguousarray(sample["q_inlet"], dtype=np.float64)
                a_inlet = np.ascontiguousarray(sample["a_inlet"], dtype=np.float64)
                p_outlet = np.ascontiguousarray(sample["p_outlet"], dtype=np.float64)
                q_outlet = np.ascontiguousarray(sample["q_outlet"], dtype=np.float64)
                a_outlet = np.ascontiguousarray(sample["a_outlet"], dtype=np.float64)
                p_in_BC = float(sample["p_in_BC"])
                p_out_BC = float(sample["p_out_BC"])
        else:
            p_inlet = np.empty(n_in, dtype=np.float64)
            q_inlet = np.empty(n_in, dtype=np.float64)
            a_inlet = np.empty(n_in, dtype=np.float64)
            p_outlet = np.empty(n_out, dtype=np.float64)
            q_outlet = np.empty(n_out, dtype=np.float64)
            a_outlet = np.empty(n_out, dtype=np.float64)
            p_in_BC = None
            p_out_BC = None

        comm.Bcast(p_inlet, root=0)
        comm.Bcast(q_inlet, root=0)
        comm.Bcast(a_inlet, root=0)
        comm.Bcast(p_outlet, root=0)
        comm.Bcast(q_outlet, root=0)
        comm.Bcast(a_outlet, root=0)
        p_in_BC = comm.bcast(p_in_BC, root=0)
        p_out_BC = comm.bcast(p_out_BC, root=0)

        self.p_inlet = p_inlet
        self.q_inlet = q_inlet
        self.a_inlet = a_inlet
        self.p_outlet = p_outlet
        self.q_outlet = q_outlet
        self.a_outlet = a_outlet
        self.p_in_BC = float(p_in_BC)
        self.p_out_BC = float(p_out_BC)


class MixedTimeSeriesRunner:
    def __init__(self, solver: TimeSeriesPerfusionSolver):
        self.solver = solver
        self.mesh = solver.mesh
        self._prepare_static_problem()

    def _prepare_static_problem(self) -> None:
        mesh = self.mesh
        solver = self.solver
        facet_tags = solver.facet_tags
        tdim = mesh.topology.dim
        fdim = tdim - 1

        U_el = element("RT", mesh.basix_cell(), 1)
        P_el = element("DG", mesh.basix_cell(), 0)
        M_el = mixed_element([U_el, P_el])
        self.M = fem.functionspace(mesh, M_el)
        self.u_trial, self.p_trial = ufl.TrialFunctions(self.M)
        self.w_test, self.v_test = ufl.TestFunctions(self.M)
        self.dx = ufl.dx(mesh)
        self.ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        self.n = ufl.FacetNormal(mesh)

        stl_files = solver._resolve_perm_region_stls(solver.perm_region_path)
        if stl_files:
            if mesh.comm.rank == 0:
                print(
                    "[perm] Using STL-based permeability regions from:",
                    ", ".join(str(p) for p in stl_files),
                    flush=True,
                )
            Kfun = solver.make_K_from_stl_regions_DG0(
                mesh,
                stl_files=stl_files,
                K_low=solver.perm_low,
                K_high=solver.perm_high,
                transition_width=solver.perm_transition_width,
            )
        else:
            K_base = 1e-7
            K_wall = 1e-7
            K_high = 1.0e-8
            d_inner = 0.01
            d_outer = 0.025
            r = 0.1
            centers = np.array(
                [
                    [0, 0.900, 0.4359],
                    [0, 0.300, 0.4359],
                    [0, -0.300, 0.4359],
                    [0, -0.900, 0.4359],
                ],
                dtype=np.float64,
            )

            INLET_MARK = getattr(solver, "arterial_concave_marker", 32)
            OUTLET_MARK = getattr(solver, "venous_concave_marker", 31)
            K_wall_blobs = solver.make_K_near_wall_DG0(
                mesh,
                facet_tags,
                wall_markers=[INLET_MARK, OUTLET_MARK],
                K_base=K_base,
                K_wall=K_wall,
                d_inner=d_inner,
                d_outer=d_outer,
            )
            K_blobs = solver.make_K_DG0(mesh, K_base, K_high, centers, r, inner_frac=0.5)
            Kfun = fem.Function(K_wall_blobs.function_space)
            Kfun.x.array[:] = K_wall_blobs.x.array[:] + K_blobs.x.array[:]
            Kfun.x.scatter_forward()

        I = ufl.Identity(mesh.geometry.dim)
        e = ufl.as_vector((1.0, 0.0, 0.0))
        perm_low = float(solver.perm_low)
        perm_high = float(solver.perm_high)
        if abs(perm_high - perm_low) > 0.0:
            alpha = (Kfun - perm_low) / (perm_high - perm_low)
            alpha = ufl.max_value(0.0, ufl.min_value(1.0, alpha))
        else:
            alpha = 0.0
        K_perp = Kfun
        Kx = perm_low + alpha * (5.0 * perm_high - perm_low)
        self.Kinv_tensor = (1.0 / K_perp) * I + ((1.0 / Kx) - (1.0 / K_perp)) * ufl.outer(e, e)

        P, _ = self.M.sub(1).collapse()
        self.q_src = solver._build_q_src(P)
        Q_local = fem.assemble_scalar(fem.form(self.q_src * self.dx))
        self.Q_tot = mesh.comm.allreduce(Q_local, op=MPI.SUM)

        solver._build_terminal_port_normal_maps()
        inlet_marks, outlet_marks, *_ = solver._get_terminal_marker_data()
        self.inlet_marks = np.asarray(inlet_marks, dtype=int)
        self.outlet_marks = np.asarray(outlet_marks, dtype=int)

        inlet_set = set(int(m) for m in inlet_marks)
        outlet_set = set(int(m) for m in outlet_marks)
        robin_set = set()
        if solver.concave_bc_mode == "robin":
            if solver.lp_arterial > 0.0:
                robin_set.add(int(solver.arterial_concave_marker))
            if solver.lp_venous > 0.0:
                robin_set.add(int(solver.venous_concave_marker))

        pressure_markers = set(outlet_set)
        if solver.concave_bc_mode == "dirichlet":
            pressure_markers.add(int(solver.arterial_concave_marker))
            pressure_markers.add(int(solver.venous_concave_marker))

        all_markers = set(solver._collect_boundary_markers())
        zero_flux_markers = all_markers - inlet_set - pressure_markers - robin_set

        self.a = (
            ufl.inner(self.Kinv_tensor * self.u_trial, self.w_test) * self.dx
            - self.p_trial * ufl.div(self.w_test) * self.dx
            + ufl.div(self.u_trial) * self.v_test * self.dx
        )
        if solver.concave_bc_mode == "robin":
            if solver.lp_arterial > 0.0:
                lp_a = fem.Constant(mesh, dfx.default_scalar_type(solver.lp_arterial))
                self.a += -(1.0 / lp_a) * ufl.dot(self.u_trial, self.n) * ufl.dot(self.w_test, self.n) * self.ds(
                    int(solver.arterial_concave_marker)
                )
            if solver.lp_venous > 0.0:
                lp_v = fem.Constant(mesh, dfx.default_scalar_type(solver.lp_venous))
                self.a += -(1.0 / lp_v) * ufl.dot(self.u_trial, self.n) * ufl.dot(self.w_test, self.n) * self.ds(
                    int(solver.venous_concave_marker)
                )

        self.a_form = fem.form(self.a)
        self.bcs = solver._build_flux_bcs_mixed(self.M, zero_flux_markers)
        self.A = assemble_matrix(self.a_form, bcs=self.bcs)
        self.A.assemble()
        self.bc_dofs = solver._collect_bc_dofs(self.bcs)

        inlet_ext_mask = np.array([solver._split_marker_facets(int(m))[0].size > 0 for m in inlet_marks], dtype=bool)
        inlet_int_mask = np.array([solver._split_marker_facets(int(m))[1].size > 0 for m in inlet_marks], dtype=bool)
        outlet_int_mask = np.array([solver._split_marker_facets(int(m))[1].size > 0 for m in outlet_marks], dtype=bool)
        self.inlet_marks_ext = np.asarray(inlet_marks, dtype=int)[inlet_ext_mask]
        self.inlet_marks_int = np.asarray(inlet_marks, dtype=int)[inlet_int_mask]
        self.outlet_marks_int = np.asarray(outlet_marks, dtype=int)[outlet_int_mask]

        self.c_vecs = solver._build_constraint_vectors(self.M, self.inlet_marks_ext, self.bc_dofs)
        self.ksp = PETSc.KSP().create(mesh.comm)
        self.ksp.setOperators(self.A)
        self.ksp.setType("preonly")
        self.ksp.getPC().setType("lu")
        self.ksp.getPC().setFactorSolverType("mumps")
        self.z_list = [solver._solve_linear(self.ksp, c) for c in self.c_vecs]

        nt = len(self.inlet_marks_ext)
        self.S = np.zeros((nt, nt), dtype=float)
        for i in range(nt):
            for j in range(nt):
                self.S[i, j] = self.c_vecs[i].dot(self.z_list[j])

        self.P1vec = fem.functionspace(
            mesh,
            element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,)),
        )
        self.projector = CachedProjector(self.P1vec)

    def _build_rhs(self):
        solver = self.solver
        _a_robin_unused, Lp = solver._build_pressure_rhs_mixed(self.u_trial, self.w_test)
        L = self.q_src * self.v_test * self.dx + Lp

        q_target = np.zeros(len(self.inlet_marks), dtype=float)
        for j, _m in enumerate(self.inlet_marks):
            if j < len(solver.inlet_marker_to_1d_idx) and len(solver.q_inlet):
                idx = int(solver.inlet_marker_to_1d_idx[j])
                if 0 <= idx < len(solver.q_inlet):
                    q_target[j] = float(solver.q_inlet[idx])
        q_in_target_in = np.abs(q_target)

        q_out_target_out = np.zeros(len(self.outlet_marks), dtype=float)
        for j, _m in enumerate(self.outlet_marks):
            if j < len(solver.outlet_marker_to_1d_idx) and len(solver.q_outlet):
                idx = int(solver.outlet_marker_to_1d_idx[j])
                if 0 <= idx < len(solver.q_outlet):
                    q_out_target_out[j] = abs(float(solver.q_outlet[idx]))

        inlet_int_mask = np.array([solver._split_marker_facets(int(m))[1].size > 0 for m in self.inlet_marks], dtype=bool)
        outlet_int_mask = np.array([solver._split_marker_facets(int(m))[1].size > 0 for m in self.outlet_marks], dtype=bool)
        q_in_target_int = np.asarray(q_in_target_in, dtype=float)[inlet_int_mask]
        q_out_target_int = np.asarray(q_out_target_out, dtype=float)[outlet_int_mask]

        L += solver._build_embedded_port_flow_rhs(
            self.v_test,
            self.inlet_marks_int,
            q_in_target_int,
            self.outlet_marks_int,
            q_out_target_int,
        )

        L_form = fem.form(L)
        b = assemble_vector(L_form)
        apply_lifting(b, [self.a_form], [self.bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, self.bcs)
        q_target_ext = np.asarray(q_target, dtype=float)[
            np.array([self.solver._split_marker_facets(int(m))[0].size > 0 for m in self.inlet_marks], dtype=bool)
        ]
        return b, q_target_ext

    def solve_current_state(self):
        solver = self.solver
        b, q_target_ext = self._build_rhs()
        z0 = solver._solve_linear(self.ksp, b)

        nt = len(self.inlet_marks_ext)
        rhs = np.zeros(nt, dtype=float)
        for i in range(nt):
            rhs[i] = q_target_ext[i] - self.c_vecs[i].dot(z0)

        if nt > 0:
            lam, *_ = np.linalg.lstsq(self.S, rhs, rcond=None)
        else:
            lam = np.zeros(0, dtype=float)

        x = z0.copy()
        for j, lj in enumerate(lam):
            x.axpy(float(lj), self.z_list[j])

        x_h = fem.Function(self.M)
        x.copy(result=x_h.x.petsc_vec)
        x_h.x.scatter_forward()

        u_sub, p_sub = x_h.split()
        u_h = u_sub.collapse()
        p_h = p_sub.collapse()

        Q_art_leak = fem.assemble_scalar(
            fem.form(ufl.dot(u_h, self.n) * self.ds(int(solver.arterial_concave_marker)))
        )
        Q_ven_leak = fem.assemble_scalar(
            fem.form(ufl.dot(u_h, self.n) * self.ds(int(solver.venous_concave_marker)))
        )
        Q_art_leak = self.mesh.comm.allreduce(Q_art_leak, op=MPI.SUM)
        Q_ven_leak = self.mesh.comm.allreduce(Q_ven_leak, op=MPI.SUM)
        boundary_audit = solver._audit_boundary_fluxes(u_h)

        if hasattr(solver, "K_tensor"):
            delattr(solver, "K_tensor")

        interface_bc = solver._compute_interface_bc(p_h, u_h, self.q_src, Q_art_leak, Q_ven_leak)
        interface_bc.update(boundary_audit)
        interface_bc.update(solver._build_mass_balance_report(interface_bc, boundary_audit, self.Q_tot))
        return p_h, u_h, interface_bc

    def run_time_series(self, checkpoint_indices: Sequence[int], checkpoint_times: Sequence[float], out_dir: Path, write_fields: bool):
        out_dir.mkdir(parents=True, exist_ok=True)
        entries: List[dict] = []

        if write_fields:
            with io.XDMFFile(self.mesh.comm, out_dir / "p.xdmf", "w") as p_writer, io.XDMFFile(
                self.mesh.comm, out_dir / "u.xdmf", "w"
            ) as u_writer:
                p_writer.write_mesh(self.mesh)
                u_writer.write_mesh(self.mesh)
                for step_idx, checkpoint_index in enumerate(checkpoint_indices):
                    time_value = float(checkpoint_times[step_idx])
                    self.solver.set_checkpoint_index(int(checkpoint_index), time_value)
                    p_h, u_h, interface_bc = self.solve_current_state()
                    u_P1 = self.projector(u_h)
                    p_writer.write_function(p_h, time_value)
                    u_writer.write_function(u_P1, time_value)

                    entry = dict(interface_bc)
                    entry["checkpoint_index"] = int(checkpoint_index)
                    entry["time"] = time_value
                    entries.append(entry)
        else:
            for step_idx, checkpoint_index in enumerate(checkpoint_indices):
                time_value = float(checkpoint_times[step_idx])
                self.solver.set_checkpoint_index(int(checkpoint_index), time_value)
                _p_h, _u_h, interface_bc = self.solve_current_state()
                entry = dict(interface_bc)
                entry["checkpoint_index"] = int(checkpoint_index)
                entry["time"] = time_value
                entries.append(entry)

        payload = {
            "checkpoint_indices": [int(i) for i in checkpoint_indices],
            "times": [float(t) for t in checkpoint_times],
            "field_output_written": bool(write_fields),
            "skip_1d": bool(entries[0].get("skip_1d", False)) if entries else bool(self.solver.skip_1d),
            "p_inlet_nodes": [entry.get("p_inlet_nodes", []) for entry in entries],
            "p_outlet_nodes": [entry.get("p_outlet_nodes", []) for entry in entries],
            "q_artery_leak": [entry.get("q_artery_leak", 0.0) for entry in entries],
            "q_venous_leak": [entry.get("q_venous_leak", 0.0) for entry in entries],
            "q_inlet": [entry.get("q_inlet", []) for entry in entries],
            "q_inlet_target": [entry.get("q_inlet_target", []) for entry in entries],
            "p_inlet_target": [entry.get("p_inlet_target", []) for entry in entries],
            "p_outlet_target": [entry.get("p_outlet_target", []) for entry in entries],
            "p_concave_inlet_bc": [entry.get("p_concave_inlet_bc", 0.0) for entry in entries],
            "p_concave_outlet_bc": [entry.get("p_concave_outlet_bc", 0.0) for entry in entries],
            "entries": entries,
        }

        if self.mesh.comm.rank == 0:
            with open(out_dir / "interface_bc_timeseries.json", "w", encoding="utf-8") as fp:
                json.dump(payload, fp, indent=2)

        return payload


def parse_args():
    ap = argparse.ArgumentParser(description="Run mixed Darcy for multiple checkpoint times in one process.")
    ap.add_argument("--bioreactor-domain", required=True)
    ap.add_argument("--facet-file", required=True)
    ap.add_argument("--mesh-inlet-file", required=True)
    ap.add_argument("--mesh-outlet-file", required=True)
    ap.add_argument("--pres-inlet-file", required=True)
    ap.add_argument("--pres-outlet-file", required=True)
    ap.add_argument("--flow-inlet-file", required=True)
    ap.add_argument("--flow-outlet-file", required=True)
    ap.add_argument("--area-inlet-file", default="")
    ap.add_argument("--area-outlet-file", default="")
    ap.add_argument("--branching-in-file", default="")
    ap.add_argument("--branching-out-file", default="")
    ap.add_argument("--skip-1d", action="store_true")
    ap.add_argument("--fallback-inlet-pressure", type=float, default=0.0)
    ap.add_argument("--fallback-outlet-pressure", type=float, default=0.0)
    ap.add_argument("--coords-inlet", nargs=3, type=float, required=True)
    ap.add_argument("--coords-outlet", nargs=3, type=float, required=True)
    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    ap.add_argument("--lp-arterial", type=float, default=0.0)
    ap.add_argument("--lp-venous", type=float, default=0.0)
    ap.add_argument("--perm-region-path", default="")
    ap.add_argument("--perm-low", type=float, default=1.0e-8)
    ap.add_argument("--perm-high", type=float, default=1.0e-6)
    ap.add_argument("--perm-transition-width", type=float, default=0.01)
    ap.add_argument("--time-indices", nargs="+", type=int, required=True)
    ap.add_argument("--time-values", nargs="*", type=float, default=[])
    ap.add_argument("--leak-pressure-output-csv", default="")
    ap.add_argument("--organoid-index", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--write-fields", action="store_true")
    return ap.parse_args()


def main():
    global current_dir
    args = parse_args()

    time_indices = resolve_selected_indices(args.time_indices)
    time_values = np.asarray(args.time_values, dtype=float)
    if time_values.size not in {0, len(time_indices)}:
        raise ValueError("--time-values must be omitted or have the same length as --time-indices")
    if time_values.size == 0:
        time_values = np.asarray(time_indices, dtype=float)

    current_dir = Path(args.out_dir).expanduser().resolve()
    current_dir.mkdir(parents=True, exist_ok=True)
    darcy_cg.current_dir = current_dir
    darcy_mixed.current_dir = current_dir

    solver = TimeSeriesPerfusionSolver(
        args.bioreactor_domain,
        args.facet_file,
        args.mesh_inlet_file,
        args.mesh_outlet_file,
        args.pres_inlet_file,
        args.pres_outlet_file,
        args.flow_inlet_file,
        args.flow_outlet_file,
        np.asarray(args.coords_inlet, dtype=float),
        np.asarray(args.coords_outlet, dtype=float),
        checkpoint_indices=time_indices,
        checkpoint_times=time_values,
        area_inlet_file=args.area_inlet_file if args.area_inlet_file else None,
        area_outlet_file=args.area_outlet_file if args.area_outlet_file else None,
        concave_bc_mode=args.concave_bc_mode,
        lp_arterial=args.lp_arterial,
        lp_venous=args.lp_venous,
        skip_1d=args.skip_1d,
        fallback_inlet_pressure=args.fallback_inlet_pressure,
        fallback_outlet_pressure=args.fallback_outlet_pressure,
        perm_region_path=args.perm_region_path,
        perm_low=args.perm_low,
        perm_high=args.perm_high,
        perm_transition_width=args.perm_transition_width,
        branching_in_file=args.branching_in_file if args.branching_in_file else None,
        branching_out_file=args.branching_out_file if args.branching_out_file else None,
        leak_pressure_output_csv=(Path(args.leak_pressure_output_csv) if args.leak_pressure_output_csv else None),
        organoid_index=int(args.organoid_index),
    )
    runner = MixedTimeSeriesRunner(solver)
    runner.run_time_series(time_indices, time_values, current_dir, write_fields=bool(args.write_fields))
    if MPI.COMM_WORLD.rank == 0:
        print("Darcy mixed time-series solve complete.")


if __name__ == "__main__":
    main()
