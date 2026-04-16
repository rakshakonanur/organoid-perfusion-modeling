import argparse
import json
import glob
from pathlib import Path
import runtime_env  # noqa: F401

import numpy as np
import ufl
import dolfinx as dfx

from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element, mixed_element
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from dolfinx.io import VTKFile
from scipy.spatial import cKDTree

import darcy as darcy_cg
from darcy import PerfusionSolver as CGPerfusionSolver, Projector

current_dir = Path(
    "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/solves"
)


class PerfusionSolver(CGPerfusionSolver):
    """
    Mixed Darcy variant preserving the same constructor/CLI/output interface as darcy.py.

    Notes:
    - Reuses all data-loading, terminal mapping, and interface JSON logic from darcy.py.
    - Replaces the pressure-only solve with a mixed (p, u) solve using BDM velocity.
    """

    def __init__(
        self,
        *args,
        perm_region_path: str = "",
        perm_low: float = 1.0e-8,
        perm_high: float = 1.0e-6,
        perm_transition_width: float = 0.01,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.perm_region_path = str(perm_region_path).strip()
        self.perm_low = float(perm_low)
        self.perm_high = float(perm_high)
        self.perm_transition_width = float(perm_transition_width)

    @staticmethod
    def _resolve_perm_region_stls(path_str: str):
        path = Path(path_str).expanduser()
        if not str(path_str).strip():
            return []
        if path.is_dir():
            return sorted(path.glob("*.stl"))
        if path.is_file():
            return [path]
        matches = [Path(p) for p in sorted(glob.glob(str(path)))]
        return [p for p in matches if p.suffix.lower() == ".stl"]

    @staticmethod
    def _build_stl_distance_tester(stl_file: Path, tol: float = 1e-9):
        try:
            import vtk
        except ImportError as exc:
            raise ImportError(
                "vtk is required to build STL-based permeability regions. "
                "Please install vtk in the Darcy runtime environment."
            ) from exc

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

        def _inside(point: np.ndarray) -> bool:
            p = np.asarray(point, dtype=float).reshape(3,)
            return bool(enclosed.IsInsideSurface(float(p[0]), float(p[1]), float(p[2])))

        def _signed_distance(point: np.ndarray) -> float:
            p = np.asarray(point, dtype=float).reshape(3,)
            return float(distance.EvaluateFunction((float(p[0]), float(p[1]), float(p[2]))))

        def _cleanup() -> None:
            try:
                enclosed.Complete()
            except Exception:
                pass

        return _inside, _signed_distance, _cleanup

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

    def make_K_from_stl_regions_DG0(self, mesh, stl_files, K_low, K_high, transition_width=0.0):
        V0 = fem.functionspace(mesh, ("DG", 0))
        Kfun = fem.Function(V0)

        if not stl_files:
            Kfun.x.array[:] = float(K_low)
            Kfun.x.scatter_forward()
            return Kfun

        tdim = mesh.topology.dim
        gdim = mesh.geometry.dim
        mesh.topology.create_connectivity(tdim, 0)
        c2v = mesh.topology.connectivity(tdim, 0)

        nloc = mesh.topology.index_map(tdim).size_local
        cell_ids = np.arange(nloc, dtype=np.int32)
        x = mesh.geometry.x
        K_local = np.full(nloc, float(K_low), dtype=np.float64)

        testers = []
        try:
            for stl_file in stl_files:
                testers.append(self._build_stl_distance_tester(Path(stl_file)))

            for c in cell_ids:
                vs = c2v.links(int(c))
                pts = x[vs, :gdim]
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
                alpha = self._smooth_transition_alpha(float(signed_dist), float(transition_width))
                K_local[c] = float(K_low) + alpha * (float(K_high) - float(K_low))
        finally:
            for _inside, _dist, cleanup in testers:
                cleanup()

        Kfun.x.array[:nloc] = K_local
        Kfun.x.scatter_forward()
        return Kfun

    def _build_terminal_flux_rhs_mixed(self, v):
        """
        Build RHS term for prescribed inlet terminal fluxes in the mixed continuity equation.
        For each inlet marker m:
          g_m = Q_m / Area_3D(m)
          RHS += ∫ g_m * v ds(m)
        """
        mesh = self.mesh
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))

        inlet_marks, _, _, _, idx_in, _ = self._get_terminal_marker_data()
        self.inlet_marker_area_3d = np.zeros(len(inlet_marks), dtype=float)
        self.inlet_marker_area_1d = np.zeros(len(inlet_marks), dtype=float)
        self.inlet_marker_q_target = np.zeros(len(inlet_marks), dtype=float)

        L_flux = 0
        for j, m in enumerate(inlet_marks):
            if j >= len(idx_in):
                continue
            Q = float(self.q_inlet[idx_in[j]]) if len(self.q_inlet) else 0.0
            scale = float(self._inlet_flux_scale_by_marker.get(int(m), 1.0))
            Q *= scale
            A_face = float(fem.assemble_scalar(fem.form(one * ds(int(m)))))
            A_ckpt = float(self.a_inlet[idx_in[j]]) if len(self.a_inlet) else 0.0
            self.inlet_marker_area_3d[j] = A_face
            self.inlet_marker_area_1d[j] = A_ckpt
            self.inlet_marker_q_target[j] = Q
            g = Q / A_face if A_face > 0.0 else 0.0
            gc = fem.Constant(mesh, dfx.default_scalar_type(g))
            L_flux += gc * v * ds(int(m))

        return L_flux

    def _build_embedded_port_flow_rhs(self, v, inlet_marks, inlet_q_in, outlet_marks, outlet_q_out):
        mesh = self.mesh
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=self.facet_tags)
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))
        L_ports = 0

        for m, Q_in in zip(np.asarray(inlet_marks, dtype=int), np.asarray(inlet_q_in, dtype=float)):
            _ext, int_facets = self._split_marker_facets(int(m))
            if int_facets.size == 0:
                continue
            area = float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * dS(int(m)))), op=MPI.SUM))
            if area <= 0.0:
                continue
            g = fem.Constant(mesh, dfx.default_scalar_type(float(Q_in) / area))
            L_ports += g * ufl.avg(v) * dS(int(m))

        for m, Q_out in zip(np.asarray(outlet_marks, dtype=int), np.asarray(outlet_q_out, dtype=float)):
            _ext, int_facets = self._split_marker_facets(int(m))
            if int_facets.size == 0:
                continue
            area = float(mesh.comm.allreduce(fem.assemble_scalar(fem.form(one * dS(int(m)))), op=MPI.SUM))
            if area <= 0.0:
                continue
            g = fem.Constant(mesh, dfx.default_scalar_type(float(Q_out) / area))
            L_ports += -g * ufl.avg(v) * dS(int(m))

        return L_ports

    @staticmethod
    def _collect_bc_dofs(bcs):
        out = []
        for bc in bcs:
            try:
                dofs = bc.dof_indices()
            except TypeError:
                dofs = bc.dof_indices
            if isinstance(dofs, tuple):
                dofs = dofs[0]
            out.append(np.asarray(dofs, dtype=np.int32))
        if not out:
            return np.array([], dtype=np.int32)
        return np.unique(np.hstack(out))

    def _assemble_system(self, a, L, bcs):
        a_form = fem.form(a)
        L_form = fem.form(L)
        A = assemble_matrix(a_form, bcs=bcs)
        A.assemble()
        b = assemble_vector(L_form)
        apply_lifting(b, [a_form], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        set_bc(b, bcs)
        return A, b, a_form

    @staticmethod
    def _solve_linear(ksp, rhs):
        x = rhs.duplicate()
        x.set(0.0)
        ksp.solve(rhs, x)
        return x

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

        ext, interior = [], []
        for f in facets:
            nc = len(f_to_c.links(int(f)))
            if nc <= 1:
                ext.append(int(f))
            else:
                interior.append(int(f))
        return np.asarray(ext, dtype=np.int32), np.asarray(interior, dtype=np.int32)

    def _build_terminal_bcs_mixed(self, M):
        """
        Build Dirichlet pressure BCs on outlet terminal facets for mixed space.
        """
        def _parent_subspace_dofs(dofs_obj):
            if isinstance(dofs_obj, (tuple, list)):
                if len(dofs_obj) == 0:
                    return np.array([], dtype=np.int32)
                return np.asarray(dofs_obj[0], dtype=np.int32)
            return np.asarray(dofs_obj, dtype=np.int32)

        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        bcs = []
        _, outlet_marks, _, _, _, idx_out = self._get_terminal_marker_data()
        P, _ = M.sub(0).collapse()

        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(mesh.topology.dim, fdim)

        for j, m in enumerate(outlet_marks):
            if j >= len(idx_out) or len(self.p_outlet) == 0:
                continue
            facets = self.facet_tags.find(int(m))
            if facets.size == 0:
                continue
            dofs_obj = fem.locate_dofs_topological((M.sub(0), P), fdim, facets)
            dofs = _parent_subspace_dofs(dofs_obj)
            if dofs.size == 0:
                continue
            pval = float(self.p_outlet[idx_out[j]])
            bcs.append(fem.dirichletbc(dfx.default_scalar_type(pval), dofs, M.sub(0)))

        if mesh.comm.rank == 0:
            print("Number of venous terminal pressure BCs:", len(bcs))
        return bcs

    def _build_wall_dirichlet_bcs_mixed(self, M):
        """
        Build Dirichlet pressure BCs on concave arterial/venous faces for mixed space.
        """
        def _parent_subspace_dofs(dofs_obj):
            if isinstance(dofs_obj, (tuple, list)):
                if len(dofs_obj) == 0:
                    return np.array([], dtype=np.int32)
                return np.asarray(dofs_obj[0], dtype=np.int32)
            return np.asarray(dofs_obj, dtype=np.int32)

        mesh = self.mesh
        facet_tags = self.facet_tags
        fdim = mesh.topology.dim - 1
        P, _ = M.sub(0).collapse()

        ART_CONCAVE = getattr(self, "arterial_concave_marker", 31)
        VEN_CONCAVE = getattr(self, "venous_concave_marker", 32)

        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(fdim, 0)
        mesh.topology.create_connectivity(mesh.topology.dim, fdim)

        art_facets = facet_tags.find(ART_CONCAVE)
        ven_facets = facet_tags.find(VEN_CONCAVE)

        if mesh.comm.rank == 0:
            print("Number of arterial concave facets tagged:", art_facets.size)
            print("Number of venous  concave facets tagged:", ven_facets.size)

        art_dofs = _parent_subspace_dofs(
            fem.locate_dofs_topological((M.sub(0), P), fdim, art_facets)
        )
        ven_dofs = _parent_subspace_dofs(
            fem.locate_dofs_topological((M.sub(0), P), fdim, ven_facets)
        )

        bcs = []
        if art_dofs.size > 0:
            bcs.append(fem.dirichletbc(dfx.default_scalar_type(self.p_in_BC), art_dofs, M.sub(0)))
        if ven_dofs.size > 0:
            bcs.append(fem.dirichletbc(dfx.default_scalar_type(self.p_out_BC), ven_dofs, M.sub(0)))
        return bcs

    def _collect_boundary_markers(self):
        local = set(int(v) for v in np.asarray(self.facet_tags.values, dtype=np.int32))
        gathered = self.mesh.comm.allgather(sorted(local))
        merged = set()
        for marks in gathered:
            merged.update(int(m) for m in marks)
        return sorted(merged)

    def _audit_boundary_fluxes(self, u_h):
        mesh = self.mesh
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(mesh)
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))

        markers = self._collect_boundary_markers()
        marker_flux = {}
        marker_area = {}

        for m in markers:
            flux_local = fem.assemble_scalar(fem.form(ufl.dot(u_h, n) * ds(int(m))))
            area_local = fem.assemble_scalar(fem.form(one * ds(int(m))))
            flux = mesh.comm.allreduce(flux_local, op=MPI.SUM)
            area = mesh.comm.allreduce(area_local, op=MPI.SUM)
            marker_flux[str(int(m))] = float(flux)
            marker_area[str(int(m))] = float(area)

        inlet_marks, outlet_marks, *_ = self._get_terminal_marker_data()
        counted_markers = set(int(m) for m in inlet_marks)
        counted_markers.update(int(m) for m in outlet_marks)
        counted_markers.add(int(self.arterial_concave_marker))
        counted_markers.add(int(self.venous_concave_marker))

        total_flux = float(sum(marker_flux.values()))
        counted_flux = float(
            sum(marker_flux.get(str(int(m)), 0.0) for m in sorted(counted_markers))
        )
        uncounted_flux = float(total_flux - counted_flux)

        if mesh.comm.rank == 0:
            print("[audit] Exterior boundary flux by marker:")
            for m in markers:
                key = str(int(m))
                print(
                    f"  marker {int(m)}: area={marker_area[key]:.12e}, flux={marker_flux[key]:.12e}"
                )
            print(
                "[audit] total exterior flux="
                f"{total_flux:.12e}, counted flux={counted_flux:.12e}, "
                f"uncounted flux={uncounted_flux:.12e}"
            )

        return {
            "boundary_flux_by_marker": marker_flux,
            "boundary_area_by_marker": marker_area,
            "total_exterior_flux": total_flux,
            "counted_interface_markers": [int(m) for m in sorted(counted_markers)],
            "counted_interface_flux": counted_flux,
            "uncounted_exterior_flux": uncounted_flux,
        }

    def _domain_centroid(self):
        x = np.asarray(self.mesh.geometry.x, dtype=float)
        local_sum = np.sum(x, axis=0)
        local_n = np.array([x.shape[0]], dtype=np.int64)
        global_sum = np.zeros_like(local_sum)
        global_n = np.zeros_like(local_n)
        self.mesh.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        self.mesh.comm.Allreduce(local_n, global_n, op=MPI.SUM)
        denom = max(int(global_n[0]), 1)
        return global_sum / float(denom)

    def _marker_average_normal(self, marker: int):
        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        gdim = mesh.geometry.dim
        facets = np.asarray(self.facet_tags.find(int(marker)), dtype=np.int32)
        if facets.size == 0:
            return np.zeros(gdim, dtype=float)

        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(fdim, 0)
        f_to_v = mesh.topology.connectivity(fdim, 0)
        X = mesh.geometry.x
        c_dom = self._domain_centroid()

        n_acc = np.zeros(gdim, dtype=float)
        for f in facets:
            vs = f_to_v.links(int(f))
            pts = X[vs]
            center = pts.mean(axis=0)
            if pts.shape[0] < 3:
                continue
            raw = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            if np.dot(raw, center - c_dom) < 0.0:
                raw *= -1.0
            n_acc += raw

        n_acc = self.mesh.comm.allreduce(n_acc, op=MPI.SUM)
        nn = float(np.linalg.norm(n_acc))
        if nn <= 0.0:
            return np.zeros(gdim, dtype=float)
        return n_acc / nn

    def _build_flux_bcs_mixed(self, M, zero_flux_markers):
        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        U, _ = M.sub(0).collapse()
        bcs = []
        self._mixed_flux_bc_functions = []

        def add_constant_vector_bc(marker: int, vec: np.ndarray):
            facets = self.facet_tags.find(int(marker))
            if facets.size == 0:
                return
            dofs = fem.locate_dofs_topological((M.sub(0), U), fdim, facets)
            if len(dofs) == 0:
                return
            arr = np.asarray(vec, dtype=float).reshape(mesh.geometry.dim,)

            def _f(x):
                vals = np.zeros((mesh.geometry.dim, x.shape[1]), dtype=dfx.default_scalar_type)
                vals[:] = arr[:, None]
                return vals

            bc_fun = fem.Function(U)
            bc_fun.interpolate(_f)
            self._mixed_flux_bc_functions.append(bc_fun)
            bcs.append(fem.dirichletbc(bc_fun, dofs, M.sub(0)))

        for marker in sorted(set(int(m) for m in zero_flux_markers)):
            add_constant_vector_bc(marker, np.zeros(mesh.geometry.dim, dtype=float))

        return bcs

    def _build_terminal_port_normal_maps(self):
        inlet_marks, outlet_marks, inlet_c, outlet_c, _idx_in, _idx_out = self._get_terminal_marker_data()

        inlet_normals = self._get_branch_port_normals(
            inlet_c, self.branch_coords_in, self.branch_normals_in, outward=True
        )
        outlet_normals = self._get_branch_port_normals(
            outlet_c, self.branch_coords_out, self.branch_normals_out, outward=True
        )

        self._inlet_port_normal_by_marker = {
            int(m): np.asarray(n, dtype=float) for m, n in zip(inlet_marks, inlet_normals)
        }
        self._outlet_port_normal_by_marker = {
            int(m): np.asarray(n, dtype=float) for m, n in zip(outlet_marks, outlet_normals)
        }

    def _marker_port_normal_constant(self, marker: int, kind: str):
        marker = int(marker)
        if kind == "inlet":
            vec = getattr(self, "_inlet_port_normal_by_marker", {}).get(marker)
        else:
            vec = getattr(self, "_outlet_port_normal_by_marker", {}).get(marker)
        if vec is None:
            return None
        vec = np.asarray(vec, dtype=float).reshape(-1)
        if vec.size != self.mesh.geometry.dim or np.linalg.norm(vec) <= 0.0:
            return None
        return fem.Constant(self.mesh, dfx.default_scalar_type(tuple(float(v) for v in vec)))

    def _build_pressure_rhs_mixed(self, u, w):
        mesh = self.mesh
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(mesh)
        Lp = 0
        ap = 0

        _, outlet_marks, _, _, _, idx_out = self._get_terminal_marker_data()
        for j, m in enumerate(outlet_marks):
            if j >= len(idx_out) or len(self.p_outlet) == 0:
                continue
            pval = float(self.p_outlet[idx_out[j]])
            pbc = fem.Constant(mesh, dfx.default_scalar_type(pval))
            ext_facets, int_facets = self._split_marker_facets(int(m))
            if ext_facets.size > 0:
                Lp += -pbc * ufl.dot(w, n) * ds(int(m))

        if self.concave_bc_mode == "dirichlet":
            p_art = fem.Constant(mesh, dfx.default_scalar_type(self.p_in_BC))
            p_ven = fem.Constant(mesh, dfx.default_scalar_type(self.p_out_BC))
            Lp += -p_art * ufl.dot(w, n) * ds(int(self.arterial_concave_marker))
            Lp += -p_ven * ufl.dot(w, n) * ds(int(self.venous_concave_marker))
        elif self.concave_bc_mode == "robin":
            if self.lp_arterial > 0.0:
                lp_a = fem.Constant(mesh, dfx.default_scalar_type(self.lp_arterial))
                p_a = fem.Constant(mesh, dfx.default_scalar_type(self.p_in_BC))
                ap += -(1.0 / lp_a) * ufl.dot(u, n) * ufl.dot(w, n) * ds(int(self.arterial_concave_marker))
                Lp += -p_a * ufl.dot(w, n) * ds(int(self.arterial_concave_marker))
            if self.lp_venous > 0.0:
                lp_v = fem.Constant(mesh, dfx.default_scalar_type(self.lp_venous))
                p_v = fem.Constant(mesh, dfx.default_scalar_type(self.p_out_BC))
                ap += -(1.0 / lp_v) * ufl.dot(u, n) * ufl.dot(w, n) * ds(int(self.venous_concave_marker))
                Lp += -p_v * ufl.dot(w, n) * ds(int(self.venous_concave_marker))

        return ap, Lp

    def _build_mass_balance_report(self, interface_bc, boundary_audit, q_src_total):
        q_inlet = np.asarray(interface_bc.get("q_inlet", []), dtype=float)
        q_outlet = np.asarray(interface_bc.get("q_outlet", []), dtype=float)
        q_inlet_port_in = np.asarray(interface_bc.get("q_inlet_port_in", []), dtype=float)
        q_outlet_port_out = np.asarray(interface_bc.get("q_outlet_port_out", []), dtype=float)
        q_inlet_target_in = np.asarray(interface_bc.get("q_inlet_target_in", []), dtype=float)
        q_outlet_target_out = np.asarray(interface_bc.get("q_outlet_target_out", []), dtype=float)
        q_art = float(interface_bc.get("q_artery_leak", 0.0))
        q_ven = float(interface_bc.get("q_venous_leak", 0.0))

        # Sign-normalized exchange summary:
        # - inlet traces are stored outward-positive on the chosen trace, so
        #   inflow into tissue is -sum(q_inlet)
        # - arterial concave flux is outward-positive on the true boundary, so
        #   inflow into tissue is -q_artery_leak
        # - venous concave flux is outward-positive, so outflow is +q_venous_leak
        inlet_exchange_into_tissue = float(
            np.sum(q_inlet_target_in)
            if q_inlet_target_in.size
            else np.sum(q_inlet_port_in)
            if q_inlet_port_in.size
            else -np.sum(q_inlet)
        )
        outlet_trace_exchange_outward = float(-np.sum(q_outlet))
        arterial_leak_into_tissue = float(-q_art)
        venous_leak_outward = float(q_ven)

        marker_flux = boundary_audit.get("boundary_flux_by_marker", {})
        wall_outward_flux = 0.0
        for key, val in marker_flux.items():
            m = int(key)
            if m in {
                int(self.arterial_concave_marker),
                int(self.venous_concave_marker),
            }:
                continue
            if m >= int(self.inlet_base_marker):
                continue
            wall_outward_flux += float(val)

        # Trusted conservation check for the mixed solve.
        outer_residual = float(boundary_audit.get("total_exterior_flux", 0.0) - float(q_src_total))
        outer_scale = sum(abs(float(v)) for v in marker_flux.values()) + abs(float(q_src_total))
        outer_rel = abs(outer_residual) / outer_scale if outer_scale > 0.0 else 0.0

        # For embedded traces, the raw q_outlet sum uses a trace convention, not a
        # global outward-normal convention. Keep it for diagnostics, but also report
        # the outlet exchange implied by the trusted global balance.
        inlet_exchange_into_tissue_direct = float(np.sum(q_inlet_port_in)) if q_inlet_port_in.size else inlet_exchange_into_tissue
        outlet_exchange_outward_direct = float(np.sum(q_outlet_port_out)) if q_outlet_port_out.size else 0.0
        outlet_exchange_outward_target = float(np.sum(q_outlet_target_out)) if q_outlet_target_out.size else 0.0
        outlet_exchange_outward_inferred = float(
            inlet_exchange_into_tissue
            + arterial_leak_into_tissue
            + float(q_src_total)
            - venous_leak_outward
            - wall_outward_flux
        )
        trace_closure_residual = float(
            inlet_exchange_into_tissue
            + arterial_leak_into_tissue
            + float(q_src_total)
            - venous_leak_outward
            - wall_outward_flux
            - outlet_trace_exchange_outward
        )
        trace_scale = (
            abs(inlet_exchange_into_tissue)
            + abs(arterial_leak_into_tissue)
            + abs(float(q_src_total))
            + abs(venous_leak_outward)
            + abs(wall_outward_flux)
            + abs(outlet_trace_exchange_outward)
        )
        trace_rel = abs(trace_closure_residual) / trace_scale if trace_scale > 0.0 else 0.0

        direct_port_balance_residual = float(
            arterial_leak_into_tissue
            + inlet_exchange_into_tissue_direct
            + float(q_src_total)
            - venous_leak_outward
            - outlet_exchange_outward_direct
            - wall_outward_flux
        )
        direct_port_balance_scale = (
            abs(arterial_leak_into_tissue)
            + abs(inlet_exchange_into_tissue_direct)
            + abs(float(q_src_total))
            + abs(venous_leak_outward)
            + abs(outlet_exchange_outward_direct)
            + abs(wall_outward_flux)
        )
        direct_port_balance_relative = (
            abs(direct_port_balance_residual) / direct_port_balance_scale
            if direct_port_balance_scale > 0.0
            else 0.0
        )

        # Tissue control-volume balance in the sign convention the coupling uses:
        #   arterial_in + inlet_in = venous_out + outlet_out
        exchange_arterial_in = arterial_leak_into_tissue
        exchange_inlet_in = inlet_exchange_into_tissue
        exchange_venous_out = venous_leak_outward
        exchange_outlet_out = (
            outlet_exchange_outward_target
            if q_outlet_target_out.size
            else outlet_exchange_outward_inferred
        )
        exchange_in_total = exchange_arterial_in + exchange_inlet_in + float(q_src_total)
        exchange_out_total = exchange_venous_out + exchange_outlet_out + wall_outward_flux
        exchange_balance_residual = exchange_in_total - exchange_out_total
        exchange_balance_scale = (
            abs(exchange_arterial_in)
            + abs(exchange_inlet_in)
            + abs(float(q_src_total))
            + abs(exchange_venous_out)
            + abs(exchange_outlet_out)
            + abs(wall_outward_flux)
        )
        exchange_balance_relative = (
            abs(exchange_balance_residual) / exchange_balance_scale
            if exchange_balance_scale > 0.0
            else 0.0
        )

        return {
            "mass_balance_outer_residual": outer_residual,
            "mass_balance_outer_relative": outer_rel,
            "mass_balance_qsrc_total": float(q_src_total),
            "mass_balance_inlet_exchange_into_tissue": inlet_exchange_into_tissue,
            "mass_balance_arterial_leak_into_tissue": arterial_leak_into_tissue,
            "mass_balance_venous_leak_outward": venous_leak_outward,
            "mass_balance_wall_outward_flux": wall_outward_flux,
            "mass_balance_outlet_trace_outward_raw": outlet_trace_exchange_outward,
            "mass_balance_outlet_port_out_direct": outlet_exchange_outward_direct,
            "mass_balance_outlet_port_out_target": outlet_exchange_outward_target,
            "mass_balance_outlet_outward_inferred": outlet_exchange_outward_inferred,
            "mass_balance_embedded_trace_residual_raw": trace_closure_residual,
            "mass_balance_embedded_trace_relative_raw": trace_rel,
            "mass_balance_direct_port_residual": direct_port_balance_residual,
            "mass_balance_direct_port_relative": direct_port_balance_relative,
            "exchange_arterial_in": exchange_arterial_in,
            "exchange_inlet_in": exchange_inlet_in,
            "exchange_venous_out": exchange_venous_out,
            "exchange_outlet_out_direct": outlet_exchange_outward_direct,
            "exchange_outlet_out_target": outlet_exchange_outward_target,
            "exchange_outlet_out": exchange_outlet_out,
            "exchange_in_total": exchange_in_total,
            "exchange_out_total": exchange_out_total,
            "exchange_balance_residual": exchange_balance_residual,
            "exchange_balance_relative": exchange_balance_relative,
            "exchange_note": (
                "Effective tissue balance uses arterial_in + inlet_in = venous_out + outlet_out; "
                "for embedded flow/flow ports, inlet_in and outlet_out use prescribed port totals when available; "
                "raw trace and direct port fluxes remain diagnostic."
            ),
            "exchange_direct_port_note": (
                "direct_port_residual evaluates Q_art + sum(Q_in_port) - Q_ven - sum(Q_out_port) + ∫qsrc dx "
                "using branch-oriented embedded port fluxes directly."
            ),
            "mass_balance_note": (
                "outer_residual is the trustworthy global conservation check; "
                "embedded_trace_residual_raw mixes one-sided dS trace fluxes and is diagnostic only."
            ),
        }

    def _get_branch_port_normals(self, marker_centroids, branch_coords, branch_normals, outward=False):
        marker_centroids = np.asarray(marker_centroids, dtype=float).reshape((-1, 3))
        branch_coords = np.asarray(branch_coords if branch_coords is not None else np.zeros((0, 3)), dtype=float).reshape((-1, 3))
        branch_normals = np.asarray(branch_normals if branch_normals is not None else np.zeros((0, 3)), dtype=float).reshape((-1, 3))
        if marker_centroids.size == 0 or branch_coords.size == 0 or branch_normals.size == 0:
            return np.zeros((len(marker_centroids), 3), dtype=float)
        tree = cKDTree(branch_coords)
        _, idx = tree.query(marker_centroids)
        normals = np.asarray(branch_normals[np.asarray(idx, dtype=int)], dtype=float)
        if outward:
            normals = -normals
        nn = np.linalg.norm(normals, axis=1, keepdims=True)
        nn[nn <= 0.0] = 1.0
        return normals / nn

    def _compute_port_fluxes_with_normals(self, u_h, markers, marker_centroids, port_normals):
        mesh = self.mesh
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=self.facet_tags)
        q_vals = []
        for m, n_port in zip(np.asarray(markers, dtype=int), np.asarray(port_normals, dtype=float)):
            n_port = np.asarray(n_port, dtype=float).reshape(-1)
            if n_port.size != mesh.geometry.dim or np.linalg.norm(n_port) <= 0.0:
                q_vals.append(0.0)
                continue
            n_const = fem.Constant(mesh, dfx.default_scalar_type(tuple(float(v) for v in n_port)))
            ext_facets, int_facets = self._split_marker_facets(int(m))
            q_expr = 0
            if ext_facets.size > 0:
                q_expr += ufl.dot(u_h, n_const) * ds(int(m))
            if int_facets.size > 0:
                q_expr += ufl.dot(ufl.avg(u_h), n_const) * dS(int(m))
            q_vals.append(float(self.mesh.comm.allreduce(fem.assemble_scalar(fem.form(q_expr)), op=MPI.SUM)))
        return np.asarray(q_vals, dtype=float)

    def _build_constraint_vectors(self, M, inlet_marks, bc_dofs):
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(self.mesh)
        w, _v = ufl.TestFunctions(M)
        c_vecs = []

        for m in inlet_marks:
            m_int = int(m)
            ext_facets, int_facets = self._split_marker_facets(m_int)
            form_expr = 0
            if ext_facets.size > 0:
                form_expr += ufl.dot(w, n) * ds(m_int)
            if int_facets.size > 0:
                n_port = self._marker_port_normal_constant(m_int, kind="inlet")
                if n_port is None:
                    form_expr += ufl.dot(ufl.avg(w), n("+")) * dS(m_int)
                else:
                    form_expr += ufl.dot(ufl.avg(w), n_port) * dS(m_int)

            c = assemble_vector(fem.form(form_expr))
            c.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
            if bc_dofs.size > 0:
                with c.localForm() as cloc:
                    arr = cloc.array
                    lo, hi = c.getOwnershipRange()
                    i0 = max(lo, int(bc_dofs.min()))
                    i1 = min(hi, int(bc_dofs.max()) + 1)
                    if i1 > i0:
                        mask = bc_dofs[(bc_dofs >= i0) & (bc_dofs < i1)] - lo
                        arr[mask] = 0.0
            c_vecs.append(c)
        return c_vecs

    @staticmethod
    def _safe_reindex(vals, idx):
        arr = np.asarray(vals)
        ids = np.asarray(idx, dtype=int)
        if ids.size == 0:
            return arr
        if np.any(ids < 0) or np.any(ids >= len(arr)):
            return arr
        return arr[ids]

    def _compute_interface_bc(self, p_h, u_h, q_src, Q_art_leak, Q_ven_leak):
        mesh = self.mesh
        gdim = mesh.geometry.dim
        comm = mesh.comm

        inlet_marks, outlet_marks, _, _, idx_in_marker, idx_out_marker = self._get_terminal_marker_data()
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        dS = ufl.Measure("dS", domain=mesh, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(mesh)
        x = ufl.SpatialCoordinate(mesh)
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))

        def _global_scalar(form_expr):
            val = float(fem.assemble_scalar(fem.form(form_expr)))
            return float(comm.allreduce(val, op=MPI.SUM))

        def _marker_stats(marker: int):
            ext_facets, int_facets = self._split_marker_facets(marker)
            n_ext = int(comm.allreduce(int(ext_facets.size), op=MPI.SUM))
            n_int = int(comm.allreduce(int(int_facets.size), op=MPI.SUM))

            if n_ext > 0:
                area = _global_scalar(one * ds(marker))
                p_int = _global_scalar(p_h * ds(marker))
                ux = [_global_scalar(u_h[k] * ds(marker)) for k in range(gdim)]
                q = _global_scalar(ufl.dot(u_h, n) * ds(marker))
                cnum = [_global_scalar(x[k] * ds(marker)) for k in range(3)]
            elif n_int > 0:
                area = _global_scalar(one * dS(marker))
                p_int = _global_scalar(p_h("+") * dS(marker))
                ux = [_global_scalar(u_h[k]("+") * dS(marker)) for k in range(gdim)]
                q = _global_scalar(ufl.dot(u_h("+"), n("+")) * dS(marker))
                cnum = [_global_scalar(x[k]("+") * dS(marker)) for k in range(3)]
            else:
                area = 0.0
                p_int = 0.0
                ux = [0.0] * gdim
                q = 0.0
                cnum = [0.0, 0.0, 0.0]

            if area > 0.0:
                p_avg = p_int / area
                u_avg = np.array([u / area for u in ux], dtype=float)
                c = np.array([v / area for v in cnum], dtype=float)
            else:
                p_avg = 0.0
                u_avg = np.zeros(gdim, dtype=float)
                c = np.zeros(3, dtype=float)
            return p_avg, u_avg, q, c

        p_inlet_raw = []
        u_inlet_raw = []
        q_inlet_raw = []
        coords_in = []
        for m in inlet_marks:
            p_avg, u_avg, q_val, c = _marker_stats(int(m))
            p_inlet_raw.append(p_avg)
            u_inlet_raw.append(u_avg)
            q_inlet_raw.append(q_val)
            coords_in.append(c)

        p_outlet_raw = []
        u_outlet_raw = []
        q_outlet_raw = []
        coords_out = []
        for m in outlet_marks:
            p_avg, u_avg, q_val, c = _marker_stats(int(m))
            p_outlet_raw.append(p_avg)
            u_outlet_raw.append(u_avg)
            q_outlet_raw.append(q_val)
            coords_out.append(c)

        p_inlet_raw = np.asarray(p_inlet_raw, dtype=float)
        p_outlet_raw = np.asarray(p_outlet_raw, dtype=float)
        u_inlet_raw = np.asarray(u_inlet_raw, dtype=float).reshape((-1, gdim)) if len(u_inlet_raw) else np.zeros((0, gdim))
        u_outlet_raw = np.asarray(u_outlet_raw, dtype=float).reshape((-1, gdim)) if len(u_outlet_raw) else np.zeros((0, gdim))
        q_inlet_raw = np.asarray(q_inlet_raw, dtype=float)
        q_outlet_raw = np.asarray(q_outlet_raw, dtype=float)
        coords_in = np.asarray(coords_in, dtype=float).reshape((-1, 3)) if len(coords_in) else np.zeros((0, 3))
        coords_out = np.asarray(coords_out, dtype=float).reshape((-1, 3)) if len(coords_out) else np.zeros((0, 3))
        port_normals_in_raw = self._get_branch_port_normals(
            coords_in, self.branch_coords_in, self.branch_normals_in, outward=False
        )
        port_normals_out_raw = self._get_branch_port_normals(
            coords_out, self.branch_coords_out, self.branch_normals_out, outward=True
        )
        q_inlet_port_raw = self._compute_port_fluxes_with_normals(
            u_h, inlet_marks, coords_in, port_normals_in_raw
        )
        q_outlet_port_raw = self._compute_port_fluxes_with_normals(
            u_h, outlet_marks, coords_out, port_normals_out_raw
        )

        f_h, _, _, _, _, cell_vol = self._compute_q_from_div_u(u_h)
        dof_coords = f_h.function_space.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        def _sample_divu(pts):
            if len(pts) == 0:
                return np.zeros(0, dtype=float)
            _, nn = tree.query(pts)
            return np.asarray(f_h.x.array[np.asarray(nn, dtype=int)], dtype=float)

        divu_inlet_raw = _sample_divu(coords_in)
        divu_outlet_raw = _sample_divu(coords_out)

        p_inlet_target_marker = self._safe_reindex(self.p_inlet, idx_in_marker)
        p_outlet_target_marker = self._safe_reindex(self.p_outlet, idx_out_marker)
        q_inlet_target_marker = self._safe_reindex(self.q_inlet, idx_in_marker)
        q_outlet_target_marker = np.abs(self._safe_reindex(self.q_outlet, idx_out_marker))

        idx_in_branch = idx_out_branch = None
        if self.branch_coords_in is not None and len(self.branch_coords_in) > 0:
            idx_in_branch = self._build_reordering_indices(coords_in, self.branch_coords_in)
        if self.branch_coords_out is not None and len(self.branch_coords_out) > 0:
            idx_out_branch = self._build_reordering_indices(coords_out, self.branch_coords_out)

        if idx_in_branch is not None:
            p_inlet = p_inlet_raw[idx_in_branch]
            u_inlet = u_inlet_raw[idx_in_branch, :]
            q_inlet = q_inlet_raw[idx_in_branch]
            q_inlet_port = q_inlet_port_raw[idx_in_branch]
            divu_in = divu_inlet_raw[idx_in_branch]
            coords_in_final = coords_in[idx_in_branch, :]
            port_normals_in = port_normals_in_raw[idx_in_branch, :]
            ids_in = self.branch_ids_in
            p_inlet_target = p_inlet_target_marker[idx_in_branch] if len(p_inlet_target_marker) else p_inlet_target_marker
            q_inlet_target = q_inlet_target_marker[idx_in_branch] if len(q_inlet_target_marker) else q_inlet_target_marker
            q_inlet_target_in = np.abs(q_inlet_target)
        else:
            p_inlet = p_inlet_raw
            u_inlet = u_inlet_raw
            q_inlet = q_inlet_raw
            q_inlet_port = q_inlet_port_raw
            divu_in = divu_inlet_raw
            coords_in_final = coords_in
            port_normals_in = port_normals_in_raw
            ids_in = None
            p_inlet_target = p_inlet_target_marker
            q_inlet_target = q_inlet_target_marker
            q_inlet_target_in = np.abs(q_inlet_target)

        if idx_out_branch is not None:
            p_outlet = p_outlet_raw[idx_out_branch]
            u_outlet = u_outlet_raw[idx_out_branch, :]
            q_outlet = q_outlet_raw[idx_out_branch]
            q_outlet_port = q_outlet_port_raw[idx_out_branch]
            divu_out = divu_outlet_raw[idx_out_branch]
            coords_out_final = coords_out[idx_out_branch, :]
            port_normals_out = port_normals_out_raw[idx_out_branch, :]
            ids_out = self.branch_ids_out
            p_outlet_target = p_outlet_target_marker[idx_out_branch] if len(p_outlet_target_marker) else p_outlet_target_marker
            q_outlet_target = q_outlet_target_marker[idx_out_branch] if len(q_outlet_target_marker) else q_outlet_target_marker
        else:
            p_outlet = p_outlet_raw
            u_outlet = u_outlet_raw
            q_outlet = q_outlet_raw
            q_outlet_port = q_outlet_port_raw
            divu_out = divu_outlet_raw
            coords_out_final = coords_out
            port_normals_out = port_normals_out_raw
            ids_out = None
            p_outlet_target = p_outlet_target_marker
            q_outlet_target = q_outlet_target_marker

        p_inlet_mean = float(np.mean(p_inlet)) if len(p_inlet) else float("nan")
        p_outlet_mean = float(np.mean(p_outlet)) if len(p_outlet) else float("nan")

        interface_bc = {
            "p_inlet_nodes": p_inlet.tolist(),
            "p_outlet_nodes": p_outlet.tolist(),
            "p_inlet_mean": p_inlet_mean,
            "p_outlet_mean": p_outlet_mean,
            "u_inlet_nodes": u_inlet.tolist(),
            "u_outlet_nodes": u_outlet.tolist(),
            "q_inlet": q_inlet.tolist(),
            "q_outlet": q_outlet.tolist(),
            "q_inlet_port_in": q_inlet_port.tolist(),
            "q_outlet_port_out": q_outlet_port.tolist(),
            "q_inlet_inward": (-q_inlet).tolist(),
            "divu_inlet": divu_in.tolist(),
            "divu_outlet": divu_out.tolist(),
            "cell_volume": float(cell_vol),
            "q_artery_leak": float(Q_art_leak),
            "q_venous_leak": float(Q_ven_leak),
            "p_concave_inlet_bc": float(self.p_in_BC),
            "p_concave_outlet_bc": float(self.p_out_BC),
            "skip_1d": bool(self.skip_1d),
            "coords_inlet": coords_in_final.tolist(),
            "coords_outlet": coords_out_final.tolist(),
            "port_normals_inlet": port_normals_in.tolist(),
            "port_normals_outlet": port_normals_out.tolist(),
            "p_inlet_target": np.asarray(p_inlet_target, dtype=float).tolist(),
            "p_outlet_target": np.asarray(p_outlet_target, dtype=float).tolist(),
            "q_inlet_target": np.asarray(q_inlet_target, dtype=float).tolist(),
            "q_inlet_target_in": np.asarray(q_inlet_target_in, dtype=float).tolist(),
            "q_outlet_target_out": np.asarray(q_outlet_target, dtype=float).tolist(),
        }
        if len(q_inlet_target) == len(q_inlet):
            interface_bc["q_inlet_residual"] = (q_inlet - q_inlet_target).tolist()
        if ids_in is not None:
            interface_bc["branch_ids_inlet"] = np.asarray(ids_in).tolist()
        if ids_out is not None:
            interface_bc["branch_ids_outlet"] = np.asarray(ids_out).tolist()
        return interface_bc

    def setup(self):
        mesh = self.mesh
        facet_tags = self.facet_tags
        tdim = mesh.topology.dim
        fdim = tdim - 1

        # Mixed Poisson / Darcy formulation:
        # flux in H(div), pressure in discontinuous L2.
        U_el = element("RT", mesh.basix_cell(), 1)
        P_el = element("DG", mesh.basix_cell(), 0)
        M_el = mixed_element([U_el, P_el])
        M = fem.functionspace(mesh, M_el)
        u, p = ufl.TrialFunctions(M)
        w, v = ufl.TestFunctions(M)

        # ---------------------------
        # K(x)
        # ---------------------------
        stl_files = self._resolve_perm_region_stls(self.perm_region_path)
        if stl_files:
            if mesh.comm.rank == 0:
                print(
                    "[perm] Using STL-based permeability regions from:",
                    ", ".join(str(p) for p in stl_files),
                    flush=True,
                )
            Kfun = self.make_K_from_stl_regions_DG0(
                mesh,
                stl_files=stl_files,
                K_low=self.perm_low,
                K_high=self.perm_high,
                transition_width=self.perm_transition_width,
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

            INLET_MARK = getattr(self, "arterial_concave_marker", 32)
            OUTLET_MARK = getattr(self, "venous_concave_marker", 31)

            K_wall_blobs = self.make_K_near_wall_DG0(
                mesh,
                facet_tags,
                wall_markers=[INLET_MARK, OUTLET_MARK],
                K_base=K_base,
                K_wall=K_wall,
                d_inner=d_inner,
                d_outer=d_outer,
            )
            K_blobs = self.make_K_DG0(mesh, K_base, K_high, centers, r, inner_frac=0.5)
            Kfun = fem.Function(K_wall_blobs.function_space)
            Kfun.x.array[:] = K_wall_blobs.x.array[:] + K_blobs.x.array[:]
            Kfun.x.scatter_forward()

        I = ufl.Identity(mesh.geometry.dim)
        e = ufl.as_vector((1.0, 0.0, 0.0))
        perm_low = float(self.perm_low)
        perm_high = float(self.perm_high)
        if abs(perm_high - perm_low) > 0.0:
            alpha = (Kfun - perm_low) / (perm_high - perm_low)
            alpha = ufl.max_value(0.0, ufl.min_value(1.0, alpha))
        else:
            alpha = 0.0

        # Keep the background isotropic and only add x-directed anisotropy inside
        # the STL-masked region (with the same smooth transition shell as Kfun).
        K_perp = Kfun
        Kx = perm_low + alpha * (5.0 * perm_high - perm_low)
        K_tensor = K_perp * I + (Kx - K_perp) * ufl.outer(e, e)

        # keep q_src in pressure space
        P, _ = M.sub(1).collapse()
        q_src = self._build_q_src(P)
        dx = ufl.dx(mesh)
        Q_tot = fem.assemble_scalar(fem.form(q_src * dx))
        if mesh.comm.rank == 0:
            print("Total ∫ q_src dx before correction =", Q_tot)
            if self.inlet_flux_correction:
                print("[warn] inlet_flux_correction is currently disabled in darcy_mixed.py.")

        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        n = ufl.FacetNormal(mesh)

        self._build_terminal_port_normal_maps()

        inlet_marks, outlet_marks, *_ = self._get_terminal_marker_data()
        inlet_set = set(int(m) for m in inlet_marks)
        outlet_set = set(int(m) for m in outlet_marks)
        robin_set = set()
        if self.concave_bc_mode == "robin":
            if self.lp_arterial > 0.0:
                robin_set.add(int(self.arterial_concave_marker))
            if self.lp_venous > 0.0:
                robin_set.add(int(self.venous_concave_marker))

        pressure_markers = set(outlet_set)
        if self.concave_bc_mode == "dirichlet":
            pressure_markers.add(int(self.arterial_concave_marker))
            pressure_markers.add(int(self.venous_concave_marker))

        all_markers = set(self._collect_boundary_markers())
        zero_flux_markers = all_markers - inlet_set - pressure_markers - robin_set

        Kinv_tensor = (1.0 / K_perp) * I + ((1.0 / Kx) - (1.0 / K_perp)) * ufl.outer(e, e)

        a = ufl.inner(Kinv_tensor * u, w) * dx - p * ufl.div(w) * dx + ufl.div(u) * v * dx
        a_robin, Lp = self._build_pressure_rhs_mixed(u, w)
        a += a_robin
        L = q_src * v * dx + Lp

        bcs = self._build_flux_bcs_mixed(M, zero_flux_markers)
        A, b, _ = self._assemble_system(a, L, bcs)
        bc_dofs = self._collect_bc_dofs(bcs)

        q_target = np.zeros(len(inlet_marks), dtype=float)
        for j, _m in enumerate(inlet_marks):
            if j < len(self.inlet_marker_to_1d_idx) and len(self.q_inlet):
                idx = int(self.inlet_marker_to_1d_idx[j])
                if 0 <= idx < len(self.q_inlet):
                    q_target[j] = float(self.q_inlet[idx])
        q_in_target_in = np.abs(q_target)
        q_out_target_out = np.zeros(len(outlet_marks), dtype=float)
        for j, _m in enumerate(outlet_marks):
            if j < len(self.outlet_marker_to_1d_idx) and len(self.q_outlet):
                idx = int(self.outlet_marker_to_1d_idx[j])
                if 0 <= idx < len(self.q_outlet):
                    q_out_target_out[j] = abs(float(self.q_outlet[idx]))

        inlet_ext_mask = np.array([self._split_marker_facets(int(m))[0].size > 0 for m in inlet_marks], dtype=bool)
        inlet_int_mask = np.array([self._split_marker_facets(int(m))[1].size > 0 for m in inlet_marks], dtype=bool)
        outlet_int_mask = np.array([self._split_marker_facets(int(m))[1].size > 0 for m in outlet_marks], dtype=bool)

        inlet_marks_ext = np.asarray(inlet_marks, dtype=int)[inlet_ext_mask]
        q_target_ext = np.asarray(q_target, dtype=float)[inlet_ext_mask]
        inlet_marks_int = np.asarray(inlet_marks, dtype=int)[inlet_int_mask]
        q_in_target_int = np.asarray(q_in_target_in, dtype=float)[inlet_int_mask]
        outlet_marks_int = np.asarray(outlet_marks, dtype=int)[outlet_int_mask]
        q_out_target_int = np.asarray(q_out_target_out, dtype=float)[outlet_int_mask]

        L += self._build_embedded_port_flow_rhs(v, inlet_marks_int, q_in_target_int, outlet_marks_int, q_out_target_int)
        A, b, _ = self._assemble_system(a, L, bcs)
        bc_dofs = self._collect_bc_dofs(bcs)
        c_vecs = self._build_constraint_vectors(M, inlet_marks_ext, bc_dofs)

        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        z0 = self._solve_linear(ksp, b)
        z_list = [self._solve_linear(ksp, c) for c in c_vecs]

        nt = len(inlet_marks_ext)
        S = np.zeros((nt, nt), dtype=float)
        rhs = np.zeros(nt, dtype=float)

        for i in range(nt):
            rhs[i] = q_target_ext[i] - c_vecs[i].dot(z0)
            for j in range(nt):
                S[i, j] = c_vecs[i].dot(z_list[j])

        if nt > 0:
            lam, *_ = np.linalg.lstsq(S, rhs, rcond=None)
        else:
            lam = np.zeros(0, dtype=float)

        x = z0.copy()
        for j, lj in enumerate(lam):
            x.axpy(float(lj), z_list[j])

        x_h = fem.Function(M)
        x.copy(result=x_h.x.petsc_vec)
        x_h.x.scatter_forward()

        u_sub, p_sub = x_h.split()
        u_h = u_sub.collapse()
        p_h = p_sub.collapse()

        # --- total leak flux across concave facet markers ---
        Q_art_leak = fem.assemble_scalar(fem.form(ufl.dot(u_h, n) * ds(int(self.arterial_concave_marker))))
        Q_ven_leak = fem.assemble_scalar(fem.form(ufl.dot(u_h, n) * ds(int(self.venous_concave_marker))))
        Q_art_leak = mesh.comm.allreduce(Q_art_leak, op=MPI.SUM)
        Q_ven_leak = mesh.comm.allreduce(Q_ven_leak, op=MPI.SUM)
        boundary_audit = self._audit_boundary_fluxes(u_h)

        # write outputs
        out_dir = current_dir / "out_darcy"
        out_dir.mkdir(exist_ok=True)

        with io.XDMFFile(mesh.comm, out_dir / "p.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(p_h)

        # Project H(div) velocity to P1 vector space for robust IO/visualization.
        P1vec = fem.functionspace(
            mesh,
            element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,)),
        )
        u_P1 = Projector(P1vec)(u_h)

        with io.XDMFFile(mesh.comm, out_dir / "u.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(u_P1)

        vtkfile_u = VTKFile(MPI.COMM_WORLD, out_dir / "u.vtu", "w")
        vtkfile_u.write_function(u_P1)

        vtkfile_p = VTKFile(MPI.COMM_WORLD, out_dir / "p.vtu", "w")
        vtkfile_p.write_function(p_h)

        # For mixed solver, interface fluxes should come directly from u_h;
        # avoid pressure-gradient fallback path in inherited interface post-processing.
        if hasattr(self, "K_tensor"):
            delattr(self, "K_tensor")

        interface_bc = self._compute_interface_bc(p_h, u_h, q_src, Q_art_leak, Q_ven_leak)
        interface_bc.update(boundary_audit)
        interface_bc.update(self._build_mass_balance_report(interface_bc, boundary_audit, Q_tot))

        if mesh.comm.rank == 0:
            with open(out_dir / "interface_bc.json", "w") as fp:
                json.dump(interface_bc, fp, indent=2)

        print("Darcy mixed solve complete.")
        return interface_bc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run mixed (BDM) Darcy perfusion solve on prepared mesh/checkpoint files.")
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
    ap.add_argument("--skip-1d", action="store_true", help="Skip inlet/outlet 1D tree loading and use fallback concave pressures.")
    ap.add_argument("--fallback-inlet-pressure", type=float, default=0.0)
    ap.add_argument("--fallback-outlet-pressure", type=float, default=0.0)
    ap.add_argument("--checkpoint-time-index", type=int, default=-1,
                    help="Checkpoint time index to read from 1D pressure/flow/area histories (-1 = last).")
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-0.175, 0.9, 0.55])
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.175, 0.9, 0.55])
    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet",
                    help="Boundary model on concave arterial/venous faces.")
    ap.add_argument("--lp-arterial", type=float, default=0.0,
                    help="Robin transfer coefficient on arterial concave face (used when --concave-bc-mode robin).")
    ap.add_argument("--lp-venous", type=float, default=0.0,
                    help="Robin transfer coefficient on venous concave face (used when --concave-bc-mode robin).")
    ap.add_argument("--inlet-flux-correction", action="store_true", default=False,
                    help="Accepted for interface compatibility. Currently disabled in mixed solver.")
    ap.add_argument("--inlet-flux-corr-max-iter", type=int, default=5,
                    help="Accepted for interface compatibility.")
    ap.add_argument("--inlet-flux-corr-relax", type=float, default=0.5,
                    help="Accepted for interface compatibility.")
    ap.add_argument("--inlet-flux-corr-tol", type=float, default=0.05,
                    help="Accepted for interface compatibility.")
    ap.add_argument(
        "--perm-region-path",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/files/stl/organoid-growth-domains/sphere",
        help="STL file, directory, or glob pattern describing high-permeability organoid regions.",
    )
    ap.add_argument("--perm-low", type=float, default=1.0e-8, help="Background permeability outside STL regions.")
    ap.add_argument("--perm-high", type=float, default=1.0e-6, help="Permeability inside STL regions.")
    ap.add_argument(
        "--perm-transition-width",
        type=float,
        default=0.01,
        help="Half-width of the smooth permeability transition shell around the STL boundary.",
    )
    ap.add_argument("--out-dir", default="", help="Optional output directory for out_darcy")
    args = ap.parse_args()

    if args.out_dir:
        current_dir = Path(args.out_dir).expanduser().resolve()
        darcy_cg.current_dir = current_dir

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
        perm_region_path=args.perm_region_path,
        perm_low=args.perm_low,
        perm_high=args.perm_high,
        perm_transition_width=args.perm_transition_width,
        branching_in_file=args.branching_in_file if args.branching_in_file else None,
        branching_out_file=args.branching_out_file if args.branching_out_file else None,
    )
    solver.setup()
