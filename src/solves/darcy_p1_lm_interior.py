import argparse
from pathlib import Path
import glob

import numpy as np
import ufl
import dolfinx as dfx

from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem
from dolfinx.fem.petsc import assemble_vector
from scipy.spatial import cKDTree

import darcy as darcy_cg
import darcy_p1_lm as lm_base

current_dir = Path(
    "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/solves"
)


class PerfusionSolver(lm_base.PerfusionSolver):
    """
    Interior-aware variant of darcy_p1_lm:
    - Outlet terminal pressures: same strong Dirichlet-on-tagged-facets behavior.
    - Inlet terminal flux constraints: uses ds for exterior markers, dS for interior markers.
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

    def _audit_boundary_fluxes(self, p_h, K_tensor):
        mesh = self.mesh
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(mesh)
        one = fem.Constant(mesh, dfx.default_scalar_type(1.0))

        markers = self._collect_boundary_markers()
        marker_flux = {}
        marker_area = {}
        for m in markers:
            ext_facets, _int_facets = self._split_marker_facets(int(m))
            if ext_facets.size == 0:
                continue
            flux_local = fem.assemble_scalar(fem.form(-ufl.dot(K_tensor * ufl.grad(p_h), n) * ds(int(m))))
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
            for m in sorted(int(k) for k in marker_flux.keys()):
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

    def _build_constraint_vectors(self, W, K_tensor, inlet_marks, bc_dofs):
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        dS = ufl.Measure("dS", domain=self.mesh, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(self.mesh)
        v = ufl.TestFunction(W)
        c_vecs = []

        for m in inlet_marks:
            m_int = int(m)
            ext_facets, int_facets = self._split_marker_facets(m_int)

            form_expr = 0
            if ext_facets.size > 0:
                # Exterior terminal facets: standard boundary flux operator.
                form_expr += -ufl.dot(K_tensor * ufl.grad(v), n) * ds(m_int)
            if int_facets.size > 0:
                # Interior terminal facets: use plus-side trace on interior measure.
                form_expr += -ufl.dot((K_tensor * ufl.grad(v))("+"), n("+")) * dS(m_int)

            c_form = fem.form(form_expr)
            c = assemble_vector(c_form)
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
                q = _global_scalar(-ufl.dot(self.K_tensor * ufl.grad(p_h), n) * ds(marker))
                cnum = [_global_scalar(x[k] * ds(marker)) for k in range(3)]
            elif n_int > 0:
                area = _global_scalar(one * dS(marker))
                p_int = _global_scalar(p_h("+") * dS(marker))
                ux = [_global_scalar(u_h[k]("+") * dS(marker)) for k in range(gdim)]
                q = _global_scalar(-ufl.dot((self.K_tensor * ufl.grad(p_h))("+"), n("+")) * dS(marker))
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

        # Marker-order arrays
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

        # Keep div(u) outputs for compatibility by sampling projected div(u) at marker centroids.
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

        idx_in_branch = idx_out_branch = None
        if self.branch_coords_in is not None and len(self.branch_coords_in) > 0:
            idx_in_branch = self._build_reordering_indices(coords_in, self.branch_coords_in)
        if self.branch_coords_out is not None and len(self.branch_coords_out) > 0:
            idx_out_branch = self._build_reordering_indices(coords_out, self.branch_coords_out)

        if idx_in_branch is not None:
            p_inlet = p_inlet_raw[idx_in_branch]
            u_inlet = u_inlet_raw[idx_in_branch, :]
            q_inlet = q_inlet_raw[idx_in_branch]
            divu_in = divu_inlet_raw[idx_in_branch]
            coords_in_final = coords_in[idx_in_branch, :]
            ids_in = self.branch_ids_in
            p_inlet_target = p_inlet_target_marker[idx_in_branch] if len(p_inlet_target_marker) else p_inlet_target_marker
            q_inlet_target = q_inlet_target_marker[idx_in_branch] if len(q_inlet_target_marker) else q_inlet_target_marker
        else:
            p_inlet = p_inlet_raw
            u_inlet = u_inlet_raw
            q_inlet = q_inlet_raw
            divu_in = divu_inlet_raw
            coords_in_final = coords_in
            ids_in = None
            p_inlet_target = p_inlet_target_marker
            q_inlet_target = q_inlet_target_marker

        if idx_out_branch is not None:
            p_outlet = p_outlet_raw[idx_out_branch]
            u_outlet = u_outlet_raw[idx_out_branch, :]
            q_outlet = q_outlet_raw[idx_out_branch]
            divu_out = divu_outlet_raw[idx_out_branch]
            coords_out_final = coords_out[idx_out_branch, :]
            ids_out = self.branch_ids_out
            p_outlet_target = p_outlet_target_marker[idx_out_branch] if len(p_outlet_target_marker) else p_outlet_target_marker
        else:
            p_outlet = p_outlet_raw
            u_outlet = u_outlet_raw
            q_outlet = q_outlet_raw
            divu_out = divu_outlet_raw
            coords_out_final = coords_out
            ids_out = None
            p_outlet_target = p_outlet_target_marker

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
            "p_inlet_target": np.asarray(p_inlet_target, dtype=float).tolist(),
            "p_outlet_target": np.asarray(p_outlet_target, dtype=float).tolist(),
            "q_inlet_target": np.asarray(q_inlet_target, dtype=float).tolist(),
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
        W = fem.functionspace(mesh, ("Lagrange", 1))
        p = ufl.TrialFunction(W)
        v = ufl.TestFunction(W)

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
                [[0, 0.900, 0.4359], [0, 0.300, 0.4359], [0, -0.300, 0.4359], [0, -0.900, 0.4359]],
                dtype=np.float64,
            )
            INLET_MARK = getattr(self, "arterial_concave_marker", 32)
            OUTLET_MARK = getattr(self, "venous_concave_marker", 31)
            K_wall_blobs = self.make_K_near_wall_DG0(
                mesh, facet_tags, wall_markers=[INLET_MARK, OUTLET_MARK],
                K_base=K_base, K_wall=K_wall, d_inner=d_inner, d_outer=d_outer
            )
            K_blobs = self.make_K_DG0(mesh, K_base, K_high, centers, r, inner_frac=0.5)
            Kfun = fem.Function(K_wall_blobs.function_space)
            Kfun.x.array[:] = K_wall_blobs.x.array[:] + K_blobs.x.array[:]
            Kfun.x.scatter_forward()

        I = ufl.Identity(mesh.geometry.dim)
        e = ufl.as_vector((1.0, 0.0, 0.0))

        alpha = 5.0
        K_perp = Kfun
        K_par = alpha * Kfun

        K_tensor = K_perp * I + (K_par - K_perp) * ufl.outer(e, e)
        self.K_tensor = K_tensor

        a = ufl.inner(K_tensor * ufl.grad(p), ufl.grad(v)) * ufl.dx
        if self.concave_bc_mode == "robin":
            a_robin, _ = self._build_concave_robin_terms(p, v)
            a += a_robin

        q_src = self._build_q_src(W)
        L = q_src * v * ufl.dx
        if self.concave_bc_mode == "robin":
            _, L_robin = self._build_concave_robin_terms(p, v)
            L += L_robin

        bcs = self._build_outlet_terminal_bcs(W)
        if self.concave_bc_mode == "dirichlet":
            bcs = self._build_wall_dirichlet_bcs(W) + bcs

        A, b, _ = self._assemble_system(a, L, bcs)
        bc_dofs = self._collect_bc_dofs(bcs)

        inlet_marks, _, _, _, idx_in, _ = self._get_terminal_marker_data()
        q_target = np.zeros(len(inlet_marks), dtype=float)
        for j, _m in enumerate(inlet_marks):
            if j < len(idx_in) and len(self.q_inlet):
                q_target[j] = float(self.q_inlet[idx_in[j]])

        c_vecs = self._build_constraint_vectors(W, K_tensor, inlet_marks, bc_dofs)

        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(A)
        ksp.setType("preonly")
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        z0 = self._solve_linear(ksp, b)
        z_list = [self._solve_linear(ksp, c) for c in c_vecs]

        nt = len(inlet_marks)
        S = np.zeros((nt, nt), dtype=float)
        rhs = np.zeros(nt, dtype=float)

        for i in range(nt):
            rhs[i] = q_target[i] - c_vecs[i].dot(z0)
            for j in range(nt):
                S[i, j] = c_vecs[i].dot(z_list[j])

        if nt > 0:
            lam, *_ = np.linalg.lstsq(S, rhs, rcond=None)
        else:
            lam = np.zeros(0, dtype=float)

        x = z0.copy()
        for j, lj in enumerate(lam):
            x.axpy(float(lj), z_list[j])

        p_h = fem.Function(W)
        x.copy(result=p_h.x.petsc_vec)
        p_h.x.scatter_forward()

        V = fem.functionspace(mesh, ("RT", 1))
        u_h = lm_base.Projector(V)(-(K_tensor * ufl.grad(p_h)))

        n = ufl.FacetNormal(mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        Q_art_leak = fem.assemble_scalar(
            fem.form(-ufl.dot(K_tensor * ufl.grad(p_h), n) * ds(int(self.arterial_concave_marker)))
        )
        Q_ven_leak = fem.assemble_scalar(
            fem.form(-ufl.dot(K_tensor * ufl.grad(p_h), n) * ds(int(self.venous_concave_marker)))
        )
        Q_art_leak = mesh.comm.allreduce(Q_art_leak, op=MPI.SUM)
        Q_ven_leak = mesh.comm.allreduce(Q_ven_leak, op=MPI.SUM)
        boundary_audit = self._audit_boundary_fluxes(p_h, K_tensor)

        out_dir = current_dir / "out_darcy"
        out_dir.mkdir(exist_ok=True)
        P1vec = fem.functionspace(mesh, ("Lagrange", 1, (mesh.geometry.dim,)))
        u_P1 = fem.Function(P1vec)
        u_P1.interpolate(u_h)
        with dfx.io.XDMFFile(mesh.comm, out_dir / "p.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(p_h)
        with dfx.io.XDMFFile(mesh.comm, out_dir / "u.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(u_P1)
        vtkfile = dfx.io.VTKFile(MPI.COMM_WORLD, out_dir / "u.vtu", "w")
        vtkfile.write_function(u_P1)
        vtkfile_p = dfx.io.VTKFile(MPI.COMM_WORLD, out_dir / "p.vtu", "w")
        vtkfile_p.write_function(p_h)

        interface_bc = self._compute_interface_bc(p_h, u_h, q_src, Q_art_leak, Q_ven_leak)
        interface_bc.update(boundary_audit)
        if mesh.comm.rank == 0:
            with open(out_dir / "interface_bc.json", "w") as fp:
                import json
                json.dump(interface_bc, fp, indent=2)
        print("Darcy P1-LM solve complete.")
        return interface_bc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run P1 Darcy LM with interior-facet terminal support.")
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
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-0.175, 0.9, 0.55])
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.175, 0.9, 0.55])
    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    ap.add_argument("--lp-arterial", type=float, default=0.0)
    ap.add_argument("--lp-venous", type=float, default=0.0)
    ap.add_argument("--inlet-flux-correction", action="store_true", default=False)
    ap.add_argument("--inlet-flux-corr-max-iter", type=int, default=5)
    ap.add_argument("--inlet-flux-corr-relax", type=float, default=0.5)
    ap.add_argument("--inlet-flux-corr-tol", type=float, default=0.05)
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
        lm_base.current_dir = current_dir

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
