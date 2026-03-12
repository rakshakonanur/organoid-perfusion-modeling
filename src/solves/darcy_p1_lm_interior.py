import argparse
from pathlib import Path

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
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-0.175, 0.9, 0.55])
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.175, 0.9, 0.55])
    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    ap.add_argument("--lp-arterial", type=float, default=0.0)
    ap.add_argument("--lp-venous", type=float, default=0.0)
    ap.add_argument("--inlet-flux-correction", action="store_true", default=False)
    ap.add_argument("--inlet-flux-corr-max-iter", type=int, default=5)
    ap.add_argument("--inlet-flux-corr-relax", type=float, default=0.5)
    ap.add_argument("--inlet-flux-corr-tol", type=float, default=0.05)
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
        inlet_flux_correction=args.inlet_flux_correction,
        inlet_flux_corr_max_iter=args.inlet_flux_corr_max_iter,
        inlet_flux_corr_relax=args.inlet_flux_corr_relax,
        inlet_flux_corr_tol=args.inlet_flux_corr_tol,
        branching_in_file=args.branching_in_file if args.branching_in_file else None,
        branching_out_file=args.branching_out_file if args.branching_out_file else None,
    )
    solver.setup()
