import argparse
import json
from pathlib import Path

import numpy as np
import ufl
import dolfinx as dfx

from mpi4py import MPI
from basix.ufl import element, mixed_element
from dolfinx import fem, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import VTKFile

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

    def setup(self):
        mesh = self.mesh
        facet_tags = self.facet_tags
        tdim = mesh.topology.dim
        fdim = tdim - 1

        # Mixed space: pressure + H(div) velocity
        P_el = element("Lagrange", mesh.basix_cell(), 1)
        U_el = element("BDM", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
        M_el = mixed_element([P_el, U_el])
        M = fem.functionspace(mesh, M_el)
        p, u = ufl.TrialFunctions(M)
        v, w = ufl.TestFunctions(M)

        # ---------------------------
        # K(x)
        # ---------------------------
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

        K_par_scale = 1.0
        K_perp_scale = 1.0
        K_par = K_par_scale * Kfun
        K_perp = K_perp_scale * Kfun

        I = ufl.Identity(mesh.geometry.dim)
        e = ufl.as_vector((1.0, 0.0, 0.0))
        K_tensor = K_perp * I + (K_par - K_perp) * ufl.outer(e, e)

        # keep q_src in pressure space
        P, _ = M.sub(0).collapse()
        q_src = self._build_q_src(P)
        dx = ufl.dx(mesh)
        Q_tot = fem.assemble_scalar(fem.form(q_src * dx))
        if mesh.comm.rank == 0:
            print("Total ∫ q_src dx before correction =", Q_tot)
            if self.inlet_flux_correction:
                print("[warn] inlet_flux_correction is currently disabled in darcy_mixed.py.")

        ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tags)
        n = ufl.FacetNormal(mesh)

        # Mixed weak form:
        # 1) Constitutive relation: u + K grad(p) = 0
        a = ufl.inner(u, w) * dx + ufl.inner(K_tensor * ufl.grad(p), w) * dx

        # 2) Continuity with explicit boundary flux handling:
        #    div(u)=q  ->  -∫ u·grad(v) + ∫(u·n)v = ∫ q v
        a += -ufl.inner(u, ufl.grad(v)) * dx

        inlet_marks, _outlet_marks, *_ = self._get_terminal_marker_data()
        inlet_set = set(int(m) for m in inlet_marks)
        robin_set = set()
        if self.concave_bc_mode == "robin":
            if self.lp_arterial > 0.0:
                robin_set.add(int(self.arterial_concave_marker))
            if self.lp_venous > 0.0:
                robin_set.add(int(self.venous_concave_marker))

        all_marks = [int(m) for m in np.unique(facet_tags.values)]
        unknown_flux_marks = [m for m in all_marks if m not in inlet_set and m not in robin_set]
        for m in unknown_flux_marks:
            a += ufl.dot(u, n) * v * ds(m)

        L = q_src * v * dx + self._build_terminal_flux_rhs_mixed(v)

        if self.concave_bc_mode == "robin":
            if self.lp_arterial > 0.0:
                lp_a = fem.Constant(mesh, dfx.default_scalar_type(self.lp_arterial))
                p_a = fem.Constant(mesh, dfx.default_scalar_type(self.p_in_BC))
                a += -lp_a * p * v * ds(int(self.arterial_concave_marker))
                L += lp_a * p_a * v * ds(int(self.arterial_concave_marker))
            if self.lp_venous > 0.0:
                lp_v = fem.Constant(mesh, dfx.default_scalar_type(self.lp_venous))
                p_v = fem.Constant(mesh, dfx.default_scalar_type(self.p_out_BC))
                a += -lp_v * p * v * ds(int(self.venous_concave_marker))
                L += lp_v * p_v * v * ds(int(self.venous_concave_marker))

        bcs = self._build_terminal_bcs_mixed(M)
        if self.concave_bc_mode == "dirichlet":
            bcs = self._build_wall_dirichlet_bcs_mixed(M) + bcs

        problem = LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            },
        )
        x_h = problem.solve()
        p_h, u_h = x_h.split()

        # --- total leak flux across concave facet markers ---
        Q_art_leak = fem.assemble_scalar(fem.form(ufl.dot(u_h, n) * ds(int(self.arterial_concave_marker))))
        Q_ven_leak = fem.assemble_scalar(fem.form(ufl.dot(u_h, n) * ds(int(self.venous_concave_marker))))
        Q_art_leak = mesh.comm.allreduce(Q_art_leak, op=MPI.SUM)
        Q_ven_leak = mesh.comm.allreduce(Q_ven_leak, op=MPI.SUM)

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
        inlet_flux_correction=args.inlet_flux_correction,
        inlet_flux_corr_max_iter=args.inlet_flux_corr_max_iter,
        inlet_flux_corr_relax=args.inlet_flux_corr_relax,
        inlet_flux_corr_tol=args.inlet_flux_corr_tol,
        branching_in_file=args.branching_in_file if args.branching_in_file else None,
        branching_out_file=args.branching_out_file if args.branching_out_file else None,
    )
    solver.setup()
