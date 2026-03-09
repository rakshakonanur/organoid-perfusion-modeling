import argparse
import json
from pathlib import Path

import numpy as np
import ufl
import dolfinx as dfx

from mpi4py import MPI
from petsc4py import PETSc
from basix.ufl import element
from dolfinx import fem, io
from dolfinx.io import VTKFile
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

import darcy as darcy_cg
from darcy import PerfusionSolver as CGPerfusionSolver, Projector

current_dir = Path(
    "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/solves"
)


class PerfusionSolver(CGPerfusionSolver):
    """
    P1 Darcy with algebraic terminal-flux constraints (Lagrange-multiplier equivalent).
    Keeps the same constructor/CLI interface as darcy.py.
    """

    def _build_outlet_terminal_bcs(self, W):
        mesh = self.mesh
        fdim = mesh.topology.dim - 1
        bcs = []
        _, outlet_marks, _, _, _, idx_out = self._get_terminal_marker_data()

        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(mesh.topology.dim, fdim)

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

        if mesh.comm.rank == 0:
            print("Number of venous terminal pressure BCs:", len(bcs))
        return bcs

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

    def _build_constraint_vectors(self, W, K_tensor, inlet_marks, bc_dofs):
        ds = ufl.Measure("ds", domain=self.mesh, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(self.mesh)
        v = ufl.TestFunction(W)
        c_vecs = []
        for m in inlet_marks:
            c_form = fem.form(-ufl.dot(K_tensor * ufl.grad(v), n) * ds(int(m)))
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
    def _solve_linear(ksp, rhs):
        x = rhs.duplicate()
        x.set(0.0)
        ksp.solve(rhs, x)
        return x

    def setup(self):
        mesh = self.mesh
        facet_tags = self.facet_tags
        W = fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1))
        p = ufl.TrialFunction(W)
        v = ufl.TestFunction(W)

        # Permeability tensor 
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
        e = ufl.as_vector((1.0, 0.0, 0.0))  # x direction

        alpha = 5.0          # 5x along x
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

        # LM mode: pressure BC only on venous terminals (+ optional concave dirichlet)
        bcs = self._build_outlet_terminal_bcs(W)
        if self.concave_bc_mode == "dirichlet":
            bcs = self._build_wall_dirichlet_bcs(W) + bcs

        A, b, _ = self._assemble_system(a, L, bcs)
        bc_dofs = self._collect_bc_dofs(bcs)

        # Build constraint rows C_m p = Q_m for inlet terminal markers
        inlet_marks, _, _, _, idx_in, _ = self._get_terminal_marker_data()
        q_target = np.zeros(len(inlet_marks), dtype=float)
        for j, _m in enumerate(inlet_marks):
            if j < len(idx_in) and len(self.q_inlet):
                q_target[j] = float(self.q_inlet[idx_in[j]])

        c_vecs = self._build_constraint_vectors(W, K_tensor, inlet_marks, bc_dofs)

        # Direct solve for constrained system via Schur complement:
        # p = z0 + sum_j lambda_j z_j, where A z0=b, A z_j=c_j
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
            # Robust solve in case of near-singularity
            lam, *_ = np.linalg.lstsq(S, rhs, rcond=None)
        else:
            lam = np.zeros(0, dtype=float)

        x = z0.copy()
        for j, lj in enumerate(lam):
            x.axpy(float(lj), z_list[j])

        p_h = fem.Function(W)
        x.copy(result=p_h.x.petsc_vec)
        p_h.x.scatter_forward()

        # Postprocess flux in an H(div)-conforming space for better facet-normal continuity.
        V = fem.functionspace(mesh, element("RT", mesh.basix_cell(), 1))
        u_h = Projector(V)(-(K_tensor * ufl.grad(p_h)))

        # Leak diagnostics on concave markers
        n = ufl.FacetNormal(mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)
        Q_art_leak = fem.assemble_scalar(fem.form(ufl.dot(u_h, n) * ds(int(self.arterial_concave_marker))))
        Q_ven_leak = fem.assemble_scalar(fem.form(ufl.dot(u_h, n) * ds(int(self.venous_concave_marker))))
        Q_art_leak = mesh.comm.allreduce(Q_art_leak, op=MPI.SUM)
        Q_ven_leak = mesh.comm.allreduce(Q_ven_leak, op=MPI.SUM)

        out_dir = current_dir / "out_darcy"
        out_dir.mkdir(exist_ok=True)
        P1vec = fem.functionspace(mesh, element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,)))
        u_P1 = fem.Function(P1vec)
        u_P1.interpolate(u_h)
        with io.XDMFFile(mesh.comm, out_dir / "p.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(p_h)
        with io.XDMFFile(mesh.comm, out_dir / "u.xdmf", "w") as f:
            f.write_mesh(mesh)
            f.write_function(u_P1)
        vtkfile = VTKFile(MPI.COMM_WORLD, out_dir / "u.vtu", "w")
        vtkfile.write_function(u_P1)
        vtkfile_p = VTKFile(MPI.COMM_WORLD, out_dir / "p.vtu", "w")
        vtkfile_p.write_function(p_h)

        interface_bc = self._compute_interface_bc(p_h, u_h, q_src, Q_art_leak, Q_ven_leak)
        if mesh.comm.rank == 0:
            with open(out_dir / "interface_bc.json", "w") as fp:
                json.dump(interface_bc, fp, indent=2)
        print("Darcy P1-LM solve complete.")
        return interface_bc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run P1 Darcy with terminal flux constraints.")
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
