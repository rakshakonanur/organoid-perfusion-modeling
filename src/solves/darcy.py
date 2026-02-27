import ufl
import numpy   as np
import dolfinx as dfx
import matplotlib.pyplot as plt
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

# -----------------------------------------------------
# Mesh importer
# -----------------------------------------------------
def import_3d_mesh(xdmf_file: str):
    with io.XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        m = xdmf.read_mesh(name="Grid")
        m.topology.create_connectivity(m.topology.dim, m.topology.dim-1)
        tags = xdmf.read_meshtags(m, name="mesh_tags")
    return m, tags

def import_mesh(bp_file: str):
    mesh = adios4dolfinx.read_mesh(filename = Path(bp_file), comm=MPI.COMM_WORLD)
    tags = adios4dolfinx.read_meshtags(filename = Path(bp_file), mesh=mesh, meshtag_name="mesh_tags")
    return mesh, tags

def import_3d_mesh_with_facets(mesh_file: str, facet_file: str):
    """
    Read 3D perfusion mesh and facet meshtags.

    mesh_file: XDMF with the volume mesh (e.g. bioreactor.xdmf).
    facet_file: XDMF with boundary facet tags (e.g. mesh_tags_well1.xdmf),
                with marker 1 = inlet surface, 2 = outlet surface.
    """
    with io.XDMFFile(MPI.COMM_WORLD, mesh_file, "r") as xdmf:
        m = xdmf.read_mesh()

    with io.XDMFFile(MPI.COMM_WORLD, facet_file, "r") as xdmf:
        m.topology.create_connectivity(m.topology.dim,     m.topology.dim-1)
        m.topology.create_connectivity(m.topology.dim-1, m.topology.dim)
        m.topology.create_connectivity(m.topology.dim-1,   0)
        facet_tags = xdmf.read_meshtags(m, name="mesh_tags")  # adjust name if needed


    return m, facet_tags
# -----------------------------------------------------
# Pressure ADIOS importer
# -----------------------------------------------------
def import_pressure_data(mesh_obj, bp_file: str):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=MPI.COMM_WORLD,
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
# Velocity ADIOS importer (vector P1)
# -----------------------------------------------------
def import_velocity_data(mesh_obj, bp_file: str):
    ts = adios4dolfinx.read_timestamps(bp_file, comm=MPI.COMM_WORLD,
                                       function_name="f")
    print("Timestamps found for velocity data:", ts)
    P1_vec = element("Lagrange", mesh_obj.basix_cell(), 1, shape=(mesh_obj.geometry.dim,))
    V = fem.functionspace(mesh_obj, P1_vec)
    u_src = fem.Function(V)
    out = []
    for t in ts:
        adios4dolfinx.read_function(bp_file, u_src,
                                    name="f", time=t)
        u_src.x.scatter_forward()
        out.append(u_src.copy())
    return out   # list of Functions


# ===================================================================
#                PERFUSION SOLVER WITH FULL COUPLING
# ===================================================================
from scipy.spatial import cKDTree

class PerfusionSolver:
    def __init__(self, bioreactor_domain, facet_file,
                 mesh_inlet_file, mesh_outlet_file,
                 pres_inlet_file, pres_outlet_file,
                 velocity_inlet_file, velocity_outlet_file,
                 inlet_coord, outlet_coord,
                 branching_in_file=None, branching_out_file=None):

        # 3D perfusion domain + boundary facets
        self.mesh, self.facet_tags = import_3d_mesh_with_facets(
            bioreactor_domain, facet_file
        )

        # Marker conventions from mesh_tags_well*.xdmf
        self.arterial_concave_marker = 31
        self.venous_concave_marker = 32
        self.inlet_base_marker = 100
        self.outlet_base_marker = 200

        # 1D tree terminal territory meshes
        self.inlet_mesh,  self.inlet_tags  = import_mesh(mesh_inlet_file)
        self.outlet_mesh, self.outlet_tags = import_mesh(mesh_outlet_file)

        # Load last time-slice of pressures/velocities
        self.p_A_k = import_pressure_data(self.inlet_mesh,  pres_inlet_file)[-1]
        self.p_V_k = import_pressure_data(self.outlet_mesh, pres_outlet_file)[-1]
        self.u_A_k = import_velocity_data(self.inlet_mesh,  velocity_inlet_file)[-1]
        self.u_V_k = import_velocity_data(self.outlet_mesh, velocity_outlet_file)[-1]

        # Extract terminal (marker=2) coordinates + pressures + flows
        self._extract_terminal_data()

        # --- branching data for inlet/outlet trees (optional but recommended) ---
        self.branch_coords_in  = None
        self.branch_ids_in     = None
        self.branch_coords_out = None
        self.branch_ids_out    = None

        if branching_in_file is not None:
            (self.branch_coords_in,
             self.branch_ids_in) = self._load_terminal_branch_coords(branching_in_file)

        if branching_out_file is not None:
            (self.branch_coords_out,
             self.branch_ids_out) = self._load_terminal_branch_coords(branching_out_file)

        # store user-input coordinates for channel inlet/outlet (3D)
        self.inlet_coord  = np.asarray(inlet_coord,  dtype=float)
        self.outlet_coord = np.asarray(outlet_coord, dtype=float)

        # sample BC pressures from 1D pressure fields at those coordinates
        self.p_in_BC  = self._sample_pressure_at_point(self.p_A_k, self.inlet_coord)
        self.p_out_BC = self._sample_pressure_at_point(self.p_V_k, self.outlet_coord)

        if self.mesh.comm.rank == 0:
            print(f"Dirichlet BC pressures: p_in = {self.p_in_BC:.3e}, "
                  f"p_out = {self.p_out_BC:.3e}")

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

        p_art = fem.Constant(mesh, dfx.default_scalar_type(self.p_in_BC))
        p_ven = fem.Constant(mesh, dfx.default_scalar_type(self.p_out_BC))

        bc_art = fem.dirichletbc(p_art, art_dofs, W)
        bc_ven = fem.dirichletbc(p_ven, ven_dofs, W)

        return [bc_art, bc_ven]


    def _load_terminal_branch_coords(self, csv_path):
        """
        Read a branchingData_*.csv and return:
          - coords: (N_term, 3) array of distal coordinates
          - ids:    (N_term,) array of branch IDs (or 0..N_term-1 if not present)

        Assumptions (tweak if your headers differ):
          * child columns contain 'child' in their name
          * distal coordinates in one of:
              (x_dist, y_dist, z_dist) or
              (xDist,  yDist,  zDist)  or
              (x, y, z) as a fallback
          * branch ID column is one of:
              branchID, branch_id, id, ID
        """
        df = pd.read_csv(csv_path)

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

        # 3) branch IDs (optional but nice to keep)
        branch_id_col = None
        for cand in ["branchID", "branch_id", "id", "ID"]:
            if cand in df.columns:
                branch_id_col = cand
                break

        if branch_id_col is not None:
            ids = df.loc[mask_terminal, branch_id_col].to_numpy()
        else:
            ids = np.arange(coords.shape[0], dtype=int)

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

        # inlet pressures/velocities
        self.p_inlet = sample_field_at_points(self.p_A_k, self.x_inlet)
        self.u_inlet = sample_field_at_points(self.u_A_k, self.x_inlet)

        # outlet pressures/velocities
        self.p_outlet = sample_field_at_points(self.p_V_k, self.x_outlet)
        self.u_outlet = sample_field_at_points(self.u_V_k, self.x_outlet)

        print(f"Found {len(self.x_inlet)} inlet terminals")
        print(f"Found {len(self.x_outlet)} outlet terminals")

    def _build_reordering_indices(self, coords_terminals, branch_coords):
        """
        For each distal coordinate in branch_coords, find the nearest
        Darcy terminal coordinate in coords_terminals.

        Returns an integer index array idx such that
            coords_terminals[idx[i], :] ~ branch_coords[i, :]
        """
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
        vals = np.unique(self.facet_tags.values)
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

        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(fdim, 0)

        f_to_v = mesh.topology.connectivity(fdim, 0)
        X = mesh.geometry.x

        centroids = np.zeros((len(markers), 3), dtype=np.float64)
        for i, m in enumerate(markers):
            facets = self.facet_tags.find(int(m))
            if facets.size == 0:
                centroids[i, :] = np.nan
                continue
            # centroid per facet then average
            c = np.zeros((facets.size, 3), dtype=np.float64)
            for j, f in enumerate(facets):
                vs = f_to_v.links(int(f))
                c[j, :] = X[vs].mean(axis=0)
            centroids[i, :] = c.mean(axis=0)
        return centroids

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

    def _build_terminal_flux_form(self, v):
        """
        Build a Neumann-type RHS term that imposes terminal velocities *facet-wise*
        on the boundary facets tagged as terminal inlet/outlet faces.

        For each terminal marker m:
          RHS += ∫ (u_m · n) * v ds(m)

        Notes
        -----
        - u_m is sampled from velocity_checkpoint_{inlet|outlet}.bp on the
          corresponding 1D terminal coordinates and mapped to 3D terminal markers.
        """
        mesh = self.mesh
        n = ufl.FacetNormal(mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)

        inlet_marks, outlet_marks = self._get_terminal_marker_sets()
        inlet_c = self._marker_centroids(inlet_marks)
        outlet_c = self._marker_centroids(outlet_marks)

        idx_in  = self._assign_markers_to_terminals(inlet_c,  self.x_inlet)
        idx_out = self._assign_markers_to_terminals(outlet_c, self.x_outlet)

        # Store for later (interface output)
        self.inlet_terminal_markers  = inlet_marks
        self.outlet_terminal_markers = outlet_marks
        self.inlet_terminal_centroids  = inlet_c
        self.outlet_terminal_centroids = outlet_c

        L_flux = 0
        # Inlet terminals
        for j, m in enumerate(inlet_marks):
            if j >= len(idx_in):
                continue
            if len(self.u_inlet) == 0:
                U = np.zeros(mesh.geometry.dim, dtype=float)
            else:
                U = np.asarray(self.u_inlet[idx_in[j]], dtype=float).reshape(mesh.geometry.dim,)
            Uc = fem.Constant(mesh, dfx.default_scalar_type(U))
            L_flux += ufl.dot(Uc, n) * v * ds(int(m))

        # Outlet terminals
        for j, m in enumerate(outlet_marks):
            if j >= len(idx_out):
                continue
            if len(self.u_outlet) == 0:
                U = np.zeros(mesh.geometry.dim, dtype=float)
            else:
                U = np.asarray(self.u_outlet[idx_out[j]], dtype=float).reshape(mesh.geometry.dim,)
            Uc = fem.Constant(mesh, dfx.default_scalar_type(U))
            L_flux += ufl.dot(Uc, n) * v * ds(int(m))

        return L_flux

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
    # 
    
    def _build_terminal_bcs(self, W):
        mesh = self.mesh
        gdim = mesh.geometry.dim

        dof_coords = W.tabulate_dof_coordinates()
        tree = cKDTree(dof_coords)

        bcs = []

        def add_bc(xpts, pvals):
            if len(xpts) == 0:
                return
            _, dofs = tree.query(xpts)
            for d, pval in zip(dofs, pvals):
                bc_val = fem.Constant(mesh, dfx.default_scalar_type(pval))
                bc = fem.dirichletbc(bc_val,
                                     np.array([d], dtype=np.int32), W)
                bcs.append(bc)

        # add_bc(self.x_inlet,  self.p_inlet)
        # add_bc(self.x_outlet, self.p_outlet)
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

        problem = LinearProblem(
            a, L,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
            }
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
                return np.zeros(0)
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

        # These replace point-wise pressure/velocity/flow samples at terminals
        p_inlet_raw  = facet_average_scalar(inlet_marks)
        p_outlet_raw = facet_average_scalar(outlet_marks)

        u_inlet_raw  = facet_average_vector(inlet_marks)
        u_outlet_raw = facet_average_vector(outlet_marks)

        q_inlet_raw  = facet_flux(inlet_marks)
        q_outlet_raw = facet_flux(outlet_marks)

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
        # defaults: Darcy order (in case branchingData not provided)
        idx_in = idx_out = None

        if self.branch_coords_in is not None and len(self.branch_coords_in) > 0:
            idx_in = self._build_reordering_indices(coords_in, self.branch_coords_in)
        if self.branch_coords_out is not None and len(self.branch_coords_out) > 0:
            idx_out = self._build_reordering_indices(coords_out, self.branch_coords_out)

        if idx_in is not None:
            p_inlet   = p_inlet_raw[idx_in]
            u_inlet   = u_inlet_raw[idx_in, :]
            q_inlet   = q_inlet_raw[idx_in]
            divu_in   = divu_inlet_raw[idx_in]
            coords_in = self.x_inlet[idx_in, :]
            ids_in    = self.branch_ids_in
        else:
            p_inlet   = p_inlet_raw
            u_inlet   = u_inlet_raw
            q_inlet   = q_inlet_raw
            divu_in   = divu_inlet_raw
            coords_in = coords_in
            ids_in    = None

        if idx_out is not None:
            p_outlet   = p_outlet_raw[idx_out]
            u_outlet   = u_outlet_raw[idx_out, :]
            q_outlet   = q_outlet_raw[idx_out]
            divu_out   = divu_outlet_raw[idx_out]
            coords_out = self.x_outlet[idx_out, :]
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
            "divu_inlet":      divu_in.tolist(),
            "divu_outlet":     divu_out.tolist(),
            "cell_volume":     cell_vol,

            # total fluxes across inlet / outlet facet markers
            "q_artery_leak":   float(Q_art_leak),
            "q_venous_leak":   float(Q_ven_leak),

            # coords for mapping back to 1D (now in branchingData order if provided)
            "coords_inlet":    coords_in.tolist(),
            "coords_outlet":   coords_out.tolist(),
        }

        # optionally store branch IDs for sanity checking
        if ids_in is not None:
            interface_bc["branch_ids_inlet"] = ids_in.tolist()
        if ids_out is not None:
            interface_bc["branch_ids_outlet"] = ids_out.tolist()

        return interface_bc

    # -----------------------------------------------------------------------------
    # Permeability field K(x) in DG0 (same as your fixed version)
    # -----------------------------------------------------------------------------
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

        tdim = mesh.topology.dim
        mesh.topology.create_connectivity(tdim, 0)  # cells -> vertices
        c2v = mesh.topology.connectivity(tdim, 0)

        # local cells only
        nloc = mesh.topology.index_map(tdim).size_local
        cell_ids = np.arange(nloc, dtype=np.int32)

        # cell midpoints
        x = mesh.geometry.x
        mids = np.zeros((nloc, mesh.geometry.dim), dtype=np.float64)
        for c in cell_ids:
            vs = c2v.links(c)
            mids[c, :] = np.mean(x[vs, :], axis=0)

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

        Kfun.x.array[:nloc] = K_local
        Kfun.x.scatter_forward()
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

        # --- 1) Collect wall facet centroids ---
        mesh.topology.create_connectivity(fdim, 0)
        f2v = mesh.topology.connectivity(fdim, 0)

        mask = np.isin(facet_tags.values, wall_markers)
        wall_facets = facet_tags.indices[mask]

        x = mesh.geometry.x
        wall_centers = np.zeros((len(wall_facets), gdim), dtype=np.float64)
        for i, f in enumerate(wall_facets):
            vs = f2v.links(f)
            wall_centers[i, :] = x[vs, :].mean(axis=0)

        # Build KD-tree for distance queries
        tree = cKDTree(wall_centers)

        # --- 2) Cell midpoints ---
        mesh.topology.create_connectivity(tdim, 0)
        c2v = mesh.topology.connectivity(tdim, 0)

        nloc = mesh.topology.index_map(tdim).size_local
        cell_ids = np.arange(nloc, dtype=np.int32)

        mids = np.zeros((nloc, gdim), dtype=np.float64)
        for c in cell_ids:
            vs = c2v.links(c)
            mids[c, :] = x[vs, :].mean(axis=0)

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
        Kfun.x.array[:nloc] = K_local 
        Kfun.x.scatter_forward()
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
        a = ufl.inner((K_tensor) * ufl.grad(p), ufl.grad(v)) * ufl.dx

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

        L = q_src * v * ufl.dx + self._build_terminal_flux_form(v)
        # Dirichlet BCs on concave inlet/outlet surfaces
        bcs = self._build_wall_dirichlet_bcs(W)

        problem = LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
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
        problem = LinearProblem(
            a, L, bcs=bcs,
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps"
            }
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

        # project velocity
        V = fem.functionspace(mesh,
                                element("DG", mesh.basix_cell(), 0,
                                        shape=(mesh.geometry.dim,)))
        projector = Projector(V)
        u_h = projector(-Kfun * grad(p_h))

         # --- total leak flux across inlet / outlet facet markers ---
        # facet_tags: 1 = inlet surface, 2 = outlet surface
        n  = ufl.FacetNormal(mesh)
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=self.facet_tags)

        INLET_MARK  = getattr(self, "arterial_concave_marker", 31)  # arterial side (concave)
        OUTLET_MARK = getattr(self, "venous_concave_marker", 32)  # venous side (concave)

        # Flux = ∫_Γ (u · n) dS
        Q_art_leak = fem.assemble_scalar(
            fem.form(ufl.dot(u_h, n) * ds(INLET_MARK))
        )
        Q_ven_leak = fem.assemble_scalar(
            fem.form(ufl.dot(u_h, n) * ds(OUTLET_MARK))
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
    bioreactor_domain = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/bioreactor_well1.xdmf"
    mesh_inlet_file   = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/tagged_branches_inlet.bp"
    mesh_outlet_file  = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/tagged_branches_outlet.bp"
    pres_inlet_file   = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/pressure_checkpoint_inlet.bp"
    pres_outlet_file  = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/pressure_checkpoint_outlet.bp"
    velocity_inlet_file   = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/velocity_checkpoint_inlet.bp"
    velocity_outlet_file  = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/velocity_checkpoint_outlet.bp"
    facet_file       = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry/mesh_tags_well1.xdmf"

    # user-chosen coords near arterial / venous channel inlet/outlet
    inlet_coord  = np.array([-0.175, 0.9, .55])  # fill with your values
    outlet_coord = np.array([0.175, 0.9, .55])

    branching_in_file  = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022526/Run8_10branches/branchingData_0.csv"
    branching_out_file = "/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation/Forest_Output/0D_Output/022526/Run8_10branches/branchingData_1.csv"

    solver = PerfusionSolver(
        bioreactor_domain, facet_file,
        mesh_inlet_file, mesh_outlet_file,
        pres_inlet_file, pres_outlet_file,
        velocity_inlet_file, velocity_outlet_file,
        inlet_coord, outlet_coord,
        branching_in_file=branching_in_file,
        branching_out_file=branching_out_file,
    )
    solver.setup()
 
