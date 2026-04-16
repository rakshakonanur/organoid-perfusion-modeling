import argparse
import json
import os
from pathlib import Path
import re
import hashlib
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
    cache_bp = mesh_path.parent / f".darcy_mesh_facets_cache_{digest}.bp"

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


################################################################################
# Transport solver
################################################################################


class TransportSolver:
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
        D_value=1e-3,
        c_in_value=1.0,
        out_file="transport_c.xdmf",
    ):
        # 3D tissue mesh + facet tags from the same geometry pipeline as Darcy.
        self.mesh, self.facet_tags = import_3d_mesh_with_facets(
            bioreactor_domain, facet_file
        )

        # Darcy velocity field projected to nodal P1 values and written as VTU.
        self.velocity = self._load_velocity(vel_file)

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
        self.D_value = float(D_value)
        self.c_in_value = float(c_in_value)
        self.out_file = str(out_file)
        self.interface_bc_file = str(interface_bc_file)

    ###########################################################################
    # Velocity import
    ###########################################################################
    def _load_velocity(self, vtu_file: str):
        reader = vtk.vtkXMLUnstructuredGridReader()
        reader.SetFileName(str(vtu_file))
        reader.Update()

        grid = reader.GetOutput()
        velocity = grid.GetPointData().GetArray("f")
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

    def _inlet_concentration(self, t: float) -> float:
        """
        Time-dependent inlet concentration. Replace this with the desired profile.
        """
        t_on = 0.0
        t_off = 100.0
        if t_on <= t <= t_off:
            return self.c_in_value
        return 0.0

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
        D = fem.Constant(mesh, dfx.default_scalar_type(self.D_value))
        delta_t = fem.Constant(mesh, dfx.default_scalar_type(self.dt))

        c = ufl.TrialFunction(W)
        w = ufl.TestFunction(W)

        c_old = fem.Function(W)
        c_h = fem.Function(W)
        c_out = fem.Function(fem.functionspace(mesh, P1))
        c_old.x.array[:] = 0.0

        dx = ufl.Measure("dx", domain=domain)
        ds_tagged = ufl.Measure("ds", domain=domain, subdomain_data=self.facet_tags)
        dS_tagged = ufl.Measure("dS", domain=domain, subdomain_data=self.facet_tags)
        n = ufl.FacetNormal(domain)

        a_time = (c * w / delta_t) * dx
        a_advect = -ufl.dot(c * self.velocity, ufl.grad(w)) * dx
        a_diffuse = ufl.dot(D * ufl.grad(c), ufl.grad(w)) * dx
        a_exchange, L_exchange = self._build_terminal_exchange_forms(
            c, w, c_in, ds_tagged, dS_tagged
        )
        a_concave, L_concave = self._build_concave_exchange_forms(
            c, w, c_in, ds_tagged, n
        )

        a = a_time + a_advect + a_diffuse + a_exchange + a_concave
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

        out = io.XDMFFile(mesh.comm, self.out_file, "w")
        out.write_mesh(mesh)

        t = 0.0
        nt = int(np.round(self.T / self.dt))
        injected_cumulative = 0.0
        removed_cumulative = 0.0

        for step in range(nt):
            t += self.dt
            c_in.value = dfx.default_scalar_type(self._inlet_concentration(t))

            b.zeroEntries()
            assemble_vector(b, L_form)
            b.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )

            solver.solve(b, c_h.x.petsc_vec)
            c_h.x.scatter_forward()
            c_old.x.array[:] = c_h.x.array

            c_out.interpolate(c_h)
            out.write_function(c_out, t)

            inlet_rates, outlet_rates = self._compute_exchange_rates(
                c_h, c_in, ds_tagged, dS_tagged
            )
            concave_in_rates, concave_out_rates = self._compute_concave_exchange_rates(
                c_h, c_in, ds_tagged, n
            )
            inlet_total = float(np.sum(inlet_rates) + np.sum(concave_in_rates))
            outlet_total = float(np.sum(outlet_rates) + np.sum(concave_out_rates))
            terminal_in_total = float(np.sum(inlet_rates))
            terminal_out_total = float(np.sum(outlet_rates))
            concave_in_total = float(np.sum(concave_in_rates))
            concave_out_total = float(np.sum(concave_out_rates))
            injected_cumulative += inlet_total * self.dt
            removed_cumulative += outlet_total * self.dt

            if mesh.comm.rank == 0:
                print(
                    f"Step {step + 1}/{nt}   t={t:.3f}   c_in={float(c_in.value):.3f}   "
                    f"inj_rate={inlet_total:.6e}   out_rate={outlet_total:.6e}   "
                    f"term_in={terminal_in_total:.6e}   term_out={terminal_out_total:.6e}   "
                    f"wall_in={concave_in_total:.6e}   wall_out={concave_out_total:.6e}   "
                    f"inj_cum={injected_cumulative:.6e}   out_cum={removed_cumulative:.6e}"
                )

        out.close()


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
        help="Darcy velocity VTU containing point-data array 'f'.",
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
    ap.add_argument("--dt", type=float, default=10.0)
    ap.add_argument("--D-value", type=float, default=1e-8)
    ap.add_argument("--c-in-value", type=float, default=1.0)
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
