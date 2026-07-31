import argparse
import csv
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
from scipy import sparse
from scipy.sparse.linalg import spsolve
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


class TransportProgressReporter:
    """Write timestep diagnostics to CSV and refresh progress plots."""

    def __init__(self, output_file: str):
        output_path = Path(output_file).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stem = output_path.stem

        self.csv_path = output_path.with_name(f"{stem}_time_history.csv")
        self.rates_path = output_path.with_name(f"{stem}_rates.png")
        self.cumulative_path = output_path.with_name(f"{stem}_cumulative.png")
        self.below_critical_path = output_path.with_name(
            f"{stem}_organoid_below_critical.png"
        )

        # Import matplotlib only on the reporting rank. Agg works both on local
        # machines and on headless cluster nodes.
        cache_root = Path(os.environ.get("TMPDIR", "/tmp")) / "transport_matplotlib_cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_root))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        self._plt = plt
        self._rows = []
        self._csv_file = self.csv_path.open("w", newline="")
        self._csv_writer = None

        self._rates_fig, self._rates_ax = plt.subplots(figsize=(8.0, 5.0))
        self._rate_lines = {
            "injection_rate_mol_per_s": self._rates_ax.plot(
                [], [], label="Injection", linewidth=2.0
            )[0],
            "consumption_rate_mol_per_s": self._rates_ax.plot(
                [], [], label="Consumption", linewidth=2.0
            )[0],
            "outflow_rate_mol_per_s": self._rates_ax.plot(
                [], [], label="Outflow", linewidth=2.0
            )[0],
        }
        self._style_axis(
            self._rates_ax,
            "Oxygen rates",
            "Rate (mol/s)",
        )
        self._rates_ax.legend()
        self._rates_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        self._cumulative_fig, self._cumulative_ax = plt.subplots(figsize=(8.0, 5.0))
        self._cumulative_lines = {
            "injected_cumulative_mol": self._cumulative_ax.plot(
                [], [], label="Injected", linewidth=2.0
            )[0],
            "consumed_cumulative_mol": self._cumulative_ax.plot(
                [], [], label="Consumed", linewidth=2.0
            )[0],
            "removed_cumulative_mol": self._cumulative_ax.plot(
                [], [], label="Removed", linewidth=2.0
            )[0],
        }
        self._style_axis(
            self._cumulative_ax,
            "Cumulative oxygen quantities",
            "Amount (mol)",
        )
        self._cumulative_ax.legend()
        self._cumulative_ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

        self._critical_fig, self._critical_ax = plt.subplots(figsize=(8.0, 5.0))
        self._critical_line = self._critical_ax.plot(
            [], [], color="tab:red", label="Organoid below critical", linewidth=2.0
        )[0]
        self._style_axis(
            self._critical_ax,
            "Organoid below critical oxygen concentration",
            "Organoid volume below critical (%)",
        )
        self._critical_ax.set_ylim(0.0, 100.0)
        self._critical_ax.legend()

    @staticmethod
    def _style_axis(ax, title: str, ylabel: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    @staticmethod
    def _set_positive_limits(ax, times, values) -> None:
        xmax = max(float(times[-1]), 1.0)
        ymax = max((float(v) for series in values for v in series), default=0.0)
        ax.set_xlim(0.0, xmax)
        ax.set_ylim(0.0, 1.08 * ymax if ymax > 0.0 else 1.0)

    @staticmethod
    def _save_atomic(fig, path: Path) -> None:
        temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
        fig.savefig(temporary, format=path.suffix.lstrip("."), dpi=150, bbox_inches="tight")
        os.replace(temporary, path)

    def update(self, row: dict) -> None:
        row_copy = dict(row)
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row_copy.keys())
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row_copy)
        self._csv_file.flush()
        self._rows.append(row_copy)

        times = np.asarray([item["time_s"] for item in self._rows], dtype=float)

        rate_values = []
        for key, line in self._rate_lines.items():
            values = np.asarray([item[key] for item in self._rows], dtype=float)
            line.set_data(times, values)
            rate_values.append(values)
        self._set_positive_limits(self._rates_ax, times, rate_values)
        self._rates_ax.set_title(f"Oxygen rates (step {row_copy['step']})")
        self._save_atomic(self._rates_fig, self.rates_path)

        cumulative_values = []
        for key, line in self._cumulative_lines.items():
            values = np.asarray([item[key] for item in self._rows], dtype=float)
            line.set_data(times, values)
            cumulative_values.append(values)
        self._set_positive_limits(self._cumulative_ax, times, cumulative_values)
        self._cumulative_ax.set_title(
            f"Cumulative oxygen quantities (step {row_copy['step']})"
        )
        self._save_atomic(self._cumulative_fig, self.cumulative_path)

        below_critical_pct = 100.0 * np.asarray(
            [
                item["organoid_below_critical_volume_fraction"]
                for item in self._rows
            ],
            dtype=float,
        )
        self._critical_line.set_data(times, below_critical_pct)
        self._critical_ax.set_xlim(0.0, max(float(times[-1]), 1.0))
        self._critical_ax.set_title(
            "Organoid below critical oxygen concentration "
            f"(step {row_copy['step']})"
        )
        self._save_atomic(self._critical_fig, self.below_critical_path)

    def close(self) -> None:
        self._csv_file.close()
        self._plt.close(self._rates_fig)
        self._plt.close(self._cumulative_fig)
        self._plt.close(self._critical_fig)


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


class NetworkTransport1D:
    """Implicit conservative advection-diffusion solver on a branching graph.

    Each branchingData row is subdivided into a configurable number of finite
    volumes. Concentration is continuous at bifurcation nodes, while the
    assembled node balance conserves advective and diffusive species flux.
    Branch flow is reconstructed from the coupled terminal checkpoint flows,
    so it is consistent with the terminal rates used by the 3D model.
    """

    def __init__(
        self,
        csv_path: str,
        terminal_checkpoint_coords,
        terminal_checkpoint_flows,
        diffusivity: float,
        dt: float,
        cells_per_segment: int = 20,
        initial_concentration: float = 0.0,
        label: str = "network",
    ):
        self.csv_path = str(csv_path)
        self.label = str(label)
        self.diffusivity = float(diffusivity)
        self.dt = float(dt)
        self.cells_per_segment = int(max(1, cells_per_segment))
        if self.diffusivity < 0.0:
            raise ValueError(f"{self.label}: diffusivity must be non-negative")
        if self.dt <= 0.0:
            raise ValueError(f"{self.label}: timestep must be positive")

        with Path(csv_path).open(newline="") as stream:
            self.rows = list(csv.DictReader(stream))
        if not self.rows:
            raise ValueError(f"No branches found in {csv_path}")

        self.branch_indices = np.arange(len(self.rows), dtype=int)
        self.children = [self._child_indices(row) for row in self.rows]
        self.terminal_branches = np.asarray(
            [i for i, children in enumerate(self.children) if not children],
            dtype=int,
        )
        roots = [
            i
            for i, row in enumerate(self.rows)
            if self._optional_int(row.get("Parent")) is None
        ]
        if len(roots) != 1:
            raise ValueError(
                f"Expected one root branch in {csv_path}, found {len(roots)}"
            )
        self.root_branch = int(roots[0])

        proximal = np.asarray(
            [
                [
                    float(row["proximalCoordsX"]),
                    float(row["proximalCoordsY"]),
                    float(row["proximalCoordsZ"]),
                ]
                for row in self.rows
            ],
            dtype=float,
        )
        distal = np.asarray(
            [
                [
                    float(row["distalCoordsX"]),
                    float(row["distalCoordsY"]),
                    float(row["distalCoordsZ"]),
                ]
                for row in self.rows
            ],
            dtype=float,
        )
        self.proximal_coords = proximal
        self.distal_coords = distal
        self.terminal_coords = distal[self.terminal_branches]
        self.lengths = np.asarray(
            [float(row["Length"]) for row in self.rows], dtype=float
        )
        self.areas = np.pi * np.asarray(
            [float(row["Radius"]) for row in self.rows], dtype=float
        ) ** 2
        if np.any(self.lengths <= 0.0) or np.any(self.areas <= 0.0):
            raise ValueError(f"Non-positive branch length or area in {csv_path}")

        checkpoint_coords = np.asarray(terminal_checkpoint_coords, dtype=float).reshape((-1, 3))
        checkpoint_flows = np.asarray(terminal_checkpoint_flows, dtype=float).reshape(-1)
        if len(checkpoint_coords) != len(checkpoint_flows) or len(checkpoint_coords) == 0:
            raise ValueError(
                f"Missing terminal checkpoint coordinates/flows for {self.label}"
            )
        distances, nearest = cKDTree(checkpoint_coords).query(self.terminal_coords)
        nearest = np.asarray(nearest, dtype=int)
        if len(checkpoint_coords) != len(self.terminal_coords):
            raise ValueError(
                f"{self.label}: found {len(checkpoint_coords)} checkpoint terminals "
                f"but {len(self.terminal_coords)} terminal branches"
            )
        if len(np.unique(nearest)) != len(nearest) or np.any(~np.isfinite(distances)):
            raise ValueError(
                f"{self.label}: checkpoint-to-branch terminal mapping is not one-to-one"
            )
        terminal_flows = checkpoint_flows[nearest]
        self.branch_flows = np.full(len(self.rows), np.nan, dtype=float)
        self.branch_flows[self.terminal_branches] = terminal_flows
        self._accumulate_branch_flows(self.root_branch)
        if np.any(~np.isfinite(self.branch_flows)):
            missing = np.flatnonzero(~np.isfinite(self.branch_flows)).tolist()
            raise ValueError(f"Could not reconstruct branch flows {missing} in {csv_path}")

        self._build_discrete_graph()
        self.concentration = np.full(
            self.n_nodes, float(initial_concentration), dtype=float
        )
        self.terminal_boundary_concentrations = np.full(
            len(self.terminal_branches), float(initial_concentration), dtype=float
        )
        self.terminal_flux_rates = np.zeros(len(self.terminal_branches), dtype=float)
        self.terminal_diffusive_flux_rates = np.zeros(
            len(self.terminal_branches), dtype=float
        )
        self.root_flux_rate = 0.0
        self.storage_rate = 0.0
        self.balance_residual = 0.0

    @staticmethod
    def _optional_int(value):
        text = str(value or "").strip()
        if not text or text.lower() == "nan":
            return None
        number = float(text)
        if number < 0:
            return None
        return int(number)

    @classmethod
    def _child_indices(cls, row):
        out = []
        for name in ("Child1", "Child2"):
            value = cls._optional_int(row.get(name))
            if value is not None:
                out.append(value)
        return out

    def _accumulate_branch_flows(self, branch: int):
        if np.isfinite(self.branch_flows[branch]):
            return float(self.branch_flows[branch])
        children = self.children[branch]
        if not children:
            raise ValueError(f"Terminal branch {branch} has no checkpoint flow")
        value = float(sum(self._accumulate_branch_flows(child) for child in children))
        self.branch_flows[branch] = value
        return value

    def _build_discrete_graph(self):
        endpoint_nodes = {}

        def endpoint_node(raw_index):
            key = int(raw_index)
            if key not in endpoint_nodes:
                endpoint_nodes[key] = len(endpoint_nodes)
            return endpoint_nodes[key]

        branch_node_paths = []
        next_node = 0
        # Allocate all branching endpoints first, preserving shared bifurcations.
        for row in self.rows:
            endpoint_node(float(row["ProximalNodeIndex"]))
            endpoint_node(float(row["DistalNodeIndex"]))
        next_node = len(endpoint_nodes)

        for row in self.rows:
            start = endpoint_nodes[int(float(row["ProximalNodeIndex"]))]
            end = endpoint_nodes[int(float(row["DistalNodeIndex"]))]
            internal = list(
                range(next_node, next_node + self.cells_per_segment - 1)
            )
            next_node += len(internal)
            branch_node_paths.append([start, *internal, end])

        self.n_nodes = int(next_node)
        self.node_coords = np.zeros((self.n_nodes, 3), dtype=float)
        for branch, row in enumerate(self.rows):
            start = endpoint_nodes[int(float(row["ProximalNodeIndex"]))]
            end = endpoint_nodes[int(float(row["DistalNodeIndex"]))]
            self.node_coords[start] = self.proximal_coords[branch]
            self.node_coords[end] = self.distal_coords[branch]
            nodes = branch_node_paths[branch]
            for local_node, node in enumerate(nodes[1:-1], start=1):
                fraction = local_node / float(self.cells_per_segment)
                self.node_coords[node] = (
                    (1.0 - fraction) * self.proximal_coords[branch]
                    + fraction * self.distal_coords[branch]
                )
        self.node_volumes = np.zeros(self.n_nodes, dtype=float)
        self.edges = []
        self.edge_branch_indices = []
        self.branch_final_edge = {}
        self.branch_first_edge = {}
        for branch, nodes in enumerate(branch_node_paths):
            dx = float(self.lengths[branch]) / self.cells_per_segment
            area = float(self.areas[branch])
            flow = float(self.branch_flows[branch])
            conductance = float(self.diffusivity * area / dx)
            for local_edge, (left, right) in enumerate(zip(nodes[:-1], nodes[1:])):
                edge_index = len(self.edges)
                self.edges.append((int(left), int(right), flow, conductance))
                self.edge_branch_indices.append(int(branch))
                volume = area * dx
                self.node_volumes[left] += 0.5 * volume
                self.node_volumes[right] += 0.5 * volume
                if local_edge == 0:
                    self.branch_first_edge[branch] = edge_index
                if local_edge == self.cells_per_segment - 1:
                    self.branch_final_edge[branch] = edge_index

        root_row = self.rows[self.root_branch]
        self.root_node = endpoint_nodes[int(float(root_row["ProximalNodeIndex"]))]
        self.terminal_nodes = np.asarray(
            [
                endpoint_nodes[
                    int(float(self.rows[branch]["DistalNodeIndex"]))
                ]
                for branch in self.terminal_branches
            ],
            dtype=int,
        )
        self._transport_matrix = self._assemble_transport_matrix()
        self.edge_branch_indices = np.asarray(
            self.edge_branch_indices, dtype=np.int32
        )

    def _assemble_transport_matrix(self):
        rows = []
        cols = []
        values = []

        def add(row, col, value):
            rows.append(int(row))
            cols.append(int(col))
            values.append(float(value))

        for left, right, flow, conductance in self.edges:
            coeff_left, coeff_right = self._edge_flux_coefficients(
                flow, conductance
            )
            # F(left -> right) = coeff_left*c_left + coeff_right*c_right.
            add(left, left, coeff_left)
            add(left, right, coeff_right)
            add(right, left, -coeff_left)
            add(right, right, -coeff_right)
        return sparse.csr_matrix(
            (values, (rows, cols)), shape=(self.n_nodes, self.n_nodes)
        )

    @staticmethod
    def _bernoulli(value):
        """Stable Bernoulli function x/(exp(x)-1) for exponential fitting."""
        x = float(value)
        if abs(x) < 1.0e-6:
            return 1.0 - 0.5 * x + x * x / 12.0 - x**4 / 720.0
        if x > 50.0:
            return x * np.exp(-x)
        if x < -50.0:
            return -x
        return x / np.expm1(x)

    @classmethod
    def _edge_flux_coefficients(cls, flow, conductance):
        """Scharfetter-Gummel flux coefficients for one vascular interval."""
        q = float(flow)
        g = float(conductance)
        if g <= np.finfo(float).tiny:
            return max(q, 0.0), min(q, 0.0)
        peclet = q / g
        return g * cls._bernoulli(-peclet), -g * cls._bernoulli(peclet)

    def _edge_flux(self, edge_index, concentration):
        left, right, flow, conductance = self.edges[int(edge_index)]
        coeff_left, coeff_right = self._edge_flux_coefficients(flow, conductance)
        return float(
            coeff_left * concentration[left]
            + coeff_right * concentration[right]
        )

    def step(self, root_concentration: float, terminal_concentrations):
        terminal_values = np.asarray(terminal_concentrations, dtype=float).reshape(-1)
        if len(terminal_values) != len(self.terminal_nodes):
            raise ValueError(
                f"{self.label}: expected {len(self.terminal_nodes)} terminal "
                f"concentrations, received {len(terminal_values)}"
            )

        old = self.concentration.copy()
        mass_over_dt = self.node_volumes / self.dt
        matrix = self._transport_matrix + sparse.diags(mass_over_dt, format="csr")
        rhs = mass_over_dt * old

        dirichlet_nodes = np.concatenate(
            (np.asarray([self.root_node], dtype=int), self.terminal_nodes)
        )
        dirichlet_values = np.concatenate(
            (
                np.asarray([float(root_concentration)], dtype=float),
                terminal_values,
            )
        )
        matrix = matrix.tolil()
        for node, value in zip(dirichlet_nodes, dirichlet_values):
            matrix.rows[int(node)] = [int(node)]
            matrix.data[int(node)] = [1.0]
            rhs[int(node)] = float(value)
        concentration = np.asarray(spsolve(matrix.tocsr(), rhs), dtype=float)
        if np.any(~np.isfinite(concentration)):
            raise RuntimeError(f"Non-finite concentration in {self.label} solve")

        self.concentration = concentration
        self.terminal_boundary_concentrations = terminal_values.copy()
        self.terminal_flux_rates = np.asarray(
            [
                self._edge_flux(self.branch_final_edge[int(branch)], concentration)
                for branch in self.terminal_branches
            ],
            dtype=float,
        )
        self.root_flux_rate = self._edge_flux(
            self.branch_first_edge[self.root_branch], concentration
        )
        dynamic_nodes = np.ones(self.n_nodes, dtype=bool)
        dynamic_nodes[dirichlet_nodes] = False
        self.storage_rate = float(
            np.dot(
                self.node_volumes[dynamic_nodes],
                concentration[dynamic_nodes] - old[dynamic_nodes],
            )
            / self.dt
        )
        # Positive root flux enters the graph; positive terminal flux leaves it.
        self.balance_residual = float(
            self.root_flux_rate
            - np.sum(self.terminal_flux_rates)
            - self.storage_rate
        )
        return self.terminal_flux_rates.copy()

    def step_with_diffusive_terminal_flux(
        self, root_concentration: float, terminal_diffusive_flux_rates
    ):
        """Advance with natural terminal diffusion and unknown terminal values.

        Flux is positive from the network into the 3D domain.  At terminal
        ``i``, the external total species flux is

            F_i = Q_i C_i + F_diff,i.

        The signed flow ``Q_i`` is positive for arterial network outflow and
        negative for venous network inflow.  The supplied diffusive flux is
        the value returned by the preceding 3D solve; it is zero initially.
        """
        diffusive_rates = np.asarray(
            terminal_diffusive_flux_rates, dtype=float
        ).reshape(-1)
        if len(diffusive_rates) != len(self.terminal_nodes):
            raise ValueError(
                f"{self.label}: expected {len(self.terminal_nodes)} terminal "
                f"diffusive fluxes, received {len(diffusive_rates)}"
            )

        old = self.concentration.copy()
        mass_over_dt = self.node_volumes / self.dt
        matrix = (
            self._transport_matrix
            + sparse.diags(mass_over_dt, format="csr")
        ).tolil()
        rhs = mass_over_dt * old

        # Terminal control-volume balance: internal transport is already in
        # _transport_matrix; add the external advective outflow implicitly and
        # move the prescribed outward diffusive flux to the right-hand side.
        for terminal_index, (node, branch) in enumerate(
            zip(self.terminal_nodes, self.terminal_branches)
        ):
            matrix[int(node), int(node)] += float(self.branch_flows[int(branch)])
            rhs[int(node)] -= float(diffusive_rates[terminal_index])

        root = int(self.root_node)
        matrix.rows[root] = [root]
        matrix.data[root] = [1.0]
        rhs[root] = float(root_concentration)
        concentration = np.asarray(spsolve(matrix.tocsr(), rhs), dtype=float)
        if np.any(~np.isfinite(concentration)):
            raise RuntimeError(f"Non-finite concentration in {self.label} solve")

        self.concentration = concentration
        self.terminal_boundary_concentrations = concentration[
            self.terminal_nodes
        ].copy()
        self.terminal_diffusive_flux_rates = diffusive_rates.copy()
        terminal_flows = self.branch_flows[self.terminal_branches]
        self.terminal_flux_rates = (
            terminal_flows * self.terminal_boundary_concentrations
            + self.terminal_diffusive_flux_rates
        )
        self.root_flux_rate = self._edge_flux(
            self.branch_first_edge[self.root_branch], concentration
        )
        dynamic_nodes = np.ones(self.n_nodes, dtype=bool)
        dynamic_nodes[root] = False
        self.storage_rate = float(
            np.dot(
                self.node_volumes[dynamic_nodes],
                concentration[dynamic_nodes] - old[dynamic_nodes],
            )
            / self.dt
        )
        self.balance_residual = float(
            self.root_flux_rate
            - np.sum(self.terminal_flux_rates)
            - self.storage_rate
        )
        return self.terminal_boundary_concentrations.copy()


class NetworkTransportVTKWriter:
    """Write a ParaView-readable PVD/VTP time series for one 1D network."""

    def __init__(self, network: NetworkTransport1D, pvd_path: str | Path):
        self.network = network
        self.pvd_path = Path(pvd_path)
        self.pvd_path.parent.mkdir(parents=True, exist_ok=True)
        self._datasets = []
        for old_path in self.pvd_path.parent.glob(f"{self.pvd_path.stem}_*.vtp"):
            old_path.unlink()
        if self.pvd_path.exists():
            self.pvd_path.unlink()

    @staticmethod
    def _add_array(data_attributes, name, values, vtk_type=None):
        array = numpy_to_vtk(
            np.ascontiguousarray(values), deep=True, array_type=vtk_type
        )
        array.SetName(str(name))
        data_attributes.AddArray(array)

    def write(self, time_value: float):
        network = self.network
        polydata = vtk.vtkPolyData()

        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(np.ascontiguousarray(network.node_coords), deep=True))
        polydata.SetPoints(points)

        lines = vtk.vtkCellArray()
        for left, right, _flow, _conductance in network.edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, int(left))
            line.GetPointIds().SetId(1, int(right))
            lines.InsertNextCell(line)
        polydata.SetLines(lines)

        self._add_array(
            polydata.GetPointData(),
            "oxygen_concentration_mol_per_cm3",
            np.asarray(network.concentration, dtype=np.float64),
        )
        self._add_array(
            polydata.GetPointData(),
            "control_volume_cm3",
            np.asarray(network.node_volumes, dtype=np.float64),
        )

        branch_ids = network.edge_branch_indices
        edge_flows = network.branch_flows[branch_ids]
        edge_areas = network.areas[branch_ids]
        edge_radii = np.sqrt(edge_areas / np.pi)
        edge_velocity = edge_flows / edge_areas
        edge_dx = network.lengths[branch_ids] / network.cells_per_segment
        edge_peclet = np.abs(edge_velocity) * edge_dx / max(
            network.diffusivity, np.finfo(float).tiny
        )
        self._add_array(
            polydata.GetCellData(),
            "branch_id",
            branch_ids,
            vtk.VTK_INT,
        )
        for name, values in (
            ("volumetric_flow_rate_cm3_per_s", edge_flows),
            ("cross_sectional_area_cm2", edge_areas),
            ("radius_cm", edge_radii),
            ("axial_velocity_cm_per_s", edge_velocity),
            ("cell_peclet_number", edge_peclet),
        ):
            self._add_array(
                polydata.GetCellData(), name, np.asarray(values, dtype=np.float64)
            )

        time_array = vtk.vtkDoubleArray()
        time_array.SetName("time_s")
        time_array.SetNumberOfTuples(1)
        time_array.SetValue(0, float(time_value))
        polydata.GetFieldData().AddArray(time_array)

        step = len(self._datasets)
        vtp_path = self.pvd_path.with_name(
            f"{self.pvd_path.stem}_{step:06d}.vtp"
        )
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(str(vtp_path))
        writer.SetInputData(polydata)
        if writer.Write() != 1:
            raise RuntimeError(f"Failed to write 1D network visualization {vtp_path}")

        self._datasets.append((float(time_value), vtp_path.name))
        self._write_pvd_index()

    def _write_pvd_index(self):
        root = ET.Element(
            "VTKFile",
            type="Collection",
            version="0.1",
            byte_order="LittleEndian",
        )
        collection = ET.SubElement(root, "Collection")
        for time_value, filename in self._datasets:
            ET.SubElement(
                collection,
                "DataSet",
                timestep=f"{time_value:.16g}",
                group="",
                part="0",
                file=filename,
            )
        try:
            ET.indent(root, space="  ")
        except AttributeError:
            pass
        temporary = self.pvd_path.with_name(f".{self.pvd_path.name}.tmp")
        ET.ElementTree(root).write(
            temporary, encoding="utf-8", xml_declaration=True
        )
        os.replace(temporary, self.pvd_path)


class NetworkTerminalHistoryWriter:
    """Stream terminal coupling rows to disk and flush every timestep."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.csv_path.open("w", newline="")
        self._writer = None

    def update(self, rows):
        rows = list(rows)
        if not rows:
            return
        if self._writer is None:
            self._writer = csv.DictWriter(
                self._stream, fieldnames=list(rows[0].keys())
            )
            self._writer.writeheader()
        self._writer.writerows(rows)
        self._stream.flush()

    def close(self):
        self._stream.close()


def _vtk_output_path(path: str | Path) -> Path:
    """Use a ParaView-friendly PVD path next to a requested output path."""
    p = Path(path)
    return p if p.suffix.lower() == ".pvd" else p.with_suffix(".pvd")


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
    # Literature-based defaults, converted to cgs.
    # Paper/SI units:
    #   D_org = 1.0e-9 m^2/s, D_M = 1.0e-9 m^2/s, D_water = 3.0e-9 m^2/s
    #   C_in = 0.2 mol/m^3, C_crit = 4.0e-2 mol/m^3, k_m = 5.0e-3 mol/m^3
    #   V_max = 2.0e-3 mol/(m^3 s)
    # cgs conversions used here:
    #   m^2/s -> cm^2/s: multiply by 1e4
    #   mol/m^3 -> mol/cm^3: divide by 1e6
    PAPER_D_ORG_CM2_PER_S = 1.07e-5
    PAPER_D_GEL_CM2_PER_S = 1.0e-5
    PAPER_D_WATER_CM2_PER_S = 3.0e-5
    PAPER_C_IN_MOL_PER_CM3 = 2.0e-7
    PAPER_C_CRIT_MOL_PER_CM3 = 4.0e-8
    PAPER_KM_MOL_PER_CM3 = 5e-9
    # PAPER_KM_MOL_PER_CM3 = 1.5e-7
    PAPER_VMAX_MOL_PER_CM3_S = 2e-9
    # PAPER_VMAX_MOL_PER_CM3_S = 6.93e-9
    PAPER_CRITICAL_TRANSITION_WIDTH_MOL_PER_CM3 = 0.5 * PAPER_C_CRIT_MOL_PER_CM3
    VELOCITY_SCALE_FACTOR = 82.5065
    # VELOCITY_SCALE_FACTOR = 1.0
    # VELOCITY_SCALE_FACTOR = 44500.0

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
        diffusion_only=False,
        marker_32_concentration=None,
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
        critical_transition_width=None,
        critical_output_file="",
        consumption_output_file="",
        critical_summary_file="",
        velocity_debug_output_file="",
        output_format="vtk",
        branching_in_file="",
        branching_out_file="",
        couple_1d_transport=False,
        network_cells_per_segment=20,
        network_coupling_relaxation=1.0,
        network_initial_concentration=0.0,
    ):
        # 3D tissue mesh + facet tags from the same geometry pipeline as Darcy.
        self.mesh, self.facet_tags = import_3d_mesh_with_facets(
            bioreactor_domain, facet_file
        )

        self.diffusion_only = bool(diffusion_only)
        self.skip_1d = bool(skip_1d or self.diffusion_only)
        self.couple_1d_transport = bool(couple_1d_transport)
        if self.couple_1d_transport and self.skip_1d:
            raise ValueError(
                "--coupled-1d-transport cannot be combined with --skip-1d or "
                "--diffusion-only"
            )
        self.network_cells_per_segment = int(network_cells_per_segment)
        if self.network_cells_per_segment < 1:
            raise ValueError("--network-cells-per-segment must be at least 1")
        self.network_coupling_relaxation = float(network_coupling_relaxation)
        if not 0.0 < self.network_coupling_relaxation <= 1.0:
            raise ValueError("--network-coupling-relaxation must be in (0, 1]")
        self.network_initial_concentration = float(network_initial_concentration)
        if self.network_initial_concentration < 0.0:
            raise ValueError("--network-initial-concentration must be non-negative")
        self.concave_exchange_mode = (
            "dirichlet"
            if self.diffusion_only
            else str(concave_exchange_mode).strip().lower()
        )
        if self.concave_exchange_mode not in {
            "velocity",
            "uniform",
            "dirichlet",
            "combined",
        }:
            raise ValueError(
                "concave_exchange_mode must be 'velocity', 'uniform', "
                f"'dirichlet', or 'combined', got {concave_exchange_mode!r}"
            )

        self._velocity_point_coords = None
        self._velocity_local_point_ids = None
        if self.diffusion_only:
            # Do not require or read a Darcy velocity file in diffusion-only mode.
            self.velocity = self._zero_velocity_function()
            self.velocity_times = np.asarray([0.0], dtype=np.float64)
            self._velocity_series = None
        else:
            # Darcy velocity field projected to nodal P1 values.
            self.velocity, self.velocity_times, self._velocity_series = (
                self._load_velocity(vel_file)
            )

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
        self.concave_exchange = []
        self.inlet_exchange = []
        self.outlet_exchange = []
        self.x_inlet = np.zeros((0, 3), dtype=float)
        self.x_outlet = np.zeros((0, 3), dtype=float)
        self.q_inlet_vals = np.zeros((0,), dtype=float)
        self.q_outlet_vals = np.zeros((0,), dtype=float)
        self.branch_coords_in = np.zeros((0, 3), dtype=float)
        self.branch_normals_in = np.zeros((0, 3), dtype=float)
        self.branch_coords_out = np.zeros((0, 3), dtype=float)
        self.branch_normals_out = np.zeros((0, 3), dtype=float)
        self.branching_in_file = ""
        self.branching_out_file = ""
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

            branching_in_file = self._resolve_terminal_branching_file(
                branching_in_file, mesh_inlet_file, "branchingData_0.csv"
            )
            branching_out_file = self._resolve_terminal_branching_file(
                branching_out_file, mesh_outlet_file, "branchingData_1.csv"
            )
            self.branching_in_file = str(branching_in_file)
            self.branching_out_file = str(branching_out_file)
            self.branch_coords_in, self.branch_normals_in = (
                self._load_terminal_branch_directions(branching_in_file)
            )
            self.branch_coords_out, self.branch_normals_out = (
                self._load_terminal_branch_directions(branching_out_file)
            )
            self._extract_terminal_data()
            self._build_terminal_exchange_data()

        if self.concave_exchange_mode == "uniform":
            self._build_concave_exchange_data(interface_bc_file)
        elif self.concave_exchange_mode in {"velocity", "combined"}:
            self._report_concave_velocity_fluxes()
        elif self.mesh.comm.rank == 0:
            print(
                "[transport] Diffusion-only mode: zero velocity, no 1D terminals, "
                "and prescribed concentrations on concave markers 31 and 32.",
                flush=True,
            )

        self.T = float(T)
        self.dt = float(dt)
        self.D_value = float(
            self.PAPER_D_ORG_CM2_PER_S if D_value is None else D_value
        )
        self.c_in_value = float(
            self.PAPER_C_IN_MOL_PER_CM3 if c_in_value is None else c_in_value
        )
        self.marker_32_concentration = (
            0.0
            if marker_32_concentration is None
            and self.concave_exchange_mode == "combined"
            else None
            if marker_32_concentration is None
            else float(marker_32_concentration)
        )
        if (
            self.marker_32_concentration is not None
            and self.concave_exchange_mode not in {"dirichlet", "combined"}
        ):
            raise ValueError(
                "--marker-32-concentration is only used with --diffusion-only "
                "or --concave-exchange-mode combined"
            )
        if self.concave_exchange_mode == "combined" and self.mesh.comm.rank == 0:
            print(
                "[transport] Combined concave boundary: Darcy upwind advection "
                f"plus Dirichlet diffusion; marker 31 concentration={self.c_in_value:.6e}, "
                f"marker 32 concentration={self.marker_32_concentration:.6e} mol/cm^3.",
                flush=True,
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
        self.critical_transition_width = float(
            self.PAPER_CRITICAL_TRANSITION_WIDTH_MOL_PER_CM3
            if critical_transition_width is None
            else critical_transition_width
        )
        self.critical_output_file = str(critical_output_file).strip()
        self.consumption_output_file = str(consumption_output_file).strip()
        self.critical_summary_file = str(critical_summary_file).strip()
        self.velocity_debug_output_file = str(velocity_debug_output_file).strip()
        self.output_format = str(output_format or "vtk").strip().lower()
        if self.output_format not in {"vtk", "xdmf", "both"}:
            raise ValueError(
                "--output-format must be one of 'vtk', 'xdmf', or 'both'; "
                f"got {self.output_format!r}."
            )

        self.network_inlet = None
        self.network_outlet = None
        self.inlet_marker_to_network_terminal = np.zeros(0, dtype=int)
        self.outlet_marker_to_network_terminal = np.zeros(0, dtype=int)
        self._coupled_terminal_flux_constants = {"inlet": [], "outlet": []}
        self._coupled_terminal_concentration_constants = {
            "inlet": [], "outlet": []
        }
        self._coupled_terminal_diffusive_rates = {
            "inlet": np.zeros(len(self.inlet_exchange), dtype=float),
            "outlet": np.zeros(len(self.outlet_exchange), dtype=float),
        }
        self._coupled_terminal_marker_rates = {
            "inlet": np.zeros(len(self.inlet_exchange), dtype=float),
            "outlet": np.zeros(len(self.outlet_exchange), dtype=float),
        }
        self.network_history = []
        if self.couple_1d_transport:
            self._initialize_coupled_network_transport()

    ###########################################################################
    # Organoid mask + cgs oxygen constants
    ###########################################################################
    @staticmethod
    def _smooth_cutoff_numpy(values, threshold, half_width):
        vals = np.asarray(values, dtype=float)
        if half_width <= 0.0:
            return (vals >= threshold).astype(float)

        s = (vals - threshold) / float(half_width)
        out = np.zeros_like(s, dtype=float)
        out[s >= 1.0] = 1.0
        mid = np.abs(s) < 1.0
        sm = s[mid]
        # COMSOL-style flc2hs quintic transition: C^2-continuous at +/-1.
        out[mid] = 0.5 + (15.0 / 16.0) * sm - (5.0 / 8.0) * sm**3 + (3.0 / 16.0) * sm**5
        return out

    @staticmethod
    def _smooth_cutoff_ufl(c_value, threshold, half_width):
        if half_width <= 0.0:
            return ufl.conditional(ufl.ge(c_value, threshold), 1.0, 0.0)

        s = (c_value - threshold) / float(half_width)
        poly = 0.5 + (15.0 / 16.0) * s - (5.0 / 8.0) * s**3 + (3.0 / 16.0) * s**5
        return ufl.conditional(
            ufl.ge(s, 1.0),
            1.0,
            ufl.conditional(ufl.le(s, -1.0), 0.0, poly),
        )

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
                f"Ccrit_width={self.critical_transition_width:.6e} mol/cm^3, "
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
            "marker_31_concentration_mol_per_cm3": float(self.c_in_value),
            "marker_32_concentration_mol_per_cm3": float(
                self.c_in_value
                if self.marker_32_concentration is None
                else self.marker_32_concentration
            ),
            "critical_oxygen_mol_per_cm3": float(self.critical_oxygen_concentration),
            "critical_transition_width_mol_per_cm3": float(self.critical_transition_width),
            "michaelis_menten_km_mol_per_cm3": float(self.mm_km),
            "michaelis_menten_vmax_mol_per_cm3_s": float(self.mm_vmax),
            "michaelis_menten_enabled": bool(self.enable_mm_consumption),
            "diffusion_only": bool(self.diffusion_only),
            "skip_1d": bool(self.skip_1d),
            "coupled_1d_transport": bool(self.couple_1d_transport),
            "network_cells_per_segment": int(self.network_cells_per_segment),
            "network_coupling_relaxation": float(
                self.network_coupling_relaxation
            ),
            "network_initial_concentration_mol_per_cm3": float(
                self.network_initial_concentration
            ),
            "network_coupling_scheme": (
                "directional_arterial_1d_concentration_imposition_without_diffusive_feedback_to_3d_to_venous_1d"
                if self.couple_1d_transport
                else "disabled"
            ),
            "arterial_network_root_concentration": "marker_31_c_in",
            "venous_network_root_concentration": "marker_32",
            "concave_exchange_mode": str(self.concave_exchange_mode),
            "arterial_synthetic_terminal_bc": (
                "disabled_skip_1d"
                if self.skip_1d
                else "1d_terminal_concentration_nitsche_imposition_plus_advective_flux_no_1d_diffusive_feedback"
                if self.couple_1d_transport
                else "danckwerts_prescribed_total_flux"
            ),
            "arterial_synthetic_terminal_total_flux": (
                "Q*C_terminal plus 3D Nitsche concentration-imposition flux; Nitsche reaction is reported as applied 3D flux but not returned to 1D"
                if self.couple_1d_transport
                else "(Q/A)*c_feed; zero total species flux when Q=0"
            ),
            "venous_synthetic_terminal_bc": (
                "disabled_skip_1d"
                if self.skip_1d
                else "advective_local_concentration_outflow_to_1d"
                if self.couple_1d_transport
                else "advective_local_concentration_outflow"
            ),
            "organoid_volume_cm3": float(org_vol),
        }
        return chi, D_field

    ###########################################################################
    # Velocity import
    ###########################################################################
    def _zero_velocity_function(self):
        V = fem.functionspace(
            self.mesh,
            element(
                "Lagrange",
                self.mesh.basix_cell(),
                1,
                shape=(self.mesh.geometry.dim,),
            ),
        )
        velocity = fem.Function(V)
        velocity.x.array[:] = 0.0
        velocity.x.scatter_forward()
        return velocity

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
        u.x.array[:] = self._scale_velocity_flat(flat)
        u.x.scatter_forward()
        return u

    def _scale_velocity_flat(self, flat_values: np.ndarray) -> np.ndarray:
        return self.VELOCITY_SCALE_FACTOR * np.asarray(flat_values, dtype=np.float64)

    def _scale_flow_values(self, flow_values: np.ndarray) -> np.ndarray:
        return self.VELOCITY_SCALE_FACTOR * np.asarray(flow_values, dtype=np.float64)

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
        u.x.array[:] = self._scale_velocity_flat(flat)
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
            u.x.array[:] = self._scale_velocity_flat(values.reshape(-1))
            u.x.scatter_forward()
            return u

        tree = cKDTree(coords)
        _, nn = tree.query(local_coords)
        self._velocity_point_coords = coords
        self._velocity_local_point_ids = np.asarray(nn, dtype=int)
        mapped = values[np.asarray(nn, dtype=int)]
        u.x.array[:] = self._scale_velocity_flat(mapped.reshape(-1))
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
        self.velocity.x.array[:] = self._scale_velocity_flat(arr)
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
    @staticmethod
    def _resolve_terminal_branching_file(explicit_path, terminal_mesh_file, filename):
        """Resolve branching metadata next to the organoid geometry by default."""
        if explicit_path:
            path = Path(explicit_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Missing terminal branching data: {path}")
            return str(path)

        mesh_path = Path(terminal_mesh_file).expanduser()
        candidates = [
            mesh_path.parent / filename,
            mesh_path.parent.parent / filename,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        return ""

    @staticmethod
    def _load_terminal_branch_directions(csv_path):
        """Return terminal distal coordinates and normalized proximal-to-distal axes."""
        if not csv_path:
            return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

        with Path(csv_path).open(newline="") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

        child_cols = [name for name in fieldnames if "child" in name.lower()]
        terminal_rows = [
            row
            for row in rows
            if not child_cols
            or all(str(row.get(name, "") or "").strip() == "" for name in child_cols)
        ]

        coordinate_sets = [
            ("distalCoordsX", "distalCoordsY", "distalCoordsZ"),
            ("x_dist", "y_dist", "z_dist"),
            ("xDist", "yDist", "zDist"),
            ("xd", "yd", "zd"),
            ("x", "y", "z"),
        ]
        coord_cols = next(
            (cols for cols in coordinate_sets if all(name in fieldnames for name in cols)),
            None,
        )
        if coord_cols is None:
            raise ValueError(f"Could not find distal coordinate columns in {csv_path}")

        coords = np.asarray(
            [[float(row[name]) for name in coord_cols] for row in terminal_rows],
            dtype=float,
        ).reshape((-1, 3))

        if all(
            name in fieldnames
            for name in ("proximalCoordsX", "proximalCoordsY", "proximalCoordsZ")
        ):
            proximal = np.asarray(
                [
                    [
                        float(row["proximalCoordsX"]),
                        float(row["proximalCoordsY"]),
                        float(row["proximalCoordsZ"]),
                    ]
                    for row in terminal_rows
                ],
                dtype=float,
            ).reshape((-1, 3))
            directions = coords - proximal
        elif all(name in fieldnames for name in ("W1", "W2", "W3")):
            # The generated branchingData files store W from distal to
            # proximal. Embedded-facet selection needs the opposite direction:
            # from the branch interior toward its tissue-facing distal end.
            directions = -np.asarray(
                [
                    [float(row["W1"]), float(row["W2"]), float(row["W3"])]
                    for row in terminal_rows
                ],
                dtype=float,
            ).reshape((-1, 3))
        else:
            raise ValueError(
                f"Could not find W1/W2/W3 or proximal coordinates in {csv_path}"
            )

        norms = np.linalg.norm(directions, axis=1)
        valid = np.isfinite(norms) & (norms > 0.0)
        if not np.all(valid):
            bad = np.flatnonzero(~valid).tolist()
            raise ValueError(f"Invalid terminal directions in {csv_path}, rows {bad}")
        directions = directions / norms[:, None]
        return coords, directions

    def _initialize_coupled_network_transport(self):
        """Construct both 1D transport trees and their 3D-marker mappings."""
        if not self.branching_in_file or not self.branching_out_file:
            raise ValueError(
                "Coupled 1D transport requires arterial and venous branchingData CSVs"
            )

        common = {
            "diffusivity": self.D_water,
            "dt": self.dt,
            "cells_per_segment": self.network_cells_per_segment,
            "initial_concentration": self.network_initial_concentration,
        }
        self.network_inlet = NetworkTransport1D(
            self.branching_in_file,
            self.x_inlet,
            self.q_inlet_vals,
            label="arterial",
            **common,
        )
        self.network_outlet = NetworkTransport1D(
            self.branching_out_file,
            self.x_outlet,
            self.q_outlet_vals,
            label="venous",
            **common,
        )

        _, _, inlet_centroids, outlet_centroids, _, _ = (
            self._get_terminal_marker_data()
        )
        self.inlet_marker_to_network_terminal = self._map_marker_centroids_to_network(
            inlet_centroids, self.network_inlet, "arterial"
        )
        self.outlet_marker_to_network_terminal = self._map_marker_centroids_to_network(
            outlet_centroids, self.network_outlet, "venous"
        )

        if self.mesh.comm.rank == 0:
            print(
                "[transport] Coupled 1D transport enabled: "
                f"arterial branches={len(self.network_inlet.rows)}, "
                f"terminals={len(self.network_inlet.terminal_nodes)}; "
                f"venous branches={len(self.network_outlet.rows)}, "
                f"terminals={len(self.network_outlet.terminal_nodes)}; "
                f"cells/branch={self.network_cells_per_segment}, "
                f"D_water={self.D_water:.6e} cm^2/s, "
                f"initial lumen concentration={self.network_initial_concentration:.6e}.",
                flush=True,
            )

    @staticmethod
    def _map_marker_centroids_to_network(marker_centroids, network, label):
        centroids = np.asarray(marker_centroids, dtype=float).reshape((-1, 3))
        if len(centroids) != len(network.terminal_coords):
            raise ValueError(
                f"{label}: found {len(centroids)} 3D terminal markers but "
                f"{len(network.terminal_coords)} terminal network branches"
            )
        distances, indices = cKDTree(network.terminal_coords).query(centroids)
        indices = np.asarray(indices, dtype=int)
        if len(np.unique(indices)) != len(indices):
            raise ValueError(f"{label}: terminal marker-to-branch mapping is not one-to-one")
        branch_scale = max(float(np.max(network.lengths)), np.finfo(float).eps)
        if np.any(~np.isfinite(distances)) or np.max(distances) > 0.25 * branch_scale:
            raise ValueError(
                f"{label}: 3D terminal marker is too far from its network endpoint "
                f"(maximum distance {np.max(distances):.6e})"
            )
        return indices

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
        self.q_inlet_vals = self._scale_flow_values(
            sample_field_at_points(self.q_inlet_fun, self.x_inlet)
        )
        self.q_outlet_vals = self._scale_flow_values(
            sample_field_at_points(self.q_outlet_fun, self.x_outlet)
        )

        if self.mesh.comm.rank == 0:
            print(f"Found {len(self.x_inlet)} inlet terminals")
            print(f"Found {len(self.x_outlet)} outlet terminals")

    ###########################################################################
    # Darcy-style marker utilities
    ###########################################################################
    def _get_terminal_marker_sets(self):
        local_vals = np.unique(np.asarray(self.facet_tags.values, dtype=np.int32))
        gathered = self.mesh.comm.allgather(local_vals.tolist())
        vals = np.asarray(
            sorted({int(value) for rank_values in gathered for value in rank_values}),
            dtype=np.int32,
        )
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
            local_sum = np.zeros(3, dtype=np.float64)
            local_count = np.asarray([0], dtype=np.int64)
            for facet in facets:
                verts = f_to_v.links(int(facet))
                local_sum += X[verts].mean(axis=0)
                local_count[0] += 1
            global_sum = np.zeros_like(local_sum)
            global_count = np.zeros_like(local_count)
            mesh.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
            mesh.comm.Allreduce(local_count, global_count, op=MPI.SUM)
            if int(global_count[0]) <= 0:
                centroids[i, :] = np.nan
            else:
                centroids[i, :] = global_sum / float(global_count[0])
        return centroids

    def _assign_markers_to_terminals(self, marker_centroids, terminal_coords):
        if marker_centroids.size == 0 or terminal_coords.size == 0:
            return np.zeros((0,), dtype=int)
        tree = cKDTree(terminal_coords)
        _, idx = tree.query(marker_centroids)
        return idx.astype(int)

    @staticmethod
    def _assign_marker_directions(marker_centroids, branch_coords, branch_directions):
        if (
            marker_centroids.size == 0
            or branch_coords.size == 0
            or branch_directions.size == 0
        ):
            return np.zeros((len(marker_centroids), 3), dtype=float)
        tree = cKDTree(branch_coords)
        _, idx = tree.query(marker_centroids)
        return np.asarray(branch_directions[np.asarray(idx, dtype=int)], dtype=float)

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

    def _marker_global_facet_counts(self, marker: int):
        exterior, interior = self._split_marker_facets(int(marker))
        n_exterior = int(self.mesh.comm.allreduce(int(exterior.size), op=MPI.SUM))
        n_interior = int(self.mesh.comm.allreduce(int(interior.size), op=MPI.SUM))
        return n_exterior, n_interior

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

        inlet_marks, outlet_marks, inlet_c, outlet_c, idx_in, idx_out = (
            self._get_terminal_marker_data()
        )
        inlet_directions = self._assign_marker_directions(
            inlet_c, self.branch_coords_in, self.branch_normals_in
        )
        outlet_directions = self._assign_marker_directions(
            outlet_c, self.branch_coords_out, self.branch_normals_out
        )

        def build_data(markers, idx_map, flows, directions, role):
            out = []
            for j, marker in enumerate(markers):
                if j >= len(idx_map):
                    continue

                n_ext, n_int = self._marker_global_facet_counts(int(marker))
                area_ext = (
                    self._global_scalar(one * ds(int(marker))) if n_ext > 0 else 0.0
                )
                area_int = (
                    self._global_scalar(one * dS(int(marker))) if n_int > 0 else 0.0
                )
                area_total = area_ext + area_int

                q_raw = float(flows[idx_map[j]]) if len(flows) else 0.0
                q_mag = abs(q_raw)
                g = q_mag / area_total if area_total > 0.0 else 0.0
                direction = (
                    np.asarray(directions[j], dtype=float)
                    if j < len(directions)
                    else np.zeros(3, dtype=float)
                )
                direction_norm = float(np.linalg.norm(direction))
                if area_int > 0.0 and (
                    not np.isfinite(direction_norm) or direction_norm <= 0.0
                ):
                    raise RuntimeError(
                        f"Embedded {role} marker {int(marker)} has no valid "
                        "branching-data direction. Pass the corresponding "
                        "--branching-*-file option."
                    )
                if direction_norm > 0.0:
                    direction = direction / direction_norm

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
                        "distal_direction": direction,
                    }
                )
            return out

        self.inlet_exchange = build_data(
            inlet_marks, idx_in, self.q_inlet_vals, inlet_directions, "inlet"
        )
        self.outlet_exchange = build_data(
            outlet_marks, idx_out, self.q_outlet_vals, outlet_directions, "outlet"
        )

        if mesh.comm.rank == 0:
            print(
                "Configured inlet terminal exchange "
                "(Danckwerts prescribed total species flux):"
            )
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

    def _velocity_debug_rows(self, time_value, ds_tagged, dS_tagged):
        """Audit the exact scaled velocity field used by the transport forms."""
        mesh = self.mesh
        n = ufl.FacetNormal(mesh)
        dx = ufl.Measure("dx", domain=mesh.ufl_domain())
        speed = ufl.sqrt(ufl.dot(self.velocity, self.velocity))
        volume = self._global_scalar(1.0 * dx)
        mean_speed = self._global_scalar(speed * dx) / volume if volume > 0.0 else 0.0

        gdim = mesh.geometry.dim
        nodal_velocity = np.asarray(self.velocity.x.array, dtype=float).reshape((-1, gdim))
        local_speed = np.linalg.norm(nodal_velocity, axis=1)
        local_min = float(np.min(local_speed)) if local_speed.size else float("inf")
        local_max = float(np.max(local_speed)) if local_speed.size else float("-inf")
        min_speed = float(mesh.comm.allreduce(local_min, op=MPI.MIN))
        max_speed = float(mesh.comm.allreduce(local_max, op=MPI.MAX))

        fieldnames = (
            "time_s",
            "surface_type",
            "role",
            "marker",
            "area_cm2",
            "checkpoint_network_to_3d_flow_cm3_per_s",
            "velocity_network_to_3d_flow_cm3_per_s",
            "velocity_to_checkpoint_ratio",
            "mean_speed_cm_per_s",
            "min_nodal_speed_cm_per_s",
            "max_nodal_speed_cm_per_s",
            "velocity_scale_factor",
        )

        def row(surface_type, role, marker, area, expected, measured):
            ratio = measured / expected if expected != 0.0 else float("nan")
            values = {
                "time_s": float(time_value),
                "surface_type": str(surface_type),
                "role": str(role),
                "marker": int(marker),
                "area_cm2": float(area),
                "checkpoint_network_to_3d_flow_cm3_per_s": float(expected),
                "velocity_network_to_3d_flow_cm3_per_s": float(measured),
                "velocity_to_checkpoint_ratio": float(ratio),
                "mean_speed_cm_per_s": float(mean_speed),
                "min_nodal_speed_cm_per_s": float(min_speed),
                "max_nodal_speed_cm_per_s": float(max_speed),
                "velocity_scale_factor": float(self.VELOCITY_SCALE_FACTOR),
            }
            return {name: values[name] for name in fieldnames}

        rows = [row("bulk", "domain", -1, volume, 0.0, 0.0)]

        # Positive values use one convention everywhere: flow into the 3D domain.
        for role, marker in (
            ("arterial_concave", int(self.arterial_concave_marker)),
            ("venous_concave", int(self.venous_concave_marker)),
        ):
            area = self._global_scalar(1.0 * ds_tagged(marker))
            flow_into_3d = -self._global_scalar(
                ufl.dot(self.velocity, n) * ds_tagged(marker)
            )
            rows.append(row("concave", role, marker, area, 0.0, flow_into_3d))

        for role, exchange in (
            ("arterial_terminal", self.inlet_exchange),
            ("venous_terminal", self.outlet_exchange),
        ):
            for data in exchange:
                marker = int(data["marker"])
                flow_into_3d = 0.0
                if data["area_exterior"] > 0.0:
                    flow_into_3d -= self._global_scalar(
                        ufl.dot(self.velocity, n) * ds_tagged(marker)
                    )
                if data["area_interior"] > 0.0:
                    direction = fem.Constant(
                        mesh,
                        dfx.default_scalar_type(
                            tuple(float(value) for value in data["distal_direction"])
                        ),
                    )
                    use_minus = ufl.gt(ufl.dot(n("+"), direction), 0.0)
                    tissue_outward_flux = ufl.conditional(
                        use_minus,
                        ufl.dot(self.velocity("-"), n("-")),
                        ufl.dot(self.velocity("+"), n("+")),
                    )
                    flow_into_3d -= self._global_scalar(
                        tissue_outward_flux * dS_tagged(marker)
                    )
                rows.append(
                    row(
                        "synthetic_terminal",
                        role,
                        marker,
                        float(data["area_total"]),
                        float(data["flow_raw"]),
                        flow_into_3d,
                    )
                )
        return rows

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
                float(self._scale_flow_values([iface_data.get("q_artery_leak", 0.0)])[0]),
            ),
            (
                "venous_concave",
                int(self.venous_concave_marker),
                float(self._scale_flow_values([iface_data.get("q_venous_leak", 0.0)])[0]),
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

    def _build_arterial_terminal_danckwerts_form(
        self, w, c_feed, ds_tagged, dS_tagged, n
    ):
        """Prescribe the total arterial species flux on synthetic terminals.

        With the terminal direction pointing from the vessel into the tissue,
        the Danckwerts condition is

            (u c - D grad(c)) . d = (Q/A) c_feed.

        After integration by parts, this is a natural total-flux condition and
        contributes only the prescribed right-hand-side flux below.  Adding a
        separate Dirichlet/Nitsche diffusion term would supply oxygen in excess
        of Q*c_feed and would no longer be a Danckwerts condition.

        Embedded dS terminals use the branching-oriented distal tissue trace.
        Exterior ds terminals have only one trace.  When Q is zero, this pure
        Danckwerts condition becomes zero total species flux.
        """
        mesh = self.mesh
        L_danckwerts = 0

        for data in self.inlet_exchange:
            marker = int(data["marker"])
            speed = fem.Constant(
                mesh, dfx.default_scalar_type(data["flux_density"])
            )
            prescribed_total_flux = speed * c_feed

            if data["area_exterior"] > 0.0:
                L_danckwerts += prescribed_total_flux * w * ds_tagged(marker)

            if data["area_interior"] > 0.0:
                direction = fem.Constant(
                    mesh,
                    dfx.default_scalar_type(
                        tuple(float(value) for value in data["distal_direction"])
                    ),
                )
                use_minus = ufl.gt(ufl.dot(n("+"), direction), 0.0)
                w_tissue = ufl.conditional(use_minus, w("-"), w("+"))
                L_danckwerts += (
                    prescribed_total_flux * w_tissue * dS_tagged(marker)
                )

        return L_danckwerts

    def _build_terminal_exchange_forms(
        self, c, w, c_in, D, ds_tagged, dS_tagged, n, h, alpha
    ):
        """
        Assemble facet exchange terms.

        In coupled 1D mode, arterial terminal concentrations are imposed on the
        3D tissue-facing trace with Nitsche terms and also set the advective
        injection Q*c_terminal. The resulting 3D Nitsche reaction is not fed
        back as an additional 1D terminal loss. Venous terminals use local
        advective outflow; their 3D trace is sampled after the solve and imposed
        as the downstream 1D inflow concentration. Outside coupled mode, arterial
        terminals use a Danckwerts prescribed-total-flux condition and venous
        terminals use local advective outflow.

        Exterior markers use ds(marker); interior embedded markers use the
        branching-oriented distal trace on dS(marker).
        """
        mesh = self.mesh

        if self.couple_1d_transport:
            a_exchange = 0
            L_exchange = 0
            self._coupled_terminal_concentration_constants = {
                "inlet": [], "outlet": []
            }
            self._coupled_terminal_flux_constants = {"inlet": [], "outlet": []}

            # Arterial network is upstream: solve it first and impose its
            # terminal concentrations on the 3D tissue-facing trace, while also
            # prescribing the advective species flux Q*c_terminal.
            for data in self.inlet_exchange:
                marker = int(data["marker"])
                boundary_concentration = fem.Constant(
                    mesh,
                    dfx.default_scalar_type(self.network_initial_concentration),
                )
                self._coupled_terminal_concentration_constants["inlet"].append(
                    boundary_concentration
                )
                signed_speed = fem.Constant(
                    mesh,
                    dfx.default_scalar_type(
                        float(data["flow_raw"]) / float(data["area_total"])
                        if data["area_total"] > 0.0 else 0.0
                    ),
                )
                if data["area_exterior"] > 0.0:
                    L_exchange += (
                        signed_speed * boundary_concentration * w
                        + alpha * D / h * boundary_concentration * w
                        - D * ufl.dot(ufl.grad(w), n) * boundary_concentration
                    ) * ds_tagged(marker)
                    a_exchange += (
                        alpha * D / h * c * w
                        - D * ufl.dot(ufl.grad(c), n) * w
                        - D * ufl.dot(ufl.grad(w), n) * c
                    ) * ds_tagged(marker)
                if data["area_interior"] > 0.0:
                    direction = fem.Constant(
                        mesh,
                        dfx.default_scalar_type(
                            tuple(float(value) for value in data["distal_direction"])
                        ),
                    )
                    use_minus = ufl.gt(ufl.dot(n("+"), direction), 0.0)
                    w_tissue = ufl.conditional(use_minus, w("-"), w("+"))
                    grad_w_tissue = ufl.conditional(
                        use_minus, ufl.grad(w)("-"), ufl.grad(w)("+")
                    )
                    normal_tissue = ufl.conditional(use_minus, n("-"), n("+"))
                    D_tissue = ufl.conditional(use_minus, D("-"), D("+"))
                    h_tissue = ufl.conditional(use_minus, h("-"), h("+"))
                    L_exchange += (
                        signed_speed * boundary_concentration * w_tissue
                        + alpha * D_tissue / h_tissue
                        * boundary_concentration * w_tissue
                        - D_tissue * ufl.dot(grad_w_tissue, normal_tissue)
                        * boundary_concentration
                    ) * dS_tagged(marker)
                    a_exchange += (
                        ufl.conditional(
                            use_minus,
                            alpha * D("-") / h("-") * c("-") * w("-"),
                            alpha * D("+") / h("+") * c("+") * w("+"),
                        )
                        - ufl.conditional(
                            use_minus,
                            D("-") * ufl.dot(ufl.grad(c)("-"), n("-")) * w("-"),
                            D("+") * ufl.dot(ufl.grad(c)("+"), n("+")) * w("+"),
                        )
                        - ufl.conditional(
                            use_minus,
                            D("-") * ufl.dot(ufl.grad(w)("-"), n("-")) * c("-"),
                            D("+") * ufl.dot(ufl.grad(w)("+"), n("+")) * c("+"),
                        )
                    ) * dS_tagged(marker)

            # Venous network is downstream. Do not impose a venous Dirichlet /
            # Nitsche concentration on the 3D terminal face here; that acted as
            # an artificial drain when the lagged 1D terminal concentration was
            # low. Instead, remove species advectively from the local 3D trace
            # and pass the solved 3D trace to the venous 1D network afterward.
            for data in self.outlet_exchange:
                marker = int(data["marker"])
                outflow_speed = fem.Constant(
                    mesh,
                    dfx.default_scalar_type(
                        abs(float(data["flow_raw"])) / float(data["area_total"])
                        if data["area_total"] > 0.0 else 0.0
                    ),
                )
                if data["area_exterior"] > 0.0:
                    a_exchange += outflow_speed * c * w * ds_tagged(marker)
                if data["area_interior"] > 0.0:
                    direction = fem.Constant(
                        mesh,
                        dfx.default_scalar_type(
                            tuple(float(value) for value in data["distal_direction"])
                        ),
                    )
                    use_minus = ufl.gt(ufl.dot(n("+"), direction), 0.0)
                    cw_tissue = ufl.conditional(
                        use_minus, c("-") * w("-"), c("+") * w("+")
                    )
                    a_exchange += outflow_speed * cw_tissue * dS_tagged(marker)
            return a_exchange, L_exchange

        a_exchange = 0
        L_exchange = self._build_arterial_terminal_danckwerts_form(
            w, c_in, ds_tagged, dS_tagged, n
        )

        for data in self.outlet_exchange:
            marker = int(data["marker"])
            g = fem.Constant(mesh, dfx.default_scalar_type(data["flux_density"]))

            if data["area_exterior"] > 0.0:
                a_exchange += g * c * w * ds_tagged(marker)
            if data["area_interior"] > 0.0:
                direction = fem.Constant(
                    mesh,
                    dfx.default_scalar_type(
                        tuple(float(value) for value in data["distal_direction"])
                    ),
                )
                use_minus = ufl.gt(ufl.dot(n("+"), direction), 0.0)
                cw_tissue = ufl.conditional(
                    use_minus, c("-") * w("-"), c("+") * w("+")
                )
                a_exchange += g * cw_tissue * dS_tagged(marker)

        return a_exchange, L_exchange

    def _advance_coupled_networks(
        self, time_value: float, arterial_root_concentration: float,
        venous_root_concentration: float
    ):
        """Solve the upstream arterial tree before the 3D transport step."""
        if not self.couple_1d_transport:
            return

        network = self.network_inlet
        marker_to_terminal = self.inlet_marker_to_network_terminal
        marker_diffusive_rates = np.zeros(len(self.inlet_exchange), dtype=float)
        self._coupled_terminal_diffusive_rates["inlet"] = marker_diffusive_rates
        network_diffusive_rates = np.empty(len(network.terminal_nodes), dtype=float)
        for marker_index, network_terminal in enumerate(marker_to_terminal):
            network_diffusive_rates[int(network_terminal)] = (
                marker_diffusive_rates[marker_index]
            )
        network.step_with_diffusive_terminal_flux(
            float(arterial_root_concentration), network_diffusive_rates
        )

        marker_total_rates = np.empty(len(self.inlet_exchange), dtype=float)
        for marker_index, (data, constant) in enumerate(
            zip(
                self.inlet_exchange,
                self._coupled_terminal_concentration_constants["inlet"],
            )
        ):
            network_terminal = int(marker_to_terminal[marker_index])
            terminal_concentration = float(
                network.terminal_boundary_concentrations[network_terminal]
            )
            constant.value = dfx.default_scalar_type(terminal_concentration)
            marker_total_rates[marker_index] = (
                float(data["flow_raw"]) * terminal_concentration
            )
        self._coupled_terminal_marker_rates["inlet"] = marker_total_rates

    def _terminal_face_concentrations(self, c_h, exchange, ds_tagged, dS_tagged):
        """Area-average the tissue-facing 3D concentration for each terminal."""
        n = ufl.FacetNormal(self.mesh)
        values = []
        for data in exchange:
            marker = int(data["marker"])
            integral = 0.0
            if data["area_exterior"] > 0.0:
                integral += self._global_scalar(c_h * ds_tagged(marker))
            if data["area_interior"] > 0.0:
                direction = fem.Constant(
                    self.mesh,
                    dfx.default_scalar_type(
                        tuple(float(value) for value in data["distal_direction"])
                    ),
                )
                c_tissue = ufl.conditional(
                    ufl.gt(ufl.dot(n("+"), direction), 0.0), c_h("-"), c_h("+")
                )
                integral += self._global_scalar(c_tissue * dS_tagged(marker))
            area = float(data["area_total"])
            values.append(integral / area if area > 0.0 else 0.0)
        return np.asarray(values, dtype=float)

    def _terminal_diffusive_flux_rates(
        self, c_h, D, exchange, concentration_constants,
        ds_tagged, dS_tagged, h, alpha
    ):
        """Return Nitsche numerical diffusion, positive network -> 3D."""
        n = ufl.FacetNormal(self.mesh)
        rates = []
        for data, boundary_concentration in zip(
            exchange, concentration_constants
        ):
            marker = int(data["marker"])
            rate = 0.0
            if data["area_exterior"] > 0.0:
                outward_numerical_flux = (
                    -D * ufl.dot(ufl.grad(c_h), n)
                    + alpha * D / h * (c_h - boundary_concentration)
                )
                rate -= self._global_scalar(
                    outward_numerical_flux * ds_tagged(marker)
                )
            if data["area_interior"] > 0.0:
                direction = fem.Constant(
                    self.mesh,
                    dfx.default_scalar_type(
                        tuple(float(value) for value in data["distal_direction"])
                    ),
                )
                use_minus = ufl.gt(ufl.dot(n("+"), direction), 0.0)
                c_tissue = ufl.conditional(
                    use_minus, c_h("-"), c_h("+")
                )
                grad_c_tissue = ufl.conditional(
                    use_minus, ufl.grad(c_h)("-"), ufl.grad(c_h)("+")
                )
                normal_tissue = ufl.conditional(
                    use_minus, n("-"), n("+")
                )
                D_tissue = ufl.conditional(use_minus, D("-"), D("+"))
                h_tissue = ufl.conditional(use_minus, h("-"), h("+"))
                outward_numerical_flux = (
                    -D_tissue * ufl.dot(grad_c_tissue, normal_tissue)
                    + alpha * D_tissue / h_tissue
                    * (c_tissue - boundary_concentration)
                )
                rate -= self._global_scalar(
                    outward_numerical_flux * dS_tagged(marker)
                )
            rates.append(rate)
        return np.asarray(rates, dtype=float)

    def _update_coupled_terminal_diffusive_fluxes(
        self, c_h, D, ds_tagged, dS_tagged, h, alpha,
        time_value, arterial_root_concentration, venous_root_concentration
    ):
        """Update coupled terminal diagnostics after the 3D solve."""
        if not self.couple_1d_transport:
            return
        iteration = int(np.rint(float(time_value) / self.dt))

        # Arterial: concentration came from 1D and is imposed on the 3D terminal
        # faces to maintain concentration continuity.  We still do not feed the
        # 3D Nitsche reaction back as an extra 1D terminal loss; the 1D artery is
        # treated as an upstream concentration source.  The reaction is still a
        # real 3D boundary flux, so include it in the reported/applied 3D rate.
        arterial_trace = self._terminal_face_concentrations(
            c_h, self.inlet_exchange, ds_tagged, dS_tagged
        )
        arterial_imposed_diffusion = np.zeros(len(self.inlet_exchange), dtype=float)
        arterial_returned_diffusion = self._terminal_diffusive_flux_rates(
            c_h,
            D,
            self.inlet_exchange,
            self._coupled_terminal_concentration_constants["inlet"],
            ds_tagged,
            dS_tagged,
            h,
            alpha,
        )
        arterial_next_diffusion = np.zeros(len(self.inlet_exchange), dtype=float)
        self._coupled_terminal_diffusive_rates["inlet"] = arterial_next_diffusion
        arterial_applied_rates = np.empty(len(self.inlet_exchange), dtype=float)
        for marker_index, data in enumerate(self.inlet_exchange):
            network_terminal = int(
                self.inlet_marker_to_network_terminal[marker_index]
            )
            terminal_concentration = float(
                self.network_inlet.terminal_boundary_concentrations[
                    network_terminal
                ]
            )
            advective_rate = float(data["flow_raw"] * terminal_concentration)
            network_total_rate = advective_rate
            applied_3d_rate = advective_rate + arterial_returned_diffusion[
                marker_index
            ]
            arterial_applied_rates[marker_index] = applied_3d_rate
            self.network_history.append(
                {
                    "coupling_iteration": iteration,
                    "time_s": float(time_value),
                    "network": "arterial",
                    "marker": int(data["marker"]),
                    "network_terminal_index": network_terminal,
                    "root_concentration_mol_per_cm3": float(
                        arterial_root_concentration
                    ),
                    "terminal_concentration_mol_per_cm3": terminal_concentration,
                    "three_d_trace_concentration_mol_per_cm3": float(
                        arterial_trace[marker_index]
                    ),
                    "concentration_continuity_residual_mol_per_cm3": float(
                        arterial_trace[marker_index] - terminal_concentration
                    ),
                    "advective_network_to_3d_flux_mol_per_s": advective_rate,
                    "imposed_diffusive_network_to_3d_flux_mol_per_s": float(
                        arterial_imposed_diffusion[marker_index]
                    ),
                    "returned_diffusive_network_to_3d_flux_mol_per_s": float(
                        arterial_returned_diffusion[marker_index]
                    ),
                    "next_relaxed_diffusive_network_to_3d_flux_mol_per_s": float(
                        arterial_next_diffusion[marker_index]
                    ),
                    "raw_network_to_3d_flux_mol_per_s": float(network_total_rate),
                    "applied_network_to_3d_flux_mol_per_s": float(applied_3d_rate),
                    "network_root_influx_mol_per_s": float(
                        self.network_inlet.root_flux_rate
                    ),
                    "network_storage_rate_mol_per_s": float(
                        self.network_inlet.storage_rate
                    ),
                    "network_balance_residual_mol_per_s": float(
                        self.network_inlet.balance_residual
                    ),
                }
            )
        self._coupled_terminal_marker_rates["inlet"] = arterial_applied_rates

        # Venous: the previous 1D terminal concentration was imposed on the 3D
        # trace before this solve. The current 3D trace is then imposed as the
        # downstream 1D inflow concentration.
        venous_trace = self._terminal_face_concentrations(
            c_h, self.outlet_exchange, ds_tagged, dS_tagged
        )
        venous_imposed_diffusion = np.zeros(len(self.outlet_exchange), dtype=float)
        venous_terminal_values = (
            self.network_outlet.terminal_boundary_concentrations.copy()
        )
        for marker_index, network_terminal in enumerate(
            self.outlet_marker_to_network_terminal
        ):
            venous_terminal_values[int(network_terminal)] = max(
                float(venous_trace[marker_index]), 0.0
            )
        venous_network_total_rates = self.network_outlet.step(
            float(venous_root_concentration), venous_terminal_values
        )

        venous_returned_diffusion = np.empty(
            len(self.outlet_exchange), dtype=float
        )
        venous_applied_rates = np.empty(len(self.outlet_exchange), dtype=float)
        for marker_index, data in enumerate(self.outlet_exchange):
            network_terminal = int(
                self.outlet_marker_to_network_terminal[marker_index]
            )
            terminal_concentration = float(
                self.network_outlet.terminal_boundary_concentrations[
                    network_terminal
                ]
            )
            advective_rate = float(data["flow_raw"] * terminal_concentration)
            venous_returned_diffusion[marker_index] = (
                float(venous_network_total_rates[network_terminal])
                - advective_rate
            )
            venous_applied_rates[marker_index] = advective_rate

        venous_next_diffusion = np.zeros(len(self.outlet_exchange), dtype=float)
        self._coupled_terminal_diffusive_rates["outlet"] = venous_next_diffusion
        self._coupled_terminal_marker_rates["outlet"] = venous_applied_rates

        for marker_index, data in enumerate(self.outlet_exchange):
            network_terminal = int(
                self.outlet_marker_to_network_terminal[marker_index]
            )
            terminal_concentration = float(
                self.network_outlet.terminal_boundary_concentrations[
                    network_terminal
                ]
            )
            advective_rate = float(data["flow_raw"] * terminal_concentration)
            network_total_rate = float(
                venous_network_total_rates[network_terminal]
            )
            self.network_history.append(
                {
                    "coupling_iteration": iteration,
                    "time_s": float(time_value),
                    "network": "venous",
                    "marker": int(data["marker"]),
                    "network_terminal_index": network_terminal,
                    "root_concentration_mol_per_cm3": float(
                        venous_root_concentration
                    ),
                    "terminal_concentration_mol_per_cm3": terminal_concentration,
                    "three_d_trace_concentration_mol_per_cm3": float(
                        venous_trace[marker_index]
                    ),
                    "concentration_continuity_residual_mol_per_cm3": float(
                        venous_trace[marker_index] - terminal_concentration
                    ),
                    "advective_network_to_3d_flux_mol_per_s": advective_rate,
                    "imposed_diffusive_network_to_3d_flux_mol_per_s": float(
                        venous_imposed_diffusion[marker_index]
                    ),
                    "returned_diffusive_network_to_3d_flux_mol_per_s": float(
                        venous_returned_diffusion[marker_index]
                    ),
                    "next_relaxed_diffusive_network_to_3d_flux_mol_per_s": float(
                        venous_next_diffusion[marker_index]
                    ),
                    "raw_network_to_3d_flux_mol_per_s": network_total_rate,
                    "applied_network_to_3d_flux_mol_per_s": float(
                        venous_applied_rates[marker_index]
                    ),
                    "network_root_influx_mol_per_s": float(
                        self.network_outlet.root_flux_rate
                    ),
                    "network_storage_rate_mol_per_s": float(
                        self.network_outlet.storage_rate
                    ),
                    "network_balance_residual_mol_per_s": float(
                        self.network_outlet.balance_residual
                    ),
                }
            )

    def _build_concave_exchange_forms(
        self, c, w, c_marker_31, c_marker_32, ds_tagged, n
    ):
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

        # Diffusion-only supply is imposed separately through the SIPG
        # Dirichlet boundary terms, not as an advective source/sink.
        if self.concave_exchange_mode == "dirichlet":
            return a_exchange, L_exchange

        if self.concave_exchange_mode == "velocity":
            qn = ufl.dot(self.velocity, n)
            q_in = 0.5 * (-qn + abs(qn))
            q_out = 0.5 * (qn + abs(qn))
            for marker in (
                int(self.arterial_concave_marker),
                int(self.venous_concave_marker),
            ):
                L_exchange += q_in * c_marker_31 * w * ds_tagged(marker)
                a_exchange += q_out * c * w * ds_tagged(marker)
            return a_exchange, L_exchange

        if self.concave_exchange_mode == "combined":
            qn = ufl.dot(self.velocity, n)
            q_in = 0.5 * (-qn + abs(qn))
            q_out = 0.5 * (qn + abs(qn))
            for marker, boundary_concentration in (
                (int(self.arterial_concave_marker), c_marker_31),
                (int(self.venous_concave_marker), c_marker_32),
            ):
                # Upwind advection: exterior concentration on inflow and the
                # tissue-side concentration on outflow.
                L_exchange += (
                    q_in * boundary_concentration * w * ds_tagged(marker)
                )
                a_exchange += q_out * c * w * ds_tagged(marker)
            return a_exchange, L_exchange

        mesh = self.mesh
        for data in self.concave_exchange:
            marker = int(data["marker"])
            g = fem.Constant(mesh, dfx.default_scalar_type(data["flux_density"]))
            if data["mode"] == "source":
                L_exchange += g * c_marker_31 * w * ds_tagged(marker)
            elif data["mode"] == "sink":
                a_exchange += g * c * w * ds_tagged(marker)

        return a_exchange, L_exchange

    def _build_concave_dirichlet_diffusion_forms(
        self, c, w, c_marker_31, c_marker_32, D, ds_tagged, n, h, alpha
    ):
        """SIPG/Nitsche concentration conditions on the two concave walls."""
        a_dirichlet = 0
        L_dirichlet = 0
        for marker, boundary_concentration in (
            (int(self.arterial_concave_marker), c_marker_31),
            (int(self.venous_concave_marker), c_marker_32),
        ):
            a_dirichlet += (
                alpha * D / h * c * w
                - D * ufl.dot(ufl.grad(c), n) * w
                - D * ufl.dot(ufl.grad(w), n) * c
            ) * ds_tagged(marker)
            L_dirichlet += (
                alpha * D / h * boundary_concentration * w
                - D * ufl.dot(ufl.grad(w), n) * boundary_concentration
            ) * ds_tagged(marker)
        return a_dirichlet, L_dirichlet

    def _compute_exchange_rates(self, c_h, c_in, ds_tagged, dS_tagged):
        """
        Return species exchange rates implied by the terminal source/sink model.

        Rates are reported in the same units as concentration times volumetric
        flow, i.e. "amount per unit time".
        """
        mesh = self.mesh
        n = ufl.FacetNormal(mesh)

        if self.couple_1d_transport:
            signed_rates = np.concatenate(
                (
                    self._coupled_terminal_marker_rates["inlet"],
                    self._coupled_terminal_marker_rates["outlet"],
                )
            )
            # Positive means network -> 3D injection; negative means 3D ->
            # network removal. Either tree may change sign when diffusion
            # dominates weak advection, so classify by the computed flux.
            return (
                np.maximum(signed_rates, 0.0),
                np.maximum(-signed_rates, 0.0),
            )

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
                direction = fem.Constant(
                    mesh,
                    dfx.default_scalar_type(
                        tuple(float(value) for value in data["distal_direction"])
                    ),
                )
                c_tissue = ufl.conditional(
                    ufl.gt(ufl.dot(n("+"), direction), 0.0), c_h("-"), c_h("+")
                )
                rate += self._global_scalar(g * c_tissue * dS_tagged(marker))
            outlet_rates.append(rate)

        return (
            np.asarray(inlet_rates, dtype=float),
            np.asarray(outlet_rates, dtype=float),
        )

    def _compute_concave_exchange_rates(
        self,
        c_h,
        c_marker_31,
        c_marker_32,
        ds_tagged,
        n,
        D=None,
        h=None,
        alpha=None,
    ):
        inflow_rates = []
        outflow_rates = []

        if self.concave_exchange_mode == "combined":
            if D is None or h is None or alpha is None:
                raise ValueError(
                    "Combined concave flux diagnostics require D, h, and alpha"
                )
            qn = ufl.dot(self.velocity, n)
            q_in = 0.5 * (-qn + abs(qn))
            q_out = 0.5 * (qn + abs(qn))
            for marker, boundary_concentration in (
                (int(self.arterial_concave_marker), c_marker_31),
                (int(self.venous_concave_marker), c_marker_32),
            ):
                advective_flux_out = q_out * c_h - q_in * boundary_concentration
                diffusive_flux_out = (
                    -D * ufl.dot(ufl.grad(c_h), n)
                    + alpha * D / h * (c_h - boundary_concentration)
                )
                total_flux_out = advective_flux_out + diffusive_flux_out
                total_flux_in = 0.5 * (
                    -total_flux_out + abs(total_flux_out)
                )
                total_flux_out_positive = 0.5 * (
                    total_flux_out + abs(total_flux_out)
                )
                inflow_rates.append(
                    self._global_scalar(total_flux_in * ds_tagged(marker))
                )
                outflow_rates.append(
                    self._global_scalar(
                        total_flux_out_positive * ds_tagged(marker)
                    )
                )
            return (
                np.asarray(inflow_rates, dtype=float),
                np.asarray(outflow_rates, dtype=float),
            )

        if self.concave_exchange_mode == "dirichlet":
            if D is None or h is None or alpha is None:
                raise ValueError(
                    "Dirichlet concave flux diagnostics require D, h, and alpha"
                )
            # Numerical outward diffusive flux associated with the SIPG
            # Dirichlet boundary residual. Negative values enter the domain.
            for marker, boundary_concentration in (
                (int(self.arterial_concave_marker), c_marker_31),
                (int(self.venous_concave_marker), c_marker_32),
            ):
                flux_out = (
                    -D * ufl.dot(ufl.grad(c_h), n)
                    + alpha * D / h * (c_h - boundary_concentration)
                )
                flux_in = 0.5 * (-flux_out + abs(flux_out))
                flux_out_positive = 0.5 * (flux_out + abs(flux_out))
                inflow_rates.append(
                    self._global_scalar(flux_in * ds_tagged(marker))
                )
                outflow_rates.append(
                    self._global_scalar(flux_out_positive * ds_tagged(marker))
                )
            return (
                np.asarray(inflow_rates, dtype=float),
                np.asarray(outflow_rates, dtype=float),
            )

        if self.concave_exchange_mode == "velocity":
            qn = ufl.dot(self.velocity, n)
            q_in = 0.5 * (-qn + abs(qn))
            q_out = 0.5 * (qn + abs(qn))
            for marker in (
                int(self.arterial_concave_marker),
                int(self.venous_concave_marker),
            ):
                inflow_rates.append(
                    self._global_scalar(q_in * c_marker_31 * ds_tagged(marker))
                )
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
                rate = self._global_scalar(g * c_marker_31 * ds_tagged(marker))
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
        cutoff = self._smooth_cutoff_numpy(
            c_prev,
            float(self.critical_oxygen_concentration),
            float(self.critical_transition_width),
        )
        rate_fun.x.array[:] = float(self.mm_vmax) * chi * cutoff / denom
        rate_fun.x.scatter_forward()

    def _compute_consumption_rate(self, c_h, organoid_indicator, dx):
        if not self.enable_mm_consumption:
            return 0.0
        km = fem.Constant(self.mesh, dfx.default_scalar_type(self.mm_km))
        vmax = fem.Constant(self.mesh, dfx.default_scalar_type(self.mm_vmax))
        c_pos = ufl.max_value(c_h, 0.0)
        cutoff = self._smooth_cutoff_ufl(
            c_pos,
            float(self.critical_oxygen_concentration),
            float(self.critical_transition_width),
        )
        rate_expr = organoid_indicator * cutoff * vmax * c_pos / (km + c_pos)
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
            "below_critical_domain_volume_fraction": float(below_vol / total_vol)
            if total_vol > 0.0
            else 0.0,
            "organoid_below_critical_volume_cm3": float(organoid_below_vol),
            "organoid_volume_cm3": float(organoid_vol),
            "below_critical_volume_fraction": float(organoid_below_vol / organoid_vol)
            if organoid_vol > 0.0
            else 0.0,
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
        c_marker_32 = (
            c_in
            if self.marker_32_concentration is None
            else fem.Constant(
                mesh, dfx.default_scalar_type(self.marker_32_concentration)
            )
        )
        delta_t = fem.Constant(mesh, dfx.default_scalar_type(self.dt))
        organoid_indicator, D = self._build_organoid_indicator_and_diffusion(W)
        mm_rate = fem.Function(W)

        c = ufl.TrialFunction(W)
        w = ufl.TestFunction(W)

        c_old = fem.Function(W)
        c_h = fem.Function(W)
        critical_mask = fem.Function(W)
        consumption_field = fem.Function(W)
        c_out = fem.Function(fem.functionspace(mesh, P1))
        critical_out = fem.Function(fem.functionspace(mesh, P1))
        consumption_out = fem.Function(fem.functionspace(mesh, P1))
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
        h = ufl.CellDiameter(domain)
        alpha = fem.Constant(mesh, dfx.default_scalar_type(10.0))
        terminal_alpha = fem.Constant(mesh, dfx.default_scalar_type(1000.0))
        a_exchange, L_exchange = self._build_terminal_exchange_forms(
            c, w, c_in, D, ds_tagged, dS_tagged, n, h, terminal_alpha
        )
        a_concave, L_concave = self._build_concave_exchange_forms(
            c, w, c_in, c_marker_32, ds_tagged, n
        )

        a = a_time + a_advect + a_diffuse + a_consumption + a_exchange + a_concave
        L = (c_old / delta_t) * w * dx + L_exchange + L_concave

        # Upwind advection flux on all interior facets.
        un = (ufl.dot(self.velocity, n) + abs(ufl.dot(self.velocity, n))) / 2.0
        interior_advection = ufl.jump(w) * (
            un("+") * c("+") - un("-") * c("-")
        )
        a += interior_advection * dS_tagged
        if self.couple_1d_transport:
            for data in [*self.inlet_exchange, *self.outlet_exchange]:
                if data["area_interior"] > 0.0:
                    a -= interior_advection * dS_tagged(int(data["marker"]))

        # SIPG diffusion for DG concentration.
        if self.concave_exchange_mode in {"dirichlet", "combined"}:
            a_dirichlet, L_dirichlet = self._build_concave_dirichlet_diffusion_forms(
                c, w, c_in, c_marker_32, D, ds_tagged, n, h, alpha
            )
            a += a_dirichlet
            L += L_dirichlet
        interior_diffusion = D("+") * alpha("+") / h("+") * ufl.dot(
            ufl.jump(w, n), ufl.jump(c, n)
        )
        interior_diffusion -= D("+") * ufl.dot(
            ufl.avg(ufl.grad(w)), ufl.jump(c, n)
        )
        interior_diffusion -= D("+") * ufl.dot(
            ufl.avg(ufl.grad(c)), ufl.jump(w, n)
        )
        a += interior_diffusion * dS_tagged
        if self.couple_1d_transport:
            for data in [*self.inlet_exchange, *self.outlet_exchange]:
                if data["area_interior"] > 0.0:
                    a -= interior_diffusion * dS_tagged(int(data["marker"]))

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
        requested_critical_path = (
            Path(self.critical_output_file)
            if self.critical_output_file
            else None
        )
        requested_consumption_path = (
            Path(self.consumption_output_file)
            if self.consumption_output_file
            else None
        )
        write_critical_visualization = requested_critical_path is not None
        write_consumption_visualization = requested_consumption_path is not None
        write_vtk = self.output_format in {"vtk", "both"}
        write_xdmf = self.output_format in {"xdmf", "both"}
        vtk_output_path = _vtk_output_path(requested_out_path)
        vtk_critical_output_path = (
            _vtk_output_path(requested_critical_path)
            if requested_critical_path is not None
            else None
        )
        vtk_consumption_output_path = (
            _vtk_output_path(requested_consumption_path)
            if requested_consumption_path is not None
            else None
        )
        xdmf_output_path = requested_out_path
        xdmf_critical_output_path = (
            requested_critical_path
            if requested_critical_path is not None
            else None
        )
        xdmf_consumption_output_path = (
            requested_consumption_path
            if requested_consumption_path is not None
            else None
        )

        if mesh.comm.rank == 0:
            # If the requested path was previously used for a temporal XDMF
            # run, remove it even when the new default output is VTK. Otherwise
            # a stale crashing .xdmf/.h5 pair can remain next to the good .pvd.
            if requested_out_path.suffix.lower() == ".xdmf":
                _remove_xdmf_with_sidecar(requested_out_path)
            if requested_critical_path is not None and requested_critical_path.suffix.lower() == ".xdmf":
                _remove_xdmf_with_sidecar(requested_critical_path)
            if requested_consumption_path is not None and requested_consumption_path.suffix.lower() == ".xdmf":
                _remove_xdmf_with_sidecar(requested_consumption_path)
            if write_vtk:
                _remove_vtk_series(vtk_output_path)
                if vtk_critical_output_path is not None and write_critical_visualization:
                    _remove_vtk_series(vtk_critical_output_path)
                if vtk_consumption_output_path is not None and write_consumption_visualization:
                    _remove_vtk_series(vtk_consumption_output_path)
            if write_xdmf:
                _remove_xdmf_with_sidecar(xdmf_output_path)
                if xdmf_critical_output_path is not None and write_critical_visualization:
                    _remove_xdmf_with_sidecar(xdmf_critical_output_path)
                if xdmf_consumption_output_path is not None and write_consumption_visualization:
                    _remove_xdmf_with_sidecar(xdmf_consumption_output_path)
        mesh.comm.barrier()

        c_out.name = "oxygen_concentration"
        critical_out.name = "below_critical_oxygen"
        consumption_out.name = "oxygen_consumption_rate"

        vtk_out = None
        vtk_critical_out = None
        vtk_consumption_out = None
        xdmf_out = None
        xdmf_critical_out = None
        xdmf_consumption_out = None
        if write_vtk:
            vtk_out = io.VTKFile(mesh.comm, str(vtk_output_path), "w")
            if write_critical_visualization and vtk_critical_output_path is not None:
                vtk_critical_out = io.VTKFile(mesh.comm, str(vtk_critical_output_path), "w")
            if write_consumption_visualization and vtk_consumption_output_path is not None:
                vtk_consumption_out = io.VTKFile(mesh.comm, str(vtk_consumption_output_path), "w")
            if mesh.comm.rank == 0:
                msg = f"[transport] Writing ParaView time series: {vtk_output_path}"
                if vtk_critical_out is not None:
                    msg += f" and {vtk_critical_output_path}"
                if vtk_consumption_out is not None:
                    msg += f" and {vtk_consumption_output_path}"
                print(msg, flush=True)
        if write_xdmf:
            xdmf_out = io.XDMFFile(mesh.comm, str(xdmf_output_path), "w")
            xdmf_out.write_mesh(mesh)
            if write_critical_visualization and xdmf_critical_output_path is not None:
                xdmf_critical_out = io.XDMFFile(mesh.comm, str(xdmf_critical_output_path), "w")
                xdmf_critical_out.write_mesh(mesh)
            if write_consumption_visualization and xdmf_consumption_output_path is not None:
                xdmf_consumption_out = io.XDMFFile(mesh.comm, str(xdmf_consumption_output_path), "w")
                xdmf_consumption_out.write_mesh(mesh)
        if write_xdmf and mesh.comm.rank == 0:
            msg = f"[transport] Writing XDMF time series: {xdmf_output_path}"
            if write_critical_visualization and xdmf_critical_output_path is not None:
                msg += f" and {xdmf_critical_output_path}"
            if write_consumption_visualization and xdmf_consumption_output_path is not None:
                msg += f" and {xdmf_consumption_output_path}"
            print(msg, flush=True)

        velocity_debug_out = None
        velocity_magnitude_debug_out = None
        velocity_debug_csv_stream = None
        velocity_debug_csv_writer = None
        velocity_debug_path = None
        velocity_magnitude_debug_path = None
        velocity_debug_csv_path = None
        velocity_magnitude = None
        velocity_magnitude_expression = None
        if self.velocity_debug_output_file:
            velocity_debug_path = _vtk_output_path(self.velocity_debug_output_file)
            velocity_magnitude_debug_path = velocity_debug_path.with_name(
                f"{velocity_debug_path.stem}_magnitude.pvd"
            )
            velocity_debug_csv_path = velocity_debug_path.with_name(
                f"{velocity_debug_path.stem}_surface_flux.csv"
            )
            if mesh.comm.rank == 0:
                velocity_debug_path.parent.mkdir(parents=True, exist_ok=True)
                _remove_vtk_series(velocity_debug_path)
                _remove_vtk_series(velocity_magnitude_debug_path)
                if velocity_debug_csv_path.exists():
                    velocity_debug_csv_path.unlink()
                velocity_debug_csv_stream = velocity_debug_csv_path.open(
                    "w", newline=""
                )
            mesh.comm.barrier()
            velocity_debug_out = io.VTKFile(
                mesh.comm, str(velocity_debug_path), "w"
            )
            velocity_magnitude_debug_out = io.VTKFile(
                mesh.comm, str(velocity_magnitude_debug_path), "w"
            )
            velocity_magnitude_space = fem.functionspace(mesh, P1)
            velocity_magnitude = fem.Function(velocity_magnitude_space)
            velocity_magnitude.name = "transport_velocity_magnitude_cm_per_s"
            velocity_magnitude_expression = fem.Expression(
                ufl.sqrt(ufl.dot(self.velocity, self.velocity)),
                velocity_magnitude_space.element.interpolation_points(),
            )
            self.velocity.name = "transport_velocity_scaled_cm_per_s"
            if mesh.comm.rank == 0:
                print(
                    "[transport] Writing exact imported/scaled velocity debug fields: "
                    f"{velocity_debug_path} and {velocity_magnitude_debug_path}; "
                    f"surface audit: {velocity_debug_csv_path}",
                    flush=True,
                )

        t = 0.0
        nt = int(np.round(self.T / self.dt))
        injected_cumulative = 0.0
        removed_cumulative = 0.0
        consumed_cumulative = 0.0
        critical_rows = []
        output_stem = Path(self.out_file).stem
        output_parent = Path(self.out_file).parent
        network_history_path = output_parent / f"{output_stem}_1d_terminal_history.csv"
        arterial_network_pvd_path = output_parent / f"{output_stem}_1d_arterial.pvd"
        venous_network_pvd_path = output_parent / f"{output_stem}_1d_venous.pvd"
        network_history_writer = None
        arterial_network_writer = None
        venous_network_writer = None
        if mesh.comm.rank == 0 and self.couple_1d_transport:
            network_history_writer = NetworkTerminalHistoryWriter(
                network_history_path
            )
            arterial_network_writer = NetworkTransportVTKWriter(
                self.network_inlet, arterial_network_pvd_path
            )
            venous_network_writer = NetworkTransportVTKWriter(
                self.network_outlet, venous_network_pvd_path
            )
            arterial_network_writer.write(0.0)
            venous_network_writer.write(0.0)
            print(
                "[transport] Writing coupled 1D visualization time series: "
                f"{arterial_network_pvd_path} and {venous_network_pvd_path}",
                flush=True,
            )
            print(
                "[transport] Streaming coupled 1D terminal history each timestep: "
                f"{network_history_path}",
                flush=True,
            )
        progress_reporter = (
            TransportProgressReporter(self.out_file) if mesh.comm.rank == 0 else None
        )
        if progress_reporter is not None:
            print(
                f"[transport] Writing timestep history: {progress_reporter.csv_path}",
                flush=True,
            )
            print(
                "[transport] Updating progress figures: "
                f"{progress_reporter.rates_path}, "
                f"{progress_reporter.cumulative_path}, "
                f"{progress_reporter.below_critical_path}",
                flush=True,
            )

        for step in range(nt):
            t += self.dt
            c_in.value = dfx.default_scalar_type(self._inlet_concentration(t))
            self._update_mm_consumption_rate(mm_rate, c_old, organoid_indicator)

            if self.couple_1d_transport:
                history_start = len(self.network_history)
                self._advance_coupled_networks(
                    t,
                    float(c_in.value),
                    float(c_marker_32.value),
                )

            velocity_changed = self._update_velocity_for_time(t)
            if velocity_changed:
                A = assemble_matrix(a_form)
                A.assemble()
                solver.setOperators(A)
            elif self.enable_mm_consumption:
                A = assemble_matrix(a_form)
                A.assemble()
                solver.setOperators(A)

            # A static velocity is written once; a time-dependent imported
            # velocity is written after every interpolation update. These are
            # the exact scaled coefficient values used by the transport form.
            if velocity_debug_out is not None and (step == 0 or velocity_changed):
                velocity_magnitude.interpolate(velocity_magnitude_expression)
                velocity_magnitude.x.scatter_forward()
                velocity_debug_out.write_function(self.velocity, t)
                velocity_magnitude_debug_out.write_function(velocity_magnitude, t)
                debug_rows = self._velocity_debug_rows(t, ds_tagged, dS_tagged)
                if mesh.comm.rank == 0:
                    if velocity_debug_csv_writer is None:
                        velocity_debug_csv_writer = csv.DictWriter(
                            velocity_debug_csv_stream,
                            fieldnames=list(debug_rows[0].keys()),
                        )
                        velocity_debug_csv_writer.writeheader()
                    velocity_debug_csv_writer.writerows(debug_rows)
                    velocity_debug_csv_stream.flush()

            b.zeroEntries()
            assemble_vector(b, L_form)
            b.ghostUpdate(
                addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE
            )

            solver.solve(b, c_h.x.petsc_vec)
            c_h.x.scatter_forward()
            if self.couple_1d_transport:
                self._update_coupled_terminal_diffusive_fluxes(
                    c_h,
                    D,
                    ds_tagged,
                    dS_tagged,
                    h,
                    terminal_alpha,
                    t,
                    float(c_in.value),
                    float(c_marker_32.value),
                )
                if mesh.comm.rank == 0:
                    network_history_writer.update(
                        self.network_history[history_start:]
                    )
                    arterial_network_writer.write(t)
                    venous_network_writer.write(t)
            c_old.x.array[:] = c_h.x.array

            c_out.interpolate(c_h)
            # c_out.x.array[:] = c_h.x.array
            # c_out.x.scatter_forward()
            critical_stats = self._critical_oxygen_stats(
                c_h,
                organoid_indicator,
                critical_mask,
                dx,
            )
            if self.enable_mm_consumption:
                km = max(float(self.mm_km), np.finfo(float).tiny)
                c_pos_vals = np.maximum(np.asarray(c_h.x.array, dtype=float), 0.0)
                chi_vals = np.asarray(organoid_indicator.x.array, dtype=float)
                cutoff_vals = self._smooth_cutoff_numpy(
                    c_pos_vals,
                    float(self.critical_oxygen_concentration),
                    float(self.critical_transition_width),
                )
                consumption_field.x.array[:] = (
                    float(self.mm_vmax)
                    * chi_vals
                    * cutoff_vals
                    * c_pos_vals
                    / (km + c_pos_vals)
                )
            else:
                consumption_field.x.array[:] = 0.0
            consumption_field.x.scatter_forward()
            critical_out.interpolate(critical_mask)
            consumption_out.interpolate(consumption_field)
            if vtk_out is not None:
                vtk_out.write_function(c_out, t)
            if vtk_critical_out is not None:
                vtk_critical_out.write_function(critical_out, t)
            if vtk_consumption_out is not None:
                vtk_consumption_out.write_function(consumption_out, t)
            if xdmf_out is not None:
                xdmf_out.write_function(c_out, t)
            if xdmf_critical_out is not None:
                xdmf_critical_out.write_function(critical_out, t)
            if xdmf_consumption_out is not None:
                xdmf_consumption_out.write_function(consumption_out, t)

            inlet_rates, outlet_rates = self._compute_exchange_rates(
                c_h, c_in, ds_tagged, dS_tagged
            )
            concave_in_rates, concave_out_rates = self._compute_concave_exchange_rates(
                c_h,
                c_in,
                c_marker_32,
                ds_tagged,
                n,
                D=D,
                h=h,
                alpha=alpha,
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
                "terminal_injection_rate_mol_per_s": float(terminal_in_total),
                "terminal_outflow_rate_mol_per_s": float(terminal_out_total),
                "concave_injection_rate_mol_per_s": float(concave_in_total),
                "concave_outflow_rate_mol_per_s": float(concave_out_total),
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
                progress_reporter.update(row)

        if progress_reporter is not None:
            progress_reporter.close()
        if network_history_writer is not None:
            network_history_writer.close()

        if vtk_out is not None:
            vtk_out.close()
        if vtk_critical_out is not None:
            vtk_critical_out.close()
        if vtk_consumption_out is not None:
            vtk_consumption_out.close()
        if velocity_debug_out is not None:
            velocity_debug_out.close()
        if velocity_magnitude_debug_out is not None:
            velocity_magnitude_debug_out.close()
        if velocity_debug_csv_stream is not None:
            velocity_debug_csv_stream.close()

        if write_xdmf:
            if xdmf_out is not None:
                xdmf_out.close()
            if xdmf_critical_out is not None:
                xdmf_critical_out.close()
            if xdmf_consumption_out is not None:
                xdmf_consumption_out.close()

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
                "critical_mask_pvd": (
                    str(vtk_critical_output_path)
                    if write_vtk and write_critical_visualization and vtk_critical_output_path is not None
                    else ""
                ),
                "consumption_pvd": (
                    str(vtk_consumption_output_path)
                    if write_vtk and write_consumption_visualization and vtk_consumption_output_path is not None
                    else ""
                ),
                "oxygen_xdmf": str(xdmf_output_path) if write_xdmf else "",
                "critical_mask_xdmf": (
                    str(xdmf_critical_output_path)
                    if write_xdmf and write_critical_visualization and xdmf_critical_output_path is not None
                    else ""
                ),
                "consumption_xdmf": (
                    str(xdmf_consumption_output_path)
                    if write_xdmf and write_consumption_visualization and xdmf_consumption_output_path is not None
                    else ""
                ),
                "time_history_csv": str(progress_reporter.csv_path),
                "network_terminal_history_csv": (
                    str(network_history_path) if self.couple_1d_transport else ""
                ),
                "arterial_network_pvd": (
                    str(arterial_network_pvd_path)
                    if self.couple_1d_transport
                    else ""
                ),
                "venous_network_pvd": (
                    str(venous_network_pvd_path)
                    if self.couple_1d_transport
                    else ""
                ),
                "rates_figure": str(progress_reporter.rates_path),
                "cumulative_figure": str(progress_reporter.cumulative_path),
                "organoid_below_critical_figure": str(
                    progress_reporter.below_critical_path
                ),
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
            diffusion_only=args.diffusion_only,
            marker_32_concentration=args.marker_32_concentration,
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
            critical_transition_width=args.critical_transition_width,
            critical_output_file=args.critical_output_file,
            consumption_output_file=args.consumption_output_file,
            critical_summary_file=args.critical_summary_file,
            velocity_debug_output_file=args.velocity_debug_output_file,
            output_format=args.output_format,
            branching_in_file=args.branching_in_file,
            branching_out_file=args.branching_out_file,
            couple_1d_transport=args.coupled_1d_transport,
            network_cells_per_segment=args.network_cells_per_segment,
            network_coupling_relaxation=args.network_coupling_relaxation,
            network_initial_concentration=args.network_initial_concentration,
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
        diffusion_only=args.diffusion_only,
        marker_32_concentration=args.marker_32_concentration,
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
        critical_transition_width=args.critical_transition_width,
        critical_output_file=_resolve_output_path(args.critical_output_file, organoid_id, batch_mode)
        if args.critical_output_file
        else "",
        consumption_output_file=_resolve_output_path(args.consumption_output_file, organoid_id, batch_mode)
        if args.consumption_output_file
        else "",
        critical_summary_file=_resolve_output_path(args.critical_summary_file, organoid_id, batch_mode)
        if args.critical_summary_file
        else "",
        velocity_debug_output_file=_resolve_output_path(
            args.velocity_debug_output_file, organoid_id, batch_mode
        )
        if args.velocity_debug_output_file
        else "",
        output_format=args.output_format,
        branching_in_file=_resolve_organoid_path(args.branching_in_file, organoid_id)
        if args.branching_in_file
        else "",
        branching_out_file=_resolve_organoid_path(args.branching_out_file, organoid_id)
        if args.branching_out_file
        else "",
        couple_1d_transport=args.coupled_1d_transport,
        network_cells_per_segment=args.network_cells_per_segment,
        network_coupling_relaxation=args.network_coupling_relaxation,
        network_initial_concentration=args.network_initial_concentration,
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
        "--branching-in-file",
        default="",
        help=(
            "Arterial branchingData CSV used to orient embedded terminal faces. "
            "Defaults to branchingData_0.csv beside the organoid geometry directory."
        ),
    )
    ap.add_argument(
        "--branching-out-file",
        default="",
        help=(
            "Venous branchingData CSV used to orient embedded terminal faces. "
            "Defaults to branchingData_1.csv beside the organoid geometry directory."
        ),
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
        choices=["velocity", "uniform", "combined"],
        default="velocity",
        help=(
            "Concave-wall model: local Darcy advective flux ('velocity'), "
            "uniform flux from interface_bc.json ('uniform'), or local Darcy "
            "advection plus Dirichlet diffusion with marker 31 at c_in and "
            "marker 32 at zero by default ('combined')."
        ),
    )
    ap.add_argument(
        "--skip-1d",
        action="store_true",
        help="Disable synthetic terminal exchange and run transport using only concave-wall exchange.",
    )
    ap.add_argument(
        "--coupled-1d-transport",
        action="store_true",
        help=(
            "Solve advection-diffusion on both arterial and venous branching "
            "networks. Arterial terminal concentrations are imposed on the "
            "3D synthetic faces and also set the advective injection "
            "Q*c_terminal, but the resulting 3D Nitsche reaction is diagnostic "
            "only and is not returned as an extra 1D terminal loss. Venous 3D "
            "terminal concentrations are sampled after the 3D solve and imposed "
            "on the venous 1D network. Existing terminal-only conditions remain "
            "the default when omitted."
        ),
    )
    ap.add_argument(
        "--network-cells-per-segment",
        type=int,
        default=1000,
        help="Finite-volume intervals per branchingData segment (default: 1000).",
    )
    ap.add_argument(
        "--network-coupling-relaxation",
        type=float,
        default=1.0,
        help=(
            "Legacy relaxation for diffusive 1D/3D terminal feedback. The "
            "current open-terminal coupled model does not return synthetic "
            "terminal diffusive fluxes, so this is a no-op for those terminals."
        ),
    )
    ap.add_argument(
        "--network-initial-concentration",
        type=float,
        default=0.0,
        help=(
            "Initial arterial and venous lumen concentration in mol/cm^3 "
            "(default: 0)."
        ),
    )
    ap.add_argument(
        "--diffusion-only",
        action="store_true",
        help=(
            "Disable Darcy advection and all 1D terminal exchange, and prescribe "
            "c_in as a Dirichlet condition on concave markers 31 and 32. The "
            "velocity and 1D flow files are not read."
        ),
    )
    ap.add_argument(
        "--marker-32-concentration",
        type=float,
        default=None,
        help=(
            "Dirichlet concentration on concave marker 32 in mol/cm^3 for a "
            "diffusion-only or combined run. Diffusion-only defaults to c_in; "
            "combined defaults to zero. Marker 31 remains at c_in."
        ),
    )
    ap.add_argument("--T", type=float, default=5000.0)
    ap.add_argument("--dt", type=float, default=50.0)
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
        help="Oxygen diffusion in organoid tissue in cm^2/s. Paper value: 1.0e-5.",
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
        help="Michaelis-Menten k_m in mol/cm^3. Paper value: 5.0e-9.",
    )
    ap.add_argument(
        "--mm-vmax",
        type=float,
        default=None,
        help="Volumetric maximal oxygen consumption in mol/(cm^3 s). Paper value: 2.0e-9.",
    )
    ap.add_argument(
        "--critical-oxygen-concentration",
        type=float,
        default=None,
        help="Critical oxygen concentration in mol/cm^3. Paper value: 4.0e-8.",
    )
    ap.add_argument(
        "--critical-transition-width",
        type=float,
        default=None,
        help=(
            "Half-width of the smooth step-down around the critical oxygen "
            "concentration, in mol/cm^3. Use 0 for a hard cutoff. Default: 5%% of C_crit."
        ),
    )
    ap.add_argument(
        "--critical-output-file",
        default="",
        help="Optional XDMF output path for the below-critical oxygen mask.",
    )
    ap.add_argument(
        "--consumption-output-file",
        default="",
        help="Optional output path for the oxygen consumption-rate field visualization.",
    )
    ap.add_argument(
        "--critical-summary-file",
        default="",
        help="Optional JSON path for critical-oxygen volume/time-series summary.",
    )
    ap.add_argument(
        "--velocity-debug-output-file",
        default="",
        help=(
            "Optional PVD path for the exact imported, time-interpolated, and "
            "scaled Darcy velocity used by the transport form. Also writes a "
            "velocity-magnitude PVD and a per-surface flow-audit CSV. Static "
            "velocity is written once; transient velocity is written whenever "
            "the imported field is updated."
        ),
    )
    ap.add_argument(
        "--output-format",
        choices=["vtk", "xdmf", "both"],
        default="vtk",
        help=(
            "Visualization format. 'vtk' writes ParaView-safe .pvd/.vtu time "
            "series. 'xdmf' writes XDMF time series. 'both' writes both."
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
