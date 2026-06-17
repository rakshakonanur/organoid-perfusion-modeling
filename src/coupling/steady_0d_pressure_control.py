from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


class _DisjointSet:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, str], tuple[int, str]] = {}

    def add(self, key: tuple[int, str]) -> None:
        if key not in self.parent:
            self.parent[key] = key

    def find(self, key: tuple[int, str]) -> tuple[int, str]:
        self.add(key)
        parent = self.parent[key]
        if parent != key:
            parent = self.find(parent)
            self.parent[key] = parent
        return parent

    def union(self, a: tuple[int, str], b: tuple[int, str]) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass(frozen=True)
class Vessel:
    vessel_id: int
    name: str
    inlet_node: int
    outlet_node: int
    resistance: float
    bc_inlet: str | None
    bc_outlet: str | None


@dataclass(frozen=True)
class BoundaryCondition:
    name: str
    bc_type: str
    value: float
    endpoint: tuple[int, str] | None = None


@dataclass(frozen=True)
class TerminalResistance:
    bc_name: str
    vessel_id: int
    node_id: int
    default_resistance: float
    default_pd: float


@dataclass(frozen=True)
class PressureMap:
    terminal_names: list[str]
    terminal_node_ids: list[int]
    d: np.ndarray
    m: np.ndarray

    def target_vector(self, targets: dict[str, float]) -> np.ndarray:
        return np.array([float(targets[name]) for name in self.terminal_names], dtype=float)

    def solve_direct(self, targets: dict[str, float]) -> tuple[np.ndarray, str]:
        target_vec = self.target_vector(targets)
        rhs = target_vec - self.d
        if self.m.shape[0] == self.m.shape[1]:
            try:
                return np.linalg.solve(self.m, rhs), "direct"
            except np.linalg.LinAlgError:
                pass
        return np.linalg.lstsq(self.m, rhs, rcond=None)[0], "least_squares"


@dataclass(frozen=True)
class FlowMap:
    terminal_names: list[str]
    a: np.ndarray
    b: np.ndarray

    def target_vector(self, targets: dict[str, float]) -> np.ndarray:
        return np.array([float(targets[name]) for name in self.terminal_names], dtype=float)

    def solve_direct(self, targets: dict[str, float]) -> tuple[np.ndarray, str]:
        target_vec = self.target_vector(targets)
        rhs = target_vec - self.a
        if self.b.shape[0] == self.b.shape[1]:
            try:
                return np.linalg.solve(self.b, rhs), "direct"
            except np.linalg.LinAlgError:
                pass
        return np.linalg.lstsq(self.b, rhs, rcond=None)[0], "least_squares"


class SteadyResistiveZeroDModel:
    """
    Build a steady resistive nodal model from an svZeroDSolver JSON deck.

    This intentionally keeps only the linear resistive part of the 0D model:
    - vessel resistance uses `R_poiseuille`
    - inertance `L` is ignored
    - capacitance `C` is ignored
    - nonlinear stenosis terms are ignored

    That makes the terminal-pressure relation affine in the terminal Pd values:

        p_terminal = d + M * Pd

    which we can then use for direct algebraic pressure control.
    """

    def __init__(
        self,
        vessels: list[Vessel],
        boundary_conditions: dict[str, BoundaryCondition],
        terminal_resistances: list[TerminalResistance],
        node_count: int,
    ) -> None:
        self.vessels = vessels
        self.boundary_conditions = boundary_conditions
        self.terminal_resistances = terminal_resistances
        self.node_count = int(node_count)

    @classmethod
    def from_solver_file(cls, path: str | Path) -> "SteadyResistiveZeroDModel":
        data = json.load(open(Path(path), "r"))
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: dict[str, Any]) -> "SteadyResistiveZeroDModel":
        bc_defs = {
            str(row["bc_name"]): BoundaryCondition(
                name=str(row["bc_name"]),
                bc_type=str(row["bc_type"]).upper(),
                value=cls._read_bc_scalar(row),
            )
            for row in data.get("boundary_conditions", [])
        }

        dsu = _DisjointSet()
        vessel_rows = data.get("vessels", [])
        zero_resistance_vessels: set[int] = set()
        for row in vessel_rows:
            vid = int(row["vessel_id"])
            dsu.add((vid, "inlet"))
            dsu.add((vid, "outlet"))
            resistance = _safe_float(
                (row.get("zero_d_element_values") or {}).get("R_poiseuille")
            )
            if np.isfinite(resistance) and resistance == 0.0:
                zero_resistance_vessels.add(vid)

        # Collapse ideal connector vessels to a single node in the algebraic
        # model. This preserves any BC attached to the vessel while avoiding
        # singular/ill-conditioned conductance rows from explicit zero-R edges.
        for vid in zero_resistance_vessels:
            dsu.union((vid, "inlet"), (vid, "outlet"))

        for row in data.get("junctions", []):
            endpoints: list[tuple[int, str]] = []
            for vid in row.get("inlet_vessels", []):
                endpoints.append((int(vid), "outlet"))
            for vid in row.get("outlet_vessels", []):
                endpoints.append((int(vid), "inlet"))
            if endpoints:
                root = endpoints[0]
                for endpoint in endpoints[1:]:
                    dsu.union(root, endpoint)

        rep_to_node: dict[tuple[int, str], int] = {}

        def node_id(endpoint: tuple[int, str]) -> int:
            rep = dsu.find(endpoint)
            if rep not in rep_to_node:
                rep_to_node[rep] = len(rep_to_node)
            return rep_to_node[rep]

        vessels: list[Vessel] = []
        terminal_resistances: list[TerminalResistance] = []
        for row in vessel_rows:
            vid = int(row["vessel_id"])
            bc_map = row.get("boundary_conditions", {}) or {}
            inlet_bc = bc_map.get("inlet")
            outlet_bc = bc_map.get("outlet")
            inlet_node = node_id((vid, "inlet"))
            outlet_node = node_id((vid, "outlet"))
            resistance = _safe_float(
                (row.get("zero_d_element_values") or {}).get("R_poiseuille")
            )
            if not np.isfinite(resistance) or resistance < 0.0:
                raise ValueError(f"Vessel {vid} has invalid R_poiseuille: {resistance}")
            resistance = max(float(resistance), np.finfo(float).tiny)
            vessels.append(
                Vessel(
                    vessel_id=vid,
                    name=str(row.get("vessel_name", f"vessel_{vid}")),
                    inlet_node=inlet_node,
                    outlet_node=outlet_node,
                    resistance=float(resistance),
                    bc_inlet=str(inlet_bc) if inlet_bc is not None else None,
                    bc_outlet=str(outlet_bc) if outlet_bc is not None else None,
                )
            )
            if outlet_bc is not None:
                bc_name = str(outlet_bc)
                bc_def = bc_defs.get(bc_name)
                if bc_def is not None and bc_def.bc_type == "RESISTANCE":
                    terminal_resistances.append(
                        TerminalResistance(
                            bc_name=bc_name,
                            vessel_id=vid,
                            node_id=outlet_node,
                            default_resistance=max(float(bc_def.value[1]), np.finfo(float).tiny)
                            if isinstance(bc_def.value, tuple)
                            else np.finfo(float).tiny,
                            default_pd=float(bc_def.value[0])
                            if isinstance(bc_def.value, tuple)
                            else 0.0,
                        )
                    )

        boundary_conditions: dict[str, BoundaryCondition] = {}
        for name, bc in bc_defs.items():
            endpoint = cls._find_bc_endpoint(vessels, name)
            boundary_conditions[name] = BoundaryCondition(
                name=bc.name,
                bc_type=bc.bc_type,
                value=bc.value,
                endpoint=endpoint,
            )

        terminal_resistances = sorted(terminal_resistances, key=lambda item: item.bc_name)
        return cls(
            vessels=vessels,
            boundary_conditions=boundary_conditions,
            terminal_resistances=terminal_resistances,
            node_count=len(rep_to_node),
        )

    @staticmethod
    def _read_bc_scalar(row: dict[str, Any]) -> float | tuple[float, float]:
        bc_type = str(row.get("bc_type", "")).upper()
        values = row.get("bc_values", {}) or {}
        if bc_type == "PRESSURE":
            raw = values.get("P", 0.0)
            if isinstance(raw, list):
                raw = raw[0]
            return float(raw)
        if bc_type == "FLOW":
            raw = values.get("Q", 0.0)
            if isinstance(raw, list):
                raw = raw[0]
            return float(raw)
        if bc_type == "RESISTANCE":
            pd = values.get("Pd", 0.0)
            r = values.get("R", 0.0)
            if isinstance(pd, list):
                pd = pd[0]
            if isinstance(r, list):
                r = r[0]
            return float(pd), float(r)
        raise ValueError(f"Unsupported boundary condition type: {bc_type}")

    @staticmethod
    def _find_bc_endpoint(vessels: Iterable[Vessel], bc_name: str) -> tuple[int, str] | None:
        for vessel in vessels:
            if vessel.bc_inlet == bc_name:
                return vessel.vessel_id, "inlet"
            if vessel.bc_outlet == bc_name:
                return vessel.vessel_id, "outlet"
        return None

    def build_pressure_map(
        self,
        terminal_resistance_overrides: dict[str, float] | None = None,
        flow_bc_overrides: dict[str, float] | None = None,
        pressure_bc_overrides: dict[str, float] | None = None,
    ) -> PressureMap:
        terminal_resistance_overrides = terminal_resistance_overrides or {}
        flow_bc_overrides = flow_bc_overrides or {}
        pressure_bc_overrides = pressure_bc_overrides or {}

        k = np.zeros((self.node_count, self.node_count), dtype=float)
        rhs = np.zeros(self.node_count, dtype=float)
        fixed_pressures: dict[int, float] = {}

        for vessel in self.vessels:
            # Zero-resistance connector vessels are collapsed onto a single
            # node in `from_data`. Re-assembling the resulting self-loop with a
            # tiny fallback resistance introduces enormous +/- conductance
            # terms that numerically erase the real diagonal entries.
            if vessel.inlet_node == vessel.outlet_node:
                continue
            g = 1.0 / max(float(vessel.resistance), np.finfo(float).tiny)
            i = vessel.inlet_node
            j = vessel.outlet_node
            k[i, i] += g
            k[j, j] += g
            k[i, j] -= g
            k[j, i] -= g

        terminal_names = [term.bc_name for term in self.terminal_resistances]
        terminal_node_ids = [term.node_id for term in self.terminal_resistances]
        s = np.zeros((self.node_count, len(self.terminal_resistances)), dtype=float)

        for col, terminal in enumerate(self.terminal_resistances):
            r_term = float(
                terminal_resistance_overrides.get(terminal.bc_name, terminal.default_resistance)
            )
            r_term = max(r_term, np.finfo(float).tiny)
            g_term = 1.0 / r_term
            node = terminal.node_id
            k[node, node] += g_term
            s[node, col] += g_term

        for bc_name, bc in self.boundary_conditions.items():
            if bc.endpoint is None:
                continue
            vessel_id, end = bc.endpoint
            vessel = next(v for v in self.vessels if v.vessel_id == vessel_id)
            node = vessel.inlet_node if end == "inlet" else vessel.outlet_node
            if bc.bc_type == "FLOW":
                q = float(flow_bc_overrides.get(bc_name, bc.value))
                rhs[node] += q if end == "inlet" else -q
            elif bc.bc_type == "PRESSURE":
                p = float(pressure_bc_overrides.get(bc_name, bc.value))
                fixed_pressures[node] = p

        if fixed_pressures:
            fixed_nodes = np.array(sorted(fixed_pressures), dtype=int)
            free_nodes = np.array(
                [node for node in range(self.node_count) if node not in fixed_pressures],
                dtype=int,
            )
        else:
            fixed_nodes = np.zeros((0,), dtype=int)
            free_nodes = np.arange(self.node_count, dtype=int)

        if free_nodes.size == 0:
            raise ValueError("No free nodes remain after applying pressure BCs.")

        k_ff = k[np.ix_(free_nodes, free_nodes)]
        rhs_f = rhs[free_nodes].copy()
        s_f = s[free_nodes, :]
        if fixed_nodes.size:
            p_fixed = np.array([fixed_pressures[int(node)] for node in fixed_nodes], dtype=float)
            rhs_f -= k[np.ix_(free_nodes, fixed_nodes)] @ p_fixed

        reg_scale = max(float(np.max(np.abs(np.diag(k_ff)))) if k_ff.size else 1.0, 1.0)
        reg_eps = 1.0e-12 * reg_scale
        reg_eye = reg_eps * np.eye(k_ff.shape[0], dtype=float)
        try:
            x0 = np.linalg.solve(k_ff, rhs_f)
        except np.linalg.LinAlgError:
            try:
                x0 = np.linalg.lstsq(k_ff, rhs_f, rcond=None)[0]
            except np.linalg.LinAlgError:
                x0 = np.linalg.solve(k_ff + reg_eye, rhs_f)
        try:
            m_free = np.linalg.solve(k_ff, s_f)
        except np.linalg.LinAlgError:
            try:
                m_free = np.linalg.lstsq(k_ff, s_f, rcond=None)[0]
            except np.linalg.LinAlgError:
                m_free = np.linalg.solve(k_ff + reg_eye, s_f)

        full_x0 = np.zeros(self.node_count, dtype=float)
        full_x0[free_nodes] = x0
        if fixed_nodes.size:
            full_x0[fixed_nodes] = p_fixed

        full_m = np.zeros((self.node_count, len(self.terminal_resistances)), dtype=float)
        full_m[free_nodes, :] = m_free

        d = full_x0[terminal_node_ids]
        m = full_m[terminal_node_ids, :]
        return PressureMap(
            terminal_names=terminal_names,
            terminal_node_ids=terminal_node_ids,
            d=d,
            m=m,
        )

    def build_flow_map(
        self,
        terminal_resistance_overrides: dict[str, float] | None = None,
        flow_bc_overrides: dict[str, float] | None = None,
        pressure_bc_overrides: dict[str, float] | None = None,
    ) -> FlowMap:
        pressure_map = self.build_pressure_map(
            terminal_resistance_overrides=terminal_resistance_overrides,
            flow_bc_overrides=flow_bc_overrides,
            pressure_bc_overrides=pressure_bc_overrides,
        )
        terminal_resistance_overrides = terminal_resistance_overrides or {}
        a = np.zeros(len(self.terminal_resistances), dtype=float)
        b = np.zeros((len(self.terminal_resistances), len(self.terminal_resistances)), dtype=float)
        for idx, terminal in enumerate(self.terminal_resistances):
            r_term = float(
                terminal_resistance_overrides.get(terminal.bc_name, terminal.default_resistance)
            )
            r_term = max(r_term, np.finfo(float).tiny)
            g_term = 1.0 / r_term
            a[idx] = g_term * pressure_map.d[idx]
            b[idx, :] = g_term * pressure_map.m[idx, :]
            b[idx, idx] -= g_term
        return FlowMap(
            terminal_names=pressure_map.terminal_names,
            a=a,
            b=b,
        )

    def solve_terminal_pressures_for_terminal_flows(
        self,
        terminal_flow_targets: dict[str, float],
        flow_bc_overrides: dict[str, float] | None = None,
        pressure_bc_overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Solve the internal steady resistive network when terminal flows are
        prescribed directly.

        This treats each terminal resistance branch as an imposed terminal flow
        at the corresponding node rather than as a conductance in the network.
        Once the terminal node pressures are known, the supporting resistance
        for a chosen distal Pd follows from:

            R = (P_terminal - Pd) / Q_target

        where the sign of Q_target matters. A non-positive R indicates that the
        chosen (Pd, Q_target) pair is not physically compatible.
        """
        flow_bc_overrides = flow_bc_overrides or {}
        pressure_bc_overrides = pressure_bc_overrides or {}

        k = np.zeros((self.node_count, self.node_count), dtype=float)
        rhs = np.zeros(self.node_count, dtype=float)
        fixed_pressures: dict[int, float] = {}

        for vessel in self.vessels:
            # Skip collapsed zero-R connector self-loops for the same reason
            # as in `build_pressure_map`.
            if vessel.inlet_node == vessel.outlet_node:
                continue
            g = 1.0 / max(float(vessel.resistance), np.finfo(float).tiny)
            i = vessel.inlet_node
            j = vessel.outlet_node
            k[i, i] += g
            k[j, j] += g
            k[i, j] -= g
            k[j, i] -= g

        terminal_node_ids = [term.node_id for term in self.terminal_resistances]
        terminal_names = [term.bc_name for term in self.terminal_resistances]
        terminal_name_set = set(terminal_names)

        for terminal in self.terminal_resistances:
            q = float(terminal_flow_targets.get(terminal.bc_name, 0.0))
            # Terminal resistance BCs are attached at vessel outlets. Positive
            # Q_target in the internal resistance convention means flow leaving
            # the internal network through that terminal node.
            rhs[int(terminal.node_id)] -= q

        for bc_name, bc in self.boundary_conditions.items():
            if bc.endpoint is None or bc_name in terminal_name_set:
                continue
            vessel_id, end = bc.endpoint
            vessel = next(v for v in self.vessels if v.vessel_id == vessel_id)
            node = vessel.inlet_node if end == "inlet" else vessel.outlet_node
            if bc.bc_type == "FLOW":
                q = float(flow_bc_overrides.get(bc_name, bc.value))
                rhs[node] += q if end == "inlet" else -q
            elif bc.bc_type == "PRESSURE":
                p = float(pressure_bc_overrides.get(bc_name, bc.value))
                fixed_pressures[node] = p

        if fixed_pressures:
            fixed_nodes = np.array(sorted(fixed_pressures), dtype=int)
            free_nodes = np.array(
                [node for node in range(self.node_count) if node not in fixed_pressures],
                dtype=int,
            )
        else:
            fixed_nodes = np.zeros((0,), dtype=int)
            free_nodes = np.arange(self.node_count, dtype=int)

        if free_nodes.size == 0:
            raise ValueError("No free nodes remain after applying pressure BCs.")

        k_ff = k[np.ix_(free_nodes, free_nodes)]
        rhs_f = rhs[free_nodes].copy()
        if fixed_nodes.size:
            p_fixed = np.array([fixed_pressures[int(node)] for node in fixed_nodes], dtype=float)
            rhs_f -= k[np.ix_(free_nodes, fixed_nodes)] @ p_fixed

        reg_scale = max(float(np.max(np.abs(np.diag(k_ff)))) if k_ff.size else 1.0, 1.0)
        reg_eps = 1.0e-12 * reg_scale
        reg_eye = reg_eps * np.eye(k_ff.shape[0], dtype=float)
        try:
            x_free = np.linalg.solve(k_ff, rhs_f)
        except np.linalg.LinAlgError:
            try:
                x_free = np.linalg.lstsq(k_ff, rhs_f, rcond=None)[0]
            except np.linalg.LinAlgError:
                x_free = np.linalg.solve(k_ff + reg_eye, rhs_f)

        x_full = np.zeros(self.node_count, dtype=float)
        x_full[free_nodes] = x_free
        if fixed_nodes.size:
            x_full[fixed_nodes] = p_fixed

        return {
            name: float(x_full[node_id])
            for name, node_id in zip(terminal_names, terminal_node_ids, strict=True)
        }

    def solve_terminal_pd(
        self,
        targets: dict[str, float],
        terminal_resistance_overrides: dict[str, float] | None = None,
        flow_bc_overrides: dict[str, float] | None = None,
        pressure_bc_overrides: dict[str, float] | None = None,
        pd_prior: dict[str, float] | None = None,
        regularization: float = 0.0,
    ) -> dict[str, float]:
        pressure_map = self.build_pressure_map(
            terminal_resistance_overrides=terminal_resistance_overrides,
            flow_bc_overrides=flow_bc_overrides,
            pressure_bc_overrides=pressure_bc_overrides,
        )
        target_vec = pressure_map.target_vector(targets)
        rhs = target_vec - pressure_map.d
        m = pressure_map.m

        if pd_prior:
            prior = np.array(
                [
                    float(
                        pd_prior.get(
                            name,
                            next(term.default_pd for term in self.terminal_resistances if term.bc_name == name),
                        )
                    )
                    for name in pressure_map.terminal_names
                ],
                dtype=float,
            )
        else:
            prior = np.array(
                [
                    next(term.default_pd for term in self.terminal_resistances if term.bc_name == name)
                    for name in pressure_map.terminal_names
                ],
                dtype=float,
            )

        lam = max(float(regularization), 0.0)
        if lam > 0.0:
            lhs = m.T @ m + lam * np.eye(m.shape[1], dtype=float)
            rhs_normal = m.T @ rhs + lam * prior
            pd = np.linalg.solve(lhs, rhs_normal)
        else:
            pd, *_ = np.linalg.lstsq(m, rhs, rcond=None)

        return {
            name: float(value)
            for name, value in zip(pressure_map.terminal_names, pd, strict=True)
        }

    def solve_terminal_pd_direct(
        self,
        targets: dict[str, float],
        terminal_resistance_overrides: dict[str, float] | None = None,
        flow_bc_overrides: dict[str, float] | None = None,
        pressure_bc_overrides: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], str, PressureMap, np.ndarray]:
        pressure_map = self.build_pressure_map(
            terminal_resistance_overrides=terminal_resistance_overrides,
            flow_bc_overrides=flow_bc_overrides,
            pressure_bc_overrides=pressure_bc_overrides,
        )
        pd, mode = pressure_map.solve_direct(targets)
        return (
            {
                name: float(value)
                for name, value in zip(pressure_map.terminal_names, pd, strict=True)
            },
            mode,
            pressure_map,
            pressure_map.target_vector(targets) - (pressure_map.d + pressure_map.m @ pd),
        )

    def solve_terminal_pd_for_flows_direct(
        self,
        targets: dict[str, float],
        terminal_resistance_overrides: dict[str, float] | None = None,
        flow_bc_overrides: dict[str, float] | None = None,
        pressure_bc_overrides: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], str, FlowMap, np.ndarray]:
        flow_map = self.build_flow_map(
            terminal_resistance_overrides=terminal_resistance_overrides,
            flow_bc_overrides=flow_bc_overrides,
            pressure_bc_overrides=pressure_bc_overrides,
        )
        pd, mode = flow_map.solve_direct(targets)
        return (
            {
                name: float(value)
                for name, value in zip(flow_map.terminal_names, pd, strict=True)
            },
            mode,
            flow_map,
            flow_map.target_vector(targets) - (flow_map.a + flow_map.b @ pd),
        )


def _format_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.6g}" for v in values) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build the steady linear resistive terminal-pressure map for an "
            "svZeroDSolver deck and optionally solve for terminal Pd values."
        )
    )
    ap.add_argument("solver_file", help="Path to solver_0d.in or similar svZeroDSolver deck.")
    ap.add_argument(
        "--dump-map",
        action="store_true",
        help="Print the affine terminal pressure map p_term = d + M Pd.",
    )
    ap.add_argument(
        "--targets-json",
        help="Optional JSON file mapping terminal BC names to desired target pressures.",
    )
    ap.add_argument(
        "--regularization",
        type=float,
        default=0.0,
        help="Tikhonov regularization strength for the Pd solve.",
    )
    ap.add_argument(
        "--direct",
        action="store_true",
        help="Use the strict circuit-analog direct solve Pd = M^{-1}(p_target - d) when possible.",
    )
    args = ap.parse_args()

    model = SteadyResistiveZeroDModel.from_solver_file(args.solver_file)
    pressure_map = model.build_pressure_map()

    print(f"nodes={model.node_count}", flush=True)
    print(f"terminals={pressure_map.terminal_names}", flush=True)
    if args.dump_map:
        print(f"d={_format_array(pressure_map.d)}", flush=True)
        print("M=", flush=True)
        for row in pressure_map.m:
            print("  " + _format_array(row), flush=True)

    if args.targets_json:
        targets = json.load(open(args.targets_json, "r"))
        if args.direct:
            pd, mode, _, residual = model.solve_terminal_pd_direct(targets=targets)
            print(f"solve_mode={mode}", flush=True)
            print(f"residual_l2={float(np.linalg.norm(residual)):.6g}", flush=True)
            print(f"residual_linf={float(np.max(np.abs(residual))):.6g}", flush=True)
            print(json.dumps(pd, indent=2, sort_keys=True), flush=True)
        else:
            pd = model.solve_terminal_pd(
                targets=targets,
                regularization=float(args.regularization),
            )
            print(json.dumps(pd, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
