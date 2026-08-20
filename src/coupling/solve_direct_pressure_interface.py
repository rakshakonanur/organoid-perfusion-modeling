#!/usr/bin/env python3
"""Solve the steady 0D--Darcy interface directly with pressure terminal BCs.

For fixed geometry and coefficients, the resistive 0D network and the reference
mixed-Darcy response map are affine.  This script constructs the complete
interface operator

    x = F(x),   x = [terminal pressures, concave leak flows],

and writes a copy of the 0D deck whose coupled terminal conditions are replaced
by constant PRESSURE conditions.  Independent-map mode solves the affine fixed
point directly.  Dynamic scaled-reference mode reproduces the existing
single-organoid pressure scaling and solves its small nonlinear fixed point
with a damped Broyden method.

The exact reference map can be built and cached automatically from a prepared
folder.  The script never modifies the input deck.  With
``--run-full-verification``, it also runs svZeroD from the solved deck and then
runs Darcy with full field output and diagnostics for downstream transport.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from coupling.steady_0d_pressure_control import SteadyResistiveZeroDModel


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _constant_pressure_bc(pressure: float) -> dict[str, Any]:
    return {
        "bc_type": "PRESSURE",
        "bc_values": {"t": [0.0, 1.0], "P": [float(pressure), float(pressure)]},
    }


def _set_constant_flow_bc(bc: dict[str, Any], flow: float) -> None:
    values = bc.get("bc_values", {}) or {}
    raw_t = values.get("t", [0.0, 1.0])
    times = list(raw_t) if isinstance(raw_t, list) and raw_t else [0.0, 1.0]
    bc["bc_type"] = "FLOW"
    bc["bc_values"] = {"t": times, "Q": [float(flow) for _ in times]}


def write_solved_interface_bcs(
    deck: dict[str, Any],
    terminal_pressures: dict[str, float],
    darcy_leak_flows: dict[tuple[int, str], float],
) -> dict[str, Any]:
    """Return a deck with pressure terminals and converged 0D leak-flow BCs."""
    output = copy.deepcopy(deck)
    changed_terminals: set[str] = set()
    changed_leaks: set[tuple[int, str]] = set()
    for bc in output.get("boundary_conditions", []):
        name = str(bc.get("bc_name", ""))
        if name in terminal_pressures:
            bc.update(_constant_pressure_bc(terminal_pressures[name]))
            changed_terminals.add(name)
            continue
        match = re.match(r"^LEAK_(ART|VEN)_(\d+)$", name, re.I)
        if not match:
            continue
        side = "arterial" if match.group(1).upper() == "ART" else "venous"
        key = (int(match.group(2)), side)
        if key not in darcy_leak_flows:
            continue
        # Same sign convention as coupler.apply_organoid_resistance_from_json:
        # Darcy stores outward concave flux; the 0D FLOW BC receives its negative.
        _set_constant_flow_bc(bc, -float(darcy_leak_flows[key]))
        changed_leaks.add(key)
    if changed_terminals != set(terminal_pressures):
        raise ValueError("Failed to replace every coupled terminal boundary condition")
    if changed_leaks != set(darcy_leak_flows):
        missing = sorted(set(darcy_leak_flows) - changed_leaks)
        raise ValueError(f"Failed to write every solved leak flow BC: {missing}")
    return output


@dataclass(frozen=True)
class DarcyMap:
    organoid: int
    path: Path
    h0: np.ndarray
    response: np.ndarray
    baseline_inputs: np.ndarray
    baseline_outputs: np.ndarray
    input_names: list[str]
    output_names: list[str]
    branch_ids_inlet: list[int]
    branch_ids_outlet: list[int]
    profile: str

    @classmethod
    def load(cls, organoid: int, path: Path) -> "DarcyMap":
        path = path.expanduser().resolve()
        metadata_path = path.with_suffix(".json")
        if not path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing Darcy map or metadata: {path}")
        metadata = _load_json(metadata_path)
        if not bool(metadata.get("globally_identified", False)):
            raise ValueError(f"Darcy map is not globally identified: {path}")
        with np.load(path) as arrays:
            h0 = np.asarray(arrays["h0"], dtype=float)
            response = np.asarray(arrays["H"], dtype=float)
            baseline_inputs = np.asarray(
                arrays.get("baseline_inputs", np.zeros(response.shape[1])), dtype=float
            )
            baseline_outputs = np.asarray(
                arrays.get("baseline_outputs", h0 + response @ baseline_inputs),
                dtype=float,
            )
        input_names = [str(value) for value in metadata.get("input_names", [])]
        output_names = [str(value) for value in metadata.get("output_names", [])]
        if response.shape != (len(output_names), len(input_names)):
            raise ValueError(
                f"Map shape {response.shape} does not match metadata "
                f"({len(output_names)}, {len(input_names)}): {path}"
            )
        if baseline_inputs.shape != (len(input_names),):
            raise ValueError(f"Invalid baseline input vector in {path}")
        return cls(
            organoid=int(organoid),
            path=path,
            h0=h0,
            response=response,
            baseline_inputs=baseline_inputs,
            baseline_outputs=baseline_outputs,
            input_names=input_names,
            output_names=output_names,
            branch_ids_inlet=[int(value) for value in metadata.get("branch_ids_inlet", [])],
            branch_ids_outlet=[int(value) for value in metadata.get("branch_ids_outlet", [])],
            profile=str(metadata.get("concave_pressure_profile", "constant")).lower(),
        )


class PressureBoundaryZeroD:
    """Evaluate the resistive 0D network with terminal pressure conditions."""

    def __init__(self, deck: dict[str, Any]) -> None:
        self.model = SteadyResistiveZeroDModel.from_data(deck)
        self.vessel_by_id = {vessel.vessel_id: vessel for vessel in self.model.vessels}
        self.vessel_by_name = {vessel.name.lower(): vessel for vessel in self.model.vessels}
        self.bc_by_name = self.model.boundary_conditions
        self.terminal_vessel_by_name: dict[str, Any] = {}
        for name, bc in self.bc_by_name.items():
            if not re.match(r"^organoid\d+_(?:inlet|outlet)_OUT\d+$", name, re.I):
                continue
            if bc.endpoint is None:
                raise ValueError(f"Terminal boundary condition {name!r} has no endpoint")
            self.terminal_vessel_by_name[name] = self.vessel_by_id[int(bc.endpoint[0])]

    def _bc_node(self, name: str) -> tuple[int, str]:
        bc = self.bc_by_name.get(name)
        if bc is None or bc.endpoint is None:
            raise ValueError(f"Boundary condition {name!r} has no network endpoint")
        vessel_id, end = bc.endpoint
        vessel = self.vessel_by_id[int(vessel_id)]
        return (vessel.inlet_node if end == "inlet" else vessel.outlet_node), end

    def evaluate(
        self,
        terminal_pressures: dict[str, float],
        darcy_leak_flows: dict[tuple[int, str], float],
    ) -> tuple[
        dict[str, float],
        dict[tuple[int, str], float],
        dict[tuple[int, str], float],
    ]:
        n = int(self.model.node_count)
        conductance = np.zeros((n, n), dtype=float)
        rhs = np.zeros(n, dtype=float)
        fixed: dict[int, float] = {}

        for vessel in self.model.vessels:
            if vessel.inlet_node == vessel.outlet_node:
                continue
            g = 1.0 / max(float(vessel.resistance), np.finfo(float).tiny)
            i, j = int(vessel.inlet_node), int(vessel.outlet_node)
            conductance[i, i] += g
            conductance[j, j] += g
            conductance[i, j] -= g
            conductance[j, i] -= g

        terminal_names = set(terminal_pressures)
        for name, pressure in terminal_pressures.items():
            node, _end = self._bc_node(name)
            previous = fixed.get(node)
            if previous is not None and not np.isclose(previous, pressure):
                raise ValueError(f"Conflicting pressure values at node {node}")
            fixed[node] = float(pressure)

        for name, bc in self.bc_by_name.items():
            if name in terminal_names or bc.endpoint is None:
                continue
            node, end = self._bc_node(name)
            if bc.bc_type == "PRESSURE":
                fixed[node] = float(bc.value)
            elif bc.bc_type == "FLOW":
                value = float(bc.value)
                upper = name.upper()
                match = re.match(r"LEAK_(ART|VEN)_(\d+)", upper)
                if match:
                    side = "arterial" if match.group(1) == "ART" else "venous"
                    key = (int(match.group(2)), side)
                    if key in darcy_leak_flows:
                        # This is the sign conversion used by the iterative coupler.
                        value = -float(darcy_leak_flows[key])
                rhs[node] += value if end == "inlet" else -value

        fixed_nodes = np.asarray(sorted(fixed), dtype=int)
        free_nodes = np.asarray([idx for idx in range(n) if idx not in fixed], dtype=int)
        if not fixed_nodes.size:
            raise ValueError("The 0D network has no pressure reference")
        if not free_nodes.size:
            raise ValueError("No free 0D nodes remain after applying terminal pressures")
        fixed_values = np.asarray([fixed[int(idx)] for idx in fixed_nodes], dtype=float)
        matrix = conductance[np.ix_(free_nodes, free_nodes)]
        reduced_rhs = rhs[free_nodes] - conductance[np.ix_(free_nodes, fixed_nodes)] @ fixed_values
        try:
            free_pressure = np.linalg.solve(matrix, reduced_rhs)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular 0D pressure-boundary system") from exc
        pressure = np.zeros(n, dtype=float)
        pressure[fixed_nodes] = fixed_values
        pressure[free_nodes] = free_pressure

        terminal_flows: dict[str, float] = {}
        for name in terminal_names:
            vessel = self.terminal_vessel_by_name.get(name)
            if vessel is None:
                raise ValueError(f"Terminal {name!r} is not present in the input deck")
            terminal_flows[name] = float(
                (pressure[vessel.inlet_node] - pressure[vessel.outlet_node])
                / float(vessel.resistance)
            )

        leak_pressures_out: dict[tuple[int, str], float] = {}
        leak_pressures_in: dict[tuple[int, str], float] = {}
        organoids = sorted({key[0] for key in darcy_leak_flows})
        for organoid in organoids:
            for side, prefix in (("arterial", "leak_art"), ("venous", "leak_ven")):
                vessel = self.vessel_by_name.get(f"{prefix}_{organoid}".lower())
                if vessel is None:
                    raise ValueError(f"Missing 0D vessel {prefix}_{organoid}")
                leak_pressures_in[(organoid, side)] = float(pressure[vessel.inlet_node])
                leak_pressures_out[(organoid, side)] = float(pressure[vessel.outlet_node])
        return terminal_flows, leak_pressures_out, leak_pressures_in


def _terminal_name(organoid: int, side: str, branch: int) -> str:
    tree = "inlet" if side == "arterial" else "outlet"
    return f"organoid{int(organoid)}_{tree}_OUT{int(branch)}"


def _profile_values(
    model: DarcyMap,
    leak_pressures: dict[tuple[int, str], float],
    face_length: float,
    compartment_spacing: float,
) -> dict[str, float]:
    organoid = int(model.organoid)
    p_art = float(leak_pressures[(organoid, "arterial")])
    p_ven = float(leak_pressures[(organoid, "venous")])
    if model.profile != "linear":
        return {"p_concave_arterial": p_art, "p_concave_venous": p_ven}
    if compartment_spacing == 0.0:
        raise ValueError("Linear profile requires nonzero compartment spacing")
    next_organoid = organoid + 1
    if (next_organoid, "arterial") not in leak_pressures:
        raise ValueError(
            f"Linear map for organoid {organoid} requires leak pressures for "
            f"organoid {next_organoid}"
        )
    ratio = float(face_length) / float(compartment_spacing)
    art_delta = float(leak_pressures[(next_organoid, "arterial")]) - p_art
    ven_delta = float(leak_pressures[(next_organoid, "venous")]) - p_ven
    return {
        "p_concave_arterial_low": p_art + art_delta * ratio,
        "p_concave_arterial_high": p_art - art_delta * ratio,
        "p_concave_venous_low": p_ven + ven_delta * ratio,
        "p_concave_venous_high": p_ven - ven_delta * ratio,
    }


def _darcy_inputs(
    model: DarcyMap,
    terminal_flows: dict[str, float],
    leak_pressures: dict[tuple[int, str], float],
    face_length: float,
    compartment_spacing: float,
) -> np.ndarray:
    profiles = _profile_values(model, leak_pressures, face_length, compartment_spacing)
    inlet_idx = 0
    outlet_idx = 0
    values: list[float] = []
    for name in model.input_names:
        if name.startswith("terminal_flux:"):
            if "inlet" in name:
                branch = model.branch_ids_inlet[inlet_idx]
                inlet_idx += 1
                value = terminal_flows[_terminal_name(model.organoid, "arterial", branch)]
            elif "outlet" in name:
                branch = model.branch_ids_outlet[outlet_idx]
                outlet_idx += 1
                value = -terminal_flows[_terminal_name(model.organoid, "venous", branch)]
            else:
                raise ValueError(f"Cannot determine side for Darcy input {name!r}")
        elif name in profiles:
            value = profiles[name]
        else:
            raise ValueError(f"No direct-coupling value for Darcy input {name!r}")
        values.append(float(value))
    if inlet_idx != len(model.branch_ids_inlet) or outlet_idx != len(model.branch_ids_outlet):
        raise ValueError(f"Terminal input count mismatch in {model.path}")
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class InterfaceLayout:
    terminal_names: list[str]
    leak_keys: list[tuple[int, str]]

    @property
    def size(self) -> int:
        return len(self.terminal_names) + len(self.leak_keys)

    def unpack(self, vector: np.ndarray) -> tuple[dict[str, float], dict[tuple[int, str], float]]:
        nt = len(self.terminal_names)
        return (
            {name: float(value) for name, value in zip(self.terminal_names, vector[:nt], strict=True)},
            {key: float(value) for key, value in zip(self.leak_keys, vector[nt:], strict=True)},
        )

    def pack(
        self,
        terminal_pressures: dict[str, float],
        leak_flows: dict[tuple[int, str], float],
    ) -> np.ndarray:
        return np.asarray(
            [terminal_pressures[name] for name in self.terminal_names]
            + [leak_flows[key] for key in self.leak_keys],
            dtype=float,
        )


def build_layout(maps: list[DarcyMap]) -> InterfaceLayout:
    terminal_names: list[str] = []
    leak_keys: list[tuple[int, str]] = []
    for model in sorted(maps, key=lambda item: item.organoid):
        terminal_names.extend(
            _terminal_name(model.organoid, "arterial", branch)
            for branch in model.branch_ids_inlet
        )
        terminal_names.extend(
            _terminal_name(model.organoid, "venous", branch)
            for branch in model.branch_ids_outlet
        )
        leak_keys.extend(((model.organoid, "arterial"), (model.organoid, "venous")))
    if len(set(terminal_names)) != len(terminal_names):
        raise ValueError("Darcy maps contain duplicate terminal names")
    return InterfaceLayout(terminal_names=terminal_names, leak_keys=leak_keys)


def build_scaled_layout(reference_map: DarcyMap, n_organoids: int) -> InterfaceLayout:
    terminal_names: list[str] = []
    leak_keys: list[tuple[int, str]] = []
    for organoid in range(1, int(n_organoids) + 1):
        terminal_names.extend(
            _terminal_name(organoid, "arterial", branch)
            for branch in reference_map.branch_ids_inlet
        )
        terminal_names.extend(
            _terminal_name(organoid, "venous", branch)
            for branch in reference_map.branch_ids_outlet
        )
        leak_keys.extend(((organoid, "arterial"), (organoid, "venous")))
    return InterfaceLayout(terminal_names=terminal_names, leak_keys=leak_keys)


def make_interface_operator(
    maps: list[DarcyMap],
    network: PressureBoundaryZeroD,
    layout: InterfaceLayout,
    face_length: float,
    compartment_spacing: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def evaluate(state: np.ndarray) -> np.ndarray:
        terminal_pressures, leak_flows = layout.unpack(np.asarray(state, dtype=float))
        terminal_q, leak_p, _leak_p_in = network.evaluate(terminal_pressures, leak_flows)
        predicted_p: dict[str, float] = {}
        predicted_leaks: dict[tuple[int, str], float] = {}
        for model in maps:
            inputs = _darcy_inputs(
                model, terminal_q, leak_p, face_length, compartment_spacing
            )
            outputs = {
                name: float(value)
                for name, value in zip(
                    model.output_names, model.h0 + model.response @ inputs, strict=True
                )
            }
            for branch in model.branch_ids_inlet:
                predicted_p[_terminal_name(model.organoid, "arterial", branch)] = outputs[
                    f"p_arterial_branch_{branch}"
                ]
            for branch in model.branch_ids_outlet:
                predicted_p[_terminal_name(model.organoid, "venous", branch)] = outputs[
                    f"p_venous_branch_{branch}"
                ]
            predicted_leaks[(model.organoid, "arterial")] = outputs["q_concave_arterial"]
            predicted_leaks[(model.organoid, "venous")] = outputs["q_concave_venous"]
        return layout.pack(predicted_p, predicted_leaks)

    return evaluate


def _reference_darcy_outputs(
    reference_map: DarcyMap,
    terminal_flows: dict[str, float],
    leak_pressures_out: dict[tuple[int, str], float],
    face_length: float,
    compartment_spacing: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    inputs = _darcy_inputs(
        reference_map,
        terminal_flows,
        leak_pressures_out,
        face_length,
        compartment_spacing,
    )
    output = {
        name: float(value)
        for name, value in zip(
            reference_map.output_names,
            reference_map.h0 + reference_map.response @ inputs,
            strict=True,
        )
    }
    arterial = {
        str(branch): output[f"p_arterial_branch_{branch}"]
        for branch in reference_map.branch_ids_inlet
    }
    venous = {
        str(branch): output[f"p_venous_branch_{branch}"]
        for branch in reference_map.branch_ids_outlet
    }
    leaks = {
        "arterial": output["q_concave_arterial"],
        "venous": output["q_concave_venous"],
    }
    return arterial, venous, leaks


def _dynamic_pressure_transform(
    reference_organoid: int,
    target_organoid: int,
    leak_pressures_in: dict[tuple[int, str], float],
    offset_anchor: str,
    drop_floor: float,
) -> dict[str, float]:
    ref_art = float(leak_pressures_in[(reference_organoid, "arterial")])
    ref_ven = float(leak_pressures_in[(reference_organoid, "venous")])
    target_art = float(leak_pressures_in[(target_organoid, "arterial")])
    target_ven = float(leak_pressures_in[(target_organoid, "venous")])
    reference_drop = ref_art - ref_ven
    target_drop = target_art - target_ven
    if abs(reference_drop) <= max(float(drop_floor), np.finfo(float).tiny):
        raise ValueError(
            "Reference 0D leak pressure drop is too small for dynamic scaling: "
            f"{reference_drop:.6g}"
        )
    scale = target_drop / reference_drop
    offset_from_artery = target_art - scale * ref_art
    offset_from_vein = target_ven - scale * ref_ven
    anchor = str(offset_anchor).strip().lower()
    if anchor == "arterial":
        offset = offset_from_artery
    elif anchor == "venous":
        offset = offset_from_vein
    else:
        raise ValueError(f"Unsupported scaled pressure anchor {offset_anchor!r}")
    return {
        "scale_factor": float(scale),
        "pressure_offset": float(offset),
        "offset_from_artery": float(offset_from_artery),
        "offset_from_vein": float(offset_from_vein),
        "offset_mismatch": float(offset_from_artery - offset_from_vein),
        "reference_pressure_drop": float(reference_drop),
        "target_pressure_drop": float(target_drop),
    }


def make_dynamic_scaled_operator(
    reference_map: DarcyMap,
    network: PressureBoundaryZeroD,
    layout: InterfaceLayout,
    n_organoids: int,
    reference_organoid: int,
    offset_anchor: str,
    face_length: float,
    compartment_spacing: float,
    drop_floor: float,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], dict[int, dict[str, float]]]]:
    if int(reference_map.organoid) != int(reference_organoid):
        raise ValueError("Reference Darcy map organoid does not match scaled reference organoid")

    def evaluate_with_transforms(
        state: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, dict[str, float]]]:
        terminal_pressures, leak_flows = layout.unpack(np.asarray(state, dtype=float))
        terminal_q, leak_p_out, leak_p_in = network.evaluate(
            terminal_pressures, leak_flows
        )
        ref_art, ref_ven, ref_leaks = _reference_darcy_outputs(
            reference_map,
            terminal_q,
            leak_p_out,
            face_length,
            compartment_spacing,
        )
        predicted_p: dict[str, float] = {}
        predicted_leaks: dict[tuple[int, str], float] = {}
        transforms: dict[int, dict[str, float]] = {}
        for organoid in range(1, int(n_organoids) + 1):
            if organoid == int(reference_organoid):
                transform = {
                    "scale_factor": 1.0,
                    "pressure_offset": 0.0,
                    "offset_from_artery": 0.0,
                    "offset_from_vein": 0.0,
                    "offset_mismatch": 0.0,
                    "reference_pressure_drop": float(
                        leak_p_in[(reference_organoid, "arterial")]
                        - leak_p_in[(reference_organoid, "venous")]
                    ),
                    "target_pressure_drop": float(
                        leak_p_in[(reference_organoid, "arterial")]
                        - leak_p_in[(reference_organoid, "venous")]
                    ),
                }
            else:
                transform = _dynamic_pressure_transform(
                    reference_organoid,
                    organoid,
                    leak_p_in,
                    offset_anchor,
                    drop_floor,
                )
            transforms[organoid] = transform
            scale = float(transform["scale_factor"])
            offset = float(transform["pressure_offset"])
            for branch in reference_map.branch_ids_inlet:
                predicted_p[_terminal_name(organoid, "arterial", branch)] = (
                    offset + scale * float(ref_art[str(branch)])
                )
            for branch in reference_map.branch_ids_outlet:
                predicted_p[_terminal_name(organoid, "venous", branch)] = (
                    offset + scale * float(ref_ven[str(branch)])
                )
            predicted_leaks[(organoid, "arterial")] = scale * float(
                ref_leaks["arterial"]
            )
            predicted_leaks[(organoid, "venous")] = scale * float(
                ref_leaks["venous"]
            )
        return layout.pack(predicted_p, predicted_leaks), transforms

    def evaluate(state: np.ndarray) -> np.ndarray:
        return evaluate_with_transforms(state)[0]

    def transforms(state: np.ndarray) -> dict[int, dict[str, float]]:
        return evaluate_with_transforms(state)[1]

    return evaluate, transforms


def _static_scaled_transforms(
    summary: dict[str, Any],
    n_organoids: int,
    reference_organoid: int,
) -> dict[int, dict[str, float]]:
    """Load fixed pressure/flow transforms from the prepared screening summary."""
    prepared = {
        int(row["organoid_id"]): dict(row)
        for row in summary.get("scaled_organoids", [])
        if "organoid_id" in row
    }
    transforms: dict[int, dict[str, float]] = {}
    missing: list[int] = []
    for organoid in range(1, int(n_organoids) + 1):
        if organoid == int(reference_organoid):
            transforms[organoid] = {
                "scale_factor": 1.0,
                "pressure_offset": 0.0,
                "offset_from_artery": 0.0,
                "offset_from_vein": 0.0,
                "offset_mismatch": 0.0,
            }
            continue
        row = prepared.get(organoid)
        if row is None or "scale_factor" not in row or "pressure_offset" not in row:
            missing.append(organoid)
            continue
        scale = float(row["scale_factor"])
        offset = float(row["pressure_offset"])
        if not np.isfinite(scale) or not np.isfinite(offset):
            raise ValueError(f"Static transform for organoid {organoid} is not finite")
        transforms[organoid] = {
            "scale_factor": scale,
            "pressure_offset": offset,
            "offset_from_artery": float(row.get("offset_from_artery", offset)),
            "offset_from_vein": float(row.get("offset_from_vein", offset)),
            "offset_mismatch": float(row.get("offset_mismatch", 0.0)),
        }
        for key in ("target_arterial_pressure", "target_venous_pressure"):
            if key in row:
                transforms[organoid][key] = float(row[key])
    if missing:
        raise ValueError(
            "Static-scaled mode requires scale_factor and pressure_offset for every "
            f"non-reference organoid; missing organoids {missing}"
        )
    return transforms


def make_static_scaled_operator(
    reference_map: DarcyMap,
    network: PressureBoundaryZeroD,
    reference_layout: InterfaceLayout,
    full_layout: InterfaceLayout,
    transforms: dict[int, dict[str, float]],
    face_length: float,
    compartment_spacing: float,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], dict[int, dict[str, float]]]]:
    """Solve only the reference interface; expand all other wells statically."""

    def evaluate(state: np.ndarray) -> np.ndarray:
        full_state = expand_static_reference_state(
            np.asarray(state, dtype=float),
            reference_layout,
            full_layout,
            reference_map,
            transforms,
        )
        terminal_pressures, leak_flows = full_layout.unpack(full_state)
        terminal_q, leak_p_out, _leak_p_in = network.evaluate(
            terminal_pressures, leak_flows
        )
        ref_art, ref_ven, ref_leaks = _reference_darcy_outputs(
            reference_map,
            terminal_q,
            leak_p_out,
            face_length,
            compartment_spacing,
        )
        predicted_p = {
            _terminal_name(reference_map.organoid, "arterial", branch): ref_art[str(branch)]
            for branch in reference_map.branch_ids_inlet
        }
        predicted_p.update({
            _terminal_name(reference_map.organoid, "venous", branch): ref_ven[str(branch)]
            for branch in reference_map.branch_ids_outlet
        })
        predicted_leaks = {
            (reference_map.organoid, "arterial"): ref_leaks["arterial"],
            (reference_map.organoid, "venous"): ref_leaks["venous"],
        }
        return reference_layout.pack(predicted_p, predicted_leaks)

    def report_transforms(_state: np.ndarray) -> dict[int, dict[str, float]]:
        return {key: dict(value) for key, value in transforms.items()}

    return evaluate, report_transforms


def expand_static_reference_state(
    reference_state: np.ndarray,
    reference_layout: InterfaceLayout,
    full_layout: InterfaceLayout,
    reference_map: DarcyMap,
    transforms: dict[int, dict[str, float]],
) -> np.ndarray:
    """Apply frozen transforms directly to one solved reference interface state."""
    reference_p, reference_q = reference_layout.unpack(
        np.asarray(reference_state, dtype=float)
    )
    full_p: dict[str, float] = {}
    full_q: dict[tuple[int, str], float] = {}
    reference_organoid = int(reference_map.organoid)
    for organoid, transform in sorted(transforms.items()):
        scale = float(transform["scale_factor"])
        offset = float(transform["pressure_offset"])
        for side, branches in (
            ("arterial", reference_map.branch_ids_inlet),
            ("venous", reference_map.branch_ids_outlet),
        ):
            for branch in branches:
                reference_name = _terminal_name(reference_organoid, side, branch)
                full_p[_terminal_name(organoid, side, branch)] = (
                    offset + scale * reference_p[reference_name]
                )
            full_q[(organoid, side)] = scale * reference_q[(reference_organoid, side)]
    return full_layout.pack(full_p, full_q)


def solve_affine_fixed_point(
    operator: Callable[[np.ndarray], np.ndarray],
    reference: np.ndarray,
    scales: np.ndarray,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    """Identify an affine operator by basis evaluation and solve x = F(x)."""
    reference = np.asarray(reference, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if reference.ndim != 1 or scales.shape != reference.shape:
        raise ValueError("Reference and basis scales must be equal-length vectors")
    if np.any(~np.isfinite(reference)) or np.any(~np.isfinite(scales)) or np.any(scales == 0.0):
        raise ValueError("Reference and basis scales must be finite; scales must be nonzero")
    f_reference = np.asarray(operator(reference), dtype=float)
    jacobian = np.empty((reference.size, reference.size), dtype=float)
    scaled_jacobian = np.empty_like(jacobian)
    for column, scale in enumerate(scales):
        perturbed = reference.copy()
        perturbed[column] += float(scale)
        response_delta = np.asarray(operator(perturbed)) - f_reference
        jacobian[:, column] = response_delta / float(scale)
        scaled_jacobian[:, column] = response_delta / scales
    affine_offset = f_reference - jacobian @ reference
    # Solve in y = D^-1 (x-reference), where D=diag(scales).  The raw
    # pressure/flow matrix mixes O(1e2) pressures with O(1e-10) flows and can
    # be catastrophically ill-conditioned even when the dimensionless fixed
    # point is well posed.
    matrix = np.eye(reference.size, dtype=float) - scaled_jacobian
    rhs = (f_reference - reference) / scales
    condition = float(np.linalg.cond(matrix))
    try:
        scaled_solution = np.linalg.solve(matrix, rhs)
        mode = "direct"
    except np.linalg.LinAlgError:
        scaled_solution = np.linalg.lstsq(matrix, rhs, rcond=float(rcond))[0]
        mode = "least_squares"
    solution = reference + scales * scaled_solution
    # Iterative refinement evaluates the actual composed operator and removes
    # amplification of floating-point error from a moderately conditioned
    # interface matrix.  No additional PDE solves occur here.
    previous_norm = float("inf")
    refinement_count = 0
    for _iteration in range(8):
        scaled_residual = (np.asarray(operator(solution)) - solution) / scales
        norm = float(np.max(np.abs(scaled_residual)))
        if not np.isfinite(norm) or norm >= previous_norm:
            break
        previous_norm = norm
        if norm <= 1.0e-10:
            break
        try:
            correction = np.linalg.solve(matrix, scaled_residual)
        except np.linalg.LinAlgError:
            correction = np.linalg.lstsq(matrix, scaled_residual, rcond=float(rcond))[0]
        candidate = solution + scales * correction
        candidate_residual = (np.asarray(operator(candidate)) - candidate) / scales
        if float(np.max(np.abs(candidate_residual))) >= norm:
            break
        solution = candidate
        refinement_count += 1
    if refinement_count:
        mode += f"_refined_{refinement_count}"
    return solution, affine_offset, jacobian, mode, condition


def solve_dynamic_fixed_point(
    operator: Callable[[np.ndarray], np.ndarray],
    reference: np.ndarray,
    scales: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int,
    minimum_step: float,
    rcond: float,
) -> tuple[np.ndarray, list[dict[str, float]], str, float]:
    """Solve a scaled nonlinear fixed point with damped good-Broyden updates."""
    reference = np.asarray(reference, dtype=float)
    scales = np.asarray(scales, dtype=float)
    if reference.shape != scales.shape or reference.ndim != 1:
        raise ValueError("Dynamic reference/scales must be equal-length vectors")
    if np.any(~np.isfinite(reference)) or np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("Dynamic reference and scales must be finite and scales positive")

    def state_from_y(y: np.ndarray) -> np.ndarray:
        return reference + scales * np.asarray(y, dtype=float)

    def residual_at_y(y: np.ndarray) -> np.ndarray:
        state = state_from_y(y)
        return (state - np.asarray(operator(state), dtype=float)) / scales

    y = np.zeros(reference.size, dtype=float)
    residual = residual_at_y(y)
    # One scaled basis sweep supplies a strong initial secant Jacobian.  It is
    # exact for the affine 0D/reference-Darcy pieces; Broyden then updates only
    # the nonlinear pressure-drop-ratio contribution from scaled wells.
    jacobian = np.empty((reference.size, reference.size), dtype=float)
    for column in range(reference.size):
        perturbed = y.copy()
        perturbed[column] += 1.0
        jacobian[:, column] = residual_at_y(perturbed) - residual

    history: list[dict[str, float]] = []
    solve_mode = "dynamic_broyden"
    stalled = False
    for iteration in range(max(int(max_iterations), 1)):
        norm = float(np.max(np.abs(residual)))
        history.append({"iteration": float(iteration), "normalized_residual_linf": norm})
        if norm <= float(tolerance):
            return state_from_y(y), history, solve_mode, float(np.linalg.cond(jacobian))
        try:
            step = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(jacobian, -residual, rcond=float(rcond))[0]
            solve_mode = "dynamic_broyden_least_squares"
        if not np.all(np.isfinite(step)):
            raise RuntimeError("Dynamic interface solver produced a non-finite step")

        accepted = False
        damping = 1.0
        trial_y = y
        trial_residual = residual
        trial_norm = norm
        while damping >= float(minimum_step):
            candidate_y = y + damping * step
            try:
                candidate_residual = residual_at_y(candidate_y)
            except (ValueError, FloatingPointError):
                damping *= 0.5
                continue
            candidate_norm = float(np.max(np.abs(candidate_residual)))
            if np.isfinite(candidate_norm) and candidate_norm < norm:
                trial_y = candidate_y
                trial_residual = candidate_residual
                trial_norm = candidate_norm
                accepted = True
                break
            damping *= 0.5
        history[-1]["accepted_step_scale"] = float(damping if accepted else 0.0)
        history[-1]["trial_normalized_residual_linf"] = float(trial_norm)
        if not accepted:
            stalled = True
            break

        delta_y = trial_y - y
        delta_residual = trial_residual - residual
        denominator = float(delta_y @ delta_y)
        if denominator > np.finfo(float).tiny:
            jacobian += np.outer(
                delta_residual - jacobian @ delta_y, delta_y
            ) / denominator
        y = trial_y
        residual = trial_residual

    # The dynamic scaling ratio can make a rank-one Broyden model temporarily
    # poor far from the solution.  The project runtime includes SciPy, so use
    # its trust-region least-squares solver as a robust fallback.  This still
    # evaluates only the dense 0D network and the cached Darcy matrix map.
    try:
        from scipy.optimize import least_squares, root
    except ImportError as exc:
        final_norm = float(np.max(np.abs(residual)))
        reason = "stalled" if stalled else "reached its iteration limit"
        raise RuntimeError(
            f"Dynamic Broyden solve {reason} with normalized residual "
            f"{final_norm:.3e}; SciPy is required for the trust-region fallback"
        ) from exc
    evaluations = max(max(int(max_iterations), 1) * (reference.size + 1), 1000)
    result = least_squares(
        residual_at_y,
        y,
        jac="2-point",
        x_scale="jac",
        ftol=max(float(tolerance) * 0.1, 1.0e-12),
        xtol=max(float(tolerance) * 0.1, 1.0e-12),
        gtol=max(float(tolerance) * 0.1, 1.0e-12),
        max_nfev=evaluations,
    )
    final_residual = residual_at_y(result.x)
    final_norm = float(np.max(np.abs(final_residual)))
    final_y = np.asarray(result.x, dtype=float)
    refinement = None
    if final_norm > float(tolerance):
        refinement = root(
            residual_at_y,
            final_y,
            method="lm",
            options={"ftol": float(tolerance) * 0.1, "xtol": float(tolerance) * 0.1,
                     "gtol": float(tolerance) * 0.1, "maxiter": evaluations},
        )
        refined_residual = residual_at_y(refinement.x)
        refined_norm = float(np.max(np.abs(refined_residual)))
        if np.isfinite(refined_norm) and refined_norm < final_norm:
            final_y = np.asarray(refinement.x, dtype=float)
            final_residual = refined_residual
            final_norm = refined_norm
    history.append({
        "iteration": float(len(history)),
        "normalized_residual_linf": final_norm,
        "scipy_function_evaluations": float(result.nfev),
        "scipy_cost": float(result.cost),
        "scipy_optimality": float(result.optimality),
        "scipy_refinement_evaluations": float(
            getattr(refinement, "nfev", 0) if refinement is not None else 0
        ),
    })
    if final_norm > float(tolerance):
        raise RuntimeError(
            "Dynamic interface trust-region solve failed: "
            f"status={result.status}, message={result.message!s}, "
            f"normalized residual={final_norm:.3e}"
        )
    return (
        state_from_y(final_y),
        history,
        "dynamic_broyden_scipy_trust_region",
        float(np.linalg.cond(jacobian)),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_region_path(raw: Path, reference_organoid: int) -> Path:
    raw = raw.expanduser().resolve()
    if raw.is_file():
        return raw
    if not raw.is_dir():
        raise FileNotFoundError(f"Permeability region does not exist: {raw}")
    candidates = [
        raw / f"organoid-{reference_organoid}.stl",
        raw / f"organoid_{reference_organoid}.stl",
        raw / f"organoid{reference_organoid}.stl",
    ]
    return next((path for path in candidates if path.is_file()), raw)


def _map_matches_requested_physics(
    map_path: Path,
    prepped_dir: Path,
    reference_organoid: int,
    perm_region: Path,
    perm_low: float,
    perm_high: float,
    transition_width: float,
    anisotropy_ratio: float,
    concave_bc_mode: str,
) -> bool:
    metadata_path = map_path.with_suffix(".json")
    if not map_path.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = _load_json(metadata_path)
        if not bool(metadata.get("globally_identified", False)):
            return False
        permeability = metadata.get("permeability", {})
        mapped_region = Path(str(permeability.get("requested_path", ""))).expanduser().resolve()
        requested_region = _resolved_region_path(perm_region, reference_organoid)
        if mapped_region != requested_region and mapped_region != perm_region.expanduser().resolve():
            return False
        requested = {
            "perm_low": float(perm_low),
            "perm_high": float(perm_high),
            "perm_transition_width": float(transition_width),
            "anisotropy_factor_inside_stl": float(anisotropy_ratio),
        }
        if any(
            not np.isclose(float(permeability.get(key, np.nan)), value, rtol=1.0e-12)
            for key, value in requested.items()
        ):
            return False
        if str(metadata.get("concave_bc_mode", "dirichlet")) != str(concave_bc_mode):
            return False
        geometry = prepped_dir / f"organoid_{reference_organoid}" / "geometry"
        for name, expected in metadata.get("geometry_sha256", {}).items():
            path = geometry / str(name)
            if not path.is_file() or _sha256(path) != str(expected):
                return False
        return bool(metadata.get("geometry_sha256", {}))
    except Exception:
        return False


def prepare_reference_map(
    *,
    prepped_dir: Path,
    reference_organoid: int,
    output: Path,
    python: Path,
    darcy_script: Path,
    perm_region: Path,
    perm_low: float,
    perm_high: float,
    transition_width: float,
    anisotropy_ratio: float,
    concave_bc_mode: str,
    rebuild: bool,
) -> Path:
    prepped_dir = prepped_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if not rebuild and _map_matches_requested_physics(
        output,
        prepped_dir,
        reference_organoid,
        perm_region,
        perm_low,
        perm_high,
        transition_width,
        anisotropy_ratio,
        concave_bc_mode,
    ):
        print(f"[darcy-map] Reusing geometry/physics-matched map {output}", flush=True)
        return output
    builder = REPO_SRC / "solves" / "build_darcy_response_map.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python.expanduser()),
        str(builder),
        "exact",
        "--baseline-run-dir",
        str(prepped_dir),
        "--organoid",
        str(int(reference_organoid)),
        "--output",
        str(output),
        "--darcy-script",
        str(darcy_script.expanduser().resolve()),
        "--concave-bc-mode",
        str(concave_bc_mode),
        "--perm-region-path",
        str(perm_region.expanduser().resolve()),
        "--perm-low",
        str(float(perm_low)),
        "--perm-high",
        str(float(perm_high)),
        "--perm-transition-width",
        str(float(transition_width)),
        "--stl-anisotropy-factor",
        str(float(anisotropy_ratio)),
    ]
    print(f"[darcy-map] Building exact reference map at {output}", flush=True)
    subprocess.run(command, check=True)
    return output


def _parse_map_spec(raw: str) -> tuple[int, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Darcy maps must use ORGANOID=PATH")
    organoid, path = raw.split("=", 1)
    try:
        organoid_idx = int(organoid)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid organoid index in {raw!r}") from exc
    return organoid_idx, Path(path)


def _baseline_reference(maps: list[DarcyMap], layout: InterfaceLayout) -> np.ndarray:
    pressures: dict[str, float] = {}
    leaks: dict[tuple[int, str], float] = {}
    for model in maps:
        output = {
            name: float(value)
            for name, value in zip(model.output_names, model.baseline_outputs, strict=True)
        }
        for branch in model.branch_ids_inlet:
            pressures[_terminal_name(model.organoid, "arterial", branch)] = output[
                f"p_arterial_branch_{branch}"
            ]
        for branch in model.branch_ids_outlet:
            pressures[_terminal_name(model.organoid, "venous", branch)] = output[
                f"p_venous_branch_{branch}"
            ]
        leaks[(model.organoid, "arterial")] = output["q_concave_arterial"]
        leaks[(model.organoid, "venous")] = output["q_concave_venous"]
    return layout.pack(pressures, leaks)


def _scaled_baseline_reference(
    reference_map: DarcyMap,
    layout: InterfaceLayout,
    n_organoids: int,
    reference_organoid: int,
    summary: dict[str, Any],
) -> np.ndarray:
    output = {
        name: float(value)
        for name, value in zip(
            reference_map.output_names, reference_map.baseline_outputs, strict=True
        )
    }
    prepared = {
        int(row["organoid_id"]): row
        for row in summary.get("scaled_organoids", [])
        if "organoid_id" in row
    }
    pressures: dict[str, float] = {}
    leaks: dict[tuple[int, str], float] = {}
    for organoid in range(1, int(n_organoids) + 1):
        row = prepared.get(organoid, {})
        scale = 1.0 if organoid == reference_organoid else float(row.get("scale_factor", 1.0))
        offset = 0.0 if organoid == reference_organoid else float(row.get("pressure_offset", 0.0))
        for branch in reference_map.branch_ids_inlet:
            pressures[_terminal_name(organoid, "arterial", branch)] = (
                offset + scale * output[f"p_arterial_branch_{branch}"]
            )
        for branch in reference_map.branch_ids_outlet:
            pressures[_terminal_name(organoid, "venous", branch)] = (
                offset + scale * output[f"p_venous_branch_{branch}"]
            )
        leaks[(organoid, "arterial")] = scale * output["q_concave_arterial"]
        leaks[(organoid, "venous")] = scale * output["q_concave_venous"]
    return layout.pack(pressures, leaks)


def _infer_organoid_count(network: PressureBoundaryZeroD) -> int:
    ids: set[int] = set()
    for name in network.terminal_vessel_by_name:
        match = re.match(r"^organoid(\d+)_", name, re.I)
        if match:
            ids.add(int(match.group(1)))
    if not ids or ids != set(range(1, max(ids) + 1)):
        raise ValueError(f"Could not infer contiguous organoid IDs from terminal BCs: {sorted(ids)}")
    return max(ids)


def run_full_verification(
    *,
    run_dir: Path,
    combined_path: Path,
    prepped_dir: Path,
    n_organoids: int,
    svzerodsolver: Path,
    run_and_split: Path,
    darcy_script: Path,
    perm_region_path: Path,
    perm_low: float,
    perm_high: float,
    transition_width: float,
    anisotropy_ratio: float,
    concave_bc_mode: str,
    offset_anchor: str,
    profile_face_length: float,
    compartment_spacing: float,
    dy_step: float,
    coords_inlet: list[float],
    coords_outlet: list[float],
    darcy_mpi_procs: int,
    darcy_mpirun_cmd: str,
    scaled_transform_mode: str,
    scaled_transforms: dict[int, dict[str, float]],
) -> dict[str, Any]:
    """Run svZeroD and a full reference Darcy solve, then synthesize scaled fields."""
    from coupling import coupler

    run_dir = run_dir.expanduser().resolve()
    prepped_dir = prepped_dir.expanduser().resolve()
    combined_path = combined_path.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    coupler.ensure_organoid_dirs(run_dir, int(n_organoids))
    coupler.sync_plot_templates(prepped_dir, run_dir, int(n_organoids))
    coupler.copy_seed_geometry(prepped_dir, run_dir, int(n_organoids))
    coupler.copy_branching_files(prepped_dir, run_dir, int(n_organoids))

    coupler.run([
        sys.executable,
        str(run_and_split.expanduser().resolve()),
        "--exe", str(svzerodsolver.expanduser()),
        "--input", str(combined_path),
        "--outdir", str(run_dir),
        "--output", "output.csv",
        "--organoid-root", str(run_dir),
        "--no-plot",
    ])
    coupler.update_1d_checkpoints(
        run_dir,
        int(n_organoids),
        np.asarray(coords_inlet, dtype=float),
        np.asarray(coords_outlet, dtype=float),
        float(dy_step),
    )

    verification_args = SimpleNamespace(
        n_organoids=int(n_organoids),
        no_synthetic_vasculature=False,
        darcy_response_map="",
        darcy_script=str(darcy_script.expanduser().resolve()),
        trial_dir=str(prepped_dir),
        first_organoid_linear_profile=True,
        concave_profile_face_length=float(profile_face_length),
        concave_profile_compartment_spacing=float(compartment_spacing),
        coords_inlet=list(coords_inlet),
        coords_outlet=list(coords_outlet),
        dy_step=float(dy_step),
        perm_region_root=str(perm_region_path.expanduser().resolve()),
        fallback_inlet_pressure=181.0,
        fallback_outlet_pressure=21.0,
        concave_bc_mode=str(concave_bc_mode),
        lp_arterial=0.0,
        lp_venous=0.0,
        perm_low=float(perm_low),
        perm_high=float(perm_high),
        perm_transition_width=float(transition_width),
        darcy_stl_anisotropy_factor=float(anisotropy_ratio),
        darcy_output_mode="full",
        darcy_diagnostics_mode="full",
        darcy_mpi_procs=int(darcy_mpi_procs),
        darcy_mpirun_cmd=str(darcy_mpirun_cmd),
        scaled_screening_mode="on",
        scaled_pressure_offset_anchor=str(offset_anchor),
    )
    scaled_cfg = coupler.load_scaled_screening_config(prepped_dir, verification_args)
    if scaled_cfg is None:
        raise ValueError("Could not load scaled-screening configuration for verification")
    scaled_cfg["transform_mode"] = str(scaled_transform_mode)
    if str(scaled_transform_mode) == "static":
        scaled_cfg["prepared_transforms"] = {
            int(key): dict(value) for key, value in scaled_transforms.items()
            if int(key) != int(scaled_cfg["reference_organoid"])
        }
    coupler.run_darcy_for_all(
        run_dir,
        verification_args,
        scaled_cfg=scaled_cfg,
        force_pde=True,
        output_mode_override="full",
        diagnostics_mode_override="full",
    )
    coupler.write_post_darcy_terminal_convergence(
        run_dir, int(n_organoids), write_cumulative=False
    )
    coupler.write_post_darcy_concave_convergence(
        run_dir, int(n_organoids), write_cumulative=False
    )
    required: list[str] = ["output.csv", "post_darcy_terminal_convergence.csv",
                           "post_darcy_concave_convergence.csv"]
    for organoid in range(1, int(n_organoids) + 1):
        for name in ("p.xdmf", "p.h5", "u.xdmf", "u.h5", "interface_bc.json"):
            required.append(f"organoid_{organoid}/out_darcy/{name}")
    missing = [name for name in required if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError("Full verification omitted transport outputs: " + ", ".join(missing))
    return {
        "run_dir": str(run_dir),
        "output_csv": str(run_dir / "output.csv"),
        "full_darcy_reference_organoid": int(scaled_cfg["reference_organoid"]),
        "scaled_field_organoids": list(range(1, int(n_organoids) + 1)),
        "transport_outputs_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deck", "--combined-in", dest="deck", type=Path, required=True,
        help="Input combined 0D JSON deck",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "independent", "dynamic-scaled", "static-scaled"],
        default="auto",
        help=(
            "Auto selects dynamic-scaled when --prepped-dir is supplied. "
            "Static-scaled holds the prepared scale_factor and pressure_offset fixed."
        ),
    )
    parser.add_argument(
        "--darcy-map",
        action="append",
        type=_parse_map_spec,
        metavar="ORGANOID=PATH",
        help="Existing globally exact Darcy map. Optional in either scaled mode.",
    )
    parser.add_argument(
        "--prepped-dir", "--trial-dir", dest="prepped_dir", type=Path,
        help="Prepared folder containing organoid_1 geometry, branching, and seed outputs.",
    )
    parser.add_argument("--n-organoids", type=int, default=0, help="Default: infer from combined.in")
    parser.add_argument("--reference-organoid", type=int, default=1)
    parser.add_argument("--scaled-screening-summary", type=Path, default=None)
    parser.add_argument(
        "--scaled-pressure-offset-anchor", choices=["arterial", "venous"], default="arterial"
    )
    parser.add_argument(
        "--reference-map-output", type=Path, default=None,
        help="Default: PREPPED_DIR/response_maps/exact_global_organoidN.npz",
    )
    parser.add_argument(
        "--map-python", type=Path, default=Path(sys.executable),
        help="Python executable containing DOLFINx/PETSc for automatic map construction.",
    )
    parser.add_argument(
        "--darcy-script", type=Path, default=REPO_SRC / "solves" / "darcy_mixed.py"
    )
    parser.add_argument(
        "--perm-region-path", "--perm-region-root", dest="perm_region_path", type=Path
    )
    parser.add_argument("--perm-low", type=float)
    parser.add_argument("--perm-high", type=float)
    parser.add_argument("--perm-transition-width", type=float, default=0.02)
    parser.add_argument(
        "--anisotropy-ratio", "--darcy-stl-anisotropy-factor",
        dest="anisotropy_ratio", type=float,
    )
    parser.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    parser.add_argument("--rebuild-darcy-response-map", action="store_true")
    parser.add_argument("--out-deck", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--linear-profile-face-length", type=float, default=0.4)
    parser.add_argument("--compartment-spacing", type=float, default=0.6)
    parser.add_argument("--pressure-basis-scale", type=float, default=1.0)
    parser.add_argument("--flow-basis-scale", type=float, default=1.0e-9)
    parser.add_argument("--rcond", type=float, default=1.0e-12)
    parser.add_argument(
        "--normalized-residual-tol", type=float, default=2.0e-4,
        help="Scaled interface residual tolerance; 2e-4 is stricter than the former 0.05%% pressure criterion.",
    )
    parser.add_argument("--dynamic-max-iterations", type=int, default=30)
    parser.add_argument("--dynamic-minimum-step", type=float, default=1.0e-4)
    parser.add_argument("--dynamic-reference-drop-floor", type=float, default=1.0e-8)
    parser.add_argument(
        "--run-full-verification", action="store_true",
        help="Run svZeroD, full reference Darcy diagnostics, and scaled-well field synthesis.",
    )
    parser.add_argument(
        "--verification-dir", type=Path, default=None,
        help="Transport-ready output directory; default is the --out-deck parent.",
    )
    parser.add_argument(
        "--svzerodsolver", type=Path,
        default=Path("/opt/miniconda3/envs/fenicsx-env/bin/svzerodsolver"),
    )
    parser.add_argument(
        "--run-and-split", type=Path, default=REPO_SRC / "prep" / "run_and_split_svzerod.py"
    )
    parser.add_argument("--dy-step", type=float, default=0.6)
    parser.add_argument("--coords-inlet", type=float, nargs=3, default=[-0.28, 0.9, 0.5375])
    parser.add_argument("--coords-outlet", type=float, nargs=3, default=[0.30, 0.9, 0.5375])
    parser.add_argument("--darcy-mpi-procs", type=int, default=1)
    parser.add_argument("--darcy-mpirun-cmd", default="mpiexec")
    args = parser.parse_args()

    deck_path = args.deck.expanduser().resolve()
    deck = _load_json(deck_path)
    network = PressureBoundaryZeroD(deck)
    n_organoids = int(args.n_organoids) if int(args.n_organoids) > 0 else _infer_organoid_count(network)
    mode = str(args.mode)
    if mode == "auto":
        mode = "dynamic-scaled" if args.prepped_dir is not None else "independent"

    map_specs = list(args.darcy_map or [])
    if len({organoid for organoid, _path in map_specs}) != len(map_specs):
        parser.error("Each organoid may have only one --darcy-map")
    prepped_dir = args.prepped_dir.expanduser().resolve() if args.prepped_dir else None
    summary: dict[str, Any] = {}
    summary_path = args.scaled_screening_summary
    if summary_path is None and prepped_dir is not None:
        candidate = prepped_dir / "scaled_screening_summary.json"
        if candidate.is_file():
            summary_path = candidate
    if summary_path is not None:
        summary_path = summary_path.expanduser().resolve()
        summary = _load_json(summary_path)
    reference_organoid = int(summary.get("reference_organoid", args.reference_organoid))

    if mode in {"dynamic-scaled", "static-scaled"}:
        if map_specs:
            if len(map_specs) != 1:
                parser.error(f"{mode} mode accepts at most one reference --darcy-map")
            map_organoid, map_path = map_specs[0]
            if int(map_organoid) != reference_organoid:
                parser.error("Reference --darcy-map index does not match reference organoid")
        else:
            if prepped_dir is None:
                parser.error(f"{mode} automatic map construction requires --prepped-dir")
            missing_physics = [
                name for name, value in (
                    ("--perm-region-path", args.perm_region_path),
                    ("--perm-low", args.perm_low),
                    ("--perm-high", args.perm_high),
                    ("--anisotropy-ratio", args.anisotropy_ratio),
                ) if value is None
            ]
            if missing_physics:
                parser.error("Automatic map construction requires " + ", ".join(missing_physics))
            map_path = args.reference_map_output or (
                prepped_dir / "response_maps" / f"exact_global_organoid{reference_organoid}.npz"
            )
            map_path = prepare_reference_map(
                prepped_dir=prepped_dir,
                reference_organoid=reference_organoid,
                output=map_path,
                python=args.map_python,
                darcy_script=args.darcy_script,
                perm_region=args.perm_region_path,
                perm_low=float(args.perm_low),
                perm_high=float(args.perm_high),
                transition_width=float(args.perm_transition_width),
                anisotropy_ratio=float(args.anisotropy_ratio),
                concave_bc_mode=str(args.concave_bc_mode),
                rebuild=bool(args.rebuild_darcy_response_map),
            )
        reference_map = DarcyMap.load(reference_organoid, map_path)
        maps = [reference_map]
        layout = build_scaled_layout(reference_map, n_organoids)
        if mode == "dynamic-scaled":
            solve_layout = layout
            operator, transform_evaluator = make_dynamic_scaled_operator(
                reference_map,
                network,
                layout,
                n_organoids=n_organoids,
                reference_organoid=reference_organoid,
                offset_anchor=str(args.scaled_pressure_offset_anchor),
                face_length=float(args.linear_profile_face_length),
                compartment_spacing=float(args.compartment_spacing),
                drop_floor=float(args.dynamic_reference_drop_floor),
            )
            reference = _scaled_baseline_reference(
                reference_map,
                layout,
                n_organoids,
                reference_organoid,
                summary,
            )
        else:
            if not summary:
                parser.error(
                    "static-scaled mode requires --scaled-screening-summary or a "
                    "scaled_screening_summary.json in --prepped-dir"
                )
            static_transforms = _static_scaled_transforms(
                summary, n_organoids, reference_organoid
            )
            solve_layout = build_layout([reference_map])
            operator, transform_evaluator = make_static_scaled_operator(
                reference_map,
                network,
                reference_layout=solve_layout,
                full_layout=layout,
                transforms=static_transforms,
                face_length=float(args.linear_profile_face_length),
                compartment_spacing=float(args.compartment_spacing),
            )
            reference = _baseline_reference([reference_map], solve_layout)
    else:
        if not map_specs:
            parser.error("independent mode requires one --darcy-map per organoid")
        maps = [DarcyMap.load(organoid, path) for organoid, path in map_specs]
        maps.sort(key=lambda item: item.organoid)
        layout = build_layout(maps)
        solve_layout = layout
        operator = make_interface_operator(
            maps,
            network,
            layout,
            face_length=float(args.linear_profile_face_length),
            compartment_spacing=float(args.compartment_spacing),
        )
        transform_evaluator = None
        reference = _baseline_reference(maps, layout)

    missing = sorted(set(layout.terminal_names) - set(network.terminal_vessel_by_name))
    extra = sorted(set(network.terminal_vessel_by_name) - set(layout.terminal_names))
    if missing or extra:
        raise ValueError(
            "0D/Darcy terminal coverage mismatch: "
            f"missing_in_deck={missing[:8]}, missing_in_maps={extra[:8]}"
        )
    scales = np.concatenate(
        (
            np.full(len(solve_layout.terminal_names), abs(float(args.pressure_basis_scale))),
            np.full(len(solve_layout.leak_keys), abs(float(args.flow_basis_scale))),
        )
    )
    nonlinear_history: list[dict[str, float]] = []
    affine_offset: np.ndarray | None = None
    jacobian: np.ndarray | None = None
    if mode == "dynamic-scaled":
        solution, nonlinear_history, solve_mode, condition = solve_dynamic_fixed_point(
            operator,
            reference,
            scales,
            tolerance=float(args.normalized_residual_tol),
            max_iterations=int(args.dynamic_max_iterations),
            minimum_step=float(args.dynamic_minimum_step),
            rcond=float(args.rcond),
        )
    else:
        solution, affine_offset, jacobian, solve_mode, condition = solve_affine_fixed_point(
            operator, reference, scales, rcond=float(args.rcond)
        )
    predicted = operator(solution)
    residual = predicted - solution
    component_scale = np.maximum(np.abs(solution), scales)
    normalized_residual = residual / component_scale
    normalized_linf = float(np.max(np.abs(normalized_residual)))
    if not np.all(np.isfinite(solution)) or normalized_linf > float(args.normalized_residual_tol):
        raise RuntimeError(
            f"Direct interface residual {normalized_linf:.3e} exceeds tolerance "
            f"{float(args.normalized_residual_tol):.3e}"
        )

    solved_reference_state = solution
    if mode == "static-scaled":
        solution = expand_static_reference_state(
            solved_reference_state,
            solve_layout,
            layout,
            reference_map,
            static_transforms,
        )
    terminal_pressures, leak_flows = layout.unpack(solution)
    terminal_q, leak_p, leak_p_in = network.evaluate(terminal_pressures, leak_flows)
    final_transforms = (
        transform_evaluator(solved_reference_state) if transform_evaluator else {}
    )
    output_deck = write_solved_interface_bcs(deck, terminal_pressures, leak_flows)

    args.out_deck.parent.mkdir(parents=True, exist_ok=True)
    with args.out_deck.open("w", encoding="utf-8") as fp:
        json.dump(output_deck, fp, indent=2)
        fp.write("\n")
    verification_result: dict[str, Any] | None = None
    if bool(args.run_full_verification):
        if prepped_dir is None:
            parser.error("--run-full-verification requires --prepped-dir")
        verification_missing = [
            name for name, value in (
                ("--perm-region-path", args.perm_region_path),
                ("--perm-low", args.perm_low),
                ("--perm-high", args.perm_high),
                ("--anisotropy-ratio", args.anisotropy_ratio),
            ) if value is None
        ]
        if verification_missing:
            parser.error(
                "--run-full-verification requires " + ", ".join(verification_missing)
            )
        verification_dir = (
            args.verification_dir.expanduser().resolve()
            if args.verification_dir is not None
            else args.out_deck.expanduser().resolve().parent
        )
        verification_result = run_full_verification(
            run_dir=verification_dir,
            combined_path=args.out_deck,
            prepped_dir=prepped_dir,
            n_organoids=n_organoids,
            svzerodsolver=args.svzerodsolver,
            run_and_split=args.run_and_split,
            darcy_script=args.darcy_script,
            perm_region_path=args.perm_region_path,
            perm_low=float(args.perm_low),
            perm_high=float(args.perm_high),
            transition_width=float(args.perm_transition_width),
            anisotropy_ratio=float(args.anisotropy_ratio),
            concave_bc_mode=str(args.concave_bc_mode),
            offset_anchor=str(args.scaled_pressure_offset_anchor),
            profile_face_length=float(args.linear_profile_face_length),
            compartment_spacing=float(args.compartment_spacing),
            dy_step=float(args.dy_step),
            coords_inlet=list(args.coords_inlet),
            coords_outlet=list(args.coords_outlet),
            darcy_mpi_procs=int(args.darcy_mpi_procs),
            darcy_mpirun_cmd=str(args.darcy_mpirun_cmd),
            scaled_transform_mode="static" if mode == "static-scaled" else "dynamic",
            scaled_transforms=final_transforms,
        )
    report_path = args.report or args.out_deck.with_suffix(".direct-interface.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "direct-0d-darcy-pressure-interface-v1",
        "input_deck": str(deck_path),
        "output_deck": str(args.out_deck.expanduser().resolve()),
        "darcy_maps": {str(model.organoid): str(model.path) for model in maps},
        "interface_mode": mode,
        "solve_mode": solve_mode,
        "interface_dimension": int(layout.size),
        "solved_interface_dimension": int(solve_layout.size),
        "statically_eliminated_interface_dimension": int(layout.size - solve_layout.size),
        "terminal_pressure_count": len(layout.terminal_names),
        "concave_leak_flow_count": len(layout.leak_keys),
        "system_condition_number": condition,
        "spectral_radius_interface_jacobian": (
            float(max((abs(value) for value in np.linalg.eigvals(jacobian)), default=0.0))
            if jacobian is not None else None
        ),
        "residual_linf": float(np.max(np.abs(residual))),
        "normalized_residual_linf": normalized_linf,
        "terminal_pressures": terminal_pressures,
        "terminal_flows_0d_signed": terminal_q,
        "concave_leak_flows_darcy": {
            f"organoid{organoid}_{side}": value
            for (organoid, side), value in leak_flows.items()
        },
        "concave_pressures_0d": {
            f"organoid{organoid}_{side}": value
            for (organoid, side), value in leak_p.items()
        },
        "concave_pressures_0d_scaling_inlet": {
            f"organoid{organoid}_{side}": value
            for (organoid, side), value in leak_p_in.items()
        },
        "scaled_transforms": {str(key): value for key, value in final_transforms.items()},
        "dynamic_scaled_transforms": (
            {str(key): value for key, value in final_transforms.items()}
            if mode == "dynamic-scaled" else {}
        ),
        "static_scaled_transforms": (
            {str(key): value for key, value in final_transforms.items()}
            if mode == "static-scaled" else {}
        ),
        "nonlinear_history": nonlinear_history,
        "affine_offset_l2": (
            float(np.linalg.norm(affine_offset)) if affine_offset is not None else None
        ),
        "full_verification": verification_result,
    }
    with report_path.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
        fp.write("\n")
    print(json.dumps({
        "out_deck": str(args.out_deck),
        "report": str(report_path),
        "interface_mode": mode,
        "solve_mode": solve_mode,
        "interface_dimension": layout.size,
        "solved_interface_dimension": solve_layout.size,
        "condition_number": condition,
        "normalized_residual_linf": normalized_linf,
        "full_verification": verification_result,
    }, indent=2))


if __name__ == "__main__":
    main()
