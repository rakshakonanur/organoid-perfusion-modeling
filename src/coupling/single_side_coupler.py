#!/usr/bin/env python3
"""Couple a genuinely arterial-only or venous-only synthetic vasculature.

This file serves two roles:

* coupling driver: prune the inactive synthetic trees from ``combined.in``,
  install a leak-referenced single-side controller, and run the standard
  coupling loop;
* Darcy worker: when launched by the coupler with ``--bioreactor-domain``, run
  the mixed Darcy solver with the inactive terminal marker family sealed.

The two concave faces always remain active. ``arterial`` therefore means that
only arterial synthetic terminals exchange with the tissue; ``venous`` means
that only venous synthetic terminals exchange with the tissue.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
REPO_SRC = REPO_ROOT / "src"
SOLVES_DIR = REPO_SRC / "solves"
for path in (REPO_SRC, SOLVES_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


ACTIVE_TO_DECK_TOKEN = {"arterial": "inlet", "venous": "outlet"}
OPPOSITE_LEAK_SIDE = {"arterial": "venous", "venous": "arterial"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _extract_cli_value(argv: list[str], flag: str, default: str = "") -> str:
    for index, token in enumerate(argv):
        if token == flag and index + 1 < len(argv):
            return str(argv[index + 1])
        if token.startswith(f"{flag}="):
            return str(token.split("=", 1)[1])
    return str(default)


def _has_cli_option(argv: list[str], flag: str) -> bool:
    return any(token == flag or token.startswith(f"{flag}=") for token in argv)


def _replace_cli_value(argv: list[str], flag: str, value: str) -> list[str]:
    out: list[str] = []
    replaced = False
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == flag:
            if not replaced:
                out.extend([flag, str(value)])
                replaced = True
            index += 2
            continue
        if token.startswith(f"{flag}="):
            if not replaced:
                out.extend([flag, str(value)])
                replaced = True
            index += 1
            continue
        out.append(token)
        index += 1
    if not replaced:
        out.extend([flag, str(value)])
    return out


def prune_combined_for_single_side(deck: dict[str, Any], active_side: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove every inactive organoid tree while retaining its concave leak."""
    active_side = str(active_side).strip().lower()
    if active_side not in ACTIVE_TO_DECK_TOKEN:
        raise ValueError(f"Unsupported single side {active_side!r}")
    inactive_token = "outlet" if active_side == "arterial" else "inlet"
    vessel_pattern = re.compile(
        rf"^organoid[_ -]?\d+[_ -]?{inactive_token}[_ -]?branch(?:\s|_|$)",
        re.IGNORECASE,
    )
    junction_pattern = re.compile(
        rf"^organoid[_ -]?\d+[_ -]?{inactive_token}[_ -]?J\d+$",
        re.IGNORECASE,
    )
    bc_pattern = re.compile(
        rf"^organoid[_ -]?\d+[_ -]?{inactive_token}[_ -]?OUT\d+$",
        re.IGNORECASE,
    )

    result = copy.deepcopy(deck)
    removed_vessels = [
        vessel
        for vessel in result.get("vessels", [])
        if vessel_pattern.match(str(vessel.get("vessel_name", "")))
    ]
    removed_ids = {int(vessel["vessel_id"]) for vessel in removed_vessels}
    result["vessels"] = [
        vessel
        for vessel in result.get("vessels", [])
        if int(vessel.get("vessel_id", -1)) not in removed_ids
    ]

    new_junctions: list[dict[str, Any]] = []
    removed_junction_names: list[str] = []
    for junction in result.get("junctions", []):
        name = str(junction.get("junction_name", ""))
        if junction_pattern.match(name):
            removed_junction_names.append(name)
            continue
        updated = copy.deepcopy(junction)
        updated["inlet_vessels"] = [
            int(vessel_id)
            for vessel_id in junction.get("inlet_vessels", [])
            if int(vessel_id) not in removed_ids
        ]
        updated["outlet_vessels"] = [
            int(vessel_id)
            for vessel_id in junction.get("outlet_vessels", [])
            if int(vessel_id) not in removed_ids
        ]
        if not updated["inlet_vessels"] or not updated["outlet_vessels"]:
            raise ValueError(
                f"Pruning {active_side} produced an incomplete junction {name!r}: {updated}"
            )
        new_junctions.append(updated)
    result["junctions"] = new_junctions

    removed_bcs = [
        bc
        for bc in result.get("boundary_conditions", [])
        if bc_pattern.match(str(bc.get("bc_name", "")))
    ]
    removed_bc_names = {str(bc.get("bc_name", "")) for bc in removed_bcs}
    result["boundary_conditions"] = [
        bc
        for bc in result.get("boundary_conditions", [])
        if str(bc.get("bc_name", "")) not in removed_bc_names
    ]

    surviving_ids = {int(vessel["vessel_id"]) for vessel in result["vessels"]}
    for junction in result["junctions"]:
        referenced = {
            int(vessel_id)
            for vessel_id in junction.get("inlet_vessels", [])
            + junction.get("outlet_vessels", [])
        }
        missing = sorted(referenced - surviving_ids)
        if missing:
            raise ValueError(
                f"Junction {junction.get('junction_name')!r} references removed vessels {missing}"
            )

    surviving_bc_names = {
        str(bc.get("bc_name", "")) for bc in result["boundary_conditions"]
    }
    for vessel in result["vessels"]:
        for bc_name in (vessel.get("boundary_conditions") or {}).values():
            if str(bc_name) not in surviving_bc_names:
                raise ValueError(
                    f"Vessel {vessel.get('vessel_name')!r} references missing BC {bc_name!r}"
                )

    for organoid_idx in range(1, 5):
        required_leak = f"LEAK_{'VEN' if active_side == 'arterial' else 'ART'}_{organoid_idx}"
        if required_leak not in surviving_bc_names:
            raise ValueError(f"Required opposite concave leak BC {required_leak} was removed")

    result["description"] = (
        f"{result.get('description', '')} [{active_side}-only synthetic vasculature]"
    ).strip()
    report = {
        "active_synthetic_side": active_side,
        "inactive_deck_token": inactive_token,
        "removed_vessel_count": len(removed_ids),
        "removed_vessel_ids": sorted(removed_ids),
        "removed_junction_count": len(removed_junction_names),
        "removed_junction_names": removed_junction_names,
        "removed_boundary_condition_count": len(removed_bc_names),
        "removed_boundary_condition_names": sorted(removed_bc_names),
        "remaining_vessel_count": len(result["vessels"]),
        "remaining_junction_count": len(result["junctions"]),
        "remaining_boundary_condition_count": len(result["boundary_conditions"]),
    }
    return result, report


def _write_pruned_combined(source: Path, destination: Path, active_side: str) -> dict[str, Any]:
    with source.open("r") as fp:
        deck = json.load(fp)
    pruned, report = prune_combined_for_single_side(deck, active_side)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(pruned, indent=2) + "\n")
    report.update({"source": str(source), "output": str(destination)})
    destination.with_name("single_side_pruning_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def _leak_pressure(output_csv: Path, organoid_idx: int, active_side: str) -> float:
    from coupling import coupler

    p_art, p_ven = coupler.read_leak_pressures_from_output(output_csv, organoid_idx)
    value = p_ven if active_side == "arterial" else p_art
    if not _finite(value):
        leak_name = "leak_ven" if active_side == "arterial" else "leak_art"
        raise RuntimeError(
            f"Missing {leak_name}_{organoid_idx} pressure in {output_csv}"
        )
    return float(value)


def _leak_referenced_error(
    active_side: str,
    p_darcy: float,
    p_0d: float,
    p_opposite_leak: float,
    pressure_scale_floor: float,
) -> dict[str, float]:
    if active_side == "arterial":
        darcy_drop = float(p_darcy) - float(p_opposite_leak)
        zero_d_drop = float(p_0d) - float(p_opposite_leak)
    else:
        darcy_drop = float(p_opposite_leak) - float(p_darcy)
        zero_d_drop = float(p_opposite_leak) - float(p_0d)
    raw_error = float(darcy_drop - zero_d_drop)
    scale = max(
        abs(darcy_drop),
        abs(zero_d_drop),
        max(float(pressure_scale_floor), sys.float_info.min),
    )
    return {
        "darcy_drop": darcy_drop,
        "zero_d_drop": zero_d_drop,
        "raw_error": raw_error,
        "scale": scale,
        "normalized_error": float(raw_error / scale),
    }


def _install_single_side_controller(config: argparse.Namespace) -> None:
    from coupling import coupler
    from coupling.coupler_hybrid_algebraic import _solve_r_for_target_flows
    from coupling.steady_0d_pressure_control import SteadyResistiveZeroDModel

    active_side = str(config.single_side)
    original_apply = coupler.apply_organoid_resistance_from_json
    original_apply_interface = coupler.apply_organoid_interface_from_json
    original_validate_darcy_inputs = coupler.validate_darcy_geometry_inputs
    original_prepare_response_maps = coupler.prepare_automatic_darcy_response_maps
    original_apply_response_map = coupler.apply_darcy_response_map
    original_run_darcy_for_all = coupler.run_darcy_for_all
    original_read_terminal_flows = coupler.read_terminal_physical_flows_from_0d_csv
    original_read_terminal_parent_pressures = coupler.read_terminal_parent_pressures_from_0d_csv
    original_read_terminal_distal_pressures = coupler.read_terminal_distal_pressures_from_0d_csv
    pending: dict[str, Any] = {"key": None, "plans": []}

    def validate_single_side_run0_layout(
        seed_run0: Path,
        run0: Path,
        n_organoids: int,
    ) -> None:
        active_key = "branch_ids_inlet" if active_side == "arterial" else "branch_ids_outlet"
        inactive_key = "branch_ids_outlet" if active_side == "arterial" else "branch_ids_inlet"
        mismatches: list[str] = []
        for organoid_idx in range(1, int(n_organoids) + 1):
            seed_path = (
                Path(seed_run0)
                / f"organoid_{organoid_idx}"
                / "out_darcy"
                / "interface_bc.json"
            )
            run_path = (
                Path(run0)
                / f"organoid_{organoid_idx}"
                / "out_darcy"
                / "interface_bc.json"
            )
            if not seed_path.is_file() or not run_path.is_file():
                mismatches.append(f"organoid_{organoid_idx} missing interface layout")
                continue
            seed_iface = json.loads(seed_path.read_text())
            run_iface = json.loads(run_path.read_text())
            seed_active = [int(value) for value in seed_iface.get(active_key, [])]
            run_active = [int(value) for value in run_iface.get(active_key, [])]
            run_inactive = [int(value) for value in run_iface.get(inactive_key, [])]
            if seed_active != run_active:
                mismatches.append(f"organoid_{organoid_idx} active {active_side} layout")
            if run_inactive:
                mismatches.append(f"organoid_{organoid_idx} inactive layout is not empty")
        if mismatches:
            raise SystemExit(
                "Invalid single-side run_0 response-map layout: " + ", ".join(mismatches)
            )

    coupler.validate_fresh_run0_layout = validate_single_side_run0_layout

    def _empty_inactive_terminal_read(
        original,
        csv_path: Path,
        branch_ids: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | None:
        # A genuine single-side split intentionally has no output.csv for the
        # removed tree. The response-map evaluator nevertheless asks for both
        # sides; an empty branch layout should therefore produce an empty array
        # instead of being treated as a missing-data failure.
        if not branch_ids:
            return np.zeros(0, dtype=float)
        return original(csv_path, branch_ids, *args, **kwargs)

    def read_single_side_terminal_flows(
        csv_path: Path,
        branch_ids: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | None:
        return _empty_inactive_terminal_read(
            original_read_terminal_flows, csv_path, branch_ids, *args, **kwargs
        )

    def read_single_side_parent_pressures(
        csv_path: Path,
        branch_ids: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | None:
        return _empty_inactive_terminal_read(
            original_read_terminal_parent_pressures,
            csv_path,
            branch_ids,
            *args,
            **kwargs,
        )

    def read_single_side_distal_pressures(
        csv_path: Path,
        branch_ids: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray | None:
        return _empty_inactive_terminal_read(
            original_read_terminal_distal_pressures,
            csv_path,
            branch_ids,
            *args,
            **kwargs,
        )

    coupler.read_terminal_physical_flows_from_0d_csv = read_single_side_terminal_flows
    coupler.read_terminal_parent_pressures_from_0d_csv = read_single_side_parent_pressures
    coupler.read_terminal_distal_pressures_from_0d_csv = read_single_side_distal_pressures

    def prepare_single_side_response_maps(
        args: argparse.Namespace,
        seed_run0: Path,
        scaled_cfg: dict[str, Any] | None,
    ) -> None:
        if str(getattr(args, "darcy_response_map", "")).strip().lower() != "auto":
            return original_prepare_response_maps(args, seed_run0, scaled_cfg)

        # The standard auto-builder caches by geometry and permeability. Those
        # are identical for arterial-only and venous-only maps, so also check
        # our side tag and force a rebuild when the cached map is for the other
        # synthetic side.
        shared_dir = Path(seed_run0) / "response_maps"
        organoid_ids = (
            [int(scaled_cfg["reference_organoid"])]
            if scaled_cfg is not None
            else list(range(1, int(args.n_organoids) + 1))
        )
        wrong_side = False
        for organoid_idx in organoid_ids:
            metadata_path = shared_dir / f"exact_global_organoid{organoid_idx}.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text())
            except Exception:
                wrong_side = True
                break
            if str(metadata.get("single_side_active", "")) != active_side:
                wrong_side = True
                break
        previous_rebuild = bool(getattr(args, "rebuild_darcy_response_map", False))
        if wrong_side:
            args.rebuild_darcy_response_map = True
        try:
            original_prepare_response_maps(args, seed_run0, scaled_cfg)
        finally:
            args.rebuild_darcy_response_map = previous_rebuild

        # Keep the map used by this run in a side-specific location. This both
        # documents the physics and prevents a later opposite-side auto build
        # from changing the artifact referenced by this result tree.
        side_dir = Path(config.coupled_root) / "response_maps" / f"{active_side}_only"
        side_dir.mkdir(parents=True, exist_ok=True)
        for organoid_idx in organoid_ids:
            stem = f"exact_global_organoid{organoid_idx}"
            for suffix in (".npz", ".json"):
                source = shared_dir / f"{stem}{suffix}"
                if not source.is_file():
                    raise RuntimeError(f"Automatic response-map build did not create {source}")
                shutil.copy2(source, side_dir / source.name)
        args.darcy_response_map = str(side_dir)

    coupler.prepare_automatic_darcy_response_maps = prepare_single_side_response_maps

    def apply_single_side_response_map(*args: Any, **kwargs: Any) -> None:
        model_path_raw = kwargs.get("model_path")
        if model_path_raw is None and len(args) >= 3:
            model_path_raw = args[2]
        if model_path_raw is None:
            raise RuntimeError("Single-side response-map evaluation received no model path")
        model_path = Path(model_path_raw)
        metadata_path = model_path.with_suffix(".json")
        if not metadata_path.is_file():
            raise RuntimeError(f"Missing single-side response-map metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        mapped_side = str(metadata.get("single_side_active", "")).strip().lower()
        if mapped_side != active_side:
            raise RuntimeError(
                f"Response map {model_path} is tagged for {mapped_side or 'no side'}, "
                f"not {active_side}-only coupling"
            )
        original_apply_response_map(*args, **kwargs)

    coupler.apply_darcy_response_map = apply_single_side_response_map

    def run_single_side_darcy_for_all(
        run_dir: Path,
        args: argparse.Namespace,
        *positional: Any,
        **kwargs: Any,
    ) -> None:
        env_overrides = {
            "COUPLING_DARCY_OUTPUT_MODE": kwargs.get("output_mode_override"),
            "COUPLING_DARCY_DIAGNOSTICS_MODE": kwargs.get("diagnostics_mode_override"),
        }
        previous_env = {name: os.environ.get(name) for name in env_overrides}
        for name, value in env_overrides.items():
            if value is not None:
                os.environ[name] = str(value)
        try:
            try:
                run_index = int(Path(run_dir).name.rsplit("_", 1)[1])
            except (IndexError, ValueError):
                run_index = -1
            requested_map = str(getattr(args, "darcy_response_map", ""))
            if run_index != 0 or not requested_map.strip():
                return original_run_darcy_for_all(
                    run_dir, args, *positional, **kwargs
                )

            # Response-map evaluation requires the preceding interface layout.
            # Run zero has no predecessor, so establish its active-only branch
            # layout with one PDE solve even when later iterations use a map.
            args.darcy_response_map = ""
            try:
                kwargs["force_pde"] = True
                original_run_darcy_for_all(run_dir, args, *positional, **kwargs)
            finally:
                args.darcy_response_map = requested_map
        finally:
            for name, previous in previous_env.items():
                if previous is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = previous

    coupler.run_darcy_for_all = run_single_side_darcy_for_all

    def validate_active_darcy_inputs(
        run_dir: Path,
        n_organoids: int,
        no_synthetic_vasculature: bool = False,
    ) -> None:
        if no_synthetic_vasculature:
            return original_validate_darcy_inputs(
                run_dir, n_organoids, no_synthetic_vasculature=True
            )
        token = ACTIVE_TO_DECK_TOKEN[active_side]
        required = [
            "bioreactor.xdmf",
            "mesh_tags.xdmf",
            f"tagged_branches_{token}.bp",
            f"pressure_checkpoint_{token}.bp",
            f"flow_checkpoint_{token}.bp",
            f"area_checkpoint_{token}.bp",
        ]
        missing: list[str] = []
        for organoid_idx in range(1, int(n_organoids) + 1):
            geometry = Path(run_dir) / f"organoid_{organoid_idx}" / "geometry"
            absent = [name for name in required if not (geometry / name).exists()]
            if absent:
                missing.append(
                    f"organoid_{organoid_idx}: {', '.join(absent)} in {geometry}"
                )
        if missing:
            raise SystemExit(
                f"Missing required {active_side}-only Darcy inputs: "
                + " | ".join(missing)
            )

    coupler.validate_darcy_geometry_inputs = validate_active_darcy_inputs

    def apply_single_side_interface(*args: Any, **kwargs: Any) -> None:
        kwargs["allow_missing_terminal_bcs"] = True
        original_apply_interface(*args, **kwargs)

    coupler.apply_organoid_interface_from_json = apply_single_side_interface

    def wrapped_apply(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        organoid_idx = int(kwargs.get("organoid_idx", args[1] if len(args) > 1 else -1))
        deck = kwargs.get("deck", args[0] if args else None)
        iface = kwargs.get("iface", args[2] if len(args) > 2 else {})
        current_run_idx = kwargs.get("current_run_idx")
        kwargs["allow_missing_terminal_bcs"] = True
        kwargs["active_terminal_side"] = active_side
        rows = original_apply(*args, **kwargs)

        if current_run_idx is None or int(current_run_idx) < int(config.controller_start_run):
            return rows
        state_key = (int(current_run_idx), id(deck))
        if pending["key"] != state_key:
            pending["key"] = state_key
            pending["plans"] = []

        bc_by_name = {
            str(bc.get("bc_name", "")): bc
            for bc in deck.get("boundary_conditions", [])
        }
        previous_output = (
            Path(config.coupled_root)
            / f"run_{int(current_run_idx) - 1}"
            / "output.csv"
        )
        p_reference = _leak_pressure(previous_output, organoid_idx, active_side)
        q_floor = max(float(config.controller_q_floor), sys.float_info.min)
        max_step = max(float(config.controller_max_log_step), 0.0)

        for row in rows:
            if str(row.get("side", "")).strip().lower() != active_side:
                continue
            bc_name = str(row.get("bc_name", ""))
            bc = bc_by_name.get(bc_name)
            if bc is None:
                continue
            q_drive = float(row.get("Q_for_R", float("nan")))
            branch_id = int(row.get("branch_id", -1))
            if not _finite(q_drive) or abs(q_drive) <= float(config.controller_bootstrap_q_threshold):
                from coupling.coupler_hybrid_algebraic import _bootstrap_target_flow_from_iface

                q_boot = _bootstrap_target_flow_from_iface(iface, active_side, branch_id)
                if _finite(q_boot):
                    q_drive = float(q_boot)
            p_darcy = float(row.get("P_interface", float("nan")))
            p_0d = float(row.get("P_0D", float("nan")))
            if not all(_finite(value) for value in (q_drive, p_darcy, p_0d)):
                continue
            error = _leak_referenced_error(
                active_side,
                p_darcy,
                p_0d,
                p_reference,
                float(config.controller_pressure_scale_floor),
            )
            delta_log = max(
                -max_step,
                min(
                    max_step,
                    -float(config.controller_log_gain)
                    * float(error["normalized_error"]),
                ),
            )
            q_target = max(abs(q_drive), q_floor) * (10.0**delta_log)
            pd_seed = float(row.get("Pd_next", float("nan")))
            if not _finite(pd_seed):
                pd_seed = float(row.get("Pd_target", p_0d))
            plan = {
                "row": row,
                "bc": bc,
                "bc_name": bc_name,
                "organoid_idx": organoid_idx,
                "side_label": active_side,
                "branch_id": branch_id,
                "q_drive": float(q_drive),
                "q_target": float(q_target),
                "q_floor": q_floor,
                "r_trial": max(float(row.get("R_next", 1.0)), 1.0),
                "pd_seed": float(pd_seed),
                "delta_log": float(delta_log),
                "error": error,
                "reference_pressure": float(p_reference),
            }
            pending["plans"].append(plan)

        if organoid_idx != int(config.n_organoids):
            return rows

        plans = list(pending["plans"])
        pending["plans"] = []
        if not plans:
            raise RuntimeError(
                f"No {active_side} terminal controller plans were built for run {current_run_idx}"
            )
        model = SteadyResistiveZeroDModel.from_data(deck)
        (
            r_by_name,
            pd_by_name,
            pterm_by_name,
            solve_mode,
            residual,
            iterations,
        ) = _solve_r_for_target_flows(
            plans=plans,
            full_model=model,
            q_floor=float(config.controller_q_floor),
            r_floor=float(config.controller_r_floor),
        )
        if len(r_by_name) != len(plans):
            missing = sorted(
                str(plan["bc_name"])
                for plan in plans
                if str(plan["bc_name"]) not in r_by_name
            )
            raise RuntimeError(f"Algebraic single-side solve missed terminal BCs: {missing}")

        for plan in plans:
            row = plan["row"]
            bc_name = str(plan["bc_name"])
            r_next = float(r_by_name[bc_name])
            pd_next = float(pd_by_name[bc_name])
            coupler.set_resistance_bc(plan["bc"], r_next, distal_pressure=pd_next)
            error = plan["error"]
            q_target = float(plan["q_target"])
            physical_q_target = q_target if active_side == "arterial" else -q_target
            row.update(
                {
                    "bc_type": "RESISTANCE",
                    "Pd_target": pd_next,
                    "Pd_target_reason": "single_side_leak_referenced_targetQ",
                    "Pd_next": pd_next,
                    "R_raw": r_next,
                    "R_next": r_next,
                    "hybrid_controller_mode": "single_side_opposite_concave_leak",
                    "hybrid_target_flow": q_target,
                    "hybrid_target_flow_raw": physical_q_target,
                    "Q_target_commanded": physical_q_target,
                    "hybrid_q_drive": float(plan["q_drive"]),
                    "hybrid_action": (
                        "leak_referenced_back_off"
                        if float(plan["delta_log"]) < -1.0e-12
                        else "leak_referenced_open_up"
                        if float(plan["delta_log"]) > 1.0e-12
                        else "leak_referenced_hold"
                    ),
                    "hybrid_delta_log10_q": float(plan["delta_log"]),
                    "hybrid_relative_pressure_error": float(error["normalized_error"]),
                    "hybrid_adaptive_log10_cap": max_step,
                    "hybrid_r_trial": float(plan["r_trial"]),
                    "hybrid_branch_pressure_error_raw": float(error["raw_error"]),
                    "hybrid_branch_pressure_error_normalized": float(error["normalized_error"]),
                    "hybrid_branch_pressure_error_scale": float(error["scale"]),
                    "hybrid_algebraic_mode": solve_mode,
                    "hybrid_algebraic_residual_linf": float(residual),
                    "hybrid_algebraic_iterations": int(iterations),
                    "P_predicted_terminal_from_algebraic": float(
                        pterm_by_name.get(bc_name, float("nan"))
                    ),
                    "single_side_active": active_side,
                    "single_side_opposite_concave_reference": OPPOSITE_LEAK_SIDE[active_side],
                    "single_side_reference_pressure": float(plan["reference_pressure"]),
                    "single_side_darcy_pressure_drop": float(error["darcy_drop"]),
                    "single_side_0d_pressure_drop": float(error["zero_d_drop"]),
                }
            )
        return rows

    coupler.apply_organoid_resistance_from_json = wrapped_apply

    original_summary_writer = coupler.write_terminal_resistance_summary

    def write_summary_with_single_side_diagnostics(
        run_dir: Path,
        rows: list[dict[str, Any]],
        write_cumulative: bool = True,
    ) -> None:
        original_summary_writer(
            run_dir,
            rows,
            write_cumulative=write_cumulative,
        )
        selected = [
            row for row in rows if str(row.get("side", "")) == active_side
        ]
        if not selected:
            return
        fields = [
            "organoid",
            "side",
            "branch_id",
            "bc_name",
            "P_interface",
            "P_0D",
            "single_side_active",
            "single_side_opposite_concave_reference",
            "single_side_reference_pressure",
            "single_side_darcy_pressure_drop",
            "single_side_0d_pressure_drop",
            "hybrid_branch_pressure_error_raw",
            "hybrid_branch_pressure_error_normalized",
            "hybrid_branch_pressure_error_scale",
            "Q_target_commanded",
            "hybrid_delta_log10_q",
            "hybrid_action",
            "R_next",
            "Pd_next",
        ]
        with (Path(run_dir) / "single_side_controller_summary.csv").open(
            "w", newline=""
        ) as fp:
            writer = csv.DictWriter(fp, fieldnames=fields)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field, "") for field in fields} for row in selected
            )

    coupler.write_terminal_resistance_summary = write_summary_with_single_side_diagnostics

    def single_side_convergence(
        *,
        run_dir: Path,
        previous_run_dir: Path,
        args: argparse.Namespace,
        current_deck: dict | None = None,
        label: str = "coupling",
    ) -> bool:
        del previous_run_dir, current_deck
        rows = coupler.read_post_darcy_terminal_convergence(
            Path(run_dir) / "post_darcy_terminal_convergence.csv"
        )
        errors: list[float] = []
        for row in rows:
            if str(row.get("side", "")).strip().lower() != active_side:
                continue
            organoid_idx = int(row.get("organoid", 0))
            p_reference = _leak_pressure(
                Path(run_dir) / "output.csv", organoid_idx, active_side
            )
            error = _leak_referenced_error(
                active_side,
                float(row["P_Darcy"]),
                float(row["P_0D_target"]),
                p_reference,
                float(config.controller_pressure_scale_floor),
            )
            errors.append(abs(float(error["normalized_error"])))
        maximum = max(errors, default=float("inf"))
        converged = bool(maximum <= float(args.tol_p))
        print(
            f"\n[check] {label}: {active_side}-only max leak-referenced "
            f"terminal error={maximum:.3e}, converged={converged}",
            flush=True,
        )
        return converged

    coupler.check_coupling_convergence = single_side_convergence


def _empty_terminal_side(data: tuple[Any, ...], active_side: str) -> tuple[Any, ...]:
    inlet_marks, outlet_marks, inlet_coords, outlet_coords, idx_in, idx_out = data
    if active_side == "arterial":
        return (
            inlet_marks,
            np.zeros(0, dtype=np.int32),
            inlet_coords,
            np.zeros((0, 3), dtype=float),
            idx_in,
            np.zeros(0, dtype=int),
        )
    return (
        np.zeros(0, dtype=np.int32),
        outlet_marks,
        np.zeros((0, 3), dtype=float),
        outlet_coords,
        np.zeros(0, dtype=int),
        idx_out,
    )


def _darcy_worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-side mixed Darcy worker")
    parser.add_argument("--bioreactor-domain", required=True)
    parser.add_argument("--facet-file", required=True)
    for name in (
        "mesh-inlet-file", "mesh-outlet-file", "pres-inlet-file", "pres-outlet-file",
        "flow-inlet-file", "flow-outlet-file", "area-inlet-file", "area-outlet-file",
        "branching-in-file", "branching-out-file", "inlet-output-csv", "outlet-output-csv",
    ):
        parser.add_argument(f"--{name}", default="")
    parser.add_argument("--coords-inlet", nargs=3, type=float, required=True)
    parser.add_argument("--coords-outlet", nargs=3, type=float, required=True)
    parser.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    parser.add_argument("--concave-pressure-profile", choices=["constant", "linear"], default="linear")
    parser.add_argument("--concave-pressure-profile-axis", choices=["x", "y", "z"], default="y")
    parser.add_argument("--arterial-pressure-profile-low", type=float, default=None)
    parser.add_argument("--arterial-pressure-profile-high", type=float, default=None)
    parser.add_argument("--venous-pressure-profile-low", type=float, default=None)
    parser.add_argument("--venous-pressure-profile-high", type=float, default=None)
    parser.add_argument("--lp-arterial", type=float, default=0.0)
    parser.add_argument("--lp-venous", type=float, default=0.0)
    parser.add_argument("--skip-1d", action="store_true")
    parser.add_argument("--fallback-inlet-pressure", type=float, default=0.0)
    parser.add_argument("--fallback-outlet-pressure", type=float, default=0.0)
    parser.add_argument("--concave-inlet-pressure", type=float, default=None)
    parser.add_argument("--concave-outlet-pressure", type=float, default=None)
    parser.add_argument("--checkpoint-time-index", type=int, default=-1)
    parser.add_argument("--inlet-flux-correction", action="store_true")
    parser.add_argument("--inlet-flux-corr-max-iter", type=int, default=5)
    parser.add_argument("--inlet-flux-corr-relax", type=float, default=0.5)
    parser.add_argument("--inlet-flux-corr-tol", type=float, default=0.05)
    parser.add_argument("--perm-region-path", default="")
    parser.add_argument("--perm-low", type=float, default=1.0e-8)
    parser.add_argument("--perm-high", type=float, default=1.0e-6)
    parser.add_argument("--perm-transition-width", type=float, default=0.01)
    parser.add_argument("--stl-anisotropy-factor", type=float, default=5.0)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--output-mode", choices=["full", "fields", "minimal"], default="full")
    parser.add_argument("--diagnostics-mode", choices=["minimal", "full"], default="minimal")
    parser.add_argument("--response-map-output", default="")
    return parser


def _run_darcy_worker(argv: list[str]) -> None:
    import darcy as darcy_cg
    import darcy_mixed

    args = _darcy_worker_parser().parse_args(argv)
    active_side = os.environ.get("COUPLING_SINGLE_SIDE", "").strip().lower()
    if active_side not in ACTIVE_TO_DECK_TOKEN:
        raise SystemExit("COUPLING_SINGLE_SIDE must be arterial or venous")
    # The inherited loader reads both checkpoint families during construction.
    # Point the disabled family at the active data so construction remains
    # topology-consistent; _get_terminal_marker_data removes that duplicate
    # family before any Darcy form, constraint, or output is built.
    if active_side == "arterial":
        args.mesh_outlet_file = args.mesh_inlet_file
        args.pres_outlet_file = args.pres_inlet_file
        args.flow_outlet_file = args.flow_inlet_file
        args.area_outlet_file = args.area_inlet_file
        args.branching_out_file = args.branching_in_file
        args.outlet_output_csv = args.inlet_output_csv
    else:
        args.mesh_inlet_file = args.mesh_outlet_file
        args.pres_inlet_file = args.pres_outlet_file
        args.flow_inlet_file = args.flow_outlet_file
        args.area_inlet_file = args.area_outlet_file
        args.branching_in_file = args.branching_out_file
        args.inlet_output_csv = args.outlet_output_csv
    if not args.perm_region_path:
        perm_region_root = os.environ.get("COUPLING_PERM_REGION_ROOT", "").strip()
        match = re.search(r"organoid_(\d+)$", str(args.out_dir).rstrip("/"))
        if perm_region_root and match:
            from coupling.coupler import resolve_perm_region_for_organoid

            args.perm_region_path = resolve_perm_region_for_organoid(
                perm_region_root, int(match.group(1))
            )
    args.perm_low = float(os.environ.get("COUPLING_PERM_LOW", args.perm_low))
    args.perm_high = float(os.environ.get("COUPLING_PERM_HIGH", args.perm_high))
    args.perm_transition_width = float(
        os.environ.get("COUPLING_PERM_TRANSITION_WIDTH", args.perm_transition_width)
    )
    args.stl_anisotropy_factor = float(
        os.environ.get("COUPLING_STL_ANISOTROPY_FACTOR", args.stl_anisotropy_factor)
    )
    args.output_mode = os.environ.get("COUPLING_DARCY_OUTPUT_MODE", args.output_mode)
    args.diagnostics_mode = os.environ.get(
        "COUPLING_DARCY_DIAGNOSTICS_MODE", args.diagnostics_mode
    )

    class SingleSidePerfusionSolver(darcy_mixed.PerfusionSolver):
        def _build_reordering_indices(self, coords_terminals, branch_coords):
            if coords_terminals is None or len(coords_terminals) == 0:
                return np.zeros(0, dtype=int)
            return super()._build_reordering_indices(coords_terminals, branch_coords)

        def _get_terminal_marker_data(self):
            filtered = _empty_terminal_side(
                super()._get_terminal_marker_data(), active_side
            )
            if active_side == "arterial":
                self.branch_ids_out = np.zeros(0, dtype=int)
                self.branch_coords_out = np.zeros((0, 3), dtype=float)
            else:
                self.branch_ids_in = np.zeros(0, dtype=int)
                self.branch_coords_in = np.zeros((0, 3), dtype=float)
            (
                self.inlet_terminal_markers,
                self.outlet_terminal_markers,
                self.inlet_terminal_centroids,
                self.outlet_terminal_centroids,
                self.inlet_marker_to_1d_idx,
                self.outlet_marker_to_1d_idx,
            ) = filtered
            return filtered

    if args.out_dir:
        darcy_mixed.current_dir = Path(args.out_dir).expanduser().resolve()
        darcy_mixed.current_dir.mkdir(parents=True, exist_ok=True)
        darcy_cg.current_dir = darcy_mixed.current_dir

    solver = SingleSidePerfusionSolver(
        args.bioreactor_domain,
        args.facet_file,
        args.mesh_inlet_file,
        args.mesh_outlet_file,
        args.pres_inlet_file,
        args.pres_outlet_file,
        args.flow_inlet_file,
        args.flow_outlet_file,
        np.asarray(args.coords_inlet, dtype=float),
        np.asarray(args.coords_outlet, dtype=float),
        area_inlet_file=args.area_inlet_file or None,
        area_outlet_file=args.area_outlet_file or None,
        concave_bc_mode=args.concave_bc_mode,
        concave_pressure_profile=args.concave_pressure_profile,
        concave_pressure_profile_axis=args.concave_pressure_profile_axis,
        arterial_pressure_profile_low=args.arterial_pressure_profile_low,
        arterial_pressure_profile_high=args.arterial_pressure_profile_high,
        venous_pressure_profile_low=args.venous_pressure_profile_low,
        venous_pressure_profile_high=args.venous_pressure_profile_high,
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
        stl_anisotropy_factor=args.stl_anisotropy_factor,
        branching_in_file=args.branching_in_file or None,
        branching_out_file=args.branching_out_file or None,
        inlet_output_csv=args.inlet_output_csv or None,
        outlet_output_csv=args.outlet_output_csv or None,
        concave_inlet_pressure=args.concave_inlet_pressure,
        concave_outlet_pressure=args.concave_outlet_pressure,
        output_mode=args.output_mode,
        diagnostics_mode=args.diagnostics_mode,
        response_map_output=args.response_map_output,
    )
    solver.setup()
    if args.response_map_output and solver.mesh.comm.rank == 0:
        metadata_path = Path(args.response_map_output).expanduser().resolve().with_suffix(".json")
        if not metadata_path.is_file():
            raise RuntimeError(f"Darcy response-map metadata was not written: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        metadata["single_side_active"] = active_side
        metadata["inactive_synthetic_side"] = OPPOSITE_LEAK_SIDE[active_side]
        if active_side == "arterial":
            metadata["branch_ids_outlet"] = []
        else:
            metadata["branch_ids_inlet"] = []
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def _coupling_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run arterial-only or venous-only synthetic-vasculature coupling."
    )
    parser.add_argument("--single-side", choices=["arterial", "venous"], required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--prepared-combined",
        default="",
        help="Output path for the pruned deck; defaults inside --coupled-root.",
    )
    parser.add_argument("--controller-start-run", type=int, default=2)
    parser.add_argument("--controller-log-gain", type=float, default=1.0)
    parser.add_argument("--controller-max-log-step", type=float, default=0.45)
    parser.add_argument("--controller-pressure-scale-floor", type=float, default=1.0)
    parser.add_argument("--controller-q-floor", type=float, default=1.0e-20)
    parser.add_argument("--controller-r-floor", type=float, default=1.0)
    parser.add_argument("--controller-bootstrap-q-threshold", type=float, default=1.0e-12)
    return parser


def _run_coupling(argv: list[str]) -> None:
    from coupling import coupler
    from coupling.coupler_hybrid_algebraic import _parse_wrapper_args

    config, coupler_argv = _coupling_parser().parse_known_args(argv)
    # Existing case launchers may still contain the old hybrid controller's
    # options. Consume them here so this file can directly replace that driver,
    # and preserve the subset that has an exact single-side equivalent.
    legacy_hybrid, coupler_argv = _parse_wrapper_args(coupler_argv)
    legacy_mapping = {
        "controller_start_run": ("--controller-start-run", "hybrid_start_run"),
        "controller_log_gain": ("--controller-log-gain", "hybrid_log_gain"),
        "controller_max_log_step": ("--controller-max-log-step", "hybrid_max_log_step"),
        "controller_pressure_scale_floor": (
            "--controller-pressure-scale-floor",
            "hybrid_pressure_scale_floor",
        ),
        "controller_q_floor": ("--controller-q-floor", "hybrid_q_floor"),
        "controller_r_floor": ("--controller-r-floor", "hybrid_r_floor"),
        "controller_bootstrap_q_threshold": (
            "--controller-bootstrap-q-threshold",
            "hybrid_bootstrap_q_threshold",
        ),
    }
    for destination, (new_flag, legacy_attribute) in legacy_mapping.items():
        legacy_flag = "--" + legacy_attribute.replace("_", "-")
        if not _has_cli_option(argv, new_flag) and _has_cli_option(argv, legacy_flag):
            setattr(config, destination, getattr(legacy_hybrid, legacy_attribute))

    template = Path(
        _extract_cli_value(coupler_argv, "--template-combined")
    ).expanduser().resolve()
    if not template.is_file():
        raise SystemExit("--template-combined must name an existing combined.in")
    coupled_root_raw = _extract_cli_value(coupler_argv, "--coupled-root")
    if not coupled_root_raw:
        raise SystemExit("--coupled-root is required")
    coupled_root = Path(coupled_root_raw).expanduser().resolve()
    prepared = (
        Path(config.prepared_combined).expanduser().resolve()
        if config.prepared_combined
        else coupled_root / "single_side_input" / "combined.in"
    )
    report = _write_pruned_combined(template, prepared, config.single_side)
    print(
        f"[single-side] Wrote {config.single_side}-only deck to {prepared}; "
        f"removed {report['removed_vessel_count']} vessels and "
        f"{report['removed_boundary_condition_count']} terminal BCs.",
        flush=True,
    )
    if config.prepare_only:
        return

    config.coupled_root = str(coupled_root)
    config.n_organoids = int(_extract_cli_value(coupler_argv, "--n-organoids", "4"))
    _install_single_side_controller(config)
    os.environ["COUPLING_SINGLE_SIDE"] = str(config.single_side)
    os.environ["COUPLING_PERM_REGION_ROOT"] = _extract_cli_value(
        coupler_argv, "--perm-region-root", ""
    )
    os.environ["COUPLING_PERM_LOW"] = _extract_cli_value(
        coupler_argv, "--perm-low", "1e-8"
    )
    os.environ["COUPLING_PERM_HIGH"] = _extract_cli_value(
        coupler_argv, "--perm-high", "1e-6"
    )
    os.environ["COUPLING_PERM_TRANSITION_WIDTH"] = _extract_cli_value(
        coupler_argv, "--perm-transition-width", "0.01"
    )
    os.environ["COUPLING_STL_ANISOTROPY_FACTOR"] = _extract_cli_value(
        coupler_argv, "--darcy-stl-anisotropy-factor", "5.0"
    )
    os.environ["COUPLING_DARCY_OUTPUT_MODE"] = _extract_cli_value(
        coupler_argv, "--darcy-output-mode", "full"
    )
    os.environ["COUPLING_DARCY_DIAGNOSTICS_MODE"] = _extract_cli_value(
        coupler_argv, "--darcy-diagnostics-mode", "minimal"
    )
    coupler_argv = _replace_cli_value(
        coupler_argv, "--template-combined", str(prepared)
    )
    coupler_argv = _replace_cli_value(
        coupler_argv, "--darcy-script", str(THIS_FILE)
    )
    sys.argv = [str(THIS_FILE), *coupler_argv]
    coupler.main()


def main() -> None:
    if "--bioreactor-domain" in sys.argv[1:]:
        _run_darcy_worker(sys.argv[1:])
    else:
        _run_coupling(sys.argv[1:])


if __name__ == "__main__":
    main()
