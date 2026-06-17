#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from coupling import coupler
from coupling.steady_0d_pressure_control import SteadyResistiveZeroDModel


def _parse_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    ap = argparse.ArgumentParser(
        description=(
            "Run the standard coupler, keep its resistance update behavior, "
            "but replace selected terminal Q_target decisions with a simpler "
            "log-space pressure-mismatch controller and realize those targets "
            "with the full algebraic resistive 0D map."
        ),
        add_help=True,
    )
    ap.add_argument(
        "--hybrid-sides",
        choices=["arterial", "venous", "both"],
        default="both",
        help="Which terminal sides should use the hybrid algebraic override.",
    )
    ap.add_argument(
        "--hybrid-organoids",
        default="all",
        help="Comma-separated organoid indices to modify, or 'all'.",
    )
    ap.add_argument(
        "--hybrid-start-run",
        type=int,
        default=2,
        help="First coupling run index where the hybrid override is applied.",
    )
    ap.add_argument(
        "--hybrid-q-floor",
        type=float,
        default=1.0e-20,
        help="Small floor used when constructing target flow magnitudes.",
    )
    ap.add_argument(
        "--hybrid-r-floor",
        type=float,
        default=1.0,
        help="Minimum resistance passed into the algebraic realization step.",
    )
    ap.add_argument(
        "--hybrid-bootstrap-q-threshold",
        type=float,
        default=1.0e-12,
        help=(
            "When |Q_for_R| is below this threshold, bootstrap from the Darcy "
            "flow target stored in interface_bc.json."
        ),
    )
    ap.add_argument(
        "--hybrid-log-gain",
        type=float,
        default=1.0,
        help="Gain applied to the relative pressure mismatch before log-step clipping.",
    )
    ap.add_argument(
        "--hybrid-max-log-step",
        type=float,
        default=0.35,
        help=(
            "Maximum absolute change in log10(|Q_target|) per iteration. "
            "0.35 is about a 2.2x multiplicative step."
        ),
    )
    ap.add_argument(
        "--hybrid-moderate-log-step",
        type=float,
        default=0.75,
        help="Adaptive cap for moderate relative pressure mismatch.",
    )
    ap.add_argument(
        "--hybrid-large-log-step",
        type=float,
        default=1.25,
        help="Adaptive cap for large relative pressure mismatch.",
    )
    ap.add_argument(
        "--hybrid-extreme-log-step",
        type=float,
        default=2.0,
        help="Adaptive cap for extreme relative pressure mismatch.",
    )
    ap.add_argument(
        "--hybrid-venous-max-log-step",
        type=float,
        default=0.10,
        help="Base log10 step cap for the stiffer venous side.",
    )
    ap.add_argument(
        "--hybrid-venous-moderate-log-step",
        type=float,
        default=0.20,
        help="Moderate log10 step cap for the venous side.",
    )
    ap.add_argument(
        "--hybrid-venous-large-log-step",
        type=float,
        default=0.30,
        help="Large log10 step cap for the venous side.",
    )
    ap.add_argument(
        "--hybrid-venous-extreme-log-step",
        type=float,
        default=0.45,
        help="Extreme log10 step cap for the venous side.",
    )
    ap.add_argument(
        "--hybrid-pressure-scale-floor",
        type=float,
        default=1.0,
        help="Floor used when normalizing the pressure mismatch.",
    )
    ap.add_argument(
        "--hybrid-arterial-negative-pressure-threshold",
        type=float,
        default=0.0,
        help=(
            "If an arterial Darcy terminal pressure drops below this value, "
            "treat it as a pathological state and force a back-off step."
        ),
    )
    ap.add_argument(
        "--hybrid-resistance-source",
        choices=["next", "previous"],
        default="next",
        help=(
            "Which resistance from the original coupler rows to hold fixed in "
            "the algebraic Pd solve."
        ),
    )
    ap.add_argument(
        "--hybrid-venous-dominance-ratio",
        type=float,
        default=5.0,
        help=(
            "If the median venous relative pressure error exceeds this multiple "
            "of the median arterial error, enter venous-dominant damping mode."
        ),
    )
    ap.add_argument(
        "--hybrid-arterial-damping-when-venous-dominant",
        type=float,
        default=0.25,
        help=(
            "Multiplier applied to arterial log10 flow steps when the venous "
            "side is dominant. Set to 0 to freeze arterial updates."
        ),
    )
    ap.add_argument(
        "--hybrid-venous-predicted-pressure-change-cap",
        type=float,
        default=0.25,
        help=(
            "Maximum allowed relative change in the algebraically predicted "
            "venous terminal pressure before shrinking the venous Q update."
        ),
    )
    ap.add_argument(
        "--hybrid-venous-predicted-pressure-shrink-min",
        type=float,
        default=0.25,
        help="Minimum multiplicative shrink factor used by the venous pre-commit guard.",
    )
    ap.add_argument(
        "--hybrid-venous-predicted-pressure-guard-iters",
        type=int,
        default=4,
        help="Maximum number of shrink attempts for the venous pre-commit guard.",
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Estimate d(pressure error)/d(log10|Q|) from recent runs and use "
            "that branchwise sensitivity to choose the next Q step."
        ),
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-history-window",
        type=int,
        default=6,
        help="Number of recent runs used to estimate branchwise dE/dlog10|Q|.",
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-min-logq-change",
        type=float,
        default=0.005,
        help="Minimum historical |delta log10|Q|| required for a sensitivity sample.",
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-min-error-change",
        type=float,
        default=0.01,
        help="Minimum historical |delta E| required for a sensitivity sample.",
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-sign-consistency",
        type=float,
        default=0.0,
        help=(
            "Minimum sign consistency required before using the branchwise "
            "dE/dlog10|Q| estimate. Default 0.0 keeps sensitivity active "
            "whenever a finite branchwise history estimate exists."
        ),
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-alpha",
        type=float,
        default=0.5,
        help=(
            "Relaxation applied to the secant-style sensitivity move: "
            "delta_log10|Q| = -alpha * error / (dE/dlog10|Q|)."
        ),
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-alpha-arterial",
        type=float,
        default=None,
        help=(
            "Optional arterial override for the secant-style sensitivity "
            "relaxation. If omitted, --hybrid-q-sensitivity-alpha is used."
        ),
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-alpha-venous",
        type=float,
        default=None,
        help=(
            "Optional venous override for the secant-style sensitivity "
            "relaxation. If omitted, --hybrid-q-sensitivity-alpha is used."
        ),
    )
    ap.add_argument(
        "--hybrid-q-sensitivity-ema-alpha",
        type=float,
        default=0.5,
        help=(
            "EMA weight used to smooth the per-branch normalized sensitivity "
            "magnitude |dE/dlog10|Q|| over recent iterations."
        ),
    )
    ap.add_argument(
        "--hybrid-alternate-sides",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Alternate which side is allowed to move strongly each run. "
            "The inactive side is damped by --hybrid-inactive-side-alpha-scale."
        ),
    )
    ap.add_argument(
        "--hybrid-alternate-start-side",
        choices=["arterial", "venous"],
        default="venous",
        help="Which side is active on the first alternating hybrid run.",
    )
    ap.add_argument(
        "--hybrid-inactive-side-alpha-scale",
        type=float,
        default=0.1,
        help="Multiplier applied to the inactive side's proposed log-space move.",
    )
    ap.add_argument(
        "--hybrid-venous-low-flow-threshold",
        type=float,
        default=5.0e-6,
        help="Q_for_R magnitude below which venous branches receive tighter trust-region caps.",
    )
    ap.add_argument(
        "--hybrid-venous-very-low-flow-threshold",
        type=float,
        default=5.0e-7,
        help="Q_for_R magnitude below which venous branches receive the strongest trust-region caps.",
    )
    ap.add_argument(
        "--hybrid-venous-low-flow-cap",
        type=float,
        default=0.01,
        help="Maximum |delta log10 Q| for low-flow venous branches.",
    )
    ap.add_argument(
        "--hybrid-venous-very-low-flow-cap",
        type=float,
        default=0.0025,
        help="Maximum |delta log10 Q| for very-low-flow venous branches.",
    )
    ap.add_argument(
        "--hybrid-venous-small-error-cap",
        type=float,
        default=0.02,
        help="Maximum |delta log10 Q| for venous branches once relative pressure error is already small.",
    )
    ap.add_argument(
        "--hybrid-alternate-venous-cap-scale",
        type=float,
        default=1.0,
        help="Extra scale applied to low-flow venous caps after alternate-side damping.",
    )
    ap.add_argument(
        "--hybrid-venous-low-flow-relaxation-alpha",
        type=float,
        default=0.15,
        help="Post-solve resistance blending factor for low-flow venous branches.",
    )
    ap.add_argument(
        "--hybrid-venous-very-low-flow-relaxation-alpha",
        type=float,
        default=0.05,
        help="Post-solve resistance blending factor for very-low-flow venous branches.",
    )
    ap.add_argument(
        "--hybrid-postsolve-worsening-tolerance",
        type=float,
        default=0.02,
        help="Allowed increase in predicted bounded branch error before post-solve relaxation is strengthened.",
    )
    return ap.parse_known_args(argv)


def _extract_cli_value(argv: list[str], flag: str, default: str) -> str:
    for idx, token in enumerate(argv):
        if token == flag and idx + 1 < len(argv):
            return str(argv[idx + 1])
        if token.startswith(f"{flag}="):
            return str(token.split("=", 1)[1])
    return str(default)


def _has_cli_flag(argv: list[str], flag: str) -> bool:
    return any(str(token) == flag for token in argv)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def _parse_organoid_selector(raw: str) -> set[int] | None:
    text = str(raw).strip().lower()
    if not text or text == "all":
        return None
    out: set[int] = set()
    for part in text.split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out


def _side_selected(config: argparse.Namespace, side_label: str) -> bool:
    selected = str(config.hybrid_sides).strip().lower()
    return selected == "both" or selected == side_label


def _bootstrap_target_flow_from_iface(
    iface: dict[str, Any],
    side_label: str,
    branch_id: int,
) -> float | None:
    if side_label == "arterial":
        branch_ids = iface.get("branch_ids_inlet", [])
        q_targets = iface.get("q_inlet_target_in", iface.get("q_inlet_target", []))
        sign = 1.0
    else:
        branch_ids = iface.get("branch_ids_outlet", [])
        q_targets = iface.get("q_outlet_target_out", iface.get("q_outlet", []))
        sign = -1.0
    if not isinstance(branch_ids, list) or not isinstance(q_targets, list):
        return None
    for idx, raw_branch in enumerate(branch_ids):
        try:
            bid = int(raw_branch)
        except Exception:
            continue
        if bid != int(branch_id) or idx >= len(q_targets):
            continue
        try:
            return float(sign * float(q_targets[idx]))
        except Exception:
            return None
    return None


def _supported_q_sign_qforr(side_label: str) -> float:
    _ = side_label
    # Q_for_R is positive for both supported arterial inflow and supported
    # venous drainage because the venous physical flow is sign-flipped when it
    # is stored in Q_for_R.
    return 1.0


def _pressure_error(side_label: str, p_darcy: float, p_0d: float) -> float:
    if not (_finite(p_darcy) and _finite(p_0d)):
        return 0.0
    if side_label == "arterial":
        return float(p_darcy) - float(p_0d)
    return float(p_0d) - float(p_darcy)


def _signed_bounded_pressure_error(side_label: str, p_darcy: float, p_0d: float) -> float:
    if not (_finite(p_darcy) and _finite(p_0d)):
        return 0.0
    scale = max(abs(float(p_darcy)), abs(float(p_0d)), 1.0)
    return float(_pressure_error(side_label, p_darcy, p_0d) / scale)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as fp:
        return list(csv.DictReader(fp))


def _physical_q_from_summary_row(row: dict[str, str], side_label: str) -> float:
    hybrid_raw = row.get("hybrid_target_flow_raw", "")
    if _finite(hybrid_raw):
        return float(hybrid_raw)
    q_for_r = float(row.get("Q_for_R", float("nan")))
    if not _finite(q_for_r):
        return float("nan")
    if side_label == "venous":
        return -abs(float(q_for_r))
    return abs(float(q_for_r))


def _q0d_magnitude_from_summary_row(row: dict[str, str]) -> float:
    q_0d = row.get("Q_0D", "")
    if _finite(q_0d):
        return abs(float(q_0d))
    q_phys = _physical_q_from_summary_row(row, str(row.get("side", "")).strip().lower())
    if _finite(q_phys):
        return abs(float(q_phys))
    return float("nan")


def _build_q_error_sensitivity_map(
    coupled_root: Path,
    current_run: int,
    history_window: int,
    min_logq_change: float,
    min_error_change: float,
    ema_alpha: float,
) -> dict[tuple[int, str, int], dict[str, float]]:
    prev_run = int(current_run) - 1
    if prev_run < 2:
        return {}

    start_run = max(1, prev_run - max(int(history_window), 2) + 1)
    history: dict[tuple[int, str, int], list[dict[str, float]]] = {}
    for run_idx in range(start_run, prev_run + 1):
        run_dir = coupled_root / f"run_{run_idx}"
        summary_rows = _read_csv_rows(run_dir / "terminal_resistance_bc_summary.csv")
        result_rows = _read_csv_rows(run_dir / "post_darcy_terminal_convergence.csv")
        if not summary_rows or not result_rows:
            continue
        result_by_key: dict[tuple[int, str, int], dict[str, str]] = {}
        for row in result_rows:
            try:
                key = (int(row.get("organoid", 0)), str(row.get("side", "")).strip().lower(), int(row.get("branch_id", -1)))
            except Exception:
                continue
            result_by_key[key] = row
        for row in summary_rows:
            try:
                key = (int(row.get("organoid", 0)), str(row.get("side", "")).strip().lower(), int(row.get("branch_id", -1)))
            except Exception:
                continue
            result = result_by_key.get(key)
            if result is None:
                continue
            q_mag = _q0d_magnitude_from_summary_row(row)
            if not _finite(q_mag):
                continue
            logq = math.log10(max(abs(float(q_mag)), sys.float_info.min))
            p_darcy = float(result.get("P_Darcy", float("nan")))
            p_0d = float(result.get("P_0D_target", float("nan")))
            if not (_finite(p_darcy) and _finite(p_0d)):
                continue
            error = _signed_bounded_pressure_error(str(key[1]), p_darcy, p_0d)
            history.setdefault(key, []).append({
                "run": float(run_idx),
                "logq": float(logq),
                "p_darcy": float(p_darcy),
                "error": float(error),
                "abs_error": float(abs(error)),
            })

    out: dict[tuple[int, str, int], dict[str, float]] = {}
    min_dx = max(float(min_logq_change), sys.float_info.min)
    ema_alpha = max(0.0, min(float(ema_alpha), 1.0))
    for key, items in history.items():
        items = sorted(items, key=lambda item: item["run"])
        signed_slopes: list[float] = []
        abs_error_slopes: list[float] = []
        improvement_ratios: list[float] = []
        ema_abs_slope = float("nan")
        for prev_item, cur_item in zip(items[:-1], items[1:]):
            dx = float(cur_item["logq"] - prev_item["logq"])
            de = float(cur_item["error"] - prev_item["error"])
            if not (_finite(dx) and _finite(de)):
                continue
            if abs(dx) <= sys.float_info.min:
                continue
            dx_eff = math.copysign(max(abs(dx), min_dx), dx)
            slope = de / dx_eff
            if not _finite(slope):
                continue
            slope_abs = abs(float(slope))
            signed_slopes.append(float(slope))
            abs_error_slopes.append(float(slope_abs))
            if not _finite(ema_abs_slope):
                ema_abs_slope = float(slope_abs)
            else:
                ema_abs_slope = float(ema_alpha) * float(slope_abs) + (1.0 - float(ema_alpha)) * float(ema_abs_slope)
            prev_abs_error = max(float(prev_item["abs_error"]), sys.float_info.min)
            improvement_ratios.append(
                float((float(prev_item["abs_error"]) - float(cur_item["abs_error"])) / prev_abs_error)
            )
        if not abs_error_slopes:
            continue
        signs = [1.0 if value > 0.0 else -1.0 if value < 0.0 else 0.0 for value in signed_slopes]
        sign_consistency = abs(sum(signs) / len(signs)) if signs else 0.0
        signed_sorted = sorted(signed_slopes)
        improvement_sorted = sorted(improvement_ratios) if improvement_ratios else [0.0]
        rolling_abs_mean = float(sum(abs_error_slopes) / len(abs_error_slopes))
        out[key] = {
            "de_dlogq_abs_mean": float(rolling_abs_mean),
            "de_dlogq_abs_ema": float(ema_abs_slope),
            "de_dlogq_signed": float(signed_sorted[len(signed_sorted) // 2]),
            "recent_de_dlogq_signed": float(signed_slopes[-1]),
            "sign_consistency": float(sign_consistency),
            "sample_count": float(len(abs_error_slopes)),
            "recent_improvement_rel": float(improvement_ratios[-1] if improvement_ratios else 0.0),
            "median_improvement_rel": float(improvement_sorted[len(improvement_sorted) // 2]),
            "oscillation_ratio": float("nan"),
        }
    return out


def _build_q_error_sensitivity_fallbacks(
    q_error_sensitivity_map: dict[tuple[int, str, int], dict[str, float]] | None,
) -> dict[tuple[int | None, str], dict[str, float]]:
    if not q_error_sensitivity_map:
        return {}

    grouped: dict[tuple[int | None, str], list[dict[str, float]]] = {}
    for (organoid_idx, side_label, _branch_id), info in q_error_sensitivity_map.items():
        if not info:
            continue
        grouped.setdefault((int(organoid_idx), str(side_label)), []).append(info)
        grouped.setdefault((None, str(side_label)), []).append(info)
        grouped.setdefault((None, "both"), []).append(info)

    fallback_map: dict[tuple[int | None, str], dict[str, float]] = {}
    for key, infos in grouped.items():
        slopes = sorted(
            float(info.get("de_dlogq_abs_ema", info.get("de_dlogq_abs_mean", float("nan"))))
            for info in infos
            if _finite(info.get("de_dlogq_abs_ema", info.get("de_dlogq_abs_mean")))
        )
        sign_consistency_vals = sorted(
            float(info["sign_consistency"])
            for info in infos
            if _finite(info.get("sign_consistency"))
        )
        sample_counts = sorted(
            float(info["sample_count"])
            for info in infos
            if _finite(info.get("sample_count"))
        )
        if not slopes:
            continue
        fallback_map[key] = {
            "de_dlogq_abs_mean": float(slopes[len(slopes) // 2]),
            "de_dlogq_abs_ema": float(slopes[len(slopes) // 2]),
            "sign_consistency": (
                float(sign_consistency_vals[len(sign_consistency_vals) // 2])
                if sign_consistency_vals
                else 1.0
            ),
            "sample_count": (
                float(sample_counts[len(sample_counts) // 2])
                if sample_counts
                else 1.0
            ),
        }
    return fallback_map


def _hybrid_target_flow(
    *,
    side_label: str,
    q_drive: float,
    p_darcy: float,
    p_0d: float,
    q_floor: float,
    pressure_scale_floor: float,
    log_gain: float,
    max_log_step: float,
    moderate_log_step: float,
    large_log_step: float,
    extreme_log_step: float,
    arterial_negative_pressure_threshold: float,
) -> tuple[float, str, float, float, float]:
    q_mag = max(abs(float(q_drive)), float(q_floor))
    error = _pressure_error(side_label, p_darcy, p_0d)
    pressure_scale = max(
        abs(float(p_darcy)) if _finite(p_darcy) else 0.0,
        abs(float(p_0d)) if _finite(p_0d) else 0.0,
        float(pressure_scale_floor),
    )
    rel_error = float(error) / float(pressure_scale)

    if (
        side_label == "arterial"
        and _finite(p_darcy)
        and float(p_darcy) < float(arterial_negative_pressure_threshold)
    ):
        pathology_strength = abs(float(p_darcy)) / max(float(pressure_scale), float(pressure_scale_floor))
        if pathology_strength >= 1.0:
            adaptive_cap = max(float(max_log_step), float(extreme_log_step))
        else:
            adaptive_cap = max(float(max_log_step), float(large_log_step))
        delta_log = -adaptive_cap
        q_target_mag = max(q_mag * (10.0 ** delta_log), float(q_floor))
        q_target = _supported_q_sign_qforr(side_label) * q_target_mag
        return (
            float(q_target),
            "arterial_negative_pressure_back_off",
            float(delta_log),
            float(rel_error),
            float(adaptive_cap),
        )

    rel_error_abs = abs(float(rel_error))
    if rel_error_abs >= 1.0:
        adaptive_cap = max(float(max_log_step), float(extreme_log_step))
    elif rel_error_abs >= 0.5:
        adaptive_cap = max(float(max_log_step), float(large_log_step))
    elif rel_error_abs >= 0.1:
        adaptive_cap = max(float(max_log_step), float(moderate_log_step))
    else:
        adaptive_cap = float(max_log_step)
    delta_log = max(-adaptive_cap, min(adaptive_cap, -float(log_gain) * rel_error))
    q_target_mag = max(q_mag * (10.0 ** delta_log), float(q_floor))
    q_target = _supported_q_sign_qforr(side_label) * q_target_mag
    if delta_log < -1.0e-12:
        action = "back_off"
    elif delta_log > 1.0e-12:
        action = "open_up"
    else:
        action = "hold"
    return float(q_target), action, float(delta_log), float(rel_error), float(adaptive_cap)


def _retarget_plan_q(plan: dict[str, Any], new_delta_log: float, action_suffix: str) -> None:
    q_drive = float(plan["q_drive"])
    q_floor = float(plan["q_floor"])
    q_mag = max(abs(q_drive), q_floor)
    q_target_mag = max(q_mag * (10.0 ** float(new_delta_log)), q_floor)
    plan["q_target"] = float(_supported_q_sign_qforr(str(plan["side_label"])) * q_target_mag)
    plan["delta_log"] = float(new_delta_log)
    plan["action"] = f"{plan['action']}_{action_suffix}"


def _set_plan_q_target_from_delta(plan: dict[str, Any], new_delta_log: float) -> None:
    q_drive = float(plan["q_drive"])
    q_floor = float(plan["q_floor"])
    q_mag = max(abs(q_drive), q_floor)
    q_target_mag = max(q_mag * (10.0 ** float(new_delta_log)), q_floor)
    plan["q_target"] = float(_supported_q_sign_qforr(str(plan["side_label"])) * q_target_mag)
    plan["delta_log"] = float(new_delta_log)


def _active_side_for_run(
    *,
    run_idx: int,
    hybrid_start_run: int,
    start_side: str,
) -> str:
    parity = max(int(run_idx) - int(hybrid_start_run), 0) % 2
    first = str(start_side).strip().lower()
    second = "arterial" if first == "venous" else "venous"
    return first if parity == 0 else second


def _apply_alternating_side_damping(
    plans: list[dict[str, Any]],
    *,
    active_side: str,
    inactive_scale: float,
) -> None:
    inactive_scale = max(0.0, min(float(inactive_scale), 1.0))
    active_side = str(active_side).strip().lower()
    if inactive_scale >= 1.0:
        return
    for plan in plans:
        side_label = str(plan.get("side_label", "")).strip().lower()
        if side_label == active_side:
            plan["hybrid_alternate_side_active"] = True
            plan["hybrid_alternate_side_scale"] = 1.0
            continue
        old_delta = float(plan.get("delta_log", 0.0))
        new_delta = float(old_delta) * inactive_scale
        if abs(new_delta - old_delta) > 1.0e-12:
            _retarget_plan_q(plan, new_delta, "alternate_damped")
        plan["hybrid_alternate_side_active"] = False
        plan["hybrid_alternate_side_scale"] = float(inactive_scale)


def _apply_q_error_sensitivity_scaling(
    plans: list[dict[str, Any]],
    q_error_sensitivity_map: dict[tuple[int, str, int], dict[str, float]] | None,
    q_error_sensitivity_fallback_map: dict[tuple[int | None, str], dict[str, float]] | None,
    gain: float,
    arterial_gain: float | None,
    venous_gain: float | None,
) -> None:
    gain = max(float(gain), 0.0)
    arterial_gain = None if arterial_gain is None else max(float(arterial_gain), 0.0)
    venous_gain = None if venous_gain is None else max(float(venous_gain), 0.0)
    global_fallback = None
    if q_error_sensitivity_fallback_map:
        global_fallback = q_error_sensitivity_fallback_map.get((None, "both"))
    for plan in plans:
        side_label = str(plan.get("side_label", "")).strip().lower()
        organoid_idx = int(plan.get("organoid_idx", -1))
        key = (
            organoid_idx,
            side_label,
            int(plan.get("branch_id", -1)),
        )
        info = q_error_sensitivity_map.get(key)
        sensitivity_source = "branch"
        if q_error_sensitivity_fallback_map:
            fallback_info = q_error_sensitivity_fallback_map.get((organoid_idx, side_label))
            if fallback_info is None:
                fallback_info = q_error_sensitivity_fallback_map.get((None, side_label))
        else:
            fallback_info = None
        if not info and q_error_sensitivity_fallback_map:
            info = fallback_info
            if info and (organoid_idx, side_label) in q_error_sensitivity_fallback_map:
                sensitivity_source = "organoid_side_fallback"
            elif info:
                sensitivity_source = "side_fallback"
        if not info and global_fallback:
            info = global_fallback
            sensitivity_source = "global_fallback"
        if not info:
            info = {"de_dlogq_abs_ema": 1.0, "sign_consistency": 1.0, "sample_count": 0.0}
            sensitivity_source = "unit_floor"
        slope_mag = float(info.get("de_dlogq_abs_ema", info.get("de_dlogq_abs_mean", float("nan"))))
        sign_consistency = float(info.get("sign_consistency", float("nan")))
        sample_count = float(info.get("sample_count", float("nan")))
        old_delta = float(plan.get("delta_log", float("nan")))
        error = float(plan.get("error", float("nan")))
        adaptive_cap = float(plan.get("adaptive_cap", 0.0))
        if not (
            _finite(slope_mag)
            and _finite(old_delta)
            and _finite(error)
        ):
            continue
        slope_mag = max(abs(float(slope_mag)), sys.float_info.min)

        local_gain = gain
        if side_label == "arterial" and arterial_gain is not None:
            local_gain = arterial_gain
        elif side_label == "venous" and venous_gain is not None:
            local_gain = venous_gain

        # Always-on normalized EMA sensitivity magnitude with direction from
        # the current mismatch sign.
        delta_mag = local_gain * abs(float(error)) / float(slope_mag)
        delta_secant = -math.copysign(delta_mag, float(error))
        if not _finite(delta_secant):
            continue
        delta_secant = max(-adaptive_cap, min(adaptive_cap, delta_secant))
        new_delta = float(delta_secant)
        if abs(new_delta) <= 1.0e-12:
            continue
        if abs(new_delta - old_delta) <= 1.0e-12:
            continue

        _retarget_plan_q(plan, float(new_delta), "branch_secant")
        plan["hybrid_sensitivity_applied"] = True
        plan["hybrid_sensitivity_multiplier"] = (
            float(abs(new_delta) / abs(old_delta))
            if abs(old_delta) > 1.0e-12
            else float("nan")
        )
        plan["hybrid_sensitivity_slope"] = float(slope_mag)
        plan["hybrid_sensitivity_sign_consistency"] = float(sign_consistency)
        plan["hybrid_sensitivity_sample_count"] = float(sample_count)
        plan["hybrid_sensitivity_stable_r_count"] = float("nan")
        plan["hybrid_q_error_sensitivity_applied"] = True
        plan["hybrid_q_error_sensitivity_slope"] = float(slope_mag)
        plan["hybrid_q_error_sensitivity_sign_consistency"] = float(sign_consistency)
        plan["hybrid_q_error_sensitivity_sample_count"] = float(sample_count)
        plan["hybrid_q_error_sensitivity_multiplier"] = (
            float(abs(new_delta) / abs(old_delta))
            if abs(old_delta) > 1.0e-12
            else float("nan")
        )
        plan["hybrid_q_error_sensitivity_source"] = str(sensitivity_source)


def _solve_r_for_target_flows(
    *,
    plans: list[dict[str, Any]],
    full_model: SteadyResistiveZeroDModel | None,
    q_floor: float,
    r_floor: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, float], str, float, int]:
    if not plans or full_model is None:
        return {}, {}, {}, "full_resistive_unavailable", float("nan"), 0

    q_by_name: dict[str, float] = {}
    pd_seed_by_name: dict[str, float] = {}
    for plan in plans:
        q_target = float(plan["q_target"])
        if str(plan["side_label"]) == "venous":
            q_target = -q_target
        q_by_name[str(plan["bc_name"])] = q_target
        pd_seed = float(plan.get("pd_seed", float("nan")))
        if not _finite(pd_seed):
            pd_seed = float(plan.get("row", {}).get("Pd_next", float("nan")))
        if not _finite(pd_seed):
            pd_seed = float(plan.get("row", {}).get("P_0D", float("nan")))
        if not _finite(pd_seed):
            pd_seed = 0.0
        pd_seed_by_name[str(plan["bc_name"])] = pd_seed

    pterm_by_name = full_model.solve_terminal_pressures_for_terminal_flows(
        terminal_flow_targets=q_by_name,
    )

    r_by_name: dict[str, float] = {}
    pd_by_name: dict[str, float] = {}
    residuals: list[float] = []
    for name, q_internal in q_by_name.items():
        p_term = float(pterm_by_name.get(name, float("nan")))
        pd_seed = float(pd_seed_by_name.get(name, 0.0))
        if not (_finite(p_term) and _finite(q_internal)) or abs(q_internal) <= max(float(q_floor), sys.float_info.min):
            continue

        raw_r = (p_term - pd_seed) / q_internal
        if _finite(raw_r) and raw_r > float(r_floor):
            r_next = float(raw_r)
            pd_next = float(pd_seed)
        else:
            r_next = max(float(r_floor), sys.float_info.min)
            pd_next = float(p_term - r_next * q_internal)

        r_by_name[name] = float(r_next)
        pd_by_name[name] = float(pd_next)
        residuals.append(abs(p_term - (pd_next + r_next * q_internal)))

    residual = max(residuals) if residuals else float("nan")
    return r_by_name, pd_by_name, pterm_by_name, "full_resistive_targetQ_solve_R", residual, 1


def _build_hybrid_override(config: argparse.Namespace):
    selected_organoids = _parse_organoid_selector(config.hybrid_organoids)
    q_floor_override = max(float(config.hybrid_q_floor), sys.float_info.min)
    r_floor_override = max(float(config.hybrid_r_floor), 1.0)
    bootstrap_q_threshold = max(float(config.hybrid_bootstrap_q_threshold), 0.0)
    start_run = max(int(config.hybrid_start_run), 1)
    pending_state: dict[str, Any] = {
        "run_idx": None,
        "deck_id": None,
        "plans": [],
        "q_error_sensitivity_map": {},
        "q_error_sensitivity_fallback_map": {},
        "row_refs": [],
    }

    original = coupler.apply_organoid_resistance_from_json

    def wrapped(
        deck: dict,
        organoid_idx: int,
        iface: dict,
        resistance_relaxation: float,
        pd_relaxation: float,
        q_floor: float,
        venous_pd_floor: float | None = 0.0,
        pd_mode: str = "interface",
        suppression_pressure_tol: float = 5.0,
        arterial_suppression_parent_margin: float = 0.0,
        venous_suppression_parent_margin: float = 0.0,
        continuity_pressure_tol: float = 5.0,
        continuity_resistance_decay: float = 0.8,
        pressure_alignment_jump_tol: float | None = None,
        active_terminal_side: str | None = "both",
        permanent_resistance_controls: bool = True,
        pressure_handoff_enabled: bool = True,
        pressure_handoff_rel_tol: float = 0.10,
        pressure_handoff_abs_tol: float = 5.0,
        inverse_flow_pd_relaxation_fraction: float = 0.5,
        inverse_flow_pd_relaxation_gamma: float = 0.5,
        inverse_flow_pd_relaxation_weight_cap: float = 10.0,
        terminal_response_correction_gain: float = 0.0,
        terminal_response_correction_min_flow_weight: float = 2.0,
        terminal_response_correction_rel_error_tol: float = 0.02,
        terminal_response_correction_error_scale: float = 0.10,
        terminal_response_correction_error_gamma: float = 1.0,
        terminal_response_correction_effective_gain_cap: float = 3.0,
        terminal_response_correction_guarded_reasons: bool = False,
        stubborn_terminal_response_map: dict[tuple[int, str, int], dict[str, float]] | None = None,
        adaptive_terminal_state_map: dict[tuple[int, str, int], dict[str, Any]] | None = None,
        terminal_pressure_sensitivity_map: dict[tuple[int, str, int], dict[str, float]] | None = None,
        terminal_sensitivity_rel_error_tol: float = 0.02,
        terminal_sensitivity_gain: float = 0.5,
        terminal_sensitivity_min_abs_slope: float = 0.01,
        terminal_sensitivity_sign_consistency: float = 0.75,
        terminal_sensitivity_max_multiplier: float = 3.0,
        pressure_match_stable_r_rel_threshold: float = 1.0e-2,
        pressure_match_min_stable_r_runs: int = 3,
        allow_missing_terminal_bcs: bool = False,
        current_run_idx: int | None = None,
    ) -> list[dict[str, Any]]:
        state_key = (int(current_run_idx) if current_run_idx is not None else None, id(deck))
        if pending_state["run_idx"] != state_key[0] or pending_state["deck_id"] != state_key[1]:
            pending_state["run_idx"] = state_key[0]
            pending_state["deck_id"] = state_key[1]
            pending_state["plans"] = []
            pending_state["row_refs"] = []
            if (
                bool(config.hybrid_q_sensitivity_control)
                and current_run_idx is not None
                and int(current_run_idx) >= start_run
            ):
                pending_state["q_error_sensitivity_map"] = _build_q_error_sensitivity_map(
                    coupled_root=Path(str(config._coupled_root)).expanduser().resolve(),
                    current_run=int(current_run_idx),
                    history_window=int(config.hybrid_q_sensitivity_history_window),
                    min_logq_change=float(config.hybrid_q_sensitivity_min_logq_change),
                    min_error_change=float(config.hybrid_q_sensitivity_min_error_change),
                    ema_alpha=float(config.hybrid_q_sensitivity_ema_alpha),
                )
                pending_state["q_error_sensitivity_fallback_map"] = _build_q_error_sensitivity_fallbacks(
                    pending_state["q_error_sensitivity_map"]
                )
            else:
                pending_state["q_error_sensitivity_map"] = {}
                pending_state["q_error_sensitivity_fallback_map"] = {}

        rows = original(
            deck=deck,
            organoid_idx=organoid_idx,
            iface=iface,
            resistance_relaxation=resistance_relaxation,
            pd_relaxation=pd_relaxation,
            q_floor=q_floor,
            venous_pd_floor=venous_pd_floor,
            pd_mode=pd_mode,
            suppression_pressure_tol=suppression_pressure_tol,
            arterial_suppression_parent_margin=arterial_suppression_parent_margin,
            venous_suppression_parent_margin=venous_suppression_parent_margin,
            continuity_pressure_tol=continuity_pressure_tol,
            continuity_resistance_decay=continuity_resistance_decay,
            pressure_alignment_jump_tol=pressure_alignment_jump_tol,
            active_terminal_side=active_terminal_side,
            permanent_resistance_controls=permanent_resistance_controls,
            pressure_handoff_enabled=pressure_handoff_enabled,
            pressure_handoff_rel_tol=pressure_handoff_rel_tol,
            pressure_handoff_abs_tol=pressure_handoff_abs_tol,
            inverse_flow_pd_relaxation_fraction=inverse_flow_pd_relaxation_fraction,
            inverse_flow_pd_relaxation_gamma=inverse_flow_pd_relaxation_gamma,
            inverse_flow_pd_relaxation_weight_cap=inverse_flow_pd_relaxation_weight_cap,
            terminal_response_correction_gain=terminal_response_correction_gain,
            terminal_response_correction_min_flow_weight=terminal_response_correction_min_flow_weight,
            terminal_response_correction_rel_error_tol=terminal_response_correction_rel_error_tol,
            terminal_response_correction_error_scale=terminal_response_correction_error_scale,
            terminal_response_correction_error_gamma=terminal_response_correction_error_gamma,
            terminal_response_correction_effective_gain_cap=terminal_response_correction_effective_gain_cap,
            terminal_response_correction_guarded_reasons=terminal_response_correction_guarded_reasons,
            stubborn_terminal_response_map=stubborn_terminal_response_map,
            adaptive_terminal_state_map=adaptive_terminal_state_map,
            terminal_pressure_sensitivity_map=terminal_pressure_sensitivity_map,
            terminal_sensitivity_rel_error_tol=terminal_sensitivity_rel_error_tol,
            terminal_sensitivity_gain=terminal_sensitivity_gain,
            terminal_sensitivity_min_abs_slope=terminal_sensitivity_min_abs_slope,
            terminal_sensitivity_sign_consistency=terminal_sensitivity_sign_consistency,
            terminal_sensitivity_max_multiplier=terminal_sensitivity_max_multiplier,
            pressure_match_stable_r_rel_threshold=pressure_match_stable_r_rel_threshold,
            pressure_match_min_stable_r_runs=pressure_match_min_stable_r_runs,
            allow_missing_terminal_bcs=allow_missing_terminal_bcs,
            current_run_idx=current_run_idx,
        )

        if current_run_idx is None or int(current_run_idx) < start_run:
            return rows
        if selected_organoids is not None and int(organoid_idx) not in selected_organoids:
            return rows

        bc_by_name = {
            str(bc.get("bc_name", "")): bc
            for bc in deck.get("boundary_conditions", [])
        }
        q_floor_local = max(float(q_floor_override), float(q_floor), sys.float_info.min)
        r_floor_local = max(float(r_floor_override), 1.0)

        for row in rows:
            side_label = str(row.get("side", "")).strip().lower()
            if not _side_selected(config, side_label):
                continue
            bc_name = str(row.get("bc_name", ""))
            bc = bc_by_name.get(bc_name)
            if bc is None:
                continue

            q_drive = float(row.get("Q_for_R", float("nan")))
            branch_id = int(row.get("branch_id", -1))
            if not _finite(q_drive):
                continue
            if abs(q_drive) <= bootstrap_q_threshold:
                q_boot = _bootstrap_target_flow_from_iface(iface, side_label, branch_id)
                if _finite(q_boot):
                    q_drive = float(q_boot)

            p_darcy = float(row.get("P_interface", float("nan")))
            p_0d = float(row.get("P_0D", float("nan")))
            q_target, action, delta_log, rel_error, adaptive_cap = _hybrid_target_flow(
                side_label=side_label,
                q_drive=q_drive,
                p_darcy=p_darcy,
                p_0d=p_0d,
                q_floor=q_floor_local,
                pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
                log_gain=float(config.hybrid_log_gain),
                max_log_step=float(
                    config.hybrid_venous_max_log_step
                    if side_label == "venous"
                    else config.hybrid_max_log_step
                ),
                moderate_log_step=float(
                    config.hybrid_venous_moderate_log_step
                    if side_label == "venous"
                    else config.hybrid_moderate_log_step
                ),
                large_log_step=float(
                    config.hybrid_venous_large_log_step
                    if side_label == "venous"
                    else config.hybrid_large_log_step
                ),
                extreme_log_step=float(
                    config.hybrid_venous_extreme_log_step
                    if side_label == "venous"
                    else config.hybrid_extreme_log_step
                ),
                arterial_negative_pressure_threshold=float(
                    config.hybrid_arterial_negative_pressure_threshold
                ),
            )

            if str(config.hybrid_resistance_source) == "previous":
                r_trial = float(row.get("R_previous", float("nan")))
            else:
                r_trial = float(row.get("R_next", float("nan")))
            if not _finite(r_trial) or r_trial <= 0.0:
                r_trial = float(row.get("R_previous", float("nan")))
            if not _finite(r_trial) or r_trial <= 0.0:
                r_trial = float(r_floor_local)
            r_trial = max(float(r_trial), float(r_floor_local), sys.float_info.min)
            pd_seed = float(row.get("Pd_next", float("nan")))
            if not _finite(pd_seed):
                pd_seed = float(row.get("Pd_target", float("nan")))
            if not _finite(pd_seed):
                pd_seed = float(row.get("P_0D", float("nan")))

            row["hybrid_controller_mode"] = "base_R_plus_log_q_target"
            row["hybrid_target_flow"] = float(q_target)
            row["hybrid_target_flow_raw"] = float(
                q_target if side_label == "arterial" else -float(q_target)
            )
            row["hybrid_q_drive"] = float(q_drive)
            row["hybrid_action"] = str(action)
            row["hybrid_delta_log10_q"] = float(delta_log)
            row["hybrid_relative_pressure_error"] = float(rel_error)
            row["hybrid_adaptive_log10_cap"] = float(adaptive_cap)
            row["hybrid_r_trial"] = float(r_trial)

            pending_state["plans"].append({
                "row": row,
                "organoid_idx": int(organoid_idx),
                "side_label": side_label,
                "bc_name": bc_name,
                "bc": bc,
                "branch_id": branch_id,
                "q_drive": float(q_drive),
                "q_target": float(q_target),
                "r_trial": float(r_trial),
                "pd_seed": float(pd_seed),
                "q_floor": float(q_floor_local),
                "delta_log": float(delta_log),
                "rel_error": float(rel_error),
                "adaptive_cap": float(adaptive_cap),
                "action": str(action),
                "error": float(_signed_bounded_pressure_error(side_label, p_darcy, p_0d)),
            })
            pending_state["row_refs"].append(row)

        if int(organoid_idx) != int(getattr(config, "_n_organoids", organoid_idx)):
            return rows

        all_plans = list(pending_state["plans"])
        pending_state["plans"] = []
        if not all_plans:
            return rows

        _apply_q_error_sensitivity_scaling(
            all_plans,
            q_error_sensitivity_map=pending_state.get("q_error_sensitivity_map", {}),
            q_error_sensitivity_fallback_map=pending_state.get("q_error_sensitivity_fallback_map", {}),
            gain=float(config.hybrid_q_sensitivity_alpha),
            arterial_gain=config.hybrid_q_sensitivity_alpha_arterial,
            venous_gain=config.hybrid_q_sensitivity_alpha_venous,
        )
        if bool(config.hybrid_alternate_sides):
            active_side = _active_side_for_run(
                run_idx=int(current_run_idx),
                hybrid_start_run=int(start_run),
                start_side=str(config.hybrid_alternate_start_side),
            )
            _apply_alternating_side_damping(
                all_plans,
                active_side=active_side,
                inactive_scale=float(config.hybrid_inactive_side_alpha_scale),
            )
        full_model: SteadyResistiveZeroDModel | None = None
        try:
            full_model = SteadyResistiveZeroDModel.from_data(deck)
        except Exception:
            full_model = None

        r_fixed_by_name: dict[str, float] = {}
        pd_solved_by_name: dict[str, float] = {}
        pterm_by_name: dict[str, float] = {}
        solve_mode = "full_resistive_unavailable"
        algebraic_residual_linf = float("nan")
        algebraic_iterations = 0
        if full_model is not None:
            try:
                (
                    r_fixed_by_name,
                    pd_solved_by_name,
                    pterm_by_name,
                    solve_mode,
                    algebraic_residual_linf,
                    algebraic_iterations,
                ) = _solve_r_for_target_flows(
                    plans=all_plans,
                    full_model=full_model,
                    q_floor=q_floor_local,
                    r_floor=r_floor_local,
                )
            except Exception:
                r_fixed_by_name = {}
                pd_solved_by_name = {}
                pterm_by_name = {}
                solve_mode = "full_resistive_failed"
                algebraic_residual_linf = float("nan")
                algebraic_iterations = 0

        for plan in all_plans:
            row = plan["row"]
            bc = plan["bc"]
            bc_name = str(plan["bc_name"])
            if bc_name not in pd_solved_by_name or bc_name not in r_fixed_by_name:
                continue
            pd_next = float(pd_solved_by_name[bc_name])
            r_next = float(r_fixed_by_name[bc_name])
            coupler.set_resistance_bc(bc, r_next, distal_pressure=pd_next)

            row["bc_type"] = "RESISTANCE"
            row["Pd_target"] = float(pd_next)
            row["Pd_target_reason"] = "hybrid_algebraic_targetQ_solve_R"
            row["Pd_next"] = float(pd_next)
            row["P_d_for_R"] = float(pd_next)
            row["R_raw"] = float(r_next)
            row["R_next"] = float(r_next)
            row["hybrid_controller_mode"] = "base_R_plus_log_q_target"
            row["hybrid_target_flow"] = float(plan["q_target"])
            row["hybrid_target_flow_raw"] = float(plan["q_target"] if str(plan["side_label"]) == "arterial" else -float(plan["q_target"]))
            row["hybrid_q_drive"] = float(plan["q_drive"])
            row["hybrid_action"] = str(plan["action"])
            row["hybrid_delta_log10_q"] = float(plan["delta_log"])
            row["hybrid_relative_pressure_error"] = float(plan["rel_error"])
            row["hybrid_adaptive_log10_cap"] = float(plan["adaptive_cap"])
            row["hybrid_r_trial"] = float(plan["r_trial"])
            row["hybrid_venous_dominant_damped"] = False
            row["hybrid_venous_median_error"] = float("nan")
            row["hybrid_arterial_median_error"] = float("nan")
            row["hybrid_previous_principle_applied"] = False
            row["hybrid_previous_principle_mode"] = ""
            row["hybrid_previous_principle_multiplier"] = float("nan")
            row["hybrid_alternate_side_active"] = bool(
                plan.get("hybrid_alternate_side_active", True)
            )
            row["hybrid_alternate_side_scale"] = float(
                plan.get("hybrid_alternate_side_scale", 1.0)
            )
            row["hybrid_q_error_sensitivity_applied"] = bool(
                plan.get("hybrid_q_error_sensitivity_applied", False)
            )
            row["hybrid_q_error_sensitivity_slope"] = float(
                plan.get("hybrid_q_error_sensitivity_slope", float("nan"))
            )
            row["hybrid_q_error_sensitivity_sign_consistency"] = float(
                plan.get("hybrid_q_error_sensitivity_sign_consistency", float("nan"))
            )
            row["hybrid_q_error_sensitivity_sample_count"] = float(
                plan.get("hybrid_q_error_sensitivity_sample_count", float("nan"))
            )
            row["hybrid_q_error_sensitivity_multiplier"] = float(
                plan.get("hybrid_q_error_sensitivity_multiplier", float("nan"))
            )
            row["hybrid_q_error_sensitivity_source"] = str(
                plan.get("hybrid_q_error_sensitivity_source", "")
            )
            row["hybrid_branch_trust_region_cap"] = float(
                plan.get("hybrid_branch_trust_region_cap", float("nan"))
            )
            row["hybrid_postsolve_relaxation_alpha"] = float(
                row.get("hybrid_postsolve_relaxation_alpha", float("nan"))
            )
            row["hybrid_predicted_bounded_error"] = float(
                row.get("hybrid_predicted_bounded_error", float("nan"))
            )
            row["hybrid_sensitivity_applied"] = bool(plan.get("hybrid_sensitivity_applied", False))
            row["hybrid_sensitivity_multiplier"] = float(
                plan.get("hybrid_sensitivity_multiplier", float("nan"))
            )
            row["hybrid_sensitivity_slope"] = float(
                plan.get("hybrid_sensitivity_slope", float("nan"))
            )
            row["hybrid_sensitivity_sign_consistency"] = float(
                plan.get("hybrid_sensitivity_sign_consistency", float("nan"))
            )
            row["hybrid_sensitivity_sample_count"] = float(
                plan.get("hybrid_sensitivity_sample_count", float("nan"))
            )
            row["hybrid_sensitivity_stable_r_count"] = float(
                plan.get("hybrid_sensitivity_stable_r_count", float("nan"))
            )
            row["hybrid_algebraic_mode"] = str(solve_mode)
            row["hybrid_algebraic_residual_linf"] = float(algebraic_residual_linf)
            row["hybrid_algebraic_iterations"] = int(algebraic_iterations)
            row["P_predicted_terminal_from_algebraic"] = float(
                pterm_by_name.get(bc_name, row.get("P_predicted_terminal_from_algebraic", float("nan")))
            )

        pending_state["row_refs"] = []

        return rows

    return wrapped


def main() -> None:
    wrapper_args, coupler_argv = _parse_wrapper_args(sys.argv[1:])
    wrapper_args._svzerodsolver = _extract_cli_value(coupler_argv, "--svzerodsolver", "svzerodsolver")
    wrapper_args._run_and_split = _extract_cli_value(coupler_argv, "--run-and-split", "../prep/run_and_split_svzerod.py")
    wrapper_args._debug = _has_cli_flag(coupler_argv, "--debug")
    wrapper_args._n_organoids = int(_extract_cli_value(coupler_argv, "--n-organoids", "1"))
    wrapper_args._coupled_root = _extract_cli_value(coupler_argv, "--coupled-root", ".")

    original_load_json = coupler.load_json

    def load_json_with_source(path: Path) -> dict:
        data = original_load_json(path)
        if isinstance(data, dict):
            data["__source_path__"] = str(Path(path))
        return data

    coupler.load_json = load_json_with_source
    coupler.apply_organoid_resistance_from_json = _build_hybrid_override(wrapper_args)
    sys.argv = [sys.argv[0], *coupler_argv]
    coupler.main()


if __name__ == "__main__":
    main()
