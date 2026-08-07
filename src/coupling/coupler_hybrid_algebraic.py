#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
        "--hybrid-opposite-side-referenced-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use opposite-side median-referenced Darcy/0D pressure-drop "
            "mismatch as the branch error after the initial pressure-drop "
            "alignment stage and before the final sensitivity stage."
        ),
    )
    ap.add_argument(
        "--hybrid-side-gauge-common-mode-control",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Once all branches in an organoid are inside the configured "
            "pressure-drop error band, apply one common-mode log-flow shift "
            "to each side to collapse/expand all branches together and reduce "
            "side-median gauge mismatch."
        ),
    )
    ap.add_argument(
        "--hybrid-side-gauge-common-mode-band",
        type=float,
        default=0.10,
        help=(
            "Maximum median absolute branch pressure-drop error allowed in "
            "an organoid before the side common-mode gauge correction is "
            "allowed to activate."
        ),
    )
    ap.add_argument(
        "--hybrid-side-gauge-common-mode-band-gate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the median absolute branch pressure-drop error to be "
            "inside --hybrid-side-gauge-common-mode-band before applying the "
            "side common-mode correction. Use "
            "--no-hybrid-side-gauge-common-mode-band-gate to apply common mode "
            "regardless of the branch pressure-drop error."
        ),
    )
    ap.add_argument(
        "--hybrid-arterial-side-gauge-common-mode-gain",
        type=float,
        default=1.0,
        help="Gain for arterial side common-mode gauge correction.",
    )
    ap.add_argument(
        "--hybrid-venous-side-gauge-common-mode-gain",
        type=float,
        default=1.0,
        help="Gain for venous side common-mode gauge correction.",
    )
    ap.add_argument(
        "--hybrid-side-gauge-common-mode-max-log-step",
        type=float,
        default=0.02,
        help="Maximum absolute common-mode log10 flow shift applied per side.",
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
        "--hybrid-q-sensitivity-start-run",
        type=int,
        default=6,
        help=(
            "First run index where the secant-style sensitivity controller is "
            "allowed to modify the bounded base update. Earlier runs use only "
            "the base bounded controller."
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
        "--hybrid-q-sensitivity-active-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Estimate sensitivity only between active-side observations. The "
            "reference controller leaves this disabled so consecutive "
            "realized-flow/post-Darcy pairs feed the magnitude EMA; enabling "
            "it skips the intervening damped run and can mix two updates."
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
        "--hybrid-organoid-deltap-scaling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Before fine branchwise updates, apply one common log-flow shift "
            "to both trees of each organoid until its 0D/Darcy median "
            "pressure-drop ratio enters the configured band."
        ),
    )
    ap.add_argument(
        "--hybrid-run2-equalize-arterial-to-venous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On run 2, hold each organoid's venous terminal flows and "
            "uniformly scale its arterial terminal flows so the two terminal "
            "totals match. This is a one-run bootstrap, not a persistent "
            "mass-balance constraint."
        ),
    )
    ap.add_argument(
        "--hybrid-organoid-deltap-gamma",
        type=float,
        default=0.5,
        help="Gain on log10(|deltaP_0D|/|deltaP_Darcy|) during coarse scaling.",
    )
    ap.add_argument(
        "--hybrid-organoid-deltap-max-log-shift",
        type=float,
        default=0.45,
        help="Maximum absolute common log10-flow shift in one coarse run.",
    )
    ap.add_argument(
        "--hybrid-organoid-deltap-ratio-lower",
        type=float,
        default=0.9,
        help="Lower |deltaP_0D|/|deltaP_Darcy| bound for fine control.",
    )
    ap.add_argument(
        "--hybrid-organoid-deltap-ratio-upper",
        type=float,
        default=1.1,
        help="Upper |deltaP_0D|/|deltaP_Darcy| bound for fine control.",
    )
    ap.add_argument(
        "--hybrid-organoid-deltap-min-in-band-runs",
        type=int,
        default=2,
        help=(
            "Number of consecutive completed runs that must keep every "
            "organoid inside the pressure-drop band before fine control."
        ),
    )
    ap.add_argument(
        "--hybrid-post-deltap-probes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "After confirmed pressure-drop alignment, run one bounded probe "
            "on each tree, then restore the saved coarse-aligned flows before "
            "sensitivity control."
        ),
    )
    ap.add_argument(
        "--hybrid-post-deltap-probe-log-step",
        type=float,
        default=0.05,
        help="Maximum absolute log10-flow move during each initial side probe.",
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


def _median(values: list[float]) -> float:
    finite_values = sorted(float(value) for value in values if _finite(value))
    if not finite_values:
        return float("nan")
    middle = len(finite_values) // 2
    if len(finite_values) % 2:
        return float(finite_values[middle])
    return 0.5 * float(finite_values[middle - 1] + finite_values[middle])


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
    p_darcy = float(p_darcy)
    p_0d = float(p_0d)
    error = _pressure_error(side_label, p_darcy, p_0d)
    # Symmetric, dimensionless normalization used by the clean reference
    # runs.  It remains well scaled when either domain has the larger pressure
    # and makes the controller invariant to the absolute pressure magnitude.
    scale = max(abs(p_darcy), abs(p_0d), 1.0)
    return float(error / scale)


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
        q_mag = abs(float(q_0d))
        if q_mag > 0.0:
            return q_mag
        return float("nan")
    q_phys = _physical_q_from_summary_row(row, str(row.get("side", "")).strip().lower())
    if _finite(q_phys):
        q_mag = abs(float(q_phys))
        if q_mag > 0.0:
            return q_mag
        return float("nan")
    return float("nan")


def _opposite_side_referenced_error_map(
    pressure_by_key: dict[tuple[int, str, int], dict[str, float]],
    *,
    pressure_scale_floor: float,
) -> dict[tuple[int, str, int], dict[str, float]]:
    floor = max(float(pressure_scale_floor), sys.float_info.min)
    out: dict[tuple[int, str, int], dict[str, float]] = {}
    organoid_ids = sorted({key[0] for key in pressure_by_key})
    for organoid_idx in organoid_ids:
        medians: dict[str, dict[str, float]] = {}
        for side_label in ("arterial", "venous"):
            entries = [
                values
                for key, values in pressure_by_key.items()
                if key[0] == organoid_idx and key[1] == side_label
            ]
            medians[side_label] = {
                "p_darcy": _median([float(values["p_darcy"]) for values in entries]),
                "p_0d": _median([float(values["p_0d"]) for values in entries]),
            }
        if not all(
            _finite(medians[side][field])
            for side in ("arterial", "venous")
            for field in ("p_darcy", "p_0d")
        ):
            continue

        for key, values in pressure_by_key.items():
            if key[0] != organoid_idx or key[1] not in {"arterial", "venous"}:
                continue
            side_label = key[1]
            opposite = "venous" if side_label == "arterial" else "arterial"
            p_darcy = float(values["p_darcy"])
            p_0d = float(values["p_0d"])
            opposite_darcy = float(medians[opposite]["p_darcy"])
            opposite_0d = float(medians[opposite]["p_0d"])
            if side_label == "arterial":
                darcy_drop = p_darcy - opposite_darcy
                zero_d_drop = p_0d - opposite_0d
            else:
                darcy_drop = opposite_darcy - p_darcy
                zero_d_drop = opposite_0d - p_0d
            raw_error = float(darcy_drop - zero_d_drop)
            scale = max(abs(float(darcy_drop)), abs(float(zero_d_drop)), floor)
            out[key] = {
                "raw_error": float(raw_error),
                "normalized_error": float(raw_error / scale),
                "scale": float(scale),
                "darcy_drop": float(darcy_drop),
                "zero_d_drop": float(zero_d_drop),
                "opposite_darcy_median": opposite_darcy,
                "opposite_0d_median": opposite_0d,
            }
    return out


def _side_median_pressure_error_map(
    pressure_by_key: dict[tuple[int, str, int], dict[str, float]],
    *,
    pressure_scale_floor: float,
) -> dict[tuple[int, str, int], dict[str, float]]:
    floor = max(float(pressure_scale_floor), sys.float_info.min)
    out: dict[tuple[int, str, int], dict[str, float]] = {}
    organoid_ids = sorted({key[0] for key in pressure_by_key})
    for organoid_idx in organoid_ids:
        medians: dict[str, dict[str, float]] = {}
        for side_label in ("arterial", "venous"):
            entries = [
                values
                for key, values in pressure_by_key.items()
                if key[0] == organoid_idx and key[1] == side_label
            ]
            medians[side_label] = {
                "p_darcy": _median([float(values["p_darcy"]) for values in entries]),
                "p_0d": _median([float(values["p_0d"]) for values in entries]),
            }
        for key, values in pressure_by_key.items():
            if key[0] != organoid_idx or key[1] not in {"arterial", "venous"}:
                continue
            side_label = str(key[1])
            median_darcy = float(medians[side_label]["p_darcy"])
            median_0d = float(medians[side_label]["p_0d"])
            if not (_finite(median_darcy) and _finite(median_0d)):
                continue
            if side_label == "arterial":
                raw_error = float(median_darcy - median_0d)
            else:
                raw_error = float(median_0d - median_darcy)
            scale = max(abs(median_darcy), abs(median_0d), floor)
            out[key] = {
                "raw_error": float(raw_error),
                "normalized_error": float(raw_error / scale),
                "scale": float(scale),
                "median_darcy": float(median_darcy),
                "median_0d": float(median_0d),
                "side_label": side_label,
            }
    return out


def _error_map_for_mode(
    pressure_by_key: dict[tuple[int, str, int], dict[str, float]],
    *,
    pressure_scale_floor: float,
    error_mode: str,
) -> dict[tuple[int, str, int], dict[str, float]]:
    mode = str(error_mode).strip().lower()
    if mode == "side_median":
        return _side_median_pressure_error_map(
            pressure_by_key,
            pressure_scale_floor=pressure_scale_floor,
        )
    if mode == "opposite_side_drop":
        return _opposite_side_referenced_error_map(
            pressure_by_key,
            pressure_scale_floor=pressure_scale_floor,
        )
    return {}


def _build_q_error_sensitivity_map(
    coupled_root: Path,
    current_run: int,
    history_window: int,
    min_logq_change: float,
    min_error_change: float,
    ema_alpha: float,
    active_only: bool,
    pressure_scale_floor: float = 1.0,
    opposite_side_referenced_error: bool = False,
    error_mode: str = "direct",
) -> dict[tuple[int, str, int], dict[str, float]]:
    prev_run = int(current_run) - 1
    if prev_run < 2:
        return {}

    # Do not seed the Jacobian estimate from run_1. The first updated run can
    # contain zero/placeholder terminal flows, which makes log(Q) secants blow
    # up and then freeze later updates.
    start_run = max(2, prev_run - max(int(history_window), 2) + 1)
    history: dict[tuple[int, str, int], list[dict[str, float]]] = {}
    for run_idx in range(start_run, prev_run + 1):
        run_dir = coupled_root / f"run_{run_idx}"
        summary_rows = _read_csv_rows(run_dir / "terminal_resistance_bc_summary.csv")
        result_rows = _read_csv_rows(run_dir / "post_darcy_terminal_convergence.csv")
        if not summary_rows or not result_rows:
            continue
        if any(
            str(row.get("hybrid_action", "")).startswith("initial_")
            or str(row.get("hybrid_action", "")).startswith("coarse_organoid_deltap_")
            for row in summary_rows
        ):
            # A coarse/common-mode phase changes the operating point without
            # providing an identifiable branch perturbation. It is a hard
            # history boundary: never pair a later probe with a pre-alignment
            # observation across this interval.
            history.clear()
            continue
        result_by_key: dict[tuple[int, str, int], dict[str, str]] = {}
        for row in result_rows:
            try:
                key = (int(row.get("organoid", 0)), str(row.get("side", "")).strip().lower(), int(row.get("branch_id", -1)))
            except Exception:
                continue
            result_by_key[key] = row
        pressure_by_key = {
            key: {
                "p_darcy": float(row.get("P_Darcy", float("nan"))),
                "p_0d": float(row.get("P_0D_target", float("nan"))),
            }
            for key, row in result_by_key.items()
            if _finite(row.get("P_Darcy")) and _finite(row.get("P_0D_target"))
        }
        referenced_errors = (
            _error_map_for_mode(
                pressure_by_key,
                pressure_scale_floor=pressure_scale_floor,
                error_mode=error_mode,
            )
            if str(error_mode).strip().lower() in {"opposite_side_drop", "side_median"}
            else {}
        )
        for row in summary_rows:
            try:
                key = (int(row.get("organoid", 0)), str(row.get("side", "")).strip().lower(), int(row.get("branch_id", -1)))
            except Exception:
                continue
            action = str(row.get("hybrid_action", "")).strip().lower()
            side_label = str(key[1])
            # During the two dedicated post-deltaP probes, only use the
            # side's own active probe row as an identification sample. The
            # inactive-baseline row from the opposite-side probe is not a
            # valid same-side sensitivity observation and poisons the slope.
            if action.startswith("post_deltap_"):
                expected_probe_action = f"post_deltap_{side_label}_probe"
                if action != expected_probe_action:
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
            if str(error_mode).strip().lower() in {"opposite_side_drop", "side_median"}:
                error_info = referenced_errors.get(key)
                if error_info is None:
                    continue
                error = float(error_info["normalized_error"])
            else:
                error = _signed_bounded_pressure_error(side_label, p_darcy, p_0d)
            active_raw = row.get("hybrid_alternate_side_active", "")
            if str(active_raw).strip():
                is_active = _safe_bool(active_raw)
            else:
                is_active = "alternate_damped" not in action
            history.setdefault(key, []).append({
                "run": float(run_idx),
                "logq": float(logq),
                "p_darcy": float(p_darcy),
                "p_0d": float(p_0d),
                "error": float(error),
                "abs_error": float(abs(error)),
                "active": float(1.0 if is_active else 0.0),
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
        if bool(active_only):
            pair_items = [
                item for item in items
                if float(item.get("active", 1.0)) >= 0.5
            ]
        else:
            pair_items = items
        for prev_item, cur_item in zip(pair_items[:-1], pair_items[1:]):
            dx = float(cur_item["logq"] - prev_item["logq"])
            de = float(cur_item["error"] - prev_item["error"])
            if not (_finite(dx) and _finite(de)):
                continue
            if abs(dx) <= sys.float_info.min:
                continue
            # Treat min_dx as denominator regularization, not a sample
            # rejection threshold.  This preserves the tiny late-stage moves
            # that made the reference controller contract smoothly without
            # allowing roundoff-sized flow changes to inflate the slope.
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


def _set_plan_q_target_magnitude(plan: dict[str, Any], target_magnitude: float) -> None:
    q_floor = float(plan["q_floor"])
    q_drive_mag = max(abs(float(plan["q_drive"])), q_floor)
    target_magnitude = max(abs(float(target_magnitude)), q_floor)
    plan["q_target"] = float(
        _supported_q_sign_qforr(str(plan["side_label"])) * target_magnitude
    )
    plan["delta_log"] = float(math.log10(target_magnitude / q_drive_mag))


def _shift_plan_target_by_delta(plan: dict[str, Any], delta_log: float, action_suffix: str) -> None:
    q_floor = float(plan["q_floor"])
    target_mag = max(abs(float(plan.get("q_target", 0.0))), q_floor)
    new_target_mag = max(target_mag * (10.0 ** float(delta_log)), q_floor)
    _set_plan_q_target_magnitude(plan, new_target_mag)
    plan["action"] = f"{plan['action']}_{action_suffix}"


def _restore_probe_baseline_on_plans(
    plans: list[dict[str, Any]],
    *,
    baseline_map: dict[tuple[int, str, int], float] | None = None,
    baseline_bc_map: dict[str, tuple[float, float]] | None = None,
    baseline_error_map: dict[tuple[int, str, int], float] | None = None,
    baseline_side_median_error_map: dict[tuple[int, str, int], float] | None = None,
) -> bool:
    restored_any = False
    for plan in plans:
        key = (
            int(plan.get("organoid_idx", -1)),
            str(plan.get("side_label", "")).strip().lower(),
            int(plan.get("branch_id", -1)),
        )
        baseline_flow = float((baseline_map or {}).get(key, float("nan")))
        if _finite(baseline_flow) and baseline_flow > sys.float_info.min:
            signed_flow = float(
                _supported_q_sign_qforr(str(plan.get("side_label", ""))) * baseline_flow
            )
            plan["q_drive"] = float(signed_flow)
            plan["q_target"] = float(signed_flow)
            plan["delta_log"] = 0.0
            plan["action"] = "post_probe_baseline"
            plan["hybrid_post_deltap_baseline_flow"] = float(baseline_flow)
            plan["hybrid_use_saved_baseline_for_sensitivity"] = True
            restored_any = True
        baseline_error = float((baseline_error_map or {}).get(key, float("nan")))
        if _finite(baseline_error):
            # The restored flow and the error consumed by the first secant
            # update must describe the same pre-probe operating point.
            plan["error"] = float(baseline_error)
            plan["rel_error"] = float(baseline_error)
            plan["hybrid_relative_pressure_error"] = float(baseline_error)
            plan["hybrid_post_probe_baseline_error"] = float(baseline_error)
        baseline_side_error = float(
            (baseline_side_median_error_map or {}).get(key, float("nan"))
        )
        if _finite(baseline_side_error):
            plan["hybrid_post_probe_baseline_side_median_error"] = float(
                baseline_side_error
            )
        baseline_bc = (baseline_bc_map or {}).get(str(plan.get("bc_name", "")))
        if baseline_bc is not None:
            plan["pd_seed"] = float(baseline_bc[1])
            plan["r_trial"] = float(baseline_bc[0])
            restored_any = True
    return restored_any


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


def _organoid_deltap_ratios_from_result_rows(
    rows: list[dict[str, str]],
) -> dict[int, float]:
    ratios: dict[int, float] = {}
    organoid_indices: set[int] = set()
    for row in rows:
        try:
            organoid_indices.add(int(row.get("organoid", 0)))
        except Exception:
            continue
    for organoid_idx in sorted(organoid_indices):
        side_rows = {
            side: [
                row for row in rows
                if int(row.get("organoid", 0)) == organoid_idx
                and str(row.get("side", "")).strip().lower() == side
            ]
            for side in ("arterial", "venous")
        }
        if not side_rows["arterial"] or not side_rows["venous"]:
            continue
        p_darcy_a = _median([float(row.get("P_Darcy", float("nan"))) for row in side_rows["arterial"]])
        p_darcy_v = _median([float(row.get("P_Darcy", float("nan"))) for row in side_rows["venous"]])
        p_0d_a = _median([float(row.get("P_0D_target", float("nan"))) for row in side_rows["arterial"]])
        p_0d_v = _median([float(row.get("P_0D_target", float("nan"))) for row in side_rows["venous"]])
        darcy_drop = abs(p_darcy_a - p_darcy_v)
        zero_d_drop = abs(p_0d_a - p_0d_v)
        if (
            _finite(darcy_drop)
            and _finite(zero_d_drop)
            and darcy_drop > sys.float_info.min
            and zero_d_drop > sys.float_info.min
        ):
            ratios[organoid_idx] = float(zero_d_drop / darcy_drop)
    return ratios


def _organoid_deltap_ratios_from_plans(
    plans: list[dict[str, Any]],
) -> dict[int, float]:
    ratios: dict[int, float] = {}
    for organoid_idx in sorted({int(plan.get("organoid_idx", -1)) for plan in plans}):
        organoid_plans = [
            plan for plan in plans
            if int(plan.get("organoid_idx", -1)) == organoid_idx
        ]
        by_side = {
            side: [
                plan for plan in organoid_plans
                if str(plan.get("side_label", "")).strip().lower() == side
            ]
            for side in ("arterial", "venous")
        }
        if not by_side["arterial"] or not by_side["venous"]:
            continue
        p_darcy_a = _median([float(plan.get("p_darcy", float("nan"))) for plan in by_side["arterial"]])
        p_darcy_v = _median([float(plan.get("p_darcy", float("nan"))) for plan in by_side["venous"]])
        p_0d_a = _median([float(plan.get("p_0d", float("nan"))) for plan in by_side["arterial"]])
        p_0d_v = _median([float(plan.get("p_0d", float("nan"))) for plan in by_side["venous"]])
        darcy_drop = abs(p_darcy_a - p_darcy_v)
        zero_d_drop = abs(p_0d_a - p_0d_v)
        if (
            _finite(darcy_drop)
            and _finite(zero_d_drop)
            and darcy_drop > sys.float_info.min
            and zero_d_drop > sys.float_info.min
        ):
            ratios[organoid_idx] = float(zero_d_drop / darcy_drop)
    return ratios


def _completed_deltap_in_band_streak(
    coupled_root: Path,
    *,
    current_run: int,
    first_run: int,
    expected_organoids: set[int],
    ratio_lower: float,
    ratio_upper: float,
) -> int:
    streak = 0
    for run_idx in range(int(current_run) - 1, int(first_run) - 1, -1):
        rows = _read_csv_rows(
            coupled_root / f"run_{run_idx}" / "post_darcy_terminal_convergence.csv"
        )
        if not rows:
            break
        ratios = _organoid_deltap_ratios_from_result_rows(rows)
        if (
            set(ratios) != set(expected_organoids)
            or any(
                ratio < float(ratio_lower) or ratio > float(ratio_upper)
                for ratio in ratios.values()
            )
        ):
            break
        streak += 1
    return int(streak)


def _post_deltap_probe_baseline_map(
    coupled_root: Path,
    *,
    current_run: int,
    first_run: int,
    first_probe_side: str,
) -> dict[tuple[int, str, int], float]:
    expected_action = f"post_deltap_{str(first_probe_side).strip().lower()}_probe"
    for run_idx in range(int(current_run) - 1, int(first_run) - 1, -1):
        rows = _read_csv_rows(
            coupled_root / f"run_{run_idx}" / "terminal_resistance_bc_summary.csv"
        )
        if not rows:
            continue
        if any(
            str(row.get("hybrid_action", "")) == expected_action
            for row in rows
        ):
            baseline: dict[tuple[int, str, int], float] = {}
            for row in rows:
                try:
                    key = (
                        int(row.get("organoid", 0)),
                        str(row.get("side", "")).strip().lower(),
                        int(row.get("branch_id", -1)),
                    )
                except Exception:
                    continue
                q_drive = row.get("hybrid_q_drive", row.get("Q_0D", ""))
                if _finite(q_drive) and abs(float(q_drive)) > sys.float_info.min:
                    baseline[key] = abs(float(q_drive))
            return baseline
        if any(
            str(row.get("hybrid_action", "")).startswith("coarse_organoid_deltap_")
            for row in rows
        ):
            break
    return {}


def _probe_baseline_state_path(coupled_root: Path) -> Path:
    return coupled_root / "hybrid_post_deltap_probe_baseline.json"


def _probe_baseline_error_map_from_entries(
    entries: list[dict[str, Any]],
    *,
    pressure_scale_floor: float,
    error_mode: str,
) -> dict[tuple[int, str, int], float]:
    pressure_by_key: dict[tuple[int, str, int], dict[str, float]] = {}
    for entry in entries:
        try:
            key = (
                int(entry["organoid"]),
                str(entry["side"]).strip().lower(),
                int(entry["branch_id"]),
            )
            p_darcy = float(entry["baseline_p_darcy"])
            p_0d = float(entry["baseline_p_0d"])
        except Exception:
            continue
        if _finite(p_darcy) and _finite(p_0d):
            pressure_by_key[key] = {
                "p_darcy": float(p_darcy),
                "p_0d": float(p_0d),
            }

    mode = str(error_mode).strip().lower()
    if mode in {"opposite_side_drop", "side_median"}:
        mapped = _error_map_for_mode(
            pressure_by_key,
            pressure_scale_floor=pressure_scale_floor,
            error_mode=mode,
        )
        return {
            key: float(info["normalized_error"])
            for key, info in mapped.items()
            if _finite(info.get("normalized_error"))
        }

    return {
        key: float(
            _signed_bounded_pressure_error(
                key[1], values["p_darcy"], values["p_0d"]
            )
        )
        for key, values in pressure_by_key.items()
    }


def _load_probe_baseline_error_map(
    coupled_root: Path,
    *,
    pressure_scale_floor: float,
    error_mode: str,
) -> dict[tuple[int, str, int], float]:
    path = _probe_baseline_state_path(coupled_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return _probe_baseline_error_map_from_entries(
        list(data.get("entries", [])),
        pressure_scale_floor=pressure_scale_floor,
        error_mode=error_mode,
    )


def _save_probe_baseline_state(
    coupled_root: Path,
    plans: list[dict[str, Any]],
    *,
    next_phase: int = 0,
    locked: bool = True,
    consumed_for_sensitivity: bool = False,
) -> None:
    entries: list[dict[str, Any]] = []
    for plan in plans:
        bc_values = dict(plan.get("bc", {}).get("bc_values", {}))
        resistance = bc_values.get("R", float("nan"))
        distal_pressure = bc_values.get("Pd", float("nan"))
        q_drive = abs(float(plan.get("q_drive", 0.0)))
        p_darcy = float(plan.get("p_darcy", float("nan")))
        p_0d = float(plan.get("p_0d", float("nan")))
        side_label = str(plan.get("side_label", "")).strip().lower()
        error = float(plan.get("error", float("nan")))
        if not _finite(error):
            error = (
                _signed_bounded_pressure_error(side_label, p_darcy, p_0d)
                if _finite(p_darcy) and _finite(p_0d)
                else float("nan")
            )
        if not (
            _finite(resistance)
            and float(resistance) > 0.0
            and _finite(distal_pressure)
            and _finite(q_drive)
            and q_drive > sys.float_info.min
        ):
            continue
        entries.append({
            "organoid": int(plan.get("organoid_idx", -1)),
            "side": str(plan.get("side_label", "")).strip().lower(),
            "branch_id": int(plan.get("branch_id", -1)),
            "bc_name": str(plan.get("bc_name", "")),
            "flow_magnitude": float(q_drive),
            "R": float(resistance),
            "Pd": float(distal_pressure),
            "baseline_error": float(error),
            "baseline_p_darcy": float(p_darcy),
            "baseline_p_0d": float(p_0d),
        })
    # Persist each definition explicitly for diagnostics. The sensitivity
    # builder also recomputes these from the saved pressures so older baseline
    # files receive the same consistent behavior.
    for mode, field in (
        ("direct", "baseline_error_direct"),
        ("opposite_side_drop", "baseline_error_opposite_side_drop"),
        ("side_median", "baseline_error_side_median"),
    ):
        error_map = _probe_baseline_error_map_from_entries(
            entries,
            pressure_scale_floor=1.0,
            error_mode=mode,
        )
        for entry in entries:
            key = (
                int(entry["organoid"]),
                str(entry["side"]).strip().lower(),
                int(entry["branch_id"]),
            )
            entry[field] = float(error_map.get(key, float("nan")))
    path = _probe_baseline_state_path(coupled_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "entries": entries,
                "next_phase": int(next_phase),
                "locked": bool(locked),
                "consumed_for_sensitivity": bool(consumed_for_sensitivity),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _load_probe_baseline_state(
    coupled_root: Path,
) -> tuple[
    dict[tuple[int, str, int], float],
    dict[str, tuple[float, float]],
    dict[str, Any],
]:
    path = _probe_baseline_state_path(coupled_root)
    if not path.exists():
        return {}, {}, {"next_phase": 0, "locked": False, "consumed_for_sensitivity": False}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}, {}, {"next_phase": 0, "locked": False, "consumed_for_sensitivity": False}
    flow_map: dict[tuple[int, str, int], float] = {}
    bc_map: dict[str, tuple[float, float]] = {}
    for entry in data.get("entries", []):
        try:
            key = (
                int(entry["organoid"]),
                str(entry["side"]).strip().lower(),
                int(entry["branch_id"]),
            )
            flow = float(entry["flow_magnitude"])
            resistance = float(entry["R"])
            distal_pressure = float(entry["Pd"])
            bc_name = str(entry["bc_name"])
        except Exception:
            continue
        if (
            _finite(flow)
            and flow > sys.float_info.min
            and _finite(resistance)
            and resistance > 0.0
            and _finite(distal_pressure)
            and bc_name
        ):
            flow_map[key] = flow
            bc_map[bc_name] = (resistance, distal_pressure)
    state = {
        "next_phase": int(data.get("next_phase", 0)),
        "locked": bool(data.get("locked", False)),
        "consumed_for_sensitivity": bool(data.get("consumed_for_sensitivity", False)),
    }
    return flow_map, bc_map, state


def _update_probe_baseline_state(
    coupled_root: Path,
    *,
    next_phase: int,
    locked: bool,
    consumed_for_sensitivity: bool | None = None,
) -> None:
    path = _probe_baseline_state_path(coupled_root)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text())
    except Exception:
        return
    data["next_phase"] = int(next_phase)
    data["locked"] = bool(locked)
    if consumed_for_sensitivity is not None:
        data["consumed_for_sensitivity"] = bool(consumed_for_sensitivity)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _build_first_post_probe_sensitivity_map(
    coupled_root: Path,
    *,
    current_run: int,
    first_run: int,
    min_logq_change: float,
    pressure_scale_floor: float = 1.0,
    opposite_side_referenced_error: bool = False,
    error_mode: str = "direct",
) -> dict[tuple[int, str, int], dict[str, float]]:
    path = _probe_baseline_state_path(coupled_root)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}

    entries = list(data.get("entries", []))
    baseline_errors = _probe_baseline_error_map_from_entries(
        entries,
        pressure_scale_floor=pressure_scale_floor,
        error_mode=error_mode,
    )
    baseline_points: dict[tuple[int, str, int], tuple[float, float]] = {}
    for entry in entries:
        try:
            key = (
                int(entry["organoid"]),
                str(entry["side"]).strip().lower(),
                int(entry["branch_id"]),
            )
            flow = float(entry["flow_magnitude"])
            error = float(baseline_errors.get(key, float("nan")))
        except Exception:
            continue
        if (
            _finite(flow)
            and flow > sys.float_info.min
            and _finite(error)
        ):
            baseline_points[key] = (
                float(math.log10(max(flow, sys.float_info.min))),
                float(error),
            )
    if not baseline_points:
        return {}

    probe_points: dict[tuple[int, str, int], tuple[float, float, float]] = {}
    for run_idx in range(int(current_run) - 1, int(first_run) - 1, -1):
        run_dir = coupled_root / f"run_{run_idx}"
        summary_rows = _read_csv_rows(run_dir / "terminal_resistance_bc_summary.csv")
        result_rows = _read_csv_rows(run_dir / "post_darcy_terminal_convergence.csv")
        if not summary_rows or not result_rows:
            continue
        result_by_key: dict[tuple[int, str, int], dict[str, str]] = {}
        for row in result_rows:
            try:
                key = (
                    int(row.get("organoid", 0)),
                    str(row.get("side", "")).strip().lower(),
                    int(row.get("branch_id", -1)),
                )
            except Exception:
                continue
            result_by_key[key] = row
        pressure_by_key = {
            key: {
                "p_darcy": float(row.get("P_Darcy", float("nan"))),
                "p_0d": float(row.get("P_0D_target", float("nan"))),
            }
            for key, row in result_by_key.items()
            if _finite(row.get("P_Darcy")) and _finite(row.get("P_0D_target"))
        }
        referenced_errors = (
            _error_map_for_mode(
                pressure_by_key,
                pressure_scale_floor=pressure_scale_floor,
                error_mode=error_mode,
            )
            if str(error_mode).strip().lower() in {"opposite_side_drop", "side_median"}
            else {}
        )
        for row in summary_rows:
            try:
                key = (
                    int(row.get("organoid", 0)),
                    str(row.get("side", "")).strip().lower(),
                    int(row.get("branch_id", -1)),
                )
            except Exception:
                continue
            side_label = str(key[1])
            expected_probe_action = f"post_deltap_{side_label}_probe"
            if str(row.get("hybrid_action", "")).strip().lower() != expected_probe_action:
                continue
            if key in probe_points or key not in baseline_points:
                continue
            # For the dedicated probe secant, use the commanded probe target
            # as the x-step, not the realized Q_0D. During the probe run the
            # solved 0D flow can still equal the saved baseline to machine
            # precision, which collapses dlogQ to zero and incorrectly drops
            # the branch from the first post-probe sensitivity map.
            q_mag = float("nan")
            target_raw = row.get("hybrid_target_flow_raw", "")
            if _finite(target_raw):
                q_mag = abs(float(target_raw))
            if not (_finite(q_mag) and q_mag > sys.float_info.min):
                q_mag = _q0d_magnitude_from_summary_row(row)
            result = result_by_key.get(key)
            if not _finite(q_mag) or result is None:
                continue
            p_darcy = float(result.get("P_Darcy", float("nan")))
            p_0d = float(result.get("P_0D_target", float("nan")))
            if not (_finite(p_darcy) and _finite(p_0d)):
                continue
            if str(error_mode).strip().lower() in {"opposite_side_drop", "side_median"}:
                error_info = referenced_errors.get(key)
                if error_info is None:
                    continue
                probe_error = float(error_info["normalized_error"])
            else:
                probe_error = float(_signed_bounded_pressure_error(side_label, p_darcy, p_0d))
            probe_points[key] = (
                float(math.log10(max(abs(float(q_mag)), sys.float_info.min))),
                float(probe_error),
                float(run_idx),
            )

    out: dict[tuple[int, str, int], dict[str, float]] = {}
    min_dx = max(float(min_logq_change), sys.float_info.min)
    for key, (probe_logq, probe_error, _run_idx) in probe_points.items():
        baseline = baseline_points.get(key)
        if baseline is None:
            continue
        base_logq, base_error = baseline
        dx = float(probe_logq - base_logq)
        de = float(probe_error - base_error)
        if not (_finite(dx) and _finite(de)):
            continue
        if abs(dx) <= sys.float_info.min:
            continue
        dx_eff = math.copysign(max(abs(dx), min_dx), dx)
        slope = de / dx_eff
        if not _finite(slope):
            continue
        out[key] = {
            "de_dlogq_abs_mean": float(abs(slope)),
            "de_dlogq_abs_ema": float(abs(slope)),
            "de_dlogq_signed": float(slope),
            "recent_de_dlogq_signed": float(slope),
            "sign_consistency": 1.0 if abs(slope) > 0.0 else 0.0,
            "sample_count": 1.0,
            "recent_improvement_rel": float("nan"),
            "median_improvement_rel": float("nan"),
            "oscillation_ratio": float("nan"),
        }
    return out


def _fine_runs_since_coarse_phase(
    coupled_root: Path,
    *,
    current_run: int,
    first_run: int,
) -> int:
    count = 0
    for run_idx in range(int(first_run), int(current_run)):
        rows = _read_csv_rows(
            coupled_root / f"run_{run_idx}" / "terminal_resistance_bc_summary.csv"
        )
        if not rows:
            continue
        actions = {str(row.get("hybrid_action", "")) for row in rows}
        if any(
            action.startswith("initial_")
            or action.startswith("coarse_organoid_deltap_")
            for action in actions
        ):
            count = 0
        else:
            count += 1
    return int(count)


def _apply_run2_arterial_venous_flow_equalization(
    plans: list[dict[str, Any]],
) -> bool:
    """Match terminal totals once without changing within-tree distributions."""
    applied = False
    organoid_indices = sorted({int(plan.get("organoid_idx", -1)) for plan in plans})
    for organoid_idx in organoid_indices:
        organoid_plans = [
            plan for plan in plans
            if int(plan.get("organoid_idx", -1)) == organoid_idx
        ]
        arterial = [
            plan for plan in organoid_plans
            if str(plan.get("side_label", "")).strip().lower() == "arterial"
        ]
        venous = [
            plan for plan in organoid_plans
            if str(plan.get("side_label", "")).strip().lower() == "venous"
        ]
        arterial_total = sum(abs(float(plan.get("q_drive", 0.0))) for plan in arterial)
        venous_total = sum(abs(float(plan.get("q_drive", 0.0))) for plan in venous)
        if (
            not arterial
            or not venous
            or not _finite(arterial_total)
            or not _finite(venous_total)
            or arterial_total <= sys.float_info.min
            or venous_total <= sys.float_info.min
        ):
            continue

        arterial_delta = math.log10(venous_total / arterial_total)
        for plan in arterial:
            _set_plan_q_target_from_delta(plan, arterial_delta)
            plan["action"] = "initial_arterial_total_equalization"
            plan["hybrid_initial_side_total_ratio"] = float(
                arterial_total / venous_total
            )
            plan["hybrid_alternate_side_active"] = True
            plan["hybrid_alternate_side_scale"] = 1.0
        for plan in venous:
            _set_plan_q_target_from_delta(plan, 0.0)
            plan["action"] = "initial_venous_hold"
            plan["hybrid_initial_side_total_ratio"] = float(
                arterial_total / venous_total
            )
            plan["hybrid_alternate_side_active"] = True
            plan["hybrid_alternate_side_scale"] = 1.0
        applied = True
    return applied


def _apply_coarse_organoid_deltap_scaling(
    plans: list[dict[str, Any]],
    *,
    gamma: float,
    max_log_shift: float,
    ratio_lower: float,
    ratio_upper: float,
    release_ready: bool = True,
) -> bool:
    """Align the 0D/Darcy pressure-drop scale without redistributing flow."""
    gamma = max(float(gamma), 0.0)
    max_log_shift = max(float(max_log_shift), 0.0)
    ratio_lower = max(float(ratio_lower), sys.float_info.min)
    ratio_upper = max(float(ratio_upper), ratio_lower)

    corrections: dict[int, float] = {}
    ratios: dict[int, float] = {}
    organoid_indices = sorted({int(plan.get("organoid_idx", -1)) for plan in plans})
    for organoid_idx in organoid_indices:
        organoid_plans = [
            plan for plan in plans
            if int(plan.get("organoid_idx", -1)) == organoid_idx
        ]
        arterial = [
            plan for plan in organoid_plans
            if str(plan.get("side_label", "")).strip().lower() == "arterial"
        ]
        venous = [
            plan for plan in organoid_plans
            if str(plan.get("side_label", "")).strip().lower() == "venous"
        ]
        if not arterial or not venous:
            continue

        p_darcy_a = _median([float(plan.get("p_darcy", float("nan"))) for plan in arterial])
        p_darcy_v = _median([float(plan.get("p_darcy", float("nan"))) for plan in venous])
        p_0d_a = _median([float(plan.get("p_0d", float("nan"))) for plan in arterial])
        p_0d_v = _median([float(plan.get("p_0d", float("nan"))) for plan in venous])
        darcy_drop = abs(p_darcy_a - p_darcy_v)
        zero_d_drop = abs(p_0d_a - p_0d_v)
        if not (
            _finite(darcy_drop)
            and _finite(zero_d_drop)
            and darcy_drop > sys.float_info.min
            and zero_d_drop > sys.float_info.min
        ):
            continue

        ratio = zero_d_drop / darcy_drop
        ratios[organoid_idx] = float(ratio)
        if ratio < ratio_lower or ratio > ratio_upper or not bool(release_ready):
            raw_delta = gamma * math.log10(ratio)
            delta = max(-max_log_shift, min(max_log_shift, raw_delta))
            if abs(delta) > 1.0e-12:
                corrections[organoid_idx] = float(delta)

    if not corrections and bool(release_ready):
        return False

    # Keep the controller phase synchronized across organoids. An organoid
    # already inside the band holds while another organoid is being aligned.
    for plan in plans:
        organoid_idx = int(plan.get("organoid_idx", -1))
        delta = float(corrections.get(organoid_idx, 0.0))
        ratio = float(ratios.get(organoid_idx, float("nan")))
        _set_plan_q_target_from_delta(plan, delta)
        plan["action"] = (
            "coarse_organoid_deltap_scaling"
            if organoid_idx in corrections
            else "coarse_organoid_deltap_hold"
        )
        plan["hybrid_organoid_deltap_ratio"] = float(ratio)
        plan["hybrid_alternate_side_active"] = True
        plan["hybrid_alternate_side_scale"] = 1.0
    return True


def _apply_post_deltap_probe_phase(
    plans: list[dict[str, Any]],
    *,
    phase: int,
    start_side: str,
    max_log_step: float,
    baseline_map: dict[tuple[int, str, int], float] | None = None,
) -> bool:
    """Generate exactly two baseline-referenced probe runs."""
    if int(phase) < 0 or int(phase) > 1:
        return False
    max_log_step = max(float(max_log_step), 0.0)
    first_side = str(start_side).strip().lower()
    second_side = "arterial" if first_side == "venous" else "venous"
    active_side = first_side if int(phase) == 0 else second_side

    baseline_by_plan: dict[int, float] = {}
    for plan in plans:
        key = (
            int(plan.get("organoid_idx", -1)),
            str(plan.get("side_label", "")).strip().lower(),
            int(plan.get("branch_id", -1)),
        )
        if int(phase) == 0:
            baseline = abs(float(plan.get("q_drive", 0.0)))
        else:
            baseline = float((baseline_map or {}).get(key, float("nan")))
        if not _finite(baseline) or baseline <= sys.float_info.min:
            return False
        baseline_by_plan[id(plan)] = float(baseline)

    for plan in plans:
        side_label = str(plan.get("side_label", "")).strip().lower()
        baseline = float(baseline_by_plan[id(plan)])
        plan["hybrid_post_deltap_baseline_flow"] = float(baseline)
        if side_label == active_side:
            old_delta = float(plan.get("delta_log", 0.0))
            probe_delta = max(-max_log_step, min(max_log_step, old_delta))
            _set_plan_q_target_magnitude(
                plan, baseline * (10.0 ** float(probe_delta))
            )
            plan["action"] = f"post_deltap_{active_side}_probe"
            plan["hybrid_alternate_side_active"] = True
            plan["hybrid_alternate_side_scale"] = 1.0
        else:
            _set_plan_q_target_magnitude(plan, baseline)
            plan["action"] = f"post_deltap_{active_side}_probe_inactive_baseline"
            plan["hybrid_alternate_side_active"] = False
            plan["hybrid_alternate_side_scale"] = 0.0
        plan["hybrid_post_deltap_probe_phase"] = int(phase)
    return True


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
        # the current mismatch sign.  Do not apply the per-terminal trust
        # region cap here: alternating-side damping and side common-mode
        # control can still change the composed move.  The trust-region cap is
        # applied later to the final terminal delta.
        delta_mag = local_gain * abs(float(error)) / float(slope_mag)
        delta_secant = -math.copysign(delta_mag, float(error))
        if not _finite(delta_secant):
            continue
        new_delta = float(delta_secant)
        plan["hybrid_branch_trust_region_cap"] = float(adaptive_cap)
        plan["hybrid_q_error_sensitivity_delta_uncapped"] = float(new_delta)
        plan["hybrid_sensitivity_delta_uncapped"] = float(new_delta)
        plan["hybrid_q_error_sensitivity_slope"] = float(slope_mag)
        plan["hybrid_q_error_sensitivity_sign_consistency"] = float(sign_consistency)
        plan["hybrid_q_error_sensitivity_sample_count"] = float(sample_count)
        plan["hybrid_q_error_sensitivity_source"] = str(sensitivity_source)
        plan["hybrid_sensitivity_slope"] = float(slope_mag)
        plan["hybrid_sensitivity_sign_consistency"] = float(sign_consistency)
        plan["hybrid_sensitivity_sample_count"] = float(sample_count)
        if abs(new_delta) <= 1.0e-12:
            plan["hybrid_q_error_sensitivity_applied"] = False
            plan["hybrid_q_error_sensitivity_status"] = "evaluated_zero_move"
            plan["hybrid_q_error_sensitivity_multiplier"] = float("nan")
            plan["hybrid_sensitivity_applied"] = False
            plan["hybrid_sensitivity_status"] = "evaluated_zero_move"
            plan["hybrid_sensitivity_multiplier"] = float("nan")
            continue
        if abs(new_delta - old_delta) <= 1.0e-12:
            plan["hybrid_q_error_sensitivity_applied"] = False
            plan["hybrid_q_error_sensitivity_status"] = "evaluated_same_as_base"
            plan["hybrid_q_error_sensitivity_multiplier"] = 1.0
            plan["hybrid_sensitivity_applied"] = False
            plan["hybrid_sensitivity_status"] = "evaluated_same_as_base"
            plan["hybrid_sensitivity_multiplier"] = 1.0
            continue

        baseline_flow = float(plan.get("hybrid_post_deltap_baseline_flow", float("nan")))
        if bool(plan.get("hybrid_use_saved_baseline_for_sensitivity", False)) and _finite(baseline_flow):
            target_magnitude = max(
                abs(float(baseline_flow)) * (10.0 ** float(new_delta)),
                float(plan.get("q_floor", sys.float_info.min)),
            )
            _set_plan_q_target_magnitude(plan, target_magnitude)
            plan["action"] = f"{plan['action']}_branch_secant"
        else:
            _retarget_plan_q(plan, float(new_delta), "branch_secant")
        plan["hybrid_sensitivity_applied"] = True
        plan["hybrid_sensitivity_status"] = "evaluated_changed_move"
        plan["hybrid_sensitivity_multiplier"] = (
            float(abs(new_delta) / abs(old_delta))
            if abs(old_delta) > 1.0e-12
            else float("nan")
        )
        plan["hybrid_sensitivity_stable_r_count"] = float("nan")
        plan["hybrid_q_error_sensitivity_applied"] = True
        plan["hybrid_q_error_sensitivity_status"] = "evaluated_changed_move"
        plan["hybrid_q_error_sensitivity_multiplier"] = (
            float(abs(new_delta) / abs(old_delta))
            if abs(old_delta) > 1.0e-12
            else float("nan")
        )


def _project_branch_secant_to_side_redistribution(
    plans: list[dict[str, Any]],
) -> None:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for plan in plans:
        side_label = str(plan.get("side_label", "")).strip().lower()
        organoid_idx = int(plan.get("organoid_idx", -1))
        grouped.setdefault((organoid_idx, side_label), []).append(plan)

    for (_organoid_idx, _side_label), side_plans in grouped.items():
        usable: list[tuple[dict[str, Any], float, float]] = []
        for plan in side_plans:
            delta = float(plan.get("delta_log", float("nan")))
            raw_weight = abs(float(plan.get("q_drive", 0.0)))
            if not (_finite(delta) and _finite(raw_weight) and raw_weight > sys.float_info.min):
                continue
            usable.append((plan, delta, raw_weight))
        if not usable:
            continue

        # Use clipped-flow weights for the projection common mode.  Raw
        # flow-weighted projection can let one runaway high-flow venous
        # terminal dominate the side common-mode estimate and subtract away
        # almost all of its own corrective branch move.  Clipping at the side
        # median preserves some flow awareness without giving the current
        # dominant branch veto power over branch-specific redistribution.
        positive_weights = sorted(weight for _plan, _delta, weight in usable)
        clip_weight = float(positive_weights[len(positive_weights) // 2])
        weighted_sum = 0.0
        weight_total = 0.0
        clipped_usable: list[tuple[dict[str, Any], float, float, float]] = []
        for plan, delta, raw_weight in usable:
            projection_weight = min(float(raw_weight), float(clip_weight))
            if not (_finite(projection_weight) and projection_weight > sys.float_info.min):
                projection_weight = 1.0
            clipped_usable.append((plan, delta, raw_weight, projection_weight))
            weighted_sum += projection_weight * delta
            weight_total += projection_weight
        if weight_total <= sys.float_info.min:
            continue
        common_mode_delta = float(weighted_sum / weight_total)
        for plan, delta, raw_weight, projection_weight in clipped_usable:
            redistribution_delta = float(delta - common_mode_delta)
            if abs(redistribution_delta - delta) > 1.0e-12:
                _set_plan_q_target_from_delta(plan, redistribution_delta)
            plan["hybrid_branch_total_pre_projection_delta"] = float(delta)
            plan["hybrid_branch_common_mode_component"] = float(common_mode_delta)
            plan["hybrid_branch_redistribution_delta"] = float(redistribution_delta)
            plan["hybrid_branch_projection_weight_raw"] = float(raw_weight)
            plan["hybrid_branch_projection_weight"] = float(projection_weight)
            plan["hybrid_branch_projection_weight_clip"] = float(clip_weight)
            plan["hybrid_branch_projection_weight_mode"] = "flow_clipped_to_side_median"


def _mark_branch_secant_projection_disabled(plans: list[dict[str, Any]]) -> None:
    """Record that branch sensitivity was used without side redistribution projection."""

    for plan in plans:
        delta = float(plan.get("delta_log", float("nan")))
        if not _finite(delta):
            continue
        plan["hybrid_branch_total_pre_projection_delta"] = float(delta)
        plan["hybrid_branch_common_mode_component"] = 0.0
        plan["hybrid_branch_redistribution_delta"] = float(delta)
        plan["hybrid_branch_projection_weight_raw"] = float("nan")
        plan["hybrid_branch_projection_weight"] = float("nan")
        plan["hybrid_branch_projection_weight_clip"] = float("nan")
        plan["hybrid_branch_projection_weight_mode"] = "disabled"


def _apply_final_trust_region_caps(plans: list[dict[str, Any]]) -> None:
    """Clip the final composed terminal move to the terminal trust region.

    The hybrid controller composes several log-flow corrections: branch
    sensitivity, alternating-side damping, and optional side-gauge common mode.
    The adaptive cap should bound the final terminal move that is actually
    handed to the algebraic 0D realization, not an intermediate branch-only
    correction.
    """

    for plan in plans:
        raw_delta = float(plan.get("delta_log", float("nan")))
        cap = float(plan.get("adaptive_cap", float("nan")))
        if not _finite(raw_delta):
            continue
        if not (_finite(cap) and cap > 0.0):
            plan["hybrid_final_uncapped_delta_log10_q"] = float(raw_delta)
            plan["hybrid_final_cap_delta_log10_q"] = float(raw_delta)
            plan["hybrid_final_cap_applied"] = False
            continue

        cap = abs(float(cap))
        clipped_delta = max(-cap, min(cap, float(raw_delta)))
        plan["hybrid_branch_trust_region_cap"] = float(cap)
        plan["hybrid_final_uncapped_delta_log10_q"] = float(raw_delta)
        plan["hybrid_final_cap_delta_log10_q"] = float(clipped_delta)
        plan["hybrid_final_cap_applied"] = bool(
            abs(clipped_delta - raw_delta) > 1.0e-12
        )
        if abs(clipped_delta - raw_delta) > 1.0e-12:
            _set_plan_q_target_from_delta(plan, float(clipped_delta))
            plan["action"] = f"{plan['action']}_final_cap"


def _apply_opposite_side_referenced_error_controller(
    plans: list[dict[str, Any]],
    *,
    config: argparse.Namespace,
    retarget: bool,
) -> None:
    pressure_by_key = {
        (
            int(plan.get("organoid_idx", -1)),
            str(plan.get("side_label", "")).strip().lower(),
            int(plan.get("branch_id", -1)),
        ): {
            "p_darcy": float(plan.get("p_darcy", float("nan"))),
            "p_0d": float(plan.get("p_0d", float("nan"))),
        }
        for plan in plans
        if _finite(plan.get("p_darcy")) and _finite(plan.get("p_0d"))
    }
    pressure_drop_error_map = _opposite_side_referenced_error_map(
        pressure_by_key,
        pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
    )
    error_map = _side_median_pressure_error_map(
        pressure_by_key,
        pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
    )
    for plan in plans:
        side_label = str(plan.get("side_label", "")).strip().lower()
        key = (
            int(plan.get("organoid_idx", -1)),
            side_label,
            int(plan.get("branch_id", -1)),
        )
        info = error_map.get(key)
        pd_info = pressure_drop_error_map.get(key)
        if info is None:
            continue
        rel_error = float(pd_info["normalized_error"]) if pd_info is not None else float("nan")
        plan["error"] = float(rel_error)
        plan["rel_error"] = float(rel_error)
        plan["hybrid_pressure_drop_error_raw"] = float(
            pd_info["raw_error"] if pd_info is not None else float("nan")
        )
        plan["hybrid_pressure_drop_error_normalized"] = float(
            pd_info["normalized_error"] if pd_info is not None else float("nan")
        )
        plan["hybrid_pressure_drop_error_scale"] = float(
            pd_info["scale"] if pd_info is not None else float("nan")
        )
        plan["hybrid_darcy_pressure_drop"] = float(
            pd_info["darcy_drop"] if pd_info is not None else float("nan")
        )
        plan["hybrid_0d_pressure_drop"] = float(
            pd_info["zero_d_drop"] if pd_info is not None else float("nan")
        )
        plan["hybrid_opposite_darcy_median"] = float(
            pd_info["opposite_darcy_median"] if pd_info is not None else float("nan")
        )
        plan["hybrid_opposite_0d_median"] = float(
            pd_info["opposite_0d_median"] if pd_info is not None else float("nan")
        )
        plan["hybrid_side_median_pressure_error_raw"] = float(info["raw_error"])
        plan["hybrid_side_median_pressure_error_normalized"] = float(info["normalized_error"])
        plan["hybrid_side_median_pressure_error_scale"] = float(info["scale"])
        plan["hybrid_side_median_darcy"] = float(info["median_darcy"])
        plan["hybrid_side_median_0d"] = float(info["median_0d"])
        plan["hybrid_combined_pressure_error"] = float(rel_error)
        if not retarget:
            continue

        if side_label == "venous":
            base_cap = float(config.hybrid_venous_max_log_step)
            moderate_cap = float(config.hybrid_venous_moderate_log_step)
            large_cap = float(config.hybrid_venous_large_log_step)
            extreme_cap = float(config.hybrid_venous_extreme_log_step)
        else:
            base_cap = float(config.hybrid_max_log_step)
            moderate_cap = float(config.hybrid_moderate_log_step)
            large_cap = float(config.hybrid_large_log_step)
            extreme_cap = float(config.hybrid_extreme_log_step)
        magnitude = abs(rel_error)
        if magnitude >= 1.0:
            adaptive_cap = max(base_cap, extreme_cap)
        elif magnitude >= 0.5:
            adaptive_cap = max(base_cap, large_cap)
        elif magnitude >= 0.1:
            adaptive_cap = max(base_cap, moderate_cap)
        else:
            adaptive_cap = base_cap
        delta = max(
            -adaptive_cap,
            min(adaptive_cap, -float(config.hybrid_log_gain) * rel_error),
        )
        _set_plan_q_target_from_delta(plan, delta)
        plan["adaptive_cap"] = float(adaptive_cap)
        if delta < -1.0e-12:
            plan["action"] = "pressure_drop_back_off"
        elif delta > 1.0e-12:
            plan["action"] = "pressure_drop_open_up"
        else:
            plan["action"] = "pressure_drop_hold"


def _project_side_gauge_common_mode_sensitivity(
    *,
    organoid_idx: int,
    side_label: str,
    side_plans: list[dict[str, Any]],
    q_error_sensitivity_map: dict[tuple[int, str, int], dict[str, float]] | None,
    q_error_sensitivity_fallback_map: dict[tuple[int | None, str], dict[str, float]] | None,
) -> tuple[float, str, float]:
    branch_slopes: list[float] = []
    for plan in side_plans:
        slope = float("nan")
        if q_error_sensitivity_map:
            key = (
                int(organoid_idx),
                str(side_label),
                int(plan.get("branch_id", -1)),
            )
            info = q_error_sensitivity_map.get(key)
            if info:
                slope = info.get("de_dlogq_abs_ema", info.get("de_dlogq_abs_mean", float("nan")))
        if not _finite(slope):
            slope = plan.get("hybrid_q_error_sensitivity_slope", float("nan"))
        if _finite(slope):
            branch_slopes.append(abs(float(slope)))
    if branch_slopes:
        branch_slopes = sorted(branch_slopes)
        return (
            float(branch_slopes[len(branch_slopes) // 2]),
            "branch_sensitivity_median",
            float(len(branch_slopes)),
        )

    if q_error_sensitivity_fallback_map:
        for key, source in (
            ((int(organoid_idx), str(side_label)), "organoid_side_fallback"),
            ((None, str(side_label)), "side_fallback"),
            ((None, "both"), "global_fallback"),
        ):
            info = q_error_sensitivity_fallback_map.get(key)
            if not info:
                continue
            slope = info.get("de_dlogq_abs_ema", info.get("de_dlogq_abs_mean", float("nan")))
            if _finite(slope):
                sample_count = info.get("sample_count", float("nan"))
                return (
                    abs(float(slope)),
                    str(source),
                    float(sample_count) if _finite(sample_count) else 0.0,
                )

    return float("nan"), "unavailable", 0.0


def _apply_side_gauge_common_mode_control(
    plans: list[dict[str, Any]],
    *,
    config: argparse.Namespace,
    q_error_sensitivity_map: dict[tuple[int, str, int], dict[str, float]] | None = None,
    q_error_sensitivity_fallback_map: dict[tuple[int | None, str], dict[str, float]] | None = None,
) -> None:
    if not bool(config.hybrid_side_gauge_common_mode_control):
        return
    band = max(float(config.hybrid_side_gauge_common_mode_band), 0.0)
    band_gate_enabled = bool(
        getattr(config, "hybrid_side_gauge_common_mode_band_gate", True)
    )
    max_step = max(float(config.hybrid_side_gauge_common_mode_max_log_step), 0.0)
    gains = {
        "arterial": max(float(config.hybrid_arterial_side_gauge_common_mode_gain), 0.0),
        "venous": max(float(config.hybrid_venous_side_gauge_common_mode_gain), 0.0),
    }
    organoid_ids = sorted({int(plan.get("organoid_idx", -1)) for plan in plans})
    for organoid_idx in organoid_ids:
        organoid_plans = [
            plan for plan in plans
            if int(plan.get("organoid_idx", -1)) == organoid_idx
        ]
        if not organoid_plans:
            continue
        branch_errors = [
            abs(float(plan.get("hybrid_pressure_drop_error_normalized", float("nan"))))
            for plan in organoid_plans
            if _finite(plan.get("hybrid_pressure_drop_error_normalized"))
        ]
        if band_gate_enabled and (
            len(branch_errors) != len(organoid_plans)
            or not branch_errors
            or _median(branch_errors) > band
        ):
            continue

        by_side = {
            side: [
                plan for plan in organoid_plans
                if str(plan.get("side_label", "")).strip().lower() == side
            ]
            for side in ("arterial", "venous")
        }
        for side_label, side_plans in by_side.items():
            if not side_plans:
                continue
            median_darcy = _median([float(plan.get("p_darcy", float("nan"))) for plan in side_plans])
            median_0d = _median([float(plan.get("p_0d", float("nan"))) for plan in side_plans])
            if not (_finite(median_darcy) and _finite(median_0d)):
                continue
            scale = max(abs(median_darcy), abs(median_0d), 1.0)
            if side_label == "arterial":
                gauge_error = float((median_darcy - median_0d) / scale)
            else:
                gauge_error = float((median_0d - median_darcy) / scale)
            saved_baseline_errors = [
                float(plan.get(
                    "hybrid_post_probe_baseline_side_median_error",
                    float("nan"),
                ))
                for plan in side_plans
                if bool(plan.get("hybrid_use_saved_baseline_for_sensitivity", False))
                and _finite(plan.get("hybrid_post_probe_baseline_side_median_error"))
            ]
            if len(saved_baseline_errors) == len(side_plans) and saved_baseline_errors:
                # On the first post-probe update, the side-total correction is
                # also applied to the restored baseline flow. Use the matching
                # saved baseline error instead of the final probe observation.
                gauge_error = float(_median(saved_baseline_errors))
            projected_slope, slope_source, slope_samples = _project_side_gauge_common_mode_sensitivity(
                organoid_idx=int(organoid_idx),
                side_label=str(side_label),
                side_plans=side_plans,
                q_error_sensitivity_map=q_error_sensitivity_map,
                q_error_sensitivity_fallback_map=q_error_sensitivity_fallback_map,
            )
            if _finite(projected_slope):
                raw_delta = -gains[side_label] * gauge_error / max(
                    abs(float(projected_slope)), sys.float_info.min
                )
            else:
                raw_delta = -gains[side_label] * gauge_error
            delta = max(-max_step, min(max_step, raw_delta))
            for plan in side_plans:
                plan["hybrid_side_gauge_common_mode_error"] = float(gauge_error)
                plan["hybrid_side_gauge_common_mode_delta"] = float(delta)
                plan["hybrid_side_gauge_common_mode_active"] = bool(abs(delta) > 1.0e-12)
                plan["hybrid_side_gauge_common_mode_slope"] = float(projected_slope)
                plan["hybrid_side_gauge_common_mode_slope_source"] = str(slope_source)
                plan["hybrid_side_gauge_common_mode_slope_samples"] = float(slope_samples)
                if abs(delta) <= 1.0e-12:
                    continue
                if delta < 0.0:
                    suffix = "side_gauge_common_mode_back_off"
                else:
                    suffix = "side_gauge_common_mode_open_up"
                _shift_plan_target_by_delta(plan, delta, suffix)


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
        "side_median_q_error_sensitivity_map": {},
        "side_median_q_error_sensitivity_fallback_map": {},
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
                and int(current_run_idx) >= max(start_run, int(config.hybrid_q_sensitivity_start_run))
            ):
                pending_state["q_error_sensitivity_map"] = _build_q_error_sensitivity_map(
                    coupled_root=Path(str(config._coupled_root)).expanduser().resolve(),
                    current_run=int(current_run_idx),
                    history_window=int(config.hybrid_q_sensitivity_history_window),
                    min_logq_change=float(config.hybrid_q_sensitivity_min_logq_change),
                    min_error_change=float(config.hybrid_q_sensitivity_min_error_change),
                    ema_alpha=float(config.hybrid_q_sensitivity_ema_alpha),
                    active_only=bool(config.hybrid_q_sensitivity_active_only),
                    pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
                    opposite_side_referenced_error=bool(config.hybrid_opposite_side_referenced_error),
                    error_mode=(
                        "opposite_side_drop"
                        if bool(config.hybrid_opposite_side_referenced_error)
                        else "direct"
                    ),
                )
                pending_state["q_error_sensitivity_fallback_map"] = _build_q_error_sensitivity_fallbacks(
                    pending_state["q_error_sensitivity_map"]
                )
                pending_state["side_median_q_error_sensitivity_map"] = _build_q_error_sensitivity_map(
                    coupled_root=Path(str(config._coupled_root)).expanduser().resolve(),
                    current_run=int(current_run_idx),
                    history_window=int(config.hybrid_q_sensitivity_history_window),
                    min_logq_change=float(config.hybrid_q_sensitivity_min_logq_change),
                    min_error_change=float(config.hybrid_q_sensitivity_min_error_change),
                    ema_alpha=float(config.hybrid_q_sensitivity_ema_alpha),
                    active_only=bool(config.hybrid_q_sensitivity_active_only),
                    pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
                    opposite_side_referenced_error=True,
                    error_mode="side_median",
                )
                pending_state["side_median_q_error_sensitivity_fallback_map"] = _build_q_error_sensitivity_fallbacks(
                    pending_state["side_median_q_error_sensitivity_map"]
                )
            else:
                pending_state["q_error_sensitivity_map"] = {}
                pending_state["q_error_sensitivity_fallback_map"] = {}
                pending_state["side_median_q_error_sensitivity_map"] = {}
                pending_state["side_median_q_error_sensitivity_fallback_map"] = {}

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
                "p_darcy": float(p_darcy),
                "p_0d": float(p_0d),
            })
            pending_state["row_refs"].append(row)

        if int(organoid_idx) != int(getattr(config, "_n_organoids", organoid_idx)):
            return rows

        all_plans = list(pending_state["plans"])
        pending_state["plans"] = []
        if not all_plans:
            return rows

        if bool(config.hybrid_opposite_side_referenced_error):
            _apply_opposite_side_referenced_error_controller(
                all_plans,
                config=config,
                retarget=False,
            )

        coupled_root = Path(str(config._coupled_root)).expanduser().resolve()
        probe_baseline_map, probe_baseline_bc_map, probe_state = _load_probe_baseline_state(
            coupled_root
        )
        probe_cycle_locked = bool(probe_state.get("locked", False))
        probe_next_phase = int(probe_state.get("next_phase", 0))
        probe_baseline_consumed = bool(
            probe_state.get("consumed_for_sensitivity", False)
        )
        ready_for_post_probe_sensitivity = (
            bool(config.hybrid_organoid_deltap_scaling)
            and bool(config.hybrid_post_deltap_probes)
            and not probe_cycle_locked
            and not probe_baseline_consumed
            and int(probe_next_phase) >= 2
            and bool(probe_baseline_map)
        )
        # The coarse organoid delta-P alignment is intended to be a one-time
        # startup phase. Once the saved post-alignment probe baseline exists,
        # never re-enter coarse scaling later in the run history even if the
        # measured ratio drifts again; the final controller should recover
        # from there via the restored drop-error moves and sensitivity stage.
        coarse_phase_completed = (
            bool(probe_cycle_locked)
            or bool(probe_baseline_consumed)
            or int(probe_next_phase) > 0
            or bool(probe_baseline_map)
        )

        run2_equalization = (
            bool(config.hybrid_run2_equalize_arterial_to_venous)
            and int(current_run_idx) == 2
            and _apply_run2_arterial_venous_flow_equalization(all_plans)
        )
        expected_organoids = {
            int(plan.get("organoid_idx", -1)) for plan in all_plans
        }
        # The current observation exists in all_plans before its post-Darcy
        # CSV is written. Read the older completed files only, then append the
        # current in-memory observation so confirmation does not cost a
        # repeated no-move solve.
        historical_in_band_streak = _completed_deltap_in_band_streak(
            coupled_root,
            current_run=int(current_run_idx) - 1,
            first_run=int(start_run),
            expected_organoids=expected_organoids,
            ratio_lower=float(config.hybrid_organoid_deltap_ratio_lower),
            ratio_upper=float(config.hybrid_organoid_deltap_ratio_upper),
        )
        current_deltap_ratios = _organoid_deltap_ratios_from_plans(all_plans)
        current_in_band = (
            set(current_deltap_ratios) == expected_organoids
            and all(
                float(config.hybrid_organoid_deltap_ratio_lower) <= ratio
                <= float(config.hybrid_organoid_deltap_ratio_upper)
                for ratio in current_deltap_ratios.values()
            )
        )
        in_band_streak = (
            int(historical_in_band_streak) + 1 if current_in_band else 0
        )
        required_in_band_runs = max(
            int(config.hybrid_organoid_deltap_min_in_band_runs), 1
        )
        baseline_restored_for_sensitivity = False
        if not run2_equalization and ready_for_post_probe_sensitivity:
            primary_error_mode = (
                "opposite_side_drop"
                if bool(config.hybrid_opposite_side_referenced_error)
                else "direct"
            )
            baseline_primary_error_map = _load_probe_baseline_error_map(
                coupled_root,
                pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
                error_mode=primary_error_mode,
            )
            baseline_side_median_error_map = _load_probe_baseline_error_map(
                coupled_root,
                pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
                error_mode="side_median",
            )
            baseline_restored_for_sensitivity = _restore_probe_baseline_on_plans(
                all_plans,
                baseline_map=probe_baseline_map,
                baseline_bc_map=probe_baseline_bc_map,
                baseline_error_map=baseline_primary_error_map,
                baseline_side_median_error_map=baseline_side_median_error_map,
            )
            if baseline_restored_for_sensitivity:
                ready_for_post_probe_sensitivity = True
        coarse_deltap_scaling = False
        if (
            not run2_equalization
            and not probe_cycle_locked
            and not baseline_restored_for_sensitivity
            and not coarse_phase_completed
        ):
            coarse_deltap_scaling = bool(
                config.hybrid_organoid_deltap_scaling
            ) and _apply_coarse_organoid_deltap_scaling(
                all_plans,
                gamma=float(config.hybrid_organoid_deltap_gamma),
                max_log_shift=float(config.hybrid_organoid_deltap_max_log_shift),
                ratio_lower=float(config.hybrid_organoid_deltap_ratio_lower),
                ratio_upper=float(config.hybrid_organoid_deltap_ratio_upper),
                release_ready=(in_band_streak >= required_in_band_runs),
            )
        fine_runs_since_coarse = _fine_runs_since_coarse_phase(
            coupled_root,
            current_run=int(current_run_idx),
            first_run=int(start_run),
        )
        post_deltap_probe_phase = False
        if (
            bool(config.hybrid_organoid_deltap_scaling)
            and bool(config.hybrid_post_deltap_probes)
            and not run2_equalization
            and not coarse_deltap_scaling
            and not baseline_restored_for_sensitivity
            and (
                probe_cycle_locked
                or (
                    in_band_streak >= required_in_band_runs
                    and int(fine_runs_since_coarse) < 2
                )
            )
        ):
            if not probe_cycle_locked:
                _save_probe_baseline_state(
                    coupled_root,
                    all_plans,
                    next_phase=0,
                    locked=True,
                    consumed_for_sensitivity=False,
                )
                probe_baseline_map, probe_baseline_bc_map, probe_state = _load_probe_baseline_state(
                    coupled_root
                )
                probe_next_phase = int(probe_state.get("next_phase", 0))
            if not probe_baseline_map:
                probe_baseline_map = _post_deltap_probe_baseline_map(
                    coupled_root,
                    current_run=int(current_run_idx),
                    first_run=int(start_run),
                    first_probe_side=str(config.hybrid_alternate_start_side),
                )
            post_deltap_probe_phase = _apply_post_deltap_probe_phase(
                all_plans,
                phase=int(probe_next_phase),
                start_side=str(config.hybrid_alternate_start_side),
                max_log_step=float(config.hybrid_post_deltap_probe_log_step),
                baseline_map=probe_baseline_map,
            )
            if post_deltap_probe_phase:
                next_phase = int(probe_next_phase) + 1
                _update_probe_baseline_state(
                    coupled_root,
                    next_phase=next_phase,
                    locked=(next_phase < 2),
                    consumed_for_sensitivity=False,
                )
        if baseline_restored_for_sensitivity:
            _update_probe_baseline_state(
                coupled_root,
                next_phase=3,
                locked=False,
                consumed_for_sensitivity=True,
            )
        sensitivity_ready = (
            bool(config.hybrid_q_sensitivity_control)
            and int(current_run_idx) >= max(
                int(start_run), int(config.hybrid_q_sensitivity_start_run)
            )
        )
        sensitivity_map = pending_state.get("q_error_sensitivity_map", {})
        side_median_sensitivity_map = pending_state.get("side_median_q_error_sensitivity_map", {})
        if baseline_restored_for_sensitivity:
            first_post_probe_map = _build_first_post_probe_sensitivity_map(
                coupled_root,
                current_run=int(current_run_idx),
                first_run=int(start_run),
                min_logq_change=float(config.hybrid_q_sensitivity_min_logq_change),
                pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
                opposite_side_referenced_error=bool(config.hybrid_opposite_side_referenced_error),
                error_mode=(
                    "opposite_side_drop"
                    if bool(config.hybrid_opposite_side_referenced_error)
                    else "direct"
                ),
            )
            if first_post_probe_map:
                sensitivity_map = dict(sensitivity_map)
                sensitivity_map.update(first_post_probe_map)
            side_median_first_post_probe_map = _build_first_post_probe_sensitivity_map(
                coupled_root,
                current_run=int(current_run_idx),
                first_run=int(start_run),
                min_logq_change=float(config.hybrid_q_sensitivity_min_logq_change),
                pressure_scale_floor=float(config.hybrid_pressure_scale_floor),
                opposite_side_referenced_error=True,
                error_mode="side_median",
            )
            if side_median_first_post_probe_map:
                side_median_sensitivity_map = dict(side_median_sensitivity_map)
                side_median_sensitivity_map.update(side_median_first_post_probe_map)
        if (
            bool(config.hybrid_opposite_side_referenced_error)
            and not run2_equalization
            and not coarse_deltap_scaling
            and not post_deltap_probe_phase
            and not sensitivity_ready
        ):
            _apply_opposite_side_referenced_error_controller(
                all_plans,
                config=config,
                retarget=True,
            )
        if (
            not run2_equalization
            and not coarse_deltap_scaling
            and not post_deltap_probe_phase
            and sensitivity_ready
        ):
            _apply_q_error_sensitivity_scaling(
                all_plans,
                q_error_sensitivity_map=sensitivity_map,
                q_error_sensitivity_fallback_map=pending_state.get("q_error_sensitivity_fallback_map", {}),
                gain=float(config.hybrid_q_sensitivity_alpha),
                arterial_gain=config.hybrid_q_sensitivity_alpha_arterial,
                venous_gain=config.hybrid_q_sensitivity_alpha_venous,
            )
            _mark_branch_secant_projection_disabled(all_plans)
        if (
            bool(config.hybrid_alternate_sides)
            and not run2_equalization
            and not coarse_deltap_scaling
            and not post_deltap_probe_phase
        ):
            if bool(config.hybrid_organoid_deltap_scaling) and bool(
                config.hybrid_post_deltap_probes
            ):
                active_side = _active_side_for_run(
                    run_idx=int(fine_runs_since_coarse),
                    hybrid_start_run=3,
                    start_side=str(config.hybrid_alternate_start_side),
                )
            else:
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
        if (
            bool(config.hybrid_opposite_side_referenced_error)
            and not run2_equalization
            and not coarse_deltap_scaling
            and not post_deltap_probe_phase
        ):
            _apply_side_gauge_common_mode_control(
                all_plans,
                config=config,
                q_error_sensitivity_map=side_median_sensitivity_map,
                q_error_sensitivity_fallback_map=pending_state.get("side_median_q_error_sensitivity_fallback_map", {}),
            )
        if (
            not run2_equalization
            and not coarse_deltap_scaling
            and not post_deltap_probe_phase
        ):
            _apply_final_trust_region_caps(all_plans)
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
            row["hybrid_organoid_deltap_ratio"] = float(
                plan.get("hybrid_organoid_deltap_ratio", float("nan"))
            )
            row["hybrid_initial_side_total_ratio"] = float(
                plan.get("hybrid_initial_side_total_ratio", float("nan"))
            )
            row["hybrid_post_deltap_probe_phase"] = int(
                plan.get("hybrid_post_deltap_probe_phase", -1)
            )
            row["hybrid_post_deltap_baseline_flow"] = float(
                plan.get("hybrid_post_deltap_baseline_flow", float("nan"))
            )
            row["hybrid_q_error_sensitivity_applied"] = bool(
                plan.get("hybrid_q_error_sensitivity_applied", False)
            )
            row["hybrid_q_error_sensitivity_status"] = str(
                plan.get("hybrid_q_error_sensitivity_status", "")
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
            row["hybrid_q_error_sensitivity_delta_uncapped"] = float(
                plan.get("hybrid_q_error_sensitivity_delta_uncapped", float("nan"))
            )
            row["hybrid_branch_total_pre_projection_delta"] = float(
                plan.get("hybrid_branch_total_pre_projection_delta", float("nan"))
            )
            row["hybrid_branch_common_mode_component"] = float(
                plan.get("hybrid_branch_common_mode_component", float("nan"))
            )
            row["hybrid_branch_redistribution_delta"] = float(
                plan.get("hybrid_branch_redistribution_delta", float("nan"))
            )
            row["hybrid_branch_projection_weight_raw"] = float(
                plan.get("hybrid_branch_projection_weight_raw", float("nan"))
            )
            row["hybrid_branch_projection_weight"] = float(
                plan.get("hybrid_branch_projection_weight", float("nan"))
            )
            row["hybrid_branch_projection_weight_clip"] = float(
                plan.get("hybrid_branch_projection_weight_clip", float("nan"))
            )
            row["hybrid_branch_projection_weight_mode"] = str(
                plan.get("hybrid_branch_projection_weight_mode", "")
            )
            row["hybrid_side_median_pressure_error_raw"] = float(
                plan.get("hybrid_side_median_pressure_error_raw", float("nan"))
            )
            row["hybrid_side_median_pressure_error_normalized"] = float(
                plan.get("hybrid_side_median_pressure_error_normalized", float("nan"))
            )
            row["hybrid_side_median_pressure_error_scale"] = float(
                plan.get("hybrid_side_median_pressure_error_scale", float("nan"))
            )
            row["hybrid_side_median_darcy"] = float(
                plan.get("hybrid_side_median_darcy", float("nan"))
            )
            row["hybrid_side_median_0d"] = float(
                plan.get("hybrid_side_median_0d", float("nan"))
            )
            row["hybrid_branch_trust_region_cap"] = float(
                plan.get("hybrid_branch_trust_region_cap", float("nan"))
            )
            row["hybrid_final_uncapped_delta_log10_q"] = float(
                plan.get("hybrid_final_uncapped_delta_log10_q", float("nan"))
            )
            row["hybrid_final_cap_delta_log10_q"] = float(
                plan.get("hybrid_final_cap_delta_log10_q", float("nan"))
            )
            row["hybrid_final_cap_applied"] = bool(
                plan.get("hybrid_final_cap_applied", False)
            )
            row["hybrid_postsolve_relaxation_alpha"] = float(
                row.get("hybrid_postsolve_relaxation_alpha", float("nan"))
            )
            row["hybrid_predicted_bounded_error"] = float(
                row.get("hybrid_predicted_bounded_error", float("nan"))
            )
            row["hybrid_sensitivity_applied"] = bool(plan.get("hybrid_sensitivity_applied", False))
            row["hybrid_sensitivity_status"] = str(
                plan.get("hybrid_sensitivity_status", "")
            )
            row["hybrid_sensitivity_multiplier"] = float(
                plan.get("hybrid_sensitivity_multiplier", float("nan"))
            )
            row["hybrid_sensitivity_slope"] = float(
                plan.get("hybrid_sensitivity_slope", float("nan"))
            )
            row["hybrid_sensitivity_delta_uncapped"] = float(
                plan.get("hybrid_sensitivity_delta_uncapped", float("nan"))
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
            row["hybrid_side_gauge_common_mode_error"] = float(
                plan.get("hybrid_side_gauge_common_mode_error", float("nan"))
            )
            row["hybrid_side_gauge_common_mode_delta"] = float(
                plan.get("hybrid_side_gauge_common_mode_delta", float("nan"))
            )
            row["hybrid_side_gauge_common_mode_active"] = bool(
                plan.get("hybrid_side_gauge_common_mode_active", False)
            )
            row["hybrid_side_gauge_common_mode_slope"] = float(
                plan.get("hybrid_side_gauge_common_mode_slope", float("nan"))
            )
            row["hybrid_side_gauge_common_mode_slope_source"] = str(
                plan.get("hybrid_side_gauge_common_mode_slope_source", "")
            )
            row["hybrid_side_gauge_common_mode_slope_samples"] = float(
                plan.get("hybrid_side_gauge_common_mode_slope_samples", float("nan"))
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
