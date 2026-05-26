#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "coupling_plot_cache"
os.environ.setdefault("MPLCONFIGDIR", str(_cache_root / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_root / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as fp:
        return list(csv.DictReader(fp))


def _run_index(path: Path) -> int:
    match = re.search(r"run_(\d+)$", path.name)
    return int(match.group(1)) if match else -1


def _percentile(values: list[float], q: float) -> float:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return float("nan")
    pos = (len(vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] * (hi - pos) + vals[hi] * (pos - lo))


def _mean(values: list[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _mean_abs(values: list[float]) -> float:
    return _mean([abs(v) for v in values if math.isfinite(v)])


def _rel_percent(a: float, b: float, floor: float = 1.0) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return float("nan")
    return 100.0 * abs(a - b) / max(min(abs(a), abs(b)), floor)


def _selected_inner_summary(run_dir: Path) -> dict[str, str]:
    rows = _read_csv(run_dir / "inner_resistance_0d_search_summary.csv")
    if not rows:
        return {}
    selected = [row for row in rows if str(row.get("selected", "")).lower() in {"true", "1", "yes"}]
    return selected[-1] if selected else rows[-1]


def _terminal_stats(run_dir: Path) -> dict[str, Any]:
    rows = _read_csv(run_dir / "terminal_resistance_bc_summary.csv")
    out: dict[str, Any] = {}
    for side in ("arterial", "venous", "all"):
        side_rows = [row for row in rows if side == "all" or row.get("side") == side]
        pi_pd: list[float] = []
        pd_interface: list[float] = []
        pi_interface: list[float] = []
        pi_interface_rel: list[float] = []
        q_abs: list[float] = []
        r_abs: list[float] = []
        pd_reasons: Counter[str] = Counter()
        r_reasons: Counter[str] = Counter()
        bc_types: Counter[str] = Counter()
        for row in side_rows:
            p_interface = _safe_float(row.get("P_interface"))
            p_0d = _safe_float(row.get("P_0D"))
            pd = _safe_float(row.get("Pd_next"))
            q = _safe_float(row.get("Q_0D"))
            r = _safe_float(row.get("R_next"))
            if math.isfinite(p_0d) and math.isfinite(pd):
                pi_pd.append(p_0d - pd)
            if math.isfinite(pd) and math.isfinite(p_interface):
                pd_interface.append(pd - p_interface)
            if math.isfinite(p_0d) and math.isfinite(p_interface):
                pi_interface.append(p_0d - p_interface)
                pi_interface_rel.append(_rel_percent(p_0d, p_interface))
            if math.isfinite(q):
                q_abs.append(abs(q))
            if math.isfinite(r) and r > 0.0:
                r_abs.append(abs(r))
            pd_reasons[str(row.get("Pd_target_reason", ""))] += 1
            r_reasons[str(row.get("R_target_reason", ""))] += 1
            bc_types[str(row.get("bc_type", ""))] += 1
        out[side] = {
            "n": len(side_rows),
            "pi_pd_mean_abs": _mean_abs(pi_pd),
            "pd_interface_mean_abs": _mean_abs(pd_interface),
            "pi_interface_mean_abs": _mean_abs(pi_interface),
            "pi_interface_rel_median": _percentile(pi_interface_rel, 50.0),
            "pi_interface_rel_p90": _percentile(pi_interface_rel, 90.0),
            "pi_interface_rel_max": max(pi_interface_rel) if pi_interface_rel else float("nan"),
            "q_abs_sum": sum(q_abs),
            "q_abs_median": _percentile(q_abs, 50.0),
            "r_abs_median": _percentile(r_abs, 50.0),
            "r_abs_p90": _percentile(r_abs, 90.0),
            "pd_reasons": pd_reasons,
            "r_reasons": r_reasons,
            "bc_types": bc_types,
        }
    return out


def collect_runs(root: Path) -> list[dict[str, Any]]:
    run_dirs = sorted(
        [path for path in root.glob("run_*") if path.is_dir() and _run_index(path) >= 0],
        key=_run_index,
    )
    records: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run = _run_index(run_dir)
        selected = _selected_inner_summary(run_dir)
        terminal = _terminal_stats(run_dir)
        if not selected and terminal["all"]["n"] == 0:
            continue
        records.append({"run": run, "selected": selected, "terminal": terminal})
    return records


def _series(records: list[dict[str, Any]], key: str, default: float = float("nan")) -> list[float]:
    vals = []
    for record in records:
        selected = record["selected"]
        vals.append(_safe_float(selected.get(key, default)))
    return vals


def _terminal_series(records: list[dict[str, Any]], side: str, key: str) -> list[float]:
    return [_safe_float(record["terminal"][side].get(key, float("nan"))) for record in records]


def _reason_series(records: list[dict[str, Any]], side: str, reason: str) -> list[int]:
    return [int(record["terminal"][side]["pd_reasons"].get(reason, 0)) for record in records]


def _plot_summary(records: list[dict[str, Any]], outdir: Path, title_suffix: str) -> Path:
    runs = [record["run"] for record in records]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(f"Coupling Convergence Summary{title_suffix}", fontsize=14)

    ax = axes[0]
    ax.plot(runs, _series(records, "score"), marker="o", ms=3, label="score")
    ax.plot(runs, _series(records, "pressure_score"), marker=".", label="pressure score")
    if any(math.isfinite(v) for v in _series(records, "implied_interface_score")):
        ax.plot(runs, _series(records, "implied_interface_score"), marker=".", label="implied interface")
    ax.plot(runs, _series(records, "jump_score"), marker=".", label="jump score")
    if any(math.isfinite(v) for v in _series(records, "flow_guard_score")):
        ax.plot(runs, _series(records, "flow_guard_score"), marker=".", label="flow guard score")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2)

    ax = axes[1]
    ax.plot(runs, _series(records, "median_error_percent"), marker="o", ms=3, label="median")
    ax.plot(runs, _series(records, "p90_error_percent"), marker=".", label="p90")
    ax.plot(runs, _series(records, "max_error_percent"), marker=".", label="max")
    ax.set_ylabel("0D-Darcy pressure error (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=3)

    ax = axes[2]
    ax.plot(runs, _terminal_series(records, "arterial", "pi_interface_rel_median"), marker="o", ms=3, label="arterial median")
    ax.plot(runs, _terminal_series(records, "arterial", "pi_interface_rel_p90"), marker=".", label="arterial p90")
    ax.plot(runs, _terminal_series(records, "venous", "pi_interface_rel_median"), marker="o", ms=3, label="venous median")
    ax.plot(runs, _terminal_series(records, "venous", "pi_interface_rel_p90"), marker=".", label="venous p90")
    ax.set_ylabel("terminal pressure error (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2)

    ax = axes[3]
    ax.plot(runs, _series(records, "jump_median_percent"), marker="o", ms=3, label="jump median")
    ax.plot(runs, _series(records, "jump_p90_percent"), marker=".", label="jump p90")
    if any(math.isfinite(v) for v in _series(records, "flow_guard_increase_fraction")):
        ax.plot(
            runs,
            [100.0 * v for v in _series(records, "flow_guard_increase_fraction")],
            marker=".",
            label="guarded flow increases (%)",
        )
    ax.set_xlabel("run")
    ax.set_ylabel("continuation / safety (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2)

    fig.tight_layout()
    out = outdir / "coupling_error_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_late(records: list[dict[str, Any]], outdir: Path, late_count: int, title_suffix: str) -> Path:
    late = records[-late_count:] if len(records) > late_count else records
    runs = [record["run"] for record in late]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)
    fig.suptitle(f"Late-Stage Coupling Diagnostics{title_suffix}", fontsize=14)

    ax = axes[0]
    ax.plot(runs, _series(late, "score"), marker="o", ms=4, label="score")
    ax.plot(runs, _series(late, "pressure_score"), marker=".", label="pressure score")
    if any(math.isfinite(v) for v in _series(late, "implied_interface_score")):
        ax.plot(runs, _series(late, "implied_interface_score"), marker=".", label="implied interface")
    ax.plot(runs, _series(late, "flow_guard_score"), marker=".", label="flow guard")
    ax.set_ylabel("score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    ax.plot(runs, _terminal_series(late, "arterial", "pd_interface_mean_abs"), marker="o", ms=4, label="arterial |Pd-Pdarcy|")
    ax.plot(runs, _terminal_series(late, "venous", "pd_interface_mean_abs"), marker="o", ms=4, label="venous |Pd-Pdarcy|")
    ax.plot(runs, _terminal_series(late, "all", "pi_pd_mean_abs"), marker=".", label="all |Pi-Pd|")
    ax.set_ylabel("pressure units")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    ax.plot(runs, _terminal_series(late, "arterial", "pi_interface_rel_p90"), marker="o", ms=4, label="arterial p90")
    ax.plot(runs, _terminal_series(late, "venous", "pi_interface_rel_p90"), marker="o", ms=4, label="venous p90")
    ax.plot(runs, _terminal_series(late, "all", "pi_interface_rel_max"), marker=".", label="all max")
    ax.set_ylabel("pressure error (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[3]
    for reason in (
        "interface",
        "resistance_implied_interface",
        "pressure_alignment",
        "parent_flow_suppression",
        "drive_limited_hold",
    ):
        vals = _reason_series(late, "all", reason)
        if any(vals):
            ax.plot(runs, vals, marker=".", label=reason)
    ax.set_xlabel("run")
    ax.set_ylabel("Pd target count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2)

    fig.tight_layout()
    out = outdir / "coupling_error_late_stage.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _plot_resistance(records: list[dict[str, Any]], outdir: Path, title_suffix: str) -> Path:
    runs = [record["run"] for record in records]
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(f"Terminal Resistance Diagnostics{title_suffix}", fontsize=14)

    ax = axes[0]
    ax.semilogy(runs, _terminal_series(records, "arterial", "r_abs_median"), marker="o", ms=3, label="arterial median R")
    ax.semilogy(runs, _terminal_series(records, "arterial", "r_abs_p90"), marker=".", label="arterial p90 R")
    ax.semilogy(runs, _terminal_series(records, "venous", "r_abs_median"), marker="o", ms=3, label="venous median R")
    ax.semilogy(runs, _terminal_series(records, "venous", "r_abs_p90"), marker=".", label="venous p90 R")
    ax.set_ylabel("|R|")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best", ncols=2)

    ax = axes[1]
    ax.plot(runs, _terminal_series(records, "arterial", "q_abs_sum"), marker="o", ms=3, label="arterial sum |Q|")
    ax.plot(runs, _terminal_series(records, "venous", "q_abs_sum"), marker="o", ms=3, label="venous sum |Q|")
    ax.set_ylabel("sum |Q_0D|")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    for reason in ("pressure_jump_calibration", "continuity_decay", "pressure_handoff", "frozen_side"):
        vals = [int(record["terminal"]["all"]["r_reasons"].get(reason, 0)) for record in records]
        if any(vals):
            ax.plot(runs, vals, marker=".", label=reason)
    ax.set_xlabel("run")
    ax.set_ylabel("R update count")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncols=2)

    fig.tight_layout()
    out = outdir / "terminal_resistance_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def _write_summary_csv(records: list[dict[str, Any]], outdir: Path) -> Path:
    out = outdir / "coupling_plot_summary.csv"
    fieldnames = [
        "run",
        "score",
        "pressure_score",
        "implied_interface_score",
        "implied_median_percent",
        "implied_p90_percent",
        "arterial_implied_score",
        "venous_implied_score",
        "arterial_implied_median_percent",
        "venous_implied_median_percent",
        "jump_score",
        "flow_guard_score",
        "median_error_percent",
        "p90_error_percent",
        "max_error_percent",
        "jump_median_percent",
        "jump_p90_percent",
        "flow_guard_increase_fraction",
        "arterial_error_median_percent",
        "arterial_error_p90_percent",
        "venous_error_median_percent",
        "venous_error_p90_percent",
        "all_pi_pd_mean_abs",
        "all_pd_interface_mean_abs",
    ]
    with out.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            selected = record["selected"]
            terminal = record["terminal"]
            writer.writerow({
                "run": record["run"],
                "score": selected.get("score", ""),
                "pressure_score": selected.get("pressure_score", ""),
                "implied_interface_score": selected.get("implied_interface_score", ""),
                "implied_median_percent": selected.get("implied_median_percent", ""),
                "implied_p90_percent": selected.get("implied_p90_percent", ""),
                "arterial_implied_score": selected.get("arterial_implied_score", ""),
                "venous_implied_score": selected.get("venous_implied_score", ""),
                "arterial_implied_median_percent": selected.get("arterial_implied_median_percent", ""),
                "venous_implied_median_percent": selected.get("venous_implied_median_percent", ""),
                "jump_score": selected.get("jump_score", ""),
                "flow_guard_score": selected.get("flow_guard_score", ""),
                "median_error_percent": selected.get("median_error_percent", ""),
                "p90_error_percent": selected.get("p90_error_percent", ""),
                "max_error_percent": selected.get("max_error_percent", ""),
                "jump_median_percent": selected.get("jump_median_percent", ""),
                "jump_p90_percent": selected.get("jump_p90_percent", ""),
                "flow_guard_increase_fraction": selected.get("flow_guard_increase_fraction", ""),
                "arterial_error_median_percent": terminal["arterial"]["pi_interface_rel_median"],
                "arterial_error_p90_percent": terminal["arterial"]["pi_interface_rel_p90"],
                "venous_error_median_percent": terminal["venous"]["pi_interface_rel_median"],
                "venous_error_p90_percent": terminal["venous"]["pi_interface_rel_p90"],
                "all_pi_pd_mean_abs": terminal["all"]["pi_pd_mean_abs"],
                "all_pd_interface_mean_abs": terminal["all"]["pd_interface_mean_abs"],
            })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create convergence plots for coupled 0D-Darcy runs.")
    parser.add_argument("--root", default="src/coupling-output", help="Coupling output root containing run_* folders.")
    parser.add_argument("--outdir", default=None, help="Directory for plots. Defaults to <root>/convergence_plots.")
    parser.add_argument("--late-count", type=int, default=25, help="Number of final runs to include in the late-stage plot.")
    parser.add_argument("--title", default="", help="Optional suffix added to plot titles.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"[error] output root not found: {root}")
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else root / "convergence_plots"
    outdir.mkdir(parents=True, exist_ok=True)

    records = collect_runs(root)
    if not records:
        raise SystemExit(f"[error] no run summaries found under {root}")

    title_suffix = f" - {args.title}" if args.title else ""
    outputs = [
        _plot_summary(records, outdir, title_suffix),
        _plot_late(records, outdir, max(1, int(args.late_count)), title_suffix),
        _plot_resistance(records, outdir, title_suffix),
        _write_summary_csv(records, outdir),
    ]
    print("[plots] wrote:")
    for path in outputs:
        print(f"  {path}")


if __name__ == "__main__":
    main()
