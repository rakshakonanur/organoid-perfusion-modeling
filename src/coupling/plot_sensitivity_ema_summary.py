#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from collections import defaultdict
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


def _q_magnitude(row: dict[str, str]) -> float:
    for key in ("Q_0D", "Q_for_R", "hybrid_target_flow", "hybrid_target_flow_raw"):
        val = _safe_float(row.get(key))
        if math.isfinite(val):
            return abs(val)
    return float("nan")


def _error_value(row: dict[str, str], error_mode: str) -> float:
    if error_mode == "bounded":
        return abs(_safe_float(row.get("terminal_pressure_error_bounded")))
    if error_mode == "rel":
        return abs(_safe_float(row.get("terminal_pressure_error_rel")))
    bounded = _safe_float(row.get("terminal_pressure_error_bounded"))
    if math.isfinite(bounded):
        return abs(bounded)
    return abs(_safe_float(row.get("terminal_pressure_error_rel")))


def _branch_label(key: tuple[int, str, int]) -> str:
    organoid, side, branch_id = key
    return f"{side} {branch_id} (org {organoid})"


def _mean_std(values: list[float]) -> tuple[float, float]:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return float(mean), 0.0
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    return float(mean), float(math.sqrt(var))


def _parse_branch_specs(raw: str) -> set[tuple[int, str, int]]:
    out: set[tuple[int, str, int]] = set()
    text = str(raw or "").strip()
    if not text:
        return out
    for part in text.split(","):
        fields = [p.strip() for p in part.split(":")]
        if len(fields) != 3:
            raise ValueError(
                "Invalid branch spec. Use organoid:side:branch_id, e.g. 1:venous:38"
            )
        organoid = int(fields[0])
        side = fields[1].lower()
        branch_id = int(fields[2])
        out.add((organoid, side, branch_id))
    return out


def build_branch_series(
    rows: list[dict[str, str]],
    *,
    ema_alpha: float,
    min_logq_change: float,
    error_mode: str,
) -> dict[tuple[int, str, int], list[dict[str, float]]]:
    grouped: dict[tuple[int, str, int], list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        try:
            key = (
                int(row.get("organoid", 0)),
                str(row.get("side", "")).strip().lower(),
                int(row.get("branch_id", -1)),
            )
            run = int(row.get("run", -1))
        except Exception:
            continue
        q_mag = _q_magnitude(row)
        err = _error_value(row, error_mode)
        logq = (
            math.log10(max(q_mag, math.ulp(1.0)))
            if math.isfinite(q_mag) and q_mag > 0.0
            else float("nan")
        )
        grouped[key].append(
            {
                "run": float(run),
                "error": float(err),
                "logq": float(logq),
                "sensitivity_sample": float("nan"),
                "sensitivity_ema": float("nan"),
            }
        )

    ema_alpha = max(0.0, min(float(ema_alpha), 1.0))
    min_logq_change = max(float(min_logq_change), math.ulp(1.0))
    for key, items in grouped.items():
        items.sort(key=lambda item: item["run"])
        ema = float("nan")
        for prev, cur in zip(items[:-1], items[1:]):
            dx = float(cur["logq"] - prev["logq"])
            de = float(cur["error"] - prev["error"])
            if math.isfinite(dx) and math.isfinite(de) and abs(dx) > math.ulp(1.0):
                dx_eff = math.copysign(max(abs(dx), min_logq_change), dx)
                sample = abs(de / dx_eff)
                cur["sensitivity_sample"] = float(sample)
                ema = sample if not math.isfinite(ema) else ema_alpha * sample + (1.0 - ema_alpha) * ema
                cur["sensitivity_ema"] = float(ema)
        if items:
            items[0]["sensitivity_ema"] = float("nan")
    return grouped


def build_side_aggregate(
    branch_series: dict[tuple[int, str, int], list[dict[str, float]]]
) -> dict[str, dict[int, dict[str, float]]]:
    aggregate: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"error": [], "sample": [], "ema": []})
    )
    for (organoid, side, branch_id), items in branch_series.items():
        _ = organoid, branch_id
        for item in items:
            run = int(item["run"])
            aggregate[side][run]["error"].append(float(item["error"]))
            aggregate[side][run]["sample"].append(float(item["sensitivity_sample"]))
            aggregate[side][run]["ema"].append(float(item["sensitivity_ema"]))

    out: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for side, run_map in aggregate.items():
        for run, vals in run_map.items():
            err_mean, err_std = _mean_std(vals["error"])
            sample_mean, sample_std = _mean_std(vals["sample"])
            ema_mean, ema_std = _mean_std(vals["ema"])
            out[side][run] = {
                "error_mean": err_mean,
                "error_std": err_std,
                "sample_mean": sample_mean,
                "sample_std": sample_std,
                "ema_mean": ema_mean,
                "ema_std": ema_std,
            }
    return out


def _plot_band(ax, xs: list[int], means: list[float], stds: list[float], color: str, label: str) -> None:
    ax.plot(xs, means, color=color, linewidth=2.2, label=label)
    upper = [m + s if math.isfinite(m) and math.isfinite(s) else float("nan") for m, s in zip(means, stds)]
    lower = [m - s if math.isfinite(m) and math.isfinite(s) else float("nan") for m, s in zip(means, stds)]
    ax.fill_between(xs, lower, upper, color=color, alpha=0.18)


def _select_individual_keys(
    branch_series: dict[tuple[int, str, int], list[dict[str, float]]],
    branch_specs: set[tuple[int, str, int]],
    side_filter: str,
    max_count: int,
) -> list[tuple[int, str, int]]:
    if branch_specs:
        keys = [key for key in branch_series if key in branch_specs]
    else:
        keys = [key for key in branch_series if side_filter == "both" or key[1] == side_filter]
        def sort_key(key: tuple[int, str, int]) -> float:
            vals = branch_series[key]
            if not vals:
                return -float("inf")
            return max((item["error"] for item in vals if math.isfinite(item["error"])), default=-float("inf"))
        keys.sort(key=sort_key, reverse=True)
        keys = keys[: max(0, int(max_count))]
    return sorted(keys, key=lambda key: (key[1], key[0], key[2]))


def make_plot(
    *,
    root: Path,
    branch_series: dict[tuple[int, str, int], list[dict[str, float]]],
    side_aggregate: dict[str, dict[int, dict[str, float]]],
    output_path: Path,
    plot_individual: bool,
    branch_specs: set[tuple[int, str, int]],
    individual_side: str,
    individual_max: int,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True, constrained_layout=True)
    side_colors = {"arterial": "#b24a2a", "venous": "#1f6a8a"}
    metrics = [
        ("error", "Mean branch error", "Branch error"),
        ("sample", "Raw sensitivity sample |dE/dlog10|Q||", "Sensitivity sample"),
        ("ema", "EMA sensitivity", "Sensitivity EMA"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for side in ("arterial", "venous"):
            run_map = side_aggregate.get(side, {})
            xs = sorted(run_map)
            if not xs:
                continue
            means = [run_map[x][f"{metric}_mean"] for x in xs]
            stds = [run_map[x][f"{metric}_std"] for x in xs]
            _plot_band(ax, xs, means, stds, side_colors[side], side.capitalize())
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    if plot_individual:
        keys = _select_individual_keys(branch_series, branch_specs, individual_side, individual_max)
        line_alpha = 0.35 if not branch_specs else 0.8
        for key in keys:
            color = side_colors.get(key[1], "#555555")
            series = branch_series[key]
            xs = [int(item["run"]) for item in series]
            ys_error = [item["error"] for item in series]
            ys_sample = [item["sensitivity_sample"] for item in series]
            ys_ema = [item["sensitivity_ema"] for item in series]
            for ax, ys in zip(axes, (ys_error, ys_sample, ys_ema)):
                ax.plot(xs, ys, color=color, alpha=line_alpha, linewidth=1.0)
            last_x = None
            last_y = None
            for x, y in reversed(list(zip(xs, ys_ema))):
                if math.isfinite(y):
                    last_x, last_y = x, y
                    break
            if last_x is None:
                for x, y in reversed(list(zip(xs, ys_error))):
                    if math.isfinite(y):
                        last_x, last_y = x, y
                        break
            if last_x is not None and last_y is not None:
                axes[2].text(last_x, last_y, _branch_label(key), color=color, fontsize=8, alpha=0.9)

    axes[-1].set_xlabel("Run")
    axes[0].legend(loc="best")
    fig.suptitle(f"Sensitivity / EMA / Error summary for {root.name}", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Plot mean arterial/venous branch error, raw sensitivity sample, "
            "and EMA sensitivity from a coupling output directory."
        )
    )
    ap.add_argument("root", help="Coupling output directory containing terminal_pd_convergence.csv")
    ap.add_argument(
        "--csv",
        default="terminal_pd_convergence.csv",
        help="CSV filename relative to root. Default: terminal_pd_convergence.csv",
    )
    ap.add_argument(
        "--output",
        default="sensitivity_ema_summary.png",
        help="Output image path. Relative paths are resolved under the root directory.",
    )
    ap.add_argument("--ema-alpha", type=float, default=0.5, help="EMA alpha for sensitivity smoothing.")
    ap.add_argument(
        "--min-logq-change",
        type=float,
        default=0.005,
        help="Minimum |delta log10|Q|| used when reconstructing sensitivity samples.",
    )
    ap.add_argument(
        "--error-mode",
        choices=["auto", "bounded", "rel"],
        default="auto",
        help="Which error column to use.",
    )
    ap.add_argument(
        "--plot-individual",
        action="store_true",
        help="Overlay individual branch traces and label them on the EMA panel.",
    )
    ap.add_argument(
        "--individual-side",
        choices=["arterial", "venous", "both"],
        default="both",
        help="Which side's branch traces to overlay when --plot-individual is used.",
    )
    ap.add_argument(
        "--individual-max",
        type=int,
        default=12,
        help="Maximum number of automatic branch overlays when --individual-branches is not specified.",
    )
    ap.add_argument(
        "--individual-branches",
        default="",
        help="Comma-separated branch specs organoid:side:branch_id, e.g. 1:venous:38,1:arterial:29",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = root / output_path

    rows = _read_csv(csv_path)
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")

    branch_series = build_branch_series(
        rows,
        ema_alpha=float(args.ema_alpha),
        min_logq_change=float(args.min_logq_change),
        error_mode=str(args.error_mode),
    )
    side_aggregate = build_side_aggregate(branch_series)
    branch_specs = _parse_branch_specs(args.individual_branches)
    make_plot(
        root=root,
        branch_series=branch_series,
        side_aggregate=side_aggregate,
        output_path=output_path,
        plot_individual=bool(args.plot_individual),
        branch_specs=branch_specs,
        individual_side=str(args.individual_side),
        individual_max=int(args.individual_max),
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
