import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_component_csv(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _group_component_rows(rows):
    grouped = {}
    for row in rows:
        name = row["name"]
        grouped.setdefault(name, {"time": [], "flow_in": [], "flow_out": [], "pressure_in": [], "pressure_out": []})
        grouped[name]["time"].append(float(row["time"]))
        grouped[name]["flow_in"].append(float(row["flow_in"]))
        grouped[name]["flow_out"].append(float(row["flow_out"]))
        grouped[name]["pressure_in"].append(float(row["pressure_in"]))
        grouped[name]["pressure_out"].append(float(row["pressure_out"]))
    for data in grouped.values():
        for key in data:
            data[key] = np.asarray(data[key], dtype=float)
    return grouped


def _load_0d_leak_waveforms(iter_dir: Path, organoid_id: int):
    candidates = [iter_dir / "split" / "other.csv", iter_dir / "output.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"Could not find 0D CSV in {iter_dir}")

    grouped = _group_component_rows(_read_component_csv(csv_path))
    art_name = f"leak_art_{organoid_id}"
    ven_name = f"leak_ven_{organoid_id}"
    if art_name not in grouped:
        raise KeyError(f"Could not find {art_name} in {csv_path}")
    if ven_name not in grouped:
        raise KeyError(f"Could not find {ven_name} in {csv_path}")

    return {
        "artery": {
            "time": grouped[art_name]["time"],
            "flow": grouped[art_name]["flow_in"],
            "name": art_name,
        },
        "venous": {
            "time": grouped[ven_name]["time"],
            "flow": grouped[ven_name]["flow_in"],
            "name": ven_name,
        },
    }


def _load_darcy_waveforms(iter_dir: Path, organoid_id: int):
    path = iter_dir / f"organoid_{organoid_id}" / "out_darcy_transient" / "interface_bc_timeseries.json"
    if not path.exists():
        raise FileNotFoundError(f"Could not find Darcy waveform file: {path}")

    data = json.loads(path.read_text())
    t = np.asarray(data["times"], dtype=float)
    q_art = np.asarray(data["q_artery_leak"], dtype=float)
    q_ven = np.asarray(data["q_venous_leak"], dtype=float)

    return {
        "artery": {
            "time": t,
            # Darcy stores outward-normal flux; negate arterial leak so positive means into tissue,
            # matching leak_art_i in the 0D CSV.
            "flow": -q_art,
            "raw_flow": q_art,
        },
        "venous": {
            "time": t,
            # For the no-vasc 0D CSV, leak_ven_i is stored with the opposite sign of
            # Darcy's outward-normal venous leak. Negate here so the plotted waveforms
            # share the same convention.
            "flow": -q_ven,
            "raw_flow": q_ven,
        },
    }


def plot_terminal_waveforms(iter_dir: Path, organoid_ids, out_dir: Path, show: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=len(organoid_ids),
        ncols=2,
        figsize=(12, 2.8 * len(organoid_ids)),
        sharex=False,
        squeeze=False,
    )

    summary_rows = []

    for row, organoid_id in enumerate(organoid_ids):
        zero_d = _load_0d_leak_waveforms(iter_dir, organoid_id)
        darcy = _load_darcy_waveforms(iter_dir, organoid_id)

        for col, key in enumerate(("artery", "venous")):
            ax = axes[row][col]
            ax.plot(
                zero_d[key]["time"],
                zero_d[key]["flow"],
                label="0D",
                linewidth=2.0,
            )
            ax.plot(
                darcy[key]["time"],
                darcy[key]["flow"],
                label="Darcy",
                linewidth=2.0,
                linestyle="--",
            )
            ax.set_title(f"Organoid {organoid_id} {'arterial' if key == 'artery' else 'venous'} leak")
            ax.set_xlabel("Time")
            ax.set_ylabel("Flow")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()

            q0 = np.asarray(zero_d[key]["flow"], dtype=float)
            qd = np.asarray(darcy[key]["flow"], dtype=float)
            if q0.shape == qd.shape and np.allclose(zero_d[key]["time"], darcy[key]["time"]):
                diff = qd - q0
                rmse = float(np.sqrt(np.mean(diff**2)))
                scale = float(np.sqrt(np.mean(q0**2))) if np.any(q0) else 0.0
                nrmse = rmse / scale if scale > 0.0 else np.nan
                max_abs = float(np.max(np.abs(diff)))
            else:
                rmse = np.nan
                nrmse = np.nan
                max_abs = np.nan

            summary_rows.append(
                {
                    "organoid_id": organoid_id,
                    "waveform": key,
                    "rmse": rmse,
                    "nrmse_rms_scaled": nrmse,
                    "max_abs_error": max_abs,
                }
            )

    fig.tight_layout()
    fig_path = out_dir / "terminal_flow_waveforms_comparison.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")

    csv_path = out_dir / "terminal_flow_waveforms_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["organoid_id", "waveform", "rmse", "nrmse_rms_scaled", "max_abs_error"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(f"Wrote figure: {fig_path}")
    print(f"Wrote summary: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot 0D vs Darcy terminal flow waveforms for each organoid."
    )
    parser.add_argument(
        "--iter-dir",
        type=Path,
        required=True,
        help="Transient iteration directory, e.g. src/coupling-output-transient/.../iter_009",
    )
    parser.add_argument(
        "--organoid-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Organoid IDs to plot",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots and summary CSV. Defaults to <iter-dir>/plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window in addition to saving files.",
    )
    args = parser.parse_args()

    iter_dir = args.iter_dir.expanduser().resolve()
    out_dir = (args.out_dir.expanduser().resolve() if args.out_dir else iter_dir / "plots")
    plot_terminal_waveforms(iter_dir, args.organoid_ids, out_dir, args.show)


if __name__ == "__main__":
    main()
