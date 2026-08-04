#!/usr/bin/env python3
"""Run and score a small oxygen-consumption sensitivity sweep.

This script can run three transport cases for each (Km, Vmax) pair:

1. diffusion-only 1000 um organoid
2. diffusion-only/no-synthetic 3000 um organoid
3. synthetic-vasculature 3000 um organoid

By default, the 1000 um diffusion-only case is used as a prescreen. If its
below-critical organoid volume is zero, the two 3000 um cases are skipped for
that parameter pair. It then scores the pair using the two phenotypes discussed
in the project:

* the 1000 um diffusion-only case should develop a below-critical/necrotic core
* the 3000 um synthetic case should improve substantially relative to the
  3000 um diffusion-only/no-synthetic case

All concentrations are in the transport solver's cgs units:
mol/cm^3 for Km and mol/(cm^3 s) for Vmax.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSPORT = REPO_ROOT / "src" / "solves" / "3d_transport.py"

DEFAULT_PYTHON = "/opt/miniconda3/envs/fenicsx-env/bin/python"

DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "src"
    / "Results"
    / "paper-results"
    / "transport-comparison"
    / "parameter-sweep"
    / "km-vmax"
)

ROOT_DIFFUSION_1000 = (
    REPO_ROOT
    / "src"
    / "Results"
    / "paper-results"
    / "transport-comparison"
    / "channel"
    / "no-vasc-1000um"
    / "run_2"
)
ROOT_DIFFUSION_3000 = (
    REPO_ROOT
    / "src"
    / "Results"
    / "paper-results"
    / "transport-comparison"
    / "channel"
    / "no-vasc-3000um"
    / "run_2"
)
ROOT_SYNTHETIC_3000 = (
    REPO_ROOT
    / "src"
    / "Results"
    / "paper-results"
    / "transport-comparison"
    / "synthetic"
    / "3000um"
    / "run_147"
)

STL_TINY = (
    REPO_ROOT / "files" / "stl" / "organoid-growth-domains" / "tiny" / "organoid-1.stl"
)
STL_CIRCLES = (
    REPO_ROOT
    / "files"
    / "stl"
    / "organoid-growth-domains"
    / "circles"
    / "organoid-1.stl"
)


@dataclass(frozen=True)
class Case:
    name: str
    root: Path
    stl: Path
    synthetic: bool


CASES = (
    Case("diffusion_1000um", ROOT_DIFFUSION_1000, STL_TINY, False),
    Case("diffusion_3000um", ROOT_DIFFUSION_3000, STL_CIRCLES, False),
    Case("synthetic_3000um", ROOT_SYNTHETIC_3000, STL_CIRCLES, True),
)


def fmt_param(value: float) -> str:
    """Stable path-friendly scientific notation."""
    return f"{value:.2e}".replace("+", "").replace("-", "m").replace(".", "p")


def combo_name(km: float, vmax: float) -> str:
    return f"km_{fmt_param(km)}__vmax_{fmt_param(vmax)}"


def final_summary_path(case_dir: Path, case: Case) -> Path:
    suffix = "synthetic" if case.synthetic else "diffusion"
    return case_dir / f"transport_critical_summary_organoid_1_{suffix}.json"


def output_paths(case_dir: Path, case: Case) -> dict[str, Path]:
    suffix = "synthetic" if case.synthetic else "diffusion"
    return {
        "oxygen": case_dir / f"transport_c_organoid_1_{suffix}.xdmf",
        "consumption": case_dir / f"transport_consumption_organoid_1_{suffix}.xdmf",
        "critical": case_dir / f"transport_critical_mask_organoid_1_{suffix}.xdmf",
        "summary": final_summary_path(case_dir, case),
    }


def build_command(
    *,
    python: str,
    case: Case,
    case_dir: Path,
    km: float,
    vmax: float,
    T: float,
    dt: float,
    output_format: str,
    coupled_1d_transport: bool,
) -> list[str]:
    org = case.root / "organoid_1"
    paths = output_paths(case_dir, case)
    cmd = [
        python,
        str(TRANSPORT),
        "--bioreactor-domain",
        str(org / "geometry" / "bioreactor.xdmf"),
        "--facet-file",
        str(org / "geometry" / "mesh_tags.xdmf"),
        "--diffusion-region-path",
        str(case.stl),
        "--c-in-value",
        "2e-7",
        "--marker-32-concentration",
        "2e-7",
        "--T",
        str(T),
        "--dt",
        str(dt),
        "--mm-km",
        str(km),
        "--mm-vmax",
        str(vmax),
        "--out-file",
        str(paths["oxygen"]),
        "--consumption-output-file",
        str(paths["consumption"]),
        "--critical-output-file",
        str(paths["critical"]),
        "--critical-summary-file",
        str(paths["summary"]),
        "--output-format",
        output_format,
    ]

    if case.synthetic:
        cmd.extend(
            [
                "--mesh-inlet-file",
                str(org / "geometry" / "tagged_branches_inlet.bp"),
                "--mesh-outlet-file",
                str(org / "geometry" / "tagged_branches_outlet.bp"),
                "--flow-inlet-file",
                str(org / "geometry" / "flow_checkpoint_inlet.bp"),
                "--flow-outlet-file",
                str(org / "geometry" / "flow_checkpoint_outlet.bp"),
                "--vel-file",
                str(org / "out_darcy" / "u.xdmf"),
                "--interface-bc-file",
                str(org / "out_darcy" / "interface_bc.json"),
                "--concave-exchange-mode",
                "combined",
            ]
        )
        if coupled_1d_transport:
            cmd.append("--coupled-1d-transport")
    else:
        cmd.append("--diffusion-only")

    return cmd


def load_final(summary: Path) -> dict[str, float | str]:
    payload = json.loads(summary.read_text())
    final = payload.get("final", {})
    metadata = payload.get("metadata", {})
    out: dict[str, float | str] = {}
    for key in (
        "organoid_below_critical_volume_fraction",
        "organoid_below_critical_volume_cm3",
        "organoid_volume_cm3",
        "below_critical_volume_cm3",
        "domain_volume_cm3",
        "below_critical_domain_volume_fraction",
        "min_concentration_mol_per_cm3",
        "max_concentration_mol_per_cm3",
        "injection_rate_mol_per_s",
        "outflow_rate_mol_per_s",
        "consumption_rate_mol_per_s",
        "terminal_injection_rate_mol_per_s",
        "terminal_outflow_rate_mol_per_s",
        "concave_injection_rate_mol_per_s",
        "concave_outflow_rate_mol_per_s",
    ):
        value = final.get(key)
        out[key] = float(value) if value is not None else float("nan")
    out["summary_file"] = str(summary)
    out["reported_km"] = float(metadata.get("michaelis_menten_km_mol_per_cm3", "nan"))
    out["reported_vmax"] = float(
        metadata.get("michaelis_menten_vmax_mol_per_cm3_s", "nan")
    )
    return out


def score_combo(
    *,
    km: float,
    vmax: float,
    combo_dir: Path,
    necrotic_threshold: float,
    benefit_threshold: float,
) -> dict[str, float | int | str]:
    summaries = {
        case.name: load_final(final_summary_path(combo_dir / case.name, case))
        for case in CASES
    }
    d1000 = summaries["diffusion_1000um"]
    d3000 = summaries["diffusion_3000um"]
    s3000 = summaries["synthetic_3000um"]

    d1000_below = float(d1000["organoid_below_critical_volume_fraction"])
    d3000_below = float(d3000["organoid_below_critical_volume_fraction"])
    s3000_below = float(s3000["organoid_below_critical_volume_fraction"])
    benefit_abs = d3000_below - s3000_below
    benefit_rel = benefit_abs / d3000_below if d3000_below > 0.0 else float("nan")

    pass_necrotic_1000 = d1000_below >= necrotic_threshold
    pass_synthetic_benefit = benefit_rel >= benefit_threshold if d3000_below > 0.0 else False

    # Smooth ranking score. The pass/fail columns are the main decision; this
    # just sorts useful candidates toward the top.
    necrotic_score = min(max(d1000_below / necrotic_threshold, 0.0), 1.0)
    benefit_score = max(benefit_rel, 0.0) if d3000_below > 0.0 else 0.0
    total_score = 0.4 * necrotic_score + 0.6 * benefit_score

    return {
        "combo": combo_name(km, vmax),
        "km_mol_per_cm3": km,
        "km_uM": km * 1e9,
        "vmax_mol_per_cm3_s": vmax,
        "vmax_mol_per_m3_s": vmax * 1e6,
        "pass_all": int(pass_necrotic_1000 and pass_synthetic_benefit),
        "pass_necrotic_1000um": int(pass_necrotic_1000),
        "pass_synthetic_benefit_3000um": int(pass_synthetic_benefit),
        "ran_3000um_cases": 1,
        "screened_out_no_1000um_critical_volume": 0,
        "score": total_score,
        "diffusion_1000um_organoid_below_critical_fraction": d1000_below,
        "diffusion_1000um_organoid_below_critical_volume_cm3": d1000[
            "organoid_below_critical_volume_cm3"
        ],
        "diffusion_1000um_organoid_volume_cm3": d1000["organoid_volume_cm3"],
        "diffusion_1000um_min_c_mol_per_cm3": d1000[
            "min_concentration_mol_per_cm3"
        ],
        "diffusion_1000um_consumption_rate_mol_per_s": d1000[
            "consumption_rate_mol_per_s"
        ],
        "diffusion_3000um_organoid_below_critical_fraction": d3000_below,
        "diffusion_3000um_organoid_below_critical_volume_cm3": d3000[
            "organoid_below_critical_volume_cm3"
        ],
        "diffusion_3000um_organoid_volume_cm3": d3000["organoid_volume_cm3"],
        "diffusion_3000um_min_c_mol_per_cm3": d3000[
            "min_concentration_mol_per_cm3"
        ],
        "diffusion_3000um_consumption_rate_mol_per_s": d3000[
            "consumption_rate_mol_per_s"
        ],
        "synthetic_3000um_organoid_below_critical_fraction": s3000_below,
        "synthetic_3000um_organoid_below_critical_volume_cm3": s3000[
            "organoid_below_critical_volume_cm3"
        ],
        "synthetic_3000um_organoid_volume_cm3": s3000["organoid_volume_cm3"],
        "synthetic_3000um_min_c_mol_per_cm3": s3000[
            "min_concentration_mol_per_cm3"
        ],
        "synthetic_3000um_consumption_rate_mol_per_s": s3000[
            "consumption_rate_mol_per_s"
        ],
        "synthetic_3000um_injection_rate_mol_per_s": s3000[
            "injection_rate_mol_per_s"
        ],
        "synthetic_3000um_outflow_rate_mol_per_s": s3000[
            "outflow_rate_mol_per_s"
        ],
        "synthetic_benefit_abs_fraction": benefit_abs,
        "synthetic_benefit_rel_fraction": benefit_rel,
        "combo_dir": str(combo_dir),
    }


def screened_out_row(
    *,
    km: float,
    vmax: float,
    combo_dir: Path,
    d1000: dict[str, float | str],
    necrotic_threshold: float,
) -> dict[str, float | int | str]:
    """Score row for a pair whose 1000um diffusion-only run has no core."""
    d1000_below = float(d1000["organoid_below_critical_volume_fraction"])
    necrotic_score = min(max(d1000_below / necrotic_threshold, 0.0), 1.0)
    nan = float("nan")
    return {
        "combo": combo_name(km, vmax),
        "km_mol_per_cm3": km,
        "km_uM": km * 1e9,
        "vmax_mol_per_cm3_s": vmax,
        "vmax_mol_per_m3_s": vmax * 1e6,
        "pass_all": 0,
        "pass_necrotic_1000um": 0,
        "pass_synthetic_benefit_3000um": 0,
        "ran_3000um_cases": 0,
        "screened_out_no_1000um_critical_volume": 1,
        "score": 0.4 * necrotic_score,
        "diffusion_1000um_organoid_below_critical_fraction": d1000_below,
        "diffusion_1000um_organoid_below_critical_volume_cm3": d1000[
            "organoid_below_critical_volume_cm3"
        ],
        "diffusion_1000um_organoid_volume_cm3": d1000["organoid_volume_cm3"],
        "diffusion_1000um_min_c_mol_per_cm3": d1000[
            "min_concentration_mol_per_cm3"
        ],
        "diffusion_1000um_consumption_rate_mol_per_s": d1000[
            "consumption_rate_mol_per_s"
        ],
        "diffusion_3000um_organoid_below_critical_fraction": nan,
        "diffusion_3000um_organoid_below_critical_volume_cm3": nan,
        "diffusion_3000um_organoid_volume_cm3": nan,
        "diffusion_3000um_min_c_mol_per_cm3": nan,
        "diffusion_3000um_consumption_rate_mol_per_s": nan,
        "synthetic_3000um_organoid_below_critical_fraction": nan,
        "synthetic_3000um_organoid_below_critical_volume_cm3": nan,
        "synthetic_3000um_organoid_volume_cm3": nan,
        "synthetic_3000um_min_c_mol_per_cm3": nan,
        "synthetic_3000um_consumption_rate_mol_per_s": nan,
        "synthetic_3000um_injection_rate_mol_per_s": nan,
        "synthetic_3000um_outflow_rate_mol_per_s": nan,
        "synthetic_benefit_abs_fraction": nan,
        "synthetic_benefit_rel_fraction": nan,
        "combo_dir": str(combo_dir),
    }


def write_scores(rows: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(
        rows,
        key=lambda r: (int(r["pass_all"]), float(r["score"])),
        reverse=True,
    )
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run Km/Vmax sweep and score necrotic-core/synthetic-benefit metrics."
    )
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--km-values",
        default="3e-9,5e-9,1e-8,2e-8",
        help="Comma-separated Km values in mol/cm^3.",
    )
    parser.add_argument(
        "--vmax-values",
        default="3e-9, 4e-9, 5e-9, 6.9e-9",
        help="Comma-separated Vmax values in mol/(cm^3 s).",
    )
    parser.add_argument("--T", type=float, default=5000.0)
    parser.add_argument("--dt", type=float, default=50.0)
    parser.add_argument(
        "--output-format",
        choices=("xdmf", "vtk", "both"),
        default="xdmf",
        help="Forwarded to 3d_transport.py. Use vtk for lighter visualization output.",
    )
    parser.add_argument(
        "--necrotic-threshold",
        type=float,
        default=0.05,
        help="Minimum 1000um diffusion-only below-critical organoid fraction.",
    )
    parser.add_argument(
        "--benefit-threshold",
        type=float,
        default=0.25,
        help="Minimum relative reduction in 3000um below-critical fraction from synthetic flow.",
    )
    parser.add_argument(
        "--critical-volume-tol",
        type=float,
        default=0.0,
        help=(
            "Only run the 3000um cases when the 1000um diffusion-only "
            "organoid below-critical volume is greater than this tolerance "
            "in cm^3. Default requires strictly nonzero volume."
        ),
    )
    parser.add_argument(
        "--score-file",
        type=Path,
        default=None,
        help="CSV path for scores. Defaults to OUTPUT_ROOT/sweep_scores.csv.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun cases even when their critical summary JSON already exists.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Do not run simulations; score existing summaries only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running simulations.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going if one case fails; failed combos will be omitted from scores.",
    )
    parser.add_argument(
        "--disable-coupled-1d-transport",
        action="store_true",
        help="For synthetic case, omit --coupled-1d-transport.",
    )
    parser.add_argument(
        "--always-run-3000um",
        action="store_true",
        help="Disable the 1000um critical-volume prescreen and run all cases.",
    )
    args = parser.parse_args(argv)

    km_values = parse_float_list(args.km_values)
    vmax_values = parse_float_list(args.vmax_values)
    score_file = args.score_file or (args.output_root / "sweep_scores.csv")

    rows: list[dict[str, float | int | str]] = []
    failures: list[str] = []

    for km in km_values:
        for vmax in vmax_values:
            combo_dir = args.output_root / combo_name(km, vmax)
            print(f"\n=== {combo_dir.name} ===", flush=True)

            combo_failed = False
            screened_out = False
            for case in CASES:
                case_dir = combo_dir / case.name
                summary = final_summary_path(case_dir, case)
                if args.score_only:
                    if case.name == "diffusion_1000um":
                        break
                    continue
                if summary.exists() and not args.force:
                    print(f"[skip] {case.name}: {summary}", flush=True)
                else:
                    case_dir.mkdir(parents=True, exist_ok=True)
                    cmd = build_command(
                        python=args.python,
                        case=case,
                        case_dir=case_dir,
                        km=km,
                        vmax=vmax,
                        T=args.T,
                        dt=args.dt,
                        output_format=args.output_format,
                        coupled_1d_transport=not args.disable_coupled_1d_transport,
                    )
                    print("[run] " + " ".join(cmd), flush=True)
                    if args.dry_run:
                        continue
                    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
                    if proc.returncode != 0:
                        msg = f"{combo_dir.name}/{case.name} failed with code {proc.returncode}"
                        failures.append(msg)
                        combo_failed = True
                        print(f"[failed] {msg}", flush=True)
                        if not args.continue_on_error:
                            return proc.returncode
                        break

                if args.dry_run:
                    continue

                if case.name == "diffusion_1000um" and not args.always_run_3000um:
                    d1000 = load_final(summary)
                    critical_volume = float(d1000["organoid_below_critical_volume_cm3"])
                    if critical_volume <= args.critical_volume_tol:
                        row = screened_out_row(
                            km=km,
                            vmax=vmax,
                            combo_dir=combo_dir,
                            d1000=d1000,
                            necrotic_threshold=args.necrotic_threshold,
                        )
                        rows.append(row)
                        write_scores(rows, score_file)
                        screened_out = True
                        print(
                            "[screened-out] 1000um diffusion-only "
                            f"critical organoid volume={critical_volume:.6e} cm^3; "
                            "skipping both 3000um cases",
                            flush=True,
                        )
                        print(f"[scores] {score_file}", flush=True)
                        break

            if args.score_only and not args.dry_run and not args.always_run_3000um:
                try:
                    d1000_summary = final_summary_path(
                        combo_dir / CASES[0].name, CASES[0]
                    )
                    d1000 = load_final(d1000_summary)
                    critical_volume = float(d1000["organoid_below_critical_volume_cm3"])
                    if critical_volume <= args.critical_volume_tol:
                        row = screened_out_row(
                            km=km,
                            vmax=vmax,
                            combo_dir=combo_dir,
                            d1000=d1000,
                            necrotic_threshold=args.necrotic_threshold,
                        )
                        rows.append(row)
                        write_scores(rows, score_file)
                        screened_out = True
                        print(
                            "[screened-out] 1000um diffusion-only "
                            f"critical organoid volume={critical_volume:.6e} cm^3; "
                            "3000um summaries not required for scoring",
                            flush=True,
                        )
                except FileNotFoundError as exc:
                    msg = f"{combo_dir.name}: missing 1000um prescreen summary: {exc}"
                    failures.append(msg)
                    print(f"[missing] {msg}", flush=True)
                    if not args.continue_on_error:
                        return 2

            if screened_out:
                continue

            if args.dry_run:
                continue
            if combo_failed:
                continue
            try:
                row = score_combo(
                    km=km,
                    vmax=vmax,
                    combo_dir=combo_dir,
                    necrotic_threshold=args.necrotic_threshold,
                    benefit_threshold=args.benefit_threshold,
                )
            except FileNotFoundError as exc:
                msg = f"{combo_dir.name}: missing summary for scoring: {exc}"
                failures.append(msg)
                print(f"[missing] {msg}", flush=True)
                if not args.continue_on_error:
                    return 2
                continue
            rows.append(row)
            write_scores(rows, score_file)
            print(
                "[score] "
                f"pass_all={row['pass_all']} "
                f"score={float(row['score']):.3f} "
                f"d1000_below={float(row['diffusion_1000um_organoid_below_critical_fraction']):.3f} "
                f"benefit={float(row['synthetic_benefit_rel_fraction']):.3f}",
                flush=True,
            )
            print(f"[scores] {score_file}", flush=True)

    if rows:
        write_scores(rows, score_file)
        print(f"\nWrote scores: {score_file}", flush=True)
    if failures:
        print("\nFailures:", flush=True)
        for item in failures:
            print(f"  - {item}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
