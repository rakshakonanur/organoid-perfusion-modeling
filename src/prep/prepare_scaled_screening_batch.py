#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1]


def _run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the repeated-geometry scaled screening pipeline for every "
            "immediate subfolder under a source root."
        )
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--trial-root", type=Path, default=REPO_SRC / "prep" / "prepped")
    parser.add_argument("--prepared-root", type=Path, default=REPO_SRC / "prep" / "scaled-screening")
    parser.add_argument("--driver-script", type=Path, default=REPO_SRC / "prep" / "prepare_scaled_screening_trial.py")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--candidate-pattern", default="Run*")
    parser.add_argument("--start-at", default="")
    parser.add_argument("--stop-after", type=int, default=0)
    parser.add_argument("--channel-xlsx", type=Path, default=REPO_SRC.parent / "files" / "excel" / "channel-resistance-sv.xlsx")
    parser.add_argument("--svzerodsolver", default="svzerodsolver")
    parser.add_argument("--n-organoids", type=int, default=4)
    parser.add_argument("--dy-step", type=float, default=0.6)
    parser.add_argument("--shift-sign", type=float, default=-1.0)
    parser.add_argument("--stl-dir", type=Path, default=REPO_SRC.parent / "files")
    parser.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    parser.add_argument("--lp-arterial", type=float, default=1e-9)
    parser.add_argument("--lp-venous", type=float, default=1e-9)
    parser.add_argument("--darcy-mpi-procs", type=int, default=1)
    parser.add_argument("--darcy-mpirun-cmd", default="mpirun")
    parser.add_argument("--darcy-solver", choices=["darcy", "darcy_mixed", "darcy_p1_lm", "darcy_p1_lm_interior"], default="darcy_mixed")
    parser.add_argument("--perm-region-root", default=str(REPO_SRC.parent / "files" / "stl" / "organoid-growth-domains" / "sphere"))
    parser.add_argument("--perm-low", type=float, default=1.0e-9)
    parser.add_argument("--perm-high", type=float, default=2.0e-7)
    parser.add_argument("--perm-transition-width", type=float, default=0.01)
    parser.add_argument("--first-organoid-linear-profile", dest="first_organoid_linear_profile", action="store_true", default=True)
    parser.add_argument("--no-first-organoid-linear-profile", dest="first_organoid_linear_profile", action="store_false")
    parser.add_argument("--concave-profile-face-length", type=float, default=0.4)
    parser.add_argument("--concave-profile-compartment-spacing", type=float, default=0.6)
    parser.add_argument("--coords-inlet", nargs=3, type=float, default=[-0.28, 0.9, 0.5375])
    parser.add_argument("--coords-outlet", nargs=3, type=float, default=[0.30, 0.9, 0.5375])
    parser.add_argument("--y-shift-step", type=float, default=0.6)
    parser.add_argument("--pressure-offset-anchor", choices=("arterial", "venous"), default="arterial")
    parser.add_argument("--no-shift-coords", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    driver_script = args.driver_script.expanduser().resolve()

    candidates = [p for p in sorted(source_root.glob(args.candidate_pattern)) if p.is_dir()]
    if args.start_at:
        candidates = [p for p in candidates if p.name >= args.start_at]
    if args.stop_after > 0:
        candidates = candidates[: int(args.stop_after)]

    if not candidates:
        raise SystemExit(f"No candidate subfolders found in {source_root} matching {args.candidate_pattern!r}")

    for idx, candidate in enumerate(candidates, start=1):
        trial_name = candidate.name
        prepared_name = candidate.name
        print(f"[batch] ({idx}/{len(candidates)}) {candidate.name}", flush=True)
        cmd = [
            sys.executable,
            str(driver_script),
            "--source-folder",
            str(candidate),
            "--trial-root",
            str(args.trial_root.expanduser().resolve()),
            "--trial-name",
            trial_name,
            "--prepared-root",
            str(args.prepared_root.expanduser().resolve()),
            "--prepared-name",
            prepared_name,
            "--channel-xlsx",
            str(args.channel_xlsx.expanduser().resolve()),
            "--svzerodsolver",
            str(args.svzerodsolver),
            "--n-organoids",
            str(int(args.n_organoids)),
            "--dy-step",
            str(float(args.dy_step)),
            "--shift-sign",
            str(float(args.shift_sign)),
            "--stl-dir",
            str(args.stl_dir.expanduser().resolve()),
            "--concave-bc-mode",
            str(args.concave_bc_mode),
            "--lp-arterial",
            str(float(args.lp_arterial)),
            "--lp-venous",
            str(float(args.lp_venous)),
            "--darcy-mpi-procs",
            str(int(args.darcy_mpi_procs)),
            "--darcy-mpirun-cmd",
            str(args.darcy_mpirun_cmd),
            "--darcy-solver",
            str(args.darcy_solver),
            "--perm-region-root",
            str(args.perm_region_root),
            "--perm-low",
            str(float(args.perm_low)),
            "--perm-high",
            str(float(args.perm_high)),
            "--perm-transition-width",
            str(float(args.perm_transition_width)),
            "--concave-profile-face-length",
            str(float(args.concave_profile_face_length)),
            "--concave-profile-compartment-spacing",
            str(float(args.concave_profile_compartment_spacing)),
            "--coords-inlet",
            *(str(v) for v in args.coords_inlet),
            "--coords-outlet",
            *(str(v) for v in args.coords_outlet),
            "--y-shift-step",
            str(float(args.y_shift_step)),
            "--pressure-offset-anchor",
            str(args.pressure_offset_anchor),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.no_shift_coords:
            cmd.append("--no-shift-coords")
        if not args.first_organoid_linear_profile:
            cmd.append("--no-first-organoid-linear-profile")
        _run_cmd(cmd)


if __name__ == "__main__":
    main()
