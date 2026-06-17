from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from coupling.steady_0d_pressure_control import SteadyResistiveZeroDModel


def _load_targets_and_overrides(summary_path: Path) -> tuple[dict[str, float], dict[str, float]]:
    targets: dict[str, float] = {}
    overrides: dict[str, float] = {}
    with open(summary_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            bc = row["bc_name"]
            raw = bc.split("organoid1_outlet_", 1)[1] if bc.startswith("organoid1_outlet_") else bc
            targets[raw] = float(row["P_interface"])
            overrides[raw] = float(row["R_next"])
    return targets, overrides


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a deck with terminal Pd values from the strict circuit solve.")
    ap.add_argument("solver_file", type=Path)
    ap.add_argument("summary_csv", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = SteadyResistiveZeroDModel.from_solver_file(args.solver_file)
    targets, overrides = _load_targets_and_overrides(args.summary_csv)
    pd, mode, _, residual = model.solve_terminal_pd_direct(
        targets=targets,
        terminal_resistance_overrides=overrides,
    )

    data = json.load(open(args.solver_file, "r"))
    pd_by_bc = {f"organoid1_outlet_{k}": v for k, v in pd.items()}
    for row in data.get("boundary_conditions", []):
        if row.get("bc_type", "").upper() != "RESISTANCE":
            continue
        name = row.get("bc_name")
        if name in pd_by_bc:
            row.setdefault("bc_values", {})["Pd"] = float(pd_by_bc[name])

    args.out.write_text(json.dumps(data, indent=2))
    print(json.dumps({"mode": mode, "residual_l2": float((residual**2).sum() ** 0.5), "residual_linf": float(abs(residual).max())}, indent=2, sort_keys=True))
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
