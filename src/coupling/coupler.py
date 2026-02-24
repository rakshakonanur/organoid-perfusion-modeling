#!/usr/bin/env python3
"""coupler_4organoid.py

Pseudotimestep coupler for the *combined* 0D file (combined.in) + 4-organoid Darcy.

Loop (fixed-point style):
  run_0:
    - copy template combined.in -> coupled/run_0/combined.in
    - set channel inlet FLOW to 0 and channel outlet PRESSURE to P0
    - (optional) zero organoid outlet FLOWs
    - run svZeroDSolver + split/copy/plot (via run_and_split_svzerod.py)
    - run 3D meshing + Darcy for all organoids (via run_all.py)

  for i = 1..N:
    - copy coupled/run_{i-1} -> coupled/run_i
    - update coupled/run_i/combined.in:
        * ramp channel inlet flow: Q = (i/N)*Q0
        * ramp channel outlet pressure: P = (1 - i/N)*P0   (drops to 0)
        * for each organoid k=1..n:
            - read coupled/run_{i-1}/organoid_k/out_darcy/interface_bc.json
            - set organoid{k}_inlet_OUT*  PRESSURE to p_inlet_nodes
            - set organoid{k}_outlet_OUT* FLOW     to q_outlet
    - run svZeroDSolver + split/copy/plot
    - run run_all.py (mesh + darcy) to produce new interface_bc.json for this run

This script does NOT copy mesh.py or darcy_P1_v2.py; it only drives existing scripts.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(2)


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"[run] {' '.join(map(str, cmd))}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def copy_tree(src: Path, dst: Path, overwrite: bool = True) -> None:
    if dst.exists() and overwrite:
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=not overwrite)


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, obj: dict) -> None:
    with path.open("w") as f:
        json.dump(obj, f, indent=4)


def _bc_value_len(bc: dict, key: str) -> int:
    vals = bc.get("bc_values", {})
    arr = vals.get(key, None)
    if isinstance(arr, list) and len(arr) > 0:
        return len(arr)
    t = vals.get("t", None)
    if isinstance(t, list) and len(t) > 0:
        return len(t)
    return 1


def ensure_flow_bc(bc: dict) -> None:
    bc["bc_type"] = "FLOW"
    vals = bc.setdefault("bc_values", {})
    t = vals.get("t", None)
    bc["bc_values"] = {"t": t} if t is not None else {}


def ensure_pressure_bc(bc: dict) -> None:
    bc["bc_type"] = "PRESSURE"
    vals = bc.setdefault("bc_values", {})
    t = vals.get("t", None)
    bc["bc_values"] = {"t": t} if t is not None else {}


def set_bc_constant(bc: dict, key: str, value: float) -> None:
    n = _bc_value_len(bc, key)
    vals = bc.setdefault("bc_values", {})
    vals[key] = [float(value) for _ in range(n)]


def find_bc(deck: dict, bc_name: str) -> Optional[dict]:
    for bc in deck.get("boundary_conditions", []):
        if str(bc.get("bc_name", "")) == bc_name:
            return bc
    return None


def find_bcs_with_prefix(deck: dict, prefix: str) -> List[dict]:
    out = []
    for bc in deck.get("boundary_conditions", []):
        name = str(bc.get("bc_name", ""))
        if name.startswith(prefix):
            out.append(bc)
    return out


def infer_base_inflow_Q(deck: dict, inflow_name: str) -> Optional[float]:
    bc = find_bc(deck, inflow_name)
    if not bc:
        return None
    if bc.get("bc_type") != "FLOW":
        return None
    q = bc.get("bc_values", {}).get("Q", [])
    if not q:
        return None
    return float(q[0])


def infer_base_outlet_P(deck: dict, outlet_name: str) -> Optional[float]:
    bc = find_bc(deck, outlet_name)
    if not bc:
        return None
    if bc.get("bc_type") != "PRESSURE":
        return None
    p = bc.get("bc_values", {}).get("P", [])
    if not p:
        return None
    return float(p[0])


def apply_channel_ramp(deck: dict, inflow_name: str, outlet_name: str, Q0: float, P0: float, scale: float) -> None:
    inflow = find_bc(deck, inflow_name)
    if inflow is None:
        die(f"Missing channel inlet bc_name='{inflow_name}' in combined deck")
    ensure_flow_bc(inflow)
    set_bc_constant(inflow, "Q", scale * Q0)

    # out = find_bc(deck, outlet_name)
    # if out is None:
    #     die(f"Missing channel outlet bc_name='{outlet_name}' in combined deck")
    # ensure_pressure_bc(out)
    # if scale > 0:
    #     set_bc_constant(out, "P", (1.0 - scale) * P0)
    # else:
    #     set_bc_constant(out, "P", 0)


def apply_organoid_interface_from_json(deck: dict, organoid_idx: int, iface: dict) -> None:
    p_in = iface.get("p_inlet_nodes", [])
    p_out = iface.get("p_outlet_nodes", [])
    artery_leak = iface.get("q_artery_leak", None)
    vein_leak = iface.get("q_venous_leak", None)

    if not isinstance(p_in, list) or not isinstance(p_out, list):
        die(f"interface_bc.json for organoid_{organoid_idx} missing p_inlet_nodes or p_outlet_nodes")

    in_prefix = f"organoid{organoid_idx}_inlet_OUT"
    out_prefix = f"organoid{organoid_idx}_outlet_OUT"
    leak_in_prefix = f"LEAK_ART_{organoid_idx}"
    leak_out_prefix = f"LEAK_VEN_{organoid_idx}"

    inlet_bcs = find_bcs_with_prefix(deck, in_prefix)
    outlet_bcs = find_bcs_with_prefix(deck, out_prefix)
    artery_leak_bcs = find_bcs_with_prefix(deck, leak_in_prefix)
    vein_leak_bcs = find_bcs_with_prefix(deck, leak_out_prefix)

    if len(inlet_bcs) == 0:
        die(f"No BCs found with prefix '{in_prefix}'")
    if len(outlet_bcs) == 0:
        die(f"No BCs found with prefix '{out_prefix}'")

    m = min(len(inlet_bcs), len(p_in))
    if len(inlet_bcs) != len(p_in):
        print(f"[warn] organoid_{organoid_idx} inlet BC count {len(inlet_bcs)} != p_inlet_nodes {len(p_in)}; using {m}")
    for j in range(m):
        ensure_pressure_bc(inlet_bcs[j])
        set_bc_constant(inlet_bcs[j], "P", float(p_in[j]))

    m = min(len(outlet_bcs), len(p_out))
    if len(outlet_bcs) != len(p_out):
        print(f"[warn] organoid_{organoid_idx} outlet BC count {len(outlet_bcs)} != p_outlet_nodes {len(p_out)}; using {m}")
    for j in range(m):
        ensure_pressure_bc(outlet_bcs[j])
        set_bc_constant(outlet_bcs[j], "P", float(p_out[j]))

    ensure_flow_bc(artery_leak_bcs[0])
    set_bc_constant(artery_leak_bcs[0], "Q", float(-artery_leak))

    ensure_flow_bc(vein_leak_bcs[0])
    set_bc_constant(vein_leak_bcs[0], "Q", float(-vein_leak))


def maybe_zero_all_organoid_outlet_flows(deck: dict, n_organoids: int) -> None:
    for k in range(1, n_organoids + 1):
        out_prefix = f"organoid{k}_outlet_OUT"
        outlet_bcs = find_bcs_with_prefix(deck, out_prefix)
        print(f"[info] zeroing organoid{k} outlet BCs: {len(outlet_bcs)} found")
        for bc in outlet_bcs:
            ensure_pressure_bc(bc)
            set_bc_constant(bc, "P", 0.0)
        in_prefix = f"organoid{k}_inlet_OUT"
        inlet_bcs = find_bcs_with_prefix(deck, in_prefix)
        print(f"[info] setting organoid{k} inlet BCs to PRESSURE: {len(inlet_bcs)} found")
        for bc in inlet_bcs:
            ensure_pressure_bc(bc)
            set_bc_constant(bc, "P", 0.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="4-organoid pseudotimestep coupler (combined.in + Darcy)")

    ap.add_argument("--template-combined", default="./combined.in", help="Path to template combined.in (JSON) to start from")
    ap.add_argument("--coupled-root", default="coupled", help="Root output directory (default: coupled)")

    ap.add_argument("--pseudo-steps", type=int, default=5, help="Number of pseudotimesteps (default: 20)")
    ap.add_argument("--n-organoids", type=int, default=4, help="Number of organoids (default: 4)")

    ap.add_argument("--channel-inlet-bc", default="INFLOW", help="bc_name for channel inlet (default: INFLOW)")
    ap.add_argument("--channel-outlet-bc", default="OUT3", help="bc_name for channel outlet (default: OUT3)")

    ap.add_argument("--channel-outlet-P0", type=float, default= 0, help="Starting channel outlet pressure (if omitted, inferred from template)")
    ap.add_argument("--channel-inlet-Q0", type=float, default= 2.5e-2, help="Target channel inlet flow (if omitted, inferred if template uses FLOW)")

    ap.add_argument("--zero-organoid-outlet-flows-at-run0", action="store_true",
                    help="At run_0, set all organoid outlet BCs to FLOW with Q=0")

    ap.add_argument("--svzerodsolver", default="svzerodsolver", help="svZeroDSolver executable")
    ap.add_argument("--run-and-split", default="./run_and_split_svzerod.py", help="Path to run_and_split_svzerod.py")
    ap.add_argument("--run-all", default="./run_all_v2.py", help="Path to run_all.py (mesh + darcy for all organoids)")

    ap.add_argument("--stl-dir", default="/Users/rakshakonanur/Documents/Research/two-channel/files", help="Directory containing organoid-1.stl, organoid-2.stl, ...")
    ap.add_argument("--trial-dir", default="/Users/rakshakonanur/Documents/Research/two-channel/src/coupled/run_0", help="Directory containing organoid_1/0D_Input_Files etc")
    ap.add_argument("--dy-step", type=float, default=0.6, help="Y shift per organoid index (default: 0.6)")
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-0.175, 0.9, 0.55],
                    help="Base inlet coords (x y z) for organoid 1")
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.175, 0.9, 0.55],
                    help="Base outlet coords (x y z) for organoid 1")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite coupled/run_* directories if they exist")
    ap.add_argument("--debug", action="store_true", help="Verbose prints")

    return ap.parse_args()


def main() -> None:
    args = parse_args()

    template = Path(args.template_combined).expanduser().resolve()
    if not template.is_file():
        die(f"Template combined.in not found: {template}")

    coupled_root = Path(args.coupled_root).expanduser().resolve()
    coupled_root.mkdir(parents=True, exist_ok=True)

    # --- build run_0 fresh ---
    run0 = coupled_root / "run_0"
    if run0.exists() and args.overwrite:
        shutil.rmtree(run0)
    run0.mkdir(parents=True, exist_ok=True)

    combined0 = run0 / "combined.in"
    shutil.copy2(template, combined0)

    deck0 = load_json(combined0)

    Q0 = args.channel_inlet_Q0
    if Q0 is None:
        Q0 = infer_base_inflow_Q(deck0, args.channel_inlet_bc)
    if Q0 is None:
        die("Could not infer channel inlet Q0 (template INFLOW is not FLOW). Provide --channel-inlet-Q0.")

    P0 = args.channel_outlet_P0
    # if P0 is None:
    #     P0 = infer_base_inflow_Q(deck0, args.channel_outlet_bc)
    # if P0 is None:
    #     die("Could not infer channel outlet P0 (template outlet is not PRESSURE). Provide --channel-outlet-P0.")

    apply_channel_ramp(deck0, args.channel_inlet_bc, args.channel_outlet_bc, Q0=Q0, P0=P0, scale=0.0)
    maybe_zero_all_organoid_outlet_flows(deck0, args.n_organoids)
    save_json(combined0, deck0)

    # --- run 0D + split/copy/plot into run_0/organoid_* ---
    run([
        sys.executable, str(Path(args.run_and_split).resolve()),
        "--exe", args.svzerodsolver,
        "--input", str(combined0),
        "--outdir", str(run0),
        "--output", "output.csv",
        "--organoid-root", str(run0),
    ] + (["--debug"] if args.debug else []))

    # --- run mesh+darcy for all organoids to generate interface_bc.json in run_0 ---
    run([
        sys.executable, str(Path(args.run_all).resolve()),
        "--stl-dir", str(Path(args.stl_dir).expanduser().resolve()),
        "--trial-dir", str(Path(args.trial_dir).expanduser().resolve()),
        "--coupled-root", str(run0),
        "--n", str(args.n_organoids),
        "--dy-step", str(args.dy_step),
        "--coords-inlet", *map(str, args.coords_inlet),
        "--coords-outlet", *map(str, args.coords_outlet),
    ])

    # --- main pseudotimestep loop ---
    N = int(args.pseudo_steps)
    if N < 1:
        die("--pseudo-steps must be >= 1")

    for i in range(1, N + 1):
        scale = min(i / float(N), 1.0)
        prev = coupled_root / f"run_{i-1}"
        cur = coupled_root / f"run_{i}"

        if cur.exists() and args.overwrite:
            shutil.rmtree(cur)
        print(f"\n=== Pseudotimestep {i}/{N}  scale={scale:.6g} ===", flush=True)
        copy_tree(prev, cur, overwrite=not args.overwrite)

        combined_i = cur / "combined.in"
        if not combined_i.exists():
            die(f"Missing {combined_i} after copying")

        deck = load_json(combined_i)

        apply_channel_ramp(deck, args.channel_inlet_bc, args.channel_outlet_bc, Q0=Q0, P0=P0, scale=scale)

        for k in range(1, args.n_organoids + 1):
            iface_path = prev / f"organoid_{k}" / "out_darcy" / "interface_bc.json"
            if not iface_path.is_file():
                die(f"Missing interface_bc.json for organoid_{k}: {iface_path}")
            iface = load_json(iface_path)
            apply_organoid_interface_from_json(deck, organoid_idx=k, iface=iface)

        save_json(combined_i, deck)

        run([
            sys.executable, str(Path(args.run_and_split).resolve()),
            "--exe", args.svzerodsolver,
            "--input", str(combined_i),
            "--outdir", str(cur),
            "--output", "output.csv",
            "--organoid-root", str(cur),
        ] + (["--debug"] if args.debug else []))

        run([
            sys.executable, str(Path(args.run_all).resolve()),
            "--stl-dir", str(Path(args.stl_dir).expanduser().resolve()),
            "--trial-dir", str(Path(args.trial_dir).expanduser().resolve()),
            "--coupled-root", str(cur),
            "--n", str(args.n_organoids),
            "--dy-step", str(args.dy_step),
            "--coords-inlet", *map(str, args.coords_inlet),
            "--coords-outlet", *map(str, args.coords_outlet),
        ])

    print("\n[done] 4-organoid pseudotimestep coupling complete.")

    N = int(args.pseudo_steps) + 10
    for i in range(int(args.pseudo_steps) + 1, N + 1):
        scale = min(i / float(N), 1.0)
        prev = coupled_root / f"run_{i-1}"
        cur = coupled_root / f"run_{i}"

        if cur.exists() and args.overwrite:
            shutil.rmtree(cur)
        print(f"\n=== Pseudotimestep {i}/{N}  scale={scale:.6g} ===", flush=True)
        copy_tree(prev, cur, overwrite=not args.overwrite)

        combined_i = cur / "combined.in"
        if not combined_i.exists():
            die(f"Missing {combined_i} after copying")

        deck = load_json(combined_i)

        for k in range(1, args.n_organoids + 1):
            iface_path = prev / f"organoid_{k}" / "out_darcy" / "interface_bc.json"
            if not iface_path.is_file():
                die(f"Missing interface_bc.json for organoid_{k}: {iface_path}")
            iface = load_json(iface_path)
            apply_organoid_interface_from_json(deck, organoid_idx=k, iface=iface)

        save_json(combined_i, deck)

        run([
            sys.executable, str(Path(args.run_and_split).resolve()),
            "--exe", args.svzerodsolver,
            "--input", str(combined_i),
            "--outdir", str(cur),
            "--output", "output.csv",
            "--organoid-root", str(cur),
        ] + (["--debug"] if args.debug else []))

        run([
            sys.executable, str(Path(args.run_all).resolve()),
            "--stl-dir", str(Path(args.stl_dir).expanduser().resolve()),
            "--trial-dir", str(Path(args.trial_dir).expanduser().resolve()),
            "--coupled-root", str(cur),
            "--n", str(args.n_organoids),
            "--dy-step", str(args.dy_step),
            "--coords-inlet", *map(str, args.coords_inlet),
            "--coords-outlet", *map(str, args.coords_outlet),
        ])


if __name__ == "__main__":
    main()
