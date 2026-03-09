#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
import csv
import os
from pathlib import Path
from typing import List, Optional

import numpy as np


def die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(2)


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"[run] {' '.join(map(str, cmd))}  (cwd={cwd})", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def resolve_mpi_launcher(user_cmd: str) -> str:
    """
    Prefer MPI launcher binaries from active Python environment when a generic
    launcher name is provided, to avoid MPI runtime mismatches.
    """
    generic = {"mpirun", "mpiexec"}
    if user_cmd:
        base = Path(user_cmd).name
        if base not in generic:
            return user_cmd

    env_bin = Path(sys.executable).resolve().parent
    candidates = [env_bin / "mpiexec.hydra", env_bin / "mpiexec", env_bin / "mpirun"]
    for cand in candidates:
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return user_cmd


def copy_tree(src: Path, dst: Path, overwrite: bool = True) -> None:
    if overwrite and dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=not overwrite)


def copy_xdmf_with_sidecars(src_xdmf: Path, dst_xdmf: Path) -> None:
    """
    Copy XDMF and referenced .h5 sidecars, rewriting HDF5 references so they
    match the destination stem (e.g. bioreactor.xdmf -> bioreactor.h5).
    """
    text = src_xdmf.read_text(encoding="utf-8", errors="ignore")
    h5_names = sorted(set(re.findall(r">([^<]+\.h5):", text)))
    if not h5_names:
        if src_xdmf.resolve() != dst_xdmf.resolve():
            shutil.copy2(src_xdmf, dst_xdmf)
        return

    if len(h5_names) == 1:
        mapping = {h5_names[0]: f"{dst_xdmf.stem}.h5"}
    else:
        mapping = {old: f"{dst_xdmf.stem}_{k}.h5" for k, old in enumerate(h5_names, start=1)}

    out_text = text
    for old, new in mapping.items():
        out_text = out_text.replace(f">{old}:", f">{new}:")
    dst_xdmf.write_text(out_text, encoding="utf-8")

    for h5_name in h5_names:
        src_h5 = src_xdmf.parent / h5_name
        if not src_h5.exists():
            raise FileNotFoundError(
                f"XDMF {src_xdmf} references missing HDF5 sidecar {src_h5}"
            )
        dst_h5 = dst_xdmf.parent / mapping[h5_name]
        if src_h5.resolve() == dst_h5.resolve():
            continue
        shutil.copy2(src_h5, dst_h5)


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
    q = vals.get("Q", None)
    new_vals = {"t": t} if t is not None else {}
    if q is not None:
        new_vals["Q"] = q
    bc["bc_values"] = new_vals


def ensure_pressure_bc(bc: dict) -> None:
    bc["bc_type"] = "PRESSURE"
    vals = bc.setdefault("bc_values", {})
    t = vals.get("t", None)
    p = vals.get("P", None)
    new_vals = {"t": t} if t is not None else {}
    if p is not None:
        new_vals["P"] = p
    bc["bc_values"] = new_vals


def set_bc_constant(bc: dict, key: str, value: float) -> None:
    n = _bc_value_len(bc, key)
    vals = bc.setdefault("bc_values", {})
    vals[key] = [float(value) for _ in range(n)]


def set_bc_relaxed(bc: dict, key: str, target: float, w: float) -> None:
    vals = bc.setdefault("bc_values", {})
    cur = 0.0
    if isinstance(vals.get(key, None), list) and len(vals[key]) > 0:
        cur = float(vals[key][0])
    new_val = (1.0 - w) * cur + w * float(target)
    set_bc_constant(bc, key, new_val)


def find_bc(deck: dict, bc_name: str) -> Optional[dict]:
    for bc in deck.get("boundary_conditions", []):
        if str(bc.get("bc_name", "")) == bc_name:
            return bc
    return None


def find_bcs_with_prefix(deck: dict, prefix: str) -> List[dict]:
    out = []
    for bc in deck.get("boundary_conditions", []):
        if str(bc.get("bc_name", "")).startswith(prefix):
            out.append(bc)
    return out


def apply_channel_ramp(
    deck: dict,
    inflow_name: str,
    Q0: float,
    scale: float,
    outlet_name: Optional[str] = None,
    outlet_P0: float = 0.0,
) -> None:
    inflow = find_bc(deck, inflow_name)
    if inflow is None:
        die(f"Missing channel inlet bc_name='{inflow_name}' in combined deck")
    ensure_flow_bc(inflow)
    set_bc_constant(inflow, "Q", scale * Q0)

    if outlet_name:
        out = find_bc(deck, outlet_name)
        if out is None:
            die(f"Missing channel outlet bc_name='{outlet_name}' in combined deck")
        ensure_pressure_bc(out)
        set_bc_constant(out, "P", scale * float(outlet_P0))


def zero_all_interface_bcs(deck: dict, n_organoids: int) -> None:
    for k in range(1, n_organoids + 1):
        inlet_bcs = find_bcs_with_prefix(deck, f"organoid{k}_inlet_OUT")
        outlet_bcs = find_bcs_with_prefix(deck, f"organoid{k}_outlet_OUT")
        leak_art = find_bcs_with_prefix(deck, f"LEAK_ART_{k}")
        leak_ven = find_bcs_with_prefix(deck, f"LEAK_VEN_{k}")

        for bc in inlet_bcs:
            ensure_pressure_bc(bc)
            set_bc_constant(bc, "P", 0.0)
        for bc in outlet_bcs:
            ensure_flow_bc(bc)
            set_bc_constant(bc, "Q", 0.0)
        for bc in leak_art:
            ensure_flow_bc(bc)
            set_bc_constant(bc, "Q", 0.0)
        for bc in leak_ven:
            ensure_flow_bc(bc)
            set_bc_constant(bc, "Q", 0.0)


def apply_organoid_interface_from_json(
    deck: dict,
    organoid_idx: int,
    iface: dict,
    relax: float,
) -> None:
    p_in = iface.get("p_inlet_nodes", [])
    q_out = iface.get("q_outlet", [])
    artery_leak = float(iface.get("q_artery_leak", 0.0))
    vein_leak = float(iface.get("q_venous_leak", 0.0))

    if not isinstance(p_in, list) or not isinstance(q_out, list):
        die(f"interface_bc.json for organoid_{organoid_idx} missing p_inlet_nodes or p_outlet_nodes")

    inlet_bcs = find_bcs_with_prefix(deck, f"organoid{organoid_idx}_inlet_OUT")
    outlet_bcs = find_bcs_with_prefix(deck, f"organoid{organoid_idx}_outlet_OUT")
    leak_art = find_bcs_with_prefix(deck, f"LEAK_ART_{organoid_idx}")
    leak_ven = find_bcs_with_prefix(deck, f"LEAK_VEN_{organoid_idx}")

    if len(inlet_bcs) == 0 or len(outlet_bcs) == 0:
        die(f"Missing organoid interface BC groups for organoid {organoid_idx}")

    m = min(len(inlet_bcs), len(p_in))
    for j in range(m):
        ensure_pressure_bc(inlet_bcs[j])
        set_bc_relaxed(inlet_bcs[j], "P", float(p_in[j]), relax)

    m = min(len(outlet_bcs), len(q_out))
    for j in range(m):
        ensure_flow_bc(outlet_bcs[j])
        set_bc_relaxed(outlet_bcs[j], "Q", float(-q_out[j]), relax)

    if leak_art:
        ensure_flow_bc(leak_art[0])
        set_bc_relaxed(leak_art[0], "Q", float(-artery_leak), relax)
    if leak_ven:
        ensure_flow_bc(leak_ven[0])
        set_bc_relaxed(leak_ven[0], "Q", float(-vein_leak), relax)


def copy_seed_geometry(seed_run0: Path, run_dir: Path, n_organoids: int) -> None:
    tmp_mesh = seed_run0 / "_tmp_mesh"
    # Reuse run_all.py geometry copier to stay aligned with pipeline behavior.
    repo_src = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_src / "prep"))
    run_all_mod = importlib.import_module("run_all")
    copy_shared_geometry_outputs = getattr(run_all_mod, "_copy_shared_geometry_outputs")

    def _copy_tmp_mesh_well(tmp_dir: Path, dst_geom: Path, k: int) -> None:
        dst_geom.mkdir(parents=True, exist_ok=True)
        # Copy canonical geometry/checkpoint files exactly as run_all.py does.
        copy_shared_geometry_outputs(tmp_dir, dst_geom, well_idx=k)

        # Also copy branched-network XDMFs (+h5 sidecars), needed by generate_1d_files.
        src_in = tmp_dir / f"branched_network_inlet_well{k}.xdmf"
        src_out = tmp_dir / f"branched_network_outlet_well{k}.xdmf"
        if src_in.exists():
            copy_xdmf_with_sidecars(src_in, dst_geom / "branched_network_inlet.xdmf")
        if src_out.exists():
            copy_xdmf_with_sidecars(src_out, dst_geom / "branched_network_outlet.xdmf")

    for k in range(1, n_organoids + 1):
        src = seed_run0 / f"organoid_{k}" / "geometry"
        dst = run_dir / f"organoid_{k}" / "geometry"
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Prefer _tmp_mesh well-specific artifacts when available, since these include
        # the canonical XDMF/HDF5 sidecar pairs for each well.
        if tmp_mesh.exists():
            _copy_tmp_mesh_well(tmp_mesh, dst, k)
        elif src.exists():
            copy_tree(src, dst, overwrite=True)
        else:
            die(f"Missing seed geometry for organoid_{k}: {src} and no fallback {tmp_mesh}")


def ensure_organoid_dirs(run_dir: Path, n_organoids: int) -> None:
    """
    Ensure run_and_split_svzerod.py destination directories exist.
    """
    for k in range(1, n_organoids + 1):
        org = run_dir / f"organoid_{k}"
        (org / "0D_Input_Files" / "inlet").mkdir(parents=True, exist_ok=True)
        (org / "0D_Input_Files" / "outlet").mkdir(parents=True, exist_ok=True)


def sync_plot_templates(seed_run0: Path, run_dir: Path, n_organoids: int) -> None:
    """
    Ensure inlet/outlet folders contain plotting assets required by
    run_and_split_svzerod.py -> plot_0d_results_to_3d.py.
    """
    required = [
        "plot_0d_results_to_3d.py",
        "plot_0d_results_at_slices.py",
        "geom.csv",
        "inflow.flow",
        "run.py",
        "solver_0d.in",
        "solver_0d_new.in",
    ]
    for k in range(1, n_organoids + 1):
        for side in ("inlet", "outlet"):
            src_dir = seed_run0 / f"organoid_{k}" / "0D_Input_Files" / side
            dst_dir = run_dir / f"organoid_{k}" / "0D_Input_Files" / side
            if not src_dir.exists():
                continue
            dst_dir.mkdir(parents=True, exist_ok=True)
            for name in required:
                src = src_dir / name
                if src.exists():
                    shutil.copy2(src, dst_dir / name)


def copy_branching_files(trial_dir: Path, run_dir: Path, n_organoids: int) -> None:
    """
    Copy branchingData_0.csv and branchingData_1.csv from trial directory into
    each run/organoid_k folder.
    """
    for k in range(1, n_organoids + 1):
        src_org = trial_dir / f"organoid_{k}"
        dst_org = run_dir / f"organoid_{k}"
        dst_org.mkdir(parents=True, exist_ok=True)
        for name in ("branchingData_0.csv", "branchingData_1.csv"):
            src = src_org / name
            if not src.exists():
                die(f"Missing branching file: {src}")
            shutil.copy2(src, dst_org / name)


def pick_existing(*paths: Path) -> Path:
    for p in paths:
        if p.exists():
            return p
    die("Missing required file. Tried:\n" + "\n".join(str(p) for p in paths))
    raise RuntimeError


def read_seed_coords_from_branching(csv_path: Path, fallback: np.ndarray) -> np.ndarray:
    try:
        with csv_path.open("r", newline="") as f:
            row = next(csv.DictReader(f))
        return np.asarray(
            [
                float(row["proximalCoordsX"]),
                float(row["proximalCoordsY"]),
                float(row["proximalCoordsZ"]),
            ],
            dtype=float,
        )
    except Exception:
        return np.asarray(fallback, dtype=float).copy()


def update_1d_checkpoints(run_dir: Path, n_organoids: int, coords_inlet: np.ndarray, coords_outlet: np.ndarray, dy_step: float) -> None:
    repo_src = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_src / "geometry"))
    mesh_mod = importlib.import_module("mesh")

    for k in range(1, n_organoids + 1):
        org = run_dir / f"organoid_{k}"
        geom = org / "geometry"
        in_dir = org / "0D_Input_Files" / "inlet"
        out_dir = org / "0D_Input_Files" / "outlet"
        if not in_dir.exists() or not out_dir.exists():
            die(f"Missing 0D input dirs for organoid_{k}: {in_dir}, {out_dir}")

        xdmf_in = pick_existing(
            geom / "branched_network_inlet.xdmf",
            geom / f"branched_network_inlet_well{k}.xdmf",
        )
        xdmf_out = pick_existing(
            geom / "branched_network_outlet.xdmf",
            geom / f"branched_network_outlet_well{k}.xdmf",
        )

        c_in = read_seed_coords_from_branching(org / "branchingData_0.csv", np.asarray(coords_inlet, dtype=float))
        c_out = read_seed_coords_from_branching(org / "branchingData_1.csv", np.asarray(coords_outlet, dtype=float))

        mesh_mod.current_dir = geom
        mesh_mod.generate_1d_files(
            xdmf_file=xdmf_in.name,
            output_dir=str(in_dir),
            file_prefix="_inlet",
            inlet_coords=c_in,
        )
        mesh_mod.generate_1d_files(
            xdmf_file=xdmf_out.name,
            output_dir=str(out_dir),
            file_prefix="_outlet",
            inlet_coords=c_out,
        )


def run_darcy_for_all(run_dir: Path, args: argparse.Namespace) -> None:
    darcy_script = Path(args.darcy_script).expanduser().resolve()
    if not darcy_script.exists():
        die(f"Missing Darcy script: {darcy_script}")

    for k in range(1, args.n_organoids + 1):
        org = run_dir / f"organoid_{k}"
        geom = org / "geometry"
        c_in = read_seed_coords_from_branching(org / "branchingData_0.csv", np.array(args.coords_inlet, dtype=float))
        c_out = read_seed_coords_from_branching(org / "branchingData_1.csv", np.array(args.coords_outlet, dtype=float))

        darcy_args = [
            "--bioreactor-domain", str(geom / "bioreactor.xdmf"),
            "--facet-file", str(geom / "mesh_tags.xdmf"),
            "--mesh-inlet-file", str(geom / "tagged_branches_inlet.bp"),
            "--mesh-outlet-file", str(geom / "tagged_branches_outlet.bp"),
            "--pres-inlet-file", str(geom / "pressure_checkpoint_inlet.bp"),
            "--pres-outlet-file", str(geom / "pressure_checkpoint_outlet.bp"),
            "--flow-inlet-file", str(geom / "flow_checkpoint_inlet.bp"),
            "--flow-outlet-file", str(geom / "flow_checkpoint_outlet.bp"),
            "--area-inlet-file", str(geom / "area_checkpoint_inlet.bp"),
            "--area-outlet-file", str(geom / "area_checkpoint_outlet.bp"),
            "--coords-inlet", *map(str, c_in),
            "--coords-outlet", *map(str, c_out),
            "--branching-in-file", str(org / "branchingData_0.csv"),
            "--branching-out-file", str(org / "branchingData_1.csv"),
            "--concave-bc-mode", str(args.concave_bc_mode),
            "--lp-arterial", str(args.lp_arterial),
            "--lp-venous", str(args.lp_venous),
            "--out-dir", str(org),
        ]
        if int(args.darcy_mpi_procs) > 1:
            mpi_launcher = resolve_mpi_launcher(str(args.darcy_mpirun_cmd))
            cmd = [
                mpi_launcher,
                "-n",
                str(args.darcy_mpi_procs),
                sys.executable,
                str(darcy_script),
                *darcy_args,
            ]
        else:
            cmd = [sys.executable, str(darcy_script), *darcy_args]
        run(cmd)


def _max_rel(a: np.ndarray, b: np.ndarray, floor: float = 1e-20) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    aa = a[:n]
    bb = b[:n]
    den = np.maximum(np.abs(bb), floor)
    return float(np.max(np.abs(aa - bb) / den))


def convergence_for_organoid(iface_path: Path, tol_q: float, tol_p: float) -> tuple[bool, float, float]:
    iface = load_json(iface_path)
    q = np.asarray(iface.get("q_inlet", []), dtype=float)
    q_t = np.asarray(iface.get("q_inlet_target", []), dtype=float)
    p_in = np.asarray(iface.get("p_inlet_nodes", []), dtype=float)
    p_in_t = np.asarray(iface.get("p_inlet_target", []), dtype=float)
    p_out = np.asarray(iface.get("p_outlet_nodes", []), dtype=float)
    p_out_t = np.asarray(iface.get("p_outlet_target", []), dtype=float)

    rq = _max_rel(q, q_t, floor=1e-20)
    rp = max(_max_rel(p_in, p_in_t, floor=1e-12), _max_rel(p_out, p_out_t, floor=1e-12))
    return (rq <= tol_q and rp <= tol_p), rq, rp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Organoid coupling driver with geometry reuse + P1-LM Darcy.")
    ap.add_argument("--template-combined", default="../prep/prepped/trial-2/combined.in")
    ap.add_argument("--trial-dir", default="../prep/prepped/trial-3",
                    help="Directory containing organoid_k/branchingData_{0,1}.csv")
    ap.add_argument("--seed-run0", default="../prep/coupled/run_0",
                    help="Existing run_0 folder with 3D meshing/tagging to reuse.")
    ap.add_argument("--coupled-root", default="../coupling-output")

    ap.add_argument("--n-organoids", type=int, default=4)
    ap.add_argument("--n-ramp", type=int, default=10)
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--tol-q", type=float, default=1e-3)
    ap.add_argument("--tol-p", type=float, default=1e-3)
    ap.add_argument("--relaxation", type=float, default=0.3)

    ap.add_argument("--channel-inlet-bc", default="INFLOW")
    ap.add_argument("--channel-inlet-Q0", type=float, default=2.5e-2)
    ap.add_argument("--channel-outlet-bc", default="OUT3")
    ap.add_argument("--channel-outlet-P0", type=float, default=100.0)
    ap.add_argument("--use-outlet-pressure-ramp", action="store_true",
                    help="Ramp channel outlet pressure from 0 at i=0 to P0 by the end of the ramp.")

    ap.add_argument("--svzerodsolver", default="svzerodsolver")
    ap.add_argument("--run-and-split", default="../prep/run_and_split_svzerod.py")
    ap.add_argument("--darcy-script", default="../solves/darcy_p1_lm.py")
    ap.add_argument("--darcy-mpi-procs", type=int, default=1,
                    help="MPI ranks for Darcy solves (1 = serial).")
    ap.add_argument("--darcy-mpirun-cmd", default="/opt/miniconda3/envs/fenicsx-env/bin/mpiexec.hydra",
                    help="MPI launcher command for Darcy when --darcy-mpi-procs > 1.")

    ap.add_argument("--dy-step", type=float, default=0.6)
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-0.18, 0.9, 0.55])
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.18, 0.9, 0.55])

    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    ap.add_argument("--lp-arterial", type=float, default=0.0)
    ap.add_argument("--lp-venous", type=float, default=0.0)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    template = Path(args.template_combined).expanduser().resolve()
    if not template.is_file():
        die(f"Template combined.in not found: {template}")

    seed_run0 = Path(args.seed_run0).expanduser().resolve()
    if not seed_run0.exists():
        die(f"Seed run_0 does not exist: {seed_run0}")
    trial_dir = Path(args.trial_dir).expanduser().resolve()
    if not trial_dir.exists():
        die(f"Trial directory does not exist: {trial_dir}")

    coupled_root = Path(args.coupled_root).expanduser().resolve()
    # coupled_root.mkdir(parents=True, exist_ok=True)

    # # Build run_0
    # run0 = coupled_root / "run_0"
    # if run0.exists() and args.overwrite:
    #     shutil.rmtree(run0)
    # run0.mkdir(parents=True, exist_ok=True)
    # combined0 = run0 / "combined.in"
    # shutil.copy2(template, combined0)
    # deck0 = load_json(combined0)
    # apply_channel_ramp(
    #     deck0,
    #     args.channel_inlet_bc,
    #     Q0=float(args.channel_inlet_Q0),
    #     scale=0.0,
    #     outlet_name=args.channel_outlet_bc if args.use_outlet_pressure_ramp else None,
    #     outlet_P0=float(args.channel_outlet_P0),
    # )
    # zero_all_interface_bcs(deck0, args.n_organoids)
    # save_json(combined0, deck0)
    # ensure_organoid_dirs(run0, args.n_organoids)
    # sync_plot_templates(seed_run0, run0, args.n_organoids)

    # run([
    #     sys.executable, str(Path(args.run_and_split).expanduser().resolve()),
    #     "--exe", args.svzerodsolver,
    #     "--input", str(combined0),
    #     "--outdir", str(run0),
    #     "--output", "output.csv",
    #     "--organoid-root", str(run0),
    # ] + (["--debug"] if args.debug else []))

    # copy_seed_geometry(seed_run0, run0, args.n_organoids)
    # copy_branching_files(trial_dir, run0, args.n_organoids)
    # update_1d_checkpoints(run0, args.n_organoids, np.array(args.coords_inlet), np.array(args.coords_outlet), args.dy_step)
    # run_darcy_for_all(run0, args)

    # Coupling loop: ramp then iterate to convergence
    total_steps = int(args.n_ramp) + int(args.max_iter)
    for i in range(13, total_steps + 1):
        scale = min(i / float(max(args.n_ramp, 1)), 1.0) if i <= args.n_ramp else 1.0
        prev = coupled_root / f"run_{i-1}"
        cur = coupled_root / f"run_{i}"
        if cur.exists() and args.overwrite:
            shutil.rmtree(cur)
        copy_tree(prev, cur, overwrite=not args.overwrite)

        combined_i = cur / "combined.in"
        deck = load_json(combined_i)
        apply_channel_ramp(
            deck,
            args.channel_inlet_bc,
            Q0=float(args.channel_inlet_Q0),
            scale=scale,
            outlet_name=args.channel_outlet_bc if args.use_outlet_pressure_ramp else None,
            outlet_P0=float(args.channel_outlet_P0),
        )

        for k in range(1, args.n_organoids + 1):
            iface_prev = prev / f"organoid_{k}" / "out_darcy" / "interface_bc.json"
            if not iface_prev.exists():
                die(f"Missing interface file: {iface_prev}")
            apply_organoid_interface_from_json(
                deck,
                k,
                load_json(iface_prev),
                relax=float(args.relaxation),
            )
        save_json(combined_i, deck)
        ensure_organoid_dirs(cur, args.n_organoids)
        sync_plot_templates(seed_run0, cur, args.n_organoids)

        run([
            sys.executable, str(Path(args.run_and_split).expanduser().resolve()),
            "--exe", args.svzerodsolver,
            "--input", str(combined_i),
            "--outdir", str(cur),
            "--output", "output.csv",
            "--organoid-root", str(cur),
        ] + (["--debug"] if args.debug else []))

        copy_seed_geometry(seed_run0, cur, args.n_organoids)
        copy_branching_files(trial_dir, cur, args.n_organoids)
        update_1d_checkpoints(cur, args.n_organoids, np.array(args.coords_inlet), np.array(args.coords_outlet), args.dy_step)
        run_darcy_for_all(cur, args)

        # convergence after ramp
        if i > args.n_ramp:
            all_ok = True
            print(f"\n[check] coupling iteration {i - args.n_ramp}/{args.max_iter}")
            for k in range(1, args.n_organoids + 1):
                iface_cur = cur / f"organoid_{k}" / "out_darcy" / "interface_bc.json"
                ok, rq, rp = convergence_for_organoid(iface_cur, tol_q=args.tol_q, tol_p=args.tol_p)
                all_ok = all_ok and ok
                print(f"  organoid_{k}: rel_q={rq:.3e}, rel_p={rp:.3e}, converged={ok}")
            if all_ok:
                print(f"\n[done] converged at run_{i}.")
                return

    print(f"\n[done] reached max iterations ({args.max_iter}) after ramp without full convergence.")


if __name__ == "__main__":
    main()
