#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import re
import shlex
import shutil
import subprocess
import sys
import csv
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np


def die(msg: str) -> None:
    print(f"[error] {msg}", file=sys.stderr)
    raise SystemExit(2)


def run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print(f"[run] {' '.join(map(str, cmd))}  (cwd={cwd})", flush=True)
    env = build_subprocess_env(cmd)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, env=env)


def run_0d_split(args: argparse.Namespace, combined_path: Path, run_dir: Path) -> None:
    run([
        sys.executable, str(Path(args.run_and_split).expanduser().resolve()),
        "--exe", args.svzerodsolver,
        "--input", str(combined_path),
        "--outdir", str(run_dir),
        "--output", "output.csv",
        "--organoid-root", str(run_dir),
        "--no-plot",
    ] + (["--debug"] if args.debug else []))


def try_run_0d_split(args: argparse.Namespace, combined_path: Path, run_dir: Path) -> bool:
    """
    Run a trial 0D solve/split without aborting the outer coupling loop.

    The inner resistance search may intentionally test unstable candidates. A
    failed candidate should simply receive an infinite score while the search
    continues with the remaining candidates.
    """
    cmd = [
        sys.executable, str(Path(args.run_and_split).expanduser().resolve()),
        "--exe", args.svzerodsolver,
        "--input", str(combined_path),
        "--outdir", str(run_dir),
        "--output", "output.csv",
        "--organoid-root", str(run_dir),
        "--no-plot",
    ] + (["--debug"] if args.debug else [])
    print(f"[run] {' '.join(map(str, cmd))}  (cwd=None)", flush=True)
    result = subprocess.run(cmd, check=False, env=build_subprocess_env(cmd))
    if result.returncode == 0:
        return True
    print(
        "[inner-resistance-0d] candidate failed "
        f"(returncode={result.returncode}); skipping this candidate",
        flush=True,
    )
    return False


def update_convergence_plots(args: argparse.Namespace, coupled_root: Path, run_index: int) -> None:
    if not bool(args.plot_each_iteration):
        return
    plot_script = Path(args.plot_convergence_script).expanduser()
    if not plot_script.is_absolute():
        plot_script = Path(__file__).resolve().parent / plot_script
    if not plot_script.exists():
        print(f"[plot] convergence plot script not found: {plot_script}", flush=True)
        return
    title = str(args.plot_title).strip() or f"through run_{run_index}"
    cmd = [
        sys.executable,
        str(plot_script),
        "--root",
        str(coupled_root),
        "--late-count",
        str(int(args.plot_late_count)),
        "--title",
        title,
    ]
    print(f"[plot] uƒating convergence plots through run_{run_index}", flush=True)
    result = subprocess.run(cmd, check=False, env=build_subprocess_env(cmd))
    if result.returncode != 0:
        print(
            f"[plot] warning: convergence plot update failed with returncode={result.returncode}; continuing.",
            flush=True,
        )


def build_subprocess_env(cmd: List[str]) -> dict:
    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    if cmd:
        launcher = Path(str(cmd[0])).name
        if launcher in {"mpiexec", "mpirun", "mpiexec.hydra"}:
            # Nested MPI launches inside Slurm job steps can inherit PMI/PMIx
            # variables that conflict with the launcher bundled in the active
            # Python environment.
            for key in (
                "PMI_FD",
                "PMI_RANK",
                "PMI_SIZE",
                "PMIX_NAMESPACE",
                "PMIX_RANK",
                "PMIX_SERVER_URI2",
                "PMIX_SERVER_URI21",
                "SLURM_MPI_TYPE",
            ):
                env.pop(key, None)
    return env


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


def build_mpi_command(user_cmd: str, nprocs: int, payload: List[str]) -> List[str]:
    launcher = resolve_mpi_launcher(str(user_cmd).strip())
    parts = shlex.split(launcher) if str(launcher).strip() else []
    if not parts:
        raise ValueError("MPI launcher command must not be empty")
    return [*parts, "-n", str(int(nprocs)), *payload]


def resolve_perm_region_for_organoid(base: str, organoid_idx: int) -> str:
    raw = str(base).strip()
    if not raw:
        return ""

    if "X" in raw:
        candidate = Path(raw.replace("X", str(organoid_idx))).expanduser()
        if candidate.suffix:
            return str(candidate.resolve())
        with_suffix = candidate.with_suffix(".stl")
        return str((with_suffix if with_suffix.exists() else candidate).resolve())

    path = Path(raw).expanduser()
    if path.is_dir():
        return str((path / f"organoid-{organoid_idx}.stl").resolve())
    return str(path.resolve())


def copy_tree(src: Path, dst: Path, overwrite: bool = True) -> None:
    if overwrite and dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, dirs_exist_ok=not overwrite)


def initialize_iteration_run(prev: Path, cur: Path, overwrite: bool) -> None:
    """
    Start a new coupling iteration without cloning the full previous run.

    The iteration only needs the previous combined.in as a mutable starting
    deck. Previous interface data are read directly from prev/out_darcy, while
    geometry, branching files, 0D outputs, and Darcy outputs are generated fresh
    later in the current run directory.
    """
    if cur.exists() and overwrite:
        shutil.rmtree(cur)
    cur.mkdir(parents=True, exist_ok=True)
    src_combined = prev / "combined.in"
    if not src_combined.exists():
        die(f"Missing previous combined.in: {src_combined}")
    shutil.copy2(src_combined, cur / "combined.in")


def link_or_copy_tree(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
    except OSError:
        copy_tree(src, dst, overwrite=True)


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


def load_scaled_screening_config(trial_dir: Optional[Path], args: argparse.Namespace) -> Optional[dict[str, Any]]:
    mode = str(getattr(args, "scaled_screening_mode", "auto")).strip().lower()
    if mode == "off":
        return None

    summary: dict[str, Any] = {}
    summary_path: Optional[Path] = None
    if trial_dir is not None:
        candidate = trial_dir / "scaled_screening_summary.json"
        if candidate.exists():
            summary_path = candidate
            summary = load_json(candidate)

    if summary_path is None and mode != "on":
        return None

    reference_organoid = int(summary.get("reference_organoid", 1))
    if reference_organoid < 1 or reference_organoid > int(args.n_organoids):
        die(
            f"Scaled-screening reference organoid {reference_organoid} is outside "
            f"1..{int(args.n_organoids)}"
        )
    shifts: dict[int, tuple[float, float, float]] = {
        reference_organoid: (0.0, 0.0, 0.0),
    }
    pressure_offset_anchor = str(getattr(args, "scaled_pressure_offset_anchor", "arterial"))

    for row in summary.get("scaled_organoids", []):
        try:
            organoid_id = int(row.get("organoid_id"))
        except Exception:
            continue
        raw_shift = row.get("shift", None)
        if isinstance(raw_shift, list) and len(raw_shift) == 3:
            try:
                shifts[organoid_id] = tuple(float(v) for v in raw_shift)
            except Exception:
                pass
        if "offset_anchor" in row:
            pressure_offset_anchor = str(row.get("offset_anchor", pressure_offset_anchor))

    for organoid_id in range(1, int(args.n_organoids) + 1):
        shifts.setdefault(
            organoid_id,
            (0.0, -float(args.dy_step) * float(organoid_id - reference_organoid), 0.0),
        )

    return {
        "enabled": True,
        "summary_path": str(summary_path) if summary_path is not None else "",
        "reference_organoid": reference_organoid,
        "pressure_offset_anchor": str(pressure_offset_anchor).strip().lower(),
        "shifts": shifts,
        "assumptions": list(summary.get("assumptions", [])),
    }


def _import_scaled_field_helpers():
    repo_src = Path(__file__).resolve().parents[1]
    solves_dir = repo_src / "solves"
    if str(solves_dir) not in sys.path:
        sys.path.insert(0, str(solves_dir))
    return importlib.import_module("compare_scaled_organoid_fields")


def _scale_pressure_value(value: Any, scale_factor: float, pressure_offset: float) -> Any:
    if isinstance(value, list):
        arr = np.asarray(value, dtype=float)
        return (pressure_offset + scale_factor * arr).tolist()
    return float(pressure_offset + scale_factor * float(value))


def _scale_nonpressure_value(value: Any, scale_factor: float) -> Any:
    if isinstance(value, list):
        arr = np.asarray(value, dtype=float)
        return (scale_factor * arr).tolist()
    return float(scale_factor * float(value))


def _shift_coords_value(value: Any, shift: tuple[float, float, float]) -> Any:
    arr = np.asarray(value, dtype=float)
    return (arr + np.asarray(shift, dtype=float)).tolist()


def _build_scaled_interface_bc(
    ref_iface: dict[str, Any],
    target_zero_d: dict[str, Any],
    pressure_transform: dict[str, float],
    scale_factor: float,
    shift: tuple[float, float, float],
) -> dict[str, Any]:
    iface = json.loads(json.dumps(ref_iface))
    pressure_offset = float(pressure_transform["pressure_offset"])

    pressure_scalar_keys = {
        "p_inlet_mean",
        "p_outlet_mean",
        "p_concave_inlet_bc",
        "p_concave_outlet_bc",
        "p_concave_inlet_bc_low",
        "p_concave_inlet_bc_high",
        "p_concave_outlet_bc_low",
        "p_concave_outlet_bc_high",
    }
    nonpressure_scalar_keys = {
        "q_artery_leak",
        "q_venous_leak",
    }
    pressure_array_keys = {
        "p_inlet_nodes",
        "p_outlet_nodes",
        "p_inlet_nodes_raw",
        "p_outlet_nodes_raw",
        "p_inlet_nodes_coupled",
        "p_outlet_nodes_coupled",
        "p_inlet_target",
        "p_outlet_target",
        "p_inlet_parent_0d",
        "p_outlet_parent_0d",
    }
    nonpressure_array_keys = {
        "q_inlet",
        "q_outlet",
        "q_inlet_inward",
        "q_inlet_target",
        "q_inlet_target_in",
        "q_outlet_target_out",
        "q_inlet_residual",
        "divu_inlet",
        "divu_outlet",
    }
    vector_keys = {
        "u_inlet_nodes",
        "u_outlet_nodes",
    }
    coord_keys = {
        "coords_inlet",
        "coords_outlet",
    }

    for key in list(iface.keys()):
        if key in pressure_scalar_keys:
            iface[key] = _scale_pressure_value(iface[key], scale_factor, pressure_offset)
        elif key in nonpressure_scalar_keys:
            iface[key] = _scale_nonpressure_value(iface[key], scale_factor)
        elif key in pressure_array_keys:
            iface[key] = _scale_pressure_value(iface[key], scale_factor, pressure_offset)
        elif key in nonpressure_array_keys:
            iface[key] = _scale_nonpressure_value(iface[key], scale_factor)
        elif key in vector_keys:
            iface[key] = _scale_nonpressure_value(iface[key], scale_factor)
        elif key in coord_keys:
            iface[key] = _shift_coords_value(iface[key], shift)

    iface["p_concave_inlet_bc"] = float(pressure_transform["target_arterial_pressure"])
    iface["p_concave_outlet_bc"] = float(pressure_transform["target_venous_pressure"])
    ref_profile = str(ref_iface.get("concave_pressure_profile", "constant")).strip().lower()
    iface["concave_pressure_profile"] = ref_profile if ref_profile in {"constant", "linear"} else "constant"
    iface["concave_pressure_profile_axis"] = str(ref_iface.get("concave_pressure_profile_axis", "y"))
    iface["scaled_concave_pressure_profile_source"] = "reference_interface_bc"
    iface["scaled_from_reference_organoid"] = int(ref_iface.get("scaled_from_reference_organoid", 1))
    iface["scaled_zero_d_q_artery_mean"] = float(target_zero_d["q_art_mean"])
    iface["scaled_zero_d_q_venous_mean"] = float(target_zero_d["q_ven_mean"])
    iface["scaled_leak_flow_source"] = "reference_darcy_scaled"
    iface["scaled_pressure_offset_anchor"] = pressure_transform["offset_anchor"]
    iface["scale_factor_from_reference"] = scale_factor
    iface["pressure_offset_from_reference"] = pressure_offset
    iface["offset_from_artery"] = float(pressure_transform["offset_from_artery"])
    iface["offset_from_vein"] = float(pressure_transform["offset_from_vein"])
    iface["offset_mismatch"] = float(pressure_transform["offset_mismatch"])
    iface["arterial_channel_drop"] = float(pressure_transform["arterial_channel_drop"])
    iface["venous_channel_drop"] = float(pressure_transform["venous_channel_drop"])
    return iface


def _write_scaled_organoid_outputs(
    run_dir: Path,
    organoid_id: int,
    ref_iface: dict[str, Any],
    ref_p: dict[str, Any],
    ref_u: dict[str, Any],
    target_zero_d: dict[str, Any],
    pressure_transform: dict[str, float],
    shift: tuple[float, float, float],
    h5py: Any,
    scaled_fields: Any,
    reference_organoid: int,
) -> dict[str, Any]:
    scale_factor = float(pressure_transform["scale_factor"])
    pressure_offset = float(pressure_transform["pressure_offset"])

    out_dir = run_dir / f"organoid_{organoid_id}" / "out_darcy"
    out_dir.mkdir(parents=True, exist_ok=True)

    scaled_p = pressure_offset + scale_factor * np.asarray(ref_p["values"], dtype=float)
    scaled_u = scale_factor * np.asarray(ref_u["values"], dtype=float)

    scaled_fields._write_xdmf_field(
        out_dir / "p.xdmf",
        np.asarray(ref_p["coords"], dtype=float) + np.asarray(shift, dtype=float),
        ref_p["topology"],
        scaled_p,
        ref_p["topology_type"],
        ref_p["attribute_type"],
        ref_p["attribute_name"],
        ref_p["attribute_center"],
        h5py,
    )
    scaled_fields._write_xdmf_field(
        out_dir / "u.xdmf",
        np.asarray(ref_u["coords"], dtype=float) + np.asarray(shift, dtype=float),
        ref_u["topology"],
        scaled_u,
        ref_u["topology_type"],
        ref_u["attribute_type"],
        ref_u["attribute_name"],
        ref_u["attribute_center"],
        h5py,
    )

    iface = _build_scaled_interface_bc(
        ref_iface=ref_iface,
        target_zero_d=target_zero_d,
        pressure_transform=pressure_transform,
        scale_factor=scale_factor,
        shift=shift,
    )
    iface["scaled_from_reference_organoid"] = int(reference_organoid)
    (out_dir / "interface_bc.json").write_text(json.dumps(iface, indent=2))
    return {
        "organoid_id": organoid_id,
        "scale_factor": scale_factor,
        "pressure_offset": pressure_offset,
        "shift": list(shift),
        "target_arterial_pressure": float(pressure_transform["target_arterial_pressure"]),
        "target_venous_pressure": float(pressure_transform["target_venous_pressure"]),
        "offset_anchor": pressure_transform["offset_anchor"],
        "offset_mismatch": float(pressure_transform["offset_mismatch"]),
    }


def replicate_reference_organoid_outputs(
    run_dir: Path,
    n_organoids: int,
    scaled_cfg: dict[str, Any],
    copy_field_outputs: bool = True,
) -> None:
    reference_organoid = int(scaled_cfg["reference_organoid"])
    ref_out = run_dir / f"organoid_{reference_organoid}" / "out_darcy"
    ref_iface_path = ref_out / "interface_bc.json"
    ref_p_path = ref_out / "p.xdmf"
    ref_u_path = ref_out / "u.xdmf"
    missing_reference = not ref_iface_path.exists() or (
        bool(copy_field_outputs)
        and (not ref_p_path.exists() or not ref_u_path.exists())
    )
    if missing_reference:
        die(
            "Scaled-screening replication requires Darcy outputs for the reference organoid; "
            f"missing one of {ref_iface_path}, {ref_p_path}, or {ref_u_path}"
        )

    replicated_rows: list[dict[str, Any]] = []
    for organoid_id in range(1, int(n_organoids) + 1):
        if organoid_id == reference_organoid:
            continue
        out_dir = run_dir / f"organoid_{organoid_id}" / "out_darcy"
        out_dir.mkdir(parents=True, exist_ok=True)
        if copy_field_outputs:
            copy_xdmf_with_sidecars(ref_p_path, out_dir / "p.xdmf")
            copy_xdmf_with_sidecars(ref_u_path, out_dir / "u.xdmf")
        shutil.copy2(ref_iface_path, out_dir / "interface_bc.json")
        replicated_rows.append(
            {
                "organoid_id": organoid_id,
                "mode": "replicated_from_reference",
                "reference_organoid": reference_organoid,
                "field_outputs_copied": bool(copy_field_outputs),
            }
        )

    summary = {
        "trial_dir": str(run_dir),
        "prepared_root": str(run_dir),
        "reference_organoid": reference_organoid,
        "scaled_organoids": replicated_rows,
        "assumptions": [
            "run_0 has zero channel ramp, so repeated wells directly reuse the reference Darcy outputs.",
            (
                "Copied wells receive the same p.xdmf, u.xdmf, and interface_bc.json values as the reference organoid."
                if copy_field_outputs
                else "Copied wells receive interface_bc.json only; p.xdmf/u.xdmf are intentionally omitted in minimal output mode."
            ),
        ],
    }
    save_json(run_dir / "scaled_screening_summary.json", summary)


def synthesize_scaled_screening_outputs(
    run_dir: Path,
    n_organoids: int,
    scaled_cfg: dict[str, Any],
) -> None:
    scaled_fields = _import_scaled_field_helpers()
    np_mod = scaled_fields._import_numpy()
    h5py = scaled_fields._import_h5py()

    reference_organoid = int(scaled_cfg["reference_organoid"])
    grouped = scaled_fields._load_0d_groups(run_dir, np_mod)
    zero_d = {
        organoid_id: scaled_fields._load_0d_organoid_summary(grouped, organoid_id, np_mod)
        for organoid_id in range(1, int(n_organoids) + 1)
    }
    ref_zero_d = zero_d[reference_organoid]

    ref_out = run_dir / f"organoid_{reference_organoid}" / "out_darcy"
    ref_iface_path = ref_out / "interface_bc.json"
    ref_p_path = ref_out / "p.xdmf"
    ref_u_path = ref_out / "u.xdmf"
    if not ref_iface_path.exists() or not ref_p_path.exists() or not ref_u_path.exists():
        die(
            "Scaled-screening mode requires Darcy outputs for the reference organoid; "
            f"missing one of {ref_iface_path}, {ref_p_path}, or {ref_u_path}"
        )

    ref_iface = json.loads(ref_iface_path.read_text())
    ref_p = scaled_fields._load_xdmf_field(ref_p_path, np_mod, h5py)
    ref_u = scaled_fields._load_xdmf_field(ref_u_path, np_mod, h5py)

    scaling_rows: list[dict[str, Any]] = []
    for organoid_id in range(1, int(n_organoids) + 1):
        if organoid_id == reference_organoid:
            continue
        shift = tuple(
            scaled_cfg["shifts"].get(
                organoid_id,
                scaled_fields._shift_for_organoid(reference_organoid, organoid_id, float(0.0)),
            )
        )
        transform = scaled_fields._pressure_transform_from_0d(
            ref_zero_d,
            zero_d[organoid_id],
            scaled_cfg["pressure_offset_anchor"],
        )
        scaling_rows.append(
            _write_scaled_organoid_outputs(
                run_dir=run_dir,
                organoid_id=organoid_id,
                ref_iface=ref_iface,
                ref_p=ref_p,
                ref_u=ref_u,
                target_zero_d=zero_d[organoid_id],
                pressure_transform=transform,
                shift=shift,
                h5py=h5py,
                scaled_fields=scaled_fields,
                reference_organoid=reference_organoid,
            )
        )

    summary = {
        "trial_dir": str(run_dir),
        "prepared_root": str(run_dir),
        "reference_organoid": reference_organoid,
        "scaled_organoids": scaling_rows,
        "assumptions": list(
            scaled_cfg.get("assumptions")
            or [
                "The same synthetic vasculature is replicated into each organoid well.",
                "Only the reference organoid is solved with Darcy; wells 2..N reuse the reference fields with y-translation and 0D-based scaling.",
                "Scaled wells write p.xdmf, u.xdmf, and interface_bc.json but do not run an independent Darcy solve.",
            ]
        ),
    }
    save_json(run_dir / "scaled_screening_summary.json", summary)


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


def set_resistance_bc(bc: dict, resistance: float, distal_pressure: float) -> None:
    bc["bc_type"] = "RESISTANCE"
    bc["bc_values"] = {
        "Pd": float(distal_pressure),
        "R": float(resistance),
    }


def set_pressure_bc_constant(bc: dict, pressure: float) -> None:
    bc["bc_type"] = "PRESSURE"
    bc["bc_values"] = {
        "t": [0.0, 1.0],
        "P": [float(pressure), float(pressure)],
    }


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


def terminal_flow_is_unsupported(
    p_darcy: float,
    p_parent: float,
    q_darcy: float,
    q_0d: float,
    mass_balance_direct_port_relative: float,
    flow_factor: float,
    pressure_rel_tol: float,
    mass_balance_tol: float,
    q_floor: float,
    pressure_floor: float = 1.0,
) -> bool:
    """Return True when 0D is asking for more terminal flow than Darcy appears to support."""
    try:
        p_darcy = float(p_darcy)
        p_parent = float(p_parent)
        q_darcy = float(q_darcy)
        q_0d = float(q_0d)
        mass_balance_direct_port_relative = float(mass_balance_direct_port_relative)
    except (TypeError, ValueError):
        return False

    if not (np.isfinite(q_darcy) and np.isfinite(q_0d)):
        return False

    q_scale = max(abs(q_darcy), float(q_floor), np.finfo(float).tiny)
    over_requested = abs(q_0d) > float(flow_factor) * q_scale
    if not over_requested:
        return False

    bad_mass_balance = (
        np.isfinite(mass_balance_direct_port_relative)
        and mass_balance_direct_port_relative > float(mass_balance_tol)
    )
    if np.isfinite(p_darcy) and np.isfinite(p_parent):
        pressure_scale = max(abs(p_parent), abs(p_darcy), float(pressure_floor))
        pressure_gap_rel = abs(p_darcy - p_parent) / pressure_scale
        bad_pressure_gap = pressure_gap_rel > float(pressure_rel_tol)
    else:
        bad_pressure_gap = False

    return bool(bad_mass_balance or bad_pressure_gap)


def pressure_scaled_terminal_target(
    current_terminal_pressure: float,
    raw_darcy_pressure: float,
    parent_pressure: float,
    q_0d: float,
    terminal_resistance: float,
    side: str,
    q_floor: float,
) -> float:
    """
    Convert a Darcy pressure overshoot into a reduced 0D flow request.

    If the Darcy solve says the tissue pressure is much farther from the parent
    pressure than the current 0D terminal pressure drop, scale the terminal flow
    down by that pressure-drop ratio and reconstruct a terminal pressure using
    the local Poiseuille resistance.
    """
    cur = float(current_terminal_pressure)
    p_darcy = float(raw_darcy_pressure)
    p_parent = float(parent_pressure)
    q_mag = abs(float(q_0d))

    if not (np.isfinite(cur) and np.isfinite(p_darcy) and np.isfinite(p_parent)):
        return cur

    current_drive = abs(cur - p_parent)
    required_drive = abs(p_darcy - p_parent)
    if not np.isfinite(required_drive) or required_drive <= 0.0:
        return cur

    flow_scale = min(1.0, current_drive / max(required_drive, np.finfo(float).tiny))
    q_next = q_mag * flow_scale

    resistance = abs(float(terminal_resistance)) if np.isfinite(terminal_resistance) else float("nan")
    if not np.isfinite(resistance) or resistance <= 0.0:
        resistance = current_drive / max(q_mag, float(q_floor), np.finfo(float).tiny)

    reduced_drive = min(current_drive, resistance * q_next)
    if not np.isfinite(reduced_drive):
        return cur

    if side == "arterial":
        return float(p_parent - reduced_drive)
    if side == "venous":
        return float(p_parent + reduced_drive)
    raise ValueError(f"Unknown terminal side {side!r}")


def set_arterial_terminal_pressure_feedback(
    bc: dict,
    p_darcy: float,
    p_parent: float,
    q_darcy: float,
    q_0d: float,
    terminal_resistance: float,
    alpha: float,
    mass_balance_direct_port_relative: float,
    support_flow_factor: float,
    support_pressure_rel_tol: float,
    support_mass_balance_tol: float,
    support_q_floor: float,
) -> None:
    vals = bc.setdefault("bc_values", {})
    cur = 0.0
    if isinstance(vals.get("P", None), list) and len(vals["P"]) > 0:
        cur = float(vals["P"][0])

    p_darcy = float(p_darcy)
    p_parent = float(p_parent)
    if not np.isfinite(p_parent):
        p_parent = cur

    unsupported = terminal_flow_is_unsupported(
        p_darcy=p_darcy,
        p_parent=p_parent,
        q_darcy=q_darcy,
        q_0d=q_0d,
        mass_balance_direct_port_relative=mass_balance_direct_port_relative,
        flow_factor=support_flow_factor,
        pressure_rel_tol=support_pressure_rel_tol,
        mass_balance_tol=support_mass_balance_tol,
        q_floor=support_q_floor,
    )
    current_drive = abs(cur - p_parent) if np.isfinite(cur) and np.isfinite(p_parent) else 0.0
    darcy_drive = abs(p_darcy - p_parent) if np.isfinite(p_darcy) and np.isfinite(p_parent) else 0.0
    pressure_over_requested = darcy_drive > current_drive * (1.0 + float(support_pressure_rel_tol))

    if not np.isfinite(p_darcy):
        target = cur
    elif unsupported or pressure_over_requested:
        target = pressure_scaled_terminal_target(
            current_terminal_pressure=cur,
            raw_darcy_pressure=p_darcy,
            parent_pressure=p_parent,
            q_0d=q_0d,
            terminal_resistance=terminal_resistance,
            side="arterial",
            q_floor=support_q_floor,
        )
    else:
        target = p_darcy
    p_next = (1.0 - float(alpha)) * cur + float(alpha) * target
    set_bc_constant(bc, "P", p_next)


def set_venous_terminal_pressure_feedback(
    bc: dict,
    p_darcy: float,
    p_parent: float,
    q_darcy: float,
    q_0d: float,
    terminal_resistance: float,
    alpha: float,
    mass_balance_direct_port_relative: float,
    support_flow_factor: float,
    support_pressure_rel_tol: float,
    support_mass_balance_tol: float,
    support_q_floor: float,
) -> None:
    vals = bc.setdefault("bc_values", {})
    cur = 0.0
    if isinstance(vals.get("P", None), list) and len(vals["P"]) > 0:
        cur = float(vals["P"][0])

    p_darcy = float(p_darcy)
    p_parent = float(p_parent)
    if not np.isfinite(p_parent):
        p_parent = cur

    unsupported = terminal_flow_is_unsupported(
        p_darcy=p_darcy,
        p_parent=p_parent,
        q_darcy=q_darcy,
        q_0d=q_0d,
        mass_balance_direct_port_relative=mass_balance_direct_port_relative,
        flow_factor=support_flow_factor,
        pressure_rel_tol=support_pressure_rel_tol,
        mass_balance_tol=support_mass_balance_tol,
        q_floor=support_q_floor,
    )
    current_drive = abs(cur - p_parent) if np.isfinite(cur) and np.isfinite(p_parent) else 0.0
    darcy_drive = abs(p_darcy - p_parent) if np.isfinite(p_darcy) and np.isfinite(p_parent) else 0.0
    pressure_over_requested = darcy_drive > current_drive * (1.0 + float(support_pressure_rel_tol))

    if not np.isfinite(p_darcy):
        target = cur
    elif unsupported or pressure_over_requested:
        target = pressure_scaled_terminal_target(
            current_terminal_pressure=cur,
            raw_darcy_pressure=p_darcy,
            parent_pressure=p_parent,
            q_0d=q_0d,
            terminal_resistance=terminal_resistance,
            side="venous",
            q_floor=support_q_floor,
        )
    else:
        target = p_darcy
    p_next = (1.0 - float(alpha)) * cur + float(alpha) * target
    set_bc_constant(bc, "P", p_next)


def terminal_resistance_by_bc_name(deck: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for vessel in deck.get("vessels", []):
        bc_name = str(vessel.get("boundary_conditions", {}).get("outlet", ""))
        if not bc_name:
            continue
        vals = vessel.get("zero_d_element_values", {})
        try:
            out[bc_name] = float(vals.get("R_poiseuille", float("nan")))
        except Exception:
            continue
    return out


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
            ensure_pressure_bc(bc)
            set_bc_constant(bc, "P", 0.0)
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
    support_flow_factor: float = 10.0,
    support_pressure_rel_tol: float = 0.1,
    support_mass_balance_tol: float = 0.1,
    support_q_floor: float = 1.0e-20,
    allow_missing_terminal_bcs: bool = False,
) -> None:
    p_in = iface.get("p_inlet_nodes_raw", iface.get("p_inlet_nodes", []))
    p_out = iface.get("p_outlet_nodes_raw", iface.get("p_outlet_nodes", []))
    p_in_parent = iface.get("p_inlet_parent_0d", iface.get("p_inlet_target", []))
    q_in = iface.get("q_inlet_port_in", iface.get("q_inlet", []))
    q_in_target = iface.get("q_inlet_target_in", iface.get("q_inlet_target", []))
    ids_in = iface.get("branch_ids_inlet", [])
    p_out_parent = iface.get("p_outlet_parent_0d", iface.get("p_outlet_target", []))
    q_out = iface.get("q_outlet_port_out", iface.get("q_outlet", []))
    q_out_target = iface.get("q_outlet_target_out", [])
    ids_out = iface.get("branch_ids_outlet", [])
    artery_leak = float(iface.get("q_artery_leak", 0.0))
    vein_leak = float(iface.get("q_venous_leak", 0.0))
    mass_balance_direct_port_relative = float(iface.get("mass_balance_direct_port_relative", 0.0))
    skip_1d = bool(iface.get("skip_1d", False))

    if not isinstance(p_in, list) or not isinstance(p_out, list):
        die(f"interface_bc.json for organoid_{organoid_idx} missing p_inlet_nodes or p_outlet_nodes")
    if not isinstance(p_in_parent, list):
        p_in_parent = []
    if not isinstance(q_in, list):
        q_in = []
    if not isinstance(q_in_target, list):
        q_in_target = []
    if not isinstance(ids_in, list):
        ids_in = []
    if not isinstance(p_out_parent, list):
        p_out_parent = []
    if not isinstance(q_out, list):
        q_out = []
    if not isinstance(q_out_target, list):
        q_out_target = []
    if not isinstance(ids_out, list):
        ids_out = []

    inlet_bcs = find_bcs_with_prefix(deck, f"organoid{organoid_idx}_inlet_OUT")
    outlet_bcs = find_bcs_with_prefix(deck, f"organoid{organoid_idx}_outlet_OUT")
    leak_art = find_bcs_with_prefix(deck, f"LEAK_ART_{organoid_idx}")
    leak_ven = find_bcs_with_prefix(deck, f"LEAK_VEN_{organoid_idx}")

    if (len(inlet_bcs) == 0 or len(outlet_bcs) == 0) and not (allow_missing_terminal_bcs or skip_1d):
        die(f"Missing organoid interface BC groups for organoid {organoid_idx}")
    if len(inlet_bcs) == 0 and len(p_in) > 0:
        die(f"Organoid {organoid_idx} has inlet pressures to apply but no inlet BC group in combined deck")
    if len(outlet_bcs) == 0 and len(p_out) > 0:
        die(f"Organoid {organoid_idx} has outlet pressures to apply but no outlet BC group in combined deck")

    inlet_bc_by_name = {str(bc.get("bc_name", "")): bc for bc in inlet_bcs}
    resistance_by_bc = terminal_resistance_by_bc_name(deck)
    m = len(p_in)
    for j in range(m):
        branch_id = int(ids_in[j]) if j < len(ids_in) else j
        bc_name = f"organoid{organoid_idx}_inlet_OUT{branch_id}"
        bc = inlet_bc_by_name.get(bc_name)
        if bc is None:
            if j >= len(inlet_bcs):
                continue
            bc = inlet_bcs[j]
            bc_name = str(bc.get("bc_name", bc_name))
        p_tissue = float(p_in[j])
        q_0d = float(q_in_target[j]) if j < len(q_in_target) else float("nan")
        parent = float(p_in_parent[j]) if j < len(p_in_parent) else float("nan")
        ensure_pressure_bc(bc)
        q_darcy = float(q_in[j]) if j < len(q_in) else float("nan")
        resistance = resistance_by_bc.get(bc_name, float("nan"))
        set_arterial_terminal_pressure_feedback(
            bc,
            p_darcy=p_tissue,
            p_parent=parent,
            q_darcy=q_darcy,
            q_0d=abs(q_0d),
            terminal_resistance=resistance,
            alpha=float(relax),
            mass_balance_direct_port_relative=mass_balance_direct_port_relative,
            support_flow_factor=float(support_flow_factor),
            support_pressure_rel_tol=float(support_pressure_rel_tol),
            support_mass_balance_tol=float(support_mass_balance_tol),
            support_q_floor=float(support_q_floor),
        )

    outlet_bc_by_name = {str(bc.get("bc_name", "")): bc for bc in outlet_bcs}
    m = len(p_out)
    for j in range(m):
        branch_id = int(ids_out[j]) if j < len(ids_out) else j
        bc_name = f"organoid{organoid_idx}_outlet_OUT{branch_id}"
        bc = outlet_bc_by_name.get(bc_name)
        if bc is None:
            if j >= len(outlet_bcs):
                continue
            bc = outlet_bcs[j]
            bc_name = str(bc.get("bc_name", bc_name))
        p_tissue = float(p_out[j])
        q_0d = float(q_out_target[j]) if j < len(q_out_target) else float("nan")
        parent = float(p_out_parent[j]) if j < len(p_out_parent) else float("nan")
        ensure_pressure_bc(bc)
        q_darcy = float(q_out[j]) if j < len(q_out) else float("nan")
        resistance = resistance_by_bc.get(bc_name, float("nan"))
        set_venous_terminal_pressure_feedback(
            bc,
            p_darcy=p_tissue,
            p_parent=parent,
            q_darcy=q_darcy,
            q_0d=q_0d,
            terminal_resistance=resistance,
            alpha=float(relax),
            mass_balance_direct_port_relative=mass_balance_direct_port_relative,
            support_flow_factor=float(support_flow_factor),
            support_pressure_rel_tol=float(support_pressure_rel_tol),
            support_mass_balance_tol=float(support_mass_balance_tol),
            support_q_floor=float(support_q_floor),
        )

    if leak_art:
        ensure_flow_bc(leak_art[0])
        set_bc_relaxed(leak_art[0], "Q", float(-artery_leak), relax)
    if leak_ven:
        ensure_flow_bc(leak_ven[0])
        set_bc_relaxed(leak_ven[0], "Q", float(-vein_leak), relax)


def _list_values(value: Any, n: int, fallback: float = float("nan")) -> list[float]:
    if isinstance(value, list):
        vals = []
        for item in value[:n]:
            try:
                vals.append(float(item))
            except Exception:
                vals.append(float(fallback))
        if len(vals) < n:
            vals.extend([float(fallback)] * (n - len(vals)))
        return vals
    try:
        scalar = float(value)
    except Exception:
        scalar = float(fallback)
    return [scalar] * n


def _terminal_reference_values(iface: dict, side: str, n: int) -> list[float]:
    if side == "inlet":
        if "p_inlet_reference" in iface:
            return _list_values(iface.get("p_inlet_reference"), n)
        fallback = iface.get(
            "p_tissue_reference_pressure",
            iface.get("p_concave_inlet_bc", float("nan")),
        )
        return _list_values(fallback, n)
    if side == "outlet":
        if "p_outlet_reference" in iface:
            return _list_values(iface.get("p_outlet_reference"), n)
        fallback = iface.get(
            "p_tissue_reference_pressure",
            iface.get("p_concave_outlet_bc", float("nan")),
        )
        return _list_values(fallback, n)
    raise ValueError(f"Unknown terminal side {side!r}")


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _pressure_continuity_rel_error(a: float, b: float, floor: float = 1.0) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    denominator = max(min(abs(a), abs(b)), float(floor))
    return float(abs(a - b) / denominator)


def _parse_float_list(value: Any, default: list[float]) -> list[float]:
    raw = str(value or "").strip()
    if not raw:
        return list(default)
    vals: list[float] = []
    for part in raw.split(","):
        parsed = _safe_float(part.strip())
        if np.isfinite(parsed):
            vals.append(float(parsed))
    return vals or list(default)


def _bc_first_numeric(bc: Optional[dict], key: str) -> float:
    if bc is None:
        return float("nan")
    vals = bc.get("bc_values", {})
    raw = vals.get(key, None)
    if isinstance(raw, list) and len(raw) > 0:
        raw = raw[0]
    try:
        return float(raw)
    except Exception:
        return float("nan")


def _rel_change(new: float, old: float, floor: float = 1.0e-12) -> float:
    if not (np.isfinite(new) and np.isfinite(old)):
        return float("nan")
    return float(abs(new - old) / max(abs(old), abs(new), float(floor)))


def _relax_positive_value(raw: float, old: float, relax: float) -> float:
    raw = max(float(raw), np.finfo(float).tiny)
    if not (np.isfinite(old) and old > 0.0):
        return raw
    w = min(max(float(relax), 0.0), 1.0)
    if w >= 1.0:
        return raw
    # Log blending avoids overshooting by orders of magnitude when R changes.
    return float(np.exp((1.0 - w) * np.log(old) + w * np.log(raw)))


def _run_index_from_path(path: Path) -> int:
    m = re.search(r"run_(\d+)$", path.name)
    return int(m.group(1)) if m else -1


def write_terminal_resistance_summary(
    run_dir: Path,
    rows: list[dict[str, Any]],
    write_cumulative: bool = True,
) -> None:
    fieldnames = [
        "run",
        "organoid",
        "side",
        "branch_id",
        "bc_name",
        "bc_type",
        "P_interface",
        "P_0D",
        "terminal_pressure_error_rel",
        "P_parent_0D",
        "P_ref_diagnostic",
        "P_d_for_R",
        "P_implied_resistance",
        "pressure_handoff_rel_gap",
        "pd_interface_rel_gap",
        "implied_interface_rel_gap",
        "pressure_handoff_abs_tol",
        "pd_interface_abs_gap",
        "implied_interface_abs_gap",
        "Pd_target",
        "Pd_target_reason",
        "suppression_pressure_tol",
        "Pd_previous",
        "Pd_next",
        "Pd_floor",
        "Pd_floor_applied",
        "Pd_floor_active_for_R",
        "Pd_abs_change",
        "Pd_rel_change",
        "Pd_relaxation_base",
        "Pd_relaxation_effective",
        "Pd_relaxation_flow_weight",
        "Pd_relaxation_q_ref",
        "terminal_response_correction_gain",
        "terminal_response_correction_effective_gain",
        "terminal_response_correction_severity",
        "terminal_response_correction_active",
        "terminal_response_correction_target",
        "terminal_response_correction_error_rel",
        "stubborn_continuity_override_active",
        "stubborn_continuity_override_relaxation",
        "stubborn_continuity_override_target",
        "stubborn_continuity_override_error_rel",
        "stubborn_terminal",
        "stubborn_terminal_gain",
        "stubborn_terminal_error_rel",
        "stubborn_terminal_recent_improvement_rel",
        "Q_0D",
        "Q_for_R",
        "Q_denominator",
        "R_previous",
        "R_raw",
        "R_target_reason",
        "R_next",
        "R_rel_change",
    ]
    summary_path = run_dir / "terminal_resistance_bc_summary.csv"
    with summary_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            if row.get("terminal_pressure_error_rel", "") == "":
                row = dict(row)
                row["terminal_pressure_error_rel"] = _pressure_continuity_rel_error(
                    _safe_float(row.get("P_0D")),
                    _safe_float(row.get("P_interface")),
                    floor=1.0,
                )
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    if not bool(write_cumulative):
        return

    cumulative_rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.parent.glob("run_*/terminal_resistance_bc_summary.csv"), key=lambda p: _run_index_from_path(p.parent)):
        with path.open("r", newline="") as fp:
            cumulative_rows.extend(csv.DictReader(fp))
    if cumulative_rows:
        cumulative_path = run_dir.parent / "terminal_pd_convergence.csv"
        with cumulative_path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in cumulative_rows:
                if row.get("terminal_pressure_error_rel", "") == "":
                    row = dict(row)
                    row["terminal_pressure_error_rel"] = _pressure_continuity_rel_error(
                        _safe_float(row.get("P_0D")),
                        _safe_float(row.get("P_interface")),
                        floor=1.0,
                    )
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def _terminal_pressure_error_rel_from_row(row: dict[str, Any]) -> float:
    p0 = _safe_float(row.get("P_0D"))
    pi = _safe_float(row.get("P_interface"))
    return _pressure_continuity_rel_error(p0, pi, floor=1.0)


def read_terminal_resistance_summary(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as fp:
        return list(csv.DictReader(fp))


def build_stubborn_terminal_response_map(
    coupled_root: Path,
    current_run: int,
    enabled: bool,
    min_rel_error: float,
    max_recent_improvement_rel: float,
    history_window: int,
    gain: float,
) -> dict[tuple[int, str, int], dict[str, float]]:
    if not enabled or gain <= 0.0:
        return {}
    prev_run = int(current_run) - 1
    if prev_run < 1:
        return {}
    prev_rows = read_terminal_resistance_summary(
        coupled_root / f"run_{prev_run}" / "terminal_resistance_bc_summary.csv"
    )
    if not prev_rows:
        return {}
    history_run = max(1, prev_run - max(int(history_window), 1))
    history_rows = read_terminal_resistance_summary(
        coupled_root / f"run_{history_run}" / "terminal_resistance_bc_summary.csv"
    )
    history_errors: dict[tuple[int, str, int], float] = {}
    for row in history_rows:
        try:
            key = (int(row.get("organoid")), str(row.get("side")), int(row.get("branch_id")))
        except Exception:
            continue
        history_errors[key] = _terminal_pressure_error_rel_from_row(row)

    out: dict[tuple[int, str, int], dict[str, float]] = {}
    for row in prev_rows:
        try:
            key = (int(row.get("organoid")), str(row.get("side")), int(row.get("branch_id")))
        except Exception:
            continue
        err = _terminal_pressure_error_rel_from_row(row)
        if not np.isfinite(err) or err < float(min_rel_error):
            continue
        old_err = history_errors.get(key, float("nan"))
        improvement = old_err - err if np.isfinite(old_err) else 0.0
        if improvement <= float(max_recent_improvement_rel):
            out[key] = {
                "gain": float(gain),
                "error_rel": float(err),
                "recent_improvement_rel": float(improvement),
            }
    return out


def _read_split_terminal_map(run_dir: Path, organoid_idx: int, side: str) -> dict[int, dict[str, float]]:
    path = run_dir / "split" / f"organoid_{organoid_idx}_{side}.csv"
    if not path.exists():
        return {}
    grouped: dict[int, dict[str, list[float]]] = {}
    with path.open("r", newline="") as fp:
        for row in csv.DictReader(fp):
            name = str(row.get("name", ""))
            m = re.search(r"branch\s+(\d+)_seg", name)
            if not m:
                continue
            branch_id = int(m.group(1))
            entry = grouped.setdefault(branch_id, {"pressure_out": [], "flow_out": []})
            entry["pressure_out"].append(_safe_float(row.get("pressure_out")))
            entry["flow_out"].append(_safe_float(row.get("flow_out")))
    out: dict[int, dict[str, float]] = {}
    for branch_id, values in grouped.items():
        out[branch_id] = {
            "pressure_out": float(np.nanmean(values["pressure_out"])),
            "flow_out": float(np.nanmean(values["flow_out"])),
        }
    return out


def score_resistance_0d_trial(
    trial_dir: Path,
    rows: list[dict[str, Any]],
    n_organoids: int,
    flow_guard_enabled: bool = True,
    flow_guard_tol: float = 0.02,
    flow_guard_weight: float = 1.0,
    pressure_weight: float = 10.0,
    inverse_flow_pressure_fraction: float = 0.5,
    inverse_flow_pressure_gamma: float = 0.5,
    inverse_flow_pressure_weight_cap: float = 10.0,
    implied_alignment_weight: float = 2.0,
    jump_weight: float = 0.05,
    max_pressure_error_weight: float = 2.0,
    stubborn_max_error_weight: float = 2.0,
    flow_guard_reject: bool = False,
    flow_guard_drive_tol: float = 1.0,
    flow_guard_q_floor: float = 1.0e-20,
) -> dict[str, float]:
    split_cache: dict[tuple[int, str], dict[int, dict[str, float]]] = {}

    def _split_value(organoid: int, side_label: str, branch_id: int) -> tuple[float, float]:
        split_side = "inlet" if side_label == "arterial" else "outlet"
        key = (int(organoid), split_side)
        if key not in split_cache:
            split_cache[key] = _read_split_terminal_map(trial_dir, int(organoid), split_side)
        data = split_cache[key].get(int(branch_id), {})
        return _safe_float(data.get("pressure_out")), _safe_float(data.get("flow_out"))

    arterial_p = [_safe_float(row.get("P_interface")) for row in rows if row.get("side") == "arterial"]
    venous_p = [_safe_float(row.get("P_interface")) for row in rows if row.get("side") == "venous"]
    pressure_span = abs(float(np.nanmean(arterial_p)) - float(np.nanmean(venous_p)))
    if not np.isfinite(pressure_span) or pressure_span <= 0.0:
        pressure_span = 1.0

    errors: list[float] = []
    active_errors: list[float] = []
    errors_by_side: dict[str, list[float]] = {"arterial": [], "venous": []}
    stubborn_errors: list[float] = []
    pressure_error_flow_samples: list[tuple[str, float, float]] = []
    proposed_implied_errors: list[float] = []
    proposed_implied_errors_by_side: dict[str, list[float]] = {"arterial": [], "venous": []}
    pressure_jumps: list[float] = []
    active_pressure_jumps: list[float] = []
    flow_guard_rel_changes: list[float] = []
    flow_guard_positive_increases: list[float] = []
    flow_guard_reductions: list[float] = []
    pressure_count = 0
    resistance_count = 0
    for row in rows:
        organoid = int(row.get("organoid", 0))
        if organoid < 1 or organoid > int(n_organoids):
            continue
        branch_id = int(row.get("branch_id", -1))
        side_label = str(row.get("side"))
        p_interface = _safe_float(row.get("P_interface"))
        if not np.isfinite(p_interface):
            continue
        p_proposed_implied = _safe_float(row.get("P_implied_resistance"))
        if np.isfinite(p_proposed_implied):
            proposed_err = abs(p_proposed_implied - p_interface) / pressure_span * 100.0
            proposed_implied_errors.append(float(proposed_err))
            if side_label in proposed_implied_errors_by_side:
                proposed_implied_errors_by_side[side_label].append(float(proposed_err))
        q_previous = _safe_float(row.get("Q_0D"))
        p_0d, q_0d = _split_value(organoid, side_label, branch_id)
        if not np.isfinite(p_0d):
            p_0d = _safe_float(row.get("P_0D"))
        if not np.isfinite(q_0d):
            q_0d = _safe_float(row.get("Q_0D"))
        if np.isfinite(p_0d):
            # This is the physically meaningful terminal-pressure mismatch:
            # the post-trial 0D terminal pressure should approach the Darcy
            # interface pressure.  Pd + RQ is only an auxiliary BC diagnostic.
            err = abs(p_0d - p_interface) / pressure_span * 100.0
            errors.append(float(err))
            if side_label in errors_by_side:
                errors_by_side[side_label].append(float(err))
            if np.isfinite(q_0d):
                pressure_error_flow_samples.append((side_label, float(err), abs(float(q_0d))))
            if row.get("R_target_reason") != "frozen_side":
                active_errors.append(float(err))
            if str(row.get("stubborn_terminal", "")).lower() == "true":
                stubborn_errors.append(float(err))
        bc_type = str(row.get("bc_type", "")).upper()
        if bc_type == "PRESSURE":
            pressure_jump = 0.0
            pressure_count += 1
        else:
            resistance = _safe_float(row.get("R_next"))
            pd_next = _safe_float(row.get("Pd_next"))
            if np.isfinite(pd_next) and np.isfinite(q_0d) and np.isfinite(resistance):
                # Candidate 0D outputs use the branch-outlet flow sign. This
                # matches SimVascular's RESISTANCE relation P = Pd + R Q.
                p_effective = pd_next + resistance * q_0d
                pressure_jump = abs(resistance * q_0d) / pressure_span * 100.0
            elif np.isfinite(p_0d):
                pressure_jump = float("nan")
            else:
                pressure_jump = float("nan")
            resistance_count += 1
        if np.isfinite(pressure_jump):
            pressure_jumps.append(float(pressure_jump))
        if row.get("R_target_reason") != "frozen_side":
            if np.isfinite(pressure_jump):
                active_pressure_jumps.append(float(pressure_jump))

        if flow_guard_enabled and np.isfinite(q_previous) and np.isfinite(q_0d):
            q_scale = max(abs(q_previous), float(flow_guard_q_floor), np.finfo(float).tiny)
            pi_previous = _safe_float(row.get("P_0D"))
            pd_previous = _safe_float(row.get("Pd_previous"))
            old_drive = abs(pi_previous - pd_previous) if np.isfinite(pi_previous) and np.isfinite(pd_previous) else float("nan")
            interface_drive = abs(pi_previous - p_interface) if np.isfinite(pi_previous) else float("nan")
            interface_would_open = (
                np.isfinite(old_drive)
                and np.isfinite(interface_drive)
                and interface_drive > old_drive + max(float(flow_guard_drive_tol), 0.0)
            )
            guarded_reason = str(row.get("Pd_target_reason", "")) in {
                "parent_flow_suppression",
                "drive_limited_hold",
            }
            if guarded_reason or interface_would_open:
                rel_flow_change = (abs(q_0d) - abs(q_previous)) / q_scale
                flow_guard_rel_changes.append(float(rel_flow_change))
                if rel_flow_change > max(float(flow_guard_tol), 0.0):
                    flow_guard_positive_increases.append(float(rel_flow_change))
                elif rel_flow_change < -max(float(flow_guard_tol), 0.0):
                    flow_guard_reductions.append(float(-rel_flow_change))

    if not errors:
        return {
            "score": float("inf"),
            "median_error_percent": float("inf"),
            "p90_error_percent": float("inf"),
            "p95_error_percent": float("inf"),
            "max_error_percent": float("inf"),
            "outlier_error_score": float("inf"),
            "active_median_error_percent": float("inf"),
            "jump_median_percent": float("inf"),
            "jump_p90_percent": float("inf"),
            "jump_max_percent": float("inf"),
            "active_jump_median_percent": float("inf"),
            "pressure_score": float("inf"),
            "inverse_flow_pressure_score": float("inf"),
            "arterial_inverse_flow_pressure_score": float("inf"),
            "venous_inverse_flow_pressure_score": float("inf"),
            "implied_interface_score": float("inf"),
            "implied_median_percent": float("inf"),
            "implied_p90_percent": float("inf"),
            "implied_max_percent": float("inf"),
            "arterial_implied_score": float("inf"),
            "venous_implied_score": float("inf"),
            "arterial_implied_median_percent": float("inf"),
            "venous_implied_median_percent": float("inf"),
            "arterial_implied_p90_percent": float("inf"),
            "venous_implied_p90_percent": float("inf"),
            "jump_score": float("inf"),
            "flow_guard_score": float("inf"),
            "flow_guard_count": 0.0,
            "flow_guard_increase_count": 0.0,
            "flow_guard_increase_fraction": 0.0,
            "flow_guard_increase_median_percent": float("inf"),
            "flow_guard_increase_p90_percent": float("inf"),
            "flow_guard_increase_max_percent": float("inf"),
            "flow_guard_reduction_median_percent": 0.0,
            "stubborn_max_error_percent": float("inf"),
            "stubborn_p90_error_percent": float("inf"),
            "stubborn_error_score": float("inf"),
            "pressure_bc_count": float(pressure_count),
            "resistance_bc_count": float(resistance_count),
        }

    errors_arr = np.array(errors, dtype=float)
    active_arr = np.array(active_errors if active_errors else errors, dtype=float)
    jump_arr = np.array(pressure_jumps if pressure_jumps else [0.0], dtype=float)
    active_jump_arr = np.array(
        active_pressure_jumps if active_pressure_jumps else pressure_jumps if pressure_jumps else [0.0],
        dtype=float,
    )
    median_err = float(np.nanmedian(errors_arr))
    p90_err = float(np.nanpercentile(errors_arr, 90.0))
    p95_err = float(np.nanpercentile(errors_arr, 95.0))
    max_err = float(np.nanmax(errors_arr))
    outlier_error_score = max_err + 0.5 * p95_err
    stubborn_arr = np.array(stubborn_errors if stubborn_errors else [0.0], dtype=float)
    stubborn_max_err = float(np.nanmax(stubborn_arr))
    stubborn_p90_err = float(np.nanpercentile(stubborn_arr, 90.0))
    stubborn_error_score = stubborn_max_err + 0.25 * stubborn_p90_err
    active_median = float(np.nanmedian(active_arr))
    jump_median = float(np.nanmedian(jump_arr))
    jump_p90 = float(np.nanpercentile(jump_arr, 90.0))
    jump_max = float(np.nanmax(jump_arr))
    active_jump_median = float(np.nanmedian(active_jump_arr))
    proposed_arr = np.array(proposed_implied_errors if proposed_implied_errors else errors, dtype=float)
    implied_median = float(np.nanmedian(proposed_arr))
    implied_p90 = float(np.nanpercentile(proposed_arr, 90.0))
    implied_max = float(np.nanmax(proposed_arr))

    def _side_implied_score(side: str) -> tuple[float, float, float]:
        vals = proposed_implied_errors_by_side.get(side, [])
        if not vals:
            return float("inf"), float("inf"), float("inf")
        arr = np.array(vals, dtype=float)
        med = float(np.nanmedian(arr))
        p90 = float(np.nanpercentile(arr, 90.0))
        maxv = float(np.nanmax(arr))
        return med + 0.5 * p90 + 0.1 * maxv, med, p90

    arterial_implied_score, arterial_implied_median, arterial_implied_p90 = _side_implied_score("arterial")
    venous_implied_score, venous_implied_median, venous_implied_p90 = _side_implied_score("venous")
    finite_side_scores = [
        value for value in (arterial_implied_score, venous_implied_score) if np.isfinite(value)
    ]
    if finite_side_scores:
        # Worst-side weighting prevents excellent arterial agreement from hiding
        # a lagging venous resistance handoff.
        implied_interface_score = (
            max(finite_side_scores) + 0.25 * float(np.mean(finite_side_scores))
        )
    else:
        implied_interface_score = implied_median + 0.5 * implied_p90 + 0.1 * implied_max

    def _side_pressure_score(side: str) -> tuple[float, float, float]:
        vals = errors_by_side.get(side, [])
        if not vals:
            return float("inf"), float("inf"), float("inf")
        arr = np.array(vals, dtype=float)
        med = float(np.nanmedian(arr))
        p90 = float(np.nanpercentile(arr, 90.0))
        maxv = float(np.nanmax(arr))
        return med + 0.5 * p90 + 0.1 * maxv, med, p90

    arterial_pressure_score, _, _ = _side_pressure_score("arterial")
    venous_pressure_score, _, _ = _side_pressure_score("venous")
    finite_pressure_scores = [
        value for value in (arterial_pressure_score, venous_pressure_score) if np.isfinite(value)
    ]
    if finite_pressure_scores:
        # Candidate selection should be driven by the true post-trial 0D
        # terminal pressure, not by the artificial resistance-implied pressure.
        # Worst-side weighting keeps a small subset of venous branches from
        # being hidden by many already-good arterial terminals.
        unweighted_pressure_score = max(finite_pressure_scores) + 0.25 * float(np.mean(finite_pressure_scores))
    else:
        unweighted_pressure_score = median_err + 0.25 * p90_err + 0.05 * max_err

    def _inverse_flow_weighted_pressure_score(side: str | None = None) -> float:
        samples = [
            (err, q_abs)
            for sample_side, err, q_abs in pressure_error_flow_samples
            if side is None or sample_side == side
        ]
        if not samples:
            return float("inf")
        q_values = np.array([q for _, q in samples if np.isfinite(q)], dtype=float)
        q_values = q_values[q_values > 0.0]
        q_ref = float(np.nanmedian(q_values)) if q_values.size else float(flow_guard_q_floor)
        q_ref = max(q_ref, float(flow_guard_q_floor), np.finfo(float).tiny)
        gamma = max(float(inverse_flow_pressure_gamma), 0.0)
        cap = max(float(inverse_flow_pressure_weight_cap), 1.0)
        weighted_sum = 0.0
        weight_sum = 0.0
        for err, q_abs in samples:
            if not np.isfinite(err):
                continue
            q_scale = max(float(q_abs), float(flow_guard_q_floor), np.finfo(float).tiny)
            weight = min((q_ref / q_scale) ** gamma, cap)
            weighted_sum += float(weight) * float(err)
            weight_sum += float(weight)
        return weighted_sum / weight_sum if weight_sum > 0.0 else float("inf")

    arterial_inverse_flow_score = _inverse_flow_weighted_pressure_score("arterial")
    venous_inverse_flow_score = _inverse_flow_weighted_pressure_score("venous")
    finite_inverse_scores = [
        value for value in (arterial_inverse_flow_score, venous_inverse_flow_score) if np.isfinite(value)
    ]
    if finite_inverse_scores:
        inverse_flow_pressure_score = max(finite_inverse_scores) + 0.25 * float(np.mean(finite_inverse_scores))
    else:
        inverse_flow_pressure_score = _inverse_flow_weighted_pressure_score(None)

    inv_fraction = min(max(float(inverse_flow_pressure_fraction), 0.0), 1.0)
    if np.isfinite(inverse_flow_pressure_score):
        pressure_score = (
            (1.0 - inv_fraction) * float(unweighted_pressure_score)
            + inv_fraction * float(inverse_flow_pressure_score)
        )
    else:
        pressure_score = float(unweighted_pressure_score)
    jump_score = jump_median + 0.25 * jump_p90 + 0.05 * jump_max
    positive_arr = np.array(flow_guard_positive_increases if flow_guard_positive_increases else [0.0], dtype=float)
    reduction_arr = np.array(flow_guard_reductions if flow_guard_reductions else [0.0], dtype=float)
    flow_guard_count = int(len(flow_guard_rel_changes))
    flow_guard_increase_count = int(len(flow_guard_positive_increases))
    flow_guard_increase_fraction = (
        float(flow_guard_increase_count / flow_guard_count)
        if flow_guard_count > 0
        else 0.0
    )
    flow_guard_increase_median = float(np.nanmedian(positive_arr) * 100.0)
    flow_guard_increase_p90 = float(np.nanpercentile(positive_arr, 90.0) * 100.0)
    flow_guard_increase_max = float(np.nanmax(positive_arr) * 100.0)
    flow_guard_reduction_median = float(np.nanmedian(reduction_arr) * 100.0)
    flow_guard_score = (
        100.0 * flow_guard_increase_fraction
        + flow_guard_increase_median
        + 0.25 * flow_guard_increase_p90
        + 0.05 * flow_guard_increase_max
    )
    score = (
        max(float(pressure_weight), 0.0) * pressure_score
        + max(float(implied_alignment_weight), 0.0) * implied_interface_score
        + max(float(jump_weight), 0.0) * jump_score
        + max(float(max_pressure_error_weight), 0.0) * outlier_error_score
        + max(float(stubborn_max_error_weight), 0.0) * stubborn_error_score
        + max(float(flow_guard_weight), 0.0) * flow_guard_score
    )
    if flow_guard_reject and flow_guard_increase_count > 0:
        # Keep the score finite so that if every candidate violates the guard,
        # the inner search still chooses the least unsafe option instead of
        # falling back to an unguarded default deck.
        score += 1.0e9
    return {
        "score": float(score),
        "median_error_percent": median_err,
        "p90_error_percent": p90_err,
        "p95_error_percent": p95_err,
        "max_error_percent": max_err,
        "outlier_error_score": float(outlier_error_score),
        "active_median_error_percent": active_median,
        "stubborn_max_error_percent": float(stubborn_max_err),
        "stubborn_p90_error_percent": float(stubborn_p90_err),
        "stubborn_error_score": float(stubborn_error_score),
        "jump_median_percent": jump_median,
        "jump_p90_percent": jump_p90,
        "jump_max_percent": jump_max,
        "active_jump_median_percent": active_jump_median,
        "pressure_score": float(pressure_score),
        "inverse_flow_pressure_score": float(inverse_flow_pressure_score),
        "arterial_inverse_flow_pressure_score": float(arterial_inverse_flow_score),
        "venous_inverse_flow_pressure_score": float(venous_inverse_flow_score),
        "implied_interface_score": float(implied_interface_score),
        "implied_median_percent": float(implied_median),
        "implied_p90_percent": float(implied_p90),
        "implied_max_percent": float(implied_max),
        "arterial_implied_score": float(arterial_implied_score),
        "venous_implied_score": float(venous_implied_score),
        "arterial_implied_median_percent": float(arterial_implied_median),
        "venous_implied_median_percent": float(venous_implied_median),
        "arterial_implied_p90_percent": float(arterial_implied_p90),
        "venous_implied_p90_percent": float(venous_implied_p90),
        "jump_score": float(jump_score),
        "flow_guard_score": float(flow_guard_score),
        "flow_guard_count": float(flow_guard_count),
        "flow_guard_increase_count": float(flow_guard_increase_count),
        "flow_guard_increase_fraction": float(flow_guard_increase_fraction),
        "flow_guard_increase_median_percent": float(flow_guard_increase_median),
        "flow_guard_increase_p90_percent": float(flow_guard_increase_p90),
        "flow_guard_increase_max_percent": float(flow_guard_increase_max),
        "flow_guard_reduction_median_percent": float(flow_guard_reduction_median),
        "pressure_bc_count": float(pressure_count),
        "resistance_bc_count": float(resistance_count),
    }


def write_inner_resistance_search_summary(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "run",
        "candidate",
        "selected",
        "resistance_relaxation_multiplier",
        "resistance_relaxation",
        "pd_relaxation",
        "continuity_resistance_decay",
        "pressure_handoff_enabled",
        "pressure_handoff_rel_tol",
        "terminal_response_correction_gain",
        "terminal_response_correction_guarded_reasons",
        "score",
        "pressure_score",
        "inverse_flow_pressure_score",
        "arterial_inverse_flow_pressure_score",
        "venous_inverse_flow_pressure_score",
        "implied_interface_score",
        "implied_median_percent",
        "implied_p90_percent",
        "implied_max_percent",
        "arterial_implied_score",
        "venous_implied_score",
        "arterial_implied_median_percent",
        "venous_implied_median_percent",
        "arterial_implied_p90_percent",
        "venous_implied_p90_percent",
        "jump_score",
        "flow_guard_score",
        "median_error_percent",
        "active_median_error_percent",
        "p90_error_percent",
        "p95_error_percent",
        "max_error_percent",
        "outlier_error_score",
        "stubborn_max_error_percent",
        "stubborn_p90_error_percent",
        "stubborn_error_score",
        "jump_median_percent",
        "active_jump_median_percent",
        "jump_p90_percent",
        "jump_max_percent",
        "flow_guard_count",
        "flow_guard_increase_count",
        "flow_guard_increase_fraction",
        "flow_guard_increase_median_percent",
        "flow_guard_increase_p90_percent",
        "flow_guard_increase_max_percent",
        "flow_guard_reduction_median_percent",
        "pressure_bc_count",
        "resistance_bc_count",
    ]
    path = run_dir / "inner_resistance_0d_search_summary.csv"
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_inner_resistance_candidate_grid(
    run_dir: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    if not rows:
        return
    fieldnames = [
        "candidate",
        "resistance_relaxation_multiplier",
        "resistance_relaxation",
        "pd_relaxation",
        "continuity_resistance_decay",
        "pressure_handoff_enabled",
        "pressure_handoff_rel_tol",
        "terminal_response_correction_gain",
        "terminal_response_correction_guarded_reasons",
    ]
    path = run_dir / "inner_resistance_0d_candidate_grid.csv"
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows):
            out = {key: row.get(key, "") for key in fieldnames}
            out["candidate"] = idx
            writer.writerow(out)
    save_json(run_dir / "inner_resistance_0d_search_config.json", config)


def apply_organoid_resistance_from_json(
    deck: dict,
    organoid_idx: int,
    iface: dict,
    resistance_relaxation: float,
    pd_relaxation: float,
    q_floor: float,
    venous_pd_floor: float | None = 0.0,
    pd_mode: str = "interface",
    suppression_pressure_tol: float = 5.0,
    continuity_pressure_tol: float = 5.0,
    continuity_resistance_decay: float = 0.8,
    pressure_alignment_jump_tol: float | None = None,
    active_terminal_side: str | None = "both",
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
    allow_missing_terminal_bcs: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skip_1d = bool(iface.get("skip_1d", False))
    artery_leak = float(iface.get("q_artery_leak", 0.0))
    vein_leak = float(iface.get("q_venous_leak", 0.0))

    inlet_bcs = find_bcs_with_prefix(deck, f"organoid{organoid_idx}_inlet_OUT")
    outlet_bcs = find_bcs_with_prefix(deck, f"organoid{organoid_idx}_outlet_OUT")
    leak_art = find_bcs_with_prefix(deck, f"LEAK_ART_{organoid_idx}")
    leak_ven = find_bcs_with_prefix(deck, f"LEAK_VEN_{organoid_idx}")

    if (len(inlet_bcs) == 0 or len(outlet_bcs) == 0) and not (allow_missing_terminal_bcs or skip_1d):
        die(f"Missing organoid interface BC groups for organoid {organoid_idx}")

    def _apply_side(side: str) -> None:
        if side == "inlet":
            bcs = inlet_bcs
            bc_prefix = f"organoid{organoid_idx}_inlet_OUT"
            p_interface = iface.get("p_inlet_nodes_raw", iface.get("p_inlet_nodes", []))
            p_0d_values = iface.get("p_inlet_target", [])
            p_parent_values = iface.get("p_inlet_parent_0d", p_0d_values)
            q_values = iface.get("q_inlet_target_in", iface.get("q_inlet_target", []))
            branch_ids = iface.get("branch_ids_inlet", [])
            side_label = "arterial"
        else:
            bcs = outlet_bcs
            bc_prefix = f"organoid{organoid_idx}_outlet_OUT"
            p_interface = iface.get("p_outlet_nodes_raw", iface.get("p_outlet_nodes", []))
            p_0d_values = iface.get("p_outlet_target", [])
            p_parent_values = iface.get("p_outlet_parent_0d", p_0d_values)
            q_values = iface.get("q_outlet_target_out", iface.get("q_outlet", []))
            branch_ids = iface.get("branch_ids_outlet", [])
            side_label = "venous"

        if not isinstance(p_interface, list):
            p_interface = []
        if not isinstance(p_0d_values, list):
            p_0d_values = []
        if not isinstance(p_parent_values, list):
            p_parent_values = []
        if not isinstance(q_values, list):
            q_values = []
        if not isinstance(branch_ids, list):
            branch_ids = []

        q_side_values: list[float] = []
        for q in q_values:
            q_val = _safe_float(q)
            if np.isfinite(q_val) and abs(q_val) > float(q_floor):
                q_side_values.append(abs(float(q_val)))
        q_side = np.array(q_side_values, dtype=float)
        q_ref_side = float(np.nanmedian(q_side)) if q_side.size else float(q_floor)
        q_ref_side = max(q_ref_side, float(q_floor), np.finfo(float).tiny)
        pd_relax_base = min(max(float(pd_relaxation), 0.0), 1.0)
        inv_pd_fraction = min(max(float(inverse_flow_pd_relaxation_fraction), 0.0), 1.0)
        inv_pd_gamma = max(float(inverse_flow_pd_relaxation_gamma), 0.0)
        inv_pd_cap = max(float(inverse_flow_pd_relaxation_weight_cap), 1.0)
        response_gain_base = max(float(terminal_response_correction_gain), 0.0)
        response_min_flow_weight = max(float(terminal_response_correction_min_flow_weight), 1.0)
        response_rel_error_tol = max(float(terminal_response_correction_rel_error_tol), 0.0)
        response_error_scale = max(float(terminal_response_correction_error_scale), response_rel_error_tol)
        response_error_gamma = max(float(terminal_response_correction_error_gamma), 0.0)
        response_effective_gain_cap = max(float(terminal_response_correction_effective_gain_cap), 0.0)
        response_guarded_reasons = bool(terminal_response_correction_guarded_reasons)

        def _effective_pd_relaxation(q_denominator: float) -> tuple[float, float]:
            if pd_relax_base <= 0.0 or inv_pd_fraction <= 0.0:
                return pd_relax_base, 1.0
            q_scale = max(float(q_denominator), float(q_floor), np.finfo(float).tiny)
            # Keep normal/high-flow terminals at the base relaxation while
            # giving small-flow, high-error terminals a stronger local move.
            flow_weight = max(1.0, min((q_ref_side / q_scale) ** inv_pd_gamma, inv_pd_cap))
            amplified = 1.0 - (1.0 - pd_relax_base) ** flow_weight
            effective = (1.0 - inv_pd_fraction) * pd_relax_base + inv_pd_fraction * amplified
            return min(max(float(effective), 0.0), 1.0), float(flow_weight)

        refs = _terminal_reference_values(iface, "inlet" if side == "inlet" else "outlet", len(p_interface))
        bc_by_name = {str(bc.get("bc_name", "")): bc for bc in bcs}
        for j, p_raw in enumerate(p_interface):
            branch_id = int(branch_ids[j]) if j < len(branch_ids) else j
            bc_name = f"{bc_prefix}{branch_id}"
            bc = bc_by_name.get(bc_name)
            if bc is None:
                if j >= len(bcs):
                    continue
                bc = bcs[j]
                bc_name = str(bc.get("bc_name", bc_name))

            p_darcy = float(p_raw)
            p_0d = float(p_0d_values[j]) if j < len(p_0d_values) else float("nan")
            p_parent = float(p_parent_values[j]) if j < len(p_parent_values) else float("nan")
            p_ref = float(refs[j]) if j < len(refs) else float("nan")
            q_0d = float(q_values[j]) if j < len(q_values) else float("nan")
            # The Darcy/interface convention stores venous drainage as positive
            # out of tissue, while the split 0D outlet trees are oriented from
            # the venous root toward the embedded terminal. SimVascular's
            # RESISTANCE BC uses the branch-outlet sign, so venous terminal
            # flows must be sign-flipped when evaluating Pd + R Q.
            q_for_r = -q_0d if side == "outlet" and np.isfinite(q_0d) else q_0d
            q_abs = abs(q_0d) if np.isfinite(q_0d) else 0.0
            q_den = max(q_abs, float(q_floor), np.finfo(float).tiny)
            stubborn_info = (
                stubborn_terminal_response_map or {}
            ).get((int(organoid_idx), side_label, int(branch_id)), {})
            stubborn_gain = _safe_float(stubborn_info.get("gain"))
            if not np.isfinite(stubborn_gain):
                stubborn_gain = 0.0
            stubborn_error_rel = _safe_float(stubborn_info.get("error_rel"))
            stubborn_recent_improvement_rel = _safe_float(
                stubborn_info.get("recent_improvement_rel")
            )
            if not np.isfinite(p_darcy):
                continue
            if not np.isfinite(p_0d):
                p_0d = _bc_first_numeric(bc, "P")
            if not np.isfinite(p_0d):
                p_0d = p_darcy
            if not np.isfinite(p_parent):
                p_parent = p_0d

            old_pd_from_resistance = _bc_first_numeric(bc, "Pd")
            old_pd = old_pd_from_resistance
            if not np.isfinite(old_pd):
                old_pd = _bc_first_numeric(bc, "P")
            old_r = _bc_first_numeric(bc, "R")
            update_this_side = active_terminal_side in (None, "both", side_label)
            if not update_this_side:
                old_bc_type = str(bc.get("bc_type", ""))
                handoff_abs_tol = max(float(pressure_handoff_abs_tol), 0.0)
                old_pressure_gap_rel = _rel_change(old_pd, p_darcy, floor=1.0)
                old_pressure_gap_abs = (
                    float(abs(old_pd - p_darcy))
                    if np.isfinite(old_pd) and np.isfinite(p_darcy)
                    else float("nan")
                )
                stale_pressure_handoff = (
                    old_bc_type.upper() == "PRESSURE"
                    and (
                        not np.isfinite(old_pressure_gap_rel)
                        or not np.isfinite(old_pressure_gap_abs)
                        or old_pressure_gap_rel > max(float(pressure_handoff_rel_tol), 0.0)
                        or old_pressure_gap_abs > handoff_abs_tol
                    )
                )
                if stale_pressure_handoff:
                    # A pressure handoff is only safe while it remains close to
                    # the current Darcy interface. If the opposite-side update
                    # moves the Darcy field, revert the frozen pressure pin to
                    # a resistance condition instead of letting it drive a
                    # large oscillation.
                    pd_next = p_darcy
                    pd_floor = float("nan")
                    pd_floor_applied = False
                    if side == "outlet" and venous_pd_floor is not None:
                        pd_floor = float(venous_pd_floor)
                        if np.isfinite(pd_floor) and pd_next < pd_floor:
                            pd_next = pd_floor
                            pd_floor_applied = True
                    r_raw = (
                        abs(old_pd - pd_next) / q_den
                        if np.isfinite(old_pd) and np.isfinite(pd_next)
                        else np.finfo(float).tiny
                    )
                    r_next = max(float(r_raw), np.finfo(float).tiny)
                    set_resistance_bc(bc, r_next, distal_pressure=pd_next)
                    rows.append({
                        "organoid": int(organoid_idx),
                        "side": side_label,
                        "branch_id": int(branch_id),
                        "bc_name": bc_name,
                        "bc_type": "RESISTANCE",
                        "P_interface": float(p_darcy),
                        "P_0D": float(p_0d),
                        "P_parent_0D": float(p_parent),
                        "P_ref_diagnostic": float(p_ref),
                        "P_d_for_R": float(pd_next),
                        "P_implied_resistance": (
                            float(pd_next + r_next * q_for_r)
                            if np.isfinite(q_0d)
                            else float("nan")
                        ),
                        "pressure_handoff_rel_gap": float("nan"),
                        "pd_interface_rel_gap": 0.0,
                        "implied_interface_rel_gap": (
                            _rel_change(pd_next + r_next * q_for_r, p_darcy, floor=1.0)
                            if np.isfinite(q_0d)
                            else float("nan")
                        ),
                        "pressure_handoff_abs_tol": float(handoff_abs_tol),
                        "pd_interface_abs_gap": 0.0,
                        "implied_interface_abs_gap": (
                            float(abs(pd_next + r_next * q_for_r - p_darcy))
                            if np.isfinite(q_0d)
                            else float("nan")
                        ),
                        "Pd_target": float(pd_next),
                        "Pd_target_reason": "pressure_handoff_reverted_frozen",
                        "suppression_pressure_tol": float(max(float(suppression_pressure_tol), 0.0)),
                        "Pd_previous": float(old_pd),
                        "Pd_next": float(pd_next),
                        "Pd_floor": float(pd_floor),
                        "Pd_floor_applied": bool(pd_floor_applied),
                        "Pd_floor_active_for_R": bool(pd_floor_applied),
                        "Pd_abs_change": (
                            float(abs(pd_next - old_pd))
                            if np.isfinite(old_pd)
                            else float("nan")
                        ),
                        "Pd_rel_change": _rel_change(pd_next, old_pd),
                        "Q_0D": float(q_0d),
                        "Q_for_R": float(q_for_r),
                        "Q_denominator": float(q_den),
                        "R_previous": float(old_r),
                        "R_raw": float(r_raw),
                        "R_target_reason": "pressure_handoff_reverted_frozen",
                        "R_next": float(r_next),
                        "R_rel_change": _rel_change(r_next, old_r),
                    })
                    continue
                rows.append({
                    "organoid": int(organoid_idx),
                    "side": side_label,
                    "branch_id": int(branch_id),
                    "bc_name": bc_name,
                    "bc_type": old_bc_type,
                    "P_interface": float(p_darcy),
                    "P_0D": float(p_0d),
                    "P_parent_0D": float(p_parent),
                    "P_ref_diagnostic": float(p_ref),
                    "P_d_for_R": float(old_pd),
                    "P_implied_resistance": float("nan"),
                    "pressure_handoff_rel_gap": float("nan"),
                    "pd_interface_rel_gap": old_pressure_gap_rel,
                    "implied_interface_rel_gap": float("nan"),
                    "pressure_handoff_abs_tol": float(handoff_abs_tol),
                    "pd_interface_abs_gap": old_pressure_gap_abs,
                    "implied_interface_abs_gap": float("nan"),
                    "Pd_target": float(old_pd),
                    "Pd_target_reason": "frozen_side",
                    "suppression_pressure_tol": float(max(float(suppression_pressure_tol), 0.0)),
                    "Pd_previous": float(old_pd),
                    "Pd_next": float(old_pd),
                    "Pd_floor": float("nan"),
                    "Pd_floor_applied": False,
                    "Pd_floor_active_for_R": False,
                    "Pd_abs_change": 0.0 if np.isfinite(old_pd) else float("nan"),
                    "Pd_rel_change": 0.0 if np.isfinite(old_pd) else float("nan"),
                    "Q_0D": float(q_0d),
                    "Q_for_R": float(q_for_r),
                    "Q_denominator": float(q_den),
                    "R_previous": float(old_r),
                    "R_raw": float(old_r),
                    "R_target_reason": "frozen_side",
                    "R_next": float(old_r),
                    "R_rel_change": 0.0 if np.isfinite(old_r) else float("nan"),
                })
                continue
            p_d_for_r = old_pd_from_resistance if np.isfinite(old_pd_from_resistance) else p_0d
            pd_floor = float("nan")
            pd_floor_active_for_resistance = False
            if side == "outlet" and venous_pd_floor is not None:
                pd_floor = float(venous_pd_floor)
                if np.isfinite(pd_floor) and p_darcy < pd_floor:
                    p_d_for_r = pd_floor
                    pd_floor_active_for_resistance = True
            r_raw = abs(p_darcy - p_d_for_r) / q_den
            r_target_reason = "pressure_jump_calibration"
            continuity_tol = max(float(continuity_pressure_tol), 0.0)
            pd_alignment_value = old_pd if np.isfinite(old_pd) else p_d_for_r
            pd_interface_gap = abs(pd_alignment_value - p_darcy) if np.isfinite(pd_alignment_value) else float("inf")
            continuity_decay_active = False
            if pd_interface_gap <= continuity_tol:
                # Once the resistance outlet pressure is already close to the
                # Darcy interface pressure, treat the added resistance as a
                # temporary continuation aid and explicitly decay it toward
                # zero. This lets the final state recover pressure continuity
                # instead of preserving a permanent pressure jump.
                decay = min(max(float(continuity_resistance_decay), 0.0), 1.0)
                if np.isfinite(old_r) and old_r > 0.0:
                    r_raw = max(old_r * decay, np.finfo(float).tiny)
                else:
                    r_raw = np.finfo(float).tiny
                r_target_reason = "continuity_decay"
                continuity_decay_active = True
            if continuity_decay_active:
                r_next = r_raw
            else:
                r_next = _relax_positive_value(r_raw, old_r, resistance_relaxation)
            pressure_tol = max(float(suppression_pressure_tol), 0.0)
            pd_target_reason = "0d_calibration" if pd_mode == "0d" else "interface"
            if pd_mode == "0d":
                pd_target = p_0d
            else:
                pressure_alignment_active = False
                interface_increases_drive = False
                if pressure_alignment_jump_tol is not None and np.isfinite(p_0d) and np.isfinite(old_pd):
                    pressure_alignment_active = abs(p_0d - old_pd) <= max(float(pressure_alignment_jump_tol), 0.0)
                if np.isfinite(p_0d) and np.isfinite(old_pd) and np.isfinite(p_darcy):
                    old_drive = abs(p_0d - old_pd)
                    aligned_drive = abs(p_0d - p_darcy)
                    # Pressure alignment is meant to remove the artificial
                    # resistance jump, not open a terminal harder. If chasing
                    # the Darcy pressure would increase the terminal pressure
                    # drive in the unsupported direction, keep the limiting
                    # logic active instead. For venous terminals, high tissue
                    # pressure is not unsupported: it is exactly the pressure
                    # state that can reverse the outlet tree and drain tissue.
                    drive_tol = max(float(pressure_alignment_jump_tol or 0.0), 0.0)
                    unsupported_drive_direction = (
                        (side == "inlet" and p_darcy > p_parent + pressure_tol)
                        or (side == "outlet" and p_darcy < p_parent - pressure_tol)
                    )
                    if aligned_drive > old_drive + drive_tol and unsupported_drive_direction:
                        interface_increases_drive = True
                        pressure_alignment_active = False
                flow_suppression_target = False
                if (
                    np.isfinite(p_parent)
                    and not pressure_alignment_active
                    and not continuity_decay_active
                ):
                    if side == "inlet":
                        # Arterial inflow is unsupported when tissue pressure exceeds
                        # its upstream parent pressure; moving Pd to the parent reduces
                        # the terminal pressure drop instead of amplifying reversal.
                        flow_suppression_target = p_darcy > p_parent + pressure_tol
                    else:
                        # Venous drainage is unsupported when tissue pressure falls
                        # below its parent pressure; chasing that lower pressure would
                        # increase suction, so target the parent instead.
                        flow_suppression_target = p_darcy < p_parent - pressure_tol
                if flow_suppression_target:
                    if np.isfinite(r_next) and np.isfinite(q_for_r):
                        # For a RESISTANCE BC the 0D-facing terminal pressure is
                        # P = Pd + R Q.  Suppressing flow toward the parent
                        # pressure must therefore target the implied pressure,
                        # not Pd itself; otherwise a large remaining RQ jump can
                        # drive the effective terminal pressure far past the
                        # parent value.
                        pd_target = p_parent - r_next * q_for_r
                    else:
                        pd_target = p_parent
                    pd_target_reason = "parent_flow_suppression"
                elif interface_increases_drive:
                    pd_target = old_pd
                    pd_target_reason = "drive_limited_hold"
                else:
                    if np.isfinite(r_next) and np.isfinite(q_for_r):
                        # A RESISTANCE BC imposes the terminal-side pressure as
                        # P = Pd + R Q, where Q uses the 0D branch-outlet sign.
                        # Target Pd so the pressure actually handed back to the
                        # 0D terminal, not just Pd itself, matches the Darcy
                        # interface pressure. This matters most after venous
                        # flow flips, because q_for_r < 0 and Pd must sit above
                        # the interface to offset the resistance drop.
                        pd_target = p_darcy - r_next * q_for_r
                        pd_target_reason = "resistance_implied_interface"
                    else:
                        pd_target = p_darcy
                    if pressure_alignment_active:
                        pd_target_reason = "pressure_alignment"
            _, pd_flow_weight_for_response = _effective_pd_relaxation(q_den)
            response_error_rel = (
                abs(p_darcy - p_0d) / max(abs(p_darcy), 1.0)
                if np.isfinite(p_darcy) and np.isfinite(p_0d)
                else float("nan")
            )
            response_target = float("nan")
            response_active = False
            response_severity = 0.0
            response_effective_gain = 0.0
            response_gain = response_gain_base + max(float(stubborn_gain), 0.0)
            response_allowed_reasons = {
                "resistance_implied_interface",
                "pressure_alignment",
                "pressure_handoff",
            }
            if response_guarded_reasons and response_gain_base > 0.0:
                # Candidate-only move: allow the cheap 0D search to test
                # pressure-response corrections even when the default update is
                # holding the terminal for flow-safety reasons. The candidate
                # score will reject it if the post-trial 0D pressure/flow gets
                # worse, so this does not become an unconditional Pd kick.
                response_allowed_reasons.update({
                    "drive_limited_hold",
                    "parent_flow_suppression",
                })
            if np.isfinite(response_error_rel) and response_error_rel > response_rel_error_tol:
                severity_den = max(
                    response_error_scale - response_rel_error_tol,
                    np.finfo(float).tiny,
                )
                response_severity = max(
                    (response_error_rel - response_rel_error_tol) / severity_den,
                    0.0,
                )
                response_severity = float(response_severity ** response_error_gamma)
                response_effective_gain = min(
                    response_gain * response_severity,
                    response_effective_gain_cap,
                )
            if (
                response_effective_gain > 0.0
                and pd_mode != "0d"
                and np.isfinite(r_next)
                and np.isfinite(q_for_r)
                and np.isfinite(p_darcy)
                and np.isfinite(p_0d)
                and pd_flow_weight_for_response >= response_min_flow_weight
                and np.isfinite(response_error_rel)
                and response_error_rel >= response_rel_error_tol
                and pd_target_reason in response_allowed_reasons
            ):
                # If Pd+RQ already matches Darcy but the actual 0D terminal
                # pressure is still lagging, compensate for the terminal-tree
                # pressure drop by deliberately targeting a stronger implied
                # boundary pressure. Candidate scoring decides whether this
                # helped the post-trial 0D pressure.
                response_target = p_darcy + response_effective_gain * (p_darcy - p_0d)
                pd_target = response_target - r_next * q_for_r
                pd_target_reason = f"{pd_target_reason}_response_correction"
                response_active = True
            if np.isfinite(old_pd):
                w_pd, pd_flow_weight = _effective_pd_relaxation(q_den)
                pd_next = (1.0 - w_pd) * old_pd + w_pd * pd_target
            else:
                w_pd, pd_flow_weight = _effective_pd_relaxation(q_den)
                pd_next = pd_target
            pd_floor_applied = False
            if side == "outlet" and venous_pd_floor is not None:
                if np.isfinite(pd_floor) and pd_next < pd_floor:
                    pd_next = pd_floor
                    pd_floor_applied = True
            p_implied = (
                pd_next + r_next * q_for_r
                if np.isfinite(pd_next) and np.isfinite(r_next) and np.isfinite(q_for_r)
                else float("nan")
            )
            pressure_handoff_rel_gap = _rel_change(p_implied, pd_next, floor=1.0)
            pd_interface_rel_gap = _rel_change(pd_next, p_darcy, floor=1.0)
            implied_interface_rel_gap = _rel_change(p_implied, p_darcy, floor=1.0)
            pd_interface_abs_gap = (
                abs(pd_next - p_darcy)
                if np.isfinite(pd_next) and np.isfinite(p_darcy)
                else float("nan")
            )
            implied_interface_abs_gap = (
                abs(p_implied - p_darcy)
                if np.isfinite(p_implied) and np.isfinite(p_darcy)
                else float("nan")
            )
            handoff_abs_tol = max(float(pressure_handoff_abs_tol), 0.0)
            pressure_handoff_active = (
                bool(pressure_handoff_enabled)
                and np.isfinite(pressure_handoff_rel_gap)
                and np.isfinite(pd_interface_rel_gap)
                and np.isfinite(implied_interface_rel_gap)
                and np.isfinite(pd_interface_abs_gap)
                and np.isfinite(implied_interface_abs_gap)
                and pressure_handoff_rel_gap <= max(float(pressure_handoff_rel_tol), 0.0)
                and pd_interface_rel_gap <= max(float(pressure_handoff_rel_tol), 0.0)
                and implied_interface_rel_gap <= max(float(pressure_handoff_rel_tol), 0.0)
                and pd_interface_abs_gap <= handoff_abs_tol
                and implied_interface_abs_gap <= handoff_abs_tol
            )
            if pressure_handoff_active:
                # Preserve the terminal pressure represented by the converged
                # resistance state instead of snapping directly to the Darcy
                # interface sample. The handoff checks above already require
                # p_implied, Pd, and p_darcy to be close.
                set_pressure_bc_constant(bc, float(p_implied))
                bc_type = "PRESSURE"
                pd_target_reason = "pressure_handoff"
                r_target_reason = "pressure_handoff"
            else:
                set_resistance_bc(bc, r_next, distal_pressure=pd_next)
                bc_type = "RESISTANCE"

            rows.append({
                "organoid": int(organoid_idx),
                "side": side_label,
                "branch_id": int(branch_id),
                "bc_name": bc_name,
                "bc_type": bc_type,
                "P_interface": float(p_darcy),
                "P_0D": float(p_0d),
                "P_parent_0D": float(p_parent),
                "P_ref_diagnostic": float(p_ref),
                "P_d_for_R": float(p_d_for_r),
                "P_implied_resistance": float(p_implied),
                "pressure_handoff_rel_gap": float(pressure_handoff_rel_gap),
                "pd_interface_rel_gap": float(pd_interface_rel_gap),
                "implied_interface_rel_gap": float(implied_interface_rel_gap),
                "pressure_handoff_abs_tol": float(handoff_abs_tol),
                "pd_interface_abs_gap": float(pd_interface_abs_gap),
                "implied_interface_abs_gap": float(implied_interface_abs_gap),
                "Pd_target": float(pd_target),
                "Pd_target_reason": str(pd_target_reason),
                "suppression_pressure_tol": float(pressure_tol),
                "Pd_previous": float(old_pd),
                "Pd_next": float(pd_next),
                "Pd_floor": float(pd_floor),
                "Pd_floor_applied": bool(pd_floor_applied),
                "Pd_floor_active_for_R": bool(pd_floor_active_for_resistance),
                "Pd_abs_change": float(abs(pd_next - old_pd)) if np.isfinite(old_pd) else float("nan"),
                "Pd_rel_change": _rel_change(pd_next, old_pd),
                "Pd_relaxation_base": float(pd_relax_base),
                "Pd_relaxation_effective": float(w_pd),
                "Pd_relaxation_flow_weight": float(pd_flow_weight),
                "Pd_relaxation_q_ref": float(q_ref_side),
                "terminal_response_correction_gain": float(response_gain),
                "terminal_response_correction_effective_gain": float(response_effective_gain),
                "terminal_response_correction_severity": float(response_severity),
                "terminal_response_correction_active": bool(response_active),
                "terminal_response_correction_target": float(response_target),
                "terminal_response_correction_error_rel": float(response_error_rel),
                "stubborn_continuity_override_active": False,
                "stubborn_continuity_override_relaxation": float("nan"),
                "stubborn_continuity_override_target": float("nan"),
                "stubborn_continuity_override_error_rel": float("nan"),
                "stubborn_terminal": bool(stubborn_gain > 0.0),
                "stubborn_terminal_gain": float(stubborn_gain),
                "stubborn_terminal_error_rel": float(stubborn_error_rel),
                "stubborn_terminal_recent_improvement_rel": float(stubborn_recent_improvement_rel),
                "Q_0D": float(q_0d),
                "Q_for_R": float(q_for_r),
                "Q_denominator": float(q_den),
                "R_previous": float(old_r),
                "R_raw": float(r_raw),
                "R_target_reason": str(r_target_reason),
                "R_next": float(r_next),
                "R_rel_change": _rel_change(r_next, old_r),
            })

    _apply_side("inlet")
    _apply_side("outlet")

    if leak_art:
        ensure_flow_bc(leak_art[0])
        set_bc_relaxed(leak_art[0], "Q", float(-artery_leak), 1.0)
    if leak_ven:
        ensure_flow_bc(leak_ven[0])
        set_bc_relaxed(leak_ven[0], "Q", float(-vein_leak), 1.0)

    return rows


def copy_seed_geometry(
    seed_run0: Path,
    run_dir: Path,
    n_organoids: int,
    no_synthetic_vasculature: bool = False,
) -> None:
    tmp_mesh = seed_run0 / "_tmp_mesh"

    required_geometry_files = [
        "bioreactor.xdmf",
        "mesh_tags.xdmf",
        "tagged_branches_inlet.bp",
        "tagged_branches_outlet.bp",
    ]
    if not no_synthetic_vasculature:
        required_geometry_files.extend([
            "pressure_checkpoint_inlet.bp",
            "pressure_checkpoint_outlet.bp",
            "flow_checkpoint_inlet.bp",
            "flow_checkpoint_outlet.bp",
            "area_checkpoint_inlet.bp",
            "area_checkpoint_outlet.bp",
        ])

    def _remove_existing_geometry(dst_geom: Path) -> None:
        if dst_geom.is_symlink() or dst_geom.is_file():
            dst_geom.unlink()
        elif dst_geom.exists():
            shutil.rmtree(dst_geom)

    def _copy_tmp_mesh_well(tmp_dir: Path, dst_geom: Path, k: int) -> None:
        # Reuse run_all.py geometry copier to stay aligned with pipeline behavior.
        repo_src = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_src / "prep"))
        run_all_mod = importlib.import_module("run_all")
        copy_shared_geometry_outputs = getattr(run_all_mod, "_copy_shared_geometry_outputs")
        dst_geom.mkdir(parents=True, exist_ok=True)
        # Copy canonical geometry/checkpoint files exactly as run_all.py does.
        copy_shared_geometry_outputs(
            tmp_dir,
            dst_geom,
            well_idx=k,
            copy_1d_artifacts=(not no_synthetic_vasculature),
        )

        # Also copy branched-network XDMFs (+h5 sidecars), needed by generate_1d_files.
        src_in = tmp_dir / f"branched_network_inlet_well{k}.xdmf"
        src_out = tmp_dir / f"branched_network_outlet_well{k}.xdmf"
        if src_in.exists():
            copy_xdmf_with_sidecars(src_in, dst_geom / "branched_network_inlet.xdmf")
        if src_out.exists():
            copy_xdmf_with_sidecars(src_out, dst_geom / "branched_network_outlet.xdmf")

    def _validate_geometry_inputs(dst_geom: Path, k: int) -> None:
        missing = [name for name in required_geometry_files if not (dst_geom / name).exists()]
        if missing:
            die(
                "Geometry preparation failed for "
                f"organoid_{k}; missing required Darcy inputs in {dst_geom}: "
                + ", ".join(missing)
            )

    for k in range(1, n_organoids + 1):
        src = seed_run0 / f"organoid_{k}" / "geometry"
        dst = run_dir / f"organoid_{k}" / "geometry"
        dst.parent.mkdir(parents=True, exist_ok=True)
        _remove_existing_geometry(dst)

        # Prefer _tmp_mesh because it stores well-specific geometry/checkpoints
        # and run_all.py knows how to remap them to the canonical Darcy names.
        # This avoids reusing stale/incomplete run_0 geometry directories.
        if tmp_mesh.exists():
            _copy_tmp_mesh_well(tmp_mesh, dst, k)
            _validate_geometry_inputs(dst, k)
            continue

        src_has_required_geometry = src.exists() and all(
            (src / name).exists() for name in required_geometry_files
        )
        src_has_branched_networks = (
            no_synthetic_vasculature
            or (
                (
                    (src / "branched_network_inlet.xdmf").exists()
                    or (src / f"branched_network_inlet_well{k}.xdmf").exists()
                )
                and (
                    (src / "branched_network_outlet.xdmf").exists()
                    or (src / f"branched_network_outlet_well{k}.xdmf").exists()
                )
            )
        )
        # Geometry is static across coupling iterations. If _tmp_mesh is not
        # available, a complete seed geometry directory can still be reused.
        if src_has_required_geometry and src_has_branched_networks:
            link_or_copy_tree(src, dst)
        else:
            die(f"Missing seed geometry for organoid_{k}: {src} and no fallback {tmp_mesh}")
        _validate_geometry_inputs(dst, k)


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


def copy_branching_files(
    trial_dir: Optional[Path],
    run_dir: Path,
    n_organoids: int,
    allow_missing: bool = False,
) -> None:
    """
    Copy branchingData_0.csv and branchingData_1.csv from trial directory into
    each run/organoid_k folder.
    """
    if trial_dir is None:
        if allow_missing:
            return
        die("copy_branching_files requires a valid trial directory when allow_missing is false")
    for k in range(1, n_organoids + 1):
        src_org = trial_dir / f"organoid_{k}"
        dst_org = run_dir / f"organoid_{k}"
        dst_org.mkdir(parents=True, exist_ok=True)
        for name in ("branchingData_0.csv", "branchingData_1.csv"):
            src = src_org / name
            if not src.exists():
                if allow_missing:
                    continue
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


def shifted_seed_fallback(base: np.ndarray, organoid_idx: int, dy_step: float) -> np.ndarray:
    coords = np.asarray(base, dtype=float).copy()
    coords[1] += -float(dy_step) * float(organoid_idx - 1)
    return coords


def _latest_named_row(csv_path: Path, vessel_name: str) -> Optional[dict]:
    if not csv_path.exists():
        return None

    latest_row: Optional[dict] = None
    latest_time = float("-inf")
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("name", "")).strip() != vessel_name:
                continue
            try:
                t = float(row.get("time", 0.0))
            except Exception:
                t = 0.0
            if latest_row is None or t >= latest_time:
                latest_row = row
                latest_time = t
    return latest_row


def _read_pressure_column(row: Optional[dict], column: str) -> Optional[float]:
    if row is None:
        return None
    raw = row.get(column, None)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except Exception:
        return None


def read_leak_pressures_from_output(
    output_csv: Path,
    organoid_idx: int,
) -> tuple[Optional[float], Optional[float]]:
    art_row = _latest_named_row(output_csv, f"leak_art_{organoid_idx}")
    ven_row = _latest_named_row(output_csv, f"leak_ven_{organoid_idx}")
    return _read_pressure_column(art_row, "pressure_out"), _read_pressure_column(ven_row, "pressure_out")

def resolve_leak_pressures_for_darcy(
    primary_output_csv: Path,
    organoid_idx: int,
    fallback_output_csv: Optional[Path] = None,
) -> tuple[Optional[float], Optional[float]]:
    p_art, p_ven = read_leak_pressures_from_output(primary_output_csv, organoid_idx)
    if fallback_output_csv is not None and (p_art is None or p_ven is None):
        fb_art, fb_ven = read_leak_pressures_from_output(fallback_output_csv, organoid_idx)
        if p_art is None:
            p_art = fb_art
        if p_ven is None:
            p_ven = fb_ven
    return p_art, p_ven


def build_first_organoid_linear_profile_from_leaks(
    primary_output_csv: Path,
    face_length: float,
    compartment_spacing: float,
    fallback_output_csv: Optional[Path] = None,
) -> Optional[dict]:
    if float(compartment_spacing) == 0.0:
        return None

    p_art_1, p_ven_1 = resolve_leak_pressures_for_darcy(
        primary_output_csv,
        1,
        fallback_output_csv=fallback_output_csv,
    )
    p_art_2, p_ven_2 = resolve_leak_pressures_for_darcy(
        primary_output_csv,
        2,
        fallback_output_csv=fallback_output_csv,
    )
    if any(v is None for v in (p_art_1, p_art_2, p_ven_1, p_ven_2)):
        return None

    ratio = float(face_length) / float(compartment_spacing)
    art_drop_12 = float(p_art_2) - float(p_art_1)
    ven_drop_12 = float(p_ven_2) - float(p_ven_1)

    # For organoid_1 there is no upstream venous well, so extrapolate one
    # compartment upstream using the same 1->2 venous pressure drop.
    p_ven_0 = float(p_ven_1) - ven_drop_12

    return {
        "profile_mode": "linear",
        "profile_axis": "y",
        # The solver applies "low/high" at min/max of the global profile axis.
        # For these faces that geometric direction is reversed relative to the
        # original left-to-right assumption, so swap the endpoints here.
        "arterial_low": float(p_art_1) + art_drop_12 * ratio,
        "arterial_high": float(p_art_1) - art_drop_12 * ratio,
        "venous_low": float(p_ven_1) + (float(p_ven_1) - p_ven_0) * ratio,
        "venous_high": float(p_ven_1) - (float(p_ven_1) - p_ven_0) * ratio,
        "arterial_center": float(p_art_1),
        "venous_center": float(p_ven_1),
        "arterial_drop_12": art_drop_12,
        "venous_drop_12": ven_drop_12,
        "venous_virtual_previous": p_ven_0,
    }


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

        c_in = read_seed_coords_from_branching(
            org / "branchingData_0.csv",
            shifted_seed_fallback(np.asarray(coords_inlet, dtype=float), k, dy_step),
        )
        c_out = read_seed_coords_from_branching(
            org / "branchingData_1.csv",
            shifted_seed_fallback(np.asarray(coords_outlet, dtype=float), k, dy_step),
        )

        mesh_mod.current_dir = geom
        inlet_output_csv = in_dir / "output.csv"
        inlet_geom_csv = in_dir / "geom.csv"
        outlet_output_csv = out_dir / "output.csv"
        outlet_geom_csv = out_dir / "geom.csv"

        inlet_generator = (
            mesh_mod.generate_1d_files_from_csv
            if hasattr(mesh_mod, "generate_1d_files_from_csv")
            and inlet_output_csv.exists()
            and inlet_geom_csv.exists()
            else mesh_mod.generate_1d_files
        )
        outlet_generator = (
            mesh_mod.generate_1d_files_from_csv
            if hasattr(mesh_mod, "generate_1d_files_from_csv")
            and outlet_output_csv.exists()
            and outlet_geom_csv.exists()
            else mesh_mod.generate_1d_files
        )

        inlet_kwargs = dict(
            xdmf_file=xdmf_in.name,
            output_dir=str(in_dir),
            file_prefix="_inlet",
            inlet_coords=c_in,
        )
        outlet_kwargs = dict(
            xdmf_file=xdmf_out.name,
            output_dir=str(out_dir),
            file_prefix="_outlet",
            inlet_coords=c_out,
        )
        if inlet_generator is mesh_mod.generate_1d_files_from_csv:
            inlet_kwargs["rewrite_branch_tags"] = True
        if outlet_generator is mesh_mod.generate_1d_files_from_csv:
            outlet_kwargs["rewrite_branch_tags"] = True

        inlet_generator(**inlet_kwargs)
        outlet_generator(**outlet_kwargs)


def required_darcy_geometry_inputs(no_synthetic_vasculature: bool = False) -> list[str]:
    required = [
        "bioreactor.xdmf",
        "mesh_tags.xdmf",
        "tagged_branches_inlet.bp",
        "tagged_branches_outlet.bp",
    ]
    if not no_synthetic_vasculature:
        required.extend([
            "pressure_checkpoint_inlet.bp",
            "pressure_checkpoint_outlet.bp",
            "flow_checkpoint_inlet.bp",
            "flow_checkpoint_outlet.bp",
            "area_checkpoint_inlet.bp",
            "area_checkpoint_outlet.bp",
        ])
    return required


def validate_darcy_geometry_inputs(run_dir: Path, n_organoids: int, no_synthetic_vasculature: bool = False) -> None:
    required = required_darcy_geometry_inputs(no_synthetic_vasculature)
    missing_by_organoid: list[str] = []
    for k in range(1, n_organoids + 1):
        geom = run_dir / f"organoid_{k}" / "geometry"
        missing = [name for name in required if not (geom / name).exists()]
        if missing:
            missing_by_organoid.append(
                f"organoid_{k}: {', '.join(missing)} in {geom}"
            )
    if missing_by_organoid:
        die(
            "Missing required Darcy geometry/checkpoint inputs before Darcy launch. "
            "The run likely has stale legacy geometry files such as pressure_inlet.bp "
            "instead of pressure_checkpoint_inlet.bp. Missing: "
            + " | ".join(missing_by_organoid)
        )


def run_darcy_for_all(
    run_dir: Path,
    args: argparse.Namespace,
    scaled_cfg: Optional[dict[str, Any]] = None,
    replicate_reference_outputs: bool = False,
) -> None:
    validate_darcy_geometry_inputs(
        run_dir,
        int(args.n_organoids),
        no_synthetic_vasculature=bool(args.no_synthetic_vasculature),
    )
    darcy_script = Path(args.darcy_script).expanduser().resolve()
    if not darcy_script.exists():
        die(f"Missing Darcy script: {darcy_script}")
    output_csv = run_dir / "output.csv"
    trial_output_csv = None
    if getattr(args, "trial_dir", ""):
        trial_output_csv = Path(args.trial_dir).expanduser().resolve() / "output.csv"

    reference_organoid = int(scaled_cfg["reference_organoid"]) if scaled_cfg is not None else 1
    solve_organoid_ids = (
        [reference_organoid]
        if scaled_cfg is not None
        else list(range(1, int(args.n_organoids) + 1))
    )

    concave_profiles_by_organoid: dict[int, dict[str, Any]] = {}
    if getattr(args, "first_organoid_linear_profile", False):
        first_organoid_profile = build_first_organoid_linear_profile_from_leaks(
            output_csv,
            face_length=float(args.concave_profile_face_length),
            compartment_spacing=float(args.concave_profile_compartment_spacing),
            fallback_output_csv=trial_output_csv,
        )
        if first_organoid_profile is not None:
            concave_profiles_by_organoid[1] = first_organoid_profile
        elif args.n_organoids >= 2 and MPI.COMM_WORLD.rank == 0:
            print(
                "[warn] Could not build first-organoid linear concave pressure profile from leak pressures; "
                "falling back to constant concave pressures.",
                flush=True,
            )

    for k in solve_organoid_ids:
        org = run_dir / f"organoid_{k}"
        geom = org / "geometry"
        c_in = read_seed_coords_from_branching(
            org / "branchingData_0.csv",
            shifted_seed_fallback(np.array(args.coords_inlet, dtype=float), k, args.dy_step),
        )
        c_out = read_seed_coords_from_branching(
            org / "branchingData_1.csv",
            shifted_seed_fallback(np.array(args.coords_outlet, dtype=float), k, args.dy_step),
        )
        perm_region_path = resolve_perm_region_for_organoid(args.perm_region_root, k) if args.perm_region_root else ""
        fallback_inlet_pressure = float(args.fallback_inlet_pressure)
        fallback_outlet_pressure = float(args.fallback_outlet_pressure)
        p_art, p_ven = resolve_leak_pressures_for_darcy(
            output_csv,
            k,
            fallback_output_csv=trial_output_csv,
        )
        if p_art is not None:
            fallback_inlet_pressure = float(p_art)
        if p_ven is not None:
            fallback_outlet_pressure = float(p_ven)

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
            "--inlet-output-csv", str(org / "0D_Input_Files" / "inlet" / "output.csv"),
            "--outlet-output-csv", str(org / "0D_Input_Files" / "outlet" / "output.csv"),
            "--concave-bc-mode", str(args.concave_bc_mode),
            "--lp-arterial", str(args.lp_arterial),
            "--lp-venous", str(args.lp_venous),
            *(["--skip-1d"] if args.no_synthetic_vasculature else []),
            "--fallback-inlet-pressure", str(fallback_inlet_pressure),
            "--fallback-outlet-pressure", str(fallback_outlet_pressure),
            "--out-dir", str(org),
        ]
        if p_art is not None:
            darcy_args.extend(["--concave-inlet-pressure", str(float(p_art))])
        if p_ven is not None:
            darcy_args.extend(["--concave-outlet-pressure", str(float(p_ven))])
        concave_profile = concave_profiles_by_organoid.get(int(k))
        if concave_profile is not None:
            darcy_args.extend([
                "--concave-pressure-profile", str(concave_profile["profile_mode"]),
                "--concave-pressure-profile-axis", str(concave_profile["profile_axis"]),
                "--arterial-pressure-profile-low", str(concave_profile["arterial_low"]),
                "--arterial-pressure-profile-high", str(concave_profile["arterial_high"]),
                "--venous-pressure-profile-low", str(concave_profile["venous_low"]),
                "--venous-pressure-profile-high", str(concave_profile["venous_high"]),
            ])
        if darcy_script.stem in {"darcy_p1_lm_interior", "darcy_mixed"} and perm_region_path:
            darcy_args.extend([
                "--perm-region-path", perm_region_path,
                "--perm-low", str(args.perm_low),
                "--perm-high", str(args.perm_high),
                "--perm-transition-width", str(args.perm_transition_width),
            ])
        if darcy_script.stem == "darcy_mixed":
            darcy_args.extend(["--output-mode", str(args.darcy_output_mode)])
        if int(args.darcy_mpi_procs) > 1:
            cmd = build_mpi_command(
                str(args.darcy_mpirun_cmd),
                int(args.darcy_mpi_procs),
                [sys.executable, str(darcy_script), *darcy_args],
            )
        else:
            cmd = [sys.executable, str(darcy_script), *darcy_args]
        run(cmd)

    if scaled_cfg is not None:
        if str(getattr(args, "darcy_output_mode", "full")) == "minimal":
            replicate_reference_organoid_outputs(
                run_dir,
                int(args.n_organoids),
                scaled_cfg,
                copy_field_outputs=False,
            )
        elif replicate_reference_outputs:
            replicate_reference_organoid_outputs(run_dir, int(args.n_organoids), scaled_cfg)
        else:
            synthesize_scaled_screening_outputs(run_dir, int(args.n_organoids), scaled_cfg)


def _max_rel(a: np.ndarray, b: np.ndarray, floor: float = 1e-20) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    aa = a[:n]
    bb = b[:n]
    den = np.maximum(np.abs(bb), floor)
    return float(np.max(np.abs(aa - bb) / den))


def _max_pressure_continuity_rel(a: np.ndarray, b: np.ndarray, floor: float = 1.0) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    aa = a[:n]
    bb = b[:n]
    den = np.maximum(np.minimum(np.abs(aa), np.abs(bb)), float(floor))
    return float(np.max(np.abs(aa - bb) / den))


def parse_0d_branch_segment_name(name: str) -> Optional[tuple[int, int]]:
    text = str(name).strip()
    patterns = [
        r"branch\s+(-?\d+)_seg(-?\d+)",
        r"branch_(-?\d+)_seg_?(-?\d+)",
        r".*branch\s*[_ ](-?\d+).*seg\s*[_ ]?(-?\d+)",
    ]
    for pat in patterns:
        m = re.match(pat, text)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None


def read_terminal_physical_flows_from_0d_csv(
    csv_path: Path,
    branch_ids: list[Any],
    side: str,
    checkpoint_time_index: int = -1,
) -> Optional[np.ndarray]:
    """
    Read terminal segment-0 flows from a split 0D output.csv.

    The returned sign convention matches interface_bc.json:
    - inlet/arterial: positive means flow into tissue.
    - outlet/venous: positive means flow out of tissue.
    """
    if not csv_path.exists():
        return None

    rows_by_branch: dict[int, list[tuple[float, float]]] = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = parse_0d_branch_segment_name(row.get("name", ""))
                if parsed is None:
                    continue
                branch, segment = parsed
                if int(segment) != 0:
                    continue
                try:
                    time = float(row.get("time", 0.0))
                    flow = float(row.get("flow_out", row.get("flow_in", "nan")))
                except Exception:
                    continue
                if np.isfinite(flow):
                    rows_by_branch.setdefault(int(branch), []).append((time, flow))
    except Exception:
        return None

    if not rows_by_branch:
        return None

    out = []
    for raw_id in branch_ids:
        try:
            branch_id = int(raw_id)
        except Exception:
            return None
        rows = sorted(rows_by_branch.get(branch_id, []), key=lambda item: item[0])
        if not rows:
            return None
        idx = int(checkpoint_time_index)
        if idx < 0:
            idx = len(rows) + idx
        idx = min(max(idx, 0), len(rows) - 1)
        flow = float(rows[idx][1])
        if side == "inlet":
            out.append(abs(flow))
        elif side == "outlet":
            out.append(-flow)
        else:
            raise ValueError(f"Unknown terminal side {side!r}")
    return np.asarray(out, dtype=float)


def read_terminal_parent_pressures_from_0d_csv(
    csv_path: Path,
    branch_ids: list[Any],
    checkpoint_time_index: int = -1,
) -> Optional[np.ndarray]:
    if not csv_path.exists():
        return None

    rows_by_branch: dict[int, list[tuple[float, float]]] = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = parse_0d_branch_segment_name(row.get("name", ""))
                if parsed is None:
                    continue
                branch, segment = parsed
                if int(segment) != 0:
                    continue
                try:
                    time = float(row.get("time", 0.0))
                    pressure = float(row["pressure_in"])
                except Exception:
                    continue
                if np.isfinite(pressure):
                    rows_by_branch.setdefault(int(branch), []).append((time, pressure))
    except Exception:
        return None

    if not rows_by_branch:
        return None

    out = []
    for raw_id in branch_ids:
        try:
            branch_id = int(raw_id)
        except Exception:
            return None
        rows = sorted(rows_by_branch.get(branch_id, []), key=lambda item: item[0])
        if not rows:
            return None
        idx = int(checkpoint_time_index)
        if idx < 0:
            idx = len(rows) + idx
        idx = min(max(idx, 0), len(rows) - 1)
        out.append(float(rows[idx][1]))
    return np.asarray(out, dtype=float)


def _terminal_arrays_for_side(iface: dict, side: str) -> tuple[list, list, list, list, list, list]:
    if side == "inlet":
        return (
            iface.get("branch_ids_inlet", []),
            iface.get("p_inlet_nodes_raw", iface.get("p_inlet_nodes", [])),
            iface.get("p_inlet_parent_0d", iface.get("p_inlet_target", [])),
            iface.get("q_inlet_port_in", iface.get("q_inlet", [])),
            iface.get("q_inlet_target_in", iface.get("q_inlet_target", [])),
            iface.get("q_inlet_target", []),
        )
    if side == "outlet":
        return (
            iface.get("branch_ids_outlet", []),
            iface.get("p_outlet_nodes_raw", iface.get("p_outlet_nodes", [])),
            iface.get("p_outlet_parent_0d", iface.get("p_outlet_target", [])),
            iface.get("q_outlet_port_out", iface.get("q_outlet", [])),
            iface.get("q_outlet_target_out", []),
            iface.get("q_outlet_target_out", []),
        )
    raise ValueError(f"Unknown terminal side {side!r}")


def build_pressure_reduction_targets(
    deck: dict,
    iface_by_organoid: dict[int, dict],
    n_organoids: int,
    support_flow_factor: float,
    support_pressure_rel_tol: float,
    support_mass_balance_tol: float,
    support_q_floor: float,
) -> dict[tuple[int, str, int], dict[str, float]]:
    resistance_by_bc = terminal_resistance_by_bc_name(deck)
    targets: dict[tuple[int, str, int], dict[str, float]] = {}

    for organoid_idx in range(1, int(n_organoids) + 1):
        iface = iface_by_organoid.get(organoid_idx)
        if iface is None or bool(iface.get("skip_1d", False)):
            continue
        mass_balance = float(iface.get("mass_balance_direct_port_relative", 0.0))

        for side in ("inlet", "outlet"):
            ids, p_raw, p_parent, q_darcy, q_target, _q_raw = _terminal_arrays_for_side(iface, side)
            if not all(isinstance(x, list) for x in (ids, p_raw, p_parent, q_darcy, q_target)):
                continue
            prefix = f"organoid{organoid_idx}_{side}_OUT"
            for j in range(min(len(ids), len(p_raw), len(p_parent), len(q_darcy), len(q_target))):
                try:
                    branch_id = int(ids[j])
                    p_darcy = float(p_raw[j])
                    parent = float(p_parent[j])
                    q_port = float(q_darcy[j])
                    q_0d = abs(float(q_target[j]))
                except Exception:
                    continue

                bc_name = f"{prefix}{branch_id}"
                bc = find_bc(deck, bc_name)
                cur = _bc_scalar_value(bc, "P")
                if cur is None:
                    continue
                cur = float(cur)

                unsupported = terminal_flow_is_unsupported(
                    p_darcy=p_darcy,
                    p_parent=parent,
                    q_darcy=q_port,
                    q_0d=q_0d,
                    mass_balance_direct_port_relative=mass_balance,
                    flow_factor=float(support_flow_factor),
                    pressure_rel_tol=float(support_pressure_rel_tol),
                    mass_balance_tol=float(support_mass_balance_tol),
                    q_floor=float(support_q_floor),
                )
                current_drive = abs(cur - parent) if np.isfinite(cur) and np.isfinite(parent) else 0.0
                darcy_drive = abs(p_darcy - parent) if np.isfinite(p_darcy) and np.isfinite(parent) else 0.0
                pressure_over_requested = darcy_drive > current_drive * (1.0 + float(support_pressure_rel_tol))
                if not (unsupported or pressure_over_requested):
                    continue
                if not np.isfinite(darcy_drive) or darcy_drive <= 0.0:
                    continue

                flow_scale = min(1.0, current_drive / max(darcy_drive, np.finfo(float).tiny))
                q_desired = q_0d * flow_scale
                resistance = abs(float(resistance_by_bc.get(bc_name, float("nan"))))
                if not np.isfinite(resistance) or resistance <= 0.0:
                    resistance = current_drive / max(q_0d, float(support_q_floor), np.finfo(float).tiny)
                desired_drive = min(current_drive, resistance * q_desired)
                if not np.isfinite(desired_drive):
                    continue

                targets[(organoid_idx, side, branch_id)] = {
                    "drive": float(desired_drive),
                    "q_desired": float(q_desired),
                    "old_parent": float(parent),
                    "old_pressure": float(cur),
                    "raw_darcy_pressure": float(p_darcy),
                    "flow_scale": float(flow_scale),
                }
    return targets


def recenter_terminal_pressures_on_latest_0d_parents(
    deck: dict,
    run_dir: Path,
    targets: dict[tuple[int, str, int], dict[str, float]],
    n_organoids: int,
) -> int:
    changed = 0
    for organoid_idx in range(1, int(n_organoids) + 1):
        for side in ("inlet", "outlet"):
            branch_ids = [
                branch_id
                for (org_idx, target_side, branch_id) in targets
                if org_idx == organoid_idx and target_side == side
            ]
            if not branch_ids:
                continue
            csv_path = run_dir / f"organoid_{organoid_idx}" / "0D_Input_Files" / side / "output.csv"
            parents = read_terminal_parent_pressures_from_0d_csv(csv_path, branch_ids)
            if parents is None:
                continue
            for branch_id, parent in zip(branch_ids, parents):
                target = targets[(organoid_idx, side, int(branch_id))]
                drive = float(target["drive"])
                if side == "inlet":
                    p_next = float(parent) - drive
                else:
                    p_next = float(parent) + drive
                bc = find_bc(deck, f"organoid{organoid_idx}_{side}_OUT{int(branch_id)}")
                if bc is None:
                    continue
                old = _bc_scalar_value(bc, "P")
                set_bc_constant(bc, "P", p_next)
                if old is None or abs(float(old) - p_next) > 1.0e-14:
                    changed += 1
    return changed


def pressure_reduction_target_flow_scores(
    run_dir: Path,
    targets: dict[tuple[int, str, int], dict[str, float]],
) -> dict[str, float]:
    sums = {"inlet": 0.0, "outlet": 0.0}
    counts = {"inlet": 0, "outlet": 0}
    for organoid_idx in sorted({key[0] for key in targets}):
        for side in ("inlet", "outlet"):
            branch_ids = [
                branch_id
                for (org_idx, target_side, branch_id) in targets
                if org_idx == organoid_idx and target_side == side
            ]
            if not branch_ids:
                continue
            csv_path = run_dir / f"organoid_{organoid_idx}" / "0D_Input_Files" / side / "output.csv"
            q_measured = read_terminal_physical_flows_from_0d_csv(csv_path, branch_ids, side)
            if q_measured is None:
                continue
            q_desired = np.asarray(
                [targets[(organoid_idx, side, int(branch_id))]["q_desired"] for branch_id in branch_ids],
                dtype=float,
            )
            n = min(q_measured.size, q_desired.size)
            if n == 0:
                continue
            den = np.maximum(np.abs(q_desired[:n]), 1.0e-20)
            sums[side] += float(np.sum(np.abs(q_measured[:n] - q_desired[:n]) / den))
            counts[side] += int(n)
    inlet = sums["inlet"] / counts["inlet"] if counts["inlet"] else 0.0
    outlet = sums["outlet"] / counts["outlet"] if counts["outlet"] else 0.0
    total_n = counts["inlet"] + counts["outlet"]
    total = (sums["inlet"] + sums["outlet"]) / total_n if total_n else 0.0
    return {
        "inlet": float(inlet),
        "outlet": float(outlet),
        "total": float(total),
        "n_inlet": float(counts["inlet"]),
        "n_outlet": float(counts["outlet"]),
    }


def _flow_score_components(
    q_darcy: np.ndarray,
    q_0d: np.ndarray,
    q_ref: np.ndarray,
    q_floor: float,
) -> tuple[float, int]:
    n = min(q_darcy.size, q_0d.size, q_ref.size)
    if n == 0:
        return 0.0, 0
    qd = np.asarray(q_darcy[:n], dtype=float)
    qz = np.asarray(q_0d[:n], dtype=float)
    qr = np.asarray(q_ref[:n], dtype=float)
    mask = np.isfinite(qd) & np.isfinite(qz) & np.isfinite(qr)
    if not np.any(mask):
        return 0.0, 0
    den = np.maximum.reduce([
        np.abs(qr[mask]),
        np.abs(qd[mask]),
        np.full(int(np.sum(mask)), float(q_floor), dtype=float),
    ])
    return float(np.sum(np.abs(qz[mask] - qd[mask]) / den)), int(np.sum(mask))


def terminal_flow_acceptance_scores(
    run_dir: Optional[Path],
    iface_by_organoid: dict[int, dict],
    n_organoids: int,
    q_floor: float,
    use_0d_csv: bool,
) -> dict[str, float]:
    sums = {"inlet": 0.0, "outlet": 0.0}
    counts = {"inlet": 0, "outlet": 0}

    for k in range(1, int(n_organoids) + 1):
        iface = iface_by_organoid.get(k)
        if iface is None or bool(iface.get("skip_1d", False)):
            continue

        for side in ("inlet", "outlet"):
            if side == "inlet":
                branch_ids = iface.get("branch_ids_inlet", [])
                q_darcy = np.asarray(iface.get("q_inlet_port_in", iface.get("q_inlet", [])), dtype=float)
                q_old = np.asarray(iface.get("q_inlet_target_in", iface.get("q_inlet_target", [])), dtype=float)
                csv_path = (
                    run_dir / f"organoid_{k}" / "0D_Input_Files" / "inlet" / "output.csv"
                    if run_dir is not None
                    else Path()
                )
            else:
                branch_ids = iface.get("branch_ids_outlet", [])
                q_darcy = np.asarray(iface.get("q_outlet_port_out", iface.get("q_outlet", [])), dtype=float)
                q_old = np.asarray(iface.get("q_outlet_target_out", []), dtype=float)
                csv_path = (
                    run_dir / f"organoid_{k}" / "0D_Input_Files" / "outlet" / "output.csv"
                    if run_dir is not None
                    else Path()
                )

            if use_0d_csv:
                q_new = read_terminal_physical_flows_from_0d_csv(csv_path, branch_ids, side)
                if q_new is None:
                    # If we cannot read the candidate 0D output, make the
                    # candidate unattractive rather than accidentally accepting it.
                    return {"inlet": float("inf"), "outlet": float("inf"), "total": float("inf")}
            else:
                q_new = q_old

            score_sum, n = _flow_score_components(q_darcy, np.asarray(q_new, dtype=float), q_old, q_floor)
            sums[side] += score_sum
            counts[side] += n

    inlet = sums["inlet"] / counts["inlet"] if counts["inlet"] else 0.0
    outlet = sums["outlet"] / counts["outlet"] if counts["outlet"] else 0.0
    total_count = counts["inlet"] + counts["outlet"]
    total = (sums["inlet"] + sums["outlet"]) / total_count if total_count else 0.0
    return {
        "inlet": float(inlet),
        "outlet": float(outlet),
        "total": float(total),
        "n_inlet": float(counts["inlet"]),
        "n_outlet": float(counts["outlet"]),
    }


def flow_scores_improved(
    candidate: dict[str, float],
    baseline: dict[str, float],
    accept_tol: float,
) -> bool:
    if not np.isfinite(candidate.get("total", float("inf"))):
        return False
    tol = float(accept_tol)
    inlet_ok = candidate["inlet"] <= baseline["inlet"] * (1.0 + tol) + tol
    outlet_ok = candidate["outlet"] <= baseline["outlet"] * (1.0 + tol) + tol
    total_ok = candidate["total"] < baseline["total"] * (1.0 - tol)
    return bool(inlet_ok and outlet_ok and total_ok)


def _bc_scalar_value(bc: Optional[dict], key: str) -> Optional[float]:
    if bc is None:
        return None
    vals = bc.get("bc_values", {})
    arr = vals.get(key, None)
    if isinstance(arr, list) and len(arr) > 0:
        try:
            return float(arr[0])
        except Exception:
            return None
    return None


def _current_leak_flow_targets(deck: dict, organoid_idx: int) -> Optional[np.ndarray]:
    leak_art = find_bcs_with_prefix(deck, f"LEAK_ART_{organoid_idx}")
    leak_ven = find_bcs_with_prefix(deck, f"LEAK_VEN_{organoid_idx}")
    q_art_bc = _bc_scalar_value(leak_art[0], "Q") if leak_art else None
    q_ven_bc = _bc_scalar_value(leak_ven[0], "Q") if leak_ven else None
    if q_art_bc is None or q_ven_bc is None:
        return None
    # Deck BCs are stored with opposite sign from Darcy leak fluxes.
    return np.asarray([-q_art_bc, -q_ven_bc], dtype=float)


def convergence_for_organoid(
    iface_path: Path,
    tol_q: float,
    tol_p: float,
    prev_iface_path: Optional[Path] = None,
    current_deck: Optional[dict] = None,
    organoid_idx: Optional[int] = None,
    pressure_compare_mode: str = "target",
) -> tuple[bool, float, float]:
    iface = load_json(iface_path)
    if bool(iface.get("skip_1d", False)):
        p = np.asarray(
            [iface.get("p_concave_inlet_bc", 0.0), iface.get("p_concave_outlet_bc", 0.0)],
            dtype=float,
        )
        if prev_iface_path is None or not prev_iface_path.exists():
            return False, float("inf"), float("inf")
        prev = load_json(prev_iface_path)
        p_t = np.asarray(
            [prev.get("p_concave_inlet_bc", 0.0), prev.get("p_concave_outlet_bc", 0.0)],
            dtype=float,
        )
        rp = _max_pressure_continuity_rel(p, p_t, floor=1.0)
        return (rp <= tol_p), float("nan"), rp

    p_in = np.asarray(iface.get("p_inlet_nodes_raw", iface.get("p_inlet_nodes", [])), dtype=float)
    p_out = np.asarray(iface.get("p_outlet_nodes_raw", iface.get("p_outlet_nodes", [])), dtype=float)

    if pressure_compare_mode == "previous":
        if prev_iface_path is None or not prev_iface_path.exists():
            return False, float("inf"), float("inf")
        prev = load_json(prev_iface_path)
        p_in_t = np.asarray(prev.get("p_inlet_nodes_raw", prev.get("p_inlet_nodes", [])), dtype=float)
        p_out_t = np.asarray(prev.get("p_outlet_nodes_raw", prev.get("p_outlet_nodes", [])), dtype=float)
    else:
        p_in_t = np.asarray(iface.get("p_inlet_target", []), dtype=float)
        p_out_t = np.asarray(iface.get("p_outlet_target", []), dtype=float)

    if p_in.size == 0 and p_out.size == 0:
        return False, float("inf"), float("inf")

    rp = max(
        _max_pressure_continuity_rel(p_in, p_in_t, floor=1.0),
        _max_pressure_continuity_rel(p_out, p_out_t, floor=1.0),
    )
    return (rp <= tol_p), float("nan"), rp


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Organoid coupling driver with geometry reuse + P1-LM Darcy.")
    ap.add_argument("--template-combined", default="../prep/scaled-screening/Run2_10branches/combined.in")
    ap.add_argument("--trial-dir", default="../prep/scaled-screening/Run2_10branches/",
                    help="Optional trial directory used to copy branchingData_{0,1}.csv into each run. Not required in --no-synthetic-vasculature mode.")
    ap.add_argument("--seed-run0", default="../prep/scaled-screening/Run2_10branches",
                    help="Existing run_0 folder with 3D meshing/tagging to reuse.")
    ap.add_argument("--coupled-root", default="../coupling-output",)

    ap.add_argument("--n-organoids", type=int, default=4)
    ap.add_argument("--n-ramp", type=int, default=1)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--start-run", type=int, default=1,
                    help="First coupling run to build. Use 1 for a fresh run from run_0, or a larger value to restart from run_{start_run-1}.")
    ap.add_argument("--tol-q", type=float, default=1e-3,
                    help="Legacy flow tolerance retained for compatibility; steady resistance-mode convergence now uses terminal pressure continuity only.")
    ap.add_argument("--tol-p", type=float, default=1e-3,
                    help="Tolerance for max |P_0D-P_Darcy|/max(min(|P_0D|,|P_Darcy|),1) across terminals.")
    ap.add_argument("--relaxation", type=float, default=1.0)
    ap.add_argument("--terminal-support-flow-factor", type=float, default=10.0,
                    help="Treat terminal flow as unsupported when |Q_0D| exceeds this multiple of |Q_Darcy_port|.")
    ap.add_argument("--terminal-support-pressure-rel-tol", type=float, default=0.1,
                    help="Relative |P_Darcy-P_parent| threshold used with flow over-request detection.")
    ap.add_argument("--terminal-support-mass-balance-tol", type=float, default=0.1,
                    help="mass_balance_direct_port_relative threshold used with flow over-request detection.")
    ap.add_argument("--terminal-support-q-floor", type=float, default=1.0e-20,
                    help="Small denominator floor for terminal flow-support detection.")
    ap.add_argument("--inner-0d-max-tries", type=int, default=10,
                    help="Maximum cheap 0D parent-recentering attempts before each Darcy solve. Use 1 to run 0D once.")
    ap.add_argument("--inner-0d-backtrack-factor", type=float, default=0.5,
                    help="Factor used to reduce terminal pressure relaxation after a rejected inner 0D attempt.")
    ap.add_argument("--inner-0d-accept-tol", type=float, default=5.0e-2,
                    help="Mean relative tolerance for stopping the inner 0D parent-recentering loop once terminal flows reach the reduced targets.")
    ap.add_argument("--terminal-bc-mode", choices=["pressure", "resistance"], default="resistance",
                    help="Use pressure terminal feedback for all runs, or use run_1 as pressure calibration and run_2+ as resistance terminal BCs.")
    ap.add_argument("--resistance-calibration-run", type=int, default=1,
                    help="Last run index that still uses pressure terminal BCs before switching to resistance BCs.")
    ap.add_argument("--terminal-resistance-q-floor", type=float, default=1.0e-20,
                    help="Denominator floor in R=|P_interface-P_0D|/max(|Q|, floor).")
    ap.add_argument("--terminal-resistance-relaxation", type=float, default=0.1,
                    help="Log-space relaxation for terminal resistance updates; 1.0 applies the calibrated R directly.")
    ap.add_argument("--terminal-pd-relaxation", type=float, default=0.1,
                    help="Linear relaxation for resistance distal pressure Pd updates.")
    ap.add_argument("--terminal-venous-pd-floor", type=float, default=None,
                    help="Optional lower bound for venous resistance Pd. Omit to allow negative distal pressure.")
    ap.add_argument("--terminal-suppression-pressure-tol", type=float, default=5.0,
                    help="Absolute pressure deadband for parent-flow suppression in resistance mode.")
    ap.add_argument("--terminal-continuity-pressure-tol", type=float, default=5.0,
                    help="When |Pd-P_interface| is below this tolerance, decay terminal resistance toward zero to recover pressure continuity.")
    ap.add_argument("--terminal-continuity-resistance-decay", type=float, default=0.95,
                    help="Target multiplicative decay for terminal resistance once Pd is aligned with the Darcy interface pressure.")
    ap.add_argument("--terminal-pressure-alignment-jump-tol", type=float, default=1.0,
                    help="When |Pi-Pd| is below this absolute tolerance, treat resistance continuation as complete and update Pd toward the Darcy interface instead of using parent-flow suppression.")
    ap.add_argument("--terminal-side-update-mode", choices=["alternate", "both"], default="both",
                    help="In resistance mode, update both terminal sides every iteration or alternate arterial/venous updates to reduce fixed-point ping-pong.")
    ap.add_argument("--terminal-alternate-first-side", choices=["arterial", "venous"], default="arterial",
                    help="First active terminal side after the resistance calibration run when --terminal-side-update-mode=alternate.")
    ap.add_argument("--terminal-pressure-handoff", action=argparse.BooleanOptionalAction, default=True,
                    help="Switch an active terminal from RESISTANCE to PRESSURE once Pi=Pd+RQ and Pd agree within the handoff tolerance.")
    ap.add_argument("--terminal-pressure-handoff-rel-tol", type=float, default=0.025,
                    help="Relative tolerance for pressure handoff, using |Pi-Pd|/max(|Pi|,|Pd|,1). Default 0.05 = 5%%.")
    ap.add_argument("--terminal-pressure-handoff-abs-tol", type=float, default=2.0,
                    help="Absolute pressure tolerance for pressure handoff. Pd and Pd+RQ must both be within this distance of the Darcy interface pressure.")
    ap.add_argument("--inner-resistance-0d-search", action=argparse.BooleanOptionalAction, default=True,
                    help="Use cheap 0D trial solves to choose the best resistance/Pd/handoff update before each Darcy solve.")
    ap.add_argument(
        "--keep-inner-resistance-trials",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep bulky _inner_resistance_0d_trials candidate folders after scoring. "
            "Default is false: keep only the candidate-grid/search-summary CSVs."
        ),
    )
    ap.add_argument("--inner-resistance-search-profile", choices=["compact", "full", "custom"], default="compact",
                    help="Candidate-grid profile. compact uses the historically useful subset; full restores the broad grid; custom uses the raw candidate-list arguments exactly.")
    ap.add_argument(
        "--terminal-pd-convergence-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Write cumulative terminal_pd_convergence.csv after each run. "
            "Disable for lean long runs; per-run terminal_resistance_bc_summary.csv is still written."
        ),
    )
    ap.add_argument("--inner-resistance-decay-candidates", default="1.0,0.85,0.70",
                    help="Comma-separated continuity resistance decay candidates for the inner 0D search.")
    ap.add_argument("--inner-resistance-relaxation-multipliers", default="0.5,1.0,2.0,4.0",
                    help="Comma-separated multipliers applied to --terminal-resistance-relaxation in the inner 0D search. These directly vary R_next, so candidates remain distinct even before pressure handoff/continuity decay activate.")
    ap.add_argument("--inner-pd-relaxation-candidates", default="0.05,0.10,0.20,0.40",
                    help="Comma-separated Pd relaxation candidates for the inner 0D search. Larger safe values can move Pd toward Darcy faster.")
    ap.add_argument("--inner-pressure-handoff-tol-candidates", default="0.05,0.10",
                    help="Comma-separated pressure handoff tolerances tested in the inner 0D search.")
    ap.add_argument("--inner-resistance-flow-guard", action=argparse.BooleanOptionalAction, default=True,
                    help="During inner 0D candidate scoring, penalize candidates that increase |Q| on terminals whose Darcy update would open the branch harder.")
    ap.add_argument("--inner-flow-guard-tol", type=float, default=0.02,
                    help="Relative |Q| increase tolerated before the inner flow guard penalizes a candidate.")
    ap.add_argument("--inner-flow-guard-weight", type=float, default=1.0,
                    help="Weight for the inner 0D flow-guard penalty added to the candidate score.")
    ap.add_argument("--inner-0d-pressure-weight", type=float, default=10.0,
                    help="Weight for post-trial 0D terminal pressure alignment P_0D versus Darcy interface pressure in the candidate score.")
    ap.add_argument("--inner-inverse-flow-pressure-fraction", type=float, default=0.5,
                    help="Fraction of the pressure score using inverse-flow weighting, so low-flow terminals influence candidate selection.")
    ap.add_argument("--inner-inverse-flow-pressure-gamma", type=float, default=0.5,
                    help="Exponent for inverse-flow pressure weighting: weight=(median(|Q|)/|Q_i|)^gamma.")
    ap.add_argument("--inner-inverse-flow-pressure-weight-cap", type=float, default=10.0,
                    help="Maximum inverse-flow pressure weight for a single terminal.")
    ap.add_argument("--terminal-inverse-flow-pd-relaxation-fraction", type=float, default=0.5,
                    help="Blend fraction for locally increasing Pd relaxation on low-flow terminals.")
    ap.add_argument("--terminal-inverse-flow-pd-relaxation-gamma", type=float, default=0.5,
                    help="Exponent for low-flow Pd relaxation amplification: multiplier=(median(|Q|)/|Q_i|)^gamma.")
    ap.add_argument("--terminal-inverse-flow-pd-relaxation-weight-cap", type=float, default=1000.0,
                    help="Maximum low-flow Pd relaxation amplification for a single terminal.")
    ap.add_argument("--inner-terminal-response-correction-gains", default="0.0, 1.0",
                    help="Comma-separated candidate gains for low-flow terminal response correction. gain=0 preserves the original update.")
    ap.add_argument("--terminal-response-correction-min-flow-weight", type=float, default=2.0,
                    help="Only apply response correction to terminals whose inverse-flow Pd weight is at least this value.")
    ap.add_argument("--terminal-response-correction-rel-error-tol", type=float, default=0.02,
                    help="Only apply response correction when |P_interface-P_0D|/max(|P_interface|,1) exceeds this threshold.")
    ap.add_argument("--terminal-response-correction-error-scale", type=float, default=0.10,
                    help="Pressure-continuity error where the response correction reaches the requested candidate gain.")
    ap.add_argument("--terminal-response-correction-error-gamma", type=float, default=1.0,
                    help="Exponent for pressure-error severity weighting of response correction.")
    ap.add_argument("--terminal-response-correction-effective-gain-cap", type=float, default=3.0,
                    help="Maximum per-terminal effective response-correction gain after pressure-error severity weighting.")
    ap.add_argument("--inner-response-correction-guarded-reasons", action=argparse.BooleanOptionalAction, default=True,
                    help="In inner 0D candidate search, allow positive response-correction candidates to test drive-limited/parent-suppressed terminals. The scored 0D trial decides whether to accept the move.")
    ap.add_argument("--stubborn-terminal-response-correction", action=argparse.BooleanOptionalAction, default=True,
                    help="Automatically add extra response correction to terminals with large, slowly improving pressure-continuity error.")
    ap.add_argument("--stubborn-terminal-min-rel-error", type=float, default=0.025,
                    help="Minimum |P_interface-P_0D|/max(|P_interface|,1) for marking a terminal as stubborn.")
    ap.add_argument("--stubborn-terminal-max-recent-improvement", type=float, default=0.005,
                    help="Maximum recent decrease in relative pressure error for a terminal to be considered stubborn.")
    ap.add_argument("--stubborn-terminal-history-window", type=int, default=1,
                    help="Number of previous coupling intervals used to estimate recent stubborn-terminal improvement.")
    ap.add_argument("--stubborn-terminal-extra-gain", type=float, default=100.0,
                    help="Extra response-correction gain added only to stubborn terminals.")
    ap.add_argument("--inner-implied-alignment-weight", type=float, default=2.0,
                    help="Weight for candidate-proposed auxiliary P_implied versus Darcy interface alignment in the inner 0D score.")
    ap.add_argument("--inner-jump-weight", type=float, default=0.05,
                    help="Weight for the temporary resistance pressure-jump penalty in the candidate score.")
    ap.add_argument("--inner-max-pressure-error-weight", type=float, default=2.0,
                    help="Weight for the global worst-terminal post-trial 0D pressure error in the inner candidate score.")
    ap.add_argument("--inner-stubborn-max-error-weight", type=float, default=2.0,
                    help="Weight for the max post-trial 0D pressure error among currently stubborn terminals in the inner candidate score.")
    ap.add_argument("--inner-flow-guard-reject", action=argparse.BooleanOptionalAction, default=False,
                    help="If enabled, reject any inner 0D candidate that increases guarded terminal |Q| beyond tolerance.")

    ap.add_argument("--channel-inlet-bc", default="INFLOW")
    ap.add_argument("--channel-inlet-Q0", type=float, default=6e-2)
    ap.add_argument("--channel-outlet-bc", default="OUT3")
    ap.add_argument("--channel-outlet-P0", type=float, default=0.0)
    ap.add_argument("--use-outlet-pressure-ramp", action="store_true",
                    help="Ramp channel outlet pressure from 0 at i=0 to P0 by the end of the ramp.")

    ap.add_argument("--svzerodsolver", default="svzerodsolver")
    ap.add_argument("--run-and-split", default="../prep/run_and_split_svzerod.py")
    ap.add_argument("--darcy-script", default="../solves/darcy_mixed.py")
    ap.add_argument("--darcy-mpi-procs", type=int, default=1,
                    help="MPI ranks for Darcy solves (1 = serial).")
    ap.add_argument("--darcy-mpirun-cmd", default="/opt/miniconda3/envs/fenicsx-env/bin/mpiexec.hydra",
                    help="MPI launcher command for Darcy when --darcy-mpi-procs > 1.")
    ap.add_argument(
        "--darcy-output-mode",
        choices=["full", "fields", "minimal"],
        default="full",
        help=(
            "Darcy output mode passed to darcy_mixed.py. Use 'minimal' during "
            "coupling iterations to write only interface_bc.json and avoid large field files."
        ),
    )
    ap.add_argument("--scaled-screening-mode", choices=["auto", "on", "off"], default="auto",
                    help="When enabled, solve only the reference organoid with Darcy and synthesize the remaining repeated-geometry wells using 0D-based scaling. 'auto' enables this when trial_dir contains scaled_screening_summary.json.")
    ap.add_argument("--scaled-pressure-offset-anchor", choices=["arterial", "venous"], default="arterial",
                    help="Fallback pressure-offset anchor for scaled-screening mode when no scaled_screening_summary.json is available.")
    ap.add_argument("--first-organoid-linear-profile", action="store_true",
                    help="For organoid_1, use organoid_2 leak pressures to build linear concave arterial/venous pressure profiles.")
    ap.add_argument("--concave-profile-face-length", type=float, default=0.4,
                    help="Half-span/length factor L1 used to scale the concave-face pressure profile from neighboring leak pressures.")
    ap.add_argument("--concave-profile-compartment-spacing", type=float, default=0.6,
                    help="Compartment spacing L2 used to scale the concave-face pressure profile from neighboring leak pressures.")
    ap.add_argument("--dy-step", type=float, default=0.6)
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-.28, 0.9, .5375])
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.30, 0.9, .5375])
    ap.add_argument("--perm-region-root", default=str(Path(__file__).resolve().parents[2] / "files" / "stl" / "organoid-growth-domains" / "sphere"),
                    help="Directory, STL path, or pattern for organoid permeability regions. If a directory is given, organoid-k uses organoid-k.stl. You can also use a path containing 'X' as the organoid index placeholder.")
    # ap.add_argument("--perm-low", type=float, default=5.0e-11)
    # ap.add_argument("--perm-high", type=float, default=1.0e-8)

    ap.add_argument("--perm-low", type=float, default=1.0e-9)
    ap.add_argument("--perm-high", type=float, default=2.0e-7)
    ap.add_argument("--perm-transition-width", type=float, default=0.01)

    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    ap.add_argument("--lp-arterial", type=float, default=0.0)
    ap.add_argument("--lp-venous", type=float, default=0.0)
    ap.add_argument("--no-synthetic-vasculature", action="store_true",
                    help="Use leak pressures to drive Darcy and skip organoid 1D terminal coupling.")
    ap.add_argument("--fallback-inlet-pressure", type=float, default=0.0,
                    help="Fallback arterial concave pressure when leak_art_k is unavailable in output.csv.")
    ap.add_argument("--fallback-outlet-pressure", type=float, default=0.0,
                    help="Fallback venous concave pressure when leak_ven_k is unavailable in output.csv.")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--plot-each-iteration", action=argparse.BooleanOptionalAction, default=True,
                    help="Regenerate convergence plots after each completed Darcy/coupling iteration.")
    ap.add_argument("--plot-convergence-script", default="plot_coupling_convergence.py",
                    help="Plotting script used by --plot-each-iteration. Relative paths are resolved next to coupler.py.")
    ap.add_argument("--plot-late-count", type=int, default=25,
                    help="Number of final runs shown in the auto-updated late-stage convergence plot.")
    ap.add_argument("--plot-title", default="",
                    help="Optional fixed title suffix for auto-updated convergence plots. Defaults to 'through run_i'.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    template = Path(args.template_combined).expanduser().resolve()
    if not template.is_file():
        die(f"Template combined.in not found: {template}")

    seed_run0 = Path(args.seed_run0).expanduser().resolve()
    if not seed_run0.exists():
        die(f"Seed run_0 does not exist: {seed_run0}")
    trial_dir: Optional[Path] = None
    if args.trial_dir:
        candidate_trial_dir = Path(args.trial_dir).expanduser().resolve()
        if candidate_trial_dir.exists():
            trial_dir = candidate_trial_dir
        elif not args.no_synthetic_vasculature:
            die(f"Trial directory does not exist: {candidate_trial_dir}")
    scaled_cfg = load_scaled_screening_config(trial_dir, args)
    if scaled_cfg is not None:
        print(
            f"[info] scaled-screening mode active: solving organoid_{scaled_cfg['reference_organoid']} "
            "with Darcy and regenerating scaled outputs for the remaining wells.",
            flush=True,
        )

    coupled_root = Path(args.coupled_root).expanduser().resolve()
    coupled_root.mkdir(parents=True, exist_ok=True)

    # Build run_0
    run0 = coupled_root / "run_0"
    if run0.exists() and args.overwrite:
        shutil.rmtree(run0)
    run0.mkdir(parents=True, exist_ok=True)
    combined0 = run0 / "combined.in"
    shutil.copy2(template, combined0)
    deck0 = load_json(combined0)
    apply_channel_ramp(
        deck0,
        args.channel_inlet_bc,
        Q0=float(args.channel_inlet_Q0),
        scale=0.0,
        outlet_name=args.channel_outlet_bc if args.use_outlet_pressure_ramp else None,
        outlet_P0=float(args.channel_outlet_P0),
    )
    zero_all_interface_bcs(deck0, args.n_organoids)
    save_json(combined0, deck0)
    ensure_organoid_dirs(run0, args.n_organoids)
    sync_plot_templates(seed_run0, run0, args.n_organoids)

    run([
        sys.executable, str(Path(args.run_and_split).expanduser().resolve()),
        "--exe", args.svzerodsolver,
        "--input", str(combined0),
        "--outdir", str(run0),
        "--output", "output.csv",
        "--organoid-root", str(run0),
        "--no-plot",
    ] + (["--debug"] if args.debug else []))

    copy_seed_geometry(
        seed_run0,
        run0,
        args.n_organoids,
        no_synthetic_vasculature=bool(args.no_synthetic_vasculature),
    )
    copy_branching_files(
        trial_dir,
        run0,
        args.n_organoids,
        allow_missing=bool(args.no_synthetic_vasculature),
    )
    if not args.no_synthetic_vasculature:
        update_1d_checkpoints(run0, args.n_organoids, np.array(args.coords_inlet), np.array(args.coords_outlet), args.dy_step)
        validate_darcy_geometry_inputs(run0, args.n_organoids, no_synthetic_vasculature=False)
    run_darcy_for_all(
        run0,
        args,
        scaled_cfg=scaled_cfg,
        replicate_reference_outputs=(scaled_cfg is not None),
    )

    # Coupling loop: ramp then iterate to convergence
    total_steps = int(args.n_ramp) + int(args.max_iter)
    start_run = max(1, int(args.start_run))
    run0 = coupled_root / "run_0"
    if start_run <= 1 and not run0.exists():
        print(f"[info] initializing {run0} from seed run_0: {seed_run0}", flush=True)
        copy_tree(seed_run0, run0, overwrite=False)
    for i in range(start_run, total_steps + 1):
        scale = min(i / float(max(args.n_ramp, 1)), 1.0) if i <= args.n_ramp else 1.0
        prev = coupled_root / f"run_{i-1}"
        cur = coupled_root / f"run_{i}"
        initialize_iteration_run(prev, cur, overwrite=bool(args.overwrite))

        combined_i = cur / "combined.in"
        base_deck = load_json(combined_i)
        iface_prev_by_organoid: dict[int, dict] = {}
        for k in range(1, args.n_organoids + 1):
            iface_prev = prev / f"organoid_{k}" / "out_darcy" / "interface_bc.json"
            if not iface_prev.exists():
                die(f"Missing interface file: {iface_prev}")
            iface_prev_by_organoid[k] = load_json(iface_prev)
        stubborn_terminal_response_map = build_stubborn_terminal_response_map(
            coupled_root,
            int(i),
            enabled=bool(args.stubborn_terminal_response_correction),
            min_rel_error=float(args.stubborn_terminal_min_rel_error),
            max_recent_improvement_rel=float(args.stubborn_terminal_max_recent_improvement),
            history_window=int(args.stubborn_terminal_history_window),
            gain=float(args.stubborn_terminal_extra_gain),
        )
        if stubborn_terminal_response_map:
            preview = sorted(stubborn_terminal_response_map.items())[:8]
            preview_text = ", ".join(
                f"org{key[0]}:{key[1]}:{key[2]} "
                f"err={value['error_rel']:.3e} improve={value['recent_improvement_rel']:.3e}"
                for key, value in preview
            )
            print(
                f"[stubborn-terminal] applying extra response gain to "
                f"{len(stubborn_terminal_response_map)} terminals: {preview_text}",
                flush=True,
            )

        def _build_candidate_deck(candidate_relax: float) -> dict:
            candidate = copy.deepcopy(base_deck)
            apply_channel_ramp(
                candidate,
                args.channel_inlet_bc,
                Q0=float(args.channel_inlet_Q0),
                scale=scale,
                outlet_name=args.channel_outlet_bc if args.use_outlet_pressure_ramp else None,
                outlet_P0=float(args.channel_outlet_P0),
            )
            for org_idx, iface_prev_data in iface_prev_by_organoid.items():
                apply_organoid_interface_from_json(
                    candidate,
                    org_idx,
                    iface_prev_data,
                    relax=float(candidate_relax),
                    support_flow_factor=float(args.terminal_support_flow_factor),
                    support_pressure_rel_tol=float(args.terminal_support_pressure_rel_tol),
                    support_mass_balance_tol=float(args.terminal_support_mass_balance_tol),
                    support_q_floor=float(args.terminal_support_q_floor),
                    allow_missing_terminal_bcs=bool(args.no_synthetic_vasculature),
                )
            return candidate

        def _build_resistance_deck(
            resistance_relaxation: float | None = None,
            pd_relaxation: float | None = None,
            continuity_resistance_decay: float | None = None,
            pressure_handoff_enabled: bool | None = None,
            pressure_handoff_rel_tol: float | None = None,
            terminal_response_correction_gain: float | None = None,
            terminal_response_correction_guarded_reasons: bool = False,
            print_active_side: bool = True,
        ) -> tuple[dict, list[dict[str, Any]]]:
            candidate = copy.deepcopy(base_deck)
            apply_channel_ramp(
                candidate,
                args.channel_inlet_bc,
                Q0=float(args.channel_inlet_Q0),
                scale=scale,
                outlet_name=args.channel_outlet_bc if args.use_outlet_pressure_ramp else None,
                outlet_P0=float(args.channel_outlet_P0),
            )
            active_terminal_side = "both"
            if str(args.terminal_side_update_mode) == "alternate":
                first_side = str(args.terminal_alternate_first_side)
                second_side = "venous" if first_side == "arterial" else "arterial"
                alternate_offset = int(i) - int(args.resistance_calibration_run) - 1
                active_terminal_side = first_side if alternate_offset % 2 == 0 else second_side
                if print_active_side:
                    print(
                        f"[terminal-resistance] updating {active_terminal_side} terminals; "
                        f"holding {second_side if active_terminal_side == first_side else first_side} terminals fixed.",
                        flush=True,
                    )
            rows: list[dict[str, Any]] = []
            for org_idx, iface_prev_data in iface_prev_by_organoid.items():
                rows.extend(
                    apply_organoid_resistance_from_json(
                        candidate,
                        org_idx,
                        iface_prev_data,
                        resistance_relaxation=float(
                            args.terminal_resistance_relaxation
                            if resistance_relaxation is None
                            else resistance_relaxation
                        ),
                        pd_relaxation=float(
                            args.terminal_pd_relaxation
                            if pd_relaxation is None
                            else pd_relaxation
                        ),
                        q_floor=float(args.terminal_resistance_q_floor),
                        venous_pd_floor=(
                            None
                            if args.terminal_venous_pd_floor is None
                            else float(args.terminal_venous_pd_floor)
                        ),
                        pd_mode=("0d" if int(i) == int(args.resistance_calibration_run) + 1 else "interface"),
                        suppression_pressure_tol=float(args.terminal_suppression_pressure_tol),
                        continuity_pressure_tol=float(args.terminal_continuity_pressure_tol),
                        continuity_resistance_decay=float(
                            args.terminal_continuity_resistance_decay
                            if continuity_resistance_decay is None
                            else continuity_resistance_decay
                        ),
                        pressure_alignment_jump_tol=float(args.terminal_pressure_alignment_jump_tol),
                        active_terminal_side=active_terminal_side,
                        pressure_handoff_enabled=bool(
                            args.terminal_pressure_handoff
                            if pressure_handoff_enabled is None
                            else pressure_handoff_enabled
                        ),
                        pressure_handoff_rel_tol=float(
                            args.terminal_pressure_handoff_rel_tol
                            if pressure_handoff_rel_tol is None
                            else pressure_handoff_rel_tol
                        ),
                        pressure_handoff_abs_tol=float(args.terminal_pressure_handoff_abs_tol),
                        inverse_flow_pd_relaxation_fraction=float(
                            args.terminal_inverse_flow_pd_relaxation_fraction
                        ),
                        inverse_flow_pd_relaxation_gamma=float(
                            args.terminal_inverse_flow_pd_relaxation_gamma
                        ),
                        inverse_flow_pd_relaxation_weight_cap=float(
                            args.terminal_inverse_flow_pd_relaxation_weight_cap
                        ),
                        terminal_response_correction_gain=float(
                            0.0
                            if terminal_response_correction_gain is None
                            else terminal_response_correction_gain
                        ),
                        terminal_response_correction_min_flow_weight=float(
                            args.terminal_response_correction_min_flow_weight
                        ),
                        terminal_response_correction_rel_error_tol=float(
                            args.terminal_response_correction_rel_error_tol
                        ),
                        terminal_response_correction_error_scale=float(
                            args.terminal_response_correction_error_scale
                        ),
                        terminal_response_correction_error_gamma=float(
                            args.terminal_response_correction_error_gamma
                        ),
                        terminal_response_correction_effective_gain_cap=float(
                            args.terminal_response_correction_effective_gain_cap
                        ),
                        terminal_response_correction_guarded_reasons=bool(
                            terminal_response_correction_guarded_reasons
                        ),
                        stubborn_terminal_response_map=stubborn_terminal_response_map,
                        allow_missing_terminal_bcs=bool(args.no_synthetic_vasculature),
                    )
                )
            for row in rows:
                row["run"] = int(i)
            return candidate, rows

        ensure_organoid_dirs(cur, args.n_organoids)
        sync_plot_templates(seed_run0, cur, args.n_organoids)

        deck: dict
        inner_tries = max(1, int(args.inner_0d_max_tries))
        use_resistance_bc = (
            str(args.terminal_bc_mode) == "resistance"
            and not bool(args.no_synthetic_vasculature)
            and int(i) > int(args.resistance_calibration_run)
        )
        if use_resistance_bc:
            first_resistance_iteration = int(i) == int(args.resistance_calibration_run) + 1
            if bool(args.inner_resistance_0d_search) and not first_resistance_iteration:
                trial_root = cur / "_inner_resistance_0d_trials"
                if trial_root.exists():
                    shutil.rmtree(trial_root)
                trial_root.mkdir(parents=True, exist_ok=True)
                decay_candidates = _parse_float_list(
                    args.inner_resistance_decay_candidates,
                    [float(args.terminal_continuity_resistance_decay)],
                )
                relaxation_multipliers = _parse_float_list(
                    args.inner_resistance_relaxation_multipliers,
                    [1.0],
                )
                pd_relaxation_candidates = _parse_float_list(
                    args.inner_pd_relaxation_candidates,
                    [float(args.terminal_pd_relaxation)],
                )
                response_correction_gains = _parse_float_list(
                    args.inner_terminal_response_correction_gains,
                    [0.0],
                )
                search_profile = str(args.inner_resistance_search_profile)
                if search_profile == "compact":
                    # Empirically, both high- and low-permeability histories
                    # almost always selected the most aggressive R relaxation.
                    # Keeping all Pd candidates preserves low-permeability
                    # flexibility while cutting the grid by roughly 4x.
                    relaxation_multipliers = [4.0]
                    if 0.9 not in pd_relaxation_candidates:
                        pd_relaxation_candidates.append(0.9)
                    response_correction_gains = sorted(
                        {0.0, *[float(v) for v in response_correction_gains if float(v) > 0.0]}
                    )
                elif search_profile == "full":
                    relaxation_multipliers = [0.5, 1.0, 2.0, 4.0]
                    if 0.9 not in pd_relaxation_candidates:
                        pd_relaxation_candidates.append(0.9)
                    decay_candidates = [1.0, 0.85, 0.7]
                    response_correction_gains = [0.0, 1.0]
                handoff_tol_candidates = (
                    _parse_float_list(
                        args.inner_pressure_handoff_tol_candidates,
                        [float(args.terminal_pressure_handoff_rel_tol)],
                    )
                    if bool(args.terminal_pressure_handoff)
                    else [float(args.terminal_pressure_handoff_rel_tol)]
                )
                candidate_specs: list[dict[str, Any]] = []
                seen_specs: set[tuple[float, float, float, float, bool, float, float, bool]] = set()
                for relax_mult in relaxation_multipliers:
                    for pd_candidate in pd_relaxation_candidates:
                        for decay_candidate in decay_candidates:
                            for handoff_tol in handoff_tol_candidates:
                                for response_gain in response_correction_gains:
                                    spec = {
                                        "resistance_relaxation_multiplier": float(relax_mult),
                                        "resistance_relaxation": min(
                                            max(float(args.terminal_resistance_relaxation) * float(relax_mult), 0.0),
                                            1.0,
                                        ),
                                        "pd_relaxation": min(max(float(pd_candidate), 0.0), 1.0),
                                        "continuity_resistance_decay": min(max(float(decay_candidate), 0.0), 1.0),
                                        "pressure_handoff_enabled": bool(args.terminal_pressure_handoff),
                                        "pressure_handoff_rel_tol": max(float(handoff_tol), 0.0),
                                        "terminal_response_correction_gain": max(float(response_gain), 0.0),
                                        "terminal_response_correction_guarded_reasons": bool(
                                            args.inner_response_correction_guarded_reasons
                                            and float(response_gain) > 0.0
                                        ),
                                    }
                                    key = (
                                        round(float(spec["resistance_relaxation_multiplier"]), 12),
                                        round(float(spec["resistance_relaxation"]), 12),
                                        round(float(spec["pd_relaxation"]), 12),
                                        round(float(spec["continuity_resistance_decay"]), 12),
                                        bool(spec["pressure_handoff_enabled"]),
                                        round(float(spec["pressure_handoff_rel_tol"]), 12),
                                        round(float(spec["terminal_response_correction_gain"]), 12),
                                        bool(spec["terminal_response_correction_guarded_reasons"]),
                                    )
                                    if key not in seen_specs:
                                        candidate_specs.append(spec)
                                        seen_specs.add(key)
                if bool(args.terminal_pressure_handoff):
                    no_handoff_specs: list[dict[str, Any]] = []
                    for relax_mult in relaxation_multipliers:
                        for pd_candidate in pd_relaxation_candidates:
                            for response_gain in response_correction_gains:
                                no_handoff = {
                                    "resistance_relaxation_multiplier": float(relax_mult),
                                    "resistance_relaxation": min(
                                        max(float(args.terminal_resistance_relaxation) * float(relax_mult), 0.0),
                                        1.0,
                                    ),
                                    "pd_relaxation": min(max(float(pd_candidate), 0.0), 1.0),
                                    "continuity_resistance_decay": float(args.terminal_continuity_resistance_decay),
                                    "pressure_handoff_enabled": False,
                                    "pressure_handoff_rel_tol": float(args.terminal_pressure_handoff_rel_tol),
                                    "terminal_response_correction_gain": max(float(response_gain), 0.0),
                                    "terminal_response_correction_guarded_reasons": bool(
                                        args.inner_response_correction_guarded_reasons
                                        and float(response_gain) > 0.0
                                    ),
                                }
                                key = (
                                    round(float(no_handoff["resistance_relaxation_multiplier"]), 12),
                                    round(float(no_handoff["resistance_relaxation"]), 12),
                                    round(float(no_handoff["pd_relaxation"]), 12),
                                    round(float(no_handoff["continuity_resistance_decay"]), 12),
                                    False,
                                    round(float(no_handoff["pressure_handoff_rel_tol"]), 12),
                                    round(float(no_handoff["terminal_response_correction_gain"]), 12),
                                    bool(no_handoff["terminal_response_correction_guarded_reasons"]),
                                )
                                if key not in seen_specs:
                                    no_handoff_specs.append(no_handoff)
                                    seen_specs.add(key)
                    if no_handoff_specs:
                        candidate_specs = no_handoff_specs + candidate_specs

                grid_config = {
                    "run": int(i),
                    "inner_resistance_search_profile": str(args.inner_resistance_search_profile),
                    "base_terminal_resistance_relaxation": float(args.terminal_resistance_relaxation),
                    "terminal_pressure_handoff_abs_tol": float(args.terminal_pressure_handoff_abs_tol),
                    "raw_inner_resistance_relaxation_multipliers": str(args.inner_resistance_relaxation_multipliers),
                    "raw_inner_pd_relaxation_candidates": str(args.inner_pd_relaxation_candidates),
                    "raw_inner_terminal_response_correction_gains": str(
                        args.inner_terminal_response_correction_gains
                    ),
                    "raw_inner_resistance_decay_candidates": str(args.inner_resistance_decay_candidates),
                    "raw_inner_pressure_handoff_tol_candidates": str(args.inner_pressure_handoff_tol_candidates),
                    "parsed_resistance_relaxation_multipliers": [float(v) for v in relaxation_multipliers],
                    "parsed_pd_relaxation_candidates": [float(v) for v in pd_relaxation_candidates],
                    "parsed_terminal_response_correction_gains": [float(v) for v in response_correction_gains],
                    "parsed_continuity_resistance_decay_candidates": [float(v) for v in decay_candidates],
                    "parsed_pressure_handoff_tol_candidates": [float(v) for v in handoff_tol_candidates],
                    "flow_guard_enabled": bool(args.inner_resistance_flow_guard),
                    "flow_guard_tol": float(args.inner_flow_guard_tol),
                    "flow_guard_weight": float(args.inner_flow_guard_weight),
                    "pressure_weight": float(args.inner_0d_pressure_weight),
                    "inverse_flow_pressure_fraction": float(args.inner_inverse_flow_pressure_fraction),
                    "inverse_flow_pressure_gamma": float(args.inner_inverse_flow_pressure_gamma),
                    "inverse_flow_pressure_weight_cap": float(args.inner_inverse_flow_pressure_weight_cap),
                    "terminal_inverse_flow_pd_relaxation_fraction": float(
                        args.terminal_inverse_flow_pd_relaxation_fraction
                    ),
                    "terminal_inverse_flow_pd_relaxation_gamma": float(
                        args.terminal_inverse_flow_pd_relaxation_gamma
                    ),
                    "terminal_inverse_flow_pd_relaxation_weight_cap": float(
                        args.terminal_inverse_flow_pd_relaxation_weight_cap
                    ),
                    "terminal_response_correction_min_flow_weight": float(
                        args.terminal_response_correction_min_flow_weight
                    ),
                    "terminal_response_correction_rel_error_tol": float(
                        args.terminal_response_correction_rel_error_tol
                    ),
                    "terminal_response_correction_error_scale": float(
                        args.terminal_response_correction_error_scale
                    ),
                    "terminal_response_correction_error_gamma": float(
                        args.terminal_response_correction_error_gamma
                    ),
                    "terminal_response_correction_effective_gain_cap": float(
                        args.terminal_response_correction_effective_gain_cap
                    ),
                    "inner_response_correction_guarded_reasons": bool(
                        args.inner_response_correction_guarded_reasons
                    ),
                    "stubborn_terminal_response_correction": bool(
                        args.stubborn_terminal_response_correction
                    ),
                    "stubborn_terminal_min_rel_error": float(args.stubborn_terminal_min_rel_error),
                    "stubborn_terminal_max_recent_improvement": float(
                        args.stubborn_terminal_max_recent_improvement
                    ),
                    "stubborn_terminal_history_window": int(args.stubborn_terminal_history_window),
                    "stubborn_terminal_extra_gain": float(args.stubborn_terminal_extra_gain),
                    "stubborn_terminal_count": int(len(stubborn_terminal_response_map)),
                    "stubborn_terminals": [
                        {
                            "organoid": int(key[0]),
                            "side": str(key[1]),
                            "branch_id": int(key[2]),
                            **value,
                        }
                        for key, value in sorted(stubborn_terminal_response_map.items())
                    ],
                    "implied_alignment_weight": float(args.inner_implied_alignment_weight),
                    "jump_weight": float(args.inner_jump_weight),
                    "max_pressure_error_weight": float(args.inner_max_pressure_error_weight),
                    "stubborn_max_error_weight": float(args.inner_stubborn_max_error_weight),
                    "flow_guard_reject": bool(args.inner_flow_guard_reject),
                    "candidate_count": int(len(candidate_specs)),
                    "candidate_specs": candidate_specs,
                }
                write_inner_resistance_candidate_grid(cur, candidate_specs, grid_config)
                print(
                    "[inner-resistance-0d] candidate grid: "
                    f"{len(candidate_specs)} candidates, "
                    f"relaxation_multipliers={grid_config['parsed_resistance_relaxation_multipliers']}, "
                    f"pd_candidates={grid_config['parsed_pd_relaxation_candidates']}, "
                    f"response_gains={grid_config['parsed_terminal_response_correction_gains']}, "
                    f"decay_candidates={grid_config['parsed_continuity_resistance_decay_candidates']}, "
                    f"handoff_tol_candidates={grid_config['parsed_pressure_handoff_tol_candidates']}",
                    flush=True,
                )

                trial_summaries: list[dict[str, Any]] = []
                best: tuple[float, dict, list[dict[str, Any]], dict[str, Any]] | None = None
                for candidate_idx, spec in enumerate(candidate_specs):
                    candidate_trial_dir = trial_root / f"candidate_{candidate_idx:02d}"
                    candidate_trial_dir.mkdir(parents=True, exist_ok=True)
                    ensure_organoid_dirs(candidate_trial_dir, int(args.n_organoids))
                    trial_combined = candidate_trial_dir / "combined.in"
                    candidate_deck, candidate_rows = _build_resistance_deck(
                        resistance_relaxation=float(spec["resistance_relaxation"]),
                        pd_relaxation=float(spec["pd_relaxation"]),
                        continuity_resistance_decay=float(spec["continuity_resistance_decay"]),
                        pressure_handoff_enabled=bool(spec["pressure_handoff_enabled"]),
                        pressure_handoff_rel_tol=float(spec["pressure_handoff_rel_tol"]),
                        terminal_response_correction_gain=float(
                            spec.get("terminal_response_correction_gain", 0.0)
                        ),
                        terminal_response_correction_guarded_reasons=bool(
                            spec.get("terminal_response_correction_guarded_reasons", False)
                        ),
                        print_active_side=(candidate_idx == 0),
                    )
                    save_json(trial_combined, candidate_deck)
                    if try_run_0d_split(args, trial_combined, candidate_trial_dir):
                        score = score_resistance_0d_trial(
                            candidate_trial_dir,
                            candidate_rows,
                            int(args.n_organoids),
                            flow_guard_enabled=bool(args.inner_resistance_flow_guard),
                            flow_guard_tol=float(args.inner_flow_guard_tol),
                            flow_guard_weight=float(args.inner_flow_guard_weight),
                            pressure_weight=float(args.inner_0d_pressure_weight),
                            inverse_flow_pressure_fraction=float(args.inner_inverse_flow_pressure_fraction),
                            inverse_flow_pressure_gamma=float(args.inner_inverse_flow_pressure_gamma),
                            inverse_flow_pressure_weight_cap=float(args.inner_inverse_flow_pressure_weight_cap),
                            implied_alignment_weight=float(args.inner_implied_alignment_weight),
                            jump_weight=float(args.inner_jump_weight),
                            max_pressure_error_weight=float(args.inner_max_pressure_error_weight),
                            stubborn_max_error_weight=float(args.inner_stubborn_max_error_weight),
                            flow_guard_reject=bool(args.inner_flow_guard_reject),
                            flow_guard_drive_tol=float(args.terminal_pressure_alignment_jump_tol),
                            flow_guard_q_floor=float(args.terminal_resistance_q_floor),
                        )
                    else:
                        score = {
                            "score": float("inf"),
                            "pressure_score": float("inf"),
                            "implied_interface_score": float("inf"),
                            "implied_median_percent": float("inf"),
                            "implied_p90_percent": float("inf"),
                            "implied_max_percent": float("inf"),
                            "arterial_implied_score": float("inf"),
                            "venous_implied_score": float("inf"),
                            "arterial_implied_median_percent": float("inf"),
                            "venous_implied_median_percent": float("inf"),
                            "arterial_implied_p90_percent": float("inf"),
                            "venous_implied_p90_percent": float("inf"),
                            "jump_score": float("inf"),
                            "flow_guard_score": float("inf"),
                            "median_error_percent": float("inf"),
                            "p90_error_percent": float("inf"),
                            "p95_error_percent": float("inf"),
                            "max_error_percent": float("inf"),
                            "outlier_error_score": float("inf"),
                            "active_median_error_percent": float("inf"),
                            "stubborn_max_error_percent": float("inf"),
                            "stubborn_p90_error_percent": float("inf"),
                            "stubborn_error_score": float("inf"),
                            "jump_median_percent": float("inf"),
                            "jump_p90_percent": float("inf"),
                            "jump_max_percent": float("inf"),
                            "active_jump_median_percent": float("inf"),
                            "flow_guard_count": 0.0,
                            "flow_guard_increase_count": 0.0,
                            "flow_guard_increase_fraction": 0.0,
                            "flow_guard_increase_median_percent": float("inf"),
                            "flow_guard_increase_p90_percent": float("inf"),
                            "flow_guard_increase_max_percent": float("inf"),
                            "flow_guard_reduction_median_percent": 0.0,
                            "pressure_bc_count": 0.0,
                            "resistance_bc_count": 0.0,
                        }
                    summary_row = {
                        "run": int(i),
                        "candidate": int(candidate_idx),
                        "selected": False,
                        **spec,
                        **score,
                    }
                    trial_summaries.append(summary_row)
                    score_value = float(score.get("score", float("inf")))
                    if np.isfinite(score_value) and (best is None or score_value < best[0]):
                        best = (score_value, candidate_deck, candidate_rows, summary_row)

                if best is None:
                    deck, resistance_rows = _build_resistance_deck()
                else:
                    deck, resistance_rows = best[1], best[2]
                    best[3]["selected"] = True
                    print(
                        "[inner-resistance-0d] selected candidate "
                        f"{best[3]['candidate']} with score={best[0]:.3e}, "
                        f"median={best[3]['median_error_percent']:.3e}%, "
                        f"p90={best[3]['p90_error_percent']:.3e}%, "
                        f"jump_median={best[3].get('jump_median_percent', float('nan')):.3e}%, "
                        f"flow_guard={best[3].get('flow_guard_score', float('nan')):.3e}",
                        flush=True,
                    )
                write_inner_resistance_search_summary(cur, trial_summaries)
                if not bool(args.keep_inner_resistance_trials):
                    shutil.rmtree(trial_root, ignore_errors=True)
            else:
                if bool(args.inner_resistance_0d_search) and first_resistance_iteration:
                    print(
                        "[inner-resistance-0d] skipping candidate search on the first "
                        "resistance iteration; calibration has no previous R/Pd state, "
                        "so all candidates are identical.",
                        flush=True,
                    )
                    save_json(
                        cur / "inner_resistance_0d_search_config.json",
                        {
                            "run": int(i),
                            "skipped": True,
                            "skip_reason": "first_resistance_iteration",
                            "inner_resistance_search_profile": str(args.inner_resistance_search_profile),
                            "base_terminal_resistance_relaxation": float(args.terminal_resistance_relaxation),
                            "base_terminal_pd_relaxation": float(args.terminal_pd_relaxation),
                            "terminal_pressure_handoff_abs_tol": float(args.terminal_pressure_handoff_abs_tol),
                            "raw_inner_resistance_relaxation_multipliers": str(
                                args.inner_resistance_relaxation_multipliers
                            ),
                            "raw_inner_pd_relaxation_candidates": str(args.inner_pd_relaxation_candidates),
                            "raw_inner_terminal_response_correction_gains": str(
                                args.inner_terminal_response_correction_gains
                            ),
                            "raw_inner_resistance_decay_candidates": str(args.inner_resistance_decay_candidates),
                            "raw_inner_pressure_handoff_tol_candidates": str(
                                args.inner_pressure_handoff_tol_candidates
                            ),
                            "flow_guard_enabled": bool(args.inner_resistance_flow_guard),
                            "flow_guard_tol": float(args.inner_flow_guard_tol),
                            "flow_guard_weight": float(args.inner_flow_guard_weight),
                            "pressure_weight": float(args.inner_0d_pressure_weight),
                            "inverse_flow_pressure_fraction": float(args.inner_inverse_flow_pressure_fraction),
                            "inverse_flow_pressure_gamma": float(args.inner_inverse_flow_pressure_gamma),
                            "inverse_flow_pressure_weight_cap": float(args.inner_inverse_flow_pressure_weight_cap),
                            "terminal_inverse_flow_pd_relaxation_fraction": float(
                                args.terminal_inverse_flow_pd_relaxation_fraction
                            ),
                            "terminal_inverse_flow_pd_relaxation_gamma": float(
                                args.terminal_inverse_flow_pd_relaxation_gamma
                            ),
                            "terminal_inverse_flow_pd_relaxation_weight_cap": float(
                                args.terminal_inverse_flow_pd_relaxation_weight_cap
                            ),
                            "terminal_response_correction_min_flow_weight": float(
                                args.terminal_response_correction_min_flow_weight
                            ),
                            "terminal_response_correction_rel_error_tol": float(
                                args.terminal_response_correction_rel_error_tol
                            ),
                            "terminal_response_correction_error_scale": float(
                                args.terminal_response_correction_error_scale
                            ),
                            "terminal_response_correction_error_gamma": float(
                                args.terminal_response_correction_error_gamma
                            ),
                            "terminal_response_correction_effective_gain_cap": float(
                                args.terminal_response_correction_effective_gain_cap
                            ),
                            "inner_response_correction_guarded_reasons": bool(
                                args.inner_response_correction_guarded_reasons
                            ),
                            "stubborn_terminal_response_correction": bool(
                                args.stubborn_terminal_response_correction
                            ),
                            "stubborn_terminal_min_rel_error": float(args.stubborn_terminal_min_rel_error),
                            "stubborn_terminal_max_recent_improvement": float(
                                args.stubborn_terminal_max_recent_improvement
                            ),
                            "stubborn_terminal_history_window": int(args.stubborn_terminal_history_window),
                            "stubborn_terminal_extra_gain": float(args.stubborn_terminal_extra_gain),
                            "stubborn_terminal_count": int(len(stubborn_terminal_response_map)),
                            "implied_alignment_weight": float(args.inner_implied_alignment_weight),
                            "jump_weight": float(args.inner_jump_weight),
                            "max_pressure_error_weight": float(args.inner_max_pressure_error_weight),
                            "stubborn_max_error_weight": float(args.inner_stubborn_max_error_weight),
                            "flow_guard_reject": bool(args.inner_flow_guard_reject),
                            "candidate_count": 0,
                        },
                    )
                deck, resistance_rows = _build_resistance_deck()
            save_json(combined_i, deck)
            write_terminal_resistance_summary(
                cur,
                resistance_rows,
                write_cumulative=bool(args.terminal_pd_convergence_csv),
            )
            run_0d_split(args, combined_i, cur)
        elif bool(args.no_synthetic_vasculature) or inner_tries <= 1:
            deck = _build_candidate_deck(float(args.relaxation))
            save_json(combined_i, deck)
            run_0d_split(args, combined_i, cur)
        else:
            target_basis_deck = copy.deepcopy(base_deck)
            apply_channel_ramp(
                target_basis_deck,
                args.channel_inlet_bc,
                Q0=float(args.channel_inlet_Q0),
                scale=scale,
                outlet_name=args.channel_outlet_bc if args.use_outlet_pressure_ramp else None,
                outlet_P0=float(args.channel_outlet_P0),
            )
            reduction_targets = build_pressure_reduction_targets(
                target_basis_deck,
                iface_prev_by_organoid,
                n_organoids=int(args.n_organoids),
                support_flow_factor=float(args.terminal_support_flow_factor),
                support_pressure_rel_tol=float(args.terminal_support_pressure_rel_tol),
                support_mass_balance_tol=float(args.terminal_support_mass_balance_tol),
                support_q_floor=float(args.terminal_support_q_floor),
            )
            print(
                f"[inner-0d] parent-recentering targets: {len(reduction_targets)} terminals",
                flush=True,
            )
            deck = _build_candidate_deck(float(args.relaxation))
            for attempt in range(inner_tries):
                save_json(combined_i, deck)
                run_0d_split(args, combined_i, cur)
                target_scores = pressure_reduction_target_flow_scores(
                    run_dir=cur,
                    targets=reduction_targets,
                )
                print(
                    "[inner-0d] attempt "
                    f"{attempt + 1}/{inner_tries}: "
                    f"target-flow-error total={target_scores['total']:.3e}, "
                    f"inlet={target_scores['inlet']:.3e}, "
                    f"outlet={target_scores['outlet']:.3e}",
                    flush=True,
                )
                if target_scores["total"] <= float(args.inner_0d_accept_tol):
                    print(
                        "[inner-0d] reduced 0D terminal flows reached the requested target tolerance.",
                        flush=True,
                    )
                    break
                if attempt + 1 >= inner_tries or not reduction_targets:
                    break
                changed = recenter_terminal_pressures_on_latest_0d_parents(
                    deck,
                    cur,
                    reduction_targets,
                    n_organoids=int(args.n_organoids),
                )
                print(
                    f"[inner-0d] recentered {changed} terminal pressures around latest 0D parent pressures.",
                    flush=True,
                )
                if changed == 0:
                    print(
                        "[inner-0d] no pressure recentering was possible; continuing to Darcy.",
                        flush=True,
                    )
                    break

        copy_seed_geometry(
            seed_run0,
            cur,
            args.n_organoids,
            no_synthetic_vasculature=bool(args.no_synthetic_vasculature),
        )
        copy_branching_files(
            trial_dir,
            cur,
            args.n_organoids,
            allow_missing=bool(args.no_synthetic_vasculature),
        )
        if not args.no_synthetic_vasculature:
            update_1d_checkpoints(cur, args.n_organoids, np.array(args.coords_inlet), np.array(args.coords_outlet), args.dy_step)
            validate_darcy_geometry_inputs(cur, args.n_organoids, no_synthetic_vasculature=False)
        run_darcy_for_all(cur, args, scaled_cfg=scaled_cfg)
        update_convergence_plots(args, coupled_root, i)

        # convergence after ramp
        if i > args.n_ramp:
            all_ok = True
            print(f"\n[check] coupling iteration {i - args.n_ramp}/{args.max_iter}")
            for k in range(1, args.n_organoids + 1):
                iface_prev = prev / f"organoid_{k}" / "out_darcy" / "interface_bc.json"
                iface_cur = cur / f"organoid_{k}" / "out_darcy" / "interface_bc.json"
                ok, rq, rp = convergence_for_organoid(
                    iface_cur,
                    tol_q=args.tol_q,
                    tol_p=args.tol_p,
                    prev_iface_path=(
                        iface_prev
                        if (args.no_synthetic_vasculature or str(args.terminal_bc_mode) == "resistance")
                        else None
                    ),
                    current_deck=(deck if args.no_synthetic_vasculature else None),
                    organoid_idx=(k if args.no_synthetic_vasculature else None),
                    pressure_compare_mode="target",
                )
                all_ok = all_ok and ok
                print(f"  organoid_{k}: rel_p={rp:.3e}, converged={ok}")
            if all_ok:
                print(f"\n[done] converged at run_{i}.")
                return

    print(f"\n[done] reached max iterations ({args.max_iter}) after ramp without full convergence.")


if __name__ == "__main__":
    main()
