#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from mpi4py import MPI

from coupler import (
    build_mpi_command,
    die,
    run,
    copy_tree,
    load_json,
    save_json,
    ensure_flow_bc,
    ensure_pressure_bc,
    find_bc,
    find_bcs_with_prefix,
    ensure_organoid_dirs,
    sync_plot_templates,
    update_1d_checkpoints,
    resolve_perm_region_for_organoid,
    read_seed_coords_from_branching,
    shifted_seed_fallback,
)


def read_waveform_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(path, dtype=float)
    except Exception as exc:
        die(f"Failed to read waveform file {path}: {exc}")

    if data.ndim == 1:
        data = np.asarray(data, dtype=float).reshape(1, -1)
    if data.shape[1] < 2:
        die(f"Waveform file must contain at least two columns (time, value): {path}")

    times = np.asarray(data[:, 0], dtype=float)
    values = np.asarray(data[:, 1], dtype=float)
    if times.size == 0:
        die(f"Waveform file is empty: {path}")
    if np.any(np.diff(times) < 0.0):
        die(f"Waveform times must be nondecreasing: {path}")
    return times, values


def merge_time_grids(*time_arrays: np.ndarray) -> np.ndarray:
    pieces = [np.asarray(arr, dtype=float).ravel() for arr in time_arrays if np.asarray(arr).size > 0]
    if not pieces:
        return np.zeros(0, dtype=float)
    merged = np.unique(np.concatenate(pieces))
    return np.asarray(merged, dtype=float)


def normalize_waveform_period(times: np.ndarray, cycle_period: float) -> np.ndarray:
    times_arr = np.asarray(times, dtype=float).ravel().copy()
    if times_arr.size == 0:
        return times_arr

    times_arr -= float(times_arr[0])
    t_end = float(times_arr[-1])
    period = float(cycle_period)
    if t_end <= 0.0:
        return np.linspace(0.0, period, times_arr.size, dtype=float)
    if np.isclose(t_end, period, rtol=1e-9, atol=1e-12):
        times_arr[-1] = period
        return times_arr
    return times_arr * (period / t_end)


def cycle_average(times: np.ndarray, values: np.ndarray) -> float:
    t = np.asarray(times, dtype=float).ravel()
    v = np.asarray(values, dtype=float).ravel()
    if t.size == 0 or v.size == 0:
        return 0.0
    if t.size == 1 or np.isclose(t[-1], t[0]):
        return float(v[0])
    return float(np.trapz(v, t) / (t[-1] - t[0]))


def bc_scalar_value(bc: Optional[dict], key: str) -> Optional[float]:
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


def make_pulse_ramp_alphas(n_stages: int) -> List[float]:
    n = max(int(n_stages), 1)
    if n == 1:
        return [1.0]
    return [float(x) for x in np.linspace(0.0, 1.0, n)]


def waveform_baseline_value(
    mode: str,
    times: np.ndarray,
    values: np.ndarray,
    steady_value: Optional[float],
) -> float:
    choice = str(mode).strip().lower()
    if choice == "steady" and steady_value is not None:
        return float(steady_value)
    if choice == "first":
        arr = np.asarray(values, dtype=float).ravel()
        return float(arr[0]) if arr.size else 0.0
    return cycle_average(times, values)


def blend_waveform(values: np.ndarray, baseline: float, alpha: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return float(baseline) + float(alpha) * (arr - float(baseline))


def interp_series(times_src: np.ndarray, values_src: np.ndarray, times_dst: np.ndarray) -> np.ndarray:
    times_src = np.asarray(times_src, dtype=float).ravel()
    values_src = np.asarray(values_src, dtype=float)
    times_dst = np.asarray(times_dst, dtype=float).ravel()

    if times_dst.size == 0:
        if values_src.ndim == 2:
            return np.zeros((0, values_src.shape[1]), dtype=float)
        return np.zeros(0, dtype=float)

    if values_src.ndim == 1:
        if values_src.size == 0:
            return np.zeros(times_dst.shape[0], dtype=float)
        if values_src.size == 1 or times_src.size == 1:
            return np.full(times_dst.shape[0], float(values_src.reshape(-1)[0]), dtype=float)
        return np.interp(times_dst, times_src, values_src)

    if values_src.ndim != 2:
        raise ValueError(f"Expected 1D or 2D series, got shape {values_src.shape}")

    if values_src.shape[1] == 0:
        return np.zeros((times_dst.shape[0], 0), dtype=float)
    cols = [interp_series(times_src, values_src[:, j], times_dst) for j in range(values_src.shape[1])]
    return np.column_stack(cols)


def repeat_rows(values: np.ndarray, n_times: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(1, -1)
    if arr.shape[1] == 0:
        return np.zeros((n_times, 0), dtype=float)
    return np.repeat(arr, n_times, axis=0)


def repeat_scalar(value: float, n_times: int) -> np.ndarray:
    return np.full(n_times, float(value), dtype=float)


def set_bc_series(bc: dict, key: str, times: np.ndarray, values: np.ndarray) -> None:
    vals = bc.setdefault("bc_values", {})
    vals["t"] = [float(t) for t in np.asarray(times, dtype=float).ravel()]
    vals[key] = [float(v) for v in np.asarray(values, dtype=float).ravel()]


def update_simulation_parameters(deck: dict, n_cycles: int, n_time_pts_per_cycle: int) -> None:
    sim = deck.setdefault("simulation_parameters", {})
    sim["number_of_cardiac_cycles"] = int(n_cycles)
    sim["number_of_time_pts_per_cardiac_cycle"] = int(max(n_time_pts_per_cycle, 2))


def apply_channel_waveforms(
    deck: dict,
    inflow_name: str,
    inflow_t: np.ndarray,
    inflow_q: np.ndarray,
    outlet_name: str,
    outlet_t: np.ndarray,
    outlet_p: np.ndarray,
) -> None:
    inflow = find_bc(deck, inflow_name)
    if inflow is None:
        die(f"Missing channel inlet bc_name='{inflow_name}' in combined deck")
    ensure_flow_bc(inflow)
    set_bc_series(inflow, "Q", inflow_t, inflow_q)

    outlet = find_bc(deck, outlet_name)
    if outlet is None:
        die(f"Missing channel outlet bc_name='{outlet_name}' in combined deck")
    ensure_pressure_bc(outlet)
    set_bc_series(outlet, "P", outlet_t, outlet_p)


def detect_n_organoids(steady_run: Path) -> int:
    indices: List[int] = []
    for child in steady_run.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if not name.startswith("organoid_"):
            continue
        try:
            indices.append(int(name.split("_", 1)[1]))
        except Exception:
            continue
    if not indices:
        die(f"Could not detect organoid_* folders in steady run: {steady_run}")
    return max(indices)


def remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def clean_iteration_outputs(iter_dir: Path, n_organoids: int) -> None:
    remove_path(iter_dir / "output.csv")
    remove_path(iter_dir / "split")

    for k in range(1, n_organoids + 1):
        org = iter_dir / f"organoid_{k}"
        remove_path(org / "out_darcy")
        remove_path(org / "out_darcy_transient")
        for side in ("inlet", "outlet"):
            side_dir = org / "0D_Input_Files" / side
            remove_path(side_dir / "output.csv")
            for child in side_dir.iterdir() if side_dir.exists() else []:
                if child.is_dir() and child.name.startswith("timeseries"):
                    remove_path(child)


def prepare_iteration_dir(steady_run: Path, iter_dir: Path, n_organoids: int) -> None:
    copy_tree(steady_run, iter_dir, overwrite=True)
    clean_iteration_outputs(iter_dir, n_organoids)
    ensure_organoid_dirs(iter_dir, n_organoids)
    sync_plot_templates(steady_run, iter_dir, n_organoids)


def load_steady_interface_state(
    steady_run: Path,
    n_organoids: int,
    coupling_times: np.ndarray,
) -> Dict[int, dict]:
    state: Dict[int, dict] = {}
    nt = int(coupling_times.size)
    for k in range(1, n_organoids + 1):
        iface_path = steady_run / f"organoid_{k}" / "out_darcy" / "interface_bc.json"
        if not iface_path.exists():
            die(f"Missing steady interface file for organoid_{k}: {iface_path}")
        iface = load_json(iface_path)
        state[k] = {
            "p_inlet_bc": repeat_rows(np.asarray(iface.get("p_inlet_nodes", []), dtype=float), nt),
            "p_outlet_bc": repeat_rows(np.asarray(iface.get("p_outlet_nodes", []), dtype=float), nt),
            "q_artery_leak_bc": repeat_scalar(-float(iface.get("q_artery_leak", 0.0)), nt),
            "q_venous_leak_bc": repeat_scalar(-float(iface.get("q_venous_leak", 0.0)), nt),
            "skip_1d": bool(iface.get("skip_1d", False)),
        }
    return state


def apply_organoid_interface_state(
    deck: dict,
    organoid_idx: int,
    state: dict,
    coupling_times: np.ndarray,
    allow_missing_terminal_bcs: bool = False,
) -> None:
    inlet_bcs = find_bcs_with_prefix(deck, f"organoid{organoid_idx}_inlet_OUT")
    outlet_bcs = find_bcs_with_prefix(deck, f"organoid{organoid_idx}_outlet_OUT")
    leak_art = find_bcs_with_prefix(deck, f"LEAK_ART_{organoid_idx}")
    leak_ven = find_bcs_with_prefix(deck, f"LEAK_VEN_{organoid_idx}")

    p_in = np.asarray(state["p_inlet_bc"], dtype=float)
    p_out = np.asarray(state["p_outlet_bc"], dtype=float)

    if (p_in.shape[1] > 0 or p_out.shape[1] > 0) and not (allow_missing_terminal_bcs or bool(state.get("skip_1d", False))):
        if len(inlet_bcs) == 0 or len(outlet_bcs) == 0:
            die(f"Missing organoid terminal pressure BC groups for organoid_{organoid_idx}")

    if len(inlet_bcs) < p_in.shape[1]:
        die(f"Organoid {organoid_idx} has {p_in.shape[1]} inlet waveforms but only {len(inlet_bcs)} inlet BCs")
    if len(outlet_bcs) < p_out.shape[1]:
        die(f"Organoid {organoid_idx} has {p_out.shape[1]} outlet waveforms but only {len(outlet_bcs)} outlet BCs")

    for j in range(p_in.shape[1]):
        ensure_pressure_bc(inlet_bcs[j])
        set_bc_series(inlet_bcs[j], "P", coupling_times, p_in[:, j])

    for j in range(p_out.shape[1]):
        ensure_pressure_bc(outlet_bcs[j])
        set_bc_series(outlet_bcs[j], "P", coupling_times, p_out[:, j])

    if leak_art:
        ensure_flow_bc(leak_art[0])
        set_bc_series(leak_art[0], "Q", coupling_times, np.asarray(state["q_artery_leak_bc"], dtype=float))
    if leak_ven:
        ensure_flow_bc(leak_ven[0])
        set_bc_series(leak_ven[0], "Q", coupling_times, np.asarray(state["q_venous_leak_bc"], dtype=float))


def read_unique_output_times(output_csv: Path) -> np.ndarray:
    if not output_csv.exists():
        die(f"Missing output.csv: {output_csv}")

    times: List[float] = []
    with output_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("time", None)
            if raw in (None, ""):
                continue
            try:
                times.append(float(raw))
            except Exception:
                continue
    if not times:
        die(f"No time values found in {output_csv}")
    return np.unique(np.asarray(times, dtype=float))


def select_time_indices(times: np.ndarray, stride: int) -> tuple[np.ndarray, np.ndarray]:
    if times.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

    step = max(int(stride), 1)
    idx = list(range(0, int(times.size), step))
    if idx[-1] != int(times.size) - 1:
        idx.append(int(times.size) - 1)
    idx_arr = np.asarray(sorted(set(idx)), dtype=int)
    return idx_arr, np.asarray(times[idx_arr], dtype=float)


def read_pressure_for_time(output_csv: Path, vessel_name: str, time_value: float) -> Optional[float]:
    best_diff = float("inf")
    best_val: Optional[float] = None
    with output_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("name", "")).strip() != vessel_name:
                continue
            try:
                t = float(row.get("time", 0.0))
            except Exception:
                continue
            diff = abs(t - float(time_value))
            if diff > best_diff:
                continue
            raw = row.get("pressure_out", None)
            if raw in (None, ""):
                continue
            try:
                best_val = float(raw)
                best_diff = diff
            except Exception:
                continue
    return best_val


def resolve_leak_pressures_for_time(
    output_csv: Path,
    organoid_idx: int,
    time_value: float,
    fallback_inlet_pressure: float,
    fallback_outlet_pressure: float,
) -> tuple[float, float]:
    p_art = read_pressure_for_time(output_csv, f"leak_art_{organoid_idx}", time_value)
    p_ven = read_pressure_for_time(output_csv, f"leak_ven_{organoid_idx}", time_value)
    if p_art is None:
        p_art = float(fallback_inlet_pressure)
    if p_ven is None:
        p_ven = float(fallback_outlet_pressure)
    return float(p_art), float(p_ven)


def build_darcy_command(
    org_dir: Path,
    geom_dir: Path,
    args: argparse.Namespace,
    coords_in: np.ndarray,
    coords_out: np.ndarray,
    checkpoint_time_index: int,
    out_dir: Path,
    time_value: Optional[float] = None,
    output_csv: Optional[Path] = None,
) -> List[str]:
    darcy_script = Path(args.darcy_script).expanduser().resolve()
    if not darcy_script.exists():
        die(f"Missing Darcy script: {darcy_script}")

    perm_region_path = resolve_perm_region_for_organoid(args.perm_region_root, int(org_dir.name.split("_")[1])) if args.perm_region_root else ""
    fallback_inlet_pressure = float(args.fallback_inlet_pressure)
    fallback_outlet_pressure = float(args.fallback_outlet_pressure)

    if args.no_synthetic_vasculature and output_csv is not None and time_value is not None:
        fallback_inlet_pressure, fallback_outlet_pressure = resolve_leak_pressures_for_time(
            output_csv,
            int(org_dir.name.split("_")[1]),
            time_value,
            fallback_inlet_pressure,
            fallback_outlet_pressure,
        )

    darcy_args = [
        "--bioreactor-domain", str(geom_dir / "bioreactor.xdmf"),
        "--facet-file", str(geom_dir / "mesh_tags.xdmf"),
        "--mesh-inlet-file", str(geom_dir / "tagged_branches_inlet.bp"),
        "--mesh-outlet-file", str(geom_dir / "tagged_branches_outlet.bp"),
        "--pres-inlet-file", str(geom_dir / "pressure_checkpoint_inlet.bp"),
        "--pres-outlet-file", str(geom_dir / "pressure_checkpoint_outlet.bp"),
        "--flow-inlet-file", str(geom_dir / "flow_checkpoint_inlet.bp"),
        "--flow-outlet-file", str(geom_dir / "flow_checkpoint_outlet.bp"),
        "--area-inlet-file", str(geom_dir / "area_checkpoint_inlet.bp"),
        "--area-outlet-file", str(geom_dir / "area_checkpoint_outlet.bp"),
        "--coords-inlet", *map(str, coords_in),
        "--coords-outlet", *map(str, coords_out),
        "--branching-in-file", str(org_dir / "branchingData_0.csv"),
        "--branching-out-file", str(org_dir / "branchingData_1.csv"),
        "--checkpoint-time-index", str(int(checkpoint_time_index)),
        "--concave-bc-mode", str(args.concave_bc_mode),
        "--lp-arterial", str(args.lp_arterial),
        "--lp-venous", str(args.lp_venous),
        *(["--skip-1d"] if args.no_synthetic_vasculature else []),
        "--fallback-inlet-pressure", str(fallback_inlet_pressure),
        "--fallback-outlet-pressure", str(fallback_outlet_pressure),
        "--out-dir", str(out_dir),
    ]

    if darcy_script.stem in {"darcy_p1_lm_interior", "darcy_mixed"} and perm_region_path:
        darcy_args.extend([
            "--perm-region-path", perm_region_path,
            "--perm-low", str(args.perm_low),
            "--perm-high", str(args.perm_high),
            "--perm-transition-width", str(args.perm_transition_width),
        ])

    if int(args.darcy_mpi_procs) > 1:
        mpi_launcher = resolve_mpi_launcher(str(args.darcy_mpirun_cmd))
        return [
            mpi_launcher,
            "-n",
            str(args.darcy_mpi_procs),
            sys.executable,
            str(darcy_script),
            *darcy_args,
        ]
    return [sys.executable, str(darcy_script), *darcy_args]


def run_darcy_time_series(
    iter_dir: Path,
    args: argparse.Namespace,
    selected_indices: np.ndarray,
    selected_times: np.ndarray,
) -> None:
    output_csv = iter_dir / "output.csv"
    for step_idx, time_index in enumerate(selected_indices):
        time_value = float(selected_times[step_idx])
        print(
            f"[darcy] time step {step_idx + 1}/{len(selected_indices)} "
            f"(checkpoint index={int(time_index)}, time={time_value:.6g})",
            flush=True,
        )
        for k in range(1, args.n_organoids + 1):
            org = iter_dir / f"organoid_{k}"
            geom = org / "geometry"
            coords_in = read_seed_coords_from_branching(
                org / "branchingData_0.csv",
                shifted_seed_fallback(np.asarray(args.coords_inlet, dtype=float), k, args.dy_step),
            )
            coords_out = read_seed_coords_from_branching(
                org / "branchingData_1.csv",
                shifted_seed_fallback(np.asarray(args.coords_outlet, dtype=float), k, args.dy_step),
            )
            out_dir = org / "out_darcy_transient" / f"time_{int(time_index):04d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = build_darcy_command(
                org,
                geom,
                args,
                coords_in,
                coords_out,
                checkpoint_time_index=int(time_index),
                out_dir=out_dir,
                time_value=time_value,
                output_csv=output_csv,
            )
            run(cmd)


def run_darcy_time_series_batched(
    iter_dir: Path,
    args: argparse.Namespace,
    selected_indices: np.ndarray,
    selected_times: np.ndarray,
    coupling_iteration: int,
) -> None:
    script = Path(args.darcy_timeseries_script).expanduser().resolve()
    if not script.exists():
        die(f"Missing Darcy time-series script: {script}")

    output_csv = iter_dir / "output.csv"
    write_fields = bool(
        int(args.darcy_field_output_every) > 0
        and int(coupling_iteration) > 0
        and int(coupling_iteration) % int(args.darcy_field_output_every) == 0
    )

    for k in range(1, args.n_organoids + 1):
        org = iter_dir / f"organoid_{k}"
        geom = org / "geometry"
        coords_in = read_seed_coords_from_branching(
            org / "branchingData_0.csv",
            shifted_seed_fallback(np.asarray(args.coords_inlet, dtype=float), k, args.dy_step),
        )
        coords_out = read_seed_coords_from_branching(
            org / "branchingData_1.csv",
            shifted_seed_fallback(np.asarray(args.coords_outlet, dtype=float), k, args.dy_step),
        )
        perm_region_path = resolve_perm_region_for_organoid(args.perm_region_root, k) if args.perm_region_root else ""
        out_dir = org / "out_darcy_transient"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(script),
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
            "--coords-inlet", *map(str, coords_in),
            "--coords-outlet", *map(str, coords_out),
            "--branching-in-file", str(org / "branchingData_0.csv"),
            "--branching-out-file", str(org / "branchingData_1.csv"),
            "--concave-bc-mode", str(args.concave_bc_mode),
            "--lp-arterial", str(args.lp_arterial),
            "--lp-venous", str(args.lp_venous),
            "--fallback-inlet-pressure", str(args.fallback_inlet_pressure),
            "--fallback-outlet-pressure", str(args.fallback_outlet_pressure),
            "--organoid-index", str(k),
            "--out-dir", str(out_dir),
            "--time-indices", *[str(int(idx)) for idx in np.asarray(selected_indices, dtype=int)],
            "--time-values", *[str(float(t)) for t in np.asarray(selected_times, dtype=float)],
        ]
        if args.no_synthetic_vasculature:
            cmd.append("--skip-1d")
            cmd.extend(["--leak-pressure-output-csv", str(output_csv)])
        if perm_region_path:
            cmd.extend([
                "--perm-region-path", perm_region_path,
                "--perm-low", str(args.perm_low),
                "--perm-high", str(args.perm_high),
                "--perm-transition-width", str(args.perm_transition_width),
            ])
        if write_fields:
            cmd.append("--write-fields")
        if int(args.darcy_mpi_procs) > 1:
            cmd = build_mpi_command(
                str(args.darcy_mpirun_cmd),
                int(args.darcy_mpi_procs),
                cmd,
            )
        run(cmd)


def run_darcy_time_series_batched_inprocess(
    iter_dir: Path,
    args: argparse.Namespace,
    selected_indices: np.ndarray,
    selected_times: np.ndarray,
    coupling_iteration: int,
) -> None:
    script = Path(args.darcy_timeseries_script).expanduser().resolve()
    if not script.exists():
        die(f"Missing Darcy time-series script: {script}")
    script_dir = str(script.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import darcy_mixed_timeseries as darcy_timeseries

    comm = MPI.COMM_WORLD
    output_csv = iter_dir / "output.csv"
    write_fields = bool(
        int(args.darcy_field_output_every) > 0
        and int(coupling_iteration) > 0
        and int(coupling_iteration) % int(args.darcy_field_output_every) == 0
    )

    for k in range(1, args.n_organoids + 1):
        org = iter_dir / f"organoid_{k}"
        geom = org / "geometry"
        coords_in = read_seed_coords_from_branching(
            org / "branchingData_0.csv",
            shifted_seed_fallback(np.asarray(args.coords_inlet, dtype=float), k, args.dy_step),
        )
        coords_out = read_seed_coords_from_branching(
            org / "branchingData_1.csv",
            shifted_seed_fallback(np.asarray(args.coords_outlet, dtype=float), k, args.dy_step),
        )
        perm_region_path = resolve_perm_region_for_organoid(args.perm_region_root, k) if args.perm_region_root else ""
        out_dir = org / "out_darcy_transient"
        out_dir.mkdir(parents=True, exist_ok=True)

        darcy_timeseries.run_job(
            bioreactor_domain=str(geom / "bioreactor.xdmf"),
            facet_file=str(geom / "mesh_tags.xdmf"),
            mesh_inlet_file=str(geom / "tagged_branches_inlet.bp"),
            mesh_outlet_file=str(geom / "tagged_branches_outlet.bp"),
            pres_inlet_file=str(geom / "pressure_checkpoint_inlet.bp"),
            pres_outlet_file=str(geom / "pressure_checkpoint_outlet.bp"),
            flow_inlet_file=str(geom / "flow_checkpoint_inlet.bp"),
            flow_outlet_file=str(geom / "flow_checkpoint_outlet.bp"),
            area_inlet_file=str(geom / "area_checkpoint_inlet.bp"),
            area_outlet_file=str(geom / "area_checkpoint_outlet.bp"),
            coords_inlet=coords_in,
            coords_outlet=coords_out,
            branching_in_file=str(org / "branchingData_0.csv"),
            branching_out_file=str(org / "branchingData_1.csv"),
            concave_bc_mode=str(args.concave_bc_mode),
            lp_arterial=float(args.lp_arterial),
            lp_venous=float(args.lp_venous),
            fallback_inlet_pressure=float(args.fallback_inlet_pressure),
            fallback_outlet_pressure=float(args.fallback_outlet_pressure),
            organoid_index=int(k),
            out_dir=str(out_dir),
            checkpoint_indices=[int(idx) for idx in np.asarray(selected_indices, dtype=int)],
            checkpoint_times=[float(t) for t in np.asarray(selected_times, dtype=float)],
            skip_1d=bool(args.no_synthetic_vasculature),
            leak_pressure_output_csv=str(output_csv) if args.no_synthetic_vasculature else None,
            perm_region_path=perm_region_path,
            perm_low=float(args.perm_low),
            perm_high=float(args.perm_high),
            perm_transition_width=float(args.perm_transition_width),
            write_fields=write_fields,
        )
        comm.barrier()


def max_rel(a: np.ndarray, b: np.ndarray, floor: float = 1e-20) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    if aa.size == 0 or bb.size == 0:
        return 0.0
    n = min(aa.size, bb.size)
    den = np.maximum(np.abs(bb[:n]), floor)
    return float(np.max(np.abs(aa[:n] - bb[:n]) / den))


def collect_interface_waveforms(
    iter_dir: Path,
    n_organoids: int,
    selected_indices: np.ndarray,
    selected_times: np.ndarray,
) -> Dict[int, dict]:
    results: Dict[int, dict] = {}
    for k in range(1, n_organoids + 1):
        timeseries_path = iter_dir / f"organoid_{k}" / "out_darcy_transient" / "interface_bc_timeseries.json"
        if timeseries_path.exists():
            raw = load_json(timeseries_path)
            results[k] = {
                "times": np.asarray(raw.get("times", selected_times), dtype=float),
                "skip_1d": bool(raw.get("skip_1d", False)),
                "p_inlet_nodes": np.asarray(raw.get("p_inlet_nodes", []), dtype=float),
                "p_outlet_nodes": np.asarray(raw.get("p_outlet_nodes", []), dtype=float),
                "q_artery_leak": np.asarray(raw.get("q_artery_leak", []), dtype=float),
                "q_venous_leak": np.asarray(raw.get("q_venous_leak", []), dtype=float),
                "q_inlet": np.asarray(raw.get("q_inlet", []), dtype=float),
                "q_inlet_target": np.asarray(raw.get("q_inlet_target", []), dtype=float),
                "p_inlet_target": np.asarray(raw.get("p_inlet_target", []), dtype=float),
                "p_outlet_target": np.asarray(raw.get("p_outlet_target", []), dtype=float),
                "p_concave_inlet_bc": np.asarray(raw.get("p_concave_inlet_bc", []), dtype=float),
                "p_concave_outlet_bc": np.asarray(raw.get("p_concave_outlet_bc", []), dtype=float),
            }
            continue

        paths = [
            iter_dir / f"organoid_{k}" / "out_darcy_transient" / f"time_{int(idx):04d}" / "out_darcy" / "interface_bc.json"
            for idx in selected_indices
        ]
        for path in paths:
            if not path.exists():
                die(f"Missing transient Darcy interface file: {path}")

        raw = [load_json(path) for path in paths]
        results[k] = {
            "times": np.asarray(selected_times, dtype=float),
            "skip_1d": bool(raw[0].get("skip_1d", False)),
            "p_inlet_nodes": np.asarray([row.get("p_inlet_nodes", []) for row in raw], dtype=float),
            "p_outlet_nodes": np.asarray([row.get("p_outlet_nodes", []) for row in raw], dtype=float),
            "q_artery_leak": np.asarray([row.get("q_artery_leak", 0.0) for row in raw], dtype=float),
            "q_venous_leak": np.asarray([row.get("q_venous_leak", 0.0) for row in raw], dtype=float),
            "q_inlet": np.asarray([row.get("q_inlet", []) for row in raw], dtype=float),
            "q_inlet_target": np.asarray([row.get("q_inlet_target", []) for row in raw], dtype=float),
            "p_inlet_target": np.asarray([row.get("p_inlet_target", []) for row in raw], dtype=float),
            "p_outlet_target": np.asarray([row.get("p_outlet_target", []) for row in raw], dtype=float),
            "p_concave_inlet_bc": np.asarray([row.get("p_concave_inlet_bc", 0.0) for row in raw], dtype=float),
            "p_concave_outlet_bc": np.asarray([row.get("p_concave_outlet_bc", 0.0) for row in raw], dtype=float),
        }
    return results


def compute_convergence(
    waveforms: Dict[int, dict],
    current_state: Dict[int, dict],
    coupling_times: np.ndarray,
    tol_q: float,
    tol_p: float,
) -> tuple[bool, Dict[int, dict]]:
    metrics: Dict[int, dict] = {}
    all_ok = True
    for k, data in waveforms.items():
        if bool(data.get("skip_1d", False)):
            q_art_target = interp_series(
                coupling_times,
                np.asarray(current_state[k]["q_artery_leak_bc"], dtype=float),
                data["times"],
            )
            q_ven_target = interp_series(
                coupling_times,
                np.asarray(current_state[k]["q_venous_leak_bc"], dtype=float),
                data["times"],
            )
            rq = max(
                max_rel(-np.asarray(data["q_artery_leak"], dtype=float), q_art_target),
                max_rel(-np.asarray(data["q_venous_leak"], dtype=float), q_ven_target),
            )
            rp = 0.0
        else:
            rq = 0.0
            rp = 0.0
            for i in range(len(data["times"])):
                rq = max(
                    rq,
                    max_rel(
                        np.asarray(data["q_inlet"][i], dtype=float),
                        np.asarray(data["q_inlet_target"][i], dtype=float),
                    ),
                )
                rp = max(
                    rp,
                    max_rel(
                        np.asarray(data["p_inlet_nodes"][i], dtype=float),
                        np.asarray(data["p_inlet_target"][i], dtype=float),
                        floor=1e-12,
                    ),
                    max_rel(
                        np.asarray(data["p_outlet_nodes"][i], dtype=float),
                        np.asarray(data["p_outlet_target"][i], dtype=float),
                        floor=1e-12,
                    ),
                )
        ok = bool(rq <= tol_q and rp <= tol_p)
        metrics[k] = {"rel_q": float(rq), "rel_p": float(rp), "converged": ok}
        all_ok = all_ok and ok
    return all_ok, metrics


def build_target_state_from_waveforms(
    waveforms: Dict[int, dict],
    coupling_times: np.ndarray,
) -> Dict[int, dict]:
    target: Dict[int, dict] = {}
    for k, data in waveforms.items():
        target[k] = {
            "p_inlet_bc": interp_series(data["times"], data["p_inlet_nodes"], coupling_times),
            "p_outlet_bc": interp_series(data["times"], data["p_outlet_nodes"], coupling_times),
            "q_artery_leak_bc": interp_series(data["times"], -np.asarray(data["q_artery_leak"], dtype=float), coupling_times),
            "q_venous_leak_bc": interp_series(data["times"], -np.asarray(data["q_venous_leak"], dtype=float), coupling_times),
            "skip_1d": bool(data.get("skip_1d", False)),
        }
    return target


def relax_state(
    current_state: Dict[int, dict],
    target_state: Dict[int, dict],
    relax: float,
) -> Dict[int, dict]:
    w = float(np.clip(relax, 0.0, 1.0))
    updated: Dict[int, dict] = {}
    for k in current_state:
        updated[k] = {
            "p_inlet_bc": (1.0 - w) * np.asarray(current_state[k]["p_inlet_bc"], dtype=float)
            + w * np.asarray(target_state[k]["p_inlet_bc"], dtype=float),
            "p_outlet_bc": (1.0 - w) * np.asarray(current_state[k]["p_outlet_bc"], dtype=float)
            + w * np.asarray(target_state[k]["p_outlet_bc"], dtype=float),
            "q_artery_leak_bc": (1.0 - w) * np.asarray(current_state[k]["q_artery_leak_bc"], dtype=float)
            + w * np.asarray(target_state[k]["q_artery_leak_bc"], dtype=float),
            "q_venous_leak_bc": (1.0 - w) * np.asarray(current_state[k]["q_venous_leak_bc"], dtype=float)
            + w * np.asarray(target_state[k]["q_venous_leak_bc"], dtype=float),
            "skip_1d": bool(target_state[k].get("skip_1d", current_state[k].get("skip_1d", False))),
        }
    return updated


def serialize_state(state: Dict[int, dict]) -> dict:
    payload = {}
    for k, data in state.items():
        payload[str(k)] = {
            "skip_1d": bool(data.get("skip_1d", False)),
            "p_inlet_bc": np.asarray(data["p_inlet_bc"], dtype=float).tolist(),
            "p_outlet_bc": np.asarray(data["p_outlet_bc"], dtype=float).tolist(),
            "q_artery_leak_bc": np.asarray(data["q_artery_leak_bc"], dtype=float).tolist(),
            "q_venous_leak_bc": np.asarray(data["q_venous_leak_bc"], dtype=float).tolist(),
        }
    return payload


def serialize_waveforms(waveforms: Dict[int, dict]) -> dict:
    payload = {}
    for k, data in waveforms.items():
        payload[str(k)] = {
            "times": np.asarray(data["times"], dtype=float).tolist(),
            "skip_1d": bool(data.get("skip_1d", False)),
            "p_inlet_nodes": np.asarray(data["p_inlet_nodes"], dtype=float).tolist(),
            "p_outlet_nodes": np.asarray(data["p_outlet_nodes"], dtype=float).tolist(),
            "q_artery_leak": np.asarray(data["q_artery_leak"], dtype=float).tolist(),
            "q_venous_leak": np.asarray(data["q_venous_leak"], dtype=float).tolist(),
            "q_inlet": np.asarray(data["q_inlet"], dtype=float).tolist(),
            "q_inlet_target": np.asarray(data["q_inlet_target"], dtype=float).tolist(),
            "p_inlet_target": np.asarray(data["p_inlet_target"], dtype=float).tolist(),
            "p_outlet_target": np.asarray(data["p_outlet_target"], dtype=float).tolist(),
            "p_concave_inlet_bc": np.asarray(data["p_concave_inlet_bc"], dtype=float).tolist(),
            "p_concave_outlet_bc": np.asarray(data["p_concave_outlet_bc"], dtype=float).tolist(),
        }
    return payload


def parse_args() -> argparse.Namespace:
    repo_src = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Transient quasi-static coupling driver restarting from a converged steady-state run."
    )
    ap.add_argument("--steady-run", default=str(repo_src / "coupling-output-no-vasc" / "run_20"),
                    help="Converged steady-state run to restart from.")
    ap.add_argument("--transient-root", default=str(repo_src / "coupling-output-transient"),
                    help="Output root for transient outer coupling iterations.")
    ap.add_argument("--n-organoids", type=int, default=4,
                    help="Number of organoids. If 0, detect from --steady-run.")
    ap.add_argument("--max-iter", type=int, default=10)
    ap.add_argument("--tol-q", type=float, default=1e-3)
    ap.add_argument("--tol-p", type=float, default=1e-3)
    ap.add_argument("--relaxation", type=float, default=0.3)
    ap.add_argument("--number-of-cardiac-cycles", type=int, default=3)
    ap.add_argument("--number-of-time-pts-per-cardiac-cycle", type=int, default=31)
    ap.add_argument("--pulse-ramp-stages", type=int, default=10,
                    help="Number of waveform-amplitude continuation stages (1 = direct full waveform).")
    ap.add_argument("--pulse-ramp-baseline", choices=["mean", "steady", "first"], default="steady",
                    help="Baseline waveform used at ramp alpha=0.")

    ap.add_argument("--channel-inlet-bc", default="INFLOW")
    ap.add_argument("--channel-outlet-bc", default="OUT3")
    ap.add_argument(
        "--inflow-waveform",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/files/sv-3d-projection/pulsatile.flow",
    )
    ap.add_argument(
        "--outlet-pressure-waveform",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/files/sv-3d-projection/pressure.flow",
    )

    ap.add_argument("--svzerodsolver", default="svzerodsolver")
    ap.add_argument("--run-and-split", default=str(repo_src / "prep" / "run_and_split_svzerod.py"))
    ap.add_argument("--darcy-script", default=str(repo_src / "solves" / "darcy_mixed.py"))
    ap.add_argument("--darcy-timeseries-script", default=str(repo_src / "solves" / "darcy_mixed_timeseries.py"))
    ap.add_argument("--darcy-mpi-procs", type=int, default=1,
                    help="MPI ranks for each Darcy solve (1 = serial).")
    ap.add_argument("--darcy-mpirun-cmd", default="/opt/miniconda3/envs/fenicsx-env/bin/mpiexec.hydra")
    ap.add_argument("--darcy-time-index-stride", type=int, default=1,
                    help="Solve Darcy every Nth transient checkpoint and always include the final checkpoint.")
    ap.add_argument("--use-batched-darcy-timeseries", action="store_true",
                    help="Run one mixed-Darcy process per organoid across all selected times and write interface_bc_timeseries.json.")
    ap.add_argument("--darcy-field-output-every", type=int, default=1,
                    help="Write p.xdmf/u.xdmf time series every N transient coupling iterations (0 disables field output).")

    ap.add_argument("--dy-step", type=float, default=0.6)
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-.28, 0.9, .5375])
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.30, 0.9, .5375])
    ap.add_argument(
        "--perm-region-root",
        default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/files/stl/organoid-growth-domains/sphere",
    )
    ap.add_argument("--perm-low", type=float, default=1.0e-9)
    ap.add_argument("--perm-high", type=float, default=2.0e-7)
    ap.add_argument("--perm-transition-width", type=float, default=0.01)

    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    ap.add_argument("--lp-arterial", type=float, default=0.0)
    ap.add_argument("--lp-venous", type=float, default=0.0)
    ap.add_argument("--no-synthetic-vasculature", action="store_true",
                    help="Skip transient 1D terminal coupling and only exchange wall leak flows.")
    ap.add_argument("--fallback-inlet-pressure", type=float, default=0.0)
    ap.add_argument("--fallback-outlet-pressure", type=float, default=0.0)

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--debug", action="store_true")
    return ap.parse_args()


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.rank
    args = parse_args()
    steady_run = Path(args.steady_run).expanduser().resolve()
    if not steady_run.exists():
        die(f"Steady restart run does not exist: {steady_run}")

    args.n_organoids = int(args.n_organoids) if int(args.n_organoids) > 0 else detect_n_organoids(steady_run)

    transient_root = Path(args.transient_root).expanduser().resolve()
    transient_root.mkdir(parents=True, exist_ok=True)

    inflow_path = Path(args.inflow_waveform).expanduser().resolve()
    outlet_path = Path(args.outlet_pressure_waveform).expanduser().resolve()
    if not inflow_path.exists():
        die(f"Missing inflow waveform file: {inflow_path}")
    if not outlet_path.exists():
        die(f"Missing outlet pressure waveform file: {outlet_path}")

    inflow_t_raw, inflow_q_full = read_waveform_file(inflow_path)
    outlet_t_raw, outlet_p_full = read_waveform_file(outlet_path)
    cycle_period = max(float(inflow_t_raw[-1]), float(outlet_t_raw[-1]))

    inflow_t_cycle = normalize_waveform_period(inflow_t_raw, cycle_period)
    outlet_t_cycle = normalize_waveform_period(outlet_t_raw, cycle_period)
    coupling_times_cycle = merge_time_grids(inflow_t_cycle, outlet_t_cycle)

    base_combined = steady_run / "combined.in"
    if not base_combined.exists():
        die(f"Missing combined.in in steady restart run: {base_combined}")
    base_deck = load_json(base_combined)

    if int(args.max_iter) <= 0:
        die("--max-iter must be positive")
    if int(args.pulse_ramp_stages) <= 0:
        die("--pulse-ramp-stages must be positive")

    coupling_times = coupling_times_cycle
    current_state = load_steady_interface_state(steady_run, args.n_organoids, coupling_times)
    if args.no_synthetic_vasculature:
        for k in current_state:
            current_state[k]["p_inlet_bc"] = np.zeros((coupling_times.size, 0), dtype=float)
            current_state[k]["p_outlet_bc"] = np.zeros((coupling_times.size, 0), dtype=float)
            current_state[k]["skip_1d"] = True

    inflow_bc_steady = bc_scalar_value(find_bc(base_deck, args.channel_inlet_bc), "Q")
    outlet_bc_steady = bc_scalar_value(find_bc(base_deck, args.channel_outlet_bc), "P")
    inflow_baseline = waveform_baseline_value(
        args.pulse_ramp_baseline,
        inflow_t_cycle,
        inflow_q_full,
        inflow_bc_steady,
    )
    outlet_baseline = waveform_baseline_value(
        args.pulse_ramp_baseline,
        outlet_t_cycle,
        outlet_p_full,
        outlet_bc_steady,
    )
    pulse_alphas = make_pulse_ramp_alphas(int(args.pulse_ramp_stages))

    last_iter_dir: Optional[Path] = None

    for stage_idx, pulse_alpha in enumerate(pulse_alphas):
        inflow_q = blend_waveform(inflow_q_full, inflow_baseline, pulse_alpha)
        outlet_p = blend_waveform(outlet_p_full, outlet_baseline, pulse_alpha)

        if len(pulse_alphas) == 1:
            stage_root = transient_root
        else:
            stage_root = transient_root / f"stage_{stage_idx:02d}_alpha_{pulse_alpha:.3f}"
            stage_root.mkdir(parents=True, exist_ok=True)

        stage_converged = False

        for it in range(int(args.max_iter)):
            iter_dir = stage_root / f"iter_{it:03d}"
            if iter_dir.exists() and not args.overwrite:
                die(f"Iteration directory already exists (use --overwrite to replace): {iter_dir}")

            if rank == 0:
                prepare_iteration_dir(steady_run, iter_dir, args.n_organoids)

                deck = json.loads(json.dumps(base_deck))
                update_simulation_parameters(
                    deck,
                    int(args.number_of_cardiac_cycles),
                    int(args.number_of_time_pts_per_cardiac_cycle),
                )
                apply_channel_waveforms(
                    deck,
                    args.channel_inlet_bc,
                    inflow_t_cycle,
                    inflow_q,
                    args.channel_outlet_bc,
                    outlet_t_cycle,
                    outlet_p,
                )
                for k in range(1, args.n_organoids + 1):
                    apply_organoid_interface_state(
                        deck,
                        k,
                        current_state[k],
                        coupling_times,
                        allow_missing_terminal_bcs=bool(args.no_synthetic_vasculature),
                    )
                save_json(iter_dir / "combined.in", deck)

                run([
                    sys.executable, str(Path(args.run_and_split).expanduser().resolve()),
                    "--exe", args.svzerodsolver,
                    "--input", str(iter_dir / "combined.in"),
                    "--outdir", str(iter_dir),
                    "--output", "output.csv",
                    "--organoid-root", str(iter_dir),
                    "--no-plot",
                ] + (["--debug"] if args.debug else []))

                if not args.no_synthetic_vasculature:
                    update_1d_checkpoints(
                        iter_dir,
                        args.n_organoids,
                        np.asarray(args.coords_inlet, dtype=float),
                        np.asarray(args.coords_outlet, dtype=float),
                        args.dy_step,
                    )

                output_times = read_unique_output_times(iter_dir / "output.csv")
                selected_indices, selected_times = select_time_indices(output_times, args.darcy_time_index_stride)
            else:
                selected_indices, selected_times = None, None
            comm.barrier()
            selected_indices = np.asarray(comm.bcast(selected_indices, root=0), dtype=int)
            selected_times = np.asarray(comm.bcast(selected_times, root=0), dtype=float)
            coupling_iteration = stage_idx * int(args.max_iter) + it + 1
            if args.use_batched_darcy_timeseries:
                if comm.size > 1:
                    if int(args.darcy_mpi_procs) != int(comm.size):
                        die(
                            f"When launching coupler_transient.py under MPI, "
                            f"--darcy-mpi-procs must match MPI size ({comm.size}), "
                            f"got {args.darcy_mpi_procs}."
                        )
                    run_darcy_time_series_batched_inprocess(iter_dir, args, selected_indices, selected_times, coupling_iteration)
                else:
                    run_darcy_time_series_batched(iter_dir, args, selected_indices, selected_times, coupling_iteration)
            else:
                if rank == 0:
                    run_darcy_time_series(iter_dir, args, selected_indices, selected_times)
            comm.barrier()

            if rank == 0:
                waveforms = collect_interface_waveforms(iter_dir, args.n_organoids, selected_indices, selected_times)
                all_ok, metrics = compute_convergence(
                    waveforms,
                    current_state,
                    coupling_times,
                    tol_q=float(args.tol_q),
                    tol_p=float(args.tol_p),
                )

                target_state = build_target_state_from_waveforms(waveforms, coupling_times)
                summary = {
                    "stage_index": stage_idx,
                    "pulse_alpha": float(pulse_alpha),
                    "iteration": it,
                    "steady_run": str(steady_run),
                    "cycle_period": float(cycle_period),
                    "number_of_cardiac_cycles": int(args.number_of_cardiac_cycles),
                    "number_of_time_pts_per_cardiac_cycle": int(args.number_of_time_pts_per_cardiac_cycle),
                    "pulse_ramp_stages": int(args.pulse_ramp_stages),
                    "pulse_ramp_baseline": str(args.pulse_ramp_baseline),
                    "channel_inflow_baseline": float(inflow_baseline),
                    "channel_outlet_baseline": float(outlet_baseline),
                    "coupling_times_per_cycle": coupling_times_cycle.tolist(),
                    "coupling_times": coupling_times.tolist(),
                    "channel_inflow_waveform": {"t": inflow_t_cycle.tolist(), "Q": inflow_q.tolist()},
                    "channel_outlet_waveform": {"t": outlet_t_cycle.tolist(), "P": outlet_p.tolist()},
                    "selected_output_times": selected_times.tolist(),
                    "selected_output_indices": selected_indices.tolist(),
                    "current_state": serialize_state(current_state),
                    "target_state": serialize_state(target_state),
                    "measured_waveforms": serialize_waveforms(waveforms),
                    "convergence": {str(k): v for k, v in metrics.items()},
                    "converged": bool(all_ok),
                }
                save_json(iter_dir / "transient_summary.json", summary)
                last_iter_dir = iter_dir

                print(
                    f"\n[check] pulse stage {stage_idx + 1}/{len(pulse_alphas)} "
                    f"(alpha={pulse_alpha:.3f}), transient coupling iteration {it + 1}/{args.max_iter}"
                )
                for k in range(1, args.n_organoids + 1):
                    metric = metrics[k]
                    print(
                        f"  organoid_{k}: rel_q={metric['rel_q']:.3e}, "
                        f"rel_p={metric['rel_p']:.3e}, converged={metric['converged']}"
                    )

                if all_ok:
                    current_state = target_state
                    stage_converged = True
                else:
                    current_state = relax_state(current_state, target_state, float(args.relaxation))
            else:
                all_ok = False
                stage_converged = False
            all_ok = bool(comm.bcast(all_ok, root=0))
            stage_converged = bool(comm.bcast(stage_converged, root=0))

            if all_ok:
                if rank == 0:
                    if stage_idx == len(pulse_alphas) - 1:
                        print(f"\n[done] transient coupling converged at {iter_dir.name}.")
                    else:
                        print(
                            f"\n[stage] pulse stage {stage_idx + 1}/{len(pulse_alphas)} "
                            f"converged at alpha={pulse_alpha:.3f}; continuing to the next stage."
                        )
                if stage_idx == len(pulse_alphas) - 1:
                    return
                break

        if rank == 0 and not stage_converged:
            print(
                f"\n[stage] pulse stage {stage_idx + 1}/{len(pulse_alphas)} "
                f"reached max iterations at alpha={pulse_alpha:.3f}."
            )

    if rank == 0 and last_iter_dir is not None:
        print(
            f"\n[done] completed pulse-ramp continuation without full convergence at the final stage. "
            f"Last results are in {last_iter_dir}."
        )
        return
    if rank == 0:
        print(f"\n[done] no transient iterations were completed.")


if __name__ == "__main__":
    main()
