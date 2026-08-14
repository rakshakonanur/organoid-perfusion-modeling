#!/usr/bin/env python3
"""Build and validate affine Darcy input/output response maps.

The mixed Darcy problem is linear while the mesh, permeability, boundary types,
and terminal marker layout remain fixed.  This tool extracts the changing
terminal-flow/concave-pressure inputs and the terminal-pressure/concave-flow
outputs from completed coupling runs and fits

    y = h0 + H @ u.

Important: a history-fitted map is exact only on the input subspace spanned by
the supplied runs.  The report records the identified rank and input dimension
so an under-excited history cannot be mistaken for a complete basis map.
The ``exact`` command instead reuses one mixed-Darcy factorization and solves
every independent terminal-flow and concave-pressure input basis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Sample:
    run: int
    inputs: np.ndarray
    outputs: np.ndarray


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _as_float_array(data: dict[str, Any], key: str, expected: int | None = None) -> np.ndarray:
    value = np.asarray(data.get(key, []), dtype=float).reshape(-1)
    if expected is not None and value.size != expected:
        raise ValueError(f"{key} has {value.size} values; expected {expected}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{key} contains non-finite values")
    return value


def _profile_inputs(data: dict[str, Any]) -> tuple[list[str], np.ndarray]:
    mode = str(data.get("concave_pressure_profile", "constant")).strip().lower()
    if mode == "linear":
        names = [
            "p_concave_arterial_low",
            "p_concave_arterial_high",
            "p_concave_venous_low",
            "p_concave_venous_high",
        ]
        values = np.asarray([
            data["p_concave_inlet_bc_low"],
            data["p_concave_inlet_bc_high"],
            data["p_concave_outlet_bc_low"],
            data["p_concave_outlet_bc_high"],
        ], dtype=float)
    else:
        names = ["p_concave_arterial", "p_concave_venous"]
        values = np.asarray([
            data["p_concave_inlet_bc"],
            data["p_concave_outlet_bc"],
        ], dtype=float)
    return names, values


def _layout(data: dict[str, Any]) -> dict[str, Any]:
    constraint_targets = _as_float_array(data, "terminal_flux_constraint_targets")
    constraint_markers = np.asarray(
        data.get("terminal_flux_constraint_markers", []), dtype=int
    ).reshape(-1)
    descriptions = [str(value) for value in data.get("terminal_flux_constraint_descriptions", [])]
    if constraint_markers.size != constraint_targets.size:
        raise ValueError("Terminal constraint marker/target counts do not match")
    if len(descriptions) != constraint_targets.size:
        raise ValueError("Terminal constraint description/target counts do not match")

    inlet_ids = np.asarray(data.get("branch_ids_inlet", []), dtype=int).reshape(-1)
    outlet_ids = np.asarray(data.get("branch_ids_outlet", []), dtype=int).reshape(-1)
    p_in = _as_float_array(data, "p_inlet_nodes", inlet_ids.size)
    p_out = _as_float_array(data, "p_outlet_nodes", outlet_ids.size)
    profile_names, _ = _profile_inputs(data)

    input_names = [
        f"terminal_flux:{description}:marker_{int(marker)}"
        for marker, description in zip(constraint_markers, descriptions)
    ] + profile_names
    output_names = (
        [f"p_arterial_branch_{int(branch)}" for branch in inlet_ids]
        + [f"p_venous_branch_{int(branch)}" for branch in outlet_ids]
        + ["q_concave_arterial", "q_concave_venous"]
    )
    return {
        "input_names": input_names,
        "output_names": output_names,
        "constraint_markers": constraint_markers.tolist(),
        "constraint_descriptions": descriptions,
        "branch_ids_inlet": inlet_ids.tolist(),
        "branch_ids_outlet": outlet_ids.tolist(),
        "concave_pressure_profile": str(data.get("concave_pressure_profile", "constant")),
        "concave_pressure_profile_axis": str(data.get("concave_pressure_profile_axis", "")),
        "n_terminal_pressures": int(p_in.size + p_out.size),
    }


def _extract_vectors(data: dict[str, Any], layout: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    markers = np.asarray(data.get("terminal_flux_constraint_markers", []), dtype=int).reshape(-1)
    descriptions = [str(value) for value in data.get("terminal_flux_constraint_descriptions", [])]
    if markers.tolist() != layout["constraint_markers"] or descriptions != layout["constraint_descriptions"]:
        raise ValueError("Terminal constraint ordering changed")
    if np.asarray(data.get("branch_ids_inlet", []), dtype=int).reshape(-1).tolist() != layout["branch_ids_inlet"]:
        raise ValueError("Arterial branch ordering changed")
    if np.asarray(data.get("branch_ids_outlet", []), dtype=int).reshape(-1).tolist() != layout["branch_ids_outlet"]:
        raise ValueError("Venous branch ordering changed")

    q = _as_float_array(data, "terminal_flux_constraint_targets", markers.size)
    profile_names, profile = _profile_inputs(data)
    if layout["input_names"][-len(profile_names):] != profile_names:
        raise ValueError("Concave profile type changed")
    p_in = _as_float_array(data, "p_inlet_nodes", len(layout["branch_ids_inlet"]))
    p_out = _as_float_array(data, "p_outlet_nodes", len(layout["branch_ids_outlet"]))
    outputs = np.concatenate((
        p_in,
        p_out,
        np.asarray([data["q_artery_leak"], data["q_venous_leak"]], dtype=float),
    ))
    inputs = np.concatenate((q, profile))
    if not (np.all(np.isfinite(inputs)) and np.all(np.isfinite(outputs))):
        raise ValueError("Input/output vector contains non-finite values")
    return inputs, outputs


def load_samples(root: Path, organoid: int) -> tuple[list[Sample], dict[str, Any], list[str]]:
    candidates = sorted(
        root.glob("run_*"),
        key=lambda path: int(path.name.split("_")[-1]),
    )
    samples: list[Sample] = []
    skipped: list[str] = []
    layout: dict[str, Any] | None = None
    for run_dir in candidates:
        try:
            run = int(run_dir.name.split("_")[-1])
        except ValueError:
            continue
        path = run_dir / f"organoid_{int(organoid)}" / "out_darcy" / "interface_bc.json"
        if not path.exists():
            skipped.append(f"run_{run}: missing {path.name}")
            continue
        try:
            data = _load_json(path)
            if layout is None:
                layout = _layout(data)
            inputs, outputs = _extract_vectors(data, layout)
            samples.append(Sample(run=run, inputs=inputs, outputs=outputs))
        except (KeyError, TypeError, ValueError) as exc:
            skipped.append(f"run_{run}: {exc}")
    if layout is None or not samples:
        raise RuntimeError(f"No usable Darcy samples found below {root}")
    return samples, layout, skipped


@dataclass
class AffineFit:
    h0: np.ndarray
    H: np.ndarray
    input_center: np.ndarray
    input_scale: np.ndarray
    singular_values: np.ndarray
    rank: int
    condition_number: float


def fit_affine(samples: list[Sample], rcond: float) -> AffineFit:
    X = np.vstack([sample.inputs for sample in samples])
    Y = np.vstack([sample.outputs for sample in samples])
    center = np.mean(X, axis=0)
    scale = np.std(X, axis=0)
    scale_floor = np.maximum(np.max(np.abs(X), axis=0), 1.0) * 1.0e-14
    active = scale > scale_floor
    safe_scale = np.where(active, scale, 1.0)
    Xn = (X - center) / safe_scale
    design = np.column_stack((np.ones(X.shape[0]), Xn[:, active]))
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        design, Y, rcond=float(rcond)
    )
    H = np.zeros((Y.shape[1], X.shape[1]), dtype=float)
    if np.any(active):
        H[:, active] = (coefficients[1:, :] / safe_scale[active, None]).T
    h0 = coefficients[0, :] - H @ center
    positive_sv = singular_values[singular_values > 0.0]
    condition = (
        float(positive_sv[0] / positive_sv[-1])
        if positive_sv.size
        else float("inf")
    )
    return AffineFit(
        h0=h0,
        H=H,
        input_center=center,
        input_scale=safe_scale,
        singular_values=singular_values,
        rank=int(rank - 1 if rank > 0 else 0),
        condition_number=condition,
    )


def predict(fit: AffineFit, inputs: np.ndarray) -> np.ndarray:
    return fit.h0 + fit.H @ np.asarray(inputs, dtype=float)


def validation_metrics(
    fit: AffineFit,
    samples: list[Sample],
    n_terminal_pressures: int,
    pressure_floor: float,
    flow_floor: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pressure_abs_all: list[float] = []
    pressure_rel_all: list[float] = []
    flow_abs_all: list[float] = []
    flow_rel_all: list[float] = []
    for sample in samples:
        predicted = predict(fit, sample.inputs)
        error = predicted - sample.outputs
        p_error = error[:n_terminal_pressures]
        p_true = sample.outputs[:n_terminal_pressures]
        q_error = error[n_terminal_pressures:]
        q_true = sample.outputs[n_terminal_pressures:]
        p_rel = np.abs(p_error) / np.maximum(np.abs(p_true), pressure_floor)
        q_rel = np.abs(q_error) / np.maximum(np.abs(q_true), flow_floor)
        pressure_abs_all.extend(np.abs(p_error).tolist())
        pressure_rel_all.extend(p_rel.tolist())
        flow_abs_all.extend(np.abs(q_error).tolist())
        flow_rel_all.extend(q_rel.tolist())
        rows.append({
            "run": sample.run,
            "pressure_max_abs": float(np.max(np.abs(p_error))),
            "pressure_rms_abs": float(np.sqrt(np.mean(p_error**2))),
            "pressure_max_rel": float(np.max(p_rel)),
            "concave_flow_max_abs": float(np.max(np.abs(q_error))),
            "concave_flow_max_rel": float(np.max(q_rel)),
        })
    summary = {
        "sample_count": len(samples),
        "pressure_max_abs": float(max(pressure_abs_all, default=0.0)),
        "pressure_rms_abs": float(
            math.sqrt(float(np.mean(np.square(pressure_abs_all))))
            if pressure_abs_all else 0.0
        ),
        "pressure_max_rel": float(max(pressure_rel_all, default=0.0)),
        "concave_flow_max_abs": float(max(flow_abs_all, default=0.0)),
        "concave_flow_max_rel": float(max(flow_rel_all, default=0.0)),
        "pressure_relative_floor": float(pressure_floor),
        "concave_flow_relative_floor": float(flow_floor),
    }
    return summary, rows


def _fit_to_arrays(fit: AffineFit) -> dict[str, np.ndarray]:
    return {
        "h0": fit.h0,
        "H": fit.H,
        "input_center": fit.input_center,
        "input_scale": fit.input_scale,
        "singular_values": fit.singular_values,
        "identified_rank": np.asarray([fit.rank], dtype=np.int64),
        "condition_number": np.asarray([fit.condition_number], dtype=float),
    }


def build(args: argparse.Namespace) -> None:
    root = Path(args.results_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[load] Reading completed Darcy outputs from {root}", flush=True)
    samples, layout, skipped = load_samples(root, args.organoid)
    print(
        f"[load] Found {len(samples)} samples, {len(layout['input_names'])} inputs, "
        f"and {len(layout['output_names'])} outputs",
        flush=True,
    )
    holdout_count = min(max(int(args.holdout_count), 0), max(len(samples) - 2, 0))
    train = samples[:-holdout_count] if holdout_count else samples
    holdout = samples[-holdout_count:] if holdout_count else []

    print(f"[fit] Fitting holdout model on {len(train)} samples", flush=True)
    holdout_fit = fit_affine(train, args.rcond)
    validation, validation_rows = validation_metrics(
        holdout_fit,
        holdout,
        layout["n_terminal_pressures"],
        args.pressure_relative_floor,
        args.flow_relative_floor,
    )
    print(
        f"[fit] Identified rank {holdout_fit.rank}/{len(layout['input_names'])}; "
        f"design condition number {holdout_fit.condition_number:.3e}",
        flush=True,
    )
    if holdout:
        print(
            f"[validate] Runs {holdout[0].run}-{holdout[-1].run}: "
            f"pressure max abs={validation['pressure_max_abs']:.6g}, "
            f"pressure RMS={validation['pressure_rms_abs']:.6g}, "
            f"concave-flow max abs={validation['concave_flow_max_abs']:.6g}",
            flush=True,
        )

    print(f"[fit] Refitting saved map on all {len(samples)} samples", flush=True)
    final_fit = fit_affine(samples, args.rcond)
    np.savez_compressed(output, **_fit_to_arrays(final_fit))
    metadata = {
        "format": "darcy-affine-response-v1",
        "construction": "history_fitted_subspace",
        "results_root": str(root),
        "organoid": int(args.organoid),
        "training_runs": [sample.run for sample in samples],
        "input_dimension": len(layout["input_names"]),
        "output_dimension": len(layout["output_names"]),
        "identified_rank": int(final_fit.rank),
        "globally_identified": bool(final_fit.rank >= len(layout["input_names"])),
        "condition_number": float(final_fit.condition_number),
        "input_names": layout["input_names"],
        "output_names": layout["output_names"],
        "layout": {key: value for key, value in layout.items() if key not in {"input_names", "output_names"}},
        "holdout_runs": [sample.run for sample in holdout],
        "holdout_validation": validation,
        "holdout_validation_by_run": validation_rows,
        "skipped_runs": skipped,
        "warning": (
            "This history-fitted map is validated only on the sampled coupling subspace. "
            "Build a one-hot PDE basis before using arbitrary independent branch perturbations."
        ),
    }
    metadata_path = output.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)
        fp.write("\n")
    print(f"[write] Response arrays: {output}", flush=True)
    print(f"[write] Metadata/validation: {metadata_path}", flush=True)


def validate_saved(args: argparse.Namespace) -> None:
    model_path = Path(args.model).expanduser().resolve()
    metadata = _load_json(model_path.with_suffix(".json"))
    arrays = np.load(model_path)
    fit = AffineFit(
        h0=arrays["h0"],
        H=arrays["H"],
        input_center=arrays["input_center"],
        input_scale=arrays["input_scale"],
        singular_values=arrays["singular_values"],
        rank=int(arrays["identified_rank"][0]),
        condition_number=float(arrays["condition_number"][0]),
    )
    root = Path(args.results_root).expanduser().resolve()
    samples, layout, _ = load_samples(root, args.organoid)
    if layout["input_names"] != metadata["input_names"]:
        raise RuntimeError("Validation input layout does not match the saved model")
    selected = [sample for sample in samples if not args.runs or sample.run in args.runs]
    summary, rows = validation_metrics(
        fit,
        selected,
        layout["n_terminal_pressures"],
        args.pressure_relative_floor,
        args.flow_relative_floor,
    )
    print(json.dumps({"summary": summary, "runs": rows}, indent=2))


def _root_coordinate(branching_file: Path) -> list[float]:
    with branching_file.open("r", encoding="utf-8-sig", newline="") as fp:
        row = next(csv.DictReader(fp))
    return [float(row[f"proximalCoords{axis}"]) for axis in "XYZ"]


def build_exact(args: argparse.Namespace) -> None:
    """Launch one mixed-Darcy assembly and build every independent basis column."""
    baseline_run = Path(args.baseline_run_dir).expanduser().resolve()
    organoid_dir = baseline_run / f"organoid_{int(args.organoid)}"
    if baseline_run.name.startswith("organoid_"):
        organoid_dir = baseline_run
    interface_file = organoid_dir / "out_darcy" / "interface_bc.json"
    interface = _load_json(interface_file)
    geometry = organoid_dir / "geometry"
    branching_in = organoid_dir / "branchingData_0.csv"
    branching_out = organoid_dir / "branchingData_1.csv"
    permeability = interface.get("permeability", {})
    perm_region_path = str(args.perm_region_path or permeability.get("requested_path", ""))
    perm_low = float(args.perm_low if args.perm_low is not None else permeability.get("perm_low", 1.0e-8))
    perm_high = float(args.perm_high if args.perm_high is not None else permeability.get("perm_high", 1.0e-6))
    perm_transition_width = float(
        args.perm_transition_width
        if args.perm_transition_width is not None
        else permeability.get("perm_transition_width", 0.01)
    )
    anisotropy_factor = float(
        args.stl_anisotropy_factor
        if args.stl_anisotropy_factor is not None
        else permeability.get("anisotropy_factor_inside_stl", 5.0)
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else output.parent / f"exact_build_organoid_{int(args.organoid)}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    required = [
        interface_file,
        geometry / "bioreactor.xdmf",
        geometry / "mesh_tags.xdmf",
        geometry / "tagged_branches_inlet.bp",
        geometry / "tagged_branches_outlet.bp",
        geometry / "pressure_checkpoint_inlet.bp",
        geometry / "pressure_checkpoint_outlet.bp",
        geometry / "flow_checkpoint_inlet.bp",
        geometry / "flow_checkpoint_outlet.bp",
        geometry / "area_checkpoint_inlet.bp",
        geometry / "area_checkpoint_outlet.bp",
        branching_in,
        branching_out,
        organoid_dir / "0D_Input_Files" / "inlet" / "output.csv",
        organoid_dir / "0D_Input_Files" / "outlet" / "output.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing exact-build inputs:\n  " + "\n  ".join(missing))

    profile = str(interface.get("concave_pressure_profile", "constant"))
    axis = str(interface.get("concave_pressure_profile_axis", "y"))
    command = [
        str(Path(args.python).expanduser()),
        str(Path(args.darcy_script).expanduser().resolve()),
        "--bioreactor-domain", str(geometry / "bioreactor.xdmf"),
        "--facet-file", str(geometry / "mesh_tags.xdmf"),
        "--mesh-inlet-file", str(geometry / "tagged_branches_inlet.bp"),
        "--mesh-outlet-file", str(geometry / "tagged_branches_outlet.bp"),
        "--pres-inlet-file", str(geometry / "pressure_checkpoint_inlet.bp"),
        "--pres-outlet-file", str(geometry / "pressure_checkpoint_outlet.bp"),
        "--flow-inlet-file", str(geometry / "flow_checkpoint_inlet.bp"),
        "--flow-outlet-file", str(geometry / "flow_checkpoint_outlet.bp"),
        "--area-inlet-file", str(geometry / "area_checkpoint_inlet.bp"),
        "--area-outlet-file", str(geometry / "area_checkpoint_outlet.bp"),
        "--branching-in-file", str(branching_in),
        "--branching-out-file", str(branching_out),
        "--inlet-output-csv", str(organoid_dir / "0D_Input_Files" / "inlet" / "output.csv"),
        "--outlet-output-csv", str(organoid_dir / "0D_Input_Files" / "outlet" / "output.csv"),
        "--coords-inlet", *[str(value) for value in _root_coordinate(branching_in)],
        "--coords-outlet", *[str(value) for value in _root_coordinate(branching_out)],
        "--concave-bc-mode", str(args.concave_bc_mode),
        "--concave-pressure-profile", profile,
        "--concave-pressure-profile-axis", axis,
        "--concave-inlet-pressure", str(interface["p_concave_inlet_bc"]),
        "--concave-outlet-pressure", str(interface["p_concave_outlet_bc"]),
        "--perm-region-path", perm_region_path,
        "--perm-low", str(perm_low),
        "--perm-high", str(perm_high),
        "--perm-transition-width", str(perm_transition_width),
        "--stl-anisotropy-factor", str(anisotropy_factor),
        "--out-dir", str(work_dir),
        "--output-mode", "minimal",
        "--diagnostics-mode", "minimal",
        "--response-map-output", str(output),
    ]
    if profile == "linear":
        command.extend([
            "--arterial-pressure-profile-low", str(interface["p_concave_inlet_bc_low"]),
            "--arterial-pressure-profile-high", str(interface["p_concave_inlet_bc_high"]),
            "--venous-pressure-profile-low", str(interface["p_concave_outlet_bc_low"]),
            "--venous-pressure-profile-high", str(interface["p_concave_outlet_bc_high"]),
        ])
    if args.concave_bc_mode == "robin":
        command.extend([
            "--lp-arterial", str(args.lp_arterial),
            "--lp-venous", str(args.lp_venous),
        ])

    print(f"[exact] Baseline: {organoid_dir}", flush=True)
    print(f"[exact] Work directory: {work_dir}", flush=True)
    print(f"[exact] Output map: {output}", flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="Fit H and run a chronological holdout validation")
    build_parser.add_argument("--results-root", required=True)
    build_parser.add_argument("--organoid", type=int, default=1)
    build_parser.add_argument("--output", required=True, help="Output .npz path")
    build_parser.add_argument("--holdout-count", type=int, default=8)
    build_parser.add_argument("--rcond", type=float, default=1.0e-12)
    build_parser.add_argument("--pressure-relative-floor", type=float, default=1.0)
    build_parser.add_argument("--flow-relative-floor", type=float, default=1.0e-10)
    build_parser.set_defaults(func=build)

    validate_parser = sub.add_parser("validate", help="Validate a saved map against completed runs")
    validate_parser.add_argument("--model", required=True)
    validate_parser.add_argument("--results-root", required=True)
    validate_parser.add_argument("--organoid", type=int, default=1)
    validate_parser.add_argument("--runs", type=int, nargs="*", default=[])
    validate_parser.add_argument("--pressure-relative-floor", type=float, default=1.0)
    validate_parser.add_argument("--flow-relative-floor", type=float, default=1.0e-10)
    validate_parser.set_defaults(func=validate_saved)

    exact_parser = sub.add_parser(
        "exact",
        help="Build globally exact H by solving every independent PDE input basis",
    )
    exact_parser.add_argument("--baseline-run-dir", required=True)
    exact_parser.add_argument("--organoid", type=int, default=1)
    exact_parser.add_argument("--output", required=True, help="Output .npz path")
    exact_parser.add_argument("--work-dir", default="")
    exact_parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable containing dolfinx/PETSc",
    )
    exact_parser.add_argument(
        "--darcy-script",
        default=str(Path(__file__).with_name("darcy_mixed.py")),
    )
    exact_parser.add_argument(
        "--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet"
    )
    exact_parser.add_argument("--lp-arterial", type=float, default=0.0)
    exact_parser.add_argument("--lp-venous", type=float, default=0.0)
    exact_parser.add_argument("--perm-region-path", default="")
    exact_parser.add_argument("--perm-low", type=float, default=None)
    exact_parser.add_argument("--perm-high", type=float, default=None)
    exact_parser.add_argument("--perm-transition-width", type=float, default=None)
    exact_parser.add_argument("--stl-anisotropy-factor", type=float, default=None)
    exact_parser.set_defaults(func=build_exact)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    parsed.func(parsed)
