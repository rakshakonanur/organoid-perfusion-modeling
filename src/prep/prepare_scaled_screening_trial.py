#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1]
if str(REPO_SRC / "prep") not in sys.path:
    sys.path.insert(0, str(REPO_SRC / "prep"))
if str(REPO_SRC / "solves") not in sys.path:
    sys.path.insert(0, str(REPO_SRC / "solves"))

import run_all
import compare_scaled_organoid_fields as scaled_fields


def _run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        shutil.copy2(src, dst)


def _copy_trial_metadata(trial_dir: Path, prepared_root: Path) -> None:
    for name in ("combined.in", "rename_map.json", "output.csv"):
        _copy_if_exists(trial_dir / name, prepared_root / name)
    _copy_if_exists(trial_dir / "split", prepared_root / "split")


def _copy_trial_organoid_inputs(trial_dir: Path, org_id: int, prepared_root: Path) -> Path:
    src_org = trial_dir / f"organoid_{org_id}"
    dst_org = prepared_root / f"organoid_{org_id}"
    dst_org.mkdir(parents=True, exist_ok=True)

    _copy_if_exists(src_org / "0D_Input_Files", dst_org / "0D_Input_Files")
    for name in ("branchingData_0.csv", "branchingData_1.csv"):
        _copy_if_exists(src_org / name, dst_org / name)
    for name in ("tissue_domain_volume_refined.vtu", "tissue_domain_volume.vtu", "well_surface.stl"):
        _copy_if_exists(src_org / name, dst_org / name)
    return dst_org


def _copy_well_geometry_from_tmp_mesh(
    tmp_mesh_dir: Path,
    dst_geom: Path,
    well_idx: int,
    copy_1d_artifacts: bool = True,
) -> None:
    dst_geom.mkdir(parents=True, exist_ok=True)
    run_all._copy_shared_geometry_outputs(
        tmp_mesh_dir,
        dst_geom,
        well_idx=well_idx,
        copy_1d_artifacts=copy_1d_artifacts,
    )

    for stem in ("branched_network_inlet", "branched_network_outlet"):
        src = tmp_mesh_dir / f"{stem}_well{well_idx}.xdmf"
        if src.exists():
            run_all._copy_xdmf_with_h5_sidecars(src, dst_geom / f"{stem}.xdmf")


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
    if arr.size == 0:
        return [] if isinstance(value, list) else value
    return (arr + np.asarray(shift, dtype=float)).tolist()


def _build_scaled_interface_bc(
    ref_iface: dict[str, Any],
    target_zero_d: dict[str, Any],
    pressure_transform: dict[str, float],
    scale_factor: float,
    shift: tuple[float, float, float],
) -> dict[str, Any]:
    iface = copy.deepcopy(ref_iface)
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
        "p_inlet_target",
        "p_outlet_target",
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

    iface["q_artery_leak"] = float(target_zero_d["q_art_mean"])
    iface["q_venous_leak"] = float(target_zero_d["q_ven_mean"])
    iface["p_concave_inlet_bc"] = float(pressure_transform["target_arterial_pressure"])
    iface["p_concave_outlet_bc"] = float(pressure_transform["target_venous_pressure"])
    iface["concave_pressure_profile"] = "scaled_from_reference"
    iface["concave_pressure_profile_axis"] = "y"
    iface["scaled_from_reference_organoid"] = 1
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
    prepared_root: Path,
    organoid_id: int,
    ref_iface: dict[str, Any],
    ref_p: dict[str, Any],
    ref_u: dict[str, Any],
    target_zero_d: dict[str, Any],
    pressure_transform: dict[str, float],
    y_shift_step: float,
    h5py: Any,
) -> dict[str, Any]:
    scale_factor = float(pressure_transform["scale_factor"])
    pressure_offset = float(pressure_transform["pressure_offset"])
    shift = scaled_fields._shift_for_organoid(1, organoid_id, y_shift_step)

    out_dir = prepared_root / f"organoid_{organoid_id}" / "out_darcy"
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


def prepare_scaled_screening_trial(args: argparse.Namespace) -> dict[str, Any]:
    use_no_synth = bool(getattr(args, "no_synthetic_vasculature", False))
    source_folder = None
    if not use_no_synth:
        if args.source_folder is None:
            raise ValueError("--source-folder is required unless --no-synthetic-vasculature is enabled.")
        source_folder = args.source_folder.expanduser().resolve()
        if not source_folder.exists():
            raise FileNotFoundError(f"Missing source folder: {source_folder}")

    trial_root = args.trial_root.expanduser().resolve()
    prepared_root_parent = args.prepared_root.expanduser().resolve()
    default_trial_name = "coupled-no-vasc-scaled" if use_no_synth else source_folder.name
    trial_name = args.trial_name.strip() if args.trial_name else default_trial_name
    prepared_name = args.prepared_name.strip() if args.prepared_name else trial_name
    trial_dir = trial_root / trial_name
    prepared_root = prepared_root_parent / prepared_name

    build_script = args.build_combined_script.expanduser().resolve()
    run_and_split_script = args.run_and_split_script.expanduser().resolve()

    build_cmd = [
        sys.executable,
        str(build_script),
        "--trial-root",
        str(trial_root),
        "--trial-name",
        trial_name,
        "--channel-xlsx",
        str(args.channel_xlsx.expanduser().resolve()),
        "--out",
        "combined.in",
        "--map",
        "rename_map.json",
        "--n-organoids",
        str(int(args.n_organoids)),
        "--dy-step",
        str(float(args.dy_step)),
        "--shift-sign",
        str(float(args.shift_sign)),
    ]
    if use_no_synth:
        build_cmd.append("--no-synthetic-vasculature")
    else:
        build_cmd.extend(["--source-folder", str(source_folder)])
    if args.overwrite:
        build_cmd.append("--overwrite-trial")
    if args.no_shift_coords:
        build_cmd.append("--no-shift-coords")
    _run_cmd(build_cmd)

    split_cmd = [
        sys.executable,
        str(run_and_split_script),
        "--exe",
        str(args.svzerodsolver),
        "--input",
        str(trial_dir / "combined.in"),
        "--outdir",
        str(trial_dir),
        "--output",
        "output.csv",
        "--organoid-root",
        str(trial_dir),
        "--no-plot",
    ]
    _run_cmd(split_cmd)

    if prepared_root.exists():
        if args.overwrite:
            shutil.rmtree(prepared_root)
        else:
            raise FileExistsError(f"Prepared output already exists: {prepared_root}")
    prepared_root.mkdir(parents=True, exist_ok=True)
    _copy_trial_metadata(trial_dir, prepared_root)

    coords_inlet_base = np.asarray(args.coords_inlet, dtype=float)
    coords_outlet_base = np.asarray(args.coords_outlet, dtype=float)
    inferred_well_spacing_y = run_all._infer_well_spacing_y_from_trial(
        trial_dir=trial_dir,
        n_wells=int(args.n_organoids),
        fallback=-float(args.dy_step),
    )
    shared_tissue_vtu = run_all._pick_existing_or_none(
        trial_dir / "organoid_1" / "tissue_domain_volume_refined.vtu",
        trial_dir / "organoid_1" / "tissue_domain_volume.vtu",
    )
    tmp_mesh_dir = prepared_root / "_tmp_mesh"

    run_all.run_one(
        i=1,
        stl_dir=args.stl_dir.expanduser().resolve(),
        trial_dir=trial_dir,
        coupled_root=prepared_root,
        coords_inlet_base=coords_inlet_base,
        coords_outlet_base=coords_outlet_base,
        dy_step=float(args.dy_step),
        well_spacing_y=float(inferred_well_spacing_y),
        n_wells=int(args.n_organoids),
        same_meshtags_as_first=bool(args.same_meshtags_as_first),
        concave_bc_mode=str(args.concave_bc_mode),
        lp_arterial=float(args.lp_arterial),
        lp_venous=float(args.lp_venous),
        no_synthetic_vasculature=use_no_synth,
        fallback_inlet_pressure=float(args.fallback_inlet_pressure),
        fallback_outlet_pressure=float(args.fallback_outlet_pressure),
        inlet_flux_correction=bool(args.inlet_flux_correction),
        inlet_flux_corr_max_iter=int(args.inlet_flux_corr_max_iter),
        inlet_flux_corr_relax=float(args.inlet_flux_corr_relax),
        inlet_flux_corr_tol=float(args.inlet_flux_corr_tol),
        shared_tissue_vtu=shared_tissue_vtu,
        tmp_mesh_dir=tmp_mesh_dir,
        reuse_mesh_from_first=True,
        use_mesh_multiwell_translation=True,
        darcy_mpi_procs=int(args.darcy_mpi_procs),
        darcy_mpirun_cmd=str(args.darcy_mpirun_cmd),
        darcy_solver=str(args.darcy_solver),
        perm_region_root=str(args.perm_region_root),
        perm_low=float(args.perm_low),
        perm_high=float(args.perm_high),
        perm_transition_width=float(args.perm_transition_width),
        first_organoid_linear_profile=bool(args.first_organoid_linear_profile),
        concave_profile_face_length=float(args.concave_profile_face_length),
        concave_profile_compartment_spacing=float(args.concave_profile_compartment_spacing),
    )

    for organoid_id in range(1, int(args.n_organoids) + 1):
        dst_org = _copy_trial_organoid_inputs(trial_dir, organoid_id, prepared_root)
        if organoid_id >= 2:
            _copy_well_geometry_from_tmp_mesh(
                tmp_mesh_dir,
                dst_org / "geometry",
                organoid_id,
                copy_1d_artifacts=(not use_no_synth),
            )

    np_mod = scaled_fields._import_numpy()
    h5py = scaled_fields._import_h5py()
    grouped = scaled_fields._load_0d_groups(trial_dir, np_mod)
    zero_d = {
        organoid_id: scaled_fields._load_0d_organoid_summary(grouped, organoid_id, np_mod)
        for organoid_id in range(1, int(args.n_organoids) + 1)
    }
    ref_zero_d = zero_d[1]
    ref_out = prepared_root / "organoid_1" / "out_darcy"
    ref_iface = json.loads((ref_out / "interface_bc.json").read_text())
    ref_p = scaled_fields._load_xdmf_field(ref_out / "p.xdmf", np_mod, h5py)
    ref_u = scaled_fields._load_xdmf_field(ref_out / "u.xdmf", np_mod, h5py)

    scaling_rows: list[dict[str, Any]] = []
    for organoid_id in range(2, int(args.n_organoids) + 1):
        transform = scaled_fields._pressure_transform_from_0d(
            ref_zero_d,
            zero_d[organoid_id],
            args.pressure_offset_anchor,
        )
        scaling_rows.append(
            _write_scaled_organoid_outputs(
                prepared_root=prepared_root,
                organoid_id=organoid_id,
                ref_iface=ref_iface,
                ref_p=ref_p,
                ref_u=ref_u,
                target_zero_d=zero_d[organoid_id],
                pressure_transform=transform,
                y_shift_step=float(args.y_shift_step),
                h5py=h5py,
            )
        )

    summary = {
        "source_folder": str(source_folder) if source_folder is not None else "",
        "trial_dir": str(trial_dir),
        "prepared_root": str(prepared_root),
        "reference_organoid": 1,
        "no_synthetic_vasculature": use_no_synth,
        "scaled_organoids": scaling_rows,
        "assumptions": [
            (
                "No synthetic vasculature is used; only organoid_1 is solved with Darcy and "
                "wells 2..N reuse organoid_1 fields with y-translation and 0D-based scaling."
                if use_no_synth
                else "The same synthetic vasculature is replicated into each organoid well."
            ),
            "Only organoid_1 is solved with Darcy; wells 2..N reuse organoid_1 fields with y-translation and 0D-based scaling.",
            "Scaled wells write p.xdmf, u.xdmf, and interface_bc.json but do not run an independent Darcy solve.",
        ],
    }
    (prepared_root / "scaled_screening_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ok] Prepared scaled screening trial at {prepared_root}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one repeated-geometry trial, run the full Darcy solve only for "
            "organoid_1, and synthesize wells 2..N using the 0D-based scaling rules "
            "from compare_scaled_organoid_fields.py."
        )
    )
    parser.add_argument("--source-folder", type=Path, default=None)
    parser.add_argument("--trial-root", type=Path, default=REPO_SRC / "prep" / "prepped")
    parser.add_argument("--trial-name", default="")
    parser.add_argument("--prepared-root", type=Path, default=REPO_SRC / "prep" / "scaled-screening")
    parser.add_argument("--prepared-name", default="081826")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--build-combined-script", type=Path, default=REPO_SRC / "prep" / "build_combined_0d_v3.py")
    parser.add_argument("--run-and-split-script", type=Path, default=REPO_SRC / "prep" / "run_and_split_svzerod.py")
    parser.add_argument("--channel-xlsx", type=Path, default=REPO_SRC.parent / "files" / "excel" / "channel-resistance-sv.xlsx")
    parser.add_argument("--svzerodsolver", default="svzerodsolver")
    parser.add_argument("--n-organoids", type=int, default=4)
    parser.add_argument("--dy-step", type=float, default=0.6)
    parser.add_argument("--shift-sign", type=float, default=-1.0)
    parser.add_argument("--no-shift-coords", action="store_true")
    parser.add_argument("--stl-dir", type=Path, default=REPO_SRC.parent / "files")
    parser.add_argument("--same-meshtags-as-first", action="store_true", default=True)
    parser.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet")
    parser.add_argument("--lp-arterial", type=float, default=1e-9)
    parser.add_argument("--lp-venous", type=float, default=1e-9)
    parser.add_argument("--fallback-inlet-pressure", type=float, default=0.0)
    parser.add_argument("--fallback-outlet-pressure", type=float, default=0.0)
    parser.add_argument("--inlet-flux-correction", action="store_true")
    parser.add_argument("--inlet-flux-corr-max-iter", type=int, default=5)
    parser.add_argument("--inlet-flux-corr-relax", type=float, default=0.5)
    parser.add_argument("--inlet-flux-corr-tol", type=float, default=0.05)
    parser.add_argument("--darcy-mpi-procs", type=int, default=1)
    parser.add_argument("--darcy-mpirun-cmd", default="mpirun")
    parser.add_argument("--darcy-solver", choices=["darcy", "darcy_mixed", "darcy_p1_lm", "darcy_p1_lm_interior"], default="darcy_mixed")
    parser.add_argument("--perm-region-root", default=str(REPO_SRC.parent / "files" / "stl" / "organoid-growth-domains" / "3000um"))
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
    parser.add_argument("--no-synthetic-vasculature", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepare_scaled_screening_trial(args)


if __name__ == "__main__":
    main()
