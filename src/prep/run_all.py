#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys
import numpy as np
import shutil
import os
import re
import importlib
import csv
from time import perf_counter

# Import your existing scripts as modules
MESH_DIR  = Path("/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry").resolve()
DARCY_DIR = Path("/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/solves").resolve()

sys.path.insert(0, str(MESH_DIR))
sys.path.insert(0, str(DARCY_DIR))

import mesh


def _load_solver_module_and_class(solver_name: str):
    mod = importlib.import_module(solver_name)
    solver_cls = getattr(mod, "PerfusionSolver")
    return mod, solver_cls

def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        # Copy contents into existing directory
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    else:
        shutil.copytree(src, dst)


def _pick_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of the candidate paths exist:\n" + "\n".join(str(p) for p in paths))


def _pick_existing_or_none(*paths: Path) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _run_cmd(cmd: list[str]) -> None:
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def _resolve_mpi_launcher(user_cmd: str) -> str:
    """
    Prefer launcher binaries from the active Python environment to avoid
    MPI launcher/runtime mismatches (e.g., Homebrew OpenMPI launcher with
    conda MPICH mpi4py runtime).
    """
    generic = {"mpirun", "mpiexec"}
    if user_cmd:
        base = Path(user_cmd).name
        if base not in generic:
            return user_cmd

    env_bin = Path(sys.executable).resolve().parent
    candidates = [
        env_bin / "mpiexec.hydra",
        env_bin / "mpiexec",
        env_bin / "mpirun",
    ]
    for cand in candidates:
        if cand.exists() and os.access(cand, os.X_OK):
            return str(cand)
    return user_cmd


def _copy_shared_geometry_outputs(
    src_geom: Path,
    dst_geom: Path,
    well_idx: int,
    copy_1d_artifacts: bool = True,
) -> None:
    """
    Copy only the geometry/checkpoint files needed by darcy.py.
    Maps translated well files to canonical names in dst:
      bioreactor_well{well_idx}.xdmf -> bioreactor.xdmf
      mesh_tags_well{well_idx}.xdmf  -> mesh_tags.xdmf
    """
    dst_geom.mkdir(parents=True, exist_ok=True)
    required = []
    optional = []
    if copy_1d_artifacts:
        required = [
            "tagged_branches_inlet.bp",
            "tagged_branches_outlet.bp",
            "pressure_checkpoint_inlet.bp",
            "pressure_checkpoint_outlet.bp",
            "flow_checkpoint_inlet.bp",
            "flow_checkpoint_outlet.bp",
        ]
        optional = [
            "area_checkpoint_inlet.bp",
            "area_checkpoint_outlet.bp",
        ]
    for name in required + optional:
        src_well = src_geom / name.replace(".bp", f"_well{well_idx}.bp")
        src_plain = src_geom / name
        src = src_well if src_well.exists() else src_plain
        if not src.exists():
            if name in required:
                raise FileNotFoundError(
                    f"Missing shared geometry artifact for well{well_idx}: "
                    f"{src_well} (or fallback {src_plain})"
                )
            continue
        dst = dst_geom / name
        if src.resolve() == dst.resolve():
            continue
        if src.is_dir():
            _copy_tree(src, dst)
        else:
            shutil.copy2(src, dst)

    src_bio = src_geom / f"bioreactor_well{well_idx}.xdmf"
    src_tag = src_geom / f"mesh_tags_well{well_idx}.xdmf"
    if not src_bio.exists() or not src_tag.exists():
        raise FileNotFoundError(
            f"Missing translated well files in {src_geom}: {src_bio.name}, {src_tag.name}"
        )
    _copy_xdmf_with_h5_sidecars(src_bio, dst_geom / "bioreactor.xdmf")
    _copy_xdmf_with_h5_sidecars(src_tag, dst_geom / "mesh_tags.xdmf")


def _copy_xdmf_with_h5_sidecars(src_xdmf: Path, dst_xdmf: Path) -> None:
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


def _build_trial_well_inputs(
    trial_dir: Path,
    n_wells: int,
    allow_missing_synthetic_inputs: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    in_vtp_dirs: list[str] = []
    out_vtp_dirs: list[str] = []
    branching_in_files: list[str] = []
    branching_out_files: list[str] = []
    for i in range(1, n_wells + 1):
        org = trial_dir / f"organoid_{i}"
        in_vtp = org / "0D_Input_Files" / "inlet"
        out_vtp = org / "0D_Input_Files" / "outlet"
        b_in = org / "branchingData_0.csv"
        b_out = org / "branchingData_1.csv"
        if not allow_missing_synthetic_inputs:
            for p in (in_vtp, out_vtp, b_in, b_out):
                if not p.exists():
                    raise FileNotFoundError(f"Missing required per-well input: {p}")
        in_vtp_dirs.append(str(in_vtp))
        out_vtp_dirs.append(str(out_vtp))
        branching_in_files.append(str(b_in))
        branching_out_files.append(str(b_out))
    return in_vtp_dirs, out_vtp_dirs, branching_in_files, branching_out_files


def _infer_well_spacing_y_from_trial(trial_dir: Path, n_wells: int, fallback: float) -> float:
    """
    Infer Y-translation between wells from branchingData_0.csv proximalCoordsY.
    Uses organoid_1 -> organoid_2 delta when available.
    """
    if n_wells < 2:
        return float(fallback)
    try:
        vals: list[float] = []
        for i in (1, 2):
            csv_path = trial_dir / f"organoid_{i}" / "branchingData_0.csv"
            with csv_path.open("r", newline="") as f:
                row = next(csv.DictReader(f))
            vals.append(float(row["proximalCoordsY"]))
        return float(vals[1] - vals[0])
    except Exception:
        return float(fallback)


def _read_seed_coords_from_branching_csv(csv_path: Path, fallback: np.ndarray) -> np.ndarray:
    """
    Read proximal seed coordinates from branchingData CSV.
    Falls back to provided coordinates if columns are unavailable.
    """
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


def _shift_seed_fallback(base: np.ndarray, organoid_idx: int, delta_y: float) -> np.ndarray:
    coords = np.asarray(base, dtype=float).copy()
    coords[1] += float(delta_y) * float(organoid_idx - 1)
    return coords


def _latest_named_row(csv_path: Path, vessel_name: str) -> dict | None:
    if not csv_path.exists():
        return None

    latest_row: dict | None = None
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


def _read_pressure_column(row: dict | None, column: str) -> float | None:
    if row is None:
        return None
    raw = row.get(column, None)
    if raw in (None, ""):
        return None
    try:
        return float(raw)
    except Exception:
        return None


def _read_leak_pressures_from_output(output_csv: Path, organoid_idx: int) -> tuple[float | None, float | None]:
    art_row = _latest_named_row(output_csv, f"leak_art_{organoid_idx}")
    ven_row = _latest_named_row(output_csv, f"leak_ven_{organoid_idx}")
    return _read_pressure_column(art_row, "pressure_out"), _read_pressure_column(ven_row, "pressure_out")


def _resolve_perm_region_for_organoid(base: str, organoid_idx: int) -> str:
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

def run_one(
    i: int,
    stl_dir: Path,
    trial_dir: Path,
    coupled_root: Path,
    coords_inlet_base: np.ndarray,
    coords_outlet_base: np.ndarray,
    dy_step: float,
    well_spacing_y: float,
    n_wells: int,
    same_meshtags_as_first: bool = True,
    concave_bc_mode: str = "dirichlet",
    lp_arterial: float = 0.0,
    lp_venous: float = 0.0,
    no_synthetic_vasculature: bool = False,
    fallback_inlet_pressure: float = 0.0,
    fallback_outlet_pressure: float = 0.0,
    inlet_flux_correction: bool = False,
    inlet_flux_corr_max_iter: int = 5,
    inlet_flux_corr_relax: float = 0.5,
    inlet_flux_corr_tol: float = 0.05,
    shared_tissue_vtu: Path | None = None,
    tmp_mesh_dir: Path | None = None,
    reuse_mesh_from_first: bool = False,
    use_mesh_multiwell_translation: bool = False,
    darcy_mpi_procs: int = 1,
    darcy_mpirun_cmd: str = "mpirun",
    darcy_solver: str = "darcy",
    perm_region_root: str = "",
    perm_low: float = 1.0e-8,
    perm_high: float = 1.0e-6,
    perm_transition_width: float = 0.01,
) -> None:
    t0 = perf_counter()
    src_org_dir = trial_dir / f"organoid_{i}"
    org_dir = coupled_root / f"organoid_{i}"
    geom_dir = org_dir / "geometry"
    out_inlet = org_dir / "0D_Input_Files" / "inlet"
    out_outlet = org_dir / "0D_Input_Files" / "outlet"

    geom_dir.mkdir(parents=True, exist_ok=True)
    out_inlet.mkdir(parents=True, exist_ok=True)
    out_outlet.mkdir(parents=True, exist_ok=True)

    print(f"[organoid_{i}] Copying 0D input folders...", flush=True)
    # 0D solver output folders + branching data come from the prepared trial directory
    src_inlet_dir = src_org_dir / "0D_Input_Files" / "inlet"
    src_outlet_dir = src_org_dir / "0D_Input_Files" / "outlet"
    if src_inlet_dir.exists():
        _copy_tree(src_inlet_dir, out_inlet)
    elif not no_synthetic_vasculature:
        raise FileNotFoundError(f"Missing required inlet directory: {src_inlet_dir}")
    if src_outlet_dir.exists():
        _copy_tree(src_outlet_dir, out_outlet)
    elif not no_synthetic_vasculature:
        raise FileNotFoundError(f"Missing required outlet directory: {src_outlet_dir}")

    print(f"[organoid_{i}] Searching for branching data in {src_org_dir}", flush=True)
    branching_in = src_org_dir / "branchingData_0.csv"
    branching_out = src_org_dir / "branchingData_1.csv"
    if not branching_in.exists() and not no_synthetic_vasculature:
        raise FileNotFoundError(f"Missing: {branching_in}")
    if not branching_out.exists() and not no_synthetic_vasculature:
        raise FileNotFoundError(f"Missing: {branching_out}")

    if shared_tissue_vtu is not None:
        tissue_vtu = shared_tissue_vtu
    else:
        tissue_vtu = _pick_existing_or_none(
            src_org_dir / "tissue_domain_volume_refined.vtu",
            src_org_dir / "tissue_domain_volume.vtu",
        )

    stl_candidates = [
        src_org_dir / "well_surface.stl",
        src_org_dir / "3d_tmp" / "well_surface.stl",
        stl_dir / "stl" / "organoid-growth-domains" / "original" / f"organoid-{i}.stl",
        stl_dir / f"organoid-{i}.stl",
        stl_dir / f"organoid_{i}.stl",
    ]
    tissue_stl = _pick_existing_or_none(*stl_candidates)

    if tissue_vtu is None and tissue_stl is None:
        missing_vtu = [
            src_org_dir / "tissue_domain_volume_refined.vtu",
            src_org_dir / "tissue_domain_volume.vtu",
        ]
        raise FileNotFoundError(
            f"[organoid_{i}] Missing tissue geometry. Looked for VTU files:\n"
            + "\n".join(f"  {p}" for p in missing_vtu)
            + "\nLooked for STL files:\n"
            + "\n".join(f"  {p}" for p in stl_candidates)
        )

    if tissue_vtu is not None:
        print(f"[organoid_{i}] Using tissue VTU: {tissue_vtu}", flush=True)
    else:
        print(f"[organoid_{i}] No VTU found, meshing directly from STL: {tissue_stl}", flush=True)

    # Use per-organoid branching seeds so Darcy concave-wall BC sampling
    # stays aligned with each organoid's translated 1D data.
    coords_inlet = _read_seed_coords_from_branching_csv(
        branching_in,
        _shift_seed_fallback(coords_inlet_base, i, well_spacing_y),
    )
    coords_outlet = _read_seed_coords_from_branching_csv(
        branching_out,
        _shift_seed_fallback(coords_outlet_base, i, well_spacing_y),
    )

    # ---- Run geometry (mesh) ----
    # Preferred flow: run mesh.py once into tmp_mesh_dir, then copy well{i} artifacts.
    if tmp_mesh_dir is not None:
        tmp_mesh_dir.mkdir(parents=True, exist_ok=True)
        if i == 1:
            t_mesh = perf_counter()
            print(f"[organoid_{i}] Starting mesh.Files(...) into {tmp_mesh_dir}...", flush=True)
            in_vtp_dirs, out_vtp_dirs, branching_in_files, branching_out_files = _build_trial_well_inputs(
                trial_dir,
                n_wells,
                allow_missing_synthetic_inputs=no_synthetic_vasculature,
            )
            mesh.current_dir = tmp_mesh_dir
            mesh.Files(
                tissue_vtu=str(tissue_vtu) if tissue_vtu is not None else None,
                tissue_stl=str(tissue_stl) if tissue_stl is not None else None,
                output_1d_inlet=str(out_inlet),
                output_1d_outlet=str(out_outlet),
                branching_data_inlet=str(branching_in),
                branching_data_outlet=str(branching_out),
                output_1d_inlet_list=in_vtp_dirs,
                output_1d_outlet_list=out_vtp_dirs,
                branching_data_inlet_list=branching_in_files,
                branching_data_outlet_list=branching_out_files,
                theta_concave_inlet_deg=60,
                theta_concave_outlet_deg=38.5,
                geo_file_inlet=str(tmp_mesh_dir / "branched_network_inlet.geo"),
                msh_file_inlet=str(tmp_mesh_dir / "branched_network_inlet.msh"),
                xdmf_file_inlet=str(tmp_mesh_dir / "branched_network_inlet.xdmf"),
                geo_file_outlet=str(tmp_mesh_dir / "branched_network_outlet.geo"),
                msh_file_outlet=str(tmp_mesh_dir / "branched_network_outlet.msh"),
                xdmf_file_outlet=str(tmp_mesh_dir / "branched_network_outlet.xdmf"),
                coords_inlet=coords_inlet,
                coords_outlet=coords_outlet,
                multiwell=True,
                same_tissue_for_all_wells=True,
                same_meshtags_for_all_wells=same_meshtags_as_first,
                well_spacing_y=well_spacing_y,
                disable_1d_terminals=no_synthetic_vasculature,
            )
            print(f"[organoid_{i}] mesh.Files(...) done in {perf_counter() - t_mesh:.1f}s", flush=True)

        src_geom = tmp_mesh_dir
        if not src_geom.exists():
            raise FileNotFoundError(f"Cannot reuse mesh: missing {src_geom}")
        print(f"[organoid_{i}] Copying well{i} mesh artifacts from {src_geom}", flush=True)
        _copy_shared_geometry_outputs(
            src_geom,
            geom_dir,
            well_idx=i,
            copy_1d_artifacts=(not no_synthetic_vasculature),
        )
    elif reuse_mesh_from_first and i > 1:
        src_geom = coupled_root / "organoid_1" / "geometry"
        if not src_geom.exists():
            raise FileNotFoundError(f"Cannot reuse mesh: missing {src_geom}")
        if use_mesh_multiwell_translation:
            print(f"[organoid_{i}] Reusing translated mesh outputs from organoid_1/well{i}", flush=True)
            _copy_shared_geometry_outputs(
                src_geom,
                geom_dir,
                well_idx=i,
                copy_1d_artifacts=(not no_synthetic_vasculature),
            )
        else:
            print(f"[organoid_{i}] Reusing mesh outputs from organoid_1", flush=True)
            _copy_tree(src_geom, geom_dir)
    else:
        t_mesh = perf_counter()
        print(f"[organoid_{i}] Starting mesh.Files(...)...", flush=True)
        mesh.current_dir = geom_dir
        mesh.Files(
            tissue_vtu=str(tissue_vtu) if tissue_vtu is not None else None,
            tissue_stl=str(tissue_stl) if tissue_stl is not None else None,
            output_1d_inlet=str(out_inlet),
            output_1d_outlet=str(out_outlet),
            branching_data_inlet=str(branching_in),
            branching_data_outlet=str(branching_out),
            geo_file_inlet=str(geom_dir / "branched_network_inlet.geo"),
            msh_file_inlet=str(geom_dir / "branched_network_inlet.msh"),
            xdmf_file_inlet=str(geom_dir / "branched_network_inlet.xdmf"),
            geo_file_outlet=str(geom_dir / "branched_network_outlet.geo"),
            msh_file_outlet=str(geom_dir / "branched_network_outlet.msh"),
            xdmf_file_outlet=str(geom_dir / "branched_network_outlet.xdmf"),
            coords_inlet=coords_inlet,
            coords_outlet=coords_outlet,
            multiwell=use_mesh_multiwell_translation,
            same_tissue_for_all_wells=use_mesh_multiwell_translation,
            same_meshtags_for_all_wells=same_meshtags_as_first,
            well_spacing_y=well_spacing_y,
            disable_1d_terminals=no_synthetic_vasculature,
        )
        # If mesh.py emitted well-suffixed outputs, normalize well1 to canonical filenames.
        if (geom_dir / "bioreactor_well1.xdmf").exists() and (geom_dir / "mesh_tags_well1.xdmf").exists():
            _copy_shared_geometry_outputs(
                geom_dir,
                geom_dir,
                well_idx=1,
                copy_1d_artifacts=(not no_synthetic_vasculature),
            )
        print(f"[organoid_{i}] mesh.Files(...) done in {perf_counter() - t_mesh:.1f}s", flush=True)

    # If mesh.Files triggers everything in __init__, we’re done.
    # If it requires an explicit call like perfusion.run(), you can add it here.
    # e.g., if hasattr(perfusion, "run"): perfusion.run()

    # ---- Run Darcy ----
    bioreactor_domain = str(geom_dir / "bioreactor.xdmf")
    facet_file = str(geom_dir / "mesh_tags.xdmf")
    mesh_inlet_file   = str(geom_dir / "tagged_branches_inlet.bp")
    mesh_outlet_file  = str(geom_dir / "tagged_branches_outlet.bp")
    pres_inlet_file   = str(geom_dir / "pressure_checkpoint_inlet.bp")
    pres_outlet_file  = str(geom_dir / "pressure_checkpoint_outlet.bp")
    flow_inlet_file   = str(geom_dir / "flow_checkpoint_inlet.bp")
    flow_outlet_file  = str(geom_dir / "flow_checkpoint_outlet.bp")
    area_inlet_file   = str(geom_dir / "area_checkpoint_inlet.bp")
    area_outlet_file  = str(geom_dir / "area_checkpoint_outlet.bp")

    print(f"[organoid_{i}] Running Darcy solve...", flush=True)
    t_darcy = perf_counter()
    solver_script = DARCY_DIR / f"{darcy_solver}.py"
    if not solver_script.exists():
        raise FileNotFoundError(f"Requested Darcy solver script not found: {solver_script}")
    perm_region_path = _resolve_perm_region_for_organoid(perm_region_root, i) if perm_region_root else ""
    resolved_fallback_inlet_pressure = float(fallback_inlet_pressure)
    resolved_fallback_outlet_pressure = float(fallback_outlet_pressure)
    if no_synthetic_vasculature:
        trial_output_csv = trial_dir / "output.csv"
        p_art, p_ven = _read_leak_pressures_from_output(trial_output_csv, i)
        if p_art is not None:
            resolved_fallback_inlet_pressure = float(p_art)
        if p_ven is not None:
            resolved_fallback_outlet_pressure = float(p_ven)
    if darcy_mpi_procs > 1:
        mpi_launcher = _resolve_mpi_launcher(darcy_mpirun_cmd)
        cmd = [
            mpi_launcher, "-n", str(darcy_mpi_procs),
            sys.executable, str(solver_script),
            "--bioreactor-domain", bioreactor_domain,
            "--facet-file", facet_file,
            "--mesh-inlet-file", mesh_inlet_file,
            "--mesh-outlet-file", mesh_outlet_file,
            "--pres-inlet-file", pres_inlet_file,
            "--pres-outlet-file", pres_outlet_file,
            "--flow-inlet-file", flow_inlet_file,
            "--flow-outlet-file", flow_outlet_file,
            "--area-inlet-file", area_inlet_file,
            "--area-outlet-file", area_outlet_file,
            "--coords-inlet", str(coords_inlet[0]), str(coords_inlet[1]), str(coords_inlet[2]),
            "--coords-outlet", str(coords_outlet[0]), str(coords_outlet[1]), str(coords_outlet[2]),
            "--branching-in-file", str(branching_in),
            "--branching-out-file", str(branching_out),
            "--concave-bc-mode", str(concave_bc_mode),
            "--lp-arterial", str(lp_arterial),
            "--lp-venous", str(lp_venous),
            *(["--skip-1d"] if no_synthetic_vasculature else []),
            "--fallback-inlet-pressure", str(resolved_fallback_inlet_pressure),
            "--fallback-outlet-pressure", str(resolved_fallback_outlet_pressure),
            *(["--inlet-flux-correction"] if inlet_flux_correction else []),
            "--inlet-flux-corr-max-iter", str(inlet_flux_corr_max_iter),
            "--inlet-flux-corr-relax", str(inlet_flux_corr_relax),
            "--inlet-flux-corr-tol", str(inlet_flux_corr_tol),
            "--out-dir", str(org_dir),
        ]
        if darcy_solver in {"darcy_p1_lm_interior", "darcy_mixed"} and perm_region_path:
            cmd.extend([
                "--perm-region-path", perm_region_path,
                "--perm-low", str(perm_low),
                "--perm-high", str(perm_high),
                "--perm-transition-width", str(perm_transition_width),
            ])
        try:
            _run_cmd(cmd)
        except subprocess.CalledProcessError as e:
            msg = str(e)
            if ("PMI_Init returned -1" in msg) or ("launcher not compatible with PMI1 client" in msg):
                print(
                    "[error] MPI launcher/runtime mismatch detected.\n"
                    f"  launcher used: {mpi_launcher}\n"
                    "Try setting --darcy-mpirun-cmd to the launcher from your active env,\n"
                    "for example: $CONDA_PREFIX/bin/mpiexec or $CONDA_PREFIX/bin/mpiexec.hydra",
                    file=sys.stderr,
                    flush=True,
                )
            raise
    else:
        darcy_mod, SolverClass = _load_solver_module_and_class(darcy_solver)
        setattr(darcy_mod, "current_dir", org_dir)
        # darcy_p1_lm_interior inherits setup() from darcy_p1_lm, whose output
        # path is resolved from the base module global `current_dir`.
        if hasattr(darcy_mod, "lm_base"):
            try:
                setattr(darcy_mod.lm_base, "current_dir", org_dir)
            except Exception:
                pass
        if hasattr(darcy_mod, "darcy_cg"):
            try:
                setattr(darcy_mod.darcy_cg, "current_dir", org_dir)
            except Exception:
                pass
        solver = SolverClass(
            bioreactor_domain, facet_file,
            mesh_inlet_file, mesh_outlet_file,
            pres_inlet_file, pres_outlet_file,
            flow_inlet_file, flow_outlet_file,
            coords_inlet, coords_outlet,
            area_inlet_file=area_inlet_file if Path(area_inlet_file).exists() else None,
            area_outlet_file=area_outlet_file if Path(area_outlet_file).exists() else None,
            concave_bc_mode=concave_bc_mode,
            lp_arterial=lp_arterial,
            lp_venous=lp_venous,
            skip_1d=no_synthetic_vasculature,
            fallback_inlet_pressure=resolved_fallback_inlet_pressure,
            fallback_outlet_pressure=resolved_fallback_outlet_pressure,
            inlet_flux_correction=inlet_flux_correction,
            inlet_flux_corr_max_iter=inlet_flux_corr_max_iter,
            inlet_flux_corr_relax=inlet_flux_corr_relax,
            inlet_flux_corr_tol=inlet_flux_corr_tol,
            branching_in_file=branching_in,
            branching_out_file=branching_out,
            **(
                {
                    "perm_region_path": perm_region_path,
                    "perm_low": perm_low,
                    "perm_high": perm_high,
                    "perm_transition_width": perm_transition_width,
                }
                if darcy_solver in {"darcy_p1_lm_interior", "darcy_mixed"} and perm_region_path
                else {}
            ),
        )
        solver.setup()
    print(f"[organoid_{i}] Darcy done in {perf_counter() - t_darcy:.1f}s", flush=True)

    print(f"[ok] organoid_{i} complete -> {org_dir} (total {perf_counter() - t0:.1f}s)", flush=True)


def parse_args():
    ap = argparse.ArgumentParser(description="Run mesh.py geometry + darcy_P1_v2.py for organoids without copying scripts.")
    ap.add_argument("--stl-dir", default="../../files", help="Directory containing organoid-1.stl, organoid-2.stl, ...")
    ap.add_argument("--trial-dir", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/prep/prepped/trial-9/", help="Directory containing organoid_1/branchingData_0.csv etc (e.g. .../input/trial-2)")
    ap.add_argument("--coupled-root", default="coupled-no-vasc/run_0", help="Output root directory (default: coupled/run_0)")
    ap.add_argument("--n", type=int, default=4, help="Number of organoids (default: 4)")
    ap.add_argument("--same-geometry-for-all-wells", action="store_true",
                    help="Use one shared tissue geometry for all organoids, falling back to STL if no VTU exists.")
    ap.add_argument("--same-meshtags-as-first", dest="same_meshtags_as_first", action="store_true", default=True,
                    help="Reuse well1 facet tags for all wells (default).")
    ap.add_argument("--retag-meshtags-per-well", dest="same_meshtags_as_first", action="store_false",
                    help="Retag facets independently for each translated well (slower).")
    ap.add_argument("--concave-bc-mode", choices=["dirichlet", "robin"], default="dirichlet",
                    help="Boundary model on concave arterial/venous faces in darcy.py.")
    ap.add_argument("--lp-arterial", type=float, default=1e-9,
                    help="Robin transfer coefficient on arterial concave face (used when --concave-bc-mode robin).")
    ap.add_argument("--lp-venous", type=float, default=1e-9,
                    help="Robin transfer coefficient on venous concave face (used when --concave-bc-mode robin).")
    ap.add_argument("--no-synthetic-vasculature", action="store_true",
                    help="Skip 1D synthetic trees and tag only the concave arterial/venous interfaces.")
    ap.add_argument("--fallback-inlet-pressure", type=float, default=0.0,
                    help="Fallback arterial concave pressure used when --no-synthetic-vasculature is enabled.")
    ap.add_argument("--fallback-outlet-pressure", type=float, default=0.0,
                    help="Fallback venous concave pressure used when --no-synthetic-vasculature is enabled.")
    ap.add_argument("--inlet-flux-correction", action="store_true",
                    help="Enable iterative per-terminal inlet flux correction in darcy.py.")
    ap.add_argument("--inlet-flux-corr-max-iter", type=int, default=5,
                    help="Maximum correction iterations.")
    ap.add_argument("--inlet-flux-corr-relax", type=float, default=0.5,
                    help="Relaxation factor for correction updates.")
    ap.add_argument("--inlet-flux-corr-tol", type=float, default=0.05,
                    help="Stop tolerance for max relative inlet flow mismatch.")
    ap.add_argument("--shared-tissue-vtu", default="",
                    help="Optional explicit shared tissue VTU path (used when --same-geometry-for-all-wells is set).")
    ap.add_argument("--reuse-mesh-from-first", dest="reuse_mesh_from_first", action="store_true", default=True,
                    help="Run mesh.py only for organoid_1 and copy geometry outputs to organoid_2..N (default: enabled).")
    ap.add_argument("--no-reuse-mesh-from-first", dest="reuse_mesh_from_first", action="store_false",
                    help="Disable mesh reuse and run mesh.py for each organoid.")
    ap.add_argument("--darcy-mpi-procs", type=int, default=1,
                    help="MPI ranks for Darcy solve (1 = serial in-process).")
    ap.add_argument("--darcy-mpirun-cmd", default="mpirun",
                    help="MPI launcher command for Darcy when --darcy-mpi-procs > 1.")
    ap.add_argument("--darcy-solver", choices=["darcy", "darcy_mixed", "darcy_p1_lm", "darcy_p1_lm_interior"], default="darcy_mixed",
                    help="Darcy solver module/script to run.")
    ap.add_argument("--perm-region-root", default="/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/files/stl/organoid-growth-domains/sphere",
                    help="Directory, STL path, or pattern for organoid permeability regions. If a directory is given, organoid-k uses organoid-k.stl. You can also use a path containing 'X' as the organoid index placeholder.")
    ap.add_argument("--perm-low", type=float, default=1.0e-9,
                    help="Background Darcy permeability outside the organoid STL region.")
    ap.add_argument("--perm-high", type=float, default=2.0e-7,
                    help="Darcy permeability inside the organoid STL region.")
    ap.add_argument("--perm-transition-width", type=float, default=0.01,
                    help="Half-width of the smooth transition shell around the organoid STL boundary.")
    ap.add_argument("--dy-step", type=float, default=0.6, help="Y shift per organoid index (default: 0.6)")
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-.28, 0.9, .5375], help="Base inlet coords (x y z) for organoid 1")
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.30, 0.9, .5375], help="Base outlet coords (x y z) for organoid 1")
    return ap.parse_args()


def main():
    args = parse_args()
    stl_dir = Path(args.stl_dir).expanduser().resolve()
    trial_dir = Path(args.trial_dir).expanduser().resolve()
    coupled_root = Path(args.coupled_root).expanduser().resolve()
    coupled_root.mkdir(parents=True, exist_ok=True)

    coords_inlet_base = np.array(args.coords_inlet, dtype=float)
    coords_outlet_base = np.array(args.coords_outlet, dtype=float)
    shared_tissue_vtu: Path | None = None
    reuse_mesh_from_first = bool(args.reuse_mesh_from_first or args.same_geometry_for_all_wells)
    tmp_mesh_dir = coupled_root / "_tmp_mesh" if reuse_mesh_from_first else None
    # If we reuse mesh from organoid_1, mesh.py should emit well1..wellN files.
    use_mesh_multiwell_translation = bool(reuse_mesh_from_first)
    inferred_well_spacing_y = _infer_well_spacing_y_from_trial(
        trial_dir=trial_dir, n_wells=args.n, fallback=-float(args.dy_step)
    )

    if args.same_geometry_for_all_wells:
        if args.shared_tissue_vtu:
            shared_tissue_vtu = Path(args.shared_tissue_vtu).expanduser().resolve()
            if not shared_tissue_vtu.exists():
                raise FileNotFoundError(f"Missing shared VTU: {shared_tissue_vtu}")
        else:
            shared_tissue_vtu = _pick_existing_or_none(
                trial_dir / "organoid_1" / "tissue_domain_volume_refined.vtu",
                trial_dir / "organoid_1" / "tissue_domain_volume.vtu",
            )

    for i in range(1, args.n + 1):
        run_one(
            i=i,
            stl_dir=stl_dir,
            trial_dir=trial_dir,
            coupled_root=coupled_root,
            coords_inlet_base=coords_inlet_base,
            coords_outlet_base=coords_outlet_base,
            dy_step=args.dy_step,
            well_spacing_y=inferred_well_spacing_y,
            n_wells=args.n,
            same_meshtags_as_first=args.same_meshtags_as_first,
            concave_bc_mode=args.concave_bc_mode,
            lp_arterial=args.lp_arterial,
            lp_venous=args.lp_venous,
            no_synthetic_vasculature=args.no_synthetic_vasculature,
            fallback_inlet_pressure=args.fallback_inlet_pressure,
            fallback_outlet_pressure=args.fallback_outlet_pressure,
            inlet_flux_correction=args.inlet_flux_correction,
            inlet_flux_corr_max_iter=args.inlet_flux_corr_max_iter,
            inlet_flux_corr_relax=args.inlet_flux_corr_relax,
            inlet_flux_corr_tol=args.inlet_flux_corr_tol,
            shared_tissue_vtu=shared_tissue_vtu,
            tmp_mesh_dir=tmp_mesh_dir,
            reuse_mesh_from_first=reuse_mesh_from_first,
            use_mesh_multiwell_translation=use_mesh_multiwell_translation,
            darcy_mpi_procs=args.darcy_mpi_procs,
            darcy_mpirun_cmd=args.darcy_mpirun_cmd,
            darcy_solver=args.darcy_solver,
            perm_region_root=args.perm_region_root,
            perm_low=args.perm_low,
            perm_high=args.perm_high,
            perm_transition_width=args.perm_transition_width,
        )


if __name__ == "__main__":
    main()
