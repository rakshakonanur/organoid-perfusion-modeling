#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np
import shutil

# Import your existing scripts as modules
MESH_DIR  = Path("/Users/rakshakonanur/Documents/Research/two-channel/src/geometry").resolve()
DARCY_DIR = Path("/Users/rakshakonanur/Documents/Research/two-channel/src/solves").resolve()

sys.path.insert(0, str(MESH_DIR))
sys.path.insert(0, str(DARCY_DIR))

import mesh
from darcy_leak import PerfusionSolver

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

def run_one(
    i: int,
    stl_dir: Path,
    trial_dir: Path,
    coupled_root: Path,
    coords_inlet_base: np.ndarray,
    coords_outlet_base: np.ndarray,
    dy_step: float,
) -> None:
    org_dir = coupled_root / f"organoid_{i}"
    geom_dir = org_dir / "geometry"
    out_inlet = org_dir / "0D_Input_Files" / "inlet"
    out_outlet = org_dir / "0D_Input_Files" / "outlet"

    geom_dir.mkdir(parents=True, exist_ok=True)
    out_inlet.mkdir(parents=True, exist_ok=True)
    out_outlet.mkdir(parents=True, exist_ok=True)

    # STL naming: prefer organoid-<i>.stl, fallback organoid_<i>.stl
    stl = stl_dir / f"organoid-{i}.stl"
    if not stl.exists():
        stl = stl_dir / f"organoid_{i}.stl"
    if not stl.exists():
        raise FileNotFoundError(f"Missing STL for organoid {i}: {stl}")

    # Branching data from trial folder
    print("Searching for branching data in ", org_dir)
    branching_in = org_dir / "branchingData_0.csv"
    branching_out = org_dir / "branchingData_1.csv"
    if not branching_in.exists():
        raise FileNotFoundError(f"Missing: {branching_in}")
    if not branching_out.exists():
        raise FileNotFoundError(f"Missing: {branching_out}")

    # _copy_tree(trial_dir / f"organoid_{i}" / "0D_Input_Files"/ "inlet", out_inlet)
    # _copy_tree(trial_dir / f"organoid_{i}" / "0D_Input_Files"/ "outlet", out_outlet)

    # Update z-coordinate: subtract 0.6*(i-1) from the 2nd component
    shift = dy_step * (i - 1)
    coords_inlet = coords_inlet_base.copy()
    coords_outlet = coords_outlet_base.copy()
    coords_inlet[1] -= shift
    coords_outlet[1] -= shift

    # ---- Run geometry (mesh) ----
    # Route geometry outputs into organoid-specific geometry folder
    mesh.current_dir = geom_dir

    perfusion = mesh.Files(
        stl_file=str(stl),
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
        msh_bioreactor=str(geom_dir / "bioreactor.msh"),
        xdmf_bioreactor=str(geom_dir / "bioreactor.xdmf"),
        coords_inlet=coords_inlet,
        coords_outlet=coords_outlet,
        single=False,
        init=True,
    )

    # If mesh.Files triggers everything in __init__, we’re done.
    # If it requires an explicit call like perfusion.run(), you can add it here.
    # e.g., if hasattr(perfusion, "run"): perfusion.run()

    # ---- Run Darcy ----
    bioreactor_domain = str(geom_dir / "bioreactor.xdmf")
    facet_file = str(geom_dir / "bioreactor_facets.xdmf")
    mesh_inlet_file   = str(geom_dir / "tagged_branches_inlet.bp")
    mesh_outlet_file  = str(geom_dir / "tagged_branches_outlet.bp")
    pres_inlet_file   = str(geom_dir / "pressure_checkpoint_inlet.bp")
    pres_outlet_file  = str(geom_dir / "pressure_checkpoint_outlet.bp")
    velocity_inlet_file   = str(geom_dir / "velocity_checkpoint_inlet.bp")
    velocity_outlet_file  = str(geom_dir / "velocity_checkpoint_outlet.bp")

    solver = PerfusionSolver(
        bioreactor_domain, facet_file,
        mesh_inlet_file, mesh_outlet_file,
        pres_inlet_file, pres_outlet_file,
        velocity_inlet_file, velocity_outlet_file,
        coords_inlet, coords_outlet,
        branching_in, branching_out
    )
    solver.setup()

    _copy_tree(DARCY_DIR/"out_darcy", org_dir/"out_darcy")

    print(f"[ok] organoid_{i} complete -> {org_dir}")


def parse_args():
    ap = argparse.ArgumentParser(description="Run mesh.py geometry + darcy_P1_v2.py for organoids without copying scripts.")
    ap.add_argument("--stl-dir", default="../files", help="Directory containing organoid-1.stl, organoid-2.stl, ...")
    ap.add_argument("--trial-dir", default="/Users/rakshakonanur/Documents/Research/two-channel/input/trial-5", help="Directory containing organoid_1/branchingData_0.csv etc (e.g. .../input/trial-2)")
    ap.add_argument("--coupled-root", default="coupled/run_0", help="Output root directory (default: coupled/run_0)")
    ap.add_argument("--n", type=int, default=4, help="Number of organoids (default: 4)")
    ap.add_argument("--dy-step", type=float, default=0.6, help="Y shift per organoid index (default: 0.6)")
    ap.add_argument("--coords-inlet", nargs=3, type=float, default=[-0.175, 0.9, 0.55], help="Base inlet coords (x y z) for organoid 1")
    ap.add_argument("--coords-outlet", nargs=3, type=float, default=[0.175, 0.9, 0.55], help="Base outlet coords (x y z) for organoid 1")
    return ap.parse_args()


def main():
    args = parse_args()
    stl_dir = Path(args.stl_dir).expanduser().resolve()
    trial_dir = Path(args.trial_dir).expanduser().resolve()
    coupled_root = Path(args.coupled_root).expanduser().resolve()
    coupled_root.mkdir(parents=True, exist_ok=True)

    coords_inlet_base = np.array(args.coords_inlet, dtype=float)
    coords_outlet_base = np.array(args.coords_outlet, dtype=float)

    for i in range(1, args.n + 1):
        run_one(
            i=i,
            stl_dir=stl_dir,
            trial_dir=trial_dir,
            coupled_root=coupled_root,
            coords_inlet_base=coords_inlet_base,
            coords_outlet_base=coords_outlet_base,
            dy_step=args.dy_step,
        )


if __name__ == "__main__":
    main()
