#!/usr/bin/env python3
"""
Run svZeroDSolver on a .in/.json input file and split the resulting CSV outputs into
separate CSVs for:
  - channel
  - organoid_1_inlet, organoid_1_outlet
  - organoid_2_inlet, organoid_2_outlet
  - ...
based on vessel-name prefixes in the output.

Also optionally:
  - strip organoid prefixes from organoid columns in split outputs
  - copy organoid split CSVs into organoid_X/0D_Input_Files/{inlet,outlet}
  - run plot_0d_results_to_3d.py in each destination directory (no arguments)
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


# -----------------------------
# Group inference helpers
# -----------------------------

# underscores are "word" chars; don't use \b. Use lookahead for [_\- ]|$.
_CHANNEL_RE = re.compile(r"(^|[^A-Za-z0-9_])channel(?=[_\-]|$)", re.IGNORECASE)
_ORG_RE     = re.compile(r"organoid[_\- ]?(\d+)[_\- ]?(inlet|outlet)(?=[_\- ]|$)", re.IGNORECASE)

# For stripping organoid prefix from columns/vessels:
# Matches organoid1_inlet_, organoid_1_inlet_, organoid-1 inlet-, etc.
def _org_prefix_re(idx: int, side: str) -> re.Pattern:
    return re.compile(rf"organoid[_\- ]*{idx}[_\- ]*{side}[_\- ]*", re.IGNORECASE)


def infer_group_from_vessel_name(vessel: str) -> Optional[str]:
    if not vessel:
        return None
    v = str(vessel).strip()

    if v.lower().startswith("channel") or _CHANNEL_RE.search(v):
        return "channel"

    m = _ORG_RE.search(v)
    if m:
        idx = int(m.group(1))
        side = m.group(2).lower()
        return f"organoid_{idx}_{side}"

    # try normalized version
    v2 = v.replace(":", "_")
    if v2.lower().startswith("channel") or _CHANNEL_RE.search(v2):
        return "channel"
    m = _ORG_RE.search(v2)
    if m:
        idx = int(m.group(1))
        side = m.group(2).lower()
        return f"organoid_{idx}_{side}"

    return None


def infer_vessel_from_column(col: str) -> Optional[str]:
    if not col:
        return None
    c = col.strip()

    # "field:vessel"
    if ":" in c:
        _, rhs = c.split(":", 1)
        rhs = rhs.strip()
        return rhs if rhs else None

    # "vessel|field" or "field|vessel"
    if "|" in c:
        lhs, rhs = c.split("|", 1)
        if infer_group_from_vessel_name(lhs):
            return lhs.strip()
        if infer_group_from_vessel_name(rhs):
            return rhs.strip()

    # suffix field: vessel_P, vessel_Q, vessel_pressure, vessel_flow
    m = re.match(r"(.+)[_\-](P|Q|p|q|pressure|flow)$", c)
    if m:
        return m.group(1).strip()

    # prefix field: P_vessel, Q_vessel
    m = re.match(r"^(P|Q|p|q|pressure|flow)[_\-](.+)$", c)
    if m:
        return m.group(2).strip()

    if infer_group_from_vessel_name(c):
        return c

    return None


def infer_time_columns(columns: List[str]) -> List[str]:
    candidates = []
    for c in columns:
        cl = c.strip().lower()
        if cl in ("t", "time", "times", "timestamp"):
            candidates.append(c)
        elif cl.endswith("_time") or cl.endswith("_t"):
            candidates.append(c)
    return candidates


# -----------------------------
# Organoid prefix stripping
# -----------------------------

def strip_organoid_prefix_from_vessel(vessel: str, idx: int, side: str) -> str:
    """Remove organoid{idx}_{side}_ prefix from a vessel string."""
    if not vessel:
        return vessel
    return _org_prefix_re(idx, side).sub("", vessel, count=1).lstrip("_- ")


def strip_organoid_prefix_from_column(col: str, idx: int, side: str) -> str:
    """
    Strip organoid prefix from a WIDE-format column name while preserving field formatting.

    Handles:
      - "Q:organoid1_inlet_branch_0_seg0" -> "Q:branch_0_seg0"
      - "organoid_1_outlet_branch_3_seg0_Q" -> "branch_3_seg0_Q"
      - "Q_organoid1_inlet_branch_0_seg0" -> "Q_branch_0_seg0"
      - "organoid1_inlet_branch_0_seg0" -> "branch_0_seg0"
    """
    c = str(col)

    # field:vessel
    if ":" in c:
        field, vessel = c.split(":", 1)
        vessel2 = strip_organoid_prefix_from_vessel(vessel.strip(), idx, side)
        return f"{field}:{vessel2}"

    # suffix field
    m = re.match(r"(.+)([_\-])(P|Q|p|q|pressure|flow)$", c)
    if m:
        vessel = m.group(1)
        sep = m.group(2)
        field = m.group(3)
        vessel2 = strip_organoid_prefix_from_vessel(vessel, idx, side)
        return f"{vessel2}{sep}{field}"

    # prefix field
    m = re.match(r"^(P|Q|p|q|pressure|flow)([_\-])(.+)$", c)
    if m:
        field = m.group(1)
        sep = m.group(2)
        vessel = m.group(3)
        vessel2 = strip_organoid_prefix_from_vessel(vessel, idx, side)
        return f"{field}{sep}{vessel2}"

    # plain vessel-like column
    return strip_organoid_prefix_from_vessel(c, idx, side)


def maybe_strip_organoid_prefixes_in_df(df, group_name: str, time_cols: List[str], vessel_col: Optional[str] = None):
    """
    Apply stripping for organoid groups:
      - WIDE: rename columns (excluding time_cols)
      - LONG: strip prefix inside vessel_col values
    """
    m = re.match(r"^organoid_(\d+)_(inlet|outlet)$", group_name)
    if not m:
        return df

    idx = int(m.group(1))
    side = m.group(2).lower()

    if vessel_col is not None and vessel_col in df.columns:
        # LONG: adjust vessel name values
        df[vessel_col] = df[vessel_col].astype(str).map(lambda v: strip_organoid_prefix_from_vessel(v, idx, side))
        return df

    # WIDE: rename columns except time
    rename_map = {}
    for c in df.columns:
        if c in time_cols:
            continue
        rename_map[c] = strip_organoid_prefix_from_column(c, idx, side)

    return df.rename(columns=rename_map)


# -----------------------------
# CSV splitting: pandas version
# -----------------------------

def split_csv_with_pandas(csv_path: Path, split_dir: Path, debug: bool = False) -> List[Path]:
    df = pd.read_csv(csv_path)

    cols = list(df.columns)
    time_cols = infer_time_columns(cols)

    # Detect long format vessel column
    vessel_col = None
    for cand in ["vessel_name", "vessel", "name", "vesselName"]:
        if cand in df.columns:
            vessel_col = cand
            break
        for c in df.columns:
            if c.lower() == cand.lower():
                vessel_col = c
                break
        if vessel_col:
            break

    written: List[Path] = []

    if vessel_col is not None:
        if debug:
            print(f"[debug] Detected LONG format via vessel column: {vessel_col}")

        groups = df[vessel_col].astype(str).apply(infer_group_from_vessel_name)
        df["_group"] = groups.fillna("other")

        if debug:
            print(f"[debug] Group counts: {df['_group'].value_counts().to_dict()}")

        for gname, gdf in df.groupby("_group", sort=True):
            gdf = gdf.drop(columns=["_group"])
            gdf = maybe_strip_organoid_prefixes_in_df(gdf, gname, time_cols=time_cols, vessel_col=vessel_col)
            out = split_dir / f"{gname}.csv"
            gdf.to_csv(out, index=False)
            written.append(out)
        return written

    if debug:
        print("[debug] Detected WIDE format (no vessel_name column)")

    # Wide format: classify columns by parsing vessel from column name
    col_groups: Dict[str, str] = {}
    for c in cols:
        if c in time_cols:
            continue
        vessel = infer_vessel_from_column(c)
        g = infer_group_from_vessel_name(vessel) if vessel else None
        col_groups[c] = g or "other"

    # Build one df per group
    group_to_cols: Dict[str, List[str]] = {}
    for c, g in col_groups.items():
        group_to_cols.setdefault(g, []).append(c)

    if debug:
        other_cols = [c for c, g in col_groups.items() if g == "other"]
        print(f"[debug] Columns classified as 'other': {len(other_cols)}")
        for c in other_cols[:20]:
            print("  [other]", c)

    for g, gcols in sorted(group_to_cols.items()):
        keep = list(dict.fromkeys(time_cols + gcols))
        gdf = df[keep].copy()
        gdf = maybe_strip_organoid_prefixes_in_df(gdf, g, time_cols=time_cols, vessel_col=None)
        out = split_dir / f"{g}.csv"
        gdf.to_csv(out, index=False)
        written.append(out)

    return written


# -----------------------------
# CSV splitting: stdlib fallback (fixed)
# -----------------------------

def split_csv_with_csv_module(csv_path: Path, split_dir: Path, debug: bool = False) -> List[Path]:
    """
    Fallback splitting if pandas isn't available.
    Supports LONG if a vessel column exists; otherwise WIDE by column grouping.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        time_cols = infer_time_columns(fieldnames)

        vessel_col = None
        for c in fieldnames:
            if c.lower() in ("vessel_name", "vessel", "name", "vesselname"):
                vessel_col = c
                break

        outputs: List[Path] = []

        if vessel_col:
            # LONG: stream rows into separate files
            writers: Dict[str, Tuple[Path, csv.DictWriter]] = {}
            file_handles: Dict[str, any] = {}
            try:
                for row in reader:
                    g = infer_group_from_vessel_name(str(row.get(vessel_col, ""))) or "other"

                    # strip prefix in vessel value for organoid groups
                    m = re.match(r"^organoid_(\d+)_(inlet|outlet)$", g)
                    if m:
                        idx = int(m.group(1))
                        side = m.group(2).lower()
                        row[vessel_col] = strip_organoid_prefix_from_vessel(str(row.get(vessel_col, "")), idx, side)

                    if g not in writers:
                        out = split_dir / f"{g}.csv"
                        fh = out.open("w", newline="")
                        w = csv.DictWriter(fh, fieldnames=fieldnames)
                        w.writeheader()
                        writers[g] = (out, w)
                        file_handles[g] = fh
                        outputs.append(out)

                    writers[g][1].writerow(row)
            finally:
                for fh in file_handles.values():
                    fh.close()
            return outputs

    # WIDE: we need a second pass; read header, classify cols, then stream rows
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        time_cols = infer_time_columns(fieldnames)

        # classify columns
        col_groups: Dict[str, str] = {}
        for c in fieldnames:
            if c in time_cols:
                continue
            vessel = infer_vessel_from_column(c)
            g = infer_group_from_vessel_name(vessel) if vessel else None
            col_groups[c] = g or "other"

        group_to_cols: Dict[str, List[str]] = {}
        for c, g in col_groups.items():
            group_to_cols.setdefault(g, []).append(c)

        if debug:
            other_cols = [c for c, g in col_groups.items() if g == "other"]
            print(f"[debug] Columns classified as 'other': {len(other_cols)}")
            for c in other_cols[:20]:
                print("  [other]", c)

        # open writers per group with filtered headers
        writers: Dict[str, Tuple[Path, csv.DictWriter, List[str]]] = {}
        file_handles: Dict[str, any] = {}
        outputs: List[Path] = []

        try:
            for g, gcols in sorted(group_to_cols.items()):
                keep = list(dict.fromkeys(time_cols + gcols))

                # apply stripping to organoid group column names (header)
                m = re.match(r"^organoid_(\d+)_(inlet|outlet)$", g)
                if m:
                    idx = int(m.group(1))
                    side = m.group(2).lower()
                    keep2 = []
                    for c in keep:
                        if c in time_cols:
                            keep2.append(c)
                        else:
                            keep2.append(strip_organoid_prefix_from_column(c, idx, side))
                    header = keep2
                else:
                    header = keep

                out = split_dir / f"{g}.csv"
                fh = out.open("w", newline="")
                w = csv.DictWriter(fh, fieldnames=header)
                w.writeheader()
                writers[g] = (out, w, keep)
                file_handles[g] = fh
                outputs.append(out)

            # stream rows and write filtered dicts (respect original keep list for lookup)
            for row in reader:
                for g, (_out, w, keep) in writers.items():
                    m = re.match(r"^organoid_(\d+)_(inlet|outlet)$", g)
                    if m:
                        idx = int(m.group(1))
                        side = m.group(2).lower()
                        out_row = {}
                        for c in keep:
                            if c in time_cols:
                                out_row[c] = row.get(c, "")
                            else:
                                out_row[strip_organoid_prefix_from_column(c, idx, side)] = row.get(c, "")
                    else:
                        out_row = {c: row.get(c, "") for c in keep}

                    w.writerow(out_row)

        finally:
            for fh in file_handles.values():
                fh.close()

        return outputs


def split_output_csv(csv_path: Path, outdir: Path, debug: bool = False) -> List[Path]:
    split_dir = outdir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    if pd is not None:
        return split_csv_with_pandas(csv_path, split_dir, debug=debug)
    return split_csv_with_csv_module(csv_path, split_dir, debug=debug)


# -----------------------------
# Copy + plot
# -----------------------------

def copy_and_plot_organoid_files(
    written_files: List[Path],
    organoid_root: Path,
    plot_script: str = "plot_0d_results_to_3d.py",
    run_plot: bool = True,
    debug: bool = False,
) -> None:
    """
    Copy split organoid CSVs into:
      <organoid_root>/organoid_X/0D_Input_Files/inlet/
      <organoid_root>/organoid_X/0D_Input_Files/outlet/
    Then run plot_script once in each destination directory.
    """
    target_dirs: Set[Path] = set()

    for p in written_files:
        m = re.match(r"^organoid_(\d+)_(inlet|outlet)\.csv$", p.name)
        if not m:
            continue

        idx = m.group(1)
        side = m.group(2).lower()

        candidates = [
            organoid_root / f"organoid_{idx}" / "0D_Input_Files" / side,
            organoid_root / f"organoid-{idx}" / "0D_Input_Files" / side,
        ]
        dest_dir = None
        for cand in candidates:
            if cand.exists():
                dest_dir = cand
                break
        if dest_dir is None:
            raise SystemExit(
                "Destination directory does not exist. Tried:\n"
                + "\n".join(str(c) for c in candidates)
            )

        dest_file = dest_dir / "output.csv"
        shutil.copy2(p, dest_file)

        if debug:
            print(f"[debug] Copied {p} -> {dest_file}")

        target_dirs.add(dest_dir)

    # Run plots only after all output.csv files are copied.
    if not run_plot:
        return
    for dest_dir in sorted(target_dirs):
        cmd = ["python3", plot_script]
        try:
            subprocess.run(cmd, cwd=str(dest_dir), check=True)
        except FileNotFoundError:
            cmd = ["python", plot_script]
            subprocess.run(cmd, cwd=str(dest_dir), check=True)
        if debug:
            print(f"[debug] Ran {plot_script} in {dest_dir}")


# -----------------------------
# Main: run solver then split
# -----------------------------

def run_solver(exe: str, input_file: Path, output_file: Path, cwd: Path) -> None:
    cmd = [exe, str(input_file), str(output_file)]
    subprocess.run(cmd, cwd=str(cwd), stdout=None, stderr=None, shell=False, check=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run svzerodsolver and split the output CSV by group (channel/organoids).")
    ap.add_argument("--exe", default="svzerodsolver", help="Executable name or path (default: svzerodsolver)")
    ap.add_argument("--input", type=str, default="./prepped/trial-8/combined.in", help="Input .in/.json file for svzerodsolver")
    ap.add_argument("--outdir", type=str, default="./prepped/trial-8", help="Output directory (cwd for solver, and where split/ is created)")
    ap.add_argument("--output", type=str, default="output.csv", help="Solver output filename (default: output.csv)")
    ap.add_argument("--no-run", action="store_true", help="Do not run solver; only split the existing output CSV")
    ap.add_argument("--debug", action="store_true", help="Print grouping/debug info while splitting")

    # New options for your workflow
    ap.add_argument(
        "--organoid-root",
        type=str,
        default="",
        help="If provided, copy organoid split CSVs to <organoid_root>/organoid_X/0D_Input_Files/{inlet,outlet} and run plot script there.",
    )
    ap.add_argument(
        "--plot-script",
        type=str,
        default="plot_0d_results_to_3d.py",
        help="Plot script name to run inside each inlet/outlet directory (default: plot_0d_results_to_3d.py)",
    )
    ap.add_argument(
        "--no-plot",
        action="store_true",
        help="Copy output.csv into organoid inlet/outlet folders but do not run plot script.",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    output_csv = outdir / args.output

    if not args.no_run:
        if not args.input:
            raise SystemExit("--input is required unless --no-run is set.")
        input_file = Path(args.input).resolve()
        run_solver(args.exe, input_file, output_csv, cwd=outdir)

    if not output_csv.exists():
        raise SystemExit(f"Expected output file not found: {output_csv}")

    written = split_output_csv(output_csv, outdir, debug=getattr(args, "debug", False))

    print("Wrote split files:")
    for p in written:
        print(f"  - {p}")

    organoid_root = Path(args.organoid_root).resolve() if args.organoid_root else outdir
    copy_and_plot_organoid_files(
        written_files=written,
        organoid_root=organoid_root,
        plot_script=args.plot_script,
        run_plot=not args.no_plot,
        debug=getattr(args, "debug", False),
    )


if __name__ == "__main__":
    main()
