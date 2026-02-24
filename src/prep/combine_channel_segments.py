#!/usr/bin/env python3
"""
Combine an existing discretized 0D channel solver (e.g., 1000 vessels in series)
into a smaller number of segments (e.g., 11), using the segment lengths in an Excel
sheet as the reference for where to place boundaries along the channel.

NEW: Also writes an updated Excel file where the Resistance and Inductance columns
are replaced with the *combined* values computed from the template solver.

Usage:
  python combine_channel_segments.py \
      --xlsx channel-resistance.xlsx \
      --solver solver_0d.json \
      --out solver_0d_template_combined_to_11.json \
      --xlsx_out channel-resistance_updated.xlsx
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def read_segments_xlsx(xlsx_path: Path) -> tuple[pd.DataFrame, list[str], np.ndarray, str, str, str | None, str | None]:
    """
    Returns:
      df_raw: original dataframe with original column names
      seg_names: list of segment names in row order
      seg_lengths: numpy array of lengths in row order
      seg_col: name of the "Segment" column in df_raw
      len_col: name of the "Length" column in df_raw
      res_col: name of the resistance column if present (else None)
      ind_col: name of the inductance column if present (else None)
    """
    df = pd.read_excel(xlsx_path, sheet_name=0)

    # Map normalized -> original column name
    norm_map = {_norm(c): c for c in df.columns}

    # Required: segment + length
    if "segment" not in norm_map:
        raise ValueError(f"Missing required column 'Segment' in {xlsx_path.name}")
    seg_col = norm_map["segment"]

    # Length can be Length or Length_
    if "length_" in norm_map:
        len_col = norm_map["length_"]
    elif "length" in norm_map:
        len_col = norm_map["length"]
    else:
        raise ValueError(f"Missing required column 'Length' (or 'Length_') in {xlsx_path.name}")

    # Optional: resistance/inductance (handle common typo "Resitance")
    res_col = None
    if "resistance" in norm_map:
        res_col = norm_map["resistance"]
    elif "resitance" in norm_map:
        res_col = norm_map["resitance"]

    ind_col = norm_map.get("inductance", None)

    seg_names = df[seg_col].astype(str).tolist()
    seg_lengths = pd.to_numeric(df[len_col], errors="raise").astype(float).to_numpy()

    if np.any(seg_lengths <= 0):
        bad = np.where(seg_lengths <= 0)[0].tolist()
        raise ValueError(f"All segment lengths must be > 0. Bad row indices: {bad}")

    return df, seg_names, seg_lengths, seg_col, len_col, res_col, ind_col


def combine_template_into_segments(
    template: dict,
    seg_names: list[str],
    seg_lengths: np.ndarray
) -> tuple[dict, list[dict]]:
    """
    Returns:
      combined_solver_json
      accum: list of dicts with keys len, R, L, C for each segment (in seg order)
    """
    vessels = template["vessels"]
    if len(vessels) < 2:
        raise ValueError("Template solver must contain at least 2 vessels to combine.")

    # Extract template lengths and params
    v_len = np.array([float(v.get("vessel_length", 0.0)) for v in vessels], dtype=float)
    if np.any(v_len < 0):
        raise ValueError("Template vessel_length must be non-negative.")
    template_total = float(v_len.sum())
    if template_total <= 0:
        raise ValueError("Template total length is zero; cannot segment by length.")

    v_R = np.array([float(v["zero_d_element_values"].get("R_poiseuille", 0.0)) for v in vessels], dtype=float)
    v_L = np.array([float(v["zero_d_element_values"].get("L", 0.0)) for v in vessels], dtype=float)
    v_C = np.array([float(v["zero_d_element_values"].get("C", 0.0)) for v in vessels], dtype=float)

    # Boundary locations along template arclength, proportional to Excel lengths
    target_total = float(seg_lengths.sum())
    seg_fracs = np.cumsum(seg_lengths) / target_total
    boundaries = seg_fracs * template_total  # last boundary = template_total

    accum = [{"len": 0.0, "R": 0.0, "L": 0.0, "C": 0.0} for _ in seg_lengths]

    seg_idx = 0
    pos = 0.0  # arclength at start of current vessel
    next_boundary = boundaries[0] if len(boundaries) else template_total

    for i in range(len(vessels)):
        Li = float(v_len[i])
        if Li <= 0:
            continue

        Ri = float(v_R[i])
        Li_inert = float(v_L[i])
        Ci = float(v_C[i])

        remaining = Li
        local_start = 0.0  # within this vessel

        while remaining > 1e-15:
            if seg_idx >= len(seg_lengths):
                seg_idx = len(seg_lengths) - 1
                next_boundary = float("inf")

            dist_to_boundary = next_boundary - (pos + local_start)
            if dist_to_boundary <= 1e-15:
                seg_idx += 1
                if seg_idx < len(boundaries):
                    next_boundary = boundaries[seg_idx]
                else:
                    next_boundary = float("inf")
                continue

            take = min(remaining, dist_to_boundary)
            frac = take / Li  # linear scaling assumption

            accum[seg_idx]["len"] += take
            accum[seg_idx]["R"] += Ri * frac
            accum[seg_idx]["L"] += Li_inert * frac
            accum[seg_idx]["C"] += Ci * frac

            remaining -= take
            local_start += take

        pos += Li

    # Build new solver JSON with 11 vessels in series
    bc = template["boundary_conditions"]
    sim = template["simulation_parameters"]

    new_vessels = []
    for j, name in enumerate(seg_names):
        vals = accum[j]
        new_vessels.append({
            "vessel_id": int(j),
            "vessel_length": float(vals["len"]),
            "vessel_name": str(name),
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "C": float(vals["C"]),
                "L": float(vals["L"]),
                "R_poiseuille": float(vals["R"]),
                "stenosis_coefficient": 0.0,
            },
        })

    # Preserve endpoint BC labels (names) from template
    inlet_bc_name = vessels[0].get("boundary_conditions", {}).get("inlet")
    outlet_bc_name = vessels[-1].get("boundary_conditions", {}).get("outlet")
    if inlet_bc_name:
        new_vessels[0]["boundary_conditions"] = {"inlet": inlet_bc_name}
    if outlet_bc_name:
        new_vessels[-1].setdefault("boundary_conditions", {})
        new_vessels[-1]["boundary_conditions"]["outlet"] = outlet_bc_name

    junctions = [{
        "junction_name": f"J{j}",
        "junction_type": "NORMAL_JUNCTION",
        "inlet_vessels": [j],
        "outlet_vessels": [j + 1],
    } for j in range(len(new_vessels) - 1)]

    combined = {
        "boundary_conditions": bc,
        "simulation_parameters": sim,
        "vessels": new_vessels,
        "junctions": junctions,
    }

    return combined, accum


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default="../files/channel-resistance-sv.xlsx",
                    help="Path to channel-resistance.xlsx")
    ap.add_argument("--solver", type=Path, default="../files/reference-sv-files/solver_0d.json",
                    help="Path to template solver_0d.json (many vessels)")
    ap.add_argument("--out", type=Path, default="../files/reference-sv-files/interpolated/solver_0d.json",
                    help="Output json path (combined to 11 segments)")
    ap.add_argument("--xlsx_out", type=Path, default="../files/channel-resistance-sv.xlsx",
                    help="Output xlsx path with updated Resistance/Inductance. "
                         "Default: <xlsx_stem>_updated.xlsx next to input xlsx.")
    args = ap.parse_args()

    df_raw, seg_names, seg_lengths, seg_col, len_col, res_col, ind_col = read_segments_xlsx(args.xlsx)

    with args.solver.open("r") as f:
        template = json.load(f)

    combined, accum = combine_template_into_segments(template, seg_names, seg_lengths)

    # Write combined JSON
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(combined, indent=2))

    # Update Excel resistance + inductance (create columns if missing)
    if res_col is None:
        res_col = "Resistance"
        df_raw[res_col] = np.nan
    if ind_col is None:
        ind_col = "Inductance"
        df_raw[ind_col] = np.nan

    # Overwrite values (row-for-row corresponds to seg_names order)
    df_raw[res_col] = [a["R"] for a in accum]
    df_raw[ind_col] = [a["L"] for a in accum]

    # Write updated Excel (do NOT overwrite by default)
    xlsx_out = args.xlsx_out
    if xlsx_out is None:
        xlsx_out = args.xlsx.with_name(f"{args.xlsx.stem}_updated{args.xlsx.suffix}")
    xlsx_out.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_excel(xlsx_out, index=False)

    print(f"[OK] Wrote {args.out} with {len(combined['vessels'])} vessels and {len(combined['junctions'])} junctions.")
    print(f"[OK] Wrote updated Excel with combined {res_col}/{ind_col}: {xlsx_out}")


if __name__ == "__main__":
    main()
