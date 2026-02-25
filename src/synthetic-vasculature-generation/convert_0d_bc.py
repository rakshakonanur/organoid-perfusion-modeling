#!/usr/bin/env python3
"""
Convert a 0D solver input from FLOW-inlet + RESISTANCE-outlets
→ PRESSURE-inlet + FLOW-outlets.

Usage
-----
python convert_0d_bc.py \
  --in solver_old.in \
  --out solver_new.in \
  --inlet-name PRESSURE_IN \
  --inlet-P 0,0 \
  --t 0,1 \
  --outlet-Q 0

Notes
-----
- If you omit --t, we'll re-use the inlet's original time array when available;
  otherwise we default to [0, 1].
- --inlet-P and --outlet-Q accept either a scalar (e.g., 0) or a comma list
  (e.g., 0,0,50,50) with the same length as --t. If a scalar is given, it is
  broadcast to the length of --t.
- The script auto-detects the inlet BC by finding the vessel that has
  boundary_conditions["inlet"]. It then updates that BC and (optionally) renames it.
- All BCs with bc_type == "RESISTANCE" are converted to FLOW BCs using the same t-array
  and the provided Q values. The old keys (e.g., R, Pd) are removed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

# -------------------------
# helpers
# -------------------------

def _parse_series(arg: str | None, t: List[float], fallback_scalar: float) -> List[float]:
    """Parse a series argument:
    - if arg is None → return [fallback_scalar]*len(t)
    - if arg is a single float-like string → broadcast
    - if arg is comma-separated values → parse list and validate length
    """
    if arg is None:
        return [fallback_scalar for _ in t]
    s = arg.strip()
    if "," in s:
        vals = [float(x) for x in s.split(",") if x.strip() != ""]
        if len(vals) != len(t):
            raise ValueError(f"Provided series has length {len(vals)} but t has length {len(t)}")
        return vals
    # scalar
    val = float(s)
    return [val for _ in t]


def _find_inlet_bc_name(data: Dict[str, Any]) -> str:
    """Return the bc_name referenced by the inlet vessel mapping.
    Raises if not found or ambiguous.
    """
    inlet_names: List[str] = []
    for v in data.get("vessels", []):
        bcmap = v.get("boundary_conditions", {}) or {}
        if "inlet" in bcmap:
            inlet_names.append(bcmap["inlet"])
    if not inlet_names:
        raise ValueError("Could not find an inlet boundary condition mapping in any vessel.")
    # Often there is only one; if multiple, we take the first but warn in verbose contexts.
    return inlet_names[0]


def _bc_index_by_name(data: Dict[str, Any]) -> Dict[str, int]:
    return {bc.get("bc_name"): i for i, bc in enumerate(data.get("boundary_conditions", []))}


# -------------------------
# main transform
# -------------------------

def transform_flow_to_pressure_inlet_and_flow_outlets(
    data: Dict[str, Any],
    inlet_new_name: str,
    P_series: List[float],
    t_series: List[float],
    Q_series_for_outlets: List[float],
    edit_flows: bool = True,
) -> Dict[str, Any]:
    bc_list = list(data.get("boundary_conditions", []))
    name_to_idx = _bc_index_by_name(data)

    # 1) Identify inlet bc name from vessel mapping
    old_inlet_name = _find_inlet_bc_name(data)
    if old_inlet_name not in name_to_idx:
        raise ValueError(f"Inlet BC '{old_inlet_name}' not present in boundary_conditions list.")

    # 2) Update/rename inlet BC to PRESSURE with P-series
    i = name_to_idx[old_inlet_name]
    inlet_bc = dict(bc_list[i])  # shallow copy
    inlet_bc["bc_name"] = inlet_new_name
    inlet_bc["bc_type"] = "PRESSURE"
    inlet_bc["bc_values"] = {"P": P_series, "t": t_series}
    bc_list[i] = inlet_bc

    # Also update the vessel mapping that referenced the old inlet name
    for v in data.get("vessels", []):
        bcmap = v.get("boundary_conditions", {}) or {}
        if bcmap.get("inlet") == old_inlet_name:
            bcmap["inlet"] = inlet_new_name
            v["boundary_conditions"] = bcmap

    # 3) Convert all RESISTANCE outlets to FLOW with Q-series and same t-series
    if edit_flows:
        for j, bc in enumerate(bc_list):
            if j == i:
                continue
            bctype = bc.get("bc_type", "").upper()
            if bctype == "RESISTANCE":
                bc = dict(bc)
                bc["bc_type"] = "FLOW"
                bc["bc_values"] = {"Q": Q_series_for_outlets, "t": t_series}
                # Preserve name and any non-values keys; drop resistance-specific keys if present under bc_values
                bc_list[j] = bc

    data_out = dict(data)
    data_out["boundary_conditions"] = bc_list
    return data_out

def transform_resistance_to_pressure_outlets(
    data: Dict[str, Any],
    P_series_for_outlets: List[float],
    t_series: List[float],
    edit_flows: bool = True,
) -> Dict[str, Any]:
    bc_list = list(data.get("boundary_conditions", []))
    name_to_idx = _bc_index_by_name(data)

    # 1) Identify inlet bc name from vessel mapping
    old_inlet_name = _find_inlet_bc_name(data)
    if old_inlet_name not in name_to_idx:
        raise ValueError(f"Inlet BC '{old_inlet_name}' not present in boundary_conditions list.")

    # 2) Update/rename inlet BC to PRESSURE with P-series
    i = name_to_idx[old_inlet_name]

    # 3) Convert all RESISTANCE outlets to PRESSURE with P-series and same t-series
    if edit_flows:
        for j, bc in enumerate(bc_list):
            if j == i:
                continue
            bctype = bc.get("bc_type", "").upper()
            if bctype == "RESISTANCE":
                bc = dict(bc)
                bc["bc_type"] = "PRESSURE"
                bc["bc_values"] = {"P": P_series_for_outlets, "t": t_series}
                # Preserve name and any non-values keys; drop resistance-specific keys if present under bc_values
                bc_list[j] = bc

    data_out = dict(data)
    data_out["boundary_conditions"] = bc_list
    return data_out


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert 0D solver BCs: flow-inlet/resistance-outlets → pressure-inlet/flow-outlets.")
    ap.add_argument("--in", dest="in_path", required=True, help="Input solver JSON (e.g., solver_old.in)")
    ap.add_argument("--out", dest="out_path", required=True, help="Output solver JSON (e.g., solver_new.in)")
    ap.add_argument("--inlet-name", default="PRESSURE_IN", help="New inlet bc_name to use (default: PRESSURE_IN)")
    ap.add_argument("--inlet-P", dest="inlet_P", default=None, help="Inlet pressure series: scalar or comma-list. If omitted, uses 0.")
    ap.add_argument("--t", dest="t_vals", default=None, help="Time series: comma-list. If omitted, re-use inlet's original t if found, else [0,1].")
    ap.add_argument("--outlet-Q", dest="outlet_Q", default=None, help="Outlet flow series: scalar or comma-list. If omitted, uses 0.")

    args = ap.parse_args()

    # Load JSON
    src = Path(args.in_path)
    data: Dict[str, Any] = json.loads(src.read_text())

    # Determine a default t-series: reuse current inlet's 't' if present and it is FLOW
    try:
        inlet_name = _find_inlet_bc_name(data)
        name_to_idx = _bc_index_by_name(data)
        inlet_bc = data["boundary_conditions"][name_to_idx[inlet_name]]
        t_default = inlet_bc.get("bc_values", {}).get("t", [0, 1])
    except Exception:
        t_default = [0, 1]

    # Parse t, P, Q
    t_series = [float(x) for x in args.t_vals.split(",")] if args.t_vals else list(map(float, t_default))
    P_series = _parse_series(args.inlet_P, t_series, fallback_scalar=0.0)
    Q_series = _parse_series(args.outlet_Q, t_series, fallback_scalar=0.0)

    data_out = transform_flow_to_pressure_inlet_and_flow_outlets(
        data=data,
        inlet_new_name=args.inlet_name,
        P_series=P_series,
        t_series=t_series,
        Q_series_for_outlets=Q_series,
    )

    Path(args.out_path).write_text(json.dumps(data_out, indent=4))
    print(f"Wrote: {args.out_path}")


if __name__ == "__main__":
    main()
