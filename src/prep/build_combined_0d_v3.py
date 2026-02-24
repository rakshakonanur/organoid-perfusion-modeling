#!/usr/bin/env python3
"""build_combined_0d.py

Create one large svZeroDSolver-style JSON input (.in) by combining:
  - organoid_#_inlet.in  (synthetic arterial-side networks)
  - organoid_#_outlet.in (synthetic venous-side networks)
  - a main channel built from channel-resistance.xlsx (or a built-in fallback table)

What it does:
  1) For each synthetic organoid file:
     - Prefix-rename every vessel_name, bc_name, and junction_name to guarantee uniqueness
     - Reindex vessel_id so there are no duplicates across files
     - Remove the inlet BC entry:
         * inlet files: removes bc_name="INFLOW" (FLOW) and removes vessel boundary_conditions["inlet"]="INFLOW"
         * outlet files: removes bc_name="PRESSURE_IN" (PRESSURE) and removes vessel boundary_conditions["inlet"]="PRESSURE_IN"
       (All other outlet/terminal BCs remain.)
     - Records a rename map (old->new ids/names) for later plotting.

  2) Builds main-channel vessels from the spreadsheet columns:
       Segment, Length, Capacitance, Resistance/Resitance, Inductance
     and creates boundary conditions:
       * arterial_seg1 inlet -> FLOW bc_name "INFLOW" (default Q=5.5e-05)
       * veinous_seg5 outlet -> PRESSURE bc_name "OUT3" (default P=79993.2)

  3) Creates channel↔organoid junctions (default mapping):
       arterial_seg1 -> (arterial_seg2 + organoid1_inlet_root)
       arterial_seg2 -> (arterial_seg3 + organoid2_inlet_root)
       arterial_seg3 -> (arterial_seg4 + organoid3_inlet_root)
       arterial_seg4 -> (arterial_seg5 + organoid4_inlet_root)
       arterial_seg5 -> u_bend
       u_bend -> veinous_seg1
       veinous_seg1 -> (veinous_seg2 + organoid4_outlet_root)
       veinous_seg2 -> (veinous_seg3 + organoid3_outlet_root)
       veinous_seg3 -> (veinous_seg4 + organoid2_outlet_root)
       veinous_seg4 -> (veinous_seg5 + organoid1_outlet_root)

You can override the channel-organoid junction wiring by providing --connections (JSON).
Run with --print-connections-template to get a starting template.

Outputs:
  - combined .in (JSON) file
  - rename_map.json (for mapping results back to original names/ids)

Usage:
  python build_combined_0d_v2.py \
    --input-dir /path/to/files \
    --channel-xlsx channel-resistance.xlsx \
    --out combined.in \
    --map rename_map.json

Or from a zip of the synthetic files:
  python build_combined_0d_v2.py \
    --synthetic-zip synthetic-0d.zip \
    --channel-xlsx channel-resistance.xlsx \
    --out combined.in \
    --map rename_map.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4))


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    raise TypeError(f"Expected list, got {type(x)}")


def _as_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    raise TypeError(f"Expected dict, got {type(x)}")


def _slug(s: str) -> str:
    # Keep original spaces (your files already use them), but remove weird characters.
    s = str(s).strip()
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^A-Za-z0-9 _\-\.]+", "_", s)
    return s


def _infer_organoid_id(path: Path) -> Optional[int]:
    m = re.search(r"organoid[_\-\s]*([0-9]+)", path.name.lower())
    return int(m.group(1)) if m else None


# -----------------------------
# Rename bookkeeping
# -----------------------------
@dataclass
class RenameRecord:
    file: str
    entity: str  # vessel / bc / junction
    old_id: str
    new_id: str
    old_name: str
    new_name: str


# -----------------------------
# Synthetic processing
# -----------------------------
def _remove_bc_entries(bcs: List[Dict[str, Any]], remove_names: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    removed = []
    keep = []
    remove_set = {n.upper() for n in remove_names}
    for bc in bcs:
        name = str(bc.get("bc_name", ""))
        if name.upper() in remove_set:
            removed.append(name)
            continue
        keep.append(bc)
    return keep, removed


def _find_root_vessel_id(vessels: List[Dict[str, Any]], inlet_bc_names: List[str]) -> Optional[int]:
    target = {n.upper() for n in inlet_bc_names}
    for v in vessels:
        bcmap = _as_dict(v.get("boundary_conditions"))
        inlet = bcmap.get("inlet")
        if isinstance(inlet, str) and inlet.upper() in target:
            return int(v["vessel_id"])
    return None


def process_synthetic_model(
    model: Dict[str, Any],
    file_label: str,
    prefix: str,
    next_vessel_id: int,
    inlet_bc_names_to_remove: List[str],
) -> Tuple[Dict[str, Any], int, List[RenameRecord], Optional[int]]:
    """Rename + reindex + remove inlet BCs."""
    rename_records: List[RenameRecord] = []

    vessels = _as_list(model.get("vessels"))
    junctions = _as_list(model.get("junctions"))
    bcs = _as_list(model.get("boundary_conditions"))

    # root vessel based on inlet BC mapping BEFORE edits
    root_old = _find_root_vessel_id(vessels, inlet_bc_names_to_remove)

    # remove inlet BC entries from BC list
    bcs_kept, removed_bc_names = _remove_bc_entries(bcs, inlet_bc_names_to_remove)

    # rename BCs (kept)
    bc_rename: Dict[str, str] = {}
    bcs_out: List[Dict[str, Any]] = []
    for bc in bcs_kept:
        old = str(bc.get("bc_name", ""))
        new = f"{prefix}{_slug(old)}" if old else f"{prefix}BC"
        bc_rename[old] = new
        bc2 = dict(bc)
        bc2["bc_name"] = new
        bcs_out.append(bc2)
        rename_records.append(RenameRecord(file_label, "bc", "", "", old, new))

    for old in removed_bc_names:
        rename_records.append(RenameRecord(file_label, "bc", "", "", old, "(REMOVED)"))

    # rename junction names
    for j in junctions:
        old = str(j.get("junction_name", ""))
        new = f"{prefix}{_slug(old)}" if old else f"{prefix}J"
        j["junction_name"] = new
        rename_records.append(RenameRecord(file_label, "junction", "", "", old, new))

    # reindex vessels + rename vessel names + update vessel bc maps
    old_to_new: Dict[int, int] = {}
    inlet_remove_set = {n.upper() for n in inlet_bc_names_to_remove}
    for v in vessels:
        old_vid = int(v["vessel_id"])
        new_vid = next_vessel_id
        next_vessel_id += 1
        old_to_new[old_vid] = new_vid

        old_name = str(v.get("vessel_name", f"vessel_{old_vid}"))
        new_name = f"{prefix}{_slug(old_name)}"
        v["vessel_id"] = new_vid
        v["vessel_name"] = new_name
        rename_records.append(RenameRecord(file_label, "vessel", str(old_vid), str(new_vid), old_name, new_name))

        bcmap = _as_dict(v.get("boundary_conditions"))
        inlet = bcmap.get("inlet")
        if isinstance(inlet, str) and inlet.upper() in inlet_remove_set:
            bcmap.pop("inlet", None)

        for k in ("inlet", "outlet"):
            if k in bcmap and isinstance(bcmap[k], str) and bcmap[k] in bc_rename:
                bcmap[k] = bc_rename[bcmap[k]]

        if bcmap:
            v["boundary_conditions"] = bcmap
        else:
            v.pop("boundary_conditions", None)

    # update junction vessel references
    for j in junctions:
        for key in ("inlet_vessels", "outlet_vessels"):
            if key in j and isinstance(j[key], list):
                j[key] = [old_to_new[int(x)] for x in j[key]]

    root_new = old_to_new[root_old] if root_old is not None else None

    model_out = {
        "description": model.get("description", {}),
        "simulation_parameters": model.get("simulation_parameters", {}),
        "boundary_conditions": bcs_out,
        "junctions": junctions,
        "vessels": vessels,
    }
    return model_out, next_vessel_id, rename_records, root_new


# -----------------------------
# Channel building
# -----------------------------
def _read_channel_table(channel_xlsx: Optional[Path]) -> List[Dict[str, float]]:
    """Return list of dict rows with keys: Segment, Length, Resistance, Capacitance, Inductance."""
    if channel_xlsx and channel_xlsx.exists():
        df = pd.read_excel(channel_xlsx)
        df.columns = [str(c).strip() for c in df.columns]
        colmap = {c.lower(): c for c in df.columns}

        def pick(*names: str) -> str:
            for n in names:
                if n.lower() in colmap:
                    return colmap[n.lower()]
            raise KeyError(f"Missing expected column. Looked for {names}. Got: {list(df.columns)}")

        c_seg = pick("Segment", "segment", "name")
        c_len = pick("Length", "length")
        c_R   = pick("Resistance", "Resitance", "resistance", "resitance")
        c_C   = pick("Capacitance", "capacitance", "C", "c")
        c_L   = pick("Inductance", "inductance", "L", "l")

        rows = []
        for _, r in df.iterrows():
            rows.append({
                "Segment": str(r[c_seg]).strip(),
                "Length": float(r[c_len]),
                "Resistance": float(r[c_R]),
                "Capacitance": float(r[c_C]) if not pd.isna(r[c_C]) else 0.0,
                "Inductance": float(r[c_L]) if not pd.isna(r[c_L]) else 0.0,
            })
        return rows

    # Fallback to your pasted table
    return [
        {"Segment":"arterial_seg1", "Length":1.95, "Resistance":7.35e2, "Capacitance":0.0, "Inductance":4.22e1},
        {"Segment":"arterial_seg2", "Length":0.6,  "Resistance":2.26e2, "Capacitance":0.0, "Inductance":1.30e1},
        {"Segment":"arterial_seg3", "Length":0.6,  "Resistance":2.26e2, "Capacitance":0.0, "Inductance":1.30e1},
        {"Segment":"arterial_seg4", "Length":0.6,  "Resistance":2.26e2, "Capacitance":0.0, "Inductance":1.30e1},
        {"Segment":"arterial_seg5", "Length":1.9,  "Resistance":7.16e2, "Capacitance":0.0, "Inductance":4.11e1},
        {"Segment":"u_bend",        "Length":2.021017612, "Resistance":2.86e2, "Capacitance":0.0, "Inductance":2.68e1},
        {"Segment":"veinous_seg1",  "Length":1.9,  "Resistance":1.87e2, "Capacitance":0.0, "Inductance":2.10e1},
        {"Segment":"veinous_seg2",  "Length":0.6,  "Resistance":5.89e1, "Capacitance":0.0, "Inductance":6.63e0},
        {"Segment":"veinous_seg3",  "Length":0.6,  "Resistance":5.89e1, "Capacitance":0.0, "Inductance":6.63e0},
        {"Segment":"veinous_seg4",  "Length":0.6,  "Resistance":5.89e1, "Capacitance":0.0, "Inductance":6.63e0},
        {"Segment":"veinous_seg5",  "Length":1.95, "Resistance":1.91e2, "Capacitance":0.0, "Inductance":2.15e1},
    ]


def build_channel_model(
    rows: List[Dict[str, float]],
    next_vessel_id: int,
    inflow_bc_name: str,
    outpress_bc_name: str,
    inflow_segment: str = "arterial_seg1",
    outpress_segment: str = "veinous_seg5",
    Q_in: float = 5.5e-05,
    P_out: float = 79993.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int], int, List[RenameRecord]]:
    vessels: List[Dict[str, Any]] = []
    bcs: List[Dict[str, Any]] = []
    seg_to_vid: Dict[str, int] = {}
    recs: List[RenameRecord] = []

    for r in rows:
        seg = str(r["Segment"]).strip()
        vid = next_vessel_id
        next_vessel_id += 1
        seg_to_vid[seg] = vid

        vessels.append({
            "vessel_id": vid,
            "vessel_length": float(r["Length"]),
            "vessel_name": f"channel_{seg}",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": float(r["Resistance"]),
                "C": float(r["Capacitance"]),
                "L": float(r["Inductance"]),
                "stenosis_coefficient": 0.0,
            },
        })

    if inflow_segment not in seg_to_vid:
        raise KeyError(f"Channel segment '{inflow_segment}' not found in channel table.")
    if outpress_segment not in seg_to_vid:
        raise KeyError(f"Channel segment '{outpress_segment}' not found in channel table.")

    bcs.append({
        "bc_name": inflow_bc_name,
        "bc_type": "FLOW",
        "bc_values": {"Q": [Q_in, Q_in], "t": [0, 1]},
    })
    bcs.append({
        "bc_name": outpress_bc_name,
        "bc_type": "PRESSURE",
        "bc_values": {"P": [P_out, P_out], "t": [0, 1]},
    })

    inflow_vid = seg_to_vid[inflow_segment]
    out_vid = seg_to_vid[outpress_segment]
    for v in vessels:
        if v["vessel_id"] == inflow_vid:
            v["boundary_conditions"] = {"inlet": inflow_bc_name}
        if v["vessel_id"] == out_vid:
            v.setdefault("boundary_conditions", {})
            v["boundary_conditions"]["outlet"] = outpress_bc_name

    recs.append(RenameRecord("channel", "bc", "", "", "INFLOW", inflow_bc_name))
    recs.append(RenameRecord("channel", "bc", "", "", "OUT_PRESS", outpress_bc_name))

    return vessels, bcs, seg_to_vid, next_vessel_id, recs

def build_leaky_vessels(
    next_vessel_id: int,
    inlet_roots: Dict[int, int],
    outlet_roots: Dict[int, int],
    R_leak: float = 0.0,
    length_leak: float = 0.01,
) -> Tuple[Dict[int, int], Dict[int, int], List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Create simple 'leaky' pseudo-vessels for each organoid inlet/outlet.

    For each organoid id in inlet_roots / outlet_roots we create:
      - an arterial-side leak vessel:  LEAK_ART_<i>
      - a venous-side  leak vessel:   LEAK_VEN_<i>

    Each is a BloodVessel with:
      R_poiseuille = R_leak, C = 0, L = 0,
    and an outlet FLOW BC with Q(t) = 0 for now.

    Returns
    -------
    leak_inlet_vids : dict[org_id -> vessel_id]   # arterial side
    leak_outlet_vids: dict[org_id -> vessel_id]   # venous side
    vessels         : list of new vessel dicts
    bcs             : list of new BC dicts
    next_vessel_id  : updated counter
    """
    leak_inlet_vids: Dict[int, int] = {}
    leak_outlet_vids: Dict[int, int] = {}
    vessels: List[Dict[str, Any]] = []
    bcs: List[Dict[str, Any]] = []

    # Arterial-side leaks (organoid inlets)
    for org_id in sorted(inlet_roots.keys()):
        vid = next_vessel_id
        next_vessel_id += 1
        leak_inlet_vids[org_id] = vid

        bc_name = f"LEAK_ART_{org_id}"
        vessels.append({
            "vessel_id": vid,
            "vessel_length": float(length_leak),
            "vessel_name": f"leak_art_{org_id}",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": float(R_leak),
                "C": 0.0,
                "L": 0.0,
                "stenosis_coefficient": 0.0,
            },
            "boundary_conditions": {
                "outlet": bc_name
            },
        })
        bcs.append({
            "bc_name": bc_name,
            "bc_type": "FLOW",
            "bc_values": {"Q": [0.0, 0.0], "t": [0.0, 1.0]},
        })

    # Venous-side leaks (organoid outlets)
    for org_id in sorted(outlet_roots.keys()):
        vid = next_vessel_id
        next_vessel_id += 1
        leak_outlet_vids[org_id] = vid

        bc_name = f"LEAK_VEN_{org_id}"
        vessels.append({
            "vessel_id": vid,
            "vessel_length": float(length_leak),
            "vessel_name": f"leak_ven_{org_id}",
            "zero_d_element_type": "BloodVessel",
            "zero_d_element_values": {
                "R_poiseuille": float(R_leak),
                "C": 0.0,
                "L": 0.0,
                "stenosis_coefficient": 0.0,
            },
            "boundary_conditions": {
                "outlet": bc_name
            },
        })
        bcs.append({
            "bc_name": bc_name,
            "bc_type": "FLOW",
            "bc_values": {"Q": [0.0, 0.0], "t": [0.0, 1.0]},
        })

    return leak_inlet_vids, leak_outlet_vids, vessels, bcs, next_vessel_id

# -----------------------------
# Channel junction wiring
# -----------------------------
def build_default_channel_junctions(
    seg_to_vid: Dict[str, int],
    inlet_roots: Dict[int, int],
    outlet_roots: Dict[int, int],
    leak_inlet_vids: Optional[Dict[int, int]] = None,
    leak_outlet_vids: Optional[Dict[int, int]] = None,
) -> List[Dict[str, Any]]:
    leak_inlet_vids = leak_inlet_vids or {}
    leak_outlet_vids = leak_outlet_vids or {}

    def vid(seg: str) -> int:
        if seg not in seg_to_vid:
            raise KeyError(f"Missing channel segment '{seg}' in channel model.")
        return seg_to_vid[seg]

    junctions: List[Dict[str, Any]] = []

    # Arterial: organoid 1..4
    art_pairs = [
        ("arterial_seg1", "arterial_seg2", 1),
        ("arterial_seg2", "arterial_seg3", 2),
        ("arterial_seg3", "arterial_seg4", 3),
        ("arterial_seg4", "arterial_seg5", 4),
    ]
    for idx, (up, dn, org) in enumerate(art_pairs, start=1):
        outs = [vid(dn), inlet_roots[org]]
        # Add arterial-side leak branch if defined for this organoid
        if org in leak_inlet_vids:
            outs.append(leak_inlet_vids[org])
        junctions.append({
            "junction_name": f"J_ART_{idx}",
            "junction_type": "NORMAL_JUNCTION",
            "inlet_vessels": [vid(up)],
            "outlet_vessels": outs,
        })

    # U-bend and venous chain
    junctions.append({
        "junction_name": "J_ART_TO_UBEND",
        "junction_type": "NORMAL_JUNCTION",
        "inlet_vessels": [vid("arterial_seg5")],
        "outlet_vessels": [vid("u_bend")],
    })
    junctions.append({
        "junction_name": "J_UBEND_TO_VEN1",
        "junction_type": "NORMAL_JUNCTION",
        "inlet_vessels": [vid("u_bend")],
        "outlet_vessels": [vid("veinous_seg1")],
    })

    # Venous: organoid 4..1
    ven_pairs = [
        ("veinous_seg1", "veinous_seg2", 4),
        ("veinous_seg2", "veinous_seg3", 3),
        ("veinous_seg3", "veinous_seg4", 2),
        ("veinous_seg4", "veinous_seg5", 1),
    ]
    for idx, (up, dn, org) in enumerate(ven_pairs, start=1):
        outs = [vid(dn), outlet_roots[org]]
        # Add venous-side leak branch if defined for this organoid
        if org in leak_outlet_vids:
            outs.append(leak_outlet_vids[org])
        junctions.append({
            "junction_name": f"J_VEN_{idx}",
            "junction_type": "NORMAL_JUNCTION",
            "inlet_vessels": [vid(up)],
            "outlet_vessels": outs,
        })

    # Defensive unique names
    seen = set()
    for j in junctions:
        name = j["junction_name"]
        if name in seen:
            k = 2
            while f"{name}_{k}" in seen:
                k += 1
            j["junction_name"] = f"{name}_{k}"
        seen.add(j["junction_name"])
    return junctions


def print_connections_template() -> str:
    tpl = {
        "junctions": [
            {
                "junction_name": "J_ART_1",
                "junction_type": "NORMAL_JUNCTION",
                "inlet_vessels": ["<channel vessel_id>"],
                "outlet_vessels": ["<channel vessel_id>", "<organoid inlet root vessel_id>"],
            }
        ]
    }
    return json.dumps(tpl, indent=4)


def load_connections_override(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and isinstance(data.get("junctions"), list):
        return data["junctions"]
    raise ValueError("Connections override must be JSON with top-level key 'junctions' (a list).")


# -----------------------------
# Input discovery
# -----------------------------
def find_organoid_files(input_dir: Path) -> Dict[int, Dict[str, Path]]:
    organoids: Dict[int, Dict[str, Path]] = {}
    for p in sorted(input_dir.rglob("organoid_*_*.in")):
        oid = _infer_organoid_id(p)
        if oid is None:
            continue
        name = p.name.lower()
        side = "inlet" if "inlet" in name else ("outlet" if "outlet" in name else "")
        if not side:
            continue
        organoids.setdefault(oid, {})[side] = p

    missing = []
    for oid, d in sorted(organoids.items()):
        if "inlet" not in d or "outlet" not in d:
            missing.append(oid)
    if missing:
        raise RuntimeError(
            f"Missing inlet/outlet file for organoids: {missing}. Expected organoid_#_inlet.in and organoid_#_outlet.in"
        )
    return organoids


def extract_zip_to_temp(zip_path: Path) -> Path:
    td = Path(tempfile.mkdtemp(prefix="synthetic_0d_"))
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(td)
    return td


# -----------------------------
# Validation
# -----------------------------
def validate_model(model: Dict[str, Any]) -> None:
    vessels = _as_list(model.get("vessels"))
    junctions = _as_list(model.get("junctions"))
    bcs = _as_list(model.get("boundary_conditions"))

    vids = [int(v["vessel_id"]) for v in vessels]
    if len(vids) != len(set(vids)):
        raise RuntimeError("Duplicate vessel_id detected in combined model.")

    bc_names = [str(b["bc_name"]) for b in bcs]
    if len(bc_names) != len(set(bc_names)):
        raise RuntimeError("Duplicate bc_name detected in combined model.")

    jn = [str(j["junction_name"]) for j in junctions]
    if len(jn) != len(set(jn)):
        raise RuntimeError("Duplicate junction_name detected in combined model.")

    bc_set = set(bc_names)
    for v in vessels:
        bcmap = v.get("boundary_conditions")
        if isinstance(bcmap, dict):
            for k in ("inlet", "outlet"):
                if k in bcmap and bcmap[k] not in bc_set:
                    raise RuntimeError(f"Vessel {v.get('vessel_name')} references missing bc_name: {bcmap[k]}")
    vid_set = set(vids)
    for j in junctions:
        for key in ("inlet_vessels", "outlet_vessels"):
            for x in j.get(key, []):
                if int(x) not in vid_set:
                    raise RuntimeError(f"Junction {j.get('junction_name')} references missing vessel_id: {x}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=str, default="/Users/rakshakonanur/Documents/Research/two-channel/input/trial-10", help="Directory containing organoid_#_inlet.in and organoid_#_outlet.in")
    ap.add_argument("--synthetic-zip", type=str, default="", help="Zip containing organoid files (alternative to --input-dir)")
    ap.add_argument("--channel-xlsx", type=str, default="/Users/rakshakonanur/Documents/Research/two-channel/files/channel-resistance-sv.xlsx", help="Path to channel-resistance.xlsx (optional; fallback table used if missing)")
    ap.add_argument("--out", type=str, default="combined.in", help="Output combined .in file (JSON)")
    ap.add_argument("--map", type=str, default="rename_map.json", help="Output rename_map.json")

    ap.add_argument("--main-inflow-name", type=str, default="INFLOW")
    ap.add_argument("--main-outlet-name", type=str, default="OUT3")
    ap.add_argument("--main-inflow-Q", type=float, default=2.5e-2)
    ap.add_argument("--main-outlet-P", type=float, default=0.0)

    ap.add_argument("--connections", type=str, default="", help="Optional JSON file overriding channel-organoid junctions")
    ap.add_argument("--print-connections-template", action="store_true", help="Print a JSON template for --connections and exit")
    args = ap.parse_args()

    if args.print_connections_template:
        print(print_connections_template())
        return

    out_path = Path(args.out).resolve()
    map_path = Path(args.map).resolve()

    temp_dir: Optional[Path] = None
    try:
        if args.synthetic_zip:
            zip_path = Path(args.synthetic_zip).resolve()
            if not zip_path.exists():
                raise FileNotFoundError(zip_path)
            temp_dir = extract_zip_to_temp(zip_path)
            input_dir = temp_dir
        else:
            if not args.input_dir:
                raise ValueError("Provide either --input-dir or --synthetic-zip")
            input_dir = Path(args.input_dir).resolve()
            if not input_dir.exists():
                raise FileNotFoundError(input_dir)

        organoids = find_organoid_files(input_dir)

        first_model = _load_json(organoids[min(organoids.keys())]["inlet"])
        combined: Dict[str, Any] = {
            "description": {"name": "combined_channel_plus_organoids"},
            "simulation_parameters": first_model.get("simulation_parameters", {}),
            "boundary_conditions": [],
            "junctions": [],
            "vessels": [],
        }

        rename_records: List[RenameRecord] = []
        next_vid = 0
        inlet_roots: Dict[int, int] = {}
        outlet_roots: Dict[int, int] = {}

        # Process synthetic organoid files
        for org_id in sorted(organoids.keys()):
            for side in ("inlet", "outlet"):
                path = organoids[org_id][side]
                model = _load_json(path)

                inlet_remove = ["INFLOW"] if side == "inlet" else ["PRESSURE_IN"]
                prefix = f"organoid{org_id}_{side}_"

                processed, next_vid, recs, root_new = process_synthetic_model(
                    model=model,
                    file_label=path.name,
                    prefix=prefix,
                    next_vessel_id=next_vid,
                    inlet_bc_names_to_remove=inlet_remove,
                )

                rename_records.extend(recs)
                combined["boundary_conditions"].extend(_as_list(processed.get("boundary_conditions")))
                combined["junctions"].extend(_as_list(processed.get("junctions")))
                combined["vessels"].extend(_as_list(processed.get("vessels")))

                if root_new is None:
                    raise RuntimeError(f"Could not find root vessel in {path.name} (expected inlet BC {inlet_remove}).")
                if side == "inlet":
                    inlet_roots[org_id] = root_new
                else:
                    outlet_roots[org_id] = root_new

        # Build channel model
                channel_xlsx = Path(args.channel_xlsx).resolve() if args.channel_xlsx else None
        channel_rows = _read_channel_table(channel_xlsx)
        channel_vessels, channel_bcs, seg_to_vid, next_vid, channel_recs = build_channel_model(
            rows=channel_rows,
            next_vessel_id=next_vid,
            inflow_bc_name=args.main_inflow_name,
            outpress_bc_name=args.main_outlet_name,
            Q_in=args.main_inflow_Q,
            P_out=args.main_outlet_P,
        )
        rename_records.extend(channel_recs)
        combined["vessels"].extend(channel_vessels)
        combined["boundary_conditions"].extend(channel_bcs)

        # --------------------------------------------------
        # Add 'leaky' pseudo-vessels branching off the
        # organoid inlet/outlet junctions.
        # These start with Q(t) = 0 and can be overwritten
        # later when coupling to the 3D gel model.
        # --------------------------------------------------
        leak_inlet_vids, leak_outlet_vids, leak_vessels, leak_bcs, next_vid = build_leaky_vessels(
            next_vessel_id=next_vid,
            inlet_roots=inlet_roots,
            outlet_roots=outlet_roots,
        )
        combined["vessels"].extend(leak_vessels)
        combined["boundary_conditions"].extend(leak_bcs)

        # Channel junctions
        if args.connections:
            junctions_channel = load_connections_override(Path(args.connections).resolve())
        else:
            junctions_channel = build_default_channel_junctions(
                seg_to_vid, inlet_roots, outlet_roots,
                leak_inlet_vids=leak_inlet_vids,
                leak_outlet_vids=leak_outlet_vids,
            )
        combined["junctions"].extend(junctions_channel)

        validate_model(combined)
        _save_json(out_path, combined)

        rename_map = {
            "records": [
                {
                    "file": r.file,
                    "entity": r.entity,
                    "old_id": r.old_id,
                    "new_id": r.new_id,
                    "old_name": r.old_name,
                    "new_name": r.new_name,
                }
                for r in rename_records
            ]
        }
        _save_json(map_path, rename_map)

    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
