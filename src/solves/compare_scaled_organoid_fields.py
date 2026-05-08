#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _import_numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise SystemExit(
            "This script requires numpy. Please run it in the same Python "
            "environment you use for the Darcy solves."
        ) from exc
    return np


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit(
            "This script requires h5py to read/write the Darcy XDMF sidecars. "
            "Please run it in the same Python environment you use for the Darcy solves."
        ) from exc
    return h5py


def _import_ckdtree():
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise SystemExit(
            "This script requires scipy for nearest-neighbor field sampling. "
            "Please run it in the same Python environment you use for the Darcy solves."
        ) from exc
    return cKDTree


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _read_component_csv(csv_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _group_component_rows(rows: list[dict[str, str]], np: Any) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        name = row["name"]
        grouped.setdefault(
            name,
            {
                "time": [],
                "flow_in": [],
                "flow_out": [],
                "pressure_in": [],
                "pressure_out": [],
            },
        )
        grouped[name]["time"].append(float(row["time"]))
        grouped[name]["flow_in"].append(float(row["flow_in"]))
        grouped[name]["flow_out"].append(float(row["flow_out"]))
        grouped[name]["pressure_in"].append(float(row["pressure_in"]))
        grouped[name]["pressure_out"].append(float(row["pressure_out"]))

    out: dict[str, dict[str, Any]] = {}
    for name, data in grouped.items():
        out[name] = {key: np.asarray(val, dtype=float) for key, val in data.items()}
    return out


def _load_0d_groups(run_dir: Path, np: Any) -> dict[str, dict[str, Any]]:
    candidates = [run_dir / "split" / "other.csv", run_dir / "output.csv"]
    csv_path = next((p for p in candidates if p.exists()), None)
    if csv_path is None:
        raise FileNotFoundError(f"Could not find 0D CSV in {run_dir}")
    return _group_component_rows(_read_component_csv(csv_path), np)


def _safe_mean(values: Any) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _load_0d_organoid_summary(
    grouped: dict[str, dict[str, Any]],
    organoid_id: int,
    np: Any,
) -> dict[str, Any]:
    art_name = f"leak_art_{organoid_id}"
    ven_name = f"leak_ven_{organoid_id}"
    if art_name not in grouped:
        raise KeyError(f"Could not find {art_name} in 0D CSV")
    if ven_name not in grouped:
        raise KeyError(f"Could not find {ven_name} in 0D CSV")

    art = grouped[art_name]
    ven = grouped[ven_name]
    if art["time"].shape != ven["time"].shape or not np.allclose(art["time"], ven["time"]):
        raise ValueError(
            f"Time samples for {art_name} and {ven_name} do not align. "
            "This scaling test expects matching times."
        )

    p_art = np.asarray(art["pressure_in"], dtype=float)
    p_ven = np.asarray(ven["pressure_in"], dtype=float)
    q_art = np.asarray(art["flow_in"], dtype=float)
    q_ven = np.asarray(ven["flow_in"], dtype=float)
    delta_p = p_art - p_ven

    return {
        "organoid_id": organoid_id,
        "time": np.asarray(art["time"], dtype=float),
        "p_art": p_art,
        "p_ven": p_ven,
        "q_art": q_art,
        "q_ven": q_ven,
        "delta_p": delta_p,
        "p_art_mean": _safe_mean(p_art),
        "p_ven_mean": _safe_mean(p_ven),
        "q_art_mean": _safe_mean(q_art),
        "q_ven_mean": _safe_mean(q_ven),
        "delta_p_mean": _safe_mean(delta_p),
        "delta_p_std": float(delta_p.std()) if len(delta_p) else float("nan"),
    }


def _parse_hdf_ref(xdmf_path: Path, text: str) -> tuple[Path, str]:
    rel_name, dataset = text.strip().split(":", 1)
    return (xdmf_path.parent / rel_name).resolve(), dataset


def _load_xdmf_field(xdmf_path: Path, np: Any, h5py: Any) -> dict[str, Any]:
    root = ET.parse(xdmf_path).getroot()
    uniform_grid = root.find("./Domain/Grid[@GridType='Uniform']")
    if uniform_grid is None:
        raise RuntimeError(f"Could not find Uniform grid in {xdmf_path}")

    topology = uniform_grid.find("Topology")
    geometry = uniform_grid.find("Geometry")
    if topology is None or geometry is None:
        raise RuntimeError(f"Missing topology/geometry in {xdmf_path}")

    topo_item = topology.find("DataItem")
    geom_item = geometry.find("DataItem")
    if topo_item is None or geom_item is None or topo_item.text is None or geom_item.text is None:
        raise RuntimeError(f"Missing topology/geometry data item in {xdmf_path}")

    collection = root.find("./Domain/Grid[@GridType='Collection']")
    if collection is None:
        raise RuntimeError(f"Could not find temporal collection in {xdmf_path}")
    field_grid = collection.findall("./Grid")[-1]
    attribute = field_grid.find("Attribute")
    if attribute is None:
        raise RuntimeError(f"Could not find Attribute in {xdmf_path}")
    attr_item = attribute.find("DataItem")
    if attr_item is None or attr_item.text is None:
        raise RuntimeError(f"Missing attribute data item in {xdmf_path}")

    topo_file, topo_dataset = _parse_hdf_ref(xdmf_path, topo_item.text)
    geom_file, geom_dataset = _parse_hdf_ref(xdmf_path, geom_item.text)
    val_file, val_dataset = _parse_hdf_ref(xdmf_path, attr_item.text)

    with h5py.File(topo_file, "r") as f:
        topology_data = f[topo_dataset][...]
    with h5py.File(geom_file, "r") as f:
        coords = f[geom_dataset][...]
    with h5py.File(val_file, "r") as f:
        values = f[val_dataset][...]

    return {
        "xdmf_path": xdmf_path,
        "coords": np.asarray(coords, dtype=float),
        "topology": np.asarray(topology_data),
        "values": np.asarray(values, dtype=float),
        "topology_type": topology.attrib.get("TopologyType", "Tetrahedron"),
        "attribute_type": attribute.attrib.get("AttributeType", "Scalar"),
        "attribute_name": attribute.attrib.get("Name", "f"),
        "attribute_center": attribute.attrib.get("Center", "Node"),
    }


def _write_xdmf_field(
    out_xdmf: Path,
    coords: Any,
    topology: Any,
    values: Any,
    topology_type: str,
    attribute_type: str,
    attribute_name: str,
    attribute_center: str,
    h5py: Any,
) -> None:
    out_xdmf.parent.mkdir(parents=True, exist_ok=True)
    out_h5 = out_xdmf.with_suffix(".h5")

    scalar = str(attribute_type).lower() == "scalar"
    values_arr = values.reshape(-1, 1) if scalar and values.ndim == 1 else values

    with h5py.File(out_h5, "w") as f:
        mesh = f.create_group("Mesh")
        grid = mesh.create_group("Grid")
        grid.create_dataset("topology", data=topology)
        grid.create_dataset("geometry", data=coords)

        fun = f.create_group("Function")
        field = fun.create_group(attribute_name)
        field.create_dataset("0", data=values_arr)

    xdmf_text = f"""<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="https://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="Grid" GridType="Uniform">
      <Topology TopologyType="{topology_type}" NumberOfElements="{topology.shape[0]}" NodesPerElement="{topology.shape[1]}">
        <DataItem Dimensions="{topology.shape[0]} {topology.shape[1]}" NumberType="Int" Format="HDF">{out_h5.name}:/Mesh/Grid/topology</DataItem>
      </Topology>
      <Geometry GeometryType="XYZ">
        <DataItem Dimensions="{coords.shape[0]} {coords.shape[1]}" Format="HDF">{out_h5.name}:/Mesh/Grid/geometry</DataItem>
      </Geometry>
    </Grid>
    <Grid Name="{attribute_name}" GridType="Collection" CollectionType="Temporal">
      <Grid Name="{attribute_name}" GridType="Uniform">
        <xi:include xpointer="xpointer(/Xdmf/Domain/Grid[@GridType='Uniform'][1]/*[self::Topology or self::Geometry])" />
        <Time Value="0" />
        <Attribute Name="{attribute_name}" AttributeType="{attribute_type}" Center="{attribute_center}">
          <DataItem Dimensions="{' '.join(str(v) for v in values_arr.shape)}" Format="HDF">{out_h5.name}:/Function/{attribute_name}/0</DataItem>
        </Attribute>
      </Grid>
    </Grid>
  </Domain>
</Xdmf>
"""
    out_xdmf.write_text(xdmf_text)


def _compute_metrics(actual: Any, predicted: Any, np: Any) -> dict[str, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    diff = predicted_arr - actual_arr

    rmse = float(np.sqrt(np.mean(diff**2)))
    mae = float(np.mean(np.abs(diff)))
    max_abs = float(np.max(np.abs(diff)))
    actual_rms = float(np.sqrt(np.mean(actual_arr**2)))
    predicted_mean = float(np.mean(predicted_arr))
    actual_mean = float(np.mean(actual_arr))
    nrmse = rmse / actual_rms if actual_rms > 0.0 else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "max_abs_error": max_abs,
        "nrmse_rms_scaled": nrmse,
        "actual_rms": actual_rms,
        "actual_mean": actual_mean,
        "predicted_mean": predicted_mean,
    }


def _field_support_coords(field: dict[str, Any], np: Any) -> Any:
    center = str(field.get("attribute_center", "Node")).strip().lower()
    coords = np.asarray(field["coords"], dtype=float)

    if center == "node":
        return coords
    if center == "cell":
        topology = np.asarray(field["topology"])
        if topology.ndim != 2:
            raise ValueError(
                f"Expected 2D topology for cell-centered field {field['xdmf_path']}, got shape {topology.shape}"
            )
        return coords[topology].mean(axis=1)

    raise ValueError(
        f"Unsupported attribute center {field.get('attribute_center')!r} in {field['xdmf_path']}"
    )


def _point_mapping(
    ref_coords: Any,
    target_coords: Any,
    shift: Any,
    coord_tol: float,
    np: Any,
    cKDTree: Any,
) -> tuple[Any, float]:
    shifted = np.asarray(ref_coords, dtype=float) + np.asarray(shift, dtype=float)
    target = np.asarray(target_coords, dtype=float)

    if shifted.shape == target.shape:
        direct_residual = float(np.max(np.abs(shifted - target)))
        if direct_residual <= coord_tol:
            indices = np.arange(target.shape[0], dtype=int)
            return indices, direct_residual

    tree = cKDTree(shifted)
    distances, indices = tree.query(target)
    max_distance = float(np.max(distances))
    return np.asarray(indices, dtype=int), max_distance


def _shift_for_organoid(reference_organoid: int, target_organoid: int, y_shift_step: float) -> tuple[float, float, float]:
    return (0.0, -y_shift_step * float(target_organoid - reference_organoid), 0.0)


def _metric_row(organoid_id: int, category: str, quantity: str, metrics: dict[str, float], **extra: Any) -> dict[str, Any]:
    row = {
        "organoid_id": organoid_id,
        "category": category,
        "quantity": quantity,
    }
    row.update(metrics)
    row.update(extra)
    return row


def _pressure_transform_from_0d(
    reference_zero_d: dict[str, Any],
    target_zero_d: dict[str, Any],
    offset_anchor: str,
) -> dict[str, float]:
    scale_factor = float(target_zero_d["delta_p_mean"] / reference_zero_d["delta_p_mean"])

    arterial_channel_drop = float(reference_zero_d["p_art_mean"] - target_zero_d["p_art_mean"])
    venous_channel_drop = float(reference_zero_d["p_ven_mean"] - target_zero_d["p_ven_mean"])

    target_arterial_pressure = float(reference_zero_d["p_art_mean"] - arterial_channel_drop)
    target_venous_pressure = float(reference_zero_d["p_ven_mean"] - venous_channel_drop)

    offset_from_artery = target_arterial_pressure - scale_factor * float(reference_zero_d["p_art_mean"])
    offset_from_vein = target_venous_pressure - scale_factor * float(reference_zero_d["p_ven_mean"])

    anchor = str(offset_anchor).strip().lower()
    if anchor == "arterial":
        pressure_offset = offset_from_artery
    elif anchor == "venous":
        pressure_offset = offset_from_vein
    else:
        raise ValueError(f"Unsupported pressure offset anchor {offset_anchor!r}; expected 'arterial' or 'venous'")

    return {
        "scale_factor": scale_factor,
        "offset_anchor": anchor,
        "arterial_channel_drop": arterial_channel_drop,
        "venous_channel_drop": venous_channel_drop,
        "target_arterial_pressure": target_arterial_pressure,
        "target_venous_pressure": target_venous_pressure,
        "offset_from_artery": offset_from_artery,
        "offset_from_vein": offset_from_vein,
        "pressure_offset": pressure_offset,
        "offset_mismatch": offset_from_artery - offset_from_vein,
    }


def _interface_metric_rows(
    organoid_id: int,
    actual_iface: dict[str, Any],
    ref_iface: dict[str, Any],
    scale_factor: float,
    pressure_offset: float,
    shift: tuple[float, float, float],
    np: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    pressure_scalar_keys = [
        "p_inlet_mean",
        "p_outlet_mean",
    ]
    nonpressure_scalar_keys = [
        "q_artery_leak",
        "q_venous_leak",
    ]
    pressure_array_keys = [
        "p_inlet_nodes",
        "p_outlet_nodes",
        "p_inlet_target",
        "p_outlet_target",
    ]
    nonpressure_array_keys = [
        "q_inlet",
        "q_inlet_inward",
        "q_inlet_target",
        "q_inlet_residual",
        "q_outlet",
    ]
    vector_keys = [
        "u_inlet_nodes",
        "u_outlet_nodes",
    ]

    for key in pressure_scalar_keys:
        if key in ref_iface and key in actual_iface:
            predicted = float(pressure_offset) + float(scale_factor) * float(ref_iface[key])
            actual = float(actual_iface[key])
            rows.append(
                _metric_row(
                    organoid_id,
                    "interface_bc",
                    key,
                    _compute_metrics(np.asarray([actual]), np.asarray([predicted]), np),
                    scale_factor=scale_factor,
                    pressure_offset=pressure_offset,
                )
            )

    for key in nonpressure_scalar_keys:
        if key in ref_iface and key in actual_iface:
            predicted = float(scale_factor) * float(ref_iface[key])
            actual = float(actual_iface[key])
            rows.append(
                _metric_row(
                    organoid_id,
                    "interface_bc",
                    key,
                    _compute_metrics(np.asarray([actual]), np.asarray([predicted]), np),
                    scale_factor=scale_factor,
                    pressure_offset=pressure_offset,
                )
            )

    for key in pressure_array_keys:
        if key in ref_iface and key in actual_iface:
            predicted = pressure_offset + scale_factor * np.asarray(ref_iface[key], dtype=float)
            actual = np.asarray(actual_iface[key], dtype=float)
            rows.append(
                _metric_row(
                    organoid_id,
                    "interface_bc",
                    key,
                    _compute_metrics(actual, predicted, np),
                    scale_factor=scale_factor,
                    pressure_offset=pressure_offset,
                )
            )

    for key in nonpressure_array_keys:
        if key in ref_iface and key in actual_iface:
            predicted = scale_factor * np.asarray(ref_iface[key], dtype=float)
            actual = np.asarray(actual_iface[key], dtype=float)
            rows.append(
                _metric_row(
                    organoid_id,
                    "interface_bc",
                    key,
                    _compute_metrics(actual, predicted, np),
                    scale_factor=scale_factor,
                    pressure_offset=pressure_offset,
                )
            )

    for key in vector_keys:
        if key in ref_iface and key in actual_iface:
            predicted = scale_factor * np.asarray(ref_iface[key], dtype=float)
            actual = np.asarray(actual_iface[key], dtype=float)
            rows.append(
                _metric_row(
                    organoid_id,
                    "interface_bc",
                    key,
                    _compute_metrics(actual, predicted, np),
                    scale_factor=scale_factor,
                    pressure_offset=pressure_offset,
                )
            )
            rows.append(
                _metric_row(
                    organoid_id,
                    "interface_bc",
                    f"{key}_magnitude",
                    _compute_metrics(
                        np.linalg.norm(actual, axis=1),
                        np.linalg.norm(predicted, axis=1),
                        np,
                    ),
                    scale_factor=scale_factor,
                    pressure_offset=pressure_offset,
                )
            )

    for key in ("coords_inlet", "coords_outlet"):
        if key in ref_iface and key in actual_iface:
            predicted = np.asarray(ref_iface[key], dtype=float) + np.asarray(shift, dtype=float)
            actual = np.asarray(actual_iface[key], dtype=float)
            rows.append(
                _metric_row(
                    organoid_id,
                    "interface_bc",
                    key,
                    _compute_metrics(actual, predicted, np),
                    scale_factor=scale_factor,
                    pressure_offset=pressure_offset,
                )
            )

    return rows


def _field_metric_rows(
    organoid_id: int,
    ref_p: dict[str, Any],
    ref_u: dict[str, Any],
    tgt_p: dict[str, Any],
    tgt_u: dict[str, Any],
    scale_factor: float,
    pressure_offset: float,
    shift: tuple[float, float, float],
    coord_tol: float,
    np: Any,
    cKDTree: Any,
) -> tuple[list[dict[str, Any]], Any, Any, float, float]:
    p_map, p_dist = _point_mapping(
        _field_support_coords(ref_p, np),
        _field_support_coords(tgt_p, np),
        shift,
        coord_tol,
        np,
        cKDTree,
    )
    u_map, u_dist = _point_mapping(
        _field_support_coords(ref_u, np),
        _field_support_coords(tgt_u, np),
        shift,
        coord_tol,
        np,
        cKDTree,
    )

    predicted_p = pressure_offset + scale_factor * np.asarray(ref_p["values"], dtype=float)[p_map]
    predicted_u = scale_factor * np.asarray(ref_u["values"], dtype=float)[u_map]
    actual_p = np.asarray(tgt_p["values"], dtype=float)
    actual_u = np.asarray(tgt_u["values"], dtype=float)

    rows = [
        _metric_row(
            organoid_id,
            "field",
            "pressure",
            _compute_metrics(actual_p, predicted_p, np),
            scale_factor=scale_factor,
            pressure_offset=pressure_offset,
            max_mapping_distance=p_dist,
        ),
        _metric_row(
            organoid_id,
            "field",
            "velocity_components",
            _compute_metrics(actual_u, predicted_u, np),
            scale_factor=scale_factor,
            pressure_offset=pressure_offset,
            max_mapping_distance=u_dist,
        ),
        _metric_row(
            organoid_id,
            "field",
            "velocity_magnitude",
            _compute_metrics(
                np.linalg.norm(actual_u, axis=1),
                np.linalg.norm(predicted_u, axis=1),
                np,
            ),
            scale_factor=scale_factor,
            pressure_offset=pressure_offset,
            max_mapping_distance=u_dist,
        ),
    ]
    return rows, predicted_p, predicted_u, p_dist, u_dist


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_order: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        headers = preferred_order or []
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
        return

    fieldnames: list[str] = []
    if preferred_order:
        for key in preferred_order:
            if any(key in row for row in rows):
                fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_zero_d_rows(
    organoid_summaries: dict[int, dict[str, Any]],
    reference_organoid: int,
    pressure_offset_anchor: str,
    np: Any,
) -> list[dict[str, Any]]:
    ref = organoid_summaries[reference_organoid]
    rows: list[dict[str, Any]] = []
    for organoid_id, data in sorted(organoid_summaries.items()):
        pressure_transform = _pressure_transform_from_0d(ref, data, pressure_offset_anchor)
        scale_factor = pressure_transform["scale_factor"]
        predicted_art_pressure = pressure_transform["pressure_offset"] + scale_factor * ref["p_art_mean"]
        predicted_ven_pressure = pressure_transform["pressure_offset"] + scale_factor * ref["p_ven_mean"]
        predicted_art_flow = scale_factor * ref["q_art_mean"]
        predicted_ven_flow = scale_factor * ref["q_ven_mean"]

        rows.extend(
            [
                {
                    "organoid_id": organoid_id,
                    "quantity": "leak_art_pressure",
                    "scale_factor": scale_factor,
                    "offset_anchor": pressure_transform["offset_anchor"],
                    "pressure_offset": pressure_transform["pressure_offset"],
                    "actual_value": data["p_art_mean"],
                    "predicted_scaled_value": predicted_art_pressure,
                    **_compute_metrics(
                        np.asarray([data["p_art_mean"]]),
                        np.asarray([predicted_art_pressure]),
                        np,
                    ),
                },
                {
                    "organoid_id": organoid_id,
                    "quantity": "leak_ven_pressure",
                    "scale_factor": scale_factor,
                    "offset_anchor": pressure_transform["offset_anchor"],
                    "pressure_offset": pressure_transform["pressure_offset"],
                    "actual_value": data["p_ven_mean"],
                    "predicted_scaled_value": predicted_ven_pressure,
                    **_compute_metrics(
                        np.asarray([data["p_ven_mean"]]),
                        np.asarray([predicted_ven_pressure]),
                        np,
                    ),
                },
                {
                    "organoid_id": organoid_id,
                    "quantity": "leak_art_flow",
                    "scale_factor": scale_factor,
                    "offset_anchor": pressure_transform["offset_anchor"],
                    "pressure_offset": pressure_transform["pressure_offset"],
                    "actual_value": data["q_art_mean"],
                    "predicted_scaled_value": predicted_art_flow,
                    **_compute_metrics(
                        np.asarray([data["q_art_mean"]]),
                        np.asarray([predicted_art_flow]),
                        np,
                    ),
                },
                {
                    "organoid_id": organoid_id,
                    "quantity": "leak_ven_flow",
                    "scale_factor": scale_factor,
                    "offset_anchor": pressure_transform["offset_anchor"],
                    "pressure_offset": pressure_transform["pressure_offset"],
                    "actual_value": data["q_ven_mean"],
                    "predicted_scaled_value": predicted_ven_flow,
                    **_compute_metrics(
                        np.asarray([data["q_ven_mean"]]),
                        np.asarray([predicted_ven_flow]),
                        np,
                    ),
                },
                {
                    "organoid_id": organoid_id,
                    "quantity": "delta_p",
                    "scale_factor": scale_factor,
                    "offset_anchor": pressure_transform["offset_anchor"],
                    "pressure_offset": pressure_transform["pressure_offset"],
                    "actual_value": data["delta_p_mean"],
                    "predicted_scaled_value": scale_factor * ref["delta_p_mean"],
                    "delta_p_std": data["delta_p_std"],
                },
                {
                    "organoid_id": organoid_id,
                    "quantity": "arterial_channel_drop",
                    "scale_factor": scale_factor,
                    "offset_anchor": pressure_transform["offset_anchor"],
                    "pressure_offset": pressure_transform["pressure_offset"],
                    "actual_value": pressure_transform["arterial_channel_drop"],
                    "predicted_scaled_value": pressure_transform["arterial_channel_drop"],
                },
                {
                    "organoid_id": organoid_id,
                    "quantity": "venous_channel_drop",
                    "scale_factor": scale_factor,
                    "offset_anchor": pressure_transform["offset_anchor"],
                    "pressure_offset": pressure_transform["pressure_offset"],
                    "actual_value": pressure_transform["venous_channel_drop"],
                    "predicted_scaled_value": pressure_transform["venous_channel_drop"],
                },
                {
                    "organoid_id": organoid_id,
                    "quantity": "pressure_offset_consistency_gap",
                    "scale_factor": scale_factor,
                    "offset_anchor": pressure_transform["offset_anchor"],
                    "pressure_offset": pressure_transform["pressure_offset"],
                    "actual_value": pressure_transform["offset_mismatch"],
                    "predicted_scaled_value": pressure_transform["offset_mismatch"],
                },
            ]
        )
    return rows


def _plot_summaries(
    out_dir: Path,
    zero_d_rows: list[dict[str, Any]],
    field_rows: list[dict[str, Any]],
    interface_rows: list[dict[str, Any]],
) -> list[str]:
    plt = _try_import_matplotlib()
    if plt is None:
        return []

    written: list[str] = []

    delta_rows = [row for row in zero_d_rows if row.get("quantity") == "delta_p"]
    if delta_rows:
        fig, ax = plt.subplots(figsize=(7, 4))
        orgs = [int(row["organoid_id"]) for row in delta_rows]
        scales = [float(row["scale_factor"]) for row in delta_rows]
        delta_p = [float(row["actual_value"]) for row in delta_rows]
        ax.plot(orgs, scales, marker="o", label="Scale factor")
        ax.set_xlabel("Organoid")
        ax.set_ylabel("Scale factor")
        ax2 = ax.twinx()
        ax2.plot(orgs, delta_p, marker="s", color="tab:red", label="0D Delta P")
        ax2.set_ylabel("0D Delta P")
        ax.set_title("0D Pressure-Drop Scaling")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / "scale_factors.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    field_rows_plot = [row for row in field_rows if row.get("quantity") in {"pressure", "velocity_magnitude"}]
    if field_rows_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        orgs = sorted({int(row["organoid_id"]) for row in field_rows_plot})
        for quantity in ("pressure", "velocity_magnitude"):
            vals = []
            for organoid_id in orgs:
                row = next(
                    row
                    for row in field_rows_plot
                    if int(row["organoid_id"]) == organoid_id and row["quantity"] == quantity
                )
                vals.append(float(row["nrmse_rms_scaled"]))
            ax.plot(orgs, vals, marker="o", label=quantity)
        ax.set_xlabel("Organoid")
        ax.set_ylabel("NRMSE")
        ax.set_title("Scaled-Field Error")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / "field_nrmse.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    interface_plot_rows = [
        row for row in interface_rows if row.get("quantity") in {"p_inlet_target", "p_outlet_target", "q_artery_leak", "q_venous_leak"}
    ]
    if interface_plot_rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        orgs = sorted({int(row["organoid_id"]) for row in interface_plot_rows})
        for quantity in ("p_inlet_target", "p_outlet_target", "q_artery_leak", "q_venous_leak"):
            vals = []
            for organoid_id in orgs:
                row = next(
                    row
                    for row in interface_plot_rows
                    if int(row["organoid_id"]) == organoid_id and row["quantity"] == quantity
                )
                vals.append(float(row["nrmse_rms_scaled"]))
            ax.plot(orgs, vals, marker="o", label=quantity)
        ax.set_xlabel("Organoid")
        ax.set_ylabel("NRMSE")
        ax.set_title("Interface-Condition Error")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        path = out_dir / "interface_nrmse.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))

    return written


def compare_scaled_fields(
    run_dir: Path,
    out_dir: Path,
    reference_organoid: int,
    organoid_ids: list[int],
    y_shift_step: float,
    coord_tol: float,
    pressure_offset_anchor: str,
    write_scaled_fields: bool,
    make_plots: bool,
) -> None:
    np = _import_numpy()
    h5py = _import_h5py()
    cKDTree = _import_ckdtree()

    run_dir = run_dir.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_label = f"organoid_{reference_organoid}"

    grouped_0d = _load_0d_groups(run_dir, np)
    organoid_summaries = {
        organoid_id: _load_0d_organoid_summary(grouped_0d, organoid_id, np)
        for organoid_id in organoid_ids
    }
    if reference_organoid not in organoid_summaries:
        raise KeyError(f"Reference organoid {reference_organoid} is not in the requested organoid IDs")

    ref_zero_d = organoid_summaries[reference_organoid]
    ref_iface_path = run_dir / f"organoid_{reference_organoid}" / "out_darcy" / "interface_bc.json"
    ref_p_path = run_dir / f"organoid_{reference_organoid}" / "out_darcy" / "p.xdmf"
    ref_u_path = run_dir / f"organoid_{reference_organoid}" / "out_darcy" / "u.xdmf"

    ref_iface = json.loads(ref_iface_path.read_text())
    ref_p = _load_xdmf_field(ref_p_path, np, h5py)
    ref_u = _load_xdmf_field(ref_u_path, np, h5py)

    zero_d_rows = _build_zero_d_rows(organoid_summaries, reference_organoid, pressure_offset_anchor, np)
    field_rows: list[dict[str, Any]] = []
    interface_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    scale_factors: dict[int, float] = {}

    for organoid_id in organoid_ids:
        zero_d = organoid_summaries[organoid_id]
        pressure_transform = _pressure_transform_from_0d(ref_zero_d, zero_d, pressure_offset_anchor)
        scale_factor = pressure_transform["scale_factor"]
        pressure_offset = pressure_transform["pressure_offset"]
        scale_factors[organoid_id] = scale_factor

        shift = _shift_for_organoid(reference_organoid, organoid_id, y_shift_step)
        iface_path = run_dir / f"organoid_{organoid_id}" / "out_darcy" / "interface_bc.json"
        p_path = run_dir / f"organoid_{organoid_id}" / "out_darcy" / "p.xdmf"
        u_path = run_dir / f"organoid_{organoid_id}" / "out_darcy" / "u.xdmf"

        actual_iface = json.loads(iface_path.read_text())
        actual_p = _load_xdmf_field(p_path, np, h5py)
        actual_u = _load_xdmf_field(u_path, np, h5py)

        interface_rows.extend(
            _interface_metric_rows(
                organoid_id=organoid_id,
                actual_iface=actual_iface,
                ref_iface=ref_iface,
                scale_factor=scale_factor,
                pressure_offset=pressure_offset,
                shift=shift,
                np=np,
            )
        )

        rows, predicted_p, predicted_u, p_dist, u_dist = _field_metric_rows(
            organoid_id=organoid_id,
            ref_p=ref_p,
            ref_u=ref_u,
            tgt_p=actual_p,
            tgt_u=actual_u,
            scale_factor=scale_factor,
            pressure_offset=pressure_offset,
            shift=shift,
            coord_tol=coord_tol,
            np=np,
            cKDTree=cKDTree,
        )
        field_rows.extend(rows)

        metadata_rows.append(
            {
                "organoid_id": organoid_id,
                "scale_factor": scale_factor,
                "offset_anchor": pressure_transform["offset_anchor"],
                "pressure_offset": pressure_offset,
                "offset_from_artery": pressure_transform["offset_from_artery"],
                "offset_from_vein": pressure_transform["offset_from_vein"],
                "offset_mismatch": pressure_transform["offset_mismatch"],
                "arterial_channel_drop": pressure_transform["arterial_channel_drop"],
                "venous_channel_drop": pressure_transform["venous_channel_drop"],
                "delta_p_mean": zero_d["delta_p_mean"],
                "delta_p_std": zero_d["delta_p_std"],
                "shift_x": shift[0],
                "shift_y": shift[1],
                "shift_z": shift[2],
                "max_pressure_mapping_distance": p_dist,
                "max_velocity_mapping_distance": u_dist,
            }
        )

        if write_scaled_fields:
            organoid_out = out_dir / f"organoid_{organoid_id}"
            _write_xdmf_field(
                organoid_out / f"p_scaled_from_{ref_label}.xdmf",
                actual_p["coords"],
                actual_p["topology"],
                predicted_p,
                actual_p["topology_type"],
                actual_p["attribute_type"],
                actual_p["attribute_name"],
                actual_p["attribute_center"],
                h5py,
            )
            _write_xdmf_field(
                organoid_out / f"u_scaled_from_{ref_label}.xdmf",
                actual_u["coords"],
                actual_u["topology"],
                predicted_u,
                actual_u["topology_type"],
                actual_u["attribute_type"],
                actual_u["attribute_name"],
                actual_u["attribute_center"],
                h5py,
            )

    summary = {
        "run_dir": str(run_dir),
        "reference_organoid": reference_organoid,
        "organoid_ids": organoid_ids,
        "y_shift_step": y_shift_step,
        "coord_tol": coord_tol,
        "pressure_offset_anchor": str(pressure_offset_anchor).strip().lower(),
        "scale_factors": scale_factors,
        "assumptions": [
            "Pressure and velocity are scaled by the same 0D pressure-drop factor.",
            "Organoid geometry is a translated copy of organoid 1 with a fixed y-step.",
            "Pressure uses one 0D leak pressure to set an additive offset after Delta P scaling.",
        ],
    }

    _write_csv(
        out_dir / "zero_d_scaling_summary.csv",
        zero_d_rows,
        preferred_order=[
            "organoid_id",
            "quantity",
            "scale_factor",
            "offset_anchor",
            "pressure_offset",
            "actual_value",
            "predicted_scaled_value",
            "rmse",
            "mae",
            "max_abs_error",
            "nrmse_rms_scaled",
        ],
    )
    _write_csv(
        out_dir / "field_comparison_summary.csv",
        field_rows,
        preferred_order=[
            "organoid_id",
            "category",
            "quantity",
            "scale_factor",
            "pressure_offset",
            "max_mapping_distance",
            "rmse",
            "mae",
            "max_abs_error",
            "nrmse_rms_scaled",
        ],
    )
    _write_csv(
        out_dir / "interface_bc_comparison_summary.csv",
        interface_rows,
        preferred_order=[
            "organoid_id",
            "category",
            "quantity",
            "scale_factor",
            "pressure_offset",
            "rmse",
            "mae",
            "max_abs_error",
            "nrmse_rms_scaled",
        ],
    )
    _write_csv(
        out_dir / "scaling_metadata.csv",
        metadata_rows,
        preferred_order=[
            "organoid_id",
            "scale_factor",
            "offset_anchor",
            "pressure_offset",
            "offset_from_artery",
            "offset_from_vein",
            "offset_mismatch",
            "arterial_channel_drop",
            "venous_channel_drop",
            "delta_p_mean",
            "delta_p_std",
            "shift_x",
            "shift_y",
            "shift_z",
            "max_pressure_mapping_distance",
            "max_velocity_mapping_distance",
        ],
    )
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

    plot_paths: list[str] = []
    if make_plots:
        plot_paths = _plot_summaries(out_dir, zero_d_rows, field_rows, interface_rows)
        if plot_paths:
            summary["plots"] = plot_paths
            (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote summary JSON: {out_dir / 'comparison_summary.json'}")
    print(f"Wrote 0D summary CSV: {out_dir / 'zero_d_scaling_summary.csv'}")
    print(f"Wrote field summary CSV: {out_dir / 'field_comparison_summary.csv'}")
    print(f"Wrote interface summary CSV: {out_dir / 'interface_bc_comparison_summary.csv'}")
    print(f"Wrote metadata CSV: {out_dir / 'scaling_metadata.csv'}")
    if write_scaled_fields:
        print(f"Wrote scaled XDMF/HDF5 fields under: {out_dir}")
    if plot_paths:
        for path in plot_paths:
            print(f"Wrote plot: {path}")
    elif make_plots:
        print("Skipped plots because matplotlib is not available in this Python environment.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scale Darcy pressure/velocity fields from organoid 1 using 0D pressure-drop ratios, "
            "then compare the scaled prediction against the raw Darcy fields and interface BCs."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("src/coupling-output-full-geometry/run_20"),
        help="Coupled steady run directory to analyze.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <run-dir>/scaled_from_organoid_<reference-organoid>",
    )
    parser.add_argument(
        "--reference-organoid",
        type=int,
        default=1,
        help="Reference organoid whose field solution will be translated and scaled.",
    )
    parser.add_argument(
        "--organoid-ids",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Organoid IDs to compare.",
    )
    parser.add_argument(
        "--y-shift-step",
        type=float,
        default=0.6,
        help="Per-organoid translation step in the y direction.",
    )
    parser.add_argument(
        "--coord-tol",
        type=float,
        default=1e-10,
        help="Coordinate tolerance used before falling back to nearest-neighbor mapping.",
    )
    parser.add_argument(
        "--pressure-offset-anchor",
        choices=("arterial", "venous"),
        default="arterial",
        help="Which 0D leak pressure sets the additive pressure offset after Delta P scaling.",
    )
    parser.add_argument(
        "--no-write-fields",
        action="store_true",
        help="Do not write scaled XDMF/HDF5 field files.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not attempt to write PNG summary plots.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir
        else run_dir / f"scaled_from_organoid_{args.reference_organoid}"
    )

    compare_scaled_fields(
        run_dir=run_dir,
        out_dir=out_dir,
        reference_organoid=args.reference_organoid,
        organoid_ids=args.organoid_ids,
        y_shift_step=float(args.y_shift_step),
        coord_tol=float(args.coord_tol),
        pressure_offset_anchor=str(args.pressure_offset_anchor),
        write_scaled_fields=not args.no_write_fields,
        make_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
