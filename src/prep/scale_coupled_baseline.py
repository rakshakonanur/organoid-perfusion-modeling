#!/usr/bin/env python3
"""Copy and uniformly scale a prepared coupled organoid baseline.

The transform is applied about one common center::

    x_scaled = center + scale * (x - center)

Known vascular lengths/radii are multiplied by ``scale`` and geometrically
derived 0D resistances are divided by ``scale**3``.  The source directory is
never modified and the destination must not already exist.

Run this with the FEniCS environment when the baseline contains HDF5, VTP, or
STL files, since that environment supplies h5py and VTK.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


BRANCH_COORD_COLUMNS = (
    "proximalCoordsX",
    "proximalCoordsY",
    "proximalCoordsZ",
    "distalCoordsX",
    "distalCoordsY",
    "distalCoordsZ",
)
BRANCH_LENGTH_COLUMNS = ("Length", "Radius", "ReducedDownstreamLength")
BRANCH_RESISTANCE_COLUMNS = ("ReducedResistance",)
IGNORED_DIRECTORY_NAMES = {
    ".git",
    "__pycache__",
    "coupling-output",
    "convergence_plots",
    "out_darcy",
    "timeseries",
}


def _scaled_coordinate(value: float, center_value: float, scale: float) -> float:
    return center_value + scale * (value - center_value)


def _parse_center(text: str) -> tuple[float, float, float]:
    if text.strip().lower() == "origin":
        return (0.0, 0.0, 0.0)
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("center must be 'origin' or X,Y,Z")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("center must contain three numbers") from exc
    if not all(math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("center values must be finite")
    return values  # type: ignore[return-value]


def _ignore_copy_entries(_directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if name in {
            ".DS_Store",
            "scaled_baseline_manifest.json",
            "scaled_screening_summary.json",
        }:
            ignored.add(name)
        elif name in IGNORED_DIRECTORY_NAMES or name.startswith("coupling-output-"):
            ignored.add(name)
    return ignored


def _scale_interface_value(value: Any, factor: float) -> Any:
    if isinstance(value, list):
        return [_scale_interface_value(item, factor) for item in value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return float(value) * factor


def scale_interface_json(
    source_path: Path,
    destination_path: Path,
    scale: float,
    center: tuple[float, float, float],
    flow_factor: float,
    scaled_stl_directory: Path | None,
) -> dict[str, Any]:
    with source_path.open() as stream:
        data = json.load(stream)
    changed: list[str] = []
    for key in ("coords_inlet", "coords_outlet"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, list) and len(row) >= 3:
                for axis in range(3):
                    row[axis] = _scaled_coordinate(float(row[axis]), center[axis], scale)
        changed.append(key)
    for key in list(data):
        if key.startswith("q_"):
            data[key] = _scale_interface_value(data[key], flow_factor)
            changed.append(key)
        elif key.startswith("u_"):
            data[key] = _scale_interface_value(data[key], flow_factor / scale**2)
            changed.append(key)
    if isinstance(data.get("cell_volume"), (int, float)):
        data["cell_volume"] = float(data["cell_volume"]) * scale**3
        changed.append("cell_volume")
    permeability = data.get("permeability")
    if isinstance(permeability, dict):
        if isinstance(permeability.get("perm_transition_width"), (int, float)):
            permeability["perm_transition_width"] = (
                float(permeability["perm_transition_width"]) * scale
            )
            changed.append("permeability.perm_transition_width")
        if scaled_stl_directory is not None:
            old_files = permeability.get("resolved_stl_files", [])
            names = [Path(str(item)).name for item in old_files]
            resolved = [
                str((scaled_stl_directory / name).resolve())
                for name in names
                if (scaled_stl_directory / name).exists()
            ]
            if resolved:
                permeability["requested_path"] = str(scaled_stl_directory.resolve())
                permeability["resolved_stl_files"] = resolved
                permeability["stl_file_count"] = len(resolved)
                changed.extend(
                    ["permeability.requested_path", "permeability.resolved_stl_files"]
                )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with destination_path.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=True)
        stream.write("\n")
    return {"kind": "darcy_interface_seed", "fields_scaled": sorted(set(changed))}


def discover_permeability_stls(source: Path, explicit_root: Path | None) -> list[Path]:
    if explicit_root is not None:
        root = explicit_root.expanduser().resolve()
        if root.is_file() and root.suffix.lower() == ".stl":
            return [root]
        if not root.is_dir():
            raise ValueError(f"Permeability STL root does not exist: {root}")
        return sorted(root.glob("*.stl"))

    parents: set[Path] = set()
    explicit_files: set[Path] = set()
    for interface_path in source.glob("organoid_*/out_darcy/interface_bc.json"):
        try:
            permeability = json.loads(interface_path.read_text()).get("permeability", {})
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(permeability, dict):
            continue
        candidates = list(permeability.get("resolved_stl_files", []))
        requested = permeability.get("requested_path")
        if requested:
            candidates.append(requested)
        for raw in candidates:
            candidate = Path(str(raw)).expanduser()
            if candidate.is_file() and candidate.suffix.lower() == ".stl":
                explicit_files.add(candidate.resolve())
                parents.add(candidate.resolve().parent)
            elif candidate.is_dir():
                parents.add(candidate.resolve())
    discovered = set(explicit_files)
    for parent in parents:
        discovered.update(path.resolve() for path in parent.glob("*.stl"))
    return sorted(discovered)


def scale_branching_csv(
    path: Path, scale: float, center: tuple[float, float, float]
) -> dict[str, Any]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    missing = [name for name in BRANCH_COORD_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError(f"{path} is missing branching coordinate columns: {missing}")

    center_by_column = dict(zip(BRANCH_COORD_COLUMNS, center + center))
    changed_values = 0
    for row in rows:
        for name in BRANCH_COORD_COLUMNS:
            raw = str(row.get(name, "")).strip()
            if raw:
                row[name] = repr(
                    _scaled_coordinate(float(raw), center_by_column[name], scale)
                )
                changed_values += 1
        for name in BRANCH_LENGTH_COLUMNS:
            raw = str(row.get(name, "")).strip()
            if raw:
                row[name] = repr(float(raw) * scale)
                changed_values += 1
        for name in BRANCH_RESISTANCE_COLUMNS:
            raw = str(row.get(name, "")).strip()
            if raw:
                row[name] = repr(float(raw) / scale**3)
                changed_values += 1

    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return {"kind": "branching_csv", "rows": len(rows), "values_scaled": changed_values}


def scale_zerod_geom_csv(
    path: Path, scale: float, center: tuple[float, float, float]
) -> dict[str, Any]:
    """Scale svVascularize geom.csv: start xyz, end xyz, length, radius."""
    rows: list[list[str]] = []
    with path.open(newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if len(row) < 8:
                raise ValueError(f"Expected at least 8 columns in {path}, got {len(row)}")
            values = [float(item) for item in row[:8]]
            for offset in (0, 3):
                for axis in range(3):
                    values[offset + axis] = _scaled_coordinate(
                        values[offset + axis], center[axis], scale
                    )
            values[6] *= scale
            values[7] *= scale
            rows.append([*(f"{value:.18e}" for value in values), *row[8:]])
    with path.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    return {
        "kind": "zerod_geom_csv",
        "rows": len(rows),
        "coordinate_columns_scaled": [0, 1, 2, 3, 4, 5],
        "length_column_scaled": 6,
        "radius_column_scaled": 7,
    }


def _scale_numeric(value: Any, factor: float) -> Any:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return value
    return float(value) * factor


def _scale_numeric_series(value: Any, factor: float) -> Any:
    if isinstance(value, list):
        return [_scale_numeric_series(item, factor) for item in value]
    return _scale_numeric(value, factor)


def scale_zerod_json(path: Path, scale: float, flow_factor: float) -> dict[str, Any] | None:
    try:
        with path.open() as stream:
            data = json.load(stream)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict) or not any(
        key in data for key in ("vessels", "boundary_conditions", "junctions")
    ):
        return None

    counts = {
        "vessel_lengths": 0,
        "poiseuille_resistances": 0,
        "inertances": 0,
        "compliances": 0,
        "stenosis_coefficients": 0,
        "resistance_bcs": 0,
        "flow_bcs": 0,
    }
    for vessel in data.get("vessels", []):
        if not isinstance(vessel, dict):
            continue
        if isinstance(vessel.get("vessel_length"), (int, float)):
            vessel["vessel_length"] = float(vessel["vessel_length"]) * scale
            counts["vessel_lengths"] += 1
        values = vessel.get("zero_d_element_values", {})
        if not isinstance(values, dict):
            continue
        rules = {
            "R_poiseuille": (scale**-3, "poiseuille_resistances"),
            "L": (scale**-1, "inertances"),
            "C": (scale**3, "compliances"),
            "stenosis_coefficient": (scale**-4, "stenosis_coefficients"),
        }
        for key, (factor, count_key) in rules.items():
            if isinstance(values.get(key), (int, float)):
                values[key] = float(values[key]) * factor
                counts[count_key] += 1

    for bc in data.get("boundary_conditions", []):
        if not isinstance(bc, dict):
            continue
        bc_type = str(bc.get("bc_type", "")).strip().upper()
        values = bc.get("bc_values", {})
        if not isinstance(values, dict):
            continue
        if bc_type == "RESISTANCE" and "R" in values:
            values["R"] = _scale_numeric_series(values["R"], scale**-3)
            counts["resistance_bcs"] += 1
        elif bc_type == "FLOW" and "Q" in values:
            values["Q"] = _scale_numeric_series(values["Q"], flow_factor)
            counts["flow_bcs"] += 1

    with path.open("w") as stream:
        json.dump(data, stream, indent=4)
        stream.write("\n")
    return {"kind": "zerod_json", **counts}


def scale_gmsh_v2(path: Path, scale: float, center: tuple[float, float, float]) -> dict[str, Any]:
    lines = path.read_text().splitlines(keepends=True)
    try:
        format_index = lines.index("$MeshFormat\n")
    except ValueError:
        format_index = lines.index("$MeshFormat\r\n") if "$MeshFormat\r\n" in lines else -1
    if format_index < 0 or format_index + 1 >= len(lines):
        raise ValueError(f"Not a Gmsh mesh: {path}")
    version_fields = lines[format_index + 1].split()
    if not version_fields or not version_fields[0].startswith("2.") or version_fields[1] != "0":
        raise ValueError(f"Only ASCII Gmsh 2.x meshes are supported: {path}")
    node_marker = next(i for i, line in enumerate(lines) if line.strip() == "$Nodes")
    count = int(lines[node_marker + 1].strip())
    for index in range(node_marker + 2, node_marker + 2 + count):
        fields = lines[index].split()
        if len(fields) != 4:
            raise ValueError(f"Unexpected Gmsh node record in {path}: {lines[index]!r}")
        xyz = [
            _scaled_coordinate(float(fields[axis + 1]), center[axis], scale)
            for axis in range(3)
        ]
        ending = "\r\n" if lines[index].endswith("\r\n") else "\n"
        lines[index] = f"{fields[0]} {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g}{ending}"
    path.write_text("".join(lines))
    return {"kind": "gmsh_v2", "points_scaled": count}


_GEO_POINT = re.compile(
    r"^(?P<prefix>\s*Point\s*\(.*?\)\s*=\s*\{)"
    r"(?P<body>[^}]*)"
    r"(?P<suffix>\}\s*;.*)$"
)


def scale_geo(path: Path, scale: float, center: tuple[float, float, float]) -> dict[str, Any]:
    output: list[str] = []
    count = 0
    for original in path.read_text().splitlines(keepends=True):
        ending = "\r\n" if original.endswith("\r\n") else "\n" if original.endswith("\n") else ""
        line = original.rstrip("\r\n")
        match = _GEO_POINT.match(line)
        if match is None:
            output.append(original)
            continue
        values = [part.strip() for part in match.group("body").split(",")]
        if len(values) != 4:
            raise ValueError(f"Unsupported Point expression in {path}: {line}")
        xyz = [
            _scaled_coordinate(float(values[axis]), center[axis], scale)
            for axis in range(3)
        ]
        mesh_size = float(values[3]) * scale
        body = ", ".join(f"{value:.17g}" for value in (*xyz, mesh_size))
        output.append(match.group("prefix") + body + match.group("suffix") + ending)
        count += 1
    path.write_text("".join(output))
    return {"kind": "gmsh_geo", "points_scaled": count, "mesh_sizes_scaled": count}


def scale_hdf5(path: Path, scale: float, center: tuple[float, float, float]) -> dict[str, Any]:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HDF5 files require h5py; run this script with the fenicsx-env Python"
        ) from exc

    referenced_geometry_datasets: set[str] = set()
    xdmf_path = path.with_suffix(".xdmf")
    if xdmf_path.exists():
        try:
            root = ET.parse(xdmf_path).getroot()
            for geometry in root.iter("Geometry"):
                for data_item in geometry.iter("DataItem"):
                    reference = str(data_item.text or "").strip()
                    if ":" not in reference:
                        continue
                    file_name, dataset_name = reference.split(":", 1)
                    if Path(file_name).name == path.name:
                        referenced_geometry_datasets.add(dataset_name.strip("/"))
        except (ET.ParseError, OSError):
            pass

    changed: list[str] = []
    with h5py.File(path, "r+") as handle:
        datasets: list[tuple[str, Any]] = []

        def collect(name: str, obj: Any) -> None:
            if isinstance(obj, h5py.Dataset) and (
                name.rsplit("/", 1)[-1].lower() == "geometry"
                or name.strip("/") in referenced_geometry_datasets
            ):
                datasets.append((name, obj))

        handle.visititems(collect)
        for name, dataset in datasets:
            if dataset.ndim != 2 or dataset.shape[1] < 3:
                raise ValueError(f"Unexpected geometry dataset shape {dataset.shape} in {path}:{name}")
            points = dataset[...]
            for axis in range(3):
                points[:, axis] = center[axis] + scale * (points[:, axis] - center[axis])
            dataset[...] = points
            changed.append(name)
    return {"kind": "hdf5", "geometry_datasets": changed}


def scale_adios_bp(
    path: Path,
    scale: float,
    center: tuple[float, float, float],
    flow_factor: float,
) -> dict[str, Any] | None:
    """Rewrite a DOLFINx ADIOS/BP checkpoint while preserving its step layout."""
    lower_name = path.name.lower()
    transforms: dict[str, Any] = {}
    if lower_name.startswith("tagged_branches_"):
        transforms["Points"] = "coordinates"
    elif lower_name.startswith("area_checkpoint_"):
        transforms["f_values"] = scale**2
    elif lower_name.startswith("flow_checkpoint_"):
        transforms["f_values"] = flow_factor
    elif lower_name.startswith("velocity_checkpoint_"):
        transforms["f_values"] = flow_factor / scale**2
    else:
        # Pressure checkpoints, for example, are physical values rather than
        # geometry and remain unchanged.
        return None

    try:
        import adios2  # type: ignore
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "ADIOS/BP checkpoints require adios2 and numpy; run this script "
            "with the fenicsx-env Python"
        ) from exc

    temporary = path.with_name(path.name + ".scaling_tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    changed_variables: list[str] = []
    try:
        with adios2.FileReader(str(path)) as reader:
            variable_metadata = reader.available_variables()
            attribute_metadata = reader.available_attributes()
            if not variable_metadata:
                raise ValueError(f"No variables found in ADIOS/BP checkpoint: {path}")
            missing = [name for name in transforms if name not in variable_metadata]
            if missing:
                raise ValueError(f"Missing expected variables {missing} in {path}")
            step_count = max(
                int(metadata.get("AvailableStepsCount", "1"))
                for metadata in variable_metadata.values()
            )
            # adios4dolfinx reads these legacy checkpoints with its BP4 path.
            # ADIOS2's current default is BP5, so preserve BP4 explicitly.
            output_adios = adios2.Adios()
            output_io = output_adios.declare_io(
                f"scale_coupled_baseline_{abs(hash(str(temporary)))}"
            )
            output_io.set_engine("BP4")
            with adios2.Stream(output_io, str(temporary), "w") as writer:
                for attribute_name in attribute_metadata:
                    writer.write_attribute(
                        attribute_name, reader.read_attribute(attribute_name)
                    )
                for step in range(step_count):
                    writer.begin_step()
                    for variable_name, metadata in variable_metadata.items():
                        available_steps = int(metadata.get("AvailableStepsCount", "1"))
                        if step >= available_steps:
                            continue
                        values = reader.read(
                            variable_name, step_selection=[step, 1]
                        )
                        transform = transforms.get(variable_name)
                        if transform == "coordinates":
                            values = np.asarray(values).copy()
                            if values.ndim != 2 or values.shape[1] < 3:
                                raise ValueError(
                                    f"Unexpected Points shape {values.shape} in {path}"
                                )
                            for axis in range(3):
                                values[:, axis] = center[axis] + scale * (
                                    values[:, axis] - center[axis]
                                )
                        elif isinstance(transform, (int, float)):
                            values = np.asarray(values) * float(transform)
                        shape = list(values.shape)
                        writer.write(
                            variable_name,
                            values,
                            shape=shape,
                            start=[0] * len(shape),
                            count=shape,
                        )
                    writer.end_step()
            changed_variables = list(transforms)
        shutil.rmtree(path)
        temporary.rename(path)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "kind": "adios_bp",
        "variables_scaled": changed_variables,
        "steps": step_count,
    }


def scale_vtk_geometry(
    path: Path, scale: float, center: tuple[float, float, float]
) -> dict[str, Any]:
    try:
        import vtk  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            f"{path.suffix} files require VTK; run this script with the fenicsx-env Python"
        ) from exc

    suffix = path.suffix.lower()
    if suffix == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
        writer = vtk.vtkXMLPolyDataWriter()
    elif suffix == ".stl":
        reader = vtk.vtkSTLReader()
        writer = vtk.vtkSTLWriter()
    else:
        raise ValueError(f"Unsupported VTK geometry suffix: {suffix}")
    reader.SetFileName(str(path))
    reader.Update()
    data = reader.GetOutput()
    if data is None or data.GetNumberOfPoints() == 0:
        raise ValueError(f"No points found in {path}")
    points = data.GetPoints()
    for index in range(points.GetNumberOfPoints()):
        point = points.GetPoint(index)
        points.SetPoint(
            index,
            *(
                _scaled_coordinate(float(point[axis]), center[axis], scale)
                for axis in range(3)
            ),
        )
    points.Modified()
    writer.SetFileName(str(path))
    writer.SetInputData(data)
    if suffix == ".stl":
        writer.SetFileTypeToBinary()
    if writer.Write() != 1:
        raise RuntimeError(f"VTK failed to write {path}")
    return {"kind": suffix.lstrip("."), "points_scaled": int(points.GetNumberOfPoints())}


def _candidate_files(root: Path) -> Iterable[Path]:
    bp_directories = sorted(path for path in root.rglob("*.bp") if path.is_dir())
    for path in bp_directories:
        if any(part in IGNORED_DIRECTORY_NAMES or part.startswith("coupling-output-") for part in path.parts):
            continue
        yield path
    for path in sorted(root.rglob("*")):
        if not path.is_file() or any(parent.suffix == ".bp" for parent in path.parents):
            continue
        if any(part in IGNORED_DIRECTORY_NAMES or part.startswith("coupling-output-") for part in path.parts):
            continue
        yield path


def scale_copied_baseline(
    source: Path,
    destination: Path,
    scale: float,
    center: tuple[float, float, float],
    flow_mode: str,
    permeability_stl_root: Path | None = None,
) -> dict[str, Any]:
    source = source.resolve()
    destination = destination.resolve()
    if not source.is_dir():
        raise ValueError(f"Source baseline is not a directory: {source}")
    if source == destination or source in destination.parents:
        raise ValueError("Destination cannot be the source or lie inside the source")
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Scale must be a positive finite number")
    flow_factor = 1.0 if flow_mode == "unchanged" else scale**2

    external_stls = discover_permeability_stls(source, permeability_stl_root)
    shutil.copytree(source, destination, ignore=_ignore_copy_entries)
    changed: list[dict[str, Any]] = []
    copied_unmodified: list[str] = []
    try:
        scaled_stl_directory: Path | None = None
        if external_stls:
            scaled_stl_directory = destination / "permeability_stls"
            scaled_stl_directory.mkdir(parents=True, exist_ok=True)
            for source_stl in external_stls:
                shutil.copy2(source_stl, scaled_stl_directory / source_stl.name)

        # out_darcy fields are derived and intentionally omitted, but the
        # previous interface JSON is needed to seed the first coupling update.
        for source_interface in source.glob("organoid_*/out_darcy/interface_bc.json"):
            relative = source_interface.relative_to(source)
            result = scale_interface_json(
                source_interface,
                destination / relative,
                scale,
                center,
                flow_factor,
                scaled_stl_directory,
            )
            changed.append({"path": str(relative), **result})

        for path in _candidate_files(destination):
            relative = str(path.relative_to(destination))
            result: dict[str, Any] | None = None
            lower_name = path.name.lower()
            suffix = path.suffix.lower()
            if lower_name.startswith("branchingdata") and suffix == ".csv":
                result = scale_branching_csv(path, scale, center)
            elif lower_name == "geom.csv" and "0D_Input_Files" in path.parts:
                result = scale_zerod_geom_csv(path, scale, center)
            elif suffix in {".in", ".json"}:
                result = scale_zerod_json(path, scale, flow_factor)
            elif suffix == ".msh":
                result = scale_gmsh_v2(path, scale, center)
            elif suffix == ".geo":
                result = scale_geo(path, scale, center)
            elif suffix in {".h5", ".hdf5"}:
                result = scale_hdf5(path, scale, center)
            elif suffix in {".vtp", ".stl"}:
                result = scale_vtk_geometry(path, scale, center)
            elif path.is_dir() and suffix == ".bp":
                result = scale_adios_bp(path, scale, center, flow_factor)
            if result is None:
                copied_unmodified.append(relative)
            else:
                changed.append({"path": relative, **result})
    except Exception:
        shutil.rmtree(destination)
        raise

    manifest = {
        "source": str(source),
        "destination": str(destination),
        "length_scale": scale,
        "center": list(center),
        "flow_boundary_mode": flow_mode,
        "flow_boundary_factor": flow_factor,
        "scaled_permeability_stl_directory": (
            str(scaled_stl_directory.resolve()) if scaled_stl_directory is not None else None
        ),
        "scaling_rules": {
            "coordinates_lengths_radii_mesh_sizes": scale,
            "areas": scale**2,
            "volumes": scale**3,
            "poiseuille_and_resistance_bc": scale**-3,
            "inertance": scale**-1,
            "compliance": scale**3,
            "stenosis_coefficient": scale**-4,
            "material_permeability_values": "unchanged",
        },
        "changed_files": changed,
        "copied_unmodified_files": copied_unmodified,
        "notes": [
            "Coupling output directories were intentionally omitted.",
            "XDMF files are copied unchanged; their referenced HDF5 geometry datasets are scaled.",
            "Pressure boundary conditions and material coefficients are unchanged.",
            "Tagged-mesh BP coordinates and area/flow/velocity checkpoints are scaled consistently.",
            "Derived out_darcy and 0D timeseries directories were intentionally omitted.",
            "Only scaled out_darcy/interface_bc.json seed files are retained for coupling initialization.",
            "scaled_screening_summary.json was omitted so coupling auto mode does not reuse stale field-scaling assumptions.",
            "Start the scaled coupling workflow from run 0; do not reuse calibrated coupling resistances.",
        ],
    }
    manifest_path = destination / "scaled_baseline_manifest.json"
    with manifest_path.open("w") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Prepared baseline directory")
    parser.add_argument("--output", type=Path, required=True, help="New scaled baseline directory")
    parser.add_argument("--scale", type=float, required=True, help="Uniform positive length scale")
    parser.add_argument(
        "--center",
        type=_parse_center,
        default=(0.0, 0.0, 0.0),
        help="Common scaling center as X,Y,Z (default: origin)",
    )
    parser.add_argument(
        "--flow-boundaries",
        choices=("unchanged", "area"),
        default="unchanged",
        help="Keep prescribed Q unchanged, or multiply FLOW boundary Q by scale^2",
    )
    parser.add_argument(
        "--permeability-stl-root",
        type=Path,
        default=None,
        help=(
            "Optional STL file/directory to copy and scale. By default it is "
            "discovered from organoid_*/out_darcy/interface_bc.json."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = scale_copied_baseline(
            source=args.source,
            destination=args.output,
            scale=float(args.scale),
            center=tuple(args.center),
            flow_mode=str(args.flow_boundaries),
            permeability_stl_root=args.permeability_stl_root,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Created {manifest['destination']} with "
        f"{len(manifest['changed_files'])} scaled files."
    )
    print(f"Audit: {Path(manifest['destination']) / 'scaled_baseline_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
