"""Shared configuration and validation for the necrotic-core tree examples."""

from pathlib import Path
import json

import numpy as np
import pyvista as pv

from generate_vasculature import Generate, REPOSITORY_ROOT


DOMAIN_DIR = (
    REPOSITORY_ROOT
    / "files/stl/organoid-growth-domains/different-shaped-domains/necrotic-experiment"
)
FULL_DOMAIN = DOMAIN_DIR / "full.stl"
CORE_DOMAIN = DOMAIN_DIR / "core.stl"
CORE_WITH_CONNECTION_DOMAIN = DOMAIN_DIR / "core-with-connection.stl"
DONUT_DOMAIN = DOMAIN_DIR / "donut.stl"
DONUT_WITH_CONNECTION_DOMAIN = DOMAIN_DIR / "donut-with-connection.stl"
OUTPUT_ROOT = Path(__file__).resolve().parent / "necrotic_experiment_output"


COMMON_PARAMETERS = {
    "num_branches": 20,
    "geom": str(FULL_DOMAIN),
    "random_seed": 49,
    "physical_clearance": 1e-6,
    "decay_probability": 0.5,
    "distance_weight_exponent": 2.0,
    "distance_weight_relaxation_batches": 10,
    "tree_threshold": 1e-6,
    "threshold_exponent": 1.0,
    "threshold_adjuster": 0.9,
    "n_points": 250,
    "n_closest_vessels": 5,
    "flow_ratio": None,
    "max_iter": 20,
    "max_candidate_batches": 50,
    "use_brute": False,
    "nonconvex_sampling": 1000,
    "bifurcation_barycentric_margin": 0.05,
    "strict_collision_checks": True,
    "exact_nonlocal_collision_checks": False,
    # The fitted implicit function develops a false outside region at the
    # sharp tube/ring union.  Use the tetrahedral mesh for vessel containment.
    "use_mesh_containment": True,
    "max_parent_daughter_turn_angle": 179.0,
    "min_daughter_separation_angle": None,
    "repair_max_parent_daughter_turn_angle": 100.0,
    "repair_min_daughter_separation_angle": 20.0,
    "tree_total_flow": 1.2e-3 / 60.0,
    "tree_terminal_pressure": 147.0,
    "tree_root_pressure": 181.0,
    "tree_fluid_density": 1.0,
    "tree_kinematic_viscosity": 0.007,
    "show_tree": False,
}


def terminal_points(tree):
    """Return distal endpoints for vessels with no downstream children."""
    terminal_ids = [
        vessel_id
        for vessel_id, adjacency in tree.vessel_map.items()
        if not adjacency["downstream"]
    ]
    terminal_ids = np.asarray(sorted(terminal_ids), dtype=int)
    return terminal_ids, tree.data[terminal_ids, 3:6]


def points_inside_surface(points, surface_path):
    surface = pv.read(surface_path).extract_surface().triangulate().clean()
    cloud = pv.PolyData(np.asarray(points, dtype=float))
    selected = cloud.select_enclosed_points(surface, tolerance=1e-7, check_surface=True)
    return np.asarray(selected["SelectedPoints"], dtype=bool)


def save_centerlines(tree, output_path):
    """Write a lightweight VTP that can be opened directly in ParaView."""
    starts = tree.data[:, 0:3]
    ends = tree.data[:, 3:6]
    points = np.empty((2 * len(tree.data), 3), dtype=float)
    points[0::2] = starts
    points[1::2] = ends
    lines = np.column_stack(
        (
            np.full(len(tree.data), 2, dtype=np.int64),
            np.arange(2 * len(tree.data), dtype=np.int64).reshape(-1, 2),
        )
    ).reshape(-1)
    centerlines = pv.PolyData(points, lines=lines)
    centerlines.cell_data["radius"] = tree.data[:, 21]
    centerlines.save(output_path)


def validate_and_save_tree(
    tree, name, *, root_start, root_direction, root_length, terminal_domain,
    output_dir, requested_bifurcations=20,
):
    """Validate fixed root geometry and terminal-domain membership."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_end = Generate._root_endpoint(root_start, root_direction, root_length).reshape(3,)
    if not np.allclose(tree.data[0, 0:3], root_start, atol=1e-12):
        raise RuntimeError(f"{name}: generated root start does not match the requested point")

    # svVascularize represents the first bifurcation by splitting the original
    # root row into collinear parent/daughter rows.  Validate the complete
    # prescribed trunk, rather than requiring it to remain a single CSV row.
    end_candidates = np.flatnonzero(
        np.linalg.norm(tree.data[:, 3:6] - expected_end.reshape(1, 3), axis=1) < 1e-10
    )
    if len(end_candidates) != 1:
        raise RuntimeError(f"{name}: prescribed root endpoint was not preserved")
    trunk_ids = []
    vessel_id = int(end_candidates[0])
    while True:
        trunk_ids.append(vessel_id)
        parent = tree.data[vessel_id, 17]
        if np.isnan(parent):
            break
        vessel_id = int(parent)
    trunk_ids.reverse()
    trunk = tree.data[np.asarray(trunk_ids, dtype=int)]
    trunk_length = float(np.sum(trunk[:, 20]))
    direction = np.asarray(root_direction, dtype=float)
    direction /= np.linalg.norm(direction)
    trunk_vectors = trunk[:, 3:6] - trunk[:, 0:3]
    transverse = trunk_vectors - np.outer(trunk_vectors @ direction, direction)
    if not np.isclose(trunk_length, root_length, atol=1e-10) or np.any(
        np.linalg.norm(transverse, axis=1) > 1e-10
    ):
        raise RuntimeError(f"{name}: prescribed fixed-length root trunk was not preserved")

    terminal_ids, endpoints = terminal_points(tree)
    inside = points_inside_surface(endpoints, terminal_domain)
    if not np.all(inside):
        failed = terminal_ids[~inside].tolist()
        raise RuntimeError(f"{name}: terminal vessels outside target domain: {failed}")

    centerline_path = output_dir / f"{name}_centerlines.vtp"
    save_centerlines(tree, centerline_path)
    summary = {
        "name": name,
        "requested_bifurcations": int(requested_bifurcations),
        "segments": int(len(tree.data)),
        "terminals": int(len(terminal_ids)),
        "root_start": tree.data[0, 0:3].tolist(),
        "root_end": expected_end.tolist(),
        "root_length": trunk_length,
        "root_segment_ids": trunk_ids,
        "terminal_domain": str(terminal_domain),
        "terminals_inside_target": int(np.count_nonzero(inside)),
        "centerlines": str(centerline_path),
    }
    (output_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def generate_and_validate(
    name, *, root_start, root_direction, root_length, terminal_domain,
    containment_domain=FULL_DOMAIN,
):
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = dict(COMMON_PARAMETERS)
    parameters.update(
        {
            "geom": str(containment_domain),
            "inlet": np.asarray(root_start, dtype=float),
            "inlet_normal": np.asarray(root_direction, dtype=float),
            "tree_root_length": float(root_length),
            "tree_terminal_geom": str(terminal_domain),
            "outdir": str(output_dir),
            "folder": name,
        }
    )

    generator = Generate()
    generator.set_parameters(**parameters)
    generator.set_assumptions(convex=False)
    generator.implicit()
    generator.tree_build()

    validate_and_save_tree(
        generator.cerm_tree,
        name,
        root_start=root_start,
        root_direction=root_direction,
        root_length=root_length,
        terminal_domain=terminal_domain,
        output_dir=output_dir,
        requested_bifurcations=parameters["num_branches"],
    )
    return generator
