from generate_vasculature import Generate
import numpy as np

from necrotic_tree_generation import CORE_DOMAIN, DONUT_DOMAIN, FULL_DOMAIN, CORE_WITH_CONNECTION_DOMAIN, DONUT_WITH_CONNECTION_DOMAIN

TUNABLE_CONFIG = {
    # Reproducibility
    "random_seed": 49,
    # Spatial growth controls
    "physical_clearance": 1e-4,
    "compete": True,
    "decay_probability": 0.5,
    # Prefer terminal-domain cells farther from existing terminal outlets.
    # 0 disables distance weighting; 1 is linear; 2 is strong.
    "distance_weight_exponent": 1.0,
    # Reduce the distance bias to zero across this many failed sample batches.
    "distance_weight_relaxation_batches": 10,
    "export_watertight": False,
    "export_shell_thickness": 0.0,
    # Minimum sampled point distance from the existing vessel surfaces.
    "tree_threshold": 1e-3,
    "forest_threshold": 1e-4,
    "threshold_exponent": 1.0,
    "threshold_adjuster": 0.9,
    "n_points": 250,
    "n_closest_vessels": 5,
    "flow_ratio": 4.0,
    "max_iter": 20,
    "max_candidate_batches": 50,
    "max_forest_collision_retries": 50,
    "use_brute": False,
    "nonconvex_sampling": 100,
    # Keep the optimized junction away from all three candidate-triangle edges.
    "bifurcation_barycentric_margin": 0.05,
    # Reject embedded outlet faces and malformed sister outlets in addition to
    # the original finite-cylinder OBB collision checks.
    "strict_collision_checks": False,
    # Lightweight face-center and sister checks stay enabled without the much
    # more restrictive all-pairs capsule collision mode.
    "reject_embedded_outlet_faces": True,
    # Refuse to export CSV, solid, or solver inputs if any terminal cap center
    # is inside another vessel after growth and repair.
    "validate_forest_outlets": True,
    # The all-pairs centerline test treats vessels as rounded capsules and is
    # considerably more restrictive for these thick trees.
    "exact_nonlocal_collision_checks": False,
    # Angular constraints in degrees. A turn of 0 is straight continuation.
    # The interior optimizer controls geometry. This only rejects near-reversals
    # during incremental growth; provisional branches may be refined later.
    "max_parent_daughter_turn_angle": 150.0,
    "min_daughter_separation_angle": 30.0,
    "repair_max_parent_daughter_turn_angle": 100.0,
    "repair_min_daughter_separation_angle": 30.0,
    # Root placement controls
    "root_volume_fraction": 1.8,
    "root_threshold_adjuster": 0.9,
    "root_max_attempts": 20,
    "root_attempts": 100,
    "root_start_on": "boundary",
    # Single-tree hemodynamics
    "tree_total_flow": 1.2e-3 / 60.0,
    "tree_terminal_pressure": 147.0,
    "tree_root_pressure": 181.0,
    "tree_fluid_density": 1.0,
    "tree_kinematic_viscosity": 0.007,
    # Forest inlet hemodynamics
    "forest_inlet_total_flow": 4.5e-3 / 60.0,
    "forest_inlet_terminal_pressure": 163.0,
    "forest_inlet_root_pressure": 181.0,
    "forest_inlet_fluid_density": 1.0,
    "forest_inlet_kinematic_viscosity": 0.007,
    "forest_inlet_bc_pressure_series": [181.0, 181.0],
    # Forest outlet hemodynamics
    "forest_outlet_total_flow": 4.5e-3 / 60.0,
    "forest_outlet_terminal_pressure": 21.0,
    "forest_outlet_root_pressure": 39.0,
    "forest_outlet_fluid_density": 1.0,
    "forest_outlet_kinematic_viscosity": 0.007,
    "forest_outlet_bc_pressure_series": [21.0, 21.0],
}

print("Select generation geometry:")
print("  0 = existing/default configuration")
print("  1 = necrotic arterial tree (terminals in core)")
print("  2 = necrotic venous tree (terminals in donut)")
print("  3 = necrotic arterial/venous forest")
generation_preset = int(input("Enter geometry selection (0, 1, 2, 3): "))
if generation_preset not in (0, 1, 2, 3):
    raise ValueError("Geometry selection must be 0, 1, 2, or 3")

runtime_config = dict(TUNABLE_CONFIG)
if generation_preset == 0:
    is_forest = int(input('Enter 1 for forest and 0 for tree: '))
elif generation_preset == 1:
    is_forest = 0
    runtime_config.update({
        "geom": str(CORE_WITH_CONNECTION_DOMAIN),
        "use_mesh_containment": True,
        "inlet": np.array([-0.28, 0.9, 0.5375]),
        "inlet_normal": np.array([1.0, 0.0, 0.0]),
        "tree_root_length": 0.225,
        "tree_terminal_geom": str(CORE_DOMAIN),
        "show_tree": False,
    })
elif generation_preset == 2:
    is_forest = 0
    runtime_config.update({
        "geom": str(DONUT_WITH_CONNECTION_DOMAIN),
        "use_mesh_containment": True,
        "inlet": np.array([0.30, 0.9, 0.5375]),
        "inlet_normal": np.array([-1.0, 0.0, 0.0]),
        # A length of 0.160 ends at x=0.14, inside donut.stl.
        "tree_root_length": 0.160,
        "tree_terminal_geom": str(DONUT_DOMAIN),
        "show_tree": False,
    })
else:
    is_forest = 1
    runtime_config.update({
        "geom": str(FULL_DOMAIN),
        "use_mesh_containment": True,
        "inlet": np.array([-0.28, 0.9, 0.5375]),
        "inlet_normal": np.array([1.0, 0.0, 0.0]),
        "outlet": np.array([0.30, 0.9, 0.5375]),
        "outlet_normal": np.array([-1.0, 0.0, 0.0]),
        "forest_inlet_root_length": 0.150,
        "forest_outlet_root_length": 0.220,
        "forest_inlet_terminal_geom": str(DONUT_DOMAIN),
        "forest_outlet_terminal_geom": str(CORE_DOMAIN),
        # The endpoint marks where the external connector enters the growth
        # domain; it must be allowed to bifurcate instead of remaining a
        # permanently protected terminal stump.
        "preserve_fixed_root_tip": False,
        # Unbiased early growth avoids locking in a far-point topology.
        "distance_weight_exponent": 2.0,
        "decay_probability": 0.5,
        "max_parent_daughter_turn_angle": 150.0,
        # Keep transactional repair within the same angular search space as
        # growth.  The former 100-degree cap made some otherwise repairable
        # venous outlets fail identically on every forest retry.
        "repair_max_parent_daughter_turn_angle": 179.0,
        # Arterial and venous solids share one physical forest, so reject and
        # repair cross-tree collisions as well as within-tree swallowed caps.
        "compete": True,
        "forest_repair_compete": True,
        # The exhaustive repair pass is optional and can be very slow. Growth
        # already uses interior junctions and exact collision rejection.
        # Growth now rejects partial outlet-face embedding directly. The
        # heuristic repair pass can move a valid sister cap into another one,
        # so leave it off and rely on the mandatory dense final audit.
        "forest_post_repair": False,
    })

rom = int(input('Enter workflow (0 = 0D, 1 = 1D, 2 = generation only): '))
if rom not in (0, 1, 2):
    raise ValueError("Workflow must be 0, 1, or 2")
num_branches = int(input('Enter the number of branches: '))
obj = Generate()
obj.set_parameters(num_branches=num_branches, **runtime_config)
obj.set_assumptions(convex = False)

tree_terminal_flow = runtime_config["tree_total_flow"] / (num_branches + 1)
forest_inlet_terminal_flow = runtime_config["forest_inlet_total_flow"] / (num_branches + 1)
forest_outlet_terminal_flow = runtime_config["forest_outlet_total_flow"] / (num_branches + 1)

forest_inlet_bc_flow_series = [forest_inlet_terminal_flow, forest_inlet_terminal_flow]
forest_outlet_bc_flow_series = [-forest_outlet_terminal_flow, -forest_outlet_terminal_flow]

if is_forest == 0:
    obj.create_directory(rom, num_branches, is_forest)
    obj.implicit()
    obj.tree_build()
    if rom == 0:
        obj.export_tree_0d_files(modify_bc=True, treeID=0, scaled=False, P=[50.0*1333.22, 50*1333.22], Q=[tree_terminal_flow, tree_terminal_flow])
        obj.run_0d_simulation(modify_bc=True, forest=False, treeID=0)
    elif rom == 1:
        obj.export_tree_0d_files() # saves the model files
        obj.export_tree_1d_files()
        obj.run_tree_1d_simulation()
else:
    if generation_preset == 3:
        num_networks = 1
        trees_per_network = [2]
        print("Using one network with two trees: arterial/core and venous/donut")
    else:
        num_networks = int(input('Enter the number of networks: '))
        trees_per_network = list(map(int, input("Enter number of trees in each network separated by space: ").split()))
    obj.create_directory(rom, num_branches, is_forest)
    # Keep the legacy preview for the default preset.  The tested necrotic
    # preset runs non-interactively and writes its geometry to the output dir.
    obj.implicit(plotVolume=(generation_preset == 0))
    obj.forest_build(number_of_networks=num_networks, trees_per_network=trees_per_network)
    if rom  == 0:
            # For pure 0D-0D coupling
            obj.export_tree_0d_files(
                modify_bc=True,
                treeID=0,
                scaled=False,
                P=runtime_config["forest_inlet_bc_pressure_series"],
                Q=forest_inlet_bc_flow_series,
            )
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=0)
            obj.export_tree_0d_files(
                modify_bc=True,
                treeID=1,
                P=runtime_config["forest_outlet_bc_pressure_series"],
                Q=forest_outlet_bc_flow_series,
            )
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=1)
            obj.plot_0d_results_to_3d_forest_both()
        # obj.export_forest_0d_files(num_cardiac_cycles=3, num_time_pts_per_cycle=5, distal_pressure=0.0)
    elif rom == 1:
            # For 1D-0D-1D coupling
            obj.export_tree_0d_files(modify_bc=True, treeID=0, scaled=False, P=[0.02*1333.22, 0.02*1333.22], Q=[forest_inlet_terminal_flow, forest_inlet_terminal_flow])
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=0)
            obj.export_tree_0d_files(modify_bc=True, treeID=1, P=[0.0*1333.22, 0.0*1333.22], Q=[-forest_outlet_terminal_flow, -forest_outlet_terminal_flow])
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=1)
            obj.plot_0d_results_to_3d_forest_both()
            obj.export_forest_1d_files()
            obj.run_forest_inlet_1d_simulation()
            obj.run_forest_outlet_1d_simulation()
