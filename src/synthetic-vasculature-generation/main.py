from generate_vasculature import Generate

TUNABLE_CONFIG = {
    # Reproducibility
    "random_seed": 49,
    # Spatial growth controls
    "physical_clearance": 1e-4,
    "compete": True,
    "decay_probability": 0.5,
    "export_watertight": False,
    "export_shell_thickness": 0.0,
    # Minimum sampled point distance from the existing vessel surfaces.
    "tree_threshold": 1e-4,
    "forest_threshold": 1e-4,
    "threshold_exponent": 1.0,
    "threshold_adjuster": 0.9,
    "n_points": 250,
    "n_closest_vessels": 2,
    "flow_ratio": None,
    "max_iter": 20,
    "use_brute": False,
    "nonconvex_sampling": 100,
    # Root placement controls
    "root_volume_fraction": 1.0,
    "root_threshold_adjuster": 0.9,
    "root_max_attempts": 20,
    "root_attempts": 100,
    "root_start_on": "boundary",
    # Single-tree hemodynamics
    "tree_total_flow": 0.03 / 60.0,
    "tree_terminal_pressure": 50.0 * 1333.22,
    "tree_root_pressure": 120.0 * 1333.22,
    # Forest inlet hemodynamics
    "forest_inlet_total_flow": 3.0e-3 / 60.0,
    "forest_inlet_terminal_pressure": 147.0,
    "forest_inlet_root_pressure": 181.0,
    "forest_inlet_fluid_density": 1.0,
    "forest_inlet_kinematic_viscosity": 0.007,
    "forest_inlet_bc_pressure_series": [181.0, 181.0],
    # Forest outlet hemodynamics
    "forest_outlet_total_flow": 3.0e-3 / 60.0,
    "forest_outlet_terminal_pressure": 21.0,
    "forest_outlet_root_pressure": 57.0,
    "forest_outlet_fluid_density": 1.0,
    "forest_outlet_kinematic_viscosity": 0.007,
    "forest_outlet_bc_pressure_series": [21.0, 21.0],
}

is_forest = int(input('Enter 1 for forest and 0 for tree: '))
rom = int(input('Enter the order of ROM (0, 1): '))
num_branches = int(input('Enter the number of branches: '))
obj = Generate()
obj.set_parameters(num_branches=num_branches, **TUNABLE_CONFIG)
obj.set_assumptions(convex = False)

tree_terminal_flow = TUNABLE_CONFIG["tree_total_flow"] / (num_branches + 1)
forest_inlet_terminal_flow = TUNABLE_CONFIG["forest_inlet_total_flow"] / (num_branches + 1)
forest_outlet_terminal_flow = TUNABLE_CONFIG["forest_outlet_total_flow"] / (num_branches + 1)

forest_inlet_bc_flow_series = [forest_inlet_terminal_flow, forest_inlet_terminal_flow]
forest_outlet_bc_flow_series = [-forest_outlet_terminal_flow, -forest_outlet_terminal_flow]

if is_forest == 0:
    obj.create_directory(rom, num_branches, is_forest)
    obj.implicit()
    obj.tree_build()
    if rom == 0:
        obj.export_tree_0d_files()
        obj.export_tree_0d_files(modify_bc=True, treeID=0, scaled=False, P=[50.0*1333.22, 50*1333.22], Q=[tree_terminal_flow, tree_terminal_flow])
        obj.run_0d_simulation(modify_bc=True, forest=True, treeID=0)
    elif rom == 1:
        obj.export_tree_0d_files() # saves the model files
        obj.export_tree_1d_files()
        obj.run_tree_1d_simulation()
else:
    num_networks = int(input('Enter the number of networks: '))
    trees_per_network = list(map(int, input("Enter number of trees in each network separated by space: ").split()))
    obj.create_directory(rom, num_branches, is_forest)
    obj.implicit(plotVolume=True)
    obj.forest_build(number_of_networks=num_networks, trees_per_network=trees_per_network)
    if rom  == 0:
            # For pure 0D-0D coupling
            obj.export_tree_0d_files(
                modify_bc=True,
                treeID=0,
                scaled=False,
                P=TUNABLE_CONFIG["forest_inlet_bc_pressure_series"],
                Q=forest_inlet_bc_flow_series,
            )
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=0)
            obj.export_tree_0d_files(
                modify_bc=True,
                treeID=1,
                P=TUNABLE_CONFIG["forest_outlet_bc_pressure_series"],
                Q=forest_outlet_bc_flow_series,
            )
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=1)
            obj.plot_0d_results_to_3d_forest_both()
        # obj.export_forest_0d_files(num_cardiac_cycles=3, num_time_pts_per_cycle=5, distal_pressure=0.0)
    else:
            # For 1D-0D-1D coupling
            obj.export_tree_0d_files(modify_bc=True, treeID=0, scaled=False, P=[0.02*1333.22, 0.02*1333.22], Q=[forest_inlet_terminal_flow, forest_inlet_terminal_flow])
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=0)
            obj.export_tree_0d_files(modify_bc=True, treeID=1, P=[0.0*1333.22, 0.0*1333.22], Q=[-forest_outlet_terminal_flow, -forest_outlet_terminal_flow])
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=1)
            obj.plot_0d_results_to_3d_forest_both()
            obj.export_forest_1d_files()
            obj.run_forest_inlet_1d_simulation()
            obj.run_forest_outlet_1d_simulation()
