from generate_vasculature import Generate

is_forest = int(input('Enter 1 for forest and 0 for tree: '))
rom = int(input('Enter the order of ROM (0, 1): '))
num_branches = int(input('Enter the number of branches: '))
obj = Generate()
obj.set_parameters(num_branches=num_branches)
obj.set_assumptions(convex = True)

if is_forest == 0:
    obj.create_directory(rom, num_branches, is_forest)
    obj.implicit()
    obj.tree_build()
    if rom == 0:
        obj.export_tree_0d_files()
        obj.export_tree_0d_files(modify_bc=True, treeID=0, scaled=False, P=[50.0*1333.22, 50*1333.22], Q=[0.03/60/num_branches, 0.03/60/num_branches])
        obj.run_0d_simulation(modify_bc=True, forest=True, treeID=0)
    elif rom == 1:
        obj.export_tree_0d_files() # saves the model files
        obj.export_tree_1d_files()
        obj.run_tree_1d_simulation()
else:
    num_networks = int(input('Enter the number of networks: '))
    trees_per_network = list(map(int, input("Enter number of trees in each network separated by space: ").split()))
    obj.create_directory(rom, num_branches, is_forest)
    obj.implicit()
    obj.forest_build(number_of_networks=num_networks, trees_per_network=trees_per_network)
    if rom  == 0:
            # For pure 0D-0D coupling
            obj.export_tree_0d_files(modify_bc=True, treeID=0, scaled=False, P=[0.5*1333.22, 0.5*1333.22], Q=[0.005/60/num_branches, 0.005/60/num_branches])
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=0)
            obj.export_tree_0d_files(modify_bc=True, treeID=1, P=[0.0*1333.22, 0*1333.22], Q=[-0.005/60/num_branches, -0.005/60/num_branches])
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=1)
            obj.plot_0d_results_to_3d_forest_both()
        # obj.export_forest_0d_files(num_cardiac_cycles=3, num_time_pts_per_cycle=5, distal_pressure=0.0)
    else:
            # For 1D-0D-1D coupling
            obj.export_tree_0d_files(modify_bc=True, treeID=0, scaled=False, P=[0.05*1333.22, 0.05*1333.22], Q=[2e-7/num_branches, 2e-7/num_branches])
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=0)
            obj.export_tree_0d_files(modify_bc=True, treeID=1, P=[0.0*1333.22, 0.0*1333.22], Q=[-2e-7/num_branches, -2e-7/num_branches])
            obj.run_0d_simulation(modify_bc=True, forest=True, treeID=1)
            obj.plot_0d_results_to_3d_forest_both()
            obj.export_forest_1d_files()
            obj.run_forest_inlet_1d_simulation()
            obj.run_forest_outlet_1d_simulation()
