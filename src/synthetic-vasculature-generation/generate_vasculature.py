import sys
sys.path.insert(0, "../../clones/svVascularize")
import pyvista as pv
import svv
from svv.domain.domain import Domain
from svv.tree.tree import Tree
from svv.forest.forest import Forest
from svv.tree.data.data import TreeParameters
import inspect
from svv.simulation.simulation import Simulation
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import chain
from tqdm import trange
import numpy as np
import pandas as pd
from pathlib import Path
import json
# from svcco.implicit.load import load3d_pv
from time import perf_counter
from datetime import datetime
import os
import vtk
import subprocess

class Generate:
    def __init__(self):
        pass

    def set_parameters(self,**kwargs):
        self.parameters = {}
        self.parameters['k']    = kwargs.get('k',2)
        self.parameters['q']   = kwargs.get('q',4)
        self.parameters['resolution']       = kwargs.get('resolution',50)
        self.parameters['buffer']       = kwargs.get('buffer',5)
        self.parameters['inlet_normal'] = kwargs.get('inlet_normal',np.array([0.5,0,0]))#.reshape(-1,1))
        self.parameters['outlet_normal'] = kwargs.get('outlet_normal',np.array([-0.5,0,0]))
        self.parameters['inlet'] = kwargs.get('inlet',np.array([-.175, -0.9, .55])) #old - [2.6,3.05,3.4], [.3,.305,.34]
        self.parameters['outlet'] = kwargs.get('outlet',np.array([.175, -0.9, .55]))
        self.parameters['num_branches'] = kwargs.get('num_branches',10)
        self.parameters['path_to_0d_solver'] = kwargs.get('path_to_0d_solver',r'/usr/local/sv/svZeroDSolver/2024-10-01/bin')
        self.parameters['path_to_1d_solver'] = kwargs.get('path_to_1d_solver',r'/usr/local/sv/oneDSolver/2025-06-26/bin/OneDSolver')
        self.parameters['outdir'] = kwargs.get('outdir',"/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation")
        self.parameters['folder'] = kwargs.get('folder','tmp')
        self.parameters['geom'] = kwargs.get('geom',"/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/files/stl/organoid-growth-domains/original/organoid-4.stl")


    def set_assumptions(self,**kwargs):
        self.homogeneous = kwargs.get('homogeneous',True)
        self.convex      = kwargs.get('convex',False)

    def implicit(self, plotVolume=False): # compute implicit domain
        mesh = pv.read(self.parameters['geom'])
        cermSurf = Domain()
        cermSurf.set_data(mesh)
        cermSurf.create()
        cermSurf.solve()
        cermSurf.build()
        print('domain constructed')
        self.cermSurf = cermSurf
    
    def tree_build(self): # build vascular tree
        cermSurf = self.cermSurf
        root = self.parameters['inlet'].reshape(1,-1)
        direction = self.parameters['inlet_normal']
        num_branches = self.parameters['num_branches']
        print("TreeParameters ref:", TreeParameters)
        print("Defined in module:", getattr(TreeParameters, "__module__", "?"))
        print("Signature:", inspect.signature(TreeParameters))
        print("Source file:", inspect.getsourcefile(TreeParameters))
        params = TreeParameters(terminal_pressure=50.0*1333.22,
                        root_pressure=120.0*1333.22,
                        terminal_flow=0.03/60/num_branches)
        cerm_tree = Tree()
        cerm_tree.parameters = params
        cerm_tree.set_domain(cermSurf)
        cerm_tree.convex = self.convex
        cerm_tree.set_root(start=root, direction=direction)
        cerm_tree.n_add(num_branches, threshold = 1e-1)
        cerm_tree.show(plot_domain=True)
        self.cerm_tree = cerm_tree
        self.data = cerm_tree.data
        self.save_data()

    def forest_build(self, number_of_networks,trees_per_network): # build vascular forest
        cermSurf = self.cermSurf
        start_points = [
            [self.parameters['inlet'].reshape(1,-1), self.parameters['outlet'].reshape(1,-1)]
        ]
        directions = [
            [self.parameters['inlet_normal'].reshape(1,-1), self.parameters['outlet_normal'].reshape(1,-1)]
        ]

        num_branches = self.parameters['num_branches']
        outdir = self.parameters['outdir']
        folder = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep
        cerm_forest = Forest(n_networks=number_of_networks, n_trees_per_network=trees_per_network,physical_clearance=2.5e-2) 
        cerm_forest.set_domain(cermSurf)
        params_inlet = TreeParameters(terminal_pressure=0.5*1333.22,
                        root_pressure=1.0*1333.22,
                        terminal_flow=5e-7/num_branches)
        params_outlet = TreeParameters(terminal_pressure=0.0*1333.22,
                        root_pressure=0.5*1333.22,
                        terminal_flow=5e-7/num_branches)
        # for i in range(number_of_networks):
        #     for j in range(trees_per_network[i]):
        #         cerm_forest.networks[i][j].parameters = params
        cerm_forest.networks[0][0].parameters = params_inlet
        cerm_forest.networks[0][1].parameters = params_outlet
        cerm_forest.set_roots(start_points,directions)
        networks = cerm_forest.add(num_branches, threshold = 1e-1) # threshold controls the length of the appended vessels
        cerm_forest.show(plot_domain=True)
        for i in range(number_of_networks):
            for j in range(trees_per_network[i]): # currently only writes the first network, can be modified to write all networks
                self.data = networks[0][j].data
                self.save_data(filename="branchingData_{}.csv".format(j))
                merged_model = networks[0][j].export_solid(watertight=False) # use watertight = True for 3d models
                os.makedirs(outdir+os.sep+"3d_tmp", exist_ok=True)
                merged_model.save(outdir+os.sep+"3d_tmp"+os.sep+"geom3D_{}.vtp".format(i))
        # cerm_forest.connect() # suppressed for now
        # cerm_forest.connections.tree_connections[0].show().show()
        self.cerm_forest = cerm_forest

    def export_tree_0d_files(self, num_cardiac_cycles = 1, num_time_pts_per_cycle = 5, distal_pressure = 0.0, modify_bc = False,
                            treeID = 1,
                             Q=[-0.0020/60,-0.0020/60], P=[0,0], t=[0,1], scaled=False): # export 0d files required for simulation
        if not hasattr(self, 'cerm_tree'):
            cerm_tree = self.cerm_forest.networks[0][treeID] # outlet tree of the first network
            if treeID == 0:
                fold = "inlet"
            else:
                fold = "outlet"
            folder = self.parameters['folder'] + os.sep + fold
        else:
            cerm_tree = self.cerm_tree
            folder = self.parameters['folder']

        path_to_0d_solver = self.parameters['path_to_0d_solver']
        outdir = self.parameters['outdir']
    
        from svv.simulation.fluid.rom.zero_d.zerod_tree import export_0d_simulation
        # sim = Simulation(tree=cerm_tree)
        export_0d_simulation(tree=cerm_tree, get_0d_solver=False, path_to_0d_solver=path_to_0d_solver,outdir=outdir,folder=folder,number_cardiac_cycles=num_cardiac_cycles,
                             number_time_pts_per_cycle=num_time_pts_per_cycle,distal_pressure=distal_pressure, geom_filename="geom.csv", scaled=scaled, capacitance=False)
        if scaled:
            edit_flows = False
        else:
            edit_flows = True

        if treeID ==  1 and modify_bc:
            from convert_0d_bc import transform_flow_to_pressure_inlet_and_flow_outlets
            data = json.load(open(os.path.join(outdir,folder,"solver_0d.in")))

            # Transform
            new_data = transform_flow_to_pressure_inlet_and_flow_outlets(
                data=data,
                inlet_new_name="PRESSURE_IN",
                P_series=P,
                t_series=t,
                Q_series_for_outlets=Q,
                edit_flows=edit_flows
            )

            # Save
            Path(outdir + os.sep + folder + os.sep + "solver_0d_new.in").write_text(json.dumps(new_data, indent=4))
            print("Wrote solver_new.in")
        elif treeID == 0 and modify_bc:
            from convert_0d_bc import transform_resistance_to_pressure_outlets
            data = json.load(open(os.path.join(outdir,folder,"solver_0d.in")))

            # Transform
            new_data = transform_resistance_to_pressure_outlets(
                data=data,
                P_series_for_outlets=P,
                t_series=t,
                edit_flows=edit_flows
            )

            # Save
            Path(outdir + os.sep + folder + os.sep + "solver_0d_new.in").write_text(json.dumps(new_data, indent=4))
            print("Wrote solver_new.in")


    def export_forest_0d_files(self, num_cardiac_cycles = 1, num_time_pts_per_cycle = 5, distal_pressure = 0.0): # export 0d files required for simulation
        cerm_forest = self.cerm_forest
        path_to_0d_solver = self.parameters['path_to_0d_solver']
        outdir = self.parameters['outdir']
        folder = self.parameters['folder']
        inlet = self.parameters['inlet']
        outlet = self.parameters['outlet']
        from svv.simulation.fluid.rom.zero_d.zerod_forest import export_0d_simulation
        networks = export_0d_simulation(forest=cerm_forest, network_id=0, inlets=[0,], get_0d_solver=True, path_to_0d_solver=path_to_0d_solver, outdir=outdir, folder=folder, number_cardiac_cycles=num_cardiac_cycles, number_time_pts_per_cycle=num_time_pts_per_cycle, distal_pressure=distal_pressure)

    def run_0d_simulation(self, modify_bc=False, forest=False, treeID=1): # run 0d simulation
        import pysvzerod
        outputDir = self.parameters['outdir'] + os.sep + self.parameters['folder']
        if forest:
            if treeID == 0:
                folder = "inlet"
            else:
                folder = "outlet"
            outputDir = outputDir + os.sep + folder

        exe = "svzerodsolver"
        if modify_bc:
            input_file = os.path.join(outputDir, "solver_0d_new.in")
        else:
            input_file = os.path.join(outputDir, "solver_0d.in")
        output_file = os.path.join(outputDir, "output.csv")

        subprocess.run([exe, input_file, output_file],
                        cwd= self.parameters['outdir'] ,
                        stdout=None,  # Display stdout in the terminal
                        stderr=None,  # Display stderr in the terminal
                        shell=False)  # shell=False is usually safer

    def plot_0d_results_to_3d(self): # export 0d results to 3d
        os.chdir(self.parameters['outdir'] + os.sep + self.parameters['folder'])
        fileName = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'plot_0d_results_to_3d.py'
        subprocess.run(['python', fileName])

    def plot_0d_results_to_3d_forest(self): # export 0d results to 3d
        os.chdir(self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'outlet')
        fileName = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'outlet' + os.sep + 'plot_0d_results_to_3d.py'
        subprocess.run(['python', fileName])


    def plot_0d_results_to_3d_forest_both(self): # export 0d results to 3d
        os.chdir(self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'inlet')
        fileName = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'inlet' + os.sep + 'plot_0d_results_to_3d.py'
        subprocess.run(['python', fileName])
        os.chdir(self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'outlet')
        fileName = self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + 'outlet' + os.sep + 'plot_0d_results_to_3d.py'
        subprocess.run(['python', fileName])

    def export_tree_1d_files(self,number_cardiac_cycles = 5,num_points=1000): # export 1d files required for simulation
        outdir = self.parameters['outdir']
        folder = self.parameters['folder']  
        print("Directory for 1D files: ", outdir)
        cerm_tree = self.cerm_tree
        merged_model = cerm_tree.export_solid(watertight=False) # use watertight = True for 3d models
        os.makedirs(outdir+os.sep+"3d_tmp", exist_ok=True)
        merged_model.save(outdir+os.sep+"3d_tmp"+os.sep+"geom3D.vtp")
    
        from svv.simulation.simulation import Simulation
        
        one_d_sim = Simulation(synthetic_object=cerm_tree, directory = outdir + os.sep + folder)
        self.data = one_d_sim.construct_1d_fluid_simulation()
        self.save_data()


    def export_forest_1d_files(self,number_cardiac_cycles = 5,num_points=1000): # export 1d files required for simulation
        outdir = self.parameters['outdir']
        folder = self.parameters['folder']
        cerm_forest = self.cerm_forest
        # merged_model = cerm_forest.export_solid() 
        # for i in np.shape(merged_model)[0]:
        #     for j in np.shape(merged_model)[1]:
        #         merged_model.save(outdir+os.sep+"3d_tmp"+os.sep+"geom3D_{}-{}.vtp".format(i,j))

        cerm_forest.networks[0][1].parameters.root_pressure = 0.0*1333.22
    
        from svv.simulation.simulation import Simulation
        names = ["inlet", "outlet"]
        # self.data = one_d_sim.construct_1d_fluid_simulation()

        for i in range(len(cerm_forest.networks)):
            for j in range(len(cerm_forest.networks[i])):
                one_d_sim = Simulation(synthetic_object=cerm_forest.networks[i][j], directory = outdir + os.sep + folder + os.sep + names[j], name = names[j])
                self.data = one_d_sim.construct_1d_fluid_simulation()

        # from svv.simulation.simulation import Simulation # currently working (maybe?)
        # one_d_sim = Simulation(synthetic_object=cerm_forest.networks[0][1], directory = outdir, name = "inlet")
        # self.data = one_d_sim.construct_1d_fluid_simulation()

        # one_d_sim = Simulation(synthetic_object=cerm_forest.networks[0][1], directory = outdir + os.sep + "outlet")
        # self.data = one_d_sim.construct_1d_fluid_simulation()
        # # _,_,self.data = cerm_tree.export_1d_simulation(steady = True, outdir=outdir, folder=folder,number_cariac_cycles=number_cardiac_cycles,num_points=num_points)
        # self.save_data()

    def run_tree_1d_simulation(self, extract_terminal_pressure = False): # run 1d simulation
        import shutil
        one_d_folder = self.parameters['outdir']  + os.sep + self.parameters['folder']
        os.chdir(one_d_folder)
        fileName = one_d_folder + os.sep + '1d_simulation_input.json'

        backup_path = fileName.replace(".json", "_backup.json")  # Create a backup file path

        # Create a backup of the original file
        shutil.copy(fileName, backup_path)
        print(f"Backup saved at: {backup_path}")

        replace = "OUTPUT TEXT"
        new_path = "OUTPUT VTK 0"
        # Read the file and store the modified content
        with open(fileName, "r") as file:
            lines = file.readlines()  # Read all lines

        # Modify the target line
        with open(fileName, "w") as file:
            for line in lines:
                if line.strip() == replace:  # Match the line (strip to ignore spaces)
                    file.write(new_path + "\n")  # Write the new line
                else:
                    file.write(line)  # Keep the other lines unchanged

        # Replace the number of finite elements in each segment

        # Replace only the 5th entry after SEGMENT
        with open(fileName, "w") as file:
            for line in lines:
                if line.startswith("SEGMENT"):  # Check if the line starts with SEGMENT
                    parts = line.split()  # Split the line into parts
                    if len(parts) > 5 and parts[4] == "5":  # Ensure the 4th entry exists and is "5"
                        parts[4] = "100"  # Replace the 4th entry
                    line = " ".join(parts) + "\n"  # Reconstruct the line
                if line.strip() == replace:  # Match the line (strip to ignore spaces)
                    file.write(new_path + "\n")  # Write the new line
                else:
                    file.write(line)  # Write the modified line
        
        path_to_1d_solver = self.parameters['path_to_1d_solver']
        
        # Use subprocess.run and set stdout and stderr to None to inherit the output to the console
        subprocess.run([path_to_1d_solver, fileName], 
                    cwd= self.parameters['outdir'] ,
                    stdout=None,  # Display stdout in the terminal
                    stderr=None,  # Display stderr in the terminal
                    shell=False)  # shell=False is usually safer 
        
    
    def run_forest_inlet_1d_simulation(self, name = "inlet"): # run 1d simulation
        import shutil
        one_d_folder = self.parameters['outdir']  + os.sep + self.parameters['folder'] + os.sep + name
        os.chdir(one_d_folder)
        fileName = one_d_folder + os.sep + '1d_simulation_input.json'

        backup_path = fileName.replace(".json", "_backup.json")  # Create a backup file path

        # Create a backup of the original file
        shutil.copy(fileName, backup_path)
        print(f"Backup saved at: {backup_path}")

        replace = "OUTPUT TEXT"
        new_path = "OUTPUT VTK 0"
        # Read the file and store the modified content
        with open(fileName, "r") as file:
            lines = file.readlines()  # Read all lines

        # Modify the target line
        with open(fileName, "w") as file:
            for line in lines:
                if line.strip() == replace:  # Match the line (strip to ignore spaces)
                    file.write(new_path + "\n")  # Write the new line
                else:
                    file.write(line)  # Keep the other lines unchanged

        # Replace the number of finite elements in each segment

        # Replace only the 5th entry after SEGMENT
        with open(fileName, "w") as file:
            for line in lines:
                if line.startswith("SEGMENT"):  # Check if the line starts with SEGMENT
                    parts = line.split()  # Split the line into parts
                    if len(parts) > 5 and parts[4] == "5":  # Ensure the 4th entry exists and is "5"
                        parts[4] = "100"  # Replace the 4th entry
                    line = " ".join(parts) + "\n"  # Reconstruct the line
                if line.strip() == replace:  # Match the line (strip to ignore spaces)
                    file.write(new_path + "\n")  # Write the new line
                else:
                    file.write(line)  # Write the modified line
        
        path_to_1d_solver = self.parameters['path_to_1d_solver']
        
        # Use subprocess.run and set stdout and stderr to None to inherit the output to the console
        subprocess.run([path_to_1d_solver, fileName], 
                    cwd= self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + name,
                    stdout=None,  # Display stdout in the terminal
                    stderr=None,  # Display stderr in the terminal
                    shell=False)  # shell=False is usually safer

    def run_forest_outlet_1d_simulation(self, name = "outlet"): # run 1d simulation
        import shutil, re
        one_d_folder = self.parameters['outdir']  + os.sep + self.parameters['folder'] + os.sep + name
        os.chdir(one_d_folder)
        fileName = one_d_folder + os.sep + '1d_simulation_input.json'

        backup_path = fileName.replace(".json", "_backup.json")  # Create a backup file path

        # Create a backup of the original file
        shutil.copy(fileName, backup_path)
        print(f"Backup saved at: {backup_path}")

        from assign_pressure_bcs import main

        sys.argv = [
            "assign_outlet_pressures.py",
            "--deck", fileName,
            "--output", self.parameters['outdir']  + os.sep + self.parameters['folder'] + os.sep + "outlet" + os.sep + "output.csv",
            "--branching", self.parameters['outdir']  + os.sep + "branchingData_1.csv",
        ]
        main()

        replace = "OUTPUT TEXT"
        new_path = "OUTPUT VTK 0"
        # Read the file and store the modified content
        with open(fileName, "r") as file:
            lines = file.readlines()  # Read all lines

        # Modify the target line
        with open(fileName, "w") as file:
            for line in lines:
                if line.strip() == replace:  # Match the line (strip to ignore spaces)
                    file.write(new_path + "\n")  # Write the new line
                else:
                    file.write(line)  # Keep the other lines unchanged

        # Replace the number of finite elements in each segment

        # Replace only the 5th entry after SEGMENT
        with open(fileName, "w") as file:
            for line in lines:
                if line.startswith("SEGMENT"):  # Check if the line starts with SEGMENT
                    parts = line.split()  # Split the line into parts
                    if len(parts) > 5 and parts[4] == "5":  # Ensure the 4th entry exists and is "5"
                        parts[4] = "100"  # Replace the 4th entry
                    line = " ".join(parts) + "\n"  # Reconstruct the line
                if line.strip() == replace:  # Match the line (strip to ignore spaces)
                    file.write(new_path + "\n")  # Write the new line
                else:
                    file.write(line)  # Write the modified line

        path_to_1d_solver = self.parameters['path_to_1d_solver']
        
        # Use subprocess.run and set stdout and stderr to None to inherit the output to the console
        subprocess.run([path_to_1d_solver, fileName], 
                    cwd= self.parameters['outdir'] + os.sep + self.parameters['folder'] + os.sep + name,
                    stdout=None,  # Display stdout in the terminal
                    stderr=None,  # Display stderr in the terminal
                    shell=False)  # shell=False is usually safer 

    def create_directory(self, rom, num_branches, is_forest): # create directory for output files
        """Creates a directory if it doesn't exist."""

        current_time = datetime.now()
        date = f"{str(current_time.month).zfill(2)}{str(current_time.day).zfill(2)}{current_time.year%2000}"

        folder = ''
        if is_forest == 1:
            folder = 'Forest_Output/'

        if rom == 1:
            folder += '1D_Output'
        elif rom == 0:
            folder += '0D_Output'

        # Current path
        dir = self.parameters['outdir']
        if not os.path.exists(dir):
            os.makedirs(dir)
            print(f"Directory '{dir}' created successfully.")
        directory_path = dir + '/' + folder + '/' + str(date)

        count = 0
        if os.path.exists(directory_path):
            for entry in os.scandir(directory_path):
                if entry.is_dir():
                    count += 1

        path_create = directory_path + '/' + 'Run' + str(count+1) + '_' + str(num_branches) + 'branches'
        if rom == 0:
            path_create += '/' + '0D_Input_Files'
        elif rom == 1:
            path_create += '/' + '1D_Input_Files'

        os.makedirs(path_create)
        print(f"Directory '{path_create}' created successfully.")

        self.parameters['outdir'] = directory_path + '/' + 'Run' + str(count+1) + '_' + str(num_branches) + 'branches'
        if rom == 0:
            self.parameters['folder'] = '0D_Input_Files'
        elif rom == 1:
            self.parameters['folder'] = '1D_Input_Files'

    def save_data(self, filename="branchingData.csv"):
        """" From Zach's code...
        data : ndarray
            This is the contiguous 2d array of vessel data forming the vascular
            tree. Each row represents a single vessel within the tree. The array
            has a shape (N,31) where N is the current number of vessel segments
            within the tree.
            The following descibe the organization and importance of the column
            indices for each vessel.

            Column indicies:
                    index: 0:2   -> proximal node coordinates
                    index: 3:5   -> distal node coordinates
                    index: 6:8   -> unit basis U
                    index: 9:11  -> unit basis V
                    index: 12:14 -> unit basis W (axial direction)
                    index: 15,16 -> children (-1 means no child)
                    index: 17    -> parent
                    index: 18    -> proximal node index (only real edges)
                    index: 19    -> distal node index (only real edges)
                    index: 20    -> length (path length)
                    index: 21    -> radius
                    index: 22    -> flow
                    index: 23    -> left bifurcation
                    index: 24    -> right bifurcation
                    index: 25    -> reduced resistance
                    index: 26    -> depth
                    index: 27    -> reduced downstream length
                    index: 28    -> root radius scaling factor
                    index: 29    -> edge that subedge belongs to
                    index: 30    -> self identifying index
        """
        

        fileName = self.parameters['outdir'] + os.sep + filename
        columnNames = ["proximalCoordsX","proximalCoordsY","proximalCoordsZ","distalCoordsX","distalCoordsY","distalCoordsZ",
                       "U1","U2","U3","V1","V2","V3","W1","W2","W3","Child1","Child2","Parent","ProximalNodeIndex","DistalNodeIndex",
                       "Length","Radius","Flow","LeftBifurcation","RightBifurcation","ReducedResistance","Depth",
                       "ReducedDownstreamLength","RootRadiusScalingFactor","Edge","Index"]
        # convert array into dataframe 
        DF = pd.DataFrame(self.data, columns=columnNames)
        # save the dataframe as a csv file 
        DF.to_csv(fileName)

if __name__ == "__main__":
    obj = CFD()
    obj.set_parameters()
    obj.set_assumptions(convex = True)
    obj.implicit(plotVolume=True)
    # obj.tree_build()
    # obj.export_tree_0d_files()
    # obj.run_0d_simulation()
    # obj.plot_0d_results_to_3d()

    # obj.forest_build(1,[2])
    # obj.export_forest_0d_files()
    # obj.run_0d_simulation()
    # obj.plot_0d_results_to_3d()

    obj.tree_build()
    obj.export_tree_1d_files()
    obj.run_tree_1d_simulation()

    # obj.forest_build(1,[2])
    # obj.export_forest_1d_files()
    # obj.run_tree_1d_simulation()

