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

    @staticmethod
    def _make_tree_parameters(**kwargs):
        """
        Support both legacy and newer svVascularize TreeParameters APIs.
        """
        try:
            return TreeParameters(**kwargs)
        except TypeError:
            params = TreeParameters()
            for key, value in kwargs.items():
                setattr(params, key, value)
            return params

    def set_parameters(self,**kwargs):
        self.parameters = {}
        self.parameters['k']    = kwargs.get('k',2)
        self.parameters['q']   = kwargs.get('q',4)
        self.parameters['resolution']       = kwargs.get('resolution',50)
        self.parameters['buffer']       = kwargs.get('buffer',5)
        self.parameters['inlet_normal'] = kwargs.get('inlet_normal',np.array([0.5,0,0]))#.reshape(-1,1))
        self.parameters['outlet_normal'] = kwargs.get('outlet_normal',np.array([-0.5,0,0]))
        self.parameters['inlet'] = kwargs.get('inlet',np.array([-.28, 0.9, .5375])) #old - [2.6,3.05,3.4], [.3,.305,.34]
        self.parameters['outlet'] = kwargs.get('outlet',np.array([.30, 0.9, .5375]))
        self.parameters['num_branches'] = kwargs.get('num_branches',10)
        self.parameters['path_to_0d_solver'] = kwargs.get('path_to_0d_solver',r'/usr/local/sv/svZeroDSolver/2024-10-01/bin')
        self.parameters['path_to_1d_solver'] = kwargs.get('path_to_1d_solver',r'/usr/local/sv/oneDSolver/2025-06-26/bin/OneDSolver')
        self.parameters['outdir'] = kwargs.get('outdir',"/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/synthetic-vasculature-generation")
        self.parameters['folder'] = kwargs.get('folder','tmp')
        self.parameters['implicit_geometry'] = kwargs.get('implicit_geometry', 'stl')
        self.parameters['geom'] = kwargs.get('geom',"/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/files/stl/organoid-growth-domains/sphere/organoid-1.stl")
        self.parameters['sphere_center'] = np.asarray(kwargs.get('sphere_center', [0.0, 0.9, 0.55]), dtype=float)
        self.parameters['sphere_radius'] = float(kwargs.get('sphere_radius', 0.06))
        self.parameters['sphere_theta_resolution'] = int(kwargs.get('sphere_theta_resolution', 60))
        self.parameters['sphere_phi_resolution'] = int(kwargs.get('sphere_phi_resolution', 60))
        self.parameters['append_cylinders'] = bool(kwargs.get('append_cylinders', True))
        self.parameters['left_cylinder_center'] = np.asarray(kwargs.get('left_cylinder_center', [-0.25, 0.9, 0.5375]), dtype=float)
        self.parameters['left_cylinder_direction'] = np.asarray(kwargs.get('left_cylinder_direction', [1.0, 0.0, 0.0]), dtype=float)
        self.parameters['left_cylinder_radius'] = float(kwargs.get('left_cylinder_radius', 0.03))
        self.parameters['left_cylinder_height'] = float(kwargs.get('left_cylinder_height', 0.11))
        self.parameters['right_cylinder_center'] = np.asarray(kwargs.get('right_cylinder_center', [0.28, 0.9, 0.5375]), dtype=float)
        self.parameters['right_cylinder_direction'] = np.asarray(kwargs.get('right_cylinder_direction', [-1.0, 0.0, 0.0]), dtype=float)
        self.parameters['right_cylinder_radius'] = float(kwargs.get('right_cylinder_radius', 0.03))
        self.parameters['right_cylinder_height'] = float(kwargs.get('right_cylinder_height', 0.11))
        self.parameters['cylinder_resolution'] = int(kwargs.get('cylinder_resolution', 60))
        # 3D mesh controls (used by svVascularize Simulation.build_meshes)
        self.parameters['mesh_hmax'] = kwargs.get('mesh_hmax', 0.001)
        self.parameters['mesh_hausd'] = kwargs.get('mesh_hausd', 1e-4)
        self.parameters['mesh_hmin_factor'] = kwargs.get('mesh_hmin_factor', 0.01)
        self.parameters['mesh_hgrad'] = kwargs.get('mesh_hgrad', 1.12)
        self.parameters['mesh_minratio'] = kwargs.get('mesh_minratio', 1.4)
        self.parameters['mesh_mindihedral'] = kwargs.get('mesh_mindihedral', 18.0)
        self.parameters['mesh_remesh_vol'] = kwargs.get('mesh_remesh_vol', True)
        self.parameters['export_watertight'] = bool(kwargs.get('export_watertight', True))
        self.parameters['export_shell_thickness'] = float(kwargs.get('export_shell_thickness', 0.0))
        # Reproducibility and growth controls
        self.parameters['random_seed'] = kwargs.get('random_seed', None)
        self.parameters['physical_clearance'] = float(kwargs.get('physical_clearance', 1e-4))
        self.parameters['compete'] = bool(kwargs.get('compete', True))
        self.parameters['decay_probability'] = float(kwargs.get('decay_probability', 0.9))
        self.parameters['tree_threshold'] = kwargs.get('tree_threshold', 1e-1)
        self.parameters['forest_threshold'] = kwargs.get('forest_threshold', 2.5e-3)
        self.parameters['threshold_exponent'] = float(kwargs.get('threshold_exponent', 1.5))
        self.parameters['threshold_adjuster'] = float(kwargs.get('threshold_adjuster', 0.9))
        self.parameters['n_points'] = int(kwargs.get('n_points', 50))
        self.parameters['n_closest_vessels'] = int(kwargs.get('n_closest_vessels', 2))
        self.parameters['flow_ratio'] = kwargs.get('flow_ratio', 20)
        self.parameters['max_iter'] = int(kwargs.get('max_iter', 20))
        self.parameters['use_brute'] = bool(kwargs.get('use_brute', False))
        self.parameters['nonconvex_sampling'] = int(kwargs.get('nonconvex_sampling', 10))
        # Root placement controls
        self.parameters['root_volume_fraction'] = float(kwargs.get('root_volume_fraction', 1.0))
        self.parameters['root_threshold_adjuster'] = float(kwargs.get('root_threshold_adjuster', 0.9))
        self.parameters['root_max_attempts'] = int(kwargs.get('root_max_attempts', 20))
        self.parameters['root_attempts'] = int(kwargs.get('root_attempts', 100))
        self.parameters['root_start_on'] = kwargs.get('root_start_on', 'boundary')
        # Tree hemodynamic controls
        self.parameters['tree_terminal_pressure'] = float(kwargs.get('tree_terminal_pressure', 50.0 * 1333.22))
        self.parameters['tree_root_pressure'] = float(kwargs.get('tree_root_pressure', 120.0 * 1333.22))
        self.parameters['tree_total_flow'] = float(kwargs.get('tree_total_flow', 0.03 / 60.0))
        self.parameters['tree_fluid_density'] = kwargs.get('tree_fluid_density', None)
        self.parameters['tree_kinematic_viscosity'] = kwargs.get('tree_kinematic_viscosity', None)
        self.parameters['tree_murray_exponent'] = kwargs.get('tree_murray_exponent', None)
        self.parameters['tree_radius_exponent'] = kwargs.get('tree_radius_exponent', None)
        self.parameters['tree_length_exponent'] = kwargs.get('tree_length_exponent', None)
        # Forest inlet hemodynamic controls
        self.parameters['forest_inlet_terminal_pressure'] = float(kwargs.get('forest_inlet_terminal_pressure', 101.0))
        self.parameters['forest_inlet_root_pressure'] = float(kwargs.get('forest_inlet_root_pressure', 181.0))
        self.parameters['forest_inlet_total_flow'] = float(kwargs.get('forest_inlet_total_flow', 6e-3 / 60.0))
        self.parameters['forest_inlet_fluid_density'] = float(kwargs.get('forest_inlet_fluid_density', 1.0))
        self.parameters['forest_inlet_kinematic_viscosity'] = float(kwargs.get('forest_inlet_kinematic_viscosity', 0.007))
        self.parameters['forest_inlet_murray_exponent'] = kwargs.get('forest_inlet_murray_exponent', None)
        self.parameters['forest_inlet_radius_exponent'] = kwargs.get('forest_inlet_radius_exponent', None)
        self.parameters['forest_inlet_length_exponent'] = kwargs.get('forest_inlet_length_exponent', None)
        # Forest outlet hemodynamic controls
        self.parameters['forest_outlet_terminal_pressure'] = float(kwargs.get('forest_outlet_terminal_pressure', 21.0))
        self.parameters['forest_outlet_root_pressure'] = float(kwargs.get('forest_outlet_root_pressure', 101.0))
        self.parameters['forest_outlet_total_flow'] = float(kwargs.get('forest_outlet_total_flow', 6e-3 / 60.0))
        self.parameters['forest_outlet_fluid_density'] = float(kwargs.get('forest_outlet_fluid_density', 1.0))
        self.parameters['forest_outlet_kinematic_viscosity'] = float(kwargs.get('forest_outlet_kinematic_viscosity', 0.007))
        self.parameters['forest_outlet_murray_exponent'] = kwargs.get('forest_outlet_murray_exponent', None)
        self.parameters['forest_outlet_radius_exponent'] = kwargs.get('forest_outlet_radius_exponent', None)
        self.parameters['forest_outlet_length_exponent'] = kwargs.get('forest_outlet_length_exponent', None)


    def set_assumptions(self,**kwargs):
        self.homogeneous = kwargs.get('homogeneous',True)
        self.convex      = kwargs.get('convex',False)

    def _root_kwargs(self):
        return {
            'volume_fraction': self.parameters['root_volume_fraction'],
            'threshold_adjuster': self.parameters['root_threshold_adjuster'],
            'max_attempts': self.parameters['root_max_attempts'],
            'attempts': self.parameters['root_attempts'],
            'start_on': self.parameters['root_start_on'],
        }

    def _growth_kwargs(self, mode: str):
        threshold_key = 'tree_threshold' if mode == 'tree' else 'forest_threshold'
        return {
            'threshold': self.parameters[threshold_key],
            'threshold_exponent': self.parameters['threshold_exponent'],
            'threshold_adjuster': self.parameters['threshold_adjuster'],
            'n_points': self.parameters['n_points'],
            'n_closest_vessels': self.parameters['n_closest_vessels'],
            'flow_ratio': self.parameters['flow_ratio'],
            'max_iter': self.parameters['max_iter'],
            'use_brute': self.parameters['use_brute'],
            'nonconvex_sampling': self.parameters['nonconvex_sampling'],
            'decay_probability': self.parameters['decay_probability'],
        }

    def _terminal_flow_from_total(self, total_flow, num_branches):
        return float(total_flow) / max(int(num_branches) + 1, 1)

    def _tree_parameter_kwargs(self, prefix: str, num_branches: int):
        params = {
            'terminal_pressure': self.parameters[f'{prefix}_terminal_pressure'],
            'root_pressure': self.parameters[f'{prefix}_root_pressure'],
            'terminal_flow': self._terminal_flow_from_total(self.parameters[f'{prefix}_total_flow'], num_branches),
        }
        optional_fields = (
            ('fluid_density', f'{prefix}_fluid_density'),
            ('kinematic_viscosity', f'{prefix}_kinematic_viscosity'),
            ('murray_exponent', f'{prefix}_murray_exponent'),
            ('radius_exponent', f'{prefix}_radius_exponent'),
            ('length_exponent', f'{prefix}_length_exponent'),
        )
        for attr_name, key in optional_fields:
            value = self.parameters.get(key)
            if value is not None:
                params[attr_name] = value
        return params

    def _apply_random_seed(self, domain):
        seed = self.parameters.get('random_seed')
        if seed is None:
            return
        domain.set_random_seed(int(seed))
        domain.set_random_generator()

    def _save_generation_config(self, filename="generation_config.json"):
        serializable = {}
        for key, value in self.parameters.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, np.generic):
                serializable[key] = value.item()
            else:
                serializable[key] = value
        serializable['homogeneous'] = bool(getattr(self, 'homogeneous', True))
        serializable['convex'] = bool(getattr(self, 'convex', False))
        serializable['svvascularize_path'] = str(Path("../../clones/svVascularize").resolve())
        outpath = Path(self.parameters['outdir']) / filename
        outpath.write_text(json.dumps(serializable, indent=2))

    def _build_cylinder(self, side: str):
        base_point = np.asarray(self.parameters[f'{side}_cylinder_center'], dtype=float).reshape(3,)
        direction = np.asarray(self.parameters[f'{side}_cylinder_direction'], dtype=float).reshape(3,)
        radius = float(self.parameters[f'{side}_cylinder_radius'])
        height = float(self.parameters[f'{side}_cylinder_height'])
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError(f"{side}_cylinder_direction must be nonzero")
        if radius <= 0.0:
            raise ValueError(f"{side}_cylinder_radius must be positive, got {radius}")
        if height <= 0.0:
            raise ValueError(f"{side}_cylinder_height must be positive, got {height}")
        unit_direction = direction / norm
        # Treat the user-provided point as the cylinder base, not the midpoint.
        center = base_point + 0.5 * height * unit_direction
        return pv.Cylinder(
            center=tuple(center),
            direction=tuple(unit_direction),
            radius=radius,
            height=height,
            resolution=max(8, int(self.parameters['cylinder_resolution'])),
        ).triangulate().clean()

    def _boolean_union_all(self, base_mesh, appendages):
        merged = base_mesh.triangulate().clean()
        for part in appendages:
            merged = merged.boolean_union(part.triangulate().clean())
            merged = merged.extract_surface().triangulate().clean()
        return merged

    def implicit(self, plotVolume=False): # compute implicit domain
        geom_kind = str(self.parameters.get('implicit_geometry', 'stl')).lower()
        if geom_kind == 'sphere':
            center = np.asarray(self.parameters['sphere_center'], dtype=float).reshape(3,)
            radius = float(self.parameters['sphere_radius'])
            if radius <= 0.0:
                raise ValueError(f"sphere_radius must be positive, got {radius}")
            mesh = pv.Sphere(
                radius=radius,
                center=tuple(center),
                theta_resolution=max(8, int(self.parameters['sphere_theta_resolution'])),
                phi_resolution=max(8, int(self.parameters['sphere_phi_resolution'])),
            ).triangulate()
            print(f"Using spherical implicit domain: center={center.tolist()}, radius={radius}")
            if self.parameters.get('append_cylinders', False):
                left_cyl = self._build_cylinder('left')
                right_cyl = self._build_cylinder('right')
                mesh = self._boolean_union_all(mesh, [left_cyl, right_cyl])
                print("Appended left and right cylinders to the spherical domain")
        else:
            mesh = pv.read(self.parameters['geom'])

        if plotVolume:
            plotter = pv.Plotter()
            plotter.add_mesh(mesh, color='lightblue', opacity=0.5)
            plotter.add_axes()
            plotter.show_grid()
            plotter.show()
        cermSurf = Domain()
        cermSurf.set_data(mesh)
        self._apply_random_seed(cermSurf)
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
        params = self._make_tree_parameters(**self._tree_parameter_kwargs('tree', num_branches))
        cerm_tree = Tree()
        cerm_tree.parameters = params
        cerm_tree.physical_clearance = self.parameters['physical_clearance']
        cerm_tree.random_seed = self.parameters['random_seed']
        cerm_tree.set_domain(cermSurf)
        cerm_tree.convex = self.convex
        cerm_tree.set_root(start=root, direction=direction, **self._root_kwargs())
        cerm_tree.n_add(num_branches, **self._growth_kwargs('tree'))
        print("Tree growth debug:", cerm_tree.growth_debug_summary())
        cerm_tree.show(plot_domain=True)
        self.cerm_tree = cerm_tree
        self.data = cerm_tree.data
        self._save_generation_config()
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
        cerm_forest = Forest(
            n_networks=number_of_networks,
            n_trees_per_network=trees_per_network,
            physical_clearance=self.parameters['physical_clearance'],
            compete=self.parameters['compete'],
        )
        cerm_forest.set_domain(cermSurf)
        params_inlet = self._make_tree_parameters(**self._tree_parameter_kwargs('forest_inlet', num_branches))
        params_outlet = self._make_tree_parameters(**self._tree_parameter_kwargs('forest_outlet', num_branches))
        # for i in range(number_of_networks):
        #     for j in range(trees_per_network[i]):
        #         cerm_forest.networks[i][j].parameters = params
        cerm_forest.networks[0][0].parameters = params_inlet
        cerm_forest.networks[0][1].parameters = params_outlet
        for network in cerm_forest.networks:
            for tree in network:
                tree.random_seed = self.parameters['random_seed']
                tree.physical_clearance = self.parameters['physical_clearance']
        cerm_forest.set_roots(start_points, directions, **self._root_kwargs())
        cerm_forest.add(num_branches, **self._growth_kwargs('forest'))
        networks = cerm_forest.networks
        for i in range(number_of_networks):
            for j in range(trees_per_network[i]):
                print(f"Forest tree {i}-{j} growth debug:", networks[i][j].growth_debug_summary())
        # cerm_forest.show(plot_domain=True)
        for i in range(number_of_networks):
            for j in range(trees_per_network[i]): # currently only writes the first network, can be modified to write all networks
                self.data = networks[0][j].data
                self.save_data(filename="branchingData_{}.csv".format(j))
                solid_outdir = outdir + os.sep + "3d_tmp"
                os.makedirs(solid_outdir, exist_ok=True)
                merged_model = networks[0][j].export_solid(
                    outdir=solid_outdir,
                    shell_thickness=self.parameters['export_shell_thickness'],
                    watertight=self.parameters['export_watertight'],
                )
                merged_model.save(solid_outdir + os.sep + "geom3D_{}.vtp".format(j))

        # cerm_forest.connect() # suppressed for now
        # cerm_forest.connections.tree_connections[0].show().show()
        # sim = Simulation(synthetic_object=cerm_forest,directory=outdir, name="tree_{}".format(j))
        # root_radii = []
        # for tr in networks[0]:
        #     try:
        #         r = float(tr.data[0, 21])
        #     except Exception:
        #         r = np.nan
        #     if np.isfinite(r) and r > 0:
        #         root_radii.append(r)

        # user_hmax = float(self.parameters['mesh_hmax'])
        # if root_radii:
        #     # Keep max element size tied to vessel scale to avoid coarse lumens.
        #     adaptive_hmax = min(user_hmax, 0.5 * min(root_radii))
        # else:
        #     adaptive_hmax = user_hmax

        # print(
        #     "Meshing params: "
        #     f"hausd={self.parameters['mesh_hausd']}, "
        #     f"hmax={adaptive_hmax:.6g}, "
        #     f"hmin_factor={self.parameters['mesh_hmin_factor']}, "
        #     f"hgrad={self.parameters['mesh_hgrad']}, "
        #     f"minratio={self.parameters['mesh_minratio']}, "
        #     f"mindihedral={self.parameters['mesh_mindihedral']}"
        # )
        # sim.build_meshes(
        #     fluid=True,
        #     tissue=True,
        #     hausd=self.parameters['mesh_hausd'],
        #     remesh_vol=self.parameters['mesh_remesh_vol'],
        #     minratio=self.parameters['mesh_minratio'],
        #     mindihedral=self.parameters['mesh_mindihedral'],
        #     mesh_hmax=adaptive_hmax,
        #     mesh_hmin_factor=self.parameters['mesh_hmin_factor'],
        #     mesh_hgrad=self.parameters['mesh_hgrad'],
        # )


        # This is not working...
        # # Detect walls, outlets, and shared boundaries
        # sim.extract_faces(crease_angle=60.0)

        # # Inspect names stored in the GeneralMesh container
        # print(sim.fluid_domain_meshes[0].faces.keys())

        # # Build and persist 3-D CFD input files
        # sim.construct_3d_fluid_simulation()
        # sim.write_3d_fluid_simulation()

        # # Result: meshes + XML under sim.file_path (e.g. ./simulations/example/)
        self.cerm_forest = cerm_forest
        self._save_generation_config()

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
                             number_time_pts_per_cycle=num_time_pts_per_cycle,distal_pressure=distal_pressure, geom_filename="geom.csv", capacitance=False)
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
        merged_model = cerm_tree.export_solid(
            shell_thickness=self.parameters['export_shell_thickness'],
            watertight=self.parameters['export_watertight'],
        )
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
