import gmsh
import meshio
import pandas as pd
import numpy as np
from dolfinx.io import XDMFFile, gmshio
import os, subprocess, shutil
from mpi4py import MPI
import dolfinx as dfx
import vtk
try:
    from geometry import branch
except ImportError:
    import branch
import logging
logger = logging.getLogger("coupler")   # use this in your code
logger.setLevel(logging.DEBUG)
from pathlib import Path
import glob
import re
from vtk.util.numpy_support import vtk_to_numpy
from basix.ufl import element
import pyvista as pv
import adios4dolfinx
from send2trash import send2trash
current_dir = Path("/Users/rakshakonanur/Documents/Research/two-channel/src/geometry")

WALL = 2
OUTLET = 1

def _get_ftetwild_path():
    path = os.environ.get("FTETWILD_PATH") or shutil.which("FloatTetwild_bin")
    if path is None:
        raise RuntimeError(
            "ERROR: cannot find fTetWild . "
            "Set $FTETWILD_PATH or add FloatTetwild_bin to your PATH."
        )
    return path

def geo_to_mesh_gmsh(geo_file, msh_file, mesh_size=0.001):
    gmsh.initialize()
    gmsh.open(str(current_dir / geo_file))

    # Generate 1D mesh (use .generate(2) for 2D, .generate(3) for 3D)
    gmsh.model.mesh.generate(1)

    # Save mesh in MSH version 2 format
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(current_dir / msh_file))

    gmsh.finalize()    

def stl_to_mesh_gmsh(stl_file, msh_file,char_len_min=0.001, char_len_max=0.01): # let final char_len_max be 0.01

    gmsh.initialize()
    gmsh.model.add("bioreactor")
    gmsh.merge(stl_file)

    gmsh.model.mesh.removeDuplicateNodes()
    
    # Classify surfaces to reconstruct the volume
    angle = 30  # angle threshold in degrees for feature edges
    force_param = True
    include_boundary = True
    curve_angle = 180
    gmsh.model.mesh.classifySurfaces(angle * (3.14159265359 / 180), include_boundary, force_param, curve_angle * (3.14159265359 / 180))
    
    # Create geometry from mesh
    gmsh.model.mesh.createGeometry()
    gmsh.model.mesh.createTopology()

    # Create volume from surface
    s = gmsh.model.getEntities(2)  # Get all surfaces
    gmsh.model.geo.addSurfaceLoop([surf[1] for surf in s], 1)
    vol = gmsh.model.geo.addVolume([1], 1)
    gmsh.model.geo.synchronize()

    # Define physical group (optional but needed for DOLFINx)
    gmsh.model.addPhysicalGroup(3, [1], 1)  # Volume
    gmsh.model.setPhysicalName(3, 1, "volume")

    p = gmsh.model.addPhysicalGroup(3, [vol], 9)
    gmsh.model.setPhysicalName(3, p, "Wall")

    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Smoothing", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", char_len_max)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", char_len_min)
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)

    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    gmsh.write(str(current_dir / msh_file))
    logger.info(f"Created mesh {current_dir/msh_file} from {stl_file}")
    gmsh.finalize()

def stl_to_mesh_ftet(stl_file, msh_file, edge_length=0.005, eps=0.0005):
    edge_length = edge_length  # ideal_edge_length = diag_of_bbox * L. (double, optional, default: 0.05) GOOD 0.007
    eps = 0.0005 # epsilon = diag_of_bbox * EPS. (double, optional, default: 1e-3) GOOD 0.0005
    float_tetwild_path = _get_ftetwild_path()
    command = f"{float_tetwild_path} -i {stl_file} -o {current_dir/msh_file} --lr {edge_length} --epsr {eps}"
    subprocess.run(command, shell=True, check=True)

def mesh_to_xdmf(msh_file, xdmf_file):
    # Convert .msh to .vtu
    mesh = meshio.read(current_dir/msh_file)
    meshio.write(current_dir/xdmf_file, mesh)

def xdmf_to_dolfinx(xdmf_file):
    with XDMFFile(MPI.COMM_WORLD, current_dir/xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    # Create connectivity between the mesh elements and their facets
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

def convert_mesh(msh_file, xdmf_file):
    """
    Read the mesh from a file and convert it to XDMF format.
    """
    # Read full Gmsh mesh
    msh = meshio.read(current_dir/msh_file)
    # Extract only "line" elements
    line_cells = [cell for cell in msh.cells if cell.type == "line"]
    if not line_cells:
        raise RuntimeError("No line cells found in the mesh.")
    # meshio expects a list of (cell_type, numpy_array)
    line_cells = [("line", line_cells[0].data)]
    # Optionally, also filter cell data for lines only
    line_cell_data = {}
    if "gmsh:physical" in msh.cell_data_dict:
        # meshio expects a dict of {cell_type: [data_array]}
        line_cell_data = {"gmsh:physical": [msh.cell_data_dict["gmsh:physical"]["line"]]}
    # Create a new mesh with only line elements
    line_mesh = meshio.Mesh(
        points=msh.points,
        cells=line_cells,
        cell_data=line_cell_data if line_cell_data else None
    )
    # Write to XDMF
    line_mesh.write(current_dir/xdmf_file)

def branch_mesh_tagging(mesh, inlet=np.array([0,0.41,0.34])):

    fdim = mesh.topology.dim - 1

    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    boundary_facets_indices = dfx.mesh.exterior_facet_indices(mesh.topology)
    tol = 1e-6

    def near_inlet(x):
        return np.isclose(x[0], inlet[0]) & np.isclose(x[1], inlet[1]) & np.isclose(x[2], inlet[2])

    inlet_facets = dfx.mesh.locate_entities_boundary(mesh, fdim, near_inlet)

    # Extract outlet facets: boundary facets excluding inlet facets
    outlet_facets = np.setdiff1d(boundary_facets_indices, inlet_facets)

    facet_indices = np.concatenate([inlet_facets, outlet_facets])
    facet_markers = np.concatenate([np.full(len(inlet_facets), 1, dtype=np.int32),
                                    np.full(len(outlet_facets), 2, dtype=np.int32)])
    facet_tag = dfx.mesh.meshtags(mesh, fdim, facet_indices, facet_markers)
    print("Facet indices: ", facet_indices, flush=True)
    print("Facet markers: ", facet_markers, flush=True)

    # Return outlet coordinates
    outlet_coords = mesh.geometry.x[facet_tag.indices[facet_tag.values == 2]]
    print("Outlet coordinates: ", outlet_coords, flush=True)

    return outlet_coords, facet_tag

def convert_vertex_tags_to_facet_tags(mesh, vertex_tags):
    """
    Convert vertex-based tags (dim=0) to facet-based tags (dim=dim-1),
    and reduce u_vec to assign only one facet per tagged vertex.
    """

    # with XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
    #     mesh = xdmf.read_mesh(name="Grid")

    tdim = mesh.topology.dim
    fdim = tdim - 1
    mesh.topology.create_connectivity(fdim, 0)  # facet -> vertex
    mesh.topology.create_connectivity(fdim, tdim)  # facet -> cell

    facet_to_vertex = mesh.topology.connectivity(fdim, 0)
    vertex_to_facet = {}

    facet_indices = []
    facet_values = []
    total_facets, total_values, sorted_facets = [],[],[]
    used_facets = set()
    used_vertices = set()

    for facet in range(mesh.topology.index_map(fdim).size_local):
        vertex_ids = facet_to_vertex.links(facet)
        tags_on_vertices = vertex_tags.values[np.isin(vertex_tags.indices, vertex_ids)]

        if len(tags_on_vertices) > 0:
            # Assign tag only if vertex hasn't been used
            for v_id in vertex_ids:
                if v_id in vertex_tags.indices and v_id not in used_vertices:
                    tag = np.bincount(tags_on_vertices).argmax()
                    facet_indices.append(facet)
                    facet_values.append(tag)
                    used_vertices.add(v_id)
                    used_facets.add(facet)
                    break  # Only one facet per vertex
    
    total_facets.append(facet_indices)
    total_values.append(np.full_like(facet_indices, OUTLET))  # Tag 0 for internal facets
    facet_indices = np.array(facet_indices, dtype=np.int32)
    facet_values = np.array(facet_values, dtype=np.int32)
    sorted_indices = np.argsort(facet_indices)
    internal_tags = dfx.mesh.meshtags(mesh, fdim, facet_indices[sorted_indices], facet_values[sorted_indices])

    # Determine the wall facets
    wall_BC_indices, wall_BC_markers = [], []
    wall_BC_facets = dfx.mesh.exterior_facet_indices(mesh.topology)
    total_facets.append(wall_BC_facets)
    total_values.append(np.full_like(wall_BC_facets, WALL)) # Tag 0 for wall BCs
    wall_BC_indices.append(wall_BC_facets)
    wall_BC_markers.append(np.full_like(wall_BC_facets, WALL))  # Tag 0 for wall BCs

    wall_BC_indices = np.hstack(wall_BC_indices).astype(np.int32)  # Ensure facets are in int32 format
    wall_BC_markers = np.hstack(wall_BC_markers).astype(np.int32)  # Ensure markers are in int32 format
    sorted_facets = np.argsort(wall_BC_indices)
    external_tags = dfx.mesh.meshtags(mesh, fdim, wall_BC_indices[sorted_facets], wall_BC_markers[sorted_facets])
    print("External facets for wall BCs:", wall_BC_indices, flush=True)

    # Remove external facets from facet_indices and facet_values
    sorted_facets = np.argsort(facet_indices)
    velocity_facets = dfx.mesh.meshtags(mesh, fdim, facet_indices[sorted_facets], facet_values[sorted_facets])
    facet_indices = np.setdiff1d(facet_indices, wall_BC_indices)
    facet_values = np.setdiff1d(facet_values, wall_BC_markers)
    print("Internal facets after removing wall BCs:", facet_indices, flush=True)

    # Save both meshtags for visualization
    total_values = np.hstack(total_values).astype(np.int32)
    total_facets = np.hstack(total_facets).astype(np.int32)
    sorted_facets = np.argsort(total_facets)
    print("Total facets:", total_facets[sorted_facets], flush=True)

    # Write mesh and tags to output files
    if mesh.comm.rank == 0:
        out_str = current_dir/'mesh_tags.xdmf'
        with XDMFFile(mesh.comm, out_str, 'w') as xdmf_out:
            xdmf_out.write_mesh(mesh)
            mesh.topology.create_connectivity(fdim, tdim)
            xdmf_out.write_meshtags(
                dfx.mesh.meshtags(mesh, fdim, total_facets[sorted_facets], total_values[sorted_facets]),
                mesh.geometry
            )

def generate_1d_files(xdmf_file, output_dir, file_prefix=None, inlet_coords=np.array([0,0.41,0.34])):

    # Load the converted XDMF mesh
    with XDMFFile(MPI.COMM_WORLD, current_dir/xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    P1 = dfx.fem.functionspace(mesh, ("CG", 1))  # Continuous Lagrange, degree 1
    P1_vec = element("Lagrange", mesh.basix_cell(), 1, shape=(mesh.geometry.dim,))
    P1vec = dfx.fem.functionspace(mesh, P1_vec)  # Vector space for velocity
    dof_coords = P1.tabulate_dof_coordinates()
    num_dofs = dof_coords.shape[0]

    velocity_fn = dfx.fem.Function(P1vec)
    pressure_fn = dfx.fem.Function(P1)
    flow_fn = dfx.fem.Function(P1)
    area_fn = dfx.fem.Function(P1)

    centerlineVel, centerlineFlow, pressure, centerlineCoords, area = load_vtp(output_dir)
    Nt = len(centerlineFlow)  # Number of timesteps

    fdim = mesh.topology.dim - 1

    mesh.topology.create_connectivity(fdim, mesh.topology.dim)
    outlet_coords, facet_tag = branch_mesh_tagging(mesh, inlet=inlet_coords)

    # If file exists, remove it
    if (current_dir / f"tagged_branches{file_prefix}.bp").exists():
        send2trash(current_dir / f"tagged_branches{file_prefix}.bp")

    adios4dolfinx.write_mesh(Path(str(current_dir / f"tagged_branches{file_prefix}.bp")), mesh, engine="BP4")
    adios4dolfinx.write_meshtags(Path(str(current_dir / f"tagged_branches{file_prefix}.bp")), mesh, facet_tag, engine="BP4")

    # Allocate u_val as a 2D array for all timesteps and dofs
    u_val = np.zeros((Nt, num_dofs))

    # Create connectivity between the mesh elements and their facets
    mesh.topology.create_connectivity(mesh.topology.dim,
                                       mesh.topology.dim - 1)
    
    vtx_vel = dfx.io.VTXWriter(
        MPI.COMM_WORLD, current_dir/f"velocity{file_prefix}.bp", [velocity_fn], engine="BP4"
    )
    
    vtx_pres = dfx.io.VTXWriter(
        MPI.COMM_WORLD, current_dir/f"pressure{file_prefix}.bp", [pressure_fn], engine="BP4"
    )
    vtx_flow = dfx.io.VTXWriter(
        MPI.COMM_WORLD, current_dir/f"flow{file_prefix}.bp", [flow_fn], engine="BP4"
    )
    vtx_area = dfx.io.VTXWriter(
        MPI.COMM_WORLD, current_dir/f"area{file_prefix}.bp", [area_fn], engine="BP4"
    )
    vtx_vel.write(0.0)
    vtx_pres.write(0.0)
    vtx_flow.write(0.0)
    vtx_area.write(0.0)
    # adios4dolfinx.write_mesh(Path("velocity.bp"), mesh, engine="BP4")
    # adios4dolfinx.write_mesh(Path("pressure.bp"), mesh, engine="BP4")
    # Commenting out nearest neighbor search
    tree = None  # KDTree for nearest neighbor search
    dt = 1

    # if checkpoint files exist, remove them
    if (current_dir / f"velocity_checkpoint{file_prefix}.bp").exists():
        send2trash(current_dir / f"velocity_checkpoint{file_prefix}.bp")
    if (current_dir / f"pressure_checkpoint{file_prefix}.bp").exists():
        send2trash(current_dir / f"pressure_checkpoint{file_prefix}.bp")
    if (current_dir / f"flow_checkpoint{file_prefix}.bp").exists():
        send2trash(current_dir / f"flow_checkpoint{file_prefix}.bp")
    if (current_dir / f"area_checkpoint{file_prefix}.bp").exists():
        send2trash(current_dir / f"area_checkpoint{file_prefix}.bp")

    for i in range(Nt):
        time = i * dt
        points = centerlineCoords  # Get the points for the current timestep
        scalar_velocity = centerlineVel[time]  # Get the velocity for the current timestep
        scalar_flow = centerlineFlow[time]  # Get the flow for the current timestep
        scalar_pressure = pressure[time]  # Get the pressure for the current timestep
        scalar_area = area  # Area is constant across timesteps

        if tree is None:
            tree = cKDTree(points)

        # Nearest neighbor interpolation
        distances, indices = tree.query(dof_coords)
        interpolated_velocity = scalar_velocity[indices]
        interpolated_pressure = scalar_pressure[indices]
        interpolated_flow = scalar_flow[indices]
        interpolated_area = scalar_area[indices]

        # Compute tangents using the next connected point
        tangents = np.zeros((len(indices), 3))
        for j, idx in enumerate(indices):
            # Simple forward difference: pick next point if possible, else previous
            if idx < len(points) - 1:
                delta = points[idx + 1] - points[idx]
            else:
                delta = points[idx] - points[idx - 1]
            tangent = delta / np.linalg.norm(delta)
            tangents[j] = tangent

        # Assign to velocity function
        velocity_fn.x.array[:] = (interpolated_velocity[:, np.newaxis] * tangents).flatten()
        pressure_fn.x.array[:] = interpolated_pressure
        flow_fn.x.array[:] = interpolated_flow
        area_fn.x.array[:] = interpolated_area

        adios4dolfinx.write_function(
            Path(current_dir / f"velocity_checkpoint{file_prefix}.bp"),
            u = velocity_fn,
            time=float(time),
            name="f")
        adios4dolfinx.write_function(
            Path(current_dir / f"pressure_checkpoint{file_prefix}.bp"),
            u = pressure_fn,
            time=float(time),
            name="f")
        adios4dolfinx.write_function(
            Path(current_dir / f"flow_checkpoint{file_prefix}.bp"),
            u = flow_fn,
            time=float(time),
            name="f")
        adios4dolfinx.write_function(
            Path(current_dir / f"area_checkpoint{file_prefix}.bp"),
            u = area_fn,
            time=float(time),
            name="f")

        # --- Write data to file with time stamp ---
        vtx_vel.write(float(time))
        vtx_pres.write(float(time))
        vtx_flow.write(float(time))
        vtx_area.write(float(time))

    vtx_vel.close()
    vtx_pres.close()
    vtx_flow.close()
    vtx_area.close()

    return outlet_coords

    
def load_vtp(directory):
    def extract_arrays(data_obj):
        return {
            data_obj.GetArray(i).GetName(): vtk_to_numpy(data_obj.GetArray(i))
            for i in range(data_obj.GetNumberOfArrays())
        }

    # Find all model files in the specified directory
    print("Searching for model files in directory:", directory)
    
    # Get a sorted list of all VTP files
    vtp_files = sorted(glob.glob(os.path.join(directory, "*.vtp")), key=lambda x: int(re.findall(r'\d+', x)[-1]))

    # if no files found, raise error
    if not vtp_files:
        vtp_files = sorted(glob.glob(os.path.join(directory+os.sep+"timeseries", "*.vtp")), key=lambda x: int(re.findall(r'\d+', x)[-1]))
    all_data = []
    points_per_section = 20  # Number of points per section to average

    centerlineVel = []  # initialize as a list
    centerlineFlow = []  # initialize as a list
    pressure = []  # initialize as a list

    for file in vtp_files:
        if file.endswith("centerlines.vtp"):
            continue
        if file.endswith("1d_model.vtp"):
            continue
        if file.endswith("total.vtp"):
            continue
        print(f"Reading: {file}")
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(file)
        reader.Update()
        polydata = reader.GetOutput()

        def extract_arrays(data_obj):
            return {
                data_obj.GetArray(i).GetName(): vtk_to_numpy(data_obj.GetArray(i))
                for i in range(data_obj.GetNumberOfArrays())
            }
        
        # Extract point coordinates
        points = polydata.GetPoints()

        # Convert to NumPy array
        num_points = points.GetNumberOfPoints()

        timestep_data = {
            "filename": file,
            "coords": np.array([points.GetPoint(i) for i in range(num_points)]),
            "point_data": extract_arrays(polydata.GetPointData()),
            "cell_data": extract_arrays(polydata.GetCellData()),
            "field_data": extract_arrays(polydata.GetFieldData())
        }


        area = timestep_data["point_data"]["Area"]  # (N,)
        flowrate_1d = timestep_data["point_data"]["Flowrate"]  # (N,)
        pressure_1d = timestep_data["point_data"]["Pressure_mmHg"]  # (N,)
        mesh_3d_coords = timestep_data["coords"]  # (N, 3)

        velocity_1d = flowrate_1d/area # calculate velocity from flowrate and area
        centerline = velocity_1d[::points_per_section] # save only one point for each cross-section
        centerlineFlowrate = flowrate_1d[::points_per_section] # save only one point for each cross-section
        centerlineFlow.append(centerlineFlowrate) # save the flowrate values for each timestep in each row

        centerlineVel.append(centerline)# save the velocity values for each timestep in each row
        pressure.append(pressure_1d[::points_per_section]) # save the pressure values for each timestep in each row
        centerlineCoords = mesh_3d_coords.reshape(-1, points_per_section, 3).mean(axis=1)

        area = area[::points_per_section]  # save only one point for each cross-section


    return centerlineVel, centerlineFlow, pressure, centerlineCoords, area


def domain_mesh_tagging(mesh, coords, tag_value=1, tol=5e-3):

    """
    Tag vertices in the mesh that are near given coordinates.

    Parameters:
        mesh (dolfinx.Mesh): The mesh object
        coords (np.ndarray): An (N, 3) array of points to tag vertices near
        tag_value (int): The marker value to assign to those vertices
        tol (float): Tolerance for proximity check

    Returns:
        dolfinx.mesh.meshtags: Vertex tags
    """
    vdim = 0  # vertex dimension
    mesh_coords = mesh.geometry.x

    # Define proximity function
    def near_coords(x):
        return np.any([np.linalg.norm(x.T - p, axis=1) < tol for p in coords], axis=0)

    # Find matching vertex indices
    near_vertex_indices = dfx.mesh.locate_entities(mesh, vdim, near_coords)

    # Tag them
    tags = np.full_like(near_vertex_indices, tag_value, dtype=np.int32)
    vertex_tags = dfx.mesh.meshtags(mesh, vdim, near_vertex_indices, tags)

    print("Tagged vertex indices:", near_vertex_indices, flush=True)
    # Ensure vertex-to-cell connectivity exists
    mesh.topology.create_connectivity(0, mesh.topology.dim)

    # Write vertex tags
    with dfx.io.XDMFFile(mesh.comm, "vertex_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(vertex_tags, mesh.geometry)

    return vertex_tags

from scipy.spatial import cKDTree

def domain_mesh_tagging_nearest(xdmf_file, coords):
    """
    Tag the nearest INTERIOR vertex in the mesh to each coordinate in coords.

    Parameters:
        xdmf_file (str or Path): Path to the mesh XDMF file
        coords (np.ndarray): An (N, 3) array of target points

    Returns:
        dolfinx.mesh.meshtags: Vertex tags
        dolfinx.mesh.Mesh:    Mesh
    """
    from dolfinx.io import XDMFFile
    from dolfinx import mesh as dfx_mesh

    # ------------------------------------------------------------------
    # 1. Load mesh
    # ------------------------------------------------------------------
    with XDMFFile(MPI.COMM_WORLD, current_dir / xdmf_file, "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")

    tdim = mesh.topology.dim
    vdim = 0  # vertex dimension

    # Connectivity needed for boundary detection
    mesh.topology.create_connectivity(vdim, tdim)

    # ------------------------------------------------------------------
    # 2. Find boundary vertices and define "interior" set
    # ------------------------------------------------------------------
    # All boundary vertices (dimension 0 on the exterior boundary)
    boundary_vertices = dfx_mesh.locate_entities_boundary(
        mesh,
        vdim,
        lambda x: np.full(x.shape[1], True, dtype=bool)
    )
    boundary_vertices = np.unique(boundary_vertices)

    mesh_coords = mesh.geometry.x
    num_vertices = mesh_coords.shape[0]

    is_boundary = np.zeros(num_vertices, dtype=bool)
    is_boundary[boundary_vertices] = True

    interior_vertices = np.where(~is_boundary)[0]

    if interior_vertices.size == 0:
        raise RuntimeError("No interior vertices found in the mesh.")

    # ------------------------------------------------------------------
    # 3. Build KDTree on interior vertices only and query
    # ------------------------------------------------------------------
    tree = cKDTree(mesh_coords[interior_vertices])
    distances, local_indices = tree.query(coords)
    # Map back to global vertex indices
    vertex_indices = interior_vertices[local_indices]

    # Remove duplicates (optional)
    unique_vertex_indices = np.unique(vertex_indices)

    # ------------------------------------------------------------------
    # 4. Create meshtags for these interior vertices
    # ------------------------------------------------------------------
    tags = np.full_like(unique_vertex_indices, OUTLET, dtype=np.int32)
    vertex_tags = dfx_mesh.meshtags(mesh, vdim, unique_vertex_indices, tags)

    print("Nearest INTERIOR tagged vertex indices:", unique_vertex_indices, flush=True)
    print("Vertex coordinates:", mesh_coords[unique_vertex_indices], flush=True)

    # ------------------------------------------------------------------
    # 5. Write mesh + tags back to file
    # ------------------------------------------------------------------
    with dfx.io.XDMFFile(mesh.comm, current_dir / xdmf_file, "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(vertex_tags, mesh.geometry)

        print(f"Total vertices: {num_vertices}")
        print(f"Tagged vertices: {vertex_tags.indices.size}")

    return vertex_tags, mesh

import numpy as np
from dolfinx import mesh as dmesh, fem
from mpi4py import MPI

import numpy as np
from dolfinx import mesh as dmesh

def tag_facet_patch_by_seed(mesh, x0, n0, theta_deg, marker):
    """
    Tag a connected patch of boundary facets starting from a seed point x0.
    Connectivity is defined by shared vertices between boundary facets.
    The patch grows over neighboring facets whose normal makes an angle
    <= theta_deg with the reference direction n0.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        3D mesh.
    x0 : array_like, shape (3,)
        Seed point on / near the desired surface.
    n0 : array_like, shape (3,)
        Reference normal direction (will be normalized).
    theta_deg : float
        Max allowed angle (degrees) between facet normal and n0.
    marker : int
        Integer tag for the selected patch.

    Returns
    -------
    facet_tags : dolfinx.mesh.MeshTags
        Meshtags on boundary facets (fdim = tdim-1) with 'marker' on
        the selected patch. Only tagged facets are present.
    """

    tdim = mesh.topology.dim
    fdim = tdim - 1
    gdim = mesh.geometry.dim
    assert gdim == 3, "This helper is written for 3D meshes."

    # --------------------------------------------------------------
    # 1) All boundary facets
    # --------------------------------------------------------------
    mesh.topology.create_connectivity(fdim, 0)
    all_bdry_facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True)
    )
    if len(all_bdry_facets) == 0:
        raise RuntimeError("No boundary facets found.")

    f_to_v = mesh.topology.connectivity(fdim, 0)
    coords = mesh.geometry.x

    nfacets = len(all_bdry_facets)
    facet_centers = np.zeros((nfacets, gdim))
    facet_normals = np.zeros((nfacets, gdim))

    # --------------------------------------------------------------
    # 2) Centers & normals for boundary facets
    # --------------------------------------------------------------
    for i, f in enumerate(all_bdry_facets):
        vs = f_to_v.links(f)
        pts = coords[vs]

        facet_centers[i, :] = pts.mean(axis=0)

        v0, v1, v2 = pts[0], pts[1], pts[2]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n)
        if n_norm > 0:
            n /= n_norm
        facet_normals[i, :] = n

    # --------------------------------------------------------------
    # 3) Vertex-based adjacency: facets that share any vertex
    # --------------------------------------------------------------
    # Build vertex -> boundary facets map
    v_to_f = {}
    for f in all_bdry_facets:
        vs = f_to_v.links(f)
        for v in vs:
            v_to_f.setdefault(int(v), []).append(int(f))

    # Build adjacency: for each boundary facet, all other boundary
    # facets sharing at least one vertex
    adj = {int(f): set() for f in all_bdry_facets}
    for f in all_bdry_facets:
        vs = f_to_v.links(f)
        neigh = set()
        for v in vs:
            neigh.update(v_to_f[int(v)])
        neigh.discard(int(f))
        adj[int(f)] = neigh

    # --------------------------------------------------------------
    # 4) Region-growing from seed facet with angle filter
    # --------------------------------------------------------------
    x0 = np.asarray(x0, dtype=float)
    n0 = np.asarray(n0, dtype=float)
    n0 /= np.linalg.norm(n0)

    cos_thr = np.cos(np.deg2rad(theta_deg))

    # Find nearest boundary facet to seed point
    d2 = np.sum((facet_centers - x0) ** 2, axis=1)
    seed_idx = int(np.argmin(d2))
    seed_facet = int(all_bdry_facets[seed_idx])

    facet_to_idx = {int(f): i for i, f in enumerate(all_bdry_facets)}

    selected = set()
    visited = set()
    stack = [seed_facet]

    while stack:
        f = stack.pop()
        if f in visited:
            continue
        visited.add(f)

        i = facet_to_idx[f]
        nf = facet_normals[i]

        # Angle condition
        if np.dot(nf, n0) >= cos_thr:
            selected.add(f)
            for nb in adj[f]:
                if nb not in visited:
                    stack.append(nb)

    if not selected:
        raise RuntimeError("Region-growing selected no facets. "
                           "Check x0, n0, and theta_deg.")

    # --------------------------------------------------------------
    # 5) Build MeshTags for selected patch
    # --------------------------------------------------------------
    facet_indices = np.array(sorted(selected), dtype=np.int32)
    facet_values  = np.full_like(facet_indices, marker, dtype=np.int32)

    facet_tags = dmesh.meshtags(mesh, fdim, facet_indices, facet_values)
    return facet_tags

import numpy as np
from dolfinx import mesh as dmesh

def tag_surface_from_seed(mesh, x0, theta_deg, marker):
    """
    Tag a connected patch of boundary facets starting from a seed point x0.
    Connectivity is defined by shared vertices between boundary facets.
    The patch grows over neighboring facets whose normal makes an angle
    <= theta_deg with the *seed facet's* normal.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        3D mesh.
    x0 : array_like, shape (3,)
        Seed point on / near the desired surface (in physical coordinates).
    theta_deg : float
        Max allowed angle (degrees) between facet normal and the seed normal.
    marker : int
        Integer tag for the selected patch.

    Returns
    -------
    facet_tags : dolfinx.mesh.MeshTags
        Meshtags on boundary facets (fdim = tdim-1) with 'marker' on
        the selected patch. Only those facets are present.
    """

    tdim = mesh.topology.dim
    fdim = tdim - 1
    gdim = mesh.geometry.dim
    assert gdim == 3, "tag_surface_from_seed is written for 3D meshes."

    # ----------------------------------------------------------
    # 1) All boundary facets
    # ----------------------------------------------------------
    mesh.topology.create_connectivity(fdim, 0)
    all_bdry_facets = dmesh.locate_entities_boundary(
        mesh, fdim, lambda x: np.full(x.shape[1], True)
    )
    if len(all_bdry_facets) == 0:
        raise RuntimeError("No boundary facets found.")

    f_to_v = mesh.topology.connectivity(fdim, 0)
    coords = mesh.geometry.x

    nfacets = len(all_bdry_facets)
    facet_centers = np.zeros((nfacets, gdim))
    facet_normals = np.zeros((nfacets, gdim))

    # ----------------------------------------------------------
    # 2) Centers & normals for boundary facets
    # ----------------------------------------------------------
    for i, f in enumerate(all_bdry_facets):
        vs = f_to_v.links(f)
        pts = coords[vs]

        # center
        facet_centers[i, :] = pts.mean(axis=0)

        # normal from first 3 vertices
        v0, v1, v2 = pts[0], pts[1], pts[2]
        n = np.cross(v1 - v0, v2 - v0)
        n_norm = np.linalg.norm(n)
        if n_norm > 0.0:
            n /= n_norm
        facet_normals[i, :] = n

    # ----------------------------------------------------------
    # 3) Vertex-based adjacency: facets sharing any vertex
    # ----------------------------------------------------------
    v_to_f = {}
    for f in all_bdry_facets:
        vs = f_to_v.links(f)
        for v in vs:
            v_to_f.setdefault(int(v), []).append(int(f))

    adj = {int(f): set() for f in all_bdry_facets}
    for f in all_bdry_facets:
        vs = f_to_v.links(f)
        neigh = set()
        for v in vs:
            neigh.update(v_to_f[int(v)])
        neigh.discard(int(f))
        adj[int(f)] = neigh

    # ----------------------------------------------------------
    # 4) Region growing from seed facet (auto normal)
    # ----------------------------------------------------------
    x0 = np.asarray(x0, dtype=float)

    # pick the boundary facet whose center is closest to x0
    d2 = np.sum((facet_centers - x0) ** 2, axis=1)
    seed_idx = int(np.argmin(d2))
    seed_facet = int(all_bdry_facets[seed_idx])

    # seed normal
    n0 = facet_normals[seed_idx].copy()
    n0_norm = np.linalg.norm(n0)
    if n0_norm == 0.0:
        raise RuntimeError("Seed facet has zero normal; pick a different x0?")
    n0 /= n0_norm

    cos_thr = np.cos(np.deg2rad(theta_deg))
    facet_to_idx = {int(f): i for i, f in enumerate(all_bdry_facets)}

    selected = set()
    visited = set()
    stack = [seed_facet]

    while stack:
        f = stack.pop()
        if f in visited:
            continue
        visited.add(f)

        i = facet_to_idx[f]
        nf = facet_normals[i]

        # ignore normal sign: treat nf and -nf as equivalent
        if abs(np.dot(nf, n0)) >= cos_thr:
            selected.add(f)
            for nb in adj[f]:
                if nb not in visited:
                    stack.append(nb)


    if not selected:
        raise RuntimeError("tag_surface_from_seed: no facets selected. "
                           "Check x0 and theta_deg.")

    facet_indices = np.array(sorted(selected), dtype=np.int32)
    facet_values  = np.full_like(facet_indices, marker, dtype=np.int32)

    facet_tags = dmesh.meshtags(mesh, fdim, facet_indices, facet_values)
    return facet_tags

def create_surface_tags(x_inlet, x_outlet, mesh, n_inlet=[1.0,0.0,0.0], n_outlet=[-1.0,0.0,0.0]):
    INLET = 1
    OUTLET = 2
    inlet_tags = tag_surface_from_seed(mesh, x_inlet, theta_deg=60, marker=INLET)
    outlet_tags = tag_surface_from_seed(mesh, x_outlet, theta_deg=60, marker=OUTLET)
    fdim = mesh.topology.dim - 1
    indices = np.concatenate([inlet_tags.indices, outlet_tags.indices]).astype(np.int32)
    values  = np.concatenate([inlet_tags.values,  outlet_tags.values]).astype(np.int32)

    facet_tags = dmesh.meshtags(mesh, fdim, indices, values)
    from dolfinx import io

    with io.XDMFFile(mesh.comm, current_dir/"bioreactor_facets.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tags, mesh.geometry)


def import_branched_mesh(branching_data_file, output_1d, geo_file="branched_network.geo", msh_file="branched_network.msh", xdmf_file="branched_network.xdmf", fileprefix="", coords=np.array([0,0.41,0.34])):
    df = pd.read_csv(branching_data_file)
    branch.write_geo_from_branching_data(df, geo_file=geo_file)
    geo_to_mesh_gmsh(geo_file=geo_file, msh_file=msh_file)
    convert_mesh(msh_file=msh_file, xdmf_file=xdmf_file)
    xdmf_to_dolfinx(xdmf_file=xdmf_file)
    outlet_coords = generate_1d_files(xdmf_file=xdmf_file, output_dir=output_1d, file_prefix=fileprefix, inlet_coords=coords)
    return outlet_coords

def create_bioreactor_mesh(stl_file, msh_file="bioreactor.msh", xdmf_file="bioreactor.xdmf", diric=None, coords_inlet=None, coords_outlet=None):
    stl_to_mesh_gmsh(stl_file, msh_file=msh_file)
    mesh_to_xdmf(msh_file=msh_file, xdmf_file=xdmf_file)
    xdmf_to_dolfinx(xdmf_file=xdmf_file)
    facet_tags, mesh = domain_mesh_tagging_nearest(xdmf_file=xdmf_file, coords=diric)
    create_surface_tags(x_inlet=coords_inlet, x_outlet=coords_outlet, mesh=mesh)
    convert_vertex_tags_to_facet_tags(mesh=mesh, vertex_tags=facet_tags)


class Files():
    def __init__(self, stl_file, output_1d_inlet, output_1d_outlet, branching_data_inlet, branching_data_outlet, init = True, single=True, coords_inlet=np.array([-.183, 0.6, .593]), coords_outlet=np.array([+.186, 0.6, .61]),
                 geo_file_inlet="branched_network_inlet.geo",
                 msh_file_inlet="branched_network_inlet.msh",
                 xdmf_file_inlet="branched_network_inlet.xdmf",
                 geo_file_outlet="branched_network_outlet.geo",
                 msh_file_outlet="branched_network_outlet.msh",
                 xdmf_file_outlet="branched_network_outlet.xdmf",
                 msh_bioreactor="bioreactor.msh",
                 xdmf_bioreactor="bioreactor.xdmf"):
        if single:
            diric = import_branched_mesh(branching_data_inlet, output_1d_inlet, inlet=True)
        else:
            diric = import_branched_mesh(branching_data_inlet, output_1d_inlet, geo_file=geo_file_inlet, msh_file=msh_file_inlet, xdmf_file=xdmf_file_inlet, fileprefix="_inlet", coords=coords_inlet)
            _ = import_branched_mesh(branching_data_outlet, output_1d_outlet, geo_file=geo_file_outlet, msh_file=msh_file_outlet, xdmf_file=xdmf_file_outlet, fileprefix="_outlet", coords=coords_outlet)
        if init:
            create_bioreactor_mesh(stl_file, msh_bioreactor, xdmf_bioreactor, diric=diric, coords_inlet=coords_inlet, coords_outlet=coords_outlet)




if __name__ == "__main__":
    # perfusion = Files(stl_file="/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/syntheticVasculature/files/geometry/cermRaksha_scaled.stl",
    #                             branching_data_file="/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/output/1D_Output/090425/Run2_50branches/1D_Input_Files/branchingData.csv")
    perfusion = Files(stl_file="/Users/rakshakonanur/Documents/Research/two-channel/files/organoid-2.stl",
                                output_1d_inlet = "/Users/rakshakonanur/Documents/Research/two-channel/input/trial-3/organoid_2/0D_Input_Files/inlet",
                                output_1d_outlet = "/Users/rakshakonanur/Documents/Research/two-channel/input/trial-3/organoid_2/0D_Input_Files/outlet",
                                branching_data_inlet="/Users/rakshakonanur/Documents/Research/two-channel/input/trial-3/organoid_2/branchingData_0.csv",
                                branching_data_outlet="/Users/rakshakonanur/Documents/Research/two-channel/input/trial-3/organoid_2/branchingData_1.csv",
                                coords_inlet=np.array([-0.183, 0.3, .593]),
                                coords_outlet=np.array([0.186, 0.3, .61]),
                                single=False,
                                init=True
                                )
