import pandas as pd
from pathlib import Path
# Run with: gmsh branched_network.geo -1 -format msh2 -o branched_network.msh
current_dir = Path("/Users/rakshakonanur/Documents/Research/Organoid-Project/coupled-multi-organoid-model/src/geometry")

def write_geo_from_branching_data(df, geo_file="branched_network.geo", elements_per_line=1000):
    point_map = {}
    geo_points = []
    geo_lines = []
    physical_lines = []
    physical_points = []
    point_id = 1
    line_id = 1

    # Track reverse mapping to find terminal nodes
    distal_points = []
    proximal_points = []
    point_coords_to_id = {}

    for idx, row in df.iterrows():
        p0 = (row["proximalCoordsX"], row["proximalCoordsY"], row["proximalCoordsZ"])
        p1 = (row["distalCoordsX"], row["distalCoordsY"], row["distalCoordsZ"])

        proximal_points.append(p0)
        distal_points.append(p1)

        if p0 not in point_map:
            point_map[p0] = point_id
            geo_points.append(f"Point({point_id}) = {{{p0[0]}, {p0[1]}, {p0[2]}, 1.0}};")
            point_coords_to_id[p0] = point_id
            point_id += 1

        if p1 not in point_map:
            point_map[p1] = point_id
            geo_points.append(f"Point({point_id}) = {{{p1[0]}, {p1[1]}, {p1[2]}, 1.0}};")
            point_coords_to_id[p1] = point_id
            point_id += 1

        start_id = point_map[p0]
        end_id = point_map[p1]

        geo_lines.append(f"Line({line_id}) = {{{start_id}, {end_id}}};")
        physical_lines.append(f"Physical Line({line_id}) = {{{line_id}}};")
        geo_lines.append(f"Transfinite Line{{{line_id}}} = {elements_per_line + 1} Using Progression 1;")

        line_id += 1

    # Inlet: first proximal point
    inlet_id = point_coords_to_id[proximal_points[0]]
    physical_points.append(f"Physical Point(1) = {{{inlet_id}}};")

    # Outlets: distal points that are not reused as proximal points
    terminal_coords = set(distal_points) - set(proximal_points)
    for i, outlet_coord in enumerate(terminal_coords, start=2):  # Start at 2 since inlet is 1
        outlet_id = point_coords_to_id[outlet_coord]
        physical_points.append(f"Physical Point({i}) = {{{outlet_id}}};")

    # Write all Gmsh code to file
    with open(current_dir/geo_file, "w") as f:
        f.write("\n".join(geo_points) + "\n")
        f.write("\n".join(geo_lines) + "\n")
        f.write("\n".join(physical_lines) + "\n")
        f.write("\n".join(physical_points) + "\n")

# Load your data and call the function

if __name__ == "__main__":
    # Example usage:
    df = pd.read_csv("/Users/rakshakonanur/Documents/Research/Synthetic_Vasculature/output/1D_Output/071725/Run5_25branches/1D_Input_Files/branchingData.csv")  # Replace with your actual path
    write_geo_from_branching_data(df, geo_file="branched_network.geo", elements_per_line=1000)