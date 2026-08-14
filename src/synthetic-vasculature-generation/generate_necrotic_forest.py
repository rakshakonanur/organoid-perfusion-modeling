"""Generate arterial/core and venous/donut trees together as one forest."""

import numpy as np

from generate_vasculature import Generate
from necrotic_tree_generation import (
    COMMON_PARAMETERS,
    CORE_DOMAIN,
    DONUT_DOMAIN,
    OUTPUT_ROOT,
    validate_and_save_tree,
)


if __name__ == "__main__":
    output_dir = OUTPUT_ROOT / "forest"
    output_dir.mkdir(parents=True, exist_ok=True)
    parameters = dict(COMMON_PARAMETERS)
    parameters.update(
        {
            "inlet": np.array([-0.28, 0.9, 0.5375]),
            "inlet_normal": np.array([1.0, 0.0, 0.0]),
            "outlet": np.array([0.30, 0.9, 0.5375]),
            "outlet_normal": np.array([-1.0, 0.0, 0.0]),
            "forest_inlet_root_length": 0.225,
            "forest_outlet_root_length": 0.160,
            "forest_inlet_terminal_geom": str(CORE_DOMAIN),
            "forest_outlet_terminal_geom": str(DONUT_DOMAIN),
            "max_parent_daughter_turn_angle": 130.0,
            "compete": True,
            "forest_post_repair": False,
            "export_forest_solids": False,
            "outdir": str(output_dir),
            "folder": "forest",
            "forest_inlet_total_flow": 3.0e-3 / 60.0,
            "forest_inlet_terminal_pressure": 147.0,
            "forest_inlet_root_pressure": 181.0,
            "forest_inlet_fluid_density": 1.0,
            "forest_inlet_kinematic_viscosity": 0.007,
            "forest_outlet_total_flow": 3.0e-3 / 60.0,
            "forest_outlet_terminal_pressure": 21.0,
            "forest_outlet_root_pressure": 57.0,
            "forest_outlet_fluid_density": 1.0,
            "forest_outlet_kinematic_viscosity": 0.007,
        }
    )

    generator = Generate()
    generator.set_parameters(**parameters)
    generator.set_assumptions(convex=False)
    generator.implicit()
    generator.forest_build(number_of_networks=1, trees_per_network=[2])

    arterial, venous = generator.cerm_forest.networks[0]
    validate_and_save_tree(
        arterial,
        "forest_arterial_core",
        root_start=parameters["inlet"],
        root_direction=parameters["inlet_normal"],
        root_length=parameters["forest_inlet_root_length"],
        terminal_domain=CORE_DOMAIN,
        output_dir=output_dir / "arterial",
    )
    validate_and_save_tree(
        venous,
        "forest_venous_donut",
        root_start=parameters["outlet"],
        root_direction=parameters["outlet_normal"],
        root_length=parameters["forest_outlet_root_length"],
        terminal_domain=DONUT_DOMAIN,
        output_dir=output_dir / "venous",
    )
