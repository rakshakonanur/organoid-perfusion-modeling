"""Generate a 20-bifurcation venous tree with terminals in donut.stl."""

from necrotic_tree_generation import (
    DONUT_DOMAIN,
    DONUT_WITH_CONNECTION_DOMAIN,
    generate_and_validate,
)


if __name__ == "__main__":
    generate_and_validate(
        "venous_donut",
        root_start=(0.30, 0.9, 0.5375),
        root_direction=(-1.0, 0.0, 0.0),
        # 0.160 ends at x=0.14, inside donut.stl.  The originally proposed
        # 0.135 length ends at x=0.165, outside both donut.stl and full.stl.
        root_length=0.160,
        terminal_domain=DONUT_DOMAIN,
        containment_domain=DONUT_WITH_CONNECTION_DOMAIN,
    )
