"""Generate a 20-bifurcation arterial tree with terminals in core.stl."""

from necrotic_tree_generation import (
    CORE_DOMAIN,
    CORE_WITH_CONNECTION_DOMAIN,
    generate_and_validate,
)


if __name__ == "__main__":
    generate_and_validate(
        "arterial_core",
        root_start=(-0.28, 0.9, 0.5375),
        root_direction=(1.0, 0.0, 0.0),
        root_length=0.225,
        terminal_domain=CORE_DOMAIN,
        containment_domain=CORE_WITH_CONNECTION_DOMAIN,
    )
