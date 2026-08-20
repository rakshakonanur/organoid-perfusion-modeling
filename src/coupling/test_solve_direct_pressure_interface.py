import unittest
from types import SimpleNamespace

import numpy as np

from coupling.solve_direct_pressure_interface import (
    _dynamic_pressure_transform,
    _static_scaled_transforms,
    InterfaceLayout,
    expand_static_reference_state,
    solve_affine_fixed_point,
    solve_dynamic_fixed_point,
    write_solved_interface_bcs,
)


class DirectPressureInterfaceTests(unittest.TestCase):
    def test_affine_fixed_point_with_mixed_component_scales(self) -> None:
        jacobian = np.array([[0.20, 2.0e9], [1.0e-11, -0.10]])
        offset = np.array([120.0, 3.0e-9])

        def operator(state: np.ndarray) -> np.ndarray:
            return offset + jacobian @ state

        expected = np.linalg.solve(np.eye(2) - jacobian, offset)
        solution, fitted_offset, fitted_jacobian, mode, condition = (
            solve_affine_fixed_point(
                operator,
                reference=np.array([100.0, 2.0e-9]),
                scales=np.array([1.0, 1.0e-9]),
                rcond=1.0e-12,
            )
        )

        self.assertEqual(mode, "direct")
        self.assertTrue(np.isfinite(condition))
        np.testing.assert_allclose(fitted_offset, offset, rtol=1.0e-12, atol=1.0e-14)
        np.testing.assert_allclose(fitted_jacobian, jacobian, rtol=1.0e-12, atol=1.0e-14)
        np.testing.assert_allclose(solution, expected, rtol=1.0e-12, atol=1.0e-14)
        np.testing.assert_allclose(operator(solution), solution, rtol=1.0e-12, atol=1.0e-14)

    def test_rejects_zero_basis_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "scales must be nonzero"):
            solve_affine_fixed_point(
                lambda state: state,
                reference=np.zeros(2),
                scales=np.array([1.0, 0.0]),
                rcond=1.0e-12,
            )

    def test_dynamic_pressure_transform_reproduces_target_endpoints(self) -> None:
        pressures = {
            (1, "arterial"): 180.0,
            (1, "venous"): 20.0,
            (2, "arterial"): 150.0,
            (2, "venous"): 30.0,
        }
        transform = _dynamic_pressure_transform(1, 2, pressures, "arterial", 1.0e-8)
        self.assertAlmostEqual(transform["scale_factor"], 0.75)
        self.assertAlmostEqual(transform["pressure_offset"], 15.0)
        self.assertAlmostEqual(15.0 + 0.75 * 180.0, 150.0)
        self.assertAlmostEqual(15.0 + 0.75 * 20.0, 30.0)
        self.assertAlmostEqual(transform["offset_mismatch"], 0.0)

    def test_dynamic_solver_handles_nonlinear_fixed_point(self) -> None:
        def operator(state: np.ndarray) -> np.ndarray:
            return np.array([1.0 + 0.1 * state[0] ** 2])

        solution, history, mode, condition = solve_dynamic_fixed_point(
            operator,
            reference=np.array([1.0]),
            scales=np.array([1.0]),
            tolerance=1.0e-10,
            max_iterations=20,
            minimum_step=1.0e-6,
            rcond=1.0e-12,
        )
        expected = (1.0 - np.sqrt(0.6)) / 0.2
        self.assertIn("dynamic_broyden", mode)
        self.assertTrue(history)
        self.assertTrue(np.isfinite(condition))
        np.testing.assert_allclose(solution, [expected], rtol=1.0e-9, atol=1.0e-10)

    def test_static_transforms_are_loaded_without_recomputation(self) -> None:
        summary = {
            "scaled_organoids": [
                {"organoid_id": 2, "scale_factor": 0.8, "pressure_offset": 10.0},
                {"organoid_id": 3, "scale_factor": 0.6, "pressure_offset": 20.0},
            ]
        }
        transforms = _static_scaled_transforms(summary, 3, 1)
        self.assertEqual(transforms[1]["scale_factor"], 1.0)
        self.assertEqual(transforms[1]["pressure_offset"], 0.0)
        self.assertEqual(transforms[2]["scale_factor"], 0.8)
        self.assertEqual(transforms[2]["pressure_offset"], 10.0)
        self.assertEqual(transforms[3]["scale_factor"], 0.6)
        self.assertEqual(transforms[3]["pressure_offset"], 20.0)

    def test_static_transforms_require_every_scaled_organoid(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing organoids \\[3\\]"):
            _static_scaled_transforms(
                {"scaled_organoids": [
                    {"organoid_id": 2, "scale_factor": 0.8, "pressure_offset": 10.0}
                ]},
                3,
                1,
            )

    def test_static_expansion_directly_scales_reference_result(self) -> None:
        reference_layout = InterfaceLayout(
            terminal_names=["organoid1_inlet_OUT1", "organoid1_outlet_OUT1"],
            leak_keys=[(1, "arterial"), (1, "venous")],
        )
        full_layout = InterfaceLayout(
            terminal_names=[
                "organoid1_inlet_OUT1", "organoid1_outlet_OUT1",
                "organoid2_inlet_OUT1", "organoid2_outlet_OUT1",
            ],
            leak_keys=[
                (1, "arterial"), (1, "venous"),
                (2, "arterial"), (2, "venous"),
            ],
        )
        expanded = expand_static_reference_state(
            np.array([100.0, 20.0, -2.0e-9, 3.0e-9]),
            reference_layout,
            full_layout,
            SimpleNamespace(organoid=1, branch_ids_inlet=[1], branch_ids_outlet=[1]),
            {
                1: {"scale_factor": 1.0, "pressure_offset": 0.0},
                2: {"scale_factor": 0.5, "pressure_offset": 10.0},
            },
        )
        np.testing.assert_allclose(
            expanded,
            [100.0, 20.0, 60.0, 20.0, -2.0e-9, 3.0e-9, -1.0e-9, 1.5e-9],
        )

    def test_solved_leaks_are_written_as_signed_flow_bcs(self) -> None:
        deck = {
            "boundary_conditions": [
                {
                    "bc_name": "organoid1_inlet_OUT1",
                    "bc_type": "RESISTANCE",
                    "bc_values": {"R": 1.0, "Pd": 0.0},
                },
                {
                    "bc_name": "LEAK_ART_1",
                    "bc_type": "FLOW",
                    "bc_values": {"t": [0.0, 0.5, 1.0], "Q": [0.0, 0.0, 0.0]},
                },
                {
                    "bc_name": "LEAK_VEN_1",
                    "bc_type": "FLOW",
                    "bc_values": {"t": [0.0, 1.0], "Q": [0.0, 0.0]},
                },
            ]
        }
        result = write_solved_interface_bcs(
            deck,
            {"organoid1_inlet_OUT1": 123.0},
            {(1, "arterial"): -2.0e-9, (1, "venous"): 3.0e-9},
        )
        by_name = {bc["bc_name"]: bc for bc in result["boundary_conditions"]}
        self.assertEqual(by_name["organoid1_inlet_OUT1"]["bc_type"], "PRESSURE")
        self.assertEqual(by_name["LEAK_ART_1"]["bc_values"]["Q"], [2.0e-9] * 3)
        self.assertEqual(by_name["LEAK_VEN_1"]["bc_values"]["Q"], [-3.0e-9] * 2)
        self.assertEqual(deck["boundary_conditions"][1]["bc_values"]["Q"], [0.0] * 3)


if __name__ == "__main__":
    unittest.main()
