import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("3d_transport.py")
SPEC = importlib.util.spec_from_file_location("transport_3d", MODULE_PATH)
TRANSPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRANSPORT)
NetworkTransport1D = TRANSPORT.NetworkTransport1D


class TwoWayNetworkTransportTests(unittest.TestCase):
    def _network(self, flow):
        temporary = tempfile.TemporaryDirectory()
        csv_path = Path(temporary.name) / "branching.csv"
        fields = [
            "Parent",
            "Child1",
            "Child2",
            "Length",
            "Radius",
            "ProximalNodeIndex",
            "DistalNodeIndex",
            "proximalCoordsX",
            "proximalCoordsY",
            "proximalCoordsZ",
            "distalCoordsX",
            "distalCoordsY",
            "distalCoordsZ",
        ]
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow(
                {
                    "Parent": "",
                    "Child1": "",
                    "Child2": "",
                    "Length": 1.0,
                    "Radius": 0.1,
                    "ProximalNodeIndex": 0,
                    "DistalNodeIndex": 1,
                    "proximalCoordsX": 0.0,
                    "proximalCoordsY": 0.0,
                    "proximalCoordsZ": 0.0,
                    "distalCoordsX": 1.0,
                    "distalCoordsY": 0.0,
                    "distalCoordsZ": 0.0,
                }
            )
        network = NetworkTransport1D(
            str(csv_path),
            np.asarray([[1.0, 0.0, 0.0]]),
            np.asarray([flow]),
            diffusivity=0.0,
            dt=0.1,
            cells_per_segment=4,
            initial_concentration=0.0,
            label="test",
        )
        network._temporary_directory = temporary
        return network

    def test_trial_does_not_mutate_and_commit_is_explicit(self):
        network = self._network(flow=2.0)
        old = network.concentration.copy()

        trial = network.solve_trial(
            [0.25], root_concentration=1.0, root_mode="dirichlet"
        )

        np.testing.assert_allclose(network.concentration, old)
        self.assertLess(abs(trial["balance_residual"]), 1.0e-12)
        network.commit_trial(trial)
        np.testing.assert_allclose(network.concentration, trial["concentration"])

    def test_venous_natural_root_is_conservative(self):
        network = self._network(flow=-2.0)

        trial = network.solve_trial([0.75], root_mode="natural_outflow")

        self.assertLess(trial["root_flux_rate"], 0.0)
        self.assertLess(trial["terminal_flux_rates"][0], 0.0)
        self.assertLess(abs(trial["balance_residual"]), 1.0e-12)
        self.assertAlmostEqual(
            trial["terminal_flux_rates"][0],
            trial["terminal_advective_flux_rates"][0],
        )

    def test_steady_trial_removes_network_storage(self):
        network = self._network(flow=2.0)

        trial = network.solve_trial(
            [0.25],
            root_concentration=1.0,
            root_mode="dirichlet",
            steady_state=True,
        )

        self.assertEqual(trial["storage_rate"], 0.0)
        self.assertLess(abs(trial["balance_residual"]), 1.0e-12)

    def test_steady_natural_outflow_is_conservative(self):
        network = self._network(flow=-2.0)

        trial = network.solve_trial(
            [0.75], root_mode="natural_outflow", steady_state=True
        )

        self.assertEqual(trial["storage_rate"], 0.0)
        self.assertLess(abs(trial["balance_residual"]), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
