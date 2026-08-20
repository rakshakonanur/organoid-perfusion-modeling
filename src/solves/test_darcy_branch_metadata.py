import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from darcy import PerfusionSolver


class TerminalBranchMetadataTests(unittest.TestCase):
    def _load(self, rows):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "branchingData.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            solver = object.__new__(PerfusionSolver)
            return solver._load_terminal_branch_metadata(path)

    @staticmethod
    def _terminal_row(**overrides):
        row = {
            "proximalCoordsX": 1.0,
            "proximalCoordsY": 2.0,
            "proximalCoordsZ": 3.0,
            "distalCoordsX": 3.0,
            "distalCoordsY": 2.0,
            "distalCoordsZ": 3.0,
            "W1": -1.0,
            "W2": 0.0,
            "W3": 0.0,
            "Child1": "",
            "Child2": "",
            "Index": 7,
        }
        row.update(overrides)
        return row

    def test_endpoint_axis_takes_precedence_over_w(self):
        _coords, ids, directions = self._load([self._terminal_row(W1=1.0)])

        np.testing.assert_array_equal(ids, [7])
        np.testing.assert_allclose(directions, [[1.0, 0.0, 0.0]])

    def test_endpoint_axis_handles_mixed_w_conventions(self):
        rows = [
            self._terminal_row(Index=4, W1=-1.0),
            self._terminal_row(Index=8, W1=1.0, distalCoordsX=-1.0),
        ]

        _coords, _ids, directions = self._load(rows)

        np.testing.assert_allclose(
            directions,
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        )

    def test_w_only_fallback_is_reversed_to_proximal_to_distal(self):
        row = self._terminal_row()
        for key in ("proximalCoordsX", "proximalCoordsY", "proximalCoordsZ"):
            row.pop(key)

        _coords, _ids, directions = self._load([row])

        np.testing.assert_allclose(directions, [[1.0, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
