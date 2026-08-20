import csv
import tempfile
import unittest
from pathlib import Path

import dolfinx as dfx
import numpy as np
import vtk
from basix.ufl import element
from dolfinx import fem
from mpi4py import MPI

from darcy_mixed import PerfusionSolver


class TerminalTraceOutputTests(unittest.TestCase):
    def test_terminal_branch_ids_are_matched_by_coordinates(self):
        solver = object.__new__(PerfusionSolver)
        markers = np.asarray([1000, 1001, 1002])
        marker_centroids = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
        )
        # Branch metadata is deliberately not in marker/checkpoint order.
        branch_coords = marker_centroids[[2, 0, 1]]
        branch_ids = np.asarray([39, 6, 12])

        marker_branch = solver._terminal_marker_branch_map(
            markers, marker_centroids, branch_coords, branch_ids
        )

        self.assertEqual(marker_branch, {1000: 6, 1001: 12, 1002: 39})

    def test_port_normals_follow_physical_arterial_and_venous_flow(self):
        solver = object.__new__(PerfusionSolver)
        inlet_direction = np.asarray([1.0, 0.0, 0.0])
        outlet_direction = np.asarray([0.0, 1.0, 0.0])
        inlet_center = np.asarray([[0.0, 0.0, 0.0]])
        outlet_center = np.asarray([[1.0, 0.0, 0.0]])
        solver.branch_coords_in = inlet_center.copy()
        solver.branch_coords_out = outlet_center.copy()
        solver.branch_normals_in = inlet_direction.reshape(1, 3)
        solver.branch_normals_out = outlet_direction.reshape(1, 3)
        solver._get_terminal_marker_data = lambda: (
            np.asarray([1000]),
            np.asarray([2000]),
            inlet_center,
            outlet_center,
            np.asarray([0]),
            np.asarray([0]),
        )

        solver._build_terminal_port_normal_maps()

        np.testing.assert_allclose(
            solver._inlet_distal_direction_by_marker[1000], inlet_direction
        )
        np.testing.assert_allclose(
            solver._inlet_port_normal_by_marker[1000], inlet_direction
        )
        np.testing.assert_allclose(
            solver._outlet_distal_direction_by_marker[2000], outlet_direction
        )
        np.testing.assert_allclose(
            solver._outlet_port_normal_by_marker[2000], -outlet_direction
        )

    def test_writes_selected_pressure_and_integrated_flux(self):
        mesh = dfx.mesh.create_unit_cube(MPI.COMM_WORLD, 1, 1, 1)
        tdim = mesh.topology.dim
        fdim = tdim - 1
        mesh.topology.create_entities(fdim)
        mesh.topology.create_connectivity(fdim, tdim)
        facet_to_cell = mesh.topology.connectivity(fdim, tdim)
        owned_facets = mesh.topology.index_map(fdim).size_local
        facet = next(
            index
            for index in range(owned_facets)
            if len(facet_to_cell.links(index)) == 2
        )
        facet_tags = dfx.mesh.meshtags(
            mesh,
            fdim,
            np.asarray([facet], dtype=np.int32),
            np.asarray([1000], dtype=np.int32),
        )

        solver = object.__new__(PerfusionSolver)
        solver.mesh = mesh
        solver.facet_tags = facet_tags
        solver.inlet_base_marker = 1000
        solver.outlet_base_marker = 2000
        facet_points = solver._entity_geometry_points(mesh, fdim, facet, 3)
        facet_center = np.mean(facet_points, axis=0)
        adjacent = np.asarray(facet_to_cell.links(facet), dtype=np.int32)
        cell_centers = np.asarray(
            [
                np.mean(solver._cell_geometry_points(mesh, int(cell), 3), axis=0)
                for cell in adjacent
            ]
        )
        direction = cell_centers[1] - cell_centers[0]
        direction /= np.linalg.norm(direction)
        selected_cell = int(adjacent[1])

        solver.x_inlet = facet_center.reshape(1, 3)
        solver.x_outlet = np.zeros((0, 3))
        solver.branch_coords_in = facet_center.reshape(1, 3)
        solver.branch_coords_out = np.zeros((0, 3))
        solver.branch_ids_in = np.asarray([7], dtype=int)
        solver.branch_ids_out = np.zeros(0, dtype=int)
        solver._inlet_distal_direction_by_marker = {1000: direction}
        solver._outlet_distal_direction_by_marker = {}

        pressure_space = fem.functionspace(mesh, ("DG", 0))
        pressure = fem.Function(pressure_space)
        for cell in range(mesh.topology.index_map(tdim).size_local):
            dof = int(pressure_space.dofmap.cell_dofs(cell)[0])
            pressure.x.array[dof] = 180.0 if cell == selected_cell else 250.0
        pressure.x.scatter_forward()

        velocity_space = fem.functionspace(
            mesh, element("RT", mesh.basix_cell(), 1)
        )
        velocity = fem.Function(velocity_space)
        velocity.interpolate(
            lambda x: np.repeat((2.0 * direction)[:, None], x.shape[1], axis=1)
        )
        velocity.x.scatter_forward()

        triangle_area = 0.5 * np.linalg.norm(
            np.cross(
                facet_points[1] - facet_points[0],
                facet_points[2] - facet_points[0],
            )
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            solver._write_terminal_trace_output(Path(tmpdir), pressure, velocity)
            output = next(Path(tmpdir).glob("terminal_traces*.pvtu"))
            reader = vtk.vtkXMLPUnstructuredGridReader()
            reader.SetFileName(str(output))
            reader.Update()
            data = reader.GetOutput().GetCellData()

            self.assertAlmostEqual(
                data.GetArray("pressure_tissue_trace").GetTuple1(0), 180.0
            )
            self.assertAlmostEqual(
                data.GetArray("volumetric_flux_into_tissue").GetTuple1(0),
                2.0 * float(triangle_area),
            )
            self.assertEqual(data.GetArray("terminal_marker").GetTuple1(0), 1000.0)
            self.assertEqual(data.GetArray("branch_id").GetTuple1(0), 7.0)

            with (Path(tmpdir) / "terminal_trace_summary.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["terminal_kind"], "arterial")
            self.assertEqual(int(rows[0]["branch_id"]), 7)
            self.assertEqual(int(rows[0]["terminal_marker"]), 1000)
            self.assertAlmostEqual(
                float(rows[0]["average_pressure_tissue_trace"]), 180.0
            )
            self.assertAlmostEqual(
                float(rows[0]["total_volumetric_flux_into_tissue"]),
                2.0 * float(triangle_area),
            )


if __name__ == "__main__":
    unittest.main()
