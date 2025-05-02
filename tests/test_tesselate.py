import unittest

import numpy as np

import crispy


class TestTesselate(unittest.TestCase):
    def setUp(self):
        self.debug = False
        np.random.seed(42)

    def test_tesselate(self):
        grains = crispy.assets.grain_map.tdxrd_map().grains
        mesh = crispy._tesselate.voronoi(grains)
        self.assertEqual(np.max(mesh.cell_data["grain_id"]), len(grains) - 1)
        self.assertEqual(np.max(mesh.cell_data["surface_grain"]), 1)

        for n in mesh.neighbours:
            self.assertLess(len(n), 16)  # verified by paraview

        points = np.random.rand(100, 3)
        mesh = crispy._tesselate.voronoi(points)
        self.assertEqual(np.max(mesh.cell_data["grain_id"]), points.shape[0] - 1)
        self.assertEqual(np.max(mesh.cell_data["surface_grain"]), 1)
        self.assertEqual(np.min(mesh.cell_data["surface_grain"]), 0)
        self.assertEqual(len(mesh.neighbours), points.shape[0])
        self.assertLess(len(mesh.neighbours[0]), points.shape[0])


if __name__ == "__main__":
    unittest.main()
