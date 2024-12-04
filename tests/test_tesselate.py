import unittest

import numpy as np

import crispy


class TestTesselate(unittest.TestCase):
    def setUp(self):
        self.debug = False
        np.random.seed(42)

    def test_tesselate(self):
        grains = crispy.assets.grainmap_id11()
        mesh = crispy.tesselate.voronoi(grains)
        self.assertEqual(np.max(mesh.cell_data["grain_id"]), len(grains) - 1)
        self.assertEqual(np.max(mesh.cell_data["surface_grain"]), 1)

        points = np.random.rand(100, 3)
        mesh = crispy.tesselate.voronoi(points)
        self.assertEqual(np.max(mesh.cell_data["grain_id"]), points.shape[0] - 1)
        self.assertEqual(np.max(mesh.cell_data["surface_grain"]), 1)
        self.assertEqual(np.min(mesh.cell_data["surface_grain"]), 0)

        # print(mesh.points)
        # mask = mesh.cell_data["grain_id"][0] == 1
        # print(mesh.points.shape)
        # print(np.array(mesh.cells_dict["polygon"]).shape)
        # simplices = np.array(mesh.cells_dict["polygon"])[mask]
        # faces = mesh.points[simplices]
        # print(simplices.shape)


if __name__ == "__main__":
    unittest.main()
