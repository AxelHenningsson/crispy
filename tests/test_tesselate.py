import unittest

import ImageD11.grain
import ImageD11.unitcell
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import crispy


class TestTesselate(unittest.TestCase):

    def setUp(self):
        self.debug = False

    def test_tesselate(self):
        grains = crispy.assets.grainmap_id11()

        x, y, z = np.array([g.translation for g in grains]).T
        xg = np.linspace(x.min(), x.max(), 16)
        yg = np.linspace(y.min(), y.max(), 16)
        zg = np.linspace(z.min(), z.max(), 16)
        tesselation = crispy.tesselate.voroni(grains, (xg, yg, zg))

        self.assertTrue(isinstance(tesselation, np.ndarray))
        np.testing.assert_allclose(np.unique(tesselation), np.arange(0, len(grains)))
        self.assertEqual(tesselation.shape[0], len(xg))
        self.assertEqual(tesselation.shape[1], len(yg))
        self.assertEqual(tesselation.shape[2], len(zg))

        seeds = np.random.rand(10, 3)
        x, y, z = seeds.T
        xg = np.linspace(x.min(), x.max(), 16)
        yg = np.linspace(y.min(), y.max(), 16)
        zg = np.linspace(z.min(), z.max(), 16)
        tesselation = crispy.tesselate.voroni(seeds, (xg, yg, zg))

        self.assertTrue(isinstance(tesselation, np.ndarray))
        np.testing.assert_allclose(np.unique(tesselation), np.arange(0, len(seeds)))
        self.assertEqual(tesselation.shape[0], len(xg))
        self.assertEqual(tesselation.shape[1], len(yg))
        self.assertEqual(tesselation.shape[2], len(zg))

if __name__ == "__main__":
    unittest.main()
