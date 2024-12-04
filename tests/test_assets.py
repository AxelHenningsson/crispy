import unittest

import ImageD11.grain
import ImageD11.unitcell
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import crispy


class TestAssets(unittest.TestCase):

    def setUp(self):
        self.debug = False

    def test_grainmap_id11(self):
        grains = crispy.assets.grainmap_id11()

        self.assertEqual(len(grains), 50)
        self.assertTrue(isinstance(grains[0], ImageD11.grain.grain))

        if self.debug:
            x, y, z = np.array([g.translation for g in grains]).T
            unit_cell = [4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0]
            ref_cell = ImageD11.unitcell.unitcell(unit_cell, symmetry=225)
            for g in grains:
                g.ref_unitcell = ref_cell
            ipf_ax = np.array([0.0, 0.0, 1.0])
            c = [g.get_ipf_colour(ipf_ax) for g in grains]
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x, y, z, c=c, cmap="viridis", s=50)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            plt.show()


if __name__ == "__main__":
    unittest.main()
