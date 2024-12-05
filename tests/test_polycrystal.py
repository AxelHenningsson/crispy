import os
import unittest

import numpy as np

import crispy


class TestPolycrystal(unittest.TestCase):
    def setUp(self):
        self.debug = False
        np.random.seed(42)

    def test_polycrystal(self):
        pc = crispy.Polycrystal(
            os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
            group_name="Fe",
        )

        path = os.path.join(
            crispy.assets._root_path, "tests/saves/FeAu_0p5_tR_ff1_grains.vtk"
        )
        pc.write(path, grains="all")

        path = os.path.join(crispy.assets._root_path, "tests/saves/FeAu_neigh_3.vtk")

        pc.write(path, neighbourhood=3)


if __name__ == "__main__":
    unittest.main()
