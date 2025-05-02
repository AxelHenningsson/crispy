import os
import unittest

import numpy as np

import crispy


class TestTDXRDMap(unittest.TestCase):
    def setUp(self):
        self.debug = False
        np.random.seed(42)

    def test_polycrystal(self):
        lattice_parameters = [2.866, 2.866, 2.866, 90.0, 90.0, 90.0]

        pc = crispy.GrainMap(
            crispy.assets.path.FEAU,
            group_name="Fe",
            lattice_parameters=lattice_parameters,
            symmetry=225,
        )

        pc.tesselate()

        path = os.path.join(
            crispy.assets._root_path, "tests/saves/FeAu_0p5_tR_ff1_grains.vtk"
        )
        pc.write(path, grains="all")

        path = os.path.join(crispy.assets._root_path, "tests/saves/FeAu_neigh_3.vtk")

        pc.write(path, neighbourhood=3)


if __name__ == "__main__":
    unittest.main()
