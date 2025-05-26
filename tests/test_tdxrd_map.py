import os
import unittest

import numpy as np

import crispy
from xfab import tools

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

    def test_from_array(self):
        # Monoclinic TiO2: 
        # Reference: https://next-gen.materialsproject.org/materials/mp-430#crystal_structure
        lattice_parameters = [4.81, 4.81, 5.04, 90, 99.86, 90 ]  # in angstroms
        symmetry = 14
        B = tools.form_b_mat(lattice_parameters) / (2 * np.pi)
        U = np.eye(3)
        translations = np.array([[0, 0, 0]])
        ubi_matrices = np.linalg.inv(U @ B).reshape(1, 3, 3)
        polycrystal = crispy.TDXRDMap.from_array(
            translations = translations,
            ubi_matrices = ubi_matrices,
            lattice_parameters = lattice_parameters ,
            symmetry= symmetry,
        )
        self.assertEqual(polycrystal.number_of_grains, 1)
        self.assertEqual(polycrystal.grains[0].ref_unitcell.symmetry, symmetry)


if __name__ == "__main__":
    unittest.main()
