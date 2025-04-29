import os
import unittest

import numpy as np

import crispy


class TestLabDCTVolume(unittest.TestCase):
    def setUp(self):
        self.debug = False
        np.random.seed(42)

    def test_polycrystal(self):
        pc = crispy.GrainMap(
            os.path.join(crispy.assets._asset_path, "lab_dct_silicon.h5"),
        )

        pc = crispy.GrainMap(
            os.path.join(crispy.assets._asset_path, "lab_dct_Al1050.h5"),
        )

        pc.texturize()

    def test__volume_walker(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32) - 1
        labels[1:4, 1:-1, 1:-1] = 0
        labels[5:-1, 1:-1, 1:-1] = 1
        labels[1:-1, 1:5, 1:-1] = 2
        labels[2, 2, 2] = 3
        number_of_grains = np.max(labels) + 1

        structure_matrix = np.zeros(
            (number_of_grains, number_of_grains), dtype=np.uint32
        )
        struct = crispy._constants.DEFAULT_STRUCT
        crispy.LabDCTVolume._volume_walker(labels, structure_matrix, struct)

        self.assertTrue(isinstance(structure_matrix, np.ndarray))
        self.assertEqual(structure_matrix.shape, (number_of_grains, number_of_grains))

        grain_sizes = np.diag(structure_matrix)
        for i in range(number_of_grains):
            self.assertEqual(
                grain_sizes[i], np.sum(labels == i), "incorrect grain size computed"
            )

        np.testing.assert_allclose(structure_matrix.T, structure_matrix)

        self.assertEqual(structure_matrix[0, 1], 0, "incorrect interface area")
        self.assertEqual(structure_matrix[0, 2], 3 * 8, "incorrect interface area")
        self.assertEqual(structure_matrix[1, 2], 4 * 8, "incorrect interface area")
        self.assertEqual(structure_matrix[0, 3], 0, "incorrect interface area")
        self.assertEqual(structure_matrix[1, 3], 0, "incorrect interface area")
        self.assertEqual(structure_matrix[2, 3], 6, "incorrect interface area")

        neighbours = np.empty((number_of_grains,), dtype=np.ndarray)
        for i in range(structure_matrix.shape[0]):
            mask = structure_matrix[i, :] > 0
            mask[i] = False
            neighbours[i] = np.where(mask)[0].astype(np.uint32)

        np.testing.assert_allclose(neighbours[0], np.array([2], dtype=np.uint32))
        np.testing.assert_allclose(neighbours[1], np.array([2], dtype=np.uint32))
        np.testing.assert_allclose(neighbours[2], np.array([0, 1, 3], dtype=np.uint32))
        np.testing.assert_allclose(neighbours[3], np.array([2], dtype=np.uint32))

    def test__filter(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32) - 1
        labels[1:4, 1:-1, 1:-1] = 0
        labels[5:-1, 1:-1, 1:-1] = 1
        labels[1:-1, 1:5, 1:-1] = 2
        labels[2, 2, 2] = 3
        number_of_voids = np.sum(labels == -1)

        sizes = np.array(
            [np.sum(labels == i) for i in range(np.max(labels) + 1)], dtype=np.uint32
        )

        min_grain_size_in_voxels = 2
        crispy.LabDCTVolume._filter(labels, sizes, min_grain_size_in_voxels)

        self.assertEqual(labels[2, 2, 2], -1)

        min_grain_size_in_voxels = sizes[0] + 1
        crispy.LabDCTVolume._filter(labels, sizes, min_grain_size_in_voxels)

        self.assertEqual(np.sum(labels == -1), number_of_voids + sizes[-1] + sizes[0])


if __name__ == "__main__":
    unittest.main()
