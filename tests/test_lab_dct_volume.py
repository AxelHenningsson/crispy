import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import crispy


class TestLabDCTVolume(unittest.TestCase):
    def setUp(self):
        self.debug = False
        np.random.seed(42)

    def test_init(self):
        try:
            pc = crispy.GrainMap(
                crispy.assets.path.SILICON,
            )
        except Exception as e:
            self.fail(f"Failed to initialize GrainMap with lab_dct_silicon.h5: {e}")

        try:
            pc = crispy.GrainMap(
                crispy.assets.path.AL1050,
            )
        except Exception as e:
            self.fail(f"Failed to initialize GrainMap with lab_dct_Al1050.h5: {e}")

        try:
            pc.texturize()
        except Exception as e:
            self.fail(f"Failed to texturize lab_dct_Al1050.h5: {e}")

    def test_center_of_mass(self):
        pc = crispy.GrainMap(
            crispy.assets.path.SILICON,
        )
        self.assertIsInstance(
            pc.center_of_mass, np.ndarray, msg="center_of_mass is not a numpy array"
        )
        self.assertEqual(
            pc.center_of_mass.shape, (3,), msg="center_of_mass has wrong shape"
        )

    def test_translate(self):
        pc = crispy.GrainMap(
            crispy.assets.path.SILICON,
        )
        center_of_mass = pc.center_of_mass
        bounding_box = np.array(pc.bounding_box)
        centroids = np.array(pc.centroids)
        X0, Y0, Z0 = pc.X.copy(), pc.Y.copy(), pc.Z.copy()
        translation = np.array([1.0, -2.0, 3.0])
        bb_translation = np.array([1.0, 1.0, -2.0, -2.0, 3.0, 3.0])

        pc.translate(translation)

        self.assertAlmostEqual(pc.center_of_mass[0], center_of_mass[0] + translation[0])
        self.assertAlmostEqual(pc.center_of_mass[1], center_of_mass[1] + translation[1])
        self.assertAlmostEqual(pc.center_of_mass[2], center_of_mass[2] + 3.0)

        np.testing.assert_allclose(
            pc.X, X0 + 1.0, err_msg="X coordinates were not translated correctly"
        )
        np.testing.assert_allclose(
            pc.Y, Y0 - 2.0, err_msg="Y coordinates were not translated correctly"
        )
        np.testing.assert_allclose(
            pc.Z, Z0 + 3.0, err_msg="Z coordinates were not translated correctly"
        )

        np.testing.assert_allclose(
            np.array(pc.bounding_box),
            bounding_box + bb_translation,
            err_msg="bounding box was not translated correctly",
        )

        np.testing.assert_allclose(
            np.array(pc.centroids),
            centroids + translation,
            err_msg="centroids were not translated correctly",
        )

    def test__coord_sum(self):
        labels = self._get_test_labels()
        x = np.arange(labels.shape[2], dtype=np.float32)
        y = np.arange(labels.shape[1], dtype=np.float32)
        z = np.arange(labels.shape[0], dtype=np.float32)
        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        number_of_grains = np.max(labels) + 1
        cs = np.zeros((number_of_grains, 3), dtype=np.float32)
        crispy.LabDCTVolume._coord_sum(X, Y, Z, labels, cs)

        self.assertGreater(
            np.sum(np.abs(cs)), 0.0, msg="_coord_sum did not compute anything"
        )

        for i in range(number_of_grains):
            mask = labels == i
            X_sum = np.sum(X[mask])
            Y_sum = np.sum(Y[mask])
            Z_sum = np.sum(Z[mask])
            self.assertAlmostEqual(
                cs[i, 0],
                X_sum,
                msg=f"_coord_sum computed incorrect X sum for grain {i}",
            )
            self.assertAlmostEqual(
                cs[i, 1],
                Y_sum,
                msg=f"_coord_sum computed incorrect Y sum for grain {i}",
            )
            self.assertAlmostEqual(
                cs[i, 2],
                Z_sum,
                msg=f"_coord_sum computed incorrect Z sum for grain {i}",
            )

    def test_rotate(self):
        pc0 = crispy.GrainMap(
            crispy.assets.path.SILICON,
        )
        pc1 = crispy.GrainMap(
            crispy.assets.path.SILICON,
        )

        try:
            pc1.rotate(np.eye(3))
        except Exception as e:
            self.fail(f"Failed to rotate from rotation matrix: {e}")
        try:
            pc1.rotate(Rotation.from_matrix(np.eye(3)))
        except Exception as e:
            self.fail(f"Failed to rotate from rotation object: {e}")
        try:
            pc1.rotate(Rotation.from_rotvec(2 * np.pi * np.array([1.0, 0.0, 0.0])))
        except Exception as e:
            self.fail(f"Failed to rotate from rotation vector: {e}")

        self.assertTrue(
            np.allclose(pc0.X, pc1.X), msg="X coordinates were not rotated correctly"
        )
        self.assertTrue(
            np.allclose(pc0.Y, pc1.Y), msg="Y coordinates were not rotated correctly"
        )
        self.assertTrue(
            np.allclose(pc0.Z, pc1.Z), msg="Z coordinates were not rotated correctly"
        )

        np.testing.assert_allclose(
            pc0.orientations,
            pc1.orientations,
            err_msg="orientations not preserved under identity rotation",
        )

        rotation1 = Rotation.from_rotvec(np.radians(10.0) * np.array([1.0, 0.0, 0.0]))
        pc1.rotate(rotation1)
        self.assertTrue(
            np.allclose(pc0.X, pc1.X),
            msg="X not preserved after rotation around x-axis",
        )

        rotation2 = Rotation.from_rotvec(np.radians(1.0) * np.array([1.0, 0.1, 2.0]))
        pc1.rotate(rotation2)

        i, j, k = 4, 10, 3
        a0 = np.array([pc0.X[i, j, k], pc0.Y[i, j, k], pc0.Z[i, j, k]])
        a1 = np.array([pc1.X[i, j, k], pc1.Y[i, j, k], pc1.Z[i, j, k]])
        R1 = rotation1.as_matrix()
        R2 = rotation2.as_matrix()
        np.testing.assert_allclose(
            R2 @ R1 @ a0, a1, err_msg="Coordinate rotation is fundamentallywrong"
        )
        for u0, u1 in zip(pc0.orientations, pc1.orientations):
            np.testing.assert_allclose(
                R2 @ R1 @ u0, u1, err_msg="Orientation rotation is fundamentally wrong"
            )
        for g0, g1 in zip(pc0.grains, pc1.grains):
            np.testing.assert_allclose(
                R2 @ R1 @ g0.u,
                g1.u,
                err_msg="Grain rotation is fundamentally wrong",
            )
        np.testing.assert_allclose(
            R2 @ R1 @ pc0.voxel_rotations.as_matrix(),
            pc1.voxel_rotations.as_matrix(),
            err_msg="Voxel rotation is fundamentally wrong",
        )

        pc1 = crispy.GrainMap(
            crispy.assets.path.SILICON,
        )
        rotation = Rotation.from_rotvec(np.radians(10.0) * np.array([1.0, 0.0, 0.0]))
        pc1.rotate(rotation)
        pc1.rotate(rotation.inv())

        np.testing.assert_allclose(
            pc0.orientations,
            pc1.orientations,
            err_msg="orientations not preserved under yclic rotation around x-axis",
        )

        self.assertTrue(
            np.allclose(pc0.Y, pc1.Y),
            msg="Y not preserved under cyclic rotation around x-axis",
        )
        self.assertTrue(
            np.allclose(pc0.Z, pc1.Z),
            msg="Z not preserved under cyclic rotation around x-axis",
        )

    def test_center_volume(self):
        pc = crispy.GrainMap(
            os.path.join(crispy.assets._asset_path, "lab_dct_silicon.h5"),
        )

        pc.center_volume()
        center_of_mass = pc.center_of_mass
        self.assertAlmostEqual(
            center_of_mass[0],
            0.0,
            msg="center_of_mass is not centered in x after center_volume() call",
        )
        self.assertAlmostEqual(
            center_of_mass[1],
            0.0,
            msg="center_of_mass is not centered in y after center_volume() call",
        )
        self.assertAlmostEqual(
            center_of_mass[2],
            0.0,
            msg="center_of_mass is not centered in z after center_volume() call",
        )

    def test_crop(self):
        pc = crispy.GrainMap(
            crispy.assets.path.SILICON,
        )
        expected_bounding_box = np.array(pc.bounding_box) // 2
        expected_bounding_box[0] += pc.voxel_size / 10.0
        expected_bounding_box[1] -= pc.voxel_size / 5.0
        pc.crop(*expected_bounding_box)
        for v1, v2 in zip(pc.bounding_box, expected_bounding_box):
            self.assertTrue(
                np.abs(v1 - v2) < pc.voxel_size,
                "bounding box is inconsistent with expected bounding box",
            )

        dx = pc.bounding_box[1] - pc.bounding_box[0]
        dy = pc.bounding_box[3] - pc.bounding_box[2]
        dz = pc.bounding_box[5] - pc.bounding_box[4]
        expected_volume_shape = (
            np.round(np.array([dz, dy, dx]) // pc.voxel_size).astype(int) + 1
        )
        np.testing.assert_allclose(
            pc.labels.shape,
            expected_volume_shape,
            err_msg="cropped labels array has inconsistent shape with bounding box and voxel size",
        )
        np.testing.assert_allclose(
            pc.X.shape,
            expected_volume_shape,
            err_msg="cropped X coordinates array has inconsistent shape with bounding box and voxel size",
        )
        np.testing.assert_allclose(
            pc.Y.shape,
            expected_volume_shape,
            err_msg="cropped Y coordinates array has inconsistent shape with bounding box and voxel size",
        )
        np.testing.assert_allclose(
            pc.Z.shape,
            expected_volume_shape,
            err_msg="cropped Z coordinates array has inconsistent shape with bounding box and voxel size",
        )

        number_non_voids = np.sum(pc.labels != -1)
        self.assertEqual(number_non_voids, len(pc.voxel_rotations))

        self.assertEqual(
            pc.number_of_grains,
            np.max(pc.labels) + 1,
            msg="number of grains is inconsistent with max of labels array",
        )
        self.assertEqual(
            pc.number_of_grains,
            len(pc.grains),
            msg="number of grains is inconsistent with length of grains",
        )
        self.assertEqual(
            pc.number_of_grains,
            len(pc.centroids),
            msg="number of grains is inconsistent with centroids shape",
        )
        self.assertEqual(
            pc.number_of_grains,
            len(pc.neighbours),
            msg="number of grains is inconsistent with neighbours shape",
        )
        self.assertEqual(
            pc.number_of_grains,
            len(pc.interface_areas),
            msg="number of grains is inconsistent with interface_areas shape",
        )
        self.assertEqual(
            pc.number_of_grains,
            len(pc.grain_sizes),
            msg="number of grains is inconsistent with grain_sizes shape",
        )

        self._check_labels(pc.labels)

    def test__volume_walker(self):
        labels = self._get_test_labels()
        number_of_grains = np.max(labels) + 1

        structure_matrix = np.zeros(
            (number_of_grains, number_of_grains), dtype=np.uint32
        )
        struct = crispy.CONSTANTS.DEFAULT_STRUCT
        crispy.LabDCTVolume._volume_walker(labels, structure_matrix, struct)

        self.assertTrue(isinstance(structure_matrix, np.ndarray))
        self.assertEqual(structure_matrix.shape, (number_of_grains, number_of_grains))

        grain_sizes = np.diag(structure_matrix)
        for i in range(number_of_grains):
            self.assertEqual(
                grain_sizes[i], np.sum(labels == i), "incorrect grain size computed"
            )

        np.testing.assert_allclose(
            structure_matrix.T,
            structure_matrix,
            err_msg="structure matrix was not symmetric",
        )

        self.assertEqual(structure_matrix[0, 1], 0, msg="incorrect interface area")
        self.assertEqual(structure_matrix[0, 2], 3 * 8, msg="incorrect interface area")
        self.assertEqual(structure_matrix[1, 2], 4 * 8, msg="incorrect interface area")
        self.assertEqual(structure_matrix[0, 3], 0, msg="incorrect interface area")
        self.assertEqual(structure_matrix[1, 3], 0, msg="incorrect interface area")
        self.assertEqual(structure_matrix[2, 3], 6, msg="incorrect interface area")

        neighbours = np.empty((number_of_grains,), dtype=np.ndarray)
        for i in range(structure_matrix.shape[0]):
            mask = structure_matrix[i, :] > 0
            mask[i] = False
            neighbours[i] = np.where(mask)[0].astype(np.uint32)

        np.testing.assert_allclose(
            neighbours[0],
            np.array([2], dtype=np.uint32),
            err_msg="incorrect neighbours from _volume_walker",
        )
        np.testing.assert_allclose(
            neighbours[1],
            np.array([2], dtype=np.uint32),
            err_msg="incorrect neighbours from _volume_walker",
        )
        np.testing.assert_allclose(
            neighbours[2],
            np.array([0, 1, 3], dtype=np.uint32),
            err_msg="incorrect neighbours from _volume_walker",
        )
        np.testing.assert_allclose(
            neighbours[3],
            np.array([2], dtype=np.uint32),
            err_msg="incorrect neighbours from _volume_walker",
        )

    def test__filter(self):
        labels = self._get_test_labels()
        number_of_voids = np.sum(labels == -1)

        sizes = np.array(
            [np.sum(labels == i) for i in range(np.max(labels) + 1)], dtype=np.uint32
        )

        min_grain_size_in_voxels = 2
        crispy.LabDCTVolume._filter(labels, sizes, min_grain_size_in_voxels)

        self.assertEqual(labels[2, 2, 2], -1, msg="failed to filter grain with size 1")

        min_grain_size_in_voxels = sizes[0] + 1
        crispy.LabDCTVolume._filter(labels, sizes, min_grain_size_in_voxels)

        self.assertEqual(
            np.sum(labels == -1),
            number_of_voids + sizes[-1] + sizes[0],
            msg="failed to filter the correct number of voxels",
        )

    def test__reset_labels(self):
        pc = crispy.GrainMap(
            os.path.join(crispy.assets._asset_path, "lab_dct_silicon.h5"),
        )  # dummy polycrystal to access class functions.
        labels = self._get_test_labels()
        original_labels = labels.copy()
        min_grain_size_in_voxels = 2
        sizes = np.array(
            [np.sum(labels == i) for i in range(np.max(labels) + 1)], dtype=np.uint32
        )
        crispy.LabDCTVolume._filter(labels, sizes, min_grain_size_in_voxels)
        pc._reset_labels(labels)
        self._check_labels(labels)
        self.assertTrue(
            np.max(labels) != np.max(original_labels), msg="labels were not reset"
        )

    def _get_test_labels(self):
        labels = np.zeros((10, 10, 10), dtype=np.int32) - 1
        labels[1:4, 1:-1, 1:-1] = 0
        labels[5:-1, 1:-1, 1:-1] = 1
        labels[1:-1, 1:5, 1:-1] = 2
        labels[2, 2, 2] = 3
        return labels

    def _check_labels(self, labels):
        unique_labels = np.unique(labels)
        min_label = np.min(unique_labels)

        self.assertTrue(
            min_label == -1 or min_label == 0, msg="Labels must start from -1 or 0"
        )
        expected_labels = np.arange(len(unique_labels)) + np.min(unique_labels)

        self.assertTrue(
            np.all(unique_labels == expected_labels),
            msg=f"Labels are not sequential in the range [-1, number_of_grains-1] with -1 for void the expected labels was {expected_labels} but the unique labels were {unique_labels}",
        )


if __name__ == "__main__":
    unittest.main()
