import os
import tempfile
import unittest

import numpy as np
from ImageD11 import transform

import crispy


class TestMIDAS(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = os.path.join(self.tempdir.name, "Grains.csv")
        self.columns = (
            ["GrainID"]
            + [f"O{i}{j}" for i in range(1, 4) for j in range(1, 4)]
            + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma"]
            + ["DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
            + [f"eFab{i}{j}" for i in range(1, 4) for j in range(1, 4)]
            + [f"eKen{i}{j}" for i in range(1, 4) for j in range(1, 4)]
            + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2"]
        )
        row = (
            [60, 0, -1, 0, 1, 0, 0, 0, 0, 1]
            + [10, 20, 30, 4.04, 4.04, 4.04, 90, 90, 90]
            + [2, 0.02, 0.01, 12, 0.95]
            + [100, 20, 30, 20, 200, 40, 30, 40, 300]
            + [400, 50, 60, 50, 500, 70, 60, 70, 600]
            + [5, 1, 0, 0, np.pi / 2]
        )
        self.data = np.array([row, row], dtype=float)
        self.data[1, 0] = 51
        self.data[1, 10:13] = [-20, -30, -40]
        self.reference_cell = [4, 4, 4, 90, 90, 90]
        self.symmetry = 225

    def write_grains(self, metadata=True, num_phases=1):
        with open(self.path, "w") as f:
            f.write(f"%NumGrains {len(self.data)}\n%NumPhases {num_phases}\n")
            if metadata:
                f.write(f"%PhaseInfo\n%\tSpaceGroup:{self.symmetry}\n")
                cell = " ".join(str(x) for x in self.reference_cell)
                f.write(f"%\tLattice Parameter: {cell}\n")
            f.write("%" + "\t".join(self.columns) + "\n")
            np.savetxt(f, self.data, delimiter="\t", fmt="%.9f")

    def test_orientation_and_fitted_lattice(self):
        self.write_grains()
        pc = crispy.GrainMap(self.path)
        g = pc.grains[0]
        np.testing.assert_allclose(g.u, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)
        np.testing.assert_allclose(g.ubi @ [0, 2 / 4.04, 0], [2, 0, 0], atol=1e-12)
        np.testing.assert_allclose(g.unitcell, [4.04, 4.04, 4.04, 90, 90, 90])
        np.testing.assert_allclose(pc.strain[0], np.eye(3) * 0.01, atol=1e-12)
        np.testing.assert_allclose(pc.centroids, [[10, 20, 30], [-20, -30, -40]])
        np.testing.assert_allclose(
            pc.reference_cell.lattice_parameters, self.reference_cell
        )
        self.assertEqual(pc.reference_cell.symmetry, 225)

    def test_hexagonal_reciprocal_vectors(self):
        self.reference_cell = [3, 3, 5, 90, 90, 120]
        self.symmetry = 194
        self.data[:, 13:19] = self.reference_cell
        self.write_grains()
        pc = crispy.GrainMap(self.path)
        # MIDAS: a* along x, b* in xy; O is a positive 90 degree rotation about z.
        q = np.array(
            [[0, -1 / 3, 0],
             [2 / (3 * np.sqrt(3)), 1 / (3 * np.sqrt(3)), 0],
             [0, 0, 1 / 5]]
        )
        np.testing.assert_allclose(pc.grains[0].ub, q, atol=1e-12)
        np.testing.assert_allclose(pc.grains[0].ubi @ q, np.eye(3), atol=1e-12)

    def test_rotation_stage_frame(self):
        self.write_grains()
        g = crispy.GrainMap(self.path).grains[0]
        optimizer = crispy.dfxm._Braggez(30, {})
        omega, wedge = 90, 30
        rotation = optimizer.R(np.radians([wedge, omega, 0, 0])).as_matrix()[0]
        # MIDAS: R_y(-wedge) R_z(omega), for both positions and reciprocal vectors.
        position = [-10 * np.sqrt(3) - 15, 10, -10 + 15 * np.sqrt(3)]
        q_lab = np.array([-np.sqrt(3), 0, -1]) / 4.04
        np.testing.assert_allclose(rotation @ g.translation, position, atol=1e-12)
        np.testing.assert_allclose(rotation @ g.ub @ [2, 0, 0], q_lab, atol=1e-12)
        origin = transform.compute_grain_origins(
            np.array([omega]), wedge=wedge,
            t_x=g.translation[0], t_y=g.translation[1], t_z=g.translation[2],
        )
        np.testing.assert_allclose(origin[:, 0], position, atol=1e-12)
        q_stage = transform.compute_g_from_k(
            q_lab[:, None], np.array([omega]), wedge=wedge
        )
        np.testing.assert_allclose(g.ubi @ q_stage[:, 0], [2, 0, 0], atol=1e-12)

    def test_beam_and_eta(self):
        self.write_grains()
        g = crispy.GrainMap(self.path).grains[0]
        optimizer = crispy.dfxm._Braggez(30, {})
        hkls = np.array([[2], [0], [0]])
        length = np.linalg.norm(g.ub @ hkls)
        beam = np.array([[1], [0], [0]]) / optimizer.wavelength
        for eta, yz in [(0, [0, 1]), (90, [-1, 0]), (-90, [1, 0]), (180, [0, -1])]:
            with self.subTest(eta=eta):
                _, target, theta = optimizer.get_unit_vectors(g.ub, hkls, eta)
                q = target * length
                k = transform.compute_k_vectors(
                    np.degrees(2 * theta), np.array([eta]), optimizer.wavelength
                )
                np.testing.assert_allclose(q, k, atol=1e-12)
                np.testing.assert_allclose(
                    target[1:, 0] / np.cos(theta[0]), yz, atol=1e-12
                )
                self.assertLess(q[0, 0], 0)
                self.assertAlmostEqual(np.linalg.norm(beam + q), np.linalg.norm(beam))

    def test_source_metadata_and_internal_ids(self):
        self.write_grains()
        pc = crispy.GrainMap(self.path)
        self.assertEqual([g.id for g in pc.grains], [0, 1])
        self.assertEqual([g.midas_id for g in pc.grains], [60, 51])
        self.assertEqual(pc.grains[0].midas["Confidence"], 0.95)
        self.assertEqual(pc.grains[0].midas["eFab12"], 20)
        self.assertEqual(pc.grains[0].midas["eKen33"], 600)
        pc.tesselate()
        self.assertTrue(all(g.volume > 0 for g in pc.grains))

    def test_midas_c_reference_vectors(self):
        path = os.path.join(
            os.path.dirname(__file__), "data", "midas_reciprocal_vectors.txt"
        )
        reference = np.loadtxt(path)
        u = np.array([[0.36, -0.48, 0.8], [0.8, 0.6, 0], [-0.48, 0.64, 0.6]])
        self.data = self.data[:1]
        self.data[:, 1:10] = u.ravel()
        for rows in reference.reshape(-1, 3, 13):
            with self.subTest(cell=rows[0, 1:7]):
                self.symmetry = int(rows[0, 0])
                self.reference_cell = rows[0, 1:7].tolist()
                self.data[:, 13:19] = self.reference_cell
                self.write_grains()
                pc = crispy.TDXRDMap.from_midas(self.path)
                hkls = rows[:, 7:10].T
                q = u @ rows[:, 10:13].T
                np.testing.assert_allclose(pc.grains[0].ub @ hkls, q, atol=1e-12)
                np.testing.assert_allclose(pc.grains[0].ubi @ q, hkls, atol=1e-12)

    def test_extended_reordered_columns(self):
        extra = [
            f"Diff{x}{stage}"
            for stage in ["Pre", "Post"] for x in ["Pos", "Ome", "Angle"]
        ]
        self.columns += extra
        self.columns[0] = "ID"
        self.data = np.column_stack((self.data, np.full((2, 6), np.nan)))
        self.columns = self.columns[::-1]
        self.data = self.data[:, ::-1]
        self.write_grains()
        pc = crispy.GrainMap(self.path)
        self.assertEqual(pc.grains[0].midas_id, 60)
        self.assertTrue(np.isnan(pc.grains[0].midas["DiffPosPost"]))
        np.testing.assert_allclose(pc.grains[0].translation, [10, 20, 30])

    def test_reference_cell_override_and_single_row(self):
        self.data = self.data[:1]
        self.write_grains(metadata=False)
        pc = crispy.GrainMap(
            self.path, lattice_parameters=self.reference_cell, symmetry=225
        )
        self.assertEqual(pc.number_of_grains, 1)
        np.testing.assert_allclose(pc.strain[0], np.eye(3) * 0.01, atol=1e-12)

    def test_missing_reference_cell(self):
        self.write_grains(metadata=False)
        with self.assertRaisesRegex(ValueError, "reference"):
            crispy.GrainMap(self.path)

    def test_invalid_space_group(self):
        self.write_grains()
        for symmetry in [-1, 0, 231, 225.5]:
            with self.subTest(symmetry=symmetry):
                with self.assertRaisesRegex(ValueError, "space group"):
                    crispy.GrainMap(self.path, symmetry=symmetry)

    def test_missing_fitted_lattice(self):
        self.columns[13] = "E11"
        self.write_grains()
        with self.assertRaisesRegex(ValueError, "column"):
            crispy.GrainMap(self.path)

    def test_multiple_phases(self):
        self.write_grains(num_phases=2)
        with self.assertRaisesRegex(ValueError, "phase"):
            crispy.GrainMap(self.path)
        self.data[1, self.columns.index("PhaseNr")] = 2
        self.write_grains()
        with self.assertRaisesRegex(ValueError, "phase"):
            crispy.GrainMap(self.path)

    def test_empty_map(self):
        self.data = self.data[:0]
        self.write_grains()
        with self.assertRaisesRegex(ValueError, "grain"):
            crispy.GrainMap(self.path)

    def test_invalid_orientation(self):
        for value in [2, -1, np.nan]:
            with self.subTest(value=value):
                self.data[:, 9] = value
                self.write_grains()
                with self.assertRaisesRegex(ValueError, "orientation"):
                    crispy.GrainMap(self.path)

    def test_invalid_fitted_cell(self):
        for cell in [[-4, 4, 4, 90, 90, 90], [4, 4, 4, 10, 10, 170]]:
            with self.subTest(cell=cell):
                self.data[:, 13:19] = cell
                self.write_grains()
                with self.assertRaisesRegex(ValueError, "lattice"):
                    crispy.GrainMap(self.path)

    def test_invalid_reference_cell(self):
        self.reference_cell = [4, 4, 4, 10, 10, 170]
        self.write_grains()
        with self.assertRaisesRegex(ValueError, "lattice"):
            crispy.GrainMap(self.path)

    def test_invalid_grain_ids(self):
        for value in [60, 1.5, np.nan]:
            with self.subTest(value=value):
                self.data[1, 0] = value
                self.write_grains()
                with self.assertRaisesRegex(ValueError, "ID"):
                    crispy.GrainMap(self.path)


if __name__ == "__main__":
    unittest.main()
