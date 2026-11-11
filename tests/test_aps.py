import unittest

import numpy as np

import crispy


class TestAPS(unittest.TestCase):
    def setUp(self):
        self.axes = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        self.bounds = {
            "mu": (-20, 0),
            "omega": (-40, 40),
            "chi": (-8, 8),
            "phi": (-8, 8),
            "detector_y": (-0.04, 1.2),
            "detector_x": (-1, 1),
        }
        self.angles = np.radians([-12, 17, -4, 5])

    def aps_rotation(self, angles):
        mu, omega, chi, phi = angles

        def rx(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        def ry(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def rz(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        return rx(mu) @ ry(omega) @ rx(phi) @ rz(chi)

    def grain_map(self, diffracting=True):
        rotation = self.aps_rotation(self.angles)
        theta = np.arcsin((12.398419874273968 / 30) / 4) if diffracting else 0
        c, s = np.cos(theta), np.sin(theta)
        tilt = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
        u = self.axes.T @ rotation.T @ tilt @ self.axes
        return crispy.TDXRDMap.from_array(
            [[10, 20, 30]], [np.linalg.inv(u / 4)],
            lattice_parameters=[4, 4, 4, 90, 90, 90], symmetry=225,
        )

    def test_rotation_axes_and_order(self):
        optimizer = crispy.dfxm._Braggez(30, self.bounds, beamline="APS")
        angles = np.column_stack((np.eye(4) * 0.3, self.angles, -self.angles))
        rotations = optimizer.R(angles.ravel()).as_matrix()
        for x, r in zip(angles.T, rotations):
            np.testing.assert_allclose(
                self.axes @ r @ self.axes.T, self.aps_rotation(x), atol=1e-12
            )

    def test_gradient_in_motor_order(self):
        optimizer = crispy.dfxm._Braggez(30, self.bounds, beamline="APS")
        normals = np.array([[1.0, 2, -3], [3, 1, 2], [2, -3, 1]])
        normals /= np.linalg.norm(normals, axis=0)
        targets = np.array([[2.0, -1, 3], [1, 2, 1], [-3, 3, 2]])
        targets /= np.linalg.norm(targets, axis=0)
        angles = np.column_stack((self.angles, -self.angles, self.angles / 2)).ravel()

        def cost(x):
            rotated = np.column_stack([
                self.aps_rotation(a) @ self.axes @ n
                for a, n in zip(x.reshape(4, -1).T, normals.T)
            ])
            return np.arccos(np.sum(rotated * (self.axes @ targets), axis=0)).sum()

        actual, jac = optimizer.cost_and_grad(angles, normals, targets)
        expected = np.zeros_like(jac)
        for i in range(len(angles)):
            step = np.zeros_like(angles)
            step[i] = 1e-6
            expected[i] = (cost(angles + step) - cost(angles - step)) / 2e-6
        self.assertAlmostEqual(actual, cost(angles))
        np.testing.assert_allclose(jac, expected, atol=1e-8)

    def test_diffraction_target(self):
        optimizer = crispy.dfxm._Braggez(30, self.bounds, beamline="APS")
        hkls = np.array([[0], [0], [2]])
        for eta in [0, 37, 90, -90, 180]:
            with self.subTest(eta=eta):
                _, target, theta = optimizer.get_unit_vectors(np.eye(3) / 4, hkls, eta)
                q = self.axes @ target[:, 0] / 2
                outgoing = q + np.array([0, 0, 1 / optimizer.wavelength])
                self.assertLess(q[2], 0)
                self.assertAlmostEqual(
                    np.linalg.norm(outgoing), 1 / optimizer.wavelength
                )
                np.testing.assert_allclose(
                    q[:2] / (np.cos(theta[0]) / 2),
                    [-np.sin(np.radians(eta)), np.cos(np.radians(eta))], atol=1e-12,
                )

    def test_align(self):
        pc = self.grain_map()
        bounds = {
            name: (a - 2, a + 2)
            for name, a in zip(["mu", "omega", "chi", "phi"], np.degrees(self.angles))
        }
        optimizer = crispy.dfxm._Braggez(30, bounds, beamline="APS")
        hkl = np.array([[0], [0], [2]])
        angles, residual, success, theta = optimizer.align(
            pc.grains[0].ub, hkl, alignment_tol=0.001, maxiter=800, ftol=1e-12
        )
        self.assertTrue(success[0])
        self.assertLess(residual[0], 0.001)
        rotation = self.aps_rotation(np.radians(angles[:, 0]))
        q = rotation @ self.axes @ pc.grains[0].ub @ hkl[:, 0]
        theta = np.radians(theta[0])
        expected = np.array([0, np.cos(theta), -np.sin(theta)]) / 2
        np.testing.assert_allclose(q, expected, atol=1e-5)
        for name, angle in zip(bounds, angles[:, 0]):
            self.assertGreaterEqual(angle, bounds[name][0])
            self.assertLessEqual(angle, bounds[name][1])





if __name__ == "__main__":
    unittest.main()
