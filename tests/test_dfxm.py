import os
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

import crispy


class TestBraggez(unittest.TestCase):
    def setUp(self):
        self.debug = False
        self.motor_bounds = {
            "mu": (0, 20),
            "omega": (-22, 22),
            "chi": (-5, 5),
            "phi": (-5, 5),
        }

    def test_init(self):
        energy = 22
        optimizer = crispy.dfxm.Braggez(energy, self.motor_bounds, epsilon=0.1)
        self.assertEqual(optimizer.energy, energy)
        self.assertEqual(optimizer.motor_bounds, self.motor_bounds)
        self.assertEqual(optimizer.epsilon, 0.1)
        self.assertFalse(optimizer.verbose)

    def test_cost_and_grad(self):
        zhat = np.array([0.0, 0.0, 1.0])
        xhat = np.array([1.0, 0.0, 0.0])
        angle = np.radians(2.23987)
        R0 = Rotation.from_rotvec(angle * xhat)
        target = R0.apply(zhat)
        nhat = zhat
        optimizer = crispy.dfxm.Braggez(
            22,
            self.motor_bounds,
        )
        c, jac = optimizer.cost_and_grad(
            np.zeros(4), nhat.reshape(3, 1), target.reshape(3, 1)
        )
        self.assertAlmostEqual(c, angle)
        self.assertEqual(jac.shape, (4,))
        self.assertTrue(np.allclose(jac, np.array([0, 0, -1, 0])))

    def test_mask(self):
        pc = crispy.Polycrystal(
            os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
            group_name="Fe",
            lattice_parameters=[4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0],
            symmetry=225,
        )
        energy = 19.1
        ub = pc.grains[15].ub
        hkls = np.array(
            [
                [2, 2, 2],
                [0, 0, 2],
            ]
        ).T
        optimizer = crispy.dfxm.Braggez(energy, self.motor_bounds)
        sol, res, success, bragg_angles = optimizer.align(
            ub, hkls, mask_unreachable=True
        )

        self.assertFalse(success[0])
        self.assertTrue(success[1])
        self.assertTrue(np.all(np.isnan(res[0])))
        self.assertTrue(np.all(np.isnan(sol[:, 0])))
        self.assertLessEqual(res[1], 1e-4)

    def test_optimizer(self):
        zhat = np.array([0.0, 0.0, 1.0])
        rotdir = np.array([1.0, 1.0, -1.0])
        rotdir /= np.linalg.norm(rotdir)
        angle = np.radians(3.234)
        R0 = Rotation.from_rotvec(angle * rotdir)
        target = R0.apply(zhat)
        nhat = zhat
        optimizer = crispy.dfxm.Braggez(22, self.motor_bounds)
        bounds = optimizer._explode_bounds(1)

        res = optimizer._minimize(
            np.zeros(4),
            nhat.reshape(3, 1),
            target.reshape(3, 1),
            bounds,
            maxiter=200,
            maxls=25,
            ftol=1e-8,
        )

        c, jac = optimizer.cost_and_grad(
            np.zeros(4), nhat.reshape(3, 1), target.reshape(3, 1)
        )

        R = optimizer.R(res.x)
        np.testing.assert_allclose(np.abs(R.apply(nhat) - target) < 1e-3, True)
        np.testing.assert_allclose(np.abs(1 - R.apply(nhat) @ target) < 1e-3, True)

        self.assertTrue(
            self.motor_bounds["mu"][0] <= res.x[0] <= self.motor_bounds["mu"][1]
        )
        self.assertTrue(
            self.motor_bounds["omega"][0] <= res.x[1] <= self.motor_bounds["omega"][1]
        )
        self.assertTrue(
            self.motor_bounds["chi"][0] <= res.x[2] <= self.motor_bounds["chi"][1]
        )
        self.assertTrue(
            self.motor_bounds["phi"][0] <= res.x[3] <= self.motor_bounds["phi"][1]
        )

    def test_good_and_bad_reflection(self):
        pc = crispy.Polycrystal(
            os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
            group_name="Fe",
            lattice_parameters=[4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0],
            symmetry=225,
        )
        energy = 19.1
        ub = pc.grains[15].ub
        hkls = np.array(
            [
                [2, 2, 2],
                [0, 0, 2],
            ]
        ).T
        optimizer = crispy.dfxm.Braggez(energy, self.motor_bounds)
        sol, res, success, bragg_angles = optimizer.align(ub, hkls)

        self.assertFalse(success[0])
        self.assertTrue(success[1])
        self.assertGreater(res[0], 5)
        self.assertLessEqual(np.radians(res[1]), 1e-4)

    def _fibonacci_sphere(self, samples):
        points = []
        phi = np.pi * (np.sqrt(5.0) - 1.0)  # golden angle in radians

        for i in range(samples):
            y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # radius at y

            theta = phi * i  # golden angle increment

            x = np.cos(theta) * radius
            z = np.sin(theta) * radius

            points.append((x, y, z))

        return np.array(points).T

    def test_many(self):
        zhat = np.array([0.0, 0.0, 1.0])
        yhat = np.array([0.0, 1.0, 0.0])
        angle = np.radians(13.05)
        R0 = Rotation.from_rotvec((-yhat) * angle)
        _target = R0.apply(zhat)

        nhat = self._fibonacci_sphere(samples=1000)

        motor_bounds = {
            "mu": (0, 20),
            "omega": (-22, 22),
            "chi": (-5, 5),
            "phi": (-5, 5),
        }

        # phi and chi on -5 to 5 increase the probaibly of success
        # with a factor of 10.

        target = np.zeros((3, nhat.shape[1]))
        target[0, :] = _target[0]
        target[1, :] = _target[1]
        target[2, :] = _target[2]

        optimizer = crispy.dfxm.Braggez(19.1, motor_bounds)
        bounds = optimizer._explode_bounds(nhat.shape[1])

        result = optimizer._minimize(
            np.zeros((4 * nhat.shape[1],)),
            nhat,
            target,
            bounds,
            maxiter=400,
            maxls=25,
            ftol=1e-8,
        )

        solution, residuals, success = optimizer._fill_unreachable(
            result,
            nhat,
            target,
            np.zeros(nhat.shape[1], dtype=bool),
            alignment_tol=1e-3,
        )

        self.assertTrue(np.sum(success) == 12)  # with 99% probability


if __name__ == "__main__":
    unittest.main()
