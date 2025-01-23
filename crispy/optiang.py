import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

import crispy


class OptiBragg(object):
    def __init__(self, energy, motor_bounds, epsilon=1e-6):
        self.energy = energy
        self.wavelength = crispy.laue.keV_to_angstrom(energy)
        self._bounds = self.unpack_bounds(motor_bounds)
        self.motor_bounds = motor_bounds
        self.xhat, self.yhat, self.zhat = np.eye(3)
        self._set_rotational_increments(epsilon)
        self.epsilon = epsilon

    def _set_rotational_increments(self, epsilon):
        self.dR_mu = self.R_mu(epsilon)
        self.dR_omega = self.R_omega(epsilon)
        self.dR_chi = self.R_chi(epsilon)
        self.dR_phi = self.R_phi(epsilon)

    def R_phi(self, phi):
        return Rotation.from_rotvec(phi * self.yhat)

    def R_chi(self, chi):
        return Rotation.from_rotvec(chi * self.xhat)

    def R_mu(self, mu):
        return Rotation.from_rotvec(mu * (-self.yhat))

    def R_omega(self, omega):
        return Rotation.from_rotvec(omega * self.xhat)

    def R(self, x):
        mu, omega, chi, phi = x
        return self.R_mu(mu) * self.R_omega(omega) * self.R_chi(chi) * self.R_phi(phi)

    def cost(self, rotation, nhat, target):
        return np.arccos(rotation.apply(nhat) @ target)

    def cost_and_jac(self, x, nhat, target):
        mu, omega, chi, phi = x
        R1 = self.R_mu(mu)
        R2 = self.R_omega(omega)
        R3 = self.R_chi(chi)
        R4 = self.R_phi(phi)

        c0 = self.cost(R1 * R2 * R3 * R4, nhat, target)

        jac = np.zeros((4,))
        jac[0] = self.cost(self.dR_mu * R1 * R2 * R3 * R4, nhat, target) - c0
        jac[1] = self.cost(R1 * self.dR_omega * R2 * R3 * R4, nhat, target) - c0
        jac[2] = self.cost(R1 * R2 * self.dR_chi * R3 * R4, nhat, target) - c0
        jac[3] = self.cost(R1 * R2 * R3 * self.dR_phi * R4, nhat, target) - c0
        jac /= self.epsilon

        return c0, jac

    def get_unit_vectors(self, ub, hkl):
        Q = ub @ hkl
        nhat = Q / np.linalg.norm(Q)
        d = 1 / np.linalg.norm(Q)
        theta = np.arcsin(self.wavelength / (2 * d))
        target = self.R_mu(theta).apply(self.zhat)
        return nhat, target

    def unpack_bounds(self, motor_bounds):
        return np.radians(
            [
                motor_bounds["mu"],
                motor_bounds["omega"],
                motor_bounds["chi"],
                motor_bounds["phi"],
            ]
        )

    def _minimize(self, x0, nhat, target):
        return minimize(
            self.cost_and_jac,
            x0=x0,
            args=(nhat, target),
            method="L-BFGS-B",
            jac=True,
            bounds=self._bounds,
            options={"maxls": 25, "maxfev": 50, "maxiter": 50, "ftol": 1e-6},
        )

    def run(self, ub, hkl):
        nhat, target = self.get_unit_vectors(ub, hkl)
        res = self._minimize(np.zeros((4,)), nhat, target)
        if res.fun > 1e-4:
            res = self._minimize(res.x, nhat, target)
        return np.degrees(res.x), np.degrees(res.fun), res.fun < 1e-4


if __name__ == "__main__":
    import os

    import numpy as np

    import crispy

    pc = crispy.Polycrystal(
        os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
        group_name="Fe",
        lattice_parameters=[4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0],
        symmetry=225,
    )

    energy = 19.1
    motor_bounds = {
        "mu": (0, 20),
        "omega": (-22, 22),
        "chi": (-5, 5),
        "phi": (-5, 5),
    }

    ub = pc.grains[15].ub
    hkl = np.array([0, 0, 2])
    optimizer = OptiBragg(energy, motor_bounds)

    import cProfile
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    sol, res, success = optimizer.run(ub, hkl)

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")
    print("")
    print("======================RESULTS======================")
    for m, title in zip(sol, ["mu", "omega", "chi", "phi"]):
        print("   " + title + ": ", np.round(m, 4), " degrees")
    print("   res ang:", res, " degrees")
    print("   success:", success)
    print("===================================================")
