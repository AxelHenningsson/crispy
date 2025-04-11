import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from xfab import tools


class Goniometer:
    """
    A class to represent a Diffraction X-ray Microscopy (DFXM) experiment setup.

    This class is designed to find the angular settings required of the microscope in order
    to make a given set of Miller indices come into diffraction at some rotation of the turntable.

    This is usefull for situations when we want to reach several reflection in DFXM
    without re-mounitng the sample

    Attributes:
        polycrystal (:obj:`crispy.Polycrystal`): The polycrystal object mounted on the goniometer.
        energy (:obj:`float`): Photon energy in keV.
        detector_distance (:obj:`float`): The detector distance in meters.
        motor_bounds (dict): Dictionary containing the motor bounds of the goniometer/hexapod
            keys are given in the below example dictionary:
            e.g. {"mu": (0, 20),
                "omega": (-22, 22),
                "chi": (-5, 5),
                "phi": (-5, 5),
                "detector_z": (-0.04, 1.96),
                "detector_y": (-0.169, 1.16),}
            These are in units of degrees (sample stage)
            and meters (for detector translations)
    """

    def __init__(self, polycrystal, energy, detector_distance, motor_bounds):
        self.polycrystal = polycrystal
        self.energy = energy
        self.wavelength = 12.398419874273968 / self.energy
        self.detector_distance = detector_distance
        self.motor_bounds = motor_bounds

    def _get_hkls(self):
        th_min = 0
        th_max = (
            np.arctan(self.motor_bounds["detector_z"][1] / self.detector_distance) / 2.0
        )
        unit_cell = self.polycrystal.reference_cell.lattice_parameters
        sgno = self.polycrystal.reference_cell.symmetry
        sintlmin = np.sin(th_min) / self.wavelength
        sintlmax = np.sin(th_max) / self.wavelength
        hkls = tools.genhkl_all(unit_cell, sintlmin, sintlmax, sgno=sgno).T
        return hkls

    def find_reflections(
        self,
        alignment_tol=1e-5,
        maxiter=None,
        maxls=25,
        ftol=None,
        mask_unreachable=False,
    ):
        """Find all reflecitions in the polycrystal reachable by the goniometer in DFXM

        Running this function will find all reflections in the polycrystal that are reachable
        by the goniometer in DFXM. The reflections are found by running an optimization
        algorithm that aligns the lattice normals with the target vectors. The target vectors
        are defined as the z-axis rotated by the Bragg angle in the xz-plane into the reverse
        direction of the beam which is assumed to propagate in the positive x-direction.
        The reflections are stored in the polycrystal object as a dictionary with the keys
        "hkl", "mu", "omega", "chi", "phi", "residual", and "theta". The values are the
        corresponding values for each reflection. The reflections are stored in the order
        they are found in the optimization. To access the reflections of a grain use:

            grain.dfxm["hkl"] # hkl vectors that can diffract
            grain.dfxm["mu"] # mu angles at which diffraction occurs
            grain.dfxm["omega"] # omega angles at which diffraction occurs
            grain.dfxm["chi"] # chi angles at which diffraction occurs
            grain.dfxm["phi"] # phi angles at which diffraction occurs
            grain.dfxm["residual"] # residuals, radians between the target vector Q-vector and the algined lattice normal.
            grain.dfxm["theta"] # Bragg angles

        Args:

            alignment_tol (float): The tolerance for determining if the optimization
                managed to align the reflection for diffraction in unit of radians,
                representing the misalignment of the lattice normal from the target vector.
                Default is 1e-5.
            maxiter (int): The maximum number of iterations for the optimization performed
                by L-BFGS-B. Default is None, which will be set to 200 * N, where N is the
                number of reflections.
            maxls (int): The maximum number of line search iterations for the optimization per
                iteration of L-BFGS-B. Default is 25.
            ftol (float): The tolerance for the optimization cost function before termination.
                Default is None, which will be set to 1e-8 / N, where N is the number of
                reflections.
            mask_unreachable (bool): If True, mask reflections that are unreachable
                by the goniometer/hexapod. A reflection is unreachable if the misalignment
                of the lattice normal from the target vector is greater than the maximum sum
                of the absolute values of the motor bounds. Default is False.
                Maskin unreachable reflections speed up the optimization by simply not
                considering them. When masking is enabled, the solution and residuals
                success arrays will hold np.nan values for the unreachable reflections.
        """
        hkls = self._get_hkls()
        bez = Braggez(self.energy, self.motor_bounds)
        for i, g in enumerate(self.polycrystal.grains):
            goni_angles, residual, success, theta = bez.align(
                g.ub,
                hkls,
                alignment_tol=1e-5,
                maxiter=None,
                maxls=25,
                ftol=None,
                mask_unreachable=False,
            )
            if np.sum(success) != 0:
                g.dfxm = {
                    "hkl": hkls[:, success],
                    "mu": goni_angles[0, success],
                    "omega": goni_angles[1, success],
                    "chi": goni_angles[2, success],
                    "phi": goni_angles[3, success],
                    "residual": residual[success],
                    "theta": theta[success],
                }
            else:
                g.dfxm = None

    def table_of_reflections(self):
        if not hasattr(self.polycrystal.grains[0], "dfxm"):
            raise ValueError("No reflections available. Run find_reflections() first.")
        tab = pd.DataFrame(
            columns=[
                "grain id",
                "h",
                "k",
                "l",
                "mu [dgr]",
                "omega [dgr]",
                "chi [dgr]",
                "phi [dgr]",
                "2 theta [dgr]",
                "eta [dgr]",
                "detector_z [mm]",
            ],
        )
        i = 0
        for gid, g in enumerate(self.polycrystal.grains):
            if g.dfxm:
                for n in range(len(g.dfxm["mu"])):
                    h, k, l = g.dfxm["hkl"][:, n].astype(int)
                    mu, omega, chi, phi = (
                        g.dfxm["mu"][n],
                        g.dfxm["omega"][n],
                        g.dfxm["chi"][n],
                        g.dfxm["phi"][n],
                    )
                    theta = g.dfxm["theta"][n]
                    detector_z = (
                        1000 * self.detector_distance * np.tan(2 * np.radians(theta))
                    )
                    tab.loc[i] = [
                        int(gid),
                        h,
                        k,
                        l,
                        mu,
                        omega,
                        chi,
                        phi,
                        2 * theta,
                        0,
                        detector_z,
                    ]
                    i += 1
        return tab


class Braggez(object):
    """
    Braggez is a class that performs optimization of the gonio angles to align the crystal
    with the target reflection. The optimization is performed using the L-BFGS-B algorithm

    This class is currently implemneted for the case of Dark Field X-ray Microscopy (DFXM) where
    the gonimeter/hexapod has 4 degrees of freedom (mu, omega, chi, phi). Stacked as:

        (1) base : mu
        (2) bottom : omega
        (3) top 1    : chi
        (4) top 2    : phi

    Here mu is a rotation about the negative y-axis, omega is a positive rotation about the
    z-axis, chi is a positive rotation about the x-axis, and phi is a positive rotation about
    the y-axis.

    The target reflection is always defined as lying in the xz-plane, with eta=0 (ez).
    This is also known as the simplified dfxm geometry.

    The mathmatical problem is defined as follows:
        Let nhat be a vector on the unit sphere representing the target reflection in its
        current position (i.e ub @ hkl). Next let target be a vector on the unit sphere
        such that target forms an angel of theta with the z-axis, where theta is the Bragg
        angle. The goal is to find a set of angles mu, omega, chi, phi of the goniometer/
        hexapod such that the reflection nhat is aligned with the target vector. Moreover,
        this must be done while considering the motor bounds of the goniometer/hexapod.

    NOTE: The solution is not unique. OptiBragg will find you one solution that satisfies these
    conditions if existent using a gradient based optimization.

    Args:
        energy (float): Energy of the X-ray beam in keV
        motor_bounds (dict): Dictionary containing the motor bounds of the goniometer/hexapod
            e.g. {"mu": (0, 20), "omega": (-22, 22), "chi": (-5, 5), "phi": (-5, 5)}
        epsilon (float): Small number used to compute the numerical derivatives during optimization
        verbose (bool): If True, print the results and iteration of the optimization

    Attributes:
        energy (float): Energy of the X-ray beam in keV
        motor_bounds (dict): Dictionary containing the motor bounds of the goniometer/hexapod
            e.g. {"mu": (0, 20), "omega": (-22, 22), "chi": (-5, 5), "phi": (-5, 5)}
        epsilon (float): Small number used to compute the numerical derivatives during optimization
        verbose (bool): If True, print the results and iteration of the optimization

    """

    def __init__(self, energy, motor_bounds, epsilon=1e-6, verbose=False):
        self.energy = energy
        self.wavelength = 12.398419874273968 / energy
        self.motor_bounds = motor_bounds

        self._xhat = np.array([[1], [0], [0]])
        self._yhat = np.array([[0], [1], [0]])
        self._zhat = np.array([[0], [0], [1]])

        self._set_rotational_increments(epsilon)
        self.epsilon = epsilon
        self.verbose = verbose

    def _set_rotational_increments(self, epsilon):
        """Set finite difference rotation for numerical derivatives"""
        self.dR_mu = self.R_mu(epsilon)
        self.dR_omega = self.R_omega(epsilon)
        self.dR_chi = self.R_chi(epsilon)
        self.dR_phi = self.R_phi(epsilon)

        self.dR_muT = self.R_mu(-epsilon)
        self.dR_omegaT = self.R_omega(-epsilon)
        self.dR_chiT = self.R_chi(-epsilon)
        self.dR_phiT = self.R_phi(-epsilon)

    def R_mu(self, mu):
        return Rotation.from_rotvec((mu * (-self._yhat)).T)

    def R_omega(self, omega):
        return Rotation.from_rotvec((omega * self._zhat).T)

    def R_chi(self, chi):
        return Rotation.from_rotvec((chi * self._xhat).T)

    def R_phi(self, phi):
        return Rotation.from_rotvec((phi * (self._yhat)).T)

    def R(self, x):
        mu, omega, chi, phi = x.reshape(4, len(x) // 4)
        return self.R_mu(mu) * self.R_omega(omega) * self.R_chi(chi) * self.R_phi(phi)

    def _arccos(self, ang, tol):
        """Simply to supress the annoying rounding warnings from numpy"""
        if np.abs(ang).max() >= 1 + tol:
            return np.arccos(ang)
        else:
            return np.arccos(np.clip(ang, -1, 1, out=ang))

    def cost_vector(self, rotation, nhat, target):
        """Array of misfits in radians for each reflection

        Args:
            rotation (:obj: scipy.spatial.transform.Rotation): The rotations, length N
            nhat (:obj: `numpy.ndarray`): The lattice plane normals, shape (3, N)
            target (:obj: `numpy.ndarray`): The target vectors, shape (3, N)

        Returns:
            :obj: `numpy.ndarray`: The cost vector in radians, shape (N,)
                each instance represents the misalignment of the lattice normal
                from the target vector.
        """
        return self._arccos(np.sum(rotation.apply(nhat.T).T * target, axis=0), tol=1e-6)

    def cost(self, rotation, nhat, target):
        """Aggregated cost function to fit many reflections simultaneously

        This is simply the sum of the cost_vector

        Args:
            rotation (:obj: scipy.spatial.transform.Rotation): The rotations, length N
            nhat (:obj: `numpy.ndarray`): The lattice plane normals, shape (3, N)
            target (:obj: `numpy.ndarray`): The target vectors, shape (3, N)

        Returns:
            :obj: `float`: The aggregated cost.
        """
        return self.cost_vector(rotation, nhat, target).sum()

    def cost_and_grad(self, x, nhat, target):
        """Computes the cost and the gradient of the cost function

        Uses a simple finite difference method to compute the gradient

        Args:
            x (:obj: `numpy.ndarray`): optimization variables flattened, shape (4*N,)
            nhat (:obj: `numpy.ndarray`): The lattice plane normals, shape (3, N)
            target (:obj: `numpy.ndarray`): The target vectors, shape (3, N)

        Returns:
            (:obj: `numpy.ndarray`): The aggregated cost.
            (:obj: `numpy.ndarray`): The gradient of the cost function, shape (4*N,)
        """
        mu, omega, chi, phi = x.reshape(4, len(x) // 4)
        R1 = self.R_mu(mu)
        R2 = self.R_omega(omega)
        R3 = self.R_chi(chi)
        R4 = self.R_phi(phi)

        # for speed optimization
        R12 = R1 * R2
        R34 = R3 * R4
        R234 = R2 * R34
        R123 = R12 * R3
        R1234 = R123 * R4
        #

        c0_vec = self.cost_vector(R1234, nhat, target)

        # second order central difference scheme
        N = nhat.shape[1]
        jac = np.zeros((4 * N,))

        jac[0:N] = self.cost_vector(
            self.dR_mu * R1234, nhat, target
        ) - self.cost_vector(self.dR_muT * R1234, nhat, target)

        jac[N : 2 * N] = self.cost_vector(
            R1 * self.dR_omega * R234, nhat, target
        ) - self.cost_vector(R1 * self.dR_omegaT * R234, nhat, target)

        jac[2 * N : 3 * N] = self.cost_vector(
            R12 * self.dR_chi * R34, nhat, target
        ) - self.cost_vector(R12 * self.dR_chiT * R34, nhat, target)

        jac[3 * N :] = self.cost_vector(
            R123 * self.dR_phi * R4, nhat, target
        ) - self.cost_vector(R123 * self.dR_phiT * R4, nhat, target)

        jac = jac / (2 * self.epsilon)

        return c0_vec.sum(), jac

    def _explode_bounds(self, N):
        """Set the bounds for all N reflections in the problem, just repeat the bounds N times..."""
        return np.repeat(
            np.radians(
                [
                    [self.motor_bounds["mu"][0], self.motor_bounds["mu"][1]],
                    [self.motor_bounds["omega"][0], self.motor_bounds["omega"][1]],
                    [self.motor_bounds["chi"][0], self.motor_bounds["chi"][1]],
                    [self.motor_bounds["phi"][0], self.motor_bounds["phi"][1]],
                ]
            ),
            N,
            axis=0,
        )

    def _minimize(self, x0, nhat, target, bounds, maxiter, maxls, ftol):
        """Scipy minimize wrapper of the L-BFGS-B algorithm"""

        return minimize(
            self.cost_and_grad,
            x0=x0,
            args=(nhat, target),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxls": maxls,
                "maxiter": maxiter,
                "ftol": ftol,
                "iprint": 2 if self.verbose else -1,
            },
        )

    def get_unit_vectors(self, ub, hkls):
        """Get the lattice plane normals, nhat, and the target vectors

        Args:
            ub (:obj: `numpy.ndarray`): The UB matrix, shape (3, 3)
            hkls (:obj: `numpy.ndarray`): The hkl vectors, shape (3, N)

        Returns:
            nhat (:obj: `numpy.ndarray`): The lattice plane normals, shape (3, N)
            target (:obj: `numpy.ndarray`): The target vectors, shape (3, N)
            theta (:obj: `numpy.ndarray`): The Bragg angle in radians, shape (N,)
        """
        Q = ub @ hkls
        nhat = Q / np.linalg.norm(Q, axis=0)
        d = 1 / np.linalg.norm(Q, axis=0)
        theta = np.arcsin(self.wavelength / (2 * d))
        target = self.R_mu(theta).apply(self._zhat.T).T
        return nhat, target, theta

    def _print_results(self, sol, res, success):
        print(
            "\n==================================RESULTS==================================\n"
        )
        for m, title in zip(sol, ["mu", "omega", "chi", "phi"]):
            print("   " + title + ": ", np.round(m, 4), " degrees")
        print("   misalignments:", res, " degrees")
        print("   success:", success)
        print(
            "\n===========================================================================\n"
        )

    def _mask_unreachable(self, nhat, target):
        c0 = np.degrees(np.arccos(np.sum(nhat * target, axis=0)))
        totrange = np.max(np.abs(self.motor_bounds["mu"]))
        totrange += np.max(np.abs(self.motor_bounds["omega"]))
        totrange += np.max(np.abs(self.motor_bounds["chi"]))
        totrange += np.max(np.abs(self.motor_bounds["phi"]))
        return c0 > totrange

    def _fill_unreachable(self, result, nhat, target, unreachable, alignment_tol):
        _solution = result.x.reshape(4, len(result.x) // 4)
        _residuals = self.cost_vector(self.R(result.x), nhat, target)
        _success = _residuals < alignment_tol

        solution = np.full((4, len(unreachable)), fill_value=np.nan)
        solution[:, ~unreachable] = _solution
        residuals = np.full(len(unreachable), fill_value=np.nan)
        residuals[~unreachable] = _residuals
        success = np.zeros(len(unreachable), dtype=bool)
        success[~unreachable] = _success

        return np.degrees(solution), np.degrees(residuals), success

    def align(
        self,
        ub,
        hkls,
        alignment_tol=1e-5,
        maxiter=None,
        maxls=25,
        ftol=None,
        mask_unreachable=False,
    ):
        """Align Q = ub @ hkls vectors with eta=0 plane to satisfy Bragg's law.

        Runs an optimization to find the goniometer angles that align the lattice plane
        normals to the target vectors. The target vectors are defined as the z-axis
        rotated by the Bragg angle in the xz-plane into the reverse direction of the
        beam which is assumed to propagate in the positive x-direction.

        Args:

            ub (:obj: `numpy.ndarray`): The UB matrix, shape (3, 3)
            hkls (:obj: `numpy.ndarray`): The hkl vectors, shape (3, N)
            alignment_tol (float): The tolerance for determining if the optimization
                managed to align the reflection for diffraction in unit of radians,
                representing the misalignment of the lattice normal from the target vector.
                Default is 1e-5.
            maxiter (int): The maximum number of iterations for the optimization perfromed
                by L-BFGS-B. Default is None, which will be set to 200 * N, where N is the
                number of reflections.
            maxls (int): The maximum number of line search iterations for the optimization per
                iteration of L-BFGS-B. Default is 25.
            ftol (float): The tolerance for the optimization cost function before termination.
                Default is None, which will be set to 1e-8 / N, where N is the number of
                reflections.
            mask_unreachable (bool): If True, mask reflections that are unreachable
                by the goniometer/hexapod. A reflection is unreachable if the misalignment
                of the lattice normal from the target vector is greater than the maximum sum
                of the absolute values of the motor bounds. Default is False.
                Maskin unreachable reflections speed up the optimization by simply not
                considering them. When masking is enabled, the solution and residuals
                success arrays will hold np.nan values for the unreachable reflections.

        Returns:
            sol (:obj: `numpy.ndarray`): The solution angles in degrees, shape (4, N)
                solution angles are in the order [mu, omega, chi, phi] such that
                solution[:, i] are the goniometer angles for the ith reflection.
            residuals (:obj: `numpy.ndarray`): The residuals in degrees, shape (N,)
                these are the misalignments of the lattice normal from the target
                vector given that the solution angles are applied.
            success (:obj: `numpy.ndarray`): The success of the optimization, shape (N,)
                True if the optimization converged within a tolerance of alignment_tol
            theta (:obj: `numpy.ndarray`): The Bragg angles in degrees, shape (N,)
        """
        nhat, target, theta = self.get_unit_vectors(ub, hkls)

        if mask_unreachable:
            unreachable = self._mask_unreachable(nhat, target)
        else:
            unreachable = np.zeros(hkls.shape[1], dtype=bool)

        if np.sum(~unreachable) == 0:
            return 0, 0, 0, 0

        nhat = nhat[:, ~unreachable]
        target = target[:, ~unreachable]
        bounds = self._explode_bounds(nhat.shape[1])

        x0 = (np.array(bounds)[:, 0] + np.array(bounds)[:, 1]) / 2.0

        if maxiter is None:
            maxiter = 200 * nhat.shape[1]
        if ftol is None:
            ftol = (1e-8) / nhat.shape[1]

        result = self._minimize(
            x0,
            nhat,
            target,
            bounds,
            maxiter,
            maxls,
            ftol,
        )

        solution, residuals, success = self._fill_unreachable(
            result,
            nhat,
            target,
            unreachable,
            alignment_tol,
        )

        if self.verbose:
            self._print_results(solution, residuals, success)

        return solution, residuals, success, np.degrees(theta)


if __name__ == "__main__":
    import os

    import crispy

    pc = crispy.Polycrystal(
        os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
        group_name="Fe",
        lattice_parameters=[4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0],
        symmetry=225,
    )

    motor_bounds = {
        "mu": (0, 20),
        "omega": (-22, 22),
        "chi": (-5, 5),
        "phi": (-5, 5),
        "detector_z": (-0.04, 1.96),
        "detector_y": (-0.169, 1.16),
    }

    detector_distance = 4.0
    energy = 19.1

    goni = crispy.dfxm.Goniometer(
        pc,
        energy,
        detector_distance,
        motor_bounds,
    )

    import cProfile
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    goni.find_reflections()
    goni.find_reflections()
    goni.find_reflections()
    goni.find_reflections()

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")
