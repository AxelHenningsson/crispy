import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from xfab import tools

# Beamline motor name for each internal angle (mu, omega, chi, phi).
_MOTOR_NAMES = {
    "ID03": {"mu": "mu", "omega": "omega", "chi": "chi", "phi": "phi"},
    "APS": {"mu": "RX", "omega": "omega", "chi": "samRZ", "phi": "samRX"},
}


def _internal_bounds(motor_bounds, beamline):
    """Return a copy of motor_bounds keyed by the internal angle names.

    Beamline motor names (e.g. ``RX``, ``samRZ``, ``samRX`` at APS) are
    accepted as aliases for ``mu``, ``chi`` and ``phi``.
    """
    bounds = dict(motor_bounds)
    for internal, motor in _MOTOR_NAMES[beamline].items():
        if motor != internal and motor in bounds:
            if internal in bounds:
                raise ValueError(
                    f'motor_bounds contains both "{motor}" and "{internal}", '
                    "which are the same motor"
                )
            bounds[internal] = bounds.pop(motor)
    return bounds


class Goniometer:
    """A class representing a Dark-Field X-ray Microscopy (DFXM) experimental setup.

    This class calculates the required microscope angular settings to bring a set of
    Miller indices into diffraction condition at specific turntable rotations.

    Attributes:
        polycrystal (:class:`crispy.Polycrystal <crispy._polycrystal.Polycrystal>`):
            The polycrystal object mounted on the goniometer.
        energy (:obj:`float`): Photon energy in keV.
        detector_distance (:obj:`float`): Distance between sample and detector
            in metres.
        motor_bounds (:obj:`dict`): Motor limits for the goniometer/hexapod in degrees
            (sample stage) and metres (detector translations). Keys are shown in the
            example below:
        eta (:obj:`float`): Azimuth scattering angle in degrees. Default is 0 degrees.
        beamline (:obj:`str`): ``"ID03"`` (default) or ``"APS"`` for 20-ID-E.
            Grain maps retain the ImageD11/MIDAS frame in both cases. APS uses
            positive rotations about the APS laboratory axes (z along the
            beam, y up) and the motor stack hexapod ``RX`` -> ``omega`` (Ry)
            -> ``samRZ`` -> ``samRX``, with ``samRX`` on top. For APS,
            ``motor_bounds`` takes these motor names (the ID03 names ``mu``,
            ``chi`` and ``phi`` are accepted as aliases for ``RX``, ``samRZ``
            and ``samRX``) and the results are reported with them. APS detector
            bounds use ``detector_y`` (vertical) and ``detector_x`` (horizontal).
        motor_names (:obj:`dict`): Beamline motor name of each internal angle
            ``mu``, ``omega``, ``chi``, ``phi``.


    .. code-block:: python

        motor_bounds = {
        "mu": (0, 20), # degrees
        "omega": (-22, 22), # degrees
        "chi": (-5, 5), # degrees
        "phi": (-5, 5), # degrees
        "detector_z": (-0.04, 1.96), # metres
        "detector_y": (-0.169, 1.16) # metres
        }

    """

    def __init__(
        self,
        polycrystal,
        energy,
        detector_distance,
        motor_bounds,
        eta=0,
        beamline="ID03",
    ):
        if beamline not in ("ID03", "APS"):
            raise ValueError('beamline must be "ID03" or "APS"')
        self.polycrystal = polycrystal
        self.energy = energy
        self.wavelength = 12.398419874273968 / self.energy
        self.detector_distance = detector_distance
        self.motor_bounds = motor_bounds
        self.eta = eta
        self.beamline = beamline
        self.motor_names = dict(_MOTOR_NAMES[beamline])
        bounds = _internal_bounds(motor_bounds, beamline)
        missing = [m for k, m in self.motor_names.items() if k not in bounds]
        if missing:
            raise ValueError(f"motor_bounds is missing {missing}")
        self._detector_vertical = "detector_y" if beamline == "APS" else "detector_z"

    def _get_hkls(self):
        """Find all h,k,l miller indices that can diffract at the given energy.

        This function will consider the symmetry of the polycrystal material phase.

        Returns:
            :obj:`numpy.ndarray`: Array of ``h,k,l`` Miller indices that can
                diffract at the given energy. ``shape (3, N)`` where ``N`` is
                the number of reflections.
        """
        th_min = 0
        th_max = (
            np.arctan(
                self.motor_bounds[self._detector_vertical][1] / self.detector_distance
            ) / 2.0
        )
        unit_cell = self.polycrystal.reference_cell.lattice_parameters
        sgno = self.polycrystal.reference_cell.symmetry
        sintlmin = np.sin(th_min) / self.wavelength
        sintlmax = np.sin(th_max) / self.wavelength
        hkls = tools.genhkl_all(unit_cell, sintlmin, sintlmax, sgno=sgno).T
        return hkls

    def find_reflections(
        self,
        alignment_tol=1e-4,
        maxiter=None,
        maxls=25,
        ftol=None,
        mask_unreachable=False,
    ):
        """Find all reflections reachable by the goniometer in DFXM in  *"simplified geometry"*.

        Example:

        .. code-block:: python

            import crispy

            polycrystal = crispy.assets.grain_map.tdxrd_map()

            motor_bounds = {
            "mu": (0, 20), # degrees
            "omega": (-22, 22), # degrees
            "chi": (-5, 5), # degrees
            "phi": (-5, 5), # degrees
            "detector_z": (-0.04, 1.96), # metres
            "detector_y": (-0.169, 1.16) # metres
            }

            goniometer = crispy.dfxm.Goniometer(polycrystal,
                                    energy=17,
                                    detector_distance=4,
                                    motor_bounds=motor_bounds)
            goniometer.find_reflections()

            # The found reflections are stored in the `dfxm` dictionary of each grain.
            reflections_grain_12 = polycrystal.grains[12].dfxm


        Output dictorary for ``reflections_grain_12``:

            - "chi":  array([-4.51990607])
            - "hkl":  array([[0.], [0.], [2.]])
            - "mu":  array([6.12038619])
            - "omega":  array([-13.18363797])
            - "phi":  array([-4.99889719])
            - "residual":  array([0.])
            - "theta":  array([14.71219735])

        The reflections are found by running an optimisation algorithm that aligns the
        lattice normals with target vectors. The target vectors are defined as the z-axis
        rotated by the Bragg angle in the xz-plane into the reverse direction of the beam,
        which is assumed to propagate in the positive x-direction.

        The reflections are stored in the :attr:`dfxm` dictionary of each grain with keys:

            - "hkl": Miller indices of diffracting planes
            - "mu": Mu angles at which diffraction occurs
            - "omega": Omega angles at which diffraction occurs
            - "chi": Chi angles at which diffraction occurs
            - "phi": Phi angles at which diffraction occurs
            - "residual": Misalignment between target Q-vector and aligned lattice normal
            - "theta": Bragg angles at which diffraction occurs

        To access the reflections of grain ``i``, use:

            polycrystal.grains[i].dfxm["hkl"]  # etc.

        Args:
            alignment_tol (:obj:`float`): Tolerance for determining if optimisation
                aligned the reflection, in degrees. Represents misalignment between
                lattice normal and target vector. Default is 1e-4.
            maxiter (:obj:`int`): Maximum iterations for L-BFGS-B optimisation.
                Default is ``None``, set to 200 * N where N is number of reflections.
            maxls (:obj:`int`): Maximum line search iterations per L-BFGS-B iteration.
                Default is 25.
            ftol (:obj:`float`): Tolerance for optimisation cost function termination.
                Default is ``None``, set to 1e-8 / N where N is number of reflections.
            mask_unreachable (:obj:`bool`): If ``True``, mask reflections unreachable by
                goniometer/hexapod. A reflection is unreachable if misalignment exceeds
                the sum of absolute motor bounds. Default is ``False``. Masking speeds up
                optimisation by excluding unreachable reflections. When enabled,
                unreachable reflections have ``np.nan`` values in solution and residuals.
        """
        hkls = self._get_hkls()
        bez = _Braggez(self.energy, self.motor_bounds, beamline=self.beamline)

        for g in self.polycrystal.grains:
            goni_angles, residual, success, theta = bez.align(
                g.ub,
                hkls,
                alignment_tol,
                maxiter,
                maxls,
                ftol,
                mask_unreachable,
                eta=self.eta,
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
                    "eta": self.eta,
                }
                self._add_motor_names(g.dfxm)
            else:
                g.dfxm = None

    def table_of_reflections(self):
        """Compile a :obj:`pandas.DataFrame` of the reflections found by :func:`find_reflections()`.

        Example:

        .. code-block:: python

            import crispy

            polycrystal = crispy.assets.grain_map.tdxrd_map()

            motor_bounds = {
            "mu": (0, 20), # degrees
            "omega": (-22, 22), # degrees
            "chi": (-5, 5), # degrees
            "phi": (-5, 5), # degrees
            "detector_z": (-0.04, 1.96), # metres
            "detector_y": (-0.169, 1.16) # metres
            }

            goniometer = crispy.dfxm.Goniometer(polycrystal,
                                    energy=17,
                                    detector_distance=4,
                                    motor_bounds=motor_bounds)
            goniometer.find_reflections()
            df = goniometer.table_of_reflections()


        Output :obj:`pandas.DataFrame` for ``df`` looks like this:

        .. image:: ../../docs/source/images/dfxm_table.png

        Raises:
            ValueError: if :func:`find_reflections()` has not been run.

        Returns:
            :obj:`pandas.DataFrame`: DataFrame of the reflections found
                by :func:`find_reflections()`. For each grain, the DataFrame contains
                column in accordance with the keys in the :attr:`dfxm` dictionary
                of the grain. At APS the angle columns are named after the
                motors: ``RX``, ``omega``, ``samRZ`` and ``samRX``. APS detector
                columns are ``detector_y [mm]`` and
                ``detector_x [mm]``, measured from the direct beam.
        """
        if not hasattr(self.polycrystal.grains[0], "dfxm"):
            raise ValueError("No reflections available. Run find_reflections() first.")
        tab = pd.DataFrame(
            columns=[
                "grain id",
                "h",
                "k",
                "l",
                f"{self.motor_names['mu']} [dgr]",
                f"{self.motor_names['omega']} [dgr]",
                f"{self.motor_names['chi']} [dgr]",
                f"{self.motor_names['phi']} [dgr]",
                "2 theta [dgr]",
                "eta [dgr]",
                f"{self._detector_vertical} [mm]",
            ],
        )
        if self.beamline == "APS":
            tab["detector_x [mm]"] = pd.Series(dtype=float)
        i = 0
        for gid, g in enumerate(self.polycrystal.grains):
            if g.dfxm:
                for n in range(len(g.dfxm["mu"])):
                    h, k, l = g.dfxm["hkl"][:, n].astype(int)
                    mu, omega, chi, phi, eta = (
                        g.dfxm["mu"][n],
                        g.dfxm["omega"][n],
                        g.dfxm["chi"][n],
                        g.dfxm["phi"][n],
                        g.dfxm["eta"],
                    )
                    theta = g.dfxm["theta"][n]
                    detector_radius = (
                        1000 * self.detector_distance * np.tan(2 * np.radians(theta))
                    )
                    row = [
                        int(gid),
                        h,
                        k,
                        l,
                        mu,
                        omega,
                        chi,
                        phi,
                        2 * theta,
                        eta,
                        detector_radius,
                    ]
                    if self.beamline == "APS":
                        row[-1] *= np.cos(np.radians(eta))
                        row.append(-detector_radius * np.sin(np.radians(eta)))
                    tab.loc[i] = row
                    i += 1
        return tab

    def find_symmetry_axis(
        self,
        alignment_tol=1e-4,
        maxiter=None,
        maxls=25,
        ftol=None,
        mask_unreachable=False,
    ):
        """Find Miller indices that can be aligned vertically within motor bounds.

        Vertical is +z at ID03 and +y at APS, both +z in the grain-map frame.

        This allows for the identification of various oblique imaging geometries in
        which multiple reflections are reachable by pure z-axis rotations.

        Example:

        .. code-block:: python

            import crispy

            polycrystal = crispy.assets.grain_map.lab_dct_volume("Al1050")

            motor_bounds = {
            "mu": (0, 20), # degrees
            "omega": (-30, 30), # degrees
            "chi": (-7, 7), # degrees
            "phi": (-7, 7), # degrees
            "detector_z": (-0.04, 1.96), # metres
            "detector_y": (-0.169, 1.16) # metres
            }

            goniometer = crispy.dfxm.Goniometer(polycrystal,
                                    energy=17,
                                    detector_distance=4,
                                    motor_bounds=motor_bounds)
            goniometer.find_symmetry_axis()


        Finds all reachable reflections in the polycrystal using an optimisation algorithm
        that aligns lattice normals with target vectors (defined as the z-axis). The
        reflections are stored in the polycrystal grain objects as dictionaries named
        ``symmetry_axis`` with the following keys:

            - "hkl": Miller indices that can diffract
            - "mu": Mu angles at which diffraction occurs
            - "omega": Omega angles at which diffraction occurs
            - "chi": Chi angles at which diffraction occurs
            - "phi": Phi angles at which diffraction occurs
            - "residual": Radians between lattice normal and z-axis

        To access the symmetry axis of grain ``i``, use:

            polycrystal.grains[i].symmetry_axis["hkl"]  # etc.

        Args:
            alignment_tol (:obj:`float`): Tolerance in degrees for determining if the
                optimisation aligned the lattice normal with the z-axis. Represents the
                misalignment between lattice normal and target vector. Default is 1e-4.
            maxiter (:obj:`int`): Maximum iterations for L-BFGS-B optimisation.
                Default is None (set to 200 * N, where N is number of reflections).
            maxls (:obj:`int`): Maximum line search iterations per L-BFGS-B iteration.
                Default is 25.
            ftol (:obj:`float`): Tolerance for optimisation cost function termination.
                Default is None (set to 1e-8 / N, where N is number of reflections).
            mask_unreachable (:obj:`bool`): If True, mask unreachable lattice normals by the
                goniometer/hexapod. A lattice normal is unreachable if its misalignment exceeds
                the maximum sum of absolute motor bounds. Default is False. Masking speeds
                up optimisation by excluding unreachable lattice normals. When enabled,
                solution and residual arrays contain np.nan for unreachable lattice normals.
        """
        hkls = self._get_hkls()
        bsym = _BraggSym(self.energy, self.motor_bounds, beamline=self.beamline)

        for g in self.polycrystal.grains:
            goni_angles, residual, success = bsym.align(
                g.ub,
                hkls,
                alignment_tol,
                maxiter,
                maxls,
                ftol,
                mask_unreachable,
            )
            if np.sum(success) != 0:
                g.symmetry_axis = {
                    "hkl": hkls[:, success],
                    "mu": goni_angles[0, success],
                    "omega": goni_angles[1, success],
                    "chi": goni_angles[2, success],
                    "phi": goni_angles[3, success],
                    "residual": residual[success],
                }
                self._add_motor_names(g.symmetry_axis)
            else:
                g.symmetry_axis = None

    def _add_motor_names(self, result):
        """Make the angles available under the beamline motor names as well."""
        for internal, motor in self.motor_names.items():
            result[motor] = result[internal]


class _Braggez(object):
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

    For APS, mu is positive about y in this internal frame and the stacking is
    unchanged (chi below phi). In APS laboratory coordinates this is
    Rx(RX) Ry(omega) Rz(samRZ) Rx(samRX), where mu = hexapod RX, chi = samRZ and
    phi = samRX. ``motor_bounds`` may use either set of names.
    Input UB matrices are always in the ImageD11/MIDAS frame.

    The target reflection is by default defined as lying in the xz-plane, with eta=0 (ez).
    This is also known as the simplified dfxm geometry.

    The mathmatical problem is defined as follows:
        Let nhat be a vector on the unit sphere representing the target reflection in its
        current position (i.e ub @ hkl). Next let target be a vector on the unit sphere
        such that target forms an angel of theta with the z-axis, where theta is the Bragg
        angle. The goal is to find a set of angles mu, omega, chi, phi of the goniometer/
        hexapod such that the reflection nhat is aligned with the target vector. Moreover,
        this must be done while considering the motor bounds of the goniometer/hexapod.

    NOTE: The solution is not unique. Braggez will find you one solution that satisfies these
    conditions if existent using a gradient based optimization.

    Args:
        energy (float): Energy of the X-ray beam in keV
        motor_bounds (dict): Dictionary containing the motor bounds of the goniometer/hexapod
            e.g. {"mu": (0, 20), "omega": (-22, 22), "chi": (-5, 5), "phi": (-5, 5)}
        epsilon (float): Small number used to compute the numerical derivatives during optimization
        verbose (bool): If True, print the results and iteration of the optimization
        beamline (str): ``"ID03"`` (default) or ``"APS"`` for 20-ID-E.

    Attributes:
        energy (float): Energy of the X-ray beam in keV
        motor_bounds (dict): Dictionary containing the motor bounds of the goniometer/hexapod
            e.g. {"mu": (0, 20), "omega": (-22, 22), "chi": (-5, 5), "phi": (-5, 5)}
        epsilon (float): Small number used to compute the numerical derivatives during optimization
        verbose (bool): If True, print the results and iteration of the optimization

    """

    def __init__(self, energy, motor_bounds, epsilon=1e-6, verbose=False, beamline="ID03"):
        if beamline not in ("ID03", "APS"):
            raise ValueError('beamline must be "ID03" or "APS"')
        self.energy = energy
        self.wavelength = 12.398419874273968 / energy
        self.motor_bounds = _internal_bounds(motor_bounds, beamline)
        self.beamline = beamline

        self._xhat = np.array([[1], [0], [0]])
        self._yhat = np.array([[0], [1], [0]])
        self._zhat = np.array([[0], [0], [1]])

        # Axes are in the grain-map frame; parameter order is mu, omega, chi, phi.
        if beamline == "ID03":
            self._rotation_axes = (-self._yhat, self._zhat, self._xhat, self._yhat)
            self._rotation_order = (0, 1, 2, 3)
        else:
            self._rotation_axes = (self._yhat, self._zhat, self._xhat, self._yhat)
            self._rotation_order = (0, 1, 2, 3)

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
        return Rotation.from_rotvec((mu * self._rotation_axes[0]).T)

    def R_omega(self, omega):
        return Rotation.from_rotvec((omega * self._rotation_axes[1]).T)

    def R_chi(self, chi):
        return Rotation.from_rotvec((chi * self._rotation_axes[2]).T)

    def R_phi(self, phi):
        return Rotation.from_rotvec((phi * self._rotation_axes[3]).T)

    def _rotations(self, x):
        mu, omega, chi, phi = x.reshape(4, len(x) // 4)
        rotations = (self.R_mu(mu), self.R_omega(omega), self.R_chi(chi), self.R_phi(phi))
        return [rotations[i] for i in self._rotation_order]

    def R(self, x):
        R1, R2, R3, R4 = self._rotations(x)
        return R1 * R2 * R3 * R4

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
            nhat (:obj:`numpy.ndarray`): The lattice plane normals, shape (3, N)
            target (:obj:`numpy.ndarray`): The target vectors, shape (3, N)

        Returns:
            :obj:`numpy.ndarray`: The cost vector in radians, shape (N,)
                each instance represents the misalignment of the lattice normal
                from the target vector.
        """
        return self._arccos(np.sum(rotation.apply(nhat.T).T * target, axis=0), tol=1e-6)

    def cost(self, rotation, nhat, target):
        """Aggregated cost function to fit many reflections simultaneously

        This is simply the sum of the cost_vector

        Args:
            rotation (:obj: scipy.spatial.transform.Rotation): The rotations, length N
            nhat (:obj:`numpy.ndarray`): The lattice plane normals, shape (3, N)
            target (:obj:`numpy.ndarray`): The target vectors, shape (3, N)

        Returns:
            :obj:`float`: The aggregated cost.
        """
        return self.cost_vector(rotation, nhat, target).sum()

    def cost_and_grad(self, x, nhat, target):
        """Computes the cost and the gradient of the cost function

        Uses a simple finite difference method to compute the gradient

        Args:
            x (:obj:`numpy.ndarray`): optimization variables flattened, shape (4*N,)
            nhat (:obj:`numpy.ndarray`): The lattice plane normals, shape (3, N)
            target (:obj:`numpy.ndarray`): The target vectors, shape (3, N)

        Returns:
            (:obj:`numpy.ndarray`): The aggregated cost.
            (:obj:`numpy.ndarray`): The gradient of the cost function, shape (4*N,)
        """
        R1, R2, R3, R4 = self._rotations(x)
        increments = (self.dR_mu, self.dR_omega, self.dR_chi, self.dR_phi)
        inverse = (self.dR_muT, self.dR_omegaT, self.dR_chiT, self.dR_phiT)
        dR1, dR2, dR3, dR4 = [increments[i] for i in self._rotation_order]
        dR1T, dR2T, dR3T, dR4T = [inverse[i] for i in self._rotation_order]

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
        i1, i2, i3, i4 = self._rotation_order

        jac[i1 * N : (i1 + 1) * N] = self.cost_vector(
            dR1 * R1234, nhat, target
        ) - self.cost_vector(dR1T * R1234, nhat, target)

        jac[i2 * N : (i2 + 1) * N] = self.cost_vector(
            R1 * dR2 * R234, nhat, target
        ) - self.cost_vector(R1 * dR2T * R234, nhat, target)

        jac[i3 * N : (i3 + 1) * N] = self.cost_vector(
            R12 * dR3 * R34, nhat, target
        ) - self.cost_vector(R12 * dR3T * R34, nhat, target)

        jac[i4 * N : (i4 + 1) * N] = self.cost_vector(
            R123 * dR4 * R4, nhat, target
        ) - self.cost_vector(R123 * dR4T * R4, nhat, target)

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

    def get_unit_vectors(self, ub, hkls, eta):
        """Get the lattice plane normals, nhat, and the target vectors

        Args:
            ub (:obj:`numpy.ndarray`): The UB matrix, shape (3, 3)
            hkls (:obj:`numpy.ndarray`): The hkl vectors, shape (3, N)
            eta (:obj:`float`): Azimuth scattering angle in degrees. Default is 0 degrees.

        Returns:
            nhat (:obj:`numpy.ndarray`): The lattice plane normals, shape (3, N)
            target (:obj:`numpy.ndarray`): The target vectors, shape (3, N)
            theta (:obj:`numpy.ndarray`): The Bragg angle in radians, shape (N,)
        """
        Q = ub @ hkls
        nhat = Q / np.linalg.norm(Q, axis=0)
        d = 1 / np.linalg.norm(Q, axis=0)
        theta = np.arcsin(self.wavelength / (2 * d))
        target = Rotation.from_rotvec((-theta * self._yhat).T).apply(self._zhat.T).T

        # implements arbitrary eta scattering angle
        R_eta = Rotation.from_rotvec(np.radians(eta) * np.array([1, 0, 0]))
        target = R_eta.apply(target.T).T

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
        _success = _residuals < np.radians(alignment_tol)

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
        alignment_tol=1e-4,
        maxiter=None,
        maxls=25,
        ftol=None,
        mask_unreachable=False,
        eta=0,
    ):
        """Align Q = ub @ hkls vectors a given eta plane to satisfy Bragg's law.

        Runs an optimization to find the goniometer angles that align the lattice plane
        normals to the target vectors. The target vectors are defined as the z-axis
        rotated by the Bragg angle in the xz-plane into the reverse direction of the
        beam which is assumed to propagate in the positive x-direction. If eta is not 0
        the target

        Args:
            ub (:obj:`numpy.ndarray`): The UB matrix, shape (3, 3)
            hkls (:obj:`numpy.ndarray`): The hkl vectors, shape (3, N)
            alignment_tol (:obj:`float`): The tolerance for determining if the optimization
                managed to align the reflection for diffraction in unit of degrees,
                representing the misalignment of the lattice normal from the target vector.
                Default is 1e-4.
            maxiter (:obj:`int`): The maximum number of iterations for the optimization performed
                by L-BFGS-B. Default is None, which will be set to 200 * N, where N is the
                number of reflections.
            maxls (:obj:`int`): The maximum number of line search iterations for the optimization per
                iteration of L-BFGS-B. Default is 25.
            ftol (:obj:`float`): The tolerance for the optimization cost function before termination.
                Default is None, which will be set to 1e-8 / N, where N is the number of
                reflections.
            mask_unreachable (:obj:`bool`): If True, mask reflections that are unreachable
                by the goniometer/hexapod. A reflection is unreachable if the misalignment
                of the lattice normal from the target vector is greater than the maximum sum
                of the absolute values of the motor bounds. Default is False. Masking
                unreachable reflections speed up the optimization by simply not
                considering them. When masking is enabled, the solution and residuals
                success arrays will hold np.nan values for the unreachable reflections.
            eta (:obj:`float`): Azimuth scattering angle in degrees. Default is 0 degrees.

        Returns:
            (:obj:`tuple`): tuple containing:

                - **sol** (:obj:`numpy.ndarray`): The solution angles in degrees,
                        shape (4, N) solution angles are in the order
                        [mu, omega, chi, phi] such that solution[:, i] are the
                        goniometer angles for the ith reflection.

                - **residuals** (:obj:`numpy.ndarray`): The residuals in degrees,
                        shape (N,) these are the misalignments of the lattice
                        normal from the target vector given that the solution
                        angles are applied.

                - **success** (:obj:`numpy.ndarray`): The success of the optimization,
                        shape (N,) True if the optimization converged within a
                        tolerance of alignment_tol

                - **theta** (:obj:`numpy.ndarray`): The Bragg angles,
                        in degrees for each reflection, shape (N,)
        """
        nhat, target, theta = self.get_unit_vectors(ub, hkls, eta)

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


class _BraggSym(_Braggez):
    """BraggSym is a class that performs optimization of the gonio angles to align
    crystal lattice normals with the lab-z axis.

    This is usefull for oblique diffraction geometry in which several symmetry equivalent
    reflections are to be imaged by rotating the sample around a symmetry axis.

    This class is currently implemneted for the case of Dark Field X-ray Microscopy (DFXM) where
    the gonimeter/hexapod has 4 degrees of freedom (mu, omega, chi, phi). Stacked as:

        (1) base : mu
        (2) bottom : omega
        (3) top 1    : chi
        (4) top 2    : phi

    Here mu is a rotation about the negative y-axis, omega is a positive rotation about the
    z-axis, chi is a positive rotation about the x-axis, and phi is a positive rotation about
    the y-axis.

    The target reflection is always defined as the z-axis.

    For APS, the axes and stacking order follow _Braggez. The target is still
    internal +z, which is APS laboratory +y.

    The mathmatical problem is defined as follows:
        Let nhat be a vector on the unit sphere representing the target reflection in its
        current position (i.e ub @ hkl). Next let target be the z-axis. The goal is to
        find a set of angles mu, omega, chi, phi of the goniometer/hexapod such that the
        reflection nhat is aligned with the target vector. Moreover, this must be done
        while considering the motor bounds of the goniometer/hexapod.

    NOTE: The solution is not unique. BraggSym will find you one solution that
        satisfies these conditions if existent using a gradient based optimization.

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

    def get_unit_vectors(self, ub, hkls):
        """
        Get the lattice plane normals and the target vectors,
        which are statically defined as the z-axis in BraggSym.

        Args:
            ub (:obj:`numpy.ndarray`): The UB matrix, shape (3, 3)
            hkls (:obj:`numpy.ndarray`): The hkl vectors, shape (3, N)

        Returns:
            (:obj:`tuple`): tuple containing:

                - **nhat** (:obj:`numpy.ndarray`):  The lattice plane
                        normals, shape (3, N)
                - **target** (:obj:`numpy.ndarray`): The target vectors, i.e.,
                        the z-axis in the lab frame, shape (3, N)
        """
        Q = ub @ hkls
        nhat = Q / np.linalg.norm(Q, axis=0)
        target = np.tile(self._zhat, nhat.shape[1])
        return nhat, target

    def align(
        self,
        ub,
        hkls,
        alignment_tol=1e-4,
        maxiter=None,
        maxls=25,
        ftol=None,
        mask_unreachable=False,
    ):
        """Align Q = ub @ hkls vectors with the lab frame z-axis.

        Runs an optimization to find the goniometer angles that align the lattice plane
        normals to the target. The target vectors are defined statically as the z-axis
        for the BraggSym class.

        Args:
            ub (:obj:`numpy.ndarray`): The UB matrix, shape (3, 3)
            hkls (:obj:`numpy.ndarray`): The hkl vectors, shape (3, N)
            alignment_tol (:obj:`float`): The tolerance for determining if the optimization
                managed to align the reflection for diffraction in unit of degrees,
                representing the misalignment of the lattice normal from the target vector.
                Default is 1e-4.
            maxiter (:obj:`int`): The maximum number of iterations for the optimization performed
                by L-BFGS-B. Default is None, which will be set to 200 * N, where N is the
                number of reflections.
            maxls (:obj:`int`): The maximum number of line search iterations for the optimization per
                iteration of L-BFGS-B. Default is 25.
            ftol (:obj:`float`): The tolerance for the optimization cost function before termination.
                Default is None, which will be set to 1e-8 / N, where N is the number of
                reflections.
            mask_unreachable (:obj:`bool`): If True, mask reflections that are unreachable
                by the goniometer/hexapod. A reflection is unreachable if the misalignment
                of the lattice normal from the target vector is greater than the maximum sum
                of the absolute values of the motor bounds. Default is False. Masking
                unreachable reflections speed up the optimization by simply not
                considering them. When masking is enabled, the solution and residuals
                success arrays will hold np.nan values for the unreachable reflections.

        Returns:
            (:obj:`tuple`): tuple containing:

                - **sol** (:obj:`numpy.ndarray`): The solution angles in degrees,
                        shape (4, N) solution angles are in the order
                        [mu, omega, chi, phi] such that solution[:, i] are the
                        goniometer angles for the ith reflection.

                - **residuals** (:obj:`numpy.ndarray`): The residuals in degrees,
                        shape (N,) these are the misalignments of the lattice
                        normal from the target vector given that the solution
                        angles are applied.

                - **success** (:obj:`numpy.ndarray`): The success of the optimization,
                        shape (N,) True if the optimization converged within a
                        tolerance of alignment_tol
        """
        nhat, target = self.get_unit_vectors(ub, hkls)

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

        return solution, residuals, success


if __name__ == "__main__":
    pass
