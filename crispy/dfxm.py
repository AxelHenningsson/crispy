import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from xfab import tools

from crispy import laue


class Goniometer:
    """
    A class to represent a Diffraction X-ray Microscopy (DFXM) experiment setup.

    This class is designed to find the angular settings required of the microscope in order
    to make a given set of Miller indices come into diffraction at some rotatin of the turntable.

    This is usefull for situations when we want to reach several reflection in DFXM
    without re-mounitng the sample

    Attributes:
        polycrystal (:obj:`crispy.Polycrystal`): The polycrystal object mounted on the goniometer.
        U (:obj:`numpy array`): Crystal orientation matrix of ``shape=(3,3)`` (unitary). the selected grain.
        unit_cell (:obj:`numpy array`): Unit cell parameters [a, b, c, alpha, beta, gamma] of ``shape=(6,)``.
        hkl (:obj:`numpy array`): Miller indices ``shape=(3,n)``.
        energy (:obj:`float`): Photon energy in keV.
        B (:obj:`numpy array`): The B matrix. ``shape=(3,3)`
        wavelength (:obj:`float`): wavelength in units of angstrom.

    """

    def __init__(self, polycrystal, energy, detector_distance):
        """
        Initialize the DFXM class with the experiment setup parameters.

        Args:
            polycrystal (:obj:`crispy.Polycrystal`): The polycrystal object.
            energy (:obj:`float`): Photon energy in keV.
            detector_distance (:obj:`float`): Distance in meters to the detector rig measured from the sample.
                if given, this number will be used to compute required detetor translations in y-lab. and z-lab.
                Defaults to None.
            motor_ranges (:obj:`dict`): Dictionary with the motor ranges for the experiment.
        """
        self.polycrystal = polycrystal
        self.unit_cell = polycrystal.reference_cell.lattice_parameters
        self.energy = energy
        self.wavelength = laue.keV_to_angstrom(self.energy)
        self.detector_distance = detector_distance
        self._ub = None  # orientation matrix of a selected grain

    def select_grain(self, grain_id):
        """
        Select a grain from the polycrystal.

        Args:
            grain_id (:obj:`int`): The grain id to select.

        Returns:
            grain (:obj:`crispy.Grain`): The grain object.
        """
        self._ub = self.polycrystal.grains[grain_id].ub[:, :]

    @property
    def ub(self):
        if self._ub is None:
            raise ValueError(
                "No grain has been selected. Use the select_grain method to select a grain."
            )
        return self._ub

    def ezbragg(self, hkl):
        """
        Align the crystal orientation matrix (U) such that the provided Miller indices (hkl) is in
        diffraction conditions forming a Bragg angle to the incident beam (assumed to propagate along x-lab).

        This procedure will bring the diffraction vector into the x-z plane such that diffraction is at
        the simplifed geometry of the Bragg condition, i.e when eta is zero.

        The procedure is as follows:
            (1) The hkl bourhgt to the x-z plane by a pure rotation in omega, i.e around the z-axis.
            (2) The hkl is brought to Bragg by a pure rotation mu, i.e around the y-axis.

        NOTE: the returned angles are omega, defined as positive around the z-axis, and mu, defined as
            positive around the negative y-axis.

        Args:
            hkl (:obj:`numpy array`): Miller indices to align with, ``shape=(3,)``.

        Returns:
            omega (:obj:`float`): The omega angle in degrees.
            mu (:obj:`float`): The mu angle in degrees

        """

        # Lattice normal in the sample
        nhat = self.ub @ hkl
        nhat /= np.linalg.norm(nhat)

        xy_projection = nhat.copy()
        xy_projection[2] = 0
        xy_projection /= np.linalg.norm(xy_projection)

        xhat, yhat, zhat = np.eye(3, 3)
        if xy_projection[0] < 0:
            omega = np.arccos(np.dot(-xhat, xy_projection))
            if xy_projection[1] < 0:
                omega = -omega
        else:
            omega = np.arccos(np.dot(xhat, xy_projection))
            if xy_projection[1] > 0:
                omega = -omega

        Rom = Rotation.from_rotvec(omega * zhat).as_matrix()
        nhat_xz = Rom @ nhat

        wedge = np.arccos(np.dot(zhat, nhat_xz))
        Q = self.ub @ hkl
        d = 1 / np.linalg.norm(Q)
        theta = np.arcsin(self.wavelength / (2 * d))

        if nhat_xz[0] > 0:  # defined as positive around negative y
            mu = wedge + theta
        else:
            mu = -(wedge - theta)

        ## testing
        Rmu = Rotation.from_rotvec(mu * (-yhat)).as_matrix()
        trial = Rmu @ Rom @ nhat
        assert np.allclose(trial[1], 0)
        assert trial[0] < 0
        assert trial[2] > 0
        assert np.cos(theta) - np.dot(trial, zhat) < 1e-10
        ##

        return np.degrees(omega), np.degrees(mu), np.degrees(theta)

    def get_reflections(self, omega_range, mu_range, theta_range):
        # generate all hkls
        # generate all G
        # for all G within wedge, find omega, mu
        # if omega, mu are within motor ranges, add to list
        unit_cell = self.polycrystal.reference_cell.lattice_parameters
        sgno = self.polycrystal.reference_cell.symmetry
        sintlmin = np.sin(np.radians(theta_range[0])) / self.wavelength
        sintlmax = np.sin(np.radians(theta_range[1])) / self.wavelength
        hkls = tools.genhkl_all(unit_cell, sintlmin, sintlmax, sgno=sgno)
        reflections = []
        for hkl in hkls:
            omega, mu, theta = self.ezbragg(hkl)
            if (
                omega_range[0] < omega < omega_range[1]
                and mu_range[0] < mu < mu_range[1]
            ):
                h, k, l = hkl
                reflections.append([h, k, l, omega, mu, theta])
        return np.array(reflections)

    def compute_reflection_table(self, omega_range, mu_range, theta_range):
        reflections = np.empty((self.polycrystal.number_of_grains,), dtype=object)
        for grain_id in range(self.polycrystal.number_of_grains):
            self.select_grain(grain_id)
            refl = self.get_reflections(omega_range, mu_range, theta_range)
            reflections[grain_id] = refl
        return reflections

    def inspect(self, hkl, rotation_axis=np.array([0, 0, 1])):
        """
        Inspect the angular settings required at diffraction for the Miller indices (hkl).

        Args:
            hkl (:obj:`numpy array`): Array of Miller indices with shape `(3, n)`.
            rotation_axis (:obj:`numpy array`): Axis of rotation ``shape=(3,)``. Defaults to
                zhat=[0,0,1].

        Returns:
            df (:obj:`pandas.DataFrame`): DataFrame with Miller with columns: 'h', 'k', 'l' , 'omega', 'theta', 'eta'.
        """
        refl_labels = [f"reflection {i}" for i in range(hkl.shape[1])]
        df = pd.DataFrame(
            index=refl_labels,
            columns=[
                "h",
                "k",
                "l",
                "omega_1",
                "omega_2",
                "eta_1",
                "eta_2",
                "theta",
                "2 theta",
                "detector y_1",
                "detector z_1",
                "detector y_2",
                "detector z_2",
            ],
        )

        G = laue.get_G(self._U, self.B, hkl)
        df.h, df.k, df.l = hkl
        omega = laue.get_omega(self._U, self.unit_cell, hkl, self.energy, rotation_axis)
        df.omega_1, df.omega_2 = np.degrees(omega)
        theta = laue.get_bragg_angle(G, self.wavelength)
        df.theta = np.degrees(theta)
        df["2 theta"] = 2 * df.theta

        eta_1, eta_2 = laue.get_eta_angle(G, omega, self.wavelength, rotation_axis)
        df.eta_1, df.eta_2 = np.degrees(eta_1), np.degrees(eta_2)

        if self.detector_distance is not None:
            # Add the detector cartesian hit cooridnates for the reflecitons.
            xhat, yhat, _ = np.eye(3, 3)
            R_tth = Rotation.from_rotvec(
                (-2 * theta * yhat[:, np.newaxis]).T
            ).as_matrix()
            R_eta1 = Rotation.from_rotvec((eta_1 * xhat[:, np.newaxis]).T).as_matrix()
            R_eta2 = Rotation.from_rotvec((eta_2 * xhat[:, np.newaxis]).T).as_matrix()

            r1 = R_eta1 @ R_tth @ xhat
            r2 = R_eta2 @ R_tth @ xhat
            dy, dz = [], []
            for r in r1:
                # s * r[0] = self.detector_distance
                s = self.detector_distance / r[0]
                dy.append((s * r)[1])
                dz.append((s * r)[2])
            # TODO add the corresponding requirements on the x of the experiment table and y,z of the detector
            df["detector y_1"] = np.array(dy)[:]
            df["detector z_1"] = np.array(dz)[:]

            dy, dz = [], []
            for r in r2:
                # s * r[0] = self.detector_distance
                s = self.detector_distance / r[0]
                dy.append((s * r)[1])
                dz.append((s * r)[2])
            # TODO add the corresponding requirements on the x of the experiment table and y,z of the detector
            df["detector y_2"] = np.array(dy)[:]
            df["detector z_2"] = np.array(dz)[:]
        else:
            df = df.drop(
                columns=["detector y_1", "detector z_1", "detector y_2", "detector z_2"]
            )

        return df


if __name__ == "__main__":
    import os

    import numpy as np
    from xfab import tools

    import crispy

    pc = crispy.Polycrystal(
        os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
        group_name="Fe",
        lattice_parameters=[4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0],
        symmetry=225,
    )

    detector_distance = 4.0
    goni = crispy.dfxm.Goniometer(pc, energy=19.1, detector_distance=detector_distance)
    z_upper_bound = 1.96
    th_max = np.degrees(np.arctan(1.91 / detector_distance)) / 2

    unit_cell = goni.polycrystal.reference_cell.lattice_parameters
    sgno = goni.polycrystal.reference_cell.symmetry
    sintlmin = np.sin(np.radians(0)) / goni.wavelength
    sintlmax = np.sin(np.radians(th_max)) / goni.wavelength
    hkls = tools.genhkl_all(unit_cell, sintlmin, sintlmax, sgno=sgno)

    import cProfile
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    refl_table = goni.compute_reflection_table(
        omega_range=[-22, 22], mu_range=[0, 20], theta_range=[0, th_max]
    )

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    for gid, refl in enumerate(refl_table):
        if len(refl) > 0:
            print(f"Grain {gid} ", refl)

            hkl = refl[0][:3]
            nhat = goni.polycrystal.grains[gid].ub @ hkl
            nhat /= np.linalg.norm(nhat)

            xy_projection = nhat.copy()
            xy_projection[2] = 0
            xy_projection /= np.linalg.norm(xy_projection)
            ang = np.arccos(
                np.dot(xy_projection, np.array([np.sign(xy_projection[0]), 0, 0]))
            )

            Rang = Rotation.from_rotvec(ang * np.array([0, 0, 1])).as_matrix()
            nhat_xz = Rang @ nhat

            wedge = np.arccos(np.dot(np.array([0, 0, 1]), nhat_xz))
            print(nhat_xz)

            print(np.degrees(wedge), refl[0][5])

            print(np.degrees(ang))
            print(nhat)
