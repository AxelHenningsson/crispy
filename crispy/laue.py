"""Collection of functions for solving time dependent Laue equations for arbitrary rigid body motions.
This module is a modiefied version of what is found in the FABLE package `xrd_simulator` by Axel Henningsson.
"""

import numpy as np


def get_G(U, B, G_hkl):
    """Compute the diffraction vector

    .. math::
        \\boldsymbol{G} = \\boldsymbol{U}\\boldsymbol{B}\\boldsymbol{G}_{hkl}

    Args:
        U (:obj:`numpy array`) Orientation matrix of ``shape=(3,3)`` (unitary).
        B (:obj:`numpy array`): Reciprocal to grain coordinate mapping matrix of ``shape=(3,3)``.
        G_hkl (:obj:`numpy array`): Miller indices, i.e the h,k,l integers (``shape=(3,n)``).


    Returns:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector. (``shape=(3,n)``)

    """
    return np.dot(np.dot(U, B), G_hkl)


def get_lattice_normal(U, B, G_hkl):
    """Compute the lattice normal vector

    Args:
        U (:obj:`numpy array`) Orientation matrix of ``shape=(3,3)`` (unitary).
        B (:obj:`numpy array`): Reciprocal to grain coordinate mapping matrix of ``shape=(3,3)``.
        G_hkl (:obj:`numpy array`): Miller indices, i.e the h,k,l integers (``shape=(3,n)``).

    Returns:
        n (:obj:`numpy array`): Sample coordinate system lattice normal. (``shape=(3,n)``)
    """
    G = get_G(U, B, G_hkl)
    return G / np.linalg.norm(G, axis=0)


def get_eta_angle(G, omega, wavelength, rotation_axis):
    """Compute a eta angle given a diffraction (G) vector.

    It is assumed that eta is defined as the angle between the projection of the
    outgoing wave vector unto the lab y-z plane and the z-lab-axis. I.e eta signifies
    the rotation of the crystal around the beam if the beam is directed along the x-lab-axis.

    Args:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector. (``shape=(3,n)``)
        omega (:obj:`numpy array`): Omega table for diffraction, each column is associated with
            a G-vector (``shape=(m,n)``)
        wavelength (:obj:`float`): Photon wavelength in units of angstrom.
        rotation_axis (:obj:`numpy array`): Axis of rotation. ``shape=(3,)`


    Returns:
        Eta angles (:obj:`numpy array`): Table of eta values in units of radians. (``shape=(m,n)``)

    """

    rx, ry, rz = rotation_axis
    K = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
    K2 = K.dot(K)

    wave_vector = 2 * np.pi * np.array([1, 0, 0]) / wavelength

    eta = np.zeros_like(omega)
    for i in range(eta.shape[0]):
        for j in range(eta.shape[1]):
            om = omega[i, j]
            if np.isnan(om):
                eta[i, j] = np.nan
            else:
                Romega = np.eye(3, 3) + np.sin(om) * K + (1 - np.cos(om)) * K2
                G_lab = Romega @ G[:, j]
                k_lab = G_lab + wave_vector
                kx, ky, kz = k_lab
                eta[i, j] = -np.sign(ky) * np.arccos(kz / np.sqrt(ky**2 + kz**2))

    return eta


def get_bragg_angle(G, wavelength):
    """Compute a Bragg angle given a diffraction (scattering) vector.

    Args:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector. (``shape=(3,n)``)
        wavelength (:obj:`float`): Photon wavelength in units of angstrom.

    Returns:
        Bragg angles (:obj:`float`): in units of radians. (``shape=(n,)``)

    """
    return np.arcsin(np.linalg.norm(G, axis=0) * wavelength / (4 * np.pi))


def get_sin_theta_and_norm_G(G, wavelength):
    """Compute a Bragg angle given a diffraction (scattering) vector.

    Args:
        G (:obj:`numpy array`): Sample coordinate system diffraction vector.
        wavelength (:obj:`float`): Photon wavelength in units of angstrom.

    Returns:
        sin(Bragg angle) (:obj:`float`): in units of radians and ||G||.

    """
    normG = np.linalg.norm(G, axis=0)
    return normG * wavelength / (4 * np.pi), normG


def get_b_matrix(unit_cell):
    """
    Calculate B matrix such that B^-T contains the reals space lattice vectors as columns.

    Args:
        unit_cell (:obj:`numpy array`): unit cell parameters [a,b,c,alpha,beta,gamma]. ``shape=(6,)`

    Returns:
        B (:obj:`numpy array`): The B matrix. ``shape=(3,3)`
    """
    a, b, c = unit_cell[0:3]
    alpha, beta, gamma = np.radians(unit_cell[3:])
    calp = np.cos(alpha)
    cbet = np.cos(beta)
    cgam = np.cos(gamma)
    salp = np.sin(alpha)
    sbet = np.sin(beta)
    sgam = np.sin(gamma)
    V = (
        a
        * b
        * c
        * np.sqrt(1 - calp * calp - cbet * cbet - cgam * cgam + 2 * calp * cbet * cgam)
    )
    astar = 2 * np.pi * b * c * salp / V
    bstar = 2 * np.pi * a * c * sbet / V
    cstar = 2 * np.pi * a * b * sgam / V
    sbetstar = V / (a * b * c * salp * sgam)
    sgamstar = V / (a * b * c * salp * sbet)
    cbetstar = (calp * cgam - cbet) / (salp * sgam)
    cgamstar = (calp * cbet - cgam) / (salp * sbet)
    B = np.array(
        [
            [astar, bstar * cgamstar, cstar * cbetstar],
            [0, bstar * sgamstar, -cstar * sbetstar * calp],
            [0, 0, cstar * sbetstar * salp],
        ]
    )
    return B


def get_rotmat(rotation_axis, angle):
    """
    Compute the rotation matrix for a given rotation axis and angle using Rodrigues' rotation formula.

    Args:
        rotation_axis (:obj:`numpy array`): Axis of rotation. `shape=(3,)`
        angle (:obj:`float`): The rotation angle in radians.

    Returns:
        rotmat (:obj:`numpy array`): Rotation matrix. `shape=(3, 3)`

    """
    rx, ry, rz = rotation_axis
    K = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
    K2 = K.dot(K)
    return np.eye(3, 3) + np.sin(angle) * K + (1 - np.cos(angle)) * K2


def get_omega(U, cell, hkl, energy, rotation_axis):
    """
    Given a set of Miller indices and a crystal state we find all omega values that will diffract.

    It is implicitly assumed that the inident beam is a long the lab x-axis.

    Args:
        U (:obj:`numpy array`): Crystal orientation matrix `shape=(3,3)`
        unit_cell (:obj:`numpy array`): unit cell parameters [a,b,c,alpha,beta,gamma]. `shape=(6,)`
        hkl (:obj:`numpy array`): Miller indices. `shape=(3,n)`
        energy (:obj:`float`): Photon energy in keV.
        rotation_axis (:obj:`numpy array`): Axis of rotation. `shape=(3,)`

    Returns:
        omega (:obj:`numpy array`): The omega values for diffraction. `shape=(n,)`
    """
    B = get_b_matrix(cell)
    G_0 = U @ B @ hkl
    wavelength = keV_to_angstrom(energy)

    wave_vector = 2 * np.pi * np.array([1, 0, 0]) / wavelength

    rx, ry, rz = rotation_axis
    K = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
    K2 = K.dot(K)
    rho_0_factor = -wave_vector.dot(K2)
    rho_1_factor = wave_vector.dot(K)
    rho_2_factor = wave_vector.dot(np.eye(3, 3) + K2)
    rho_0s = rho_0_factor.dot(G_0)
    rho_1s = rho_1_factor.dot(G_0)
    rho_2s = rho_2_factor.dot(G_0) + np.sum((G_0 * G_0), axis=0) / 2.0

    omega = _solve_tan_half_angle_equation(rho_0s, rho_1s, rho_2s)

    return omega


def angstrom_to_keV(wavelength):
    """Convert wavelength in angstrom to energy in KeV.

    Args:
        wavelength (:obj:`float`): wavelength in units of angstrom.

    Returns:
        energy (:obj:`float`): Photon energy in keV.
    """
    return keV_to_angstrom(wavelength)


def keV_to_angstrom(energy):
    """Cnnvert KeV energy to wavelength in angstrom.

    Args:
        energy (:obj:`float`): Photon energy in keV.

    Returns:
        wavelength (:obj:`float`): wavelength in units of angstrom.
    """
    h = 6.62607015 * 1e-34
    c = 299792458.0
    eV_to_joules = 1.60217663 * 1e-19
    return 1e10 * (h * c) / (energy * 1e3 * eV_to_joules)


def _solve_tan_half_angle_equation(rho_0, rho_1, rho_2):
    """Find all solutions, :obj:`t`, to the equation (maximum 2 solutions exists)

    .. math::
        \\rho_0 \\cos(t \\Delta \\omega) + \\rho_1 \\sin(t \\Delta \\omega) + \\rho_2 = 0. \\quad\\quad (1)

    by rewriting as

    .. math::
        (\\rho_2 - \\rho_0) s^2 + 2 \\rho_1 s + (\\rho_0 + \\rho_2) = 0. \\quad\\quad (2)

    where

    .. math::
        s = \\tan(t \\Delta \\omega / 2). \\quad\\quad (3)

    and

        .. math:: \\Delta \\omega

    is a rotation angle

    Args:
        \\rho_0,\\rho_1,\\rho_2 (:obj:`float`): Coefficients \\rho_0,\\rho_1 and \\rho_2 of equation (1).
        delta_omega (:obj:`float`): Radians of rotation.

    Returns:
        (:obj:`tuple` of :obj:`numpy.array`): 2 Arrays of solutions of matching shape to input. if no solutions exist on
        an interval the corresponding instances in the solution array holds np.nan values.

    """

    denominator = rho_2 - rho_0

    m = np.abs(denominator) < 1e-12
    denominator[m] = 0
    a = np.divide(
        rho_1, denominator, out=np.full_like(rho_0, np.nan), where=denominator != 0
    )
    b = np.divide(
        rho_0 + rho_2,
        denominator,
        out=np.full_like(rho_0, np.nan),
        where=denominator != 0,
    )

    # handle loss of numerical precision case
    m = np.abs(rho_0 + rho_2) < 1e-12
    b[m] = 0

    rootval = a**2 - b

    leadingterm = -a
    rootval[rootval < 0] = np.nan
    s1 = leadingterm + np.sqrt(rootval)
    s2 = leadingterm - np.sqrt(rootval)
    om1 = 2 * np.arctan(s1)
    om2 = 2 * np.arctan(s2)

    # handle single solution case
    m1 = (np.abs(rho_0 - rho_2) < 1e-12) * (np.abs(rho_1) > 1e-12)
    m2 = (np.abs(rho_0 - rho_2) < 1e-12) * (np.abs(rho_1) < 1e-12)
    om1[m1] = 2 * np.arctan(-rho_0[m1] / rho_1[m1])
    om1[m2] = np.pi
    om2[m1] = np.nan
    om2[m2] = np.nan

    om1[om1 < 0] += 2 * np.pi
    om2[om2 < 0] += 2 * np.pi

    om2[(om1 == 0) * (om2 == 0)] += 2 * np.pi

    return np.array([om1, om2])
    return np.array([om1, om2])
