"""Write a fake MIDAS ``Grains.csv`` for interstitial-free (IF) steel.

The file layout follows MIDAS/FF_HEDM/Example/GrainsSim.csv (ProcessGrains.c).
Everything is in the MIDAS / ESRF frame: x along the beam, y, z up.

Texture: a weak BCC rolling texture with the rolling direction (RD) along
lab z, the sheet normal (ND) along lab x and the transverse direction (TD)
along lab y:

    alpha fibre  <110> || RD   (from {001}<110> to {111}<110>)
    gamma fibre  {111} || ND
    the rest is random

Run:  python make_fake_if_steel_grains.py
"""

import os

import numpy as np
from scipy.spatial.transform import Rotation

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
SEED = 0
N_GRAINS = 1000
OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Grains_IF_steel.csv")

SPACE_GROUP = 229  # Im-3m, ferrite
A0 = 2.8665  # angstrom
STRAIN_STD = 3e-4  # scatter of the fitted cell around the reference
ANGLE_STD = 0.02  # degrees, scatter of the fitted cell angles

FRACTION_ALPHA = 0.25
FRACTION_GAMMA = 0.30  # remaining grains are random
TEXTURE_SPREAD = 7.0  # degrees, scatter about the ideal fibres

SAMPLE_RADIUS = 250.0  # micrometres, cylinder axis along z
SAMPLE_HEIGHT = 600.0  # micrometres
MEAN_GRAIN_RADIUS = 30.0  # micrometres
BEAM_THICKNESS = SAMPLE_HEIGHT


def unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def orientation(nd, rd):
    """Crystal-to-sample matrix with crystal ``nd`` along x and ``rd`` along z."""
    nd, rd = unit(nd), unit(rd)
    td = np.cross(rd, nd)
    return np.array([nd, td, rd])


def about(axis, angles):
    """Rotation matrices about ``axis`` for an array of angles in degrees."""
    rotvec = np.radians(angles)[:, None] * np.asarray(axis, dtype=float)
    return Rotation.from_rotvec(rotvec).as_matrix()


def make_orientations(rng):
    n_alpha = int(FRACTION_ALPHA * N_GRAINS)
    n_gamma = int(FRACTION_GAMMA * N_GRAINS)
    n_random = N_GRAINS - n_alpha - n_gamma

    # Alpha fibre: start at {001}<110> and turn about RD (lab z) by up to
    # +-54.7 degrees, which reaches {111}<110>.
    start = orientation(nd=[0, 0, 1], rd=[1, 1, 0])
    angles = rng.uniform(-54.74, 54.74, n_alpha)
    alpha = about([0, 0, 1], angles) @ start

    # Gamma fibre: {111} along ND (lab x), any rotation about ND.
    start = orientation(nd=[1, 1, 1], rd=[1, -1, 0])
    angles = rng.uniform(0, 360, n_gamma)
    gamma = about([1, 0, 0], angles) @ start

    fibres = np.concatenate((alpha, gamma))
    scatter = Rotation.from_rotvec(
        np.radians(rng.normal(0, TEXTURE_SPREAD / np.sqrt(3), (len(fibres), 3)))
    ).as_matrix()
    fibres = scatter @ fibres

    # Present each grain through a random cubic symmetry equivalent, as an
    # indexer would, so that the texture is not visible in the raw matrices.
    random = Rotation.from_quat(rng.normal(size=(n_random, 4))).as_matrix()
    sym = Rotation.create_group("O").as_matrix()
    fibres = fibres @ sym[rng.integers(0, len(sym), len(fibres))]

    u = np.concatenate((fibres, random))
    return u[rng.permutation(N_GRAINS)]


def make_positions(rng, min_distance=MEAN_GRAIN_RADIUS):
    """Random grain centres in a cylinder, no two closer than min_distance."""
    points = np.empty((0, 3))
    while len(points) < N_GRAINS:
        r = SAMPLE_RADIUS * np.sqrt(rng.uniform())
        a = rng.uniform(0, 2 * np.pi)
        p = np.array([r * np.cos(a), r * np.sin(a), SAMPLE_HEIGHT * rng.uniform(-0.5, 0.5)])
        if len(points) == 0 or np.linalg.norm(points - p, axis=1).min() > min_distance:
            points = np.vstack((points, p))
    return points


def main():
    rng = np.random.default_rng(SEED)
    u = make_orientations(rng)
    xyz = make_positions(rng)

    cells = np.empty((N_GRAINS, 6))
    cells[:, :3] = A0 * (1 + rng.normal(0, STRAIN_STD, (N_GRAINS, 3)))
    cells[:, 3:] = 90 + rng.normal(0, ANGLE_STD, (N_GRAINS, 3))

    radius = rng.lognormal(np.log(MEAN_GRAIN_RADIUS), 0.3, N_GRAINS)
    confidence = rng.uniform(0.85, 1.0, N_GRAINS)
    diff_pos = np.abs(rng.normal(0, 5, N_GRAINS))  # micrometres
    diff_ome = np.abs(rng.normal(0, 0.05, N_GRAINS))  # degrees
    diff_angle = np.abs(rng.normal(0, 0.02, N_GRAINS))  # degrees
    euler = Rotation.from_matrix(u).as_euler("ZXZ", degrees=True)

    columns = (
        ["GrainID"]
        + [f"O{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
        + ["X", "Y", "Z", "a", "b", "c", "alpha", "beta", "gamma"]
        + ["DiffPos", "DiffOme", "DiffAngle", "GrainRadius", "Confidence"]
        + [f"eFab{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
        + [f"eKen{i}{j}" for i in (1, 2, 3) for j in (1, 2, 3)]
        + ["RMSErrorStrain", "PhaseNr", "Eul0", "Eul1", "Eul2"]
    )
    data = np.column_stack(
        [
            np.arange(1, N_GRAINS + 1),
            u.reshape(N_GRAINS, 9),
            xyz,
            cells,
            diff_pos,
            diff_ome,
            diff_angle,
            radius,
            confidence,
            np.zeros((N_GRAINS, 19)),  # eFab, eKen, RMSErrorStrain (as in GrainsSim.csv)
            np.ones(N_GRAINS),  # PhaseNr
            euler,
        ]
    )
    assert data.shape[1] == len(columns)

    with open(OUTPUT, "w") as f:
        f.write(f"%NumGrains {N_GRAINS}\n")
        f.write("%BeamCenter 0.000000\n")
        f.write(f"%BeamThickness {BEAM_THICKNESS:f}\n")
        f.write("%GlobalPosition 0.000000\n")
        f.write("%NumPhases 1\n")
        f.write(f"%PhaseInfo\n%\tSpaceGroup:{SPACE_GROUP}\n")
        f.write(
            "%\tLattice Parameter: "
            + " ".join(f"{x:f}" for x in [A0, A0, A0, 90, 90, 90])
            + "\n"
        )
        f.write("%" + "\t".join(columns) + "\n")
        for row in data:
            f.write(f"{int(row[0])}\t" + "\t".join(f"{x:f}" for x in row[1:]) + "\t\n")
    print(f"Wrote {N_GRAINS} grains to {OUTPUT}")


if __name__ == "__main__":
    main()
