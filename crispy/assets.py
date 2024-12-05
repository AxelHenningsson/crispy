"""Module to load example data and phantoms."""

import os

import ImageD11.unitcell
import matplotlib.pyplot as plt
import numpy as np

import crispy
import crispy.read
import crispy.tesselate
import crispy.vizualise


# path to the root directory of the repository
_root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))

# path to the assets directory
_asset_path = os.path.join(_root_path, "assets")


def grainmap_id11():
    """Load a grain map from the ID11 beamline at ESRF on a FeAu sample.

    Returns:
        :obj:`list` of :obj:`ImageD11.grain.grain`: List of grains in the grain map.
    """
    filename = os.path.join(_asset_path, "FeAu_0p5_tR_ff1_grains.h5")
    return crispy.read.grains(filename, group_name="Fe")


if __name__ == "__main__":
    grains = grainmap_id11()
    x, y, z = np.array([g.translation for g in grains]).T

    xg = np.linspace(x.min(), x.max(), 256)
    yg = np.linspace(y.min(), y.max(), 256)
    zg = np.linspace(z.min(), z.max(), 128)

    import cProfile
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    tessmap = crispy.tesselate.voroni(grains, (xg, yg, zg))

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    crispy.vizualise.save_voxels("test", tessmap, (xg, yg, zg))

    unit_cell = [4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0]
    spacegroup = 225

    ref_cell = ImageD11.unitcell.unitcell(unit_cell, spacegroup)
    for g in grains:
        g.ref_unitcell = ref_cell

    ipf_ax = np.array([0.0, 0.0, 1.0])
    c = [g.get_ipf_colour(ipf_ax) for g in grains]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(x, y, z, c=c, cmap="viridis", s=50)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()
