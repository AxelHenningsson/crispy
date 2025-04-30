"""Module to load example data and phantoms."""

import os

import numpy as np

import crispy
import crispy._read
import crispy._tesselate
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
    return crispy._read.grains(filename, group_name="Fe")


if __name__ == "__main__":
    pass