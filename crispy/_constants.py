import numpy as np


class CONSTANTS:
    """Constants for the crispy package.

    This class contains constants that are used throughout the crispy package.

    This is considered a private class and should not be used outside of the
    crispy package.
    """

    # to easily interface with xfab for misorientation calculations
    CRYSTAL_SYSTEM_STR_TO_INT = {
        "triclinic": 1,
        "monoclinic": 2,
        "orthorhombic": 3,
        "tetragonal": 4,
        "trigonal": 5,
        "hexagonal": 6,
        "cubic": 7,
    }

    # to easily interface with xfab for misorientation calculations
    SPACEGROUP_TO_CRYSTAL_SYSTEM = np.empty((230,), dtype="object")
    SPACEGROUP_TO_CRYSTAL_SYSTEM[0:2] = "triclinic"
    SPACEGROUP_TO_CRYSTAL_SYSTEM[3:16] = "monoclinic"
    SPACEGROUP_TO_CRYSTAL_SYSTEM[16:75] = "orthorhombic"
    SPACEGROUP_TO_CRYSTAL_SYSTEM[75:143] = "tetragonal"
    SPACEGROUP_TO_CRYSTAL_SYSTEM[143:168] = "trigonal"
    SPACEGROUP_TO_CRYSTAL_SYSTEM[168:195] = "hexagonal"
    SPACEGROUP_TO_CRYSTAL_SYSTEM[195:] = "cubic"

    # structuring element for 3D binary image processing for lab-dct volumes
    DEFAULT_STRUCT = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
