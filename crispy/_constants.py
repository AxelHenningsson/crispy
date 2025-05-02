import numpy as np


class CONSTANTS:
    # to easily interface with xfab for misorientation calculations
    _CRYSTAL_SYSTEM_STR_TO_INT = {
        "triclinic": 1,
        "monoclinic": 2,
        "orthorhombic": 3,
        "tetragonal": 4,
        "trigonal": 5,
        "hexagonal": 6,
        "cubic": 7,
    }

    _SPACEGROUP_TO_CRYSTAL_SYSTEM = np.empty((230,), dtype="object")
    _SPACEGROUP_TO_CRYSTAL_SYSTEM[0:2] = "triclinic"
    _SPACEGROUP_TO_CRYSTAL_SYSTEM[3:16] = "monoclinic"
    _SPACEGROUP_TO_CRYSTAL_SYSTEM[16:75] = "orthorhombic"
    _SPACEGROUP_TO_CRYSTAL_SYSTEM[75:143] = "tetragonal"
    _SPACEGROUP_TO_CRYSTAL_SYSTEM[143:168] = "trigonal"
    _SPACEGROUP_TO_CRYSTAL_SYSTEM[168:195] = "hexagonal"
    _SPACEGROUP_TO_CRYSTAL_SYSTEM[195:] = "cubic"

    # structuring element for 3D binary image processing
    DEFAULT_STRUCT = np.array(
        [
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
