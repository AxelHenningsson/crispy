import os
import unittest

import numpy as np

import crispy


class TestLabDCTVolume(unittest.TestCase):
    def setUp(self):
        self.debug = False
        np.random.seed(42)

    def test_polycrystal(self):
        pc = crispy.GrainMap(
            os.path.join(crispy.assets._asset_path, "lab_dct_silicon.h5"),
        )

        pc = crispy.GrainMap(
            os.path.join(crispy.assets._asset_path, "lab_dct_Al1050.h5"),
        )

        pc.texturize()


if __name__ == "__main__":
    unittest.main()
