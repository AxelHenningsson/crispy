"""Module to load example data and phantoms.

Here you will find example datasets and phantoms that can be used to test and
demonstrate the functionality of the crispy package. These assets are stored in
the `assets` directory of the repository.

"""

import os

from ._lab_dct_volume import LabDCTVolume
from ._tdxrd_map import TDXRDMap

# path to the root directory of the repository
_root_path = os.path.abspath(os.path.join(os.path.abspath(__file__), "..", ".."))

# path to the assets directory
_asset_path = os.path.join(_root_path, "assets")


class path:
    """Absolute paths to various assets.

    Currently availabel example datasets are:
        - .../crispy/assets/FeAu_0p5_tR_ff1_grains.h5
        - .../crispy/assets/lab_dct_silicon.h5
        - .../crispy/assets/lab_dct_Al1050.h5

    """

    FEAU = os.path.join(_asset_path, "FeAu_0p5_tR_ff1_grains.h5")
    SILICON = os.path.join(_asset_path, "lab_dct_silicon.h5")
    AL1050 = os.path.join(_asset_path, "lab_dct_Al1050.h5")


class grain_map:
    """Class to load example grain maps.

    Various assets and example dataset are available in the `assets` directory.

    """

    def tdxrd_map():
        """Load a grain map from the ID11 beamline at ESRF on a FeAu sample.

        Returns:
            :obj:`list` of :obj:`ImageD11.grain.grain`: List of grains in the grain map.
        """
        return TDXRDMap(
            path.FEAU,
            group_name="Fe",
            lattice_parameters=[4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0],
            symmetry=225,  # cubic fcc
        )

    def lab_dct_volume(name="Al1050"):
        """Load a lab-dct volume.

        Args:
            name (:obj:`str`): The name of the lab-dct volume to load.
                This can be one of the following:
                - "Al1050": A lab-dct volume of a small polycrystalline Al1050 sample.
                - "silicon": A lab-dct volume of three silicon single crystals shards.
                Defaults to "Al1050".

        Raises:
            ValueError: If the name is not one of the available assets.

        Returns:
            :obj:`crispy._lab_dct_volume.LabDCTVolume`: Lab-dct volume.

        """

        if name == "Al1050":
            return LabDCTVolume(path.AL1050)
        elif name == "silicon":
            return LabDCTVolume(path.SILICON)
        else:
            raise ValueError(f"No asset found for name = {name}")


if __name__ == "__main__":
    pass
