import h5py
import numpy as np

from ._lab_dct_volume import LabDCTVolume
from ._tdxrd_map import TDXRDMap


class GrainMap:
    """Factory class for creating :class:`crispy.Polycrystal <crispy._polycrystal.Polycrystal>` objects.

    Creates :class:`crispy.Polycrystal <crispy._polycrystal.Polycrystal>` objects from
    either a list of :obj:`ImageD11.grain.grain` objects or an HDF5 file containing
    grain information. Supports both 3DXRD grain maps and lab-DCT grain volumes.

    Example:
        .. code-block:: python

            import crispy

            # Load a 3DXRD grain map
            path_to_h5_3dxrd = crispy.assets.path.FEAU
            grain_map_3dxrd = GrainMap(path_to_h5_3dxrd)

            # Load a lab-DCT grain volume
            path_to_h5_lab_dct = crispy.assets.path.AL1050
            grain_map_lab_dct = GrainMap(path_to_h5_lab_dct)

            # Access grains and reference cell
            print(grain_map_3dxrd.grains)
            print(grain_map_lab_dct.grains)
            print(grain_map_3dxrd.reference_cell)
            print(grain_map_lab_dct.reference_cell)

    The factory returns either a :class:`crispy.TDXRDMap <crispy._tdxrd_map.TDXRDMap>`
    (for 3DXRD) or a :class:`crispy.LabDCTVolume <crispy._lab_dct_volume.LabDCTVolume>`
    (for lab-DCT). Both types contain a `grains` attribute - a list of
    `ImageD11.grain.grain <https://github.com/FABLE-3DXRD/ImageD11/blob/f1766f525288a6c49f20f98ee7d95ea9233db430/ImageD11/grain.py#L49>`_
    objects representing individual grains. They also contain a reference cell that
    describes the lattice parameters and symmetry group of the material phase.

    Any resulting :class:`crispy.Polycrystal <crispy._polycrystal.Polycrystal>` can be
    mounted on a :obj:`crispy.dfxm.Goniometer` to analyze accessible diffraction
    reflections for Dark-Field X-ray Microscopy.

    Args:
        grain_data (:obj:`str` or :obj:`list` of :obj:`ImageD11.grain.grain`):
            Either an absolute path to an HDF5 file containing grain data (3DXRD map
            or lab-DCT volume), or a list of ImageD11 grain objects. Example files
            can be found in the `crispy.assets` module.
        group_name (:obj:`str`): Name of the HDF5 group containing grain information.
            Only used for 3DXRD HDF5 files. Defaults to "grains".
        lattice_parameters (:obj:`numpy.ndarray`): Crystal lattice parameters
            [a, b, c, alpha, beta, gamma]. Defaults to None.
        symmetry (:obj:`int`): Symmetry group of the phase. Defaults to None.

    Returns:
        Either :class:`crispy.TDXRDMap <crispy._tdxrd_map.TDXRDMap>` or
        :class:`crispy.LabDCTVolume <crispy._lab_dct_volume.LabDCTVolume>`.

    Raises:
        ValueError: If grain_data is neither a list of grain objects nor a valid
            HDF5 file path.
    """

    def __new__(
        cls, grain_data, group_name="grains", lattice_parameters=None, symmetry=None
    ):
        if isinstance(grain_data, (list, np.ndarray)):
            return TDXRDMap(grain_data, group_name, lattice_parameters, symmetry)
        elif isinstance(grain_data, str):
            if not grain_data.endswith(".h5"):
                raise ValueError("File path does not end with .h5")
            with h5py.File(grain_data, "r") as f:
                if "LabDCT" in list(f.keys()):
                    return LabDCTVolume(grain_data)
                else:
                    return TDXRDMap(
                        grain_data, group_name, lattice_parameters, symmetry
                    )
        else:
            raise ValueError(
                "grain_data must be an array of ImageD11.grain.grain objects or a path \
                    to an HDF5 file."
            )


if __name__ == "__main__":
    pass
