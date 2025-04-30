import h5py
import numpy as np

import crispy


class GrainMap:
    """Factory class to represent a polycrystal.

    This class is a factory for creating polycrystal objects. It can create
    polycrystal objects from a list of ImageD11.grain.grain objects or from
    a path to an HDF5 file containing the grain information. The class also
    supports loading grain information from a lab-dct grain volume file.

    Depending on the input, it will return either a crispy._tdxrd_map.TDXRDMap
    or a crispy._dct_volume.LabDCTVolume. Common to the GrainMap is the
    attribute `grains`, which is a list of ImageD11.grain.grain objects
    representing the grains in the polycrystal. Moeover a polycrystal object
    always hold a reference cell, which is a unitcell object describing the
    lattice parameters and symmetry group of the material phase.

    Any polycrystal object can be mounted on a crispy.dfxm.Goniometer object to

    Args:
        grain_data (:obj:`str` or :obj:`list` of :obj:`ImageD11.grain.grain`):
            Absolute path to the HDF5 file containing the grain information, or
            alternatively a list of ImageD11.grain.grain objects. This may also
            be a path to a lab-dct grain volume file.
        group_name (:obj:`str`): The name of the group in the HDF5 file containing
            the grain information. Defaults to "grains". Only used if
            `grain_data` is a string to an HDF5 file with a 3DXRD grain map.
        lattice_parameters (:obj:`numpy array`): The lattice parameters of the
            polycrystal [a, b, c, alpha, beta, gamma]. Defaults to None.
        symmetry (:obj:`int`): The symmetry group of the phase. Defaults to None.

    Returns:
        A crispy._tdxrd_map.TDXRDMap or a crispy._dct_volume.LabDCTVolume.
    """

    def __new__(
        cls, grain_data, group_name="grains", lattice_parameters=None, symmetry=None
    ):
        if isinstance(grain_data, (list, np.ndarray)):
            return crispy.TDXRDMap(grain_data, group_name, lattice_parameters, symmetry)
        elif isinstance(grain_data, str):
            if not grain_data.endswith(".h5"):
                raise ValueError("File path does not end with .h5")
            with h5py.File(grain_data, "r") as f:
                if "LabDCT" in list(f.keys()):
                    return crispy.LabDCTVolume(grain_data)
                else:
                    return crispy.TDXRDMap(
                        grain_data, group_name, lattice_parameters, symmetry
                    )
        else:
            raise ValueError(
                "grain_data must be an array of ImageD11.grain.grain objects or a path to an HDF5 file."
            )

if __name__ == "__main__":
    pass