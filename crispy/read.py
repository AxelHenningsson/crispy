import ImageD11.grain
import numpy as np

def grains(filename, group_name="grains"):
    """Read the grains from a HDF5 file. Follows the ImageD11 format.

    Args:
        filename (:obj:`str`): Absolute path to the HDF5 file containing the grain
            information.
        group_name (:obj:`str`): The name of the group in the HDF5 file containing
            the grain information. Defaults to "grains".

    Returns:
        :obj:`list` of :obj:`ImageD11.grain.grain`: List of grains in the HDF5 file.
    """
    return np.array( ImageD11.grain.read_grain_file_h5(filename, group_name) )


if __name__ == "__main__":
    pass
