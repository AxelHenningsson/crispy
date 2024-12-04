import numpy as np
import meshio


def save_voxels(file, data, coord):
    """write a tesselated map to a paraview readable file.

    Args:
        file (:obj:`string`): Absolute path ending with desired filename.
        data (:obj:`numpy.array`): scalar data to be saved, shape=(m,n,o).
        coord (:obj:`iterable`): x,y,z coordinates.
    """
    x, y, z = coord
    points = np.meshgrid(x, y, z, indexing="ij")
    points = np.array(points).reshape(3, len(x) * len(y) * len(z))

    cells = [("vertex", np.array([[i] for i in range(points.shape[0])]))]
    if len(file.split(".")) == 1:
        filename = file + ".xdmf"
    else:
        filename = file

    voxel_size_x = x[1] - x[0]
    voxel_size_y = y[1] - y[0]
    voxel_size_z = z[1] - z[0]

    voxel_size = np.zeros_like(points)
    voxel_size[0] = voxel_size_x
    voxel_size[1] = voxel_size_y
    voxel_size[2] = voxel_size_z

    meshio.Mesh(
        points.T,
        cells,
        point_data={
            "data": data.flatten(),
            "Voxel Size X": np.ones((points.shape[1],)) * voxel_size_x,
            "Voxel Size Y": np.ones((points.shape[1],)) * voxel_size_y,
            "Voxel Size Z": np.ones((points.shape[1],)) * voxel_size_z,
            "Voxel Size": voxel_size.T,
        },
    ).write(filename)
