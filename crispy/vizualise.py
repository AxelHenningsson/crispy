"""The vizualisation module contains functions genreally targeting visualising voronoi tesselations
of polycrystals. I.e these function are usefull for :class:`crispy.TDXRDMap` <crispy._tdxrd_map.TDXRDMap>

For lab-DCT volumes, see :func:`crispy.LabDCTVolume.write()` <crispy._lab_dct_volume.LabDCTVolume.write>
for writing to paraview files to disc for visualization.
"""

import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from . import _tesselate
from ._lab_dct_volume import LabDCTVolume


def _snap_to_bounds(ax, polycrystal):
    """Set the aspect ratio of the plot to be equal to the extent of the polycrystal + 10% padding."""
    bounds = polycrystal.bounding_box
    extent = polycrystal.extent
    ax.set_xlim(bounds[0, 0] - 0.1 * extent[0], bounds[0, 1] + 0.1 * extent[0])
    ax.set_ylim(bounds[1, 0] - 0.1 * extent[1], bounds[1, 1] + 0.1 * extent[1])
    ax.set_zlim(bounds[2, 0] - 0.1 * extent[2], bounds[2, 1] + 0.1 * extent[2])
    ax.set_box_aspect([extent[0], extent[1], extent[2]])


def _xyz_labels(ax):
    ax.set_xlabel("X um", labelpad=20)
    ax.set_ylabel("Y um", labelpad=20)
    ax.set_zlabel("Z um", labelpad=20)


def mesh(polycrystal, neighbourhood=None, select=None):
    """Plot the Voronoi tesselation of the polycrystal.

    Example:

    .. code-block:: python

        import matplotlib.pyplot as plt
        import numpy as np
        import crispy

        polycrystal = crispy.assets.grain_map.tdxrd_map()
        polycrystal.tesselate()
        polycrystal.colorize(np.eye(3))

        fig, ax = crispy.vizualise.mesh(polycrystal)
        plt.show()

    .. image:: ../../docs/source/images/mesh.png

    Args:
        polycrystal (:obj:`crispy.TDXRDMap`): The polycrystal object.
        neighbourhood (:obj:`int`, optional): When not None, the grain
            with number -neighbourhood- and all of its neighbours are
            plotted. Default is None, in which case the entire polycrystal
            mesh is plotted.
        select (:obj:`list` of :obj:`int`, optional): List of grain indices
            to plot. Default is None.

    Returns:
        :obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`
    """
    if isinstance(polycrystal, LabDCTVolume):
        raise ValueError(
            "Lab-DCT volumes are not yet supported for mesh visualization."
        )
    assert polycrystal._mesh is not None, (
        "Mesh not yet generated. Call polycrystal.tesselate() first."
    )

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(9, 7))

    if neighbourhood is not None:
        grains_index = polycrystal._select_grains(None, neighbourhood)
        local_geometry = polycrystal._extract_geom(grains_index)
        mesh = _tesselate._build_mesh(*local_geometry)
        grains = polycrystal.grains[grains_index]
    elif select is not None:
        grains_index = select
        local_geometry = polycrystal._extract_geom(grains_index)
        mesh = _tesselate._build_mesh(*local_geometry)
        grains = polycrystal.grains[grains_index]
    else:
        mesh = polycrystal._mesh
        grains = polycrystal.grains

    if hasattr(grains[0], "rgb"):
        facecolors = []
        for i in range(len(mesh.cells_dict["polygon"])):
            gid = mesh.cell_data["grain_id"][0][i]
            facecolors.append(tuple(polycrystal.grains[gid].rgb[:, 0]) + (1,))
        axis = (
            str(polycrystal._ipf_axes[0].round(3))
            .replace(" ", ",")
            .replace("[", "")
            .replace("]", "")
        )
        coloring = "ipf (view-axis x,y,z = {})".format(axis)
    else:
        cmap = cm.rainbow
        norm = Normalize(vmin=0, vmax=polycrystal.number_of_grains)
        facecolors = [
            cmap(norm(mesh.cell_data["grain_id"][0][i]))
            for i in range(len(mesh.cells_dict["polygon"]))
        ]
        coloring = "grain id"

    triangles = [mesh.points[poly] for poly in mesh.cells_dict["polygon"]]

    ax.add_collection(
        Poly3DCollection(triangles, alpha=0.3, linewidths=0, facecolors=facecolors)
    )

    _snap_to_bounds(ax, polycrystal)

    if neighbourhood is not None:
        title = f"Voronoi Tesselation of Grain {neighbourhood} and Neighbours"
    else:
        title = "Voronoi Tesselation of the Polycrystal"

    title = f"{title}\nColoring by {coloring}"
    ax.set_title(title)

    _xyz_labels(ax)

    return fig, ax


if __name__ == "__main__":
    pass
