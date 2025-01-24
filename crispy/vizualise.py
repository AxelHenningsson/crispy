import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import crispy


def _crispy_styling():
    fontsize = 18
    ticksize = 18
    plt.style.use("dark_background")
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.rcParams["font.family"] = "Times New Roman"


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


def centroids(polycrystal):
    """Plot the centroids of the grains in the polycrystal.

    Returns:
        :obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`
    """
    _crispy_styling()
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(9, 7))
    x, y, z = polycrystal.centroids.T
    if hasattr(polycrystal.grains[0], 'rgb'):
        color = [g.rgb[:,0] for g in polycrystal.grains]
        ax.scatter3D(x, y, z, c=color, s=50)
        axis = str(polycrystal._ipf_axes[0].round(3)).replace("  ", ",").replace("[", "").replace("]", "")
        coloring = "ipf (view-axis x,y,z = {})".format(axis)
    else:
        ax.scatter3D(x, y, z, c=range(len(x)), cmap="rainbow", s=50)
        coloring = "grain id"
    title = "Centroids of the Polycrystal"
    title = f"{title}\nColoring by {coloring}"
    ax.set_title(title)
    _xyz_labels(ax)
    _snap_to_bounds(ax, polycrystal)
    return fig, ax


def mesh(
    polycrystal, neighbourhood=None
):  # TODO: add args for surface/interiror grains, colorings, etc.
    """Plot the Voronoi tesselation of the polycrystal.

    Args:
        polycrystal (:obj:`crispy.Polycrystal`): The polycrystal object.
        neighbourhood (:obj:`int`, optional): When not None, the grain
            with number -neighbourhood- and all of its neighbours are
            plotted. Default is None, in which case the entire polycrystal
            mesh is plotted.

    Returns:
        :obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`
    """
    assert polycrystal._mesh is not None, (
        "Mesh not yet generated. Call polycrystal.tesselate() first."
    )

    _crispy_styling()

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d"), figsize=(9, 7))

    if neighbourhood is not None:
        grains_index = polycrystal._select_grains(None, neighbourhood)
        local_geometry = polycrystal._extract_geom(grains_index)
        mesh = crispy.tesselate._build_mesh(*local_geometry)
        grains = polycrystal.grains[grains_index]
    else:
        mesh = polycrystal._mesh
        grains = polycrystal.grains

    if hasattr(grains[0], 'rgb'):
        facecolors = []
        for i in range(len(mesh.cells_dict["polygon"])):
            gid = mesh.cell_data["grain_id"][0][i]
            facecolors.append( tuple(polycrystal.grains[gid].rgb[:, 0]) + (1,) )
        axis = str(polycrystal._ipf_axes[0].round(3)).replace(" ", ",").replace("[", "").replace("]", "")
        coloring = "ipf (view-axis x,y,z = {})".format(axis)
    else:
        cmap = cm.rainbow
        norm = Normalize(vmin=0, vmax=polycrystal.number_of_grains)
        facecolors = [cmap(norm(mesh.cell_data["grain_id"][0][i])) for i in range(len(mesh.cells_dict["polygon"]))]
        coloring = "grain id"

    triangles = [mesh.points[poly] for poly in mesh.cells_dict["polygon"]]


    ax.add_collection(Poly3DCollection(triangles, alpha=0.3, linewidths=0, facecolors=facecolors))

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
    import os

    import crispy

    pc = crispy.Polycrystal(
        os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
        group_name="Fe",
        lattice_parameters=  [4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0] ,
        symmetry=225
    )

    pc.tesselate()

    fig, ax = mesh(pc, neighbourhood=2)
    plt.show()

    fig, ax = centroids(pc)
    plt.show()
    fig, ax = centroids(pc)
    plt.show()
    plt.show()
