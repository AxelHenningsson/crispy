import numpy as np

import crispy


class Polycrystal:
    """Class to represent a polycrystal.

    Args:
        grainfile (:obj:`str`): Absolute path to the HDF5 file containing the grain
            information.
        group_name (:obj:`str`): The name of the group in the HDF5 file containing
            the grain information. Defaults to "grains".
    """

    def __init__(self, grainfile, group_name="grains"):
        self._grains = crispy.read.grains(grainfile, group_name)
        self._mesh = crispy.tesselate.voronoi(self._grains)

    def write(self, path, grains="all", neighbourhood=None):
        """Write the polycrystal to a file readable by Paraview.

        Args:
            path (:obj:`str`): Absolute path to the file to write the polycrystal
                to, must end in .vtk.
            grains (:obj:`list` of :obj:`int` or :obj:`int` or :obj:`str`): The
                grain ids to write. Defaults to "all" in which case all grains
                in the polycyrtsal are written.
            neighbourhood (:obj:`int`): Integer specifying the grain id to write and all
                of its neighbours. Default is None. in which case the grains specified
                via the grains argument are written.

        """
        if neighbourhood is None and isinstance(grains, str) and grains == "all":
            self._mesh.write(path)
        else:
            grains = self._select_grains(grains, neighbourhood)
            local_geometry = self._extract_geom(grains)
            mesh = crispy.tesselate._build_mesh(*local_geometry)
            mesh.write(path)

    def _select_grains(self, grains, neighbourhood):
        """Patch the grain index list to include neighbours, handle int, etc."""
        if isinstance(neighbourhood, int):
            i = neighbourhood
            grains = np.concatenate((self._mesh.neighbours[i], [i]))
        elif isinstance(grains, int):
            grains = [grains]
        return grains

    def _extract_geom(self, grains):
        """Rebuild the mesh geometry arrays kepping only the specified grains."""
        vertices = []
        simplices = []
        grain_id = []
        surface_grain = []

        for i in grains:
            mask = self._mesh.cell_data["grain_id"][0] == i

            simp = self._mesh.cells_dict["polygon"][mask]
            vert = self._mesh.points[np.unique(simp)]

            simp -= np.min(simp)
            simp += len(vertices)

            vertices.extend(list(vert))
            simplices.extend(list(simp))
            grain_id.extend([i] * len(simp))
            surface_grain.extend(self._mesh.cell_data["surface_grain"][0][mask])

        return vertices, simplices, grain_id, surface_grain


if __name__ == "__main__":
    import os

    pc = crispy.Polycrystal(
        os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
        group_name="Fe",
    )

    path = os.path.join(crispy.assets._root_path, "sandbox/FeAu_neigh_1.vtk")
    pc.write(path, neighbourhood=3)
