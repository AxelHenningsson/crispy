import ImageD11.grain
import numpy as np
import xfab
import xfab.symmetry

import crispy


class Polycrystal:
    """Class to represent a polycrystal.

    Args:
        grainfile (:obj:`str` or :obj:`list` of :obj:`ImageD11.grain.grain`):
            Absolute path to the HDF5 file containing the grain information, or
            alternatively a list of ImageD11.grain.grain objects.
        group_name (:obj:`str`): The name of the group in the HDF5 file containing
            the grain information. Defaults to "grains".
    """

    def __init__(self, grainfile, group_name="grains"):
        if isinstance(grainfile, list):
            self._grains = grainfile
        else:
            self._grains = crispy.read.grains(grainfile, group_name)
        self._mesh = None

    @classmethod
    def from_array(
        cls,
        translations,
        ubi_matrices,
    ):
        if len(translations) != len(ubi_matrices):
            raise ValueError("translations and ubi_matrices must have the same length")
        grains = [ImageD11.grain.grain(ubi) for ubi in ubi_matrices]
        for g, t in zip(grains, translations):
            g.translation = t
        return cls(grains)

    def tesselate(self):
        """Perform a voronoi tesselation using crispy.tesselate.voronoi

        Generates the polycrystal mesh with one convex polyhedra per grain.
        """
        self._mesh = crispy.tesselate.voronoi(self._grains)

    def texturize(self, crystal_system):
        """Compute the misorientation between all grain neighbours.

        NOTE: uses xfab.symmetry.Umis.

        Args:
            crystal_system (:obj:int): crystal_system number must be one of 1: Triclinic, 2: Monoclinic,
                3: Orthorhombic, 4: Tetragonal, 5: Trigonal, 6: Hexagonal, 7: Cubic
        """
        self.misorientations = np.empty((len(self._grains),), dtype=np.ndarray)
        for gi in range(len(self._grains)):
            u = self._grains[gi].u
            self.misorientations[gi] = np.zeros((len(self._mesh.neighbours[gi]),))
            for i, ni in enumerate(self._mesh.neighbours[gi]):
                mis = xfab.symmetry.Umis(u, self._grains[ni].u, crystal_system)
                self.misorientations[gi][i] = np.min(mis[:, 1])

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
        if self._mesh is None:
            raise ValueError(
                "Mesh is None, did you forget to call the tesselate() method?"
            )
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
    import cProfile
    import os
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    pc = crispy.Polycrystal(
        os.path.join(crispy.assets._asset_path, "FeAu_0p5_tR_ff1_grains.h5"),
        group_name="Fe",
    )

    pc.tesselate()

    path = os.path.join(crispy.assets._root_path, "sandbox/FeAu_neigh_1.vtk")
    pc.write(path, neighbourhood=3)

    ngrains = 1000
    translations = np.random.rand(ngrains, 3)
    translations[:, 2] *= 2
    from scipy.spatial.transform import Rotation

    ubi_matrices = Rotation.random(ngrains).as_matrix()
    pc = crispy.Polycrystal.from_array(translations, ubi_matrices)
    pc.tesselate()
    pc.texturize(crystal_system=7)

    path = os.path.join(crispy.assets._root_path, "sandbox/random_poly_cryst.vtk")

    pc.write(path, grains="all")

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")
    print("\n\nCPU time is : ", t2 - t1, "s")
