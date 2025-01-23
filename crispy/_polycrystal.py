import ImageD11.grain
import numpy as np
import xfab
import xfab.symmetry
from orix.plot import IPFColorKeyTSL
from orix.vector.vector3d import Vector3d
import crispy

class Polycrystal:
    """Class to represent a polycrystal.

    Args:
        grainfile (:obj:`str` or :obj:`list` of :obj:`ImageD11.grain.grain`):
            Absolute path to the HDF5 file containing the grain information, or
            alternatively a list of ImageD11.grain.grain objects.
        group_name (:obj:`str`): The name of the group in the HDF5 file containing
            the grain information. Defaults to "grains".


    Attributes:
        grains (:obj:`list` of :obj:`ImageD11.grain.grain`): List of grains in the polycrystal.
        number_of_grains (:obj:`int`): Number of grains in the polycrystal.
        bounding_box (:obj:`numpy array`): The bounding box of the polycrystal. Shape=(3,2).
        extent (:obj:`numpy array`): The extent of the polycrystal. Shape=(3,).
        centroids (:obj:`numpy array`): All the grain centroid postions in the polycrystal. Shape=(N, 3).
        u (:obj:`numpy array`): All the U matrices of the grains in the polycrystal. Shape=(N, 3, 3).
        orientations (:obj:`numpy array`): Alias for u.
        b (:obj:`numpy array`): All the B matrices of the grains in the polycrystal. Shape=(N, 3, 3).
        neighbours (:obj:`numpy array`): The neighbours of each grain in the polycrystal. shape=(N,).
            each sub-array contains the indices of the neighbours of the corresponding grain.
            is None before calling the tesselate() method.
        misorientations (:obj:`numpy array`): The misorientations between all grain neighbours. shape=(N,).
            each sub-array contains the misorientations between the corresponding grain and its neighbours.
            is None before calling the texturize() method.
    """

    def __init__(
        self, grainfile, group_name="grains", lattice_parameters=None, symmetry=None
    ):
        if isinstance(grainfile, list):
            self.grains = grainfile
        else:
            self.grains = crispy.read.grains(grainfile, group_name)

        if lattice_parameters and symmetry:
            self.reference_cell = ImageD11.unitcell.unitcell(
                lattice_parameters, symmetry
            )
        else:
            try:
                self.reference_cell = self.grains[0].ref_cell
            except:
                raise ValueError(
                    "No reference cell parameters passed and the grains do not contain a ref_cell attribute"
                )

        self.number_of_grains = len(self.grains)

        self._mesh = None
        self._misorientations = None

    @property
    def neighbours(self):
        """Return the neighbours of each grain in the polycrystal."""
        if self._mesh is None:
            raise ValueError(
                "Mesh is None, did you forget to call the tesselate() method?"
            )
        return self._mesh.neighbours.copy()

    @property
    def misorientations(self):
        """Return the misorientations between all grain neighbours."""
        if self._misorientations is None:
            raise ValueError(
                "Misorientations are None, did you forget to call the texturize() method?"
            )
        return self._misorientations.copy()

    @property
    def crystal_system(self):
        """Return the crystal system of the polycrystal as as a string

        i.e : triclinic, monoclinic, orthorhombic, tetragonal,
            trigonal, hexagonal or cubic

        """
        return crispy.CONSTANTS._SPACEGROUP_TO_CRYSTAL_SYSTEM[self.reference_cell.symmetry]

    @property
    def bounding_box(self):
        """Return the bounding box of the polycrystal, shape=(3,2)

        Format is: [ [min_x, max_x], [min_y, max_y], [min_z, max_z] ].

        computation is done by taking the min and max of the centroids.
        """
        x, y, z = self.centroids.T
        return np.array([[x.min(), x.max()], [y.min(), y.max()], [z.min(), z.max()]])

    @property
    def extent(self):
        """Return the extent of the polycrystal.

        Format is: [ xwidth, ywidth, zwidth ].

        computation is done by taking the min and max of the centroids.
        """
        bounds = self.bounding_box
        return bounds[:, 1] - bounds[:, 0]

    @property
    def centroids(self):
        """Return all the grain centroid postions in the polycrystal. shape=(N, 3)"""
        return np.array([g.translation for g in self.grains])

    @property
    def ubis(self):
        """Return all the UBI matrices of the grains in the polycrystal. shape=(N, 3, 3)"""
        return np.array([g.ubi for g in self.grains])

    @property
    def u(self):
        """Return all the U matrices of the grains in the polycrystal. shape=(N, 3, 3)"""
        return np.array([g.u for g in self.grains])

    @property
    def orientations(self):  # alias for u
        return self.u

    @property
    def b(self):
        """Return all the B matrices of the grains in the polycrystal. shape=(N, 3, 3)"""
        return np.array([g.b for g in self.grains])

    @classmethod
    def from_array(
        cls,
        translations,
        ubi_matrices,
    ):
        """Instantiate the polycrystal from pure translations and ubi matrices.

        This circumvents the need to read a grain file and ImageD11.grain.grain objects.

        Args:
            translations (:obj: `iterable` of :obj:`numpy.ndarray`): 3D grain center translations each of shape=(3,).
            ubi_matrices (:obj: `iterable` of :obj:`numpy.ndarray`): ubi matrices each of shape=(3, 3).

        Returns:
            :obj:`crispy.Polycrystal`: The polycrystal object.
        """
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
        self._mesh = crispy.tesselate.voronoi(self.grains)

    def texturize(self):
        """Compute the misorientation between all grain neighbours.

        NOTE: uses xfab.symmetry.Umis.

        This function sets the misorientations attribute of the polycrystal object,
        such that self.misorientations[i] is a shape=(n,) array of misorientations
        between grain number i and its n neighbours. Each grain can have a different
        number of neighbours. The misorientation between grain i and its j-th neighbour
        is always the minimum misorientation between all possible orientations taking
        into account the crystal system symmetry.

        Args:
            crystal_system (:obj:str): crystal_system must be one of triclinic,
                monoclinic, orthorhombic, tetragonal, trigonal, hexagonal, cubic
        """
        _crystal_system = crispy.CONSTANTS._CRYSTAL_SYSTEM_STR_TO_INT[self.crystal_system]
        self._misorientations = np.empty((len(self.grains),), dtype=np.ndarray)
        for gi in range(len(self.grains)):
            u = self.grains[gi].u
            self._misorientations[gi] = np.zeros((len(self._mesh.neighbours[gi]),))
            for i, ni in enumerate(self._mesh.neighbours[gi]):
                mis = xfab.symmetry.Umis(u, self.grains[ni].u, _crystal_system)
                self._misorientations[gi][i] = np.min(mis[:, 1])

    def colorize(self, ipf_axes):
        """Compute the IPF colors for the polycrystal using orix (https://github.com/pyxem/orix)

        The implementation is based on the ImageD11 interpretation of orix.

        sets a :obj:`numpy array`: of RGB values for each grain in the polycrystal. Shape=(3, k).
            self.grain.rgb[:, k] is the RGB value of grain when viewed along the k-th axis.

        Args:
            axes (:obj:`numpy array`): The viewing axes to compute the IPF colors for, default is the
                x, y, and z axes, i.e np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]). axes[i] is the i-th
                viewing axis.
        """
        rgb = self._ipf_colors(ipf_axes)
        self._ipf_axes = ipf_axes
        for i in range(self.number_of_grains):
            self.grains[i].rgb = rgb[i]

    def _ipf_colors(self, axes=np.eye(3)):
        """Compute the IPF colors for the polycrystal using orix (https://github.com/pyxem/orix)

        The implementation is based on the ImageD11 interpretation of orix.

        Args:
            axes (:obj:`numpy array`): The viewing axes to compute the IPF colors for, default is the
                x, y, and z axes, i.e np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]). axes[i] is the i-th
                viewing axis.

        Returns:
            :obj:`numpy array`: The RGB values for each grain in the polycrystal. Shape=(N, 3, 3).
                RGB[i, :, k] is the RGB value of grain i when viewed along the k-th axis.
        """
        rgb = np.zeros((self.number_of_grains, 3, axes.shape[0]))
        point_group = self.reference_cell.orix_phase.point_group
        orix_orien = self.reference_cell.get_orix_orien(self.u)
        for i, axis in enumerate(axes):
            ipfkey = IPFColorKeyTSL(point_group, direction=Vector3d(axis))
            rgb[:, :, i] = ipfkey.orientation2color(orix_orien)
        return rgb

    def write(self, path, grains="all", neighbourhood=None):
        """Write the polycrystal to a file readable by Paraview.

        Args:
            path (:obj:`str`): Absolute path to the file to write the polycrystal
                to, must end in .vtk.
            grains (:obj:`list` of :obj:`int` or :obj:`int` or :obj:`str`): The
                grain ids to write. Defaults to "all" in which case all grains
                in the polycrystal are written.
            neighbourhood (:obj:`int`): When not None, the grains argument is overwritten
                such that the grain number -neighbourhood- and all of its neighbours are
                written to file. Default is None.

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
    pc.texturize(crystal_system="cubic")

    path = os.path.join(crispy.assets._root_path, "sandbox/random_poly_cryst.vtk")

    pc.write(path, grains="all")

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")
    print("\n\nCPU time is : ", t2 - t1, "s")
    print("\n\nCPU time is : ", t2 - t1, "s")
    print("\n\nCPU time is : ", t2 - t1, "s")
    print("\n\nCPU time is : ", t2 - t1, "s")
