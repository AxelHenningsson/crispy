import numpy as np
from orix.plot import IPFColorKeyTSL
from orix.vector.vector3d import Vector3d

from crispy._constants import _SPACEGROUP_TO_CRYSTAL_SYSTEM


class Polycrystal:
    """Meta class to represent a polycrystal.

    This class is a meta class for creating polycrystal objects. Examples of
    incarnations of this class are TDXRDMap and LabDCTVolume. To create a
    polycrystal object, use the factory class crispy._grain_map.GrainMap which
    will dispatch to the correct class depending on the input data.

    Common to all polycrystal objects is the attribute `grains`, which is a
    list of ImageD11.grain.grain objects representing the grains in the
    polycrystal. Moeover a polycrystal object always hold a reference cell,
    which is a unitcell object describing the lattice parameters and symmetry
    group of the material phase.

    Any polycrystal object can be mounted on a crispy.dfxm.Goniometer object to
    compute available reciprocal space vectors and the corresponding
    diffraction angles for DFXM experiments.

    Attributes:
        reference_cell (:obj:`ImageD11.unitcell`): The reference cell of the
            polycrystal. This is a unitcell object describing the lattice
            parameters and symmetry group of the material phase.
        grains (:obj:`list` of :obj:`ImageD11.grain.grain`): A list of
            ImageD11.grain.grain objects representing the grains in the
            polycrystal.
        number_of_grains (:obj:`int`): The number of grains in the
            polycrystal.
        _misorientations (:obj:`numpy array`): The misorientations between
            all grain neighbours. This is a 2D array of shape (N, ...) where N
            is the number of grains in the polycrystal. i.e the misorientations
            between grain i and all of its neighbours are stored in
            misorientations[i]. The shape of the second dimension depends
            on the number of neighbours of grain i.
        neighbours (:obj:`numpy array`): The neighbours of each grain in the
            polycrystal. This is a 2D array of shape (N, M) where N is the
            number of grains in the polycrystal and M is the number of
            neighbours of each grain. i.e the neighbours of grain i are stored
            in neighbours[i]. with M being the number of neighbours of grain i,
            -- a variable number of neighbours is allowed.

    """

    def __init__(self):
        self.reference_cell = None
        self.grains = None
        self.number_of_grains = None
        self._misorientations = None
        self.neighbours = None

    def texturize(self):
        """Compute the misorientations between all grain neighbours."""
        pass

    @property
    def misorientations(self):
        """Return the misorientations between all grain neighbours."""
        if self._misorientations is None:
            raise ValueError(
                "Misorientations are None, did you forget to call the texturize() method?"
            )
        return self._misorientations.copy()

    @property
    def ubi(self):
        """Return all the UBI matrices of the grains in the polycrystal. shape=(N, 3, 3)"""
        return np.array([g.ubi for g in self.grains])

    @property
    def u(self):
        """Return all the U matrices of the grains in the polycrystal. shape=(N, 3, 3)"""
        return np.array([g.u for g in self.grains])

    @property
    def orientation(self):  # alias for u
        return self.u

    @property
    def b(self):
        """Return all the B matrices of the grains in the polycrystal. shape=(N, 3, 3)"""
        return np.array([g.b for g in self.grains])

    @property
    def strain(self):
        """Return all the sample strain tensors of the grains in the polycrystal. shape=(N, 3, 3)"""
        dzero_cell = self.reference_cell.lattice_parameters
        return np.array([g.eps_sample_matrix(dzero_cell) for g in self.grains])

    @property
    def crystal_system(self):
        """Return the crystal system of the polycrystal as as a string

        i.e : triclinic, monoclinic, orthorhombic, tetragonal,
            trigonal, hexagonal or cubic

        """
        return _SPACEGROUP_TO_CRYSTAL_SYSTEM[self.reference_cell.symmetry]

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


if __name__ == "__main__":
    pass
