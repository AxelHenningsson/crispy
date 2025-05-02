import numpy as np
from orix.plot import IPFColorKeyTSL
from orix.vector.vector3d import Vector3d

from ._constants import CONSTANTS


class Polycrystal:
    """Base class for representing polycrystalline materials.

    This class serves as a base for specialized polycrystal implementations like
    :class:`crispy.TDXRDMap <crispy._tdxrd_map.TDXRDMap>` and
    :class:`crispy.LabDCTVolume <crispy._lab_dct_volume.LabDCTVolume>`. To create a
    polycrystal object, use the :class:`crispy.GrainMap <crispy._grain_map.GrainMap>`
    factory class which dispatches to the appropriate implementation.

    All polycrystal objects share common attributes including a list of
    :obj:`ImageD11.grain.grain` objects representing individual grains and a reference
    cell describing the material's lattice parameters and symmetry group.

    Any polycrystal object can be mounted on a :obj:`crispy.dfxm.Goniometer` to analyze
    accessible diffraction reflections for Dark-Field X-ray Microscopy experiments.

    Attributes:
        reference_cell (:obj:`ImageD11.unitcell`): Reference cell describing the
            material's lattice parameters and symmetry group.
        grains (:obj:`list` of :obj:`ImageD11.grain.grain`): List of grain objects
            representing individual crystals in the polycrystal.
        number_of_grains (:obj:`int`): Total number of grains in the polycrystal.
        neighbours (:obj:`numpy.ndarray`): Array of ``shape (N, ...)`` containing grain
            neighbour indices where ``N`` is the number of grains and ``...`` depends on the
            number of neighbours per grain. I.e., if grain ``i`` has 3 neighbours, then
            ``neighbours[i, :]`` contains the indices of the 3 neighbours.
    """

    def __init__(self):
        self.reference_cell = None
        self.grains = None
        self.number_of_grains = None
        self._misorientations = None
        self.neighbours = None

    def texturize(self):
        """Compute the :attr:`misorientations` between all grain neighbours.

        Calculates and stores misorientation angles between adjacent grains in the
        polycrystal. This is an in-place operation that mutates the state of the
        :class:`crispy.Polycrystal <crispy._polycrystal.Polycrystal>` object such
        that the :attr:`misorientations` attribute is set.
        """
        pass

    @property
    def misorientations(self):
        """Return the misorientations between all grain neighbours.

        Returns:
            :obj:`numpy.ndarray`: Array of misorientation angles between adjacent grains.
                ``shape=(N, ...)`` where ``N`` is the number of grains and the
                second dimension, ``...``, depends on the number of neighbours per grain.
                I.e., if grain ``i`` has 3 neighbours, then ``misorientations[i, :]``
                contains the misorientation angles between grain ``i`` and its
                3 neighbours following the order of the attribute :attr:`neighbours`.

        Raises:
            ValueError: If misorientations have not been computed via texturize().
        """
        if self._misorientations is None:
            raise ValueError(
                "Misorientations are None, did you forget to call texturize()?"
            )
        return self._misorientations.copy()

    @property
    def ubi(self):
        """Return the UBI matrices (sample frame reciprocal unit cell matrices) of all grains.

        Returns:
            :obj:`numpy.ndarray`: Array of UBI matrices
                with ``shape=(N, 3, 3)``, where ``N`` is the number of grains.
        """
        return np.array([g.ubi for g in self.grains])

    @property
    def u(self):
        """Return the U matrices (orientation matrices) of all grains.

        Returns:
            :obj:`numpy.ndarray`: Array of U matrices
                with ``shape=(N, 3, 3)``, where ``N`` is the number of grains.
        """
        return np.array([g.u for g in self.grains])

    @property
    def orientation(self):
        """Syntax sugar for :attr:`u`.

        Returns:
            :obj:`numpy.ndarray`: Array of U matrices
                with ``shape=(N, 3, 3)``, where ``N`` is the number of grains.
        """
        return self.u

    @property
    def b(self):
        """Return the B matrices (crystal frame reciprocal unit cell matrices) of all grains.

        Returns:
            :obj:`numpy.ndarray`: Array of B matrices
                with ``shape=(N, 3, 3)``, where ``N`` is the number of grains.
        """
        return np.array([g.b for g in self.grains])

    @property
    def strain(self):
        """Return the sample frame strain tensors of all grains.

        Returns:
            :obj:`numpy.ndarray`: Array of strain tensors
                with ``shape=(N, 3, 3)``, where ``N`` is the number of grains.
        """
        dzero_cell = self.reference_cell.lattice_parameters
        return np.array([g.eps_sample_matrix(dzero_cell) for g in self.grains])

    @property
    def crystal_system(self):
        """Return the crystal system name.

        Returns:
            :obj:`str`: One of: triclinic, monoclinic, orthorhombic, tetragonal,
                trigonal, hexagonal, or cubic.
        """
        return CONSTANTS.SPACEGROUP_TO_CRYSTAL_SYSTEM[self.reference_cell.symmetry]

    def _ipf_colors(self, axes=np.eye(3)):
        """Compute inverse pole figure (IPF) colors.

        Uses the ImageD11 interpretation of orix (https://github.com/pyxem/orix) to
        calculate IPF colors for specified viewing directions.

        Args:
            axes (:obj:`numpy.ndarray`): View directions to compute IPF colors.
                Default is identity matrix representing ``x``, ``y``, ``z`` axes.
                ``shape=(k, 3)`` where ``k`` is the number of directions.

        Returns:
            :obj:`numpy.ndarray`: RGB color values with ``shape=(N, 3, k)``,
                where ``N`` is the number of grains and ``k`` is the number of view
                directions. ``RGB[i,:,j]`` gives colors for grain ``i`` viewed
                along axis ``j``.
        """
        rgb = np.zeros((self.number_of_grains, 3, axes.shape[0]))
        point_group = self.reference_cell.orix_phase.point_group
        orix_orien = self.reference_cell.get_orix_orien(self.u)
        for i, axis in enumerate(axes):
            ipfkey = IPFColorKeyTSL(point_group, direction=Vector3d(axis))
            rgb[:, :, i] = ipfkey.orientation2color(orix_orien)
        return rgb

    def colorize(self, ipf_axes):
        """Compute and store IPF colors for all grains.

        Uses orix (https://github.com/pyxem/orix) to calculate inverse pole figure
        colors for specified viewing directions and stores them in grain objects
        in the :attr:`grains` attribute under the `rgb` attribute.

        Example:

        .. code-block:: python

            import crispy
            import numpy as np

            polycrystal = crispy.assets.grain_map.tdxrd_map()

            # Colorize the polycrystal with the y, and z axes
            polycrystal.colorize(np.array([[0, 1, 0], [0, 0, 1]]))

            # these can be accessed as follows
            grain = polycrystal.grains[0]
            ipf_rgb_color_x_view = grain.rgb[:, 0]
            ipf_rgb_color_y_view = grain.rgb[:, 1]


        Args:
            ipf_axes (:obj:`numpy.ndarray`): Viewing directions to compute IPF
                colors for. ``shape=(k, 3)`` where ``k`` is the number of directions.
        """
        rgb = self._ipf_colors(ipf_axes)
        self._ipf_axes = ipf_axes
        for i in range(self.number_of_grains):
            self.grains[i].rgb = rgb[i]

    def write(self, file_path, *args, **kwargs):
        """Write the polycrystal to a file readable by Paraview.

        The grain volume will be written to the specified file path,
        the file format is ``.xdmf`` or ``.vtk``. The appropriate extension will
        automatically be added if it is not already present.

        Args:
            file_path (:obj:`str`): Absolute path to the file to write the polycrystal
                to, must end in ``.xdmf`` or ``.vtk``.
        """
        pass


if __name__ == "__main__":
    pass
