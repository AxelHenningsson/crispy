import os

import h5py
import ImageD11
import meshio
import numba
import numpy as np
import xfab
import xfab.symmetry
from scipy.spatial.transform import Rotation

from crispy._constants import _CRYSTAL_SYSTEM_STR_TO_INT
from crispy._polycrystal import Polycrystal


class LabDCTVolume(Polycrystal):
    """Class to represent a voxelated lab-dct grain volume.

    This class is used to represent a voxelated grain volume in the lab
    frame. The class is used to read the grain volume data from a h5
    file in the lab-dct format and to provide methods to manipulate the
    grain volume data. This class is a specific implementation of the
    Polycrystal class and can be interfaced with the crispy.dfxm.Goniometer
    to search for dfxm reflections. This may be usefull in
    preperation for synchrotron measurements to determine what type of
    reflections and orientation that will be available in the sample.

    Note that the lab-dct (as implemented by xnovo) coordinate system aligns
    with the ESR ID11 and ID03 coordinate systems in that the z axis is
    pointing towards the ceiling, the x axis is pointing along the beam
    photon propagation direction and the y axis is pointing to the left
    as seen by a traveling photon. Loading a lab dct map with LabDCTVolume
    will unify the coordinate system to be the same as that of crispy.TDXRDMap.

    Args:
        file_path (str): The path to the h5 file containing the grain
            volume data. The file should be in the lab-dct format.
            The file should contain the following datasets:
            - LabDCT/Spacing: The voxel size in microns.
            - LabDCT/Data/GrainId: The grain id for each voxel.
            - LabDCT/Data/Rodrigues: The Rodrigues vector for each voxel.
            - PhaseInfo/Phase01/UnitCell: The unit cell parameters.
            - PhaseInfo/Phase01/SpaceGroup: The space group of the phase.

    Attributes:
        reference_cell (ImageD11.unitcell): The reference cell of the
            grain volume.
        voxel_size (float): The voxel size in microns.
        labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
            axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
            increase from low to high along the positive axis direction.
        X (np.ndarray): The x coordinates of the voxels in microns. shape=(m,n,o)
        Y (np.ndarray): The y coordinates of the voxels in microns. shape=(m,n,o)
        Z (np.ndarray): The z coordinates of the voxels in microns. shape=(m,n,o)
        orientations (np.ndarray): The orientations of the grains in
            Rodrigues vector format. shape=(number_of_grains, 3, 3)
        B0 (np.ndarray): The B0 matrix for the grain volume. I.e the
            recirprocal lattice vectors in the lab frame (upper triangular
            3,3 matrix).
        grains (np.ndarray): The grains in the grain volume. len=number_of_grains
        number_of_grains (int): The number of grains in the grain volume.

    """

    def __init__(self, file_path):
        if not os.path.exists(file_path) or file_path.endswith(".h5") is False:
            raise ValueError("File path is not valid or does not end with .h5")

        self.reference_cell = self._get_reference_cell(file_path)
        self.B0 = self._get_B0(self.reference_cell)

        self.voxel_size = self._get_voxel_size(file_path)
        self.labels = self._get_grainid(file_path)
        self.Z, self.Y, self.X = self._get_voxel_grid(self.voxel_size, self.labels)
        self.rodrigues = self._get_rodrigues(file_path)
        self._misorientations = None

        self._update_state()

    def _update_state(self):
        """Update the state of the grain volume after a crop or filter.

        This will update the labels, orientations, grains, number of
        grains, grain sizes and bounding box of the grain volume.
        The labels will be reset to start from -1,0,1,2,...,n. where
        -1 is the void label. The orientations will be updated to
        reflect the new label etc.
        """
        self._reset_labels()
        self.orientations = self._get_orientations(self.rodrigues, self.labels)
        self.grains = self._get_grains(self.B0, self.orientations, self.reference_cell)
        self.number_of_grains = len(self.grains)
        self.grain_sizes = self._get_grain_sizes(self.number_of_grains, self.labels)
        self.bounding_box = self._get_bounding_box(self.X, self.Y, self.Z)
        self.neighbours = self._get_neighbours(self.labels, self.number_of_grains)
        if self._misorientations is not None:
            self.texturize()

    def _get_neighbours(self, labels, number_of_grains, struct=None):
        """Compute neighbours of each grian label in the grain volume.

        By default, neighbourhood is defined though the 3D structuring element:
            struct = np.array(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ],
                dtype=uint8,
            )

        Edge voxels are not included in the neighbourhood calculation.

        Args:
            labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
                axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
                increase from low to high along the positive axis direction.
            number_of_grains (int): The number of grains in the grain volume.
                The number of grains is the maximum label id + 1.
                The number of grains is the number of unique labels in the
                labels array.
            struct (np.ndarray): The structuring element to use for the
                neighbourhood. The structuring element should be a 3D
                boolean array with the same shape as the labels array.
                The default structuring element is the one defined above.
                Struct shape must be odd in all dimensions.
        Returns:
            neighbours (np.ndarray): The neighbours of each grain in the
                grain volume. shape=(number_of_grains, number_of_grains)
                The neighbours are defined as the labels of the grains that
                are connected to the grain with the same label.
        """
        if struct is None:
            struct = np.array(
                [
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                    [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
                    [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            )
        assert isinstance(struct, np.ndarray), (
            "Structuring element must be a numpy array"
        )
        assert struct.ndim == 3, "Structuring element must be 3D"
        assert (
            struct.shape[0] % 2 == 1
            and struct.shape[1] % 2 == 1
            and struct.shape[2] % 2 == 1
        ), "Structuring element must have odd shape in all dimensions"

        _neighbours = np.zeros((number_of_grains, number_of_grains), dtype=np.int32) - 1
        counters = LabDCTVolume._neighwalk(labels, _neighbours, struct)

        neighbours = np.empty((number_of_grains,), dtype=np.ndarray)
        for i in range(_neighbours.shape[0]):
            neighbours[i] = _neighbours[i, : counters[i]]

        return neighbours

    @staticmethod
    @numba.njit(parallel=True)
    def _neighwalk(labels, neighbours, struct):
        """Walk a 3D labeled array and find the neighbours of each label.

        Neighbours are defined though the 3D structuring element struct.

        This is a parallelized and compiled function that is used internally
        to speed up the computation of the neighbours of each while avoiding
        any bloated memory usage.

        Args:
            labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
                axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
                increase from low to high along the positive axis direction.
            neighbours (np.ndarray): The neighbours of each grain in the
                grain volume. shape=(number_of_grains, number_of_grains).
            struct (np.ndarray): The structuring element to use for the
                neighbourhood. The structuring element should be a 3D
                uint8 array of 1s and 0s.

        Returns:
            counters (np.ndarray): The number of neighbours for each grain.
                shape=(number_of_grains,).
        """
        mask = labels > -1
        m, n, o = struct.shape[0] // 2, struct.shape[1] // 2, struct.shape[2] // 2
        counters = np.zeros((neighbours.shape[0],), dtype=np.int32)
        for i in numba.prange(1, labels.shape[0] - 1):
            for j in range(1, labels.shape[1] - 1):
                for k in range(1, labels.shape[2] - 1):
                    if mask[i, j, k]:
                        label = labels[i, j, k]
                        for ii in range(struct.shape[0]):
                            for jj in range(struct.shape[1]):
                                for kk in range(struct.shape[2]):
                                    if (
                                        struct[ii, jj, kk]
                                        and mask[
                                            i + ii - m,
                                            j + jj - n,
                                            k + kk - o,
                                        ]
                                    ):
                                        nlabel = labels[
                                            i + ii - m,
                                            j + jj - n,
                                            k + kk - o,
                                        ]
                                        if (
                                            nlabel != label
                                            and nlabel not in neighbours[label]
                                        ):
                                            neighbours[label, counters[label]] = nlabel
                                            counters[label] += 1
        return counters

    def write(self, file_path):
        """Write the grain volume to paraview readable formats.

        The grain volume will be written to the specified file path,
        example of formats are .vtk and .xdmf.

        The resulting point data will contain the following attributes:
            - spacing_in_um: The voxel size in microns.
            - ipf_x: The x component of the IPF color.
            - ipf_y: The y component of the IPF color.
            - ipf_z: The z component of the IPF color.
            - labels: The grain id for each voxel.
            - grain_sizes: The size of each grain in units of voxels.

        The data format is sparse in the sense that only non-void voxels
        will be written to the file.

        Args:
            file_path (str): The path to the file to write the grain
                volume to. The file should be in the paraview readable
                format. The file should end with .vtk or .xdmf or similar.

        """
        m = self.labels > -1
        coordinates = np.array([self.X[m], self.Y[m], self.Z[m]]).T

        _rgb = self._ipf_colors()
        rgb = np.zeros((coordinates.shape[0], 3, 3))
        for i, gid in enumerate(self.labels[m]):
            rgb[i, :, :] = _rgb[gid, :, :]

        grain_sizes = np.zeros((coordinates.shape[0],))
        for i, gid in enumerate(self.labels[m]):
            grain_sizes[i] = self.grain_sizes[gid]

        cells = [("vertex", np.array([[i] for i in range(coordinates.shape[0])]))]
        meshio.Mesh(
            coordinates,
            cells,
            point_data={
                "spacing_in_um": np.ones((coordinates.shape[0],)) * self.voxel_size,
                "ipf_x": rgb[:, :, 0],
                "ipf_y": rgb[:, :, 1],
                "ipf_z": rgb[:, :, 2],
                "labels": self.labels[m],
                "grain_sizes": grain_sizes,
            },
        ).write(file_path)

    def texturize(self):
        """Compute the misorientation between all grain neighbours.

        NOTE: uses xfab.symmetry.Umis.

        This function sets the misorientations attribute of the polycrystal object,
        such that self.misorientations[i] is a shape=(n,) array of misorientations
        between grain number i and its n neighbours. Each grain can have a different
        number of neighbours. The misorientation between grain i and its j-th neighbour
        is always the minimum misorientation between all possible orientations taking
        into account the crystal system symmetry.

        """
        _crystal_system = _CRYSTAL_SYSTEM_STR_TO_INT[self.crystal_system]
        self._misorientations = np.empty((len(self.grains),), dtype=np.ndarray)
        for gi in range(len(self.grains)):
            u = self.grains[gi].u
            self._misorientations[gi] = np.zeros((len(self.neighbours[gi]),))
            for i, ni in enumerate(self.neighbours[gi]):
                mis = xfab.symmetry.Umis(u, self.grains[ni].u, _crystal_system)
                self._misorientations[gi][i] = np.min(mis[:, 1])

    def _get_bounding_box(self, X, Y, Z):
        """Get the bounding box of the grain volume.

        The bounding box is defined as the minimum and maximum coordinates
        of the voxels in the grain volume. The bounding box is returned
        as a tuple of (xmin, xmax, ymin, ymax, zmin, zmax).

        Args:
            X (np.ndarray): The x coordinates of the voxels in microns.
                shape=(m,n,o)
            Y (np.ndarray): The y coordinates of the voxels in microns.
                shape=(m,n,o)
            Z (np.ndarray): The z coordinates of the voxels in microns.
                shape=(m,n,o)

        Returns:
            tuple: The bounding box of the grain volume.
                (xmin, xmax, ymin, ymax, zmin, zmax)
        """
        return np.array(
            [np.min(X), np.max(X), np.min(Y), np.max(Y), np.min(Z), np.max(Z)]
        )

    def _get_grain_sizes(self, number_of_grains, labels):
        """Get the sizes of the grains in the grain volume in units of voxels.

        The sizes are calculated by counting the number of voxels in each
        grain. The sizes are returned as a numpy array with the same
        shape as the labels array.

        Args:
            number_of_grains (int): The number of grains in the grain volume.
            labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
                axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
                increase from low to high along the positive axis direction.

        Returns:
            np.ndarray: The sizes of the grains in units of voxels.
                shape=(number_of_grains,)
        """
        sizes = np.zeros(number_of_grains, dtype=np.int32)
        return LabDCTVolume._grain_sizes(labels, sizes)

    @staticmethod
    @numba.njit(parallel=True)
    def _grain_sizes(labels, sizes):
        """Parallelized function to calculate the sizes of the grains."""
        for i in numba.prange(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(labels.shape[2]):
                    if labels[i, j, k] != -1:
                        sizes[labels[i, j, k]] += 1
        return sizes

    def filter(self, min_grain_size_in_voxels):
        """Filter the grains based on their size.

        Grains with a size smaller than grain_size will be removed from
        the voxel volume. Attributes will be updated accordingly.

        Args:
            min_grain_size_in_voxels (int): The minimum size of the grains to keep in
                units of voxels.
        """
        LabDCTVolume._filter(
            self.labels,
            self.rodrigues,
            self.grain_sizes,
            min_grain_size_in_voxels,
        )
        self._update_state()

    @staticmethod
    @numba.njit(parallel=True)
    def _filter(labels, rodrigues, sizes, min_grain_size_in_voxels):
        """Parallelized function to filter on grain size."""
        for i in numba.prange(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(labels.shape[2]):
                    if (
                        labels[i, j, k] != -1
                        and sizes[labels[i, j, k]] < min_grain_size_in_voxels
                    ):
                        labels[i, j, k] = -1
                        rodrigues[i, j, k, :] = 0

    def _reset_labels(self):
        """Decrement labels to start from -1,0,1,2,...,n where -1 is the void label."""
        mask = self.labels > -1
        unique_labels = np.unique(self.labels[mask])
        mapping = np.zeros((unique_labels.max() + 1,), dtype=np.int32)
        for new_label, old_label in enumerate(unique_labels):
            mapping[old_label] = new_label
        LabDCTVolume._relabel(self.labels, mapping, mask)

    @staticmethod
    @numba.njit(parallel=True)
    def _relabel(labels, mapping, mask):
        for i in numba.prange(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(labels.shape[2]):
                    if mask[i, j, k]:
                        labels[i, j, k] = mapping[labels[i, j, k]]

    def _get_unique_labels(self):
        """Get the unique labels from the labels array."""
        mask = self.labels > -1
        unique_labels = np.unique(self.labels[mask])
        return unique_labels

    def crop(
        self,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        zmin=None,
        zmax=None,
    ):
        """Crop the voxel volume to the specified bounds.

        The voxel volume will be cropped to the specified bounds. The
        attributes will be updated accordingly pruning any grains that
        are outside the bounds.

        Args:
            xmin (float): The minimum x coordinate of the crop.
            xmax (float): The maximum x coordinate of the crop.
            ymin (float): The minimum y coordinate of the crop.
            ymax (float): The maximum y coordinate of the crop.
            zmin (float): The minimum z coordinate of the crop.
            zmax (float): The maximum z coordinate of the crop.
        """
        if xmin is None:
            xmin = np.min(self.X[0, 0, :])
        if xmax is None:
            xmax = np.max(self.X[0, 0, :])
        if ymin is None:
            ymin = np.min(self.Y[0, :, 0])
        if ymax is None:
            ymax = np.max(self.Y[0, :, 0])
        if zmin is None:
            zmin = np.min(self.Z[:, 0, 0])
        if zmax is None:
            zmax = np.max(self.Z[:, 0, 0])
        if xmin > xmax or ymin > ymax or zmin > zmax:
            raise ValueError(
                "Invalid crop bounds: xmin > xmax or ymin > ymax or zmin > zmax"
            )

        k1, k2 = (
            np.argmin(np.abs(self.X[0, 0, :] - xmin)),
            np.argmin(np.abs(self.X[0, 0, :] - xmax)),
        )
        j1, j2 = (
            np.argmin(np.abs(self.Y[0, :, 0] - ymin)),
            np.argmin(np.abs(self.Y[0, :, 0] - ymax)),
        )
        i1, i2 = (
            np.argmin(np.abs(self.Z[:, 0, 0] - zmin)),
            np.argmin(np.abs(self.Z[:, 0, 0] - zmax)),
        )

        self.labels = self.labels[i1:i2, j1:j2, k1:k2]
        self.X = self.X[i1:i2, j1:j2, k1:k2]
        self.Y = self.Y[i1:i2, j1:j2, k1:k2]
        self.Z = self.Z[i1:i2, j1:j2, k1:k2]
        self.rodrigues = self.rodrigues[i1:i2, j1:j2, k1:k2]

        self._update_state()

    def translate(self, translation):
        """Translate the voxel volume by the specified translation.

        The voxel volume will be translated by the specified translation.
        The attributes will be updated accordingly.

        This is usefull for realigning the voxel volume to be used in
        diffraction simulations or in integration with a synchrotron
        measurement setting.

        Args:
            translation (np.ndarray): The translation vector in units of microns.
                The translation vector should be of shape (3,).
        """
        pass

    def rotate(self, rotation):
        """Rotate the voxel volume by the specified rotation.

        The voxel volume will be rotated by the specified rotation.
        The attributes will be updated accordingly.

        This will update both voxel coordinates and grain orientations.

        This is usefull for realigning the voxel volume to be used in
        diffraction simulations or in integration with a synchrotron
        measurement setting.

        Args:
            rotation (np.ndarray): The rotation matrix. The rotation matrix
                should be of shape (3, 3).
        """
        pass

    def _get_rodrigues(self, file_path):
        """Get the Rodrigues vector from the h5 file."""
        with h5py.File(file_path, "r") as f:
            rodrigues = f["LabDCT/Data/Rodrigues"][...].astype(np.float32)
        return rodrigues

    def _get_grains(self, B0, orientations, reference_cell):
        """Get the grains from the orientations and B0 matrix.

        Args:
            B0 (np.ndarray): The B0 matrix for the grain volume.
                I.e the recirprocal lattice vectors in the lab frame
                (upper triangular 3,3 matrix).
            orientations (np.ndarray): The orientations of the grains
                in Rodrigues vector format. shape=(number_of_grains, 3, 3)
            reference_cell (ImageD11.unitcell): The reference cell of
                the grain volume.

        Returns:
            np.ndarray: The grains in the grain volume. len=number_of_grains
        """
        grains = []
        for u in orientations:
            grains.append(ImageD11.grain.grain(np.linalg.inv(u @ B0)))
            grains[-1].reference_cell = reference_cell
        return np.array(grains)

    def _get_reference_cell(self, file_path):
        """Construct the reference cell from the h5 file.

        NOTE: only one phase is supported at the moment.

        Args:
            file_path (str): The path to the h5 file containing the
                grain volume data. The file should be in the lab-dct format.

        Returns:
            ImageD11.unitcell: The reference cell of the grain volume.
                The reference cell is constructed from the unit cell
                parameters and the space group of the phase.
        """
        with h5py.File(file_path, "r") as f:
            if len(f["PhaseInfo"].keys()) > 1:
                raise NotImplementedError("Multiple phases not implemented yet")
            lattice_parameters = f["PhaseInfo/Phase01/UnitCell"][()]
            symmetry = int(f["PhaseInfo/Phase01/SpaceGroup"][0])
        return ImageD11.unitcell.unitcell(lattice_parameters, symmetry)

    def _get_voxel_size(self, file_path):
        """Get the voxel size from the h5 file.

        NOTE: only isotropic voxel size is supported at the moment.

        Args:
            file_path (str): The path to the h5 file containing the
                grain volume data. The file should be in the lab-dct format.

        Returns:
            float: The voxel size in microns.
        """
        with h5py.File(file_path, "r") as f:
            voxel_size = f["LabDCT/Spacing"][()][:] * 1e3
            if not np.allclose(voxel_size, voxel_size[0]):
                raise NotImplementedError(
                    "non isotropic voxel size not implemented yet"
                )
        return voxel_size[0]

    def _get_grainid(self, file_path):
        """Get the grain id from the h5 file.

        Args:
            file_path (str): The path to the h5 file containing the
                grain volume data. The file should be in the lab-dct format.

        Returns:
            labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
                axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
                increase from low to high along the positive axis direction.
        """
        with h5py.File(file_path, "r") as f:
            labels = f["LabDCT/Data/GrainId"][()].astype(np.int32) - 1
        return labels

    def _get_voxel_grid(self, voxel_size, labels):
        """Get the voxel grid from the voxel size and labels.

        Args:
            voxel_size (float): The voxel size in microns.
            labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
                axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
                increase from low to high along the positive axis direction.

        Returns:
            X (np.ndarray): The x coordinates of the voxels in microns. shape=(m,n,o)
            Y (np.ndarray): The y coordinates of the voxels in microns. shape=(m,n,o)
            Z (np.ndarray): The z coordinates of the voxels in microns. shape=(m,n,o)
        """
        x = voxel_size * (np.arange(0, labels.shape[2], 1) - labels.shape[2] // 2)
        y = voxel_size * (np.arange(0, labels.shape[1], 1) - labels.shape[1] // 2)
        z = voxel_size * (np.arange(0, labels.shape[0], 1) - labels.shape[0] // 2)
        return np.meshgrid(z, y, x, indexing="ij")

    def _get_orientations(self, rodrigues, labels):
        """Get the orientations from the h5 file.

        Args:
            rodrigues (np.ndarray): The Rodrigues vector for each voxel
                shape=(m,n,o,3).
            labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
                axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
                increase from low to high along the positive axis direction.

        Returns:
            orientations (np.ndarray): The orientations of the grains
                in Rodrigues vector format. shape=(number_of_grains, 3, 3)
        """
        mask = labels > -1
        index = np.argsort(labels[mask])
        rods = rodrigues[mask][index]
        gids = labels[mask][index]
        ii = np.where(np.diff(gids, append=0) != 0)[0]
        uni_rod = rods[ii]
        norm = np.linalg.norm(uni_rod, axis=1)
        euler_axis = uni_rod / norm[:, np.newaxis]
        theta = 2 * np.arctan(norm)
        euler_axis *= theta[:, np.newaxis]
        return Rotation.from_rotvec(euler_axis).as_matrix()

    def _get_B0(self, reference_cell):
        """Get the B0 matrix from the reference cell.

        NOTE: The B0 matrix is not scaled by 2 pp. This is to remain
        consistent with the ImageD11 package.

        Args:
            reference_cell (ImageD11.unitcell): The reference cell of
                the grain volume.

        Returns:
            np.ndarray: The B0 matrix for the grain volume. I.e the
                recirprocal lattice vectors in the lab frame
                (upper triangular 3,3 matrix).
        """

        a, b, c, alpha, beta, gamma = reference_cell.lattice_parameters

        alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)

        calp = np.cos(alpha)
        cbet = np.cos(beta)
        cgam = np.cos(gamma)
        salp = np.sin(alpha)
        sbet = np.sin(beta)
        sgam = np.sin(gamma)

        angular = np.sqrt(
            1 - calp * calp - cbet * cbet - cgam * cgam + 2 * calp * cbet * cgam
        )
        V = a * b * c * angular

        astar = b * c * salp / V
        bstar = a * c * sbet / V
        cstar = a * b * sgam / V

        sbetstar = V / (a * b * c * salp * sgam)
        sgamstar = V / (a * b * c * salp * sbet)

        cbetstar = (calp * cgam - cbet) / (salp * sgam)
        cgamstar = (calp * cbet - cgam) / (salp * sbet)

        return np.array(
            [
                [astar, bstar * cgamstar, cstar * cbetstar],
                [0, bstar * sgamstar, -cstar * sbetstar * calp],
                [0, 0, cstar * sbetstar * salp],
            ]
        )


# Precompile on import
LabDCTVolume._neighwalk.compile(
    (
        numba.types.Array(numba.int32, 3, "C"),  # labels volume
        numba.types.Array(numba.int32, 2, "C"),  # neighbours
        numba.types.Array(numba.uint8, 3, "C"),  # struct
    )
)

LabDCTVolume._grain_sizes.compile(
    (
        numba.types.Array(numba.int32, 3, "C"),  # labels
        numba.types.Array(numba.int32, 1, "C"),  # grain sizes
    )
)

LabDCTVolume._filter.compile(
    (
        numba.types.Array(numba.int32, 3, "C"),  # labels volume
        numba.types.Array(numba.float32, 4, "C"),  # rodrigues volume
        numba.types.Array(numba.int32, 1, "C"),  # grain sizes
        numba.types.int32,  # min_grain_size_in_voxels
    )
)

LabDCTVolume._relabel.compile(
    (
        numba.types.Array(numba.int32, 3, "C"),  # labels volume
        numba.types.Array(numba.int32, 1, "C"),  # mapping
        numba.types.Array(numba.uint8, 3, "C"),  # mask
    )
)

if __name__ == "__main__":
    pass
