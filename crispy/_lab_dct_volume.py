import os

import h5py
import ImageD11.grain
import ImageD11.unitcell
import meshio
import numba
import numpy as np
import xfab
import xfab.symmetry
from scipy.spatial.transform import Rotation

from crispy._constants import _CRYSTAL_SYSTEM_STR_TO_INT, DEFAULT_STRUCT
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
            matrix vector format. shape=(number_of_grains, 3, 3)
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
        self.voxel_rotations = self._get_voxel_rotations(file_path, self.labels)
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
        self._check_labels()

        self.orientations = self._get_orientations(self.voxel_rotations, self.labels)
        self.grains = self._get_grains(self.B0, self.orientations, self.reference_cell)
        self.number_of_grains = len(self.grains)

        if np.max(self.labels) != self.number_of_grains - 1:
            raise ValueError(
                "Number of grains is not consistent with the maximum label"
            )

        self.bounding_box = self._get_bounding_box(self.X, self.Y, self.Z)
        self.neighbours, self.interface_areas, self.grain_sizes = self._spatial_search(
            self.labels, self.number_of_grains
        )
        self.centroids = self._get_centroids(self.grain_sizes)
        if self._misorientations is not None:
            self.texturize()

    def _check_labels(self):
        """Check that labels are sequential with no gaps starting from -1 or 0.

        Raises:
            ValueError: If labels are not sequential or have gaps.
        """

        unique_labels = np.unique(self.labels)
        min_label = np.min(unique_labels)

        if not min_label == -1 or min_label == 0:
            raise ValueError("Labels must start from -1 or 0")

        expected_labels = np.arange(len(unique_labels)) + np.min(unique_labels)
        if not np.all(unique_labels == expected_labels):
            print("unique_labels", unique_labels)
            print("expected_labels", expected_labels)
            raise ValueError(
                "Labels are not sequential in the range [-1, number_of_grains-1] with -1 for void"
            )

    def _spatial_search(self, labels, number_of_grains, struct=DEFAULT_STRUCT):
        """Compute spatial properties of grains in the voxel volume.

        By default, neighbourhoods are defined though the 3D structuring element:
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
                grain volume. shape=(number_of_grains,)
                The neighbours are defined as the labels of the grains that
                are connected to the grain with the same label.
            interface_areas (np.ndarray): The interface areas between each
                pair of grains in the grain volume in units of voxels.
                shape=(number_of_grains,)
            grain_sizes (np.ndarray): The size of each grain in the grain
                volume in units of voxels. shape=(number_of_grains,)
        """
        assert isinstance(struct, np.ndarray), (
            "Structuring element must be a numpy array"
        )
        assert struct.ndim == 3, "Structuring element must be 3D"
        assert (
            struct.shape[0] % 2 == 1
            and struct.shape[1] % 2 == 1
            and struct.shape[2] % 2 == 1
        ), "Structuring element must have odd shape in all dimensions"
        assert np.max(labels) + 1 == number_of_grains, (
            "Labels must be in range [-1, number_of_grains-1] with -1 for void"
        )
        assert np.min(labels) == -1, (
            "Labels must be in range [-1, number_of_grains-1] with -1 for void"
        )
        assert labels.dtype == np.int32, "Labels must be of type int32"

        structure_matrix = np.zeros(
            (number_of_grains, number_of_grains), dtype=np.uint32
        )
        LabDCTVolume._volume_walker(labels, structure_matrix, struct)

        return self._unpack_structure_matrix(structure_matrix, number_of_grains)

    def _unpack_structure_matrix(self, structure_matrix, number_of_grains):
        """Unpack a structure matrix into neighbours, interface areas and grain sizes.

        At any one point in the voxel volume if a voxel of label i neighbours
        a voxel of label j, the structure_matrix[i, j] is incremented by 1
        counting up the interface area between the two grains. The diagonal
        of the structure_matrix, representing the neighbourhood of a grain
        with itself, is reserved for the grain size such that whenever a
        non-void voxel is traversed, the structure_matrix[i, i] is incremented
        by 1 counting up the number of voxels in the grain.

        Args:
            structure_matrix (np.ndarray): The structure matrix to unpack.
                shape=(number_of_grains, number_of_grains)
            number_of_grains (int): The number of grains in the grain volume.

        Returns:
            neighbours (np.ndarray): The neighbours of each grain in the
                grain volume. shape=(number_of_grains,)
                The neighbours are defined as the labels of the grains that
                are connected to the grain with the same label.
            interface_areas (np.ndarray): The interface areas between each
                pair of grains in the grain volume in units of voxels.
                shape=(number_of_grains,)
            grain_sizes (np.ndarray): The size of each grain in the grain
                volume in units of voxels. shape=(number_of_grains,)
        """
        grain_sizes = np.diag(structure_matrix)
        neighbours = np.empty((number_of_grains,), dtype=np.ndarray)
        interface_areas = np.empty((number_of_grains,), dtype=np.ndarray)
        for i in range(structure_matrix.shape[0]):
            mask = structure_matrix[i, :] > 0
            mask[i] = False
            neighbours[i] = np.where(mask)[0][1:].astype(np.uint32)
            interface_areas[i] = structure_matrix[i, neighbours[i]]
        return neighbours, interface_areas, grain_sizes

    @staticmethod
    @numba.njit(parallel=True)
    def _volume_walker(labels, structure_matrix, struct):
        """Walk a 3D labeled array and find the neighbours of each label.

        Neighbours are defined though the 3D structuring element struct.

        This is a parallelized and compiled function that is used internally
        to speed up the computation of the neighbours of each while avoiding
        any bloated memory usage.

        The filling of the input structure_matrix gives information about:
            (1) The size of each grain in the grain volume.
            (2) The number index of neighbours of each grain in the grain volume.
            (3) The interface area between each pair of grains in the grain volume.

        Algorithm description:
            The deployed algorihtm evolves around the filling of a so named
            structure_matrix. This is simply a N x N matrix of unsigned 32 bit
            integers. The matrix is filled by walking the 3D labeled array and
            checking the neighbours of each label. The neighbours are defined
            though the 3D structuring element struct. At any one point in the
            voxel volume if a voxel of label i neighbours a voxel of label j,
            the structure_matrix[i, j] is incremented by 1 counting up the
            interface area between the two grains. The diagonal of the
            structure_matrix, representing the neighbourhood of a grain
            with itself, is reserved for the grain size such that whenever a
            non-void voxel is traversed, the structure_matrix[i, i] is incremented
            by 1 counting up the number of voxels in the grain.

            NOTE: This algorithm depends on the fact that the labels are
            non-negative integers starting from 0 and ending at N-1 where N
            is the number of grains in the grain volume. A voxel with a label
            of -1 is considered to be void.

        Args:
            labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
                axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
                increase from low to high along the positive axis direction.
            structure_matrix (np.ndarray): shape=(number_of_grains, number_of_grains)
                of type uint32. The structure matrix is used to store the
                number of neighbours for each grain, and the grain sizes.
            struct (np.ndarray): The structuring element to use for the
                neighbourhood. The structuring element should be a 3D
                uint8 array of 1s and 0s.

        """

        num_labels = structure_matrix.shape[0]
        zdim, ydim, xdim = labels.shape
        m, n, o = struct.shape[0] // 2, struct.shape[1] // 2, struct.shape[2] // 2
        mask = labels > -1

        # Each thread gets its own local structure matrix
        thread_count = numba.get_num_threads()
        local_matrices = np.zeros(
            (thread_count, num_labels, num_labels), dtype=np.uint32
        )

        z_chunks = np.array_split(np.arange(zdim), thread_count)

        for thread_id in numba.prange(thread_count):
            for i in z_chunks[thread_id]:
                for j in range(ydim):
                    for k in range(xdim):
                        if mask[i, j, k]:
                            label = labels[i, j, k]
                            local_matrices[thread_id, label, label] += 1

                            if (
                                i != 0
                                and j != 0
                                and k != 0
                                and i != zdim - 1
                                and j != ydim - 1
                                and k != xdim - 1
                            ):
                                for ii in range(struct.shape[0]):
                                    for jj in range(struct.shape[1]):
                                        for kk in range(struct.shape[2]):
                                            if struct[ii, jj, kk]:
                                                _i = i + ii - m
                                                _j = j + jj - n
                                                _k = k + kk - o
                                                if mask[_i, _j, _k]:
                                                    nlabel = labels[_i, _j, _k]
                                                    if label != nlabel:
                                                        local_matrices[
                                                            thread_id, label, nlabel
                                                        ] += 1
        # Reduction step (serial)
        for t in range(thread_count):
            for i in range(num_labels):
                for j in range(num_labels):
                    structure_matrix[i, j] += local_matrices[t, i, j]

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

    def _update_voxel_rotations(self, old_labels, new_labels):
        """Update, in-place, the per voxel orientations after cropping or filtering.

        self.voxel_rotations is the per voxel rotation for non-void voxels
        as a iterable flat structure, thus it requires special attention
        to update the orientations after cropping or filtering the 3D volume.

        Args:
            old_labels (np.ndarray): The labels of the grain volume before
                cropping or filtering.
            new_labels (np.ndarray): The labels of the grain volume after
                cropping or filtering.
        """
        # this is what is in voxel_rotations before filtering
        old_mask = old_labels > -1

        # after filtering these are the non-void voxels
        new_mask = new_labels > -1

        # voxels that are void in the new mask but not in the old mask
        # are removed from the voxel_rotations here
        to_keep = new_mask[old_mask]
        self.voxel_rotations = self.voxel_rotations[to_keep]

    @property
    def center_of_mass(self):
        """The center of mass of the voxel volume in units of microns.

        The center of mass is defined as the arithmetic mean of the
        coordinates of the non-void voxels.

        returns:
            numpy.ndarray of shape = (3,)

        """
        mask = self.labels > -1
        return np.array(
            [
                np.mean(self.X[mask]),
                np.mean(self.Y[mask]),
                np.mean(self.Z[mask]),
            ]
        )

    def _get_centroids(self, grain_sizes):
        """Centroids of each grain in the voxel volume in units of microns.

        returns:
            numpy.ndarray of shape = (number_of_grains, 3)
        """
        cs = np.zeros((self.number_of_grains, 3), dtype=np.float32)
        LabDCTVolume._coord_sum(self.X, self.Y, self.Z, self.labels, cs)
        return cs / grain_sizes[:, np.newaxis]

    @staticmethod
    @numba.njit(parallel=True)
    def _coord_sum(X, Y, Z, labels, cs):
        # this is not threadsafe by default as we plan to add at similar
        # locations in the cs array. Each thread gets a local copy of cs
        # and at the end we sum them up.

        n_threads = numba.get_num_threads()
        n_labels = cs.shape[0]
        thread_cs = np.zeros((n_threads, n_labels, 3), dtype=cs.dtype)
        mask = labels > -1

        z_chunks = np.array_split(np.arange(labels.shape[0]), n_threads)

        for thread_id in numba.prange(n_threads):
            for i in z_chunks[thread_id]:
                for j in range(labels.shape[1]):
                    for k in range(labels.shape[2]):
                        if mask[i, j, k]:
                            label = labels[i, j, k]

                            # here there is thread-breaking, hence the thread_id
                            # is used to get the local copy of cs and add the
                            # coordinates to it.
                            thread_cs[thread_id, label, 0] += X[i, j, k]
                            thread_cs[thread_id, label, 1] += Y[i, j, k]
                            thread_cs[thread_id, label, 2] += Z[i, j, k]

        for t in range(n_threads):
            for l in range(n_labels):
                cs[l, 0] += thread_cs[t, l, 0]
                cs[l, 1] += thread_cs[t, l, 1]
                cs[l, 2] += thread_cs[t, l, 2]

    def center_volume(self):
        """Center the volume around the center of mass."""
        self.translate(-self.center_of_mass)

    def filter(self, min_grain_size_in_voxels):
        """Filter the grains based on their size.

        Grains with a size smaller than grain_size will be removed from
        the voxel volume. Attributes will be updated accordingly.

        Args:
            min_grain_size_in_voxels (int): The minimum size of the grains to keep in
                units of voxels.
        """
        old_labels = self.labels.copy()
        LabDCTVolume._filter(
            self.labels,
            self.grain_sizes,
            min_grain_size_in_voxels,
        )
        self._update_voxel_rotations(old_labels, self.labels)
        self._update_state()

    @staticmethod
    @numba.njit(parallel=True, cache=True)
    def _filter(labels, sizes, min_grain_size_in_voxels):
        """Parallelized function to filter on grain size."""
        mask = labels > -1
        for i in numba.prange(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(labels.shape[2]):
                    if mask[i, j, k]:
                        label = labels[i, j, k]
                        if sizes[label] < min_grain_size_in_voxels:
                            # this is threadsafe by default as each thread works on a different
                            # part of the labels array, i.e we have that i,j,k are unique.
                            labels[i, j, k] = -1

    def _reset_labels(self):
        """Decrement labels to start from -1,0,1,2,...,n where -1 is the void label."""
        mask = self.labels > -1
        unique_labels = np.unique(self.labels[mask])
        mapping = np.zeros((unique_labels.max() + 1,), dtype=np.int32)
        for new_label, old_label in enumerate(unique_labels):
            mapping[old_label] = new_label
        LabDCTVolume._relabel(self.labels, mapping, mask)

    @staticmethod
    @numba.njit(parallel=True, cache=True)
    def _relabel(labels, mapping, mask):
        for i in numba.prange(labels.shape[0]):
            for j in range(labels.shape[1]):
                for k in range(labels.shape[2]):
                    if mask[i, j, k]:
                        # this is threadsafe by default as each thread works on a different
                        # part of the labels array, i.e we have that i,j,k are unique.
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
        xmin = np.min(self.X[0, 0, :]) if xmin is None else xmin
        xmax = np.max(self.X[0, 0, :]) if xmax is None else xmax
        ymin = np.min(self.Y[0, :, 0]) if ymin is None else ymin
        ymax = np.max(self.Y[0, :, 0]) if ymax is None else ymax
        zmin = np.min(self.Z[:, 0, 0]) if zmin is None else zmin
        zmax = np.max(self.Z[:, 0, 0]) if zmax is None else zmax

        self._validate_crop_bounds(
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
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

        old_labels = self.labels.copy()
        vol = np.ones_like(self.labels, dtype=bool)
        vol[i1:i2, j1:j2, k1:k2] = False
        self.labels[vol] = -1
        self._update_voxel_rotations(old_labels, self.labels)

        self.labels = self.labels[i1:i2, j1:j2, k1:k2]
        self.X = self.X[i1:i2, j1:j2, k1:k2]
        self.Y = self.Y[i1:i2, j1:j2, k1:k2]
        self.Z = self.Z[i1:i2, j1:j2, k1:k2]

        self._update_state()

    def _validate_crop_bounds(self, xmin, xmax, ymin, ymax, zmin, zmax):
        """Validate the crop bounds."""

        x0, x1, y0, y1, z0, z1 = self.bounding_box
        assert xmin >= x0, "Crop bounds are outside the bounding box in x"
        assert xmax <= x1, "Crop bounds are outside the bounding box in x"
        assert ymin >= y0, "Crop bounds are outside the bounding box in y"
        assert ymax <= y1, "Crop bounds are outside the bounding box in y"
        assert zmin >= z0, "Crop bounds are outside the bounding box in z"
        assert zmax <= z1, "Crop bounds are outside the bounding box in z"

        assert xmax >= xmin + self.voxel_size, "Crop bounds are too small in x"
        assert ymax >= ymin + self.voxel_size, "Crop bounds are too small in y"
        assert zmax >= zmin + self.voxel_size, "Crop bounds are too small in z"

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
        dx, dy, dz = translation
        self.X += dx
        self.Y += dy
        self.Z += dz
        self.bounding_box = self._get_bounding_box(self.X, self.Y, self.Z)
        for g in self.grains:
            if hasattr(g, "translation") and g.translation is not None:
                g.translation += translation
        self.centroids += translation

    def rotate(self, rotation):
        """Rotate the voxel volume by the specified rotation.

        The voxel volume will be rotated by the specified rotation.
        The attributes will be updated accordingly.

        This will update both voxel coordinates and grain orientations.

        This is usefull for realigning the voxel volume to be used in
        diffraction simulations or in integration with a synchrotron
        measurement setting.

        Args:
            rotation (np.ndarray or scipy.spatial.transform.Rotation): The rotation
                matrix, rotation vector or rotation object to rotate the voxel volume by.
                if this is a rotation vector, it should be of shape (3,) with the norm
                equal to the angle of rotation in radians. The direction of the vector
                is the axis of rotation.
        """
        if isinstance(rotation, Rotation):
            scipy_rot = rotation
        elif isinstance(rotation, np.ndarray):
            if rotation.shape == (3,):
                scipy_rot = Rotation.from_rotvec(rotation)
            elif rotation.shape == (3, 3):
                scipy_rot = Rotation.from_matrix(rotation)
            else:
                raise ValueError("Rotation matrix must be of shape (3, 3)")
        else:
            raise ValueError(
                "Rotation must be a rotation matrix or rotation vector or scipy.spatial.transform.Rotation object"
            )

        rotation_matrix = scipy_rot.as_matrix()
        c1, c2, c3 = rotation_matrix.T

        rotated_X = c1[0] * self.X + c2[0] * self.Y + c3[0] * self.Z
        rotated_Y = c1[1] * self.X + c2[1] * self.Y + c3[1] * self.Z
        rotated_Z = c1[2] * self.X + c2[2] * self.Y + c3[2] * self.Z

        for g in self.grains:
            if hasattr(g, "translation") and g.translation is not None:
                g.translation = rotation_matrix @ g.translation
            ubi = np.linalg.inv(rotation_matrix @ g.u @ g.B)
            g.set_ubi(ubi)

        self.voxel_rotations = scipy_rot * self.voxel_rotations
        self.orientations = self._get_orientations(self.voxel_rotations, self.labels)
        self.X = rotated_X
        self.Y = rotated_Y
        self.Z = rotated_Z
        self.bounding_box = self._get_bounding_box(self.X, self.Y, self.Z)

    def _get_voxel_rotations(self, file_path, labels):
        """Get the Rodrigues vector from the h5 file."""
        with h5py.File(file_path, "r") as f:
            rodrigues = f["LabDCT/Data/Rodrigues"][...].astype(np.float32)
        rodrigues = rodrigues[labels > -1]
        norm = np.linalg.norm(rodrigues, axis=1)
        euler_axis = rodrigues / norm[:, np.newaxis]
        theta = 2 * np.arctan(norm)
        euler_axis *= theta[:, np.newaxis]
        return Rotation.from_rotvec(euler_axis)

    def _get_grains(self, B0, orientations, reference_cell):
        """Get the grains from the orientations and B0 matrix.

        Args:
            B0 (np.ndarray): The B0 matrix for the grain volume.
                I.e the recirprocal lattice vectors in the lab frame
                (upper triangular 3,3 matrix).
            orientations (scipy.spatial.transform.Rotation): The orientations
                of the grains as 3x3 rotation matrices.
                shape=(number_of_grains, 3, 3)
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

    def _get_orientations(self, voxel_rotations, labels):
        """Get the orientations from the h5 file.

        Args:
            voxel_rotations (scipy.spatial.transform.Rotation): A
                scipy.spatial.transform.Rotation object
                per voxel in a flat manner skipping void voxels.
            labels (np.ndarray): The grain id for each voxel shape=(m,n,o)
                axis=0 is z, axis=1 is y, axis=2 is x. coordinates always
                increase from low to high along the positive axis direction.

        Returns:
            orientations (np.ndarray): The orientations of the grains
                in matrix format. shape=(number_of_grains, 3, 3)
        """
        mask = labels > -1
        index = np.argsort(labels[mask])
        gids = labels[mask][index]
        ii = np.where(np.diff(gids, append=0) != 0)[0]
        unique_rotations = voxel_rotations[index][ii]
        return unique_rotations.as_matrix()

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
LabDCTVolume._volume_walker.compile(
    (
        numba.types.Array(numba.int32, 3, "C"),  # labels volume
        numba.types.Array(numba.uint32, 2, "C"),  # structure matrix
        numba.types.Array(numba.uint8, 3, "C"),  # struct
    )
)

LabDCTVolume._filter.compile(
    (
        numba.types.Array(numba.int32, 3, "C"),  # labels volume
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

LabDCTVolume._coord_sum.compile(
    (
        numba.types.Array(numba.float32, 3, "C"),  # X
        numba.types.Array(numba.float32, 3, "C"),  # Y
        numba.types.Array(numba.float32, 3, "C"),  # Z
        numba.types.Array(numba.int32, 3, "C"),  # labels
        numba.types.Array(numba.float32, 2, "C"),  # cs
    )
)

if __name__ == "__main__":
    pass
if __name__ == "__main__":
    pass
