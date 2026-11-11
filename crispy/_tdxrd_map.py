import re

import ImageD11.grain
import ImageD11.unitcell
import numpy as np
import xfab
import xfab.symmetry

from . import _tesselate
from ._constants import CONSTANTS
from ._polycrystal import Polycrystal


class TDXRDMap(Polycrystal):
    """Class to represent a polycrystal as a list of grain centroids and orientations.

    Args:
        grainfile (:obj:`str` or :obj:`list` of :obj:`ImageD11.grain.grain`):
            Absolute path to the HDF5 file containing the grain information, or
            alternatively a list of ImageD11.grain.grain objects.
        group_name (:obj:`str`): The name of the group in the HDF5 file containing
            the grain information. Defaults to "grains".


    Attributes:
        grains (:obj:`list` of :obj:`ImageD11.grain.grain`): List of grains in the polycrystal.
        number_of_grains (:obj:`int`): Number of grains in the polycrystal.
        u (:obj:`numpy.ndarray`): All the U matrices of the grains in the polycrystal. Shape=(N, 3, 3).
        orientations (:obj:`numpy.ndarray`): Alias for u.
        b (:obj:`numpy.ndarray`): All the B matrices of the grains in the polycrystal. Shape=(N, 3, 3).
        misorientations (:obj:`numpy.ndarray`): The misorientations between all grain neighbours. shape=(N,).
            each sub-array contains the misorientations between the corresponding grain and its neighbours.
            is None before calling the texturize() method.
    """

    def __init__(
        self, grainfile, group_name="grains", lattice_parameters=None, symmetry=None
    ):
        if isinstance(grainfile, list):
            self.grains = np.array(grainfile)
        else:
            self.grains = np.array(
                ImageD11.grain.read_grain_file_h5(grainfile, group_name)
            )

        if lattice_parameters and symmetry:
            self.reference_cell = ImageD11.unitcell.unitcell(
                lattice_parameters, symmetry
            )
        else:
            if hasattr(self.grains[0], "ref_unitcell"):
                self.reference_cell = self.grains[0].ref_unitcell
            else:
                raise ValueError(
                    "No reference cell parameters and/or symmetry passed and the grains do not contain a ref_cell attribute"
                )

        self.number_of_grains = len(self.grains)

        self._mesh = None
        self._misorientations = None

        self.add_grain_attr(self.grains)

    def add_grain_attr(self, grains):
        for grain_id, g in enumerate(grains):
            if not hasattr(g, "id"):
                g.id = grain_id

            if hasattr(g, "intensity_info"):
                pattern = r"sum_of_all = (?P<sum_of_all>[\d.]+).*?median = (?P<median>[\d.]+) , min = (?P<min>[\d.]+) , max = (?P<max>[\d.]+) , mean = (?P<mean>[\d.]+) , std = (?P<std>[\d.]+) , n = (?P<n>\d+)"
                vars = re.search(pattern, g.intensity_info).groupdict()
                g.sum_peak_intensity = float(vars["sum_of_all"])
                g.median_peak_intensity = float(vars["median"])
                g.mean_peak_intensity = float(vars["median"])
                g.min_peak_intensity = float(vars["min"])
                g.max_peak_intensity = float(vars["max"])
                g.std_peak_intensity = float(vars["std"])
                g.number_of_peaks = int(vars["n"])

    @property
    def neighbours(self):
        """Return the neighbours of each grain in the polycrystal."""
        if self._mesh is None:
            raise ValueError(
                "Mesh is None, did you forget to call the tesselate() method?"
            )
        return self._mesh.neighbours.copy()

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
    def sum_peak_intensity(self):
        """Return sum intensity of all indexed peaks for each grain. shape=(N,)"""
        return np.array([g.sum_peak_intensity for g in self.grains])

    @property
    def number_of_peaks(self):
        """Return number of indexed peaks for each grain. shape=(N,)"""
        return np.array([g.number_of_peaks for g in self.grains])

    @property
    def median_peak_intensity(self):
        """Return the median peak intensity for all grains. shape=(N,)"""
        return np.array([g.median_peak_intensity for g in self.grains])

    @property
    def mean_peak_intensity(self):
        """Return the mean peak intensity for all grains. shape=(N,)"""
        return np.array([g.mean_peak_intensity for g in self.grains])

    @property
    def min_peak_intensity(self):
        """Return the minimum peak intensity for all grains. shape=(N,)"""
        return np.array([g.min_peak_intensity for g in self.grains])

    @property
    def max_peak_intensity(self):
        """Return the maximum peak intensity for all grains. shape=(N,)"""
        return np.array([g.max_peak_intensity for g in self.grains])

    @property
    def std_peak_intensity(self):
        """Return the standard deviation of peak intensities for all grains. shape=(N,)"""
        return np.array([g.std_peak_intensity for g in self.grains])

    @classmethod
    def from_array(
        cls, translations, ubi_matrices, lattice_parameters=None, symmetry=None
    ):
        """Instantiate the polycrystal from pure translations and ubi matrices.

        This circumvents the need to read a grain file and ImageD11.grain.grain objects.

        Args:
            translations (:obj:`iterable` of :obj:`numpy.ndarray`): 3D grain center translations each of shape=(3,).
            ubi_matrices (:obj:`iterable` of :obj:`numpy.ndarray`): ubi matrices each of shape=(3, 3).
            lattice_parameters (:obj:`iterable` of :obj:`numpy.ndarray`): lattice parameters each of shape=(6,).
            symmetry (:obj:`int`): space group number.

        Returns:
            :obj:`crispy._tdxrd_map.TDXRDMAP`: The polycrystal object.
        """
        if len(translations) != len(ubi_matrices):
            raise ValueError("translations and ubi_matrices must have the same length")
        grains = [ImageD11.grain.grain(ubi) for ubi in ubi_matrices]
        reference_cell = ImageD11.unitcell.unitcell(lattice_parameters, symmetry)
        for g, t in zip(grains, translations):
            g.translation = t
            g.ref_unitcell = reference_cell
        return cls(grains)

    @classmethod
    def from_midas(cls, grainfile, lattice_parameters=None, symmetry=None):
        """Load a single-phase MIDAS ``Grains.csv`` with fitted lattice parameters.

        Columns are read by name. Orientations and positions retain the MIDAS
        rotation-stage frame; no sample or laboratory frame rotation is applied.
        MIDAS and ImageD11 both use crystal-to-sample orientation matrices and
        the same reciprocal basis, in inverse angstroms without a 2*pi factor.

        Each grain has a consecutive ``id``, its original ``midas_id``, and a
        ``midas`` dictionary containing all source columns without conversion.
        Positions and radii are in micrometres. The eFab/eKen columns are kept
        in microstrain, separate from the strain computed from the fitted cell.

        Args:
            grainfile (:obj:`str`): Path to the MIDAS grain table.
            lattice_parameters (:obj:`iterable`): Reference cell parameters
                [a, b, c, alpha, beta, gamma], in angstroms and degrees.
                Defaults to the reference lattice in the file header.
            symmetry (:obj:`int`): Space group number. Defaults to the file header.

        Returns:
            :obj:`crispy.TDXRDMap`: The imported grain map.

        Raises:
            ValueError: If required columns or reference cell data are missing,
                the table is empty, contains multiple phases, or has invalid grains.
        """
        columns = None
        with open(grainfile) as f:
            for line in f:
                if not line.lstrip().startswith("%"):
                    continue
                header = line.lstrip().lstrip("%").strip()
                if "O11" in header.split():
                    columns = header.split()
                    break
                if header.startswith("NumPhases") and int(header.split()[1]) != 1:
                    raise ValueError("MIDAS imports require a single phase")
                if header.startswith("SpaceGroup:") and symmetry is None:
                    symmetry = int(header.split(":", 1)[1])
                if (
                    header.startswith("Lattice Parameter:")
                    and lattice_parameters is None
                ):
                    lattice_parameters = [
                        float(x) for x in header.split(":", 1)[1].split()
                    ]
            if columns is None:
                raise ValueError("Missing MIDAS column header")
            rows = [
                line
                for line in f
                if line.strip() and not line.lstrip().startswith("%")
            ]

        if not rows:
            raise ValueError("MIDAS table contains no grains")
        if lattice_parameters is None or symmetry is None:
            raise ValueError(
                "MIDAS reference lattice parameters and symmetry are required"
            )
        if symmetry not in range(1, 231):
            raise ValueError("MIDAS symmetry must be a space group number from 1 to 230")
        symmetry = int(symmetry)

        id_column = "GrainID" if "GrainID" in columns else "ID"
        orientation_columns = [
            f"O{i}{j}" for i in range(1, 4) for j in range(1, 4)
        ]
        cell_columns = ["a", "b", "c", "alpha", "beta", "gamma"]
        required = (
            [id_column] + orientation_columns + ["X", "Y", "Z"] + cell_columns
        )
        if len(set(columns)) != len(columns) or any(
            c not in columns for c in required
        ):
            raise ValueError("Missing or duplicate MIDAS grain columns")
        data = np.loadtxt(rows, ndmin=2)
        if data.shape[1] != len(columns):
            raise ValueError("MIDAS data does not match the column header")

        ids = data[:, columns.index(id_column)]
        if (
            not np.all(np.isfinite(ids))
            or np.any(ids != np.floor(ids))
            or len(np.unique(ids)) != len(ids)
        ):
            raise ValueError("MIDAS grain IDs must be unique integers")
        if "PhaseNr" in columns:
            phases = data[:, columns.index("PhaseNr")]
            if not np.all(np.isfinite(phases)) or len(np.unique(phases)) != 1:
                raise ValueError("MIDAS imports require a single phase")

        orientations = data[:, [columns.index(c) for c in orientation_columns]]
        orientations = orientations.reshape(-1, 3, 3)
        # MIDAS tables may round orientation entries to six decimal places.
        if (
            not np.all(np.isfinite(orientations))
            or not np.allclose(
                orientations @ orientations.transpose(0, 2, 1),
                np.eye(3),
                atol=1e-5,
                rtol=0,
            )
            or np.any(np.linalg.det(orientations) <= 0)
        ):
            raise ValueError("Invalid MIDAS orientation matrix")
        translations = data[:, [columns.index(c) for c in ["X", "Y", "Z"]]]
        if not np.all(np.isfinite(translations)):
            raise ValueError("Invalid MIDAS grain position")
        fitted_cells = data[:, [columns.index(c) for c in cell_columns]]
        cells = np.vstack((lattice_parameters, fitted_cells))
        if (
            not np.all(np.isfinite(cells))
            or np.any(cells[:, :3] <= 0)
            or np.any(cells[:, 3:] <= 0)
            or np.any(cells[:, 3:] >= 180)
        ):
            raise ValueError("Invalid MIDAS lattice parameters")
        ca, cb, cg = np.cos(np.radians(cells[:, 3:])).T
        if np.any(1 + 2 * ca * cb * cg - ca**2 - cb**2 - cg**2 <= 0):
            raise ValueError("MIDAS lattice angles must define a positive volume")

        reference_cell = ImageD11.unitcell.unitcell(lattice_parameters, symmetry)
        grains = []
        for row, u, t, cell, gid in zip(
            data, orientations, translations, fitted_cells, ids
        ):
            B = ImageD11.unitcell.unitcell(cell).B
            ubi = np.linalg.inv(u @ B)
            g = ImageD11.grain.grain(ubi, translation=t)
            g.ref_unitcell = reference_cell
            g.midas_id = int(gid)
            g.midas = dict(zip(columns, row))
            grains.append(g)
        return cls(grains)

    def tesselate(self):
        """Perform a voronoi tesselation using crispy.tesselate.voronoi

        Generates the polycrystal mesh with one convex polyhedra per grain.
        """
        self._mesh = _tesselate.voronoi(list(self.grains))

        # propagate polygon mesh data to the grain objects for easy access.
        grain_volumes = self._mesh.cell_data["grain_volumes"][0]
        grain_id = self._mesh.cell_data["grain_id"][0]
        is_on_boundary = self._mesh.cell_data["surface_grain"][0]
        triangle_nodes = self._mesh.cells[0].data
        nodes = self._mesh.points
        for g in self.grains:
            m = g.id == grain_id

            g.volume = grain_volumes[m][0]
            g.equivalent_sphere_radii = (3 * g.volume / (4 * np.pi)) ** (1 / 3)
            g.is_on_boundary = is_on_boundary[m][0] == 1

            high = nodes[triangle_nodes[m, :].flatten()].max(axis=0)
            low = nodes[triangle_nodes[m, :].flatten()].min(axis=0)
            width = high - low

            g.bounding_box_lower_corner_x = low[0]
            g.bounding_box_width_x = width[0]
            g.bounding_box_lower_corner_y = low[1]
            g.bounding_box_width_y = width[1]
            g.bounding_box_lower_corner_z = low[2]
            g.bounding_box_width_z = width[2]

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
        _crystal_system = CONSTANTS.CRYSTAL_SYSTEM_STR_TO_INT[self.crystal_system]
        self._misorientations = np.empty((len(self.grains),), dtype=np.ndarray)
        for gi in range(len(self.grains)):
            u = self.grains[gi].u
            self._misorientations[gi] = np.zeros((len(self._mesh.neighbours[gi]),))
            for i, ni in enumerate(self._mesh.neighbours[gi]):
                mis = xfab.symmetry.Umis(u, self.grains[ni].u, _crystal_system)
                self._misorientations[gi][i] = np.min(mis[:, 1])

    def write(self, file_path, grains="all", neighbourhood=None):
        """Write the polycrystal to a file readable by Paraview.

        The grain volume will be written to the specified file path,
        the file format is ``.xdmf`` or ``.vtk``. The appropriate extension will
        automatically be added if it is not already present.

        Args:
            file_path (:obj:`str`): Absolute path to the file to write the polycrystal
                to, must end in ``.xdmf`` or ``.vtk``.
            grains (:obj:`list` of :obj:`int` or :obj:`int` or :obj:`str`): The
                grain ids to write. Defaults to "all" in which case all grains
                in the polycrystal are written.
            neighbourhood (:obj:`int`): When not None, the grains argument is overwritten
                such that the grain number -neighbourhood- and all of its neighbours are
                written to file. Default is None.

        """
        file_path += ".vtk" if not file_path.endswith(".vtk") else ""

        if self._mesh is None:
            raise ValueError(
                "Mesh is None, did you forget to call the tesselate() method?"
            )
        elif neighbourhood is None and isinstance(grains, str) and grains == "all":
            mesh = self._mesh
        else:
            grains = self._select_grains(grains, neighbourhood)
            local_geometry = self._extract_geom(grains)
            mesh = _tesselate._build_mesh(*local_geometry)

        # Add all x,y,z ipf colors to the mesh before writing to disc.
        rgb = self._ipf_colors()

        mesh.cell_data["ipf-x"] = [np.zeros((len(mesh.cell_data["grain_id"][0]), 3))]
        mesh.cell_data["ipf-y"] = [np.zeros((len(mesh.cell_data["grain_id"][0]), 3))]
        mesh.cell_data["ipf-z"] = [np.zeros((len(mesh.cell_data["grain_id"][0]), 3))]
        for i, gid in enumerate(mesh.cell_data["grain_id"][0]):
            mesh.cell_data["ipf-x"][0][i] = rgb[gid, :, 0]
            mesh.cell_data["ipf-y"][0][i] = rgb[gid, :, 1]
            mesh.cell_data["ipf-z"][0][i] = rgb[gid, :, 2]

        mesh.write(file_path)

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
        grain_volumes = []

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
            grain_volumes.extend(self._mesh.cell_data["grain_volumes"][0][mask])

        return vertices, simplices, grain_id, grain_volumes, surface_grain


if __name__ == "__main__":
    pass
