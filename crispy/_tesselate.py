import ImageD11.grain
import meshio
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection


def _extract_bounds(seeds, bounds):
    """Get bounding box for tesselation.

    Args:
        points (:obj:`numpy.ndarray`): Numpy array of shape=(N,3) specifying the seeds
        bounds (:obj:`list` of :obj:`tuple`, optional): Bounds for the tesselation

    Returns:
        :obj:`tuple`: Tuple of the bounds in x, y, z dimensions
    """
    if bounds is None:
        xmin, xmax = seeds[:, 0].min(), seeds[:, 0].max()
        ymin, ymax = seeds[:, 1].min(), seeds[:, 1].max()
        zmin, zmax = seeds[:, 2].min(), seeds[:, 2].max()

        epsilon = 1e-8
        xmax += epsilon
        xmin -= epsilon
        ymax += epsilon
        ymin -= epsilon
        zmax += epsilon
        zmin -= epsilon

    else:
        xmin, xmax = bounds[0]
        ymin, ymax = bounds[1]
        zmin, zmax = bounds[2]
    return xmin, xmax, ymin, ymax, zmin, zmax


def _get_plane_maps(points):
    """Get distance and normal maps for each point in the tesselation.

    Args:
        points (:obj:`numpy.ndarray`): Numpy array of shape=(N,3) specifying the seeds

    Returns:
        :obj:`tuple`: Tuple of the distance and normal maps for each point in the
                tesselation,
            dmap, nmap, of shapes (N, N-1) and (N, N-1, 3) respectively. The dmap
                contains the
            distances of each point to all other points in the tesselation, while the
                nmap contains the normal vectors of the planes defined by each point and
                all other points in the tesselation such that the normal vector is
                pointing towards the other point.
    """
    dmap = np.zeros((points.shape[0], points.shape[0] - 1))
    nmap = np.zeros((points.shape[0], points.shape[0] - 1, 3))
    for i in range(points.shape[0]):
        x1 = points[i]
        v = points - x1
        norm = np.linalg.norm(v, axis=1)
        m = norm != 0
        norm = norm[m]
        v = v[m]
        dmap[i] = norm
        nmap[i] = v / norm[:, np.newaxis]
    return dmap, nmap


def _get_halfspaces(point, distance, normals, bounds):
    """Get halfspaces for each point in the tesselation.

    These are stacked inequalities of the form Ax - b >= 0, where A is the normal
    and b is the distance to the origin of the plane.

    Args:
        point (:obj:`numpy.ndarray`): Numpy array of shape=(3,) specifying the seed
        distance (:obj:`numpy.ndarray`): Numpy array of shape=(N-1,) specifying the
            distances of each point to all other points in the tesselation
        normals (:obj:`numpy.ndarray`): Numpy array of shape=(N-1,3) specifying the
            normal vectors of the planes defined by each point and all other points
            in the tesselation.
        bounds (:obj:`tuple`): Tuple of the bounds in x, y, z dimensions

    Returns:
        :obj:`numpy.ndarray`: Numpy array of shape=(N,4) specifying the halfspaces
            i.e the [A; b] array.
    """
    A = normals
    b = distance / 2.0
    x, y, z = point

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    limits = np.array(
        [
            [1, 0, 0, np.abs(xmax - x)],
            [-1, 0, 0, np.abs(xmin - x)],
            [0, 1, 0, np.abs(ymax - y)],
            [0, -1, 0, np.abs(ymin - y)],
            [0, 0, 1, np.abs(zmax - z)],
            [0, 0, -1, np.abs(zmin - z)],
        ]
    )
    limits[:, -1] *= -1
    halfspaces = np.hstack((A, -b.reshape(len(b), 1)))
    halfspaces = np.vstack((halfspaces, limits))
    return halfspaces


def _is_on_boundary(verts, bounds):
    """Check if a polyhedron is on the boundary of the tesselation.

    Args:
        verts (:obj:`numpy.ndarray`): Numpy array of shape=(N,3) specifying the vertices
            of the polyhedron
        bounds (:obj:`tuple`): Tuple of the bounds in x, y, z dimensions

    Returns:
        :obj:`int`: 1 if the polyhedron is on the boundary of the tesselation,
            0 otherwise.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    x, y, z = verts.T
    return int(
        np.abs(x.max() - xmax) < 1e-8
        or np.abs(y.max() - ymax) < 1e-8
        or np.abs(z.max() - zmax) < 1e-8
        or np.abs(x.min() - xmin) < 1e-8
        or np.abs(y.min() - ymin) < 1e-8
        or np.abs(z.min() - zmin) < 1e-8
    )


def _build_mesh(
    vertices, simplices, grain_id, grain_volumes, surface_grain, neighbours=None
):
    """construct a mesh object from the vertices and simplices.

    Args:
        vertices (:obj:`numpy.ndarray`): numpy arrays of shape=(N,3)
            specifying the vertices of each polyhedron
        simplices (:obj:`numpy.ndarray`): numpy arrays of shape=(M,3)
            specifying the simplices of each polyhedron
        grain_id (:obj:`list` of :obj:`int`): List of integers specifying the
            grain id of each polyhedron
        grain_volumes (:obj:`list` of :obj:`float`): List of floats specifying the
            grain volume of each polyhedron
        surface_grain (:obj:`list` of :obj:`int`): List of integers specifying if
            the polyhedron is on the boundary
        neighbours (:obj:`numpy.ndarray`, optional): Numpy array of shape=(N,) specifying
            the grain neighbours of each grain polyhedron. Defaults to None.

    Returns:
        :obj:`meshio.Mesh`: Mesh object of the tesselation.
    """

    mesh = meshio.Mesh(
        vertices,
        cells=[("polygon", simplices)],
        cell_data={
            "grain_id": [grain_id],
            "surface_grain": [surface_grain],
            "grain_volumes": [grain_volumes],
        },
    )
    mesh.neighbours = neighbours
    return mesh


def _extract_points(seeds):
    """Extract the seed points from the input.

    Args:
        seeds (:obj:`numpy.ndarray` or :obj:`list` of :obj:`ImageD11.grain.grain`):
            List of grains or a numpy array of shape=(N,3) specifying the seed
            coordinates of the voroni tesselation.

    Returns:
        :obj:`numpy.ndarray`: Numpy array of shape=(N,3) specifying the seeds
    """
    if isinstance(seeds[0], ImageD11.grain.grain):
        return np.array([g.translation for g in seeds])
    elif isinstance(seeds, np.ndarray) and seeds.ndim == 2 and seeds.shape[1] == 3:
        return seeds
    else:
        raise ValueError("seeds must be a numpy array or a list of grains")


def voronoi(seeds, bounds=None):
    """Voroni Tesselate a set of seeds into a 3D voxel volume.

    The returned mesh is a 3D voxel volume of the voroni tesselation of the seeds.
    It will contain a series of polyhedra, one for each seed.

    Args:
        seeds (:obj:`numpy.ndarray` or :obj:`list` of :obj:`ImageD11.grain.grain`):
            List of grains or a numpy array of shape=(N,3) specifying the seed
            coordinates of the voroni tesselation.
        bounds (:obj:`list` of :obj:`tuple`, optional): Bounds for the tesselation
            as a tuple of 3 numpy arrays specifying the limits of the tesselation in x, y, z
            i.e bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]. Defaults to None, in which
            case the bounds are inferred from the seeds as the minimum and maximum values in each
            respective dimension padded by 1e-8 in each dimension.

    Returns:
        :obj:`meshio.Mesh`: Mesh object of the tesselation with cell data as grain_id and surface_grain.
            see meshi documentation for more details: https://github.com/nschloe/meshio

    """
    points = _extract_points(seeds)
    bounds = _extract_bounds(points, bounds)

    dmap, nmap = _get_plane_maps(points)
    vertices, simplices = [], []
    grain_id, surface_grain = [], []
    neighbours = np.empty(dtype=np.ndarray, shape=len(points))
    grain_volumes = []
    ms = 0
    for i in range(len(points)):
        halfspaces = _get_halfspaces(points[i], dmap[i], nmap[i], bounds)
        hs = HalfspaceIntersection(halfspaces, interior_point=np.array([0, 0, 0]))
        hull = ConvexHull(hs.intersections)
        verts = hs.intersections + points[i]
        simps = hull.simplices
        neigh = hs.dual_vertices + (hs.dual_vertices >= i)
        neigh = neigh[neigh < len(points)]
        neighbours[i] = neigh
        vertices.append(verts)
        simplices.append(simps + ms)
        ms += np.max(simps) + 1
        grain_id.extend([i] * len(simps))
        grain_volumes.extend([hull.volume] * len(simps))
        surface_grain.extend([_is_on_boundary(verts, bounds)] * len(simps))
    mesh = _build_mesh(
        np.concatenate(vertices),
        np.concatenate(simplices),
        grain_id,
        grain_volumes,
        surface_grain,
        neighbours,
    )

    return mesh


if __name__ == "__main__":
    pass
