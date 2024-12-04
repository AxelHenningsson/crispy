import warnings

import ImageD11.grain
import numpy as np
from numba import njit

def mesh_voronoi():
    # TODO: this is a draft
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    import meshio


    np.random.seed(42)  # For reproducibility
    # points = np.array([[0.3,0.3,0.3], [-0.3,-0.3,-0.3]])
    points = np.random.rand(10000, 3) - 0.5
    points[:, 2] *= 2

    xmin, xmax = -0.5, 0.5
    ymin, ymax = -0.5, 0.5
    zmin, zmax = -1, 1

    import cProfile
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    dmap = []
    nmap = []
    for i in range(points.shape[0]):
        x1 = points[i]
        v = points - x1
        norm = np.linalg.norm(v, axis=1)
        m = norm != 0
        norm = norm[m]
        v = v[m]
        dmap.append(norm)
        nmap.append(v / norm[:, np.newaxis])
    dmap = np.array(dmap)
    nmap = np.array(nmap)

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    # Stacked Inequalities of the form Ax + b <= 0 in format [A; b]

    simplices = []
    vertices = []
    grain_id = []
    surface_grain = []
    ms = 0
    for i in range(len(points)):
        A = nmap[i]
        b = dmap[i] / 2.0
        x, y, z = points[i]

        limits = np.array(
            [
                [1, 0, 0, np.abs(xmax) - x],
                [-1, 0, 0, np.abs(xmin) + x],
                [0, 1, 0, np.abs(ymin) - y],
                [0, -1, 0, np.abs(ymax) + y],
                [0, 0, 1, np.abs(zmin) - z],
                [0, 0, -1, np.abs(zmax) + z],
            ]
        )

        limits[:, -1] *= -1
        halfspaces = np.hstack((A, -b.reshape(len(b), 1)))
        halfspaces = np.vstack((halfspaces, limits))
        interior_point = np.array([0, 0, 0])

        hs = HalfspaceIntersection(halfspaces, interior_point)
        hull = ConvexHull(hs.intersections)

        verts = hs.intersections + points[i]
        vertices.append(verts)
        simplices.append(hull.simplices + ms)
        ms += np.max(hull.simplices) + 1

        grain_id.extend([i] * len(hull.simplices))

        if (
            np.abs(verts.T[0].max() - xmax) < 1e-8
            or np.abs(verts.T[1].max() - ymax) < 1e-8
            or np.abs(verts.T[2].max() - zmax) < 1e-8
            or np.abs(verts.T[0].min() - xmin) < 1e-8
            or np.abs(verts.T[1].min() - ymin) < 1e-8
            or np.abs(verts.T[2].min() - zmin) < 1e-8
        ):
            surface_grain.extend([1] * len(hull.simplices))
        else:
            surface_grain.extend([0] * len(hull.simplices))

    vertices = np.concatenate(vertices)
    simplices = np.concatenate(simplices)

    cells = [("polygon", simplices)]
    mesh = meshio.Mesh(
        vertices,
        cells=cells,
        cell_data={"grain_id": [grain_id], "surface_grain": [surface_grain]},
    )
    mesh.write("myhull.vtk")


def voroni(seeds, coord):
    """Voroni Tesselate a set of seeds into a 3D voxel volume.

    Args:
        seeds (:obj:`numpy array` or :obj:`list` of :obj:`ImageD11.grain.grain`):
            List of grains or a numpy array of shape=(N,3) specifying the seed
            coordinates of the voroni tesselation.
        coord (:obj:`tuple` of :obj:`numpy array`): tuple of x,y,z 1D numpy
            coordinate arrays which define the mesh of voxels that will be
            deployed during voroni tessselation.

    Returns:
        :obj:`numpy array`: The tesselated voxel volume of shape=(len(x), len(y), len(z))
            of type uint16. Each value in the array corresponds to the index in of the
            corresponding seed.
    """
    if isinstance(seeds[0], ImageD11.grain.grain):
        seed_points = np.array([g.translation for g in seeds])
    elif isinstance(seeds, np.ndarray):
        seed_points = seeds
    else:
        raise ValueError("seeds must be ImageD11.grain.grain or nump.ndarray")

    seed_labels = np.arange(0, seed_points.shape[0], dtype=np.int16)

    X, Y, Z = np.meshgrid(*coord, indexing="ij")
    tesselation = label_volume(X, Y, Z, seed_labels, seed_points)

    if len(np.unique(tesselation)) != len(seed_labels):
        warnings.warn("Tesselation resolution is too coase to capture all grains!")

    return tesselation

@njit
def label_volume(X, Y, Z, seed_labels, seed_points):
    """Do the actual tesselation."""
    labeled_volume = np.ones(X.shape, dtype=np.uint16)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                x = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                a = seed_points - x
                labeled_volume[i, j, k] = seed_labels[np.argmin(np.sum(a * a, axis=1))]
    return labeled_volume

def _tesselate():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull, HalfspaceIntersection
    import meshio

    if __name__ == "__main__":

        np.random.seed(42)  # For reproducibility
        # points = np.array([[0.3,0.3,0.3], [-0.3,-0.3,-0.3]])
        points = np.random.rand(10000, 3) - 0.5
        points[:, 2] *= 2

        xmin, xmax = -0.5, 0.5
        ymin, ymax = -0.5, 0.5
        zmin, zmax = -1, 1

        import cProfile
        import pstats
        import time

        pr = cProfile.Profile()
        pr.enable()
        t1 = time.perf_counter()

        dmap = []
        nmap = []
        for i in range(points.shape[0]):
            x1 = points[i]
            v = points - x1
            norm = np.linalg.norm(v, axis=1)
            m = norm != 0
            norm = norm[m]
            v = v[m]
            dmap.append(norm)
            nmap.append(v / norm[:, np.newaxis])
        dmap = np.array(dmap)
        nmap = np.array(nmap)

        t2 = time.perf_counter()
        pr.disable()
        pr.dump_stats("tmp_profile_dump")
        ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
        ps.print_stats(15)
        print("\n\nCPU time is : ", t2 - t1, "s")

        # Stacked Inequalities of the form Ax + b <= 0 in format [A; b]

        simplices = []
        vertices = []
        grain_id = []
        surface_grain = []
        ms = 0
        for i in range(len(points)):
            A = nmap[i]
            b = dmap[i] / 2.0
            x, y, z = points[i]

            limits = np.array(
                [
                    [1, 0, 0, np.abs(xmax) - x],
                    [-1, 0, 0, np.abs(xmin) + x],
                    [0, 1, 0, np.abs(ymin) - y],
                    [0, -1, 0, np.abs(ymax) + y],
                    [0, 0, 1, np.abs(zmin) - z],
                    [0, 0, -1, np.abs(zmax) + z],
                ]
            )

            limits[:, -1] *= -1
            halfspaces = np.hstack((A, -b.reshape(len(b), 1)))
            halfspaces = np.vstack((halfspaces, limits))
            interior_point = np.array([0, 0, 0])

            hs = HalfspaceIntersection(halfspaces, interior_point)
            hull = ConvexHull(hs.intersections)

            verts = hs.intersections + points[i]
            vertices.append(verts)
            simplices.append(hull.simplices + ms)
            ms += np.max(hull.simplices) + 1

            grain_id.extend([i] * len(hull.simplices))

            if (
                np.abs(verts.T[0].max() - xmax) < 1e-8
                or np.abs(verts.T[1].max() - ymax) < 1e-8
                or np.abs(verts.T[2].max() - zmax) < 1e-8
                or np.abs(verts.T[0].min() - xmin) < 1e-8
                or np.abs(verts.T[1].min() - ymin) < 1e-8
                or np.abs(verts.T[2].min() - zmin) < 1e-8
            ):
                surface_grain.extend([1] * len(hull.simplices))
            else:
                surface_grain.extend([0] * len(hull.simplices))

        vertices = np.concatenate(vertices)
        simplices = np.concatenate(simplices)

        cells = [("polygon", simplices)]
        mesh = meshio.Mesh(
            vertices,
            cells=cells,
            cell_data={"grain_id": [grain_id], "surface_grain": [surface_grain]},
        )
        mesh.write("myhull.vtk")



        
if __name__ == "__main__":

    seeds = np.random.rand(16, 3)
    print(seeds)
    # seeds = np.array([[0,0,0],[0.5,0.5,1]])
    x, y, z = seeds.T
    xg = np.linspace(0, 1, 32)
    yg = np.linspace(0, 1, 32)
    zg = np.linspace(0, 1, 32)

    from scipy.spatial import Voronoi



    import cProfile
    import pstats
    import time

    import crispy

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    #tessmap = crispy.tesselate.voroni(seeds, (xg, yg, zg))
    vor = Voronoi(seeds)
    vertices = vor.vertices

    
    print(vertices)


    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    crispy.vizualise.save_voxels("test", tessmap, (xg, yg, zg))
