#  Roadmap for crispy

The goal of crispy is to provide a simple and easy to use API for 

    (a) loading a set of pointlike grains
        (1) From a grain file (ImageD11)
        (2) From a scattered numpy data

    (b) Tesselating them into a mesh
        (1) By Voronoi tesselation

    (c) Rendering them various ways
        (1) By paraview loading (3D)
        (2) By matplotlib cuts (2D)
        (3) By matplotlib scatters (3d overwiew)

    (d) Computing inherent-local mesh properties
        (1) Neighbours (this is the key one)
        (2) Misorienations between neighbours
        (3) Aspect ratios (x,y,z bounds)
        (4) Volumes
        (5) Surface areas

    (e) Computing diffraction mesh properties
        (1) translations needed to bring to beam center
        (2) hkl-dependent rotations needed to diffract,
            - omega, chi, phi, mu
        (3) Detector/diffraction positions
            - tth, eta, y, z

# Goal for today

*Implement inherent-local mesh properties*
- [ ] Neighbours (this is the key one)
- [ ] Misorienations between neighbours
- [ ] Aspect ratios (x,y,z bounds)
- [ ] Volumes
- [ ] Surface areas

*Computing diffraction mesh properties*
- [ ] hkl-dependent rotations needed to diffract
