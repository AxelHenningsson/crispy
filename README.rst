crispy - a CRystal population metric InSpector written in PYthon.
=====================================================================

crispy is a Python package for analyzing crystal population metrics from 3DXRD and
lab-DCT data targeting interfacing between diffraction contrast modalities.

One of the central features of crispy is the provision of per-grain reflections
diffraction information for Dark-Field X-ray Microscopy (DFXM) by analyzing
3DXRD grain maps and lab-DCT grain volumes.

Interfaces for reading, analyzing, and visualizing grain maps are provided for

    * **3DXRD grain maps** (3D scatter of grain centroids and grain orientations)

    * **lab-DCT voxel volumes** (3D voxel volumes where each voxel is a single crystal grain)

Usecase
=====================================================================

We can load a grain map from disc as

.. code-block:: python

    import crispy

    # this could be a 3DXRD grain map or a lab-DCT grain volume
    path_to_my_grain_map = crispy.assets.path.AL1050
    grain_map = crispy.GrainMap( path_to_my_grain_map )

In this example we have a lab-DCT grain volume, we do some filtering to remove grains that are not of interest

.. code-block:: python

    grain_map.filter( min_grain_size_in_voxels = 200 )
    grain_map.prune_boundary_grains()

Many more operations for manipulating grain maps are available -- check out the docs!

We can now write the grain map to a file to  visualize in paraview

.. code-block:: python

    grain_map.write("grain_map.xdmf")


.. image:: ../../docs/source/images/readme_grains.png

To search for accessible reflections in DFXM mode for ``eta=0``, we can use the :obj:`crispy.dfxm.Goniometer` class as

.. code-block:: python

    motor_bounds = {
    "mu": (0, 20), # degrees
    "omega": (-30, 30), # degrees
    "chi": (-7, 7), # degrees
    "phi": (-7, 7), # degrees
    "detector_z": (-0.04, 1.96), # metres
    "detector_y": (-0.169, 1.16) # metres
    }

    goniometer = crispy.dfxm.Goniometer(grain_map,
                            energy=17,
                            detector_distance=4,
                            motor_bounds=motor_bounds)

    goniometer.find_reflections()

The resulting reflections can be accessed as

.. code-block:: python

    polycrystal.grains[i].dfxm

Providing a dictionary with refleciton information for each grain.

.. code-block:: python

    {'hkl': array([[0.],
            [0.],
            [2.]]),
    'mu': array([12.7388667]),
    'omega': array([1.78967645]),
    'chi': array([-4.23236192]),
    'phi': array([-2.77428247]),
    'residual': array([0.]),
    'theta': array([10.37543294])}

Alternatively, we can generate a :obj:`pandas.DataFrame` with reflection information for all grains as

.. code-block:: python

    df = goniometer.table_of_reflections()


.. image:: ../../docs/source/images/readme_df.png


It is also possible to load a 3DXRD grain map from a file tesselate and visualize.

.. code-block:: python

    path_to_my_grain_map = crispy.assets.path.FEAU
    grain_map = crispy.GrainMap( path_to_my_grain_map )
    grain_map.tesselate()
    grain_map.colorize( np.eye(3) )
    crispy.visualize.mesh( grain_map )

.. image:: ../../docs/source/images/readme_tdxrd.png


Installation
=====================================================================

To install ``crispy`` from source, run

.. code-block:: bash

    git clone https://github.com/AxelHenningsson/crispy.git
    cd crispy
    pip install -e .


Documentation
=====================================================================

The extended documentation is about to be released at an externally hosted website.