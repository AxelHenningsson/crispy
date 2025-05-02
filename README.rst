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

Installation
------------

To install ``crispy`` from source, run

.. code-block:: bash

    git clone https://github.com/AxelHenningsson/crispy.git
    cd crispy
    pip install -e .




