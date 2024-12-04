import numpy as np
import matplotlib.pyplot as plt 
import h5py
import hdf5plugin
import ImageD11.grain

def grains(filename, group_name='grains'):
    return ImageD11.grain.read_grain_file_h5(filename, group_name)

if __name__=='__main__':
    pass