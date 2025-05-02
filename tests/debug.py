# %%
import crispy

# this could be a 3DXRD grain map or a lab-DCT grain volume
path_to_my_grain_map = crispy.assets.path.AL1050
grain_map = crispy.GrainMap( path_to_my_grain_map )

# %%
grain_map.filter( min_grain_size_in_voxels = 200 )
grain_map.prune_boundary_grains()

# %%
grain_map.write("grain_map.xdmf")

# %%
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
grain_map.grains[0].dfxm

# %%
df = goniometer.table_of_reflections()
df
# %%
import matplotlib.pyplot as plt
fontsize = 22
ticksize= 22
plt.rcParams['font.size'] = fontsize
plt.rcParams['xtick.labelsize'] = ticksize
plt.rcParams['ytick.labelsize'] = ticksize
plt.rcParams['font.family'] = 'Times New Roman'
plt.style.use('dark_background')

# %%
import numpy as np

path_to_my_grain_map = crispy.assets.path.FEAU
print(crispy.assets.path.FEAU)
grain_map = crispy.GrainMap( path_to_my_grain_map, 
                            group_name='Fe', 
                            lattice_parameters=[2.8665, 2.8665, 4.504, 90, 90, 120],
                            symmetry=225)
grain_map.tesselate()
grain_map.colorize( np.array([[0, 1, 0]]) )
crispy.vizualise.mesh( grain_map )