import h5py
import time
import sys, os
import argparse
import numpy as np
sys.path.append('../')

from os.path import join
import matplotlib.pyplot as plt
from LevelSetPy.Grids import createGrid
from LevelSetPy.Visualization import implicit_mesh
from LevelSetPy.Visualization.color_utils import cm_colors

parser = argparse.ArgumentParser(description='Visualization')
parser.add_argument('--flock_num', '-fn', type=int, default=0, help='Which flock\'s brat to optimize?' )
parser.add_argument('--silent', '-si', action='store_false', help='silent debug print outs' )
parser.add_argument('--fname', '-fn', type=str, help='which BRAT to load?' )
args = parser.parse_args()
args.verbose = True if not args.silent else False

save = True
flname = "Flock_1"
base_path = "/opt/flock1"
fname = 'opt/murmurations_flock1_02-04-22_00-35.hdf5'

# recreate spacing
gmin = np.asarray([[-1.5, -1.5, -np.pi]]).T
gmax = np.asarray([[1.5, 1.5, np.pi] ]).T
grid = createGrid(gmin, gmax, 101, 2)
spacing = tuple(grid.dx.flatten().tolist())

def see(n, obj):
    keys = []
    for k, v in obj.attrs.items():
        keys.append((k,v))
    return keys

verbose = False

fontdict = {'fontsize':16, 'fontweight':'bold'}
plt.ion()
fig = plt.figure(1, figsize=(16,9), dpi=100)
ax = plt.subplot(111, projection='3d')

with h5py.File(fname, 'r+') as df:
    if verbose:
        df.visititems(see)

    value_key = [k for k in df.keys()][0]
    keys = [key for key in df[value_key]]

    color_len = 15
    colors = [iter(plt.cm.Spectral(np.linspace(0, 1, color_len))),
              iter(plt.cm.ocean(np.linspace(0, 1, color_len))),
              iter(plt.cm.viridis(np.linspace(0, 1, color_len))),
              iter(plt.cm.rainbow(np.linspace(0, 1, color_len))),
              iter(plt.cm.coolwarm(np.linspace(0, 1, color_len))),
              iter(plt.cm.magma(np.linspace(0, 1, color_len))),
              iter(plt.cm.YlOrRd(np.linspace(0, 1, color_len))),
              iter(plt.cm.rainbow(np.linspace(0, 1, color_len))),
              iter(plt.cm.summer(np.linspace(0, 1, color_len))),
              iter(plt.cm.nipy_spectral(np.linspace(0, 1, color_len))),
              iter(plt.cm.autumn(np.linspace(0, 1, color_len))),
              iter(plt.cm.PuRd(np.linspace(0, 1, color_len))),
              iter(plt.cm.twilight(np.linspace(0, 1, color_len))),
              iter(plt.cm.inferno(np.linspace(0, 1, color_len))),
              iter(plt.cm.copper(np.linspace(0, 1, color_len))),
              iter(plt.cm.cubehelix(np.linspace(0, 1, color_len))),
              ]
    idx = 0
    # load them brats for a flock
    for key in keys:
        brt = np.asarray(df[f"{value_key}/{key}"])


        # if idx%color_len==0:
        #     color_iter = colors[idx]
        mesh_bundle=implicit_mesh(brt, level=0, spacing=spacing, edge_color=None, \
                                 face_color=[.3,.4,.8])

        ax.grid('on')
        plt.cla()

        ax.add_collection3d(mesh_bundle.mesh)
        xlim = (mesh_bundle.verts[:, 0].min(), mesh_bundle.verts[:,0].max())
        ylim = (mesh_bundle.verts[:, 1].min(), mesh_bundle.verts[:,1].max())
        zlim = (mesh_bundle.verts[:, 2].min(), mesh_bundle.verts[:,2].max())

        ax.set_xlim3d(*xlim)
        ax.set_ylim3d(*ylim)
        ax.set_zlim3d(*zlim)

        ax.tick_params(axis='both', which='major', labelsize=10)

        ax.set_xlabel(rf'x$_1^{1}$ (m)', fontdict=fontdict)
        ax.set_ylabel(rf'x$_2^{1}$ (m)', fontdict=fontdict)
        ax.set_zlabel(rf'$\omega^{1} (deg)$',fontdict=fontdict)

        time_step = float(key.split(sep="_")[-1])#*1000
        # print('timestep: ', time_step)
        ax.set_title(f'{flname}\'s BRAT at {time_step} secs.', fontdict=fontdict)
        ax.view_init(azim=-30, elev=30)

        fig.canvas.draw()
        fig.canvas.flush_events()

        if save:
            lname = fname.split(sep="_")[0].split(sep="/")[-1]
            fig.savefig(join(base_path, rf"{lname}_{idx:0>4}.jpg"), \
                                bbox_inches='tight',facecolor='None')

        time.sleep(1e-5)

        idx+= 1

# do this so pyplot doesn't close abruptly in the end
plt.show()

plt.ioff()
