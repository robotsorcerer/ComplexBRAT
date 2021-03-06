import h5py
import time
import glob
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
parser.add_argument('--silent', '-si', action='store_true', help='silent debug print outs' )
parser.add_argument('--delete', '-dt', action='store_true', help='silent debug print outs' )
parser.add_argument('--fname', '-fn', type=str, default='murmurations_flock_01_02-06-22_17-43.hdf5', help='which BRAT to load?' )
parser.add_argument('--start', '-st', type=int, default=1, help='what key in the index to resume from' )
parser.add_argument('--end', '-ed', type=int, help='what key in the index to resume from' )
args = parser.parse_args()
args.verbose = True if not args.silent else False
verbose = not args.silent

save = True
base_path = "/opt/murmurations/"
fname = join(base_path, args.fname)

fontdict = {'fontsize':18, 'fontweight':'bold'}

def visualize_brats(spacing, value, cidx=0):
    colors = [ 
                plt.cm.Spectral(.9), plt.cm.rainbow(.9),
                plt.cm.ocean(.8), plt.cm.cubehelix(.8),
                plt.cm.viridis(.2), plt.cm.coolwarm(.8),
                plt.cm.magma(.8), plt.cm.summer(.8),
                plt.cm.nipy_spectral(.8), plt.cm.autumn(.8),
                plt.cm.twilight(.8), plt.cm.inferno_r(.8),
                plt.cm.copper(.8)
                ]        
    color = colors[cidx]

    mesh_bundle=implicit_mesh(value, level=0, spacing=spacing, edge_color=None, \
                                 face_color=color)
    
    plt.ion()
    fig = plt.figure(1, figsize=(25,16), dpi=100)
    ax = plt.subplot(111, projection='3d')

    ax.add_collection3d(mesh_bundle.mesh)
    xlim = (mesh_bundle.verts[:, 0].min(), mesh_bundle.verts[:,0].max())
    ylim = (mesh_bundle.verts[:, 1].min(), mesh_bundle.verts[:,1].max())
    zlim = (mesh_bundle.verts[:, 2].min(), mesh_bundle.verts[:,2].max())

    ax.set_xlim3d(*xlim)
    ax.set_ylim3d(*ylim)
    ax.set_zlim3d(*zlim)

    ax.tick_params(axis='both', which='major', labelsize=10)

    ax.set_xlabel(rf'x$_1^{int("01")}$ (m)', fontdict=fontdict)
    ax.set_ylabel(rf'x$_2^{int("01")}$ (m)', fontdict=fontdict)
    ax.set_zlabel(rf'$\omega^{int("01")} (deg)$',fontdict=fontdict)

    ax.set_title(f'Flock {int(lname)}\'s BRAT.', fontdict=fontdict)
    azim=60 if int(lname)%2==0 else -30
    if lname=='00': azim=-75
    ax.view_init(azim=azim, elev=30)

    if save:
        savepath=join(base_path, rf"flock_{lname}")
        os.makedirs(savepath) if not os.path.exists(savepath) else None
        fig.savefig(join(savepath, rf"{idx:0>4}.jpg"), bbox_inches='tight',facecolor='None')

    fig.canvas.draw()
    fig.canvas.flush_events()

def see(n, obj):
    keys = []
    for k, v in obj.attrs.items():
        keys.append((k,v))
    return keys

with h5py.File(fname, 'r+') as df:
    if verbose:
        df.visititems(see)

    value_key = [k for k in df.keys()][0]
    keys = [key for key in df[value_key]]
    if 'spacing' in value_key:
        spacing = np.asarray(df["value/spacing"])
    else:
        gmin = np.asarray([[-1.5, -1.5, -np.pi]]).T
        gmax = np.asarray([[1.5, 1.5, np.pi] ]).T
        grid = createGrid(gmin, gmax, 101, 2)
        spacing = grid.dx.flatten()        
    spacing = tuple(spacing.tolist())

    print(f"Num BRATs in this flock: {len(keys)}")
            
    lname = fname.split(sep="_")[2]
    cidx = int(lname)

    if not args.end:
        args.end = -1

    idx = args.start-1

    if args.delete:
        oldfiles = glob.glob(join(base_path, rf"flock_{lname}", '*.*'))
        for f in oldfiles:  os.remove(f) 

    # load them brats for a flock
    for key in keys[args.start:args.end]:
        brt = np.asarray(df[f"{value_key}/{key}"])
        print(f"On BRAT: {idx+1}/{len(keys[:args.end])}--{len(keys)}")

        visualize_brats(spacing, brt, cidx)
        time_step = float(key.split(sep="_")[-1])
        

        time.sleep(1e-5)

        idx+= 1

# do this so pyplot doesn't close abruptly at the end
plt.show()
plt.ioff()
