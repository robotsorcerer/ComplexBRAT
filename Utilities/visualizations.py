__all__ = ["visualize_init_avoid_tube"]

import matplotlib.pyplot as plt
from LevelSetPy.Grids import createGrid
import numpy as np

def visualize_init_avoid_tube(flock, save=True, fname=None, title=''):
	"""
		For a flock, whose mesh has been precomputed, 
		visualize the initial backward avoid tube.
	"""
	# visualize avoid set 
	fontdict = {'fontsize':16, 'fontweight':'bold'}
	mesh_bundle = flock.mesh_bundle

	fig = plt.figure(1, figsize=(16,9), dpi=100)
	ax = plt.subplot(111, projection='3d')
	ax.add_collection3d(mesh_bundle.mesh)


	xlim = (mesh_bundle.verts[:, 0].min(), mesh_bundle.verts[:,0].max())
	ylim = (mesh_bundle.verts[:, 1].min(), mesh_bundle.verts[:,1].max())
	zlim = (mesh_bundle.verts[:, 2].min(), mesh_bundle.verts[:,2].max())

	# create grid that contains just this zero-level set to avoid computational craze 
	gmin = np.asarray([[xlim[0], ylim[0], zlim[0]]]).T
	gmax = np.asarray([[xlim[1], ylim[1], zlim[1]] ]).T

	# create reduced grid upon which this zero level set dwells
	flock.grid_zero = createGrid(gmin, gmax, flock.grid.N, 2)

	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	ax.set_zlim(zlim)

	ax.grid('on')
	ax.tick_params(axis='both', which='major', labelsize=10)

	ax.set_xlabel(rf'x$_1^{flock.label}$ (m)', fontdict=fontdict)
	ax.set_ylabel(rf'x$_2^{flock.label}$ (m)', fontdict=fontdict)
	ax.set_zlabel(rf'$\omega^{flock.label} (rad)$',fontdict=fontdict)

	if title:
		ax.set_title(title, fontdict=fontdict)
	else:
		ax.set_title(f'Flock {flock.label}\'s ({flock.N} Agents) Payoff.', fontdict=fontdict)
	ax.view_init(azim=-45, elev=30)

	if save:
		fig.savefig(fname, bbox_inches='tight',facecolor='None')