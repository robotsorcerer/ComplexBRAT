__comment__     = "Solves the Complex Backward Reach Avoid Tubes for a Murmuration of Starlings."
__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux License"
__comment__ 	= "Evader at origin"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"
__date__ 		= "Nov. 2021"

import copy
import time
import h5py
import logging
import argparse
import sys, os
import random
import cupy as cp
import numpy  as np
from math import pi
import numpy.linalg as LA
import matplotlib as mpl
# mpl.use('Qt5Agg')
import matplotlib.pyplot as plt

from os.path import abspath, join, dirname, expanduser
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append(abspath(join('..')))

from Libs import *

from LevelSetPy.Grids import *
from LevelSetPy.DynamicalSystems import *
from LevelSetPy.Utilities import *
from LevelSetPy.Visualization import *
from LevelSetPy.BoundaryCondition import *
from LevelSetPy.InitialConditions import *


from BRATVisualization.rcbrt_visu import RCBRTVisualizer

parser = argparse.ArgumentParser(description='Hamilton-Jacobi Analysis')
parser.add_argument('--silent', '-si', action='store_false', help='silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_false', help='save BRS/BRT at end of sim' )
parser.add_argument('--visualize', '-vz', action='store_false', default=True, help='visualize level sets?' )
parser.add_argument('--show_flock_payoff', '-sp', action='store_false', default=False, help='visualize level sets?' )
parser.add_argument('--load_brt', '-lb', action='store_true', help='load saved brt?' )
parser.add_argument('--stochastic', '-st', action='store_true', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--compute_traj', '-ct', action='store_false', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--verify', '-vf', action='store_true', default=True, help='visualize level sets?' )
parser.add_argument('--elevation', '-el', type=float, default=5., help='elevation angle for target set plot.' )
parser.add_argument('--direction', '-dr',  action='store_true',  help='direction to grow the level sets. Negative by default.' )
parser.add_argument('--azimuth', '-az', type=float, default=10., help='azimuth angle for target set plot.' )
parser.add_argument('--pause_time', '-pz', type=float, default=.3, help='pause time between successive updates of plots' )
args = parser.parse_args()
args.verbose = True if not args.silent else False

print(f'args:  {args}')

if not args.silent:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

# Turn off pyplot's spurious dumps on screen
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

u_bound = 1
w_bound = deg2rad(45)
fontdict = {'fontsize':16, 'fontweight':'bold'}

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
		plt.savefig(fname, bbox_inches='tight',facecolor='None')

def get_avoid_brt(flock, compute_mesh=True, color='crimson'):
	"""
		Get the avoid BRT for this flock. That is, every bird 
		within a flock must avoid one another.

		Parameters:
		==========
		.flock: This flock of vehicles.
		.compute_mesh: compute mesh of local payoffs for each bird in this flock?
	"""
	for vehicle in flock.vehicles:
		# make the radius of the target setthe turn radius of this vehicle
		vehicle.payoff = shapeCylinder(flock.grid, 2, center=flock.update_state(vehicle.cur_state), \
										radius=vehicle.payoff_width)
	"""
		Now compute the overall payoff for the flock
	   	by taking a union of all the avoid sets.
	"""
	flock.payoff = shapeUnion([veh.payoff for veh in flock.vehicles])
	if compute_mesh:
		spacing=tuple(flock.grid.dx.flatten().tolist())
		flock.mesh_bundle = implicit_mesh(flock.payoff, 0, spacing,edge_color=None, face_color=color)    
		
	return flock 


def get_flock(gmin, gmax, num_points, num_agents, init_xyzs, label,\
				periodic_dims=2, reach_rad=.2, avoid_rad=.3,
				base_path='', save=True, color='blue'):
	"""
		Params
		======
		gmin: minimum bounds of the grid
		gmax: maximum bounds of the grid
		num_points: number of points on the grid
		num_agents: number of agents in this flock
		init_xyzs: initial positions of the birds in this flock
		label: label of this flock among all flocks
		periodic_dims: periodic dimensions
		reach_rad: reach for external disturbance
		avoid_rad: avoid radius for topological interactions
	"""
	global u_bound, w_bound

	assert gmin.ndim==2, 'gmin must be of at least 2 dims'
	assert gmax.ndim==2, 'gmax must be of at least 2 dims'

	gmin = to_column_mat(gmin)
	gmax = to_column_mat(gmax)

	grid = createGrid(gmin, gmax, num_points, periodic_dims)
	
	vehicles = [Bird(grid, u_bound, w_bound, np.expand_dims(init_xyzs[i], 1) , random.random(), \
					center=np.zeros((3,1)), neigh_rad=3, label=i+1, init_random=False) \
					for i in range(num_agents)]                
	flock = Flock(grid, vehicles, label=label, reach_rad=.2, avoid_rad=.3)
	get_avoid_brt(flock, compute_mesh=True, color=color)

	if args.visualize and args.show_flock_payoff:
		visualize_init_avoid_tube(flock, save, fname=join(base_path, f"flock_{flock.label}.jpg"))         
		plt.show() 

	return flock


def main(args):
	# global params
	gmin = np.asarray([[-1.5, -1.5, -np.pi]]).T
	gmax = np.asarray([[1.5, 1.5, np.pi] ]).T
	num_agents = 7

	H         = 0.4
	H_STEP    = 0.05
	neigh_rad = 0.4
	reach_rad = .2

	# Please note the way I formulate the initial states here. Linear speed is constant but heading is different.
	INIT_XYZS = np.array([[neigh_rad*np.cos((i/6)*np.pi/4), neigh_rad*np.sin((i/6)*np.pi/4), H+i*H_STEP] for i in range(num_agents)])

	# color thingy
	color = iter(plt.cm.inferno_r(np.linspace(.25, 1, num_agents)))

	# save shenanigans
	base_path = join(expanduser("~"), "Documents/Papers/Safety/WAFR2022/figures")
	flock0 = get_flock(gmin, gmax, 101, num_agents, INIT_XYZS, label=1, periodic_dims=2, \
						reach_rad=.2, avoid_rad=.3, base_path=base_path, color=next(color))	
	flock1 = get_flock(gmin, gmax, 101, num_agents-1, 1.1*INIT_XYZS, label=2,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3, base_path=base_path, color=next(color))	           
	flock2 = get_flock(gmin, gmax, 101, num_agents, -1.1*INIT_XYZS, label=3,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3, base_path=base_path, color=next(color))	                         
	flock3 = get_flock(gmin, gmax, 101, num_agents-1, 1.5*INIT_XYZS, label=4,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3, base_path=base_path, color=next(color))	                             
	flock4 = get_flock(gmin, gmax, 101, num_agents, -1.5*INIT_XYZS, label=5,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3, base_path=base_path, color=next(color))	                
	flock5 = get_flock(gmin, gmax, 101, num_agents-1, 2.0*INIT_XYZS, label=6,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3, base_path=base_path, color=next(color))	          
	flock6 = get_flock(gmin, gmax, 101, num_agents, -1.8*INIT_XYZS, label=7,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3, base_path=base_path, color=next(color))	

	# after creating value function, make state space cupy objects
	g = flock1.grid
	g.xs = [cp.asarray(x) for x in g.xs]
	finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
				innerData = Bundle({'grid':g,
					'hamFunc': flock1.hamiltonian,
					'partialFunc': flock1.dissipation,
					'dissFunc': artificialDissipationGLF,
					'CoStateCalc': upwindFirstENO2,
					}),
					positive = False,  # direction to grow the updated level set
				))

	# Visualization paramters
	spacing = tuple(g.dx.flatten().tolist())
	params = Bundle(
					{"grid": g,
					 'disp': True,
					 'labelsize': 16,
					 'labels': "Initial 0-LevelSet",
					 'linewidth': 2,
					 'elevation': 10,
					 'azimuth': 5,
					 'mesh': flock0.mesh_bundle,
					 'pause_time': args.pause_time,
					 'title': f'Initial BRT. Flock with {flock1.N} agents.',
					 'level': 0, # which level set to visualize
					 'winsize': (16,9),
					 'fontdict': {'fontsize':18, 'fontweight':'bold'},
					 "savedict": Bundle({"save": True,
									"savename": "murmur",
									"savepath": join(expanduser("~"),
									"Documents/Papers/Safety/WAFR2022/figures/")
								 })
					})

	if args.load_brt:
		args.save = False
		brt = np.load("data/murmurations.npz")
	else:
		if args.visualize:
			viz = RCBRTVisualizer(params=params)
		t_range = [0, 50]
		t_plot = (t_range[1] - t_range[0]) / t_range[-1] 
		small = 100*eps

		# Loop through t_range (subject to a little roundoff).
		t_now = t_range[0]
		start_time = cputime()
		itr_start = cp.cuda.Event()
		itr_end = cp.cuda.Event()

		brt = [flock0.payoff]
		meshes, brt_time = [], []
		value_rolling = cp.asarray(copy.copy(flock1.payoff))

		colors = iter(plt.cm.ocean(np.linspace(.25, 2, 100)))
		color = next(colors)
		options = Bundle(dict(factorCFL=0.7, stats='on', singleStep='on'))
		
		idx = 0
		while(t_range[1] - t_now > small * t_range[1]):
			itr_start.record()
			cpu_start = cputime()
			time_step = f"{t_now}/{t_range[-1]}"

			# Reshape data array into column vector for ode solver call.
			y0 = value_rolling.flatten()

			# How far to step?
			t_span = np.hstack([ t_now, min(t_range[1], t_now + t_plot) ])

			# Integrate a timestep.
			t, y, _ = odeCFL2(termRestrictUpdate, t_span, y0, odeCFLset(options), finite_diff_data)
			cp.cuda.Stream.null.synchronize()
			t_now = t

			# Get back the correctly shaped data array
			value_rolling = y.reshape(g.shape)

			if args.visualize:
				value_rolling_np = value_rolling.get()
				mesh_bundle=implicit_mesh(value_rolling_np, level=0, spacing=spacing,
									edge_color=None,  face_color=color)
				viz.update_tube(mesh_bundle, time_step, True)

			if args.save:
				if args.visualize:
					fig = plt.gcf()
					fig.savefig(join(base_path,
						rf"murmurations_{t_now}.jpg"), bbox_inches='tight',facecolor='None')
				
				# save this brt
				savename = join("data/", rf"murmurations.hdf5")
				if os.path.exists(savename):
					os.remove(savename)

				with h5py.File(savename, 'a') as h5file:
					h5file.create_dataset(f'value/time_{t_now:0>3.3f}', data=value_rolling_np, compression="gzip")

			idx += 1
			itr_end.record()
			itr_end.synchronize()
			cpu_end = cputime()

			info(f't: {time_step} | GPU time: {(cp.cuda.get_elapsed_time(itr_start, itr_end)):.2f} | CPU Time: {(cpu_end-cpu_start):.2f}, | Targ bnds {min(y):.2f}/{max(y):.2f} Norm: {np.linalg.norm(y, 2):.2f}')

	if args.verify:
		x0 = np.array([[1.25, 0, pi]])

		#examine to see if the initial state is in the BRS/BRT
		gexam = copy.deepcopy(g)

if __name__ == '__main__':
	main(args)
