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
import logging
import argparse
import sys, os
import random
import cupy as cp
import numpy  as np
from math import pi
import numpy.linalg as LA
import matplotlib.pyplot as plt

from os.path import abspath, join, dirname, expanduser
sys.path.append(dirname(dirname(abspath(__file__))))
sys.path.append(abspath(join('..')))

from Libs import *

from LevelSetPy.Grids import *
from LevelSetPy.Utilities import *
from LevelSetPy.Visualization import *
from LevelSetPy.BoundaryCondition import *
# from LevelSetPy.DynamicalSystems import *
from LevelSetPy.InitialConditions import *


from BRATVisualization.rcbrt_visu import RCBRTVisualizer

parser = argparse.ArgumentParser(description='Hamilton-Jacobi Analysis')
parser.add_argument('--silent', '-si', action='store_false', help='silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_false', help='save BRS/BRT at end of sim' )
parser.add_argument('--visualize', '-vz', action='store_false', help='visualize level sets?' )
parser.add_argument('--load_brt', '-lb', action='store_true', help='load saved brt?' )
parser.add_argument('--stochastic', '-st', action='store_true', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--compute_traj', '-ct', action='store_false', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--verify', '-vf', action='store_true', default=True, help='visualize level sets?' )
parser.add_argument('--elevation', '-el', type=float, default=5., help='elevation angle for target set plot.' )
parser.add_argument('--direction', '-dr',  action='store_true',  help='direction to grow the level sets. Negative by default.' )
parser.add_argument('--azimuth', '-az', type=float, default=15., help='azimuth angle for target set plot.' )
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
w_bound = 1
fontdict = {'fontsize':16, 'fontweight':'bold'}


def get_flock(gmin, gmax, num_points, num_agents, init_xyzs, label,\
				periodic_dims=2, \
				reach_rad=.2, avoid_rad=.3):
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
	
	vehicles = [Bird(grid, 1, 1, np.expand_dims(init_xyzs[i], 1) , random.random(), \
						   center=np.zeros((3,1)), neigh_rad=3, \
						   label=i+1, init_random=False) for i in range(num_agents)]                
	flock = Flock(grid, vehicles, label=label, reach_rad=.2, avoid_rad=.3)

	return flock

def get_avoid_brt(flock, compute_mesh=True):
	"""
		Get the avoid BRT for this flock. That is, every bird 
		within a flock must avoid one another.

		Parameters:
		==========
		.flock: This flock of vehicles.
		.compute_mesh: compute mesh of local payoffs for each bird in this flock?
	"""
	idx=.3
	color = plt.cm.ocean(flock.label)

	for vehicle in flock.vehicles:
		vehicle_state = vehicle.cur_state
		# make the radius of the target setthe turn radius of this vehicle
		vehicle.payoff = shapeCylinder(flock.grid, 2, center=flock.position(vehicle_state), \
										radius=vehicle_state[-1].take(0))
		spacing=tuple(flock.grid.dx.flatten().tolist())
		# if compute_mesh:
		# 	vehicle.mesh_bundle   = implicit_mesh(vehicle.payoff, level=0, spacing=spacing, edge_color='r', face_color='k')
	
	"""
		Now compute the overall payoff for the flock
	   	by taking a union of all the avoid sets.
	"""
	prev_vehicle = flock.vehicles[0]
	for idx, vehicle in enumerate(flock.vehicles[1:]):
		flock.payoff = shapeUnion(prev_vehicle.payoff, vehicle.payoff)
		prev_vehicle = vehicle
	
	if compute_mesh:
		spacing=tuple(flock.grid.dx.flatten().tolist())
		flock.mesh_bundle = implicit_mesh(flock.payoff, 0, spacing, edge_color='.4', face_color='c')    
	
	return flock 

def main(args):
	# global params

	gmin = np.asarray([[-1, -1, -np.pi]]).T
	gmax = np.asarray([[1, 1, np.pi] ]).T
	num_agents = 7

	H         = .4 #.1
	H_STEP    = .05
	neigh_rad = 0.3

	# Please note the way I formulate the initial states here. Linear speed is constant but heading is different.
	INIT_XYZS = np.array([[neigh_rad*np.cos((i/6)*2*np.pi+np.pi/2), neigh_rad*np.sin((i/6)*2*np.pi+np.pi/2), H+i*H_STEP] for i in range(num_agents)])
	
	flock0 = get_flock(gmin, gmax, 101, num_agents, INIT_XYZS, label=1,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3)
	# add other flocks to this state space
	flock1 = get_flock(gmin, gmax, 101, num_agents, 1.1*INIT_XYZS, label=2,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3)
	flock2 = get_flock(gmin, gmax, 101, num_agents, -1.1*INIT_XYZS, label=3,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3)

	flock3 = get_flock(gmin, gmax, 101, num_agents, 1.5*INIT_XYZS, label=4,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3)
	flock4 = get_flock(gmin, gmax, 101, num_agents, -1.5*INIT_XYZS, label=5,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3)

	flock6 = get_flock(gmin, gmax, 101, num_agents, 2.0*INIT_XYZS, label=6,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3)						
	flock5 = get_flock(gmin, gmax, 101, num_agents, -2.0*INIT_XYZS, label=7,\
						periodic_dims=2, reach_rad=.2, avoid_rad=.3)
	get_avoid_brt(flock0, compute_mesh=True)
	# visualize_init_avoid_tube(flock0, save=True, fname=join(expanduser("~"), "Documents/Papers/Safety/WAFR2022", \
	# 									f"figures/flock_{flock0.label}.jpg"))
	
	# after creating value function, make state space cupy objects
	g = flock0.grid
	g.xs = [cp.asarray(x) for x in g.xs]
	finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
				innerData = Bundle({'grid':g,
					'hamFunc': flock0.hamiltonian,
					'partialFunc': flock0.dissipation,
					'dissFunc': artificialDissipationGLF,
					'CoStateCalc': upwindFirstENO2,
					}),
					positive = False,  # direction to grow the updated level set
				))
	t_range = [0, 2.5]

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
					 'title': f'Flock {flock0.label}\'s Avoid Tube. Num Agents={flock0.N}',
					 'level': 0, # which level set to visualize
					 'winsize': (16,9),
					 'fontdict': {'fontsize':18, 'fontweight':'bold'},
					 "savedict": Bundle({"save": True,
									"savename": "murmur",
									"savepath": join(expanduser("~"),
									"Documents/Papers/Safety/WAFR2022/figures/")
								 })
					}
					)
	args.spacing = spacing
	args.init_mesh = flock0.mesh_bundle; args.params = params

	if args.load_brt:
		args.save = False
		brt = np.load("data/murmurations.npz")
	else:
		if args.visualize:
			viz = RCBRTVisualizer(params=params)
		t_plot = (t_range[1] - t_range[0]) / 5 #10
		small = 100*eps
		options = Bundle(dict(factorCFL=0.95, stats='on', singleStep='off'))

		# Loop through t_range (subject to a little roundoff).
		t_now = t_range[0]
		start_time = cputime()
		itr_start = cp.cuda.Event()
		itr_end = cp.cuda.Event()

		brt = [flock0.payoff]
		meshes, brt_time = [], []
		value_rolling = cp.asarray(copy.copy(flock0.payoff))

		while(t_range[1] - t_now > small * t_range[1]):
			itr_start.record()
			cpu_start = cputime()
			time_step = f"{t_now}/{t_range[-1]}"

			# Reshape data array into column vector for ode solver call.
			y0 = value_rolling.flatten()

			# How far to step?
			t_span = cp.hstack([ t_now, min(t_range[1], t_now + t_plot) ])

			# Integrate a timestep.
			t, y, _ = odeCFL2(termRestrictUpdate, t_span, y0, odeCFLset(options), finite_diff_data)
			cp.cuda.Stream.null.synchronize()
			t_now = t

			# Get back the correctly shaped data array
			value_rolling = y.reshape(g.shape)

			if args.visualize:
				value_rolling_np = value_rolling.get()
				mesh_bundle=implicit_mesh(value_rolling_np, level=0, spacing=args.spacing,
									edge_color='.35',  face_color='magenta')
				viz.update_tube(mesh_bundle, time_step)
				# store this brt
				brt.append(value_rolling_np); brt_time.append(t_now); meshes.append(mesh_bundle)

			if args.save:
				fig = plt.gcf()
				fig.savefig(join(expanduser("~"),"Documents/Papers/Safety/WAFR2022",
					rf"figures/murmurations_{t_now}.jpg"), bbox_inches='tight',facecolor='None')
				# save this brt

			itr_end.record()
			itr_end.synchronize()
			cpu_end = cputime()

			info(f't: {time_step} | GPU time: {(cp.cuda.get_elapsed_time(itr_start, itr_end)):.2f} | CPU Time: {(cpu_end-cpu_start):.2f}, | Targ bnds {min(y):.2f}/{max(y):.2f} Norm: {np.linalg.norm(y, 2):.2f}')

		# if not args.load_brt:
		# 	os.makedirs("data") if not os.path.exists("data") else None
		# 	np.savez_compressed("data/rcbrt.npz", brt=np.asarray(brt), \
		# 		meshes=np.asarray(meshes), brt_time=np.asarray(brt_time))

	if args.verify:
		x0 = np.array([[1.25, 0, pi]])

		#examine to see if the initial state is in the BRS/BRT
		gexam = copy.deepcopy(g)

if __name__ == '__main__':
	main(args)
