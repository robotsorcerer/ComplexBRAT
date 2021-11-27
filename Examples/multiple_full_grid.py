__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Hamilton-Jacobi Analysis in Python"
__license__ 	= "Molux License"
__comment__ 	= "2 Evaders 2 Pursuers"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import copy
import time
import logging
import argparse
import sys, os
import cupy as cp
import numpy  as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

sys.path.append('../')
from LevelSetPy.Utilities import *
from LevelSetPy.Visualization import *
from LevelSetPy.DynamicalSystems import *
from LevelSetPy.Grids import createGrid
from LevelSetPy.InitialConditions import shapeCylinder
from LevelSetPy.SpatialDerivative import upwindFirstENO2
from LevelSetPy.ExplicitIntegration.Dissipation import artificialDissipationGLF

from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))
from BRATSolver.brt_solver import solve_brt


parser = argparse.ArgumentParser(description='Hamilton-Jacobi Analysis')
parser.add_argument('--silent', '-si', action='store_false', help='silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_true', help='save BRS/BRT at end of sim' )
parser.add_argument('--visualize', '-vz', action='store_true', help='visualize level sets?' )
parser.add_argument('--load_brt', '-lb', action='store_true', default=True, help='load saved brt?' )
parser.add_argument('--stochastic', '-st', action='store_true', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--compute_traj', '-ct', action='store_false', help='Run trajectories with stochastic dynamics?' )
parser.add_argument('--verify', '-vf', action='store_true', default=True, help='visualize level sets?' )
parser.add_argument('--elevation', '-el', type=float, default=5., help='elevation angle for target set plot.' )
parser.add_argument('--azimuth', '-az', type=float, default=5., help='azimuth angle for target set plot.' )
parser.add_argument('--pause_time', '-pz', type=float, default=4, help='pause time between successive updates of plots' )
args = parser.parse_args()
args.verbose = True if not args.silent else False

print(f'args:  {args}')

if not args.silent:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
else:
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
# Turn off pyplot's spurious dumps on screen
logging.getLogger('matplotlib.font_manager').disabled = True
logger = logging.getLogger(__name__)

obj = Bundle({})

def get_obj():
	# by default all vehicle velocities share similar velocity characteristics
	
	pdDims = 2; N = 100
	v, w = +1, +1
	obj.ve, obj.vp = v, v
	obj.we, obj.wp = -w, w
	

	# get player (pursuer) 1's state space
	gmin = np.array(([[-5, -5, -pi]])).T
	gmax = np.array(([[0, 0, pi]])).T
	obj.p1 = Bundle({'pursuer':Bundle({}), 'evader':Bundle({})})
	obj.p1.pursuer.grid = createGrid(gmin, gmax, N, pdDims)
	obj.p1.pursuer.center = np.array(([[-2.5, -2.5, 0]]),dtype=np.float64).T
	obj.p1.pursuer.radius = 0.5

	# get player (evader) 2's state space
	gmin = np.array(([[0, 0, pi]])).T
	gmax = np.array(([[5, 5, 3*pi]])).T
	obj.p1.evader.grid = createGrid(gmin, gmax, N, pdDims)
	obj.p1.evader.center = np.array(([[2.5, 2.5, 2*pi]]),dtype=np.float64).T
	obj.p1.evader.radius = .5

	# get player (pursuer) 3's state space
	gmin = np.array(([[5, 5, 3*pi]])).T
	gmax = np.array(([[10, 10, 5*pi]])).T
	obj.p2 = Bundle({'pursuer':Bundle({}), 'evader':Bundle({})})
	obj.p2.pursuer.grid = createGrid(gmin, gmax, N, pdDims)
	obj.p2.pursuer.center = np.array(([[7.5, 7.5, 4*pi]]),dtype=np.float64).T
	obj.p2.pursuer.radius = .5

	# get player (evader) 4's state space
	gmin = np.array(([[10, 10, 5*pi]])).T
	gmax = np.array(([[15, 15, 7*pi]])).T
	obj.p2.evader.grid = createGrid(gmin, gmax, N, pdDims)
	obj.p2.evader.center = np.array(([[12.5, 12.5, 6*pi]]),dtype=np.float64).T
	obj.p2.evader.radius = .5

	# Full grid
	gmin = np.array(([[-5, -5, -pi]])).T
	gmax = np.array(([[15, 15, 7*pi]])).T
	obj.full_grid = createGrid(gmin, gmax, N, pdDims)

	'''
		Here, the table is symmetric, so we end up with the upper triangular
		capture or avoid results of the differential game. (See paper.)
	'''
	obj.p1.pursuer.xdot = dubins_absolute(obj, obj.p1.pursuer)
	obj.p1.evader.xdot  = dubins_absolute(obj, obj.p1.evader)
	obj.p2.pursuer.xdot = dubins_absolute(obj, obj.p2.pursuer)
	obj.p2.evader.xdot  = dubins_absolute(obj, obj.p2.evader)

	# after creating value function, make state space cupy objects
	obj.p1.pursuer.grid.xs = [cp.asarray(x) for x in obj.p1.pursuer.grid.xs]

	# Compute target set of all four vehicles
	value_func = shapeRectangleByCorners(obj.full_grid, lower=-3, upper=13)
	# we now have a large value function, decompose the value w.r.t to the
	# basis of the four vehicles to get its correspondiung decomposition ihnto diff bases


	obj.p1_term = obj.v_e - obj.v_p * cp.cos(obj.grid.xs[2])
	obj.p2_term = -obj.v_p * cp.sin(obj.grid.xs[2])
	obj.alpha = [ cp.abs(obj.p1_term) + cp.abs(obj.omega_e * obj.grid.xs[1]), \
					cp.abs(obj.p2_term) + cp.abs(obj.omega_e * obj.grid.xs[0]), \
					obj.omega_e + obj.omega_p ]

	return obj 

def get_target(g):
	cylinder = shapeCylinder(g.grid, g.axis_align, g.center, g.radius)
	return cylinder

def get_hamiltonian_func(t, data, deriv, finite_diff_data):
	global obj
	ham_value = deriv[0] * obj.p1_term + \
				deriv[1] * obj.p2_term - \
				obj.omega_e*np.abs(deriv[0]*obj.grid.xs[1] - \
				deriv[1] * obj.grid.xs[0] - deriv[2])  + \
				obj.omega_p * np.abs(deriv[2])

	return ham_value, finite_diff_data

def get_partial_func(t, data, derivMin, derivMax, \
			  schemeData, dim):
	"""
		Calculate the extrema of the absolute value of the partials of the
		analytic Hamiltonian with respect to the costate (gradient).
	"""
	global obj

	assert dim>=0 and dim <3, "grid dimension has to be between 0 and 2 inclusive."

	return obj.alpha[dim]

def main(args):
	obj = get_obj()
  	data0 = get_target(obj)
	data = cp.asarray(copy.copy(data0))
	finite_diff_data = Bundle({'grid': obj.grid, 'hamFunc': get_hamiltonian_func,
								'partialFunc': get_partial_func,
								'dissFunc': artificialDissipationGLF,
								'derivFunc': upwindFirstENO2,
								})

	# Visualization paramters
	spacing = tuple(obj.grid.dx.flatten().tolist())
	init_mesh = implicit_mesh(data0, level=0, spacing=spacing, edge_color='b', face_color='b')
	params = Bundle(
					{"grid": obj.grid,
					 'disp': True,
					 'labelsize': 16,
					 'labels': "Initial 0-LevelSet",
					 'linewidth': 2,
					 'data': data,
					 'elevation': args.elevation,
					 'azimuth': args.azimuth,
					 'mesh': init_mesh,
					 'init_conditions': False,
					 'pause_time': args.pause_time,
					 'level': 0, # which level set to visualize
					 'winsize': (16,9),
					 'fontdict': Bundle({'fontsize':12, 'fontweight':'bold'}),
					 "savedict": Bundle({"save": False,
					 				"savename": "rcbrt",
					 				"savepath": "../jpeg_dumps/rcbrt"})
					 })

	args.obj = obj; args.spacing = spacing
	args.init_mesh = init_mesh; args.params = params 
	t_range = [0, 2.5]

	if args.load_brt:
		args.save = False
		brt = np.load("data/rcbrt.npz")
	else:
		opt_t, brt = solve_brt(args, t_range, data, finite_diff_data)

	if args.save:
		import os
		os.makedirs("data") if not os.path.exists("data") else None
		np.savez_compressed("data/rcbrt.npz", brt=data.get())

	if args.verify:
		x0 = np.array([[1.25, 0, pi]])

		#examine to see if the initial state is in the BRS/BRT
		gexam = copy.deepcopy(obj.grid)

		#we should be doing a kdtree or flann search here
		# state_val  = eval(obj.grid, x0, brt)

		# # g, data = augmentPeriodic(obj.grid, brt)
		# for i in range(obj.grid.dim):
		# 	if (isfield(obj.grid, 'bdry') and id(obj.grid.bdry[i])==id(addGhostPeriodic)):
				


if __name__ == '__main__':
	main(args)