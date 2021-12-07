__all__ = ['double_integrator_ttr']

__author__ 		= "Lekan Molu"
__copyright__ 	= "2021, Large Hamilton-Jacobi Analysis."
__license__ 	= "Molux License"
__comment__ 	= "Double Integrator Dynamics Time to reach origin."
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"

import copy
import argparse
import logging
import cupy as cp
import numpy as np

import sys
from os.path import abspath, join, dirname
sys.path.append(dirname(dirname(abspath(__file__))))
from BRATVisualization.rcbrt_visu import RCBRTVisualizer

sys.path.append(abspath(join('..')))
from LevelSetPy.Utilities import *
from LevelSetPy.Grids import createGrid
from LevelSetPy.Helper import postTimeStepTTR
from LevelSetPy.Visualization import implicit_mesh
from LevelSetPy.DynamicalSystems import DoubleIntegrator
from LevelSetPy.SpatialDerivative import upwindFirstENO2, upwindFirstWENO5
from LevelSetPy.ExplicitIntegration import artificialDissipationGLF
from LevelSetPy.ExplicitIntegration.Integration import odeCFL3, odeCFL2, odeCFLset
from LevelSetPy.ExplicitIntegration.Term import termRestrictUpdate, termLaxFriedrichs

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from os.path import dirname, abspath, join
sys.path.append(dirname(dirname(abspath(__file__))))

parser = argparse.ArgumentParser(description='Hamilton-Jacobi Analysis')
parser.add_argument('--silent', '-si', action='store_true', help='silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_true', help='save BRS/BRT at end of sim' )
parser.add_argument('--visualize', '-vz', action='store_true', help='visualize level sets?' )
parser.add_argument('--load_brt', '-lb', action='store_false', help='load saved brt?' )
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


def main(args):
	gmin = np.array(([[-1, -1]]),dtype=np.float64).T
	gmax = np.array(([[1, 1]]),dtype=np.float64).T
	g = createGrid(gmin, gmax, 101, None)

	eps_targ = 1.0
	u_bound = 1
	target_rad = .2 #eps_targ * np.max(g.dx)
	dint = DoubleIntegrator(g, u_bound)
	attr = dint.mttr() - target_rad

	value_func = copy.copy(attr)
	value_func_init = copy.copy(value_func)

	attr = np.maximum(0, attr)

	#turn the state space over to the gpu
	g.xs = [cp.asarray(x) for x in g.xs]
	
	finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
				innerData = Bundle({'grid':g, 'hamFunc': dint.hamiltonian,
					'partialFunc': dint.dynamics,
					'dissFunc': artificialDissipationGLF,
					'derivFunc': upwindFirstENO2,
					'input_bound': u_bound,
					'innerFunc': termLaxFriedrichs,
					'positive': False,  # direction to grow the updated level set
					}),
					positive = False,  # direction to grow the updated level set
				))

	small = 100*eps
	t_span = np.linspace(0, 2.0, 20)
	options = Bundle(dict(factorCFL=0.75, stats='on', maxStep=realmax, \
						  singleStep='off', postTimestep=postTimeStepTTR))
	
	y = copy.copy(value_func.flatten())
	y, finite_diff_data = postTimeStepTTR(0, y, finite_diff_data)
	value_func = cp.asarray(copy.copy(y.reshape(g.shape)))

	# Visualization paramters
	spacing = tuple(g.dx.flatten().tolist())
	
	params = Bundle(
					{"grid": g,
					 'disp': True,
					 'labelsize': 16,
					 'labels': "Initial 0-LevelSet",
					 'linewidth': 2,
					 'data': value_func,
					 'elevation': args.elevation,
					 'azimuth': args.azimuth,
					 'mesh': value_func,
					 'init_conditions': False,
					 'pause_time': args.pause_time,
					 'level': 0, # which level set to visualize
					 'winsize': (16,9),
					 'fontdict': Bundle({'fontsize':12, 'fontweight':'bold'}),
					 "savedict": Bundle({"save": False,
					 				"savename": "rcbrt",
					 				"savepath": "../jpeg_dumps/rcbrt"})
					 })

	if args.visualize:
		viz = RCBRTVisualizer(params=args.params)

	value_func_all = [value_func]

	cur_time, max_time = 0, t_span[-1]
	step_time = (t_span[-1]-t_span[0])/8.0
	
	start_time = cputime()
	itr_start = cp.cuda.Event()
	itr_end = cp.cuda.Event()

	while max_time-cur_time > small * max_time:
		itr_start.record()
		cpu_start = cputime()

		time_step = f"{cur_time:.2f}/{max_time:.2f}"

		y0 = value_func.flatten()

		#How far to integrate
		t_span = np.hstack([cur_time, min(max_time, cur_time + step_time)])

		# one step of integration
		t, y, finite_diff_data = odeCFL2(termRestrictUpdate, t_span, \
									     y0, options, finite_diff_data)
		cp.cuda.Stream.null.synchronize()
		cur_time = t if np.isscalar(t) else t[-1]

		value_func = cp.reshape(y, g.shape)

		if args.visualize:
			data_np = value_func.get()
			mesh=implicit_mesh(data_np, level=0, spacing=spacing,
								edge_color='None',  face_color='red')
			viz.update_tube(data_np, mesh, args.pause_time)
		
		itr_end.record()
		itr_end.synchronize()
		cpu_end = cputime()
		
		info(f't: {time_step} | GPU time: {(cp.cuda.get_elapsed_time(itr_start, itr_end)):.2f} | CPU Time: {(cpu_end-cpu_start):.2f}, | Targ bnds {min(y):.2f}/{max(y):.2f} Norm: {np.linalg.norm(y, 2):.2f}')
        
		# store this brt
		value_func_all.append(value_func.get())
		
	end_time = cputime()
	info(f'Total BRS/BRT execution time {(end_time - start_time):.4f} seconds.')

	return t, value_func_all

if __name__ == '__main__':
	main(args)