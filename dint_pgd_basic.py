
import sys
import copy
import time
import logging
import argparse
import cupy as cp
import numpy as np
import scipy.linalg as la
from os.path import abspath, join
import matplotlib.pyplot as plt

sys.path.append(abspath(join('..')))
from BRATVisualization.DIVisu import DoubleIntegratorVisualizer
sys.path.append(abspath(join('../..')))
from LevelSetPy.Utilities import *
from LevelSetPy.Grids import createGrid
from LevelSetPy.Helper import postTimeStepTTR
from LevelSetPy.Visualization import implicit_mesh
from LevelSetPy.DynamicalSystems import DoubleIntegrator

# POD Decomposition
from LevelSetPy.POD import *

# Chambolle-Pock for Total Variation De-Noising
from LevelSetPy.Optimization import chambollepock

# Value co-state and Lax-Friedrichs upwinding schemes
from LevelSetPy.InitialConditions import *
from LevelSetPy.SpatialDerivative import upwindFirstENO2
from LevelSetPy.ExplicitIntegration import artificialDissipationGLF
from LevelSetPy.ExplicitIntegration.Integration import odeCFL2, odeCFLset
from LevelSetPy.ExplicitIntegration.Term import termRestrictUpdate, termLaxFriedrichs

parser = argparse.ArgumentParser(description='Hamilton-Jacobi Analysis')
parser.add_argument('--silent', '-si', action='store_true', help='silent debug print outs' )
parser.add_argument('--save', '-sv', action='store_true', help='save BRS/BRT at end of sim' )
parser.add_argument('--visualize', '-vz', action='store_false', help='visualize level sets?' )
parser.add_argument('--load_brt', '-lb', action='store_false', help='load saved brt?' )
parser.add_argument('--init_cond', '-ic', type=str, default='circle', help='visualize level sets?' )
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


dint = DoubleIntegrator
u_bound = 1

def preprocessing():
    global dint, u_bound

    gmin = np.array(([[-1, -1]]),dtype=np.float64).T
    gmax = np.array(([[1, 1]]),dtype=np.float64).T
    g = createGrid(gmin, gmax, 101, None)

    eps_targ = 1.0
    target_rad = .2

    dint = DoubleIntegrator(g, u_bound)
    attr = dint.mttr() - target_rad
    if strcmp(args.init_cond, 'attr'):
        value_func_init = attr
    elif strcmp(args.init_cond, 'circle'):
        value_func_init = shapeSphere(g, zeros(g.dim, 1), target_rad)
    elif strcmp(args.init_cond, 'square'):
        value_func_init = shapeRectangleByCenter(g, zeros(g.dim, 1), \
                                    2*target_rad*ones(g.dim,1))

    attr = np.maximum(0, attr)

    return g, attr, value_func_init

def show_attr(g, attr, xis):
    ### pick a bunch of initial conditions
    Delta = lambda u: u  # u is either +1 or -1

    # how much timesteps to consider
    t = np.linspace(-1, 1, 100)
    u = 1
    # do implicit euler integration to obtain x1 and x2
    x1p = np.empty((len(xis), g.xs[0].shape[1], len(t)))
    x2p = np.empty((len(xis), g.xs[1].shape[1], len(t)))
    # states under negative control law
    x1m  = np.empty((len(xis), g.xs[0].shape[1], len(t)))
    x2m = np.empty((len(xis), g.xs[1].shape[1], len(t)))

    plt.ion()
    fig, [ax1,ax2] = plt.subplots(1, 2, figsize=(12,4))

    for i in range(len(xis)):
        for k in range(len(t)):
            x2p[i, :,k] = xis[i][1] + Delta(u) * t[k]
            x1p[i, :,k] = xis[i][0] + .5 * Delta(u) * x2p[i,:,k]**2 - .5 * Delta(u) * xis[i][1]**2
            # state trajos under -ve control law
            x2m[i, :,k] = xis[i][1] + Delta(-u) * t[k]
            x1m[i, :,k] = xis[i][0] + .5 * Delta(-u) * x2p[i,:,k]**2 - .5 * Delta(-u) * xis[i][1]**2


    # Plot a few snapshots for different initial conditions.
    color = iter(plt.cm.viridis(np.linspace(.25, 1, 2*len(xis))))
    for init_cond in range(len(xis)):
        # state trajectories are unique for every initial cond
        # so pick last state
        ax1.plot(x1p[init_cond, -1, :], x2p[init_cond, -1, :], color=next(color))#, label=rf"$u=+1$")
        ax1.plot(x1m[init_cond, -1, :], x2m[init_cond, -1, :], '--', color=next(color))#, label=rf"$u=+1$")
        ax1.grid('on')

    ax1.set_ylabel(rf"Linear Speed, $x_1$ ($m/s$)")
    ax1.set_xlabel(rf"Position $m$")
    ax1.set_title(rf"Forced Trajectories. Solid: $u = +1$ | Dashed: $u=-1$")
    ax1.legend(loc="lower right", fontsize=8, bbox_to_anchor=(1.01,.05))


    # Plot all vectograms(snapshots) in space and time.
    cdata = ax2.pcolormesh(g.xs[0], g.xs[1], attr, shading="nearest", cmap="magma")
    plt.colorbar(cdata, ax=ax2, extend="both")
    ax2.set_xlabel(r"Position($(m)$)")
    ax2.set_ylabel(r"Velocity $(m/s)$")
    ax2.set_title(r"State Plane")

    fig.suptitle("Isochrones for the Double Integral Plant.")

    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(args.pause_time)

    fig.suptitle("Isochrones for the Double Integral Plant.")

def show_init_levels(g, attr, value_func_init):

    f, (ax1, ax2) = plt.subplots(1,2,figsize=(12, 4))

    ax1.contour(g.xs[0], g.xs[1], attr, colors='red')
    ax1.set_title('Analytical TTR')
    ax1.set_xlabel(r"$x_1 (m)$")
    ax1.set_ylabel(r"$x_2 (ms^{-1})$")
    ax1.set_xlim([-1.02, 1.02])
    ax1.set_ylim([-1.01, 1.01])
    ax1.grid()

    ax2.contour(g.xs[0], g.xs[1], value_func_init, colors='blue')
    ax2.set_title('Numerical TTR')
    ax2.set_xlabel(r"$x_1 (m)$")
    ax2.set_xlim([-1.02, 1.02])
    ax2.set_ylim([-1.01, 1.01])
    ax2.grid()

    f.suptitle(f"Levelsets")

    f.canvas.draw()
    f.canvas.flush_events()
    time.sleep(args.pause_time)

def main(g, attr, value_init):
    global dint, u_bound
    #turn the state space over to the gpu
    g.xs = [cp.asarray(x) for x in g.xs]
    finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
    			innerData = Bundle({'grid':g,
                    'hamFunc': dint.hamiltonian,
    				'partialFunc': dint.dynamics,
    				'dissFunc': artificialDissipationGLF,
    				'derivFunc': upwindFirstENO2,
    				}),
    				positive = False,  # direction to grow the updated level set
    			))

    small = 100*eps
    t_span = np.linspace(0, 2.0, 20)
    options = Bundle(dict(factorCFL=0.75, stats='on', maxStep=realmax, 						singleStep='off', postTimestep=postTimeStepTTR))

    y = copy.copy(value_init.flatten())
    y, finite_diff_data = postTimeStepTTR(0, y, finite_diff_data)
    value_func = cp.asarray(copy.copy(y.reshape(g.shape)))

    U, Sigma, V = la.svd(value_init, full_matrices=False)

    # sanity checks
    print(U.shape, Sigma.shape, V.shape)
    np.allclose(U@np.diag(Sigma)@V, value_init)

    # informed reduced basis rank
    min_rank = minimal_projection_error(value_init, U, eps=1e-5, plot=True)
    print('min_rank ', min_rank)

    Ur, Sigmar, Vr = U[:,:min_rank], Sigma[:min_rank], V[:,:min_rank]
    # be sure the singular values are arranged in descending order

    plt.plot(Sigmar)
    plt.grid('on')
    plt.title('Singular values from DMD')
    f = plt.gcf()

    f.canvas.draw()
    f.canvas.flush_events()
    time.sleep(args.pause_time)

    # project value function in reduced basis
    value_rob = Ur.T @ value_init @ Ur
    # value_rob = value0_cpu @ Ur
    print(f'Value function in ROB: {value_rob.shape}')

    value_scaled = np.zeros_like(value_init)

    oddIndices = cell(g.dim)
    evenIndices = cell(g.dim)
    for i in range(g.dim):
        evenIndices = np.arange(0, value_scaled.shape[i], 2, dtype=np.intp)
        oddIndices = np.arange(1, value_scaled.shape[i], 2, dtype=np.intp)

    # interpolate data onto new value structure for ease of maniupulation
    value_scaled[np.ix_(evenIndices, evenIndices)] = value_rob[:-1,:-1]
    value_scaled[np.ix_(oddIndices, oddIndices)] = value_rob[2:,2:]

    value_rob = cp.asarray(value_scaled)
    del value_scaled

    # Visualization paramters
    spacing = tuple(g.dx.flatten().tolist())
    #turn the state space over to the gpu
    g.xs = [cp.asarray(x) for x in g.xs]

    params = Bundle(
    					{'grid': g,
    					#'pgd_grid': g,
    					'disp': True,
    					'labelsize': 16,
    					'labels': "Initial 0-LevelSet",
    					'linewidth': 2,
    					'elevation': args.elevation,
    					'azimuth': args.azimuth,
    					'mesh': value_func.get(),
    					'pgd_mesh': value_rob.get(),
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
    	viz = DoubleIntegratorVisualizer(params)

    value_func_all = np.zeros((len(t_span),)+value_func.shape)
    value_pgd_all = np.zeros((len(t_span),)+value_rob.shape)

    cur_time, max_time = 0, t_span[-1]
    step_time = (t_span[-1]-t_span[0])/8.0

    start_time = cputime()
    itr_start = cp.cuda.Event()
    itr_end = cp.cuda.Event()
    
    idx = 0

    while max_time-cur_time > small * max_time:
    	itr_start.record()
    	cpu_start = cputime()

    	time_step = f"{cur_time:.2f}/{max_time:.2f}"

    	y0 = value_func.flatten()
    	y0_rob = cp.asarray(value_rob.flatten())

    	#How far to integrate
    	t_span = np.hstack([cur_time, min(max_time, cur_time + step_time)])

    	# advance one step of integration
    	t, y, finite_diff_data = odeCFL2(termRestrictUpdate, t_span, y0, options, finite_diff_data)
    	t_rob, y_rob, finite_diff_rob = odeCFL2(termRestrictUpdate, t_span, y0_rob, options, finite_diff_data)
    	cp.cuda.Stream.null.synchronize()
    	cur_time = t if np.isscalar(t) else t[-1]

    	value_func = cp.reshape(y, g.shape)
    	value_rob = cp.reshape(y_rob, g.shape)

    	if args.visualize:
    		ls_mesh = value_func.get()
    		pgd_mesh = value_rob.get()
    		viz.update_tube(attr, ls_mesh, pgd_mesh, cur_time, delete_last_plot=True)

    	itr_end.record()
    	itr_end.synchronize()
    	cpu_end = cputime()

    	print(f't: {time_step} | GPU time: {(cp.cuda.get_elapsed_time(itr_start, itr_end)):.2f} | CPU Time: {(cpu_end-cpu_start):.2f}')
    	print(f'Bnds {min(y):.2f}/{max(y):.2f} | PGD Bnds {min(y_rob):.2f}/{max(y_rob):.2f} | Norm: {np.linalg.norm(y, 2):.2f} | PGD Norm: {np.linalg.norm(y_rob, 2):.2f} ')

    	# store this brt
    	value_func_all[idx] = ls_mesh
    	value_pgd_all[idx] = pgd_mesh
        
        idx += 1

    end_time = cputime()
    print(f'Total BRS/BRT execution time {(end_time - start_time):.4f} seconds.')


if __name__ == '__main__':
    g, attr, value_func_init = preprocessing()
    print("Showing analytical time to reach.")

    xis = [(1,0), (.5, 0),  (0,0), (-.5, 0), (-1,0),]
    show_attr(g, attr, xis)

    print("Showing initial level sets.")
    show_init_levels(g, attr, value_func_init)

    print("Evolving the level set.")
    main(g, attr, value_func_init)
