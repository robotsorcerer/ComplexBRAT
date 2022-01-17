#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import random
import hashlib
import numpy as np
import numpy.linalg as LA
from os.path import abspath, join
sys.path.append(abspath(join('../..')))
sys.path.append(abspath(join('..')))

from math import pi
from os.path import expanduser, join

from LevelSetPy.Utilities import *
from LevelSetPy.BoundaryCondition import *
from LevelSetPy.Visualization import *
# from LevelSetPy.DynamicalSystems import *
from LevelSetPy.Grids import *
from LevelSetPy.DynamicalSystems import *
from LevelSetPy.InitialConditions import *
# from LevelSetPy.SpatialDerivative import *
# from LevelSetPy.ExplicitIntegration.Dissipation import *


import cupy as cp 
from LevelSetPy.SpatialDerivative import upwindFirstENO2
from LevelSetPy.ExplicitIntegration.Integration import odeCFL2, odeCFLset
from LevelSetPy.ExplicitIntegration.Dissipation import artificialDissipationGLF
from LevelSetPy.ExplicitIntegration.Term import termRestrictUpdate, termLaxFriedrichs


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# In[2]:



def get_flock(gmin, gmax, num_points, num_agents, init_xyzs, label,				periodic_dims=2, 				reach_rad=.2, avoid_rad=.3):
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
	
	vehicles = [BirdSingle(grid, 1, 1, np.expand_dims(init_xyzs[i], 1) , random.random(), 						   center=np.zeros((3,1)), neigh_rad=3, 						   label=i+1, init_random=False) for i in range(num_agents)]                
	flock = BirdFlock(grid, vehicles, label=label, reach_rad=.2, avoid_rad=.3)

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
		vehicle.payoff = shapeCylinder(flock.grid, 2, center=flock.position(vehicle_state), 										radius=vehicle.cur_state[-1].take(0))
		spacing=tuple(flock.grid.dx.flatten().tolist())
		if compute_mesh:
			vehicle.mesh   = implicit_mesh(vehicle.payoff, level=0, spacing=spacing, edge_color='r', face_color='k')
	
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
		mesh = implicit_mesh(flock.payoff, 0, spacing, edge_color='.4', face_color='c')    
		flock.mesh, flock.verts = mesh.mesh, mesh.verts
	
	return flock 

def visualize_init_avoid_tube(flock, save=True, fname=None, title=''):
	"""
		For a flock, whose mesh has been precomputed, 
		visualize the initial backward avoid tube.
	"""
	# visualize avoid set 
	fontdict = {'fontsize':16, 'fontweight':'bold'}

	fig = plt.figure(1, figsize=(16,9), dpi=100)
	ax = plt.subplot(111, projection='3d')
	ax.add_collection3d(flock.mesh)


	xlim = (flock.verts[:, 0].min(), flock.verts[:,0].max())
	ylim = (flock.verts[:, 1].min(), flock.verts[:,1].max())
	zlim = (flock.verts[:, 2].min(), flock.verts[:,2].max())

	# create grid that contains just this zero-level set to avoid computational craze 
	gmin = np.asarray([[xlim[0], ylim[0], zlim[0]]]).T
	gmax = np.asarray([[xlim[1], ylim[1], zlim[1]] ]).T

	# create reduced grid upon which this zero level set dwells
	flock.grid_zero = createGrid(gmin, gmax, flock0.grid.N, 2)

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
		ax.set_title(f'Flock {flock.label}\'s Avoid Tube. Num_agents={flock.N}', fontdict=fontdict)
	ax.view_init(azim=-45, elev=30)

	if save:
		fig.savefig(fname, bbox_inches='tight',facecolor='None')


# ## Define 1st Flock 

# In[3]:



gmin = np.asarray([[-1, -1, -np.pi]]).T
gmax = np.asarray([[1, 1, np.pi] ]).T
num_agents = 5

H         = .1
H_STEP    = .05
neigh_rad = 0.3

# Please note the way I formulate the initial states here. Linear speed is constant but heading is different.
INIT_XYZS = np.array([[neigh_rad*np.cos((i/6)*2*np.pi+np.pi/2), neigh_rad*np.sin((i/6)*2*np.pi+np.pi/2), H+i*H_STEP] for i in range(num_agents)])
flock0 = get_flock(gmin, gmax, 101, num_agents, INIT_XYZS, 1, 2, .2, .3)
get_avoid_brt(flock0, compute_mesh=True)
visualize_init_avoid_tube(flock0, save=True, fname=join(expanduser("~"), "Documents/Papers/Safety/WAFR2022", 									f"figures/flock_{flock0.label}.jpg"))


# In[4]:


spacing=tuple(flock0.grid.dx.flatten().tolist())

mesh = implicit_mesh(flock0.vehicles[0].payoff, level=0, spacing=spacing, edge_color=None, face_color='orange')
mesh.label = 0

visualize_init_avoid_tube(mesh, save=True, 							fname=join(expanduser("~"), "Documents/Papers/Safety/WAFR2022", 									f"figures/payoff_single.jpg"), 
							title='Zero-Level RCBRT Payoff for Agent 1 in Flock 1')


# In[7]:


# See if we can update the RCBRT using the new grid
g = flock0.grid_zero
dubins_rel = DubinsVehicleRel(g, 1, 1, expand(INIT_XYZS[0], 1))

# after creating value function, make state space cupy objects
g.xs = [cp.asarray(x) for x in g.xs]
finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
            innerData = Bundle({'grid':g,
                'hamFunc': dubins_rel.hamiltonian,
                'partialFunc': dubins_rel.dissipation,
                'dissFunc': artificialDissipationGLF,
                'CoStateCalc': upwindFirstENO2,
                }),
                positive = False,  # direction to grow the updated level set
            ))

t_range = [0, 2.5]

params = Bundle(
                {"grid": g,
                'disp': True,
                'labelsize': 16,
                'labels': "Initial 0-LevelSet",
                'linewidth': 2,
                'data': flock0.payoff,
                'elevation': 10,
                'azimuth': 10,
                'mesh': Bundle(dict(mesh=flock0.mesh, verts=flock0.verts)),
                'init_conditions': False,
                'pause_time': .3,
                'level': 0, # which level set to visualize
                'winsize': (16,9),
                'fontdict': Bundle({'fontsize':18, 'fontweight':'bold'}),
                "savedict": Bundle({"save": False,
                            "savename": "dint_basic.jpg",
                            "savepath": join(expanduser("~"),
                            "Documents/Papers/Safety/PGDReach/figures")
                            })
                }
                )


# In[6]:


from BRATVisualization.rcbrt_visu import RCBRTVisualizer


# In[ ]:


viz = RCBRTVisualizer(params=params)
t_plot = (t_range[1] - t_range[0]) / 10
small = 100*eps
options = Bundle(dict(factorCFL=0.95, stats='on', singleStep='off'))

# Loop through t_range (subject to a little roundoff).
t_now = t_range[0]
start_time = cputime()
itr_start = cp.cuda.Event()
itr_end = cp.cuda.Event()

brt = [value_init]
meshes, brt_time = [], []
value_rolling = cp.asarray(copy.copy(value_init))

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
			mesh=implicit_mesh(value_rolling_np, level=0, spacing=args.spacing,
								edge_color=None,  face_color='maroon')
			viz.update_tube(value_rolling_np, mesh, time_step)
			# store this brt
			brt.append(value_rolling_np); brt_time.append(t_now); meshes.append(mesh)

		if args.save:
			fig = plt.gcf()
			fig.savefig(join(expanduser("~"),"Documents/Papers/Safety/WAFR2022",
				rf"figures/rcbrt_{t_now}.jpg"),
				bbox_inches='tight',facecolor='None')

		itr_end.record()
		itr_end.synchronize()
		cpu_end = cputime()

		info(f't: {time_step} | GPU time: {(cp.cuda.get_elapsed_time(itr_start, itr_end)):.2f} | CPU Time: {(cpu_end-cpu_start):.2f}, | Targ bnds {min(y):.2f}/{max(y):.2f} Norm: {np.linalg.norm(y, 2):.2f}')


# In[10]:


gmax       = gmin
gmin       = gmin*3
gcenters   = (gmax-gmin)/2
# print(gcenters)

num_agents = 8  
H          = .1
H_STEP     = .05
neigh_rad  = 0.3

# Please note the way I formulate the initial states here. Linear speed is constant but heading is different.
INIT_XYZS = np.array([[gcenters[0,0]*np.cos((i/6)*2*np.pi+np.pi/2), gcenters[1,0]*np.sin((i/6)*2*np.pi+np.pi/2), H+i*H_STEP] for i in range(num_agents)])
flock1 = get_flock(gmin, gmax, 101, num_agents, INIT_XYZS, 2, 2, .2, .3)
flock1_avoid_brt = get_avoid_brt(flock1, compute_mesh=True)
visualize_init_avoid_tube(flock1, save=True, xlim=(0, 3), ylim=(1.5,3), zlim=(0,10.9))


# ### Get Avoid Tube for Flock III
# 

# In[ ]:



gmin = np.asarray([[-2, -2, -2*np.pi]]).T
gmax = np.asarray([[2, 2, 2*np.pi] ]).T
num_agents = 8  

H         = .1
H_STEP    = .05
neigh_rad = 0.3

# Please note the way I formulate the initial states here. Linear speed is constant but heading is different.
INIT_XYZS = np.array([[neigh_rad*np.cos((i/6)*2*np.pi+np.pi/2), neigh_rad*np.sin((i/6)*2*np.pi+np.pi/2), H+i*H_STEP] for i in range(num_agents)])
flock1 = get_flock(gmin, gmax, 101, num_agents, INIT_XYZS, 2, 2, .2, .3)
flock1_avoid_brt = get_avoid_brt(flock1, compute_mesh=True)
visualize_init_avoid_tube(flock1, save=True, xlim=(0, 3), ylim=(1.5,3), zlim=(0,10.9))


# This one for target set involving leaderless coordination. Every agent within the flock must maintain a metric distance from one the other. There is a repulsive force between each agent and a predator.

# In[ ]:


# it feels like we also need create the case for when birds in a flock have a sparse interaction.


# In[ ]:


# compute nearest neigbors on a grid


# In[ ]:


# Now create classes for flocks splitting, flocks contraction, and flocks expansion.


# In[ ]:


dubins_flock


# In[ ]:



brt = np.load("../data/rcbrt.npz")['brt']
brt_time = np.load("../data/rcbrt.npz")['brt_time']
brt_time += [brt_time[-1]]
                                  
f = plt.figure(figsize=(16,9))
gs  = gridspec.GridSpec(2,2, f)


def plot_brt(ax, mesh, time_step):
	ax.grid('on')
	ax.add_collection3d(mesh)  
	xlim = (0, 1.75)
	ylim = (0, 1.75)
	zlim = (0, 2*np.pi)

	ax.set_xlim3d(*xlim)
	ax.set_ylim3d(*ylim)
	ax.set_zlim3d(*zlim)
	ax.set_title(rf'BRT at {time_step}/2.5 secs.', 			fontdict={'fontsize':18, 'fontweight':'bold'})


# slices to plot:
slc = [0, 3, 7, 10]
brt_time = np.linspace(0, 2.0, 11)
ax = [plt.subplot(gs[0, 0], projection='3d'), 
	  plt.subplot(gs[0, 1], projection='3d'),
	  plt.subplot(gs[1, 0], projection='3d'),
	  plt.subplot(gs[1, 1], projection='3d')]
i=0
for slc_num in slc:		
	mesh = implicit_mesh(brt[slc_num], level=0, spacing=tuple(g.dx.flatten().tolist()),
									edge_color=None,  face_color='orchid')
	plot_brt(ax[i], mesh, brt_time[slc_num])
	i+=1	
plt.show()


# In[ ]:


def controllability(g, gr, attr, value_func_init):
	fontdict = {'fontsize':28, 'fontweight':'bold'}
	f, (ax1, ax2) = plt.subplots(1,2,figsize=(16, 6))

	ax1.contour(g.xs[0], g.xs[1], attr, colors='red')
	ax1.set_title('Analytical TTR', fontdict =fontdict)
	ax1.set_xlabel(r"$x_1 (m)$", fontdict =fontdict)
	ax1.set_ylabel(r"$x_2 (ms^{-1})$", fontdict =fontdict)
	ax1.tick_params(axis='both', which='major', labelsize=28)
	ax1.tick_params(axis='both', which='minor', labelsize=18)
	ax1.set_xlim([-1.02, 1.02])
	ax1.set_ylim([-1.01, 1.01])
	ax1.grid()

	ax2.contour(gr.xs[0], gr.xs[1], value_func_init, colors='blue')
	ax2.set_title('Numerical TTR', fontdict =fontdict)
	ax2.set_xlabel(r"$x_1 (m)$", fontdict =fontdict)
	ax2.set_xlim([-1.02, 1.02])
	ax2.set_ylim([-1.01, 1.01])
	ax2.grid('on')
	ax2.tick_params(axis='both', which='major', labelsize=28)
	ax2.tick_params(axis='both', which='minor', labelsize=18)
	ax2.legend(loc="center left", fontsize=8) 

	f.suptitle(f"Levelsets")

	f.canvas.draw()
	f.canvas.flush_events()
	time.sleep(args.pause_time)


# In[ ]:


g.xs = [g.xs[i].get() for i in range(g.dim)]
gr.xs = [gr.xs[i].get() for i in range(gr.dim)]
view_noncontrollability(g, gr, attr, value_rob)


# In[ ]:


gs


# In[ ]:


p1, p2, mode= obj.p1.pursuer, obj.p1.evader, 'capture'

assert isfield(p1, 'center'), 'player I must have a center '                                'defined for its capture equation.'
assert isfield(p2, 'center'), 'player II must have a center '                                'defined for its capture equation.'
assert isfield(p1, 'grid'), 'player I must have its grid info'
assert isfield(p2, 'grid'), 'player II must have its grid info'

x1 = p1.grid.xs
x2 = p2.grid.xs

p1_dyn  = cell(3)
p2_dyn = cell(3)

p1_dyn[0] = obj.vp*np.cos(x1[2])
p1_dyn[1] = obj.vp*np.sin(x1[2])
p1_dyn[2] = obj.we

p2_dyn[0] = obj.ve*np.cos(x2[2])
p2_dyn[1] = obj.ve*np.sin(x2[2])
p2_dyn[2] = obj.wp

data_agent1 = np.zeros(p1.grid.shape)
data_agent2 = np.zeros(p2.grid.shape)

if isfield(p1, 'center') and numel(p1.center==1):
    p1.center = p1.center*np.ones((p1.grid.dim,1), dtype=np.float64)
if isfield(p2, 'center') and numel(p2.center==1):
    p2.center = p2.center*np.ones((p2.grid.dim,1), dtype=np.float64)
        


# ### This for the Double Integrator

# In[ ]:


import cupy as cp
import numpy as np

import sys
from os.path import abspath, join
sys.path.append(abspath(join('../..')))
from LevelSetPy.Utilities import *
from LevelSetPy.Grids import createGrid
from LevelSetPy.Helper import postTimeStepTTR
from LevelSetPy.Visualization import implicit_mesh
from LevelSetPy.DynamicalSystems import DoubleIntegrator
from LevelSetPy.SpatialDerivative import upwindFirstWENO5
from LevelSetPy.ExplicitIntegration import artificialDissipationGLF
from LevelSetPy.ExplicitIntegration.Integration import odeCFL3, odeCFLset
from LevelSetPy.ExplicitIntegration.Term import termRestrictUpdate, termLaxFriedrichs


# In[ ]:


gmin = np.array(([[-1, -1]]),dtype=np.float64).T
gmax = np.array(([[1, 1]]),dtype=np.float64).T
g = createGrid(gmin, gmax, 101, None)

eps_targ = 1.0
u_bound = 1
target_rad = .2 #eps_targ * np.max(g.dx)
dint = DoubleIntegrator(g, u_bound)
value_time2reach = dint.min_time2reach() - target_rad
value_time2reach = np.maximum(0, value_time2reach)


# In[ ]:


above_curve=dint.grid.xs[0]>dint.Gamma
below_curve=dint.grid.xs[0]<dint.Gamma
on_curve=dint.grid.xs[0]==dint.Gamma

reach_term1  = (dint.grid.xs[1] + np.emath.sqrt(4*dint.grid.xs[0] +                          2 * dint.grid.xs[1]**2))*above_curve
reach_term2 =  (-dint.grid.xs[1]+np.emath.sqrt(-4*dint.grid.xs[0] +                 2 * dint.grid.xs[1]**2) )*below_curve
reach_term3 = np.abs(dint.grid.xs[1]) * on_curve
reach_time = reach_term1.real + reach_term2.real + reach_term3


# In[ ]:


dataIn = cp.arange(1, 13).reshape(3, 4)
print('dataIn\n', dataIn)
indices =[ [3], [5]]
print()
print(dataIn[cp.ix_(*indices)])


# ### Multiple vehicles

# In[ ]:



obj = Bundle({})


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
obj.p1.pursuer.basis = np.array(([[1,0,0]]))

# get player (evader) 2's state space
gmin = np.array(([[0, 0, pi]])).T
gmax = np.array(([[5, 5, 3*pi]])).T
obj.p1.evader.grid = createGrid(gmin, gmax, N, pdDims)
obj.p1.evader.center = np.array(([[2.5, 2.5, 2*pi]]),dtype=np.float64).T
obj.p1.evader.radius = .5
obj.p1.evader.basis = np.array(([[0,1,0]]))

# get player (pursuer) 3's state space
gmin = np.array(([[5, 5, 3*pi]])).T
gmax = np.array(([[10, 10, 5*pi]])).T
obj.p2 = Bundle({'pursuer':Bundle({}), 'evader':Bundle({})})
obj.p2.pursuer.grid = createGrid(gmin, gmax, N, pdDims)
obj.p2.pursuer.center = np.array(([[7.5, 7.5, 4*pi]]),dtype=np.float64).T
obj.p2.pursuer.radius = .5
obj.p2.pursuer.basis = np.array(([[0,0,1]]))

# get player (evader) 4's state space
gmin = np.array(([[10, 10, 5*pi]])).T
gmax = np.array(([[15, 15, 7*pi]])).T
obj.p2.evader.grid = createGrid(gmin, gmax, N, pdDims)
obj.p2.evader.center = np.array(([[12.5, 12.5, 6*pi]]),dtype=np.float64).T
obj.p2.evader.radius = .5
obj.p2.evader.basis = np.array(([[0,0,0,1]]))

# Full grid
gmin = np.array(([[-5, -5, -pi]])).T
gmax = np.array(([[15, 15, 7*pi]])).T
obj.full_grid = createGrid(gmin, gmax, N, pdDims)


# In[ ]:


obj.p1.pursuer.xdot = dubins_absolute(obj, obj.p1.pursuer)
obj.p1.evader.xdot  = dubins_absolute(obj, obj.p1.evader)
obj.p2.pursuer.xdot = dubins_absolute(obj, obj.p2.pursuer)
obj.p2.evader.xdot  = dubins_absolute(obj, obj.p2.evader)
 
value_func = shapeRectangleByCorners(obj.full_grid, lower=-3, upper=13)


# In[ ]:


# we now have a large value function, decompose the value w.r.t to the
# basis of the four vehicles to get its correspondiung decomposition into diff bases


# In[ ]:


# compose rhe full grid as a tensor of all players
obj.grid_compose_vs = obj.p1.pursuer.grid.vs + obj.p1.evader.grid.vs +                       obj.p2.pursuer.grid.vs + obj.p2.evader.grid.vs
print([x.shape for x in obj.grid_compose_vs])

# too big
#FullTensor = np.meshgrid(*obj.grid_compose_vs, indexing='ij')


# In[ ]:


# direction cosine of p1.pursuer w.r.t p1.evader
obj.p1.pursuer.basis_p1evader = np.dot(obj.p1.pursuer.basis.T, obj.p1.evader.basis)


# In[ ]:


p1_pursuer = np.asarray(obj.p1.pursuer.grid.xs).transpose([1,2,3,0])
p1_evader = np.asarray(obj.p1.evader.grid.xs).transpose([1,2,3,0])
p2_pursuer = np.asarray(obj.p2.pursuer.grid.xs).transpose([1,2,3,0])
p2_evader = np.asarray(obj.p2.evader.grid.xs).transpose([1,2,3,0])
FT = np.concatenate((p1_pursuer, p1_evader), )
FT.shape

# use HOSVD to compute optimal rotation vector of one agent's state space to another's state space

def kabsch(P, Q):
    """
    Using the Kabsch algorithm with two sets of paired point P and Q, centered
    around the centroid. Each vector set is represented as an NxD
    matrix, where D is the the dimension of the space.
    The algorithm works in three steps:
    - a centroid translation of P and Q (assumed done before this function
      call)
    - the computation of a covariance matrix C
    - computation of the optimal rotation matrix U
    For more info see http://en.wikipedia.org/wiki/Kabsch_algorithm
    Parameters
    ----------
    P : array
        (N,D) matrix, where N is points and D is dimension.
    Q : array
        (N,D) matrix, where N is points and D is dimension.
    Returns
    -------
    U : matrix
        Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(P), Q)

    # Computation of the optimal rotation matrix
    # This can be done using singular value decomposition (SVD)
    # Getting the sign of the det(V)*(W) to decide
    # whether we need to correct our rotation matrix to ensure a
    # right-handed coordinate system.
    # And finally calculating the optimal rotation matrix U
    # see http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = np.linalg.svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


# In[ ]:


a = np.arange(9).reshape(3,3)
b, c = 2*a, 3*a


# In[ ]:


np.minimum(a, b, c)


# In[ ]:


a = a.flatten()
a[1:-1]


# In[ ]:




