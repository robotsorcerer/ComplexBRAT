__all__ = ["solve_brt"]
__comment__ = "!!!DEPRECATED!!! We now solve all in the main function."

import copy
import cupy as cp
import numpy as np

from LevelSetPy.Visualization import implicit_mesh
from LevelSetPy.Utilities import Bundle, cputime, eps, info
from LevelSetPy.ExplicitIntegration.Integration import odeCFL2, odeCFL3, odeCFLset

from LevelSetPy.ExplicitIntegration.Term import termRestrictUpdate

from BRATVisualization.rcbrt_visu import RCBRTVisualizer

def solve_brt(args, t_range, data, finite_diff_data):

	if args.visualize:
		viz = RCBRTVisualizer(params=args.params)


	t_plot = (t_range[1] - t_range[0]) / 10
	# """
	# ---------------------------------------------------------------------------
	#  Restrict the Hamiltonian so that reachable set only grows.
	#    The Lax-Friedrichs approximation scheme MUST already be completely set up.
	# """
	# innerData = copy.copy(finite_diff_data)
	# del finite_diff_data

	# # Wrap the true Hamiltonian inside the term approximation restriction routine.
	# finite_diff_data = Bundle(dict(innerFunc = termLaxFriedrichs,
	# 							   innerData = innerData,
	# 							   positive = False,  # direction to grow the updated level set
	# 							))

	small = 100*eps
	options = Bundle(dict(factorCFL=0.95, stats='on', singleStep='off'))

	# Loop through t_range (subject to a little roundoff).
	t_now = t_range[0]
	start_time = cputime()
	itr_start = cp.cuda.Event()
	itr_end = cp.cuda.Event()

	data_all = [data]

	while(t_range[1] - t_now > small * t_range[1]):
		itr_start.record()
		cpu_start = cputime()
		time_step = f"{t_now}/{t_range[-1]}"

		# Reshape data array into column vector for ode solver call.
		y0 = data.flatten()

		# How far to step?
		t_span = np.hstack([ t_now, min(t_range[1], t_now + t_plot) ])

		# Integrate a timestep.
		t, y, _ = odeCFL2(termRestrictUpdate, t_span, y0, odeCFLset(options), finite_diff_data)
		cp.cuda.Stream.null.synchronize()
		t_now = t

		# Get back the correctly shaped data array
		data = y.reshape(grid.shape)

		if args.visualize:
			data_np = data.get()
			mesh=implicit_mesh(data_np, level=0, spacing=args.spacing,
								edge_color='None',  face_color='red')
			viz.update_tube(data_np, mesh, time_step)

		itr_end.record()
		itr_end.synchronize()
		cpu_end = cputime()

		info(f't: {time_step} | GPU time: {(cp.cuda.get_elapsed_time(itr_start, itr_end)):.2f} | CPU Time: {(cpu_end-cpu_start):.2f}, | Targ bnds {min(y):.2f}/{max(y):.2f} Norm: {np.linalg.norm(y, 2):.2f}')

		# store this brt
		data_all.append(data.get())
	# # if we are done, update target set on frame II
	# if args.visualize:
	# 	viz.update_tube(data_np, args.init_mesh, time_step)

	end_time = cputime()
	info(f'Total BRS/BRT execution time {(end_time - start_time):.4f} seconds.')

	return t, data_all
