__helper = """
                lifter = Lift(value_func)
                x, cost_hist = lifter.solve(cp.asarray(value_rob))
            """

from LevelSetPy.Optimization import chambollepock
import cupy as cp

class Lift(object):
    def __init__(self, value_orig = None, max_iter=1500):
        self.value_orig = value_orig # prescribed_dose
        self.max_iter = max_iter
        self.cpk     = chambollepock()

    def cost_func(self, b):
        "Solve the total variation problem under L2 regularization"
        def buff(K, x):
            cost = 0
            for i in range(len(K)):
                cost+=cp.sum(cp.power(K[i](x)-b[i],2))
            cost = 0.5 * cost
            return cost
        return buff

    def solve(self, value_rom=None):
        'Solves for the beamlet intensities, x. Returns x as well as a history of the cost function'
        # bundle = get_mask(case, self.f_mask, self.f_dij)
        A      = self.value_orig
        self.cpk.variables.reset_variables()
        self.cpk.define.K_datamatrix(A, sparsemat=False, sparsetype='csr')
        x_testing = 'cc_l2s'#cp.proxops.avail_ops[3]

        self.cpk.define.dualprox(operation=[x_testing]*6, b=value_rom)
        self.cpk.define.primalprox(operation=self.cpk.proxops.prox_I_lb_ub(lb=0))
        self.cpk.variables.costfunc = self.cost_func(b=value_rom)
        self.cpk.variables.maxiter  = self.max_iter

        x, cost_hist    = self.cpk.optimizer.lazy_cpk(continue_opt=False)

        return x, cost_hist