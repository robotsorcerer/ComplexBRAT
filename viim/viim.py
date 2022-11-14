import numpy as np


def levelsetstepsign(payoff, gam, h, dt, ren, eps):
    """
    evolves payoffs and builds the signed distance functions
    =========
    payoff: a list of cp or np arrays
    """
    (n, m) = payoff[0].shape
    me = .000001
    for i in range(len(payoff)):
        u = np.reshape(payoff[i], (n, m), order='F') - eps
        for t in range(ren):
            ex = np.pad(u, ((0, 1), (0, 0)), 'wrap')
            fx = np.diff(ex, 1, 0) / h
            ey = np.pad(u, ((0, 0), (0, 1)), 'wrap')
            fy = np.diff(ey, 1, 1) / h

            ex = np.pad(u, ((1, 0), (0, 0)), 'wrap')
            bx = np.diff(ex, 1, 0) / h
            ey = np.pad(u, ((0, 0), (1, 0)), 'wrap')
            by = np.diff(ey, 1, 1) / h

            normgradc = np.sqrt(((fx + bx) / 2) ** 2 + ((fy + by) / 2) ** 2 + me)

            normgrad = np.sqrt(fx ** 2 + fy ** 2 + me)
            ex = np.pad(fx / normgrad, ((1, 0), (0, 0)), 'wrap')
            bx = np.diff(ex, 1, 0) / h
            ey = np.pad(fy / normgrad, ((0, 0), (1, 0)), 'wrap')
            by = np.diff(ey, 1, 1) / h

            curvature = bx + by
            u = u + dt * gam[i] * curvature * normgradc

        payoff[i] = u

    return payoff


def dictmapping(payoff, h, level, FLAG, map, funcs, DMIIM):
    """
    UNDER CONSTRUCTION
    Flag:  0 == taxicab metric
           1 == First-order accurate redistancing (e.g. fast marching method of Sethian;
                    equivalently, method of Tsitsiklis)
           2 == Directional optimization (biquadratic interpolation)
           3 == Directional optimization (bicubic interpolation)
    """
    c = len(payoff)
    (n, m) = payoff[0].shape
    all = list(range(c))
    redists = np.zeros((c, n, m))
    for i in range(c):
        temp = np.reshape(payoff[i], (n, m), order='F')

    return payoff

def viim(phi, gamma, h, dt, eps, N, ren, map, width, FLAG, funcs, DMIIM):
    for k in range(1, N):
        phi = levelsetstepsign(phi, gamma, h, dt, ren, eps)
        phi = dictmapping(phi, h, width, FLAG, map, funcs, DMIIM)
    return phi
