import numpy as np


def poisson_arrival_times(flux, exp_time, tau, taufac):
    # this is the easiest thing to do (poisson), can later implement other
    # arrival time statistics, i.e. MR

    N = max(int(tau * 1e6 / taufac), 1)

    # Generate discretized time bins.
    t = np.arange(0, int(exp_time * 1e6), N)
    n = np.random.poisson(np.ones(t.shape) * flux / 1e6 * N)

    tlist = t[n > 0] * 1.

    tlist += N * np.random.rand(len(tlist))
    return tlist