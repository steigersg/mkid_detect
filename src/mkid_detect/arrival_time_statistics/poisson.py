import numpy as np


def poisson_arrival_times(flux, exp_time, tau, taufac):
    """Get a list of Poisson distributed arrival times

    Parameters
    ----------
    flux: float
        Input flux in photons/s.
    exp_time: float
        Exposure time in s.
    tau: float
        Decorrelation timescale (s).
    taufac: float
        Bin fraction for dicretization (us).

    Returns
    -------
    tlist: list
        Photon arrival times following Poisson statistics with decorrelation time tau.
    """
    # Find size of time bins.
    N = max(int(tau * 1e6 / taufac), 1)

    # Generate discretized time bins.
    t = np.arange(0, int(exp_time * 1e6), N)

    # Find expected number of photons per bin.
    n = np.random.poisson(np.ones(t.shape) * flux / 1e6 * N)

    tlist = t[n > 0] * 1.

    # Add a random number to find exact time for each photon within the bin.
    tlist += N * np.random.rand(len(tlist))
    return tlist