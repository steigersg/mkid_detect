"""
Get modified Rician (MR) distributed arrival times for a given input flux. MR distributed arrival
time statistics are applicable for off-axis stellar speckles behind an extreme AO system. For more
details see https://doi.org/10.48550/arXiv.1906.03354 and https://doi.org/10.48550/arXiv.2209.06312
"""

import numpy as np
from scipy import special, interpolate
from mkid_detect.arrival_time_statistics.statistics_utils import corrsequence


def mr_arrival_times(Ic, Is, exp_time, tau, taufac):
    """Get the list of MR distributed arrival times.

    Parameters
    ----------
    Ic: float
        Ic (coherent intensity) value.
    Is: float
        Is (time-varying intensity) value.
    exp_time: float
        Exposure time to use (s).
    tau: float
        Decorrelation timescale (s).
    taufac: float
        Bin fraction for dicretization (us).
    Returns
    -------
    tlist: list
        Photon arrival times following MR statistics with decorrelation time tau.
    """
    # Find size of time bins.
    N = max(int(tau * 1e6 / taufac), 1)

    # Generate discretized time bins.
    t, normal = corrsequence(int(exp_time * 1e6 / N), tau * 1e6 / N)

    uniform = 0.5 * (special.erf(normal / np.sqrt(2)) + 1)
    t *= N
    f = mr_icdf(Ic, Is)

    # Calculate expected intensity.
    I = f(uniform) / 1e6

    # Find expected number of photons per bin.
    n = np.random.poisson(I * N)

    tlist = t[n > 0] * 1.

    # Add a random number to find exact time for each photon within the bin.
    tlist += N * np.random.rand(len(tlist))
    return tlist


def mr_icdf(Ic, Is, interpmethod='cubic'):
    """Get the inverse CDF of the modified Rician.

    Parameters
    ----------
    Ic: float
        Ic (coherent intensity) value.
    Is: float
        Is (time-varying intensity) value.
    interpmethod: str
        Interpolation method to use for interpolate.interp1d.

    Returns
    -------
    func:
        interpolation function f for the inverse CDF of the MR.
    """

    if Is <= 0 or Ic < 0:
        raise ValueError("Cannot compute modified Rician CDF with Is<=0 or Ic<0.")

    # Compute mean and variance of modified Rician, compute CDF by
    # going 15 sigma to either side (but starting no lower than zero).
    # Use 1000 points, or about 30 points/sigma.

    mu = Ic + Is
    sig = np.sqrt(Is ** 2 + 2 * Ic * Is)
    I1 = max(0, mu - 15 * sig)
    I2 = mu + 15 * sig
    I = np.linspace(I1, I2, 1000)

    # Grid spacing.  Set I to be offset by dI/2 to give the
    # trapezoidal rule by direct summation.

    dI = I[1] - I[0]
    I += dI / 2

    # Modified Rician PDF, and CDF by direct summation of intensities
    # centered on the bins to be integrated.  Enforce normalization at
    # the end since our integration scheme is off by a part in 1e-6 or
    # something.

    # p_I = 1./Is*np.exp(-(Ic + I)/Is)*special.iv(0, 2*np.sqrt(I*Ic)/Is)
    p_I = modified_rician(I, Ic, Is)

    cdf = np.cumsum(p_I) * dI
    cdf /= cdf[-1]

    # The integral is defined with respect to the bin edges.

    I = np.asarray([0] + list(I + dI / 2))
    cdf = np.asarray([0] + list(cdf))

    # The interpolation scheme doesn't want duplicate values.  Pick
    # the unique ones, and then return a function to compute the
    # inverse of the CDF.

    i = np.unique(cdf, return_index=True)[1]
    return interpolate.interp1d(cdf[i], I[i], kind=interpmethod)


def modified_rician(I, Ic, Is):
    """Define a modified Rician function."""
    mr = 1. / Is * np.exp((2 * np.sqrt(I * Ic) - (Ic + I)) / Is) * special.ive(0, 2 * np.sqrt(I * Ic) / Is)
    return mr