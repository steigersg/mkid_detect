import numpy as np
from scipy import special, interpolate, integrate
from mkid_detect.arrival_time_statistics.statistics_utils import corrsequence
from scipy.stats import gamma


def gamma_arrival_times(flux, exp_time, tau, taufac, mean_strehl=0.9, beta=5, alpha=30):
    """Get the list of Gamma distributed arrival times.

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
    mean_strehl: float
        Median Strehl ratio over the exposure.
    beta: float
        Beta factor that defines the Gamma distribution.
    alpha: float
        Alpha factor that defines the Gamma distribution.

    Returns
    -------
    tlist: list
        Photon arrival times following Gamma statistics with decorrelation time tau.
    """

    N = max(int(tau * 1e6 / taufac), 1)

    t, normal = corrsequence(int(exp_time * 1e6 / N), tau * 1e6 / N)
    uniform = 0.5 * (special.erf(normal / np.sqrt(2)) + 1)
    t *= N
    f = gamma_icdf(mean_strehl, flux, bet=beta, alph=alpha)

    I = f(uniform) / 1e6
    n = np.random.poisson(I * N)

    tlist = t[n > 0] * 1.

    tlist += N * np.random.rand(len(tlist))
    return tlist


def p_A(x, gam=None, bet=None, alph=None):
    """Define the Gamma PDF"""
    pdf = gamma.pdf(x, alph, loc=gam, scale=1 / bet)
    return pdf


def gamma_icdf(median_strehl, Ip, bet=None, alph=None, interpmethod='cubic'):
    """

    Parameters
    ----------
    median_strehl: float
    Ip: float
        Intensity components that follows the Gamma distributed arrival times.
    bet: float
        Beta factor that defines the Gamma distribution.
    alph: float
        Alpha factor that defines the Gamma distribution.
    interpmethod: str
        Interpolation method to pass to interpolate.interp1d.

    Returns
    -------
    func:
        interpolation function f for the inverse CDF of the Gamma function.
    """
    # compute mean and variance of gamma distribution
    sr1 = 0.1  # max(0, mu - 15 * sig)
    sr2 = 1.0  # min(mu + 15 * sig, 1.)
    sr = np.linspace(sr1, sr2, 1000)
    gam = -(median_strehl + (1 - median_strehl))
    p_I = (1. / (2 * np.sqrt(sr))) * (p_A(np.sqrt(sr), gam=gam, alph=alph, bet=bet) +
                                      p_A(-np.sqrt(sr), gam=gam, alph=alph, bet=bet))
    norm = integrate.simps(p_I)
    p_I /= norm
    # go from strehls to intensities
    I = (sr * Ip) / median_strehl
    dI = I[1] - I[0]
    I += dI / 2
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