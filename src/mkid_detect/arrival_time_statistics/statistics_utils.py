import numpy as np


def corrsequence(Ttot, tau):
    """
    Generate a sequence of correlated Gaussian noise, correlation time
    tau.  Algorithm is recursive and from Markus Deserno.  The
    recursion is implemented as an explicit for loop but has a lower
    computational cost than converting uniform random variables to
    modified-Rician random variables.

    Arguments:
    Ttot: int, the total integration time in microseconds.
    tau: float, the correlation time in microseconds.

    Returns:
    t: a list of integers np.arange(0, Ttot)
    r: a correlated Gaussian random variable, zero mean and unit variance, array of length Ttot

    """

    t = np.arange(Ttot)
    g = np.random.normal(0, 1, Ttot)
    r = np.zeros(g.shape)
    f = np.exp(-1. / tau)
    sqrt1mf2 = np.sqrt(1 - f ** 2)
    r = recursion(r, g, f, sqrt1mf2, g.shape[0])
    return t, r


def recursion(r, g, f, sqrt1mf2, n):
    for i in range(1, n):
        r[i] = r[i - 1]*f + g[i]*sqrt1mf2
    return r


