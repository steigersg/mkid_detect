import numpy as np


def cosmic_rays(xdim, ydim, cr_rate, exp_time, pixel_pitch):
    """Generates photon entries caused by a cosmic ray hit on the detector.

    Calculates the expected number of cosmic ray hits given the cosmic ray rate,
    pixel dimensions, and pixel pitch. Then generates and returns the expected photon entries.
    This function assumes a cosmic ray event will light up every pixel on the
    array within a 20 microsecond window as per https://iopscience.iop.org/article/10.3847/1538-3881/ac5833.

    Parameters
    ----------
    xdim: int
        Number of pixels in x-dimension.
    ydim: int
        Number of pixels in x-dimension.
    cr_rate:
        Cosmic ray rate (events/cm^2/s)
    exp_time: float
        Total duration of the observation.
    pixel_pitch: float
        Pixel pitch (cm).

    Returns
    -------
    cr_xs: np.ndarray
        Array of x-coordinates where a cosmic ray photon hit.
    cr_ys: np.ndarray
        Array of y-coordinates where a cosmic ray photon hit.
    cr_wvls: np.ndarray
        Array of registered wavelengths for each cosmic ray photon hit.
    cr_times: np.ndarray
        Array of times when a cosmic ray photon was registered.
    """
    # to start just assume all pixels get saturated
    array_area = (xdim * pixel_pitch) * (ydim * pixel_pitch)
    hits_per_sec = array_area * cr_rate
    total_hits = int(hits_per_sec * exp_time)

    # each hit happens ar a random time on a random pixel
    possible_times = np.arange(0, exp_time * 1e6, 10, dtype=int)

    hit_times = np.random.choice(possible_times, int(total_hits))

    # use meshgrid to get every coordinate
    grid = np.meshgrid(np.arange(0, xdim, 1), np.arange(0, ydim, 1))
    cr_xs = grid[0].flatten()
    cr_ys = grid[1].flatten()

    # give every pixel 1 count within +/- 10 microseconds of the hit time
    # later can add fireball to get better timing
    cr_times = []
    for i, time in enumerate(hit_times):
        cr_times.append(np.random.choice(np.arange(time-10, time+10, 1), len(cr_xs)))

    cr_times = np.array(cr_times, dtype=int)
    # generate a random wavelength
    cr_wvls = np.random.choice(np.arange(100, 1000, 1), size=(total_hits, len(cr_xs)))

    cr_xs = np.tile(cr_xs, (total_hits, 1))
    cr_ys = np.tile(cr_ys, (total_hits, 1))
    return cr_xs, cr_ys, cr_wvls, cr_times
