import time
import numpy as np
import matplotlib.pyplot as plt
from utils import remove_deadtime
from cosmic_rays import cosmic_rays
from photon_list import PhotonList


class MKIDDetect:
    def __init__(self, cr_rate, sat_rate, QE, R, R_std, dead_time, pixel_pitch, dark_photon_rate):
        """ Create an MKID output photon list for a given input fluxmap.

        This class includes the sim_output utility which returns an output photon list
        for a given input fluxmap. A series of wavelength dependent flux maps can also be
        given.

        Parameters
        ----------
        cr_rate: float
            Cosmic ray rate (events/cm^2/s).
        sat_rate: float
            Saturation rate of the MKID (counts/s/wvl).
        QE: float
            Quantum Efficiency.
        R: int
            MKID average energy resolution.
        R_std: float
            Standard deviation in energy resolution per pixel.
        dead_time: int
            MKID pixel dead time (microseconds).
        pixel_pitch: float
            Pixel pitch (cm).
        dark_photon_rate:
            Rate of expected counts due to background sources of radiation (counts/s).
        save_dir: str
            Directory where to save generated HDF5 files.
        """
        self.cr_rate = cr_rate
        self.sat_rate = sat_rate
        self.QE = QE
        self.R = R
        self.R_std = R_std
        self.dead_time = dead_time  # in us
        self.dark_photon_rate = dark_photon_rate
        self.start = 0
        self.R_map = None
        self.pixel_pitch = pixel_pitch  # in cm

        self.tau = 0.1  # photon correlation time
        self.taufac = 500

    def get_photon_wavelengths(self, true_wvl, R, size):
        """Get array of wavelengths degraded by resolution R.

        Generates a wavelength for a number of photons given the true
        wavelength that those photons should have and the energy resolution of
        the pixel.

        Parameters
        ----------
        true_wvl: float
            The true wavelength of the photon (in nm)
        R: float
            The energy resolution to use.
        size: int
            How many draws to make from the normal distribution defined by R.

        Returns
        -------
        measured_wvl: np.ndarray
            Array of photon wavelengths with uncertainty given by R.
        """
        del_E = true_wvl / R
        measured_wvl = np.random.normal(loc=true_wvl, scale=del_E, size=size)
        return measured_wvl

    def get_photon_arrival_times(self, flux, exp_time):
        """Get array of arrival times.

        Returns an array of expected photon arrival times given an input flux.
        Accounts for poisson statistics and the MKID array deadtime.

        Parameters
        ----------
        flux: float
            Expected flux in the given pixel (counts/s)
        exp_time: float
            Total duration of the observation.

        Returns
        -------
        keep_times: list
            Photon arrival times (microseconds).
        """
        # this is the easiest thing to do (poisson), can later implement other
        # arrival time statistics, i.e. MR

        N = max(int(self.tau * 1e6 / self.taufac), 1)

        # Generate discretized time bins.
        t = np.arange(0, int(exp_time * 1e6), N)
        n = np.random.poisson(np.ones(t.shape) * flux / 1e6 * N)

        tlist = t[n > 0] * 1.

        tlist += N * np.random.rand(len(tlist))

        # Remove photons that arrive too close to the preceding photon.
        keep_times = remove_deadtime(tlist, dead_time=self.dead_time)

        return keep_times

    def estimate_table_size(self, exp_time, fluxmap, num_wavelengths):
        total_photons = np.sum(fluxmap * exp_time * num_wavelengths)
        # 1.6 MB for 1e5 photons
        estimated_memory = (1.6 * total_photons) / 1.0e5
        return estimated_memory

    def sim_output(self, fluxmap, exp_time, wavelengths, max_mem=10.0,  save_dir=''):
        """Simulate an MKID output.

        Given an (optionally wavelength dependent) input flux map, exposure time,
        and list of wavelengths corresponding to the flux map, outputs a simulated
        MKID photon list including any noise sources.

        Parameters
        ----------
        fluxmap: np.ndarray
            Array of fluxes for each pixel at each wavelength.
        exp_time: float
            Total duration of the observation.
        wavelengths: list
            Discretized wavelengths to use for this observation.
        max_mem: float
            Maxmimum size of ourput HDF5 mile (MB)

        Returns
        -------
        pl: PhotonList
            Instance of the PhotonList class containing all the photons for a
            given fluxmap and noise parameters.
        """
        dims = np.shape(fluxmap)
        assert dims[0] == len(wavelengths)

        for map in fluxmap:
            map *= self.QE
            map += self.dark_photon_rate * exp_time

        if not self.R_map:
            self.R_map = np.random.normal(loc=self.R, scale=self.R_std, size=np.shape(fluxmap[0]))

        estimated_total_mem = self.estimate_table_size(exp_time, fluxmap, len(wavelengths))

        if estimated_total_mem > max_mem:
            raise MemoryError('The file you are asking for is bigger than the set maximum. '
                              'Try decreasing your integration time.')

        self.start = int(time.time())

        pl = PhotonList(start=self.start)

        if not self.R_map:
            self.R_map = np.random.normal(loc=self.R, scale=self.R_std, size=np.shape(fluxmap[0]))

        for i, wvl in enumerate(wavelengths):
            for (x, y), val in np.ndenumerate(fluxmap[i]):
                measured_times = self.get_photon_arrival_times(val, exp_time)
                measured_wvls = self.get_photon_wavelengths(wvl, self.R_map[x, y], size=len(measured_times))

                if len(measured_times) > (self.sat_rate * exp_time):
                    measured_times = measured_times[:self.sat_rate * exp_time]
                    measured_wvls = measured_wvls[:self.sat_rate * exp_time]

                xs = np.full(np.shape(measured_times), x)
                ys = np.full(np.shape(measured_times), y)

                pl.add_photons(measured_times, measured_wvls, xs, ys)

        cr_xs, cr_ys, cr_wvls, cr_times = cosmic_rays(np.shape(fluxmap)[1], np.shape(fluxmap)[2],
                                                      self.cr_rate, exp_time, self.pixel_pitch)
        for j, hit in enumerate(cr_times):
            pl.add_photons(cr_times[j], cr_wvls[j], cr_xs[j], cr_ys[j])

        return pl


if __name__ == '__main__':
    mkid = MKIDDetect(1, 5000, 0.9, 1000, 0.2, 10,  0.03, dark_photon_rate=1e-3)

    focal_image = np.ones((50, 50))

    for (x, y), i in np.ndenumerate(focal_image):
        if x == y:
            focal_image[x, y] = 10

    pl = mkid.sim_output([focal_image], 1, [600])
    im = pl.generate_image()
    plt.imshow(im)
    plt.colorbar()
    plt.show()