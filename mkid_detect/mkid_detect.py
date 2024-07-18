from photon_list import PhotonList
import time
import numpy as np
from utils import remove_deadtime
from cosmic_rays import cosmic_rays

class MKIDDetect:
    def __init__(self, cr_rate, sat_rate, QE, R, R_std, dead_time, pixel_pitch, save_dir=''):
        self.cr_rate = cr_rate
        self.sat_rate = sat_rate
        self.QE = QE
        self.R = R
        self.R_std = R_std
        self.dead_time = dead_time  # in us
        self.save_dir = save_dir
        self.start = 0
        self.R_map = None
        self.pixel_pitch = pixel_pitch  # in cm

        self.tau = 0.1  # photon correlation time
        self.taufac = 500

    def get_photon_wavelengths(self, true_wvl, R, size):
        """
        Generates a wavelength for a number of photons given the true
        wavelength that those photons should have and the energy reosltuion of
        your pixel.

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

        """
        del_E = true_wvl / R
        return np.random.normal(loc=true_wvl, scale=del_E, size=size)

    def get_photon_arrival_times(self, flux, exp_time):
        # this is the easiest thing to do (poisson), can later implement other
        # arrival time statistics, i.e. MR
        # exp_time is in seconds
        # flux is in photons/s

        N = max(int(self.tau * 1e6 / self.taufac), 1)

        # generate discretize time bins
        t = np.arange(0, int(exp_time * 1e6), N)
        n = np.random.poisson(np.ones(t.shape) * flux / 1e6 * N)

        tlist = t[n > 0] * 1.

        tlist += N * np.random.rand(len(tlist))

        keep = remove_deadtime(tlist, dead_time=self.dead_time)

        return keep

    def sim_output(self, fluxmap, exp_time, wavelengths):
        # fluxmap in phot/pix/s/nm
        # exp_time in seconds
        dims = np.shape(fluxmap)
        assert dims[0] == len(wavelengths)

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
        for i, hit in enumerate(cr_times):
            pl.add_photons(cr_times[i], cr_wvls[i], cr_xs[i], cr_ys[i])

        return pl


if __name__ == '__main__':
    mkid = MKIDDetect(1, 5000, 0.9, 1000, 0.2, 10,  0.03)

    focal_image = np.ones((50, 50))

    for (x, y), i in np.ndenumerate(focal_image):
        if x == y:
            focal_image[x, y] = 10

    pl = mkid.sim_output([focal_image], 1, [600])
    im = pl.generate_image()
    plt.imshow(im)
    plt.colorbar()
    plt.show()