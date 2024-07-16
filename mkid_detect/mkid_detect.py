from photon_list import PhotonList
import time
import numpy as np
from utils import remove_deadtime

class MKIDDetect:
    def __init__(self, cr_rate, sat_rate, QE, R, R_std, dead_time, max_phot_per_file, save_dir=''):
        self.cr_rate = cr_rate
        self.sat_rate = sat_rate
        self.QE = QE
        self.R = R
        self.R_std = R_std
        self.dead_time = dead_time # in us
        self.max_phot_per_file = max_phot_per_file
        self.save_dir = save_dir
        self.start = 0
        self.R_map = None
        self.tau = 0.1  # photon correlation time
        self.taufac = 500

    def get_photon_wavelengths(self, true_wvl, R, size):
        del_E = true_wvl / R
        return np.random.normal(loc=true_wvl, scale=del_E, size=size)

    def get_photon_arrival_times(self, I, exp_time):
        # exp_time is in seconds
        # I is in photons/s

        N = max(int(self.tau * 1e6 / self.taufac), 1)
        # generate discretize time bins for constant background
        t = np.arange(0, int(exp_time * 1e6), N)
        n = np.random.poisson(np.ones(t.shape) * I / 1e6 * N)

        tlist = t[n > 0] * 1.

        tlist += N * np.random.rand(len(tlist))

        #keep = remove_deadtime(tlist) # write in C
        keep = [int(t) for t in tlist]

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

                pl.add_photons(x, y, measured_wvls, measured_times)
