import time
from hcipy.util import large_poisson
import numpy as np
from mkid_detect.utils import remove_deadtime
from mkid_detect.cosmic_rays import cosmic_rays
from mkid_detect.photon_list import PhotonList
from mkid_detect.logger import logger
from tqdm import tqdm


PhotonNumpyType = np.dtype([('time', np.uint32),
                            ('wavelength', np.float32),
                            ('x', np.uint32),
                            ('y', np.uint32)])


class MKIDDetect:
    def __init__(self, cr_rate, sat_rate, QE, R, R_std, dead_time, pixel_pitch,
                 dark_photon_rate, dead_pixel_mask=None, hot_pixel_mask=None):
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
        dead_pixel_mask: np.ndarray
            Boolean array specifying the location of dead pixels.
        hot_pixel_mask: np.ndarray
            Boolean array specifying the location of hot pixels.

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
        self.dead_pixel_mask = dead_pixel_mask
        self.hot_pixel_mask = hot_pixel_mask

        self.tau = 0.1  # photon correlation time
        self.taufac = 500  # bin fraction (us)

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

    def estimate_table_size(self, exp_time, fluxmap):
        """Estimate size of HDF5 file from expected number of photons."""
        fluxmap[fluxmap > self.sat_rate] = self.sat_rate
        photons = [f * exp_time for f in fluxmap]
        total_photons = np.sum(photons)

        # 1.6 MB for 1e5 photons
        estimated_memory = (1.6 * total_photons) / 1.0e5

        return estimated_memory

    def sim_output(self, fluxmap, exp_time, wavelengths, max_mem=10.0,  save_dir=''):
        """Simulate an MKID exposure

        Given an (optionally wavelength dependent) input flux map, exposure time,
        and list of wavelengths corresponding to the flux map, outputs a series of
        simulated MKID photon list including any noise sources. The output will be broken
        up into multiple PhotonLists such that the size of any one file does not
        exceed the specified max_mem value.

        Parameters
        ----------
        fluxmap: np.ndarray
            Array of fluxes for each pixel at each wavelength.
        exp_time: float
            Total duration of the observation (s).
        wavelengths: list
            Discretized wavelengths to use for this observation.
        max_mem: float
            Maxmimum size of output HDF5 mile (MB).
        save_dir: str
            The directory to save the generated HDF5 files to.

        Returns
        -------
        pls: list
            Output PhotonLists for the desired observation.
        """
        pls = []
        start_time = int(time.time())
        estimated_total_mem = self.estimate_table_size(exp_time, fluxmap)

        if estimated_total_mem > 1e6:
            raise MemoryError("This file will be greater than 1 TB, please lower your desired exposure time.")

        if estimated_total_mem < max_mem:
            start_times = np.array([start_time])
            exp_times = np.array([exp_time])

        else:
            num_files = int(estimated_total_mem / max_mem)
            logger.info(f"Using {num_files} files for generating total exposure")
            exp_time /= num_files
            start_times = np.array([int(start_time + i*exp_time) for i in range(num_files)])
            exp_times = np.array([exp_time for i in start_times])

        logger.info(f"Simulating {len(exp_times)} photon lists from the provided data.")

        for i, t in enumerate(exp_times):
            pl = self.sim_exposure(fluxmap, t, wavelengths, max_mem=max_mem, start_time=start_times[i], save_dir=save_dir)
            pls.append(pl)

        return pls

    def sim_exposure(self, fluxmap, exp_time, wavelengths, max_mem=10.0, start_time=None, save_dir=''):
        """Simulate an MKID exposure.

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
        start_time: int
            Unix timestamp for the start of the observation. Used for HDF5 file name.
        save_dir: str
            The directory to save the generated HDF5 files to.

        Returns
        -------
        pl: PhotonList
            Instance of the PhotonList class containing all the photons for a
            given fluxmap and noise parameters.
        """
        dims = np.shape(fluxmap)
        if dims[0] != len(wavelengths):
            raise AssertionError(f"Spectral dimensions of the fluxmap ({dims[0]}) does "
                                 f"not match the specified wavelengths ({wavelengths})")

        if self.dead_pixel_mask is None:
            self.dead_pixel_mask = np.zeros_like(fluxmap[0])

        if self.hot_pixel_mask is None:
            self.hot_pixel_mask = np.zeros_like(fluxmap[0])

        for flux in fluxmap:
            flux *= ~self.dead_pixel_mask
            flux[self.hot_pixel_mask] = self.sat_rate
            flux *= self.QE
            flux += self.dark_photon_rate * exp_time

        fluxmap = large_poisson(fluxmap)

        if self.R_map is None:
            self.R_map = np.random.normal(loc=self.R, scale=self.R_std, size=np.shape(fluxmap[0]))

        estimated_total_mem = self.estimate_table_size(exp_time, fluxmap)

        if int(estimated_total_mem) > max_mem:
            raise MemoryError('The file you are asking for is bigger than the set maximum. '
                              'Try decreasing your integration time.')

        # Calculate maximum number of expected photons.
        hits_per_sec = (dims[1] * self.pixel_pitch) * (dims[2] * self.pixel_pitch) * self.cr_rate
        total_hits = int(hits_per_sec * exp_time)

        max_photons = int(np.sum(fluxmap * exp_time) + (total_hits * dims[1] * dims[2]))

        photons = np.zeros(max_photons, dtype=PhotonNumpyType)
        photons = np.ascontiguousarray(photons)

        total_phot = 0

        with tqdm(total=max_photons) as pbar:
            for i, wvl in enumerate(wavelengths):
                for (x, y), val in np.ndenumerate(fluxmap[i]):
                    if val > self.sat_rate:
                        val = self.sat_rate

                    measured_times = self.get_photon_arrival_times(val, exp_time)
                    measured_wvls = get_photon_wavelengths(wvl, self.R_map[x, y], size=len(measured_times))

                    xs = np.full(np.shape(measured_times), x)
                    ys = np.full(np.shape(measured_times), y)

                    n_phot = len(measured_times)
                    new_phot = total_phot + n_phot

                    photons[total_phot:new_phot]['time'] = measured_times
                    photons[total_phot:new_phot]['wavelength'] = measured_wvls
                    photons[total_phot:new_phot]['x'] = xs
                    photons[total_phot:new_phot]['y'] = ys

                    total_phot += n_phot
                    pbar.update(n_phot)

        if self.cr_rate > 0:
            logger.info('Adding cosmic ray counts.')
            cr_xs, cr_ys, cr_wvls, cr_times = cosmic_rays(np.shape(fluxmap)[1], np.shape(fluxmap)[2],
                                                              self.cr_rate, exp_time, self.pixel_pitch)
            for j, hit in enumerate(cr_times):
                n_phot = len(cr_times[j])
                new_phot = total_phot + n_phot

                photons[total_phot:new_phot]['time'] = cr_times[j]
                photons[total_phot:new_phot]['wavelength'] = cr_wvls[j]
                photons[total_phot:new_phot]['x'] = cr_xs[j]
                photons[total_phot:new_phot]['y'] = cr_ys[j]

                total_phot += n_phot

        # Remove any leftover photon rows.
        photons = photons[photons['time'] != 0]

        # Instantiate PhotonList.
        if not start_time:
            start_time = int(time.time())
        pl = PhotonList(start=start_time, save_dir=save_dir)

        # Add photons to the photon list.
        pl.add_photons(np.array(photons['time']),
                       np.array(photons['wavelength']),
                       np.array(photons['x']),
                       np.array(photons['y']))

        pl.close()
        logger.info("Photon list generation completed.")

        return pl

    def sim_output_image(self, fluxmap, exp_time, wavelengths):
        """ Lightweight method for simulating an output without making photon lists.

        Note that this integrates over all the temporal information available to an MKID which
        may not be desired.

        Parameters
        ----------
        fluxmap: np.ndarray
            Array of fluxes for each pixel at each wavelength.
        exp_time: float
            Total duration of the observation (s).
        wavelengths: list
            Discretized wavelengths to use for this observation.

        Returns
        -------
        images: np.ndarray
            Array of the (optionally wavelength dependent) images output by the MKID detector
        """
        dims = np.shape(fluxmap)
        images = np.zeros_like(fluxmap)

        if dims[0] != len(wavelengths):
            raise AssertionError(f"Spectral dimensions of the fluxmap ({dims[0]}) does "
                                 f"not match the specified wavelengths ({wavelengths})")

        if self.dead_pixel_mask is None:
            self.dead_pixel_mask = np.zeros_like(fluxmap[0])

        if self.hot_pixel_mask is None:
            self.hot_pixel_mask = np.zeros_like(fluxmap[0])

        for i, flux in enumerate(fluxmap):
            flux *= ~self.dead_pixel_mask
            flux[self.hot_pixel_mask] = self.sat_rate
            flux *= self.QE

            flux *= exp_time
            flux += self.dark_photon_rate * exp_time

            # Add one photon per pixel per cosmic ray hit.
            array_area = (np.shape(flux)[0] * self.pixel_pitch) * (np.shape(flux)[1] * self.pixel_pitch)
            hits_per_sec = array_area * self.cr_rate
            total_hits = int(hits_per_sec * exp_time)
            flux += np.full_like(flux, total_hits)

            # Finally, add Poisson noise.
            images[i] = large_poisson(flux)

        return images


def get_photon_wavelengths(true_wvl, R, size):
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
