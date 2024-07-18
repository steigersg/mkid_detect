import numpy as np
import tables
from tables import *

class Photon(IsDescription):
    time = UInt32Col()
    wavelength = Float64Col()
    x = UInt16Col()
    y = UInt16Col()


class PhotonList:
    def __init__(self, start):
        """

        Parameters
        ----------
        start: float
            Start time of the observation (Unix timestamp).
        """
        self.start = start
        self.name = f"{self.start}.h5"

        self.table = None
        self.h5file = open_file(self.name, mode="w", title="Photon Table")

        group = self.h5file.create_group("/", 'MKID', 'MKID Information')

        self.table = self.h5file.create_table(group, 'readout', Photon, "MKID Readout File")

    def add_photons(self, times, wavelengths, x, y):
        """

        Parameters
        ----------
        times: np.ndarray
            Times of photons to be added (microseconds).
        wavelengths: np.ndarray
            Wavelengths of photons to be added (nm).
        x: np.ndarray
            x-coordinates of photons to be added.
        y: np.ndarray
            y-coordinates of photons to be added.

        Returns
        -------

        """
        self.table.append((times, wavelengths, x, y))
        self.table.flush()

    def get_column(self, col_name):
        """

        Parameters
        ----------
        col_name: str
            Name of column to query. Options are 'x', 'y', 'time', 'wavelength'.

        Returns
        -------

        """
        table = self.h5file.root.MKID.readout
        return [x[col_name] for x in table.iterrows()]

    def query_photons(self, start_wvl=None, stop_wvl=None, start_time=None, stop_time=None, pixel=None):
        """

        Parameters
        ----------
        start_wvl: float
            Starting wavelength (nm).
        stop_wvl: float
            Ending wavelength (nm).
        start_time: float
            Start time (s).
        stop_time:
            Stop time (s).
        pixel: (int, int)
            (x, y) pixel coordinate. If None will return the whole array.

        Returns
        -------

        """
        if pixel is not None:
            pixel_condition = f'(x == {pixel[0]}) & (y == {pixel[1]})'
        else:
            pixel_condition = ''

        if start_wvl is None and stop_wvl is None:
            wvl_condition = ''
        elif start_wvl and stop_wvl:
            wvl_condition = f'({start_wvl} < wavelength) & (wavelength < {stop_wvl})'
        elif start_wvl and not stop_wvl:
            wvl_condition = f'({start_wvl} < wavelength)'
        else:
            wvl_condition = f'(wavelength < {stop_wvl})'

        if start_time is None and stop_time is None:
            time_condition = '(0 < time)'
        elif start_time and stop_time:
            time_condition = f'({start_time*1e6} < time) & (time < {stop_time*1e6})'
        elif start_time and not stop_time:
            time_condition = f'({start_time*1e6} < time)'
        else:
            time_condition = f'(time < {stop_time*1e6})'

        condition = time_condition
        if wvl_condition != '':
            condition += '&' + wvl_condition
        if pixel_condition != '':
            condition += '&' + pixel_condition

        return self.table.read_where(condition)

    def generate_image(self, start_wvl=None, stop_wvl=None, start_time=None, stop_time=None):
        """

        Parameters
        ----------
        start_wvl: float
            Starting wavelength (nm).
        stop_wvl: float
            Ending wavelength (nm).
        start_time: float
            Start time (s).
        stop_time:
            Stop time (s).

        Returns
        -------

        """
        x_dim = np.max(self.get_column('x'))
        y_dim = np.max(self.get_column('y'))
        image = np.zeros((x_dim, y_dim))

        for (x, y), i in np.ndenumerate(image):
            photons = self.query_photons(start_wvl, stop_wvl, start_time, stop_time, pixel=(x, y))
            image[x, y] = len(photons)

        return image

