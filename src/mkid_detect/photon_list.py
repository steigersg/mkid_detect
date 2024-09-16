import os
import numpy as np
import tables
from tables import *
from tqdm import tqdm


class Photon(IsDescription):
    time = UInt32Col()
    wavelength = Float64Col()
    x = UInt16Col()
    y = UInt16Col()


class PhotonList:
    def __init__(self, start, save_dir=''):
        """Create and manipulate a PyTable containing photon entries.

        Contains methods for adding, querying, and generating images from
        a table of photons with properties given by the Photon class.

        Parameters
        ----------
        start: float
            Start time of the observation (Unix timestamp).
        """
        self.start = start
        self.save_dir = save_dir
        self.name = os.path.join(self.save_dir, f"{self.start}.h5")

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.table = None
        self.h5file = open_file(self.name, mode="w", title="Photon Table")

        group = self.h5file.create_group("/", 'MKID', 'MKID Information')

        self.table = self.h5file.create_table(group, 'readout', Photon, "MKID Readout File")

    @property
    def table_size(self):
        return self.table.size_on_disk

    def remove(self):
        os.remove(self.name)

    def add_photons(self, times, wavelengths, x, y):
        """Add photons to the PhotonList.

        Given array of times, wavelengths, x-coordinates, and y-coordinates, adds
        those photons to the end of the photon table.

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
        """Returns a column of the photon list.

        Returns the column of the photon table for the specified column name.

        Parameters
        ----------
        col_name: str
            Name of column to query. Options are 'x', 'y', 'time', 'wavelength'.

        Returns
        -------
        column: list
            Values of the specified column.
        """
        if self.h5file.isopen:
            table = self.h5file.root.MKID.readout
            column = [x[col_name] for x in table.iterrows()]
        else:
            with tables.open_file(self.name, mode='r') as h5file:
                table = h5file.root.MKID.readout
                column = [x[col_name] for x in table.iterrows()]
        return column

    def query_photons(self, start_wvl=None, stop_wvl=None, start_time=None, stop_time=None, pixel=None):
        """Returns photons satisfying the specified query conditions.

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
        filtered_table: Table
            Photons satisfying the query conditions.
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

        if self.h5file.isopen:
            filtered_table = self.table.read_where(condition)
        else:
            with tables.open_file(self.name, mode='r') as h5file:
                table = h5file.root.MKID.readout
                filtered_table = table.read_where(condition)
        return filtered_table

    def generate_image(self, start_wvl=None, stop_wvl=None, start_time=None, stop_time=None):
        """Generate an image from a PhotonList.

        Generates a 2D image array from the PhotonList satisfying the given
        conditions.

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
        image: np.ndarray
            The output image (counts).
        """
        x_dim = np.max(self.get_column('x'))
        y_dim = np.max(self.get_column('y'))
        image = np.zeros((x_dim, y_dim))

        with tqdm(total=x_dim*y_dim) as pbar:
            for (x, y), i in np.ndenumerate(image):
                photons = self.query_photons(start_wvl, stop_wvl, start_time, stop_time, pixel=(x, y))
                image[x, y] = len(photons)
                pbar.update(1)

        return image

    def close(self):
        self.h5file.close()
