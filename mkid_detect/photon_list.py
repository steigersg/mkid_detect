from tables import *

class Photon(IsDescription):
    time = UInt32Col()
    wavelength = Float64Col()
    x = UInt16Col()
    y = UInt16Col()


class PhotonList:
    def __init__(self, start):
        self.start = start
        self.name = f"{self.start}.h5"

        self.table = None
        self.h5file = open_file(self.name, mode="w", title="Photon Table")

        group = self.h5file.create_group("/", 'MKID', 'MKID Information')

        self.table = self.h5file.create_table(group, 'readout', Photon, "MKID Readout File")

    def add_photons(self, times, wavelengths, x, y):
        self.table.append((times, wavelengths, x, y))
        self.table.flush()

    def get_column(self, col_name):
        table = self.h5file.root.MKID.readout
        return [x[col_name] for x in table.iterrows()]

    def query_photons(self, start_wvl=None, stop_wvl=None, start_time=None, stop_time=None):
        if start_wvl is None and stop_wvl is None:
            wvl_condition = ''
        elif start_wvl and stop_wvl:
            wvl_condition = f'({start_wvl} < wavelength) & (wavelength < {stop_wvl})'
        elif start_wvl and not stop_wvl:
            wvl_condition = f'({start_wvl} < wavelength)'
        else:
            wvl_condition = f'(wavelength < {stop_wvl})'

        if start_time is None and stop_time is None:
            time_condition = ''
        elif start_time and stop_time:
            time_condition = f'({start_time*1e6} < time) & (time < {stop_time*1e6})'
        elif start_time and not stop_time:
            time_condition = f'({start_time*1e6} < time)'
        else:
            time_condition = f'(time < {stop_time*1e6})'

        if wvl_condition == '':
            condition = time_condition
        elif time_condition == '':
            condition = wvl_condition
        else:
            condition = wvl_condition + '&' + time_condition

        return self.table.read_where(condition)
