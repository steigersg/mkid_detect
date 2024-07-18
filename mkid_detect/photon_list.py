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
