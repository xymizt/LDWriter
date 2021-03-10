""" Parser for MoTec ld files

Code created through reverse engineering the data format.

Credits to gotzl
"""

import datetime
import struct

import numpy as np


class ldData(object):
    """Container for parsed data of an ld file.

    Allows reading and writing.
    """

    def __init__(self, head, channs):
        self.head = head
        self.channs = channs

    def __getitem__(self, item):
        if not isinstance(item, int):
            col = [n for n, x in enumerate(self.channs) if x.name == item]
            if len(col) != 1:
                raise Exception("Could get column", item, col)
            item = col[0]
        return self.channs[item]

    def __iter__(self):
        return iter([x.name for x in self.channs])

    @classmethod
    def frompd(cls, df, units, parameters):
        # type: (pd.DataFrame) -> ldData
        """Create and ldData object from a pandas DataFrame.

        Example:
        import pandas as pd
        import numpy as np
        from ldparser import ldData

        # create test dataframe
        df = pd.DataFrame(np.random.randn(6,4),columns=list('ABCD'))
        print(df)
        # create an lddata object from the dataframe
        l = ldData.frompd(df)
        # write an .ld file
        l.write('/tmp/test.ld')

        # just to check, read back the file
        l = ldData.fromfile('/tmp/test.ld')
        # create pandas dataframe
        df = pd.DataFrame(data={c: l[c].data for c in l})
        print(df)

        """

        # for now, fix datatype and frequency
        freq, dtype = 20 if parameters['-f'] == "" else int(parameters['-f']), np.float32

        # pointer to meta data of first channel
        meta_ptr = struct.calcsize(ldHead.fmt) + 1000  # Give ample space for event data

        # list of columns to read - only accept numeric data
        cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

        # pointer to data of first channel
        chanheadsize = struct.calcsize(ldChan.fmt)

        # set up the event data
        event_ptr = 2000  # arbitrary byte position between end of head and start of meta_ptr
        event = ldEvent(parameters['-e'], parameters['-s'], parameters['-c'], 0, None)

        data_ptr = meta_ptr + len(cols) * chanheadsize

        # create a header
        head = ldHead(meta_ptr, data_ptr, event_ptr,  event,
                       parameters['-d'],  parameters['-i'], parameters['-v'],
                       datetime.datetime.now(),
                       parameters['-c'], parameters['-e'], parameters['-s'])

        # create the channels, meta data and associated data
        channs, prev, next = [], 0, meta_ptr + chanheadsize
        for n, col in enumerate(cols):
            # create mocked channel header

            chan = ldChan(None,
                          meta_ptr, prev, next if n < len(cols)-1 else 0,
                          data_ptr, len(df[col]),
                          dtype, freq, 0, 1, 1, 0,  # shift, mul, scale, dec
                          col, col, units[n])

            # link data to the channel
            chan._data = df[col].to_numpy(dtype)

            # calculate pointers to the previous/next channel meta data
            prev = meta_ptr
            meta_ptr = next
            next += chanheadsize

            # increment data pointer for next channel
            data_ptr += chan._data.nbytes

            channs.append(chan)

        return cls(head, channs)

    @classmethod
    def fromfile(cls, f):
        # type: (str) -> ldData
        """Parse data of an ld file
        """
        return cls(*read_ldfile(f))

    def write(self, f):
        # type: (str) -> ()
        """Write an ld file containing the current header information and channel data
        """

        # convert the data using scale/shift etc before writing the data
        conv_data = lambda c: ((c.data / c.mul) - c.shift) * c.scale / pow(10., -c.dec)

        with open(f, 'wb') as f_:
            self.head.write(f_, len(self.channs))
            f_.seek(self.channs[0].meta_ptr)
            list(map(lambda c: c[1].write(f_, c[0]), enumerate(self.channs)))  # save channel head
            list(map(lambda c: f_.write(conv_data(c)), self.channs))  # Save data


class ldEvent(object):
    fmt = '<64s64s1024sH'

    def __init__(self, name, session, comment, venue_ptr, venue):
        self.name, self.session, self.comment, self.venue_ptr, self.venue = \
            name, session, comment, venue_ptr, venue

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldEvent
        """Parses and stores the event information in an ld file
        """
        name, session, comment, venue_ptr = struct.unpack(
            ldEvent.fmt, f.read(struct.calcsize(ldEvent.fmt)))
        name, session, comment = map(decode_string, [name, session, comment])

        print((name, session, comment, venue_ptr))

        venue = None
        if venue_ptr > 0:
            f.seek(venue_ptr)
            venue = ldVenue.fromfile(f)

        return cls(name, session, comment, venue_ptr, venue)

    def write(self, f):
        f.write(struct.pack(ldEvent.fmt,
                            self.name.encode(),
                            self.session.encode(),
                            self.comment.encode(),
                            self.venue_ptr))

        if self.venue_ptr > 0:
            f.seek(self.venue_ptr)
            self.venue.write(f)

    def __str__(self):
        return "%s; venue: %s"%(self.name, self.venue)


class ldVenue(object):
    fmt = '<64s1034xH'

    def __init__(self, name, vehicle_ptr, vehicle):
        self.name, self.vehicle_ptr, self.vehicle = name, vehicle_ptr, vehicle

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldVenue
        """Parses and stores the venue information in an ld file
        """
        name, vehicle_ptr = struct.unpack(ldVenue.fmt, f.read(struct.calcsize(ldVenue.fmt)))

        vehicle = None
        if vehicle_ptr > 0:
            f.seek(vehicle_ptr)
            vehicle = ldVehicle.fromfile(f)
        return cls(decode_string(name), vehicle_ptr, vehicle)

    def write(self, f):
        f.write(struct.pack(ldVenue.fmt, self.name.encode(), self.vehicle_ptr))

        if self.vehicle_ptr > 0:
            f.seek(self.vehicle_ptr)
            self.vehicle.write(f)

    def __str__(self):
        return "%s; vehicle: %s"%(self.name, self.vehicle)


class ldVehicle(object):
    fmt = '<64s128xI32s32s'

    def __init__(self, id, weight, type, comment):
        self.id, self.weight, self.type, self.comment = id, weight, type, comment

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldVehicle
        """Parses and stores the vehicle information in an ld file
        """
        id, weight, type, comment = struct.unpack(ldVehicle.fmt, f.read(struct.calcsize(ldVehicle.fmt)))
        id, type, comment = map(decode_string, [id, type, comment])
        return cls(id, weight, type, comment)

    def write(self, f):
        f.write(struct.pack(ldVehicle.fmt, self.id.encode(), self.weight, self.type.encode(), self.comment.encode()))

    def __str__(self):
        return "%s (type: %s, weight: %i, %s)"%(self.id, self.type, self.weight, self.comment)


class ldHead(object):
    fmt = '<' + (
        "I4x"     # ldmarker
        "II"      # chann_meta_ptr chann_data_ptr
        "20x"     # ??
        "I"       # event_ptr
        "24x"     # ??
        "HHH"     # unknown static (?) numbers
        "I"       # device serial
        "8s"      # device type
        "H"       # device version
        "H"       # unknown static (?) number
        "I"       # num_channs
        "4x"      # ??
        "16s"     # date
        "16x"     # ??
        "16s"     # time
        "16x"     # ??
        "64s"     # driver
        "64s"     # vehicleid
        "64x"     # ??
        "64s"     # venue
        "64x"     # ??
        "1024x"   # ??
        "I"       # enable "pro logging" (some magic number?)
        "66x"     # ??
        "64s"     # short comment
        "126x"    # ??
        "64s"     # event
        "64s"     # session
    )

    def __init__(self, meta_ptr, data_ptr, aux_ptr, aux, driver, vehicleid, venue, datetime, short_comment, event, session):
        self.meta_ptr, self.data_ptr, self.aux_ptr, self.aux, self.driver, self.vehicleid, \
        self.venue, self.datetime, self.short_comment, self.event, self.session = meta_ptr, data_ptr, aux_ptr, aux, \
                                                driver, vehicleid, venue, datetime, short_comment, event, session

    @classmethod
    def fromfile(cls, f):
        # type: (file) -> ldHead
        """Parses and stores the header information of an ld file
        """
        (_, meta_ptr, data_ptr, aux_ptr,
            _, _, _,
            _, _, _, _, n,
            date, time,
            driver, vehicleid, venue,
            _, short_comment, event, session) = struct.unpack(ldHead.fmt, f.read(struct.calcsize(ldHead.fmt)))

        date, time, driver, vehicleid, venue, short_comment, event, session = \
            map(decode_string, [date, time, driver, vehicleid, venue, short_comment, event, session])

        try:
            # first, try to decode datatime with seconds
            _datetime = datetime.datetime.strptime(
                    '%s %s'%(date, time), '%d/%m/%Y %H:%M:%S')
        except ValueError:
            _datetime = datetime.datetime.strptime(
                '%s %s'%(date, time), '%d/%m/%Y %H:%M')

        aux = None
        if aux_ptr > 0:
            f.seek(aux_ptr)
            aux = ldEvent.fromfile(f)
        return cls(meta_ptr, data_ptr, aux_ptr, aux, driver, vehicleid, venue, _datetime, short_comment, event, session)

    def write(self, f, n):
        f.write(struct.pack(ldHead.fmt,
                            0x40,
                            self.meta_ptr, self.data_ptr, self.aux_ptr,
                            0, 0x4240, 0xf,
                            88, "ACL".encode(), 170, 128, n,
                            self.datetime.date().strftime("%d/%m/%Y").encode(),
                            self.datetime.time().strftime("%H:%M:%S").encode(),
                            self.driver.encode(), self.vehicleid.encode(), self.venue.encode(),
                            11336, self.short_comment.encode(), self.event.encode(), self.session.encode(),
                            ))

        if self.aux_ptr > 0:
            f.seek(self.aux_ptr)
            self.aux.write(f)

    def __str__(self):
        return 'driver:    %s\n' \
               'vehicleid: %s\n' \
               'venue:     %s\n' \
               'event:     %s\n' \
               'session:   %s\n' \
               'short_comment: %s\n' \
               'event_long:    %s'%(
            self.driver, self.vehicleid, self.venue, self.event, self.session, self.short_comment, self.aux)


class ldChan(object):
    """Channel (meta) data

    Parses and stores the channel meta data of a channel in a ld file.
    Needs the pointer to the channel meta block in the ld file.
    The actual data is read on demand using the 'data' property.
    """

    fmt = '<' + (
        "IIII"    # prev_addr next_addr data_ptr n_data
        "H"       # some counter?
        "HHH"     # datatype datatype rec_freq
        "HHHh"    # shift mul scale dec_places
        "32s"     # name
        "8s"      # short name
        "12s"     # unit
        "40x"     # ? (40 bytes for ACC, 32 bytes for acti)
    )

    def __init__(self, _f, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len,
                 dtype, freq, shift, mul, scale, dec,
                 name, short_name, unit):

        self._f = _f
        self.meta_ptr = meta_ptr
        self._data = None

        (self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len,
        self.dtype, self.freq,
        self.shift, self.mul, self.scale, self.dec,
        self.name, self.short_name, self.unit) = prev_meta_ptr, next_meta_ptr, data_ptr, data_len,\
                                                 dtype, freq,\
                                                 shift, mul, scale, dec,\
                                                 name, short_name, unit

    @classmethod
    def fromfile(cls, _f, meta_ptr):
        # type: (str, int) -> ldChan
        """Parses and stores the header information of an ld channel in a ld file
        """
        with open(_f, 'rb') as f:
            f.seek(meta_ptr)

            (prev_meta_ptr, next_meta_ptr, data_ptr, data_len, _,
             dtype_a, dtype, freq, shift, mul, scale, dec,
             name, short_name, unit) = struct.unpack(ldChan.fmt, f.read(struct.calcsize(ldChan.fmt)))

        name, short_name, unit = map(decode_string, [name, short_name, unit])

        if dtype_a in [0x07]:
            dtype = [None, np.float16, None, np.float32][dtype-1]
        elif dtype_a in [0, 0x03, 0x05]:
            dtype = [None, np.int16, None, np.int32][dtype-1]
        else: raise Exception('Datatype %i not recognized'%dtype_a)

        return cls(_f, meta_ptr, prev_meta_ptr, next_meta_ptr, data_ptr, data_len,
                   dtype, freq, shift, mul, scale, dec, name, short_name, unit)

    def write(self, f, n):
        if self.dtype == np.float16 or self.dtype == np.float32:
            dtype_a = 0x07
            dtype = {np.float16: 2, np.float32: 4}[self.dtype]
        else:
            dtype_a = 0x03
            dtype = {np.int16: 2, np.int32: 4}[self.dtype]

        f.write(struct.pack(ldChan.fmt,
                            self.prev_meta_ptr, self.next_meta_ptr, self.data_ptr, self.data_len,
                            0x2ee1+n, dtype_a, dtype, self.freq, self.shift, self.mul, self.scale, self.dec,
                            self.name.encode(), self.short_name.encode(), self.unit.encode(encoding='ISO-8859-1')))

    @property
    def data(self):
        # type: () -> np.array
        """ Read the data words of the channel
        """
        if self._data is None:
            # jump to data and read
            with open(self._f, 'rb') as f:
                f.seek(self.data_ptr)
                try:
                    self._data = np.fromfile(f,
                                            count=self.data_len, dtype=self.dtype)

                    self._data = (self._data/self.scale * pow(10., -self.dec) + self.shift) * self.mul

                    if len(self._data) != self.data_len:
                        raise ValueError("Not all data read!")

                except ValueError as v:
                    print(v, self.name, self.freq,
                          hex(self.data_ptr), hex(self.data_len),
                          hex(len(self._data)), hex(f.tell()))
                    # raise v
        return self._data

    def __str__(self):
        return 'chan %s (%s) [%s], %i Hz'%(
            self.name,
            self.short_name, self.unit,
            self.freq)


def decode_string(bytes):
    # type: (bytes) -> str
    """decode the bytes and remove trailing zeros
    """
    try:
        return bytes.decode('ascii').strip().rstrip('\0').strip()
    except Exception as e:
        print("Could not decode string: %s - %s"%(e, bytes))
        return ""
        # raise e


def read_channels(f_, meta_ptr):
    # type: (str, int) -> list
    """ Read channel data inside ld file

    Cycles through the channels inside an ld file,
     starting with the one where meta_ptr points to.
     Returns a list of ldchan objects.
    """
    chans = []
    while meta_ptr:
        chan_ = ldChan.fromfile(f_, meta_ptr)
        chans.append(chan_)
        meta_ptr = chan_.next_meta_ptr
    return chans


def read_ldfile(f_):
    # type: (str) -> (ldHead, list)
    """ Read an ld file, return header and list of channels
    """
    head_ = ldHead.fromfile(open(f_,'rb'))
    chans = read_channels(f_, head_.meta_ptr)
    return head_, chans


def read_units(file):
    # Removes the first row and return it as a list of units
    # Saves a new csv without units row
    with open(file) as f:
        lines = f.readlines()
        unitsRow = lines.pop(1)
        units = list(unitsRow.split(','))
    
    newFile, _ = os.path.splitext(os.path.basename(file))
    newFile = os.path.join("tmp", newFile + "NO_UNITS.csv")

    with open(newFile, 'w') as f:
        f.writelines(lines)

    return units, newFile


def convert_ld(csvFilePath, targetPath, parameters):
    print(os.path.basename(csvFilePath))
    fileName, _ = os.path.splitext(os.path.basename(csvFilePath))

    # separate units from csv
    units, file = read_units(csvFilePath)

    # create the dataframe
    df = pd.read_csv(file)

    # load the data based on the dataframe
    l = ldData.frompd(df, units, parameters)
    # write an .ld file
    file, _ = os.path.splitext(os.path.basename(file))
    l.write(os.path.join(targetPath, file + ".ld"))


def next_value(ls, value):
    try:
        return ls.pop(ls.index(value)+1)
    except (IndexError, ValueError):
        return ""


if __name__ == '__main__':

    import sys, os, glob
    import pandas as pd
    import numpy as np

    # Handle single file
    if '-p' in sys.argv:
        args = sys.argv[1:]
    # Handle folder
    else:
        args = sys.argv[2:]

    commands = [
        '-p',  # single file path
        '-f',  # frequency
        '-c',  # comment
        '-e',  # event
        '-s',  # session
        '-d',  # driver
        '-i',  # vehicle id
        '-v',   # test venue
        '--folder',
        '--help'
    ]

    if '--help' in sys.argv:
        print(
        """
            -p C:\\single\\file\\nospaces\\path.csv
            -f frequencyInteger
            -c comment
            -e event
            -s session
            -d driver
            -i vehicleID
            -v venue
            --folder C:\\single\\folder\\nospaces # target folder where to save LD files
            --help  # help
        """
        )
        exit()

    parameters = dict(map(lambda p: (p, next_value(args, p)), commands))
    print(parameters)

    targetFolder = parameters['--folder'] if '--folder' in args else 'ldfiles'

    if '-p' in args:
        file = parameters['-p']
        convert_ld(file, targetFolder, parameters)

    else:
        for f in glob.glob('%s/*.csv'%sys.argv[1]):
            convert_ld(f, targetFolder, parameters)

    # Clear temp folder
    import shutil
    for file_name in os.listdir('tmp'):
        file_path = os.path.join('tmp', file_name)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason %s" % (file_path, e))

    # just to check, read back the file
    # l = ldData.fromfile('ldfiles\\CADC150_CS_plus23_ID034_LID496_201001_merge_GlobAttNO_UNITS.ld')
    # # create pandas dataframe
    # df = pd.DataFrame(data={c: l[c].data for c in l})
    # print(df)

    # Plot channels
    # from itertools import groupby
    # import matplotlib.pyplot as plt
    #     l = ldData.fromfile(f)
    #     # create plots for all channels with the same frequency
    #     for f, g in groupby(l.channs, lambda x:x.freq):
    #         df = pd.DataFrame({i.name.lower(): i.data for i in g})
    #         df.plot()
    #         plt.show()


