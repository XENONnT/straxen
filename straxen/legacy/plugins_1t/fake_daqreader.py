from straxen.plugins.raw_records.daqreader import DAQReader
from immutabledict import immutabledict


class Fake1TDAQReader(DAQReader):
    provides = ("raw_records", "raw_records_diagnostic", "raw_records_aqmon")

    data_kind = immutabledict(zip(provides, provides))
