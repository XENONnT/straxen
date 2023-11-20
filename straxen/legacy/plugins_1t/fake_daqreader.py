from straxen import DAQReader
from immutabledict import immutabledict

import strax

export, __all__ = strax.exporter()

@export
class Fake1TDAQReader(DAQReader):
    provides = (
        'raw_records',
        'raw_records_diagnostic',
        'raw_records_aqmon')

    data_kind = immutabledict(zip(provides, provides))
