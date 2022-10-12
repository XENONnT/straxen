import numba
import numpy as np
import strax
from immutabledict import immutabledict
from strax.processing.general import _touching_windows
import straxen

from straxen.plugins.defaults import HE_PREAMBLE
from straxen.plugins.peaklets.peaklets import Peaklets
from straxen.plugins.defaults import HE_PREAMBLE
import strax
import straxen

import numpy as np
import numba
import pandas as pd

import typing as ty
from immutabledict import immutabledict

export, __all__ = strax.exporter()

@export
class PeakletsHighEnergy(Peaklets):
    __doc__ = HE_PREAMBLE + Peaklets.__doc__
    depends_on = 'records_he'
    provides = 'peaklets_he'
    data_kind = 'peaklets_he'
    __version__ = '0.0.2'
    child_plugin = True
    save_when = strax.SaveWhen.TARGET

    n_he_pmts = straxen.URLConfig(track=False, default=752, infer_type=False,
                                  help="Maximum channel of the he channels")

    he_channel_offset = straxen.URLConfig(track=False, default=500, infer_type=False,
                                          help="Minimum channel number of the he channels")

    le_to_he_amplification = straxen.URLConfig(default=20, track=True, infer_type=False,
                                               help="Difference in amplification between low energy and high "
                                                    "energy channels")

    peak_min_pmts_he = straxen.URLConfig(default=2, infer_type=False,
                                         child_option=True, parent_option_name='peak_min_pmts',
                                         track=True,
                                         help="Minimum number of contributing PMTs needed to define a peak")

    saturation_correction_on_he = straxen.URLConfig(default=False, infer_type=False,
                                                    child_option=True, parent_option_name='saturation_correction_on',
                                                    track=True,
                                                    help='On off switch for saturation correction for High Energy'
                                                         ' channels')

    hit_min_amplitude_he = straxen.URLConfig(
        default='cmt://hit_thresholds_he?version=ONLINE&run_id=plugin.run_id', track=True, infer_type=False,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_tpc_pmts, or a number,'
             'or a string like "pmt_commissioning_initial" which means calling'
             'hitfinder_thresholds.py'
             'or a tuple like (correction=str, version=str, nT=boolean),'
             'which means we are using cmt.'
    )

    def infer_dtype(self):
        return strax.peak_dtype(n_channels=self.n_he_pmts)

    def setup(self):
        self.to_pe = self.gain_model
        buffer_pmts = np.zeros(self.he_channel_offset)
        self.to_pe = np.concatenate((buffer_pmts, self.to_pe))
        self.to_pe *= self.le_to_he_amplification

        self.hit_thresholds = self.hit_min_amplitude_he

        self.channel_range = self.channel_map['he']

    def compute(self, records_he, start, end):
        result = super().compute(records_he, start, end)
        return result['peaklets']

