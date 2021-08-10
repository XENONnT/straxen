import numpy as np
from copy import copy
from immutabledict import immutabledict

import strax, straxen

export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number."),
)
class nVetoExtTimings(strax.Plugin):
    """
    Plugin which computes the time differnce from external trigger timing to hitlets_nv.
    The nearlest `time` of `raw_records_nv` before `hitlets_nv` time are used as the
    external trigger timings.

    Note:
        Channel dependence for trigger timing has been ignored. It should be the same,
        But unfortunetely not.
    """
    depends_on = ('raw_records_nv', 'hitlets_nv')
    provides = 'ext_timings_nv'
    data_kind = 'hitlets_nv'

    compressor = 'zstd'
    ends_with = '_nv'
    __version__ = '0.0.1'

    def infer_dtype(self):
        dtype = copy(strax.time_dt_fields)
        dtype += [(('Delta time from trigger timing [ns]', 'delta_time'), np.int16),]
        return dtype

    def setup(self):
        self.channel_range = self.config['channel_map']['nveto']

    def compute(self, hitlets_nv, raw_records_nv):

        time_range_dtype = copy(strax.time_fields)
        time_range_dtype += [(('PMT channel', 'channel'), np.int16)]
        rr_time_ranges = np.zeros(len(raw_records_nv), dtype=time_range_dtype)
        rr_time_ranges['time'] = raw_records_nv['time']
        rr_time_ranges['endtime'] = raw_records_nv['time'] + \
                                    raw_records_nv['pulse_length'] * raw_records_nv['dt']
        rr_time_ranges['channel'] = raw_records_nv['channel']

        ext_timings_nv = np.zeros_like(hitlets_nv, dtype=self.infer_dtype())
        ext_timings_nv['time'] = hitlets_nv['time']
        ext_timings_nv['length'] = hitlets_nv['length']
        ext_timings_nv['dt'] = hitlets_nv['dt']

        # numpy access with fancy index returns copy, not view
        # This for-loop is required to substitute in one by one
        for ch in np.arange(self.channel_range[0], self.channel_range[1] + 1):
            fancy_i_ch = hitlets_nv['channel']==ch
            fancy_i_ch = np.arange(len(fancy_i_ch))[fancy_i_ch]
            _rr_time_ranges = rr_time_ranges[rr_time_ranges['channel']==ch]
            _hitlets_nv = hitlets_nv[fancy_i_ch]
            _rr_index = strax.fully_contained_in(_hitlets_nv, _rr_time_ranges)
            _t_delta = _hitlets_nv['time'] - _rr_time_ranges[_rr_index]['time']

            for res_i, t_del in zip(fancy_i_ch, _t_delta):
                ext_timings_nv['delta_time'][res_i] = t_del

        return ext_timings_nv
