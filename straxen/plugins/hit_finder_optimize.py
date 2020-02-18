import strax
import numpy as np
import numba

@strax.takes_config(
    strax.Option(
        'to_hit_counts',
        default='not defined yet',  # noqa
        help='path into which the data shall be stored'),
    strax.Option(
        'hit_height_threshold',
        default=tuple(np.arange(4, 40., 1)),
        help='Hitfinder height threshold in ADC counts above baseline'),
    strax.Option(
        'height_over_noise_threshold',
        default=tuple(np.arange(1.5, 20, 0.5)),
        help='height-over-noise threshold in ADC counts above baseline'),
    strax.Option(
        'window',
        default=(145, 170),
        help='Search window for hits'),
    strax.Option(
        'nbaseline',
        default=26,
        help='Number of samples to estimate baseline rms'),
    strax.Option(
        'start_ch',
        default=0,
        help='Index of the first channel'),
    strax.Option(
        'end_ch',
        default=254,
        help='Index of the last channel'),
    strax.Option(
        'channels_for_special_care',
        default=[],
        help='Channels which should not be included in this analysis')
)
class HitCounting(strax.Plugin):
    """
    """
    __version__ = '0.0.1'

    parallel = 'True'
    depends_on = 'raw_records'
    # compressor = 'lz4'

    provides = ('hit_statics', 'hit_dynamics')
    data_kind = {k: k for k in provides}

    def infer_dtype(self):
        """
        """
        nchannels = self.config['end_ch'] - self.config['start_ch']

        dtype = [(("Discrimination kind specific threshold", 'threshold'), np.int32),
                 (("Hits per channel", 'nhits'), np.float32, nchannels),
                 (("Number of events per channel", "N"), np.int32, nchannels),
                 (("Length of a single wf", "length"), np.int32),
                 (("Time resolution in ns", "dt"), np.int16),
                 (("Time of the chunk for time range?", "time"), np.int64)]

        return {k: dtype for k in self.provides}

    # def setup(self):
    #     """
    #     Setting-up storage array?
    #     """
    #     nchannels = self.config['end_ch'] - self.config['start_ch']
    #     dtype = [(("Discrimination kind specific threshold", 'threshold'), np.int32),
    #              (("Hits per channel", 'nhits'), np.float32, nchannels),
    #              (("Number of events per channel", "N"), np.int32, nchannels),
    #              (("Length of a single wf", "length"), np.int32),
    #              (("Time resolution in ns", "dt"), np.int16),
    #              (("Time of the chunk for time range?", "time"), np.int64)]
    #
    #
    #     # Init storage:
    #     nhits_hadc = np.zeros(len(self.config['hit_height_threshold']), dtype=dtype)
    #     nhits_hon = np.zeros(len(self.config['hit_height_threshold']), dtype=dtype)

    def compute(self, raw_records):
        """

        :param raw_records:
        :return:

        TODO: Remove copy and past stuff....
        """
        nchannels = self.config['end_ch'] - self.config['start_ch']
        dtype = [(("Discrimination kind specific threshold", 'threshold'), np.int32),
                 (("Hits per channel", 'nhits'), np.float32, nchannels),
                 (("Number of events per channel", "N"), np.int32, nchannels),
                 (("Length of a single wf", "length"), np.int32),
                 (("Time resolution in ns", "dt"), np.int16),
                 (("Time of the chunk for time range?", "time"), np.int64)]

        # Init storage:
        nhits_hadc = np.zeros(len(self.config['hit_height_threshold']), dtype=dtype)
        nhits_hon = np.zeros(len(self.config['height_over_noise_threshold']), dtype=dtype)

        # Masking out not needed channels:
        channels = np.arange(self.config['start_ch'], self.config['end_ch'], 1, dtype=int)
        # Take out the special cases:
        if self.config['channels_for_special_care']:
            mask = np.invert(np.isin(channels, self.config['channels_for_special_care']))
            channels = channels[mask]
        mask = np.isin(raw_records['channel'], channels)
        raw_records = raw_records[mask]

        # Get basic information:
        nhits_hadc['length'] = raw_records['length'][0]
        nhits_hon['length'] = raw_records['length'][0]

        nhits_hadc['dt'] = raw_records['dt'][0]
        nhits_hon['dt'] = raw_records['dt'][0]

        nhits_hadc['time'] = raw_records['time'][0]   # TODO: Ask about time parameter, needed for seconds_range?
        nhits_hon['time'] = raw_records['time'][0]    #  wouldn't work here I guess..

        res = _count_rr_in_channel(raw_records, len(channels), self.config['start_ch'])
        nhits_hadc['N'] += res
        nhits_hon['N'] += res

        # Compute with static height thresholds:
        for ind, hadc in enumerate(self.config['hit_height_threshold']):
            h = strax.find_hits(raw_records, hadc, hadc, nbaseline=self.config['nbaseline'])
            res = _count_hit_in_channel(h,
                                        len(channels),
                                        self.config['start_ch'],
                                        window=np.array(self.config['window'], np.int16))
            nhits_hadc['nhits'][ind] += res
            nhits_hadc['threshold'][ind] = hadc

        # Compute with dynamic height-over-noise thresholds:
        for ind, hon in enumerate(self.config['height_over_noise_threshold']):
            h = strax.find_hits(raw_records, hon, hon, nbaseline=self.config['nbaseline'], static=True)
            res = _count_hit_in_channel(h,
                                        len(channels),
                                        self.config['start_ch'],
                                        window=np.array(self.config['window'], np.int16))
            nhits_hon['nhits'][ind] += res
            nhits_hon['threshold'][ind] = hon

        return dict(hit_statics=nhits_hadc, hit_dynamics=nhits_hon)


@numba.njit
def _count_hit_in_channel(h, channels, start_id=0, window=np.array([145, 175], np.int16)):
    """

    :param h:
    :param channels:
    :param start_id:
    :param window:
    :return:
    """
    res = np.zeros(channels)
    for ch in range(channels):
        mask = (h['left'] >= window[0]) & (h['left'] < window[1])
        res[ch] = np.sum(h[mask]['channel'] == (ch + start_id))
    return res


@numba.njit
def _count_rr_in_channel(rr, channels, start_id=0):
    """

    :param rr:
    :param channels:
    :param start_id:
    :return:
    """
    res = np.zeros(channels, np.int32)
    for ch in range(channels):
        res[ch] = np.sum(rr['channel'] == (ch + start_id))
    return res