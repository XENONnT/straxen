import strax


@strax.takes_config(
    strax.Option(
        'to_hit_counts',
        default='not defined yet',  # noqa
        help='path into which the data shall be stored'),
    strax.Option(
        'hit_height_threshold',
        default=np.arange(0, 40., 1),
        help='Hitfinder height threshold in ADC counts above baseline'),
    strax.Option(
        'height_over_noise_threshold',
        default=np.arange(0, 20, 0.5),
        help='height-over-noise threshold in ADC counts above baseline'),
    strax.Option(
        'window',
        default=np.array([145, 170], np.int16),
        help='Search window for hits'),
    strax.Option(
        'nbaseline',
        default=26,
        help='Number of samples to estimate baseline rms'),
    strax.Option(
        'nChannels',
        default=254,
        help='Number of PMT channels'),
    strax.Option(
        'start_id',
        default=0,
        help='Channel number of the very first PMT')
)
class HitCounting(strax.Plugin):
    """
    """
    __version__ = '0.0.1'

    # parallel = 'False'
    depends_on = 'raw_records'
    # compressor = 'lz4'

    provides = ('hit_statics', 'hit_dynamics')
    data_kind = {k: k for k in provides}

    def infer_dtype(self):
        """
        """
        dtype = [(("Discrimination kind specific threshold", 'threshold'), np.int32),
                 (("Hits per channel", 'nhits'), np.float32, self.config['nChannels']),
                 (("Number of events per channel", "N"), np.int32, self.config['nChannels']),
                 (("Length of a single wf", "length"), np.int32),
                 (("Time resolution in ns", "dt"), np.int16),
                 (("Time of the chunk for time range?", "time"), np.int64)]

        return {k: dtype for k in self.provides}

    def setup(self):
        """
        Setting-up storage array?
        """
        # Init storage:
        self.nhits_hadc = np.zeros(len(self.config['hit_height_threshold']), dtype=dtype)
        self.nhits_hon = np.zeros(len(self.config['hit_height_threshold']), dtype=dtype)

    def compute(self, raw_records):
        '''

        TODO: Remove copy and past stuff....
        '''
        # Get basic information:
        self.nhits_hadc['length'] = raw_records['length'][0]
        self.nhits_hon['length'] = raw_records['length'][0]

        self.nhits_hadc['dt'] = raw_records['dt'][0]
        self.nhits_hon['dt'] = raw_records['dt'][0]

        self.nhits_hadc['time'] = raw_records['time'][0]
        self.nhits_hon['time'] = raw_records['time'][0]

        res = _count_rr_in_channel(raw_records, self.config['nChannels'], self.config['start_id'])
        self.nhits_hadc['N'] += res
        self.nhits_hon['N'] += res

        # Compute with static height thresholds:
        for ind, hadc in enumerate(self.config['hit_height_threshold']):
            h = strax.find_hits(raw_records, hadc, hadc, nbaseline=self.config['nbaseline'])
            res = _count_hit_in_channel(h,
                                        self.config['nChannels'],
                                        self.config['start_id'],
                                        window=self.config['window'])
            self.nhits_hadc['nhits'][ind] += res
            self.nhits_hadc['threshold'][ind] = hadc

        # Compute with dynmaic height-over-noise thresholds:
        for ind, hon in enumerate(self.config['height_over_noise_threshold']):
            h = strax.find_hits(raw_records, hon, hon, nbaseline=self.config['nbaseline'], static=True)
            res = _count_hit_in_channel(h,
                                        self.config['nChannels'],
                                        self.config['start_id'],
                                        window=self.config['window'])
            self.nhits_hon['nhits'][ind] += res
            self.nhits_hon['threshold'][ind] = hon

        return dict(hit_statics=self.nhits_hadc, hit_dynamics=self.nhits_hon)


@numba.njit
def _count_hit_in_channel(h, channels, start_id=0, winodw=np.array([145, 175], np.int16)):
    res = np.zeros(channels)
    for ch in range(channels):
        mask = (h['left'] >= winodw[0]) & (h['left'] < winodw[1])
        res[ch] = np.sum(h[mask]['channel'] == (ch + start_id))
    return res


@numba.njit
def _count_rr_in_channel(rr, channels, start_id=0):
    res = np.zeros(channels, np.int32)
    for ch in range(channels):
        res[ch] = np.sum(rr['channel'] == (ch + start_id))
    return res