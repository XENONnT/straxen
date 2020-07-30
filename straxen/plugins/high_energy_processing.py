import strax
import straxen
import numpy as np

export, __all__ = strax.exporter()

# These are also needed in peaklets, since hitfinding is repeated
HITFINDER_OPTIONS_he = tuple([
    strax.Option(
        'hit_min_amplitude_he',track=False,
        default=tuple(np.ones(800, dtype=np.int16) * 15),
        help='Minimum hit amplitude in ADC counts above baseline. '
             'See straxen.hit_min_amplitude for options.'
    )])


@export
@strax.takes_config(
    strax.Option(
        'n_tpc_pmts_he',track=False,default=800
        ),
    strax.Option(
        'parent_version',track=True,default=straxen.PulseProcessing.__version__
        ),
    *HITFINDER_OPTIONS_he)
class PulseProcessingHe(straxen.PulseProcessing):
    __version__ = '0.0.1'
    rechunk_on_save = False
    depends_on = 'raw_records_he'
    provides = 'records_he'
    data_kind = 'records_he'

    def infer_dtype(self):
        dtype = strax.record_dtype(strax.DEFAULT_RECORD_LENGTH)
        return dtype

    def setup(self):
        self.hev_enabled=False
        self.config['n_tpc_pmts'] = self.config['n_tpc_pmts_he']
        self.config['hit_min_amplitude'] = self.config['hit_min_amplitude_he']

    def compute(self, raw_records_he, start, end):
        result = super().compute(raw_records_he, start,end)
        return result['records']

    
@export
@strax.takes_config(
    strax.Option(
        'n_tpc_pmts_he',track=False,default=800
        ),
    strax.Option(
        'he_channels_offset',track=False,default=500
        ),
    strax.Option(
        'amplification',track=False,default=20
        ),
    strax.Option(
        'parent_version',track=True,default=straxen.Peaklets.__version__
        ),
    *HITFINDER_OPTIONS_he
)
class PeakletsHe(straxen.Peaklets):
    depends_on = 'records_he'
    provides = 'peaklets_he'
    data_kind = 'peaklets_he'
    __version__ = '0.0.1'
    
    def infer_dtype(self):
        return strax.peak_dtype(
                        n_channels=self.config['n_tpc_pmts_all'])

    def setup(self):
        self.config['hit_min_amplitude'] = self.config['hit_min_amplitude_he']  
        self.to_pe = straxen.get_to_pe(self.run_id,
                                       self.config['gain_model'],
                                       n_tpc_pmts=self.config['n_tpc_pmts'])
        buffer_pmts = np.zeros(self.config['he_channels_offset'])
        self.to_pe = np.concatenate((buffer_pmts,self.to_pe))
        self.to_pe *= self.config['amplification']

    def compute(self, records_he, start, end):
        result = super().compute(records_he, start,end)
        return result['peaklets']
        
        
@export
@strax.takes_config(
    strax.Option(
        'parent_version',track=True,default=straxen.PeakletClassification.__version__
        ),
)
class PeakletClassificationHe(straxen.PeakletClassification):
    """Classify peaklets as unknown, S1, or S2."""
    provides = 'peaklet_classification_he'
    depends_on = ('peaklets_he',)
    __version__ = '0.0.1'

    def compute(self, peaklets_he):
        return super().compute(peaklets_he)


@export
@strax.takes_config(
    strax.Option(
        'parent_version',track=True,default=straxen.MergedS2s.__version__
        ),
)
class MergedS2sHe(straxen.MergedS2s):
    """Merge together peaklets if we believe they form a single peak instead
    """
    depends_on = ('peaklets_he', 'peaklet_classification_he')
    data_kind = 'merged_s2s_he'
    provides = 'merged_s2s_he'
    __version__ = '0.0.1'
    
    def infer_dtype(self):
        return strax.unpack_dtype(self.deps['peaklets_he'].dtype_for('peaklets_he'))

    def compute(self, peaklets_he):
        return super().compute(peaklets_he)


@export
@strax.takes_config(
    strax.Option(
        'parent_version',track=True,default=straxen.Peaks.__version__
        ),
)
class PeaksHe(straxen.Peaks):
    depends_on = ('peaklets_he', 'peaklet_classification_he', 'merged_s2s_he')
    data_kind = 'peaks_he'
    provides = 'peaks_he'
    __version__ = '0.0.1'
    
    def infer_dtype(self):
        return self.deps['peaklets_he'].dtype_for('peaklets')

    def compute(self, peaklets_he, merged_s2s_he):
        return super().compute(peaklets_he, merged_s2s_he)


@export
@strax.takes_config(
    strax.Option(
        'parent_version',track=True,default=straxen.PeakBasics.__version__
        ),
)
class PeakBasicsHe(straxen.PeakBasics):
    __version__ = '0.0.1'
    depends_on = 'peaks_he'
    provides = 'peak_basics_he'

    def compute(self, peaks_he):
        return super().compute(peaks_he)
