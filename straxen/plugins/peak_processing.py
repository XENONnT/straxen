import json
from packaging.version import parse as parse_version
import tempfile

import numpy as np
import numba

import strax
from straxen.common import get_to_pe, pax_file, get_resource, first_sr1_run
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('peak_gap_threshold', default=300,
                 help="No hits for this many ns triggers a new peak"),
    strax.Option('peak_left_extension', default=30,
                 help="Include this many ns left of hits in peaks"),
    strax.Option('peak_right_extension', default=30,
                 help="Include this many ns right of hits in peaks"),
    strax.Option('peak_min_pmts', default=2,
                 help="Minimum contributing PMTs needed to define a peak"),
    strax.Option('single_channel_peaks', default=False,
                 help='Whether single-channel peaks should be reported'),
    strax.Option('peak_split_min_height', default=25,
                 help="Minimum height in PE above a local sum waveform"
                      "minimum, on either side, to trigger a split"),
    strax.Option('peak_split_min_ratio', default=4,
                 help="Minimum ratio between local sum waveform"
                      "minimum and maxima on either side, to trigger a split"),
    strax.Option('diagnose_sorting', track=False, default=False,
                 help="Enable runtime checks for sorting and disjointness"),
    strax.Option(
        'to_pe_file',
        default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',
        help='Link to the to_pe conversion factors'))
class Peaks(strax.Plugin):
    depends_on = ('records',)
    data_kind = 'peaks'
    parallel = 'process'
    rechunk_on_save = True

    __version__ = '0.1.0'

    def infer_dtype(self):
        self.to_pe = get_to_pe(self.run_id,self.config['to_pe_file'])
        return strax.peak_dtype(n_channels=len(self.to_pe))

    def compute(self, records):
        r = records

        hits = strax.find_hits(r)

        # Remove hits in zero-gain channels
        # they should not affect the clustering!
        hits = hits[self.to_pe[hits['channel']] != 0]

        hits = strax.sort_by_time(hits)

        peaks = strax.find_peaks(
            hits, self.to_pe,
            gap_threshold=self.config['peak_gap_threshold'],
            left_extension=self.config['peak_left_extension'],
            right_extension=self.config['peak_right_extension'],
            min_channels=self.config['peak_min_pmts'],
            result_dtype=self.dtype)
        strax.sum_waveform(peaks, r, self.to_pe)

        peaks = strax.split_peaks(
            peaks, r, self.to_pe,
            min_height=self.config['peak_split_min_height'],
            min_ratio=self.config['peak_split_min_ratio'])

        strax.compute_widths(peaks)

        if self.config['diagnose_sorting']:
            assert np.diff(r['time']).min() >= 0, "Records not sorted"
            assert np.diff(hits['time']).min() >= 0, "Hits not sorted"
            assert np.all(peaks['time'][1:]
                          >= strax.endtime(peaks)[:-1]), "Peaks not disjoint"

        return peaks


@export
@strax.takes_config(
    strax.Option('n_top_pmts', default=127,
                 help="Number of top PMTs")
)
class PeakBasics(strax.Plugin):
    __version__ = "0.0.1"
    parallel = True
    depends_on = ('peaks',)
    dtype = [
        (('Start time of the peak (ns since unix epoch)',
          'time'), np.int64),
        (('End time of the peak (ns since unix epoch)',
          'endtime'), np.int64),
        (('Peak integral in PE',
            'area'), np.float32),
        (('Number of PMTs contributing to the peak',
            'n_channels'), np.int16),
        (('PMT number which contributes the most PE',
            'max_pmt'), np.int16),
        (('Area of signal in the largest-contributing PMT (PE)',
            'max_pmt_area'), np.int32),
        (('Width (in ns) of the central 50% area of the peak',
            'range_50p_area'), np.float32),
        (('Fraction of area seen by the top array',
            'area_fraction_top'), np.float32),

        (('Length of the peak waveform in samples',
          'length'), np.int32),
        (('Time resolution of the peak waveform in ns',
          'dt'), np.int16),
        ]

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), self.dtype)
        for q in 'time length dt area'.split():
            r[q] = p[q]
        r['endtime'] = p['time'] + p['dt'] * p['length']
        r['n_channels'] = (p['area_per_channel'] > 0).sum(axis=1)
        r['range_50p_area'] = p['width'][:, 5]
        r['max_pmt'] = np.argmax(p['area_per_channel'], axis=1)
        r['max_pmt_area'] = np.max(p['area_per_channel'], axis=1)

        n_top = self.config['n_top_pmts']
        area_top = p['area_per_channel'][:, :n_top].sum(axis=1)
        # Negative-area peaks get 0 AFT - TODO why not NaN?
        m = p['area'] > 0
        r['area_fraction_top'][m] = area_top[m]/p['area'][m]
        return r


@export
@strax.takes_config(
    strax.Option(
        'nn_architecture',
        help='Path to JSON of neural net architecture',
        default_by_run=[
            (0, pax_file('XENON1T_tensorflow_nn_pos_20171217_sr0.json')),
            (first_sr1_run, pax_file('XENON1T_tensorflow_nn_pos_20171217_sr1.json'))]),   # noqa

    strax.Option(
        'nn_weights',
        help='Path to HDF5 of neural net weights',
        default_by_run=[
            (0, pax_file('XENON1T_tensorflow_nn_pos_weights_20171217_sr0.h5')),
            (first_sr1_run, pax_file('XENON1T_tensorflow_nn_pos_weights_20171217_sr1.h5'))]),   # noqa

    strax.Option('min_reconstruction_area',
                 help='Skip reconstruction if area (PE) is less than this',
                 default=10))
class PeakPositions(strax.Plugin):
    dtype = [('x', np.float32,
              'Reconstructed S2 X position (cm), uncorrected'),
             ('y', np.float32,
              'Reconstructed S2 Y position (cm), uncorrected')]
    depends_on = ('peaks',)

    # TODO
    # Parallelization doesn't seem to make it go faster
    # Is there much pure-python stuff in tensorflow?
    # Process-level paralellization might work, but you'd have to do setup
    # in each process, which probably negates the benefits,
    # except for huge chunks
    parallel = False

    n_top_pmts = 127

    __version__ = '0.0.1'

    def setup(self):
        import tensorflow as tf
        self.has_tf2 = parse_version(tf.__version__) > parse_version('2.0.a')
        if self.has_tf2:
            keras = tf.keras
        else:
            import keras

        nn_json = get_resource(self.config['nn_architecture'])
        nn = keras.models.model_from_json(nn_json)

        bad_pmts = json.loads(nn_json)['badPMTList']
        self.pmt_mask = ~np.in1d(np.arange(self.n_top_pmts),
                                 bad_pmts)

        with tempfile.NamedTemporaryFile() as f:
            f.write(get_resource(self.config['nn_weights'],
                                 fmt='binary'))
            nn.load_weights(f.name)
        self.nn = nn

        if not self.has_tf2:
            # Workaround for using keras/tensorflow in a threaded environment.
            # See: https://github.com/keras-team/keras/issues/
            # 5640#issuecomment-345613052
            self.nn._make_predict_function()
            self.graph = tf.get_default_graph()

    def compute(self, peaks):
        # Keep large peaks only
        peak_mask = peaks['area'] > self.config['min_reconstruction_area']
        x = peaks['area_per_channel'][peak_mask, :]

        if len(x) == 0:
            # Nothing to do, and .predict crashes on empty arrays
            return dict(x=np.zeros(0, dtype=np.float32),
                        y=np.zeros(0, dtype=np.float32))

        # Keep good top PMTS
        x = x[:, :self.n_top_pmts][:, self.pmt_mask]

        # Normalize
        with np.errstate(divide='ignore', invalid='ignore'):
            x /= x.sum(axis=1).reshape(-1, 1)

        result = np.ones((len(peaks), 2), dtype=np.float32) * float('nan')

        if self.has_tf2:
            y = self.nn.predict(x)
        else:
            with self.graph.as_default():
                y = self.nn.predict(x)

        result[peak_mask, :] = y

        # Convert from mm to cm... why why why
        result /= 10

        return dict(x=result[:, 0], y=result[:, 1])


@export
@strax.takes_config(
    strax.Option('s1_max_width', default=150,
                 help="Maximum (IQR) width of S1s"),
    strax.Option('s1_min_n_channels', default=3,
                 help="Minimum number of PMTs that must contribute to a S1"),
    strax.Option('s2_min_area', default=10,
                 help="Minimum area (PE) for S2s"),
    strax.Option('s2_min_width', default=200,
                 help="Minimum width for S2s"))
class PeakClassification(strax.Plugin):
    __version__ = '0.0.1'
    depends_on = ('peak_basics',)
    dtype = [
        ('type', np.int8, 'Classification of the peak.')]
    parallel = True

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), dtype=self.dtype)

        is_s1 = p['n_channels'] >= self.config['s1_min_n_channels']
        is_s1 &= p['range_50p_area'] < self.config['s1_max_width']
        r['type'][is_s1] = 1

        is_s2 = p['area'] > self.config['s2_min_area']
        is_s2 &= p['range_50p_area'] > self.config['s2_min_width']
        r['type'][is_s2] = 2

        return r

@export
@strax.takes_config(
    strax.Option('min_area_fraction', default=0.5,
                 help='The area of competing peaks must be at least '
                      'this fraction of that of the considered peak'),
    strax.Option('nearby_window', default=int(1e7),
                 help='Peaks starting within this time window (on either side)'
                      'in ns count as nearby.'),
)
class NCompeting(strax.OverlapWindowPlugin):
    depends_on = ('peak_basics',)
    dtype = [
        ('n_competing', np.int32,
            'Number of nearby larger or slightly smaller peaks')]

    def get_window_size(self):
        return 2 * self.config['nearby_window']

    def compute(self, peaks):
        return dict(n_competing=self.find_n_competing(
            peaks,
            window=self.config['nearby_window'],
            fraction=self.config['min_area_fraction']))

    @staticmethod
    @numba.jit(nopython=True, nogil=True, cache=True)
    def find_n_competing(peaks, window, fraction):
        n = len(peaks)
        t = peaks['time']
        a = peaks['area']
        results = np.zeros(n, dtype=np.int32)

        left_i = 0
        right_i = 0
        for i, peak in enumerate(peaks):
            while t[left_i] + window < t[i] and left_i < n - 1:
                left_i += 1
            while t[right_i] - window < t[i] and right_i < n - 1:
                right_i += 1
            results[i] = np.sum(a[left_i:right_i + 1] > a[i] * fraction)

        return results - 1
