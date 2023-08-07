import numpy as np
import straxen
import strax

export, __all__ = strax.exporter()


@export
class PeakletsMainAlt(strax.Plugin):
    """
    Concatenate peaklets that constitute the main and alternative S1s and S2s.
    Allow waveform watching by analysts, while significantly decrease storage pressure.
    """

    depends_on = ('peaklets', 'event_basics')
    provides = ('peaklets_main_alt')
    data_kind = ('peaklets_main_alt')
    parallel = 'process'
    compressor = 'zstd'
    save_when = strax.SaveWhen.ALWAYS

    __version__ = '0.0.0'

    n_tpc_pmts = straxen.URLConfig(
        type=int,
        help='Number of TPC PMTs')

    sum_waveform_top_array = straxen.URLConfig(
        default=True,
        type=bool,
        help='Digitize the sum waveform of the top array separately'
    )

    def setup(self):
        return

    def infer_dtype(self):
        return strax.peak_dtype(
                n_channels=self.n_tpc_pmts,
                digitize_top=self.sum_waveform_top_array,
            )

    def fake_mask(self, df, start_field='time', end_field='end_time', pad=10e3):
        """
        Create a fake mask to touch the peaklets
        :param df: events
        :param start_field: start time
        :param end_field: end time
        :param pad: pad time for safety
        :return:
        """
        # Make sure peak exist first, inexist peak has time = -1
        mask_peak_exist = df[start_field] > 0
        # Fake containers to touch contain the peaklets
        fake_mask = np.zeros(mask_peak_exist.sum(), dtype=[('time', np.float64), ('endtime', np.float64)])
        fake_mask['time'] = df[mask_peak_exist][start_field] - pad
        fake_mask['endtime'] = df[mask_peak_exist][end_field] + pad
        return fake_mask

    def compute(self, peaklets, events):
        # Only save peaklets in main/alt_s1 main/alt_s2
        peaklets_main_alt_id = []

        for tag in ['s1', 'alt_s1', 's2', 'alt_s2']:
            result = strax.touching_windows(peaklets,
                                            fake_mask(events, f'{tag}_time', f'{tag}_endtime'))  # touch the peaklets
            if len(result) > 0:
                result = np.concatenate([np.arange(result[i][0], result[i][1], dtype=int) for i in
                                         range(len(result))])  # find the index of the touched peaklets
                peaklets_main_alt_id.append(np.array(result))
            print(result)
        # peaklets_main_alt = peaklets[np.array(peaklets_main_alt_id)]
        peaklets_main_alt = peaklets[np.sort(np.concatenate(peaklets_main_alt_id))]  # sort by time
        peaklets_main_alt = peaklets_main_alt[np.argsort(peaklets_main_alt['time'])]  # sort again just in case

        return peaklets_main_alt
