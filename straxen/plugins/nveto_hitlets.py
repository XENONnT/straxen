import numba
import numpy as np
from immutabledict import immutabledict

import strax
import straxen
export, __all__ = strax.exporter()

__all__ = ['nVETOHitlets']


@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_nv',
        default=(3, 15),
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'hit_min_amplitude_nv',
        default=20, track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number.'),
    strax.Option(
        'min_split_nv',
        default=2**14, track=True,
        help='Minimum height difference [ADC counts] between local minimum and maximum, '
             'that a pulse get split.'),
    strax.Option(
        'min_split_ratio_nv',
        default=0, track=True,
        help='Min ratio between local maximum and minimum to split pulse (zero to switch this off).'),
    strax.Option(
        'entropy_template_nv',
        default='flat', track=True,
        help='Template data is compared with in conditional entropy. Can be either "flat" or an template array.'),
    strax.Option(
        'entropy_square_data_nv',
        default=False, track=True,
        help='Parameter which decides if data is first squared before normalized and compared to the template.'),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number."),
    strax.Option(
        'to_pe_file_nv',
        default='/dali/lgrandi/wenz/strax_data/HdMdata_strax_v0_9_0/swt_gains.npy',  # noqa
        help='URL of the to_pe conversion factors. Expect gains in units ADC/sample.'),
)
class nVETOHitlets(strax.Plugin):
    """
    Plugin which computes the nveto hitlets and their parameters.
    """
    __version__ = '0.0.1'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'records_nv'

    provides = 'hitlets_nv'
    data_kind = 'hitlets_nv'

    dtype = strax.hitlet_dtype()

    def setup(self):
        #TODO: Unify with TPC and add adc thresholds
        self.to_pe = straxen.get_resource(self.config['to_pe_file_nv'], fmt='npy')

    def compute(self, records_nv):
        # Search again for hits in records:
        hits = strax.find_hits(records_nv, min_amplitude=self.config['hit_min_amplitude_nv'])

        # Merge concatenate overlapping  within a channel. This is important
        # in case hits were split by record boundaries. In case we
        # accidentally concatenate two PMT signals we split them later again.
        hits = strax.concat_overlapping_hits(hits,
                                             self.config['save_outside_hits_nv'],
                                             self.config['channel_map']['nveto'])
        hits = strax.sort_by_time(hits)

        # Now convert hits into temp_hitlets including the data field:
        temp_hitlets = np.zeros(len(hits), strax.hitlet_dtype(n_sample=hits['length'].max()))
        strax.refresh_hit_to_hitlets(hits, temp_hitlets)
        del hits

        # Get hitlet data and split hitlets:
        strax.get_hitlets_data(temp_hitlets, records_nv, to_pe=self.to_pe)

        temp_hitlets = strax.split_peaks(temp_hitlets,
                                         records_nv,
                                         self.to_pe,
                                         data_type='hitlets',
                                         algorithm='local_minimum',
                                         min_height=self.config['min_split_nv'],
                                         min_ratio=self.config['min_split_ratio_nv'],
                                         result_dtype=strax.hitlet_dtype(n_sample=hits['length'].max()),
                                         )

        # Compute other hitlet properties:
        strax.hitlet_properties(temp_hitlets)
        strax.conditional_entropy(temp_hitlets, template='flat', square_data=False)

        # Remove data field:
        names = [name for name in temp_hitlets.dtype.names if name is not 'data']
        hitlets = temp_hitlets[names]
        return hitlets
