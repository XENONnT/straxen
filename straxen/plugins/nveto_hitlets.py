import numba
import numpy as np
from immutabledict import immutabledict

import strax
import straxen
export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_nv',
        default=(3, 15), track=True,
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'hit_min_amplitude_nv',
        default=20, track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number.'),
    strax.Option(
        'min_split_nv',
        default=100, track=True,
        help='Minimum height difference pe/sample between local minimum and maximum, '
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
        default=straxen.aux_repo + '/c5800ea686f06f0149af30b2db9c08b6216ecb36/n_veto_gains.npy?raw=true',  # noqa
        help='URL of the to_pe conversion factors. Expect gains in units ADC/sample.'),
)
class nVETOHitlets(strax.Plugin):
    """
    Plugin which computes the nveto hitlets and their parameters.
    
    Hitlets are an extension of regular hits. They include the left
    and right extension. The plugin does the following:
        1. Generate hitlets which includes these sub-steps:
            * Apply left and right hit extension and concatenate
            overlapping hits.
            * Generate temp. hitelts and look for their waveforms in
            their corresponding records.
            * Split hitlets if they satisfy the set criteria.
        2. Compute the properties of the hitlets.

    Note:
        Hitlets are getting chopped if extended in not recorded regions.
    """
    __version__ = '0.0.2'

    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    depends_on = 'records_nv'

    provides = 'hitlets_nv'
    data_kind = 'hitlets_nv'

    dtype = strax.hitlet_dtype()

    def setup(self):
        # TODO: Unify with TPC and add adc thresholds
        self.to_pe = straxen.get_resource(self.config['to_pe_file_nv'], fmt='npy')

    def compute(self, records_nv, start, end):
        # Search again for hits in records:
        hits = strax.find_hits(records_nv, min_amplitude=self.config['hit_min_amplitude_nv'])

        # Merge concatenate overlapping  within a channel. This is important
        # in case hits were split by record boundaries. In case we
        # accidentally concatenate two PMT signals we split them later again.
        hits = strax.concat_overlapping_hits(hits,
                                             self.config['save_outside_hits_nv'],
                                             self.config['channel_map']['nveto'],
                                             start,
                                             end)
        hits = strax.sort_by_time(hits)

        # Now convert hits into temp_hitlets including the data field:
        if len(hits):
            nsamples = hits['length'].max()
        else:
            nsamples = 0
        temp_hitlets = np.zeros(len(hits), strax.hitlet_with_data_dtype(n_samples=nsamples))

        # Generating hitlets and copying relevant information from hits to hitlets.
        # These hitlets are not stored in the end since this array also contains a data
        # field which we will drop later.
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
                                         min_ratio=self.config['min_split_ratio_nv']
                                         )

        # Compute other hitlet properties:
        # We have to loop here 3 times over all hitlets...
        strax.hitlet_properties(temp_hitlets)
        entropy = strax.conditional_entropy(temp_hitlets, template='flat', square_data=False)
        temp_hitlets['entropy'][:] = entropy
        strax.compute_widths(temp_hitlets)

        # Remove data field:
        hitlets = np.zeros(len(temp_hitlets), dtype=strax.hitlet_dtype())
        drop_data_field(temp_hitlets, hitlets)

        return hitlets

@numba.njit
def drop_data_field(old_hitlets, new_hitlets):
    """
    Function which copies everything except for the data field.
    If anyone know a better and faster way please let me know....

    :param old_hitlets:
    :param new_hitlets:
    :return:
    """
    n_hitlets = len(old_hitlets)
    for i in range(n_hitlets):
        o = old_hitlets[i]
        n = new_hitlets[i]

        n['time'] = o['time']
        n['length'] = o['length']
        n['dt'] = o['dt']
        n['channel'] = o['channel']
        n['hit_length'] = o['hit_length']
        n['area'] = o['area']
        n['amplitude'] = o['amplitude']
        n['time_amplitude'] = o['time_amplitude']
        n['entropy'] = o['entropy']
        n['width'][:] = o['width'][:]
        n['area_decile_from_midpoint'][:] = o['area_decile_from_midpoint'][:]
        n['fwhm'] = o['fwhm']
        n['fwtm'] = o['fwtm']
        n['left'] = o['left']
        n['low_left'] = o['low_left']
        n['record_i'] = o['record_i']
