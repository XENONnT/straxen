import numba
import numpy as np
from immutabledict import immutabledict

import strax
import straxen

from straxen.get_corrections import is_cmt_option

export, __all__ = strax.exporter()

MV_PREAMBLE = 'Muno-Veto Plugin: Same as the corresponding nVETO-PLugin.\n'


@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_nv',
        default=(3, 15), track=True, infer_type=False,
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'hit_min_amplitude_nv',
        default=('hit_thresholds_nv', 'ONLINE', True), track=True, infer_type=False,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number, '
             'or a string like "pmt_commissioning_initial" which means calling '
             'hitfinder_thresholds.py, '
             'or a tuple like (correction=str, version=str, nT=boolean), '
             'which means we are using cmt.'),
    strax.Option(
        'min_split_nv',
        default=0.063, track=True, infer_type=False,
        help='Minimum height difference pe/sample between local minimum and maximum, '
             'that a pulse get split.'),
    strax.Option(
        'min_split_ratio_nv',
        default=0.75, track=True, infer_type=False,
        help='Min ratio between local maximum and minimum to split pulse (zero to switch this '
             'off).'),
    strax.Option(
        'entropy_template_nv',
        default='flat', track=True, infer_type=False,
        help='Template data is compared with in conditional entropy. Can be either "flat" or an '
             'template array.'),
    strax.Option(
        'entropy_square_data_nv',
        default=False, track=True, infer_type=False,
        help='Parameter which decides if data is first squared before normalized and compared to '
             'the template.'),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number."),
    strax.Option('gain_model_nv',
                 default=("to_pe_model_nv", "ONLINE", True), infer_type=False,
                 help='PMT gain model. Specify as (model_type, model_config, nT = True)'),
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
    __version__ = '0.1.1'

    parallel = 'process'
    rechunk_on_save = True
    compressor = 'zstd'

    depends_on = 'records_nv'

    provides = 'hitlets_nv'
    data_kind = 'hitlets_nv'
    ends_with = '_nv'

    dtype = strax.hitlet_dtype()

    def setup(self):
        self.channel_range = self.config['channel_map']['nveto']
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1

        to_pe = straxen.get_correction_from_cmt(self.run_id,
                                                self.config['gain_model_nv'])

        # Create to_pe array of size max channel:
        self.to_pe = np.zeros(self.channel_range[1] + 1, dtype=np.float32)
        self.to_pe[self.channel_range[0]:] = to_pe[:]

        # Check config of `hit_min_amplitude_nv` and define hit thresholds
        # if cmt config
        if is_cmt_option(self.config['hit_min_amplitude_nv']):
            self.hit_thresholds = straxen.get_correction_from_cmt(
                self.run_id, self.config['hit_min_amplitude_nv'])
        # if hitfinder_thresholds config
        elif isinstance(self.config['hit_min_amplitude_nv'], str):
            self.hit_thresholds = straxen.hit_min_amplitude(
                self.config['hit_min_amplitude_nv'])
        else:  # int or array
            self.hit_thresholds = self.config['hit_min_amplitude_nv']

    def compute(self, records_nv, start, end):
        hits = strax.find_hits(records_nv, min_amplitude=self.hit_thresholds)
        hits = remove_switched_off_channels(hits, self.to_pe)

        temp_hitlets = strax.create_hitlets_from_hits(hits,
                                                      self.config['save_outside_hits_nv'],
                                                      self.channel_range,
                                                      chunk_start=start,
                                                      chunk_end=end)
        del hits

        # Get hitlet data and split hitlets:
        temp_hitlets = strax.get_hitlets_data(temp_hitlets,
                                              records_nv,
                                              to_pe=self.to_pe,
                                              min_hitlet_sample=600)

        temp_hitlets = strax.split_peaks(temp_hitlets,
                                         None,  # Only needed for peak splitting
                                         records_nv,
                                         None,  # Only needed for peak splitting
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

        # Remove data field:
        hitlets = np.zeros(len(temp_hitlets), dtype=strax.hitlet_dtype())
        strax.copy_to_buffer(temp_hitlets, hitlets, '_copy_hitlets')
        return hitlets


def remove_switched_off_channels(hits, to_pe):
    """Removes hits which were found in a channel without any gain.
    :param hits: Hits found in records.
    :param to_pe: conversion factor from ADC per sample.
    :return: Hits
    """
    channel_off = np.argwhere(to_pe == 0).flatten()
    mask_off = np.isin(hits['channel'], channel_off)
    hits = hits[~mask_off]
    return hits


@export
@strax.takes_config(
    strax.Option(
        'save_outside_hits_mv',
        default=(2, 5), track=True, infer_type=False,
        child_option=True, parent_option_name='save_outside_hits_nv',
        help='Save (left, right) samples besides hits; cut the rest'),
    strax.Option(
        'hit_min_amplitude_mv', infer_type=False,
        default=('hit_thresholds_mv', 'ONLINE', True), track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_mveto_pmts, or a number, '
             'or a string like "pmt_commissioning_initial" which means calling '
             'hitfinder_thresholds.py, '
             'or a tuple like (correction=str, version=str, nT=boolean),'
             'which means we are using cmt.'),
    strax.Option(
        'min_split_mv',
        default=100, track=True, infer_type=False,
        child_option=True, parent_option_name='min_split_nv',
        help='Minimum height difference pe/sample between local minimum and maximum, '
             'that a pulse get split.'),
    strax.Option(
        'min_split_ratio_mv',
        default=0, track=True, infer_type=False,
        child_option=True, parent_option_name='min_split_ratio_nv',
        help='Min ratio between local maximum and minimum to split pulse (zero to switch this '
             'off).'),
    strax.Option(
        'entropy_template_mv',
        default='flat', track=True, infer_type=False,
        child_option=True, parent_option_name='entropy_template_nv',
        help='Template data is compared with in conditional entropy. Can be either "flat" or a '
             'template array.'),
    strax.Option(
        'entropy_square_data_mv',
        default=False, track=True, infer_type=False,
        child_option=True, parent_option_name='entropy_square_data_nv',
        help='Parameter which decides if data is first squared before normalized and compared to '
             'the template.'),
    strax.Option('gain_model_mv',
                 default=("to_pe_model_mv", "ONLINE", True), infer_type=False,
                 child_option=True, parent_option_name='gain_model_nv',
                 help='PMT gain model. Specify as (model_type, model_config)'),
)
class muVETOHitlets(nVETOHitlets):
    __doc__ = MV_PREAMBLE + nVETOHitlets.__doc__
    __version__ = '0.0.2'
    depends_on = 'records_mv'

    provides = 'hitlets_mv'
    data_kind = 'hitlets_mv'
    child_plugin = True

    dtype = strax.hitlet_dtype()

    def setup(self):
        self.channel_range = self.config['channel_map']['mv']
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1

        to_pe = straxen.get_correction_from_cmt(
            self.run_id, self.config['gain_model_mv'])

        # Create to_pe array of size max channel:
        self.to_pe = np.zeros(self.channel_range[1] + 1, dtype=np.float32)
        self.to_pe[self.channel_range[0]:] = to_pe[:]

        # Check config of `hit_min_amplitude_mv` and define hit thresholds
        # if cmt config
        if is_cmt_option(self.config['hit_min_amplitude_mv']):
            self.hit_thresholds = straxen.get_correction_from_cmt(self.run_id,
                                                                  self.config[
                                                                      'hit_min_amplitude_mv'])
        # if hitfinder_thresholds config
        elif isinstance(self.config['hit_min_amplitude_mv'], str):
            self.hit_thresholds = straxen.hit_min_amplitude(
                self.config['hit_min_amplitude_mv'])
        else:  # int or array
            self.hit_thresholds = self.config['hit_min_amplitude_mv']

    def compute(self, records_mv, start, end):
        return super().compute(records_mv, start, end)
