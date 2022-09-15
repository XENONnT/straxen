import numpy as np
from immutabledict import immutabledict

import strax
import straxen

from straxen.plugins.veto_pulse_processing import MV_PREAMBLE, NV_HIT_DEFAULTS, MV_HIT_DEFAULTS

export, __all__ = strax.exporter()


@export
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

    dtype = strax.hitlet_dtype()

    save_outside_hits_nv = straxen.URLConfig(
        default=NV_HIT_DEFAULTS['save_outside_hits_nv'], track=True, infer_type=False,
        help='Save (left, right) samples besides hits; cut the rest')

    hit_min_amplitude_nv = straxen.URLConfig(
        infer_type=False,
        default=NV_HIT_DEFAULTS['hit_min_amplitude_nv'], track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_nveto_pmts, or a number, '
             'or a string like "pmt_commissioning_initial" which means calling '
             'hitfinder_thresholds.py, '
             'or a tuple like (correction=str, version=str, nT=boolean), '
             'which means we are using cmt.')

    min_split_nv = straxen.URLConfig(
        default=0.063, track=True, infer_type=False,
        help='Minimum height difference pe/sample between local minimum and maximum, '
             'that a pulse get split.')

    min_split_ratio_nv = straxen.URLConfig(
        default=0.75, track=True, infer_type=False,
        help='Min ratio between local maximum and minimum to split pulse (zero to switch this '
             'off).')

    entropy_template_nv = straxen.URLConfig(
        default='flat', track=True, infer_type=False,
        help='Template data is compared with in conditional entropy. Can be either "flat" or an '
             'template array.')

    entropy_square_data_nv = straxen.URLConfig(
        default=False, track=True, infer_type=False,
        help='Parameter which decides if data is first squared before normalized and compared to '
             'the template.')

    channel_map = straxen.URLConfig(track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number.")

    gain_model_nv = straxen.URLConfig(
                 default="cmt://to_pe_model_nv?version=ONLINE&run_id=plugin.run_id", infer_type=False,
                 help='PMT gain model. Specify as (model_type, model_config, nT = True)')

    def setup(self):
        self.channel_range = self.channel_map['nveto']
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1

        to_pe = self.gain_model_nv

        # Create to_pe array of size max channel:
        self.to_pe = np.zeros(self.channel_range[1] + 1, dtype=np.float32)
        self.to_pe[self.channel_range[0]:] = to_pe[:]

        # Check config of `hit_min_amplitude_nv` and define hit thresholds
        # if cmt config
        self.hit_thresholds = self.hit_min_amplitude_nv
        
    def compute(self, records_nv, start, end):
        records_nv = remove_switched_off_channels(records_nv, self.to_pe)
        hits = strax.find_hits(records_nv, min_amplitude=self.hit_thresholds)

        temp_hitlets = strax.create_hitlets_from_hits(hits,
                                                      self.save_outside_hits_nv,
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
                                         min_height=self.min_split_nv,
                                         min_ratio=self.min_split_ratio_nv,
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
class muVETOHitlets(nVETOHitlets):
    __doc__ = MV_PREAMBLE + nVETOHitlets.__doc__
    __version__ = '0.0.2'
    depends_on = 'records_mv'

    provides = 'hitlets_mv'
    data_kind = 'hitlets_mv'
    child_plugin = True

    dtype = strax.hitlet_dtype()

    save_outside_hits_mv = straxen.URLConfig(
        default=MV_HIT_DEFAULTS['save_outside_hits_mv'], track=True, infer_type=False,
        child_option=True, parent_option_name='save_outside_hits_nv',
        help='Save (left, right) samples besides hits; cut the rest')

    hit_min_amplitude_mv = straxen.URLConfig(
        infer_type=False,
        default=MV_HIT_DEFAULTS['hit_min_amplitude_mv'], track=True,
        help='Minimum hit amplitude in ADC counts above baseline. '
             'Specify as a tuple of length n_mveto_pmts, or a number, '
             'or a string like "pmt_commissioning_initial" which means calling '
             'hitfinder_thresholds.py, '
             'or a tuple like (correction=str, version=str, nT=boolean),'
             'which means we are using cmt.')

    min_split_mv = straxen.URLConfig(
        default=100, track=True, infer_type=False,
        child_option=True, parent_option_name='min_split_nv',
        help='Minimum height difference pe/sample between local minimum and maximum, '
             'that a pulse get split.')

    min_split_ratio_mv = straxen.URLConfig(
        default=0, track=True, infer_type=False,
        child_option=True, parent_option_name='min_split_ratio_nv',
        help='Min ratio between local maximum and minimum to split pulse (zero to switch this '
             'off).')

    entropy_template_mv = straxen.URLConfig(
        default='flat', track=True, infer_type=False,
        child_option=True, parent_option_name='entropy_template_nv',
        help='Template data is compared with in conditional entropy. Can be either "flat" or a '
             'template array.')

    entropy_square_data_mv = straxen.URLConfig(
        default=False, track=True, infer_type=False,
        child_option=True, parent_option_name='entropy_square_data_nv',
        help='Parameter which decides if data is first squared before normalized and compared to '
             'the template.')

    gain_model_mv = straxen.URLConfig(
                 default="cmt://to_pe_model_mv?version=ONLINE&run_id=plugin.run_id",infer_type=False,
                 child_option=True, parent_option_name='gain_model_nv',
                 help='PMT gain model. Specify as (model_type, model_config)')

    def setup(self):
        self.channel_range = self.channel_map['mv']
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1

        to_pe = self.gain_model_mv
        # Create to_pe array of size max channel:
        self.to_pe = np.zeros(self.channel_range[1] + 1, dtype=np.float32)
        self.to_pe[self.channel_range[0]:] = to_pe[:]

        self.hit_thresholds = self.hit_min_amplitude_mv

    def compute(self, records_mv, start, end):
        return super().compute(records_mv, start, end)
