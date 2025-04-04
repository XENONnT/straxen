import numpy as np
from immutabledict import immutabledict

import strax
import straxen

from straxen.plugins.defaults import NV_HIT_DEFAULTS

export, __all__ = strax.exporter()


@export
class nVETOHitlets(strax.Plugin):
    """Plugin which computes the nveto hitlets and their parameters.

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

    __version__ = "0.1.1"

    parallel = "process"
    rechunk_on_save = True
    # To reduce the number of chunks, we increase the target size
    # This would not harm memory usage, because we rechunk on load
    chunk_target_size_mb = 2000
    compressor = "zstd"

    depends_on = "records_nv"

    provides = "hitlets_nv"
    data_kind = "hitlets_nv"

    dtype = strax.hitlet_dtype()

    save_outside_hits_nv = straxen.URLConfig(
        default=NV_HIT_DEFAULTS["save_outside_hits_nv"],
        track=True,
        infer_type=False,
        help="Save (left, right) samples besides hits; cut the rest",
    )

    hit_min_amplitude_nv = straxen.URLConfig(
        infer_type=False,
        default=NV_HIT_DEFAULTS["hit_min_amplitude_nv"],
        track=True,
        help=(
            "Minimum hit amplitude in ADC counts above baseline. "
            "Specify as a tuple of length n_nveto_pmts, or a number."
        ),
    )

    min_split_nv = straxen.URLConfig(
        default=0.063,
        track=True,
        infer_type=False,
        help=(
            "Minimum height difference pe/sample between local minimum and maximum, "
            "that a pulse get split."
        ),
    )

    min_split_ratio_nv = straxen.URLConfig(
        default=0.75,
        track=True,
        infer_type=False,
        help=(
            "Min ratio between local maximum and minimum to split pulse (zero to switch this off)."
        ),
    )

    channel_map = straxen.URLConfig(
        track=False,
        type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) channel number.",
    )

    gain_model_nv = straxen.URLConfig(
        default=(
            "list-to-array://"
            "xedocs://pmt_area_to_pes"
            "?as_list=True"
            "&sort=pmt"
            "&detector=neutron_veto"
            "&run_id=plugin.run_id"
            "&version=ONLINE"
            "&attr=value"
        ),
        infer_type=False,
        help="PMT gain model. Specify as (model_type, model_config, nT = True)",
    )

    def setup(self):
        self.channel_range = self.channel_map["nveto"]
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1

        to_pe = self.gain_model_nv

        # Create to_pe array of size max channel:
        self.to_pe = np.zeros(self.channel_range[1] + 1, dtype=np.float32)
        self.to_pe[self.channel_range[0] :] = to_pe[:]

        # Assign attribute that might be used in daughter classes:
        self.hit_thresholds = self.hit_min_amplitude_nv

    def compute(self, records_nv, start, end):
        records_nv = remove_switched_off_channels(records_nv, self.to_pe)
        hits = strax.find_hits(records_nv, min_amplitude=self.hit_thresholds)

        temp_hitlets = strax.create_hitlets_from_hits(
            hits, self.save_outside_hits_nv, self.channel_range, chunk_start=start, chunk_end=end
        )
        del hits

        # Get hitlet data and split hitlets:
        temp_hitlets = strax.get_hitlets_data(
            temp_hitlets, records_nv, to_pe=self.to_pe, min_hitlet_sample=600
        )

        temp_hitlets = strax.split_peaks(
            temp_hitlets,
            None,  # Only needed for peak splitting
            records_nv,
            None,  # Only needed for peak splitting
            self.to_pe,
            data_type="hitlets",
            algorithm="local_minimum",
            min_height=self.min_split_nv,
            min_ratio=self.min_split_ratio_nv,
        )

        # Compute other hitlet properties:
        # We have to loop here 3 times over all hitlets...
        strax.hitlet_properties(temp_hitlets)

        # Remove data field:
        hitlets = np.zeros(len(temp_hitlets), dtype=strax.hitlet_dtype())
        strax.copy_to_buffer(temp_hitlets, hitlets, "_copy_hitlets")
        return strax.sort_by_time(hitlets)


def remove_switched_off_channels(records, to_pe):
    """Removes records of channels which gain was set to zero.

    :param records Hits found in records.
    :param to_pe: conversion factor from ADC per sample.
    :return: records

    """
    channel_off = np.argwhere(to_pe == 0).flatten()
    mask_off = np.isin(records["channel"], channel_off)
    records = records[~mask_off]
    return records
