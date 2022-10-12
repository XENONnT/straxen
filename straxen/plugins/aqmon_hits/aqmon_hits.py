from enum import IntEnum
import numpy as np
import strax
import straxen

from  straxen.plugins.raw_records.daqreader import ARTIFICIAL_DEADTIME_CHANNEL

export, __all__ = strax.exporter()


# More info about the acquisition monitor can be found here:
# https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:alexelykov:acquisition_monitor

@export
class AqmonChannels(IntEnum):
    """Mapper of named aqmon channels to ints"""
    MV_TRIGGER = 797
    GPS_SYNC = 798
    ARTIFICIAL_DEADTIME = ARTIFICIAL_DEADTIME_CHANNEL
    # Analogue sum waveform
    SUM_WF = 800
    # GPS sync acquisition monitor
    GPS_SYNC_AM = 801
    # HighEnergyVeto
    HEV_STOP = 802
    HEV_START = 803
    # To avoid confusion with HEV, these are the high energy boards
    BUSY_HE_STOP = 804
    BUSY_HE_START = 805
    # Low energy boards (main chain)
    BUSY_STOP = 806
    BUSY_START = 807
    # nVETO GPS_Sync
    GPS_SYNC_NV = 813
    # mVETO GPS_Sync
    GPS_SYNC_MV = 1084


@export
class AqmonHits(strax.Plugin):
    """
    Find hits in acquisition monitor data. These hits could be
    then used by other plugins for deadtime calculations,
    GPS SYNC analysis, etc.
    """
    save_when = strax.SaveWhen.TARGET
    __version__ = '1.1.2'
    hit_min_amplitude_aqmon = straxen.URLConfig(
        default=(
            # Analogue signals
            (50, (int(AqmonChannels.SUM_WF),)),
            # Digital signals, can set a much higher threshold
            (1500, (
                int(AqmonChannels.MV_TRIGGER),
                int(AqmonChannels.GPS_SYNC),
                int(AqmonChannels.GPS_SYNC_AM),
                int(AqmonChannels.HEV_STOP),
                int(AqmonChannels.HEV_START),
                int(AqmonChannels.BUSY_HE_STOP),
                int(AqmonChannels.BUSY_HE_START),
                int(AqmonChannels.BUSY_STOP),
                int(AqmonChannels.BUSY_START),)),
            # Fake signals, 0 meaning that we won't find hits using
            # strax but just look for starts and stops
            (0, (int(AqmonChannels.ARTIFICIAL_DEADTIME),)),
        ),
        track=True,
        help='Minimum hit threshold in ADC*counts above baseline. Specified '
             'per channel in the format (threshold, (chx,chy),)',
    )
    baseline_samples_aqmon = straxen.URLConfig(
        default=10,
        track=True,
        help='Number of samples to use at the start of the pulse to determine the baseline'
    )
    check_raw_record_aqmon_overlaps = straxen.URLConfig(
        default=True,
        track=False,
        help='Crash if any of the pulses in raw_records_aqmon overlap with others '
             'in the same channel'
    )

    depends_on = 'raw_records_aqmon'
    provides = 'aqmon_hits'
    data_kind = 'aqmon_hits'

    dtype = strax.hit_dtype

    def compute(self, raw_records_aqmon):
        not_allowed_channels = (set(np.unique(raw_records_aqmon['channel']))
                                - set(self.aqmon_channels))
        if not_allowed_channels:
            raise ValueError(
                f'Unknown channel {not_allowed_channels}. Only know {self.aqmon_channels}')

        if self.check_raw_record_aqmon_overlaps:
            straxen.check_overlaps(raw_records_aqmon,
                                   n_channels = max(AqmonChannels).value + 1
            )

        records = strax.raw_to_records(raw_records_aqmon)
        strax.zero_out_of_bounds(records)
        strax.baseline(records, baseline_samples=self.baseline_samples_aqmon, flip=True)
        aqmon_hits = self.find_aqmon_hits_per_channel(records)
        aqmon_hits = strax.sort_by_time(aqmon_hits)
        return aqmon_hits

    @property
    def aqmon_channels(self):
        return [channel for hit_and_channel_list in self.hit_min_amplitude_aqmon
                for channel in hit_and_channel_list[1]]

    def find_aqmon_hits_per_channel(self, records):
        """Allow different thresholds to be applied to different channels"""
        aqmon_thresholds = np.zeros(np.max(self.aqmon_channels) + 1)
        for hit_threshold, channels in self.hit_min_amplitude_aqmon:
            aqmon_thresholds[np.array(channels)] = hit_threshold

        # Split the artificial deadtime ones and do those separately if there are any
        is_artificial = records['channel'] == AqmonChannels.ARTIFICIAL_DEADTIME
        aqmon_hits = strax.find_hits(records[~is_artificial],
                                     min_amplitude=aqmon_thresholds)

        if np.sum(is_artificial):
            aqmon_hits = np.concatenate([
                aqmon_hits, self.get_deadtime_hits(records[is_artificial])])
        return aqmon_hits

    def get_deadtime_hits(self, artificial_deadtime):
        """
        Actually, the artificial deadtime hits are already an interval so
        we only have to copy the appropriate hits
        """
        hits = np.zeros(len(artificial_deadtime), dtype=self.dtype)
        hits['time'] = artificial_deadtime['time']
        hits['dt'] = artificial_deadtime['dt']
        hits['length'] = artificial_deadtime['length']
        hits['channel'] = artificial_deadtime['channel']
        return hits
