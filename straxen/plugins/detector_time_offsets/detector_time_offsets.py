import strax
import straxen
import numpy as np
from straxen.plugins.aqmon_hits.aqmon_hits import AqmonChannels


class DetectorSynchronization(strax.Plugin):
    """Plugin which computes the synchronization delay between TPC and vetos.

    Reference:
        * xenon:xenonnt:dsg:mveto:sync_monitor
    """

    __version__ = "0.0.3"
    depends_on = ("raw_records_aqmon", "raw_records_aqmon_nv", "raw_records_aux_mv")
    provides = "detector_time_offsets"
    data_kind = "detector_time_offsets"

    tpc_internal_delay = straxen.URLConfig(
        default={"0": 4917, "020380": 10137},
        type=dict,
        track=True,
        help="Internal delay between aqmon and regular TPC channels ins [ns]",
    )
    adc_threshold_nim_signal = straxen.URLConfig(
        default=500, type=int, track=True, help="Threshold in [adc] to search for the NIM signal"
    )
    # This value is only valid for SR0:
    epsilon_offset = straxen.URLConfig(
        default=76, type=int, track=True, help="Measured missing offset for nveto in [ns]"
    )
    sync_max_delay = strax.Config(default=11e3, help="max delay DetectorSynchronization [ns]")
    sync_expected_min_clock_distance = straxen.URLConfig(
        default=9.9e9, help="min clock distance DetectorSynchronization [ns]"
    )
    sync_expected_max_clock_distance = straxen.URLConfig(
        default=10.1e9, help="max clock distance DetectorSynchronization [ns]"
    )

    def infer_dtype(self):
        dtype = []
        dtype += strax.time_fields
        dtype += [
            (("Time offset for nV to synchronize with TPC in [ns]", "time_offset_nv"), np.int64),
            (("Time offset for mV to synchronize with TPC in [ns]", "time_offset_mv"), np.int64),
        ]
        return dtype

    def compute(self, raw_records_aqmon, raw_records_aqmon_nv, raw_records_aux_mv, start, end):
        rr_tpc = raw_records_aqmon
        rr_nv = raw_records_aqmon_nv
        rr_mv = raw_records_aux_mv

        extra_offset = 0
        _mask_tpc = rr_tpc["channel"] == AqmonChannels.GPS_SYNC
        if not np.any(_mask_tpc):
            # For some runs in the beginning no signal has been acquired here.
            # In that case we have to add the internal DAQ delay as an extra offset later.
            _mask_tpc = rr_tpc["channel"] == AqmonChannels.GPS_SYNC_AM
            extra_offset = self.get_delay()

        hits_tpc = self.get_nim_edge(rr_tpc[_mask_tpc], self.config["adc_threshold_nim_signal"])
        hits_tpc["time"] += extra_offset

        _mask_mveto = rr_mv["channel"] == AqmonChannels.GPS_SYNC_MV
        hits_mv = self.get_nim_edge(rr_mv[_mask_mveto], self.config["adc_threshold_nim_signal"])

        _mask_nveto = rr_nv["channel"] == AqmonChannels.GPS_SYNC_NV
        hits_nv = self.get_nim_edge(rr_nv[_mask_nveto], self.config["adc_threshold_nim_signal"])
        nveto_extra_offset = 0
        if not len(hits_nv):
            # During SR0 sync signal was not recorded properly for the
            # neutron-veto, hence take waveform itself as "hits".
            _mask_nveto &= rr_nv["record_i"] == 0
            nveto_extra_offset = self.config["epsilon_offset"]
            hits_nv = rr_nv[_mask_nveto]
        hits_nv["time"] += nveto_extra_offset

        offsets_mv = self.estimate_delay(hits_tpc, hits_mv)
        offsets_nv = self.estimate_delay(hits_tpc, hits_nv)
        assert len(offsets_mv) == len(offsets_nv), "Unequal number of sync signals!"

        result = np.zeros(len(offsets_mv), dtype=self.dtype)
        result["time"] = hits_tpc["time"]
        result["endtime"] = strax.endtime(hits_tpc)
        result["time_offset_nv"] = offsets_nv
        result["time_offset_mv"] = offsets_mv

        return result

    def get_delay(self):
        delay = 0
        for run_id, _delay in self.config["tpc_internal_delay"].items():
            if int(self.run_id) >= int(run_id):
                delay = _delay
        return delay

    @staticmethod
    def get_nim_edge(raw_records, threshold=500):
        records = strax.raw_to_records(raw_records)
        strax.baseline(records)
        hits = strax.find_hits(records, min_amplitude=threshold)
        return hits

    def estimate_delay(self, hits_det0, hits_det1):
        """Function to estimate the average offset between two hits."""
        err_value = -10000000000

        offsets = []
        prev_time = 0
        for ind in range(len(hits_det0)):
            offset = self.find_offset_nearest(hits_det1["time"], hits_det0["time"][ind])
            if ind:
                # Cannot compute time to prev for first event
                time_to_prev = hits_det0["time"][ind] - prev_time
            else:
                time_to_prev = 10e9

            # Additional check to avoid spurious signals
            _correct_distance_to_prev_lock = time_to_prev >= self.sync_expected_min_clock_distance
            _correct_distance_to_prev_lock = time_to_prev < self.sync_expected_max_clock_distance
            if (abs(offset) < self.sync_max_delay) & _correct_distance_to_prev_lock:
                offsets.append(offset)
                prev_time = hits_det0["time"][ind]
            else:
                # Add err_value in case offset is not valid
                offsets.append(err_value)
                prev_time = hits_det0["time"][ind]

        return np.array(offsets)

    def find_offset_nearest(self, array, value):
        if not len(array):
            return -self.sync_max_delay
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return value - array[idx]
