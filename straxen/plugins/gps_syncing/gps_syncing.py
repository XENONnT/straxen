import strax
import straxen
import utilix
import numpy as np
import pandas as pd
import datetime

from straxen.plugins.aqmon_hits.aqmon_hits import AqmonChannels
from scipy.interpolate import interp1d


class GpsSync(strax.OverlapWindowPlugin):
    """Correct the event times to GPS time.

    1. Finds the TTL GPS pulses coming into the AM from the gps
    module and their pairs coming from the module for the
    correspondent run.
      2. Corrects the timestamp of all events by linearly interpolating
    between the previous and next sync pulses.

    """

    __version__ = "0.2.1"
    depends_on = ("aqmon_hits", "event_basics")
    provides = "gps_sync"
    data_kind = "events"

    gps_channel = AqmonChannels.GPS_SYNC_AM
    _is_tpc = True  # If true tpc_internal_delay is subtracted from final
    # GPS results. Has to be set to False when inherited
    # for the neutron-veto.

    tpc_internal_delay = straxen.URLConfig(
        default={"0": 4917, "020380": 10137, "043040": 0},
        type=dict,
        track=True,
        help="Internal delay between aqmon and regular TPC channels ins [ns]",
    )

    gps_truth_vicinity = straxen.URLConfig(
        default=4.5,
        type=float,
        track=True,
        help=(
            "+/- time interval in [s] in which the GPS truth should be searched for each GPS pulse."
        ),
    )

    dtype = strax.time_fields + [(("GPS absolute time [ns]", "t_gps"), np.int64)]

    def get_window_size(self):
        # Use a large window to ensure that multiple GPS pulses are found
        # within the current chunk. Specified window is in nanoseconds.
        return int(120 * 10**9)

    def setup(self):
        # Load GPS-module pulses
        self.gps_times = self.load_gps_array()
        self.delay = 0
        if self._is_tpc:
            self.delay = self.get_delay()

    def load_gps_array(self):
        """Function which load GPS time information from data base and converts timestamps back into
        nanoseconds unix time."""
        gps_info = self.gps_times_from_runid(self.run_id)
        gps_info["pulse_time"] = np.int64(gps_info["gps_sec"] * 1e9) + np.int64(gps_info["gps_ns"])
        gps_array = np.sort(gps_info["pulse_time"])
        return gps_array

    def gps_times_from_runid(self, run_id):
        """Fetches the mongodb looking for the gps_sync collection for timestamps between the start
        and end times of a given run_id."""
        rundb = utilix.xent_collection()
        gps_times = utilix.xent_collection(collection="gps_sync", database="xenonnt")

        if isinstance(run_id, str):
            run_id = int(run_id)

        query = {"number": run_id}

        doc = rundb.find_one(query, projection={"start": 1, "end": 1})

        assert not (doc is None), "No match for run_id %s when computing GPS times." % self.run_id

        start_t = doc["start"].replace(tzinfo=datetime.timezone.utc).timestamp()
        end_t = doc["end"].replace(tzinfo=datetime.timezone.utc).timestamp()

        query = {"gps_sec": {"$gte": start_t - 11, "$lte": end_t + 11}, "channel": 0}

        return pd.DataFrame(gps_times.find(query))

    @staticmethod
    def match_gps_pulses_with_truth(gps_pulse_time, gps_truth_time, vicinity):
        """Function which matches GPS truth from the file stored in the rundb with the corresponding
        GPS pulse recorded by the acquisition monitor.

        :param gps_pulse_time: Unix ns timestamps corresponding to GPS pulse starts
        :param gps_truth_time: Unix ns timestamps of the GPS pulses
        :param vicinity: +/- vicinity in [s] for which GPS pulse is matched with GPS truth.

        """
        _gps_pulses = np.zeros(len(gps_pulse_time), strax.time_fields)
        _gps_pulses["time"] = gps_pulse_time - int(vicinity * 10**9)
        _gps_pulses["endtime"] = gps_pulse_time + int(vicinity * 10**9)

        _gps_truth = np.zeros(len(gps_truth_time), strax.time_fields)
        _gps_truth["time"] = gps_truth_time
        _gps_truth["endtime"] = gps_truth_time + 1

        truth_in_pulse_index = strax.fully_contained_in(_gps_truth, _gps_pulses)

        matched_gps_truth = gps_truth_time[truth_in_pulse_index >= 0]
        truth_in_pulse_index = truth_in_pulse_index[truth_in_pulse_index >= 0]

        matched_gps_pulses = gps_pulse_time[truth_in_pulse_index]

        return matched_gps_truth, matched_gps_pulses

    def compute_time_array(self, l_daq_sync, l_gps_sync, l_daq_evt):
        """Function which computes for each event the corresponding GPS time.

        :param l_daq_sync: Unix time ns timestamps for the recorded gps pulses
        :param l_gps_sync: Unix time ns timestamps for the GPS truth.
        :param l_daq_evt: Unix time ns timestamps for the events.

        """
        _has_no_gps_but_events = (len(l_daq_sync) == 0 or len(l_gps_sync) == 0) and len(l_daq_evt)
        if _has_no_gps_but_events:
            raise ValueError(
                f'Cannot compute GPS correction for "{self.run_id}". '
                f"There were {len(l_gps_sync)} GPS found in the run-docs, "
                "which are marching recorded GPS pulses, "
                f"with {len(l_daq_evt)} events."
            )

        _has_no_events = not len(l_daq_evt)
        if _has_no_events:
            return np.zeros(0, dtype=np.int64)

        # Important before interpolate convert times in unix ns times
        # into ns since run start, otherwise interpolation gets bit
        # issue with the conversion from int64 to float64.
        first_gps_time = l_daq_sync[0]

        interp_gps = interp1d(
            l_daq_sync - first_gps_time,
            l_gps_sync - first_gps_time,
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        event_times_gps = interp_gps(l_daq_evt - first_gps_time)

        # Now convert back into int64 and add offset again
        event_times_gps = event_times_gps.astype(np.int64)
        return event_times_gps + first_gps_time

    def load_aqmon_array(self, hits):
        gps_hits = hits[hits["channel"] == self.gps_channel]
        aqmon_array = gps_hits["time"]
        return aqmon_array

    def get_delay(self):
        delay = 0
        for run_id, _delay in self.tpc_internal_delay.items():
            if int(self.run_id) >= int(run_id):
                delay = _delay
        return delay

    def compute(self, aqmon_hits, events):
        hits = aqmon_hits
        evts = events

        # Load pulses from aqmon
        gps_pulse_times = self.load_aqmon_array(hits)

        # Match GPS pulses with GPS truth, interpolate and extrapolate results
        # in order to include data at run boundaries.
        # Since the clock drift is tiny an extrapolation should be sufficient.
        matched_gps_truth, matched_gps_pulses = self.match_gps_pulses_with_truth(
            gps_pulse_times, self.gps_times, self.gps_truth_vicinity
        )

        t_events_gps = self.compute_time_array(matched_gps_pulses, matched_gps_truth, evts["time"])

        ans = dict()
        ans["time"] = evts["time"]
        ans["endtime"] = evts["endtime"]
        ans["t_gps"] = t_events_gps - self.delay

        return ans


class GpsSyncDAQVeto(GpsSync):
    """Plugin which computes veto_intervals using GPS times.

    Required to compute correctly total lifetime loss of experiment.

    """

    __version__ = "0.0.1"
    depends_on = ("aqmon_hits", "veto_intervals")
    provides = "veto_intervals_gps_sync"
    data_kind = "veto_intervals"
    child_plugin = True

    def compute(self, aqmon_hits, veto_intervals):
        res = super().compute(aqmon_hits, veto_intervals)
        return res


class GpsSync_nv(GpsSync):
    """Computes absolute GPS time for nveto data."""

    adc_threshold_nim_signal = straxen.URLConfig(
        default=500, type=int, track=True, help="Threshold in [adc] to search for the NIM signal"
    )
    # This value is only valid for SR0:
    epsilon_offset_nv_gps_pulse = straxen.URLConfig(
        default=76, type=int, track=True, help="Measured missing offset for nveto in [ns]"
    )

    __version__ = "0.1.0"
    depends_on = ("raw_records_aqmon_nv", "events_nv")
    provides = "events_gps_nv"
    data_kind = "events_nv"

    gps_channel = AqmonChannels.GPS_SYNC_NV
    _is_tpc = False

    dtype = strax.time_fields + [
        (("GPS absolute time [ns]", "time_sync_gps"), np.int64),
    ]

    def setup(self):
        super().setup()

    @staticmethod
    def get_nim_edge(raw_records, threshold=500):
        records = strax.raw_to_records(raw_records)
        strax.baseline(records)
        hits = strax.find_hits(records, min_amplitude=threshold)
        return hits

    def compute(self, raw_records_aqmon_nv, events_nv):
        rr_nv = raw_records_aqmon_nv
        _mask_nveto = rr_nv["channel"] == self.gps_channel
        hits_nv = self.get_nim_edge(rr_nv[_mask_nveto], self.adc_threshold_nim_signal)
        nveto_extra_offset = 0
        if not len(hits_nv):
            # During SR0 sync signal was not recorded properly for the
            # neutron-veto, hence take waveform itself as "hits".
            _mask_nveto &= rr_nv["record_i"] == 0
            nveto_extra_offset = self.epsilon_offset_nv_gps_pulse
            hits_nv = rr_nv[_mask_nveto]
        hits_nv["time"] += nveto_extra_offset

        res = super().compute(aqmon_hits=hits_nv, events=events_nv)
        gps_nv = np.zeros(len(res["time"]), self.dtype)
        gps_nv["time"] = res["time"]
        gps_nv["endtime"] = res["endtime"]
        gps_nv["time_sync_gps"] = res["t_gps"]
        return gps_nv


class GpsSync_mv(GpsSync):
    """Computes absolute GPS time for mveto data."""

    adc_threshold_nim_signal = straxen.URLConfig(
        default=500, type=int, track=True, help="Threshold in [adc] to search for the NIM signal"
    )

    __version__ = "0.1.0"
    depends_on = ("raw_records_aux_mv", "events_mv")
    provides = "events_gps_mv"
    data_kind = "events_mv"

    gps_channel = AqmonChannels.GPS_SYNC_MV
    _is_tpc = False

    dtype = strax.time_fields + [
        (("GPS absolute time [ns]", "time_sync_gps"), np.int64),
    ]

    def setup(self):
        super().setup()

    @staticmethod
    def get_nim_edge(raw_records, threshold=500):
        records = strax.raw_to_records(raw_records)
        strax.baseline(records)
        hits = strax.find_hits(records, min_amplitude=threshold)
        return hits

    def compute(self, raw_records_aux_mv, events_mv):
        rr_mv = raw_records_aux_mv
        _mask_nveto = rr_mv["channel"] == self.gps_channel
        hits_mv = self.get_nim_edge(rr_mv[_mask_nveto], self.adc_threshold_nim_signal)

        res = super().compute(aqmon_hits=hits_mv, events=events_mv)
        gps_mv = np.zeros(len(res["time"]), self.dtype)
        gps_mv["time"] = res["time"]
        gps_mv["endtime"] = res["endtime"]
        gps_mv["time_sync_gps"] = res["t_gps"]
        return gps_mv
