import numpy as np
import strax
import straxen
from immutabledict import immutabledict

export, __all__ = strax.exporter()


@export
class NVDarkRateMonitoring(strax.Plugin):
    """Plugin to monitor the dark rate, the baseline and the baseline RMS of the Neutron Veto PMTs.

    From the "raw_records_coin_nv" data format,
    raw records within the first 10 seconds of the run are selected.
    Among these, only raw records with the variable "record_i" equals
    0 are included in the monitor calculation.

    The plugin returns an array of 120 elements (one for each PMT),
    where each element contains:
    - the number of the channel;
    - the dark rate of the channel;
    - the dark rate error of the channel;
    - the mean of the baseline means of the channel;
    - the RMS of the baseline means of the channel;
    - the mean of baseline RMSs of the channel;
    - the RMS of the baseline RMSs of the channel.

    """

    __version__ = "0.0.6"
    depends_on = ("raw_records_coin_nv",)
    provides = "monitor_nv"
    data_kind = "monitor_nv"

    # Configurations
    channel_map = straxen.URLConfig(
        track=False,
        type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) channel number.",
    )

    run_start = straxen.URLConfig(
        default="runstart://plugin.run_id?",
        track=False,
        infer_type=False,
        help="Returns run start in utc unix time in ns.",
    )

    keep_n_seconds_for_monitoring = straxen.URLConfig(
        default=10,
        track=False,
        infer_type=False,
        help="Number of seconds which should be used for monitoring.",
    )

    baseline_software_trigger_samples_nv = straxen.URLConfig(
        infer_type=False,
        default=26,
        track=True,
        help="Number of samples used in baseline rms calculation",
    )

    def setup(self):
        """Setup method to initialize channel range and baseline samples."""
        self.baseline_samples = self.baseline_software_trigger_samples_nv
        self.channel_range = self.channel_map["nveto"]
        self.channels_to_monitor = np.arange(self.channel_range[0], self.channel_range[1] + 1)

    def infer_dtype(self):
        """Infer dtype based on channel range."""
        dtype = strax.time_fields + [
            (("Channel", "channel"), np.int32),
            (("Dark rate [Hz]", "dark_rate"), np.float32),
            (("Dark rate error [Hz]", "dark_rate_error"), np.float32),
            (("Baselines means mean [ADCc]", "baselines_means_mean"), np.float32),
            (("Baselines means RMS [ADCc]", "baselines_means_rms"), np.float32),
            (("Baselines RMSs mean [ADCc]", "baselines_rmss_mean"), np.float32),
            (("Baselines RMSs RMS [ADCc]", "baselines_rmss_rms"), np.float32),
        ]
        return dtype

    def compute(self, raw_records_coin_nv):
        # If the chunk is empty there is nothing to monitor
        if len(raw_records_coin_nv) == 0:
            return np.zeros(0, dtype=self.dtype)

        run_start = self.run_start
        # Upper time limit of the monitoring window
        t_limit = run_start + self.keep_n_seconds_for_monitoring * straxen.units.s

        # If the first record of the chunk is already past the monitoring
        # window, the whole chunk is outside it: nothing to do
        if raw_records_coin_nv["time"][0] >= t_limit:
            return np.zeros(0, dtype=self.dtype)

        # Calculate effective monitoring time for this chunk
        chunk_start = raw_records_coin_nv["time"][0]
        chunk_end = strax.endtime(raw_records_coin_nv).max()

        # The effective interval is the overlap between the chunk and the
        # monitoring window
        effective_start = max(chunk_start, run_start)
        effective_end = min(chunk_end, t_limit)
        effective_time = (effective_end - effective_start) / straxen.units.s

        # If there is no real overlap, there is nothing to monitor
        # in this chunk
        if effective_time <= 0:
            return np.zeros(0, dtype=self.dtype)

        # Data selection
        mask = (raw_records_coin_nv["record_i"] == 0) & (raw_records_coin_nv["time"] < t_limit)
        r = raw_records_coin_nv[mask]

        # Initialize the output array: one element per each of the 120 channels
        res = np.zeros(len(self.channels_to_monitor), dtype=self.dtype)
        res["channel"] = self.channels_to_monitor
        res["time"] = effective_start
        res["endtime"] = effective_end

        # If no data is left after the filter, still return the array with
        # channel/time/endtime populated (the rest stays at zero)
        if len(r) == 0:
            return res

        # Count how many records were found for each channel
        ch_found, counts = np.unique(r["channel"], return_counts=True)
        count_map = dict(zip(ch_found, counts))

        # Compute statistics for each channel
        for i, ch in enumerate(self.channels_to_monitor):
            if ch in count_map:
                ch_data = r[r["channel"] == ch]
                n_events = count_map[ch]

                # Dark rate = number of events / effective monitoring time
                res["dark_rate"][i] = n_events / effective_time
                # Poissonian statistical error on the count
                res["dark_rate_error"][i] = np.sqrt(n_events) / effective_time

                # Baseline mean statistics
                res["baselines_means_mean"][i] = np.mean(ch_data["baseline"])
                res["baselines_means_rms"][i] = np.std(ch_data["baseline"])

                # Compute the waveform RMS
                n_samples = min(self.baseline_samples, ch_data["data"].shape[1])
                wfs = ch_data["data"][:, :n_samples]
                rms_per_pulse = np.std(wfs, axis=1)

                # Mean and RMS of the RMS values
                res["baselines_rmss_mean"][i] = np.mean(rms_per_pulse)
                res["baselines_rmss_rms"][i] = np.std(rms_per_pulse)
            else:
                # No data for this channel in this chunk:
                # dark_rate set to 0 (no counts), everything else NaN
                res["dark_rate"][i] = 0
                res["dark_rate_error"][i] = np.nan
                res["baselines_means_mean"][i] = np.nan
                res["baselines_means_rms"][i] = np.nan
                res["baselines_rmss_mean"][i] = np.nan
                res["baselines_rmss_rms"][i] = np.nan

        return res
