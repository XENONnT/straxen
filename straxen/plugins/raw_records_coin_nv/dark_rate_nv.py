import datetime
from immutabledict import immutabledict
import strax
import numpy as np
from itertools import chain

# This makes sure shorthands for only the necessary functions
# are made available under straxen.[...]
export, __all__ = strax.exporter()

@export


class NVDarkRateMonitoring(strax.Plugin): 
    """
    Plugin to monitor the dark rate, the baseline and the baseline RMS of the Neutron Veto PMTs.

    From the "raw_records_coin_nv" data format, raw records within the first 10 seconds of the run are selected. Among these, only raw records
    with the variable "record_i" equals 0 are included in the monitor calculation.
        
    The plugin returns an array of 120 elements (one for each PMT), where each element contains:
    - the number of the channel;
    - the dark rate of the channel;
    - the dark rate error of the channel;
    - the mean of the baseline means of the channel;
    - the RMS of the baseline means of the channel;     
    - the mean of baseline RMSs of the channel;
    - the RMS of the baseline RMSs of the channel.     
    """

    __version__ = '0.0.11'
    depends_on = ('raw_records_coin_nv',)
    provides   = 'dark_rate_nv'
    data_kind  = 'dark_rate_nv'

    dtype =  strax.time_fields + [
            (('Channel', 'channel'),np.int32),
            (('Dark rate [Hz]','dark_rate'),np.float32),
            (('Dark rate error [Hz]','dark_rate_error'),np.float32),
            (('Baselines means mean [ADCc]', 'baselines_means_mean'),np.float32),
            (('Baselines means RMS [ADCc]', 'baselines_means_rms'),np.float32),
            (('Baselines RMSs mean [ADCc]', 'baselines_rmss_mean'), np.float32),
            (('Baselines RMSs RMS [ADCc]', 'baselines_rmss_rms'), np.float32)
            ]

    def compute(self, raw_records_coin_nv):

        if len(raw_records_coin_nv) == 0:
            return np.zeros(0, dtype=self.dtype)
        
        # Initialize the absolute run start time based on the first received chunk.
        # This value is stored as an instance attribute to persist across chunks.
        if not hasattr(self, '_true_start'):
            self._true_start = raw_records_coin_nv['time'][0]
        
        run_start = self._true_start
        # Define the cut-off at 10 seconds (1e10 nanoseconds) from run start.   
        t_limit = run_start + int(1e10)
            
        # If the current chunk duration is less than 10 seconds or starts after the 10 seconds threshold from the run start time, skip processing.
        # Returns an empty array with the required dtype to maintain pipeline consistency.
        if raw_records_coin_nv['time'][0] >= t_limit or raw_records_coin_nv['time'][-1]-raw_records_coin_nv['time'][0] < 1e10:
            return np.zeros(0, dtype=self.dtype)

        # Apply boolean masking to select only:
        # - record_i == 0: the start of each pulse (avoids counting fragments),
        # - time < t_limit: raw records within the valid 10 seconds window.
        mask = (raw_records_coin_nv['record_i'] == 0) & (raw_records_coin_nv['time'] < t_limit)
        r = raw_records_coin_nv[mask]
    
        # Prepare the result array with one element per monitored PMT channel.
        channels_to_monitor = np.array([i for i in range(2000,2120)])
        res = np.zeros(len(channels_to_monitor), dtype=self.dtype)

        # Set the channel and time information in the resulting array.
        res['channel'] = channels_to_monitor
        res['time'] = run_start
        res['endtime'] = t_limit

        # If no records are found in the first 10s, return the initialized empty results.
        if len(r) == 0:
            return res
    
        # Identify which channels fired and the frequency per channel.
        ch_found, counts = np.unique(r['channel'], return_counts=True)
        count_map = dict(zip(ch_found, counts))

        # Iterate through each configured channel to compute specific information.
        for i, ch in enumerate(channels_to_monitor):
            if ch in count_map:
                # Isolate data for the specific channel
                ch_data = r[r['channel'] == ch]
                n_events = count_map[ch]

                # Rate calculated as raw records per 10 seconds. Poissonian statistical error: sqrt(N) / time.
                res['dark_rate'][i] = n_events / 10.0
                res['dark_rate_error'][i] = np.sqrt(n_events) / 10.0
                    
                # Mean and RMS of the baseline stored in raw records.
                res['baselines_means_mean'][i] = np.mean(ch_data['baseline'])
                res['baselines_means_rms'][i] = np.std(ch_data['baseline'])
                    
                # Calculate baseline RMSs directly from the first 26 samples of the waveforms.
                wfs = ch_data['data'][:, :26]
                rms_per_pulse = np.std(wfs, axis=1)

                # Aggregate mean and standard deviation of baseline RMSs across all channel pulses.
                res['baselines_rmss_mean'][i] = np.mean(rms_per_pulse)
                res['baselines_rmss_rms'][i] = np.std(rms_per_pulse)
            else:
                #Assign 0 or NaN for channels with no activity in the window.
                res['dark_rate'][i] = 0
                res['dark_rate_error'][i]= np.nan
                res['baselines_means_mean'][i] = np.nan
                res['baselines_rmss_rms'][i] = np.nan
    
        return res