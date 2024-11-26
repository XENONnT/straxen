import numpy as np
import strax
import straxen
import pandas as pd
import numba
import warnings
import os

class LoneHitEventBasics(straxen.plugins.event_basics.EventBasics):
    '''EventBasics class adapted to work with the event times triggered by the neutron veto coincidences. The default event_basics class
    gives an error when no peak is found in the event, which is not expected to happen with the default event class building on a 
    triggering peak. Now this can happen since an event is no longer triggered by peaks in the TPC but in the neutron veto.
    If no S2 is found, the event is discarded.
    '''
    provides = 'event_basics'
    data_kind='events'
    depends_on = ['peak_basics','peak_positions','peak_proximity','events_nv_triggered',]
    save_when = strax.SaveWhen.ALWAYS
    __version__ = '0.0.1'
    child_plugin = True
    
    allow_s1s_before_max_drift_time = straxen.URLConfig(
        default=False,
        infer_type=False,
        help="Allow S1s before one drift time from main S2 to become main or alternative S1",
    )
    gain_model = straxen.URLConfig(
        infer_type=False,
        help='PMT gain model. Specify as URL or explicit value',
        default='cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id',
    )
    lonehit_coincidence_window = straxen.URLConfig(
        default=(600,600),
        infer_type=False,
        help='Left and right width of coincidence window for lonehits'
    )
    lonehit_filename = straxen.URLConfig(
        default='28112023_siob2wxxm5_lonehits_in_nv_events_topCW5d9m.hdf5',
        infer_type=False,
        help='This plugin version avoids direct dependency from lone_hits. Instead, exported lone hits in an NV concidence time window are loaded from this file.'
    )
    lonehit_filepath = straxen.URLConfig(
        default='/scratch/midway2/jjakob/exported_data/',
        infer_type=False,
        track = False,
        help='Path where the file is stored should not affect lineage, so set track to False and use a separate config.'
    )

    def setup(self):
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)
        (self.left_window,self.right_window) = self.lonehit_coincidence_window
        self.to_pe = self.gain_model
        self.path_to_lonehit_file = os.path.join(self.lonehit_filepath, self.lonehit_filename)
        stored_lonehits = pd.read_hdf(self.path_to_lonehit_file,key='data')
        dtype=strax.time_dt_fields + [(('Channel number of lone hit', 'channel'), np.int16), (('Area of lone hit', 'area'), np.float32)]
        self.all_lone_hits = np.zeros(len(stored_lonehits),dtype=dtype)
        for key in stored_lonehits.keys():
            self.all_lone_hits[key]=stored_lonehits[key]
        self.all_lone_hits=self.all_lone_hits[self.all_lone_hits['time'].argsort()]

    def infer_dtype(self):
        dtype = super().infer_dtype() 
        dtype += [('lonehit_area', np.float32, 'Area of lone hit [PE]')]
        dtype += [(("Total number of triggering NV peaks in event",'nv_n_triggering'), np.int32)] 
        dtype += [(("Absolute time of lone hit [ns]",'lonehit_time'), np.int64)]
        dtype += [(("PMT number of lone hit",'lonehit_channel'), np.int16)]
        dtype += [(("Number of lonehits in coincidence window",'n_lonehits'), np.int32)]
        dtype += [(("Drift time from lone hit [ns]",'lonehit_drift_time'), np.int64)]
        dtype += [(("Center time of triggering NV peak [ns]",'abs_nv_center_time'), np.int64)]
        return dtype


    def get_largest_hit(self, hits):
        '''Get all hits that are in the nv event trigger window and return the largest'''
        largest_hit_index = np.argsort(hits['area'])[-1]
        return hits[largest_hit_index]


    def fill_events(self, result_buffer, events, split_hits, split_peaks):
        """Loop over the events and peaks within that event"""
        for event_i, current_event in enumerate(events):
            hits_in_event_i = split_hits[event_i]
            peaks_in_event_i = split_peaks[event_i]
            # remove all S2s that are before the trigger since with a longer event window we otherwise might have S2s before the trigger
            mask_s2s_before_trigger = (peaks_in_event_i['type']==2)&(peaks_in_event_i['time']<current_event['abs_nv_event_time'])
            peaks_in_event_i=peaks_in_event_i[~mask_s2s_before_trigger]

            n_peaks = len(peaks_in_event_i)
            n_s2_peaks = np.sum(peaks_in_event_i['type']==2)
            result_buffer[event_i]['n_peaks'] = n_peaks
            if not n_s2_peaks:
                continue #...instead of: raise ValueError(f'No peaks within event?\n{events[event_i]}')
            self.fill_result_i(result_buffer[event_i], peaks_in_event_i)
            #We did the normal peak search at this point so we should have an S2, otherwise we would have "continue"d already.
            #Don't explicitly care about what is happening with the peaks, just check if lone_hit is in window and if so store its information
            mask_lonehits_in_time_coincidence = hits_in_event_i['time'] < (current_event['abs_nv_event_time'] + self.right_window)
            mask_lonehits_in_time_coincidence &= hits_in_event_i['time'] > (current_event['abs_nv_event_time'] - self.left_window)
            lonehits_in_coincidence_window = hits_in_event_i[mask_lonehits_in_time_coincidence]
            if len(lonehits_in_coincidence_window):
                largest_hit = self.get_largest_hit(lonehits_in_coincidence_window)
                converted_area = self.to_pe[largest_hit['channel']] * largest_hit['area']
                result_buffer[event_i]['lonehit_area'] = converted_area
                result_buffer[event_i]['lonehit_time'] = largest_hit['time']
                result_buffer[event_i]['lonehit_channel'] = largest_hit['channel']
                result_buffer[event_i]['n_lonehits'] = len(lonehits_in_coincidence_window)
                result_buffer[event_i]['lonehit_drift_time'] = result_buffer[event_i]['s2_center_time'] - result_buffer[event_i]['lonehit_time']
                if np.isnan(result_buffer[event_i]['drift_time']): 
                    # a bit unconsistent but useful for S2width cut: if no S1 was found (== drift time is nan),
                    # replace the drift time with the lone hit drift time
                    result_buffer[event_i]['drift_time'] = result_buffer[event_i]['lonehit_drift_time']
            result_buffer[event_i]['abs_nv_center_time'] = current_event['abs_nv_event_time']
            result_buffer[event_i]['nv_n_triggering'] = current_event['n_triggering']

                    
    def compute(self, event_candidates, peaks):
        lone_hits=self.all_lone_hits
        result = np.zeros(len(event_candidates), dtype=self.dtype)
        self.set_nan_defaults(result)
        result['n_lonehits'] = 0
        
        lone_hits = lone_hits[self.to_pe[lone_hits['channel']] != 0]
        split_hits = strax.split_by_containment(lone_hits, event_candidates)
        if np.sum([len(hits_in_event_i) for hits_in_event_i in split_hits]) == 0:
            warnings.warn(f"No hits found in any event. Check if for {self.run_id} lone_hits are provided in {self.path_to_lonehit_file}.")
        split_peaks = strax.split_by_containment(peaks, event_candidates)

        result['time'] = event_candidates['time']
        result['endtime'] = event_candidates['endtime']
        result['event_number'] = event_candidates['event_number']

        self.fill_events(result, event_candidates, split_hits, split_peaks)
        
        result = result[result['s2_center_time']>0] # is it problematic that the event number field is now not continously increasing?
        
        return result