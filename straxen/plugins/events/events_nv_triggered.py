import numpy as np
import strax
import straxen
import numba

@strax.takes_config(
    strax.Option('cut_area_boundary_nv_nr', default=(39.0, 99.23), type=(tuple, list),
                 help='Area boundary for selection in neutron veto'
                 ),
    strax.Option(name='max_drift_length',
                 default=straxen.tpc_z, type=(int, float),
                 help='Total length of the TPC from the bottom of gate to the '
                      'top of cathode wires [cm]',
                 ),
    strax.Option('left_nv_event_extension', default=int(0.25e6), type=(int, float),
                 help='Extend events this many ns to the left from each '
                      'triggering peak. This extension is added to the maximum '
                      'drift time',
                 ),
    strax.Option('right_nv_event_extension', default=int(0.25e6), type=(int, float),
                 help='Extend events this many ns to the right from each '
                      'triggering peak',
                 ),
    strax.Option('cut_min_n_contributing_pmt_nv', default=4, type=int,
                 help='Minimal number of contributing PMTs'
                 ),
    strax.Option('cut_center_time_lower_boundary_nv', default=(-39.3, 0.0504, 44.5), type=(tuple, list),
                 help='Parameters for the boundary function.'
                 ),
    strax.Option('cut_center_time_upper_boundary_nv', default=(31.8, 0.0136, 72.3), type=(tuple, list),
                 help='Parameters for the boundary function.'
                 ),
    )


class nVS1Events(strax.OverlapWindowPlugin):
    '''Calculates a selection in the neutron veto and uses this as a triggering "peak". About a drift time after this peak is then defined as an event. For safety, also one drift time before the trigger is added to the event such that is covers two full drift times.'''
    depends_on = ['events_nv','events_sync_nv']

    provides = 'events_nv_triggered'
    data_kind = 'event_candidates' # Needs different data kind as length of event_basics will differ after rejecting events with no S2 peak
    __version__ = '0.0.1'
    
    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    dtype = strax.time_fields + [(("Event number in this dataset","event_number"),np.int64)] + [(("Center time of triggering nVeto event",'nv_center_time'), np.float32)] + [(("Absolut nv center time",'abs_nv_event_time'), np.int64)] + [(("nv center time, relative to start time",'nv_event_time'), np.float64)] + [(("Total area of all hitlets in event [PE]",'area'), np.float32)] + [(("Total number of hitlets in event",'n_hits'), np.int32)]+ [(("Total number of triggering nveto peaks in event",'n_triggering'), np.int32)] 
    events_seen = 0
    
    @staticmethod
    def boundary(x, A, lamb, c):
        return A*np.exp(-lamb*x) + c

    def setup(self):
        self.lower_bound, self.upper_bound = self.config['cut_area_boundary_nv_nr']
        self.drift_time_max = int(self.config['max_drift_length'] / self.electron_drift_velocity)
        self.left_extension = self.config['left_nv_event_extension'] + self.drift_time_max
        self.right_extension = self.drift_time_max + self.config['right_nv_event_extension']
    
    def get_window_size(self):
        return 10 * (self.config['left_nv_event_extension']
                     + self.drift_time_max
                     + self.config['right_nv_event_extension'])
    
    def compute(self, events_nv, start, end):
        ##Requirements for triggering neutron veto event stolen from cutax
        _is_triggering = (events_nv['area'] > self.lower_bound)&(events_nv['area']<self.upper_bound)
        _is_triggering &= events_nv['n_contributing_pmt'] >= self.config['cut_min_n_contributing_pmt_nv']
        _is_triggering &= events_nv['center_time'] >= self.boundary(events_nv['area'],*self.config['cut_center_time_lower_boundary_nv'])
        _is_triggering &= events_nv['center_time'] < self.boundary(events_nv['area'],*self.config['cut_center_time_upper_boundary_nv'])
        
        triggers = events_nv[_is_triggering]
        triggers['endtime']=triggers['endtime_sync']
        triggers['time']=triggers['time_sync']
        
        ##Join nearby triggers
        t0, t1 = strax.find_peak_groups(
            triggers,
            gap_threshold=self.left_extension + self.right_extension + 1,
            left_extension= self.left_extension,
            right_extension= self.right_extension)
        
        ##Don't extend beyond the chunck boundaries
        t0 = np.clip(t0, start, end)
        t1 = np.clip(t1, start, end)

        result = np.zeros(len(t0), self.dtype)
        result['time']=t0
        result['endtime']=t1
        result['event_number'] = np.arange(len(result)) + self.events_seen
        self.events_seen += len(result)

        for trigger_i, (t0_i,t1_i) in enumerate(zip(t0,t1)):
            ##If there is more than one trigger in the considered time range, take the first.
            current = triggers[(triggers['endtime']<t1_i)&(triggers['time']>t0_i)]
            assert len(current), 'Time range of event triggered by nVeto does not contain any event in nVeto??'
            result[trigger_i]['n_triggering'] = len(current)
            first_hit = 0
            result[trigger_i]['area']=current[first_hit]['area']
            result[trigger_i]['n_hits']=current[first_hit]['n_hits']
            result[trigger_i]['nv_center_time']=current[first_hit]['center_time']
            result[trigger_i]['abs_nv_event_time']=current[first_hit]['time']+np.int64(current[first_hit]['center_time'])
            result[trigger_i]['nv_event_time'] = np.float64(current[first_hit]['time']-result[trigger_i]['time']) + current[first_hit]['center_time']
        
        return result
