import strax
import straxen
import numpy as np
export, __all__ = strax.exporter()


@export
class EventInfoMS(strax.Plugin):
    """
    Plugin to collect multiple-scatter event observables
    """
    __version__ = '0.0.2'
    depends_on = (
        'event_info',
        'peak_basics', 'peak_per_event', 'peak_corrections', 'peak_positions')
    provides = 'event_ms_naive'
    save_when = strax.SaveWhen.TARGET

    # config options don't double cache things from the resource cache!
    g1 = straxen.URLConfig(
        default='bodega://g1?bodega_version=v2',
        help='S1 gain in PE / photons produced',
    )
    g2 = straxen.URLConfig(
        default='bodega://g2?bodega_version=v2',
        help='S2 gain in PE / electrons produced',
    )
    lxe_w = straxen.URLConfig(
        default=13.7e-3,
        help='LXe work function in quanta/keV'
    )
    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )
    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z, infer_type=False,
        help='Total length of the TPC from the bottom of gate to the '
             'top of cathode wires [cm]')

    ms_window_fac = straxen.URLConfig(
        default=1.01, type=(int, float),
        help='Max drift time window to look for peaks in multiple scatter events'
    )

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (('Sum of S1 areas in event',
              's1_sum'), np.float32),
            (('Corrected S1 area based on average position of S2s in event',
              'cs1_multi'), np.float32),
            (('Corrected S1 area based on average position of S2s in event before time-dep LY correction',
              'cs1_multi_wo_timecorr'), np.float32),
            (('Sum of S2 areas in event',
              's2_sum'), np.float32),
            (('Sum of corrected S2 areas in event',
              'cs2_sum'), np.float32),
            (('Sum of corrected S2 areas in event S2 before elife correction',
              'cs2_wo_timecorr_sum'), np.float32),
            (('Sum of corrected S2 areas in event before SEG/EE and elife corrections',
              'cs2_wo_elifecorr_sum'), np.float32),
            (('Average of S2 area fraction top in event',
              'cs2_area_fraction_top_avg'), np.float32),
            (('Sum of the energy estimates in event',
              'ces_sum'), np.float32),
            (('Sum of the charge estimates in event',
              'e_charge_sum'), np.float32),
            (('Average x position of S2s in event',
              'x_avg'), np.float32),
            (('Average y position of S2s in event',
              'y_avg'), np.float32),
            (('Average observed z position of energy deposits in event',
              'z_obs_avg'), np.float32),
            (('Number of S2s in event',
              'multiplicity'), np.int32),
        ]
        return dtype

    def setup(self):
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)

    def cs1_to_e(self, x):
        return self.lxe_w * x / self.g1

    def cs2_to_e(self, x):
        return self.lxe_w * x / self.g2

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.infer_dtype())

        # Assign peaks features to main S1 and main S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            cond = (sp['type'] == 2) & (sp['drift_time'] > 0)
            cond &= (sp['drift_time'] < self.ms_window_fac * self.drift_time_max) & (sp['cs2'] > 0)
            result['s2_sum'][event_i] = np.nansum(sp[cond]['area'])
            result['cs2_sum'][event_i] = np.nansum(sp[cond]['cs2'])
            result['cs2_wo_timecorr_sum'][event_i] = np.nansum(sp[cond]['cs2_wo_timecorr'])
            result['cs2_wo_elifecorr_sum'][event_i] = np.nansum(sp[cond]['cs2_wo_elifecorr'])         
            result['s1_sum'][event_i] = np.nansum(sp[sp['type'] == 1]['area'])

            if np.sum(sp[cond]['cs2']) > 0: 
                result['cs1_multi_wo_timecorr'][event_i] = event['s1_area'] * np.average(
                    sp[cond]['s1_xyz_correction_factor'], weights=sp[cond]['cs2'])
                result['cs1_multi'][event_i] = result['cs1_multi_wo_timecorr'][event_i] * np.average(
                    sp[cond]['s1_rel_light_yield_correction_factor'], weights=sp[cond]['cs2'])
                result['x_avg'][event_i] = np.average(sp[cond]['x'], weights=sp[cond]['cs2'])
                result['y_avg'][event_i] = np.average(sp[cond]['y'], weights=sp[cond]['cs2'])
                result['z_obs_avg'][event_i] = np.average(sp[cond]['z_obs_ms'], weights=sp[cond]['cs2'])
                result['cs2_area_fraction_top_avg'][event_i] = np.average(
                    sp[cond]['cs2_area_fraction_top'], weights=sp[cond]['cs2'])   
                result['multiplicity'][event_i] = len(sp[cond]['area'])

        el = self.cs1_to_e(result['cs1_multi'])
        ec = self.cs2_to_e(result['cs2_sum'])
        result['ces_sum'] = el + ec
        result['e_charge_sum'] = ec
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)
        return result
