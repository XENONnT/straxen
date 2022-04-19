import strax
import numpy as np
from straxen import OnlineMonitor
from immutabledict import immutabledict

#Find a way to import hotspot_monitor.py where the plots are made

export, __all__ = strax.exporter()

class OnlineHotspotMonitor(strax.Plugin):
    """
    Plugin to write data needed for the online hotspot monitor to the 
    online-monitor collection in the runs-database. Data that is written by
    this plugin should be small such as to not overload the runs-
    database.

    This plugin takes 'peak_basics' and 'peak_positions_mlp'. Although 
    they are not strictly related, they are aggregated into a single data_type
    in order to minimize the number of documents in the online monitor.

    Produces 'online_hotspot_monitor' with info on the peaks and their
    positions.
    """
    depends_on = ('peak_basics', 'peak_positions_mlp')
    provides = 'online_hotspot_monitor'
    data_kind = 'online_hotspot_monitor'
    __version__ = '0.0.1'
    rechunk_on_save = False #??

    def infer_dtype(self):
        # n_bins_area_width = self.config['area_vs_width_nbins']
        # bounds_area_width = self.config['area_vs_width_bounds']

        # n_bins = self.config['online_peak_monitor_nbins']

        # n_tpc_pmts = self.config['n_tpc_pmts']
        dtype = [
            (('Start time of the chunk', 'time'),
             np.int64),
            (('Peak integral in PE',
            'area'), np.float32),
            (('Reconstructed mlp peak x-position','x_mlp'), 
             np.float32), 
            (('Reconstructed mlp peak y-position','y_mlp'), 
             np.float32), 
            (('Width (in ns) of the central 50% area of the peak',
            'range_50p_area'), np.float32),
            (('Time between 10% and 50% area quantiles [ns]',
            'rise_time'), np.float32),
            (('End time of the chunk', 'endtime'),
             np.int64),
        ]
        print(dtype)
        return dtype

    def compute(self,peaks,start,end):
        # Make results en 1D array
        res = np.zeros(len(peaks), dtype=self.dtype)
        res['time'] = start

        area = peaks['area']
        range_50p_area = peaks['range_50p_area']

        res['x_mlp'] = peaks['x_mlp']
        res['y_mlp'] = peaks['y_mlp']
        res['rise_time'] = peaks['rise_time']
        res['endtime'] = end

        print(res)
        return res


