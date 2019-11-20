import straxen
import strax
import pandas as pd
from tqdm import tqdm
import numpy as np
import numba
from numba import njit

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('LED_window',
                 default=(125, 250),
                 help="Window (samples) where we expect signal in LED calibration")
)


class LEDCalibration(strax.Plugin):
    
    '''
    Preliminary version, several parameters to set during commisioning.
    LEDCalibration returns: channel, time, dt, lenght, Area, amplitudeLED and amplitudeNOISE.
    The new variables are:
    - Area: Area computed in the given window, averaged over 6 windows that have the same starting sample and different end samples.
    - amplitudeLED: peak amplitude of the LED on run in the given window.
    - amplitudeNOISE: amplitude of the LED on run in a window far from the signal one.
    '''
    
    __version__ = '0.0.5'
    depends_on = 'raw_records'
    # Options below copied from other plugins, need to be reviewed by an expert
    data_kind = 'led_cal_0' 
    compressor = 'zstd'
    parallel = 'process'
    rechunk_on_save = False
    
    dtype = [('area', np.int32, 'Area averaged in integration windows'),
             ('amplitudeLED', np.int32, 'Amplitude in LED window'),
             ('amplitudeNOISE', np.int32, 'Amplitude in off LED window'),
             ('channel', np.int16, 'Channel'),
             ('time', np.int64, 'Start time of the interval (ns since unix epoch)'),
             ('dt', np.int16, 'Time resolution in ns'),
             ('length', np.int32, 'Length of the interval in samples')
            ]
    
    def get_amplitude(raw_records, window):
        
        '''
        Needed for the SPE computation.
        Take the maximum in two different regions, where there is the signal and where there is not.
        '''

        on = []
        off = []
        for r in raw_records:
            amp_LED = np.max(r['data'][window[0]:window[1]])
            amp_NOISE = np.max(r['data'][2*window[0]:2*window[1]])
            on.append(amp_LED)
            off.append(amp_NOISE)
        on = np.array(on, dtype=[('amplitudeLED', '<i4')])
        off = np.array(off, dtype=[('amplitudeNOISE', '<i4')])
        return on, off

    def get_area(raw_records, window):
        
        '''
        Needed for the gain computation.
        Sum the data in the defined window to get the area.
        This is done in 6 integration window and it returns the average area.
        '''
        
        left = window[0]
        end_pos = [window[1]+2*i for i in range(6)]
        n_channel_s = np.arange(0, 249, 1) # TODO: to change during nT commissioning or add in configuration options
        Area = np.array(raw_records[['channel', 'area']],dtype=[('channel','int16'),('area','float32')])

        for n_channel in n_channel_s:
            wf_tmp = raw_records[raw_records['channel'] == n_channel]
            area = 0
            for right in end_pos:
                area += wf_tmp['data'][:,left:right].sum(axis=1)

        mask = np.where(Area['channel'] == n_channel)[0]
        Area['area'][mask] = area.astype(np.float)/6.
        return Area

    
    def compute(self, raw_records):
        
        r = raw_records[raw_records['channel'] < 248] # TODO: to change during nT commissioning or add in configuration options
        temp = np.zeros(len(r), dtype=self.dtype)
        
        temp['channel'] = r['channel']
        temp['time'] = r['time']
        temp['dt'] = r['dt']
        temp['length'] = r['length']
        
        on, off = get_amplitude(r, self.config['LED_window'])
        temp['amplitudeLED'] = on
        temp['amplitudeNOISE'] = off

        area = get_area(r, self.config['LED_window'])
        temp['area'] = area['area']
        
        return temp
