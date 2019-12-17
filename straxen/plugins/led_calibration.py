'''
Dear nT analyser, 
if you want to complain please contact: chiara@physik.uzh.ch, gvolta@physik.uzh.ch
'''

import strax
import numpy as np
import pandas as pd
from tqdm import tqdm
import numba
from numba import njit

#export, __all__ = strax.exporter()
# QUESTION: what does 'export, __all__ = strax.exporter()' do?
# ANSWER: [fill in]

@strax.takes_config(
    strax.Option('LED_window',
                 default=(125, 250),
                 help="Window (samples) where we expect signal in LED calibration"),
    strax.Option('noise_window',
                 default=(350, 475),
                 help="Window (samples) to analysis the noise")
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
    
    __version__ = '0.0.8'
    print(__version__)
    depends_on = ('raw_records',)
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
    
    def compute(self, raw_records):
        
        r = raw_records[raw_records['channel'] < 248] # TODO: to change during nT commissioning or add in configuration options
        temp = np.zeros(len(r), dtype=self.dtype)
        
        temp['channel'] = r['channel']
        temp['time'] = r['time']
        temp['dt'] = r['dt']
        temp['length'] = r['length']
        
        on, off = get_amplitude(r, self.config['LED_window'], self.config['noise_window'])
        temp['amplitudeLED'] = on
        temp['amplitudeNOISE'] = off

        area = get_area(r, self.config['LED_window'])
        temp['area'] = area['area']
        
        return temp

# QUESTIONS: can some nice functions of numba.njit be used? What does @export mean?
# ANSWERS: [fill in]
    
def get_amplitude(raw_records, LED_window, noise_window):
    '''
    Needed for the SPE computation.
    Take the maximum in two different regions, where there is the signal and where there is not.
    '''
    on = np.zeros(len(raw_records), dtype=[('amplitudeLED', '<i4')])
    off = np.zeros(len(raw_records), dtype=[('amplitudeNOISE', '<i4')])
    #on  = ((np.max(r['data'][LED_window[0]:LED_window[1]])) for r in raw_records)
    #off = ((np.max(r['data'][noise_window[0]:noise_window[1]])) for r in raw_records)
    i = 0
    for r in raw_records:
        on[i] = np.max(r['data'][LED_window[0]:LED_window[1]])
        off[i] = np.max(r['data'][noise_window[0]:noise_window[1]])
        i=i+1
    #    on.append(amp_LED)
    #    off.append(amp_NOISE)
    #on = np.array(on, dtype=[('amplitudeLED', '<i4')])
    #off = np.array(off, dtype=[('amplitudeNOISE', '<i4')])
    return on, off

def get_area(raw_records, LED_window):
    '''
    Needed for the gain computation.
    Sum the data in the defined window to get the area.
    This is done in 6 integration window and it returns the average area.
    '''
    left = LED_window[0]
    end_pos = [LED_window[1]+2*i for i in range(6)]
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
