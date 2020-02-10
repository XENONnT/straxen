'''
Dear nT analyser, 
if you want to complain please contact: chiara@physik.uzh.ch, gvolta@physik.uzh.ch
'''

import strax
import numpy as np
import pandas as pd
from tqdm import tqdm
import numba
import resource
from numba import njit

#export, __all__ = strax.exporter()
# QUESTION: what does 'export, __all__ = strax.exporter()' do?
# ANSWER: [fill in]

@strax.takes_config(
    strax.Option('led_window',
                 default=(150, 275),
                 help="Window (samples) where we expect the signal in LED calibration"),
    strax.Option('noise_window',
                 default=(0, 125),
                 help="Window (samples) to analysis the noise"),
    strax.Option('channel_list',
                 default=(0,248),
                 help="Three different light level for XENON1T: (0,36), (37,126), (127,248). Defalt value: all the PMTs")
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
    
    __version__ = '0.1.1'
    depends_on = ('raw_records',)
    # Options below copied from other plugins, need to be reviewed by an expert
    data_kind = 'led_cal' 
    compressor = 'zstd'
    parallel = 'process'
    rechunk_on_save = False
    
    dtype = [('area', np.int32, 'Area averaged in integration windows'),
             ('amplitude_led', np.int32, 'Amplitude in LED window'),
             ('amplitude_noise', np.int32, 'Amplitude in off LED window'),
             ('channel', np.int16, 'Channel'),
             ('time', np.int64, 'Start time of the interval (ns since unix epoch)'),
             ('dt', np.int16, 'Time resolution in ns'),
             ('length', np.int32, 'Length of the interval in samples')
            ]
    
    def compute(self, raw_records):
        #print('gigabytes: ', (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)/1000000)
        r = raw_records[(raw_records['channel'] >= self.config['channel_list'][0])&(raw_records['channel'] <= self.config['channel_list'][1])]
        # TODO: to change during nT commissioning or add in configuration options
        temp = np.zeros(len(r), dtype=self.dtype)
        
        temp['channel'] = r['channel']
        temp['time'] = r['time']
        temp['dt'] = r['dt']
        temp['length'] = r['length']
        
        on, off = get_amplitude(r, self.config['led_window'], self.config['noise_window'])
        temp['amplitude_led'] = on['amplitude_led']
        temp['amplitude_noise'] = off['amplitude_noise']

        area = get_area(r, self.config['led_window'])
        temp['area'] = area['area']

        
        return temp

# QUESTIONS: can some nice functions of numba.njit be used? What does @export mean?
# ANSWERS: [fill in]

#@njit
def get_amplitude(raw_records, led_window, noise_window):
    '''
    Needed for the SPE computation.
    Take the maximum in two different regions, where there is the signal and where there is not.
    '''
    on = np.zeros((len(raw_records)), dtype=[('channel','int16'),('amplitude_led', '<i4')])
    off = np.zeros((len(raw_records)), dtype=[('channel','int16'),('amplitude_noise', '<i4')])
    for i, r in enumerate(raw_records):
        on['amplitude_led'][i] = np.max(r['data'][led_window[0]:led_window[1]])
        on['channel'][i] = r['channel']
        off['amplitude_noise'][i] = np.max(r['data'][noise_window[0]:noise_window[1]])
        off['channel'][i] = r['channel']
    return on, off

def get_area(raw_records, led_window):
    '''
    Needed for the gain computation.
    Sum the data in the defined window to get the area.
    This is done in 6 integration window and it returns the average area.
    '''
    left = led_window[0]
    end_pos = [led_window[1]+2*i for i in range(6)]

    Area = np.zeros((len(raw_records)), dtype=[('channel','int16'),('area','float32')])
    for right in end_pos:
        Area['area'] += raw_records['data'][:, left:right].sum(axis=1)
    Area['channel'] = raw_records['channel']
    Area['area'] = Area['area']/float(len(end_pos))
        
    return Area
