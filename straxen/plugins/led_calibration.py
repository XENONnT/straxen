'''
Dear nT analyser, 
if you want to complain please contact: chiara@physik.uzh.ch, gvolta@physik.uzh.ch, kazama@isee.nagoya-u.ac.jp
'''

import strax
import numba
import numpy as np

# This makes sure shorthands for only the necessary functions
# are made available under straxen.[...]
export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('baseline_window',
                 default=(0,50),
                 help="Window (samples) for baseline calculation."),
    strax.Option('led_window',
                 default=(50, 115),
                 help="Window (samples) where we expect the signal in LED calibration"),
    strax.Option('noise_window',
                 default=(120, 185),
                 help="Window (samples) to analysis the noise"),
    strax.Option('channel_list',
                 default=(0,248),
                 help="Three different light level for XENON1T: (0,36), (37,126), (127,248). Defalt value: all the PMTs"))
class LEDCalibration(strax.Plugin):
    """
    Preliminary version, several parameters to set during commisioning.
    LEDCalibration returns: channel, time, dt, lenght, Area, amplitudeLED and amplitudeNOISE.
    The new variables are:
    - Area: Area computed in the given window, averaged over 6 windows that have the same starting sample and different end samples.
    - amplitudeLED: peak amplitude of the LED on run in the given window.
    - amplitudeNOISE: amplitude of the LED on run in a window far from the signal one.
    """
    
    __version__ = '0.1.3'
    depends_on = ('raw_records',)
    data_kind = 'led_cal' 
    compressor = 'zstd'
    parallel = 'process'
    rechunk_on_save = False
    
    dtype = [('area', np.float64, 'Area averaged in integration windows'),
             ('amplitude_led', np.int32, 'Amplitude in LED window'),
             ('amplitude_noise', np.int32, 'Amplitude in off LED window'),
             ('channel', np.int16, 'Channel'),
             ('time', np.int64, 'Start time of the interval (ns since unix epoch)'),
             ('dt', np.int16, 'Time resolution in ns'),
             ('length', np.int32, 'Length of the interval in samples')
            ]
    
    def compute(self, raw_records):
        r = raw_records[(raw_records['channel'] >= self.config['channel_list'][0])&(raw_records['channel'] <= self.config['channel_list'][1])]
        # TODO: to change during nT commissioning or add in configuration options
        temp = np.zeros(len(r), dtype=self.dtype)
        
        temp['channel'] = r['channel']
        temp['time'] = r['time']
        temp['dt'] = r['dt']
        temp['length'] = r['length']
        
        on, off = get_amplitude(r, self.config['led_window'], self.config['noise_window'], self.config['baseline_window'])
        temp['amplitude_led'] = on['amplitude']
        temp['amplitude_noise'] = off['amplitude']

        area = get_area(r, self.config['led_window'], self.config['noise_window'], self.config['baseline_window'])
        temp['area'] = area['area']

        
        return temp

# QUESTIONS: can some nice functions of numba.njit be used? What does @export mean?
# ANSWERS: [fill in]

_on_off_dtype = np.dtype([('channel', 'int16'),
                          ('amplitude', '<i4')])

@numba.njit(nogil=True, cache=True)
def get_amplitude(raw_records, led_window, noise_window, baseline_window):
    '''
    Needed for the SPE computation.
    Take the maximum in two different regions, where there is the signal and where there is not.
    '''
    left_bsl  = baseline_window[0]
    right_bsl = baseline_window[-1]
    bsl_diff = 1.0 * (right_bsl-left_bsl)
    
    on = np.zeros((len(raw_records)), dtype=_on_off_dtype)
    off = np.zeros((len(raw_records)), dtype=_on_off_dtype)
    for i, r in enumerate(raw_records):
        r['data'][:] = np.abs(r['data'] - np.sum(r['data'][left_bsl:right_bsl])/bsl_diff)
        on[i]['amplitude'] = safe_max(r['data'][led_window[0]:led_window[1]])
        on[i]['channel'] = r['channel']
        off[i]['amplitude'] = safe_max(r['data'][noise_window[0]:noise_window[1]])
        off[i]['channel'] = r['channel']
    return on, off


@numba.njit(nogil=True, cache=True)
def safe_max(w):
    if not len(w):
        return 0
    return w.max()


def get_area(raw_records, led_window, noise_window, baseline_window):
    '''
    Needed for the gain computation.
    Sum the data in the defined window to get the area.
    This is done in 6 integration window and it returns the average area.
    '''
    left = led_window[0]
    end_pos = [led_window[1]+2*i for i in range(6)]

    left_bsl  = baseline_window[0]
    right_bsl = baseline_window[-1]
    
    Area = np.zeros((len(raw_records)), dtype=[('channel','int16'),('area','float64')])
    for right in end_pos:
        Area['area'] += raw_records['data'][:, left:right].sum(axis=1)
        Area['area'] -= float(right-left)*raw_records['data'][:, left_bsl:right_bsl].sum(axis=1)/(right_bsl-left_bsl)
    Area['channel'] = raw_records['channel']
    Area['area'] = Area['area']/float(len(end_pos))
        
    return Area
