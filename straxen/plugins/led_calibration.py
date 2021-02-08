'''
Dear nT analyser, 
if you want to complain please contact: chiara@physik.uzh.ch, gvolta@physik.uzh.ch, kazama@isee.nagoya-u.ac.jp
'''
import datetime
import strax
import numba
import numpy as np

# This makes sure shorthands for only the necessary functions
# are made available under straxen.[...]
export, __all__ = strax.exporter()

channel_list = [i for i in range(494)]
@export
@strax.takes_config(
    strax.Option('baseline_window',
                 default=(0,40),
                 help="Window (samples) for baseline calculation."),
    strax.Option('led_window',
                 default=(78, 116),
                 help="Window (samples) where we expect the signal in LED calibration"),
    strax.Option('noise_window',
                 default=(10, 48),
                 help="Window (samples) to analysis the noise"),
    strax.Option('channel_list',
                 default=(tuple(channel_list)),
                 help="List of PMTs. Defalt value: all the PMTs"))

class LEDCalibration(strax.Plugin):
    """
    Preliminary version, several parameters to set during commissioning.
    LEDCalibration returns: channel, time, dt, lenght, Area,
    amplitudeLED and amplitudeNOISE.
    The new variables are:
        - Area: Area computed in the given window, averaged over 6
        windowsthat have the same starting sample and different end
        samples.
        - amplitudeLED: peak amplitude of the LED on run in the given
        window.
        - amplitudeNOISE: amplitude of the LED on run in a window far
         from the signal one.
    """
    
    __version__ = '0.2.3'
    depends_on = ('raw_records',)
    data_kind = 'led_cal' 
    compressor = 'zstd'
    parallel = 'process'
    rechunk_on_save = False

    dtype = [('area', np.float32, 'Area averaged in integration windows'),
             ('amplitude_led', np.float32, 'Amplitude in LED window'),
             ('amplitude_noise', np.float32, 'Amplitude in off LED window'),
             ('channel', np.int16, 'Channel'),
             ('time', np.int64, 'Start time of the interval (ns since unix epoch)'),
             ('dt', np.int16, 'Time resolution in ns'),
             ('length', np.int32, 'Length of the interval in samples')]
    
    def compute(self, raw_records):
        '''
        The data for LED calibration are build for those PMT which belongs to channel list. 
        This is used for the different ligh levels. As defaul value all the PMTs are considered.
        '''
        mask = np.where(np.in1d(raw_records['channel'], self.config['channel_list']))[0]
        rr   = raw_records[mask]
        r    = get_records(rr, baseline_window=self.config['baseline_window'])
        del rr, raw_records

        temp = np.zeros(len(r), dtype=self.dtype)
        strax.copy_to_buffer(r, temp, "_recs_to_temp_led")

        on, off = get_amplitude(r, self.config['led_window'], self.config['noise_window'])
        temp['amplitude_led']   = on['amplitude']
        temp['amplitude_noise'] = off['amplitude']

        area = get_area(r, self.config['led_window'])
        temp['area'] = area['area']
        return temp


def get_records(raw_records, baseline_window):
    """
    Determine baseline as the average of the first baseline_samples
    of each pulse. Subtract the pulse float(data) from baseline.
    """

    record_length = np.shape(raw_records.dtype['data'])[0]

    _dtype = [(('Start time since unix epoch [ns]', 'time'), '<i8'),
              (('Length of the interval in samples', 'length'), '<i4'),
              (('Width of one sample [ns]', 'dt'), '<i2'),
              (('Channel/PMT number', 'channel'), '<i2'),
              (('Length of pulse to which the record belongs (without zero-padding)', 'pulse_length'), '<i4'),
              (('Fragment number in the pulse', 'record_i'), '<i2'),
              (('Waveform data in raw ADC counts', 'data'), 'f4', (record_length,))]

    records = np.zeros(len(raw_records), dtype=_dtype)
    strax.copy_to_buffer(raw_records, records, "_rr_to_r_led")

    mask = np.where((records['record_i'] == 0) & (records['length'] == 160))[0]
    records = records[mask]
    bl = records['data'][:, baseline_window[0]:baseline_window[1]].mean(axis=1)
    records['data'][:, :160] = -1. * (records['data'][:, :160].transpose() - bl[:]).transpose()
    return records


_on_off_dtype = np.dtype([('channel', 'int16'),
                          ('amplitude', 'float32')])


def get_amplitude(records, led_window, noise_window):
    """
    Needed for the SPE computation.
    Take the maximum in two different regions, where there is the signal and where there is not.
    """   
    on = np.zeros((len(records)), dtype=_on_off_dtype)
    off = np.zeros((len(records)), dtype=_on_off_dtype)
    on['amplitude'] = np.max(records['data'][:, led_window[0]:led_window[1]], axis=1)
    on['channel']   = records['channel']
    off['amplitude'] = np.max(records['data'][:, noise_window[0]:noise_window[1]], axis=1)
    off['channel']   = records['channel']
    return on, off

_area_dtype = np.dtype([('channel', 'int16'),
                        ('area', 'float32')])

def get_area(records, led_window):
    """
    Needed for the gain computation.
    Sum the data in the defined window to get the area.
    This is done in 6 integration window and it returns the average area.
    """
    left = led_window[0]
    end_pos = [led_window[1]+2*i for i in range(6)]
  
    Area = np.zeros((len(records)), dtype=_area_dtype)
    for right in end_pos:
        Area['area'] += records['data'][:, left:right].sum(axis=1)
    Area['channel'] = records['channel']
    Area['area']    = Area['area']/float(len(end_pos))
        
    return Area
