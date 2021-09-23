import strax
import straxen

import numpy as np

from straxen.get_corrections import is_cmt_option

import numba
export, __all__ = strax.exporter()


# define new data type for afterpulse data
dtype_ap = [(('Channel/PMT number', 'channel'),
              '<i2'),
            (('Time resolution in ns', 'dt'),
              '<i2'),
            (('Start time of the interval (ns since unix epoch)', 'time'),
              '<i8'),
            (('Length of the interval in samples', 'length'),
              '<i4'),
            (('Integral in ADC x samples', 'area'),
              '<i4'),
            (('Pulse area in PE', 'area_pe'),
              '<f4'),
            (('Sample index in which hit starts', 'left'),
              '<i2'),
            (('Sample index in which hit area succeeds 10% of total area', 'sample_10pc_area'),
              '<i2'),
            (('Sample index in which hit area succeeds 50% of total area', 'sample_50pc_area'),
              '<i2'),
            (('Sample index of hit maximum', 'max'),
              '<i2'),
            (('Index of first sample in record just beyond hit (exclusive bound)', 'right'),
              '<i2'),
            (('Height of hit in ADC counts', 'height'),
              '<i4'),
            (('Height of hit in PE', 'height_pe'),
              '<f4'),
            (('Delay of hit w.r.t. LED hit in same WF, in samples', 'tdelay'),
              '<i2'),
            (('Internal (temporary) index of fragment in which hit was found', 'record_i'),
              '<i4'),
            (('Index of sample in record where integration starts',
              'left_integration'),
              np.int16),
            (('Index of first sample beyond integration region',
              'right_integration'),
              np.int16),
            (('ADC threshold applied in order to find hits',
              'threshold'),
              np.float32),
           ]


@export
@strax.takes_config(
                    strax.Option('gain_model',
                                 help='PMT gain model. Specify as (model_type, model_config)',
                                ),
                    strax.Option('n_tpc_pmts',
                                 type=int,
                                 help="Number of PMTs in TPC",
                                ),
                    strax.Option('LED_window_left',
                                 default=50,
                                 help='Left boundary of sample range for LED pulse integration',
                                ),
                    strax.Option('LED_window_right',
                                 default=100,
                                 help='Right boundary of sample range for LED pulse integration',
                                ),
                    strax.Option('baseline_samples',
                                 default=40,
                                 help='Number of samples to use at start of WF to determine the baseline',
                                ),
                    strax.Option('hit_min_amplitude',
                                 track=True,
                                 default=('hit_thresholds_tpc', 'ONLINE', True),
                                 help='Minimum hit amplitude in ADC counts above baseline. '
                                      'Specify as a tuple of length n_tpc_pmts, or a number,'
                                      'or a string like "pmt_commissioning_initial" which means calling'
                                      'hitfinder_thresholds.py'
                                      'or a tuple like (correction=str, version=str, nT=boolean),'
                                      'which means we are using cmt.',
                                ),
                    strax.Option('hit_min_height_over_noise',
                                 default=4,
                                 help='Minimum hit amplitude in numbers of baseline_rms above baseline.'
                                      'Actual threshold used is max(hit_min_amplitude, hit_min_height_over_noise * baseline_rms).',
                                ),
                    strax.Option('hit_left_extension',
                                 default=2,
                                 help='Extend hits by this many samples left',
                                ),
                    strax.Option('hit_right_extension',
                                 default=20,
                                 help='Extend hits by this many samples right',
                                ),
                   )
class LEDAfterpulseProcessing(strax.Plugin):
    
    __version__ = '0.4.0'
    depends_on = 'raw_records'
    data_kind = 'afterpulses'
    provides = 'afterpulses'
    compressor = 'zstd'
    parallel = 'process'
    rechunk_on_save = True
    
    dtype = dtype_ap
    
    def setup(self):
        
        self.to_pe = straxen.get_correction_from_cmt(self.run_id, self.config['gain_model'])
        
        # Check config of `hit_min_amplitude` and define hit thresholds
        # if cmt config
        if is_cmt_option(self.config['hit_min_amplitude']):
            self.hit_thresholds = straxen.get_correction_from_cmt(self.run_id,
                self.config['hit_min_amplitude'])
        # if hitfinder_thresholds config
        elif isinstance(self.config['hit_min_amplitude'], str):
            self.hit_thresholds = straxen.hit_min_amplitude(
                self.config['hit_min_amplitude'])
        else: # int or array
            self.hit_thresholds = self.config['hit_min_amplitude']
            
        print("I'm here :)")
        print(self.dtype)

        
    def compute(self, raw_records):
        
        # Convert everything to the records data type -- adds extra fields.
        records = strax.raw_to_records(raw_records)
        del raw_records
        
        # channel split: only need the TPC PMT channels
        r = records[records['channel'] < self.config['n_tpc_pmts']]
        del records
        
        # calculate baseline and baseline rms
        strax.baseline(r,
                       baseline_samples=self.config['baseline_samples'],
                       flip=True)
 
        # find all hits
        hits = strax.find_hits(r,
                               min_amplitude = self.hit_thresholds,
                               min_height_over_noise = self.config['hit_min_height_over_noise'],
                              )
        
        # sort hits first by record_i, then by time
        hits_sorted = np.sort(hits, order=('record_i', 'time'))
        
        # find LED hits and afterpulses within the same WF
        hits_ap = find_ap(hits_sorted,
                          r,
                          LED_window_left=self.config['LED_window_left'],
                          LED_window_right=self.config['LED_window_right'],
                          hit_left_extension=self.config['hit_left_extension'],
                          hit_right_extension=self.config['hit_right_extension'],
                         )
    
        hits_ap['area_pe'] = hits_ap['area'] * self.to_pe[hits_ap['channel']]
        hits_ap['height_pe'] = hits_ap['height'] * self.to_pe[hits_ap['channel']]
        
        return hits_ap


@export
def find_ap(hits, records, LED_window_left, LED_window_right, hit_left_extension, hit_right_extension):
    buffer = np.zeros(len(hits), dtype=dtype_ap)
    res = _find_ap(hits, records, LED_window_left, LED_window_right, hit_left_extension, hit_right_extension, buffer=buffer)
    return res
            

@numba.jit(nopython=True, nogil=True, cache=True)
def _find_ap(hits, records, LED_window_left, LED_window_right, hit_left_extension, hit_right_extension, buffer=None):
    # hits need to be sorted by record_i, then time!
    
    offset = 0
    
    is_LED = False
    t_LED = None
    
    prev_record_i = hits[0]['record_i']
    record_data = records[prev_record_i]['data']
    record_len = records[prev_record_i]['length']
    baseline_fpart = records[prev_record_i]['baseline'] % 1
    
    for h_i, h in enumerate(hits):
        
        if h['record_i'] > prev_record_i:
            # start of a new record
            is_LED = False
            # only increment buffer if the old one is not empty! this happens when no (LED) hit is found in the previous record
            if not buffer[offset]['time'] == 0:
                offset += 1
            prev_record_i = h['record_i']
            record_data = records[prev_record_i]['data']
            baseline_fpart = records[prev_record_i]['baseline'] % 1
        
        res = buffer[offset]

            
        if h['left'] < LED_window_left:
            # if hit is before LED window: discard
            continue
            
        if (h['left'] >= LED_window_left) & (h['left'] < LED_window_right):
            # hit is in LED window
            if not is_LED:
                # this is the first hit in the LED window
                res['time'] = h['time']
                res['dt'] = h['dt']
                res['channel'] = h['channel']
                res['left'] = h['left']
                res['right'] = h['right']
                res['record_i'] = h['record_i']
                res['threshold'] = h['threshold']
                res['height'] = h['height']
                
                res['left_integration'] =  h['left'] - hit_left_extension
                res['right_integration'] = h['right'] + hit_right_extension

                res['length'] = res['right_integration'] - res['left_integration']
                
                # need to add baseline_fpart to area
                hit_data = record_data[res['left_integration']:res['right_integration']]
                res['area'] = hit_data.sum() + res['length'] * baseline_fpart
                
                res['sample_10pc_area'] = res['left_integration'] + get_sample_area_quantile(hit_data, 0.1, baseline_fpart)
                res['sample_50pc_area'] = res['left_integration'] + get_sample_area_quantile(hit_data, 0.5, baseline_fpart)
                
                res['max'] = res['left_integration'] + hit_data.argmax()
                
                # set the LED time in the current WF
                t_LED = res['sample_10pc_area']
                
                is_LED = True
                
                continue
            
            # more hits in LED window: extend the first (merging all hits in the LED window)
            res['right'] = h['right']
            res['right_integration'] = h['right'] + hit_right_extension

            res['length'] = res['right_integration'] - res['left_integration']
            res['height'] = max(res['height'], h['height'])
            
            hit_data = record_data[res['left_integration']:res['right_integration']]
            res['area'] = hit_data.sum() + res['length'] * baseline_fpart

            res['sample_10pc_area'] = res['left_integration'] + get_sample_area_quantile(hit_data, 0.1, baseline_fpart)
            res['sample_50pc_area'] = res['left_integration'] + get_sample_area_quantile(hit_data, 0.5, baseline_fpart)

            res['max'] = res['left_integration'] + hit_data.argmax()

            t_LED = res['sample_10pc_area']
            
            continue
            
        # Here begins a new hit after the LED window
        
        if (h['left'] >= LED_window_right) and not is_LED:
            # no LED hit found: ignore and go to next hit (until new record begins)
            continue
        
        ## if a hit is completely inside the previous hit's right_extension, then skip it (because it's already included in the previous hit)
        if h['right'] <= res['right_integration']:
            continue
            
        ## if a hit only partly overlaps with the previous hit's right_extension, merge them (extend previous hit by this one)
        if h['left'] <= res['right_integration']:
            
            res['right'] = h['right']
            res['right_integration'] = h['right'] + hit_right_extension
            if res['right_integration'] > record_len:
                res['right_integration'] = record_len
            res['length'] = res['right_integration'] - res['left_integration']
            res['height'] = max(res['height'], h['height'])
            
            hit_data = record_data[res['left_integration']:res['right_integration']]
            res['area'] = hit_data.sum() + res['length'] * baseline_fpart
            
            res['sample_10pc_area'] = res['left_integration'] + get_sample_area_quantile(hit_data, 0.1, baseline_fpart)
            res['sample_50pc_area'] = res['left_integration'] + get_sample_area_quantile(hit_data, 0.5, baseline_fpart)

            res['max'] = res['left_integration'] + hit_data.argmax()
            
            res['tdelay'] = res['sample_10pc_area'] - t_LED
            
            continue
                   
        # an actual new hit increases the buffer index
        offset += 1 
        res = buffer[offset]
        
        res['time'] = h['time']
        res['dt'] = h['dt']
        res['channel'] = h['channel']
        res['left'] = h['left']
        res['right'] = h['right']
        res['record_i'] = h['record_i']
        res['threshold'] = h['threshold']
        res['height'] = h['height']

        res['left_integration'] =  h['left'] - hit_left_extension
        res['right_integration'] = h['right'] + hit_right_extension
        if res['right_integration'] > record_len:
            res['right_integration'] = record_len
        res['length'] = res['right_integration'] - res['left_integration']

        hit_data = record_data[res['left_integration']:res['right_integration']]
        res['area'] = hit_data.sum() + res['length'] * baseline_fpart

        res['sample_10pc_area'] = res['left_integration'] + get_sample_area_quantile(hit_data, 0.1, baseline_fpart)
        res['sample_50pc_area'] = res['left_integration'] + get_sample_area_quantile(hit_data, 0.5, baseline_fpart)

        res['max'] = res['left_integration'] + hit_data.argmax()
        
        res['tdelay'] = res['sample_10pc_area'] - t_LED
    
    return buffer[:offset]


@export
@numba.jit(nopython=True, nogil=True, cache=True)
def get_sample_area_quantile(data, quantile, baseline_fpart):
    '''
    returns first sample index in hit where integrated area of hit is above total area
    '''
    
    area = 0
    area_tot = data.sum() + len(data) * baseline_fpart

    for d_i, d in enumerate(data):
        area += d + baseline_fpart
        if area > (quantile * area_tot):
            return d_i
        if (d_i == len(data)-1):
            # if no quantile was found, something went wrong
            # (negative area due to wrong baseline, caused by real events that by coincidence fall in the first samples of the trigger window)
            #print('no quantile found: set to 0')
            return 0
