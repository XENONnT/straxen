import strax
import straxen

import numpy as np

import numba
export, __all__ = strax.exporter()

# define new data type for afterpulse data
dtype_ap = [(('Channel/PMT number','channel'),
              '<i2'),                                         # = PMT number
             (('Time resolution in ns', 'dt'),
               '<i2'),                                        # sample width = 10ns
             (('Start time of the interval (ns since unix epoch)', 'time'),
               '<i8'),                                        # start time of hit
             (('Length of the interval in samples', 'length'), 
               '<i4'),                                        # length of hit
             (('Integral in ADC x samples', 'area'),
               '<i4'),                                        # area of hit
             (('Pulse area in PE', 'area_pe'),
               '<f4'),                                        # area converted to PE
             (('Index of sample in record in which hit starts', 'left'),
               '<i2'),                                        # start sample of hit
             (('Index of sample in record in which the hit area suceeds 10% of its total area', 'sample_10pc_area'),
               '<i2'),                                        # 10% area quantile sample of hit
             (('Index of sample in record in which the hit area suceeds 50% of its total area', 'sample_50pc_area'),
               '<i2'),                                        # 50% area quantile sample of hit        
             (('Index of sample in record in which the hit height suceeds 50% of maximum height', 'sample_50pc_height'),
               '<i2'),                                        # sample where hit height first exceeds 10% of maxium height
             (('Index of sample in record of hit maximum', 'max'),
               '<i2'),                                        # sample of hit maximum
             (('Index of first sample in record just beyond hit (exclusive bound)', 'right'),
               '<i2'),                                        # first sample after last sample of hit
             (('Height of hit in ADC counts', 'height'),
               '<i4'),                                        # height of hit in ADC counts
             (('Height of hit in PE', 'height_pe'),
               '<f4'),                                        # height of hit converted to PE equivalent
             (('Delay of hit w.r.t. LED hit in same WF, in samples', 'tdelay'),
               '<i2'),                                        # time delay of hit w.r.t to corresponding LED hit in the same WF in samples
             (('Internal (temporary) index of fragment in which hit was found', 'record_i'),
               '<i4')                                         # record index in which hit was found
            ]


@export
@strax.takes_config(
    strax.Option('gain_model',
                 help='PMT gain model. Specify as (model_type, model_config)'),
    strax.Option('n_tpc_pmts',
                 type=int,
                 help="Number of PMTs in TPC"),
    strax.Option('LED_window_left',
                 default=50,
                 help='Left boundary of sample range for LED pulse integration'),
    strax.Option('LED_window_right',
                 default=90,
                 help='Right boundary of sample range for LED pulse integration'),
    strax.Option('baseline_samples',
                 default=40,
                 help='Number of samples to use at the start of the pulse to determine the baseline'),
    )
class LEDAfterpulses(strax.Plugin):
    
    __version__ = '0.1.2'
    depends_on = 'raw_records'
    data_kind = 'afterpulses'
    provides = 'afterpulses'
    compressor = 'zstd'
    parallel = 'process'
    rechunk_on_save = True
    
    dtype = dtype_ap
    
    def setup(self):
        
        self.to_pe = straxen.get_to_pe(self.run_id, self.config['gain_model'], n_pmts=self.config['n_tpc_pmts'])
#         print('setup:\n   plugin version = ', self.__version__,
#               #'\n   hit_threshold = ', self.config['hit_threshold'],
#               '\n   gain_model = ', self.config['gain_model'],
#               '\n   LED_window = ', self.config['LED_window_left'], '-', self.config['LED_window_right'],
#               '\n\n to_pe =', self.to_pe,
#              )

        
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
        
        # find all hits in all WFs
        hits = find_hits_ap(r,
                            LED_window_left=self.config['LED_window_left'],
                            LED_window_right=self.config['LED_window_right'],
                           )
        
        # copy hits to result and convert hit area to PE
        result = np.zeros(len(hits), dtype=self.dtype)
        for name in hits.dtype.names:
            result[name] = hits[name]        
        result['area_pe'] = result['area'] * self.to_pe[result['channel']]
        result['height_pe'] = result['height'] * self.to_pe[result['channel']]
        
        return result


    
# new data type for hits in AP data
hit_ap_dtype = [(('Channel/PMT number', 'channel'), '<i2'),
             (('Time resolution in ns', 'dt'), '<i2'),
             (('Start time of the interval (ns since unix epoch)', 'time'), '<i8'),
             (('Length of the interval in samples', 'length'), '<i4'),
             (('Integral of hit in ADC counts x samples', 'area'), '<i4'),
             (('Index of sample in record in which hit starts', 'left'), '<i2'),
             (('Index of sample in record in which the hit area suceeds 10% of its total area', 'sample_10pc_area'), '<i2'),
             (('Index of sample in record in which the hit area suceeds 50% of its total area', 'sample_50pc_area'), '<i2'),
             (('Index of sample in record in which the hit height suceeds 50% of maximum height', 'sample_50pc_height'), '<i2'),
             (('Index of sample in record of hit maximum', 'max'), '<i2'),
             (('Index of first sample in record just beyond hit (exclusive bound)', 'right'), '<i2'),
             (('Height of hit in ADC counts', 'height'), '<i4'),
             (('Time delay w.r.t. LED hit', 'tdelay'), '<i2'),
             (('Internal (temporary) index of fragment in which hit was found', 'record_i'), '<i4')
            ]
            
    
# strax hitfinder modified for AP data
#   - all samples above threshold in LED window are combined into one LED hit
#   - added timing parameters for hits
#   - gives delay time of all hits in WF w.r.t. LED hit


@export
@strax.growing_result(hit_ap_dtype, chunk_size=int(1e2))
@numba.jit(nopython=True, nogil=True, cache=True)
def find_hits_ap(records, LED_window_left, LED_window_right, _result_buffer=None):

    buffer = _result_buffer
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])
    offset = 0

    for record_i, r in enumerate(records):
        in_interval = False
        hit_start = -1
        area = height = t_LED = 0
        
        LED_hit = False
        
        # get hit finding threshold: 3 times the baseline rms (standard dev)
        threshold = r['baseline_rms'] * 3

        # start hitfinder at beginning of LED window
        for i in range(LED_window_left, samples_per_record):
            
            x = r['data'][i]
            above_threshold = x > threshold
        
            if (i>=LED_window_left) & (i<=LED_window_right):
                # start of LED integration
                if above_threshold:
                    if area == 0:
                        # start of LED hit (first sample above threshold in LED window)
                        hit_start = i
                    area += x
                    height = max(height, x)
                    if height == x:
                        hit_max = i
                    
                    # end of LED hit (last sample above threshold in LED window)
                    hit_end = i
                        
                    if i == LED_window_right:
                        # end of LED window
                        in_interval = True
                        LED_hit = True
                    else:
                        continue
                else:
                    if (i == LED_window_right):
                        # end of LED window
                        if area == 0: #no LED pulse found, go to next record
                            break
                        in_interval = True
                        LED_hit = True
                    continue

            # continue after LED window (same as strax hitfinder)        
            if not in_interval and above_threshold:
                # Start of a hit
                in_interval = True
                hit_start = i
                
                
            if in_interval:
                if not above_threshold:
                    # Hit ends at the start of this sample
                    #if not LED_hit: # (LED hit end is determined already)
                    hit_end = i
                    in_interval = False
                    
                else:
                    area += x
                    height = max(height, x)
                    if height == x:
                        # the sample at which hit is at maximum
                        hit_max = i

                    if i == samples_per_record - 1:
                        # Hit ends at the *end* of this sample
                        # (because the record ends)
                        hit_end = i + 1
                        in_interval = False

                if not in_interval:
                    # Hit is done, add it to the result
                    if hit_end == hit_start:
                        #print(r['time'], r['channel'], hit_start)
                        raise ValueError(
                            "Caught attempt to save zero-length hit!")
                    
                    if hit_end == hit_start + 1:
                        # ignore hits with only 1 sample above threshold
                        continue
                    
                    res = buffer[offset]
                    res['left'] = hit_start
                    res['right'] = hit_end
                    res['time'] = r['time'] + hit_start * r['dt']
                    # Note right bound is exclusive, no + 1 here:
                    res['length'] = hit_end - hit_start
                    res['dt'] = r['dt']
                    res['channel'] = r['channel']
                    res['record_i'] = record_i
                    res['max'] = hit_max

                    # Store areas and height. 
                    baseline_fpart = r['baseline'] % 1
                    area += res['length'] * baseline_fpart
                    res['area'] = area
                    res['height'] = height + baseline_fpart
                    

                    ### Additional hit timing parameters

                    # Get area quantiles (first sample where integrated area of hit is > x % total area)
                    data = r['data'][hit_start:hit_end]
                    area_sum = 0
                    q10 = False
                    for data_i, d in enumerate(data):
                        area_sum += d
                        
                        #if area_sum < 0.1*area:
                        #    continue
                        #else:
                        #    res['sample_10pc_area'] = hit_start + data_i
                        #    break
                        
                        if area_sum > 0.1 * area:
                            if not q10:
                                q10 = True
                                res['sample_10pc_area'] = hit_start + data_i
                            if area_sum > 0.5 * area:
                                res['sample_50pc_area'] = hit_start + data_i
                                break
                        
                        
                    
                    # Get sample index where 50% of height is exceeded
                    for data_i, d in enumerate(data):
                        if d < 0.5*(height + baseline_fpart):
                            continue
                        else:
                            res['sample_50pc_height'] = hit_start + data_i
                            break

                    # Get time of LED hit
                    if LED_hit:
                        t_LED = res['sample_10pc_area']
                        LED_hit = False

                    # store time difference (in samples) between hit and LED hit in same WF
                    res['tdelay'] = res['sample_10pc_area'] - t_LED

                    # reset for next hit
                    area = height = 0
                    

                    # Yield buffer to caller if needed
                    offset += 1
                    if offset == len(buffer):
                        yield offset
                        offset = 0
                        
    yield offset
