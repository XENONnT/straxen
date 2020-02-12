import strax
import straxen

import numpy as np

from straxen.common import get_to_pe
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
             (('Index of sample in record in which the hit height suceeds 50% of maximum height', 'sample_50pc_height'),
               '<i2'),                                        # sample where hit height first exceeds 10% of maxium height
             (('Index of sample in record of hit maximum', 'max'),
               '<i2'),                                        # sample of hit maximum
             (('Index of first sample in record just beyond hit (exclusive bound)', 'right'),
               '<i2'),                                        # first sample after last sample of hit
             (('Height of hit in ADC counts', 'height'),
               '<i4'),                                        # height of pulse
             (('Time delay w.r.t. LED hit', 'tdelay'),
               '<i2'),                                        # time delay of pulse w.r.t to corresponding LED hit in the same WF in samples
             (('Internal (temporary) index of fragment in which hit was found', 'record_i'),
               '<i4')                                         # record index in which hit was found
            ]


@strax.takes_config(
    strax.Option('n_pmts',
                 default=248,
                 help="Number of PMTs in TPC"),
    strax.Option('to_pe_file',
                 default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',
                 help='Link to the to_pe conversion factors'),
    strax.Option('LED_window',
                 default=[140,180],
                 help='Sample range for LED pulse integration'),
    strax.Option('hit_threshold',
                 default=15,
                 help='Hitfinder threshold in ADC counts above baseline'),
    )
class AP(strax.Plugin):
    
    __version__ = '0.0.2.5'
    depends_on = 'raw_records'
    data_kind = 'afterpulses'
    provides = 'afterpulses'
    compressor = 'zstd'
    #compressor = 'lz4'
    parallel = 'process'
    rechunk_on_save = False
    
    dtype = dtype_ap
    
    def setup(self):
        self.to_pe = get_to_pe(self.run_id, self.config['to_pe_file'])
        print('setup:\n   plugin version = ', self.__version__,
              '\n   hit_threshold = ', self.config['hit_threshold'],
              '\n   to_pe_file = ', self.config['to_pe_file'],
              '\n   LED_window = ', self.config['LED_window'],
             )
        
    def compute(self, raw_records):
        
        r = raw_records[raw_records['channel'] < self.config['n_pmts']]
        
        
        #hits = strax.find_hits(r, threshold=self.config['hit_threshold'])
        hits = find_hits_ap(r, threshold=self.config['hit_threshold'], LED_window=self.config['LED_window'])
                
        res = np.zeros(len(hits), dtype=self.dtype)
        for name in hits.dtype.names:
            res[name] = hits[name]        
        
        res['area_pe'] = res['area'] * self.to_pe[res['channel']]
        
        return res


hit_ap_dtype = [(('Channel/PMT number', 'channel'), '<i2'),
             (('Time resolution in ns', 'dt'), '<i2'),
             (('Start time of the interval (ns since unix epoch)', 'time'), '<i8'),
             (('Length of the interval in samples', 'length'), '<i4'),
             (('Integral in ADC x samples', 'area'), '<i4'),
             (('Index of sample in record in which hit starts', 'left'), '<i2'),
             (('Index of sample in record in which the hit area suceeds 10% of its total area', 'sample_10pc_area'), '<i2'),
             (('Index of sample in record in which the hit height suceeds 50% of maximum height', 'sample_50pc_height'), '<i2'),   
             (('Index of sample in record of hit maximum', 'max'), '<i2'),
             (('Index of first sample in record just beyond hit (exclusive bound)', 'right'), '<i2'),
             (('Height of hit in ADC counts', 'height'), '<i4'),
             (('Time delay w.r.t. LED hit', 'tdelay'), '<i2'),
             (('Internal (temporary) index of fragment in which hit was found', 'record_i'), '<i4')
            ]
            

@export
@strax.growing_result(hit_ap_dtype, chunk_size=int(1e2))
@numba.jit(nopython=True, nogil=True, cache=True)
def find_hits_ap(records, threshold, LED_window, _result_buffer=None):

    buffer = _result_buffer
    if not len(records):
        return
    samples_per_record = len(records[0]['data'])
    #LED_window = [140,180]
    #LED_range = list(range(140,180))
    #print(LED_range)
    offset = 0

    for record_i, r in enumerate(records):
        in_interval = False
        hit_start = -1
        area = height = t_LED = 0
        
        LED_hit = False

        for i in range(LED_window[0], samples_per_record):
            # We can't use enumerate over r['data'],
            # numba gives errors if we do.
            # TODO: file issue?
            
            x = r['data'][i]
            above_threshold = x > threshold
        
        
            if (i>=LED_window[0]) & (i<=LED_window[1]):
                # start of LED integration
                if above_threshold:
                    if area==0:
                        # start of LED hit
                        hit_start = i
                    area += x
                    height = max(height, x)
                    if height == x:
                        hit_max = i
                        
                    if i==LED_window[1]:
                        in_interval = True
                        LED_hit = True
                    else:
                        continue
                else:
                    if i==LED_window[1]:
                        in_interval = True
                        LED_hit = True
                    continue
        
        
            if not in_interval and above_threshold:
                # Start of a hit
                in_interval = True
                hit_start = i
                
                
            if in_interval:
                if not above_threshold:
                    # Hit ends at the start of this sample
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
                    # print('saving hit')
                    # Hit is done, add it to the result
                    if hit_end == hit_start:
                        print(r['time'], r['channel'], hit_start)
                        raise ValueError(
                            "Caught attempt to save zero-length hit!")
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
                    
                    # Get 10% area quantile (first sample where integrated area is > 10% total area)
                    data = r['data'][hit_start:hit_end]
                    area_sum = 0
                    for data_i, d in enumerate(data):
                        area_sum += d
                        if area_sum < 0.1*area:
                            continue
                        else:
                            res['sample_10pc_area'] = hit_start + data_i
                            break
                    
                    # Store areas and height.
                    baseline_fpart = r['baseline'] % 1
                    area += res['length'] * baseline_fpart
                    res['area'] = area
                    res['height'] = height + baseline_fpart
                    
                    # Get sample index where 50% of height is exceeded
                    for data_i, d in enumerate(data):
                        if d < 0.5*(height + baseline_fpart):
                            continue
                        else:
                            res['sample_50pc_height'] = hit_start + data_i
                            break
                    
                    if LED_hit:
                        t_LED = res['sample_10pc_area']
                        LED_hit = False
                    
                    res['tdelay'] = res['sample_10pc_area'] - t_LED
                                        
                    area = height = 0
                    

                    # Yield buffer to caller if needed
                    offset += 1
                    if offset == len(buffer):
                        yield offset
                        offset = 0
                        
    yield offset


