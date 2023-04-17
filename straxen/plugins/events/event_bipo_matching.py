import strax
import numpy as np
import numba
import straxen


export, __all__ = strax.exporter()

@export
class BiPo214Matching(strax.Plugin):
    
    depends_on=('bi_po_variables', )   
    provides = 'bi_po_214_matching'

    __version__ = "2.1.4"
    
    rechunk_on_save=False
    
    def infer_dtype(self): 
        
        dtype = strax.time_fields + [
                (f's2_bi_match', np.int,
                 f'Index of the S2 reconstructed as a Bismuth peak'),
                (f's2_po_match', np.int,
                 f'Index of the S2 reconstructed as a Polonium peak')]
        
        return dtype

    def setup(self):
        
        self.tol = 3000 # 2 mus tolerance window
        
    def compute(self,events):
        
        result = np.ones(len(events), dtype=self.dtype)
        result['time'] = events['time']
        result['endtime'] = events['endtime']

        # Calculate the delta_t between the two S1s
        dt = events['s1_center_time_0'] - events['s1_center_time_1']  

        L = []
        for n in range(10):
            L.append(str(n))

        combs = list(itertools.combinations(L,2))

        ds1_u = (dt + self.tol)
        ds1_l = (dt - self.tol)
        for i in np.where(ds1_l < self.tol):
            ds1_l[i] = 0

        flag = np.repeat('-1_-1', len(dt))
        overall_cross_mask = np.full(len(dt), False)

        # try every possible combination    
        for comb in combs:
    
            # compute dt for this combination
            dtcomb = events['s2_center_time_'+comb[1]] - events['s2_center_time_'+comb[0]]            
            dtcomb = np.abs(dtcomb)
            
            # mask true if match
            mask = (ds1_l < dtcomb) & (ds1_u > dtcomb) 
            
            #mask &= (events['s2_area_fraction_top_'+comb[0]] > .55)
            #mask &= (events['s2_area_fraction_top_'+comb[1]] > .55)

            # check if we found a match for something that was already matched
            # if two combinations are possible discard the event
            cross_mask = mask & (flag != '-1_-1')
            overall_cross_mask |= cross_mask
            
            flag = np.where(mask , comb[0]+'_'+comb[1], flag)

        flag = np.where(overall_cross_mask , '-2_-2', flag)
        
        # This part is very badly coded but ok..
        
        flag_bi = []
        flag_po = []
        
        for i, f in enumerate(flag):
            c = list(map(int, f.split('_')))
            if c[0] >= 0:
                if events['s2_center_time_%i'%c[1]][i] - events['s2_center_time_%i'%c[0]][i] < 0:
                    flag_bi.append(c[1])
                    flag_po.append(c[0])
                else:
                    flag_bi.append(c[0])
                    flag_po.append(c[1])
            else:
                    flag_bi.append(c[0])
                    flag_po.append(c[1])       
        

        result['s2_bi_match'] = flag_bi
        result['s2_po_match'] = flag_po
        
        return result