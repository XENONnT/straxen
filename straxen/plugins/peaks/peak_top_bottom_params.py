import numpy as np
import strax
import straxen
from warnings import warn
export, __all__ = strax.exporter()

@export
class PeakTopBottomParams(strax.Plugin):
    """
    Pluging that computes timing characteristics of top and bottom waveforms 
    based on waveforms stored at peak level
    """
    depends_on = ('peaks', 'peak_basics')
    provides = 'peak_top_bottom_params'
    __version__ = '0.0.0'

    def infer_dtype(self):
        dtype = []
        peak_basics_fields = self.deps['peak_basics'].dtype.fields
        self.arrs = ['top', 'bot']
        for arr_ in self.arrs:
            dtype+=[((f'Central time for {arr_} PMTs [ ns ]',
                      f'center_time_{arr_}'),
                      peak_basics_fields['center_time'][0])]
            dtype+=[((f'Time between 10% and 50% area quantiles for {arr_} PMTs [ns]',
                      f'rise_time_{arr_}'),
                      peak_basics_fields['rise_time'][0])]
            dtype+=[((f'Width (in ns) of the central 50% area of the peak for {arr_} PMTs',
                      f'range_50p_area_{arr_}'),
                      peak_basics_fields['range_50p_area'][0])]
            dtype+=[((f'Width (in ns) of the central 90% area of the peak for {arr_} PMTs',
                      f'range_90p_area_{arr_}'),
                      peak_basics_fields['range_90p_area'][0])]
        dtype += strax.time_fields
        return dtype

    def compute(self, peaks):
        result = np.zeros(peaks.shape, dtype=self.dtype)
        peak_dtype = self.deps['peaks'].dtype
        for arr_ in self.arrs:
            fpeaks_ = np.zeros(peaks.shape[0], dtype=peak_dtype)
            if arr_ == 'top':
                fpeaks_['data']=peaks['data_top']
                fpeaks_['area']=peaks['area']*peaks[f'area_fraction_top']
            elif arr_ == 'bot':
                fpeaks_['data']=(peaks['data']-peaks['data_top'])
                fpeaks_['area']=peaks['area']*(1.-peaks['area_fraction_top'])
            elif arr_ == 'tot':
                # This one is ony
                fpeaks_['data']=peaks[f'{type_}_data']
                fpeaks_['area']=peaks[f'{type_}_area']
            else:
                raise RuntimeError(f'Received unknown array type : '+ arr_)
            fpeaks_['length']=peaks[f'length']
            fpeaks_['dt']=peaks[f'dt']
            mask=(fpeaks_['area']>0)
            # computing center times
            with np.errstate(divide='ignore', invalid='ignore'):
                recalc_ctime = np.sum(fpeaks_['data']*(np.arange(0, fpeaks_['data'].shape[1])), axis=1 )
                recalc_ctime/=fpeaks_['area']
                recalc_ctime*=fpeaks_['dt']
                recalc_ctime[~mask]=0.0
            result[f'center_time_{arr_}']=peaks['time']
            result[f'center_time_{arr_}'][mask]+=recalc_ctime[mask].astype(int)
            # computing widths times
            strax.compute_widths(fpeaks_)
            result[f'rise_time_{arr_}'][:]=np.nan
            result[f'rise_time_{arr_}'][mask]= -fpeaks_['area_decile_from_midpoint'][mask][:, 1]
            result[f'range_50p_area_{arr_}'][:]=np.nan
            result[f'range_50p_area_{arr_}'][mask] = fpeaks_['width'][mask][:, 5]
            result[f'range_90p_area_{arr_}'][:]=np.nan
            result[f'range_90p_area_{arr_}'][mask] = fpeaks_['width'][mask][:, 9]
        result['time'], result['endtime'] = peaks['time'], strax.endtime(peaks)

        return result
