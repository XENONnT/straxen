import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import configparser as cp
import time
import datetime 

import strax
import straxen


def get_scalingspectrum(data_led, data_noise, channels = np.arange(0, 494, 1), bad_channels=None):
        ''' 
        Function that subtract out the contribution of the noise to the amplitude spectrum. 
        Then it scale down the off-time amplitude spectrum such that the total counts below 
        the (3-7) ADC count threshold is the same as in the LED spectrum.

        Args:
            1. data_led: signal amplitude array from a PMT(s). The array should have channel variable.
            2. data_noise: noise amplitude array from a PMT(s). The array should have channel variable.
            3. channels: list of PMTs n° to analize.
        Note:
            1. the fraction of SPE signals with amplitude below a threshold of (3-10) ADC counts is assume to be small.
            2. the spectrum also contains contributions of 2 or more photoelectrons. From the scaling down factor 
               of the noise s, assuming a Poisson distribution of photoelectrons we estimate that the average number of 
               photoelectrons (occupancy) in the LED run was lambda = -ln(s) = 0.566.
            3. the fraction of events with 2 or more photoelectrons is then 1-exp(-lambda)(1+lambda) = 0.111. The contribution 
               of 2 or more photoelectrons leads to a slight over-estimate in the acceptances calculated.
        Output:
            1. SPE: array with histograms info.
        '''
        if bad_channels == None:
            bad_channels = [ ]

        datatype = [('channel', np.int16),
                    ('spectrum led', object),   ('bins led', object),
                    ('spectrum noise', object), ('bins noise', object),
                    ('spectrum noise scaled 2 bin', object), ('occupancy 2 bin', np.float32),
                    ('spectrum noise scaled 3 bin', object), ('occupancy 3 bin', np.float32),
                    ('spectrum noise scaled 4 bin', object), ('occupancy 4 bin', np.float32),
                    ('spectrum noise scaled 5 bin', object), ('occupancy 5 bin', np.float32),
                    ('spectrum noise scaled 6 bin', object), ('occupancy 6 bin', np.float32)]

        SPE = np.zeros((len(channels)), dtype = datatype)

        for j, ch in enumerate(channels):
            if ch in bad_channels:
                    SPE[j]['channel']        = ch
                    SPE[j]['spectrum led']   = np.NaN
                    SPE[j]['bins led']       = np.NaN
                    SPE[j]['spectrum noise'] = np.NaN
                    SPE[j]['bins noise']     = np.NaN
                    for i in range(12,17):                   
                        SPE[j]['spectrum noise scaled '+str(i-10)+' bin'] = np.NaN
                        SPE[j]['occupancy '+str(i-10)+' bin'] = np.NaN
            else:
                arr_led   = data_led[data_led['channel'] == ch]
                arr_noise = data_noise[data_noise['channel'] == ch]

                led, bins_led     = np.histogram(arr_led['amplitude_led'], bins=np.arange(-10, 500, 1))
                noise, bins_noise = np.histogram(arr_noise['amplitude_noise'], bins=np.arange(-10, 500, 1))

                SPE[j]['channel']        = ch
                SPE[j]['spectrum led']   = led
                SPE[j]['bins led']       = bins_led
                SPE[j]['spectrum noise'] = noise
                SPE[j]['bins noise']     = bins_noise

                for i in range(12,17):
                    scaling_coeff = led[:i+1].sum()/noise[:i+1].sum()
                    SPE[j]['spectrum noise scaled '+str(i-10)+' bin'] = noise*scaling_coeff
                    SPE[j]['occupancy '+str(i-10)+' bin'] = -np.log(scaling_coeff)

        return SPE

def get_speacceptance(data_spe, data_noise, channels = np.arange(0, 494, 1), bad_channels=None):
        ''' 
        Function that compute SPE acceptance. 

        Args:
            1. data_led: signal amplitude array from a PMT(s). The array should have channel variable.
            2. data_noise: noise amplitude array from a PMT(s). The array should have channel variable.
            3. channels: list of PMTs n° to analize.
        Note:
            1. the acceptance as a function of amplitude (threshold) is defined as the fraction of 
               noise-subtracted single photoelectron spectrum above that amplitude.
        Output:
            1. SPE_acceptance: array with histograms info.
        '''
        if bad_channels == None:
            bad_channels = [ ]

        datatype = [('channel', np.int16),
                    ('Acceptance @ 15 ADC 2 bin', np.float32), ('Threshold for 0.9 acceptance 2 bin', np.float32),
                    ('SPE acceptance 2 bin', object), ('bins center SPE acceptance 2 bin', object),
                    ('noise-subtracted spectrum 2 bin', object), ('error of noise-subtracted spectrum 2 bin', object),
                    ('Acceptance @ 15 ADC 3 bin', np.float32), ('Threshold for 0.9 acceptance 3 bin', np.float32),
                    ('SPE acceptance 3 bin', object), ('bins center SPE acceptance 3 bin', object),
                    ('noise-subtracted spectrum 3 bin', object), ('error of noise-subtracted spectrum 3 bin', object),
                    ('Acceptance @ 15 ADC 4 bin', np.float32), ('Threshold for 0.9 acceptance 4 bin', np.float32),
                    ('SPE acceptance 4 bin', object), ('bins center SPE acceptance 4 bin', object),
                    ('noise-subtracted spectrum 4 bin', object), ('error of noise-subtracted spectrum 4 bin', object),
                    ('Acceptance @ 15 ADC 5 bin', np.float32), ('Threshold for 0.9 acceptance 5 bin', np.float32),
                    ('SPE acceptance 5 bin', object), ('bins center SPE acceptance 5 bin', object),
                    ('noise-subtracted spectrum 5 bin', object), ('error of noise-subtracted spectrum 5 bin', object),
                    ('Acceptance @ 15 ADC 6 bin', np.float32), ('Threshold for 0.9 acceptance 6 bin', np.float32),
                    ('SPE acceptance 6 bin', object), ('bins center SPE acceptance 6 bin', object),
                    ('noise-subtracted spectrum 6 bin', object), ('error of noise-subtracted spectrum 6 bin', object)]

        SPE = get_scalingspectrum(data_spe, data_noise, channels, bad_channels)
        SPE_acceptance = np.zeros((len(channels)), dtype = datatype)

        for j, ch in enumerate(channels):
            if ch in bad_channels:
                for i in range(12,17):
                    SPE_acceptance[j]['channel'] = ch
                    SPE_acceptance[j]['Acceptance @ 15 ADC '+str(i-10)+' bin'] = np.NaN
                    SPE_acceptance[j]['Threshold for 0.9 acceptance '+str(i-10)+' bin'] = np.NaN
                    SPE_acceptance[j]['SPE acceptance '+str(i-10)+' bin'] = np.NaN
                    SPE_acceptance[j]['bins center SPE acceptance '+str(i-10)+' bin'] = np.NaN
                    SPE_acceptance[j]['noise-subtracted spectrum '+str(i-10)+' bin'] = np.NaN
                    SPE_acceptance[j]['error of noise-subtracted spectrum '+str(i-10)+' bin'] = np.NaN
            else:            
                arr = SPE[SPE['channel'] == ch]
                SPE_acceptance[j]['channel'] = ch

                for i in range(12,17):
                    try:
                        diff = arr['spectrum led'][0] - arr['spectrum noise scaled '+str(i-10)+' bin'][0]
                        sigma_diff = np.sqrt(arr['spectrum led'][0] + arr['spectrum noise scaled '+str(i-10)+' bin'][0])

                        res =  1. - np.cumsum(diff)/np.sum(diff)
                        res = np.clip(res, 0, 1)
                        x = arr['bins led'][0]
                        x_center = 0.5 * (x[1:] + x[:-1])
                        pos_15ADC = np.where(x==15)
                        pos_acc90 = np.where(res<0.9)

                        SPE_acceptance[j]['Acceptance @ 15 ADC '+str(i-10)+' bin'] = res[pos_15ADC[0]][0]
                        SPE_acceptance[j]['Threshold for 0.9 acceptance '+str(i-10)+' bin'] = x_center[pos_acc90[0][0]]
                        SPE_acceptance[j]['SPE acceptance '+str(i-10)+' bin'] = res
                        SPE_acceptance[j]['bins center SPE acceptance '+str(i-10)+' bin'] = x_center
                        SPE_acceptance[j]['noise-subtracted spectrum '+str(i-10)+' bin'] = diff
                        SPE_acceptance[j]['error of noise-subtracted spectrum '+str(i-10)+' bin'] = sigma_diff
                    except:
                        warnings.warn('Something went wrong in PMT: '+str(ch))
                        for i in range(12,17):
                            SPE_acceptance[j]['channel'] = ch
                            SPE_acceptance[j]['Acceptance @ 15 ADC '+str(i-10)+' bin'] = np.NaN
                            SPE_acceptance[j]['Threshold for 0.9 acceptance '+str(i-10)+' bin'] = np.NaN
                            SPE_acceptance[j]['SPE acceptance '+str(i-10)+' bin'] = np.NaN
                            SPE_acceptance[j]['bins center SPE acceptance '+str(i-10)+' bin'] = np.NaN
                            SPE_acceptance[j]['noise-subtracted spectrum '+str(i-10)+' bin'] = np.NaN
                            SPE_acceptance[j]['error of noise-subtracted spectrum '+str(i-10)+' bin'] = np.NaN

        return SPE_acceptance

def get_moments(data, channels=np.arange(0,494, 1), bad_channels=None):
    ''' 
    Function that compute first and second moments (mean and variance) of data distribution.

    Args:
        1. data: PMT(s) array. The array should have channel variable.
        2. channels: list of PMTs n° to analize.

    Note:
        1. this function is used for gain calculation.

    Output:
        1. moments: array with mean and variance.

    ''' 
    if bad_channels == None:
        bad_channels = [ ]

    datatype = [('channel', np.int16), 
                ('mean', np.float32), 
                ('variance', np.float32)]
    moments =  np.zeros(len(channels), dtype = datatype)

    for i, ch in enumerate(channels):
        if ch in bad_channels:
            moments[i]['channel']  = ch
            moments[i]['mean']     = np.NaN
            moments[i]['variance'] = np.NaN
        else:
            area = data[data['channel']==ch]['area']
            hist, bins  = np.histogram(area, range=(-1000, 20000), bins=21000)
            mids = 0.5*(bins[1:] + bins[:-1])

            moments[i]['channel']  = ch
            moments[i]['mean']     = np.average(mids, weights=hist)
            moments[i]['variance'] = np.average((mids - np.average(mids, weights=hist))**2, weights=hist)

    return moments

def get_occupancy(data_s, data_b, channels=np.arange(0,494, 1), bad_channels=None, order=10):
    ''' 
    Function that occupancy (poisson parameter) of data distribution.

    #TODO: comments the important steps 

    Args:
        1. data_s: signal PMT(s) array.
        2. data_b: noise PMT(s) array.
        3. channels: list of PMTs n° to analize.

    Note:

    Output:
        1. Occupancy:
            - estimated occupancy: 
            - estimated occupancy error:
            - iteration: 
            - occupancy: 
            - occupancy error:
            - threshold: 
            - occupancy smooth: 
            - scaling factor:
            - entries: 
    '''  
    if bad_channels == None:
        bad_channels = [ ]

    datatype = [('channel', np.int16), 
                ('estimated occupancy', np.float32), 
                ('estimated occupancy error', np.float32),
                ('iteration', np.float32),
                ('occupancy', object), ('occupancy error', object), ('threshold', object), 
                ('occupancy smooth', object), ('scaling factor', np.float32), 
                ('entries', np.float32)]

    Occupancy =  np.zeros(len(channels), dtype = datatype)

    for i, ch in enumerate(channels):
        if ch in bad_channels:
            Occupancy[i]['channel']                   = ch
            Occupancy[i]['estimated occupancy']       = np.NaN
            Occupancy[i]['estimated occupancy error'] = np.NaN
            Occupancy[i]['iteration']                 = np.NaN
            Occupancy[i]['occupancy']                 = np.NaN
            Occupancy[i]['occupancy error']           = np.NaN
            Occupancy[i]['threshold']                 = np.NaN
            Occupancy[i]['occupancy smooth']          = np.NaN
            Occupancy[i]['scaling factor']            = np.NaN
            Occupancy[i]['entries']                   = np.NaN
        else:

            moments_s = get_moments(data=data_s, channels=[ch])
            moments_b = get_moments(data=data_b, channels=[ch])
            area_s = data_s[data_s['channel']==ch]['area']
            signal, bins     = np.histogram(area_s, range=(-1000, 20000), bins=21000)

            area_b = data_b[data_b['channel']==ch]['area']
            background, bins = np.histogram(area_b, range=(-1000, 20000), bins=21000)

            E_s = moments_s[moments_s['channel']==ch]['mean'][0]

            if E_s > 0:
                threshold = -35
            else:
                threshold = 0

            ini_threshold = threshold    
            end_threshold = 50
            start = np.digitize(-1000, bins)

            occupancy     = []
            occupancy_err = []
            thr           = []

            tot_entries_b = np.sum(background)

            while threshold < end_threshold: 
                bin_threshold = np.digitize(threshold, bins)

                Ab = np.sum(background[start:bin_threshold])
                As = np.sum(signal[start:bin_threshold])

                if Ab > 0 and As > 0:
                    f = Ab/tot_entries_b

                    l = -np.log(As/Ab)

                    l_err = np.sqrt((np.exp(l) + 1. - 2.*(Ab/tot_entries_b))/Ab)

                    if l_err/l <= 0.05:
                        occupancy.append(l)
                        occupancy_err.append(l_err)
                        thr.append(threshold)
                threshold += 1

            num = len(occupancy) - 1
            if num % 2 == 0:
                num = num - 1
            try:
                occupancy_smooth = savgol_filter(occupancy, num, order)
                dummy = occupancy_smooth.argsort()[::-1]
                for idx in range(0, len(dummy)):
                    if occupancy_err[dummy[idx]]/occupancy[dummy[idx]] < 0.01:           
                        estimated_occupancy = occupancy[dummy[idx]]
                        estimated_occupancy_err = occupancy_err[dummy[idx]]
                        itr = dummy[idx]
                        break
                    else:
                        estimated_occupancy = 0
                        estimated_occupancy_err = 0
                        itr = 0

                Occupancy[i]['channel']                   = ch
                Occupancy[i]['estimated occupancy']       = estimated_occupancy
                Occupancy[i]['estimated occupancy error'] = estimated_occupancy_err
                Occupancy[i]['iteration']                 = itr
                Occupancy[i]['occupancy']                 = occupancy
                Occupancy[i]['occupancy error']           = occupancy_err
                Occupancy[i]['threshold']                 = thr
                Occupancy[i]['occupancy smooth']          = occupancy_smooth
                Occupancy[i]['scaling factor']            = f
                Occupancy[i]['entries']                   = tot_entries_b

            except ValueError:
                try:
                    occupancy_smooth = savgol_filter(occupancy, num, num-1)
                    dummy = occupancy_smooth.argsort()[::-1]
                    for idx in range(0, len(dummy)):
                        if occupancy_err[dummy[idx]]/occupancy[dummy[idx]] < 0.02:           
                            estimated_occupancy = occupancy[dummy[idx]]
                            estimated_occupancy_err = occupancy_err[dummy[idx]]
                            itr = dummy[idx]
                            break
                        else:
                            estimated_occupancy = 0
                            estimated_occupancy_err = 0
                            itr = 0

                    Occupancy[i]['channel']                   = ch
                    Occupancy[i]['estimated occupancy']       = estimated_occupancy
                    Occupancy[i]['estimated occupancy error'] = estimated_occupancy_err
                    Occupancy[i]['iteration']                 = itr
                    Occupancy[i]['occupancy']                 = occupancy
                    Occupancy[i]['occupancy error']           = occupancy_err
                    Occupancy[i]['threshold']                 = thr
                    Occupancy[i]['occupancy smooth']          = occupancy_smooth
                    Occupancy[i]['scaling factor']            = f
                    Occupancy[i]['entries']                   = tot_entries_b

                except:
                    Occupancy[i]['channel']                   = ch
                    Occupancy[i]['estimated occupancy']       = np.NaN
                    Occupancy[i]['estimated occupancy error'] = np.NaN
                    Occupancy[i]['iteration']                 = np.NaN
                    Occupancy[i]['occupancy']                 = np.NaN
                    Occupancy[i]['occupancy error']           = np.NaN
                    Occupancy[i]['threshold']                 = np.NaN
                    Occupancy[i]['occupancy smooth']          = np.NaN
                    Occupancy[i]['scaling factor']            = np.NaN
                    Occupancy[i]['entries']                   = np.NaN

    return Occupancy

def get_gainconversion(mu):
    ''' 
    Function that computed the gain from SPE ADC count.

    Args:
        1. mu: SPE ADC signal.

    Note:

    Output:
        1. gain: multiplication PMT factor.
    '''  
    Z = 50
    A = 10
    e = 1.6021766208e-19
    f = 1e8
    r = 2.25/16384

    gain = mu*r/(Z*A*f*e*1e6)

    return gain

def get_gain(self, data_s, data_b, channels=np.arange(0,494, 1), bad_channels=None, order=10):
    ''' 
    Function that computed the gain from the occupancy.

    #TODO: comments the important steps 

    Args:
        1. data_s: signal PMT(s) array.
        2. data_b: noise PMT(s) array.
        3. channels: list of PMTs n° to analize.

    Note:

    Output:
        1. Gain: multiplication PMT factor.
    '''
    if bad_channels == None:
        bad_channels = [ ]
    datatype = [('channel', np.int16), 
                ('gain', np.float32), 
                ('gain error', np.float32),
                ('gain statistics error', np.float32),
                ('gain sistematics error', np.float32)]

    Gain = np.zeros(len(channels), dtype = datatype)

    for i, ch in enumerate(channels):
        if ch in bad_channels:
            Gain[i]['channel']    = ch
            Gain[i]['gain']       = np.NaN
            Gain[i]['gain error'] = np.NaN
            Gain[i]['gain statistics error'] = np.NaN
            Gain[i]['gain sistematics error'] = np.NaN
        else:
            moments_s = get_moments(data=data_s, channels=[ch])
            moments_b = get_moments(data=data_b, channels=[ch])
            Occupancy = get_occupancy(data_s=data_s, data_b=data_b, channels=[ch])

            E_s = moments_s[moments_s['channel']==ch]['mean']
            V_s = moments_s[moments_s['channel']==ch]['variance']
            E_b = moments_b[moments_b['channel']==ch]['mean']
            V_b = moments_b[moments_b['channel']==ch]['variance']
            occupancy     = Occupancy[Occupancy['channel']==ch]['estimated occupancy']
            occupancy_err = Occupancy[Occupancy['channel']==ch]['estimated occupancy error']
            tot_N = Occupancy[Occupancy['channel']==ch]['entries']
            f_b   = Occupancy[Occupancy['channel']==ch]['scaling factor']

            if occupancy >= 0:
                EPsi = (E_s - E_b)/occupancy
                VPsi = (V_s - V_b)/occupancy - EPsi**2
                EPsi_stat_err = (occupancy*(EPsi**2 + VPsi) + 2.*V_b)/(tot_N*occupancy**2) + (EPsi*EPsi*(np.exp(occupancy) + 1. - 2.*f_b))/(f_b*tot_N*occupancy**2)
                EPsi_sys_err = (E_s - E_b)*occupancy_err/(occupancy**2)

                gain     = get_gainconversion(EPsi)
                gain_err = get_gainconversion(np.sqrt(EPsi_stat_err)) + get_gainconversion(EPsi_sys_err)

                Gain[i]['channel']    = ch
                Gain[i]['gain']       = gain
                Gain[i]['gain error'] = gain_err
                Gain[i]['gain statistics error']  = get_gainconversion(np.sqrt(EPsi_stat_err)) 
                Gain[i]['gain sistematics error'] = get_gainconversion(EPsi_sys_err)
            else:
                Gain[i]['channel']    = ch
                Gain[i]['gain']       = np.NaN
                Gain[i]['gain error'] = np.NaN
                Gain[i]['gain statistics error']  = np.NaN
                Gain[i]['gain sistematics error'] = np.NaN
    return Gain

def get_gainfunction(V, A, k):
    ''' 
    Gain function.

    Args:
        1. V: voltage value
        2. A k: parameter for gain function

    Note:

    Output:
        1. gain value

    '''  
    n = 12
    #k = 0.672
    #a = 0.018
    return A * V**(k*n)

'''
TODO: comment the mini analysis
'''

@straxen.mini_analysis(requires=('led_calibration',))
def xenonnt_occupancy(context, run_id, led_cal, seconds_range,
                      channels=np.arange(0,494, 1), bad_channels=None, order=10,
                      save_gain=False, folder_gain=None):
    
    if bad_channels == None:
        bad_channels = [ ]
        
    data_led, data_noise = led_cal[led_cal['run_id']==run_id[0]], led_cal[led_cal['run_id']==run_id[1]]
    
    ### Selection of those events where at least 10 PMTs have seen something
    good_time = [ ]
    bad_time  = [ ]
    for chunk in st.get_iter(run_id[0], 'led_calibration', max_workers=20, seconds_range=seconds_range,
                             keep_columns = ('channel', 'area', 'time')):
        led_data_ = chunk.data
        timestamp = np.unique(led_data_['time'])
        time_led_good = led_data_[led_data_['area']>50]['time']
        for t in timestamp:
            if len(np.where(time_led_good==t)[0]) > 10:
                good_time.append(t)
            else:
                bad_time.append(t)
    print('Check if good and bad are different: ', good_time == bad_time)
    print('Events with at least on PMT with area > 50: ', len(good_time))
    print('Events without at least on PMT with area < 50: ', len(bad_time))
    mask = np.isin(data_led['time'], good_time)
    data_led = data_led[mask]
    
    ### Fill led_area and noise_area for gain anc occupancy estimation
    led_area = np.zeros(len(data_led), dtype = np.dtype([('channel', 'int16'), ('area', 'float32')]))
    led_area['channel'] = data_led['channel']
    led_area['area'] = data_led['area']
    
    noise_area = np.empty(0, dtype = np.dtype([('channel', 'int16'), ('area', 'float32')]))
    for ch in channels:
        PMT_area = np.zeros(len(led_area[led_area['channel']==ch]), 
                            dtype = np.dtype([('channel', 'int16'), ('area', 'float32')]))
        idx = len(led_area[led_area['channel']==ch]['channel'])
        PMT_area['channel'] = led_area[led_area['channel']==ch]['channel']
        PMT_area['area']    = data_noise[data_noise['channel']==ch]['area'][:idx]
        noise_area = np.concatenate((noise_area, PMT_area)) 
        
    ### Check about LED and NOISE staticts! Warning is rised if the difference is grater than +/- 5 percent.
    for ch in channels:
        if ch not in bad_channels:
            len1 = len(led_area[led_area['channel']==ch]['area'])
            len2 = len(noise_area[noise_area['channel']==ch]['area'])
            if len1 !=0:
                diff = ((len1 - len2)/len1)*100
                if np.abs(diff)>5:
                    warnings.warn('In PMT ch '+str(ch)+' noise statistics is different about '+str(diff)+' percent compare to LED statistics.')
            else:
                bad_channels.append(ch)
                print('len(data_led) of channel %s is zero'%(str(ch)))
    
    occupancy = get_occupancy(led_area, noise_area, channels=channels, bad_channels=bad_channels, order=order)
    
    ### Save gain and occupancy
    if save_gain==True:
        save_in = folder_gain+'occupancy_timeselection_'+run[0]+'_'+run[1]
        print('Gain: ', save_in)
        np.savez(save_in, x=occupancy)
        
    return occupancy

@straxen.mini_analysis(requires=('led_calibration',))
def xenonnt_gain(context, run_id, led_cal, seconds_range,
                 channels=np.arange(0,494, 1), bad_channels=None, order=10,
                 save_gain=False, folder_gain=None):
    
    if bad_channels == None:
        bad_channels = [ ]
        
    data_led, data_noise = led_cal[led_cal['run_id']==run_id[0]], led_cal[led_cal['run_id']==run_id[1]]
    
    ### Selection of those events where at least 10 PMTs have seen something
    good_time = [ ]
    bad_time  = [ ]
    for chunk in st.get_iter(run_id[0], 'led_calibration', max_workers=20, seconds_range=seconds_range,
                             keep_columns = ('channel', 'area', 'time')):
        led_data_ = chunk.data
        timestamp = np.unique(led_data_['time'])
        time_led_good = led_data_[led_data_['area']>50]['time']
        for t in timestamp:
            if len(np.where(time_led_good==t)[0]) > 10:
                good_time.append(t)
            else:
                bad_time.append(t)
    print('Check if good and bad are different: ', good_time == bad_time)
    print('Events with at least on PMT with area > 50: ', len(good_time))
    print('Events without at least on PMT with area < 50: ', len(bad_time))
    mask = np.isin(data_led['time'], good_time)
    data_led = data_led[mask]
    
    ### Fill led_area and noise_area for gain anc occupancy estimation
    led_area = np.zeros(len(data_led), dtype = np.dtype([('channel', 'int16'), ('area', 'float32')]))
    led_area['channel'] = data_led['channel']
    led_area['area'] = data_led['area']
    
    noise_area = np.empty(0, dtype = np.dtype([('channel', 'int16'), ('area', 'float32')]))
    for ch in channels:
        PMT_area = np.zeros(len(led_area[led_area['channel']==ch]), 
                            dtype = np.dtype([('channel', 'int16'), ('area', 'float32')]))
        idx = len(led_area[led_area['channel']==ch]['channel'])
        PMT_area['channel'] = led_area[led_area['channel']==ch]['channel']
        PMT_area['area']    = data_noise[data_noise['channel']==ch]['area'][:idx]
        noise_area = np.concatenate((noise_area, PMT_area))
    
    ### Saving histogram for database
    led   = [ ]
    noise = [ ]
    ADC   = np.arange(-100, 21000, 1)
    for ch in channels:
        noise_hist, bins = np.histogram(noise_area[noise_area['channel']==ch]['area'], bins=ADC)
        led_hist, bins   = np.histogram(led_area[led_area['channel']==ch]['area'], bins=ADC)
        led.append({'channel': ch, 'data': led_hist})
        noise.append({'channel': ch, 'data': noise_hist})
    
    ### Check about LED and NOISE staticts! Warning is rised if the difference is grater than +/- 5 percent.
    for ch in channels:
        if ch not in bad_channels:
            len1 = len(led_area[led_area['channel']==ch]['area'])
            len2 = len(noise_area[noise_area['channel']==ch]['area'])
            if len1 !=0:
                diff = ((len1 - len2)/len1)*100
                if np.abs(diff)>5:
                    warnings.warn('In PMT ch '+str(ch)+' noise statistics is different about '+str(diff)+' percent compare to LED statistics.')
            else:
                bad_channels.append(ch)
                print('len(data_led) of channel %s is zero'%(str(ch)))
    
    gain = get_gain(led_area, noise_area, channels=channels, bad_channels=bad_channels, order=order)
    
    ### Save gain and occupancy
    if save_gain==True:
        save_in = folder_gain+'/histogram/area_'+run[0]+'_'+run[1]
        print('Histogram for db: ', save_in)
        np.savez(save_in, x=led, y=noise)
        
        save_in = folder_gain+'gain_timeselection_'+run[0]+'_'+run[1]
        print('Gain: ', save_in)
        np.savez(save_in, x=gain)
    
    return gain

@straxen.mini_analysis(requires=('led_calibration',))
def xenonnt_spespectrum(context, run_id, led_cal, seconds_range,
                        channels=np.arange(0,494, 1), bad_channels=None,
                        save_spe=False, folder_spe=None):
    
    if bad_channels == None:
        bad_channels = [ ]
        
    data_led, data_noise = led_cal[led_cal['run_id']==run_id[0]], led_cal[led_cal['run_id']==run_id[1]]
    
    ### Selection of those events where at least 10 PMTs have seen something
    good_time = [ ]
    bad_time  = [ ]
    for chunk in st.get_iter(run_id[0], 'led_calibration', max_workers=20, seconds_range=seconds_range,
                             keep_columns = ('channel', 'area', 'time')):
        led_data_ = chunk.data
        timestamp = np.unique(led_data_['time'])
        time_led_good = led_data_[led_data_['area']>50]['time']
        for t in timestamp:
            if len(np.where(time_led_good==t)[0]) > 10:
                good_time.append(t)
            else:
                bad_time.append(t)
    print('Check if good and bad are different: ', good_time == bad_time)
    print('Events with at least on PMT with area > 50: ', len(good_time))
    print('Events without at least on PMT with area < 50: ', len(bad_time))
    mask = np.isin(data_led['time'], good_time)
    data_led = data_led[mask]
    
    ### Fill led_area and noise_area for gain anc occupancy estimation
    led_amplitude = np.zeros(len(data_led), dtype = np.dtype([('channel', 'int16'),('amplitude_led', 'float32')]))
    led_amplitude['channel'] = data_led['channel']
    led_amplitude['amplitude_led'] = data_led['amplitude_led']
    
    noise_amplitude = np.empty(0, dtype = np.dtype([('channel', 'int16'), ('amplitude_noise', 'float32')]))
    for ch in channels:
        PMT_amplitude = np.zeros(len(led_amplitude[led_amplitude['channel']==ch]),
                                 dtype = np.dtype([('channel', 'int16'), ('amplitude_noise', 'float32')]))
        idx = len(led_amplitude[led_amplitude['channel']==ch]['channel'])
        PMT_amplitude['channel']            = led_amplitude[led_amplitude['channel']==ch]['channel']
        PMT_amplitude['amplitude_noise']    = data_noise[data_noise['channel']==ch]['amplitude_noise'][:idx]
        noise_amplitude = np.concatenate((noise_amplitude, PMT_amplitude)) 
        
    ### Check about LED and NOISE staticts! Warning is rised if the difference is grater than +/- 5 percent.
    for ch in channels:
        if ch not in bad_ch:
            len1 = len(led_amplitude[led_amplitude['channel']==ch]['amplitude_led'])
            len2 = len(noise_amplitude[noise_amplitude['channel']==ch]['amplitude_noise'])
            if len1 !=0:
                diff = ((len1 - len2)/len1)*100
                if np.abs(diff)>5:
                    warnings.warn('In PMT ch '+str(ch)+' noise statistics is different about '+str(diff)+' percent compare to LED statistics.')
            else:
                bad_ch.append(ch)
                print('len(data_led) of channel %s is zero'%(str(ch)))
    spe_spectrum = get_scalingspectrum(led_amplitude, noise_amplitude, 
                                       channels=channels, bad_channels=bad_channels)
    if save_spe==True:
        save_in = folder_spe+'spespectrum_timeselection_'+run[0]+'_'+run[1]
        print('Histogram for db: ', save_in)
        np.savez(save_in, x=spe_spectrum)
        
    return spe_spectrum


@straxen.mini_analysis(requires=('led_calibration',))
def xenonnt_speacceptance(context, run_id, led_cal, seconds_range,
                          channels=np.arange(0,494, 1), bad_channels=None,
                          save_spe=False, folder_spe=None):
    
    if bad_channels == None:
        bad_channels = [ ]
        
    data_led, data_noise = led_cal[led_cal['run_id']==run_id[0]], led_cal[led_cal['run_id']==run_id[1]]
    
    ### Selection of those events where at least 10 PMTs have seen something
    good_time = [ ]
    bad_time  = [ ]
    for chunk in st.get_iter(run_id[0], 'led_calibration', max_workers=20, seconds_range=seconds_range,
                             keep_columns = ('channel', 'area', 'time')):
        led_data_ = chunk.data
        timestamp = np.unique(led_data_['time'])
        time_led_good = led_data_[led_data_['area']>50]['time']
        for t in timestamp:
            if len(np.where(time_led_good==t)[0]) > 10:
                good_time.append(t)
            else:
                bad_time.append(t)
    print('Check if good and bad are different: ', good_time == bad_time)
    print('Events with at least on PMT with area > 50: ', len(good_time))
    print('Events without at least on PMT with area < 50: ', len(bad_time))
    mask = np.isin(data_led['time'], good_time)
    data_led = data_led[mask]
    
    ### Fill led_area and noise_area for gain anc occupancy estimation
    led_amplitude = np.zeros(len(data_led), dtype = np.dtype([('channel', 'int16'),('amplitude_led', 'float32')]))
    led_amplitude['channel'] = data_led['channel']
    led_amplitude['amplitude_led'] = data_led['amplitude_led']
    
    noise_amplitude = np.empty(0, dtype = np.dtype([('channel', 'int16'), ('amplitude_noise', 'float32')]))
    for ch in channels:
        PMT_amplitude = np.zeros(len(led_amplitude[led_amplitude['channel']==ch]),
                                 dtype = np.dtype([('channel', 'int16'), ('amplitude_noise', 'float32')]))
        idx = len(led_amplitude[led_amplitude['channel']==ch]['channel'])
        PMT_amplitude['channel']            = led_amplitude[led_amplitude['channel']==ch]['channel']
        PMT_amplitude['amplitude_noise']    = data_noise[data_noise['channel']==ch]['amplitude_noise'][:idx]
        noise_amplitude = np.concatenate((noise_amplitude, PMT_amplitude)) 
        
    ### Check about LED and NOISE staticts! Warning is rised if the difference is grater than +/- 5 percent.
    for ch in channels:
        if ch not in bad_ch:
            len1 = len(led_amplitude[led_amplitude['channel']==ch]['amplitude_led'])
            len2 = len(noise_amplitude[noise_amplitude['channel']==ch]['amplitude_noise'])
            if len1 !=0:
                diff = ((len1 - len2)/len1)*100
                if np.abs(diff)>5:
                    warnings.warn('In PMT ch '+str(ch)+' noise statistics is different about '+str(diff)+' percent compare to LED statistics.')
            else:
                bad_ch.append(ch)
                print('len(data_led) of channel %s is zero'%(str(ch)))
                
    spe_acceptance = get_speacceptance(led_amplitude, noise_amplitude, 
                                       channels=channels, bad_channels=bad_channels)
    
    if save_spe==True:
        save_in = folder_spe+'spespectrum_timeselection_'+run[0]+'_'+run[1]
        print('Histogram for db: ', save_in)
        np.savez(save_in, x=spe_acceptance)
        
    return spe_acceptance