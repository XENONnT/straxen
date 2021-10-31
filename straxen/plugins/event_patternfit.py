import strax
import straxen
from straxen.get_corrections import get_correction_from_cmt
import numpy as np
import numba
from straxen.numbafied_scipy import numba_gammaln, numba_betainc
from scipy.special import loggamma

export, __all__ = strax.exporter()

@export
@strax.takes_config(
    strax.Option('s1_optical_map', help='S1 (x, y, z) optical/pattern map.',
                 default='XENONnT_s1_xyz_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl'),
    strax.Option('s2_optical_map', help='S2 (x, y) optical/pattern map.',
                 default='XENONnT_s2_xy_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl'),
    strax.Option('s1_aft_map', help='Date drive S1 area fraction top map.',
                 default='s1_aft_dd_xyz_XENONnT_Kr83m_41500eV_19Oct2021.json'),
    strax.Option('mean_pe_per_photon', help='Mean of full VUV single photon response',
                 default=1.2),
    strax.Option('gain_model',
                 help='PMT gain model. Specify as (model_type, model_config)'),
    strax.Option('n_tpc_pmts', type=int,
                 help='Number of TPC PMTs'),
    strax.Option('n_top_pmts', type=int,
                 help='Number of top TPC PMTs'),
    strax.Option('s1_min_area_pattern_fit',
                 help='Skip EventPatternFit reconstruction if S1 area (PE) is less than this',
                 default=2),
    strax.Option('s2_min_area_pattern_fit',
                 help='Skip EventPatternFit reconstruction if S2 area (PE) is less than this',
                 default=10),
    strax.Option('store_per_channel', default=False, type=bool,
                 help='Store normalized LLH per channel for each peak'),
    strax.Option('max_r_pattern_fit', default=straxen.tpc_r, type=float,
                 help='Maximal radius of the peaks where llh calculation will be performed'),
    strax.Option(name='electron_drift_velocity',
                 help='Vertical electron drift velocity in cm/ns (1e4 m/ms)',
                 default=("electron_drift_velocity", "ONLINE", True)),
    strax.Option(name='electron_drift_time_gate',
                 help='Electron drift time from the gate in ns',
                 default=("electron_drift_time_gate", "ONLINE", True)),
)

class EventPatternFit(strax.Plugin):
    '''
    Plugin that provides patter information for events
    '''
    
    depends_on = ('event_area_per_channel', 'event_basics', 'event_positions')
    provides = 'event_pattern_fit'
    __version__ = '0.0.9'

    def infer_dtype(self):
        dtype = [('s2_2llh', np.float32,
                  'Modified Poisson likelihood value for main S2 in the event'),
                 ('alt_s2_2llh', np.float32,
                  'Modified Poisson likelihood value for alternative S2'),
                 ('s1_2llh', np.float32,
                  'Modified Poisson likelihood value for main S1'),
                 ('s1_top_2llh', np.float32,
                  'Modified Poisson likelihood value for main S1, calculated from top array'),
                 ('s1_bottom_2llh', np.float32,
                  'Modified Poisson likelihood value for main S1, calculated from bottom array'),
                 ('s1_area_fraction_top_continuous_probability', np.float32,
                  'Continuous binomial test for S1 area fraction top'),
                 ('s1_area_fraction_top_discrete_probability', np.float32,
                  'Discrete binomial test for S1 area fraction top'),
                 ('s1_photon_fraction_top_continuous_probability', np.float32,
                  'Continuous binomial test for S1 photon fraction top'),
                 ('s1_photon_fraction_top_discrete_probability', np.float32,
                  'Discrete binomial test for S1 photon fraction top'),
                 ('alt_s1_area_fraction_top_continuous_probability', np.float32,
                  'Continuous binomial test for alternative S1 area fraction top'),
                 ('alt_s1_area_fraction_top_discrete_probability', np.float32,
                  'Discrete binomial test for alternative S1 area fraction top'),
                 ('alt_s1_photon_fraction_top_continuous_probability', np.float32,
                  'Continuous binomial test for alternative S1 photon fraction top'),
                 ('alt_s1_photon_fraction_top_discrete_probability', np.float32,
                  'Discrete binomial test for alternative S1 photon fraction top')]
        
        if self.config['store_per_channel']:
            dtype += [
                (('2LLH per channel for main S2', 's2_2llh_per_channel'),
                 np.float32, (self.config['n_top_pmts'], )),
                (('2LLH per channel for alternative S2', 'alt_s2_2llh_per_channel'),
                 np.float32, (self.config['n_top_pmts'], )),
                (('Pattern main S2', 's2_pattern'),
                 np.float32, (self.config['n_top_pmts'], )),
                (('Pattern alt S2', 'alt_s2_pattern'),
                 np.float32, (self.config['n_top_pmts'], )),
                (('Pattern for main S1', 's1_pattern'),
                 np.float32, (self.config['n_tpc_pmts'], )),
                (('2LLH per channel for main S1', 's1_2llh_per_channel'),
                 np.float32, (self.config['n_tpc_pmts'], )),
            ]
        dtype += strax.time_fields
        return dtype
    
    def setup(self):
        self.electron_drift_velocity = get_correction_from_cmt(self.run_id, self.config['electron_drift_velocity'])
        self.electron_drift_time_gate = get_correction_from_cmt(self.run_id, self.config['electron_drift_time_gate'])
        self.mean_pe_photon = self.config['mean_pe_per_photon']
        
        # Getting S1 AFT maps
        self.s1_aft_map = straxen.InterpolatingMap(
            straxen.get_resource(
                self.config['s1_aft_map'],
                fmt=self._infer_map_format(self.config['s1_aft_map'])))
                    
        # Getting optical maps
        self.s1_pattern_map = straxen.InterpolatingMap(
            straxen.get_resource(
                self.config['s1_optical_map'],
                fmt=self._infer_map_format(self.config['s1_optical_map'])))
        self.s2_pattern_map = straxen.InterpolatingMap(
            straxen.get_resource(
                self.config['s2_optical_map'],
                fmt=self._infer_map_format(self.config['s2_optical_map'])))
        
        # Getting gain model to get dead PMTs
        self.to_pe = straxen.get_correction_from_cmt(self.run_id, self.config['gain_model'])
        self.dead_PMTs = np.where(self.to_pe == 0)[0]
        self.pmtbool = ~np.in1d(np.arange(0, self.config['n_tpc_pmts']), self.dead_PMTs)
        self.pmtbool_top = self.pmtbool[:self.config['n_top_pmts']]
        self.pmtbool_bottom = self.pmtbool[self.config['n_top_pmts']:self.config['n_tpc_pmts']]
        
    def compute(self, events):
        
        result = np.zeros(len(events), dtype=self.dtype)
        result['time'] = events['time']
        result['endtime'] = strax.endtime(events)

        # Computing LLH values for S1s
        self.compute_s1_llhvalue(events, result)
        
        # Computing LLH values for S2s
        self.compute_s2_llhvalue(events, result)
        
        # Computing binomial test for s1 area fraction top
        s1_area_fraction_top_probability = np.vectorize(_s1_area_fraction_top_probability)
        positions = np.vstack([events['x'], events['y'], events['z']]).T
        aft_prob = self.s1_aft_map(positions)
        
        alt_s1_interaction_drift_time = events['s2_center_time']-events['alt_s1_center_time']
        alt_s1_interaction_z = -self.electron_drift_velocity*(alt_s1_interaction_drift_time-self.electron_drift_time_gate)
        alt_positions = np.vstack([events['x'], events['y'], alt_s1_interaction_z]).T     
        alt_aft_prob = self.s1_aft_map(alt_positions)
        
        # main s1 events
        mask_s1 = ~np.isnan(aft_prob)
        mask_s1 &= ~np.isnan(events['s1_area'])
        mask_s1 &= ~np.isnan(events['s1_area_fraction_top'])
        
        # default value is nan, it will be overwrite if the event satisfy the requirements
        result['s1_area_fraction_top_continuous_probability'][:] = np.nan
        result['s1_area_fraction_top_discrete_probability'][:] = np.nan
        result['s1_photon_fraction_top_continuous_probability'][:] = np.nan
        result['s1_photon_fraction_top_discrete_probability'][:] = np.nan
        
        # compute binomial test only if we have events that have valid aft prob, s1 area and s1 aft
        if np.sum(mask_s1):
            arg = aft_prob[mask_s1], events['s1_area'][mask_s1], events['s1_area_fraction_top'][mask_s1]
            result['s1_area_fraction_top_continuous_probability'][mask_s1] = s1_area_fraction_top_probability(*arg)
            result['s1_area_fraction_top_discrete_probability'][mask_s1] = s1_area_fraction_top_probability(*arg, 'discrete')
            arg = aft_prob[mask_s1], events['s1_area'][mask_s1]/self.config['mean_pe_per_photon'], events['s1_area_fraction_top'][mask_s1]
            result['s1_photon_fraction_top_continuous_probability'][mask_s1] = s1_area_fraction_top_probability(*arg)
            result['s1_photon_fraction_top_discrete_probability'][mask_s1] = s1_area_fraction_top_probability(*arg, 'discrete')
        
        # alternative s1 events
        mask_alt_s1 = ~np.isnan(alt_aft_prob)
        mask_alt_s1 &= ~np.isnan(events['alt_s1_area'])
        mask_alt_s1 &= ~np.isnan(events['alt_s1_area_fraction_top'])
        
        # default value is nan, it will be ovewrite if the event satisfy the requirments
        result['alt_s1_area_fraction_top_continuous_probability'][:] = np.nan
        result['alt_s1_area_fraction_top_discrete_probability'][:] = np.nan
        result['alt_s1_photon_fraction_top_continuous_probability'][:] = np.nan
        result['alt_s1_photon_fraction_top_discrete_probability'][:] = np.nan
        
        # compute binomial test only if we have events that have valid aft prob, alt s1 area and alt s1 aft
        if np.sum(mask_alt_s1):
            arg = aft_prob[mask_alt_s1], events['alt_s1_area'][mask_alt_s1], events['alt_s1_area_fraction_top'][mask_alt_s1]
            result['alt_s1_area_fraction_top_continuous_probability'][mask_alt_s1] = s1_area_fraction_top_probability(*arg)
            result['alt_s1_area_fraction_top_discrete_probability'][mask_alt_s1] = s1_area_fraction_top_probability(*arg, 'discrete')
            arg = aft_prob[mask_alt_s1], events['alt_s1_area'][mask_alt_s1]/self.config['mean_pe_per_photon'], events['alt_s1_area_fraction_top'][mask_alt_s1]
            result['alt_s1_photon_fraction_top_continuous_probability'][mask_alt_s1] = s1_area_fraction_top_probability(*arg)
            result['alt_s1_photon_fraction_top_discrete_probability'][mask_alt_s1] = s1_area_fraction_top_probability(*arg, 'discrete')
                
        return result

    def compute_s1_llhvalue(self, events, result):
        # Selecting S1s for pattern fit calculation
        # - must exist (index != -1)
        # - must have total area larger minimal one
        # - must have positive AFT
        x, y, z = events['x'], events['y'], events['z']
        cur_s1_bool = events['s1_area']>self.config['s1_min_area_pattern_fit']
        cur_s1_bool &= events['s1_index']!=-1
        cur_s1_bool &= events['s1_area_fraction_top']>=0
        cur_s1_bool &= np.isfinite(x)
        cur_s1_bool &= np.isfinite(y)
        cur_s1_bool &= np.isfinite(z)
        cur_s1_bool &= (x**2 + y**2) < self.config['max_r_pattern_fit']**2
        
        # default value is nan, it will be ovewrite if the event satisfy the requirments
        result['s1_2llh'][:] = np.nan
        result['s1_top_2llh'][:] = np.nan
        result['s1_bottom_2llh'][:] = np.nan
        
        # Making expectation patterns [ in PE ]
        if np.sum(cur_s1_bool):
            s1_map_effs = self.s1_pattern_map(np.array([x, y, z]).T)[cur_s1_bool, :]
            s1_area = events['s1_area'][cur_s1_bool]
            s1_pattern = s1_area[:, None]*(s1_map_effs[:, self.pmtbool])/np.sum(s1_map_effs[:, self.pmtbool], axis=1)[:, None] 

            s1_pattern_top = (events['s1_area_fraction_top'][cur_s1_bool]*s1_area)
            s1_pattern_top = s1_pattern_top[:, None]*((s1_map_effs[:, :self.config['n_top_pmts']])[:, self.pmtbool_top])
            s1_pattern_top /= np.sum((s1_map_effs[:, :self.config['n_top_pmts']])[:, self.pmtbool_top], axis=1)[:, None] 
            s1_pattern_bottom = ((1-events['s1_area_fraction_top'][cur_s1_bool])*s1_area)
            s1_pattern_bottom = s1_pattern_bottom[:, None]*((s1_map_effs[:, self.config['n_top_pmts']:])[:, self.pmtbool_bottom])
            s1_pattern_bottom /= np.sum((s1_map_effs[:, self.config['n_top_pmts']:])[:, self.pmtbool_bottom], axis=1)[:, None] 

            # Getting pattern from data
            s1_area_per_channel_ = events['s1_area_per_channel'][cur_s1_bool,:]
            s1_area_per_channel = s1_area_per_channel_[:, self.pmtbool]
            s1_area_per_channel_top = (s1_area_per_channel_[:, :self.config['n_top_pmts']])[:, self.pmtbool_top]
            s1_area_per_channel_bottom = (s1_area_per_channel_[:, self.config['n_top_pmts']:])[:, self.pmtbool_bottom]

            # Top and bottom
            arg1 = s1_pattern/self.mean_pe_photon, s1_area_per_channel, self.mean_pe_photon
            arg2 = s1_area_per_channel/self.mean_pe_photon, s1_area_per_channel, self.mean_pe_photon
            norm_llh_val = (neg2llh_modpoisson(*arg1) - neg2llh_modpoisson(*arg2))
            result['s1_2llh'][cur_s1_bool] = np.sum(norm_llh_val, axis=1)

            # If needed to stire - store only top and bottom array, but not together
            if self.config['store_per_channel']:
                # Storring pattern information
                store_patterns = np.zeros((s1_pattern.shape[0], self.config['n_tpc_pmts']) )  
                store_patterns[:, self.pmtbool] = s1_pattern
                result['s1_pattern'][cur_s1_bool] = store_patterns
                # Storing actual LLH values
                store_2LLH_ch = np.zeros((norm_llh_val.shape[0], self.config['n_tpc_pmts']) )
                store_2LLH_ch[:, self.pmtbool] = norm_llh_val
                result['s1_2llh_per_channel'][cur_s1_bool] = store_2LLH_ch

            # Top
            arg1 = s1_pattern_top/self.mean_pe_photon, s1_area_per_channel_top, self.mean_pe_photon
            arg2 = s1_area_per_channel_top/self.mean_pe_photon, s1_area_per_channel_top, self.mean_pe_photon
            norm_llh_val = (neg2llh_modpoisson(*arg1) - neg2llh_modpoisson(*arg2))
            result['s1_top_2llh'][cur_s1_bool] = np.sum(norm_llh_val, axis=1)

            # Bottom
            arg1 = s1_pattern_bottom/self.mean_pe_photon, s1_area_per_channel_bottom, self.mean_pe_photon
            arg2 = s1_area_per_channel_bottom/self.mean_pe_photon, s1_area_per_channel_bottom, self.mean_pe_photon
            norm_llh_val = (neg2llh_modpoisson(*arg1) - neg2llh_modpoisson(*arg2))
            result['s1_bottom_2llh'][cur_s1_bool] = np.sum(norm_llh_val, axis=1)
            
    def compute_s2_llhvalue(self, events, result):
        for t_ in ['s2', 'alt_s2']:
            # Selecting S2s for pattern fit calculation
            # - must exist (index != -1)
            # - must have total area larger minimal one
            # - must have positive AFT
            x, y = events[t_+'_x'], events[t_+'_y']
            cur_s2_bool = (events[t_+'_area']>self.config['s2_min_area_pattern_fit'])
            cur_s2_bool &= (events[t_+'_index']!=-1)
            cur_s2_bool &= (events[t_+'_area_fraction_top']>0)
            cur_s2_bool &= (x**2 + y**2) < self.config['max_r_pattern_fit']**2
            
            # default value is nan, it will be ovewrite if the event satisfy the requirments
            result[t_+'_2llh'][:] = np.nan
            
            # Making expectation patterns [ in PE ]
            if np.sum(cur_s2_bool):
                s2_map_effs = self.s2_pattern_map(np.array([x, y]).T)[cur_s2_bool, 0:self.config['n_top_pmts']]
                s2_map_effs = s2_map_effs[:, self.pmtbool_top]
                s2_top_area = (events[t_+'_area_fraction_top']*events[t_+'_area'])[cur_s2_bool]
                s2_pattern  = s2_top_area[:, None]*s2_map_effs/np.sum(s2_map_effs, axis=1)[:,None]

                # Getting pattern from data
                s2_top_area_per_channel = events[t_+'_area_per_channel'][cur_s2_bool, 0:self.config['n_top_pmts']]
                s2_top_area_per_channel = s2_top_area_per_channel[:, self.pmtbool_top]

                # Calculating LLH, this is shifted Poisson
                # we get area expectation and we need to scale them to get
                # photon expectation
                norm_llh_val = (neg2llh_modpoisson(
                                     mu    = s2_pattern/self.mean_pe_photon, 
                                     areas = s2_top_area_per_channel, 
                                     mean_pe_photon=self.mean_pe_photon)
                                        - 
                                neg2llh_modpoisson(
                                     mu    = s2_top_area_per_channel/self.mean_pe_photon, 
                                     areas = s2_top_area_per_channel, 
                                     mean_pe_photon=self.mean_pe_photon)
                               )
                result[t_+'_2llh'][cur_s2_bool] = np.sum(norm_llh_val, axis=1)

                if self.config['store_per_channel']:
                    store_patterns = np.zeros((s2_pattern.shape[0], self.config['n_top_pmts']) )
                    store_patterns[:, self.pmtbool_top] = s2_pattern
                    result[t_+'_pattern'][cur_s2_bool] = store_patterns#:s2_pattern[cur_s2_bool]

                    store_2LLH_ch = np.zeros((norm_llh_val.shape[0], self.config['n_top_pmts']) )
                    store_2LLH_ch[:, self.pmtbool_top] = norm_llh_val
                    result[t_+'_2llh_per_channel'][cur_s2_bool] = store_2LLH_ch
                    
    @staticmethod
    def _infer_map_format(map_name, known_formats=('pkl', 'json', 'json.gz')):
        for fmt in known_formats:
            if map_name.endswith(fmt):
                return fmt
        raise ValueError(f'Extension of {map_name} not in {known_formats}')


def neg2llh_modpoisson(mu=None, areas=None, mean_pe_photon=1.0):
    """
    Modified poisson distribution with proper normalization for shifted poisson. 
    
    mu - expected number of photons per channel
    areas  - observed areas per channel
    mean_pe_photon - mean of area responce for one photon
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        fraction = areas/mean_pe_photon
        res = 2.*(mu -
                  (fraction)*np.log(mu) +
                  loggamma((fraction)+1) +
                  np.log(mean_pe_photon)
                 )
    is_zero = areas <= 0    # If area equals or smaller than 0 - assume 0
    res[is_zero] = 2.*mu[is_zero]
    # if zero channel has negative expectation, assume LLH to be 0 there
    # this happens in the normalization factor calculation when mu is received from area
    neg_mu = mu < 0.0
    res[is_zero | neg_mu] = 0.0
    return res


# continuous and discrete binomial test
# https://github.com/poliastro/cephes/blob/master/src/bdtr.c

@numba.vectorize([numba.float64(numba.float64, numba.float64, numba.float64)])
def binom_pmf(k, n, p):
    scale_log = numba_gammaln(n + 1) - numba_gammaln(n - k + 1) - numba_gammaln(k + 1)
    ret_log = scale_log + k * np.log(p) + (n - k) * np.log(1 - p)
    return np.exp(ret_log)

@numba.njit
def binom_cdf(k, n, p):
    if k < 0:
        return np.nan
    if k == n:
        return 1.0
    dn = n - k
    if k == 0:
        dk = np.exp(dn * np.log(1.0 - p))
    else:
        dk = k + 1
        dk = numba_betainc(dn, dk, 1.0 - p)
    return dk

@numba.njit
def binom_sf(k, n, p):
    if k < 0:
        return 1.0
    if k == n:
        return 0.0
    dn = n - k
    if k == 0:
        if p < .01:
            dk = -np.expm1(dn * np.log1p(-p))
        else:
            dk = 1.0 - np.exp(dn * np.log(1.0 - p))
    else:
        dk = k + 1
        dk = numba_betainc(dk, dn, p)
    return dk

@numba.njit
def _find_interval(n, p):
    mu = n*p
    sigma = np.sqrt(n*p*(1-p))
    if mu-2*sigma < 0:
        s1_min_range = 0
    else:
        s1_min_range = mu-2*sigma
    if mu+2*sigma > n:
        s1_max_range = n
    else:
        s1_max_range = mu+2*sigma
    return s1_min_range, s1_max_range

@numba.njit
def _get_min_and_max(s1_min, s1_max, n, p, shift):
    s1 = np.arange(s1_min, s1_max, 0.01)
    ds1 = binom_pmf(s1, n, p)
    s1argmax = np.argmax(ds1)
    if np.argmax(ds1) - shift > 0:
        minimum = s1[s1argmax - shift]
    else:
        minimum = s1[0]
    maximum = s1[s1argmax + shift]
    return minimum, maximum, s1[s1argmax]

@numba.njit
def binom_test(k, n, p):
    """
    The main purpose of this algorithm is to find the value j on the
    other side of the mean that has the same probability as k, and
    integrate the tails outward from k and j. In the case where either
    k or j are zero, only the non-zero tail is integrated.
    """
    # define the S1 interval for finding the maximum
    _s1_min_range, _s1_max_range = _find_interval(n, p)

    # compute the binomial probability for each S1 and define the Binom range for later
    minimum, maximum, s1max = _get_min_and_max(_s1_min_range, _s1_max_range, n, p, shift=2)
   
    # comments TODO
    _d0 = binom_pmf(0, n, p)
    _dmax = binom_pmf(s1max, n, p)
    _dn = binom_pmf(n, n, p)
    d = binom_pmf(k, n, p)
    rerr = 1 + 1e-7
    d = d * rerr
    _d0 = _d0 * rerr
    _dmax = _dmax * rerr 
    _dn = _dn * rerr
    # define number of interaction for finding the the value j
    # the exceptional case of n<=0, is avoid since n_iter is at least 2
    if n > 0:
        n_iter = int(np.round_(np.log10(n)) + 1)
        n_iter = max(n_iter, 2)
    else:
        n_iter = 2
    # comments TODO
    if k < minimum:
        if (_d0 >= d) and (_d0 > _dmax):
            n_iter, j_min, j_max = -1, 0, 0
        elif _dn > d:
            n_iter, j_min, j_max = -2, 0, 0
        else:
            j_min, j_max = s1max, n
        def _check_(d, y0, y1):
            return (d>y1) and (d<=y0)
    # comments TODO
    elif k>maximum:
        if _d0 >= d:
            n_iter, j_min, j_max = -1, 0, 0
        else:
            j_min, j_max = 0, s1max
        def _check_(d, y0, y1):
            return (d>=y0) and (d<y1)
    # comments TODO
    else:
        n_iter, j_min, j_max = 0, k, k
        def _check_(d, y0, y1):
            return (d>=y0) and (d<y1)
    
    # comments TODO
    if (d==0):
        pval = 0.0
    # comments TODO
    else:
        for i in range(n_iter):
            n_pts = int(j_max - j_min)
            if (i<2) and (n_pts < 50): 
                n_pts = 50
            j_range = np.linspace(j_min, j_max, n_pts)
            y = binom_pmf(j_range, n, p)
            for ii in range(n_pts - 1):
                if _check_(d, y[ii], y[ii + 1]):
                    j_min, j_max = j_range[ii], j_range[ii + 1]
                    break

        j = max(min((j_min + j_max) / 2, n), 0)
        
        # here we use n_iter also for decide to do one-side (left/right) test 
        # or two-side test
        if (k * j == 0)and(n_iter==-1):
            pval = binom_sf(max(k, j), n, p)
        elif (k * j == 0)and(n_iter==-2):
            pval = binom_cdf(max(k, j), n, p)
        else:
            pval = binom_cdf(min(k, j), n, p) + binom_sf(max(k, j), n, p)
        pval = min(1.0, pval)
    return pval

def _s1_area_fraction_top_probability(aft_prob, area_tot, area_fraction_top, mode='continuous'):
    '''
    Wrapper that does the S1 AFT probability calculation for you
    '''
    
    area_top = area_tot * area_fraction_top
    
    # Raise a warning in case one of these three condition is verified
    # and return binomial test equal to nan since they are not physical
    # k: size_top, n: size_tot, p: aft_prob
    do_test = True
    if area_tot < area_top:
        # warnings.warn(f'n {area_tot} must be >= k {area_top}')
        binomial_test = np.nan
        do_test = False
    if (aft_prob > 1.0) or (aft_prob < 0.0):
        # warnings.warn(f'p {aft_prob} must be in range [0, 1]')
        binomial_test = np.nan
        do_test = False
    if area_top < 0:
        # warnings.warn(f'k {area_top} must be >= 0')
        binomial_test = np.nan
        do_test = False
        
    if do_test:
        if mode == 'discrete':
            binomial_test = binom_pmf(area_top, area_tot, aft_prob)
        else:
            binomial_test = binom_test(area_top, area_tot, aft_prob)
        
    return binomial_test
