import numpy as np
import strax
import straxen
import numba

class PeaksSubtypes(strax.Plugin):
    """
    Subtyping Peaks
    This plugin scans forward in time and catagorize peaks into subtypes based on their correlations with nearby peaks.
    Reference note: https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:peak_subtyping_study
    :returns: an integer index for each peak. Please refer to this note: https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:peak_subtyping_dictionary
    """

    __version__ = '0.2.0'
    provides = 'subtype_mask'
    depends_on = ('peak_basics')
    dtype = [('subtype', np.int16, 'time subtyping of peaks')]+strax.time_fields

    ref_se_gain = straxen.URLConfig(
        default='bodega://se_gain?bodega_version=v1',
        help='Nominal single electron (SE) gain in PE / electron extracted.')

    ref_se_span = straxen.URLConfig(
        default='bodega://se_spread?bodega_version=v0',
        help = "SE spread value"
    )

    electron_drift_velocity = straxen.URLConfig(
        default='cmt://'
                'electron_drift_velocity'
                '?version=ONLINE&run_id=plugin.run_id',
        cache=True,
        help='Vertical electron drift velocity in cm/ns (1e4 m/ms)'
    )

    max_drift_length = straxen.URLConfig(
        default=straxen.tpc_z, type=(int, float),
        help='Total length of the TPC from the bottom of gate to the '
             'top of cathode wires [cm]',
    )

    ls2_threshold_ne = straxen.URLConfig(
        default = 5, type=int, # e-
        help="cutoff between small S2 and large S2, 5e- for now"
    )

    s1_s2_window_fac = straxen.URLConfig(
        default = 1.1, 
        help = "window to look for pS2 after S1 in multiples of full drift time,"
               "slightly larger to incorporate spread uncertainties of e- drifting"
    )

    other_ls2 = straxen.URLConfig(
        default = 0.5, 
        help = "threshold to consider other large S2s within a window"
    )

    s2_merge_window_fac = straxen.URLConfig(
        default = 0.1, 
        help = "if two S2s happen within this window we merge the S2 areas. Not used for now"
    )

    s2_discard_window_fac = straxen.URLConfig(
        default = 3, 
        help="if two S2s happen as distant as this, then they are not coupled. Not used for now"
    )

    def set_vals(self):
        self.full_dt = int(self.max_drift_length / self.electron_drift_velocity)
        self.se_gain = self.ref_se_gain['all_tpc']
        self.se_span = self.ref_se_span['all_tpc']
        self.ls2_threshold = self.ls2_threshold_ne*self.se_gain+np.sqrt(self.ls2_threshold_ne)*self.se_span
        self.s1_s2_window = self.s1_s2_window_fac*self.full_dt
        self.s2_merge_window = self.s2_merge_window_fac*self.full_dt
        self.s2_discard_window = self.s2_discard_window_fac*self.full_dt

    @staticmethod
    @numba.njit()
    def find_max_ind(array):
        max_ind = 0
        max_val = array[0]

        for i,item in enumerate(array):
            if item > max_val:
                max_val = item
                max_ind = i

        return max_ind,max_val

    def mark_with_s1(
        self,
        p_t, p_a, p_type,
        mask,
        full_dt, ls2_threshold, other_ls2,
        s1_s2_window, se_gain
        ):
        '''
        Look after each S1s and classify:
        1. mark the largest peak within s1_s2_window as pS2 (exceeding ls2_threshold)
        2. extend window after pS2
        3. mark all other large S2s as olS2. 
        4. mark all other small S2s as photoionization (PH)

        if pS2 identified, extend the window to 2fdt after such pS2 and mark S2s:
        1. other large S2 if an S2 is larger than ls2_threshold and half the pS2 size
        2. photoionization if not olS2
        '''

        dmask = mask[mask == 6]
        ts = p_t[mask == 6] #time
        tps = p_type[mask == 6] #type
        pas = p_a[mask == 6] #area
        s1s_t = ts[tps == 1] # s1 time
        s1s_a = pas[tps == 1] # s1 area
        s2s_t = ts[tps == 2] # s2 time
        s2s_a = pas[tps == 2] # s2 area
        s1_mask = dmask[tps == 1] # define s1 subtypes
        s2_mask = dmask[tps == 2] # define s2 subtypes

        for i in range(len(s1s_t)):
            # locate pS2:
            ps2_cand_cond = (s2s_t > s1s_t[i]) & (s2s_t <= s1s_t[i] + s1_s2_window)
            if len(s2s_a[ps2_cand_cond]):
                # do we have pS2 or a bunch of small S2s
                max_ind,max_val = self.find_max_ind(s2s_a[ps2_cand_cond])
                max_time = s2s_t[ps2_cand_cond][max_ind]
                if max_val >= ls2_threshold:
                    # pS2 located
                    whole_window_cond = (s2s_t > s1s_t[i])
                    whole_window_cond &= (s2s_t <= s2s_t[ps2_cand_cond][max_ind] + 2 * full_dt)
                    s2_mask_tmp = s2_mask[whole_window_cond]
                    s2_mask_tmp[
                        (s2s_a[whole_window_cond] >= np.max([ls2_threshold,other_ls2 * max_val]))
                        & (s2_mask[whole_window_cond] != 22)
                        & (s2s_t[whole_window_cond] < max_time)] = 261
                    s2_mask_tmp[
                        (s2s_a[whole_window_cond] >= np.max([ls2_threshold,other_ls2 * max_val]))
                        & (s2_mask[whole_window_cond] != 22)
                        & (s2s_t[whole_window_cond] >= max_time)] = 262
                    s2_mask_tmp[
                        (s2s_a[whole_window_cond] < np.max([ls2_threshold,other_ls2 * max_val]))
                        & (s2_mask[whole_window_cond] != 22)
                        & (s2s_t[whole_window_cond] < max_time)] = 271
                    s2_mask_tmp[
                        (s2s_a[whole_window_cond] < np.max([ls2_threshold,other_ls2 * max_val]))
                        & (s2_mask[whole_window_cond] != 22)
                        & (s2s_t[whole_window_cond] >= max_time)] = 272
                    # assign the pS2 and S1
                    s2_mask_tmp[s2s_t[whole_window_cond] == max_time] = 22
                    s1_mask[i] = 11
                    s2_mask[whole_window_cond] = s2_mask_tmp
                else:
                    #no pS2, loneS1 spotted
                    whole_window_cond = (s2s_t > s1s_t[i]) & (s2s_t <= s1s_t[i] + full_dt)
                    if s1s_a[i] < 0.5 * se_gain:
                        # Potential misclassified SE signal
                        s2_mask[whole_window_cond] = 273
                        s1_mask[i] = 13
                    else:
                        s2_mask[whole_window_cond] = 271
                        s1_mask[i] = 12
            else:
                if s1s_a[i] < 0.5*se_gain:
                    # Potential misclassified SE signal
                    s1_mask[i] = 13
                else:
                    s1_mask[i] = 12
        dmask[tps == 1] = s1_mask
        dmask[tps == 2] = s2_mask
        mask[mask == 6] = dmask

    def mark_s2s(
        self,
        p_t, p_a,
        mask,
        full_dt, ls2_threshold, other_ls2,
        s1_s2_window
        ):
        '''
        After marking all peaks after S1s, all that's left are S2s.
        One extra occasion is the "fakeS2".
        If a small S2 is identified and a pS2 can be paired up with it, such small S2 is marked "fake S2". 
        '''

        dmask = mask[mask == 6] 
        pt = p_t[mask == 6]
        pa = p_a[mask == 6]

        # load only the ones not being classified
        while len(dmask[dmask == 6]) > 0:
            start_s2_t = pt[dmask == 6][0]
            start_s2_a = pa[dmask == 6][0]

            # first mark lonepS2s
            if start_s2_a >= ls2_threshold:
                # assign this S2 as lonepS2
                dmask_tmp = dmask[dmask == 6]
                dmask_tmp[0] = 25
                dmask[dmask == 6] = dmask_tmp
                # mark upto 2 fdt into olS2 or PH
                whole_window_cond = (pt > start_s2_t) & (pt <= start_s2_t + 2 * full_dt)
                mask_tmp = dmask[whole_window_cond]
                mask_tmp[pa[whole_window_cond] >= ls2_threshold] = 262
                mask_tmp[pa[whole_window_cond] < ls2_threshold] = 272
                dmask[whole_window_cond] = mask_tmp
            else:
                # A small S2 might actually be an S1, so here I also mark suspicious couplings
                ps2_cand_cond = (pt > start_s2_t) & (pt <= start_s2_t + s1_s2_window)
                if len(pa[ps2_cand_cond]):
                    # do we have pS2 or a bunch of small S2s
                    max_ind,max_val = self.find_max_ind(pa[ps2_cand_cond])
                    max_time = pt[ps2_cand_cond][max_ind]
                    if max_val>=ls2_threshold:
                        # a suspicious coupling spotted. label this s2 as fakeS2
                        dmask_tmp = dmask[dmask==6]
                        dmask_tmp[0] = 28
                        dmask[dmask == 6] = dmask_tmp
                        whole_window_cond = (pt > start_s2_t) & (pt <= max_time + 2 * full_dt)
                        mask_tmp = dmask[whole_window_cond]
                        mask_tmp[
                            (pa[whole_window_cond] >= np.max([ls2_threshold,other_ls2 * max_val]))
                            & (pt[whole_window_cond] < max_time)] = 29
                        mask_tmp[
                            (pa[whole_window_cond] < np.max([ls2_threshold,other_ls2 * max_val]))
                            &(pt[whole_window_cond] < max_time)] = 20
                        mask_tmp[
                            (pa[whole_window_cond] >= np.max([ls2_threshold,other_ls2 * max_val]))
                            & (pt[whole_window_cond] >= max_time)] = 262
                        mask_tmp[
                            (pa[whole_window_cond] < np.max([ls2_threshold,other_ls2 * max_val]))
                            & (pt[whole_window_cond] >= max_time)] = 272
                        mask_tmp[pt[whole_window_cond] == max_time] = 22
                        dmask[whole_window_cond] = mask_tmp
                    else:
                        dmask_tmp = dmask[dmask == 6]
                        dmask_tmp[0] = 21
                        dmask[dmask == 6] = dmask_tmp
                else:
                    dmask_tmp = dmask[dmask == 6]
                    dmask_tmp[0] = 21
                    dmask[dmask == 6] = dmask_tmp
        mask[mask == 6] = dmask

    def compute(self,peaks):
        self.set_vals()
        self.mask = np.ones(len(peaks)) * 6
        self.mask[peaks['type'] == 0] = 0

        self.mark_with_s1(
            peaks['time'], peaks['area'], peaks['type'],
            self.mask,
            self.full_dt, self.ls2_threshold, self.other_ls2,
            self.s1_s2_window, self.se_gain)

        self.mark_s2s(
            peaks['time'], peaks['area'],
            self.mask,
            self.full_dt, self.ls2_threshold, self.other_ls2, self.s1_s2_window)

        return dict(
            time=peaks['time'],
            endtime=strax.endtime(peaks),
            subtype = self.mask
        )
