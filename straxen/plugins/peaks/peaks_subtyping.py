from enum import IntEnum

import numpy as np
import numba
import strax
import straxen


class PeakSubtyping(IntEnum):
    """Peak Subtyping Dictionary"""
    Undefined = 6
    # Unknown (0)
    Junk = 0
    # S1 (1)
    isoS1 = 10        # S1 without nearby peaks
    S1 = 11           # Regular S1 with a matched pS2
    loneS1 = 12       # Regular S1 without a matched pS2
    sloneS1 = 13      # Regular S1 without a matched pS2, with area < 0.5SE
    # S2 (2)
    DE = 21           # Delayed Extracted few electron peak
    pS2 = 22          # S2 matched with an S1, with area >= 5SE
    isoDE = 23        # S2 without nearby peaks, with area < 5SE
    isopS2 = 24       # S2 without nearby peaks, with area >= 5SE
    lonepS2 = 25      # S2 with area >= 5SE, without a matched S1 but not categorized as other large S2s
    S1olS2 = 261      # S2 with a nearby pS2, with area >= max(0.5 * pS2, 5SE), after S1 but before S2
    S2olS2 = 262      # S2 with a nearby pS2, with area >= max(0.5 * pS2, 5SE), after S2
    S1PH = 271        # Photoionization S2s, with area < max(0.5 * pS2, 5SE), after S1 but before S2
    S2PH = 272        # Photoionization S2s, with area < max(0.5 * pS2, 5SE), after S2
    slS1PH = 273      # Photoionization S2s after a sloneS1
    fakeS2 = 28       # S2 with area < 5SE, with a paired pS2 satisfying S1-S2 time correlation
    fakeS2_olS2 = 29  # olS2 after the fakeS2 and before the associated pS2
    fakeS2_PH = 20    # Photoionization after the fakeS2 and before the associated pS2


class PeaksSubtypes(strax.Plugin):
    """
    Subtyping Peaks
    This plugin scans forward in time and catagorize peaks into subtypes based on their correlations with nearby peaks.
    Reference note: https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:peak_subtyping_study
    :returns: an integer index for each peak. 
      Please refer to this note: https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:peak_subtyping_dictionary
    """

    __version__ = '0.2.0'
    provides = 'subtype_mask'
    depends_on = ('peak_basics')
    save_when = strax.SaveWhen.NEVER
    dtype = [('subtype', np.int16, 'time subtyping of peaks')] + strax.time_fields

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
        default=5., type=(int, float),
        help="cutoff between small S2 and large S2, 5e- for now"
    )

    s1_s2_window_fac = straxen.URLConfig(
        default=1.1, type=(int, float),
        help="window to look for pS2 after S1 in multiples of full drift time,"
             "slightly larger to incorporate spread uncertainties of e- drifting"
    )

    other_ls2_fac = straxen.URLConfig(
        default=0.5, type=(int, float),
        help="threshold to consider other large S2s within a window"
    )

    mis_s2_fac = straxen.URLConfig(
        default=0.5, type=(int, float),
        help="threshold to consider potential misclassified SE"
    )

    after_s1_window_ext_fac = straxen.URLConfig(
        default=1, type=(int, float),
        help="extend scanning window after identified large S1s by this much full drift time, if no S2 is found in s1_s2_window"
    )

    after_s2_window_ext_fac = straxen.URLConfig(
        default=2, type=(int, float),
        help="extend scanning window after identified primary S2 by this much full drift time"
    )

    def setup(self):
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)
        self.se_gain = self.ref_se_gain['all_tpc']
        self.se_span = self.ref_se_span['all_tpc']
        self.ls2_threshold = self.ls2_threshold_ne * self.se_gain
        self.mis_s2_threshold = self.mis_s2_fac * self.se_gain
        self.ls2_threshold += np.sqrt(self.ls2_threshold_ne) * self.se_span
        self.s1_s2_window = self.s1_s2_window_fac * self.drift_time_max

    def mark_with_s1(
        self,
        peaks,
        mask,
        drift_time_max, ls2_threshold, other_ls2_fac,
        s1_s2_window,
        after_s1_window_ext_fac, after_s2_window_ext_fac
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

        undefined_mask = (mask == PeakSubtyping.Undefined)
        dmask = mask[undefined_mask]
        ts = peaks['time'][undefined_mask]  # time
        tps = peaks['type'][undefined_mask]  # type
        pas = peaks['area'][undefined_mask]  # area
        s1s_t = ts[tps == 1]  # s1 time
        s1s_a = pas[tps == 1]  # s1 area
        s2s_t = ts[tps == 2]  # s2 time
        s2s_a = pas[tps == 2]  # s2 area
        s1_mask = dmask[tps == 1]  # define s1 subtypes
        s2_mask = dmask[tps == 2]  # define s2 subtypes

        for i, s1_t in enumerate(s1s_t):
            # locate pS2:
            ps2_cand_cond = (s2s_t > s1_t) & (s2s_t <= s1_t + s1_s2_window)
            if ps2_cand_cond.sum():
                # do we have pS2 or a bunch of small S2s
                max_ind = np.nanargmax(s2s_a[ps2_cand_cond])
                max_val = s2s_a[ps2_cand_cond][max_ind]
                max_time = s2s_t[ps2_cand_cond][max_ind]
                if max_val >= ls2_threshold:
                    # pS2 located
                    whole_window_cond = (s2s_t > s1_t)
                    whole_window_cond &= (s2s_t <= s2s_t[ps2_cand_cond][max_ind] + after_s2_window_ext_fac * drift_time_max)
                    s2_mask_tmp = s2_mask[whole_window_cond]
                    small_s2s_a = s2s_a[whole_window_cond] < np.max([ls2_threshold, other_ls2_fac * max_val])
                    not_pS2 = s2_mask_tmp != PeakSubtyping.pS2
                    within_dt_s2s_t = s2s_t[whole_window_cond] < max_time
                    s2_mask_tmp[
                        ~small_s2s_a & not_pS2 & within_dt_s2s_t] = PeakSubtyping.S1olS2
                    s2_mask_tmp[
                        ~small_s2s_a & not_pS2 & ~within_dt_s2s_t] = PeakSubtyping.S2olS2
                    s2_mask_tmp[
                        small_s2s_a & not_pS2 & within_dt_s2s_t] = PeakSubtyping.S1PH
                    s2_mask_tmp[
                        small_s2s_a & not_pS2 & ~within_dt_s2s_t] = PeakSubtyping.S2PH
                    # assign the pS2 and S1
                    s2_mask_tmp[s2s_t[whole_window_cond] == max_time] = PeakSubtyping.pS2
                    s1_mask[i] = PeakSubtyping.S1
                    s2_mask[whole_window_cond] = s2_mask_tmp
                else:
                    # no pS2, loneS1 spotted
                    whole_window_cond = (s2s_t > s1_t)
                    whole_window_cond &= (s2s_t <= s1_t + after_s1_window_ext_fac * drift_time_max)
                    if s1s_a[i] < self.mis_s2_threshold:
                        # Potential misclassified SE signal
                        s2_mask[whole_window_cond] = PeakSubtyping.slS1PH
                        s1_mask[i] = PeakSubtyping.sloneS1
                    else:
                        s2_mask[whole_window_cond] = PeakSubtyping.S1PH
                        s1_mask[i] = PeakSubtyping.loneS1
            else:
                if s1s_a[i] < self.mis_s2_threshold:
                    # Potential misclassified SE signal
                    s1_mask[i] = PeakSubtyping.sloneS1
                else:
                    s1_mask[i] = PeakSubtyping.loneS1
        dmask[tps == 1] = s1_mask
        dmask[tps == 2] = s2_mask
        mask[undefined_mask] = dmask

    def mark_s2s(
        self,
        peaks,
        mask,
        drift_time_max, ls2_threshold, other_ls2_fac,
        s1_s2_window, after_s2_window_ext_fac
        ):
        '''
        After marking all peaks after S1s, all that's left are S2s.
        One extra occasion is the "fakeS2".
        If a small S2 is identified and a pS2 can be paired up with it, such small S2 is marked "fake S2". 
        '''

        undefined_mask = (mask == PeakSubtyping.Undefined)
        dmask = mask[undefined_mask]
        pt = peaks['time'][undefined_mask]  # time
        pa = peaks['area'][undefined_mask]  # area

        # load only the ones not being classified
        # add a termination number of while loops
        max_iter = len(peaks)
        current_step = 0
        while (dmask == PeakSubtyping.Undefined).sum() > 0 and current_step < max_iter:
            current_step += 1
            start_s2_t = pt[dmask == PeakSubtyping.Undefined][0]
            start_s2_a = pa[dmask == PeakSubtyping.Undefined][0]

            # first mark lonepS2s
            if start_s2_a >= ls2_threshold:
                # assign this S2 as lonepS2
                dmask_tmp = dmask[dmask == PeakSubtyping.Undefined]
                dmask_tmp[0] = PeakSubtyping.lonepS2
                dmask[dmask == PeakSubtyping.Undefined] = dmask_tmp
                # mark upto 2 fdt into olS2 or PH
                whole_window_cond = (pt > start_s2_t) & (pt <= start_s2_t + after_s2_window_ext_fac * drift_time_max)
                mask_tmp = dmask[whole_window_cond]
                small_ps = pa[whole_window_cond] < ls2_threshold
                mask_tmp[~small_ps] = PeakSubtyping.S2olS2
                mask_tmp[small_ps] = PeakSubtyping.S2PH
                dmask[whole_window_cond] = mask_tmp
            else:
                # A small S2 might actually be an S1, so here I also mark suspicious couplings
                ps2_cand_cond = (pt > start_s2_t) & (pt <= start_s2_t + s1_s2_window)
                if len(pa[ps2_cand_cond]):
                    # do we have pS2 or a bunch of small S2s
                    max_ind = np.nanargmax(pa[ps2_cand_cond])
                    max_val = pa[ps2_cand_cond][max_ind]
                    max_time = pt[ps2_cand_cond][max_ind]
                    if max_val >= ls2_threshold:
                        # a suspicious coupling spotted. label this s2 as fakeS2
                        dmask_tmp = dmask[dmask == PeakSubtyping.Undefined]
                        dmask_tmp[0] = PeakSubtyping.fakeS2
                        dmask[dmask == PeakSubtyping.Undefined] = dmask_tmp
                        whole_window_cond = (pt > start_s2_t) & (pt <= max_time + after_s2_window_ext_fac * drift_time_max)
                        mask_tmp = dmask[whole_window_cond]
                        small_pa = pa[whole_window_cond] < np.max([ls2_threshold, other_ls2_fac * max_val])
                        within_dt_pt = pt[whole_window_cond] < max_time
                        mask_tmp[
                            ~small_pa & within_dt_pt] = PeakSubtyping.fakeS2_olS2
                        mask_tmp[
                            small_pa & within_dt_pt] = PeakSubtyping.fakeS2_PH
                        mask_tmp[
                            ~small_pa & ~within_dt_pt] = PeakSubtyping.S2olS2
                        mask_tmp[
                            small_pa & ~within_dt_pt] = PeakSubtyping.S2PH
                        mask_tmp[pt[whole_window_cond] == max_time] = PeakSubtyping.pS2
                        dmask[whole_window_cond] = mask_tmp
                    else:
                        dmask_tmp = dmask[dmask == PeakSubtyping.Undefined]
                        dmask_tmp[0] = PeakSubtyping.DE
                        dmask[dmask == PeakSubtyping.Undefined] = dmask_tmp
                else:
                    dmask_tmp = dmask[dmask == PeakSubtyping.Undefined]
                    dmask_tmp[0] = PeakSubtyping.DE
                    dmask[dmask == PeakSubtyping.Undefined] = dmask_tmp
        mask[undefined_mask] = dmask

    def compute(self, peaks):
        mask = np.ones(len(peaks)) * PeakSubtyping.Undefined
        mask[peaks['type'] == 0] = PeakSubtyping.Junk

        self.mark_with_s1(
            peaks, mask,
            self.drift_time_max, self.ls2_threshold, self.other_ls2_fac,
            self.s1_s2_window, self.after_s1_window_ext_fac, self.after_s2_window_ext_fac)

        self.mark_s2s(
            peaks, mask,
            self.drift_time_max, self.ls2_threshold, self.other_ls2_fac,
            self.s1_s2_window, self.after_s2_window_ext_fac)

        n_undefined = (mask == PeakSubtyping.Undefined).sum()
        if n_undefined:
            raise RuntimeError(f'We still find {n_undefined} peaks undefined!')

        return dict(
            time=peaks['time'],
            endtime=strax.endtime(peaks),
            subtype=mask
        )
