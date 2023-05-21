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


class PeaksSubtypes(strax.OverlapWindowPlugin):
    """
    Subtyping Peaks
    This plugin scans forward in time and catagorize peaks into subtypes
    based on their correlations with nearby peaks.
    Reference note:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:peak_subtyping_study
    :returns: an integer index for each peak.
      Please refer to this note:
      https://xe1t-wiki.lngs.infn.it/doku.php?id=jlong:peak_subtyping_dictionary
    """

    __version__ = '0.2.1'
    provides = 'subtype_mask'
    depends_on = ('peak_basics')
    save_when = strax.SaveWhen.NEVER
    subtype_dtype = np.int16

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

    large_s2_threshold_ne = straxen.URLConfig(
        default=5., type=(int, float),
        help="cutoff between small S2 and large S2, 5e- for now"
    )

    other_large_s2_fac = straxen.URLConfig(
        default=0.5, type=(int, float),
        help="threshold to consider other large S2s within a window"
    )

    mis_s2_fac = straxen.URLConfig(
        default=0.5, type=(int, float),
        help="threshold to consider potential misclassified SE"
    )

    s1_s2_window_fac = straxen.URLConfig(
        default=1.1, type=(int, float),
        help="window to look for pS2 after S1 in multiples of full drift time,"
             "slightly larger to incorporate spread uncertainties of e- drifting"
    )

    after_s1_window_ext_fac = straxen.URLConfig(
        default=1.0, type=(int, float),
        help="extend scanning window after identified large S1s by this much full drift time,"
             "if no S2 is found in s1_s2_window"
    )

    after_s2_window_ext_fac = straxen.URLConfig(
        default=2.0, type=(int, float),
        help="extend scanning window after identified primary S2 by this much full drift time"
    )

    def infer_dtype(self):
        dtype = [('subtype', np.int16, 'time subtyping of peaks')] + strax.time_fields
        return dtype

    def setup(self):
        self.drift_time_max = int(self.max_drift_length / self.electron_drift_velocity)
        self.se_gain = self.ref_se_gain['all_tpc']
        self.se_span = self.ref_se_span['all_tpc']
        self.large_s2_threshold = self.large_s2_threshold_ne * self.se_gain
        self.large_s2_threshold += np.sqrt(self.large_s2_threshold_ne) * self.se_span
        self.mis_s2_threshold = self.mis_s2_fac * self.se_gain
        self.s1_s2_window = self.s1_s2_window_fac * self.drift_time_max
        self.after_s1_window_ext = self.after_s1_window_ext_fac * self.drift_time_max
        self.after_s2_window_ext = self.after_s2_window_ext_fac * self.drift_time_max

    def get_window_size(self):
        return 10 * max(self.s1_s2_window, self.after_s1_window_ext, self.after_s2_window_ext)

    @staticmethod
    def mark_with_s1(
        peaks, subtype_dtype,
        mis_s2_threshold, large_s2_threshold, other_large_s2_fac,
        s1_s2_window, after_s1_window_ext, after_s2_window_ext):
        '''
        Look after each S1s and classify:
        1. mark the largest peak within s1_s2_window as pS2 (exceeding large_s2_threshold)
        2. extend window after pS2
        3. mark all other large S2s as olS2. 
        4. mark all other small S2s as photoionization (PH)

        if pS2 identified, extend the window to after_s2_window_ext after such pS2 and mark S2s:
        1. other large S2 if an S2 is larger than large_s2_threshold and half the pS2 size
        2. photoionization if not olS2

        The order of subtyping assignment is important
        and reveals the priorities of the subtypings
        '''

        mask = np.ones(len(peaks), dtype=subtype_dtype)
        type_1_mask = (peaks['type'] == 1)
        type_2_mask = (peaks['type'] == 2)
        s1_area = peaks['area'][type_1_mask]
        s2_area = peaks['area'][type_2_mask]
        s1_subtype = np.ones(type_1_mask.sum(), dtype=subtype_dtype) * PeakSubtyping.Undefined
        s2_subtype = np.ones(type_2_mask.sum(), dtype=subtype_dtype) * PeakSubtyping.Undefined

        # Define S1-S2 pairing window
        _peaks = peaks[type_2_mask].copy()
        _peaks['time'] = _peaks['center_time']
        _peaks['endtime'] = _peaks['time']
        _peaks = np.sort(_peaks, order='time')

        _window = np.zeros(type_1_mask.sum(), dtype=strax.time_fields)
        _window['time'] = peaks['center_time'][type_1_mask]
        _window['endtime'] = _window['time'] + s1_s2_window
        tw_s1_s2 = strax.touching_windows(_peaks, _window)

        # Find index of max S2 following S1
        s2_max_index = np.zeros(len(_window), dtype=np.int32)
        for i, tw_12 in enumerate(tw_s1_s2):
            if tw_12[1] - tw_12[0] == 0 or np.all(np.isnan(s2_area[tw_12[0]:tw_12[1]])):
                s2_max_index[i] = -1
            else:
                # do we have pS2 or a bunch of small S2s
                s2_max_index[i] = np.nanargmax(s2_area[tw_12[0]:tw_12[1]]) + tw_12[0]

        # Assign the pS2 first, this might be overwritten by other subtyping later
        pS2_index = np.arange(type_2_mask.sum())[s2_max_index[s2_max_index != -1]]
        s2_subtype[pS2_index] = PeakSubtyping.pS2

        no_pS2_mask = (s2_max_index == -1)
        s2_max_area = s2_area[s2_max_index]
        s2_max_area[no_pS2_mask] = np.nan
        s2_max_time = _peaks['center_time'][s2_max_index]

        # Identify the loneS1, S1 and potential misclassified SE
        lone_s1 = tw_s1_s2[:, 1] - tw_s1_s2[:, 0] == 0
        mis_s1_mask = s1_area < mis_s2_threshold
        large_ps2_mask = s2_max_area >= large_s2_threshold
        s1_subtype[~lone_s1] = PeakSubtyping.S1
        s1_subtype[(mis_s1_mask & lone_s1) | (mis_s1_mask & ~large_ps2_mask)] = PeakSubtyping.sloneS1
        s1_subtype[(~mis_s1_mask & lone_s1) | (~mis_s1_mask & ~large_ps2_mask)] = PeakSubtyping.loneS1

        # Assign S1PH and slS1PH if pS2 not found
        # S1 following window after S1 if pS2 not found
        _window = np.zeros(type_1_mask.sum(), dtype=strax.time_fields)
        _window['time'] = peaks['center_time'][type_1_mask]
        _window['endtime'] = _window['time'] + after_s1_window_ext
        tw_after_s1 = strax.touching_windows(_peaks, _window)

        S1PH_index = np.hstack([
            np.arange(tw_a1[0], tw_a1[1]) for tw_a1 in tw_after_s1[~large_ps2_mask & ~mis_s1_mask]])
        slS1PH_index = np.hstack([
            np.arange(tw_a1[0], tw_a1[1]) for tw_a1 in tw_after_s1[~large_ps2_mask & mis_s1_mask]])
        S1PH_index = np.unique(S1PH_index)
        slS1PH_index = np.unique(slS1PH_index)
        s2_subtype[S1PH_index] = PeakSubtyping.S1PH
        s2_subtype[slS1PH_index] = PeakSubtyping.slS1PH

        # Assign S1PH and S1olS2 after S1 before pS2
        # S1 following window, to identify S1 photoionization
        _window = np.zeros((~no_pS2_mask).sum(), dtype=strax.time_fields)
        _window['time'] = peaks['center_time'][type_1_mask][~no_pS2_mask]
        _window['endtime'] = s2_max_time[~no_pS2_mask]
        tw_after_s1_before_s2 = strax.touching_windows(_peaks, _window)

        new_s2_max_area = s2_max_area[~no_pS2_mask]
        large_ps2_mask = new_s2_max_area >= large_s2_threshold
        small_s2s = np.hstack([
            s2_area[tw_a1b2[0]:tw_a1b2[1]] < np.max([
                large_s2_threshold, other_large_s2_fac * s2_a]) for tw_a1b2, s2_a in zip(
            tw_after_s1_before_s2[large_ps2_mask], new_s2_max_area[large_ps2_mask])])
        after_s1_before_s2_index = np.hstack([
            np.arange(tw_a1b2[0], tw_a1b2[1]) for tw_a1b2 in tw_after_s1_before_s2[large_ps2_mask]])
        S1PH_index = np.unique(after_s1_before_s2_index[small_s2s])
        S1olS2_index = np.unique(after_s1_before_s2_index[~small_s2s])
        s2_subtype[S1PH_index] = PeakSubtyping.S1PH
        s2_subtype[S1olS2_index] = PeakSubtyping.S1olS2

        # Assign S2PH and S2olS2 after pS2
        # S2 following window, to identify S2 photoionization
        s2_max_time_argsort = np.argsort(s2_max_time[~no_pS2_mask])
        # Need to make sure that the container is sorted in time
        _window = np.zeros((~no_pS2_mask).sum(), dtype=strax.time_fields)
        _window['time'] = s2_max_time[~no_pS2_mask][s2_max_time_argsort]
        _window['endtime'] = _window['time'] + after_s2_window_ext
        tw_after_s1_after_s2 = strax.touching_windows(_peaks, _window)

        sorted_s2_max_area = s2_max_area[~no_pS2_mask][s2_max_time_argsort]
        large_ps2_mask = sorted_s2_max_area >= large_s2_threshold
        small_s2s = np.hstack([
            s2_area[tw_a1a2[0]:tw_a1a2[1]] < np.max([
                large_s2_threshold, other_large_s2_fac * s2_a]) for tw_a1a2, s2_a in zip(
            tw_after_s1_after_s2[large_ps2_mask], sorted_s2_max_area[large_ps2_mask])])
        after_s1_after_s2_index = np.hstack([
            np.arange(tw_a1a2[0], tw_a1a2[1]) for tw_a1a2 in tw_after_s1_after_s2[large_ps2_mask]])
        S2PH_index = np.unique(after_s1_after_s2_index[small_s2s])
        S2olS2_index = np.unique(after_s1_after_s2_index[~small_s2s])
        s2_subtype[S2PH_index] = PeakSubtyping.S2PH
        s2_subtype[S2olS2_index] = PeakSubtyping.S2olS2

        mask[type_1_mask] = s1_subtype
        mask[type_2_mask] = s2_subtype
        return mask

    def mark_s2s(self, peaks, mask):
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
            if start_s2_a >= self.large_s2_threshold:
                # assign this S2 as lonepS2
                dmask_tmp = dmask[dmask == PeakSubtyping.Undefined]
                dmask_tmp[0] = PeakSubtyping.lonepS2
                dmask[dmask == PeakSubtyping.Undefined] = dmask_tmp
                # mark upto 2 fdt into olS2 or PH
                whole_window_cond = (pt > start_s2_t) & (
                    pt <= start_s2_t + self.after_s2_window_ext)
                mask_tmp = dmask[whole_window_cond]
                small_ps = pa[whole_window_cond] < self.large_s2_threshold
                mask_tmp[~small_ps] = PeakSubtyping.S2olS2
                mask_tmp[small_ps] = PeakSubtyping.S2PH
                dmask[whole_window_cond] = mask_tmp
            else:
                # A small S2 might actually be an S1, so here I also mark suspicious couplings
                ps2_cand_cond = (pt > start_s2_t) & (pt <= start_s2_t + self.s1_s2_window)
                if len(pa[ps2_cand_cond]):
                    # do we have pS2 or a bunch of small S2s
                    max_ind = np.nanargmax(pa[ps2_cand_cond])
                    max_val = pa[ps2_cand_cond][max_ind]
                    max_time = pt[ps2_cand_cond][max_ind]
                    if max_val >= self.large_s2_threshold:
                        # a suspicious coupling spotted. label this s2 as fakeS2
                        dmask_tmp = dmask[dmask == PeakSubtyping.Undefined]
                        dmask_tmp[0] = PeakSubtyping.fakeS2
                        dmask[dmask == PeakSubtyping.Undefined] = dmask_tmp
                        whole_window_cond = (pt > start_s2_t)
                        whole_window_cond &= (pt <= max_time + self.after_s2_window_ext)
                        mask_tmp = dmask[whole_window_cond]
                        small_pa = pa[whole_window_cond] < np.max([
                            self.large_s2_threshold, self.other_large_s2_fac * max_val])
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
        result = np.zeros(len(peaks), self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = strax.endtime(peaks)
        result['subtype'] = np.ones(len(peaks)) * PeakSubtyping.Undefined
        result['subtype'][peaks['type'] == 0] = PeakSubtyping.Junk

        undefined_mask = result['subtype'] == PeakSubtyping.Undefined
        result['subtype'][undefined_mask] = self.mark_with_s1(
            peaks[undefined_mask],
            self.subtype_dtype,
            self.mis_s2_threshold, self.large_s2_threshold, self.other_large_s2_fac,
            self.s1_s2_window, self.after_s1_window_ext, self.after_s2_window_ext)

        # self.mark_with_s1(peaks, mask, self.s1_s2_window, self.mis_s2_threshold, self.large_s2_threshold, self.after_s1_window_ext_fac, self.after_s2_window_ext_fac, self.drift_time_max, self.other_large_s2_fac)
        # self.mark_s2s(peaks, mask)

        # n_undefined = (mask == PeakSubtyping.Undefined).sum()
        # if n_undefined:
        #     raise RuntimeError(f'We still find {n_undefined} peaks undefined!')

        return result
