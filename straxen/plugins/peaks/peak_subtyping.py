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

    Note: 
      In this plugin, the timing of peak is only defined by their center_time.
      And the length of peaks are zero.
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
        # we need to set those window sizes as integers because they are used to define containers
        self.s1_s2_window = int(self.s1_s2_window_fac * self.drift_time_max)
        self.after_s1_window_ext = int(self.after_s1_window_ext_fac * self.drift_time_max)
        self.after_s2_window_ext = int(self.after_s2_window_ext_fac * self.drift_time_max)

    def get_window_size(self):
        return 10 * max(self.s1_s2_window, self.after_s1_window_ext, self.after_s2_window_ext)

    @staticmethod
    def mark_peaks_after_s1s(
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

        # prepare for referred variables
        mask = np.ones(len(peaks), dtype=subtype_dtype) * PeakSubtyping.Undefined
        type_1_mask = (peaks['type'] == 1)
        type_2_mask = (peaks['type'] == 2)

        _s1_area = peaks['area'][type_1_mask]
        _s2_area = peaks['area'][type_2_mask]

        base_times = peaks['center_time']
        base_areas = peaks['area']
        base_indices = np.arange(len(peaks))

        # find S1-pS2 pairing
        _peaks = peaks[type_2_mask].copy()
        # Only use center_time because we used center_time to define the drift time
        _peaks['time'] = _peaks['center_time']
        # touchng_windows needs non-zero length of things
        _peaks['endtime'] = _peaks['time'] + 1

        _window = np.zeros(type_1_mask.sum(), dtype=strax.time_fields)
        _window['time'] = peaks['center_time'][type_1_mask]
        _window['endtime'] = _window['time'] + s1_s2_window
        # Here the _window can be overlapping
        tw_s1_s2 = strax.touching_windows(_peaks, _window)

        # Find index of max S2 of each group of S2s following S1
        _ps2_max_indices = np.ones(len(_window), dtype=np.int32) * -1
        for i, tw_12 in enumerate(tw_s1_s2):
            if tw_12[1] - tw_12[0] == 0:
                continue
            else:
                # do we have pS2 or a bunch of small S2s
                _ps2_max_indices[i] = np.argmax(_s2_area[tw_12[0]:tw_12[1]]) + tw_12[0]
        _no_pS2_mask = (_ps2_max_indices == -1)
        ps2_area = _s2_area[_ps2_max_indices]
        # if there is no pS2, set the area to be nan
        ps2_area[_no_pS2_mask] = np.nan

        # identify the loneS1, S1 and potential misclassified SE,
        # if the S2 is not large enough to be a pS2
        _large_ps2_mask = ps2_area >= large_s2_threshold
        # if the S1 is a lone S1, it has no pS2 or no large enough pS2
        lone_s1 = _no_pS2_mask | ~_large_ps2_mask
        # if the S1 is a misclassified S2
        _mis_s1_mask = _s1_area < mis_s2_threshold
        # get indices of loneS1, sloneS2, S1, pS2
        loneS1_indices = base_indices[type_1_mask][lone_s1 & ~_mis_s1_mask]
        sloneS1_indices = base_indices[type_1_mask][lone_s1 & _mis_s1_mask]
        S1_indices = base_indices[type_1_mask][~lone_s1]
        # sometimes the same pS2 can be paired up with multiple S1s
        # also need to make sure that the pS2_indices are sorted, but np.unique does that
        _ps2_max_indices = np.unique(_ps2_max_indices[~lone_s1])
        pS2_indices = base_indices[type_2_mask][_ps2_max_indices]

        # limitation of window size
        # when we see a leading peak who can cause the correlation,
        # the search of photon-ionization should stop
        limitation = np.sort(np.hstack(
            [
                base_times[loneS1_indices],
                base_times[sloneS1_indices],
                base_times[S1_indices],
                base_times[pS2_indices],
            ]
        ))

        # containers triggered by each leading peaks
        # here we start to divide the time line by non-overlapping containers
        loneS1_containers = limited_containers(
            base_times[loneS1_indices], after_s1_window_ext, limitation)
        sloneS1_containers = limited_containers(
            base_times[sloneS1_indices], after_s1_window_ext, limitation)
        S1_containers = limited_containers(
            base_times[S1_indices], s1_s2_window, limitation)
        pS2_containers = limited_containers(
            base_times[pS2_indices], after_s2_window_ext, limitation)

        # extra check, will be deleted after debugging
        for i_a, a in enumerate([loneS1_containers, sloneS1_containers, S1_containers, pS2_containers]):
            for i_b, b in enumerate([loneS1_containers, sloneS1_containers, S1_containers, pS2_containers]):
                if i_a == i_b:
                    continue
                r = strax.touching_windows(a, b)
                assert (r[:, 1] - r[:, 0]).max() == 0, ''\
                    + 'triggering containers should not be overlapping'

        _peaks = peaks.copy()
        _peaks['time'] = _peaks['center_time']
        # touchng_windows needs non-zero length of things
        _peaks['endtime'] = _peaks['time'] + 1

        # assign loneS1's & sloneS1's following peaks
        tw_after_s1 = strax.touching_windows(_peaks, loneS1_containers)
        S1PH_indices = combine_indices(tw_after_s1)
        mask[S1PH_indices] = PeakSubtyping.S1PH
        tw_after_s1 = strax.touching_windows(_peaks, sloneS1_containers)
        slS1PH_indices = combine_indices(tw_after_s1)
        mask[slS1PH_indices] = PeakSubtyping.slS1PH

        # assign S1's following peaks
        tw_after_s1_before_ps2 = strax.touching_windows(_peaks, S1_containers)
        _S1_small_s2s_thresholds = np.vstack(
            [
                np.full(len(S1_containers), large_s2_threshold),
                other_large_s2_fac * ps2_area[~lone_s1],
            ]
        ).max(axis=0)
        S1PH_indices, S1olS2_indices = combine_indices_ref(
            tw_after_s1_before_ps2, base_areas, _S1_small_s2s_thresholds)
        mask[S1PH_indices] = PeakSubtyping.S1PH
        mask[S1olS2_indices] = PeakSubtyping.S1olS2

        # assign pS2's following peaks
        tw_after_s1_after_ps2 = strax.touching_windows(_peaks, pS2_containers)
        _pS2_small_s2s_thresholds = np.vstack(
            [
                np.full(len(pS2_containers), large_s2_threshold),
                other_large_s2_fac * _s2_area[_ps2_max_indices],
            ]
        ).max(axis=0)
        S2PH_indices, S2olS2_indices = combine_indices_ref(
            tw_after_s1_after_ps2, base_areas, _pS2_small_s2s_thresholds)
        mask[S2PH_indices] = PeakSubtyping.S2PH
        mask[S2olS2_indices] = PeakSubtyping.S2olS2

        # assign loneS1, S1, sloneS2, pS2
        mask[loneS1_indices] = PeakSubtyping.loneS1
        mask[sloneS1_indices] = PeakSubtyping.sloneS1
        mask[S1_indices] = PeakSubtyping.S1
        mask[pS2_indices] = PeakSubtyping.pS2
        return mask

    @staticmethod
    def mark_other_s2s(
        peaks, subtype_dtype,
        large_s2_threshold, other_large_s2_fac,
        s1_s2_window, after_s2_window_ext):
        '''
        After marking all peaks after S1s, all that's left are S2s.
        One extra occasion is the fakeS2.
        If a small S2 is identified and a pS2 can be paired up with it, such small S2 is marked fakeS2.
        '''
        # prepare for referred variables
        mask = np.ones(len(peaks), dtype=subtype_dtype) * PeakSubtyping.Undefined
        type_1_mask = (peaks['area'] < large_s2_threshold)
        type_2_mask = (peaks['area'] >= large_s2_threshold)

        _s2_area = peaks['area'][type_2_mask]

        base_times = peaks['center_time']
        base_areas = peaks['area']
        base_indices = np.arange(len(peaks))

        # find lonepS2-pS2 pairing
        _peaks = peaks[type_2_mask].copy()
        # Only use center_time because we used center_time to define the drift time
        _peaks['time'] = _peaks['center_time']
        # touchng_windows needs non-zero length of things
        _peaks['endtime'] = _peaks['time'] + 1

        _window = np.zeros(type_1_mask.sum(), dtype=strax.time_fields)
        _window['time'] = peaks['center_time'][type_1_mask]
        _window['endtime'] = _window['time'] + s1_s2_window
        # Here the _window can be overlapping
        tw_s1_s2 = strax.touching_windows(_peaks, _window)

        # Find index of max S2 of each group of S2s following lonepS2
        _ps2_max_indices = np.ones(len(_window), dtype=np.int32) * -1
        for i, tw_12 in enumerate(tw_s1_s2):
            if tw_12[1] - tw_12[0] == 0:
                continue
            else:
                # do we have pS2 or a bunch of small S2s
                _ps2_max_indices[i] = np.argmax(_s2_area[tw_12[0]:tw_12[1]]) + tw_12[0]
        _no_pS2_mask = (_ps2_max_indices == -1)
        ps2_area = _s2_area[_ps2_max_indices]
        # if there is no pS2, set the area to be nan
        ps2_area[_no_pS2_mask] = np.nan

        # identify the lonepS2,
        # if the S2 is not large enough to be a pS2
        _large_ps2_mask = ps2_area >= large_s2_threshold
        # if the pS2 is a lone pS2, it has no pS2 or no large enough pS2
        lone_fakes2 = _no_pS2_mask | ~_large_ps2_mask
        # get indices of fakeS2, pS2, lonepS2
        fakeS2_indices = base_indices[type_1_mask][~lone_fakes2]
        # sometimes the same pS2 can be paired up with multiple slonepS2,
        # also need to make sure that the pS2_indices are sorted, but np.unique does that
        _ps2_max_indices = np.unique(_ps2_max_indices[~lone_fakes2])
        pS2_indices = base_indices[type_2_mask][_ps2_max_indices]
        type_2_base_indices = np.arange(type_2_mask.sum())
        _loneps2_max_indices = type_2_base_indices[
            ~np.isin(type_2_base_indices, _ps2_max_indices)]
        lonepS2_indices = base_indices[type_2_mask][_loneps2_max_indices]

        # limitation of window size
        # when we see a leading peak who can cause the correlation,
        # the search of photon-ionization should stop
        limitation = np.sort(np.hstack(
            [
                base_times[fakeS2_indices],
                base_times[pS2_indices],
                base_times[lonepS2_indices],
            ]
        ))

        # containers triggered by each leading peaks
        # here we start to divide the time line by non-overlapping containers
        fakeS2_containers = limited_containers(
            base_times[fakeS2_indices], s1_s2_window, limitation)
        pS2_containers = limited_containers(
            base_times[pS2_indices], after_s2_window_ext, limitation)
        lonepS2_containers = limited_containers(
            base_times[lonepS2_indices], after_s2_window_ext, limitation)

        # extra check, will be deleted after debugging
        for i_a, a in enumerate([fakeS2_containers, pS2_containers, lonepS2_containers]):
            for i_b, b in enumerate([fakeS2_containers, pS2_containers, lonepS2_containers]):
                if i_a == i_b:
                    continue
                r = strax.touching_windows(a, b)
                assert (r[:, 1] - r[:, 0]).max() == 0, ''\
                    + 'triggering containers should not be overlapping'

        _peaks = peaks.copy()
        _peaks['time'] = _peaks['center_time']
        # touchng_windows needs non-zero length of things
        _peaks['endtime'] = _peaks['time'] + 1

        # assign fakeS2's following peaks
        tw_after_fakes2_before_s2 = strax.touching_windows(_peaks, fakeS2_containers)
        _fakeS2_small_s2s_thresholds = np.vstack(
            [
                np.full(len(fakeS2_containers), large_s2_threshold),
                other_large_s2_fac * ps2_area[~lone_fakes2],
            ]
        ).max(axis=0)
        fakeS2_PH_indices, fakeS2_olS2_indices = combine_indices_ref(
            tw_after_fakes2_before_s2, base_areas, _fakeS2_small_s2s_thresholds)
        mask[fakeS2_PH_indices] = PeakSubtyping.fakeS2_PH
        mask[fakeS2_olS2_indices] = PeakSubtyping.fakeS2_olS2

        # assign pS2's following peaks
        tw_after_fakes2_after_s2 = strax.touching_windows(_peaks, pS2_containers)
        _pS2_small_s2s_thresholds = np.vstack(
            [
                np.full(len(pS2_containers), large_s2_threshold),
                other_large_s2_fac * _s2_area[_ps2_max_indices],
            ]
        ).max(axis=0)
        S2PH_indices, S2olS2_indices = combine_indices_ref(
            tw_after_fakes2_after_s2, base_areas, _pS2_small_s2s_thresholds)
        mask[S2PH_indices] = PeakSubtyping.S2PH
        mask[S2olS2_indices] = PeakSubtyping.S2olS2

        # assign lonepS2's following peaks
        tw_after_lonepS2 = strax.touching_windows(_peaks, lonepS2_containers)
        S2PH_indices, S2olS2_indices = combine_indices_ref(
            tw_after_lonepS2, base_areas, large_s2_threshold)
        mask[S2PH_indices] = PeakSubtyping.S2PH
        mask[S2olS2_indices] = PeakSubtyping.S2olS2

        # assign fakeS2, pS2, lonepS2
        mask[fakeS2_indices] = PeakSubtyping.fakeS2
        mask[pS2_indices] = PeakSubtyping.pS2
        mask[lonepS2_indices] = PeakSubtyping.lonepS2

        undefined_mask = (mask == PeakSubtyping.Undefined)
        mask[undefined_mask] = PeakSubtyping.DE
        return mask

    def compute(self, peaks):
        # Sort the peaks first by center_time
        peaks = np.sort(peaks, order='center_time')

        result = np.zeros(len(peaks), self.dtype)
        result['time'] = peaks['time']
        result['endtime'] = strax.endtime(peaks)
        result['subtype'] = np.ones(len(peaks)) * PeakSubtyping.Undefined
        junk_mask = (peaks['type'] != 1) & (peaks['type'] != 2)
        junk_mask |= np.isnan(peaks['area'])
        result['subtype'][junk_mask] = PeakSubtyping.Junk

        # mark with S1-pS2 pairing
        undefined_mask = result['subtype'] == PeakSubtyping.Undefined
        result['subtype'][undefined_mask] = self.mark_peaks_after_s1s(
            peaks[undefined_mask],
            self.subtype_dtype,
            self.mis_s2_threshold, self.large_s2_threshold, self.other_large_s2_fac,
            self.s1_s2_window, self.after_s1_window_ext, self.after_s2_window_ext)

        # mark with lonepS2 and fakeS2-pS2 pairing
        undefined_mask = result['subtype'] == PeakSubtyping.Undefined
        result['subtype'][undefined_mask] = self.mark_other_s2s(
            peaks[undefined_mask],
            self.subtype_dtype,
            self.large_s2_threshold, self.other_large_s2_fac,
            self.s1_s2_window, self.after_s2_window_ext)

        # check if there is any undefined peaks
        n_undefined = (result['subtype'] == PeakSubtyping.Undefined).sum()
        if n_undefined:
            raise RuntimeError(f'We still find {n_undefined} peaks undefined!')

        # sort the result by time
        result = np.sort(result, order='time')

        return result


def limited_containers(start, length, limitation):
    """
    Build container given start, length and limitation of end time.
    The retuned containers should not be overlapping
    :param start: sorted array of interval start points
    :param length: length of each interval, len(start) == len(length)
    """
    assert np.all(length >= 0)
    if not hasattr(length, '__len__'):
        length = np.full(len(start), length)
    # limitations must be sorted,
    # and it is a combination of given limitation and start
    # because we need to make sure the returned containers are not overlapping
    limitation = np.unique(np.hstack([start, limitation]))
    containers = np.zeros(len(start), dtype=strax.time_fields)
    containers['time'] = start
    containers['endtime'] = _limited_containers(start, length, limitation)
    assert np.all(containers['time'][1:] - containers['endtime'][:-1] >= 0)
    return containers


@numba.njit
def _limited_containers(start, length, limitation):
    """
    Construct the endtime in jit accelerated way
    """
    endtime = start + length
    n = len(limitation)
    i = 0
    for container_i, (time_i, endtime_i) in enumerate(zip(start, endtime)):
        while i < n - 1 and time_i >= limitation[i]:
            i += 1
        if limitation[i] < endtime_i:
            endtime[container_i] = limitation[i]
    return endtime


def combine_indices(result):
    """
    Combine the indices from touching_windows results
    :param result: touching_windows results, each row is a pair of indices
    """
    indices = []
    for r in result:
        indices.append(np.arange(r[0], r[1]))
    indices = np.unique(np.hstack(indices))
    return indices


def combine_indices_ref(result, areas, reference_areas):
    """
    Combine the indices from touching_windows results, based on the areas.
    If the areas larger than the reference areas, combine them into large_indices,
    if not, combine them into small_indices.
    :param result: touching_windows results, each row is a pair of indices
    :param areas: areas of the peaks of the cooresponding indices
    :param reference_areas: reference areas to compare with for each pair of indices
    """
    if not hasattr(reference_areas, '__len__'):
        reference_areas = np.full(len(result), reference_areas)
    assert len(result) == len(reference_areas), ''\
        + 'result and reference_areas must have the same length'
    small_indices = []
    large_indices = []
    for r, ref in zip(result, reference_areas):
        small_indices.append(np.arange(r[0], r[1])[areas[r[0]:r[1]] <= ref])
        large_indices.append(np.arange(r[0], r[1])[areas[r[0]:r[1]] > ref])
    small_indices = np.unique(np.hstack(small_indices))
    large_indices = np.unique(np.hstack(large_indices))
    return small_indices, large_indices
