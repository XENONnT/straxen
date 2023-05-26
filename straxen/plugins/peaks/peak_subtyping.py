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
    save_when = strax.SaveWhen.EXPLICIT
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
        dtype = strax.time_fields + [
            ('subtype', self.subtype_dtype, 'subtyping of peaks')]
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
    def pair_lone_s1_and_ps2(peaks, type_1_mask, type_2_mask, extension, large_s2_threshold):
        # find S1-pS2 pairing
        s1_window = np.zeros(type_1_mask.sum(), dtype=strax.time_fields)
        s1_window['time'] = peaks['center_time'][type_1_mask]
        s1_window['endtime'] = s1_window['time'] + extension

        s2_peaks = peaks[type_2_mask].copy()
        # Only use center_time because we used center_time to define the drift time
        s2_peaks['time'] = s2_peaks['center_time']
        # touchng_windows needs non-zero length of things
        s2_peaks['endtime'] = s2_peaks['time'] + 1

        # Here the s1_window can be overlapping
        tw_s1_ps2 = strax.touching_windows(s2_peaks, s1_window)
        # Find index of max S2 of each group of S2s following S1
        _ps2_max_indices = np.ones(len(s1_window), dtype=np.int32) * -1
        for i, tw_12 in enumerate(tw_s1_ps2):
            if tw_12[1] - tw_12[0] == 0:
                continue
            else:
                # do we have pS2 or a bunch of small S2s
                index_s2_with_max_area = np.argmax(
                    s2_peaks['area'][tw_12[0]:tw_12[1]]) + tw_12[0]
                _ps2_max_indices[i] = index_s2_with_max_area
        _no_pS2_mask = (_ps2_max_indices == -1)
        ps2_area = s2_peaks['area'][_ps2_max_indices]
        # if there is no pS2, set the area to be nan
        ps2_area[_no_pS2_mask] = np.nan

        # identify the loneS1, S1 and potential misclassified SE,
        # if the S2 is not large enough to be a pS2
        _large_ps2_mask = ps2_area >= large_s2_threshold
        # if the S1 is a lone S1, it has no pS2 or no large enough pS2
        lone_S1_mask = _no_pS2_mask | ~_large_ps2_mask
        max_areas_after_S1 = ps2_area[~lone_S1_mask]

        # sometimes the same pS2 can be paired up with multiple S1s
        _no_lones1_ps2_max_indices = _ps2_max_indices[~lone_S1_mask]
        # also need to make sure that the pS2_indices are sorted, but np.unique does that
        _ps2_max_indices = np.unique(_no_lones1_ps2_max_indices)

        # if pS2 sits between an S1-pS2 pair, it is not a pS2!
        _ps2_peaks = np.zeros(len(_ps2_max_indices), dtype=strax.time_fields)
        # peaks' times are the pS2s' times
        _ps2_peaks['time'] = s2_peaks['time'][_ps2_max_indices]
        _ps2_peaks['endtime'] = _ps2_peaks['time'] + 1
        _window = np.zeros((~lone_S1_mask).sum(), dtype=strax.time_fields)
        # window's times are the S1s' times
        _window['time'] = s1_window['time'][~lone_S1_mask]
        # window's endtimes are the pS2s' times
        _window['endtime'] = s2_peaks['time'][_no_lones1_ps2_max_indices]
        tw_s1_ps2 = strax.touching_windows(_ps2_peaks, _window)
        _not_pS2_indices = combine_indices(tw_s1_ps2)
        _pS2_base_indices = np.arange(len(_ps2_max_indices))
        _are_pS2_indices = _pS2_base_indices[
            ~np.isin(_pS2_base_indices, _not_pS2_indices)]

        # update the pS2 indices, remove the ones that covered by S1-pS2 pair
        pS2_indices = _ps2_max_indices[_are_pS2_indices]
        return lone_S1_mask, max_areas_after_S1, pS2_indices

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

        base_times = peaks['center_time']
        base_areas = peaks['area']
        base_indices = np.arange(len(peaks))

        lone_S1_mask, max_areas_after_S1, pS2_indices = PeaksSubtypes.pair_lone_s1_and_ps2(
            peaks, type_1_mask, type_2_mask, s1_s2_window, large_s2_threshold)

        # get indices of pS2, loneS1, sloneS1, S1
        pS2_indices = base_indices[type_2_mask][pS2_indices]

        # if the S1 is a misclassified S2
        _mis_s1_mask = peaks['area'][type_1_mask] < mis_s2_threshold
        # get indices of loneS1, sloneS2, S1, pS2
        loneS1_indices = base_indices[type_1_mask][lone_S1_mask & ~_mis_s1_mask]
        sloneS1_indices = base_indices[type_1_mask][lone_S1_mask & _mis_s1_mask]
        S1_indices = base_indices[type_1_mask][~lone_S1_mask]

        _peaks = peaks.copy()
        _peaks['time'] = _peaks['center_time']
        # touchng_windows needs non-zero length of things
        _peaks['endtime'] = _peaks['time'] + 1

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
        # assign loneS1's & sloneS1's following peaks
        S1PH_indices = find_indices_after(
            base_times[loneS1_indices], _peaks, after_s1_window_ext, limitation)
        mask[S1PH_indices] = PeakSubtyping.S1PH
        slS1PH_indices = find_indices_after(
            base_times[sloneS1_indices], _peaks, after_s1_window_ext, limitation)
        mask[slS1PH_indices] = PeakSubtyping.slS1PH

        # assign S1's following peaks
        # here the areas are fractions of potential pS2s, but not the real pS2s
        # because some pS2s are filtered out
        S1PH_indices, S1olS2_indices = find_indices_after_ref(
            base_times[S1_indices], peaks, s1_s2_window,
            limitation, large_s2_threshold, other_large_s2_fac * max_areas_after_S1)
        mask[S1PH_indices] = PeakSubtyping.S1PH
        mask[S1olS2_indices] = PeakSubtyping.S1olS2

        max_areas_after_pS2 = base_areas[pS2_indices]
        # assign pS2's following peaks
        S2PH_indices, S2olS2_indices = find_indices_after_ref(
            base_times[pS2_indices], peaks, after_s2_window_ext,
            limitation, large_s2_threshold, other_large_s2_fac * max_areas_after_pS2)
        mask[S2PH_indices] = PeakSubtyping.S2PH
        mask[S2olS2_indices] = PeakSubtyping.S2olS2

        # assign loneS1, sloneS1, S1, pS2
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
        If a small S2 is identified and a pS2 can be paired up with it,
        such small S2 is marked fakeS2.
        '''
        # prepare for referred variables
        mask = np.ones(len(peaks), dtype=subtype_dtype) * PeakSubtyping.Undefined

        base_times = peaks['center_time']
        base_areas = peaks['area']
        base_indices = np.arange(len(peaks))

        # find fakeS2
        _ps2_mask = (peaks['area'] >= large_s2_threshold)
        _fakes2_mask = ~_ps2_mask

        # find fakeS2-pS2 pairing
        _fakes2_peaks = peaks[_fakes2_mask].copy()
        # Only use center_time because we used center_time to define the drift time
        _fakes2_peaks['time'] = _fakes2_peaks['center_time']
        # touchng_windows needs non-zero length of things
        _fakes2_peaks['endtime'] = _fakes2_peaks['time'] + 1

        # fakeS2s should not live within after_s2_window_ext after a large S2
        fakeS2_containers = limited_containers(
            base_times[_ps2_mask], s1_s2_window,
            base_times[_ps2_mask] + after_s2_window_ext, forward=False)

        tw_fakes2_ps2 = strax.touching_windows(_fakes2_peaks, fakeS2_containers)

        _no_fakeS2_mask = (tw_fakes2_ps2[:, 1] - tw_fakes2_ps2[:, 0] == 0)
        _fakes2_ps2_indices = np.unique(tw_fakes2_ps2[:, 0][~_no_fakeS2_mask])
        # fakeS2 are those before large S2s
        _fakes2_indices = base_indices[_fakes2_mask][_fakes2_ps2_indices]

        # repeat what we have done in the first function
        type_1_mask = _fakes2_mask & np.isin(np.arange(len(peaks)), _fakes2_indices)
        type_2_mask = ~type_1_mask

        lone_fakeS2_mask, max_areas_after_fakeS2, pS2_indices = PeaksSubtypes.pair_lone_s1_and_ps2(
            peaks, type_1_mask, type_2_mask, s1_s2_window, large_s2_threshold)

        # get indices of pS2, loneS1, sloneS1, S1
        pS2_indices = base_indices[type_2_mask][pS2_indices]
        fakeS2_indices = base_indices[type_1_mask][~lone_fakeS2_mask]
        assert np.all(base_areas[pS2_indices] > large_s2_threshold)

        # limitation of window size
        # when we see a leading peak who can cause the correlation,
        # the search of photon-ionization should stop
        limitation = np.sort(np.hstack(
            [
                base_times[fakeS2_indices],
                base_times[pS2_indices],
            ]
        ))

        _peaks = peaks.copy()
        _peaks['time'] = _peaks['center_time']
        # touchng_windows needs non-zero length of things
        _peaks['endtime'] = _peaks['time'] + 1

        # assign fakeS2's following peaks
        fakeS2_PH_indices, fakeS2_olS2_indices = find_indices_after_ref(
            base_times[fakeS2_indices], peaks, s1_s2_window,
            limitation, large_s2_threshold, other_large_s2_fac * max_areas_after_fakeS2)
        mask[fakeS2_PH_indices] = PeakSubtyping.fakeS2_PH
        mask[fakeS2_olS2_indices] = PeakSubtyping.fakeS2_olS2

        max_areas_after_pS2 = base_areas[pS2_indices]
        # assign pS2's following peaks
        S2PH_indices, S2olS2_indices = find_indices_after_ref(
            base_times[pS2_indices], peaks, after_s2_window_ext,
            limitation, large_s2_threshold, other_large_s2_fac * max_areas_after_pS2)
        mask[S2PH_indices] = PeakSubtyping.S2PH
        mask[S2olS2_indices] = PeakSubtyping.S2olS2

        # what else? undefined large peaks are lonepS2
        lonepS2_indices = (mask == PeakSubtyping.Undefined) & _ps2_mask

        # assign lonepS2's following peaks
        S2PH_indices, S2olS2_indices = find_indices_after_ref(
            base_times[lonepS2_indices], peaks, after_s2_window_ext,
            limitation, large_s2_threshold, large_s2_threshold)
        mask[S2PH_indices] = PeakSubtyping.S2PH
        mask[S2olS2_indices] = PeakSubtyping.S2olS2

        # assign fakeS2, pS2, lonepS2
        mask[fakeS2_indices] = PeakSubtyping.fakeS2
        mask[pS2_indices] = PeakSubtyping.pS2
        mask[lonepS2_indices] = PeakSubtyping.lonepS2

        # remaining small S2s are DE
        undefined_mask = (mask == PeakSubtyping.Undefined)
        mask[undefined_mask] = PeakSubtyping.DE
        return mask

    def compute(self, peaks):
        # Sort the peaks first by center_time
        # but remember the original order
        argsort = np.argsort(peaks['center_time'])
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

        # sort the result by time
        _result = result.copy()
        result[argsort] = _result

        return result


def find_indices_after(anchor, peaks, extension, limitation):
    containers = limited_containers(
        anchor, extension, limitation)
    indices_widows = strax.touching_windows(peaks, containers)
    peaks_indices = combine_indices(indices_widows)
    return peaks_indices


def find_indices_after_ref(
        anchor, peaks, extension, 
        limitation, common_threshold, reference_areas):
    # assign S1's following peaks
    containers = limited_containers(
        anchor, extension, limitation)
    if not hasattr(reference_areas, '__len__'):
        reference_areas = np.full(len(anchor), reference_areas)
    else:
        reference_areas = np.vstack(
            [
                np.full(len(containers), common_threshold),
                reference_areas,
            ]
        ).max(axis=0)
    indices_widows = strax.touching_windows(peaks, containers)
    small_indices, large_indices = combine_indices_ref(
        indices_widows, peaks['area'], reference_areas)
    return small_indices, large_indices


def limited_containers(anchor, length, limitation, forward=True):
    """
    Build container given anchor, length and limitation of end time.
    The retuned containers should not be overlapping
    :param anchor: sorted array of interval anchor points
    :param length: length of each interval, len(anchor) == len(length)
    """
    assert np.all(length >= 0)
    if not hasattr(length, '__len__'):
        length = np.full(len(anchor), length)
    # limitations must be sorted,
    # and it is a combination of given limitation and anchor
    # because we need to make sure the returned containers are not overlapping
    # prevent numpy change the dtype to float
    # limitation needs also be sorted
    if len(limitation) > 0:
        limitation = np.unique(np.hstack([anchor, limitation]))
    else:
        limitation = anchor.copy()
    containers = np.zeros(len(anchor), dtype=strax.time_fields)
    if forward:
        containers['time'] = anchor
        containers['endtime'] = _limited_containers_forward(anchor, length, limitation)
    else:
        containers['endtime'] = anchor
        containers['time'] = _limited_containers_reverse(anchor, length, limitation)
    assert np.all(containers['time'][1:] - containers['endtime'][:-1] >= 0)
    return containers


@numba.njit
def _limited_containers_forward(time, length, limitation):
    """
    Construct the endtime in jit accelerated way
    """
    endtime = time + length
    n = len(limitation)
    i = 0
    for j in range(len(time)):
        while i < n - 1 and time[j] >= limitation[i]:
            i += 1
        endtime[j] = min(endtime[j], limitation[i])
    return endtime


@numba.njit
def _limited_containers_reverse(endtime, length, limitation):
    """
    Construct the time in jit accelerated way
    """
    time = endtime - length
    n = len(limitation)
    i = n - 1
    for j in np.arange(len(endtime))[::-1]:
        while i > 0 and endtime[j] <= limitation[i]:
            i -= 1
        time[j] = max(time[j], limitation[i])
    return time


def combine_indices(result):
    """
    Combine the indices from touching_windows results
    :param result: touching_windows results, each row is a pair of indices
    """
    if len(result) == 0:
        return np.empty(0, dtype=np.int)
    length = np.sum(result[:, 1] - result[:, 0])
    indices = np.zeros(length, dtype=np.int64)
    i = 0
    for r in result:
        l = r[1] - r[0]
        segment = np.arange(r[0], r[1])
        indices[i : i + l] = segment
        i += l
    indices = np.unique(indices[:i])
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
    if len(result) == 0:
        return np.array([], dtype=np.int), np.array([], dtype=np.int)
    if not hasattr(reference_areas, '__len__'):
        reference_areas = np.full(len(result), reference_areas)
    assert len(result) == len(reference_areas), ''\
        + 'result and reference_areas must have the same length'
    length = np.sum(result[:, 1] - result[:, 0])
    indices = np.zeros(length, dtype=np.int64)
    sizemask = np.zeros(length, dtype=bool)
    i = 0
    for r, ref in zip(result, reference_areas):
        l = r[1] - r[0]
        segment = np.arange(r[0], r[1])
        indices[i : i + l] = segment
        sizemask[i : i + l] = (areas[r[0]:r[1]] <= ref)
        i += l
    small_indices = np.unique(indices[:i][sizemask[:i]])
    large_indices = np.unique(indices[:i][~sizemask[:i]])
    return small_indices, large_indices
