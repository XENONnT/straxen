import numpy as np
import strax
from strax.processing.peak_splitting import natural_breaks_gof

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('divide_90p_width', default=10,
                 help="The peak is smoothed by dividing the 90p width by"
                      "this number, and coverting it into number of samples."
                      "This is then the 'n' used in the smoothing kernel"
                      "shown below."),
    strax.Option('smoothing_power', default=3,
                 help="The power used in the smoothing filter with a kernel of"
                      "(1-(x/n)^p)^p, where p is the power"),
    strax.Option('percentage_threshold', default=0.1,
                 help="The height threshold for the peak as a percentage"
                      "of the maximum, used to reject the low parts"
                      "of the peak in order to find the local extrema."),
    strax.Option('percent_valley_height', default=0.9,
                 help="The percentage of the valley height of the deepest"
                      "valley for which to calculate the valley width"),
)
class LocalMinimumInfo(strax.LoopPlugin):
    """
    Looks for the main S2 peak in an event, finds the local minimum (if there is one),
    and looks to compute some figures of merit such as the max goodness of split,
    width of the valley, and height of the valley.
    """
    depends_on = ('event_basics', 'peaks')
    provides = 'local_min_info_2'
    parallel = 'process'
    compressor = 'zstd'

    __version__ = '0.1.0'
    dtype = strax.time_fields + [(('Maximum Goodness of Split',
                                   's2_max_gos'), np.float32),
                                 (('Number of local maxima of the smoothed peak',
                                   's2_num_loc_max'), np.int16),
                                 (('Full gap at p% of the valley height of the deepest valley [ns],'
                                   'by default p = 90', 's2_valley_gap'), np.float32),
                                 (('Valley depth over max height of the deepest valley',
                                   's2_valley_height_ratio'), np.float32)]

    def compute_loop(self, event, peaks):
        """
        This finds the maxima and minima for the main S2 peak and calculates its info such as
        the number of local maxima, the depth of the deepest local minimum over the maximum height
        of the peak (s2_valley_height_ratio), and the width of the local minimum valley at
        90% of the valley height.
        :param event: The event
        :param peaks: The peaks belonging to the event, only the main S2 peak is considered,
        if there is none, this plugin returns none
        :return: Returns a dictionary containing all of the fields above for each main S2 peak
        as well as the timing information.

        """
        max_gos = np.nan
        num_loc_maxes = 0
        valley_gap = np.nan
        valley_height_ratio = np.nan

        if event['s2_area'] > 0:
            p = peaks[event['s2_index']]

            smoothing_number = int(p['width'][9] / (self.config['divide_90p_width'] * p['dt'])) + 1
            smoothed_peak = power_smooth(p['data'][:p['length']],
                                         smoothing_number,
                                         self.config['smoothing_power'])

            # Set data below precentage threshold on both side to zeros
            left, right = bounds_above_percentage_height(smoothed_peak, self.config['percentage_threshold'])

            # Maximum GOS calculation for data above percentage
            max_gos = np.max(natural_breaks_gof(p['data'][slice(left, right)],
                                                p['dt']))

            # Local minimum based information
            maxes, mins = identify_local_extrema(smoothed_peak)
            maxes = maxes[(maxes >= left) & (maxes < right)]
            mins = mins[(mins >= left) & (mins < right)]
            num_loc_maxes = len(maxes)

            valley_gap, valley = full_gap_percent_valley(smoothed_peak,
                                                         maxes,
                                                         mins,
                                                         self.config['percent_valley_height'],
                                                         p['dt'])

            valley_height_ratio = valley / np.max(smoothed_peak)

        return dict(time=event['time'],
                    endtime=event['endtime'],
                    s2_max_gos=max_gos,
                    s2_num_loc_max=num_loc_maxes,
                    s2_valley_gap=valley_gap,
                    s2_valley_height_ratio=valley_height_ratio)


def full_gap_percent_valley(smoothp, max_loc, min_loc, pv, dt):
    """
    Full gap at percent valley. The width of the valley at "pv" of the valley height
    :param smoothp: The smoothed peak
    :param max_loc: Location of every local maximum of the peak
    :param min_loc: Location of every local minimum of the peak
    :param pv: "Percent value", the percent of the valley height for which to calculate the gap
    :param dt: The time of one sample in ns
    :return: The gap in ns, the depth in PE
    """
    # Only do this for peaks which have number of maxes-number of mins = 1, since otherwise
    # this local minimum finding doesn't make sense. Furthermore, it gets rid of those peaks
    # which start at some high value likely because they are the tails of another peak
    # or something.

    if (len(max_loc) - len(min_loc) == 1) & (len(min_loc) > 0):
        gaps = np.zeros(len(min_loc))
        gap_heights = np.zeros(len(min_loc))
        for j in range(len(min_loc)):
            gh = np.min(smoothp[[max_loc[j], max_loc[j + 1]]] - smoothp[min_loc[j]])

            height_pv = (smoothp[min_loc[j]] + gh * pv)

            above_hpv = smoothp > height_pv
            above_hpv[:max_loc[j]] = True
            left_crossing = np.argmin(above_hpv)

            above_hpv = smoothp > height_pv
            above_hpv[:min_loc[j]] = False
            right_crossing = np.argmax(above_hpv)

            gaps[j] = right_crossing - left_crossing
            gap_heights[j] = gh

        max_gap = np.max(gap_heights)
        valley_depth = gap_heights[np.argmax(gap_heights)]
        return max_gap * dt, valley_depth
    else:
        return 0, 0


def bounds_above_percentage_height(p, percent):
    """
    :param p: The peak
    :param percent: The percentage of the maximum height to cut the peak,
    this is to reject the tails and noise before and after the bulk of the peak
    """
    above_pecent_height = np.where(p >= np.max(p) * percent)[0]
    assert len(above_pecent_height) >= 1, 'At least one sample is above %f fraction of the peak'

    return above_pecent_height[0], above_pecent_height[-1] + 1


def identify_local_extrema(smoothp):
    """
    Identifies local minima and maxima by comparing each point to the one before and after it
    :param smoothp: smoothed peak
    :return: The locations of the minima, the locations of the maxima
    """
    larger_than_next = (smoothp > np.pad(smoothp[:-1], (1, 0))).astype('int')

    max_loc = np.where(np.diff(larger_than_next) < 0)[0]
    min_loc = np.where(np.diff(larger_than_next) > 0)[0]

    return max_loc, min_loc


def power_smooth(origindata, cover_num, power):
    """
    A smoothing filter to get rid of the noise in peaks so that we don't find too many local extrema
    that are just noisy
    :param origindata: Original peak
    :param cover_num: The cover number for smoothing, high cover numbers mean you smooth over a larger region
    :param power: The power of the smoothing, essentially tunes how much your smoothing filter looks like a square
    :return: The smoothed waveform
    """
    x_zeroed = np.arange(-cover_num, cover_num + 1)
    weight = (1 - np.abs(x_zeroed / cover_num) ** power) ** power
    weight = weight / np.sum(weight)

    smoothed_data = np.convolve(origindata, weight)[cover_num:-cover_num]
    return smoothed_data