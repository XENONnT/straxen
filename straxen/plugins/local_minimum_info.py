import numpy as np
import strax
from strax.processing.peak_splitting import natural_breaks_gof

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option('divide_90p_width_localmin', default=7., type = float,
                 help="The peak is smoothed by dividing the 90p width by"
                      "this number, and coverting it into number of samples."
                      "This is then the 'n' used in the smoothing kernel"
                      "shown below."),
    strax.Option('smoothing_power_localmin', default=3., type = float,
                 help="The power used in the smoothing filter with a kernel of"
                      "(1-(x/n)^p)^p, where p is the power"),
    strax.Option('percentage_threshold_localmin', default=0.1, type = float,
                 help="The height threshold for the peak as a percentage"
                      "of the maximum, used to reject the low parts"
                      "of the peak in order to find the local extrema."),
    strax.Option('percent_valley_height', default=0.9, type = float,
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
    data_kind = 'events'
    provides = 'event_local_min_info'
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
            
            smoothing_number = p['width'][9] / p['dt']
            smoothing_number = np.ceil(smoothing_number / self.config['divide_90p_width_localmin'])
            smoothed_peak = power_smooth(p['data'][:p['length']],
                                         int(smoothing_number),
                                         self.config['smoothing_power_localmin'])
            
            if len(smoothed_peak)>0:
                # Set data below percentage threshold on both side to zeros
                left, right = bounds_above_percentage_height(smoothed_peak,
                                                             self.config['percentage_threshold_localmin'])

                # Maximum GOS calculation for data above percentage
                max_gos = np.max(natural_breaks_gof(p['data'][left: right],
                                                    p['dt']))

                # Local minimum based information
                maxes, mins = identify_local_extrema(smoothed_peak)

                maxes = maxes[np.logical_and(maxes >= left, maxes < right)]
                mins = mins[np.logical_and(mins >= left, mins < right)]

                num_loc_maxes = len(maxes)

                valley_gap, valley = full_gap_percent_valley(smoothed_peak,
                                                             maxes,
                                                             mins,
                                                             self.config['percent_valley_height'],
                                                             p['dt'])

                valley_height_ratio = valley / np.max(smoothed_peak)

        return {'time': event['time'],
                'endtime': event['endtime'],
                's2_max_gos': max_gos,
                's2_num_loc_max': num_loc_maxes,
                's2_valley_gap': valley_gap,
                's2_valley_height_ratio': valley_height_ratio}


def full_gap_percent_valley(smoothp, max_loc, min_loc, pv, dt):
    """
    Full gap at percent valley. The width of the valley at "pv" of the valley height
    Only do this for peaks which have number of maxes-number of mins = 1, since otherwise
    this local minimum finding doesn't make sense. Furthermore, it gets rid of those peaks
    which start at some high value likely because they are the tails of another peak
    or something.
    :param smoothp: The smoothed peak
    :param max_loc: Location of every local maximum of the peak
    :param min_loc: Location of every local minimum of the peak
    :param pv: "Percent value", the percent of the valley height for which to calculate the gap
    :param dt: The time of one sample in ns
    :return: The gap in ns, the depth in PE
    """
    n_gap = len(min_loc)
    p_length = len(smoothp)
    if ~((len(max_loc) - n_gap != 1)&(len(min_loc)>0)):
        return 0, 0
    else:
        gaps = np.zeros(n_gap)
        gap_heights = np.zeros(len(min_loc))
        for j in range(n_gap):
            gh = np.min(smoothp[max_loc[j:j + 2]])
            gh -= smoothp[min_loc[j]]

            height_pv = (smoothp[min_loc[j]] + gh * pv)

            above_hpv = smoothp > height_pv
            above_hpv |= np.arange(p_length) < max_loc[j]
            left_crossing = np.argmin(above_hpv)

            above_hpv = smoothp > height_pv
            above_hpv &= np.arange(p_length) >= min_loc[j]
            right_crossing = np.argmax(above_hpv)

            gaps[j] = right_crossing - left_crossing
            gap_heights[j] = gh

        max_gap = gaps[np.argmax(gap_heights)]
        valley_depth = gap_heights[np.argmax(gap_heights)]
        return max_gap * dt, valley_depth

def bounds_above_percentage_height(p, percent):
    """
    Finding the index bounds of the peak above given percentage
    :param p: The peak
    :param percent: The percentage of the maximum height to cut the peak,
    this is to reject the tails and noise before and after the bulk of the peak
    :return: The left and right (exclusive) index of samples above the percent
    """
    percent_height = np.max(p) * percent
    above_percent_height = np.where(p >= percent_height)[0]

    return above_percent_height[0], above_percent_height[-1] + 1


def identify_local_extrema(smoothp):
    """
    Identifies local minima and maxima by comparing each point to the one before and after it
    :param smoothp: smoothed peak
    :return: The locations of the minima, the locations of the maxima
    """
    larger_than_next = (smoothp > np.pad(smoothp[:-1], (1, 0)))
    larger_than_next = larger_than_next.astype('int')

    max_loc = np.where(np.diff(larger_than_next) < 0)[0]
    min_loc = np.where(np.diff(larger_than_next) > 0)[0]

    return max_loc, min_loc


def power_smooth(origindata, cover_num, power):
    """
    A smoothing filter to get rid of the noise in peaks so that we don't find too many local extrema
    that are just noisy
    :param origindata: Original peak
    :param cover_num: The cover number for smoothing
    high cover numbers mean you smooth over a larger region
    :param power: The power of the smoothing
    essentially tunes how much your smoothing filter looks like a square
    :return: The smoothed waveform
    """
    x_zeroed = np.arange(-cover_num, cover_num + 1)
    weight = (1 - np.abs(x_zeroed / cover_num) ** power) ** power
    weight = weight / np.sum(weight)

    smoothed_data = np.convolve(origindata, weight)[cover_num:-cover_num]
    return smoothed_data
