import numpy as np
from scipy.stats import chi2
import strax

from straxen import units

export, __all__ = strax.exporter()


class S2Width(strax.Plugin):
    """S2 Width cut based on diffusion model
    The S2 width cut compares the S2 width to what we could expect based on its depth in the detector. The inputs to
    this are the drift velocity and the diffusion constant. The allowed variation in S2 width is greater at low
    energy (since it is fluctuating statistically) Ref: (arXiv:1102.2865)
    It should be applicable to data regardless of if it ER or NR;
    above cS2 = 1e5 pe ERs the acceptance will go down due to track length effects.
    around S2 = 1e5 pe there are beta-gamma merged peaks from Pb214 that extends the S2 width
    Tune the diffusion model parameters based on fax data according to note:
    https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:sim:notes:tzhu:width_cut_tuning#toy_fax_simulation
    Contact: Tianyu <tz2263@columbia.edu>, Yuehuan <weiyh@physik.uzh.ch>, Jelle <jaalbers@nikhef.nl>
    ported from lax.sciencerun1.py"""
    depends_on = ('event_info',)
    provides = 'cut_S2_width'
    dtype = [('cut_S2width', np.float32, 'S2 Width cut')]

    version = 1

    diffusion_constant = 25.26 * ((units.cm)**2) / units.s
    v_drift = 1.440 * (units.um) / units.ns
    scg = 23.0  # s2_secondary_sc_gain in pax config
    scw = 258.41  # s2_secondary_sc_width median
    SigmaToR50 = 1.349
    DriftTimeFromGate = 1.6 * units.us

    def s2_width_model(self, drift_time):
        """Diffusion model
        """
        return np.sqrt(2 * self.diffusion_constant * (drift_time - self.DriftTimeFromGate) / self.v_drift ** 2)

    def nElectron(self, events):
        return np.clip(events['s2_area'],0,5000) / self.scg

    def normWidth(self, events):
        return (np.square(events['s2_range_50p_area'] / self.SigmaToR50) -np.square(self.scw) /
                np.square(self.s2_width_model(events['drift_time'])))

    def logpdf(self, events):
        return chi2.logpdf(self.normWidth(events) * (self.nElectron(events) - 1), self.nElectron(events))

    def compute(self, events):
        arr = np.all([self.logpdf(events) >-14],axis=0)
#        arr = self.logpdf(events)
        return dict(s2width=arr)


class S1SingleScatter(strax.Plugin):
    """Requires only one valid interaction between the largest S2, and any S1 recorded before it.
    The S1 cut checks that any possible secondary S1s recorded in a waveform, could not have also
    produced a valid interaction with the primary S2. To check whether an interaction between the
    second largest S1 and the largest S2 is valid, we use the S2Width cut. If the event would pass
    the S2Width cut, a valid second interaction exists, and we may have mis-identified which S1 to
    pair with the primary S2. Therefore we cut this event. If it fails the S2Width cut the event is
    not removed.
    Current version is developed on calibration data (pax v6.8.0). It is described in this note:
    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:jacques:s1_single_scatter_cut_sr1
    It should be applicable to data regardless whether it is ER or NR.
    Contact: Jacques Pienaar, <jpienaar@uchicago.edu>
    ported from lax.sciencerun1.py
    """
    depends_on = 'event_info'
    provides = 'cut_S1SingleScatter'

    dtype = [('cut_S1SingleScatter', np.bool, 'S1 Single Scatter cut')]
    version = 1


    def compute(self, events):

        mask = alt_s1_interaction_drift_time > self.S2width.DriftTimeFromGate
        alt_n_electron = np.clip(events[mask]['s2_area'], 0, 5000) / self.S2width.scg

        # Alternate S1 relative width
        alt_rel_width = np.square(events[mask]['s2_range_50p_area'] / self.S2width.SigmaToR50)- np.square(self.S2width.scw)
        alt_rel_width /= np.square(self.S2width.s2_width_model(self.S2width,
                                                               events[mask]['alt_s1_interaction_drift_time']))

        alt_interaction_passes = chi2.logpdf(alt_rel_width * (alt_n_electron - 1), alt_n_electron) > - 20

#        arr = np.all([events[mask])
        df.loc[mask, (self.name())] = True ^ alt_interaction_passes

        return df

    def compute(self, events):
        arr = np.all([(-92.9 < events['z']), (-9 > events['z']),
                      (36.94 > np.sqrt(events['x']**2 + events['y']**2))],
                     axis=0)
        return dict(cut_fiducial_cylinder=arr)



class S2SingleScatter(strax.Plugin):
    """Check that largest other S2 area is smaller than some bound.
    The single scatter is to cut an event if its largest_other_s2 is too large.
    As the largest_other_s2 takes a greater value when they originated from some real scatters
    in comparison, those from photo-ionization in single scatter cases would be smaller.
    https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:cut:s2single
    Contact: Tianyu Zhu <tz2263@columbia.edu>
    ported from lax.sciencerun1.py
    """
    depends_on= 'event_info'
    provides = 'cut_S2SingleScatter'
    dtype = [('cut_S2SingleScatter', np.bool, 'S2 Single Scatter cut')]
    version = 4
    allowed_range = (0, np.inf)
    variable = 'temp'


    def name(self):
        return 'Cut%s' % self.__class__.__name__


    @classmethod
    def other_s2_bound(cls, s2_area):
        rescaled_s2_0 = s2_area * 0.00832 + 72.3
        rescaled_s2_1 = s2_area * 0.03 - 109

        another_term_0 = 1 / (np.exp((s2_area - 23300) * 5.91e-4) + 1)
        another_term_1 = 1 / (np.exp((23300 - s2_area) * 5.91e-4) + 1)

        return rescaled_s2_0 * another_term_0 + rescaled_s2_1 * another_term_1

    def compute(self, events):
        largest_other_s2_is_nan = np.isnan(events['s2_largest_other'])
        arr = np.all([largest_other_s2_is_nan | (events['s2_largest_other'] < self.other_s2_bound(events['s2_area']))],
                    axis=0)
        return dict(cut_s2_singelscatter = arr)


class S2Threshold(strax.Plugin):
    """The S2 energy at which the trigger is perfectly efficient.
    See: https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:analysis:firstresults:daqtriggerpaxefficiency
    Contact: Jelle Aalbers <aalbers@nikhef.nl>
    ported from lax.sciencerun1.py
    """
    depends_on = 'event_info'
    provides = 'S2Threshold'
    dtype = [('cut_s2threshold', np.bool, 's2 must be larger then 200 PE')]

    def compute(self, events):
        arr = np.all([events['s2_area']>200],
                     axis=0)
        return dict(cut_s2threshold=arr)

class S2AreaFractionTop(strax.Plugin):
    """Cuts events with an unusual fraction of S2 on top array.
    Primarily cuts gas events with a particularly large S2 AFT, also targets some
    strange / junk / other events with a low AFT.
    This cut has been checked on S2 ranges between 0 and 50 000 pe.
    Described in the note at: xenon:xenon1t:analysis:firstresults:s2_aft_cut_summary
    Contact: Adam Brown <abrown@physik.uzh.ch>,
    ported from lax.sciencerun1.py
    """
    depends_on = ('event_info',)
    provides = 'cut_S2AreaFractionTop'
    dtype = [('cut_S2AreaFractionTop', np.bool, 'Cut on S2 AFT')]

    def compute(self, events):
        arr = np.all([events['s2_area_fraction_top']<upperlimit(events['s2_area_fraction_top']),
                    events['s2_area_fraction_top']>lowerlimit(events['s2_area_fraction_top'])])
        return dict(cut_S2AreaFractionTop=arr)



class FiducialCylinder1T(strax.Plugin):
    """Implementation of fiducial volume cylinder 1T,
    ported from lax.sciencerun0.py"""
    depends_on = ('event_positions',)
    provides = 'fiducial_cylinder_1t'
    dtype = [('cut_fiducial_cylinder', np.bool, 'One tonne fiducial cylinder')]

    def compute(self, events):
        arr = np.all([(-92.9 < events['z']), (-9 > events['z']),
                      (36.94 > np.sqrt(events['x']**2 + events['y']**2))],
                     axis=0)
        return dict(cut_fiducial_cylinder=arr)


class S1MaxPMT(strax.LoopPlugin):
    """Removes events where the largest hit in S1 is too large
    port from lax.sciencerun0.py"""
    depends_on = ('events', 'event_basics', 'peak_basics')
    dtype = [('cut_s1_max_pmt', np.bool, 'S1 max PMT cut')]

    def compute_loop(self, event, peaks):
        ret = dict(cut_s1_max_pmt=True)
        if not len(peaks) or np.isnan(event['s1_index']):
            return ret

        peak = peaks[event['s1_index']]
        max_channel = peak['max_pmt_area']
        ret['cut_s1_max_pmt'] = (max_channel < 0.052 * event['s1_area'] + 4.15)
        return ret


class S1LowEnergyRange(strax.Plugin):
    """Pass only events with cs1<200"""
    depends_on = ('events', 'corrected_areas')
    dtype = [('cut_s1_low_energy_range', np.bool, "Event under 200pe")]

    def compute(self, events):
        ret = np.all([events['cs1'] < 200], axis=0)
        return dict(cut_s1_low_energy_range=ret)


class SR1Cuts(strax.MergeOnlyPlugin):
    depends_on = ['fiducial_cylinder_1t', 's1_max_pmt', 's1_low_energy_range']
    save_when = strax.SaveWhen.ALWAYS


class FiducialEvents(strax.Plugin):
    depends_on = ['event_info', 'fiducial_cylinder_1t']
    data_kind = 'fiducial_events'

    def infer_dtype(self):
        return strax.merged_dtype([self.deps[d].dtype
                                   for d in self.depends_on])

    def compute(self, events):
        return events[events['cut_fiducial_cylinder']]
