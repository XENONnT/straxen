import strax
import numpy as np
import straxen
import numba

export, __all__ = strax.exporter()


@export
class S2FWHM2(strax.Plugin):
    """This is a default plugin, like used by many plugins in straxen. It finds the full-width half-
    maximum of the main and alternate S2 peak for each event.

    This is the exact same code as S2FWHM, except for the fact that I set the first FWHM of each
    chunk to a NaN, and the 'random_stuff' field to -1.

    """

    __version__ = "0.0.1"

    depends_on = ("event_basics", "peaks")
    provides = "s2_fwhm_2"
    data_kind = "events"

    smoothing = straxen.URLConfig(
        default=False,
        type=bool,
        track=True,
        help="Flag for whether or not the waveform is smoothed or not.",
    )
    averaging_samples = straxen.URLConfig(
        default=3, type=int, track=True, help="Number of samples to average over."
    )

    # We really could just type the dtype, but for demonstration purposes
    # we can use infer_dtype
    def infer_dtype(self):
        dtype = [
            (("Start time since unix epoch [ns]", "time"), np.int64),
            (("Exclusive end time since unix epoch [ns]", "endtime"), np.int64),
            (("FWHM for s2 S2 in the event", "s2_fwhm"), np.float32),
            (("FWHM for alt_s2 S2 in the event", "alt_s2_fwhm"), np.float32),
            (("An extra field just to detect something", "random_stuff"), np.int32),
        ]

        return dtype

    def compute(self, events, peaks):
        result = np.zeros(len(events), self.dtype)
        result["time"] = events["time"]
        result["endtime"] = events["endtime"]

        if len(events) > 0:
            pks_per_ev = strax.split_by_containment(peaks, events)
            pk_buffer = {
                t: -999 * np.ones((len(events), len(peaks["data"][0]))) for t in ["s2", "alt_s2"]
            }
            pk_dt = {t: 10 * np.ones(len(events)) for t in ["s2", "alt_s2"]}

            for t in ["s2", "alt_s2"]:
                for i, (ev, pk_ev) in enumerate(zip(events, pks_per_ev)):
                    pk_buffer[t][i] = pk_ev[ev[f"{t}_index"]]["data"]
                    pk_dt[t][i] = pk_ev[ev[f"{t}_index"]]["dt"]

            for t in ["s2", "alt_s2"]:
                if self.smoothing:
                    pk_buffer[t] = smooth(pk_buffer[t], self.averaging_samples)
                result[f"{t}_fwhm"] = fwhm(pk_buffer[t], pk_dt[t])

            result["s2_fwhm"][0] = np.nan
            result["random_stuff"][0] = -1

        return result


@numba.njit()
def fwhm(wfs, dts):
    result = -np.ones(len(wfs))
    for i, (w, dt) in enumerate(zip(wfs, dts)):
        if w[0] == -999:
            result[i] = -1
        else:
            above_half = np.where(w >= 0.5 * np.max(w))[0]
            result[i] = (above_half[-1] - above_half[0]) * dt
    return result


@numba.njit()
def smooth(wfs, smooth_samps):
    smooth_wfs = -999 * np.ones((len(wfs), len(wfs[0]) + smooth_samps - 1))
    for wf_i, wf in enumerate(wfs):
        if wf[0] != -999:
            smooth_wfs[wf_i] = np.convolve(wf, np.ones(smooth_samps) / smooth_samps)

    return smooth_wfs
