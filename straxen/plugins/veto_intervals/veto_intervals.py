import typing
import numpy as np
import strax

from straxen.plugins.aqmon_hits.aqmon_hits import AqmonChannels

export, __all__ = strax.exporter()


# ### Veto hardware ###:
# V1495 busy veto module:
# Generates a 25 ns NIM pulse whenever a veto begins and a 25 ns NIM signal when it ends.
# A new start signal can occur only after the previous busy instance ended.
# 1ms (1e6 ns) - minimum busy veto length, or until the board clears its memory

# DDC10 High Energy Veto:
# 10ms (1e7 ns) - fixed HE veto length in XENON1T DDC10,
# in XENONnT it will be calibrated based on the length of large S2 SE tails
# The start/stop signals for the HEV are generated by the V1495 board


@export
class VetoIntervals(strax.ExhaustPlugin):
    """Find pairs of veto start and veto stop signals and the veto.

    duration between them:
     - busy_* <= V1495 busy veto for tpc channels
     - busy_he_* <= V1495 busy veto for high energy tpc channels
     - hev_* <= DDC10 hardware high energy veto
     - straxen_deadtime <= special case of deadtime introduced by the
       DAQReader-plugin

    """

    __version__ = "1.1.1"
    depends_on = "aqmon_hits"
    provides = "veto_intervals"
    data_kind = "veto_intervals"

    def infer_dtype(self):
        dtype = [
            (("veto interval [ns]", "veto_interval"), np.int64),
            (("veto signal type", "veto_type"), np.str_("U30")),
        ]
        dtype += strax.time_fields
        return dtype

    def setup(self):
        self.veto_names = ["busy_", "busy_he_", "hev_"]
        self.channel_map = {aq_ch.name.lower(): int(aq_ch) for aq_ch in AqmonChannels}

    def compute(self, aqmon_hits, start, end):
        # Allocate a nice big buffer and throw away the part we don't need later
        result = np.zeros(len(aqmon_hits) * len(self.veto_names), self.dtype)
        vetos_seen = 0

        for veto_name in self.veto_names:
            veto_hits_start = channel_select(aqmon_hits, self.channel_map[veto_name + "start"])
            veto_hits_stop = channel_select(aqmon_hits, self.channel_map[veto_name + "stop"])

            veto_hits_start, veto_hits_stop = self.handle_starts_and_stops_outside_of_run(
                veto_hits_start=veto_hits_start,
                veto_hits_stop=veto_hits_stop,
                chunk_start=start,
                chunk_end=end,
                veto_name=veto_name,
            )
            n_vetos = len(veto_hits_start)

            result["time"][vetos_seen : vetos_seen + n_vetos] = veto_hits_start["time"]
            result["endtime"][vetos_seen : vetos_seen + n_vetos] = veto_hits_stop["time"]
            result["veto_type"][vetos_seen : vetos_seen + n_vetos] = veto_name + "veto"

            vetos_seen += n_vetos

        # Straxen deadtime is special, it's a start and stop with no data
        # but already an interval so easily used here
        artificial_deadtime = aqmon_hits[
            (aqmon_hits["channel"] == AqmonChannels.ARTIFICIAL_DEADTIME)
        ]
        n_artificial = len(artificial_deadtime)

        if n_artificial:
            result[vetos_seen:n_artificial]["time"] = artificial_deadtime["time"]
            result[vetos_seen:n_artificial]["endtime"] = strax.endtime(artificial_deadtime)
            result[vetos_seen:n_artificial]["veto_type"] = "straxen_deadtime_veto"
            vetos_seen += n_artificial

        result = result[:vetos_seen]
        result["veto_interval"] = result["endtime"] - result["time"]
        argsort = strax.stable_argsort(result["time"])
        result = result[argsort]
        return result

    def handle_starts_and_stops_outside_of_run(
        self,
        veto_hits_start: np.ndarray,
        veto_hits_stop: np.ndarray,
        chunk_start: int,
        chunk_end: int,
        veto_name: str,
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """We might be missing one start or one stop at the end of the run, set it to the chunk
        endtime if this is the case."""
        # Just for traceback info that we declare this here
        extra_start = []
        extra_stop = []
        missing_a_final_stop = (
            len(veto_hits_start)
            and len(veto_hits_stop)
            and veto_hits_start[-1]["time"] > veto_hits_stop["time"][-1]
        )
        missing_a_final_stop = missing_a_final_stop or (
            len(veto_hits_start) and not len(veto_hits_stop)
        )
        if missing_a_final_stop:
            # There is one *start* of the //end// of the run -> the
            # **stop** is missing (because it's outside of the run),
            # let's add one **stop** at the //end// of this chunk
            extra_stop = self.fake_hit(chunk_end)
            veto_hits_stop = np.concatenate([veto_hits_stop, extra_stop])
        if len(veto_hits_stop) - len(veto_hits_start) == 1:
            # There is one *stop* of the //beginning// of the run
            # -> the **start** is missing (because it's from before
            # starting the run), # let's add one **start** at the
            # //beginning// of this chunk
            extra_start = self.fake_hit(chunk_start)
            veto_hits_start = np.concatenate([extra_start, veto_hits_start])

        something_is_wrong = len(veto_hits_start) != len(veto_hits_stop)

        message = (
            f"Got inconsistent number of {veto_name} starts "
            f"{len(veto_hits_start)}) / stops ({len(veto_hits_stop)})."
        )
        if len(extra_start):
            message += " Despite the fact that we inserted one extra start at the beginning of the run."  # noqa
        elif len(extra_stop):
            message += (
                " Despite the fact that we inserted one extra stop at the end of the run."  # noqa
            )
        if something_is_wrong:
            raise ValueError(message)

        if np.any(veto_hits_start["time"] > veto_hits_stop["time"]):
            raise ValueError("Found veto's starting before the previous stopped")

        return veto_hits_start, veto_hits_stop

    @staticmethod
    def fake_hit(start, dt=1, length=1):
        hit = np.zeros(1, strax.hit_dtype)
        hit["time"] = start
        hit["dt"] = dt
        hit["length"] = length
        return hit


# Don't use @numba since numba doesn't like masking arrays, use numpy
def channel_select(rr, ch):
    """Return data from start/stop veto channel in the acquisition monitor (AM)"""
    return rr[rr["channel"] == ch]
