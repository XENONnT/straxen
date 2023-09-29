import strax
import numpy as np
from immutabledict import immutabledict
import straxen

export, __all__ = strax.exporter()


@export
class OnlineMonitorNV(strax.Plugin):
    """Plugin to write data of nVeto detector to the online-monitor. Data that is written by this
    plugin should be small (~MB/chunk) to not overload the runs-database.

    This plugin takes 'hitlets_nv' and 'events_nv'. Although they are not strictly related, they are
    aggregated into a single data_type in order to minimize the number of documents in the online
    monitor.

    Produces 'online_monitor_nv' with info on the hitlets_nv and events_nv

    """

    __version__ = "0.0.4"

    depends_on = ("hitlets_nv", "events_nv")
    provides = "online_monitor_nv"
    data_kind = "online_monitor_nv"

    # Needed in case we make again an muVETO child.
    ends_with = "_nv"

    channel_map = straxen.URLConfig(
        track=False,
        type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) " "channel number.",
    )

    events_area_bounds = straxen.URLConfig(
        type=tuple,
        default=(-0.5, 130.5),
        help="Boundaries area histogram of events_nv_area_per_chunk [PE]",
    )

    events_area_nbins = straxen.URLConfig(
        type=int,
        default=131,
        help="Number of bins of histogram of events_nv_area_per_chunk, " "defined value 1 PE/bin",
    )

    def infer_dtype(self):
        self.channel_range = self.channel_map["nveto"]
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1
        return veto_monitor_dtype(self.ends_with, self.n_channel, self.events_area_nbins)

    def compute(self, hitlets_nv, events_nv, start, end):
        # General setup
        res = np.zeros(1, dtype=self.dtype)
        res["time"] = start
        res["endtime"] = end

        # Count number of hitlets_nv per PMT
        hitlets_channel_count, _ = np.histogram(
            hitlets_nv["channel"],
            bins=self.n_channel,
            range=[self.channel_range[0], self.channel_range[1] + 1],
        )
        res[f"hitlets{self.ends_with}_per_channel"] = hitlets_channel_count

        # Count number of events_nv with coincidence cut
        res[f"events{self.ends_with}_per_chunk"] = len(events_nv)
        sel = events_nv["n_contributing_pmt"] >= 4
        res[f"events{self.ends_with}_4coinc_per_chunk"] = np.sum(sel)
        sel = events_nv["n_contributing_pmt"] >= 5
        res[f"events{self.ends_with}_5coinc_per_chunk"] = np.sum(sel)
        sel = events_nv["n_contributing_pmt"] >= 8
        res[f"events{self.ends_with}_8coinc_per_chunk"] = np.sum(sel)
        sel = events_nv["n_contributing_pmt"] >= 10
        res[f"events{self.ends_with}_10coinc_per_chunk"] = np.sum(sel)

        # Get histogram of events_nv_area per chunk
        events_area, bins_ = np.histogram(
            events_nv["area"], bins=self.events_area_nbins, range=self.events_area_bounds
        )
        res[f"events{self.ends_with}_area_per_chunk"] = events_area
        return res


def veto_monitor_dtype(veto_name: str = "_nv", n_pmts: int = 120, n_bins: int = 131) -> list:
    dtype = []
    dtype += strax.time_fields  # because mutable
    dtype += [
        (
            (f"hitlets{veto_name} per channel", f"hitlets{veto_name}_per_channel"),
            (np.int64, n_pmts),
        ),
        (
            (f"events{veto_name}_area per chunk", f"events{veto_name}_area_per_chunk"),
            np.int64,
            n_bins,
        ),
        ((f"events{veto_name} per chunk", f"events{veto_name}_per_chunk"), np.int64),
        (
            (f"events{veto_name} 4-coincidence per chunk", f"events{veto_name}_4coinc_per_chunk"),
            np.int64,
        ),
        (
            (f"events{veto_name} 5-coincidence per chunk", f"events{veto_name}_5coinc_per_chunk"),
            np.int64,
        ),
        (
            (f"events{veto_name} 8-coincidence per chunk", f"events{veto_name}_8coinc_per_chunk"),
            np.int64,
        ),
        (
            (f"events{veto_name} 10-coincidence per chunk", f"events{veto_name}_10coinc_per_chunk"),
            np.int64,
        ),
    ]
    return dtype
