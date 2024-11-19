import strax
import straxen
import numpy as np
import numba
import typing as ty
from immutabledict import immutabledict

export, __all__ = strax.exporter()


@export
class nVETOEvents(strax.OverlapWindowPlugin):
    """Plugin which computes the boundaries of veto events."""

    __version__ = "0.1.0"

    depends_on = "hitlets_nv"
    provides = "events_nv"
    data_kind = "events_nv"
    compressor = "zstd"
    events_seen = 0

    event_left_extension_nv = straxen.URLConfig(
        default=0, track=True, type=int, help="Extends event window this many [ns] to the left."
    )

    event_resolving_time_nv = straxen.URLConfig(
        default=300, track=True, type=int, help="Resolving time for window coincidence [ns]."
    )

    event_min_hits_nv = straxen.URLConfig(
        default=4,
        track=True,
        type=int,
        help="Minimum number of fully confined hitlets to define an event.",
    )

    channel_map = straxen.URLConfig(
        track=False,
        type=immutabledict,
        help="immutabledict mapping subdetector to (min, max) channel number",
    )

    def infer_dtype(self):
        self.name_event_number = "event_number_nv"
        self.channel_range = self.channel_map["nveto"]
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1
        return veto_event_dtype(self.name_event_number, self.n_channel)

    def get_window_size(self):
        return self.event_left_extension_nv + self.event_resolving_time_nv + 1

    def compute(self, hitlets_nv, start, end):
        events, hitlets_ids_in_event = find_veto_events(
            hitlets_nv,
            self.event_min_hits_nv,
            self.event_resolving_time_nv,
            self.event_left_extension_nv,
            event_number_key=self.name_event_number,
            n_channel=self.n_channel,
        )

        # Compute basic properties:
        if len(hitlets_ids_in_event):
            compute_nveto_event_properties(
                events, hitlets_nv, hitlets_ids_in_event, start_channel=self.channel_range[0]
            )

        # Get eventids:
        n_events = len(events)
        events[self.name_event_number] = np.arange(n_events) + self.events_seen
        self.events_seen += n_events
        return events


def veto_event_dtype(
    name_event_number: str = "event_number_nv",
    n_pmts: int = 120,
) -> list:
    dtype = []
    dtype += strax.time_fields  # because mutable
    dtype += [
        (("Veto event number in this dataset", name_event_number), np.int64),
        (("Total area of all hitlets in event [pe]", "area"), np.float32),
        (("Total number of hitlets in events", "n_hits"), np.int32),
        (("Total number of contributing channels", "n_contributing_pmt"), np.uint8),
        (("Area in event per channel [pe]", "area_per_channel"), np.float32, n_pmts),
        (
            (
                "Area weighted mean time of the event relative to the event start [ns]",
                "center_time",
            ),
            np.float32,
        ),
        (("Weighted variance of time [ns]", "center_time_spread"), np.float32),
        (
            ("Minimal amplitude-to-amplitude gap between neighboring hitlets [ns]", "min_gap"),
            np.int8,
        ),
        (
            ("Maximal amplitude-to-amplitude gap between neighboring hitlets [ns]", "max_gap"),
            np.int8,
        ),
    ]
    return dtype


@numba.njit(cache=True, nogil=True)
def compute_nveto_event_properties(
    events: np.ndarray,
    hitlets: np.ndarray,
    contained_hitlets_ids: np.ndarray,
    start_channel: int = 2000,
):
    """Computes properties of the neutron-veto events. Writes results directly to events.

    :param events: Events for which properties should be computed
    :param hitlets: hitlets which were used to build the events.
    :param contained_hitlets_ids: numpy array of the shape n x 2 which holds the indices of the
        hitlets contained in the corresponding event.
    :param start_channel: Integer specifying start channel, e.g. 2000 for nveto.

    """
    for e, (s_i, e_i) in zip(events, contained_hitlets_ids):
        hitlets_in_event = hitlets[s_i:e_i]
        event_area = np.sum(hitlets_in_event["area"])
        e["area"] = event_area
        e["n_hits"] = len(hitlets_in_event)
        e["n_contributing_pmt"] = len(np.unique(hitlets_in_event["channel"]))

        t = hitlets_in_event["time"] - hitlets_in_event[0]["time"]

        dt = np.diff(hitlets_in_event["time"] + hitlets_in_event["time_amplitude"])
        e["min_gap"] = np.min(dt)
        e["max_gap"] = np.max(dt)

        if event_area:
            e["center_time"] = np.sum(t * hitlets_in_event["area"]) / event_area
            if e["n_hits"] > 1 and e["center_time"]:
                w = hitlets_in_event["area"] / e["area"]  # normalized weights
                # Definition of variance
                e["center_time_spread"] = np.sqrt(
                    np.sum(w * np.power(t - e["center_time"], 2)) / np.sum(w)
                )
            else:
                e["center_time_spread"] = np.inf

        # Compute per channel properties:
        for hit in hitlets_in_event:
            ch = hit["channel"] - start_channel
            e["area_per_channel"][ch] += hit["area"]


@export
def find_veto_events(
    hitlets: np.ndarray,
    coincidence_level: int,
    resolving_time: int,
    left_extension: int,
    event_number_key: str = "event_number_nv",
    n_channel: int = 120,
) -> ty.Tuple[np.ndarray, np.ndarray]:
    """Function which find the veto events as a nfold concidence in a given resolving time window.
    All hitlets which touch the event window contribute.

    :param hitlets: Hitlets which shall be used for event creation.
    :param coincidence_level: int, coincidence level.
    :param resolving_time: int, resolving window for coincidence in ns.
    :param left_extension: int, left event extension in ns.
    :param event_number_key: str, field name for the event number
    :param n_channel: int, number of channels in detector.
    :return: events, hitelt_ids_per_event

    """
    # Find intervals which satisfy requirement:
    event_intervals = straxen.plugins.nveto_recorder.find_coincidence(
        hitlets,
        coincidence_level,
        resolving_time,
        left_extension,
    )

    # Find all hitlets which touch the coincidence windows:
    # (we cannot use fully_contained in here since some muon signals
    # may be larger than 300 ns)
    hitlets_ids_in_event = strax.touching_windows(hitlets, event_intervals)

    # For some rare cases long signals may touch two intervals, in that
    # case we merge the intervals in the subsequent function:
    hitlets_ids_in_event = _solve_ambiguity(hitlets_ids_in_event)

    # Now we can create the veto events:
    events = np.zeros(
        len(hitlets_ids_in_event), dtype=veto_event_dtype(event_number_key, n_channel)
    )
    _make_event(hitlets, hitlets_ids_in_event, events)
    return events, hitlets_ids_in_event


@numba.njit(cache=True, nogil=False)
def _solve_ambiguity(contained_hitlets_ids: np.ndarray) -> np.ndarray:
    """Function which solves the ambiguity if a single hitlets overlaps with two event intervals.

    This can happen for muon signals which have a long tail, since we define the coincidence window
    as a fixed window. Hence those tails can extend beyond the fixed window.

    """
    res = np.zeros(contained_hitlets_ids.shape, dtype=contained_hitlets_ids.dtype)

    if not len(res):
        # Return empty result
        return res

    offset = 0
    start_i, end_i = contained_hitlets_ids[0]
    for e_i, ids in enumerate(contained_hitlets_ids[1:]):
        if end_i > ids[0]:
            # Current and next interval overlap so just updated the end
            # index.
            end_i = ids[1]
        else:
            # They do not overlap store indices:
            res[offset] = [start_i, end_i]
            offset += 1
            # Init next interval:
            start_i, end_i = ids

    # Last event:
    res[offset, :] = [start_i, end_i]
    offset += 1
    return res[:offset]


@numba.njit(cache=True, nogil=True)
def _make_event(hitlets: np.ndarray, hitlet_ids: np.ndarray, res: np.ndarray):
    """Function which sets veto event time and endtime."""
    for ei, ids in enumerate(hitlet_ids):
        hit = hitlets[ids[0] : ids[1]]
        res[ei]["time"] = hit[0]["time"]
        endtime = np.max(strax.endtime(hit))
        res[ei]["endtime"] = endtime
