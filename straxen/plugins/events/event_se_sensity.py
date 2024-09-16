import numpy as np
import strax


class EventSEDensity(strax.Plugin):
    """This plugin is designed to calculate the single electron rate density for main and alt S2 in
    events.

    References:
        * v0.1.0: xenon:xenonnt:ac:sr1:hotspot_veto_cut:wimp_roi

    """

    __version__ = "0.0.0"
    depends_on = ("event_basics", "peak_se_density")
    provides = "event_se_density"

    def infer_dtype(self):
        dtype = []
        for s_i in [1, 2]:
            for main_or_alt, description in zip(["", "alt_"], ["main", "alternate"]):
                dtype += [
                    (
                        (
                            f"Neiboring single-electron score of {description} S{s_i} [Hz/cm^2]",
                            f"{main_or_alt}s{s_i}_se_nearby_probability",
                        ),
                        np.float32,
                    ),
                ]
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        split_peaks = strax.split_by_containment(peaks, events)
        result = np.zeros(len(events), self.dtype)

        # 1. Assign peaks features to main S2 and alt S2 in the event
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            for type_ in ["s1", "alt_s1", "s2", "alt_s2"]:
                type_index = event[f"{type_}_index"]
                if type_index != -1:
                    result[f"{type_}_se_nearby_probability"][event_i] = sp["se_nearby_probability"][
                        type_index
                    ]

        # 2. Set time and endtime for events
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)
        return result
