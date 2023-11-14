from immutabledict import immutabledict
import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@export
class EventAreaPerChannel(strax.Plugin):
    """Simple plugin that provides area per channel for main and alternative S1/S2 in the event."""

    depends_on = ("event_basics", "peaks")
    provides = ("event_area_per_channel", "event_n_channel")
    data_kind = immutabledict(zip(provides, ("events", "events")))
    __version__ = "0.1.1"

    compressor = "zstd"
    save_when = immutabledict({
        "event_area_per_channel": strax.SaveWhen.EXPLICIT,
        "event_n_channel": strax.SaveWhen.ALWAYS,
    })

    n_top_pmts = straxen.URLConfig(default=straxen.n_top_pmts, type=int, help="Number of top PMTs")

    def infer_dtype(self):
        # setting data type from peak dtype
        pfields_ = self.deps["peaks"].dtype_for("peaks").fields
        # populating data type
        infoline = {
            "s1": "main S1",
            "s2": "main S2",
            "alt_s1": "alternative S1",
            "alt_s2": "alternative S2",
        }
        dtype = []
        # populating APC
        ptypes = ["s1", "s2", "alt_s1", "alt_s2"]
        for type_ in ptypes:
            dtype += [
                (
                    (f"Area per channel for {infoline[type_]}", f"{type_}_area_per_channel"),
                    pfields_["area_per_channel"][0],
                )
            ]
            dtype += [
                (
                    (f"Length of the interval in samples for {infoline[type_]}", f"{type_}_length"),
                    pfields_["length"][0],
                )
            ]
            dtype += [
                (
                    (f"Width of one sample for {infoline[type_]} [ns]", f"{type_}_dt"),
                    pfields_["dt"][0],
                )
            ]
        # populating S1 n channel properties
        n_channel_dtype = [
            (("Main S1 count of contributing PMTs", "s1_n_channels"), np.int16),
            (("Main S1 top count of contributing PMTs", "s1_top_n_channels"), np.int16),
            (("Main S1 bottom count of contributing PMTs", "s1_bottom_n_channels"), np.int16),
        ]
        return {
            "event_area_per_channel": dtype + n_channel_dtype + strax.time_fields,
            "event_n_channel": n_channel_dtype + strax.time_fields,
        }

    def compute(self, events, peaks):
        area_per_channel = np.zeros(len(events), self.dtype["event_area_per_channel"])
        area_per_channel["time"] = events["time"]
        area_per_channel["endtime"] = strax.endtime(events)
        n_channel = np.zeros(len(events), self.dtype["event_n_channel"])
        n_channel["time"] = events["time"]
        n_channel["endtime"] = strax.endtime(events)

        split_peaks = strax.split_by_containment(peaks, events)
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            for type_ in ["s1", "s2", "alt_s1", "alt_s2"]:
                type_index = event[f"{type_}_index"]
                if type_index != -1:
                    type_area_per_channel = sp["area_per_channel"][type_index]
                    area_per_channel[f"{type_}_area_per_channel"][event_i] = type_area_per_channel
                    area_per_channel[f"{type_}_length"][event_i] = sp["length"][type_index]
                    area_per_channel[f"{type_}_dt"][event_i] = sp["dt"][type_index]
                    if type_ == "s1":
                        area_per_channel["s1_n_channels"][event_i] = (
                            type_area_per_channel > 0
                        ).sum()
                        area_per_channel["s1_top_n_channels"][event_i] = (
                            type_area_per_channel[: self.config["n_top_pmts"]] > 0
                        ).sum()
                        area_per_channel["s1_bottom_n_channels"][event_i] = (
                            type_area_per_channel[self.config["n_top_pmts"] :] > 0
                        ).sum()
        for field in ["s1_n_channels", "s1_top_n_channels", "s1_bottom_n_channels"]:
            n_channel[field] = area_per_channel[field]
        result = {
            "event_area_per_channel": area_per_channel,
            "event_n_channel": n_channel,
        }
        return result
