import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@export
class EventWaveform(strax.Plugin):
    """Simple plugin that provides (data) waveforms for main and alternative S1/S2 in the event."""

    depends_on = ("event_basics", "peaks")
    provides = "event_waveform"
    __version__ = "0.0.1"

    compressor = "zstd"
    save_when = strax.SaveWhen.EXPLICIT

    n_top_pmts = straxen.URLConfig(default=straxen.n_top_pmts, type=int, help="Number of top PMTs")

    def _have_data(self, field):
        return field in self.deps["peaks"].dtype_for("peaks").names

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
        # populating waveform samples
        ptypes = ["s1", "s2", "alt_s1", "alt_s2"]
        for type_ in ptypes:
            dtype += [
                (
                    (f"Waveform for {infoline[type_]} [ PE / sample ]", f"{type_}_data"),
                    pfields_["data"][0],
                )
            ]
            if self._have_data("data_top"):
                dtype += [
                    (
                        (
                            f"Top waveform for {infoline[type_]} [ PE / sample ]",
                            f"{type_}_data_top",
                        ),
                        pfields_["data_top"][0],
                    )
                ]
            if self._have_data("data_start"):
                dtype += [
                    (
                        (
                            f"Waveform Start for {infoline[type_]} [ PE / sample ]",
                            f"{type_}_data_start",
                        ),
                        pfields_["data_start"][0],
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
        dtype += strax.time_fields
        return dtype

    def compute(self, events, peaks):
        result = np.zeros(len(events), self.dtype)
        result["time"] = events["time"]
        result["endtime"] = strax.endtime(events)

        split_peaks = strax.split_by_containment(peaks, events)
        for event_i, (event, sp) in enumerate(zip(events, split_peaks)):
            for type_ in ["s1", "s2", "alt_s1", "alt_s2"]:
                type_index = event[f"{type_}_index"]
                if type_index != -1:
                    result[f"{type_}_length"][event_i] = sp["length"][type_index]
                    result[f"{type_}_data"][event_i] = sp["data"][type_index]
                    if self._have_data("data_top"):
                        result[f"{type_}_data_top"][event_i] = sp["data_top"][type_index]
                    if self._have_data("data_start"):
                        result[f"{type_}_data_start"][event_i] = sp["data_start"][type_index]
                    result[f"{type_}_dt"][event_i] = sp["dt"][type_index]
        return result
