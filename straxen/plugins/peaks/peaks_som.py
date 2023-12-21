import strax
import numpy as np
from straxen.plugins.peaks.peaks import Peaks

export, __all__ = strax.exporter()


@export
class PeaksSOM(Peaks):
    """Same as Peaks but include in addition SOM type field to be propagated to event_basics.

    Thus, only change dtype.

    """

    __version__ = "0.0.1"
    child_plugin = True

    def infer_dtype(self):
        peaklet_classification_dtype = self.deps["peaklet_classification"].dtype_for(
            "peaklet_classification"
        )
        peaklets_dtype = self.deps["peaklets"].dtype_for("peaklets")
        # The merged dtype is argument position dependent!
        # It must be first classification then peaklet
        # Otherwise strax will raise an error when checking for the returned dtype!
        merged_s2s_dtype = strax.merged_dtype((peaklet_classification_dtype, peaklets_dtype))
        return merged_s2s_dtype

    def compute(self, peaklets, merged_s2s):
        result = super().compute(peaklets, merged_s2s)

        # For merged S2s SOM and straxen type are undefined:
        _is_merged_s2 = np.isin(result["time"], merged_s2s["time"]) & np.isin(
            strax.endtime(result), strax.endtime(merged_s2s)
        )
        result["straxen_type"][_is_merged_s2] = -1
        result["som_sub_type"][_is_merged_s2] = -1

        return result


@export
class PeaksSOMClassification(strax.Plugin):
    """Plugin which propagates S1 SOM infromation to peaks if straxen.PeakClassification is still
    used."""

    depends_on = ("peak_basics", "peaklet_classification_som")
    __version__ = "0.0.1"

    provides = "peak_som_classification"

    def infer_dtype(self):
        dtype_peaklets = strax.time_fields + [
            ("som_sub_type", np.int32, "SOM subtype of the peak(let)"),
            ("som_type", np.int8, "SOM type of the peak(let)"),
            ("loc_x_som", np.int16, "x location of the peak(let) in the SOM"),
            ("loc_y_som", np.int16, "y location of the peak(let) in the SOM"),
        ]
        return dtype_peaklets

    def compute(self, peaklets, peaks):
        result = np.zeros(len(peaks), self.dtype)
        result[:] = -1
        result["time"] = peaks["time"]
        result["endtime"] = peaks["endtime"]

        # Only select S1 and update their SOM fields:
        _is_s1_peaks = peaks["type"] == 1
        _is_s1_som_peaklets = peaklets["straxen_type"] == 1
        _are_same_s1_peaks = np.all(
            result[_is_s1_peaks]["time"] == peaklets[_is_s1_som_peaklets]["time"]
        )
        assert _are_same_s1_peaks, "S1 peaks and SOM S1 peaklets are not identical?!?"

        # Copy "manually" because only subset of data which does not work
        # with copy to buffer:
        for field in ("som_sub_type", "som_type", "loc_x_som", "loc_y_som"):
            result[field][_is_s1_peaks] = peaklets[field][_is_s1_som_peaklets]

        return result
