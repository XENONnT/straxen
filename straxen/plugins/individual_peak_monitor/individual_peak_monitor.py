import numpy as np
import strax
import straxen

export, __all__ = strax.exporter()


@export
class IndividualPeakMonitor(strax.Plugin):
    """Plugin to write data needed for the online SE monitor to the online- monitor collection in
    the runs-database. Data that is written by this plugin should be small such as to not overload
    the runs- database. If the peaks are large, random max_bytes of data are selected from the
    peaks.

    This plugin takes 'peak_basics' and 'peak_positions_mlp'. Although they are not strictly
    related, they are aggregated into a single data_type in order to minimize the number of
    documents in the online monitor.

    Produces 'individual_peak_monitor' with info on the peaks and their positions.

    """

    online_max_bytes = straxen.URLConfig(
        default=6e6, track=True, help="Maximum amount of bytes of data for MongoDB document"
    )

    depends_on = ("peak_basics", "peak_positions_mlp")
    provides = "individual_peak_monitor"
    data_kind = "individual_peak_monitor"
    __version__ = "0.0.1"

    def infer_dtype(self):
        dtype = [
            (("Peak integral in PE", "area"), np.float32),
            (("Reconstructed mlp peak x-position", "x_mlp"), np.float32),
            (("Reconstructed mlp peak y-position", "y_mlp"), np.float32),
            (("Width (in ns) of the central 50% area of the peak", "range_50p_area"), np.float32),
            (("Fraction of original peaks array length that is saved", "weight"), np.float32),
        ] + strax.time_fields
        return dtype

    def compute(self, peaks):
        peaks_size = peaks.nbytes

        if peaks_size > self.online_max_bytes:
            # Calculate fraction of the data that can be kept
            # to reduce datasize
            new_len = int(len(peaks) / peaks_size * self.online_max_bytes)
            idx = np.random.choice(np.arange(len(peaks)), replace=False, size=new_len)
            data = peaks[strax.stable_sort(idx)]

        else:  # peaks_size <= self.max_bytes:
            data = peaks
        res = np.zeros(len(data), dtype=self.dtype)
        res["time"] = data["time"]
        res["x_mlp"] = data["x_mlp"]
        res["y_mlp"] = data["y_mlp"]
        res["area"] = data["area"]
        res["range_50p_area"] = data["range_50p_area"]
        res["endtime"] = data["endtime"]

        if len(data):
            res["weight"] = len(peaks) / len(data)
        else:
            res["weight"] = 0

        return res
