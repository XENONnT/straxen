import numpy as np
import strax
import straxen


export, __all__ = strax.exporter()


@export
class PeakBasicsVanilla(strax.Plugin):
    """Compute the basic peak-properties, thereby dropping structured arrays.

    NB: This plugin can therefore be loaded as a pandas DataFrame.

    """

    __version__ = "0.1.7"
    depends_on = "peaks"
    provides = "peak_basics"

    check_peak_sum_area_rtol = straxen.URLConfig(
        default=None,
        track=False,
        infer_type=False,
        help=(
            "Check if the sum area and the sum of area per "
            "channel are the same. If None, don't do the "
            "check. To perform the check, set to the desired "
            " rtol value used e.g. '1e-4' (see np.isclose)."
        ),
    )

    n_top_pmts = straxen.URLConfig(default=straxen.n_top_pmts, type=int, help="Number of top PMTs")

    def infer_dtype(self):
        dtype = strax.time_fields + [
            (("Weighted average center time of the peak [ns]", "center_time"), np.int64),
            (("Peak integral in PE", "area"), np.float32),
            (("Number of hits contributing at least one sample to the peak", "n_hits"), np.int32),
            (("Number of PMTs contributing to the peak", "n_channels"), np.int16),
            (("Number of top PMTs contributing to the peak", "top_n_channels"), np.int16),
            (("PMT number which contributes the most PE", "max_pmt"), np.int16),
            (("Area of signal in the largest-contributing PMT (PE)", "max_pmt_area"), np.float32),
            (("Total number of saturated channels", "n_saturated_channels"), np.int16),
            (("Width (in ns) of the central 50% area of the peak", "range_50p_area"), np.float32),
            (("Width (in ns) of the central 90% area of the peak", "range_90p_area"), np.float32),
            (("Fraction of area seen by the top array", "area_fraction_top"), np.float32),
            (("Length of the peak waveform in samples", "length"), np.int32),
            (("Time resolution of the peak waveform in ns", "dt"), np.int16),
            (("Weighted relative median time of the peak [ns]", "median_time"), np.float32),
            (("Time between 10% and 50% area quantiles [ns]", "rise_time"), np.float32),
            (
                ("Number of PMTs with hits within tight range of mean", "tight_coincidence"),
                np.int16,
            ),
            (("Classification of the peak(let)", "type"), np.int8),
            (("Peak is merged from peaklet", "merged"), bool),
            (
                ("Largest time difference between apexes of hits inside peak [ns]", "max_diff"),
                np.int32,
            ),
            (
                ("Smallest time difference between apexes of hits inside peak [ns]", "min_diff"),
                np.int32,
            ),
            (
                (
                    "First channel/PMT number inside peak (sorted by apexes of hits)",
                    "first_channel",
                ),
                np.int16,
            ),
            (
                ("Last channel/PMT number inside peak (sorted by apexes of hits)", "last_channel"),
                np.int16,
            ),
        ]
        return dtype

    def compute(self, peaks):
        p = peaks
        r = np.zeros(len(p), self.dtype)
        needed_fields = "time center_time length dt median_time area area_fraction_top type "
        needed_fields += "merged max_diff min_diff first_channel last_channel"
        for q in needed_fields.split():
            r[q] = p[q]
        r["endtime"] = p["time"] + p["dt"] * p["length"]
        r["n_channels"] = (p["area_per_channel"] > 0).sum(axis=1)
        r["top_n_channels"] = (p["area_per_channel"][:, : self.n_top_pmts] > 0).sum(axis=1)
        r["n_hits"] = p["n_hits"]
        r["range_50p_area"] = p["width"][:, 5]
        r["range_90p_area"] = p["width"][:, 9]
        r["max_pmt"] = np.argmax(p["area_per_channel"], axis=1)
        r["max_pmt_area"] = np.max(p["area_per_channel"], axis=1)
        r["tight_coincidence"] = p["tight_coincidence"]
        r["n_saturated_channels"] = p["n_saturated_channels"]
        r["rise_time"] = -p["area_decile_from_midpoint"][:, 1]

        if self.check_peak_sum_area_rtol is not None:
            area_total = p["area_per_channel"].sum(axis=1)
            self.check_area(area_total, p, self.check_peak_sum_area_rtol)
        return r

    @staticmethod
    def check_area(area_per_channel_sum, peaks, rtol) -> None:
        """Check if the area of the sum-wf is the same as the total area (if the area of the peak is
        positively defined).

        :param area_per_channel_sum: the summation of the peaks['area_per_channel'] which will be
            checked against the values of peaks['area'].
        :param peaks: array of peaks.
        :param rtol: relative tolerance for difference between area_per_channel_sum and
            peaks['area']. See np.isclose.
        :raises: ValueError if the peak area and the area-per-channel sum are not sufficiently close

        """
        positive_area = peaks["area"] > 0
        if not np.sum(positive_area):
            return

        is_close = np.isclose(
            area_per_channel_sum[positive_area],
            peaks[positive_area]["area"],
            rtol=rtol,
        )

        if not is_close.all():
            for peak in peaks[positive_area][~is_close]:
                print("bad area")
                strax.print_record(peak)

            p_i = np.where(~is_close)[0][0]
            peak = peaks[positive_area][p_i]
            area_fraction_off = 1 - area_per_channel_sum[positive_area][p_i] / peak["area"]
            message = (
                "Area not calculated correctly, it's "
                f"{100 * area_fraction_off} % off, time: {peak['time']}"
            )
            raise ValueError(message)
