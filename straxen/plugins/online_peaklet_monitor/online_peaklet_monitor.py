import numpy as np

import strax
import straxen

export, __all__ = strax.exporter()


@export
class OnlinePeakletMonitor(strax.Plugin):
    """Online monitor of the peaklet level, for live processing that stops at peaklets.

    ``online_peak_monitor`` needs ``peak_basics``, and therefore ``merged_s2s``, which in turn needs
    peaklet position reconstruction. When the DAQ processes only up to ``peaklets`` (for instance to
    keep up during high-rate calibrations), none of the existing online monitor data types can be
    made. This plugin provides the equivalent information from ``peaklets`` alone, at the cost of
    working with pre-merge peaklets rather than peaks.

    Its main purpose is monitoring the single-electron (SE) rate and gain, which is well suited to
    the peaklet level: single electrons are small S2s that are essentially never merged into larger
    S2s, so skipping ``merged_s2s`` does not bias them. The SE area spectrum is stored as a
    histogram rather than as a fitted number, so that the selection window can be revisited later
    without reprocessing.

    Like the other online monitor plugins, the output is written to the online monitor MongoDB,
    where one chunk becomes one BSON document with a hard 16 MB limit. The output size is therefore
    bounded by construction: the number of rows per chunk is ``peaklet_monitor_max_rows`` at most,
    independent of the rate and of the chunking of the input.

    """

    __version__ = "0.0.1"

    # Index of the width field holding the central 50% area range, as used by peak_basics
    width_index_50p = 5

    depends_on = ("peaklets", "peaklet_classification")
    provides = "online_peaklet_monitor"
    data_kind = "online_peaklet_monitor"

    peaklet_monitor_time_bin = straxen.URLConfig(
        type=int,
        default=int(10e9),
        help=(
            "Time bin of the peaklet monitor [ns]. Chunks shorter than this produce a single bin; "
            "longer chunks are divided into bins of at most this length."
        ),
    )

    peaklet_monitor_max_rows = straxen.URLConfig(
        type=int,
        default=200,
        track=True,
        help=(
            "Maximum number of rows (time bins) emitted per chunk. Bounds the size of the MongoDB "
            "document of the online monitor, at the cost of a coarser time binning for very long "
            "chunks."
        ),
    )

    peaklet_area_vs_width_nbins = straxen.URLConfig(
        type=int,
        default=40,
        help=(
            "Number of bins for the peaklet area vs width histogram of the online monitor. "
            "NB: this is a 2D histogram"
        ),
    )

    peaklet_area_vs_width_bounds = straxen.URLConfig(
        type=tuple,
        default=((-2, 5), (0, 5)),
        help="Boundaries of the log-log histogram of peaklet area vs width",
    )

    se_monitor_nbins = straxen.URLConfig(
        type=int,
        default=200,
        help="Number of bins of the single-electron area spectrum",
    )

    se_monitor_area_bounds = straxen.URLConfig(
        type=tuple,
        default=(0.0, 100.0),
        help="Boundaries of the single-electron area spectrum [PE]",
    )

    se_monitor_window = straxen.URLConfig(
        type=tuple,
        default=(15.0, 70.0),
        help=(
            "Area window counted as single electrons [PE]. The same window is used for the SE gain "
            "estimate, which is the mode of the spectrum inside it."
        ),
    )

    n_tpc_pmts = straxen.URLConfig(type=int, help="Number of TPC PMTs")

    def infer_dtype(self):
        n_bins_area_width = self.peaklet_area_vs_width_nbins
        dtype = strax.time_fields + [
            (("Number of peaklets", "n_peaklets"), np.int32),
            (("Peaklet rate [Hz]", "peaklet_rate"), np.float32),
            (("Number of S2-like peaklets", "n_s2_peaklets"), np.int32),
            (("Number of single-electron candidates", "n_se"), np.int32),
            (("Single-electron rate [Hz]", "se_rate"), np.float32),
            (("Single electron gain, mode of the spectrum [PE]", "se_gain"), np.float32),
            (
                ("Area spectrum of S2-like peaklets [PE]", "se_area_hist"),
                (np.int32, self.se_monitor_nbins),
            ),
            (("Bounds of the area spectrum [PE]", "se_area_bounds"), (np.float32, 2)),
            (
                ("Area fraction top histogram of single-electron candidates", "se_aft_hist"),
                (np.int32, 50),
            ),
            (
                ("Summed area per channel of single-electron candidates [PE]", "se_per_channel"),
                (np.float32, self.n_tpc_pmts),
            ),
            (
                ("Peaklet area vs width histogram (log-log)", "area_vs_width_hist"),
                (np.int64, (n_bins_area_width, n_bins_area_width)),
            ),
            (
                ("Peaklet area vs width edges (log-space)", "area_vs_width_bounds"),
                (np.float64, (2, 2)),
            ),
        ]
        return dtype

    def compute(self, peaklets, start, end):
        edges = self.time_bin_edges(start, end)
        res = np.zeros(len(edges) - 1, dtype=self.dtype)
        res["time"] = edges[:-1]
        res["endtime"] = edges[1:]
        res["se_area_bounds"] = self.se_monitor_area_bounds
        res["area_vs_width_bounds"] = self.peaklet_area_vs_width_bounds
        livetime = (res["endtime"] - res["time"]) / 1e9

        if not len(peaklets):
            return res

        # Which time bin does each peaklet belong to?
        bin_i = np.searchsorted(edges, peaklets["time"], side="right") - 1
        bin_i = np.clip(bin_i, 0, len(res) - 1)

        res["n_peaklets"] = np.bincount(bin_i, minlength=len(res))
        res["peaklet_rate"] = res["n_peaklets"] / livetime

        # Always cut out unphysical peaklets, as in the peak-level monitor
        physical = (peaklets["area"] > 0) & (peaklets["width"][:, self.width_index_50p] > 0)
        res["area_vs_width_hist"] = self.area_width_hist(
            bin_i[physical], peaklets[physical], len(res)
        )

        # Single electrons are small S2s, so only S2-like peaklets are considered
        is_s2 = peaklets["type"] == 2
        res["n_s2_peaklets"] = np.bincount(bin_i[is_s2], minlength=len(res))
        res["se_area_hist"] = self.binned_hist(
            bin_i[is_s2],
            peaklets["area"][is_s2],
            len(res),
            self.se_monitor_nbins,
            self.se_monitor_area_bounds,
        )

        se_low, se_high = self.se_monitor_window
        is_se = is_s2 & (peaklets["area"] > se_low) & (peaklets["area"] < se_high)
        res["n_se"] = np.bincount(bin_i[is_se], minlength=len(res))
        res["se_rate"] = res["n_se"] / livetime
        res["se_aft_hist"] = self.binned_hist(
            bin_i[is_se], peaklets["area_fraction_top"][is_se], len(res), 50, (0, 1)
        )
        np.add.at(res["se_per_channel"], bin_i[is_se], peaklets["area_per_channel"][is_se])
        res["se_gain"] = self.se_gain_estimate(res["se_area_hist"])
        return res

    def time_bin_edges(self, start, end):
        """Divide [start, end) into bins of at most ``peaklet_monitor_time_bin``, never more than
        ``peaklet_monitor_max_rows`` of them."""
        n_bins = int(np.ceil((end - start) / self.peaklet_monitor_time_bin))
        n_bins = np.clip(n_bins, 1, self.peaklet_monitor_max_rows)
        return np.linspace(start, end, n_bins + 1).astype(np.int64)

    @staticmethod
    def binned_hist(bin_i, values, n_rows, n_bins, bounds):
        """Histogram ``values`` in ``n_bins`` between ``bounds``, separately per time bin."""
        low, high = bounds
        value_i = np.floor((values - low) / (high - low) * n_bins).astype(np.int64)
        keep = (value_i >= 0) & (value_i < n_bins)
        flat = np.bincount(
            bin_i[keep] * n_bins + value_i[keep],
            minlength=n_rows * n_bins,
        )
        return flat.reshape(n_rows, n_bins)

    def area_width_hist(self, bin_i, peaklets, n_rows):
        """Make the area vs width 2D-histogram, separately per time bin.

        The axes match ``online_peak_monitor``: the stored histogram is indexed ``[time bin, width,
        area]``.

        """
        n_bins = self.peaklet_area_vs_width_nbins
        (area_low, area_high), (width_low, width_high) = self.peaklet_area_vs_width_bounds
        log_area = np.log10(peaklets["area"])
        log_width = np.log10(peaklets["width"][:, self.width_index_50p])

        area_i = np.floor((log_area - area_low) / (area_high - area_low) * n_bins).astype(np.int64)
        width_i = np.floor((log_width - width_low) / (width_high - width_low) * n_bins).astype(
            np.int64
        )
        keep = (area_i >= 0) & (area_i < n_bins) & (width_i >= 0) & (width_i < n_bins)
        flat = np.bincount(
            (bin_i[keep] * n_bins + width_i[keep]) * n_bins + area_i[keep],
            minlength=n_rows * n_bins * n_bins,
        )
        return flat.reshape(n_rows, n_bins, n_bins)

    def se_gain_estimate(self, se_area_hist):
        """Mode of the area spectrum inside the single-electron window, as in
        ``online_peak_monitor``."""
        low, high = self.se_monitor_area_bounds
        n_bins = self.se_monitor_nbins
        edges = np.linspace(low, high, n_bins + 1)
        bin_centers = (edges[1:] + edges[:-1]) / 2
        in_window = (bin_centers > self.se_monitor_window[0]) & (
            bin_centers < self.se_monitor_window[1]
        )
        gain = np.zeros(len(se_area_hist), dtype=np.float32)
        if not np.any(in_window):
            return gain
        has_data = se_area_hist[:, in_window].sum(axis=1) > 0
        gain[has_data] = bin_centers[in_window][
            np.argmax(se_area_hist[has_data][:, in_window], axis=1)
        ]
        return gain
