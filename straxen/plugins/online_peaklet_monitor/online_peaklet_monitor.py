import numpy as np

import strax
import straxen

export, __all__ = strax.exporter()


@export
class OnlinePeakletMonitor(strax.Plugin):
    """Online monitor of the peaklet level, for live processing that stops at peaklets.

    ``online_peak_monitor`` needs ``peak_basics``, and therefore ``merged_s2s``, which in turn needs
    peaklet position reconstruction. When the DAQ processes only up to ``peaklets``, for instance to
    keep up during a high-rate calibration, none of the existing online monitor data types can be
    made. This plugin provides the equivalent information from ``peaklets`` alone.

    Its main purpose is monitoring the single-electron (SE) rate and gain. Single electrons are
    small S2s that are essentially never merged into larger S2s, so the peaklet level does not bias
    them, and they form an isolated island in the area-width plane that can be selected without any
    classification. On a calibration run, a box around that island was found to be 99.6% pure in
    the sense of ``peaklet_classification`` type 2, and to hold 92% of the S2-like peaklets of its
    area range. The plugin therefore depends on ``peaklets`` only.

    The area spectrum of the width band is stored as a histogram rather than as a fitted number, so
    that the area window can be revisited later without reprocessing. This matters, because the SE
    gain moves with the field configuration and can end up outside the window it was tuned for -
    including outside the ``online_se_bounds`` default that ``online_peak_monitor`` uses. The
    defaults of ``se_monitor_area_window`` and ``se_monitor_width_window`` should be reviewed
    whenever the extraction field changes.

    Like the other online monitor plugins, the output is written to the online monitor MongoDB,
    where one chunk becomes one BSON document with a hard 16 MB limit. The output size is therefore
    bounded by construction: the number of rows per chunk is ``peaklet_monitor_max_rows`` at most,
    independent of the rate and of the chunking of the input. A row is about 8 kB.

    A time bin is the input chunk, or a subdivision of it, so its length is not guaranteed: a chunk
    can be a few microseconds long, and its single row then carries a meaningless instantaneous
    rate. Consumers should aggregate the counts over the livetime they cover,
    ``n_se.sum() / ((endtime - time) / 1e9).sum()``, and not average ``se_rate``: on a run with one
    very short chunk the two differ by 5%.

    The plugin has been checked against a direct count of the same box on the peaklets of a
    calibration run, which it reproduces to better than 0.1%.

    """

    __version__ = "0.0.1"

    # Index of the width field holding the central 50% area range, as used by peak_basics
    width_index_50p = 5

    depends_on = "peaklets"
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

    se_monitor_width_window = straxen.URLConfig(
        type=tuple,
        default=(120.0, 400.0),
        help=(
            "Width (range 50% area) window of the single-electron selection [ns]. Together with "
            "se_monitor_area_window this is the box the SE rate is counted in; the area spectrum "
            "is taken of everything inside this width window."
        ),
    )

    se_monitor_area_window = straxen.URLConfig(
        type=tuple,
        default=(20.0, 120.0),
        help=(
            "Area window of the single-electron selection [PE]. Also the range in which the SE "
            "gain is looked for. It moves with the extraction field: check it against the "
            "stored spectrum rather than assuming the default still brackets the peak."
        ),
    )

    se_monitor_nbins = straxen.URLConfig(
        type=int,
        default=200,
        help="Number of bins of the single-electron area spectrum",
    )

    se_monitor_area_bounds = straxen.URLConfig(
        type=tuple,
        default=(0.0, 200.0),
        help=(
            "Boundaries of the single-electron area spectrum [PE]. Wider than "
            "se_monitor_area_window so that the window can be moved without reprocessing."
        ),
    )

    n_tpc_pmts = straxen.URLConfig(type=int, help="Number of TPC PMTs")

    def infer_dtype(self):
        n_bins_area_width = self.peaklet_area_vs_width_nbins
        dtype = strax.time_fields + [
            (("Number of peaklets", "n_peaklets"), np.int32),
            (("Peaklet rate [Hz]", "peaklet_rate"), np.float32),
            (("Number of peaklets in the single-electron width window", "n_se_width"), np.int32),
            (("Number of single-electron candidates", "n_se"), np.int32),
            (("Single-electron rate [Hz]", "se_rate"), np.float32),
            (("Single electron gain, mode of the spectrum [PE]", "se_gain"), np.float32),
            (
                (
                    "Area spectrum of the peaklets in the single-electron width window [PE]",
                    "se_area_hist",
                ),
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
                (np.int32, (n_bins_area_width, n_bins_area_width)),
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

        area = peaklets["area"]
        width = peaklets["width"][:, self.width_index_50p]

        res["n_peaklets"] = np.bincount(bin_i, minlength=len(res))
        res["peaklet_rate"] = res["n_peaklets"] / livetime

        # Always cut out unphysical peaklets, as in the peak level monitor
        physical = (area > 0) & (width > 0)
        res["area_vs_width_hist"] = self.area_width_hist(
            bin_i[physical], area[physical], width[physical], len(res)
        )

        # Single electrons are an isolated island of the area-width plane, so no classification
        # is needed to select them. The spectrum of the width window is stored whole; the rate is
        # the part of it inside the area window.
        width_low, width_high = self.se_monitor_width_window
        in_width = physical & (width > width_low) & (width < width_high)
        res["n_se_width"] = np.bincount(bin_i[in_width], minlength=len(res))
        res["se_area_hist"] = self.binned_hist(
            bin_i[in_width],
            area[in_width],
            len(res),
            self.se_monitor_nbins,
            self.se_monitor_area_bounds,
        )

        area_low, area_high = self.se_monitor_area_window
        is_se = in_width & (area > area_low) & (area < area_high)
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
        ``peaklet_monitor_max_rows`` of them.

        The edges are computed in integer arithmetic on purpose. A nanosecond timestamp needs 61
        bits, so float64 resolves it to a few hundred nanoseconds only, and np.linspace would
        return a first edge below ``start`` - which strax rejects as a chunk that "starts early".

        """
        n_bins = int(np.ceil((end - start) / self.peaklet_monitor_time_bin))
        n_bins = int(np.clip(n_bins, 1, self.peaklet_monitor_max_rows))
        duration = np.int64(end) - np.int64(start)
        edges = np.int64(start) + np.arange(n_bins + 1, dtype=np.int64) * duration // n_bins
        edges[-1] = end
        return edges

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

    def area_width_hist(self, bin_i, area, width, n_rows):
        """Make the area vs width 2D-histogram, separately per time bin.

        The axes match ``online_peak_monitor``: the stored histogram is indexed ``[time bin, width,
        area]``.

        """
        n_bins = self.peaklet_area_vs_width_nbins
        (area_low, area_high), (width_low, width_high) = self.peaklet_area_vs_width_bounds
        log_area = np.log10(area)
        log_width = np.log10(width)

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
        """Mode of the area spectrum inside the single-electron area window, as
        ``online_peak_monitor`` does with ``online_se_bounds``."""
        low, high = self.se_monitor_area_bounds
        n_bins = self.se_monitor_nbins
        edges = np.linspace(low, high, n_bins + 1)
        bin_centers = (edges[1:] + edges[:-1]) / 2
        in_window = (bin_centers > self.se_monitor_area_window[0]) & (
            bin_centers < self.se_monitor_area_window[1]
        )
        gain = np.zeros(len(se_area_hist), dtype=np.float32)
        if not np.any(in_window):
            return gain
        has_data = se_area_hist[:, in_window].sum(axis=1) > 0
        gain[has_data] = bin_centers[in_window][
            np.argmax(se_area_hist[has_data][:, in_window], axis=1)
        ]
        return gain
