import numpy as np
import strax
import straxen

from straxen.plugins.defaults import DEFAULT_POSREC_ALGO, FAKE_MERGED_S2_TYPE

export, __all__ = strax.exporter()


@export
class PeakPositionsNT(strax.MergeOnlyPlugin):
    """Merge the reconstructed algorithms of the different algorithms into a single one that can be
    used in Event Basics.

    Select one of the plugins to provide the 'x' and 'y' to be used further down the chain. Since we
    already have the information needed here, there is no need to wait until events to make the
    decision.

    Since the computation is trivial as it only combined the three input plugins, don't save this
    plugins output.

    """

    provides = "peak_positions"
    depends_on = (
        "peak_positions_mlp",
        "peak_positions_cnf",
    )
    save_when = strax.SaveWhen.NEVER
    __version__ = "0.0.0"

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )

    def infer_dtype(self):
        dtype = strax.merged_dtype([self.deps[d].dtype_for(d) for d in self.depends_on])
        dtype += [
            ("x", np.float32, "Reconstructed S2 X position (cm), uncorrected"),
            ("y", np.float32, "Reconstructed S2 Y position (cm), uncorrected"),
        ]
        return dtype

    def compute(self, peaks):
        result = {dtype: peaks[dtype] for dtype in peaks.dtype.names}
        algorithm = self.default_reconstruction_algorithm
        for xy in ("x", "y"):
            result[xy] = peaks[f"{xy}_{algorithm}"]
        return result


@export
class PeakletPositionsNT(PeakPositionsNT):

    __version__ = "0.0.0"
    provides = "peaklet_positions"
    depends_on = (
        "peaklet_positions_mlp",
        "peaklet_positions_cnf",
    )

    def compute(self, peaklets):
        return super().compute(peaklets)


@export
class MergedPeakPositionsNT(strax.Plugin):

    __version__ = "0.0.0"

    depends_on = ("peaklet_positions", "peaklet_classification", "merged_s2s")
    data_kind = "peaks"
    provides = "peak_positions"

    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO, help="default reconstruction algorithm that provides (x,y)"
    )

    merge_without_s1 = straxen.URLConfig(
        default=True,
        infer_type=False,
        help=(
            "If true, S1s will be igored during the merging. "
            "It's now possible for a S1 to be inside a S2 post merging"
        ),
    )

    def infer_dtype(self):
        dtype = self.deps["peaklet_positions"].dtype_for("peaklet_positions")
        return dtype

    def compute(self, peaklets, merged_s2s):
        # Remove fake merged S2s from dirty hack, see above
        merged_s2s = merged_s2s[merged_s2s["type"] != FAKE_MERGED_S2_TYPE]

        if self.merge_without_s1:
            is_s1_peaklets = peaklets["type"] == 1
            _peaklets = peaklets[~is_s1_peaklets]
        else:
            _peaklets = peaklets
        windows = strax.touching_windows(_peaklets, merged_s2s)

        _merged_s2 = np.zeros(len(merged_s2s), dtype=peaklets.dtype)
        indices = np.full(len(_peaklets), -1)

        for i, (start, end) in enumerate(windows):
            indices[start:end] = i
            for name in peaklets.dtype.names:
                _merged_s2[name][i] = np.nanmean(_peaklets[name][start:end], axis=0)

        _merged_s2["time"] = merged_s2s["time"]
        _merged_s2["endtime"] = strax.endtime(merged_s2s)

        # TODO: We have to make sure that the sorting here is the same to in the Peaks plugin
        # because maybe different peaklets can have same time
        _result = strax.sort_by_time(np.concatenate([_peaklets[indices == -1], _merged_s2]))

        if self.merge_without_s1:
            _result = strax.sort_by_time(np.concatenate([peaklets[is_s1_peaklets], _result]))

        result = np.zeros(len(_result), dtype=self.dtype)
        strax.copy_to_buffer(_result, result, "_copy_requested_peak_positions_fields")
        return result
