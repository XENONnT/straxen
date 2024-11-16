import numpy as np
import strax
import straxen
from straxen.plugins.hitlets_nv.hitlets_nv import nVETOHitlets
from straxen.plugins.defaults import MV_PREAMBLE, MV_HIT_DEFAULTS

export, __all__ = strax.exporter()


@export
class muVETOHitlets(nVETOHitlets):
    __doc__ = MV_PREAMBLE + (nVETOHitlets.__doc__ or "")
    __version__ = "0.0.2"
    depends_on = "records_mv"

    provides = "hitlets_mv"
    data_kind = "hitlets_mv"
    child_plugin = True

    dtype = strax.hitlet_dtype()

    save_outside_hits_mv = straxen.URLConfig(
        default=MV_HIT_DEFAULTS["save_outside_hits_mv"],
        track=True,
        infer_type=False,
        child_option=True,
        parent_option_name="save_outside_hits_nv",
        help="Save (left, right) samples besides hits; cut the rest",
    )

    hit_min_amplitude_mv = straxen.URLConfig(
        infer_type=False,
        default=MV_HIT_DEFAULTS["hit_min_amplitude_mv"],
        track=True,
        help=(
            "Minimum hit amplitude in ADC counts above baseline. "
            "Specify as a tuple of length n_mveto_pmts, or a number, "
            "or a tuple like (correction=str, version=str, nT=boolean),"
            "which means we are using cmt."
        ),
    )

    min_split_mv = straxen.URLConfig(
        default=100,
        track=True,
        infer_type=False,
        child_option=True,
        parent_option_name="min_split_nv",
        help=(
            "Minimum height difference pe/sample between local minimum and maximum, "
            "that a pulse get split."
        ),
    )

    min_split_ratio_mv = straxen.URLConfig(
        default=0,
        track=True,
        infer_type=False,
        child_option=True,
        parent_option_name="min_split_ratio_nv",
        help=(
            "Min ratio between local maximum and minimum to split pulse (zero to switch this off)."
        ),
    )

    entropy_template_mv = straxen.URLConfig(
        default="flat",
        track=True,
        infer_type=False,
        child_option=True,
        parent_option_name="entropy_template_nv",
        help=(
            'Template data is compared with in conditional entropy. Can be either "flat" or a '
            "template array."
        ),
    )

    entropy_square_data_mv = straxen.URLConfig(
        default=False,
        track=True,
        infer_type=False,
        child_option=True,
        parent_option_name="entropy_square_data_nv",
        help=(
            "Parameter which decides if data is first squared before normalized and compared to "
            "the template."
        ),
    )

    gain_model_mv = straxen.URLConfig(
        default="cmt://to_pe_model_mv?version=ONLINE&run_id=plugin.run_id",
        infer_type=False,
        child_option=True,
        parent_option_name="gain_model_nv",
        help="PMT gain model. Specify as (model_type, model_config)",
    )

    def setup(self):
        self.channel_range = self.channel_map["mv"]
        self.n_channel = (self.channel_range[1] - self.channel_range[0]) + 1

        to_pe = self.gain_model_mv
        # Create to_pe array of size max channel:
        self.to_pe = np.zeros(self.channel_range[1] + 1, dtype=np.float32)
        self.to_pe[self.channel_range[0] :] = to_pe[:]

        self.hit_thresholds = self.hit_min_amplitude_mv

    def compute(self, records_mv, start, end):
        return super().compute(records_mv, start, end)
