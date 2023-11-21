import strax
import straxen
from straxen import pax_file, first_sr1_run
from straxen.common import get_resource
import tempfile
import numpy as np
import json
import os

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        "nn_architecture",
        infer_type=False,
        help="Path to JSON of neural net architecture",
        default_by_run=[
            (0, pax_file("XENON1T_tensorflow_nn_pos_20171217_sr0.json")),
            (
                first_sr1_run,
                straxen.aux_repo
                + "3548132b55f81a43654dba5141366041e1daaf01/strax_files/"
                "XENON1T_tensorflow_nn_pos_20171217_sr1_reformatted.json",
            ),
        ],
    ),  # noqa
    strax.Option(
        "nn_weights",
        infer_type=False,
        help="Path to HDF5 of neural net weights",
        default_by_run=[
            (0, pax_file("XENON1T_tensorflow_nn_pos_weights_20171217_sr0.h5")),
            (first_sr1_run, pax_file("XENON1T_tensorflow_nn_pos_weights_20171217_sr1.h5")),
        ],
    ),  # noqa
    strax.Option(
        "min_reconstruction_area",
        help="Skip reconstruction if area (PE) is less than this",
        default=10,
        infer_type=False,
    ),
    strax.Option(
        "n_top_pmts", default=straxen.n_top_pmts, infer_type=False, help="Number of top PMTs"
    ),
)
class PeakPositions1T(strax.Plugin):
    """Compute the S2 (x,y)-position based on a neural net."""

    dtype = [
        ("x", np.float32, "Reconstructed S2 X position (cm), uncorrected"),
        ("y", np.float32, "Reconstructed S2 Y position (cm), uncorrected"),
    ] + strax.time_fields
    depends_on = "peaks"
    provides = "peak_positions"

    # Parallelization doesn't seem to make it go faster
    # Is there much pure-python stuff in tensorflow?
    # Process-level paralellization might work, but you'd have to do setup
    # in each process, which probably negates the benefits,
    # except for huge chunks
    parallel = False

    __version__ = "0.1.1"

    def setup(self):
        import tensorflow as tf

        keras = tf.keras
        nn_conf = get_resource(self.config["nn_architecture"], fmt="json")
        # badPMTList was inserted by a very clever person into the keras json
        # file. Let's delete it to prevent future keras versions from crashing.
        # Do NOT try `del nn_conf['badPMTList']`! See get_resource docstring
        # for the gruesome details.
        bad_pmts = nn_conf["badPMTList"]
        nn = keras.models.model_from_json(
            json.dumps({k: v for k, v in nn_conf.items() if k != "badPMTList"})
        )
        self.pmt_mask = ~np.in1d(np.arange(self.config["n_top_pmts"]), bad_pmts)

        # Keras needs a file to load its weights. We can't put the load
        # inside the context, then it would break on Windows,
        # because there temporary files cannot be opened again.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(get_resource(self.config["nn_weights"], fmt="binary"))
            fname = f.name
        nn.load_weights(fname)
        os.remove(fname)
        self.nn = nn

    def compute(self, peaks):
        result = np.ones(len(peaks), dtype=self.dtype)
        result["time"], result["endtime"] = peaks["time"], strax.endtime(peaks)
        result["x"] *= float("nan")
        result["y"] *= float("nan")

        # Keep large peaks only
        peak_mask = peaks["area"] > self.config["min_reconstruction_area"]
        if not np.sum(peak_mask):
            # Nothing to do, and .predict crashes on empty arrays
            return result

        # Input: normalized hitpatterns in good top PMTs
        _in = peaks["area_per_channel"][peak_mask, :]
        _in = _in[:, : self.config["n_top_pmts"]][:, self.pmt_mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            _in /= _in.sum(axis=1).reshape(-1, 1)

        # Output: positions in mm (unfortunately), so convert to cm
        _out = self.nn.predict(_in) / 10

        # Set output in valid rows. Do NOT try result[peak_mask]['x']
        # unless you want all NaN positions (boolean masks make a copy unless
        # they are used as the last index)
        result["x"][peak_mask] = _out[:, 0]
        result["y"][peak_mask] = _out[:, 1]
        return result
