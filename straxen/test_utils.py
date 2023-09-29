import strax
import straxen
import tarfile
import io
import os
from warnings import warn
from os import environ as os_environ
from immutabledict import immutabledict
from importlib import import_module
import numpy as np


export, __all__ = strax.exporter()


nt_test_run_id = "012882"


@export
def download_test_data(
    test_data="https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/353b2c60a01e96f67e4ba544ce284bd91241964d/strax_files/strax_test_data_straxv1.1.0.tar",  #  noqa
):
    """Downloads strax test data to strax_test_data in the current
    directory."""
    blob = straxen.common.get_resource(test_data, fmt="binary")
    f = io.BytesIO(blob)
    tf = tarfile.open(fileobj=f)
    tf.extractall()


@export
def _overwrite_testing_function_file(function_file):
    """For testing purposes allow this function file to be loaded from
    HOME/testing_folder."""
    if not _is_on_pytest():
        # If we are not on a pytest, never try using a local file.
        return function_file

    home = os.environ.get("HOME")
    if home is None:
        # Impossible to load from non-existent folder
        return function_file

    testing_file = os.path.join(home, function_file)

    if os.path.exists(testing_file):
        # For testing purposes allow loading from 'home/testing_folder'
        warn(
            f"Using local function: {function_file} from {testing_file}! "
            f"If you are not integrated testing on github you should "
            f"absolutely remove this file. (See #559)"
        )
        function_file = testing_file

    return function_file


def is_installed(module):
    try:
        import_module(module)
        return True
    except ModuleNotFoundError:
        return False


@export
def _is_on_pytest():
    """Check if we are on a pytest."""
    return "PYTEST_CURRENT_TEST" in os_environ


def _get_fake_daq_reader():
    class DAQReader(straxen.DAQReader):
        """Dummy version of the DAQ reader to make sure that all the testing
        data produced here will have a different lineage."""

        __version__ = "MOCKTESTDATA"

    return DAQReader


def nt_test_context(
    target_context="xenonnt_online", deregister=(), keep_default_storage=False, **kwargs
) -> strax.Context:
    """Get a dummy context with full nt-like data simulated data (except aqmon)
    to allow testing plugins.

    :param target_context: Which contexts from straxen.contexts to test
    :param deregister: a list of plugins from the context
    :param keep_default_storage: if to True, keep the default context
        storage. Usually, you don't need this since all the data will be
        stored in a separate test data folder.
    :param kwargs: Any kwargs are passed to the target-context
    :return: a context
    """
    if not straxen.utilix_is_configured(warning_message=False):
        kwargs.setdefault("_database_init", False)

    st = getattr(straxen.contexts, target_context)(**kwargs)
    st.set_config(
        {"diagnose_sorting": True, "diagnose_overlapping": True, "store_per_channel": True}
    )
    st.register(_get_fake_daq_reader())
    download_test_data(
        "https://raw.githubusercontent.com/XENONnT/"
        "strax_auxiliary_files/"
        "f0d177401e11408b273564f0e29df77528e83d26/"
        "strax_files/"
        "012882-raw_records-z7q2d2ye2t.tar"
    )
    if keep_default_storage:
        st.storage += [strax.DataDirectory("./strax_test_data")]
    else:
        st.storage = [strax.DataDirectory("./strax_test_data")]
    assert st.is_stored(nt_test_run_id, "raw_records"), os.listdir(st.storage[-1].path)

    to_remove = list(deregister)
    for plugin in to_remove:
        del st._plugin_class_registry[plugin]
    return st


def create_unique_intervals(size, time_range=(0, 40), allow_zero_length=True):
    """Hypothesis stragtegy which creates unqiue time intervals.

    :param size: Number of intervals desired. Can be less if non-unique
        intervals are found.
    :param time_range: Time range in which intervals should be.
    :param allow_zero_length: If true allow zero length intervals.
    """
    from hypothesis import strategies

    strat = strategies.lists(
        elements=strategies.integers(*time_range), min_size=size * 2, max_size=size * 2
    ).map(lambda x: _convert_to_interval(x, allow_zero_length))
    return strat


def _convert_to_interval(time_stamps, allow_zero_length):
    time_stamps = np.sort(time_stamps)
    intervals = np.zeros(len(time_stamps) // 2, strax.time_dt_fields)
    intervals["dt"] = 1
    intervals["time"] = time_stamps[::2]
    intervals["length"] = time_stamps[1::2] - time_stamps[::2]

    if not allow_zero_length:
        intervals = intervals[intervals["length"] > 0]
    return np.unique(intervals)


@strax.takes_config(
    strax.Option("secret_time_offset", default=0, track=False),
    strax.Option("recs_per_chunk", default=10, track=False),
    strax.Option(
        "n_chunks",
        default=2,
        track=False,
        help="Number of chunks for the dummy raw records we are writing here",
    ),
    strax.Option(
        "channel_map",
        track=False,
        type=immutabledict,
        help="frozendict mapping subdetector to (min, max) " "channel number.",
    ),
)
class DummyRawRecords(strax.Plugin):
    """Provide dummy raw records for the mayor raw_record types."""

    provides = straxen.daqreader.DAQReader.provides
    parallel = "process"
    depends_on = tuple()
    data_kind = immutabledict(zip(provides, provides))
    rechunk_on_save = False
    dtype = {p: strax.raw_record_dtype() for p in provides}

    def setup(self):
        self.channel_map_keys = {
            "he": "he",
            "nv": "nveto",
            "aqmon": "aqmon",
            "aux_mv": "aux_mv",
            "s_mv": "mv",
        }  # s_mv otherwise same as aux in endswith

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < self.config["n_chunks"]

    def compute(self, chunk_i):
        t0 = chunk_i + self.config["secret_time_offset"]
        if chunk_i < self.config["n_chunks"] - 1:
            # One filled chunk
            r = np.zeros(self.config["recs_per_chunk"], self.dtype["raw_records"])
            r["time"] = t0
            r["length"] = r["dt"] = 1
            r["channel"] = np.arange(len(r))
        else:
            # One empty chunk
            r = np.zeros(0, self.dtype["raw_records"])

        res = {}
        for p in self.provides:
            rr = np.copy(r)
            # Add detector specific channel offset:
            for key, channel_key in self.channel_map_keys.items():
                if channel_key not in self.config["channel_map"]:
                    # Channel map for 1T is different.
                    continue
                if p.endswith(key):
                    first_channel, last_channel = self.config["channel_map"][channel_key]
                    rr["channel"] += first_channel
                    if key == "aqmon":
                        # explicitly clip these channels as we have an additional check higher in the chain
                        first_channel = int(min(straxen.AqmonChannels))
                        last_channel = int(max(straxen.AqmonChannels))

                    rr = rr[(rr["channel"] >= first_channel) & (rr["channel"] < last_channel)]
            res[p] = self.chunk(start=t0, end=t0 + 1, data=rr, data_type=p)
        return res
