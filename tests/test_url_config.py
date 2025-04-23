import os
import json
import random
import warnings
from bson import json_util
from datetime import datetime
import pickle
import tempfile
import fsspec
import unittest
import pandas as pd
import numpy as np
import utilix.rundb
import strax
import straxen
from straxen.test_utils import nt_test_context, nt_test_run_id
from straxen.plugins.defaults import DEFAULT_POSREC_ALGO


class DummyObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@straxen.URLConfig.register("random")
def generate_random(_):
    return random.random()


@straxen.URLConfig.register("range")
def generate_range(length):
    length = int(length)
    return np.arange(length)


@straxen.URLConfig.register("unpicklable")
def return_lamba(_):
    return lambda x: x


@straxen.URLConfig.register("large-array")
def large_array(_):
    return np.ones(1_000_000).tolist()


@straxen.URLConfig.register("object-list")
def object_list(length):
    length = int(length)
    return [DummyObject(a=i, b=i + 1) for i in range(length)]


@straxen.URLConfig.preprocessor
def formatter(config, **kwargs):
    if not isinstance(config, str):
        return config
    try:
        config = config.format(**kwargs)
    except KeyError:
        pass
    return config


GLOBAL_VERSIONS = {"global_v1": {"test_config": "v0"}}


@straxen.URLConfig.preprocessor
def replace_global_version(config, name=None, **kwargs):
    if name is None:
        return

    if not isinstance(config, str):
        return config

    if straxen.URLConfig.SCHEME_SEP not in config:
        return config

    version = straxen.URLConfig.kwarg_from_url(config, "version")

    if version is None:
        return config

    if version.startswith("global_") and version in GLOBAL_VERSIONS:
        version = GLOBAL_VERSIONS[version].get(name, version)
        config = straxen.URLConfig.format_url_kwargs(config, version=version)
    return config


class ExamplePlugin(strax.Plugin):
    depends_on = ()
    dtype = strax.time_fields
    provides = "test_data"
    test_config = straxen.URLConfig(
        default=42,
    )
    cached_config = straxen.URLConfig(default=666, cache=1)


class AlgorithmPlugin(strax.Plugin):
    depends_on = "raw_records"
    dtype = strax.time_fields
    provides = "test_data"
    allow_superrun = True
    default_reconstruction_algorithm = straxen.URLConfig(
        default=DEFAULT_POSREC_ALGO,
    )
    superruns_test_config_a = straxen.URLConfig(
        default=(
            'take://json://{"'
            + nt_test_run_id
            + '":0,"_'
            + nt_test_run_id
            + '":1}?take=plugin.run_id'
        ),
    )
    superruns_test_config_b = straxen.URLConfig(
        default='take://json://{"cnf":0,"mlp":1}?take=plugin.default_reconstruction_algorithm',
    )
    global_version_test_config = straxen.URLConfig(
        default="format://{version}?version=v0",
    )


class TestURLConfig(unittest.TestCase):
    def setUp(self):
        st = nt_test_context("xenonnt")
        st.register(ExamplePlugin)
        self.st = st

    def test_default(self):
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, 42)

    def test_literal(self):
        self.st.set_config({"test_config": 666})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, 666)

    def test_leading_zero_int(self):
        self.st.set_config({"test_config": "format://{value}?value=0666"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, "0666")
        self.assertIsInstance(p.test_config, str)

    def test_json_protocol(self):
        self.st.set_config({"test_config": 'json://{"a":0}'})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, {"a": 0})

    def test_format_protocol(self):
        self.st.set_config({"test_config": "format://{run_id}?run_id=plugin.run_id"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, nt_test_run_id)

    def test_fsspec_protocol(self):
        with fsspec.open("memory://test_file.json", mode="w") as f:
            json.dump({"value": 999}, f)
        self.st.set_config(
            {"test_config": "take://json://fsspec://memory://test_file.json?take=value"}
        )
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, 999)

    def test_take_nested(self):
        self.st.set_config(
            {
                "test_config": (
                    'take://json://{"a":{"aa":0,"ab":1},"b":{"ba":2,"bb":3}}?take=b&take=ba'
                )
            }
        )
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, 2)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_bodedga_get(self):
        """Just a didactic example."""
        self.st.set_config(
            {
                "test_config": (
                    "take://resource://XENONnT_numbers.json?fmt=json&take=g1&take=v2&take=value"
                )
            }
        )
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        # Either g1 is 0, bodega changed or someone broke URLConfigs
        self.assertTrue(p.test_config)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test xedocs.")
    def test_itp_dict(self, ab_value=20, cd_value=21, dump_as="json"):
        """Test that we are getting ~the same value from interpolating at the central date in a
        dict.

        :param ab_value, cd_value: some values to test against
        :param dump_as: Write as csv or as json file

        """
        central_datetime = (
            utilix.rundb.xent_collection()
            .find_one({"number": int(nt_test_run_id)}, projection={"start": 1})
            .get("start", "QUERY FAILED!")
        )
        fake_file = {
            "time": [
                datetime(2000, 1, 1).timestamp() * 1e9,
                central_datetime.timestamp() * 1e9,
                datetime(2040, 1, 1).timestamp() * 1e9,
            ],
            "ab": [10, ab_value, 30],
            "cd": [11, cd_value, 31],
        }

        temp_dir = tempfile.TemporaryDirectory()

        if dump_as == "json":
            fake_file_name = os.path.join(temp_dir.name, "test_seg.json")
            with open(fake_file_name, "w") as f:
                json.dump(fake_file, f)
        elif dump_as == "csv":
            # This example also works well with dataframes!
            fake_file_name = os.path.join(temp_dir.name, "test_seg.csv")
            pd.DataFrame(fake_file).to_csv(fake_file_name)
        else:
            raise ValueError

        self.st.set_config(
            {
                "test_config": (
                    "itp_dict://"
                    "resource://"
                    f"{fake_file_name}"
                    "?run_id=plugin.run_id"
                    f"&fmt={dump_as}"
                    "&itp_keys=ab,cd"
                )
            }
        )
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertIsInstance(p.test_config, dict)
        assert np.isclose(p.test_config["ab"], ab_value, rtol=1e-3)
        assert np.isclose(p.test_config["cd"], cd_value, rtol=1e-3)
        temp_dir.cleanup()

    def test_itp_dict_csv(self):
        self.test_itp_dict(dump_as="csv")

    def test_rekey(self):
        original_dict = {"a": 1, "b": 2, "c": 3}
        check_dict = {"anew": 1, "bnew": 2, "cnew": 3}

        temp_dir = tempfile.TemporaryDirectory()

        fake_file_name = os.path.join(temp_dir.name, "test_dict.json")
        with open(fake_file_name, "w") as f:
            json.dump(original_dict, f)

        self.st.set_config(
            {
                "test_config": (
                    f"rekey_dict://resource://{fake_file_name}?"
                    "fmt=json&replace_keys=a,b,c"
                    "&with_keys=anew,bnew,cnew"
                )
            }
        )
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, check_dict)
        temp_dir.cleanup()

    def test_print_protocol_desc(self):
        straxen.URLConfig.print_protocols()

    def test_cache(self):
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")

        # sanity check that default value is not affected
        self.assertEqual(p.cached_config, 666)
        self.st.set_config({"cached_config": "random://abc"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")

        # value is randomly generated when accessed so if
        # its equal when we access it again, its coming from the cache
        cached_value = p.cached_config
        self.assertEqual(cached_value, p.cached_config)

        # now change the config to which will generate a new number
        self.st.set_config({"cached_config": "random://dfg"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")

        # sanity check that the new value is still consistent i.e. cached
        self.assertEqual(p.cached_config, p.cached_config)

        # test if previous value is evicted, since cache size is 1
        self.assertNotEqual(cached_value, p.cached_config)

        # verify pickalibility of objects in cache dont affect plugin pickalibility
        self.st.set_config({"cached_config": "unpicklable://dfg"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        with self.assertRaises(AttributeError):
            pickle.dumps(p.cached_config)
        pickle.dumps(p)

    def test_cache_size(self):
        """Test the cache helper functions."""
        # make sure the value has a detectable size
        self.st.set_config({"cached_config": "large-array://dfg"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")

        # fetch the value so its stored in the cache
        p.cached_config

        # cache should now have finite size
        self.assertGreater(straxen.config_cache_size_mb(), 0.0)

        # test if clearing cache works as expected
        straxen.clear_config_caches()
        self.assertEqual(straxen.config_cache_size_mb(), 0.0)

    def test_filter_kwargs(self):
        all_kwargs = dict(a=1, b=2, c=3)

        # test a function that takes only a seubset of the kwargs
        def func1(a=None, b=None):
            return

        filtered1 = straxen.filter_kwargs(func1, all_kwargs)
        self.assertEqual(filtered1, dict(a=1, b=2))
        func1(**filtered1)

        # test function that accepts wildcard kwargs
        def func2(**kwargs):
            return

        filtered2 = straxen.filter_kwargs(func2, all_kwargs)
        self.assertEqual(filtered2, all_kwargs)
        func2(**filtered2)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test xedocs.")
    def test_dry_evaluation(self):
        """Check that running a dry evaluation can be done outside of the context of a URL config
        and yield the same result."""
        plugin_url = (
            "xedocs://electron_drift_velocities?attr=value&run_id=plugin.run_id&version=ONLINE"
        )
        self.st.set_config({"test_config": plugin_url})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        correct_val = p.test_config

        # We can also get it from one of these methods
        dry_val1 = straxen.URLConfig.evaluate_dry(
            f"xedocs://electron_drift_velocities?attr=value&run_id={nt_test_run_id}&version=ONLINE"
        )
        dry_val2 = straxen.URLConfig.evaluate_dry(
            f"xedocs://electron_drift_velocities?attr=value&version=ONLINE", run_id=nt_test_run_id
        )

        # All methods should yield the same
        assert correct_val == dry_val1 == dry_val2

        # However dry-evaluation does NOT allow loading the plugin.run_id
        # as in the plugin_url and should complain about that
        with self.assertRaises(ValueError):
            straxen.URLConfig.evaluate_dry(plugin_url)

    def test_objects_to_dict(self):
        n = 3
        self.st.set_config(
            {"test_config": f"objects-to-dict://object-list://{n}?key_attr=a&value_attr=b"}
        )
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, {i: i + 1 for i in range(n)})

    def test_list_to_array(self):
        n = 3
        self.st.set_config({"test_config": f"list-to-array://object-list://{n}"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertIsInstance(p.test_config, np.ndarray)

    def test_format_preprocessor(self):
        self.st.set_config({"test_config": "{name}:{run_id}"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, f"test_config:{nt_test_run_id}")
        self.assertEqual(p.test_config, p.config["test_config"])

    def test_global_version_preprocessor(self):
        self.st.set_config({"test_config": "fake://url?version=global_v1"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, "fake://url?version=v0")

    def test_sort_url_kwargs(self):
        """URLConfig preprocessor to rearange the order of arguments given buy a url to ensure the
        same url with a different hash order gives the same hash."""
        url = "xedocs://electron_lifetimes?run_id=034678&version=v5&attr=value"
        intended_url = "xedocs://electron_lifetimes?attr=value&run_id=034678&version=v5"
        preprocessed_url = straxen.config.preprocessors.sort_url_kwargs(url)
        self.assertEqual(intended_url, preprocessed_url)

    def test_xedocs_global_version_hash_coinsistency(self):
        # Same URLs but the queries are in a different order
        st1 = self.st.new_context(
            config={"test_config": "fake://electron_lifetimes?run_id=25000&version=v5&attr=value"}
        )
        st2 = self.st.new_context(
            config={"test_config": "fake://electron_lifetimes?attr=value&run_id=25000&version=v5"}
        )
        self.assertEqual(
            st1.key_for("025000", "corrected_areas").lineage_hash,
            st2.key_for("025000", "corrected_areas").lineage_hash,
        )

    def test_global_version_not_changed(self):
        """
        - if no global version is matched, the url version should not be changed
        - if config is not matched, the url version should not be changed
        """
        assert "global_v2" not in GLOBAL_VERSIONS
        self.st.set_config({"test_config": "fake://url?version=global_v2"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, "fake://url?version=global_v2")

        class ExamplePluginNew(ExamplePlugin):
            test_config_new = straxen.URLConfig(
                default="rootisthesourceofallevil",
            )

        self.st.register(ExamplePluginNew)
        self.st.set_config({"test_config_new": "fake://url?version=global_v1"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config_new, "fake://url?version=global_v1")

    @unittest.skipIf(
        not straxen.utilix_is_configured(), "No db access, cannot test run_doc protocol."
    )
    def test_run_doc_protocol(self):
        self.st.set_config({"test_config": "run_doc://mode?run_id=plugin.run_id"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, "tpc_commissioning")

        self.st.set_config({"test_config": "run_doc://fake_key?run_id=plugin.run_id&default=42"})
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(p.test_config, 42)

        with self.assertRaises(ValueError):
            self.st.set_config({"test_config": "run_doc://mode?run_id=plugin.run_id"})
            p = self.st.get_single_plugin("999999999", "test_data")
            return p.test_config

    def test_regex_url_warnings(self):

        url = "xedocs://electron_lifetimes?verion=v5&att=value"  # url with typos
        self.st.set_config({"test_config": url})

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            # Trigger a warning.
            self.st.get_single_plugin(nt_test_run_id, "test_data")

            # Verify the warning
            assert len(w) != 0, "Error, warning dispatcher not working"

    def test_pad_array(self):
        """Test that pad_array works as expected."""
        n = 3
        self.st.set_config(
            {"test_config": f"pad-array://range://{n}?pad_left=2&pad_right=3&pad_value=0"}
        )
        p = self.st.get_single_plugin(nt_test_run_id, "test_data")
        self.assertEqual(len(p.test_config), 8)
        self.assertEqual(p.test_config[0], 0)
        self.assertEqual(p.test_config[-1], 0)

    def test_not_cmt_check(self):
        """Expect error when using cmt."""
        with self.assertRaises(NotImplementedError):
            straxen.config.check_urls("cmt")

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_superruns_safeguard(self):
        """Test that the superruns safeguard works as expected."""

        # test all configs can be checked
        st = self.st.new_context()
        for data_type in st._plugin_class_registry:
            if st._plugin_class_registry[data_type].depends_on:
                st._plugin_class_registry[data_type].allow_superrun = True
        configs = st.time_dependent_configs(
            "_" + nt_test_run_id, ("event_info", "events_nv", "events_mv")
        )
        st.hashed_url_configs(configs)

        # test the safeguard works
        st = self.st.new_context()
        st.set_context_config(
            {"plugin_attr_convert": st.context_config["plugin_attr_convert"] + ("take",)}
        )
        st.register((AlgorithmPlugin,))
        start = pd.to_datetime(0, unit="ns", utc=True)
        end = pd.to_datetime(1, unit="ns", utc=True)
        run_doc = {"name": nt_test_run_id, "start": start, "end": end}
        with open(st.storage[0]._run_meta_path(str(nt_test_run_id)), "w") as fp:
            json.dump(run_doc, fp, default=json_util.default)
        st.define_run("_" + nt_test_run_id, [nt_test_run_id])
        st.get_components("_" + nt_test_run_id, ("test_data",))
        default = AlgorithmPlugin.takes_config["superruns_test_config_a"].default
        st.set_config({"superruns_test_config_a": default.replace("run_id", "_run_id")})
        with self.assertRaises(NotImplementedError):
            # the raise NotImplementedError is expected from compute method
            st.get_components("_" + nt_test_run_id, ("test_data",), combining=True)
        with self.assertRaises(ValueError):
            # the raise ValueError is expected from get_components in the wrapper
            st.get_components("_" + nt_test_run_id, ("test_data",))

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_global_version_safeguard(self):
        """Test that the global_version safeguard works as expected."""

        # test all configs can be checked
        st = self.st.new_context()
        st.set_context_config({"xedocs_version": "global_OFFLINE"})
        st.register((AlgorithmPlugin,))
        st.set_config({"global_version_test_config": "format://{version}?version=ONLINE"})
        with self.assertRaises(ValueError):
            st.get_components(nt_test_run_id, ("test_data",))
