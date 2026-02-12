import os
import strax
from unittest import TestCase
import tempfile
from straxen.test_utils import nt_test_run_id
import straxen


class PluginTestAccumulator:
    """Accumulator for test functions for unit-testing such that all plugin related unit tests can
    be run on the same data within a single unit-test.

    Use example:
    ```python
        from _core import PluginTestAccumulator


        @PluginTestAccumulator.register('test_example')
        def test_example(self, # You should always accept self as an argument!):
            raise ValueError('Test failed')
    ```

    """

    # See URLConfigs for the original inspiration.
    @classmethod
    def register(cls, test_name, func=None):
        def wrapper(func):
            if not isinstance(test_name, str):
                raise ValueError("test_name name must be a string.")
            if not test_name.startswith("test"):
                raise ValueError(f"Tests should start with test_.., got {test_name} for {func}")
            if hasattr(cls, test_name):
                raise ValueError(f"{test_name} already used!")
            setattr(cls, test_name, func)
            return func

        return wrapper(func) if func is not None else wrapper


class PluginTestCase(TestCase):
    """Class for type hinting of PluginTest."""

    run_id: str
    st: strax.Context


class SetupContextNt(PluginTestCase):
    """Class for managing tests that depend on specific plugins and require a bit of data to run the
    test (provided by straxen.test_utils.nt_test_context).

    Don't add tests directly, but add using the `@PluginTestAccumulator.register`-decorator (see
    straxen/tests/plugins/README.md)

    """

    exclude_plugins = (
        "events_gps_nv",
        "events_gps_mv",
        "gps_sync",
        "veto_intervals_gps_sync",
        "ref_mon_nv",
        "peak_s1_positions_cnn",
        "event_s1_positions_cnn",
    )

    @classmethod
    def setUpClass(cls) -> None:
        """Common setup for all the tests.

        We need some data which we don't delete but reuse to prevent a lot of computations in this
        class. Only after running all the tests, we run the cleanup.

        """
        # Context can be controlled via STRAXEN_TEST_CONTEXT environment variable.
        # Default is 'xenonnt' (SOM plugins), set to 'xenonnt_online' to test vanilla plugins.
        context_name = os.environ.get("STRAXEN_TEST_CONTEXT", "xenonnt")
        cls.st = straxen.test_utils.nt_test_context(context_name)
        cls.run_id = nt_test_run_id

        # Remove excluded plugins from registry (GPS and CNN only)
        for plugin_name in cls.exclude_plugins:
            cls.st._plugin_class_registry.pop(plugin_name, None)

        # For online context, set ML model configs to None to avoid model download issues
        # Plugins will still run but return NaN for position fields
        if context_name == "xenonnt_online":
            cls.st.set_config(
                {
                    "tf_model_mlp": None,
                }
            )

        # Make sure that we only write to the temp-dir we cleanup after each test
        cls.st.storage[0].readonly = True
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.st.storage.append(strax.DataDirectory(cls.tempdir.name))

    @classmethod
    def tearDownClass(cls) -> None:
        """Removes test data after tests are done."""
        cls.tempdir.cleanup()


def run_pytest_from_main():
    """Build new unit test for provided functions.

    For example, you might want to run it for a single module, in that case you don't want to run
    ALL the tests. So you can do e.g. `python peak_building.py` where we only collect the tests
    defined in that module.

    """
    import unittest

    class Test(SetupContextNt, PluginTestAccumulator):
        pass

    test_suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(test_suite)
