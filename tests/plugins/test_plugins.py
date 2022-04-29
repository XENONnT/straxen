import strax
import straxen
import unittest
import tempfile
from straxen.test_utils import nt_test_run_id
from _core import PluginTestCase, PluginTestAccumulator

# Need import to attach new tests to the PluginTestAccumulator
import bayes_plugin
import event_building
import peak_building
import posrec_plugins


# Don't bother with remote tests
@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class PluginTest(PluginTestCase, PluginTestAccumulator):
    """
    Class for managing tests that depend on specific plugins and
    require a bit of data to run the test
    (provided by straxen.test_utils.nt_test_context).

    Don't add tests directly, but add using the
    `@PluginTestAccumulator.register`-decorator (see
    straxen/tests/plugins/README.md)
    """
    exclude_plugins = 'events_sync_mv', 'events_sync_nv'

    @classmethod
    def setUpClass(cls) -> None:
        """
        Common setup for all the tests. We need some data which we
        don't delete but reuse to prevent a lot of computations in this
        class. Only after running all the tests, we run the cleanup.
        """
        cls.st = straxen.test_utils.nt_test_context()
        cls.run_id = nt_test_run_id

        # Make sure that we only write to the temp-dir we cleanup after each test
        cls.st.storage[0].readonly = True
        cls.tempdir = tempfile.TemporaryDirectory()
        cls.st.storage.append(strax.DataDirectory(cls.tempdir.name))

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Removes test data after tests are done.
        """
        cls.tempdir.cleanup()


# Very important step! We add a test for each of the plugins
for _target in set(straxen.test_utils.nt_test_context()._plugin_class_registry.values()):
    # Only run one test per plugin (even if it provides multiple targets)
    _target = strax.to_str_tuple(_target.provides)[0]
    if _target in PluginTest.exclude_plugins:
        continue

    test_name = f'test_{_target}'
    if hasattr(PluginTestAccumulator, test_name):
        # We already made a dedicated test, let's skip this
        continue

    # pylint: disable=cell-var-from-loop
    @PluginTestAccumulator.register(test_name)
    def _make(self, target=_target):
        self.st.make(self.run_id, target)
