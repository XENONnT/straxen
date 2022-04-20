import strax
import straxen
import unittest
from straxen.test_utils import nt_test_run_id
import os
import shutil
from _core import PluginTestAccumulator

# Need import to attach new tests to the PluginTestAccumulator
import event_building


# Don't bother with remote tests
@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class PluginTest(unittest.TestCase, PluginTestAccumulator):
    exclude_plugins = 'events_sync_mv', 'events_sync_nv'

    @classmethod
    def setUpClass(cls) -> None:
        """
        Common setup for all the tests. We need some data which we
        don't delete but reuse to prevent a lot of computations in this
        class
        """
        cls.st = straxen.test_utils.nt_test_context()
        cls.run_id = nt_test_run_id

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Removes test data after tests are done.
        """
        path = os.path.abspath(cls.st.storage[-1].path)
        for file in os.listdir(path):
            shutil.rmtree(os.path.join(path, file))


# Very important step! We add a test for each of the plugins
for _target in set(straxen.test_utils.nt_test_context()._plugin_class_registry.values()):
    # Only run one test per plugin (even if it provides multiple targets)
    _target = strax.to_str_tuple(_target.provides)[0]

    if _target in PluginTest.exclude_plugins:
        continue

    # pylint disable=cell-var-from-loop
    @PluginTestAccumulator.register(f'test_{_target}')
    def _make(self, target=_target):
        self.st.make(self.run_id, target)
