import strax
import straxen
import unittest
from _core import PluginTestAccumulator, SetupContextNt

# Need import to attach new tests to the PluginTestAccumulator
import bayes_plugin
import event_building
import peak_building
import posrec_plugins
import pulse_processing
import nv_processing


# Don't bother with remote tests
@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class PluginTest(SetupContextNt, PluginTestAccumulator):
    """_CoreTest with tests registered"""


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


class TestEmptyRecords(PluginTest):
    """Run the tests again, but this time with empty raw-records"""

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.st.register(straxen.test_utils.DummyRawRecords)
        cls.st.set_context_config(dict(forbid_creation_of=()))
