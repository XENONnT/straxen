import straxen
import unittest
from straxen.test_utils import nt_test_run_id
import os
import shutil
from importlib import import_module

_magic_key = 'plugin_test_'


# Somehow the registering cases some issues with importing this main
# class elsewhere (pytest won't recognize the setUpClass anymore, ideas
# to fix are welcome)
@unittest.skipIf(not __file__.endswith('test_plugins.py'),
                 'Only run from main testing file')
# Don't bother with remote tests
@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class PluginTest(unittest.TestCase):
    exclude_plugins = 'events_sync_mv', 'events_sync_nv'

    @classmethod
    def register(cls, test_name, func=None):
        # See URLConfigs for the original insparation.
        def wrapper(func):
            if not isinstance(test_name, str):
                raise ValueError('test_name name must be a string.')

            setattr(cls, test_name, func)
            return func
        return wrapper(func) if func is not None else wrapper

    @classmethod
    def setUpClass(cls) -> None:
        """
        Common setup for all the tests. We need some data which we
        don't delete but reuse to prevent a lot of computations in this
        class
        """
        cls.st = straxen.test_utils.nt_test_context()
        cls.run_id = nt_test_run_id
        # cls.st.make(cls.run_id, 'records')

    @classmethod
    def tearDownClass(cls) -> None:
        """
        Removes test data after tests are done.
        """
        path = os.path.abspath(cls.st.storage[-1].path)
        for file in os.listdir(path):
            shutil.rmtree(os.path.join(path, file))


# Very important step! We add a test for each of the plugins
for t in straxen.test_utils.nt_test_context()._plugin_class_registry.keys():
    if t in PluginTest.exclude_plugins:
        continue

    @PluginTest.register(f'test_{t}')
    def _make(self, target=t):
        self.st.make(self.run_id, target)


# There is probably a cleaner step but add all the functions in any .py
# file in this dir to the tests
for file in os.listdir(os.path.dirname(os.path.abspath(__file__))):
    if not file.endswith('.py') and file != str(__file__):
        continue

    module = import_module(file[:-len('.py')])

    for attr, _object in module.__dict__.items():
        if attr.startswith(_magic_key):
            PluginTest.register(f'test_{attr[len(_magic_key):]}', _object)
