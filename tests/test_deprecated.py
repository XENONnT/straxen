"""These tests are for deprecated functions that we will remove in future releases.

This is as such a bit of a "to do list" of functions to remove from straxen

"""

import straxen
import unittest


class TestDeprecated(unittest.TestCase):
    def test_context_config_overwrite(self):
        with self.assertWarns(DeprecationWarning):
            straxen.contexts.xenonnt_online(
                _database_init=False, _context_config_overwrite={"a": 1}
            )
