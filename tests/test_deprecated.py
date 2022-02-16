"""
These tests are for deprecated functions that we will remove in future releases

This is as such a bit of a "to do list" of functions to remove from straxen
"""

import straxen
import matplotlib.pyplot as plt
import unittest


class TestDeprecated(unittest.TestCase):
    def test_context_config_overwrite(self):
        with self.assertWarns(DeprecationWarning):
            straxen.contexts.xenonnt_online(
                _database_init=False,
                _context_config_overwrite={'a': 1})

    def test_tight_layout(self):
        with self.assertWarns(DeprecationWarning):
            plt.scatter([1], [2])
            straxen.quiet_tight_layout()
            plt.clf()
