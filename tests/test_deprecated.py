"""
These tests are for deprecated functions that we will remove in future releases

This is as such a bit of a "to do list" of functions to remove from straxen
"""

import straxen
import matplotlib.pyplot as plt


def test_tight_layout():
    plt.scatter([1], [2])
    straxen.quiet_tight_layout()
    plt.clf()
