import straxen
import matplotlib.pyplot as plt


def test_tight_layout():
    plt.scatter([1], [2])
    straxen.quiet_tight_layout()
    plt.clf()
