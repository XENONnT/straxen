import unittest
from hypothesis import strategies, given, settings, example
from straxen.plugins.event_patternfit import binom_test


@settings(deadline=None)
@given(strategies.floats(0., 1.), strategies.floats(1e-3, 1.-1e-3), strategies.floats(1., 1000))
@example(0., 0.05, 10.) #AFT_observed, AFT_expected (from xyz), S1_total
def test_patternfit_stats(aftobs, aft, s1tot):
    s1top = aftobs*s1tot
    assert(binom_test(s1top, s1tot, aft) == 1.)