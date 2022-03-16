import unittest
from hypothesis import strategies, given, settings, example
from straxen.plugins.event_patternfit import binom_pmf, binom_cdf


@settings(deadline=None)
@given(strategies.floats(0.,1.),strategies.floats(1e-3,1.-1e-3),strategies.floats(1.,1000))
@example(1.,0.45,10.) #AFT_observed, AFT_expected (from xyz), S1_total
def test_patternfit_stats(aftobs, aft, s1tot):
    #PDF is positive:
    s1top = aftobs*s1tot
    assert(0. <= binom_pmf(s1top, s1tot, aft))
    #CDF is positive, between 0 and 1, and 0 at 0 and 1 at s1tot:
    assert(0. <= binom_cdf(s1top, s1tot, aft))
    assert(binom_cdf(s1top, s1tot, aft) <=1.)
    assert(0. == binom_cdf(0., s1tot, aft))
    assert(1. == binom_cdf(s1tot, s1tot, aft))
