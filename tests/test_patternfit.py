from hypothesis import strategies, given, settings, example
from straxen.plugins.events.event_pattern_fit import (
    binom_test,
    lbinom_pmf_mode,
    lbinom_pmf,
    lbinom_pmf_inverse,
    binom_sf,
)
import numpy as np
import scipy.stats as sps


@settings(deadline=None)
@given(strategies.floats(0.0, 1.0), strategies.floats(0.01, 0.99), strategies.floats(2.0, 1000))
def test_patternfit_stats(aftobs, aft, s1tot):
    s1top = aftobs * s1tot
    assert (binom_test(s1top, s1tot, aft) >= 0) & (binom_test(s1top, s1tot, aft) <= 1)


@settings(deadline=None)
@given(strategies.floats(0.0, 1.0), strategies.floats(0.01, 0.99), strategies.floats(2.0, 1000))
def test_inverse_pdf(aftobs, aft, s1tot):
    s1top = aftobs * s1tot
    # code stolen from binom_test, checking if the other point on

    k = s1top
    n = s1tot
    p = aft

    mode = lbinom_pmf_mode(0, n, 0, (n, p))
    distance = abs(mode - k)
    target = lbinom_pmf(k, n, p)

    if k < mode:
        j_min = mode
        j_max = min(mode + 2.0 * distance, n)
        j = lbinom_pmf_inverse(j_min, j_max, target, (n, p))
    else:
        j_min = max(mode - 2.0 * distance, 0)
        j_max = mode
        j = lbinom_pmf_inverse(j_min, j_max, target, (n, p))
    if np.isnan(j) or j == n:
        # if no other crossing is found, we cannot check if the two give same pdf
        assert True
    else:
        pdf_value = lbinom_pmf(j, n, p)
        # values in log, it is not _really_ crucial that they're precisely equal
        np.testing.assert_almost_equal(pdf_value, target, decimal=2)


@settings(deadline=None)
@strategies.composite
def n_and_k(draw):
    n = draw(strategies.integers(2, 1000))
    k = draw(strategies.integers(0, n))
    return n, k


@settings(deadline=None)
@given(nk=n_and_k(), p=strategies.floats(0.01, 0.99))
@example((10, 5), 0.5)
def test_pmf(nk, p):
    # test that at integer n, k binom_pmf agrees with the scipy result
    n, k = nk
    sps_lpmf = sps.binom(n, p).logpmf(k)
    straxen_lpmf = lbinom_pmf(k, n, p)
    np.testing.assert_almost_equal(straxen_lpmf, sps_lpmf, decimal=2)


@settings(deadline=None)
@given(nk=n_and_k(), p=strategies.floats(0.01, 0.99))
def test_pvalue(nk, p):
    # test that at integer n, k binom_pmf agrees with the scipy result
    n, k = nk
    sps_pval = sps.binom(n, p).sf(k + 0.1)
    straxen_pval = binom_sf(k, n, p)
    np.testing.assert_almost_equal(straxen_pval, sps_pval, decimal=2)
