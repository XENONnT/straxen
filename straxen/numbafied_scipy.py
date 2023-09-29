import numba
import ctypes
import strax
from scipy.special.cython_special import loggamma, betainc, gammaln

export, __all__ = strax.exporter()
# Numba versions of scipy functions:
# Based on http://numba.pydata.org/numba-doc/latest/extending/high-level.html but
# omitting the address step.
# Unfortunately this approach does not allow to cache the resulting numba functions.
# Any ideas in this regard would be welcome.
# All subsequent function will treat their inputs as 64bit floats:
double = ctypes.c_double

functype = ctypes.CFUNCTYPE(double, double)
gammaln_float64 = functype(gammaln)


@export
@numba.njit
def numba_gammaln(x):
    return gammaln_float64(x)


functype = ctypes.CFUNCTYPE(double, double, double, double)
betainc_float64 = functype(betainc)


@export
@numba.njit
def numba_betainc(x1, x2, x3):
    return betainc_float64(x1, x2, x3)


functype = ctypes.CFUNCTYPE(double, double)
loggamma_float64 = functype(loggamma)


@export
@numba.njit
def numba_loggamma(x):
    return loggamma_float64(x)
