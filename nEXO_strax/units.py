"""Define unit system for pax (i.e., seconds, etc.)

This sets up variables for the various unit abbreviations, ensuring we always
have a 'consistent' unit system.  There are almost no cases that you should
change this without talking with a maintainer.
"""

from __future__ import division

# From physics.nist.gov, January 2015
electron_charge_SI = 1.602176565 * 10 ** (-19)
boltzmannConstant_SI = 1.3806488 * 10 ** (-23)

m = 10 ** 2  # distances in cm
s = 10 ** 9  # times in ns
eV = 1  # energies in eV
C = 1 / electron_charge_SI  # Charge in # electrons, so voltage in Volts
K = 1  # Temperature in Kelvins

# derived units
Hz = 1 / s
J = eV / electron_charge_SI
kg = J * s ** 2 / m ** 2
V = J / C
A = C / s
N = J / m
Pa = N / m ** 2
bar = 10 ** 5 * Pa
Ohm = V / A

# 10 ^ -3 base units
mm = 10 ** (-3) * m
ms = 10 ** (-3) * s
mK = 10 ** (-3) * K
mC = 10 ** (-3) * C
meV = 10 ** (-3) * eV

mHz = 10 ** (-3) * Hz
mJ = 10 ** (-3) * J
g = 10 ** (-3) * kg
mV = 10 ** (-3) * V
mA = 10 ** (-3) * A
mN = 10 ** (-3) * N
mPa = 10 ** (-3) * Pa
mbar = 10 ** (-3) * bar
mOhm = 10 ** (-3) * Ohm

# 10 ^ -6 base units
um = 10 ** (-6) * m
us = 10 ** (-6) * s
uK = 10 ** (-6) * K
uC = 10 ** (-6) * C
ueV = 10 ** (-6) * eV

uHz = 10 ** (-6) * Hz
uJ = 10 ** (-6) * J
mg = 10 ** (-6) * kg
uV = 10 ** (-6) * V
uA = 10 ** (-6) * A
uN = 10 ** (-6) * N
uPa = 10 ** (-6) * Pa
ubar = 10 ** (-6) * bar
uOhm = 10 ** (-6) * Ohm

# 10 ^ -9 base units
nm = 10 ** (-9) * m
ns = 10 ** (-9) * s
nK = 10 ** (-9) * K
nC = 10 ** (-9) * C
neV = 10 ** (-9) * eV

nHz = 10 ** (-9) * Hz
nJ = 10 ** (-9) * J
ug = 10 ** (-9) * kg
nV = 10 ** (-9) * V
nA = 10 ** (-9) * A
nN = 10 ** (-9) * N
nPa = 10 ** (-9) * Pa
nbar = 10 ** (-9) * bar
nOhm = 10 ** (-9) * Ohm

# 10 ^ 3 base units
km = 10 ** 3 * m
ks = 10 ** 3 * s
kK = 10 ** 3 * K
kC = 10 ** 3 * C
keV = 10 ** 3 * eV

kHz = 10 ** 3 * Hz
kJ = 10 ** 3 * J
Mg = 10 ** 3 * kg
kV = 10 ** 3 * V
kA = 10 ** 3 * A
kN = 10 ** 3 * N
kOhm = 10 ** 3 * Ohm
kbar = 10 ** 3 * bar
kPa = 10 ** 3 * Pa


# 10 ^ 6 base units
Mm = 10 ** 6 * m
Ms = 10 ** 6 * s
MK = 10 ** 6 * K
MC = 10 ** 6 * C
MeV = 10 ** 6 * eV

MHz = 10 ** 6 * Hz
MJ = 10 ** 6 * J
Gg = 10 ** 6 * kg
MV = 10 ** 6 * V
MA = 10 ** 6 * A
MN = 10 ** 6 * N
MOhm = 10 ** 6 * Ohm
Mbar = 10 ** 6 * bar
MPa = 10 ** 6 * Pa


# 10 ^ 9 base units
Gm = 10 ** 9 * m
Gs = 10 ** 9 * s
GK = 10 ** 9 * K
GC = 10 ** 9 * C
GeV = 10 ** 9 * eV

GHz = 10 ** 9 * Hz
GJ = 10 ** 9 * J
GV = 10 ** 9 * V
GA = 10 ** 9 * A
GN = 10 ** 9 * N
GOhm = 10 ** 9 * Ohm
Gbar = 10 ** 9 * bar
GPa = 10 ** 9 * Pa

# other units
cm = 10 ** (-2) * m
ng = 10 ** (-12) * kg
# Townsend (unit for reduced electric field)
Td = 10 ** (-17) * V / cm ** 2  # noqa

electron_charge = electron_charge_SI * C
boltzmannConstant = boltzmannConstant_SI * J / K
