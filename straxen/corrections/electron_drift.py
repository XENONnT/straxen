'''
Electron Drift Velocity and Drift Time at Gate

The Understanding the shape of S2s is a prerequisite for calibrating the trigger, processor, and the S2 width cut. Drift time and velocity are important parameters needed for the event z-reconstruction and to understand the S2 shape.
The drift time is calculated as the time difference between events originate at gate and cathode. The drift velocity is then extracted dividing the XENONnT cathode-gate distance of 1485 mm. Since z coordinate starts at the gate,
while the drift time starts at the liquid level, therefore we need to subtract the electron drift time at the gate to get
the proper z coordinate, as well as to remove events happening above the gate.

See note on drift velocity, diffusion constant, and drift time at the gate
    https://xe1t-wiki.lngs.infn.it/doku.php?id=dandrea:diffusionstudywithkrdata

'''

import strax
import rframe

from .base_corrections import TimeIntervalCorrection, TimeSampledCorrection

export, __all__ = strax.exporter()


@export
class ElectronDriftVelocity(TimeIntervalCorrection):
    _NAME = "electron_drift_velocities"
    value: float

@export
class DriftTimeGate(TimeIntervalCorrection):
    _NAME = "electron_drift_time_gates"
    value: float
