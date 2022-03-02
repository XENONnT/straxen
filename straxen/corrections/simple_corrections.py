import strax
import rframe

from .base_corrections import TimeIntervalCorrection, TimeSampledCorrection

export, __all__ = strax.exporter()


@export
class BaselineSamples(TimeIntervalCorrection):
    _NAME = "baseline_samples"
    detector: str = rframe.Index()
    value: int

@export
class ElectronDriftVelocity(TimeIntervalCorrection):
    _NAME = "electron_drift_velocities"
    value: float

@export
class ElectronLifetime(TimeIntervalCorrection):
    _NAME = "electron_lifetimes"
    value: float


@export
class HitThresholds(TimeIntervalCorrection):
    _NAME = "hit_thresholds"
    detector: str = rframe.Index()
    pmt: int = rframe.Index()

    value: int

@export
class RelExtractionEff(TimeSampledCorrection):
    _NAME = "rel_extraction_eff"
    value: float

