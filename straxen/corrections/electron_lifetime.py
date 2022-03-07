'''
# Electron lifetime
Thanks to the continuous purification, we expect an improved electron lifetime over time. However, a significant number of variations are expected in the XENONnT detector (due to operations etc), we thus need a model to evaluate the evolution of e-life for time dependent S2 corrections. Thanks to purity monitor, we will have sufficient amount of data for such a study in XENONnT.

Currently we are using data from the purity monitor via SCADA interface to predict elife values in the next 6 hours, then this information is pass to corrections DB and updated there. (See e-log for updates)

See [description in the Team C overview page](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:reconstruction_team#electron_lifetimeevolution_and_correction)

'''

import strax

from .base_corrections import TimeIntervalCorrection

export, __all__ = strax.exporter()


@export
class ElectronLifetime(TimeIntervalCorrection):
    _NAME = "electron_lifetimes"
    value: float
