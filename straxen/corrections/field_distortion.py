'''
# Field distortion correction
The detector is designed to have uniform electric field, however, it's never perfectly uniform, especially near the TPC edges. If PTFE panels gets charged up, this is even more serious. In order to make analysis much easier, we can correct the field distortion effect. We can do two different approach, the performance is evaluated by studying the uniformity of reconstruct events (Kr83m, Rn220 etc):

 * Simulation driven. It's performance in XENON100 detector was pretty good, but not in XENON1T.
 * Purely data-driven approach. By definition its performance on data is good.

See [description in the Team C overview page](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:reconstruction_team#correction_against_field_distortion)

'''

import strax

from .base_references import BaseMap

export, __all__ = strax.exporter()


@export
class FdcMap(BaseMap):
    _NAME = "fdc_maps"
    fmt = 'json.gz'
