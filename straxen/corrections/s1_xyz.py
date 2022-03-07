'''
# S1 (x,y,z) correction
S1 signals in LXeTPC has a large position dependence across the whole volume. In order to make a precise energy reconstruction, we need a “correction” map to overcome its spatial dependence, as the case for S2.

See [description in the Team C overview page](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:reconstruction_team#s1_x_y_z_correction)

The jupyter notebook in this folder is replaced by [this](https://github.com/XENONnT/nton/blob/master/nton/analyses/corrections/s1correction.py) script in the nton file

'''

import strax

from .base_references import BaseMap

export, __all__ = strax.exporter()


@export
class S1XYZMap(BaseMap):
    _NAME = "s1_xyz_maps"    

