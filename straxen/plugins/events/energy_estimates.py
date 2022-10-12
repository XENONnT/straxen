import numpy as np
import straxen



import strax
export, __all__ = strax.exporter()

@export
class EnergyEstimates(strax.Plugin):
    """
    Plugin which converts cS1 and cS2 into energies (from PE to KeVee).
    """
    __version__ = '0.1.1'
    depends_on = ['corrected_areas']
    dtype = [
        ('e_light', np.float32, 'Energy in light signal [keVee]'),
        ('e_charge', np.float32, 'Energy in charge signal [keVee]'),
        ('e_ces', np.float32, 'Energy estimate [keVee]')
    ] + strax.time_fields
    save_when = strax.SaveWhen.TARGET

    # config options don't double cache things from the resource cache!
    g1 = straxen.URLConfig(
        default='bodega://g1?bodega_version=v2',
        help="S1 gain in PE / photons produced",
    )
    g2 = straxen.URLConfig(
        default='bodega://g2?bodega_version=v2',
        help="S2 gain in PE / electrons produced",
    )
    lxe_w = straxen.URLConfig(
        default=13.7e-3,
        help="LXe work function in quanta/keV"
    )

    def compute(self, events):
        el = self.cs1_to_e(events['cs1'])
        ec = self.cs2_to_e(events['cs2'])
        return dict(e_light=el,
                    e_charge=ec,
                    e_ces=el + ec,
                    time=events['time'],
                    endtime=strax.endtime(events))

    def cs1_to_e(self, x):
        return self.lxe_w * x / self.g1

    def cs2_to_e(self, x):
        return self.lxe_w * x / self.g2