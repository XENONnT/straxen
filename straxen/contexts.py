import strax
import straxen

export, __all__ = strax.exporter()



@export
def demo_context():
    """Return strax context used in the straxen demo notebook"""
    return strax.Context(
            storage=[strax.DataDirectory('./strax_data'),
                     strax.DataDirectory(straxen.straxen_dir + '/data',
                                         readonly=True)],
            register=straxen.plugins.pax_interface.RecordsFromPax,
            register_all=straxen.plugins.plugins)
