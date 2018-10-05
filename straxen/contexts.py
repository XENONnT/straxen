import strax
import straxen


def demo():
    """Return strax context used in the straxen demo notebook"""
    return straxen.XENONContext(
            storage=[strax.DataDirectory('./strax_data'),
                     strax.DataDirectory(straxen.straxen_dir + '/data',
                                         readonly=True)],
            register=straxen.plugins.pax_interface.RecordsFromPax,
            register_all=straxen.plugins.plugins)


def xenon1t_analysis():
    """Return strax context used for XENON1T re-analysis with
    the latest strax version
    """
    return straxen.XENONContext(
        storage=straxen.RunDB(),
        register=straxen.plugins.pax_interface.RecordsFromPax,
        register_all=straxen.plugins.plugins)
