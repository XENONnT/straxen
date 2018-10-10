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


def xenon1t_analysis(local_only=False):
    """Return strax context used for XENON1T re-analysis with
    the latest strax version
    """
    return straxen.XENONContext(
        storage=straxen.RunDB(local_only=local_only),
        register=straxen.plugins.pax_interface.RecordsFromPax,
        register_all=straxen.plugins.plugins,
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=('raw_records',),
    )