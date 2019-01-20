import strax
import straxen


def demo():
    """Return strax context used in the straxen demo notebook"""
    return straxen.XENONContext(
            storage=[strax.DataDirectory('./strax_data'),
                     strax.DataDirectory(straxen.straxen_dir + '/data',
                                         readonly=True)],
            register=straxen.plugins.pax_interface.RecordsFromPax,
            register_all=straxen.plugins.plugins,
            forbid_creation_of=('raw_records',),
    )


def xenon1t_analysis(local_only=False):
    """Return strax context used for XENON1T re-analysis with
    the latest strax version
    """
    return straxen.XENONContext(
        storage=[
            straxen.RunDB(local_only=local_only),
            strax.DataDirectory('./strax_data'),
        ],
        register=straxen.plugins.pax_interface.RecordsFromPax,
        register_all=straxen.plugins.plugins,
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=('raw_records',),
    )


def nt_daq_test_analysis(local_data_dir='./strax_data'):
    """Return strax test for analysis of the nT DAQ test data"""
    return straxen.XENONContext(
        storage = [
            straxen.RunDB(
                mongo_url='mongodb://{username}:{password}@gw:27019/xenonnt',
                mongo_collname='run',
                runid_field='number',
                mongo_dbname='xenonnt'),
            # TODO: can we avoid having to declare this as another frontend?
            strax.DataDirectory('./strax_data_jelle')],
        register_all=straxen.plugins.plugins)
