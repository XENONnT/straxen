import strax
import straxen as sx


common_opts = dict(
    register_all=[sx.daqreader, sx.pulse_processing,
                  sx.peak_processing, sx.event_processing, sx.cuts],
    store_run_fields=(
        'name', 'number', 'reader.ini.name',
        'tags.name',
        'start', 'end', 'trigger.events_built',
        'tags.name'),
    check_available=('raw_records', 'records', 'peaks',
                     'events', 'event_info'))


def demo():
    """Return strax context used in the straxen demo notebook"""
    sx.download_test_data()
    return strax.Context(
            storage=[strax.DataDirectory('./strax_data'),
                     strax.DataDirectory('./strax_test_data')],
            register=sx.RecordsFromPax,
            forbid_creation_of=('raw_records',),
            **common_opts)


def xenon1t_analysis(local_only=False):
    """Return strax context used for XENON1T re-analysis with
    the latest strax version
    """
    return strax.Context(
        storage=[
            sx.RunDB(local_only=local_only),
            strax.DataDirectory(
                '/dali/lgrandi/aalbers/strax_data_raw',
                take_only='raw_records',
                deep_scan=False,
                readonly=True),
            strax.DataDirectory('./strax_data'),
        ],
        register=sx.RecordsFromPax,
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=('raw_records',),
        **common_opts)


def nt_daq_test_analysis():
    """Return strax test for analysis of the nT DAQ test data"""
    return strax.Context(
        storage = [
            sx.RunDB(
                mongo_url='mongodb://{username}:{password}@gw:27019/xenonnt',
                mongo_collname='run',
                runid_field='number',
                mongo_dbname='xenonnt'),
            # TODO: can we avoid having to declare this as another frontend?
            strax.DataDirectory('./strax_data_jelle')],
        **common_opts)


def strax_workshop_dali():
    return strax.Context(
        storage=[
            strax.DataDirectory(
                '/dali/lgrandi/aalbers/strax_data_raw',
                take_only='raw_records',
                deep_scan=False,
                readonly=True),
            strax.DataDirectory(
                '/dali/lgrandi/aalbers/strax_data',
                readonly=True,
                provide_run_metadata=False),
            strax.DataDirectory('./strax_data',
                                provide_run_metadata=False)],
        register=sx.plugins.pax_interface.RecordsFromPax,
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=('raw_records',),
        **common_opts)
