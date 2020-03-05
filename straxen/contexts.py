import strax
import straxen


common_opts = dict(
    register_all=[
        straxen.daqreader,
        straxen.pulse_processing,
        straxen.peaklet_processing,
        straxen.peak_processing,
        straxen.event_processing,
        straxen.cuts],
    store_run_fields=(
        'name', 'number',
        'reader.ini.name', 'tags.name',
        'start', 'end', 'livetime',
        'trigger.events_built'),
    check_available=('raw_records', 'records', 'peaklets',
                     'events', 'event_info'))


def demo():
    """Return strax context used in the straxen demo notebook"""
    straxen.download_test_data()
    return strax.Context(
            storage=[strax.DataDirectory('./strax_data'),
                     strax.DataDirectory('./strax_test_data')],
            register=straxen.RecordsFromPax,
            forbid_creation_of=('raw_records',),
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
        register=straxen.plugins.pax_interface.RecordsFromPax,
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=('raw_records',),
        **common_opts)


def fake_daq():
    """Context for processing fake DAQ data in the current directory"""
    return strax.Context(
        storage=[strax.DataDirectory('./strax_data',
                                     provide_run_metadata=False),
                 # Fake DAQ puts run doc JSON in same folder:
                 strax.DataDirectory('./from_fake_daq',
                                     readonly=True)],
        config=dict(input_dir='./from_fake_daq'),
        **common_opts)


nveto_common_opts = dict(
    register_all=[
        straxen.nveto_daqreader,
        straxen.nveto_recorder,
        straxen.nveto_pulse_processing],
    store_run_fields=(
        'name', 'number',
        'reader.ini.name', 'tags.name',
        'start', 'end', 'livetime',
        'trigger.events_built'),
    check_available=('nveto_pre_raw_records',
                     'nveto_raw_records',
                     'nveto_diagnostic_lone_records',
                     'nveto_lone_records_count',
                     'nveto_records',
                     'nveto_pulses',
                     'nveto_pulse_basics',
                     ))


def strax_nveto_hdm_test():
    return strax.Context(
        storage=[
            strax.DataDirectory(
                '/dali/lgrandi/hiraide/data/nveto/DaqTestHdM',
                take_only='nveto_pre_raw_records',
                deep_scan=False,
                readonly=True),
            strax.DataDirectory(
                '/dali/lgrandi/wenz/strax_data/HdMtest',
                provide_run_metadata=False)],
        **nveto_common_opts)