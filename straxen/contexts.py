import warnings

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
                     strax.DataDirectory('./strax_test_data', readonly=True)],
            register=straxen.RecordsFromPax,
            forbid_creation_of=('raw_records',),
            config=dict(check_raw_record_overlaps=False),
            **common_opts)


def strax_workshop_dali():
    warnings.warn(
        "The strax_workshop_dali context is deprecated and will "
        "be removed in April 2020. Please use "
        "straxen.contexts.xenon1t_dali() instead.",
        DeprecationWarning)
    return xenon1t_dali()


def xenon1t_dali():
    return strax.Context(
        storage=[
            strax.DataDirectory(
                '/dali/lgrandi/xenon1t/strax_converted/raw',
                take_only='raw_records',
                provide_run_metadata=True,
                deep_scan=False,
                readonly=True),
            strax.DataDirectory(
                '/dali/lgrandi/xenon1t/strax_converted/processed',
                readonly=True,
                provide_run_metadata=False),
            strax.DataDirectory('./strax_data',
                                provide_run_metadata=False)],
        register=straxen.plugins.pax_interface.RecordsFromPax,
        config=dict(check_raw_record_overlaps=False),
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
        config=dict(daq_input_dir='./from_fake_daq',
                    daq_chunk_duration=int(2e9),
                    n_readout_threads=8,
                    daq_overlap_chunk_duration=int(2e8)),
        **common_opts)
