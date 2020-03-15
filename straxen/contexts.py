import warnings

from frozendict import frozendict
import strax
import straxen


common_opts = dict(
    register_all=[
        straxen.pulse_processing,
        straxen.peaklet_processing,
        straxen.peak_processing,
        straxen.event_processing],
    store_run_fields=(
        'name', 'number',
        'reader.ini.name', 'tags.name',
        'start', 'end', 'livetime',
        'trigger.events_built'),
    check_available=('raw_records', 'records', 'peaklets',
                     'events', 'event_info'))

x1t_common_config = dict(
    check_raw_record_overlaps=False,
    n_tpc_pmts=248,
    channel_map=frozendict(
        # (Minimum channel, maximum channel)
        tpc=(0, 247),
        diagnostic=(248, 253),
        aqmon=(254, 999)))

xnt_common_config = dict(
    n_tpc_pmts=493,
    channel_map=frozendict(
         # (Minimum channel, maximum channel)
         tpc=(0, 493),
         lowgain=(500, 752),
         aqmon=(799, 807),
         tpc_blank=(999, 999),
         mv=(1000, 1083),
         mv_blank=(1999, 1999),
    ))


##
# XENONnT
##

def nt_simulation():
    import wfsim
    return strax.Context(
        storage='./strax_data',
        register=wfsim.RawRecordsFromFax,
        config=dict(
            nchunk=1,
            event_rate=1,
            chunk_size=10,
            detector='XENONnT',
            fax_config='https://raw.githubusercontent.com/XENONnT/'
                       'strax_auxiliary_files/master/fax_files/fax_config_nt.json',
            **x1t_common_config),
        **straxen.contexts.common_opts)


##
# XENON1T
##

def demo():
    """Return strax context used in the straxen demo notebook"""
    straxen.download_test_data()
    return strax.Context(
            storage=[strax.DataDirectory('./strax_data'),
                     strax.DataDirectory('./strax_test_data', readonly=True)],
            register=straxen.RecordsFromPax,
            forbid_creation_of=('raw_records',),
            config=dict(**x1t_common_config),
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
                    daq_compressor='lz4',
                    n_readout_threads=8,
                    daq_overlap_chunk_duration=int(2e8),
                    **x1t_common_config),
        register=straxen.Fake1TDAQReader,
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
        config=dict(**x1t_common_config),
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=('raw_records',),
        **common_opts)
