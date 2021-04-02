from immutabledict import immutabledict
import strax
import straxen


common_opts = dict(
    register_all=[
        straxen.event_processing,
        straxen.double_scatter],
    # Register all peak/pulse processing by hand as 1T does not need to have
    # the high-energy plugins.
    register=[
        straxen.PulseProcessing,
        straxen.Peaklets,
        straxen.PeakletClassification,
        straxen.MergedS2s,
        straxen.Peaks,
        straxen.PeakBasics,
        straxen.PeakProximity],
    check_available=('raw_records', 'peak_basics'),
    store_run_fields=(
        'name', 'number',
        'start', 'end', 'livetime', 'mode'))

xnt_common_config = dict(
    n_tpc_pmts=straxen.n_tpc_pmts,
    n_top_pmts=straxen.n_top_pmts,
    gain_model=("CMT_model", ("to_pe_model", "ONLINE")),
    gain_model_nv=("CMT_model", ("to_pe_model_nv", "ONLINE")),
    gain_model_mv=("to_pe_constant", "adc_mv"),
    channel_map=immutabledict(
        # (Minimum channel, maximum channel)
        # Channels must be listed in a ascending order!
        tpc=(0, 493),
        he=(500, 752),  # high energy
        aqmon=(790, 807),
        aqmon_nv=(808, 815),  # nveto acquisition monitor
        tpc_blank=(999, 999),
        mv=(1000, 1083),
        aux_mv=(1084, 1087),  # Aux mv channel 2 empty  1 pulser  and 1 GPS
        mv_blank=(1999, 1999),
        nveto=(2000, 2119),
        nveto_blank=(2999, 2999)),
    # Clustering/classification parameters
    s1_max_rise_time=100,
    s2_xy_correction_map=("CMT_model", ('s2_xy_map', "ONLINE"), True),
    fdc_map=("CMT_model", ('fdc_map', "ONLINE"), True),
)

# Plugins in these files have nT plugins, E.g. in pulse&peak(let)
# processing there are plugins for High Energy plugins. Therefore do not
# st.register_all in 1T contexts.
xnt_common_opts = common_opts.copy()
xnt_common_opts['register'] = common_opts['register'] + [
    straxen.PeakPositionsCNN,
    straxen.PeakPositionsMLP,
    straxen.PeakPositionsGCN,
    straxen.PeakPositionsNT,
    straxen.PeakBasicsHighEnergy,
    straxen.PeaksHighEnergy,
    straxen.PeakletsHighEnergy,
    straxen.PeakletClassificationHighEnergy,
    straxen.MergedS2sHighEnergy,
]
xnt_common_opts['register_all'] = common_opts['register_all'] + [
    straxen.nveto_recorder,
    straxen.veto_pulse_processing,
    straxen.veto_hitlets,
    straxen.veto_events,
    straxen.acqmon_processing,
    straxen.pulse_processing,
    straxen.peaklet_processing,
    straxen.online_monitor,
]

##
# XENONnT
##


def xenonnt_online(output_folder='./strax_data',
                   we_are_the_daq=False,
                   _minimum_run_number=7157,
                   _database_init=True,
                   _forbid_creation_of=None,
                   _rucio_path='/dali/lgrandi/rucio/',
                   _raw_path='/dali/lgrandi/xenonnt/raw',
                   _processed_path='/dali/lgrandi/xenonnt/processed',
                   _add_online_monitor_frontend=False,
                   _context_config_overwrite=None,
                   **kwargs):
    """
    XENONnT online processing and analysis

    :param output_folder: str, Path of the strax.DataDirectory where new
        data can be stored
    :param we_are_the_daq: bool, if we have admin access to upload data
    :param _minimum_run_number: int, lowest number to consider
    :param _database_init: bool, start the database (for testing)
    :param _forbid_creation_of: str/tuple, of datatypes to prevent form
        being written (raw_records* is always forbidden).
    :param _rucio_path: str, path of rucio
    :param _raw_path: str, common path of the raw-data
    :param _processed_path: str. common path of output data
    :param _context_config_overwrite: dict, overwrite config
    :param _add_online_monitor_frontend: bool, should we add the online
        monitor storage frontend.
    :param kwargs: dict, context options
    :return: strax.Context
    """
    context_options = {
        **straxen.contexts.xnt_common_opts,
        **kwargs}

    st = strax.Context(
        config=straxen.contexts.xnt_common_config,
        **context_options)
    st.register([straxen.DAQReader, straxen.LEDCalibration])

    st.storage = [
        straxen.RunDB(
            readonly=not we_are_the_daq,
            minimum_run_number=_minimum_run_number,
            runid_field='number',
            new_data_path=output_folder,
            rucio_path=_rucio_path,
        )] if _database_init else []
    if not we_are_the_daq:
        st.storage += [
            strax.DataDirectory(
                _raw_path,
                readonly=True,
                take_only=straxen.DAQReader.provides),
            strax.DataDirectory(
                _processed_path,
                readonly=True,
            )]
        if output_folder:
            st.storage.append(
                strax.DataDirectory(output_folder))

        st.context_config['forbid_creation_of'] = straxen.daqreader.DAQReader.provides
        if _forbid_creation_of is not None:
            st.context_config['forbid_creation_of'] += strax.to_str_tuple(_forbid_creation_of)
    # Only the online monitor backend for the DAQ
    if _database_init and (_add_online_monitor_frontend or we_are_the_daq):
        st.storage += [straxen.OnlineMonitor(
            readonly=not we_are_the_daq,
            take_only=('veto_intervals',
                       'online_peak_monitor',
                       'event_basics',))]

    # Remap the data if it is before channel swap (because of wrongly cabled
    # signal cable connectors) These are runs older than run 8797. Runs
    # newer than 8796 are not affected. See:
    # https://github.com/XENONnT/straxen/pull/166 and
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:daq:sector_swap
    st.set_context_config({'apply_data_function': (straxen.common.remap_old,)})
    if _context_config_overwrite is not None:
        st.set_context_config(_context_config_overwrite)
    return st


def xenonnt_led(**kwargs):
    st = xenonnt_online(**kwargs)
    st.context_config['check_available'] = ('raw_records', 'led_calibration')
    # Return a new context with only raw_records and led_calibration registered
    st = st.new_context(
        replace=True,
        config=st.config,
        storage=st.storage,
        **st.context_config)
    st.register([straxen.DAQReader, straxen.LEDCalibration])
    return st


# This gain model is the average to_pe. For something more fancy use the CMT
def xenonnt_simulation(output_folder='./strax_data'):
    import wfsim
    xnt_common_config['gain_model'] = ('to_pe_constant', 0.01)
    st = strax.Context(
        storage=strax.DataDirectory(output_folder),
        config=dict(detector='XENONnT',
                    fax_config='fax_config_nt_design.json',
                    check_raw_record_overlaps=True,
                    **straxen.contexts.xnt_common_config,
                    ),
        **straxen.contexts.xnt_common_opts)
    st.register(wfsim.RawRecordsFromFaxNT)
    return st


def xenonnt_temporary_five_pmts(**kwargs):
    """Temporary context for selected PMTs"""
    # Start from the online context
    st_online = xenonnt_online(**kwargs)

    temporary_five_pmts_config = {
        'gain_model': ('CMT_model', ("to_pe_model", "xenonnt_temporary_five_pmts")),
        'peak_min_pmts': 2,
        'peaklet_gap_threshold': 300,
    }
    # If there are any config overwrites in the kwargs, us those,
    # otherwise use the config as in the dict above.

    for k in list(temporary_five_pmts_config.keys()):
        if k in kwargs:
            temporary_five_pmts_config[k] = kwargs[k]

    # Copy the online context and change the configuration here
    st = st_online.new_context()
    st.set_config(temporary_five_pmts_config)

    return st


def xenonnt_initial_commissioning(*args, **kwargs):
    raise ValueError(
        'Use xenonnt_online. See' 
        'https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:commissioning:straxen_contexts#update_09_nov_20')  # noqa

##
# XENON1T
##


x1t_context_config = {
    **common_opts,
    **dict(
        check_available=('raw_records', 'records', 'peaklets',
                         'events', 'event_info'),
        free_options=('channel_map',),
        store_run_fields=tuple(
            [x for x in common_opts['store_run_fields'] if x != 'mode']
            + ['trigger.events_built', 'reader.ini.name']))}
x1t_context_config.update(
    dict(register=common_opts['register'] +
                  [straxen.PeakPositions1T,
                   straxen.RecordsFromPax,
         ]))

x1t_common_config = dict(
    check_raw_record_overlaps=False,
    allow_sloppy_chunking=True,
    n_tpc_pmts=248,
    n_top_pmts=127,
    channel_map=immutabledict(
        # (Minimum channel, maximum channel)
        tpc=(0, 247),
        diagnostic=(248, 253),
        aqmon=(254, 999)),
    hev_gain_model=('to_pe_per_run',
                    'to_pe.npy'),
    gain_model=('to_pe_per_run',
                'to_pe.npy'),
    pmt_pulse_filter=(
        0.012, -0.119,
        2.435, -1.271, 0.357, -0.174, -0., -0.036,
        -0.028, -0.019, -0.025, -0.013, -0.03, -0.039,
        -0.005, -0.019, -0.012, -0.015, -0.029, 0.024,
        -0.007, 0.007, -0.001, 0.005, -0.002, 0.004, -0.002),
    tail_veto_threshold=int(1e5),
    # Smaller right extension since we applied the filter
    peak_right_extension=30,
    peak_min_pmts=2,
    save_outside_hits=(3, 3),
    hit_min_amplitude='XENON1T_SR1',
    peak_split_gof_threshold=(
        None,  # Reserved
        ((0.5, 1), (3.5, 0.25)),
        ((2, 1), (4.5, 0.4))),
    left_event_extension=int(1e6),
    right_event_extension=int(1e6),
    elife_conf=straxen.aux_repo + '3548132b55f81a43654dba5141366041e1daaf01/strax_files/elife.npy',
)



def demo():
    """Return strax context used in the straxen demo notebook"""
    straxen.download_test_data()

    st = strax.Context(
        storage=[strax.DataDirectory('./strax_data'),
                 strax.DataDirectory('./strax_test_data',
                                     deep_scan=True,
                                     provide_run_metadata=True,
                                     readonly=True)],
        forbid_creation_of=straxen.daqreader.DAQReader.provides,
        config=dict(**x1t_common_config),
        **x1t_context_config)

    # Use configs that are always available
    st.set_config(dict(
        hev_gain_model=('to_pe_per_run', straxen.aux_repo +
                        '3548132b55f81a43654dba5141366041e1daaf01/strax_files/to_pe.npy'),
        gain_model=('to_pe_per_run', straxen.aux_repo +
                    '3548132b55f81a43654dba5141366041e1daaf01/strax_files/to_pe.npy'),
    ))
    return st


def fake_daq():
    """Context for processing fake DAQ data in the current directory"""
    st = strax.Context(
        storage=[strax.DataDirectory('./strax_data'),
                 # Fake DAQ puts run doc JSON in same folder:
                 strax.DataDirectory('./from_fake_daq',
                                     provide_run_metadata=True,
                                     readonly=True)],
        config=dict(daq_input_dir='./from_fake_daq',
                    daq_chunk_duration=int(2e9),
                    daq_compressor='lz4',
                    n_readout_threads=8,
                    daq_overlap_chunk_duration=int(2e8),
                    **x1t_common_config),
        **x1t_context_config)
    st.register(straxen.Fake1TDAQReader)
    return st


def xenon1t_dali(output_folder='./strax_data', build_lowlevel=False, **kwargs):
    context_options = {
        **x1t_context_config,
        **kwargs}

    st = strax.Context(
        storage=[
            strax.DataDirectory(
                '/dali/lgrandi/xenon1t/strax_converted/raw',
                take_only='raw_records',
                provide_run_metadata=True,
                readonly=True),
            strax.DataDirectory(
                '/dali/lgrandi/xenon1t/strax_converted/processed',
                readonly=True),
            strax.DataDirectory(output_folder)],
        config=dict(**x1t_common_config),
        # When asking for runs that don't exist, throw an error rather than
        # starting the pax converter
        forbid_creation_of=(
            straxen.daqreader.DAQReader.provides if build_lowlevel
            else straxen.daqreader.DAQReader.provides + ('records', 'peaklets')),
        **context_options)
    return st


def xenon1t_led(**kwargs):
    st = xenon1t_dali(**kwargs)
    st.context_config['check_available'] = ('raw_records', 'led_calibration')
    # Return a new context with only raw_records and led_calibration registered
    st = st.new_context(
        replace=True,
        config=st.config,
        storage=st.storage,
        **st.context_config)
    st.register([straxen.RecordsFromPax, straxen.LEDCalibration])
    return st


def xenon1t_simulation(output_folder='./strax_data'):
    import wfsim
    st = strax.Context(
        storage=strax.DataDirectory(output_folder),
        config=dict(
            fax_config='fax_config_1t.json',
            detector='XENON1T',
            **straxen.contexts.x1t_common_config),
        **straxen.contexts.common_opts)
    st.register(wfsim.RawRecordsFromFax1T)
    return st
