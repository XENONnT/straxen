from immutabledict import immutabledict
import strax
import straxen
from copy import deepcopy

common_opts = dict(
    register_all=[
        straxen.event_processing,
        straxen.double_scatter,
    ],
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
    gain_model=("to_pe_model", "ONLINE", True),
    gain_model_nv=("to_pe_model_nv", "ONLINE", True),
    gain_model_mv=("to_pe_model_mv", "ONLINE", True),
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
    # Event level parameters
    s2_xy_correction_map=('s2_xy_map', "ONLINE", True),
    fdc_map=('fdc_map', "ONLINE", True),
    s1_xyz_correction_map=("s1_xyz_map", "ONLINE", True),
    g1=0.1426,
    g2=11.55,
)
# these are placeholders to avoid calling cmt with non integer run_ids. Better solution pending.
# s1,s2 and fd corrections are still problematic
xnt_simulation_config = deepcopy(xnt_common_config)
xnt_simulation_config.update(gain_model=("to_pe_placeholder", True),
                             gain_model_nv=("adc_nv", True),
                             gain_model_mv=("adc_mv", True),
                             elife_conf=('elife_constant', 1e6),
                             )

# Plugins in these files have nT plugins, E.g. in pulse&peak(let)
# processing there are plugins for High Energy plugins. Therefore do not
# st.register_all in 1T contexts.
xnt_common_opts = common_opts.copy()
xnt_common_opts.update({
    'register': common_opts['register'] + [straxen.PeakPositionsCNN,
                                           straxen.PeakPositionsMLP,
                                           straxen.PeakPositionsGCN,
                                           straxen.PeakPositionsNT,
                                           straxen.PeakBasicsHighEnergy,
                                           straxen.PeaksHighEnergy,
                                           straxen.PeakletsHighEnergy,
                                           straxen.PeakletClassificationHighEnergy,
                                           straxen.MergedS2sHighEnergy,
                                           straxen.EventInfo,
                                          ],
    'register_all': common_opts['register_all'] + [straxen.veto_veto_regions,
                                                   straxen.nveto_recorder,
                                                   straxen.veto_pulse_processing,
                                                   straxen.veto_hitlets,
                                                   straxen.veto_events,
                                                   straxen.acqmon_processing,
                                                   straxen.pulse_processing,
                                                   straxen.peaklet_processing,
                                                   straxen.online_monitor,
                                                   straxen.event_area_per_channel,
                                                   straxen.event_patternfit,
                                                   ],
    'use_per_run_defaults': False,
})

##
# XENONnT
##


def xenonnt(cmt_version='global_ONLINE', **kwargs):
    """XENONnT context"""
    st = straxen.contexts.xenonnt_online(**kwargs)
    st.apply_cmt_version(cmt_version)
    return st


def xenonnt_online(output_folder='./strax_data',
                   use_rucio=False,
                   we_are_the_daq=False,
                   _minimum_run_number=7157,
                   _maximum_run_number=None,
                   _database_init=True,
                   _forbid_creation_of=None,
                   _rucio_path='/dali/lgrandi/rucio/',
                   _include_rucio_remote=False,
                   _raw_path='/dali/lgrandi/xenonnt/raw',
                   _processed_path='/dali/lgrandi/xenonnt/processed',
                   _add_online_monitor_frontend=False,
                   _context_config_overwrite=None,
                   **kwargs):
    """
    XENONnT online processing and analysis

    :param output_folder: str, Path of the strax.DataDirectory where new
        data can be stored
    :param use_rucio: bool, whether or not to use the rucio frontend
    :param we_are_the_daq: bool, if we have admin access to upload data
    :param _minimum_run_number: int, lowest number to consider
    :param _maximum_run_number: Highest number to consider. When None
        (the default) consider all runs that are higher than the
        minimum_run_number.
    :param _database_init: bool, start the database (for testing)
    :param _forbid_creation_of: str/tuple, of datatypes to prevent form
        being written (raw_records* is always forbidden).
    :param _include_rucio_remote: allow remote downloads in the context
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
            maximum_run_number=_maximum_run_number,
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
                strax.DataDirectory(output_folder,
                                    provide_run_metadata=True,
                                   ))

        st.context_config['forbid_creation_of'] = straxen.daqreader.DAQReader.provides
        if _forbid_creation_of is not None:
            st.context_config['forbid_creation_of'] += strax.to_str_tuple(_forbid_creation_of)

    # if we said so, add the rucio frontend to storage
    if use_rucio:
        st.storage.append(straxen.rucio.RucioFrontend(
            include_remote=straxen.RUCIO_AVAILABLE,
            staging_dir=output_folder,
        ))

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
    st.set_context_config({'apply_data_function': (straxen.remap_old,
                                                   straxen.check_loading_allowed,
                                                   )})
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


def xenonnt_simulation(output_folder='./strax_data',
                       cmt_run_id='020280',
                       cmt_version='v3',
                       fax_config='fax_config_nt_design.json',
                       cmt_option_overwrite=immutabledict(),
                       overwrite_from_fax_file=False,
                       _cmt_run_id_proc_only=None,
                       _forbid_creation_of=None,
                       _config_overlap=immutabledict(drift_time_gate='electron_drift_time_gate',
                                                     drift_velocity_liquid='electron_drift_velocity',
                                                     electron_lifetime_liquid='elife_conf'),
                       **kwargs):
    """
    Context with CMT options updated with cmt_run_id.
    CMT options can also be overwritten via fax config file.
    :param output_folder: Output folder for strax data.
    :param cmt_run_id: Run id which should be used to load corrections.
    :param cmt_version: Global version for corrections to be loaded.
    :param fax_config: Fax config file to use.
    :param cmt_option_overwrite: Dictionary to overwrite CMT settings. Keys
        must be valid CMT option keys.
    :param overwrite_from_fax_file: If true overwrites CMT options with
        constants from fax file.
    :param _cmt_run_id_proc_only: Run id just for > raw_records processing if diverge from
    cmt_run_id.
    :param _forbid_creation_of: str/tuple, of datatypes to prevent form
        being written (e.g. 'raw_records' for read only simulation context).
    :param _config_overlap: Dictionary of options to overwrite. Keys
        must be simulation config keys, values must be valid CMT option keys.
    :param kwargs: Additional kwargs taken by strax.Context.
    :return: strax.Context instance
    """
    import wfsim
    st = strax.Context(
        storage=strax.DataDirectory(output_folder),
        config=dict(detector='XENONnT',
                    fax_config=fax_config,
                    check_raw_record_overlaps=True,
                    **straxen.contexts.xnt_common_config,),
        **straxen.contexts.xnt_common_opts, **kwargs)
    st.register(wfsim.RawRecordsFromFaxNT)

    if _forbid_creation_of is not None:
        st.context_config['forbid_creation_of'] += strax.to_str_tuple(_forbid_creation_of)

    st.apply_cmt_version(f'global_{cmt_version}')
    if not cmt_run_id:
        raise ValueError('You have to specify a run_id which should be used to initialize '
                         'the corrections.')
    if _cmt_run_id_proc_only is None:
        _cmt_run_id_proc_only = cmt_run_id

    # Setup processing
    # Replace default cmt options with cmt_run_id tag + cmt run id
    cmt_options = straxen.get_corrections.get_cmt_options(st)
    for option in cmt_options:
        st.config[option] = tuple(['cmt_run_id', _cmt_run_id_proc_only, *cmt_options[option]])

    # Take fax config and put into context option
    if overwrite_from_fax_file:
        fax_config = straxen.get_resource(fax_config, fmt='json')

        for fax_field, cmt_field in _config_overlap.items():
            st.config[cmt_field] = tuple([cmt_options[cmt_field][0] + '_constant',
                                          fax_config[fax_field]])

    # Setup simulation
    # Pass the CMT options to simulation (override all other config input methods)
    else:
        fax_config_override_from_cmt = dict()

        for fax_field, cmt_field in _config_overlap.items():
            fax_config_override_from_cmt[fax_field] = tuple(['cmt_run_id', cmt_run_id, *cmt_options[cmt_field]])
        st.set_config({'fax_config_override_from_cmt': fax_config_override_from_cmt})

    # Fix gain model
    st.set_config({'gain_model_mc': tuple(['cmt_run_id', cmt_run_id, *cmt_options['gain_model']])})

    # User customized overwrite
    cmt_options = straxen.get_corrections.get_cmt_options(st)  # This includes gain_model_mc
    for option in cmt_option_overwrite:
        if option not in cmt_options:
            raise ValueError(f'Overwrite option {option} is not using CMT by default '
                             'you should just use set config')
        _name_index = 2 if 'cmt_run_id' in cmt_options[option] else 0
        st.config[option] = (cmt_options[option][_name_index] + '_constant', cmt_option_overwrite[option])

    # Only for simulations
    st.set_config({"event_info_function": "disabled"})

    return st


##
# XENON1T
##


x1t_context_config = {
    **common_opts,
    **dict(
        check_available=('raw_records', 'records', 'peaklets',
                         'events', 'event_info'),
        free_options=('channel_map',),
        use_per_run_defaults=True,
        store_run_fields=tuple(
            [x for x in common_opts['store_run_fields'] if x != 'mode']
            + ['trigger.events_built', 'reader.ini.name']))}
x1t_context_config.update(
    dict(register=common_opts['register'] +
                  [straxen.PeakPositions1T,
                   straxen.RecordsFromPax,
                   straxen.EventInfo1T,
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
    # Records
    hev_gain_model=('to_pe_model', "v1", False),
    pmt_pulse_filter=(
        0.012, -0.119,
        2.435, -1.271, 0.357, -0.174, -0., -0.036,
        -0.028, -0.019, -0.025, -0.013, -0.03, -0.039,
        -0.005, -0.019, -0.012, -0.015, -0.029, 0.024,
        -0.007, 0.007, -0.001, 0.005, -0.002, 0.004, -0.002),
    hit_min_amplitude='XENON1T_SR1',
    tail_veto_threshold=int(1e5),
    save_outside_hits=(3, 3),
    # Peaklets
    peaklet_gap_threshold=350,
    gain_model=('to_pe_model', "v1", False),
    peak_split_gof_threshold=(
        None,  # Reserved
        ((0.5, 1), (3.5, 0.25)),
        ((2, 1), (4.5, 0.4))),
    peak_min_pmts=2,
    # MergedS2s
    s2_merge_gap_thresholds=((1.7, 5.0e3), (4.0, 500.), (5.0, 0.)),
    # Peaks
    # Smaller right extension since we applied the filter
    peak_right_extension=30,
    s1_max_rise_time=60,
    s1_max_rise_time_post100=150,
    s1_min_coincidence=3,
    # Events*
    left_event_extension=int(0.3e6),
    right_event_extension=int(1e6),
    elife_conf=('elife_xenon1t', 'v1', False),
    electron_drift_velocity=("electron_drift_velocity_constant", 1.3325e-4),
    max_drift_length=96.9,
    electron_drift_time_gate=("electron_drift_time_gate_constant", 1700),
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
        hev_gain_model=('1T_to_pe_placeholder', False),
        gain_model=('1T_to_pe_placeholder', False),
        elife_conf=('elife_constant', 1e6),
        electron_drift_velocity=("electron_drift_velocity_constant", 1.3325e-4),
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
            **x1t_common_config),
        **x1t_context_config)
    st.register(wfsim.RawRecordsFromFax1T)
    return st
