from immutabledict import immutabledict
import strax
import straxen
from copy import deepcopy
from straxen import HAVE_ADMIX
import os
import warnings
import typing as ty
from pandas.util._decorators import deprecate_kwarg

from straxen.common import pax_file

common_opts = dict(
    register_all=[
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
        straxen.PeakProximity,
        straxen.Events,
        straxen.EventBasics,
        straxen.EventPositions,
        straxen.CorrectedAreas,
        straxen.EnergyEstimates,
    ],
    check_available=('raw_records', 'peak_basics'),
    store_run_fields=(
        'name', 'number',
        'start', 'end', 'livetime', 'mode', 'source'))

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
    fdc_map=('fdc_map', "ONLINE", True),
)
# these are placeholders to avoid calling cmt with non integer run_ids. Better solution pending.
# s1,s2 and fd corrections are still problematic
xnt_simulation_config = deepcopy(xnt_common_config)
xnt_simulation_config.update(gain_model=("to_pe_placeholder", True),
                             gain_model_nv=("adc_nv", True),
                             gain_model_mv=("adc_mv", True),
                             elife=1e6,
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
                                           straxen.S2ReconPosDiff,
                                           straxen.PeakBasicsHighEnergy,
                                           straxen.PeaksHighEnergy,
                                           straxen.PeakletsHighEnergy,
                                           straxen.PeakletClassificationHighEnergy,
                                           straxen.MergedS2sHighEnergy,
                                           straxen.EventInfo,
                                           straxen.PeakShadow,
                                           straxen.EventShadow,
                                           ],
    'register_all': common_opts['register_all'] + [straxen.nveto_recorder,
                                                   straxen.veto_pulse_processing,
                                                   straxen.veto_hitlets,
                                                   straxen.veto_events,
                                                   straxen.acqmon_processing,
                                                   straxen.pulse_processing,
                                                   straxen.peaklet_processing,
                                                   straxen.online_monitor,
                                                   straxen.event_area_per_channel,
                                                   straxen.event_patternfit,
                                                   straxen.event_processing,
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


@deprecate_kwarg('_minimum_run_number', 'minimum_run_number')
@deprecate_kwarg('_maximum_run_number', 'maximum_run_number')
@deprecate_kwarg('_include_rucio_remote', 'include_rucio_remote')
@deprecate_kwarg('_add_online_monitor_frontend', 'include_online_monitor')
def xenonnt_online(output_folder: str = './strax_data',
                   we_are_the_daq: bool = False,
                   minimum_run_number: int = 7157,
                   maximum_run_number: ty.Optional[int] = None,

                   # Frontends
                   include_rucio_remote: bool = False,
                   include_online_monitor: bool = False,
                   include_rucio_local: bool = False,

                   # Frontend options
                   download_heavy: bool = False,
                   _rucio_path: str = '/dali/lgrandi/rucio/',
                   _rucio_local_path: ty.Optional[str] = None,
                   _raw_path: ty.Optional[str] = '/dali/lgrandi/xenonnt/raw',
                   _processed_path: ty.Optional[str] = '/dali/lgrandi/xenonnt/processed',

                   # Testing options
                   _context_config_overwrite: ty.Optional[dict] = None,
                   _database_init: bool = True,
                   _forbid_creation_of: ty.Optional[dict] = None,
                   **kwargs):
    """
    XENONnT online processing and analysis

    :param output_folder: str, Path of the strax.DataDirectory where new
        data can be stored
    :param we_are_the_daq: bool, if we have admin access to upload data
    :param minimum_run_number: int, lowest number to consider
    :param maximum_run_number: Highest number to consider. When None
        (the default) consider all runs that are higher than the
        minimum_run_number.
    :param include_rucio_remote: add the rucio remote frontend to the
        context
    :param include_online_monitor: add the online monitor storage frontend.
    :param include_rucio_local: add the rucio local storage frontend.
        This is only needed if one wants to do a fuzzy search in the
        data the runs database is out of sync with rucio
    :param download_heavy: bool, whether or not to allow downloads of
        heavy data (raw_records*, less the aqmon)

    :param _rucio_path: str, path of rucio
    :param _rucio_local_path: str, path of local RSE of rucio. Only use
        for testing!
    :param _raw_path: str, common path of the raw-data
    :param _processed_path: str. common path of output data
    :param _context_config_overwrite: dict, overwrite config
    :param _database_init: bool, start the database (for testing)
    :param _forbid_creation_of: str/tuple, of datatypes to prevent form
        being written (raw_records* is always forbidden).
    :param kwargs: dict, context options
    :return: strax.Context
    """
    context_options = {
        **straxen.contexts.xnt_common_opts,
        **kwargs}

    st = strax.Context(
        config=straxen.contexts.xnt_common_config,
        **context_options)
    st.register([straxen.DAQReader,
                 straxen.LEDCalibration,
                 straxen.LEDAfterpulseProcessing])

    st.storage = [
        straxen.RunDB(
            readonly=not we_are_the_daq,
            minimum_run_number=minimum_run_number,
            maximum_run_number=maximum_run_number,
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
            st.storage += [strax.DataDirectory(output_folder,
                                               provide_run_metadata=True,
                                               )]
        st.context_config['forbid_creation_of'] = straxen.daqreader.DAQReader.provides
        if _forbid_creation_of is not None:
            st.context_config['forbid_creation_of'] += strax.to_str_tuple(_forbid_creation_of)

    # Add the rucio frontend if we are able to
    if include_rucio_remote and HAVE_ADMIX:
        rucio_frontend = straxen.RucioRemoteFrontend(
            staging_dir=os.path.join(output_folder, 'rucio'),
            download_heavy=download_heavy,
        )
        st.storage += [rucio_frontend]

    if include_rucio_local:
        rucio_local_frontend = straxen.RucioLocalFrontend(path=_rucio_local_path)
        st.storage += [rucio_local_frontend]

    # Only the online monitor backend for the DAQ
    if _database_init and (include_online_monitor or we_are_the_daq):
        st.storage += [straxen.OnlineMonitor(
            readonly=not we_are_the_daq,
            take_only=('veto_intervals',
                       'online_peak_monitor',
                       'event_basics',
                       'online_monitor_nv',
                       'online_monitor_mv',
                       ))]

    # Remap the data if it is before channel swap (because of wrongly cabled
    # signal cable connectors) These are runs older than run 8797. Runs
    # newer than 8796 are not affected. See:
    # https://github.com/XENONnT/straxen/pull/166 and
    # https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:daq:sector_swap
    st.set_context_config({'apply_data_function': (straxen.remap_old,
                                                   straxen.check_loading_allowed,
                                                   )})
    if _context_config_overwrite is not None:
        warnings.warn(f'_context_config_overwrite is deprecated, please pass to context as kwargs',
                      DeprecationWarning)
        st.set_context_config(_context_config_overwrite)

    return st


def _parse_xenonnt_online_kwargs(argument_mapping, **kwargs):
    parsed_kwargs = {}
    for correct_kwarg, old_kwarg in argument_mapping:
        if old_kwarg in kwargs:
            value = kwargs.pop(old_kwarg)
            warnings.warn(f'Use {correct_kwarg} instead of {old_kwarg}',
                          DeprecationWarning)
        else:
            value = kwargs.pop(correct_kwarg)
        parsed_kwargs[correct_kwarg] = value
    list_of_kwargs = [parsed_kwargs[k[0]] for k in argument_mapping]
    return list_of_kwargs, kwargs


def xenonnt_led(**kwargs):
    st = xenonnt_online(**kwargs)
    st.set_context_config(
        {'check_available': ('raw_records', 'led_calibration'),
         'free_options': list(xnt_common_config.keys())
         })
    # Return a new context with only raw_records and led_calibration registered
    st = st.new_context(
        replace=True,
        config=st.config,
        storage=st.storage,
        **st.context_config)
    st.register([straxen.DAQReader,
                 straxen.LEDCalibration,
                 straxen.nVETORecorder,
                 straxen.nVETOPulseProcessing,
                 straxen.nVETOHitlets,
                 straxen.nVetoExtTimings, ])
    st.set_config({"coincidence_level_recorder_nv": 1})
    return st


def xenonnt_simulation(
                output_folder='./strax_data',
                wfsim_registry='RawRecordsFromFaxNT',
                cmt_run_id_sim=None,
                cmt_run_id_proc=None,
                cmt_version='global_ONLINE',
                fax_config='fax_config_nt_design.json',
                overwrite_from_fax_file_sim=False,
                overwrite_from_fax_file_proc=False,
                cmt_option_overwrite_sim=immutabledict(),
                cmt_option_overwrite_proc=immutabledict(),
                _forbid_creation_of=None,
                _config_overlap=immutabledict(
                            drift_time_gate='electron_drift_time_gate',
                            drift_velocity_liquid='electron_drift_velocity',
                            electron_lifetime_liquid='elife',
                ),
                **kwargs):
    """
    The most generic context that allows for setting full divergent
    settings for simulation purposes

    It makes full divergent setup, allowing to set detector simulation
    part (i.e. for wfsim up to truth and  raw_records). Parameters _sim
    refer to detector simulation parameters.

    Arguments having _proc in their name refer to detector parameters that
    are used for processing of simulations as done to the real detector
    data. This means starting from already existing raw_records and finishing
    with higher level data, such as peaks, events etc.

    If only one cmt_run_id is given, the second one will be set automatically,
    resulting in CMT match between simulation and processing. However, detector
    parameters can be still overwritten from fax file or manually using cmt
    config overwrite options.

    CMT options can also be overwritten via fax config file.
    :param output_folder: Output folder for strax data.
    :param wfsim_registry: Name of WFSim plugin used to generate data.
    :param cmt_run_id_sim: Run id for detector parameters from CMT to be used
        for creation of raw_records.
    :param cmt_run_id_proc: Run id for detector parameters from CMT to be used
        for processing from raw_records to higher level data.
    :param cmt_version: Global version for corrections to be loaded.
    :param fax_config: Fax config file to use.
    :param overwrite_from_fax_file_sim: If true sets detector simulation
        parameters for truth/raw_records from from fax_config file istead of CMT
    :param overwrite_from_fax_file_proc:  If true sets detector processing
        parameters after raw_records(peaklets/events/etc) from from fax_config
        file instead of CMT
    :param cmt_option_overwrite_sim: Dictionary to overwrite CMT settings for
        the detector simulation part.
    :param cmt_option_overwrite_proc: Dictionary to overwrite CMT settings for
        the data processing part.
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
    st.register(getattr(wfsim, wfsim_registry))

    # Make sure that the non-simulated raw-record types are not requested
    st.deregister_plugins_with_missing_dependencies()

    if straxen.utilix_is_configured(
            warning_message='Bad context as we cannot set CMT since we '
                            'have no database access'''):
        st.apply_cmt_version(cmt_version)

    if _forbid_creation_of is not None:
        st.context_config['forbid_creation_of'] += strax.to_str_tuple(_forbid_creation_of)

    # doing sanity checks for cmt run ids for simulation and processing
    if (not cmt_run_id_sim ) and (not cmt_run_id_proc ):
        raise RuntimeError("cmt_run_id_sim and cmt_run_id_proc are None. "
                           "You have to specify at least one CMT run id. ")
    if (cmt_run_id_sim and cmt_run_id_proc ) and (cmt_run_id_sim!=cmt_run_id_proc):
        print("INFO : divergent CMT runs for simulation and processing")
        print("    cmt_run_id_sim".ljust(25), cmt_run_id_sim)
        print("    cmt_run_id_proc".ljust(25), cmt_run_id_proc)
    else:
        cmt_id = cmt_run_id_sim  or cmt_run_id_proc
        cmt_run_id_sim  = cmt_id
        cmt_run_id_proc = cmt_id

    # Replace default cmt options with cmt_run_id tag + cmt run id
    cmt_options_full = straxen.get_corrections.get_cmt_options(st)

    # prune to just get the strax options
    cmt_options = {key: val['strax_option']
                   for key, val in cmt_options_full.items()}

    # First, fix gain model for simulation
    st.set_config({'gain_model_mc': 
                        ('cmt_run_id', cmt_run_id_sim, *cmt_options['gain_model'])})
    fax_config_override_from_cmt = dict()
    for fax_field, cmt_field in _config_overlap.items():
        value = cmt_options[cmt_field]

        # URL configs need to be converted to the expected format
        if isinstance(value, str):
            opt_cfg = cmt_options_full[cmt_field]
            version = straxen.URLConfig.kwarg_from_url(value, 'version')
            # We now allow the cmt name to be different from the config name
            # WFSim expects the cmt name
            value = (opt_cfg['correction'], version, True)

        fax_config_override_from_cmt[fax_field] = ('cmt_run_id', cmt_run_id_sim,
                                                   *value)
    st.set_config({'fax_config_override_from_cmt': fax_config_override_from_cmt})

    # and all other parameters for processing
    for option in cmt_options:
        value = cmt_options[option]
        if isinstance(value, str):
            # for URL configs we can just replace the run_id keyword argument
            # This will become the proper way to override the run_id for cmt configs
            st.config[option] = straxen.URLConfig.format_url_kwargs(value, run_id=cmt_run_id_proc)
        else:
            # FIXME: Remove once all cmt configs are URLConfigs
            st.config[option] = ('cmt_run_id', cmt_run_id_proc, *value)

    # Done with "default" usage, now to overwrites from file
    #
    # Take fax config and put into context option
    if overwrite_from_fax_file_proc or overwrite_from_fax_file_sim:
        fax_config = straxen.get_resource(fax_config, fmt='json')
        for fax_field, cmt_field in _config_overlap.items():
            if overwrite_from_fax_file_proc:
                if isinstance(cmt_options[cmt_field], str):
                    # URLConfigs can just be set to a constant
                    st.config[cmt_field] = fax_config[fax_field]
                else:
                    # FIXME: Remove once all cmt configs are URLConfigs
                    st.config[cmt_field] = (cmt_options[cmt_field][0] + '_constant',
                                            fax_config[fax_field])
            if overwrite_from_fax_file_sim:
                # CMT name allowed to be different from the config name
                # WFSim needs the cmt name
                cmt_name = cmt_options_full[cmt_field]['correction']

                st.config['fax_config_override_from_cmt'][fax_field] = (
                         cmt_name + '_constant', fax_config[fax_field])

    # And as the last step - manual overrrides, since they have the highest priority
    # User customized for simulation
    for option in cmt_option_overwrite_sim:
        if option not in cmt_options:
            raise ValueError(f'Overwrite option {option} is not using CMT by default '
                             'you should just use set config')
        if option not in _config_overlap.values():
            raise ValueError(f'Overwrite option {option} does not have mapping from '
                             f'CMT to fax config!')
        for fax_key, cmt_key in _config_overlap.items():
            if cmt_key == option:
                cmt_name = cmt_options_full[option]['correction']
                st.config['fax_config_override_from_cmt'][fax_key] = (
                                            cmt_name + '_constant',
                                            cmt_option_overwrite_sim[option])
            del(fax_key, cmt_key)
    # User customized for simulation
    for option in cmt_option_overwrite_proc:
        if option not in cmt_options:
            raise ValueError(f'Overwrite option {option} is not using CMT by default '
                             'you should just use set config')
        
        if isinstance(cmt_options[option], str):
            # URLConfig options can just be set to constants, no hacks needed
            # But for now lets keep things consistent for people
            st.config[option] = cmt_option_overwrite_proc[option]
        else:
            # CMT name allowed to be different from the config name
            # WFSim needs the cmt name
            cmt_name = cmt_options_full[option]['correction']
            st.config[option] = (cmt_name + '_constant', 
                                 cmt_option_overwrite_proc[option])
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
    s1_max_rise_time_post100=150,
    s1_min_coincidence=3,
    # Events*
    left_event_extension=int(0.3e6),
    right_event_extension=int(1e6),
    elife=1e6,
    electron_drift_velocity=("electron_drift_velocity_constant", 1.3325e-4),
    max_drift_length=96.9,
    electron_drift_time_gate=("electron_drift_time_gate_constant", 1700),
    se_gain=28.2,
    avg_se_gain=28.2,
    rel_extraction_eff=1.0,
    s1_xyz_map=f'itp_map://resource://{pax_file("XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json")}?fmt=json',  # noqa
    s2_xy_map=f'itp_map://resource://{pax_file("XENON1T_s2_xy_ly_SR1_v2.2.json")}?fmt=json',
    g1=0.1426,
    g2=11.55/(1 - 0.63),
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
        elife=1e6,
        electron_drift_velocity=("electron_drift_velocity_constant", 1.3325e-4),
        se_gain=28.2,
        avg_se_gain=28.2,
        rel_extraction_eff=1.0,
        s1_xyz_map=f'itp_map://resource://{pax_file("XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json")}?fmt=json',
        s2_xy_map=f'itp_map://resource://{pax_file("XENON1T_s2_xy_ly_SR1_v2.2.json")}?fmt=json',
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
    st.set_context_config(
        {'check_available': ('raw_records', 'led_calibration'),
         'free_options': list(x1t_context_config.keys())
         })
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
    st.deregister_plugins_with_missing_dependencies()
    return st
