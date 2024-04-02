from immutabledict import immutabledict
import strax
import straxen
from copy import deepcopy
from straxen import HAVE_ADMIX
import os
import warnings
import typing as ty
from pandas.util._decorators import deprecate_kwarg
import socket
from straxen.plugins.peaklets.peaklet_classification_som import PeakletClassificationSOM

common_opts = dict(
    register_all=[],
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
        straxen.EventInfoDouble,
        straxen.DistinctChannels,
    ],
    check_available=('raw_records', 'peak_basics'),
    store_run_fields=(
        'name', 'number',
        'start', 'end', 'livetime', 'mode', 'source'))

xnt_common_config = dict(
    n_tpc_pmts=straxen.n_tpc_pmts,
    n_top_pmts=straxen.n_top_pmts,
    gain_model="list-to-array://"
               "xedocs://pmt_area_to_pes"
               "?as_list=True"
               "&sort=pmt"
               "&detector=tpc"
               "&run_id=plugin.run_id"
               "&version=ONLINE"
               "&attr=value",
    gain_model_nv="list-to-array://"
                  "xedocs://pmt_area_to_pes"
                  "?as_list=True"
                  "&sort=pmt"
                  "&detector=neutron_veto"
                  "&run_id=plugin.run_id"
                  "&version=ONLINE"
                  "&attr=value",
    gain_model_mv="list-to-array://"
                  "xedocs://pmt_area_to_pes"
                  "?as_list=True"
                  "&sort=pmt"
                  "&detector=muon_veto"
                  "&run_id=plugin.run_id"
                  "&version=ONLINE"
                  "&attr=value",
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
    fdc_map='itp_map://'
            'resource://'
            'xedocs://fdc_maps'
            '?algorithm=plugin.default_reconstruction_algorithm'
            '&version=ONLINE'
            '&attr=value'
            '&run_id=plugin.run_id'
            '&fmt=binary'
            '&scale_coordinates=plugin.coordinate_scales',
    z_bias_map='itp_map://'
               'resource://'
               'XnT_z_bias_map_chargeup_20230329.json.gz?'
               'fmt=json.gz'
               '&method=RegularGridInterpolator',
)
# these are placeholders to avoid calling cmt with non integer run_ids. Better solution pending.
# s1,s2 and fd corrections are still problematic
xnt_simulation_config = deepcopy(xnt_common_config)
xnt_simulation_config.update(gain_model="legacy-to-pe://to_pe_placeholder",
                             gain_model_nv="legacy-to-pe://adc_nv",
                             gain_model_mv="legacy-to-pe://adc_mv",
                             elife=1e6,
                             )

# Plugins in these files have nT plugins, E.g. in pulse&peak(let)
# processing there are plugins for High Energy plugins. Therefore, do not
# st.register_all in 1T contexts.
xnt_common_opts = common_opts.copy()
xnt_common_opts.update({
    'register': common_opts['register'] + [],
    'register_all': common_opts['register_all'] + [straxen.plugins,
                                                   ],
    'use_per_run_defaults': False,
})


##
# XENONnT
##


def xenonnt(global_version='global_ONLINE',
            _from_cutax=False, **kwargs):
    """XENONnT context"""
    if not _from_cutax and global_version != 'global_ONLINE':
        warnings.warn('Don\'t load a context directly from straxen, '
                      'use cutax instead!')
        
    st = straxen.contexts.xenonnt_online(global_version=global_version, **kwargs)

    return st


def xenonnt_som(global_version='global_ONLINE', xedocs_version=None,
                _from_cutax=False, **kwargs):
    """XENONnT context for the SOM"""

    st = straxen.contexts.xenonnt(global_version=global_version, xedocs_version=xedocs_version,
                                  _from_cutax=_from_cutax, **kwargs)
    del st._plugin_class_registry['peaklet_classification']
    st.register(PeakletClassificationSOM)

    return st


def find_rucio_local_path(include_rucio_local, _rucio_local_path):
    """
    Check the hostname to determine which rucio local path to use. Note that access to
    /dali/lgrandi/rucio/ is possible only if you are on dali compute node or login node.

    :param include_rucio_local: add the rucio local storage frontend.
        This is only needed if one wants to do a fuzzy search in the
        data the runs database is out of sync with rucio
    :param _rucio_local_path: str, path of local RSE of rucio. Only use
        for testing!
    """
    hostname = socket.gethostname()
    # if you are on dali compute node, do nothing
    if ('dali' in hostname) and ('login' not in hostname):
        _include_rucio_local = include_rucio_local
        __rucio_local_path = _rucio_local_path
    # Assumed the only other option is 'midway' or login nodes, 
    # where we have full access to dali and project space. 
    # This doesn't make sense outside XENON but we don't care.
    else:
        _include_rucio_local = True
        __rucio_local_path = '/project/lgrandi/rucio/'
        print('You specified _auto_append_rucio_local=True and you are not on dali compute nodes,'
              'so we will add the following rucio local path: ', __rucio_local_path)

    return _include_rucio_local, __rucio_local_path


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
                   _auto_append_rucio_local: bool = True,
                   _rucio_path: str = '/dali/lgrandi/rucio/',
                   _rucio_local_path: ty.Optional[str] = None,
                   _raw_paths: ty.Optional[str] = ['/dali/lgrandi/xenonnt/raw'],
                   _processed_paths: ty.Optional[ty.List[str]] = ['/dali/lgrandi/xenonnt/processed',
                                                                  '/project2/lgrandi/xenonnt/processed',
                                                                  '/project/lgrandi/xenonnt/processed'],

                   # Testing options
                   _context_config_overwrite: ty.Optional[dict] = None,
                   _database_init: bool = True,
                   _forbid_creation_of: ty.Optional[dict] = None,
                   global_version: str = "global_ONLINE",
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

    :param _auto_append_rucio_local: bool, whether or not to automatically append the 
        rucio local path
    :param _rucio_path: str, path of rucio
    :param _rucio_local_path: str, path of local RSE of rucio. Only use
        for testing!
    :param _raw_paths: list[str], common path of the raw-data
    :param _processed_paths: list[str]. common paths of output data
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

    if _auto_append_rucio_local:
        include_rucio_local, _rucio_local_path = find_rucio_local_path(
            include_rucio_local, _rucio_local_path)

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
        for _raw_path in _raw_paths:
            st.storage += [
                strax.DataDirectory(
                    _raw_path,
                    readonly=True,
                    take_only=straxen.DAQReader.provides)]
        for _processed_path in _processed_paths:
            st.storage += [strax.DataDirectory(
                _processed_path,
                readonly=True)]

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
                       'individual_peak_monitor',
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

    if global_version is not None:
        st.apply_xedocs_configs(version=global_version, **kwargs)

    return st


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


def xenonnt_simulation_offline(output_folder: str = './strax_data',
                               wfsim_registry: str = 'RawRecordsFromFaxNT',
                               run_id: ty.Optional[str] = None,
                               global_version: ty.Optional[str] = None,
                               fax_config: ty.Optional[str] = None,
                               ):
    """
    :param output_folder: strax_data folder
    :param wfsim_registry: Raw_records generation mechanism,
                           'RawRecordsFromFaxNT', 'RawRecordsFromMcChain', etc,
                           https://github.com/XENONnT/WFSim/blob/master/wfsim/strax_interface.py
    :param run_id: Real run_id to use to fetch the corrections
    :param global_version: Global versions
                           https://github.com/XENONnT/corrections/tree/master/XENONnT/global_versions
    :param fax_config: WFSim configuration files
                       https://github.com/XENONnT/private_nt_aux_files/blob/master/sim_files/fax_config_nt_sr0_v4.json
    :return: strax context for simulation
    """
    if run_id is None:
        raise ValueError("Specify a run_id to load the corrections")
    if global_version is None:
        raise ValueError("Specify a correction global version")
    if fax_config is None:
        raise ValueError("Specify a simulation configuration file")

    import wfsim
    # General strax context, register common plugins
    st = strax.Context(storage=strax.DataDirectory(output_folder),
                       **straxen.contexts.xnt_common_opts)
    # Register simulation configs required by WFSim plugins
    st.config.update(dict(detector='XENONnT',
                          fax_config=fax_config,
                          check_raw_record_overlaps=True,
                          **straxen.contexts.xnt_common_config))
    # Register WFSim raw_records plugin to overwrite real data raw_records
    wfsim_plugin = getattr(wfsim, wfsim_registry)
    st.register(wfsim_plugin)
    for plugin_name in wfsim_plugin.provides:
        assert 'wfsim' in str(st._plugin_class_registry[plugin_name])
    # Register offline global corrections same as real data
    st.apply_xedocs_configs(version=global_version)
    # Real data correction is run_id dependent,
    # but in sim we often use run_id not in the runDB
    # So we switch the run_id dependence to a specific run -> run_id
    local_versions = st.config
    for config_name, url_config in local_versions.items():
        if isinstance(url_config, str):
            if 'run_id' in url_config:
                local_versions[config_name] = straxen.URLConfig.format_url_kwargs(url_config, run_id=run_id)
    st.config = local_versions
    # In simulation, the raw_records generation depends on gain measurement
    st.config['gain_model_mc'] = st.config['gain_model']
    # No blinding in simulations
    st.config["event_info_function"] = "disabled"
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
                    **straxen.contexts.xnt_common_config, ),
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
    if (not cmt_run_id_sim) and (not cmt_run_id_proc):
        raise RuntimeError("cmt_run_id_sim and cmt_run_id_proc are None. "
                           "You have to specify at least one CMT run id. ")
    if (cmt_run_id_sim and cmt_run_id_proc) and (cmt_run_id_sim != cmt_run_id_proc):
        print("INFO : divergent CMT runs for simulation and processing")
        print("    cmt_run_id_sim".ljust(25), cmt_run_id_sim)
        print("    cmt_run_id_proc".ljust(25), cmt_run_id_proc)
    else:
        cmt_id = cmt_run_id_sim or cmt_run_id_proc
        cmt_run_id_sim = cmt_id
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
            del (fax_key, cmt_key)
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
# XENON1T, see straxen/legacy
##


def demo():
    """Return strax context used in the straxen demo notebook"""
    return straxen.legacy.contexts_1t.demo()


def fake_daq():
    """Context for processing fake DAQ data in the current directory"""
    return straxen.legacy.contexts_1t.fake_daq()


def xenon1t_dali(output_folder='./strax_data', build_lowlevel=False, **kwargs):
    return straxen.legacy.contexts_1t.xenon1t_dali(output_folder=output_folder, build_lowlevel=build_lowlevel, **kwargs)


def xenon1t_led(**kwargs):
    return straxen.legacy.contexts_1t.xenon1t_led(**kwargs)


def xenon1t_simulation(output_folder='./strax_data'):
    return straxen.legacy.contexts_1t.xenon1t_simulation(output_folder=output_folder)
