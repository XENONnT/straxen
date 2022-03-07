import tempfile
import strax
import straxen
from straxen.test_utils import nt_test_run_id, DummyRawRecords, testing_config_1T, test_run_id_1T
from immutabledict import immutabledict


def _run_plugins(st,
                 make_all=False,
                 run_id=nt_test_run_id,
                 from_scratch=False,
                 **process_kwargs):
    """
    Try all plugins (except the DAQReader) for a given context (st) to see if
    we can really push some (empty) data from it and don't have any nasty
    problems like that we are referring to some non existant dali folder.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        if from_scratch:
            st.storage = [strax.DataDirectory(temp_dir)]
            # As we use a temporary directory we should have a clean start
            assert not st.is_stored(run_id, 'raw_records'), 'have RR???'

        # Don't concern ourselves with rr_aqmon et cetera
        _forbidden_plugins = tuple([p for p in
                                    straxen.daqreader.DAQReader.provides
                                    if p not in
                                    st._plugin_class_registry['raw_records'].provides])
        st.set_context_config({'forbid_creation_of': _forbidden_plugins})

        # Create event info
        target = 'event_info'
        st.make(run_id=run_id,
                targets=target,
                **process_kwargs)

        # The stuff should be there
        assert st.is_stored(run_id, target), f'Could not make {target}'

        if not make_all:
            return

        end_targets = set(st._get_end_targets(st._plugin_class_registry))
        for data_type in end_targets - set(_forbidden_plugins):
            if data_type in straxen.DAQReader.provides:
                continue
            st.make(run_id, data_type)
        # Now make sure we can get some data for all plugins
        all_datatypes = set(st._plugin_class_registry.keys())
        assert 'pulse_counts' in data_type
        for data_type in all_datatypes - set(_forbidden_plugins):
            savewhen = st._plugin_class_registry[data_type].save_when
            if isinstance(savewhen, (dict, immutabledict)):
                savewhen = savewhen[data_type]
            should_be_stored = savewhen == strax.SaveWhen.ALWAYS
            if data_type == 'pulse_counts':
              assert should_be_stored
            if should_be_stored:
                is_stored = st.is_stored(run_id, data_type)
                assert is_stored, f"{data_type} did not save correctly!"
    print("Wonderful all plugins work (= at least they don't fail), bye bye")

def _update_context(st, max_workers, nt=True):
    # Ignore strax-internal warnings
    st.set_context_config({'free_options': tuple(st.config.keys())})
    if not nt:
        st.register(DummyRawRecords)
        if straxen.utilix_is_configured(warning_message=False):
            # Set some placeholder gain as this takes too long for 1T to load from CMT
            st.set_config({k: v for k, v in testing_config_1T.items() if
                           k in ('hev_gain_model', 'gain_model')})
        else:
            st.set_config(testing_config_1T)

    if max_workers - 1:
        st.set_context_config({
            'allow_multiprocess': True,
            'allow_lazy': False,
            'timeout': 60,  # we don't want to build travis for ever
        })
    print('--- Plugins ---')
    for k, v in st._plugin_class_registry.items():
        print(k, v)


def _test_child_options(st, run_id):
    """
    Test which checks if child options are handled correctly.
    """
    # Register all used plugins
    plugins = []
    already_seen = []
    for data_type in st._plugin_class_registry.keys():
        if data_type in already_seen or data_type in straxen.DAQReader.provides:
            continue

        p = st.get_single_plugin(run_id, data_type)
        plugins.append(p)
        already_seen += p.provides

    # Loop over all plugins and check if child options were propagated to the parent:
    for p in plugins:
        for option_name, option in p.takes_config.items():
            # Check if option is a child option:
            if option.child_option:
                # Get corresponding parent option. Do not have to test if
                # parent option name is defined this is already done in strax
                parent_name = option.parent_option_name

                # Now check if parent config was replaced with child:
                t = p.config[parent_name] == p.config[option_name]
                assert t, (f'This is strange the child option "{option_name}" was set to '
                           f'{p.config[option_name]}, but the corresponding parent config'
                           f' "{parent_name}" has the value {p.config[parent_name]}. '
                           f'Please check the options of {p.__class__.__name__} and if '
                           'it is a child plugin (child_plugin=True)!')

                # Test if parent names were removed from the lineage:
                t = parent_name in p.lineage[p.provides[-1]][2]
                assert not t, (f'Found "{parent_name}" in the lineage of {p.__class__.__name__}. '
                               f'This should not have happend since "{parent_name}" is a child of '
                               f'"{option_name}"!')


def test_1T(ncores=1):
    if ncores == 1:
        print('-- 1T lazy mode --')
    st = straxen.contexts.xenon1t_dali()
    _update_context(st, ncores, nt=False)

    # Register the 1T plugins for this test as well
    st.register_all(straxen.plugins.x1t_cuts)
    for _plugin, _plugin_class in st._plugin_class_registry.items():
        if 'cut' in str(_plugin).lower():
            _plugin_class.save_when = strax.SaveWhen.ALWAYS

    # Run the test
    _run_plugins(st, make_all=True, max_workers=ncores, run_id=test_run_id_1T, from_scratch=True)

    # Test issue #233
    st.search_field('cs1')

    # set all the configs to be non-CMT
    st.set_config(testing_config_1T)
    _test_child_options(st, test_run_id_1T)

    print(st.context_config)


def test_nT(ncores=1):
    if ncores == 1:
        print('-- nT lazy mode --')
    init_database = straxen.utilix_is_configured(warning_message=False)
    st = straxen.test_utils.nt_test_context(
        _database_init=init_database,
        use_rucio=False,
        deregister=('events_sync_nv', 'events_sync_mv')
    )
    _update_context(st, ncores, nt=True)
    # Lets take an abandoned run where we actually have gains for in the CMT
    _run_plugins(st, make_all=True, max_workers=ncores, run_id=nt_test_run_id)
    # Test issue #233
    st.search_field('cs1')
    # Test of child plugins:
    _test_child_options(st, nt_test_run_id)
    print(st.context_config)


def test_nT_mutlticore():
    print('nT multicore')
    test_nT(3)
