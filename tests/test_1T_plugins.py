"""Test for 1T plugins, nT plugins are tested in the ./plugins directory"""
import tempfile
import strax
import straxen
from straxen.test_utils import DummyRawRecords
from immutabledict import immutabledict

test_run_id_1T = '180423_1021'

testing_config_1T = dict(
    hev_gain_model='legacy-to-pe://1T_to_pe_placeholder',
    gain_model='legacy-to-pe://1T_to_pe_placeholder',
    elife=1e6,
    electron_drift_velocity=1e-4,
    electron_drift_time_gate=1700,
)


def _run_plugins(st, make_all=False, run_id=test_run_id_1T, **process_kwargs):
    """
    Try all plugins (except the DAQReader) for a given context (st) to see if
    we can really push some (empty) data from it and don't have any nasty
    problems like that we are referring to some non existant dali folder.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        st.storage = [strax.DataDirectory(temp_dir)]
        # As we use a temporary directory we should have a clean start
        assert not st.is_stored(run_id, 'raw_records'), 'have RR???'

        if not make_all:
            return

        end_targets = set(st._get_end_targets(st._plugin_class_registry))
        if st.context_config['allow_multiprocess']:
            st.make(run_id, list(end_targets), allow_multiple=True, **process_kwargs)
        else:
            for data_type in end_targets:
                st.make(run_id, data_type)
        # Now make sure we can get some data for all plugins
        all_datatypes = set(st._plugin_class_registry.keys())
        for data_type in all_datatypes:
            savewhen = st._plugin_class_registry[data_type].save_when
            if isinstance(savewhen, (dict, immutabledict)):
                savewhen = savewhen[data_type]
            should_be_stored = savewhen == strax.SaveWhen.ALWAYS
            if should_be_stored:
                is_stored = st.is_stored(run_id, data_type)
                assert is_stored, f"{data_type} did not save correctly!"
    print("Wonderful all plugins work (= at least they don't fail), bye bye")


def _update_context(st, max_workers):
    # Ignore strax-internal warnings
    st.set_context_config({'free_options': tuple(st.config.keys()),
                           'forbid_creation_of': ()})

    st.register(DummyRawRecords)
    st.set_config(testing_config_1T)

    if max_workers - 1:
        st.set_context_config({
            'allow_multiprocess': True,
            'allow_lazy': False,
            'timeout': 120,  # we don't want to build travis for ever
            'allow_shm': strax.processor.SHMExecutor is None,
        })


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


def test_1T(ncores=2):
    st = straxen.contexts.xenon1t_dali()
    _update_context(st, ncores)
    st.register_all(straxen.legacy.plugins_1t.x1t_cuts)
    for _plugin, _plugin_class in st._plugin_class_registry.items():
        if 'cut' in str(_plugin).lower():
            _plugin_class.save_when = strax.SaveWhen.ALWAYS

    # Run the test
    _run_plugins(st, make_all=True, max_workers=ncores, run_id=test_run_id_1T)
    # set all the configs to be non-CMT
    st.set_config(testing_config_1T)
    _test_child_options(st, test_run_id_1T)

    print(st.context_config)
