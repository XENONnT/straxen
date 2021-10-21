import tempfile
import strax
import numpy as np
from immutabledict import immutabledict
import straxen
from straxen.common import pax_file, aux_repo
from straxen.test_utils import nt_test_run_id

##
# Tools
##

test_run_id_1T = '180423_1021'

testing_config_1T = dict(
    hev_gain_model=('1T_to_pe_placeholder', False),
    gain_model=('1T_to_pe_placeholder', False),
    elife_conf=('elife_constant', 1e6),
    electron_drift_velocity=("electron_drift_velocity_constant", 1e-4),
    electron_drift_time_gate=("electron_drift_time_gate_constant", 1700),
)


@strax.takes_config(
    strax.Option('secret_time_offset', default=0, track=False),
    strax.Option('recs_per_chunk', default=10, track=False),
    strax.Option('n_chunks', default=2, track=False,
                 help='Number of chunks for the dummy raw records we are writing here'),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="frozendict mapping subdetector to (min, max) "
                      "channel number.")
)
class DummyRawRecords(strax.Plugin):
    """
    Provide dummy raw records for the mayor raw_record types
    """
    provides = ('raw_records',
                'raw_records_he',
                'raw_records_nv',
                'raw_records_aqmon',
                'raw_records_aux_mv',
                'raw_records_mv'
                )
    parallel = 'process'
    depends_on = tuple()
    data_kind = immutabledict(zip(provides, provides))
    rechunk_on_save = False
    dtype = {p: strax.raw_record_dtype() for p in provides}

    def setup(self):
        self.channel_map_keys = {'he': 'he',
                                 'nv': 'nveto',
                                 'aqmon': 'aqmon',
                                 'aux_mv': 'aux_mv',
                                 's_mv': 'mv',
                                 }  # s_mv otherwise same as aux in endswith

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < self.config['n_chunks']

    def compute(self, chunk_i):
        t0 = chunk_i + self.config['secret_time_offset']
        if chunk_i < self.config['n_chunks'] - 1:
            # One filled chunk
            r = np.zeros(self.config['recs_per_chunk'], self.dtype['raw_records'])
            r['time'] = t0
            r['length'] = r['dt'] = 1
            r['channel'] = np.arange(len(r))
        else:
            # One empty chunk
            r = np.zeros(0, self.dtype['raw_records'])

        res = {}
        for p in self.provides:
            rr = np.copy(r)
            # Add detector specific channel offset:
            for key, channel_key in self.channel_map_keys.items():
                if channel_key not in self.config['channel_map']:
                    # Channel map for 1T is different.
                    continue
                if p.endswith(key):
                    s, e = self.config['channel_map'][channel_key]
                    rr['channel'] += s
            res[p] = self.chunk(start=t0, end=t0 + 1, data=rr, data_type=p)
        return res


# Don't concern ourselves with rr_aqmon et cetera
forbidden_plugins = tuple([p for p in
                           straxen.daqreader.DAQReader.provides
                           if p not in DummyRawRecords.provides])


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
        for p in end_targets - set(forbidden_plugins):
            if 'raw' in p:
                continue
            st.make(run_id, p)
        # Now make sure we can get some data for all plugins
        all_datatypes = set(st._plugin_class_registry.keys())
        for p in all_datatypes - set(forbidden_plugins):
            should_be_stored = (st._plugin_class_registry[p].save_when ==
                                strax.SaveWhen.ALWAYS)
            if should_be_stored:
                is_stored = st.is_stored(run_id, p)
                assert is_stored, f"{p} did not save correctly!"
    print("Wonderful all plugins work (= at least they don't fail), bye bye")


def _update_context(st, max_workers, nt=True):
    # Change config to allow for testing both multiprocessing and lazy mode
    st.set_context_config({'forbid_creation_of': forbidden_plugins})
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


##
# Tests
##


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
    test_nT(2)

# Disable the test below as it saves some time in travis and gives limited new
# information as most development is on nT-plugins.
# def test_1T_mutlticore():
#     print('1T multicore')
#     test_1T(2)
