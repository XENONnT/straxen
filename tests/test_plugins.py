import tempfile
import strax
import numpy as np
from immutabledict import immutabledict
import straxen
from straxen.common import pax_file, aux_repo
##
# Tools
##
# Let's make a dummy map for NVeto
nveto_pmt_dummy_df = {'channel': list(range(2000, 2120)),
                      'x': list(range(120)),
                      'y': list(range(120)),
                      'z': list(range(120))}

# Some configs are better obtained from the strax_auxiliary_files repo.
# Let's use small files, we don't want to spend a lot of time downloading
# some file.
testing_config_nT = dict(
    nn_architecture=
    aux_repo + 'f0df03e1f45b5bdd9be364c5caefdaf3c74e044e/fax_files/mlp_model.json',
    nn_weights=
    aux_repo + 'f0df03e1f45b5bdd9be364c5caefdaf3c74e044e/fax_files/mlp_model.h5',
    gain_model=("to_pe_placeholder", True),
    s2_xy_correction_map=pax_file('XENON1T_s2_xy_ly_SR0_24Feb2017.json'),
    elife_conf=('elife_constant', 1e6),
    baseline_samples_nv=10,
    fdc_map=pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz'),
    gain_model_nv=("adc_nv", True),
    gain_model_mv=("adc_mv", True),
    nveto_pmt_position_map=nveto_pmt_dummy_df,
    s1_xyz_correction_map=pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR0_pax-680_fdc-3d_v0.json'),
    electron_drift_velocity=("electron_drift_velocity_constant", 1e-4),
    s1_aft_map=aux_repo + 'ffdadba3439ae7922b19f5dd6479348b253c09b0/strax_files/s1_aft_UNITY_xyz_XENONnT.json',
    s2_optical_map=aux_repo + '8a6f0c1a4da4f50546918cd15604f505d971a724/strax_files/s2_map_UNITY_xy_XENONnT.json',
    s1_optical_map=aux_repo + '8a6f0c1a4da4f50546918cd15604f505d971a724/strax_files/s1_lce_UNITY_xyz_XENONnT.json',
    electron_drift_time_gate=("electron_drift_time_gate_constant", 2700),
    hit_min_amplitude='pmt_commissioning_initial',
    hit_min_amplitude_nv=20,
    hit_min_amplitude_mv=80,
    hit_min_amplitude_he='pmt_commissioning_initial_he'
)

testing_config_1T = dict(
    hev_gain_model=('1T_to_pe_placeholder', False),
    gain_model=('1T_to_pe_placeholder', False),
    elife_conf=('elife_constant', 1e6),
    electron_drift_velocity=("electron_drift_velocity_constant", 1e-4),
    electron_drift_time_gate=("electron_drift_time_gate_constant", 1700),
)

test_run_id_nT = '008900'
test_run_id_1T = '180423_1021'


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
                 run_id=test_run_id_nT,
                 **proces_kwargs):
    """
    Try all plugins (except the DAQReader) for a given context (st) to see if
    we can really push some (empty) data from it and don't have any nasty
    problems like that we are referring to some non existant dali folder.
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        st.storage = [strax.DataDirectory(temp_dir)]

        # As we use a temporary directory we should have a clean start
        assert not st.is_stored(run_id, 'raw_records'), 'have RR???'

        # Create event info
        target = 'event_info'
        st.make(run_id=run_id,
                targets=target,
                **proces_kwargs)

        # The stuff should be there
        assert st.is_stored(run_id, target), f'Could not make {target}'

        if not make_all:
            return

        end_targets = set(st._get_end_targets(st._plugin_class_registry))
        for p in end_targets-set(forbidden_plugins):
            st.make(run_id, p)
        # Now make sure we can get some data for all plugins
        all_datatypes = set(st._plugin_class_registry.keys())
        for p in all_datatypes-set(forbidden_plugins):
            should_be_stored = (st._plugin_class_registry[p].save_when ==
                                strax.SaveWhen.ALWAYS)
            if should_be_stored:
                is_stored = st.is_stored(run_id, p)
                assert is_stored, f"{p} did not save correctly!"
    print("Wonderful all plugins work (= at least they don't fail), bye bye")


def _update_context(st, max_workers, fallback_gains=None, nt=True):
    # Change config to allow for testing both multiprocessing and lazy mode
    st.set_context_config({'forbid_creation_of': forbidden_plugins})
    # Ignore strax-internal warnings
    st.set_context_config({'free_options': tuple(st.config.keys())})
    st.register(DummyRawRecords)
    if nt and not straxen.utilix_is_configured():
        st.set_config(testing_config_nT)
        del st._plugin_class_registry['peak_positions_mlp']
        del st._plugin_class_registry['peak_positions_cnn']
        del st._plugin_class_registry['peak_positions_gcn']
        st.register(straxen.PeakPositions1T)
        print(f"Using {st._plugin_class_registry['peak_positions']} for posrec tests")
        st.set_config({'gain_model': fallback_gains})

    elif not nt:
        if straxen.utilix_is_configured():
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
    _run_plugins(st, make_all=True, max_wokers=ncores, run_id=test_run_id_1T)

    # Test issue #233
    st.search_field('cs1')

    # set all the configs to be non-CMT
    st.set_config(testing_config_1T)
    _test_child_options(st, test_run_id_1T)

    print(st.context_config)


def test_nT(ncores=1):
    if ncores == 1:
        print('-- nT lazy mode --')
    st = straxen.contexts.xenonnt_online(_database_init=straxen.utilix_is_configured(),
                                         use_rucio=False)
    offline_gain_model = ("to_pe_placeholder", True)
    _update_context(st, ncores, fallback_gains=offline_gain_model, nt=True)
    # Lets take an abandoned run where we actually have gains for in the CMT
    _run_plugins(st, make_all=True, max_wokers=ncores, run_id=test_run_id_nT)
    # Test issue #233
    st.search_field('cs1')
    # Test of child plugins:
    _test_child_options(st, test_run_id_nT)
    print(st.context_config)


def test_nT_mutlticore():
    print('nT multicore')
    test_nT(2)

# Disable the test below as it saves some time in travis and gives limited new
# information as most development is on nT-plugins.
# def test_1T_mutlticore():
#     print('1T multicore')
#     test_1T(2)
