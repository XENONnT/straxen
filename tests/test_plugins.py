import tempfile
import strax
import numpy as np
from immutabledict import immutabledict
from strax.testutils import run_id, recs_per_chunk
import straxen

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
    straxen.aux_repo + 'f0df03e1f45b5bdd9be364c5caefdaf3c74e044e/fax_files/mlp_model.json',
    nn_weights=
    straxen.aux_repo + 'f0df03e1f45b5bdd9be364c5caefdaf3c74e044e/fax_files/mlp_model.h5',
    gain_model=
    ('to_pe_per_run',
     straxen.aux_repo + '58e615f99a4a6b15e97b12951c510de91ce06045/fax_files/to_pe_nt.npy'),
    s2_xy_correction_map=straxen.pax_file('XENON1T_s2_xy_ly_SR0_24Feb2017.json'),
    elife_conf=straxen.aux_repo + '3548132b55f81a43654dba5141366041e1daaf01/strax_files/elife.npy',
    baseline_samples_nv=10,
    fdc_map=straxen.pax_file('XENON1T_FDC_SR0_data_driven_3d_correction_tf_nn_v0.json.gz'),
    gain_model_nv=("to_pe_constant", "adc_nv"),
    nveto_pmt_position_map=nveto_pmt_dummy_df,
)

testing_config_1T = dict(
    hev_gain_model=('to_pe_constant', 0.0085),
    gain_model=('to_pe_constant', 0.0085),
    elife_conf=straxen.aux_repo + '3548132b55f81a43654dba5141366041e1daaf01/strax_files/elife.npy',
)

test_run_id_nT = '008900'


@strax.takes_config(
    strax.Option('secret_time_offset', default=0, track=False),
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
                                 's_mv': 'mv',}  # s_mv otherwise same as aux in endswith

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < self.config['n_chunks']

    def compute(self, chunk_i):
        t0 = chunk_i + self.config['secret_time_offset']
        if chunk_i < self.config['n_chunks'] - 1:
            # One filled chunk
            r = np.zeros(recs_per_chunk, self.dtype['raw_records'])
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
                 run_id=run_id,
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

        # I'm only going to do this for nT because:
        #  A) Doing this many more times does not give us much more
        #     info (everything above already worked fine)
        #  B) Most development will be on nT, 1T may get less changes
        #     in the near future
        if make_all:
            # Now make sure we can get some data for all plugins
            for p in list(st._plugin_class_registry.keys()):
                if p not in forbidden_plugins:
                    st.get_array(run_id=run_id,
                                 targets=p,
                                 **proces_kwargs)

                    # Check for types that we want to save that they are stored.
                    if (int(st._plugin_class_registry['peaks'].save_when) >
                            int(strax.SaveWhen.TARGET)):
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


def _test_child_options(st):
    """
    Test which checks if child options are handled correctly.
    """
    # Register all used plugins
    plugins = []
    already_seen = []
    for data_type in st._plugin_class_registry.keys():
        if data_type in already_seen or data_type in straxen.DAQReader.provides:
            continue

        p = st.get_single_plugin('0', data_type)
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
    _run_plugins(st, make_all=True, max_wokers=ncores)
    # Test issue #233
    st.search_field('cs1')
    _test_child_options(st)
    print(st.context_config)


def test_nT(ncores=1):
    if ncores == 1:
        print('-- nT lazy mode --')
    st = straxen.contexts.xenonnt_online(_database_init=straxen.utilix_is_configured())
    offline_gain_model = ('to_pe_constant', 'gain_placeholder')
    _update_context(st, ncores, fallback_gains=offline_gain_model, nt=True)
    # Lets take an abandoned run where we actually have gains for in the CMT
    _run_plugins(st, make_all=True, max_wokers=ncores, run_id=test_run_id_nT)
    # Test issue #233
    st.search_field('cs1')
    # Test of child plugins:
    _test_child_options(st)
    print(st.context_config)


def test_nT_mutlticore():
    print('nT multicore')
    test_nT(2)

# Disable the test below as it saves some time in travis and gives limited new
# information as most development is on nT-plugins.
# def test_1T_mutlticore():
#     print('1T multicore')
#     test_1T(2)
