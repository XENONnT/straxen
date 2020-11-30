import tempfile
import strax
import straxen
import numpy as np
from immutabledict import immutabledict
from strax.testutils import run_id, recs_per_chunk
import os

# Number of chunks for the dummy raw records we are writing here
N_CHUNKS = 2

##
# Tools
##


@strax.takes_config(
    strax.Option('secret_time_offset', default=0, track=False)
)
class DummyRawRecords(strax.Plugin):
    """
    Provide dummy raw records for the mayor raw_record types
    """
    provides = ('raw_records',
                'raw_records_he',
                'raw_records_nv',
                'raw_records_aqmon')
    parallel = 'process'
    depends_on = tuple()
    data_kind = immutabledict(zip(provides, provides))
    rechunk_on_save = False
    dtype = {p: strax.raw_record_dtype() for p in provides}

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < N_CHUNKS

    def compute(self, chunk_i):
        t0 = chunk_i + self.config['secret_time_offset']
        if chunk_i < N_CHUNKS - 1:
            r = np.zeros(recs_per_chunk, self.dtype['raw_records'])
            r['time'] = t0
            r['length'] = r['dt'] = 1
            r['channel'] = np.arange(len(r))
        else:
            r = np.zeros(0, self.dtype['raw_records'])
        res = {p: self.chunk(start=t0, end=t0 + 1, data=r, data_type=p)
               for p in self.provides}
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


def _update_context(st, max_workers, fallback_gains=None):
    # Change config to allow for testing both multiprocessing and lazy mode
    st.set_context_config({'forbid_creation_of': forbidden_plugins})
    st.register(DummyRawRecords)
    try:
        if straxen.uconfig is None:
            raise ValueError('uconfig did not import')
        # If you want to have quicker checks: always raise an ValueError
        # as the CMT does take quite long to load the right corrections.
        if max_workers > 1 and fallback_gains is not None:
            raise ValueError(
                'Use fallback gains for multicore to save time on tests')
    except ValueError:
        # Okay so we cannot initize the runs-database. Let's just use some
        # fallback values if they are specified.
        if ('gain_model' in st.config and
                st.config['gain_model'][0] == 'CMT_model'):
            if fallback_gains is None:
                # If you run into this error, go to the test_nT() - test and
                # see for example how it is done there.
                raise ValueError('Context uses CMT_model but no fallback_gains '
                                 'are specified in test_plugins.py for this '
                                 'context being tested')
            else:
                st.set_config({'gain_model': fallback_gains})
    if max_workers - 1:
        st.set_context_config({
            'allow_multiprocess': True,
            'allow_lazy': False,
            'timeout': 60,  # we don't want to build travis for ever
        })

##
# Tests
##


def test_1T(ncores=1):
    if ncores == 1:
        print('-- 1T lazy mode --')
    st = straxen.contexts.xenon1t_dali()
    _update_context(st, ncores)

    # Register the 1T plugins for this test as well
    st.register_all(straxen.plugins.x1t_cuts)
    _run_plugins(st, make_all=False, max_wokers=ncores)
    # Test issue #233
    st.search_field('cs1')
    print(st.context_config)


def test_nT(ncores=1):
    if ncores == 1:
        print('-- nT lazy mode --')
    st = straxen.contexts.xenonnt_online(_database_init=False)
    offline_gain_model = ('to_pe_constant', 'gain_placeholder')
    _update_context(st, ncores, fallback_gains=offline_gain_model)
    # Lets take an abandoned run where we actually have gains for in the CMT
    _run_plugins(st, make_all=True, max_wokers=ncores, run_id='008900')
     # Test issue #233
    st.search_field('cs1')
    print(st.context_config)


def test_nT_mutlticore():
    print('nT multicore')
    test_nT(2)

# Disable the test below as it saves some time in travis and gives limited new
# information as most development is on nT-plugins.
# def test_1T_mutlticore():
#     print('1T multicore')
#     test_1T(2)
