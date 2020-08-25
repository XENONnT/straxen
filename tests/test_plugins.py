import tempfile
import strax
import straxen
import numpy as np
from immutabledict import immutabledict
from strax.testutils import run_id, recs_per_chunk

# Number of chunks for the dummy raw records we are writing here
N_CHUNKS = 2


@strax.takes_config(
    strax.Option('secret_time_offset', default=0, track=False)
)
class DummyRawRecords(strax.Plugin):
    """
    Provide dummy raw records for the mayor raw_record types
    """
    provides = ('raw_records', 'raw_records_he', 'raw_records_nv',)
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
        r = np.zeros(recs_per_chunk, self.dtype['raw_records'])
        t0 = chunk_i + self.config['secret_time_offset']
        r['time'] = t0
        r['length'] = r['dt'] = 1
        r['channel'] = np.arange(len(r))

        res = {p: self.chunk(start=t0, end=t0 + 1, data=r, data_type=p)
               for p in self.provides}
        return res


def test_all_plugins():
    """
    Try all plugins (except the DAQReader) in both multicore and signle core mode to see
    if we can really push some (empty) data from it and don't have any nasty problems like
    that we are referring to some non existant dali folder.
    """
    # Test both lazy and not lazy multicore/single core
    for max_workers in [2, 1]:
        for it, (context_name, st) in enumerate(
                {"nT": straxen.contexts.xenonnt_online(_database_init=False),
                 "1T": straxen.contexts.xenon1t_dali()}.items()
        ):
            print(f'Testing {context_name} context')
            with tempfile.TemporaryDirectory() as temp_dir:

                # Don't concern ourselves with rr_aqmon et cetera
                forbidden_plugins = tuple([p for p in
                                           straxen.daqreader.DAQReader.provides
                                           if p not in DummyRawRecords.provides])

                # Change config to allow for testing both multiprocessing and lazy mode
                st.set_context_config({'forbid_creation_of': forbidden_plugins})
                if max_workers - 1:
                     st.set_context_config({
                         'allow_multiprocess': True,
                         'allow_lazy': False,
                         'timeout': 60,  # we don't want to build travis for ever
                     })

                st.register(DummyRawRecords)
                st.storage = [strax.DataDirectory(temp_dir)]

                # As we use a temporary directory we should have a clean start
                assert not st.is_stored(run_id, 'raw_records'), 'have RR???'

                # 1Ts NN seems funky, let's ignore it for now?
                target = {'nT': 'event_info', '1T': 'peak_basics'}[context_name]

                # Create stuff
                st.make(run_id=run_id,
                        targets=target,
                        max_workers=max_workers)
                # The stuff should be there
                assert st.is_stored(run_id, target), f'Could not make {target}'

                # I'm only going to do this for nT because:
                #  A) Doing this many more times does not give us much more
                #     info (everything above already worked fine)
                #  B) Most development will be on nT, 1T may get less changes
                #     in the near future
                if context_name == 'nT':
                    # First make some _he stuff for multiprocessing
                    st.make(run_id, 'peaks_he', max_workers=max_workers)

                    # Now make sure we can get some data for all plugins
                    for p in list(st._plugin_class_registry.keys()):
                        if p not in forbidden_plugins:
                            st.get_array(run_id=run_id,
                                         targets=p,
                                         max_workers=max_workers)

                            # Check for types that we want to save that they are stored.
                            if (int(st._plugin_class_registry['peaks'].save_when) >
                                    int(strax.SaveWhen.TARGET)):
                                is_stored = st.is_stored(run_id, p)
                                assert is_stored, f"{p} did not save correctly!"

    print("Wonderful all plugins work (= at least they don't fail), bye bye")
