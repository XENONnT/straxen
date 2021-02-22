import tempfile
import os

import numpy as np
import straxen

test_run_id_1T = '180423_1021'


def test_straxen():
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("Temporary directory is ", temp_dir)
            os.chdir(temp_dir)

            print("Downloading test data (if needed)")
            st = straxen.contexts.demo()
            # Ignore strax-internal warnings
            st.set_context_config({'free_options': tuple(st.config.keys())})

            run_df = st.select_runs(available='raw_records')
            print(run_df)
            run_id = run_df.iloc[0]['name']
            assert run_id == test_run_id_1T

            print("Test processing")
            df = st.get_df(run_id, 'event_info')

            assert len(df) > 0
            assert 'cs1' in df.columns
            assert df['cs1'].sum() > 0
            assert not np.all(np.isnan(df['x'].values))

            print('Test common.get_livetime_sec')
            events = st.get_array(run_id, 'peaks')
            straxen.get_livetime_sec(st, test_run_id_1T, things=events)
            # TODO: find a way to break up the tests
            # surely pytest has common startup/cleanup?

            print("Test mini analysis")
            @straxen.mini_analysis(requires=('raw_records',))
            def count_rr(raw_records):
                return len(raw_records)

            n = st.count_rr(test_run_id_1T)
            assert n > 100

        # On windows, you cannot delete the current process'
        # working directory, so we have to chdir out first.
        finally:
            os.chdir('..')
