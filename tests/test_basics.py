import tempfile
import os

import straxen


def test_build_event_info():
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Temporary directory is ", temp_dir)
        os.chdir(temp_dir)

        print("Downloading test data (if needed)")
        straxen.download_test_data()
        st = straxen.contexts.demo()

        run_df = st.select_runs(available='raw_records')
        run_id = run_df.iloc[0]['name']
        assert run_id == '180423_1021'

        print("Test processing")
        df = st.get_df(run_id, 'event_info')
        assert len(df) > 0
        assert 'cs1' in df.columns
        assert df['cs1'].sum() > 0
