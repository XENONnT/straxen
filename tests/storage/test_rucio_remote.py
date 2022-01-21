import unittest
import straxen
import os
import strax
import shutil


@unittest.skipIf(not straxen.HAVE_ADMIX, "Admix is not installed")
class TestRucioRemote(unittest.TestCase):
    """
    Test loading data from the rucio remote frontend
    """
    def setUp(self) -> None:
        self.run_id = '009104'
        self.staging_dir = './test_rucio_remote'

    def get_context(self, download_heavy: bool) -> strax.Context:
        os.makedirs(self.staging_dir, exist_ok=True)
        context = straxen.contexts.xenonnt_online(
            output_folder=os.path.join(self.staging_dir, 'output'),
            include_rucio_remote=True,
            download_heavy=download_heavy,
            _rucio_path=self.staging_dir,
            _raw_path=os.path.join(self.staging_dir, 'raw'),
            _database_init=False,
            _processed_path=os.path.join(self.staging_dir, 'processed'),
        )
        return context

    def tearDown(self):
        if os.path.exists(self.staging_dir):
            shutil.rmtree(self.staging_dir)

    def test_download_no_heavy(self):
        st = self.get_context(download_heavy=False)
        with self.assertRaises(strax.DataNotAvailable):
            rr = st.get_array(self.run_id, 'raw_records')
            assert False, len(rr)

    def test_download_with_heavy(self):
        st = self.get_context(download_heavy=True)
        rr = st.get_array(self.run_id, 'raw_records')
        assert len(rr)
    
    def test_download_with_heavy_and_high_level(self):
        st = self.get_context(download_heavy=True)
        pc = st.get_array(self.run_id, 'pulse_counts')
        assert len(pc)

    def check_empty_context(self, context):
        for sf in context.storage:
            assert not context._is_stored_in_sf(self.run_id, 'raw_records', sf), sf