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
        self.staging_dir = './.test_rucio_remote'

    def get_context(self, download_heavy: bool) -> strax.Context:
        os.makedirs(self.staging_dir, exist_ok=True)
        context = straxen.contexts.xenonnt_online(
            include_rucio_remote=True,
            download_heavy=download_heavy,
            _raw_path=os.path.join(self.staging_dir, 'raw'),
            _database_init=False,
            _processed_path=os.path.join(self.staging_dir, 'processed'),
        )
        return context

    def tearDown(self):
        shutil.rmtree(self.staging_dir)

    def test_download_no_heavy(self):
        st = self.get_context(download_heavy=False)
        with self.assertRaises(strax.DataNotAvailable):
            st.get_array(self.run_id, 'raw_records')

    def test_download_with_heavy(self):
        st = self.get_context(download_heavy=True)
        rr = st.get_array(self.run_id, 'raw_records')
        assert len(rr)
