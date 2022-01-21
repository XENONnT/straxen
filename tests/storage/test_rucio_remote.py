import unittest
import straxen
import os
import strax
import shutil


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

    @unittest.skipIf(not straxen.HAVE_ADMIX, "Admix is not installed")
    def test_download_no_heavy(self):
        st = self.get_context(download_heavy=False)
        with self.assertRaises(strax.DataNotAvailable):
            rr = st.get_array(self.run_id, 'raw_records')
            assert False, len(rr)

    @unittest.skipIf(not straxen.HAVE_ADMIX, "Admix is not installed")
    def test_download_with_heavy(self):
        st = self.get_context(download_heavy=True)
        rr = st.get_array(self.run_id, 'raw_records')
        assert len(rr)

    @unittest.skipIf(not straxen.HAVE_ADMIX, "Admix is not installed")
    def test_download_with_heavy_and_high_level(self):
        st = self.get_context(download_heavy=True)
        pc = st.get_array(self.run_id, 'pulse_counts')
        assert len(pc)

    @unittest.skipIf(not straxen.HAVE_ADMIX, "Admix is not installed")
    def check_empty_context(self, context):
        for sf in context.storage:
            assert not context._is_stored_in_sf(self.run_id, 'raw_records', sf), sf

    def test_did_to_dirname(self):
        did = 'xnt_038697:raw_records_aqmon-rfzvpzj4mf'
        assert 'xnt_' not in straxen.rucio_remote.did_to_dirname(did)
        with self.assertRaises(RuntimeError):
            straxen.rucio_remote.did_to_dirname('a-b-c')
