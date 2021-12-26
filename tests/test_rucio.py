import straxen
import unittest
import strax
import socket


class TestBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """
        For testing purposes, slightly alter the RucioFrontend such that
         we can run tests outside of dali too
        """
        if not straxen.utilix_is_configured():
            return
        if 'rcc' not in socket.getfqdn():
            # If we are not on RCC, for testing, add some dummy site
            straxen.RucioFrontend.local_rses = {'UC_DALI_USERDISK': r'.rcc.',
                                                'test_rucio': f'{socket.getfqdn()}'}
            straxen.RucioFrontend.get_rse_prefix = lambda *x: 'test_rucio'

        # Some non-existing keys that we will try finding in the test cases.
        cls.test_keys = [
            strax.DataKey(run_id=run_id,
                          data_type='dtype',
                          lineage={'dtype': ['Plugin', '0.0.0.', {}], }
                          )
            for run_id in ('-1', '-2')
        ]

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_load_context_defaults(self):
        """Don't fail immediately if we start a context due to Rucio"""
        st = straxen.contexts.xenonnt_online(_minimum_run_number=10_000,
                                             _maximum_run_number=10_010,
                                             )
        st.select_runs()

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_find_local(self):
        """Make sure that we don't find the non existing data"""
        rucio = straxen.RucioFrontend(include_remote=False,)
        self.assertRaises(strax.DataNotAvailable,
                          rucio.find,
                          self.test_keys[0]
                          )

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_find_several_local(self):
        """Let's try finding some keys (won't be available)"""
        rucio = straxen.RucioFrontend(include_remote=False,)
        print(rucio)
        found = rucio.find_several(self.test_keys)
        # We shouldn't find any of these
        assert found == [False for _ in self.test_keys]

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_find_several_remote(self):
        """
        Let's try running a find_several with the include remote.
        This should fail but when no rucio is installed or else it
        shouldn't find any data.
        """
        try:
            rucio = straxen.RucioFrontend(include_remote=True,)
        except ImportError:
            pass
        else:
            found = rucio.find_several(self.test_keys)
            # We shouldn't find any of these
            assert found == [False for _ in self.test_keys]

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_find_local(self):
        """Make sure that we don't find the non existing data"""
        run_db = straxen.RunDB(rucio_path='./rucio_test')
        with self.assertRaises(strax.DataNotAvailable):
            run_db.find(self.test_keys[0])
