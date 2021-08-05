import straxen
import unittest
import strax
import socket


class TestBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not straxen.utilix_is_configured():
            return
        if 'rcc' not in socket.getfqdn():
            # If we are not on RCC, for testing, add some dummy site
            straxen.RucioFrontend.local_rses = {'UC_DALI_USERDISK': r'.rcc.',
                                                'test_rucio': f'{socket.getfqdn()}'}
            straxen.RucioFrontend.get_rse_prefix = lambda *x: 'test_rucio'

        cls.test_keys = [
            strax.DataKey(run_id=run_id,
                          data_type='dtype',
                          lineage={'dtype': ['Plugin', '0.0.0.', {}],}
                          )
            for run_id in ('-1', '-2')
        ]

    @classmethod
    def tearDownClass(cls):
        pass

    def test_load_context_defaults(self):
        if not straxen.utilix_is_configured():
            return
        st = straxen.contexts.xenonnt_online(_minimum_run_number=10_000,
                                             _maximum_run_number=10_010,
                                             )
        st.select_runs()

    def test_find_local(self):
        if not straxen.utilix_is_configured():
            return
        rucio = straxen.RucioFrontend(
            include_remote=False,
        )
        self.assertRaises(strax.DataNotAvailable,
                          rucio.find,
                          self.test_keys[0]
                          )


    def test_find_several_local(self):
        if not straxen.utilix_is_configured():
            return
        rucio = straxen.RucioFrontend(
            include_remote=False,
        )
        rucio.find_several(self.test_keys)
        print(rucio)

    def test_find_several_remote(self):
        if not straxen.utilix_is_configured():
            return
        try:
            rucio = straxen.RucioFrontend(
                include_remote=True,
            )
        except ImportError:
            pass
        else:
            rucio.find_several(self.test_keys)
