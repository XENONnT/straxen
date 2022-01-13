import warnings
import pytz
import numpy as np
import straxen
import unittest
import requests


class SCInterfaceTest(unittest.TestCase):
    def setUp(self):
        self.resources_available()
        # Simple query test:
        # Query 5 s of data:
        self.start = 1609682275000000000
        # Add micro-second to check if query does not fail if inquery precsion > SC precision
        self.start += 10**6
        self.end = self.start + 5*10**9

    def test_wrong_querries(self):
        parameters = {'SomeParameter': 'XE1T.CTPC.Board06.Chan011.VMon'}

        with self.assertRaises(ValueError):
            # Runid but no context
            df = self.sc.get_scada_values(parameters,
                                          run_id='1',
                                          every_nth_value=1,
                                          query_type_lab=False, )

        with self.assertRaises(ValueError):
            # No time range specified
            df = self.sc.get_scada_values(parameters,
                                          every_nth_value=1,
                                          query_type_lab=False, )

        with self.assertRaises(ValueError):
            # Start larger end
            df = self.sc.get_scada_values(parameters,
                                          start=2,
                                          end=1,
                                          every_nth_value=1,
                                          query_type_lab=False, )

        with self.assertRaises(ValueError):
            # Start and/or end not in ns unix time
            df = self.sc.get_scada_values(parameters,
                                          start=1,
                                          end=2,
                                          every_nth_value=1,
                                          query_type_lab=False, )

    def test_pmt_names(self):
        """
        Tests different query options for pmt list.
        """
        pmts_dict = self.sc.find_pmt_names(pmts=12, current=True)
        assert 'PMT12_HV' in pmts_dict.keys()
        assert 'PMT12_I' in pmts_dict.keys()
        assert pmts_dict['PMT12_HV'] == 'XE1T.CTPC.BOARD04.CHAN003.VMON'

        pmts_dict = self.sc.find_pmt_names(pmts=(12, 13))
        assert 'PMT12_HV' in pmts_dict.keys()
        assert 'PMT13_HV' in pmts_dict.keys()

        with self.assertRaises(ValueError):
            self.sc.find_pmt_names(pmts=12, current=False, hv=False)

    def test_token_expires(self):
        self.sc.token_expires_in()

    def test_convert_timezone(self):
        parameters = {'SomeParameter': 'XE1T.CTPC.Board06.Chan011.VMon'}
        df = self.sc.get_scada_values(parameters,
                                      start=self.start,
                                      end=self.end,
                                      every_nth_value=1,
                                      query_type_lab=False, )

        df_strax = straxen.convert_time_zone(df, tz='strax')
        assert df_strax.index.dtype.type is np.int64

        df_etc = straxen.convert_time_zone(df, tz='Etc/GMT+0')
        assert df_etc.index.dtype.tz is pytz.timezone('Etc/GMT+0')

    def test_query_sc_values(self):
        """
        Unity test for the SCADAInterface. Query a fixed range and check if 
        return is correct.
        """
        print('Testing SCADAInterface')
        parameters = {'SomeParameter': 'XE1T.CTPC.Board06.Chan011.VMon'}
        df = self.sc.get_scada_values(parameters,
                                      start=self.start,
                                      end=self.end,
                                      every_nth_value=1,
                                      query_type_lab=False, )

        assert df['SomeParameter'][0] // 1 == 1253, 'First values returned is not corrrect.'
        assert np.all(np.isnan(df['SomeParameter'][1:])), 'Subsequent values are not correct.'

        print('Testing forwardfill option:')
        parameters = {'SomeParameter': 'XE1T.CRY_FCV104FMON.PI'}
        df = self.sc.get_scada_values(parameters,
                                      start=self.start,
                                      end=self.end,
                                      fill_gaps='forwardfill',
                                      every_nth_value=1,
                                      query_type_lab=False,)
        assert np.all(np.isclose(df[:4], 2.079859)), 'First four values deviate from queried values.'
        assert np.all(np.isclose(df[4:], 2.117820)), 'Last two values deviate from queried values.'
        print('Testing interpolation option:')
        self.sc.get_scada_values(parameters,
                                 start=self.start,
                                 end=self.end,
                                 fill_gaps='interpolation',
                                 every_nth_value=1,
                                 query_type_lab=False,)

        print('Testing down sampling and averaging option:')
        parameters = {'SomeParameter': 'XE1T.CRY_TE101_TCRYOBOTT_AI.PI'}
        df_all = self.sc.get_scada_values(parameters,
                                          start=self.start,
                                          end=self.end,
                                          fill_gaps='forwardfill',
                                          every_nth_value=1,
                                          query_type_lab=False, )

        df = self.sc.get_scada_values(parameters,
                                      start=self.start,
                                      end=self.end,
                                      fill_gaps='forwardfill',
                                      down_sampling=True,
                                      every_nth_value=2,
                                      query_type_lab=False,)

        assert np.all(df_all[::2] == df), 'Downsampling did not return the correct values.'

        df = self.sc.get_scada_values(parameters,
                                      start=self.start,
                                      end=self.end,
                                      fill_gaps='forwardfill',
                                      every_nth_value=2,
                                      query_type_lab=False,)

        # Compare average for each second value:
        for ind, i in enumerate([0, 2, 4]):
            is_correct = np.isclose(np.mean(df_all[i:i + 2]), df['SomeParameter'][ind])
            assert is_correct, 'Averaging is incorrect.'

        # Testing lab query type:
        df = self.sc.get_scada_values(parameters,
                                      start=self.start,
                                      end=self.end,
                                      query_type_lab=True,)
        is_sorrect = np.all(df['SomeParameter'] // 1 == -96)
        assert is_sorrect, 'Not all values are correct for query type lab.'

    @staticmethod
    def test_average_scada():
        t = np.arange(0, 100, 10)
        t_t, t_a = straxen.scada._average_scada(t / 1e9, t, 1)
        assert len(t_a) == len(t), 'Scada deleted some of my 10 datapoints!'

    def resources_available(self):
        """
        Exception to skip test if external requirements are not met. Otherwise define 
        Scada interface as self.sc.
        """
        if not straxen.utilix_is_configured('scada','scdata_url',):
            self.skipTest("Cannot test scada since we have no access to xenon secrets.)")

        try:
            self.sc = straxen.SCADAInterface(use_progress_bar=False)
            self.sc.get_new_token()
        except requests.exceptions.SSLError:
            self.skipTest("Cannot reach database since HTTPs certifcate expired.")
