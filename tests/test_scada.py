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

        # Test ffill option:
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

        print('Testing downsampling and averaging option:')
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
            assert np.isclose(np.mean(df_all[i:i + 2]), df['SomeParameter'][ind]), 'Averaging is incorrect.'

        # Testing lab query type:
        df = self.sc.get_scada_values(parameters,
                                      start=self.start,
                                      end=self.end,
                                      query_type_lab=True,)

        assert np.all(df['SomeParameter'] // 1 == -96), 'Not all values are correct for query type lab.'

        
    def test_average_scada(self):
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
