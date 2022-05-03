@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class TestS2LocalMinimum(unittest.TestCase):
    def setUp(self):
        st = straxen.test_utils.nt_test_context()
        st.register(straxen.local_minimum_info.LocalMinimumInfo)
        self.target = 'event_local_min_info'
        self.run_id = straxen.test_utils.nt_test_run_id
        self.st = st
        
        #A fake peak to test the functions on
        self.test_time = np.arange(200)
        self.std1, self.std2 = 15, 20
        
        #peak 1: relatively narrow peaks, separated closely
        #peak 2: relatively wide peaks, separated closely
        #peak 3: relatively wide peaks, separated far
        self.test_peak_1 = np.exp(-(self.test_time-75)**2/(2*std1**2))+np.exp(-(self.test_time-125)**2/(2*std1**2))
        self.test_peak_2 = np.exp(-(self.test_time-75)**2/(2*std2**2))+np.exp(-(self.test_time-125)**2/(2*std2**2))
        self.test_peak_3 = np.exp(-(self.test_time-50)**2/(2*std2**2))+np.exp(-(self.test_time-150)**2/(2*std2**2))
    
    def test_identify_local_extrema(self):
        
        """
        Tests whether the local extrema are identified properly
        """
        
        max1, min1 = straxen.local_minimum_info.identify_local_extrema(self.test_peak_1)
        max2, min2 = straxen.local_minimum_info.identify_local_extrema(self.test_peak_2)
        
        assert np.all(min1 == min2)
        assert max2-min2 == 1
        assert max1-min1 == 1
        assert min1[0]==100
        assert len(max1) == 2
        assert len(max2) == 2
    
    def test_full_gap_percent_valley(self):
        
        """
        Tests if the gaps and valleys are identified properly
        """
        
        max1, min1 = straxen.local_minimum_info.identify_local_extrema(self.test_peak_1)
        max2, min2 = straxen.local_minimum_info.identify_local_extrema(self.test_peak_2)
        max3, min3 = straxen.local_minimum_info.identify_local_extrema(self.test_peak_3)
        
        valley_gap1, valley1 = straxen.local_minimum_info.full_gap_percent_valley(test_peak_1, max1, min1, 0.9, 1)
        valley_gap2, valley2 = straxen.local_minimum_info.full_gap_percent_valley(test_peak_2, max2, min2, 0.9, 1)
        valley_gap3, valley3 = straxen.local_minimum_info.full_gap_percent_valley(test_peak_3, max3, min3, 0.9, 1)
        
        #peak 2 should have the shortest gap and the shortest valley
        #peak 3 should have the deepest valley
        #peak 3 should have the widest gap
        
        assert (valley_gap2<valley_gap1)&(valley_gap2<valley_gap3)
        assert (valley2<valley1)&(valley2<valley3)
        assert (valley3>valley1)&(valley3>valley2)
        assert (valley_gap3>valley_gap1)&(valley_gap3>valley_gap2)
        
        
