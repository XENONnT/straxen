import os
import json
import strax
import straxen
import fsspec
from straxen.test_utils import nt_test_context, nt_test_run_id
import unittest
import pickle
import random
import numpy as np
from datetime import datetime


@straxen.URLConfig.register('random')
def generate_random(_):
    return random.random()


@straxen.URLConfig.register('unpicklable')
def return_lamba(_):
    return lambda x: x


@straxen.URLConfig.register('large-array')
def large_array(_):
    return np.ones(1_000_000).tolist()


class ExamplePlugin(strax.Plugin):
    depends_on = ()
    dtype = strax.time_fields
    provides = ('test_data',)
    test_config = straxen.URLConfig(default=42,)
    cached_config = straxen.URLConfig(default=666, cache=1)

    def compute(self):
        pass


class TestURLConfig(unittest.TestCase):
    def setUp(self):
        st = nt_test_context()
        st.register(ExamplePlugin)
        self.st = st

    def test_default(self):
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 42)

    def test_literal(self):
        self.st.set_config({'test_config': 666})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 666)

    def test_leading_zero_int(self):
        self.st.set_config({'test_config': 'format://{value}?value=0666'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, '0666')
        self.assertIsInstance(p.test_config, str)


    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test CMT.")
    def test_cmt_protocol(self):
        self.st.set_config({'test_config': 'cmt://elife?version=v1&run_id=plugin.run_id'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertTrue(abs(p.test_config-219203.49884000001) < 1e-2)

    def test_json_protocol(self):
        self.st.set_config({'test_config': 'json://[1,2,3]'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, [1, 2, 3])

    def test_format_protocol(self):
        self.st.set_config({'test_config': 'format://{run_id}?run_id=plugin.run_id'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, nt_test_run_id)

    def test_fsspec_protocol(self):
        with fsspec.open('memory://test_file.json', mode='w') as f:
            json.dump({"value": 999}, f)
        self.st.set_config(
            {'test_config': 'take://json://fsspec://memory://test_file.json?take=value'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 999)

    def test_chained(self):
        self.st.set_config({'test_config': 'take://json://[1,2,3]?take=0'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 1)

    def test_take_nested(self):
        self.st.set_config({'test_config': 'take://json://{"a":[1,2,3]}?take=a&take=0'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 1)

    @unittest.skipIf(not straxen.utilix_is_configured(),
                     "No db access, cannot test!")
    def test_bodedga_get(self):
        """Just a didactic example"""
        self.st.set_config({
            'test_config':
                'take://'
                'resource://'
                'XENONnT_numbers.json'
                '?fmt=json'
                '&take=g1'
                '&take=v2'
                '&take=value'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        # Either g1 is 0, bodega changed or someone broke URLConfigs
        self.assertTrue(p.test_config)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test CMT.")
    def test_seg_file(self):
        fake_file = {'times': [datetime(2000, 1, 1).timestamp() * 1e9,
                               datetime(2020, 1, 1).timestamp() * 1e9,
                               datetime(2040, 1, 1).timestamp() * 1e9],
                         'gain_ab': [10, 20, 30],
                         'gain_cd': [11, 21, 31]
                         }
        fake_file_name = 'test_seg.json'

        with open(fake_file_name, 'w') as f:
            json.dump(fake_file, f)

        self.st.set_config({'test_config': f'seg_file://resource://{fake_file_name}?run_id=plugin.run_id&fmt=json'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertIsInstance(p.test_config, dict)
        os.remove(fake_file_name)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test CMT.")
    def test_ee_file(self):
        fake_file = {'timestamps': [datetime(2000, 1, 1).timestamp() * 1e9,
                                    datetime(2020, 1, 1).timestamp() * 1e9,
                                    datetime(2040, 1, 1).timestamp() * 1e9],
                     'ee_ab': [0.1, 0.2, 0.3],
                     'ee_cd': [0.4, 0.5, 0.6]
                     }
        fake_file_name = 'test_ee.json'

        with open(fake_file_name, 'w') as f:
            json.dump(fake_file, f)

        self.st.set_config({'test_config': f'ee_file://resource://{fake_file_name}?run_id=plugin.run_id&fmt=json'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertIsInstance(p.test_config, dict)

    def test_print_protocol_desc(self):
        straxen.URLConfig.print_protocols()

    def test_cache(self):
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')

        # sanity check that default value is not affected
        self.assertEqual(p.cached_config, 666)
        self.st.set_config({'cached_config': 'random://abc'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')

        # value is randomly generated when accessed so if
        # its equal when we access it again, its coming from the cache
        cached_value = p.cached_config
        self.assertEqual(cached_value, p.cached_config)

        # now change the config to which will generate a new number
        self.st.set_config({'cached_config': 'random://dfg'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')

        # sanity check that the new value is still consistent i.e. cached
        self.assertEqual(p.cached_config, p.cached_config)

        # test if previous value is evicted, since cache size is 1
        self.assertNotEqual(cached_value, p.cached_config)

        # verify pickalibility of objects in cache dont affect plugin pickalibility
        self.st.set_config({'cached_config': 'unpicklable://dfg'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        with self.assertRaises(AttributeError):
            pickle.dumps(p.cached_config)
        pickle.dumps(p)

    def test_cache_size(self):
        '''test the cache helper functions
        '''
        # make sure the value has a detectable size
        self.st.set_config({'cached_config': 'large-array://dfg'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')

        # fetch the value so its stored in the cache
        value = p.cached_config

        # cache should now have finite size
        self.assertGreater(straxen.config_cache_size_mb(), 0.0)

        # test if clearing cache works as expected
        straxen.clear_config_caches()
        self.assertEqual(straxen.config_cache_size_mb(), 0.0)

    def test_filter_kwargs(self):
        all_kwargs = dict(a=1, b=2, c=3)

        # test a function that takes only a seubset of the kwargs
        def func1(a=None, b=None):
            return
        
        filtered1 = straxen.filter_kwargs(func1, all_kwargs)
        self.assertEqual(filtered1, dict(a=1, b=2))
        func1(**filtered1)


        # test function that accepts wildcard kwargs
        def func2(**kwargs):
            return
        filtered2 = straxen.filter_kwargs(func2, all_kwargs)
        self.assertEqual(filtered2, all_kwargs)
        func2(**filtered2)
