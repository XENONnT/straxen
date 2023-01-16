import os
import json
import tempfile
import pandas as pd
import strax
import straxen
import fsspec
import utilix.rundb
from straxen.test_utils import nt_test_context, nt_test_run_id
import unittest
import pickle
import random
import numpy as np
from datetime import datetime


class DummyObject:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


@straxen.URLConfig.register('random')
def generate_random(_):
    return random.random()


@straxen.URLConfig.register('unpicklable')
def return_lamba(_):
    return lambda x: x


@straxen.URLConfig.register('large-array')
def large_array(_):
    return np.ones(1_000_000).tolist()


@straxen.URLConfig.register('object-list')
def object_list(length):
    length = int(length)
    return [DummyObject(a=i, b=i+1) for i in range(length)]


@straxen.URLConfig.preprocessor
def formatter(config, **kwargs):
    if not isinstance(config, str):
        return config
    try:
        config = config.format(**kwargs)
    except KeyError:
        pass
    return config


GLOBAL_VERSIONS = {
    'global_v1': {
        'test_config': 'v0'
    }
}


@straxen.URLConfig.preprocessor
def replace_global_version(config, name=None, **kwargs):
    if name is None:
        return

    if not isinstance(config, str):
        return config

    if not straxen.URLConfig.SCHEME_SEP in config:
        return config

    version = straxen.URLConfig.kwarg_from_url(config, 'version')

    if version is None:
        return config

    if version.startswith('global_') and version in GLOBAL_VERSIONS:
        version = GLOBAL_VERSIONS[version].get(name, version)
        config = straxen.URLConfig.format_url_kwargs(config, version=version)
    return config


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
    def test_itp_dict(self, ab_value=20, cd_value=21, dump_as='json'):
        """
        Test that we are getting ~the same value from interpolating at the central date in a dict

        :param ab_value, cd_value: some values to test against
        :param dump_as: Write as csv or as json file
        """
        central_datetime = utilix.rundb.xent_collection().find_one(
            {'number': int(nt_test_run_id)},
            projection={'start': 1}
        ).get('start', 'QUERY FAILED!')
        fake_file = {'time': [datetime(2000, 1, 1).timestamp() * 1e9,
                              central_datetime.timestamp() * 1e9,
                              datetime(2040, 1, 1).timestamp() * 1e9],
                     'ab': [10, ab_value, 30],
                     'cd': [11, cd_value, 31]
                         }

        temp_dir = tempfile.TemporaryDirectory()

        if dump_as == 'json':
            fake_file_name = os.path.join(temp_dir.name, 'test_seg.json')
            with open(fake_file_name, 'w') as f:
                json.dump(fake_file, f)
        elif dump_as == 'csv':
            # This example also works well with dataframes!
            fake_file_name = os.path.join(temp_dir.name, 'test_seg.csv')
            pd.DataFrame(fake_file).to_csv(fake_file_name)
        else:
            raise ValueError

        self.st.set_config({'test_config':f'itp_dict://'
                                          f'resource://'
                                          f'{fake_file_name}'
                                          f'?run_id=plugin.run_id'
                                          f'&fmt={dump_as}'
                                          f'&itp_keys=ab,cd'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertIsInstance(p.test_config, dict)
        assert np.isclose(p.test_config['ab'], ab_value, rtol=1e-3)
        assert np.isclose(p.test_config['cd'], cd_value, rtol=1e-3)
        temp_dir.cleanup()

    def test_itp_dict_csv(self):
        self.test_itp_dict(dump_as='csv')

    def test_rekey(self):
        original_dict = {'a': 1, 'b': 2, 'c': 3}
        check_dict = {'anew': 1, 'bnew': 2, 'cnew': 3}

        temp_dir = tempfile.TemporaryDirectory()

        fake_file_name = os.path.join(temp_dir.name, 'test_dict.json')
        with open(fake_file_name, 'w') as f:
            json.dump(original_dict, f)

        self.st.set_config({'test_config': f'rekey_dict://resource://{fake_file_name}?'
                                           f'fmt=json&replace_keys=a,b,c'
                                           f'&with_keys=anew,bnew,cnew'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, check_dict)
        temp_dir.cleanup()

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

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test CMT.")
    def test_dry_evaluation(self):
        """
        Check that running a dry evaluation can be done outside of the
        context of a URL config and yield the same result.
        """
        plugin_url = 'cmt://electron_drift_velocity?run_id=plugin.run_id&version=v3'
        self.st.set_config({'test_config': plugin_url})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        correct_val = p.test_config

        # We can also get it from one of these methods
        dry_val1 = straxen.URLConfig.evaluate_dry(
            f'cmt://electron_drift_velocity?run_id={nt_test_run_id}&version=v3')
        dry_val2 = straxen.URLConfig.evaluate_dry(
            f'cmt://electron_drift_velocity?version=v3', run_id=nt_test_run_id)

        # All methods should yield the same
        assert correct_val == dry_val1 == dry_val2

        # However dry-evaluation does NOT allow loading the plugin.run_id
        # as in the plugin_url and should complain about that
        with self.assertRaises(ValueError):
            straxen.URLConfig.evaluate_dry(plugin_url)

    def test_objects_to_dict(self):
        n = 3
        self.st.set_config({'test_config': f'objects-to-dict://object-list://{n}?key_attr=a&value_attr=b'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, {i:i+1 for i in range(n)})

    def test_list_to_array(self):
        n = 3
        self.st.set_config({'test_config': f'list-to-array://object-list://{n}'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertIsInstance(p.test_config, np.ndarray)

    def test_format_preprocessor(self):
        self.st.set_config({'test_config': '{name}:{run_id}'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, f'test_config:{nt_test_run_id}')
        self.assertEqual(p.test_config, p.config['test_config'])

    def test_global_version_preprocessor(self):
        self.st.set_config({'test_config': 'fake://url?version=global_v1'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 'fake://url?version=v0')
    
    def test_global_version_not_changed(self):
        """
          - if no global version is matched, the url version should not be changed
          - if config is not matched, the url version should not be changed
        """
        assert 'global_v2' not in GLOBAL_VERSIONS
        self.st.set_config({'test_config': 'fake://url?version=global_v2'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 'fake://url?version=global_v2')
        
        self.st.set_config({'test_config_new': 'fake://url?version=global_v1'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config_new, 'fake://url?version=global_v1')
