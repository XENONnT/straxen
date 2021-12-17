import json
import strax
import straxen
import fsspec
from straxen.test_utils import nt_test_context, nt_test_run_id
import unittest


class TestPlugin(strax.Plugin):
    depends_on = ()
    dtype = strax.time_fields
    provides = ('test_data',)
    test_config = straxen.URLConfig(default=42,)

    def compute(self):
        pass

@straxen.URLConfig.register_preprocessor('take')
def increment_take(url, arg, take, increment_take=False):
    if increment_take:
        return url+f"?take={take+1}"
    return url

class TestURLConfig(unittest.TestCase):
    def setUp(self):
        st = nt_test_context()
        st.register(TestPlugin)
        self.st = st

    def test_default(self):
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 42)

    def test_literal(self):
        self.st.set_config({'test_config': 666})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, 666)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test CMT.")
    def test_cmt_protocol(self):
        self.st.set_config({'test_config': 'cmt://elife?version=v1&run_id=plugin.run_id'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertTrue(abs(p.test_config-219203.49884000001)<1e-2)

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test CMT.")
    def test_cmt_preprocessor(self):
        self.st.set_config({'test_config': 'cmt://elife?version=global_v1'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.config['test_config'], 'cmt://elife?version=v1')

    def test_json_protocol(self):
        self.st.set_config({'test_config': 'json://[1,2,3]'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        self.assertEqual(p.test_config, [1,2,3])

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

    def test_preprocessor(self):
        # The take preprocessor will increment the take parameter by one in increment_take is True
        self.st.set_config({'test_config': 'take://json://[1,2,3]?take=0&increment_take=True'})
        p = self.st.get_single_plugin(nt_test_run_id, 'test_data')
        
        # verify that the second item was taken i.e. the preprocessor
        # has incremented the value of take from 0 to 1
        self.assertEqual(p.test_config, 2)

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

    def test_print_protocol_desc(self):
        straxen.URLConfig.print_protocols()
