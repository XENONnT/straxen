import unittest
import strax
import os
import straxen
import shutil
import json
from bson import json_util


class TestRucioLocal(unittest.TestCase):
    def setUp(self) -> None:
        self.test_keys = [
            strax.DataKey(run_id=run_id,
                          data_type='dtype',
                          lineage={'dtype': ['Plugin', '0.0.0', {}], }
                          )
            for run_id in ('-1', '-2')
        ]
        self.rucio_path = './.test_rucio'
        self.write_test_data()

    def tearDown(self) -> None:
        shutil.rmtree(self.rucio_path)

    # def test_find(self):
    #     rucio_local = straxen.RucioLocalFrontend(rucio_dir = self.rucio_path)
    #     find_result = rucio_local.find(self.test_keys[0])
    #     assert len(find_result) and find_result[0] == 'RucioLocalBackend', find_result
    #
    # def test_find_several(self):
    #     rucio_local = straxen.RucioLocalFrontend(rucio_dir=self.rucio_path)
    #     find_several_results = rucio_local.find_several(self.test_keys)
    #     assert find_several_results, find_several_results
    #     for find_result in find_several_results:
    #         assert len(find_result) and find_result[0] == 'RucioLocalBackend', find_result

    def test_find_fuzzy(self):
        changed_keys = []
        rucio_local = straxen.RucioLocalFrontend(rucio_dir=self.rucio_path)
        for key in self.test_keys:
            changed_key = strax.DataKey(
                run_id=key.run_id,
                data_type=key.data_type,
                lineage={'dtype': ['Plugin', '1.0.0', {}], }
            )
            changed_keys += [changed_key]

            # We shouldn't find this data
            with self.assertRaises(strax.DataNotAvailable):
                rucio_local.find(changed_key)

        # Also find several shouldn't work
        find_several_keys = rucio_local.find_several(changed_keys)
        self.assertFalse(any(find_several_keys))

        # Now test fuzzy
        with self.assertWarns(UserWarning):
            find_several_keys_fuzzy = rucio_local.find_several(
                changed_keys,
                fuzzy_for=changed_keys[0].data_type,
            )
        self.assertTrue(all(find_several_keys_fuzzy))

    def write_test_data(self):
        os.makedirs(self.rucio_path, exist_ok=True)
        for key in self.test_keys:
            did = straxen.key_to_rucio_did(key)
            md_did = f'{did}-metadata.json'
            path = straxen.storage.rucio_local.rucio_path(self.rucio_path, md_did)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            self.write_md(path, {'writing_ended': 1,
                                 'chunks': [],
                                 "lineage_hash": key.lineage_hash,
                                 "lineage": key.lineage,
                                 })

    @staticmethod
    def write_md(path, content: dict):
        with open(path, mode='w') as f:
            f.write(json.dumps(content, default=json_util.default))

