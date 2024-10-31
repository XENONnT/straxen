import socket
import unittest
import strax
import os
import straxen
import shutil
import json
from bson import json_util


class TestRucioLocal(unittest.TestCase):
    """Test the behavoir of how the Rucio Local frontend should behave."""

    def setUp(self) -> None:
        self.test_keys = [
            strax.DataKey(
                run_id=run_id,
                data_type="dtype",
                lineage={
                    "dtype": ["Plugin", "0.0.0", {}],
                },
            )
            for run_id in ("-1", "-2")
        ]
        self.rucio_path = "./.test_rucio"
        self.write_test_data()

    def tearDown(self) -> None:
        shutil.rmtree(self.rucio_path)

    def test_find(self):
        rucio_local = straxen.RucioLocalFrontend(path=self.rucio_path)
        find_result = rucio_local.find(self.test_keys[0])
        assert len(find_result) and find_result[0] == "RucioLocalBackend", find_result

    def test_find_several(self):
        rucio_local = straxen.RucioLocalFrontend(path=self.rucio_path)
        find_several_results = rucio_local.find_several(self.test_keys)
        assert find_several_results, find_several_results
        for find_result in find_several_results:
            assert len(find_result) and find_result[0] == "RucioLocalBackend", find_result

    def test_find_fuzzy(self):
        changed_keys = []
        rucio_local = straxen.RucioLocalFrontend(path=self.rucio_path)
        for key in self.test_keys:
            changed_key = strax.DataKey(
                run_id=key.run_id,
                data_type=key.data_type,
                lineage={
                    "dtype": ["Plugin", "1.0.0", {}],
                },
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
            metadata = {
                "writing_ended": 1,
                "chunks": [
                    {
                        "filename": f"{key.data_type}-{key.lineage_hash}-000000",
                    },
                ],
                "lineage_hash": key.lineage_hash,
                "lineage": key.lineage,
            }
            self.write_md(self.rucio_path, did, metadata)
            self.write_chunks(self.rucio_path, did, [c["filename"] for c in metadata["chunks"]])

    @staticmethod
    def write_md(rucio_path, did, content: dict):
        md_did = strax.RUN_METADATA_PATTERN % did
        md_path = straxen.storage.rucio_local.rucio_path(rucio_path, md_did)
        os.makedirs(os.path.split(md_path)[0], exist_ok=True)
        with open(md_path, mode="w") as f:
            f.write(json.dumps(content, default=json_util.default))

    @staticmethod
    def write_chunks(rucio_path, did, file_names):
        for file_name in file_names:
            file_did = did.split(":")[0] + ":" + file_name
            chunk_path = straxen.storage.rucio_local.rucio_path(rucio_path, file_did)
            os.makedirs(os.path.split(chunk_path)[0], exist_ok=True)
            with open(chunk_path, mode="w") as f:
                f.write(file_name)


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class TestBasics(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """For testing purposes, slightly alter the RucioRemoteFrontend such that we can run tests
        outside of dali too."""
        # Some non-existing keys that we will try finding in the test cases.
        cls.test_keys = [
            strax.DataKey(
                run_id=run_id,
                data_type="dtype",
                lineage={
                    "dtype": ["Plugin", "0.0.0.", {}],
                },
            )
            for run_id in ("-1", "-2")
        ]

    def test_load_context_defaults(self):
        """Don't fail immediately if we start a context due to Rucio."""
        st = straxen.contexts.xenonnt_online(
            minimum_run_number=10_000,
            maximum_run_number=10_010,
        )
        st.select_runs()

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_find_several_remote(self):
        """Let's try running a find_several with the include remote.

        This should fail but when no rucio is installed or else it shouldn't find any data.

        """
        try:
            rucio = straxen.RucioRemoteFrontend()
        except ImportError:
            pass
        else:
            found = rucio.find_several(self.test_keys)
            # We shouldn't find any of these
            assert found == [False for _ in self.test_keys]

    @unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_find_local_rundb(self):
        """Make sure that we don't find the non existing data."""
        run_db = straxen.RunDB(rucio_path="./rucio_test")
        with self.assertRaises(strax.DataNotAvailable):
            run_db.find(self.test_keys[0])

    def test_determine_rse(self):
        class DummyLocalRucio(straxen.RucioLocalFrontend):
            local_prefixes = {"any": "./any"}
            local_rses = {}

        dummy_class = DummyLocalRucio(path="./")
        assert dummy_class.determine_rse() is None

        update = {"any": socket.getfqdn()}
        dummy_class.local_rses = update
        assert dummy_class.determine_rse() == "any"

        # now the init should also work
        DummyLocalRucio.local_rses = update
        assert DummyLocalRucio().path == "./any"

        with self.assertRaises(ValueError):
            dummy_class.local_rses.update({"some other!": socket.getfqdn()})
            dummy_class.determine_rse()

    @unittest.skipIf(not straxen.utilix_is_configured(), "No DB access")
    @unittest.skipIf(
        socket.getfqdn() in straxen.RucioLocalFrontend.local_rses,
        "Testing useless frontends only works on hosts where it's not supposed to work",
    )
    def test_useless_frontend(self):
        """Test that using a rucio-local frontend on a non-RSE listed site doesn't cause issues when
        registered."""
        rucio_local = straxen.RucioLocalFrontend()
        assert rucio_local.path is None
        with self.assertRaises(strax.DataNotAvailable):
            rucio_local.find(self.test_keys[0])
        # Do a small test that we did not break everything by having a useless fontend
        st = straxen.test_utils.nt_test_context(
            minimum_run_number=10_000,
            maximum_run_number=10_005,
            include_rucio_local=True,
            keep_default_storage=True,
        )
        self.assertTrue(
            str(rucio_local.__class__) in [str(sf.__class__) for sf in st.storage],
            "Rucio local did not get registered??",
        )
        st.select_runs()
