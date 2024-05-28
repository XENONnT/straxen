import os
import re
import unittest
import strax
from strax.testutils import Records, Peaks
import straxen
import shutil
import tempfile
import pymongo
import datetime
import socket
from straxen import RunDB, mongo_uri_not_set


@unittest.skipIf(mongo_uri_not_set(), "No access to test database")
class TestRunDBFrontend(unittest.TestCase):
    """Test the saving behavior of the context with the straxen.RunDB.

    Requires write access to some pymongo server, the URI of witch is to be set
    as an environment variable under:

        TEST_MONGO_URI

    At the moment this is just an empty database but you can also use some free
    ATLAS mongo server.

    """

    _run_test = True

    @classmethod
    def setUpClass(cls) -> None:
        # Just to make sure we are running some mongo server, see test-class docstring
        cls.test_run_ids = ["0", "1"]
        cls.all_targets = ("peaks", "records")

        uri = os.environ.get("TEST_MONGO_URI")
        db_name = "test_rundb"
        cls.collection_name = "test_rundb_coll"
        client = pymongo.MongoClient(uri)
        cls.database = client[db_name]
        collection = cls.database[cls.collection_name]
        cls.path = os.path.join(tempfile.gettempdir(), "strax_data")
        # assert cls.collection_name not in cls.database.list_collection_names()

        if not straxen.utilix_is_configured():
            # Bit of an ugly hack but there is no way to get around this
            # function even though we don't need it
            straxen.rundb.utilix.rundb.xent_collection = lambda *args, **kwargs: collection

        cls.rundb_sf = straxen.RunDB(
            readonly=False,
            runid_field="number",
            new_data_path=cls.path,
            minimum_run_number=-1,
            rucio_path="./strax_test_data",
        )
        cls.rundb_sf.client = client
        cls.rundb_sf.collection = collection

        # Extra test for regexes
        class RunDBTestLocal(RunDB):
            """Change class to mathc current host too."""

            hosts = {"bla": f"{socket.getfqdn()}"}

        cls.rundb_sf_with_current_host = RunDBTestLocal(
            readonly=False,
            runid_field="number",
            new_data_path=cls.path,
            minimum_run_number=-1,
            rucio_path="./strax_test_data",
        )
        cls.rundb_sf_with_current_host.client = client
        cls.rundb_sf_with_current_host.collection = collection

        cls.st = strax.Context(
            register=[Records, Peaks],
            storage=[cls.rundb_sf],
            use_per_run_defaults=False,
            config=dict(bonus_area=0),
        )

    def setUp(self) -> None:
        for run_id in self.test_run_ids:
            self.collection.insert_one(_rundoc_format(run_id))
        assert not self.is_all_targets_stored

    def tearDown(self):
        self.database[self.collection_name].drop()
        if os.path.exists(self.path):
            print(f"rm {self.path}")
            shutil.rmtree(self.path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.database[cls.collection_name].drop()

    @property
    def collection(self):
        return self.database[self.collection_name]

    @property
    def is_all_targets_stored(self) -> bool:
        """This should always be False as one of the targets (records) is not stored in mongo."""
        return all(
            [all([self.st.is_stored(r, t) for t in self.all_targets]) for r in self.test_run_ids]
        )

    def test_finding_runs(self):
        rdb = self.rundb_sf
        col = self.database[self.collection_name]
        assert col.find_one() is not None
        query = rdb.number_query()
        assert col.find_one(query) is not None
        runs = self.st.select_runs()
        assert len(runs) == len(self.test_run_ids)

    def test_write_and_load(self):
        assert not self.is_all_targets_stored

        # Make ALL the data
        # NB: the context writes to ALL the storage frontends that are susceptible
        for t in self.all_targets:
            self.st.make(self.test_run_ids, t)

        for r in self.test_run_ids:
            print(self.st.available_for_run(r))
        assert self.is_all_targets_stored

        # Double check that we can load data from mongo even if we cannot make it
        self.st.context_config["forbid_creation_of"] = self.all_targets
        peaks = self.st.get_array(self.test_run_ids, self.all_targets[-1])
        assert len(peaks)
        runs = self.st.select_runs(available=self.all_targets)
        assert len(runs) == len(self.test_run_ids)

        # Insert a new run number and check that it's not marked as available
        self.database[self.collection_name].insert_one(_rundoc_format(3))
        self.st.runs = None  # Reset
        all_runs = self.st.select_runs()
        available_runs = self.st.select_runs(available=self.all_targets)
        assert len(available_runs) == len(self.test_run_ids)
        assert len(all_runs) == len(self.test_run_ids) + 1

    def test_lineage_changes(self):
        st = strax.Context(
            register=[Records, Peaks],
            storage=[self.rundb_sf],
            use_per_run_defaults=True,
        )
        lineages = [st.key_for(r, "peaks").lineage_hash for r in self.test_run_ids]
        assert len(set(lineages)) > 1
        with self.assertRaises(ValueError):
            # Lineage changing per run is not allowed!
            st.select_runs(available="peaks")

    def test_fuzzy(self):
        """See that fuzzy for does not work yet with the RunDB."""
        fuzzy_st = self.st.new_context(fuzzy_for=self.all_targets)
        with self.assertWarns(UserWarning):
            fuzzy_st.is_stored(self.test_run_ids[0], self.all_targets[0])
        with self.assertWarns(UserWarning):
            keys = [fuzzy_st.key_for(r, self.all_targets[0]) for r in self.test_run_ids]
            self.rundb_sf.find_several(keys, fuzzy_for=self.all_targets)

    def test_invalids(self):
        """Test a couble of invalid ways of passing arguments to the RunDB."""
        with self.assertRaises(ValueError):
            straxen.RunDB(
                runid_field="numbersdfgsd",
            )
        with self.assertRaises(ValueError):
            r = self.test_run_ids[0]
            keys = [self.st.key_for(r, t) for t in self.all_targets]
            self.rundb_sf.find_several(keys, fuzzy_for=self.all_targets)
        with self.assertRaises(strax.DataNotAvailable):
            self.rundb_sf.find(self.st.key_for("_super-run", self.all_targets[0]))
        with self.assertRaises(strax.DataNotAvailable):
            self.rundb_sf._find(
                self.st.key_for("_super-run", self.all_targets[0]),
                write=False,
                allow_incomplete=False,
                fuzzy_for=[],
                fuzzy_for_options=[],
            )

    def test_rucio_format(self):
        """Test that document retrieval works for rucio files in the RunDB."""
        rucio_id = "999999"
        target = self.all_targets[-1]
        key = self.st.key_for(rucio_id, target)
        self.assertFalse(rucio_id in self.test_run_ids)
        rd = _rundoc_format(rucio_id)
        did = straxen.key_to_rucio_did(key)
        location = None
        for host_alias, regex in self.database.userdisks.items():
            if re.match(regex, self.database.hostname):
                location = host_alias
        rd["data"] = [
            {
                "host": "rucio-catalogue",
                "status": "transferred",
                "did": did,
                "number": int(rucio_id),
                "type": target,
            }
        ]
        if location is not None:
            rd["data"][0]["location"] = location
        self.database[self.collection_name].insert_one(rd)

        # Make sure we get the backend key using the _find option
        self.assertTrue(
            self.rundb_sf_with_current_host._find(
                key,
                write=False,
                allow_incomplete=False,
                fuzzy_for=None,
                fuzzy_for_options=None,
            )[1]
            == did,
        )
        with self.assertRaises(strax.DataNotAvailable):
            # Now, this same test should fail if we have a rundb SF
            # without our host added to the regex
            self.rundb_sf._find(
                key,
                write=False,
                allow_incomplete=False,
                fuzzy_for=None,
                fuzzy_for_options=None,
            )
        with self.assertRaises(strax.DataNotAvailable):
            # Although we did insert a document, we should get a data
            # not available error as we did not actually save any data
            # on the rucio folder
            self.rundb_sf_with_current_host.find(key)
        with self.assertRaises(strax.DataNotAvailable):
            # Same, just double checking!
            self.rundb_sf.find(key)


def _rundoc_format(run_id):
    start = datetime.datetime.fromtimestamp(0) + datetime.timedelta(days=int(run_id))
    end = start + datetime.timedelta(days=1)
    doc = {
        "comments": [{"comment": "some testdoc", "date": start, "user": "master user"}],
        "data": [],
        "detectors": ["tpc"],
        "mode": "test",
        "number": int(run_id),
        "source": "none",
        "start": start,
        "end": end,
        "user": "master user",
    }
    return doc
