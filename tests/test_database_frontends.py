import unittest
import strax
from strax.testutils import Records, Peaks
import straxen
import os
import shutil
import tempfile
import pymongo
import datetime


def mongo_uri_not_set():
    return 'TEST_MONGO_URI' not in os.environ


class TestRunDBFrontend(unittest.TestCase):
    """
    Test the saving behavior of the context with the straxen.RunDB

    Requires write access to some pymongo server, the URI of witch is to be set
    as an environment variable under:

        TEST_MONGO_URI

    At the moment this is just an empty database but you can also use some free
    ATLAS mongo server.
    """
    _run_test = True

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        if 'TEST_MONGO_URI' not in os.environ:
            return
        self.test_run_ids = ['0', '1']
        self.all_targets = ('peaks', 'records')

        uri = os.environ.get('TEST_MONGO_URI')
        db_name = 'test_rundb'
        self.collection_name = 'test_rundb_coll'
        client = pymongo.MongoClient(uri)
        self.database = client[db_name]
        collection = self.database[self.collection_name]
        self.path = os.path.join(tempfile.gettempdir(), 'strax_data')
        assert self.collection_name not in self.database.list_collection_names()

        if not straxen.utilix_is_configured():
            # Bit of an ugly hack but there is no way to get around this
            # function even though we don't need it
            straxen.rundb.utilix.rundb.xent_collection = lambda *args, **kwargs: collection

        self.rundb_sf = straxen.RunDB(readonly=False,
                                      runid_field='number',
                                      new_data_path=self.path,
                                      minimum_run_number=-1,
                                      )
        self.rundb_sf.client = client
        self.rundb_sf.collection = collection

        self.st = strax.Context(register=[Records, Peaks],
                                storage=[self.rundb_sf],
                                use_per_run_defaults=False,
                                config=dict(bonus_area=0),
                                )
        for run_id in self.test_run_ids:
            collection.insert_one(_rundoc_format(run_id))
        assert not self.is_all_targets_stored

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    def tearDown(self):
        self.database[self.collection_name].drop()
        if os.path.exists(self.path):
            print(f'rm {self.path}')
            shutil.rmtree(self.path)

    @property
    def is_all_targets_stored(self) -> bool:
        """This should always be False as one of the targets (records) is not stored in mongo"""
        return all([all(
            [self.st.is_stored(r, t) for t in self.all_targets])
            for r in self.test_run_ids])

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    def test_finding_runs(self):
        rdb = self.rundb_sf
        col = self.database[self.collection_name]
        assert col.find_one() is not None
        query = rdb.number_query()
        assert col.find_one(query) is not None
        runs = self.st.select_runs()
        assert len(runs) == len(self.test_run_ids)

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
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
        self.st.context_config['forbid_creation_of'] = self.all_targets
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

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    def test_lineage_changes(self):
        st = strax.Context(register=[Records, Peaks],
                           storage=[self.rundb_sf],
                           use_per_run_defaults=True,
                           )
        lineages = [st.key_for(r, 'peaks').lineage_hash for r in self.test_run_ids]
        assert len(set(lineages)) > 1
        with self.assertRaises(ValueError):
            # Lineage changing per run is not allowed!
            st.select_runs(available='peaks')


def _rundoc_format(run_id):
    start = datetime.datetime.fromtimestamp(0) + datetime.timedelta(days=int(run_id))
    end = start + datetime.timedelta(days=1)
    doc = {
        'comments': [{'comment': 'some testdoc',
                      'date': start,
                      'user': 'master user'}],
        'data': [],
        'detectors': ['tpc'],

        'mode': 'test',
        'number': int(run_id),
        'source': 'none',
        'start': start,
        'end': end,
        'user': 'master user'}
    return doc
