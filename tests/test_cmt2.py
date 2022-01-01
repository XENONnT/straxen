
import pydantic
import unittest
import straxen
import os
import pymongo
from hypothesis import example, given, strategies as st


class SimpleCorrection(straxen.BaseCorrectionSchema):
    name = 'simple_correction'
    index = straxen.IntegerIndex(name='version')
    value: pydantic.confloat(gt=0, lt=1e6)
    
class SomeTimeIntervalCorrection(straxen.TimeIntervalCorrection):
    name = 'time_interval_correction'
    value: int

class SomeSampledCorrection(straxen.TimeSampledCorrection):
    name = 'sampled_correction'
    value: int


def mongo_uri_not_set():
    return 'TEST_MONGO_URI' not in os.environ


class TestCorrectionDataframes(unittest.TestCase):
    """
    Requires write access to some pymongo server, the URI of witch is to be set
    as an environment variable under:

        TEST_MONGO_URI

    """
    _run_test = True

    def setUp(self):
        # Just to make sure we are running some mongo server, see test-class docstring
        if 'TEST_MONGO_URI' not in os.environ:
            self._run_test = False
            return
        uri = os.environ.get('TEST_MONGO_URI')
        db_name = 'test_cmt2'
        self.dfs = straxen.CorrectionDataframes.from_mongodb(url=uri, dbname=db_name)

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    def tearDown(self):
        self.dfs.db.client.drop_database(self.dfs.db.name)
    
    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(SimpleCorrection.builds())
    def test_insert_simple(self, record):
        idx, doc = record
        rdf = self.dfs.simple_correction
        rdf.loc[idx] = doc
        self.assertEqual(rdf.at[idx, 'value'], doc.value)
        rdf.db.drop_collection(rdf.name)
    
