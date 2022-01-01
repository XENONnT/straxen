
import pydantic
import unittest
import straxen
import os
import pymongo
from hypothesis import settings, given, strategies as st



def mongo_uri_not_set():
    return 'TEST_MONGO_URI' not in os.environ


class SimpleCorrection(straxen.BaseCorrectionSchema):
    name = 'simple_correction'
    index = straxen.IntegerIndex(name='version')
    value: pydantic.confloat(gt=0, lt=2**32-1)
    

class SomeSampledCorrection(straxen.TimeSampledCorrection):
    name = 'sampled_correction'
    value: pydantic.confloat(lt=2**32-1)


class SomeTimeIntervalCorrection(straxen.TimeIntervalCorrection):
    name = 'time_interval_correction'
    value: pydantic.confloat(lt=2**32-1)


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
    def test_simple_correction(self, record):
        idx, doc = record
        rdf = self.dfs.simple_correction
        rdf.loc[idx] = doc
        self.assertEqual(rdf.at[idx, 'value'], doc.value)
        alt_doc = doc.copy()
        alt_doc.value += 1
        with self.assertRaises(IndexError):
            rdf.loc[idx] = alt_doc

        rdf.loc[idx+1] = alt_doc
        self.assertEqual(rdf.at[idx+1, 'value'], alt_doc.value)
        rdf.db.drop_collection(rdf.name)
    
    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.lists(SomeSampledCorrection.builds(), min_size=2, unique=True))
    @settings(deadline=None)
    def test_sampled_correctoin(self, records):

        rdf = self.dfs.sampled_correction
        rdf.db.drop_collection(rdf.name)

        # we must sort by time before inserting
        # since values are interpolated in time 
        records.sort(key=lambda x: x[0][1])

        for idx, doc in records:
            v,dt = idx
            while v<1:
                v += 1
            rdf.loc[(v,dt)] = doc
        
        df = rdf.loc[:]

        self.assertEqual(len(df), len(records))


        # test version 0
        rdf.db.drop_collection(rdf.name)
        cutoff = straxen.corrections_settings.clock.cutoff_datetime(buffer=3600)

        after_cutoff = list(filter(lambda x: x[0][1]>cutoff, records))

        for idx, doc in after_cutoff:
            _, dt = idx
            rdf.loc[(0,dt)] = doc

        df = rdf.loc[:]

        


