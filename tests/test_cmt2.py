
import pydantic
import unittest
import time
import pytz
import straxen
import os
import pymongo
import datetime
from hypothesis import settings, given, assume, strategies as st
from straxen.remote_dataframes.schema import InsertionError



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
        rdf = self.dfs[SimpleCorrection.name]
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
    @given(st.lists(SomeSampledCorrection.builds(version={'min_value':1, 'max_value':1}),
                    min_size=3, unique_by=lambda x: x[0]))
    @settings(deadline=None)
    def test_sampled_correction_version_1(self, records):
        assume(all([r[0][0]==1] for r in records))
        
        records = sorted(records, key=lambda x: x[0][1])

        for (idx1, doc1), (idx2, doc2) in zip(records[:-1], records[1:]):
            # Require minimum 10 second spacing between samples
            # otherwise we get into trouble with rounding
            assume((idx2[1] - idx1[1])>datetime.timedelta(seconds=10))
        rdf = self.dfs[SomeSampledCorrection.name]
        rdf.db.drop_collection(rdf.name)

        # we must sort by time before inserting
        # since values are interpolated in time 
        
        for idx, doc in records:
            # v, dt = idx
            # dt = dt
            rdf.loc[idx] = doc

        for (idx1, doc1), (idx2, doc2) in zip(records[:-1], records[1:]):
            
            if doc1.value <1e-2 or doc2.value<1e-2:
                continue
            
            if abs(doc2.value - doc1.value)<1e-2:
                continue
            
            half_deltat = (idx2[1] - idx1[1])/2
            
            if half_deltat < datetime.timedelta(seconds=10):
                continue

            dt = (idx1[1] + half_deltat)
            val = rdf.at[(1, dt), 'value']

            half_value = (doc2.value+doc1.value)/2
            # require better than 1% accuracy
            thresh = max(1e-2*doc1.value, 1e-2*doc2.value)
            diff = abs(half_value- val)

            self.assertLessEqual(diff, thresh)
            

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.lists(SomeSampledCorrection.builds(version={'min_value':0, 'max_value':0}),
            min_size=5, unique_by=lambda x: x[0]))
    @settings(deadline=None)
    def test_sampled_correction_version_0(self, records):
        assume(all([r[0][0]==0] for r in records))

        rdf = self.dfs[SomeSampledCorrection.name]
        rdf.db.drop_collection(rdf.name)

        # we must sort by time before inserting
        # since values are interpolated in time 
        records = sorted(records, key=lambda x: x[0][1])

        cutoff = straxen.corrections_settings.clock.cutoff_datetime(buffer=3600)
        now = straxen.corrections_settings.clock.current_datetime()

        # after_cutoff = list(filter(lambda x: x[0][1]>cutoff and not x[0][0], records))
        for idx, doc in records:
            cutoff = straxen.corrections_settings.clock.cutoff_datetime(buffer=1)
            if idx[1]>cutoff:
                rdf.loc[idx] = doc
            elif idx[1]<cutoff-datetime.timedelta(seconds=1):
                with self.assertRaises(InsertionError):
                    rdf.loc[idx] = doc
        df = rdf.loc[:]

        
    # @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    # @given(st.lists(SomeTimeIntervalCorrection.builds(),
    #                 min_size=1, unique_by=lambda x: x[0]))
    # @settings(deadline=None)
    # def test_interval_correctoin(self, records):

    #     rdf = self.dfs[SomeTimeIntervalCorrection.name]
    #     rdf.db.drop_collection(rdf.name)

    #     # we must sort by time before inserting
    #     # since values are interpolated in time 
    #     records.sort(key=lambda x: x[0][1])

        

    #     prev = records[0][0][1][1]

    #     for idx, doc in records:
    #         version, (left, right) = idx
    #         if prev<=left:
    #             rdf.loc[idx] = doc
    #             prev = right
