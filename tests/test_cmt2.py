
import pydantic
import unittest
import time
import pytz
import straxen
import os
import pymongo
import datetime
import numpy as np
import pandas as pd
from hypothesis import settings, given, assume, strategies as st
from straxen.remote_dataframes.schema import InsertionError



def mongo_uri_not_set():
    return 'TEST_MONGO_URI' not in os.environ


@st.composite
def non_overlapping_interval_lists(draw, elements=st.datetimes(), min_size=2):
    elem = draw(st.lists(elements, unique=True, min_size=min_size*2).map(sorted))
    return list(zip(elem[:-1:2], elem[1::2]))


@st.composite
def non_overlapping_interval_ranges(draw, elements=st.datetimes(), min_size=2):
    elem = draw(st.lists(elements, unique=True, min_size=min_size).map(sorted))
    return list(zip(elem[:-1], elem[1:]))


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
        
        rdf.db.drop_collection(rdf.name)

        rdf.loc[idx] = doc
        self.assertEqual(rdf.at[idx, 'value'], doc.value)
        alt_doc = doc.copy()
        alt_doc.value += 1
        with self.assertRaises(IndexError):
            rdf.loc[idx] = alt_doc

        rdf.loc[idx+1] = alt_doc
        self.assertEqual(rdf.at[idx+1, 'value'], alt_doc.value)
        self.assertNotEqual(rdf.at[idx, 'value'], alt_doc.value)
        
        df = rdf.sel()
        self.assertIsInstance(df, pd.DataFrame)

        df = rdf.sel(version=idx)
        self.assertIsInstance(df, pd.DataFrame)
    
    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.lists(SomeSampledCorrection.builds(version={'min_value':1, 'max_value':1}),
                    min_size=3, unique_by=lambda x: x[0]))
    @settings(deadline=None)
    def test_sampled_correction_version1(self, records):
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
            rdf.loc[idx] = doc

        for (idx1, doc1), (idx2, doc2) in zip(records[:-1], records[1:]):

            dt = idx1[1] + (idx2[1] - idx1[1])/2
            val = rdf.at[(1, dt), 'value']
            if abs(val) == float('nan'):
                continue
            half_value = (doc2.value+doc1.value)/2
            # require better than 1% accuracy
            thresh = max(1e-2*abs(doc1.value), 1e-2*abs(doc2.value))
       
            diff = abs(half_value - val)

            if diff == float('inf'):
                continue

            if np.isnan(diff):
                continue

            self.assertLessEqual(diff, thresh)
            
        df = rdf.loc[:]
        self.assertIsInstance(df, pd.DataFrame)


    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.lists(SomeSampledCorrection.builds(version={'min_value':0, 'max_value':0}),
            min_size=5, unique_by=lambda x: x[0]))
    @settings(deadline=None)
    def test_sampled_correction_version0(self, records):
        assume(all([r[0][0]==0] for r in records))

        rdf = self.dfs[SomeSampledCorrection.name]
        rdf.db.drop_collection(rdf.name)

        # we must sort by time before inserting
        # since values are interpolated in time 
        records = sorted(records, key=lambda x: x[0][1])

        for idx, doc in records:
            cutoff = straxen.corrections_settings.clock.cutoff_datetime(buffer=1)
            if idx[1]>cutoff:
                rdf.loc[idx] = doc
            elif idx[1]<cutoff-datetime.timedelta(seconds=1):
                with self.assertRaises(InsertionError):
                    rdf.loc[idx] = doc
        df = rdf.loc[:]
        self.assertIsInstance(df, pd.DataFrame)

        
    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.lists(SomeTimeIntervalCorrection.builds(version={'min_value':1, 'max_value':1}),
                    min_size=2, unique_by=lambda x: (x[0][0], x[0][1][0])))
    @settings(deadline=None)
    def test_interval_correction_version1(self, records):

        rdf = self.dfs[SomeTimeIntervalCorrection.name]
        rdf.db.drop_collection(rdf.name)

        # we must sort by time before inserting
        # since values are interpolated in time 
        records = sorted(records, key=lambda x: x[0][1][0])

        for (idx1, doc1), (idx2, _) in zip(records[:-1], records[1:]):
    
            version = idx1[0]
            interval = (idx1[1][0], idx2[1][0])
            assume((interval[1] - interval[0])>datetime.timedelta(seconds=10))

            idx = (version, interval)
            rdf.loc[idx] = doc1
            dt = interval[0] + (interval[1] - interval[0])/2
            if dt in interval:
                continue
            self.assertEqual(rdf.at[(version, dt), 'value'], doc1.value)
        
        df = rdf.sel()
        self.assertIsInstance(df, pd.DataFrame)