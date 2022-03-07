
from typing import List
import pydantic
import numbers
import unittest
import pytz
import straxen
import os
import datetime
import pandas as pd

from hypothesis import settings, given, assume, strategies as st
from rframe.schema import InsertionError

# enforce datetimes are in the a reasonable range
# pandas nanosecond resolution has trouble with extreme dates
datetimes = st.datetimes(min_value=datetime.datetime(2000, 1, 1, 0, 0),
                         max_value=datetime.datetime(2232, 1, 1, 0, 0))

def mongo_uri_not_set():
    return 'TEST_MONGO_URI' not in os.environ


def make_datetime_index(start, stop, step='1d'):
    if isinstance(start, numbers.Number):
        start = pd.to_datetime(start, unit='s', utc=True)
    if isinstance(stop, numbers.Number):
        stop = pd.to_datetime(stop, unit='s', utc=True)
    return pd.date_range(start, stop, freq=step)


def make_datetime_interval_index(start, stop, step='1d'):
    if isinstance(start, numbers.Number):
        start = pd.to_datetime(start, unit='s', utc=True)
    if isinstance(stop, numbers.Number):
        stop = pd.to_datetime(stop, unit='s', utc=True)
    return pd.interval_range(start, stop, periods=step)


@st.composite
def non_overlapping_interval_lists(draw, elements=st.datetimes(), min_size=2):
    elem = draw(st.lists(elements, unique=True, min_size=min_size*2).map(sorted))
    return list(zip(elem[:-1:2], elem[1::2]))


@st.composite
def non_overlapping_interval_ranges(draw, elements=st.datetimes(), min_size=2):
    elem = draw(st.lists(elements, unique=True, min_size=min_size).map(sorted))
    return list(zip(elem[:-1], elem[1:]))

class SimpleCorrection(straxen.BaseCorrectionSchema):
    _NAME = 'simple_correction'

    # enforce float is in the range that mongodb can handle
    value: pydantic.confloat(gt=-2**31, lt=2**31)
    

class SomeSampledCorrection(straxen.TimeSampledCorrection):
    _NAME = 'sampled_correction'

    # enforce float is in the range that mongodb can handle
    value: pydantic.confloat(gt=-2**31, lt=2**31)


class SomeTimeIntervalCorrection(straxen.TimeIntervalCorrection):
    _NAME = 'time_interval_correction'

    # enforce float is in the range that mongodb can handle
    value: pydantic.confloat(gt=-2**31, lt=2**31)


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
        self.dfs = straxen.CorrectionFrames.from_mongodb(url=uri, db=db_name)

    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    def tearDown(self):
        self.dfs.db.client.drop_database(self.dfs.db.name)
    
    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.builds(SimpleCorrection))
    def test_simple_correction(self, doc: SimpleCorrection):
        idx = doc.version
        name = SimpleCorrection._NAME
        rdf = self.dfs[name]
        
        self.dfs.db.drop_collection(name)

        rdf.loc[idx] = doc
        self.assertEqual(rdf.at[idx, 'value'], doc.value)
        alt_doc = doc.copy()
        alt_doc.value += 1
        with self.assertRaises(IndexError):
            rdf.loc[idx] = alt_doc

        rdf.loc[idx+'1'] = alt_doc
        self.assertEqual(rdf.at[idx+'1', 'value'], alt_doc.value)
        self.assertNotEqual(rdf.at[idx, 'value'], alt_doc.value)
        
        df = rdf.sel()
        self.assertIsInstance(df, pd.DataFrame)

        df = rdf.sel(version=idx)
        self.assertIsInstance(df, pd.DataFrame)
    
    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.lists(st.builds(SomeSampledCorrection,
                            version=st.just('v1'),
                            time=datetimes),
                    min_size=3, unique_by=lambda x: (x.time, x.version)))
    @settings(deadline=None)
    def test_sampled_correction_v1(self, docs: List[SomeSampledCorrection]):
        
        docs = sorted(docs, key=lambda x: x.time)

        for doc1, doc2 in zip(docs[:-1], docs[1:]):
            # Require minimum 10 second spacing between samples
            # otherwise we get into trouble with rounding
            assume((doc2.time - doc1.time)>datetime.timedelta(seconds=10))

        name = SomeSampledCorrection._NAME
        rdf = self.dfs[name]
        self.dfs.db.drop_collection(name)

        # we must sort by time before inserting
        # since values are interpolated in time 
        
        for doc in docs:
            assume(not pd.isna(doc.value))
            assume(abs(doc.value) < float('inf'))
            rdf.loc[doc.index_labels_tuple] = doc

        for doc1, doc2 in zip(docs[:-1], docs[1:]):
            dt = doc1.time + (doc2.time - doc1.time)/2
            val = rdf.at[('v1', dt), 'value']
            if abs(val) == float('nan'):
                continue
            half_value = (doc2.value+doc1.value)/2
            diff = abs(half_value-val)
            thresh = max(1e-2*abs(doc2.value), 1e-2*abs(doc1.value), 1e-2)
            self.assertLessEqual(diff, thresh)

        df = rdf.loc[:]
        self.assertIsInstance(df, pd.DataFrame)


    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.lists(st.builds(SomeSampledCorrection,
                            version=st.just('ONLINE'),
                            time=datetimes),
                    min_size=3, unique_by=lambda x: (x.version, x.time)))
    @settings(deadline=None)
    def test_sampled_correction_online(self, docs: List[SomeSampledCorrection]):
        name = SomeSampledCorrection._NAME
        rdf = self.dfs[name]
        self.dfs.db.drop_collection(name)

        # we must sort by time before inserting
        # since values are interpolated in time 
        docs = sorted(docs, key=lambda x: x.time)

        for doc in docs:
            clock = straxen.corrections_settings.clock
            cutoff = clock.cutoff_datetime(buffer=1)
            time = doc.time
            if clock.utc:
                time = time.replace(tzinfo=pytz.UTC)
            if time>cutoff:
                rdf.loc[doc.index_labels_tuple] = doc
            elif time<cutoff-datetime.timedelta(seconds=1):
                with self.assertRaises(InsertionError):
                    rdf.loc[doc.index_labels_tuple] = doc
        df = rdf.loc[:]
        self.assertIsInstance(df, pd.DataFrame)

        
    @unittest.skipIf(mongo_uri_not_set(), "No access to test database")
    @given(st.lists(st.builds(SomeTimeIntervalCorrection,
                    version=st.just('v1'), 
                    time=datetimes,
                    ),
                    min_size=1,
                    unique_by=lambda x: (x.version, x.time.left),
                    )
            )
    @settings(deadline=None)
    def test_interval_correction_v1(self, docs: List[SomeTimeIntervalCorrection]):
        name = SomeTimeIntervalCorrection._NAME
        rdf = self.dfs[name]
        self.dfs.db.drop_collection(name)

        # we must sort by time before inserting
        # since values are interpolated in time 
        docs = sorted(docs, key=lambda x: x.time.left)
        last = docs[-1].time.left + datetime.timedelta(days=10)
        times = sorted([doc.time.left for doc in docs]) + [last]

        for doc,left,right in zip(docs, times[:-1], times[1:]):
            # very small differences may be hard to test for correctness
            assume((right-left) > datetime.timedelta(seconds=10))

            doc.time.left = left
            doc.time.right = right
            
        for doc in docs:
            rdf.loc[doc.index_labels_tuple] = doc
      
            dt = doc.time.left + (doc.time.right - doc.time.left)/2
            val = rdf.at[(doc.version, dt), 'value']
            self.assertEqual(val, doc.value)
            
        df1 = rdf.sel()
        self.assertIsInstance(df1, pd.DataFrame)
