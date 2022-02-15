"""
Test certain interactions with the runsdatabase.
NB! this only works if one has access to the database. This does not
work e.g. on travis jobs and therefore the tests failing locally will
not show up in Pull Requests.
"""
import straxen
import os
import unittest
from pymongo import ReadPreference
import warnings


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
class TestSelectRuns(unittest.TestCase):
    def test_select_runs(self, check_n_runs=2):
        """
        Test (if we have a connection) if we can perform strax.select_runs
            on the last two runs in the runs collection

        :param check_n_runs: int, the number of runs we want to check
        """
        self.assertTrue(check_n_runs >= 1)
        st = straxen.contexts.xenonnt_online(use_rucio=False)
        run_col = st.storage[0].collection

        # Find the latest run in the runs collection
        last_run = run_col.find_one(projection={'number': 1},
                                    sort=[('number', -1)]
                                    ).get('number')

        # Set this number as the minimum run number. This limits the
        # amount of documents checked and therefore keeps the test short.
        st.storage[0].minimum_run_number = int(last_run) - (check_n_runs - 1)
        st.select_runs()


@unittest.skipIf(not straxen.utilix_is_configured(),
                 "Cannot download because utilix is not configured")
class TestDownloader(unittest.TestCase):
    def test_downloader(self):
        """Test if we can download a small file from the downloader"""
        downloader = straxen.MongoDownloader()
        path = downloader.download_single('to_pe_nt.npy')
        self.assertTrue(os.path.exists(path))


def _patch_om_init(take_only):
    """
    temp patch since om = straxen.OnlineMonitor() does not work with utilix
    """
    header = 'RunDB'
    user = straxen.uconfig.get(header, 'pymongo_user')
    pwd = straxen.uconfig.get(header, 'pymongo_password')
    url = straxen.uconfig.get(header, 'pymongo_url').split(',')[-1]
    uri = f"mongodb://{user}:{pwd}@{url}"
    return straxen.OnlineMonitor(uri=uri, take_only=take_only)


@unittest.skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_online_monitor(target='online_peak_monitor', max_tries=3):
    """
    See if we can get some data from the online monitor before max_tries

    :param target: target to test getting from the online monitor
    :param max_tries: number of queries max allowed to get a non-failing
        run
    """
    st = straxen.contexts.xenonnt_online(use_rucio=False)
    straxen.get_mongo_uri()
    om = _patch_om_init(target)
    st.storage = [om]
    k=st.key_for('0', 'online_peak_monitor')
    from pprint import pprint
    pprint(k.lineage)
    assert False, (k, k.lineage)
    max_run = None
    for i in range(max_tries):
        query = {'provides_meta': True, 'data_type': target}
        if max_run is not None:
            # One run failed before, lets try a more recent one.
            query.update({'number': {"$gt": int(max_run)}})
        collection = om.db[om.col_name].with_options(
            read_preference=ReadPreference.SECONDARY_PREFERRED)
        some_run = collection.find_one(query,
                                       projection={'number': 1,
                                                   'metadata': 1,
                                                   'lineage_hash': 1,
                                                   })
        if some_run is not None and some_run.get('lineage_hash', False):
            if some_run['lineage_hash'] != st.key_for("0", target).lineage_hash:
                # We are doing a new release, therefore there is no
                # matching data. This makes sense.
                return
        if some_run is None or some_run.get('number', None) is None:
            print(f'Found None')
            continue
        elif 'exception' in some_run.get('metadata', {}):
            # Did find a run, but it is bad, we need to find another one
            print(f'Found {some_run.get("number", "No number")} with errors')
            max_run = some_run.get("number", -1)
            continue
        else:
            # Correctly written
            run_id = f'{some_run["number"]:06}'
            break
    else:
        if collection.find_one() is not None:
            raise FileNotFoundError(f'No non-failing {target} found in the online '
                                    f'monitor after {max_tries}. Looked for:\n'
                                    f'{st.key_for("0", target)}')
        warnings.warn(f'Did not find any data in {om.col_name}!')
        return
    st.get_array(run_id, target, seconds_range=(0, 1), allow_incomplete=True)
