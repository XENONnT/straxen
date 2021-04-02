"""
Test certain interactions with the runsdatabase.
NB! this only works if one has access to the database. This does not
work e.g. on travis jobs and therefore the tests failing locally will
not show up in Pull Requests.
"""

import straxen
import os
from warnings import warn
from .test_plugins import test_run_id_nT


def test_select_runs(check_n_runs=2):
    """
    Test (if we have a connection) if we can perform strax.select_runs
        on the last two runs in the runs collection

    :param check_n_runs: int, the number of runs we want to check
    """

    if not straxen.utilix_is_configured():
        warn('Makes no sense to test the select runs because we do not '
             'have access to the database.')
        return
    assert check_n_runs >= 1
    st = straxen.contexts.xenonnt_online()
    run_col = st.storage[0].collection

    # Find the latest run in the runs collection
    last_run = run_col.find_one(projection={'number': 1},
                                sort=[('number', -1)]
                                ).get('number')

    # Set this number as the minimum run number. This limits the
    # amount of documents checked and therefore keeps the test short.
    st.storage[0].minimum_run_number = int(last_run) - (check_n_runs - 1)
    st.select_runs()


def test_downloader():
    """Test if we can download a small file from the downloader"""
    if not straxen.utilix_is_configured():
        warn('Cannot download because utilix is not configured')
        return

    downloader = straxen.MongoDownloader()
    path = downloader.download_single('to_pe_nt.npy')
    assert os.path.exists(path)


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


def test_online_monitor(target='online_peak_monitor', max_tries=3):
    """
    See if we can get some data from the online monitor before max_tries

    :param target: target to test getting from the online monitor
    :param max_tries: number of queries max allowed to get a non-failing
        run
    """
    if not straxen.utilix_is_configured():
        warn('Cannot test online monitor because utilix is not configured')
        return
    st = straxen.contexts.xenonnt_online()
    om = _patch_om_init(target)
    st.storage = [om]
    max_run = None
    for i in range(max_tries):
        query = {'provides_meta': True, 'data_type': target}
        if max_run is not None:
            # One run failed before, lets try a more recent one.
            query.update({'number': {"$gt": int(max_run)}})
        some_run = om.db[om.col_name].find_one(query,
                                               projection={'number': 1,
                                                           'metadata': 1})
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
        raise FileNotFoundError(f'No non-failing {target} found in the online '
                                f'monitor after {max_tries}. Looked for:\n'
                                f'{st.key_for("0", target)}')
    st.get_array(run_id, target, seconds_range=(0, 1), allow_incomplete=True)
