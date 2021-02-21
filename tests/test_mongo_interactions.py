"""
Test certain interactions with the runsdatabase.
NB! this only works if one has access to the database. This does not
work e.g. on travis jobs and therefore the tests failing locally will
not show up in Pull Requests.
"""

import straxen
import os
from warnings import warn

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
    st.storage[0].minimum_run_number = int(last_run) - (check_n_runs-1)
    st.select_runs()


def test_downloader():
    if not straxen.utilix_is_configured():
        warn('Cannot download because utilix is not configured')
        return

    downloader = straxen.MongoDownloader()
    path = downloader.download_single('to_pe_nt.npy')
    assert os.path.exists(path)
