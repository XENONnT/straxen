"""
Test that we do not run into the issue solved in this PR. This is technically
speaking a strax issue but would take quite some time to setup properly. If
this test is failing, it's most likely relsted to a change in strax.plugin.py

For more info see:
https://github.com/AxFoundation/strax/pull/345
"""

import straxen
import os
import tempfile
from .test_basics import test_run_id_1T


# Offending peak-timestamps. These were causing the issues described in
# https://github.com/AxFoundation/strax/pull/345. They were hand picked
# time ranges that caused issues in the past  when strax was unable to
# unify the time ranges of the input chunks(e.g. strax v0.12.4, straxen
# v0.12.3).
OFFENDING_PEAK_TIMESTAMPS = [(1518690942041190780, 1518690942041191160),
                             (1518690942041191060, 1518690942041191550),
                             (1518690942090637380, 1518690942090637900),
                             (1518690942090637800, 1518690942090638590),
                             (1518690942090638500, 1518690942090641040),
                             (1518690942090657090, 1518690942090659430),
                             (1518690942090659330, 1518690942090660770),
                             (1518690942090660690, 1518690942090663690),
                             (1518690942090663600, 1518690942090666300),
                             (1518690942090666330, 1518690942090669210),
                             (1518690942090669120, 1518690942090671500),
                             (1518690942090671430, 1518690942090674090),
                             (1518690942090673990, 1518690942090678090)]


def test_time_selection():
    """Forcefully run into an error if strax #345 is not merged."""
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            print("Temporary directory is ", temp_dir)
            os.chdir(temp_dir)

            print("Downloading test data (if needed)")
            st = straxen.contexts.demo()
            # Ignore strax-internal warnings
            st.set_context_config({'free_options': tuple(st.config.keys())})

            print("Making peak basics")
            st.make(test_run_id_1T, 'peak_basics')

            print("Making sure that we have the data")
            for t in ('peaklets', 'peak_basics'):
                assert st.is_stored(test_run_id_1T, t), f'{t} should be stored'

            print("Testing if we can open the offending time ranges")
            for t0, t1 in OFFENDING_PEAK_TIMESTAMPS:
                # This will run into ValueErrors and RuntimeErrors if
                # there is a problem with the plugin handling in strax:
                # https://github.com/AxFoundation/strax/pull/345
                st.get_array(test_run_id_1T, targets='peaks', time_range=(t0, t1))

        # On windows, you cannot delete the current process'
        # working directory, so we have to chdir out first.
        finally:
            os.chdir('..')
