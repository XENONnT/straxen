import strax
import straxen
import tarfile
import io
import os
from warnings import warn
from os import environ as os_environ

export, __all__ = strax.exporter()


@export
def download_test_data(test_data='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/353b2c60a01e96f67e4ba544ce284bd91241964d/strax_files/strax_test_data_straxv1.1.0.tar',  #  noqa
                       ):
    """Downloads strax test data to strax_test_data in the current directory"""
    blob = straxen.common.get_resource(test_data, fmt='binary')
    f = io.BytesIO(blob)
    tf = tarfile.open(fileobj=f)
    tf.extractall()

@export
def _overwrite_testing_function_file(function_file):
    """For testing purposes allow this function file to be loaded from HOME/testing_folder"""
    if not _is_on_pytest():
        # If we are not on a pytest, never try using a local file.
        return function_file

    home = os.environ.get('HOME')
    if home is None:
        # Impossible to load from non-existent folder
        return function_file

    testing_file = os.path.join(home, function_file)

    if os.path.exists(testing_file):
        # For testing purposes allow loading from 'home/testing_folder'
        warn(f'Using local function: {function_file} from {testing_file}! '
             f'If you are not integrated testing on github you should '
             f'absolutely remove this file. (See #559)')
        function_file = testing_file

    return function_file


@export
def _is_on_pytest():
    """Check if we are on a pytest"""
    return 'PYTEST_CURRENT_TEST' in os_environ


nt_test_run_id = '012882'


def nt_test_context(target_context='xenonnt_online',
                    **kwargs):
    st = getattr(straxen.contexts, target_context)(**kwargs)
    st._plugin_class_registry['raw_records'].__version__ = "MOCKTESTDATA"
    st.storage = [strax.DataDirectory('./strax_test_data')]
    download_test_data('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/8b304bde43260eb47b4d666c244a386ac5a25b51/strax_files/012882-raw_records-z7q2d2ye2t.tar')
    return st
