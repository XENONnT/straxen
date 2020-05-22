import ast
import configparser
import gzip
import inspect
import io
import json
import os
import os.path as osp
import pickle
import socket
import sys
import tarfile
import urllib.request

import numpy as np
import pandas as pd

import strax
export, __all__ = strax.exporter()
__all__ += ['straxen_dir', 'first_sr1_run', 'tpc_r', 'aux_repo',
            'n_tpc_pmts', 'n_top_pmts']

straxen_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

aux_repo = 'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/'

tpc_r = 66.4   # Not really radius, but apothem: from MC paper draft 1.0
n_tpc_pmts = 494
n_top_pmts = 253


@export
def pmt_positions(xenon1t=False):
    """Return pandas dataframe with PMT positions
    columns: array (top/bottom), i (PMT number), x, y
    """
    if xenon1t:
        # Get PMT positions from the XENON1T config without PAX
        config = configparser.ConfigParser()
        config.read_string(
            get_resource('https://raw.githubusercontent.com/XENON1T/pax/master/pax/config/XENON1T.ini'))
        pmt_config = ast.literal_eval(config['DEFAULT']['pmts'])
        return pd.DataFrame([
            dict(x=q['position']['x'],
                 y=q['position']['y'],
                 i=q['pmt_position'],
                 array=q.get('array','other'))
            for q in pmt_config[:248]])
    else:
        return get_resource(
            aux_repo + '874de2ffe41147719263183b89d26c9ee562c334/pmt_positions_xenonnt.csv',
            fmt='csv')


# In-memory resource cache
_resource_cache = dict()

# Formats for which the original file is text, not binary
_text_formats = ['text', 'csv', 'json']


# Placeholder for resource management system in the future?
@export
def get_resource(x, fmt='text'):
    """Return contents of file or URL x
    :param fmt: Format to parse contents into

    Do NOT mutate the result you get. Make a copy if you're not sure.
    If you mutate resources it will corrupt the cache, cause terrible bugs in
    unrelated code, tears unnumbered ye shall shed, not even the echo of
    your lamentations shall pass over the mountains, etc.
    """
    if x in _resource_cache:
        # Retrieve from in-memory cache
        return _resource_cache[x]

    if '://' in x:
        # Web resource; look first in on-disk cache
        # to prevent repeated downloads.
        cache_fn = strax.utils.deterministic_hash(x)
        cache_folders = ['./resource_cache',
                         '/tmp/straxen_resource_cache',
                         '/dali/lgrandi/strax/resource_cache']
        for cache_folder in cache_folders:
            try:
                os.makedirs(cache_folder, exist_ok=True)
            except (PermissionError, OSError):
                continue
            cf = osp.join(cache_folder, cache_fn)
            if osp.exists(cf):
                result = get_resource(cf, fmt=fmt)
                break
        else:
            print(f'Did not find {cache_fn} in cache, downloading {x}')
            result = urllib.request.urlopen(x).read()
            is_binary = fmt not in _text_formats
            if not is_binary:
                result = result.decode()

            # Store in as many caches as possible
            m = 'wb' if is_binary else 'w'
            available_cf = None
            for cache_folder in cache_folders:
                if not osp.exists(cache_folder):
                    continue
                cf = osp.join(cache_folder, cache_fn)
                try:
                    with open(cf, mode=m) as f:
                        f.write(result)
                except Exception:
                    pass
                else:
                    available_cf = cf
            if available_cf is None:
                raise RuntimeError(
                    f"Could not store {x} in on-disk cache,"
                    "none of the cache directories are writeable??")

            # Retrieve result from file-cache
            # (so we only need one format-parsing logic)
            result = get_resource(available_cf, fmt=fmt)

    else:
        # File resource
        if fmt in ['npy', 'npy_pickle']:
            result = np.load(x, allow_pickle=fmt == 'npy_pickle')
            if isinstance(result, np.lib.npyio.NpzFile):
                # Slurp the arrays in the file, so the result can be copied,
                # then close the file so its descriptors does not leak.
                result_slurped = {k: v[:] for k, v in result.items()}
                result.close()
                result = result_slurped
        elif fmt == 'pkl':
            with open(x, 'rb') as f:
                result = pickle.load(f)
        elif fmt == 'pkl.gz':
            with gzip.open(x, 'rb') as f:
                result = pickle.load(f)
        elif fmt == 'json.gz':
            with gzip.open(x, 'rb') as f:
                result = json.load(f)
        elif fmt == 'json':
            with open(x, mode='r') as f:
                result = json.load(f)
        elif fmt == 'binary':
            with open(x, mode='rb') as f:
                result = f.read()
        elif fmt == 'text':
            with open(x, mode='r') as f:
                result = f.read()
        elif fmt == 'csv':
            result = pd.read_csv(x)
        else:
            raise ValueError(f"Unsupported format {fmt}!")

    # Store in in-memory cache
    _resource_cache[x] = result

    return result


@export
def get_elife(run_id,elife_file):
    x = get_resource(elife_file, fmt='npy')
    run_index = np.where(x['run_id'] == int(run_id))[0]
    if not len(run_index):
        # Gains not known: using placeholders
        e = 623e3
    else:
        e = x[run_index[0]]['e_life']
    return e


@export
def get_secret(x):
    """Return secret key x. In order of priority, we search:

      * Environment variable: uppercase version of x
      * xenon_secrets.py (if included with your straxen installation)
      * A standard xenon_secrets.py located on the midway analysis hub
        (if you are running on midway)
    """
    env_name = x.upper()
    if env_name in os.environ:
        return os.environ[env_name]

    message = (f"Secret {x} requested, but there is no environment "
               f"variable {env_name}, ")
    try:
        from . import xenon_secrets
    except ImportError:
        message += ("nor was there a valid xenon_secrets.py "
                    "included with your straxen installation, ")

        # If on midway, try loading a standard secrets file instead
        if 'rcc' in socket.getfqdn():
            path_to_secrets = '/project2/lgrandi/xenonnt/xenon_secrets.py'
            if os.path.exists(path_to_secrets):
                sys.path.append(osp.dirname(path_to_secrets))
                import xenon_secrets
                sys.path.pop()
            else:
                raise ValueError(
                    message + ' nor could we load the secrets module from '
                              f'{path_to_secrets}, even though you seem '
                              'to be on the midway analysis hub.')

        else:
            raise ValueError(
                message + 'nor are you on the midway analysis hub.')

    if hasattr(xenon_secrets, x):
        return getattr(xenon_secrets, x)
    raise ValueError(message + " and the secret is not in xenon_secrets.py")


@export
def download_test_data():
    """Downloads strax test data to strax_test_data in the current directory"""
    blob = get_resource('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/11929e7595e178dd335d59727024847efde530fb/strax_test_data_straxv0.9.tar',
                        fmt='binary')
    f = io.BytesIO(blob)
    tf = tarfile.open(fileobj=f)
    tf.extractall()


@export
def get_livetime_sec(context, run_id, things=None):
    """Get the livetime of a run in seconds. If it is not in the run metadata,
    estimate it from the data-level metadata of the data things.
    """
    try:
        md = context.run_metadata(run_id,
                                  projection=('start', 'end', 'livetime'))
    except strax.RunMetadataNotAvailable:
        if things is None:
            raise
        return (strax.endtime(things[-1]) - things[0]['time']) / 1e9
    else:
        if 'livetime' in md:
            return md['livetime']
        else:
            return (md['end'] - md['start']).total_seconds()



##
# Old XENON1T Stuff
##


first_sr1_run = 170118_1327


@export
def pax_file(x):
    """Return URL to file hosted in the pax repository master branch"""
    return 'https://raw.githubusercontent.com/XENON1T/pax/master/pax/data/' + x
