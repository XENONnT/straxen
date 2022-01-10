import ast
import configparser
import gzip
import inspect
import typing as ty
import commentjson
import json
import os
import os.path as osp
import pickle
import dill
import urllib.request
import tqdm
import numpy as np
import pandas as pd
from re import match
import numba
from warnings import warn
import strax
import straxen

export, __all__ = strax.exporter()
__all__ += ['straxen_dir', 'first_sr1_run', 'tpc_r', 'tpc_z', 'aux_repo',
            'n_tpc_pmts', 'n_top_pmts', 'n_hard_aqmon_start', 'ADC_TO_E',
            'n_nveto_pmts', 'n_mveto_pmts', 'tpc_pmt_radius', 'cryostat_outer_radius',
            'perp_wire_angle', 'perp_wire_x_rot_pos', 'INFINITY_64BIT_SIGNED']

straxen_dir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

aux_repo = 'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/'

tpc_r = 66.4  # [CM], Not really radius, but apothem: from MC paper draft 1.0
cryostat_outer_radius = 81.5  # [cm] radius of the outer cylinder wall.
tpc_z = 148.6515  # [CM], distance between the bottom of gate and top of cathode wires
n_tpc_pmts = 494
n_top_pmts = 253
n_hard_aqmon_start = 800

n_nveto_pmts = 120
n_mveto_pmts = 84

tpc_pmt_radius = 7.62 / 2  # cm

perp_wire_angle = np.deg2rad(30)
perp_wire_x_rot_pos = 13.06  #[cm]

# Convert from ADC * samples to electrons emitted by PMT
# see pax.dsputils.adc_to_pe for calculation. Saving this number in straxen as
# it's needed in analyses
ADC_TO_E = 17142.81741

# See https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:daq:sector_swap
LAST_MISCABLED_RUN = 8796
TSTART_FIRST_CORRECTLY_CABLED_RUN = 1596036001000000000

INFINITY_64BIT_SIGNED = 9223372036854775807


@export
def rotate_perp_wires(x_obs: np.ndarray,
                      y_obs: np.ndarray,
                      angle_extra: ty.Union[float, int] = 0):
    """
    Returns x and y in the rotated plane where the perpendicular wires
    area vertically aligned (parallel to the y-axis). Accepts addition to the
    rotation angle with `angle_extra` [deg]

    :param x_obs: array of x coordinates
    :param y_obs: array of y coordinates
    :param angle_extra: extra rotation in [deg]
    :return: x_rotated, y_rotated
    """
    if len(x_obs) != len(y_obs):
        raise ValueError('x and y are not of the same length')
    angle_extra_rad = np.deg2rad(angle_extra)
    x_rot = (np.cos(perp_wire_angle + angle_extra_rad) * x_obs
             - np.sin(perp_wire_angle + angle_extra_rad) * y_obs)
    y_rot = (np.sin(perp_wire_angle + angle_extra_rad) * x_obs
             + np.cos(perp_wire_angle + angle_extra_rad) * y_obs)
    return x_rot, y_rot


@export
def pmt_positions(xenon1t=False):
    """Return pandas dataframe with PMT positions
    columns: array (top/bottom), i (PMT number), x, y
    """
    if xenon1t:
        # Get PMT positions from the XENON1T config without PAX
        config = configparser.ConfigParser()
        config.read_string(
            resource_from_url('https://raw.githubusercontent.com/XENON1T/pax/master/pax/config/XENON1T.ini'))
        pmt_config = ast.literal_eval(config['DEFAULT']['pmts'])
        return pd.DataFrame([
            dict(x=q['position']['x'],
                 y=q['position']['y'],
                 i=q['pmt_position'],
                 array=q.get('array', 'other'))
            for q in pmt_config[:248]])
    else:
        return resource_from_url(
            aux_repo + '874de2ffe41147719263183b89d26c9ee562c334/pmt_positions_xenonnt.csv',
            fmt='csv')


# In-memory resource cache
_resource_cache = dict()

# Formats for which the original file is text, not binary
_text_formats = ['text', 'csv', 'json']


@export
def open_resource(file_name: str, fmt='text'):
    """
    Open file
    :param file_name: str, file to open
    :param fmt: format of the file
    :return: opened file
    """
    cached_name = _cache_name(file_name, fmt)
    if cached_name in _resource_cache:
        # Retrieve from in-memory cache
        return _resource_cache[cached_name]
    # File resource
    if fmt in ['npy', 'npy_pickle']:
        result = np.load(file_name, allow_pickle=fmt == 'npy_pickle')
        if isinstance(result, np.lib.npyio.NpzFile):
            # Slurp the arrays in the file, so the result can be copied,
            # then close the file so its descriptors does not leak.
            result_slurped = {k: v[:] for k, v in result.items()}
            result.close()
            result = result_slurped
    elif fmt == 'pkl':
        with open(file_name, 'rb') as f:
            result = pickle.load(f)  # nosec
    elif fmt == 'pkl.gz':
        with gzip.open(file_name, 'rb') as f:
            result = pickle.load(f)  # nosec
    elif fmt == 'dill':
        with open(file_name, 'rb') as f:
            result = dill.load(f)  # nosec
    elif fmt == 'dill.gz':
        with gzip.open(file_name, 'rb') as f:
            result = dill.load(f)  # nosec
    elif fmt == 'json.gz':
        with gzip.open(file_name, 'rb') as f:
            result = json.load(f)
    elif fmt == 'json':
        with open(file_name, mode='r') as f:
            result = commentjson.load(f)
    elif fmt == 'binary':
        with open(file_name, mode='rb') as f:
            result = f.read()
    elif fmt in ['text', 'txt']:
        with open(file_name, mode='r') as f:
            result = f.read()
    elif fmt == 'csv':
        result = pd.read_csv(file_name)
    else:
        raise ValueError(f"Unsupported format {fmt}!")

    # Store in in-memory cache
    _resource_cache[cached_name] = result

    return result


@export
def get_resource(x: str, fmt='text'):
    """
    Get the resource from an online source to be opened here. We will
        sequentially try the following:
            1. Load if from memory if we asked for it before;
            2. load it from a file if the path exists;
            3. (preferred option) Load it from our database
            4. Load the file from some URL (e.g. raw github content)

    :param x: str, either it is :
        A.) a path to the file;
        B.) the identifier of the file as it's stored under in the database;
        C.) A URL to the file (e.g. raw github content).

    :param fmt: str, format of the resource x
    :return: the opened resource file x opened according to the
        specified format
    """
    # 1. load from memory
    cached_name = _cache_name(x, fmt)
    if cached_name in _resource_cache:
        return _resource_cache[cached_name]
    # 2. load from file
    elif os.path.exists(x):
        return open_resource(x, fmt=fmt)
    # 3. load from database
    elif straxen.uconfig is not None:
        downloader = straxen.MongoDownloader()
        if x in downloader.list_files():
            path = downloader.download_single(x)
            return open_resource(path, fmt=fmt)
    # 4. load from URL
    if '://' in x:
        return resource_from_url(x, fmt=fmt)
    raise FileNotFoundError(
        f'Cannot open {x} because it is either not stored or we '
        f'cannot download it from anywhere.')


def _cache_name(name: str, fmt: str)->str:
    """Return a name under which to store the requested name with the given format in the _cache"""
    return f'{fmt}::{name}'


# Legacy loader for public URL files
def resource_from_url(html: str, fmt='text'):
    """
    Return contents of file or URL html
    :param html: str, html to the file you are requesting e.g. raw github content
    :param fmt: str, format to parse contents into

    Do NOT mutate the result you get. Make a copy if you're not sure.
    If you mutate resources it will corrupt the cache, cause terrible bugs in
    unrelated code, tears unnumbered ye shall shed, not even the echo of
    your lamentations shall pass over the mountains, etc.
    :return: The file opened as specified per it's format
    """

    if '://' not in html:
        raise ValueError('Can only open urls!')

    # Web resource; look first in on-disk cache
    # to prevent repeated downloads.
    cache_fn = strax.utils.deterministic_hash(html)
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
            result = open_resource(cf, fmt=fmt)
            break
    else:
        print(f'Did not find {cache_fn} in cache, downloading {html}')
        # disable bandit
        result = urllib.request.urlopen(html).read()
        is_binary = fmt not in _text_formats
        if not is_binary:
            result = result.decode()

        # Store in as many caches as possible
        m = 'wb' if is_binary else 'w'
        available_cf = None
        for cache_folder in cache_folders:
            if not osp.exists(cache_folder):
                continue
            if not os.access(cache_folder, os.W_OK):
                continue
            cf = osp.join(cache_folder, cache_fn)
            with open(cf, mode=m) as f:
                f.write(result)
            available_cf = cf
        if available_cf is None:
            raise RuntimeError(
                f"Could not store {html} in on-disk cache,"
                "none of the cache directories are writeable??")

        # Retrieve result from file-cache
        # (so we only need one format-parsing logic)
        result = open_resource(available_cf, fmt=fmt)
    return result


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


@export
def pre_apply_function(data, run_id, target, function_name='pre_apply_function'):
    """
    Prior to returning the data (from one chunk) see if any function(s) need to
    be applied.

    :param data: one chunk of data for the requested target(s)
    :param run_id: Single run-id of of the chunk of data
    :param target: one or more targets
    :param function_name: the name of the function to be applied. The
        function_name.py should be stored in the database.
    :return: Data where the function is applied.
    """
    if function_name not in _resource_cache:
        # only load the function once and put it in the resource cache
        function_file = f'{function_name}.py'
        function_file = straxen.test_utils._overwrite_testing_function_file(function_file)
        function = get_resource(function_file, fmt='txt')
        # pylint: disable=exec-used
        exec(function)
        # Cache the function to reduce reloading & eval operations
        _resource_cache[function_name] = locals().get(function_name)
    data = _resource_cache[function_name](data, run_id, strax.to_str_tuple(target))
    return data


@export
def check_loading_allowed(data, run_id, target,
                          max_in_disallowed=1,
                          disallowed=('event_positions',
                                      'corrected_areas',
                                      'energy_estimates')
                          ):
    """
    Check that the loading of the specified targets is not
    disallowed

    :param data: chunk of data
    :param run_id: run_id of the run
    :param target: list of targets requested by the user
    :param max_in_disallowed: the max number of targets that are
        in the disallowed list
    :param disallowed: list of targets that are not allowed to be
        loaded simultaneously by the user
    :return: data
    :raise: RuntimeError if more than max_in_disallowed targets
        are requested
    """
    n_targets_in_disallowed = sum([t in disallowed for t in
                                   strax.to_str_tuple(target)])
    if n_targets_in_disallowed > max_in_disallowed:
        raise RuntimeError(
            f'Don\'t load {disallowed} separately, use "event_info" instead')
    return data


@export
def remap_channels(data, verbose=True, safe_copy=False, _tqdm=False, ):
    """
    There were some errors in the channel mapping of old data as described in
        https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:daq:sector_swap
        using this function, we can convert old data to reflect the right channel map
        while loading the data. We convert both the field 'channel' as well as anything
        that is an array of the same length of the number of channels.

    :param data: numpy array of pandas dataframe

    :param verbose: print messages while converting data

    :param safe_copy: if True make a copy of the data prior to performing manipulations.
        Will prevent overwrites of the internal references but does require more memory.
    
    :param _tqdm: bool (try to) add a tqdm wrapper to show the progress

    :return: Correctly mapped data
    """
    # This map shows which channels were recabled. We now have to do the same in software
    # for old runs.
    remap = get_resource(
        aux_repo + '/ecb6da7bd4deb98cd0a4e83b3da81c1e67505b16/remapped_channels_since_20200729_17.20UTC.csv',
        fmt='csv')

    def convert_channel(_data, replace=('channel', 'max_pmt')):
        """
        Given an array, replace the 'channel' entry if we had to remap it according to the
            map of channels to be remapped

        :param _data: data whereof the replace entries should be changed according to the
            remapping of channels

        :param replace: entries (keys/numpy-dtypes) that should be changed in the data

        :return: remapped data where each of the replace entries has been replaced
        """
        data_keys = get_dtypes(_data)

        # loop over the things to replace
        for _rep in replace:
            if _rep not in data_keys:
                # Apparently this data doesn't have the entry we want to replace
                continue
            if _rep == 'channel' and _dat['channel'].ndim != 1:
                # Only convert channel if they are flat and not nested.
                continue
            # Make a buffer we can overwrite and replace with an remapped array
            buff = np.array(_data[_rep])
            buff = _swap_values_in_array(np.array(_data[_rep]),
                                         buff,
                                         np.array(remap['PMT_new'].values),
                                         np.array(remap['PMT_old'].values))
            _data[_rep] = buff
            if verbose:
                print(f'convert_channel::\tchanged {_rep}')
        # Not needed for np.array as the internal memory already reflects it but it is
        # needed for pd.DataFrames.
        return _data

    def remap_single_entry(_data, _array_entry):
        """
        Remap the data of a array field (_entry) in the data. For example, remap
            saturated_channel (which is of length n_pmts) where the entries of the PMT_old
            will be replaced by the entries of PMT_new and vise versa.

        :param _data: reshuffle the _data according to for _entry according to the map of
            channels to be remapped

        :param _array_entry: key or dtype of the data. NB: should be array whereof the length
            equals the number of PMTs!

        :return: correctly mapped data
        """
        _k = get_dtypes(_data)
        if _array_entry not in _k:
            raise ValueError(f'remap_single_entry::\tcannot remap {_array_entry} in data '
                             f'with fields {_k}.')
        buff = np.array(_data[_array_entry])
        for _, _row in remap.iterrows():
            pmt_new, pmt_old = _row['PMT_new'], _row['PMT_old']
            buff[:, pmt_new] = _data[_array_entry][:, pmt_old]
        _data[_array_entry] = buff
        # Not needed for np.array but for pd.DataFrames
        return _data

    def convert_channel_like(channel_data, n_chs=n_tpc_pmts):
        """
        Look for entries in the data of n_chs length. If found, assume it should be
            remapped according to the map

        :param channel_data: data to be converted according to the map of channels to be
            remapped. This data is checked for any entries (dtype names) that have a
            length equal to the n_chs and if so, is remapped accordingly

        :param n_chs: the number of channels

        :return: correctly mapped data
        """

        if not len(channel_data):
            return channel_data
        # Create a buffer to overright
        buffer = channel_data.copy()
        for k in strax.utils.tqdm(get_dtypes(channel_data), disable=not _tqdm):
            if np.iterable(channel_data[k][0]) and len(channel_data[k][0]) == n_chs:
                if verbose:
                    print(f'convert_channel_like::\tupdate {k}')
                buffer = remap_single_entry(buffer, k)
        return buffer

    # Take the last two samples as otherwise the pd.DataFrame gives an unexpected output.
    # I would have preferred st.estimate_run_start(f'00{last_miscabled_run}')) but st is
    # not per se initialized.
    if np.any(data['time'][-2:] > TSTART_FIRST_CORRECTLY_CABLED_RUN):
        raise ValueError(f'Do not remap the data after run 00{LAST_MISCABLED_RUN}')

    if safe_copy:
        # Make sure we make a new entry as otherwise some internal buffer of numpy arrays
        # may yield puzzling results as internal buffers may also reflect the change.
        _dat = data.copy()
        del data
    else:
        # Just continue with data
        _dat = data

    # Do the conversion(s)
    _dat = convert_channel(_dat)
    if not isinstance(_dat, pd.DataFrame):
        # pd.DataFrames are flat arrays and thus cannot have channel_like arrays in them
        _dat = convert_channel_like(_dat)

    return _dat


@export
def remap_old(data, targets, run_id, works_on_target=''):
    """
    If the data is of before the time sectors were re-cabled, apply a software remap
        otherwise just return the data is it is.
    :param data: numpy array of data with at least the field time. It is assumed the data
        is sorted by time
    :param targets: targets in the st.get_array to get
    :param run_id: required positional argument of apply_function_to_data in strax
    :param works_on_target: regex match string to match any of the targets. By default set
        to '' such that any target in the targets would be remapped (which is what we want
        as channels are present in most data types). If one only wants records (no
        raw-records) and peaks* use e.g. works_on_target = 'records|peaks'.
    """

    if np.any(data['time'][:2] >= TSTART_FIRST_CORRECTLY_CABLED_RUN):
        # We leave the 'new' data be
        pass
    elif not np.any([match(works_on_target, t) for t in strax.to_str_tuple(targets)]):
        # None of the targets are such that we want to remap
        pass
    elif len(data):
        # select the old data and do the remapping for this
        mask = data['time'] < TSTART_FIRST_CORRECTLY_CABLED_RUN
        data = data.copy()
        data[mask] = remap_channels(data[mask])
    return data


@export
def get_dtypes(_data):
    """
    Return keys/dtype names of pd.DataFrame or numpy array

    :param _data: data to get the keys/dtype names

    :return: keys/dtype names
    """
    if isinstance(_data, np.ndarray):
        _k = _data.dtype.names
    elif isinstance(_data, pd.DataFrame):
        _k = _data.keys()
    return _k


@numba.jit(nopython=True, nogil=True, cache=True)
def _swap_values_in_array(data_arr, buffer, items, replacements):
    """
    Fill buffer for item[k] -> replacements[k]
    :param data_arr: numpy array of data
    :param buffer: copy of data_arr where the replacements of items will be saved
    :param items: array of len x containing values that are in data_arr and need to be
        replaced with the corresponding item in replacements
    :param replacements: array of len x containing the values that should replace the
        corresponding item in items
    :return: the buffer reflecting the changes
    """
    for i, val in enumerate(data_arr):
        for k, it in enumerate(items):
            if val == it:
                buffer[i] = replacements[k]
                break
    return buffer


##
# Old XENON1T Stuff
##


first_sr1_run = 170118_1327


@export
def pax_file(x):
    """Return URL to file hosted in the pax repository master branch"""
    return 'https://raw.githubusercontent.com/XENON1T/pax/master/pax/data/' + x
