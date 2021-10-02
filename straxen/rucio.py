import socket
import re
import json
from bson import json_util
import os
import glob
import hashlib
import time
from utilix import xent_collection
import strax

export, __all__ = strax.exporter()


class TooMuchDataError(Exception):
    pass


class DownloadError(Exception):
    pass


@export
class RucioFrontend(strax.StorageFrontend):
    """
    Uses the rucio client for the data find.
    """
    local_rses = {'UC_DALI_USERDISK': r'.rcc.'}
    local_did_cache = None
    local_rucio_path = None

    # Some attributes to set if we have the remote backend
    _did_client = None
    _id_not_found_error = None
    _rse_client = None

    def __init__(self,
                 include_remote=False,
                 download_heavy=False,
                 staging_dir='./strax_data',
                 *args, **kwargs):
        """
        :param include_remote: Flag specifying whether or not to allow rucio downloads from remote sites
        :param download_heavy: option to allow downloading of heavy data through RucioRemoteBackend
        :param args: Passed to strax.StorageFrontend
        :param kwargs: Passed to strax.StorageFrontend
        """
        super().__init__(*args, **kwargs)
        self.readonly = True
        self.collection = xent_collection()

        # check if there is a local rse for the host we are running on
        hostname = socket.getfqdn()
        local_rse = None
        for rse, host_regex in self.local_rses.items():
            if re.search(host_regex, hostname):
                if local_rse is not None:
                    raise ValueError(f"The regex {host_regex} matches two RSEs {rse} and {local_rse}. "
                                     f"I'm not sure what to do with that.")
                local_rse = rse

        # if there is no local host and we don't want to include the
        # remote ones, we can't do anything
        if local_rse is None and not include_remote:
            raise RuntimeError(f"Could not find a local RSE for hostname {hostname}, "
                               f"and include_remote is False.")

        self.local_rse = local_rse
        self.include_remote = include_remote

        self.backends = []
        if local_rse:
            # get the rucio prefix for the local rse, and setup strax
            # rucio backend to read from that path
            rucio_prefix = self.get_rse_prefix(local_rse)
            self.backends.append(RucioLocalBackend(rucio_prefix))
            self.local_rucio_path = rucio_prefix

        if include_remote:
            self._set_remote_imports()
            self.backends.append(RucioRemoteBackend(staging_dir, download_heavy=download_heavy))

    def __repr__(self):
        # List the relevant attributes
        attributes = ('include_remote', 'readonly', 'path', 'exclude', 'take_only')
        representation = f'{self.__class__.__module__}.{self.__class__.__name__}'
        for attr in attributes:
            if hasattr(self, attr) and getattr(self, attr):
                representation += f', {attr}: {getattr(self, attr)}'
        return representation

    def _set_remote_imports(self):
        try:
            from rucio.client.rseclient import RSEClient
            from rucio.client.didclient import DIDClient
            from rucio.common.exception import DataIdentifierNotFound
            self._did_client = DIDClient()
            self._id_not_found_error = DataIdentifierNotFound
            self._rse_client = RSEClient()
        except (ModuleNotFoundError, RuntimeError) as e:
            raise ImportError('Cannot work with Rucio remote backend') from e

    def find_several(self, keys, **kwargs):
        if not len(keys):
            return []

        ret = []
        for key in keys:
            did = key_to_rucio_did(key)
            if self.did_is_local(did):
                ret.append(('RucioLocalBackend', did))
            else:
                ret.append(False)
        return ret

    def _find(self, key: strax.DataKey, write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        did = key_to_rucio_did(key)
        if allow_incomplete or write:
            raise RuntimeError(f'Allow incomplete/writing is not allowed for '
                               f'{self.__class.__name__} since data might not be '
                               f'continuous')
        if self.did_is_local(did):
            return "RucioLocalBackend", did
        elif self.include_remote:
            # only do this part if we include the remote backend
            try:
                # check if the DID exists
                scope, name = did.split(':')
                self._did_client.get_did(scope, name)
                return "RucioRemoteBackend", did
            except self._id_not_found_error:
                pass

        if fuzzy_for or fuzzy_for_options:
            matches_to = self._match_fuzzy(key,
                                           fuzzy_for,
                                           fuzzy_for_options)
            if matches_to:
                return matches_to

        raise strax.DataNotAvailable

    def get_rse_prefix(self, rse):
        if self._rse_client is not None:
            rse_info = self._rse_client.get_rse(rse)
            prefix = rse_info['protocols'][0]['prefix']
        elif self.local_rse == 'UC_DALI_USERDISK':
            # If rucio is not loaded but we are on dali, look here:
            prefix = '/dali/lgrandi/rucio/'
        else:
            raise ValueError(f'We are not on dali and cannot load rucio')
        return prefix

    def did_is_local(self, did):
        """
        Determines whether or not a given did is on a local RSE. If
        there is no local RSE, returns False.

        :param did: Rucio DID string
        :return: boolean for whether DID is local or not.
        """
        try:
            md = self._get_backend("RucioLocalBackend").get_metadata(did)
        except (strax.DataNotAvailable, strax.DataCorrupted):
            return False

        return self._all_chunk_stored(md, did)

    def _all_chunk_stored(self, md: dict, did: str) -> bool:
        """
        Check if all the chunks are stored that are claimed in the
        metadata-file
        """
        scope, name = did.split(':')
        for chunk in md.get('chunks', []):
            if chunk.get('filename'):
                _did = f"{scope}:{chunk['filename']}"
                ch_path = rucio_path(self.local_rucio_path, _did)
                if not os.path.exists(ch_path):
                    return False
        return True

    def _match_fuzzy(self,
                     key: strax.DataKey,
                     fuzzy_for: tuple,
                     fuzzy_for_options: tuple,
                     ) -> tuple:
        # fuzzy for local backend
        pattern = os.path.join(self.get_rse_prefix(self.local_rse),
                               f'xnt_{key.run_id}/*/*/{key.data_type}*metadata.json')
        mds = glob.glob(pattern)
        for md in mds:
            md_dict = read_md(md)
            if self._matches(md_dict['lineage'],
                             # Convert lineage dict to json like to compare
                             json.loads(json.dumps(key.lineage, sort_keys=True)),
                             fuzzy_for,
                             fuzzy_for_options):
                fuzzy_lineage_hash = md_dict['lineage_hash']
                did = f'xnt_{key.run_id}:{key.data_type}-{fuzzy_lineage_hash}'
                self.log.warning(f'Was asked for {key} returning {md}')
                if self._all_chunk_stored(md_dict, did):
                    return 'RucioLocalBackend', did


@export
class RucioLocalBackend(strax.FileSytemBackend):
    """Get data from local rucio RSE"""
    def __init__(self, rucio_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rucio_dir = rucio_dir

    def get_metadata(self, did: str, **kwargs):
        scope, name = did.split(':')
        number, dtype, hsh = parse_did(did)
        metadata_json = f'{dtype}-{hsh}-metadata.json'
        metadata_did = f'{scope}:{metadata_json}'

        metadata_path = rucio_path(self.rucio_dir, metadata_did)
        folder = os.path.join('/', *metadata_path.split('/')[:-1])
        if not os.path.exists(folder):
            raise strax.DataNotAvailable(f"No folder for metadata at {metadata_path}")
        if not os.path.exists(metadata_path):
            raise strax.DataCorrupted(f"Folder exists but no metadata at {metadata_path}")

        with open(metadata_path, mode='r') as f:
            return json.loads(f.read())

    def _read_chunk(self, did, chunk_info, dtype, compressor):
        scope, name = did.split(':')
        did = f"{scope}:{chunk_info['filename']}"
        fn = rucio_path(self.rucio_dir, did)
        return strax.load_file(fn, dtype=dtype, compressor=compressor)

    def _saver(self, **kwargs):
        raise NotImplementedError(
            "Cannot save directly into rucio (yet), upload with admix instead")


@export
class RucioRemoteBackend(strax.FileSytemBackend):
    """Get data from remote Rucio RSE"""

    # datatypes we don't want to download since they're too heavy
    heavy_types = ['raw_records', 'raw_records_nv', 'raw_records_he']

    def __init__(self, staging_dir, download_heavy=False, **kwargs):
        """
        :param staging_dir: Path (a string) where to save data. Must be a writable location.
        :param *args: Passed to strax.FileSystemBackend
        :param **kwargs: Passed to strax.FileSystemBackend
        """

        if os.path.exists(staging_dir):
            if not os.access(staging_dir, os.W_OK):
                raise PermissionError(f"You told the rucio backend to download data to {staging_dir}, "
                                      f"but that path is not writable by your user")
        else:
            try:
                os.makedirs(staging_dir)
            except OSError:
                raise PermissionError(f"You told the rucio backend to download data to {staging_dir}, "
                                      f"but that path is not writable by your user")

        super().__init__(**kwargs)
        self.staging_dir = staging_dir
        self.download_heavy = download_heavy
        # Do it only when we actually load rucio
        from rucio.client.downloadclient import DownloadClient
        self.download_client = DownloadClient()

    def get_metadata(self, dset_did, rse='UC_OSG_USERDISK', **kwargs):
        base_dir = os.path.join(self.staging_dir, did_to_dirname(dset_did))

        # define where the metadata will go (or where it already might be)
        number, dtype, hsh = parse_did(dset_did)
        metadata_file = f"{dtype}-{hsh}-metadata.json"
        metadata_path = os.path.join(base_dir, metadata_file)

        # download if it doesn't exist
        if not os.path.exists(metadata_path):
            metadata_did = f'{dset_did}-metadata.json'
            did_dict = dict(did=metadata_did,
                            base_dir=base_dir,
                            no_subdir=True,
                            rse=rse
                            )
            print(f"Downloading {metadata_did}")
            self._download([did_dict])

        # check again
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found at {metadata_path}")

        with open(metadata_path, mode='r') as f:
            return json.loads(f.read())

    def _read_chunk(self, dset_did, chunk_info, dtype, compressor, rse="UC_OSG_USERDISK"):
        base_dir = os.path.join(self.staging_dir, did_to_dirname(dset_did))
        chunk_file = chunk_info['filename']
        chunk_path = os.path.join(base_dir, chunk_file)
        if not os.path.exists(chunk_path):
            number, datatype, hsh = parse_did(dset_did)
            if datatype in self.heavy_types and not self.download_heavy:
                error_msg = ("For space reasons we don't want to have everyone "
                             "downloading raw data. If you know what you're "
                             "doing, pass download_heavy=True to the Rucio "
                             "frontend. If not, check your context and/or ask "
                             "someone if this raw data is needed locally.")
                raise DownloadError(error_msg)
            scope, name = dset_did.split(':')
            chunk_did = f"{scope}:{chunk_file}"
            print(f"Downloading {chunk_did}")
            did_dict = dict(did=chunk_did,
                            base_dir=base_dir,
                            no_subdir=True,
                            rse=rse,
                            )
            self._download([did_dict])

        # check again
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"No chunk file found at {chunk_path}")

        return strax.load_file(chunk_path, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata, **kwargs):
        raise NotImplementedError("Cannot save directly into rucio (yet), upload with admix instead")

    def _download(self, did_dict_list):
        # need to pass a list of dicts
        # let's try 3 times
        success = False
        _try = 1
        while _try <= 3 and not success:
            if _try > 1:
                for did_dict in did_dict_list:
                    did_dict['rse'] = None
            try:
                self.download_client.download_dids(did_dict_list)
                success = True
            except KeyboardInterrupt:
                raise
            except Exception:
                sleep = 3**_try
                print(f"Download try #{_try} failed. Sleeping for {sleep} seconds and trying again...")
                time.sleep(sleep)
            _try += 1
        if not success:
            raise DownloadError(f"Error downloading from rucio.")


class RucioSaver(strax.Saver):
    """
    TODO Saves data to rucio if you are the production user
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError


def rucio_path(root_dir, did):
    """Convert target to path according to rucio convention.
    See the __hash method here: https://github.com/rucio/rucio/blob/1.20.15/lib/rucio/rse/protocols/protocol.py"""
    scope, filename = did.split(':')
    # disable bandit
    rucio_md5 = hashlib.md5(did.encode('utf-8')).hexdigest() # nosec
    t1 = rucio_md5[0:2]
    t2 = rucio_md5[2:4]
    return os.path.join(root_dir, scope, t1, t2, filename)


def parse_did(did):
    """Parses a Rucio DID and returns a tuple of (number:int, dtype:str, hash: str)"""
    scope, name = did.split(':')
    number = int(scope.split('_')[1])
    dtype, hsh = name.split('-')
    return number, dtype, hsh


def did_to_dirname(did):
    """Takes a Rucio dataset DID and returns a dirname like used by strax.FileSystemBackend"""
    # make sure it's a DATASET did, not e.g. a FILE
    if len(did.split('-')) != 2:
        raise RuntimeError(f"The DID {did} does not seem to be a dataset DID. Is it possible you passed a file DID?")
    dirname = did.replace(':', '-').replace('xnt_', '')
    return dirname


def key_to_rucio_did(key: strax.DataKey) -> str:
    """Convert a strax.datakey to a rucio did field in rundoc"""
    return f'xnt_{key.run_id}:{key.data_type}-{key.lineage_hash}'


def key_to_rucio_meta(key: strax.DataKey) -> str:
    return f'{str(key.data_type)}-{key.lineage_hash}-metadata.json'


def read_md(path: str) -> json:
    with open(path, mode='r') as f:
        md = json.loads(f.read(),
                        object_hook=json_util.object_hook)
    return md


def list_datasets(scope):
    from rucio.client.client import Client
    rucio_client = Client()
    datasets = [d for d in rucio_client.list_dids(scope, {'type': 'dataset'}, type='dataset')]
    return datasets
