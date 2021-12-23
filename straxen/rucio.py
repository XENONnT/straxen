import glob
import hashlib
import json
import os
import re
import socket
from warnings import warn
import numpy as np
import strax
from bson import json_util
from utilix import xent_collection

try:
    import admix
    from rucio.common.exception import DataIdentifierNotFound
    HAVE_ADMIX = True
except ImportError:
    HAVE_ADMIX = False


export, __all__ = strax.exporter()


class TooMuchDataError(Exception):
    pass


@export
class RucioLocalFrontend(strax.StorageFrontend):
    """
    Storage that loads from rucio by assuming the rucio file naming
    convention without access to the rucio database.

    Normally, you don't need this StorageFrontend as it should return
    the same data as the RunDB frontend
    """
    storage_type = strax.StorageType.LOCAL
    # Todo fix duplication
    local_prefixes = {
        'UC_DALI_USERDISK': '/dali/lgrandi/rucio/',
        'SDSC_USERDISK': '/expanse/lustre/projects/chi135/shockley/rucio',
    }

    def __init__(self, *args, **kwargs):

        # check if there is a local rse for the host we are running on
        hostname = socket.getfqdn()
        local_rse = None
        for rse, host_regex in self.local_rses.items():
            if re.search(host_regex, hostname):
                if local_rse is not None:
                    raise ValueError(f"The regex {host_regex} matches two RSEs {rse} and {local_rse}. "
                                     f"I'm not sure what to do with that.")
                local_rse = rse
        super().__init__(*args, **kwargs)
        # This frontend is naive, neither smart nor flexible
        self.readonly = True
        self.path = self.local_prefixes[local_rse]
        self.backends = [RucioLocalBackend(self.path)]
    # end TODO

    def _find(self, key: strax.DataKey, write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        did = key_to_rucio_did(key)
        if allow_incomplete or write:
            raise RuntimeError(f'Allow incomplete/writing is not allowed for '
                               f'{self.__class.__name__} since data might not be '
                               f'continuous')
        if self.did_is_local(did):
            return self.backends[0].__class__.__name__, did

        if fuzzy_for or fuzzy_for_options:
            matches_to = self._match_fuzzy(key,
                                           fuzzy_for,
                                           fuzzy_for_options)
            if matches_to:
                return matches_to

        raise strax.DataNotAvailable

    def did_is_local(self, did):
        """
        Determines whether or not a given did is on a local RSE. If
        there is no local RSE, returns False.

        :param did: Rucio DID string
        :return: boolean for whether DID is local or not.
        """
        try:
            md = self._get_backend("RucioLocalBackend").get_metadata(did)
        except (strax.DataNotAvailable, strax.DataCorrupted, KeyError):
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
                ch_path = rucio_path(self.path, _did)
                if not os.path.exists(ch_path):
                    return False
        return True

    def _match_fuzzy(self,
                     key: strax.DataKey,
                     fuzzy_for: tuple,
                     fuzzy_for_options: tuple,
                     ) -> tuple:
        pattern = os.path.join(
            self.get_rse_prefix(self.local_rse),
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
                    return self.backends[0].__class__.__name__, did


@export
class RucioRemoteFrontend(strax.StorageFrontend):
    """
    Uses the rucio client for the data find.
    """

    storage_type = strax.StorageType.REMOTE
    local_rses = {'UC_DALI_USERDISK': r'.rcc.',
                  'SDSC_USERDISK': r'.sdsc.'
                  }
    local_did_cache = None
    path = None
    local_prefixes = {'UC_DALI_USERDISK': '/dali/lgrandi/rucio/',
                      'SDSC_USERDISK': '/expanse/lustre/projects/chi135/shockley/rucio',
                      }

    def __init__(self,
                 include_remote=True,
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

        self.local_rse = local_rse
        self.include_remote = include_remote

        self.backends = []

        if not include_remote:
            raise ValueError
        if HAVE_ADMIX:
            self.backends = [
                RucioRemoteBackend(staging_dir, download_heavy=download_heavy),
            ]
        else:
            self.log.warning("You passed use_remote=True to rucio fronted, "
                             "but you don't have access to admix/rucio! Using local backed only.")

    def find_several(self, keys, **kwargs):
        # for performance, dont do find_several with this storage frontend
        # we basically do the same query we would do in the RunDB plugin
        return np.zeros_like(keys, dtype=bool).tolist()

    def _find(self, key: strax.DataKey, write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        did = key_to_rucio_did(key)
        if allow_incomplete or write:
            raise RuntimeError(f'Allow incomplete/writing is not allowed for '
                               f'{self.__class.__name__} since data might not be '
                               f'continuous')
        try:
            rules = admix.rucio.list_rules(did, state="OK")
            if len(rules):
                return "RucioRemoteBackend", did
        except DataIdentifierNotFound:
            pass

        raise strax.DataNotAvailable

    def find(self,
             key: strax.DataKey,
             write=False,
             check_broken=False,
             **kwargs):
        # Overwrite defaults of super().find()
        return super().find(key, write, check_broken, **kwargs)

    def get_rse_prefix(self, rse):
        return admix.rucio.get_rse_prefix(rse)


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

    # for caching RSE locations
    dset_cache = {}

    def __init__(self, staging_dir, download_heavy=False, **kwargs):
        """
        :param staging_dir: Path (a string) where to save data. Must be
            a writable location.
        :param download_heavy: Whether or not to allow downloads of the
            heaviest data (raw_records*, less aqmon and MV)
        :param **kwargs: Passed to strax.FileSystemBackend
        """
        mess = (f"You told the rucio backend to download data to {staging_dir}, "
                f"but that path is not writable by your user")
        if os.path.exists(staging_dir):
            if not os.access(staging_dir, os.W_OK):

                raise PermissionError(mess)
        else:
            try:
                os.makedirs(staging_dir)
            except OSError:
                raise PermissionError(mess)
        super().__init__(**kwargs)
        self.staging_dir = staging_dir
        self.download_heavy = download_heavy

    def _get_metadata(self, dset_did, **kwargs):
        if dset_did in self.dset_cache:
            rse = self.dset_cache[dset_did]
        else:
            rses = admix.rucio.get_rses(dset_did)
            rse = admix.downloader.determine_rse(rses)
            self.dset_cache[dset_did] = rse

        metadata_did = f'{dset_did}-metadata.json'
        downloaded = admix.download(metadata_did, rse=rse, location=self.staging_dir)
        if len(downloaded) != 1:
            raise ValueError(f"{metadata_did} should be a single file. "
                             f"We found {len(downloaded)}.")
        metadata_path = downloaded[0]
        # check again
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"No metadata found at {metadata_path}")

        with open(metadata_path, mode='r') as f:
            return json.loads(f.read())

    def _read_chunk(self, dset_did, chunk_info, dtype, compressor):
        base_dir = os.path.join(self.staging_dir, did_to_dirname(dset_did))
        chunk_file = chunk_info['filename']
        chunk_path = os.path.abspath(os.path.join(base_dir, chunk_file))
        if not os.path.exists(chunk_path):
            number, datatype, hsh = parse_did(dset_did)
            if datatype in self.heavy_types and not self.download_heavy:
                error_msg = ("For space reasons we don't want to have everyone "
                             "downloading raw data. If you know what you're "
                             "doing, pass download_heavy=True to the Rucio "
                             "frontend. If not, check your context and/or ask "
                             "someone if this raw data is needed locally.")
                warn(error_msg)
                raise strax.DataNotAvailable
            scope, name = dset_did.split(':')
            chunk_did = f"{scope}:{chunk_file}"
            if dset_did in self.dset_cache:
                rse = self.dset_cache[dset_did]
            else:
                rses = admix.rucio.get_rses(dset_did)
                rse = admix.downloader.determine_rse(rses)
                self.dset_cache[dset_did] = rse

            downloaded = admix.download(chunk_did, rse=rse, location=self.staging_dir)
            if len(downloaded) != 1:
                raise ValueError(f"{chunk_did} should be a single file. "
                                 f"We found {len(downloaded)}.")
            assert chunk_path == downloaded[0]

        # check again
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"No chunk file found at {chunk_path}")

        return strax.load_file(chunk_path, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata, **kwargs):
        raise NotImplementedError("Cannot save directly into rucio (yet), "
                                  "upload with admix instead")


class RucioSaver(strax.Saver):
    """
    TODO Saves data to rucio if you are the production user
    """
    def __init__(self, *args, **kwargs):
        raise NotImplementedError


def rucio_path(root_dir, did):
    """
    Convert target to path according to rucio convention.
    See the __hash method here:
    https://github.com/rucio/rucio/blob/1.20.15/lib/rucio/rse/protocols/protocol.py
    """
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


def read_md(path: str) -> json:
    with open(path, mode='r') as f:
        md = json.loads(f.read(),
                        object_hook=json_util.object_hook)
    return md
