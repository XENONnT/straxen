import glob
import hashlib
import json
import os
import re
import socket
import warnings

import strax
from bson import json_util
from .rucio_remote import key_to_rucio_did, parse_rucio_did

export, __all__ = strax.exporter()


@export
class RucioLocalFrontend(strax.StorageFrontend):
    """
    Storage that loads from rucio by assuming the rucio file naming
    convention without access to the rucio database.

    Normally, you don't need this StorageFrontend as it should return
    the same data as the RunDB frontend
    """
    storage_type = strax.StorageType.LOCAL
    local_prefixes = {'UC_DALI_USERDISK': '/dali/lgrandi/rucio/',
                      'SDSC_USERDISK': '/expanse/lustre/projects/chi135/shockley/rucio',
                      }
    local_rses = {'UC_DALI_USERDISK': r'.rcc.',
                  'SDSC_USERDISK': r'.sdsc.'
                  }

    def __init__(self, rucio_dir=None, *args, **kwargs):
        kwargs.setdefault('readonly', True)
        super().__init__(*args, **kwargs)
        if rucio_dir is None:
            local_rse = self.determine_rse()
            self.path = self.local_prefixes[local_rse]
        else:
            self.path = rucio_dir
        self.backends = [RucioLocalBackend(self.path)]

    def determine_rse(self):
        # check if there is a local rse for the host we are running on
        hostname = socket.getfqdn()
        local_rse = None
        for rse, host_regex in self.local_rses.items():
            if re.search(host_regex, hostname):
                if local_rse is not None:
                    raise ValueError(
                        f"The regex {host_regex} matches two RSEs {rse} and"
                        f" {local_rse}. I'm not sure what to do with that.")
                local_rse = rse
        return local_rse

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
            self.path,
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
                warnings.warn(f'Was asked for {key} returning {md}', UserWarning)
                if self._all_chunk_stored(md_dict, did):
                    return self.backends[0].__class__.__name__, did


@export
class RucioLocalBackend(strax.FileSytemBackend):
    """Get data from local rucio RSE"""

    def __init__(self, rucio_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rucio_dir = rucio_dir

    def get_metadata(self, did: str, **kwargs):
        scope, name = did.split(':')
        number, dtype, hsh = parse_rucio_did(did)
        metadata_json = f'{dtype}-{hsh}-metadata.json'
        metadata_did = f'{scope}:{metadata_json}'

        metadata_path = rucio_path(self.rucio_dir, metadata_did)
        folder = os.path.split(metadata_path)[0]
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


def rucio_path(root_dir, did):
    """
    Convert target to path according to rucio convention.
    See the __hash method here:
    https://github.com/rucio/rucio/blob/1.20.15/lib/rucio/rse/protocols/protocol.py
    """
    scope, filename = did.split(':')
    # disable bandit
    rucio_md5 = hashlib.md5(did.encode('utf-8')).hexdigest()  # nosec
    t1 = rucio_md5[0:2]
    t2 = rucio_md5[2:4]
    return os.path.join(root_dir, scope, t1, t2, filename)


def read_md(path: str) -> json:
    with open(path, mode='r') as f:
        md = json.loads(f.read(),
                        object_hook=json_util.object_hook)
    return md
