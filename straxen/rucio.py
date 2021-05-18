import socket
import re
from tqdm import tqdm
import json
import os
import hashlib
from rucio.client.client import Client
from rucio.client.rseclient import RSEClient
from rucio.client.replicaclient import ReplicaClient
from rucio.client.downloadclient import DownloadClient
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

    def __init__(self, include_remote=False,
                 staging_dir='./strax_data',
                 minimum_run_number=7157,
                 runs_to_consider=None,
                 *args, **kwargs):
        """
        :param include_remote: Flag specifying whether or not to allow rucio downloads from remote sites (TODO: not implemented yet)
        :param minimum_run_number: only consider run numbers larger than this
        :param runs_to_consider: list of runs to consider, so we don't automatically search for ALL data.
        :param args: Passed to strax.StorageFrontend
        :param kwargs: Passed to strax.StorageFrontend
        """
        super().__init__(*args, **kwargs)
        # initialize rucio clients
        # TODO eventually just have admix calls for this?
        self.rucio_client = Client()
        self.rse_client = RSEClient()
        self.replica_client = ReplicaClient()
        self.collection = xent_collection()

        # check if there is a local rse for the host we are running on
        hostname = socket.getfqdn()
        local_rse = None
        for rse, host_regex in self.local_rses.items():
            if re.search(host_regex, hostname):
                if local_rse is None:
                    local_rse = rse
                else:
                    raise ValueError(f"The regex {host_regex} matches two RSEs {rse} and {local_rse}. "
                                     f"I'm not sure what to do with that.")

        # if there is no local host and we don't want to include the remote ones, we can't do anything
        if local_rse is None and not include_remote:
            raise RuntimeError(f"Could not find a local RSE for hostname {hostname}, and include_remote is False.")

        # get the rucio prefix for the local rse, and setup strax rucio backend to read from that path

        self.backends = []
        if local_rse:
            rucio_prefix = self.get_rse_prefix(local_rse)
            self.backends.append(RucioLocalBackend(rucio_prefix))

        self.local_rse = local_rse

        if include_remote:
            self.backends.append(RucioRemoteBackend(staging_dir))

        # find run numbers to consider
        if runs_to_consider:
            self.runs_to_consider = runs_to_consider
        else:
            if minimum_run_number is None:
                minimum_run_number = 0
            query = {'number': {'$gte': minimum_run_number}}
            self.runs_to_consider = [r['number'] for r in self.collection.find(query, {'number': 1})]

    def _scan_runs(self, store_fields):
        if self.local_rse is None:
            # Don't allow this to be called since it will just loop over all data.
            raise TooMuchDataError("We don't want use the _scan_runs for the remote rucio backed "
                                   "as it will just return all our data. ")
        else:
            if self.local_did_cache is None:
                datasets = self.get_rse_datasets(self.local_rse)
                self.local_did_cache = datasets

            # sometimes there's crap in rucio from testing days, so lets do a query to find the real TPC data there
            # plus we will have MC data in rucio as well at some point
            query = {'data.did': {'$in': datasets}}
            projection = {field: 1 for field in store_fields}
            # don't care about the _id
            projection['_id'] = 0

            cursor = self.collection.find(query, projection=projection)
            for doc in cursor:
                yield doc

    def find_several(self, keys, **kwargs):
        if kwargs['fuzzy_for'] or kwargs['fuzzy_for_options']:
            raise NotImplementedError("Can't do fuzzy with RunDB yet.")
        if not len(keys):
            return []

        if self.local_did_cache is None:
            datasets = self.get_rse_datasets(self.local_rse)
            self.local_did_cache = datasets

        ret = []
        for key in keys:
            did = self.key_to_rucio_did(key)
            backend_key = f'{key.run_id}-{key.data_type}-{key.lineage_hash}'
            if did in self.local_did_cache and self.did_is_local(did):
                ret.append((strax.rucio.__name__, backend_key))
            else:
                ret.append(False)
        return ret

    def _find(self, key: strax.DataKey, write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        if fuzzy_for or fuzzy_for_options:
            raise NotImplementedError("Can't do fuzzy with rucio yet.")

        did = self.key_to_rucio_did(key)
        if self.did_is_local(did):
            backend_key = f'{key.run_id}-{key.data_type}-{key.lineage_hash}'
            return "RucioLocalBackend", backend_key
        else:
            return "RucioRemoteBackend", did

    def get_rse_prefix(self, rse):
        """TODO will eventually do an admix call here"""
        rse_info = self.rse_client.get_rse(rse)
        prefix = rse_info['protocols'][0]['prefix']
        return prefix

    def did_is_local(self, did):
        """
        Determines whether or not a given did is on a local RSE. If there is no local RSE, returns False.
        :param did: Rucio DID string
        :return: boolean for whether DID is local or not.
        """
        if self.local_rse is None:
            return False
        rules = self.list_rules(did, rse_expression=self.local_rse, state='OK')
        if len(rules):
            return True
        return False

    def list_rules(self, did, **filters):
        """
        Fetches list of replication rules using the rucio client.
        :param did: Rucio DID string
        :param filters: Kwargs that allow for selecting only certain rules.
        For example, rse_expression={rse} will return only the rules at the RSE {rse}. Another useful one is state='OK'
        :return: List of dictionaries with replication rule information.
        """
        scope, name = did.split(':')
        rules = self.rucio_client.list_did_rules(scope, name)
        # get rules that pass some filter(s)
        ret = []
        for rule in rules:
            selected = True
            for key, val in filters.items():
                if rule[key] != val:
                    selected = False
            if selected:
                ret.append(rule)
        return ret

    def get_rse_datasets(self, rse):
        datasets = self.replica_client.list_datasets_per_rse(rse)
        ret = []
        for d in tqdm(datasets, desc=f'Finding all datasets at {rse}'):
            try:
                number = int(d['scope'].split('_')[1])
            except (ValueError, IndexError):
                continue
            if number in self.runs_to_consider:
                did = f"{d['scope']}:{d['name']}"
                ret.append(did)
        return ret

    @staticmethod
    def key_to_rucio_did(key: strax.DataKey):
        """Convert a strax.datakey to a rucio did field in rundoc"""
        return f'xnt_{key.run_id}:{key.data_type}-{key.lineage_hash}'


@export
class RucioLocalBackend(strax.FileSytemBackend):
    """Get data from local rucio RSE"""
    def __init__(self, rucio_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rucio_dir = rucio_dir

    def get_metadata(self, dirname:str, **kwargs):
        prefix = strax.dirname_to_prefix(dirname)
        metadata_json = f'{prefix}-metadata.json'
        fn = rucio_path(self.rucio_dir, metadata_json, dirname)
        folder = os.path.join('/', *fn.split('/')[:-1])
        if not os.path.exists(folder):
            raise strax.DataNotAvailable(f"No folder for metadata at {fn}")
        if not os.path.exists(fn):
            raise strax.DataCorrupted(f"Folder exists but no metadata at {fn}")

        with open(fn, mode='r') as f:
            return json.loads(f.read())

    def _read_chunk(self, dirname, chunk_info, dtype, compressor):
        fn = rucio_path(self.rucio_dir, chunk_info['filename'], dirname)
        return strax.load_file(fn, dtype=dtype, compressor=compressor)

    def _saver(self, **kwargs):
        raise NotImplementedError(
            "Cannot save directly into rucio (yet), upload with admix instead")


@export
class RucioRemoteBackend(strax.FileSytemBackend):
    """Get data from remote Rucio RSE"""
    def __init__(self, staging_dir, *args, **kwargs):
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

        super().__init__(*args, **kwargs)
        self.staging_dir = staging_dir
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
            scope, name = dset_did.split(':')
            chunk_did = f"{scope}:{chunk_file}"
            print(f"Downloading {chunk_did}")
            did_dict = dict(did=chunk_did,
                            base_dir=base_dir,
                            no_subdir=True,
                            rse=rse
                            )
            self._download([did_dict])

        # check again
        if not os.path.exists(chunk_path):
            raise FileNotFoundError(f"No chunk file found at {chunk_path}")

        return strax.load_file(chunk_path, dtype=dtype, compressor=compressor)

    def _saver(self, dirname, metadata):
        raise NotImplementedError("Cannot save directly into rucio (yet), upload with admix instead")

    def _download(self, did_dict_list):
        # need to pass a list of dicts
        # let's try 3 times
        success = False
        _try = 1
        while _try <= 3 and not success:
            if _try > 1:
                did_dict_list['rse'] = None
            try:
                self.download_client.download_dids(did_dict_list)
                success = True
            except KeyboardInterrupt:
                raise
            except:
                sleep = 3**_try
                print(f"Download try #{_try} failed. Sleeping for {sleep} seconds and trying again...")
            _try += 1
        if not success:
            raise DownloadError(f"Error downloading from rucio.")


class RucioSaver(strax.Saver):
    """
    TODO Saves data to rucio if you are the production user
    """

    def __init__(self):
        raise NotImplementedError


def rucio_path(root_dir, filename, dirname):
    """Convert target to path according to rucio convention.
    See the __hash method here: https://github.com/rucio/rucio/blob/1.20.15/lib/rucio/rse/protocols/protocol.py"""
    scope = "xnt_"+dirname.split('-')[0]
    rucio_did = "{0}:{1}".format(scope, filename)
    rucio_md5 = hashlib.md5(rucio_did.encode('utf-8')).hexdigest()
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
