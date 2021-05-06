import socket
import re
from tqdm import tqdm
from rucio.client.client import Client
from rucio.client.rseclient import RSEClient
from rucio.client.replicaclient import ReplicaClient
from utilix import xent_collection
import strax

export, __all__ = strax.exporter()


@export
class RucioFrontend(strax.StorageFrontend):
    local_rses = {'UC_DALI_USERDISK': r'.rcc.'}
    local_did_cache = None

    def __init__(self, include_remote=False,
                 minimum_run_number=7157,
                 runs_to_consider=None,
                 *args, **kwargs):
        """
        :param include_remote: Flag specifying whether or not to allow rucio downloads from remote sites
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        # initialize rucio clients
        # TODO eventually will just have admix calls for this
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
        self.backends = [strax.rucio(self.get_rse_prefix(local_rse))]

        self.local_rse = local_rse

        if include_remote:
            # TODO add remote rucio backend?
            # TODO Or maybe this should become an option in the already-existing rucio backend
            raise NotImplementedError

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
            # TODO do we really need to scan over all the data that we have? This is resource-intensive for little gain
            raise NotImplementedError
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
        for k in keys:
            did = self.key_to_rucio_did(k)
            backend_key = f'{k.run_id}-{k.data_type}-{k.lineage_hash}'
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
            return strax.rucio.__name__, backend_key
        else:
            # TODO download data from other RSE
            raise strax.DataNotAvailable

    def get_rse_prefix(self, rse):
        """TODO will eventually do an admix call here"""
        rse_info = self.rse_client.get_rse(rse)
        prefix = rse_info['protocols'][0]['prefix']
        return prefix

    def did_is_local(self, did):
        if self.local_rse is None:
            return False
        rules = self.list_rules(did, rse_expression=self.local_rse, state='OK')
        if len(rules) > 0:
            return True
        else:
            return False

    def list_rules(self, did, **filters):
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
