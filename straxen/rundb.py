import os
import re
import typing
import socket
from warnings import warn
import botocore.client
from tqdm import tqdm
import pymongo
from copy import deepcopy
import strax
import straxen
export, __all__ = strax.exporter()


default_mongo_url = 'xenon-rundb.grid.uchicago.edu:27017/xenonnt'
backup_mongo_urls = ('fried.rice.edu:27017/xenonnt',
                     'xenon1t-daq.lngs.infn.it:27017/xenonnt')
default_mongo_dbname = 'xenonnt'
default_mongo_collname = 'runs'


@export
class RunDB(strax.StorageFrontend):
    """Frontend that searches RunDB MongoDB for data.
    """
    # Dict of alias used in rundb: regex on hostname
    hosts = {
        'dali': r'^dali.*rcc.*',
    }

    provide_run_metadata = True

    def __init__(self,
                 mongo_url=None,
                 mongo_dbname=None,
                 mongo_collname=None,
                 minimum_run_number=7157,
                 runid_field='name',
                 local_only=False,
                 new_data_path=None,
                 reader_ini_name_is_mode=False,
                 rucio_path=None,
                 *args, **kwargs):
        """
        :param mongo_url: URL to Mongo runs database (including auth)
        :param local_only: Do not show data as available if it would have to be
        downloaded from a remote location.
        :param new_data_path: Path where new files are to be written.
            Defaults to None: do not write new data
            New files will be registered in the runs db!
            TODO: register under hostname alias (e.g. 'dali')
        :param runid_field: Rundb field to which strax's run_id concept
            corresponds. Can be either
            - 'name': values must be strings, for XENON1T
            - 'number': values must be ints, for XENONnT DAQ tests
        :param reader_ini_name_is_mode: If True, will overwrite the 'mode'
        field with 'reader.ini.name'.

        Other (kw)args are passed to StorageFrontend.__init__

        """
        super().__init__(*args, **kwargs)
        self.local_only = local_only
        self.new_data_path = new_data_path
        self.reader_ini_name_is_mode = reader_ini_name_is_mode
        self.minimum_run_number = minimum_run_number
        self.rucio_path = rucio_path
        if self.new_data_path is None:
            self.readonly = True
        self.runid_field = runid_field

        if self.runid_field not in ['name', 'number']:
            raise ValueError("Unrecognized runid_field option %s" % self.runid_field)

        self.hostname = socket.getfqdn()

        if mongo_url is None:
            mongo_url = get_mongo_url(self.hostname)

        self.client = pymongo.MongoClient(mongo_url)

        if mongo_dbname is None:
            mongo_dbname = default_mongo_dbname
        if mongo_collname is None:
            mongo_collname = default_mongo_collname
        self.collection = self.client[mongo_dbname][mongo_collname]

        self.backends = [
            strax.FileSytemBackend(),
        ]

        # Construct mongo query for runs with available data.
        # This depends on the machine you're running on.
        self.available_query = [{'host': self.hostname}]

        # Go through known host aliases
        for host_alias, regex in self.hosts.items():
            if re.match(regex, self.hostname):
                self.available_query.append({'host': host_alias})

        if self.rucio_path is not None:
            self.backends.append(strax.rucio(self.rucio_path))
            # When querying for rucio, add that it should be dali-userdisk
            self.available_query.append({'host': 'rucio-catalogue',
                                         'location': 'UC_DALI_USERDISK'})

    def _data_query(self, key):
        """Return MongoDB query for data field matching key"""
        return {
            'data': {
                '$elemMatch': {
                    'type': key.data_type,
                    'meta.lineage': key.lineage,
                    '$or': self.available_query}}}

    def _find(self, key: strax.DataKey,
              write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        if fuzzy_for or fuzzy_for_options:
            raise NotImplementedError("Can't do fuzzy with RunDB yet.")

        # Check if the run exists
        if self.runid_field == 'name':
            run_query = {'name': str(key.run_id)}
        else:
            run_query = {'number': int(key.run_id)}

        # Check that we are in rucio backend
        if self.rucio_path is not None:
            rucio_key = self.key_to_rucio_did(key)
            dq = {
                'data': {
                    '$elemMatch': {
                        # TODO can we query smart on the lineage_hash?
                        'type': key.data_type,
                        'did': rucio_key,
                        'protocol': 'rucio'}}}
            doc = self.collection.find_one({**run_query, **dq},
                                           projection=dq)
            if doc is not None:
                datum = doc['data'][0]
                assert datum.get('did', '') == rucio_key, f'Expected {rucio_key} got data on {datum["location"]}'
                backend_name, backend_key = datum['protocol'], f'{key.run_id}-{key.data_type}-{key.lineage_hash}'
                return backend_name, backend_key

        dq = self._data_query(key)
        doc = self.collection.find_one({**run_query, **dq},
                                       projection=dq)

        if doc is None:
            # Data was not found
            if not write:
                raise strax.DataNotAvailable

            output_path = os.path.join(self.new_data_path, str(key))

            if self.new_data_path is not None:
                doc = self.collection.find_one(run_query, projection={'_id'})
                if not doc:
                    raise ValueError(f"Attempt to register new data for non-existing run {key.run_id}")   # noqa
                self.collection.find_one_and_update(
                    {'_id': doc['_id']},
                    {'$push': {'data': {
                        'location': output_path,
                        'host': self.hostname,
                        'type': key.data_type,
                        'protocol': strax.FileSytemBackend.__name__,
                        # TODO: duplication with metadata stuff elsewhere?
                        'meta': {'lineage': key.lineage}
                    }}})

            return (strax.FileSytemBackend.__name__,
                    output_path)

        datum = doc['data'][0]

        if write and not self._can_overwrite(key):
            raise strax.DataExistsError(at=datum['location'])

        return datum['protocol'], datum['location']

    def find_several(self, keys: typing.List[strax.DataKey], **kwargs):
        if kwargs['fuzzy_for'] or kwargs['fuzzy_for_options']:
            raise NotImplementedError("Can't do fuzzy with RunDB yet.")
        if not len(keys):
            return []
        if not len(set([k.lineage_hash for k in keys])) == 1:
            raise ValueError("find_several keys must have same lineage")
        if not len(set([k.data_type for k in keys])) == 1:
            raise ValueError("find_several keys must have same data type")
        keys = list(keys)   # Context used to pass a set

        if self.runid_field == 'name':
            run_query = {'name': {'$in': [key.run_id for key in keys]}}
        else:
            run_query = {f'{self.runid_field}': {'$in': [int(key.run_id) for key in keys]}}
        dq = self._data_query(keys[0])

        # dict.copy is sometimes not sufficient for nested dictionary
        projection = deepcopy(dq)
        projection.update({
            k: True
            for k in f'name number'.split()})

        results_dict = dict()
        for doc in self.collection.find(
                {**run_query, **dq}, projection=projection):
            # If you get a key error here there might be something off with the
            # projection
            datum = doc['data'][0]

            if self.runid_field == 'name':
                dk = doc['name']
            else:
                dk = f'{doc["number"]:06}'

            results_dict[dk] = datum['protocol'], datum['location']
        return [results_dict.get(k.run_id, False)
                for k in keys]

    def _list_available(self, key: strax.DataKey,
              allow_incomplete, fuzzy_for, fuzzy_for_options):
        if fuzzy_for or fuzzy_for_options:
            raise NotImplementedError("Can't do fuzzy with RunDB yet.")
        if allow_incomplete:
            raise NotImplementedError("Can't allow_incomplete with RunDB yet")

        q = self._data_query(key)
        if self.minimum_run_number:
            q['number'] = {'$gt': self.minimum_run_number}
        cursor = self.collection.find(
            q,
            projection=[self.runid_field])
        return [x[self.runid_field] for x in cursor]

    def _scan_runs(self, store_fields):
        if self.minimum_run_number:
            query = {'number': {'$gt': self.minimum_run_number}}
        else:
            query = {}
        projection = strax.to_str_tuple(list(store_fields))
        # Replace fields by their subfields if requested only take the most
        # "specific" projection
        projection = [f1 for f1 in projection
                      if not any([f2.startswith(f1+".") for f2 in projection])]
        cursor = self.collection.find(
            filter=query,
            projection=projection)
        for doc in tqdm(cursor, desc='Fetching run info from MongoDB',
                        total=cursor.count()):
            del doc['_id']
            if self.reader_ini_name_is_mode:
                doc['mode'] = \
                    doc.get('reader', {}).get('ini', {}).get('name', '')
            yield doc

    def run_metadata(self, run_id, projection=None):
        if self.runid_field == 'name':
            run_id = str(run_id)
        else:
            run_id = int(run_id)
        if isinstance(projection, str):
            projection = {projection: 1}
        elif isinstance(projection, (list, tuple)):
            projection = {x: 1 for x in projection}

        doc = self.collection.find_one(
            {self.runid_field: run_id},
            projection=projection)
        if doc is None:
            raise strax.DataNotAvailable
        if self.reader_ini_name_is_mode:
            doc['mode'] = doc.get('reader', {}).get('ini', {}).get('name', '')
        return doc

    @staticmethod
    def key_to_rucio_did(key: strax.DataKey):
        """Convert a strax.datakey to a rucio did field in rundoc"""
        return f'xnt_{key.run_id}:{key.data_type}-{key.lineage_hash}'


def get_mongo_url(hostname):
    """
    Read url for mongo by reading the username and password from
    straxen.get_secret.

    :param hostname: The name of the host currently working on. If
    this is an event-builder, we can use the gateway to
    authenticate. Else we use either of the hosts in
    default_mongo_url and backup_mongo_urls.
    """
    if hostname.endswith('xenon.local'):
        # So we are running strax on an event builder
        username = straxen.get_secret('mongo_rdb_username')
        password = straxen.get_secret('mongo_rdb_password')
        url_base = 'xenon1t-daq:27017'
        mongo_url = f"mongodb://{username}:{password}@{url_base}"
    else:
        username = straxen.get_secret('rundb_username')
        password = straxen.get_secret('rundb_password')

        # try connection to the mongo database in this order
        mongo_connections = [default_mongo_url, *backup_mongo_urls]
        for url_base in mongo_connections:
            try:
                mongo_url = f"mongodb://{username}:{password}@{url_base}"
                # Force server timeout if we cannot connect ot this url. If this
                # does not raise an error, break and use this url
                pymongo.MongoClient(mongo_url).server_info()
                break
            except pymongo.errors.ServerSelectionTimeoutError:
                warn(f'Cannot connect to to Mongo url: {url_base}')
                if url_base == mongo_connections[-1]:
                    raise pymongo.errors.ServerSelectionTimeoutError(
                        'Cannot connect to any Mongo url')
    return mongo_url
