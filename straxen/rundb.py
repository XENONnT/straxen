import os
import re
import socket

import botocore.client
from tqdm import tqdm
import pymongo

import strax
import straxen
export, __all__ = strax.exporter()


default_mongo_url = (
    'mongodb://{username}:{password}@rundbcluster-shard-00-00-cfaei.'
    'gcp.mongodb.net:27017,rundbcluster-shard-00-01-cfaei.gcp.mongodb.net'
    ':27017,rundbcluster-shard-00-02-cfaei.gcp.mongodb.net:27017/test?'
    'ssl=true&replicaSet=RunDBCluster-shard-0&authSource=admin')
default_mongo_dbname = 'xenon1t'
default_mongo_collname = 'runs'


@export
class RunDB(strax.StorageFrontend):
    """Frontend that searches RunDB MongoDB for data.

    Loads appropriate backends ranging from Files to S3.
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
                 runid_field='name',
                 s3_kwargs=None,
                 local_only=False,
                 new_data_path=None,
                 *args, **kwargs):
        """
        :param mongo_url: URL to Mongo runs database (including auth)
        :param local_only: Do not show data as available if it would have to be
        downloaded from a remote location.
        :param new_data_path: Path where new files are to be written.
            Defaults to None: do not write new data
            New files will be registered in the runs db!
            TODO: register under hostname alias (e.g. 'dali')
        :param s3_kwargs: Arguments to initialize S3 backend (including auth)
        :param runid_field: Rundb field to which strax's run_id concept
            corresponds. Can be either
            - 'name': values must be strings, for XENON1T
            - 'number': values must be ints, for XENONnT DAQ tests

        Other (kw)args are passed to StorageFrontend.__init__

        TODO: disable S3 if secret keys not known
        """
        super().__init__(*args, **kwargs)
        self.local_only = local_only
        self.new_data_path = new_data_path
        if self.new_data_path is None:
            self.readonly = True

        self.runid_field = runid_field

        if self.runid_field not in ['name', 'number']:
            raise ValueError("Unrecognized runid_field option %s" % self.runid_field)

        if s3_kwargs is None:
            s3_kwargs = dict(
                aws_access_key_id=straxen.get_secret('s3_access_key_id'),
                aws_secret_access_key=straxen.get_secret('s3_secret_access_key'),      # noqa
                endpoint_url='http://ceph-s3.mwt2.org',
                service_name='s3',
                config=botocore.client.Config(
                    connect_timeout=5,
                    retries=dict(max_attempts=10)))

        if mongo_url is None:
            mongo_url = default_mongo_url
        self.client = pymongo.MongoClient(mongo_url.format(
            username=straxen.get_secret('rundb_username'),
            password=straxen.get_secret('rundb_password')))

        if mongo_dbname is None:
            mongo_dbname = default_mongo_dbname
        if mongo_collname is None:
            mongo_collname = default_mongo_collname
        self.collection = self.client[mongo_dbname][mongo_collname]

        self.backends = [
            strax.S3Backend(**s3_kwargs),
            strax.FileSytemBackend(),
        ]

        # Construct mongo query for runs with available data.
        # This depends on the machine you're running on.
        self.hostname = socket.getfqdn()
        self.available_query = [{'host': self.hostname}]
        if not self.local_only:
            self.available_query.append({'host': 'ceph-s3'})

        # Go through known host aliases
        for host_alias, regex in self.hosts.items():
            if re.match(regex, self.hostname):
                self.available_query.append({'host': host_alias})

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
            run_query = {'name': key.run_id}
        else:
            run_query = {'number': int(key.run_id)}
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

    def _list_available(self, key: strax.DataKey,
              allow_incomplete, fuzzy_for, fuzzy_for_options):
        if fuzzy_for or fuzzy_for_options:
            raise NotImplementedError("Can't do fuzzy with RunDB yet.")

        dq = self._data_query(key)
        cursor = self.collection.find(
            dq,
            projection=[self.runid_field])
        return [x[self.runid_field] for x in cursor]

    def _scan_runs(self, store_fields):
        cursor = self.collection.find(
            filter={},
            projection=strax.to_str_tuple(
                list(store_fields) + ['reader.ini.name']))
        for doc in tqdm(cursor, desc='Fetching run info from MongoDB',
                        total=cursor.count()):
             # Remove the Mongo document ID and add the run mode
            del doc['_id']
            doc.setdefault('mode',
                           doc.get('reader', {}).get('ini', {}).get('name', ''))
            yield doc

    def run_metadata(self, run_id, projection=None):
        doc = self.collection.find_one({'name': run_id}, projection=projection)
        if doc is None:
            raise strax.DataNotAvailable
        return doc
