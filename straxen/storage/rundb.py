import os
import re
import typing
import socket
from typing import Dict, Union, Iterator
from copy import deepcopy
import strax
from .rucio_remote import key_to_rucio_did
from .rucio_local import RucioLocalBackend

import warnings

try:
    import utilix
except (RuntimeError, FileNotFoundError):
    # We might be on a travis job
    pass
from straxen import uconfig

export, __all__ = strax.exporter()


@export
class RunDB(strax.StorageFrontend):
    """Frontend that searches RunDB MongoDB for data."""

    storage_type = strax.StorageType.LOCAL
    # Dict of alias used in rundb: regex on hostname
    hosts = {
        "dali": r"^dali.*rcc.*|fried.rice.edu",
        "midway": r"^midway2.*rcc.*|^midway.*rcc.*",
    }
    userdisks = {
        "UC_DALI_USERDISK": r"^dali.*rcc.*|fried.rice.edu",
        "UC_MIDWAY_USERDISK": r"^midway2.*rcc.*|^midway.*rcc.*",
    }

    provide_run_metadata = True
    progress_bar = False

    def __init__(
        self,
        minimum_run_number=7157,
        maximum_run_number=None,
        runid_field="name",
        local_only=False,
        new_data_path=None,
        reader_ini_name_is_mode=False,
        rucio_path=None,
        mongo_url=None,
        mongo_user=None,
        mongo_password=None,
        mongo_database=None,
        *args,
        **kwargs,
    ):
        """
        :param minimum_run_number: Lowest number to consider
        :param maximum_run_number: Highest number to consider. When None
            (the default) consider all runs that are higher than the
            minimum_run_number.
        :param runid_field: Rundb field to which strax's run_id concept
            corresponds. Can be either
            - 'name': values must be strings, for XENON1T
            - 'number': values must be ints, for XENONnT DAQ tests
        :param local_only: Do not show data as available if it would
            have to be downloaded from a remote location.
        :param new_data_path: Path where new files are to be written.
            Defaults to None: do not write new data
            New files will be registered in the runs db!
        :param reader_ini_name_is_mode: If True, will overwrite the
            'mode' field with 'reader.ini.name'.
        :param rucio_path: What is the base path where Rucio is mounted
        :param mongo_url: URL to Mongo runs database (excl auth)
        :param mongo_user: user to Mongo runs database
        :param mongo_password: password to Mongo runs database
        :param mongo_database: database name of Mongo runs database

        Other (kw)args are passed to StorageFrontend.__init__
        """
        super().__init__(*args, **kwargs)
        self.local_only = local_only
        self.new_data_path = new_data_path
        self.reader_ini_name_is_mode = reader_ini_name_is_mode
        self.minimum_run_number = minimum_run_number
        self.maximum_run_number = maximum_run_number
        self.rucio_path = rucio_path
        if self.new_data_path is None:
            self.readonly = True
        self.runid_field = runid_field

        if self.runid_field not in ["name", "number"]:
            raise ValueError("Unrecognized runid_field option %s" % self.runid_field)

        self.hostname = socket.getfqdn()
        if not self.readonly and self.hostname.endswith("xenon.local"):
            # We want admin access to start writing data!
            mongo_url = uconfig.get("rundb_admin", "mongo_rdb_url")
            mongo_user = uconfig.get("rundb_admin", "mongo_rdb_username")
            mongo_password = uconfig.get("rundb_admin", "mongo_rdb_password")
            mongo_database = uconfig.get("rundb_admin", "mongo_rdb_database")

        # setup mongo kwargs...
        # utilix.rundb.pymongo_collection will take the following variables as kwargs
        #     url: mongo url, including auth
        #     user: the user
        #     password: the password for the above user
        #     database: the mongo database name
        # finally, it takes the collection name as an arg (not a kwarg).
        # if no collection arg is passed, it defaults to the runsDB collection
        # See github.com/XENONnT/utilix/blob/master/utilix/rundb.py for more details
        mongo_kwargs = {
            "url": mongo_url,
            "user": mongo_user,
            "password": mongo_password,
            "database": mongo_database,
        }
        self.collection = utilix.rundb.xent_collection(**mongo_kwargs)

        # Do not delete the client!
        self.client = self.collection.database.client

        self.backends = [
            strax.FileSytemBackend(),
        ]

        # Construct mongo query for runs with available data.
        # This depends on the machine you're running on.
        self.available_query = [{"host": self.hostname}]

        # Go through known host aliases
        for host_alias, regex in self.hosts.items():
            if re.match(regex, self.hostname):
                self.available_query.append({"host": host_alias})

        # When querying for rucio, add that it should be dali-userdisk (when on dali)
        if self.rucio_path is not None and any(
            re.match(regex, self.hostname) for regex in self.hosts.values()
        ):
            for host_alias, regex in self.userdisks.items():
                if re.match(regex, self.hostname):
                    self.backends.append(RucioLocalBackend(self.rucio_path))
                    self.available_query.append(
                        {
                            "host": "rucio-catalogue",
                            "location": host_alias,
                            "status": "transferred",
                        }
                    )

    def _data_query(self, key):
        """Return MongoDB query for data field matching key."""
        return {
            "data": {
                "$elemMatch": {
                    "type": key.data_type,
                    "$and": [
                        {
                            "$or": [
                                {"meta.lineage": key.lineage},
                                {
                                    "did": {"$regex": f"/*{key.data_type}-{key.lineage_hash}"},
                                },
                            ]
                        },
                        {"$or": self.available_query},
                    ],
                }
            }
        }

    def _find(self, key: strax.DataKey, write, allow_incomplete, fuzzy_for, fuzzy_for_options):
        if key.run_id.startswith("_"):
            # Superruns are currently not supprorted..
            raise strax.DataNotAvailable

        if fuzzy_for or fuzzy_for_options:
            warnings.warn("Can't do fuzzy with RunDB yet. Only returning exact matches")

        # Check if the run exists
        run_query: Dict[str, Union[str, int]]
        if self.runid_field == "name":
            run_query = {"name": str(key.run_id)}
        else:
            run_query = {"number": int(key.run_id)}

        # Check that we are in rucio backend
        if self.rucio_path is not None:
            rucio_key = key_to_rucio_did(key)
            rucio_available_query = self.available_query[-1]
            dq = {
                "data": {
                    "$elemMatch": {
                        "type": key.data_type,
                        "did": rucio_key,
                        **rucio_available_query,
                    },
                }
            }
            doc = self.collection.find_one(
                {
                    **run_query,
                    **dq,
                },
                projection=dq,
            )
            if doc is not None:
                datum = doc["data"][0]
                error_message = f'Expected {rucio_key} got data on {datum["location"]}'
                if datum.get("did", "") != rucio_key:
                    raise RuntimeError(error_message)
                backend_name = "RucioLocalBackend"
                backend_key = key_to_rucio_did(key)
                return backend_name, backend_key

        dq = self._data_query(key)
        doc = self.collection.find_one({**run_query, **dq}, projection=dq)

        if doc is None:
            # Data was not found
            if not write:
                raise strax.DataNotAvailable

            output_path = os.path.join(self.new_data_path, str(key))

            if self.new_data_path is not None:
                doc = self.collection.find_one(run_query, projection={"_id"})
                if not doc:
                    raise ValueError(
                        f"Attempt to register new data for non-existing run {key.run_id}"
                    )
                self.collection.find_one_and_update(
                    {"_id": doc["_id"]},
                    {
                        "$push": {
                            "data": {
                                "location": output_path,
                                "host": self.hostname,
                                "type": key.data_type,
                                "protocol": strax.FileSytemBackend.__name__,
                                "meta": {"lineage": key.lineage},
                            }
                        }
                    },
                )

            return (strax.FileSytemBackend.__name__, output_path)
        datum = doc["data"][0]

        if datum["host"] == "rucio-catalogue":
            raise strax.DataNotAvailable

        if write and not self._can_overwrite(key):
            raise strax.DataExistsError(at=datum["location"])

        return datum["protocol"], datum["location"]

    def find_several(self, keys: typing.List[strax.DataKey], **kwargs):
        if kwargs.get("fuzzy_for", False) or kwargs.get("fuzzy_for_options", False):
            warnings.warn("Can't do fuzzy with RunDB yet. Only returning exact matches")
        if not keys:
            return []
        if not len(set([k.lineage_hash for k in keys])) == 1:
            raise ValueError("find_several keys must have same lineage")
        if not len(set([k.data_type for k in keys])) == 1:
            raise ValueError("find_several keys must have same data type")
        keys = list(keys)  # Context used to pass a set

        if self.runid_field == "name":
            run_query = {"name": {"$in": [key.run_id for key in keys]}}
        else:
            run_query = {
                f"{self.runid_field}": {
                    "$in": [int(key.run_id) for key in keys if not key.run_id.startswith("_")]
                }
            }
        dq = self._data_query(keys[0])

        # dict.copy is sometimes not sufficient for nested dictionary
        projection = deepcopy(dq)
        projection.update({k: True for k in f"name number".split()})
        results_dict = self._parse_documents(
            self.collection.find({**run_query, **dq}, projection=projection),
        )

        return [results_dict.get(k.run_id, False) for k in keys]

    def _parse_documents(self, documents: Iterator):
        results_dict = dict()
        for doc in documents:
            # If you get a key error here there might be something off with the
            # projection
            datum = doc["data"][0]

            if self.runid_field == "name":
                dk = doc["name"]
            else:
                dk = f'{doc["number"]:06}'
            for required_field in ("protocol", "location"):
                if required_field not in datum:
                    raise ValueError(f"Missing field {required_field} in {datum} from {doc}")
            results_dict[dk] = datum["protocol"], datum["location"]
        return results_dict

    def _scan_runs(self, store_fields):
        query = self.number_query()
        projection = strax.to_str_tuple(list(store_fields))
        # Replace fields by their subfields if requested only take the most
        # "specific" projection
        projection = [
            f1 for f1 in projection if not any([f2.startswith(f1 + ".") for f2 in projection])
        ]
        cursor = self.collection.find(filter=query, projection=projection)
        for doc in strax.utils.tqdm(
            cursor,
            desc="Fetching run info from MongoDB",
            total=self.collection.count_documents(query),
            disable=not self.progress_bar,
        ):
            del doc["_id"]
            if self.reader_ini_name_is_mode:
                doc["mode"] = doc.get("reader", {}).get("ini", {}).get("name", "")
            yield doc

    def run_metadata(self, run_id, projection=None):
        if run_id.startswith("_"):
            # Superruns are currently not supported..
            raise strax.DataNotAvailable

        if self.runid_field == "name":
            run_id = str(run_id)
        else:
            run_id = int(run_id)
        if isinstance(projection, str):
            projection = {projection: 1}
        elif isinstance(projection, (list, tuple)):
            projection = {x: 1 for x in projection}

        doc = self.collection.find_one({self.runid_field: run_id}, projection=projection)
        if doc is None:
            raise strax.DataNotAvailable
        if self.reader_ini_name_is_mode:
            doc["mode"] = doc.get("reader", {}).get("ini", {}).get("name", "")
        return doc

    def number_query(self):
        q_number = {}
        if self.minimum_run_number:
            q_number = {"$gt": self.minimum_run_number}
        if self.maximum_run_number is not None:
            q_number.update({"$lt": self.maximum_run_number})
        if q_number:
            return {"number": q_number}
        return {}
