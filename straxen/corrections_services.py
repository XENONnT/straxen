"""Return corrections from corrections DB
"""
from datetime import datetime
from datetime import timezone
import pytz
from bson.son import SON
import pymongo
import numpy as np
from warnings import warn

import strax
import straxen
from .rundb import default_mongo_url, backup_mongo_urls

export, __all__ = strax.exporter()


@export
class CmtServices():
    """
    A class that returns corrections
    Corrections are set of parameters to be applied in the analysis
    stage to remove detector effects. Information on the strax implementation
    can be found at https://github.com/AxFoundation/strax/blob/master/strax/corrections.py
    """
    def __init__(self, host=None, username='nt_analysis', password=None):
        """
        :param host: DB host
        :param username: DB username
        :param password: DB password
        """
        self.host = host
        self.username = username
        self.password = password
        if self.username.endswith('analysis'):
            self.password = straxen.get_secret('rundb_password')

        if host is None:
            mongo_connections = [default_mongo_url, *backup_mongo_urls]
            for url in mongo_connections:
                try:
                    mongo_url = f'mongodb://{url}'

                    self.interface = strax.CorrectionsInterface(
                        host=mongo_url,
                        username=self.username,
                        password=self.password,
                        database_name='corrections')
                    break
                except pymongo.errors.ServerSelectionTimeoutError:
                    warn(f'Cannot connect to to Mongo url: {url}')
                    if url == mongo_connections[-1]:
                        raise pymongo.errors.ServerSelectionTimeoutError(
                            'Cannot connect to any Mongo url')
        else:
            self.interface = strax.CorrectionsInterface(
                    host=self.host,
                    username=self.username,
                    password=self.password,
                    database_name='corrections')

    def get_corrections_config(self, run_id, correction=None, config_model=None):
        """
        Get context configuration for a given correction
        :param run_id: run id from runDB
        :param correction: correction's name (str type)
        :param config_model: configuration model (dict type)
        :return: correction value(s)
        """

        if not isinstance(config_model, tuple):
            raise ValueError(f'config_model {config_model} must be a tuple')
        if len(config_model) == 3:
            model_type, global_version, xenon1t = config_model
        elif len(config_model) == 2:
            model_type, global_version = config_model
            xenon1t = False

        if correction == 'pmt_gains':
            if model_type == 'to_pe_model':
                to_pe = self.get_pmt_gains(run_id, global_version, xenon1t)
                return to_pe

            elif model_type == 'to_pe_constant':
                n_tpc_pmts = straxen.n_tpc_pmts
                if xenon1t:
                    n_tpc_pmts = 248
                cte_value = global_version
                to_pe = np.ones(n_tpc_pmts, dtype=np.float32) * cte_value
                return to_pe

            else:
                raise ValueError(f'{model_type} not implemented')

        elif correction == 'elife':
            if model_type == 'elife_model':
                elife = self.get_elife(run_id, global_version, xenon1t)
                return elife
            elif model_type == 'elife_constant':
                cte_value = global_version
                elife = float(cte_value)
                return elife
        else:
            raise ValueError(f'{corection} not found')

    def get_elife(self, run_id, global_version='v1', xenon1t=False):
        """
        Smart logic to return electron lifetime correction
        :param run_id: run id from runDB
        :param global_version: global version
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :return: electron lifetime correction value
        """
        when = CmtServices.get_time(run_id, xenon1t)
        if xenon1t:
            df_global = self.interface.read('global_xenon1t')
        else:
            df_global = self.interface.read('global')
        try:
            for correction, version in df_global.iloc[-1][global_version].items():
                if 'elife' in correction:
                    df = self.interface.read(correction)
                    df = self.interface.interpolate(df, when)
        except KeyError:
            raise ValueError(f'Global version {global_version} not found')

        return df.loc[df.index == when, global_version].values[0]

    def get_pmt_gains(self, run_id, global_version='v1', xenon1t=False):
        """
        Smart logic to return pmt gains to PE values.
        :param run_id: run id from runDB
        :param global_version: global version
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :retrun: array of pmt gains to PE values
        """
        when = CmtServices.get_time(run_id, xenon1t)
        if xenon1t:
            df_global = self.interface.read('global_xenon1t')
        else:
            df_global = self.interface.read('global')
        try:
            # equivalent to 'to_pe' in gains_model
            gains = []
            for correction, version in df_global.iloc[-1][global_version].items():
                if 'pmt' in correction:
                    df = self.interface.read(correction)
                    df = self.interface.interpolate(df, when)
                    gains.append(df.loc[df.index == when, version].values[0])
                pmt_gains = np.asarray(gains, dtype=np.float32)

        except KeyError:
            raise ValueError(f'Global version {global_version} not found')

        return pmt_gains

    def get_lce(self, run_id, s, position, global_version='v1', xenon1t=False):
        """
        Smart logic to return light collection eff map values.
        :param run_id: run id from runDB
        :param s: S1 map or S2 map
        :param postion: event position
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        """
        raise NotImplementedError

    def get_fdc(slef, run_id, position, global_version='v1', xenon1t=False):
        """
        Smart logic to return field distortion map values.
        :param run_id: run id from runDB
        :param postion: event position
        :param xenon1t: boolean xenon1t data=False/True
        """
        raise NotImplementedError

    @staticmethod
    def get_time(run_id, xenon1t=False):
        """
        Smart logic to return start time from runsDB
        :param run_id: run id from runDB
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :retrun: run start time
        """
        # xenonnt use int
        if not xenon1t:
            run_id = int(run_id)

        username = straxen.get_secret('rundb_username')
        rundb_password = straxen.get_secret('rundb_password')
        mongo_url = f'mongodb://{username}@xenon-rundb.grid.uchicago.edu:27017/xenonnt'
        collection = pymongo.MongoClient(mongo_url, password=rundb_password)['xenonnt']['runs']

        pipeline = [
                {'$match':
                    {"number": run_id, "detectors": "tpc", "end":
                        {"$exists": True}}},
                {"$project": {'time': '$start', 'number': 1, '_id': 0}},
                {"$sort": SON([("time", 1)])}]

        if xenon1t:
            collection = pymongo.MongoClient(mongo_url, password=rundb_password)['run']['runs_new']
            pipeline = [
                    {'$match':
                        {"name": run_id, "detector": "tpc", "end":
                            {"$exists": True}}},
                    {"$project": {'time': '$start', 'name': 1, '_id': 0}},
                    {"$sort": SON([("time", 1)])}]

        # to save it in datetime format
        time = datetime.now(tz=timezone.utc)
        rundb_info = list(collection.aggregate(pipeline))
        if not len(rundb_info):
            raise ValueError(f'run_id = {run_id} not found')
        for t in collection.aggregate(pipeline):
            time = t['time']
        return time.replace(tzinfo=pytz.utc)
