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
        :param host: corrections DB host
        :param username: corrections DB username
            nt_analysis user has read only permission to corrections DB
            cmt user has r/w permission to corrections DB and read permission to runsDB
        :param password: DB password
        """
        self.host = host
        self.username = username
        if password is not None:
            self.password = password
        elif self.username.endswith('analysis'):
            self.password = straxen.get_secret('rundb_password')
        else:
            raise ValueError('No password for {user_name}')
        # Initialize DBs
        if host is None:
            mongo_connections = [default_mongo_url, *backup_mongo_urls]
            for url in mongo_connections:
                try:
                    if self.username.endswith('analysis'):
                        mongo_url = f'mongodb://{url}'
                    else:
                        mongo_url = url[:-14]
                    # Initialize correction DB
                    self.interface = strax.CorrectionsInterface(
                        host=mongo_url,
                        username=self.username,
                        password=self.password,
                        database_name='corrections')
                    # Initialize runs DB
                    runsdb_user = straxen.get_secret('rundb_username')
                    runsdb_mongo_url = f'mongodb://{runsdb_user}@{url}'
                    self.collection = pymongo.MongoClient(
                            runsdb_mongo_url,
                            password=straxen.get_secret(
                                'rundb_password'))['xenonnt']['runs']
                    self.collection_1t = pymongo.MongoClient(
                            runsdb_mongo_url,
                            password=straxen.get_secret(
                                'rundb_password'))['run']['runs_new']
                    break
                except pymongo.errors.ServerSelectionTimeoutError:
                    warn(f'Cannot connect to to Mongo url: {url}')
                    if url == mongo_connections[-1]:
                        raise pymongo.errors.ServerSelectionTimeoutError(
                            'Cannot connect to any Mongo url')
        else:
            raise PermissionError(f'Trying to use an invalid host(non xenon host)')

    def get_corrections_config(self, run_id, correction=None, config_model=None):
        """
        Get context configuration for a given correction
        :param run_id: run id from runDB
        :param correction: correction's name (str type)
        :param config_model: configuration model (tuple type)
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
                if type(global_version) != float and type(global_version) != np.ndarray:
                    raise ValueError(
                            f'User specify a model type {model_type} '
                            f'and provide a {type(global_version)} to be used')
                cte_value = global_version
                to_pe = np.ones(n_tpc_pmts, dtype=np.float32) * cte_value
                if len(to_pe) != n_tpc_pmts:
                    raise ValueError(f'to_pe length does not match {n_tpc_pmts}')
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
            raise ValueError(f'{correction} not found')

    def get_elife(self, run_id, global_version='v1', xenon1t=False):
        """
        Smart logic to return electron lifetime correction
        :param run_id: run id from runDB
        :param global_version: global version
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :return: electron lifetime correction value
        """
        when = self.get_time(run_id, xenon1t)
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
        :return: array of pmt gains to PE values
        """
        when = self.get_time(run_id, xenon1t)
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
        # be cautious with very early runs
        test = np.isnan(pmt_gains)
        if test.all():
            raise ValueError(f'to_pe(PMT gains) values are NaN, no data available'
                             f' for {run_id} in the gain model with version {global_version},'
                             f' please set a cte values for {run_id}')
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

    def get_fdc(self, run_id, position, global_version='v1', xenon1t=False):
        """
        Smart logic to return field distortion map values.
        :param run_id: run id from runDB
        :param position: event position
        :param xenon1t: boolean xenon1t data=False/True
        """
        raise NotImplementedError

    def get_time(self, run_id, xenon1t=False):
        """
        Smart logic to return start time from runsDB
        :param run_id: run id from runDB
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :return: run start time
        """
        runsdb_collection = pymongo.MongoClient()
        if xenon1t:
            runsdb_collection = self.collection_1t
        else:
            # xenonnt use int
            run_id = int(run_id)
            runsdb_collection = self.collection

        rundoc = runsdb_collection.find_one(
                {'name' if xenon1t else 'number': run_id, 'end': {'$exists': 1}},
                {'start': 1})
        if rundoc is None:
            raise ValueError(f'run_id = {run_id} not found')
        time = rundoc['start']
        return time.replace(tzinfo=pytz.utc)
