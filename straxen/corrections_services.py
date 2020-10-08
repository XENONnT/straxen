"""Return corrections from corrections DB
"""
import pytz
import pymongo
import numpy as np
from warnings import warn
from functools import lru_cache

import strax
import straxen
from .rundb import default_mongo_url, backup_mongo_urls

export, __all__ = strax.exporter()


# TODO inherit the patent class
@export
class CorrectionsManagementServices():
    """
    A class that returns corrections
    Corrections are set of parameters to be applied in the analysis
    stage to remove detector effects. Information on the strax implementation
    can be found at https://github.com/AxFoundation/strax/blob/master/strax/corrections.py
    """
    def __init__(self, host=None, username='nt_analysis', password=None, is_nt=True):
        """
        :param host: corrections DB host
        :param username: corrections DB username
            nt_analysis user has read only permission to corrections DB
            cmt user has r/w permission to corrections DB and read permission to runsDB
        :param password: DB password
        :param is_nt: bool if True we are looking at nT if False we are looking at 1T
        """
        # TODO not needed? Should it ever be not None??
        self.host = host
        self.username = username
        self.is_nt = is_nt

        if password is not None:
            self.password = password
        elif self.username.endswith('analysis'):
            self.password = straxen.get_secret('rundb_password')
        else:
            raise ValueError(f'No password for {username}')
        # TODO avoid duplicate code with RunDB.py
        # Initialize runDB to get start-times
        if host is None:
            mongo_connections = [default_mongo_url, *backup_mongo_urls]
            for url in mongo_connections:
                try:
                    if self.username.endswith('analysis'):
                        mongo_url = f'mongodb://{url}'
                    else:
                        # TODO make this cleaner
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
                    if self.is_nt:
                        # TODO make cleaner
                        self.collection = pymongo.MongoClient(
                            runsdb_mongo_url,
                            password=straxen.get_secret(
                                'rundb_password'))['xenonnt']['runs']
                    else:
                        # TODO make cleaner
                        self.collection = pymongo.MongoClient(
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

    def __repr__(self):
        return str(f'{"XENONnT " if self.is_nt else "XENON1T"}-Corrections_Management_Services')

    def get_corrections_config(self, run_id, correction=None, config_model=None):
        """
        Get context configuration for a given correction
        :param run_id: run id from runDB
        :param correction: correction's name (str type)
        :param config_model: configuration model (tuple type)
        :return: correction value(s)
        """

        if not isinstance(config_model, (tuple, list)) or not (2 <= len(config_model) <= 3):
            raise ValueError(f'config_model {config_model} must be a tuple')
        if len(config_model) == 3:
            # TODO
            #  Fix this because we shouldn't be mixing 1t and nT
            model_type, global_version, xenon1t = config_model
        elif len(config_model) == 2:
            model_type, global_version = config_model
            # TODO obsolete
            xenon1t = False

        if correction == 'pmt_gains':
            to_pe = self.get_pmt_gains(run_id, model_type, global_version, xenon1t=not self.is_nt)
            return to_pe
        elif correction == 'elife':
            elife = self.get_elife(run_id, model_type, global_version, xenon1t=not self.is_nt)
            return elife
        else:
            raise ValueError(f'{correction} not found')

    # TODO add option to extract 'when', the start time might not be the best
    #  entry for e.g. for super runs
    # cache results, this would help when looking at the same gains
    @lru_cache(maxsize=None)
    def _get_correction(self, run_id, correction, global_version, xenon1t=False):
        """
        Smart logic to get correction from DB
        :param run_id: run id from runDB
        :param correction: correction's name, key word (str type)
        :param global_version: global version (str type)
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :return: correction value(s)
        """
        when = self.get_start_time(run_id, not self.is_nt)
        if not self.is_nt:
            df_global = self.interface.read('global_xenon1t')
        else:
            df_global = self.interface.read('global')
        try:
            values = []
            for it_correction, version in df_global.iloc[-1][global_version].items():
                if correction in it_correction:
                    df = self.interface.read(it_correction)
                    df = self.interface.interpolate(df, when)
                    values.append(df.loc[df.index == when, version].values[0])
                corrections = np.asarray(values)
        except KeyError:
            raise ValueError(f'Global version {global_version} not found for correction {correction}')
        # for single value corrections, e.g. elife correction
        if len(corrections) == 1:
            return float(corrections)
        else:
            return corrections

    def get_elife(self, run_id, model_type, global_version, xenon1t=False):
        """
        Smart logic to return electron lifetime correction
        :param run_id: run id from runDB
        :param model_type: choose either elife_model or elife_constant
        :param global_version: global version
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :return: electron lifetime correction value
        """
        if model_type == 'elife_model':
            elife = self._get_correction(run_id, 'elife', global_version, not self.is_nt)
            return elife
        elif model_type == 'elife_constant':
            if not isinstance(global_version, float):
                raise ValueError(
                    f'User must specify a model type {model_type} '
                    f'and provide a {type(global_version)} to be used')
            cte_value = global_version
            elife = float(cte_value)
            return elife
        else:
            raise ValueError(f'model type {model_type} not implemented for electron lifetime')

    def get_pmt_gains(self, run_id, model_type, global_version, xenon1t=False):
        """
        Smart logic to return pmt gains to PE values.
        :param run_id: run id from runDB
        :param model_type: Choose either to_pe_model or to_pe_constant
        :param global_version: global version
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :return: array of pmt gains to PE values
        """
        if model_type == 'to_pe_model':
            to_pe = self._get_correction(run_id, 'pmt', global_version, not self.is_nt)
            # be cautious with very early runs
            test = np.isnan(to_pe)
            if test.all():
                raise ValueError(
                        f'to_pe(PMT gains) values are NaN, no data available'
                        f' for {run_id} in the gain model with version {global_version},'
                        f' please set a cte values for {run_id}')
            return to_pe
        elif model_type == 'to_pe_constant':
            n_tpc_pmts = straxen.n_tpc_pmts
            if not self.is_nt:
                n_tpc_pmts = 248
            if not isinstance(global_version, (float, np.ndarray)):
                raise ValueError(
                        f'User specify a model type {model_type} '
                        f'and provide a {type(global_version)} to be used')
            cte_value = global_version
            to_pe = np.ones(n_tpc_pmts, dtype=np.float32) * cte_value
            if len(to_pe) != n_tpc_pmts:
                raise ValueError(f'to_pe length does not match {n_tpc_pmts}')
            return to_pe
        else:
            raise ValueError(f'{model_type} not implemented for to_pe values')

    def get_lce(self, run_id, s, position, global_version='v1', xenon1t=False):
        """
        Smart logic to return light collection eff map values.
        :param run_id: run id from runDB
        :param s: S1 map or S2 map
        :param global_version:
        :param position: event position
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        """
        raise NotImplementedError

    def get_fdc(self, run_id, position, global_version='v1', xenon1t=False):
        """
        Smart logic to return field distortion map values.
        :param run_id: run id from runDB
        :param position: event position
        :param global_version:
        :param xenon1t: boolean xenon1t data=False/True
        """
        raise NotImplementedError

    # TODO change to st.estimate_start_time
    def get_start_time(self, run_id, xenon1t=False):
        """
        Smart logic to return start time from runsDB
        :param run_id: run id from runDB
        :param xenon1t: boolean, whether you are processing xenon1t data or not
        :return: run start time
        """

        if self.is_nt:
            # xenonnt use int
            run_id = int(run_id)

        rundoc = self.collection.find_one(
                {'number' if self.is_nt else 'name': run_id},
                {'start': 1})
        if rundoc is None:
            raise ValueError(f'run_id = {run_id} not found')
        time = rundoc['start']
        return time.replace(tzinfo=pytz.utc)
