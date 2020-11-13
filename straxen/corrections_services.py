"""Return corrections from corrections DB
"""
import pytz
import numpy as np
from functools import lru_cache
import configparser
import strax
import straxen
from straxen import uconfig

export, __all__ = strax.exporter()


@export
class CorrectionsManagementServices():
    """
    A class that returns corrections
    Corrections are set of parameters to be applied in the analysis
    stage to remove detector effects. Information on the strax implementation
    can be found at https://github.com/AxFoundation/strax/blob/master/strax/corrections.py
    """
    def __init__(self, username=None, password=None, mongo_url=None, is_nt=True):
        """
        :param username: corrections DB username
            read the .xenon_config for the users "pymongo_user" has
            readonly permissions to the corrections DB
            the "CMT admin user" has r/w permission to corrections DB
            and read permission to runsDB
        :param password: DB password
        :param is_nt: bool if True we are looking at nT if False we are looking at 1T
        """
        # TODO avoid duplicated code with the RunDB.py?
        # Basic setup
        if username is not None:
            self.username = username
        else:
            self.username = uconfig.get('RunDB', 'pymongo_user')
        if password is not None:
            self.password = password
        else:
            self.password = uconfig.get('RunDB', 'pymongo_password')

        if mongo_url is None:
            mongo_url = uconfig.get('RunDB', 'pymongo_url')

        # Setup the interface
        self.interface = strax.CorrectionsInterface(
            host=f'mongodb://{mongo_url}',
            username=self.username,
            password=self.password,
            database_name='corrections')
        # Use the same client as the CorrectionsInterface
        client = self.interface.client

        self.is_nt = is_nt
        if self.is_nt:
            self.collection = client['xenonnt']['runs']
        else:
            self.collection = client['run']['runs_new']

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(f'{"XENONnT " if self.is_nt else "XENON1T"}'
                   f'-Corrections_Management_Services')

    def get_corrections_config(self, run_id, correction=None, config_model=None):
        """
        Get context configuration for a given correction
        :param run_id: run id from runDB
        :param correction: correction's name (str type)
        :param config_model: configuration model (tuple type)
        :return: correction value(s)
        """

        if not isinstance(config_model, (tuple, list)) or len(config_model) != 2:
            raise ValueError(f'config_model {config_model} must be a tuple')
        model_type, global_version = config_model

        if correction == 'pmt_gains':
            return self.get_pmt_gains(run_id, model_type, global_version)
        elif correction == 'elife':
            return self.get_elife(run_id, model_type, global_version)
        else:
            raise ValueError(f'{correction} not found')

    # TODO add option to extract 'when'. Also, the start time might not be the best
    #  entry for e.g. for super runs
    # cache results, this would help when looking at the same gains
    @lru_cache(maxsize=None)
    def _get_correction(self, run_id, correction, global_version):
        """
        Smart logic to get correction from DB
        :param run_id: run id from runDB
        :param correction: correction's name, key word (str type)
        :param global_version: global version (str type)
        :return: correction value(s)
        """
        when = self.get_start_time(run_id)
        df_global = self.interface.read('global' if self.is_nt else 'global_xenon1t')

        try:
            values = []
            for it_correction, version in df_global.iloc[-1][global_version].items():
                if correction in it_correction:
                    df = self.interface.read(it_correction)
                    if global_version == 'ONLINE':
                        # We don't want to have different versions based
                        # on when something was processed therefore
                        # don't interpolate but forward fill.
                        df = self.interface.interpolate(df, when, how='fill')
                    else:
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

    def get_elife(self, run_id, model_type, global_version):
        """
        Smart logic to return electron lifetime correction
        :param run_id: run id from runDB
        :param model_type: choose either elife_model or elife_constant
        :param global_version: global version, or float (if model_type == elife_constant) 
        :return: electron lifetime correction value
        """
        if model_type == 'elife_model':
            return self._get_correction(run_id, 'elife', global_version)

        elif model_type == 'elife_constant':
            # This is nothing more than just returning the value we put in
            if not isinstance(global_version, float):
                raise ValueError(f'User specify a model type {model_type} '
                                 f'and should provide a float. Got: '
                                 f'{type(global_version)}')
            return float(global_version)

        else:
            raise ValueError(f'model type {model_type} not implemented for electron lifetime')

    # TODO create a propper dict for 'to_pe_constant' and 'global_version' as
    #  the 'global_version' is not a version but an array/float for
    #  model_type = 'to_pe_constant'
    def get_pmt_gains(self, run_id, model_type, global_version, gain_dtype = np.float32):
        """
        Smart logic to return pmt gains to PE values.
        :param run_id: run id from runDB
        :param model_type: Choose either to_pe_model or to_pe_constant
        :param global_version: global version or a constant value or an array (if
        model_type == to_pe_constant)
        :param gain_dtype: dtype of the gains to be returned as array
        :return: array of pmt gains to PE values
        """
        if model_type == 'to_pe_model':
            to_pe = self._get_correction(run_id, 'pmt', global_version)
            # be cautious with very early runs, check that not all are None
            if np.isnan(to_pe).all():
                raise ValueError(
                        f'to_pe(PMT gains) values are NaN, no data available'
                        f' for {run_id} in the gain model with version '
                        f'{global_version}, please set constant values for '
                        f'{run_id}')

        elif model_type == 'to_pe_constant':
            n_tpc_pmts = straxen.n_tpc_pmts
            if not self.is_nt:
                # TODO can we prevent these kind of hard codes using the context?
                n_tpc_pmts = straxen.contexts.x1t_common_config['n_tpc_pmts']

            if not isinstance(global_version, (float, np.ndarray, int)):
                raise ValueError(f'User must specify a model type {model_type} '
                                 f'and provide a float/array to be used. Got: '
                                 f'{type(global_version)}')

            # Generate an array of values and multiply by the 'global_version'
            to_pe = np.ones(n_tpc_pmts, dtype=gain_dtype) * global_version
            if len(to_pe) != n_tpc_pmts:
                raise ValueError(f'to_pe length does not match {n_tpc_pmts}. '
                                 f'Check that {global_version} is either of '
                                 f'length {n_tpc_pmts} or a float')

        else:
            raise ValueError(f'{model_type} not implemented for to_pe values')

        # Double check the dtype of the gains
        to_pe = np.array(to_pe, dtype=gain_dtype)

        # Double check that all the gains are found, None is not allowed
        # since strax processing does not handle this well. If a PMT is
        # off it's gain should be 0.
        if np.any(np.isnan(to_pe)):
            pmts_affected = np.argwhere(np.isnan(to_pe))[:, 0]
            raise GainsNotFoundError(
                f'Gains returned by CMT are None for PMT_i = {pmts_affected}. '
                f'Cannot proceed with processing. Report to CMT-maintainers.')
        return to_pe

    def get_lce(self, run_id, s, position, global_version='v1'):
        """
        Smart logic to return light collection eff map values.
        :param run_id: run id from runDB
        :param s: S1 map or S2 map
        :param global_version:
        :param position: event position
        """
        raise NotImplementedError

    def get_fdc(self, run_id, position, global_version='v1'):
        """
        Smart logic to return field distortion map values.
        :param run_id: run id from runDB
        :param position: event position
        :param global_version: global version (str type)
        """
        raise NotImplementedError

    # TODO change to st.estimate_start_time
    def get_start_time(self, run_id):
        """
        Smart logic to return start time from runsDB
        :param run_id: run id from runDB
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


class GainsNotFoundError(Exception):
    """Fatal error if a None value is returned by the corrections"""
    pass
