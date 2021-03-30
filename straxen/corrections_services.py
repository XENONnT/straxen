"""Return corrections from corrections DB
"""
import pytz
import numpy as np
from functools import lru_cache
import strax
import utilix
import straxen
import os

export, __all__ = strax.exporter()

corrections_w_file = ['mlp_model', 'gcn_model', 'cnn_model',
                      's2_xy_map', 's1_xy_map', 'fdc_map_mlp',
                      'fdc_map_gcn', 'fdc_map_cnn']

single_value_corrections = ['elife', 'baseline_samples_nv',
                            'electron_drift_velocity']


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

        mongo_kwargs = {'url': mongo_url,
                        'user': username,
                        'password': password,
                        'database': 'corrections'}
        corrections_collection = utilix.rundb.xent_collection(**mongo_kwargs)

        # Do not delete the client!
        self.client = corrections_collection.database.client

        # Setup the interface
        self.interface = strax.CorrectionsInterface(
            self.client,
            database_name='corrections')

        self.is_nt = is_nt
        if self.is_nt:
            self.collection = self.client['xenonnt']['runs']
        else:
            self.collection = self.client['run']['runs_new']

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return str(f'{"XENONnT " if self.is_nt else "XENON1T"}'
                   f'-Corrections_Management_Services')

    def get_corrections_config(self, run_id, config_model=None):
        """
        Get context configuration for a given correction
        :param run_id: run id from runDB
        :param config_model: configuration model (tuple type)
        :return: correction value(s)
        """

        if not isinstance(config_model, (tuple, list)) or len(config_model) != 2:
            raise ValueError(f'config_model {config_model} must be a tuple')
        model_type, global_version = config_model

        if 'to_pe_model' in model_type:
            return self.get_pmt_gains(run_id, model_type, global_version)
        elif model_type in single_value_corrections:
            return self._get_correction(run_id, model_type, global_version)
        elif model_type in corrections_w_file:
            return self.get_config_from_cmt(run_id, model_type, global_version)
        else:
            raise ValueError(f"{config_model} not found, currently these are "
                             f"available {single_value_corrections} and "
                             f"{corrections_w_file} ")

    # TODO add option to extract 'when'. Also, the start time might not be the best
    # entry for e.g. for super runs
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
                    if global_version in ('ONLINE', 'xenonnt_temporary_five_pmts'):
                        # We don't want to have different versions based
                        # on when something was processed therefore
                        # don't interpolate but forward fill.
                        df = self.interface.interpolate(df, when, how='fill')
                    if correction in corrections_w_file:
                        # is this the best solution?
                        df = self.interface.interpolate(df, when, how='fill')
                    else:
                        df = self.interface.interpolate(df, when)
                    values.append(df.loc[df.index == when, version].values[0])
            corrections = np.asarray(values)
        except KeyError:
            raise ValueError(f"Global version {global_version} not found for correction {correction}")

        else:
            return corrections

    def get_pmt_gains(self, run_id, model_type, global_version,
                      cacheable_versions=('ONLINE',),
                      gain_dtype=np.float32):
        """
        Smart logic to return pmt gains to PE values.
        :param run_id: run id from runDB
        :param model_type: to_pe_model (gain model)
        :param global_version: global version
        :param cacheable_versions: versions that are allowed to be
        cached in ./resource_cache
        :param gain_dtype: dtype of the gains to be returned as array
        :return: array of pmt gains to PE values
        """
        to_pe = None
        cache_name = None

        if 'to_pe_model' in model_type:
            # Get the detector name based on the requested model_type
            # This also will be used to the cachable name convention
            # pmt == TPC, n_veto == n_veto's PMT, etc
            detector_names = {'to_pe_model': 'pmt',
                              'to_pe_model_nv': 'n_veto',
                              'to_pe_model_mv': 'mu_veto'}
            target_detector = detector_names[model_type]

            if global_version in cacheable_versions:
                # Try to load from cache, if it does not exist it will be created below
                cache_name = cacheable_naming(run_id, model_type, global_version)
                try:
                    to_pe = straxen.get_resource(cache_name, fmt='npy')
                except (ValueError, FileNotFoundError):
                    pass

            if to_pe is None:
                to_pe = self._get_correction(run_id, target_detector, global_version)

            # be cautious with very early runs, check that not all are None
            if np.isnan(to_pe).all():
                raise ValueError(
                    f"to_pe(PMT gains) values are NaN, no data available "
                    f"for {run_id} in the gain model with version "
                    f"{global_version}, please set constant values for "
                    f"{run_id}")

        else:
            raise ValueError(f"{model_type} not implemented for to_pe values")

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

        if (cache_name is not None
                and global_version in cacheable_versions
                and not os.path.exists(cache_name)):
            # This is an array we can save since it's in the cacheable
            # versions but it has not been saved yet. Next time we need
            # it, we can get it from our cache.
            np.save(cache_name, to_pe, allow_pickle=False)
        return to_pe

    def get_config_from_cmt(self, run_id, model_type, global_version='ONLINE'):
        """
        Smart logic to return NN weights file name to be downloader by
        straxen.MongoDownloader()
        :param run_id: run id from runDB
        :param model_type: model type and neural network type; model_mlp,
        or model_gcn or model_cnn
        :param global_version: global version
        :param return: NN weights file name
        """
        if model_type not in corrections_w_file:
            raise ValueError(f"{model_type} is not stored in CMT "
                             f"please check, these are available {corrections_w_file}")

        file_name = self._get_correction(run_id, model_type, global_version)

        if not file_name:
            raise ValueError(f"You have the right option but could not find a file"
                             f"Please contact CMT manager and yell at him")

        return file_name

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


def cacheable_naming(*args, fmt='.npy', base='./resource_cache/'):
    """Convert args to consistent naming convention for array to be cached"""
    if not os.path.exists(base):
        try:
            os.mkdir(base)
        except (FileExistsError, PermissionError):
            pass
    for arg in args:
        if not type(arg) == str:
            raise TypeError(f'One or more args of {args} are not strings')
    return base + '_'.join(args) + fmt


class GainsNotFoundError(Exception):
    """Fatal error if a None value is returned by the corrections"""
    pass
