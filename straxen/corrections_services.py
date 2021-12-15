"""Return corrections from corrections DB
"""
import warnings
import pytz
import numpy as np
from functools import lru_cache
import strax
import utilix
import straxen
import os
from urllib.parse import urlparse, parse_qs

export, __all__ = strax.exporter()

corrections_w_file = ['mlp_model', 'cnn_model', 'gcn_model',
                      's2_xy_map_mlp', 's2_xy_map_cnn', 's2_xy_map_gcn', 's2_xy_map',
                      's1_xyz_map_mlp', 's1_xyz_map_cnn', 's1_xyz_map_gcn',
                      'fdc_map_mlp', 'fdc_map_cnn', 'fdc_map_gcn']

single_value_corrections = ['elife_xenon1t', 'elife', 'baseline_samples_nv',
                            'electron_drift_velocity', 'electron_drift_time_gate',
                            'se_gain', 'rel_extraction_eff']

arrays_corrections = ['hit_thresholds_tpc', 'hit_thresholds_he',
                      'hit_thresholds_nv', 'hit_thresholds_mv']

# needed because we pass these names as strax options which then get paired with the default reconstruction algorithm
# important for apply_cmt_version
posrec_corrections_basenames = ['s1_xyz_map', 'fdc_map', 's2_xy_map']


class CMTVersionError(Exception):
    pass

class CMTnanValueError(Exception):
    pass

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
            raise ValueError(f'config_model {config_model} must be a tuple of length 2')
        model_type, version = config_model

        if 'to_pe_model' in model_type:
            return self.get_pmt_gains(run_id, model_type, version)
        elif model_type in single_value_corrections or model_type in arrays_corrections:
            return self._get_correction(run_id, model_type, version)
        elif model_type in corrections_w_file:
            return self.get_config_from_cmt(run_id, model_type, version)
        else:
            raise ValueError(f"{model_type} not found, currently these are "
                             f"available {single_value_corrections}, {arrays_corrections} and "
                             f"{corrections_w_file} ")

    # TODO add option to extract 'when'. Also, the start time might not be the best
    # entry for e.g. for super runs
    # cache results, this would help when looking at the same gains
    @lru_cache(maxsize=None)
    def _get_correction(self, run_id, correction, version):
        """
        Smart logic to get correction from DB
        :param run_id: run id from runDB
        :param correction: correction's name, key word (str type)
        :param version: local version (str type)
        :return: correction value(s)
        """
        when = self.get_start_time(run_id)

        try:
            values = []

            # hack to workaround to group all pmts
            # because every pmt is its own dataframe...of course
            if correction in {'pmt', 'n_veto', 'mu_veto'}:
                # get lists of pmts 
                df_global = self.interface.read('global_xenonnt' if self.is_nt else 'global_xenon1t')
                gains = df_global['global_ONLINE'][0]  # global is where all pmts are grouped
                pmts = list(gains.keys())
                for it_correction in pmts: # loop over all PMTs
                    if correction in it_correction:
                        df = self.interface.read_at(it_correction, when)
                        if df[version].isnull().values.any():
                            raise CMTnanValueError(f"For {it_correction} there are NaN values, this means no correction available "
                                                   f"for {run_id} in version {version}, please check e-logbook for more info ")
 
                        if version in 'ONLINE':
                            df = self.interface.interpolate(df, when, how='fill')
                        else:
                            df = self.interface.interpolate(df, when)
                        values.append(df.loc[df.index == when, version].values[0])
            else:
                df = self.interface.read_at(correction, when)
                if df[version].isnull().values.any():
                    raise CMTnanValueError(f"For {correction} there are NaN values, this means no correction available "
                                           f"for {run_id} in version {version}, please check e-logbook for more info ")
 
                if correction in corrections_w_file or correction in arrays_corrections or version in 'ONLINE':
                    df = self.interface.interpolate(df, when, how='fill')
                else:
                    df = self.interface.interpolate(df, when)
                values.append(df.loc[df.index == when, version].values[0])
            corrections = np.asarray(values)
        except KeyError:
            if "global" in version:
                raise ValueError(f"User is not allowed to pass {version} global version are not allowed")
            raise ValueError(f"Version {version} not found for correction {correction}, please check")
            
        else:
            return corrections

    def get_pmt_gains(self, run_id, model_type, version,
                      cacheable_versions=('ONLINE',),
                      gain_dtype=np.float32):
        """
        Smart logic to return pmt gains to PE values.
        :param run_id: run id from runDB
        :param model_type: to_pe_model (gain model)
        :param version: version
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

            if version in cacheable_versions:
                # Try to load from cache, if it does not exist it will be created below
                cache_name = cacheable_naming(run_id, model_type, version)
                try:
                    to_pe = straxen.get_resource(cache_name, fmt='npy')
                except (ValueError, FileNotFoundError):
                    pass

            if to_pe is None:
                to_pe = self._get_correction(run_id, target_detector, version)

            # be cautious with very early runs, check that not all are None
            if np.isnan(to_pe).any():
                raise ValueError(
                    f"to_pe(PMT gains) values are NaN, no data available "
                    f"for {run_id} in the gain model with version")

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
                and version in cacheable_versions
                and not os.path.exists(cache_name)):
            # This is an array we can save since it's in the cacheable
            # versions but it has not been saved yet. Next time we need
            # it, we can get it from our cache.
            np.save(cache_name, to_pe, allow_pickle=False)
        return to_pe

    def get_config_from_cmt(self, run_id, model_type, version='ONLINE'):
        """
        Smart logic to return NN weights file name to be downloader by
        straxen.MongoDownloader()
        :param run_id: run id from runDB
        :param model_type: model type and neural network type; model_mlp,
        or model_gcn or model_cnn
        :param version: version
        :param return: NN weights file name
        """
        if model_type not in corrections_w_file:
            raise ValueError(f"{model_type} is not stored in CMT "
                             f"please check, these are available {corrections_w_file}")

        file_name = self._get_correction(run_id, model_type, version)

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

    def get_local_versions(self, global_version):
        """Returns a dict of local versions for a given global version. Use 'latest' to get newest version"""
        # check that 'global' is in the passed string.

        if global_version == 'latest':
            # CMT appends columns to the global versions dataframe, so taking last one is the latests
            global_version = self.global_versions[-1]

        if 'global' not in global_version:
            warnings.warn("'global' does not appear in the passed global version. Are you sure this right?")
        # CMT generates a global version, global version is just a set of local versions
        # With this we can do pretty easy bookkeping for offline contexts

        cmt_global = self.interface.read('global_xenonnt')
        if global_version not in cmt_global:
            avail_global_versions_string = '\n'.join([f'\t\t{v}' for v in self.global_versions])
            raise ValueError(f"Global version {global_version} not found! "
                             f"Try one of these:\n{avail_global_versions_string}")
        # get local versions from CMT global version
        local_versions = cmt_global[global_version][0]

        # to make returned dictionary more manageable, we prune all the per-PMT corrections
        # first rename to more clear variable
        local_versions['to_pe_model'] = local_versions['pmt_000_gain_xenonnt']
        local_versions['to_pe_model_nv'] = local_versions['n_veto_000_gain_xenonnt']
        local_versions['to_pe_model_mv'] = local_versions['mu_veto_000_gain_xenonnt']

        # drop the per-PMT corrections
        pruned_local_versions = {key: val for key, val in local_versions.items() if "_gain_xenonnt" not in key}
        return pruned_local_versions

    @property
    def global_versions(self):
        return self.interface.read('global_xenonnt').columns.tolist()


def cacheable_naming(*args, fmt='.npy', base='./resource_cache/'):
    """Convert args to consistent naming convention for array to be cached"""
    if not os.path.exists(base):
        try:
            os.mkdir(base)
        except (FileExistsError, PermissionError):
            pass
    for arg in args:
        if not isinstance(arg, str):
            raise TypeError(f'One or more args of {args} are not strings')
    return base + '_'.join(args) + fmt


class GainsNotFoundError(Exception):
    """Fatal error if a None value is returned by the corrections"""


def get_cmt_local_versions(global_version):
    cmt = CorrectionsManagementServices()
    return cmt.get_local_versions(global_version)


args_idx = lambda x: x.rfind('?') if '?' in x else None


def replace_version(url, version):
    kwargs = {k:v[0] for k,v in parse_qs(urlparse(url).query).items()}
    kwargs['version'] = version
    args = [f"{k}={v}" for k,v in kwargs.items()]
    args_str = "&".join(args)
    return f'{url[:args_idx(url)]}?{args_str}'


def correction_name_from_urlstring(url):
    # only care about what happens after cmt
    before_url, cmt, after_url = url.partition('cmt://')
    tmp_config = straxen.URLConfig(default=after_url)
    new_url, kwargs = tmp_config.split_url_kwargs(after_url)
    return tmp_config.dispatch(new_url, **kwargs)


@strax.Context.add_method
def apply_cmt_version(context: strax.Context, cmt_global_version: str):
    """Sets all the relevant correction variables
    :param cmt_global_version: A specific CMT global version, or 'latest' to get the newest one
    :returns None
    """
    local_versions = get_cmt_local_versions(cmt_global_version)

    # get the position algorithm we are using
    # I feel like this should be easier...
    posrec_option = 'default_reconstruction_algorithm'
    if posrec_option in context.config:
        posrec_algo = context.config[posrec_option]
    else:
        posrec_algo = context._plugin_class_registry['peak_positions'].takes_config[posrec_option].default

    cmt_options = straxen.get_corrections.get_cmt_options(context)

    # catch here global versions that are not compatible with this straxen version
    # this happens if a new correction was added to CMT that was not used in a fixed version
    # we want this error to occur in order to keep fixed global versions
    cmt_config = dict()
    failed_keys = []

    for option, value in cmt_options.items():
        # might need to modify correction name to include position reconstruction algo
        # if the config is a URL, get the correction name in CMT
        if isinstance(value, str) and 'cmt://' in value:
            correction_name = correction_name_from_urlstring(value)
        # if it's still a tuple, it's the first element
        else:
            correction_name = value[0]
            # this is a bit of a mess, but posrec configs are treated differently in the tuples
            if correction_name in posrec_corrections_basenames:
                correction_name += f"_{posrec_algo}"

        # now see if our correction is in our local_versions dict
        if correction_name in local_versions:
            if isinstance(value, str) and 'cmt://' in value:
                new_value = replace_version(value, local_versions[correction_name])
            # if it is a tuple, make a new tuple
            else:
                new_value = (value[0], local_versions[correction_name], value[2])
        else:
            failed_keys.append(correction_name)
            continue

        cmt_config[option] = new_value

    if len(failed_keys):
        failed_keys = ', '.join(failed_keys)
        msg = f"CMT version {cmt_global_version} is not compatible with this straxen version! " \
              f"CMT {cmt_global_version} is missing these corrections: {failed_keys}"

        # only raise a warning if we are working with the online context
        if cmt_global_version == "global_ONLINE":
            warnings.warn(msg)
        else:
            raise CMTVersionError(msg)

    context.set_config(cmt_config)
