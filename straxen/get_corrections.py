import numpy as np
import strax
import straxen
from warnings import warn
from functools import wraps
from straxen.corrections_services import corrections_w_file
from straxen.corrections_services import single_value_corrections
from straxen.corrections_services import arrays_corrections


export, __all__ = strax.exporter()
__all__ += ['FIXED_TO_PE']


def correction_options(get_correction_function):
    """
    A wrapper function for functions here in the get_corrections module
    Search for special options like ["cmt_run_id", "prefix", "suffix"] and apply arg shuffling accordingly
    Example confs:
        ('cmt_run_id', cmt_run_id, 'to_pe_model', 'ONLINE', True)
        ('suffix', suffix, 'cmt_run_id', cmt_run_id, 'to_pe_model', 'ONLINE', True)

    :param get_correction_function: A function here in the get_corrections module
    :returns: The function wrapped with the option search
    """
    @wraps(get_correction_function)
    def correction_options_wrapped(run_id, conf, *arg):
        if isinstance(conf, tuple):
            set_prefix = ['prefix', False, None]
            set_suffix = ['suffix', False, None]
            set_cmt_run_id = ['cmt_run_id', False, None]

            for tag in [set_cmt_run_id, set_prefix, set_suffix]:
                if tag[0] in conf:
                    i_tag = conf.index(tag[0])
                    tag[:] = [tag[0], True, conf[i_tag + 1]]
                    conf = [item for i, item in enumerate(conf) if i not in [i_tag, i_tag + 1]]

            if len(conf) == 1:
                conf = conf[0]
            else:
                if set_prefix[1]:
                    conf[0] = set_prefix[2] + '_' + conf[0]
                if set_suffix[1]:
                    conf[0] = conf[0] + '_' + set_suffix[2]
                if set_cmt_run_id[1]:
                    run_id = set_cmt_run_id[2]

                conf = tuple(conf)

        # Else use the get corrections as they are
        return get_correction_function(run_id, conf, *arg)

    return correction_options_wrapped


@export
@correction_options
def get_correction_from_cmt(run_id, conf):
    """
    Get correction from CMT general format is
    conf = ('correction_name', 'version', True)
    where True means looking at nT runs, e.g. 
    get_correction_from_cmt(run_id, conf[:2])
    special cases:
    version can be replaced by consant int, float or array
    when user specify value(s)
    :param run_id: run id from runDB
    :param conf: configuration 
    :return: correction value(s)
    """
 
    if isinstance(conf, str):
        # Legacy support for pax files
        return conf

    elif isinstance(conf, tuple) and len(conf) == 2:
        model_conf, cte_value = conf[:2]

        # special case constant to_pe values 
        if model_conf in FIXED_TO_PE:
            correction = FIXED_TO_PE[model_conf]
            return correction

        # special case constant single value or list of values.
        elif 'constant' in model_conf:
            if not isinstance(cte_value, (float, int, str, list)):
                raise ValueError(f"User specify a model type {model_conf} "
                                 "and should provide a number or list of numbers. Got: "
                                 f"{type(cte_value)}")
            correction = cte_value
            return correction

    elif isinstance(conf, tuple) and len(conf) == 3:
        model_conf, global_version, is_nt = conf[:3]
        cmt = straxen.CorrectionsManagementServices(is_nt=is_nt)
        correction = cmt.get_corrections_config(run_id, conf[:2])
        if correction.size == 0:
            raise ValueError(f"Could not find a value for {model_conf} "
                             f"please check it is implemented in CMT. ")

        if model_conf in corrections_w_file: # file's name (maps, NN, etc) 
            correction = ' '.join(map(str, correction))
            return correction

        elif model_conf in single_value_corrections:
            if 'samples' in model_conf: # int baseline samples, etc
                return int(correction)
            else:
                return float(correction) # float elife, drift velocity, etc
        
        elif model_conf in arrays_corrections:
            np_correction = correction.reshape(correction.size)
            np_correction = np_correction.astype(np.int16)  # not sure if straxen can handle dtype:object therefore specify dtype
            return np_correction
        
        return correction
    
    else:
        raise ValueError("Wrong configuration. "
                         "Please use the following format: "
                         "(config->str, model_config->str or number, is_nT->bool) "
                         f"User specify {conf} please modify")


@export
def get_cmt_resource(run_id, conf, fmt=''):
    """
    Get resource with CMT correction file name
    """
    return straxen.get_resource(get_correction_from_cmt(run_id, conf), fmt=fmt)


@export
def is_cmt_option(config):
    """
    Check if the input configuration is cmt style.
    """
    return _is_cmt_option(None, config)


@correction_options
def _is_cmt_option(run_id, config):
    # Compatibilty with 
    if isinstance(config, str) and "cmt://" in config:
        return True
    is_cmt = (isinstance(config, tuple)
              and len(config)==3
              and isinstance(config[0], str)
              and isinstance(config[1], (str, int, float))
              and isinstance(config[2], bool))
    
    return is_cmt


def get_cmt_options(context):
    """
    Function which loops over all plugin configs and returns dictionary
    with option name as key and current settings as values.

    :param context: Context with registered plugins.
    """
    cmt_options = {}
    for data_type, plugin in context._plugin_class_registry.items():
        for option_key, option in plugin.takes_config.items():
            if (option_key in context.config and
                straxen.get_corrections.is_cmt_option(context.config[option_key])
                ):
                cmt_options[option_key] = context.config[option_key]
            elif straxen.get_corrections.is_cmt_option(option.default):
                cmt_options[option_key] = option.default

    return cmt_options


FIXED_TO_PE = {
    'to_pe_placeholder': np.repeat(0.0085, straxen.n_tpc_pmts),
    '1T_to_pe_placeholder' : np.array([0.007, 0., 0., 0.008, 0.004, 0.008, 0.004, 0.008, 0.007, 0.005, 0.007, 0.006, 0., 0.006, 0.008, 0.007, 0.006, 0.009,0.007, 0.007, 0.007, 0.012, 0.004, 0.008, 0.005, 0.008, 0., 0., 0.007, 0.007, 0.004, 0., 0.004, 0.007, 0., 0.005,0.007, 0.007, 0.005, 0.005, 0.008, 0.006, 0.005, 0.007, 0.006, 0.007, 0.008, 0.005, 0.008, 0.008, 0.005, 0.005, 0.007, 0.008, 0.005, 0.009, 0.004, 0.005, 0.01 , 0.008, 0.006, 0.016, 0., 0.005, 0.005, 0., 0.01, 0.008, 0.004, 0.006, 0.005, 0., 0.008, 0., 0.004, 0.004, 0.006, 0.005, 0.012, 0., 0.005,0.004, 0.004, 0.008, 0.007, 0.012, 0., 0., 0., 0.007, 0.007, 0., 0.005, 0.008, 0.006, 0.004, 0.004, 0.006, 0.008,0.008, 0.008, 0.006, 0., 0.007, 0.005, 0.005, 0.005, 0.007,0.004, 0.008, 0.007, 0.008, 0.008, 0.006, 0.006, 0.01, 0.005,0.008, 0., 0.012, 0.007, 0.004, 0.008, 0.007, 0.007, 0.008,0.003, 0.004, 0.007, 0.006, 0., 0.005, 0.004, 0.005, 0., 0., 0.004, 0., 0.004, 0., 0.004, 0., 0.011, 0.005,0.006, 0.005, 0.004, 0.004, 0., 0.007, 0., 0.004, 0., 0.005, 0.006, 0.007, 0.005, 0.008, 0.004, 0.006, 0.008, 0.007,0., 0.008, 0.008, 0.007, 0.007, 0., 0.008, 0.004, 0.004,0.005, 0.004, 0.007, 0.008, 0.004, 0.006, 0.006, 0., 0.007,0.004, 0.004, 0.005, 0., 0.008, 0.004, 0.004, 0.004, 0.008,0.008, 0., 0.006, 0.005, 0.004, 0.005, 0.008, 0.008, 0.008,0., 0.005, 0.008, 0., 0.008, 0., 0.004, 0.012, 0., 0.005, 0.007, 0.009, 0.005, 0.004, 0.004, 0., 0., 0.004,0.004, 0.011, 0.004, 0.004, 0.007, 0.004, 0.005, 0.004, 0.005,0.007, 0.004, 0.006, 0.006, 0.004, 0.008, 0.005, 0.007, 0.007,0., 0.004, 0.007, 0.008, 0.004, 0., 0.007, 0.004, 0.004, 0.004, 0., 0.004, 0.005, 0.004]),
    # Gains which will preserve all areas in adc counts.
    # Useful for debugging and tests.
    'adc_tpc': np.ones(straxen.n_tpc_pmts),
    'adc_mv': np.ones(straxen.n_mveto_pmts),
    'adc_nv': np.ones(straxen.n_nveto_pmts)
}
