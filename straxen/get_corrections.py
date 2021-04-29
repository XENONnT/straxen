import numpy as np
import strax
import straxen
from warnings import warn
from functools import wraps
from straxen.corrections_services import corrections_w_file

export, __all__ = strax.exporter()
__all__ += ['FIXED_TO_PE']


def correction_options(get_correction_function):
    """
    A wrapper function for functions here in the get_corrections module
    Search for special options like ["MC", ] and apply arg shuffling accordingly
    Example: get_to_pe(
        run_id, ('MC', cmt_run_id, 'CMT_model', ('to_pe_model', 'ONLINE')), n_tpc_pmts])

    :param get_correction_function: A function here in the get_corrections module
    :returns: The function wrapped with the option search
    """
    @wraps(get_correction_function)
    def correction_options_wrapped(run_id, conf, *arg):
        if isinstance(conf, tuple):
            # MC chain simulation can put run_id inside configuration
            if 'MC' in conf:
                tag, cmt_run_id, *conf = conf
                if tag != 'MC':
                    raise ValueError('Get corrections require input in the from of tuple '
                                     '("MC", run_id, *conf) when "MC" option is invoked')
                return get_correction_function(cmt_run_id, tuple(conf), *arg)

        # Else use the get corrections as they are
        return get_correction_function(run_id, conf, *arg)

    return correction_options_wrapped


@export
@correction_options
def get_to_pe(run_id, conf, n_pmts):
    """
    To pe values for TPC, n-veto mu-veto PMTs, this is also now as gains
    :param run_id: run id from runDB
    :param conf: configuration 
    :n_pmts: number of PMTs
    :return: array of to_pe values
    """
    if isinstance(conf, str) and conf.startswith('https://raw'):
        # Legacy support for pax files
        x = straxen.get_resource(conf, fmt='npy')
        run_index = np.where(x['run_id'] == int(run_id))[0]
        if not len(run_index):
            # Electron lifetime not known: using placeholders
            e = 623e3
        else:
            e = x[run_index[0]]['e_life']
        return float(e)

    elif isinstance(conf, tuple) and len(conf) == 3:
        is_nt = conf[-1]
        model_conf, global_version = conf[:2]
        if 'to_pe_model' in model_conf:
            corrections = straxen.CorrectionsManagementServices(is_nt=is_nt)
            to_pe = corrections.get_corrections_config(run_id, conf[:2])

            return to_pe
        else:
            raise ValueError("Wrong configuration. "
                         "Please use the following format: "
                         "(config->str, model_config->str or number, is_nT->bool) "
                         f"User specify {conf} please modify")


    elif isinstance(conf, tuple) and len(conf) == 2:
        model_conf = conf[0]
        if model_conf in FIXED_TO_PE:
            to_pe = FIXED_TO_PE[model_conf]
            if len(to_pe) != n_pmts:
                raise ValueError(
                    f"Gain model {model_conf} resulted in a to_pe "
                    f"of length {len(to_pe)}, but n_pmts is {n_pmts}!")
                
            return to_pe
        else:
            raise ValueError(
                "User must add his/her own values to FIXED_TO_PE as: "
                "FIXED_TO_PE ={str(my_constant_gains): np.repeat(0.005, straxen.n_tpc_pmts)}")            

    else:
        raise NotImplementedError(f"To pe model type {conf} not implemented")

FIXED_TO_PE = {
    'to_pe_placeholder': np.repeat(0.0085, straxen.n_tpc_pmts),
    '1T_to_pe_placeholder' : np.array([0.007, 0., 0., 0.008, 0.004, 0.008, 0.004, 0.008, 0.007, 0.005, 0.007, 0.006, 0., 0.006, 0.008, 0.007, 0.006, 0.009,0.007, 0.007, 0.007, 0.012, 0.004, 0.008, 0.005, 0.008, 0., 0., 0.007, 0.007, 0.004, 0., 0.004, 0.007, 0., 0.005,0.007, 0.007, 0.005, 0.005, 0.008, 0.006, 0.005, 0.007, 0.006, 0.007, 0.008, 0.005, 0.008, 0.008, 0.005, 0.005, 0.007, 0.008, 0.005, 0.009, 0.004, 0.005, 0.01 , 0.008, 0.006, 0.016, 0., 0.005, 0.005, 0., 0.01, 0.008, 0.004, 0.006, 0.005, 0., 0.008, 0., 0.004, 0.004, 0.006, 0.005, 0.012, 0., 0.005,0.004, 0.004, 0.008, 0.007, 0.012, 0., 0., 0., 0.007, 0.007, 0., 0.005, 0.008, 0.006, 0.004, 0.004, 0.006, 0.008,0.008, 0.008, 0.006, 0., 0.007, 0.005, 0.005, 0.005, 0.007,0.004, 0.008, 0.007, 0.008, 0.008, 0.006, 0.006, 0.01, 0.005,0.008, 0., 0.012, 0.007, 0.004, 0.008, 0.007, 0.007, 0.008,0.003, 0.004, 0.007, 0.006, 0., 0.005, 0.004, 0.005, 0., 0., 0.004, 0., 0.004, 0., 0.004, 0., 0.011, 0.005,0.006, 0.005, 0.004, 0.004, 0., 0.007, 0., 0.004, 0., 0.005, 0.006, 0.007, 0.005, 0.008, 0.004, 0.006, 0.008, 0.007,0., 0.008, 0.008, 0.007, 0.007, 0., 0.008, 0.004, 0.004,0.005, 0.004, 0.007, 0.008, 0.004, 0.006, 0.006, 0., 0.007,0.004, 0.004, 0.005, 0., 0.008, 0.004, 0.004, 0.004, 0.008,0.008, 0., 0.006, 0.005, 0.004, 0.005, 0.008, 0.008, 0.008,0., 0.005, 0.008, 0., 0.008, 0., 0.004, 0.012, 0., 0.005, 0.007, 0.009, 0.005, 0.004, 0.004, 0., 0., 0.004,0.004, 0.011, 0.004, 0.004, 0.007, 0.004, 0.005, 0.004, 0.005,0.007, 0.004, 0.006, 0.006, 0.004, 0.008, 0.005, 0.007, 0.007,0., 0.004, 0.007, 0.008, 0.004, 0., 0.007, 0.004, 0.004, 0.004, 0., 0.004, 0.005, 0.004]),
    # Gains which will preserve all areas in adc counts.
    # Useful for debugging and tests.
    'adc_tpc': np.ones(straxen.n_tpc_pmts),
    'adc_mv': np.ones(straxen.n_mveto_pmts),
    'adc_nv': np.ones(straxen.n_nveto_pmts)
}


@export
@correction_options
def get_correction_from_cmt(run_id, conf):
    """
    Get correction from CMT
    :param run_id: run id from runDB
    :param conf: configuration 
    :return: correction
    """
 
    if isinstance(conf, str) and conf.startswith('https://raw'):
        # Legacy support for pax files
        return conf
    if isinstance(conf, tuple) and len(conf) == 3:
        is_nt = conf[-1]
        model_conf, global_version = conf[:2]
        correction = global_version  # in case is a single value
        if 'constant' in model_conf:
            if not isinstance(global_version, (float, int)):
                raise ValueError(f"User specify a model type {model_conf} "
                                 "and should provide a number. Got: "
                                 f"{type(global_version)}")
        else:
            cmt = straxen.CorrectionsManagementServices(is_nt=is_nt)
            correction = cmt.get_corrections_config(run_id, conf[:2])
            if correction.size == 0:
                raise ValueError(f"Could not find a value for {model_conf} "
                                 "please check it is implemented in CMT. "
                                 f"for nT = {is_nt}")
            if model_conf in corrections_w_file:
                this_file = ' '.join(map(str, correction))
                return this_file

        if 'samples' in model_conf:
            return int(correction)

        else:
            return float(correction)

    else:
        raise ValueError("Wrong configuration. "
                         "Please use the following format: "
                         "(config->str, model_config->str or number, is_nT->bool) "
                         f"User specify {conf} please modify")
