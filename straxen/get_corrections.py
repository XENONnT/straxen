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

    if isinstance(conf, str) and conf.startswith('https://raw'):
        # Legacy support for pax files
        x = straxen.get_resource(elife_conf, fmt='npy')
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
                    f"Gain model {gain_model} resulted in a to_pe "
                    f"of length {len(to_pe)}, but n_pmts is {n_pmts}!")
                
            return to_pe
        else:
            raise ValueError(
                "User must add his/her own values to FIXED_TO_PE as: "
                "FIXED_TO_PE ={str(my_constant_gains): np.repeat(0.005, straxen.n_tpc_pmts)}")            

    else:
        raise NotImplementedError(f"Gain model type {model_conf} not implemented")

FIXED_TO_PE = {
    'to_pe_placeholder': np.repeat(0.0085, straxen.n_tpc_pmts),
    '1T_to_pe_placeholder' : np.array([0.0072452, 0., 0., 0.00813198, 0.00438829, 0.0079016, 0.00357636, 0.00752925, 0.00743175, 0.00483737, 0.00706977, 0.00586599, 0., 0.00556236, 0.00797391, 0.00704167, 0.00640926, 0.00850643, 0.00714517, 0.00742941, 0.00715024, 0.01209479, 0.00397228, 0.00754782, 0.00540989, 0.00763518, 0., 0., 0.00659082, 0.00727863, 0.00422917, 0., 0.00413345, 0.0070529, 0., 0.00536641, 0.00743007, 0.00704821, 0.00456053, 0.00518346, 0.00752871, 0.00564258, 0.00459131, 0.00734189, 0.00612753, 0.00655326, 0.00759871, 0.00476416, 0.00802808, 0.00760099, 0.0045909, 0.00460675, 0.00714769, 0.00800725, 0.0046979, 0.00866425, 0.00374124, 0.00496461, 0.01035307, 0.00758412, 0.00603282, 0.01618727, 0., 0.00450775, 0.00483394, 0., 0.00981709, 0.00780705, 0.00357422, 0.00565691, 0.00456281, 0., 0.00803384, 0., 0.00358838, 0.0036835, 0.00588289, 0.00513244, 0.01175829, 0., 0.0050855, 0.0040911, 0.00386543, 0.00816158, 0.0067502, 0.01204568, 0., 0., 0., 0.00651278, 0.00742206, 0., 0.00491631, 0.00769847, 0.00582819, 0.00406426, 0.00400214, 0.00577728, 0.00814137, 0.00763981, 0.00761573, 0.00554446, 0., 0.00715309, 0.00503238, 0.00459783, 0.00492299, 0.00745983, 0.00357002, 0.00759856, 0.00717, 0.00816608, 0.00767994, 0.00604421, 0.00587048, 0.00964442, 0.00468335, 0.00829553, 0., 0.01194368, 0.00698784, 0.00363265, 0.00751866, 0.00745633, 0.00745376, 0.00769538, 0.00348097, 0.00362781, 0.00746885, 0.00638465, 0., 0.00512748, 0.00372908, 0.00523452, 0., 0., 0.00444336, 0., 0.00388768, 0., 0.00359526, 0., 0.01064339, 0.00510303, 0.00646788, 0.00508451, 0.00428273, 0.00350879, 0., 0.00728879, 0., 0.00425876, 0., 0.00471755, 0.00627014, 0.00728991, 0.00537106, 0.00848785, 0.00385989, 0.00556679, 0.00751004, 0.00731159, 0., 0.00779083, 0.00760312, 0.00667208, 0.00731918, 0., 0.00759461, 0.00381062, 0.00359802, 0.00521435, 0.00429271, 0.00735201, 0.00776976, 0.00399636, 0.00622611, 0.00585986, 0., 0.0073606, 0.00358086, 0.00358621, 0.00526905, 0., 0.00770451, 0.00405925, 0.00430389, 0.00421667, 0.0076157 , 0.00764557, 0., 0.00642047, 0.0048628 , 0.00372671, 0.00510367, 0.0076145 , 0.00765479, 0.00770955, 0., 0.00526101, 0.00803205, 0., 0.00792361, 0., 0.00440089, 0.01168991, 0., 0.004523, 0.00704132, 0.00884788, 0.00513383, 0.00445313, 0.00404645, 0., 0., 0.00354486, 0.00417448, 0.01137993, 0.00366378, 0.00368374, 0.00726986, 0.00358424, 0.00503405, 0.00360588, 0.00541749, 0.00742327, 0.00374311, 0.00621805, 0.00647762, 0.00370024, 0.00771456, 0.00457882, 0.00746426, 0.00674823, 0., 0.00366232, 0.00749439, 0.00756156, 0.00365761, 0., 0.00733116, 0.00405427, 0.00375715, 0.00410382, 0., 0.00429508, 0.00494906, 0.00378778]),
    # Gains which will preserve all areas in adc counts.
    # Useful for debugging and tests.
    'adc_tpc': np.ones(straxen.n_tpc_pmts),
    'adc_mv': np.ones(straxen.n_mveto_pmts),
    'adc_nv': np.ones(straxen.n_nveto_pmts)
}


@export
@correction_options
def get_correction_from_cmt(run_id, conf):
    if isinstance(conf, str) and conf.startswith('https://raw'):
        # Legacy support for pax files
        return conf
    if isinstance(conf, tuple) and len(conf) == 3:
        is_nt = conf[-1]
        model_conf, global_version = conf[:2]
        correction = global_version  # in case is a single value
        if 'constant' in model_conf:
            if not isinstance(global_version, (float, int)):
                raise ValueError(f"User specify a model type {model_type} "
                                 "and should provide a number. Got: "
                                 f"{type(global_version)}")
        else:
            cmt = straxen.CorrectionsManagementServices(is_nt=is_nt)
            correction = cmt.get_corrections_config(run_id, conf[:2])
            if correction.size == 0:
                raise ValueError(f"Could not find a value for {model_type} "
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

