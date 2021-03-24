import numpy as np
import strax
import straxen
from warnings import warn

export, __all__ = strax.exporter()
__all__ += ['FIXED_TO_PE']


@export
def get_to_pe(run_id, gain_model, n_pmts):
    if not isinstance(gain_model, tuple):
        raise ValueError(f"gain_model must be a tuple")
    if not len(gain_model) == 2:
        raise ValueError(f"gain_model must have two elements: "
                         f"the model type and its specific configuration")
    model_type, model_conf = gain_model

    if model_type == 'disabled':
        # Somebody messed up
        raise RuntimeError("Attempt to use a disabled gain model")
    if model_type == 'CMT_model':
        if not isinstance(model_conf, tuple) or len(model_conf) != 2:
            # Raise a value error if the condition is not met. We should have:
            # ("CMT_model", -> To specify that we want to use the online
            #                  corrections management tool
            #   (
            #   "to_pe_model", -> This is to specify that we want  the gains
            #   "v1", -> The version of the correction 'v1' is the online version
            #   )
            raise ValueError('CMT gain model should be similar to:'
                             '("CMT_model", ("to_pe_model", "v1"). Instead got:'
                             f'{model_conf}')
        # is this the best way to do this?
        is_nt = n_pmts == straxen.n_tpc_pmts or n_pmts == straxen.n_nveto_pmts or n_pmts == straxen.n_mveto_pmts

        corrections = straxen.CorrectionsManagementServices(is_nt=is_nt)
        to_pe = corrections.get_corrections_config(run_id, model_conf)

        return to_pe

    elif model_type == 'to_pe_per_run':
        warn("to_pe_per_run will be replaced by CorrectionsManagementSevices",
             DeprecationWarning, 2)
        # Load a npy file specifying a run_id -> to_pe array
        to_pe_file = model_conf
        x = straxen.get_resource(to_pe_file, fmt='npy')
        run_index = np.where(x['run_id'] == int(run_id))[0]
        if not len(run_index):
            # Gains not known: using placeholders
            run_index = [-1]
        to_pe = x[run_index[0]]['to_pe']

    elif model_type == 'to_pe_constant':
        if model_conf in FIXED_TO_PE:
            return FIXED_TO_PE[model_conf]

        try:
            # Uniform gain, specified as a to_pe factor
            to_pe = np.ones(n_pmts, dtype=np.float32) * model_conf
        except np.core._exceptions.UFuncTypeError as e:
            raise (str(e) +
                   f'\nTried multiplying by {model_conf}. Insert a number instead.')
    else:
        raise NotImplementedError(f"Gain model type {model_type} not implemented")

    if len(to_pe) != n_pmts:
        raise ValueError(
            f"Gain model {gain_model} resulted in a to_pe "
            f"of length {len(to_pe)}, but n_pmts is {n_pmts}!")
    return to_pe


FIXED_TO_PE = {
    # just some dummy placeholder for nT gains
    'gain_placeholder': np.repeat(0.0085, straxen.n_tpc_pmts),
    # Gains which will preserve all areas in adc counts.
    # Useful for debugging and tests.
    'adc_tpc': np.ones(straxen.n_tpc_pmts),
    'adc_mv': np.ones(straxen.n_mveto_pmts),
    'adc_nv': np.ones(straxen.n_nveto_pmts)
}


@export
def get_elife(run_id, elife_conf):
    if isinstance(elife_conf, tuple) and len(elife_conf) == 3:
        # We want to use corrections management
        is_nt = elife_conf[-1]
        cmt = straxen.CorrectionsManagementServices(is_nt=is_nt)

        e = cmt.get_corrections_config(run_id, elife_conf[:2])

    elif isinstance(elife_conf, str):
        warn("get_elife will be replaced by CorrectionsManagementSevices",
             DeprecationWarning, 2)
        # Let's remove these functions and only rely on the CMT in the future
        x = straxen.get_resource(elife_conf, fmt='npy')
        run_index = np.where(x['run_id'] == int(run_id))[0]
        if not len(run_index):
            # Electron lifetime not known: using placeholders
            e = 623e3
        else:
            e = x[run_index[0]]['e_life']
    else:
        raise ValueError(
            'Wrong elife model. Either specify a string (url) or the '
            'Corrections Management Tools format: '
            '(model_type->str, model_config->str, is_nT->bool)'
            '')
    return float(e)


@export
def get_config_from_cmt(run_id, conf):
    if isinstance(conf, str) and conf.startswith('https://raw'):
        # Legacy support for pax files
        return conf
    if not isinstance(conf, tuple):
        raise ValueError(f'conf must be a tuple')
    if not len(conf) == 3:
        raise ValueError(f'conf must have three elements: '
                         f'the model type, its specific configuration '
                         f'and detector (True = nT)')
    model_type, model_conf, is_nt = conf
    if model_type == 'CMT_model':
        cmt = straxen.CorrectionsManagementServices(is_nt=is_nt)
        this_file = cmt.get_corrections_config(run_id, model_conf)
        this_file = ' '.join(map(str, this_file))

    else:
        raise ValueError(f'Wrong NN configuration, please look at this {nn_config} '
                         'and modify it accordingly')

    return this_file
