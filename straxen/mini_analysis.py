import inspect
import textwrap

import holoviews as hv
import pandas as pd
import strax
from strax.context import select_docs
import straxen


export, __all__ = strax.exporter()

ma_doc = """
This is a straxen mini-analysis.
The method takes run_id as its only positional argument,
and additional arguments through keywords only.

The function requires the data types: {requires}.
Unless you specify this through data_kind = array keyword arguments,
this data will be loaded automatically. 

The function takes the same selection arguments as context.get_array:
""" + select_docs

_hv_bokeh_initialized = False

@export
def mini_analysis(requires=tuple(), hv_bokeh=False):

    def decorator(f):
        parameters = inspect.signature(f).parameters

        def wrapped_f(context: strax.Context, run_id: str, **kwargs):
            # Validate arguments
            known_kwargs = (
                'time_range seconds_range time_within time_selection '
                'selection_str t_reference to_pe config').split()
            for k in kwargs:
                if k not in known_kwargs and k not in parameters:
                    # Python itself also raises TypeError for invalid kwargs
                    raise TypeError(f"Unknown argument {k} for {f.__name__}")

            if 'config' in kwargs:
                context = context.new_context(config=kwargs['config'])

            # Say magic words to enable holoviews
            if hv_bokeh:
                global _hv_bokeh_initialized
                if not _hv_bokeh_initialized:
                    hv.extension('bokeh')
                    _hv_bokeh_initialized = True

            # TODO: This is a placeholder until the corrections system
            # is more fully developed
            if 'to_pe' in parameters and 'to_pe' not in kwargs:
                kwargs['to_pe'] = straxen.get_to_pe(
                    run_id,
                    'https://raw.githubusercontent.com/XENONnT/'
                    'strax_auxiliary_files/master/to_pe.npy')

            # Prepare selection arguments
            kwargs['time_range'] = context.to_absolute_time_range(
                run_id,
                targets=requires,
                **{k: kwargs.get(k)
                   for k in ('time_range seconds_range time_within'.split())})
            kwargs.setdefault('time_selection', 'fully_contained')
            kwargs.setdefault('selection_str', None)

            if ('t_reference' in parameters
                    and kwargs.get('t_reference') is None):
                kwargs['t_reference'] = context.estimate_run_start(
                    run_id, requires)

            # Load required data
            deps_by_kind = strax.group_by_kind(
                requires, context=context, require_time=False)
            for dkind, dtypes in deps_by_kind.items():
                if dkind in kwargs:
                    # Already have data, just apply cuts
                    kwargs[dkind] = context.apply_selection(
                        kwargs[dkind],
                        selection_str=kwargs['selection_str'],
                        time_range=kwargs['time_range'],
                        time_selection=kwargs['time_selection'])
                else:
                    kwargs[dkind] = context.get_array(
                        run_id,
                        dtypes,
                        selection_str=kwargs['selection_str'],
                        time_range=kwargs['time_range'],
                        time_selection=kwargs['time_selection'])

            # If user did not give time kwargs, but the function expects
            # a time_range, add them based on the time range of the data
            # (if there is no data, otherwise give (NaN, NaN))
            if kwargs.get('time_range') is None:
                base_dkind = list(deps_by_kind.keys())[0]
                x = kwargs[base_dkind]
                x0 = x.iloc[0] if isinstance(x, pd.DataFrame) else x[0]
                kwargs['time_range'] = (
                    (x0['time'], strax.endtime(x).max()) if len(x)
                    else (float('nan'), float('nan')))

            # Pass only the arguments the function wants
            to_pass = dict()
            for k in parameters:
                if k == 'run_id':
                    to_pass['run_id'] = run_id
                elif k == 'context':
                    to_pass['context'] = context
                elif k in kwargs:
                    to_pass[k] = kwargs[k]
                # If we get here, let's hope the function defines a default...

            return f(**to_pass)

        wrapped_f.__name__ = f.__name__

        if hasattr(f, '__doc__') and f.__doc__:
            doc_lines = f.__doc__.splitlines()
            wrapped_f.__doc__ = (doc_lines[0]
                               + '\n'
                               + textwrap.dedent(
                        '\n'.join(doc_lines[1:])))
        else:
            wrapped_f.__doc__ = \
                ('Straxen mini-analysis for which someone was too lazy'
                 'to write a proper docstring')

        wrapped_f.__doc__ += ma_doc.format(requires=', '.join(requires))

        strax.Context.add_method(wrapped_f)
        return wrapped_f

    return decorator
