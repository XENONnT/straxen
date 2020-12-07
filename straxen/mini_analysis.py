import inspect
import textwrap

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
def mini_analysis(requires=tuple(),
                  hv_bokeh=False,
                  warn_beyond_sec=None,
                  default_time_selection='touching'):
    def decorator(f):
        parameters = inspect.signature(f).parameters

        def wrapped_f(context: strax.Context, run_id: str, **kwargs):
            # Validate arguments
            known_kwargs = (
                'time_range seconds_range time_within time_selection '
                'ignore_time_warning '
                'selection_str t_reference to_pe config').split()
            for k in kwargs:
                if k not in known_kwargs and k not in parameters:
                    # Python itself also raises TypeError for invalid kwargs
                    raise TypeError(f"Unknown argument {k} for {f.__name__}")

            if 'config' in kwargs:
                context = context.new_context(config=kwargs['config'])
            if 'config' in parameters:
                kwargs['config'] = context.config

            # Say magic words to enable holoviews
            if hv_bokeh:
                global _hv_bokeh_initialized
                if not _hv_bokeh_initialized:
                    import holoviews
                    holoviews.extension('bokeh')
                    _hv_bokeh_initialized = True

            # TODO: This is a placeholder until the corrections system
            # is more fully developed
            if 'to_pe' in parameters and 'to_pe' not in kwargs:
                kwargs['to_pe'] = straxen.get_to_pe(
                    run_id,
                    context.config['gain_model'],
                    context.config['n_tpc_pmts'])

            # Prepare selection arguments
            kwargs['time_range'] = context.to_absolute_time_range(
                run_id,
                targets=requires,
                **{k: kwargs.get(k)
                   for k in ('time_range seconds_range time_within'.split())})
            kwargs.setdefault('time_selection', default_time_selection)
            kwargs.setdefault('selection_str', None)

            kwargs['t_reference'], _ = context.estimate_run_start_and_end(
                run_id, requires)

            if warn_beyond_sec is not None and not kwargs.get('ignore_time_warning'):
                tr = kwargs['time_range']
                if tr is None:
                    sec_requested = float('inf')
                else:
                    sec_requested = (tr[1] - tr[0]) / int(1e9)
                if sec_requested > warn_beyond_sec:
                    tr_str = "the entire run" if tr is None else f"{sec_requested} seconds"
                    raise ValueError(
                        f"The author of this mini analysis recommends "
                        f"not requesting more than {warn_beyond_sec} seconds. "
                        f"You are requesting {tr_str}. If you wish to proceed, "
                        "pass ignore_time_warning = True.")

            # Load required data, if any
            if len(requires):
                deps_by_kind = strax.group_by_kind(
                    requires, context=context)
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
                            time_selection=kwargs['time_selection'],
                            # Arguments for new context, if needed
                            config=kwargs.get('config'),
                            register=kwargs.get('register'),
                            storage=kwargs.get('storage', tuple()),
                            progress_bar=False,
                        )

                # If user did not give time kwargs, but the function expects
                # a time_range, try to add one based on the time range of the data
                base_dkind = list(deps_by_kind.keys())[0]
                x = kwargs[base_dkind]
                if len(x) and kwargs.get('time_range') is None:
                    x0 = x.iloc[0] if isinstance(x, pd.DataFrame) else x[0]
                    try:
                        kwargs.setdefault('time_range', (x0['time'],
                                                         strax.endtime(x).max()))

                    except AttributeError:
                        # If x is a holoviews dataset, this will fail.
                        pass

            if 'seconds_range' in parameters:
                if kwargs.get('time_range') is None:
                    scr = None
                else:
                    scr = tuple([(t - kwargs['t_reference']) / int(1e9)
                                 for t in kwargs['time_range']])
                kwargs.setdefault('seconds_range', scr)

            kwargs.setdefault('run_id', run_id)
            kwargs.setdefault('context', context)

            if 'kwargs' in parameters:
                # Likely this will be passed to another mini-analysis
                to_pass = kwargs
                # Do not pass time_range and seconds_range both (unless explicitly requested)
                # strax does not like that
                if 'seconds_range' in to_pass and not 'seconds_range' in parameters:
                    del to_pass['seconds_range']
                if 'time_within' in to_pass and not 'time_within' in parameters:
                    del to_pass['time_within']
            else:
                # Pass only arguments the function wants
                to_pass = {k: v for k, v in kwargs.items()
                           if k in parameters}
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
