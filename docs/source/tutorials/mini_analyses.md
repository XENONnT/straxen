Mini-analyses
==============
Straxen includes some "canned" mini-analyses that you can invoke easily from any context. These usually make a single plot, though they could also return information. Just pass them a run_id and/or some time selection options:

.. code-block:: python

    st = strax.contexts.demo()
    st.waveform_display(run_id, seconds_range=(0, 0.1))

You can also run the analysis on a part of a run by specifying a time range and/or a selection string, just like for `get_df` and `get_array`.


Developing mini-analyses
-------------------------

A mini-analysis is just a function. It can do anything, such as making plots or returning values. Declaring your function a mini-analysis only means:
- It becomes available as a context method;
- It's required data is automatically loaded using that context if the user does not pass it.

To declare a function a mini-analysis, just decorate it with `@straxen.mini_analysis`. Here's an example:

.. code-block:: python

    @straxen.mini_analysis(requires=[
        'records',
        ['peaks', ('peaks', 'peak_basics', 'peak_classification')]
    ])
    def waveform_display(to_pe, t0, t1, records, peaks):

Note this adds the function as a method for *all* strax contexts in the current python session. So don't call mini-analyses you plan to commit to straxen `plot` or `doit` (and certainly not `get_df`...).

The `requires` argument declares that the function takes data of two kinds: `records` and `peaks`. The `peaks` array must contain data from both `peaks`, `peak_basics`, and `peak_classification`. This declaration might be simplified in the future.

Besides the data kinds (`records` and `peaks` in the example above), a mini-analysis function can take the following special arguments (in any order, as they are passed via **kwargs):
  * context: the strax Context being used
  * run_id: the run_id being loaded
  * to_pe: the PMT gains applying to the run_id being loaded
  * t_reference: start time of the run in epoch ns (unless overriden)
  * selection_str: Selection string used to get this data
  * time_range: (start, stop) time in ns since the epoch of the selected data interval
  * time_selection: kind of time selection used (fully_contained, touching, or skip)


Your analysis will always get these arguments (if you add them to your function), even if the user does not pass them. For time_range, you will get the absolute time in ns even if the user uses one of the other time arguments (such as seconds_range). If no time_range is passed by the user, you will get the time range of the full run. If run metadata is not available, this will be estimated from the data that is passed in / loaded (and if that is empty, you will get (NaN, Nan)).

If your analysis takes any other arguments, they must be keyword arguments. For example, `plot_pmt_pattern` takes an extra `array` argument:

.. code-block:: python

    @straxen.mini_analysis(requires=('records',))
    def plot_pmt_pattern(*, records, to_pe, array='bottom'):


Mini-analysis or plugin?
--------------------------

A mini-analysis takes all data from (a part of) a run and produces a single output, often a plot. A plugin produces data incrementally for a run, e.g. events, or new properties of events.

If you histogram or fit something, you'll want to make a mini-analysis (or just keep your code separate from straxen). If you compute some property for several things (peaks, events, or even chunks or seconds etc.) in a run, you want to make a plugin.
