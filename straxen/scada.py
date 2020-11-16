import urllib
import requests
import pandas as pd
import numba
import numpy as np
import warnings

import strax
import straxen

from straxen import uconfig

export, __all__ = strax.exporter()


@export
class SCADAInterface:

    def __init__(self, context=None):
        """
        Interface to excess the XENONnT slow control data via python.

        :param context: Context you are using e.g. st. This is needed
            if you would like to query data via run_ids.
        """
        try:
            self.SCData_URL = uconfig.get('scada', 'scdata_url')
            self.SCLastValue_URL = uconfig.get('scada', 'sclastvalue_url')
            self.SCADA_SECRETS = dict(QueryType=uconfig.get('scada', 'querytype'),
                                      username=uconfig.get('scada', 'username'),
                                      api_key=uconfig.get('scada', 'api_key')
                                      )
        except ValueError:
            raise ValueError(f'Cannot load SCADA information, from your xenon'
                             ' config. SCADAInterface cannot be used.')
        self.context = context

    def get_scada_values(self,
                         parameters,
                         start=None,
                         end=None,
                         run_id=None,
                         time_selection_kwargs={'full_range': True},
                         interpolation=False,
                         filling_kwargs={},
                         down_sampling=False,
                         value_every_seconds=1):
        """
        Function which returns XENONnT slow control values for a given
        set of parameters and time range.

        The time range can be either defined by a start and end time or
        via the run_id, target and context.

        :param parameters: dictionary containing the names of the
            requested scada-parameters. The keys are used as identifier
            of the parameters in the returned pandas.DataFrame.
        :param start: int representing the start time of the interval
            in ns unix time.
        :param end: same as start but as end.
        :param run_id: Id of the run. Can also be specified as a list or
            tuple of run ids. In this case we will return the time
            range lasting between the start of the first and endtime
            of the second run.
        :param time_selection_kwargs: Keyword arguments taken by
            st.to_absolute_time_range(). Default: full_range=True.
        :param interpolation: Boolean which decided to either forward
            fill empty values or to interpolate between existing ones.
        :param filling_kwargs: Kwargs applied to pandas .ffill() or
            .interpolate().
        :param down_sampling: Boolean which indicates whether to
            donw_sample result or to apply moving average. Moving average
            is deactivated in case of interpolated data.
        :param value_every_seconds: Defines with which time difference
            values should be returned. Must be an integer!
            Default: one value per 1 seconds.
        :return: pandas.DataFrame containing the data of the specified
            parameters.
        """
        if not isinstance(parameters, dict):
            mes = 'The argument "parameters" has to be specified as a dict.'
            raise ValueError(mes)

        if np.all((run_id, self.context)):
            # User specified a valid context and run_id, so get the start
            # and end time for our query:
            if isinstance(run_id, (list, tuple)):
                run_id = np.sort(run_id)  # Do not trust the user's
                start, _ = self.context.to_absolute_time_range(run_id[0], **time_selection_kwargs)
                _, end = self.context.to_absolute_time_range(run_id[-1], **time_selection_kwargs)
            else:
                start, end = self.context.to_absolute_time_range(run_id, **time_selection_kwargs)
        elif np.any(run_id):
            mes = ('You are trying to query slow control data via run_ids' 
                  ' but you have not specified the context you are '
                   'working with. Please set the context either via '
                   '.st = YOURCONTEXT, or when initializing the '
                   'interface.')
            raise ValueError(mes)

        if not np.all((start, end)):
            # User has not specified any vaild start and end time
            mes = ('You have to specify either a run_id and context.'
                   ' E.g. call get_scada_values(parameters, run_id=run)'
                   ' or you have to specifiy a valid start and end time '
                   'in utc unix time ns.')
            raise ValueError(mes)

        now = np.datetime64('now')
        if (end // 10**9) > now.astype(np.int64):
            mes = ('You are asking for an endtime which is in the future,'
                   ' I may be written by a physicist, but I am neither self-'
                   'aware nor can I predict the future like they can. You '
                   f'asked for the endtime: {end // 10**9} but current utc '
                   f'time is {now.astype(np.int64)}. I will return for the values for the '
                   'corresponding times as nans instead.')
            warnings.warn(mes)

        # Now loop over specified parameters and get the values for those.
        for ind, (k, p) in enumerate(parameters.items()):
            print(f'Start to query {k}: {p}')
            temp_df = self._query_single_parameter(start, end,
                                                   k, p,
                                                   value_every_seconds=value_every_seconds,
                                                   interpolation=interpolation,
                                                   filling_kwargs=filling_kwargs,
                                                   down_sampling=down_sampling
                                                   )

            if ind:
                m = np.all(df.loc[:, 'time'] == temp_df.loc[:, 'time'])
                mes = ('This is odd somehow the time stamps for the query of'
                       f' {p} does not match the other time stamps.')
                assert m, mes
                df = pd.concat((df, temp_df[k]), axis=1)
            else:
                df = temp_df

        # Adding timezone information and rename index:
        df.set_index('time', inplace=True)
        df = df.tz_localize(tz='UTC')
        df.index.rename('time UTC', inplace=True)

        if (end // 10**9) > now.astype(np.int64):
            df.loc[now:, :] = np.nan

        return df

    def _query_single_parameter(self,
                                start,
                                end,
                                parameter_key,
                                parameter_name,
                                interpolation,
                                filling_kwargs,
                                down_sampling,
                                value_every_seconds=1):
        """
        Function to query the values of a single parameter from SCData.

        :param start: Start time in ns unix time
        :param end: End time in ns unix time
        :param parameter_key: Key to identify queryed parameter in the
            DataFrame
        :param parameter_name: Parameter name in Scada/historian database.
        :param value_every_seconds: Defines with which time difference
            values should be returned. Must be an integer!
            Default: one value per 1 seconds.

        :returns: DataFrame with a time and parameter_key column.
        """
        if value_every_seconds < 1:
            mes = ("Scada takes only values every second. Cannot ask for a"
                   " higher sampling rate than one value per second. However"
                   f" you asked for one value every {value_every_seconds} seconds.")
            raise ValueError(mes)
        if not isinstance(value_every_seconds, int):
            raise ValueError('"value_every_seconds" must be an int!')

        # First we have to create an array where we can fill values with
        # the sampling frequency of scada:
        # TODO: Add a check in case user queries to many values. If yes read
        #  the data in chunks. How much are too many?
        seconds = np.arange(start, end + 1, 10**9)  # +1 to make sure endtime is included
        df = pd.DataFrame()
        df.loc[:, 'time'] = seconds
        df['time'] = df['time'].astype('<M8[ns]')
        df.set_index('time', inplace=True)

        # Check if first value is in requested range:
        query = self.SCADA_SECRETS.copy()
        query['name'] = parameter_name
        query['EndDateUnix'] = (start // 10**9) + 1  # +1 since it is end before exclusive
        query = urllib.parse.urlencode(query)
        values = requests.get(self.SCLastValue_URL + query)

        try:
            temp_df = pd.read_json(values.text)
        except ValueError:
            mes = values.text  # returns a dictionary as a string
            mes = eval(mes)
            raise ValueError(f'SCADA raised the following error "{mes["message"]}" '
                             f'when looking for your parameter "{parameter_name}"')

        # Store value as first value in our df
        df.loc[df.index.values[0], parameter_key] = temp_df['value'][0]

        # Query values between start+1 and end time:
        query = self.SCADA_SECRETS.copy()
        query["StartDateUnix"] = (start // 10**9) + 1
        query["EndDateUnix"] = (end // 10**9)
        query['name'] = parameter_name
        query = urllib.parse.urlencode(query)
        values = requests.get(self.SCData_URL + query)

        try:
            # Here we cannot do any better since the Error message returned
            # by the scada api is always the same...
            temp_df = pd.read_json(values.text)
            df.loc[temp_df['timestampseconds'], parameter_key] = temp_df.loc[:, 'value'].values
        except ValueError:
            pass

        # Let user decided whether to ffill or interpolate:
        if interpolation:
            df.interpolate(**filling_kwargs, inplace=True)
        else:
            # Now fill values in between like Scada would do:
            df.ffill(**filling_kwargs, inplace=True)
        df.reset_index(inplace=True)

        # Step 4. Down-sample data if asked for:
        if value_every_seconds > 1:
            if interpolation and not down_sampling:
                warnings.warn('Cannot use interpolation and running average at the same time.'
                              ' Deactivated the running average, switch to down_sampling instead.')
                down_sampling = True

            if down_sampling:
                df = df[::value_every_seconds]
            else:
                nt, nv = _average_scada(df['time'].astype(np.int64).values,
                                        df[parameter_key].values,
                                        value_every_seconds)
                df = pd.DataFrame()
                df['time'] = nt.astype('<M8[ns]')
                df[parameter_key] = nv

        return df

    def find_scada_parameter(self):
        # TODO: Add function which returns SCADA sensor names by short Name
        raise NotImplementedError('Feature not implemented yet.')


@export
def convert_time_zone(df, tz):
    """
    Function which converts the current time zone of a given
    pd.DataFrame into another timezone.

    :param df: pandas.DataFrame containing the Data. Index must be a
        datetime object with time zone information.
    :param tz: str representing the timezone the index should be
        converted to. See the notes for more information.
    :return: pandas.DataFrame with converted time index.

    Notes:
        1. ) The input pandas.DataFrame must be indexed via datetime
        objects which are timezone aware.

        2.)  You can find a complete list of available timezones via:
        ```
        import pytz
        pytz.all_timezones
        ```
        You can also specify 'strax' as timezone which will convert the
        time index into a 'strax time' equivalent.
        The default timezone of strax is UTC.
    """
    if tz == 'strax':
        df = df.tz_convert(tz='UTC')
        df.index = df.index.astype(np.int64)
        df.index.rename(f'time strax', inplace=True)
    else:
        df = df.tz_convert(tz=tz)
        df.index.rename(f'time {tz}', inplace=True)
    return df


@numba.njit
def _average_scada(times, values, nvalues):
    """
    Function which down samples scada values.

    :param times: Unix times of the data points.
    :param values: Corresponding sensor value
    :param nvalues: Number of samples we average over.
    :return: new time values and
    """
    if len(times) % nvalues:
        nsamples = (len(times) // nvalues) - 1
    else:
        nsamples = (len(times) // nvalues)
    res = np.zeros(nsamples, dtype=np.float32)
    new_times = np.zeros(nsamples, dtype=np.int64)
    for ind in range(nsamples):
        res[ind] = np.mean(values[ind * nvalues:(ind + 1) * nvalues])
        new_times[ind] = np.mean(times[ind * nvalues:(ind + 1) * nvalues])

    return new_times, res
