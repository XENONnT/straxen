import urllib
import requests
import pandas as pd
import numba
import numpy as np
import warnings
import strax
import straxen

import sys
if any('jupyter' in arg for arg in sys.argv):
    # In some cases we are not using any notebooks,
    # Taken from 44952863 on stack overflow thanks!
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from straxen import uconfig
export, __all__ = strax.exporter()


@export
class SCADAInterface:

    def __init__(self, context=None, use_progress_bar=True):
        """
        Interface to excess the XENONnT slow control data via python.

        :param context: Context you are using e.g. st. This is needed
            if you would like to query data via run_ids.
        :param use_progress_bar: Use a progress bar in the Scada interface
        """
        try:
            self.SCData_URL = uconfig.get('scada', 'scdata_url')
            self.SCLastValue_URL = uconfig.get('scada', 'sclastvalue_url')
            self.SCADA_SECRETS = dict(QueryType=uconfig.get('scada', 'querytype'),
                                      username=uconfig.get('scada', 'username'),
                                      api_key=uconfig.get('scada', 'api_key')
                                      )

            # Load parameters from the database.
            self.pmt_file = straxen.get_resource('PMTmap_SCADA.json',
                                                 fmt='json')
        except ValueError as e:
            raise ValueError(f'Cannot load SCADA information, from your xenon'
                             ' config. SCADAInterface cannot be used.') from e

        # Use a tqdm progress bar if requested. If a user does not want
        # a progress bar, just wrap it by a tuple
        self._use_progress_bar = use_progress_bar
        self.context = context

    def get_scada_values(self,
                         parameters,
                         start=None,
                         end=None,
                         run_id=None,
                         time_selection_kwargs=None,
                         fill_gaps=None,
                         filling_kwargs=None,
                         down_sampling=False,
                         every_nth_value=1):
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
        :param query_type_lab: Mode on how to query data from the historians.
            Can be either False (default) to get raw data or True to get
            data which was interpolated by historian. Useful if large
            time ranges have to be queried.
        :param time_selection_kwargs: Keyword arguments taken by
            st.to_absolute_time_range(). Default: {"full_range": True}
        :param fill_gaps: Decides how to fill gaps in which no data was
            recorded. Only needed for query_type_lab=False. Can be either
            None, "interpolation" or "forwardfill".None keeps the gaps
            (default), "interpolation" uses pandas.interpolate and
            "forwardfill" pandas.ffill. See
            https://pandas.pydata.org/docs/ for more information. You
            can change the filling options of the methods with the
            filling_kwargs.
        :param filling_kwargs: Kwargs applied to pandas .ffill() or
            .interpolate(). Only needed for query_type_lab=False.
        :param down_sampling: Boolean which indicates whether to
            donw_sample result or to apply average. The averaging
            is deactivated in case of interpolated data. Only needed
            for query_type_lab=False.
        :param every_nth_value: Defines over how many values we compute
            the average or the nth sample in case we down sample the
            data. In case query_type_lab=True every nth second is
            returned.
        :return: pandas.DataFrame containing the data of the specified
            parameters.
        """
        query_type_lab = False

        if not filling_kwargs:
            filling_kwargs = {}

        if not time_selection_kwargs:
            time_selection_kwargs = {'full_range': True}

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
        elif run_id:
            mes = ('You are trying to query slow control data via run_ids' 
                   ' but you have not specified the context you are '
                   'working with. Please set the context either via '
                   '.st = YOURCONTEXT, or when initializing the '
                   'interface.')
            raise ValueError(mes)

        if not np.all((start, end)):
            # User has not specified any valid start and end time
            mes = ('You have to specify either a run_id and context.'
                   ' E.g. call get_scada_values(parameters, run_id=run)'
                   ' or you have to specify a valid start and end time '
                   'in utc unix time ns.')
            raise ValueError(mes)

        _fill_gaps = [None, 'None', 'interpolation', 'forwardfill']

        if fill_gaps not in _fill_gaps:
            raise ValueError(f'Wrong argument for "fill_gaps", must be either {_fill_gaps}.' 
               f' You specified "{fill_gaps}"')

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
        iterator = enumerate(parameters.items())
        if self._use_progress_bar:
            # wrap using progress bar
            iterator = tqdm(iterator, total=len(parameters), desc='Load parameters')
        for ind, (k, p) in iterator:
            temp_df = self._query_single_parameter(start, end,
                                                   k, p,
                                                   every_nth_value=every_nth_value,
                                                   fill_gaps=fill_gaps,
                                                   filling_kwargs=filling_kwargs,
                                                   down_sampling=down_sampling,
                                                   query_type_lab=query_type_lab
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
                                fill_gaps,
                                filling_kwargs,
                                down_sampling,
                                query_type_lab=False,
                                every_nth_value=1):
        """
        Function to query the values of a single parameter from SCData.

        :param start: Start time in ns unix time
        :param end: End time in ns unix time
        :param parameter_key: Key to identify queried parameter in the
            DataFrame
        :param parameter_name: Parameter name in Scada/historian database.
        :param fill_gaps: Decides how to fill gaps in which no data was
            recorded. Only needed for query_type_lab=False. Can be either
            None, "interpolation" or "forwardfill".None keeps the gaps
            (default), "interpolation" uses pandas.interpolate and
            "forwardfill" pandas.ffill. See
            https://pandas.pydata.org/docs/ for more information. You
            can change the filling options of the methods with the
            filling_kwargs.
        :param filling_kwargs: Keyword arguments forwarded to pandas.ffill
            or pandas.interpolate.
        :param every_nth_value: Defines over how many values we compute
            the average or the nthed sample in case we down sample the
            data.

        :returns: DataFrame with a time and parameter_key column.
        """
        if every_nth_value < 1:
            mes = ("SCADA takes only values every second. Cannot ask for a"
                   " higher sampling rate than one value per second. However"
                   f" you asked for one value every {every_nth_value} seconds.")
            raise ValueError(mes)
        if not isinstance(every_nth_value, int):
            raise ValueError('"value_every_seconds" must be an int!')

        # First we have to create an array where we can fill values:
        if query_type_lab:
            # In the lab case we get interpolated data without nans so the df can be set
            # accordingly.
            seconds = np.arange(start, end + 1, 10**9 * every_nth_value)
        else:
            seconds = np.arange(start, end + 1, 10**9)  # +1 to make sure endtime is included

        df = pd.DataFrame()
        df.loc[:, 'time'] = seconds
        df['time'] = df['time'].astype('<M8[ns]')
        df.set_index('time', inplace=True)

        # Init parameter query:
        query = self.SCADA_SECRETS.copy()  # Have to copy dict since it is not immutable
        query['name'] = parameter_name

        # Check if first value is in requested range:
        temp_df = self._query(query,
                              self.SCLastValue_URL,
                              end=(start // 10**9) + 1)  # +1 since it is end before exclusive

        # Store value as first value in our df
        df.loc[df.index.values[0], parameter_key] = temp_df['value'][0]

        # Query values between start+1 and endtime:
        offset = 0
        ntries = 0
        max_tries = 40000  # This corresponds to ~23 years
        while ntries < max_tries:
            temp_df = self._query(query,
                                  self.SCData_URL,
                                  start=(start//10**9)+1+offset,
                                  end=(end//10**9)+1,
                                  query_type_lab=query_type_lab,
                                  seconds_interval=every_nth_value,
                                  raise_error_message=False  # No valid value in query range... 
                                  )  # +1 since it is end before exclusive
            if temp_df.empty:
                # In case WebInterface does not return any data, e.g. if query range too small
                break
            times = (temp_df['timestampseconds'].values*10**9).astype('<M8[ns]')
            df.loc[times, parameter_key] = temp_df.loc[:, 'value'].values

            endtime = temp_df['timestampseconds'].values[-1].astype(np.int64)
            offset += len(temp_df)
            ntries += 1
            if not (len(temp_df) == 35000 and endtime != end // 10**9):
                # Max query are 35000 values, if end is reached the
                # length of the dataframe is either smaller or the last
                # time value is equivalent to queried range.
                break

        # Let user decided whether to ffill, interpolate or keep gaps:
        if fill_gaps == 'interpolation':
            df.interpolate(**filling_kwargs, inplace=True)

        if fill_gaps == 'forwardfill':
            # Now fill values in between like Scada would do:
            df.ffill(**filling_kwargs, inplace=True)

        # Step 4. Down-sample data if asked for:
        df.reset_index(inplace=True)
        if every_nth_value > 1:
            if down_sampling:
                df = df[::every_nth_value]
            else:
                nt, nv = _average_scada(df['time'].astype(np.int64).values,
                                        df[parameter_key].values,
                                        every_nth_value)
                df = pd.DataFrame()
                df['time'] = nt.astype('<M8[ns]')
                df[parameter_key] = nv

        return df

    @staticmethod
    def _query(query, 
               api, 
               start=None, 
               end=None, 
               query_type_lab=False, 
               seconds_interval=None,
              raise_error_message=True):
        """
        Helper to reduce code. Asks for data and returns result. Raises error
        if api returns error.
        """
        if start:
            query["StartDateUnix"] = start
        if end:
            query['EndDateUnix'] = end

        if query_type_lab:
            query['QueryType'] = 'lab'
            query['interval'] = seconds_interval
        else:
            query['QueryType'] = 'rawbytime'
            query.pop('interval', None)  # Interval only works with lab

        query_url = urllib.parse.urlencode(query)
        values = requests.get(api + query_url)
        values = values.json()
            
        temp_df = pd.DataFrame(columns=('timestampseconds', 'value'))
        if isinstance(values, dict) and raise_error_message:
            query_status = values['status']
            query_message = values['message']
            raise ValueError(f'SCADAapi has not returned values for the '
                             f'parameter "{query["name"]}". It returned the '
                             f'status "{query_status}" with the message "{query_message}".')
        if isinstance(values, list):
            temp_df = pd.DataFrame(values)
        return temp_df

    def find_scada_parameter(self):
        raise NotImplementedError('Feature not implemented yet.')

    def find_pmt_names(self, pmts=None, hv=True, current=False):
        """
        Function which returns a list of PMT parameter names to be
        called in SCADAInterface.get_scada_values. The names refer to
        the high voltage of the PMTs, not their current.

        Thanks to Hagar and Giovanni who provided the file.

        :param pmts: Optional parameter to specify which PMT parameters
            should be returned. Can be either a list or array of channels
            or just a single one.
        :param hv: Bool if true names of high voltage channels are
            returned.
        :param current: Bool if true names for the current channels are
            returned.
        :return: dictionary containing short names as keys and scada
            parameter names as values.
        """
        if not (hv or current):
            raise ValueError('Either one "hv" or "current" must be true.')

        if isinstance(pmts, np.ndarray):
            # convert to a simple list since otherwise we get ambiguous errors
            pmts = list(pmts)
            
        if not hasattr(pmts, '__iter__'):
            # If single PMT convert it to itterable
            pmts = [pmts]

        # Getting parameter names for all PMTs:
        # The file contains the names for the HV channels
        if pmts:
            pmts_v = {k: v for k, v in self.pmt_file.items() if int(k[3:]) in pmts}
        else:
            pmts_v = self.pmt_file

        res = {}
        # Now get all relevant names:
        for key, value in pmts_v.items():
            if hv:
                res[key+'_HV'] = value
            if current:
                res[key+'_I'] = value[:-4] + 'IMON'

        return res


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
