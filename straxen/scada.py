import urllib
import requests

import pandas as pd
import numba
import numpy as np

import strax
import straxen

from datetime import datetime
from datetime import timedelta
import time
import pytz

import getpass
import warnings
from configparser import NoOptionError
import sys

export, __all__ = strax.exporter()


# Fancy tqdm style in notebooks
tqdm = strax.utils.tqdm


@export
class SCADAInterface:

    def __init__(self, context=None, use_progress_bar=True):
        """
        Interface to access the XENONnT slow control data via python.

        :param context: Context you are using e.g. st. This is needed
            if you would like to query data via run_ids.
        :param use_progress_bar: Use a progress bar in the Scada interface
        """
        self.we_are_straxen = False
        self._token_expire_time = None
        self._token = None
        self.pmt_file_found = True
        try:
            self.SCLogin_url = straxen.uconfig.get('scada', 'sclogin_url')
            self.SCData_URL = straxen.uconfig.get('scada', 'scdata_url')
            self.SCLastValue_URL = straxen.uconfig.get('scada', 'sclastvalue_url')

        except ValueError as e:
            raise ValueError(f'Cannot load SCADA information, from your xenon'
                             ' config. SCADAInterface cannot be used.') from e

        try:
            # Load parameters from the database.
            self.pmt_file = straxen.get_resource('PMTmap_SCADA.json', fmt='json')
        except FileNotFoundError:
            warnings.warn('Cannot find PMT map, "find_pmt_names" cannot be used.')
            self.pmt_file_found = False

        # Use a tqdm progress bar if requested. If a user does not want
        # a progress bar, just wrap it by a tuple
        self._use_progress_bar = use_progress_bar
        self.context = context

        self.we_are_straxen = True
        self.get_new_token()

    def get_scada_values(self,
                         parameters,
                         start=None,
                         end=None,
                         run_id=None,
                         query_type_lab=True,
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
            Can be either False to get raw data or True (default) to get
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

        if not filling_kwargs:
            filling_kwargs = {}

        if not isinstance(parameters, dict):
            mes = 'The argument "parameters" has to be specified as a dict.'
            raise ValueError(mes)

        start, end, now = self._get_and_check_start_end(run_id,
                                                        start,
                                                        end,
                                                        time_selection_kwargs
                                                        )

        _fill_gaps = [None, 'None', 'interpolation', 'forwardfill']
        if fill_gaps not in _fill_gaps:
            raise ValueError(f'Wrong argument for "fill_gaps", must be either {_fill_gaps}.'
                             f' You specified "{fill_gaps}"')

        if not self._token:
            # User has not asked for a token yet:
            self._get_token()

        # Check if token will expire soon, if so renew the token before we query
        # the parameters:
        hrs, mins = self._token_expires_in()
        if hrs == 0 and mins < 30:
            print('Your token will expire in less than 30 min please get first a new one:')
            self._get_token()

        # Now loop over specified parameters and get the values for those.
        for ind, (k, p) in tqdm(
                enumerate(parameters.items()),
                total=len(parameters),
                desc='Load parameters',
                disable=not self._use_progress_bar,
                ):
            try:
                temp_df = self._query_single_parameter(start, end,
                                                       k, p,
                                                       every_nth_value=every_nth_value,
                                                       fill_gaps=fill_gaps,
                                                       filling_kwargs=filling_kwargs,
                                                       down_sampling=down_sampling,
                                                       query_type_lab=query_type_lab)
                
                if ind:
                    m = np.all(df.loc[:, 'time'] == temp_df.loc[:, 'time'])
                    if ind and not m:
                        raise ValueError('This is odd somehow the time stamps for the query of'
                                         f' {p} does not match the previous timestamps.')
            except ValueError as e:
                warnings.warn(f'Was not able to load parameters for "{k}". The reason was: "{e}".'
                              f'Continue without {k}.')
                temp_df = pd.DataFrame(columns=(k,))
            
            if ind:
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

    def _get_and_check_start_end(self, run_id, start, end, time_selection_kwargs):
        """
        Helper function which clusters all time related checks and reduces complexity
        of get_scada_values.
        """
        if not time_selection_kwargs:
            time_selection_kwargs = {'full_range': True}

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
                   '.context = YOURCONTEXT, or when initializing the '
                   'interface.')
            raise ValueError(mes)

        if not np.all((start, end)):
            # User has not specified any valid start and end time
            mes = ('You have to specify either a run_id and context.'
                   ' E.g. call get_scada_values(parameters, run_id=run)'
                   ' or you have to specify a valid start and end time '
                   'in utc unix time ns.')
            raise ValueError(mes)

        if end < start:
            raise ValueError('You specified an endtime which is smaller '
                             'than the start time.')

        if (np.log10(start) < 18) or (np.log10(end) < 18):
            raise ValueError('Expected the time to be in ns unix time (number with 19 digits or more).'
                             ' Have you specified the time maybe in seconds or micro-seconds?')

        now = np.datetime64('now')
        if (end // 10**9) > now.astype(np.int64):
            mes = ('You are asking for an endtime which is in the future,'
                   ' I may be written by a physicist, but I am neither self-'
                   'aware nor can I predict the future like they can. You '
                   f'asked for the endtime: {end // 10**9} but current utc '
                   f'time is {now.astype(np.int64)}. I will return for the values for the '
                   'corresponding times as nans instead.')
            warnings.warn(mes)

        # Chop start/end time if precision is higher then seconds level.
        start = (start//10**9)*10**9
        end = (end//10**9)*10**9

        return int(start), int(end), now

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
        query = {'name': parameter_name}

        if not query_type_lab:
            # Check if first value is in requested range:
            # This is only needed in case of raw data since here it can 
            # happen that the user queries a range without any data.
            temp_df = self._query(query,
                                  self.SCLastValue_URL,
                                  end=(start // 10**9) + 1)  # +1 since it is end before exclusive

            # Store value as first value in our df
            df.loc[df.index.values[0], parameter_key] = temp_df['value'][0]
            offset = 1
        else:
            offset = 0

        one_year_in_ns = int(24*3600*360*10**9)
        starts = np.arange(start+offset, end, one_year_in_ns)
        if len(starts):
            ends = starts + one_year_in_ns
            ends = np.clip(ends, a_max=end, a_min=0)
        else:
            ends = np.array([end])
            starts = np.array([start])

        for start_query, end_query in zip(starts, ends):
            self._query_data_per_year(parameter_key,
                                      query,
                                      start_query,
                                      end_query,
                                      query_type_lab,
                                      every_nth_value,
                                      df,
                                      )

        # Let user decided whether to ffill, interpolate or keep gaps:
        if fill_gaps == 'interpolation':
            df.interpolate(**filling_kwargs, inplace=True)

        if fill_gaps == 'forwardfill':
            # Now fill values in between like Scada would do:
            df.ffill(**filling_kwargs, inplace=True)

        # Step 4. Down-sample data if asked for:
        df.reset_index(inplace=True)
        if every_nth_value > 1 and not query_type_lab:
            # If the user asks for down sampling do so, but only for 
            # raw_data, lab query type is already interpolated and down sampled
            # by the historian. 
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

    def _query_data_per_year(self, 
                             parameter_name,
                             query,
                             start,
                             end,
                             query_type_lab,
                             seconds_interval,
                             result_dataframe,
                             ):
        """
        The SCADA API cannot handle query ranges lasting longer than
        one year. So in case the user specifies a longer time range
        we have to chunk the time requests in steps of years.

        Updates the resulting dataframe in place.
        """
        ntries = 0
        # This corresponds to a bit more than one year assuming 1 value per second:
        max_tries = 1000
        while ntries < max_tries:
            # Although we step the query already in years we also have to
            # do the query in a whole loop as we can only query 35000
            # data points at any given time, however it is not possible
            # to know the sampling rate of the queried parameter apriori.
            temp_df = self._query(query,
                                  self.SCData_URL,
                                  start=(start // 10**9),
                                  end=(end // 10**9),
                                  query_type_lab=query_type_lab,
                                  seconds_interval=seconds_interval,
                                  raise_error_message=False,  # No valid value in query range...
                                  )  # +1 since it is end before exclusive
            if temp_df.empty:
                # In case WebInterface does not return any data, e.g. if query range too small
                break

            times = (temp_df['timestampseconds'].values * 10**9).astype('<M8[ns]')
            result_dataframe.loc[times, parameter_name] = temp_df.loc[:, 'value'].values
            endtime = temp_df['timestampseconds'].values[-1].astype(np.int64)*10**9
            start = endtime  # Next query should start at the last time seen.
            ntries += 1
            if not (len(temp_df) == 35000 and endtime != end // 10**9):
                # Max query are 35000 values, if end is reached the
                # length of the dataframe is either smaller or the last
                # time value is equivalent to queried range.
                break

        return result_dataframe

    def _query(self,
               query,
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

        # Configure query url
        query_url = urllib.parse.urlencode(query)
        self._query_url = api + query_url
        # Security check if url is a real url and not something like file://
        if not self._query_url.lower().startswith('https'):
            raise ValueError('The query URL should start with https! '
                             f'Current URL: {self._query_url}')

        response = requests.get(self._query_url,
                                headers={'Authorization': self._token})
        if response.status_code == 401:
            # Invalid token so we have to get a new one,
            # this should actually never happen, but you never know...
            print('Your token is invalid. It may have expired please get a new one:')
            # If the user puts in the wrong credentials the query will fail.
            self._get_token()
            response = requests.get(self._query_url,
                                    headers={'Authorization': self._token})

        if response.status_code != 200:
            # Check if we get any status code different from 200 == ok
            # If yes raise the corresponding status:
            response.raise_for_status()

        # Read database response and check if query was valid:
        values = response.json()

        temp_df = pd.DataFrame(columns=('timestampseconds', 'value'))
        if isinstance(values, dict) and raise_error_message:
            # Not valid, why:
            query_status = values['status']
            query_message = values['message']
            raise ValueError(f'SCADAapi has not returned values for the '
                             f'parameter "{query["name"]}". It returned the '
                             f'status "{query_status}" with the message "{query_message}".')
        if isinstance(values, list):
            # Valid so return dataframe
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
        if not self.pmt_file_found:
            raise ValueError('json file containing the PMT information was not found. '
                             '"find_pmt_names" cannot be used.')

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
                res[key + '_HV'] = value
            if current:
                res[key + '_I'] = value[:-4] + 'IMON'

        return res

    def get_new_token(self):
        """
        Function to renew the token of the current session.
        """
        self._get_token()

    def _get_token(self):
        """
        Function which asks for user credentials to receive a personalized
        security token. The token is required to query any data from the
        slow control historians.
        """
        if not self.we_are_straxen:
            username, password =  self._ask_for_credentials()
        else:
            try:
                username = straxen.uconfig.get('scada', 'straxen_username')
                password = straxen.uconfig.get('scada', 'straxen_password')
            except (AttributeError, NoOptionError):
                # If section does not exist Fall back to user credentials
                username, password = self._ask_for_credentials()

        login_query = {'username': username,
                       'password': password,
                       }
        res = requests.post(self.SCLogin_url,
                            data=login_query)

        res = res.json()
        if 'token' not in res.keys():
            raise ValueError('Cannot get security token from Slow Control web API. '
                             f'API returned the following reason: {res["Message"]}')

        self._token = res['token']
        toke_start_time = datetime.now(tz=pytz.timezone('utc'))
        hours_added = timedelta(hours=3)
        self._token_expire_time = toke_start_time + hours_added
        print('Received token, the token is valid for 3 hrs.\n',
              f'from {toke_start_time.strftime("%d.%m. %H:%M:%S")} UTC\n',
              f'till {self._token_expire_time.strftime("%d.%m. %H:%M:%S")} UTC\n'
              'We will automatically refresh the token for you :). '
              'Have a nice day and a fruitful analysis!'
              )

    @staticmethod
    def _ask_for_credentials():
        print('Please, enter your Xe1TViewer/SCADA credentials:')
        time.sleep(1)
        username = getpass.getpass('Xenon Username: ')
        password = getpass.getpass('Xenon Password: ')
        return username, password

    def token_expires_in(self):
        """
        Function which displays how long until the current token expires.
        """
        if self._token_expire_time:
            print(f'The current token expires at {self._token_expire_time.strftime("%d.%m. %H:%M:%S")} UTC')
            hrs, mins = self._token_expires_in()
            print(f'Which is in {hrs} h and {mins} min.')
        else:
            raise ValueError('You do not have any valid token yet. Please call '
                             '"get_new_token" first".')

    def _token_expires_in(self):
        """
        Computes hrs and minutes until token expires.
        """
        now = datetime.now(tz=pytz.timezone('utc'))
        dt = (self._token_expire_time - now).seconds  # time delta in seconds
        hrs = dt // 3600
        mins = dt % 3600 // 60
        return hrs, mins


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
        n_samples = (len(times) // nvalues) - 1
    else:
        n_samples = (len(times) // nvalues)
    res = np.zeros(n_samples, dtype=np.float32)
    new_times = np.zeros(n_samples, dtype=np.int64)
    for ind in range(n_samples):
        res[ind] = np.mean(values[ind * nvalues:(ind + 1) * nvalues])
        new_times[ind] = np.mean(times[ind * nvalues:(ind + 1) * nvalues])

    return new_times, res
