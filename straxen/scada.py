import urllib
import requests
import pandas as pd
import numba
import numpy as np
import warnings
import json
import ast
import strax

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
        except ValueError as e:
            raise ValueError(f'Cannot load SCADA information, from your xenon'
                             ' config. SCADAInterface cannot be used.') from e
            
        try:
            # Better to cache the file since is not large:
            with open(uconfig.get('scada', 'pmt_parameter_names')) as f:
                self.pmt_file = json.load(f)
        except (FileNotFoundError, ValueError):
            warnings.warn(('Cannot load PMT parameter names from parameter file.' 
                          ' "find_pmt_names" is disabled for this session.'))
            self.pmt_file = None
        try: 
            with open(uconfig.get('scada', 'parameter_readout_rate')) as f:
                self.read_out_rates = json.load(f)
        except (FileNotFoundError, ValueError) as e:
            raise FileNotFoundError(
                'Cannot load file containing parameter sampling rates.') from e

        self.context = context

    def get_scada_values(self,
                         parameters,
                         start=None,
                         end=None,
                         run_id=None,
                         time_selection_kwargs=None,
                         interpolation=False,
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
        :param time_selection_kwargs: Keyword arguments taken by
            st.to_absolute_time_range(). Default: {"full_range": True}
        :param interpolation: Boolean which decided to either forward
            fill empty values or to interpolate between existing ones.
        :param filling_kwargs: Kwargs applied to pandas .ffill() or
            .interpolate().
        :param down_sampling: Boolean which indicates whether to
            donw_sample result or to apply average. The averaging
            is deactivated in case of interpolated data.
        :param every_nth_value: Defines over how many values we compute
            the average or the nth sample in case we down sample the
            data.
        :return: pandas.DataFrame containing the data of the specified
            parameters.
        """
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

        now = np.datetime64('now')
        if (end // 10**9) > now.astype(np.int64):
            mes = ('You are asking for an endtime which is in the future,'
                   ' I may be written by a physicist, but I am neither self-'
                   'aware nor can I predict the future like they can. You '
                   f'asked for the endtime: {end // 10**9} but current utc '
                   f'time is {now.astype(np.int64)}. I will return for the values for the '
                   'corresponding times as nans instead.')
            warnings.warn(mes)

        self._test_sampling_rate(parameters)

        # Now loop over specified parameters and get the values for those.
        for ind, (k, p) in tqdm(enumerate(parameters.items()), total=len(parameters)):
            temp_df = self._query_single_parameter(start, end,
                                                   k, p,
                                                   every_nth_value=every_nth_value,
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

    def _test_sampling_rate(self, parameters):
        """
        Function which test if the specified parameters share all the
        same sampling rates. If not they cannot be put into a single
        DataFrame and an error is raised.

        :param parameters: input parameter names.
        """
        # Check if queried parameters share the same readout rate if not raise error:
        for rate, parameter_names in self.read_out_rates.items():
            if not hasattr(parameter_names, '__iter__'):
                parameter_names = [parameter_names]
            # Loop over different readout rates. If they belong to the same readout rate...
            input_parameter_names = np.array([v for v in parameters.values()])
            m = np.isin(input_parameter_names, parameter_names)

            if not (np.all(m) or np.all(~m)):
                # ...either all parameters are true or false.
                same_rate = input_parameter_names[m]
                not_same_rate = input_parameter_names[~m]
                raise ValueError(('Not all parameters of your inquiry share the same readout rates. '
                                  f'The parameters {same_rate} are read out every {rate} seconds while '
                                  f'{not_same_rate} are not. For the your and the developers sanity please make '
                                  'two separate inquiries.'))

            if np.all(m):
                # Yes all parameters share the same readout rate:
                self.readout_rate = int(rate)
                self.base = 0
            else:
                self.readout_rate = None

    def _query_single_parameter(self,
                                start,
                                end,
                                parameter_key,
                                parameter_name,
                                interpolation,
                                filling_kwargs,
                                down_sampling,
                                every_nth_value=1):
        """
        Function to query the values of a single parameter from SCData.

        :param start: Start time in ns unix time
        :param end: End time in ns unix time
        :param parameter_key: Key to identify queried parameter in the
            DataFrame
        :param parameter_name: Parameter name in Scada/historian database.
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

        # First we have to create an array where we can fill values with
        # the sampling frequency of scada:
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
        except ValueError as e:
            mes = values.text
            query_message = ast.literal_eval(mes)
            raise ValueError(f'SCADA raised a value error when looking for '
                             f'your parameter "{parameter_name}". The error '
                             f'was {query_message}') from e

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
            self._raw_data = temp_df
            df.loc[temp_df['timestampseconds'], parameter_key] = temp_df.loc[:, 'value'].values
        except ValueError:
            pass

        # Let user decided whether to ffill or interpolate:
        if interpolation:
            df.interpolate(**filling_kwargs, inplace=True)
        else:
            # Now fill values in between like Scada would do:
            df.ffill(**filling_kwargs, inplace=True)
        
        # Step 3.5 In case the sampling rate is not 1 s we have to drop values 
        # and but all columns to the same base:
        if self.readout_rate:
            if not self.base:
                self.base = temp_df['timestampseconds']  # Only contains values which changed
                # In case there are earlier values, but nothing was record we have to go back
                # according to the readout rate to find the true base
                self.base = df[temp_df['timestampseconds'][0]::-self.readout_rate].index.values[-1]
            df = df[self.base::self.readout_rate]

        # Step 4. Down-sample data if asked for:
        df.reset_index(inplace=True)
        if every_nth_value > 1:
            if interpolation and not down_sampling:
                warnings.warn('Cannot use interpolation and running average at the same time.'
                              ' Deactivated the running average, switch to down_sampling instead.')
                down_sampling = True

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
        if not self.pmt_file:
            raise ValueError(('Cannot load PMT parameter names from parameter file.' 
                          ' "find_pmt_names" is disabled in this session.'))
        
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
