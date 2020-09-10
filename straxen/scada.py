import urllib
import requests
import pandas as pd
import numba
import numpy as np

import strax
from .SCADA_SECRETS import SCData_URL, SCLastValue_URL, SCADA_SECRETS

export, __all__ = strax.exporter()

def convert_labtime_to_unix():
    #TODO: Add convinient CET/Lab time to unix time converter
    raise NotImplementedError

def find_scada_parameter():
    # TODO: Add function which returns SCADA sensor names by short Name
    raise NotImplementedError

@export
def get_scada_values(parameters,
                     start=None,
                     end=None,
                     context=None,
                     run_id=None,
                     target=None,
                     seconds_range=None,
                     value_every_seconds=1):
    """
    Function which returns XENONnT slow control values for a given set
    of parameters and time range.

    The time range can be either defined by a start and end time or via
    the run_id, target and context.

    :param parameters: dictionary containing the names of the requested
        scada-parameters. The keys are used as identifier of the
        parameters in the returned pandas.DataFrame.
    :param start: int representing the start time of the interval in ns
        unix time.
    :param end: same as start but as end.
    :param context: Context you are working with (e.g. st).
    :param run_id: Id of the run.
    :param target: Target for which the run start and end should be used.
    :param seconds_range: Range of seconds since the run start.
    :param value_every_seconds: Defines with which time difference
        values should be returned. Must be an integer!
        Default: one value per 1 seconds.
    :return: pandas.DataFrame containing the data of the specified
        parameters.
    """
    if not isinstance(parameters, dict):
        mes = 'The argument "parameters" has to be specified as a dict.'
        raise ValueError(mes)

    if np.all((run_id, context, target)):
        # User specified a valid context and run_id, so get the start
        # and end time for our query:
        meta = context.get_meta(run_id, target)
        # TODO: Target is only used to identify a stored data_kind metadata. Could be removed since start and end
        #  should be the same for all data_kinds.
        start = meta['start']  # TODO: This will fail for raw_records. Does not have start or end field; why?

        if seconds_range:
            start += seconds_range[0] * 10**9
            end = start + seconds_range[1] * 10**9
        else:
            end = meta['end']

    if np.all((start, end)):
        # User specified a vaild start and end time, so there is not
        # anything which needs to be done.
        pass

    else:
        mes = ('You have to specify either a run_id, context and target.'
               ' E.g. call get_scada_values(parameters, run_id=run,'
               ' target=raw_records", context=st) or you have to specifiy'
               'a valid start and end time.')
        raise ValueError(mes)

    # Now loop over specified parameters and get the values for those.
    for ind, (k, p) in enumerate(parameters.items()):
        print(f'Start to query {k}: {p}') #TODO: Remove me?
        temp_df = _query_single_parameter(start, end, k, p, value_every_seconds)

        if ind:
            m = np.all(df.loc[:, 'time'] == temp_df.loc[:, 'time'])
            mes = ('This is odd somehow the time stamps for the query of'
                   f' {p} does not match the other time stamps.')
            assert m, mes
            df = pd.concat((df, temp_df[k]), axis=1)
        else:
            df = temp_df
    return df


def _query_single_parameter(start,
                           end,
                           parameter_key,
                           parameter_name,
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
    query = SCADA_SECRETS
    query["StartDateUnix"] = start // 10**9
    query["EndDateUnix"] = end // 10**9
    query['name'] = parameter_name
    query = urllib.parse.urlencode(query)
    values = requests.get(SCData_URL + query)

    # Step 1.: Get the Scada values:
    # TODO: If no value can be found this will throw an error...
    # TODO: If parameter name does not exist will throw a different type of error.
    # Solution query first last value since which returns last value
    # + time stamp. If timestamp before start then do not query SCData:
    temp_df = pd.read_json(values.text)

    # Step 2.: Fill dataframe with 1 seconds spacing:
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
    #  the data in chunks.
    seconds = np.arange(start, end, 10**9)
    df = pd.DataFrame()
    df.loc[:, 'time'] = seconds
    df['time'] = df['time'].astype('<M8[ns]')
    df.set_index('time', inplace=True)

    df.loc[temp_df['timestampseconds'], parameter_key] = temp_df.loc[:, 'value'].values

    # Fill the very first values if needed:
    # TODO: Replace me with something meaningful, use GetSCLast for it....
    # TODO: Move this function before GetSCData, will help to avoid no data in time range error.
    df.iloc[0, 0] = 0

    # Now fill values in between like Scada would do:
    df.ffill(inplace=True)
    df.reset_index(inplace=True)

    # Step 4. Downsample data if asked for:
    if value_every_seconds > 1:
        nt, nv = _downsample_scada(df['time'].astype(np.int64).values,
                                   df[parameter_key].values,
                                   value_every_seconds)
        df = pd.DataFrame()
        df['time'] = nt.astype('<M8[ns]')
        df[parameter_key] = nv

    return df


@numba.njit
def _downsample_scada(times, values, nvalues):
    '''
    Function which downsamples scada values.

    Downsampling means simply taking the mean of the corresponding time
    range.
    '''
    if len(times) % nvalues:
        nsamples = (len(times) // nvalues) - 1
    else:
        nsamples = (len(times) // nvalues)
    res = np.zeros(nsamples, dtype=np.float32)
    new_times = np.zeros(nsamples, dtype=np.int64)  # TODO: Think about the time binning...
    for ind in range(nsamples):
        res[ind] = np.mean(values[ind * nvalues:(ind + 1) * nvalues])
        new_times[ind] = times[ind * nvalues]

    return new_times, res