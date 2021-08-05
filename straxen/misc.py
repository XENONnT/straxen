import numpy as np
import pandas as pd
import socket
import strax
import straxen
import sys
import warnings
import datetime
import pytz
from os import environ as os_environ
from importlib import import_module
from configparser import NoSectionError

export, __all__ = strax.exporter()


@export
def dataframe_to_wiki(df, float_digits=5, title='Awesome table', 
                      force_int=tuple()):
    """Convert a pandas dataframe to a dokuwiki table 
    (which you can copy-paste onto the XENON wiki)
    :param df: dataframe to convert
    :param float_digits: Round float-ing point values to this number of digits.
    :param title: title of the table.
    """
    table = '^ %s ' % title + '^' * (len(df.columns) - 1) + '^\n'
    table += '^ ' + ' ^ '.join(df.columns) + ' ^\n'

    def do_round(x):
        if isinstance(x, float):
            return round(x, float_digits)
        return x
    force_int = np.where(np.in1d(df.columns.values,
                                 strax.to_str_tuple(force_int)))[0]

    for _, row in df.iterrows():
        table += "| " + ' | '.join([
            str(int(x) if i in force_int else do_round(x))
            for i, x in enumerate(row.values.tolist())]) + ' |\n'
    return table


@export
def print_versions(modules=('strax', 'straxen', 'cutax'), return_string=False):
    """
    Print versions of modules installed.

    :param modules: Modules to print, should be str, tuple or list. E.g.
        print_versions(modules=('strax', 'straxen', 'wfsim',
        'cutax', 'pema'))
    :param return_string: optional. Instead of printing the message,
        return a string
    :return: optional, the message that would have been printed
    """
    message = (f'Working on {socket.getfqdn()} with the following '
               f'versions and installation paths:')
    py_version = sys.version.replace(' (', '\t(').replace('\n', '')
    message += f"\npython\tv{py_version}"
    for m in strax.to_str_tuple(modules):
        try:
            mod = import_module(m)
            message += f'\n{m}'
            if hasattr(mod, '__version__'):
                message += f'\tv{mod.__version__}'
            if hasattr(mod, '__path__'):
                message += f'\t{mod.__path__[0]}'
        except (ModuleNotFoundError, ImportError):
            print(f'{m} is not installed')
    if return_string:
        return message
    print(message)


@export
def utilix_is_configured(header='RunDB', section='xent_database') -> bool:
    """
    Check if we have the right connection to
    :return: bool, can we connect to the Mongo database?
    """
    try:
        return (hasattr(straxen.uconfig, 'get') and
                straxen.uconfig.get(header, section) is not None)
    except NoSectionError:
        return False


@export
class TimeWidgets:

    def __init__(self):
        """Creates interactive time widgets which allow to convert a
        human readble date and time into a start and endtime in unix time
        nano-seconds.
        """
        self._created_widgets = False

    def create_widgets(self):
        """
        Creates time and time zone widget for simpler time querying.

        Note:
            Please be aware that the correct format for the time field
            is HH:MM.
        """
        utcend = datetime.datetime.utcnow()
        deltat = datetime.timedelta(minutes=60)
        utcstart = utcend - deltat
        import ipywidgets as widgets

        self._start_widget = self._create_date_and_time_widget(utcstart, 'Start')
        self._end_widget = self._create_date_and_time_widget(utcend, 'End')
        self._time_zone_widget = self._create_time_zone_widget()
        self._created_widgets = True

        return widgets.VBox([widgets.HBox([self._time_zone_widget, widgets.HTML(value="ns:")]),
                             self._start_widget,
                             self._end_widget])

    def get_start_end(self):
        """
        Returns start and end time of the specfied time interval in
        nano-seconds utc unix time.
        """
        if not self._created_widgets:
            raise ValueError('Please run first "create_widgets". ')

        time_zone = self._time_zone_widget.options[self._time_zone_widget.value][0]

        start, start_ns = self._convert_to_datetime(self._start_widget, time_zone)
        start = start.astimezone(tz=pytz.UTC)
        start = start.timestamp()
        start = int(start * 10**9) + start_ns
        end, end_ns = self._convert_to_datetime(self._end_widget, time_zone)
        end = end.astimezone(tz=pytz.UTC)
        end = end.timestamp()
        end = int(end * 10**9) + end_ns

        if start > end:
            warnings.warn('Start time is larger than endtime are you '
                          'sure you wanted this?')
        return start, end

    @staticmethod
    def _create_date_and_time_widget(date_and_time, widget_describtion):
        import ipywidgets as widgets
        from ipywidgets import Layout
        date = datetime.date(date_and_time.year,
                             date_and_time.month,
                             date_and_time.day)
        date = widgets.DatePicker(description=widget_describtion,
                                  value=date,
                                  layout=Layout(width='225px'),
                                  disabled=False)

        time = "{:02d}:{:02d}".format(int(date_and_time.hour),
                                      int(date_and_time.minute))
        time = widgets.Text(value=time,
                            layout=Layout(width='75px'),
                            disabled=False)

        time_ns = widgets.Text(value='0',
                               layout=Layout(width='150px'),
                               disabled=False)

        return widgets.HBox([date, time, time_ns])

    @staticmethod
    def _create_time_zone_widget():
        import ipywidgets as widgets
        _time_zone_widget = widgets.Dropdown(options=[('CET', 0), ('UTC', 1)],
                                             value=0,
                                             description='Time Zone:',
                                             )
        return _time_zone_widget

    @staticmethod
    def _convert_to_datetime(time_widget, time_zone):
        """
        Converts values of widget into a timezone aware datetime object.

        :param time_widget: Widget Box containing a DatePicker and
            two text widget. The first text widget is used to set a day
            time. The second the time in nano-seconds.
        :param time_zone: pytz.timezone allowed string for a timezone.

        :returns: timezone aware datetime object.
        """
        date_and_time = [c.value for c in time_widget.children]
        try:
            hour_and_minutes = datetime.datetime.strptime(date_and_time[1], '%H:%M')
        except ValueError as e:
            raise ValueError('Cannot convert time into datetime object. '
                             f'Expected the following formating HH:MM. {e}')

        time = datetime.datetime.combine(date_and_time[0], datetime.time())
        time = time.replace(hour=hour_and_minutes.hour,
                            minute=hour_and_minutes.minute)
        time_zone = pytz.timezone(time_zone)
        time = time_zone.localize(time)

        time_ns = int(date_and_time[2])

        return time, time_ns


@export
def convert_array_to_df(array: np.ndarray) -> pd.DataFrame:
    """
    Converts the specified array into a DataFrame drops all higher
    dimensional fields during the process.

    :param array: numpy.array to be converted.
    :returns: DataFrame with higher dimensions dropped.
    """
    keys = [key for key in array.dtype.names if array[key].ndim == 1]
    return pd.DataFrame(array[keys])


@export
def _is_on_pytest():
    """Check if we are on a pytest"""
    return 'PYTEST_CURRENT_TEST' in os_environ
