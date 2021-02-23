import numpy as np
import pandas as pd
import socket
import strax
import straxen
import sys
export, __all__ = strax.exporter()
from configparser import NoSectionError


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

def _force_int(df, columns):
    for c in strax.to_str_tuple(columns):
        df[c] = df[c].values.astype(np.int64)
    return df


@export
def print_versions(modules=('strax', 'straxen'), return_string=False):
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
            # pylint: disable=exec-used
            exec(f'import {m}')
            # pylint: disable=eval-used
            message += f'\n{m}\tv{eval(m).__version__}\t{eval(m).__path__[0]}'
        except (ModuleNotFoundError, ImportError):
            print(f'{m} is not installed')
    if return_string:
        return message
    print(message)


@export
def utilix_is_configured(header='RunDB', section='pymongo_database') -> bool:
    """
    Check if we have the right connection to
    :return: bool, can we connect to the Mongo database?
    """
    try:
        return (hasattr(straxen.uconfig, 'get') and
                straxen.uconfig.get(header, section) is not None)
    except NoSectionError:
        return False
