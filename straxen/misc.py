import numpy as np
import pandas as pd

import strax
import straxen
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

def _force_int(df, columns):
    for c in strax.to_str_tuple(columns):
        df[c] = df[c].values.astype(np.int64)
    return df
