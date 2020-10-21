"""Create datastructure documentation page

This will add a page with various svg graphs and html tables
describing the datastructure: dependencies, columns provided,
and configuration options that apply to each plugins.

For extra credit, the SVGs are clickable.
"""
from collections import defaultdict
import os
import shutil
import pandas as pd
import graphviz
import strax
import straxen
import numpy as np


this_dir = os.path.dirname(os.path.realpath(__file__))

page_header = """
{title}
========================

This page is an autogenerated reference for all the plugins in straxen's
`xenonnt_online` context. 

Colors indicate data kinds. To load tables with different data kinds,
you currently need more than one `get_df` (or `get_array`) commands.

"""


template = """
{data_type}
--------------------------------------------------------

Description
~~~~~~~~~~~~~~~~~~~~~~

Provided by plugin: {p.__class__.__name__}

Data kind: {kind}

{docstring}


Columns provided
~~~~~~~~~~~~~~~~~~~~~~
.. raw:: html

{columns}


Dependencies
~~~~~~~~~~~~~~~~~~~~~~
.. raw:: html

{svg}


Configuration options
~~~~~~~~~~~~~~~~~~~~~~~

These are all options that affect this data type. 
This also includes options taken by dependencies of this datatype,
because changing any of those options affect this data indirectly.

.. raw:: html

{config_options}


"""

titles = {'': 'Straxen datastructure',
          '_he': "Straxen datastructure for high energy channels",
          '_nv': "Straxen datastructure for neutron veto",
          '_mv': "Straxen datastructure for muon veto",
}
tree_suffices = list(titles.keys())

kind_colors = dict(
    events='#ffffff',
    peaks='#98fb98',
    hitlets='#0066ff',
    peaklets='#d9ff66',
    merged_s2s='#ccffcc',
    records='#ffa500',
    raw_records='#ff4500',
    raw_records_coin='#ff4500')

suffices = ['_he', '_nv', "_mv"]
for suffix in suffices:
    to_copy = list(kind_colors.keys())
    for c in to_copy:
        kind_colors[c + suffix] = kind_colors[c]


def add_spaces(x):
    """Add four spaces to every line in x

    This is needed to make html raw blocks in rst format correctly
    """
    y = ''
    if isinstance(x, str):
        x = x.split('\n')
    for q in x:
        y += '    ' + q
    return y


def skip(p, d, suffix, data_type):
    """
    Can we skip this plugin in our for loop (a bunch of if statements to check if we can
        continue).
    :param p: strax.plugin
    :param d: dependency of the strax.plugin
    :param suffix: any of the tree_suffices needed for the logic
    :param data_type: data type (sting)
    :return: bool if we can continue (do not include this one in the data-structure if
        True)
    """
    if suffix not in p.data_kind_for(d):
        # E.g. don't bother with raw_records_nv stuff for mv
        return True
    elif suffix == '' and np.any([s in p.data_kind_for(d) for s in tree_suffices if s != '']):
        # E.g. don't bother with raw_records_nv stuff for tpc ('' is always in a string)
        return True
    elif 'raw_records' in p.data_kind_for(d):
        if 'raw_records' in data_type:
            # fine
            pass
        elif f'raw_records{suffix}' != p.data_kind_for(d) and 'aqmon' in p.data_kind_for(d):
            # skip aqmon raw_records in dependency graph
            return True
    return False


def get_plugins_deps(st):
    """
    For a given Strax.Context return the dependencies per plugin split by the known
        tree_suffices.
    :param st: Strax.Context
    :return: dict of default dicts containing the number of dependencies.
    """
    plugins_by_deps = {k: defaultdict(list) for k in tree_suffices}
    for suffix in tree_suffices:
        for pn, p in st._plugin_class_registry.items():
            if suffix not in pn:
                continue
            elif suffix == '' and np.any([s in pn for s in tree_suffices if s != '']):
                continue
            plugins = st._get_plugins((pn,), run_id='0')
            plugins_by_deps[suffix][len(plugins)].append(pn)
    return plugins_by_deps


def get_context():
    """
    Need to init a context without initializing the runs_db as that requires the
        appropriate passwords.
    :return: straxen context that mimics the xenonnt_online context without the rundb init
    """
    st = straxen.contexts.xenonnt_online(_database_init=False)
    st.context_config['forbid_creation_of'] = straxen.daqreader.DAQReader.provides
    return st


def build_datastructure_doc():

    pd.set_option('display.max_colwidth', int(1e9))

    st = get_context()

    # Too lazy to write proper graph sorter
    # Make dictionary {total number of dependencies below -> list of plugins}

    plugins_by_deps = get_plugins_deps(st)

    # Make graph for each suffix ('' referring to TPC)
    for suffix in tree_suffices:
        out = page_header.format(title=titles[suffix])
        print(f'------------ {suffix} ------------')
        os.makedirs(this_dir + f'/graphs{suffix}', exist_ok=True)
        for n_deps in list(reversed(sorted(list(plugins_by_deps[suffix].keys())))):
            for data_type in plugins_by_deps[suffix][n_deps]:
                plugins = st._get_plugins((data_type,), run_id='0')

                # Create dependency graph
                g = graphviz.Digraph(format='svg')
                # g.attr('graph', autosize='false', size="25.7,8.3!")
                for d, p in plugins.items():
                    if skip(p, d, suffix, data_type):
                        continue
                    g.node(d,
                           style='filled',
                           href='#' + d.replace('_', '-'),
                           fillcolor=kind_colors.get(p.data_kind_for(d), 'grey'))
                    for dep in p.depends_on:
                        g.edge(d, dep)

                fn = this_dir + f'/graphs{suffix}/' + data_type
                g.render(fn)
                with open(fn + '.svg', mode='r') as f:
                    svg = add_spaces(f.readlines()[5:])

                config_df = st.show_config(d).sort_values(by='option')

                # Shorten long default values
                config_df['default'] = [
                    x[:10] + '...' + x[-10:]
                    if isinstance(x, str) and len(x) > 30 else x
                    for x in config_df['default'].values]

                p = plugins[data_type]

                out += template.format(p=p, svg=svg, data_type=data_type,
                    columns=add_spaces(st.data_info(data_type).to_html(index=False)),
                    kind=p.data_kind_for(data_type),
                    docstring=p.__doc__ if p.__doc__ else '(no plugin description)',
                    config_options=add_spaces(config_df.to_html(index=False)))

        with open(this_dir + f'/reference/datastructure{suffix}.rst', mode='w') as f:
            f.write(out)

        shutil.rmtree(this_dir + f'/graphs{suffix}')


try:
    if __name__ == '__main__':
        build_datastructure_doc()
except KeyError:
    # Whatever
    pass
