import concurrent.futures
import fnmatch
import re

from tqdm import tqdm
import numpy as np
import pandas as pd
import pymongo

import strax
import straxen

export, __all__ = strax.exporter()


@export
@strax.takes_config(
    strax.Option(
        'prescan_runs', default=False,
         help="Query runs db on initialization to populate self.runs with "
              "a dataframe of basic run info. If False (default),"
              " will do this on the first call to run_selection."),
)
class XENONContext(strax.Context):
    """Strax context with extra methods appropriate to XENON analysis
    (i.e. replacing functionality of hax)
    """
    rundb : pymongo.collection
    runs: pd.DataFrame

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        for st in self.storage:
            if isinstance(st, straxen.RunDB):
                self.rundb = st.collection
                break
        else:
            self.log.debug(
                "XENONContext without rundb frontend initialized: some "
                "functionality will be unavailable")
            self.rundb = None

        if self.context_config['prescan_runs']:
            self.scan_runs()

    def scan_runs(self,
                  extra_fields=tuple()):
        """Update self.runs with runs currently available in the runs db,
        used for run_selection calls.
        :param extra_fields: Additional fields from run doc to include.
        """
        base_fields = ['name', 'number', 'reader.ini.name', 'tags.name',
                       'start', 'end', 'trigger.events_built', 'tags.name']

        if self.rundb is None:
            raise RuntimeError("Cannot scan runs db if no "
                               "rundb frontend is registered in the context.")

        docs = []
        cursor = self.rundb.find(
            filter={},
            projection=(base_fields + list(extra_fields)))
        for doc in tqdm(cursor, desc='Loading run info', total=cursor.count()):
            # TODO: Perhaps we should turn this query into an aggregation
            # to return also availability of key data types
            # (records, peaks, events?)

            # Process and flatten the doc
            # Convert tags to single string
            doc['tags'] = ','.join([t['name']
                                    for t in doc.get('tags', [])])
            doc = straxen.flatten_dict(doc, separator='__')
            del doc['_id']  # Remove the Mongo document ID
            doc = straxen.flatten_dict(doc, separator='__')
            docs.append(doc)

        self.runs = pd.DataFrame(docs)

    def run_selection(self, run_mode=None,
                      include_tags=None, exclude_tags=None,
                      pattern_type='fnmatch', ignore_underscore=True):
        """Return pandas.DataFrame with basic info from runs
        that match selection criteria.
        :param include: String or list of strings of patterns
            for required tags
        :param exclude: String / list of strings of patterns
            for forbidden tags.
            Exclusion criteria  have higher priority than inclusion criteria.
        :param run_mode: Pattern to match run modes (reader.ini.name)
        :param pattern_type: Type of pattern matching to use.
            Defaults to 'fnmatch', which means you can use
            unix shell-style wildcards (?, *).
            The alternative is 're', which means you can use
            full python regular expressions.
        :param ignore_underscore: Ignore the underscore at the start of tags
            (indicating some degree of officialness or automation).

        Examples:
         - `run_selection(include_tags='blinded')`
            select all datasets with a blinded or _blinded tag.
         - `run_selection(include_tags='*blinded')`
            ... with blinded or _blinded, unblinded, blablinded, etc.
         - `run_selection(include_tags=['blinded', 'unblinded'])`
           ... with blinded OR unblinded, but not blablinded.
         - `run_selection(include_tags='blinded',
                          exclude_tags=['bad', 'messy'])`
           select blinded dsatasets that aren't bad or messy
        """
        if self.runs is None:
            self.scan_runs()
        dsets = self.runs.copy()

        if pattern_type not in ('re', 'fnmatch'):
            raise ValueError("Pattern type must be 're' or 'fnmatch'")

        if run_mode is not None:
            modes = dsets['reader__ini__name'].values
            mask = np.zeros(len(modes), dtype=np.bool_)
            if pattern_type == 'fnmatch':
                for i, x in enumerate(modes):
                    mask[i] = fnmatch.fnmatch(x, run_mode)
            elif pattern_type == 're':
                for i, x in enumerate(modes):
                    mask[i] = bool(re.match(run_mode, x))
            dsets = dsets[mask]

        if include_tags is not None:
            dsets = dsets[_tags_match(dsets,
                                      include_tags,
                                      pattern_type,
                                      ignore_underscore)]

        if exclude_tags is not None:
            dsets = dsets[True ^ _tags_match(dsets,
                                             exclude_tags,
                                             pattern_type,
                                             ignore_underscore)]

        return dsets

    def get_array(self, run_id,
                  *args,
                  max_workers=None,
                  **kwargs) -> np.ndarray:
        # TODO: blinding
        # TODO: cut history tracking
        if isinstance(run_id, (str, int)):
            # Single run: strax can handle this by itself
            return super().get_array(run_id,
                                     *args,
                                     max_workers=max_workers,
                                     **kwargs)

        if max_workers is None:
            # ProcessPoolExecutor defaults to using lots of cores
            # unless we say:
            max_workers = 1

        # Multiple run case
        # Probably we'll want to use dask for this in the future,
        # to enable cut history tracking.
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers) as exc:
            futures = [exc.submit(super().get_array, r, *args, **kwargs)
                       for r in run_id]
            results = [f.result()
                       for f in tqdm(
                            concurrent.futures.as_completed(futures),
                            desc="Loading %d runs" % len(run_id),
                            total=len(run_id))]
            return np.concatenate(results)


XENONContext.get_array.__doc__ = strax.Context.__doc__


def _tags_match(dsets, patterns, pattern_type, ignore_underscore):
    result = np.zeros(len(dsets), dtype=np.bool)

    if isinstance(patterns, str):
        patterns = [patterns]

    for i, tags in enumerate(dsets.tags):
        result[i] = any([any([_tag_match(tag, pattern,
                                         pattern_type,
                                         ignore_underscore)
                              for tag in tags.split(',')
                              for pattern in patterns])])

    return result


def _tag_match(tag, pattern, pattern_type, ignore_underscore):
    if ignore_underscore and tag.startswith('_'):
        tag = tag[1:]
    if pattern_type == 'fnmatch':
        return fnmatch.fnmatch(tag, pattern)
    elif pattern_type == 're':
        return bool(re.match(pattern, tag))
    raise NotImplementedError


@export
def count_tags(ds):
    """Return how often each tag occurs in the datasets DataFrame ds"""
    from collections import Counter
    from itertools import chain
    all_tags = chain(*[ts.split(',')
                       for ts in ds['tags'].values])
    return Counter(all_tags)
