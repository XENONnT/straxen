import sys
from sys import getsizeof, stderr
import inspect
import warnings
import datetime
import fnmatch
from immutabledict import immutabledict
import pytz
from itertools import chain
from collections import defaultdict, OrderedDict, deque
from importlib import import_module
from platform import python_version
import typing as ty
from copy import deepcopy

import numpy as np
import pandas as pd
import socket
import graphviz
import strax
import straxen
from git import Repo, InvalidGitRepositoryError
from configparser import NoSectionError

try:
    # pylint: disable=redefined-builtin
    from reprlib import repr
except ImportError:
    pass

_is_jupyter = any("jupyter" in arg for arg in sys.argv)


export, __all__ = strax.exporter()
__all__.extend(["kind_colors"])


kind_colors = dict(
    events="#ffffff",
    peaks="#98fb98",
    hitlets="#0066ff",
    peaklets="#d9ff66",
    merged_s2s="#ccffcc",
    lone_hits="#CAFF70",
    records="#ffa500",
    raw_records="#ff4500",
    raw_records_aqmon="#ff4500",
    raw_records_aux_mv="#ff4500",
    online_peak_monitor="deepskyblue",
    online_monitor="deepskyblue",
)


@export
def dataframe_to_wiki(df, float_digits=5, title="Awesome table", force_int: ty.Tuple = ()):
    """Convert a pandas dataframe to a dokuwiki table (which you can copy-paste onto the XENON wiki)

    :param df: dataframe to convert
    :param float_digits: format float to this number of digits.
    :param title: title of the table.
    :param force_int: tuple of column names to force to be integers

    """
    table = "^ %s " % title + "^" * (len(df.columns) - 1) + "^\n"
    table += "^ " + " ^ ".join(df.columns) + " ^\n"

    def format_float(x):
        if isinstance(x, float):
            return f"{x:.{float_digits}f}"
        return x

    force_int = np.where(np.in1d(df.columns.values, strax.to_str_tuple(force_int)))[0]

    for _, row in df.iterrows():
        table += (
            "| "
            + " | ".join(
                [
                    str(int(x) if i in force_int else format_float(x))
                    for i, x in enumerate(row.values.tolist())
                ]
            )
            + " |\n"
        )
    return table


@export
def print_versions(
    modules=("strax", "straxen", "cutax"),
    print_output=not _is_jupyter,
    include_python=True,
    return_string=False,
    include_git=True,
    include_all_local=True,
):
    """Print versions of modules installed.

    :param modules: Modules to print, should be str, tuple or list. E.g.
        print_versions(modules=('numpy', 'dddm',))
    :param return_string: optional. Instead of printing the message, return a string
    :param include_git: Include the current branch and latest commit hash
    :param include_all_local: Include all local modules (not in /opt/XENONnT/).
    :return: optional, the message that would have been printed

    """
    versions = defaultdict(list)
    if include_python:
        versions["module"] = ["python"]
        versions["version"] = [python_version()]
        versions["path"] = [sys.executable]
        versions["git"] = [None]

    local_modules = []    
    if include_all_local:
        # Only activate local-module detection if container paths are present
        module_values = sys.modules.values()
        in_container = any(
            mod and getattr(mod, '__file__', '').startswith('/opt/XENONnT/')
            for mod in module_values
        )
        if in_container:
            for mod_name, mod in list(sys.modules.items()):
                mod_version = getattr(mod, '__version__', None)
                mod_file = getattr(mod, '__file__', '')
                if mod_version and mod_file and not mod_file.startswith("/opt/XENONnT/"):
                    local_modules.append(mod_name.split(".")[0])

    
    modules = list(set(modules) | set(local_modules))
    for m in strax.to_str_tuple(modules):
        result = _version_info_for_module(m, include_git=include_git)
        if result is None:
            continue
        version, path, git_info = result
        versions["module"].append(m)
        versions["version"].append(version)
        versions["path"].append(path)
        versions["git"].append(git_info)
    df = pd.DataFrame(versions)
    info = f"Host {socket.getfqdn()}\n{df.to_string(index=False,)}"
    if print_output:
        print(info)
    if return_string:
        return info
    return df


def _version_info_for_module(module_name, include_git):
    try:
        mod = import_module(module_name)
    except (ModuleNotFoundError, ImportError):
        print(f"{module_name} is not installed")
        return
    git = None
    version = mod.__dict__.get("__version__", None)
    module_path_list = mod.__dict__.get("__path__") or [None]
    module_path = module_path_list[0]
    if include_git:
        try:
            repo = Repo(module_path, search_parent_directories=True)
        except InvalidGitRepositoryError:
            # not a git repo
            pass
        else:
            try:
                branch = repo.active_branch
            except TypeError:
                branch = "unknown"
            try:
                commit_hash = repo.head.object.hexsha
            except TypeError:
                commit_hash = "unknown"
            git = f"branch:{branch} | {commit_hash[:7]}"
    return version, module_path, git


@export
def utilix_is_configured(
    header: str = "RunDB",
    section: str = "xent_database",
    warning_message: ty.Union[None, bool, str] = None,
) -> bool:
    """Check if we have the right connection to.

    :param header: Which header to check in the utilix config file
    :param section: Which entry in the header to check to exist
    :param warning_message: If utilix is not configured, warn the user. if None -> generic warning
        if str -> use the string to warn if False -> don't warn
    :return: bool, can we connect to the Mongo database?

    """
    try:
        is_configured = (
            hasattr(straxen.uconfig, "get") and straxen.uconfig.get(header, section) is not None
        )
    except NoSectionError:
        is_configured = False

    should_report = bool(warning_message) or warning_message is None
    if not is_configured and should_report:
        if warning_message is None:
            warning_message = "Utilix is not configured, cannot proceed"
        warnings.warn(str(warning_message))
    return is_configured


@export
class TimeWidgets:
    def __init__(self):
        """Creates interactive time widgets which allow to convert a human readble date and time
        into a start and endtime in unix time nano- seconds."""
        self._created_widgets = False

    def create_widgets(self):
        """Creates time and time zone widget for simpler time querying.

        Note:
            Please be aware that the correct format for the time field
            is HH:MM.

        """
        utcend = datetime.datetime.utcnow()
        deltat = datetime.timedelta(minutes=60)
        utcstart = utcend - deltat
        import ipywidgets as widgets

        self._start_widget = self._create_date_and_time_widget(utcstart, "Start")
        self._end_widget = self._create_date_and_time_widget(utcend, "End")
        self._time_zone_widget = self._create_time_zone_widget()
        self._created_widgets = True

        return widgets.VBox(
            [
                widgets.HBox([self._time_zone_widget, widgets.HTML(value="ns:")]),
                self._start_widget,
                self._end_widget,
            ]
        )

    def get_start_end(self):
        """Returns start and end time of the specfied time interval in nano- seconds utc unix
        time."""
        if not self._created_widgets:
            raise ValueError('Please run first "create_widgets". ')

        time_zone = self._time_zone_widget.options[self._time_zone_widget.value][0]

        start, start_ns = self._convert_to_datetime(self._start_widget, time_zone)
        start = start.astimezone(tz=pytz.UTC)
        start = start.timestamp()
        start = int(start * straxen.units.s) + start_ns
        end, end_ns = self._convert_to_datetime(self._end_widget, time_zone)
        end = end.astimezone(tz=pytz.UTC)
        end = end.timestamp()
        end = int(end * straxen.units.s) + end_ns

        if start > end:
            warnings.warn("Start time is larger than endtime are you sure you wanted this?")
        return start, end

    @staticmethod
    def _create_date_and_time_widget(date_and_time, widget_describtion):
        import ipywidgets as widgets
        from ipywidgets import Layout

        date = datetime.date(date_and_time.year, date_and_time.month, date_and_time.day)
        date = widgets.DatePicker(
            description=widget_describtion, value=date, layout=Layout(width="225px"), disabled=False
        )

        time = "{:02d}:{:02d}".format(int(date_and_time.hour), int(date_and_time.minute))
        time = widgets.Text(value=time, layout=Layout(width="75px"), disabled=False)

        time_ns = widgets.Text(value="0", layout=Layout(width="150px"), disabled=False)

        return widgets.HBox([date, time, time_ns])

    @staticmethod
    def _create_time_zone_widget():
        import ipywidgets as widgets

        _time_zone_widget = widgets.Dropdown(
            options=[("CET", 0), ("UTC", 1)],
            value=0,
            description="Time Zone:",
        )
        return _time_zone_widget

    @staticmethod
    def _convert_to_datetime(time_widget, time_zone):
        """Converts values of widget into a timezone aware datetime object.

        :param time_widget: Widget Box containing a DatePicker and two text widget. The first text
            widget is used to set a day time. The second the time in nano-seconds.
        :param time_zone: pytz.timezone allowed string for a timezone.
        :return: timezone aware datetime object.

        """
        date_and_time = [c.value for c in time_widget.children]
        try:
            hour_and_minutes = datetime.datetime.strptime(date_and_time[1], "%H:%M")
        except ValueError as e:
            raise ValueError(
                "Cannot convert time into datetime object. "
                f"Expected the following formating HH:MM. {e}"
            )

        time = datetime.datetime.combine(date_and_time[0], datetime.time())
        time = time.replace(hour=hour_and_minutes.hour, minute=hour_and_minutes.minute)
        time_zone = pytz.timezone(time_zone)
        time = time_zone.localize(time)

        time_ns = int(date_and_time[2])

        return time, time_ns


@strax.Context.add_method
def extract_latest_comment(self):
    """Extract the latest comment in the runs-database. This just adds info to st.runs.

    Example:
        st.extract_latest_comment()
        st.select_runs(available=('raw_records'))

    """
    if self.runs is None or "comments" not in self.runs.keys():
        self.scan_runs(store_fields=("comments",))
        latest_comments = _parse_to_last_comment(self.runs["comments"])
        self.runs["comments"] = latest_comments
    return self.runs


def _parse_to_last_comment(comments):
    """Unpack to get the last comment (hence the -1) or give '' when there is none."""
    return [(c[-1]["comment"] if hasattr(c, "__len__") else "") for c in comments]


@export
def convert_array_to_df(array: np.ndarray) -> pd.DataFrame:
    """Converts the specified array into a DataFrame drops all higher dimensional fields during the
    process.

    :param array: numpy.array to be converted.
    :return: DataFrame with higher dimensions dropped.

    """
    keys = [key for key in array.dtype.names if array[key].ndim == 1]
    return pd.DataFrame(array[keys])


@export
def filter_kwargs(func, kwargs):
    """Filter out keyword arguments that are not in the call signature of func and return filtered
    kwargs dictionary."""
    params = inspect.signature(func).parameters
    if any([str(p).startswith("**") for p in params.values()]):
        # if func accepts wildcard kwargs, return all
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


@export
class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed.

    copied from
    https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306

    """

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val


@export
def total_size(o, handlers=None, verbose=False):
    """Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses: tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    from: https://code.activestate.com/recipes/577504/

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {
        tuple: iter,
        list: iter,
        deque: iter,
        dict: dict_handler,
        set: iter,
        frozenset: iter,
    }
    if handlers is not None:
        all_handlers.update(handlers)  # user handlers take precedence
    seen = set()  # track which object id's have already been seen
    default_size = getsizeof(0)  # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:  # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


@strax.Context.add_method
def dependency_tree(
    st,
    target="event_info",
    include_class=False,
    include_level=False,
    exclude_pattern=None,
    dump_plot=True,
    to_dir="./",
    format="svg",
):
    """Plot the dependency graph.

    :param target: str of the target plugin to check
    :param include_class: bool, whether include class name in label
    :param exclude_pattern: str, list or tuple of patterns to exclude
    :param dump_plot: bool, if True, save the plot to the to_dir
    :param to_dir: str, directory to save the plot
    :param format: str, format of the plot

    """
    if exclude_pattern is None:
        exclude_pattern = []
    elif isinstance(exclude_pattern, str):
        exclude_pattern = [exclude_pattern]

    if not isinstance(exclude_pattern, (list, tuple)):
        raise ValueError("exclude_pattern should be a str or list or tuple!")

    st._fixed_plugin_cache = None

    plugins = st._get_plugins((target,), run_id="0")
    graph = graphviz.Digraph(name=f"{to_dir}/{target}", strict=True)
    graph.attr(bgcolor="transparent")
    for d, p in plugins.items():
        if any([fnmatch.fnmatch(d, pattern) for pattern in exclude_pattern]):
            continue
        label = d
        if include_class:
            label += f"\n{p.__class__.__name__}"
        if include_level:
            label += f"\nlevel: {st.tree_levels[d]['level']} "
            label += f"order: {st.tree_levels[d]['order']}"
        graph.node(
            d,
            label=label,
            style="filled",
            fillcolor=kind_colors.get(p.data_kind_for(d), "grey"),
        )
        for dep in p.depends_on:
            graph.edge(d, dep)

    # dump the plot if need
    if dump_plot:
        graph.render(format=format)


@strax.Context.add_method
def storage_graph(
    st,
    run_id,
    target,
    graph=None,
    not_stored=None,
    include_class=False,
    dump_plot=True,
    to_dir="./",
    format="svg",
):
    """Plot the dependency graph indicating the storage of the plugins.

    :param target: str of the target plugin to check
    :param graph: graphviz.graphs.Digraph instance
    :param not_stored: set of plugins which are not stored
    :param dump_plot: bool, if True, save the plot to the to_dir
    :param to_dir: str, directory to save the plot
    :param format: str, format of the plot
    :return: all plugins that will be calculated when running self.make(run_id, target)

    The colors used in the graph represent the following storage states:
    - grey: strax.SaveWhen.NEVER
    - red: strax.SaveWhen.EXPLICIT
    - orange: strax.SaveWhen.TARGET
    - yellow: strax.SaveWhen.ALWAYS
    - green: target is stored

    """
    save_when_colors = {
        strax.SaveWhen.NEVER: "grey",
        strax.SaveWhen.EXPLICIT: "red",
        strax.SaveWhen.TARGET: "orange",
        strax.SaveWhen.ALWAYS: "yellow",
    }
    if not_stored is None:
        not_stored = set()
    # the set of plugins which are not stored
    if st.is_stored(run_id, target):
        # if the plugin is stored, fill in green
        fillcolor = "green"
    else:
        save_when = deepcopy(st._plugin_class_registry[target]().save_when)
        if isinstance(save_when, immutabledict):
            save_when = save_when[target]
        # if it is not stored, fill in the color according to save_when
        fillcolor = save_when_colors[save_when]

    if graph is None:
        graph = graphviz.Digraph(name=f"{to_dir}/{run_id}-{target}", strict=True)
        graph.attr(bgcolor="transparent")
    else:
        if not isinstance(graph, graphviz.graphs.Digraph):
            raise ValueError("graph should be a graphviz.Digraph instance!")
    # add a node of target
    p = st._Context__get_plugin(run_id, target)
    graph.node(
        target,
        label=f"{target}\n{p.__class__.__name__}" if include_class else None,
        style="filled",
        fillcolor=fillcolor,
    )
    if (not st.is_stored(run_id, target)) and (target not in not_stored):
        not_stored.add(target)
        depends_on = deepcopy(st._plugin_class_registry[target]().depends_on)
        depends_on = strax.to_str_tuple(depends_on)
        for dep in depends_on:
            # only add the node to the graph but not save the plot
            not_stored.update(
                st.storage_graph(
                    run_id,
                    dep,
                    graph=graph,
                    not_stored=not_stored,
                    include_class=include_class,
                    dump_plot=False,
                    to_dir=to_dir,
                )
            )
            graph.edge(target, dep)

    # dump the plot if need
    if dump_plot:
        graph.render(format=format)
    return not_stored
