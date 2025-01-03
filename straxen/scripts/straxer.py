"""Process a single run with straxen."""

import argparse
import datetime
import logging
import time
import os
import psutil
import json
import importlib
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process a single run with straxen",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "run_id", metavar="RUN_ID", type=str, help="ID of the run to process; usually the run name."
    )
    parser.add_argument(
        "--package", default="straxen", help="Where to load the context from (straxen/cutax/pema)"
    )
    parser.add_argument("--context", default="xenonnt_online", help="Name of context to use")
    parser.add_argument(
        "--target",
        default="event_info",
        nargs="*",
        help="Target final data type to produce. Can be a list for multicore mode.",
    )
    parser.add_argument(
        "--context_kwargs", type=json.loads, help="Use a json-file to load the context with"
    )
    parser.add_argument(
        "--register_from_file", type=str, help="do st.register_all from a specified file"
    )
    parser.add_argument(
        "--config_kwargs", type=json.loads, help="Use a json-file to set the context to"
    )
    parser.add_argument(
        "--from_scratch",
        action="store_true",
        help=(
            "Start processing at raw_records, regardless of what data is available. "
            "Saving will ONLY occur to ./strax_data! If you already have the target"
            "data in ./strax_data, you need to delete it there first."
        ),
    )
    parser.add_argument(
        "--max_messages",
        default=4,
        type=int,
        help=(
            "Size of strax's internal mailbox buffers. "
            "Lower to reduce memory usage, at increasing risk of deadlocks."
        ),
    )
    parser.add_argument(
        "--timeout", default=None, type=int, help="Strax' internal mailbox timeout in seconds"
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help=(
            "Number of worker threads/processes. "
            "Strax will multithread (1/plugin) even if you set this to 1."
        ),
    )
    parser.add_argument(
        "--notlazy",
        action="store_true",
        help="Forbid lazy single-core processing. Not recommended.",
    )
    parser.add_argument("--multiprocess", action="store_true", help="Allow multiprocessing.")
    parser.add_argument(
        "--multi_target",
        action="store_true",
        help=(
            "Allow st.make to be called with multiple targets at once "
            "(otherwise loop over the target list)"
        ),
    )
    parser.add_argument(
        "--shm", action="store_true", help="Allow passing data via /dev/shm when multiprocessing."
    )
    parser.add_argument(
        "--profile_to",
        default="",
        help="Filename to output profile information to. If omitted,no profiling will occur.",
    )
    parser.add_argument(
        "--profile_ram",
        action="store_true",
        help=(
            "Use memory_profiler for a more accurate measurement of the "
            "peak RAM usage of the process."
        ),
    )
    parser.add_argument(
        "--diagnose_sorting",
        action="store_true",
        help="Diagnose sorting problems during processing",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging to stdout")
    parser.add_argument(
        "--build_lowlevel",
        action="store_true",
        help="Build low-level data even if the context forbids it.",
    )
    parser.add_argument(
        "--only_strax_data", action="store_true", help="Only use ./strax_data (if not on dali)."
    )
    parser.add_argument("--add_folder", default="", help="Also add folder to st.storage")
    parser.add_argument(
        "--print_alive",
        default=300,
        help="Print that straxer is still running every this many [seconds]",
    )
    return parser.parse_args()


def setup_context(args):
    # reimport to be safe
    import strax
    import straxen

    context_module = importlib.import_module(f"{args.package}.contexts")
    st = getattr(context_module, args.context)()

    if args.context_kwargs:
        logging.info(f"set context kwargs {args.context_kwargs}")
        st = getattr(context_module, args.context)(**args.context_kwargs)

    if args.config_kwargs:
        logging.info(f"set context options to {args.config_kwargs}")
        st.set_config(to_dict_tuple(args.config_kwargs))

    if args.register_from_file:
        register_to_context(st, args.register_from_file)

    if args.diagnose_sorting:
        st.set_config(dict(diagnose_sorting=True))

    st.context_config["allow_multiprocess"] = args.multiprocess
    st.context_config["allow_shm"] = args.shm
    st.context_config["allow_lazy"] = not (args.notlazy is True)

    if args.timeout is not None:
        st.context_config["timeout"] = args.timeout
    st.context_config["max_messages"] = args.max_messages

    if args.build_lowlevel:
        st.context_config["forbid_creation_of"] = tuple()
    else:
        st.context_config["forbid_creation_of"] = straxen.DAQReader.provides

    if args.from_scratch:
        for q in st.storage:
            q.take_only = ("raw_records",)
        st.storage.append(
            strax.DataDirectory("./strax_data", overwrite="always", provide_run_metadata=False)
        )
    if args.only_strax_data:
        for sf in st.storage:
            # Set all others to read only
            sf.readonly = True
        for sf in st.storage:
            if hasattr(sf, "path"):
                if sf.path == "./strax_data":
                    break
        else:
            st.storage += [strax.DataDirectory("./strax_data")]

    if args.add_folder != "":
        for sf in st.storage:
            # Set all others to read only
            sf.readonly = True
        if os.path.exists(args.add_folder):
            st.storage += [strax.DataDirectory(args.add_folder)]

    if st.is_stored(args.run_id, args.target):
        logging.warning("This data is already available. Straxer is done")
        sys.exit(0)
    return st


def run(args):
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.info(f"Starting processing of run {args.run_id} until {args.target}")

    # These imports take a bit longer, so it's nicer
    # to do them after argparsing (so --help is fast)
    import strax
    import straxen
    import pandas as pd

    # For showing data availability below
    pd.options.display.max_rows = 999
    logging.info(
        straxen.print_versions(
            tuple({"strax", "straxen", args.package}), return_string=True, print_output=False
        ),
    )

    logging.info("Starting context")
    st = setup_context(args)
    logging.info("Context started")

    # Reactivate after https://github.com/XENONnT/straxen/issues/586
    logging.info(f"Checking availabilty")
    logging.info(f"Available\n{str(st.available_for_run(args.run_id))}")

    logging.info("Infer start/end")
    try:
        md = st.run_metadata(args.run_id)
        t_start = md["start"].replace(tzinfo=datetime.timezone.utc).timestamp()
        t_end = md["end"].replace(tzinfo=datetime.timezone.utc).timestamp()
        st.config["run_start_time"] = md["start"].timestamp()
        st.context_config["free_options"] = tuple(
            list(st.context_config["free_options"]) + ["run_start_time"]
        )
    except strax.RunMetadataNotAvailable:
        for t in ("raw_records", "records", "peaklets"):
            if st.is_stored(args.run_id, t):
                break
        t_start, t_end = st.estimate_run_start_and_end(args.run_id, t)
        t_start, t_end = t_start / 1e9, t_end / 1e9
    run_duration = t_end - t_start
    logging.info(f"Infer start/end: run is {run_duration:.1f} s")

    start_alive_thread(logging, args.print_alive)

    process = psutil.Process(os.getpid())
    peak_ram = 0

    def run_make(targets):
        """For target or a list of targets, run st.make and print the progress."""
        nonlocal peak_ram
        if not args.multi_target:
            logging.info(f"Checking if {targets} is stored")
            if st.is_stored(args.run_id, targets):
                logging.info(f"{args.run_id}:{targets} is stored")
                return
        logging.info(f"Start processing {args.run_id}:{targets}")

        def get_results():
            kwargs = dict(
                run_id=args.run_id,
                targets=targets,
                max_workers=int(args.workers),
                allow_multiple=args.multi_target,
                progress_bar=False,
                save=strax.to_str_tuple(targets),
            )

            if args.profile_to:
                with strax.profile_threaded(args.profile_to):
                    yield from st.get_iter(**kwargs)
            else:
                yield from st.get_iter(**kwargs)

        clock_start = None
        for i, d in enumerate(get_results()):
            mem_mb = process.memory_info().rss / 1e6
            peak_ram = max(mem_mb, peak_ram)

            if not len(d):
                logging.info(f"Got chunk {i}, but it is empty! Using {mem_mb:.1f} MB RAM.")
                continue

            # Compute detector/data time left
            time_end = d.end / 1e9
            dt = time_end - t_start
            time_left = t_end - time_end

            msg = (
                f"Got {len(d)} items. "
                f"Now {dt:.1f} sec / {100 * dt / run_duration:.1f}% into the run. "
                f"Using {mem_mb:.1f} MB RAM. "
            )
            if clock_start is not None:
                # Compute processing job clock time left
                d_clock = time.time() - clock_start
                clock_time_left = time_left / (dt / d_clock)
                msg += f"ETA {clock_time_left:.2f} sec."
            else:
                clock_start = time.time()
            logging.info(msg)
        logging.info(f"{targets} finished! Took {time.time() - clock_start:.1f} s")

    proc_start = time.time()
    if args.multi_target:
        run_make(args.target)
    else:
        for target in strax.to_str_tuple(args.target):
            run_make(target)
    logging.info(
        f"Straxer is done in {time.time() - proc_start :.1f} s! peak RAM usage was"
        f" ~{peak_ram:.1f} MB."
    )


def register_to_context(st, module: str):
    if not os.path.exists(module):
        raise FileNotFoundError(f"No such file {module}")
    assert module.endswith(".py"), "only py files please!"
    folder, file = os.path.split(module)
    sys.path.append(folder)
    to_register = importlib.import_module(os.path.splitext(file)[0])
    st.register_all(to_register)
    logging.info(f"Successfully registered {file}. Printing plugins")

    for key, plugin in st._plugin_class_registry.items():
        logging.info(f"{key}\t{plugin}")


def to_dict_tuple(res: dict):
    """Convert list configs to tuple configs."""
    res = res.copy()
    for k, v in res.copy().items():
        if isinstance(v, list):
            # Remove lists to tuples
            res[k] = tuple(_v if not isinstance(_v, list) else tuple(_v) for _v in v)
    return res


def print_is_alive(log, print_timeout):
    while True:
        log.info("Straxer still running")
        time.sleep(print_timeout)


def start_alive_thread(log, print_timeout):
    from threading import Thread

    thread = Thread(
        name="Ping alive", target=print_is_alive, args=(log, print_timeout), daemon=True
    )
    log.info(f"Starting thread to ping that we are still running")
    thread.start()


def main():
    args = parse_args()
    if args.profile_ram:
        from memory_profiler import memory_usage

        mem = memory_usage(proc=(run, (args,)))
        print(f"Memory profiler says peak RAM usage was: {max(mem):.1f} MB")
        sys.exit()
    else:
        sys.exit(run(args))


if __name__ == "__main__":
    main()
