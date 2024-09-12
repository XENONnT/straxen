import os
import argparse
import pymongo
import hug
import socket
from datetime import datetime, timedelta
import pytz
import time
import threading

import strax
import straxen

st: strax.Context
max_load_mb = 1000
max_return_mb = 10


def load_context(name, extra_dirs=None):
    global st
    st = getattr(straxen.contexts, name)()
    for sf in st.storage:
        sf.readonly = True

    if extra_dirs is not None:
        for d in extra_dirs:
            if os.access(d, os.W_OK) is not True:
                raise IOError(f"No writing access to {d}. Check mountpoints ")
            st.storage.append(strax.DataDirectory(d, readonly=True))

    st.context_config["allow_incomplete"] = True
    st.context_config["forbid_creation_of"] = (
        "*",
        # In case someone has an old strax that doesn't support * here:
        "raw_records",
        "records",
        "peaklets",
        "peaks",
        "event_basics",
        "event_info",
    )


@hug.exception(Exception)
def handle_exception(exception):
    return {"error": str(exception)}


@hug.get("/get_data")
def get_data(
    run_id: hug.types.text,
    target: hug.types.text,
    max_n: hug.types.number = 1000,
    start: hug.types.float_number = None,
    end: hug.types.float_number = None,
    selection_str: hug.types.text = None,
):
    try:
        st
    except NameError:
        return "Context not loaded???"

    if len(run_id) != 6 and run_id.isdecimal():
        run_id = "%06d" % int(run_id)

    if max_n <= 0:
        max_n = None
    t0, _ = st.estimate_run_start_and_end(run_id, target)
    if start is None or end is None:
        time_range = None
    else:
        time_range = [t0 + int(start * int(1e9)), t0 + int(end * int(1e9))]

    md = st.get_metadata(run_id, target)  # don't catch exception, already nice
    if not md["chunks"]:
        raise ValueError(
            "No chunks available -- either the first chunk has "
            "yet to be written, or something is wrong."
        )
    full_run_bytes = st.size_mb(run_id, target) * int(1e6)

    if time_range is None:
        # Time range is the full run. Find the second range corresponding
        # to that (as best we can)
        if "chunks" not in md or not md["chunks"]:
            print(f"Run {run_id} has no chunks??")
        else:
            time_range = [md["chunks"][0]["start"], md["chunks"][-1]["end"]]

    if time_range is None or not max_n:
        # We have to load all the data
        load_bytes = full_run_bytes

    else:
        # Let's see how much we really need to load, given max_n.
        load_at_least_n = 0
        load_bytes = 0
        for chunk_info in md["chunks"]:
            if chunk_info["start"] > time_range[1] or chunk_info["end"] < time_range[0]:
                # None of this chunk will be loaded
                continue
            load_bytes += chunk_info["nbytes"]

            if chunk_info["start"] > time_range[0] and chunk_info["end"] < time_range[1]:
                # All of this chunk will be loaded
                load_at_least_n += chunk_info["n"]

            if load_at_least_n > max_n:
                # No point loading any more, we already exceeded max_n
                time_range[1] = chunk_info["end"]
                break

    if load_bytes > max_load_mb * int(1e6):
        raise ValueError(
            f"Cannot load {target} for run {run_id}: it would "
            f"take more than {max_load_mb} MB RAM to load."
        )

    x = st.get_array(run_id, target, time_range=time_range, selection_str=selection_str)
    if max_n:
        x = x[:max_n]
    size_mb = x.nbytes / int(1e6)
    if size_mb > max_return_mb:
        raise ValueError(
            f"Not converting {target} for run {run_id} "
            f"to json since the binary data is {size_mb:.1f} MB. "
            "Try lowering max_n."
        )

    return [dict(zip(row.dtype.names, row)) for row in x]


def now(plus=0):
    return datetime.now(pytz.utc) + timedelta(seconds=plus)


def _set_state():
    """Inform the bootstrax collection we're running microstrax."""
    # DAQ database
    daq_db_name = "daq"
    daq_uri = straxen.get_mongo_uri(
        header="rundb_admin",
        user_key="mongo_daq_username",
        pwd_key="mongo_daq_password",
        url_key="mongo_daq_url",
    )
    daq_client = pymongo.MongoClient(daq_uri)
    daq_db = daq_client[daq_db_name]
    bs_coll = daq_db["eb_monitor"]
    while True:
        microstrax_state = dict(
            host=socket.getfqdn(),
            pid=os.getpid(),
            time=now(),
            state="hosting microstrax",
            max_load_mb=args.max_load_mb,
            max_return_mb=args.max_return_mb,
            context=args.context,
            n_dirs=len(args.extra_dirs),
        )
        try:
            daq_db.command("ping")
        except Exception as timeout_error:
            print(f"Ran into {timeout_error}. Can be bad. Let's take a nap")
        bs_coll.insert_one(microstrax_state)
        time.sleep(9)


def set_state():
    """Open thread to set the state of microstrax."""
    update_thread = threading.Thread(name="Update_db", target=_set_state)
    update_thread.setDaemon(True)
    update_thread.start()


def main():
    load_context(args.context, extra_dirs=args.extra_dirs)
    set_state()

    while True:
        try:
            hug.API(__name__).http.serve(port=args.port)
        except pymongo.errors.NotMasterError:
            print("Ran into the infamous pymongo.errors.NotMasterError. Sleep for a sec")
            time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Start a microservice to return strax data as json"
    )
    parser.add_argument(
        "--context", default="xenonnt_online", help="Name of straxen context to use"
    )
    parser.add_argument("--port", default=8000, type=int, help="HTTP port to serve on")
    parser.add_argument("--max_load_mb", default=1000, type=int)
    parser.add_argument("--max_return_mb", default=10, type=int)
    parser.add_argument("--extra_dirs", nargs="*", help="Extra directories to look for data")
    args = parser.parse_args()

    max_load_mb = args.max_load_mb
    max_return_mb = args.max_return_mb

    main()
