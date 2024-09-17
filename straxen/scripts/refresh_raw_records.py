import argparse
from ast import literal_eval
import json
import os
import os.path as osp

import numpy as np
from tqdm import tqdm

import strax
import straxen


def main():
    parser = argparse.ArgumentParser(description="Refresh strax raw_records created with < v0.9.0")
    parser.add_argument("--parent_folder", default=".", help="Folder with raw_records folders.")
    parser.add_argument(
        "--no_run_metadata",
        action="store_true",
        help=(
            "Refresh even if you have lost the run metadata. "
            "Some useful sanity checks will be disabled."
        ),
    )
    parser.add_argument(
        "--procrustean",
        action="store_true",
        help=(
            "Delete records that fall across chunk boundaries. "
            "Use as a last resort for triggerless data you do not want "
            "to reconvert from pax properly."
        ),
    )
    parser.add_argument(
        "--xnt",
        action="store_true",
        help="Set if records came from the DAQReader, not RecordsFromPax",
    )
    parser.add_argument("run_id", help="run_id to convert")
    args = parser.parse_args()
    run_id = args.run_id
    parent_folder = args.parent_folder

    ##
    # Prepare context
    ##
    st = strax.Context(
        storage=[
            strax.DataDirectory(
                parent_folder,
                # We WILL overwrite your data
                # just not through the usual means:
                readonly=True,
            )
        ],
        **straxen.contexts.common_opts,
    )
    if args.xnt:
        st.register(straxen.DAQReader)
    else:
        st.register(straxen.RecordsFromPax)

    ##
    # Get metadata
    ##
    folder = st.storage[0].find(st.key_for(run_id, "raw_records"), fuzzy_for="raw_records")[1]
    md = st.get_metadata(run_id, "raw_records")
    metadata_fn = os.path.join(folder, strax.dirname_to_prefix(folder) + "-metadata.json")
    assert osp.exists(metadata_fn)
    dtype = np.dtype(literal_eval(md["dtype"]))
    record_length = strax.record_length_from_dtype(dtype)

    if not args.no_run_metadata:
        run_md = st.run_metadata(run_id)
        run_start, run_end = [
            int(x.timestamp()) * int(1e9) for x in [run_md["start"], run_md["end"]]
        ]
    else:
        run_start, run_end = None, None

    if not len(md["chunks"]):
        raise ValueError("Cannot convert data: no chunks!")
    if "start" in md["chunks"][0]:
        raise ValueError("This data was already converted")

    ##
    # Convert data
    ##
    last_endtime = 0
    for i, c in enumerate(tqdm(md["chunks"], desc=f"Converting raw_records for {run_id}")):
        filename = osp.join(folder, c["filename"])
        rr = strax.load_file(filename, dtype=dtype, compressor=md["compressor"])

        if not len(rr):
            raise ValueError("Cannot convert data with empty chunks")

        if rr[0]["time"] < last_endtime:
            if args.procrustean:
                to_cut = rr["time"] < last_endtime
                print(f"[!!] Removing {to_cut.sum()} records from chunk {i} to remove overlaps!")
                rr = rr[~to_cut]
            else:
                raise ValueError(
                    f"Cannot convert data: chunk {i}'s data starts "
                    f"at {rr[0]['time']} while the previous chunk's data "
                    f"ended at {last_endtime}"
                )

        if i == 0:
            if run_start is not None:
                c["start"] = run_start
            else:
                c["start"] = rr[0]["time"]
        else:
            c["start"] = last_endtime
        c["end"] = last_endtime = strax.endtime(rr).max()

        new_rr = np.zeros(len(rr), dtype=strax.raw_record_dtype(record_length))
        strax.copy_raw_records(rr, new_rr)
        if "baseline" in rr.dtype.fields:
            # Undo baselining
            new_rr["data"] = rr["baseline"][:, np.newaxis] - rr["data"]
            strax.zero_out_of_bounds(new_rr)

        c["run_id"] = run_id
        c["nbytes"] = new_rr.nbytes
        c["filesize"] = strax.save_file(filename, new_rr, compressor=md["compressor"])
        # We must rewrite these too, the chunk count could have changed
        # if args.procrustean
        c["n"] = len(rr)
        c["first_time"] = rr[0]["time"]
        c["first_endtime"] = strax.endtime(rr[0])
        c["last_time"] = rr[-1]["time"]
        c["last_endtime"] = strax.endtime(rr[-1])

    if run_start is None:
        run_start = md["chunks"][0]["start"]
        run_end = md["chunks"][-1]["start"]

    ##
    # Set and write out new metadata
    ##
    md["start"] = run_start
    md["end"] = run_end
    md["run_id"] = run_id
    md["data_kind"] = "raw_records"
    md["converted_from_old_strax"] = md["strax_version"]
    md["strax_version"] = strax.__version__
    md["dtype"] = np.dtype(strax.raw_record_dtype(record_length)).descr.__repr__()

    with open(metadata_fn, mode="w") as f:
        f.write(json.dumps(md, sort_keys=True, indent=4, cls=strax.NumpyJSONEncoder))


if __name__ == "__main__":
    main()
