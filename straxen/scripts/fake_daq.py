import argparse
import os
from copy import copy
import shutil
import time

import numpy as np
import strax
import straxen

parser = argparse.ArgumentParser(description="Fake DAQ to test XENONnT eventbuilder prototype")

parser.add_argument(
    "--input_path",
    default="./test_input_data",
    help="Directory with input data (used to fake new run)",
)
parser.add_argument("--input_run", default="180215_1029", help="Run id of input data")
parser.add_argument("--detector", default="tpc", help="Specifies the detector type (tpc/nveto).")
parser.add_argument("--output", default="./from_fake_daq", help="Output directory")
parser.add_argument(
    "--output_run", default=None, help="Output run id to use. If omitted, use same as input"
)
parser.add_argument("--compressor", default="lz4", help="Compressor to use for live records")
parser.add_argument(
    "--rate",
    default=0,
    type=int,
    help="Output rate in MBraw/sec. If omitted, emit data as fast as possible",
)
parser.add_argument(
    "--realtime", action="store_true", help="Emit data at same pace as it was acquired"
)
parser.add_argument("--shm", action="store_true", help="Operate in /dev/shm")
parser.add_argument(
    "--no_run_metadata",
    action="store_true",
    help=(
        "Produce Fake DAQ data even if you have lost the run metadata. "
        "Some useful sanity checks will be disabled."
    ),
)
parser.add_argument("--chunk_duration", default=2.0, type=float, help="Chunk size in sec (not ns)")
parser.add_argument(
    "--stop_after",
    default=float("inf"),
    type=float,
    help="Stop after this much MB written/loaded in",
)
parser.add_argument(
    "--sync_chunk_duration",
    default=0.2,
    type=float,
    help="Synchronization chunk size in sec (not ns)",
)
args = parser.parse_args()

if args.shm:
    output_dir = "/dev/shm/from_fake_daq"
else:
    output_dir = args.output
output_run = args.output_run if args.output_run else args.input_run


def main():
    global output_dir

    # Get context for reading
    st = strax.Context(
        storage=strax.DataDirectory(args.input_path, provide_run_metadata=True, readonly=True),
        register=straxen.plugins.pax_interface.RecordsFromPax,
        config=straxen.contexts.x1t_common_config,
        **straxen.contexts.common_opts,
    )

    n_readout_threads = 8
    if args.detector == "tpc":
        n_channels = st.config["n_tpc_pmts"]
    elif args.detector == "nveto":
        n_channels = st.config["n_nveto_pmts"]
    else:
        raise ValueError("Detector type not supported.")
    channels_per_reader = np.ceil(n_channels / n_readout_threads)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Copy over metadata
    run_start = None
    if not args.no_run_metadata:
        run_md = st.run_metadata(args.input_run)
        st2 = st.new_context(storage=strax.DataDirectory(output_dir), replace=True)
        run_md.setdefault("strax_defaults", dict())
        run_md["strax_defaults"]["n_readout_threads"] = n_readout_threads
        run_md["strax_defaults"]["compressor"] = args.compressor
        st2.storage[0].write_run_metadata(output_run, run_md)
        del st2
        run_start = int(int(1e9) * int(run_md["start"].timestamp()))

    if args.rate:
        print("Preparing payload data: slurping into memory")

    chunk_sizes = []
    chunk_data_compressed = []

    if args.detector == "tpc":
        source = st.get_iter(args.input_run, "raw_records")
        sampling = 10  # Hardcoded these numbers since records might be empty.
    elif args.detector == "nveto":
        source = st.get_iter(args.input_run, "nveto_pre_raw_records")
        sampling = 2
    else:
        raise ValueError("Detector type not supported.")

    buffer: strax.Chunk = next(source)
    payload_t_start = payload_t_end = buffer.start
    input_exhausted = False

    chunk_i = -1
    while len(buffer) or not input_exhausted:
        chunk_i += 1
        desired_end = payload_t_end + int(  # endtime of last chunk
            int(1e9) * (args.sync_chunk_duration if chunk_i % 2 else args.chunk_duration)
        )
        while buffer.end < desired_end:
            try:
                buffer = strax.Chunk.concatenate([buffer, next(source)])
            except StopIteration:
                input_exhausted = True
                break
        t_0 = time.time()

        # NB: this is not a regular strax chunk split!
        keep = buffer.data["time"] < desired_end
        records = buffer.data[keep]
        buffer.data = buffer.data[~keep]
        buffer.start = 0  # We don't use buffer.start anymore, fortunately
        payload_t_end = desired_end

        # Restore baseline, clear metadata, fix time
        if run_start is None:
            run_start = records["time"][0]
        records["time"] = records["time"] - run_start
        assert np.all(records["time"] % sampling == 0)

        chunk_sizes.append(records.nbytes)
        result = []
        for reader_i in range(n_readout_threads):
            first_channel = reader_i * channels_per_reader
            r = records[
                (records["channel"] >= first_channel)
                & (records["channel"] < first_channel + channels_per_reader)
            ]
            r = strax.io.COMPRESSORS[args.compressor]["compress"](r)
            result.append(r)

        if args.rate:
            # Slurp into memory
            chunk_data_compressed.append(result)
        else:
            # Simulate realtime DAQ / emit data immediately
            # Cannot slurp in advance, else time would be offset.
            write_chunk(chunk_i, result)
            if chunk_i % 2 == 0:
                dt = args.chunk_duration
            else:
                dt = args.sync_chunk_duration

            t_sleep = dt - (time.time() - t_0)
            wrote_mb = chunk_sizes[chunk_i] / 1e6

            print(
                f"{chunk_i}: wrote {wrote_mb:.1f} MB_raw"
                + (f", sleep for {t_sleep:.2f} s" if args.realtime else "")
            )
            if args.realtime:
                if t_sleep < 0:
                    if chunk_i % 2 == 0:
                        print("Fake DAQ too slow :-(")
                else:
                    time.sleep(t_sleep)

        if sum(chunk_sizes) / 1e6 > args.stop_after:
            # TODO: background thread does not terminate!
            break

    if args.rate:
        total_raw = sum(chunk_sizes) / 1e6
        total_comp = sum([len(y) for x in chunk_data_compressed for y in x]) / 1e6
        total_dt = (payload_t_end - payload_t_start) / int(1e9)
        print(
            f"Prepared {len(chunk_sizes)} chunks "
            f"spanning {total_dt:.1f} sec, "
            f"{total_raw:.2f} MB raw "
            f"({total_comp:.2f} MB compressed)"
        )
        if args.rate:
            takes = total_raw / args.rate
        else:
            takes = total_dt
        input(f"Press enter to start DAQ for {takes:.1f} sec")

        # Emit at fixed rate
        for chunk_i, reader_data in enumerate(chunk_data_compressed):
            t_0 = time.time()

            write_chunk(chunk_i, reader_data)

            wrote_mb = chunk_sizes[chunk_i] / 1e6
            t_sleep = wrote_mb / args.rate - (time.time() - t_0)

            print(f"{chunk_i}: wrote {wrote_mb:.1f} MB_raw, sleep for {t_sleep:.2f} s")
            if t_sleep < 0:
                if chunk_i % 2 == 0:
                    print("Fake DAQ too slow :-(")
            else:
                time.sleep(t_sleep)

    end_dir = output_dir + "/THE_END"
    os.makedirs(end_dir)
    for i in range(n_readout_threads):
        with open(end_dir + f"/{i:06d}", mode="w") as f:
            f.write("That's all folks!")

    print("Fake DAQ done")


def write_to_dir(c, outdir):
    tempdir = outdir + "_temp"
    os.makedirs(tempdir)
    for reader_i, x in enumerate(c):
        with open(f"{tempdir}/reader_{reader_i}", "wb") as f:
            f.write(copy(x))  # Copy needed for honest shm writing?
    os.rename(tempdir, outdir)


def write_chunk(chunk_i, reader_data):
    big_chunk_i = chunk_i // 2

    if chunk_i % 2 != 0:
        write_to_dir(reader_data, output_dir + "/%06d_post" % big_chunk_i)
        write_to_dir(reader_data, output_dir + "/%06d_pre" % (big_chunk_i + 1))
    else:
        write_to_dir(reader_data, output_dir + "/%06d" % big_chunk_i)


if __name__ == "__main__":
    main()
