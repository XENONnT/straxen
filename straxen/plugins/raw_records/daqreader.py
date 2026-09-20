import os
import glob
import warnings
from typing import Tuple
from collections import Counter
from immutabledict import immutabledict

import numpy as np
import numba
import strax

export, __all__ = strax.exporter()
__all__.extend(["ARTIFICIAL_DEADTIME_CHANNEL"])

# Just below the TPC acquisition monitor, see
# https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:daq:channel_groups
ARTIFICIAL_DEADTIME_CHANNEL = 799


class ArtificialDeadtimeInserted(UserWarning):
    pass


@export
@strax.takes_config(
    # All these must have track=False, so the raw_records hash never changes!
    # DAQ settings -- should match settings given to redax
    strax.Option(
        "record_length", default=110, track=False, type=int, help="Number of samples per raw_record"
    ),
    strax.Option(
        "max_digitizer_sampling_time",
        default=10,
        track=False,
        type=int,
        help="Highest interval time of the digitizer sampling times(s) used.",
    ),
    strax.Option(
        "run_start_time",
        type=float,
        track=False,
        default=0,
        help="time of start run (s since unix epoch)",
    ),
    strax.Option(
        "daq_chunk_duration",
        track=False,
        default=int(5e9),
        type=int,
        help="Duration of regular chunks in ns",
    ),
    strax.Option(
        "daq_overlap_chunk_duration",
        track=False,
        default=int(5e8),
        type=int,
        help="Duration of intermediate/overlap chunks in ns",
    ),
    strax.Option(
        "daq_compressor",
        default="lz4",
        track=False,
        help="Algorithm used for (de)compressing the live data",
    ),
    strax.Option(
        "readout_threads",
        type=dict,
        track=False,
        help=(
            "Dictionary of the readout threads where the keys "
            "specify the reader and value the number of threads"
        ),
    ),
    strax.Option("daq_input_dir", type=str, track=False, help="Directory where readers put data"),
    # DAQReader settings
    strax.Option(
        "safe_break_in_pulses",
        default=strax.DEFAULT_CHUNK_SPLIT_NS,
        track=False,
        infer_type=False,
        help=(
            "Time (ns) between pulses indicating a safe break "
            "in the datastream -- gaps of this size cannot be "
            "interior to peaklets."
        ),
    ),
    strax.Option(
        "channel_map",
        track=False,
        type=immutabledict,
        infer_type=False,
        help="immutabledict mapping subdetector to (min, max) channel number.",
    ),
)
class DAQReader(strax.Plugin):
    """Read the XENONnT DAQ-live_data from redax and split it to the appropriate raw_record data-
    types based on the channel-map.

    Does nothing whatsoever to the live_data; not even baselining.

    Provides:
     - raw_records: (tpc)raw_records.
     - raw_records_he: raw_records for the high energy boards
       digitizing the top PMT-array at lower amplification.
     - raw_records_nv: neutron veto raw_records; only stored temporary
       as the software coincidence trigger not applied yet.
     - raw_records_mv: muon veto raw_records.
     - raw_records_aqmon: raw_records for the acquisition monitor (_nv
       for neutron veto).
     - raw_records_sc: raw_records for the neutron gun backscatter PMT
       (_sc)

    """

    provides: Tuple[str, ...] = (
        "raw_records",
        "raw_records_he",  # high energy
        "raw_records_aqmon",
        "raw_records_nv",  # nveto raw_records (will not be stored long term)
        "raw_records_aqmon_nv",
        "raw_records_aux_mv",
        "raw_records_sc",
        "raw_records_mv",  # mveto has to be last due to lineage
    )

    data_kind = immutabledict(zip(provides, provides))
    depends_on: Tuple = tuple()
    parallel = "process"
    rechunk_on_load = True
    chunk_source_size_mb = strax.DEFAULT_CHUNK_SIZE_MB  # 200 MB
    rechunk_on_save = immutabledict(
        raw_records=False,
        raw_records_he=False,
        raw_records_aqmon=True,
        raw_records_nv=False,
        raw_records_aqmon_nv=True,
        raw_records_aux_mv=True,
        raw_records_sc=False,
        raw_records_mv=False,
    )
    chunk_target_size_mb = 500
    compressor = "lz4"
    __version__ = "0.0.0"
    input_timeout = 300

    def infer_dtype(self):
        return {
            d: strax.raw_record_dtype(samples_per_record=self.config["record_length"])
            for d in self.provides
        }

    def setup(self):
        self.t0 = int(self.config["run_start_time"]) * int(1e9)
        self.dt_max = self.config["max_digitizer_sampling_time"]
        self.n_readout_threads = sum(self.config["readout_threads"].values())
        if self.config["safe_break_in_pulses"] > min(
            self.config["daq_chunk_duration"], self.config["daq_overlap_chunk_duration"]
        ):
            raise ValueError(
                "Chunk durations must be larger than the minimum safe break"
                " duration (preferably a lot larger!)"
            )

    def _path(self, chunk_i):
        return self.config["daq_input_dir"] + f"/{chunk_i:06d}"

    def _chunk_paths(self, chunk_i):
        """Return paths to previous, current and next chunk If any of them does not exist, or they
        are not yet populated with data from all readers, their path is replaced by False."""
        p = self._path(chunk_i)
        result = []
        for q in [p + "_pre", p, p + "_post"]:
            if os.path.exists(q):
                n_files = self._count_files_per_chunk(q)
                if n_files >= self.n_readout_threads:
                    result.append(q)
                else:
                    print(
                        f"Found incomplete folder {q}: "
                        f"contains {n_files} files but expected "
                        f"{self.n_readout_threads}. "
                        "Waiting for more data."
                    )
                    if self.source_finished():
                        # For low rates, different threads might end in a
                        # different chunck at the end of a run,
                        # still keep the results in this case.
                        print(f"Run finished correctly nonetheless: saving the results")
                        result.append(q)
                    else:
                        result.append(False)
            else:
                result.append(False)
        return tuple(result)

    @staticmethod
    def _partial_chunk_to_thread_name(partial_chunk):
        """Convert name of part of the chunk to the thread_name that wrote it."""
        return "_".join(partial_chunk.split("_")[:-1])

    def _count_files_per_chunk(self, path_chunk_i):
        """Check that the files in the chunks have names consistent with the readout threads."""
        counted_files = Counter(
            [self._partial_chunk_to_thread_name(p) for p in os.listdir(path_chunk_i)]
        )
        for thread, n_counts in counted_files.items():
            if thread not in self.config["readout_threads"]:
                raise ValueError(f"Bad data for {path_chunk_i}. Got {thread}")
            if n_counts > self.config["readout_threads"][thread]:
                raise ValueError(
                    f"{thread} wrote {n_counts}, expected{self.config['readout_threads'][thread]}"
                )
        return sum(counted_files.values())

    def source_finished(self):
        end_dir = self.config["daq_input_dir"] + "/THE_END"
        if not os.path.exists(end_dir):
            return False
        else:
            return self._count_files_per_chunk(end_dir) >= self.n_readout_threads

    def is_ready(self, chunk_i):
        ended = self.source_finished()
        pre, current, post = self._chunk_paths(chunk_i)
        next_ahead = os.path.exists(self._path(chunk_i + 1))
        if current and (
            (pre and post or chunk_i == 0 and post or ended and (pre and not next_ahead))
        ):
            return True
        return False

    def _load_chunk(self, path, start, end, kind="central"):
        records = [
            strax.load_file(
                fn, compressor=self.config["daq_compressor"], dtype=self.dtype_for("raw_records")
            )
            for fn in sorted(glob.glob(f"{path}/*"))
        ]
        records = np.concatenate(records)
        records = strax.sort_by_time(records)

        first_start, last_start, last_end = None, None, None
        if len(records):
            first_start, last_start = records[0]["time"], records[-1]["time"]
            # Records are sorted by (start)time and are of variable length.
            # Their end-times can differ. In the most pessimistic case we have
            # to look back one record length for each channel.
            tot_channels = np.sum([np.diff(x) + 1 for x in self.config["channel_map"].values()])
            look_n_samples = self.config["record_length"] * tot_channels
            last_end = strax.endtime(records[-look_n_samples:]).max()
            if first_start < start or last_start >= end:
                raise ValueError(
                    f"Bad data from DAQ: chunk {path} should contain data "
                    f"that starts in [{start}, {end}), but we see start times "
                    f"ranging from {first_start} to {last_start}."
                )

        if kind == "central":
            result = records
            break_time = None
        else:
            # Find a time at which we can safely partition the data.
            min_gap = self.config["safe_break_in_pulses"]
            if not len(records) or last_end + min_gap < end:
                # There is enough room at the end of the data
                break_time = end - min_gap
                result = records if kind == "post" else records[:0]
            else:
                # Let's hope there is some quiet time in the middle
                try:
                    result, break_time = strax.from_break(
                        records,
                        safe_break=min_gap,
                        # Records from the last chunk can extend as far as:
                        not_before=(start + self.config["record_length"] * self.dt_max),
                        left=kind == "post",
                        tolerant=False,
                    )
                except strax.NoBreakFound:
                    # We still have to break somewhere, but this can involve
                    # throwing away data.
                    # Let's do it at the end of the chunk
                    # Maybe find a better time, e.g. a longish-but-not-quite
                    # satisfactory gap
                    break_time = end - min_gap

                    # Mark the region where data /might/ be removed with
                    # artificial deadtime.
                    dead_time_start = break_time - self.config["record_length"] * self.dt_max
                    warnings.warn(
                        f"Data in {path} is so dense that no {min_gap} "
                        "ns break exists: data loss inevitable. "
                        "Inserting artificial deadtime between "
                        f"{dead_time_start} and {end}.",
                        ArtificialDeadtimeInserted,
                    )

                    if kind == "pre":
                        # Give the artificial deadtime past the break
                        result = self._artificial_dead_time(
                            start=break_time, end=end, dt=self.dt_max
                        )
                    else:
                        # Remove data that would stick out
                        result = records[strax.endtime(records) <= break_time]
                        # Add the artificial deadtime until the break
                        result = strax.sort_by_time(
                            np.concatenate(
                                [
                                    result,
                                    self._artificial_dead_time(
                                        start=dead_time_start, end=break_time, dt=self.dt_max
                                    ),
                                ]
                            )
                        )
        return result, break_time

    def _artificial_dead_time(self, start, end, dt):
        return strax.dict_to_rec(
            dict(
                time=[start],
                length=[(end - start) // dt],
                dt=[dt],
                channel=[ARTIFICIAL_DEADTIME_CHANNEL],
            ),
            self.dtype_for("raw_records"),
        )

    def compute(self, chunk_i):
        dt_central = self.config["daq_chunk_duration"]
        dt_overlap = self.config["daq_overlap_chunk_duration"]

        t_start = chunk_i * (dt_central + dt_overlap)
        t_end = t_start + dt_central

        pre, current, post = self._chunk_paths(chunk_i)

        break_pre = t_start
        break_post = t_end

        # Keep the only references in this mutable list.
        parts = [None, None, None]

        if pre:
            if chunk_i == 0:
                warnings.warn(
                    f"DAQ is being sloppy: there should be no pre dir {pre} "
                    "for chunk 0. We're ignoring it.",
                    UserWarning,
                )
            else:
                parts[0], break_pre = self._load_chunk(
                    path=pre,
                    start=t_start - dt_overlap,
                    end=t_start,
                    kind="pre",
                )

        parts[1], _ = self._load_chunk(
            path=current,
            start=t_start,
            end=t_end,
            kind="central",
        )

        if post:
            parts[2], break_post = self._load_chunk(
                path=post,
                start=t_end,
                end=t_end + dt_overlap,
                kind="post",
            )

        # Make this once
        if not hasattr(self, "_channel_to_detector"):
            self._channel_to_detector = _make_channel_to_detector(
                self.config["channel_map"]
            )
        result_arrays, output_sorted = split_channel_ranges_from_parts(
            parts,
            self.config["channel_map"],
            self._channel_to_detector,
            self.dtype_for("raw_records"),
            self.t0,
        )

        del parts

        # Fallback only. Absolute time has been added and sorted
        # If a detector output is unexpectedly unsorted, fall back to
        # the standard allocating sort for that detector only.
        # Could also be optimized, this path should never execute
        subdetectors = tuple(self.config["channel_map"])
        for i, sorted_ in enumerate(output_sorted):
            if not sorted_:
                warnings.warn(
                    f"{subdetectors[i]} output was not sorted; "
                    "falling back to strax.sort_by_time"
                )
                result_arrays[i] = strax.sort_by_time(result_arrays[i])

        # Convert to strax chunks
        result = dict()
        for i, subd in enumerate(subdetectors):
            # Ignore data from the 'blank' channels, corresponding to
            # channels that have nothing connected
            if subd.endswith("blank"):
                continue

            result_name = "raw_records"
            if subd.startswith("nveto"):
                result_name += "_nv"
            elif subd != "tpc":
                result_name += "_" + subd
            result[result_name] = self.chunk(
                start=self.t0 + break_pre,
                end=self.t0 + break_post,
                data=result_arrays[i],
                data_type=result_name,
            )

        print(f"Read chunk {chunk_i:06d} from DAQ")
        for r in result.values():
            # Print data rate / data type if any
            if r._mbs() > 0:
                print(f"\t{r}")
        return result

@numba.njit(nogil=True)
def count_by_detector_and_get_dt(
    records,
    channel_to_detector,
    counts,
    detector_dt,
    detector_seen,
):
    """
    Count records per detector and record the first sample width for each

    Unknown or unmapped channels are treated as invalid DAQ data and raise a
    `ValueError`.
    """
    for i in range(len(records)):
        ch = records[i]["channel"]

        if ch < 0 or ch >= len(channel_to_detector):
            raise ValueError("Bad data from DAQ: data in unknown channel")

        d = channel_to_detector[ch]

        if d < 0:
            raise ValueError("Bad data from DAQ: data in unknown channel")

        counts[d] += 1

        # This becomes exactly output[d]["dt"][0]
        if not detector_seen[d]:
            detector_dt[d] = records[i]["dt"]
            detector_seen[d] = True

def _make_channel_to_detector(channel_map):
    """
    Produces a lut mapping from channel (int) to detector (tpc, nv, mv,...) 
    """
    max_channel = max(right for left, right in channel_map.values())

    lut = np.full(max_channel + 1, -1, dtype=np.int16)

    for d, (left, right) in enumerate(channel_map.values()):
        if np.any(lut[left:right + 1] != -1):
            raise ValueError("Overlapping channel ranges")

        lut[left:right + 1] = d

    return lut

@numba.njit(nogil=True, cache=True)
def scatter_part(
    records,
    channel_to_detector,
    outputs,
    positions,
    time_offsets,
    last_time,
    last_channel,
    output_sorted,
):
    """
    Scatter raw records into preallocated detector outputs.

    Records are routed according to ``channel_to_detector`` and copied
    directly into their final detector-specific buffers. The detector
    time offset is applied during the copy, avoiding a later full-array
    pass.

    Ordering state is preserved across successive calls through 
    `positions`, `last_time`, and `last_channel`. 
    Once an output is found to be unsorted, `output_sorted[d]` 
    remains False.

    Assumes `outputs` have already been allocated to exact final sizes 
    and `positions` contains the next write position for each detector.
    """
    for i in range(len(records)):
        ch = records[i]["channel"]
        d = channel_to_detector[ch]

        j = positions[d]
        t = records[i]["time"]

        # j > 0 means this detector has already received a record.
        if j:
            if output_sorted[d]:
                lt = last_time[d]

                if (
                    t < lt
                    or (
                        t == lt
                        and ch < last_channel[d]
                    )
                ):
                    output_sorted[d] = False

        last_time[d] = t
        last_channel[d] = ch

        # Fetch detector output once.
        out = outputs[d]

        # Full record copy.
        out[j] = records[i]

        # Fold the old later full-array time adjustment into this write.
        out[j]["time"] = t + time_offsets[d]

        positions[d] = j + 1

@export
def split_channel_ranges_from_parts(
    parts,
    channel_map,
    channel_to_detector, 
    dtype,
    t0,
):
    """
    Rewrite of split_channel_ranges to remove concatenation in compute. 

    `parts` MUST be a mutable list.

    The entries are cleared as soon as they have been copied into
    their final detector arrays.
    """
    n_detectors = len(channel_map)

    counts = np.zeros(n_detectors, dtype=np.int64)

    # dt dtype in raw_records is int16 currently, 
    # but int64 here is tiny and makes the offset 
    # arithmetic straightforward.
    detector_dt = np.zeros(n_detectors, dtype=np.int64)
    detector_seen = np.zeros(n_detectors, dtype=np.bool_)

    # determine final output sizes and the first dt for each
    # detector.
    for x in parts:
        if x is not None:
            count_by_detector_and_get_dt(
                x,
                channel_to_detector,
                counts,
                detector_dt,
                detector_seen,
            )

    # Final Buffer
    outputs = numba.typed.List()

    for n in counts:
        outputs.append(
            np.empty(int(n), dtype=dtype)
        )

    # Time offset that compute previously applied afterwards.
    time_offsets = np.zeros(n_detectors, dtype=np.int64)

    for d in range(n_detectors):
        if detector_seen[d]:
            dt = detector_dt[d]
            time_offsets[d] = dt * (np.int64(t0) // dt)

    positions = np.zeros(n_detectors, dtype=np.int64)

    # Sortedness maintained while copying.
    last_time = np.zeros(n_detectors, dtype=np.int64)
    last_channel = np.zeros(n_detectors, dtype=np.int64)
    output_sorted = np.ones(n_detectors, dtype=np.bool_)

    # copy directly to final buffers.
    for i in range(len(parts)):
        x = parts[i]

        if x is None:
            continue

        scatter_part(
            x,
            channel_to_detector,
            outputs,
            positions,
            time_offsets,
            last_time,
            last_channel,
            output_sorted,
        )

        # Critical: caller can not retain a reference.
        parts[i] = None

    if not np.array_equal(positions, counts):
        raise RuntimeError("Internal error while splitting records")

    return outputs, output_sorted

@export
@numba.njit(nogil=True, cache=True)
def split_channel_ranges(records, channel_ranges):
    """Return numba.List of record arrays in channel_ranges.

    ~2.5x as fast as a naive implementation with np.in1d

    """
    n_subdetectors = len(channel_ranges)
    which_detector = np.zeros(len(records), dtype=np.int8)
    n_in_detector = np.zeros(n_subdetectors, dtype=np.int64)

    # First loop to count number of records per detector
    for r_i, r in enumerate(records):
        for d_i in range(n_subdetectors):
            left, right = channel_ranges[d_i]
            if r["channel"] > right:
                continue
            elif r["channel"] >= left:
                which_detector[r_i] = d_i
                n_in_detector[d_i] += 1
                break
            else:
                # channel_ranges should be sorted ascending.
                print(r["time"], r["channel"], channel_ranges)
                raise ValueError("Bad data from DAQ: data in unknown channel!")

    # Allocate memory
    results = numba.typed.List()
    for d_i in range(n_subdetectors):
        results.append(np.empty(n_in_detector[d_i], dtype=records.dtype))

    # Second loop to fill results
    # This is slightly faster than using which_detector == d_i masks,
    # since it only needs one loop over the data.
    n_placed = np.zeros(n_subdetectors, dtype=np.int64)
    for r_i, r in enumerate(records):
        d_i = which_detector[r_i]
        results[d_i][n_placed[d_i]] = r
        n_placed[d_i] += 1

    return results
