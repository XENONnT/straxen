import glob
import os
import shutil
import warnings
from collections import Counter
from immutabledict import immutabledict
import numpy as np
import numba

import strax

export, __all__ = strax.exporter()
__all__ += ['ARTIFICIAL_DEADTIME_CHANNEL']


# Just below the TPC acquisition monitor, see
# https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:dsg:daq:channel_groups
ARTIFICIAL_DEADTIME_CHANNEL = 799


class ArtificialDeadtimeInserted(UserWarning):
    pass


@export
@strax.takes_config(

    # All these must have track=False, so the raw_records hash never changes!

    # DAQ settings -- should match settings given to redax
    strax.Option('record_length', default=110, track=False, type=int,
                 help="Number of samples per raw_record"),
    strax.Option('max_digitizer_sampling_time',
                 default=10, track=False, type=int,
                 help="Highest interval time of the digitizer sampling times(s) used."),
    strax.Option('run_start_time', type=float, track=False, default=0,
                 help="time of start run (s since unix epoch)"),
    strax.Option('daq_chunk_duration', track=False,
                 default=int(5e9), type=int,
                 help="Duration of regular chunks in ns"),
    strax.Option('daq_overlap_chunk_duration', track=False,
                 default=int(5e8), type=int,
                 help="Duration of intermediate/overlap chunks in ns"),
    strax.Option('daq_compressor', default="lz4", track=False,
                 help="Algorithm used for (de)compressing the live data"),
    strax.Option('readout_threads', type=dict, track=False,
                 help="Dictionary of the readout threads where the keys "
                      "specify the reader and value the number of threads"),
    strax.Option('daq_input_dir', type=str, track=False,
                 help="Directory where readers put data"),

    # DAQReader settings
    strax.Option('safe_break_in_pulses', default=1000, track=False,
                 help="Time (ns) between pulses indicating a safe break "
                      "in the datastream -- gaps of this size cannot be "
                      "interior to peaklets."),
    strax.Option('erase', default=False, track=False,
                 help="Delete reader data after processing"),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number."))
class DAQReader(strax.Plugin):
    """
    Read the XENONnT DAQ-live_data from redax and split it to the
    appropriate raw_record data-types based on the channel-map.

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
    """
    provides = (
        'raw_records',
        'raw_records_he',  # high energy
        'raw_records_aqmon',
        'raw_records_nv',  # nveto raw_records (will not be stored long term)
        'raw_records_aqmon_nv',
        'raw_records_aux_mv',
        'raw_records_mv',    # mveto has to be last due to lineage
    )

    data_kind = immutabledict(zip(provides, provides))
    depends_on = tuple()
    parallel = 'process'
    rechunk_on_save = immutabledict(
        raw_records=False,
        raw_records_he=False,
        raw_records_aqmon=True,
        raw_records_nv=False,
        raw_records_aqmon_nv=True,
        raw_records_aux_mv=True,
        raw_records_mv=False,
    )
    compressor = 'lz4'
    __version__ = '0.0.0'

    def infer_dtype(self):
        return {
            d: strax.raw_record_dtype(
                samples_per_record=self.config["record_length"])
            for d in self.provides}

    def setup(self):
        self.t0 = int(self.config['run_start_time']) * int(1e9)
        self.dt_max = self.config['max_digitizer_sampling_time']
        self.n_readout_threads = sum(self.config['readout_threads'].values())
        if (self.config['safe_break_in_pulses']
                > min(self.config['daq_chunk_duration'],
                      self.config['daq_overlap_chunk_duration'])):
            raise ValueError(
                "Chunk durations must be larger than the minimum safe break"
                " duration (preferably a lot larger!)")

    def _path(self, chunk_i):
        return self.config["daq_input_dir"] + f'/{chunk_i:06d}'

    def _chunk_paths(self, chunk_i):
        """Return paths to previous, current and next chunk
        If any of them does not exist, or they are not yet populated
        with data from all readers, their path is replaced by False.
        """
        p = self._path(chunk_i)
        result = []
        for q in [p + '_pre', p, p + '_post']:
            if os.path.exists(q):
                n_files = self._count_files_per_chunk(q)
                if n_files >= self.n_readout_threads:
                    result.append(q)
                else:
                    print(f"Found incomplete folder {q}: "
                          f"contains {n_files} files but expected "
                          f"{self.n_readout_threads}. "
                          f"Waiting for more data.")
                    if self.source_finished():
                        # For low rates, different threads might end in a
                        # different chunck at the end of a run,
                        # still keep the results in this case.
                        print(f"Run finished correctly nonetheless: "
                              f"saving the results")
                        result.append(q)
                    else:
                        result.append(False)
            else:
                result.append(False)
        return tuple(result)

    @staticmethod
    def _partial_chunk_to_thread_name(partial_chunk):
        """Convert name of part of the chunk to the thread_name that wrote it"""
        return '_'.join(partial_chunk.split('_')[:-1])

    def _count_files_per_chunk(self, path_chunk_i):
        """
        Check that the files in the chunks have names consistent with
        the readout threads
        """
        counted_files = Counter(
            [self._partial_chunk_to_thread_name(p) for p in os.listdir(path_chunk_i)])
        for thread, n_counts in counted_files.items():
            if thread not in self.config['readout_threads']:
                raise ValueError(f'Bad data for {path_chunk_i}. Got {thread}')
            if n_counts > self.config['readout_threads'][thread]:
                raise ValueError(f'{thread} wrote {n_counts}, expected'
                                 f'{self.config["readout_threads"][thread]}')
        return sum(counted_files.values())

    def source_finished(self):
        end_dir = self.config["daq_input_dir"] + '/THE_END'
        if not os.path.exists(end_dir):
            return False
        else:
            return self._count_files_per_chunk(end_dir) >= self.n_readout_threads

    def is_ready(self, chunk_i):
        ended = self.source_finished()
        pre, current, post = self._chunk_paths(chunk_i)
        next_ahead = os.path.exists(self._path(chunk_i + 1))
        if (current and (
                (pre and post
                 or chunk_i == 0 and post
                 or ended and (pre and not next_ahead)))):
            return True
        return False

    def _load_chunk(self, path, start, end, kind='central'):
        records = [
            strax.load_file(
                fn,
                compressor=self.config["daq_compressor"],
                dtype=self.dtype_for('raw_records'))
            for fn in sorted(glob.glob(f'{path}/*'))]
        records = np.concatenate(records)
        records = strax.sort_by_time(records)

        first_start, last_start, last_end = None, None, None
        if len(records):
            first_start, last_start = records[0]['time'], records[-1]['time']
            # Records are sorted by (start)time and are of variable length.
            # Their end-times can differ. In the most pessimistic case we have
            # to look back one record length for each channel.
            tot_channels = np.sum([np.diff(x)+1 for x in
                                   self.config['channel_map'].values()])
            look_n_samples = self.config["record_length"] * tot_channels
            last_end = strax.endtime(records[-look_n_samples:]).max()
            if first_start < start or last_start >= end:
                raise ValueError(
                    f"Bad data from DAQ: chunk {path} should contain data "
                    f"that starts in [{start}, {end}), but we see start times "
                    f"ranging from {first_start} to {last_start}.")

        if kind == 'central':
            result = records
            break_time = None
        else:
            # Find a time at which we can safely partition the data.
            min_gap = self.config['safe_break_in_pulses']
            if not len(records) or last_end + min_gap < end:
                # There is enough room at the end of the data
                break_time = end - min_gap
                result = records if kind == 'post' else records[:0]
            else:
                # Let's hope there is some quiet time in the middle
                try:
                    result, break_time = strax.from_break(
                        records,
                        safe_break=min_gap,
                        # Records from the last chunk can extend as far as:
                        not_before=(start
                                    + self.config['record_length'] * self.dt_max),
                        left=kind == 'post',
                        tolerant=False)
                except strax.NoBreakFound:
                    # We still have to break somewhere, but this can involve
                    # throwing away data.
                    # Let's do it at the end of the chunk
                    # TODO: find a better time, e.g. a longish-but-not-quite
                    # satisfactory gap
                    break_time = end - min_gap

                    # Mark the region where data /might/ be removed with
                    # artificial deadtime.
                    dead_time_start = (
                            break_time - self.config['record_length'] * self.dt_max)
                    warnings.warn(
                        f"Data in {path} is so dense that no {min_gap} "
                        f"ns break exists: data loss inevitable. "
                        f"Inserting artificial deadtime between "
                        f"{dead_time_start} and {end}.",
                        ArtificialDeadtimeInserted)

                    if kind == 'pre':
                        # Give the artificial deadtime past the break
                        result = self._artificial_dead_time(
                            start=break_time, end=end, dt=self.dt_max)
                    else:
                        # Remove data that would stick out
                        result = records[strax.endtime(records) <= break_time]
                        # Add the artificial deadtime until the break
                        result = strax.sort_by_time(
                            np.concatenate([result,
                                            self._artificial_dead_time(
                                                start=dead_time_start,
                                                end=break_time, dt=self.dt_max)]))

        if self.config['erase']:
            shutil.rmtree(path)
        return result, break_time

    def _artificial_dead_time(self, start, end, dt):
        return strax.dict_to_rec(
            dict(time=[start],
                 length=[(end - start) // dt],
                 dt=[dt],
                 channel=[ARTIFICIAL_DEADTIME_CHANNEL]),
            self.dtype_for('raw_records'))

    def compute(self, chunk_i):
        dt_central = self.config['daq_chunk_duration']
        dt_overlap = self.config['daq_overlap_chunk_duration']

        t_start = chunk_i * (dt_central + dt_overlap)
        t_end = t_start + dt_central

        pre, current, post = self._chunk_paths(chunk_i)
        r_pre, r_post = None, None
        break_pre, break_post = t_start, t_end

        if pre:
            if chunk_i == 0:
                warnings.warn(
                    f"DAQ is being sloppy: there should be no pre dir {pre} "
                    f"for chunk 0. We're ignoring it.",
                    UserWarning)
            else:
                r_pre, break_pre = self._load_chunk(
                    path=pre,
                    start=t_start - dt_overlap,
                    end=t_start,
                    kind='pre')

        r_main, _ = self._load_chunk(
            path=current,
            start=t_start,
            end=t_end,
            kind='central')

        if post:
            r_post, break_post = self._load_chunk(
                path=post,
                start=t_end,
                end=t_end + dt_overlap,
                kind='post')

        # Concatenate the result.
        records = np.concatenate([
            x for x in (r_pre, r_main, r_post)
            if x is not None])

        # Split records by channel
        result_arrays = split_channel_ranges(
            records,
            np.asarray(list(self.config['channel_map'].values())))
        del records

        # Convert to strax chunks
        result = dict()
        for i, subd in enumerate(self.config['channel_map']):
            if len(result_arrays[i]):
                # dt may differ per subdetector
                dt = result_arrays[i]['dt'][0]
                # Convert time to time in ns since unix epoch.
                # Ensure the offset is a whole digitizer sample
                result_arrays[i]["time"] += dt * (self.t0 // dt)

            # Ignore data from the 'blank' channels, corresponding to
            # channels that have nothing connected
            if subd.endswith('blank'):
                continue

            result_name = 'raw_records'
            if subd.startswith('nveto'):
                result_name += '_nv'
            elif subd != 'tpc':
                result_name += '_' + subd
            result[result_name] = self.chunk(
                start=self.t0 + break_pre,
                end=self.t0 + break_post,
                data=result_arrays[i],
                data_type=result_name)

        print(f"Read chunk {chunk_i:06d} from DAQ")
        for r in result.values():
            # Print data rate / data type if any
            if r._mbs() > 0:
                print(f"\t{r}")
        return result


@export
class Fake1TDAQReader(DAQReader):
    provides = (
        'raw_records',
        'raw_records_diagnostic',
        'raw_records_aqmon')

    data_kind = immutabledict(zip(provides, provides))


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
            if r['channel'] > right:
                continue
            elif r['channel'] >= left:
                which_detector[r_i] = d_i
                n_in_detector[d_i] += 1
                break
            else:
                # channel_ranges should be sorted ascending.
                print(r['time'], r['channel'], channel_ranges)
                raise ValueError(
                    "Bad data from DAQ: data in unknown channel!")

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
