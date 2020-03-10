import glob
import os
import shutil
import warnings

import numpy as np

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
    strax.Option('daq_chunk_duration', track=False,
                 default=int(5e9), type=int,
                 help="Duration of regular chunks in ns"),
    strax.Option('daq_overlap_chunk_duration', track=False,
                 default=int(5e8), type=int,
                 help="Duration of intermediate/overlap chunks in ns"),
    strax.Option('daq_input_dir', type=str, track=False,
                 help="Directory where readers put data"),
    strax.Option('n_readout_threads', type=int, track=False,
                 help="Number of readout threads producing strax data files"),
    strax.Option('safe_break_in_pulses', default=1000, track=False,
                 help="Time (ns) between pulses indicating a safe break "
                      "in the datastream -- gaps of this size cannot be "
                      "interior to peaklets."),
    strax.Option('erase', default=False, track=False,
                 help="Delete reader data after processing"),
    strax.Option('compressor', default="blosc", track=False,
                 help="Algorithm used for (de)compressing the live data"),
    strax.Option('record_length', default=110, track=False, type=int,
                 help="Number of samples per raw_record"),
    strax.Option('digitizer_sampling_resolution',
                 default=10, track=False, type=int,
                 help="Digitizer sampling resolution"),
    strax.Option('run_start_time', type=float, track=False, default=0,
                 help="time of start run (s since unix epoch)"))
class DAQReader(strax.Plugin):
    """Read the XENONnT DAQ

    Does nothing whatsoever to the pulse data; not even baselining.
    """
    provides = 'raw_records'
    depends_on = tuple()
    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    def infer_dtype(self):
        return strax.raw_record_dtype(
            samples_per_record=self.config["record_length"])

    def setup(self):
        self.t0 = int(self.config['run_start_time']) * int(1e9)
        self.dt = self.config['digitizer_sampling_resolution']

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
                n_files = len(os.listdir(q))
                if n_files >= self.config['n_readout_threads']:
                    result.append(q)
                else:
                    print(f"Found incomplete folder {q}: "
                          f"contains {n_files} files but expected "
                          f"{self.config['n_readout_threads']}. "
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

    def source_finished(self):
        end_dir = self.config["daq_input_dir"] + '/THE_END'
        if not os.path.exists(end_dir):
            return False
        else:
            return len(os.listdir(end_dir)) >= self.config['n_readout_threads']

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
                compressor=self.config["compressor"],
                dtype=self.dtype_for('raw_records'))
            for fn in sorted(glob.glob(f'{path}/*'))]
        records = np.concatenate(records)
        records = strax.sort_by_time(records)

        first_start, last_start, last_end = None, None, None
        if len(records):
            first_start, last_start = records[0]['time'], records[-1]['time']
            # Records are fixed length, so we also know the last end:
            last_end = strax.endtime(records[-1])

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
                                    + self.config['record_length'] * self.dt),
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
                            break_time - self.config['record_length'] * self.dt)
                    warnings.warn(
                        f"Data in {path} is so dense that no {min_gap} "
                        f"ns break exists: data loss inevitable. "
                        f"Inserting artificial deadtime between "
                        f"{dead_time_start} and {end}.",
                        ArtificialDeadtimeInserted)

                    if kind == 'pre':
                        # Give the artificial deadtime past the break
                        result = self._artificial_dead_time(
                            start=break_time, end=end)
                    else:
                        # Remove data that would stick out
                        result = records[strax.endtime(records) <= break_time]
                        # Add the artificial deadtime until the break
                        result = strax.sort_by_time(
                            np.concatenate([result,
                                            self._artificial_dead_time(
                                                start=dead_time_start,
                                                end=break_time)]))

        if self.config['erase']:
            shutil.rmtree(path)
        return result, break_time

    def _artificial_dead_time(self, start, end):
        return strax.dict_to_rec(
            dict(time=[start],
                 length=[(end - start) // self.dt],
                 dt=[self.dt],
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

        if len(records):
            # Convert time to time in ns since unix epoch.
            # Ensure the offset is a whole digitizer sample
            records["time"] += self.dt * (self.t0 // self.dt)

        result = self.chunk(
            start=self.t0 + break_pre,
            end=self.t0 + break_post,
            data=records)
        print(f"Read chunk {chunk_i:04d}, {result} from DAQ")
        return result
