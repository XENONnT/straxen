import glob
import os
import shutil

import numpy as np

import strax


__all__ = ['DAQReader']


@strax.takes_config(
    strax.Option('safe_break_in_pulses', default=1000,
                 help="Time (ns) between pulse starts indicating a safe break "
                      "in the datastream -- peaks will not span this."),
    strax.Option('input_dir', type=str, track=False,
                 help="Directory where readers put data"),
    strax.Option('n_readout_threads', type=int, track=False,
                 help="Number of readout threads producing strax data files"),
    strax.Option('erase', default=False, track=False,
                 help="Delete reader data after processing"),
    strax.Option('compressor', default="blosc", track=False,
                 help="Algorithm used for (de)compressing the live data"),
    strax.Option('run_start_time', default=0., type=float, track=False,
                 help="time of start run (s since unix epoch)"))
class DAQReader(strax.Plugin):
    provides = 'raw_records'
    depends_on = tuple()
    dtype = strax.record_dtype()
    parallel = 'process'
    rechunk_on_save = False
    compressor = 'lz4'

    def _path(self, chunk_i):
        return self.config["input_dir"] + f'/{chunk_i:06d}'

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
        end_dir = self.config["input_dir"] + '/THE_END'
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

    def _load_chunk(self, path, kind='central'):
        records = [strax.load_file(fn,
                                   compressor=self.config["compressor"],
                                   dtype=strax.record_dtype())
                   for fn in sorted(glob.glob(f'{path}/*'))]
        records = np.concatenate(records)
        records = strax.sort_by_time(records)
        if kind == 'central':
            result = records
        else:
            result = strax.from_break(
                records,
                safe_break=self.config['safe_break_in_pulses'],
                left=kind == 'post',
                tolerant=True)
        if self.config['erase']:
            shutil.rmtree(path)
        return result

    def compute(self, chunk_i):
        pre, current, post = self._chunk_paths(chunk_i)
        if chunk_i == 0 and pre:
            pre = False
            print(f"There should be no {pre} dir for chunk 0: ignored")
        records = np.concatenate(
            ([self._load_chunk(pre, kind='pre')] if pre else [])
            + [self._load_chunk(current)]
            + ([self._load_chunk(post, kind='post')] if post else [])
        )

        strax.baseline(records)
        strax.integrate(records)

        if len(records):
            # Convert time to time in ns since unix epoch.
            # Ensure the offset is a whole digitizer sample
            t0 = int(self.config["run_start_time"] * int(1e9))
            dt = records[0]['dt']
            t0 = dt * (t0 // dt)
            records["time"] += t0

            timespan_sec = (records[-1]['time'] - records[0]['time']) / 1e9
            print(f'{chunk_i}: read {records.nbytes/1e6:.2f} MB '
                  f'({len(records)} records, '
                  f'{timespan_sec:.2f} sec) from readers')
        else:
            print(f'{chunk_i}: read an empty chunk!')

        return records
