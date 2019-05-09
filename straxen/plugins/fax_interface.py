"""Convert fax instructions to raw records
"""
import numpy as np
import os
import glob

import strax

export, __all__ = strax.exporter()


def records_needed(pulse_length, samples_per_record):
    """Return records needed to store pulse_length samples"""
    return np.ceil(pulse_length / samples_per_record).astype(np.int)


@export
def pax_to_records(input_filename,
                   samples_per_record=strax.DEFAULT_RECORD_LENGTH,
                   events_per_chunk=10):
    """Return pulse records array from pax zip input_filename

    This only works if you have pax installed in your strax environment,
    which is somewhat tricky.
    """

    # Monkeypatch matplotlib so pax is importable
    # See https://github.com/XENON1T/pax/pull/734
    import matplotlib
    matplotlib._cntr = None

    from pax import core  # Pax is not a dependency

    mypax = core.Processor('XENON1T', config_dict=dict(
        pax=dict(
            look_for_config_in_runs_db=False,
            plugin_group_names=['input'],
            encoder_plugin=None,
            input_name=input_filename),
        # Fast startup: skip loading big maps
        WaveformSimulator=dict(
            s1_light_yield_map='placeholder_map.json',
            s2_light_yield_map='placeholder_map.json',
            s1_patterns_file=None,
            s2_patterns_file=None)))

    print(f"Starting conversion, {events_per_chunk} evt/chunk")

    results = []

    def finish_results():
        nonlocal
        results
        records = np.concatenate(results)
        # In strax data, records are always stored
        # sorted, baselined and integrated
        records = strax.sort_by_time(records)
        strax.baseline(records)
        strax.integrate(records)
        print("Returning %d records" % len(records))
        results = []
        return records

    for event in mypax.get_events():
        event = mypax.process_event(event)

        if not len(event.pulses):
            # Triggerless pax data contains many empty events
            # at the end. With the fixed events per chunk setting
            # this can lead to empty files, which confuses strax.
            continue

        pulse_lengths = np.array([p.length
                                  for p in event.pulses])

        n_records_tot = records_needed(pulse_lengths,
                                       samples_per_record).sum()
        records = np.zeros(n_records_tot,
                           dtype=strax.record_dtype(samples_per_record))
        output_record_index = 0  # Record offset in data

        for p in event.pulses:
            n_records = records_needed(p.length, samples_per_record)

            for rec_i in range(n_records):
                r = records[output_record_index]
                r['time'] = (event.start_time
                             + p.left * 10
                             + rec_i * samples_per_record * 10)
                r['channel'] = p.channel
                r['pulse_length'] = p.length
                r['record_i'] = rec_i
                r['dt'] = 10

                # How much are we storing in this record?
                if rec_i != n_records - 1:
                    # There's more chunks coming, so we store a full chunk
                    n_store = samples_per_record
                    assert p.length > samples_per_record * (rec_i + 1)
                else:
                    # Just enough to store the rest of the data
                    # Note it's not p.length % samples_per_record!!!
                    # (that would be zero if we have to store a full record)
                    n_store = p.length - samples_per_record * rec_i

                assert 0 <= n_store <= samples_per_record
                r['length'] = n_store

                offset = rec_i * samples_per_record
                r['data'][:n_store] = p.raw_data[offset:offset + n_store]
                output_record_index += 1

        results.append(records)
        if len(results) >= events_per_chunk:
            yield finish_results()

    mypax.shutdown()

    if len(results):
        y = finish_results()
        if len(y):
            yield y


@export
@strax.takes_config(
    strax.Option('fax_dir', default='/data/', track=False,
                 help="Directory with fax instructions"),
    strax.Option('events_per_chunk', default=50, track=False,
                 help="Number of events to yield per chunk"),
    strax.Option('samples_per_record', default=strax.DEFAULT_RECORD_LENGTH, track=False,
                 help="Number of samples per record")
)
class RecordsFromFax(strax.Plugin):
    provides = 'raw_records'
    data_kind = 'raw_records'
    compressor = 'zstd'
    depends_on = tuple()
    parallel = False
    rechunk_on_save = False

    def infer_dtype(self):
        return strax.record_dtype(self.config['samples_per_record'])

    def iter(self, *args, **kwargs):
        if not os.path.exists(self.config['fax_dir']):
            raise FileNotFoundError(self.config['fax_dir'])
        input_dir = os.path.join(self.config['fax_dir'], self.run_id)
        fax_file = sorted(glob.glob(input_dir + '/XENON*.csv'))
        for file_i, in_fn in enumerate(pax_files):
            if (self.config['stop_after_zips']
                    and file_i >= self.config['stop_after_zips']):
                break
            yield from pax_to_records(
                in_fn,
                samples_per_record=self.config['samples_per_record'],
                events_per_chunk=self.config['events_per_chunk'])
