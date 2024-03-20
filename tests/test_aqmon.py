import os.path
from typing import List, Optional

from unittest import TestCase, skipIf
import numpy as np
import strax
import straxen
import shutil
from straxen.test_utils import nt_test_run_id as test_run_id_nT
from straxen import VetoIntervals, VetoProximity


class DummyAqmonHits(strax.Plugin):
    """Dummy plugin to make some aqmon hits that may span ON and OFF signals over chunks.

    There are two channels, that signify ON or OFF of a logic signal

    We are interested in the deadtime, which is the time difference between ON and OFF singals. This
    plugin computes that deadtime, even if we have missed the first ON (so we start with OFF) or
    missed the last OFF (such that the last signal is an ON). If we miss one ON at the start or one
    OFF at the end, we should just consider all time before the first OFF or all the time after the
    last ON as deadtime.

    """

    vetos_per_chunk = strax.Config(
        default=list(range(1, 10)),
        help=(
            "The number of ON/OFF signals per chunk, preferably a "
            "combination of odd and even such that we have both "
            "unmatched ON/OFF singals"
        ),
    )
    start_with_channel_on = strax.Config(
        default=True, help="If True, start with an ON signal, otherwise start if OFF"
    )
    channel_on = strax.Config(
        default=straxen.AqmonChannels.BUSY_START,
        help="ON channel. Just some channel known to the VetoIntervals plugin.",
        type=int,
    )
    channel_off = strax.Config(
        default=straxen.AqmonChannels.BUSY_STOP,
        help="OFF channel. Just some channel known to the VetoIntervals plugin",
        type=int,
    )
    veto_duration_max = strax.Config(default=300e9, help="Max duration of one veto [ns].")
    # Start from scratch
    depends_on = ()
    parallel = False
    provides = straxen.AqmonHits.provides
    dtype = straxen.AqmonHits.dtype
    save_when = strax.SaveWhen.NEVER

    # Keep track for we need this from plugin to plugin
    _last_channel_was_off = True
    _last_endtime = 0

    # Will overwrite this with a mutable default in the test
    TOTAL_DEADTIME: Optional[List] = None
    TOTAL_SIGNALS: Optional[List] = None

    # This value should be larger than the duration of a single hit
    gap_ns_between_chunks = 10

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < len(self.vetos_per_chunk)

    def compute(self, chunk_i):
        if chunk_i == 0:
            self._last_channel_was_off = self.start_with_channel_on

        n_vetos = self.vetos_per_chunk[chunk_i]
        res = np.zeros(n_vetos, self.dtype)

        # Add some randomly increasing times (larger than previous chunk
        res["time"] = (
            self._last_endtime
            + 1
            + np.cumsum(np.random.randint(low=2, high=self.veto_duration_max, size=n_vetos))
        )
        res["dt"] = 1
        res["length"] = 1

        if self._last_channel_was_off:
            # Previous chunk, we ended with an off, so now we start with an ON
            res["channel"][::2] = self.channel_on
            res["channel"][1::2] = self.channel_off
            starts = res[::2]["time"]
            stops = res[1::2]["time"]

            # Sum the time differences between starts and stops to get the deadtime
            self.TOTAL_DEADTIME += [np.sum(stops - starts[: len(stops)])]

        else:
            # Previous chunk, we ended with an ON, so now we start with an OFF
            res["channel"][1::2] = self.channel_on
            res["channel"][::2] = self.channel_off
            starts = res[1::2]["time"]
            stops = res[::2]["time"]

            # Ignore the first stop (stops last from previous chunk)
            self.TOTAL_DEADTIME += [np.sum(stops[1:] - starts[: len(stops) - 1])]

            # Additionally, add the deadtime that spans the chunks (i.e. the first stop)
            # The gap_ns_between_chunks may look a bit magic, but it's correct
            if chunk_i != 0:
                self.TOTAL_DEADTIME += [self.gap_ns_between_chunks]
            self.TOTAL_DEADTIME += [stops[0] - self._last_endtime]

        # Track the total number of signals both ON and OFF
        self.TOTAL_SIGNALS += [n_vetos]

        previous_end = self._last_endtime
        self._last_endtime = res["time"][-1] + self.gap_ns_between_chunks
        self._last_channel_was_off = res["channel"][-1] == self.channel_off
        if len(res) > 1:
            assert np.sum(res["channel"])

        if chunk_i == len(self.vetos_per_chunk) - 1 and not self._last_channel_was_off:
            # There is one ON without an OFF at the end of the run in
            # the last chunk. This should fill all the way to the end as
            # deadtime:
            dt = self._last_endtime - res["time"][-1]
            self.TOTAL_DEADTIME += [dt]
        return self.chunk(start=previous_end, end=self._last_endtime, data=res)


class DummyEventBasics(strax.Plugin):
    """Get evenly spaced random duration events that don't overlap."""

    n_chunks = strax.Config(default=3)
    event_time_range = strax.Config(
        default=(0, 3e6), help="Where to span the durations of chunks over"
    )
    events_per_chunk = strax.Config(default=20)
    event_durations = strax.Config(default=(1000, 20_000), help="event durations (min, max) ns")
    depends_on = ()
    save_when = strax.SaveWhen.ALWAYS
    provides = "event_basics"
    data_kind = "events"
    dtype = strax.time_fields
    _events = None

    def source_finished(self):
        return True

    def is_ready(self, chunk_i):
        return chunk_i < self.n_chunks

    def get_events_this_chunk(self, chunk_i):
        if self._events is None:
            res = np.zeros(int(self.events_per_chunk * self.n_chunks), dtype=self.dtype)
            times = np.linspace(*self.event_time_range, len(res))
            res["time"] = times

            endtimes = res["time"] + np.random.randint(*self.event_durations, size=len(res))
            # Don't allow overlapping events
            endtimes[:-1] = np.clip(endtimes[:-1], 0, res["time"][1:] - 1)
            res["endtime"] = endtimes
            self._events = np.split(res, self.n_chunks)
        return self._events[chunk_i]

    def chunk_start(self, chunk_i) -> int:
        # Get the start time of the requested chunk
        assert self._events is not None, "run get_events_this_chunk first!"
        if chunk_i == 0:
            return self._events[chunk_i]["time"][0]
        return self._events[chunk_i - 1]["endtime"][-1]

    def get_chunk_end(self, chunk_i):
        assert self._events is not None, "run get_events_this_chunk first!"
        if chunk_i == self.n_chunks:
            return self.event_time_range[1]
        return self._events[chunk_i]["endtime"][-1]

    def compute(self, chunk_i):
        events = self.get_events_this_chunk(chunk_i)
        events_within = events["endtime"] < self.event_time_range[1]
        events = events[events_within]

        return self.chunk(
            start=self.chunk_start(chunk_i), end=self.get_chunk_end(chunk_i), data=events
        )


class TestAqmonProcessing(TestCase):
    def setUp(self) -> None:
        st = straxen.test_utils.nt_test_context().new_context()
        # I'm going to deregister all plugins, since I don't want to
        # get a thousand warnings that some config is not used, make
        # sure to mark all configs as "free options".
        st.set_context_config({"free_options": list(st.config.keys())})
        st._plugin_class_registry = {}

        st.set_config(dict(veto_proximity_window=10**99))
        self.TOTAL_DEADTIME: List = []
        self.TOTAL_SIGNALS: List = []

        class DeadTimedDummyAqHits(DummyAqmonHits):
            # Add mutible defaults to give results in the tests below
            TOTAL_DEADTIME = self.TOTAL_DEADTIME
            TOTAL_SIGNALS = self.TOTAL_SIGNALS

        class DummyVi(VetoIntervals):
            save_when = strax.SaveWhen.ALWAYS

        class DummyVp(VetoProximity):
            save_when = strax.SaveWhen.NEVER

        st.register(DeadTimedDummyAqHits)
        st.register(DummyVi)
        st.register(DummyVp)
        st.register(DummyEventBasics)
        self.st = st
        self.run_id = test_run_id_nT
        self.assertFalse(np.sum(self.TOTAL_DEADTIME))
        self.assertFalse(st.is_stored(self.run_id, "aqmon_hits"))
        self.assertFalse(st.is_stored(self.run_id, "veto_intervals"))

    def test_dummy_plugin_works(self):
        """The simplest plugin."""
        self.st.make(self.run_id, "aqmon_hits")
        self.assertGreater(np.sum(self.TOTAL_DEADTIME), 0)
        self.assertGreater(np.sum(self.TOTAL_SIGNALS), 0)
        events = self.st.get_array(self.run_id, "event_basics")
        self.assertTrue(len(events))

    def test_veto_intervals(self, options=None):
        if options is not None:
            self.st.set_config(options)
        veto_intervals = self.st.get_array(self.run_id, "veto_intervals")
        # We should have roughly 1/2 the number of ON/OFF signals of intervals
        # Roughly, since we might have patched an ON/OFF
        self.assertAlmostEqual(len(veto_intervals), sum(self.TOTAL_SIGNALS) / 2, delta=2)
        self.assertTrue(np.sum(self.TOTAL_DEADTIME))
        self.assertEqual(np.sum(veto_intervals["veto_interval"]), np.sum(self.TOTAL_DEADTIME))

    def test_veto_intervals_with_missing_on(self):
        self.test_veto_intervals(dict(start_with_channel_on=False))

    def test_make_veto_proximity(self):
        """I'm not going to do something fancy here, just checking if we can run the code."""
        veto_intervals = self.st.get_array(self.run_id, "veto_intervals")
        self.st.set_config(dict(event_time_range=[0, int(veto_intervals["endtime"][-1])]))
        self.st.make(self.run_id, "event_basics")
        for c in self.st.get_iter(self.run_id, "veto_proximity"):
            print(c)
        return self.st

    @skipIf(not straxen.test_utils.is_installed("cutax"), "cutax not installed")
    def test_cut_daq_reader(self):
        st = self.test_make_veto_proximity()
        import cutax

        st.register(cutax.cuts.DAQVeto)
        st.make(self.run_id, "cut_daq_veto")

    def tearDown(self) -> None:
        for sf in self.st.storage:
            if not sf.readonly:
                p = getattr(sf, "path")
                if os.path.exists(p):
                    shutil.rmtree(p)
