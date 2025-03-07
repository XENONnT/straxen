# mypy: disable-error-code="has-type"

import os
import shutil
import unittest
import strax
import straxen
from straxen import download_test_data, get_resource
from straxen.plugins.raw_records.daqreader import (
    ArtificialDeadtimeInserted,
    DAQReader,
    ARTIFICIAL_DEADTIME_CHANNEL,
)
from datetime import timezone, datetime
from time import sleep
import numpy as np


class DummyDAQReader(DAQReader):
    """Dummy version of DAQReader with different provides and different lineage."""

    provides = (
        "raw_records",
        "raw_records_nv",
        "raw_records_aqmon",
    )
    dummy_version = strax.Config(
        default=None,
        track=True,
        help="Extra option to make sure that we are getting a different lineage if we want to",
    )
    data_kind = dict(zip(provides, provides))
    rechunk_on_save = False

    def _path(self, chunk_i):
        path = super()._path(chunk_i)
        path_exists = os.path.exists(path)
        print(f"looked for {chunk_i} on {path}. Is found = {path_exists}")
        return path


class TestDAQReader(unittest.TestCase):
    """
    Test DAQReader with a few chunks of amstrax data:
    https://github.com/XAMS-nikhef/amstrax


    This class is structured with three parts:
      - A. The test(s) where we execute some tests to make sure the
      DAQ-reader works well;
      - B. Setup and teardown logic which downloads/removes test data if
      we run this test so that we get a fresh sample of data every time
      we run this test;
      - C. Some utility functions for part A and B (like setting the
      context etc).
    """

    run_id = "999999"
    run_doc_name = "rundoc_999999.json"
    live_data_path = f"./live_data/{run_id}"
    rundoc_file = "https://raw.githubusercontent.com/XAMS-nikhef/amstrax_files/73681f112d748f6cd0e95045970dd29c44e983b0/data/rundoc_999999.json"  # noqa
    data_file = "https://raw.githubusercontent.com/XAMS-nikhef/amstrax_files/73681f112d748f6cd0e95045970dd29c44e983b0/data/999999.tar"  # noqa

    # # Part A. the actual tests
    def test_make(self) -> None:
        """Test if we can run the daq-reader without chrashing and if we actually stored the data
        after making it."""
        run_id = self.run_id
        for target, plugin_class in self.st._plugin_class_registry.items():
            self.st.make(run_id, target)
            sleep(0.5)  # allow os to rename the file
            if plugin_class.save_when >= strax.SaveWhen.TARGET:
                self.assertTrue(
                    self.st.is_stored(run_id, target),
                )

    @unittest.mock.patch.object(
        straxen.AqMonChannelOccupancy,
        "get_v1495_config",
        straxen.AqMonChannelOccupancy.get_fake_config,
    )
    def test_insert_deadtime(self):
        """In the DAQ reader, we need a mimimium quiet period to say where we can start/end a chunk.

        Test that this information gets propagated to the ARTIFICIAL_DEADTIME_CHANNEL (in raw-
        records-aqmon) if we set this value to an unrealistic value of 0.5 s.

        """
        st = self.st.new_context()
        st.set_config(
            {
                "safe_break_in_pulses": int(0.5e9),
                "dummy_version": "test_insert_deadtime",
            }
        )

        with self.assertWarns(ArtificialDeadtimeInserted):
            st.make(self.run_id, "raw_records")

        rr_aqmon = st.get_array(self.run_id, "raw_records_aqmon")
        self.assertTrue(len(rr_aqmon))

        st.register(straxen.AqmonHits)
        st.register(straxen.VetoIntervals)
        veto_intervals = st.get_array(self.run_id, "veto_intervals")
        assert np.sum(veto_intervals["veto_interval"]), "No artificial deadtime parsed!"

    def test_invalid_setting(self):
        """The safe break in pulses cannot be longer than the chunk size."""
        st = self.st.new_context()
        st.set_config(
            {
                "safe_break_in_pulses": int(3600e9),
                "dummy_version": "test_invalid_setting",
            }
        )
        with self.assertRaises(ValueError):
            st.make(self.run_id, "raw_records")

    # # Part B. data-download and cleanup
    @classmethod
    def setUpClass(cls) -> None:
        st = strax.Context()
        st.register(DummyDAQReader)
        st.storage = [strax.DataDirectory("./daq_test_data")]
        st.set_config({"daq_input_dir": cls.live_data_path})
        cls.st = st

    @classmethod
    def tearDownClass(cls) -> None:
        path_live = f"live_data/{cls.run_id}"
        if os.path.exists(path_live):
            shutil.rmtree(path_live)
            print(f"rm {path_live}")

    def setUp(self) -> None:
        if not os.path.exists(self.live_data_path):
            print(f"Fetch {self.live_data_path}")
            self.download_test_data()
        rd = self.get_metadata()
        st = self.set_context_config(self.st, rd)
        self.st = st
        self.assertFalse(self.st.is_stored(self.run_id, "raw_records"))

    def tearDown(self) -> None:
        data_path = self.st.storage[0].path
        if os.path.exists(data_path):
            shutil.rmtree(data_path)
            print(f"rm {data_path}")

    # # Part C. Some utility functions for A & B
    def download_test_data(self):
        download_test_data(self.data_file)
        self.assertTrue(os.path.exists(self.live_data_path))

    def get_metadata(self):
        md = get_resource(self.rundoc_file, fmt="json")
        # This is a flat dict but we need to have a datetime object,
        # since this is only a test, let's just replace it with a
        # placeholder
        md["start"] = datetime.now()
        return md

    @staticmethod
    def set_context_config(st, run_doc):
        """Update context with fields needed by the DAQ reader."""
        daq_config = run_doc["daq_config"]
        st.set_context_config(dict(forbid_creation_of=tuple()))
        st.set_config(
            {
                "channel_map": dict(
                    # (Minimum channel, maximum channel)
                    # Channels must be listed in a ascending order!
                    tpc=(0, 1),
                    nveto=(1, 2),
                    aqmon=(ARTIFICIAL_DEADTIME_CHANNEL, ARTIFICIAL_DEADTIME_CHANNEL + 1),
                )
            }
        )
        update_config = {
            "readout_threads": daq_config["processing_threads"],
            "record_length": daq_config["strax_fragment_payload_bytes"] // 2,
            "max_digitizer_sampling_time": 10,
            "run_start_time": run_doc["start"].replace(tzinfo=timezone.utc).timestamp(),
            "daq_chunk_duration": int(daq_config["strax_chunk_length"] * 1e9),
            "daq_overlap_chunk_duration": int(daq_config["strax_chunk_overlap"] * 1e9),
            "daq_compressor": daq_config.get("compressor", "lz4"),
        }
        print(f"set config to {update_config}")
        st.set_config(update_config)
        return st
