import json
import os
import tempfile
import unittest
import strax
from straxen.storage.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    IOBenchmarkHarness,
    SyntheticDataGenerator,
)


class TestBenchmarkHarness(unittest.TestCase):
    """Test suite for the straxen I/O Benchmarking Harness."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="straxen_bench_test_")
        self.run_id = "050000"

    def tearDown(self):
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_synthetic_data_generator(self):
        """Test synthesizing valid Strax datasets across multiple data types."""
        for target in ["records", "peaks", "events"]:
            dest = f"file://{self.temp_dir}/{target}_data"
            key, run_folder = SyntheticDataGenerator.create_dataset(
                destination_url=dest,
                target_type=target,
                run_id=self.run_id,
                n_chunks=3,
                chunk_size_mb=0.5,
                compressor="zstd",
            )
            self.assertEqual(key.data_type, target)
            self.assertEqual(key.run_id, self.run_id)

            prefix = f"{target}-{key.lineage_hash}"
            dataset_dir = os.path.join(self.temp_dir, f"{target}_data", str(key))
            md_path = os.path.join(dataset_dir, f"{prefix}-metadata.json")
            self.assertTrue(os.path.exists(md_path))

            with open(md_path, "r") as f:
                meta = json.load(f)
            self.assertEqual(len(meta["chunks"]), 3)
            self.assertEqual(meta["data_type"], target)

            # Load first chunk and verify array integrity
            chunk_file = os.path.join(dataset_dir, meta["chunks"][0]["filename"])
            self.assertTrue(os.path.exists(chunk_file))
            dtype, _ = SyntheticDataGenerator.get_dtype_and_generator(target)
            data = strax.load_file(chunk_file, dtype=dtype, compressor="zstd")
            self.assertGreater(len(data), 0)

    def test_harness_workloads_memory(self):
        """Test executing all workload types in memory."""
        config = BenchmarkConfig(
            targets=["records"],
            run_id=self.run_id,
            storage_types=["memory"],
            n_chunks=3,
            chunk_size_mb=0.5,
            compressor="zstd",
            workloads=["bulk_array", "streaming_iter", "time_slice", "metadata_probe"],
            workers=[1],
            iterations=2,
            warmup=1,
        )

        harness = IOBenchmarkHarness(config)
        report = harness.run()

        self.assertIsInstance(report, BenchmarkReport)
        self.assertEqual(len(report.results), 4)

        for res in report.results:
            self.assertEqual(res.storage_type, "memory")
            self.assertEqual(res.target, "records")
            self.assertGreater(res.wall_time_s, 0.0)
            self.assertGreater(res.throughput_mb_s, 0.0)
            self.assertGreater(res.record_rate_hz, 0.0)

            if res.workload == "streaming_iter":
                self.assertIsNotNone(res.ttfc_ms)
                self.assertGreater(res.ttfc_ms, 0.0)
                self.assertGreater(len(res.chunk_latencies_ms), 0)

    def test_harness_posix_and_file_comparison(self):
        """Test profiling and comparing POSIX and file-based fsspec storage."""
        config = BenchmarkConfig(
            targets=["records"],
            run_id=self.run_id,
            storage_types=["posix", "file"],
            n_chunks=2,
            chunk_size_mb=0.5,
            compressor="zstd",
            workloads=["bulk_array"],
            workers=[1],
            iterations=1,
            warmup=0,
        )

        harness = IOBenchmarkHarness(config)
        report = harness.run(temp_dir=self.temp_dir)

        self.assertEqual(len(report.results), 2)

        # Verify table formatting
        table_str = report.to_markdown_table()
        self.assertIn("posix", table_str)
        self.assertIn("file", table_str)
        self.assertIn("Throughput (MB/s)", table_str)

        # Verify JSON export
        json_path = os.path.join(self.temp_dir, "benchmark_report.json")
        report.to_json(json_path)
        self.assertTrue(os.path.exists(json_path))

        with open(json_path, "r") as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["target"], "records")

    def test_multi_threaded_loader(self):
        """Test multi-threaded chunk retrieval."""
        config = BenchmarkConfig(
            targets=["records"],
            run_id=self.run_id,
            storage_types=["memory"],
            n_chunks=4,
            chunk_size_mb=0.5,
            compressor="zstd",
            workloads=["bulk_array"],
            workers=[1, 2],
            iterations=1,
            warmup=0,
        )

        harness = IOBenchmarkHarness(config)
        report = harness.run()

        self.assertEqual(len(report.results), 2)
        self.assertEqual(report.results[0].workers, 1)
        self.assertEqual(report.results[1].workers, 2)

    def test_cli_execution(self):
        """Test CLI benchmark script."""
        from straxen.scripts.benchmark_io import main

        out_json = os.path.join(self.temp_dir, "cli_results.json")
        test_argv = [
            "--backends",
            "memory",
            "--targets",
            "records",
            "--n-chunks",
            "2",
            "--chunk-size-mb",
            "0.2",
            "--workloads",
            "bulk_array",
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--output",
            out_json,
        ]

        main(test_argv)
        self.assertTrue(os.path.exists(out_json))
