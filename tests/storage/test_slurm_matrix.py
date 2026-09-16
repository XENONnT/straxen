import json
import os
import tempfile
import unittest
import unittest.mock
from straxen.scripts.benchmark_slurm import main as cli_main
from straxen.storage.slurm import (
    BenchmarkMatrix,
    SlurmConfig,
    SlurmMatrixRunner,
)


class TestSlurmMatrix(unittest.TestCase):
    """Test SLURM Benchmark Matrix generation, execution, and reporting."""

    def setUp(self):
        self.matrix = BenchmarkMatrix(
            backends=["posix", "memory"],
            targets=["records"],
            workloads=["bulk_array", "streaming_iter"],
            workers=[1, 2],
            chunk_sizes_mb=[0.5],
            n_chunks=3,
            iterations=1,
            warmup=0,
        )
        self.slurm_cfg = SlurmConfig(
            partition="caslake",
            account="pi-lgrandi",
            qos="caslake",
            cpus_per_task=4,
            mem_per_cpu="4G",
            time="00:30:00",
            max_concurrent_jobs=10,
            env_setup="module load python",
        )
        self.runner = SlurmMatrixRunner(
            matrix=self.matrix,
            slurm_config=self.slurm_cfg,
        )

    def test_matrix_expansion(self):
        tasks = self.matrix.expand()
        # 2 backends * 1 target * 2 workloads * 2 workers * 1 chunk_size = 8
        self.assertEqual(len(tasks), 8)

        for i, t in enumerate(tasks):
            self.assertEqual(t["task_id"], i)
            self.assertIn(t["backend"], ["posix", "memory"])
            self.assertEqual(t["target"], "records")
            self.assertIn(t["workload"], ["bulk_array", "streaming_iter"])
            self.assertIn(t["workers"], [1, 2])
            self.assertEqual(t["chunk_size_mb"], 0.5)

    def test_sbatch_script_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.runner.generate_scripts(output_dir=tmp_dir)

            self.assertTrue(os.path.isfile(paths["manifest"]))
            self.assertTrue(os.path.isfile(paths["array_script"]))
            self.assertTrue(os.path.isfile(paths["local_script"]))
            self.assertEqual(paths["n_tasks"], "8")

            with open(paths["array_script"], "r") as f:
                content = f.read()

            self.assertIn("#SBATCH --job-name=strax_bench", content)
            self.assertIn("#SBATCH --account=pi-lgrandi", content)
            self.assertIn("#SBATCH --partition=caslake", content)
            self.assertIn("#SBATCH --qos=caslake", content)
            self.assertIn("#SBATCH --cpus-per-task=4", content)
            self.assertIn("#SBATCH --mem-per-cpu=4G", content)
            self.assertIn("#SBATCH --array=0-7%10", content)
            self.assertIn("module load python", content)
            self.assertIn("benchmark_slurm run-task", content)

    def test_singularity_prefix_generation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            s_img = "/project2/lgrandi/xenonnt/singularity-images/test.simg"
            cfg = SlurmConfig(singularity_image=s_img)
            runner = SlurmMatrixRunner(matrix=self.matrix, slurm_config=cfg)
            paths = runner.generate_scripts(output_dir=tmp_dir)

            with open(paths["array_script"], "r") as f:
                content = f.read()

            expected_prefix = (
                f"singularity exec -B /project:/project -B /project2:/project2 "
                f"-B /dali:/dali {s_img}"
            )
            self.assertIn(expected_prefix, content)

    def test_dry_run_submission(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = self.runner.submit(output_dir=tmp_dir, dry_run=True)
            self.assertEqual(res["status"], "dry_run")
            self.assertIn("sbatch", res["command"])
            self.assertEqual(res["n_tasks"], 8)

    def test_mock_sbatch_submission(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mock_res = unittest.mock.MagicMock()
            mock_res.stdout = "Submitted batch job 98765432\n"

            with unittest.mock.patch("subprocess.run", return_value=mock_res) as mock_run:
                res = self.runner.submit(output_dir=tmp_dir, dry_run=False)
                mock_run.assert_called_once()
                self.assertEqual(res["status"], "submitted")
                self.assertEqual(res["job_id"], "98765432")

    def test_execute_task(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            paths = self.runner.generate_scripts(output_dir=tmp_dir)
            out_file = SlurmMatrixRunner.execute_task(
                manifest_path=paths["manifest"],
                task_id=0,
                output_dir=paths["results_dir"],
            )
            self.assertTrue(os.path.isfile(out_file))

            with open(out_file, "r") as f:
                data = json.load(f)

            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["task_id"], 0)
            self.assertIn("throughput_mb_s", data[0])
            self.assertIn("wall_time_s", data[0])

    def test_results_aggregation_and_speedups(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            res_dir = os.path.join(tmp_dir, "results")
            os.makedirs(res_dir, exist_ok=True)

            r_posix = [
                {
                    "task_id": 0,
                    "storage_type": "posix",
                    "target": "records",
                    "workload": "bulk_array",
                    "workers": 1,
                    "chunk_size_mb": 2.0,
                    "throughput_mb_s": 500.0,
                    "record_rate_hz": 2000000.0,
                    "ttfc_ms": None,
                    "peak_rss_mb": 200.0,
                    "cpu_percent": 90.0,
                }
            ]
            r_xrootd = [
                {
                    "task_id": 1,
                    "storage_type": "xrootd",
                    "target": "records",
                    "workload": "bulk_array",
                    "workers": 1,
                    "chunk_size_mb": 2.0,
                    "throughput_mb_s": 1250.0,
                    "record_rate_hz": 5000000.0,
                    "ttfc_ms": None,
                    "peak_rss_mb": 210.0,
                    "cpu_percent": 95.0,
                }
            ]

            with open(os.path.join(res_dir, "result_0.json"), "w") as f:
                json.dump(r_posix, f)
            with open(os.path.join(res_dir, "result_1.json"), "w") as f:
                json.dump(r_xrootd, f)

            report = SlurmMatrixRunner.collect_results(results_dir=res_dir)
            self.assertEqual(len(report.results), 2)

            speedups = report.compute_speedups(baseline_backend="posix")
            self.assertEqual(len(speedups), 1)
            self.assertEqual(speedups[0]["test_backend"], "xrootd")
            self.assertAlmostEqual(speedups[0]["speedup_ratio"], 2.5)

            table = report.to_markdown_table()
            self.assertIn("Speedup Ratios vs POSIX Baseline", table)
            self.assertIn("2.50x", table)

            json_out = os.path.join(tmp_dir, "summary.json")
            report.to_json(json_out)
            self.assertTrue(os.path.isfile(json_out))

            plot_out = os.path.join(tmp_dir, "plot.png")
            report.plot_matrix(plot_out)
            self.assertTrue(os.path.isfile(plot_out))

    def test_cli_modes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 1. Generate CLI
            gen_ret = cli_main(
                [
                    "generate",
                    "--backends",
                    "posix",
                    "memory",
                    "--targets",
                    "records",
                    "--workloads",
                    "bulk_array",
                    "--workers",
                    "1",
                    "--output-dir",
                    tmp_dir,
                ]
            )
            self.assertEqual(gen_ret, 0)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "submit_array.sbatch")))

            # 2. Submit --dry-run CLI
            sub_ret = cli_main(
                [
                    "submit",
                    "--backends",
                    "posix",
                    "memory",
                    "--targets",
                    "records",
                    "--workloads",
                    "bulk_array",
                    "--workers",
                    "1",
                    "--output-dir",
                    tmp_dir,
                    "--dry-run",
                ]
            )
            self.assertEqual(sub_ret, 0)

            # 3. Run-task CLI
            manifest = os.path.join(tmp_dir, "matrix_manifest.json")
            results_dir = os.path.join(tmp_dir, "results")
            task_ret = cli_main(
                [
                    "run-task",
                    "--manifest",
                    manifest,
                    "--task-id",
                    "0",
                    "--output-dir",
                    results_dir,
                ]
            )
            self.assertEqual(task_ret, 0)
            self.assertTrue(os.path.isfile(os.path.join(results_dir, "result_0.json")))

            # 4. Collect CLI
            collect_ret = cli_main(
                [
                    "collect",
                    "--output-dir",
                    tmp_dir,
                    "--output",
                    os.path.join(tmp_dir, "collected.json"),
                    "--plot",
                    os.path.join(tmp_dir, "collected.png"),
                ]
            )
            self.assertEqual(collect_ret, 0)
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "collected.json")))
            self.assertTrue(os.path.isfile(os.path.join(tmp_dir, "collected.png")))


if __name__ == "__main__":
    unittest.main()
