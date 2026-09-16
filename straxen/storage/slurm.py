"""SLURM Benchmark Matrix Runner for Midway and cluster environments.

Automates multi-dimensional storage I/O benchmarking across storage tiers
(POSIX, XRootD, memory, scratch), data targets, workloads, and worker counts.
"""

from dataclasses import dataclass, field
import itertools
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
import strax
from straxen.storage.benchmark import (
    BenchmarkConfig,
    BenchmarkReport,
    IOBenchmarkHarness,
)

try:
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

export, __all__ = strax.exporter()
__all__.extend(
    [
        "SlurmConfig",
        "BenchmarkMatrix",
        "MatrixReport",
        "SlurmMatrixRunner",
    ]
)

log = logging.getLogger("straxen.storage.slurm")


@export
@dataclass
class SlurmConfig:
    """SLURM scheduler configuration for Midway / cluster jobs."""

    partition: str = "caslake"
    account: str = "pi-lgrandi"
    qos: Optional[str] = None
    time: str = "01:00:00"
    cpus_per_task: int = 4
    mem_per_cpu: Optional[str] = "4G"
    mem: Optional[str] = None
    job_name: str = "strax_bench"
    output_dir: str = "./slurm_benchmarks"
    max_concurrent_jobs: int = 20
    env_setup: Optional[str] = None
    singularity_image: Optional[str] = None
    extra_sbatch_args: Dict[str, str] = field(default_factory=dict)


@export
@dataclass
class BenchmarkMatrix:
    """Definition of multi-dimensional storage benchmark parameter space."""

    backends: List[str] = field(default_factory=lambda: ["posix", "xrootd"])
    targets: List[str] = field(default_factory=lambda: ["records", "peaks"])
    workloads: List[str] = field(default_factory=lambda: ["bulk_array", "streaming_iter"])
    workers: List[int] = field(default_factory=lambda: [1, 2, 4])
    chunk_sizes_mb: List[float] = field(default_factory=lambda: [2.0])
    n_chunks: int = 5
    iterations: int = 3
    warmup: int = 1
    compressor: str = "zstd"
    run_id: str = "050000"
    xrootd_url: str = "root://midway-origin.uchicago.edu/"
    subpath: str = "xenon/xenonnt/processed"

    def expand(self) -> List[Dict[str, Any]]:
        """Generate Cartesian product of matrix permutations with unique task IDs."""
        tasks: List[Dict[str, Any]] = []
        task_id = 0
        for b, t, w, wrk, cs in itertools.product(
            self.backends,
            self.targets,
            self.workloads,
            self.workers,
            self.chunk_sizes_mb,
        ):
            tasks.append(
                {
                    "task_id": task_id,
                    "backend": b,
                    "target": t,
                    "workload": w,
                    "workers": wrk,
                    "chunk_size_mb": cs,
                    "n_chunks": self.n_chunks,
                    "iterations": self.iterations,
                    "warmup": self.warmup,
                    "compressor": self.compressor,
                    "run_id": self.run_id,
                    "xrootd_url": self.xrootd_url,
                    "subpath": self.subpath,
                }
            )
            task_id += 1
        return tasks


@export
class MatrixReport:
    """Aggregates and analyzes benchmark results across a SLURM matrix execution."""

    def __init__(
        self,
        results: List[Dict[str, Any]],
        manifest: Optional[List[Dict[str, Any]]] = None,
    ):
        self.results = results
        self.manifest = manifest or []

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to nested dictionary."""
        return {
            "total_results": len(self.results),
            "results": self.results,
            "manifest": self.manifest,
            "speedups": self.compute_speedups(),
        }

    def to_json(self, filepath: str) -> None:
        """Export merged matrix report to JSON file."""
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def compute_speedups(
        self, baseline_backend: str = "posix"
    ) -> List[Dict[str, Any]]:
        """Calculate throughput speedup ratios relative to a baseline storage tier."""
        speedups: List[Dict[str, Any]] = []

        lookup: Dict[Tuple[str, str, int, float], Dict[str, float]] = {}
        for r in self.results:
            key = (
                r["target"],
                r["workload"],
                r["workers"],
                r.get("chunk_size_mb", 2.0),
            )
            if key not in lookup:
                lookup[key] = {}
            lookup[key][r["storage_type"]] = r["throughput_mb_s"]

        for (tgt, wld, wrk, cs), st_dict in lookup.items():
            base_tp = st_dict.get(baseline_backend)
            if not base_tp or base_tp <= 0:
                continue

            for st, tp in st_dict.items():
                if st == baseline_backend:
                    continue
                ratio = tp / base_tp
                speedups.append(
                    {
                        "target": tgt,
                        "workload": wld,
                        "workers": wrk,
                        "chunk_size_mb": cs,
                        "baseline_backend": baseline_backend,
                        "test_backend": st,
                        "baseline_throughput_mb_s": base_tp,
                        "test_throughput_mb_s": tp,
                        "speedup_ratio": ratio,
                    }
                )
        return speedups

    def to_markdown_table(self) -> str:
        """Format results as a comprehensive Markdown table."""
        if not self.results:
            return "No matrix benchmark results found."

        headers = [
            "Storage",
            "Target",
            "Workload",
            "Workers",
            "Throughput (MB/s)",
            "Record Rate (k/s)",
            "TTFC (ms)",
            "Peak RSS (MB)",
            "CPU %",
        ]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        sorted_results = sorted(
            self.results,
            key=lambda r: (
                r["target"],
                r["workload"],
                r["workers"],
                r["storage_type"],
            ),
        )

        for r in sorted_results:
            ttfc_s = f"{r['ttfc_ms']:.1f}" if r.get("ttfc_ms") is not None else "-"
            rss_s = f"{r['peak_rss_mb']:.1f}" if r.get("peak_rss_mb") is not None else "-"
            cpu_s = f"{r['cpu_percent']:.1f}%" if r.get("cpu_percent") is not None else "-"
            rate_k = r.get("record_rate_hz", 0.0) / 1000.0

            row = [
                r["storage_type"],
                r["target"],
                r["workload"],
                str(r["workers"]),
                f"{r['throughput_mb_s']:.2f}",
                f"{rate_k:.1f}",
                ttfc_s,
                rss_s,
                cpu_s,
            ]
            lines.append("| " + " | ".join(row) + " |")

        speedups = self.compute_speedups()
        if speedups:
            lines.append("\n### Speedup Ratios vs POSIX Baseline\n")
            sp_headers = [
                "Target",
                "Workload",
                "Workers",
                "Comparison",
                "Throughput Ratio",
            ]
            lines.append("| " + " | ".join(sp_headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(sp_headers)) + " |")
            for sp in speedups:
                comp = f"{sp['test_backend']} vs {sp['baseline_backend']}"
                ratio_s = f"{sp['speedup_ratio']:.2f}x"
                lines.append(
                    f"| {sp['target']} | {sp['workload']} | {sp['workers']} | {comp} | {ratio_s} |"
                )

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print markdown summary to standard out."""
        print("\n" + "=" * 90)
        print("                     SLURM BENCHMARK MATRIX REPORT")
        print("=" * 90)
        print(self.to_markdown_table())
        print("=" * 90 + "\n")

    def plot_matrix(self, output_path: str) -> None:
        """Generate multi-panel comparison chart of matrix throughput."""
        if not HAVE_MATPLOTLIB:
            log.warning("Matplotlib not available; skipping matrix plot.")
            return

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        targets = sorted(list({r["target"] for r in self.results}))
        if not targets:
            return

        fig, axes = plt.subplots(
            nrows=len(targets),
            ncols=1,
            figsize=(10, 4 * len(targets)),
            squeeze=False,
            dpi=150,
        )

        for i, target in enumerate(targets):
            ax = axes[i, 0]
            subset = [r for r in self.results if r["target"] == target]

            series: Dict[str, Dict[int, float]] = {}
            for r in subset:
                s_key = f"{r['storage_type']} ({r['workload']})"
                if s_key not in series:
                    series[s_key] = {}
                series[s_key][r["workers"]] = r["throughput_mb_s"]

            for s_key, data_pts in series.items():
                workers_sorted = sorted(data_pts.keys())
                vals = [data_pts[w] for w in workers_sorted]
                ax.plot(
                    workers_sorted,
                    vals,
                    marker="o",
                    linewidth=2,
                    label=s_key,
                )

            ax.set_title(f"Target: {target} Throughput vs Worker Threads")
            ax.set_xlabel("Workers (Threads)")
            ax.set_ylabel("Throughput (MB/s)")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()

        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        log.info(f"Saved matrix plot to {output_path}")


@export
class SlurmMatrixRunner:
    """Orchestrates generation, submission, execution, and harvesting of SLURM benchmarks."""

    def __init__(
        self,
        matrix: Optional[BenchmarkMatrix] = None,
        slurm_config: Optional[SlurmConfig] = None,
    ):
        self.matrix = matrix or BenchmarkMatrix()
        self.slurm_config = slurm_config or SlurmConfig()

    def generate_scripts(
        self, output_dir: Optional[str] = None
    ) -> Dict[str, str]:
        """Generate manifest, job array script, and helper submission scripts.

        :param output_dir: Destination directory for generated artifacts.
        :return: Dictionary containing paths to generated files.
        """
        out_dir = os.path.abspath(output_dir or self.slurm_config.output_dir)
        logs_dir = os.path.join(out_dir, "logs")
        results_dir = os.path.join(out_dir, "results")
        os.makedirs(logs_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        tasks = self.matrix.expand()
        manifest_path = os.path.join(out_dir, "matrix_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(tasks, f, indent=2)

        n_tasks = len(tasks)
        if n_tasks == 0:
            raise ValueError("Benchmark matrix produced 0 tasks.")

        sc = self.slurm_config
        qos_line = f"#SBATCH --qos={sc.qos}\n" if sc.qos else ""
        mem_line = f"#SBATCH --mem={sc.mem}\n" if sc.mem else (
            f"#SBATCH --mem-per-cpu={sc.mem_per_cpu}\n" if sc.mem_per_cpu else ""
        )

        extra_lines = ""
        for k, v in sc.extra_sbatch_args.items():
            extra_lines += f"#SBATCH --{k}={v}\n"

        env_setup_block = sc.env_setup or "# Default environment"

        singularity_prefix = ""
        if sc.singularity_image:
            singularity_prefix = (
                f"singularity exec -B /project:/project -B /project2:/project2 "
                f"-B /dali:/dali {sc.singularity_image} "
            )

        max_concurrent = sc.max_concurrent_jobs or 20

        array_script_content = f"""#!/bin/bash
#SBATCH --job-name={sc.job_name}
#SBATCH --account={sc.account}
#SBATCH --partition={sc.partition}
{qos_line}#SBATCH --time={sc.time}
#SBATCH --cpus-per-task={sc.cpus_per_task}
{mem_line}#SBATCH --output={logs_dir}/job_%A_%a.out
#SBATCH --error={logs_dir}/job_%A_%a.err
#SBATCH --array=0-{n_tasks - 1}%{max_concurrent}
{extra_lines}
set -e

echo "=== Straxen SLURM Benchmark Task ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"

{env_setup_block}

TASK_ID="${{SLURM_ARRAY_TASK_ID}}"
MANIFEST="{manifest_path}"
RESULTS="{results_dir}"

{singularity_prefix}python -m straxen.scripts.benchmark_slurm run-task \\
    --manifest "$MANIFEST" \\
    --task-id "$TASK_ID" \\
    --output-dir "$RESULTS"

echo "Task $TASK_ID completed successfully at $(date)"
"""
        array_script_path = os.path.join(out_dir, "submit_array.sbatch")
        with open(array_script_path, "w") as f:
            f.write(array_script_content)
        os.chmod(array_script_path, 0o755)

        local_runner_content = f"""#!/bin/bash
set -e
MANIFEST="{manifest_path}"
RESULTS="{results_dir}"

echo "Running all {n_tasks} matrix tasks sequentially..."
for ((i=0; i<{n_tasks}; i++)); do
    echo "Running task $i / {n_tasks}..."
    python -m straxen.scripts.benchmark_slurm run-task \\
        --manifest "$MANIFEST" \\
        --task-id "$i" \\
        --output-dir "$RESULTS"
done
echo "All tasks complete. Collecting results..."
python -m straxen.scripts.benchmark_slurm collect --results-dir "$RESULTS"
"""
        local_script_path = os.path.join(out_dir, "run_local.sh")
        with open(local_script_path, "w") as f:
            f.write(local_runner_content)
        os.chmod(local_script_path, 0o755)

        return {
            "manifest": manifest_path,
            "array_script": array_script_path,
            "local_script": local_script_path,
            "output_dir": out_dir,
            "results_dir": results_dir,
            "logs_dir": logs_dir,
            "n_tasks": str(n_tasks),
        }

    def submit(
        self, output_dir: Optional[str] = None, dry_run: bool = True
    ) -> Dict[str, Any]:
        """Generate scripts and submit to SLURM (or simulate via dry_run).

        :param output_dir: Target output directory.
        :param dry_run: If True, generate scripts without invoking sbatch.
        :return: Dictionary with submission metadata and Job ID.
        """
        paths = self.generate_scripts(output_dir)
        cmd = ["sbatch", paths["array_script"]]

        if dry_run:
            log.info(f"Dry run: generated scripts in {paths['output_dir']}.")
            return {
                "status": "dry_run",
                "command": " ".join(cmd),
                "paths": paths,
                "n_tasks": int(paths["n_tasks"]),
            }

        try:
            res = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            output = res.stdout.strip()
            job_id = output.split()[-1] if output else "unknown"
            log.info(f"Submitted SLURM array job {job_id}")
            return {
                "status": "submitted",
                "job_id": job_id,
                "command": " ".join(cmd),
                "output": output,
                "paths": paths,
                "n_tasks": int(paths["n_tasks"]),
            }
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            log.error(f"Failed to submit SLURM job: {e}")
            return {
                "status": "error",
                "error": str(e),
                "command": " ".join(cmd),
                "paths": paths,
            }

    @staticmethod
    def execute_task(
        manifest_path: str, task_id: int, output_dir: str
    ) -> str:
        """Execute a single task definition from the manifest.

        :param manifest_path: Path to matrix_manifest.json.
        :param task_id: Zero-indexed task ID to execute.
        :param output_dir: Output directory to write result_<task_id>.json.
        :return: Path to generated result file.
        """
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        task = next((t for t in manifest if t["task_id"] == task_id), None)
        if task is None:
            raise ValueError(f"Task ID {task_id} not found in {manifest_path}")

        cfg = BenchmarkConfig(
            targets=[task["target"]],
            run_id=task["run_id"],
            storage_types=[task["backend"]],
            n_chunks=task["n_chunks"],
            chunk_size_mb=task["chunk_size_mb"],
            compressor=task["compressor"],
            workloads=[task["workload"]],
            workers=[task["workers"]],
            iterations=task["iterations"],
            warmup=task["warmup"],
            xrootd_url=task["xrootd_url"],
            subpath=task["subpath"],
        )

        harness = IOBenchmarkHarness(cfg)
        report: BenchmarkReport = harness.run()
        results_data = report.to_dict()

        for r in results_data:
            r["task_id"] = task_id
            r["chunk_size_mb"] = task["chunk_size_mb"]

        os.makedirs(output_dir, exist_ok=True)
        out_file = os.path.join(output_dir, f"result_{task_id}.json")
        with open(out_file, "w") as f:
            json.dump(results_data, f, indent=2)

        return out_file

    @staticmethod
    def collect_results(
        results_dir: str, manifest_path: Optional[str] = None
    ) -> MatrixReport:
        """Scan directory for result_*.json and aggregate into MatrixReport."""
        results: List[Dict[str, Any]] = []

        if not os.path.isdir(results_dir):
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        for fname in sorted(os.listdir(results_dir)):
            if fname.startswith("result_") and fname.endswith(".json"):
                full_path = os.path.join(results_dir, fname)
                try:
                    with open(full_path, "r") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            results.extend(data)
                        elif isinstance(data, dict):
                            results.append(data)
                except (json.JSONDecodeError, OSError) as e:
                    log.warning(f"Could not read {full_path}: {e}")

        manifest_data = []
        if manifest_path and os.path.isfile(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    manifest_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                pass

        return MatrixReport(results=results, manifest=manifest_data)
