"""I/O Benchmarking Harness for straxen.

Measures, profiles, and compares data access performance across storage tiers (POSIX disk, XRootD
streaming, memory, and local fsspec).

"""

from dataclasses import asdict, dataclass, field
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import strax
import straxen

try:
    import psutil

    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

try:
    import matplotlib.pyplot as plt

    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

export, __all__ = strax.exporter()
__all__.extend(
    [
        "BenchmarkConfig",
        "BenchmarkResult",
        "BenchmarkReport",
        "SyntheticDataGenerator",
        "IOBenchmarkHarness",
    ]
)

log = logging.getLogger("straxen.benchmark")


def get_synthetic_plugin_name(target_type: str) -> str:
    """Return uniform class name for synthetic plugin across generator and harness."""
    return f"{target_type.capitalize()}SyntheticPlugin"


@dataclass
class BenchmarkConfig:
    """Configuration for an I/O benchmark run."""

    targets: List[str] = field(default_factory=lambda: ["records"])
    run_id: str = "050000"
    storage_types: List[str] = field(default_factory=lambda: ["posix", "memory"])
    n_chunks: int = 5
    chunk_size_mb: float = 2.0
    compressor: str = "zstd"
    workloads: List[str] = field(default_factory=lambda: ["bulk_array", "streaming_iter"])
    workers: List[int] = field(default_factory=lambda: [1])
    iterations: int = 3
    warmup: int = 1
    xrootd_url: str = "root://midway-origin.uchicago.edu/"
    subpath: str = "xenon/xenonnt/processed"
    time_slice_fraction: float = 0.2


@dataclass
class BenchmarkResult:
    """Telemetry and performance metrics for a single benchmark workload."""

    storage_type: str
    target: str
    workload: str
    workers: int
    n_chunks: int
    total_records: int
    uncompressed_mb: float
    wall_time_s: float
    throughput_mb_s: float
    record_rate_hz: float
    ttfc_ms: Optional[float] = None
    chunk_latencies_ms: List[float] = field(default_factory=list)
    latency_mean_ms: Optional[float] = None
    latency_median_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_min_ms: Optional[float] = None
    latency_max_ms: Optional[float] = None
    peak_rss_mb: Optional[float] = None
    delta_rss_mb: Optional[float] = None
    cpu_user_s: Optional[float] = None
    cpu_sys_s: Optional[float] = None
    cpu_percent: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SyntheticDataGenerator:
    """Generates synthetic Strax datasets for isolated benchmarking."""

    @staticmethod
    def get_dtype_and_generator(target_type: str):
        """Return (numpy dtype, item generator function) for standard targets."""
        if target_type == "raw_records":
            dtype = np.dtype(strax.raw_record_dtype())

            def fill_fn(data, n, start_time):
                data["time"] = start_time + np.arange(n) * 10
                data["length"] = 110
                data["dt"] = 2
                data["channel"] = np.random.randint(0, 494, size=n)

        elif target_type == "records":
            dtype = np.dtype(strax.record_dtype())

            def fill_fn(data, n, start_time):
                data["time"] = start_time + np.arange(n) * 10
                data["length"] = 110
                data["dt"] = 2
                data["channel"] = np.random.randint(0, 494, size=n)

        elif target_type == "peaks":
            dtype = np.dtype(strax.peak_dtype())

            def fill_fn(data, n, start_time):
                data["time"] = start_time + np.arange(n) * 100
                data["length"] = 10
                data["dt"] = 2
                data["area"] = np.random.uniform(10, 1000, size=n)

        elif target_type in ("events", "event_info"):
            fields = strax.time_fields + [
                ("s1_area", np.float32),
                ("s2_area", np.float32),
                ("cs1", np.float32),
                ("cs2", np.float32),
                ("x", np.float32),
                ("y", np.float32),
                ("z", np.float32),
            ]
            dtype = np.dtype(fields)

            def fill_fn(data, n, start_time):
                data["time"] = start_time + np.arange(n) * 1000
                data["endtime"] = data["time"] + 100
                data["s1_area"] = np.random.uniform(5, 50, size=n)
                data["s2_area"] = np.random.uniform(500, 5000, size=n)
                data["x"] = np.random.uniform(-60, 60, size=n)
                data["y"] = np.random.uniform(-60, 60, size=n)
                data["z"] = np.random.uniform(-140, 0, size=n)

        else:
            fields = strax.time_fields + [("data", np.float32)]
            dtype = np.dtype(fields)

            def fill_fn(data, n, start_time):
                data["time"] = start_time + np.arange(n) * 100
                data["endtime"] = data["time"] + 10

        return dtype, fill_fn

    @classmethod
    def create_dataset(
        cls,
        destination_url: str,
        target_type: str = "records",
        run_id: str = "050000",
        n_chunks: int = 5,
        chunk_size_mb: float = 2.0,
        compressor: str = "zstd",
        rucio_mode: bool = False,
        scope_prefix: str = "xnt_",
    ) -> Tuple[strax.DataKey, str]:
        """Generate and write a synthetic Strax dataset to destination URL/path."""
        import fsspec
        from .rucio_deterministic import (
            chunk_to_rucio_did,
            key_to_rucio_dids,
            rucio_deterministic_path,
        )

        dtype, fill_fn = cls.get_dtype_and_generator(target_type)
        itemsize = max(1, dtype.itemsize)
        target_bytes = int(chunk_size_mb * 1024 * 1024)
        n_items_per_chunk = max(1, target_bytes // itemsize)

        plugin_name = get_synthetic_plugin_name(target_type)
        lineage: Dict[str, Any] = {target_type: (plugin_name, "0.0.0", {})}
        key = strax.DataKey(run_id=run_id, data_type=target_type, lineage=lineage)

        fs, base_path = fsspec.core.url_to_fs(destination_url)
        clean_base = base_path.rstrip("/")

        if rucio_mode:
            dataset_did, metadata_did = key_to_rucio_dids(key, scope_prefix=scope_prefix)
            scope = dataset_did.split(":", 1)[0]
            target_key_str = f"{clean_base}/{dataset_did}"
        else:
            run_folder = f"{clean_base}/{key}"
            fs.makedirs(run_folder, exist_ok=True)
            target_key_str = run_folder

        chunks_meta = []
        current_time = 0
        for i in range(n_chunks):
            chunk_fn = f"{target_type}-{key.lineage_hash}-{i:06d}"
            if rucio_mode:
                chunk_did = chunk_to_rucio_did(scope, chunk_fn)
                rel_chunk_path = rucio_deterministic_path(chunk_did)
                chunk_file_path = f"{clean_base}/{rel_chunk_path}"
                parent_dir = chunk_file_path.rsplit("/", 1)[0]
                fs.makedirs(parent_dir, exist_ok=True)
            else:
                chunk_file_path = f"{run_folder}/{chunk_fn}"

            chunk_data = np.zeros(n_items_per_chunk, dtype=dtype)
            fill_fn(chunk_data, n_items_per_chunk, current_time)

            with fs.open(chunk_file_path, mode="wb") as f:
                strax.save_file(f, chunk_data, compressor=compressor)

            chunk_start = int(chunk_data["time"][0])
            chunk_end = int(strax.endtime(chunk_data)[-1])
            current_time = chunk_end

            chunk_meta = {
                "filename": chunk_fn,
                "n": n_items_per_chunk,
                "start": chunk_start,
                "end": chunk_end,
                "run_id": run_id,
            }
            chunks_meta.append(chunk_meta)

        metadata = {
            "strax_version": strax.__version__,
            "run_id": run_id,
            "data_type": target_type,
            "data_kind": target_type,
            "dtype": dtype.descr.__repr__(),
            "compressor": compressor,
            "lineage_hash": key.lineage_hash,
            "lineage": {target_type: [plugin_name, "0.0.0", {}]},
            "writing_ended": time.time(),
            "chunks": chunks_meta,
        }

        if rucio_mode:
            rel_md_path = rucio_deterministic_path(metadata_did)
            md_path = f"{clean_base}/{rel_md_path}"
            parent_dir = md_path.rsplit("/", 1)[0]
            fs.makedirs(parent_dir, exist_ok=True)
        else:
            prefix = f"{target_type}-{key.lineage_hash}"
            md_fn = strax.RUN_METADATA_PATTERN % prefix
            md_path = f"{run_folder}/{md_fn}"

        with fs.open(md_path, mode="wb") as f:
            f.write(json.dumps(metadata, indent=2).encode("utf-8"))

        return key, target_key_str


class BenchmarkReport:
    """Compiles, compares, and exports benchmark metrics."""

    def __init__(self, results: List[BenchmarkResult]):
        self.results = results

    def to_dict(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.results]

    def to_json(self, filepath: str) -> None:
        """Export results to JSON."""
        parent = os.path.dirname(filepath)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_markdown_table(self) -> str:
        """Generate a formatted markdown comparison table."""
        if not self.results:
            return "No benchmark results recorded."

        headers = [
            "Storage",
            "Target",
            "Workload",
            "Workers",
            "Wall Time (s)",
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

        for r in self.results:
            ttfc_str = f"{r.ttfc_ms:.1f}" if r.ttfc_ms is not None else "-"
            rss_str = f"{r.peak_rss_mb:.1f}" if r.peak_rss_mb is not None else "-"
            cpu_str = f"{r.cpu_percent:.1f}%" if r.cpu_percent is not None else "-"
            rate_k = r.record_rate_hz / 1000.0

            row = [
                r.storage_type,
                r.target,
                r.workload,
                str(r.workers),
                f"{r.wall_time_s:.3f}",
                f"{r.throughput_mb_s:.2f}",
                f"{rate_k:.1f}",
                ttfc_str,
                rss_str,
                cpu_str,
            ]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print markdown table to standard output."""
        print("\n" + "=" * 95)
        print("                        STRAXEN I/O BENCHMARK RESULTS")
        print("=" * 95)
        print(self.to_markdown_table())
        print("=" * 95 + "\n")

    def plot_comparison(self, output_path: str) -> None:
        """Generate a comparison chart if matplotlib is available."""
        if not HAVE_MATPLOTLIB:
            log.warning("Matplotlib not available; skipping plot generation.")
            return

        grouped: Dict[str, Dict[str, float]] = {}
        for r in self.results:
            key = f"{r.target}:{r.workload}:w{r.workers}"
            if key not in grouped:
                grouped[key] = {}
            grouped[key][r.storage_type] = r.throughput_mb_s

        labels = list(grouped.keys())
        storage_types = sorted({s for g in grouped.values() for s in g.keys()})

        x = np.arange(len(labels))
        width = 0.8 / len(storage_types) if storage_types else 0.8

        fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

        for i, stype in enumerate(storage_types):
            values = [grouped[lbl].get(stype, 0.0) for lbl in labels]
            ax.bar(x + i * width, values, width, label=stype)

        ax.set_ylabel("Throughput (MB/s)")
        ax.set_title("Straxen Storage I/O Throughput Benchmark")
        ax.set_xticks(x + width * (len(storage_types) - 1) / 2)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        log.info(f"Saved benchmark plot to {output_path}")


@export
class IOBenchmarkHarness:
    """Orchestrates I/O benchmarking across storage backends and workloads."""

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.process = psutil.Process() if HAVE_PSUTIL else None

    def _sample_rss_mb(self) -> float:
        if self.process:
            return self.process.memory_info().rss / (1024 * 1024)
        return 0.0

    def _sample_cpu_times(self):
        if self.process:
            return self.process.cpu_times()
        return None

    def _build_synthetic_plugin(self, target_type: str, dtype):
        """Construct dynamic dummy plugin with matching name and version."""
        plugin_name = get_synthetic_plugin_name(target_type)

        class DynamicBenchmarkPlugin(strax.Plugin):
            provides = target_type
            depends_on: tuple = tuple()
            data_kind = target_type
            __version__ = "0.0.0"

            def infer_dtype(self):
                return dtype

        DynamicBenchmarkPlugin.__name__ = plugin_name
        return DynamicBenchmarkPlugin

    def _setup_context_and_data(
        self,
        storage_type: str,
        target_type: str,
        temp_dir: str,
    ) -> Tuple[strax.Context, int, float]:
        """Configure storage frontend and create synthetic dataset if needed."""
        dtype, _ = SyntheticDataGenerator.get_dtype_and_generator(target_type)
        plugin = self._build_synthetic_plugin(target_type, dtype)

        if storage_type == "posix":
            posix_path = os.path.join(temp_dir, "posix_data")
            _, _ = SyntheticDataGenerator.create_dataset(
                destination_url=posix_path,
                target_type=target_type,
                run_id=self.config.run_id,
                n_chunks=self.config.n_chunks,
                chunk_size_mb=self.config.chunk_size_mb,
                compressor=self.config.compressor,
            )
            frontend = strax.DataDirectory(posix_path, readonly=True)

        elif storage_type == "xrootd":
            xrootd_dest = f"file://{temp_dir}/xrootd_data"
            _, _ = SyntheticDataGenerator.create_dataset(
                destination_url=xrootd_dest,
                target_type=target_type,
                run_id=self.config.run_id,
                n_chunks=self.config.n_chunks,
                chunk_size_mb=self.config.chunk_size_mb,
                compressor=self.config.compressor,
            )
            frontend = straxen.XRootDFrontend(
                redirector_url=f"file://{temp_dir}/xrootd_data",
                subpath="",
            )

        elif storage_type == "memory":
            mem_url = f"memory://bench_{target_type}"
            _, _ = SyntheticDataGenerator.create_dataset(
                destination_url=mem_url,
                target_type=target_type,
                run_id=self.config.run_id,
                n_chunks=self.config.n_chunks,
                chunk_size_mb=self.config.chunk_size_mb,
                compressor=self.config.compressor,
            )
            frontend = straxen.XRootDFrontend(
                redirector_url=mem_url,
                subpath="",
            )

        elif storage_type == "file":
            file_dest = f"file://{temp_dir}/fsspec_file_data"
            _, _ = SyntheticDataGenerator.create_dataset(
                destination_url=file_dest,
                target_type=target_type,
                run_id=self.config.run_id,
                n_chunks=self.config.n_chunks,
                chunk_size_mb=self.config.chunk_size_mb,
                compressor=self.config.compressor,
            )
            frontend = straxen.XRootDFrontend(
                redirector_url=file_dest,
                subpath="",
            )
        else:
            raise ValueError(f"Unknown storage_type {storage_type}")

        st = strax.Context(storage=[frontend])
        st.register(plugin)

        total_records = int(
            self.config.n_chunks
            * max(1, int((self.config.chunk_size_mb * 1024 * 1024) / dtype.itemsize))
        )
        uncompressed_mb = (total_records * dtype.itemsize) / (1024 * 1024)

        return st, total_records, uncompressed_mb

    def run_workload(
        self,
        st: strax.Context,
        target: str,
        workload: str,
        workers: int,
        total_records: int,
        uncompressed_mb: float,
        storage_type: str,
    ) -> BenchmarkResult:
        """Execute a specific workload pattern under sampling."""
        # Warmup passes
        for _ in range(self.config.warmup):
            if workload == "bulk_array":
                _ = st.get_array(self.config.run_id, target, max_workers=workers)
            elif workload == "streaming_iter":
                for _ in st.get_iter(self.config.run_id, target, max_workers=workers):
                    pass
            elif workload == "metadata_probe":
                _ = st.is_stored(self.config.run_id, target)

        wall_times = []
        ttfc_list = []
        all_chunk_latencies = []

        rss_before = self._sample_rss_mb()
        peak_rss = rss_before
        cpu_before = self._sample_cpu_times()

        for _ in range(self.config.iterations):
            t0 = time.perf_counter()
            ttfc = None

            if workload == "bulk_array":
                arr = st.get_array(self.config.run_id, target, max_workers=workers)
                t1 = time.perf_counter()
                rec_count = len(arr)

            elif workload == "streaming_iter":
                rec_count = 0
                t_prev = t0
                for chunk in st.get_iter(self.config.run_id, target, max_workers=workers):
                    now = time.perf_counter()
                    if ttfc is None:
                        ttfc = (now - t0) * 1000.0
                    all_chunk_latencies.append((now - t_prev) * 1000.0)
                    t_prev = now
                    rec_count += len(chunk)
                t1 = time.perf_counter()

            elif workload == "time_slice":
                meta = st.get_metadata(self.config.run_id, target)
                chunks = meta.get("chunks", [])
                if chunks:
                    full_span = chunks[-1]["end"] - chunks[0]["start"]
                    slice_span = int(full_span * self.config.time_slice_fraction)
                    t_start = chunks[0]["start"]
                    t_end = int(t_start + max(1, slice_span))
                    arr = st.get_array(
                        self.config.run_id,
                        target,
                        time_range=(t_start, t_end),
                        max_workers=workers,
                    )
                    rec_count = len(arr)
                else:
                    rec_count = 0
                t1 = time.perf_counter()

            elif workload == "metadata_probe":
                _ = st.is_stored(self.config.run_id, target)
                t1 = time.perf_counter()
                rec_count = total_records

            else:
                raise ValueError(f"Unknown workload {workload}")

            wall_times.append(t1 - t0)
            if ttfc is not None:
                ttfc_list.append(ttfc)

            current_rss = self._sample_rss_mb()
            if current_rss > peak_rss:
                peak_rss = current_rss

        cpu_after = self._sample_cpu_times()
        rss_after = self._sample_rss_mb()

        mean_wall = float(np.mean(wall_times)) if wall_times else 0.0
        throughput = (uncompressed_mb / mean_wall) if mean_wall > 0 else 0.0
        rate = (rec_count / mean_wall) if mean_wall > 0 else 0.0

        mean_ttfc = float(np.mean(ttfc_list)) if ttfc_list else None

        lat_mean = float(np.mean(all_chunk_latencies)) if all_chunk_latencies else None
        lat_median = float(np.median(all_chunk_latencies)) if all_chunk_latencies else None
        lat_p95 = float(np.percentile(all_chunk_latencies, 95)) if all_chunk_latencies else None
        lat_min = float(np.min(all_chunk_latencies)) if all_chunk_latencies else None
        lat_max = float(np.max(all_chunk_latencies)) if all_chunk_latencies else None

        cpu_user = (cpu_after.user - cpu_before.user) if (cpu_before and cpu_after) else None
        cpu_sys = (cpu_after.system - cpu_before.system) if (cpu_before and cpu_after) else None
        cpu_pct = (
            (((cpu_user + cpu_sys) / (mean_wall * self.config.iterations)) * 100.0)
            if (cpu_user is not None and cpu_sys is not None and mean_wall > 0)
            else None
        )

        return BenchmarkResult(
            storage_type=storage_type,
            target=target,
            workload=workload,
            workers=workers,
            n_chunks=self.config.n_chunks,
            total_records=rec_count,
            uncompressed_mb=uncompressed_mb,
            wall_time_s=mean_wall,
            throughput_mb_s=throughput,
            record_rate_hz=rate,
            ttfc_ms=mean_ttfc,
            chunk_latencies_ms=all_chunk_latencies,
            latency_mean_ms=lat_mean,
            latency_median_ms=lat_median,
            latency_p95_ms=lat_p95,
            latency_min_ms=lat_min,
            latency_max_ms=lat_max,
            peak_rss_mb=peak_rss,
            delta_rss_mb=(rss_after - rss_before),
            cpu_user_s=cpu_user,
            cpu_sys_s=cpu_sys,
            cpu_percent=cpu_pct,
        )

    def run(self, temp_dir: Optional[str] = None) -> BenchmarkReport:
        """Execute the configured benchmark suite and return a BenchmarkReport."""
        import tempfile

        cleanup_temp = False
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="straxen_bench_")
            cleanup_temp = True

        results = []
        try:
            for stype in self.config.storage_types:
                for target in self.config.targets:
                    st, total_records, uncompressed_mb = self._setup_context_and_data(
                        storage_type=stype,
                        target_type=target,
                        temp_dir=temp_dir,
                    )
                    for workload in self.config.workloads:
                        for workers in self.config.workers:
                            log.info(
                                f"Benchmarking: storage={stype}, target={target}, "
                                f"workload={workload}, workers={workers}"
                            )
                            result = self.run_workload(
                                st=st,
                                target=target,
                                workload=workload,
                                workers=workers,
                                total_records=total_records,
                                uncompressed_mb=uncompressed_mb,
                                storage_type=stype,
                            )
                            results.append(result)
        finally:
            if cleanup_temp and os.path.exists(temp_dir):
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

        return BenchmarkReport(results)
