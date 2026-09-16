"""Command-line interface for the straxen I/O Benchmarking Harness.

Usage:
    straxen_benchmark_io --backends posix memory --targets records --n-chunks 5
    straxen_benchmark_io --backends posix xrootd --workers 1 2 --output res.json
"""

import argparse
import sys
from straxen.storage.benchmark import (
    BenchmarkConfig,
    IOBenchmarkHarness,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Benchmark Straxen I/O across storage tiers (POSIX, XRootD, memory, file)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["posix", "memory"],
        choices=["posix", "xrootd", "memory", "file"],
        help="Storage tiers to profile and compare",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["records"],
        choices=["records", "peaks", "events", "event_info", "raw_records"],
        help="Data types to benchmark",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=["bulk_array", "streaming_iter"],
        choices=["bulk_array", "streaming_iter", "time_slice", "metadata_probe"],
        help="Access workloads to benchmark",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="050000",
        help="Target run ID",
    )
    parser.add_argument(
        "--n-chunks",
        type=int,
        default=5,
        help="Number of chunks per dataset",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=float,
        default=2.0,
        help="Target uncompressed chunk size in megabytes",
    )
    parser.add_argument(
        "--compressor",
        type=str,
        default="zstd",
        choices=["zstd", "blosc", "lz4", "bz2"],
        help="Compression algorithm for synthesized chunks",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1],
        help="Worker thread counts for multi-threaded loader",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Measurement repetitions per workload",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations before sampling",
    )
    parser.add_argument(
        "--xrootd-url",
        type=str,
        default="root://midway-origin.uchicago.edu/",
        help="XRootD redirector base URL when profiling remote tier",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save benchmark results as JSON",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save comparison chart image (.png)",
    )
    return parser.parse_args(args)


def main(argv=None):
    args = parse_args(argv)

    config = BenchmarkConfig(
        targets=args.targets,
        run_id=args.run_id,
        storage_types=args.backends,
        n_chunks=args.n_chunks,
        chunk_size_mb=args.chunk_size_mb,
        compressor=args.compressor,
        workloads=args.workloads,
        workers=args.workers,
        iterations=args.iterations,
        warmup=args.warmup,
        xrootd_url=args.xrootd_url,
    )

    harness = IOBenchmarkHarness(config)
    report = harness.run()
    report.print_summary()

    if args.output:
        report.to_json(args.output)
        print(f"Results exported to {args.output}")

    if args.plot:
        report.plot_comparison(args.plot)
        print(f"Comparison plot saved to {args.plot}")


if __name__ == "__main__":
    sys.exit(main())
