"""Command-line interface for the SLURM Benchmark Matrix Runner on Midway.

Modes:
  generate: Build manifest and SLURM array job scripts.
  submit:   Build and submit job array to SLURM (supports --dry-run).
  run-task: Worker executor invoked by SLURM array task.
  collect:  Harvest worker task JSON outputs into summary report and plots.

"""

import argparse
import os
import sys
from straxen.storage.slurm import (
    BenchmarkMatrix,
    MatrixReport,
    SlurmConfig,
    SlurmMatrixRunner,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Straxen SLURM Benchmark Matrix Runner for Midway",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Common matrix args
    def add_matrix_args(p):
        p.add_argument(
            "--backends",
            nargs="+",
            default=["posix", "xrootd"],
            choices=["posix", "xrootd", "memory", "file"],
            help="Storage tiers to profile",
        )
        p.add_argument(
            "--targets",
            nargs="+",
            default=["records", "peaks"],
            choices=["records", "peaks", "events", "event_info", "raw_records"],
            help="Data targets to benchmark",
        )
        p.add_argument(
            "--workloads",
            nargs="+",
            default=["bulk_array", "streaming_iter"],
            choices=["bulk_array", "streaming_iter", "time_slice", "metadata_probe"],
            help="Workloads to benchmark",
        )
        p.add_argument(
            "--workers",
            type=int,
            nargs="+",
            default=[1, 2, 4],
            help="Worker thread concurrency levels",
        )
        p.add_argument(
            "--chunk-sizes-mb",
            type=float,
            nargs="+",
            default=[2.0],
            help="Chunk sizes in MB",
        )
        p.add_argument("--n-chunks", type=int, default=5, help="Chunks per dataset")
        p.add_argument("--iterations", type=int, default=3, help="Repetitions per test")
        p.add_argument("--warmup", type=int, default=1, help="Warmup iterations")
        p.add_argument(
            "--compressor",
            type=str,
            default="zstd",
            choices=["zstd", "blosc", "lz4", "bz2"],
            help="Compression algorithm",
        )
        p.add_argument("--run-id", type=str, default="050000", help="Run ID")
        p.add_argument(
            "--xrootd-url",
            type=str,
            default="root://midway-origin.uchicago.edu/",
            help="XRootD redirector URL",
        )

    # Common SLURM args
    def add_slurm_args(p):
        p.add_argument(
            "--partition",
            type=str,
            default="caslake",
            help="SLURM partition (caslake, broadwl, dali, etc.)",
        )
        p.add_argument(
            "--account",
            type=str,
            default="pi-lgrandi",
            help="SLURM account name",
        )
        p.add_argument("--qos", type=str, default=None, help="SLURM QOS")
        p.add_argument("--time", type=str, default="01:00:00", help="Wall clock limit")
        p.add_argument(
            "--cpus-per-task",
            type=int,
            default=4,
            help="CPUs allocated per job",
        )
        p.add_argument(
            "--mem-per-cpu",
            type=str,
            default="4G",
            help="Memory allocated per CPU",
        )
        p.add_argument(
            "--max-concurrent",
            type=int,
            default=20,
            help="SLURM job array concurrency limit",
        )
        p.add_argument(
            "--output-dir",
            type=str,
            default="./slurm_benchmarks",
            help="Directory for logs, scripts, and results",
        )
        p.add_argument(
            "--env-setup",
            type=str,
            default=None,
            help="Custom shell commands or script to setup environment",
        )
        p.add_argument(
            "--singularity-image",
            type=str,
            default=None,
            help="Path to Singularity container image",
        )

    # Subcommand: generate
    p_gen = subparsers.add_parser("generate", help="Generate SLURM array scripts")
    add_matrix_args(p_gen)
    add_slurm_args(p_gen)

    # Subcommand: submit
    p_sub = subparsers.add_parser("submit", help="Submit benchmark matrix to SLURM")
    add_matrix_args(p_sub)
    add_slurm_args(p_sub)
    p_sub.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Generate scripts without invoking sbatch",
    )

    # Subcommand: run-task
    p_task = subparsers.add_parser("run-task", help="Execute single matrix task")
    p_task.add_argument("--manifest", type=str, required=True, help="Path to manifest JSON")
    p_task.add_argument("--task-id", type=int, required=True, help="Task index to run")
    p_task.add_argument("--output-dir", type=str, required=True, help="Directory for output")

    # Subcommand: collect
    p_col = subparsers.add_parser("collect", help="Aggregate matrix results into report")
    p_col.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Directory containing result_*.json",
    )
    p_col.add_argument(
        "--output-dir",
        type=str,
        default="./slurm_benchmarks",
        help="Root benchmark output directory",
    )
    p_col.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export merged JSON report",
    )
    p_col.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path to save comparison plot image (.png)",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.subcommand in ("generate", "submit"):
        matrix = BenchmarkMatrix(
            backends=args.backends,
            targets=args.targets,
            workloads=args.workloads,
            workers=args.workers,
            chunk_sizes_mb=args.chunk_sizes_mb,
            n_chunks=args.n_chunks,
            iterations=args.iterations,
            warmup=args.warmup,
            compressor=args.compressor,
            run_id=args.run_id,
            xrootd_url=args.xrootd_url,
        )
        slurm_cfg = SlurmConfig(
            partition=args.partition,
            account=args.account,
            qos=args.qos,
            time=args.time,
            cpus_per_task=args.cpus_per_task,
            mem_per_cpu=args.mem_per_cpu,
            output_dir=args.output_dir,
            max_concurrent_jobs=args.max_concurrent,
            env_setup=args.env_setup,
            singularity_image=args.singularity_image,
        )
        runner = SlurmMatrixRunner(matrix=matrix, slurm_config=slurm_cfg)

        if args.subcommand == "generate":
            paths = runner.generate_scripts(args.output_dir)
            print(f"Generated SLURM matrix in {paths['output_dir']}")
            print(f"Total tasks: {paths['n_tasks']}")
            print(f"Array script: {paths['array_script']}")
            print(f"Manifest: {paths['manifest']}")
            return 0

        elif args.subcommand == "submit":
            res = runner.submit(output_dir=args.output_dir, dry_run=args.dry_run)
            if res.get("status") == "dry_run":
                print(f"[Dry Run] Generated scripts in {res['paths']['output_dir']}")
                print(f"Command to submit: {res['command']}")
            elif res.get("status") == "submitted":
                print(f"Submitted SLURM job array: {res.get('job_id')}")
            else:
                print(f"Submission failed: {res.get('error')}", file=sys.stderr)
                return 1
            return 0

    elif args.subcommand == "run-task":
        out_file = SlurmMatrixRunner.execute_task(
            manifest_path=args.manifest,
            task_id=args.task_id,
            output_dir=args.output_dir,
        )
        print(f"Completed task {args.task_id} -> {out_file}")
        return 0

    elif args.subcommand == "collect":
        res_dir = args.results_dir
        manifest_path = None
        if not res_dir:
            res_dir = os.path.join(args.output_dir, "results")
            man_candidate = os.path.join(args.output_dir, "matrix_manifest.json")
            if os.path.isfile(man_candidate):
                manifest_path = man_candidate

        report: MatrixReport = SlurmMatrixRunner.collect_results(
            results_dir=res_dir,
            manifest_path=manifest_path,
        )
        report.print_summary()

        if args.output:
            report.to_json(args.output)
            print(f"Exported merged report to {args.output}")

        if args.plot:
            report.plot_matrix(args.plot)
            print(f"Saved matrix plot to {args.plot}")

        return 0


if __name__ == "__main__":
    sys.exit(main())
