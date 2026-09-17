"""Command-line interface for the straxen Environment & Runtime Validator.

Usage:
    straxen_validate_environment
    straxen_validate_environment --skip-network --skip-mounts
    straxen_validate_environment --output report.json
    straxen_validate_environment --strict

"""

import argparse
import json
import sys
from straxen.storage.validator import EnvironmentValidator


def build_parser():
    parser = argparse.ArgumentParser(
        description="Validate straxen runtime, XRootD tools, SciTokens, and cluster mounts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--skip-network",
        action="store_true",
        default=False,
        help="Skip remote XRootD network connectivity checks",
    )
    parser.add_argument(
        "--skip-mounts",
        action="store_true",
        default=False,
        help="Skip cluster filesystem mount checks",
    )
    parser.add_argument(
        "--xrootd-url",
        type=str,
        default=None,
        help="Custom XRootD redirector URL to ping",
    )
    parser.add_argument(
        "--check-paths",
        nargs="+",
        default=None,
        help="Custom filesystem paths to verify",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export JSON validation report",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output raw JSON report to stdout",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit with non-zero code if any WARN or FAIL status is found",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    validator = EnvironmentValidator(
        xrootd_url=args.xrootd_url,
        cluster_paths=args.check_paths,
    )

    report = validator.validate_all(
        skip_network=args.skip_network,
        skip_mounts=args.skip_mounts,
    )

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print_summary()

    if args.output:
        report.to_json(args.output)
        print(f"Report exported to {args.output}")

    if report.overall_status == "UNHEALTHY":
        return 1
    if args.strict and report.overall_status != "HEALTHY":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
