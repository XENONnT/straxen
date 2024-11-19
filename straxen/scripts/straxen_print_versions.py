"""Small utility to print versions of software.

Example:
    straxen_print_versions strax straxen cutax wfsim pema

"""

import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Print software versions using straxen.print_versions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "packages",
        metavar="PACKAGES",
        default=["strax", "straxen", "cutax"],
        nargs="*",
        help="Which packages to print installations of",
    )
    args = parser.parse_args()

    from strax import to_str_tuple
    from straxen import print_versions

    print_versions(to_str_tuple(args.packages))


if __name__ == "__main__":
    main()
