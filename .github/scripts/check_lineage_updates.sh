#!/bin/bash
set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <first_branch_name> <second_branch_name>"
    exit 1
fi

old_branch="$1"
new_branch="$2"

top_level=$(git rev-parse --show-toplevel)
current_branch=$(git rev-parse --abbrev-ref HEAD)

# run cleanup on exit
function cleanup {
    git checkout $current_branch
    ./report_pr_changes --branch old --computation clean
}
trap cleanup EXIT

cd $top_level/bin

git checkout $old_branch
./report_pr_changes --branch old --computation lineage_hash
git checkout $new_branch
./report_pr_changes --branch new --computation lineage_hash
./report_pr_changes --branch new --computation hash_comparison
./report_pr_changes --branch new --computation print_added_plugin
./report_pr_changes --branch new --computation changed_affected_plugin
git checkout $old_branch
./report_pr_changes --branch old --computation changed_affected_plugin
./report_pr_changes --branch old --computation report_changes
