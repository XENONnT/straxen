"""Pytest plugin for monitoring straxen plugin performance.

This module provides pytest hooks to automatically measure and report time and memory usage for
straxen plugin tests.

"""

import pytest
import json
import csv
import os
from pathlib import Path

from straxen.performance_monitor import (
    PerformanceCollector,
    measure_plugin_performance,
    HAS_PSUTIL,
)


def _restore_original_compute(original_compute):
    """Helper function for unpickling: returns the original compute method."""
    return original_compute


class _PerformanceComputeWrapper:
    """Picklable wrapper for plugin compute methods.

    This class wraps plugin compute() methods to add performance monitoring. It's implemented as a
    class to be picklable (nested functions can't be pickled).

    """

    def __init__(self, plugin_instance, original_compute, collector):
        self.plugin_instance = plugin_instance
        self.original_compute = original_compute
        self.collector = collector

    def __call__(self, *args, **kwargs):
        return measure_plugin_performance(
            self.plugin_instance, self.original_compute, *args, **kwargs
        )

    def __reduce__(self):
        # Return the original compute method when pickling
        # This allows the plugin to be pickled without the wrapper
        return (_restore_original_compute, (self.original_compute,))


def pytest_addoption(parser):
    """Add command-line options for performance monitoring."""
    parser.addoption(
        "--monitor-performance",
        action="store_true",
        default=False,
        help="Enable performance monitoring for plugin tests",
    )
    parser.addoption(
        "--monitor-tracemalloc",
        action="store_true",
        default=False,
        help="Enable detailed tracemalloc memory profiling (adds overhead)",
    )
    parser.addoption(
        "--performance-output-dir",
        action="store",
        default="test-results",
        help="Directory to save performance metrics (default: test-results)",
    )


def pytest_configure(config):
    """Configure pytest with performance monitoring markers and register reporter."""
    config.addinivalue_line(
        "markers",
        "monitor_performance: mark test to monitor plugin performance",
    )

    # Register terminal reporter if monitoring is enabled
    if config.getoption("--monitor-performance") or os.environ.get(
        "STRAXEN_MONITOR_PERFORMANCE", ""
    ).lower() in ("1", "true", "yes"):
        reporter = PerformanceTerminalReporter(config)
        config.pluginmanager.register(reporter, "performance_terminal_reporter")


@pytest.fixture(scope="session", autouse=True)
def performance_collector(request):
    """Session-scoped fixture that manages the performance collector.

    Only monitors plugins when:
    1. Monitoring is enabled (--monitor-performance or STRAXEN_MONITOR_PERFORMANCE=1)
    2. Test is in the PluginTest class (standard tests with real data)

    """
    config = request.config

    # Only enable if explicitly requested via CLI or env var
    # NOTE: Auto-enable in CI is disabled due to pickle compatibility issues
    enabled = config.getoption("--monitor-performance") or os.environ.get(
        "STRAXEN_MONITOR_PERFORMANCE", ""
    ).lower() in ("1", "true", "yes")

    if not enabled:
        yield None
        return

    if not HAS_PSUTIL:
        pytest.warn(
            "psutil not installed, performance monitoring will have limited functionality",
            RuntimeWarning,
        )

    use_tracemalloc = config.getoption("--monitor-tracemalloc") or os.environ.get(
        "STRAXEN_MONITOR_TRACEMALLOC", ""
    ).lower() in ("1", "true", "yes")

    # Initialize collector
    collector = PerformanceCollector()
    collector.reset()
    collector.enable(use_tracemalloc=use_tracemalloc)

    # Monkey-patch strax.Plugin.__init__ to wrap compute methods
    import strax

    _original_plugin_init = strax.Plugin.__init__

    def wrapped_plugin_init(self, *args, **kwargs):
        _original_plugin_init(self, *args, **kwargs)
        # Wrap the compute method of this instance using a picklable wrapper
        if hasattr(self, "compute") and callable(self.compute):
            self.compute = _PerformanceComputeWrapper(self, self.compute, collector)

    strax.Plugin.__init__ = wrapped_plugin_init

    yield collector

    # Restore original methods
    strax.Plugin.__init__ = _original_plugin_init
    collector.disable()

    # Generate reports
    output_dir = config.getoption("--performance-output-dir")
    _generate_performance_reports(collector, output_dir)

    # Print summary to console (timing only, RAM is unreliable)
    summary = collector.get_summary()
    if summary:
        print(f"\n{'=' * 70}")
        print("PLUGIN PERFORMANCE SUMMARY (Timing)")
        print(f"{'=' * 70}\n")
        print(f"{'Plugin':<40} {'Avg Time (ms)':>15} {'Executions':>12}")
        print(f"{'-' * 70}")

        sorted_plugins = sorted(
            summary.items(),
            key=lambda x: x[1]["total_time_ms"],
            reverse=True,
        )

        for target, stats in sorted_plugins:
            avg_time = stats["total_time_ms"] / stats["count"]
            count = stats["count"]
            print(f"{target:<40} {avg_time:>15.2f} {count:>12}")

        print(f"{'=' * 70}")
        print(f"Total: {len(summary)} plugins measured")
        print(f"{'=' * 70}\n")


def _generate_performance_reports(collector: PerformanceCollector, output_dir: str):
    """Generate performance report files."""
    metrics = collector.get_metrics()

    if not metrics:
        return

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate CSV report
    csv_file = output_path / "plugin_performance.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "plugin_name",
                "target",
                "execution_time_ms",
                "ram_before_mb",
                "ram_after_mb",
                "ram_delta_mb",
                "ram_peak_mb",
                "tracemalloc_peak_mb",
                "tracemalloc_delta_mb",
            ],
        )
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric.to_dict())

    # Generate JSON report
    json_file = output_path / "plugin_performance.json"
    with open(json_file, "w") as f:
        json.dump(
            {
                "metrics": [m.to_dict() for m in metrics],
                "summary": collector.get_summary(),
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 70}")
    print(f"Performance reports saved:")
    print(f"  CSV:  {csv_file}")
    print(f"  JSON: {json_file}")
    print(f"{'=' * 70}\n")


@pytest.fixture
def monitor_plugin_performance(performance_collector):
    """Fixture that can be used to enable monitoring for specific tests."""
    return performance_collector


def pytest_runtest_logreport(report):
    """Hook to display performance metrics after each test."""
    if report.when != "call":
        return

    collector = PerformanceCollector()
    if not collector.enabled:
        return

    # Metrics are displayed by the PerformanceTerminalReporter
    pass


class PerformanceTerminalReporter:
    """Terminal reporter for displaying performance metrics after each test."""

    def __init__(self, config):
        self.config = config
        self.collector = PerformanceCollector()
        self.last_metric_count = 0

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):
        """Display metrics after each test call phase completes."""
        outcome = yield  # noqa: F841
        if call.when != "call":
            return

        if not self.collector.enabled:
            return

        # Only report for PluginTest class (not TestEmptyRecords, etc.)
        if not (hasattr(item, "cls") and item.cls and item.cls.__name__ == "PluginTest"):
            return

        metrics = self.collector.get_metrics()
        new_metrics = metrics[self.last_metric_count :]
        self.last_metric_count = len(metrics)

        if new_metrics:
            # Aggregate metrics by plugin_name and target
            aggregated = {}
            for metric in new_metrics:
                key = (metric.plugin_name, metric.target)
                if key not in aggregated:
                    aggregated[key] = {
                        "time_ms": [],
                    }
                aggregated[key]["time_ms"].append(metric.execution_time_ms)

            # Print aggregated results (timing only)
            print()
            for (plugin_name, target), data in aggregated.items():
                count = len(data["time_ms"])
                avg_time = sum(data["time_ms"]) / count
                max_time = max(data["time_ms"])

                time_str = f"{avg_time:.1f}ms"
                if count > 1:
                    time_str += f" (max={max_time:.1f}ms, {count}x)"

                print(f"  📊 {plugin_name} ({target}): {time_str}")
