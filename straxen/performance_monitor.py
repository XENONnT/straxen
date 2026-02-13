"""Performance monitoring utilities for tracking plugin execution time and memory usage.

This module provides tools to measure and report the time and RAM consumption of strax plugin
compute() methods during testing.

"""

import time
import tracemalloc
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import threading
import os

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


@dataclass
class PluginMetrics:
    """Container for plugin performance metrics."""

    plugin_name: str
    target: str
    execution_time_ms: float
    ram_before_mb: float
    ram_after_mb: float
    ram_delta_mb: float
    ram_peak_mb: Optional[float] = None
    tracemalloc_peak_mb: Optional[float] = None
    tracemalloc_delta_mb: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert metrics to dictionary."""
        return {
            "plugin_name": self.plugin_name,
            "target": self.target,
            "execution_time_ms": round(self.execution_time_ms, 3),
            "ram_before_mb": round(self.ram_before_mb, 2),
            "ram_after_mb": round(self.ram_after_mb, 2),
            "ram_delta_mb": round(self.ram_delta_mb, 2),
            "ram_peak_mb": round(self.ram_peak_mb, 2) if self.ram_peak_mb else None,
            "tracemalloc_peak_mb": (
                round(self.tracemalloc_peak_mb, 2) if self.tracemalloc_peak_mb else None
            ),
            "tracemalloc_delta_mb": (
                round(self.tracemalloc_delta_mb, 2) if self.tracemalloc_delta_mb else None
            ),
        }


class PerformanceCollector:
    """Singleton collector for plugin performance metrics."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.metrics: List[PluginMetrics] = []
        self.enabled = False
        self.use_tracemalloc = False
        self._initialized = True

    def enable(self, use_tracemalloc: bool = False):
        """Enable performance monitoring."""
        self.enabled = True
        self.use_tracemalloc = use_tracemalloc
        if use_tracemalloc:
            if not tracemalloc.is_tracing():
                tracemalloc.start()

    def disable(self):
        """Disable performance monitoring."""
        self.enabled = False
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    def reset(self):
        """Clear all collected metrics."""
        self.metrics = []

    def add_metric(self, metric: PluginMetrics):
        """Add a metric to the collection."""
        self.metrics.append(metric)

    def get_metrics(self) -> List[PluginMetrics]:
        """Get all collected metrics."""
        return self.metrics.copy()

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all plugins."""
        summary = {}
        for metric in self.metrics:
            key = metric.target
            if key not in summary:
                summary[key] = {
                    "count": 0,
                    "total_time_ms": 0,
                    "max_time_ms": 0,
                    "total_ram_delta_mb": 0,
                    "max_ram_delta_mb": 0,
                    "max_ram_peak_mb": 0,
                }

            summary[key]["count"] += 1
            summary[key]["total_time_ms"] += metric.execution_time_ms
            summary[key]["max_time_ms"] = max(summary[key]["max_time_ms"], metric.execution_time_ms)
            summary[key]["total_ram_delta_mb"] += metric.ram_delta_mb
            summary[key]["max_ram_delta_mb"] = max(
                summary[key]["max_ram_delta_mb"], metric.ram_delta_mb
            )
            if metric.ram_peak_mb:
                summary[key]["max_ram_peak_mb"] = max(
                    summary[key]["max_ram_peak_mb"], metric.ram_peak_mb
                )

        return summary


def get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    if not HAS_PSUTIL:
        return 0.0

    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def measure_plugin_performance(plugin_instance, original_compute, *args, **kwargs):
    """Wrapper function to measure plugin compute() performance.

    Args:
        plugin_instance: The plugin instance
        original_compute: The original compute method
        *args, **kwargs: Arguments to pass to compute()

    Returns:
        The result of the original compute() method

    """
    collector = PerformanceCollector()

    if not collector.enabled:
        # Monitoring disabled, just call original
        return original_compute(*args, **kwargs)

    # Get plugin info
    plugin_name = plugin_instance.__class__.__name__
    provides = plugin_instance.provides
    if isinstance(provides, tuple):
        target = provides[0]  # Use first target
    else:
        target = provides

    # Measure memory before
    ram_before_mb = get_memory_usage_mb()

    # Start tracemalloc if enabled
    tracemalloc_snapshot_before = None
    if collector.use_tracemalloc and tracemalloc.is_tracing():
        tracemalloc_snapshot_before = tracemalloc.take_snapshot()

    # Measure execution time
    start_time = time.perf_counter()

    try:
        result = original_compute(*args, **kwargs)
    finally:
        # Always measure even if compute fails
        end_time = time.perf_counter()
        execution_time_ms = (end_time - start_time) * 1000

        # Measure memory after
        ram_after_mb = get_memory_usage_mb()
        ram_delta_mb = ram_after_mb - ram_before_mb

        # Get tracemalloc stats if enabled
        tracemalloc_peak_mb = None
        tracemalloc_delta_mb = None
        if collector.use_tracemalloc and tracemalloc.is_tracing():
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc_peak_mb = peak_bytes / (1024 * 1024)

            if tracemalloc_snapshot_before:
                snapshot_after = tracemalloc.take_snapshot()
                top_stats = snapshot_after.compare_to(tracemalloc_snapshot_before, "lineno")
                delta_bytes = sum(stat.size_diff for stat in top_stats)
                tracemalloc_delta_mb = delta_bytes / (1024 * 1024)

        # Create metric
        metric = PluginMetrics(
            plugin_name=plugin_name,
            target=target,
            execution_time_ms=execution_time_ms,
            ram_before_mb=ram_before_mb,
            ram_after_mb=ram_after_mb,
            ram_delta_mb=ram_delta_mb,
            ram_peak_mb=ram_after_mb,  # Use after as peak for psutil
            tracemalloc_peak_mb=tracemalloc_peak_mb,
            tracemalloc_delta_mb=tracemalloc_delta_mb,
        )

        collector.add_metric(metric)

    return result
