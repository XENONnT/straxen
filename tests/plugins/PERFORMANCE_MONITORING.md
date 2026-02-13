# Plugin Performance Monitoring

This document describes how to use the performance monitoring feature for straxen plugin tests.

## Overview

The performance monitoring system automatically tracks execution time and RAM usage for each plugin's `compute()` method during test runs. This helps identify performance bottlenecks and memory-intensive operations.

**Behavior**:
- **In CI/GitHub Actions**: Automatically enabled for all test runs
- **Locally**: Opt-in via `--monitor-performance` flag or environment variable

## CI/CD Integration

### GitHub Actions

Performance monitoring is **automatically enabled** in GitHub Actions CI runs. The system:

1. **Detects CI environment** - Uses the `CI=true` environment variable
2. **Collects metrics** - Monitors all plugin tests during the normal pytest run
3. **Shows inline output** - Metrics appear in test logs after each test
4. **Displays summary** - A dedicated step shows the performance summary table
5. **Uploads artifacts** - CSV and JSON files are saved as workflow artifacts

### Viewing Results in GitHub

**In PR test logs**:
1. Go to the "Checks" tab of your PR
2. Click on the test job (e.g., "pytest_py3.10")
3. Scroll through the test output to see per-plugin metrics like:
   ```
   📊 PulseProcessing (records):
      ⏱️  Time: 1050.22 ms
      💾 RAM Delta: +64.02 MB (peak: 377.22 MB)
   ```
4. Look for the "Display performance summary" step to see the aggregate table

**Download artifacts**:
1. Go to the workflow run page
2. Scroll to the "Artifacts" section at the bottom
3. Download `performance-results-py{version}-{test}` 
4. Extract to get CSV/JSON files

### Artifact Retention

Performance artifacts are kept for **30 days** after the workflow run.

## Usage

### Local Development (Opt-In)

To enable performance monitoring locally, add the `--monitor-performance` flag:

```bash
pytest tests/plugins/ --monitor-performance
```

### Disabling in CI

If you need to disable performance monitoring in CI (e.g., for debugging):

```yaml
# In .github/workflows/pytest.yml
env:
  STRAXEN_MONITOR_PERFORMANCE: "0"  # Disable monitoring
```

Or temporarily via environment variable:
```bash
export STRAXEN_MONITOR_PERFORMANCE=0
pytest tests/plugins/
```

### Advanced Options

**Enable detailed memory profiling with tracemalloc** (adds overhead):
```bash
pytest tests/plugins/ --monitor-performance --monitor-tracemalloc
```

**Specify custom output directory**:
```bash
pytest tests/plugins/ --monitor-performance --performance-output-dir=my-results/
```

## Output

### Console Output

During test execution, you'll see metrics printed after each test:

```
  📊 PulseProcessing (records):
     ⏱️  Time: 1050.22 ms
     💾 RAM Delta: +64.02 MB (peak: 377.22 MB)
```

At the end of the test session, a summary table is displayed:

```
======================================================================
PLUGIN PERFORMANCE SUMMARY
======================================================================

Plugin                         Avg Time (ms)   Avg RAM (MB)   
------------------------------------------------------------
peaklets                          3983.22              100.32
records                            951.58               59.98
======================================================================
Total: 2 plugins measured
======================================================================
```

### File Artifacts

Performance data is saved to two formats in the output directory (default: `test-results/`):

1. **CSV format** (`plugin_performance.csv`): Easy to import into spreadsheets
   ```csv
   plugin_name,target,execution_time_ms,ram_before_mb,ram_after_mb,ram_delta_mb,...
   PulseProcessing,records,1050.22,313.20,377.22,64.02,...
   ```

2. **JSON format** (`plugin_performance.json`): Includes summary statistics
   ```json
   {
     "metrics": [...],
     "summary": {
       "records": {
         "count": 2,
         "total_time_ms": 2100.57,
         "max_time_ms": 1050.35,
         ...
       }
     }
   }
   ```

## Metrics Explained

- **execution_time_ms**: Time taken to execute the plugin's compute() method
- **ram_before_mb**: Process RSS memory before compute()
- **ram_after_mb**: Process RSS memory after compute()
- **ram_delta_mb**: Change in memory (after - before)
- **ram_peak_mb**: Peak memory during compute() (currently same as after)
- **tracemalloc_peak_mb**: Python object memory peak (if --monitor-tracemalloc enabled)
- **tracemalloc_delta_mb**: Python object memory delta (if --monitor-tracemalloc enabled)

## Implementation Details

The monitoring system works by:
1. Monkey-patching `strax.Plugin.__init__()` to wrap each plugin instance's `compute()` method
2. Measuring memory (via psutil) and time before/after compute() execution
3. Collecting metrics in a singleton `PerformanceCollector`
4. Generating reports at the end of the test session

### Performance Overhead

- **Basic monitoring** (psutil only): <5% overhead
- **With tracemalloc**: 10-20% overhead (use only when needed)

## Troubleshooting

**No metrics collected?**
- Ensure tests actually execute plugin compute() methods (not just loading cached data)
- Verify `--monitor-performance` flag is set

**psutil not available?**
- Install with: `pip install psutil`
- Monitoring will work with reduced functionality

**High memory variance?**
- Memory measurements can vary based on system state and garbage collection
- Run multiple times and look at trends rather than absolute numbers

## Examples

### Run specific plugin tests with monitoring:
```bash
pytest tests/plugins/test_plugins.py::PluginTest::test_peaklets --monitor-performance -v
```

### Monitor all plugin tests and save to custom directory:
```bash
pytest tests/plugins/ --monitor-performance --performance-output-dir=benchmark-results/
```

### Use in CI/CD:
```bash
# In your CI script
pytest tests/plugins/ --monitor-performance
# Artifacts in test-results/ can be uploaded as build artifacts
```

## Future Enhancements

Potential improvements:
- Per-chunk metrics for multi-chunk processing
- Memory profiling at function level
- Historical trend tracking
- Performance regression detection
- Integration with pytest-benchmark
