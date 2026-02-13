# How Performance Monitoring Shows Up in GitHub CI

## 📋 Overview

When you push a PR, the performance monitoring system automatically runs and displays results in multiple places.

## 🔍 Where to Find Results

### 1. In the PR "Checks" Tab

**Path**: PR → "Checks" tab → Select test job (e.g., "pytest_py3.10")

**What you'll see in the test log**:

```
tests/plugins/test_plugins.py::PluginTest::test_records PASSED

  📊 PulseProcessing (records):
     ⏱️  Time: 1050.22 ms
     💾 RAM Delta: +64.02 MB (peak: 377.22 MB)
  
  📊 PulseProcessing (records):
     ⏱️  Time: 1050.35 ms
     💾 RAM Delta: +64.02 MB (peak: 377.22 MB)

tests/plugins/test_plugins.py::PluginTest::test_peaklets PASSED

  📊 Peaklets (peaklets):
     ⏱️  Time: 3983.22 ms
     💾 RAM Delta: +100.32 MB (peak: 450.50 MB)
```

### 2. In the "Display performance summary" Step

After all tests complete, a dedicated workflow step shows:

```
📊 Plugin Performance Summary:

============================================================
Plugin                         Avg Time (ms)   Avg RAM (MB)   
------------------------------------------------------------
peaklets                          3983.22              100.32
records                           1050.29               64.02
merged_s2s                         845.67               45.12
event_info                         512.34               28.50
============================================================
Total: 4 plugins measured
```

### 3. As Downloadable Artifacts

**Path**: Workflow run → Bottom of page → "Artifacts" section

**Available downloads**:
- `performance-results-py3.10-pytest`
- `performance-results-py3.11-pytest`
- `performance-results-py3.10-pytest_no_database`

Each contains:
- `plugin_performance.csv` - Spreadsheet-friendly format
- `plugin_performance.json` - Detailed metrics with summary

## 📊 Sample JSON Output

```json
{
  "metrics": [
    {
      "plugin_name": "PulseProcessing",
      "target": "records",
      "execution_time_ms": 1050.22,
      "ram_before_mb": 313.20,
      "ram_after_mb": 377.22,
      "ram_delta_mb": 64.02,
      "ram_peak_mb": 377.22,
      "tracemalloc_peak_mb": null,
      "tracemalloc_delta_mb": null
    },
    ...
  ],
  "summary": {
    "records": {
      "count": 2,
      "total_time_ms": 2100.57,
      "max_time_ms": 1050.35,
      "total_ram_delta_mb": 128.04,
      "max_ram_delta_mb": 64.02,
      "max_ram_peak_mb": 377.22
    },
    ...
  }
}
```

## 🎯 Example Use Cases

### Use Case 1: Review PR Performance Impact

**Before merging a PR**, check if it affects plugin performance:

1. Go to the PR's "Checks" tab
2. Find the "Display performance summary" step
3. Compare time/memory vs. previous runs
4. Look for unexpected increases

### Use Case 2: Identify Slow Plugins

Download artifacts and open CSV in Excel/Google Sheets:
```csv
plugin_name,target,execution_time_ms,ram_delta_mb
Peaklets,peaklets,3983.22,100.32
PulseProcessing,records,1050.22,64.02
```

Sort by `execution_time_ms` to find bottlenecks.

### Use Case 3: Track Performance Over Time

1. Download artifacts from multiple workflow runs
2. Compare metrics across commits/branches
3. Create trend charts in your favorite tool

## ⚙️ Configuration

### For Regular PRs
No configuration needed! It just works.

### To Disable Temporarily
In your PR, modify `.github/workflows/pytest.yml`:
```yaml
env:
  STRAXEN_MONITOR_PERFORMANCE: "0"
```

### To Enable More Verbose Profiling
Add to workflow:
```yaml
run: |
  pytest --monitor-tracemalloc  # Detailed Python object tracking
```

## 🚀 Benefits

✅ **Zero overhead for developers** - Automatic in CI  
✅ **Non-intrusive** - Doesn't slow down local development  
✅ **Always up-to-date** - Metrics on every PR  
✅ **Historical data** - Download artifacts for trend analysis  
✅ **Easy comparison** - See performance changes in PRs  

## 📝 Notes

- Artifacts are retained for **30 days**
- Performance can vary slightly between CI runs (±5%)
- Use metrics for relative comparisons, not absolute benchmarks
- CI uses GitHub-hosted runners (Ubuntu, 2 CPU cores)
