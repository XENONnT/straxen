import os
import sys

# List of major components and whether they have tests
components = {
    "scripts/bootstrax.py": {"size": 2231, "test": None},
    "analyses/bokeh_waveform_plot.py": {"size": 1243, "test": None},
    "scripts/ajax.py": {"size": 1001, "test": None},
    "scripts/restrax.py": {"size": 961, "test": None},
    "scada.py": {"size": 678, "test": "test_scada.py"},
    "common.py": {"size": 570, "test": "test_common.py"},
    "misc.py": {"size": 561, "test": "test_misc.py"},
    "holoviews_utils.py": {"size": 554, "test": "test_holoviews_utils.py"},
    "itp_map.py": {"size": 444, "test": "test_itp_map.py"},
    "contexts.py": {"size": 367, "test": "test_contexts.py"},
    "config/url_config.py": {"size": 514, "test": "test_url_config.py"},
}

print("Coverage Analysis")
print("=" * 60)
print(f"{'Component':<40} {'Size':<8} {'Test?'}")
print("-" * 60)

untested = []
for comp, info in sorted(components.items(), key=lambda x: x[1]["size"], reverse=True):
    has_test = "✓" if info["test"] else "✗"
    print(f"{comp:<40} {info['size']:<8} {has_test}")
    if not info["test"]:
        untested.append((comp, info["size"]))

print("\n" + "=" * 60)
print(f"\nUntested large files ({len(untested)}):")
total_untested = sum(s for _, s in untested)
for comp, size in untested:
    print(f"  - {comp} ({size} lines)")
print(f"\nTotal untested lines: {total_untested}")
