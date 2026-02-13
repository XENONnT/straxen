# Compare memory-related settings between v2.2.7 and daq-vanilla

import subprocess
import re

def get_plugin_versions(ref, plugin_name):
    cmd = f"git show {ref}:straxen/plugins/**/{plugin_name}.py 2>/dev/null | grep '__version__'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)', result.stdout)
        return match.group(1) if match else "not found"
    return "file not found"

plugins = [
    "peaklets/peaklets.py",
    "merged_s2s/merged_s2s.py", 
    "peaks/peaks.py",
    "events/events.py",
]

print("Plugin Version Comparison")
print("=" * 70)
print(f"{'Plugin':<40} {'v2.2.7':<15} {'daq-vanilla':<15}")
print("-" * 70)

for plugin in plugins:
    plugin_name = plugin.split('/')[-1].replace('.py', '')
    old_ver = get_plugin_versions("v2.2.7", plugin)
    new_ver = get_plugin_versions("daq-vanilla", plugin)
    changed = "🔴" if old_ver != new_ver else ""
    print(f"{plugin:<40} {old_ver:<15} {new_ver:<15} {changed}")
