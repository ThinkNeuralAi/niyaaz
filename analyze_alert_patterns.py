"""
Analysis of alert logging patterns across modules
"""
import os
import re

modules_dir = "modules"

print("=" * 80)
print("ANALYZING ALERT LOGGING PATTERNS")
print("=" * 80)

modules = {
    'ppe_monitoring.py': {},
    'person_smoking_detection.py': {},
    'material_theft_monitor.py': {},
    'unauthorized_entry_monitor.py': {},
    'crowd_detection.py': {},
    'table_service_monitor.py': {},
    'service_discipline_monitor.py': {},
    'cash_detection.py': {},
    'bag_detection.py': {},
}

for module_name in modules:
    module_path = os.path.join(modules_dir, module_name)
    if os.path.exists(module_path):
        with open(module_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Check for gif_recorder usage
        has_gif_recorder = 'AlertGifRecorder' in content or 'gif_recorder' in content
        
        # Count log_alert and save_alert_gif calls
        log_alert_count = len(re.findall(r'\.log_alert\(', content))
        save_alert_gif_count = len(re.findall(r'\.save_alert_gif\(', content))
        
        modules[module_name] = {
            'has_gif_recorder': has_gif_recorder,
            'log_alert_calls': log_alert_count,
            'save_alert_gif_calls': save_alert_gif_count,
        }
        
        print(f"\n{module_name}:")
        print(f"  Has GIF Recorder: {has_gif_recorder}")
        print(f"  log_alert() calls: {log_alert_count}")
        print(f"  save_alert_gif() calls: {save_alert_gif_count}")
        
        if has_gif_recorder and log_alert_count > 0 and save_alert_gif_count > 0:
            print(f"  ⚠️ ISSUE: Calls BOTH log_alert() and save_alert_gif() - may create duplicate records")
        elif has_gif_recorder and log_alert_count > 0:
            print(f"  ❌ ISSUE: Has GIF recording but ONLY calls log_alert() (no save_alert_gif)")
        elif has_gif_recorder and save_alert_gif_count > 0:
            print(f"  ✅ OK: Has GIF recording and calls save_alert_gif()")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

problematic = []
for module_name, info in modules.items():
    if info.get('has_gif_recorder') and info.get('log_alert_calls', 0) > 0:
        problematic.append((module_name, info))

if problematic:
    print(f"\nFound {len(problematic)} modules to fix:")
    for module_name, info in problematic:
        print(f"  - {module_name} ({info['log_alert_calls']} log_alert calls, {info['save_alert_gif_calls']} save_alert_gif calls)")
else:
    print("\nNo issues found!")
