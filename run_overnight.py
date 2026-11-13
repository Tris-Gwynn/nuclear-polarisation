import subprocess
import sys
import time

# Overnight checks. Each script's output is logged to its own file, and a
# failure does not stop the remaining scripts.
#
# 6_generate_rate_check.py must already be edited to run only slow_triplet
# at N_B0 = 5000 and save to Figure_6/figure6_slow_triplet_5000.npz.
# The Figure 6 plot script is not run, since it expects all three rate cases.

ROOT = "/home/tristengwynn/nuclear-polarisation"

scripts = [
    ("Figure_8/8_generate_grid.py",            "log_orientation_map.txt"),
    ("Figure_6/6_generate_rate_check.py",      "log_slow_triplet_5000.txt"),
]

status = {}
for script, log in scripts:
    path = f"{ROOT}/{script}"
    print(f"--- Starting {script} (log: {log}) ---", flush=True)
    t0 = time.time()
    with open(f"{ROOT}/{log}", "w") as f:
        result = subprocess.run([sys.executable, path], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    minutes = (time.time() - t0) / 60
    status[script] = "OK" if result.returncode == 0 else f"FAILED (exit {result.returncode})"
    print(f"--- {script}: {status[script]} after {minutes:.1f} min ---\n", flush=True)

print("Summary:")
for script, s in status.items():
    print(f"  {script}: {s}")