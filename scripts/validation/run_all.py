#!/usr/bin/env python
"""Run every ``check_*.py`` of this folder (each in its own process) and print one line per script with its runtime.

    python scripts/validation/run_all.py [--quick] [--outdir DIR] [--skip tng,fake]
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--quick", action="store_true")
ap.add_argument("--outdir", default=None)
ap.add_argument("--skip", default="", help="comma-separated substrings of script names to skip (e.g. tng,fake)")
a = ap.parse_args()
here = Path(__file__).parent
skip = [s for s in a.skip.split(",") if s]
rows = []
for script in sorted(here.glob("check_*.py")):
    if any(s in script.name for s in skip):
        continue
    cmd = [sys.executable, str(script)] + (["--quick"] if a.quick else []) + (["--outdir", a.outdir] if a.outdir else [])
    t = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    last = [l for l in p.stdout.splitlines() if l.startswith("== ") and "PASS" in l]
    rows.append((script.name, p.returncode, time.perf_counter() - t, last[-1] if last else p.stderr.strip().splitlines()[-1:]))
    print(f"{'PASS' if p.returncode == 0 else 'FAIL'}  {script.name:36s} {rows[-1][2]:6.0f} s  {rows[-1][3]}")
print(f"{sum(r[1] == 0 for r in rows)}/{len(rows)} scripts passed")
sys.exit(0 if all(r[1] == 0 for r in rows) else 1)
