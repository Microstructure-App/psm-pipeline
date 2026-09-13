#!/usr/bin/env python3
"""
harvest_runtime.py -- recover the TRUE wall-clock runtime of the H1 sweep from
the filesystem, with no human timing and no human participants.

WHY THIS EXISTS
---------------
H4 needs a measured throughput figure for the psm route. Timing a person
requires human-participants approval at ISEF-affiliated competitions (TJAS,
Regeneron STS). Timing a COMPUTER does not. The 270 archived DREAM.3D volumes
from the H1 sweep each carry a filesystem modification time, so the sweep
timed itself. This script reads those timestamps back.

The sweep was resumable (Ctrl-C, close laptop, rerun), so the raw first-to-last
span almost certainly contains idle gaps. This script separates:

    span_total      -- first mtime to last mtime (includes any idle gaps)
    active_compute  -- span with idle gaps removed  <-- USE THIS ONE
    per_run_median  -- median seconds between consecutive completions

A gap is any inter-run interval longer than GAP_FACTOR x the median interval.
Both numbers are printed so you can report the honest one and disclose the other.

USAGE
-----
    python harvest_runtime.py "outputs/H1_final_2026-07-12/volumes"
    python harvest_runtime.py "outputs/H1_final_2026-07-12/volumes" --ext .npz
    python harvest_runtime.py <dir> --csv runtime_evidence.csv

CAVEAT TO DISCLOSE IN METHODS
-----------------------------
mtimes are a retrospective proxy, not an instrumented timer. They are accurate
provided the files were not copied in a way that rewrote timestamps. If your
archive folder was created with a copy that reset mtimes, ALL files will share
nearly the same timestamp -- the script detects this and refuses to report.
Going forward, log timestamps inside the pipeline itself so the measurement is
instrumented rather than inferred.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from datetime import datetime
from pathlib import Path

GAP_FACTOR = 10.0        # interval > GAP_FACTOR x median  =>  idle gap
FLAT_TOL_SEC = 2.0       # if the whole span is under this, mtimes were reset


def collect(root: Path, ext: str) -> list:
    files = sorted(p for p in root.rglob(f"*{ext}") if p.is_file())
    if not files:
        print(f"No '*{ext}' files under {root}", file=sys.stderr)
        sys.exit(1)
    return sorted(((p, p.stat().st_mtime) for p in files), key=lambda t: t[1])


def analyse(stamps: list) -> dict:
    times = [t for _, t in stamps]
    n = len(times)
    span = times[-1] - times[0]

    if span < FLAT_TOL_SEC:
        print("REFUSING TO REPORT: all modification times are within "
              f"{FLAT_TOL_SEC}s of each other. The archive was almost certainly "
              "created by a copy that reset timestamps, so these files carry no "
              "timing information. Re-derive from the original output folder, or "
              "instrument the pipeline and re-run a subset.", file=sys.stderr)
        sys.exit(2)

    deltas = [times[i + 1] - times[i] for i in range(n - 1)]
    med = statistics.median(deltas)
    threshold = GAP_FACTOR * med if med > 0 else float("inf")

    gaps = [d for d in deltas if d > threshold]
    active = sum(d for d in deltas if d <= threshold)

    return {
        "n_runs": n,
        "first": times[0],
        "last": times[-1],
        "span_total_s": span,
        "active_compute_s": active,
        "idle_s": sum(gaps),
        "n_gaps": len(gaps),
        "per_run_median_s": med,
        "per_run_mean_active_s": active / max(1, n - 1),
        "per_run_min_s": min(deltas),
        "per_run_max_active_s": max((d for d in deltas if d <= threshold), default=0.0),
    }


def hrs(sec: float) -> float:
    return sec / 3600.0


def report(a: dict, root: Path, ext: str) -> None:
    ts = lambda s: datetime.fromtimestamp(s).strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n=== H1 sweep runtime, recovered from {a['n_runs']} '{ext}' files ===")
    print(f"    source: {root}")
    print(f"    first completion : {ts(a['first'])}")
    print(f"    last  completion : {ts(a['last'])}")
    print()
    print(f"  span (wall clock, incl. idle) : {hrs(a['span_total_s']):8.2f} h")
    print(f"  idle gaps removed             : {hrs(a['idle_s']):8.2f} h  "
          f"({a['n_gaps']} gap{'s' if a['n_gaps'] != 1 else ''})")
    print(f"  ACTIVE COMPUTE                : {hrs(a['active_compute_s']):8.2f} h"
          "   <-- report this")
    print()
    print(f"  per-reconstruction, median    : {a['per_run_median_s']:8.1f} s")
    print(f"  per-reconstruction, mean      : {a['per_run_mean_active_s']:8.1f} s")
    print(f"  per-reconstruction, min / max : {a['per_run_min_s']:.1f} s / "
          f"{a['per_run_max_active_s']:.1f} s")
    print()

    thr = a["n_runs"] / hrs(a["active_compute_s"]) if a["active_compute_s"] else 0
    print(f"  THROUGHPUT: {thr:.1f} independent 3D reconstructions per compute-hour")
    print()

    lo = a["per_run_min_s"] / 3600.0
    hi = a["per_run_max_active_s"] / 3600.0
    print("--- paste into h4_cost_model.py ---")
    print(f'PSM_MACHINE_SPEC = "TODO: CPU / RAM / OS of the machine that ran this"')
    print(f'PSM_RECON_HOURS_EACH = ({lo:.4f}, {hi:.4f})   # per reconstruction')
    print(f'PSM_SWEEP_RUNS = {a["n_runs"]}')
    print(f'PSM_SWEEP_ACTIVE_HOURS = {hrs(a["active_compute_s"]):.3f}')
    print("-----------------------------------\n")

    if a["n_gaps"]:
        print("NOTE: idle gaps were detected and excluded. Disclose both numbers "
              "in Methods -- report active compute, state the total span, and say "
              "the sweep was resumable and run across sessions.\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", help="folder containing the archived volumes")
    ap.add_argument("--ext", default=".npz", help="file extension (default .npz)")
    ap.add_argument("--csv", help="write per-file timestamps as evidence")
    a = ap.parse_args()

    root = Path(a.directory).expanduser()
    if not root.exists():
        print(f"No such directory: {root}", file=sys.stderr)
        return 1

    stamps = collect(root, a.ext)
    res = analyse(stamps)
    report(res, root, a.ext)

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "file", "mtime_epoch", "mtime_iso", "delta_s"])
            prev = None
            for i, (p, t) in enumerate(stamps):
                w.writerow([i, p.name, f"{t:.3f}",
                            datetime.fromtimestamp(t).isoformat(),
                            "" if prev is None else f"{t - prev:.3f}"])
                prev = t
        print(f"Evidence CSV written: {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
