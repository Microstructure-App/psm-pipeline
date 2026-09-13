"""
diagnose.py -- why does the 3D pore fraction not match the 2D input?

    python diagnose.py

Reads outputs/batch/metrics_long.csv and the archived .npz volumes, and prints
everything needed to identify where the pore phase is being lost or invented.
Answers, in order:

  1. 2D IN vs 3D OUT, per phase, per sample.  Is only pore wrong, or are the
     phases scrambled?  If NiO and BZCYYb also land on the wrong values, the
     phase-label mapping is inverted somewhere and pore is a symptom, not the
     disease.

  2. What labels are actually IN the volume?  If DREAM.3D emits a 0 label
     (unassigned voxels that RemoveFlaggedFeatures did not fill), then
     volume_fractions_3d -- which counts only labels 1, 2, 3 -- is dividing by
     the full voxel count while ignoring the zeros, and every fraction is
     silently wrong.

  3. Grain size vs the 50-VOXEL DELETION THRESHOLD.  The template runs
     MultiThresholdObjects (NumElements < 50) -> RemoveFlaggedFeatures
     (FillRemovedFeatures = 1).  At 0.1 um voxels, a 50-voxel grain is a sphere
     of ~0.46 um diameter.  The measured pore mean diameter is ~0.4-0.5 um, so
     essentially EVERY pore grain DREAM.3D packs is a candidate for deletion,
     and the freed volume is handed to its neighbours.  This prints, per sample,
     what fraction of the pore size distribution sits below the cutoff.

  4. Can the target even be represented?  Volume = dims * voxel^3.  A 1.2% pore
     target in a 500 um^3 box is 6 um^3.  If the mean packed grain is larger
     than that, DREAM.3D cannot land anywhere near the target no matter what.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path("outputs/batch/metrics_long.csv")
VOXEL_UM = 0.1                 # BatchConfig default
DIMS = (100, 100, 50)
DELETE_BELOW_VOXELS = 50       # template filter 05: NumElements < 50

pd.set_option("display.width", 200, "display.max_columns", 40)


def main() -> None:
    if not CSV.exists():
        print(f"no CSV at {CSV}")
        return
    df = pd.read_csv(CSV)
    df = df[df.get("status", "ok") == "ok"]
    phases = [c[5:] for c in df.columns if c.startswith("vf2d_")
              and c != "vf2d_Unclassified"]

    print("=" * 78)
    print("1. WHAT WENT IN (2D)  vs  WHAT CAME OUT (3D)      [median over runs]")
    print("=" * 78)
    rows = []
    for s, g in df.groupby("sample"):
        r = {"sample": s}
        for p in phases:
            i = g[f"vf2d_{p}"].median()
            o = g[f"vf3d_{p}"].median() if f"vf3d_{p}" in g else float("nan")
            r[f"{p} in"] = round(100 * i, 2)
            r[f"{p} out"] = round(100 * o, 2)
            r[f"{p} x"] = round(o / i, 1) if i > 0 else float("inf")
        r["3D sums to"] = round(100 * sum(g[f"vf3d_{p}"].median() for p in phases), 1)
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n  'x' = out/in. If it is ~1.0 the phase reproduced. If NiO and BZCYYb")
    print("  are ~1.0 and only pore is off, the packing is the problem. If they are")
    print("  ALL off, suspect the phase-label mapping instead.")
    print("  If '3D sums to' is well under 100, the volume contains UNLABELLED voxels.")

    print("\n" + "=" * 78)
    print("2. LABELS ACTUALLY PRESENT IN THE GENERATED VOLUME")
    print("=" * 78)
    npzs = sorted(glob.glob("outputs/batch/volumes/*.npz"))
    if not npzs:
        print("  no archived volumes found")
    else:
        for f in npzs[:3]:
            vol = np.load(f)["vol"]
            u, c = np.unique(vol, return_counts=True)
            frac = {int(a): round(100 * b / vol.size, 2) for a, b in zip(u, c)}
            print(f"  {Path(f).name:28s} shape {vol.shape}  labels {frac}")
        print("\n  Expected: exactly {1, 2, 3} = Pore, NiO, BZCYYb.")
        print("  A label 0 means RemoveFlaggedFeatures left voxels unassigned ->")
        print("  volume_fractions_3d counts them in the denominator but not the")
        print("  numerator, so every reported fraction is deflated.")

    print("\n" + "=" * 78)
    print("3. PORE GRAINS vs THE 50-VOXEL DELETION THRESHOLD")
    print("=" * 78)
    d_cut_um = 2 * (3 * DELETE_BELOW_VOXELS / (4 * math.pi)) ** (1 / 3) * VOXEL_UM
    print(f"  voxel = {VOXEL_UM} um; a {DELETE_BELOW_VOXELS}-voxel sphere has diameter "
          f"{d_cut_um:.3f} um")
    print(f"  -> the template DELETES every grain smaller than {d_cut_um:.3f} um\n")
    print(f"  {'sample':8s} {'mu':>7s} {'sigma':>7s} {'mean D um':>10s} "
          f"{'% of pore grains BELOW cutoff':>32s}")
    for s, g in df.groupby("sample"):
        mu = g["mu_Pore"].median()
        sg = g["sigma_Pore"].median()
        meand = g["meand_Pore_um"].median()
        # lognormal CDF at the cutoff diameter
        z = (math.log(d_cut_um) - mu) / sg
        below = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        print(f"  {s:8s} {mu:7.3f} {sg:7.3f} {meand:10.3f} {100*below:31.1f}%")
    print("\n  If this is near 100%, DREAM.3D packs the pore grains and the template")
    print("  then deletes almost all of them, handing the space to NiO/BZCYYb.")

    print("\n" + "=" * 78)
    print("4. IS THE TARGET EVEN REPRESENTABLE?")
    print("=" * 78)
    vol_um3 = DIMS[0] * DIMS[1] * DIMS[2] * VOXEL_UM ** 3
    print(f"  synthetic volume: {DIMS} voxels @ {VOXEL_UM} um = {vol_um3:.0f} um^3\n")
    print(f"  {'sample':8s} {'pore target':>12s} {'= um^3':>9s} {'mean grain um^3':>16s} "
          f"{'grains needed':>14s}")
    for s, g in df.groupby("sample"):
        vf = g["vf2d_Pore"].median()
        target = vf * vol_um3
        meand = g["meand_Pore_um"].median()
        gvol = (math.pi / 6) * meand ** 3
        n = target / gvol if gvol > 0 else float("inf")
        flag = "  <-- FEWER THAN 1 GRAIN FITS" if n < 1 else ""
        print(f"  {s:8s} {100*vf:11.2f}% {target:9.2f} {gvol:16.4f} {n:14.0f}{flag}")
    print("\n  If 'grains needed' is a healthy number (tens to hundreds), the target")
    print("  is representable and the loss is happening downstream (see #3).")


if __name__ == "__main__":
    main()
