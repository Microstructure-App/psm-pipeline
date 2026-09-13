"""H3 step 3 (FIXED v2): compare synthetic volumes against the real stack.

Fix vs v1: DREAM.3D writes phases 1-indexed (1=Pore, 2=LSCF, 3=GDC in the
order h3_generate.py listed them; label 0 is reserved). v1 assumed 0-indexed
labels, so every synthetic metric was computed on the wrong phase and GDC
(label 3) was silently dropped. This version remaps labels to 0/1/2 after
loading each VTK, asserts the result is sane, and prints progress.

The 25 volumes in h3_outputs/ are unaffected — do NOT regenerate. Just rerun:

    python h3\\h3_compare.py
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.chdir(HERE)

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from psm import dream3d as d3d
from psm import evaluation as ev
from psm.tortuosity import all_phase_tortuosity
from psm.boundaries import analyze_boundaries

VOX = 0.040
NAMES = {0: "Pore", 1: "LSCF", 2: "GDC"}
OUT = Path("h3_outputs")

# --- pre-registered bounds (locked before generation, 2026-07-19) ----------
BOUNDS = {"VF": 0.05,        # absolute, volume-fraction points
          "TPB": 0.30,       # relative
          "Tau": 0.20,       # relative
          "Conn": 0.15}      # absolute


def remap_dream3d_labels(vol):
    """DREAM.3D ensembles are 1-indexed (0 reserved). Shift to 0-indexed so
    labels line up with NAMES. Safe even if a phase failed to pack: ids are
    preserved, so {1,3} -> {0,2} keeps the right identities."""
    u = np.unique(vol)
    if 0 not in u:
        vol = vol - 1
        u = np.unique(vol)
    assert set(u.tolist()) <= {0, 1, 2}, f"unexpected phase labels: {u}"
    return vol


def metrics(vol):
    row = {}
    vf = ev.volume_fractions_3d(vol, NAMES)
    for n, f in vf.items():
        row[f"VF_{n}"] = f
    for r in all_phase_tortuosity(vol, NAMES, axis=0):
        p = r["Phase"]
        row[f"Tau_{p}"] = float(r.get("Tortuosity", float("nan")))
        row[f"Conn_{p}"] = float(r.get("ConnectivityRatio", float("nan")))
        row[f"Percolates_{p}"] = bool(r.get("Percolates"))
    b = analyze_boundaries(vol, NAMES, voxel_size_um=VOX)
    row["TPB_density_um2"] = float(b.tpb_density_um_per_um3)
    for (x, y), area in b.dpb_area_um2.items():
        row[f"DPB_{x}-{y}_um2"] = float(area)
    return row


def main():
    t0 = time.time()
    print("[1/3] real stack ...", flush=True)
    raw = tifffile.imread("Segmented_LSCF_GDC_Supercrop.tif")
    real = np.zeros_like(raw, dtype=np.uint8)
    real[raw == 105] = 1
    real[raw == 255] = 2
    real_row = metrics(real)
    OUT.mkdir(exist_ok=True)
    json.dump(real_row, open(OUT / "h3_real_metrics_local.json", "w"),
              indent=1)
    print(f"      done ({time.time()-t0:.0f}s)", flush=True)

    manifest = json.load(open(OUT / "h3_manifest.json"))
    rows = []
    for i, m in enumerate(manifest, 1):
        t1 = time.time()
        vol = remap_dream3d_labels(d3d.load_phase_volume_vtk(m["vtk"]))
        r = metrics(vol)
        r.update(slice=m["slice"], rep=m["rep"])
        rows.append(r)
        print(f"[2/3] volume {i}/{len(manifest)} "
              f"(slice {m['slice']} rep {m['rep']}) "
              f"{time.time()-t1:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "h3_synthetic_metrics.csv", index=False)

    print("[3/3] verdict ...", flush=True)
    verdict = []
    for col, kind in [("VF_Pore", "VF"), ("VF_LSCF", "VF"), ("VF_GDC", "VF"),
                      ("TPB_density_um2", "TPB"),
                      ("Tau_Pore", "Tau"), ("Tau_LSCF", "Tau"),
                      ("Tau_GDC", "Tau"),
                      ("Conn_Pore", "Conn"), ("Conn_LSCF", "Conn"),
                      ("Conn_GDC", "Conn")]:
        x = df[col].replace([np.inf, -np.inf], np.nan)
        mean, sd = float(x.mean()), float(x.std())
        n_perc = None
        pcol = col.replace("Tau_", "Percolates_")
        if col.startswith("Tau_") and pcol in df:
            n_perc = int(df[pcol].sum())
        ref = real_row[col]
        if kind in ("VF", "Conn"):
            err, bound = abs(mean - ref), BOUNDS[kind]
        else:
            err = (abs(mean - ref) / abs(ref)
                   if ref and math.isfinite(ref) else float("nan"))
            bound = BOUNDS[kind]
        verdict.append(dict(metric=col, real=round(ref, 4),
                            synth_mean=round(mean, 4),
                            synth_sd=round(sd, 4),
                            n_percolating=n_perc,
                            error=round(err, 4) if math.isfinite(err)
                            else None,
                            bound=bound,
                            within=bool(math.isfinite(err) and err <= bound)))
    vd = pd.DataFrame(verdict)
    vd.to_csv(OUT / "h3_verdict.csv", index=False)
    print(vd.to_string(index=False))
    print(f"Total {time.time()-t0:.0f}s. Wrote h3_outputs/h3_verdict.csv",
          flush=True)


if __name__ == "__main__":
    main()