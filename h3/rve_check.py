"""Gate 3: RVE convergence check on the real LSCF-GDC stack.

Two probes:
  A) nested centered subvolumes at size fractions f of each axis
     -> does each metric converge toward the full-volume value?
  B) 8 corner subvolumes at f=0.6 -> spatial scatter (heterogeneity).

Pre-registered convergence criterion (per metric): |m(0.8) - m(1.0)| /
|m(1.0)| < 0.05  -> converged.  Tortuosity uses tau values only when finite
in every window; percolation loss in any window is reported explicitly.
"""
import numpy as np, json
import pandas as pd
from metrics3d import full_report

vol = np.load("real_vol.npy")
names = {0: "Pore", 1: "LSCF", 2: "GDC"}
VOX = 0.040
Z, Y, X = vol.shape

rows = []
for f in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    dz, dy, dx = int(Z * f), int(Y * f), int(X * f)
    z0, y0, x0 = (Z - dz) // 2, (Y - dy) // 2, (X - dx) // 2
    sub = vol[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
    rep = full_report(sub, names, VOX)
    rep["window"] = f"center_f={f}"
    rep["frac"] = f
    rows.append(rep)

# corner windows at f=0.6
f = 0.6
dz, dy, dx = int(Z * f), int(Y * f), int(X * f)
for i, (z0, y0, x0) in enumerate([(0, 0, 0), (0, 0, X - dx), (0, Y - dy, 0),
                                  (0, Y - dy, X - dx), (Z - dz, 0, 0),
                                  (Z - dz, 0, X - dx), (Z - dz, Y - dy, 0),
                                  (Z - dz, Y - dy, X - dx)]):
    sub = vol[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx]
    rep = full_report(sub, names, VOX)
    rep["window"] = f"corner{i}_f={f}"
    rep["frac"] = f
    rows.append(rep)

df = pd.DataFrame(rows)
df.to_csv("rve_sweep.csv", index=False)

full = df[df.window == "center_f=1.0"].iloc[0]
w08 = df[df.window == "center_f=0.8"].iloc[0]
metrics = ["VF_Pore", "VF_LSCF", "VF_GDC", "TPB_density_um2",
           "Conn_Pore", "Conn_LSCF", "Conn_GDC",
           "Tau_Pore", "Tau_LSCF", "Tau_GDC",
           "DPB_Pore-LSCF_um1", "DPB_Pore-GDC_um1", "DPB_LSCF-GDC_um1"]
verdict = {}
for m in metrics:
    a, b = w08[m], full[m]
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        verdict[m] = dict(rel_change=None, converged=False, note="non-finite")
        continue
    rc = abs(a - b) / abs(b)
    verdict[m] = dict(rel_change=round(float(rc), 4), converged=bool(rc < 0.05))

corners = df[df.window.str.startswith("corner")]
scatter = {m: dict(mean=float(corners[m].replace([np.inf], np.nan).mean()),
                   cv=float(corners[m].replace([np.inf], np.nan).std()
                            / abs(corners[m].replace([np.inf], np.nan).mean()))
                   if corners[m].replace([np.inf], np.nan).mean() else None)
           for m in metrics}
json.dump(dict(convergence=verdict, corner_scatter_f06=scatter),
          open("rve_verdict.json", "w"), indent=1)
print(json.dumps(verdict, indent=1))
print("\ncorner CV (f=0.6):")
for m in metrics:
    print(f"  {m:24s} cv={scatter[m]['cv']}")
