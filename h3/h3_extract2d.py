"""H3 step 1: extract 2D statistics from single slices of the real stack.

Each chosen z-slice plays the role of "the one cheap 2D SEM image".
Segmentation is trivial (the stack is already labeled), so this isolates
exactly the part of the pipeline H3 tests: 2D PSD statistics -> DREAM.3D
synthesis -> 3D metrics.

Measurement uses psm.psd.measure_phase with app defaults (watershed on,
h_max=0.3, min_area_px=15) so the numbers are the app's numbers.

Output: h3_2d_stats.json  (one record per slice: per-phase VF, mu, sigma,
counts) + h3_2d_stats.csv for eyeballing.
"""
import json
import numpy as np
import pandas as pd

from psm import psd as psdmod

VOX_UM = 0.040                      # deck-derived nominal, see gate 2 notes
SLICES = [15, 45, 75, 105, 135]     # spread across the 151-slice stack
NAMES = {0: "Pore", 1: "LSCF", 2: "GDC"}

vol = np.load("real_vol.npy")

records = []
rows = []
for z in SLICES:
    sl = vol[z]                     # (283, 120) labeled image
    rec = dict(slice=z, voxel_um=VOX_UM,
               height_px=int(sl.shape[0]), width_px=int(sl.shape[1]))
    tot = sl.size
    for p, nm in NAMES.items():
        mask = sl == p
        vf = float(mask.sum()) / tot
        psd = psdmod.measure_phase(mask, VOX_UM, nm)
        rec[f"VF_{nm}"] = vf
        rec[f"mu_{nm}"] = psd.lognormal_mu
        rec[f"sigma_{nm}"] = psd.lognormal_sigma
        rec[f"n_particles_{nm}"] = int(psd.summary["AcceptedFeatureCount"])
        rec[f"meanD_{nm}_um"] = psd.summary["MeanDiameter_um"]
        rows.append(dict(slice=z, phase=nm, VF=vf,
                         mu=psd.lognormal_mu, sigma=psd.lognormal_sigma,
                         n=psd.summary["AcceptedFeatureCount"],
                         meanD_um=psd.summary["MeanDiameter_um"],
                         D50_um=psd.summary["D50_um"]))
    records.append(rec)

json.dump(dict(voxel_um=VOX_UM, slices=SLICES,
               target_dims_xyz=[int(vol.shape[2]), int(vol.shape[1]),
                                int(vol.shape[0])],
               records=records),
          open("h3_2d_stats.json", "w"), indent=1)
df = pd.DataFrame(rows)
df.to_csv("h3_2d_stats.csv", index=False)
print(df.to_string(index=False))
