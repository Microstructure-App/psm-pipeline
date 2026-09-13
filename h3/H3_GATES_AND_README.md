# H3 — Fidelity study: gates, data, and run instructions

Date: 2026-07-19. Real dataset: `Segmented_LSCF_GDC_Supercrop.tif`
(151 x 283 x 120 voxels, labels {0, 105, 255}), provenance
`USC_Update_10NOV2021.pptx` (Sample 1, pristine LSCF+GDC).

## Gate 1 — intensity-to-phase mapping: PASS

Mapping adopted: 0 = Pore, 105 = LSCF, 255 = GDC. Evidence: (a) the deck's
own segmented 2D image classifies as 58.1% black / 20.5% LSCF-blue
(#0064AF) / 21.4% GDC-green (#247324) against the slide-3 legend, matching
the stack's 61.7 / 21.2 / 17.1 and confirming the high porosity is genuine;
(b) a 50:50 wt% LSCF-GDC composite implies ~53:47 LSCF:GDC by volume
(densities ~6.36 vs ~7.24 g/cm3) — the stack's solid split is 55:45 as
mapped and would be 45:55 if swapped; (c) BSE brightness ordering (GDC has
the higher mean atomic number) puts GDC at 255. Residual caveat: the 50:50
composition needs Dr. Jin's confirmation, but a solid-label swap would only
rename phases — every H3 metric is symmetric under it.

## Gate 2 — voxel dimensions: provisionally resolved at 40 nm isotropic

The Sample 1 3D render (`ppt/media/image3.gif`) carries labeled MATLAB
axes: Y ~ 12 um, X ~ 5 um, Z ~ 6 um. Against 283 / 120 / 151 voxels each
axis independently gives 39.7-42.4 nm, i.e. a near-isotropic ~40 nm voxel.
Adopted: 40 nm nominal, +/-6% band, pending Dr. Jin's confirmation.
Dimensionless metrics (VF, tau, connectivity) are unaffected by the band;
TPB density scales as 1/a^2 (+/-12%).

## Gate 3 — RVE convergence: MIXED (decision required)

Criterion (pre-registered): |m(f=0.8) - m(f=1.0)| / |m(1.0)| < 5% on nested
centered windows; spatial scatter from 8 corner windows at f=0.6. Results
(`rve_sweep.csv`, `rve_verdict.json`):

- Converged: VF_LSCF (1.3%), Conn_Pore (0.03%), Tau_Pore (0%),
  DPB Pore-LSCF (4.2%).
- Not converged: VF_GDC (21.5%), Conn_LSCF (25.3%), Conn_GDC (17.4%),
  TPB density (8.5%), VF_Pore (6.4%, marginal).
- Corner CVs: pore metrics <1-14%; solid metrics 25-48%.

Consequence: the volume is NOT an RVE for solid-phase metrics. Two honest
readings: (i) strict pre-registration -> abandon H3; (ii) descoped design —
because the 2D slices feeding the synthesis come from this same volume,
"does the app's 2D->3D synthesis reproduce the 3D metrics of its own parent
volume" remains internally valid; what dies is generalization to LSCF-GDC
cathodes broadly. Pore-phase claims survive either reading. Decision owner:
Ishveer + Dr. Jin. The paper must state whichever reading is used.

## Real-stack reference metrics (40 nm, tau along z = milling axis)

See `real_reference_metrics.json` (computed with `metrics3d.py` in the
sandbox; `h3_compare.py` recomputes the same quantities with the psm
modules as the authoritative numbers — the two must agree to ~1%).

VF 0.617/0.212/0.171 (Pore/LSCF/GDC); Conn 0.999/0.465/0.449;
Tau(z) 1.000/1.487/1.073, all phases percolate; TPB 8.28 um^-2;
DPB Pore-LSCF 1.74, Pore-GDC 2.29, LSCF-GDC 0.37 um^-1.

## 2D single-slice statistics (`h3_2d_stats.json` / `.csv`)

Slices z = 15, 45, 75, 105, 135; measured with psm.psd.measure_phase, app
defaults (watershed on, h_max = 0.3, min_area_px = 15), 40 nm/px.
Slice-to-slice scatter is large (VF_Pore 0.54-0.72, VF_LSCF 0.14-0.30) —
report this as input variance of the single-image premise.

## KNOWN BUG — pin scikit-image < 0.26

scikit-image 0.26.0's `watershed` is OOM-killed on ordinary inputs (first
hit: slice-45 LSCF mask). 0.24 works. Add `scikit-image<0.26` to
requirements.txt; check the local version before running Stage 3.

## Run order (Windows machine, repo root)

1. Lock the error bounds in `h3_compare.py` (BOUNDS) with Dr. Jin BEFORE
   generating anything. Proposed: VF +/-5 pp abs; TPB +/-30% rel;
   tau +/-20% rel; Conn +/-0.15 abs.
2. Copy `h3_2d_stats.json`, `h3_generate.py`, `h3_compare.py`, and
   `Segmented_LSCF_GDC_Supercrop.tif` into the repo root next to `psm/`.
3. `python h3_generate.py --replicates 5`   (25 runs, ~5.1 M voxels each)
   If a run is slow, drop to `--replicates 3` — 15 volumes still give a
   usable replicate band.
4. `python h3_compare.py` -> `h3_outputs/h3_verdict.csv` is the H3 table.
5. Cross-check `h3_outputs/h3_real_metrics_local.json` against
   `real_reference_metrics.json`; investigate any >1% disagreement.

## Files in this package

- `h3_2d_stats.json`, `h3_2d_stats.csv` — 2D inputs to synthesis
- `h3_generate.py`, `h3_compare.py` — Windows-side scripts
- `h3_extract2d.py` — how the 2D stats were produced (rerunnable locally)
- `metrics3d.py`, `rve_check.py` — sandbox analysis code (provenance)
- `real_reference_metrics.json` — independent cross-check numbers
- `rve_sweep.csv`, `rve_verdict.json` — Gate 3 raw data and verdict
