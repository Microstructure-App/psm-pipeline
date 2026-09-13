# PCFC Electrode Microstructure Pipeline — User Manual

A free, lightweight desktop app that turns a **2D SEM image** of a proton-conducting
fuel cell (PCFC) electrode into **3D microstructure metrics**: phase volume
fractions, particle-size distributions, tortuosity, double-phase-boundary (DPB)
interfacial areas, and triple-phase-boundary (TPB) density.

This manual is written to be handed to a new lab member with no prior context.

---

## 1. What the app is (and what it is not)

**It is** a Python port of Dr. Jin's MATLAB microstructure workflow, wrapped in a
click-through interface. You give it a segmented SEM image; it gives you the same
statistics the MATLAB pipeline produced, plus tortuosity, boundary metrics, and 3D
renders.

**It is not** a 3D imaging tool. It never sees a real 3D scan. Instead it measures
2D statistics and asks DREAM.3D to *synthesize* a 3D volume with the same
statistics. Every 3D number therefore inherits a **2D->3D limitation** that must be
stated when you report results — a known, intentional method constraint, not a bug.

---

## 2. Installing and running

### Development mode (day to day)

1. Open the project in VS Code with the virtual environment activated.
2. From the project root:
   ```bash
   streamlit run app.py
   ```
3. The app opens in a browser tab (or the pywebview window if packaged).

The "i icon" help buttons use `st.popover`, which needs **Streamlit >= 1.32**; on
older versions the app falls back to expanders automatically.

### DREAM.3D (needed only for Stage 4's one-click run)

- DREAM.3D **6.5.171** legacy Windows build, unzipped to a short path such as
  `C:\DREAM3D\`.
- The runner is `PipelineRunner.exe`. The legacy zip is **flat — there is no
  `bin/` subfolder**. The correct path looks like:
  ```
  C:\DREAM3D\DREAM3D-6.5.171-Win64\PipelineRunner.exe
  ```
- The app remembers this path in `~/.psm_pipeline_settings.json`, so you set it once.

### Packaging (end-of-project only)

Distribution target is a single `.exe` via **pywebview + PyInstaller**. This is
end-of-project work and does not affect day-to-day development. Two habits keep
packaging painless: never hardcode paths to bundled files (use a `resource_path()`
helper at package time), and keep user-installation paths as configurable settings
(already done via the settings file above).

---

## 3. The stages

The left sidebar is a **map**. Click any stage to jump to it; the arrow marks where
you are and a check marks completed pipeline stages. Loaded data persists as you
move, so you can revisit an earlier stage without redoing work. Each stage also has
**Back / Next** buttons at the bottom.

### Stage 1 - Setup & Calibration
- **Scale bar (um):** the physical length printed on the SEM image.
- **Scale bar (pixels):** measure that same bar in pixels in any image viewer.
- Together these set **um/pixel**, used by every downstream size.
- Upload the SEM image. **Pre-crop** the black border, scale bar, and text out of
  the image first — a stray black border biases thresholding badly. An optional
  in-app crop is available if you didn't crop beforehand.

### Stage 2 - Phase Segmentation
- Choose the number of phases — usually **3** for a PCFC electrode (Pore, the
  electronic conductor NiO/Ni, and the proton conductor BZCYYb). The DREAM.3D
  template expects exactly 3.
- Set an intensity band (0 = black ... 255 = white) for each phase. Pixels outside
  every band are tracked as **Unclassified**, not silently reassigned.
- **Auto-suggest (multi-Otsu)** gives a starting guess only — verify it against the
  histogram.
- **Sanity check:** a realistic pore volume fraction is ~20-40%. If pore reads near
  zero the border/scale bar leaked in or the Pore band is wrong. The app warns you
  when pore < 5%.

### Stage 3 - Particle Sizes
- Measures the particle-size distribution per phase and fits a **lognormal (mu,
  sigma)** in log-space. These feed DREAM.3D.
- **Watershed** splits touching particles — a known source of difference vs.
  MATLAB, so record whether it was on or off.
- **Connectivity caveat:** scikit-image labels with 4-connectivity by default;
  MATLAB's `bwconncomp` defaults to 8-connectivity, producing small particle-count
  differences.
- If a phase has too few particles to fit, mu/sigma come back **NaN** and the
  generator falls back to template defaults. The app warns you when this happens —
  watch for it, because it can hide a bad segmentation.

### Stage 4 - Generate 3D Structure
- DREAM.3D takes the **statistics** (phase fraction + mu/sigma), *not* the image,
  and synthesizes a statistically equivalent 3D volume.
- **Option A** exports the pipeline JSON to run yourself.
- **Option B** runs DREAM.3D headlessly in one click and loads the result straight
  into Stage 5. Set the `PipelineRunner.exe` path once (remembered afterward).
- The VTK output writes to `%TEMP%\microstructure.vtk` and is overwritten each run.
  For runs worth keeping, copy them out or point to a persistent `outputs/` folder.

### Stage 5 - 3D Analysis
- **Tortuosity** along the chosen transport axis. The volume is stored as
  **(z, y, x)**, so axis 0 = z. (Dr. Jin's MATLAB uses (x, y, z); the indexing
  difference is documented in `visualize3d.py`.)
- **DPB** interfacial areas per phase pair, and **TPB** density (voxel-edge method).
- **3D visualization** (needs exactly 3 phases): a three-phase voxel scatter and a
  2x2 boundary figure with voxel counts. Controls: *max scatter points per phase*,
  and boundary-figure detail (*coarse / medium / fine* — higher is slower).
- **Two TPB numbers:** the density (voxel-edge, `boundaries.py`) and the boundary
  render's count (3x3x3 voxel-count, `visualize3d.py`) use different methods —
  cross-validation references, not identical values.

---

## 4. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Pore fraction ~2% | Border/scale bar in ROI, or wrong Pore band | Re-crop; adjust Pore threshold in Stage 2 |
| DREAM.3D "could not read pipeline file" | NaN mu/sigma written to JSON | Fix segmentation so phases have enough particles; the generator guards NaN with `math.isfinite` |
| "PipelineRunner not found" | Wrong path / expecting a `bin/` folder | Use the flat path `...\DREAM3D-6.5.171-Win64\PipelineRunner.exe` |
| Particle counts differ from MATLAB | 4- vs 8-connectivity, or watershed settings | Note the settings; expected difference |
| 3D render says "requires exactly 3 phases" | Volume has != 3 labels | Segment into 3 phases before generating |
| No "i icon" buttons, expanders instead | Streamlit < 1.32 | Upgrade Streamlit for popovers (optional) |

---

## 5. Glossary

- **PSD** — particle size distribution.
- **Lognormal mu, sigma** — parameters of the fitted size distribution, in log-space.
- **Volume fraction** — fraction of classified pixels assigned to a phase.
- **Tortuosity** — how winding a transport path is through a phase (>= 1).
- **DPB** — double-phase boundary; interfacial area between two phases.
- **TPB** — triple-phase boundary; where all three phases meet (electrochemically
  active sites).
- **DREAM.3D StatsGenerator** — builds a synthetic 3D volume from phase statistics.
- **VTK / HDF5** — file formats the 3D volume is read from.

---

## 6. Known limitations (state these when reporting)

1. **2D->3D:** all 3D metrics come from a synthetic volume built from 2D statistics.
2. **Watershed and connectivity** settings affect particle counts.
3. **TPB method mismatch** between the voxel-edge density and the 3x3x3 voxel count.
4. The synthetic volume is *statistically* equivalent, not a reconstruction of the
   specific imaged region.

---

## 11. Stage 6 — Final Evaluation (new)

Stage 6 answers the question the whole pipeline exists for: **is this a good
electrode microstructure?** It shows a scorecard where every metric has:

- a **plain-language definition** (what the number physically means),
- its **value** for the current sample,
- the **criterion** it is judged against, and
- a status: ✅ good · 🟡 acceptable · ❌ poor.

An overall verdict banner summarizes the card: *Good* (all green),
*Acceptable with caveats* (no reds), or *Needs attention* (any red).

### Default criteria (literature-derived, editable)

| Metric | Good | Acceptable | Rationale |
|---|---|---|---|
| Pore volume fraction | 0.25–0.40 | 0.15–0.50 | Composite electrodes are sintered to ~30–40% porosity; below ~15% gas transport chokes, above ~50% the solid networks weaken. |
| Percolation (each phase) | yes | — | A phase with no connected path across the transport axis conducts nothing at all. |
| Geometric tortuosity τ | ≤ 2.0 | ≤ 3.0 | Well-connected electrode phases typically measure τ ≈ 1.5–3 by shortest-path methods. |
| Connectivity ratio | ≥ 0.90 | ≥ 0.50 | Voxels outside the largest cluster are isolated, dead volume. |
| TPB density (voxel-edge) | ≥ 2.0 µm⁻² | ≥ 0.5 µm⁻² | Reported active TPB densities for Ni-YSZ anodes span ~1–10 µm⁻². |

All bands are editable in the **Evaluation criteria & calibration** expander
and persist between sessions (stored in `~/.psm_pipeline_settings.json`).

### Calibration after MATLAB cross-validation

If comparing against Dr. Jin's MATLAB pipeline reveals a *systematic* ratio
between the two implementations, set the **τ calibration factor** and/or the
**TPB calibration factor**. They multiply the reported values before judging
and before saving to the comparison table. Any non-1.0 factor must be
documented in the paper's methods section.

### Sample comparison (OFA8 / OFA10 / OFA11)

Run each electrode's image through Stages 1–5 (pick its Sample ID in Stage 1),
then click **Save results** in Stage 6. Saved rows persist in
`outputs/sample_results.json`, survive restarts, and can be overwritten by
re-saving the same ID — so results remain correctable after validation.
Download the full table as `OFA_sample_comparison.csv` for the paper.

---

## 12. Interactive 3D views (Stage 5, updated)

The 3D renders are now **interactive Plotly figures**: drag to rotate, scroll
to zoom, hover for voxel coordinates. Each plot has its own data table directly
underneath it, so every interfacial area and voxel count sits next to the
picture it belongs to.

The original static matplotlib renderer is still available via the *Static
matplotlib render* checkbox — use it to save publication PNGs in the exact
style of the MATLAB reference figures.

Reminder that applies to both renderers: the **triple-phase voxel count**
(3×3×3 neighborhood, from `boundarydef2.m`) and the **TPB density**
(voxel-edge method, `psm/boundaries.py`) are different definitions and are
cross-validation references — report one and cite the other as a cross-check.

---

## 13. Security notes for the packaged app

- The Streamlit server binds to **127.0.0.1 only** (`.streamlit/config.toml`),
  so the app is never reachable from other machines on the lab network.
- Usage telemetry is disabled; nothing leaves the machine.
- Stage 4 refuses to execute any program that is not named
  `PipelineRunner.exe`, even if the settings file is edited to point elsewhere.
- All outputs are written to the local `outputs/` folder next to the app, with
  sample names sanitized before being used in file names.
- When building the `.exe` with PyInstaller, include `.streamlit/config.toml`
  and the `outputs/` folder logic; do not add `--console=false` debugging
  backdoors, and sign the executable if the lab distributes it.
