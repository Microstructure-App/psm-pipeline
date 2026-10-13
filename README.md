# PCFC Electrode Microstructure Pipeline

A Python application that takes an SEM image of a proton-conducting fuel-cell
(PCFC) composite electrode and runs the full analysis chain:

1. **Segmentation** of phases by manual intensity thresholds (any number of
   named phases; pore/NiO/BZCYYb for the standard 3-phase electrode).
2. **Particle size distribution** + volume fractions, with a lognormal fit per phase.
3. **DREAM.3D** pipeline export pre-filled with the measured statistics, which
   synthesizes a statistically equivalent 3D voxel volume (and STL files).
4. **3D structure analysis** of the synthesized volume: geometric tortuosity,
   percolation, double-phase-boundary (DPB) interfacial area, and
   triple-phase-boundary (TPB) density.

This is a Python port of the lab's MATLAB reference (manual histogram
thresholding + watershed PSD, per PDF_Matlab__updated.m) plus the camp
connectivity/tortuosity scripts, restructured into a reusable package with both
a Streamlit app and a CLI. `segmentation.py` can also auto-suggest threshold
ranges (multi-Otsu or GMM) that you then refine by hand.

---

## Project layout

```
psm_pipeline/
├── psm/
│   ├── segmentation.py   # SEM -> phase map by manual intensity thresholds
│   ├── psd.py            # watershed split, shape filter, lognormal fit
│   ├── dream3d.py        # write StatsGenerator JSON, run pipeline, load volume
│   ├── tortuosity.py     # BFS geometric tortuosity + percolation (2D/3D)
│   └── boundaries.py     # DPB interfacial area + TPB density (3D)
├── app.py                # Streamlit GUI (run end to end in the browser)
├── run_cli.py            # command-line entry point
└── requirements.txt
```

Drop the camp template pipeline JSON
(`3phasemicrostructure_stl_removeisolated.json`) into this folder.

---

## Setup in VS Code (Windows)

1. **Install Python 3.10–3.12** from python.org (check "Add to PATH").
2. **Install VS Code** and the Microsoft **Python** extension.
3. Open this folder in VS Code: `File > Open Folder...`.
4. Open a terminal (`` Ctrl+` ``) and create a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
   On macOS/Linux the activate line is `source .venv/bin/activate`.
5. Select the interpreter: `Ctrl+Shift+P` -> "Python: Select Interpreter" ->
   pick the `.venv` one.

---

## Running it

### Streamlit app (recommended for the PhD students)
```
streamlit run app.py
```
A browser tab opens. Go top to bottom: upload an SEM image, enter the scale-bar
length (measure it in pixels in any image viewer), set each phase's intensity
range on the histogram (or click "Auto-suggest"), run segmentation, compute the
PSD, export the DREAM.3D pipeline, then later load the DREAM.3D output volume for
the 3D analysis.

### Command line
```
# Segmentation (manual thresholds) + PSD + write a DREAM.3D pipeline:
python run_cli.py analyze --image OFA8_SEM.png --scale-um 30 --scale-px 200 ^
    --thresholds "Pore:0-40,NiO:60-130,BZCYYb:150-255" ^
    --template 3phasemicrostructure_stl_removeisolated.json --out results ^
    --dims 100 100 50 --voxel 0.1

# ...or let multi-Otsu suggest the ranges for you to refine:
python run_cli.py analyze --image OFA8_SEM.png --scale-um 30 --scale-px 200 ^
    --auto "Pore,NiO,BZCYYb" --out results

# (optional) also run DREAM.3D automatically:
#   add  --dream3d-bin C:\DREAM3D\bin

# Analyze the synthesized 3D volume:
python run_cli.py volume --input results/microstructure.dream3d --voxel 0.1 --axis 0
```

### Running DREAM.3D by hand
`write_pipeline` routes every output into your chosen folder, so the old
hard-coded Windows paths are gone. Run the generated JSON with the DREAM.3D
6.5.171 `PipelineRunner`:
```
C:\DREAM3D\bin\PipelineRunner.exe -p results\microstructure_pipeline.json
```
Open the resulting `.xdmf` (not the raw `.dream3d`) in ParaView and color by the
`Phases` array, as before.

---

## How the pieces connect to DREAM.3D StatsGenerator

The lognormal `(mu, sigma)` from `psd.py` feed StatsGenerator **unchanged**.
StatsGenerator's `FeatureSize Distribution` -> `Average` / `Standard Deviation`
are the mean and standard deviation of `ln(D)` (i.e. lognormal mu and sigma).
This was confirmed against the template: for the original "gold" phase
(mu = 0.7, sigma = 0.3) the bins reproduce exactly from
`arange(exp(mu - 2*sigma), exp(mu + 2*sigma), step)`, giving min diameter
`exp(0.1) = 1.105`. So no unit conversion is needed between the PSD fit and the
pipeline.

`dream3d.py` regenerates `BinNumber`, `Bin Count` and `Feature_Diameter_Info`
from `(mu, sigma)` using configurable `min_cutoff_sigma` / `max_cutoff_sigma`
(default ±3σ) and rebroadcasts the per-bin shape arrays so their lengths stay
consistent. **Open the generated pipeline in the StatsGenerator GUI once to eyeball
the curves** — the per-phase sigma cutoffs in the original template were hand-set
(2σ, 4σ, 5σ for the three phases), so the ±3σ default is a sane starting point,
not a guaranteed match.

---

## What is tested vs. what needs your machine

Tested here on synthetic data and verified numerically:
- Threshold segmentation (manual + multi-Otsu suggest) and volume fractions.
- Watershed PSD, lognormal fit, summary table and PDF plots.
- DREAM.3D JSON writer (valid pipeline, consistent bin/array lengths, paths).
- Tortuosity BFS (matches the camp MATLAB `estimateTortuosity` behavior).
- TPB/DPB on a hand-checkable case and a synthetic 3D volume.
- ASCII VTK reader round-trip.

Needs your environment to fully exercise:
- **`PipelineRunner`** must be installed (DREAM.3D 6.5.171) to actually
  synthesize the volume.
- **`.dream3d` (HDF5) reading** uses the standard 6.x layout
  (`/DataContainers/SyntheticVolumeDataContainer/CellData/Phases`). If your
  output uses different container/array names, pass them to
  `load_phase_volume_h5(...)`, or read the ASCII `.vtk` instead.
- A **real SEM image with a scale bar** for the actual segmentation. The GMM
  thresholds adapt automatically, but check the segmentation overlay and adjust
  `max_expected_pore_fraction` to your real pore content.

## A note on the tortuosity definition

`geometric_tortuosity` returns the **shortest** connected path from the entry
face to the exit face divided by the straight distance, exactly as the camp
MATLAB does. On the 5×5 teaching matrix this gives τ = 1.5 (shortest path = 6
steps)— the slide traces one specific illustrative path that is not the minimum. The slide itself notes you should use
the shortest valid path, which is what this code computes.
