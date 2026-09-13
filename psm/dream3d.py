"""
DREAM.3D bridge.

Three jobs:
  1. write_pipeline()  -- take the camp template pipeline JSON and inject the
       measured phase fractions + lognormal (mu, sigma) per phase, regenerate
       the StatsGenerator bins, parameterize the hard-coded Windows output
       paths, and set the synthetic volume dimensions / resolution.
  2. run_pipeline()    -- call the DREAM.3D 6.5 PipelineRunner on the JSON.
  3. load_phase_volume() -- read the resulting 3D phase array back into numpy,
       from either the .dream3d (HDF5) file or the ASCII .vtk file.

Why mu / sigma map straight through
------------------------------------
StatsGenerator's "FeatureSize Distribution" -> "Average" and
"Standard Deviation" are the mean and standard deviation of ln(D), i.e. the
lognormal mu and sigma. This was confirmed against the template: for the
"gold" phase (mu=0.7, sigma=0.3) the template's min diameter exp(0.1)=1.105
and bins reproduce exactly from arange(exp(mu-2s), exp(mu+2s), step). So the
(mu, sigma) coming out of psd.py go in here unchanged.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseStats:
    """One phase, as measured by segmentation + PSD."""

    name: str
    phase_fraction: float
    mu: float                # lognormal mean of ln(D)
    sigma: float             # lognormal std of ln(D)
    # StatsGenerator cutoffs, in multiples of sigma. The template used 2-5
    # depending on phase; +/-3 sigma is a reasonable default. Confirm in the
    # StatsGenerator GUI after writing.
    min_cutoff_sigma: float = 3.0
    max_cutoff_sigma: float = 3.0
    num_bins: int = 10


def _regen_size_distribution(stats_obj: dict, ph: PhaseStats) -> None:
    """Rewrite one StatsDataArray phase entry in place for new (mu, sigma).

    Updates Name, PhaseFraction, the FeatureSize Distribution Average/StdDev,
    the BinNumber array, Bin Count and Feature_Diameter_Info, and rebroadcasts
    the per-bin shape-distribution arrays so their lengths stay consistent with
    the new bin count (values are taken from the template's first bin, which
    keeps grains roughly equiaxed as in the original).
    """
    minD = float(np.exp(ph.mu - ph.min_cutoff_sigma * ph.sigma))
    maxD = float(np.exp(ph.mu + ph.max_cutoff_sigma * ph.sigma))
    if not (maxD > minD):                       # degenerate (sigma ~ 0)
        maxD = minD * 1.5 + 1e-3
    bin_step = (maxD - minD) / max(ph.num_bins, 1)
    bins = np.arange(minD, maxD, bin_step)
    if bins.size == 0:
        bins = np.array([minD])
    n_bins = int(bins.size)

    stats_obj["Name"] = ph.name
    stats_obj["PhaseFraction"] = float(ph.phase_fraction)
    stats_obj["FeatureSize Distribution"] = {
        "Average": float(ph.mu),
        "Standard Deviation": float(ph.sigma),
    }
    stats_obj["BinNumber"] = [float(b) for b in bins]
    stats_obj["Bin Count"] = n_bins
    # Feature_Diameter_Info = [binStep, maxDiameter, minDiameter]
    stats_obj["Feature_Diameter_Info"] = [float(bin_step), float(maxD), float(minD)]

    # Keep per-bin shape arrays the right length for n_bins.
    def _broadcast(dist_key: str, array_keys: list[str]) -> None:
        dist = stats_obj.get(dist_key)
        if not isinstance(dist, dict):
            return
        for k in array_keys:
            vals = dist.get(k)
            if isinstance(vals, list) and vals:
                dist[k] = [float(vals[0])] * n_bins

    _broadcast("FeatureSize Vs B Over A Distributions", ["Alpha", "Beta"])
    _broadcast("FeatureSize Vs C Over A Distributions", ["Alpha", "Beta"])
    _broadcast("FeatureSize Vs Omega3 Distributions", ["Alpha", "Beta"])
    _broadcast("FeatureSize Vs Neighbors Distributions",
               ["Average", "Standard Deviation"])


def write_pipeline(
    template_path: str,
    phases: list[PhaseStats],
    output_dir: str,
    out_prefix: str = "microstructure",
    dimensions: tuple[int, int, int] = (100, 100, 50),
    resolution_um: float = 0.1,
    save_path: str | None = None,
) -> str:
    """Create a runnable DREAM.3D pipeline JSON from measured phase stats.

    Returns the path of the written pipeline JSON. All DREAM.3D outputs
    (.dream3d, .vtk, STL files) are routed into `output_dir`.
    """
    with open(template_path, "r") as f:
        pipe = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # --- StatsGenerator (filter "00") -------------------------------------
    sg = pipe["00"]
    stats_arrays = sg["StatsDataArray"]
    # numeric phase keys "1","2","3" in order
    phase_keys = sorted(
        (k for k in stats_arrays if k.isdigit()), key=lambda x: int(x)
    )
    if len(phases) != len(phase_keys):
        raise ValueError(
            f"template has {len(phase_keys)} phases but {len(phases)} provided"
        )
    for key, ph in zip(phase_keys, phases):
        _regen_size_distribution(stats_arrays[key], ph)

    # --- Initialize Synthetic Volume (filter "01") ------------------------
    init = pipe["01"]
    init["Dimensions"] = {"x": dimensions[0], "y": dimensions[1], "z": dimensions[2]}
    init["Resolution"] = {"x": resolution_um, "y": resolution_um, "z": resolution_um}
    init["BoxDimensions"] = (
        f"X Range: 0 to {dimensions[0]*resolution_um} "
        f"(Delta: {dimensions[0]*resolution_um})\n"
        f"Y Range: 0 to {dimensions[1]*resolution_um} "
        f"(Delta: {dimensions[1]*resolution_um})\n"
        f"Z Range: 0 to {dimensions[2]*resolution_um} "
        f"(Delta: {dimensions[2]*resolution_um})"
    )

    # --- parameterize every hard-coded output path -----------------------
    dream3d_out = os.path.join(output_dir, f"{out_prefix}.dream3d")
    vtk_out = os.path.join(output_dir, f"{out_prefix}.vtk")
    stl_dir = os.path.join(output_dir, "stl")
    os.makedirs(stl_dir, exist_ok=True)
    for fid, filt in pipe.items():
        if not isinstance(filt, dict):
            continue
        name = filt.get("Filter_Name")
        if name == "DataContainerWriter":
            filt["OutputFile"] = dream3d_out
        elif name == "VtkRectilinearGridWriter":
            filt["OutputFile"] = vtk_out
        elif name == "WriteStlFile":
            filt["OutputStlDirectory"] = stl_dir

    save_path = save_path or os.path.join(output_dir, f"{out_prefix}_pipeline.json")
    with open(save_path, "w") as f:
        json.dump(pipe, f, indent=4)
    return save_path


def run_pipeline(pipeline_json: str, dream3d_bin_dir: str) -> subprocess.CompletedProcess:
    """Run a pipeline with the DREAM.3D 6.5 PipelineRunner.

    dream3d_bin_dir is the folder holding PipelineRunner(.exe), e.g.
    C:\\DREAM3D\\bin on Windows. Raises if the executable is not found.
    """
    exe = "PipelineRunner.exe" if os.name == "nt" else "PipelineRunner"
    exe_path = os.path.join(dream3d_bin_dir, exe)
    if not os.path.exists(exe_path):
        raise FileNotFoundError(
            f"PipelineRunner not found at {exe_path}. Point dream3d_bin_dir at "
            f"the DREAM.3D 6.5.171 'bin' folder."
        )
    return subprocess.run(
        [exe_path, "-p", pipeline_json],
        capture_output=True, text=True, check=False,
    )


# ----------------------------------------------------------------------------
# Read the synthetic volume back into numpy
# ----------------------------------------------------------------------------
def load_phase_volume_h5(
    dream3d_file: str,
    container: str = "SyntheticVolumeDataContainer",
    array: str = "Phases",
) -> np.ndarray:
    """Load the 3D phase array from a .dream3d (HDF5) file -> (z, y, x) int array."""
    import h5py

    with h5py.File(dream3d_file, "r") as h5:
        dc = h5["DataContainers"][container]
        phases = np.asarray(dc["CellData"][array]).squeeze()
        # geometry dimensions are stored x, y, z
        try:
            dims = np.asarray(dc["_SIMPL_GEOMETRY"]["DIMENSIONS"]).astype(int)
            nx, ny, nz = int(dims[0]), int(dims[1]), int(dims[2])
            vol = phases.reshape((nz, ny, nx))
        except Exception:
            vol = phases  # already shaped
    return vol.astype(np.int32)


def load_phase_volume_vtk(vtk_file: str, array: str = "Phases") -> np.ndarray:
    """Parse an ASCII legacy VTK rectilinear grid (WriteBinaryFile=0).

    Returns the named CELL_DATA scalar as a (z, y, x) int array.
    """
    with open(vtk_file, "r") as f:
        tokens = f.read().split()

    def _read_block(key: str, n: int, start: int) -> tuple[list[float], int]:
        vals, i = [], start
        while len(vals) < n and i < len(tokens):
            try:
                vals.append(float(tokens[i]))
            except ValueError:
                pass
            i += 1
        return vals, i

    dims = None
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "DIMENSIONS":
            dims = (int(tokens[i + 1]), int(tokens[i + 2]), int(tokens[i + 3]))
            i += 4
            continue
        if t == "SCALARS" and i + 1 < len(tokens) and tokens[i + 1] == array:
            # skip "SCALARS name type [ncomp]" + "LOOKUP_TABLE name"
            j = i + 2
            while j < len(tokens) and tokens[j] != "LOOKUP_TABLE":
                j += 1
            j += 2  # past LOOKUP_TABLE + its name
            nx, ny, nz = dims
            ncells = (nx - 1) * (ny - 1) * (nz - 1)  # cell data
            vals, _ = _read_block(array, ncells, j)
            arr = np.asarray(vals).reshape((nz - 1, ny - 1, nx - 1))
            return arr.astype(np.int32)
        i += 1
    raise ValueError(f"could not find SCALARS '{array}' in {vtk_file}")
