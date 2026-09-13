"""
Per-phase particle size distribution (PSD).

Python port of STEP 8 of the PhD MATLAB reference:
  - watershed separation of touching particles (distance transform + h-maxima)
  - shape / size rejection (eccentricity, solidity, min/max diameter)
  - lognormal fit -> (mu, sigma) in log space, plus mean particle size
  - experimental PDF (histogram) and D10/D50/D90 percentiles

The lognormal (mu, sigma) come straight out of this module and feed
DREAM.3D StatsGenerator unchanged -- see dream3d.py for why they map
directly to StatsGenerator's "Average" and "Standard Deviation" fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy import stats
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
from psm.matlab_watershed import matlab_watershed_split


@dataclass
class PSDConfig:
    """Filtering knobs. Defaults follow the MATLAB reference."""

    min_area_px: int = 15
    min_diameter_um: float = 0.05
    # maximum diameter (um) to drop unrealistic clumps; inf = no cap
    max_diameter_um: float = float("inf")
    # Shape filter is OFF by default to match the updated MATLAB reference,
    # which rejects on size only. Turn it on to also drop elongated / non-compact
    # objects (the older OFA8_SEM behavior).
    apply_shape_filter: bool = False
    max_eccentricity: float = 0.95   # ~1 == very elongated (only if filter on)
    min_solidity: float = 0.60       # ~1 == compact      (only if filter on)
    num_bins: int = 10
    use_watershed: bool = True
    h_max: float = 0.3               # lower -> more splitting


@dataclass
class PhasePSD:
    name: str
    diameters_um: np.ndarray
    lognormal_mu: float              # mean of ln(D)
    lognormal_sigma: float           # std of ln(D)
    mean_particle_size_um: float     # exp(mu + sigma^2/2)
    summary: dict
    pdf_bin_centers: np.ndarray
    pdf_density: np.ndarray
    fit_x: np.ndarray
    fit_y: np.ndarray
    accepted_mask: np.ndarray


def _watershed_split(mask: np.ndarray, h_max: float) -> np.ndarray:
    """Split touching particles. Delegates to the MATLAB-parity port
    in psm/matlab_watershed.py, which reproduces bwdist -> imhmax ->
    watershed(-D) with explicit regional-minima markers and mask-restricted
    flooding (equivalent output, avoids skimage's slow background plateau)."""
    return matlab_watershed_split(mask, h_max)

def measure_phase(
    mask: np.ndarray,
    um_per_px: float,
    name: str,
    cfg: PSDConfig | None = None,
    max_diameter_um: float | None = None,
    h_max: float | None = None,
) -> PhasePSD:
    """Measure one phase mask -> PhasePSD with stats, PDF and lognormal fit."""
    cfg = cfg or PSDConfig()
    max_d = cfg.max_diameter_um if max_diameter_um is None else max_diameter_um
    hmax = cfg.h_max if h_max is None else h_max

    bw = remove_small_objects(np.asarray(mask, dtype=bool), min_size=cfg.min_area_px)
    particle_mask = _watershed_split(bw, hmax) if cfg.use_watershed else bw
    particle_mask = remove_small_objects(particle_mask, min_size=cfg.min_area_px)

    labels = ndi.label(particle_mask, structure=np.ones((3, 3), dtype=int))[0]
    props = regionprops(labels)

    accepted_labels = np.zeros(int(labels.max()) + 1, dtype=bool)
    diam_um = []
    rows = []
    for p in props:
        d_um = p.equivalent_diameter_area * um_per_px
        area_um2 = p.area * um_per_px ** 2
        keep = (
            d_um >= cfg.min_diameter_um
            and d_um <= max_d
            and p.area >= cfg.min_area_px
        )
        if cfg.apply_shape_filter:
            keep = keep and (
                p.eccentricity <= cfg.max_eccentricity
                and p.solidity >= cfg.min_solidity
            )
        if keep:
            accepted_labels[p.label] = True
            diam_um.append(d_um)
            rows.append(
                dict(
                    Phase=name,
                    EquivalentDiameter_um=d_um,
                    Area_um2=area_um2,
                    Eccentricity=p.eccentricity,
                    Solidity=p.solidity,
                )
            )

    # Vectorized accept mask: O(pixels), not O(n_particles x pixels).
    # The old `accepted[labels == p.label] = True` per-particle scan was the
    # Stage-3 bottleneck that looked like a hang on finely-split masks.
    accepted = accepted_labels[labels]

    diam_um = np.asarray(diam_um, dtype=float)
    n_raw = len(props)
    n_accepted = len(diam_um)

    # ---- lognormal fit (log-space mu/sigma) -------------------------------
    if n_accepted >= 2:
        # floc=0 fixes the location at 0 -> a true 2-parameter lognormal,
        # matching MATLAB lognfit / fitdist(...,'Lognormal'). mu, sigma are the
        # mean and std of ln(D); these feed DREAM.3D StatsGenerator unchanged.
        shape, _loc, scale = stats.lognorm.fit(diam_um, floc=0)
        sigma = float(shape)
        mu = float(np.log(scale))
        # Mean / std of the fitted lognormal, in micrometers (not mu/sigma).
        lognormal_mean = float(np.exp(mu + sigma ** 2 / 2.0))
        lognormal_std = float(
            np.sqrt((np.exp(sigma ** 2) - 1.0) * np.exp(2 * mu + sigma ** 2))
        )
        x = np.linspace(diam_um.min(), diam_um.max(), 300)
        y = stats.lognorm.pdf(x, shape, loc=0, scale=scale)
    else:
        mu = sigma = lognormal_mean = lognormal_std = float("nan")
        x = np.array([])
        y = np.array([])
    # MPS_um in the MATLAB summary is the arithmetic mean equivalent diameter.
    mps = float(np.mean(diam_um)) if n_accepted else float("nan")

    if n_accepted > 0:
        density, edges = np.histogram(diam_um, bins=cfg.num_bins, density=True)
        centers = edges[:-1] + np.diff(edges) / 2.0
    else:
        density = np.array([])
        centers = np.array([])

    summary = dict(
        Phase=name,
        RawFeatureCount=n_raw,
        AcceptedFeatureCount=n_accepted,
        MeanDiameter_um=float(np.mean(diam_um)) if n_accepted else float("nan"),
        MedianDiameter_um=float(np.median(diam_um)) if n_accepted else float("nan"),
        StdDiameter_um=float(np.std(diam_um, ddof=1)) if n_accepted > 1 else float("nan"),
        D10_um=float(np.percentile(diam_um, 10)) if n_accepted else float("nan"),
        D50_um=float(np.percentile(diam_um, 50)) if n_accepted else float("nan"),
        D90_um=float(np.percentile(diam_um, 90)) if n_accepted else float("nan"),
        Lognormal_mu=mu,
        Lognormal_sigma=sigma,
        LognormalMean_um=lognormal_mean,
        LognormalStd_um=lognormal_std,
        MPS_um=mps,
    )
    return PhasePSD(
        name=name,
        diameters_um=diam_um,
        lognormal_mu=mu,
        lognormal_sigma=sigma,
        mean_particle_size_um=mps,   # arithmetic mean diameter (MATLAB MPS_um)
        summary=summary,
        pdf_bin_centers=centers,
        pdf_density=density,
        fit_x=x,
        fit_y=y,
        accepted_mask=accepted,
    )


def summary_table(phase_psds: list[PhasePSD], vol_fractions: dict) -> pd.DataFrame:
    """Build the per-phase summary DataFrame (one row per phase)."""
    rows = []
    for psd in phase_psds:
        row = dict(psd.summary)
        row["VolumeFraction"] = vol_fractions.get(psd.name, float("nan"))
        rows.append(row)
    cols = [
        "Phase", "VolumeFraction", "RawFeatureCount", "AcceptedFeatureCount",
        "MeanDiameter_um", "MedianDiameter_um", "StdDiameter_um",
        "D10_um", "D50_um", "D90_um",
        "Lognormal_mu", "Lognormal_sigma", "MPS_um",
    ]
    return pd.DataFrame(rows)[cols]