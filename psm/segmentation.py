"""
SEM image -> phase segmentation by manual intensity thresholds.

Updated to match the lab's current MATLAB reference (PDF_Matlab__updated.m):
segmentation is now manual, histogram-based thresholding over an arbitrary
number of user-named phases, in 8-bit intensity space (0-255). Each phase is
an intensity range [low, high]; pixels outside every range are "unclassified"
and reported separately (not reassigned).

For convenience this module can also *suggest* starting thresholds
automatically (multi-Otsu, or the older GMM approach), which the user then
refines by hand -- the standard SEM practice.

Typical PCFC electrode ordering by brightness:
    Pore   = darkest range
    NiO    = mid (dark solid)
    BZCYYb = brightest range
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skimage import color, io
from skimage.filters import threshold_multiotsu


@dataclass
class PhaseThreshold:
    """One named phase defined by an 8-bit intensity range [low, high]."""
    name: str
    low: int          # 0-255 inclusive
    high: int         # 0-255 inclusive


@dataclass
class SegmentationResult:
    phase_id: np.ndarray                 # uint16; 0 = unclassified, 1..N = phases
    gray8: np.ndarray                    # the 8-bit ROI actually analyzed
    thresholds: list                     # list[PhaseThreshold] in phase order
    masks: dict = field(default_factory=dict)   # name -> bool mask
    unclassified_mask: np.ndarray = None


def read_sem_as_gray8(path: str) -> np.ndarray:
    """Robust SEM reader -> uint8 grayscale in [0, 255].

    Handles grayscale, RGB, RGBA / extra channels, and float images. Mirrors
    readSEMasGray8 in the MATLAB reference.
    """
    img = io.imread(path)
    if img.ndim == 3:
        if img.shape[2] >= 3:
            gray = color.rgb2gray(img[:, :, :3])      # float [0,1]
        else:
            gray = img[:, :, 0]
    else:
        gray = img

    if gray.dtype == np.uint8:
        return gray
    g = gray.astype(float)
    if g.min() < 0 or g.max() > 1:
        g = (g - g.min()) / (np.ptp(g) + 1e-12)         # mat2gray
    return (g * 255.0).round().astype(np.uint8)


def microns_per_pixel(scale_bar_pixels: float, scale_bar_um: float) -> float:
    """Scale-bar length in pixels + physical length -> um/pixel."""
    if scale_bar_pixels <= 0:
        raise ValueError("scale_bar_pixels must be positive")
    return scale_bar_um / float(scale_bar_pixels)


def crop_roi(gray8: np.ndarray, roi=None) -> np.ndarray:
    """Crop to roi = (row0, col0, height, width). None -> full image."""
    if roi is None:
        return gray8
    r0, c0, h, w = (int(round(v)) for v in roi)
    return gray8[r0 : r0 + h, c0 : c0 + w]


def check_threshold_ranges(thresholds: list) -> tuple[bool, str]:
    """Validate ranges: within 0-255, low<=high, and no overlap between phases."""
    for t in thresholds:
        if not (0 <= t.low <= 255 and 0 <= t.high <= 255):
            return False, "Thresholds must be between 0 and 255."
        if t.low > t.high:
            return False, f"Phase '{t.name}': low must be <= high."
    for i in range(len(thresholds)):
        for j in range(i + 1, len(thresholds)):
            a, b = thresholds[i], thresholds[j]
            if a.low <= b.high and b.low <= a.high:
                return False, f"Phases '{a.name}' and '{b.name}' ranges overlap."
    return True, ""


def segment_by_thresholds(
    gray8: np.ndarray, thresholds: list, fill_unclassified: bool = False
) -> SegmentationResult:
    """Assign each pixel to the phase whose [low, high] range contains it.

    phase_id is 0 where no range matches (unclassified). If fill_unclassified
    is True, those pixels are assigned to the nearest classified pixel's phase
    (useful when a gap-free map is needed, e.g. 2-D boundary checks).
    """
    ok, msg = check_threshold_ranges(thresholds)
    if not ok:
        raise ValueError(msg)

    phase_id = np.zeros(gray8.shape, dtype=np.uint16)
    masks = {}
    for p, t in enumerate(thresholds, start=1):
        m = (gray8 >= t.low) & (gray8 <= t.high)
        phase_id[m] = p
        masks[t.name] = m
    unclassified = phase_id == 0

    if fill_unclassified and unclassified.any():
        from scipy import ndimage as ndi
        classified = phase_id > 0
        _, (ir, ic) = ndi.distance_transform_edt(~classified, return_indices=True)
        phase_id[unclassified] = phase_id[ir[unclassified], ic[unclassified]]
        masks = {t.name: phase_id == p for p, t in enumerate(thresholds, start=1)}
        unclassified = np.zeros_like(unclassified)

    return SegmentationResult(
        phase_id=phase_id, gray8=gray8, thresholds=list(thresholds),
        masks=masks, unclassified_mask=unclassified,
    )


def volume_fractions(result: SegmentationResult) -> dict:
    """Area / volume fraction of each phase plus the unclassified remainder."""
    total = result.phase_id.size
    vf = {
        t.name: float(np.count_nonzero(result.phase_id == p)) / total
        for p, t in enumerate(result.thresholds, start=1)
    }
    vf["Unclassified"] = float(np.count_nonzero(result.unclassified_mask)) / total
    return vf


# ----------------------------------------------------------------------------
# Optional automatic threshold *suggestions* (refine by hand afterward)
# ----------------------------------------------------------------------------
def suggest_thresholds_multiotsu(gray8: np.ndarray, names: list) -> list:
    """Propose intensity ranges by multi-Otsu (len(names) classes)."""
    n = len(names)
    if n < 2:
        return [PhaseThreshold(names[0], 0, 255)]
    cuts = threshold_multiotsu(gray8, classes=n).astype(int)
    edges = [0, *cuts.tolist(), 255]
    out = []
    for i, name in enumerate(names):
        low = edges[i] if i == 0 else edges[i] + 1
        out.append(PhaseThreshold(name, int(low), int(edges[i + 1])))
    return out


def suggest_thresholds_gmm(gray8: np.ndarray, names: list, random_state: int = 1) -> list:
    """Propose ranges via a Gaussian mixture (the older auto approach).

    Components are sorted by mean intensity and split at the midpoints between
    adjacent component means; ranges are then handed back for manual tweaking.
    """
    from sklearn.mixture import GaussianMixture
    n = len(names)
    if n < 2:
        return [PhaseThreshold(names[0], 0, 255)]
    gm = GaussianMixture(n_components=n, n_init=5, random_state=random_state)
    gm.fit(gray8.reshape(-1, 1).astype(float))
    means = np.sort(gm.means_.ravel())
    cuts = ((means[:-1] + means[1:]) / 2.0).round().astype(int)
    edges = [0, *cuts.tolist(), 255]
    out = []
    for i, name in enumerate(names):
        low = edges[i] if i == 0 else edges[i] + 1
        out.append(PhaseThreshold(name, int(low), int(edges[i + 1])))
    return out
