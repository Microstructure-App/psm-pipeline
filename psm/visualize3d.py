"""
3D visualization and boundary cross-validation for synthetic phase volumes.

This module ports two of Dr. Jin's MATLAB reference scripts into Python so the
Streamlit app can render the DREAM.3D output inline (no MATLAB / ParaView round
trip):

  * volumetricplot2.m  -> plot_three_phase()
        Three-phase voxel scatter render (Ni/YSZ/Pore -> NiO/BZCYYb/Pore here).

  * boundarydef2.m     -> plot_boundary_subplots() and the boundary detectors.
        2x2 subplot figure: transparent phase isosurfaces + boundary point
        clouds for each two-phase pair and the triple-phase boundary.

Two important notes carried over from the handoff:

1. Coordinate convention. Dr. Jin's MATLAB indexes ``phaseVolume(x, y, z)``.
   The app's volume (from ``d3d.load_phase_volume_vtk``) is indexed
   ``vol[z, y, x]``. The boundary math is symmetric across the three axes, so
   the detectors port directly; only the *plotting* needs to map array axes to
   display axes, which the helpers below handle explicitly.

2. TPB method. ``detect_triple_phase_boundary`` here uses Dr. Jin's 3x3x3
   neighborhood-convolution definition (a voxel is a TPB voxel if its 27-voxel
   neighborhood contains all three phases). This is DIFFERENT from the
   voxel-edge line-length method in ``psm.boundaries``. The two give different
   numbers by construction. Use the counts from this module as a *cross-
   validation reference* alongside the boundaries-module TPB density, and state
   the method choice in the abstract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage as ndi

Color = Tuple[float, float, float]

# ----------------------------------------------------------------------------
# Phase -> color mapping
# ----------------------------------------------------------------------------
# Dr. Jin's palette: metal = red, electrolyte = blue, pore = light cyan-gray.
# We map by phase *role* (keyword in the name) so the colors stay semantically
# correct no matter what integer label DREAM.3D assigns to each phase. Anything
# unrecognized falls back to a per-label palette.

_METAL_COLOR: Color = (1.00, 0.00, 0.00)   # red    (Ni / NiO)
_ELECTRO_COLOR: Color = (0.00, 0.00, 1.00)  # blue   (YSZ / BZCYYb)
_PORE_COLOR: Color = (0.75, 0.85, 0.85)    # light  (pore)
_FALLBACK_PALETTE: Sequence[Color] = (
    (0.85, 0.10, 0.10), (0.10, 0.30, 0.85), (0.70, 0.80, 0.80),
    (0.20, 0.65, 0.20), (0.85, 0.55, 0.10), (0.55, 0.20, 0.75),
)

# Boundary-pair / triple colors (from boundarydef2.m).
BOUNDARY_COLORS: Dict[str, Color] = {
    "metal-electro": (0.85, 0.00, 0.00),   # dark red
    "metal-pore": (0.00, 0.65, 0.00),      # green
    "electro-pore": (0.45, 0.00, 0.85),    # purple
    "triple": (1.00, 0.65, 0.00),          # orange
}


def _role_of(name: str) -> str:
    """Classify a phase name into 'metal' | 'electro' | 'pore' | 'other'."""
    n = name.lower()
    if "pore" in n or "void" in n:
        return "pore"
    if "ni" in n or "metal" in n:        # Ni, NiO
        return "metal"
    if "ysz" in n or "bzcy" in n or "electro" in n or "ceria" in n or "bz" in n:
        return "electro"
    return "other"


def phase_colors(names: Dict[int, str]) -> Dict[int, Color]:
    """Assign a display color to each phase label based on its role."""
    colors: Dict[int, Color] = {}
    fallback_i = 0
    for lab in sorted(names):
        role = _role_of(names[lab])
        if role == "metal":
            colors[lab] = _METAL_COLOR
        elif role == "electro":
            colors[lab] = _ELECTRO_COLOR
        elif role == "pore":
            colors[lab] = _PORE_COLOR
        else:
            colors[lab] = _FALLBACK_PALETTE[fallback_i % len(_FALLBACK_PALETTE)]
            fallback_i += 1
    return colors


def _pair_color(role_a: str, role_b: str) -> Color:
    roles = {role_a, role_b}
    if roles == {"metal", "electro"}:
        return BOUNDARY_COLORS["metal-electro"]
    if roles == {"metal", "pore"}:
        return BOUNDARY_COLORS["metal-pore"]
    if roles == {"electro", "pore"}:
        return BOUNDARY_COLORS["electro-pore"]
    return (0.30, 0.30, 0.30)


# ----------------------------------------------------------------------------
# Boundary detection (ports of boundarydef2.m helpers)
# ----------------------------------------------------------------------------
def detect_two_phase_boundary(vol: np.ndarray, pa: int, pb: int) -> np.ndarray:
    """Face-contact (6-neighbor) boundary between phases ``pa`` and ``pb``.

    Faithful port of ``detectTwoPhaseBoundary``: a voxel is on the boundary if
    it directly face-touches a voxel of the other phase along x, y, or z. BOTH
    voxels of each touching pair are marked. Returns a boolean mask the same
    shape as ``vol``.
    """
    mask_a = vol == pa
    mask_b = vol == pb
    boundary = np.zeros(vol.shape, dtype=bool)

    nd = vol.ndim
    for axis in range(nd):
        def sl(start: Optional[int], stop: Optional[int]) -> tuple:
            s = [slice(None)] * nd
            s[axis] = slice(start, stop)
            return tuple(s)

        lo = sl(0, -1)    # indices 0 .. n-2  (MATLAB 1:end-1)
        hi = sl(1, None)  # indices 1 .. n-1  (MATLAB 2:end)

        # A on the low side touching B on the high side
        contact = mask_a[lo] & mask_b[hi]
        boundary[lo] |= contact
        boundary[hi] |= contact

        # B on the low side touching A on the high side
        contact = mask_b[lo] & mask_a[hi]
        boundary[lo] |= contact
        boundary[hi] |= contact

    return boundary


def detect_triple_phase_boundary(vol: np.ndarray, labels: Sequence[int]) -> np.ndarray:
    """Triple-phase boundary via the 3x3x3 neighborhood method.

    Faithful port of the ``convn(mask, ones(3,3,3), 'same') > 0`` logic: a voxel
    is a TPB voxel if its 27-voxel neighborhood contains at least one voxel of
    every one of the three ``labels``. ``convn(box) > 0`` is exactly a binary
    dilation by a 3x3x3 cube, which is what we use here.
    """
    if len(labels) != 3:
        raise ValueError(f"triple-phase boundary needs exactly 3 labels, got {len(labels)}")
    struct = np.ones((3, 3, 3), dtype=bool)
    near = [ndi.binary_dilation(vol == lab, structure=struct) for lab in labels]
    return near[0] & near[1] & near[2]


@dataclass
class BoundaryCounts:
    """Voxel-count reference numbers from Dr. Jin's boundary definitions."""
    phase_voxels: Dict[int, int]
    pair_voxels: Dict[Tuple[int, int], int]
    triple_voxels: int


def boundary_voxel_counts(vol: np.ndarray, labels: Sequence[int]) -> BoundaryCounts:
    """Compute the boundary voxel counts reported by boundarydef2.m.

    These are *voxel counts*, distinct from the interfacial area (um^2) and TPB
    length (um) computed by ``psm.boundaries``. They exist for cross-validation
    and for the figure subtitles.
    """
    labels = list(labels)
    phase_voxels = {lab: int(np.count_nonzero(vol == lab)) for lab in labels}

    pair_voxels: Dict[Tuple[int, int], int] = {}
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            pair_voxels[(a, b)] = int(np.count_nonzero(detect_two_phase_boundary(vol, a, b)))

    triple = 0
    if len(labels) == 3:
        triple = int(np.count_nonzero(detect_triple_phase_boundary(vol, labels)))

    return BoundaryCounts(phase_voxels=phase_voxels,
                          pair_voxels=pair_voxels,
                          triple_voxels=triple)


# ----------------------------------------------------------------------------
# Plotting helpers
# ----------------------------------------------------------------------------
def _voxel_points(mask: np.ndarray, max_points: int, rng: np.random.Generator
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x, y, z) display coords of voxels in ``mask``, capped in count.

    ``mask`` is indexed [z, y, x]; we return display coords x, y, z. If there
    are more than ``max_points`` voxels we randomly subsample (preserves spatial
    spread better than striding).
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        empty = np.empty(0)
        return empty, empty, empty
    if max_points and idx.size > max_points:
        idx = rng.choice(idx, size=max_points, replace=False)
    z, y, x = np.unravel_index(idx, mask.shape)
    return x, y, z


def _auto_downsample(shape: Sequence[int], target_max_dim: int = 22) -> int:
    """Integer downsample factor so the largest dimension is <= target."""
    m = max(shape)
    if m <= target_max_dim:
        return 1
    return int(np.ceil(m / target_max_dim))


def _add_isosurface(ax, mask: np.ndarray, color: Color, alpha: float,
                    scale: float = 1.0) -> None:
    """Add a transparent isosurface of ``mask`` to a 3D axis (port of the
    ``patch(isosurface(...))`` calls in plotBoundarySubplot).

    ``scale`` multiplies the vertex coordinates, used when ``mask`` is a
    downsampled copy so the surface still fills the full-resolution box.
    """
    if not mask.any():
        return
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    try:
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    except (ValueError, RuntimeError):
        return
    if scale != 1.0:
        verts = verts * scale
    # verts columns are (dim0=z, dim1=y, dim2=x); reorder triangles to (x, y, z)
    tris = verts[faces][:, :, [2, 1, 0]]
    mesh = Poly3DCollection(tris, alpha=alpha)
    mesh.set_facecolor(color)
    mesh.set_edgecolor("none")
    ax.add_collection3d(mesh)


def _set_box(ax, shape: Sequence[int]) -> None:
    """Equal-aspect box matching the (z, y, x) volume shape."""
    nz, ny, nx = shape
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    try:
        ax.set_box_aspect((nx, ny, nz))
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Public plotting API
# ----------------------------------------------------------------------------
def plot_three_phase(vol: np.ndarray,
                     names: Dict[int, str],
                     colors: Optional[Dict[int, Color]] = None,
                     max_points_per_phase: int = 15000,
                     voxel_size_um: Optional[float] = None,
                     elev: float = 22.0,
                     azim: float = -60.0,
                     seed: int = 0):
    """Three-phase voxel scatter render (port of volumetricplot2.m).

    Solid phases are drawn opaque; pore (and any 'light' phase) is drawn
    semi-transparent so the solid network reads clearly, matching Dr. Jin's
    alpha choices. Voxel counts are capped per phase for responsive inline
    rendering. Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    if colors is None:
        colors = phase_colors(names)
    rng = np.random.default_rng(seed)
    labels = sorted(names)

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")

    # Draw pore/light phases first (so opaque solids render on top).
    order = sorted(labels, key=lambda l: 0 if _role_of(names[l]) == "pore" else 1)
    for lab in order:
        mask = vol == lab
        x, y, z = _voxel_points(mask, max_points_per_phase, rng)
        if x.size == 0:
            continue
        is_pore = _role_of(names[lab]) == "pore"
        ax.scatter(x, y, z,
                   s=6, marker="s",
                   c=[colors[lab]],
                   alpha=0.10 if is_pore else 0.9,
                   edgecolors="none",
                   depthshade=False,
                   label=f"{names[lab]} ({int(mask.sum()):,} vox)")

    unit = "voxels" if voxel_size_um is None else "\u00b5m"
    if voxel_size_um is not None:
        # relabel ticks in microns
        for setter, n in ((ax.set_xlim, vol.shape[2]),
                          (ax.set_ylim, vol.shape[1]),
                          (ax.set_zlim, vol.shape[0])):
            setter(0, n)
    _set_box(ax, vol.shape)
    ax.set_xlabel(f"X ({unit})")
    ax.set_ylabel(f"Y ({unit})")
    ax.set_zlabel(f"Z ({unit})")
    ax.set_title("Three-Phase Microstructure", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=8, framealpha=0.9)
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()
    return fig


def plot_boundary_subplots(vol: np.ndarray,
                           names: Dict[int, str],
                           colors: Optional[Dict[int, Color]] = None,
                           counts: Optional[BoundaryCounts] = None,
                           iso_downsample: Optional[int] = None,
                           max_boundary_points: int = 6000,
                           phase_alpha: float = 0.12,
                           elev: float = 22.0,
                           azim: float = -60.0,
                           seed: int = 0):
    """2x2 boundary figure (port of boundarydef2.m's main figure).

    For each of the three two-phase pairs and the triple-phase boundary, draws
    the three phases as faint isosurfaces (context) plus the boundary voxels as
    a colored point cloud. Returns a matplotlib Figure.

    Isosurfaces are computed on a downsampled copy of each phase mask for
    responsiveness (context only); boundary point clouds use the full-resolution
    detection, capped in count. Boundary *counts* in the subplot titles come
    from ``counts`` (full-resolution) if provided, else are computed here.
    """
    import matplotlib.pyplot as plt

    labels = sorted(names)
    if len(labels) != 3:
        raise ValueError(f"boundary subplots expect exactly 3 phases, got {len(labels)}")
    if colors is None:
        colors = phase_colors(names)
    if counts is None:
        counts = boundary_voxel_counts(vol, labels)
    rng = np.random.default_rng(seed)

    d = iso_downsample if iso_downsample is not None else _auto_downsample(vol.shape)
    vol_ds = vol[::d, ::d, ::d] if d > 1 else vol
    masks_ds = {lab: (vol_ds == lab) for lab in labels}

    a, b, c = labels
    pairs = [(a, b), (a, c), (b, c)]

    fig = plt.figure(figsize=(11, 9))

    def _draw(ax, boundary_mask, b_color, title, n_count, marker_size):
        for lab in labels:
            _add_isosurface(ax, masks_ds[lab], colors[lab], phase_alpha, scale=float(d))
        x, y, z = _voxel_points(boundary_mask, max_boundary_points, rng)
        if x.size:
            ax.scatter(x, y, z, s=marker_size, c=[b_color],
                       edgecolors="none", depthshade=False)
        _set_box(ax, vol.shape)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title(f"{title}\n{n_count:,} boundary voxels",
                     fontsize=11, fontweight="bold")
        ax.view_init(elev=elev, azim=azim)

    # three two-phase pairs
    for k, (pa, pb) in enumerate(pairs):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        bmask = detect_two_phase_boundary(vol, pa, pb)
        col = _pair_color(_role_of(names[pa]), _role_of(names[pb]))
        title = f"{names[pa]}\u2013{names[pb]} Boundary"
        _draw(ax, bmask, col, title, counts.pair_voxels[(pa, pb)], marker_size=6)

    # triple-phase boundary
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    tmask = detect_triple_phase_boundary(vol, labels)
    tname = "\u2013".join(names[l] for l in labels) + " Triple-Phase Boundary"
    _draw(ax, tmask, BOUNDARY_COLORS["triple"], tname, counts.triple_voxels, marker_size=12)

    fig.tight_layout()
    return fig


def counts_dataframe(counts: BoundaryCounts, names: Dict[int, str]):
    """Tidy DataFrame of boundary voxel counts for display in the app."""
    import pandas as pd
    rows = []
    for (a, b), n in counts.pair_voxels.items():
        rows.append({"Boundary": f"{names[a]}\u2013{names[b]}",
                     "Type": "two-phase (face contact)",
                     "VoxelCount": n})
    rows.append({"Boundary": "\u2013".join(names[l] for l in sorted(names)),
                 "Type": "triple-phase (3x3x3)",
                 "VoxelCount": counts.triple_voxels})
    return pd.DataFrame(rows)
