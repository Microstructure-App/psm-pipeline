"""
Geometric tortuosity and percolation for a labeled phase volume.

Generalizes the camp 2D BFS tortuosity (pdf_analysis_2d.m) to 3D voxel
volumes. For a chosen phase and a transport axis, it does a breadth-first
search through the phase from the entry face to the exit face and reports:

    tortuosity tau = (shortest connected path length) / (straight distance)

plus whether the phase percolates (a connected path exists at all) and the
connectivity ratio (largest connected cluster / total phase voxels), matching
the camp definitions in the lecture slides.

Works on 2D arrays too (treat as a single-slice volume).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class TortuosityResult:
    phase: int
    axis: int
    percolates: bool
    tortuosity: float            # NaN if no connected path
    path_length: float           # in voxel steps; NaN if none
    straight_distance: float
    connectivity_ratio: float    # largest cluster / total phase voxels


def _neighbors_offsets(ndim: int) -> list[tuple]:
    """6-connectivity in 3D / 4-connectivity in 2D (face neighbors only)."""
    offs = []
    for d in range(ndim):
        for s in (-1, 1):
            o = [0] * ndim
            o[d] = s
            offs.append(tuple(o))
    return offs


def connectivity_ratio(mask: np.ndarray) -> float:
    """Largest connected cluster size / total phase voxels (face connectivity)."""
    total = int(mask.sum())
    if total == 0:
        return 0.0
    structure = ndi.generate_binary_structure(mask.ndim, 1)  # face connectivity
    labels, n = ndi.label(mask, structure=structure)
    if n == 0:
        return 0.0
    sizes = np.bincount(labels.ravel())[1:]
    return float(sizes.max()) / total


def geometric_tortuosity(
    phase_map: np.ndarray, phase: int, axis: int = 0
) -> TortuosityResult:
    """BFS shortest path through `phase` from the low face to high face of `axis`.

    The straight distance is (n_along_axis - 1) voxel steps; tortuosity is the
    shortest path length divided by that. Returns NaN tortuosity if the phase
    does not span the volume along `axis`.
    """
    mask = phase_map == phase
    ndim = mask.ndim
    n_axis = mask.shape[axis]
    straight = float(n_axis - 1)

    # entry slice = first plane along axis, exit slice = last plane
    entry_sel = [slice(None)] * ndim
    entry_sel[axis] = 0
    exit_sel = [slice(None)] * ndim
    exit_sel[axis] = n_axis - 1

    dist = np.full(mask.shape, -1, dtype=np.int64)
    q: deque = deque()

    # seed all phase voxels on the entry face
    entry_face = mask[tuple(entry_sel)]
    for idx in np.argwhere(entry_face):
        full = list(idx[:axis]) + [0] + list(idx[axis:])
        coord = tuple(full)
        dist[coord] = 0
        q.append(coord)

    offsets = _neighbors_offsets(ndim)
    shape = mask.shape
    while q:
        cur = q.popleft()
        d0 = dist[cur]
        for off in offsets:
            nb = tuple(cur[i] + off[i] for i in range(ndim))
            if all(0 <= nb[i] < shape[i] for i in range(ndim)):
                if mask[nb] and dist[nb] == -1:
                    dist[nb] = d0 + 1
                    q.append(nb)

    exit_dist = dist[tuple(exit_sel)]
    reached = exit_dist[exit_dist >= 0]
    cr = connectivity_ratio(mask)

    if reached.size == 0:
        return TortuosityResult(
            phase=phase, axis=axis, percolates=False, tortuosity=float("nan"),
            path_length=float("nan"), straight_distance=straight,
            connectivity_ratio=cr,
        )
    path_len = float(reached.min())
    tau = path_len / straight if straight > 0 else float("nan")
    return TortuosityResult(
        phase=phase, axis=axis, percolates=True, tortuosity=tau,
        path_length=path_len, straight_distance=straight, connectivity_ratio=cr,
    )


def all_phase_tortuosity(
    phase_map: np.ndarray, phases: dict[int, str], axis: int = 0
) -> list[dict]:
    """Run tortuosity for every phase; return a list of plain-dict rows."""
    rows = []
    for pid, name in phases.items():
        r = geometric_tortuosity(phase_map, pid, axis=axis)
        rows.append(
            dict(
                Phase=name, Axis=axis, Percolates=r.percolates,
                Tortuosity=r.tortuosity, PathLength_vox=r.path_length,
                StraightDistance_vox=r.straight_distance,
                ConnectivityRatio=r.connectivity_ratio,
            )
        )
    return rows
