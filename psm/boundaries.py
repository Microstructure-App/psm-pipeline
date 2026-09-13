"""
Double-phase-boundary (DPB) and triple-phase-boundary (TPB) analysis.

For a PCFC composite electrode the electrochemically active sites are the
triple-phase boundaries where the ion conductor (BZCYYb), the electron
conductor (Ni / NiO) and the gas pore all meet. This module quantifies:

  * DPB -- interfacial AREA between each pair of phases, by counting shared
           voxel faces and multiplying by the voxel face area.
  * TPB -- LINE density where three different phases meet, by the standard
           voxel-edge method: every interior voxel edge is shared by the four
           voxels around it; if those four voxels contain all three phases the
           edge is a TPB segment of length = voxel size.

TPB density is reported as total TPB length / total volume (units 1/length^2,
e.g. um / um^3 = um^-2), the usual figure of merit for SOFC/PCFC electrodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


@dataclass
class BoundaryResult:
    voxel_size_um: float
    dpb_area_um2: dict = field(default_factory=dict)     # ("A","B") -> area
    tpb_length_um: float = 0.0
    total_volume_um3: float = 0.0
    tpb_density_um_per_um3: float = 0.0                   # length / volume


def double_phase_boundaries(
    phase_map: np.ndarray, phase_names: dict[int, str], voxel_size_um: float
) -> dict:
    """Interfacial area (um^2) for every pair of phases via shared voxel faces."""
    face_area = voxel_size_um ** 2
    counts: dict = {}
    ndim = phase_map.ndim
    for axis in range(ndim):
        a = np.take(phase_map, range(0, phase_map.shape[axis] - 1), axis=axis)
        b = np.take(phase_map, range(1, phase_map.shape[axis]), axis=axis)
        diff = a != b
        ai = a[diff].ravel()
        bi = b[diff].ravel()
        for x, y in zip(ai, bi):
            key = tuple(sorted((int(x), int(y))))
            counts[key] = counts.get(key, 0) + 1
    return {
        (phase_names.get(k[0], k[0]), phase_names.get(k[1], k[1])): c * face_area
        for k, c in counts.items()
    }


def triple_phase_boundary(
    phase_map: np.ndarray, voxel_size_um: float, required_phases: set | None = None
) -> tuple[float, int]:
    """Total TPB length (um) and segment count via the voxel-edge method.

    An interior edge runs parallel to one axis and is surrounded by the four
    voxels in the perpendicular 2x2 block. If those four voxels contain all
    three (or all of `required_phases`) phase labels, the edge is a TPB segment.
    """
    if phase_map.ndim != 3:
        raise ValueError("TPB requires a 3D volume")
    pm = phase_map
    n_segments = 0
    # For an edge parallel to `axis`, the 2x2 block spans the other two axes.
    for axis in range(3):
        other = [d for d in range(3) if d != axis]
        d1, d2 = other
        # Build the four shifted views that share an edge parallel to `axis`.
        sl = [slice(None)] * 3
        def view(o1, o2):
            s = list(sl)
            s[d1] = slice(o1, pm.shape[d1] - 1 + o1)
            s[d2] = slice(o2, pm.shape[d2] - 1 + o2)
            return pm[tuple(s)]
        v00, v01, v10, v11 = view(0, 0), view(0, 1), view(1, 0), view(1, 1)
        stack = np.stack([v00, v01, v10, v11], axis=-1)
        stack = np.sort(stack, axis=-1)
        # number of distinct labels among the 4 voxels
        distinct = 1 + np.sum(np.diff(stack, axis=-1) != 0, axis=-1)
        if required_phases is None:
            is_tpb = distinct >= 3
        else:
            req = np.array(sorted(required_phases))
            present = np.zeros(stack.shape[:-1], dtype=bool)
            present = np.all(
                [np.any(stack == r, axis=-1) for r in req], axis=0
            )
            is_tpb = present
        n_segments += int(np.count_nonzero(is_tpb))
    return n_segments * voxel_size_um, n_segments


def analyze_boundaries(
    phase_map: np.ndarray,
    phase_names: dict[int, str],
    voxel_size_um: float,
    tpb_required_phases: set | None = None,
) -> BoundaryResult:
    """Full DPB + TPB analysis for a 3D phase volume."""
    dpb = double_phase_boundaries(phase_map, phase_names, voxel_size_um)
    tpb_len, _ = triple_phase_boundary(
        phase_map, voxel_size_um, required_phases=tpb_required_phases
    )
    total_vol = phase_map.size * voxel_size_um ** 3
    density = tpb_len / total_vol if total_vol > 0 else 0.0
    return BoundaryResult(
        voxel_size_um=voxel_size_um,
        dpb_area_um2=dpb,
        tpb_length_um=tpb_len,
        total_volume_um3=total_vol,
        tpb_density_um_per_um3=density,
    )
