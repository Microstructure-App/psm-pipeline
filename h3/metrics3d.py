"""3D microstructure metrics for the H3 study.

Definitions deliberately mirror the psm package:
  - geometric tortuosity: BFS shortest 6-connected path entry face -> exit
    face, divided by straight-line distance (camp MATLAB estimateTortuosity
    behavior; see project README).
  - TPB density: voxel-edge method (edges whose 4 incident voxels contain
    all three phases), length / volume  [um^-2]  (psm/boundaries.py method).
  - DPB: phase-pair face area / volume  [um^-1].
  - connectivity ratio: largest connected component voxels / phase voxels.

Volume convention: vol[z, y, x] (VTK loader convention), phase ids 0..2.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

STRUCT6 = ndi.generate_binary_structure(3, 1)


def volume_fractions(vol: np.ndarray, n_phases: int = 3) -> dict:
    tot = vol.size
    return {p: float((vol == p).sum()) / tot for p in range(n_phases)}


def connectivity(vol: np.ndarray, phase: int) -> dict:
    mask = vol == phase
    n_vox = int(mask.sum())
    if n_vox == 0:
        return dict(ratio=float("nan"), n_components=0, n_voxels=0)
    lab, n = ndi.label(mask, structure=STRUCT6)
    if n == 0:
        return dict(ratio=float("nan"), n_components=0, n_voxels=n_vox)
    sizes = np.bincount(lab.ravel())[1:]
    return dict(ratio=float(sizes.max()) / n_vox, n_components=int(n),
                n_voxels=n_vox)


def geometric_tortuosity(vol: np.ndarray, phase: int, axis: int = 0) -> dict:
    """BFS shortest path from entry face to exit face along `axis`.

    Returns tau = min geodesic steps / (extent-1); inf if not percolating.
    """
    mask = np.moveaxis(vol == phase, axis, 0)  # BFS along axis 0
    nz = mask.shape[0]
    if not mask[0].any() or not mask[-1].any():
        return dict(tau=float("inf"), percolates=False)
    dist = np.full(mask.shape, -1, dtype=np.int32)
    front = np.zeros_like(mask)
    front[0] = mask[0]
    dist[front] = 0
    d = 0
    reached = front.copy()
    while front.any():
        d += 1
        grown = ndi.binary_dilation(front, structure=STRUCT6) & mask & ~reached
        if not grown.any():
            break
        dist[grown] = d
        reached |= grown
        front = grown
        # BFS explores strictly in order of distance, so the first arrival at
        # any exit-face voxel is the global minimum over the exit face.
        if grown[-1].any():
            break
    exit_d = dist[-1][mask[-1]]
    exit_d = exit_d[exit_d >= 0]
    if exit_d.size == 0:
        return dict(tau=float("inf"), percolates=False)
    tau = float(exit_d.min()) / float(nz - 1)
    return dict(tau=tau, percolates=True)


def tpb_density(vol: np.ndarray, voxel_um: float) -> dict:
    """Voxel-edge TPB: an edge parallel to axis k is a TPB segment when the
    4 voxels sharing it (in the perpendicular plane) contain all 3 phases.
    Returns total length (um) and density (um^-2)."""
    n_edges = 0
    for ax in range(3):
        v = np.moveaxis(vol, ax, 0)
        a = v[:, :-1, :-1]
        b = v[:, :-1, 1:]
        c = v[:, 1:, :-1]
        d = v[:, 1:, 1:]
        stack = np.stack([a, b, c, d], axis=0)
        has = [(stack == p).any(axis=0) for p in range(3)]
        tpb = has[0] & has[1] & has[2]
        n_edges += int(tpb.sum())
    vol_um3 = vol.size * voxel_um ** 3
    length_um = n_edges * voxel_um
    return dict(n_edges=n_edges, length_um=length_um,
                density_um2=length_um / vol_um3)


def dpb_areas(vol: np.ndarray, voxel_um: float) -> dict:
    """Interfacial area per phase pair via face counting -> um^-1 (area/vol)."""
    pairs = {(0, 1): 0, (0, 2): 0, (1, 2): 0}
    for ax in range(3):
        v = np.moveaxis(vol, ax, 0)
        a, b = v[:-1], v[1:]
        diff = a != b
        for (p, q) in pairs:
            pairs[(p, q)] += int((((a == p) & (b == q)) |
                                  ((a == q) & (b == p))).sum())
    vol_um3 = vol.size * voxel_um ** 3
    face_um2 = voxel_um ** 2
    return {f"{p}-{q}": cnt * face_um2 / vol_um3 for (p, q), cnt in pairs.items()}


def full_report(vol: np.ndarray, names: dict, voxel_um: float,
                tort_axis: int = 0) -> dict:
    vf = volume_fractions(vol)
    out = dict(voxel_um=voxel_um, shape=list(vol.shape))
    for p, nm in names.items():
        out[f"VF_{nm}"] = vf[p]
        c = connectivity(vol, p)
        out[f"Conn_{nm}"] = c["ratio"]
        t = geometric_tortuosity(vol, p, axis=tort_axis)
        out[f"Tau_{nm}"] = t["tau"]
        out[f"Percolates_{nm}"] = t["percolates"]
    tpb = tpb_density(vol, voxel_um)
    out["TPB_density_um2"] = tpb["density_um2"]
    dpb = dpb_areas(vol, voxel_um)
    for k, v in dpb.items():
        p, q = int(k[0]), int(k[2])
        out[f"DPB_{names[p]}-{names[q]}_um1"] = v
    return out
