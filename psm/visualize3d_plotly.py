"""
Interactive (rotatable) 3D visualization of phase volumes using Plotly.

This module is the interactive counterpart of ``psm.visualize3d`` (which renders
static matplotlib figures). It reuses the SAME boundary detectors and the SAME
phase-role color mapping, so the two renderers always agree on what a boundary
is and which phase is which color — only the drawing backend differs.

Why Plotly: figures render as WebGL in the Streamlit page, so the user can
rotate, zoom, and hover with the mouse — no re-render round trip. This also
survives PyInstaller/pywebview packaging because it is pure HTML/JS (no native
GUI toolkit needed).

Method notes carried over (IMPORTANT for reporting):
  * The volume is indexed ``vol[z, y, x]`` (VTK loader convention). Dr. Jin's
    MATLAB indexes ``(x, y, z)``. Detection math is axis-symmetric; plotting
    maps array axes -> display axes explicitly.
  * The triple-phase figure here uses the 3x3x3 neighborhood VOXEL-COUNT
    definition (Dr. Jin's boundarydef2.m). The TPB *density* in
    ``psm.boundaries`` uses the voxel-EDGE method. Different definitions,
    different numbers — cross-validation references, not interchangeable.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

import plotly.graph_objects as go

from .visualize3d import (
    BOUNDARY_COLORS,
    BoundaryCounts,
    Color,
    _auto_downsample,
    _pair_color,
    _role_of,
    _voxel_points,
    detect_triple_phase_boundary,
    detect_two_phase_boundary,
    phase_colors,
)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _rgb(c: Color, alpha: Optional[float] = None) -> str:
    """(r,g,b) floats in 0..1 -> plotly 'rgb(...)' / 'rgba(...)' string."""
    r, g, b = (int(round(255 * x)) for x in c)
    if alpha is None:
        return f"rgb({r},{g},{b})"
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _scene_layout(shape: Sequence[int], voxel_size_um: Optional[float] = None) -> dict:
    """Equal-aspect 3D scene matching the (z, y, x) volume shape."""
    nz, ny, nx = shape
    unit = "vox" if voxel_size_um is None else "µm"
    s = 1.0 if voxel_size_um is None else float(voxel_size_um)
    m = max(nx, ny, nz)
    return dict(
        xaxis=dict(title=f"X ({unit})", range=[0, nx * s]),
        yaxis=dict(title=f"Y ({unit})", range=[0, ny * s]),
        zaxis=dict(title=f"Z ({unit})", range=[0, nz * s]),
        aspectmode="manual",
        aspectratio=dict(x=nx / m, y=ny / m, z=nz / m),
        camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
    )


def _mesh_isosurface(mask: np.ndarray, color: Color, opacity: float,
                     name: str, scale: float = 1.0,
                     voxel_size_um: Optional[float] = None) -> Optional[go.Mesh3d]:
    """Marching-cubes surface of ``mask`` as a translucent Mesh3d (context)."""
    if not mask.any():
        return None
    from skimage import measure
    try:
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), level=0.5)
    except (ValueError, RuntimeError):
        return None
    if scale != 1.0:
        verts = verts * scale
    if voxel_size_um is not None:
        verts = verts * float(voxel_size_um)
    # verts columns are (z, y, x) -> display (x, y, z)
    return go.Mesh3d(
        x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=_rgb(color), opacity=opacity, name=name,
        showlegend=False, hoverinfo="skip", flatshading=True,
    )


def _boundary_scatter(mask: np.ndarray, color: Color, name: str,
                      max_points: int, seed: int, size: int = 2,
                      voxel_size_um: Optional[float] = None) -> Optional[go.Scatter3d]:
    rng = np.random.default_rng(seed)
    x, y, z = _voxel_points(mask, max_points, rng)
    if x.size == 0:
        return None
    if voxel_size_um is not None:
        s = float(voxel_size_um)
        x, y, z = x * s, y * s, z * s
    return go.Scatter3d(
        x=x, y=y, z=z, mode="markers", name=name,
        marker=dict(size=size, color=_rgb(color), opacity=0.9),
        hovertemplate=f"{name}<br>x=%{{x}}, y=%{{y}}, z=%{{z}}<extra></extra>",
    )


# ----------------------------------------------------------------------------
# Public figures
# ----------------------------------------------------------------------------
def three_phase_figure(vol: np.ndarray,
                       names: Dict[int, str],
                       colors: Optional[Dict[int, Color]] = None,
                       max_points_per_phase: int = 15000,
                       voxel_size_um: Optional[float] = None,
                       seed: int = 0) -> go.Figure:
    """Interactive three-phase voxel render (Plotly port of volumetricplot2.m).

    Solid phases opaque, pore semi-transparent, exactly like the matplotlib /
    MATLAB versions. Drag to rotate, scroll to zoom, hover for coordinates.
    """
    if colors is None:
        colors = phase_colors(names)
    rng = np.random.default_rng(seed)
    fig = go.Figure()

    # Pore first so opaque solids draw over it.
    order = sorted(names, key=lambda l: 0 if _role_of(names[l]) == "pore" else 1)
    for lab in order:
        mask = vol == lab
        x, y, z = _voxel_points(mask, max_points_per_phase, rng)
        if x.size == 0:
            continue
        if voxel_size_um is not None:
            s = float(voxel_size_um)
            x, y, z = x * s, y * s, z * s
        is_pore = _role_of(names[lab]) == "pore"
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            name=f"{names[lab]} ({int(mask.sum()):,} vox)",
            marker=dict(size=2, color=_rgb(colors[lab]),
                        opacity=0.15 if is_pore else 0.9),
            hovertemplate=f"{names[lab]}<br>x=%{{x}}, y=%{{y}}, z=%{{z}}<extra></extra>",
        ))

    fig.update_layout(
        scene=_scene_layout(vol.shape, voxel_size_um),
        title="Three-Phase Microstructure (drag to rotate)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0.0),
        margin=dict(l=0, r=0, t=60, b=0), height=620,
    )
    return fig


def pair_boundary_figure(vol: np.ndarray,
                         names: Dict[int, str],
                         pa: int, pb: int,
                         colors: Optional[Dict[int, Color]] = None,
                         iso_downsample: Optional[int] = None,
                         max_boundary_points: int = 6000,
                         phase_opacity: float = 0.10,
                         voxel_size_um: Optional[float] = None,
                         seed: int = 0) -> Tuple[go.Figure, int]:
    """One two-phase boundary view: faint phase surfaces + boundary point cloud.

    Returns (figure, full-resolution boundary voxel count) so the app can print
    the exact number directly under the plot it belongs to.
    """
    labels = sorted(names)
    if colors is None:
        colors = phase_colors(names)
    d = iso_downsample if iso_downsample is not None else _auto_downsample(vol.shape)
    vol_ds = vol[::d, ::d, ::d] if d > 1 else vol

    fig = go.Figure()
    for lab in labels:
        mesh = _mesh_isosurface(vol_ds == lab, colors[lab], phase_opacity,
                                names[lab], scale=float(d),
                                voxel_size_um=voxel_size_um)
        if mesh is not None:
            fig.add_trace(mesh)

    bmask = detect_two_phase_boundary(vol, pa, pb)
    n = int(np.count_nonzero(bmask))
    col = _pair_color(_role_of(names[pa]), _role_of(names[pb]))
    sc = _boundary_scatter(bmask, col, f"{names[pa]}–{names[pb]} boundary",
                           max_boundary_points, seed,
                           voxel_size_um=voxel_size_um)
    if sc is not None:
        fig.add_trace(sc)

    fig.update_layout(
        scene=_scene_layout(vol.shape, voxel_size_um),
        title=f"{names[pa]}–{names[pb]} boundary",
        showlegend=False, margin=dict(l=0, r=0, t=45, b=0), height=430,
    )
    return fig, n


def triple_boundary_figure(vol: np.ndarray,
                           names: Dict[int, str],
                           colors: Optional[Dict[int, Color]] = None,
                           iso_downsample: Optional[int] = None,
                           max_boundary_points: int = 6000,
                           phase_opacity: float = 0.10,
                           voxel_size_um: Optional[float] = None,
                           seed: int = 0) -> Tuple[go.Figure, int]:
    """Triple-phase boundary view (3x3x3 neighborhood method).

    Returns (figure, TPB voxel count). Remember: this count is NOT the same
    quantity as the voxel-edge TPB density from ``psm.boundaries``.
    """
    labels = sorted(names)
    if len(labels) != 3:
        raise ValueError(f"triple-phase figure needs exactly 3 phases, got {len(labels)}")
    if colors is None:
        colors = phase_colors(names)
    d = iso_downsample if iso_downsample is not None else _auto_downsample(vol.shape)
    vol_ds = vol[::d, ::d, ::d] if d > 1 else vol

    fig = go.Figure()
    for lab in labels:
        mesh = _mesh_isosurface(vol_ds == lab, colors[lab], phase_opacity,
                                names[lab], scale=float(d),
                                voxel_size_um=voxel_size_um)
        if mesh is not None:
            fig.add_trace(mesh)

    tmask = detect_triple_phase_boundary(vol, labels)
    n = int(np.count_nonzero(tmask))
    sc = _boundary_scatter(tmask, BOUNDARY_COLORS["triple"], "Triple-phase boundary",
                           max_boundary_points, seed, size=3,
                           voxel_size_um=voxel_size_um)
    if sc is not None:
        fig.add_trace(sc)

    tname = "–".join(names[l] for l in labels)
    fig.update_layout(
        scene=_scene_layout(vol.shape, voxel_size_um),
        title=f"{tname} triple-phase boundary",
        showlegend=False, margin=dict(l=0, r=0, t=45, b=0), height=430,
    )
    return fig, n
