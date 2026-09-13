"""psm.scalebar -- click-to-measure scale-bar calibration helpers.

Supports the Stage 1 "Measure scale bar by clicking" tool in app.py.
The user clicks the two end ticks of the scale bar on the SEM databar;
these helpers convert display-space clicks to native pixel coordinates,
optionally snap each click to the nearest bright vertical tick mark, and
return the pixel distance between the two points.

Design notes
------------
* All functions are pure numpy -- no Streamlit imports -- so they can be
  unit-tested headlessly (mirrors the matlab_parity_check.py philosophy).
* Snap-to-tick exists because a hand click on a downscaled preview is only
  accurate to ~3-5 native px per click (~1% error on an 800 px bar). SEM
  databars (FEI/Thermo style) terminate the bar with short vertical end
  ticks that are taller than the 2-3 px bar body, so within a small search
  window the tick column is identified as the column with the largest
  count of bright pixels in a vertical band around the click.
* If no column in the window is convincingly tick-like (e.g. the user is
  measuring a plain bar with no end ticks, or clicked empty space), the
  click is returned unchanged and `snapped=False` -- the tool degrades
  gracefully to raw-click measurement.
"""

from __future__ import annotations

import numpy as np

# Intensity above which a databar pixel counts as "bright" (bars/ticks are
# rendered white-on-black on FEI/Thermo databars; 180 tolerates JPEG halos).
_BRIGHT = 180


def bottom_strip(gray: np.ndarray, frac: float = 0.22) -> tuple[np.ndarray, int]:
    """Return (strip, y_offset) = the bottom `frac` of the image.

    The SEM databar lives at the bottom of the frame, so showing only this
    strip lets the click widget display it much larger (better click
    accuracy) than showing the whole image.

    Parameters
    ----------
    gray : 2-D uint8 array, full SEM image (pre-crop, databar included).
    frac : fraction of image height to keep, measured from the bottom.

    Returns
    -------
    strip : 2-D uint8 view of the bottom rows.
    y_offset : row index in the full image where the strip starts
               (full_y = y_offset + strip_y).
    """
    h = gray.shape[0]
    y0 = max(0, int(round(h * (1.0 - frac))))
    return gray[y0:, :], y0


def snap_to_tick(gray: np.ndarray, x: int, y: int,
                 search: int = 25, band: int = 22,
                 thresh: int = _BRIGHT) -> tuple[int, bool]:
    """Refine a clicked x to the nearest scale-bar end tick, if one exists.

    Scans columns within +/-`search` px of the click. For each column it
    counts bright pixels in a vertical band of +/-`band` rows around the
    clicked y. The horizontal bar body contributes only its own thickness
    (~2-3 bright rows) to *every* column, whereas an end tick is a locally
    taller vertical stroke -- so the tick column stands out as the max.

    Returns (x_refined, snapped). If no column looks tick-like (max count
    below 6, or not clearly above the window's typical bar-body count),
    the original x is returned with snapped=False.
    """
    h, w = gray.shape
    x = int(round(x)); y = int(round(y))
    if not (0 <= x < w and 0 <= y < h):
        return x, False

    c_lo, c_hi = max(0, x - search), min(w, x + search + 1)
    r_lo, r_hi = max(0, y - band), min(h, y + band + 1)
    win = (gray[r_lo:r_hi, c_lo:c_hi] > thresh)
    counts = win.sum(axis=0)               # bright rows per column
    if counts.size == 0:
        return x, False

    best_local = int(np.argmax(counts))
    best = int(counts[best_local])
    typical = float(np.median(counts))     # ~bar-body thickness (or 0)

    # A tick must be (a) a real vertical stroke, (b) clearly taller than
    # whatever the bar body contributes across the whole window.
    if best >= 6 and best >= typical + 3:
        return c_lo + best_local, True
    return x, False


def measure_bar_px(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    """Pixel length of the scale bar from two clicked endpoints.

    SEM scale bars are horizontal, so the physical length is the
    horizontal separation |x2 - x1|; using the Euclidean distance would
    let vertical click jitter inflate the measurement.
    """
    return float(abs(p2[0] - p1[0]))


def microns_per_pixel_from_clicks(p1: tuple[int, int], p2: tuple[int, int],
                                  bar_um: float) -> float:
    """um/px implied by two clicked endpoints and the bar's printed label."""
    d = measure_bar_px(p1, p2)
    if d <= 0:
        raise ValueError("The two clicked points have the same x coordinate; "
                         "click the two opposite ends of the scale bar.")
    return bar_um / d
