"""
find_roi.py -- work out the analysis crop, and PROVE it is right.

    python find_roi.py images/OFA8.tif images/OFA10.tif images/OFA11.tif

WHAT HAS TO BE CUT, AND WHY
---------------------------
An OFA cross-section image contains four horizontal zones, top to bottom:

  1. BLACK BAND      vacuum / mounting resin above the electrode surface.
                     ~160 rows of pure black. Every pixel of it falls in the
                     pore intensity band, so leaving it in reports the vacuum
                     as porosity.

  2. DENSE LAYER     a bright, non-porous skin (electrolyte / dense surface)
                     between the free surface and the porous composite.
                     Its porosity is literally 0.0%, and it is almost entirely
                     the bright (BZCY) phase. Leaving it in is the subtle
                     killer: on OFA8 it moves NiO from 43.0% to 37.2% -- a
                     SIX POINT error -- while the pore fraction still looks
                     roughly plausible, so nothing obviously screams.

  3. POROUS ELECTRODE   <-- the only zone that should be analyzed.

  4. DATABAR         the black strip with mag / WD / HV / scale bar.
                     ~118 rows. Also pure black -> also read as pore.

Cutting 1 and 4 but not 2 is worse than useless: it looks like it worked.

HOW EACH BOUNDARY IS FOUND
--------------------------
  black band + databar : per-row MEAN INTENSITY, median-filtered before the
      search. Mean intensity, NOT the fraction of near-black pixels -- that was
      a real bug in the first version. OFA8's top band is vacuum (pure black,
      row mean ~28), but OFA10's and OFA11's are mounting RESIN: dark grey and
      textured, row mean 40-60. A "fraction of pixels below 40" test sees the
      vacuum and walks straight past the resin, so OFA10/OFA11 came back with
      no top band at all and volume fractions wrong by 7-11 percentage points.
      Row mean separates cleanly on all three: bands 28-60, electrode 160-235.
      The median filter additionally stops the databar's white text rows and
      its white bottom border line from terminating the walk early.

  dense layer : per-row fraction of pixels inside the PORE intensity band.
      The dense layer's porosity is ~0; the porous electrode's is a few
      percent. Scanning down from the top of the micrograph, the dense layer
      ends where row porosity first rises and STAYS above a floor.

THE SELF-CHECK IS THE POINT
---------------------------
Detecting bands is guesswork. Reproducing the volume fractions that were
already validated against the MATLAB reference is proof. Pass --check and the
script segments the proposed ROI with that sample's validated bands and prints
the volume fractions next to the MATLAB targets. If they do not match, the
crop is wrong -- no matter how sensible the preview looks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------------
# Per-sample intensity bands (validated against the MATLAB reference) and the
# volume fractions they must reproduce. The self-check compares against these.
# ---------------------------------------------------------------------------
BANDS = {
    "OFA8":  [("Pore", 0, 100), ("NiO", 101, 174), ("BZCYYb", 175, 255)],
    "OFA10": [("Pore", 0, 75),  ("NiO", 76, 180),  ("BZCYYb", 181, 255)],
    "OFA11": [("Pore", 0, 59),  ("NiO", 60, 190),  ("BZCYYb", 191, 255)],
}
MATLAB_VF = {                       # Pore, NiO, BZCYYb  (percent)
    "OFA8":  (1.19, 43.0, 55.8),
    "OFA10": (0.46, 51.3, 48.3),
    "OFA11": (1.36, 57.0, 41.6),
}

MEAN_CUT = 120.0      # row-mean below this is band/databar, above is electrode.
                      # Measured: bands 28-60, dense layer 210-235, porous 160-195.
MEDIAN_K = 21         # median-filter window: spans the databar's text rows
MIN_BAND = 20         # ignore "bands" thinner than this
PORE_FLOOR = 0.004    # 0.4% row porosity: above this we are out of the dense layer
PORE_RUN = 40         # ...and it must stay above it for this many rows


def to_gray8(path: str) -> np.ndarray:
    im = Image.open(path)
    if im.mode not in ("L", "I;16", "I"):
        im = im.convert("L")
    a = np.asarray(im)
    if a.dtype != np.uint8:
        a = a.astype(float)
        a = (a - a.min()) / (np.ptp(a) + 1e-12)     # np.ptp, not a.ptp (NumPy 2)
        a = (a * 255).round().astype(np.uint8)
    return a


def find_black_bands(gray: np.ndarray) -> tuple[int, int, np.ndarray]:
    """(first micrograph row, first databar row, row-mean profile for the preview)."""
    mean = gray.mean(axis=1)
    m = median_filter(mean, size=MEDIAN_K)
    H = gray.shape[0]

    top = 0
    while top < H and m[top] < MEAN_CUT:
        top += 1
    if top < MIN_BAND:
        top = 0                                   # no convincing top band

    bot = H - 1
    while bot >= 0 and m[bot] < MEAN_CUT:
        bot -= 1
    bot += 1
    if H - bot < MIN_BAND:
        bot = H                                   # no convincing databar
    return top, bot, mean


def find_dense_layer_end(gray: np.ndarray, top: int, bot: int,
                         pore_hi: int) -> int:
    """First row of the POROUS electrode (i.e. where the dense skin ends)."""
    pore = (gray[top:bot] <= pore_hi).mean(axis=1)
    n = len(pore)
    for i in range(n - PORE_RUN):
        if np.all(pore[i:i + PORE_RUN] > PORE_FLOOR):
            return top + i
    return top                                    # no dense layer detected


def preview(gray: np.ndarray, r0: int, r1: int, prof: np.ndarray,
            dense_end: int, out: Path) -> None:
    H, W = gray.shape
    rgb = np.dstack([gray] * 3).astype(np.uint8)
    # everything discarded gets tinted red
    for band in ((0, r0), (r1, H)):
        s, e = band
        if e > s:
            rgb[s:e, :, 0] = np.minimum(255, rgb[s:e, :, 0].astype(int) + 70)
    if r0 > 0:
        rgb[max(0, r0 - 3):r0 + 3, :] = [255, 0, 0]        # KEEP starts here
    if r1 < H:
        rgb[r1 - 3:min(H, r1 + 3), :] = [255, 0, 0]        # KEEP ends here
    if dense_end > 0:
        rgb[max(0, dense_end - 2):dense_end + 2, :] = [255, 160, 0]   # dense edge
    bw = max(40, W // 24)
    p = prof / 255.0                              # row-mean, normalised for display
    for y in range(H):
        n = int(p[y] * bw)
        if n:
            rgb[y, W - n:] = [0, 200, 255]
    xt = W - int((MEAN_CUT / 255.0) * bw)         # the decision threshold
    rgb[:, max(0, xt - 1):xt + 1] = [255, 255, 0]
    Image.fromarray(rgb).save(out)


def check_vf(gray: np.ndarray, r0: int, r1: int, sample: str) -> str:
    """Segment the proposed ROI and compare with the validated MATLAB numbers."""
    if sample not in BANDS:
        return "  (no validated bands on file for this sample -- check skipped)"
    roi = gray[r0:r1]
    got, lines = [], []
    for nm, lo, hi in BANDS[sample]:
        got.append(100.0 * ((roi >= lo) & (roi <= hi)).mean())
    tgt = MATLAB_VF[sample]
    lines.append(f"  {'phase':<8}{'this ROI':>10}{'MATLAB':>10}{'diff':>9}")
    worst = 0.0
    for (nm, _, _), g, t in zip(BANDS[sample], got, tgt):
        d = g - t
        worst = max(worst, abs(d))
        lines.append(f"  {nm:<8}{g:9.2f}%{t:9.2f}%{d:+8.2f}pp")
    verdict = ("  ROI CONFIRMED -- reproduces the validated segmentation."
               if worst <= 1.5 else
               "  !! ROI REJECTED -- does not reproduce the validated numbers.\n"
               "     Do NOT run the sweep with this crop. Compare against the\n"
               "     crop you used in Stage 1 of the app.")
    return "\n".join(lines) + "\n" + verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--check", action="store_true", default=True,
                    help="verify the ROI against the validated MATLAB fractions")
    ap.add_argument("--keep-dense", action="store_true",
                    help="do NOT cut the dense layer (you almost never want this)")
    args = ap.parse_args()

    rois = {}
    for path in args.images:
        p = Path(path)
        if not p.exists():
            print(f"!! not found: {p}", file=sys.stderr)
            continue
        sample = p.stem
        gray = to_gray8(str(p))
        H, W = gray.shape

        top, bot, prof = find_black_bands(gray)
        pore_hi = BANDS.get(sample, [("Pore", 0, 100)])[0][2]
        dense_end = top if args.keep_dense else find_dense_layer_end(
            gray, top, bot, pore_hi)

        r0, r1 = dense_end, bot
        roi = (int(r0), 0, int(r1 - r0), int(W))
        rois[sample] = roi

        out = p.parent / f"roi_preview_{sample}.png"
        preview(gray, r0, r1, prof, dense_end, out)

        print(f"\n{'='*62}\n{p.name}   ({H} rows x {W} cols)\n{'='*62}")
        print(f"  top band (vacuum/resin) rows    0 - {top-1:<5d}  ({top} rows)  CUT")
        print(f"  dense layer (skin)    rows {top:6d} - {dense_end-1:<5d}  "
              f"({dense_end-top} rows)  {'KEPT' if args.keep_dense else 'CUT'}")
        print(f"  POROUS ELECTRODE      rows {r0:6d} - {r1-1:<5d}  ({r1-r0} rows)  KEEP")
        print(f"  databar               rows {bot:6d} - {H-1:<5d}  ({H-bot} rows)  CUT")
        print(f"\n  ROI = {roi}")
        print(f"  preview: {out}")
        if args.check:
            print()
            print(check_vf(gray, r0, r1, sample))
        for g in (3, 4):
            th, tw = (r1 - r0) // g, W // g
            warn = "" if min(th, tw) >= 350 else "   <-- small for a stable PSD"
            print(f"  {g}x{g} grid -> {th} x {tw} px tiles{warn}")

    if rois:
        print(f"\n{'='*62}\nPaste into run_batch.py\n{'='*62}")
        if len(set(rois.values())) == 1:
            print(f"  ROI = {next(iter(rois.values()))}")
        else:
            print("  ROIs differ -- give each SampleSpec its own (last argument):")
            for s, r in rois.items():
                print(f"    {s}: {r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())