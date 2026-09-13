"""
PCFC Electrode Microstructure Pipeline -- Streamlit app (wizard / "map" layout).

Run with:   streamlit run app.py

Stages:
  1. Setup & Calibration   -- sample ID, SEM image, scale bar.
  2. Phase Segmentation    -- intensity thresholds -> phase map + volume fractions.
  3. Particle Sizes        -- PSD + lognormal (mu, sigma) per phase.
  4. Generate 3D Structure -- DREAM.3D pipeline export / one-click headless run.
  5. 3D Analysis           -- tortuosity + TPB/DPB + interactive 3D visualization.
  6. Results & Export      -- metrics with provenance, replicate statistics, ranking.
  * Manual                 -- in-app user guide.

Session state carries results between stages; recorded runs persist as a list in
outputs/sample_results.json (schema 2), so repeated reconstructions of the same
sample coexist and their spread can be reported.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import pathlib
import re
import tempfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from psm import segmentation as seg
from psm import psd as psdmod
from psm import dream3d as d3d
from psm import dream3d_runner as dr
from psm import visualize3d as viz
from psm import visualize3d_plotly as vpl
from psm import evaluation as ev
from psm.tortuosity import all_phase_tortuosity
from psm.boundaries import analyze_boundaries
from psm import scalebar as sb

from PIL import Image, ImageDraw

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAVE_IMG_COORDS = True
except ImportError:
    HAVE_IMG_COORDS = False

try:
    # streamlit-drawable-canvas 0.9.3 fetches its background image through
    # streamlit.elements.image.image_to_url, which Streamlit >=1.41 moved to
    # streamlit.elements.lib.image_utils (with a new signature). Bridge the
    # old call site before importing the canvas so backgrounds still render.
    import streamlit.elements.image as _st_image_mod
    if not hasattr(_st_image_mod, "image_to_url"):
        from streamlit.elements.lib.image_utils import (
            image_to_url as _image_to_url_new)
        from streamlit.elements.lib.layout_utils import (
            LayoutConfig as _LayoutConfig)

        def _image_to_url_compat(image, width, clamp, channels,
                                 output_format, image_id):
            try:
                layout = _LayoutConfig(
                    width=width if isinstance(width, int) and width > 0
                    else None)
            except Exception:
                layout = _LayoutConfig()
            return _image_to_url_new(image, layout, clamp, channels,
                                     output_format, image_id)

        _st_image_mod.image_to_url = _image_to_url_compat
    from streamlit_drawable_canvas import st_canvas
    HAVE_CANVAS = True
except Exception:
    HAVE_CANVAS = False

# ---------------------------------------------------------------------------
# Paths & persistent user settings
# ---------------------------------------------------------------------------
APP_VERSION = "1.0.0"
APP_DIR = pathlib.Path(__file__).parent
OUTPUT_DIR = pathlib.Path(os.environ.get("PSM_OUTPUT_DIR", APP_DIR / "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SETTINGS_FILE = pathlib.Path.home() / ".psm_pipeline_settings.json"
SAMPLES_FILE = OUTPUT_DIR / "sample_results.json"


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def save_settings(s: dict) -> None:
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))


def load_runs() -> list:
    """Load recorded runs, migrating the old {sample_id: record} format.

    Schema 1 stored one record per sample, keyed by sample ID, overwritten on
    every re-save. Schema 2 stores a list of runs so repeated reconstructions
    of the same sample coexist and their spread can be reported. Old files are
    converted in memory and rewritten on the next save; nothing is discarded.
    """
    if not SAMPLES_FILE.exists():
        return []
    try:
        raw = json.loads(SAMPLES_FILE.read_text())
    except json.JSONDecodeError:
        return []

    if isinstance(raw, dict) and "runs" in raw:            # schema 2
        return list(raw.get("runs", []))
    if isinstance(raw, list):                              # bare list
        return raw
    if isinstance(raw, dict):                              # schema 1 -> migrate
        out = []
        for sample_id, rec in raw.items():
            if not isinstance(rec, dict):
                continue
            r = dict(rec)
            r.setdefault("Sample", sample_id)
            r.setdefault("Replicate", 1)
            r.setdefault("RunID", f"legacy{abs(hash(sample_id)) % 10**6:06d}")
            r.setdefault("Timestamp", "")
            r["MigratedFromSchema1"] = True
            out.append(r)
        return out
    return []


def save_runs(runs: list) -> None:
    SAMPLES_FILE.write_text(json.dumps(
        {"schema_version": 2, "runs": runs}, indent=2, default=str))


def safe_name(s: str) -> str:
    """Sanitize a sample ID for use in file names (packaging/security)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", s.strip())[:40] or "sample"


def validate_runner(path: str) -> tuple[bool, str]:
    """Only allow the real DREAM.3D PipelineRunner to be executed.

    The path is user-editable and persisted in a settings file, so before the
    app launches a subprocess it checks that the file exists and is actually
    named PipelineRunner(.exe). This prevents a tampered settings file from
    pointing the app at an arbitrary program.
    """
    if not os.path.isfile(path):
        return False, f"File not found: {path}"
    base = os.path.basename(path).lower()
    if base not in ("pipelinerunner.exe", "pipelinerunner"):
        return False, ("For safety this app only runs a file named "
                       "PipelineRunner.exe (the DREAM.3D headless runner), "
                       f"not '{os.path.basename(path)}'.")
    return True, ""


# ---------------------------------------------------------------------------
# Page + light styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PCFC Microstructure Pipeline",
                   page_icon="🔬", layout="wide")
ss = st.session_state

st.markdown("""
<style>
  /* tighter, cleaner sidebar nav buttons */
  section[data-testid="stSidebar"] button { border-radius: 8px; }
  /* verdict banners */
  .verdict-good {background:#e7f5e7;border-left:6px solid #2e7d32;
                 padding:0.8rem 1rem;border-radius:6px;}
  .verdict-ok   {background:#fff8e1;border-left:6px solid #f9a825;
                 padding:0.8rem 1rem;border-radius:6px;}
  .verdict-poor {background:#fdecea;border-left:6px solid #c62828;
                 padding:0.8rem 1rem;border-radius:6px;}
  .stMetric {background:#fafafa;border:1px solid #eee;border-radius:8px;
             padding:0.4rem 0.6rem;}
</style>
""", unsafe_allow_html=True)


def rerun():
    """Force an immediate rerun (works on old and new Streamlit)."""
    (getattr(st, "rerun", None) or st.experimental_rerun)()


# ---------------------------------------------------------------------------
# Stage registry + navigation (the "map")
# ---------------------------------------------------------------------------
STEPS = [
    ("setup",    "1 · Setup & Calibration"),
    ("segment",  "2 · Phase Segmentation"),
    ("psd",      "3 · Particle Sizes"),
    ("generate", "4 · Generate 3D Structure"),
    ("analyze",  "5 · 3D Analysis"),
    ("evaluate", "6 · Results & Export"),
    ("manual",   "📖 Manual"),
]
N_PIPELINE = 6  # first six are the ordered pipeline; the last is the manual.

if "step" not in ss:
    ss.step = 0
if "runs" not in ss:
    ss.runs = load_runs()


def goto(i: int):
    ss.step = max(0, min(i, len(STEPS) - 1))
    rerun()


def sidebar_map():
    with st.sidebar:
        st.markdown("### 🗺️ Pipeline map")
        sample = ss.get("sample_id", "")
        if sample:
            st.caption(f"Current sample: **{sample}**")
        done_flags = {
            0: "g8" in ss,        # Setup done once an image is loaded/cropped
            1: "res" in ss,       # Segmentation done
            2: "psds" in ss,      # Particle sizes done
            3: "vol" in ss,       # 3D structure generated
            4: "bres" in ss,      # 3D metrics computed
            5: bool(ss.get("runs")),  # at least one run recorded
        }
        n_done = sum(bool(done_flags.get(i)) for i in range(N_PIPELINE))
        st.progress(n_done / N_PIPELINE,
                    text=f"{n_done} / {N_PIPELINE} stages complete")
        for i, (_key, label) in enumerate(STEPS):
            current = i == ss.step
            done = done_flags.get(i, False) and not current
            prefix = "▶ " if current else ("✓ " if done else "   ")
            if st.button(prefix + label, key=f"nav_{i}",
                         use_container_width=True,
                         type="primary" if current else "secondary"):
                goto(i)
        st.divider()
        st.caption("Data stays loaded as you move between stages. "
                   "Click a stage to jump to it.")


def nav_footer():
    """Back / Next buttons, shown on the pipeline stages only."""
    if ss.step >= N_PIPELINE:
        return
    st.divider()
    b, _, n = st.columns([1, 6, 1])
    with b:
        if ss.step > 0 and st.button("← Back", use_container_width=True):
            goto(ss.step - 1)
    with n:
        if ss.step < N_PIPELINE - 1 and st.button("Next →", use_container_width=True,
                                                   type="primary"):
            goto(ss.step + 1)


# ===========================================================================
# STAGE 1 -- Setup & Calibration
# ===========================================================================
def step_setup():
    st.header("1 · Setup & Calibration")
    st.write("Name the sample, upload its SEM image, and tell the app how many microns "
             "each pixel is worth. Everything downstream (particle sizes, boundaries) "
             "depends on this calibration.")

    st.subheader("Sample")
    c0, c0b = st.columns([1, 2])
    choice = c0.selectbox(
        "Sample ID", ["OFA8", "OFA10", "OFA11", "Custom…"], key="sample_choice",
        help="Which electrode this image belongs to. The three OFA samples are the "
             "study set (OFA8 = sedimentary, OFA10 = literature-preferred, "
             "OFA11 = blistered). Results saved in Stage 6 are grouped by this ID, "
             "so the three electrodes can be compared side by side.")
    if choice == "Custom…":
        ss.sample_id = c0b.text_input("Custom sample name", value=ss.get("sample_id", ""),
                                      key="sample_custom")
    else:
        ss.sample_id = choice

    c1, c2 = st.columns([2, 3])
    with c1:
        st.subheader("Scale bar")
        ss.setdefault("scale_bar_um", 30.0)
        ss.setdefault("scale_bar_px", 200.0)
        scale_bar_um = st.number_input(
            "Scale bar length (µm)", min_value=0.1, key="scale_bar_um",
            help="The physical length printed next to the scale bar on the microscope "
                 "image, in microns (e.g. '30 µm'). Together with the pixel measurement "
                 "below it converts every pixel measurement into real distances — if it "
                 "is wrong, every size and boundary number downstream is wrong by the "
                 "same factor.")
        scale_bar_px = st.number_input(
            "Scale bar length (pixels)", min_value=1.0, key="scale_bar_px",
            help="The length of that same scale bar measured in pixels. Open the image "
                 "in any viewer (even MS Paint), hover over each end of the bar, and "
                 "subtract the x-coordinates. Typical SEM scale bars are 100–400 px.")
        um_per_px = seg.microns_per_pixel(scale_bar_px, scale_bar_um)
        ss.um_per_px = um_per_px
        st.metric("Resolution", f"{um_per_px:.5f} µm / pixel")

    with c2:
        st.subheader("SEM image")
        upload = st.file_uploader(
            "SEM image", type=["png", "tif", "tiff", "jpg", "jpeg", "bmp"], key="sem_upload",
            help="For best results the image should show only the sample surface. "
                 "Black borders, the scale-bar graphic, text labels, and epoxy mounting "
                 "material all have their own gray levels, so if left in they corrupt "
                 "the phase fractions. Crop them out beforehand or with the crop tool "
                 "below.")
        if upload is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False,
                                              suffix=os.path.splitext(upload.name)[1])
            tmp.write(upload.read())
            tmp.close()
            ss.g8_full = seg.read_sem_as_gray8(tmp.name)
            st.success(f"Loaded {upload.name} — shape {ss.g8_full.shape}")


    # ---- Optional: measure the scale bar by clicking ----------------------
    if "g8_full" in ss:
        with st.expander("📏 Measure scale bar by clicking (recommended)",
                         expanded=False):
            if not HAVE_IMG_COORDS:
                st.warning(
                    "This tool needs the `streamlit-image-coordinates` package. "
                    "Install it in your environment and restart the app:\n\n"
                    "```\npip install streamlit-image-coordinates\n```")
            else:
                st.write(
                    "Click the **two end ticks** of the scale bar in the databar "
                    "below (order doesn't matter). The measured pixel length is "
                    "computed from the horizontal distance between the clicks, so "
                    "small vertical wobble is harmless. With *snap* on, each click "
                    "is refined to the nearest bright tick mark automatically.")

                frac = st.slider(
                    "Portion of image shown (from the bottom)", 0.10, 1.00, 0.22,
                    0.02, key="sb_frac",
                    help="The databar sits at the bottom of the frame; showing "
                         "only that strip makes it larger and clicks more "
                         "accurate. Slide to 1.0 to see the whole image.")
                do_snap = st.checkbox(
                    "Snap clicks to nearest tick mark", value=True, key="sb_snap",
                    help="Refines each click to the nearest bright vertical "
                         "stroke (the bar's end tick) within ±25 px. If no tick "
                         "is found near the click, the raw click is kept.")

                strip, _y0 = sb.bottom_strip(ss.g8_full, frac)
                disp_w = int(min(strip.shape[1], 1200))
                scale = strip.shape[1] / disp_w

                # Redraw stored clicks as red marker lines on the strip.
                marked = Image.fromarray(strip).convert("RGB")
                drw = ImageDraw.Draw(marked)
                for (mx, _my) in ss.get("sb_points", []):
                    drw.line([(mx, 0), (mx, strip.shape[0])],
                             fill=(255, 60, 60), width=max(2, int(round(scale))))

                click = streamlit_image_coordinates(marked, width=disp_w,
                                                    key="sb_click")
                if click is not None:
                    raw = (int(round(click["x"] * scale)),
                           int(round(click["y"] * scale)))
                    if raw != ss.get("sb_last_click"):
                        ss.sb_last_click = raw
                        pt = raw
                        if do_snap:
                            xr, _snapped = sb.snap_to_tick(strip, raw[0], raw[1])
                            pt = (xr, raw[1])
                        pts = list(ss.get("sb_points", []))
                        pts.append(pt)
                        ss.sb_points = pts[-2:]   # keep the two most recent
                        st.rerun()                # redraw markers immediately

                pts = ss.get("sb_points", [])
                cA, cB = st.columns([1, 3])
                if cA.button("Reset clicks", key="sb_reset"):
                    ss.sb_points = []
                    ss.sb_last_click = None
                    st.rerun()
                cB.caption(f"Clicks recorded: {len(pts)}/2"
                           + (f" — x = {[p[0] for p in pts]}" if pts else ""))

                if len(pts) == 2:
                    bar_px = sb.measure_bar_px(pts[0], pts[1])
                    if bar_px <= 0:
                        st.error("Both clicks landed on the same column — click "
                                 "the two opposite ends of the bar.")
                    else:
                        bar_um = st.number_input(
                            "Printed bar length (µm)", min_value=0.1,
                            value=float(ss.get("scale_bar_um", 30.0)),
                            key="sb_um_label",
                            help="The number written next to the bar on the "
                                 "databar (e.g. '30 µm'). The bar's physical "
                                 "length, not the HFW field.")
                        st.metric("Measured scale bar",
                                  f"{bar_px:.0f} px  →  "
                                  f"{bar_um / bar_px:.5f} µm / pixel")

                        def _apply_scalebar_measurement():
                            # Runs as an on_click callback, i.e. *before* the
                            # next script run — the only safe point to write
                            # to widget-bound session-state keys.
                            ss["scale_bar_px"] = float(round(bar_px, 1))
                            ss["scale_bar_um"] = float(ss["sb_um_label"])

                        st.button("✅ Apply to calibration", type="primary",
                                  key="sb_apply",
                                  on_click=_apply_scalebar_measurement,
                                  help="Writes the measured values into the "
                                       "Scale bar fields above; the µm/pixel "
                                       "resolution updates instantly.")

    if "g8_full" in ss:
        st.subheader(
            "Optional: crop region of interest",
            help="Restrict the analysis to a rectangular region containing only sample. "
                 "Drag a rectangle directly on the image (recommended) or type the "
                 "row/column bounds by hand; the preview updates live so you can check "
                 "that borders, labels, and the scale bar are excluded.")
        g = ss.g8_full
        H, W = g.shape

        # Canonical ROI (r0, c0, h, w) lives in a plain session-state key so it
        # survives switching between the drag and numeric tools. It is reset to
        # the full frame whenever an image with a different shape is loaded.
        if ss.get("crop_roi_shape") != (H, W):
            ss.crop_roi_box = (0, 0, H, W)
            ss.crop_roi_shape = (H, W)

        def _clamp_box(r0, c0, h, w):
            """Force an (r0, c0, h, w) box to lie fully inside the image."""
            r0 = int(min(max(r0, 0), H - 1))
            c0 = int(min(max(c0, 0), W - 1))
            h = int(min(max(h, 1), H - r0))
            w = int(min(max(w, 1), W - c0))
            return r0, c0, h, w

        CROP_MODES = ["🖱️ Drag rectangle", "🔢 Numeric entry"]
        mode = st.radio("Crop tool", CROP_MODES, horizontal=True,
                        key="crop_mode", label_visibility="collapsed")

        if mode == CROP_MODES[0]:
            if not HAVE_CANVAS:
                st.warning(
                    "Drag-to-crop needs the `streamlit-drawable-canvas` "
                    "package. Install it and restart the app:\n\n"
                    "```\npip install streamlit-drawable-canvas\n```\n\n"
                    "Until then, switch to **Numeric entry** above.")
            else:
                st.write(
                    "**Press, drag, release** — the red box is visible while "
                    "you drag, and the crop applies the moment you let go. "
                    "Draw a new box any time to replace the old one.")

                # Downscale for display with a *known* factor so drawn
                # rectangles map back to full-resolution pixels exactly.
                disp_w = int(min(W, 950))
                disp_scale = disp_w / W
                disp_h = max(1, int(round(H * disp_scale)))
                pil_disp = Image.fromarray(g).convert("RGB").resize(
                    (disp_w, disp_h))

                # Seed the canvas with the currently-applied ROI so the
                # selection stays visible, and key it on that ROI so the
                # canvas resets (showing only the latest box) whenever the
                # ROI changes — from a drag, the buttons, or numeric entry.
                rr0, cc0, chh, cww = _clamp_box(*ss.crop_roi_box)
                init_objs = []
                if (rr0, cc0, chh, cww) != (0, 0, H, W):
                    init_objs.append({
                        "type": "rect", "version": "4.4.0",
                        "originX": "left", "originY": "top",
                        "left": int(cc0 * disp_scale),
                        "top": int(rr0 * disp_scale),
                        "width": max(1, int(cww * disp_scale)),
                        "height": max(1, int(chh * disp_scale)),
                        "fill": "rgba(255, 75, 75, 0.12)",
                        "stroke": "#FF4B4B", "strokeWidth": 2,
                        "scaleX": 1, "scaleY": 1})

                result = st_canvas(
                    background_image=pil_disp, height=disp_h, width=disp_w,
                    drawing_mode="rect", stroke_color="#FF4B4B",
                    fill_color="rgba(255, 75, 75, 0.12)", stroke_width=2,
                    initial_drawing={"version": "4.4.0",
                                     "objects": init_objs},
                    update_streamlit=True, display_toolbar=False,
                    key=f"crop_canvas_{H}x{W}_{rr0}_{cc0}_{chh}_{cww}")

                # The newest rectangle on the canvas is the selection. Map it
                # back to full resolution and apply if it genuinely moved
                # (>2 px tolerance absorbs display-scale rounding, so a bare
                # click on the canvas can't jitter the ROI).
                if result is not None and result.json_data:
                    rects = [o for o in result.json_data.get("objects", [])
                             if o.get("type") == "rect"]
                    if rects:
                        o = rects[-1]
                        w_px = abs(o.get("width", 0) * o.get("scaleX", 1))
                        h_px = abs(o.get("height", 0) * o.get("scaleY", 1))
                        x0 = o.get("left", 0)
                        y0 = o.get("top", 0)
                        if o.get("originX") == "right":
                            x0 -= w_px
                        if o.get("originY") == "bottom":
                            y0 -= h_px
                        if w_px >= 5 and h_px >= 5:
                            cand = _clamp_box(round(y0 / disp_scale),
                                              round(x0 / disp_scale),
                                              round(h_px / disp_scale),
                                              round(w_px / disp_scale))
                            if max(abs(a - b) for a, b in
                                   zip(cand, (rr0, cc0, chh, cww))) > 2:
                                ss.crop_prev_box = ss.crop_roi_box
                                ss.crop_roi_box = cand
                                st.rerun()   # remount canvas on the new ROI

                b1, b2 = st.columns([1, 1])
                if b1.button("⏪ Undo last selection", key="crop_undo",
                             disabled="crop_prev_box" not in ss,
                             help="Restores the ROI from before the most "
                                  "recent box — handy after a stray drag."):
                    ss.crop_roi_box = ss.pop("crop_prev_box")
                    st.rerun()
                if b2.button("↩️ Reset to full image", key="crop_reset"):
                    ss.crop_prev_box = ss.crop_roi_box
                    ss.crop_roi_box = (0, 0, H, W)
                    st.rerun()
                st.caption("Tip: switch to **Numeric entry** to nudge the "
                           "bounds by a pixel or two without re-dragging.")
        else:
            d0, d1, d2, d3 = _clamp_box(*ss.crop_roi_box)
            c1, c2, c3, c4 = st.columns(4)
            r0 = c1.number_input("row0", 0, H - 1, d0, key="crop_r0")
            col0 = c2.number_input("col0", 0, W - 1, d1, key="crop_c0")
            hh = c3.number_input("height", 1, H, d2, key="crop_h")
            ww = c4.number_input("width", 1, W, d3, key="crop_w")
            ss.crop_roi_box = _clamp_box(r0, col0, hh, ww)

        rr0, cc0, hh, ww = _clamp_box(*ss.crop_roi_box)
        ss.g8 = seg.crop_roi(g, (rr0, cc0, hh, ww))
        st.caption("Applied crop: "
                   + ("none — analyzing the full image"
                      if (rr0, cc0, hh, ww) == (0, 0, H, W)
                      else f"row0={rr0}, col0={cc0}, height={hh}, width={ww}"))

        cimg, chist = st.columns(2)
        with cimg:
            fig, ax = plt.subplots(); ax.imshow(ss.g8, cmap="gray", vmin=0, vmax=255)
            ax.set_title("Grayscale ROI"); ax.axis("off"); st.pyplot(fig)
        with chist:
            fig, ax = plt.subplots(); ax.hist(ss.g8.ravel(), bins=256, range=(0, 255))
            ax.set_title("Intensity histogram"); ax.set_xlabel("0=black … 255=white")
            st.pyplot(fig)
        st.info("Image ready. Go to **Stage 2 · Phase Segmentation** to pick thresholds.")


# ===========================================================================
# STAGE 2 -- Phase Segmentation
# ===========================================================================
def step_segment():
    st.header("2 · Phase Segmentation")
    if "g8" not in ss:
        st.warning("No image yet. Go back to **Stage 1** and upload an SEM image first.")
        return

    g8 = ss.g8
    n_phases = st.number_input(
        "Number of phases", 1, 6, 3, key="n_phases",
        help="How many distinct materials (phases) to identify. In a PCFC composite "
             "electrode there are usually exactly three: pore (darkest), the electron "
             "conductor NiO, and the ion conductor BZCYYb (brightest). The DREAM.3D "
             "template in Stage 4 requires exactly 3 phases.")

    default_names = ["Pore", "NiO", "BZCYYb", "Phase_4", "Phase_5", "Phase_6"]

    # Seed each slider's default once. Because the sliders below use a `key`,
    # Streamlit persists their value under that key and IGNORES the `value=`
    # argument on every rerun after the first -- which is exactly why the button
    # appeared dead. The fix: write the suggestion straight into the slider keys,
    # then rerun.
    for i in range(int(n_phases)):
        ss.setdefault(f"rng{i}", (i * 80, min(i * 80 + 79, 255)))

    if st.button("Auto-suggest thresholds (multi-Otsu)",
                 help="Automatically proposes an intensity band per phase using the "
                      "multi-Otsu method, which places cuts at the valleys of the "
                      "histogram. Treat it as a starting point: on noisy or "
                      "low-contrast images the valleys can land in the wrong place, "
                      "so adjust the sliders while watching both the histogram and "
                      "the colored preview."):
        names = [ss.get(f"nm{i}", default_names[i]) for i in range(int(n_phases))]
        for i, t in enumerate(seg.suggest_thresholds_multiotsu(g8, names)):
            ss[f"rng{i}"] = (int(t.low), int(t.high))
        st.rerun()

    st.caption("Intensity scale: 0 = black … 255 = white. Pixels outside every band are tracked "
               "as *Unclassified* (not silently reassigned).")

    thresholds = []
    for i in range(int(n_phases)):
        c1, c2 = st.columns([1, 2])
        nm = c1.text_input(f"Phase {i+1} name", value=default_names[i], key=f"nm{i}")
        rng = c2.slider(f"{nm} intensity range", 0, 255, key=f"rng{i}")
        thresholds.append(seg.PhaseThreshold(nm, rng[0], rng[1]))    

    ok, msg = seg.check_threshold_ranges(thresholds)
    if not ok:
        st.warning(msg)
    else:
        if st.button("Run segmentation", type="primary"):
            ss.res = seg.segment_by_thresholds(g8, thresholds)
            ss.names = [t.name for t in thresholds]

    if "res" in ss:
        res = ss.res
        disp = np.zeros((*res.phase_id.shape, 3))
        cmap = plt.get_cmap("tab10")
        for p, t in enumerate(res.thresholds, start=1):
            disp[res.phase_id == p] = cmap((p - 1) % 10)[:3]
        fig, ax = plt.subplots(); ax.imshow(disp); ax.set_title("Phase segmentation")
        ax.axis("off"); st.pyplot(fig)

        ss.vf = seg.volume_fractions(res)
        st.subheader(
            "Volume fractions",
            help="The share of classified pixels belonging to each phase. Pixels "
                 "outside every intensity band are counted separately as Unclassified "
                 "rather than forced into a phase. Sanity check before moving on: a "
                 "realistic electrode pore fraction is roughly 0.20–0.40. This "
                 "pipeline is known to systematically UNDER-estimate pore fraction "
                 "when the pore band is set too narrow, so if Pore reads only a few "
                 "percent, widen its band before trusting anything downstream.")
        vf_df = pd.DataFrame([{"Phase": k, "VolumeFraction": v} for k, v in ss.vf.items()])
        st.dataframe(vf_df)
        pore_vf = next((v for k, v in ss.vf.items() if "pore" in k.lower()), None)
        if pore_vf is not None and pore_vf < 0.05:
            st.error(f"Pore fraction is only {pore_vf:.1%} — far below the realistic "
                     "20–40% range. This matches the known under-segmentation issue; "
                     "widen the Pore intensity band. A pore fraction this low will "
                     "also make DREAM.3D fail or produce a meaningless volume.")


# ===========================================================================
# STAGE 3 -- Particle Sizes (PSD)
# ===========================================================================
def step_psd():
    st.header("3 · Particle Sizes")
    if "res" not in ss:
        st.warning("Run **Stage 2 · Phase Segmentation** first.")
        return

    st.subheader("PSD options")
    c1, c2 = st.columns(2)
    with c1:
        min_area = st.number_input(
            "Min object area (px)", value=15, min_value=1, key="min_area",
            help="Objects smaller than this many pixels are treated as noise and "
                 "dropped before sizes are measured. Raise it if speckle noise is "
                 "producing hundreds of tiny 'particles'; lower it if genuine fine "
                 "particles are being discarded.")
        use_ws = st.checkbox(
            "Watershed split touching particles", value=True, key="use_ws",
            help="Watershed splitting separates particles that touch, so a clump is "
                 "not measured as one large particle. It generally increases the "
                 "particle count and lowers the mean size. NOTE for MATLAB "
                 "cross-validation: scikit-image and MATLAB watershed differ, so the "
                 "cleanest comparison is with watershed OFF in both pipelines.")
    with c2:
        h_max = st.slider(
            "Watershed h-max (lower = more splitting)", 0.1, 2.0, 0.30, 0.05, key="h_max",
            help="Controls how aggressively watershed splits touching particles. Lower "
                 "values split more readily; higher values keep more clusters intact. "
                 "0.30 matches the MATLAB reference (imhmax h=0.3).")
        shape_filter = st.checkbox(
            "Apply shape filter (eccentricity/solidity)", value=False, key="shape_filter",
            help="Removes objects that are too elongated or too irregular to be "
                 "genuine particles. OFF by default to match the updated MATLAB "
                 "reference, which filters on size only.")
        num_bins = st.number_input(
            "PDF bins", value=10, min_value=2, key="num_bins",
            help="Number of bins for the particle-size histogram. More bins give "
                 "finer resolution but need more particles to look smooth; 10 matches "
                 "the MATLAB reference plots.")

    if st.button("Compute PSD for all phases", type="primary"):
        cfg = psdmod.PSDConfig(min_area_px=int(min_area), use_watershed=use_ws, h_max=h_max,
            apply_shape_filter=shape_filter, num_bins=int(num_bins))
        ss.psd_cfg = cfg          # recorded with the run in Stage 6
        with st.spinner("Computing PSD for all phases... (watershed on large masks can take ~10-30 s)"):
            ss.psds = [psdmod.measure_phase(ss.res.masks[n], ss.um_per_px, n, cfg) for n in ss.names]
    
    if "psds" in ss:
        table = psdmod.summary_table(ss.psds, ss.vf)
        ss.psd_table = table
        st.subheader(
            "Size distribution results",
            help="Each phase's size distribution is summarized by a lognormal fit — a "
                 "mean (µ) and spread (σ) of ln(diameter) — the standard model for "
                 "particle sizes and exactly what DREAM.3D's StatsGenerator expects. "
                 "Phases with very few detected particles fit poorly (µ or σ may even "
                 "be NaN); a NaN fit cannot be sent to DREAM.3D, so fix the "
                 "segmentation for that phase first.")
        st.dataframe(table)
        cols = st.columns(min(len(ss.psds), 3))
        for col, psd in zip(cols * 2, ss.psds):
            with col:
                fig, ax = plt.subplots()
                if psd.pdf_density.size:
                    w = (psd.pdf_bin_centers[1] - psd.pdf_bin_centers[0]) if psd.pdf_bin_centers.size > 1 else 0.1
                    ax.bar(psd.pdf_bin_centers, psd.pdf_density, width=w, alpha=0.6)
                if psd.fit_x.size:
                    ax.plot(psd.fit_x, psd.fit_y, "k-", lw=2)
                ax.set_title(f"{psd.name} PDF"); ax.set_xlabel("d [µm]"); ax.set_ylabel("pdf")
                st.pyplot(fig)
        st.download_button("Download summary CSV", table.to_csv(index=False),
                           f"{safe_name(ss.get('sample_id', 'sample'))}_SEM_summary.csv",
                           "text/csv")


# ===========================================================================
# STAGE 4 -- Generate 3D Structure (DREAM.3D)
# ===========================================================================
def step_generate():
    st.header(
        "4 · Generate 3D Structure",
        help="The 3D structure is generated from the measured statistics (phase "
             "fractions and particle-size distributions), not reconstructed from the "
             "image itself. Any metric computed from it describes a statistically "
             "equivalent structure rather than the exact imaged region — the standard "
             "2D→3D limitation, which must be stated when reporting results.")
    st.write("This step takes the statistics you measured — phase fraction and the size "
             "distribution per phase — and synthesizes a statistically equivalent 3D volume.")

    if "psds" not in ss:
        st.warning("Compute particle sizes in **Stage 3** first.")
        return

    st.subheader("Template & grid")
    template = st.text_input(
        "Template pipeline JSON path", value="3D_generator_simple.json", key="template",
        help="Path to the DREAM.3D pipeline template that defines how the 3D volume is "
             "built. The bundled 3D_generator_simple.json expects exactly 3 phases and "
             "the DREAM.3D 6.5.171 JSON format.")
    c1, c2, c3 = st.columns(3)
    nx = c1.number_input("nx", 10, 1000, 100,
                         help="Number of voxels along x. Bigger volumes give smoother "
                              "statistics but take longer to generate and analyze.")
    ny = c2.number_input("ny", 10, 1000, 100, help="Number of voxels along y.")
    nz = c3.number_input("nz", 10, 1000, 50, help="Number of voxels along z.")
    res_um = st.number_input(
        "Voxel size (µm)", 0.001, 10.0, 0.1, format="%.3f", key="gen_res_um",
        help="The physical edge length of one voxel, in microns. Choose it so that a "
             "typical particle (see the mean diameters from Stage 3) spans at least "
             "5–10 voxels; otherwise particles cannot be resolved. Stage 5 reuses "
             "this value for physical areas and lengths.")

    st.subheader("DREAM.3D runner")
    settings = load_settings()
    default_runner = settings.get(
        "runner_exe", r"C:\DREAM3D\DREAM3D-6.5.171-Win64\PipelineRunner.exe")
    runner_exe = st.text_input(
        "PipelineRunner.exe path", value=default_runner, key="runner_exe",
        help="Full path to DREAM.3D's headless PipelineRunner.exe (in the DREAM.3D "
             "bin folder — NOT DREAM3D.exe, which opens the GUI). Set it once; it is "
             "saved between sessions. For safety the app refuses to execute any file "
             "not named PipelineRunner.exe.")
    if runner_exe != default_runner:
        settings["runner_exe"] = runner_exe
        save_settings(settings)

    # ---- Option A: export the pipeline JSON --------------------------------
    st.subheader("Option A — export pipeline JSON")
    if st.button("Write pipeline JSON"):
        real = [p for p in ss.psds if p.name != "Unclassified"]
        if not os.path.exists(template):
            st.error(f"Template not found: {template}")
        elif len(real) != 3:
            st.error(f"Template expects 3 phases; you have {len(real)}. "
                     "Use a 3-phase segmentation or adjust the template.")
        else:
            phases = [d3d.PhaseStats(name=p.name, phase_fraction=float(ss.vf[p.name]),
                                     mu=float(p.lognormal_mu), sigma=float(p.lognormal_sigma))
                      for p in real]
            outdir = tempfile.mkdtemp()
            path = d3d.write_pipeline(template, phases, outdir, out_prefix="microstructure",
                                      dimensions=(int(nx), int(ny), int(nz)), resolution_um=res_um)
            ss.pipeline_text = open(path).read()
            st.success("Pipeline written.")
            st.code(f"PipelineRunner -p {os.path.basename(path)}", language="bash")

    if "pipeline_text" in ss:
        st.download_button("Download pipeline JSON", ss.pipeline_text,
                           "microstructure_pipeline.json", "application/json")

    # ---- Option B: one-click headless run ----------------------------------
    st.subheader("Option B — run DREAM.3D now (headless)")
    if st.button("Generate 3D microstructure (run DREAM.3D)", type="primary"):
        real = [p for p in ss.psds if p.name != "Unclassified"]
        run_ok, run_msg = validate_runner(runner_exe)
        if not os.path.exists(template):
            st.error(f"Template not found: {template}")
        elif len(real) != 3:
            st.error(f"Template expects 3 phases; you have {len(real)}.")
        elif not run_ok:
            st.error(run_msg)
        else:
            phases = [dr.PhaseParams(phase_fraction=float(ss.vf[p.name]),
                                     mu=float(p.lognormal_mu),
                                     sigma=float(p.lognormal_sigma),
                                     name=p.name) for p in real]
            # persistent, per-sample output (no more %TEMP% overwriting)
            out_vtk = str(OUTPUT_DIR /
                          f"microstructure_{safe_name(ss.get('sample_id', 'sample'))}.vtk")

            bad = [p.name for p in real
                   if not (math.isfinite(p.lognormal_mu) and math.isfinite(p.lognormal_sigma))]
            if bad:
                st.warning(f"Lognormal fit failed for: {', '.join(bad)}. "
                           "Using template defaults for those phases. Check segmentation.")

            try:
                with st.spinner("Running DREAM.3D..."):
                    vtk_path = dr.generate_microstructure(
                        template_path=template,
                        phases=phases,
                        output_vtk=out_vtk,
                        pipeline_runner_exe=runner_exe,
                        dims=(int(nx), int(ny), int(nz)),
                        resolution_um=res_um,
                        stream=True)
                ss.vol = d3d.load_phase_volume_vtk(vtk_path)
                ss.res_um = float(res_um)
                ss.gen_meta = {                    # recorded with the run in Stage 6
                    "template": str(template),
                    "dims": [int(nx), int(ny), int(nz)],
                    "resolution_um": float(res_um),
                    "vtk": str(vtk_path),
                }
                # a freshly generated volume invalidates all cached results
                for k in ("fig_threephase", "fig_boundaries", "boundary_counts",
                          "pfig_three", "pfig_pairs", "pfig_triple",
                          "tort_rows", "bres", "vf3d", "names3d"):
                    ss.pop(k, None)
                st.success(f"Generated volume {ss.vol.shape}, saved to `{vtk_path}`; "
                           "ready in **Stage 5**.")
            except Exception as e:
                st.error(str(e))


# ===========================================================================
# STAGE 5 -- 3D Analysis
# ===========================================================================
def step_analyze():
    st.header("5 · 3D Analysis")
    st.write("Tortuosity, interfacial area (DPB), triple-phase-boundary (TPB) density, "
             "and interactive 3D renders of the structure. Each plot has its own data "
             "table directly underneath so every number is next to the picture it "
             "belongs to.")

    if "vol" not in ss:
        st.warning("No 3D volume yet. Generate one in **Stage 4** first.")
        return

    vox = st.number_input(
        "Voxel size for 3D analysis (µm)", 0.001, 10.0,
        float(ss.get("res_um", 0.1)), format="%.3f", key="analysis_vox",
        help="The physical size of one voxel edge, in microns, used to convert voxel "
             "counts into physical areas and lengths. It defaults to the value used "
             "in Stage 4 — leave it unchanged unless you deliberately want the metrics "
             "rescaled.")
    axis = st.selectbox(
        "Transport axis", [0, 1, 2], format_func=lambda a: {0: "z", 1: "y", 2: "x"}[a],
        key="axis",
        help="Tortuosity is measured for transport along this direction through the "
             "structure. Pick the through-thickness direction of the electrode — the "
             "direction gas and charge must actually travel in the cell.")

    labels = sorted(int(x) for x in np.unique(ss.vol))
    names = {lab: (ss.get("names", [])[lab - 1] if ss.get("names") and 0 < lab <= len(ss["names"])
                   else f"Phase_{lab}") for lab in labels}
    ss.names3d = names

    # --- numeric analysis (cached; BFS on big volumes takes a while) --------
    if st.button("Compute / refresh 3D metrics", type="primary"):
        with st.spinner("Running tortuosity BFS and boundary analysis ..."):
            ss.tort_rows = all_phase_tortuosity(ss.vol, names, axis=int(axis))
            ss.bres = analyze_boundaries(ss.vol, names, voxel_size_um=vox)
            ss.vf3d = ev.volume_fractions_3d(ss.vol, names)
            ss.vox_used = float(vox)
            ss.axis_used = int(axis)

    if "bres" not in ss:
        st.info("Click **Compute / refresh 3D metrics** to run tortuosity + boundary "
                "analysis (a few seconds on a 100×100×50 volume).")
    else:
        st.subheader(
            "Tortuosity & percolation",
            help="Geometric tortuosity τ = shortest connected path / straight distance "
                 "along the transport axis, per phase, by breadth-first search. "
                 "'Percolates' says whether the phase spans the volume at all; "
                 "'ConnectivityRatio' is the fraction of the phase in its largest "
                 "connected cluster (1.0 = no isolated islands).")
        st.dataframe(pd.DataFrame(ss.tort_rows))

        b = ss.bres
        st.subheader(
            "Interfacial area & TPB (voxel-edge method)",
            help="DPB interfacial areas count shared voxel faces × face area. TPB "
                 "density is total triple-line length / volume by the voxel-EDGE "
                 "method. The 3×3×3 voxel COUNTS shown under the boundary plots below "
                 "are a different definition (Dr. Jin's boundarydef2.m) — the two are "
                 "cross-validation references, not interchangeable numbers; always "
                 "state which method you report.")
        m1, m2, m3 = st.columns(3)
        m1.metric("TPB density (µm⁻²)", f"{b.tpb_density_um_per_um3:.4f}")
        m2.metric("TPB length (µm)", f"{b.tpb_length_um:.2f}")
        m3.metric("Analyzed volume (µm³)", f"{b.total_volume_um3:.2f}")
        st.dataframe(pd.DataFrame([{"Pair": f"{a}–{c}", "InterfacialArea_um2": v}
                                   for (a, c), v in b.dpb_area_um2.items()]))

    # --- 3D visualization ----------------------------------------------------
    st.subheader("3D visualization (interactive — drag to rotate, scroll to zoom)")
    if len(labels) != 3:
        st.info(f"The phase-render and boundary views require exactly 3 phases; "
                f"this volume has {len(labels)} "
                f"({', '.join(names[l] for l in labels)}).")
        return

    cc1, cc2, cc3 = st.columns(3)
    max_pts = cc1.number_input(
        "Max scatter points / phase", 2000, 60000, 15000, step=1000,
        help="Cap on how many voxels are drawn per phase. Higher looks denser but "
             "renders and rotates more slowly in the browser.")
    iso_detail = cc2.radio(
        "Surface detail (higher = slower)",
        options=["coarse", "medium", "fine"], index=0, horizontal=True,
        help="Resolution of the translucent phase surfaces that give the boundary "
             "point clouds context. Coarse is fine for exploring; use fine for "
             "figures.")
    static_mode = cc3.checkbox(
        "Static matplotlib render (for paper figures)", value=False,
        help="Renders the original non-interactive matplotlib figures instead — "
             "right-click-save friendly and identical to the MATLAB reference style.")
    ds_map = {"coarse": None, "medium": 4, "fine": 3}

    v1, v2 = st.columns(2)
    render_three = v1.button("Render three-phase view")
    render_bound = v2.button("Render boundary views")

    if static_mode:
        # ---- legacy matplotlib path (kept for exportable static figures) ----
        if render_three:
            with st.spinner("Rendering three-phase scatter ..."):
                ss.fig_threephase = viz.plot_three_phase(
                    ss.vol, names, max_points_per_phase=int(max_pts))
        if render_bound:
            with st.spinner("Detecting boundaries and rendering ..."):
                ss.boundary_counts = viz.boundary_voxel_counts(ss.vol, labels)
                ss.fig_boundaries = viz.plot_boundary_subplots(
                    ss.vol, names, counts=ss.boundary_counts,
                    iso_downsample=ds_map[iso_detail])
        if "fig_threephase" in ss:
            st.pyplot(ss.fig_threephase)
        if "boundary_counts" in ss:
            st.dataframe(viz.counts_dataframe(ss.boundary_counts, names))
        if "fig_boundaries" in ss:
            st.pyplot(ss.fig_boundaries)
        return

    # ---- interactive Plotly path -------------------------------------------
    if render_three:
        with st.spinner("Building interactive three-phase view ..."):
            ss.pfig_three = vpl.three_phase_figure(
                ss.vol, names, max_points_per_phase=int(max_pts))
    if render_bound:
        with st.spinner("Detecting boundaries and building interactive views ..."):
            pairs = [(labels[0], labels[1]), (labels[0], labels[2]),
                     (labels[1], labels[2])]
            ss.pfig_pairs = []
            for pa, pb in pairs:
                fig, n = vpl.pair_boundary_figure(
                    ss.vol, names, pa, pb, iso_downsample=ds_map[iso_detail])
                ss.pfig_pairs.append((pa, pb, fig, n))
            tfig, tn = vpl.triple_boundary_figure(
                ss.vol, names, iso_downsample=ds_map[iso_detail])
            ss.pfig_triple = (tfig, tn)

    if "pfig_three" in ss:
        st.plotly_chart(ss.pfig_three, use_container_width=True)
        # data directly under the plot it belongs to
        rows = []
        for lab in labels:
            nvox = int(np.count_nonzero(ss.vol == lab))
            rows.append({"Phase": names[lab], "Voxels": nvox,
                         "VolumeFraction": nvox / ss.vol.size})
        st.caption("Data for the three-phase view above:")
        st.dataframe(pd.DataFrame(rows))

    if "pfig_pairs" in ss:
        st.markdown("#### Two-phase boundaries")
        cols = st.columns(3)
        for col, (pa, pb, fig, n) in zip(cols, ss.pfig_pairs):
            with col:
                st.plotly_chart(fig, use_container_width=True)
                key = (names[pa], names[pb])
                area = ss.bres.dpb_area_um2.get(key) if "bres" in ss else None
                data = {"Boundary": f"{names[pa]}–{names[pb]}",
                        "BoundaryVoxels (face contact)": f"{n:,}"}
                if area is not None:
                    data["InterfacialArea (µm²)"] = f"{area:.2f}"
                st.table(pd.DataFrame([data]).T.rename(columns={0: "value"}))
        if "bres" not in ss:
            st.caption("Interfacial areas in µm² appear here after you click "
                       "**Compute / refresh 3D metrics** above.")

    if "pfig_triple" in ss:
        st.markdown("#### Triple-phase boundary")
        tfig, tn = ss.pfig_triple
        tcol, dcol = st.columns([2, 1])
        with tcol:
            st.plotly_chart(tfig, use_container_width=True)
        with dcol:
            st.markdown("**Data for this plot:**")
            trows = {"TPB voxels (3×3×3 method)": f"{tn:,}"}
            if "bres" in ss:
                trows["TPB length, voxel-edge (µm)"] = f"{ss.bres.tpb_length_um:.2f}"
                trows["TPB density, voxel-edge (µm⁻²)"] = f"{ss.bres.tpb_density_um_per_um3:.4f}"
            st.table(pd.DataFrame([trows]).T.rename(columns={0: "value"}))
            st.caption("The voxel COUNT (3×3×3 neighborhood, boundarydef2.m) and the "
                       "voxel-edge DENSITY are different definitions by construction. "
                       "Report one and cite the other as a cross-check.")

    if "bres" in ss:
        st.success("Metrics computed. Continue to **Stage 6 · Results & Export** to "
                   "record this reconstruction and compare samples.")


# ===========================================================================
# STAGE 6 -- Results & Export
# ===========================================================================
def _provenance_dict() -> dict:
    """Everything needed to reproduce the run now, recorded with the run.

    Written into every saved run and every export. If a field is missing it is
    recorded as None rather than omitted, so a gap in the record is visible
    instead of invisible.
    """
    prov: dict = {
        "app_version": APP_VERSION,
        "sample_id": ss.get("sample_id"),
        "um_per_px": ss.get("um_per_px"),
        "scale_bar_um": ss.get("scale_bar_um"),
        "scale_bar_px": ss.get("scale_bar_px"),
        "crop_box": list(ss.get("crop_roi_box")) if ss.get("crop_roi_box") else None,
        "voxel_um": ss.get("vox_used"),
        "transport_axis": {0: "z", 1: "y", 2: "x"}.get(ss.get("axis_used")),
        "volume_dims": list(ss.vol.shape) if "vol" in ss else None,
        "dream3d_seed": None,   # not yet exposed by Stage 4 -- see note in manual
    }
    gm = ss.get("gen_meta") or {}
    prov["dream3d_template"] = gm.get("template")
    prov["dream3d_resolution_um"] = gm.get("resolution_um")

    res = ss.get("res")
    if res is not None and getattr(res, "thresholds", None):
        prov["segmentation_bands"] = {
            t.name: [int(t.low), int(t.high)] for t in res.thresholds}
    else:
        prov["segmentation_bands"] = None

    cfg = ss.get("psd_cfg")
    if cfg is not None:
        prov["psd_config"] = {
            "min_area_px": getattr(cfg, "min_area_px", None),
            "use_watershed": getattr(cfg, "use_watershed", None),
            "h_max": getattr(cfg, "h_max", None),
            "apply_shape_filter": getattr(cfg, "apply_shape_filter", None),
            "num_bins": getattr(cfg, "num_bins", None),
        }
    else:
        prov["psd_config"] = None
    return prov


def _current_metric_rows():
    """MetricRows for whatever is currently in session state, or None."""
    if "bres" not in ss or "tort_rows" not in ss or "vf3d" not in ss:
        return None
    tpb_count = None
    try:
        if "pfig_triple" in ss:
            tpb_count = int(ss.pfig_triple[1])
    except Exception:
        tpb_count = None
    return ev.build_metric_rows(
        tortuosity_rows=ss.tort_rows,
        boundary_result=ss.bres,
        vf3d=ss.vf3d,
        vf2d=ss.get("vf"),
        psds=ss.get("psds"),
        tpb_voxel_count=tpb_count,
    )


def step_evaluate():
    st.header("6 · Results & Export")
    st.write("Reports every metric with its units, method, and provenance; records "
             "repeated reconstructions so their spread is visible; and compares the "
             "saved samples against each other. This stage applies no quality "
             "thresholds of its own — see *Compare against your own criteria* below "
             "if you want pass/fail bands.")

    rows = _current_metric_rows()

    # ---- metric definitions -------------------------------------------------
    with st.expander("📚 What each metric means"):
        for key in ("volume_fraction", "percolation", "tortuosity", "connectivity",
                    "tpb_density", "dpb_area", "replicates"):
            st.markdown(ev.DEFINITIONS[key])
            st.markdown("")

    # =======================================================================
    # A. Current sample
    # =======================================================================
    if rows is None:
        st.warning("No metrics yet for the current sample. Run **Stage 5 · 3D "
                   "Analysis** and click *Compute / refresh 3D metrics* first.")
    else:
        sample = ss.get("sample_id", "sample")
        st.subheader(f"Measured values — {sample}")

        table = pd.DataFrame([r.as_dict() for r in rows])
        prov_filter = st.multiselect(
            "Show provenance", sorted(table["Provenance"].unique()),
            default=sorted(table["Provenance"].unique()),
            help="'2D-measured' rows come straight from the segmented micrograph. "
                 "'synthetic-3D' rows are measured on the DREAM.3D volume built "
                 "from those 2D statistics and inherit the synthesis error. Keeping "
                 "them visibly separated is the point of this table.")
        st.dataframe(table[table["Provenance"].isin(prov_filter)],
                     use_container_width=True, hide_index=True)

        st.download_button(
            "⬇️ Download this sample's metrics (CSV)",
            table.to_csv(index=False),
            f"{safe_name(sample)}_metrics.csv", "text/csv")

        # ---- record this run ------------------------------------------------
        st.subheader("Record this reconstruction")
        st.caption("DREAM.3D synthesis is stochastic, so one volume is one draw. "
                   "Re-run Stage 4 and Stage 5 and record each result as a further "
                   "replicate; the spread across replicates is the reconstruction "
                   "noise floor.")
        existing = [r for r in ss.runs if str(r.get("Sample")) == str(sample)]
        next_rep = max([int(r.get("Replicate", 0)) for r in existing], default=0) + 1
        c1, c2 = st.columns([1, 3])
        rep = c1.number_input("Replicate #", 1, 999, next_rep, key="rep_no",
                              help="Auto-incremented. Runs are appended, never "
                                   "overwritten — remove one explicitly below if it "
                                   "was a mistake.")
        if c2.button(f"💾 Record run for {sample} (replicate {int(rep)})",
                     type="primary"):
            rec = ev.metrics_record(sample, rows, replicate=int(rep),
                                    provenance=_provenance_dict())
            ss.runs.append(rec)
            save_runs(ss.runs)
            st.success(f"Recorded. {len(ss.runs)} run(s) on file "
                       f"({len(existing) + 1} for {sample}).")
            rerun()

    # =======================================================================
    # B. Recorded runs, replicate spread, ranking
    # =======================================================================
    st.divider()
    st.subheader("Recorded runs")

    if not ss.runs:
        st.caption("No runs recorded yet. Run each sample through Stages 1–5 and "
                   "record it here to build the comparison.")
        return

    runs_df = pd.DataFrame(ss.runs)
    labels = ev.metric_labels(rows) if rows else {}
    directions = ev.metric_directions(rows) if rows else {}

    counts = runs_df.groupby("Sample").size().reset_index(name="Replicates recorded")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    with st.expander("All recorded runs (raw)"):
        st.dataframe(runs_df, use_container_width=True, hide_index=True)
        rm = st.selectbox("Remove a run", ["—"] + [
            f"{r['Sample']} · rep {r.get('Replicate', '?')} · {r.get('RunID', '?')}"
            for r in ss.runs])
        if rm != "—" and st.button("Remove selected run"):
            rid = rm.rsplit("· ", 1)[-1].strip()
            ss.runs = [r for r in ss.runs if r.get("RunID") != rid]
            save_runs(ss.runs)
            rerun()

    # ---- aggregation ------------------------------------------------------
    agg = ev.aggregate_runs(ss.runs, labels=labels)
    agg_df = pd.DataFrame(agg)

    st.subheader("Replicate statistics (median and IQR)")
    st.caption("Median and interquartile range rather than mean and standard "
               "deviation: n is small and the reconstruction distribution is not "
               "assumed normal. With fewer than 4 replicates the reported range is "
               "the observed min–max, and n is shown so the reader can tell.")
    show = agg_df[agg_df["n"] > 0][
        ["Sample", "Metric", "n", "median", "q25", "q75", "IQR", "min", "max"]]
    st.dataframe(show, use_container_width=True, hide_index=True)

    # ---- ranking ----------------------------------------------------------
    if runs_df["Sample"].nunique() >= 2:
        st.subheader("Sample ranking")
        st.caption("Ranking compares samples against each other and needs no "
                   "thresholds, which is why it is the only comparative judgement "
                   "this tool makes. Metrics with no defensible favourable "
                   "direction are ordered descending and labelled as such — do not "
                   "read those orderings as quality orderings.")
        rank_df = pd.DataFrame(ev.rank_samples(agg, directions))
        metric_choices = sorted(rank_df["Metric"].unique())
        picked = st.multiselect(
            "Metrics to show", metric_choices,
            default=[m for m in metric_choices
                     if "TPB density" in m or "Connectivity" in m
                     or "tortuosity" in m][:6] or metric_choices[:6])
        if picked:
            st.dataframe(
                rank_df[rank_df["Metric"].isin(picked)].sort_values(
                    ["Metric", "Rank"]),
                use_container_width=True, hide_index=True)
            for key in sorted(set(
                    rank_df[rank_df["Metric"].isin(picked)]["MetricKey"])):
                note = ev.spread_overlap_note(agg, key)
                if note:
                    st.caption(f"• {labels.get(key, key)}: {note}")

    # =======================================================================
    # C. Opt-in user criteria
    # =======================================================================
    st.divider()
    with st.expander("📐 Compare against your own criteria (optional, off by default)"):
        st.caption(
            "This tool ships with no quality thresholds. If you want pass/fail "
            "bands, enter them here together with the source you took them from — "
            "a paper, a standard, or 'own judgement'. The source is recorded with "
            "the result and appears in the export, so any judgement in your output "
            "is attributable to you rather than to the software.")

        settings = load_settings()
        saved = settings.get("user_criteria", [])
        keys = sorted(labels) if labels else []
        if not keys:
            st.info("Compute metrics for a sample first so the metric list is known.")
        else:
            seed = pd.DataFrame(saved) if saved else pd.DataFrame(
                [{"metric_key": "", "lo": None, "hi": None, "source": ""}])
            for col in ("metric_key", "lo", "hi", "source"):
                if col not in seed.columns:
                    seed[col] = None
            try:
                edited = st.data_editor(
                    seed[["metric_key", "lo", "hi", "source"]],
                    num_rows="dynamic", use_container_width=True, hide_index=True,
                    column_config={
                        "metric_key": st.column_config.SelectboxColumn(
                            "Metric", options=keys, required=False),
                        "lo": st.column_config.NumberColumn("Lower bound"),
                        "hi": st.column_config.NumberColumn("Upper bound"),
                        "source": st.column_config.TextColumn(
                            "Source (required)", width="large"),
                    })
            except Exception:
                st.warning("Your Streamlit version does not support the criteria "
                           "editor. Criteria can still be set by editing "
                           "'user_criteria' in ~/.psm_pipeline_settings.json.")
                edited = seed

            crits, problems = [], []
            for _, r in edited.iterrows():
                mk = str(r.get("metric_key") or "").strip()
                if not mk:
                    continue
                c = ev.UserCriterion(
                    metric_key=mk,
                    lo=None if pd.isna(r.get("lo")) else float(r["lo"]),
                    hi=None if pd.isna(r.get("hi")) else float(r["hi"]),
                    source=str(r.get("source") or ""))
                ok, msg = c.is_valid()
                (crits if ok else problems).append(c if ok else f"{mk}: {msg}")
            for p in problems:
                st.warning(p)

            if st.button("Save criteria"):
                settings["user_criteria"] = [
                    {"metric_key": c.metric_key, "lo": c.lo, "hi": c.hi,
                     "source": c.source} for c in crits]
                save_settings(settings)
                st.success("Criteria saved with their sources.")

            if rows and crits:
                jt = pd.DataFrame(ev.apply_user_criteria(rows, crits))
                if not jt.empty:
                    st.markdown("**Current sample against your criteria**")
                    st.dataframe(jt, use_container_width=True, hide_index=True)

    # =======================================================================
    # D. Export
    # =======================================================================
    st.divider()
    st.subheader("Export")
    e1, e2, e3 = st.columns(3)
    e1.download_button("⬇️ All runs (CSV)", runs_df.to_csv(index=False),
                       "PSM_runs.csv", "text/csv",
                       help="One row per recorded reconstruction, provenance "
                            "columns included. This is the file to attach as "
                            "supplementary data.")
    e2.download_button("⬇️ Replicate statistics (CSV)", agg_df.to_csv(index=False),
                       "PSM_replicate_statistics.csv", "text/csv")
    bundle = {
        "schema_version": ev.SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "exported": _dt.datetime.now().replace(microsecond=0).isoformat(),
        "caveats": ev.STANDING_CAVEATS,
        "runs": ss.runs,
        "replicate_statistics": agg,
    }
    e3.download_button("⬇️ Full record (JSON)", json.dumps(bundle, indent=2,
                                                          default=str),
                       "PSM_record.json", "application/json",
                       help="Runs, replicate statistics, provenance, and the "
                            "standing caveats in one file.")

    st.markdown("**Reporting notes carried with every export**")
    for c in ev.STANDING_CAVEATS:
        st.caption("• " + c)



# ===========================================================================
# MANUAL (in-app)
# ===========================================================================
IN_APP_MANUAL = """
## PCFC Microstructure Pipeline — quick manual

**What this app does.** It turns a 2D SEM image of a PCFC electrode into 3D
microstructure metrics: it segments phases, measures particle-size distributions,
synthesizes a statistically equivalent 3D volume with DREAM.3D, computes
tortuosity and phase-boundary metrics, and judges the result against
literature-based criteria.

**The stages (use the map on the left):**

1. **Setup & Calibration** — pick the sample ID (OFA8/OFA10/OFA11), upload the
   SEM image and set the scale bar (µm ↔ pixels).
2. **Phase Segmentation** — set an intensity band (0–255) for each phase.
   Pixels outside every band stay *Unclassified*. Check the volume fractions:
   a realistic pore fraction is ~20–40%.
3. **Particle Sizes** — measure the size distribution and fit a lognormal
   (µ, σ) per phase. These feed DREAM.3D.
4. **Generate 3D Structure** — DREAM.3D synthesizes a 3D volume **from the
   statistics**, not from the image. Output is saved per sample in `outputs/`.
5. **3D Analysis** — tortuosity, DPB interfacial area, TPB density, and
   interactive 3D renders (drag to rotate). Every plot has its data directly
   underneath it. Needs exactly 3 phases.
6. **Results & Export** — every metric with its units, method, and provenance
   (2D-measured vs. synthetic-3D); record each reconstruction as a replicate
   and see the median and IQR across them; rank the samples against each
   other; export runs, replicate statistics, and provenance. No quality
   thresholds are applied unless you enter your own, with a source.

**Things to report honestly:**
- The 3D volume is synthetic, built from 2D statistics (2D→3D limitation).
- Watershed on/off and 4- vs 8-connectivity change particle counts.
- The two TPB numbers (voxel-edge density vs. 3×3×3 voxel count) use different
  methods and are cross-validation references, not identical values.
- DREAM.3D synthesis is stochastic. One volume is one draw; record several
  replicates per sample and report the median with its IQR.
- Any pass/fail bands shown in Stage 6 were entered by you, not by the tool,
  and are exported with the source you attributed them to.

  Full details, troubleshooting, and a glossary are in `USER_MANUAL.md`.
"""


def step_manual():
    st.header("📖 Manual")
    st.markdown(IN_APP_MANUAL)
    manual_path = APP_DIR / "USER_MANUAL.md"
    if manual_path.exists():
        st.download_button("Download full manual (USER_MANUAL.md)",
                           manual_path.read_text(encoding="utf-8"), "USER_MANUAL.md", "text/markdown")
    else:
        st.caption("Place USER_MANUAL.md next to app.py to offer it for download here.")
    if st.button("← Back to pipeline"):
        goto(0)


# ===========================================================================
# Dispatch
# ===========================================================================
st.title("🔬 PCFC Electrode Microstructure Pipeline")
sidebar_map()

_DISPATCH = {
    "setup": step_setup,
    "segment": step_segment,
    "psd": step_psd,
    "generate": step_generate,
    "analyze": step_analyze,
    "evaluate": step_evaluate,
    "manual": step_manual,
}
_DISPATCH[STEPS[ss.step][0]]()
nav_footer()