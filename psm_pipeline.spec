# -*- mode: python ; coding: utf-8 -*-
# PSM Pipeline -- PyInstaller spec.
# Run from the repo root, inside the project venv:  pyinstaller --noconfirm --clean psm_pipeline.spec
#
# Design notes:
#   * ONEDIR build (not onefile): faster startup, fewer antivirus false
#     positives, and Streamlit's static assets behave better.
#   * Optional packages are auto-detected: if a package isn't installed in
#     this venv it is silently skipped, so the spec keeps working as the
#     dependency list evolves (plotly, streamlit-image-coordinates, etc.).
#   * DREAM.3D is deliberately NOT bundled -- Stage 4's one-click run calls
#     the user's local PipelineRunner.exe, whose path lives in
#     ~/.psm_pipeline_settings.json exactly as in the dev version.

import glob
import importlib.util
import os

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = [("app.py", ".")]
binaries = []
hiddenimports = []

# DREAM.3D pipeline template JSONs -- Stage 4 loads these by relative path.
for _json in ["3D_generator_simple.json"]:
    if os.path.exists(_json):
        datas.append((_json, "."))
if os.path.isdir("templates"):
    datas.append(("templates", "templates"))

# psm package: freeze whatever `import psm` resolves to in this venv
# (works for both the psm/ subdirectory layout and the pip install -e . layout).
hiddenimports += collect_submodules("psm")
if os.path.isdir("psm"):
    datas.append(("psm", "psm"))

# Packages that need full collection (data files + metadata + submodules).
# Required ones raise if missing; optional ones are skipped if not installed.
REQUIRED_COLLECT = ["streamlit", "webview"]
OPTIONAL_COLLECT = [
    "PIL",
    "matplotlib",
    "plotly",
    "skimage",
    "imageio",
    "tifffile",
    "h5py",
    "streamlit_image_coordinates",
    "streamlit_drawable_canvas",
]

def _collect(pkg):
    d, b, h = collect_all(pkg)
    datas.extend(d)
    binaries.extend(b)
    hiddenimports.extend(h)

for pkg in REQUIRED_COLLECT:
    _collect(pkg)

for pkg in OPTIONAL_COLLECT:
    if importlib.util.find_spec(pkg) is not None:
        _collect(pkg)

# Submodule-only collection (hooks handle their data, but lazy imports inside
# psm modules can hide these from PyInstaller's static analysis).
for pkg in ["sklearn", "scipy", "pandas"]:
    if importlib.util.find_spec(pkg) is not None:
        hiddenimports += collect_submodules(pkg)

a = Analysis(
    ["launcher.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PSM_Pipeline",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,   # flip to True temporarily if you need to see tracebacks
    icon=None,       # set to "psm.ico" if/when you make one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="PSM_Pipeline",
)
