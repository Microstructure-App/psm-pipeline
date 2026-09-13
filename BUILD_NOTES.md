# PSM Pipeline — .exe build notes

## One required edit to app.py (do this first)

When frozen, `app.py` lives inside the read-only-ish `_internal` bundle folder, so
`outputs/` must move to a per-user location. The launcher sets `PSM_OUTPUT_DIR`
for you; `app.py` just has to honor it.

Find this exact text:

```python
OUTPUT_DIR = APP_DIR / "outputs"
```

Replace with this exact text:

```python
OUTPUT_DIR = pathlib.Path(os.environ.get("PSM_OUTPUT_DIR", APP_DIR / "outputs"))
```

Dev behavior is unchanged (env var absent → same `outputs/` as today). Frozen
behavior: results land in `C:\Users\<you>\PSM_Pipeline\outputs\`.

## Build steps

1. Copy `launcher.py`, `psm_pipeline.spec`, and `build.bat` into the repo root
   (next to `app.py`).
2. Activate the project venv.
3. Test unfrozen first: `python launcher.py` — a native window should open with
   the app inside. If this works, the frozen build almost certainly will.
4. Run `build.bat`.
5. Result: `dist\PSM_Pipeline\PSM_Pipeline.exe`. Ship the whole folder as a zip.

## What is and is not inside the exe

Bundled: Python, Streamlit, numpy/scipy/skimage/pandas/matplotlib, plotly,
h5py, the psm package, pywebview.

NOT bundled (by design):

- **DREAM.3D 6.5.171** — Stage 4's one-click run calls the user's local
  `PipelineRunner.exe`; the path is configured in-app and stored in
  `~/.psm_pipeline_settings.json`, same as the dev version. Document this as a
  prerequisite in the README ("Stage 4 headless run requires DREAM.3D 6.5.171;
  all other stages work standalone").
- **WebView2 Runtime** — pywebview renders through Microsoft Edge WebView2,
  preinstalled on essentially all updated Windows 10/11 machines. If a target
  machine somehow lacks it, the free "Evergreen" installer from Microsoft fixes
  it. Worth one line in the README.

## Troubleshooting

- **Window opens but blank / server never starts** → read
  `C:\Users\<you>\PSM_Pipeline\logs\server.log`; the child process's full
  traceback is there.
- **Silent crash on launch** → set `console=True` in the spec, rebuild, run
  from a terminal to see the traceback.
- **`ModuleNotFoundError` at runtime** → a lazy import PyInstaller couldn't
  see. Add the package name to `OPTIONAL_COLLECT` in the spec and rebuild.
- **Antivirus quarantine** → expected occasionally with PyInstaller output;
  onedir (already chosen) minimizes it. For the symposium, distribute via the
  GitHub Release page so the download has a stable provenance.
- **First launch is slow (~5–15 s)** → normal; Streamlit initializes and
  Windows Defender scans the new binaries once. Subsequent launches are faster.

## Why onedir + child-process (for your Methods/repo docs)

- Onefile self-extracts to a temp dir on every launch (slower, more AV flags).
- Streamlit installs POSIX signal handlers at startup, which is only legal on
  a process's main thread — so the server runs as a child process of the same
  exe rather than a background thread. This is the standard pattern for
  Streamlit desktop wrappers.
