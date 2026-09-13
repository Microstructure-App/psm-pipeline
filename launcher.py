"""
PSM Pipeline desktop launcher.

Architecture (this pattern matters):
  - The exe re-invokes ITSELF with `--run-streamlit <port>` as a child process.
  - The child runs the Streamlit server on its own main thread (Streamlit
    installs signal handlers, which raise ValueError if attempted from a
    background thread -- this is why we do NOT use threading here).
  - The parent waits for the port to open, then shows a pywebview window
    pointed at http://127.0.0.1:<port>.
  - Closing the window terminates the server child.

Works both frozen (PyInstaller) and unfrozen (python launcher.py) for testing.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

APP_NAME = "PSM Pipeline"
CHILD_FLAG = "--run-streamlit"

# Windows: CREATE_NO_WINDOW so the server child never flashes a console.
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_dir() -> str:
    """Directory containing bundled data files (app.py, psm/)."""
    if is_frozen():
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def user_data_dir() -> str:
    """Writable per-user directory for outputs + logs (survives reinstalls)."""
    d = os.path.join(os.path.expanduser("~"), "PSM_Pipeline")
    os.makedirs(os.path.join(d, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    return d


def find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def wait_for_server(port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.25)
    return False


# ---------------------------------------------------------------------------
# Child mode: run the Streamlit server (main thread of its own process)
# ---------------------------------------------------------------------------
def run_streamlit_server(port: int) -> None:
    # Relative resource paths (Stage 4's template JSON, etc.) resolve against
    # the bundle dir -- matching dev, where cwd is the repo root.
    os.chdir(bundle_dir())
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("PSM_OUTPUT_DIR", os.path.join(user_data_dir(), "outputs"))
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

    from streamlit import config as st_config
    from streamlit.web import bootstrap

    app_path = os.path.join(bundle_dir(), "app.py")
    if not os.path.exists(app_path):
        print(f"FATAL: app.py not found at {app_path}", flush=True)
        sys.exit(2)

    options = {
        "global.developmentMode": False,
        "server.headless": True,
        "server.address": "127.0.0.1",
        "server.port": port,
        "server.fileWatcherType": "none",
        "server.runOnSave": False,
        "server.enableCORS": True,
        "server.enableXsrfProtection": True,
        "browser.gatherUsageStats": False,
        "client.toolbarMode": "minimal",
        "theme.base": "light",
        # Root font size in px (default 16). Scales text AND widget chrome,
        # including the canvas-rendered dataframes that injected CSS can't
        # reach. Requires Streamlit >= 1.43; older versions skip it below.
        "theme.baseFontSize": 20,
    }
    for key, value in options.items():
        try:
            st_config.set_option(key, value)
        except Exception as exc:
            # An unrecognized option on an older Streamlit must not be fatal:
            # the app is still perfectly usable at the default size.
            print(f"config: skipped {key}={value!r} ({exc})", flush=True)

    try:
        # Streamlit >= ~1.18 signature: run(path, is_hello, args, flag_options)
        bootstrap.run(app_path, False, [], {})
    except TypeError:
        # Older signature: run(path, command_line, args, flag_options)
        bootstrap.run(app_path, "", [], {})


# ---------------------------------------------------------------------------
# Parent mode: spawn server child, open the window
# ---------------------------------------------------------------------------
def spawn_server(port: int, log_path: str) -> subprocess.Popen:
    if is_frozen():
        cmd = [sys.executable, CHILD_FLAG, str(port)]
    else:
        cmd = [sys.executable, os.path.abspath(__file__), CHILD_FLAG, str(port)]
    log = open(log_path, "w", buffering=1, encoding="utf-8", errors="replace")
    return subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW,
    )


def fail_dialog(message: str) -> None:
    if os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, APP_NAME, 0x10)
    else:
        print(message, file=sys.stderr)


def main() -> None:
    if CHILD_FLAG in sys.argv:
        port = int(sys.argv[sys.argv.index(CHILD_FLAG) + 1])
        run_streamlit_server(port)
        return

    port = find_free_port()
    log_path = os.path.join(user_data_dir(), "logs", "server.log")
    proc = spawn_server(port, log_path)

    if not wait_for_server(port):
        proc.terminate()
        fail_dialog(
            "The analysis server failed to start within 120 s.\n\n"
            f"See the log for details:\n{log_path}"
        )
        sys.exit(1)

    import webview  # deferred: heavy import, and only the parent needs it

    webview.create_window(
        APP_NAME,
        f"http://127.0.0.1:{port}",
        width=1440,
        height=900,
        min_size=(1100, 720),
        confirm_close=False,
    )
    try:
        webview.start()
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    os._exit(0)


if __name__ == "__main__":
    main()
