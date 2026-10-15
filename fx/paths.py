"""Where Bokin FX keeps things.

- Its own files (the page, the version) sit next to the code, or inside the PyInstaller bundle.
- Data (projects, the log, downloaded updates): %APPDATA%\\Bokin\\FX on Windows, ~/.local/share/Bokin/FX
  elsewhere; BOKIN_FX_DATA overrides it (tests, a server).
- Finished videos: Videos\\Bokin FX in the user's folder (BOKIN_FX_OUT overrides it).
"""
import hashlib
import os
import sys

APP_ID = "bokin-fx"  # what /api/ping answers: the installer and a second start look for it
PORTS = range(7890, 7900)


def resource_dir():
    """The folder of index.html and version.txt: the bundle when installed, else this package."""
    return getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))


def version():
    """'1.0.<build>' for an installed app (CI writes version.txt), 'dev' when run from the source folder."""
    try:
        with open(os.path.join(resource_dir(), "version.txt"), encoding="utf-8") as f:
            return f.read().strip() or "dev"
    except OSError:
        return "dev"


def frozen():
    return bool(getattr(sys, "frozen", False))


def data_dir():
    path = os.environ.get("BOKIN_FX_DATA")
    if not path:
        if os.name == "nt":
            base = os.environ.get("APPDATA") or os.path.expanduser("~")
        else:
            base = os.environ.get("XDG_DATA_HOME") or os.path.join(os.path.expanduser("~"), ".local", "share")
        path = os.path.join(base, "Bokin", "FX")
    os.makedirs(path, exist_ok=True)
    return path


def owner():
    """Who this copy of the app works for (its data folder): another Windows user's copy, on the same shared
    127.0.0.1 ports, answers with a different value and is left alone."""
    return hashlib.sha256(os.path.normcase(os.path.abspath(data_dir())).encode("utf-8")).hexdigest()[:16]


def projects_dir():
    path = os.path.join(data_dir(), "projects")
    os.makedirs(path, exist_ok=True)
    return path


def updates_dir():
    return os.path.join(data_dir(), "updates")


def out_dir():
    path = os.environ.get("BOKIN_FX_OUT") or os.path.join(os.path.expanduser("~"), "Videos", "Bokin FX")
    os.makedirs(path, exist_ok=True)
    return path


def log_path():
    return os.path.join(data_dir(), "app.log")
