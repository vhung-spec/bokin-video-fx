"""Installs the built installer for real and checks the installed app (CI, on Windows).

  python packaging/ci_install_test.py install <BokinFX-Setup-X.exe>
      a fresh silent install, in the folder setup offers the first time (Bokin FX on the first fixed drive other than
      Windows' own, else the user's programs folder); the installed app starts and serves its page; setup run again
      while the app is running closes it (BokinFX.exe --close) and installs over it, in the same folder; run with
      another folder (/DIR=), the app moves there: the old copy's files are gone, Windows' list of apps names the new
      folder, nothing of the user's data is touched.

  python packaging/ci_install_test.py update <BokinFX-Setup-Y.exe> <Y>
      the installed app finds version Y on a stand-in for GitHub (in the newest release's manifest, while the API
      refuses as for an office over its limit of requests), downloads and verifies it; "Cập nhật ngay"
      (POST /api/update) installs it and the installer starts the app again, now in version Y.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_releases  # noqa: E402

APP_ID = "{10897464-A25C-47D4-89B9-E0110ED62F27}"  # (BokinFX.iss)
LOG = os.path.join(os.environ.get("APPDATA", ""), "Bokin", "FX", "app.log")


def installed():
    """The folder Bokin FX is installed in, as Windows' list of apps has it (None: not installed)."""
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_ID}_is1") as key:
            return os.path.normpath(winreg.QueryValueEx(key, "InstallLocation")[0])
    except OSError:
        return None


def exe():
    return os.path.join(installed() or "", "BokinFX.exe")
BASE = "http://127.0.0.1:7890"


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {} if body is None else {"X-Bokin": "1", "Content-Type": "application/json"}
    try:
        with urllib.request.urlopen(urllib.request.Request(BASE + path, data, headers), timeout=5) as r:
            raw = r.read()
            return json.loads(raw) if r.headers.get_content_type() == "application/json" else raw.decode("utf-8")
    except (OSError, ValueError):
        return None


def stop(message):
    try:
        with open(LOG, encoding="utf-8", errors="replace") as f:
            print("".join(f.readlines()[-40:]))
    except OSError:
        pass
    sys.exit("FAILED: " + message)


def wait(test, seconds, what):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if test():
            print("ok:", what, flush=True)
            return
        time.sleep(1)
    stop(what)


def setup(installer, *more):
    code = subprocess.run([installer, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", *more]).returncode
    if code != 0:
        stop(f"setup ended with code {code}")


def start_app(update_api, update_site=None):
    env = dict(os.environ, BOKIN_NO_BROWSER="1", BOKIN_UPDATE_API=update_api)
    if update_site:
        env["BOKIN_UPDATE_SITE"] = update_site
    return subprocess.Popen([exe()], env=env)


def offered():
    """The folder setup offers the first time (as BokinFX.iss's DefaultDir works it out)."""
    import ctypes
    win = os.environ.get("SystemDrive", "C:")[0].upper()
    for letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
        root = letter + ":\\"
        if letter != win and ctypes.windll.kernel32.GetDriveTypeW(root) == 3:
            try:
                probe = os.path.join(root, "bokin-fx-ci-probe")
                os.mkdir(probe)
                os.rmdir(probe)
                return os.path.join(root, "Bokin FX")
            except OSError:
                continue
    return os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Bokin FX")


def install(installer):
    setup(installer)
    where = installed()
    print("installed in", where, flush=True)
    if not where or not os.path.exists(exe()):
        stop(f"not installed: {where}")
    if os.path.normcase(where) != os.path.normcase(os.path.normpath(offered())):
        stop(f"installed in {where}, not in the folder offered the first time, {offered()}")
    app = start_app("http://127.0.0.1:9/none")
    wait(lambda: (call("/api/ping") or {}).get("app") == "bokin-fx", 60, "the installed app answers")
    wait(lambda: "Bokin FX" in (call("/") or ""), 10, "the installed app serves its page")
    setup(installer)
    wait(lambda: app.poll() is not None, 20, "setup closed the running app before installing over it")
    if os.path.normcase(installed() or "") != os.path.normcase(where):
        stop(f"installed over it somewhere else: {installed()}")
    # moved to another folder: the app there, the old copy's files gone, the user's data kept
    data = os.path.join(os.environ.get("APPDATA", ""), "Bokin", "FX")
    keep = os.path.join(data, "ci-keep.txt")
    os.makedirs(data, exist_ok=True)
    with open(keep, "w") as f:
        f.write("mine")
    moved = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Bokin FX moved")
    setup(installer, f"/DIR={moved}")
    if os.path.normcase(installed() or "") != os.path.normcase(moved) or not os.path.exists(exe()):
        stop(f"not moved to {moved}: {installed()}")
    left = os.path.exists(where) and os.listdir(where)
    if left:
        stop(f"the old copy left files in {where}: {left}")
    if not os.path.exists(keep):
        stop("moving the app took the user's data")
    os.remove(keep)
    print("ok: moved to", moved, "and the old folder is gone", flush=True)
    app = start_app("http://127.0.0.1:9/none")
    wait(lambda: (call("/api/ping") or {}).get("app") == "bokin-fx", 60, "the moved app answers")
    call("/api/quit", {})
    wait(lambda: app.poll() is not None, 30, "the moved app quits")


def update(installer, version):
    fake = fake_releases.serve(f"fx-v{version}", installer, 8765)
    # the newest release's manifest, while the API refuses as it does for an office over its limit of requests
    app = start_app("http://127.0.0.1:8765/limited", "http://127.0.0.1:8765/site")
    wait(lambda: ((call("/api/state") or {}).get("update") or {}).get("ready"), 120,
         "the app found the next version, downloaded it and checked it")
    if not (call("/api/update", {}) or {}).get("ok"):
        stop("the page could not start the update")
    wait(lambda: app.poll() is not None, 30, "the app closed for the update")
    wait(lambda: (call("/api/ping") or {}).get("version") == version, 240, f"the app runs again, in version {version}")
    call("/api/quit", {})
    fake.shutdown()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "install":
        install(os.path.abspath(sys.argv[2]))
    elif len(sys.argv) >= 4 and sys.argv[1] == "update":
        update(os.path.abspath(sys.argv[2]), sys.argv[3])
    else:
        sys.exit(__doc__)
