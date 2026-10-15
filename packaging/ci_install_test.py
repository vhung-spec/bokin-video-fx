"""Installs the built installer for real and checks the installed app (CI, on Windows).

  python packaging/ci_install_test.py install <BokinFX-Setup-X.exe>
      a fresh silent install; the installed app starts and serves its page; setup run again while the app is
      running closes it (BokinFX.exe --close) and installs over it.

  python packaging/ci_install_test.py update <BokinFX-Setup-Y.exe> <Y>
      the installed app finds version Y on a stand-in for GitHub, downloads and verifies it; "Cập nhật ngay"
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

EXE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "Bokin FX", "BokinFX.exe")
LOG = os.path.join(os.environ.get("APPDATA", ""), "Bokin", "FX", "app.log")
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


def setup(installer):
    code = subprocess.run([installer, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]).returncode
    if code != 0:
        stop(f"setup ended with code {code}")


def start_app(update_api):
    env = dict(os.environ, BOKIN_NO_BROWSER="1", BOKIN_UPDATE_API=update_api)
    return subprocess.Popen([EXE], env=env)


def install(installer):
    setup(installer)
    if not os.path.exists(EXE):
        stop(f"not installed: {EXE}")
    app = start_app("http://127.0.0.1:9/none")
    wait(lambda: (call("/api/ping") or {}).get("app") == "bokin-fx", 60, "the installed app answers")
    wait(lambda: "Bokin FX" in (call("/") or ""), 10, "the installed app serves its page")
    setup(installer)
    wait(lambda: app.poll() is not None, 20, "setup closed the running app before installing over it")


def update(installer, version):
    fake = fake_releases.serve(f"fx-v{version}", installer, 8765)
    app = start_app("http://127.0.0.1:8765/releases")
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
