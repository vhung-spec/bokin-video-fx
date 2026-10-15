"""Bokin FX on a computer: one local server, the page in a window of its own, updates, and quitting by itself.

- One copy runs at a time for each Windows user (a named mutex): a second start only opens a window on the
  running one, waiting for it if it is still starting.
- The window is Edge (or Chrome) in app mode (--app=<url>): no tabs or address bar.
- When no page has asked anything for a while (its window was closed) and nothing is being made, the app quits,
  installing a downloaded update on the way out. "Cập nhật ngay" on the page installs at once and the installer
  starts the app again; its open window reconnects by itself.
- The installer first runs the installed BokinFX.exe --close: it asks a running app to quit through /api/quit,
  and answers 3 (setup then stops with a message) while that app is making a video.
"""
import argparse
import json
import os
import shutil
import sys
import time
import urllib.request
import webbrowser
from concurrent.futures import ThreadPoolExecutor

from . import paths, server, updater

IDLE_EXIT = 300  # seconds without the page, then quit (a minimised window may only ask once a minute)
BUSY = 3  # exit code of --close while the running app is making a video
LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # this PC itself: never through a proxy
_guard = None  # the single-instance mutex (or lock file), held for the life of the process


def log_to_file():
    """An installed app has no console: what it prints goes to app.log in the data folder."""
    path = paths.log_path()
    try:
        if os.path.exists(path) and os.path.getsize(path) > 2_000_000:
            os.replace(path, path + ".old")
    except OSError:
        pass
    sys.stdout = sys.stderr = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115 - for the app's life


def ping(port, timeout=1.0):
    try:
        with LOCAL.open(f"http://127.0.0.1:{port}/api/ping", timeout=timeout) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def running_port():
    """The port of this user's running Bokin FX (all ports asked at once: on Windows a closed port takes
    about a second to say so), or None. Another Windows user's copy answers with another owner."""
    with ThreadPoolExecutor(len(paths.PORTS)) as pool:
        answers = list(pool.map(ping, paths.PORTS))
    for port, answer in zip(paths.PORTS, answers):
        if answer and answer.get("app") == paths.APP_ID and answer.get("owner") == paths.owner():
            return port
    return None


def first_copy():
    """True for the first copy of the app started by this user; later ones see the guard taken."""
    global _guard
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
        _guard = kernel32.CreateMutexW(None, False, f"Local\\BokinFX-{paths.owner()}")
        return ctypes.get_last_error() != 183  # ERROR_ALREADY_EXISTS
    import fcntl
    _guard = open(os.path.join(paths.data_dir(), "app.lock"), "w")  # noqa: SIM115 - held while running
    try:
        fcntl.flock(_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def close_running(wait=20):
    """--close, for the installer: a running Bokin FX is asked to quit. 0 once none runs, BUSY if it refuses."""
    port = running_port()
    if port is None:
        return 0
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/quit", data=b"{}", method="POST",
                                 headers={"X-Bokin": "1", "Content-Type": "application/json"})
    try:
        with LOCAL.open(req, timeout=5) as r:
            if json.loads(r.read()).get("busy"):
                return BUSY
    except (OSError, ValueError):
        pass
    deadline = time.time() + wait
    while time.time() < deadline and ping(port):
        time.sleep(0.3)
    return 0


def show(url):
    """The page in its own window: Edge or Chrome in app mode, else the default browser."""
    if os.environ.get("BOKIN_NO_BROWSER"):
        return
    if os.name == "nt":
        import subprocess
        bases = [os.environ.get(k) for k in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA")]
        for exe in (r"Microsoft\Edge\Application\msedge.exe", r"Google\Chrome\Application\chrome.exe"):
            for base in filter(None, bases):
                path = os.path.join(base, exe)
                if os.path.exists(path):
                    subprocess.Popen([path, f"--app={url}", "--window-size=1440,920"])
                    return
    webbrowser.open(url)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="BokinFX")
    ap.add_argument("--selftest", action="store_true", help="make a real video and check it, then quit")
    ap.add_argument("--updated", action="store_true", help="started again by the installer after an update")
    ap.add_argument("--close", action="store_true", help="ask a running Bokin FX to quit (for the installer)")
    args = ap.parse_args(argv)
    if args.close:
        return close_running()
    if paths.frozen() or os.environ.get("BOKIN_LOG_FILE"):
        log_to_file()
    if args.selftest:
        from . import selftest
        return selftest.run()

    if not first_copy():  # another copy is running, or still starting: show it once it answers
        deadline = time.time() + 40
        while time.time() < deadline:
            port = running_port()
            if port:
                show(f"http://127.0.0.1:{port}/")
                return 0
            time.sleep(0.5)
        return 0
    port = running_port()  # a copy from a version without the guard
    if port:
        show(f"http://127.0.0.1:{port}/")
        return 0
    shutil.rmtree(os.path.join(paths.data_dir(), "tmp"), ignore_errors=True)  # uploads cut off by a quit
    app = server.App(desktop=True)
    srv = None
    for port in paths.PORTS:
        try:
            srv = server.start(app, port)
            break
        except OSError:
            continue
    if srv is None:
        print("no free port in 7890-7899", flush=True)
        return 1
    url = f"http://127.0.0.1:{port}/"
    print(f"Bokin FX {paths.version()} at {url}", flush=True)
    if updater.parse_version(paths.version()):
        app.updater = updater.AutoUpdate(paths.version(), paths.updates_dir()).start()
    started = time.time()
    if not args.updated:
        show(url)

    shown = not args.updated
    while not app.quit.wait(2):
        if not shown and time.time() - started > 10:
            shown = True
            if app.seen <= started:  # no window came back after the update: open one
                show(url)
        if time.time() - app.seen > IDLE_EXIT and not app.busy():
            print("no page for a while: quitting", flush=True)
            if app.updater and app.updater.state.get("ready"):
                app.updater.run_installer(restart=False)
            break
    else:
        if app.relaunch and app.updater:
            app.updater.run_installer(restart=True)
        print("quitting", flush=True)
    srv.shutdown()
    return 0


def entry():
    code = 1
    try:
        code = main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else 0
    except BaseException:  # noqa: BLE001 - the log is the only place an installed app can say it
        import traceback
        traceback.print_exc()
    finally:
        try:
            sys.stdout.flush()
        except (OSError, ValueError, AttributeError):
            pass
    os._exit(code)  # worker threads may still be in ffmpeg reads
