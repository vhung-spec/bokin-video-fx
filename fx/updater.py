"""Self-update from this repository's GitHub releases (a public repository: no login, no key).

A release is tagged fx-v<version> and carries the installer BokinFX-Setup-<version>.exe; GitHub Actions only
publishes one after building, installing and testing it. AutoUpdate looks when the app starts, every 30 minutes,
and when the page asks (each time its window opens, or the version is clicked),
fetches a newer installer in the background and keeps it only if its size and SHA-256 are the ones GitHub lists for
the file (the asset's "digest", else the "SHA-256:" line the workflow writes into the release notes). A release
whose file cannot be checked is never used.

The installer runs when the app quits by itself, or at once when the page asks ("Cập nhật ngay"), and then starts
the app again. Only api.github.com and GitHub's download servers are contacted, with nothing about the computer.
Whoever can publish releases in the repository can update every installed app: keep that account protected.
"""
import collections
import hashlib
import json
import os
import re
import subprocess
import threading
import time
import urllib.request

RELEASES = "https://api.github.com/repos/vhung-spec/bokin-video-fx/releases?per_page=50"
TAG = "fx-v"
INSTALLER = re.compile(r"^BokinFX-Setup-[0-9.]+\.exe$")
CHECK_EVERY = 30 * 60
SOONEST = 60  # seconds between two checks the page asks for

Release = collections.namedtuple("Release", "version numbers url name size sha256")


def parse_version(text):
    """'1.0.12' -> (1, 0, 12); None for 'dev' and anything else that is not a version number."""
    parts = str(text).strip().split(".")
    if not all(p.isdigit() for p in parts):
        return None
    return tuple(int(p) for p in parts)


def _open(url, accept):
    req = urllib.request.Request(url, headers={"Accept": accept, "User-Agent": "BokinFX-update"})
    return urllib.request.urlopen(req, timeout=60)


def latest(api=None):
    """The newest published release that has a checkable installer, or None."""
    with _open(api or os.environ.get("BOKIN_UPDATE_API") or RELEASES, "application/vnd.github+json") as r:
        listing = json.loads(r.read())
    found = None
    for rel in listing:
        tag = rel.get("tag_name") or ""
        numbers = parse_version(tag[len(TAG):]) if tag.startswith(TAG) else None
        if not numbers or rel.get("draft") or rel.get("prerelease") or (found and numbers <= found.numbers):
            continue
        for asset in rel.get("assets") or []:
            if not INSTALLER.match(asset.get("name") or ""):
                continue
            sha = (asset.get("digest") or "").lower().partition("sha256:")[2]
            if not re.fullmatch(r"[0-9a-f]{64}", sha):
                noted = re.search(r"SHA-256:\s*([0-9a-fA-F]{64})", rel.get("body") or "")
                sha = noted.group(1).lower() if noted else ""
            if sha:
                found = Release(".".join(map(str, numbers)), numbers, asset["browser_download_url"],
                                asset["name"], asset.get("size"), sha)
            break
    return found


def verified(path, release):
    if release.size is not None and os.path.getsize(path) != release.size:
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest() == release.sha256


def _keep_only(folder, name):
    for other in os.listdir(folder):
        if other != name:
            try:
                os.remove(os.path.join(folder, other))
            except OSError:
                pass


def fetch(release, folder):
    """The release's installer in `folder`, downloaded once and verified. Older files there are deleted only
    once this one is in place, so a ready installer never disappears before its replacement is."""
    os.makedirs(folder, exist_ok=True)
    target = os.path.join(folder, release.name)
    if os.path.exists(target) and verified(target, release):
        _keep_only(folder, release.name)
        return target
    partial = target + ".download"
    with _open(release.url, "application/octet-stream") as r, open(partial, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    if not verified(partial, release):
        os.remove(partial)
        raise ValueError("downloaded installer does not match the published SHA-256")
    os.replace(partial, target)
    _keep_only(folder, release.name)
    return target


class AutoUpdate:
    """state: what the page shows. ready = an installer for a newer version is downloaded and verified."""

    def __init__(self, current, folder, every=CHECK_EVERY):
        self.numbers, self.folder, self.every = parse_version(current), folder, every
        self.installer = None
        self.state = {"current": current, "available": None, "ready": False, "error": None, "checked": None}
        self.wake = threading.Event()

    def start(self):
        if self.numbers is not None:  # a build from the source folder ("dev") never updates itself
            threading.Thread(target=self._run, daemon=True, name="fx-update").start()
        return self

    def _run(self):
        time.sleep(3)
        while True:
            self.check()
            self.wake.wait(self.every)
            self.wake.clear()

    def check_soon(self):
        """Look again now (the page asked), unless that was done a moment ago."""
        if time.time() - (self.state.get("checked") or 0) > SOONEST:
            self.wake.set()

    def check(self):
        try:
            release = latest()
            self.state["checked"] = time.time()
            if release and release.numbers > self.numbers:
                installer = fetch(release, self.folder)
                self.installer = installer
                self.state.update(available=release.version, ready=True)
                print(f"update {release.version} ready: it is installed when the app closes", flush=True)
            self.state["error"] = None
        except Exception as e:  # noqa: BLE001 - offline or GitHub busy: next time
            self.state["error"] = str(e)[:200]

    def run_installer(self, restart):
        """Start the verified installer on its own, silently (it closes this app). False if there is none."""
        if not self.installer or not os.path.exists(self.installer):
            return False
        args = [self.installer, "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART"]
        if restart:
            args.append("/RELAUNCH=1")
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        subprocess.Popen(args, close_fds=True, creationflags=flags)
        print(f"running the installer of {self.state['available']}", flush=True)
        return True
