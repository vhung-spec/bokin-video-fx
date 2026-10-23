import hashlib
import json
import os
import ssl
import time
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from tests import helpers
from fx import updater

INSTALLER = b"MZ fake installer " * 1000


class FakeGitHub(BaseHTTPRequestHandler):
    releases = []
    manifest = None  # (the newest release's manifest; None: it has none)
    ranges = True  # (downloads can go on from where they stopped)
    sent = 0  # (bytes of installers sent)

    def log_message(self, *a):
        pass

    def answer(self, code, body=b"", headers=()):
        self.send_response(code)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/releases"):
            return self.answer(200, json.dumps(self.releases).encode())
        if self.path == "/limited":  # (an Internet connection over the API's limit of requests)
            return self.answer(403, b'{"message": "API rate limit exceeded"}', [("X-RateLimit-Remaining", "0")])
        if self.path == "/site/releases/latest/download/" + updater.MANIFEST and self.manifest is not None:
            return self.answer(200, json.dumps(self.manifest).encode())
        if self.path.startswith(("/download/", "/site/releases/download/fx-v")):
            start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-") or 0) if FakeGitHub.ranges else 0
            FakeGitHub.sent += len(INSTALLER) - start
            if start:
                return self.answer(206, INSTALLER[start:], [("Content-Range", f"bytes {start}-{len(INSTALLER) - 1}/"
                                                                               f"{len(INSTALLER)}")])
            return self.answer(200, INSTALLER)
        self.answer(404)


class Updates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeGitHub)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"
        os.environ["BOKIN_UPDATE_API"] = cls.base + "/releases"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        os.environ.pop("BOKIN_UPDATE_API", None)

    def release(self, tag, digest=True, body="", draft=False, name=None):
        name = name or f"BokinFX-Setup-{tag[4:]}.exe"
        sha = hashlib.sha256(INSTALLER).hexdigest()
        asset = {"name": name, "size": len(INSTALLER), "browser_download_url": f"{self.base}/download/{name}"}
        if digest:
            asset["digest"] = "sha256:" + sha
        return {"tag_name": tag, "draft": draft, "prerelease": False, "assets": [asset],
                "body": body.replace("<sha>", sha)}

    def tearDown(self):
        os.environ.pop("BOKIN_UPDATE_SITE", None)
        os.environ["BOKIN_UPDATE_API"] = self.base + "/releases"
        FakeGitHub.manifest, FakeGitHub.ranges = None, True

    def test_public_releases(self):
        """The source repository is private (its releases need a login): the apps read the installers' repository,
        its newest release's manifest first (a plain download, not the API), and the workflow publishes it."""
        self.assertTrue(updater.RELEASES.startswith("https://api.github.com/repos/vhung-spec/bokin-fx-releases/releases"))
        self.assertEqual(updater.SITE, "https://github.com/vhung-spec/bokin-fx-releases")
        os.environ.pop("BOKIN_UPDATE_API")
        self.assertEqual(updater._sources(), (updater.SITE, updater.RELEASES))
        os.environ["BOKIN_UPDATE_API"] = self.base + "/releases"
        self.assertEqual(updater._sources(), (None, self.base + "/releases"))  # (a stand-in: never the real GitHub)
        with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".github", "workflows",
                               "bokin-fx.yml"), encoding="utf-8") as f:
            self.assertIn("release/" + updater.MANIFEST, f.read())

    def test_manifest_first(self):
        """The newest release's manifest, a plain download: no request to the API (an office shares its limit)."""
        os.environ["BOKIN_UPDATE_SITE"] = self.base + "/site"
        os.environ["BOKIN_UPDATE_API"] = self.base + "/limited"
        sha = hashlib.sha256(INSTALLER).hexdigest()
        FakeGitHub.manifest = {"version": "1.0.40", "name": "BokinFX-Setup-1.0.40.exe", "size": len(INSTALLER),
                               "sha256": sha}
        rel = updater.latest()
        self.assertEqual((rel.version, rel.sha256, rel.size), ("1.0.40", sha, len(INSTALLER)))
        self.assertEqual(rel.url, self.base + "/site/releases/download/fx-v1.0.40/BokinFX-Setup-1.0.40.exe")
        u = updater.AutoUpdate("1.0.39", os.path.join(helpers.ROOT, "updates5"))
        u.check()
        self.assertTrue(u.state["ready"])
        self.assertIsNone(u.state["reason"])
        # a release without a manifest (an older one), or one not to use: the API's list
        os.environ["BOKIN_UPDATE_API"] = self.base + "/releases"
        FakeGitHub.releases = [self.release("fx-v1.0.12")]
        for manifest in (None, dict(FakeGitHub.manifest, name="evil.exe"), dict(FakeGitHub.manifest, sha256="x"),
                         dict(FakeGitHub.manifest, version="dev"), dict(FakeGitHub.manifest, size="9")):
            FakeGitHub.manifest = manifest
            self.assertEqual(updater.latest().version, "1.0.12", manifest)

    def test_why_a_look_failed(self):
        """The page says why: the API's limit for this Internet connection, GitHub out of reach, a download broken."""
        os.environ["BOKIN_UPDATE_API"] = self.base + "/limited"
        u = updater.AutoUpdate("1.0.4", os.path.join(helpers.ROOT, "updates6"))
        u.check()
        self.assertEqual(u.state["reason"], "limit")
        self.assertIn("403", u.state["error"])
        os.environ["BOKIN_UPDATE_API"] = "http://127.0.0.1:9/none"
        u.check()
        self.assertEqual(u.state["reason"], "offline")
        os.environ["BOKIN_UPDATE_SITE"] = "http://127.0.0.1:9/none"  # (out of reach: the API is not asked as well)
        u.check()
        self.assertEqual(u.state["reason"], "offline")
        os.environ.pop("BOKIN_UPDATE_SITE")
        os.environ["BOKIN_UPDATE_API"] = self.base + "/releases"
        FakeGitHub.releases = [self.release("fx-v1.0.5")]
        FakeGitHub.releases[0]["assets"][0]["digest"] = "sha256:" + "0" * 64
        u.check()
        self.assertEqual((u.state["reason"], u.state["ready"], u.state["downloading"]), ("download", False, None))
        FakeGitHub.releases = [self.release("fx-v1.0.5")]
        u.check()
        self.assertTrue(u.state["ready"])
        self.assertIsNone(u.state["error"])
        self.assertIsNone(u.state["reason"])
        self.assertEqual(updater.reason(ValueError("x")), "other")

    def test_a_certificate_only_windows_trusts(self):
        """Where Python's own check refuses GitHub's certificate ("unable to get local issuer certificate": a Windows
        that has never fetched GitHub's root, since Edge and Chrome keep their own lists), the look is made again with
        Windows' own check, which fetches the root, and every look after it is made so at once; another error is not
        tried again. With no such check to be had, the page says it was the certificate."""
        real = urllib.request.urlopen
        windows = object()  # (stands for Windows' own check: the fake GitHub is plain HTTP)
        asked = []

        def urlopen(request, timeout=None, context=None):  # (Python's own check refuses every certificate)
            asked.append(context)
            if context is None and "127.0.0.1:9/" not in request.full_url:
                raise urllib.error.URLError(ssl.SSLCertVerificationError(
                    1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer "
                       "certificate (_ssl.c:1010)"))
            return real(request, timeout=timeout)

        FakeGitHub.releases = [self.release("fx-v1.0.8")]
        with mock.patch.object(updater.urllib.request, "urlopen", urlopen), \
                mock.patch.object(updater, "_system_first", False), \
                mock.patch.object(updater, "system_context", lambda: windows):
            u = updater.AutoUpdate("1.0.7", os.path.join(helpers.ROOT, "updates-cert"))
            u.check()
            self.assertTrue(u.state["ready"])
            self.assertIsNone(u.state["reason"])
            self.assertEqual(asked, [None, windows, windows])  # (the list, then the installer: Windows' check at once)
            # out of reach: not tried again (the same answer)
            asked.clear()
            with mock.patch.object(updater, "_system_first", False):
                os.environ["BOKIN_UPDATE_API"] = "http://127.0.0.1:9/none"
                u.check()
                self.assertEqual((u.state["reason"], asked), ("offline", [None]))
                os.environ["BOKIN_UPDATE_API"] = self.base + "/releases"
        # no Windows' check to be had: the page says it was the certificate
        with mock.patch.object(updater.urllib.request, "urlopen", urlopen), \
                mock.patch.object(updater, "_system_first", False), \
                mock.patch.object(updater, "system_context", lambda: None):
            u = updater.AutoUpdate("1.0.7", os.path.join(helpers.ROOT, "updates-cert2"))
            u.check()
            self.assertEqual(u.state["reason"], "cert")
            self.assertIn("CERTIFICATE_VERIFY_FAILED", u.state["error"])

    @unittest.skipUnless(os.name == "nt", "Windows' own certificate check")
    def test_windows_checks_github(self):
        """On Windows, Windows' own check (truststore, in the app) takes GitHub's certificates: the newest release's
        manifest, github.com handing it on to its download server."""
        context = updater.system_context()
        self.assertIsNotNone(context)
        try:
            with urllib.request.urlopen(f"{updater.SITE}/releases/latest/download/{updater.MANIFEST}", timeout=30,
                                        context=context) as r:
                self.assertIn("version", json.loads(r.read()))
        except urllib.error.URLError as e:
            if updater.reason(e) == "cert":
                raise
            self.skipTest(f"GitHub out of reach: {e}")

    def test_download_goes_on(self):
        """A download that broke off goes on from where it stopped; from the start if the server sends it all."""
        FakeGitHub.releases = [self.release("fx-v1.0.7")]
        rel = updater.latest()
        for ranges in (True, False):
            FakeGitHub.ranges = ranges
            folder = os.path.join(helpers.ROOT, f"updates7{ranges}")
            os.makedirs(folder)
            with open(os.path.join(folder, rel.name + ".download"), "wb") as f:
                f.write(INSTALLER[:5000])
            FakeGitHub.sent = 0
            path = updater.fetch(rel, folder)
            self.assertEqual(open(path, "rb").read(), INSTALLER)
            self.assertEqual(FakeGitHub.sent, len(INSTALLER) - (5000 if ranges else 0))
            self.assertEqual(os.listdir(folder), [rel.name])
        folder = os.path.join(helpers.ROOT, "updates8")  # (a broken piece: thrown away, the next look starts again)
        os.makedirs(folder)
        FakeGitHub.ranges = True
        with open(os.path.join(folder, rel.name + ".download"), "wb") as f:
            f.write(b"x" * 5000)
        with self.assertRaises(ValueError):
            updater.fetch(rel, folder)
        self.assertEqual(os.listdir(folder), [])
        self.assertEqual(open(updater.fetch(rel, folder), "rb").read(), INSTALLER)

    def test_release_notes_and_manifest(self):
        """The workflow's notes and manifest (packaging/release_notes.py): what the updater reads."""
        import subprocess
        import sys
        folder = os.path.join(helpers.ROOT, "release")
        os.makedirs(folder)
        installer = os.path.join(folder, "BokinFX-Setup-1.0.70.exe")
        with open(installer, "wb") as f:
            f.write(INSTALLER)
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run([sys.executable, os.path.join(root, "packaging", "release_notes.py"), installer, "1.0.70",
                        folder], check=True)
        sha = hashlib.sha256(INSTALLER).hexdigest()
        with open(os.path.join(folder, "notes.md"), encoding="utf-8") as f:
            self.assertIn(f"SHA-256: {sha}", f.read())
        with open(os.path.join(folder, updater.MANIFEST), encoding="utf-8") as f:
            FakeGitHub.manifest = json.load(f)
        os.environ["BOKIN_UPDATE_SITE"] = self.base + "/site"
        rel = updater.latest()
        self.assertEqual((rel.version, rel.name, rel.size, rel.sha256), ("1.0.70", "BokinFX-Setup-1.0.70.exe",
                                                                       len(INSTALLER), sha))

    def test_versions(self):
        self.assertEqual(updater.parse_version("1.0.12"), (1, 0, 12))
        self.assertIsNone(updater.parse_version("dev"))
        self.assertIsNone(updater.parse_version("1.0.x"))
        self.assertGreater(updater.parse_version("1.0.10"), updater.parse_version("1.0.9"))

    def test_latest_and_fetch(self):
        FakeGitHub.releases = [self.release("fx-v1.0.9"), self.release("fx-v1.0.12", digest=False, body="SHA-256: <sha>"),
                               self.release("fx-v1.0.30", draft=True), self.release("loop-v9.0.0"),
                               self.release("fx-v1.0.20", digest=False),  # nothing to check it with: skipped
                               self.release("fx-v1.0.25", name="evil.exe")]
        rel = updater.latest()
        self.assertEqual(rel.version, "1.0.12")
        folder = os.path.join(helpers.ROOT, "updates")
        path = updater.fetch(rel, folder)
        self.assertEqual(open(path, "rb").read(), INSTALLER)
        bad = rel._replace(sha256="0" * 64, name="BokinFX-Setup-1.0.13.exe")
        with self.assertRaises(ValueError):
            updater.fetch(bad, folder)
        self.assertEqual(os.listdir(folder), [rel.name])  # the bad one is gone; the good one stays until replaced

    def test_ready_installer_kept_until_the_next_is_in(self):
        folder = os.path.join(helpers.ROOT, "updates4")
        FakeGitHub.releases = [self.release("fx-v1.0.8")]
        first = updater.fetch(updater.latest(), folder)
        broken = updater.latest()._replace(version="1.0.9", name="BokinFX-Setup-1.0.9.exe", sha256="f" * 64)
        with self.assertRaises(ValueError):
            updater.fetch(broken, folder)
        self.assertTrue(os.path.exists(first))  # still there to install

    def test_auto_update_state(self):
        FakeGitHub.releases = [self.release("fx-v1.0.5")]
        u = updater.AutoUpdate("1.0.4", os.path.join(helpers.ROOT, "updates2"))
        u.check()
        self.assertTrue(u.state["ready"])
        self.assertEqual(u.state["available"], "1.0.5")
        same = updater.AutoUpdate("1.0.5", os.path.join(helpers.ROOT, "updates3"))
        same.check()
        self.assertFalse(same.state["ready"])
        self.assertIsNone(same.state["error"])
        self.assertFalse(same.run_installer(restart=True))

    def test_the_page_can_ask_for_a_check(self):
        """A window opened, or the version clicked: the app looks again at once, not within a minute twice."""
        FakeGitHub.releases = [self.release("fx-v1.0.5")]
        u = updater.AutoUpdate("1.0.5", os.path.join(helpers.ROOT, "updates4"), every=3600).start()
        for _ in range(100):
            if u.state["checked"]:
                break
            time.sleep(0.1)
        first = u.state["checked"]
        self.assertIsNotNone(first)
        self.assertFalse(u.check_soon())  # a moment after the first look: nothing
        self.assertFalse(u.wake.is_set())
        u.state["checked"] = first - 120  # as if two minutes had passed
        FakeGitHub.releases = [self.release("fx-v1.0.6")]
        self.assertTrue(u.check_soon())
        for _ in range(100):
            if u.state["ready"]:
                break
            time.sleep(0.1)
        self.assertEqual(u.state["available"], "1.0.6")
        self.assertIsNone(u.state["downloading"])
        self.assertGreater(u.state["checked"], first - 120)  # set once the installer is in, not before

    def test_every_five_minutes(self):
        self.assertEqual(updater.CHECK_EVERY, 5 * 60)


if __name__ == "__main__":
    unittest.main()
