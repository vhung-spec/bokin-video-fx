import hashlib
import json
import os
import time
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tests import helpers
from fx import updater

INSTALLER = b"MZ fake installer " * 1000


class FakeGitHub(BaseHTTPRequestHandler):
    releases = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/releases"):
            body = json.dumps(self.releases).encode()
        elif self.path.startswith("/download/"):
            body = INSTALLER
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


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
