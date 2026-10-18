import datetime
import json
import os
import socket
import tempfile
import unittest
import urllib.error
import urllib.request

from tests import helpers
from tests.test_server import request
from bokin_license import store as license_store
from fx import licensing, server


class Licence(unittest.TestCase):
    """Before a key for this computer is activated the app shows only its activation screen: the server refuses
    the work (projects, media, export) and still answers the installer and the updater."""

    @classmethod
    def setUpClass(cls):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cls.srv = server.start(server.App(desktop=False), port)
        cls.base = f"http://127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def setUp(self):
        self.empty = tempfile.TemporaryDirectory()
        license_store.default_folder = lambda: self.empty.name  # a computer with no key
        licensing.status(fresh=True)

    def tearDown(self):
        license_store.default_folder = lambda: helpers.LICENSE_FOLDER
        licensing.status(fresh=True)
        self.empty.cleanup()

    def post(self, path, body):
        return request(self.base + path, json.dumps(body).encode(), {"X-Bokin": "1", "Content-Type": "application/json"},
                       "POST")

    def test_refused_until_activated(self):
        for path in ("/api/schema", "/api/project?id=0123456789ab", "/api/keys", "/media/out?name=a.mp4"):
            code, body, _ = request(self.base + path)
            self.assertEqual((code, json.loads(body).get("license")), (403, True), path)
        code, body, _ = self.post("/api/export", {"id": "0123456789ab"})
        self.assertEqual((code, json.loads(body).get("license")), (403, True))
        # the page, its activation screen, and what the installer and the updater ask
        self.assertEqual(request(self.base + "/")[0], 200)
        self.assertEqual(json.loads(request(self.base + "/api/ping")[1])["app"], "bokin-fx")
        self.assertIn("update", json.loads(request(self.base + "/api/state")[1]))
        self.assertEqual(self.post("/api/update/check", {})[0], 200)
        view = json.loads(request(self.base + "/api/license")[1])
        self.assertFalse(view["licensed"])
        self.assertRegex(view["machine"], r"^[A-Z2-7]{4}(-[A-Z2-7]{4}){3}$")

    def test_activation(self):
        machine = licensing.status()["machine"]
        cases = [("", "Hãy dán key"), ("xin chào", "định dạng"),
                 (helpers.test_key(["loop"]), "không dành cho Bokin FX"),
                 (helpers.test_key(machine="AAAA-BBBB-CCCC-DDDD"), "máy khác"),
                 (helpers.test_key(expires=datetime.date(2021, 1, 1)), "hết hạn"),
                 (helpers.test_key(secret=bytes(32)), "không hợp lệ")]  # signed by someone else
        for key, says in cases:
            code, body, _ = self.post("/api/license", {"key": key})
            self.assertEqual(code, 400, key)
            self.assertIn(says, json.loads(body)["error"], key)
        self.assertEqual(license_store.stored_keys(self.empty.name), [])  # a refused key is never kept
        self.assertEqual(request(self.base + "/api/schema")[0], 403)
        until = datetime.date.today() + datetime.timedelta(days=30)
        bundle = helpers.test_key(["fx", "loop"], machine, until)
        code, body, _ = self.post("/api/license", {"key": " " + bundle[:30] + "\n" + bundle[30:] + " "})  # (as pasted)
        self.assertEqual(code, 200, body)
        view = json.loads(body)
        self.assertEqual((view["licensed"], view["expires"], view["products"]),
                         (True, until.isoformat(), ["Bokin FX", "Bokin Loop"]))
        self.assertEqual(request(self.base + "/api/schema")[0], 200)  # the app works now
        self.assertEqual(json.loads(request(self.base + "/api/license")[1])["licensed"], True)
        self.assertTrue(os.path.exists(os.path.join(self.empty.name, "keys.txt")))

    def test_a_key_that_runs_out_meanwhile(self):
        """A key with an end date checked again before an export, and at least every minute."""
        self.assertEqual(self.post("/api/license", {"key": helpers.test_key()})[0], 200)
        self.assertEqual(request(self.base + "/api/schema")[0], 200)
        license_store.default_folder = lambda: os.path.join(self.empty.name, "gone")  # (as if it ran out)
        self.assertEqual(request(self.base + "/api/schema")[0], 200)  # (read again within a minute)
        self.assertEqual(self.post("/api/export", {"id": "0123456789ab"})[0], 403)  # an export asks now
        self.assertEqual(request(self.base + "/api/schema")[0], 403)


if __name__ == "__main__":
    unittest.main()
