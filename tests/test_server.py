import json
import os
import socket
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request

from tests import helpers
from fx import server


def request(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        cls.app = server.App(desktop=False)
        cls.srv = server.start(cls.app, port)
        cls.base = f"http://127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def post(self, path, body=None, raw=None, headers=None):
        h = {"X-Bokin": "1", "Content-Type": "application/json"}
        h.update(headers or {})
        data = raw if raw is not None else json.dumps(body or {}).encode()
        return request(self.base + path, data, h, "POST")

    def test_only_its_own_page(self):
        self.assertEqual(request(self.base + "/")[0], 200)
        self.assertEqual(request(self.base + "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(request(self.base + "/api/preview", b"{}", {"Content-Type": "application/json"}, "POST")[0], 403)
        self.assertEqual(self.post("/api/open", {})[0], 400)  # not on a server

    def test_project_flow(self):
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "server.mp4"), 160, 90, 12)
        with open(clip, "rb") as f:
            name = urllib.parse.quote("Quán cà phê.mp4")
            code, body, _ = self.post(f"/api/upload?kind=video&name={name}", raw=f.read(),
                                      headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(code, 200, body)
        project = json.loads(body)
        pid = project["id"]
        self.assertEqual(project["name"], "Quán cà phê")
        self.assertEqual((project["mask_w"], project["mask_h"]), (160, 90))
        code, body, _ = self.post("/api/save", {"id": pid, "layers": [{"id": "../../x", "kind": "snow"}]})
        self.assertEqual(code, 400)
        code, body, _ = self.post("/api/save", {"id": pid, "layers": [{"id": "snow1", "kind": "snow", "params": {"amount": 50}}]})
        self.assertEqual(code, 200, body)
        self.assertEqual(self.post(f"/api/mask?id={pid}&name=..%2F..%2Fx", raw=b"x")[0], 400)
        self.assertEqual(self.post(f"/api/mask?id={pid}&name=snow1", raw=b"not a png")[0], 400)
        self.assertEqual(request(self.base + f"/api/mask?id={pid}&name=..%2Fproject")[0], 404)
        code, body, headers = request(self.base + f"/api/look?id={pid}&t=3&w=160")
        self.assertEqual((code, headers.get("Content-Type")), (200, "image/jpeg"))
        self.assertEqual(request(self.base + f"/api/frame?id={pid}&t=999")[0], 200)  # clamped to the last frame
        self.assertEqual(request(self.base + "/media/out?name=..%2F..%2Fsecret.mp4")[0], 404)
        self.assertEqual(request(self.base + f"/media/preview?id={pid}&name=..%2Finput.mp4")[0], 404)

        code, body, _ = self.post("/api/preview", {"id": pid})
        self.assertEqual(code, 200, body)
        job = json.loads(body)["id"]
        for _ in range(240):
            state = json.loads(request(self.base + "/api/state")[1])
            mine = next(j for j in state["jobs"] if j["id"] == job)
            if mine["state"] not in ("queued", "running"):
                break
            time.sleep(0.25)
        self.assertEqual(mine["state"], "done", mine)
        code, body, headers = request(self.base + f"/media/preview?id={pid}&name={mine['output']}",
                                      headers={"Range": "bytes=0-99"})
        self.assertEqual((code, len(body)), (206, 100))
        self.assertTrue(headers["Content-Range"].startswith("bytes 0-99/"))

        code, body, _ = self.post("/api/export", {"id": pid, "seconds": 2})
        job = json.loads(body)["id"]
        for _ in range(480):
            mine = next(j for j in json.loads(request(self.base + "/api/state")[1])["jobs"] if j["id"] == job)
            if mine["state"] not in ("queued", "running"):
                break
            time.sleep(0.25)
        self.assertEqual(mine["state"], "done", mine)
        self.assertTrue(mine["output"].startswith("Quán cà phê-fx"))
        self.assertTrue(os.path.exists(os.path.join(os.environ["BOKIN_FX_OUT"], mine["output"])))
        self.assertEqual(self.post("/api/delete", {"id": pid})[0], 200)
        self.assertEqual(request(self.base + f"/api/project?id={pid}")[0], 400)

    def test_quit_refused_while_busy(self):
        self.assertFalse(self.app.jobs.busy())
        code, body, _ = self.post("/api/quit", {})
        self.assertEqual(json.loads(body), {"ok": True})
        self.assertTrue(self.app.quit.is_set())
        self.app.quit.clear()


if __name__ == "__main__":
    unittest.main()
