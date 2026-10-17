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


def png_bytes():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("L", (160, 90), 255).save(buf, "PNG")
    return buf.getvalue()


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
        code, _, headers = request(self.base + "/")
        self.assertEqual(code, 200)
        self.assertEqual(headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(headers.get("Content-Security-Policy"), "frame-ancestors 'none'")
        self.assertEqual(request(self.base + "/api/state", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(request(self.base + "/api/export", b"{}", {"Content-Type": "application/json"}, "POST")[0], 403)
        self.assertEqual(self.post("/api/open", {})[0], 400)  # not on a server

    def test_keys_and_guide(self):
        code, body, _ = request(self.base + "/api/keys")
        self.assertEqual(code, 200)
        v = json.loads(body)
        self.assertEqual(v["keys"]["undo"], ["Ctrl+KeyZ"])
        self.assertIn("names", v)
        self.assertEqual(request(self.base + "/api/keys", json.dumps({"keys": {}}).encode(),
                                 {"Content-Type": "application/json"}, "POST")[0], 403)  # (no X-Bokin header)
        code, body, _ = self.post("/api/keys", {"keys": {"razorTool": ["KeyD", "Ctrl+KeyW"], "x": ["KeyQ"]}})
        self.assertEqual(code, 200)
        v = json.loads(body)
        self.assertEqual(v["custom"], {"razorTool": ["KeyD"]})
        self.assertEqual(json.loads(request(self.base + "/api/keys")[1])["keys"]["razorTool"], ["KeyD"])
        self.assertEqual(json.loads(self.post("/api/keys", {"keys": {}})[1])["custom"], {})
        self.assertEqual(self.post("/api/keys", raw=b"[1]")[0], 400)
        code, body, headers = request(self.base + "/api/guide")
        self.assertEqual(code, 200)
        self.assertTrue(headers["Content-Type"].startswith("text/markdown"))
        self.assertIn("Phím tắt", body.decode("utf-8"))

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
        self.assertEqual(request(self.base + f"/media/fx?id={pid}&name=..%2Finput.mp4")[0], 404)
        self.assertEqual(request(self.base + "/media/lib?id=..%2F..&what=thumb")[0], 400)

        # "Lưu thành bản khác": a new project, the old one as it was
        self.assertEqual(self.post("/api/project/copy", {"id": "../x", "name": "a"})[0], 400)
        code, body, _ = self.post("/api/project/copy", {"id": pid, "name": "Quán cà phê - bản 2"})
        self.assertEqual(code, 200, body)
        copy = json.loads(body)
        self.assertEqual(copy["name"], "Quán cà phê - bản 2")
        self.assertEqual([lay["id"] for lay in json.loads(request(self.base + f"/api/project?id={copy['id']}")[1])["layers"]],
                         ["snow1"])
        self.assertEqual(self.post("/api/delete", {"id": copy["id"]})[0], 200)

        # the light copy of the video, made in the background after the upload
        self.wait_for(lambda: json.loads(request(self.base + f"/api/project?id={pid}")[1])["proxy"])
        code, body, headers = request(self.base + f"/media/base?id={pid}", headers={"Range": "bytes=0-99"})
        self.assertEqual((code, len(body)), (206, 100))
        self.assertTrue(headers["Content-Range"].startswith("bytes 0-99/"))

        # an effect in the library, used by a clip, with its light copy
        with open(clip, "rb") as f:
            code, body, _ = self.post("/api/upload?kind=library&name=snow.mp4", raw=f.read(),
                                      headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(code, 200, body)
        item = json.loads(body)
        self.assertEqual(request(self.base + f"/media/lib?id={item['id']}&what=thumb")[0], 200)
        self.wait_for(lambda: next(i for i in json.loads(request(self.base + "/api/state")[1])["library"]
                                   if i["id"] == item["id"])["proxy"])
        self.assertEqual(request(self.base + f"/media/lib?id={item['id']}&what=proxy")[0], 200)
        project = json.loads(self.post("/api/save", {"id": pid, "layers": [
            {"id": "snow1", "kind": "snow", "params": {"amount": 50}, "start": 0, "end": 200},
            {"id": "file1", "kind": "file", "lib": item["id"], "start": 100, "end": 400}]})[1])
        self.assertEqual([lay["kind"] for lay in project["layers"]], ["snow", "file"])
        self.assertEqual(self.post("/api/library/delete", {"id": item["id"]})[0], 400)  # in use

        # a built-in effect's preview clip
        answer = json.loads(self.post("/api/fxclip", {"id": pid, "layer": "snow1"})[1])
        self.assertFalse(answer["ready"])
        self.wait_for(lambda: json.loads(self.post("/api/fxclip", {"id": pid, "layer": "snow1"})[1])["ready"])
        self.assertEqual(request(self.base + f"/media/fx?id={pid}&name={answer['name']}")[0], 200)
        self.assertEqual(self.post("/api/fxclip", {"id": pid, "layer": "file1"})[0], 400)

        # a cut clip's area is copied to its second half
        self.post("/api/save", {"id": pid, "layers": project["layers"] + [dict(project["layers"][0], id="snow2")]})
        self.assertEqual(self.post(f"/api/mask?id={pid}&name=snow1", raw=png_bytes())[0], 200)
        code, body, _ = self.post("/api/copymask", {"id": pid, "src": "snow1", "dst": "snow2"})
        self.assertEqual(code, 200, body)
        self.assertIn("snow2", json.loads(body)["masks"])
        self.assertEqual(self.post("/api/copymask", {"id": pid, "src": "snow1", "dst": "../x"})[0], 400)

        code, body, _ = self.post("/api/export", {"id": pid})
        job = json.loads(body)["id"]
        mine = self.wait_for(lambda: next(j for j in json.loads(request(self.base + "/api/state")[1])["jobs"]
                                          if j["id"] == job and j["state"] not in ("queued", "running")), 600)
        self.assertEqual(mine["state"], "done", mine)
        self.assertTrue(mine["output"].startswith("Quán cà phê-fx-30s"), mine["output"])
        long_name = urllib.parse.quote("Quán " + "x" * 250 + ".mp4")  # a long name keeps its extension
        with open(clip, "rb") as f:
            code, body, _ = self.post(f"/api/upload?kind=video&name={long_name}", raw=f.read(),
                                      headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(code, 200, body)
        self.assertEqual(self.post("/api/delete", {"id": json.loads(body)["id"]})[0], 200)
        self.assertTrue(os.path.exists(os.path.join(os.environ["BOKIN_FX_OUT"], mine["output"])))
        self.assertEqual(self.post("/api/delete", {"id": pid})[0], 200)
        self.assertEqual(request(self.base + f"/api/project?id={pid}")[0], 400)

    def wait_for(self, check, tries=240):
        for _ in range(tries):
            try:
                got = check()
            except StopIteration:
                got = None
            if got:
                return got
            time.sleep(0.25)
        self.fail("timed out")

    def test_old_clips_get_every_setting(self):
        """A rain or snow clip saved before their settings grew gets every setting, as drawn, when opened."""
        from fx import effects, projects
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "server-old.mp4"), 160, 90, 12)
        project = projects.create(clip, "old.mp4")
        tid = project["tracks"][0]["id"]
        project = projects.apply(project, {"layers": [
            {"id": "rainol", "kind": "rain", "track": tid, "params": {"amount": 60}},
            {"id": "snowol", "kind": "snow", "track": tid, "params": {"blur": 70}}]})
        path = os.path.join(projects.folder(project["id"]), "project.json")
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        saved["layers"][0]["params"] = {"amount": 60, "speed": 60, "length": 50, "size": 35, "wind": 10, "depth": 50,
                                        "opacity": 45, "color": "#d4dce6"}  # as the previous version saved it
        with open(path, "w", encoding="utf-8") as f:
            json.dump(saved, f)
        code, body, _ = request(self.base + f"/api/project?id={project['id']}")
        self.assertEqual(code, 200)
        layers = {lay["id"]: lay for lay in json.loads(body)["layers"]}
        self.assertEqual(layers["rainol"]["params"], effects.clean_params("rain", {**saved["layers"][0]["params"]}))
        self.assertEqual(set(layers["snowol"]["params"]), {d["key"] for d in effects.SCHEMA["snow"]["params"]})
        self.assertEqual(layers["rainol"]["params"]["amount"], 60)

    def test_refused_request_body_is_not_run(self):
        """A cross-site POST is refused, and its body is never read as a second request."""
        import socket as sk
        port = int(self.base.rsplit(":", 1)[1])
        inner = (f"POST /api/quit HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nX-Bokin: 1\r\n"
                 "Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}").encode()
        outer = (f"POST /api/export HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: text/plain\r\n"
                 f"Content-Length: {len(inner)}\r\n\r\n").encode() + inner
        with sk.create_connection(("127.0.0.1", port), timeout=5) as c:
            c.sendall(outer)
            got = b""
            while True:
                chunk = c.recv(65536)
                if not chunk:
                    break
                got += chunk
        self.assertIn(b"403", got.split(b"\r\n", 1)[0])
        self.assertEqual(got.count(b"HTTP/1.1"), 1)
        self.assertFalse(self.app.quit.is_set())

    def test_wrong_type_refused_before_upload(self):
        code, body, headers = self.post("/api/upload?kind=video&name=" + urllib.parse.quote("a" * 300 + ".exe"), raw=b"x" * 10,
                                        headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(code, 400)
        self.assertEqual(headers.get("Connection"), "close")

    def test_ping_names_its_owner(self):
        from fx import paths
        answer = json.loads(request(self.base + "/api/ping")[1])
        self.assertEqual(answer["owner"], paths.owner())

    def test_quit_refused_while_busy(self):
        self.assertFalse(self.app.jobs.busy())
        code, body, _ = self.post("/api/quit", {})
        self.assertEqual(json.loads(body), {"ok": True})
        self.assertTrue(self.app.quit.is_set())
        self.app.quit.clear()


if __name__ == "__main__":
    unittest.main()
