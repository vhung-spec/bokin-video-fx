import json
import os
import socket
import unittest

import numpy as np
from PIL import Image

from tests import helpers
from tests.test_server import request
from fx import colourmatch, library, overlay, projects, server, video

W, H, P = 320, 180, 24


def scene_video(path):
    """A warm evening: dark orange on the left, bright warm on the right."""
    xx = np.mgrid[0:H, 0:W][1].astype(np.float32) / W
    left = np.array([70, 42, 22], np.float32)
    right = np.array([235, 190, 140], np.float32)
    f = np.where((xx < 0.5)[..., None], left, right) + 6 * np.sin(np.arange(W) / 3)[None, :, None]
    enc = video.Encoder(path, W, H, "24", crf=18, preset="veryfast")
    for _ in range(P):
        enc.write(np.clip(f, 0, 255).astype(np.uint8))
    enc.close()
    return path


def neutral_png(path):
    """A light grey, slightly cool bird-sized blob on a transparent ground."""
    yy, xx = np.mgrid[0:120, 0:120]
    a = (((xx - 60) / 45.0) ** 2 + ((yy - 60) / 30.0) ** 2 < 1) * 255
    g = 150 + 60 * (xx / 120.0)
    rgba = np.dstack([g * 0.95, g, g * 1.05, a]).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgba, "RGBA").save(path)
    return path


def sparks_on_black(path):
    """White sparks on black (laid with Screen)."""
    rng = np.random.default_rng(3)
    enc = video.Encoder(path, 160, 90, "24", crf=18, preset="veryfast")
    for _ in range(12):
        f = np.zeros((90, 160, 3), np.uint8)
        for x, y in rng.integers(0, [156, 86], (40, 2)):
            f[y:y + 4, x:x + 4] = 250
        enc.write(f)
    enc.close()
    return path


def cast(rgb, weight):
    m = colourmatch._stats(rgb, weight)[0]
    return m / float(m @ colourmatch.LUMA)


class Match(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = scene_video(os.path.join(helpers.ROOT, "evening.mp4"))
        cls.project = projects.create(src, "evening.mp4")
        cls.folder = projects.folder(cls.project["id"])
        cls.bird = library.add(neutral_png(os.path.join(helpers.ROOT, "bird.png")), "bird.png")
        cls.sparks = library.add(sparks_on_black(os.path.join(helpers.ROOT, "sparks.mp4")), "sparks.mp4")

    def layer(self, item, **params):
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "filemm", "kind": "file", "lib": item["id"], "start": 0, "end": P, "params": params}]})
        return project, project["layers"][0]

    def test_a_clip_takes_on_the_light_and_colour_where_it_lies(self):
        project, layer = self.layer(self.bird, blend="normal", fit="contain", scale=30, x=-25)
        p = colourmatch.match(self.folder, project, layer)
        self.assertEqual(set(p), {"bright", "contrast", "sat", "hue", "temp", "tint_amt", "tint"})
        self.assertLess(p["bright"], -20)  # a light grey bird in the dark left half: darker
        self.assertTrue(p["temp"] > 0 or p["tint_amt"] > 0, p)  # and warmer
        path, info = library.source_path(self.bird), self.bird["info"]
        rgb, weight = colourmatch.clip(path, info, "normal")
        scene_rgb, scene_weight = colourmatch.scene(self.folder, project, layer, layer["params"], info)
        before, after, want = cast(rgb, weight), cast(colourmatch._apply(rgb, p), weight), cast(scene_rgb, scene_weight)
        self.assertLess(float(np.abs(after - want).sum()), 0.7 * float(np.abs(before - want).sum()))
        # its settings are the sliders' own: whole numbers in their ranges, a colour the page's picker takes
        ranges = {d["key"]: d for d in overlay.SCHEMA["params"]}
        for k in ("bright", "contrast", "sat", "hue", "temp", "tint_amt"):
            self.assertIsInstance(p[k], int)
            self.assertTrue(ranges[k]["min"] <= p[k] <= ranges[k]["max"], k)
        self.assertRegex(p["tint"], r"^#[0-9a-f]{6}$")
        # moved to the bright right half: lighter than on the left
        project, layer = self.layer(self.bird, blend="normal", fit="contain", scale=30, x=25)
        self.assertGreater(colourmatch.match(self.folder, project, layer)["bright"], p["bright"] + 30)

    def test_light_laid_over_only_gets_darker(self):
        project, layer = self.layer(self.sparks, blend="screen", scale=60, x=-20)
        p = colourmatch.match(self.folder, project, layer)
        self.assertLess(p["bright"], 0)  # white sparks over a dark evening: dimmer, never brighter
        self.assertEqual(p["contrast"], 0)
        project, layer = self.layer(self.sparks, blend="screen", scale=60, x=20)
        self.assertLessEqual(colourmatch.match(self.folder, project, layer)["bright"], 0)

    def test_a_clip_like_its_scene_stays_as_it_is(self):
        rng = np.random.default_rng(0)
        px = np.clip(rng.normal([0.5, 0.4, 0.3], 0.1, (30000, 3)), 0, 1).astype(np.float32)
        w = np.ones(len(px), np.float32)
        p = colourmatch.settings(px[:10000], w[:10000], px, w, "normal")
        self.assertLessEqual(max(abs(p[k]) for k in ("bright", "contrast", "sat", "temp", "tint_amt")), 3, p)

    def test_endpoint(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        srv = server.start(server.App(desktop=False), port)
        try:
            project, layer = self.layer(self.bird, blend="normal", fit="contain", scale=30, x=-25)
            post = lambda body: request(f"http://127.0.0.1:{port}/api/match", json.dumps(body).encode(),
                                        {"X-Bokin": "1", "Content-Type": "application/json"}, "POST")
            code, body, _ = post({"id": project["id"], "layer": layer["id"]})
            self.assertEqual(code, 200, body)
            self.assertLess(json.loads(body)["params"]["bright"], 0)
            code, body, _ = post({"id": project["id"], "layer": "nothere"})
            self.assertEqual((code, json.loads(body)["error"]), (400, "Không tìm thấy clip."))
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
