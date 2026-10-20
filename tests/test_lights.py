"""Twinkling string lights (fx/lights.py): a placed effect drawn as light."""
import json
import os
import socket
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from tests import helpers
from tests.test_server import request
from fx import bulbs, effects, engine, lights, projects, server, video

LAYOUTS = [k for k in lights.LAYOUTS if k != "scene"]  # (the lights in the video: their own tests)
MODES = [o["value"] for o in effects.SCHEMA["lights"]["params"][1]["options"]]


def make(params=None, w=640, h=360, frames=144, fps=24, seed=3):
    return effects.make("lights", params or {}, seed, frames, fps, w, h)


def picture(fx, t, w, h):
    pic = np.zeros((h, w, 3), np.float32)
    canvas = effects.Canvas(w, h)
    fx.draw(t, canvas, None)
    canvas.flush(pic, fx.blend)
    return pic


class Lights(unittest.TestCase):
    def test_every_way_loops(self):
        """Every layout and every way of twinkling: frame P is frame 0 again."""
        for layout in LAYOUTS:
            for mode in MODES:
                for frames, fps in ((144, 24), (150, 25)):
                    fx = make({"layout": layout, "mode": mode, "speed": 100}, 320, 180, frames, fps)
                    diff = np.abs(picture(fx, 0, 320, 180) - picture(fx, frames, 320, 180)).max()
                    self.assertLess(diff, 1 / 512, (layout, mode, frames))

    def test_bulbs_in_the_box(self):
        """The strands lie inside the box (a nudge off the line at most), as many bulbs as asked for."""
        for layout in LAYOUTS:
            few, many = (make({"layout": layout, "amount": a}) for a in (0, 100))
            self.assertLess(len(few.x), len(many.x), layout)
            self.assertGreater(len(many.x), 300, layout)
            for fx in (few, many):
                self.assertGreater(fx.x.min(), -0.03, layout)
                self.assertLess(fx.x.max(), 1.03, layout)
                self.assertGreater(fx.y.min(), -0.03, layout)
                self.assertLess(fx.y.max(), 1.08, layout)

    def test_the_tree(self):
        """Round a cone: narrow at the top, wide at the bottom; the bulbs at the back mostly hidden by the tree."""
        fx = make({"jitter": 0, "back": 0})
        top, low = fx.y < 0.3, fx.y > 0.8
        self.assertLess(np.ptp(fx.x[top]), 0.5 * np.ptp(fx.x[low]))
        self.assertAlmostEqual(float(np.ptp(fx.x[low])), 0.62, delta=0.06)  # (the width setting)
        self.assertGreater((fx.vis < 0.01).mean(), 0.3)  # (at the back)
        self.assertGreater((fx.vis > 0.99).mean(), 0.3)
        seen = make({"jitter": 0, "back": 100})
        self.assertLess((seen.vis < 0.01).mean(), 0.01)

    def test_chase_runs_along_the_strand(self):
        """Chasing: a little later each bulb is as bright as the one before it was."""
        fx = make({"mode": "chase", "depth": 100})
        step = 1 / (fx.m_chase * lights.CHASE)
        a, _ = fx.levels(0.3)
        b, _ = fx.levels(0.3 + step)
        one = np.flatnonzero(fx.strand == 0)
        self.assertTrue(np.allclose(b[one[1:]], a[one[:-1]], atol=1e-9))
        self.assertGreater(np.ptp(a), 0.5)

    def test_flash_and_steady(self):
        flash = make({"mode": "flash", "depth": 100})
        level, _ = flash.levels(0.25 / flash.m_flash)  # (a quarter of a flash: one group on, the other off)
        self.assertGreater(level[flash.group == 0].min(), 0.99)
        self.assertLess(level[flash.group == 1].max(), 0.01)
        steady = make({"mode": "steady"})
        levels = np.array([steady.levels(t / 144)[0] for t in range(144)])
        self.assertGreater(levels.min(), 0.9)

    def test_colours_in_turn(self):
        fx = make({"palette": "multi", "jitter": 0})
        one = np.flatnonzero(fx.strand == 0)[:10]
        colours = [tuple(np.round(c, 3)) for c in fx.c_halo[one]]
        self.assertEqual(len(set(colours[:5])), 5)
        self.assertEqual(colours[:5], colours[5:10])
        own = make({"palette": "one", "color": "#00ff00"})
        self.assertTrue(np.allclose(own.c_halo, [0, 1, 0]))

    def test_light_only(self):
        """Drawn as light (Screen): it only brightens; nothing outside the box but a halo's glow."""
        fx = make({"scale": 50, "x": 20, "glow": 0, "rays": 0})
        pic = picture(fx, 10, 640, 360)
        self.assertGreater(pic.max(), 0.5)
        x0, y0, w, h = fx.box()
        lit = np.argwhere(pic.max(axis=2) > 0.02)
        self.assertGreaterEqual(lit[:, 1].min(), x0 - 4)
        self.assertLessEqual(lit[:, 1].max(), x0 + w + 4)

    def test_preview_clip_is_the_box(self):
        """Its preview clip is the box alone: the same bulbs as in the video, wherever the box is."""
        fx = make({"scale": 50, "x": 10}, 1000, 600)
        x0, y0, w, h = fx.box()
        self.assertEqual((x0, y0, w, h), (450.0, 150.0, 300.0, 300.0))
        frame = picture(fx, 7, 1000, 600)[150:450, 450:750]
        clip = effects.make("lights", engine.fx_clip_params("lights", {"scale": 50, "x": 10}), 3, 144, 24, 300, 300)
        alone = picture(clip, 7, 300, 300)
        self.assertLess(float(np.abs(frame - alone).max()), 1e-5)
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 48, "seconds": 2.0}
        self.assertEqual(engine.fx_clip_frame(info, "lights"), (640, 640))
        project = {"video": info, "length": 30}
        a = engine.fx_clip_name(project, {"kind": "lights", "params": {"scale": 50, "x": 10}})
        self.assertEqual(a, engine.fx_clip_name(project, {"kind": "lights", "params": {"scale": 20, "y": -30}}))
        self.assertTrue(a.endswith(".mp4"))  # (light on black, laid with Screen)


def tree_with_lights(w=640, h=360, n=60, seed=1):
    """A dark tree with n small lights on it (a clipped white middle, a glow of its colour): the picture, the lights'
    places (pixels) and colours."""
    im = Image.new("RGB", (w, h), (20, 18, 26))
    ImageDraw.Draw(im).polygon([(w / 2, 20), (w / 2 - 150, 330), (w / 2 + 150, 330)], fill=(24, 60, 36))
    g = np.random.default_rng(seed)
    glow = Image.new("RGB", (w, h), 0)
    pts, cols = [], []
    while len(pts) < n:
        y = 60 + 250 * g.random()
        x = w / 2 + (g.random() - 0.5) * 2 * 130 * (y - 20) / 310
        if any((x - a) ** 2 + (y - b) ** 2 < 100 for a, b in pts):
            continue
        col = [(255, 70, 50), (70, 255, 110), (90, 130, 255)][len(pts) % 3]
        ImageDraw.Draw(glow).ellipse([x - 4, y - 4, x + 4, y + 4], fill=col)
        pts.append((x, y))
        cols.append(col)
    glow = glow.filter(ImageFilter.GaussianBlur(2))
    arr = np.clip(np.asarray(im).astype(int) + np.asarray(glow).astype(int), 0, 255).astype(np.uint8)
    im = Image.fromarray(arr)
    for x, y in pts:
        ImageDraw.Draw(im).ellipse([x - 1.3, y - 1.3, x + 1.3, y + 1.3], fill=(255, 252, 245))
    return np.asarray(im), np.array(pts), np.array(cols)


class OnTheLightsInTheVideo(unittest.TestCase):
    """String lights on the lights a video already has: found under the box (fx/bulbs.py), they flare and twinkle
    where they are, in their own colours; bulbs or only their light."""

    def test_found_once_each_where_they_are(self):
        rgb, pts, cols = tree_with_lights()
        rect = (100, 0, 540, 360)
        found = np.array(bulbs.spots(rgb.astype(np.float32) / 255, rect))
        x = rect[0] + found[:, 0] * (rect[2] - rect[0])
        y = rect[1] + found[:, 1] * (rect[3] - rect[1])
        d = np.hypot(x[:, None] - pts[None, :, 0], y[:, None] - pts[None, :, 1])
        self.assertGreater((d.min(axis=1) < 2.5).mean(), 0.95)  # (each on a light)
        self.assertGreater((d.min(axis=0) < 2.5).mean(), 0.9)  # (nearly every light)
        self.assertLessEqual(len(found), len(pts))  # (once each)
        near = d.argmin(axis=1)
        hue = np.argmax(found[:, 3:6], axis=1)  # (its colour: red, green or blue, as its glow)
        self.assertGreater((hue == np.argmax(cols[near], axis=1)).mean(), 0.9)
        # only inside the box, and inside the area when one is painted
        half = np.array(bulbs.spots(rgb.astype(np.float32) / 255, (100, 0, 320, 360)))
        self.assertLess(len(half), 0.7 * len(found))
        area = np.zeros(rgb.shape[:2], np.float32)
        area[:180] = 1
        top = np.array(bulbs.spots(rgb.astype(np.float32) / 255, rect, area))
        self.assertTrue(len(top) and (top[:, 1] < 0.51).all())
        self.assertEqual(bulbs.spots(np.zeros((90, 160, 3), np.float32), (0, 0, 160, 90)), [])

    def test_drawn_where_they_are(self):
        spots = [[0.25, 0.5, 1.0, 1.0, 0.2, 0.1], [0.75, 0.25, 0.5, 0.1, 0.3, 1.0]]
        fx = make({"layout": "scene", "spots": spots, "amount": 100, "palette": "video", "jitter": 100})
        self.assertTrue(np.allclose(fx.x, [0.25, 0.75]) and np.allclose(fx.y, [0.5, 0.25]))
        self.assertTrue(np.allclose(fx.c_halo[0], [1.0, 0.2, 0.1]) and np.allclose(fx.c_halo[1], [0.1, 0.3, 1.0]))
        self.assertLess(fx.bright[1], fx.bright[0])  # (a fainter light in the video flares less)
        few = make({"layout": "scene", "spots": spots * 20, "amount": 0})
        self.assertEqual(len(few.x), 10)  # (a quarter of them at "amount" 0)
        none = make({"layout": "scene"})
        self.assertEqual(float(picture(none, 5, 640, 360).max()), 0.0)

    def test_only_their_light(self):
        """"Chỉ ánh sáng": no bulb of its own (no hot core), its light only."""
        bulb, glow = (make({"bulb": b, "glow": 0, "rays": 0, "jitter": 0, "brightness": 25}) for b in ("bulb", "glow"))
        a, b = picture(bulb, 9, 640, 360), picture(glow, 9, 640, 360)
        self.assertGreater(float(b.max()), 0.2)
        self.assertLess(float(b.max()), 0.85 * float(a.max()))
        self.assertEqual(effects.SCHEMA["lights"]["params"][0]["presets"]["scene"]["bulb"], "glow")

    def test_settings_keep_only_points(self):
        p = effects.clean_params("lights", {"spots": [[0.5, 0.5, 1, 1, 0, 0], "junk", [1, 2], [float("nan")] * 6,
                                                      [2, -1, "0.5", 0, 0, 0], {"x": 1}]})
        self.assertEqual(p["spots"], [[0.5, 0.5, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.5, 0.0, 0.0, 0.0]])
        self.assertEqual(effects.clean_params("lights", {"spots": "x"})["spots"], [])
        many = effects.clean_params("lights", {"spots": [[0.1] * 6] * 5000})["spots"]
        self.assertEqual(len(many), lights.SPOTS)

    def test_endpoint(self):
        """/api/bulbs: the lights in the project's video under the clip's box."""
        rgb, pts, _ = tree_with_lights()
        path = os.path.join(helpers.ROOT, "tree-lights.mp4")
        enc = video.Encoder(path, 640, 360, "24", crf=12, preset="veryfast")
        for _ in range(24):
            enc.write(rgb)
        enc.close()
        project = projects.create(path, "tree-lights.mp4")
        project = projects.apply(projects.load(project["id"]), {"layers": [
            {"id": "lights1", "kind": "lights", "start": 0, "end": 24, "params": {"scale": 100, "layout": "scene"}}]})
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        srv = server.start(server.App(desktop=False), port)
        try:
            def post(body):
                return request(f"http://127.0.0.1:{port}/api/bulbs", json.dumps(body).encode(),
                               {"X-Bokin": "1", "Content-Type": "application/json"}, "POST")
            code, body, _ = post({"id": project["id"], "layer": "lights1"})
            self.assertEqual(code, 200, body)
            self.assertGreater(len(json.loads(body)["spots"]), 0.8 * len(pts))
            code, body, _ = post({"id": project["id"], "layer": "nothere"})
            self.assertEqual((code, json.loads(body)["error"]), (400, "Không tìm thấy clip dây đèn."))
        finally:
            srv.shutdown()


if __name__ == "__main__":
    unittest.main()
