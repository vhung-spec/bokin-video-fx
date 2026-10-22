"""String lights drawn by hand anywhere on the picture (fx/garland.py, "Dây đèn tự vẽ"): a full-frame effect drawn as
light, its strands kept as places in the frame."""
import json
import math
import os
import unittest

import numpy as np

from tests import helpers
from fx import effects, engine, garland, lights, projects, video

MODES = [o["value"] for o in effects.SCHEMA["garland"]["params"][4]["options"]]
S, C, H = garland.START, garland.CLICK, garland.HAND
# a swag of clicks (hanging), an upright one, and a stretch drawn by hand (an arch)
STRANDS = [[0.1, 0.3, S, 0.5], [0.5, 0.3, C, 0], [0.9, 0.3, C, 0],
           [0.5, 0.45, S, 0.5], [0.5, 0.95, C, 0],
           [0.1, 0.9, S, 1.0]] + [[0.1 + 0.03 * i, 0.9 - 0.15 * math.sin(math.pi * i / 10), H, 0] for i in range(1, 11)]


def make(params=None, w=640, h=360, frames=144, fps=24, seed=3):
    return effects.make("garland", dict({"strands": STRANDS, "jitter": 0}, **(params or {})), seed, frames, fps, w, h)


def picture(fx, t, w, h):
    pic = np.zeros((h, w, 3), np.float32)
    canvas = effects.Canvas(w, h)
    fx.draw(t, canvas, None)
    canvas.flush(pic, fx.blend)
    return pic


class Garland(unittest.TestCase):
    def test_every_way_loops(self):
        """Every way of twinkling, bulbs or only their light: frame P is frame 0 again."""
        for mode in MODES:
            for bulb in ("bulb", "glow"):
                for frames, fps in ((144, 24), (150, 25)):
                    fx = make({"mode": mode, "bulb": bulb, "speed": 100, "jitter": 60}, 320, 180, frames, fps)
                    diff = np.abs(picture(fx, 0, 320, 180) - picture(fx, frames, 320, 180)).max()
                    self.assertLess(diff, 1 / 512, (mode, bulb, frames))

    def test_bulbs_along_each_strand(self):
        """One spacing on every strand ("gap", as on a real string): a longer strand has more bulbs, half a step in
        from its ends; denser at a higher setting; a strand shorter than a step: one bulb, in its middle."""
        fx = make({"palette": "multi"})
        gap = garland.gap_of(50)
        for i in range(3):
            on = fx.strand == i
            steps = np.hypot(np.diff(fx.x[on]), np.diff(fx.y[on]))
            self.assertLess(float(np.ptp(steps)), 0.25 * gap, i)  # (even along it: chords of a hanging wire)
            self.assertAlmostEqual(float(steps.mean()), gap, delta=0.12 * gap)
        upright = fx.strand == 1
        self.assertAlmostEqual(float(fx.y[upright].min()), 0.45 + 0.5 * 0.5 / upright.sum(), delta=1e-6)
        self.assertEqual(int(upright.sum()), int(math.floor(0.5 / gap + 0.5)))
        dense = make({"gap": 100})
        self.assertGreater(len(dense.x), 2.5 * len(fx.x))
        short = make({"strands": [[0.5, 0.5, S, 0], [0.505, 0.5, C, 0]]})
        self.assertEqual(len(short.x), 1)
        self.assertAlmostEqual(float(short.x[0]), 0.5025 * 640 / 360, delta=1e-6)
        # colours in turn along each strand
        one = np.flatnonzero(fx.strand == 0)[:10]
        self.assertEqual(len({tuple(np.round(c, 3)) for c in fx.c_halo[one[:5]]}), 5)

    def test_each_strand_its_own_sag(self):
        """Between two clicked points the wire hangs (a level span the most: a quarter of its width in its middle at
        sag 1), each strand as much as its own sag; an upright span does not, nor a stretch drawn by hand."""
        level = [[0.1, 0.3, S, 1.0], [0.9, 0.3, C, 0]]
        fx = make({"strands": level + [[0.1, 0.8, S, 0.0], [0.9, 0.8, C, 0]]})
        middle = (fx.strand == 0) & (np.abs(fx.x - 0.5 * 640 / 360) < 0.03)
        self.assertAlmostEqual(float(fx.y[middle].mean()), 0.3 + 0.25 * 0.8 * 640 / 360, delta=0.01)
        self.assertLess(float(np.abs(fx.y[fx.strand == 1] - 0.8).max()), 1e-9)  # (its own sag: 0)
        self.assertLess(float(np.abs(make().x[make().strand == 1] - 0.5 * 640 / 360).max()), 1e-9)  # (upright)
        arch = make()
        on = arch.strand == 2  # (sag 1, but drawn by hand: it follows the hand, under the arch's top)
        self.assertGreater(float(arch.y[on].min()), 0.9 - 0.15 - 0.01)

    def test_places_in_the_frame(self):
        """Any size of the same shape (the preview clip, the video) lays the same bulbs at the same places; nothing at
        all without a strand."""
        a, b = make({}, 1920, 1080), make({}, 1280, 720)
        self.assertTrue(np.allclose(a.x, b.x) and np.allclose(a.y, b.y))
        pa, pb = picture(make({}, 640, 360), 9, 640, 360), picture(make({}, 1280, 720), 9, 1280, 720)
        big = np.asarray(pb.reshape(360, 2, 640, 2, 3).mean(axis=(1, 3)))
        self.assertGreater(float(pa.max()), 0.5)
        self.assertLess(float(np.abs(pa - big).mean()), 0.01)
        self.assertEqual(float(picture(make({"strands": []}), 5, 640, 360).max()), 0.0)

    def test_at_most_cap_bulbs(self):
        """Very many long strands, densely: no more than CAP bulbs, still evenly spaced."""
        many = [[x, y, S if i == 0 else C, 0.2] for k in range(40) for i, (x, y) in
                enumerate([(0.0, k / 40), (1.0, k / 40 + 0.01)])]
        fx = make({"strands": many, "gap": 100})
        self.assertLessEqual(len(fx.x), garland.CAP)
        self.assertGreater(len(fx.x), 0.9 * garland.CAP)
        counts = np.bincount(fx.strand)
        self.assertLessEqual(int(np.ptp(counts)), 1)

    def test_settings(self):
        """Its strands: points of four numbers 0..1 (anything else left out, at most POINTS); no box, nothing to
        place or turn; the bulbs, twinkling and colours of the string lights; the page's cap of bulbs."""
        cls, sch = effects.KINDS["garland"], effects.SCHEMA["garland"]
        self.assertTrue(cls.BOX is None and not cls.TURNS and cls.blend == "screen" and cls.PREVIEW == 1280)
        keys = [d["key"] for d in sch["params"]]
        self.assertFalse({"x", "y", "scale", "rot_x", "layout", "amount"} & set(keys))
        self.assertEqual(keys[:3], ["strands", "gap", "jitter"])
        self.assertEqual(sch["bulbs_most"], garland.CAP)
        self.assertNotIn("video", [o["value"] for o in sch["params"][5]["options"]])
        p = effects.clean_params("garland", {"strands": [[0.2, 0.2, 1, 0.4], [3, -2, 0, 0], "x", [1, 2, 3],
                                                         [float("nan"), 0, 0, 0]]})
        self.assertEqual(p["strands"], [[0.2, 0.2, 1.0, 0.4], [1.0, 0.0, 0.0, 0.0]])
        many = effects.clean_params("garland", {"strands": [[0.1, 0.1, 0, 0]] * 9999})["strands"]
        self.assertEqual(len(many), garland.POINTS)

    def test_preview_clip(self):
        """Light on black over the whole frame (H.264, laid with Screen), as many pixels as a 16:9 clip 1280 wide."""
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 48, "seconds": 2.0}
        self.assertEqual(engine.fx_clip_frame(info, "garland"), (1280, 720))
        project = {"video": info, "length": 30}
        name = engine.fx_clip_name(project, {"kind": "garland", "params": {"strands": STRANDS}})
        self.assertTrue(name.endswith(".mp4"))
        other = engine.fx_clip_name(project, {"kind": "garland", "params": {"strands": STRANDS[:3]}})
        self.assertNotEqual(name, other)


class FromTheStringLights(unittest.TestCase):
    """Strands drawn by hand with an older version were string lights (layout "draw", their points in its box): a
    project opened now has them as Dây đèn tự vẽ, where they were, about as many bulbs, about as big."""

    def test_moved_on_load(self):
        path = os.path.join(helpers.ROOT, "garland-old.mp4")
        enc = video.Encoder(path, 640, 360, "24", crf=20, preset="veryfast")
        for _ in range(24):
            enc.write(np.full((360, 640, 3), 40, np.uint8))
        enc.close()
        project = projects.create(path, "garland-old.mp4")
        old = {"layout": "draw", "scale": 177.8, "x": 0, "y": 0, "sag": 40, "bulbs": 80, "size": 40,
               "palette": "video", "mode": "chase",
               "strands": [[0.1, 0.4, 1], [0.5, 0.4, 0], [0.9, 0.45, 0], [0.3, 0.6, 1], [0.6, 0.7, 0]]}
        file = os.path.join(projects.folder(project["id"]), "project.json")
        with open(file, encoding="utf-8") as f:
            raw = json.load(f)
        raw["layers"] = [{"id": "lights1", "kind": "lights", "name": "Dây đèn lấp lánh", "track": raw["tracks"][0]["id"],
                          "start": 0, "end": 24, "seed": 3, "params": old},
                         {"id": "lights2", "kind": "lights", "name": "Cây thông", "track": raw["tracks"][1]["id"],
                          "start": 0, "end": 24, "seed": 3, "params": {"layout": "tree", "scale": 50}}]
        with open(file, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        loaded = projects.load(project["id"])
        moved, tree = loaded["layers"]
        self.assertEqual((moved["kind"], moved["name"]), ("garland", "Dây đèn tự vẽ"))
        self.assertEqual((tree["kind"], tree["name"], tree["params"]["layout"]), ("lights", "Cây thông", "tree"))
        p = effects.clean_params("garland", moved["params"])
        self.assertEqual(p["mode"], "chase")
        self.assertEqual(p["palette"], "warm")  # (the lights' own colours in the video: no more)
        # where they were on the picture: the box (a square 177.8% of the height, in the middle) made the frame
        bx, by, _, bh = effects.place_box((1, 1), {"scale": 177.8, "x": 0, "y": 0}, 640, 360)
        for q, (u, v, new) in zip(p["strands"], old["strands"]):
            self.assertAlmostEqual(q[0], (bx + u * bh) / 640, delta=1e-4)
            self.assertAlmostEqual(q[1], (by + v * bh) / 360, delta=1e-4)
            self.assertEqual(q[2], 1.0 if new else 0.0)
            self.assertEqual(q[3], 0.4 if new else 0.0)
        # about as many bulbs and about as big as before
        before = effects.make("lights", {}, 3, 48, 24, 640, 360)  # (for its core size at "size" 40 in a box)
        now = effects.make("garland", p, 3, 48, 24, 640, 360)
        self.assertLess(abs(len(now.x) - 80), 6)
        core_px = float(np.median(now.core)) * 360
        was_px = float(np.median(before.core)) * bh
        self.assertLess(abs(core_px - was_px) / was_px, 0.12)
        # saved, it stays one: no string lights drawn by hand any more
        saved = projects.apply(loaded, {"layers": loaded["layers"]})
        self.assertEqual([lay["kind"] for lay in saved["layers"]], ["garland", "lights"])
        self.assertIsNone(lights.Lights.moved({"layout": "tree"}, 640, 360))


if __name__ == "__main__":
    unittest.main()
