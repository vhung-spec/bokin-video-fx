"""Twinkling string lights (fx/lights.py): a placed effect drawn as light."""
import unittest

import numpy as np

from fx import effects, engine, lights

LAYOUTS = list(lights.LAYOUTS)
MODES = [o["value"] for o in effects.SCHEMA["lights"]["params"][2]["options"]]


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
        self.assertEqual(a, engine.fx_clip_name(project, {"kind": "lights", "params": {"scale": 45, "y": -30}}))
        # (sized across a step: a clip of another size, engine.SNUG)
        self.assertNotEqual(a, engine.fx_clip_name(project, {"kind": "lights", "params": {"scale": 20, "y": -30}}))
        self.assertTrue(a.endswith(".mp4"))  # (light on black, laid with Screen)

    def test_only_their_light(self):
        """"Chỉ ánh sáng": no bulb of its own (no hot core), its light only."""
        bulb, glow = (make({"bulb": b, "glow": 0, "rays": 0, "jitter": 0, "brightness": 25}) for b in ("bulb", "glow"))
        a, b = picture(bulb, 9, 640, 360), picture(glow, 9, 640, 360)
        self.assertGreater(float(b.max()), 0.2)
        self.assertLess(float(b.max()), 0.85 * float(a.max()))

    def test_big_box_sharper_clip(self):
        """A big box of small bulbs (a tree filling the picture): a preview clip PREVIEW high (tiny bulbs blur at 640);
        a smaller one no sharper than the box is in the copy of the video the page plays, a step up (sizing the box
        does not make it again and again)."""
        info = {"w": 1920, "h": 1080, "fps": 24, "frames": 144}
        size = lambda scale: engine.fx_clip_frame(info, "lights", {"kind": "lights", "params": {"scale": scale}})
        self.assertEqual(size(60), (480, 480))
        self.assertEqual(size(66), (480, 480))
        self.assertEqual(size(100), (1280, 1280))
        self.assertEqual(size(177.8), (1280, 1280))
        self.assertEqual(engine.fx_clip_frame(info, "steam", {"kind": "steam", "params": {"scale": 177.8}}),
                         (640, 640))  # (only an effect with a PREVIEW)

    def test_ways_to_lay_them(self):
        """Round a tree, swags, a curtain or scattered (strands drawn by hand: Dây đèn tự vẽ, fx/garland.py; the
        lights a video has are no longer looked for): a value of an older version falls back to the tree."""
        by = {d["key"]: d for d in effects.SCHEMA["lights"]["params"]}
        self.assertEqual([o["value"] for o in by["layout"]["options"]], ["tree", "swag", "curtain", "scatter"])
        self.assertEqual(by["back"]["show"], {"layout": ["tree"]})
        self.assertNotIn("video", [o["value"] for o in by["palette"]["options"]])
        self.assertFalse({"spots", "strands", "bulbs", "sag"} & set(by))
        p = effects.clean_params("lights", {"layout": "scene", "palette": "video", "spots": [[0.5] * 6]})
        self.assertEqual((p["layout"], p["palette"], "spots" in p), ("tree", "warm", False))


if __name__ == "__main__":
    unittest.main()
