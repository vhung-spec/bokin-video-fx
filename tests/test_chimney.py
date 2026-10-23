"""Smoke from the chimneys in a picture (fx/chimney.py, "Khói ống khói"): a full-frame effect, each chimney clicked on
the picture, its plume the smoke's own (fx/steam.py, its chimney style) in a box standing on its mouth."""
import json
import os
import unittest

import numpy as np

from tests import helpers
from fx import effects  # (first: it loads every effect's module)
from fx import chimney, engine, overlay, projects, steam, video

W, H = 640, 360
STACKS = [[0.25, 0.8, 0.05, 0.5], [0.7, 0.55, 0.025, 0.5]]  # a near chimney, and one farther away (higher, smaller)


def make(params=None, w=W, h=H, frames=144, fps=24, seed=3, light=None):
    return effects.make("chimney", dict({"stacks": STACKS}, **(params or {})), seed, frames, fps, w, h, light)


def picture(fx, t, base=0.2):
    pic = np.full((fx.h, fx.w, 3), base, np.float32)
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, None)
    canvas.flush(pic, fx.blend)
    return pic


def cover(fx, t):
    """How much smoke covers each pixel at frame t (on black, how far it is from black)."""
    return np.abs(picture(fx, t, 0.0)).max(axis=2)


def box_of(stack, tall=50, w=W, h=H):
    x0, y0, side = chimney.plume_box(*stack[:3], tall, w, h)
    return int(round(x0)), int(round(y0)), int(round(side))


class Chimney(unittest.TestCase):
    def test_every_way_loops(self):
        """In any wind, for any loop: frame P is frame 0 again."""
        for frames, fps in ((144, 24), (150, 25)):
            for params in ({}, {"breeze": -100, "gust": 100, "curl": 100}, {"tall": 100, "speed": 100}):
                fx = make(params, 320, 180, frames, fps)
                self.assertLess(float(np.abs(picture(fx, 0) - picture(fx, frames)).max()), 1 / 512, (frames, params))

    def test_from_each_mouth_in_its_box(self):
        """Each chimney's smoke rises from its mouth, inside the box chimney.plume_box() lays (as the page lays it):
        smoke just over each mouth, none below it, none outside the boxes."""
        fx = make({"breeze": 0, "gust": 0})
        self.assertEqual(len(fx.plumes), 2)
        seen = sum(cover(fx, t) for t in range(0, 144, 6))
        inside = np.zeros((H, W), bool)
        for stack in STACKS:
            x0, y0, side = box_of(stack)
            inside[max(0, y0):y0 + side, max(0, x0):x0 + side] = True
            mx, my, mw = stack[0] * W, stack[1] * H, stack[2] * W
            above = seen[int(my - 0.6 * side):int(my - 0.5 * mw), int(mx - mw):int(mx + mw)]
            self.assertGreater(float(above.max()), 0.2, stack)
            self.assertLess(float(seen[int(my + mw):, int(mx - mw):int(mx + mw)].max()), 1e-3, stack)  # (none below)
        self.assertLess(float(seen[~inside].max()), 1e-3)
        plume = {tuple(np.round(p.box(), 6)) for p in fx.plumes}
        self.assertEqual(plume, {tuple(np.round((*chimney.plume_box(*s[:3], 50, W, H)[:2],
                                                 *[chimney.plume_box(*s[:3], 50, W, H)[2]] * 2), 6)) for s in STACKS})

    def test_its_size_follows_the_mouth(self):
        """A mouth twice as wide: a box twice as big; "Khói bay cao" 100: twice as high as at 50, 0: half."""
        x0, y0, side = chimney.plume_box(0.5, 0.8, 0.04, 50, W, H)
        self.assertAlmostEqual(side, 0.04 * W * chimney.TALL)
        self.assertAlmostEqual(chimney.plume_box(0.5, 0.8, 0.08, 50, W, H)[2], 2 * side)
        self.assertAlmostEqual(chimney.plume_box(0.5, 0.8, 0.04, 100, W, H)[2], 2 * side)
        self.assertAlmostEqual(chimney.plume_box(0.5, 0.8, 0.04, 0, W, H)[2], side / 2)
        # the mark (the smoke's ellipse) on the mouth: as wide, its middle where the chimney was clicked
        fx = make({"stacks": [[0.5, 0.8, 0.04, 0.5]]})
        plume = fx.plumes[0]
        bx, by, bw, bh = plume.box()
        self.assertAlmostEqual(plume.p["rim"] / 100 * bw, 0.04 * W)
        self.assertAlmostEqual(by + bh - steam.MARK / 2 * 0.04 * W, 0.8 * H)
        self.assertAlmostEqual(bx + bw / 2, 0.5 * W)

    def test_far_ones_behind(self):
        """The higher up the picture a chimney is, the farther away: its smoke is drawn first."""
        fx = make({"stacks": [[0.2, 0.9, 0.04, 0.5], [0.5, 0.4, 0.02, 0.5], [0.8, 0.6, 0.03, 0.5]]})
        ys = [p.box()[1] + p.box()[3] for p in fx.plumes]
        self.assertEqual(ys, sorted(ys))

    def test_how_much_smoke_each(self):
        """Each chimney's own amount: none at 0 (no plume at all), more at 1 than at 0.5."""
        one = [0.5, 0.8, 0.05]
        self.assertEqual(len(make({"stacks": [one + [0.0]]}).plumes), 0)
        some, more = (sum(cover(fx, t).sum() for t in range(0, 144, 6))
                      for fx in (make({"stacks": [one + [m]], "amount": 40}) for m in (0.5, 1.0)))
        self.assertGreater(more, 1.2 * some)
        self.assertEqual(len(make({"stacks": []}).plumes), 0)
        self.assertEqual(float(np.abs(picture(make({"stacks": []}), 5) - np.float32(0.2)).max()), 0.0)

    def test_one_wind_for_all(self):
        """The wind is the same for every chimney: with it from the left every plume leans right, from the right left;
        no two plumes puff in step (each its own seed)."""
        for breeze, sign in ((100, 1), (-100, -1)):
            fx = make({"breeze": breeze, "gust": 0})
            seen = sum(cover(fx, t) for t in range(0, 144, 6))
            for stack in STACKS:
                x0, y0, side = box_of(stack)
                part = seen[max(0, y0):y0 + side, max(0, x0):x0 + side]
                cols = np.arange(part.shape[1]) + max(0, x0) - stack[0] * W
                mid = float((part.sum(axis=0) * cols).sum() / part.sum())
                self.assertGreater(sign * mid, 0.03 * side, (breeze, stack))
        fx = make({"stacks": [[0.3, 0.8, 0.05, 0.5], [0.7, 0.8, 0.05, 0.5]], "breeze": 0, "gust": 0})
        both = [cover(fx, t) for t in range(0, 144, 6)]
        a = np.stack([c[:, :W // 2] for c in both])
        b = np.stack([c[:, W // 2:][:, ::-1] for c in both])  # (the other one, mirrored: in step, it would match)
        self.assertGreater(float(np.abs(a - b).mean()), 0.002)

    def test_lit_by_the_scene(self):
        """Each plume is lit by the light where it stands: one by a warm lamp is warmer and brighter than one in the
        dark, in the same clip."""
        scene = np.full((H, W, 3), 12, np.uint8)
        yy, xx = np.mgrid[0:H, 0:W]
        lamp = np.exp(-((xx - 160) ** 2 + (yy - 200) ** 2) / (2 * 50.0 ** 2))[..., None] * np.array([255, 190, 110])
        scene = np.clip(scene + lamp, 0, 255).astype(np.uint8)
        light = effects.light_map([scene])
        fx = make({"stacks": [[0.25, 0.8, 0.05, 0.5], [0.75, 0.8, 0.05, 0.5]], "lit": 100}, light=light)
        near, far = sorted(fx.plumes, key=lambda p: p.box()[0])
        self.assertGreater(float(np.mean(near.colours[1])), float(np.mean(far.colours[1])) + 0.1)
        self.assertGreater(float(near.colours[1][0] - near.colours[1][2]), 0.05)  # (warm)
        unlit = make({"lit": 0}, light=light)
        self.assertEqual([tuple(p.colours[0]) for p in unlit.plumes], [tuple(steam._rgb("#c3c4c6"))] * 2)

    def test_settings(self):
        """Its chimneys: points of four numbers 0..1 (anything else left out, at most STACKS); no box, no place, no
        turn; the smoke's own sliders at the values its chimney style sets them; the page's numbers."""
        cls, sch = effects.KINDS["chimney"], effects.SCHEMA["chimney"]
        self.assertTrue(cls.BOX is None and not cls.TURNS and cls.LIT and cls.blend == "over" and cls.PREVIEW == 1280)
        keys = [d["key"] for d in sch["params"]]
        self.assertEqual(keys[:2], ["stacks", "tall"])
        self.assertFalse({"x", "y", "scale", "rot_x", "rim", "reach", "style", "look"} & set(keys))
        defaults = effects.defaults("chimney")
        for key in chimney.SHARED:
            self.assertEqual(defaults[key], steam.STYLE_PRESETS["chimney"].get(key, effects.defaults("steam")[key]), key)
        self.assertEqual((sch["stacks_most"], sch["mouth"], sch["much"]), (chimney.STACKS, chimney.MOUTH, chimney.MUCH))
        self.assertEqual(sch["plume"], {"ratio": steam.MARK, "tall": chimney.TALL, "side": chimney.SIDE})
        p = effects.clean_params("chimney", {"stacks": [[0.2, 0.3, 0.05, 0.5], [3, -2, 0.1, 9], "x", [1, 2, 3],
                                                         [float("nan"), 0, 0, 0]]})
        self.assertEqual(p["stacks"], [[0.2, 0.3, 0.05, 0.5], [1.0, 0.0, 0.1, 1.0]])
        many = effects.clean_params("chimney", {"stacks": [[0.1, 0.1, 0.02, 0.5]] * 999})["stacks"]
        self.assertEqual(len(many), chimney.STACKS)

    def test_preview_clip(self):
        """See-through smoke over the whole frame (WebM), as many pixels as a 16:9 clip 1280 wide; another clip for
        other chimneys."""
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 48, "seconds": 2.0}
        self.assertEqual(engine.fx_clip_frame(info, "chimney"), (1280, 720))
        project = {"video": info, "length": 30}
        name = engine.fx_clip_name(project, {"kind": "chimney", "params": {"stacks": STACKS, "lit": 0}})
        self.assertTrue(name.endswith(".webm"))
        other = engine.fx_clip_name(project, {"kind": "chimney", "params": {"stacks": STACKS[:1], "lit": 0}})
        self.assertNotEqual(name, other)


class FromTheSmoke(unittest.TestCase):
    """Smoke from a chimney made with the smoke (its style "chimney", 1.0.60 to 1.0.62: a box placed, perhaps turned,
    its mark on the mouth): a project opened now has it as Khói ống khói, one chimney where the mark was, as wide, its
    smoke as high, its wind and look."""

    def test_moved_on_load(self):
        path = os.path.join(helpers.ROOT, "chimney-old.mp4")
        enc = video.Encoder(path, 640, 360, "24", crf=20, preset="veryfast")
        for _ in range(24):
            enc.write(np.full((360, 640, 3), 40, np.uint8))
        enc.close()
        project = projects.create(path, "chimney-old.mp4")
        old = dict(steam.STYLE_PRESETS["chimney"], style="chimney", scale=60, x=-12, y=-20, breeze=-45, amount=80)
        turned = dict(old, rot_y=30, rot_z=8)
        file = os.path.join(projects.folder(project["id"]), "project.json")
        with open(file, encoding="utf-8") as f:
            raw = json.load(f)
        raw["layers"] = [{"id": "smoke1", "kind": "steam", "name": "Khói", "track": raw["tracks"][0]["id"],
                          "start": 0, "end": 24, "seed": 3, "params": old},
                         {"id": "smoke2", "kind": "steam", "name": "Mái nhà", "track": raw["tracks"][1]["id"],
                          "start": 0, "end": 24, "seed": 4, "params": turned},
                         {"id": "cup1", "kind": "steam", "name": "Khói", "track": raw["tracks"][1]["id"],
                          "start": 0, "end": 24, "seed": 5, "params": dict(steam.STYLE_PRESETS["cup"], style="cup")}]
        with open(file, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        loaded = projects.load(project["id"])
        moved, turn, cup = loaded["layers"]
        self.assertEqual((moved["kind"], moved["name"]), ("chimney", "Khói ống khói"))
        self.assertEqual((turn["kind"], turn["name"]), ("chimney", "Mái nhà"))  # (a name of its own stays)
        self.assertEqual((cup["kind"], cup["params"]["style"]), ("steam", "cup"))
        p = effects.clean_params("chimney", moved["params"])
        self.assertEqual((p["breeze"], p["amount"], p["tall"]), (-45, 80, 50))
        # where it was: the box (a square 60% of the height) made the frame, its mark on its bottom edge, rim wide
        bx, by, side, _ = effects.place_box((1, 1), old, 640, 360)
        (x, y, wid, much), = p["stacks"]
        mw = old["rim"] / 100 * side
        self.assertAlmostEqual(x * 640, bx + side / 2, delta=0.1)
        self.assertAlmostEqual(y * 360, by + side - steam.MARK / 2 * mw, delta=0.1)
        self.assertAlmostEqual(wid * 640, mw, delta=0.1)
        self.assertEqual(much, chimney.MUCH)
        # its smoke rises in the same box as before
        now = effects.make("chimney", p, 3, 48, 24, 640, 360).plumes[0].box()
        for a, b in zip(now, (bx, by, side, side)):
            self.assertAlmostEqual(a, b, delta=0.5)
        # turned: where its mark was in the frame (the turn of a file clip, as the page shows it)
        q = effects.clean_params("chimney", turn["params"])["stacks"][0]
        rect = effects.sheet_rect("steam", effects.clean_params("steam", dict(turned, style="cup")), 640, 360)
        hm = overlay.homography(effects.clean_params("steam", dict(turned, style="cup")), rect, 640, 360)[0]
        mid = hm @ (rect[2] / 2, rect[3] - steam.MARK / 2 * old["rim"] / 100 * rect[2], 1.0)
        self.assertAlmostEqual(q[0] * 640, mid[0] / mid[2], delta=0.1)
        self.assertAlmostEqual(q[1] * 360, mid[1] / mid[2], delta=0.1)
        # saved, it stays one: no smoke from a chimney made with the smoke any more
        saved = projects.apply(loaded, {"layers": loaded["layers"]})
        self.assertEqual([lay["kind"] for lay in saved["layers"]], ["chimney", "chimney", "steam"])
        self.assertIsNone(steam.moved({"style": "cup"}, 640, 360))


if __name__ == "__main__":
    unittest.main()
