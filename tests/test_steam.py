import unittest

import numpy as np
from PIL import Image

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, engine, steam

W, H = 320, 180
PLACE = ("x", "y", "scale")
RANGES = [p for p in steam.SCHEMA["params"] if p["type"] == "range" and p["key"] not in PLACE]
MAX = {p["key"]: p["max"] for p in RANGES}
MIN = {p["key"]: p["min"] for p in RANGES}
BIG = {"scale": 90, "x": 0, "y": 0}  # a big box in the small test frame


def frame(fx, t, base=0.3, mask=None):
    pic = np.full((fx.h, fx.w, 3), base, np.float32)
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, mask)
    canvas.flush(pic, fx.blend)
    return pic


def cover(fx, t, mask=None):
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, mask)
    return canvas.a.copy()


def inside(fx):
    """Rows and columns of the frame inside the effect's box (with a pixel to spare for rounding)."""
    x0, y0, w, h = fx.box()
    return slice(max(0, int(y0) - 1), int(y0 + h) + 2), slice(max(0, int(x0) - 1), int(x0 + w) + 2)


class Loops(unittest.TestCase):
    """Frame P is frame 0 again with every look and at the extremes, at any loop length and frame rate."""

    CASES = [dict(steam.LOOKS[name], look=name) for name in steam.LOOKS] + [
        MAX, dict(MIN, amount=100, opacity=100), {"breeze": -100, "gust": 100, "curl": 100, "speed": 100},
        {"speed": 0, "reach": 100, "curl": 0, "spread": 100}]

    def test_frame_p_is_frame_zero(self):
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
            for params in self.CASES if frames != 720 else self.CASES[:1]:
                fx = effects.make("steam", {**params, **BIG}, 3, frames, fps, W, H)
                first = frame(fx, 0)
                self.assertGreater(np.abs(first - 0.3).max(), 0.02, params)  # (there is steam to see)
                diff = np.abs(first - frame(fx, frames)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")

    def test_seam_drawn_frame_by_frame(self):
        """An export draws frame after frame: the steam it carries on to at frame P is the steam a new start finds
        at frame 0."""
        for frames, fps in ((144, 24), (150, 25), (90, 30)):
            for params in ({}, self.CASES[-2]):
                going = effects.make("steam", {**params, **BIG}, 5, frames, fps, W, H)
                for t in range(frames - 40, frames):
                    frame(going, t)
                fresh = effects.make("steam", {**params, **BIG}, 5, frames, fps, W, H)
                diff = np.abs(frame(going, frames) - frame(fresh, 0)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")

    def test_any_order(self):
        """Frame by frame or straight to a frame, the steam is the same."""
        a = effects.make("steam", BIG, 2, 144, 24, W, H)
        for t in range(20, 31):
            frame(a, t)
        b = effects.make("steam", BIG, 2, 144, 24, W, H)
        self.assertLess(np.abs(frame(a, 31) - frame(b, 31)).max(), 1e-4)

    def test_rises_and_changes(self):
        fx = effects.make("steam", BIG, 3, 144, 24, W, H)
        self.assertGreater(np.abs(frame(fx, 0) - frame(fx, 12)).max(), 0.02)
        # every bit of steam is higher a tick later (row a, the age in ticks, is row a + 1 a tick later)
        before, after = fx._state(0)[1][:-1], fx._state(1)[1][1:]
        self.assertGreater(float((after - before).min()), 0)

    def test_same_seed_same_steam(self):
        a, b = (effects.make("steam", BIG, 9, 144, 24, W, H) for _ in range(2))
        c = effects.make("steam", BIG, 10, 144, 24, W, H)
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))
        self.assertFalse(np.array_equal(frame(a, 17), frame(c, 17)))

    def test_between_frames(self):
        """A time between two frames is drawn too (between them)."""
        fx = effects.make("steam", BIG, 3, 144, 24, W, H)
        a, b, mid = cover(fx, 10), cover(fx, 11), cover(fx, 10.5)
        self.assertLessEqual(np.abs(mid - (a + b) / 2).mean(), np.abs(a - b).mean() + 1e-6)


class Box(unittest.TestCase):
    """A placed effect: the steam rises from the middle of its box's bottom and stays inside the box."""

    def test_steam_stays_in_its_box(self):
        for params in ({"scale": 40, "x": 20, "y": -10},
                       {"scale": 60, "x": -25, "y": 15, "breeze": 100, "spread": 100, "curl": 100, "reach": 100,
                        "rim": 100, "gust": 100}):
            fx = effects.make("steam", params, 2, 144, 24, 640, 360)
            rows, cols = inside(fx)
            for t in (0, 40, 90):
                a = cover(fx, t)
                self.assertGreater(a[rows, cols].max(), 0.05, params)
                a[rows, cols] = 0
                self.assertEqual(a.max(), 0, params)

    def test_from_the_rim_up(self):
        fx = effects.make("steam", {"scale": 90, "x": 0, "y": 0}, 4, 144, 24, 640, 360)
        x0, y0, w, h = fx.box()
        total = sum(cover(fx, t) for t in range(0, 144, 12))
        box = total[int(y0):int(y0 + h), int(x0):int(x0 + w)]
        bh, bw = box.shape
        low = box[int(bh * 0.85):]  # just above the rim: in the middle, not at the sides
        self.assertGreater(low[:, int(bw * 0.3):int(bw * 0.7)].mean(), 5 * low[:, :int(bw * 0.2)].mean() + 1e-4)
        self.assertGreater(low[:, int(bw * 0.3):int(bw * 0.7)].mean(), 5 * low[:, -int(bw * 0.2):].mean() + 1e-4)
        self.assertLess(box[:int(bh * 0.02)].max(), 0.01)  # faded before the top
        self.assertLess(box[:, :int(bw * 0.02)].max(), 0.01)  # and the sides
        self.assertLess(box[:, -int(bw * 0.02):].max(), 0.01)

    def test_reach(self):
        def upper(reach):
            fx = effects.make("steam", {"scale": 90, "reach": reach}, 4, 144, 24, 640, 360)
            x0, y0, w, h = fx.box()
            a = sum(cover(fx, t) for t in range(0, 144, 24))
            return a[int(y0):int(y0 + h * 0.5), int(x0):int(x0 + w)].mean()
        self.assertLess(upper(30), 1e-4)
        self.assertGreater(upper(100), 0.005)

    def test_breeze_leans(self):
        def lean(breeze):
            fx = effects.make("steam", {"scale": 90, "breeze": breeze, "gust": 0}, 4, 144, 24, 640, 360)
            x0, y0, w, h = fx.box()
            a = sum(cover(fx, t) for t in range(0, 144, 24))[int(y0):int(y0 + h * 0.6), int(x0):int(x0 + w)]
            return float((a.sum(axis=0) * np.arange(a.shape[1])).sum() / a.sum() / a.shape[1] - 0.5)
        self.assertGreater(lean(100), 0.05)
        self.assertLess(lean(-100), -0.05)
        self.assertLess(abs(lean(0)), 0.1)

    def test_preview_clip_is_the_box(self):
        """The page's preview clip (the box alone, at its own size) looks like the steam drawn in the frame."""
        info = {"w": 960, "h": 540, "fps": "24", "frames": 144, "seconds": 6.0}
        cw, ch = engine.fx_clip_frame(info, "steam")
        self.assertEqual((cw, ch), (540, 540))
        for params in ({"scale": 50, "x": 12, "y": -5}, {"scale": 30, "x": -20, "y": 10, "look": "long",
                                                         **steam.LOOKS["long"]}):
            clip = effects.make("steam", engine.fx_clip_params("steam", params), 4, 144, 24, cw, ch)
            full = effects.make("steam", params, 4, 144, 24, 960, 540)
            self.assertEqual(tuple(round(v) for v in clip.box()), (0, 0, cw, ch))
            for t in (0, 70):
                rows, cols = inside(full)
                x0, y0, w, h = full.box()
                bx, by = int(round(x0)), int(round(y0))
                bw, bh = int(round(x0 + w)) - bx, int(round(y0 + h)) - by
                a = cover(full, t)[by:by + bh, bx:bx + bw]
                b = np.asarray(Image.fromarray(cover(clip, t)).resize((bw, bh), Image.Resampling.BILINEAR))
                self.assertGreater(a.max(), 0.05)
                self.assertLess(np.abs(a - b).mean(), 0.05 * a.mean() + 1e-4, params)
                self.assertLess(np.abs(a - b).max(), 0.25 * a.max(), params)

    def test_off_the_frame_and_extreme_sizes(self):
        for params in ({"scale": 80, "x": 60, "y": 40}, {"scale": 1}, {"scale": 300}, {"scale": 10, "x": 100},
                       {"scale": 50, "y": -100}):
            fx = effects.make("steam", params, 2, 144, 24, W, H)
            a = cover(fx, 33)
            self.assertTrue(np.isfinite(a).all())
            self.assertLessEqual(a.max(), 1.0)
        fx = effects.make("steam", {"scale": 10, "x": 100, "y": 100}, 2, 144, 24, W, H)  # nowhere in the frame
        self.assertEqual(cover(fx, 5).max(), 0)


class Picture(unittest.TestCase):
    def test_keeps_out_of_its_area(self):
        """Outside the clip's area (mask 0) the picture stays exactly as it was; a half mask halves the steam."""
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        fx = effects.make("steam", dict(BIG, amount=100, opacity=100), 3, 144, 24, W, H)
        pic = frame(fx, 7, mask=mask)
        self.assertTrue(np.array_equal(pic[:, :W // 2], np.full((H, W // 2, 3), 0.3, np.float32)))
        self.assertGreater(pic[:, W // 2:].max(), 0.33)
        half = cover(fx, 7, np.full((H, W), 0.5, np.float32))
        self.assertAlmostEqual(float(half.max()), float(cover(fx, 7).max()) * 0.5, places=5)

    def test_nothing_without_steam(self):
        for params in ({"amount": 0}, {"opacity": 0}):
            fx = effects.make("steam", {**BIG, **params}, 3, 144, 24, W, H)
            self.assertTrue(np.array_equal(frame(fx, 5), np.full((H, W, 3), 0.3, np.float32)), params)

    def test_shows_on_dark_and_bright(self):
        """Laid over the picture: pale steam lightens a dark room and still shows (a little) on a bright one."""
        fx = effects.make("steam", BIG, 3, 144, 24, W, H)
        self.assertGreater(frame(fx, 9, base=0.05).max(), 0.15)
        bright = frame(fx, 9, base=0.75)
        self.assertGreater(bright.max(), 0.76)
        self.assertLessEqual(bright.max(), 1.0)

    def test_colour(self):
        fx = effects.make("steam", dict(BIG, color="#ff0000", opacity=100), 3, 144, 24, W, H)
        pic = frame(fx, 9, base=0.0)
        self.assertGreater(pic[..., 0].max(), 0.1)
        self.assertEqual(pic[..., 1:].max(), 0.0)


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_looks_set_them(self):
        params = steam.SCHEMA["params"]
        self.assertTrue(all(p.get("group") for p in params))
        look = next(p for p in params if p["key"] == "look")
        defaults = effects.defaults("steam")
        self.assertEqual(look["value"], "light")
        keys = {p["key"] for p in params} - set(PLACE) - {"look", "color"}
        for name, preset in look["presets"].items():
            self.assertEqual(set(preset), keys, name)  # every setting but the place and the colour
            self.assertEqual(effects.clean_params("steam", preset), {**defaults, **preset}, name)  # in range
        self.assertEqual(look["presets"]["light"], {k: defaults[k] for k in keys})  # the default look: defaults
        self.assertEqual({o["value"] for o in look["options"]}, set(steam.LOOKS))

    def test_box(self):
        self.assertEqual(steam.Steam.BOX, (1, 1))
        fx = effects.make("steam", {}, 1, 144, 24, 1920, 1080)
        x0, y0, w, h = fx.box()
        self.assertAlmostEqual(h, 1080 * effects.defaults("steam")["scale"] / 100)
        self.assertAlmostEqual(x0 + w / 2, 960)


if __name__ == "__main__":
    unittest.main()
