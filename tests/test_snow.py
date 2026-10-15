import math
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, snow

W, H = 320, 180


def frame(fx, t, w=W, h=H, mask=None, base=0.3):
    pic = np.full((h, w, 3), base, np.float32)
    canvas = effects.Canvas(w, h)
    fx.draw(t, canvas, mask)
    canvas.flush(pic, fx.blend)
    return pic


def coverage(fx, t):
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, None)
    return canvas.a


MAX = {p["key"]: p["max"] for p in snow.SCHEMA["params"] if p["type"] == "range"}
MIN = {p["key"]: p["min"] for p in snow.SCHEMA["params"] if p["type"] == "range"}


class Loops(unittest.TestCase):
    """Frame P is frame 0 again with every look and every kind of flake, at any loop length and frame rate."""

    CASES = [dict(snow.LOOKS[name], look=name) for name in snow.LOOKS] + [
        MAX, dict(MIN, amount=100), {"flake": "crystal", "size": 100, "spin": 100, "twinkle": 100},
        {"wind": -100, "gust": 100, "swirl": 100, "sway": 100, "near": 100, "glow": 100},
        {"flake": "soft", "speed": 100, "depth": 100, "blur": 0}]

    def test_frame_p_is_frame_zero(self):
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
            for params in self.CASES:
                w, h = (640, 360) if params.get("size") == 100 else (W, H)  # big enough to show crystals
                fx = effects.make("snow", params, 3, frames, fps, w, h)
                diff = np.abs(frame(fx, 0, w, h) - frame(fx, frames, w, h)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")

    def test_crystals_turn_and_glint(self):
        """Crystals are really drawn (and turn), and glints come and go: the case above covers them."""
        fx = effects.make("snow", {"flake": "crystal", "size": 100, "spin": 100, "twinkle": 100}, 3, 144, 24, 640, 360)
        self.assertTrue(fx.crystal.any())
        self.assertTrue((fx.turns != 0).any())
        self.assertTrue((fx.glints > 0).any())
        soft = effects.make("snow", {"flake": "soft", "size": 100}, 3, 144, 24, 640, 360)
        self.assertFalse(soft.crystal.any())

    def test_no_flake_jumps_while_the_shutter_is_open(self):
        """A flake leaving at the bottom starts again at the top: its smear must not span the frame."""
        fx = effects.make("snow", {"speed": 100, "amount": 60, "near": 100}, 3, 144, 24, W, H)
        for t in range(144):
            x0, y0, _ = fx.at(t, -fx.SHUTTER / 2)
            x1, y1, _ = fx.at(t, fx.SHUTTER / 2)
            self.assertLess(float(np.hypot(x1 - x0, y1 - y0).max()), 0.25 * H, t)

    def test_gust_curve_is_periodic(self):
        self.assertAlmostEqual(float(snow._GUST[0]), float(snow._GUST[-1]))
        self.assertAlmostEqual(float(snow._GUST.max() - snow._GUST.min()), 1.0, places=6)
        a = snow.gust_push(np.array([0.3, 0.3 + 2 * math.pi * 5]))
        self.assertAlmostEqual(float(a[0]), float(a[1]), places=9)


class Settings(unittest.TestCase):
    def test_old_settings_keep_their_meaning(self):
        """Old projects keep what they had: the first version's settings, ranges and defaults are all there."""
        old = {"amount": (0, 100, 45), "size": (0, 100, 40), "speed": (0, 100, 35), "wind": (-100, 100, 10),
               "sway": (0, 100, 40), "depth": (0, 100, 60), "blur": (0, 100, 40), "opacity": (0, 100, 85)}
        by = {p["key"]: p for p in snow.SCHEMA["params"]}
        for key, (lo, hi, value) in old.items():
            self.assertEqual((by[key]["min"], by[key]["max"], by[key]["value"]), (lo, hi, value), key)
        self.assertEqual(by["color"]["value"], "#ffffff")
        self.assertEqual(snow.Snow.VERSION, 2)
        self.assertEqual(snow.Snow.blend, "over")

    def test_every_setting_has_a_heading(self):
        for p in snow.SCHEMA["params"]:
            self.assertTrue(p.get("group"), p["key"])
            self.assertTrue(p["label"], p["key"])

    def test_looks(self):
        """Each look sets every setting but the colours, to values the page accepts; "Tuyết vừa" is the defaults."""
        look = next(p for p in snow.SCHEMA["params"] if p["key"] == "look")
        self.assertEqual({o["value"] for o in look["options"]}, set(snow.LOOKS))
        self.assertIs(look["presets"], snow.LOOKS)
        keys = {p["key"] for p in snow.SCHEMA["params"]} - {"look", "color", "far_color"}
        for name, preset in snow.LOOKS.items():
            self.assertEqual(set(preset), keys, name)
            self.assertEqual(effects.clean_params("snow", preset), {**effects.defaults("snow"), **preset}, name)
        self.assertEqual(snow.LOOKS["medium"], {k: v for k, v in effects.defaults("snow").items() if k in keys})

    def test_nothing_when_amount_is_zero(self):
        fx = effects.make("snow", dict(MAX, amount=0), 3, 144, 24, W, H)
        self.assertTrue(np.array_equal(frame(fx, 5), np.full((H, W, 3), 0.3, np.float32)))

    def test_stays_in_its_area(self):
        mask = np.zeros((360, 640), np.float32)
        mask[:, 320:] = 1
        fx = effects.make("snow", dict(MAX, flake="crystal"), 3, 144, 24, 640, 360)
        pic = frame(fx, 7, 640, 360, mask)
        self.assertTrue(np.array_equal(pic[:, :320], np.full((360, 320, 3), 0.3, np.float32)))
        self.assertGreater(pic[:, 320:].max(), 0.6)

    def test_looks_or_light_do_not_move_the_flakes(self):
        """Settings of how flakes look leave every flake where it was; more flakes keep the ones already there."""
        base = effects.make("snow", {}, 4, 144, 24, W, H)
        x, y, _ = base.at(37)
        for change in ({"twinkle": 100}, {"glow": 100}, {"opacity": 10}, {"color": "#ffd0d0"}, {"flake": "soft"},
                       {"spin": 100}, {"far_color": "#000000"}, {"look": "blizzard"}):
            other = effects.make("snow", change, 4, 144, 24, W, H)
            ox, oy, _ = other.at(37)
            self.assertTrue(np.array_equal(x, ox) and np.array_equal(y, oy), change)
        nearer = effects.make("snow", {"near": 100}, 4, 144, 24, W, H)
        nx, ny, _ = nearer.at(37)
        self.assertTrue(np.array_equal(x[~base.near], nx[~nearer.near]))
        more = effects.make("snow", {"amount": 70}, 4, 144, 24, W, H)
        mx, my, _ = more.at(37)
        here = {(round(a, 3), round(b, 3)) for a, b in zip(mx, my)}
        self.assertTrue(all((round(a, 3), round(b, 3)) in here for a, b in zip(x, y)))

    def test_preview_looks_like_the_video(self):
        """Sizes are given for a 1080-line frame: the small preview is covered about as much as the full video."""
        small = effects.make("snow", {}, 2, 144, 24, 640, 360)
        full = effects.make("snow", {}, 2, 144, 24, 1920, 1080)
        a = float(np.mean([coverage(small, t).mean() for t in (0, 50, 100)]))
        b = float(np.mean([coverage(full, t).mean() for t in (0, 50, 100)]))
        self.assertLess(abs(a / b - 1), 0.3, (a, b))

    def test_crystal_designs_are_symmetric(self):
        for shape in range(len(snow.CRYSTAL_KINDS)):
            a = np.asarray(snow.crystal(shape, 58), np.float32) / 255.0
            self.assertGreater(a.max(), 0.9)
            self.assertLess(float(np.abs(a - a[:, ::-1]).mean()), 0.02, shape)


if __name__ == "__main__":
    unittest.main()
