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

    CRYSTALS = {"flake": "crystal", "size": 80, "spin": 100, "twinkle": 100}
    CASES = [dict(snow.LOOKS[name], look=name) for name in snow.LOOKS] + [
        MAX, dict(MIN, amount=100), {"flake": "crystal", "size": 100, "spin": 100, "twinkle": 100}, CRYSTALS,
        {"wind": -100, "gust": 100, "swirl": 100, "sway": 100, "near": 100, "glow": 100, "haze": 100},
        {"flake": "soft", "speed": 100, "depth": 100, "blur": 0}]

    def test_frame_p_is_frame_zero(self):
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
            for params in self.CASES:
                w, h = (640, 360) if params.get("size") == 100 else (W, H)  # big enough to show crystals
                fx = effects.make("snow", params, 3, frames, fps, w, h)
                diff = np.abs(frame(fx, 0, w, h) - frame(fx, frames, w, h)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")

    def test_crystals_turn_and_glint(self):
        """Crystals are really drawn with their arms (and turn), even in the small frames above, and glints come
        and go: the cases above cover them."""
        for params, (w, h) in ((self.CRYSTALS, (W, H)), ({**self.CRYSTALS, "size": 100}, (640, 360))):
            fx = effects.make("snow", params, 3, 144, 24, w, h)
            self.assertTrue((fx.crystal & (fx.ink_r == 0)).any(), params)
            self.assertTrue((fx.turns != 0).any())
            self.assertTrue((fx.glints > 0).any())
        soft = effects.make("snow", {"flake": "soft", "size": 100}, 3, 144, 24, 640, 360)
        self.assertFalse(soft.crystal.any())

    def test_gusts_never_blow_against_the_wind(self):
        """Between bursts the wind calms down; it never pushes the flakes back."""
        for wind in (10, -40):
            fx = effects.make("snow", {"wind": wind, "gust": 100, "sway": 0, "swirl": 0}, 3, 720, 24, W, H)
            xs = np.array([fx.at(t)[0] for t in range(721)])
            s = np.array([np.mod(fx.s0 + fx.k * fx.len * fx.tau(t), fx.len) for t in range(721)])
            step = np.diff(xs, axis=0) * np.sign(wind)
            step = step[np.diff(s, axis=0) > 0]  # not where a flake starts again at the top
            self.assertGreater(float(step.min()), 0.0, wind)

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

    def test_alike_at_every_size(self):
        """Sizes and paths are worked out in a 1080-line frame: the preview, the video and 4K have the same flakes
        in the same places (in frame heights), and the same ones are crystals."""
        for params in ({}, {"flake": "crystal"}, dict(snow.LOOKS["sparkle"]), {"size": 100},
                       dict(snow.LOOKS["blizzard"])):
            made = [effects.make("snow", params, 2, 144, 24, w, h) for w, h in ((640, 360), (1920, 1080), (3840, 2160))]
            first = made[0]
            for fx in made[1:]:
                self.assertTrue(np.array_equal(first.crystal, fx.crystal), params)
                (xa, ya, on_a), (xb, yb, on_b) = first.at(50), fx.at(50)
                self.assertTrue(np.array_equal(on_a, on_b), params)
                for a, b in ((xa, xb), (ya, yb)):
                    self.assertLess(float(np.abs(a / first.h - b / fx.h).max()), 1e-9, params)
        self.assertGreater(effects.make("snow", {"flake": "crystal"}, 2, 144, 24, 640, 360).crystal.mean(), 0.1)

    def test_natural_snow_is_soft_clumps(self):
        """In the natural style (the default, and the heavy and dreamy looks) crystals are rare, small, soft and
        near the camera: no field of crisp stars."""
        for params in ({}, {"size": 100}, dict(snow.LOOKS["heavy"]), dict(snow.LOOKS["dreamy"])):
            fx = effects.make("snow", params, 2, 144, 24, 1920, 1080)
            self.assertLess(float(fx.crystal.mean()), 0.01, params)
            if fx.crystal.any():
                self.assertTrue(fx.softc[fx.crystal].all())
                self.assertGreaterEqual(float(fx.z[fx.crystal].min()), 0.7)
                self.assertLessEqual(float(fx.tip[fx.crystal].max()), snow.RARE_TIP[1] + 1e-9)
        self.assertFalse(effects.make("snow", {"flake": "crystal"}, 2, 144, 24, 1920, 1080).softc.any())

    def test_out_of_focus_flakes_are_bright_soft_orbs(self):
        """At the lens a flake is a bright, soft orb, brightest in the middle (no darker middle, no ring) with a
        faint glow around it; still or moving."""
        for dx, dy in ((0.0, 0.0), (4.0, 30.0)):
            c = snow.flakes(*[np.array([v]) for v in (80.3, 80.6, 24.0, 10.0, 1.0, dx, dy)], 80, 80)[0]
            row = c[80, 80:]  # from the middle outwards
            self.assertTrue((np.diff(row) <= 1e-6).all(), (dx, dy))
            self.assertGreater(float(row[0]), 0.7)
            self.assertGreater(float(row[30]), 0.01)  # the glow, beyond the disc
        fx = effects.make("snow", {}, 2, 144, 24, 1920, 1080)
        self.assertGreater(float(np.median(fx.opacity[fx.near])), 0.6)

    def test_heavy_snow_and_blizzard(self):
        """Heavy snow and a blizzard have many more flakes, and a blizzard veils the scene in a whiteout that also
        shows over a bright sky."""
        made = {name: effects.make("snow", dict(snow.LOOKS[name]), 2, 144, 24, W, H) for name in snow.LOOKS}
        self.assertGreater(made["heavy"].n, 3 * made["medium"].n)
        self.assertGreater(made["blizzard"].n, 6 * made["medium"].n)
        for t in (10, 60):
            a = coverage(made["blizzard"], t)
            self.assertGreater(float(a.min()), 0.04)
            self.assertGreater(float(a.mean()), 0.15)
            sky = frame(made["blizzard"], t, base=0.72)
            self.assertGreater(float(sky.mean()), 0.75)  # whiter
            self.assertGreater(float(coverage(made["heavy"], t).mean()), 10 * float(coverage(made["medium"], t).mean()))

    def test_glints_show_by_default(self):
        """A few clear glints at a time, almost always; more of them up the slider, none at 0."""

        def glinting(twinkle):
            fx = effects.make("snow", {"twinkle": twinkle}, 2, 720, 24, 1920, 1080)
            return np.array([((fx.glints * np.maximum(np.sin(2 * math.pi * fx.q * fx.tau(t) + fx.qph), 0)
                               ** fx.SHARP) > 0.3).sum() for t in range(0, 720, 6)])

        bright = glinting(15)
        self.assertGreater(float((bright > 0).mean()), 0.9)
        self.assertGreater(float(bright.mean()), 2)
        counts = [float(glinting(tw).mean()) for tw in (0, 15, 40, 70, 100)]
        self.assertEqual(counts[0], 0.0)
        self.assertTrue(all(a * 1.5 < b for a, b in zip(counts[1:], counts[2:])), counts)

    def test_smear_keeps_the_ink(self):
        """A moving flake is spread along its way, as bright in all as when still."""
        for r in (0.6, 1.2, 3.0, 12.0):
            still = snow.flakes(*[np.array([v]) for v in (20.3, 20.6, r, 1.0, 0.0, 0.0, 0.0)], 40, 40)[0].sum()
            moving = snow.flakes(*[np.array([v]) for v in (20.3, 20.6, r, 1.0, 0.0, 3.0, 9.0)], 40, 40)[0].sum()
            self.assertLess(abs(moving / still - 1), 0.15, r)

    def test_haze(self):
        fx = effects.make("snow", {"haze": 100}, 2, 144, 24, W, H)
        a = coverage(fx, 10)
        self.assertGreater(float(a.min()), 0.05)
        self.assertGreater(float(a[:20].mean()), float(a[-20:].mean()))  # thicker towards the top

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
                       {"spin": 100}, {"far_color": "#000000"}, {"look": "blizzard"}, {"haze": 100}):
            other = effects.make("snow", change, 4, 144, 24, W, H)
            ox, oy, _ = other.at(37)
            self.assertTrue(np.array_equal(x, ox) and np.array_equal(y, oy), change)
        nearer = effects.make("snow", {"near": 100}, 4, 144, 24, W, H)
        nx, ny, _ = nearer.at(37)
        self.assertTrue(np.array_equal(x[~base.near], nx[~nearer.near]))
        for amount in (70, 100):  # (from 60 up, many more far flakes)
            more = effects.make("snow", {"amount": amount}, 4, 144, 24, W, H)
            mx, my, _ = more.at(37)
            here = {(round(a, 3), round(b, 3)) for a, b in zip(mx, my)}
            self.assertTrue(all((round(a, 3), round(b, 3)) in here for a, b in zip(x, y)), amount)
            x, y = mx, my

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
