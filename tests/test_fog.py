"""Drifting fog (fx/fog.py): banks of fog drifting with the wind, in a band of the frame, lit by the scene."""
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects  # (first: it loads every effect's module)
from fx import fog

W, H = 480, 270
MAX = {p["key"]: p["max"] for p in fog.SCHEMA["params"] if p["type"] == "range"}


def make(params=None, frames=240, fps=24, w=W, h=H, seed=3, light=None):
    return effects.make("fog", {**fog.DEFAULTS, **(params or {})}, seed, frames, fps, w, h, light=light)


def drawn(fx, t, mask=None):
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, mask)
    return canvas.rgb.copy(), canvas.a.copy()


class Loop(unittest.TestCase):
    def test_frame_p_is_frame_0(self):
        cases = [dict(fog.LOOKS[k], look=k) for k in fog.LOOKS] + [
            MAX, {"wind": -100, "swirl": 0, "depth": 0}, {"wind": 0}, {"swirl": 100, "patchy": 100, "size": 0},
            {"even": 100, "top": 0}]
        for frames, fps in ((240, 24), (150, 25), (90, 30), (1440, 24)):
            for params in cases:
                fx = make(params, frames, fps)
                for a, b in zip(drawn(fx, 0), drawn(fx, frames)):
                    self.assertLess(float(np.abs(a - b).max()), 1e-6, (frames, fps, params))

    def test_no_jumps(self):
        """From one frame to the next the fog changes about as much as from any other to the next: no frame unlike the
        rest, also where one copy of a layer gives way to the other (each copy starts again where its weight is
        nothing); with no wind it only changes slowly (its fine wisps do not boil: the flow that bends them turns only a
        little in a loop)."""
        for params in ({}, {"wind": 100, "swirl": 100}, {"wind": -60, "patchy": 90}, {"wind": 0}):
            fx = make(params, 720, 24)
            steps = [float(np.abs(fx.coverage(t / 720) - fx.coverage((t + 1) / 720)).max()) for t in range(0, 720, 2)]
            self.assertLess(max(steps), 2.5 * float(np.median(steps)), params)
            if params == {"wind": 0}:
                self.assertLess(max(steps), 0.06)  # (a flow turning fast stirs them by a third a frame)

    def test_drifts_with_the_wind(self):
        """With the wind the fog moves its way (what moves right: brighter where it was darker just to its left);
        with no wind it only changes shape in place."""
        def flow(wind):
            fx = make({"wind": wind, "top": 0, "lower": 0, "patchy": 30}, 720, 24)
            total = 0.0
            for t in range(0, 720, 24):
                a, b = fx.coverage(t / 720), fx.coverage((t + 2) / 720)
                total += float(((b - a) * np.gradient(a, axis=1)).sum())
            return total
        right, left, still = flow(80), flow(-80), flow(0)
        self.assertLess(right, 0)  # (C(x - vt): dC/dt = -v dC/dx)
        self.assertGreater(left, 0)
        self.assertLess(abs(still), 0.25 * min(abs(right), abs(left)))


class Picture(unittest.TestCase):
    def test_in_its_band(self):
        """Nothing above its top edge (beyond the fade and the billows), nothing below a bottom edge above the
        frame's; denser toward the bottom when asked (ground fog)."""
        fx = make({"top": 40, "bottom": 75, "soft": 20, "lower": 0})
        rows = (np.arange(H) + 0.5) / H
        for t in (0, 77, 160):
            _, a = drawn(fx, t)
            self.assertLess(float(a[rows < 0.40 - 0.1 * 1.8].max()), 1e-3)
            self.assertLess(float(a[rows > 0.75 + 0.1 * 1.8].max()), 1e-3)
            self.assertGreater(float(a[(rows > 0.45) & (rows < 0.7)].mean()), 0.1)
        ground = make({"top": 20, "lower": 100})
        _, a = drawn(ground, 50)
        self.assertGreater(float(a[rows > 0.85].mean()), 1.5 * float(a[(rows > 0.25) & (rows < 0.4)].mean()))

    def test_patchy(self):
        """"Khe hở": an even veil at the left, separate banks with clear air between them at the right."""
        clear = []
        for patchy in (0, 100):
            fx = make({"patchy": patchy, "top": 0, "lower": 0})
            clear.append(float((fx.coverage(0.3) < 0.02).mean()))
        self.assertLess(clear[0], 0.02)
        self.assertGreater(clear[1], 0.15)

    def test_even(self):
        """"Đều khắp khung": as much fog, its swings much smaller (no clear air between banks), yet still drifting."""
        base = {"top": 0, "lower": 0, "patchy": 0, "amount": 40, "opacity": 60}
        uneven, even = make({**base, "even": 0}), make({**base, "even": 100})
        a, b = uneven.coverage(0.3), even.coverage(0.3)
        self.assertLess(abs(float(b.mean()) - float(a.mean())), 0.1 * float(a.mean()))
        self.assertLess(float(b.std() / b.mean()), 0.2 * float(a.std() / a.mean()))
        self.assertGreater(float(b.min()), 0.8 * float(b.mean()))
        self.assertGreater(float(np.abs(even.coverage(0.3 + 1 / 24) - b).max()), 0.0)

    def test_haze_over_the_whole_frame(self):
        """The light even haze look covers the whole frame, top to bottom, lightly; nowhere much lighter along the top
        edge (fog above the frame shades it as fog in the frame does)."""
        fx = make(dict(fog.LOOKS["haze"]))
        c = fx.coverage(0.4)
        rows = c.mean(axis=1)
        self.assertGreater(float(c.min()), 0.15)
        self.assertLess(float(c.max()), 0.6)
        self.assertLess(float(np.ptp(rows)), 0.2 * float(rows.mean()))
        for params in ({"top": 0, "lower": 0, "patchy": 0, "amount": 100}, dict(fog.LOOKS["haze"])):
            fx = make(params)
            shade = fx._shade(fx.depth_at(0.4)).mean(axis=1)
            self.assertLess(abs(float(shade[:4].mean()) - float(shade[40:120].mean())), 0.02, params)

    def test_how_thick(self):
        thin, thick = (float(make({"amount": a, "top": 0}).coverage(0.2).mean()) for a in (20, 90))
        self.assertGreater(thick, 2 * thin)
        self.assertLessEqual(float(make({"amount": 100, "opacity": 60, "top": 0}).coverage(0.2).max()), 0.6 + 1e-6)
        rgb, a = drawn(make({"amount": 0}), 10)
        self.assertEqual(float(a.max()), 0.0)

    def test_alike_at_every_size(self):
        """Worked out on a grid of frame heights: the preview, the video and 4K show the same fog."""
        small, big = make(w=640, h=360), make(w=3840, h=2160)
        self.assertTrue(np.array_equal(small.coverage(0.4), big.coverage(0.4)))

    def test_its_own_colour_or_the_scene_light(self):
        """Without the scene's light it is its own colour; lit by the scene it glows near a light and is darker in a
        dark place."""
        rgb, a = drawn(make({"top": 0, "lower": 0, "patchy": 0, "amount": 100, "opacity": 100}), 10)
        own = effects._rgb(fog.SCHEMA["params"][-1]["value"])
        hit = a > 0.5
        seen = rgb[hit] / a[hit][:, None]  # (its own colour, a little darker under fog above it: lit from above)
        shade = seen / own
        self.assertLess(float(shade.max()), 1.01)
        self.assertGreater(float(shade.min()), 1 - fog.SHADE - 0.02)
        self.assertLess(float(np.abs(shade - shade.mean(axis=1, keepdims=True)).max()), 0.01)  # (its own hue)
        self.assertGreater(float(np.ptp(shade[:, 0])), 0.05)  # (a body: some of it in its own shade)
        scene = np.full((H, W, 3), 12, np.uint8)
        scene[100:140, 300:330] = 255  # (a lamp, right of the middle)
        light = effects.light_map([scene])
        fx = make({"top": 0, "lower": 0, "patchy": 0, "amount": 100, "opacity": 100, "lit": 100}, light=light)
        rgb, a = drawn(fx, 10)
        colour = rgb / np.maximum(a, 1e-6)[..., None]
        near, far = colour[110:130, 300:330].mean(), colour[110:130, 20:60].mean()
        self.assertGreater(near, far + 0.15)
        self.assertLess(far, float(own.mean()) - 0.2)

    def test_stays_in_its_area(self):
        fx = make({"top": 0})
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        _, a = drawn(fx, 30, mask)
        self.assertEqual(float(a[:, :W // 2].max()), 0.0)
        self.assertGreater(float(a[:, W // 2:].max()), 0.3)


class Settings(unittest.TestCase):
    def test_looks(self):
        """Each look sets every one of the fog's own settings but the colour, to values the page accepts."""
        look = next(p for p in fog.SCHEMA["params"] if p["key"] == "look")
        self.assertEqual({o["value"] for o in look["options"]}, set(fog.LOOKS))
        keys = set(fog.DEFAULTS)
        for name, preset in look["presets"].items():
            self.assertEqual(set(preset), keys, name)
            cleaned = effects.clean_params("fog", preset)
            self.assertEqual({k: cleaned[k] for k in keys}, preset, name)
        defaults = effects.defaults("fog")
        self.assertEqual({k: defaults[k] for k in keys}, fog.DEFAULTS)

    def test_every_setting_has_a_heading(self):
        for p in fog.SCHEMA["params"]:
            self.assertTrue(p.get("group"), p["key"])
        self.assertTrue(fog.SCHEMA["note"])

    def test_drawn_over_and_turns(self):
        cls = effects.KINDS["fog"]
        self.assertEqual((cls.blend, cls.lens, cls.BOX, cls.LIT, cls.TURNS), ("over", False, None, True, True))


if __name__ == "__main__":
    unittest.main()
