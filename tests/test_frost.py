"""Frost and snow on a window pane (fx/frost.py)."""
import unittest

import numpy as np
from PIL import Image

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, frost

LOOKS = effects.SCHEMA["frost"]["params"][0]["presets"]


def make(params=None, w=640, h=360, frames=144, fps=24, seed=3):
    return effects.make("frost", params or {}, seed, frames, fps, w, h)


def cover(fx, t):
    """How much of each pixel the frost hides at frame t (drawn over black and over white)."""
    out = []
    for base in (0.0, 1.0):
        pic = np.full((fx.h, fx.w, 3), base, np.float32)
        canvas = effects.Canvas(fx.w, fx.h)
        fx.draw(t, canvas, None)
        canvas.flush(pic, fx.blend)
        out.append(pic)
    return 1 - (out[1] - out[0]).mean(axis=2)


class Frost(unittest.TestCase):
    def test_every_look_loops(self):
        for look, params in LOOKS.items():
            for frames, fps in ((144, 24), (150, 25)):
                fx = make(dict(params, look=look, creep=100, melt=100), 320, 180, frames, fps)
                self.assertLess(float(np.abs(cover(fx, 0) - cover(fx, frames)).max()), 1 / 512, look)

    def test_at_the_edges(self):
        """Frost round the edges and in the corners, the middle of the pane clear."""
        c = cover(make({"amount": 0, "sparkle": 0}), 0)
        h, w = c.shape
        self.assertGreater(c[:h // 12].mean(), 0.2)
        self.assertGreater(c[:, :w // 20].mean(), 0.2)
        self.assertLess(c[h // 3:2 * h // 3, w // 3:2 * w // 3].mean(), 0.02)
        thick = cover(make({"cover": 100, "amount": 0, "sparkle": 0}), 0)
        self.assertGreater(thick.mean(), c.mean() + 0.1)

    def test_snow_on_the_bottom_edge(self):
        c = cover(make({"sill": 50, "amount": 0, "sparkle": 0, "opacity": 100}), 0)
        self.assertGreater(c[-3:].min(), 0.95)
        none = cover(make({"sill": 0, "cover": 0, "ferns": 0, "rime": 0, "veil": 0, "amount": 0, "sparkle": 0}), 0)
        self.assertLess(float(none.max()), 1e-6)

    def test_same_frost_at_any_size(self):
        """The ferns are made in frame heights: a small preview shows the frost of the video."""
        small, big = (cover(make({"amount": 0, "sparkle": 0, "creep": 0}, w, h), 0) for w, h in ((480, 270),
                                                                                              (1280, 720)))
        a, b = (np.asarray(Image.fromarray(c.astype(np.float32), "F").resize((64, 36), Image.Resampling.BOX))
                for c in (small, big))
        self.assertLess(float(np.abs(a - b).mean()), 0.03)

    def test_creeps_back_and_out(self):
        """Still when it does not creep; creeping, it is smallest halfway through the loop."""
        still = make({"creep": 0, "amount": 0, "sparkle": 0})
        self.assertTrue(np.array_equal(cover(still, 10), cover(still, 80)))
        moving = make({"creep": 100, "amount": 0, "sparkle": 0})
        self.assertLess(cover(moving, 72).sum(), 0.9 * cover(moving, 0).sum())

    def test_flakes_land_and_melt(self):
        """A flake lands at a new place each time, while it is unseen (no jump)."""
        fx = make({"amount": 100, "melt": 100}, frames=720, fps=24)
        self.assertGreater(fx.k_fl.max(), 1)
        i = int(np.argmax(fx.k_fl))
        self.assertFalse(np.array_equal(fx.places[i, 0], fx.places[i, 1]))
        for age in (0.0, 0.9999):
            land = frost._smooth(age / 0.04)
            melt = frost._smooth((age - fx.hold) / (0.86 - fx.hold))
            self.assertLess(float((land * (1 - melt)).max()), 1e-3)
        a, b = cover(fx, 0), cover(fx, 100)
        self.assertGreater(float(np.abs(a - b).max()), 0.3)

    def test_lit_by_the_scene(self):
        """Lit by the scene behind the glass: in front of a bright place the frost is bright, in front of a dark one
        dim (frosted glass shows the scene blurred); "lit" 0: its own colour, as without the scene's light."""
        w, h = 640, 360
        scene = np.zeros((h, w, 3), np.uint8)
        scene[:, : w // 2] = 230  # (bright on the left, dark on the right)
        light = effects.light_map([scene])
        p = {"amount": 0, "sparkle": 0, "sill": 0, "creep": 0, "ferns": 0, "opacity": 100}
        lit = effects.make("frost", p, 3, 144, 24, w, h, light)
        own = effects.make("frost", dict(p, lit=0), 3, 144, 24, w, h, light)
        plain = effects.make("frost", p, 3, 144, 24, w, h)
        for fx in (lit, own, plain):
            fx.draw(0, effects.Canvas(w, h), None)
        rgb, cover = lit._frost_at(0)
        colour = rgb.sum(axis=2) / np.maximum(cover, 1e-6) / 3
        edge = cover > 0.3
        left, right = edge.copy(), edge.copy()
        left[:, w // 4:] = False
        right[:, : 3 * w // 4] = False
        self.assertGreater(colour[left].mean(), colour[right].mean() + 0.3)
        self.assertTrue(np.allclose(own._frost_at(0)[0], plain._frost_at(0)[0], atol=1e-5))
        self.assertEqual(effects.KINDS["frost"].LIT, True)


if __name__ == "__main__":
    unittest.main()
