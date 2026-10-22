"""A painted area flickering (fx/flicker.py): a lens inside the clip's area only."""
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, flicker

W, H = 320, 180
MODES = list(flicker.MODES)
_yy, _xx = np.mgrid[0:H, 0:W]
AREA = (((_xx - 160) / 90.0) ** 2 + ((_yy - 90) / 70.0) ** 2 < 1).astype(np.float32)


def scene():
    """Dark on the left, bright and warm on the right."""
    pic = np.zeros((H, W, 3), np.float32)
    pic[:, : W // 2] = 0.08
    pic[:, W // 2:] = (0.9, 0.55, 0.25)
    return pic


def make(params=None, frames=144, fps=24):
    return effects.make("flicker", params or {}, 3, frames, fps, W, H)


def frame(fx, t, mask=AREA):
    pic = scene()
    fx.apply(t, pic, mask, 1.0)
    return pic


class Flicker(unittest.TestCase):
    def test_every_way_loops(self):
        for mode in MODES:
            for frames, fps in ((144, 24), (150, 25)):
                fx = make({"mode": mode, "speed": 100, "amount": 100}, frames, fps)
                self.assertLess(float(np.abs(frame(fx, 0) - frame(fx, frames)).max()), 1 / 512, mode)
                levels = fx.level((np.arange(frames) + 0.5) / frames)
                self.assertGreater(float(np.ptp(levels)), 1.0, mode)  # (it does flicker)

    def test_only_inside_the_area(self):
        fx = make({"amount": 100})
        self.assertTrue(np.array_equal(frame(fx, 7, None), scene()))  # (no area: nothing)
        changed = np.abs(frame(fx, 7) - scene()).max(axis=2)
        self.assertEqual(float(changed[AREA == 0].max()), 0.0)
        # the soft rim: little change right at the outline, the full change well inside
        rim = (AREA > 0) & (np.abs(np.hypot((_xx - 160) / 90.0, (_yy - 90) / 70.0) - 1) < 0.03)
        inside = np.hypot((_xx - 160) / 90.0, (_yy - 90) / 70.0) < 0.5
        t = int(np.argmax(np.abs(fx.level((np.arange(144) + 0.5) / 144))))
        changed = np.abs(frame(fx, t) - scene()).max(axis=2)
        self.assertLess(float(changed[rim].mean()), 0.3 * float(changed[inside & (_xx > 170)].mean()))

    def test_bright_parts_flicker_more_warmer_when_bright(self):
        fx = make({"amount": 100, "focus": 100, "warm": 100, "edge": 0})
        levels = fx.level((np.arange(144) + 0.5) / 144)
        dim, bright = int(np.argmin(levels)), int(np.argmax(levels))
        a, b = frame(fx, dim), frame(fx, bright)
        left, right = (AREA > 0) & (_xx < 150), (AREA > 0) & (_xx > 170)
        swing = np.abs(b - a).mean(axis=2)
        self.assertGreater(float(swing[right].mean()), 5 * float(swing[left].mean()))
        base = scene()
        r, bl = a[..., 0][right].mean() / base[..., 0][right].mean(), a[..., 2][right].mean() / base[..., 2][right].mean()
        self.assertLess(bl, r)  # (dimmer: redder, blue drops most)

    def test_a_faulty_lamp_is_mostly_steady(self):
        fx = make({"mode": "faulty"})
        levels = fx.level((np.arange(2000) + 0.5) / 2000)
        self.assertGreater(float((levels > 0.8).mean()), 0.6)
        self.assertLess(float(levels.min()), -0.8)

    def test_settings(self):
        cls = effects.KINDS["flicker"]
        self.assertTrue(cls.AREA and cls.lens and cls.UNDER)
        self.assertFalse(cls.TURNS)
        sch = effects.SCHEMA["flicker"]
        self.assertTrue(sch["area_tip"] and sch["area_empty"] and "Vẽ vùng" in sch["note"])
        self.assertFalse({"x", "y", "scale", "rot_x"} & {p["key"] for p in sch["params"]})


if __name__ == "__main__":
    unittest.main()
