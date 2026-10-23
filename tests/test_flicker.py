"""A painted area flickering (fx/flicker.py): a lens inside the clip's area only; a flame round the picture's own
light, a lamp or a sign going off."""
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


def sign():
    """A sign at night: a brick wall (its bricks and mortar), lit letters burnt out to white, their glow round them."""
    pic = np.zeros((H, W, 3), np.float32)
    pic[:] = (0.07, 0.05, 0.06)
    pic[(_yy % 12 < 2) | (_xx % 30 < 2)] = (0.04, 0.03, 0.035)
    glow = np.exp(-((_xx - 160) / 70.0) ** 2 - ((_yy - 90) / 30.0) ** 2)[..., None]
    lin = pic ** 2 + glow * np.array([0.5, 0.08, 0.3], np.float32)
    letters = (np.abs(_yy - 90) < 12) & (np.abs(_xx - 160) < 60) & (_xx % 14 < 4)
    lin[letters] = 1.0
    return np.sqrt(np.clip(lin, 0, 1)).astype(np.float32), letters, glow[..., 0]


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
                self.assertGreater(float(np.ptp(levels)), 0.99, mode)  # (it does flicker)

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
        """On as in the video most of the time; now and then a burst of quick flickers right off."""
        fx = make({"mode": "faulty"}, 720, 24)
        levels = fx.level((np.arange(7200) + 0.5) / 7200)
        self.assertGreater(float((levels > -0.02).mean()), 0.8)
        self.assertLess(float(levels.min()), -0.98)
        self.assertLessEqual(float(levels.max()), 0.0)
        off = np.diff((levels < -0.5).astype(int))
        self.assertGreaterEqual(int((off == 1).sum()), 4)  # (a burst: several flickers)

    def test_clearly_seen(self):
        """At its first settings a flame's light swings a fifth or so either way where it lights (lit places brighter
        and dimmer than in the video), while the dark places of the area stay as they are (painted wide: no outline)."""
        fx = make({"edge": 0})
        right, left = (AREA > 0) & (_xx > 170), (AREA > 0) & (_xx < 150)
        mean = np.array([frame(fx, t)[right].mean() for t in range(144)])
        base = scene()[right].mean()
        self.assertGreater(float(mean.std() / base), 0.1)
        self.assertGreater(float(mean.max() / base), 1.15)
        self.assertLess(float(mean.min() / base), 0.8)
        for t in (0, 40, 90):
            self.assertLess(float(np.abs(frame(fx, t)[left] - scene()[left]).max()), 1e-6)

    def test_a_burnt_out_highlight_shows_its_colour_dimmed(self):
        """A carved pumpkin's opening burnt out to yellow (its red at white) dimmed: oranger (its red holds more than
        its green, as a dimmer candle's light), not a darker yellow."""
        pic = np.zeros((H, W, 3), np.float32)
        pic[:] = (1.0, 0.84, 0.43)
        fx = make({"edge": 0, "warm": 0})
        t = int(np.argmin(fx.level((np.arange(144) + 0.5) / 144)))
        fx.apply(t, pic, AREA, 1.0)
        r, g, b = (float(v) for v in pic[90, 160])
        self.assertLess(g, 0.6)
        self.assertGreater(r / g, 1.2 * (1.0 / 0.84))

    def test_signs_go_off(self):
        """A sign blinking (at the settings its way sets): off, its letters and their glow go as dark as the wall
        behind them; on, it is as in the video; the wall, bricks and all, stays as it is."""
        pic, letters, glow = sign()
        p = dict(effects.defaults("flicker"), mode="pulse", edge=0)
        p.update(flicker.MODES["pulse"]["preset"])
        fx = make(p)
        levels = fx.level((np.arange(144) + 0.5) / 144)
        off, on = int(np.argmin(levels)), int(np.argmax(levels))
        dark, lit = pic.copy(), pic.copy()
        fx.apply(off, dark, AREA, 1.0)
        fx.apply(on, lit, AREA, 1.0)
        self.assertLess(float(np.abs(lit - pic).max()), 0.01)
        wall = (AREA > 0) & (glow < 0.01)
        self.assertLess(float(np.abs(dark[wall] - pic[wall]).max()), 0.01)  # (as it was, within 2 levels of 255)
        self.assertLess(float(dark[letters].mean()), 0.3)  # (unlit glass)
        haze = (AREA > 0) & (glow > 0.3) & ~letters
        self.assertLess(float(dark[haze].mean()) - float(pic[wall].mean()), 0.03)  # (its glow gone)

    def test_lit_bit_by_bit(self):
        """A sign lit along its length (left to right by default; or right to left, top to bottom, bottom to top):
        while it is being lit, its start is on and its end still off; then all of it on; it ends off."""
        for way, start, end in (("right", (90, 75), (90, 245)), ("left", (90, 245), (90, 75)),
                                ("down", (35, 160), (145, 160)), ("up", (145, 160), (35, 160))):
            fx = make({"mode": "sweep", "way": way, "edge": 0, "amount": 100})
            ph = ((np.arange(1440) + 0.5) / 1440 * fx.m + fx.ph / flicker.TAU) % 1.0
            during = int(np.argmin(np.abs(ph - flicker.WIPE / 2))) / 10
            held = int(np.argmin(np.abs(ph - (flicker.WIPE + flicker.HOLD) / 2))) / 10
            gone = int(np.argmin(np.abs(ph - (flicker.DOWN + 1) / 2))) / 10
            pic = np.repeat(np.where(AREA > 0, 0.8, 0.15)[..., None], 3, 2).astype(np.float32)  # (a lit sign on a
            # darker wall)
            a, b, c = pic.copy(), pic.copy(), pic.copy()
            fx.apply(during, a, AREA, 1.0)
            fx.apply(held, b, AREA, 1.0)
            fx.apply(gone, c, AREA, 1.0)
            self.assertGreater(float(a[start].mean()), 0.75, way)
            self.assertLess(float(a[end].mean()), 0.2, way)
            self.assertGreater(float(b[end].mean()), 0.75, way)
            self.assertLess(float(c[start].mean()), 0.2, way)

    def test_settings(self):
        cls = effects.KINDS["flicker"]
        self.assertTrue(cls.AREA and cls.lens and cls.UNDER)
        self.assertFalse(cls.TURNS)
        sch = effects.SCHEMA["flicker"]
        self.assertTrue(sch["area_tip"] and sch["area_empty"] and "Vẽ vùng" in sch["note"])
        self.assertFalse({"x", "y", "scale", "rot_x"} & {p["key"] for p in sch["params"]})
        params = {p["key"]: p for p in sch["params"]}
        self.assertEqual(params["way"]["show"], {"mode": ["sweep"]})  # (the way a sign is lit: only for that)
        presets = params["mode"]["presets"]
        self.assertEqual(set(presets), {o["value"] for o in params["mode"]["options"]})  # (every way: its own)
        for mode in ("pulse", "faulty", "sweep"):  # (a sign's own colour, quite dark when off)
            self.assertEqual(presets[mode]["warm"], 0)
            self.assertGreaterEqual(presets[mode]["amount"], 70)


if __name__ == "__main__":
    unittest.main()
