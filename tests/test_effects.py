import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects

W, H = 320, 180


KINDS = list(effects.KINDS)
_yy, _xx = np.mgrid[0:H, 0:W]
SCENE = np.dstack([0.2 + 0.5 * (((_xx // 16) + (_yy // 16)) % 2), 0.3 + 0.4 * _xx / W, 0.25 + 0.5 * _yy / H]).astype(
    np.float32)  # something to blur and see through (a lens effect changes the picture under it)


def frame(fx, t, base=0.3, mask=None):
    pic = np.full((H, W, 3), base, np.float32) if base is not None else SCENE.copy()
    if fx.lens:
        fx.apply(t, pic, mask, 1.0)
        return pic
    canvas = effects.Canvas(W, H)
    fx.draw(t, canvas, mask)
    canvas.flush(pic, fx.blend)
    return pic


class Periodic(unittest.TestCase):
    """Frame P must be frame 0 again, whatever the loop length, frame rate and settings."""

    CASES = [{}, {"wind": -100, "speed": 100, "amount": 100}, {"wind": 100, "speed": 0, "sway": 100},
             {"depth": 0, "blur": 100, "size": 100}]

    def test_every_kind_returns_to_frame_zero(self):
        for kind in KINDS:
            for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
                for params in self.CASES + ([{"style": "dust"}] if kind == "glow" else [{"style": "petal"}]):
                    if kind in ("snow", "glow") or frames != 720:
                        fx = effects.make(kind, params, 3, frames, fps, W, H)
                        diff = np.abs(frame(fx, 0, None) - frame(fx, frames, None)).max()
                        self.assertLess(diff, 1 / 512, f"{kind} {params} P={frames} fps={fps}")

    def test_moves_between(self):
        for kind in KINDS:
            fx = effects.make(kind, {"amount": 80}, 3, 144, 24, W, H)
            if kind != "glass" or getattr(fx, "moves", True):  # (fogged glass alone does not move)
                self.assertGreater(np.abs(frame(fx, 0, None) - frame(fx, 36, None)).max(), 0.05, kind)

    def test_same_seed_same_picture(self):
        a = effects.make("snow", {}, 9, 144, 24, W, H)
        b = effects.make("snow", {}, 9, 144, 24, W, H)
        c = effects.make("snow", {}, 10, 144, 24, W, H)
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))
        self.assertFalse(np.array_equal(frame(a, 17), frame(c, 17)))

    def test_nothing_when_amount_is_zero(self):
        for kind in ("snow", "leaves", "rain", "glow"):  # (water on glass: its haze stays)
            fx = effects.make(kind, {"amount": 0}, 3, 144, 24, W, H)
            self.assertTrue(np.array_equal(frame(fx, 5), np.full((H, W, 3), 0.3, np.float32)), kind)

    def test_every_kind_keeps_out_of_its_area(self):
        """Outside the clip's area (mask 0) the picture stays exactly as it was, for every kind."""
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        for kind in KINDS:
            fx = effects.make(kind, {"amount": 100}, 3, 144, 24, W, H)
            pic = frame(fx, 7, None, mask)
            self.assertLess(float(np.abs(pic[:, :W // 2] - SCENE[:, :W // 2]).max()), 1e-5, kind)  # (float rounding)

    def test_mask_limits_the_effect(self):
        fx = effects.make("snow", {"amount": 100, "size": 100}, 3, 144, 24, W, H)
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        canvas = effects.Canvas(W, H)
        pic = np.full((H, W, 3), 0.3, np.float32)
        fx.draw(5, canvas, mask)
        canvas.flush(pic, fx.blend)
        self.assertTrue(np.array_equal(pic[:, :W // 2], np.full((H, W // 2, 3), 0.3, np.float32)))
        self.assertGreater(pic[:, W // 2:].max(), 0.5)


class Params(unittest.TestCase):
    def test_clean_params(self):
        p = effects.clean_params("snow", {"amount": 1e9, "wind": "-500", "color": "red", "junk": 1})
        self.assertEqual(p["amount"], 100)
        self.assertEqual(p["wind"], -100)
        self.assertEqual(p["color"], "#ffffff")
        self.assertNotIn("junk", p)
        p = effects.clean_params("leaves", {"style": "../../etc"})
        self.assertEqual(p["style"], "autumn")
        p = effects.clean_params("file", {"fade": "1.26", "start": float("nan")})
        self.assertEqual((p["fade"], p["start"]), (1.3, 0))

    def test_place_and_size_finely(self):
        """A clip is moved about a pixel at a time and sized in tenths of a percent, not whole percents."""
        p = effects.clean_params("file", {"x": 20.45, "y": -7.333, "scale": 87.66})
        self.assertEqual((p["x"], p["y"], p["scale"]), (20.45, -7.35, 87.7))


if __name__ == "__main__":
    unittest.main()
