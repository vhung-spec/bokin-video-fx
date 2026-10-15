import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects

W, H = 320, 180


def frame(fx, t, base=0.3):
    canvas = effects.Canvas(W, H)
    pic = np.full((H, W, 3), base, np.float32)
    fx.draw(t, canvas, None)
    canvas.flush(pic, fx.blend)
    return pic


class Periodic(unittest.TestCase):
    """Frame P must be frame 0 again, whatever the loop length, frame rate and settings."""

    CASES = [{}, {"wind": -100, "speed": 100, "amount": 100}, {"wind": 100, "speed": 0, "sway": 100},
             {"depth": 0, "blur": 100, "size": 100}]

    def test_every_kind_returns_to_frame_zero(self):
        for kind in ("snow", "leaves", "rain", "glow"):
            for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
                for params in self.CASES + ([{"style": "dust"}] if kind == "glow" else [{"style": "petal"}]):
                    if kind in ("snow", "glow") or frames != 720:
                        fx = effects.make(kind, params, 3, frames, fps, W, H)
                        diff = np.abs(frame(fx, 0) - frame(fx, frames)).max()
                        self.assertLess(diff, 1 / 512, f"{kind} {params} P={frames} fps={fps}")

    def test_moves_between(self):
        for kind in ("snow", "leaves", "rain", "glow"):
            fx = effects.make(kind, {"amount": 80}, 3, 144, 24, W, H)
            self.assertGreater(np.abs(frame(fx, 0) - frame(fx, 36)).max(), 0.05, kind)

    def test_same_seed_same_picture(self):
        a = effects.make("snow", {}, 9, 144, 24, W, H)
        b = effects.make("snow", {}, 9, 144, 24, W, H)
        c = effects.make("snow", {}, 10, 144, 24, W, H)
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))
        self.assertFalse(np.array_equal(frame(a, 17), frame(c, 17)))

    def test_nothing_when_amount_is_zero(self):
        for kind in ("snow", "leaves", "rain", "glow"):
            fx = effects.make(kind, {"amount": 0}, 3, 144, 24, W, H)
            self.assertTrue(np.array_equal(frame(fx, 5), np.full((H, W, 3), 0.3, np.float32)), kind)

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


if __name__ == "__main__":
    unittest.main()
