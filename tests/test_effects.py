import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects

W, H = 320, 180


KINDS = list(effects.KINDS)
_yy, _xx = np.mgrid[0:H, 0:W]
SCENE = np.dstack([0.2 + 0.5 * (((_xx // 16) + (_yy // 16)) % 2), 0.3 + 0.4 * _xx / W, 0.25 + 0.5 * _yy / H]).astype(
    np.float32)  # something to blur and see through (a lens effect changes the picture under it)
# an area painted over most of the frame, for the effects that work only inside one (Effect.AREA: rustling leaves)
AREA = (((_xx - W * 0.45) / (W * 0.42)) ** 2 + ((_yy - H * 0.55) / (H * 0.42)) ** 2 < 1).astype(np.float32)
# what an effect needs drawn by hand to show anything (Dây đèn tự vẽ: its strands; Khói ống khói: a chimney; one
# only: a frame on its own costs about a second a chimney, its smoke made from the start; several in test_chimney)
DRAWN = {"garland": {"strands": [[0.1, 0.3, 1, 0.4], [0.5, 0.35, 0, 0], [0.9, 0.3, 0, 0], [0.2, 0.8, 1, 0],
                                 [0.3, 0.7, 0.5, 0], [0.4, 0.75, 0.5, 0]]},
         "chimney": {"stacks": [[0.3, 0.85, 0.05, 0.5]]}}
MORE = {"hearth": [{"wood": "oak", "stack": "heap", "embers": 100}]}  # (more of an effect's own: its logs)


def frame(fx, t, base=0.3, mask=None):
    pic = np.full((H, W, 3), base, np.float32) if base is not None else SCENE.copy()
    if fx.lens:
        fx.apply(t, pic, AREA if mask is None and fx.AREA else mask, 1.0)
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
                for params in self.CASES + ([{"style": "dust"}] if kind == "glow" else [{"style": "petal"}]) + \
                        MORE.get(kind, []):
                    if kind in ("snow", "glow") or frames != 720:
                        fx = effects.make(kind, dict(params, **DRAWN.get(kind, {})), 3, frames, fps, W, H)
                        diff = np.abs(frame(fx, 0, None) - frame(fx, frames, None)).max()
                        self.assertLess(diff, 1 / 512, f"{kind} {params} P={frames} fps={fps}")

    def test_every_kind_turned_returns_to_frame_zero(self):
        """Placed and turned in 3D (drawn on a sheet, warped onto the frame: sheet.Sheet), frame P is frame 0 again:
        the turn does not move. (The candle does not turn.)"""
        from PIL import Image
        from fx import sheet
        img = Image.fromarray((SCENE * 255 + 0.5).astype(np.uint8))
        for kind in KINDS:
            if not effects.KINDS[kind].TURNS:
                continue
            for frames, fps in ((144, 24), (150, 25)):
                for params in ({"rot_x": 20, "rot_y": -35, "rot_z": 10, "persp": 70},
                               {"rot_z": -60, "scale": 60, "x": 15, "amount": 100, "wind": 80}):
                    fx = sheet.Sheet(kind, params, 3, frames, fps, W, H)
                    a, b = (np.asarray(fx.draw(t, img.copy())).astype(int) for t in (0, frames))
                    self.assertLessEqual(int(np.abs(a - b).max()), 1, f"{kind} {params} P={frames} fps={fps}")

    def test_moves_between(self):
        for kind in KINDS:
            fx = effects.make(kind, {"amount": 80, **DRAWN.get(kind, {})}, 3, 144, 24, W, H)
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

    def test_an_area_effect_needs_its_area(self):
        """An effect that works only inside the clip's area (rustling leaves) changes nothing without one: never the
        whole picture."""
        for kind, cls in effects.KINDS.items():
            if cls.AREA:
                fx = effects.make(kind, {"amount": 100}, 3, 144, 24, W, H)
                pic = SCENE.copy()
                fx.apply(7, pic, None, 1.0)
                self.assertTrue(np.array_equal(pic, SCENE), kind)
                fx.apply(7, pic, AREA, 1.0)
                self.assertFalse(np.array_equal(pic, SCENE), kind)

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
        p = effects.clean_params("leaves", {"shape": "../../etc", "season": None, "red": "x"})
        self.assertEqual((p["maple"], p["oak"], p["scene"], p["season"], p["red"]), (True, False, "maple", "mixed", 40))
        self.assertNotIn("shape", p)
        p = effects.clean_params("leaves", {"style": "../../etc"})  # (an old clip's: its leaves, in autumn colours)
        self.assertEqual((p["maple"], p["birch"], p["season"]), (True, True, "redgold"))
        p = effects.clean_params("leaves", {"scene": "../../etc", "oak": "yes", "petal": None})
        self.assertEqual((p["scene"], p["oak"], p["petal"], p["maple"]), ("maple", False, False, True))
        p = effects.clean_params("file", {"fade": "1.26", "start": float("nan")})
        self.assertEqual((p["fade"], p["start"]), (1.3, 0))

    def test_place_and_size_finely(self):
        """A clip is moved about a pixel at a time and sized in tenths of a percent, not whole percents."""
        p = effects.clean_params("file", {"x": 20.45, "y": -7.333, "scale": 87.66})
        self.assertEqual((p["x"], p["y"], p["scale"]), (20.45, -7.35, 87.7))


if __name__ == "__main__":
    unittest.main()


class Placed(unittest.TestCase):
    def test_box_and_its_preview_clip(self):
        """A placed effect lies in its box as a file clip with fit "contain" is placed; its preview clip is the box
        alone, the same wherever the box is."""
        from fx import engine

        class Dot(effects.Effect):
            BOX = (1, 2)

            def draw(self, t, canvas, mask):
                x0, y0, w, h = self.box()
                canvas.over(round(x0), round(y0), np.ones((round(h), round(w)), np.float32), np.ones(3, np.float32),
                            1.0, mask)
        effects.KINDS["dottest"] = Dot
        effects.SCHEMA["dottest"] = {"label": "test", "params": effects._place(10)}
        try:
            fx = effects.make("dottest", {"scale": 10, "x": 20, "y": -10}, 1, 48, 24, 1920, 1080)
            x0, y0, w, h = fx.box()
            self.assertAlmostEqual(h, 108)
            self.assertAlmostEqual(w, 54)
            self.assertAlmostEqual(x0, (1920 - 54) / 2 + 384)
            self.assertAlmostEqual(y0, (1080 - 108) / 2 - 108)
            info = {"w": 1920, "h": 1080, "fps": "24", "frames": 48, "seconds": 2.0}
            self.assertEqual(engine.fx_clip_frame(info, "dottest"), (320, 640))
            project = {"video": info, "length": 30}
            a = {"kind": "dottest", "params": {"scale": 10, "x": 20, "y": -10}}
            b = {"kind": "dottest", "params": {"scale": 30, "x": -5, "y": 0}}
            self.assertEqual(engine.fx_clip_name(project, a), engine.fx_clip_name(project, b))
            whole = effects.make("dottest", engine.fx_clip_params("dottest", a["params"]), 1, 48, 24, 320, 640)
            self.assertEqual(tuple(round(v) for v in whole.box()), (0, 0, 320, 640))  # the clip is the box
        finally:
            del effects.KINDS["dottest"], effects.SCHEMA["dottest"]
