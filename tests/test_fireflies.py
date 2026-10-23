"""Fireflies (fx/fireflies.py): a placed effect of small lights wandering and flashing in their box."""
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects  # (first: it loads every effect's module)
from fx import engine, fireflies, overlay

W, H = 480, 270
AT = {"scale": 90, "x": 0, "y": 0}


def make(params=None, frames=240, fps=24, w=W, h=H, seed=3):
    return effects.make("fireflies", {**AT, **(params or {})}, seed, frames, fps, w, h)


def light(fx, t, w=None, h=None):
    canvas = effects.Canvas(w or fx.w, h or fx.h)
    fx.draw(t, canvas, None)
    return canvas.rgb.copy()


class Loop(unittest.TestCase):
    def test_frame_p_is_frame_0(self):
        for frames, fps in ((240, 24), (150, 25), (90, 30)):
            for params in ({}, {"amount": 100, "speed": 100, "blink": 100, "wander": 100}):
                fx = make(params, frames, fps)
                self.assertLess(float(np.abs(light(fx, 0) - light(fx, frames)).max()), 1e-6, (frames, fps, params))

    def test_they_wander_slowly(self):
        """Each moves smoothly, a little each frame, and gets about over the loop."""
        fx = make({}, 720, 24)
        _, _, _, h = fx.box()
        xs, ys = zip(*(fx.where(t / 720) for t in range(0, 720, 2)))
        x, y = np.array(xs) * h, np.array(ys) * h  # (pixels: steps of 2 frames)
        step = np.hypot(np.diff(x, axis=0), np.diff(y, axis=0))
        self.assertLess(float(step.max()), 0.02 * h)  # (no jumps)
        self.assertGreater(float(np.ptp(x, axis=0).mean()), 0.05 * h)  # (they get about)

    def test_flashes(self):
        """A firefly's flash comes up quickly and fades more slowly; with "Nhấp nháy" 100 it is dark between
        flashes, with 0 it glows steadily."""
        fx = make({"twinkle": 100}, 720, 24)
        tau = np.arange(720 * 4) / (720 * 4)
        lit = np.array([fx.flashes(t) for t in tau])  # (time x fireflies)
        self.assertLess(float(lit.min()), 1e-3)
        self.assertGreater(float(lit.max(axis=0).min()), 0.99)  # (each flashes fully)
        i = 0
        f = lit[:, i]
        peak = int(np.argmax(f))
        up = np.flatnonzero(f[:peak + 1] < 0.1)
        rise = peak - (int(up[-1]) if len(up) else 0)
        after = np.flatnonzero(f[peak:] < 0.1)
        fall = int(after[0]) if len(after) else len(f) - peak
        self.assertLess(rise, fall)
        steady = make({"twinkle": 0}, 720, 24)
        self.assertLess(float(np.ptp([steady.flashes(t) for t in tau[::40]])), 1e-9)

    def test_moves(self):
        fx = make()
        self.assertGreater(float(np.abs(light(fx, 0) - light(fx, 12)).max()), 0.05)


class Picture(unittest.TestCase):
    def test_only_in_its_box(self):
        """Their light stays in their box (they fade out toward its edges: nothing is cut)."""
        for params in ({}, {"amount": 100, "size": 100, "depth": 100, "wander": 100}, {"x": 40, "scale": 40}):
            fx = make(params)
            x0, y0, w, h = fx.box()
            outside = np.ones((H, W), bool)
            outside[max(0, int(y0)):int(np.ceil(y0 + h)), max(0, int(x0)):int(np.ceil(x0 + w))] = False
            for t in (0, 50, 133):
                rgb = light(fx, t)
                self.assertLess(float(rgb[outside].max(initial=0.0)), 2 / 255, (params, t))

    def test_how_many(self):
        self.assertEqual(make({"amount": 0}).n, 0)
        self.assertTrue(np.array_equal(light(make({"amount": 0}), 5), np.zeros((H, W, 3), np.float32)))
        counts = [make({"amount": a}).n for a in (10, 35, 80, 100)]
        self.assertEqual(counts, sorted(counts))
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[-1], fireflies.count(100))

    def test_their_colour(self):
        """Lime yellow-green light by default; warm yellow, green, or a colour chosen."""
        for tone, want in (("lime", (0.78, 1.0, 0.31)), ("warm", (1.0, 0.88, 0.4))):
            fx = make({"tone": tone, **effects.SCHEMA["fireflies"]["params"][1]["presets"][tone], "amount": 60})
            rgb = sum(light(fx, t) for t in (10, 40, 70))
            total = rgb.reshape(-1, 3).sum(axis=0)
            got = total / total.max()
            self.assertLess(float(np.abs(got - np.array(want)).max()), 0.15, (tone, got))

    def test_near_ones_bigger_and_softer(self):
        """"Gần xa": with depth some are near (bigger, dimmer at the middle); with none, all alike."""
        flat, deep = make({"depth": 0}), make({"depth": 100})
        self.assertLess(float(np.ptp(flat.core / flat.core.min())), 0.7)
        self.assertGreater(float(deep.core.max() / deep.core.min()), 2.0)
        near = int(np.argmax(deep.core))
        self.assertLess(deep.glow[near] / deep.core[near], np.median(deep.glow / deep.core))


class Preview(unittest.TestCase):
    def test_preview_clip_is_the_box_alone(self):
        """The preview clip (the box alone, at the clip's size) is the box in the whole frame, at another size."""
        from PIL import Image
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 240}
        layer = {"kind": "fireflies", "params": dict(AT)}
        cw, ch = engine.fx_clip_frame(info, "fireflies", layer)
        self.assertAlmostEqual(cw / ch, 16 / 9, places=1)
        whole = make({}, 240, 24, 1920, 1080)
        alone = effects.make("fireflies", engine.fx_clip_params("fireflies", AT), 3, 240, 24, cw, ch)
        x0, y0, bw, bh = whole.box()
        for t in (0, 77):
            clip = np.minimum(light(alone, t), 1)
            big = np.minimum(light(whole, t), 1)
            part = np.asarray(Image.fromarray((big * 255 + 0.5).astype(np.uint8)).resize(
                (cw, ch), Image.Resampling.BOX, box=(x0, y0, x0 + bw, y0 + bh)), np.float32) / 255
            self.assertLess(float(np.abs(part - clip).mean()), 0.004, t)
            self.assertGreater(float(clip.max()), 0.3, t)

    def test_light_on_black(self):
        cls = fireflies.Fireflies
        self.assertEqual((cls.blend, cls.lens, cls.TURNS, cls.BOX), ("screen", False, False, (16, 9)))
        project = {"video": {"w": 1920, "h": 1080, "fps": "24", "frames": 144, "seconds": 6.0}, "length": 30}
        a = engine.fx_clip_name(project, {"kind": "fireflies", "params": {}})
        self.assertTrue(a.endswith(".mp4"))  # (H.264 on black, laid with Screen)
        self.assertEqual(a, engine.fx_clip_name(project, {"kind": "fireflies", "params": {"x": 12}}))  # (the box alone)


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_label(self):
        keys = [p["key"] for p in fireflies.SCHEMA["params"]]
        self.assertEqual(len(keys), len(set(keys)))
        for p in fireflies.SCHEMA["params"]:
            self.assertTrue(p.get("group") and p["label"], p["key"])
        self.assertFalse(set(overlay.ROTATIONS) & set(keys))
        self.assertTrue(fireflies.SCHEMA["note"])

    def test_looks_and_tones(self):
        """Each look sets every setting but the colour and the place; the first look is the defaults. A tone sets
        the colour, and a colour chosen by hand is "Tự chọn" (the page follows it: match)."""
        params = {p["key"]: p for p in fireflies.SCHEMA["params"]}
        look, tone = params["look"], params["tone"]
        keys = set(params) - {"look", "tone", "color", "x", "y", "scale"}
        defaults = effects.defaults("fireflies")
        for name, preset in look["presets"].items():
            self.assertEqual(set(preset), keys, name)
            self.assertEqual(effects.clean_params("fireflies", preset), {**defaults, **preset}, name)
        self.assertEqual(look["presets"][look["value"]], {k: v for k, v in defaults.items() if k in keys})
        self.assertEqual(tone["match"], ["color"])
        self.assertEqual({o["value"] for o in tone["options"]} - set(tone["presets"]), {"custom"})
        self.assertEqual(tone["presets"][tone["value"]], {"color": defaults["color"]})


if __name__ == "__main__":
    unittest.main()
