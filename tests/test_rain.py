import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, rain

W, H = 320, 180
LOOKS = next(p for p in rain.SCHEMA["params"] if p["key"] == "look")["presets"]


def frame(fx, t, pic=None, mask=None, w=W, h=H):
    pic = np.full((h, w, 3), 0.3, np.float32) if pic is None else pic.copy()
    canvas = effects.Canvas(w, h)
    fx.draw(t, canvas, mask)
    canvas.flush(pic, fx.blend)
    return pic


def everything(**extra):
    """Every part of the rain at once: splashes on a ground, sheets, mist, lightning, strong gusts."""
    return dict({"ground": 55, "splash": 100, "ripple": 100, "veil": 100, "mist": 100, "lightning": 100,
                 "flashes": 10, "gust": 100, "waves": 100, "blur": 100, "edge": 100}, **extra)


class Loops(unittest.TestCase):
    CASES = [{}, everything(), everything(wind=-100, speed=100, amount=100, length=100),
             everything(wind=100, speed=0, size=100, depth=100), everything(depth=0, variety=100, ground=10),
             {"amount": 3, "gust": 100, "waves": 100}] + [dict(v, look=k) for k, v in LOOKS.items()]

    def test_frame_p_is_frame_0(self):
        """Every part (drops, gusts, waves, sheets, splashes, lightning) returns to frame 0 after one loop."""
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
            for params in self.CASES:
                fx = effects.make("rain", params, 3, frames, fps, W, H)
                diff = np.abs(frame(fx, 0) - frame(fx, frames)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")
                t = frames // 3  # and any frame, a loop later
                self.assertLess(np.abs(frame(fx, t) - frame(fx, t + frames)).max(), 1 / 512, f"{params} t={t}")

    def test_signals_are_periodic(self):
        fx = effects.make("rain", everything(), 3, 150, 25, W, H)
        for table in (fx.gusting, fx.carried, fx.density):
            self.assertAlmostEqual(float(rain._at(table, 0.0)), float(rain._at(table, 1.0)), places=12)
            self.assertAlmostEqual(float(rain._at(table, 0.999999)), float(rain._at(table, 0.0)), places=4)
        for t in (0.0, 0.3, 0.71):
            np.testing.assert_allclose(fx.flash_at(t), fx.flash_at(t + 1.0), atol=1e-9)


class Look(unittest.TestCase):
    def test_nothing_when_amount_is_zero(self):
        for params in ({"amount": 0}, everything(amount=0, lightning=0)):
            fx = effects.make("rain", params, 3, 144, 24, W, H)
            for t in (0, 50):
                self.assertTrue(np.array_equal(frame(fx, t), np.full((H, W, 3), 0.3, np.float32)), params)

    def test_keeps_out_of_its_area(self):
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        fx = effects.make("rain", everything(amount=100), 3, 144, 24, W, H)
        flashes = [t for t in range(144) if fx.flash_at(t / 144)[0] > 0.3]
        self.assertTrue(flashes)
        for t in (7, flashes[0]):
            pic = frame(fx, t, mask=mask)
            self.assertLess(float(np.abs(pic[:, :W // 2] - 0.3).max()), 1e-5)
            self.assertGreater(float(np.abs(pic[:, W // 2:] - 0.3).max()), 0.05)

    def test_moves(self):
        fx = effects.make("rain", {}, 3, 144, 24, W, H)
        self.assertGreater(np.abs(frame(fx, 0) - frame(fx, 3)).max(), 0.05)

    def test_same_seed_same_picture(self):
        a, b, c = (effects.make("rain", everything(), s, 144, 24, W, H) for s in (9, 9, 10))
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))
        self.assertFalse(np.array_equal(frame(a, 17), frame(c, 17)))

    def test_more_rain_keeps_the_drops_there_were(self):
        """Raising the amount adds drops; the ones already falling stay where they were."""
        few = effects.make("rain", {"amount": 20}, 4, 144, 24, W, H)
        more = effects.make("rain", {"amount": 70}, 4, 144, 24, W, H)
        self.assertGreater(more.n, 2 * few.n)
        have = set(zip(np.round(more.xe, 9), np.round(more.s0, 9)))
        self.assertTrue(all(d in have for d in zip(np.round(few.xe, 9), np.round(few.s0, 9))))

    def test_same_look_at_every_size(self):
        """Sizes are in pixels of a 1080-line frame: the preview, 1080p and 4K hold about as much rain."""
        cover = []
        for w, h in ((640, 360), (1920, 1080), (3840, 2160)):
            fx = effects.make("rain", {"veil": 0, "mist": 0}, 5, 240, 24, w, h)
            canvas = effects.Canvas(w, h)
            total = 0.0
            for t in (3, 60, 150):
                fx.draw(t, canvas, None)
                total += float(canvas.a.mean())
                canvas.reset()
            cover.append(total)
        self.assertLess(max(cover) / min(cover), 1.25, cover)

    def test_lightning(self):
        """Off at 0; at a whole number of times a loop otherwise, and the sky lights up."""
        fx = effects.make("rain", {"lightning": 0}, 3, 750, 25, W, H)
        self.assertEqual(max(fx.flash_at(t / 750)[0] for t in range(750)), 0.0)
        for flashes, seconds in ((2, 30), (10, 30), (3, 60), (1, 6)):
            frames = seconds * 25
            fx = effects.make("rain", {"lightning": 80, "flashes": flashes}, 3, frames, 25, W, H)
            levels = np.array([fx.flash_at(t / frames)[0] for t in range(frames)])
            strikes = int(np.sum((levels[1:] > 0.5) & (levels[:-1] < 0.05)))
            self.assertEqual(strikes, max(1, round(flashes * seconds / 30)), (flashes, seconds))
            t = int(np.argmax(levels))
            self.assertGreater(frame(fx, t)[:H // 3].mean(), frame(fx, (t + frames // 2) % frames)[:H // 3].mean())

    def test_gusts_lean_the_rain(self):
        """Gusts add to the wind's slant now and then, smoothly, the same way as the wind."""
        calm = effects.make("rain", {"gust": 0, "wind": 30}, 3, 750, 25, W, H)
        self.assertEqual(float(np.abs(calm.gusting).max()), 0.0)
        for wind in (30, -30):
            fx = effects.make("rain", {"gust": 100, "wind": wind}, 3, 750, 25, W, H)
            lean = np.degrees(np.arctan(fx.gusting))
            self.assertGreater(float(np.abs(lean).max()), 20)
            self.assertTrue(np.all(np.sign(lean[np.abs(lean) > 0.01]) == np.sign(wind)))
            self.assertLess(float(np.abs(np.diff(lean)).max()), 0.5)  # no jumps (2048 steps a loop)
            slants = []  # of the streaks on screen, in the strongest gust and between gusts
            for i in (np.argmax(np.abs(lean)), np.argmin(np.abs(lean))):
                _, _, ux, uy, _, _, show = fx.drops(int(i / len(lean) * 750))
                slants.append(np.degrees(np.arctan2(ux[show], uy[show])).mean() * np.sign(wind))
            self.assertGreater(slants[0], slants[1] + 10)

    def test_streaks_stop_at_the_ground(self):
        layer = rain.Layer(40, 60)
        one = np.array([1.0])
        layer.streaks(one * 10.3, one * 2.0, one * 12.1, one * 50.0, one * 1.0, one * 0.8, one * 30.0)
        cover = layer.view()
        self.assertEqual(float(cover[30:].max()), 0.0)  # nothing below the ground
        self.assertTrue(all(cover[r].max() > 0 for r in range(2, 30)))  # the rest of the streak is there
        self.assertEqual(float(cover[:2].max()), 0.0)

    def test_streak_coverage(self):
        """A streak thinner than a pixel covers that share of it; brighter towards its head."""
        cover = []
        for width in (0.25, 0.5, 1.0, 2.0):
            layer = rain.Layer(40, 120)
            one = np.array([1.0])
            layer.streaks(one * 20.4, one * 10.0, one * 20.4, one * 110.0, one * width, one * 1.0, one * 999.0)
            cover.append(layer.view())
        sums = [float(c.sum()) for c in cover]
        for a, b in zip(sums, sums[1:]):
            self.assertAlmostEqual(b / a, 2.0, delta=0.05)
        rows = cover[2].sum(axis=1)
        self.assertGreater(rows[95], 2 * rows[25])  # head (row 110) brighter than the tail (row 10)
        self.assertLessEqual(float(cover[2].max()), 1.0 + 1e-6)

    def test_splashes_only_with_a_ground(self):
        """No ground in view: nothing below where the drops already are; a ground: splashes and rings on it."""
        bare = effects.make("rain", {"ground": 100, "splash": 100, "ripple": 100}, 3, 144, 24, W, H)
        wet = effects.make("rain", {"ground": 60, "splash": 100, "ripple": 100}, 3, 144, 24, W, H)
        dry = effects.make("rain", {"ground": 60, "splash": 0, "ripple": 0}, 3, 144, 24, W, H)
        self.assertTrue(np.all(bare.hit > 2))
        band = slice(int(0.62 * H), H)
        more = sum(float(np.abs(frame(wet, t) - frame(dry, t))[band].sum()) for t in range(0, 144, 12))
        self.assertGreater(more, 1.0)


class Settings(unittest.TestCase):
    def test_old_settings_keep_their_meaning(self):
        """Projects made before keep every setting they had, with the same range."""
        old = {"amount": (45, 0, 100), "speed": (60, 0, 100), "length": (50, 0, 100), "size": (35, 0, 100),
               "wind": (10, -100, 100), "depth": (50, 0, 100), "opacity": (45, 0, 100)}
        by_key = {p["key"]: p for p in rain.SCHEMA["params"]}
        for key, (_, lo, hi) in old.items():
            self.assertEqual((by_key[key]["min"], by_key[key]["max"]), (lo, hi), key)
        self.assertEqual(by_key["color"]["type"], "color")
        p = effects.clean_params("rain", {"amount": 80, "wind": -40, "color": "#aabbcc"})
        self.assertEqual((p["amount"], p["wind"], p["color"]), (80, -40, "#aabbcc"))
        self.assertEqual((p["ground"], p["lightning"]), (100, 0))  # new parts start out of the way
        self.assertEqual(rain.Rain.VERSION, 2)

    def test_every_setting_has_a_group(self):
        groups = [p.get("group") for p in rain.SCHEMA["params"]]
        self.assertTrue(all(groups))
        seen = []
        for g in groups:  # each heading once: a group's settings sit together
            if not seen or seen[-1] != g:
                self.assertNotIn(g, seen)
                seen.append(g)

    def test_looks(self):
        """Choosing a look sets valid values; the default look is the defaults."""
        by_key = {p["key"]: p for p in rain.SCHEMA["params"]}
        defaults = effects.defaults("rain")
        self.assertEqual(defaults["look"], "medium")
        self.assertEqual({k: defaults[k] for k in LOOKS["medium"]}, LOOKS["medium"])
        for name, values in LOOKS.items():
            for key, v in values.items():
                self.assertIn(key, by_key, name)
                self.assertNotIn(key, ("look", "ground", "color"), name)  # the scene's own, kept
                self.assertTrue(by_key[key]["min"] <= v <= by_key[key]["max"], (name, key))
            self.assertEqual(effects.clean_params("rain", dict(defaults, **values)), dict(defaults, **values))


if __name__ == "__main__":
    unittest.main()
