import time
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, leaves

W, H = 320, 180


def frame(fx, t, w=W, h=H, mask=None, base=0.3):
    pic = np.full((h, w, 3), base, np.float32)
    canvas = effects.Canvas(w, h)
    fx.draw(t, canvas, mask)
    canvas.flush(pic, fx.blend)
    return pic


MAX = {p["key"]: p["max"] for p in leaves.SCHEMA["params"] if p["type"] == "range"}
MIN = {p["key"]: p["min"] for p in leaves.SCHEMA["params"] if p["type"] == "range"}
OLD = {"style": "autumn", "amount": 30, "size": 45, "speed": 35, "wind": 15, "spin": 50, "sway": 50, "depth": 50,
       "blur": 30, "opacity": 100}  # a clip saved by the first leaves effect


class Loops(unittest.TestCase):
    """Frame P is frame 0 again with every look, kind of leaf and way of falling, at any loop length."""

    CASES = [dict(leaves.LOOKS[name], look=name) for name in leaves.LOOKS] + [
        {"shape": shape} for shape in ("mixed", "oval", "petal")] + [
        MAX, dict(MIN, amount=100), dict(MAX, wind=-100), {"spin": 100, "sway": 100, "gust": 100, "motion": 100},
        {"spin": 0, "sway": 0, "gust": 0, "motion": 0, "blur": 0, "haze": 0}, {"big": 0, "small": 100}, OLD]

    def test_frame_p_is_frame_zero(self):
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
            for params in self.CASES:
                fx = effects.make("leaves", params, 3, frames, fps, W, H)
                diff = np.abs(frame(fx, 0) - frame(fx, frames)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")
                if frames == 144:  # every leaf where it was, turned as it was
                    a, b = fx.at(0), fx.at(frames)
                    for u, v in zip(a[:2], b[:2]):
                        self.assertLess(float(np.abs(u - v).max()), 1e-6, params)
                    for u, v in zip(fx.pose(a[3], a[4]), fx.pose(b[3], b[4])):
                        self.assertLess(float(np.abs(np.cos(u) - np.cos(v)).max()), 1e-6, params)

    def test_leaves_move(self):
        fx = effects.make("leaves", {"amount": 80}, 3, 144, 24, W, H)
        self.assertGreater(np.abs(frame(fx, 0) - frame(fx, 6)).max(), 0.1)

    def test_every_way_of_falling(self):
        """Most leaves glide from side to side; with spin, some tumble (showing their back half the time) and some
        spiral; the gliding ones slow down at the ends of their swings."""
        fx = effects.make("leaves", {"amount": 100, "spin": 80}, 3, 720, 24, 1920, 1080)
        modes = np.bincount(fx.mode, minlength=3) / fx.n
        self.assertGreater(modes[leaves.GLIDE], 0.5)
        self.assertGreater(modes[leaves.TUMBLE], 0.1)
        self.assertGreater(modes[leaves.SPIRAL], 0.08)
        tumble = fx.mode == leaves.TUMBLE
        self.assertTrue((fx.flips[tumble] != 0).all())
        pitch = np.array([fx.pose(*fx.at(t)[3:])[2] for t in range(0, 720, 3)])
        backs = (np.cos(pitch[:, tumble]) < 0).mean(axis=0)
        self.assertTrue(((backs > 0.3) & (backs < 0.7)).all(), backs)
        calm = effects.make("leaves", {"amount": 100, "spin": 0}, 3, 720, 24, W, H)
        self.assertTrue((calm.mode == leaves.GLIDE).mean() > 0.8)
        self.assertFalse((calm.mode == leaves.TUMBLE).any())

    def test_gusts_never_blow_against_the_wind(self):
        """Between gusts the wind calms down; it never pushes the leaves back (their swing aside)."""
        for wind in (20, -50):
            fx = effects.make("leaves", {"wind": wind, "gust": 100, "sway": 0, "spin": 0}, 3, 720, 24, W, H)
            xs = np.array([fx.at(t)[0] for t in range(721)])
            s = np.array([np.mod(fx.s0 + fx.k * fx.len * fx.tau(t), fx.len) for t in range(721)])
            step = np.diff(xs, axis=0) * np.sign(wind)
            step = step[np.diff(s, axis=0) > 0]  # not where a leaf starts again at the top
            self.assertGreater(float(step.min()), 0.0, wind)

    def test_no_leaf_jumps_while_the_shutter_is_open(self):
        fx = effects.make("leaves", {"speed": 100, "amount": 60, "motion": 100}, 3, 144, 24, W, H)
        for t in range(144):
            x0, y0 = fx.at(t, -fx.SHUTTER / 2)[:2]
            x1, y1 = fx.at(t, fx.SHUTTER / 2)[:2]
            self.assertLess(float(np.hypot(x1 - x0, y1 - y0).max()), 0.25 * H, t)


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading(self):
        for p in leaves.SCHEMA["params"]:
            self.assertTrue(p.get("group"), p["key"])
            self.assertTrue(p["label"], p["key"])
        self.assertEqual(effects.SCHEMA["leaves"], leaves.SCHEMA)
        self.assertIs(effects.KINDS["leaves"], leaves.Leaves)
        self.assertGreater(leaves.Leaves.VERSION, 1)  # (the first leaves' preview clips are made again)

    def test_looks_and_seasons(self):
        """Each look sets every setting but the kind of leaf and its colours ("Lá rơi vừa" is the defaults); each
        season sets how many leaves are red, yellow and green; all to values the page accepts."""
        by = {p["key"]: p for p in leaves.SCHEMA["params"]}
        self.assertIs(by["look"]["presets"], leaves.LOOKS)
        self.assertIs(by["season"]["presets"], leaves.SEASONS)
        self.assertEqual({o["value"] for o in by["look"]["options"]}, set(leaves.LOOKS))
        self.assertEqual({o["value"] for o in by["season"]["options"]}, set(leaves.SEASONS))
        keys = set(by) - {"look", "shape", "season", "red", "yellow", "green"}
        defaults = effects.defaults("leaves")
        for name, preset in leaves.LOOKS.items():
            self.assertEqual(set(preset), keys, name)
            self.assertEqual(effects.clean_params("leaves", {**defaults, **preset}), {**defaults, **preset}, name)
        self.assertEqual(leaves.LOOKS["medium"], {k: v for k, v in defaults.items() if k in keys})
        for name, preset in leaves.SEASONS.items():
            self.assertEqual(set(preset), {"red", "yellow", "green"}, name)
        self.assertEqual(leaves.SEASONS[defaults["season"]],
                         {k: defaults[k] for k in ("red", "yellow", "green")})  # (the default season is the default)
        self.assertEqual(defaults["shape"], "maple")

    def test_old_clips_still_load(self):
        """A clip saved by the first leaves effect gets every setting of this one, meaning what it meant: its kind of
        leaves and colours, about as big and as many leaves."""
        p = effects.clean_params("leaves", OLD)
        self.assertEqual(set(p), set(effects.defaults("leaves")))
        self.assertEqual((p["shape"], p["season"], p["red"], p["yellow"], p["green"]),
                         ("mixed", "redgold", 50, 50, 0))
        for key in ("speed", "wind", "spin", "sway", "depth", "blur", "opacity"):
            self.assertEqual(p[key], OLD[key], key)
        self.assertLess(abs(leaves.size_px(p["small"]) - 0.65 * 48.1), 3)
        self.assertLess(abs(leaves.size_px(p["big"]) - 1.7 * 48.1), 6)
        self.assertEqual(p["amount"], 20)  # (as many leaves on screen as before)
        self.assertEqual(effects.clean_params("leaves", p), p)  # (cleaned again: as it is)
        for style, (shape, season) in (("green", ("oval", "summer")), ("petal", ("petal", "mixed"))):
            q = effects.clean_params("leaves", dict(OLD, style=style))
            self.assertEqual((q["shape"], q["season"]), (shape, season))
        self.assertEqual(effects.clean_params("leaves", {"size": "junk", "amount": float("nan")})["amount"],
                         effects.defaults("leaves")["amount"])
        old = effects.make("leaves", OLD, 3, 144, 24, 640, 360)
        new = effects.make("leaves", {}, 3, 144, 24, 640, 360)
        self.assertLess(abs(old.n / new.n - OLD["amount"] * leaves.OLD_FULL / (35 * leaves.Leaves.FULL)), 0.25)
        self.assertGreater(np.abs(frame(old, 10, 640, 360) - 0.3).max(), 0.1)

    def test_colour_mix(self):
        """How many leaves are red, yellow and green follows the three sliders; the same leaves keep their places
        when the mix changes."""
        for mix in ({"red": 100, "yellow": 0, "green": 0}, {"red": 0, "yellow": 0, "green": 100},
                    {"red": 50, "yellow": 50, "green": 0}, leaves.SEASONS["mixed"], leaves.SEASONS["early"],
                    {"red": 0, "yellow": 0, "green": 0}):
            fx = effects.make("leaves", dict(mix, amount=100), 3, 720, 24, 1920, 1080)
            shares = np.bincount(fx.family, minlength=3) / fx.n
            wanted = np.array([mix["red"], mix["yellow"], mix["green"]], float)
            wanted = wanted / wanted.sum() if wanted.sum() else np.full(3, 1 / 3)
            self.assertLess(float(np.abs(shares - wanted).max()), 0.1, (mix, shares))
            self.assertTrue((shares[wanted == 0] == 0).all(), mix)
        a = effects.make("leaves", {"red": 100, "yellow": 0, "green": 0}, 3, 144, 24, W, H)
        b = effects.make("leaves", {"red": 0, "yellow": 30, "green": 70}, 3, 144, 24, W, H)
        self.assertTrue(np.array_equal(a.at(40)[0], b.at(40)[0]))

    def test_kinds_of_leaf(self):
        for shape, kinds in (("maple", {"maple"}), ("oval", {"oval"}), ("petal", {"petal"}),
                             ("mixed", {"maple", "oval"})):
            fx = effects.make("leaves", {"shape": shape, "amount": 100}, 3, 144, 24, W, H)
            self.assertEqual(set(fx.kind), kinds, shape)
        mixed = effects.make("leaves", {"shape": "mixed", "amount": 100}, 3, 144, 24, 1920, 1080)
        self.assertGreater((mixed.kind == "maple").mean(), 0.5)

    def test_big_and_small_leaves(self):
        """The leaves' sizes span "Lá nhỏ nhất" to "Lá to nhất" (either way round), far ones small, near ones big;
        petals are smaller than leaves."""
        for big, small in ((78, 25), (60, 10), (20, 90)):
            fx = effects.make("leaves", {"big": big, "small": small, "amount": 60}, 3, 720, 24, 1920, 1080)
            lo, hi = sorted((leaves.size_px(small), leaves.size_px(big)))
            self.assertLess(abs(fx.size.min() / lo - 1), 0.08, (big, small))
            self.assertLess(abs(fx.size.max() / hi - 1), 0.08, (big, small))
            self.assertGreater(np.corrcoef(fx.z, np.log(fx.size))[0, 1], 0.6)
            if hi / lo > 2:  # big and small at once: a few really big, more small ones
                self.assertGreater((fx.size > lo * (hi / lo) ** 0.75).sum(), 2)
                self.assertGreater((fx.size < lo * (hi / lo) ** 0.35).sum(), fx.n // 4)
        same = effects.make("leaves", {"big": 50, "small": 50}, 3, 144, 24, W, H)
        self.assertLess(float(np.ptp(same.size)), 1e-6)
        petals = effects.make("leaves", {"shape": "petal"}, 3, 144, 24, W, H)
        self.assertAlmostEqual(float(petals.size.max()), leaves.size_px(78) * leaves.PETAL, delta=8)
        for v in (0, 25, 50, 100):
            self.assertAlmostEqual(leaves.size_slider(leaves.size_px(v)), v)

    def test_alike_at_every_size(self):
        """Sizes and paths are worked out in a 1080-line frame: the preview, the video and 4K show the same leaves
        in the same places (in frame heights); a small frame draws them from smaller pictures."""
        made = [effects.make("leaves", {}, 2, 144, 24, w, h) for w, h in ((640, 360), (1920, 1080), (3840, 2160))]
        for fx in made[1:]:
            (xa, ya, on_a, _, _), (xb, yb, on_b, _, _) = made[0].at(50), fx.at(50)
            self.assertTrue(np.array_equal(on_a, on_b))
            self.assertLess(float(np.abs(xa / made[0].h - xb / fx.h).max()), 1e-9)
            self.assertLess(float(np.abs(ya / made[0].h - yb / fx.h).max()), 1e-9)
            self.assertTrue(np.array_equal(made[0].family, fx.family))
        self.assertEqual([fx.master for fx in made], [72, 288, 288])
        self.assertEqual(effects.make("leaves", {"big": 100}, 2, 144, 24, 1280, 720).master, 288)
        self.assertEqual(effects.make("leaves", {"big": 70}, 2, 144, 24, 1280, 720).master, 144)

    def test_same_seed_same_picture(self):
        a = effects.make("leaves", {}, 9, 144, 24, W, H)
        b = effects.make("leaves", {}, 9, 144, 24, W, H)
        c = effects.make("leaves", {}, 10, 144, 24, W, H)
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))
        self.assertFalse(np.array_equal(frame(a, 17), frame(c, 17)))
        leaves._SPRITES.clear()  # (made again: the same pictures)
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))

    def test_nothing_outside_the_area(self):
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        fx = effects.make("leaves", {"amount": 100, "big": 100}, 3, 144, 24, W, H)
        pic = frame(fx, 7, mask=mask)
        self.assertTrue(np.array_equal(pic[:, :W // 2], np.full((H, W // 2, 3), 0.3, np.float32)))
        self.assertGreater(float(np.abs(pic[:, W // 2:] - 0.3).max()), 0.1)


def solidity(sprite):
    """How much of its convex hull a leaf's picture fills (deep sinuses between lobes: much less than 1)."""
    a = np.asarray(sprite.front[0], np.float32)[..., 3] / 255
    ys, xs = np.nonzero(a > 0.5)
    pts = sorted(set(zip(xs.tolist(), ys.tolist())))

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    hull = half(pts)[:-1] + half(pts[::-1])[:-1]
    area = 0.5 * abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(hull, hull[1:] + hull[:1])))
    return float((a > 0.5).sum()) / area


class Pictures(unittest.TestCase):
    def test_maple_leaves(self):
        """A maple leaf: one piece with deep sinuses between its lobes (much less of its outline filled than an
        ordinary leaf's), coloured as its colour, a duller and lighter back."""
        hues = {"red": lambda c: c[0] > 1.6 * c[1] and c[0] > 2 * c[2],
                "yellow": lambda c: c[0] > 0.7 and c[1] > 0.45 and c[2] < 0.5 * c[1],
                "green": lambda c: c[1] > c[0] and c[1] > 1.3 * c[2]}
        for family, hue in hues.items():
            for v in range(leaves.VARIANTS["maple"]):
                s = leaves.make_sprite("maple", family, v, 144)
                a = np.asarray(s.front[0], np.float32)[..., 3] / 255
                self.assertTrue(hue(s.rgb), (family, v, s.rgb))
                self.assertLess(solidity(s), 0.82, (family, v))  # (an ordinary leaf: about 0.94)
                front = np.asarray(s.front[0], np.float32)[..., :3].sum(axis=(0, 1)) / max(a.sum(), 1)
                back = np.asarray(s.back[0], np.float32)[..., :3].sum(axis=(0, 1)) / max(a.sum(), 1)
                self.assertGreater(back.mean(), front.mean(), (family, v))  # lighter
                self.assertLess(np.ptp(back), np.ptp(front) + 1, (family, v))  # duller
        for v in range(leaves.VARIANTS["oval"]):
            self.assertGreater(solidity(leaves.make_sprite("oval", "green", v, 144)), 0.88, v)

    def test_every_design_differs(self):
        seen = set()
        for kind, count in leaves.VARIANTS.items():
            for v in range(count):
                s = leaves.make_sprite(kind, "red", v, 72)
                seen.add(np.asarray(s.front[0])[..., 3].tobytes())
        self.assertEqual(len(seen), sum(leaves.VARIANTS.values()))

    def test_pictures_kept_within_bounds(self):
        """The pictures kept take a bounded share of memory, and every design at the largest size fits in it (they
        are not made again and again while a video is made)."""
        leaves._SPRITES.clear()
        try:
            for kind, count in leaves.VARIANTS.items():
                for family in leaves.FAMILIES:
                    for v in range(count):
                        leaves.sprite(kind, family, v, 288)
            everything = len(leaves._SPRITES)
            self.assertEqual(everything, sum(leaves.VARIANTS.values()) * 3 - 2 * leaves.VARIANTS["petal"])
            size = sum(sum(len(img.tobytes()) for img in s.front + s.back) for s in leaves._SPRITES.values())
            self.assertLessEqual(size, leaves.MAX_BYTES)
            leaves.MAX_BYTES = 4 << 20  # (less room: the oldest go)
            for v in range(30):
                leaves.sprite("maple", leaves.FAMILIES[v % 3], v % leaves.VARIANTS["maple"], 144)
            self.assertLessEqual(sum(s.bytes for s in leaves._SPRITES.values()), leaves.MAX_BYTES)
            self.assertIn(("maple", "green", 9, 144), leaves._SPRITES)  # (the last one made)
        finally:
            leaves.MAX_BYTES = 80 << 20
            leaves._SPRITES.clear()


class Speed(unittest.TestCase):
    def test_a_1080p_frame_costs_about_as_much_as_snow(self):
        """At the defaults a 1080p frame of leaves takes no more than about twice what one of snow takes (their
        pictures made first, as for a video)."""
        def cost(kind):
            fx = effects.make(kind, {}, 3, 720, 24, 1920, 1080)
            canvas = effects.Canvas(1920, 1080)
            pic = np.full((1080, 1920, 3), 0.3, np.float32)
            for t in range(0, 720, 40):
                fx.draw(t, canvas, None)
                canvas.flush(pic, fx.blend)
            best = []
            for t in range(5, 720, 90):
                t0 = time.perf_counter()
                fx.draw(t, canvas, None)
                canvas.flush(pic, fx.blend)
                best.append(time.perf_counter() - t0)
            return float(np.median(best))
        snow, leaf = cost("snow"), cost("leaves")
        self.assertLess(leaf, 2.5 * snow + 0.01, (leaf, snow))


if __name__ == "__main__":
    unittest.main()
