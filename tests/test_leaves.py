import math
import time
import unittest

import numpy as np
from PIL import Image

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, engine, leafshapes, leaves

W, H = 320, 180


def frame(fx, t, w=W, h=H, mask=None, base=0.3):
    pic = np.full((h, w, 3), base, np.float32)
    canvas = effects.Canvas(w, h)
    fx.draw(t, canvas, mask)
    canvas.flush(pic, fx.blend)
    return pic


def only(*shapes, **extra):
    """Settings with just these shapes ticked."""
    return {**{s: s in shapes for s in leaves.SHAPES}, **extra}


MAX = {p["key"]: p["max"] for p in leaves.SCHEMA["params"] if p["type"] == "range"}
MIN = {p["key"]: p["min"] for p in leaves.SCHEMA["params"] if p["type"] == "range"}
OLD = {"style": "autumn", "amount": 30, "size": 45, "speed": 35, "wind": 15, "spin": 50, "sway": 50, "depth": 50,
       "blur": 30, "opacity": 100}  # a clip saved by the first leaves effect
REMAKE = dict(leaves.LOOKS["medium"], look="medium", shape="maple", season="mixed", red=40, yellow=35, green=25)
# (a clip saved by the remake before the shapes: one kind of leaf)


class Loops(unittest.TestCase):
    """Frame P is frame 0 again with every look, every shape on its own, every "Loại lá" and way of falling, at any
    loop length."""

    CASES = ([dict(leaves.LOOKS[name], look=name) for name in leaves.LOOKS]
             + [only(s) for s in leaves.SHAPES] + [dict(leaves.SCENES[s], scene=s) for s in leaves.SCENES]
             + [MAX, dict(MIN, amount=100), dict(MAX, wind=-100),
                {"spin": 100, "sway": 100, "gust": 100, "motion": 100},
                {"spin": 0, "sway": 0, "gust": 0, "motion": 0, "blur": 0, "haze": 0, "glow": 0},
                {"big": 0, "small": 100}, OLD, REMAKE, only()])

    def test_frame_p_is_frame_zero(self):
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
            for params in self.CASES:
                fx = effects.make("leaves", params, 3, frames, fps, W, H)
                diff = np.abs(frame(fx, 0) - frame(fx, frames)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")
                if frames == 144:  # every leaf where it was, turned as it was, lit as it was
                    a, b = fx.at(0), fx.at(frames)
                    for u, v in zip(a[:2], b[:2]):
                        self.assertLess(float(np.abs(u - v).max()), 1e-6, params)
                    pa, pb = fx.pose(a[3], a[4]), fx.pose(b[3], b[4])
                    for u, v in zip(pa, pb):
                        self.assertLess(float(np.abs(np.cos(u) - np.cos(v)).max()), 1e-6, params)
                    for u, v in zip(fx.light(pa[0], pa[2]), fx.light(pb[0], pb[2])):
                        self.assertLess(float(np.abs(u - v).max()), 1e-6, params)

    def test_every_shape_on_its_own_loops_at_1080p(self):
        """Each shape alone, big and many, at full size (its largest pictures): the loop closes exactly."""
        for s in leaves.SHAPES:
            fx = effects.make("leaves", only(s, amount=60, big=95), 4, 96, 24, 1920, 1080)
            diff = np.abs(frame(fx, 0, 1920, 1080) - frame(fx, 96, 1920, 1080)).max()
            self.assertLess(diff, 1 / 512, s)
            self.assertGreater(np.abs(frame(fx, 10, 1920, 1080) - 0.3).max(), 0.2, s)  # (something drawn)

    def test_leaves_move(self):
        fx = effects.make("leaves", {"amount": 80}, 3, 144, 24, W, H)
        self.assertGreater(np.abs(frame(fx, 0) - frame(fx, 6)).max(), 0.1)

    def test_no_spinning_without_spin(self):
        """At "Xoay, lật mặt, lộn vòng" 0 no leaf turns round, spirals or tumbles (it still tilts as it swings)."""
        fx = effects.make("leaves", {"spin": 0, "amount": 100}, 3, 720, 24, 1920, 1080)
        self.assertTrue((fx.mode == leaves.GLIDE).all())
        yaw = np.array([fx.pose(*fx.at(t)[3:])[1] for t in range(0, 721, 4)])
        self.assertLess(float(np.ptp(yaw, axis=0).max()), 0.1 * 2 * np.pi)

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

    def test_swing_keeps_pace_with_the_fall(self):
        """A gliding leaf swings sideways at most about 3 times as fast as it falls and hardly ever rises at the ends
        of a swing, however slow the fall; a slower fall swings slower, and big leaves swing slower than small."""
        periods = {}
        for params in ({}, {"speed": 0}, {"speed": 100}, {"speed": 0, "sway": 100}, dict(leaves.LOOKS["dreamy"]),
                       only("willow", "almond", "petal")):
            fx = effects.make("leaves", dict(params, gust=0), 3, 1800, 30, 1920, 1080)
            g = np.flatnonzero(fx.mode == leaves.GLIDE)
            xy = np.array([fx.at(t)[:2] for t in range(0, 1801)])[:, :, g] * 30  # px a second apart
            s = np.array([np.mod(fx.s0 + fx.k * fx.len * fx.tau(t), fx.len) for t in range(0, 1801)])[:, g]
            go = np.diff(s, axis=0) > 0  # (not where a leaf starts again at the top)
            vx, vy = np.diff(xy[:, 0], axis=0), np.diff(xy[:, 1], axis=0)
            for i in range(len(g)):
                fall = np.median(vy[go[:, i], i])
                self.assertLess(np.abs(vx[go[:, i], i]).max(), 3.3 * fall, params)
            self.assertLess(float((vy[go] < 0).mean()), 0.005, params)  # (rising: a frame in 200 at most)
            periods[str(params)] = np.median(fx.loop_s / fx.m[g])
            if not params:
                own = fx.own[g]
                big, small = fx.loop_s / fx.m[g][own > 0.75], fx.loop_s / fx.m[g][own < 0.25]
                self.assertGreater(np.median(big), 1.2 * np.median(small))
        self.assertGreater(periods[str({"speed": 0})], 1.8 * periods[str({})])
        self.assertLess(periods[str({"speed": 100})], 0.8 * periods[str({})])

    def test_gusts_never_blow_against_the_wind(self):
        """Between gusts the wind calms down; it never pushes the leaves back (their swing aside)."""
        for wind in (20, -50):
            fx = effects.make("leaves", {"wind": wind, "gust": 100, "sway": 0, "spin": 0}, 3, 720, 24, W, H)
            xs = np.array([fx.at(t)[0] for t in range(721)])
            s = np.array([np.mod(fx.s0 + fx.k * fx.len * fx.tau(t), fx.len) for t in range(721)])
            step = np.diff(xs, axis=0) * np.sign(wind)
            step = step[np.diff(s, axis=0) > 0]  # not where a leaf starts again at the top
            self.assertGreater(float(step.min()), 0.0, wind)

    def test_lanes_spread_evenly(self):
        """The leaves come down all across the frame (no band left empty for the whole loop), and each time a leaf
        comes down again it is another leaf: shifted a little, another design."""
        for params in ({}, dict(leaves.LOOKS["few"]), only("almond", "birch")):
            for seed in (1, 2, 3):
                fx = effects.make("leaves", params, seed, 720, 24, 1920, 1080)
                hist = np.zeros(8)
                for t in range(0, 720, 12):
                    x, y, on = fx.at(t)[:3]
                    hist += np.histogram(x[on & (y > 0) & (y < 1080)], bins=8, range=(0, 1920))[0]
                hist /= hist.mean()
                self.assertGreater(hist.min(), 0.45, (params, seed, hist))
                self.assertLess(hist.max(), 1.6, (params, seed, hist))
        fx = effects.make("leaves", {"sway": 0, "gust": 0, "spin": 0}, 3, 720, 24, 1920, 1080)
        again = np.flatnonzero(fx.k > 1)
        self.assertGreater(len(again), fx.n // 2)
        for i in again:  # at the same place along its path, one time round later
            t = 100.0
            later = t + fx.P / fx.k[i]
            self.assertEqual((fx.passes(later)[i] - fx.passes(t)[i]) % fx.k[i], 1)
            self.assertAlmostEqual(fx.at(later)[1][i], fx.at(t)[1][i], 6)
            self.assertGreater(abs(fx.at(later)[0][i] - fx.at(t)[0][i]), 2.0)  # another place across
            self.assertNotEqual(fx.design(later)[i], fx.design(t)[i])  # another design

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
        own = len(leaves.SCHEMA["params"])  # (then the place and turn every full-frame effect has: test_sheet)
        self.assertEqual(effects.SCHEMA["leaves"], dict(leaves.SCHEMA, params=effects.SCHEMA["leaves"]["params"]))
        self.assertEqual(effects.SCHEMA["leaves"]["params"][:own], leaves.SCHEMA["params"])
        self.assertEqual([p["group"] for p in effects.SCHEMA["leaves"]["params"][own:]],
                         ["Vị trí và cỡ"] * 3 + ["Xoay 3D"] * 4)
        self.assertIs(effects.KINDS["leaves"], leaves.Leaves)
        self.assertEqual(leaves.Leaves.VERSION, 4)  # (the remake's preview clips are made again)

    def test_shapes_are_checkboxes(self):
        """Each shape is a checkbox of its own under one heading, the two maples ticked at first; "Loại lá" and
        "Kiểu lá rơi" come first, under "Kiểu"."""
        by = {p["key"]: p for p in leaves.SCHEMA["params"]}
        keys = [p["key"] for p in leaves.SCHEMA["params"]]
        self.assertEqual(keys[:2], ["scene", "look"])
        self.assertEqual({by[k]["group"] for k in ("scene", "look")}, {"Kiểu"})
        self.assertEqual(keys[2:2 + len(leaves.SHAPES)], list(leaves.SHAPES))
        for s in leaves.SHAPES:
            self.assertEqual(by[s]["type"], "toggle", s)
            self.assertEqual(by[s]["group"], leaves.SHAPES_GROUP, s)
            self.assertEqual(by[s]["value"], s in ("maple", "japanese"), s)
        self.assertEqual(len({by[s]["label"] for s in leaves.SHAPES}), len(leaves.SHAPES))
        p = effects.clean_params("leaves", {s: "true" if s == "oak" else 0 for s in leaves.SHAPES})
        self.assertEqual([s for s in leaves.SHAPES if p[s]], ["oak"])

    def test_looks_and_seasons(self):
        """Each "Kiểu lá rơi" sets how the leaves fall, how many and how big ("Lá rơi vừa" is the defaults), not
        which shapes, their colours or the light; each season sets how many leaves are red, yellow and green; all
        to values the page accepts."""
        by = {p["key"]: p for p in leaves.SCHEMA["params"]}
        self.assertIs(by["look"]["presets"], leaves.LOOKS)
        self.assertIs(by["season"]["presets"], leaves.SEASONS)
        self.assertEqual({o["value"] for o in by["look"]["options"]}, set(leaves.LOOKS))
        self.assertEqual({o["value"] for o in by["season"]["options"]}, set(leaves.SEASONS))
        keys = set(by) - {"scene", "look", "season", "red", "yellow", "green", "glow"} - set(leaves.SHAPES)
        defaults = effects.defaults("leaves")
        for name, preset in leaves.LOOKS.items():
            self.assertEqual(set(preset), keys, name)
            self.assertEqual(effects.clean_params("leaves", {**defaults, **preset}), {**defaults, **preset}, name)
        self.assertEqual(leaves.LOOKS["medium"], {k: v for k, v in defaults.items() if k in keys})
        for name, preset in leaves.SEASONS.items():
            self.assertEqual(set(preset), {"red", "yellow", "green"}, name)
        self.assertEqual(leaves.SEASONS[defaults["season"]],
                         {k: defaults[k] for k in ("red", "yellow", "green")})  # (the default season is the default)

    def test_kinds_of_leaves_at_once(self):
        """Each "Loại lá" ticks its shapes (and only those) and sets its season's colours (cherry petals leave the
        colours as they are); the defaults are "Rừng phong"; the page follows the ticked shapes back to the "Loại
        lá" (match), or to "Tự chọn" when none has them."""
        by = {p["key"]: p for p in leaves.SCHEMA["params"]}
        scene = by["scene"]
        self.assertIs(scene["presets"], leaves.SCENES)
        self.assertEqual(scene["match"], list(leaves.SHAPES))
        values = [o["value"] for o in scene["options"]]
        self.assertEqual(values, list(leaves.SCENES) + [leaves.CUSTOM])
        self.assertNotIn(leaves.CUSTOM, scene["presets"])
        defaults = effects.defaults("leaves")
        self.assertEqual(defaults["scene"], "maple")
        self.assertEqual(leaves.scene_of(defaults), "maple")
        names = {}
        for name, preset in leaves.SCENES.items():
            p = effects.clean_params("leaves", {**defaults, **preset})
            self.assertEqual(p, {**defaults, **preset}, name)
            ticked = tuple(s for s in leaves.SHAPES if p[s])
            self.assertTrue(ticked, name)
            names[ticked] = name
            self.assertEqual(leaves.scene_of(p), name)
            if "petal" not in ticked:
                self.assertIn(preset["season"], leaves.SEASONS, name)
                self.assertEqual({k: preset[k] for k in ("red", "yellow", "green")}, leaves.SEASONS[preset["season"]])
        self.assertEqual(len(names), len(leaves.SCENES))  # (no two with the same shapes)
        self.assertEqual(leaves.SCENES["almond"]["season"], "winter")  # (a bàng tree in winter: red, orange, a few
        # yellow and green)
        self.assertEqual(leaves.SEASONS["winter"], {"red": 55, "yellow": 35, "green": 10})
        self.assertEqual(leaves.SCENES["ginkgo"]["season"], "gold")
        self.assertEqual(leaves.scene_of(only("oak", "willow")), leaves.CUSTOM)
        labels = {o["value"]: o["label"] for o in scene["options"]}
        self.assertIn("bàng", labels["almond"].lower())
        self.assertIn("anh đào", labels["petal"].lower())

    def test_shapes_ticked(self):
        """The leaves are the shapes ticked, in about equal shares; none ticked: maple leaves; the same settings, the
        same leaves (ticking another shape makes a new mix: other shapes and places)."""
        for s in leaves.SHAPES:
            fx = effects.make("leaves", only(s, amount=100), 3, 144, 24, W, H)
            self.assertEqual(set(fx.kind.tolist()), {s}, s)
        fx = effects.make("leaves", only(amount=100), 3, 144, 24, W, H)
        self.assertEqual(set(fx.kind.tolist()), {"maple"})
        for shapes in (("maple", "japanese"), ("ginkgo", "oak", "almond"), leaves.SHAPES):
            fx = effects.make("leaves", only(*shapes, amount=100), 3, 720, 24, 1920, 1080)
            self.assertEqual(set(fx.kind.tolist()), set(shapes))
            share = np.array([(fx.kind == s).mean() for s in shapes])
            self.assertLess(float(np.abs(share - 1 / len(shapes)).max()), 0.35 / len(shapes) + 0.03, (shapes, share))
        self.assertGreater(fx.n, 100)
        again = effects.make("leaves", only(*leaves.SHAPES, amount=100), 3, 720, 24, 1920, 1080)
        self.assertTrue(np.array_equal(fx.kind, again.kind))  # (the same seed: the same leaves)

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

    def test_summer_is_green(self):
        """Green leaves turn yellow or red at their edges only when some leaves have turned already: in summer every
        leaf is plain green; early in autumn many green leaves are turning."""
        summer = effects.make("leaves", dict(leaves.SEASONS["summer"], amount=100), 3, 720, 24, 1920, 1080)
        self.assertTrue((summer.palette == 2).all())
        early = effects.make("leaves", dict(leaves.SEASONS["early"], amount=100), 3, 720, 24, 1920, 1080)
        green = early.family == 2
        self.assertGreater((early.palette[green] == 3).mean(), 0.25)
        self.assertTrue((early.palette[~green] == early.family[~green]).all())

    def test_each_shape_as_big_as_it_is(self):
        """Each shape keeps its size beside the others (a bàng or a maple leaf big, a birch or willow leaf small,
        a ginkgo leaf between), all between "Lá nhỏ nhất" and "Lá to nhất"; the biggest shape ticked reaches "Lá to
        nhất", even on its own."""
        lo, hi = leaves.size_px(22), leaves.size_px(82)
        rel = leaves.REL
        for big in ("almond", "maple", "oak"):
            for small in ("birch", "willow", "petal"):
                self.assertGreater(rel[big], 1.3 * rel[small], (big, small))
        self.assertGreater(rel["ginkgo"], 1.1 * rel["birch"])
        self.assertLess(rel["ginkgo"], 0.8 * rel["maple"])
        self.assertEqual(max(rel, key=rel.get), "almond")
        fx = effects.make("leaves", only(*leaves.SHAPES, amount=100), 3, 720, 24, 1920, 1080)
        self.assertTrue((fx.size >= lo - 1e-6).all() and (fx.size <= hi + 1e-6).all())
        for s in leaves.SHAPES:
            ceiling = max(lo, hi * rel[s] / rel["almond"])
            mine = fx.size[fx.kind == s]
            self.assertTrue((mine <= ceiling + 1e-6).all(), s)
            self.assertGreater(float(mine.max()), 0.75 * ceiling, s)  # (its nearest about as big as it gets)
        self.assertGreater(float(fx.size[fx.kind == "almond"].max()), 0.9 * hi)
        for s in leaves.SHAPES:  # alone: the whole range (petals: smaller than leaves, as they always were)
            alone = effects.make("leaves", only(s, amount=100), 3, 720, 24, 1920, 1080)
            want = hi * (leaves.REL["petal"] / leaves.REL["maple"] if s == "petal" else 1.0)
            self.assertLess(abs(float(alone.size.max()) / want - 1), 0.08, s)

    def test_big_and_small_leaves(self):
        """The leaves' sizes span "Lá nhỏ nhất" to "Lá to nhất" (either way round), far ones small, near ones big."""
        for big, small in ((78, 25), (60, 10), (20, 90)):
            fx = effects.make("leaves", {"big": big, "small": small, "amount": 60}, 3, 720, 24, 1920, 1080)
            lo, hi = sorted((leaves.size_px(small), leaves.size_px(big)))
            maple = fx.kind == "maple"
            self.assertLess(abs(fx.size.min() / lo - 1), 0.08, (big, small))
            self.assertLess(abs(fx.size[maple].max() / hi - 1), 0.08, (big, small))
            self.assertGreater(np.corrcoef(fx.z, np.log(fx.size))[0, 1], 0.6)
            if hi / lo > 2:  # big and small at once: a few really big, more small ones
                self.assertGreater((fx.size > lo * (hi / lo) ** 0.75).sum(), 2)
                self.assertGreater((fx.size < lo * (hi / lo) ** 0.35).sum(), fx.n // 4)
        same = effects.make("leaves", {"big": 50, "small": 50}, 3, 144, 24, W, H)
        self.assertLess(float(np.ptp(same.size)), 1e-6)
        for v in (0, 25, 50, 100):
            self.assertAlmostEqual(leaves.size_slider(leaves.size_px(v)), v)

    def test_clearly_big_and_small_at_once(self):
        """In the frame, at the defaults and in every look: the biggest tenth of the leaves at least 3 times the
        size of the smallest tenth, and some reach 80% of "Lá to nhất"."""
        for name in leaves.LOOKS:
            fx = effects.make("leaves", dict(leaves.LOOKS[name], look=name), 11, 720, 24, 1920, 1080)
            sizes = []
            for t in range(0, 720, 30):
                x, y, on = fx.at(t)[:3]
                sizes.append(fx.size[on & (x >= 0) & (x < 1920) & (y >= 0) & (y < 1080)])
            s = np.concatenate(sizes)
            self.assertGreater(np.percentile(s, 90) / np.percentile(s, 10), 3, name)
            self.assertGreater((s >= 0.8 * leaves.size_px(leaves.LOOKS[name]["big"])).sum(), 0, name)
            self.assertGreaterEqual(leaves.LOOKS[name]["big"], 82, name)

    def test_old_clips_still_load(self):
        """A clip saved by the first leaves effect gets every setting of this one, meaning what it meant: its kind of
        leaves (maple leaves and ordinary ones: now both maples and birch) and colours, about as big and as many
        leaves."""
        p = effects.clean_params("leaves", OLD)
        self.assertEqual(set(p), set(effects.defaults("leaves")))
        self.assertEqual([s for s in leaves.SHAPES if p[s]], ["maple", "japanese", "birch"])
        self.assertEqual((p["scene"], p["season"], p["red"], p["yellow"], p["green"]),
                         (leaves.CUSTOM, "redgold", 50, 50, 0))
        for key in ("speed", "wind", "spin", "sway", "depth", "blur", "opacity"):
            self.assertEqual(p[key], OLD[key], key)
        self.assertLess(abs(leaves.size_px(p["small"]) - 0.7 * 48.1), 3)
        self.assertLess(abs(leaves.size_px(p["big"]) - 1.8 * 48.1), 6)
        self.assertEqual(p["amount"], 15)  # (as many leaves in the frame as before)
        self.assertEqual(effects.clean_params("leaves", p), p)  # (cleaned again: as it is)
        for style, (shapes, season) in (("green", (("birch",), "summer")), ("petal", (("petal",), "mixed"))):
            q = effects.clean_params("leaves", dict(OLD, style=style))
            self.assertEqual((tuple(s for s in leaves.SHAPES if q[s]), q["season"]), (shapes, season))
        self.assertEqual(effects.clean_params("leaves", {"size": "junk", "amount": float("nan")})["amount"],
                         effects.defaults("leaves")["amount"])
        # the first leaves effect showed about 13.5 leaves in a 1080p frame with these settings, its maple leaves'
        # area about 2830 px² each on average (measured with it)
        counts, areas = [], []
        for seed in (3, 4, 5):
            old = effects.make("leaves", OLD, seed, 720, 24, 1920, 1080)
            for t in range(0, 720, 12):
                x, y, on = old.at(t)[:3]
                seen = on & (x >= 0) & (x < 1920) & (y >= 0) & (y < 1080)
                counts.append(seen.sum())
                areas.append(old.size[seen & (old.kind == "maple")] ** 2)
        self.assertLess(abs(np.mean(counts) / 13.5 - 1), 0.15, np.mean(counts))
        self.assertLess(abs(np.mean(np.concatenate(areas)) / 2830 - 1), 0.2)
        self.assertGreater(np.abs(frame(old, 10, 640, 360) - 0.3).max(), 0.1)

    def test_remade_clips_still_load(self):
        """A clip saved by the remake (one kind of leaf: "shape") gets the shapes that kind was: the same kind of
        leaves keep falling, every other setting as it was."""
        for shape, shapes in (("maple", ("maple", "japanese")), ("mixed", ("maple", "japanese", "birch")),
                              ("oval", ("birch",)), ("petal", ("petal",)), ("junk", ("maple", "japanese")),
                              (["x"], ("maple", "japanese"))):
            p = effects.clean_params("leaves", dict(REMAKE, shape=shape))
            self.assertEqual(tuple(s for s in leaves.SHAPES if p[s]), shapes, shape)
            self.assertNotIn("shape", p)
            self.assertEqual(p["scene"], leaves.scene_of(p), shape)
            for key, v in REMAKE.items():
                if key != "shape":
                    self.assertEqual(p[key], v, (shape, key))
        self.assertEqual(effects.clean_params("leaves", dict(REMAKE, shape="maple"))["scene"], "maple")
        self.assertEqual(effects.clean_params("leaves", dict(REMAKE, shape="petal"))["scene"], "petal")
        # ticked shapes win over an old kind saved beside them
        p = effects.clean_params("leaves", dict(only("oak"), shape="petal"))
        self.assertEqual([s for s in leaves.SHAPES if p[s]], ["oak"])
        # the old maple leaves fall as before: both kinds of maple, the biggest as big as "Lá to nhất"
        old = effects.make("leaves", dict(REMAKE, amount=100), 7, 720, 24, 1920, 1080)
        self.assertEqual(set(old.kind.tolist()), {"maple", "japanese"})
        self.assertLess(abs(float(old.size[old.kind == "maple"].max()) / leaves.size_px(82) - 1), 0.08)

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
            self.assertTrue(np.array_equal(made[0].kind, fx.kind))
        self.assertEqual([int(fx.master.max()) for fx in made], [72, 288, 288])
        self.assertEqual(int(effects.make("leaves", {"big": 100}, 2, 144, 24, 1280, 720).master.max()), 288)
        self.assertEqual(int(effects.make("leaves", {"big": 70}, 2, 144, 24, 1280, 720).master.max()), 144)
        fx = made[1]  # (each leaf's pictures as large as it: the small far ones small)
        share = np.array([leaves.SHARE[k] for k in fx.kind])
        self.assertTrue(((fx.master >= 288) == (share * 144 < 1.1 * fx.longest)).all())
        self.assertGreater((fx.master <= 144).mean(), 0.5)

    def test_same_seed_same_picture(self):
        a = effects.make("leaves", {}, 9, 144, 24, W, H)
        b = effects.make("leaves", {}, 9, 144, 24, W, H)
        c = effects.make("leaves", {}, 10, 144, 24, W, H)
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))
        self.assertFalse(np.array_equal(frame(a, 17), frame(c, 17)))
        leaves._SPRITES.clear()  # (made again: the same pictures)
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))
        d = effects.make("leaves", only(*leaves.SHAPES), 9, 144, 24, 1280, 720)
        first = frame(d, 30, 1280, 720)
        leaves._SPRITES.clear()
        self.assertTrue(np.array_equal(first, frame(d, 30, 1280, 720)))

    def test_nothing_outside_the_area(self):
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        fx = effects.make("leaves", {"amount": 100, "big": 100}, 3, 144, 24, W, H)
        pic = frame(fx, 7, mask=mask)
        self.assertTrue(np.array_equal(pic[:, :W // 2], np.full((H, W // 2, 3), 0.3, np.float32)))
        self.assertGreater(float(np.abs(pic[:, W // 2:] - 0.3).max()), 0.1)

    def test_preview_clip_is_sharper(self):
        """The leaves' preview clip has as many pixels as 960 x 540 (small leaves blur at 640 wide), never fewer than
        the usual one (a tall video: the usual, the page shows it small) nor more than the video; every other
        effect's is as it was, its name too (no preview made again for them)."""
        info = {"w": 1920, "h": 1080, "fps": "30", "frames": 300, "seconds": 10.0, "file": "input.mp4"}
        self.assertEqual(engine.fx_clip_frame(info, "leaves"), (960, 540))
        self.assertEqual(engine.fx_clip_frame(dict(info, w=3840, h=2160), "leaves"), (960, 540))
        self.assertEqual(engine.fx_clip_frame(dict(info, w=1080, h=1920), "leaves"), (640, 1138))  # (as snow's)
        self.assertEqual(engine.fx_clip_frame(dict(info, w=1080, h=1920), "snow"), (640, 1138))
        self.assertEqual(engine.fx_clip_frame(dict(info, w=1440, h=1080), "leaves"), (830, 622))
        self.assertEqual(engine.fx_clip_frame(dict(info, w=640, h=360), "leaves"), (640, 360))  # (never larger)
        for kind in ("snow", "glow", "glass"):
            self.assertEqual(engine.fx_clip_frame(info, kind), (640, 360), kind)
        self.assertEqual(engine.fx_clip_frame(info, "rain"), (1280, 720))  # (thin streaks: its own PREVIEW)
        project = {"id": "names", "video": info, "length": 30, "layers": []}
        names = {"snow": "fx-2a4ed27c4984.webm", "glow": "fx-15f81175c33d.mp4",
                 "glass": "fx-c4e2c2d87bcc.webm", "candle": "fx-273a3828f2a8.webm"}
        for kind, name in names.items():  # (measured before the leaves' shapes; rain and steam were remade since)
            self.assertEqual(engine.fx_clip_name(project, {"id": "x1", "kind": kind, "params": {}, "seed": 5}), name)
        self.assertNotEqual(engine.fx_clip_name(project, {"id": "x1", "kind": "leaves", "params": {}, "seed": 5}),
                            "fx-203e86c65a77.webm")


def solidity(inside):
    """How much of its convex hull a leaf's blade fills (deep sinuses between lobes: much less than 1)."""
    ys, xs = np.nonzero(inside)
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
    return float(inside.sum()) / area


def blade(kind, family, v, n=200):
    """A leaf design's blade (no stem) on a grid of leaf units, y up: inside (bool), and the grid."""
    design, _ = leafshapes.design_of(kind, family, v)
    x0, y0, x1, y1 = leafshapes._bounds(design)
    gx, gy = np.meshgrid(np.linspace(x0, x1, n, dtype=np.float32), np.linspace(y1, y0, n, dtype=np.float32))
    return leafshapes._blade(design, gx, gy)[0] < 0, gx, gy


def hsv(img):
    a = np.asarray(img, np.float32) / 255
    m = a[..., 3] > 0.5
    rgb = a[..., :3][m] / a[..., 3][m][:, None]
    hi, lo = rgb.max(1), rgb.min(1)
    d = np.maximum(hi - lo, 1e-6)
    r, g, b = rgb.T
    h = np.where(hi == r, ((g - b) / d) % 6, np.where(hi == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
    return h, (hi - lo) / np.maximum(hi, 1e-6), rgb @ np.array([0.2126, 0.7152, 0.0722])


class Pictures(unittest.TestCase):
    def test_each_shape_looks_like_itself(self):
        """Measured on the blades: maples cut deep between their lobes (a Japanese maple's deepest), an oak's lobes
        cut in too; the others whole; a willow leaf long and narrow, a bàng leaf longer than wide and widest beyond
        its middle, a birch or heart-shaped leaf about as wide as long, a ginkgo fan wider than long; a heart-shaped
        leaf notched at its stem, a ginkgo often notched at its edge."""
        def ratio(inside, gx, gy):  # length (along y) / width (along x) of the blade
            ys, xs = gy[inside], gx[inside]
            return float(np.ptp(ys) / np.ptp(xs))
        maple = []
        for v in range(4):
            sol = {k: solidity(blade(k, "green", v)[0]) for k in leafshapes.SHAPES}
            self.assertLess(sol["japanese"], 0.68, v)
            self.assertLess(sol["maple"], 0.9, v)
            maple.append(sol["maple"])
            self.assertLess(sol["oak"], 0.85, v)
            for k in ("ginkgo", "almond", "birch", "willow", "heart", "petal"):
                self.assertGreater(sol[k], 0.9, (k, v, sol[k]))
            r = {k: ratio(*blade(k, "green", v)) for k in ("willow", "almond", "birch", "heart", "ginkgo")}
            self.assertGreater(r["willow"], 4.5, v)
            self.assertTrue(1.45 < r["almond"] < 2.3, (v, r["almond"]))
            self.assertTrue(0.95 < r["birch"] < 1.6, (v, r["birch"]))
            self.assertTrue(0.9 < r["heart"] < 1.4, (v, r["heart"]))
            self.assertLess(r["ginkgo"], 0.85, v)
            inside, gx, gy = blade("almond", "green", v)
            rows = inside.sum(axis=1)  # (width at each height, top row first)
            widest_y = gy[np.argmax(rows), 0]
            self.assertGreater(widest_y, 0.55, v)  # (obovate: widest near its end)
            inside, gx, gy = blade("heart", "green", v, 300)
            col = inside[:, np.argmin(np.abs(gx[0] - 0.0))]  # down the middle: the notch reaches up to the stem
            low = gy[:, 0][np.flatnonzero(col)].min()
            side = inside[:, np.argmin(np.abs(gx[0] - 0.2))]
            self.assertGreater(low - gy[:, 0][np.flatnonzero(side)].min(), 0.05, v)
        self.assertLess(np.mean(maple), 0.83)
        notched = 0
        for v in range(6):
            design, _ = leafshapes.design_of("ginkgo", "yellow", v)
            notched += design["notch"] > 0.08
        self.assertGreaterEqual(notched, 2)

    def test_maple_leaves_are_alike_on_both_sides(self):
        """A maple leaf's two halves mirror each other (each a little different, nothing missing on one side)."""
        xs = np.linspace(-1.4, 1.4, 320, dtype=np.float32)
        ys = np.linspace(1.3, -0.7, 240, dtype=np.float32)
        gx, gy = np.meshgrid(xs, ys)
        for kind in ("maple", "japanese"):
            for palette in leaves.PALETTES:
                for v in range(leaves.VARIANTS[kind]):
                    design, _ = leafshapes.design_of(kind, palette, v)
                    inside = leafshapes._blade(design, gx, gy)[0] < 0
                    mirror = inside[:, ::-1]
                    least = 0.9 if kind == "maple" else 0.85  # (a Japanese maple's narrow lobes: a little less)
                    self.assertGreaterEqual((inside & mirror).sum() / (inside | mirror).sum(), least,
                                            (kind, palette, v))

    def test_maple_lobes_end_in_points(self):
        """Every main lobe of every maple design narrows to a point (an oak's or a bàng leaf's end is rounded): at 97%
        of its length it is less than a fifth as wide as at its widest."""
        def width(lobe, t):  # across its (bent) axis at t of its length
            x0, y0 = lobe.axis(t * lobe.L)
            x1, y1 = lobe.axis(t * lobe.L + 1e-3)
            tx, ty = (x1 - x0) / math.hypot(x1 - x0, y1 - y0), (y1 - y0) / math.hypot(x1 - x0, y1 - y0)
            q = np.linspace(-0.6, 0.6, 2401, dtype=np.float32)
            d = lobe.field((x0 + q * ty)[None, :], (y0 - q * tx)[None, :])[0][0]
            return float((d < 0).sum() * (q[1] - q[0]))
        for palette in leaves.PALETTES:
            for v in range(leaves.VARIANTS["maple"]):
                design, _ = leafshapes.design_of("maple", palette, v)
                for lobe in design["lobes"]:
                    if lobe.kind == "lobe":
                        self.assertLess(width(lobe, 0.97), 0.2 * width(lobe, lobe.widest), (palette, v))
        for kind in ("oak", "almond"):
            design, _ = leafshapes.design_of(kind, "red", 0)
            lobe = design["lobes"][0]
            self.assertGreater(width(lobe, 0.97), 0.25 * width(lobe, lobe.widest), kind)

    def test_oak_leaves_look_like_oak_leaves(self):
        """An English oak's leaf: broad rounded lobes, the sinuses between them cut in a third to two thirds of the
        way to the midrib (not to it: no fern; measured across, against the lobes either side), widest beyond its
        middle (its lobes smaller towards the stem). The oak before measured 0.83-0.87 here."""
        deepest = []
        xs = np.linspace(-0.7, 0.7, 1401, dtype=np.float32)
        c = int(np.argmin(np.abs(xs)))
        for palette in leaves.PALETTES:
            for v in range(leaves.VARIANTS["oak"]):
                design, _ = leafshapes.design_of("oak", palette, v)
                mid = design["lobes"][0]
                sides, along = [[], []], np.linspace(0.15, 0.8, 200)
                for s in along:
                    bx, by = mid.axis(s)
                    row = np.full((1, xs.size), by, np.float32)
                    inside = leafshapes._blade(design, (bx + xs)[None, :], row)[0][0] < 0
                    sides[0].append(xs[c] - xs[c - int(np.argmax(~inside[c::-1])) + 1])
                    sides[1].append(xs[c + int(np.argmax(~inside[c:])) - 1] - xs[c])
                for w in map(np.array, sides):
                    cuts = [1 - w[i] / min(w[:i].max(), w[i:].max()) for i in range(1, len(w) - 1)
                            if w[i] < w[i - 1] and w[i] <= w[i + 1]]
                    deepest.append(max(cuts))
                    self.assertTrue(0.1 < deepest[-1] < 0.7, (palette, v, deepest[-1]))
                width = np.array(sides[0]) + np.array(sides[1])
                self.assertGreater(along[int(np.argmax(width))], 0.45, (palette, v))  # (obovate)
                self.assertLess(width[along < 0.3].max(), 0.85 * width.max(), (palette, v))
        self.assertTrue(0.25 < np.median(deepest) < 0.6, np.median(deepest))

    def test_heart_leaves_are_lopsided(self):
        """A linden's leaf is a little lopsided: one lobe by the stem bigger than the other; its upper sides and its
        lobes toothed alike, its point short."""
        for v in range(leaves.VARIANTS["heart"]):
            inside, gx, gy = blade("heart", "yellow", v, 300)
            low = inside & (gy < 0.2)
            left, right = (low & (gx < 0)).sum(), (low & (gx > 0)).sum()
            self.assertGreater(max(left, right) / min(left, right), 1.04, v)
            design, _ = leafshapes.design_of("heart", "yellow", v)
            th = np.linspace(-math.pi, math.pi, 3601)
            _, teeth = leafshapes._heart_radius(design, th)
            for part in ((th > 0.5) & (th < 1.5), (th > 2.0) & (th < 2.6)):  # (up the side, and on a lobe)
                self.assertGreater(float(np.ptp(teeth[part])), 0.8 * design["teeth"][0], v)
            self.assertLess(design["point"][0], 0.12, v)

    def test_natural_colours(self):
        """Each shape in its own colours: red maples red (not orange), a ginkgo's "red" a deep golden orange (it
        never goes red), an oak's red russet brown, a bàng leaf's red to orange; yellows yellow, greens green; a
        willow's green yellowish and its back silvery; turning leaves warmer than green ones; a petal pink on both
        sides."""
        def mean(kind, family, view="front", size=72):
            out = []
            for v in range(3):
                h, s, l = hsv(getattr(leafshapes.make_sprite(kind, family, v, size), view)[0])
                out.append((np.median(h), s.mean(), l.mean()))
            return np.array(out).mean(axis=0)

        def reds(kind, most):  # (hues round 0: their share near red)
            h = np.concatenate([hsv(leafshapes.make_sprite(kind, "red", v, 72).front[0])[0] for v in range(3)])
            return ((h <= most) | (h >= 340)).mean()
        for kind in ("maple", "japanese"):
            self.assertGreater(reds(kind, 12), 0.6, kind)
            self.assertGreater(mean(kind, "red")[1], 0.6, kind)
        self.assertGreater(reds("almond", 25), 0.7)
        h, _, _ = mean("ginkgo", "red")
        self.assertTrue(24 < h < 45, h)  # (golden orange)
        h, s, l = mean("oak", "red")
        self.assertTrue(10 < h < 32 and l < 0.4, (h, l))  # (russet: brown-red, dark)
        for kind in leafshapes.SHAPES[:-1]:
            h, s, l = mean(kind, "yellow")
            self.assertTrue(30 < h < 62, (kind, h))
            h, s, l = mean(kind, "green")
            self.assertTrue(62 < h < 130, (kind, h))
            rgb = {p: [leafshapes.make_sprite(kind, p, v, 72).rgb for v in range(3)] for p in ("green", "turning")}
            warm = {p: np.mean([c[0] / c[1] for c in cs]) for p, cs in rgb.items()}
            self.assertGreater(warm["turning"], warm["green"] + 0.1, kind)
        hw, _, _ = mean("willow", "green")
        hm, _, _ = mean("maple", "green")
        self.assertLess(hw, hm)  # (a willow's green: more yellow)
        _, sat_back, l_back = mean("willow", "green", "back")
        _, sat_front, l_front = mean("willow", "green", "front")
        self.assertLess(sat_back, 0.7 * sat_front)
        self.assertGreater(l_back, l_front + 0.1)
        for v in range(leaves.VARIANTS["petal"]):
            s = leafshapes.make_sprite("petal", "red", v, 144)
            (hf, sf, lf), (_, sb, lb) = hsv(s.front[0]), hsv(s.back[0])
            self.assertGreater(sf.mean(), 0.15, v)
            self.assertGreater(sb.mean(), 0.15, v)
            self.assertLess(abs(lb.mean() / lf.mean() - 1), 0.05, v)
            self.assertTrue(((hf > 300) | (hf < 15)).mean() > 0.8, v)  # (pink)

    def test_petals_glow_pink(self):
        """A cherry petal lit from behind glows brighter and stays pink (no peach or cream)."""
        def mean_hue(img):
            h, s, l = hsv(img)
            return math.degrees(math.atan2(np.sin(np.radians(h)).mean(), np.cos(np.radians(h)).mean())) % 360, l
        for v in range(leaves.VARIANTS["petal"]):
            spr = leafshapes.make_sprite("petal", "red", v, 144)
            (hf, lf), (hl, ll) = mean_hue(spr.front[0]), mean_hue(spr.lit[0])
            self.assertLess(abs((hl - hf + 180) % 360 - 180), 15, v)
            self.assertGreater(ll.mean(), lf.mean(), v)
            half = Image.blend(spr.front[0], spr.lit[0], 0.5)  # (as a falling petal glows)
            self.assertLess(abs((mean_hue(half)[0] - hf + 180) % 360 - 180), 15, v)

    def test_bang_leaves_in_winter(self):
        """Bàng leaves in winter: red ones from wine red through brick and orange to russet brown (not all one red),
        their thick midrib and veins showing lighter on the front; yellow ones a clear yellow, no darker band down the
        middle; the "Lá bàng mùa đông" mix has orange-yellow and a few green leaves among the red."""
        hues = []
        for v in range(leaves.VARIANTS["almond"]):
            design, _ = leafshapes.design_of("almond", "red", v)
            spr = leafshapes.make_sprite("almond", "red", v, 288)
            h, s, l = hsv(spr.front[0])
            hues.append(math.degrees(math.atan2(np.sin(np.radians(h)).mean(), np.cos(np.radians(h)).mean())))
            main = _main_veins(design, 288)
            front = np.asarray(spr.front[0], np.float32)[..., :3].sum(axis=2)
            near = (np.asarray(spr.front[0])[..., 3] > 250) & (main < 0.05)
            self.assertGreater(front[main > 0.6].mean(), 1.15 * front[near].mean(), v)
        self.assertGreater(max(hues) - min(hues), 18, hues)  # (each leaf's mean: wine red to orange)
        for v in range(leaves.VARIANTS["almond"]):
            a = np.asarray(leafshapes.make_sprite("almond", "yellow", v, 288).front[0], np.float32) / 255
            rgb = a[..., :3] / np.maximum(a[..., 3:], 1e-6)
            lum = rgb @ np.array([0.3, 0.55, 0.15], np.float32)
            band, rest = [], []
            for r in range(a.shape[0]):
                cols = np.flatnonzero(a[r, :, 3] > 0.9)
                if len(cols) >= 40:
                    c = (cols.min() + cols.max()) / 2
                    band += lum[r, cols[np.abs(cols - c) < 8]].tolist()
                    rest += lum[r, cols[np.abs(cols - c) >= 8]].tolist()
            self.assertGreater(np.mean(band), 0.98 * np.mean(rest), v)
            self.assertGreater(np.mean(rest), 0.55, v)
            hues.append(float(np.median(hsv(leafshapes.make_sprite("almond", "yellow", v, 144).front[0])[0])))
        self.assertGreater(max(hues) - min(hues), 25, hues)  # (red and yellow leaves together)
        fx = effects.make("leaves", dict(leaves.SCENES["almond"], amount=100), 3, 720, 24, 1920, 1080)
        shares = np.bincount(fx.family, minlength=3) / fx.n
        self.assertTrue(0.4 < shares[0] < 0.7 and shares[1] > 0.2 and shares[2] > 0.03, shares)

    def test_colour_varies_across_a_leaf(self):
        """No flat fills: across a leaf its colour and lightness vary (more than its grain alone would)."""
        for kind in leafshapes.SHAPES:
            for family in ("red", "yellow", "green"):
                h, s, l = hsv(leafshapes.make_sprite(kind, family, 1, 144).front[0])
                self.assertGreater(float(np.percentile(l, 90) - np.percentile(l, 10)), 0.06, (kind, family))

    def test_back_and_lit_from_behind(self):
        """The back is duller and lighter than the front (veins standing out); lit from behind, a leaf glows:
        brighter and warmer than its front, its veins darker than the blade beside them."""
        for kind in leafshapes.SHAPES[:-1]:
            for family in ("red", "yellow", "green"):
                spr = leafshapes.make_sprite(kind, family, 0, 144)
                a = np.asarray(spr.front[0], np.float32)[..., 3] / 255
                front, back, lit = (np.asarray(img[0], np.float32)[..., :3].sum(axis=(0, 1)) / max(a.sum(), 1)
                                    for img in (spr.front, spr.back, spr.lit))
                if kind != "ginkgo":  # (a ginkgo's back is much like its front)
                    self.assertGreater(back.mean(), front.mean(), (kind, family))
                luma = np.array([0.3, 0.55, 0.15])
                self.assertGreater(lit @ luma, front @ luma + 4, (kind, family))  # (0..255)
                self.assertGreater(lit[0] / max(lit[2], 1), front[0] / max(front[2], 1) * 0.98, (kind, family))
        design, rng = leafshapes.design_of("maple", "yellow", 2)
        spr = leafshapes.make_sprite("maple", "yellow", 2, 288)
        main = _main_veins(design, 288)
        lit = np.asarray(spr.lit[0], np.float32)[..., :3].sum(axis=2)
        front = np.asarray(spr.front[0], np.float32)[..., :3].sum(axis=2)
        near = (np.asarray(spr.front[0])[..., 3] > 250) & (main < 0.05)
        on = main > 0.6
        self.assertLess(lit[on].mean(), 0.85 * lit[near].mean())  # (lit: the veins dark)
        self.assertGreater(front[on].mean(), 0.97 * front[near].mean())  # (in front: no darker than the blade)

    def test_every_design_differs(self):
        seen = set()
        for kind, count in leaves.VARIANTS.items():
            for v in range(count):
                s = leafshapes.make_sprite(kind, "red", v, 72)
                seen.add(np.asarray(s.front[0])[..., 3].tobytes())
        self.assertEqual(len(seen), sum(leaves.VARIANTS.values()))

    def test_crisp_edges_no_halo(self):
        """The edge is antialiased over about a pixel at every size (no hard steps, no smudge), and never brighter
        than what it covers (premultiplied colour no more than its coverage: no light rim on a dark ground)."""
        for kind in leafshapes.SHAPES:
            spr = leafshapes.make_sprite(kind, "yellow", 0, 288)
            for img in spr.front + spr.back + spr.lit:
                a = np.asarray(img, np.int32)
                self.assertTrue((a[..., :3] <= a[..., 3:4] + 1).all(), (kind, img.size))
            a = np.asarray(spr.front[0], np.float32)[..., 3] / 255
            part = ((a > 0.02) & (a < 0.9 * a.max())).sum()
            edge = np.abs(np.diff(a > 0.5, axis=0)).sum() + np.abs(np.diff(a > 0.5, axis=1)).sum()
            self.assertTrue(0.6 < part / edge < 2.6, (kind, part / edge))

    def test_pictures_kept_within_bounds(self):
        """The pictures kept take a bounded share of memory (the least recently used go first); one clip draws from
        no more designs than fit in it at once, however many shapes are ticked (none is made again and again)."""
        leaves._SPRITES.clear()
        most = leafshapes.MAX_BYTES
        try:
            for params, size in ((only(*leaves.SHAPES, amount=100, big=100), (3840, 2160)), ({}, (1920, 1080)),
                                 (dict(only(*leaves.SHAPES), **leaves.LOOKS["many"]), (1920, 1080))):
                fx = effects.make("leaves", params, 3, 720, 24, *size)
                pictures = {}
                for t in range(0, 720, 3):
                    for i, d in enumerate(fx.design(t)):
                        key = (fx.kind[i], fx.palette[i] if fx.kind[i] != "petal" else 0, d)
                        pictures[key] = max(pictures.get(key, 0), fx.master[i])
                self.assertEqual(pictures, fx.sizes, params)  # (each design made once, as large as needed)
                self.assertLessEqual(sum(map(leafshapes.sprite_bytes, pictures.values())),
                                     leaves.CLIP_SHARE * most, params)
                self.assertTrue((fx.count >= (2 if size[0] < 2000 else 1)).all(), params)  # (4K: their largest)
            single = effects.make("leaves", only("maple", amount=100), 3, 720, 24, 1920, 1080)
            self.assertTrue((single.count == leaves.VARIANTS["maple"]).all())  # (one shape: every design)
            leafshapes.MAX_BYTES = 4 << 20  # (less room: the least recently used go)
            for v in range(30):
                leafshapes.sprite("maple", leaves.FAMILIES[v % 3], v % leaves.VARIANTS["maple"], 144)
                leafshapes.sprite("maple", "red", 0, 144)  # (used again and again: kept)
            self.assertLessEqual(sum(s.bytes for s in leaves._SPRITES.values()), leafshapes.MAX_BYTES)
            self.assertIn(("maple", "red", 0, 144), leaves._SPRITES)
            self.assertIn(("maple", leaves.FAMILIES[29 % 3], 29 % leaves.VARIANTS["maple"], 144), leaves._SPRITES)
            self.assertEqual(leafshapes.sprite_bytes(288), leafshapes.make_sprite("oak", "red", 0).bytes)
        finally:
            leafshapes.MAX_BYTES = most
            leaves._SPRITES.clear()

    def test_several_clips_share_the_pictures(self):
        """Three leaves clips drawn together (as one export worker draws them) fit their pictures in memory at once:
        over a whole loop, twice, no picture is made twice (here in less room, and small)."""
        most, made = leafshapes.MAX_BYTES, []
        make = leafshapes.make_sprite
        leaves._SPRITES.clear()
        try:
            leafshapes.MAX_BYTES = 6 << 20
            leafshapes.make_sprite = lambda *a: made.append(a) or make(*a)
            clips = [effects.make("leaves", params, seed, 720, 24, 640, 360) for seed, params in
                     ((1, {"amount": 100}), (2, only("ginkgo", amount=100)), (3, only("almond", "oak", amount=100)))]
            for fx in clips:  # (each within its share, and with more than one design where it can)
                self.assertLessEqual(sum(map(leafshapes.sprite_bytes, fx.sizes.values())),
                                     leaves.CLIP_SHARE * leafshapes.MAX_BYTES)
            self.assertGreater(int(clips[0].count.min()), 1)
            canvas = effects.Canvas(640, 360)
            for _ in range(2):
                for t in range(0, 720, 12):
                    for fx in clips:
                        fx.draw(t, canvas, None)
                        canvas.reset()
            self.assertEqual(len(made), len(set(made)))
            self.assertLessEqual(sum(s.bytes for s in leaves._SPRITES.values()), leafshapes.MAX_BYTES)
            self.assertGreater(len(made), 20)
        finally:
            leafshapes.MAX_BYTES, leafshapes.make_sprite = most, make
            leaves._SPRITES.clear()

    def test_no_holes(self):
        """No design of any shape, in any colour, has a hole through its blade (two lobes' points shutting in a bit
        of a notch, say), at the largest pictures; the blades themselves hardly ever do (a pixel or two at most,
        filled)."""
        n = 288
        for kind in leafshapes.SHAPES:
            for family in (("red",) if kind == "petal" else leaves.PALETTES):
                for v in range(leaves.VARIANTS[kind]):
                    design, _ = leafshapes.design_of(kind, family, v)
                    *_, alpha, _, filled = leafshapes._blade_picture(design, n)
                    self.assertFalse(leafshapes.enclosed(alpha < 0.999).any(), (kind, family, v))
                    self.assertLessEqual(filled, 3, (kind, family, v))
        # the fill itself: a hole shut in by the blade is found, a notch open to the edge is not
        bg = np.ones((12, 12), bool)
        bg[2:9, 2:10] = False
        bg[4, 4] = bg[5, 6] = bg[6, 6] = True  # (two holes)
        bg[3:6, 9] = True  # (a notch from the edge)
        self.assertEqual(np.argwhere(leafshapes.enclosed(bg)).tolist(), [[4, 4], [5, 6], [6, 6]])


def _main_veins(design, n):
    """The main veins of a design as drawn into its pictures (n px), 0..1."""
    x0, y0, x1, y1 = leafshapes._bounds(design)
    n0 = 64
    ppu0 = (n0 - 4) / max(x1 - x0, y1 - y0)
    gx, gy = leafshapes._grid(n0, (x0 + x1) / 2, (y0 + y1) / 2, ppu0)
    a0 = np.clip(0.5 - leafshapes._blade(design, gx, gy)[0] * ppu0, 0, 1)
    cx, cy = float((a0 * gx).sum() / a0.sum()), float((a0 * gy).sum() / a0.sum())
    half = max(cx - x0, x1 - cx, cy - y0, y1 - cy) + 0.02
    ppu = (n / 2 - 2) / half
    from PIL import Image, ImageDraw
    big = n * leafshapes.SS
    img = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(img)
    for line in leafshapes._veins(design)[0]:
        leafshapes._poly_stroke(draw, line, ppu * leafshapes.SS, cx - n / 2 / ppu, cy + n / 2 / ppu, 255)
    return np.asarray(img.resize((n, n), Image.Resampling.BOX), np.float32) / 255


class Drawing(unittest.TestCase):
    def test_long_side_never_enlarged(self):
        """Whatever its slant, a leaf is drawn from a picture at least as long as it is seen (no blur that comes and
        goes as it turns); seen edge-on it is a solid line, no gaps."""
        spr = leafshapes.sprite("maple", "red", 0, 288)
        for length in (30, 70, 95, 150, 250):
            img = spr.image(False, length)
            self.assertGreaterEqual(spr.extent * img.size[0] / spr.size, min(length, spr.extent) - 1e-6)
        for squash in (0.05, 0.1, 0.3):
            out, x0, y0 = leaves.place(spr, False, 120, 0.4, 0.0, squash, 200.0, 200.0, 2)
            a = np.asarray(out, np.float32)[..., 3] / 255
            # across the leaf (squashed up-down in the picture, so the line is its width, turned by the roll):
            # every step of the way covered
            along = [a[int(round(200 - y0 + t * np.sin(0.4))), int(round(200 - x0 + t * np.cos(0.4)))]
                     for t in np.linspace(-30, 30, 61)]
            self.assertGreater(min(along), 0.2, squash)
        # in one step or two (squashed more than SQUASHED), the same leaf: alike either side of the switch
        one = leaves.place(spr, False, 120, 0.3, 0.6, leaves.SQUASHED + 0.001, 150.0, 150.0, 2)
        two = leaves.place(spr, False, 120, 0.3, 0.6, leaves.SQUASHED - 0.001, 150.0, 150.0, 2)
        self.assertEqual(one[1:], two[1:])
        a, b = (np.asarray(p[0], np.float32) / 255 for p in (one, two))
        self.assertLess(float(np.abs(a - b).mean()), 0.01)
        self.assertLess(abs(float(a[..., 3].sum() / b[..., 3].sum()) - 1), 0.01)
        # far leaves a little softer: drawn from a picture up to a third smaller, never more
        img = spr.image(False, 60 * 0.72)
        self.assertGreaterEqual(spr.extent * img.size[0] / spr.size, 60 * 0.72 - 1e-6)

    def test_every_leaf_drawn_large_enough(self):
        """Every leaf on screen comes from pictures at least as large as it is seen, whatever its shape (a ginkgo's
        long stem or a willow's narrow blade do not shrink it): made that large at once (its shape's SHARE holds for
        every design), never a smaller picture first."""
        fx = effects.make("leaves", only(*leaves.SHAPES, amount=100, big=100), 3, 144, 24, 1920, 1080)
        for i in np.flatnonzero(fx.longest > 50):
            for d in range(int(fx.count[i])):
                spr = fx.picture(i, d)
                self.assertTrue(spr.extent >= fx.longest[i] - 1e-6 or spr.size == leaves.MIPS[0], (fx.kind[i], d))
        for kind in leaves.SHAPES:
            for v in range(leaves.VARIANTS[kind]):
                for family in (("red",) if kind == "petal" else leaves.PALETTES):
                    spr = leafshapes.make_sprite(kind, family, v, 72)
                    self.assertGreaterEqual(spr.extent / spr.size, leaves.SHARE[kind], (kind, family, v))

    def test_light(self):
        """Face on, a leaf is about as bright as its picture; turned to the light in front, brighter; turned away,
        darker; the sun behind it makes it glow most when it faces the camera from in front of the sun, not at all
        at "Nắng chiếu xuyên lá" 0."""
        fx = effects.make("leaves", {"glow": 100, "amount": 100}, 3, 144, 24, 1920, 1080)
        fx.chi = np.zeros(fx.n)
        fx.sun = np.ones(fx.n)
        fx.sheen = np.zeros(fx.n)
        zero = np.zeros(fx.n)
        lit, glow = fx.light(zero, zero)  # face on
        self.assertTrue(np.allclose(lit, 0.75 + 0.35 * 0.72, atol=0.01))
        self.assertTrue((glow > 0.5).all())
        towards, _ = fx.light(zero, np.full(fx.n, -0.6))  # (its face turned up, to the light above)
        away, glow_away = fx.light(zero, np.full(fx.n, 0.6))
        self.assertTrue((towards > lit + 0.03).all() and (away < lit - 0.03).all())
        edge_lit, edge_glow = fx.light(zero, np.full(fx.n, math.pi / 2))
        self.assertTrue((edge_glow < glow).all())
        turned, turned_glow = fx.light(zero, np.full(fx.n, math.pi))  # (its back seen: lit the same)
        self.assertTrue(np.allclose(turned, lit) and np.allclose(turned_glow, glow))
        none = effects.make("leaves", {"glow": 0, "amount": 100}, 3, 144, 24, 1920, 1080)
        self.assertEqual(float(np.abs(none.light(zero, zero)[1]).max()), 0.0)

    def test_backlight_makes_leaves_glow(self):
        """At "Nắng chiếu xuyên lá" 100 the leaves are brighter and warmer on the whole than at 0."""
        a = effects.make("leaves", {"glow": 0, "amount": 100}, 3, 144, 24, 640, 360)
        b = effects.make("leaves", {"glow": 100, "amount": 100}, 3, 144, 24, 640, 360)
        pa, pb = frame(a, 20, 640, 360, base=0.0), frame(b, 20, 640, 360, base=0.0)
        self.assertGreater(pb.sum(), 1.05 * pa.sum())
        self.assertGreater(pb[..., 0].sum() / pb[..., 2].sum(), pa[..., 0].sum() / pa[..., 2].sum())

    def test_smear(self):
        """Motion blur spreads a leaf evenly along the way it moves: nothing lost, its middle where it was, as long
        as the way, its edges along the way smooth (no zipper); a long smear costs about what a short one does."""
        arr = np.zeros((40, 30, 4), np.uint8)
        arr[15:25, 10:20] = 255
        img = Image.fromarray(arr, "RGBa")
        for dx, dy in ((0, 12), (5, 20), (-30, 8), (17.5, -17.2), (0.8, 0.9), (9.3, 21.7)):
            out, x0, y0 = leaves.smear(img, dx, dy)
            a = np.asarray(out, np.float32)[..., 3] / 255
            self.assertLess(abs(float(a.sum()) / 100.0 - 1), 0.02, (dx, dy))
            h, w = a.shape
            cx = float((a * np.arange(w)).sum() / a.sum()) + x0
            cy = float((a * np.arange(h)[:, None]).sum() / a.sum()) + y0
            self.assertLess(abs(cx - 14.5) + abs(cy - 19.5), 0.15, (dx, dy))
            ys, xs = np.nonzero(a > 0.004)
            self.assertLess(abs(np.ptp(xs) + 1 - 10 - abs(dx)) if abs(dx) > abs(dy) else
                            abs(np.ptp(ys) + 1 - 10 - abs(dy)), 3, (dx, dy))
        # a disc moving along a slant: as the many copies of it along the way, each moved by a fraction of a pixel,
        # would make it (whole-pixel shears were up to 0.24 off at its edges: a fine zipper along them)
        yy, xx = np.mgrid[0:60, 0:60]
        disc = np.clip(20.5 - np.hypot(xx - 29.5, yy - 29.5), 0, 1).astype(np.float32)
        for dx, dy in ((7.3, 17.9), (3.1, 12.4), (-11.0, 14.5), (17.5, -17.2)):
            out, x0, y0 = leaves.smear(Image.fromarray((np.dstack([disc] * 4) * 255).astype(np.uint8), "RGBa"), dx, dy)
            got = np.asarray(out, np.float32)[..., 3] / 255
            h, w = got.shape
            want = np.zeros((h, w), np.float32)
            for t in (np.arange(200) + 0.5) / 200 - 0.5:
                want += np.asarray(Image.fromarray(disc, "F").transform(
                    (w, h), Image.Transform.AFFINE, (1, 0, x0 - t * dx, 0, 1, y0 - t * dy),
                    resample=Image.Resampling.BILINEAR))
            want /= 200
            self.assertLess(float(np.abs(got - want).max()), 0.1, (dx, dy))
        big = Image.fromarray(np.random.default_rng(1).integers(0, 256, (200, 200, 4), np.uint8), "RGBa")

        def cost(dx, dy):  # a second per million px of what it makes
            t0 = time.perf_counter()
            for _ in range(5):
                out = leaves.smear(big, dx, dy)[0]
            return (time.perf_counter() - t0) / (5 * out.size[0] * out.size[1]) * 1e6
        cost(4, 6)
        self.assertLess(cost(60, 90), 3 * cost(4, 6))  # (not 24 copies against 7)


class Speed(unittest.TestCase):
    @staticmethod
    def cost(kind, params, size=(1920, 1080)):
        fx = effects.make(kind, params, 3, 720, 24, *size)
        canvas = effects.Canvas(*size)
        pic = np.full((size[1], size[0], 3), 0.3, np.float32)
        for t in range(0, 720, 40):  # (their pictures made first, as for a video)
            fx.draw(t, canvas, None)
            canvas.flush(pic, fx.blend)
        best = []
        for t in range(5, 720, 90):
            t0 = time.perf_counter()
            fx.draw(t, canvas, None)
            canvas.flush(pic, fx.blend)
            best.append(time.perf_counter() - t0)
        return float(np.median(best))

    def test_a_1080p_frame_costs_about_as_much_as_snow(self):
        """At the defaults a 1080p frame of leaves takes no more than about twice what one of snow takes (measured:
        the leaves before their shapes took 55 ms, now 49 ms; snow about 25 ms); every shape at once about as much
        as the defaults."""
        snow, leaf = self.cost("snow", {}), self.cost("leaves", {})
        self.assertLess(leaf, 2.5 * snow + 0.01, (leaf, snow))
        windy = self.cost("leaves", dict(leaves.LOOKS["windy"], motion=100))  # (many leaves, fast, smeared)
        self.assertLess(windy, 4 * snow + 0.01, (windy, snow))
        every = self.cost("leaves", only(*leaves.SHAPES))
        self.assertLess(every, 1.4 * leaf + 0.005, (every, leaf))

    def test_every_kind_of_leaves_costs_about_as_much(self):
        """Each "Loại lá" at 1080p costs no more than about one and a half times the defaults (a ginkgo's long stem
        or a willow's narrow blade: only the box round the leaf is drawn). Measured: defaults 44 ms, ginkgo 67 (113
        when its whole square picture was drawn), birch 61, heart 59, the others 36-52 (bound: 1.75x, for noise)."""
        leaf = self.cost("leaves", {})
        for name in leaves.SCENES:
            leaves._SPRITES.clear()  # (each on its own, as one clip renders)
            cost = self.cost("leaves", dict(leaves.SCENES[name], scene=name))
            self.assertLess(cost, 1.75 * leaf + 0.005, (name, cost, leaf))

    def test_pictures_made_quickly(self):
        """The pictures a 1080p clip needs are made in a few seconds, however many shapes are ticked (measured: 3 s
        at the defaults, 2 s with every shape)."""
        for params in ({}, only(*leaves.SHAPES)):
            leaves._SPRITES.clear()
            t0 = time.perf_counter()
            fx = effects.make("leaves", params, 3, 720, 24, 1920, 1080)
            for i in range(fx.n):
                for j in range(int(min(fx.k[i], fx.count[i]))):
                    fx.picture(i, int((fx.variant[i] + 3 * j) % fx.count[i]))
            self.assertLess(time.perf_counter() - t0, 12, params)
        leaves._SPRITES.clear()


if __name__ == "__main__":
    unittest.main()
