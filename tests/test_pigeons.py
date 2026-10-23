"""Pigeons (fx/pigeons.py): a placed effect of rock doves drawn by code, walking about and pecking on the ground or on
a ledge, or flying round in a flock or past, lit by the scene where their box stands."""
import math
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects  # (first: it loads every effect's module)
from fx import engine, overlay, pigeons

W, H = 480, 270
AT = {"scale": 90, "x": 0, "y": 0}
MODES = ("ground", "ledge", "flock", "pass")
DAY = {"lit": 0}  # (lit by daylight, not by a scene)


def make(params=None, frames=240, fps=24, w=W, h=H, seed=3, light=None):
    return effects.make("pigeons", {**AT, **(params or {})}, seed, frames, fps, w, h, light)


def drawn(fx, t):
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, None)
    return canvas.rgb.copy(), canvas.a.copy()


def lowest(part):
    """The lowest point of a part (an ellipsoid: name, centre, rotation, radii), above the ground."""
    _, c, rot, r = part
    return float(c[2] - math.sqrt(float(((rot[2, :] * r) ** 2).sum())))


def walks(walker, least=2.5):
    """The walks of a walker's day long enough to be at full speed in their middle (start, length: seconds)."""
    return [(start, d) for kind, start, d in walker.segs if kind == "walk" and d > least]


class Loop(unittest.TestCase):
    def test_frame_p_is_frame_0(self):
        for mode in MODES:
            for frames, fps in ((240, 24), (150, 25)):
                fx = make({"mode": mode, "amount": 70}, frames, fps)
                a, b = drawn(fx, 0), drawn(fx, frames)
                self.assertLess(float(np.abs(a[0] - b[0]).max()), 1e-6, (mode, frames, fps))
                self.assertLess(float(np.abs(a[1] - b[1]).max()), 1e-6, (mode, frames, fps))

    def test_they_move(self):
        for mode in MODES:
            fx = make({"mode": mode, "amount": 70, "busy": 100})
            a = sum(np.abs(drawn(fx, 0)[1] - drawn(fx, t)[1]).max() for t in (6, 30, 90))
            self.assertGreater(float(a), 0.5, mode)

    def test_each_walker_comes_home(self):
        """A bird on the ground goes round a closed path of its own, as far as it walks over the loop (once round,
        or more often round a smaller path): at the loop's end where it started, facing the same way. Along a ledge
        its path goes no farther from home than its room, however long it walks, and turns round at its ends in a
        narrow half circle (TURN)."""
        for seed in range(6):
            for busy in (0.0, 0.5, 1.0):
                for loop, ledge, room in ((20.0, False, None), (20.0, False, 0.8), (20.0, True, 2.0),
                                          (60.0, True, 1.0)):
                    walker = pigeons.Walker(np.random.default_rng(seed), loop, busy, 1.0, ledge, room)
                    x0, y0, a0 = walker.where(0.0)
                    x1, y1, a1 = walker.where(loop - 1e-9)
                    self.assertLess(math.hypot(x1 - x0, y1 - y0), 1e-3, (seed, busy, ledge))
                    self.assertLess(abs(math.remainder(a1 - a0, math.tau)), 0.05, (seed, busy, ledge))
                    if ledge:
                        self.assertLessEqual(float(np.abs(walker.px).max()), room * 1.01, (seed, busy, loop))
                        self.assertLessEqual(float(np.abs(walker.py).max()), pigeons.TURN * 1.01, (seed, busy, loop))


class Walking(unittest.TestCase):
    """As a pigeon walks: its planted foot stays put on the ground, its head is held still and thrust forward a
    step at a time; as it pecks, its beak reaches the ground."""

    def test_a_planted_foot_stays_put(self):
        walker = pigeons.Walker(np.random.default_rng(4), 30.0, 0.8, 1.0)
        start, d = walks(walker)[0]
        for side in (0, 1):
            ground = []
            for t in np.linspace(start + 0.4, start + d - 0.4, 400):
                foot = walker.pose(t).feet[side]
                if foot[2] == 0.0:  # (planted: where it is on the ground, along the way the bird goes)
                    ground.append(walker.walked_at(t) + foot[0])
                else:
                    ground.append(None)
            runs, run = [], []
            for g in ground + [None]:
                if g is None:
                    if len(run) > 5:
                        runs.append(run)
                    run = []
                else:
                    run.append(g)
            self.assertGreater(len(runs), 2)
            for run in runs:
                self.assertLess(max(run) - min(run), 0.004, side)  # (units: a bird is 0.86 long)

    def test_its_feet_take_turns(self):
        walker = pigeons.Walker(np.random.default_rng(4), 30.0, 0.8, 1.0)
        start, d = walks(walker)[0]
        for t in np.linspace(start + 0.4, start + d - 0.4, 200):
            feet = walker.pose(t).feet
            self.assertTrue(feet[0][2] == 0.0 or feet[1][2] == 0.0, t)  # (always one on the ground)
            self.assertLess(max(abs(feet[0][0]), abs(feet[1][0])), pigeons.STRIDE / 4 + 1e-9)

    def test_its_head_held_still_then_thrust(self):
        """Its head stays nearly still on the ground's way most of the time, and goes forward fast a bob a step (two
        a stride)."""
        walker = pigeons.Walker(np.random.default_rng(4), 30.0, 0.8, 1.0)
        start, d = walks(walker)[0]
        ts = np.linspace(start + 0.5, start + d - 0.5, 2000)
        head = np.array([walker.walked_at(t) + walker.pose(t).head[0] for t in ts])
        body = np.array([walker.walked_at(t) for t in ts])
        speed, at = np.diff(head) / np.diff(ts), np.diff(body) / np.diff(ts)
        still = np.abs(speed) < 0.2 * at.mean()
        self.assertGreater(float(still.mean()), 0.55)
        thrusts = int(np.sum(np.diff(still.astype(int)) == -1))
        steps = (body[-1] - body[0]) / (pigeons.STRIDE / 2)
        self.assertLessEqual(abs(thrusts - steps), 2.0, (thrusts, steps))

    def test_its_beak_reaches_the_ground(self):
        for seed in range(4):
            walker = pigeons.Walker(np.random.default_rng(seed), 30.0, 0.2, 1.0)
            pecks = [(start, d) for kind, start, d in walker.segs if kind == "peck"]
            self.assertTrue(pecks, seed)
            start, d = pecks[0]
            low = min(lowest([p for p in pigeons.parts(walker.pose(t)) if p[0] == "beak"][0])
                      for t in np.linspace(start, start + d, 600))
            self.assertLess(low, 0.01, seed)
            self.assertGreater(low, -0.01, seed)  # (not into the ground)

    def test_standing_on_its_feet(self):
        parts = pigeons.parts(pigeons.Pose())
        self.assertAlmostEqual(min(lowest(p) for p in parts), 0.0, places=6)
        self.assertEqual([p[0] for p in parts if abs(lowest(p)) < 1e-6], ["toe", "toe"])


class Picture(unittest.TestCase):
    def test_only_in_its_box(self):
        """Nothing outside the box (its preview clip is the box alone): the birds on the ground walk in it, a flock
        goes round in it, birds passing come in and go out at its sides."""
        for mode in MODES:
            for params in ({}, {"amount": 100, "size": 100, "depth": 100, "busy": 100, "speed": 100},
                           {"amount": 100, "size": 100, "x": 30, "y": -20, "scale": 40}):
                fx = make({"mode": mode, **params})
                x0, y0, w, h = fx.box()
                outside = np.ones((H, W), bool)  # (the pixels the box's edges cross are partly inside)
                outside[max(0, math.floor(y0)):math.ceil(y0 + h), max(0, math.floor(x0)):math.ceil(x0 + w)] = False
                for t in (0, 50, 133, 201):
                    self.assertEqual(float(drawn(fx, t)[1][outside].max(initial=0.0)), 0.0, (mode, params, t))

    def test_a_flock_is_whole_in_its_box(self):
        """A flock's loop is made small enough for its box: the box's edges never cut a bird of it."""
        cut = []
        real = pigeons._inside

        def inside(ix, iy, shape, x0, y0, w, h):
            out = real(ix, iy, shape, x0, y0, w, h)
            if out is not None and out.min() < 1:
                cut.append((ix, iy))
            return out
        pigeons._inside = inside
        try:
            for params in ({}, {"amount": 100, "size": 100, "depth": 100}):
                fx = make({"mode": "flock", **params})
                for t in range(0, 240, 20):
                    drawn(fx, t)
        finally:
            pigeons._inside = real
        self.assertEqual(cut, [])

    def test_how_many(self):
        for mode in MODES:
            self.assertEqual(len(make({"mode": mode, "amount": 0}).u), 0)
            self.assertEqual(float(drawn(make({"mode": mode, "amount": 0}), 5)[1].max()), 0.0)
            counts = [len(make({"mode": mode, "amount": a}).u) for a in (10, 35, 80, 100)]
            self.assertEqual(counts, sorted(counts), mode)
            self.assertGreater(counts[0], 0, mode)
            self.assertEqual(counts[-1], pigeons.CAP[mode], mode)

    def test_their_coats(self):
        """A white dove is pale, a dark one dark, a brown one reddish, a rock dove grey with a green neck."""
        def colour(coat):
            fx = make({"coat": coat, "amount": 60, "size": 100, **DAY}, w=960, h=540)
            rgb, a = drawn(fx, 0)
            return rgb.reshape(-1, 3).sum(axis=0) / max(float(a.sum()), 1e-9)
        white, grey, dark, brown = (colour(c) for c in ("white", "grey", "dark", "brown"))
        luma = np.array([0.2126, 0.7152, 0.0722])
        self.assertGreater(white @ luma, grey @ luma)
        self.assertGreater(grey @ luma, dark @ luma)
        self.assertGreater(brown[0], brown[2] * 1.1)
        self.assertLess(abs(grey[0] - grey[2]), 0.12)  # (grey: not red, not blue)
        coat = pigeons.Coat("grey", np.random.default_rng(0))
        self.assertGreater(coat.c["green"][1], coat.c["green"][0] * 1.3)  # (its sheen)

    def test_lit_by_the_scene(self):
        """In a dark scene the birds are darker, in a warm one warmer; with "Sáng theo cảnh" 0 the scene does not
        light them."""
        dark = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.0], np.float32)
        warm = np.array([1.0, 0.6, 0.3, 0.5, 0.35, 0.2, 0.0], np.float32)

        def colour(params, light):
            fx = make({"mode": "ground", "amount": 60, "size": 90, **params}, w=960, h=540, light=light)
            rgb, a = drawn(fx, 30)
            return rgb.reshape(-1, 3).sum(axis=0) / max(float(a.sum()), 1e-9)
        day = colour(DAY, None)
        self.assertLess(colour({"lit": 100}, dark).sum(), day.sum() * 0.3)
        hot = colour({"lit": 100}, warm)
        self.assertGreater(hot[0] / hot[2], day[0] / day[2] * 1.3)
        self.assertLess(float(np.abs(colour(DAY, dark) - day).max()), 1e-6)

    def test_far_ones_smaller(self):
        """On the ground, the farther up the box (the smaller their feet's y) the smaller ("Gần xa"); with no depth,
        all alike."""
        for depth, smaller in ((100, True), (0, False)):
            fx = make({"amount": 100, "depth": depth})
            x0, y0, w, h = fx.box()
            birds = fx._ground(0, x0, y0, w, h)
            ys, scales = np.array([b[1] for b in birds]), np.array([b[2] for b in birds])
            if smaller:
                self.assertGreater(float(np.corrcoef(ys, scales)[0, 1]), 0.8)
                self.assertGreater(float(scales.max() / scales.min()), 1.4)
            else:
                self.assertLess(float(np.ptp(scales)), 1e-9)

    def test_a_ledge_is_a_line(self):
        """On a ledge they stand on one line near the box's bottom edge, walking to and fro along it."""
        fx = make({"mode": "ledge", "amount": 100})
        x0, y0, w, h = fx.box()
        for t in (0, 77, 160):
            feet = np.array([b[1] for b in fx._ground(t, x0, y0, w, h)])
            self.assertLess(float(np.ptp(feet)), 0.04 * h, t)
            self.assertGreater(float(feet.min()), y0 + 0.9 * h, t)

    def test_passing_birds_cross_the_box(self):
        """Birds passing go across the box, each way as chosen, facing the way they go."""
        for way, heading in (("right", 0.0), ("left", math.pi)):
            fx = make({"mode": "pass", "way": way, "amount": 50}, 480, 24)
            x0, y0, w, h = fx.box()
            xs = np.array([[b[0] for b in fx._flying(t, x0, y0, w, h)] for t in range(0, 480, 2)])
            step = np.diff(xs, axis=0)
            step = step[np.abs(step) < 0.5 * w]  # (not where one starts again at the other side)
            self.assertTrue(np.all(step > 0) if way == "right" else np.all(step < 0), way)
            self.assertLess(float(xs.min()), x0)
            self.assertGreater(float(xs.max()), x0 + w)
            for b in fx._flying(33, x0, y0, w, h):
                self.assertLess(abs(math.remainder(b[3] - heading, math.tau)), 0.2, way)

    def test_a_flock_goes_round(self):
        """A flock wheels round together: the flock across much of the box, each of its birds round (the inner ones
        round a smaller loop), with no jumps."""
        fx = make({"mode": "flock", "amount": 50}, 480, 24)
        x0, y0, w, h = fx.box()
        xs = np.array([[b[0] for b in fx._flying(t, x0, y0, w, h)] for t in range(0, 480, 4)])
        self.assertGreater(float(np.ptp(xs.mean(axis=1))), 0.4 * w)
        self.assertGreater(float(np.ptp(xs, axis=0).min()), 0.1 * w)
        step = np.abs(np.diff(xs, axis=0))
        self.assertLess(float(step.max()), 0.05 * w)


class Preview(unittest.TestCase):
    def test_preview_clip_is_the_box_alone(self):
        """The preview clip (the box alone, at the clip's size) is the box in the whole frame, at another size."""
        from PIL import Image
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 240}
        for mode in ("ground", "flock"):
            params = {**AT, "mode": mode, "amount": 50, **DAY}
            layer = {"kind": "pigeons", "params": params}
            cw, ch = engine.fx_clip_frame(info, "pigeons", layer)
            self.assertAlmostEqual(cw / ch, 16 / 9, places=1)
            whole = make(params, 240, 24, 1920, 1080)
            alone = effects.make("pigeons", engine.fx_clip_params("pigeons", params), 3, 240, 24, cw, ch)
            x0, y0, bw, bh = whole.box()
            for t in (0, 77):
                big, clip = drawn(whole, t)[1], drawn(alone, t)[1]
                part = np.asarray(Image.fromarray((big * 255 + 0.5).astype(np.uint8)).resize(
                    (cw, ch), Image.Resampling.BOX, box=(x0, y0, x0 + bw, y0 + bh)), np.float32) / 255
                self.assertLess(float(np.abs(part - clip).mean()), 0.004, (mode, t))
                self.assertGreater(float(clip.mean()), 0.003, (mode, t))

    def test_drawn_over_the_picture(self):
        cls = pigeons.Pigeons
        self.assertEqual((cls.blend, cls.lens, cls.TURNS, cls.BOX, cls.LIT), ("over", False, False, (16, 9), True))
        project = {"video": {"w": 1920, "h": 1080, "fps": "24", "frames": 144, "seconds": 6.0}, "length": 30}
        a = engine.fx_clip_name(project, {"kind": "pigeons", "params": DAY})
        self.assertTrue(a.endswith(".webm"))  # (with see-through, laid Normal)
        self.assertEqual(a, engine.fx_clip_name(project, {"kind": "pigeons", "params": {**DAY, "x": 12}}))  # (the box
        # alone)


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_label(self):
        keys = [p["key"] for p in pigeons.SCHEMA["params"]]
        self.assertEqual(len(keys), len(set(keys)))
        for p in pigeons.SCHEMA["params"]:
            self.assertTrue(p.get("group") and p["label"], p["key"])
        self.assertFalse(set(overlay.ROTATIONS) & set(keys))
        self.assertTrue(pigeons.SCHEMA["note"])
        self.assertEqual(effects.defaults("pigeons")["mode"], "ground")

    def test_shown_where_they_count(self):
        """Walking about, the view from above and the shadow: on the ground or a ledge; the way they fly: passing."""
        show = {p["key"]: p.get("show") for p in pigeons.SCHEMA["params"]}
        for key in ("busy", "view", "shadow"):
            self.assertEqual(show[key], {"mode": ["ground", "ledge"]}, key)
        self.assertEqual(show["way"], {"mode": ["pass"]})
        for key in ("mode", "coat", "amount", "size", "depth", "speed", "lit"):
            self.assertIsNone(show[key], key)
        modes = [o[0] if isinstance(o, (list, tuple)) else o["value"]
                 for o in next(p for p in pigeons.SCHEMA["params"] if p["key"] == "mode")["options"]]
        self.assertEqual(sorted(modes), sorted(MODES))

    def test_coats_and_views(self):
        """Each coat a picture of its own; the view from above changes how they are seen."""
        a = drawn(make({"coat": "white", **DAY}), 10)[0]
        b = drawn(make({"coat": "dark", **DAY}), 10)[0]
        self.assertGreater(float(np.abs(a - b).max()), 0.3)
        low = drawn(make({"view": 0, **DAY}), 10)[1]
        high = drawn(make({"view": 100, **DAY}), 10)[1]
        self.assertGreater(float(np.abs(low - high).max()), 0.3)


if __name__ == "__main__":
    unittest.main()
