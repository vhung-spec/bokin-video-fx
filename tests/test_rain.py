import math
import os
import time
import unittest

import numpy as np
from PIL import Image

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, engine, projects, rain, sheet, video

W, H = 320, 180
LOOKS = next(p for p in rain.SCHEMA["params"] if p["key"] == "look")["presets"]


def night(w=W, h=H):
    """A dark street at night with one warm lamp at the right (uint8 RGB)."""
    yy, xx = np.mgrid[0:h, 0:w]
    pic = np.zeros((h, w, 3), np.uint8) + np.array([14, 12, 11], np.uint8)
    lamp = (xx - 0.8 * w) ** 2 + (yy - 0.3 * h) ** 2 < (0.03 * h) ** 2
    pic[lamp] = (255, 205, 120)
    return pic


LIGHT = effects.light_map([night()])


def round_lamp(r, w=W, h=H):
    """Where the frame lies within r frame heights of night()'s lamp."""
    yy, xx = np.mgrid[0:h, 0:w]
    return (xx - 0.8 * w) ** 2 + (yy - 0.3 * h) ** 2 < (r * h) ** 2


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
             {"amount": 3, "gust": 100, "waves": 100}, everything(size=0, speed=0, length=0, ground=40)] + [
        dict(v, look=k) for k, v in LOOKS.items()]

    def test_frame_p_is_frame_0(self):
        """Every part (drops, gusts, waves, sheets, splashes, lightning) returns to frame 0 after one loop."""
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
            for params in self.CASES:
                fx = effects.make("rain", params, 3, frames, fps, W, H)
                diff = np.abs(frame(fx, 0) - frame(fx, frames)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")
                t = frames // 3  # and any frame, a loop later
                self.assertLess(np.abs(frame(fx, t) - frame(fx, t + frames)).max(), 1 / 512, f"{params} t={t}")

    def test_lit_frame_p_is_frame_0(self):
        """Lit by the scene (its light is still, so time drops out), every look and the strongest settings loop."""
        for params in [everything(lit=100), everything(lit=40, wind=-80), {"lit": 100}] + [
                dict(v, look=k, ground=60) for k, v in LOOKS.items()]:
            fx = effects.make("rain", params, 3, 150, 25, W, H, LIGHT)
            self.assertLess(np.abs(frame(fx, 0) - frame(fx, 150)).max(), 1 / 512, params)
            self.assertLess(np.abs(frame(fx, 61) - frame(fx, 211)).max(), 1 / 512, params)

    def test_each_pass_in_another_lane(self):
        """A drop comes down somewhere else each time (no streak runs down the same line twice a second), the same
        place a loop later; it changes lanes only out of sight, above or below the frame."""
        for look in ("medium", "storm"):
            fx = effects.make("rain", dict(LOOKS[look], look=look, gust=0), 3, 750, 25, W, H)
            for i in np.arange(0, fx.n, max(1, fx.n // 40)):
                k = int(fx.k[i])
                start = -fx.s0[i] / (fx.k[i] * fx.len[i])  # (when its pass 0 begins)
                passes = np.array([float(fx.lanes(start + (j + 0.5) / k, [i])[0]) for j in range(k)])
                self.assertEqual(len(np.unique(np.round(passes, 9))), k, (look, i))
                if k > 1:
                    step = np.abs(np.diff(np.append(passes, passes[0])))
                    self.assertGreater(float(step.min()), 0.3 * float(fx.room[i]), (look, i))
                self.assertAlmostEqual(float(fx.lanes(start + 1.5 / k, [i])[0]),
                                       float(fx.lanes(start + 1.5 / k + 1.0, [i])[0]))
            before = None
            for t in range(0, 300):
                lane = fx.lanes(t / 750)
                hx, hy, ux, uy, length, s, show = fx.drops(t)
                if before is not None:
                    moved = np.flatnonzero(np.abs(lane - before[0]) > 1e-9)
                    step = fx.k[moved] * fx.len[moved] / 750  # (how far along its path a drop goes in a frame)
                    self.assertTrue(np.all(s[moved] <= step * 1.0001), (look, t))  # just starting from above
                    # and it was at the end of the last one, going out below (the path's ends lie off the frame)
                    self.assertTrue(np.all(before[1][moved] >= fx.len[moved] - step * 1.0001), (look, t))
                before = lane, s

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
            lit, dark = frame(fx, t), frame(fx, (t + frames // 2) % frames)
            self.assertGreater(lit[:H // 3].mean(), dark[:H // 3].mean())
            # light from the sky: far more at the top than near the ground
            gain = (lit - dark).mean(axis=(1, 2))
            self.assertGreater(gain[:H // 6].mean(), 4 * gain[-H // 6:].mean())

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

    def test_sheets_go_with_the_wind_in_gusts(self):
        """The sheets of rain drift with the wind, faster in a gust, never back against it."""
        for wind in (40, -40):
            fx = effects.make("rain", {"look": "storm", "wind": wind, "gust": 75}, 3, 720, 24, W, H)
            xs = np.linspace(0, 4, 800, endpoint=False)  # one tile across (4 frame heights)
            step = xs[1] - xs[0]
            for t in range(0, 720, 3):
                a = fx.pattern(t / 720, xs, np.full_like(xs, 0.5))
                b = fx.pattern((t + 1) / 720, xs, np.full_like(xs, 0.5))
                lags = np.arange(-40, 41)
                best = lags[np.argmax([np.dot(a - a.mean(), np.roll(b, int(k)) - b.mean()) for k in lags])]
                moved = -best * step  # (b rolled back by what it moved matches a)
                self.assertGreaterEqual(moved * np.sign(wind), 0, f"wind {wind}, frame {t}: {moved:.3f}")

    def test_streaks_stop_at_the_ground(self):
        layer = rain.Layer(40, 60)
        one = np.array([1.0])
        layer.streaks(one * 10.3, one * 2.0, one * 12.1, one * 50.0, one * 1.0, one * 0.8, one * 30.0)
        cover = layer.view()
        self.assertEqual(float(cover[30:].max()), 0.0)  # nothing below the ground
        self.assertTrue(all(cover[r].max() > 0 for r in range(1, 30)))  # the rest of the streak is there
        self.assertEqual(float(cover[:1].max()), 0.0)  # (its round end reaches half its width past the tail)

    def test_streak_coverage(self):
        """A streak thinner than a pixel covers that share of it; even along its length (a drop at a steady speed
        while the shutter is open), round at both ends."""
        cover = []
        for width in (0.25, 0.5, 1.0, 2.0, 4.0):
            layer = rain.Layer(40, 130)
            one = np.array([1.0])
            layer.streaks(one * 20.4, one * 10.0, one * 20.4, one * 110.0, one * width, one * 1.0, one * 999.0)
            cover.append(layer.view())
        sums = [float(c.sum()) for c in cover]
        for a, b in zip(sums, sums[1:]):
            self.assertAlmostEqual(b / a, 2.0, delta=0.05)
        for c, width in zip(cover, (0.25, 0.5, 1.0, 2.0, 4.0)):
            rows = c.sum(axis=1)
            middle = rows[15:105]
            self.assertLess(float(middle.max() - middle.min()), 0.01 * float(middle.max()), width)  # even
            self.assertAlmostEqual(float(middle.mean()), width, delta=0.01 * width)
            # round ends: fading in before the tail and out past the head, as much there as an even end would be
            self.assertTrue(0 < rows[9] < 0.5 * width and 0 < rows[110] < 0.5 * width, width)
            self.assertAlmostEqual(float(rows[:15].sum()), 5 * width, delta=0.1 * width)
            self.assertAlmostEqual(float(rows[105:].sum()), 5 * width, delta=0.1 * width)
            self.assertEqual(float(rows[112 + int(width):].max()), 0.0)
        self.assertLessEqual(float(cover[2].max()), 1.0 + 1e-6)
        self.assertGreater(float(cover[4][50].max()), 0.99)  # a wide drop covers its middle

    def test_streaks_do_not_pop_at_the_sides(self):
        """A drop is dropped only once its whole streak (and its blur) is out of the frame: the frame is the same
        as one drawn with no drop left out, at 4K and with long, slanted, blurred streaks."""
        cases = [(3840, 2160, LOOKS["storm"], (100,)),
                 (1920, 1080, {"wind": -100, "length": 100, "speed": 100, "depth": 100, "gust": 100}, (100, 333)),
                 (1920, 1080, {"wind": 100, "length": 100, "size": 100, "depth": 100, "blur": 100}, (100, 333))]
        for w, h, params, times in cases:
            fx = effects.make("rain", params, 3, 750, 25, w, h)
            for t in times:
                culled = frame(fx, t, w=w, h=h)
                fx.in_view = lambda hx, ux, length, show: np.flatnonzero(show)
                every = frame(fx, t, w=w, h=h)
                del fx.in_view
                self.assertTrue(np.array_equal(culled, every), (w, params, t))

    def test_soft_streaks_have_soft_ends(self):
        """Out of focus, a streak is blurred along its length as across it: its head fades out past where it is,
        and it fades into the ground instead of stopping on a line."""
        one = np.array([1.0])
        for cut, end in ((999.0, 150), (120.0, 120)):
            layer = rain.Layer(80, 200)
            layer.streaks(one * 40, one * 20, one * 40, one * 150, one * 10, one * 1.0, one * cut, soft=True)
            rows = layer.view().sum(axis=1)
            self.assertLess(float(np.abs(np.diff(rows)).max()), 0.15 * rows.max(), cut)  # no steps
            self.assertGreater(rows[end + 3], 0.1 * rows.max(), cut)  # past the end, fading
            self.assertEqual(float(rows[end + 11:].max()), 0.0, cut)

    def test_rain_comes_in_waves(self):
        """At 100 the rain is several times as dense at its heaviest as in a lull; at 0 it is steady."""
        for waves, low, high in ((0, 1.0, 1.05), (100, 3.0, 99)):
            fx = effects.make("rain", {"waves": waves}, 3, 750, 25, 640, 360)
            seen = [int(fx.drops(t)[-1].sum()) for t in range(0, 750, 5)]
            self.assertTrue(low <= max(seen) / min(seen) <= high, (waves, min(seen), max(seen)))

    def test_defaults_lay_no_haze(self):
        """Medium rain (the defaults) keeps the dark parts of the picture as they were between the drops; sheets of
        rain lie only here and there, with clear air between them."""
        dark = np.full((H, W, 3), 0.03, np.float32)
        for seed in (1, 2, 3):
            fx = effects.make("rain", {}, seed, 750, 25, W, H)
            for t in (10, 300, 600):
                lift = (frame(fx, t, pic=dark) - dark).mean(axis=2)
                self.assertLess(float(np.median(lift)), 0.5 / 255, (seed, t))
            sheets = effects.make("rain", {"veil": 100, "opacity": 0, "edge": 0}, seed, 750, 25, W, H)
            shares, peaks = [], []
            for t in range(0, 750, 75):
                lift = (frame(sheets, t, pic=dark) - dark).mean(axis=2)
                shares.append(float((lift > 1 / 255).mean()))
                peaks.append(float(lift.max()))
            self.assertLess(np.mean(shares), 0.35, seed)
            self.assertGreater(max(peaks), 0.05, seed)

    def test_small_drops_fall_slower_and_wander(self):
        """Drizzle (small drops) falls slower and drifts a little; rain of big drops is only a little faster."""
        fine = effects.make("rain", {"size": 12, "speed": 8}, 3, 750, 25, W, H)
        mid = effects.make("rain", {}, 3, 750, 25, W, H)
        big = effects.make("rain", {"size": 100}, 3, 750, 25, W, H)
        self.assertAlmostEqual(mid.mean_speed, 2.03, places=2)
        self.assertLess(fine.mean_speed, 0.4 * mid.mean_speed)
        self.assertLess(effects.make("rain", {"size": 12}, 3, 750, 25, W, H).mean_speed, 0.9 * mid.mean_speed)
        self.assertLess(big.mean_speed, 1.25 * mid.mean_speed)  # (thicker drops: not a much faster rain)
        self.assertGreater(float(fine.wander.max()), 2 * float(mid.wander.max()))
        self.assertEqual(float(big.wander.max()), 0.0)
        self.assertTrue(np.all(fine.wm == np.round(fine.wm)) and np.all(fine.wm >= 1))  # whole cycles a loop

    def test_splashes_alike_at_every_size(self):
        """Splashes and rings on the ground cover about as much of it at 1080p as at 4K."""
        cover = []
        for w, h in ((1920, 1080), (3840, 2160)):
            wet = effects.make("rain", {"ground": 70, "splash": 100, "ripple": 100}, 3, 750, 25, w, h)
            dry = effects.make("rain", {"ground": 70, "splash": 0, "ripple": 0}, 3, 750, 25, w, h)
            total = 0.0
            for t in (40, 300):
                for fx, sign in ((wet, 1), (dry, -1)):
                    canvas = effects.Canvas(w, h)
                    fx.draw(t, canvas, None)
                    total += sign * float(canvas.a[int(0.7 * h):].mean())
            cover.append(total)
        self.assertGreater(min(cover), 0)
        self.assertLess(max(cover) / min(cover), 1.4, cover)

    def test_heavy_rain_on_the_ground(self):
        """Heavy rain on a paved ground fizzes: drops land all over it (far more than the streaks drawn), each a
        crown of droplets and spreading rings; they change a good share of the ground."""
        w, h = 1920, 1080
        pave = np.zeros((h, w, 3), np.float32) + 0.3
        pave[: int(0.45 * h)] = 0.7  # (a bright sky)
        light = effects.light_map([np.asarray(Image.fromarray((pave * 255).astype(np.uint8)).resize((960, 540)))])
        look = dict(LOOKS["heavy"], look="heavy", ground=45)
        wet = effects.make("rain", look, 3, 750, 25, w, h, light)
        dry = effects.make("rain", dict(look, splash=0, ripple=0), 3, 750, 25, w, h, light)
        self.assertGreater(len(wet.hits["k"]), 5000)
        self.assertTrue(np.all(wet.hits["k"] == np.round(wet.hits["k"])))  # (whole landings a loop)
        d = np.abs(frame(wet, 100, pave, w=w, h=h) - frame(dry, 100, pave, w=w, h=h)).max(axis=2)
        self.assertGreater(float((d[int(0.46 * h):] > 2 / 255).mean()), 0.05)

    def test_splashes_only_with_a_ground(self):
        """No ground in view: nothing below where the drops already are; a ground: splashes and rings on it."""
        bare = effects.make("rain", {"ground": 100, "splash": 100, "ripple": 100}, 3, 144, 24, W, H)
        wet = effects.make("rain", {"ground": 60, "splash": 100, "ripple": 100}, 3, 144, 24, W, H)
        dry = effects.make("rain", {"ground": 60, "splash": 0, "ripple": 0}, 3, 144, 24, W, H)
        self.assertTrue(np.all(bare.hit > 2))
        band = slice(int(0.62 * H), H)
        more = sum(float(np.abs(frame(wet, t) - frame(dry, t))[band].sum()) for t in range(0, 144, 12))
        self.assertGreater(more, 1.0)


class Drops(unittest.TestCase):
    """What the drops are like at 1080p and 25 frames a second: as a camera sees rain."""

    def on_screen(self, params, times=(40, 200, 400)):
        fx = effects.make("rain", params, 2, 750, 25, 1920, 1080)
        seen = []
        for t in times:
            hx, hy, ux, uy, length, s, show = fx.drops(t)
            seen.append(fx.in_view(hx, ux, length, show & (hy > 0) & (hy < 1080)))
        return fx, np.concatenate(seen)

    def test_medium_rain(self):
        """Mưa vừa: a fair share of drops 2 px wide or more, few hair-thin ones, some soft near ones; it crosses the
        frame in about half a second, its streaks half a frame's fall long (a 180 degree shutter)."""
        fx, i = self.on_screen({})
        width, speed = fx.width[i] / fx.px, fx.speed[i] * 1080 / 25  # px of 1080, px a frame
        self.assertTrue(1.5 < np.median(width) < 2.4, np.median(width))
        self.assertTrue(0.35 < (width >= 2).mean() < 0.6, (width >= 2).mean())
        self.assertLess(np.percentile(width, 10), 1.2)  # (and far ones: fine)
        self.assertTrue(0.04 < (fx.defocus[i] > 0.5).mean() < 0.15, (fx.defocus[i] > 0.5).mean())
        self.assertTrue(70 < np.median(speed) < 110, np.median(speed))
        self.assertAlmostEqual(fx.shutter * 25, 0.525, places=3)

    def test_thin_drops_are_far_and_slow(self):
        """One perspective: how wide a drop looks and how fast it crosses the screen go together, so a thin drop never
        races across; the depth setting spreads them."""
        for depth in (30, 50, 100):
            fx, i = self.on_screen({"depth": depth})
            self.assertGreater(np.corrcoef(np.log(fx.width[i]), np.log(fx.speed[i]))[0, 1], 0.7, depth)
            thin = fx.width[i] / fx.px < 1.2
            if thin.any():
                self.assertLess(np.median(fx.speed[i][thin]), np.median(fx.speed[i]), depth)
        flat, i = self.on_screen({"depth": 0})
        self.assertLess(float(np.ptp(flat.speed[i])) / float(np.mean(flat.speed[i])), 0.35)

    def test_bigger_drops_not_much_faster(self):
        """Thicker drops ("Độ dày hạt") are wider, hardly faster."""
        thin, i = self.on_screen({"size": 10})
        thick, k = self.on_screen({"size": 100})
        self.assertGreater(np.median(thick.width[k]), 2.5 * np.median(thin.width[i]))
        self.assertLess(np.median(thick.speed[k]), 1.4 * np.median(thin.speed[i]))

    def test_small_drops_lean_more_in_the_wind(self):
        fx, i = self.on_screen({"wind": 60, "gust": 0})
        slant = np.degrees(np.arctan2(fx.sin[i], fx.cos[i]))
        self.assertTrue(13 < np.median(slant) < 23, np.median(slant))
        self.assertGreater(float(np.ptp(slant)), 8)

    def test_heavier_looks_have_bigger_drops(self):
        """Heavier looks have wider drops than medium rain; drizzle is fine and slow."""
        med, i = self.on_screen(dict(LOOKS["medium"], look="medium"))
        for name in ("heavy", "storm"):
            fx, k = self.on_screen(dict(LOOKS[name], look=name))
            self.assertGreater(np.median(fx.width[k]), np.median(med.width[i]), name)
        fine, k = self.on_screen(dict(LOOKS["drizzle"], look="drizzle"))
        self.assertLess(np.median(fine.width[k]), np.median(med.width[i]))
        self.assertLess(np.median(fine.speed[k]), 0.5 * np.median(med.speed[i]))


class Lit(unittest.TestCase):
    """Rain in the scene's light: a drop shows the light around it."""

    def colours(self, params, light):
        fx = effects.make("rain", dict({"amount": 100, "veil": 0, "mist": 0, "edge": 0}, **params), 3, 144, 24, W, H,
                          light)
        canvas = effects.Canvas(W, H)
        fx.draw(20, canvas, None)
        a = canvas.a.copy()
        rgb = canvas.rgb / np.maximum(a, 1e-6)[..., None]
        return rgb, a

    def test_bright_and_warm_near_a_light(self):
        rgb, a = self.colours({"lit": 100}, LIGHT)
        seen = a > 0.05
        near, far = round_lamp(0.15), round_lamp(9.0) & ~round_lamp(0.6)
        luma = rgb @ effects.LUMA
        self.assertGreater(luma[seen & near].mean(), 3 * luma[seen & far].mean())
        warm = rgb[seen & near]
        self.assertGreater(warm[:, 0].mean(), 1.1 * warm[:, 2].mean())  # (the lamp's colour)
        rgb, a = self.colours({}, LIGHT)  # (the default: a little of its own colour, so some rain shows in the dark)
        dark = (rgb @ effects.LUMA)[(a > 0.05) & far].mean()
        self.assertGreater(dark, 0.1)
        self.assertLess(dark, 0.35)

    def test_its_own_colour_without_the_light(self):
        """At 0, or without a picture (no video), the rain is drawn in its own colour, the same either way."""
        own = effects._rgb(effects.defaults("rain")["color"])
        for params, light in (({"lit": 0}, LIGHT), ({"lit": 100}, None)):
            rgb, a = self.colours(params, light)
            np.testing.assert_allclose(rgb[a > 0.05], np.broadcast_to(own, rgb[a > 0.05].shape), atol=2e-3)
        a = effects.make("rain", everything(lit=0), 3, 144, 24, W, H, LIGHT)
        b = effects.make("rain", everything(lit=60), 3, 144, 24, W, H)
        self.assertTrue(np.array_equal(frame(a, 30), frame(b, 30)))

    def test_lit_mist_leaves_the_dark_dark(self):
        """Mist and sheets of rain glow round the lights and leave the unlit dark as it was (in their own colour:
        a grey veil over everything)."""
        dark = np.full((H, W, 3), 0.03, np.float32)
        lifts = {}
        for lit in (0, 100):
            fx = effects.make("rain", {"mist": 100, "veil": 100, "opacity": 0, "edge": 0, "lit": lit}, 3, 750, 25,
                              W, H, LIGHT)
            lifts[lit] = (frame(fx, 100, pic=dark) - dark).mean(axis=2)
        far = (slice(None), slice(0, int(0.3 * W)))
        near = (slice(int(0.15 * H), int(0.45 * H)), slice(int(0.72 * W), int(0.88 * W)))
        self.assertLess(lifts[100][far].mean(), 0.25 * lifts[0][far].mean())
        self.assertGreater(lifts[100][near].mean(), 4 * lifts[100][far].mean())

    def test_light_map(self):
        """Glints round a small light; a big bright sky is no light source (rain against it barely shows)."""
        m = effects.light_map([night(640, 360)])
        lh, lw = m.shape[:2]
        self.assertEqual(m.shape, (effects.LIGHT_H, 240, 6))
        lum = effects.light(m) @ effects.LUMA
        x, y = round(0.8 * lw), round(0.3 * lh)
        self.assertGreater(lum[y, x + 6], 20 * lum[y, x - 60])  # (50 px of 1080 from the lamp, and far from it)
        sky = np.zeros((360, 640, 3), np.uint8) + 40
        sky[:150] = 225
        e = effects.light(effects.light_map([sky]))
        self.assertLess(float((e @ effects.LUMA)[:40].max()), float(effects.linear(225 / 255.0)) * 1.02)
        self.assertIsNone(effects.light_map([]))

    def test_daylight_is_no_lamp(self):
        """Only clipped highlights in a dark scene glint: not white paint, a bright sky between dark shapes or a
        clipped sky in daylight. Where the picture is clipped (the sun), the light is at least white."""
        yy, xx = np.mgrid[0:360, 0:640]
        day = np.zeros((360, 640, 3), np.uint8) + np.array([90, 95, 100], np.uint8)
        day[:120] = 205  # an overcast sky
        day[40:110, 200:440] = (40, 60, 45)  # a tree, with the sky showing through a gap
        day[60:80, 300:320] = 218
        day[300:306, 100:540:40] = 220  # white paint on the road
        e = effects.light(effects.light_map([day]))
        self.assertLess(float(e.max()), float(effects.linear(220 / 255.0)) * 1.05)
        blown = day.copy()
        blown[:120] = 255  # a blown-out sky in daylight, through the leaves too
        blown[60:80, 300:320] = 255
        sun = (xx - 500) ** 2 + (yy - 50) ** 2 < 30 ** 2
        e = effects.light(effects.light_map([blown]))
        self.assertLess(float(e.max()), 1.0 + 1e-3)  # (no glint: never brighter than white)
        lh, lw = e.shape[:2]
        self.assertGreater(float(e[round(50 / 360 * lh), round(500 / 640 * lw)].min()), 0.95)  # (the sun: white)
        self.assertTrue(sun.any())
        # a rain in it: the light is nowhere brighter than white, so no drop flashes
        for scene in (day, blown):
            fx = effects.make("rain", {}, 3, 144, 24, W, H, effects.light_map([scene]))
            self.assertIsNone(fx._colours()[2])
        fx = effects.make("rain", {}, 3, 144, 24, W, H, LIGHT)  # (and by a lamp at night some do)
        self.assertIsNotNone(fx._colours()[2])

    def lay(self, params, pic, light=None, t=60):
        """Rain at 640 x 360 laid over `pic` (float RGB) in its own light."""
        h, w = pic.shape[:2]
        light = effects.light_map([(pic * 255 + 0.5).astype(np.uint8)]) if light is None else light
        fx = effects.make("rain", params, 3, 750, 25, w, h, light)
        return frame(fx, t, pic=pic, w=w, h=h)

    def test_lights_stay_bright(self):
        """Nothing in the rain darkens a light brighter than white: a lamp, a neon tube, the sun (the mist and the
        sheets of rain glow round them instead)."""
        yy, xx = np.mgrid[0:360, 0:640]
        dark = np.zeros((360, 640, 3), np.float32) + 0.05
        lamp = (xx - 500) ** 2 + (yy - 100) ** 2 < 14 ** 2
        tube = (np.abs(yy - 200) < 3) & (np.abs(xx - 160) < 80)
        dark[lamp] = (1.0, 0.95, 0.85)
        dark[tube] = (0.4, 1.0, 1.0)
        day = np.zeros((360, 640, 3), np.float32) + np.array([0.75, 0.78, 0.7], np.float32)
        sun = (xx - 450) ** 2 + (yy - 90) ** 2 < 45 ** 2
        day[sun] = 1.0
        day[250:] = 0.15
        for name, values in LOOKS.items():
            for pic, core in ((dark, lamp), (dark, tube), (day, sun)):
                out = self.lay(dict(values, look=name), pic)
                self.assertGreaterEqual(float(out[core].mean()), float(pic[core].mean()) - 0.005, name)
            r = np.hypot(xx - 500, yy - 100)
            halo, far = (r >= 14) & (r < 24), r > 150
            lit = self.lay(dict(values, look=name, opacity=0, mist=80), dark) - dark
            flat = self.lay(dict(values, look=name, opacity=0, mist=80, lit=0), dark) - dark
            self.assertGreater(float(lit[halo].mean()), float(flat[halo].mean()), name)  # (a glow round it)
            self.assertLess(float(lit[far].mean()), 0.25 * float(lit[halo].mean()), name)  # (and dark away from it)

    def test_shows_by_the_light_not_in_the_dark(self):
        """A lit window in a dark wall, the default rain: it shows more beside the window than far along the wall,
        and the far wall's black stays nearly black (drizzle's mist too); against a flat bright sky it barely
        shows."""
        yy, xx = np.mgrid[0:360, 0:640]
        wall = np.zeros((360, 640, 3), np.float32) + 0.03
        window = (np.abs(xx - 200) < 70) & (np.abs(yy - 170) < 90)
        wall[window] = (0.9, 0.62, 0.35)
        light = effects.light_map([(wall * 255 + 0.5).astype(np.uint8)])
        beside = (np.abs(xx - 300) < 25) & (np.abs(yy - 170) < 90)
        far = (xx > 540) & (np.abs(yy - 170) < 90)
        lifts = {"beside": [], "far": []}
        for t in range(40, 400, 40):
            out = self.lay({"amount": 100}, wall, light, t)
            d = (out - wall) @ effects.LUMA
            lifts["beside"].append(d[beside])
            lifts["far"].append(d[far])
        beside, far_ = (np.concatenate(lifts[k]) for k in ("beside", "far"))
        self.assertGreater(float(beside.mean()), 1.5 * float(far_.mean()))
        self.assertLess(float(np.percentile(far_, 99.5)), 0.2)
        black = np.median((self.lay(dict(LOOKS["drizzle"], look="drizzle"), wall, light) @ effects.LUMA)[far])
        self.assertLess(float(black), 1.3 * 0.03)
        sky = np.zeros((360, 640, 3), np.float32) + np.array([0.82, 0.84, 0.86], np.float32)
        for name in ("medium", "heavy"):
            d = (self.lay(dict(LOOKS[name], look=name), sky) - sky) @ effects.LUMA
            self.assertLess(float(np.percentile(np.abs(d), 99.5)), 0.04, name)

    def test_turned_sheet_sees_the_light_of_its_place(self):
        """A rain placed or turned in 3D gets the light of the place on the frame where each part of it lands."""
        nearly = sheet.Sheet("rain", {"rot_z": 0.1}, 3, 150, 25, W, H, light=LIGHT)
        got = nearly.fx.light
        self.assertEqual(got.shape[0], LIGHT.shape[0])
        np.testing.assert_allclose(effects.light(got)[5:-5, 5:-5], effects.light(LIGHT)[5:-5, 5:-5], rtol=0.15,
                                   atol=0.01)
        moved = sheet.Sheet("rain", {"x": -45}, 3, 150, 25, W, H, light=LIGHT).fx.light
        lum = effects.light(moved) @ effects.LUMA  # (moved left: its right part lies on the frame's middle)
        brightest = float((effects.light(LIGHT) @ effects.LUMA).max())
        self.assertLess(float(lum[:, -lum.shape[1] // 3:].max()), 0.5 * brightest)
        a = np.asarray(nearly.draw(0, Image.fromarray(night())))
        b = np.asarray(nearly.draw(150, Image.fromarray(night())))
        self.assertLessEqual(int(np.abs(a.astype(int) - b).max()), 1)  # (it loops)


class Video(unittest.TestCase):
    """The light comes from the project's video: read once, handed to every piece of work."""

    @classmethod
    def setUpClass(cls):
        path = os.path.join(helpers.ROOT, "night.mp4")
        enc = video.Encoder(path, W, H, "24", crf=18, preset="veryfast")
        for _ in range(12):
            enc.write(night())
        enc.close()
        cls.project = projects.create(path, "night.mp4")
        cls.folder = projects.folder(cls.project["id"])

    def test_the_video_lights_the_rain(self):
        light = engine.scene_light(self.folder, self.project)
        self.assertIsNotNone(light)
        lum = effects.light(light) @ effects.LUMA
        self.assertGreater(lum[:, -lum.shape[1] // 4:].max(), 10 * lum[:, :lum.shape[1] // 4].max())
        self.assertIs(engine.scene_light(self.folder, self.project), light)  # (read once)
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "rainlt", "kind": "rain", "params": {"lit": 100}, "seed": 1},
            {"id": "raintn", "kind": "rain", "params": {"rot_y": 20}, "seed": 2}]})
        scene = engine.Scene(self.folder, project, (W, H))
        self.assertIs(scene.layers[0][0].light, light)
        self.assertIsNotNone(scene.layers[1][0].fx.light)  # (turned: on its sheet)
        handed = engine.found_files(self.folder, project)
        self.assertIs(handed["_light"], light)
        self.assertIs(engine.scene_light("nowhere", handed), light)  # (a worker reads nothing)
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def test_light_of_the_whole_loop(self):
        """The light is the loop's average, from frames spread over all of it: a lamp lit only in the first second
        counts for that second, one lit for the rest of the loop counts for the rest."""
        path = os.path.join(helpers.ROOT, "lamps.mp4")
        enc = video.Encoder(path, W, H, "24", crf=18, preset="veryfast")
        yy, xx = np.mgrid[0:H, 0:W]
        first = (xx - 0.25 * W) ** 2 + (yy - 0.3 * H) ** 2 < (0.03 * H) ** 2
        later = (xx - 0.75 * W) ** 2 + (yy - 0.3 * H) ** 2 < (0.03 * H) ** 2
        for t in range(144):
            f = np.zeros((H, W, 3), np.uint8) + 14
            f[first if t < 24 else later] = (255, 205, 120)
            enc.write(f)
        enc.close()
        project = projects.create(path, "lamps.mp4")
        lum = effects.light(engine.scene_light(projects.folder(project["id"]), project)) @ effects.LUMA
        lh, lw = lum.shape
        on_first, on_later = lum[round(0.3 * lh), round(0.25 * lw)], lum[round(0.3 * lh), round(0.75 * lw)]
        self.assertGreater(on_later, 3 * on_first)
        self.assertGreater(on_first, 3 * lum[-5, 5])

    def test_preview_clip_names(self):
        """Its preview clip is remade for another video when the rain is lit by it, not when it is not; the rain's
        preview clip is wider than other effects' (its streaks are thin)."""
        project = projects.load(self.project["id"])
        other = dict(project, video=dict(project["video"], file="input-x.mp4"))
        for lit, same in ((70, False), (0, True)):
            layer = {"kind": "rain", "params": {"lit": lit}, "seed": 1}
            names = {engine.fx_clip_name(p, layer) for p in (project, other)}
            self.assertEqual(len(names) == 1, same, lit)
        info = {"w": 1920, "h": 1080}
        self.assertEqual(engine.fx_clip_frame(info, "rain"), (1280, 720))
        self.assertEqual(engine.fx_clip_frame(info, "snow"), (640, 360))
        self.assertEqual(engine.fx_clip_frame({"w": 960, "h": 540}, "rain"), (960, 540))

    def test_preview_clip_is_lit(self):
        """The preview clip carries the light, as the export draws it."""
        params = {"amount": 100, "lit": 100, "veil": 0}
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "rainpv", "kind": "rain", "params": params, "seed": 1}]})
        name = engine.fx_clip(self.folder, project, "rainpv", processes=1)
        path = os.path.join(self.folder, "clips", name)
        info = video.probe(path, overlay=True)
        with video.Frames(path, info, alpha=True, count=1) as frames:
            first = next(iter(frames)).astype(np.float32) / 255
        fx = effects.make("rain", params, 1, 720, 24, W, H, engine.scene_light(self.folder, project))
        canvas = effects.Canvas(W, H)
        fx.draw(0, canvas, None)
        drawn = canvas.rgb / np.maximum(canvas.a, 1e-6)[..., None] @ effects.LUMA
        luma = first[..., :3] @ effects.LUMA
        near, far = round_lamp(0.15), round_lamp(9.0) & ~round_lamp(0.6)
        for part in (near, far):
            self.assertAlmostEqual(luma[part & (first[..., 3] > 0.2)].mean(),
                                   drawn[part & (canvas.a > 0.2)].mean(), delta=0.04)
        self.assertGreater(luma[near & (first[..., 3] > 0.2)].mean(), luma[far & (first[..., 3] > 0.2)].mean() + 0.1)
        projects.apply(projects.load(self.project["id"]), {"layers": []})


class Speed(unittest.TestCase):
    def test_a_default_frame_in_time(self):
        """The speed budget: the default rain at 1080p lit by a scene, as an export draws it, measured at 65 ms a frame
        (the rain before it: 64 ms; heavy rain on a ground with its splashes: 118 ms, 91 ms before); here, for a busy
        machine, under 0.35 s. Each round's frames are timed together, the best round counts."""
        light = effects.light_map([night(1920, 1080)])
        fx = effects.make("rain", {}, 2, 750, 25, 1920, 1080, light)
        canvas = effects.Canvas(1920, 1080)
        pic = np.full((1080, 1920, 3), 0.2, np.float32)
        fx.draw(0, canvas, None)
        canvas.reset()
        best = math.inf
        for r in range(3):
            start = time.perf_counter()
            for t in (10 + r, 20 + r, 30 + r):
                fx.draw(t, canvas, None)
                canvas.flush(pic, fx.blend)
            best = min(best, (time.perf_counter() - start) / 3)
        self.assertLess(best, 0.35)


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
        self.assertEqual(rain.Rain.VERSION, 4)

    def test_old_looks_become_the_new_ones(self):
        """A clip saved before the rain was lit by the scene (no "lit") that kept a look as it was gets that look as
        it is now; one changed by hand keeps its numbers; settings saved now are left alone."""
        for name, old in rain.OLD_PRESETS.items():
            saved = dict(effects.defaults("rain"), **old, look=name)
            del saved["lit"]
            p = effects.clean_params("rain", saved)
            self.assertEqual({k: p[k] for k in LOOKS[name]}, LOOKS[name], name)
            self.assertEqual(p["lit"], effects.defaults("rain")["lit"])
            changed = dict(saved, amount=old["amount"] - 7)
            p = effects.clean_params("rain", changed)
            self.assertEqual({k: p[k] for k in old}, {k: changed[k] for k in old}, name)
        mine = dict(effects.defaults("rain"), **rain.OLD_PRESETS["drizzle"], look="drizzle")  # (has "lit")
        self.assertEqual(effects.clean_params("rain", mine), mine)
        self.assertEqual(effects.clean_params("rain", {"amount": 60})["amount"], 60)

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
                self.assertNotIn(key, ("look", "ground", "color", "lit"), name)  # the scene's own, kept
                self.assertTrue(by_key[key]["min"] <= v <= by_key[key]["max"], (name, key))
            self.assertEqual(effects.clean_params("rain", dict(defaults, **values)), dict(defaults, **values))


if __name__ == "__main__":
    unittest.main()
