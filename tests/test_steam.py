import json
import os
import time
import unittest

import numpy as np
from PIL import Image

from tests import helpers
from fx import effects, engine, projects, sheet, steam, video

W, H = 320, 180
PLACE = ("x", "y", "scale")
RANGES = [p for p in steam.SCHEMA["params"] if p["type"] == "range" and p["key"] not in PLACE]
MAX = {p["key"]: p["max"] for p in RANGES}
MIN = {p["key"]: p["min"] for p in RANGES}
BIG = {"scale": 90, "x": 0, "y": 0}  # a big box in the small test frame
STYLES = {o["value"]: dict(steam.STYLE_PRESETS[o["value"]], style=o["value"])
          for o in next(p for p in steam.SCHEMA["params"] if p["key"] == "style")["options"]}


def frame(fx, t, base=0.3, mask=None):
    pic = np.full((fx.h, fx.w, 3), base, np.float32)
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, mask)
    canvas.flush(pic, fx.blend)
    return pic


def cover(fx, t, mask=None):
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, mask)
    return canvas.a.copy()


def inside(fx):
    """Rows and columns of the frame inside the effect's box (with a pixel to spare for rounding)."""
    x0, y0, w, h = fx.box()
    return slice(max(0, int(y0) - 1), int(y0 + h) + 2), slice(max(0, int(x0) - 1), int(x0 + w) + 2)


def in_box(fx, a):
    x0, y0, w, h = fx.box()
    return a[int(round(y0)):int(round(y0 + h)), int(round(x0)):int(round(x0 + w))]


def mean_cover(params, frames=720, step=15, size=(640, 360)):
    """The coverage in the box, averaged over frames spread over the whole loop."""
    fx = effects.make("steam", params, 3, frames, 24, *size)
    return sum(in_box(fx, cover(fx, t)) for t in range(0, frames, step)) / len(range(0, frames, step)), fx


def light_of(colour, w=W, h=H, lamp=None):
    """A scene's light map (effects.light_map) of a flat picture of one colour, a bright lamp on one side if asked."""
    pic = np.full((h, w, 3), colour, np.uint8)
    if lamp == "left":
        pic[:, :w // 3] = 255
    return effects.light_map([pic])


class Loops(unittest.TestCase):
    """Frame P is frame 0 again with every style and look and at the extremes, at any loop length and frame rate."""

    CASES = [dict(STYLES[s], **steam.LOOKS[name], look=name) for s in STYLES for name in steam.LOOKS] + [
        MAX, dict(MIN, amount=100, opacity=100), {"breeze": -100, "gust": 100, "curl": 100, "speed": 100},
        {"speed": 0, "reach": 100, "curl": 0, "spread": 100}, dict(STYLES["pot"], **MAX),
        dict(STYLES["incense"], breeze=60, gust=100, curl=100)]

    def test_frame_p_is_frame_zero(self):
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24)):
            for params in self.CASES if frames != 720 else self.CASES[:1] + self.CASES[4:5] + self.CASES[8:9]:
                fx = effects.make("steam", {**params, **BIG}, 3, frames, fps, W, H)
                first = frame(fx, 0)
                self.assertGreater(np.abs(first - 0.3).max(), 0.02, params)  # (there is steam to see)
                diff = np.abs(first - frame(fx, frames)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")

    def test_seam_drawn_frame_by_frame(self):
        """An export draws frame after frame: the steam it carries on to at frame P is the steam a new start finds
        at frame 0 (the whirls too: a new start finds the whirls born before it)."""
        for frames, fps in ((144, 24), (150, 25), (90, 30)):
            for params in ({}, self.CASES[-4], STYLES["incense"], STYLES["pot"]):
                going = effects.make("steam", {**params, **BIG}, 5, frames, fps, W, H)
                for t in range(frames - 40, frames):
                    frame(going, t)
                fresh = effects.make("steam", {**params, **BIG}, 5, frames, fps, W, H)
                diff = np.abs(frame(going, frames) - frame(fresh, 0)).max()
                self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")

    def test_any_order(self):
        """Frame by frame or straight to a frame, the steam is the same."""
        for params in ({}, STYLES["pot"]):
            a = effects.make("steam", {**params, **BIG}, 2, 144, 24, W, H)
            for t in range(20, 31):
                frame(a, t)
            b = effects.make("steam", {**params, **BIG}, 2, 144, 24, W, H)
            self.assertLess(np.abs(frame(a, 31) - frame(b, 31)).max(), 1e-4, params)

    def test_rises_and_changes(self):
        fx = effects.make("steam", BIG, 3, 144, 24, W, H)
        self.assertGreater(np.abs(frame(fx, 0) - frame(fx, 12)).max(), 0.02)
        # every bit of steam is higher a tick later (row a, the age in ticks, is row a + 1 a tick later), the whirls'
        # points too
        before, after = fx._state(0), fx._state(1)
        self.assertGreater(float((after[1][1:] - before[1][:-1]).min()), 0)
        self.assertGreater(float((after[3][1:] - before[3][:-1]).min()), 0)

    def test_same_seed_same_steam(self):
        a, b = (effects.make("steam", BIG, 9, 144, 24, W, H) for _ in range(2))
        c = effects.make("steam", BIG, 10, 144, 24, W, H)
        self.assertTrue(np.array_equal(frame(a, 17), frame(b, 17)))
        self.assertFalse(np.array_equal(frame(a, 17), frame(c, 17)))

    def test_few_frames_a_second(self):
        """A video of few frames a second (more than one tick a frame) still carries the steam on from the frame
        before, and gets the same steam as frame after frame or a fresh start."""
        seq = effects.make("steam", BIG, 3, 64, 8, W, H)
        steps = []
        step = seq._step
        seq._step = lambda *a: steps.append(1) or step(*a)
        drawn = {t: cover(seq, t) for t in range(10, 30)}
        self.assertLess(len(steps), seq.L + 3 * 20)  # (not a whole lifetime of steam each frame)
        skip = effects.make("steam", BIG, 3, 64, 8, W, H)
        for t in range(10, 30, 3):
            self.assertTrue(np.array_equal(cover(skip, t), drawn[t]), t)
        fresh = effects.make("steam", BIG, 3, 64, 8, W, H)
        self.assertLess(np.abs(cover(fresh, 29) - drawn[29]).max(), 1e-4)

    def test_between_frames(self):
        """A time between two frames is drawn too (between them)."""
        fx = effects.make("steam", BIG, 3, 144, 24, W, H)
        a, b, mid = cover(fx, 10), cover(fx, 11), cover(fx, 10.5)
        self.assertLessEqual(np.abs(mid - (a + b) / 2).mean(), np.abs(a - b).mean() + 1e-6)

    def test_breathes_but_never_goes(self):
        """How much steam there is comes and goes over the loop, but it never all but vanishes; and nothing pops
        from one frame to the next."""
        for params in ({}, STYLES["incense"], STYLES["pot"]):
            fx = effects.make("steam", {**params, **BIG}, 4, 720, 24, W, H)
            totals = np.array([cover(fx, t).sum() for t in range(0, 720, 8)])
            self.assertGreater(totals.min(), 0.45 * np.median(totals), params)
            run = [cover(fx, t) for t in range(300, 330)]
            steps = [np.abs(a - b).mean() for a, b in zip(run, run[1:])]
            self.assertLess(max(steps), 2.5 * np.median(steps), params)

    def test_nothing_pops_at_a_tick(self):
        """The air moves on ticks (20 a second), frames lie between them: right over the surface, where new smoke
        comes in, the step from a frame on a tick to the next is no bigger than the others (no pulse 4 or 10 times a
        second)."""
        for params in STYLES.values():
            for fps, per in ((24, 6), (30, 3)):  # (a frame on a tick every `per` frames)
                fx = effects.make("steam", {**params, "scale": 90}, 3, 10 * fps, fps, 640, 360)
                x0, y0, w, h = fx.box()
                rows = slice(int(round(y0 + h * (1 - 2 * fx.yc - 0.1))), int(round(y0 + h)))
                run = np.stack([cover(fx, t)[rows, int(x0):int(x0 + w)] for t in range(60, 60 + 40 * per + 1)])
                steps = np.abs(np.diff(run, axis=0)).mean(axis=(1, 2))
                on = np.arange(60, 60 + steps.size) % per == 0
                self.assertLess(steps[on].mean(), 1.15 * steps[~on].mean(), (params["style"], fps))


class Box(unittest.TestCase):
    """A placed effect: the steam rises from the mark at the middle of its box's bottom and stays inside the box."""

    def test_steam_stays_in_its_box(self):
        for params in ({"scale": 40, "x": 20, "y": -10},
                       {"scale": 60, "x": -25, "y": 15, "breeze": 100, "spread": 100, "curl": 100, "reach": 100,
                        "rim": 100, "gust": 100},
                       dict(STYLES["pot"], scale=60, x=10, amount=100, spread=100, breeze=-100),
                       dict(STYLES["incense"], scale=50, y=20, curl=100, gust=100)):
            fx = effects.make("steam", params, 2, 144, 24, 640, 360)
            rows, cols = inside(fx)
            for t in (0, 40, 90):
                a = cover(fx, t)
                self.assertGreater(a[rows, cols].max(), 0.05, params)
                a[rows, cols] = 0
                self.assertEqual(a.max(), 0, params)

    def test_from_the_mark_up(self):
        """It rises from the mark (an ellipse sitting on the box's bottom edge, MARK as high as wide) right from the
        surface, densest low down and thinning as it rises (never a lump high up); faded before the box's top and
        sides."""
        mean, fx = mean_cover({"scale": 95})
        bh, bw = mean.shape
        self.assertAlmostEqual(fx.yc, steam.MARK / 2 * effects.defaults("steam")["rim"] / 100)
        rows = mean.mean(axis=1)[::-1]  # (from the bottom)
        mark = int(round(2 * fx.yc * bh))  # the mark's top, rows from the bottom
        low = rows[mark:mark + int(0.06 * bh)].mean()
        self.assertGreater(low, 0.5 * rows.max())  # (it does not float above the surface)
        tenths = [rows[int(bh * f):int(bh * (f + 0.1))].mean() for f in np.arange(0.2, 1.0, 0.1)]
        self.assertTrue(all(b <= a * 1.05 for a, b in zip(tenths, tenths[1:])), tenths)  # (thinner as it rises)
        surface = mean[bh - mark:]
        self.assertGreater(surface[:, int(bw * 0.4):int(bw * 0.6)].mean(), 5 * surface[:, :int(bw * 0.2)].mean())
        self.assertLess(mean[:int(bh * 0.02)].max(), 0.01)  # faded before the top
        self.assertLess(mean[:, :int(bw * 0.02)].max(), 0.01)  # and the sides
        self.assertLess(mean[:, -int(bw * 0.02):].max(), 0.01)

    def test_fades_before_the_sides(self):
        """However it spreads or leans, the steam thins out before the box's sides (each strand a little sooner or
        later: no straight edge), not cut off by them."""
        for name, params in STYLES.items():
            for look in steam.LOOKS:
                fx = effects.make("steam", dict(params, **steam.LOOKS[look], look=look, scale=95), 3, 720, 24, 640,
                                  360)
                for t in range(0, 720, 48):
                    a = in_box(fx, cover(fx, t))
                    n = int(0.06 * a.shape[1])
                    self.assertLess(max(a[:, :n].max(), a[:, -n:].max()), 0.05, (name, look, t))

    def test_never_a_flat_lump(self):
        """Where much steam overlaps it covers more, but ever more slowly: never all its opacity (a flat grey
        lump), not even a dense pot before a bright light (it glows), at its thickest."""
        light = light_of((70, 60, 50), 640, 360, lamp="left")
        for name, params in STYLES.items():
            for look in ("light", "thick", "strong"):
                p = dict(params, **steam.LOOKS["thick" if look == "strong" else look], scale=95)
                if look == "strong":
                    p.update(amount=100, opacity=100)
                fx = effects.make("steam", p, 3, 720, 24, 640, 360, light)
                for t in range(0, 720, 90):
                    a = in_box(fx, cover(fx, t))
                    self.assertLessEqual(a.max(), 0.951 * p["opacity"] / 100, (name, look, t))
                    self.assertLess((a > 0.9 * p["opacity"] / 100).sum(), 0.15 * (a > 0.05).sum(), (name, look, t))

    def test_a_veil_over_the_pot(self):
        """Over a pot the steam is an even veil right over the water (no row of tongues, each from its own point),
        thicker than over a cup."""
        def peaks(row):  # (bumps standing well clear of the dips on both sides)
            d = np.sign(np.diff(row))
            ext = row[[0, *(np.flatnonzero(np.diff(d) != 0) + 1), row.size - 1]]
            dip = 0.15 * row.max()
            return int(((ext[1:-1] - ext[:-2] > dip) & (ext[1:-1] - ext[2:] > dip)).sum())
        for look in steam.LOOKS:
            fx = effects.make("steam", dict(STYLES["pot"], **steam.LOOKS[look], look=look, scale=95), 3, 720, 24,
                              640, 360)
            counts = []
            for t in range(0, 720, 60):
                a = in_box(fx, cover(fx, t))
                bh, bw = a.shape
                half = 0.7 * fx.a * bh
                for f in (1.0, 0.5):  # (the mark's middle, and in front of it)
                    row = a[int(round(bh * (1 - f * fx.yc))), int(bw / 2 - half):int(bw / 2 + half)]
                    self.assertGreater(row.mean(), 0.03, (look, t))
                    counts.append(peaks(row))
            self.assertLess(np.mean(counts), 2.5, look)  # (the old pot: 4 to 5, up to 11)

    def test_a_kettle(self):
        """A pot's steam from a narrow mark (a kettle's spout): fewer points, less steam in all (not all of a pot's
        heaped on a few pixels), a short clear stretch above the spout, then a plume."""
        pot = effects.make("steam", STYLES["pot"], 3, 720, 24, 640, 360)
        fx = effects.make("steam", dict(STYLES["pot"], rim=12, scale=95), 3, 720, 24, 640, 360)
        self.assertEqual(fx.n, steam.FEW)
        self.assertLess(fx.gain * fx.n, 0.5 * pot.gain * pot.n)
        mean = sum(in_box(fx, cover(fx, t)) for t in range(0, 720, 30)) / 24
        rows = mean.max(axis=1)[::-1]  # (from the bottom)
        mark = int(round(2 * fx.yc * mean.shape[0]))
        self.assertLess(rows[:mark + 2].max(), 0.25 * rows.max())  # (clear just above the spout)
        self.assertGreater(rows[mark:mark + int(0.25 * mean.shape[0])].max(), 0.15)
        self.assertLess(mean.max(), 0.9 * STYLES["pot"]["opacity"] / 100)

    def test_no_comb(self):
        """No strands rise from the same places all the time (a comb): over a whole loop, the steam just above the
        surface is spread, not a row of stripes (the old steam had 7 to 10 of them)."""
        def peaks(profile):
            top, out = profile.max(), 0
            for i in range(1, profile.size - 1):
                if profile[i] >= profile[i - 1] and profile[i] > profile[i + 1] and profile[i] > 0.15 * top:
                    if profile[i] - max(profile[max(0, i - 12):i].min(), profile[i + 1:i + 13].min()) > 0.06 * top:
                        out += 1
            return out
        for params in ({}, {"look": "thick", **steam.LOOKS["thick"]}, {"rim": 100}, STYLES["pot"]):
            mean, fx = mean_cover(dict(params, scale=95))
            bh = mean.shape[0]
            lo, hi = int(bh * (1 - fx.yc - 0.1)), int(bh * (1 - fx.yc - 0.03))
            self.assertLessEqual(peaks(mean[lo:hi].mean(axis=0)), 4, params)

    def test_reach(self):
        def upper(reach):
            fx = effects.make("steam", {"scale": 90, "reach": reach}, 4, 144, 24, 640, 360)
            x0, y0, w, h = fx.box()
            a = sum(cover(fx, t) for t in range(0, 144, 24))
            return a[int(y0):int(y0 + h * 0.5), int(x0):int(x0 + w)].mean()
        self.assertLess(upper(30), 1e-4)
        self.assertGreater(upper(100), 0.005)

    def test_breeze_leans(self):
        def lean(breeze):
            fx = effects.make("steam", {"scale": 90, "breeze": breeze, "gust": 0}, 4, 144, 24, 640, 360)
            x0, y0, w, h = fx.box()
            a = sum(cover(fx, t) for t in range(0, 144, 24))[int(y0):int(y0 + h * 0.6), int(x0):int(x0 + w)]
            return float((a.sum(axis=0) * np.arange(a.shape[1])).sum() / a.sum() / a.shape[1] - 0.5)
        self.assertGreater(lean(100), 0.1)
        self.assertLess(lean(-100), -0.1)
        self.assertLess(abs(lean(0)), 0.1)
        self.assertGreater(lean(steam.LOOKS["breeze"]["breeze"]), lean(steam.DEFAULTS["breeze"]) + 0.08)  # the look

    def test_wide_rim(self):
        """A wide rim has more points, as close together as at half the box: no fringe of strands far apart."""
        half, wide = (effects.make("steam", {"rim": r}, 2, 144, 24, W, H) for r in (50, 100))
        self.assertEqual(wide.n, 2 * half.n)
        self.assertTrue(np.array_equal(wide.ux[:half.n], half.ux))  # (the same points, and more)
        self.assertAlmostEqual(wide.a, 2 * half.a)

    def test_incense_is_a_thin_smooth_thread(self):
        """Incense: one thin thread straight up from its tip (a calm stretch), that only higher up waves, curls and
        spreads."""
        fx = effects.make("steam", dict(STYLES["incense"], scale=100), 6, 720, 24, 1920, 1080)
        x0, y0, w, h = fx.box()
        for t in (0, 200, 470):
            a = in_box(fx, cover(fx, t))
            bh, bw = a.shape
            for f, most in ((0.04, 6), (0.1, 6), (0.18, 10)):  # (heights above the tip, of the box; pixels)
                row = a[int(bh * (1 - fx.yc - f))]
                on = np.flatnonzero(row > 0.5 * row.max())
                self.assertGreater(row.max(), 0.15, (t, f))
                self.assertLessEqual(on[-1] - on[0] + 1, most, (t, f))  # (one thread, a few pixels wide in 1080)
                self.assertLess(abs((on[0] + on[-1]) / 2 - bw / 2), 0.04 * bw, (t, f))  # (straight up)
            high = a[int(bh * 0.3):int(bh * 0.6)]
            cols = np.flatnonzero(high.max(axis=0) > 0.05)
            self.assertGreater(cols[-1] - cols[0], 0.08 * bw, t)  # (higher up it waves and spreads)

    def test_pot_is_wide_and_dense(self):
        """A pot: dense steam over the whole surface, much more of it than over a cup, rising quicker."""
        pot, fx = mean_cover(dict(STYLES["pot"], scale=95))
        cup, _ = mean_cover({"scale": 95})
        bh, bw = pot.shape
        low = pot[int(bh * (1 - 2 * fx.yc - 0.15)):int(bh * (1 - 2 * fx.yc))]
        across = low.mean(axis=0)
        rim = slice(int(bw * (0.5 - 0.35 * fx.a)), int(bw * (0.5 + 0.35 * fx.a)))
        self.assertGreater(across[rim].min(), 0.25 * across.max())  # (over the whole surface)
        self.assertGreater(pot.mean(), 3 * cup.mean())
        self.assertGreater(fx.speed, effects.make("steam", {}, 3, 720, 24, W, H).speed * 1.4)

    def test_preview_clip_is_the_box(self):
        """The page's preview clip (the box alone, at its own size, lit by the light where the box stands) looks like
        the steam drawn in the frame."""
        info = {"w": 960, "h": 540, "fps": "24", "frames": 144, "seconds": 6.0}
        cw, ch = engine.fx_clip_frame(info, "steam")
        self.assertEqual((cw, ch), (540, 540))
        scene = np.zeros((540, 960, 3), np.uint8)
        scene[:, :300] = (255, 200, 120)  # (a warm lamp on the left)
        scene[:, 300:] = (30, 24, 20)
        light = effects.light_map([scene])
        for params in ({"scale": 50, "x": 12, "y": -5}, {"scale": 30, "x": -20, "y": 10, "look": "long",
                                                         **steam.LOOKS["long"]},
                       dict(STYLES["pot"], scale=60, x=5), dict(STYLES["incense"], scale=70)):
            placed = effects.placed_light("steam", light, params, 960, 540)
            clip = effects.make("steam", engine.fx_clip_params("steam", params), 4, 144, 24, cw, ch, placed)
            full = effects.make("steam", params, 4, 144, 24, 960, 540, light)
            self.assertTrue(np.allclose(clip.tone, full.tone), params)  # (lit the same)
            self.assertEqual(tuple(round(v) for v in clip.box()), (0, 0, cw, ch))
            for t in (0, 70):
                x0, y0, w, h = full.box()
                bx, by = int(round(x0)), int(round(y0))
                bw, bh = int(round(x0 + w)) - bx, int(round(y0 + h)) - by
                canvas = effects.Canvas(960, 540)
                full.draw(t, canvas, None)
                a = canvas.a[by:by + bh, bx:bx + bw]
                rgb = canvas.rgb[by:by + bh, bx:bx + bw]
                small = effects.Canvas(cw, ch)
                clip.draw(t, small, None)
                b = np.asarray(Image.fromarray(small.a).resize((bw, bh), Image.Resampling.BILINEAR))
                c = np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(small.rgb[..., i])).resize(
                    (bw, bh), Image.Resampling.BILINEAR)) for i in range(3)], 2)
                self.assertGreater(a.max(), 0.05)
                self.assertLess(np.abs(a - b).mean(), 0.06 * a.mean() + 1e-4, params)
                self.assertLess(np.abs(a - b).max(), 0.3 * a.max(), params)
                self.assertLess(np.abs(rgb - c).mean(), 0.06 * rgb.mean() + 1e-4, params)

    def test_preview_clip_name(self):
        """Lit by the scene, the preview clip is lit by the light where the box stands (a little rounded): moved
        into other light, a new one is made; nudged, or turned, not (it is lit the same); unlit, it is the same
        wherever the box is."""
        src = os.path.join(helpers.ROOT, "steam-lamp.mp4")
        enc = video.Encoder(src, 320, 180, "24", crf=18, preset="veryfast")
        for _ in range(12):
            f = np.full((180, 320, 3), 40, np.uint8)
            f[:, :80] = (255, 210, 150)  # (a warm lamp on the left)
            enc.write(f)
        enc.close()
        project = projects.create(src, "steam-lamp.mp4")

        def name(**p):
            return engine.fx_clip_name(project, {"id": "s1", "kind": "steam", "params": p, "seed": 3})
        self.assertNotEqual(name(x=-30), name(x=30))
        self.assertNotEqual(name(scale=20, x=-30), name(scale=60, x=-30))
        self.assertEqual(name(x=10), name(x=10.5))  # (nudged)
        self.assertEqual(name(x=10), name(x=10, rot_y=30, rot_z=5))
        self.assertEqual(name(x=-30, lit=0), name(x=30, lit=0, scale=40))
        self.assertNotEqual(name(style="incense"), name(style="pot"))
        light = engine.scene_light(projects.folder(project["id"]), project)
        exact = effects.placed_light("steam", light, {"x": 10}, 320, 180)
        rounded = effects.placed_light("steam", light, {"x": 10}, 320, 180, rounded=True)
        self.assertLess(np.abs(np.sqrt(exact[:6]) - np.sqrt(rounded[:6])).max(), 1 / 120)
        clip = effects.make("steam", {"style": "incense"}, 1, 144, 24, 640, 640)
        self.assertLess(clip.PREVIEW_CRF, 34)  # (thin threads: a little more quality)

    def test_as_strong_at_any_speed_or_reach(self):
        """Quicker steam is not much fainter (as much steam along a wisp), and steam that does not rise far still
        shows."""
        def strength(params):
            fx = effects.make("steam", {"scale": 100, **params}, 4, 144, 24, 270, 270)
            frames = [cover(fx, t) for t in range(0, 144, 8)]
            return np.mean([a.mean() for a in frames]), np.median([a.max() for a in frames])
        mean, peak = strength({})
        for speed in (0, 100):
            self.assertLess(abs(strength({"speed": speed})[0] / mean - 1), 0.45, speed)
        self.assertGreater(strength({"reach": 20})[1], 0.5 * peak)

    def test_off_the_frame_and_extreme_sizes(self):
        for params in ({"scale": 80, "x": 60, "y": 40}, {"scale": 1}, {"scale": 300}, {"scale": 10, "x": 100},
                       {"scale": 50, "y": -100}, dict(STYLES["pot"], scale=300), dict(STYLES["incense"], scale=1),
                       {"rim": 1}, {"rim": 1, "style": "pot"}):
            fx = effects.make("steam", params, 2, 144, 24, W, H)
            a = cover(fx, 33)
            self.assertTrue(np.isfinite(a).all())
            self.assertLessEqual(a.max(), 1.0)
        fx = effects.make("steam", {"scale": 10, "x": 100, "y": 100}, 2, 144, 24, W, H)  # nowhere in the frame
        self.assertEqual(cover(fx, 5).max(), 0)

    def test_big_box_drawn_finely(self):
        """A box taller than GRID_MAX is drawn on a grid of GRID_MAX rows, enlarged; a smaller one at its own size."""
        fx = effects.make("steam", {"scale": 100}, 2, 144, 24, 1920, 1080)
        got = fx.cover(30, fx.box()[3])
        self.assertEqual(got[4], (steam.GRID_MAX, steam.GRID_MAX))
        a = cover(fx, 30)
        self.assertGreater(a.max(), 0.05)
        small = effects.make("steam", {"scale": 45}, 2, 144, 24, 1920, 1080)
        self.assertEqual(small.cover(30, small.box()[3])[4], (486, 486))

    def test_turned(self):
        """Turned in 3D (on its sheet: its box), it loops and stays within its turned box; lit, it is lit as upright
        (where its box stands)."""
        light = light_of((60, 50, 40), 640, 360, lamp="left")
        params = {"scale": 60, "x": -10, "rot_y": 30, "rot_x": 10, "persp": 60}
        s = sheet.Sheet("steam", params, 3, 144, 24, 640, 360, light=light)
        upright = effects.make("steam", {"scale": 60, "x": -10}, 3, 144, 24, 640, 360, light)
        self.assertTrue(np.allclose(s.fx.tone, upright.tone))
        pic = Image.new("RGB", (640, 360), (40, 40, 40))
        a, b = (np.asarray(s.draw(t, pic.copy())).astype(int) for t in (0, 144))
        self.assertLessEqual(int(np.abs(a - b).max()), 1)
        self.assertGreater(int(np.abs(a - 40).max()), 10)


class Picture(unittest.TestCase):
    def test_keeps_out_of_its_area(self):
        """Outside the clip's area (mask 0) the picture stays exactly as it was; a half mask halves the steam."""
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        fx = effects.make("steam", dict(BIG, amount=100, opacity=100), 3, 144, 24, W, H)
        pic = frame(fx, 7, mask=mask)
        self.assertTrue(np.array_equal(pic[:, :W // 2], np.full((H, W // 2, 3), 0.3, np.float32)))
        self.assertGreater(pic[:, W // 2:].max(), 0.33)
        half = cover(fx, 7, np.full((H, W), 0.5, np.float32))
        self.assertAlmostEqual(float(half.max()), float(cover(fx, 7).max()) * 0.5, places=5)

    def test_nothing_without_steam(self):
        for params in ({"amount": 0}, {"opacity": 0}):
            fx = effects.make("steam", {**BIG, **params}, 3, 144, 24, W, H, light_of((90, 80, 70)))
            self.assertTrue(np.array_equal(frame(fx, 5), np.full((H, W, 3), 0.3, np.float32)), params)

    def test_shows_on_dark_and_bright(self):
        """Laid over the picture: pale steam lightens a dark room and still shows (a little) on a bright one."""
        fx = effects.make("steam", BIG, 3, 144, 24, W, H)
        self.assertGreater(frame(fx, 9, base=0.05).max(), 0.15)
        bright = frame(fx, 9, base=0.75)
        self.assertGreater(bright.max(), 0.76)
        self.assertLessEqual(bright.max(), 1.0)

    def test_colour(self):
        """Not lit by the scene (no light, or lit 0): its own colour."""
        for light, lit in ((None, 70), (light_of((200, 120, 60)), 0)):
            fx = effects.make("steam", dict(BIG, color="#ff0000", opacity=100, lit=lit), 3, 144, 24, W, H, light)
            pic = frame(fx, 9, base=0.0)
            self.assertGreater(pic[..., 0].max(), 0.1)
            self.assertEqual(pic[..., 1:].max(), 0.0)

    def test_lit_by_the_scene(self):
        """Lit by the scene, steam shows the light that falls on it: warm under a warm lamp, dim in a dim room,
        brighter on the side the light comes from; its own colour only tints it; before a bright light behind it
        more of it shows (it glows)."""
        warm = effects.make("steam", BIG, 3, 144, 24, W, H, light_of((255, 170, 90)))
        dim = effects.make("steam", BIG, 3, 144, 24, W, H, light_of((40, 40, 40)))
        r, g, b = warm.colours[1]
        self.assertGreater(r, b + 0.1)
        self.assertLess(max(dim.colours[1]), 0.7 * max(warm.colours[1]))
        self.assertGreater(cover(warm, 9).sum(), 1.5 * cover(dim, 9).sum())  # (a bright light behind: it glows)
        side = effects.make("steam", BIG, 3, 144, 24, W, H, light_of((50, 40, 30), lamp="left"))
        self.assertLess(side.tone[6], -0.2)  # (the light is on the left)
        self.assertGreater(sum(side.colours[1]), sum(side.colours[0]) + 0.05)
        canvas = effects.Canvas(W, H)
        side.draw(9, canvas, None)
        a = canvas.a
        lum = canvas.rgb @ effects.LUMA / np.maximum(a, 1e-6)
        dx = np.zeros_like(a)
        dx[:, 1:-1] = a[:, 2:] - a[:, :-2]
        strong = a > 0.1
        left_edges = strong & (dx > 0.02)  # (rising to the right: the strand's left side, toward the light)
        right_edges = strong & (dx < -0.02)
        self.assertGreater(lum[left_edges].mean(), lum[right_edges].mean() + 0.02)
        blue = effects.make("steam", dict(BIG, color="#b0c4ff"), 3, 144, 24, W, H, light_of((200, 200, 200)))
        self.assertGreater(blue.colours[1][2], blue.colours[1][0])  # (its colour tints the light)

    def test_light_at(self):
        """The light of where a box stands: the brightest light around it, what is behind it, and its side."""
        light = light_of((50, 40, 30), 640, 360, lamp="left")
        tone = steam.light_at(light, (200, 60, 200, 200), 640, 360)
        self.assertEqual(tone.shape, (7,))
        self.assertGreater(tone[:3].min(), 0.8)  # (the lamp is near it)
        self.assertLess(tone[6], 0)
        far = steam.light_at(light, (440, 60, 200, 200), 640, 360)
        self.assertLess(far[:3].max(), tone[:3].min())
        self.assertGreater(far[6], tone[6])


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_styles_and_looks_set_them(self):
        params = steam.SCHEMA["params"]
        self.assertTrue(all(p.get("group") for p in params))
        by = {p["key"]: p for p in params}
        defaults = effects.defaults("steam")
        self.assertEqual((defaults["style"], defaults["look"]), ("cup", "light"))
        keys = set(by) - set(PLACE)
        # a style: every setting but the place, the light and the style itself (its own rim and colour)
        # (smoke from a chimney: Khói ống khói, fx/chimney.py, since 1.0.63)
        self.assertEqual([o["value"] for o in by["style"]["options"]], ["cup", "incense", "pot"])
        for name, preset in by["style"]["presets"].items():
            self.assertEqual(set(preset), keys - {"style", "lit"}, name)
            self.assertEqual(effects.clean_params("steam", preset), {**defaults, **preset}, name)  # in range
            self.assertEqual(preset["color"], steam.STYLES[name]["colour"])
        self.assertEqual(by["style"]["presets"]["cup"], {k: defaults[k] for k in keys - {"style", "lit"}})
        # a look: how it moves and how much of it there is (not its style, rim, light or colour)
        motion = keys - {"style", "look", "rim", "lit", "color"}
        for name, preset in by["look"]["presets"].items():
            self.assertEqual(set(preset), motion, name)
            self.assertEqual(effects.clean_params("steam", preset), {**defaults, **preset}, name)
        self.assertEqual({o["value"] for o in by["look"]["options"]}, set(steam.LOOKS))
        self.assertEqual(by["rim"]["min"], 1)  # (incense: a point)

    def test_mark(self):
        """The page's mark: an ellipse as wide as "rim", MARK as high as wide, sitting inside the box on its bottom
        edge; the steam rises from its middle."""
        mark = steam.SCHEMA["mark"]
        self.assertEqual(mark["width"], "rim")
        self.assertEqual(mark["ratio"], steam.MARK)
        self.assertTrue(mark["sits"] and mark["tip"] and mark["label"])
        for rim in (3, 50, 100):
            fx = effects.make("steam", {"rim": rim}, 1, 144, 24, W, H)
            self.assertAlmostEqual(fx.yc, steam.MARK / 2 * rim / 100)
            self.assertAlmostEqual(fx.e, fx.yc)

    def test_box(self):
        self.assertEqual(steam.Steam.BOX, (1, 1))
        fx = effects.make("steam", {}, 1, 144, 24, 1920, 1080)
        x0, y0, w, h = fx.box()
        self.assertAlmostEqual(h, 1080 * effects.defaults("steam")["scale"] / 100)
        self.assertAlmostEqual(x0 + w / 2, 960)

    def test_old_settings(self):
        """Settings saved before the styles (no "style") are a cup's steam with the same settings, drawn the new
        way, lit by the scene (the new default)."""
        old = {"look": "thick", "amount": 90, "rim": 60, "reach": 80, "spread": 55, "speed": 40, "curl": 45,
               "breeze": 6, "gust": 25, "soft": 55, "opacity": 75, "color": "#f4f1ec", "scale": 45, "x": 3, "y": -8}
        p = effects.clean_params("steam", old)
        self.assertEqual(p, {**effects.defaults("steam"), **old, "style": "cup"})
        self.assertEqual(effects.clean_params("steam", p), p)  # (once only)
        fx = effects.make("steam", old, 3, 144, 24, W, H)
        self.assertGreater(cover(fx, 5).max(), 0.05)

    def test_old_mark_stays_on_the_cup(self):
        """Its mark was centred on the box's bottom edge; now it sits inside the box. An old project's box moves down
        by half the mark's height (for the video's shape), so the mark stays where it was on the cup: once (a project
        saved since keeps its place)."""
        for w, h in ((1920, 1080), (1080, 1920), (640, 640)):
            old = {"rim": 60, "scale": 40, "x": 5, "y": -10}
            new = steam.upgrade_in(old, w, h)
            x0, y0, bw, bh = effects.box_of("steam", old, w, h)
            n0, ny0, nbw, nbh = effects.box_of("steam", new, w, h)
            centre = ny0 + nbh - steam.MARK / 2 * 0.6 * nbw  # the new mark's middle
            self.assertAlmostEqual(centre, y0 + bh, delta=0.03 * h / 100 + 1e-6)  # (y on steps of 0.05%)
            self.assertEqual((n0, nbw), (x0, bw))
            self.assertEqual(new["style"], "cup")
            self.assertEqual(steam.upgrade_in(new, w, h), new)  # (once only)
        self.assertEqual(effects.upgrade_in("snow", {"amount": 3}, 640, 360), {"amount": 3})

    def test_old_turned_mark_stays_on_the_cup(self):
        """Turned in 3D, the box is moved so that the new mark's middle is where the old one was in the frame."""
        from fx import overlay

        def mark(params, rim, now):
            p = effects.clean_params("steam", params)
            rect = effects.sheet_rect("steam", p, 1920, 1080)
            q = overlay.homography(p, rect, 1920, 1080)[0] @ (
                rect[2] / 2, rect[3] - (steam.MARK / 2 * rim / 100 * rect[2] if now else 0), 1.0)
            return q[:2] / q[2]
        for turn in ({"rot_z": 30}, {"rot_x": 50, "persp": 80}, {"rot_y": 40, "rot_z": 15, "persp": 60}):
            old = {"rim": 60, "scale": 45, "x": 3, "y": 10, **turn}
            new = steam.upgrade_in(old, 1920, 1080)
            self.assertLess(np.hypot(*(mark(old, 60, False) - mark(new, 60, True))), 1.0, turn)
            self.assertEqual(steam.upgrade_in(new, 1920, 1080), new)

    def test_old_project_loads(self):
        """An old project's steam clip: loaded, it is a cup's steam with its settings, its box moved so the mark
        stays on the cup; saved again, it keeps that place."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "steam-old.mp4"), 320, 180, 12)
        project = projects.create(src, "steam-old.mp4")
        project = projects.apply(project, {"layers": [{"id": "st01", "kind": "steam", "params": {}, "seed": 2}]})
        path = os.path.join(projects.folder(project["id"]), "project.json")
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        old = {"look": "light", "amount": 45, "rim": 50, "scale": 50, "x": 0, "y": -20, "opacity": 55}
        saved["layers"][0]["params"] = dict(old)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(saved, f)
        loaded = projects.load(project["id"])
        p = loaded["layers"][0]["params"]
        self.assertEqual(p["style"], "cup")
        self.assertAlmostEqual(p["y"], -20 + steam.MARK / 2 * 0.5 * 50, places=6)
        self.assertEqual(projects.load(project["id"])["layers"][0]["params"], p)  # (the same each time)
        again = projects.apply(loaded, {"layers": loaded["layers"]})
        self.assertEqual(projects.load(again["id"])["layers"][0]["params"]["y"],
                         effects.clean_params("steam", p)["y"])


class Speed(unittest.TestCase):
    def test_a_frame_is_quick(self):
        """A frame (the steam lit by the scene) drawn frame after frame as an export does, against laying a picture
        as big as its box over the frame (what any effect there costs at least), timed here on the same machine:
        a default 1080p frame about 2 times that (about 20 ms on a 4-core laptop; the old steam took 16 ms, 1.3
        times), a box at the finest grid or in 4K about as much as laying it."""
        def took(w, h, draw, flush):
            times = []
            for t in range(101, 125):
                t0 = time.perf_counter()
                draw(t)
                flush()
                times.append(time.perf_counter() - t0)
            return float(np.median(times))

        for (w, h), params, most in (((1920, 1080), {}, 3.0), ((1920, 1080), STYLES["pot"], 3.0),
                                     ((1920, 1080), dict(STYLES["incense"], scale=100), 2.0),
                                     ((1920, 1080), {"scale": 100}, 2.0), ((3840, 2160), {}, 2.0)):
            light = light_of((60, 45, 35), w, h, lamp="left")
            fx = effects.make("steam", params, 7, 720, 24, w, h, light)
            canvas = effects.Canvas(w, h)
            pic = np.zeros((h, w, 3), np.float32)
            fx.draw(100, canvas, None)
            canvas.flush(pic, fx.blend)
            steam_s = took(w, h, lambda t: fx.draw(t, canvas, None), lambda: canvas.flush(pic, fx.blend))
            x0, y0, bw, bh = (int(round(v)) for v in fx.box())
            sheet = np.random.default_rng(1).random((min(bh, h), min(bw, w))).astype(np.float32) * 0.5
            lay = took(w, h, lambda t: canvas.over(max(0, x0), max(0, y0), sheet, (0.9, 0.9, 0.9), 1.0, None),
                       lambda: canvas.flush(pic, "over"))
            self.assertLess(steam_s, most * lay, (w, h, params, steam_s, lay))
            self.assertLess(steam_s, 0.5, (w, h, params))


if __name__ == "__main__":
    unittest.main()
