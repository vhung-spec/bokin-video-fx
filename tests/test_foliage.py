"""Rustling leaves (fx/foliage.py): a lens that warps the picture inside the clip's area, so most checks go through
apply(). The motion is read back exactly from a ramp picture: bilinear sampling of a linear ramp gives the very
position it was read from (with "Lá lấp lánh" at 0: the shimmer would change the ramp's values)."""
import io
import os
import time
import unittest

import numpy as np
from PIL import Image, ImageFilter

from tests import helpers
from fx import effects  # (first: it loads every effect's module)
from fx import engine, foliage, paths, projects, video

LOOKS = list(foliage.LOOKS)
PRESETS = effects.SCHEMA["foliage"]["params"][0]["presets"]
K = 1 / 2048  # the ramp's step a pixel


def ellipse(w, h, cx=0.5, cy=0.55, rx=0.3, ry=0.35):
    """An area painted over a bush: an ellipse (centre and radii as fractions of the frame's height and width)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    return (((xx / w - cx) / rx) ** 2 + ((yy / h - cy) / ry) ** 2 < 1).astype(np.float32)


def texture(w, h, seed=1):
    """Something leafy to warp: smooth blobs and fine grain."""
    g = np.random.default_rng(seed)
    coarse = np.asarray(Image.fromarray(g.random((h // 8 + 2, w // 8 + 2, 3)).astype(np.float32)[..., 0], "F").resize(
        (w, h), Image.Resampling.BICUBIC), np.float32)
    fine = g.random((h, w, 3)).astype(np.float32)
    return np.clip(0.15 + 0.4 * coarse[..., None] + 0.3 * fine, 0, 1).astype(np.float32)


_RAMPS = {}


def ramp(w, h):
    if (w, h) not in _RAMPS:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _RAMPS[(w, h)] = np.dstack([xx * K, yy * K, np.full((h, w), 0.5, np.float32)]).astype(np.float32)
    return _RAMPS[(w, h)].copy()


def make(params=None, frames=900, fps=30, w=640, h=360, seed=3, look="gentle"):
    return effects.make("foliage", {**PRESETS[look], **(params or {})}, seed, frames, fps, w, h)


def moved(fx, t, mask, gain=1.0):
    """How far each pixel moved at frame t (dx, dy: pixels of this frame), read back from a ramp."""
    pic = ramp(fx.w, fx.h)
    before = pic.copy()
    fx.apply(t, pic, mask, gain)
    return -(pic[..., 0] - before[..., 0]) / K, -(pic[..., 1] - before[..., 1]) / K


def _painted_near(mask, margin):
    """How many painted pixels each (2 margin + 1)-square round a pixel holds (nothing beyond the frame)."""
    h, w = mask.shape
    s = np.pad((mask > 0.5).astype(np.int64), margin + 1).cumsum(0).cumsum(1)
    k = 2 * margin + 1
    return s[k:k + h, k:k + w] - s[:h, k:k + w] - s[k:k + h, :w] + s[:h, :w]


def inside(mask, margin):
    """The part of the area at least `margin` pixels in from its outline."""
    return _painted_near(mask, margin) == (2 * margin + 1) ** 2


def around(mask, margin):
    """The area and `margin` pixels round it."""
    return _painted_near(mask, margin) > 0


class Loops(unittest.TestCase):
    def test_frame_p_is_frame_0(self):
        """Every look, a breeze both ways, the slowest and quickest rustling: frame P is frame 0 again, whatever the
        loop's length and frame rate."""
        w, h = 320, 180
        pic, mask = texture(w, h), ellipse(w, h)
        cases = [{"look": look} for look in LOOKS] + [
            {"look": "strong", "direction": "left", "breeze": 100, "speed": 100, "shimmer": 100},
            {"look": "gentle", "speed": 0, "cluster": 0, "edge": 0}, {"look": "medium", "cluster": 100, "amount": 100}]
        for case in cases:
            params = dict(case)
            look = params.pop("look")
            for frames, fps in ((144, 24), (150, 25), (90, 30), (900, 30)):
                fx = make(params, frames, fps, w, h, look=look)
                a, b = pic.copy(), pic.copy()
                fx.apply(0, a, mask, 1.0)
                fx.apply(frames, b, mask, 1.0)
                self.assertLess(float(np.abs(a - b).max()), 1 / 512, (case, frames, fps))
                c = pic.copy()
                fx.apply(frames // 3, c, mask, 1.0)
                self.assertGreater(float(np.abs(c - a).max()), 0.02, (case, frames, fps))  # (and it does move)

    def test_moves_smoothly(self):
        """From one frame to the next each pixel moves a little: the sway is smooth, never a jitter."""
        fx = make({"shimmer": 0}, look="strong")
        mask = ellipse(640, 360)
        deep = inside(mask, 20)
        prev = None
        steps, most = [], 0.0
        for t in range(0, 60):
            dx, dy = moved(fx, t, mask)
            most = max(most, float(np.abs(dx[deep]).max()))
            if prev is not None:
                steps.append(float(np.abs(dx[deep] - prev[deep]).max()))
            prev = dx
        self.assertGreater(most, 0.3)
        self.assertLess(max(steps), 0.15 * most + 0.02, (max(steps), most))

    def test_no_drift_over_the_loop(self):
        """Averaged over the loop every pixel is where it was (a sway, not a slide; a breeze pushes and lets go)."""
        for look in ("gentle", "strong"):
            fx = make({"shimmer": 0}, 120, 30, 320, 180, look=look)
            mask = ellipse(320, 180)
            acc = np.zeros((180, 320), np.float32)
            for t in range(120):
                acc += moved(fx, t, mask)[0]
            deep = inside(mask, 10)
            self.assertLess(float(np.abs(acc / 120)[deep].max()), 0.12 * fx.px * 3, look)


class Area(unittest.TestCase):
    def test_nothing_without_an_area(self):
        fx = make(look="strong")
        pic = texture(640, 360)
        out = pic.copy()
        fx.apply(40, out, None, 1.0)
        self.assertTrue(np.array_equal(out, pic))
        fx.apply(40, out, np.zeros((360, 640), np.float32), 1.0)
        self.assertTrue(np.array_equal(out, pic))

    def test_outside_the_area_exactly_the_picture(self):
        pic = texture(640, 360)
        mask = ellipse(640, 360)
        for look in LOOKS:
            for gain in (1.0, 0.4):
                fx = make({"edge": 0}, look=look)
                out = pic.copy()
                fx.apply(77, out, mask, gain)
                self.assertTrue(np.array_equal(out[mask == 0], pic[mask == 0]), (look, gain))
                self.assertFalse(np.array_equal(out, pic), (look, gain))

    def test_motion_fades_toward_the_outline(self):
        """With "Giữ yên viền" a thin rim along the outline does not move: at the default the outermost 2 pixels of a
        1080-line frame, the motion full 10 pixels in; at 100 a wider rim; at 0 the leaves move up to the outline."""
        w, h = 1920, 1080
        mask = ellipse(w, h)
        band = (mask > 0) & ~inside(mask, 2)  # the outermost 2 pixels of the area
        mid = inside(mask, 5) & ~inside(mask, 7)
        ten = inside(mask, 10) & ~inside(mask, 12)
        deep = inside(mask, 40)

        def mean_moved(params):
            fx = make({"shimmer": 0, **params}, w=w, h=h, look="strong")
            out = []
            for t in range(0, 300, 60):
                d = np.hypot(*moved(fx, t, mask))
                out.append([float(d[band].mean()), float(d[mid].mean()), float(d[ten].mean()), float(d[deep].mean())])
            return np.mean(out, axis=0)
        near, between, tens, far = mean_moved({"edge": 40})
        self.assertLess(near, 0.1 * far)
        self.assertGreater(between, 0.4 * far)
        self.assertGreater(tens, 0.85 * far)
        near, between, tens, far = mean_moved({"edge": 100})
        self.assertLess(between, 0.1 * far)
        near, between, tens, far = mean_moved({"edge": 0})
        self.assertGreater(near, 0.5 * far)

    def test_moves_all_over_a_narrow_bush(self):
        """A bush painted as a strip 60 pixels tall (1080p) rustles all over it, not only in its middle: 8 pixels in
        from its outline the leaves move nearly as much as with no still rim at all."""
        w, h = 1920, 1080
        mask = np.zeros((h, w), np.float32)
        mask[500:560, 200:1700] = 1
        rows = np.zeros((h, w), bool)
        rows[508:552, 400:1500] = True

        def mean_moved(params):
            fx = make({"shimmer": 0, **params}, w=w, h=h, look="strong")
            return np.mean([float(np.hypot(*moved(fx, t, mask))[rows].mean()) for t in range(0, 300, 50)])
        self.assertGreater(mean_moved({}), 0.9 * mean_moved({"edge": 0}))

    def test_outline_of_a_bush_stays_still(self):
        """A bright bush on a dark ground, its area painted along its outline: the outline does not move (its
        outermost leaves barely), while inside the leaves rustle."""
        w, h = 1920, 1080
        mask = ellipse(w, h)
        pic = np.full((h, w, 3), 0.1, np.float32)
        pic[mask > 0] = texture(w, h)[mask > 0] * 0.5 + 0.45
        outline = around(mask, 2) & ~inside(mask, 2)  # from 2 px out to 2 px in
        deep = inside(mask, 40)
        for look in LOOKS:
            fx = make(look=look, w=w, h=h)
            for t in (0, 333):
                out = pic.copy()
                fx.apply(t, out, mask, 1.0)
                change = np.abs(out - pic).max(axis=2)
                self.assertLess(float(change[outline].max()), 10 / 255, (look, t))
                self.assertLess(float(change[outline].mean()), 0.1 * float(change[deep].mean()), (look, t))

    def test_crown_cut_by_the_frame_moves_there(self):
        """Where the area meets the frame's edge (a crown cut by the top of the picture) it is no outline: the
        leaves move up to the edge, and nothing is read from beyond it."""
        w, h = 640, 360
        mask = np.zeros((h, w), np.float32)
        mask[:120, 100:540] = 1
        fx = make({"shimmer": 0}, w=w, h=h, look="strong")
        top = np.zeros((h, w), bool)
        top[:3, 200:440] = True
        deep = np.zeros((h, w), bool)
        deep[40:80, 200:440] = True
        near, far = [], []
        for t in range(0, 300, 30):
            d = np.hypot(*moved(fx, t, mask))
            near.append(float(d[top].mean()))
            far.append(float(d[deep].mean()))
        self.assertGreater(np.mean(near), 0.5 * np.mean(far))
        out = texture(w, h)
        fx.apply(50, out, mask, 1.0)
        self.assertTrue(np.isfinite(out).all())

    def test_gain_fades_the_motion(self):
        """A fade makes the leaves sway less (no crossfade with the still picture: no doubled leaves)."""
        fx = make({"shimmer": 0}, look="strong")
        mask = ellipse(640, 360)
        deep = inside(mask, 30)
        full = moved(fx, 210, mask)[0]
        half = moved(fx, 210, mask, 0.5)[0]
        self.assertLess(float(np.abs(half - 0.5 * full)[deep].max()), 2e-3)


class Motion(unittest.TestCase):
    """At 1080p, against the reference measured in a real video of bushes in a light breeze (0.5 to 1.5 px, clusters
    nearly independent beyond about 30 px, periods of 2 to 4 s and slower drifts)."""

    @classmethod
    def setUpClass(cls):
        w, h = 1920, 1080
        cls.mask = np.zeros((h, w), np.float32)
        cls.mask[300:700, 600:1100] = 1  # (a square area; the points read lie well inside it)
        cls.series = {}
        for look, ts in (("gentle", range(0, 450, 3)), ("strong", range(0, 450, 9))):  # (15 s)
            fx = make({"shimmer": 0, "edge": 0}, 900, 30, w, h, look=look)
            cls.series[look] = np.stack([np.stack(moved(fx, t, cls.mask))[:, 380:620:2, 680:1020:2] for t in ts])

    def test_amplitude_like_the_reference(self):
        s = self.series["gentle"]  # (time, axis, y, x)
        sd = float(s.std())
        self.assertGreater(sd, 0.3)  # pixels of a 1080-line frame, each axis
        self.assertLess(sd, 0.6)
        self.assertLess(float(np.percentile(np.hypot(s[:, 0], s[:, 1]), 99.9)), 2.5)
        self.assertGreater(float(self.series["strong"].std()), 1.3 * sd)
        self.assertLess(float(np.percentile(np.abs(self.series["strong"]), 99.9)), 6.0)

    def test_clusters_sway_on_their_own(self):
        s = self.series["gentle"][:, 0]  # dx (time, y, x), every 2nd pixel

        def corr(dy, dx):
            a = s[:, :s.shape[1] - dy, :s.shape[2] - dx].reshape(len(s), -1)
            b = s[:, dy:, dx:].reshape(len(s), -1)
            a, b = a - a.mean(0), b - b.mean(0)
            return float(((a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0))).mean())
        self.assertGreater(corr(0, 1), 0.85)  # 2 px apart: one cluster
        self.assertLess(corr(0, 15), 0.3)  # 30 px apart: nearly independent
        self.assertLess(abs(corr(20, 20)), 0.15)  # 56 px apart: independent

    def test_periods_of_a_few_seconds(self):
        s = self.series["gentle"][:, 0].reshape(150, -1)
        spec = np.abs(np.fft.rfft(s - s.mean(0), axis=0)).mean(axis=1)
        hz = np.fft.rfftfreq(150, 3 / 30)
        peak = float(hz[1:][np.argmax(spec[1:])])
        self.assertGreater(peak, 0.1)
        self.assertLess(peak, 0.6)
        fast = make({"shimmer": 0, "speed": 100}, 900, 30, 320, 180)
        slow = make({"shimmer": 0, "speed": 0}, 900, 30, 320, 180)
        self.assertGreater(min(fast.octaves[0]["m"]), max(slow.octaves[0]["m"]))

    def test_breeze_pushes_one_way(self):
        w, h = 640, 360
        mask = ellipse(w, h)
        deep = inside(mask, 20)
        for side in ("right", "left"):
            fx = make({"shimmer": 0, "breeze": 100, "direction": side, "amount": 0}, 300, 30, w, h)
            mean = np.array([float(moved(fx, t, mask)[0][deep].mean()) for t in range(0, 300, 5)])
            sign = 1 if side == "right" else -1
            self.assertGreater(sign * mean.max() if sign > 0 else -mean.min(), 0.3 * fx.px * 3, side)
            self.assertLess(abs(float(mean.mean())), 0.25 * float(np.abs(mean).max()), side)  # (pushed and let go)


class Shimmer(unittest.TestCase):
    def test_a_few_percent_of_light(self):
        """Leaves catching the light: a flat patch of leaves gets a few percent brighter or darker, no more."""
        w, h = 640, 360
        mask = ellipse(w, h)
        deep = inside(mask, 20)
        flat = np.full((h, w, 3), 0.5, np.float32)
        fx = make(look="gentle", w=w, h=h)
        swings = []
        for t in range(0, 900, 45):
            out = flat.copy()
            fx.apply(t, out, mask, 1.0)
            swings.append(out[..., 1][deep] / 0.5 - 1)
        swings = np.concatenate(swings)
        self.assertGreater(float(swings.std()), 0.01)
        self.assertLess(float(np.abs(swings).max()), 0.2)
        out = flat.copy()
        make({"shimmer": 0}, w=w, h=h).apply(99, out, mask, 1.0)
        self.assertLess(float(np.abs(out - flat).max()), 1e-6)  # (a flat picture warped is the same flat picture)


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_label(self):
        for p in effects.SCHEMA["foliage"]["params"]:
            self.assertTrue(p.get("group") and p.get("label"), p["key"])
        self.assertIn("Vẽ vùng", effects.SCHEMA["foliage"]["note"])

    def test_looks(self):
        keys = {p["key"] for p in effects.SCHEMA["foliage"]["params"]} - {"look"}
        for look in LOOKS:
            self.assertEqual(set(PRESETS[look]), keys, look)
        a, b, c = (effects.clean_params("foliage", PRESETS[k]) for k in ("gentle", "medium", "strong"))
        self.assertLess(a["amount"], b["amount"])
        self.assertLess(b["amount"], c["amount"])
        self.assertLess(a["breeze"], c["breeze"])

    def test_a_lens_that_does_not_turn_or_move(self):
        cls = effects.KINDS["foliage"]
        self.assertTrue(cls.lens and cls.UNDER and cls.AREA)
        self.assertFalse(cls.TURNS or cls.BOX)
        self.assertFalse(effects.on_sheet("foliage", {"rot_y": 30, "x": 10}))


class Same(unittest.TestCase):
    def test_same_seed_same_picture(self):
        pic, mask = texture(320, 180), ellipse(320, 180)
        a, b, c = pic.copy(), pic.copy(), pic.copy()
        make(seed=4, w=320, h=180).apply(31, a, mask, 1.0)
        make(seed=4, w=320, h=180).apply(31, b, mask, 1.0)
        make(seed=5, w=320, h=180).apply(31, c, mask, 1.0)
        self.assertTrue(np.array_equal(a, b))
        self.assertFalse(np.array_equal(a, c))

    def test_the_same_motion_at_any_size(self):
        """At the preview's size the leaves move as in the export (the same field, scaled): what the live preview
        shows is what the video gets."""
        big = make({"shimmer": 0}, 900, 30, 1920, 1080, look="medium")
        small = make({"shimmer": 0}, 900, 30, 1280, 720, look="medium")
        mb, ms = ellipse(1920, 1080), ellipse(1280, 720)
        for t in (0, 200, 555):
            bx = moved(big, t, mb)[0]
            sx = moved(small, t, ms)[0]
            # the big frame's motion at the small frame's pixel centres (as the preview is the picture made small)
            yy, xx = np.mgrid[0:720, 0:1280].astype(np.float32)
            by = np.clip(((yy + 0.5) * 1.5 - 0.5).astype(int), 0, 1079)
            bxx = np.clip(((xx + 0.5) * 1.5 - 0.5).astype(int), 0, 1919)
            want = bx[by, bxx] / 1.5
            deep = inside(ms, 30)
            r = float(np.corrcoef(want[deep], sx[deep])[0, 1])
            self.assertGreater(r, 0.97, t)
            self.assertAlmostEqual(float(sx[deep].std() / want[deep].std()), 1.0, delta=0.06)

    def test_any_order_of_frames(self):
        """A frame is the same whether frames before it were drawn or not (the renderer's pieces start anywhere)."""
        pic, mask = texture(320, 180), ellipse(320, 180)
        fx = make(w=320, h=180, look="strong")
        a = pic.copy()
        fx.apply(400, a, mask, 1.0)
        fresh = make(w=320, h=180, look="strong")
        for t in (3, 77, 2):
            fresh.apply(t, pic.copy(), mask, 1.0)
        b = pic.copy()
        fresh.apply(400, b, mask, 1.0)
        self.assertTrue(np.array_equal(a, b))


class Speed(unittest.TestCase):
    def test_fast_enough(self):
        """1080p, an area filling a quarter of the frame: well within a frame's share of a render (aim: under 60 ms
        on a 4-core laptop; the bound here leaves room for slow test machines)."""
        w, h = 1920, 1080
        mask = ellipse(w, h, 0.25, 0.75, 0.25, 0.25)
        pic = texture(w, h)
        fx = make(w=w, h=h, look="medium")
        fx.apply(0, pic.copy(), mask, 1.0)
        pics = [pic.copy() for _ in range(8)]
        t0 = time.perf_counter()
        for t, p in enumerate(pics):
            fx.apply(t * 11, p, mask, 1.0)
        ms = (time.perf_counter() - t0) / len(pics) * 1000
        print(f"\nrustling leaves, 1080p, a quarter of the frame: {ms:.0f} ms a frame")
        self.assertLess(ms, 150)


class Preview(unittest.TestCase):
    """Its preview clip: the picture under the clip's area, rustling, only its change kept, made again when the
    area is painted."""

    def test_preview_keeps_only_its_change(self):
        w, h = 320, 180
        pic, mask = texture(w, h), ellipse(w, h)
        after = pic.copy()
        make(w=w, h=h, look="strong").apply(50, after, mask, 1.0)
        rgba = engine.change_rgba(pic, after, 1.0).astype(np.float32) / 255
        a = rgba[..., 3:]
        laid = rgba[..., :3] * a + pic * (1 - a)
        self.assertLess(float(np.abs(laid - np.clip(after, 0, 1)).max()), 1.01 / 255)
        far = ~around(mask, 8)
        self.assertEqual(float(a[..., 0][far].max()), 0.0)  # outside the area: clear

    def test_max5_is_pillows_max_filter(self):
        """change_rgba's spread (max5) is exactly Pillow's MaxFilter(5), only faster: older clips stay the same."""
        g = np.random.default_rng(2)
        for shape in ((50, 70), (3, 9), (1, 1), (7, 2)):
            a = (g.random(shape) ** 4 * 255).astype(np.uint8)
            want = np.asarray(Image.fromarray(a).filter(ImageFilter.MaxFilter(5)))
            self.assertTrue(np.array_equal(engine.max5(a), want), shape)

    def test_bands_cover_the_area(self):
        g = np.random.default_rng(5)
        where = np.zeros((200, 300), bool)
        where[10:60, 200:290] = g.random((50, 90)) > 0.9  # (a crown up right, a bush down left)
        where[150:190, 10:40] = True
        cover = np.zeros_like(where)
        for x0, y0, x1, y1 in engine._bands_of(where, 8):
            cover[y0:y1, x0:x1] = True
        self.assertTrue(cover[around(where.astype(np.float32), 8)].all())
        self.assertLess(cover.mean(), 0.9)
        self.assertEqual(engine._bands_of(np.zeros((10, 10), bool), 8), [])

    def test_name_follows_the_area(self):
        """The preview clip is made again when the clip's area is painted differently, or a still area is painted
        near it; not for a still area elsewhere, nor when nothing is painted (an empty clip), nor for the clip's
        "Mép mềm" (the area is taken as painted: engine.area_feather)."""
        info = {"w": 480, "h": 270, "fps": "24", "frames": 144, "seconds": 6.0, "file": "input.mp4"}
        project = {"id": "f0l1agetest", "video": info, "length": 30, "freeze": {"feather": 6},
                   "still": {"source": "frame", "frame": 0}}
        layer = {"id": "leaf01", "kind": "foliage", "params": {}, "seed": 2}
        folder = os.path.join(paths.projects_dir(), project["id"])
        os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
        stamp = [0]

        def paint(name, *rects, value=255):
            m = Image.new("L", (480, 270), 0)
            for x, y, w, h in rects:
                m.paste(value, (x, y, x + w, y + h))
            path = engine.mask_path(folder, name)
            m.save(path)
            stamp[0] += 1
            os.utime(path, ns=(1, 10 ** 9 * stamp[0]))
            return engine.fx_clip_name(project, layer)

        nowhere = engine.fx_clip_name(project, layer)
        self.assertTrue(nowhere.endswith(".webm"))
        self.assertEqual(paint("leaf01"), nowhere)  # nothing painted
        self.assertEqual(paint("leaf01", (0, 0, 480, 270)), nowhere)  # painted all over: the whole frame
        bush = paint("leaf01", (300, 150, 120, 80))
        self.assertNotEqual(bush, nowhere)
        self.assertNotEqual(paint("leaf01", (300, 150, 121, 80)), bush)
        self.assertEqual(paint("leaf01", (300, 150, 120, 80)), bush)  # (undone: the clip made before)
        self.assertEqual(engine.fx_clip_name(project, dict(layer, feather=12)), bush)
        self.assertEqual((engine.area_feather(dict(layer, feather=12)), engine.area_feather({"kind": "snow", "feather": 12}),
                          engine.area_feather({"kind": "file", "feather": 7})), (0, 12, 7))
        self.assertNotEqual(engine.fx_clip_name(project, dict(layer, params={"amount": 70})), bush)
        self.assertEqual(paint("freeze", (10, 10, 40, 40)), bush)  # a still area far away
        self.assertNotEqual(paint("freeze", (290, 160, 20, 20)), bush)  # one on the bush
        self.assertNotEqual(paint("freeze", (270, 140, 10, 10)), bush)  # one just beside it (its soft edge)

    def test_preview_clip_shows_what_the_export_does(self):
        """Made for a real project: the clip laid on the video (as the page lays it) moves the leaves as the exact
        frame does."""
        from fx import timeline
        work = os.path.join(helpers.ROOT, "foliage-preview")
        os.makedirs(work, exist_ok=True)
        w, h = 320, 180
        src = os.path.join(work, "still.mp4")
        still = (texture(w, h, 7) * 255 + 0.5).astype(np.uint8)
        enc = video.Encoder(src, w, h, "24", crf=12, preset="veryfast")
        for _ in range(24):
            enc.write(still)
        enc.close()
        project = projects.create(src, "bush.mp4")
        folder = projects.folder(project["id"])
        total = timeline.project_frames(project)
        project = projects.apply(project, {"tracks": project["tracks"], "layers": [
            {"id": "leaf01", "kind": "foliage", "params": PRESETS["strong"], "seed": 3,
             "track": project["tracks"][0]["id"], "start": 0, "end": total}]})
        mw, mh = projects.mask_size(project["video"])
        buf = io.BytesIO()
        Image.fromarray((ellipse(mw, mh) * 255).astype(np.uint8)).save(buf, "PNG")
        project = projects.save_mask(project, "leaf01", buf.getvalue())
        name = engine.fx_clip(folder, project, "leaf01", processes=1)
        self.assertEqual(name, engine.fx_clip_name(project, project["layers"][0]))
        path = os.path.join(folder, "clips", name)
        info = video.probe(path, overlay=True)
        self.assertTrue(info["alpha"])
        self.assertEqual((info["w"], info["h"], info["frames"]), (w, h, total))
        with video.Frames(path, info, alpha=True, count=60) as frames:
            clip = [f.astype(np.float32) for f in frames]
        base = video.read_frame(engine.input_path(folder, project), project["video"], 0).astype(np.float32)
        looks = engine.Looks()
        deep = inside(ellipse(w, h), 12)
        corr = []
        for t in (10, 30, 55):
            a = clip[t][..., 3:] / 255
            live = base * (1 - a) + clip[t][..., :3] * a
            exact = looks.frame(folder, project, t, w).astype(np.float32)
            corr.append(float(np.corrcoef((live - base)[deep].ravel(), (exact - base)[deep].ravel())[0, 1]))
            self.assertLess(float(np.abs(live - exact).mean()), 3, t)  # (levels of 255, the whole frame)
        self.assertGreater(min(corr), 0.8, corr)


if __name__ == "__main__":
    unittest.main()
