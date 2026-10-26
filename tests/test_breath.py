"""A sleeping dog breathing (fx/breath.py): a placed lens effect, the picture's own pixels moved up and back down;
the clip's area (the dog painted) is what moves."""
import io
import math
import os
import unittest

import numpy as np
from PIL import Image

from tests import helpers
from fx import effects  # (first: it loads every effect's module)
from fx import breath, engine, paths, projects, video

W, H = 480, 270
BOX = {"scale": 80, "x": 0, "y": 5}  # (a box 288 x 216, its bottom edge at y 256)
STEADY = {"vary": 0}


def make(params=None, frames=300, fps=30, w=W, h=H, seed=3):
    return effects.make("breath", {**BOX, **(params or {})}, seed, frames, fps, w, h)


def lens(fx, t, pic, mask=None, gain=1.0, kept=None):
    out = pic.copy()
    fx.apply(t, out, mask, gain, kept)
    return out


def body(fx):
    """The body's outline in the frame, in pixels: (left, top, right, bottom)."""
    x0, y0, w, h = fx.box()
    bw, bh = fx.body
    return x0 + (0.5 - bw / 2) * w, y0 + (1 - bh) * h, x0 + (0.5 + bw / 2) * w, y0 + h


def stripes(w=W, h=H, period=12.0):
    """Level stripes, smooth (no pixel is a step): something whose rise shows in every column."""
    yy = np.mgrid[0:h, 0:w][0].astype(np.float32) + 0.5
    v = 0.5 + 0.35 * np.sin(2 * np.pi * yy / period)
    return np.dstack([v, 0.8 * v + 0.1, 0.6 * v + 0.2]).astype(np.float32)


def scene(fx):
    """A dog on its bed by a wall: a golden body (an ellipse standing on the bed line, its top on the body's top),
    a dark wall with mortar lines behind it, a head beside it, a striped bed below."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    left, top, right, bottom = body(fx)
    pic = np.dstack([0.22 + 0.05 * ((yy // 9) % 2), 0.18 + 0.04 * ((xx // 23) % 2), 0.15 + 0 * xx]).astype(np.float32)
    cx, cy = (left + right) / 2, bottom
    dog = ((xx - cx) / ((right - left) / 2)) ** 2 + ((yy - cy) / (bottom - top)) ** 2 < 1
    fur = 0.05 * np.sin(xx * 0.9) * np.sin(yy * 1.3)
    pic[dog] = np.stack([0.85 + fur, 0.55 + fur, 0.25 + 0 * fur], -1)[dog]
    head = ((xx - left + 10) ** 2 + (yy - bottom + 25) ** 2) < 22 ** 2
    pic[head] = (0.8, 0.5, 0.22)
    pic[yy > bottom] = (0.5, 0.42, 0.35)
    return pic


def dog_area(fx, w=W, h=H):
    """The dog of scene() painted (its body and head, a little past its fur's edge), as a clip's area: 0..1."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32) + 0.5
    left, top, right, bottom = (v * w / W if i % 2 == 0 else v * h / H for i, v in enumerate(body(fx)))
    cx, cy = (left + right) / 2, bottom
    dog = ((xx - cx) / ((right - left) / 2 + 1)) ** 2 + ((yy - cy) / (bottom - top + 1)) ** 2 < 1
    head = ((xx - left + 10 * w / W) ** 2 + (yy - bottom + 25 * h / H) ** 2) < (23 * h / H) ** 2
    return ((dog | head) & (yy < bottom)).astype(np.float32)


def edge_at(pic, x, y_from, y_to):
    """Where (rows, a fraction) the picture's first channel crosses the middle of its range going down column x."""
    col = pic[y_from:y_to, x, 0]
    mid = (col.min() + col.max()) / 2
    i = int(np.flatnonzero((col[:-1] < mid) != (col[1:] < mid))[0])
    return y_from + i + 0.5 + (mid - col[i]) / (col[i + 1] - col[i])


class Breathing(unittest.TestCase):
    def test_frame_p_is_frame_0(self):
        for frames, fps in ((144, 24), (150, 25), (900, 30)):
            for params in ({}, {"vary": 100, "rate": 40, "amount": 100}, {"rate": 6, "frame_h": 80, "frame_w": 20}):
                fx = make(params, frames, fps)
                self.assertEqual(fx.level(0), fx.level(frames), (frames, fps, params))
                pic = stripes()
                self.assertTrue(np.array_equal(lens(fx, 0, pic), lens(fx, frames, pic)), (frames, fps, params))

    def test_breaths_a_loop(self):
        """As many breaths a loop as the rate's nearest whole number (at least one)."""
        for rate, frames, fps, want in ((18, 900, 30, 9), (12, 900, 30, 6), (40, 1800, 30, 40), (26, 900, 30, 13),
                                       (6, 144, 24, 1)):
            fx = make({"rate": rate, **STEADY}, frames, fps)
            levels = np.array([fx.level(t / 8) for t in range(8 * frames)])
            ups = int(((levels[:-1] < 0.5) & (levels[1:] >= 0.5)).sum() + (levels[-1] < 0.5 <= levels[0]))
            self.assertEqual(ups, want, (rate, frames))
            self.assertAlmostEqual(float(levels.max()), 1.0, 3)

    def test_smooth(self):
        """No jump: from frame to frame the belly moves a little, its speed changes smoothly; where one breath
        becomes the next, and where the loop closes, it is at rest."""
        for params in ({}, {"vary": 100, "rate": 30}):
            fx = make(params)
            levels = np.array([fx.level(t) for t in range(-2, fx.P + 3)])
            self.assertLess(float(np.abs(np.diff(levels)).max()), 0.08, params)
            self.assertLess(float(np.abs(np.diff(levels, 2)).max()), 0.03, params)
            for end in np.concatenate([[0.0], fx.ends]):  # (each breath's end, the next one's start)
                t = ((end - fx.start) % 1.0) * fx.P
                self.assertLess(fx.level(t), 1e-9, (params, end))
                self.assertLess(max(fx.level(t + d) for d in (-0.25, 0.25)), 0.002, (params, end))  # (no slope)

    def test_a_breath(self):
        """In smoothly, out by itself (quicker at first), then a pause at rest; no slope at either end."""
        self.assertEqual(breath.breath(0.0), 0.0)
        self.assertAlmostEqual(breath.breath(breath.INHALE), 1.0)
        self.assertEqual(breath.breath(0.95), 0.0)
        self.assertLess(breath.breath(0.01), 0.003)
        self.assertGreater(breath.breath(breath.INHALE + 0.3 * breath.EXHALE), 0.5)  # (out quickly at first)
        self.assertLess(1 - breath.breath(breath.INHALE + 0.01), 0.004)
        rising = [breath.breath(p) for p in np.linspace(0, breath.INHALE, 50)]
        falling = [breath.breath(p) for p in np.linspace(breath.INHALE, 1, 80)]
        self.assertTrue(all(b >= a for a, b in zip(rising, rising[1:])))
        self.assertTrue(all(b <= a for a, b in zip(falling, falling[1:])))

    def test_each_breath_its_own(self):
        """Breaths differ in length and depth as much as "Mỗi hơi thở một khác" says (not at all at 0), the longer
        ones deeper; on average as deep as "Bụng phồng" says."""
        same = make(STEADY, 900, 30)
        self.assertTrue(np.allclose(np.diff(np.concatenate([[0], same.ends])), 1 / len(same.ends)))
        self.assertTrue(np.allclose(same.depths, 1.0))
        some = make({"vary": 100}, 900, 30)
        lengths = np.diff(np.concatenate([[0], some.ends]))
        self.assertGreater(float(lengths.max() / lengths.min()), 1.1)
        self.assertGreater(float(some.depths.max() - some.depths.min()), 0.15)
        self.assertAlmostEqual(float(some.depths.mean()), 1.0)
        self.assertGreater(float(np.corrcoef(lengths, some.depths)[0, 1]), 0.0)

    def test_two_dogs_not_together(self):
        a, b = make({}, seed=3), make({}, seed=4)
        self.assertGreater(max(abs(a.level(t) - b.level(t)) for t in range(0, 300, 5)), 0.5)


class Picture(unittest.TestCase):
    def test_at_rest_as_it_was(self):
        """Between breaths (breathed out) the picture is exactly as it was."""
        fx = make()
        pic = scene(fx)
        rest = [t for t in range(fx.P) if fx.level(t) == 0.0]
        self.assertGreater(len(rest), 10)
        for t in rest[::5]:
            self.assertTrue(np.array_equal(lens(fx, t, pic), pic), t)

    def test_the_top_of_the_body_rises(self):
        """The top of the body rises as far as "Bụng phồng" says in its middle (RISE of the body's height at 100), less
        toward its sides; the bed stays."""
        for amount in (50, 100):
            fx = make({"amount": amount, **STEADY})
            pic = scene(fx)
            t = max(range(fx.P), key=fx.level)
            out = lens(fx, t, pic)
            left, top, right, bottom = body(fx)
            want = fx.level(t) * breath.RISE * amount / 100 * (bottom - top)
            mid = int((left + right) / 2)
            got = edge_at(pic, mid, int(top) - 15, int(top) + 15) - edge_at(out, mid, int(top) - 15, int(top) + 15)
            self.assertAlmostEqual(got, want, delta=0.15 * want, msg=amount)
            near_side = int(left + 0.12 * (right - left))
            y = int(top + 0.45 * (bottom - top))  # (the ellipse's outline there)
            side = edge_at(pic, near_side, y - 25, y + 25) - edge_at(out, near_side, y - 25, y + 25)
            self.assertLess(side, 0.5 * got, amount)
            self.assertTrue(np.array_equal(out[int(bottom) + 1:], pic[int(bottom) + 1:]), amount)  # (the bed)

    def test_nothing_beside_or_above_moves(self):
        """Nothing moves beside the body (its head), above the band the rise fades out over, or outside the box."""
        fx = make({"amount": 100, "vary": 100})
        pic = scene(fx)
        left, top, right, bottom = body(fx)
        band = top - breath.FADE * (bottom - top)
        still = np.ones((H, W), bool)
        still[max(0, math.floor(band)):math.ceil(bottom), math.floor(left):math.ceil(right)] = False
        moved = np.zeros((H, W), bool)
        for t in range(0, fx.P, 7):
            moved |= np.abs(lens(fx, t, pic) - pic).max(axis=2) > 0
        self.assertFalse((moved & still).any())
        self.assertGreater(int(moved.sum()), 0.2 * int((~still).sum()))

    def test_the_band_above_the_body(self):
        """Above the body the rise fades out over FADE of its height (the wall behind its back is pressed a little,
        then let go): fully at the top of the body, half way up the band about half, nothing at its top; never
        higher than the box's top edge."""
        fx = make({"amount": 100, **STEADY})
        rows, cols = fx._profiles((0, 0, W, H), *fx.box())
        x0, y0, w, h = fx.box()
        left, top, right, bottom = body(fx)
        band = breath.FADE * (bottom - top)

        def at(y):
            return float(np.interp(y, np.arange(H) + 0.5, rows))
        self.assertAlmostEqual(at(top), 1.0, 2)
        self.assertAlmostEqual(at(top - band / 2), 0.5, 1)
        self.assertLess(at(top - band), 0.01)
        self.assertLess(at(bottom - 0.5), 0.001)
        tall = make({"amount": 100, "frame_h": 80})
        rows, _ = tall._profiles((0, 0, W, H), *tall.box())
        self.assertTrue(np.all(rows[:math.floor(tall.box()[1])] == 0))  # (the box's top edge: the band cut short)
        self.assertGreater(float(rows.max()), 0.99)

    def test_never_folds(self):
        """The warp never folds: down a column each pixel is read from no higher than the one above it (the slope of
        the rise stays under 1), however far it rises and however short the band above the body."""
        for params in ({"amount": 100, "vary": 100, "frame_h": 80}, {"amount": 100, "vary": 100, "frame_h": 20},
                       {"amount": 100, "vary": 100, "frame_w": 20, "scale": 30}):
            fx = make(params)
            rows, cols = fx._profiles((0, 0, W, H), *fx.box())
            most = max(fx.level(t / 4) for t in range(4 * fx.P)) * fx.rise * fx.box()[3]
            slope = np.abs(np.diff(rows)).max() * most
            self.assertLess(float(slope), breath.STEEPEST + 0.02, params)

    def test_sharp(self):
        """Fine detail stays sharp as it moves (a cubic read): fine stripes half a pixel up keep most of their
        contrast (a linear read keeps 71% of stripes 4 pixels apart there), and nothing rings past the pixels it
        lies between."""
        fx = make({"amount": 100, **STEADY})
        pic = stripes(period=4.0)
        x0, y0, w, h = fx.box()
        left, top, right, bottom = body(fx)
        mid = int((left + right) / 2)
        for t in range(fx.P):
            out = lens(fx, t, pic)
            col = fx.level(t) * fx.rise * h  # (how far the middle column's top rises now)
            if abs(col % 1 - 0.5) < 0.08:
                break
        else:
            self.fail("no half pixel")
        rows = slice(int(top) - 3, int(top) + 4)
        swing = float(np.ptp(out[rows, mid, 0]))
        self.assertGreater(swing, 0.8 * float(np.ptp(pic[rows, mid, 0])))
        self.assertLessEqual(float(out[..., 0].max()), float(pic[..., 0].max()) + 1e-6)
        self.assertGreaterEqual(float(out[..., 0].min()), float(pic[..., 0].min()) - 1e-6)

    def test_mask_and_gain(self):
        """A fade (gain) rises less, not a crossfade with the still picture (no doubled outline); where the clip's
        area is 0 and nothing of it rises into, nothing moves."""
        fx = make({"amount": 100, **STEADY})
        pic = scene(fx)
        t = max(range(fx.P), key=fx.level)
        left, top, right, bottom = body(fx)
        mid = int((left + right) / 2)
        y0, y1 = int(top) - 15, int(top) + 15
        full = edge_at(pic, mid, y0, y1) - edge_at(lens(fx, t, pic), mid, y0, y1)
        half_pic = lens(fx, t, pic, gain=0.5)
        half = edge_at(pic, mid, y0, y1) - edge_at(half_pic, mid, y0, y1)
        self.assertAlmostEqual(half, full / 2, delta=0.1 * full)
        col = half_pic[y0:y1, mid, 0]
        crossings = int(((col[:-1] < 0.5) != (col[1:] < 0.5)).sum())
        self.assertEqual(crossings, 1)  # (one outline, not two)
        mask = np.ones((H, W), np.float32)
        mask[:, mid:] = 0
        out = lens(fx, t, pic, mask)
        self.assertTrue(np.array_equal(out[:, mid:], pic[:, mid:]))
        self.assertFalse(np.array_equal(out[:, :mid], pic[:, :mid]))
        self.assertTrue(np.array_equal(lens(fx, t, pic, gain=0.0), pic))

    def test_the_dog_painted_moves_over_the_wall(self):
        """The clip's area is what moves (the owner, 2026-10-26: "tự tô phần bụng con chó ... để không dính hiệu ứng
        vào tường"): the dog painted rises as without an area, its back over the wall; the wall and the bed, not
        painted, stay exactly as they were (never stretched), but where the back rose over them."""
        fx = make({"amount": 100, **STEADY})
        pic = scene(fx)
        area = dog_area(fx)
        t = max(range(fx.P), key=fx.level)
        out = lens(fx, t, pic, area)
        plain = lens(fx, t, pic)
        left, top, right, bottom = body(fx)
        mid = int((left + right) / 2)
        y0, y1 = int(top) - 15, int(top) + 15
        risen = edge_at(pic, mid, y0, y1) - edge_at(out, mid, y0, y1)
        self.assertAlmostEqual(risen, edge_at(pic, mid, y0, y1) - edge_at(plain, mid, y0, y1), delta=0.2)
        lift = fx.level(t) * fx.rise * fx.box()[3]
        self.assertGreater(risen, 0.8 * lift)
        # the wall: exactly as it was, but for the few pixels over the back the dog rose into
        reach = np.zeros_like(area, bool)
        for d in range(int(math.ceil(lift)) + 2):
            reach[:H - d] |= area[d:] > 0
        wall = ~reach
        self.assertTrue(np.array_equal(out[wall], pic[wall]))
        self.assertFalse(np.array_equal(plain[wall], pic[wall]))  # (without an area the wall is pressed)
        covered = (area == 0) & reach & (np.abs(out - pic).max(axis=2) > 0.1)
        self.assertGreater(int(covered[:, mid - 20:mid + 20].sum()), 40)  # (the back rose over the wall there)
        inside = area > 0
        deep = np.zeros_like(inside)
        deep[int(top) + 8:int(bottom) - 8, int(left) + 30:int(right) - 30] = True
        self.assertTrue(np.allclose(out[deep & inside], plain[deep & inside], atol=1e-6))  # (inside: as without)

    def test_the_dog_painted_softly(self):
        """A new clip's area is soft over a few pixels (SCHEMA "feather"; the page's own is 20): the back still rises
        nearly all the way. (Soft within what is painted, what moves near the outline moves only partly: an edge as
        soft as the page's own 20 pixels would keep the back from rising.)"""
        from PIL import ImageFilter
        fx = make({"amount": 100, **STEADY})
        pic = scene(fx)
        t = max(range(fx.P), key=fx.level)
        left, top, right, bottom = body(fx)
        mid = int((left + right) / 2)
        y0, y1 = int(top) - 15, int(top) + 15
        painted = Image.fromarray((dog_area(fx) * 255).astype(np.uint8))

        def risen(feather):  # (as engine.mask_image softens a clip's area: within its outline)
            m = painted.filter(ImageFilter.GaussianBlur(feather)).point(engine.INSIDE) if feather else painted
            out = lens(fx, t, pic, np.asarray(m, np.float32) / 255)
            return edge_at(pic, mid, y0, y1) - edge_at(out, mid, y0, y1)
        lift = fx.level(t) * fx.rise * fx.box()[3]
        self.assertLessEqual(breath.SCHEMA["feather"], 3)
        self.assertGreater(risen(breath.SCHEMA["feather"]), 0.75 * lift)
        self.assertLess(risen(20), 0.5 * lift)

    def test_the_dog_painted_turned(self):
        """Turned in 3D, the area painted on the frame is what moves on the sheet too: the wall beside and above the
        dog stays exactly as it was."""
        from PIL import ImageFilter
        from fx import sheet
        p = {**BOX, "rot_z": 8, "persp": 60, "amount": 100, **STEADY}
        upright = make()
        pic = scene(upright)
        area = Image.fromarray((dog_area(upright) * 255).astype(np.uint8))
        img = Image.fromarray((pic * 255 + 0.5).astype(np.uint8))
        on = sheet.Sheet("breath", p, 3, 300, 30, W, H, weight=area)
        off = sheet.Sheet("breath", p, 3, 300, 30, W, H)
        t = max(range(300), key=on.fx.level)
        a = np.asarray(on.draw(t, img.copy())).astype(int)
        b = np.asarray(off.draw(t, img.copy())).astype(int)
        near = np.asarray(area.filter(ImageFilter.MaxFilter(31))) > 0  # (the dog, and as far as its back rises)
        base = np.asarray(img).astype(int)
        self.assertEqual(int(np.abs(a - base)[~near].max()), 0)
        self.assertGreater(int(np.abs(b - base)[~near].max()), 8)  # (without an area the wall above is pressed)
        self.assertGreater(int((np.abs(a - base).max(axis=2) > 8).sum()), 50)
        self.assertEqual(int(np.abs(np.asarray(on.draw(0, img.copy())).astype(int) -
                                    np.asarray(on.draw(300, img.copy())).astype(int)).max()), 0)

    def test_off_the_frame_and_tiny(self):
        """A box partly off the frame, or tiny, does what it can and never fails."""
        for params in ({"x": 45, "y": -40}, {"x": -60}, {"scale": 1}, {"x": 100, "y": 100}):
            fx = make({"amount": 100, **params})
            pic = scene(make())
            for t in range(0, fx.P, 25):
                out = lens(fx, t, pic)
                self.assertTrue(np.isfinite(out).all(), params)

    def test_its_change_a_layer(self):
        """Turned, it says how much of each pixel it keeps (kept): its change is exactly a colour laid over the
        picture at 1 - kept (what the sheet warps onto the frame), everything kept where nothing changes; the picture
        it makes is the same as upright."""
        fx = make({"amount": 100})
        pic = scene(fx)
        for t in (max(range(fx.P), key=fx.level), 40):
            plain = lens(fx, t, pic)
            kept = np.ones((H, W), np.float32)
            out = lens(fx, t, pic, kept=kept)
            self.assertTrue(np.array_equal(plain, out), t)
            added = out - pic * kept[..., None]
            self.assertGreaterEqual(float(added.min()), -1e-5, t)
            self.assertLessEqual(float((added - (1 - kept)[..., None]).max()), 1e-5, t)
            self.assertTrue(np.all(kept[np.abs(out - pic).max(axis=2) == 0] == 1.0), t)

    def test_turned(self):
        """Turned in 3D (a dog lying on a slope): drawn upright on its sheet, its change laid over the frame, only in
        its quad; frame P is frame 0."""
        from PIL import Image, ImageDraw, ImageFilter
        from fx import sheet
        p = {**BOX, "scale": 60, "rot_z": 15, "rot_x": 20, "persp": 60, "amount": 100}
        self.assertTrue(effects.on_sheet("breath", p))
        img = Image.fromarray((scene(make()) * 255 + 0.5).astype(np.uint8))
        fx = sheet.Sheet("breath", p, 3, 300, 30, W, H)
        a = np.asarray(fx.draw(0, img.copy())).astype(int)
        b = np.asarray(fx.draw(300, img.copy())).astype(int)
        self.assertEqual(int(np.abs(a - b).max()), 0)
        quad = Image.new("L", (W, H), 0)
        ImageDraw.Draw(quad).polygon(effects.quad("breath", p, W, H), fill=255)
        near = np.asarray(quad.filter(ImageFilter.MaxFilter(5))) > 0
        peak = max(range(300), key=fx.fx.level)
        changed = np.abs(np.asarray(fx.draw(peak, img.copy())).astype(int) - np.asarray(img).astype(int)).max(axis=2)
        self.assertFalse(((changed > 0) & ~near).any())
        self.assertGreater(int((changed > 8).sum()), 50)

    def test_preview_from_the_picture_under_the_box(self):
        """The box alone (as the preview clip is made) on the picture under the box, resized to the clip, gives what
        the whole frame gives there: every size is relative to the box."""
        from PIL import Image
        whole = make({"amount": 100})
        pic = scene(whole)
        x0, y0, bw, bh = whole.box()
        cw, ch = round(bw * 2), round(bh * 2)
        alone = effects.make("breath", engine.fx_clip_params("breath", {**BOX, "amount": 100}), 3, 300, 30, cw, ch)

        def under(a):
            return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).resize(
                (cw, ch), Image.Resampling.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh)), np.float32) / 255
        for t in (max(range(300), key=whole.level), 40):
            got = under(pic)
            alone.apply(t, got, None, 1.0)
            want = under(lens(whole, t, pic))
            self.assertLess(float(np.abs(got - want).mean()), 0.01, t)


class Preview(unittest.TestCase):
    def test_name_follows_the_area(self):
        """Its preview clip is made with its area (what moves), so it is made again when the area is painted
        differently, its softness changes, or (passing behind the foreground) the foreground is painted; nothing
        painted and no area differ (nothing moves, everything moves)."""
        info = {"w": 480, "h": 270, "fps": "24", "frames": 144, "seconds": 6.0, "file": "input.mp4"}
        project = {"id": "br3athtest", "video": info, "length": 30, "fg": {"feather": 4}}
        layer = {"id": "dog001", "kind": "breath", "params": dict(BOX), "seed": 2}
        folder = os.path.join(paths.projects_dir(), project["id"])
        os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
        stamp = [0]

        def paint(name, *rects, lay=layer):
            m = Image.new("L", (480, 270), 0)
            for x, y, w, h in rects:
                m.paste(255, (x, y, x + w, y + h))
            path = engine.mask_path(folder, name)
            m.save(path)
            stamp[0] += 1
            os.utime(path, ns=(1, 10 ** 9 * stamp[0]))
            return engine.fx_clip_name(project, lay)

        everything = engine.fx_clip_name(project, layer)
        self.assertTrue(everything.endswith(".webm"))
        blank = paint("dog001")
        self.assertNotEqual(blank, everything)
        dog = paint("dog001", (150, 120, 180, 100))
        self.assertNotEqual(dog, blank)
        self.assertNotEqual(paint("dog001", (150, 120, 181, 100)), dog)
        self.assertEqual(paint("dog001", (150, 120, 180, 100)), dog)  # (undone: the clip made before)
        self.assertNotEqual(engine.fx_clip_name(project, dict(layer, feather=6)), dog)
        self.assertEqual(paint("fg", (10, 10, 30, 30)), dog)  # (not behind the foreground)
        behind = engine.fx_clip_name(project, dict(layer, behind=True))
        self.assertNotEqual(behind, dog)
        self.assertNotEqual(paint("fg", (10, 10, 31, 30), lay=dict(layer, behind=True)), behind)

    def test_preview_clip_shows_what_the_export_does(self):
        """Made for a real project with the dog painted: the clip laid on the video as the page lays it (no area over
        it) gives the exact frame; the wall above the back stays the video's own."""
        from fx import timeline
        work = os.path.join(helpers.ROOT, "breath-preview")
        os.makedirs(work, exist_ok=True)
        w, h = 320, 180
        fx = make(frames=144, fps=24, w=w, h=h)
        src = os.path.join(work, "still.mp4")
        still = (np.asarray(Image.fromarray((scene(make()) * 255 + 0.5).astype(np.uint8)).resize((w, h))))
        enc = video.Encoder(src, w, h, "24", crf=12, preset="veryfast")
        for _ in range(24):
            enc.write(still)
        enc.close()
        project = projects.create(src, "dog.mp4")
        folder = projects.folder(project["id"])
        total = timeline.project_frames(project)
        params = {**BOX, "amount": 100, **STEADY}
        project = projects.apply(project, {"tracks": project["tracks"], "layers": [
            {"id": "dog001", "kind": "breath", "params": params, "seed": 3,
             "track": project["tracks"][0]["id"], "start": 0, "end": total}]})
        mw, mh = projects.mask_size(project["video"])
        buf = io.BytesIO()
        Image.fromarray((dog_area(make(), mw, mh) * 255).astype(np.uint8)).save(buf, "PNG")
        project = projects.save_mask(project, "dog001", buf.getvalue())
        name = engine.fx_clip(folder, project, "dog001", processes=1)
        self.assertEqual(name, engine.fx_clip_name(project, project["layers"][0]))
        path = os.path.join(folder, "clips", name)
        info = video.probe(path, overlay=True)
        self.assertTrue(info["alpha"])
        x0, y0, bw, bh = effects.box_of("breath", params, w, h)
        cw, ch = info["w"], info["h"]
        base = video.read_frame(engine.input_path(folder, project), project["video"], 0).astype(np.float32)
        fx = make(params, total, 24, w, h)
        peak = max(range(0, min(total, 120)), key=fx.level)
        with video.Frames(path, info, alpha=True, count=peak + 1) as frames:
            for i, frame in enumerate(frames):
                clip = frame.astype(np.float32)
        a = clip[..., 3:] / 255
        laid = np.asarray(Image.fromarray(base.astype(np.uint8)).resize(
            (cw, ch), Image.Resampling.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh)), np.float32)
        live = laid * (1 - a) + clip[..., :3] * a
        exact = engine.Looks().frame(folder, project, peak, w).astype(np.float32)
        exact = np.asarray(Image.fromarray(np.clip(exact, 0, 255).astype(np.uint8)).resize(
            (cw, ch), Image.Resampling.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh)), np.float32)
        self.assertLess(float(np.abs(live - exact).mean()), 2.5)  # (levels of 255, the box)
        self.assertGreater(float(np.abs(exact - laid).max()), 30)  # (it moved)
        # the wall above the back: the video's own in the exact frame, see-through in the clip
        top = int(round(bh * (1 - fx.body[1]) - 2 * fx.rise * bh * 1.6)) - 2
        self.assertGreater(top, 4)
        self.assertLess(float(np.abs(exact[:top] - laid[:top]).max()), 3)
        self.assertLess(float(a[:top].max()), 0.05)


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_label(self):
        keys = [p["key"] for p in effects.SCHEMA["breath"]["params"]]
        self.assertEqual(len(keys), len(set(keys)))
        for p in effects.SCHEMA["breath"]["params"]:
            self.assertTrue(p.get("group"), p["key"])
            self.assertTrue(p["label"], p["key"])
        for k in ("look", "amount", "rate", "vary", "frame_w", "frame_h", "x", "y", "scale", "rot_z", "persp"):
            self.assertIn(k, keys)

    def test_looks(self):
        """Each look sets the breathing (not the body or the place); the default look is the defaults."""
        look = next(p for p in breath.SCHEMA["params"] if p["key"] == "look")
        self.assertEqual({o["value"] for o in look["options"]}, set(breath.LOOKS))
        defaults = effects.defaults("breath")
        for name, preset in look["presets"].items():
            self.assertEqual(set(preset), {"amount", "rate", "vary"}, name)
            self.assertEqual(effects.clean_params("breath", preset), {**defaults, **preset}, name)
        self.assertEqual(look["presets"][look["value"]], {k: defaults[k] for k in ("amount", "rate", "vary")})
        self.assertGreater(look["presets"]["small"]["rate"], defaults["rate"])  # (a small dog, a cat: quicker)
        self.assertLess(look["presets"]["deep"]["rate"], defaults["rate"])

    def test_the_body_in_its_box(self):
        """The body is an outline on the box's bottom edge (the page drags its sides and top: "frame"), always a
        rectangle; it never fills the box's height, so the rise always has room above it to fade out."""
        frame = breath.SCHEMA["frame"]
        self.assertEqual((frame["fixed"], frame["width"], frame["height"]), ("rect", "frame_w", "frame_h"))
        self.assertTrue(frame["label"] and frame["tip"] and frame["side_tip"] and frame["top_tip"])
        high = next(p for p in breath.SCHEMA["params"] if p["key"] == "frame_h")
        self.assertLessEqual(high["max"], 80)
        self.assertEqual(breath.Breath.BOX, (4.0, 3.0))
        fx = make({"frame_w": 60, "frame_h": 50})
        x0, y0, w, h = fx.box()
        self.assertEqual(body(fx), (x0 + 0.2 * w, y0 + 0.5 * h, x0 + 0.8 * w, y0 + h))

    def test_a_lens_made_from_the_picture_under_it(self):
        """A lens that turns, its preview clip made from the picture under its box (a transparent WebM laid as it
        is, named by the box's place)."""
        cls = breath.Breath
        self.assertTrue(cls.lens and cls.UNDER and cls.TURNS and cls.BOX and cls.MOVES)
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 144, "seconds": 6.0}
        a = engine.fx_clip_name({"video": info, "length": 30}, {"kind": "breath", "params": {}})
        b = engine.fx_clip_name({"video": info, "length": 30}, {"kind": "breath", "params": {"x": 5}})
        self.assertTrue(a.endswith(".webm"))
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
