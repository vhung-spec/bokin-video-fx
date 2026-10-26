"""The candle flame (fx/candle.py): a placed lens effect, so most checks go through apply()."""
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects  # (first: it loads every effect's module)
from fx import candle, engine

W, H = 480, 270
MAX = {p["key"]: p["max"] for p in candle.SCHEMA["params"] if p["type"] == "range" and p["key"] not in "xy"}
MIN = {p["key"]: p["min"] for p in candle.SCHEMA["params"] if p["type"] == "range" and p["key"] not in "xy"}
STILL = {"flicker": 0, "sway": 0, "flutter": 0}


def place(cx, base, height, w=W, h=H):
    """The settings that put the box's flame base at (cx, base) for a flame `height` pixels tall."""
    bw, bh = candle.Candle.BOX
    bh_px = height / candle.FLAME
    bw_px = bh_px * bw / bh
    return {"scale": bh_px / min(w / bw, h / bh) * 100, "x": (cx - bw_px / 2 - (w - bw_px) / 2) / w * 100,
            "y": (base - candle.BASE * bh_px - (h - bh_px) / 2) / h * 100}


CX, BASE, TALL = 300.0, 160.0, 24.0
AT = place(CX, BASE, TALL)


def scene(flame=True, wide=1.0):
    """A dim warm room, a candle, and (if `flame`) a plain old flame on it: an orange oval with a white core."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) + 0.5
    pic = np.dstack([0.25 + 0.1 * xx / W, 0.15 + 0.05 * yy / H, 0.1 + 0.0 * xx]).astype(np.float32)
    pic[(np.abs(xx - CX) < 14) & (yy > BASE + 2)] = (0.9, 0.86, 0.75)  # the wax
    pic[(np.abs(xx - CX) < 0.8) & (yy > BASE - 2) & (yy < BASE + 6)] = (0.1, 0.07, 0.05)  # the wick
    if flame:
        oval = ((xx - CX) / (4.5 * wide)) ** 2 + ((yy - (BASE - TALL / 2)) / (TALL / 2)) ** 2
        pic[oval < 1] = (1.0, 0.75, 0.35)
        pic[oval < 0.35] = (1.0, 0.96, 0.85)
    return pic


SCENE = scene()


def soft_flame(pic, cx, base, tall, half_w, lean=0.0, blob=None):
    """A soft, bright flame (light added with Screen) on `pic`: a teardrop `tall` pixels high whose tip leans
    `lean` pixels aside; and a loose soft blob (x, y, radius) if given."""
    yy, xx = np.mgrid[0:pic.shape[0], 0:pic.shape[1]].astype(np.float32) + 0.5
    u = (base - yy) / tall
    uc = np.clip(u, 0, 1)
    r = half_w * np.sqrt(uc / 0.32) * ((1 - uc) / 0.68) ** 1.3 + 0.3
    xc = cx + lean * uc ** 2
    light = np.clip(1.4 - np.abs(xx - xc) / r, 0, 1) * ((u > -0.02) & (u < 1))
    if blob is not None:
        bx, by, br = blob
        light = np.maximum(light, np.exp(-((xx - bx) ** 2 + (yy - by) ** 2) / (2 * br * br)) * 1.1)
    light = np.clip(light, 0, 1)[..., None] * np.array([1.0, 0.75, 0.4], np.float32)
    return (1 - (1 - pic) * (1 - light)).astype(np.float32)


def lens(fx, t, pic=SCENE, mask=None, gain=1.0):
    out = pic.copy()
    fx.apply(t, out, mask, gain)
    return out


def light(fx, t, w=None, h=None):
    canvas = effects.Canvas(w or fx.w, h or fx.h)
    fx.draw(t, canvas, None)
    return canvas.rgb.copy()


def make(params, frames=144, fps=24, w=W, h=H, seed=3):
    return effects.make("candle", {**AT, **params}, seed, frames, fps, w, h)


def box_mask(fx, w=W, h=H):
    x0, y0, bw, bh = fx.box()
    m = np.zeros((h, w), bool)
    m[max(0, int(np.floor(y0))):int(np.ceil(y0 + bh)), max(0, int(np.floor(x0))):int(np.ceil(x0 + bw))] = True
    return m


class Loops(unittest.TestCase):
    CASES = [{}, {"mode": "new"}, MAX, MIN, {**MAX, "mode": "new"}, {**MIN, "mode": "new"},
             {"flutter": 100, "speed": 100}, {"flutter": 100, "speed": 0, "mode": "new"}] + [
        dict(p, look=k) for k, p in next(p for p in candle.SCHEMA["params"] if p["key"] == "look")["presets"].items()]

    def test_frame_p_is_frame_0(self):
        """The exact frame and the preview come back to frame 0 after one loop, whatever the settings."""
        mask = np.zeros((H, W), np.float32)
        mask[60:200, 200:400] = 0.8
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24), (24, 24), (1800, 30)):
            for params in self.CASES:
                fx = make(params, frames, fps)
                for m in (None, mask):
                    diff = np.abs(lens(fx, 0, mask=m) - lens(fx, frames, mask=m)).max()
                    self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")
                self.assertLess(np.abs(light(fx, 0) - light(fx, frames)).max(), 1 / 512, f"draw {params}")

    def test_motion_is_periodic(self):
        """Every signal of the motion, flutters included, is the same one loop later."""
        for frames, fps in ((144, 24), (720, 24), (37, 30)):
            fx = make({"flutter": 100, "flicker": 100, "sway": 100}, frames, fps)
            self.assertGreaterEqual(len(fx.f_at), 0 if frames < 144 else 1)
            for t in (0, 5, frames // 3):
                np.testing.assert_allclose(fx._motion(t), fx._motion(t + frames), atol=1e-9)

    def test_moves(self):
        for mode in ("alive", "new"):
            fx = make({"mode": mode})
            self.assertGreater(np.abs(lens(fx, 0) - lens(fx, 30)).max(), 0.05, mode)

    def test_no_motion_faster_than_a_quarter_of_the_frame_rate(self):
        """Faster than that, the frames show a jitter, not a flame moving: every rate (waves, flutters, the
        ripple up the flame) stays at or under fps / 4, even at the fastest pace."""
        for fps, frames in ((24, 144), (24, 720), (30, 900), (25, 750)):
            fx = make({"speed": 100, "flutter": 100}, frames, fps)
            rates = np.concatenate([w.m for w in fx.waves] + [fx.f_m, [fx.m_rip]]) / fx.loop_s
            self.assertTrue(len(fx.f_m))
            self.assertLessEqual(float(rates.max()), fps / 4 + 1e-9, (fps, frames))
            # the fast waves still differ from each other (eased below the limit, not cut to it)
            self.assertGreater(len(set(fx.waves[1].m)), 1)

    def test_order_of_frames_does_not_matter(self):
        """A frame is the same whatever was made before it, and the same for the same seed."""
        for mode in ("alive", "new"):
            fx = make({"mode": mode, "flutter": 100})
            first = lens(fx, 50)
            for t in (51, 3, 140, 77):
                lens(fx, t)
            self.assertTrue(np.array_equal(first, lens(fx, 50)))
            self.assertTrue(np.array_equal(first, lens(make({"mode": mode, "flutter": 100}), 50)))
            self.assertFalse(np.array_equal(first, lens(make({"mode": mode, "flutter": 100}, seed=4), 50)))

    def test_no_motion_when_calm(self):
        """With no flicker, sway or flutter the flame stands still (only its box's light stays on it)."""
        for mode in ("alive", "new"):
            fx = make({"mode": mode, **STILL})
            self.assertTrue(np.array_equal(lens(fx, 0), lens(fx, 77)), mode)

    def test_alive_and_calm_changes_nothing(self):
        """Bringing a flame to life with no motion, no light thrown, no halo and its size and brightness as they
        are leaves the picture exactly as it was."""
        for pic in (SCENE, soft_flame(scene(False), CX, BASE, TALL, 5)):
            fx = make({"mode": "alive", **STILL, "light": 0, "halo": 0, "size": 50, "brightness": 50, "warmth": 50})
            for t in (0, 33, 101):
                self.assertTrue(np.array_equal(lens(fx, t, pic), pic))


class Place(unittest.TestCase):
    def test_box_fits_the_flame(self):
        """The box puts the flame's base at its bottom middle: the settings above put it where they say."""
        fx = make({})
        x0, y0, w, h, cx, ay, hf = fx._geom()
        for got, want in ((cx, CX), (ay, BASE), (hf, TALL)):  # (the page keeps x, y, scale to 0.05, 0.1)
            self.assertLess(abs(got - want), 0.2)
        self.assertAlmostEqual(w / h, candle.Candle.BOX[0] / candle.Candle.BOX[1])

    def test_table_scene_flames(self):
        """The two flames of the 1920 x 1080 test scene (bases (1220, 458) and (1390, 550), tips at 397 and
        501) take these settings (as the integrator's notes say)."""
        for (cx, base, tip), want in (((1220, 458, 397), (13.54, -11.55, 14.1)),
                                      ((1390, 550, 501), (22.4, -2.25, 11.3))):
            got = place(cx, base, base - tip, 1920, 1080)
            self.assertEqual((round(got["x"], 2), round(got["y"], 2), round(got["scale"], 1)), want)

    def test_changes_only_its_box_without_light(self):
        """With no light thrown round it, everything it does stays inside its box (both ways); with light, the
        flicker reaches further, as far as the light reaches."""
        for mode in ("alive", "new"):
            for params in ({"light": 0}, {"light": 0, **MAX}, {"light": 0, "flutter": 100, "sway": 100}):
                fx = make({"mode": mode, **params})
                inside = box_mask(fx)
                for t in (0, 13, 29, 101):
                    changed = np.abs(lens(fx, t) - SCENE).max(axis=2) > 0
                    self.assertFalse((changed & ~inside).any(), (mode, params, t))
                    self.assertTrue(changed.any())
            fx = make({"mode": mode, "light": 100, "reach": 100})
            changed = np.abs(lens(fx, 3) - SCENE).max(axis=2) > 1e-4
            self.assertTrue((changed & ~box_mask(fx)).any(), mode)

    def test_draw_stays_in_its_box(self):
        for mode in ("alive", "new"):
            for params in ({}, MAX, {"size": 100, "sway": 100, "flutter": 100, "halo": 100}):
                fx = make({"mode": mode, **params})
                for t in (0, 40, 77):
                    lit = light(fx, t).max(axis=2) > 0
                    self.assertFalse((lit & ~box_mask(fx)).any(), (mode, params))
                    self.assertTrue(lit.any())

    def test_nothing_cut_at_the_box_edges(self):
        """The flame and its glow fade out well inside the box, even at the most: the preview clip (the box
        alone) shows no cut edge."""
        for params in ({"mode": "new", **MAX}, {"mode": "alive", **MAX},
                       {"mode": "new", "size": 100, "sway": 100, "flutter": 100, "flicker": 100, "speed": 100}):
            fx = effects.make("candle", {**params, "x": 0, "y": 0, "scale": 100}, 2, 720, 24, 120, 160)
            for t in range(0, 720, 9):
                rgb = light(fx, t)
                edges = (rgb[:2], rgb[-2:], rgb[:, :2], rgb[:, -2:])
                self.assertLess(max(float(e.max()) for e in edges), 1 / 255, (params, t))

    def test_preview_clip_is_the_box_alone(self):
        """The preview clip (the box alone, x = y = 0, scale 100, on a canvas of the box's shape) is the same
        picture as the box in the whole frame, at another size."""
        from PIL import Image
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 144}
        cw, ch = engine.fx_clip_frame(info, "candle")
        self.assertAlmostEqual(cw / ch, candle.Candle.BOX[0] / candle.Candle.BOX[1], places=2)
        for params in ({}, {"mode": "new"}, {"mode": "new", "look": "breeze", "flutter": 70, "sway": 65},
                       {"mode": "new", "size": 100, "halo": 100, "warmth": 0}):
            full_params = {**params, "x": 13.54, "y": -11.55, "scale": 14.1}
            alone = effects.make("candle", engine.fx_clip_params("candle", full_params), 1, 144, 24, cw, ch)
            self.assertEqual(tuple(round(v) for v in alone.box()), (0, 0, cw, ch))
            whole = effects.make("candle", full_params, 1, 144, 24, 1920, 1080)
            for t in (0, 50):
                clip = np.minimum(light(alone, t), 1)
                x0, y0, bw, bh = whole.box()
                big = np.minimum(light(whole, t), 1)
                # the frame's box at the clip's size (its pixels at the clip's pixel centres)
                part = Image.fromarray((big * 255 + 0.5).astype(np.uint8)).resize(
                    (cw, ch), Image.Resampling.BILINEAR, box=(x0, y0, x0 + bw, y0 + bh))
                part = np.asarray(part, np.float32) / 255
                self.assertLess(float(np.abs(part - clip).mean()), 0.004, (params, t))
                self.assertLess(float(np.abs(part - clip).max()), 0.2, (params, t))
                small = np.asarray(Image.fromarray((clip * 255 + 0.5).astype(np.uint8)).resize(
                    (round(bw), round(bh)), Image.Resampling.BOX), np.float32) / 255
                self.assertGreater(small.max(), 0.5 if params.get("mode") == "new" else 0.03)

    def test_preview_is_the_exact_frame_on_black(self):
        """On a black picture the new flame's exact frame is what the preview lays over it (Screen)."""
        black = np.zeros((H, W, 3), np.float32)
        for params in ({"mode": "new"}, {"mode": "new", **MAX}, {"mode": "alive", "halo": 100}):
            fx = make(params)
            for t in (0, 33):
                pic = black.copy()
                canvas = effects.Canvas(W, H)
                fx.draw(t, canvas, None)
                canvas.flush(pic, fx.blend)
                self.assertLess(float(np.abs(lens(fx, t, black) - pic).max()), 1e-5, params)

    def test_off_the_frame_and_tiny(self):
        """A box partly or wholly off the picture, or a tiny frame: no error, nothing outside the picture."""
        for params in ({"x": 100, "y": 0}, {"x": -100, "y": -100}, {"x": 49, "y": 47, "scale": 300},
                       {"scale": 1}, {"scale": 300}):
            for mode in ("alive", "new"):
                fx = make({"mode": mode, **params, "light": 100, "reach": 100})
                out = lens(fx, 5)
                self.assertTrue(np.isfinite(out).all(), params)
                light(fx, 5)
        for w, h in ((8, 6), (1, 1), (33, 2)):
            for mode in ("alive", "new"):
                fx = effects.make("candle", {"mode": mode}, 1, 48, 24, w, h)
                pic = np.full((h, w, 3), 0.3, np.float32)
                fx.apply(3, pic, None, 1.0)
                self.assertTrue(np.isfinite(pic).all())
                light(fx, 3)
        fx = make({"x": -100, "y": -100, "scale": 1, "light": 0})
        self.assertTrue(np.array_equal(lens(fx, 5), SCENE))


class Picture(unittest.TestCase):
    def test_mask_and_gain(self):
        """Outside its area (mask 0) or at gain 0 the picture stays exactly as it was; elsewhere it is
        picture + gain * mask * (full - picture)."""
        mask = np.zeros((H, W), np.float32)
        mask[:, :int(CX)] = 1  # half the flame
        mask[:, int(CX) + 3:] = 0.5
        for mode in ("alive", "new"):
            fx = make({"mode": mode, "light": 100})
            full = lens(fx, 9)
            out = lens(fx, 9, mask=mask, gain=0.6)
            zero = mask == 0
            self.assertTrue(zero.any())
            self.assertTrue(np.array_equal(out[zero], SCENE[zero]))
            want = SCENE + 0.6 * mask[..., None] * (full - SCENE)
            self.assertLess(float(np.abs(out - want).max()), 1e-5)
            self.assertTrue(np.array_equal(lens(fx, 9, gain=0.0), SCENE))
            self.assertLess(float(np.abs(lens(fx, 9, gain=0.5) - (SCENE + 0.5 * (full - SCENE))).max()), 1e-5)

    def test_alive_moves_the_old_flame_from_its_base(self):
        """The old flame sways and stretches, held at its base: what is below the base never moves."""
        fx = make({"mode": "alive", "light": 0, "halo": 0, "flicker": 100, "sway": 100, "flutter": 100})
        frames = [lens(fx, t) for t in range(0, 144, 6)]
        below = int(BASE) + 1
        for f in frames:
            self.assertTrue(np.array_equal(f[below:], SCENE[below:]))
        flame = frames[0][int(BASE - TALL):int(BASE)]
        moved = max(float(np.abs(f[int(BASE - TALL):int(BASE)] - flame).max()) for f in frames)
        self.assertGreater(moved, 0.3)
        # its tip goes up and down (the flame stretches), its middle left and right (it sways)
        tips = [np.flatnonzero((f[:, int(CX) - 2:int(CX) + 3] @ candle.LUMA).max(axis=1) > 0.6)[0] for f in frames]
        self.assertGreater(max(tips) - min(tips), 0)
        mids = [float(((f[int(BASE - 0.8 * TALL), :] @ candle.LUMA) * np.arange(W)).sum()
                      / (f[int(BASE - 0.8 * TALL), :] @ candle.LUMA).sum()) for f in frames]
        self.assertGreater(max(mids) - min(mids), 0.01)

    def test_alive_moves_only_the_flame(self):
        """Only the flame's own light moves: a dark line just beside it (a frame, a jar's rim) and the room round
        it stay exactly as they were, frame after frame, even in a breeze."""
        pic = soft_flame(scene(False), CX, BASE, TALL, 2.5)  # (its light reaches 0.16 flame heights aside)
        x = int(CX + 0.3 * TALL)
        pic[int(BASE - 1.9 * TALL):int(BASE) - 1, x - 1:x + 2] = (0.08, 0.06, 0.05)
        pic[int(BASE - 1.9 * TALL):int(BASE) - 1, int(CX - 0.6 * TALL)] = (0.6, 0.45, 0.3)  # (a bright line)
        line = np.zeros((H, W), bool)
        line[int(BASE - 1.9 * TALL):int(BASE) - 1, x - 1:x + 2] = True
        for params in ({}, dict(candle.LOOKS["breeze"])):
            fx = make({"mode": "alive", **params, "light": 0, "halo": 0})
            moved = 0.0
            for t in range(144):
                out = np.round(lens(fx, t, pic) * 255)  # (the same in the video: 8 bits)
                self.assertTrue(np.array_equal(out[line], np.round(pic[line] * 255)), (params, t))
                self.assertTrue(np.array_equal(out[:, int(CX - 0.6 * TALL)],
                                               np.round(pic[:, int(CX - 0.6 * TALL)] * 255)))
                out /= 255
                moved = max(moved, float(np.abs(out - pic).max()))
            self.assertGreater(moved, 0.2, params)  # (and the flame did move)

    def test_new_flame_hides_broken_flames(self):
        """The broken flames a made-up video shows (a blob come loose 1.3 flame heights up, a tip bent 0.6 flame
        heights aside, a flame 1.6 flame heights tall) are gone: with the new flame drawn, the picture is the
        room with the new flame (within 0.1) everywhere but on the new flame itself."""
        room = scene(False)
        cases = {"blob": soft_flame(room, CX, BASE, TALL, 5, blob=(CX + 2.4, BASE - 1.3 * TALL, 2.4)),
                 "bent": soft_flame(room, CX, BASE, TALL * 1.05, 4.5, lean=0.6 * TALL),
                 "tall": soft_flame(room, CX, BASE, 1.6 * TALL, 5),
                 "wide": soft_flame(room, CX, BASE, 0.85 * TALL, 8)}
        params = {"mode": "new", **STILL, "light": 0}
        want = lens(make(params), 0, room)
        black = np.zeros_like(room)
        flame = lens(make({**params, "halo": 0}), 0, black).max(axis=2) > 0.04
        for name, pic in cases.items():
            self.assertGreater(float(np.abs(pic - room)[~flame].max()), 0.5, name)  # (broken outside the new one)
            got = lens(make(params), 0, pic)
            self.assertLess(float(np.abs(got - want)[~flame].max()), 0.1, name)
            kept = lens(make({**params, "cover": 0}), 0, pic)
            self.assertGreater(float(np.abs(kept - want)[~flame].max()), 0.4, name)

    def test_new_flame_keeps_what_is_darker(self):
        """A flame only adds light, so hiding it never takes away what is darker than the room round it (the
        wick, a dark frame or line behind)."""
        pic = soft_flame(scene(False), CX, BASE, TALL, 5)
        x = int(CX + 0.25 * TALL)
        pic[int(BASE - 1.5 * TALL):int(BASE) - 1, x:x + 2] = (0.05, 0.04, 0.03)
        out = lens(make({"mode": "new", **STILL, "light": 0, "halo": 0, "size": 0}), 0, pic)
        self.assertLess(float(np.abs(out[int(BASE - 1.5 * TALL):int(BASE) - 1, x:x + 2] - 0.04).max()), 0.03)

    def test_preview_from_the_picture_under_the_box(self):
        """The box alone (as the preview clip is made) on the picture under the box, resized to the clip, gives
        what the whole frame gives there: the integrator can make the live preview from it."""
        from PIL import Image
        pic = soft_flame(scene(False), CX, BASE, TALL, 5)
        for mode in ("alive", "new"):
            params = {"mode": mode, "light": 0, **AT}
            whole = make(params)
            x0, y0, bw, bh = whole.box()
            cw, ch = round(bw * 4), round(bh * 4)
            alone = effects.make("candle", engine.fx_clip_params("candle", params), 3, 144, 24, cw, ch)

            def under(a):
                return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).resize(
                    (cw, ch), Image.Resampling.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh)), np.float32) / 255
            for t in (0, 50):
                got = under(pic)
                alone.apply(t, got, None, 1.0)
                want = under(lens(whole, t, pic))
                self.assertLess(float(np.abs(got - want).mean()), 0.02, (mode, t))

    def test_preview_keeps_only_its_change(self):
        """The preview clip (engine.change_rgba) laid with Normal blend on the picture under the box gives the
        effect's picture; where nothing changes it is clear and where little changes (the halo's faint glow) it is
        nearly so, so the clips of lower tracks show there."""
        from PIL import Image, ImageFilter
        pic = soft_flame(scene(False), CX, BASE, TALL, 5)
        for halo in (40, 0):
            params = {"mode": "alive", "light": 0, "halo": halo, **AT}
            x0, y0, bw, bh = make(params).box()
            cw, ch = round(bw * 4), round(bh * 4)
            alone = effects.make("candle", engine.fx_clip_params("candle", params), 3, 144, 24, cw, ch)
            before = np.asarray(Image.fromarray((pic * 255 + 0.5).astype(np.uint8)).resize(
                (cw, ch), Image.Resampling.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh)), np.float32) / 255
            after = before.copy()
            alone.apply(50, after, None, 1.0)
            rgba = engine.change_rgba(before, after, np.ones((ch, cw, 1), np.float32)).astype(np.float32) / 255
            a = rgba[..., 3:]
            laid = rgba[..., :3] * a + before * (1 - a)
            self.assertLess(float(np.abs(laid - np.clip(after, 0, 1)).max()), 1.01 / 255, halo)
            self.assertGreater(float(a.max()), 0.9, halo)  # the flame
            self.assertLess(float((a > 0.1).mean()), 0.2, halo)  # most of the box: nearly clear
            if halo == 0:
                changed = np.abs(after - before).max(axis=2) > 0
                clear = ~np.asarray(Image.fromarray(changed.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(15)),
                                    bool)
                self.assertGreater(clear.mean(), 0.5)
                self.assertEqual(float(a[..., 0][clear].max()), 0.0)  # far from any change: clear

    def test_alive_keeps_the_white_core_white(self):
        """When the light dims, what the camera saw white stays white (it does not turn grey)."""
        fx = make({"mode": "alive", "flicker": 100, "light": 100, "brightness": 0})
        core = (SCENE.min(axis=2) > 0.84)
        core[:int(BASE - 0.5 * TALL)] = False  # (its top half: the flame shrinks below it now and then)
        lows = []
        for t in range(0, 144, 3):
            out = lens(fx, t)
            lows.append(float(out[core].max(axis=1).min()))
        self.assertGreater(min(lows), 0.97)

    def test_new_flame_covers_the_old_one(self):
        """Drawing a new flame hides the old one, even a wider one: where the old flame was and the new one is
        not, the room shows again."""
        wide = scene(wide=2.2)
        empty = scene(flame=False)
        spot = (int(BASE - TALL * 0.45), int(CX + 8.5))  # inside the wide old flame, beside the new one
        self.assertGreater(wide[spot].min(), 0.3)
        kept = lens(make({"mode": "new", "cover": 0, "halo": 0, "light": 0}), 0, wide)
        covered = lens(make({"mode": "new", "cover": 100, "halo": 0, "light": 0}), 0, wide)
        self.assertGreater(float(np.abs(kept[spot] - empty[spot]).max()), 0.3)
        self.assertLess(float(np.abs(covered[spot] - empty[spot]).max()), 0.12)

    def test_new_flame_looks_like_a_flame(self):
        """A white-yellow core, warmer edges and tip, a dim bluish base, a soft halo round it."""
        fx = make({"mode": "new", "flicker": 0, "sway": 0, "flutter": 0}, w=960, h=540)
        x0, y0, w, h, cx, ay, hf = fx._geom()
        black = np.zeros((540, 960, 3), np.float32)
        out = lens(fx, 0, black)
        c = int(cx)
        core = out[int(ay - 0.38 * hf), c]
        tip = out[int(ay - 0.9 * hf), c]
        self.assertGreater(core.min(), 0.85)  # near white
        self.assertLess(tip[2] / tip[0], 0.5)  # orange
        base_side = out[int(ay - 0.04 * hf), int(cx + 0.06 * hf)]
        self.assertGreater(base_side[2], base_side[1] * 0.6)  # bluish, not yellow
        self.assertLess(base_side.max(), 0.4)  # and dim
        halo = out[int(ay - 0.5 * hf), int(cx + 0.6 * hf)]
        self.assertGreater(halo[0], 0.01)
        self.assertGreater(halo[0], halo[2])

    def test_light_flickers_on_the_room(self):
        """The light round the flame goes up and down in time with it, more near it, nothing far away."""
        fx = make({"mode": "alive", "light": 100, "flicker": 100, "halo": 0})
        near = (int(BASE + 20), int(CX + 30))
        far = (5, 5)
        values = np.array([lens(fx, t)[near] for t in range(0, 144, 4)])
        self.assertGreater(float(values[:, 0].max() - values[:, 0].min()), 0.02)
        self.assertTrue(all(np.array_equal(lens(fx, t)[far], SCENE[far]) for t in (0, 50)))
        fx0 = make({"mode": "alive", "light": 0, "halo": 0})
        self.assertTrue(np.array_equal(lens(fx0, 7)[near], SCENE[near]))


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_label(self):
        for p in candle.SCHEMA["params"]:
            self.assertTrue(p.get("group"), p["key"])
            self.assertTrue(p["label"], p["key"])
        keys = [p["key"] for p in candle.SCHEMA["params"]]
        self.assertEqual(len(keys), len(set(keys)))
        for k in ("look", "mode", "x", "y", "scale"):
            self.assertIn(k, keys)

    def test_looks(self):
        """Each look sets every setting but the way (mode), the cover and the place; the default look is the
        defaults."""
        look = next(p for p in candle.SCHEMA["params"] if p["key"] == "look")
        self.assertEqual({o["value"] for o in look["options"]}, set(candle.LOOKS))
        keys = {p["key"] for p in candle.SCHEMA["params"]} - {"look", "mode", "cover", "x", "y", "scale"}
        defaults = effects.defaults("candle")
        for name, preset in look["presets"].items():
            self.assertEqual(set(preset), keys, name)
            self.assertEqual(effects.clean_params("candle", preset), {**defaults, **preset}, name)
        self.assertEqual(look["presets"][look["value"]], {k: v for k, v in defaults.items() if k in keys})
        self.assertEqual(defaults["mode"], "alive")

    def test_anchor(self):
        """Where the flame's base and tip go in the box (marks on the page's box): its bottom middle, and the top of
        the flame the effect takes the video's flame to be (at size 50), with room above it for the halo."""
        self.assertEqual(candle.SCHEMA["anchor"], [0.5, candle.BASE])
        self.assertEqual(candle.SCHEMA["tip"], [0.5, candle.BASE - candle.FLAME])
        self.assertTrue(candle.SCHEMA["anchor_hint"] and candle.SCHEMA["tip_hint"])
        self.assertGreater(candle.SCHEMA["tip"][1], 0.25)  # room for the halo
        fx = make({})
        x0, y0, w, h = fx.box()
        g = fx._geom()
        self.assertAlmostEqual(g[4], x0 + 0.5 * w)
        self.assertAlmostEqual(g[5], y0 + candle.BASE * h)
        self.assertAlmostEqual(g[5] - g[6] * fx.size_k, y0 + candle.SCHEMA["tip"][1] * h)

    def test_screen_blend_and_lens(self):
        """A lens whose preview clip is made from the picture under its box (a transparent WebM laid as it is, its
        name changing with the box's place)."""
        self.assertEqual(candle.Candle.blend, "screen")
        self.assertTrue(candle.Candle.lens and candle.Candle.UNDER)
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 144, "seconds": 6.0}
        a = engine.fx_clip_name({"video": info, "length": 30}, {"kind": "candle", "params": {}})
        b = engine.fx_clip_name({"video": info, "length": 30}, {"kind": "candle", "params": {"x": 5}})
        self.assertTrue(a.endswith(".webm"))
        self.assertNotEqual(a, b)
        self.assertEqual(engine.fx_clip_frame(info, "candle", {"kind": "candle", "params": {"scale": 13}}),
                         (70, 94))  # no bigger than the box is in the copy of the video the page plays (1280 wide)

    def test_preview_follows_the_still_area_under_its_box(self):
        """The preview clip is made again when the still area under its box changes (its mask, softness, picture),
        not for a still area elsewhere in the frame."""
        import os
        from PIL import Image
        from fx import paths
        info = {"w": 480, "h": 270, "fps": "24", "frames": 144, "seconds": 6.0, "file": "input.mp4"}
        project = {"id": "c4ndl3t35t", "video": info, "length": 30, "freeze": {"feather": 10},
                   "still": {"source": "frame", "frame": 0}}
        layer = {"kind": "candle", "params": dict(AT)}
        masks = os.path.join(paths.projects_dir(), project["id"], "masks")
        os.makedirs(masks, exist_ok=True)
        x0, y0, bw, bh = effects.box_of("candle", AT, W, H)
        bare = engine.fx_clip_name(project, layer)

        stamp = [0]

        def paint(*rects):
            m = Image.new("L", (W, H), 0)
            for x, y, w, h in rects:
                m.paste(255, (int(x), int(y), int(x + w), int(y + h)))
            m.save(os.path.join(masks, "freeze.png"))
            stamp[0] += 1
            os.utime(os.path.join(masks, "freeze.png"), ns=(1, 10 ** 9 * stamp[0]))  # (a new stamp each time)
            return engine.fx_clip_name(project, layer)
        far = paint((5, 5, 30, 30))  # far from the box (top left; the box is right of the middle)
        self.assertGreater(x0, 120)
        self.assertEqual(far, bare)
        self.assertEqual(engine.fx_clip_name(dict(project, still={"source": "frame", "frame": 9}), layer), bare)
        self.assertEqual(engine.fx_clip_name(dict(project, freeze={"feather": 20}), layer), bare)
        over = paint((x0 + 2, y0 + 2, 8, 8))
        self.assertNotEqual(over, bare)
        # more still area elsewhere, or that taken away again: the clip stays
        self.assertEqual(paint((x0 + 2, y0 + 2, 8, 8), (5, 5, 30, 30)), over)
        self.assertEqual(paint((x0 + 2, y0 + 2, 8, 8)), over)
        self.assertNotEqual(paint((x0 + 2, y0 + 2, 9, 8)), over)  # changed under the box: made again
        self.assertEqual(paint((x0 + 2, y0 + 2, 8, 8)), over)  # (undone: the clip made before)
        self.assertNotEqual(engine.fx_clip_name(dict(project, still={"source": "frame", "frame": 9}), layer), over)
        self.assertNotEqual(engine.fx_clip_name(dict(project, freeze={"feather": 20}), layer), over)
        near = paint((x0 - 30, y0 + 10, 20, 20))  # 10 px left of the box: its soft edge (10 px) reaches it
        self.assertNotEqual(near, bare)
        moved = engine.fx_clip_name(project, {"kind": "candle", "params": dict(AT, x=AT["x"] + 30)})
        self.assertNotEqual(moved, near)


if __name__ == "__main__":
    unittest.main()
