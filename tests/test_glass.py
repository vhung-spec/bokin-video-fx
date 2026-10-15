"""Water on a window pane (fx/glass.py): a lens effect, so most checks go through apply()."""
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects, glass

W, H = 320, 180
_yy, _xx = np.mgrid[0:H, 0:W]
SCENE = np.dstack([0.2 + 0.5 * (((_xx // 16) + (_yy // 16)) % 2), 0.3 + 0.4 * _xx / W, 0.25 + 0.5 * _yy / H]).astype(
    np.float32)
MAX = {p["key"]: p["max"] for p in glass.SCHEMA["params"] if p["type"] == "range"}
MIN = {p["key"]: p["min"] for p in glass.SCHEMA["params"] if p["type"] == "range"}


def lens(fx, t, mask=None, gain=1.0, scene=SCENE):
    pic = scene.copy()
    fx.apply(t, pic, mask, gain)
    return pic


def over(fx, t, mask=None):
    canvas = effects.Canvas(W, H)
    fx.draw(t, canvas, mask)
    return canvas.rgb.copy(), canvas.a.copy()


class Loops(unittest.TestCase):
    CASES = [{}, MAX, MIN, {"refog": 20, "speed": 0, "drops": 100}, {"refog": 1, "speed": 100, "hits": 100},
             {"size": 100, "bead_size": 100, "wiggle": 100, "refraction": 100}] + [
        glass.SCHEMA["params"][0]["presets"][k] for k in glass.LOOKS]

    def test_frame_p_is_frame_0(self):
        """The exact frame and the preview both come back to frame 0 after one loop, whatever the settings."""
        mask = np.zeros((H, W), np.float32)
        mask[20:150, 30:300] = 1
        for frames, fps in ((144, 24), (150, 25), (90, 30), (720, 24), (1800, 30), (24, 24)):
            for params in self.CASES:
                fx = glass.Glass(effects.clean_params("glass", params), 3, frames, fps, W, H)
                for m in (None, mask):
                    diff = np.abs(lens(fx, 0, m) - lens(fx, frames, m)).max()
                    self.assertLess(diff, 1 / 512, f"{params} P={frames} fps={fps}")
                (r0, a0), (r1, a1) = over(fx, 0), over(fx, frames)
                self.assertLess(max(np.abs(r0 - r1).max(), np.abs(a0 - a1).max()), 1 / 512, f"draw {params}")

    def test_order_of_frames_does_not_matter(self):
        """A frame is the same whatever was drawn before it (the buffer is cleared of the last frame's drops)."""
        fx = effects.make("glass", {"drops": 100, "hits": 100}, 3, 144, 24, W, H)
        first = lens(fx, 50)
        for t in (51, 3, 140, 77):
            lens(fx, t)
        self.assertTrue(np.array_equal(first, lens(fx, 50)))
        fresh = effects.make("glass", {"drops": 100, "hits": 100}, 3, 144, 24, W, H)
        self.assertTrue(np.array_equal(first, lens(fresh, 50)))


class Area(unittest.TestCase):
    def test_nothing_changes_outside_the_area_or_at_gain_0(self):
        mask = np.zeros((H, W), np.float32)
        mask[40:120, 100:230] = 1
        mask[60:100, 150:180] = 0.5
        for params in ({}, MAX):
            fx = effects.make("glass", params, 3, 144, 24, W, H)
            for t in (0, 33, 90):
                pic = lens(fx, t, mask)
                self.assertTrue(np.array_equal(pic[mask == 0], SCENE[mask == 0]))
                self.assertFalse(np.array_equal(pic[mask == 1], SCENE[mask == 1]))
                self.assertTrue(np.array_equal(lens(fx, t, mask, 0.0), SCENE))
                canvas = effects.Canvas(W, H)
                fx.draw(t, canvas, mask)
                self.assertEqual(float(canvas.a[mask == 0].max()), 0.0)
        empty = np.zeros((H, W), np.float32)
        self.assertTrue(np.array_equal(lens(effects.make("glass", {}, 3, 144, 24, W, H), 5, empty), SCENE))
        self.assertTrue(np.array_equal(lens(effects.make("glass", {"opacity": 0}, 3, 144, 24, W, H), 5), SCENE))

    def test_gain_mixes_in_proportion(self):
        fx = effects.make("glass", {}, 3, 144, 24, W, H)
        full, half = lens(fx, 20), lens(fx, 20, gain=0.5)
        self.assertLess(np.abs(half - (SCENE + 0.5 * (full - SCENE))).max(), 1e-5)

    def test_tiny_and_odd_areas(self):
        for w, h in ((1, 1), (2, 1), (7, 5), (33, 17)):
            fx = effects.make("glass", MAX, 3, 48, 24, w, h)
            pic = np.full((h, w, 3), 0.4, np.float32)
            fx.apply(5, pic, None, 1.0)
            self.assertTrue(np.isfinite(pic).all())
            canvas = effects.Canvas(w, h)
            fx.draw(5, canvas, None)
        fx = effects.make("glass", MAX, 3, 48, 24, W, H)
        for box in ((50, 51, 60, 61), (0, 3, 0, W), (H - 2, H, W - 1, W)):
            mask = np.zeros((H, W), np.float32)
            mask[box[0]:box[1], box[2]:box[3]] = 1
            pic = lens(fx, 7, mask)
            self.assertTrue(np.array_equal(pic[mask == 0], SCENE[mask == 0]))


class Look(unittest.TestCase):
    def test_fog_blurs_and_whitens(self):
        plain = {"beads": 0, "drops": 0, "hits": 0}
        fogged = lens(effects.make("glass", {**plain, "haze": 100, "blur": 100}, 3, 144, 24, W, H), 0)
        clear = lens(effects.make("glass", {**plain, "haze": 0, "blur": 100}, 3, 144, 24, W, H), 0)
        self.assertTrue(np.array_equal(clear, SCENE))  # no fog, nothing out of focus: unchanged
        edges = lambda p: np.abs(np.diff(p, axis=1)).mean()  # noqa: E731
        self.assertLess(edges(fogged), 0.5 * edges(SCENE))
        self.assertGreater(fogged.mean(), SCENE[..., :].mean() - 0.02)
        # out of focus on clear glass: blurred, not lighter
        far = lens(effects.make("glass", {**plain, "haze": 0, "blur": 100, "defocus": 100}, 3, 144, 24, W, H), 0)
        self.assertLess(edges(far), 0.5 * edges(SCENE))
        self.assertLess(abs(float(far.mean() - SCENE.mean())), 0.02)

    def test_light_fog_blurs_instead_of_dimming(self):
        """Fogged glass spreads the light: no sharp copy of the scene is left under the blur (it used to be a
        crossfade, so a crisp edge stayed visible under a halo)."""
        scene = np.full((H, W, 3), 0.2, np.float32)
        scene[:, W // 2:] = 0.8
        fx = effects.make("glass", {"beads": 0, "drops": 0, "hits": 0, "haze": 35, "uneven": 0, "blur": 50},
                          3, 144, 24, W, H)
        row = lens(fx, 0, scene=scene)[H // 2, :, 1]
        jump = np.abs(np.diff(row))
        self.assertLess(float(jump.max()), 0.12)  # the step of 0.6 is spread out

    def test_a_drop_shows_the_scene_upside_down(self):
        """Dark above, light below: inside a big bead the top is lighter than the bottom (and the other way round
        when drops do not bend the light)."""
        w, h = W * 4, H * 4
        ramp = np.dstack([np.repeat((np.arange(h, dtype=np.float32) / h)[:, None], w, 1)] * 3)
        for refraction, flipped in ((50, True), (0, False)):
            fx = effects.make("glass", {"haze": 0, "blur": 0, "drops": 0, "hits": 0, "beads": 100, "bead_size": 100,
                                        "highlight": 0, "rim": 0, "refraction": refraction}, 3, 144, 24, w, h)
            pic = ramp.copy()
            fx.apply(0, pic, None, 1.0)
            b, seen = fx.bead, 0
            for k in np.argsort(-b["r"])[:20]:  # the biggest beads, away from the edges
                x, y, r = b["cx"][k], b["cy"][k], b["r"][k]
                if r < 4 or not (w * 0.1 < x < w * 0.9 and h * 0.3 < y < h * 0.7):
                    continue
                cols = slice(int(x - 0.3 * r), int(x + 0.3 * r) + 1)
                top = pic[int(y - 0.6 * r):int(y - 0.2 * r), cols].mean()
                bottom = pic[int(y + 0.2 * r) + 1:int(y + 0.6 * r) + 1, cols].mean()
                self.assertEqual(top > bottom, flipped, (refraction, x, y, r))
                seen += 1
            self.assertGreater(seen, 2)

    def test_drops_see_the_scene_not_the_fog(self):
        """Drops that barely bend the light (level 0 of the picture) show the scene behind, not the fogged glass
        around them."""
        params = {"blur": 0, "drops": 0, "hits": 0, "beads": 100, "bead_size": 100, "highlight": 0, "rim": 0,
                  "refraction": 8}
        fogged = effects.make("glass", {**params, "haze": 100}, 3, 144, 24, W, H)
        clear = effects.make("glass", {**params, "haze": 0}, 3, 144, 24, W, H)
        a, b = lens(fogged, 0), lens(clear, 0)
        fogged._fields(0)
        inside = fogged.buf[glass.A] > 0.99
        self.assertGreater(int(inside.sum()), 50)
        veil = fogged.veil * fogged.fog * (1 - fogged.buf[glass.CLR])
        # inside a drop only the thin veil laid over the whole glass differs
        expect = b + veil[..., None] * (fogged.colour - b)
        self.assertLess(float(np.abs(a - expect)[inside].mean()), 0.004)

    def test_running_drops_never_vanish_in_place(self):
        """A slot's cycles are chained: a drop that stops is carried on by the next cycle from the same spot, the
        same size, and a chain ends only by running off the bottom."""
        for params in ({}, {"speed": 0, "drops": 100}, {"speed": 100, "size": 100, "wiggle": 100}):
            fx = effects.make("glass", params, 3, 720, 24, W, H)
            for i in range(fx.r_n):
                c = int(fx.r_c[i])
                self.assertTrue(fx.r_new[i, 0] and fx.r_last[i, c - 1])
                for v in range(c):
                    end = fx.r_y0[i, v] + fx.r_L[i, v]
                    if fx.r_last[i, v]:
                        self.assertGreater(end - fx.r_R[i, v], H)  # off the bottom
                        continue
                    self.assertFalse(fx.r_new[i, v + 1])
                    self.assertAlmostEqual(fx.r_y0[i, v + 1], end, places=3)
                    x_end = fx._x(np.array([i]), np.array([v]), np.array([end]))[0]
                    self.assertAlmostEqual(fx.r_xk[i, v + 1, 0], x_end, places=3)
                # at every cycle boundary the drop is where and as big as it was a moment before
                for v in range(c):
                    if fx.r_last[i, (v - 1) % c]:
                        continue
                    tau = np.mod((v - fx.r_phi[i]) / c, 1.0)
                    before, after = fx._running(tau - 1e-7), fx._running(tau + 1e-7)
                    for k in range(4):
                        self.assertAlmostEqual(before[k][i], after[k][i], delta=0.05, msg=(params, i, v, k))

    def test_rain_hits_stay_then_dry_up_slowly(self):
        fx = effects.make("glass", {"hits": 60}, 3, 720, 24, W, H)
        self.assertGreater(fx.h_n, 10)
        life = (fx.h_end - fx.h_sa) * fx.h_T[:, None]
        self.assertGreater(float(life[:, 0].min()), 3.0)  # seconds on the glass
        self.assertGreaterEqual(float((fx.h_ev * fx.h_T[:, None])[:, 0].min()), 1.5 - 1e-9)  # drying up

    def test_buffer_only_as_big_as_the_area(self):
        fx = effects.make("glass", {}, 3, 144, 24, W, H)
        mask = np.zeros((H, W), np.float32)
        mask[40:120, 100:230] = 1
        lens(fx, 5, mask)
        self.assertEqual(fx.buf.shape, (8, 80, 130))
        self.assertTrue(np.array_equal(lens(fx, 5, mask), lens(effects.make("glass", {}, 3, 144, 24, W, H), 5, mask)))

    def test_drops_run_and_clear_the_fog(self):
        fx = effects.make("glass", {"drops": 100, "beads": 0, "hits": 0}, 3, 240, 24, W, H)
        cleared = []
        for t in range(0, 240, 20):
            fx._fields(t)
            cleared.append(float((fx.buf[glass.CLR] > 0.5).mean()))
        self.assertGreater(max(cleared), 0.005)  # trails
        self.assertGreater(np.abs(lens(fx, 0) - lens(fx, 12)).max(), 0.05)

    def test_moving_a_slider_keeps_the_beads(self):
        a = effects.make("glass", {}, 3, 144, 24, W, H)
        b = effects.make("glass", {"highlight": 90, "haze": 70, "speed": 80, "refraction": 10}, 3, 144, 24, W, H)
        self.assertTrue(np.array_equal(a.bead["cx"], b.bead["cx"]))
        more = effects.make("glass", {"beads": 80}, 3, 144, 24, W, H)
        self.assertGreater(len(more.bead["cx"]), len(a.bead["cx"]))
        self.assertGreater(np.isin(a.bead["cx"], more.bead["cx"]).mean(), 0.85)  # more beads: the old ones stay

    def test_same_scene_at_any_size(self):
        """Sizes are in pixels of a 1080-line frame: the preview and the video show the same glass."""
        small = effects.make("glass", {}, 3, 144, 24, 640, 360)
        big = effects.make("glass", {}, 3, 144, 24, 1920, 1080)
        self.assertAlmostEqual(len(small.bead["cx"]) / len(big.bead["cx"]), 1, delta=0.1)
        self.assertAlmostEqual(float(np.median(big.bead["r"] / big.px)), float(np.median(small.bead["r"] / small.px)),
                               delta=0.2)

    def test_looks_and_params(self):
        keys = {p["key"]: p for p in glass.SCHEMA["params"]}
        for look, preset in glass.SCHEMA["params"][0]["presets"].items():
            for k, v in preset.items():
                self.assertIn(k, keys, look)
                self.assertTrue(keys[k]["min"] <= v <= keys[k]["max"], (look, k))
        self.assertEqual(glass.SCHEMA["params"][0]["presets"]["rain"], glass.DEFAULTS)
        for k, v in glass.DEFAULTS.items():
            self.assertEqual(keys[k]["value"], v, k)
        self.assertTrue(glass.Glass.lens)


if __name__ == "__main__":
    unittest.main()
