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
             {"refog": 20, "speed": 100, "drops": 100},  # (trails of many earlier cycles still fogging over)
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


class Release(unittest.TestCase):
    """What is as big as the frame is freed while a clip is not showing (the renderer calls release()) and made
    again for the next frame drawn: a clip duplicated or cut many times holds about one clip's worth at a time."""

    def test_frames_are_the_same_after_release(self):
        mask = np.zeros((H, W), np.float32)
        mask[30:160, 40:150] = 1
        mask[30:160, 170:290] = 0.7
        for params in ({}, {"look": "storm", "refog": 20}, MAX):
            for m in (None, mask):
                kept = effects.make("glass", params, 3, 720, 24, W, H)
                freed = effects.make("glass", params, 3, 720, 24, W, H)
                for k, t in enumerate((5, 6, 300, 301, 302, 719, 0)):  # (it goes off and on again between frames)
                    if k % 2 == 0:
                        freed.release()
                        for name in ("bbox", "buf", "flat", "dirty", "beads_all", "b_idx", "b_id", "b_f", "fog",
                                     "mottle"):
                            self.assertIsNone(getattr(freed, name), name)
                        freed.release()  # (again: nothing to do)
                    self.assertTrue(np.array_equal(lens(kept, t, m, 0.9), lens(freed, t, m, 0.9)), (params, t))
                    if k % 3 == 0:
                        freed.release()
                    a, b = effects.Canvas(W, H), effects.Canvas(W, H)
                    kept.draw(t, a, m)
                    freed.draw(t, b, m)
                    self.assertTrue(np.array_equal(a.rgb, b.rgb) and np.array_equal(a.a, b.a), ("draw", params, t))

    def test_nothing_that_grows_with_the_frame_is_kept(self):
        """Made, and released after a frame: no array left of even a quarter of a frame plane (what the layout
        needs is made small), even with the most and biggest beads."""
        w, h = 1280, 720
        for params in ({}, {"beads": 100, "bead_size": 100}):
            fx = effects.make("glass", params, 3, 720, 24, w, h)

            def biggest():
                sizes = [0]
                for v in vars(fx).values():
                    for a in (v.values() if isinstance(v, dict) else v if isinstance(v, (tuple, list)) else [v]):
                        if isinstance(a, np.ndarray):
                            sizes.append(a.size)
                return max(sizes)

            self.assertLess(biggest(), w * h // 4)  # (nothing that big made before a frame is drawn)
            fx.apply(3, np.full((h, w, 3), 0.5, np.float32), None, 1.0)
            self.assertGreaterEqual(biggest(), 8 * w * h)  # the drop buffer
            fx.release()
            self.assertLess(biggest(), w * h // 4)

    def test_duplicated_clips_hold_one_clips_worth(self):
        """Six copies of a clip one after the other, one frame drawn in each, the others released: the memory used
        at most is about that of one copy drawing a frame, not six times as much."""
        import tracemalloc
        w, h = 640, 360
        scene = np.full((h, w, 3), 0.5, np.float32)
        clips = [effects.make("glass", {}, 3, 720, 24, w, h) for _ in range(6)]
        tracemalloc.start()
        try:
            base = tracemalloc.get_traced_memory()[0]
            clips[0].apply(60, scene.copy(), None, 1.0)
            one = tracemalloc.get_traced_memory()[1] - base
            clips[0].release()
            tracemalloc.reset_peak()
            for k, fx in enumerate(clips):
                for other in clips:
                    if other is not fx:
                        other.release()
                fx.apply(120 * k + 60, scene.copy(), None, 1.0)
            six = tracemalloc.get_traced_memory()[1] - base
        finally:
            tracemalloc.stop()
        self.assertGreater(one, 8 * w * h * 4)  # (the drop buffer alone)
        self.assertLess(six, 1.25 * one)


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

    def test_slow_refog_keeps_the_running_drops(self):
        """How slowly trails fog over does not change the running drops: as many runs, the same paths and times
        (it used to make each slot go round less often: 20 s cut the runs per 30 s to a third)."""
        ref = effects.make("glass", {"refog": 1}, 3, 900, 30, W, H)
        self.assertGreater(float(ref.r_c.sum()), 2 * ref.r_n)
        for refog in (6, 15, 20):
            fx = effects.make("glass", {"refog": refog}, 3, 900, 30, W, H)
            self.assertEqual(fx.r_n, ref.r_n)
            for name in ("r_c", "r_T", "r_phi", "r_y0", "r_L", "r_R", "r_sr", "r_se", "r_xk", "r_yk"):
                self.assertTrue(np.array_equal(getattr(fx, name), getattr(ref, name)), (refog, name))
            for t in (0, 111, 450, 899):
                a, b = fx._running(fx.tau(t)), ref._running(ref.tau(t))
                for k in range(6):  # (all but how open the top is: its trail fogs over more slowly)
                    self.assertTrue(np.array_equal(a[k], b[k]), (refog, t, k))

    def test_trails_last_refog_seconds_through_later_cycles(self):
        """With trails slow to fog over, a slot's drop runs again (twice, and more) while the trail it left earlier
        is still clear (fogged over long before with the default 6 s)."""
        w, h, fps, frames = 640, 360, 30, 1800
        params = {"refog": 20, "speed": 100, "drops": 100, "beads": 0, "hits": 0}
        fx = effects.make("glass", params, 3, frames, fps, w, h)
        quick = effects.make("glass", {**params, "refog": 6}, 3, frames, fps, w, h)
        self.assertGreater(int(fx.r_back.max()), 2)
        slow, fast = [], []
        for i in np.argsort(fx.r_T):
            T, c = fx.r_T[i], int(fx.r_c[i])
            if T > 4.75:  # (two cycles and a bit is under half of 20 s)
                break
            for v in range(0, c, 3):
                y = fx.r_y0[i, v] + 0.5 * fx.r_L[i, v]
                if fx.r_L[i, v] < 6 * fx.r_R[i, v] or not 0.2 * h < y < 0.8 * h:
                    continue
                x = fx._x(np.array([i]), np.array([v]), np.array([y]))[0]
                if not 0.05 * w < x < 0.95 * w:
                    continue
                passed = (v + fx._phase_at(i, v, np.array([y]))[0] - fx.r_phi[i]) / c  # loop phase
                rows, cols = slice(int(y) - 1, int(y) + 2), slice(int(x) - 1, int(x) + 2)
                t = (passed + (2 * T + 0.5) / fx.loop_s) % 1.0 * frames
                for e, out in ((fx, slow), (quick, fast)):
                    e._fields(t)
                    out.append(float(e.buf[glass.CLR][rows, cols].mean()))
        self.assertGreater(len(slow), 10)
        self.assertGreater(min(slow), 0.2)  # two cycles and a bit later, still clear
        # (with 6 s, fogged over: unless another drop or trail is there now)
        self.assertGreater(np.mean(np.array(fast) < np.array(slow) - 0.2), 0.9)
        # on the whole, more of the glass is kept clear by trails
        share = [np.mean([(e._fields(t) or (e.buf[glass.CLR] > 0.5).mean()) for t in range(0, frames, 150)])
                 for e in (quick, fx)]
        self.assertGreater(share[1], 1.5 * share[0])

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
