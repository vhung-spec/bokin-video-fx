"""The fire in a fireplace (fx/hearth.py): a placed lens effect whose flames are a small fluid simulation, made frame
after frame and kept; most checks go through apply()."""
import multiprocessing
import unittest

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects  # (first: it loads every effect's module)
from fx import engine, hearth, logs, overlay

W, H = 480, 270
AT = {"scale": 80, "x": 0, "y": 5}  # the box over much of the frame
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_yy, _xx = np.mgrid[0:H, 0:W]
ROOM = np.dstack([0.3 + 0.2 * (((_xx // 12) + (_yy // 8)) % 2), 0.22 + 0.1 * _xx / W, 0.18 + 0.1 * _yy / H]).astype(
    np.float32)  # a brick wall, its rows and joints
BLACK = np.zeros((H, W, 3), np.float32)


def make(params=None, frames=96, fps=24, w=W, h=H, seed=3):
    return effects.make("hearth", {**AT, **(params or {})}, seed, frames, fps, w, h)


def lens(fx, t, pic=ROOM, mask=None, gain=1.0):
    out = pic.copy()
    fx.apply(t, out, mask, gain)
    return out


def box_of(fx):
    """The box's pixels in the frame: (rows, columns) slices, and the box itself."""
    x0, y0, w, h = fx.box()
    r = fx._rect(x0, y0, x0 + w, y0 + h)
    return (slice(r[1], r[3]), slice(r[0], r[2])), (x0, y0, w, h)


def _glow_elsewhere(k, seed, frames, fps, t):
    """(In another process) the glow of a fire at frame t."""
    return hearth._Fire(k, seed, frames, fps).frame(t)[0].tobytes()


def _logs_elsewhere(params, t):
    """(In another process) a frame with logs."""
    from fx import effects as fx_effects
    return lens(fx_effects.make("hearth", {**AT, **params}, 3, 96, 24, W, H), t, ROOM).tobytes()


LOG_KEYS = {"wood", "stack", "log_size", "char", "view"}
FRAME_KEYS = {"frame", "frame_w", "frame_h", "curve"}
WITH_LOGS = {"wood": "oak", "light": 0}


class Loop(unittest.TestCase):
    def test_frame_p_is_frame_0(self):
        for frames, fps in ((96, 24), (90, 30)):
            fx = make({}, frames, fps)
            for t in (0, 7):
                self.assertTrue(np.array_equal(lens(fx, t), lens(fx, t + frames)), (frames, fps, t))

    def test_no_jump_where_the_loop_closes(self):
        """From the loop's last frame to its first the flames change as from any frame to the next (the crossfade
        that closes it)."""
        for frames, fps, params in ((96, 24, {}), (150, 25, {"speed": 100, "amount": 100}), (90, 30, {"speed": 0})):
            f = make(params, frames, fps).frames()
            g = [f.frame(t)[0] for t in range(frames)]
            step = [float(np.abs(g[t + 1] - g[t]).mean()) for t in range(frames - 1)]
            seam = float(np.abs(g[0] - g[-1]).mean())
            self.assertGreater(min(step), 0.0, params)  # (it moves every frame)
            self.assertLess(seam, 1.5 * float(np.median(step)), (params, seam, float(np.median(step))))
            self.assertLess(max(step), 2.5 * float(np.median(step)), params)

    def test_made_as_far_as_asked(self):
        """A frame asked for makes the frames up to it only (a paused frame does not wait for the whole loop); the
        rest are made as they are asked for, and kept."""
        fx = make({"bed": 55}, 96, 24)
        f = fx.frames()
        self.assertEqual(f.made, 0)
        a = f.frame(10)[0].copy()
        self.assertEqual(f.made, f.join + 11)
        f.frame(3)
        self.assertEqual(f.made, f.join + 11)
        self.assertTrue(np.array_equal(f.frame(10)[0], a))
        f.frame(95)
        self.assertEqual(f.made, f.join + 96)
        self.assertIsNone(f.fluid)  # (all made: the simulation's fields let go)
        self.assertEqual(f.kept.nbytes, (f.join + 96) * hearth.GRID[0] * hearth.GRID[1])

    def test_kept_for_the_next_clip(self):
        """Clips (or scenes made again) with the same fire share it; another fire for other flames, not for other
        colours, light or place."""
        a = make({}).frames()
        self.assertIs(make({"bright": 90, "warmth": 10, "light": 0, "cover": 60, "x": 20}).frames(), a)
        self.assertIsNot(make({"height": 70}).frames(), a)
        self.assertIsNot(make({}, seed=4).frames(), a)
        self.assertIsNot(make({}, 120).frames(), a)

    def test_the_same_fire_in_another_process(self):
        """A video is made in pieces in several processes: each makes the fire from its start, and the same."""
        fx = make({}, 96, 24)
        here = fx.frames().frame(40)[0].tobytes()
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            there = pool.apply(_glow_elsewhere, (fx.k, fx.seed, 96, 24.0, 40))
        self.assertEqual(here, there)

    def test_moves(self):
        fx = make()
        self.assertGreater(float(np.abs(lens(fx, 0, BLACK) - lens(fx, 3, BLACK)).mean()), 0.003)
        self.assertGreater(float(np.abs(lens(fx, 0, BLACK) - lens(fx, 40, BLACK)).mean()), 0.01)


class Picture(unittest.TestCase):
    def test_flames_rise_from_the_bed(self):
        """On black: flames over the bed (its ellipse on the box's bottom edge, "bed" wide), brightest low, none
        at the box's top; as wide as the bed, not wider."""
        for bed in (40, 70, 100):
            fx = make({"bed": bed, "light": 0, "embers": 0})
            (rows, cols), (x0, y0, w, h) = box_of(fx)
            lit = np.mean([lens(fx, t, BLACK)[rows, cols] @ LUMA for t in (10, 30, 50, 70)], axis=0)
            n = lit.shape[0]
            low, top = float(lit[n // 2:].mean()), float(lit[:n // 7].mean())
            self.assertGreater(low, 0.06, bed)
            self.assertLess(top, low / 8, bed)
            xs = (np.arange(cols.start, cols.stop) + 0.5 - x0) / w * 2 - 1  # (-1 .. 1 across the box)
            across = lit[n // 2:].mean(axis=0)
            on = np.abs(xs[across > 0.1 * across.max()])
            self.assertGreater(float(on.max()), 0.65 * bed / 100, bed)
            self.assertLess(float(on.max()), bed / 100 + 0.25, bed)

    def test_the_colours_of_a_fire(self):
        """White-yellow where brightest, orange, dark red where faint; never blue or green."""
        fx = make({"light": 0})
        rgb = np.concatenate([lens(fx, t, BLACK).reshape(-1, 3) for t in (12, 44)])
        luma = rgb @ LUMA
        hot, faint = rgb[luma > 0.75], rgb[(luma > 0.04) & (luma < 0.2)]
        self.assertGreater(len(hot), 50)
        self.assertGreater(len(faint), 50)
        self.assertTrue(np.all(hot[:, 0] >= hot[:, 1] - 1e-3) and np.all(hot[:, 1] >= hot[:, 2] - 1e-3))
        self.assertGreater(float(hot[:, 1].mean()), 0.7)  # (yellow to white)
        self.assertLess(float((faint[:, 1] / faint[:, 0]).mean()), 0.5)  # (red to orange)
        lit = rgb[luma > 0.02]
        self.assertTrue(np.all(lit[:, 2] <= lit[:, 1] + 1e-3) and np.all(lit[:, 1] <= lit[:, 0] + 1e-3))

    def test_only_in_its_box(self):
        """Nothing outside its box changes: not the flames, nor their glow, nor their light, nor an old fire put
        out."""
        for params in ({"light": 100, "cover": 100, "embers": 100}, {"x": 40, "y": 30, "scale": 40}):
            fx = make(params)
            (rows, cols), _ = box_of(fx)
            out = lens(fx, 20)
            outside = np.ones((H, W), bool)
            outside[rows, cols] = False
            self.assertTrue(np.array_equal(out[outside], ROOM[outside]), params)
            self.assertFalse(np.array_equal(out, ROOM), params)

    def test_nothing_cut_at_the_box_edges(self):
        for params in ({}, {"bed": 100, "height": 100, "amount": 100, "embers": 100, "light": 100}):
            fx = make(params)
            (rows, cols), _ = box_of(fx)
            for t in (5, 45):
                part = lens(fx, t, BLACK)[rows, cols]
                rims = (part[:2], part[-2:], part[:, :2], part[:, -2:])
                self.assertLess(max(float(e.max()) for e in rims), 2 / 255, (params, t))

    def test_mask_and_gain(self):
        fx = make({"light": 100, "cover": 50})
        whole = lens(fx, 9)
        half = lens(fx, 9, gain=0.5)
        self.assertLess(float(np.abs((half - ROOM) - 0.5 * (whole - ROOM)).max()), 1e-5)
        mask = np.zeros((H, W), np.float32)
        mask[:, W // 2:] = 1
        masked = lens(fx, 9, mask=mask)
        self.assertTrue(np.array_equal(masked[:, :W // 2], ROOM[:, :W // 2]))
        self.assertLess(float(np.abs(masked[:, W // 2:] - whole[:, W // 2:]).max()), 1e-6)
        self.assertTrue(np.array_equal(lens(fx, 9, gain=0.0), ROOM))

    def test_light_flickers_on_the_firebox(self):
        """The fire's light on what is round it: brighter, warm, as bright as the fire is now."""
        fx, dark = make({"light": 100}), make({"light": 0})
        (rows, cols), (x0, y0, w, h) = box_of(fx)
        spot = (slice(int(y0 + 0.2 * h), int(y0 + 0.3 * h)), slice(int(x0 + 0.08 * w), int(x0 + 0.14 * w)))
        gain = []
        for t in range(0, 96, 4):
            a, b = lens(fx, t)[spot], lens(dark, t)[spot]
            gain.append(float(a[..., 0].mean() / b[..., 0].mean()))
            self.assertGreater(float(a[..., 0].mean() - b[..., 0].mean()), float(a[..., 2].mean() - b[..., 2].mean()))
        self.assertGreater(min(gain), 1.03)
        self.assertGreater(max(gain) - min(gain), 0.02)  # (it flickers)

    def test_an_old_fire_put_out(self):
        """A still, bright fire in the video ("Che lửa cũ" 100): its bright places dimmed under the new flames to
        a dim glow; the dark firebox round it as it was."""
        pic = np.full((H, W, 3), 0.06, np.float32)
        fx0 = make({"cover": 0, "light": 0})
        (rows, cols), (x0, y0, w, h) = box_of(fx0)
        yy, xx = np.mgrid[0:H, 0:W]
        old = (((xx - (x0 + 0.5 * w)) / (0.25 * w)) ** 2 + ((yy - (y0 + 0.7 * h)) / (0.2 * h)) ** 2) < 1
        pic[old] = (1.0, 0.62, 0.15)
        pic[old & ((((xx - (x0 + 0.5 * w)) / (0.1 * w)) ** 2 + ((yy - (y0 + 0.75 * h)) / (0.08 * h)) ** 2) < 1)] = (
            1.0, 0.97, 0.85)  # (its burnt-out white core)
        fx1 = make({"cover": 100, "light": 0})
        a, b = lens(fx0, 30, pic), lens(fx1, 30, pic)
        self.assertLess(float((b[old] @ LUMA).mean()), 0.75 * float((a[old] @ LUMA).mean()))
        far = np.zeros((H, W), bool)
        far[rows, cols] = True
        far &= (np.abs(xx - (x0 + 0.5 * w)) > 0.45 * w) & (yy < y0 + 0.5 * h)
        self.assertLess(float(np.abs(a[far] - b[far]).max()), 0.01)

    def test_off_the_frame_and_tiny(self):
        """A box partly or wholly off the picture, or a tiny frame: no error, nothing outside the picture."""
        for params in ({"x": 100, "y": 0}, {"x": -100, "y": -100}, {"x": 49, "y": 47, "scale": 300},
                       {"scale": 1}, {"scale": 300}):
            fx = make({**params, "light": 100, "cover": 100, "embers": 100})
            self.assertTrue(np.isfinite(lens(fx, 5)).all(), params)
        for w, h in ((8, 6), (1, 1), (33, 2)):
            fx = effects.make("hearth", {"cover": 100, "light": 100}, 1, 48, 24, w, h)
            pic = np.full((h, w, 3), 0.3, np.float32)
            fx.apply(3, pic, None, 1.0)
            self.assertTrue(np.isfinite(pic).all())
        self.assertTrue(np.array_equal(lens(make({"x": -100, "y": -100, "scale": 1}), 5), ROOM))


class Logs(unittest.TestCase):
    """Logs ("Củi"): a pile of logs drawn by code on the bed, charred, lit by the fire, under its flames."""

    def logs_of(self, fx):
        """Where the logs are (the box's part in the frame: rows, columns) and what is traced there."""
        (rows, cols), (x0, y0, w, h) = box_of(fx)
        lay = fx._layout(fx._rect(x0, y0, x0 + w, y0 + h), x0, y0, w, h)
        px0, py0, px1, py1 = lay["pile"]
        return (slice(rows.start + py0, rows.start + py1), slice(cols.start + px0, cols.start + px1)), lay["logs"]

    def test_none_as_before(self):
        """Without logs (a clip made before them, or "none") the fire is drawn as it was."""
        a, b = make({}), make({"wood": "none", "stack": "heap", "log_size": 90})
        self.assertIsNone(a.pile)
        self.assertNotIn("base", a.k)
        self.assertEqual(a.k, b.k)
        for t in (0, 30):
            self.assertTrue(np.array_equal(lens(a, t), lens(b, t)))

    def test_loops(self):
        for params in ({"wood": "oak"}, {"wood": "birch", "stack": "heap", "embers": 100},
                       {"wood": "brown", "stack": "two", "char": 100}):
            fx = make(params)
            for t in (0, 11):
                self.assertTrue(np.array_equal(lens(fx, t), lens(fx, t + 96)), (params, t))

    def test_on_the_bed_in_the_box(self):
        """The logs stand on the bed (its ellipse on the box's bottom edge), about as wide as it, inside the box:
        nothing outside it changes; they cover what is there."""
        for params in ({"wood": "oak"}, {"wood": "oak", "bed": 100, "log_size": 100, "stack": "heap"},
                       {"wood": "birch", "bed": 40, "view": 0}, {"wood": "brown", "view": 100, "x": 30, "scale": 50}):
            fx = make({**params, "light": 100, "embers": 100})
            (rows, cols), (x0, y0, w, h) = box_of(fx)
            out = lens(fx, 20)
            outside = np.ones((H, W), bool)
            outside[rows, cols] = False
            self.assertTrue(np.array_equal(out[outside], ROOM[outside]), params)
            where, laid = self.logs_of(fx)
            cover = np.zeros((H, W), np.float32)
            cover[where] = laid["alpha"]
            ys, xs = np.nonzero(cover > 0.5)
            self.assertGreater(len(ys), 0.01 * w * h, params)
            self.assertFalse((cover[outside] > 0).any(), params)
            bed = fx.bed * w
            self.assertGreater(ys.max(), y0 + h - hearth.MARK * bed - 0.03 * h, params)  # (down on the bed)
            self.assertLess(ys.min(), y0 + h - 0.08 * h, params)
            self.assertGreater(ys.min(), y0 + 0.35 * h, params)  # (not up in the flames)
            across = (xs.max() - xs.min()) / bed
            self.assertTrue(0.7 < across < 1.25 or bed > 0.9 * w, (params, across))
            self.assertLess(abs((xs.max() + xs.min()) / 2 - (x0 + w / 2)), 0.12 * w, params)

    def test_the_flames_behind_the_logs_in_front(self):
        """Where a log in front of the fire covers the picture, its charred wood shows, not the flames behind it
        (their colours have hardly any blue); flames rise above the logs."""
        fx = make({**WITH_LOGS, "height": 70, "embers": 0})
        where, laid = self.logs_of(fx)
        front = (laid["alpha"] > 0.99) & (laid["front"] > 0.99)
        self.assertGreater(int(front.sum()), 200)
        above = (slice(where[0].start - (where[0].stop - where[0].start), where[0].start), where[1])
        for t in (10, 40, 70):
            out = lens(fx, t, BLACK)
            wood = out[where][front]
            self.assertGreater(float(wood[:, 2].sum() / max(wood[:, 0].sum(), 1e-6)), 0.3, t)
            flames = out[above].reshape(-1, 3)
            self.assertGreater(float((flames @ LUMA).max()), 0.6, t)

    def test_lit_by_the_fire(self):
        """The logs are lit by the flames (brighter as they are), more on their tops than under them; their cracks
        glow low down, breathing."""
        dim, bright = make({**WITH_LOGS, "bright": 0}), make({**WITH_LOGS, "bright": 100})
        where, laid = self.logs_of(dim)
        on = laid["alpha"] > 0.99
        a = np.mean([lens(dim, t, BLACK)[where][on] @ LUMA for t in (5, 25)])
        b = np.mean([lens(bright, t, BLACK)[where][on] @ LUMA for t in (5, 25)])
        self.assertGreater(b, 1.2 * a)
        frames = [lens(bright, t, BLACK)[where] for t in range(0, 96, 8)]
        moving = np.std([f[on] @ LUMA for f in frames], axis=0)
        self.assertGreater(float(moving.mean()), 0.002)  # (flickering with the fire)
        cold = make({**WITH_LOGS, "embers": 0})
        hot = make({**WITH_LOGS, "embers": 100})
        rows = laid["alpha"].shape[0]
        low = on.copy()
        low[: int(rows * 0.6)] = False
        red = [float((lens(f, 30, BLACK)[where][low][:, 0] - lens(f, 30, BLACK)[where][low][:, 2]).mean())
               for f in (cold, hot)]
        self.assertGreater(red[1], red[0] + 0.01)  # (the embers in the cracks, glowing)

    def test_each_wood_and_way(self):
        """Every wood and every way of stacking draws logs of its own look, the settings changing them as they say."""
        looks = {}
        for wood in ("oak", "brown", "birch"):
            fx = make({**WITH_LOGS, "wood": wood, "char": 0})
            where, laid = self.logs_of(fx)
            on = laid["alpha"] > 0.99
            looks[wood] = laid["albedo"][on].mean(axis=0)
        self.assertGreater(float(looks["birch"].mean()), 2 * float(looks["oak"].mean()))  # (white bark)
        self.assertGreater(float(looks["brown"][0] / looks["brown"][2]), float(looks["oak"][0] / looks["oak"][2]))
        counts = {way: len(make({**WITH_LOGS, "stack": way}).pile.logs) for way in ("two", "stack", "heap")}
        self.assertEqual(counts, {"two": 2, "stack": 3, "heap": 5})
        small, big = make({**WITH_LOGS, "log_size": 0}).pile, make({**WITH_LOGS, "log_size": 100}).pile
        self.assertGreater(big.r, 1.5 * small.r)
        fresh, burnt = make({**WITH_LOGS, "char": 0}), make({**WITH_LOGS, "char": 100})
        dark = [float(self.logs_of(f)[1]["albedo"][self.logs_of(f)[1]["alpha"] > 0.99].mean()) for f in (fresh, burnt)]
        self.assertLess(dark[1], dark[0])  # (charred: darker)
        low, high = make({**WITH_LOGS, "view": 0}), make({**WITH_LOGS, "view": 100})
        self.assertGreater(self.logs_of(high)[1]["alpha"].sum(), self.logs_of(low)[1]["alpha"].sum())  # (more seen)
        other = make(WITH_LOGS, seed=4).pile.logs[0].p0  # ("Đổi kiểu ngẫu nhiên": laid otherwise)
        self.assertNotEqual(other.tolist(), make(WITH_LOGS).pile.logs[0].p0.tolist())

    def test_a_log_where_it_is(self):
        """A log is where its axis and radius say: rays meet a plain one (no lumps) at its surface, facing out."""
        rng = np.random.default_rng(1)
        log = logs.Log((-0.3, 0.0, 0.1), (0.3, 0.0, 0.1), 0.1, "oak", rng)
        log.shape[:] = 0.0
        log.shape[:, 0] = 1.0
        o = np.array([[x, -1.0, 0.1 + dz] for x in (-0.1, 0.0, 0.2) for dz in (-0.05, 0.0, 0.08)])
        t = log.trace(o, np.array([0.0, 1.0, 0.0]))
        dz = o[:, 2] - 0.1
        self.assertLess(float(np.abs(t - (1.0 - np.sqrt(0.1 ** 2 - dz ** 2))).max()), 1e-4)
        p = o + t[:, None] * np.array([0.0, 1.0, 0.0])
        n = log.normal(*log.local(p))
        want = (p - np.array([p[:, 0], np.zeros(len(p)), np.full(len(p), 0.1)]).T) / 0.1
        self.assertLess(float(np.abs(n - want).max()), 1e-2)
        past = np.array([[0.0, -1.0, 0.25], [0.45, -1.0, 0.1]])  # (above it, beyond its end)
        self.assertTrue(np.isinf(log.trace(past, np.array([0.0, 1.0, 0.0]))).all())

    def test_the_same_logs_in_another_process(self):
        """A video is made in pieces in several processes: each traces the logs, and the same."""
        params = {"wood": "birch", "stack": "heap"}
        here = lens(make(params), 40, ROOM).tobytes()
        with multiprocessing.get_context("spawn").Pool(1) as pool:
            there = pool.apply(_logs_elsewhere, (params, 40))
        self.assertEqual(here, there)

    def test_settings(self):
        """No logs unless chosen; the logs' settings show once a wood is; changing the logs' wood or how burnt makes
        no new fire, their size or place does (the flames rise from among them)."""
        by_key = {p["key"]: p for p in hearth.SCHEMA["params"]}
        self.assertEqual(by_key["wood"]["value"], "none")
        self.assertEqual({o["value"] for o in by_key["wood"]["options"]} - {"none"}, set(logs.WOODS))
        for key in LOG_KEYS - {"wood"}:
            self.assertEqual(by_key[key]["show"], {"wood": list(logs.WOODS)}, key)
            self.assertEqual(by_key[key]["group"], by_key["wood"]["group"], key)
        k = make(WITH_LOGS).k
        self.assertGreater(k["base"], 0)
        for key, v in (("wood", "birch"), ("char", 10), ("light", 100), ("x", 20)):
            self.assertEqual(make({**WITH_LOGS, key: v}).k, k, key)
        for key, v in (("log_size", 90), ("stack", "two"), ("view", 0)):
            self.assertNotEqual(make({**WITH_LOGS, key: v}).k, k, key)


class Frame(unittest.TestCase):
    """Khung lò: the fireplace's opening in the picture (an outline on the box's bottom edge); the fire, its light
    and its logs only inside it."""

    def opening(self, fx):
        """Which pixels of the frame are inside the opening (well inside, well outside: past its soft edge)."""
        (rows, cols), (x0, y0, w, h) = box_of(fx)
        fw, fh = fx.frame_w * w, fx.frame_h * h
        rx, ry = hearth.frame_radii(fx.frame, fw, fh, fx.p["curve"])
        yy, xx = np.mgrid[0:H, 0:W] + 0.5
        u = xx - (x0 + (w - fw) / 2)
        v = yy - (y0 + h - fh)
        inside = (u >= 0) & (u <= fw) & (v >= 0) & (yy <= y0 + h)
        for corner, cx in (((u < rx), rx), ((u > fw - rx), fw - rx)):  # (the top corners: quarter ellipses)
            corner = corner & (v < ry)
            inside &= ~corner | (((u - cx) / max(rx, 1e-9)) ** 2 + ((v - ry) / max(ry, 1e-9)) ** 2 <= 1)
        from PIL import Image, ImageFilter
        img = Image.fromarray((inside * 255).astype(np.uint8))
        well_in = np.asarray(img.filter(ImageFilter.MinFilter(7))) > 0
        well_out = np.asarray(img.filter(ImageFilter.MaxFilter(7))) == 0
        return well_in, well_out

    def test_none_as_before(self):
        a, b = make({}), make({"frame": "none", "frame_w": 30, "frame_h": 40, "curve": 100})
        for t in (0, 30):
            self.assertTrue(np.array_equal(lens(a, t), lens(b, t)))

    def test_only_inside_the_opening(self):
        """Nothing outside the opening changes (past its soft edge: not the flames, their glow, their light, an old
        fire put out, the logs); inside it the fire burns as without the frame, down to the floor."""
        for params in ({"frame": "rect", "frame_w": 60, "frame_h": 70},
                       {"frame": "arch", "frame_w": 50, "frame_h": 90, "curve": 100},
                       {"frame": "round", "frame_w": 85, "frame_h": 60, "curve": 80, "wood": "oak"}):
            fx = make({**params, "light": 100, "cover": 100, "embers": 100, "height": 100})
            free = make({"light": 100, "cover": 100, "embers": 100, "height": 100, "wood": params.get("wood", "none")})
            well_in, well_out = self.opening(fx)
            for t in (10, 50):
                out, whole = lens(fx, t), lens(free, t)
                self.assertTrue(np.array_equal(out[well_out], ROOM[well_out]), (params, t))
                if "wood" not in params:  # (the same fire inside; with logs, the pile fits the opening instead)
                    self.assertLess(float(np.abs(out[well_in] - whole[well_in]).max()), 1e-5, (params, t))
                self.assertGreater(float(np.abs(out[well_in] - ROOM[well_in]).max()), 0.2, (params, t))

    def test_the_opening_as_the_page_draws_it(self):
        """Its top corners as CSS rounds a box's corners (the page draws the outline so): square; an arch over the
        whole width, a half circle at 100; rounded corners."""
        self.assertEqual(hearth.frame_radii("rect", 100, 80, 60), (0.0, 0.0))
        self.assertEqual(hearth.frame_radii("arch", 100, 80, 100), (50.0, 50.0))
        self.assertEqual(hearth.frame_radii("arch", 100, 30, 100), (50.0, 30.0))  # (no higher than the opening)
        self.assertEqual(hearth.frame_radii("arch", 100, 80, 50), (50.0, 25.0))
        rx, ry = hearth.frame_radii("round", 100, 80, 100)
        self.assertEqual(rx, ry)
        self.assertLess(rx, 50)
        fx = make({"frame": "arch", "frame_w": 50, "frame_h": 90, "curve": 100})
        (rows, cols), (x0, y0, w, h) = box_of(fx)
        lay = fx._layout(fx._rect(x0, y0, x0 + w, y0 + h), x0, y0, w, h)
        opening = lay["frame"]
        fw = 0.5 * w
        top = int(round(0.1 * h)) + 2  # (just under the top of the arch, at its middle: inside)
        self.assertGreater(float(opening[top, int(w / 2)]), 0.9)
        self.assertLess(float(opening[top, int((w - fw) / 2) + 2]), 0.1)  # (beside it, in the corner: outside)
        self.assertGreater(float(opening[-1, int(w / 2)]), 0.99)  # (the floor: not faded)

    def test_settings(self):
        """The frame's sizes show once a shape is chosen, its curve for an arch or rounded corners; the page has its
        outline's keys; it makes no new fire; logs fit in it."""
        by_key = {p["key"]: p for p in hearth.SCHEMA["params"]}
        self.assertEqual(by_key["frame"]["value"], "none")
        self.assertEqual({o["value"] for o in by_key["frame"]["options"]} - {"none"}, set(hearth.FRAMES))
        for key in ("frame_w", "frame_h"):
            self.assertEqual(by_key[key]["show"], {"frame": list(hearth.FRAMES)}, key)
        self.assertEqual(by_key["curve"]["show"], {"frame": ["arch", "round"]})
        spec = hearth.SCHEMA["frame"]
        self.assertEqual([spec[k] for k in ("shape", "width", "height", "curve")], ["frame", "frame_w", "frame_h",
                                                                                     "curve"])
        self.assertTrue(spec["label"] and spec["tip"])
        self.assertEqual(make({"frame": "arch", "frame_w": 40}).k, make({}).k)
        narrow = make({"frame": "rect", "frame_w": 40, "wood": "oak", "bed": 100}).pile
        xs = [abs(x) for lg in narrow.logs for x in (lg.p0[0], lg.end()[0])]
        self.assertLess(max(xs) + narrow.r, 0.4 * 4 / 3 / 2 + 1e-6)


class Preview(unittest.TestCase):
    def test_preview_from_the_picture_under_the_box(self):
        """The box alone (as the preview clip is made) on the picture under the box, resized to the clip, gives
        what the whole frame gives there."""
        from PIL import Image
        info = {"w": W, "h": H, "fps": "24", "frames": 96, "seconds": 4.0}
        for params in ({}, {"cover": 80, "light": 100, "bed": 50}, {"wood": "oak", "stack": "heap"},
                       {"frame": "arch", "frame_w": 60, "curve": 100, "light": 100}):
            whole = make(params)
            x0, y0, bw, bh = whole.box()
            layer = {"kind": "hearth", "params": {**AT, **params}}
            cw, ch = engine.fx_clip_frame(info, "hearth", layer)
            alone = effects.make("hearth", engine.fx_clip_params("hearth", {**AT, **params}), 3, 96, 24, cw, ch)
            self.assertEqual(tuple(round(v) for v in alone.box()), (0, 0, cw, ch))

            def under(a):
                return np.asarray(Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).resize(
                    (cw, ch), Image.Resampling.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh)), np.float32) / 255
            for t in (0, 50):
                got = under(ROOM)
                alone.apply(t, got, None, 1.0)
                want = under(lens(whole, t))
                self.assertLess(float(np.abs(got - want).mean()), 0.02, (params, t))

    def test_a_lens_from_the_picture_under_its_box(self):
        cls = hearth.Hearth
        self.assertEqual((cls.blend, cls.lens, cls.UNDER, cls.TURNS, cls.BOX), ("screen", True, True, False, (4, 3)))
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 144, "seconds": 6.0}
        project = {"video": info, "length": 30}
        a = engine.fx_clip_name(project, {"kind": "hearth", "params": {}})
        self.assertTrue(a.endswith(".webm"))
        self.assertNotEqual(a, engine.fx_clip_name(project, {"kind": "hearth", "params": {"x": 5}}))
        self.assertNotEqual(a, engine.fx_clip_name(project, {"kind": "hearth", "params": {"height": 60}}))
        w, h = engine.fx_clip_frame(info, "hearth", {"kind": "hearth", "params": {"scale": 20}})
        self.assertAlmostEqual(w / h, 4 / 3, places=1)
        self.assertLessEqual(h, 1080 * 0.2 + 2)  # (no bigger than the box is in the video)


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_label(self):
        keys = [p["key"] for p in hearth.SCHEMA["params"]]
        self.assertEqual(len(keys), len(set(keys)))
        for p in hearth.SCHEMA["params"]:
            self.assertTrue(p.get("group") and p["label"], p["key"])
        for k in ("look", "bed", "height", "cover", "x", "y", "scale"):
            self.assertIn(k, keys)
        self.assertFalse(set(overlay.ROTATIONS) & set(keys))  # (a fire rises straight up: no turn)
        self.assertTrue(hearth.SCHEMA["note"])

    def test_looks(self):
        """Each look sets every setting but where the fire is, how wide, the old fire put out, the logs and the
        fireplace's opening; the first look is the defaults."""
        look = next(p for p in hearth.SCHEMA["params"] if p["key"] == "look")
        self.assertEqual({o["value"] for o in look["options"]}, set(hearth.LOOKS))
        keys = {p["key"] for p in hearth.SCHEMA["params"]} - {"look", "bed", "cover", "x", "y", "scale"} - LOG_KEYS - \
            FRAME_KEYS
        defaults = effects.defaults("hearth")
        for name, preset in look["presets"].items():
            self.assertEqual(set(preset), keys, name)
            self.assertEqual(effects.clean_params("hearth", preset), {**defaults, **preset}, name)
        self.assertEqual(look["presets"][look["value"]], {k: v for k, v in defaults.items() if k in keys})

    def test_the_bed_mark(self):
        """The page's mark: a flat ellipse sitting on the box's bottom edge, as wide as "bed"; the fire's places
        are on it."""
        mark = hearth.SCHEMA["mark"]
        self.assertEqual((mark["width"], mark["ratio"], mark["sits"]), ("bed", hearth.MARK, True))
        self.assertTrue(mark["label"] and mark["tip"])
        fx = make({"bed": 60})
        fluid = hearth._Fluid(fx.k, fx.seed)
        gw, gh = hearth.GRID
        u = fluid.at / gw * 2 - 1
        self.assertLessEqual(float(np.abs(u).max()), 0.6)
        self.assertGreater(float(np.abs(u).max()), 0.45)
        self.assertLessEqual(float(fluid.up.max()), 1.5 + hearth.MARK * 0.6 * gw + 1e-6)  # (in the ellipse's height)

    def test_what_makes_the_fire_again(self):
        """Only the flames' own settings make the fire again (seconds of work); the colours, the light, the old
        fire and the place change only how it is laid."""
        k = make({}).k
        for key, v in (("bright", 90), ("warmth", 0), ("embers", 100), ("light", 0), ("cover", 100), ("x", 30),
                       ("scale", 20)):
            self.assertEqual(make({key: v}).k, k, key)
        for key, v in (("bed", 30), ("height", 90), ("amount", 10), ("flicker", 90), ("speed", 80)):
            self.assertNotEqual(make({key: v}).k, k, key)


if __name__ == "__main__":
    unittest.main()
