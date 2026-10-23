"""A painted area flickering (fx/flicker.py): a lens inside the clip's area only, the whole area darker by the same
share of its light and back to the video's; its preview clip a black veil."""
import os
import unittest

import numpy as np
from PIL import Image

from tests import helpers
from fx import effects  # (first: it loads every effect's module)
from fx import engine, flicker, paths, projects, video

W, H = 320, 180
MODES = list(flicker.MODES)
_yy, _xx = np.mgrid[0:H, 0:W]
AREA = (((_xx - 160) / 90.0) ** 2 + ((_yy - 90) / 70.0) ** 2 < 1).astype(np.float32)
INSIDE = np.hypot((_xx - 160) / 90.0, (_yy - 90) / 70.0) < 0.6  # (well inside: past the soft rim)


def scene():
    """Dark on the left, bright and warm on the right, and a pattern of every colour across it."""
    pic = np.zeros((H, W, 3), np.float32)
    pic[:, : W // 2] = 0.08
    pic[:, W // 2:] = (0.9, 0.55, 0.25)
    dots = (_xx % 9 < 3) & (_yy % 7 < 2)
    pic[dots] = np.stack([0.2 + 0.7 * _xx / W, 0.9 - 0.6 * _yy / H, 0.5 + 0.0 * _xx], -1)[dots]
    return pic


def sign():
    """A sign at night: a brick wall (its bricks and mortar), lit letters burnt out to white, their glow round them."""
    pic = np.zeros((H, W, 3), np.float32)
    pic[:] = (0.07, 0.05, 0.06)
    pic[(_yy % 12 < 2) | (_xx % 30 < 2)] = (0.04, 0.03, 0.035)
    glow = np.exp(-((_xx - 160) / 70.0) ** 2 - ((_yy - 90) / 30.0) ** 2)[..., None]
    lin = pic ** 2 + glow * np.array([0.5, 0.08, 0.3], np.float32)
    letters = (np.abs(_yy - 90) < 12) & (np.abs(_xx - 160) < 60) & (_xx % 14 < 4)
    lin[letters] = 1.0
    return np.sqrt(np.clip(lin, 0, 1)).astype(np.float32), letters


def make(params=None, frames=144, fps=24, w=W, h=H):
    return effects.make("flicker", params or {}, 3, frames, fps, w, h)


def frame(fx, t, mask=AREA, pic=None):
    pic = scene() if pic is None else pic.copy()
    fx.apply(t, pic, mask, 1.0)
    return pic


def darkest(fx, frames=144):
    return int(np.argmax(fx.darkness((np.arange(frames) + 0.5) / frames)))


class Flicker(unittest.TestCase):
    def test_every_way_loops(self):
        for mode in MODES:
            for frames, fps in ((144, 24), (150, 25)):
                fx = make({"mode": mode, "speed": 100, "amount": 100}, frames, fps)
                self.assertLess(float(np.abs(frame(fx, 0) - frame(fx, frames)).max()), 1 / 512, mode)
                dark = fx.darkness((np.arange(frames) + 0.5) / frames)
                self.assertGreater(float(np.ptp(dark)), 0.5, mode)  # (it does flicker)
                self.assertTrue(0 <= float(dark.min()) and float(dark.max()) <= 1, mode)

    def test_only_inside_the_area(self):
        fx = make({"amount": 100})
        self.assertTrue(np.array_equal(frame(fx, 7, None), scene()))  # (no area: nothing)
        t = darkest(fx)
        changed = np.abs(frame(fx, t) - scene()).max(axis=2)
        self.assertEqual(float(changed[AREA == 0].max()), 0.0)
        # the soft rim: little change right at the outline, the full change well inside
        rim = (AREA > 0) & (np.abs(np.hypot((_xx - 160) / 90.0, (_yy - 90) / 70.0) - 1) < 0.03)
        self.assertLess(float(changed[rim].mean()), 0.3 * float(changed[INSIDE & (_xx > 170)].mean()))

    def test_the_whole_area_alike_its_picture_kept(self):
        """Every pixel well inside the area keeps the same share of its light, dark or bright, of any colour, in
        each of its channels: nothing in it changes but how bright it is (no colour, no shape, no part of it more
        than another), and it is never brighter than in the video."""
        for mode in MODES:
            fx = make({"mode": mode, "edge": 0, "way": "down"})
            for t in range(0, 144, 9):
                before = scene()
                after = frame(fx, t)
                self.assertLessEqual(float((after - before).max()), 0.0, mode)
                if mode == "sweep":
                    continue  # (lit along its length: the same share along each row, see test_lit_bit_by_bit)
                lit = before > 0.01
                share = after[INSIDE & lit.all(axis=2)] / before[INSIDE & lit.all(axis=2)]
                self.assertLess(float(np.ptp(share)), 1e-5, (mode, t))
        fx = make({"mode": "sweep", "edge": 0, "way": "down", "amount": 100})
        after, before = frame(fx, 40), scene()
        share = after[INSIDE] / np.maximum(before[INSIDE], 1e-6)
        rows = np.broadcast_to(_yy[..., None], (H, W, 3))[INSIDE]
        for y in np.unique(rows)[::7]:
            self.assertLess(float(np.ptp(share[rows == y])), 1e-4, y)  # (one share along each row)

    def test_clearly_seen(self):
        """At its first settings (a candle, "Độ nháy" 50) the area is as bright as in the video most of the time and
        flutters darker, at its darkest half as bright: clearly seen, never flat."""
        fx = make({"edge": 0})
        share = np.array([float(frame(fx, t)[INSIDE & (_xx > 170)].mean()) for t in range(144)]) / float(
            scene()[INSIDE & (_xx > 170)].mean())
        self.assertGreater(float(share.max()), 0.97)
        self.assertLess(float(share.min()), 0.55)
        self.assertGreater(float(share.std()), 0.05)
        self.assertGreater(float(np.median(share)), 0.8)

    def test_a_faulty_lamp_is_mostly_steady(self):
        """On as in the video most of the time; now and then a burst of quick flickers right off."""
        fx = make({"mode": "faulty"}, 720, 24)
        levels = fx.level((np.arange(7200) + 0.5) / 7200)
        self.assertGreater(float((levels > -0.02).mean()), 0.8)
        self.assertLess(float(levels.min()), -0.98)
        self.assertLessEqual(float(levels.max()), 0.0)
        off = np.diff((levels < -0.5).astype(int))
        self.assertGreaterEqual(int((off == 1).sum()), 4)  # (a burst: several flickers)

    def test_signs_go_off(self):
        """A sign blinking (at the settings its way sets): on, exactly as in the video; off, the area painted round
        it a fifth as bright, letters, glow and wall alike (its bricks are still bricks: only what is painted
        flickers)."""
        pic, letters = sign()
        p = dict(effects.defaults("flicker"), mode="pulse", edge=0)
        p.update(flicker.MODES["pulse"]["preset"])
        fx = make(p)
        dark = fx.darkness((np.arange(144) + 0.5) / 144)
        off, on = int(np.argmax(dark)), int(np.argmin(dark))
        a, b = pic.copy(), pic.copy()
        fx.apply(off, a, AREA, 1.0)
        fx.apply(on, b, AREA, 1.0)
        self.assertLess(float(np.abs(b - pic).max()), 1e-3)
        inside = INSIDE[..., None] & (pic > 0.01)
        self.assertLess(float(np.abs(a[inside] / pic[inside] - 0.2).max()), 0.01)
        self.assertLess(float(a[letters & INSIDE].mean()), 0.25)
        self.assertTrue(np.array_equal(a[AREA == 0], pic[AREA == 0]))

    def test_lit_bit_by_bit(self):
        """A sign lit along its length (left to right by default; or right to left, top to bottom, bottom to top):
        while it is being lit, its start is on and its end still off; then all of it on; it ends off."""
        for way, start, end in (("right", (90, 75), (90, 245)), ("left", (90, 245), (90, 75)),
                                ("down", (35, 160), (145, 160)), ("up", (145, 160), (35, 160))):
            fx = make({"mode": "sweep", "way": way, "edge": 0, "amount": 100})
            ph = ((np.arange(1440) + 0.5) / 1440 * fx.m + fx.ph / flicker.TAU) % 1.0
            during = int(np.argmin(np.abs(ph - flicker.WIPE / 2))) / 10
            held = int(np.argmin(np.abs(ph - (flicker.WIPE + flicker.HOLD) / 2))) / 10
            gone = int(np.argmin(np.abs(ph - (flicker.DOWN + 1) / 2))) / 10
            pic = np.repeat(np.where(AREA > 0, 0.8, 0.15)[..., None], 3, 2).astype(np.float32)  # (a lit sign on a
            # darker wall)
            a, b, c = pic.copy(), pic.copy(), pic.copy()
            fx.apply(during, a, AREA, 1.0)
            fx.apply(held, b, AREA, 1.0)
            fx.apply(gone, c, AREA, 1.0)
            self.assertGreater(float(a[start].mean()), 0.75, way)
            self.assertLess(float(a[end].mean()), 0.2, way)
            self.assertGreater(float(b[end].mean()), 0.75, way)
            self.assertLess(float(c[start].mean()), 0.2, way)

    def test_settings(self):
        cls = effects.KINDS["flicker"]
        self.assertTrue(cls.AREA and cls.lens and cls.UNDER and cls.VEIL)
        self.assertFalse(cls.TURNS)
        sch = effects.SCHEMA["flicker"]
        self.assertTrue(sch["area_tip"] and sch["area_empty"] and "Vẽ vùng" in sch["note"])
        params = {p["key"]: p for p in sch["params"]}
        self.assertEqual(list(params), ["mode", "way", "amount", "speed", "edge"])  # (no colour, no "bright parts
        # more": the picture stays as it is)
        self.assertEqual(params["way"]["show"], {"mode": ["sweep"]})  # (the way a sign is lit: only for that)
        presets = params["mode"]["presets"]
        self.assertEqual(set(presets), {o["value"] for o in params["mode"]["options"]})  # (every way: its own)
        for mode in ("pulse", "faulty", "sweep"):  # (a sign: quite dark when off)
            self.assertGreaterEqual(presets[mode]["amount"], 70)
        for mode in ("candle", "fire", "breathe"):
            self.assertEqual(presets[mode]["amount"], 50)

    def test_older_clips_still_load(self):
        """A clip made before (its colour and "bright parts more" settings) opens with its way, amount, speed and
        rim as they were."""
        p = effects.clean_params("flicker", {"mode": "faulty", "amount": 70, "speed": 30, "focus": 80, "warm": 40,
                                             "edge": 10})
        self.assertEqual(p, {"mode": "faulty", "way": "right", "amount": 70, "speed": 30, "edge": 10})


class Preview(unittest.TestCase):
    def test_the_veil_is_the_flicker(self):
        """veil() laid as black, that see-through, over the picture is exactly what apply() makes of it."""
        for mode in MODES:
            fx = make({"mode": mode, "amount": 90})
            for t in (0, 33, 71):
                (x0, y0, x1, y1), dark = fx.veil(t, AREA)
                laid = scene()
                laid[y0:y1, x0:x1] *= 1 - dark[..., None]
                self.assertLess(float(np.abs(laid - frame(fx, t)).max()), 1e-6, mode)
        self.assertIsNone(make({"amount": 0}).veil(5, AREA))
        self.assertIsNone(make().veil(5, None))

    def test_preview_clip_is_a_black_veil(self):
        """The preview clip: black, as see-through as the light the area keeps (veil(), at the clip's size), nothing
        outside the area; made without reading the video."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "flick.mp4"), 320, 180, 24)
        project = projects.create(src, "flick.mp4")
        folder = projects.folder(project["id"])
        tid = project["tracks"][0]["id"]
        params = {"mode": "pulse", "amount": 100}
        project = projects.apply(project, {"layers": [{"id": "flick1", "kind": "flicker", "track": tid, "start": 0,
                                                       "end": 720, "seed": 3, "params": params}]})
        os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
        Image.fromarray((AREA * 255).astype(np.uint8)).save(engine.mask_path(folder, "flick1"))
        source = os.path.join(folder, project["video"]["file"])
        os.rename(source, source + ".away")  # (not read)
        try:
            name = engine.fx_clip(folder, project, "flick1", processes=1)
        finally:
            os.rename(source + ".away", source)
        path = os.path.join(folder, "clips", name)
        info = video.probe(path, overlay=True)
        size = (info["w"], info["h"])
        fx = effects.make("flicker", params, 3, engine.project_frames(project), 24, *size)
        area = np.asarray(engine.mask_image(folder, "flick1", size, 0, 180, inside=True), np.float32) / 255
        t = int(np.argmax(fx.darkness((np.arange(48) + 0.5) / engine.project_frames(project))))
        with video.Frames(path, info, alpha=True, start=t, count=1) as frames:
            got = next(iter(frames)).astype(np.float32) / 255
        (x0, y0, x1, y1), dark = fx.veil(t, area)
        want = np.zeros(size[::-1], np.float32)
        want[y0:y1, x0:x1] = dark
        self.assertGreater(float(want.max()), 0.95)
        self.assertLess(float(np.abs(got[..., 3] - want).mean()), 0.01)
        self.assertLess(float(np.percentile(np.abs(got[..., 3] - want), 99)), 0.05)
        self.assertLess(float(got[..., :3][got[..., 3] > 0.2].max()), 0.08)  # (black)

    def test_name_follows_the_area_only(self):
        """The preview clip is made again when the clip's area is painted differently; not for another video, nor
        for a still area painted near it (nothing of the picture is in it)."""
        info = {"w": 480, "h": 270, "fps": "24", "frames": 144, "seconds": 6.0, "file": "input.mp4"}
        project = {"id": "f1ickertest", "video": info, "length": 30, "freeze": {"feather": 6},
                   "still": {"source": "frame", "frame": 0}}
        layer = {"id": "flick2", "kind": "flicker", "params": {}, "seed": 2}
        folder = os.path.join(paths.projects_dir(), project["id"])
        os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
        stamp = [0]

        def paint(name, x, y, w, h):
            m = Image.new("L", (480, 270), 0)
            m.paste(255, (x, y, x + w, y + h))
            path = engine.mask_path(folder, name)
            m.save(path)
            stamp[0] += 1
            os.utime(path, ns=(1, 10 ** 9 * stamp[0]))
            return engine.fx_clip_name(project, layer)

        first = paint("flick2", 100, 60, 120, 80)
        self.assertTrue(first.endswith(".webm"))
        self.assertEqual(paint("freeze", 90, 50, 150, 100), first)  # (a still area right there)
        self.assertEqual(engine.fx_clip_name(dict(project, video=dict(info, file="input-2.mp4")), layer), first)
        self.assertNotEqual(paint("flick2", 100, 60, 130, 80), first)


if __name__ == "__main__":
    unittest.main()
