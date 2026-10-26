"""Built-in effects placed and turned in 3D like a file clip (sheet.py): drawn on a sheet, warped onto the frame."""
import inspect
import math
import os
import tempfile
import time
import unittest

import numpy as np
from PIL import Image, ImageFilter

from tests import helpers  # noqa: F401 - temporary data folder
from tests.test_overlay import _css_corners
from fx import effects, engine, overlay, sheet

W, H = 320, 180
TURN = {"rot_x": 25, "rot_y": -30, "rot_z": 12, "persp": 60, "scale": 70, "x": 6, "y": -4}
_yy, _xx = np.mgrid[0:H, 0:W]
SCENE = np.dstack([40 + 150 * (((_xx // 16) + (_yy // 16)) % 2), 60 + 100 * _xx / W, 70 + 120 * _yy / H]).astype(
    np.uint8)  # something a lens changes


class Dot(effects.Effect):
    """A test effect: a white square at 30% across and 40% down its sheet, whatever its size."""

    def draw(self, t, canvas, mask):
        s = max(2, round(self.h / 30) // 2 * 2)
        canvas.over(round(0.3 * self.w) - s // 2, round(0.4 * self.h) - s // 2, np.ones((s, s), np.float32),
                    np.ones(3, np.float32), 1.0, mask)


class Spot(effects.Effect):
    """A test lens: it turns the picture white only in a small square around the middle of its sheet (or nowhere)."""
    lens = True
    where = (0.5, 0.5)

    def apply(self, t, picture, mask, gain, kept=None):
        if self.where is None:
            return
        x, y, r = round(self.where[0] * self.w), round(self.where[1] * self.h), max(2, self.h // 20)
        square = picture[y - r:y + r, x - r:x + r]
        square += gain * (1.0 - square)
        if kept is not None:
            kept[y - r:y + r, x - r:x + r] = 1.0 - gain

    def draw(self, t, canvas, mask):
        pass


class Kinds:
    """Test kinds added for a test, taken away after it."""

    def __init__(self, **kinds):
        self.kinds = kinds

    def __enter__(self):
        for name, cls in self.kinds.items():
            effects.KINDS[name] = cls
            effects.SCHEMA[name] = {"label": name, "params": list(effects._PLACE + effects._TURN)}

    def __exit__(self, *exc):
        for name in self.kinds:
            del effects.KINDS[name], effects.SCHEMA[name]


def make_scene(kind, params, size=(W, H), area=None):
    """engine.Scene with one clip of `kind` over the whole video (30 s: 720 frames)."""
    folder = tempfile.mkdtemp(prefix="sheet-", dir=helpers.ROOT)
    project = {"id": "sheettest", "video": {"w": size[0], "h": size[1], "fps": "24", "frames": 48, "seconds": 2.0,
                                            "file": "x.mp4"}, "length": 30, "tracks": [{"id": "t1", "on": True}],
               "layers": [{"id": "a1", "kind": kind, "params": params, "seed": 5, "track": "t1", "start": 0,
                           "end": 720}]}
    if area is not None:
        os.makedirs(os.path.join(folder, "masks"))
        area.save(engine.mask_path(folder, "a1"))
    return engine.Scene(folder, project, size)


def scene_frames(kind, params, times, area=None):
    """Frames of make_scene() over SCENE."""
    scene = make_scene(kind, params, (W, H), area)
    try:
        return scene, [scene.frame(t, SCENE) for t in times]
    finally:
        scene.close()


class Plain(unittest.TestCase):
    def test_not_placed_nor_turned_is_the_old_way(self):
        """Old clips (no place, no turn saved) and new ones at the defaults are drawn on the frame exactly as before:
        the same float path, byte for byte, and the same preview clip."""
        new = {"x": 0, "y": 0, "scale": 100, "rot_x": 0, "rot_y": 0, "rot_z": 0, "persp": 50}
        info = {"w": W, "h": H, "fps": "24", "frames": 48, "seconds": 2.0, "file": "input.mp4"}
        for kind in ("snow", "rain", "leaves", "glow", "glass", "steam"):
            p = effects.clean_params(kind, {"amount": 80})
            old = {k: v for k, v in p.items() if k not in new or kind == "steam" and k in ("x", "y", "scale")}
            self.assertFalse(effects.on_sheet(kind, old), kind)
            self.assertFalse(effects.on_sheet(kind, p), kind)
            self.assertFalse(effects.on_sheet(kind, dict(p, persp=100)), kind)  # (perspective alone turns nothing)
            scene, got = scene_frames(kind, old, (0, 31, 117))
            self.assertNotIsInstance(scene.layers[0][0], overlay.Laid)
            fx = effects.make(kind, old, 5, 720, 24, W, H)
            for t, frame in zip((0, 31, 117), got):  # the classic way, written out
                pic = SCENE.astype(np.float32) * (1 / 255.0)
                if fx.lens:
                    fx.apply(t, pic, None, 1.0)
                else:
                    canvas = effects.Canvas(W, H)
                    fx.draw(t, canvas, None)
                    canvas.flush(pic, fx.blend)
                np.clip(pic, 0.0, 1.0, out=pic)
                self.assertEqual(frame.tobytes(), (pic * 255.0 + 0.5).astype(np.uint8).tobytes(), (kind, t))
            _, again = scene_frames(kind, p, (0, 31, 117))
            self.assertTrue(all(a.tobytes() == b.tobytes() for a, b in zip(got, again)), kind)
            project = {"video": info, "length": 30}
            cls = effects.KINDS[kind]
            turn = {k: v for k, v in TURN.items() if not cls.BOX or k not in ("x", "y", "scale")}
            names = {engine.fx_clip_name(project, {"kind": kind, "params": q, "seed": 5})
                     for q in (old, p, dict(p, **turn))}
            self.assertEqual(len(names), 1, kind)  # the preview clip never depends on the turn (nor the place...)
            if cls.BOX and cls.LIT:  # (...but a placed effect lit by the scene is lit by the light where it stands:
                # test_steam, with a video; this project has none to read, so its light is not known)
                moved = {engine.fx_clip_name(project, {"kind": kind, "params": dict(q, x=10), "seed": 5})
                         for q in (p, dict(p, lit=0))}
                self.assertIn(names.pop(), moved, kind)
                self.assertIn(engine.fx_clip_name(project, {"kind": kind, "params": dict(p, lit=0), "seed": 5}), moved)

    def test_settings(self):
        """Every effect but the candle and the rustling leaves has a file clip's place and turn settings, under the
        same headings (the fire in a fireplace too, since 2026-10-26: a fireplace seen from one side); the candle has
        no turn at all (its flame rises straight up), nor the fireflies (a swarm of points), nor the pigeons (they
        stand on the ground, fly level); the rustling leaves
        neither a turn nor a place (they move the picture's own leaves where they are), nor a painted area
        flickering, nor string lights drawn by hand on the picture, nor the smoke of its chimneys (where its things
        are)."""
        file = {p["key"]: p for p in overlay.SCHEMA["params"]}
        for kind, cls in effects.KINDS.items():
            by = {p["key"]: p for p in effects.SCHEMA[kind]["params"]}
            if kind in ("candle", "fireflies", "pigeons", "foliage", "flicker", "garland", "chimney"):
                self.assertFalse(cls.TURNS)
                self.assertFalse(set(overlay.ROTATIONS) & set(by))
                self.assertFalse(effects.on_sheet(kind, {"rot_x": 30, "rot_z": 10}))
                if kind in ("foliage", "flicker", "garland", "chimney"):
                    self.assertFalse({"x", "y", "scale", "persp"} & set(by))
                    self.assertFalse(effects.on_sheet(kind, {"x": 10, "scale": 50}))
                continue
            self.assertTrue(cls.TURNS, kind)
            if cls.lens:  # (a lens that turns says how much of the picture under it it keeps: sheet.Sheet)
                self.assertIn("kept", inspect.signature(cls.apply).parameters, kind)
            for key in overlay.ROTATIONS + ("persp",):
                self.assertEqual({k: v for k, v in by[key].items() if k != "group"},
                                 {k: v for k, v in file[key].items() if k != "group"}, (kind, key))
                self.assertEqual(by[key]["group"], "Xoay 3D")
            for key in ("x", "y", "scale"):
                self.assertEqual(by[key]["group"], "Vị trí và cỡ", (kind, key))
                self.assertEqual(by[key]["label"], file[key]["label"], (kind, key))
                if not cls.BOX:
                    self.assertEqual(by[key], dict(file[key], group="Vị trí và cỡ"), (kind, key))
            self.assertTrue(effects.on_sheet(kind, {"rot_y": 5}), kind)
            self.assertEqual(effects.on_sheet(kind, {"x": 10}), not cls.BOX, kind)  # (steam moves in its box)


class Geometry(unittest.TestCase):
    def test_corners_are_the_css_transform(self):
        """The sheet's corners: the frame's shape placed as a file clip of the frame's size, fit "contain", turned
        as the page's CSS lays its preview clip; a placed effect's: its box, turned."""
        rng = np.random.default_rng(4)
        for _ in range(30):
            p = {"rot_x": float(rng.uniform(-80, 80)), "rot_y": float(rng.uniform(-80, 80)),
                 "rot_z": float(rng.uniform(-180, 180)), "persp": int(rng.integers(0, 101)),
                 "scale": float(rng.uniform(10, 300)), "x": float(rng.uniform(-50, 50)),
                 "y": float(rng.uniform(-50, 50))}
            for kind in ("snow", "steam"):
                q = effects.clean_params(kind, p)
                want = _css_corners(q, effects.sheet_rect(kind, q, 640, 360), 640, 360)
                for g, w in zip(effects.quad(kind, q, 640, 360), want):
                    self.assertAlmostEqual(g[0], w[0], places=6)
                    self.assertAlmostEqual(g[1], w[1], places=6)
            fp = effects.clean_params("file", dict(p, fit="contain"))
            self.assertEqual(effects.quad("rain", p, 640, 360), overlay.quad(fp, {"w": 640, "h": 360}, 640, 360))
        flat = effects.quad("snow", {}, 640, 360)
        self.assertEqual([tuple(round(v, 9) for v in c) for c in flat], [(0, 0), (640, 0), (640, 360), (0, 360)])

    def test_what_is_drawn_lands_where_the_homography_takes_it(self):
        with Kinds(dottest=Dot):
            for p in (TURN, {"rot_z": 90, "scale": 50}, {"rot_y": 60, "persp": 100, "scale": 80},
                      {"x": 20, "y": -15, "scale": 40}, {"rot_x": -50, "rot_z": -140, "persp": 30, "scale": 120}):
                s = sheet.Sheet("dottest", p, 1, 24, 24, 640, 360)
                got = np.asarray(s.draw(0, Image.new("RGB", (640, 360))))[..., 0].astype(float)
                cw, ch = s.size
                size = max(2, round(ch / 30) // 2 * 2)
                cx, cy = round(0.3 * cw) - size // 2 + size / 2, round(0.4 * ch) - size // 2 + size / 2
                u, v, w = s.hm @ np.array([cx * s.step[0], cy * s.step[1], 1.0])
                ys, xs = np.nonzero(got > 8)
                mx = (xs * got[ys, xs]).sum() / got[ys, xs].sum() + 0.5
                my = (ys * got[ys, xs]).sum() / got[ys, xs].sum() + 0.5
                self.assertLess(math.hypot(mx - u / w, my - v / w), 1.0, (p, (mx, my), (u / w, v / w)))

    def test_the_same_scene_as_the_preview_clip(self):
        """Drawn on a sheet rounded to whole pixels (here in a frame of the page's width), the effect is the scene
        of the video (16:9) and of its preview clip: as many leaves, in the same places (one more or less would move
        many of them)."""
        for seed in range(1, 31):
            for p in ({"rot_x": 20, "rot_y": 35, "rot_z": -8, "persp": 70, "scale": 80, "amount": 80},
                      {"rot_y": 10, "scale": 63.3, "amount": 100}):
                s = sheet.Sheet("leaves", p, seed, 720, 24, 706, 397, None, 16 / 9)
                clip = effects.make("leaves", effects.sheet_params("leaves", p), seed, 720, 24, 640, 360)
                self.assertEqual(s.fx.n, clip.n, (seed, p, s.size))
                a, b = s.fx.at(48)[:2], clip.at(48)[:2]
                for u, v in zip(a, b):  # (in frame heights)
                    self.assertLess(float(np.abs(u / s.fx.h - v / clip.h).max()), 1e-6, (seed, p))

    def test_drawn_at_the_size_the_warp_needs(self):
        """Never finer than a frame pixel where the sheet shows largest, never more than the effect at its placed
        size: a small or far sheet is cheap; a big one is held to SHEET_MOST frames."""
        s = sheet.Sheet("snow", {"scale": 30, "rot_z": 10}, 1, 24, 24, 1920, 1080)
        self.assertEqual(s.size, (576, 324))
        s = sheet.Sheet("snow", {"rot_z": 5}, 1, 24, 24, 1920, 1080)
        self.assertEqual(s.size, (1920, 1080))
        s = sheet.Sheet("snow", {"rot_x": 75, "persp": 0, "scale": 40}, 1, 24, 24, 1920, 1080)
        self.assertEqual(s.size, (768, 432))  # (one scale for both ways: an effect's flakes keep their shape)
        s = sheet.Sheet("snow", {"rot_y": 70, "persp": 100, "scale": 30, "x": 30}, 1, 24, 24, 1920, 1080)
        self.assertLessEqual(s.size[1], 324)  # (never finer than at its placed size, however near it comes)
        s = sheet.Sheet("snow", {"rot_z": 5, "scale": 300}, 1, 24, 24, 1920, 1080)
        self.assertLessEqual(s.size[0] * s.size[1], sheet.SHEET_MOST * 1920 * 1080 * 1.01)
        s = sheet.Sheet("steam", {"scale": 45, "rot_y": 20}, 1, 24, 24, 1920, 1080)
        self.assertEqual(s.size[0], s.size[1])  # the box is square
        self.assertLessEqual(s.size[1], round(1080 * 0.45) + 1)

    def test_edge_on_and_out_of_sight(self):
        for p in ({"rot_y": 80, "rot_x": 80, "persp": 100, "scale": 10}, {"x": 100, "y": 100, "rot_z": 10},
                  {"x": -100, "scale": 50, "rot_y": 10}):
            for kind in ("snow", "glass"):
                s = sheet.Sheet(kind, p, 1, 24, 24, 40, 24)
                self.assertTrue(s.empty, (kind, p))
                img = Image.fromarray(SCENE[:24, :40])
                self.assertEqual(np.asarray(s.draw(0, img.copy())).tobytes(), np.asarray(img).tobytes())
        for kind in ("snow", "rain", "leaves", "glow", "glass", "steam"):  # tiny and steep: no error
            for p in ({"rot_x": 79, "rot_y": -79, "persp": 100, "scale": 10}, {"scale": 300, "rot_x": 80, "persp": 100}):
                s = sheet.Sheet(kind, p, 2, 24, 24, W, H)
                got = np.asarray(s.draw(3, Image.fromarray(SCENE)))
                self.assertEqual(got.shape, (H, W, 3))


class Laying(unittest.TestCase):
    def test_soft_outline_and_nothing_outside(self):
        """Outside the turned sheet the frame is untouched; its edge is soft (partly covered pixels)."""
        with Kinds(whitetest=type("White", (effects.Effect,), {"draw": lambda self, t, canvas, mask: canvas.over(
                0, 0, np.ones((self.h, self.w), np.float32), np.ones(3, np.float32), 1.0, mask)})):
            p = dict(TURN)
            s = sheet.Sheet("whitetest", p, 1, 24, 24, W, H)
            got = np.asarray(s.draw(0, Image.new("RGB", (W, H), (128, 128, 128))))[..., 0].astype(int)
            cover = np.zeros((H, W), np.uint8)
            x0, y0, x1, y1 = s.box
            cover[y0:y1, x0:x1] = np.asarray(s.cover)
            self.assertTrue((got[cover == 0] == 128).all())
            self.assertTrue((got[cover == 255] == 255).all())
            edge = (cover > 0) & (cover < 255)
            self.assertGreater(int(edge.sum()), 100)
            self.assertGreaterEqual(int(got.min()), 128)  # (no dark fringe)

    def test_area_and_foreground_after_the_warp(self):
        """The clip's area (with "behind the foreground") is in frame space, after the warp: outside it the frame
        stays exactly as it was, for a turned snow and a turned water on glass."""
        area = Image.new("L", (W, H), 0)
        area.paste(255, (W // 2, 0, W, H))
        for kind, params in (("snow", dict(TURN, amount=100, size=100)), ("glass", dict(TURN, haze=80)),
                             ("steam", {"scale": 80, "rot_y": 30, "rot_z": 10})):
            _, (plain,) = scene_frames(kind, params, (40,))
            scene, (got,) = scene_frames(kind, params, (40,), area=area)
            self.assertIsInstance(scene.layers[0][0], sheet.Sheet)
            left = slice(0, W // 2 - 1)
            self.assertEqual(got[:, left].tobytes(), SCENE[:, left].tobytes(), kind)
            self.assertFalse((plain[:, left] == SCENE[:, left]).all(), kind)  # (it shows there without the area)
            right = slice(W // 2 + 1, W)
            self.assertLessEqual(int(np.abs(got[:, right].astype(int) - plain[:, right]).max()), 1, kind)

    def test_blends(self):
        """Light (Screen) only brightens; Normal covers."""
        img = Image.fromarray(SCENE)
        s = sheet.Sheet("glow", dict(TURN, amount=100, brightness=100), 1, 24, 24, W, H)
        got = np.asarray(s.draw(5, img.copy())).astype(int)
        self.assertTrue((got >= SCENE.astype(int)).all())
        self.assertGreater(int((got > SCENE.astype(int) + 20).sum()), 20)

    def test_fades(self):
        """A fade lays it weaker (gain), as an upright clip; none at 0."""
        img = Image.fromarray(SCENE)
        s = sheet.Sheet("snow", dict(TURN, amount=100), 1, 24, 24, W, H)
        full = np.asarray(s.draw(5, img.copy())).astype(int) - SCENE
        half = np.asarray(s.draw(5, img.copy(), 0.5)).astype(int) - SCENE
        self.assertEqual(np.asarray(s.draw(5, img.copy(), 0.0)).tobytes(), SCENE.tobytes())
        big = np.abs(full) > 40
        self.assertGreater(int(big.sum()), 20)
        self.assertLess(float(np.abs(half[big] - full[big] / 2).max()), 3.0)


class Lens(unittest.TestCase):
    def test_only_the_change_is_laid(self):
        """A lens turned: the picture under the sheet goes onto the sheet, the effect is applied there, and only
        its change comes back. Where it changes nothing, and outside the sheet, every pixel stays as it was."""
        img = Image.fromarray(SCENE)
        with Kinds(spottest=Spot):
            Spot.where = None  # it changes nothing at all
            s = sheet.Sheet("spottest", TURN, 1, 24, 24, W, H)
            self.assertEqual(np.asarray(s.draw(0, img.copy())).tobytes(), SCENE.tobytes())
            Spot.where = (0.5, 0.5)  # only a small square in its middle
            s = sheet.Sheet("spottest", TURN, 1, 24, 24, W, H)
            got = np.asarray(s.draw(0, img.copy())).astype(int)
            diff = np.abs(got - SCENE).max(axis=2)
            ys, xs = np.nonzero(diff)
            cw, ch = s.size
            u, v, w = s.hm @ np.array([0.5 * cw * s.step[0], 0.5 * ch * s.step[1], 1.0])
            self.assertLess(math.hypot(xs.mean() + 0.5 - u / w, ys.mean() + 0.5 - v / w), 1.5)
            reach = (max(2, ch // 20) * math.sqrt(2) + 2) * 2 + 2  # the square and the bilinear's reach, on the frame
            self.assertLess(float(np.hypot(xs + 0.5 - u / w, ys + 0.5 - v / w).max()), reach)
            self.assertGreater(int(diff.max()), 100)  # (the square is white)
            Spot.where = (0.5, 0.5)

    def test_water_on_glass_turned(self):
        """Water on glass turned: it changes the picture inside the quad (fog, drops) and nothing outside it, and
        the frame's edge is repeated under the sheet past the frame (no dark band at the frame's edge)."""
        img = Image.fromarray(SCENE)
        s = sheet.Sheet("glass", dict(TURN, haze=90, blur=80), 3, 48, 24, W, H)
        got = np.asarray(s.draw(7, img.copy())).astype(int)
        cover = np.zeros((H, W), np.uint8)
        x0, y0, x1, y1 = s.box
        cover[y0:y1, x0:x1] = np.asarray(s.cover)
        diff = np.abs(got - SCENE).max(axis=2)
        self.assertTrue((diff[cover == 0] == 0).all())
        self.assertGreater(float(diff[cover == 255].mean()), 5)
        s = sheet.Sheet("glass", {"rot_z": 30, "scale": 150, "haze": 90, "blur": 80, "rim": 0}, 3, 48, 24, W, H)
        flat = Image.new("RGB", (W, H), (200, 180, 160))
        got = np.asarray(s.draw(7, flat.copy())).astype(int)
        self.assertGreater(int(got[2:-2, 2:-2].min()), 120)  # (a black band would pull the fogged glass down)

    def test_water_on_glass_blurs_as_upright(self):
        """Turned by a hair, or fitted to a window seen at an angle, fogged glass blurs the scene behind it as
        upright glass does: the frame's fine detail does not show crisp through the fog (the glass says how much of
        each pixel it keeps, and the frame is laid with that). Clear glass keeps the frame's own pixels, exactly,
        wherever it changes nothing."""
        rng = np.random.default_rng(3)
        fine = np.dstack([60 + 120 * (((_xx // 3) + (_yy // 3)) % 2), 80 + 90 * ((_xx // 5) % 2),
                          100 + 60 * np.sin(_yy / 3.0)])  # fine detail, and grain
        fine = np.clip(fine + rng.normal(0, 10, (H, W, 3)), 0, 255).astype(np.uint8)

        def edges(a):  # how crisp: the mean of a Laplacian
            a = a.astype(float).mean(axis=2)
            return np.abs(4 * a[1:-1, 1:-1] - a[:-2, 1:-1] - a[2:, 1:-1] - a[1:-1, :-2] - a[1:-1, 2:])

        window = {"scale": 45, "x": 22, "y": -12, "rot_x": 10, "rot_y": -35, "rot_z": 4, "persp": 70}
        clear = {"haze": 0, "defocus": 0, "drops": 0, "hits": 0, "beads": 40}
        for params in ({"haze": 100, "blur": 100}, {}, clear):
            fx = effects.make("glass", params, 3, 48, 24, W, H)
            pic = fine.astype(np.float32) * (1 / 255.0)
            fx.apply(5, pic, None, 1.0)
            upright = (np.clip(pic, 0, 1) * 255 + 0.5).astype(np.uint8)
            for turn in ({"rot_z": 0.1}, window):
                s = sheet.Sheet("glass", dict(params, **turn), 3, 48, 24, W, H, None, W / H)
                got = np.asarray(s.draw(5, Image.fromarray(fine)))
                cover = np.zeros((H, W), np.uint8)
                x0, y0, x1, y1 = s.box
                cover[y0:y1, x0:x1] = np.asarray(s.cover)
                inner = np.asarray(Image.fromarray(cover).filter(ImageFilter.MinFilter(7)))[1:-1, 1:-1] == 255
                diff = np.abs(got.astype(int) - fine).max(axis=2)
                self.assertFalse(diff[cover == 0].any(), (params, turn))  # nothing outside the quad
                was, up, now = (float(edges(a)[inner].mean()) for a in (fine, upright, got))
                if params is clear:  # the frame's own detail, untouched where the glass is clear
                    self.assertGreater(float((diff[1:-1, 1:-1][inner] == 0).mean()), 0.85, turn)
                    self.assertLess(abs(now / was - 1), 0.05, (turn, was, now))
                else:  # as blurred as upright glass (adding only the change brought back 0.4 of the detail)
                    self.assertLess(now, 1.2 * up, (params, turn, was, up, now))
                    self.assertLess(now, 0.12 * was, (params, turn, was, up, now))


class Loops(unittest.TestCase):
    def test_turned_returns_to_frame_zero(self):
        """The turn does not move: frame P is frame 0 again."""
        img = Image.fromarray(SCENE)
        for kind in ("snow", "rain", "leaves", "glow", "glass", "steam"):
            for frames, fps in ((144, 24), (90, 30)):
                s = sheet.Sheet(kind, dict(TURN, amount=80), 3, frames, fps, W, H)
                a = np.asarray(s.draw(0, img.copy())).astype(int)
                b = np.asarray(s.draw(frames, img.copy())).astype(int)
                self.assertLessEqual(int(np.abs(a - b).max()), 1, (kind, frames))
                if kind != "glass":
                    self.assertGreater(int(np.abs(np.asarray(s.draw(frames // 3, img.copy())).astype(int) - a).max()),
                                       20, kind)

    def test_a_mix_and_a_cut(self):
        """Two halves of a turned clip with a Mix on the cut: the picture of the uncut clip (as upright)."""
        from fx import projects
        src = helpers.make_loop(os.path.join(helpers.ROOT, "sheetmix.mp4"), W, H, 24)
        project = projects.create(src, "sheetmix.mp4")
        folder, tid = projects.folder(project["id"]), project["tracks"][0]["id"]
        snow = {"id": "snoww", "kind": "snow", "track": tid, "start": 48, "end": 192, "fade_in": 0.5, "fade_out": 0.5,
                "seed": 5, "params": dict(TURN, amount=60)}
        cut = [dict(snow, id="snowc1", end=120, fade_out=0), dict(snow, id="snowc2", start=120, fade_in=0, mix=1.0)]
        times = [100, 112, 119, 120, 123, 131]

        def frames(layers):
            p = projects.apply(projects.load(project["id"]), {"layers": layers})
            scene = engine.Scene(folder, p, (W, H))
            try:
                return {t: scene.frame(t, SCENE).astype(int) for t in times}
            finally:
                scene.close()
        a, b = frames([snow]), frames(cut)
        for t in times:
            self.assertLessEqual(int(np.abs(a[t] - b[t]).max()), 1, t)


class Memory(unittest.TestCase):
    def test_sheets_share_their_scratch(self):
        """The sheets of a scene, and its upright effects, share one canvas and one lens buffer (they are drawn one
        after another, and each leaves the canvas empty): the frames are the very frames drawn with memory of
        their own, and several turned clips hold one canvas, as big as the biggest, not one each."""
        clips = [("snow", {}), ("rain", dict(TURN, amount=70)), ("glass", dict(TURN, haze=70)),
                 ("glow", {"rot_z": 20, "scale": 140}), ("leaves", {"scale": 60, "x": -20}),
                 ("glass", {"rot_y": 30, "scale": 80}), ("snow", {"rot_x": 30, "scale": 200, "persp": 80}),
                 ("steam", {"rot_y": 25}), ("glow", {})]
        folder = tempfile.mkdtemp(prefix="sheet-", dir=helpers.ROOT)
        project = {"id": "sheetmem", "video": {"w": W, "h": H, "fps": "24", "frames": 48, "seconds": 2.0,
                                               "file": "x.mp4"}, "length": 30,
                   "tracks": [{"id": f"t{i}", "on": True} for i in range(len(clips))],
                   "layers": [{"id": f"a{i}", "kind": kind, "params": p, "seed": 5 + i, "track": f"t{i}", "start": 0,
                               "end": 720} for i, (kind, p) in enumerate(clips)]}

        def frames():
            scene = engine.Scene(folder, project, (W, H))
            try:
                return scene, [scene.frame(t, SCENE) for t in (0, 9, 40, 9)]
            finally:
                scene.close()
        scene, shared = frames()
        sheets = [d for d, _, _ in scene.layers if isinstance(d, sheet.Sheet)]
        self.assertEqual(len(sheets), 7)
        blocks = sheets[0].scratch.blocks
        self.assertTrue(all(s.scratch is sheets[0].scratch for s in sheets))
        self.assertEqual(blocks["canvas"].size, max(s.scratch_size()[0] for s in sheets))  # (the biggest: 200%)
        self.assertGreater(blocks["canvas"].size, W * H * 4)
        self.assertTrue(np.shares_memory(scene.canvas.rgb, blocks["canvas"]))
        self.assertFalse(blocks["canvas"].any())  # (left empty)
        floats = sheet.Scratch.floats
        try:  # each asking for memory of its own
            sheet.Scratch.floats = lambda self, name, n: np.zeros(n, np.float32)
            _, own = frames()
        finally:
            sheet.Scratch.floats = floats
        for a, b in zip(shared, own):
            self.assertEqual(a.tobytes(), b.tobytes())
        self.assertEqual(shared[1].tobytes(), shared[3].tobytes())


class Speed(unittest.TestCase):
    def test_turned_at_1080p_in_time(self):
        """The speed budget: a full-frame rain, snow, leaves or water on glass turned at 1080p costs about twice the
        upright one at most (measured: 1.4 to 1.9 times; here, for a busy machine, 2.5 times and a little more).
        Each round's frames are timed together, the best round counts."""
        base = np.zeros((1080, 1920, 3), np.uint8) + 60
        base[::7] = 200  # (something for the glass to blur)
        for kind in ("rain", "snow", "leaves", "glass"):
            scenes = {label: make_scene(kind, p, (1920, 1080)) for label, p in (
                ("upright", {}), ("turned", {"rot_x": 20, "rot_y": -25, "rot_z": 8, "persp": 50}))}
            try:
                for scene in scenes.values():
                    scene.frame(0, base)
                times = {label: math.inf for label in scenes}
                for r in range(3):
                    for label, scene in scenes.items():
                        start = time.perf_counter()
                        for t in (7 + r, 14 + r):
                            scene.frame(t, base)
                        times[label] = min(times[label], (time.perf_counter() - start) / 2)
            finally:
                for scene in scenes.values():
                    scene.close()
            self.assertLess(times["turned"], 2.5 * times["upright"] + 0.05, f"{kind}: {times}")
            self.assertLess(times["turned"], 1.5, f"{kind}: {times}")


if __name__ == "__main__":
    unittest.main()
