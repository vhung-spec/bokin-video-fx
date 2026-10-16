import math
import os
import subprocess
import unittest

import numpy as np

from tests import helpers
from fx import effects, overlay, video


def dominant(parts):
    return max(parts, key=lambda p: p[1])[0]


class Timeline(unittest.TestCase):
    def test_weights_add_up(self):
        for seconds, start, fade in ((10, 0, 1), (10, 3, 2), (2.5, 0, 1), (6, 0, 0), (0.5, 0, 1), (30, 25, 1)):
            for t in range(144):
                parts = overlay.timeline(t, 24, 144, seconds, start, fade)
                self.assertAlmostEqual(sum(w for _, w in parts), 1.0, places=6)
                for when, _ in parts:
                    self.assertGreaterEqual(when, 0)
                    self.assertLessEqual(when, seconds + 1e-9)

    def test_seam_continues_the_file(self):
        """What frame P-1 mostly shows is what comes just before what frame 0 shows."""
        for seconds, fade in ((10, 1), (12, 2.5), (3, 1), (7, 0.5)):
            last = dominant(overlay.timeline(143, 24, 144, seconds, 0, fade))
            first = dominant(overlay.timeline(0, 24, 144, seconds, 0, fade))
            self.assertAlmostEqual(last + 1 / 24, first, places=6, msg=f"{seconds} s file, fade {fade}")

    def test_no_dissolve_inside_the_loop(self):
        parts = overlay.timeline(60, 24, 144, 10, 0, 1)
        self.assertEqual(len(parts), 1)
        self.assertAlmostEqual(parts[0][0], 1 + 60 / 24)

    def test_no_fade_with_a_short_file(self):
        """A fade of 0 with a file shorter than the loop (or a start past its end) must not divide by zero."""
        for frames, fps, seconds, start in ((144, 24000 / 1001, 3.003, 0), (144, 24, 2.0, 2.0), (144, 24, 2.0, 5.0),
                                            (150, 25, 1.04, 0), (90, 30, 0.5, 0.1)):
            for t in range(frames):
                parts = overlay.timeline(t, fps, frames, seconds, start, 0)
                self.assertAlmostEqual(sum(w for _, w in parts), 1.0, places=6)
                self.assertTrue(all(when >= 0 for when, _ in parts))

    def test_clip_plays_the_file_through(self):
        """A clip shorter than the video plays the file from its start, frame after frame, and repeats it with
        a dissolve when the clip is longer than the file."""
        for seconds, need in ((10, 4), (2.5, 8), (1.0, 3)):
            prev = None
            for i in range(int(need * 24)):
                parts = overlay.playing(i / 24, 24, need, seconds, 0, 0.5)
                self.assertAlmostEqual(sum(w for _, w in parts), 1.0, places=6)
                main = dominant(parts)
                if prev is not None and len(parts) == 1:
                    self.assertTrue(abs(main - prev - 1 / 24) < 1e-6 or main < prev, (seconds, i, main, prev))
                prev = main
        self.assertEqual(overlay.playing(3.0, 24, 4, 10, 2, 1), [(5.0, 1.0)])  # from "start" on
        # a clip as long as its file (50 frames at 24 fps), the file's length kept to the millisecond (2.083 s):
        # played once, not once and then again from its start
        self.assertIsNone(overlay._plan(2.083, 0.0, 0.5, 50 / 24, 24)[1])
        self.assertEqual(len(overlay.playing(49 / 24, 24, 50 / 24, 2.083, 0.0, 0.5)), 1)

    def test_kinds_of_files(self):
        self.assertTrue(overlay.is_image("a.PNG") and not overlay.is_image("a.mov"))
        self.assertTrue(overlay.is_video("a.webm") and not overlay.is_video("a.exe"))


def ffmpeg(*args):
    subprocess.run([video.ffmpeg_exe(), "-v", "error", "-y", *args], check=True)


class Files(unittest.TestCase):
    def test_frames_by_number_when_audio_starts_first(self):
        """Readers opened anywhere in the file give frame i, also when the audio starts before the video."""
        path = os.path.join(helpers.ROOT, "late-video.mp4")
        ffmpeg("-f", "lavfi", "-i", "sine=d=4", "-itsoffset", "0.5", "-f", "lavfi", "-i",
               "testsrc2=s=64x48:r=24:d=2", "-map", "0:a", "-map", "1:v", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", path)
        info = video.probe(path, overlay=True)
        with video.Frames(path, info, alpha=True) as frames:
            truth = [f.copy() for f in frames]
        for i in (30, 5, 40, 12):
            cursors = overlay.Cursors(path, info, (info["w"], info["h"]), True)
            try:
                self.assertTrue(np.array_equal(cursors.get(i), truth[i]), f"frame {i}")
            finally:
                cursors.close()

    def test_gif_transparency_is_read_from_the_pixels(self):
        solid, holes = os.path.join(helpers.ROOT, "solid.gif"), os.path.join(helpers.ROOT, "holes.gif")
        ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x48:r=10:d=1", solid)
        from PIL import Image, ImageDraw
        frames = []
        for k in range(3):  # a red square moving over a see-through background
            im = Image.new("RGBA", (64, 48), (0, 0, 0, 0))
            ImageDraw.Draw(im).rectangle([5 + 10 * k, 10, 25 + 10 * k, 30], fill=(255, 0, 0, 255))
            frames.append(im)
        frames[0].save(holes, save_all=True, append_images=frames[1:], duration=100, loop=0, disposal=2)
        self.assertFalse(overlay.probe_file(solid)["alpha"])
        self.assertTrue(overlay.probe_file(holes)["alpha"])

    def test_hdr_is_refused(self):
        path = os.path.join(helpers.ROOT, "hlg.mp4")
        # the tags written into the H.264 stream itself (as cameras do), the same with every ffmpeg build
        ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-bsf:v", "h264_metadata=video_full_range_flag=0:colour_primaries=9:transfer_characteristics=18:"
               "matrix_coefficients=9", path)
        try:
            info = video.probe(path)
        except video.VideoError as e:
            self.assertIn("HDR", str(e))
            return
        described = subprocess.run([video.ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True, text=True).stderr
        self.fail(f"an HLG / BT.2020 video was accepted: {info}\n{described}")


if __name__ == "__main__":
    unittest.main()


class SeeThroughMov(unittest.TestCase):
    def test_prores_4444_with_alpha(self):
        """A MOV in ProRes 4444 with an alpha channel (the usual way effects with a transparent background come) is
        read as see-through, and only its shape covers the picture."""
        path = os.path.join(helpers.ROOT, "leaves-alpha.mov")
        graph = ("color=c=black@0.0:s=160x90:r=25:d=0.4,format=rgba[bg];color=c=orange:s=40x20:r=25:d=0.4,"
                 "format=rgba[fg];[bg][fg]overlay=60:30:format=auto,format=rgba")
        subprocess.run([video.ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i", graph, "-c:v", "prores_ks",
                        "-profile:v", "4444", "-pix_fmt", "yuva444p10le", path], check=True)
        info = overlay.probe_file(path)
        self.assertTrue(info["alpha"])
        p = {"fit": "contain", "scale": 100, "x": 0, "y": 0, "blend": "normal", "opacity": 100, "start": 0, "fade": 0}
        pic = np.asarray(overlay.Overlay(path, info, p, 10, 25, 160, 90).picture(2))
        self.assertEqual(pic.shape[2], 4)
        self.assertEqual(pic[5, 5, 3], 0)  # see-through around the shape
        self.assertEqual(pic[40, 80, 3], 255)  # the shape itself


class Flips(unittest.TestCase):
    P = {"fit": "contain", "scale": 100, "x": 0, "y": 0, "blend": "normal", "opacity": 100, "start": 0, "fade": 0,
         "flip_h": False, "flip_v": False}

    def test_a_picture_turned_over_and_partly_outside(self):
        """Made bigger, moved and turned over: the part in the frame is that part of the turned-over picture."""
        from PIL import Image
        path = os.path.join(helpers.ROOT, "quarters.png")
        im = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
        for i, colour in enumerate([(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255), (255, 255, 0, 255)]):
            im.paste(colour, ((i % 2) * 100, (i // 2) * 50, (i % 2) * 100 + 100, (i // 2) * 50 + 50))
        im.save(path)
        info = overlay.probe_file(path)
        p = dict(self.P, scale=150, x=20, flip_h=True, flip_v=True)
        got = np.asarray(overlay.Overlay(path, info, p, 1, 25, 200, 100).picture(0), np.float32)
        whole = im.resize((300, 150), Image.Resampling.LANCZOS).transpose(Image.Transpose.ROTATE_180)
        want = np.asarray(whole.crop((10, 25, 210, 125)), np.float32)  # placed at (-10, -25)
        self.assertLess(float(np.abs(got - want)[3:-3, 3:-3].mean()), 2.0)

    def test_a_video_turned_over(self):
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "flip.mp4"), 160, 90, 4)
        info = overlay.probe_file(clip)
        p = dict(self.P, scale=50)  # inside the frame
        plain = np.asarray(overlay.Overlay(clip, info, p, 4, 24, 320, 180).picture(1))
        turned = np.asarray(overlay.Overlay(clip, info, dict(p, flip_h=True), 4, 24, 320, 180).picture(1))
        self.assertTrue((turned == plain[:, ::-1]).all())
        upside = np.asarray(overlay.Overlay(clip, info, dict(p, flip_v=True), 4, 24, 320, 180).picture(1))
        self.assertTrue((upside == plain[::-1]).all())


class Colours(unittest.TestCase):
    def test_settings_that_change_nothing(self):
        self.assertIsNone(overlay.colour_ops(effects.clean_params("file", {})))
        self.assertIsNone(overlay.colour_ops(effects.clean_params("file", {"tint": "#00ff00"})))  # no amount

    def test_black_stays_black_and_transparency_stays(self):
        """An effect on black laid with Screen must not get a veil from its colour settings; transparency is kept."""
        from PIL import Image
        im = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
        im.paste((255, 255, 255, 128), (4, 0, 8, 4))
        for p in ({"contrast": -100}, {"contrast": 100, "bright": 100}, {"sat": 100, "hue": 90, "temp": 100},
                  {"tint_amt": 100, "tint": "#ff8030"}):
            out = np.asarray(overlay.coloured(im, overlay.colour_ops(effects.clean_params("file", p))))
            self.assertEqual(out[:, :4, :3].max(), 0, p)
            self.assertTrue((out[..., 3] == np.asarray(im)[..., 3]).all(), p)

    def test_what_each_setting_does(self):
        from PIL import Image
        grey = Image.new("RGB", (2, 2), (128, 128, 128))
        orange = Image.new("RGB", (2, 2), (230, 120, 40))

        def px(im, **p):
            return np.asarray(overlay.coloured(im, overlay.colour_ops(effects.clean_params("file", p))), int)[0, 0]
        self.assertGreater(px(grey, bright=50)[0], 160)
        self.assertLess(px(grey, bright=-50)[0], 100)
        self.assertLess(px(Image.new("RGB", (2, 2), (60, 60, 60)), contrast=80)[0], 60)  # darks darker
        self.assertLess(np.ptp(px(orange, sat=-100)), 2)  # grey
        warm = px(grey, temp=100)
        self.assertGreater(warm[0], warm[2] + 30)
        white = px(Image.new("RGB", (2, 2), (255, 255, 255)), tint_amt=100, tint="#ff8030")
        self.assertEqual(list(white), [255, 128, 48])


def _css_corners(p, rect, w, h):
    """The corners as the browser lays them, written out point by point from the CSS definitions (checked in
    Chromium): flips, rotateZ, rotateY, rotateX about the middle, then perspective(d): scaled by d / (d - z)."""
    x, y, sw, sh = rect
    a, b, c = (math.radians(p.get(k, 0)) for k in ("rot_x", "rot_y", "rot_z"))
    d = overlay.homography(p, rect, w, h)[1]
    out = []
    for u, v in ((0, 0), (sw, 0), (sw, sh), (0, sh)):
        px, py, pz = (u - sw / 2) * (-1 if p.get("flip_h") else 1), (v - sh / 2) * (-1 if p.get("flip_v") else 1), 0.0
        px, py = px * math.cos(c) - py * math.sin(c), px * math.sin(c) + py * math.cos(c)
        px, pz = px * math.cos(b) + pz * math.sin(b), -px * math.sin(b) + pz * math.cos(b)
        py, pz = py * math.cos(a) - pz * math.sin(a), py * math.sin(a) + pz * math.cos(a)
        k = d / (d - pz) if d else 1.0
        out.append((x + sw / 2 + px * k, y + sh / 2 + py * k))
    return out


class Turned(unittest.TestCase):
    """A file clip turned in 3D (Xoay 3D): warped onto the quad the page's CSS transform gives."""

    @classmethod
    def setUpClass(cls):
        from PIL import Image, ImageDraw
        rng = np.random.default_rng(5)
        cls.noise = os.path.join(helpers.ROOT, "turn-noise.png")  # 160 x 120, blocks of 2 x 2 pixels
        blocks = (rng.random((60, 80, 3)) * 255).astype(np.uint8)
        Image.fromarray(np.repeat(np.repeat(blocks, 2, 0), 2, 1)).save(cls.noise)
        cls.dot = os.path.join(helpers.ROOT, "turn-dot.png")  # a red dot on black
        im = Image.new("RGB", (200, 100), (0, 0, 0))
        ImageDraw.Draw(im).rectangle([148, 18, 151, 21], fill=(255, 0, 0))
        im.save(cls.dot)
        cls.white = os.path.join(helpers.ROOT, "turn-white.png")
        Image.new("RGB", (120, 80), (255, 255, 255)).save(cls.white)
        cls.holes = os.path.join(helpers.ROOT, "turn-holes.png")  # see-through but for a square
        im = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
        im.paste((0, 255, 0, 255), (30, 30, 70, 70))
        im.save(cls.holes)

    @staticmethod
    def params(**p):
        return effects.clean_params("file", dict({"fit": "contain", "blend": "normal", "fade": 0}, **p))

    @staticmethod
    def draw(path, p, w=400, h=300, bg=(0, 0, 0)):
        from PIL import Image
        o = overlay.Overlay(path, overlay.probe_file(path), p, 1, 24, w, h)
        try:
            return o, np.asarray(o.draw(0, Image.new("RGB", (w, h), bg)))
        finally:
            o.close()

    def test_no_turn_is_the_plain_way(self):
        """All three turns 0 (whatever the perspective): the plain way, so old projects render exactly as before."""
        old = {k: v for k, v in self.params(scale=50, x=7.3, flip_h=True).items()
               if k not in ("rot_x", "rot_y", "rot_z", "persp")}  # a clip saved before there was any turning
        a, pa = self.draw(self.noise, old)
        self.assertIsNone(a.warp)
        for persp in (0, 50, 100):
            b, pb = self.draw(self.noise, self.params(scale=50, x=7.3, flip_h=True, persp=persp))
            self.assertIsNone(b.warp)
            self.assertEqual(pa.tobytes(), pb.tobytes())
        self.assertFalse(overlay.rotated(self.params()))
        self.assertTrue(overlay.rotated(self.params(rot_y=0.1)))

    def test_z_90_is_a_quarter_turn(self):
        """Turned 90 degrees in the plane, the picture is the upright one turned a quarter clockwise (its middle on a
        pixel corner: every pixel lands on a pixel)."""
        info = overlay.probe_file(self.noise)
        p = self.params(scale=50)
        plain = np.asarray(overlay.Overlay(self.noise, info, p, 1, 24, 400, 300).picture(0))
        o = overlay.Overlay(self.noise, info, dict(p, rot_z=90), 1, 24, 400, 300)
        got = np.asarray(o.picture(0))
        self.assertEqual(o.box, (125, 50, 275, 250))
        want = np.rot90(plain, k=-1)
        self.assertEqual(got.shape[:2], want.shape[:2])
        self.assertLessEqual(int(np.abs(got[..., :3].astype(int) - want[..., :3]).max()), 1)
        self.assertEqual(int(got[..., 3].min()), 255)  # the outline on pixel edges: nothing half covered
        o = overlay.Overlay(self.noise, info, dict(p, rot_z=-90), 1, 24, 400, 300)
        self.assertEqual(o.box, (125, 50, 275, 250))
        self.assertLessEqual(int(np.abs(np.asarray(o.picture(0))[..., :3].astype(int) -
                                        np.rot90(plain, k=1)[..., :3]).max()), 1)

    def test_corners_are_the_css_transform(self):
        info = overlay.probe_file(self.noise)
        rng = np.random.default_rng(9)
        for _ in range(40):
            p = self.params(rot_x=float(rng.uniform(-80, 80)), rot_y=float(rng.uniform(-80, 80)),
                            rot_z=float(rng.uniform(-180, 180)), persp=int(rng.integers(0, 101)),
                            scale=float(rng.uniform(10, 300)), x=float(rng.uniform(-50, 50)),
                            y=float(rng.uniform(-50, 50)), flip_h=bool(rng.integers(2)), flip_v=bool(rng.integers(2)),
                            fit=["cover", "contain"][int(rng.integers(2))])
            got, want = overlay.quad(p, info, 640, 360), _css_corners(p, overlay.placement(p, info, 640, 360), 640, 360)
            for g, wnt in zip(got, want):
                self.assertAlmostEqual(g[0], wnt[0], places=6)
                self.assertAlmostEqual(g[1], wnt[1], places=6)
        # rotateX(30): the top edge goes away (shorter), the bottom comes nearer (longer); the middle stays
        q = overlay.quad(self.params(rot_x=30), info, 640, 360)
        self.assertLess(q[1][0] - q[0][0], q[2][0] - q[3][0])
        self.assertAlmostEqual((q[0][0] + q[1][0]) / 2, 320, places=6)

    def test_the_picture_follows_the_corners(self):
        """A dot of the file lands where the homography takes it."""
        info = overlay.probe_file(self.dot)
        for p in (self.params(rot_x=25, rot_y=-35, rot_z=20, persp=70), self.params(rot_y=50, persp=100, flip_h=True),
                  self.params(rot_z=-150, rot_x=-40, persp=0, scale=80, x=10)):
            _, got = self.draw(self.dot, p)
            rect = overlay.placement(p, info, 400, 300)
            hm, _ = overlay.homography(p, rect, 400, 300)
            k = rect[2] / info["w"]
            u, v, wt = hm @ np.array([150 * k, 20 * k, 1.0])
            red = got[..., 0].astype(float) * (got[..., 1] < 80)
            ys, xs = np.nonzero(red > 60)
            cx = (xs * red[ys, xs]).sum() / red[ys, xs].sum() + 0.5
            cy = (ys * red[ys, xs]).sum() / red[ys, xs].sum() + 0.5
            self.assertLess(math.hypot(cx - u / wt, cy - v / wt), 1.0, (p, (cx, cy), (u / wt, v / wt)))

    def test_soft_outline_without_a_dark_fringe(self):
        """The outline is soft (partly covered pixels), and a white picture laid on grey never darkens it."""
        _, got = self.draw(self.white, self.params(rot_z=30, rot_x=20, persp=60), bg=(128, 128, 128))
        g = got[..., 0].astype(int)
        self.assertGreaterEqual(int(g.min()), 127)
        self.assertGreater(int(((g > 135) & (g < 245)).sum()), 100)  # the soft outline
        self.assertGreater(int((g == 255).sum()), 2000)

    def test_see_through_stays_see_through(self):
        _, got = self.draw(self.holes, self.params(rot_z=45, rot_y=30, persp=50), bg=(10, 20, 30))
        green = (got[..., 1] > 200) & (got[..., 0] < 40)
        self.assertGreater(int(green.sum()), 1000)
        # only the square's own soft edge differs from the background: nothing of its see-through part shows
        other = ~green & (np.abs(got.astype(int) - (10, 20, 30)).max(axis=2) > 3)
        self.assertLess(int(other.sum()), 0.25 * int(green.sum()))

    def test_edge_on_and_out_of_sight(self):
        """Turned (almost) edge-on, very big and turned, or off the frame: no error, no blow-up, nothing or a
        clean picture. No corner is ever at or behind the eye (NEAR)."""
        from PIL import Image
        info = overlay.probe_file(self.noise)
        for p in (self.params(rot_x=80, rot_y=80, persp=100, scale=300, fit="cover"),
                  self.params(rot_x=-80, rot_y=80, rot_z=33, persp=100, scale=300),
                  self.params(rot_y=-80, persp=100, scale=300, x=100), self.params(rot_x=80, rot_y=-80, persp=0)):
            rect = overlay.placement(p, info, 400, 300)
            hm, _ = overlay.homography(p, rect, 400, 300)
            w = hm[2] @ np.array([[0, rect[2], rect[2], 0], [0, 0, rect[3], rect[3]], [1, 1, 1, 1]])
            self.assertGreater(float(w.min()), 0.19, p)
            _, got = self.draw(self.noise, p)
            self.assertEqual(got.shape, (300, 400, 3))
        tiny = os.path.join(helpers.ROOT, "turn-tiny.png")
        Image.new("RGB", (400, 200), (255, 255, 255)).save(tiny)
        o, got = self.draw(tiny, self.params(scale=10, rot_x=80, rot_y=80, persp=100), w=40, h=20)
        self.assertTrue(o.empty)  # a sliver of less than a pixel: nothing
        self.assertEqual(int(got.max()), 0)
        o, got = self.draw(self.noise, self.params(x=100, y=100, rot_z=10))
        self.assertTrue(o.empty)
        self.assertEqual(int(got.max()), 0)

    def test_area_and_blends(self):
        """The clip's area (its weight), Screen, Add and Multiply work turned as they do upright."""
        from PIL import Image
        info = overlay.probe_file(self.white)
        weight = Image.new("L", (400, 300), 0)
        weight.paste(255, (0, 0, 200, 300))  # its left half only
        for blend, bg, inside in (("screen", (40, 40, 40), 255), ("multiply", (200, 100, 50), (200, 100, 50)),
                                  ("add", (40, 40, 40), 255)):
            o = overlay.Overlay(self.white, info, self.params(rot_z=20, blend=blend, scale=70), 1, 24, 400, 300, weight)
            got = np.asarray(o.draw(0, Image.new("RGB", (400, 300), bg))).astype(int)
            self.assertTrue((got[:, 200:] == bg).all(), blend)  # outside its area: untouched
            self.assertTrue((got[150, 190] == inside).all(), (blend, got[150, 190]))

    def test_only_what_shows_is_read(self):
        """A video made big and turned: ffmpeg cuts only the source pixels under what shows."""
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "turn-src.mp4"), 640, 360, 4)
        info = overlay.probe_file(clip)
        o = overlay.Overlay(clip, info, self.params(fit="cover", scale=300, rot_z=15, rot_x=10), 4, 24, 320, 180)
        try:
            x0, y0, x1, y1 = o.cursors.src
            self.assertLess((x1 - x0) * (y1 - y0), 0.3 * 640 * 360)
            pic = o.picture(2)
            self.assertEqual((pic.size, pic.mode), ((320, 180), "RGBA"))
        finally:
            o.close()

    def test_a_1080p_clip_turned_in_time(self):
        """The speed budget: a 1080p video over the whole frame, turned, costs well under a second a frame."""
        import time
        from PIL import Image
        clip = os.path.join(helpers.ROOT, "turn-1080.mp4")
        rng = np.random.default_rng(2)
        enc = video.Encoder(clip, 1920, 1080, "24", crf=24, preset="ultrafast")
        for _ in range(4):
            enc.write((rng.random((1080, 1920, 3)) * 255).astype(np.uint8))
        enc.close()
        base = Image.fromarray((rng.random((1080, 1920, 3)) * 255).astype(np.uint8))
        p = self.params(blend="screen", fit="cover", rot_x=12, rot_y=-18, rot_z=6)
        o = overlay.Overlay(clip, overlay.probe_file(clip), p, 4, 24, 1920, 1080)
        try:
            o.draw(0, base.copy())
            start = time.perf_counter()
            for t in (1, 2, 3):
                o.draw(t, base.copy())
            per = (time.perf_counter() - start) / 3
        finally:
            o.close()
        self.assertLess(per, 1.5, f"{per:.2f} s a frame")
