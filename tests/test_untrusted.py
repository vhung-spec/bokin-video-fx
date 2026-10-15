import os
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from tests import helpers
from fx import library, overlay, paths, projects, video


class Untrusted(unittest.TestCase):
    """Files and settings from outside: never too costly, never opened by a decoder the app does not need."""

    def test_thin_strip_is_made_only_where_it_shows(self):
        strip = os.path.join(helpers.ROOT, "strip.png")
        Image.new("RGBA", (10000, 4), (255, 255, 255, 255)).save(strip)
        info = overlay.probe_file(strip)
        p = {"fit": "cover", "scale": 300, "x": 0, "y": 0, "blend": "screen", "opacity": 100, "start": 0, "fade": 1}
        with mock.patch.object(Image.Image, "resize", wraps=None, autospec=True,
                               side_effect=lambda im, size, *a, **k: self.assertLessEqual(size[0] * size[1], 320 * 180)
                               or Image.new(im.mode, size)):
            ov = overlay.Overlay(strip, info, p, 24, 24, 320, 180)
        self.assertEqual(ov.picture(0).size, (320, 180))

    def test_thin_video_strip(self):
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "vstrip.mp4"), 2000, 16, 4)
        info = overlay.probe_file(clip)
        p = {"fit": "cover", "scale": 100, "x": 13, "y": 0, "blend": "screen", "opacity": 100, "start": 0, "fade": 0}
        ov = overlay.Overlay(clip, info, p, 4, 24, 320, 180)
        self.assertLess(ov.cursors.size[0], 320 + 2 * 180 / 16 + 2)
        self.assertEqual(ov.picture(1).size, (320, 180))

    def test_high_frame_rate_refused(self):
        before = set(os.listdir(paths.projects_dir()))
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "fast.mp4"), 64, 64, 2, "5001")
        with self.assertRaises(projects.UserError):
            projects.create(clip, "fast.mp4")
        self.assertEqual(set(os.listdir(paths.projects_dir())), before)

    def test_eps_named_png_never_reaches_ghostscript(self):
        eps = os.path.join(helpers.ROOT, "x.png")
        with open(eps, "w") as f:
            f.write("%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 100 100\nshowpage\n")
        with mock.patch("PIL.EpsImagePlugin.Ghostscript", side_effect=AssertionError("Ghostscript")):
            with self.assertRaises(video.VideoError):
                overlay.probe_file(eps)
        jpg = os.path.join(helpers.ROOT, "ok.jpg")
        Image.new("RGB", (40, 30)).save(jpg)
        self.assertEqual(overlay.probe_file(jpg)["w"], 40)

    def test_rename_keeps_slashes(self):
        png = os.path.join(helpers.ROOT, "n.png")
        Image.new("RGB", (40, 30)).save(png)
        item = library.add(png, "n.png")
        self.assertEqual(library.rename(item["id"], "Mưa 1/2")["name"], "Mưa 1 2")
        self.assertEqual(library.rename(item["id"], "A\\B")["name"], "A B")
        self.assertEqual(library.rename(item["id"], "v1.2")["name"], "v1.2")


class Pictures(unittest.TestCase):
    def test_a_gif_kept_under_a_png_name_still_opens(self):
        """Older versions kept pictures by content; such a file must still open (EPS never)."""
        gif = os.path.join(helpers.ROOT, "old.png")
        Image.new("RGB", (40, 30), (200, 10, 10)).save(gif, "GIF")
        self.assertEqual(overlay.probe_file(gif)["w"], 40)

    def test_a_cut_video_keeps_its_colour_in_line(self):
        """A clip made bigger and moved is read in part (crop before scale); on 4:2:0 video the part starts on even
        pixels, so the colour lines up with the picture as when the whole frame is scaled and then cut."""
        clip = os.path.join(helpers.ROOT, "col420.mp4")
        yy, xx = np.mgrid[0:90, 0:160]
        stripes = np.where(((yy // 2 + xx // 2) % 2)[..., None] == 1, [230, 30, 40], [20, 200, 230]).astype(np.uint8)
        enc = video.Encoder(clip, 160, 90, "24", crf=10, preset="veryfast")
        for _ in range(4):
            enc.write(stripes)
        enc.close()
        info = overlay.probe_file(clip)
        p = {"fit": "contain", "scale": 150, "x": 1, "y": 0, "blend": "normal", "opacity": 100, "start": 0,
             "fade": 0}  # the part read starts on source pixel 25, 15 (odd) before it is made even
        ov = overlay.Overlay(clip, info, p, 4, 24, 320, 180)
        sw, sh = ov.cursors.size
        part = np.asarray(ov.picture(1), np.float32)
        x0, y0, x1, y1 = ov.box
        k = min(320 / 160, 180 / 90) * 1.5  # contain, then 150 %: as Overlay places it
        whole_w, whole_h = round(160 * k), round(90 * k)
        with video.Frames(clip, info, (whole_w, whole_h), start=1, count=1) as frames:
            whole = next(iter(frames)).astype(np.float32)
        ox = round((320 - whole_w) / 2 + p["x"] / 100 * 320)
        oy = round((180 - whole_h) / 2 + p["y"] / 100 * 180)
        ref = whole[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        inner = (slice(4, -4), slice(4, -4))  # the outer pixels are filtered from a little less around them
        self.assertLess(float(np.abs(part[..., :3][inner] - ref[inner]).mean()), 2.0)
