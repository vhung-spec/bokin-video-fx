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
