import os
import subprocess
import unittest

import numpy as np

from tests import helpers
from fx import overlay, video


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
        ffmpeg("-f", "lavfi", "-i", "testsrc2=s=64x48:r=24:d=1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-color_primaries", "bt2020", "-color_trc", "arib-std-b67", "-colorspace", "bt2020nc", path)
        with self.assertRaises(video.VideoError):
            video.probe(path)


if __name__ == "__main__":
    unittest.main()
