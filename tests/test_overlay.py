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
