import os
import re
import struct
import subprocess
import unittest

from fx import guide, keys, video


class VideoGuide(unittest.TestCase):
    """The video guide (HUONG-DAN.mp4) as the page plays it: an MP4 with its index at the front, H.264, silent, the
    length its chapters say, no tool's name in it, not too big for the installer."""

    def test_the_file(self):
        path = guide.video_path()
        self.assertTrue(os.path.exists(path), path)
        size = os.path.getsize(path)
        self.assertLess(size, 40e6)  # (every update downloads it again: kept small)
        with open(path, "rb") as f:
            data = f.read()
        boxes, at = [], 0
        while at + 8 <= len(data):
            n, kind = struct.unpack(">I4s", data[at:at + 8])
            boxes.append(kind)
            at += n if n >= 8 else len(data)
        self.assertEqual(boxes[:2], [b"ftyp", b"moov"])  # (its index first: it plays at once, from any chapter)
        for word in (b"x264", b"Lavf", b"Lavc", b"ffmpeg", b"Piper", b"vais"):
            self.assertEqual(data.find(word), -1, word)
        info = subprocess.run([video.ffmpeg_exe(), "-hide_banner", "-i", path], capture_output=True,
                              stdin=subprocess.DEVNULL).stderr.decode("utf-8", "replace")
        self.assertIn("Video: h264", info)
        self.assertNotIn("Audio:", info)  # (silent: the owner chose it without the voice)
        h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info).groups()
        self.assertAlmostEqual(int(h) * 3600 + int(m) * 60 + float(s), guide.LENGTH, delta=0.2)

    def test_chapters(self):
        at = [a for a, _ in guide.CHAPTERS]
        self.assertEqual(at[0], 0.0)
        self.assertEqual(at, sorted(at))
        self.assertLess(at[-1], guide.LENGTH)
        self.assertEqual(guide.CHAPTERS[1][1], "1. Kích hoạt app")  # (the activation screen opens the video there)
        view = guide.view()
        self.assertTrue(view["ready"])
        self.assertEqual(len(view["chapters"]), len(guide.CHAPTERS))

    def test_the_guide_has_its_button(self):
        with open(keys.guide_path(), encoding="utf-8") as f:
            self.assertIn("<!-- video -->", f.read())


if __name__ == "__main__":
    unittest.main()
