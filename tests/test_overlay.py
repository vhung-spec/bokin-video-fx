import unittest

from tests import helpers  # noqa: F401 - temporary data folder
from fx import overlay


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

    def test_kinds_of_files(self):
        self.assertTrue(overlay.is_image("a.PNG") and not overlay.is_image("a.mov"))
        self.assertTrue(overlay.is_video("a.webm") and not overlay.is_video("a.exe"))


if __name__ == "__main__":
    unittest.main()
