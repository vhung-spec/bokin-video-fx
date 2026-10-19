import os
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

from tests import helpers  # noqa: F401  (the test data folders)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "brand"))
import make_icons  # noqa: E402


class Brand(unittest.TestCase):
    """The Bokin icon family: every app has its own code and colour, its letters are in the outlines kept, and its
    Windows icon holds every size drawn on its own."""

    def test_family(self):
        codes = [v[0] for v in make_icons.APPS.values()]
        self.assertEqual(len(set(codes)), len(codes))
        self.assertEqual(len({v[2] for v in make_icons.APPS.values()}), len(codes))
        for code in codes:
            for ch in code:
                self.assertIn(ch, make_icons.FONT["glyphs"])

    def test_icon(self):
        with tempfile.TemporaryDirectory() as d:
            path = make_icons.write_ico("fx", os.path.join(d, "fx.ico"))
            with Image.open(path) as im:
                self.assertTrue({(16, 16), (32, 32), (48, 48), (256, 256)} <= set(im.info["sizes"]))
        big = np.asarray(make_icons.tile("fx", 256))
        self.assertEqual(tuple(big[128, 4, :3]), make_icons._hex("#5AA2FF"))  # the tile's colour
        self.assertEqual(big[0, 0, 3], 0)  # rounded: its corner is clear
        ink = np.all(np.abs(big[..., :3].astype(int) - make_icons._hex("#0B1C3D")) < 8, axis=2)
        self.assertGreater(ink[150:, :128].mean(), 0.15)  # the letters, at the bottom left
        self.assertGreater(ink[20:60, 175:235].mean(), 0.1)  # the loop, at the top right
        small = np.asarray(make_icons.tile("fx", 16))
        ink = np.all(np.abs(small[..., :3].astype(int) - make_icons._hex("#0B1C3D")) < 60, axis=2)
        self.assertEqual(ink[:5, 10:].sum(), 0)  # no loop on a tiny icon

    def test_svg(self):
        svg = make_icons.svg("loop")
        self.assertTrue(svg.startswith("<svg") and svg.endswith("</svg>") and "#FFA24D" in svg)
        self.assertNotIn("<text", svg)  # letters as outlines: no font needed to show it
        self.assertIn("evenodd", make_icons.small_svg("loop"))


if __name__ == "__main__":
    unittest.main()
