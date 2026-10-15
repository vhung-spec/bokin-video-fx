"""Draws the app icon (a snowflake on a dark blue tile) into packaging/icon.ico. Run before PyInstaller."""
import math
import os

from PIL import Image, ImageDraw

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.ico")


def tile(size):
    s = size * 4  # drawn large, then scaled down smoothly
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=s * 0.22, fill=(32, 52, 86, 255))
    c, r, w = s / 2, s * 0.34, max(2, int(s * 0.07))
    for k in range(6):
        a = math.radians(90 + 60 * k)
        x, y = c + r * math.cos(a), c - r * math.sin(a)
        d.line([(c, c), (x, y)], fill=(214, 236, 255, 255), width=w)
        for side in (-1, 1):  # the small branches
            b = a + side * math.radians(40)
            mx, my = c + r * 0.6 * math.cos(a), c - r * 0.6 * math.sin(a)
            d.line([(mx, my), (mx + r * 0.32 * math.cos(b), my - r * 0.32 * math.sin(b))],
                   fill=(214, 236, 255, 255), width=max(2, int(w * 0.75)))
    d.ellipse([c - w, c - w, c + w, c + w], fill=(214, 236, 255, 255))
    return im.resize((size, size), Image.Resampling.LANCZOS)


if __name__ == "__main__":
    tile(256).save(OUT, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print(OUT)
