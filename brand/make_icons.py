"""The Bokin app icons: one family for every app of the company (chosen 2026-10-19, "hướng B").

Each app is a tile of its own colour with its two-letter code in a deep shade of that colour at the bottom left, and
the Bokin mark, a small open loop, at the top right. The letters are Be Vietnam Pro ExtraBold (SIL Open Font
License 1.1; their outlines are in glyphs.json, so no font is needed here). Below 32 pixels the loop is left out:
it would only be a smudge.

    python brand/make_icons.py            every app: out/<app>.svg, .ico and .png (1024), and out/sheet.png

Only Pillow and numpy: packaging/make_icon.py of each app calls write_ico() before its build. A new app: a line in
APPS (a code nobody has, a colour of a hue the others do not use).
"""
import json
import math
import os
import re

import numpy as np
from PIL import Image, ImageChops, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# id: (code, name, tile colour, letter colour)
APPS = {
    "studio-music": ("Sm", "Bokin Studio Music", "#FF6B97", "#3A0D1E"),
    "playlist-studio": ("Pl", "Bokin Playlist Studio", "#B49BFF", "#241446"),
    "fx": ("Fx", "Bokin FX", "#5AA2FF", "#0B1C3D"),
    "loop": ("Lp", "Bokin Loop", "#FFA24D", "#3A1C04"),
    "motion-studio": ("Mo", "Bokin Motion Studio", "#3DDBC9", "#04302B"),
    "upscaler": ("Up", "Bokin Upscaler", "#C9E66A", "#222D06"),
    "key-center": ("Kc", "Bokin Key Center", "#D9DEE7", "#1A1D23"),
    "web": ("Wb", "Bokin Web", "#F2C54B", "#352805"),
}

RADIUS = 0.23  # of the tile's side
LETTERS = 0.44  # the letters' size (em) as a share of the side
LEFT, BOTTOM = 0.12, 0.07  # the line of letters from the left and bottom edges (as the design page laid it)
TRACKING = -0.03  # em between letters
MARK = (0.7, 0.1, 0.2)  # the loop's box: left, top, side (shares of the side)
MARK_FROM = 32  # pixels: smaller icons have no loop
ICO_SIZES = (16, 20, 24, 32, 40, 48, 64, 96, 128, 256)

with open(os.path.join(HERE, "glyphs.json"), encoding="utf-8") as _f:
    FONT = json.load(_f)


def _hex(c):
    return tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))


def _layout(code, side):
    """(x, y, scale) of each letter's outline (font units, y down from the baseline) on a tile `side` wide: as the
    page laid the line (CSS line-height 1 on the font's ascender and descender, its bottom BOTTOM above the edge)."""
    em = LETTERS * side
    k = em / FONT["upm"]
    content = (FONT["ascender"] - FONT["descender"]) / FONT["upm"]  # (descender is negative)
    baseline_in_box = (1 - content) / 2 + FONT["ascender"] / FONT["upm"]
    baseline = side * (1 - BOTTOM) - em * (1 - baseline_in_box)
    x, out = LEFT * side, []
    for ch in code:
        out.append((x, baseline, k))
        x += FONT["glyphs"][ch]["adv"] * k + TRACKING * em
    return out


def _contours(d):
    """The contours of a glyph's outline (M, L, Q, Z) as lists of points, curves flattened."""
    contours, cur, at = [], [], (0.0, 0.0)
    for op, args in re.findall(r"([MLQZ])([^MLQZ]*)", d):
        n = [float(v) for v in args.split()]
        if op == "M":
            if cur:
                contours.append(cur)
            at = (n[0], n[1])
            cur = [at]
        elif op == "L":
            at = (n[0], n[1])
            cur.append(at)
        elif op == "Q":
            (cx, cy), (ex, ey) = (n[0], n[1]), (n[2], n[3])
            for i in range(1, 9):
                t = i / 8
                cur.append(((1 - t) ** 2 * at[0] + 2 * (1 - t) * t * cx + t * t * ex,
                            (1 - t) ** 2 * at[1] + 2 * (1 - t) * t * cy + t * t * ey))
            at = (ex, ey)
        elif op == "Z" and cur:
            contours.append(cur)
            cur = []
    if cur:
        contours.append(cur)
    return contours


def _letters_mask(code, side):
    """The letters as an "L" mask of side x side, even-odd filled (the holes of o, p, b stay open)."""
    mask = Image.new("1", (side, side), 0)
    for ch, (x, y, k) in zip(code, _layout(code, side)):
        for contour in _contours(FONT["glyphs"][ch]["d"]):
            one = Image.new("1", (side, side), 0)
            ImageDraw.Draw(one).polygon([(x + px * k, y + py * k) for px, py in contour], fill=1)
            mask = ImageChops.logical_xor(mask, one)
    return mask.convert("L")


def tile(app, size):
    """The icon of an app (an id of APPS) as an RGBA image size x size, drawn large and scaled down."""
    code, _, colour, ink = APPS[app]
    s = size * (8 if size < 64 else 4)
    im = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, s - 1, s - 1], radius=s * RADIUS, fill=_hex(colour) + (255,))
    ink_rgba = _hex(ink) + (255,)
    if size >= MARK_FROM:  # the loop: an open circle (the gap at the top right), round ends
        left, top, box = (v * s for v in MARK)
        u = box / 20  # its drawing is 20 units wide
        cx, cy, r, w = left + 10 * u, top + 10 * u, 5.6 * u, 2.6 * u
        d.arc([cx - r - w / 2, cy - r - w / 2, cx + r + w / 2, cy + r + w / 2], 0, 315, fill=ink_rgba, width=round(w))
        for a in (0, 315):
            ex, ey = cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))
            d.ellipse([ex - w / 2, ey - w / 2, ex + w / 2, ey + w / 2], fill=ink_rgba)
    letters = Image.new("RGBA", (s, s), ink_rgba)
    im.paste(letters, (0, 0), _letters_mask(code, s))
    return im.resize((size, size), Image.Resampling.LANCZOS)


def svg(app, mark=True):
    """The icon as an SVG (1024 x 1024), the letters as outlines."""
    code, name, colour, ink = APPS[app]
    s = 1024
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {s} {s}" role="img" aria-label="{name}">',
             f'<rect width="{s}" height="{s}" rx="{s * RADIUS:g}" fill="{colour}"/>']
    if mark:
        left, top, box = (v * s for v in MARK)
        parts.append(f'<path transform="translate({left:g} {top:g}) scale({box / 20:g})" d="M15.6 10a5.6 5.6 0 1 1-1.64-3.96" '
                     f'fill="none" stroke="{ink}" stroke-width="2.6" stroke-linecap="round"/>')
    for ch, (x, y, k) in zip(code, _layout(code, s)):
        parts.append(f'<path transform="translate({x:.2f} {y:.2f}) scale({k:.5f})" fill="{ink}" fill-rule="evenodd" '
                     f'd="{FONT["glyphs"][ch]["d"]}"/>')
    return "".join(parts) + "</svg>"


def _baked(d, x, y, k):
    """A glyph's outline moved and scaled into place, to one decimal (short enough for a page's markup)."""
    def pair(m):
        px, py = float(m.group(1)), float(m.group(2))
        return f"{x + px * k:.1f} {y + py * k:.1f}"
    return re.sub(r"(-?[\d.]+) (-?[\d.]+)", pair, d)


def small_svg(app, side=64):
    """The icon as a short SVG for a page (its tab, its corner): no loop (it is drawn under 32 pixels there)."""
    code, name, colour, ink = APPS[app]
    paths = "".join(_baked(FONT["glyphs"][ch]["d"], x, y, k) for ch, (x, y, k) in zip(code, _layout(code, side)))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}">'
            f'<rect width="{side}" height="{side}" rx="{side * RADIUS:g}" fill="{colour}"/>'
            f'<path fill="{ink}" fill-rule="evenodd" d="{paths}"/></svg>')


def write_ico(app, path):
    """The Windows icon of an app: every size drawn on its own (not one picture scaled)."""
    images = [tile(app, n) for n in ICO_SIZES]
    images[-1].save(path, format="ICO", sizes=[(n, n) for n in ICO_SIZES], append_images=images[:-1])
    return path


def main():
    os.makedirs(OUT, exist_ok=True)
    for app in APPS:
        with open(os.path.join(OUT, app + ".svg"), "w", encoding="utf-8") as f:
            f.write(svg(app))
        tile(app, 1024).save(os.path.join(OUT, app + ".png"))
        write_ico(app, os.path.join(OUT, app + ".ico"))
    sizes = (256, 64, 48, 32, 24, 16)
    sheet = Image.new("RGBA", (24 + len(APPS) * 280, 24 + 280 + 90), (15, 17, 21, 255))
    for i, app in enumerate(APPS):
        x = 24 + i * 280
        big = tile(app, 256)
        sheet.alpha_composite(big, (x, 24))
        cx = x
        for n in sizes[1:]:
            sheet.alpha_composite(tile(app, n), (cx, 24 + 256 + 20 + (64 - n)))
            cx += n + 10
    sheet.save(os.path.join(OUT, "sheet.png"))
    print(OUT)


if __name__ == "__main__":
    main()
