"""Falling snow (see effects.py for how every effect loops).

Flakes at every depth: far ones tiny, faint, slow and a little bluer (many more of them in heavy snow); nearer ones
bigger and quicker; a few right at the lens, out of focus, that whip past as bright soft orbs. Most flakes are soft
round clumps, as real snow mostly looks. Six-armed ice crystals, drawn by code, that turn and tumble, are the
crystal style; the natural style only has a rare small, soft one near the camera. Which flakes are crystals is
decided in a 1080-line frame, so the small preview, the video and 4K show the same ones. Every flake flutters (side
sway at two whole-number frequencies per loop, slowing at the ends of each swing), drifts in gentle swirls (a
smooth flow field that changes with tau) and with the wind, which blows in gusts that sweep across the frame (always
with the wind), and some of them glint now and then. A haze of snow too far to see flake by flake can veil the
scene: streaky drifts of it blow along with the snow, up to a whiteout in a blizzard.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import sprites
from .effects import Falling, _choice, _colour, _lerp, _num, _rgb

LOOKS = {  # the "Kiểu tuyết" presets: every setting but the colours ("medium" is the defaults, added below)
    "light": {"flake": "mixed", "amount": 22, "size": 35, "speed": 25, "sway": 45, "swirl": 25, "spin": 35,
              "wind": 5, "gust": 15, "depth": 60, "blur": 35, "near": 10, "haze": 0, "twinkle": 15, "glow": 0,
              "opacity": 80},
    "heavy": {"flake": "mixed", "amount": 90, "size": 55, "speed": 40, "sway": 35, "swirl": 35, "spin": 40,
              "wind": 15, "gust": 30, "depth": 70, "blur": 50, "near": 40, "haze": 30, "twinkle": 10, "glow": 0,
              "opacity": 90},
    "blizzard": {"flake": "soft", "amount": 100, "size": 45, "speed": 90, "sway": 30, "swirl": 70, "spin": 60,
                 "wind": 75, "gust": 80, "depth": 75, "blur": 60, "near": 45, "haze": 75, "twinkle": 0, "glow": 0,
                 "opacity": 95},
    "sparkle": {"flake": "crystal", "amount": 35, "size": 60, "speed": 25, "sway": 40, "swirl": 25, "spin": 50,
                "wind": 5, "gust": 15, "depth": 60, "blur": 30, "near": 15, "haze": 0, "twinkle": 70, "glow": 35,
                "opacity": 90},
    "dreamy": {"flake": "mixed", "amount": 30, "size": 75, "speed": 15, "sway": 60, "swirl": 40, "spin": 35,
               "wind": 5, "gust": 10, "depth": 80, "blur": 70, "near": 45, "haze": 0, "twinkle": 20, "glow": 25,
               "opacity": 85},
}

SCHEMA = {"label": "Tuyết rơi", "params": [
    _choice("look", "Kiểu tuyết", "medium",
            [("light", "Tuyết nhẹ"), ("medium", "Tuyết vừa"), ("heavy", "Tuyết dày"), ("blizzard", "Bão tuyết"),
             ("sparkle", "Tuyết lấp lánh"), ("dreamy", "Bông to bay chậm")], presets=LOOKS, group="Kiểu"),
    _choice("flake", "Hình bông tuyết", "mixed",
            [("soft", "Tròn mềm"), ("crystal", "Tinh thể sáu cánh"),
             ("mixed", "Tự nhiên (bông tròn, thỉnh thoảng thấy cánh)")], group="Kiểu"),
    _num("amount", "Mật độ", 45, group="Bông tuyết"), _num("size", "Cỡ bông", 40, group="Bông tuyết"),
    _num("speed", "Tốc độ rơi", 35, group="Chuyển động"), _num("sway", "Đung đưa", 40, group="Chuyển động"),
    _num("swirl", "Xoáy nhẹ theo gió", 35, group="Chuyển động"),
    _num("spin", "Xoay, lật bông tinh thể", 40, group="Chuyển động"),
    _num("wind", "Gió (trái ← → phải)", 10, -100, 100, group="Gió"),
    _num("gust", "Gió giật từng cơn", 25, group="Gió"),
    _num("depth", "Chiều sâu (xa nhỏ, gần to)", 60, group="Chiều sâu"),
    _num("blur", "Nhoè bông ở gần", 40, group="Chiều sâu"),
    _num("near", "Bông to sát ống kính", 25, group="Chiều sâu"),
    _num("haze", "Màn tuyết mờ (trắng xoá khi bão)", 0, group="Chiều sâu"),
    _num("twinkle", "Lấp lánh", 15, group="Ánh sáng"), _num("glow", "Quầng sáng quanh bông", 0, group="Ánh sáng"),
    _num("opacity", "Độ đậm", 85, group="Màu"), _colour("color", "Màu", "#ffffff", group="Màu"),
    _colour("far_color", "Màu bông ở xa", "#e4ebf4", group="Màu")]}
LOOKS["medium"] = {p["key"]: p["value"] for p in SCHEMA["params"] if p["key"] in LOOKS["light"]}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _grid(hx, hy, fx, fy):
    """Pixel centres around a particle at sub-pixel offset (fx, fy) of its own pixel (see sprites.py)."""
    x0, y0 = -int(math.ceil(hx)), -int(math.ceil(hy))
    xs = np.arange(x0, -x0 + 1, dtype=np.float32) + 0.5 - fx
    ys = np.arange(y0, -y0 + 1, dtype=np.float32) + 0.5 - fy
    return x0, y0, xs[None, :], ys[:, None]


GRIDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 19, 22, 26, 30, 35, 40, 46, 53, 61, 70, 80, 92, 106, 122, 140,
                  160, 185, 212, 245, 280, 320])  # half-widths of the grids flakes are drawn on together
GLOW = 0.25  # the share of an out-of-focus flake's light in the faint glow around it
GLOW_SPREAD = 0.85  # that glow's standard deviation, in radii
CORE = 0.45  # how much brighter in the middle than at the edge an out-of-focus flake is (beyond a sharp one's)


def flakes(x, y, r, soft, bokeh, dx, dy, gx, gy):
    """Round flakes, worked out together: each a disc of radius `r` px with edges `soft` px wide, a little
    brighter in the middle, smeared over the (dx, dy) px it moves while a camera's shutter is open. `bokeh` 0..1:
    how far out of focus it is; such a flake is a bright soft orb with a faint glow around it (never darker in
    the middle). x, y: centres (all arrays of n). They share one grid of (2 gx + 1) x (2 gy + 1) pixels around
    their own pixel, wide enough for each. Returns the coverages, (n, 2 gy + 1, 2 gx + 1)."""

    def col(v):
        return np.asarray(v, np.float32)[:, None, None]

    soft = np.maximum(soft, 0.7)
    move = np.hypot(dx, dy)
    xs = (np.arange(-gx, gx + 1, dtype=np.float32) + 0.5)[None, None, :] - col(x - np.floor(x))
    ys = (np.arange(-gy, gy + 1, dtype=np.float32) + 0.5)[None, :, None] - col(y - np.floor(y))
    moving = move > 0.5
    m = np.where(moving, move, 1.0)
    ux, uy = col(np.where(moving, dx / m, 1.0)), col(np.where(moving, dy / m, 0.0))
    along = xs * ux + ys * uy  # a still flake: along x, so d below is the plain distance
    across = xs * uy - ys * ux
    across *= across
    half = col(np.where(moving, move / 2, 0.0))
    beyond = np.maximum(np.abs(along) - half, 0.0)
    d = np.sqrt(beyond * beyond + across)
    rr = col(r)
    out = np.clip((rr - d) / col(soft) + 0.5, 0.0, 1.0)
    out = out * out * (3 - 2 * out)
    if moving.any():  # how long the disc covers each pixel while it moves: the streak fades at both ends (the
        # corners of that rounded off over the edge's width, so a soft flake leaves no seams in its streak)
        chord = np.sqrt(np.maximum(rr * rr - across, 0.25))
        k = col(soft) ** 2
        rounded = (np.asarray(bokeh) > 0).any()

        def least(a, b):
            return 0.5 * (a + b - np.sqrt((a - b) ** 2 + k)) if rounded else np.minimum(a, b)

        stay = (least(along + chord, half) + least(chord - along, half)) / col(m)
        out *= np.where(col(moving) > 0, np.clip(stay, 0.0, 1.0), 1.0)
    tiny = r < 1.5  # a flake smaller than a pixel only covers part of it
    if tiny.any():
        total = out.sum(axis=(1, 2))
        area = math.pi * r * r
        out *= col(np.where(tiny & (total > area) & (area > 0), area / np.maximum(total, 1e-9), 1.0))
    b = np.asarray(bokeh, np.float32)
    shaded = (r >= 2.0) | (b > 0)
    if shaded.any():
        q = np.minimum(d / rr, 1.0)
        out = np.where(col(shaded) > 0, out * (1.0 - col(0.3 + CORE * b) * q * q), out)
    if (b > 0).any():  # the glow: a wide soft spot, smeared too (as much light in all as when still)
        sd = col(GLOW_SPREAD * r)
        glow = np.exp(d * d / (-2 * sd * sd)) / (1 + 2 * half / (2.5066 * sd))
        g = col(GLOW * b)
        out = out * (1 - g) + glow * g
    return out.astype(np.float32, copy=False)


def lay(canvas, ids, alpha, prgb, mask):
    """Like Canvas.over_rgba, but only for the pixels `ids` (flat indices into the frame), one row of `prgb` each:
    the many flakes, scattered all over the frame, are laid over it at little cost."""
    w = canvas.w
    if mask is not None:
        m = mask.reshape(-1)[ids]
        alpha, prgb = alpha * m, prgb * m[:, None]
    rows, cols = ids // w, ids % w
    y0, x0 = int(rows.min()), int(cols.min())
    canvas._place(x0, y0, int(rows.max()) + 1 - y0, int(cols.max()) + 1 - x0)  # (the box it touched)
    rgb, cov = canvas.rgb.reshape(-1, 3), canvas.a.reshape(-1)
    inv = 1.0 - alpha
    rgb[ids] = rgb[ids] * inv[:, None] + prgb
    cov[ids] = cov[ids] * inv + alpha


def _cells(x, y, gx, gy):
    """Pixel centres around particles at x, y (arrays of n), on grids of (2 gx + 1) x (2 gy + 1) pixels around
    their own pixels: xs (n, 1, 2 gx + 1), ys (n, 2 gy + 1, 1)."""
    fx = (x - np.floor(x)).astype(np.float32)[:, None, None]
    fy = (y - np.floor(y)).astype(np.float32)[:, None, None]
    xs = (np.arange(-gx, gx + 1, dtype=np.float32) + 0.5)[None, None, :] - fx
    ys = (np.arange(-gy, gy + 1, dtype=np.float32) + 0.5)[None, :, None] - fy
    return xs, ys


def halos(x, y, spread, gx, gy):
    """halo() of particles at x, y (arrays of n), worked out together: (n, 2 gy + 1, 2 gx + 1)."""
    xs, ys = _cells(x, y, gx, gy)
    return np.exp((xs * xs + ys * ys) * (-0.5 / (spread * spread)).astype(np.float32)[:, None, None])


def glints(x, y, length, gx, gy):
    """glint() of particles at x, y (arrays of n), worked out together: (n, 2 gy + 1, 2 gx + 1)."""
    xs, ys = _cells(x, y, gx, gy)
    ln = np.asarray(length, np.float32)[:, None, None]
    w = 2 * 0.45 ** 2
    rays = np.maximum(np.exp(-ys * ys / w) * np.clip(1 - np.abs(xs) / ln, 0, 1) ** 2.5,
                      np.exp(-xs * xs / w) * np.clip(1 - np.abs(ys) / ln, 0, 1) ** 2.5)
    return np.minimum(0.8 * rays + np.exp(-(xs * xs + ys * ys) / 2.0), 1.0)


def _grid_of(reach):
    """The grid half-widths (from GRIDS, or more) for particles reaching `reach` px from their centres."""
    need = np.ceil(reach)
    return np.maximum(GRIDS[np.minimum(np.searchsorted(GRIDS, need), len(GRIDS) - 1)], need).astype(int)


def halo(spread, fx, fy):
    """The faint glow around a flake: a wide soft spot, `spread` px (its standard deviation), peak 1."""
    x0, y0, xs, ys = _grid(2.5 * spread, 2.5 * spread, fx, fy)
    return np.exp((xs * xs + ys * ys) * (-0.5 / (spread * spread))), x0, y0


def glint(length, fx, fy):
    """A glint of light on a flake: a bright point with four thin rays `length` px long."""
    x0, y0, xs, ys = _grid(length + 1.5, length + 1.5, fx, fy)
    w = 2 * 0.45 ** 2
    rays = np.maximum(np.exp(-ys * ys / w) * np.clip(1 - np.abs(xs) / length, 0, 1) ** 2.5,
                      np.exp(-xs * xs / w) * np.clip(1 - np.abs(ys) / length, 0, 1) ** 2.5)
    core = np.exp(-(xs * xs + ys * ys) / 2.0)
    return np.minimum(0.8 * rays + core, 1.0).astype(np.float32), x0, y0


def smear(cover, dx, dy):
    """`cover` smeared over the (dx, dy) px it moves while the shutter is open: copies along the way, at most a
    pixel apart (each one split between four pixels). Returns (coverage, x0, y0): where its top-left pixel now sits
    relative to the original's."""
    n = int(math.ceil(math.hypot(dx, dy))) + 1
    h, w = cover.shape
    px, py = int(math.ceil(abs(dx) / 2)) + 1, int(math.ceil(abs(dy) / 2)) + 1
    out = np.zeros((h + 2 * py, w + 2 * px), np.float32)
    for k in range(n):
        o = k / (n - 1) - 0.5
        sx, sy = px + o * dx, py + o * dy
        ix, iy = math.floor(sx), math.floor(sy)
        fx, fy = sx - ix, sy - iy
        out[iy:iy + h, ix:ix + w] += cover * ((1 - fx) * (1 - fy))
        out[iy:iy + h, ix + 1:ix + 1 + w] += cover * (fx * (1 - fy))
        out[iy + 1:iy + 1 + h, ix:ix + w] += cover * ((1 - fx) * fy)
        out[iy + 1:iy + 1 + h, ix + 1:ix + 1 + w] += cover * (fx * fy)
    out *= 1 / n
    return out, -px, -py


# --- ice crystals -------------------------------------------------------------------------------------------

CRYSTAL_KINDS = ("dendrite", "fern", "plate", "star", "sector", "dendrite", "fern", "plate")
CRYSTAL_SIZES = (6, 8, 12, 16, 22, 30, 42, 58, 80, 112, 156)  # px across; the arms reach 0.45 of it from the middle
SMALLEST = 2.5  # a crystal whose arms reach less than this (output px) cannot show them: drawn as a round flake
RARE_TIP = (9.0, 15.0)  # the natural style's rare crystals: arms reaching this far (px of a 1080-line frame)
_DESIGNS, _CRYSTALS = {}, {}


def crystal_design(shape):
    """The design of crystal `shape`, the same on all six arms: its branches (where, how long), plates, widths."""
    if shape not in _DESIGNS:
        rng = np.random.default_rng([shape, 6021])
        kind = CRYSTAL_KINDS[shape % len(CRYSTAL_KINDS)]
        pairs = {"dendrite": rng.integers(3, 5), "fern": rng.integers(5, 8), "plate": rng.integers(1, 3),
                 "star": 1, "sector": 0}[kind]
        reach = {"dendrite": rng.uniform(0.4, 0.55), "fern": rng.uniform(0.22, 0.32), "plate": rng.uniform(0.2, 0.3),
                 "star": rng.uniform(0.15, 0.22), "sector": 0.0}[kind]
        at = np.linspace(rng.uniform(0.25, 0.4), rng.uniform(0.72, 0.86), pairs) if pairs else []
        branches = [(float(f), float(reach * (1 - f) ** 0.7 * rng.uniform(0.85, 1.15))) for f in at]
        _DESIGNS[shape] = {
            "kind": kind, "branches": branches, "twigs": kind == "dendrite",
            "width": rng.uniform(0.045, 0.07) * (1.6 if kind == "sector" else 1.0),
            "plate": {"plate": rng.uniform(0.16, 0.24), "sector": rng.uniform(0.5, 0.62),
                      "star": rng.uniform(0.08, 0.12)}.get(kind, rng.uniform(0.05, 0.1)),
            "tip": rng.uniform(0.045, 0.065) if kind == "plate" else 0.0}
    return _DESIGNS[shape]


def _hexagon(cx, cy, r, turn):
    return [(cx + r * math.cos(turn + k * math.pi / 3), cy + r * math.sin(turn + k * math.pi / 3)) for k in range(6)]


def crystal(shape, size, soft=False):
    """Crystal `shape` drawn `size` px across as coverage 0..1 (a Pillow "L" image): six arms with side branches,
    a translucent plate and a soft core in the middle, and a soft icy glow. No stroke is thinner than about a
    pixel. `soft`: a crystal tumbling in the air a little out of focus, a soft clump with a hint of its arms."""
    key = (shape, size, soft)
    if key in _CRYSTALS:
        return _CRYSTALS[key]
    design = crystal_design(shape)
    ss = 4  # supersampling
    big = size * ss
    c, arm = big / 2, 0.45 * big
    width = max(design["width"] * arm, 0.9 * ss)
    plate = Image.new("L", (big, big), 0)
    lines = Image.new("L", (big, big), 0)
    core = Image.new("L", (big, big), 0)
    pd, ld = ImageDraw.Draw(plate), ImageDraw.Draw(lines)
    rc = (0.4 if soft else 0.22) * arm  # the soft core: small crystals read as flakes with a hint of arms
    ImageDraw.Draw(core).ellipse((c - rc, c - rc, c + rc, c + rc), fill=200 if soft else 120)

    def bar(x0, y0, x1, y1, w0, w1):  # a tapered stroke
        ln = math.hypot(x1 - x0, y1 - y0) or 1.0
        nx, ny = -(y1 - y0) / ln, (x1 - x0) / ln
        ld.polygon([(x0 + nx * w0 / 2, y0 + ny * w0 / 2), (x1 + nx * w1 / 2, y1 + ny * w1 / 2),
                    (x1 - nx * w1 / 2, y1 - ny * w1 / 2), (x0 - nx * w0 / 2, y0 - ny * w0 / 2)], fill=255)

    up = -math.pi / 2
    pd.polygon(_hexagon(c, c, design["plate"] * arm, up), fill=90 if design["kind"] == "sector" else 150)
    thin = max(0.55 * width, 0.9 * ss)
    for k in range(6):
        a = up + k * math.pi / 3
        dx, dy = math.cos(a), math.sin(a)
        bar(c, c, c + dx * arm, c + dy * arm, width, max(0.5 * width, 0.9 * ss))
        for f, ln in design["branches"]:
            bx, by = c + dx * f * arm, c + dy * f * arm
            for side in (-1, 1):
                b = a + side * math.pi / 3
                ex, ey = bx + math.cos(b) * ln * arm, by + math.sin(b) * ln * arm
                bar(bx, by, ex, ey, thin, max(0.6 * thin, 0.9 * ss))
                if design["twigs"] and ln > 0.14:  # little twigs on long branches, parallel to the arm
                    tx, ty = bx + math.cos(b) * 0.55 * ln * arm, by + math.sin(b) * 0.55 * ln * arm
                    for s2 in (-1, 1):
                        t2 = b + s2 * math.pi / 3
                        bar(tx, ty, tx + math.cos(t2) * 0.3 * ln * arm, ty + math.sin(t2) * 0.3 * ln * arm,
                            max(0.8 * thin, 0.9 * ss), 0.9 * ss)
        if design["tip"]:
            pd.polygon(_hexagon(c + dx * arm * 0.94, c + dy * arm * 0.94, design["tip"] * arm, up), fill=170)
    blur = ImageFilter.GaussianBlur(max(0.8, 0.05 * size))
    core = np.asarray(core.resize((size, size), Image.Resampling.LANCZOS).filter(blur), np.float32)
    cover = np.maximum(np.maximum(np.asarray(plate.resize((size, size), Image.Resampling.LANCZOS), np.float32),
                                  np.asarray(lines.resize((size, size), Image.Resampling.LANCZOS), np.float32)),
                       core) / 255.0
    glow = np.asarray(Image.fromarray((cover * 255).astype(np.uint8)).filter(blur), np.float32) / 255.0
    cover = np.clip(cover + 0.3 * glow, 0.0, 1.0)
    if soft:  # out of focus: the arms blur into a clump, brightest in the middle
        wide = ImageFilter.GaussianBlur(max(1.0, 0.07 * size))
        cover = np.asarray(Image.fromarray((cover * 255).astype(np.uint8)).filter(wide), np.float32) / 255.0
        cover = 0.6 * cover + 0.4 * glow + 0.3 * core / 255.0
        cover = np.clip(0.95 * cover / max(float(cover.max()), 1e-6), 0.0, 1.0)
    img = Image.fromarray((cover * 255 + 0.5).astype(np.uint8), "L")
    _CRYSTALS[key] = img
    return img


def crystal_size(tip):
    """The crystal sprite (px across) drawn for arms reaching `tip` px: the smallest that is big enough."""
    return next((s for s in CRYSTAL_SIZES if 0.45 * s >= tip), CRYSTAL_SIZES[-1])


_INK = {}


def crystal_ink(shape, size, soft=False):
    """How much of its square crystal `shape` covers when drawn `size` px across (0..1)."""
    if (shape, size, soft) not in _INK:
        _INK[shape, size, soft] = float(np.asarray(crystal(shape, size, soft), np.float32).mean()) / 255.0
    return _INK[shape, size, soft]


# --- gusts and the haze ---------------------------------------------------------------------------------------

def _gust_curve(n=1024, sharp=4.0):
    """How far a gust has pushed the flakes over one gust cycle (0..2 pi), from -0.5 to 0.5, beyond a steady
    drift: the wind blows in short bursts (exp(sharp * (cos - 1))) and calmly between them. The curve has the
    bursts' mean taken away, so it is periodic; `lead` is that mean as a drift per radian of the cycle (in the
    curve's units), which the flakes' straight paths carry instead, so the push never turns against the wind."""
    th = np.arange(n) * (2 * math.pi / n)
    push = np.exp(sharp * (np.cos(th) - 1))
    mean = float(push.mean())
    curve = np.cumsum(push - mean)
    span = float(curve.max() - curve.min())
    curve -= 0.5 * (curve.max() + curve.min())
    curve /= span
    return np.append(curve, curve[0]), mean / span * n / (2 * math.pi)


_GUST, GUST_LEAD = _gust_curve()


def gust_push(theta):
    return np.interp(np.mod(theta, 2 * math.pi) * (len(_GUST) - 1) / (2 * math.pi), np.arange(len(_GUST)), _GUST)


def drifts(rng, n, along, across):
    """A smooth random pattern that wraps around (an n x n tile), standard deviation 1: about `along` blobs a tile
    one way and `across` the other, so the blobs are streaks when `across` is the larger."""
    f = np.fft.fftfreq(n) * n
    g = np.fft.rfftfreq(n) * n
    amp = np.exp(-(f[:, None] / along) ** 2 - (g[None, :] / across) ** 2)
    amp[0, 0] = 0.0
    spec = amp * np.exp(2j * math.pi * rng.random(amp.shape))
    field = np.fft.irfft2(spec, s=(n, n))
    return (field / max(float(field.std()), 1e-9)).astype(np.float32)


def sample(field, u, v):
    """`field` (a tile that wraps around) at u, v (in tiles), bilinear."""
    n = field.shape[0]
    u, v = u * n, v * n
    i, j = np.floor(u), np.floor(v)
    fu, fv = (u - i).astype(np.float32), (v - j).astype(np.float32)
    i0, j0 = i.astype(np.int64) % n, j.astype(np.int64) % n
    i1, j1 = (i0 + 1) % n, (j0 + 1) % n
    return ((field[i0, j0] * (1 - fv) + field[i0, j1] * fv) * (1 - fu)
            + (field[i1, j0] * (1 - fv) + field[i1, j1] * fv) * fu)


class Snow(Falling):
    CAP = 9000  # flakes with fixed random draws
    NEAR = 240  # flakes right at the lens (their own draws)
    VERSION = 2
    SHARP = 16  # how brief a glint is
    SHUTTER = 0.35  # a frame's share the camera's shutter is open: how much a moving flake smears
    CRISP = 0.8  # crystals smear a little less, so their arms stay clear
    FAR = 0.35  # the far band: many more of these tiny flakes when the snow is heavy
    TILE = (3.0, 2.0)  # the haze's drifting pattern: one tile, frame heights along and across the snow's way

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 27))
        v = np.random.default_rng([self.seed, 1103, 2]).random((self.NEAR, 27))
        flow = np.random.default_rng([self.seed, 1103, 3])
        u = np.concatenate([u, v])
        near = np.arange(len(u)) >= self.CAP
        ds = p["depth"] / 100
        z = np.where(near, 1.0 + 0.5 * u[:, 0], u[:, 0] ** 1.5)  # 0 far .. 1 near; above 1: at the lens
        zc = np.minimum(z, 1.0)
        heavy = min(max((p["amount"] - 50) / 50, 0.0), 1.0)  # heavy snow: many more far flakes, a little stronger

        # size, focus and strength (pixels of a 1080-line frame); out of focus (bokeh 0..1): bigger, softer, a
        # glow around it, and at the lens a bright orb rather than a faint smudge
        size_k = (1 - ds) + ds * (0.2 + 2.5 * zc ** 2)
        radius = _lerp(0.9, 4.5, p["size"] / 100) * size_k * (0.7 + 0.6 * u[:, 4])
        radius = np.where(near, _lerp(8, 26, p["size"] / 100) * (0.6 + 0.9 * u[:, 4]), radius)
        bokeh = p["blur"] / 100 * np.clip((zc - 0.8) / 0.2, 0, 1) ** 1.5
        bokeh = np.where(near, 0.5 + 0.5 * p["blur"] / 100, bokeh)
        soft = np.maximum(0.75, (0.25 + 0.25 * bokeh) * radius)
        radius = radius * (1 + 0.3 * bokeh)
        opacity = (p["opacity"] / 100 * _lerp(0.45 + 0.2 * heavy, 1.0, zc ** 0.6) * (0.8 + 0.2 * u[:, 8])
                   * (1 - 0.15 * bokeh))
        opacity = np.where(near, np.minimum(1.0, p["opacity"] / 100 * _lerp(0.7, 1.15, u[:, 8])) * (1 - 0.1 * bokeh),
                           opacity)
        glow = p["glow"] / 100
        mix = _smooth(zc ** 0.7)

        style = p["flake"]
        tip = 2.0 * radius  # how far a crystal's arms reach
        spin = p["spin"] / 100

        # motion (frame heights): the wind's slant, plus the steady drift of the gusts (their bursts come on top)
        speed_k = (1 - ds) + ds * (0.3 + 1.5 * zc ** 1.5)
        speed = _lerp(0.08, 0.8, p["speed"] / 100) * speed_k * (0.85 + 0.3 * u[:, 3])
        speed = speed * np.where(near, _lerp(2.6, 4.0, u[:, 3]), 1.0)  # at the lens they whip past
        angle = math.radians(45) * p["wind"] / 100 + np.radians(5) * (u[:, 9] - 0.5)
        sway = p["sway"] / 100 * _lerp(0.008, 0.05, u[:, 5]) * (0.4 + 0.8 * z)
        self.swirl = p["swirl"] / 100 * 0.07
        swirl = self.swirl * (0.35 + 0.65 * z)
        self.gust_sign = 1.0 if p["wind"] >= 0 else -1.0
        push = p["gust"] / 100 * 0.32 * (0.3 + 0.7 * z)
        self.gm = max(1.0, round(self.loop_s / 9))  # gusts per loop
        drift = self.gust_sign * push * GUST_LEAD * 2 * math.pi * self.gm / self.loop_s
        vx, vy = speed * np.sin(angle) + drift, speed * np.cos(angle)
        angle, speed = np.arctan2(vx, vy), np.hypot(vx, vy)
        streak = self.SHUTTER * speed * self.h / self.fps  # pixels a flake smears over

        # ice crystals, decided in a 1080-line frame: in the crystal style the sharp flakes big enough to show their
        # arms; in the natural style a rare one among the nearest sharp flakes, small and soft (most real snow
        # falls as soft clumps, and many crisp stars read as stickers)
        sharp = (~near) & (bokeh < 0.25) & (streak < 1.6 * tip * px)
        rare = (style == "mixed") & (zc >= 0.7) & (u[:, 13] < 0.1) & (tip >= RARE_TIP[0])
        crystal_ = sharp & (((style == "crystal") & (tip >= 6)) | rare)
        softc = crystal_ & (style == "mixed")
        opacity = np.where(crystal_ & ~softc, np.minimum(1.0, 1.15 * opacity), opacity)  # thin arms: as bright
        # how far a flake's picture reaches from its centre: wherever it may be (whatever its kind and glow) for
        # its path, so those settings leave every flake where it was; as it is, to skip the flakes out of sight
        halo_px = np.where(glow > 0, 2.5 * (1.2 * np.where(crystal_, tip, radius) + 1.5), 0.0)
        reach = np.maximum(np.maximum(np.where(crystal_, tip, radius) + soft, 2 * radius), halo_px) * px + 3
        most = (np.maximum(tip + soft, 2.5 * (1.2 * tip + 1.5)) + 3) * px  # (in a 1080-line frame: alike at any size)
        tip = np.where(softc, np.minimum(tip, RARE_TIP[1]), tip)
        margin_y = (most + streak) / self.h + 0.3 * sway + swirl + 0.01
        margin_x = most / self.h + 1.4 * sway + swirl + 0.5 * push + 0.02
        share = self.paths(u[:, 1], u[:, 2], speed, angle, margin_y, margin_x)

        # how many: about `wanted` on screen at a time (counted with the margins around it, where the flakes
        # also are), the first candidates of each band; drawn far to near. Heavy snow has many more flakes, far
        # ones most (fewer extra ones when they are big: big wet flakes fall as fewer clumps)
        scale = self.aspect / (16 / 9)
        wider = [(1 + 2 * float(margin_x[k].mean()) / self.aspect) * (1 + 2 * float(margin_y[k].mean()))
                 for k in (~near, near)]
        wanted = p["amount"] / 100 * 1100 * scale * wider[0]
        wanted_near = p["near"] / 100 * math.sqrt(p["amount"] / 100) * 28 * scale * wider[1]
        far = np.flatnonzero(z[:self.CAP] < self.FAR)
        mid = np.flatnonzero(z[:self.CAP] >= self.FAR)
        part = float(share[far].sum() / max(share[:self.CAP].sum(), 1e-9))  # the far band's part of them
        more = heavy ** 1.5 * min(max((100 - p["size"]) / 45, 0.0), 1.0)
        pick = np.concatenate([mid[self._first(share[mid], wanted * (1 - part) * (1 + 1.2 * more))],
                               far[self._first(share[far], wanted * part * (1 + 3 * more))],
                               self.CAP + self._first(share[self.CAP:], wanted_near)])
        order = pick[np.argsort(z[pick], kind="stable")]
        for name in ("cos", "sin", "k", "len", "extra", "s0", "xe", "speed", "my"):
            setattr(self, name, getattr(self, name)[order])
        u = u[order]
        self.n = len(order)
        self.z = z[order]
        self.radius, self.soft, self.bokeh = radius[order] * px, soft[order] * px, bokeh[order]
        self.opacity, self.reach, self.mix = opacity[order], reach[order], mix[order]
        self.far_rgb, self.near_rgb = _rgb(p["far_color"]), _rgb(p["color"])
        self.colour = self.far_rgb * (1 - self.mix[:, None]) + self.near_rgb * self.mix[:, None]
        self.near = near[order]
        self.sharp = self.bokeh < 0.3
        self.together = (~self.near) & (self.bokeh < 0.5)  # drawn together (big soft flakes: one by one)
        self.sway, self.swirl_k, self.push = sway[order], swirl[order], push[order]
        self.m1, self.m2 = self.cycles(_lerp(0.12, 0.35, u[:, 6])), self.cycles(_lerp(0.5, 1.0, u[:, 10]))
        self.phi1, self.phi2 = u[:, 7] * 2 * math.pi, u[:, 11] * 2 * math.pi
        self.a2 = _lerp(0.15, 0.4, u[:, 12])

        # the flow: three slow waves of a stream function; the flakes move along its curl (they swirl, never
        # bunch up), far flakes see it smaller, and each depth sees another part of it
        f = flow.random((3, 6))
        turn = f[:, 0] * 2 * math.pi
        wave = 2 * math.pi / _lerp(0.35, 0.9, f[:, 1])
        self.fk = np.stack([wave * np.cos(turn), wave * np.sin(turn)])  # (2, 3)
        self.fdir = np.stack([-np.sin(turn), np.cos(turn)])
        self.fm = np.rint(_lerp(0.03, 0.09, f[:, 2]) * self.loop_s)  # whole cycles per loop (0: a still pattern)
        self.fph = f[:, 3] * 2 * math.pi
        self.famp = np.array([1.0, 0.7, 0.5]) / 1.4
        self.fzoom = _lerp(1.7, 1.0, np.minimum(self.z, 1.0))
        self.fdepth = (self.z * 2.3)[:, None] * (f[:, 4] * 2 * math.pi)[None, :]
        self.gph = float(f[0, 5]) * 2 * math.pi

        # crystals turn (whole turns per loop) and tumble (seen at a slant they look narrower)
        self.crystal, self.softc = crystal_[order], softc[order]
        self.shape = (u[:, 14] * 8).astype(int) % 8
        self.tip = tip[order] * px
        self.rot0 = u[:, 15] * 2 * math.pi
        self.turns = np.sign(u[:, 16] - 0.5) * np.rint(spin * _lerp(0.02, 0.15, u[:, 17]) * self.loop_s)
        self.wobble = spin * 0.5 * u[:, 18]
        self.mw, self.phw = self.cycles(_lerp(0.1, 0.3, u[:, 19])), u[:, 20] * 2 * math.pi
        self.tumble = spin * _lerp(0.15, 0.5, u[:, 21])
        self.mt, self.pht = self.cycles(_lerp(0.08, 0.25, u[:, 22])), u[:, 23] * 2 * math.pi

        # the glow: sharp flakes big enough to light the air around them
        self.halo = np.where(self.sharp & (self.radius >= 0.9 * px), 0.25 * glow, 0.0)
        self.spread = 1.2 * np.where(self.crystal, self.tip, self.radius) + 1.5 * px

        # glints: a share of the sharp flakes catch the light now and then; the slider gives more of them,
        # brighter and with longer rays (a few at a time by default)
        tw = self.tw = p["twinkle"] / 100
        self.glints = np.where((u[:, 24] < 0.03 + 0.4 * tw) & self.sharp & (tw > 0), 0.5 + 0.5 * tw, 0.0)
        self.q = self.cycles(_lerp(0.1, 0.35, u[:, 25]))
        self.qph = u[:, 26] * 2 * math.pi

        # a crystal too small (in this frame) to show its arms is drawn as a round flake with as much ink
        self.ink_r = np.zeros(self.n)
        for i in np.flatnonzero(self.crystal & (self.tip < SMALLEST)):
            ink = crystal_ink(int(self.shape[i]), crystal_size(self.tip[i]), bool(self.softc[i]))
            self.ink_r[i] = self.tip[i] / 0.45 * math.sqrt(ink / math.pi)

        # the haze: snow too far and too thick to see flake by flake. A veil in the far colour (thicker towards the
        # top, where a scene is mostly furthest) in streaky drifts that blow along the snow's way (two patterns, one
        # slower, each a whole number of tiles a loop), swelling as a gust sweeps through; a whiteout at the top of
        # the slider
        self.haze = p["haze"] / 100 * 0.42 if p["amount"] > 0 else 0.0
        if self.haze:
            step = 8 * self.px  # worked out on a coarse grid (smooth anyway), then scaled up
            gw, gh = math.ceil(self.w / step) + 2, math.ceil(self.h / step) + 2
            self.hx = ((np.arange(gw) + 0.5) * step / self.h)[None, :]
            self.hy = ((np.arange(gh) + 0.5) * step / self.h)[:, None]
            self.hbox = (0.0, 0.0, self.w / step, self.h / step)
            body = ~self.near
            way = float(np.median(np.arctan2(self.sin[body], self.cos[body]))) if body.any() else 0.0
            pace = float(np.median(self.speed[body])) if body.any() else 0.1  # frame heights a second
            lu, lv = self.TILE
            self.hu = (self.hx * math.sin(way) + self.hy * math.cos(way)) / lu
            self.hv = (self.hx * math.cos(way) - self.hy * math.sin(way)) / lv
            self.hfields = [drifts(flow, 64, 2.0, 3.0), drifts(flow, 96, 3.0, 7.0)]
            self.hm = [max(1.0, round(k * pace * self.loop_s / lu)) for k in (0.5, 1.0)]
            self.htop = _lerp(1.0, 0.65, np.clip(self.hy, 0, 1))
            self.hgust = 0.9 * p["gust"] / 100

        self._acc = None  # the flakes in focus, added up (made when first needed, kept empty between frames)
        self.halos = sprites.SpriteCache(halo)
        self.sparks = sprites.SpriteCache(lambda key, fx, fy: glint(key, fx, fy))

    def veil(self, t):
        """The haze at frame t: coverage over the whole frame."""
        tau = self.tau(t)
        a = sample(self.hfields[0], self.hu - self.hm[0] * tau, self.hv)
        b = sample(self.hfields[1], self.hu - self.hm[1] * tau, self.hv + 0.37)
        d = np.exp(0.3 * a + 0.22 * b - 0.07)  # (about 1 on average, never below nothing)
        if self.hgust:  # the same bursts as the flakes' gusts, sweeping across from the windward side
            theta = 2 * math.pi * self.gm * tau + self.gph - 2 * math.pi * 0.35 * self.gust_sign * self.hx / self.aspect
            d = d + self.hgust * (np.exp(4 * (np.cos(theta) - 1)) - 0.2)
        d = np.minimum(self.haze * self.htop * np.maximum(d, 0.0), 0.9).astype(np.float32)
        big = Image.fromarray(d, "F").resize((self.w, self.h), Image.Resampling.BILINEAR, box=self.hbox)
        return np.asarray(big, np.float32)

    @staticmethod
    def _first(share, wanted):
        """The first candidates that make about `wanted` on screen at a time."""
        n = int(np.searchsorted(np.cumsum(share), wanted)) + 1 if wanted > 0 else 0
        return np.arange(min(n, len(share)))

    def at(self, t, dt=0.0):
        """x, y (pixels) of every flake at frame t + dt, and whether it is on its visible stretch at frame t. Along
        its path a flake is followed from where it is at t (not started again at the top during dt)."""
        s = np.mod(self.s0 + self.k * self.len * self.tau(t), self.len)
        on = s >= self.extra
        d = s + self.k * self.len * (dt / self.P) - self.extra
        x, y = self.xe + d * self.sin, -self.my + d * self.cos
        tau = self.tau(t + dt)
        sway = 2 * math.pi * self.m1 * tau + self.phi1
        dx = self.sway * (np.sin(sway) + self.a2 * np.sin(2 * math.pi * self.m2 * tau + self.phi2))
        dy = 0.18 * self.sway * np.sin(2 * sway)  # a flake falls slower at the ends of its swing
        if self.swirl > 0:
            zx, zy = x * self.fzoom, y * self.fzoom
            phase = (zx[:, None] * self.fk[0] + zy[:, None] * self.fk[1] + 2 * math.pi * self.fm * tau + self.fph
                     + self.fdepth)
            c = np.cos(phase) * self.famp
            dx = dx + self.swirl_k * (c @ self.fdir[0])
            dy = dy + self.swirl_k * (c @ self.fdir[1])
        if self.push.any():  # a gust sweeps across the frame from the side the wind comes from
            theta = 2 * math.pi * self.gm * tau + self.gph - 2 * math.pi * 0.35 * self.gust_sign * x / self.aspect
            dx = dx + self.gust_sign * self.push * gust_push(theta)
        return (x + dx) * self.h, (y + dy) * self.h, on

    def draw(self, t, canvas, mask):
        veil = self.veil(t) if self.haze else None
        if not self.n:
            if veil is not None:
                self._lay(canvas, mask, veil)
            return
        x, y, on = self.at(t)
        xb, yb, _ = self.at(t, -self.SHUTTER / 2)
        xa, ya, _ = self.at(t, self.SHUTTER / 2)
        mx, my = xa - xb, ya - yb  # pixels moved while the shutter is open
        tau = self.tau(t)
        spark = self.glints * np.maximum(np.sin(2 * math.pi * self.q * tau + self.qph), 0.0) ** self.SHARP
        opacity = np.minimum(1.0, self.opacity * (1 + 1.5 * spark))
        turn = self.rot0 + 2 * math.pi * self.turns * tau + self.wobble * np.sin(2 * math.pi * self.mw * tau + self.phw)
        squash = 1 - self.tumble * (0.5 - 0.5 * np.cos(2 * math.pi * self.mt * tau + self.pht))
        reach = self.reach + 0.5 * np.hypot(mx, my)
        seen = on & (x > -reach) & (x < self.w + reach) & (y > -reach) & (y < self.h + reach) & (opacity > 0.002)
        round_ = seen & ~(self.crystal & (self.ink_r == 0))
        r = np.where(self.ink_r > 0, self.ink_r, self.radius)
        lit = seen & (spark > 0.02) & (self.radius >= 0.6 * self.px)  # glinting now
        ray = np.minimum(self.tip * (1.8 + 2.0 * self.tw) + 3 * self.px, 40 * self.px)
        # most flakes, with their glow and glints, are worked out together and laid over the frame in one go
        # (behind the others); those at the lens or far out of focus and the crystals one by one, far to near
        group = round_ & self.together
        dots = self._together(canvas, mask, veil, np.flatnonzero(round_), group, x, y, r, mx, my, opacity,
                              np.flatnonzero(group & lit), spark, ray)
        for i in np.flatnonzero(seen & ~group):
            if self.halo[i] > 0:  # the glow first, the flake over it
                ix, iy = math.floor(x[i]), math.floor(y[i])
                cover, x0, y0 = self.halos.get(sprites.bucket(self.spread[i], 1.1), x[i] - ix, y[i] - iy)
                canvas.over(ix + x0, iy + y0, cover, self.colour[i], self.halo[i] * opacity[i], mask)
            if i in dots:
                cover, x0, y0 = dots[i]
                canvas.over(x0, y0, cover, self.colour[i], opacity[i], mask)
            else:
                self._crystal(i, x[i], y[i], turn[i], squash[i], mx[i], my[i], opacity[i], canvas, mask)
            if lit[i]:
                ix, iy = math.floor(x[i]), math.floor(y[i])
                cover, x0, y0 = self.sparks.get(sprites.bucket(ray[i]), x[i] - ix, y[i] - iy)
                canvas.over(ix + x0, iy + y0, cover, self.colour[i], min(1.0, 1.2 * self.opacity[i]) * spark[i],
                            mask)

    def _add(self, cover, idx, x, y, gx, gy, weight):
        """Adds coverages `cover` (n, 2 gy + 1, 2 gx + 1) of flakes `idx` at x, y, times `weight`, to the sums."""
        w, h = self.w, self.h
        ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
        cols = (ix - gx)[:, None, None] + np.arange(2 * gx + 1)[None, None, :]
        rows = (iy - gy)[:, None, None] + np.arange(2 * gy + 1)[None, :, None]
        ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
        at = (rows * w + cols)[ok]
        wt = (cover * np.asarray(weight, np.float32)[:, None, None])[ok]
        np.add.at(self._acc[0], at, wt)
        np.add.at(self._acc[1], at, wt * np.broadcast_to(self.mix[idx].astype(np.float32)[:, None, None],
                                                         ok.shape)[ok])

    def _together(self, canvas, mask, veil, idx, group, x, y, r, mx, my, opacity, lit, spark, ray):
        """The round flakes `idx` (a crystal too small to show its arms is one, with as much ink), worked out
        together, a group of flakes on the same grid size at a time. Those in `group` (with their glow, and the
        glints of those in `lit`) are added up and laid over the frame at once (where they overlap, they cover it),
        over the haze `veil` if any; returns the others, {flake: (coverage, x0, y0)}."""
        if self._acc is None:
            self._acc = np.zeros((2, self.w * self.h), np.float32)
        out = {}
        if len(idx):
            rr = r[idx]
            soft, bokeh, dx, dy = np.maximum(self.soft[idx], 0.7), self.bokeh[idx], mx[idx], my[idx]
            reach = np.maximum(rr + soft / 2, np.where(bokeh > 0, 2.2 * GLOW_SPREAD * rr, 0.0)) + 1
            gx, gy = _grid_of(reach + np.abs(dx) / 2), _grid_of(reach + np.abs(dy) / 2)
            key = gx * 100000 + gy
            for k in np.unique(key):
                g = np.flatnonzero(key == k)
                hx, hy = int(gx[g[0]]), int(gy[g[0]])
                f = idx[g]
                cover = flakes(x[f], y[f], rr[g], soft[g], bokeh[g], dx[g], dy[g], hx, hy)
                add = group[f]
                for j in np.flatnonzero(~add):
                    out[int(f[j])] = (cover[j], math.floor(x[f[j]]) - hx, math.floor(y[f[j]]) - hy)
                if add.any():
                    f = f[add]
                    self._add(cover[add], f, x[f], y[f], hx, hy, opacity[f])
        glow = idx[group[idx] & (self.halo[idx] > 0)]
        if len(glow):  # the glow around them
            gx = _grid_of(2.5 * self.spread[glow])
            for k in np.unique(gx):
                f = glow[gx == k]
                self._add(halos(x[f], y[f], self.spread[f], k, k), f, x[f], y[f], k, k, self.halo[f] * opacity[f])
        if len(lit):  # glints
            gx = _grid_of(ray[lit] + 1.5)
            for k in np.unique(gx):
                f = lit[gx == k]
                self._add(glints(x[f], y[f], ray[f], k, k), f, x[f], y[f], k, k,
                          np.minimum(1.0, 1.2 * self.opacity[f]) * spark[f])
        self._lay(canvas, mask, veil)
        return out

    def _lay(self, canvas, mask, veil=None):
        """Lays the sums (covered more than once: just covered) over the haze `veil` if any, over the frame, and
        empties them: the pixels they touched, or the whole frame at once (with the haze, or when they touched much
        of it)."""
        if self._acc is None:
            self._acc = np.zeros((2, self.w * self.h), np.float32)
        acc, far, near = self._acc, self.far_rgb, self.near_rgb - self.far_rgb
        ids = None if veil is not None else np.flatnonzero(acc[0])
        if ids is None or len(ids) > acc.shape[1] // 8:
            total = acc[0].reshape(self.h, self.w)
            alpha = np.minimum(total, 1.0)
            mixed = acc[1].reshape(self.h, self.w) / np.maximum(total, 1.0)
            if veil is not None:  # the haze (in the far colour) behind the flakes
                alpha *= 1.0 - veil
                alpha += veil
            if mask is not None:
                alpha *= mask
                mixed *= mask
            acc[:] = 0.0
            empty = canvas.box is None  # (nothing drawn yet: nothing to cover)
            canvas._place(0, 0, self.h, self.w)
            inv = 1.0 - alpha
            for c in range(3):  # (a channel at a time: quicker)
                v = canvas.rgb[..., c]
                if empty:
                    np.multiply(alpha, far[c], out=v)
                else:
                    v *= inv
                    v += alpha * far[c]
                v += mixed * near[c]
            if empty:
                canvas.a[:] = alpha
            else:
                canvas.a *= inv
                canvas.a += alpha
        elif len(ids):
            total, mixed = acc[0][ids], acc[1][ids]
            acc[0][ids] = 0.0
            acc[1][ids] = 0.0
            k = 1.0 / np.maximum(total, 1.0)
            cover, mixed = total * k, mixed * k
            lay(canvas, ids, cover, cover[:, None] * far + mixed[:, None] * near, mask)

    def _crystal(self, i, x, y, turn, squash, mx, my, opacity, canvas, mask):
        tip = self.tip[i]
        size = crystal_size(tip)
        scale = tip / (0.45 * size)
        half = 0.5 * size * scale + 1.5
        x0, y0 = math.floor(x - half), math.floor(y - half)
        w, h = math.ceil(x + half) - x0, math.ceil(y + half) - y0
        ca, sa = math.cos(turn), math.sin(turn)
        sx, sy = scale, scale * max(squash, 0.2)
        ox, oy = x0 - x, y0 - y
        data = (ca / sx, sa / sx, (ca * ox + sa * oy) / sx + size / 2,
                -sa / sy, ca / sy, (-sa * ox + ca * oy) / sy + size / 2)
        img = crystal(int(self.shape[i]), size, bool(self.softc[i])).transform(
            (w, h), Image.Transform.AFFINE, data, resample=Image.Resampling.BILINEAR)
        cover = np.asarray(img, np.float32) * (1 / 255.0)
        mx, my = mx * self.CRISP, my * self.CRISP
        if math.hypot(mx, my) >= 0.7:
            cover, dx0, dy0 = smear(cover, mx, my)
            x0, y0 = x0 + dx0, y0 + dy0
        canvas.over(x0, y0, cover, self.colour[i], opacity, mask)
