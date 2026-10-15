"""Particle images, all made by code: soft dots (snow), streaks (rain), glows (fireflies, dust) and leaves.

Dots, streaks and glows are float32 coverage arrays drawn at a sub-pixel offset: a particle at x = 103.4 is drawn
from pixel 103 with fx = 0.4, so slow particles glide instead of stepping a whole pixel at a time. Each function
returns (array, x0, y0): where the array's top-left pixel sits relative to the particle's own pixel.

Leaves are premultiplied RGBA images (Pillow mode "RGBa") at several sizes; the leaf effect turns and squashes
them every frame.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PHASES = 4  # cached sprites: sub-pixel positions per pixel and axis


def _grid(hx, hy, fx, fy):
    x0, y0 = -int(math.ceil(hx)), -int(math.ceil(hy))
    xs = np.arange(x0, -x0 + 1, dtype=np.float32) + 0.5 - fx
    ys = np.arange(y0, -y0 + 1, dtype=np.float32) + 0.5 - fy
    return x0, y0, xs[None, :], ys[:, None]


def dot(radius, soft, streak, angle, fx, fy):
    """A round dot: solid inside `radius`, fading over `soft` px, smeared over `streak` px along the direction of
    motion `angle` (radians from straight down, positive to the right), as a camera shutter would."""
    soft = max(soft, 0.7)
    dx, dy = math.sin(angle), math.cos(angle)
    reach = radius + soft / 2 + 1
    x0, y0, xs, ys = _grid(abs(dx) * streak / 2 + reach, abs(dy) * streak / 2 + reach, fx, fy)
    n = 1 if streak < 0.5 else min(32, int(math.ceil(streak / max(0.5, min(radius, 1.5)))) + 1)
    acc = np.zeros((ys.shape[0], xs.shape[1]), np.float32)
    for i in range(n):
        o = 0.0 if n == 1 else (i / (n - 1) - 0.5) * streak
        a = np.clip((radius - np.sqrt((xs - o * dx) ** 2 + (ys - o * dy) ** 2)) / soft + 0.5, 0.0, 1.0)
        acc += a * a * (3 - 2 * a)
    acc /= n
    area = math.pi * radius * radius + 2 * radius * streak  # a dot smaller than a pixel only covers part of it
    total = float(acc.sum())
    if total > area > 0:
        acc *= area / total
    return acc, x0, y0


def streak(length, width, angle, fx, fy):
    """A rain streak: its head at the particle, its tail `length` px behind it (against the motion `angle`),
    brighter at the head. A streak thinner than a pixel covers that share of it."""
    dx, dy = math.sin(angle), math.cos(angle)
    reach = max(width, 1.0) / 2 + 1.5
    x0, y0, xs, ys = _grid(abs(dx) * length + reach, abs(dy) * length + reach, fx, fy)
    along = xs * dx + ys * dy  # 0 at the head, -length at the tail
    across = np.abs(-xs * dy + ys * dx)
    w = max(width, 1.0)
    profile = np.clip(w / 2 + 0.5 - across, 0.0, 1.0) * min(1.0, width)
    t = np.clip(-along / max(length, 1e-3), 0.0, 1.0)  # 0 head .. 1 tail
    beyond = np.where(along > 0, along, np.where(along < -length, -length - along, 0.0))
    ends = np.clip(1.0 - beyond, 0.0, 1.0)
    return (profile * ends * (1.0 - 0.75 * t)).astype(np.float32), x0, y0


def glow(radius, fx, fy):
    """A point of light: a bright core and a wide soft halo, peak 1."""
    core, halo = max(0.5, 0.5 * radius), 2.4 * max(radius, 0.6)
    reach = 3 * halo + 1
    x0, y0, xs, ys = _grid(reach, reach, fx, fy)
    d2 = xs * xs + ys * ys
    out = (np.exp(-d2 / (2 * core * core)) + 0.45 * np.exp(-d2 / (2 * halo * halo))) / 1.45
    target = 2 * math.pi * (core * core + 0.45 * halo * halo) / 1.45  # what the curve holds when well sampled
    total = float(out.sum())
    if total > target > 0:
        out *= target / total
    return out.astype(np.float32), x0, y0


class SpriteCache:
    """Sprites by (shape key, sub-pixel phase), made on first use."""

    def __init__(self, make):
        self.make, self.items = make, {}

    def get(self, key, fx, fy):
        qx, qy = min(int(fx * PHASES), PHASES - 1), min(int(fy * PHASES), PHASES - 1)
        item = self.items.get((key, qx, qy))
        if item is None:
            item = self.items[(key, qx, qy)] = self.make(key, (qx + 0.5) / PHASES, (qy + 0.5) / PHASES)
        return item


def bucket(value, step=1.06, floor=0.25):
    """Rounds a size on a log scale (6 % steps) so particles of nearly the same size share a sprite."""
    if value <= floor:
        return floor
    return round(step ** round(math.log(value) / math.log(step)), 4)


# --- leaves -------------------------------------------------------------------------------------------------

LEAF_PALETTES = {
    # (inner colour, edge colour) pairs
    "autumn": [((236, 170, 52), (196, 74, 30)), ((226, 128, 40), (170, 52, 32)), ((214, 160, 60), (150, 92, 40)),
               ((240, 190, 70), (210, 110, 40)), ((200, 80, 40), (140, 40, 30)), ((190, 140, 70), (120, 80, 40))],
    "green": [((132, 170, 70), (70, 112, 40)), ((150, 178, 80), (92, 130, 50)), ((110, 150, 60), (60, 96, 36)),
              ((170, 190, 90), (110, 140, 60))],
    "petal": [((253, 232, 238), (242, 176, 200)), ((252, 222, 232), (236, 160, 188)),
              ((255, 240, 244), (246, 196, 214))],
}
LEAF_SHAPES = {"autumn": ["oval", "maple", "round", "narrow", "maple", "oval"],
               "green": ["oval", "narrow", "round", "oval"],
               "petal": ["petal"]}
SS = 4  # supersampling of the outlines


def _outline(shape, s, rng):
    """Polygon of a leaf in an s x s box: stem at the bottom centre, tip at the top."""
    c = s / 2
    if shape == "maple":
        pts = []
        for k in range(720):
            phi = math.radians(k / 2)
            tips = 1 - abs(math.sin(2.5 * (phi + math.pi / 2)))  # five pointed lobes, the middle one up
            low = math.exp(-((phi - math.pi / 2) / 0.95) ** 2)  # the two lobes next to the stem are small
            r = 0.27 + (0.2 - 0.09 * low) * tips ** 1.3 + 0.012 * abs(math.sin(24 * phi))
            r *= 1 - 0.5 * math.exp(-((phi - math.pi / 2) / 0.2) ** 2)  # the notch at the stem
            pts.append((c + r * s * math.cos(phi), 0.5 * s + r * s * math.sin(phi)))
        return pts
    width, a, b = {"oval": (0.5, 0.75, 0.85), "narrow": (0.22, 1.0, 1.0), "round": (0.72, 0.6, 0.7),
                   "petal": (0.62, 0.55, 0.55)}[shape]
    bend = rng.uniform(-0.04, 0.04)
    teeth = 0.0 if shape == "petal" else rng.uniform(0.0, 0.03)
    left, right = [], []
    for k in range(121):
        u = k / 120
        y = s * (0.95 - 0.9 * u)
        hw = s * width / 2 * math.sin(math.pi * u ** a) ** b * (1 + teeth * math.sin(2 * math.pi * 14 * u))
        x = c + s * bend * math.sin(math.pi * u)
        left.append((x - hw, y))
        right.append((x + hw, y))
    return left + right[::-1]


def _smooth_noise(rng, size, cells=6):
    small = Image.fromarray((rng.random((cells, cells)) * 255).astype(np.uint8))
    return np.asarray(small.resize((size, size), Image.Resampling.BICUBIC), np.float32) / 255.0


class Leaf:
    """One leaf: front and back as premultiplied RGBA images at 256, 128, 64 and 32 px."""

    SIZES = (256, 128, 64, 32)

    def __init__(self, style, index, seed):
        rng = np.random.default_rng([seed, index, 7])
        s = self.SIZES[0]
        shapes = LEAF_SHAPES[style]
        shape = shapes[index % len(shapes)]
        inner, edge = LEAF_PALETTES[style][int(rng.integers(len(LEAF_PALETTES[style])))]
        jitter = rng.uniform(0.92, 1.06) * rng.uniform(0.97, 1.03, 3)  # brightness, and a little hue
        inner = np.clip(np.array(inner) * jitter, 0, 255) / 255.0
        edge = np.clip(np.array(edge) * jitter, 0, 255) / 255.0

        big = Image.new("L", (s * SS, s * SS), 0)
        ImageDraw.Draw(big).polygon([(x * SS, y * SS) for x, y in _outline(shape, s, rng)], fill=255)
        if shape == "petal":  # the small notch at the tip of a cherry petal
            ImageDraw.Draw(big).polygon([((s / 2 - 0.06 * s) * SS, 0.0), ((s / 2 + 0.06 * s) * SS, 0.0),
                                         (s / 2 * SS, 0.14 * s * SS)], fill=0)
        veins = Image.new("L", (s * SS, s * SS), 0)
        if shape != "petal":
            d = ImageDraw.Draw(veins)
            d.line([(s / 2 * SS, 0.98 * s * SS), (s / 2 * SS, 0.1 * s * SS)], fill=255, width=int(0.018 * s * SS))
            for k in range(1, 6):
                y = (0.9 - 0.14 * k) * s
                for side in (-1, 1):
                    d.line([(s / 2 * SS, y * SS), ((s / 2 + side * 0.22 * s) * SS, (y - 0.12 * s) * SS)],
                           fill=150, width=int(0.008 * s * SS))
        alpha = np.asarray(big.resize((s, s), Image.Resampling.LANCZOS), np.float32) / 255.0
        vein = np.asarray(veins.resize((s, s), Image.Resampling.LANCZOS), np.float32) / 255.0 * alpha
        # how far inside the leaf a pixel is: colour goes from the inner colour to the edge colour
        depth = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(s * 0.07)),
                           np.float32) / 255.0
        if shape == "petal":  # deeper pink at the base, paler towards the tip
            inside = np.clip((0.95 - np.arange(s, dtype=np.float32) / s) / 0.6, 0.0, 1.0)[:, None, None] \
                * np.ones((1, s, 1), np.float32)
            mottle = 0.04
        else:
            inside = np.clip((depth - 0.35) * 2.2, 0.0, 1.0)[..., None]
            mottle = 0.16
        rim = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2)),
                         np.float32) / 255.0
        shade = (1 - mottle * 0.75 + mottle * _smooth_noise(rng, s))[..., None] * (0.8 + 0.2 * rim[..., None])
        front = (edge + (inner - edge) * inside) * shade
        front = front * (1 - 0.35 * vein[..., None]) + np.clip(inner * 1.25, 0, 1) * 0.35 * vein[..., None]
        grey = front.mean(axis=2, keepdims=True)
        back = np.clip((front + (grey - front) * 0.3) * 1.1, 0, 1)[:, ::-1]
        back = back * (1 - 0.45 * vein[:, ::-1, None]) + 0.92 * 0.45 * vein[:, ::-1, None]
        self.front = self._mips(front, alpha)
        self.back = self._mips(back, alpha[:, ::-1])
        # the stem end sits at the bottom centre; the leaf effect turns the image around its centre

    def _mips(self, rgb, alpha):
        a = alpha[..., None]
        img = Image.fromarray((np.dstack([np.clip(rgb, 0, 1) * a, a]) * 255 + 0.5).astype(np.uint8), "RGBa")
        return [img if size == self.SIZES[0] else img.resize((size, size), Image.Resampling.LANCZOS)
                for size in self.SIZES]

    def image(self, back, length):
        """The smallest of the ready sizes that is still at least `length` px (the largest if none is)."""
        mips = self.back if back else self.front
        for img in reversed(mips):
            if img.size[0] >= length:
                return img
        return mips[0]
