"""Particle images, all made by code: soft dots, streaks and glows (fireflies, dust). (Leaves: leaves.py.)

Dots, streaks and glows are float32 coverage arrays drawn at a sub-pixel offset: a particle at x = 103.4 is drawn
from pixel 103 with fx = 0.4, so slow particles glide instead of stepping a whole pixel at a time. Each function
returns (array, x0, y0): where the array's top-left pixel sits relative to the particle's own pixel.
"""
import math

import numpy as np

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
