"""Falling rain (see effects.py for how every effect loops)."""
import math

import numpy as np

from . import sprites
from .effects import Falling, _colour, _lerp, _num, _rgb

SCHEMA = {"label": "Mưa", "params": [
    _num("amount", "Mật độ", 45), _num("speed", "Tốc độ rơi", 60), _num("length", "Độ dài vệt", 50),
    _num("size", "Độ dày hạt", 35), _num("wind", "Gió (nghiêng)", 10, -100, 100),
    _num("depth", "Chiều sâu", 50), _num("opacity", "Độ đậm", 45), _colour("color", "Màu", "#d4dce6")]}


class Rain(Falling):
    CAP = 6000

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 8))
        z = u[:, 0]
        ds = p["depth"] / 100
        speed = _lerp(1.4, 4.2, p["speed"] / 100) * ((1 - ds) + ds * (0.55 + 0.9 * z)) * (0.9 + 0.2 * u[:, 3])
        self.angle = math.radians(30) * p["wind"] / 100
        angle = self.angle + np.radians(3) * (u[:, 7] - 0.5)
        width = _lerp(0.6, 1.8, p["size"] / 100) * ((1 - ds) + ds * (0.5 + z)) * px
        shutter = _lerp(0.25, 1.1, p["length"] / 100)
        longest = shutter * float(speed.max()) * 1.3 / self.fps
        share = self.paths(u[:, 1], u[:, 2], speed, angle, longest + 0.02, longest + 0.04)
        wanted = p["amount"] / 100 * 1400 * self.aspect / (16 / 9)
        opacity = np.minimum(1.0, p["opacity"] / 100 * _lerp(0.25, 0.85, z) * (0.75 + 0.5 * u[:, 5]))
        kept = self.choose(share, wanted, {"z": z, "width": width, "opacity": opacity})
        self.__dict__.update(kept)
        self.gusts(0.006 * abs(p["wind"]) / 100)
        length = shutter * self.speed * self.h / self.fps
        self.keys = [(sprites.bucket(ln, 1.1, 1.0), sprites.bucket(wd, 1.1, 0.3)) for ln, wd in zip(length, self.width)]
        self.colour = _rgb(p["color"])
        self.cache = sprites.SpriteCache(lambda key, fx, fy: sprites.streak(key[0], key[1], self.angle, fx, fy))

    def draw(self, t, canvas, mask):
        x, y, on = self.where(t)
        x = x + self.gust_at(self.tau(t)) * (0.3 + 0.7 * self.z)
        xs, ys = x * self.h, y * self.h
        for i in np.flatnonzero(on & (xs > -100) & (xs < self.w + 100)):
            ix, iy = math.floor(xs[i]), math.floor(ys[i])
            cover, x0, y0 = self.cache.get(self.keys[i], xs[i] - ix, ys[i] - iy)
            canvas.over(ix + x0, iy + y0, cover, self.colour, self.opacity[i], mask)
