"""Falling snow (see effects.py for how every effect loops)."""
import math

import numpy as np

from . import sprites
from .effects import Falling, _colour, _lerp, _num, _rgb

SCHEMA = {"label": "Tuyết rơi", "params": [
    _num("amount", "Mật độ", 45), _num("size", "Cỡ bông", 40), _num("speed", "Tốc độ rơi", 35),
    _num("wind", "Gió (trái ← → phải)", 10, -100, 100), _num("sway", "Đung đưa", 40),
    _num("depth", "Chiều sâu (xa nhỏ, gần to)", 60), _num("blur", "Nhoè bông ở gần", 40),
    _num("opacity", "Độ đậm", 85), _colour("color", "Màu", "#ffffff")]}


class Snow(Falling):
    CAP = 4000

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 10))
        z = u[:, 0] ** 1.6
        ds = p["depth"] / 100
        size_k = (1 - ds) + ds * (0.3 + 2.2 * z ** 1.5)
        speed_k = (1 - ds) + ds * (0.45 + 1.3 * z)
        radius = _lerp(0.9, 4.2, p["size"] / 100) * size_k * (0.75 + 0.5 * u[:, 4])
        defocus = p["blur"] / 100 * np.clip((z - 0.55) / 0.45, 0, 1) ** 1.5
        soft = np.maximum(0.75, 0.3 * radius) + defocus * radius * 2.5
        radius = radius * (1 + 0.8 * defocus)
        opacity = p["opacity"] / 100 * _lerp(0.5, 0.95, z) * (0.8 + 0.2 * u[:, 8]) * (1 - 0.5 * defocus)
        self.angle = math.radians(45) * p["wind"] / 100
        angle = self.angle + np.radians(4) * (u[:, 9] - 0.5)
        sway = p["sway"] / 100 * _lerp(0.004, 0.025, u[:, 5]) * (0.5 + 0.7 * z)
        speed = _lerp(0.12, 0.85, p["speed"] / 100) * speed_k * (0.85 + 0.3 * u[:, 3])
        reach_px = float((radius + soft).max()) * px + 2
        margin_y = reach_px / self.h + 0.25 * float(speed.max()) / self.fps
        share = self.paths(u[:, 1], u[:, 2], speed, angle, margin_y, margin_y + float(sway.max()) + 0.03)
        wanted = p["amount"] / 100 * 900 * self.aspect / (16 / 9)
        kept = self.choose(share, wanted, {"z": z, "radius": radius, "soft": soft, "opacity": opacity,
                                           "sway": sway, "m": self.cycles(_lerp(0.15, 0.5, u[:, 6])),
                                           "phi": u[:, 7] * 2 * math.pi})
        self.__dict__.update(kept)
        self.gusts(0.02 * abs(p["wind"]) / 100 + 0.012 * p["sway"] / 100)
        streak = 0.5 * self.speed * self.h / self.fps  # a 180-degree shutter: half a frame of motion
        self.keys = [(sprites.bucket(r * px), sprites.bucket(s * px), sprites.bucket(st, floor=0.5))
                     for r, s, st in zip(self.radius, self.soft, streak)]
        self.colour = _rgb(p["color"])
        self.cache = sprites.SpriteCache(lambda key, fx, fy: sprites.dot(key[0], key[1], key[2], self.angle, fx, fy))

    def draw(self, t, canvas, mask):
        x, y, on = self.where(t)
        tau = self.tau(t)
        x = x + self.sway * np.sin(2 * math.pi * self.m * tau + self.phi) + self.gust_at(tau) * (0.3 + 0.7 * self.z)
        xs, ys = x * self.h, y * self.h
        for i in np.flatnonzero(on & (xs > -60) & (xs < self.w + 60)):
            ix, iy = math.floor(xs[i]), math.floor(ys[i])
            cover, x0, y0 = self.cache.get(self.keys[i], xs[i] - ix, ys[i] - iy)
            canvas.over(ix + x0, iy + y0, cover, self.colour, self.opacity[i], mask)
