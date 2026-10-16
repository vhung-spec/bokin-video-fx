"""Steam rising from a cup (a placed effect: Effect.BOX). A starter: a soft puff rising in its box."""
import math

import numpy as np

from .effects import Effect, _colour, _num, _place, _rgb

SCHEMA = {"label": "Khói cốc cà phê", "params": [
    *_place(12), _num("opacity", "Độ đậm", 60, group="Màu"), _colour("color", "Màu khói", "#f2efe9", group="Màu")]}


class Steam(Effect):
    BOX = (1, 1.6)

    def draw(self, t, canvas, mask):
        x0, y0, w, h = self.box()
        r = max(1.0, 0.18 * w)
        rise = (self.tau(t) * 3) % 1.0  # three puffs a loop
        cx, cy = x0 + w / 2, y0 + h - r - rise * (h - 2 * r)
        n = int(2 * r) + 3
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        cover = np.exp(-((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * (r / 2) ** 2)) * math.sin(math.pi * rise)
        canvas.over(int(cx - n / 2), int(cy - n / 2), cover.astype(np.float32), _rgb(self.p["color"]),
                    self.p["opacity"] / 100, mask)
