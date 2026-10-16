"""A candle flame (a placed lens effect: Effect.BOX, apply()). A starter: a flickering glow in its box."""
import math

import numpy as np

from .effects import Canvas, Effect, _colour, _num, _place, _rgb

SCHEMA = {"label": "Lửa nến", "params": [
    *_place(8), _num("glow", "Quầng sáng", 50, group="Ánh sáng"), _colour("color", "Màu lửa", "#ffb35a", group="Màu")]}


class Candle(Effect):
    BOX = (1, 2.5)
    blend = "screen"
    lens = True

    def draw(self, t, canvas, mask):
        x0, y0, w, h = self.box()
        r = max(1.0, 0.4 * w)
        n = int(2 * r) + 3
        yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
        level = 0.6 + 0.4 * math.sin(2 * math.pi * 7 * self.tau(t))
        glow = np.exp(-((xx - n / 2) ** 2 + (yy - n / 2) ** 2) / (2 * (r / 2) ** 2)).astype(np.float32)
        canvas.light(int(x0 + w / 2 - n / 2), int(y0 + h * 0.6 - n / 2), glow * level * self.p["glow"] / 100,
                     _rgb(self.p["color"]), mask)

    def apply(self, t, picture, mask, gain):
        if gain <= 0:
            return
        canvas = Canvas(self.w, self.h)
        before = picture.copy()
        self.draw(t, canvas, None)
        canvas.flush(picture, self.blend)
        k = gain if mask is None else gain * mask[..., None]
        picture[:] = before + (picture - before) * k
