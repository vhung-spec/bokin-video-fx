"""Fireflies ("Đom đóm"): small lights that wander slowly round flower bushes and flash now and then, as fireflies on a
summer evening (a placed effect: Effect.BOX; drawn as light, laid with Screen; see effects.py for how every effect
loops).

The box goes over the bushes (the flowers, the grass, a little air above them): the fireflies live in it, each round
a place of its own, wandering slowly in loops and bobbing (sums of sines, whole cycles a loop); each flashes on its
own beat as a firefly does: its light comes up quickly, stays a moment, and fades more slowly, dark or faintly lit
between flashes ("Nhấp nháy"). Near ones are bigger and softer (out of focus, as a camera sees them), far ones
small points; a hot, paler core and a halo of their colour. Toward the box's edges they fade out: nothing is cut.

Inside the box every length is in box heights (as the string lights'), so the preview clip (the box alone) and the
video show the same.
"""
import math

import numpy as np

from . import sprites
from .effects import Effect, _choice, _colour, _lerp, _num, _place, _rgb
from .lights import _core, _halo

TAU = 2 * math.pi
CAP = 160  # fireflies at most (the first ones: the same random draws, however many there are)
EDGE = 0.07  # they fade out within this much of the box's edges (box heights)
SOFT = 0.04  # and their light within this much (none beyond the box: its preview clip is the box alone)
LIFT = 0.012  # how far a firefly rises as it flashes (box heights)
RISE, HOLD, FADE = 0.12, 0.1, 0.32  # a flash: seconds its light takes to come up, stays, and takes to fade (by e)
HALO, CORE, BLOOM = 0.9, 2.6, 0.12  # light at the middle of its halo, its core, its wide bloom (times its light)
TONES = {"lime": "#c8ff4f", "warm": "#ffe066", "green": "#8dff6a"}

DEFAULTS = {"amount": 35, "size": 45, "brightness": 70, "twinkle": 65, "blink": 50, "wander": 50, "speed": 45,
            "depth": 45}
LOOKS = {
    "garden": {},
    "many": {"amount": 80, "size": 38, "twinkle": 65, "wander": 45, "depth": 55},
    "few": {"amount": 12, "size": 50, "brightness": 80, "twinkle": 85, "wander": 60},
    "dreamy": {"amount": 30, "size": 70, "brightness": 60, "twinkle": 45, "speed": 30, "depth": 90},
}

SCHEMA = {
    "label": "Đom đóm",
    "note": "Đom đóm bay lượn, nháy sáng quanh khóm hoa. (1) Bấm Suốt video. (2) Kéo khung trắng tới khóm hoa, kéo góc "
            "cho khung phủ khóm hoa và khoảng trời phía trên: đom đóm chỉ bay trong khung, gần mép khung thì mờ dần. "
            "Nhiều khóm hoa: một khung lớn phủ hết, hoặc mỗi khóm một clip (Ctrl+C, Ctrl+V rồi kéo sang khóm khác). "
            "(3) Muốn đom đóm bay ra sau lá, cành phía trước: tô lá, cành đó vào Tiền cảnh rồi bật Bay ra sau tiền "
            "cảnh cho clip này. Đẹp nhất trong cảnh chiều tối, ban đêm.",
    "params": [
        _choice("look", "Kiểu", "garden", [("garden", "Đom đóm trong vườn"), ("many", "Nhiều đom đóm (mùa đom đóm)"),
                                           ("few", "Vài con lác đác"), ("dreamy", "Mơ màng (to, nhoè)")],
                presets={k: {**DEFAULTS, **v} for k, v in LOOKS.items()}, group="Kiểu"),
        _choice("tone", "Màu ánh sáng", "lime", [("lime", "Vàng chanh (như đom đóm thật)"), ("warm", "Vàng ấm"),
                                                 ("green", "Xanh lá"), ("custom", "Tự chọn (ở dưới)")],
                presets={k: {"color": v} for k, v in TONES.items()}, match=["color"], group="Kiểu"),
        _colour("color", "Màu (tự chọn)", TONES["lime"], group="Kiểu"),
        _num("amount", "Số đom đóm", DEFAULTS["amount"], group="Đom đóm"),
        _num("size", "Cỡ", DEFAULTS["size"], group="Đom đóm"),
        _num("brightness", "Độ sáng", DEFAULTS["brightness"], group="Đom đóm"),
        _num("depth", "Gần xa (con gần to, nhoè)", DEFAULTS["depth"], group="Đom đóm"),
        _num("twinkle", "Nhấp nháy (0: sáng đều)", DEFAULTS["twinkle"], group="Nháy sáng"),
        _num("blink", "Nhịp nháy (thưa ← → dày)", DEFAULTS["blink"], group="Nháy sáng"),
        _num("wander", "Bay lượn xa", DEFAULTS["wander"], group="Bay"),
        _num("speed", "Tốc độ bay", DEFAULTS["speed"], group="Bay"),
        *_place(50)]}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def count(amount):
    """How many fireflies an "amount" (0..100) makes."""
    return int(round(CAP * 0.75 * (amount / 100) ** 1.3))


class Fireflies(Effect):
    blend = "screen"
    BOX = (16, 9)
    TURNS = False  # (a swarm of points: nothing to turn)
    PREVIEW = 1280  # its preview clip's height when its box is big in the video

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        u = np.random.default_rng([self.seed, 5501]).random((CAP, 24))[:count(p["amount"])]
        self.n = len(u)
        wide = self.BOX[0] / self.BOX[1]  # (the box's width, box heights)
        self.wide = wide
        # where each lives (box heights), how near it is (0 far .. 1 near: most are far)
        self.hx = EDGE + u[:, 0] * (wide - 2 * EDGE)
        self.hy = 0.12 + 0.8 * u[:, 1] ** 0.8
        near = p["depth"] / 100 * u[:, 2] ** 2
        # wandering: three loops each way (slow, middle, quick), whole cycles a loop, as far as "Bay lượn xa" says
        pace = 2.0 ** ((p["speed"] - 50) / 50)
        reach = _lerp(0.2, 1.8, p["wander"] / 100) * (0.6 + 0.8 * u[:, 3])
        rates = np.stack([_lerp(0.025, 0.06, u[:, 4]), _lerp(0.07, 0.16, u[:, 5]), _lerp(0.18, 0.4, u[:, 6])])
        self.mx = self.cycles(rates * pace)
        self.my = self.cycles(np.stack([_lerp(0.03, 0.07, u[:, 7]), _lerp(0.08, 0.18, u[:, 8]),
                                        _lerp(0.2, 0.45, u[:, 9])]) * pace)
        self.ax = np.array([0.1, 0.045, 0.015])[:, None] * reach
        self.ay = np.array([0.055, 0.03, 0.01])[:, None] * reach
        self.px_ = u[:, 10:13].T * TAU
        self.py_ = u[:, 13:16].T * TAU
        # flashing: its own beat (whole flashes a loop), a quick rise, a moment lit, a slower fade
        beat = 2.0 ** ((p["blink"] - 50) / 50)
        self.mf = self.cycles(_lerp(0.15, 0.4, u[:, 16]) * beat)
        self.pf = u[:, 17]
        period = self.loop_s / self.mf  # (seconds a beat)
        self.rise, self.hold = RISE / period, HOLD / period  # (shares of a beat)
        self.fade = FADE / period
        self.tail = np.exp(-(1 - self.rise - self.hold) / self.fade)  # (what is left at the beat's end: taken away)
        self.tw = p["twinkle"] / 100
        self.glow = _lerp(0.55, 1.0, u[:, 18]) * p["brightness"] / 100 * 1.6
        # sizes (box heights): near ones bigger and softer, as much light spread wider
        size = _lerp(0.5, 1.6, p["size"] / 100) * 0.0045 * (0.75 + 0.5 * u[:, 19])
        self.core = size * (1 + 2.5 * near)
        self.halo = size * (4.5 + 4.0 * near)
        self.bloom = size * (10.0 + 6.0 * near)
        self.glow /= 1 + 1.5 * near  # (a near one's light spread over a bigger disc: dimmer at its middle)
        colour = _rgb(p["color"])
        self.c_halo = colour.astype(np.float32)
        self.c_core = (colour + (1 - colour) * 0.6).astype(np.float32)  # (its middle burns paler)
        self._clip = None
        self.cache = (sprites.SpriteCache(lambda key, fx, fy: _core(key, fx, fy)),
                      sprites.SpriteCache(lambda key, fx, fy: _halo(key, fx, fy)))  # (halos and blooms alike)

    def where(self, tau):
        """Where each firefly is at loop phase tau (box heights from the box's top left)."""
        x = self.hx + (self.ax * np.sin(TAU * self.mx * tau + self.px_)).sum(axis=0)
        y = self.hy + (self.ay * np.sin(TAU * self.my * tau + self.py_)).sum(axis=0)
        return x, y

    def flash(self, tau):
        """Each firefly's flash at loop phase tau: 0 (between flashes) .. 1."""
        ph = (self.mf * tau + self.pf) % 1.0
        up = _smooth(ph / self.rise)
        down = (np.exp(-np.maximum(ph - self.rise - self.hold, 0.0) / self.fade) - self.tail) / (1 - self.tail)
        return np.where(ph < self.rise, up, np.where(ph < self.rise + self.hold, 1.0, down))

    def flashes(self, tau):
        """How lit each firefly is at loop phase tau (0..1): its flash, or its faint glow between flashes."""
        return (1 - self.tw) * 0.6 + self.tw * self.flash(tau)

    def _inside(self, x0, y0, w, h, mask):
        """Where their light may fall: inside the box, softly to its edges (its preview clip is the box alone), and
        the clip's area."""
        key = (x0, y0, w, h)
        if self._clip is None or self._clip[0] != key or self._clip[1] is not mask:
            ys = (np.arange(self.h, dtype=np.float32) + 0.5 - y0) / h
            xs = (np.arange(self.w, dtype=np.float32) + 0.5 - x0) / h
            inside = (_smooth(np.minimum(ys, 1 - ys) / SOFT)[:, None] *
                      _smooth(np.minimum(xs, w / h - xs) / SOFT)[None, :]).astype(np.float32)
            self._clip = (key, mask, inside if mask is None else inside * mask)  # (kept while the box and area stay)
        return self._clip[2]

    def draw(self, t, canvas, mask):
        x0, y0, w, h = self.box()
        if h < 2 or not self.n:
            return
        mask = self._inside(x0, y0, w, h, mask)
        tau = self.tau(t)
        x, y = self.where(tau)
        flash = self.flash(tau)
        y = y - LIFT * flash  # (a firefly rises a little as it flashes)
        edge = _smooth(np.minimum(np.minimum(x, self.wide - x), np.minimum(y, 1 - y)) / EDGE)
        lit = ((1 - self.tw) * 0.6 + self.tw * flash) * self.glow * edge
        xs, ys = x0 + x * h, y0 + y * h
        core, halo = self.cache
        reach = self.bloom * h * 2.6 + 2
        for i in np.flatnonzero(lit > 0.004):
            if xs[i] + reach[i] < 0 or ys[i] + reach[i] < 0 or xs[i] - reach[i] > self.w or ys[i] - reach[i] > self.h:
                continue
            ix, iy = math.floor(xs[i]), math.floor(ys[i])
            fx, fy = xs[i] - ix, ys[i] - iy
            if lit[i] > 0.25:  # (a wide bloom when it flashes; too faint to see between flashes)
                spot, sx, sy = halo.get(sprites.bucket(max(self.bloom[i] * h, 0.5)), fx, fy)
                canvas.light(ix + sx, iy + sy, spot, self.c_halo * (BLOOM * lit[i]), mask)
            spot, sx, sy = halo.get(sprites.bucket(max(self.halo[i] * h, 0.5)), fx, fy)
            canvas.light(ix + sx, iy + sy, spot, self.c_halo * (HALO * lit[i]), mask)
            spot, sx, sy = core.get(sprites.bucket(max(self.core[i] * h, 0.3)), fx, fy)
            canvas.light(ix + sx, iy + sy, spot, self.c_core * (CORE * lit[i]), mask)
