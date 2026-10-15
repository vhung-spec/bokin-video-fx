"""Falling rain (see effects.py for how every effect loops).

A drop is drawn as the streak it leaves while the camera's shutter is open: faint at its tail, brightest near its
head, row by row with exact coverage (a streak thinner than a pixel covers that share of it). Far drops are fine,
faint and slow on screen; near ones thicker, brighter, faster and a little out of focus (drawn on a half-size layer,
which softens them, and blurred past their ends too). Small drops fall slower and wander a little: drizzle drifts.
Gusts add to the wind now and then: a drop drifts with the air it falls through, so its streak leans with the gust.
Around the drops: soft sheets of rain blown along, a light mist, splashes and rings where drops reach the ground, and
flashes of lightning.

All of it is a function of tau = t / P with whole numbers of cycles per loop (gusts, sheets, rain coming in waves,
lightning), so frame P is frame 0 again.
"""
import math

import numpy as np
from PIL import Image

from .effects import Falling, _choice, _colour, _lerp, _num, _rgb

PRESETS = {
    "drizzle": {"amount": 70, "size": 12, "length": 12, "speed": 8, "opacity": 38, "variety": 35, "wind": 6,
                "gust": 10, "waves": 10, "depth": 40, "blur": 20, "veil": 45, "mist": 30, "splash": 20, "ripple": 25,
                "lightning": 0},
    "medium": {"amount": 45, "size": 35, "length": 50, "speed": 60, "opacity": 45, "variety": 50, "wind": 10,
               "gust": 20, "waves": 15, "depth": 50, "blur": 35, "veil": 30, "mist": 0, "splash": 50, "ripple": 30,
               "lightning": 0},
    "heavy": {"amount": 78, "size": 48, "length": 60, "speed": 75, "opacity": 50, "variety": 55, "wind": 15,
              "gust": 35, "waves": 25, "depth": 60, "blur": 45, "veil": 50, "mist": 20, "splash": 70, "ripple": 45,
              "lightning": 0},
    "storm": {"amount": 95, "size": 55, "length": 70, "speed": 90, "opacity": 50, "variety": 60, "wind": 40,
              "gust": 75, "waves": 45, "depth": 65, "blur": 50, "veil": 70, "mist": 30, "splash": 85, "ripple": 55,
              "lightning": 55, "flashes": 3},
}


def _p(key, label, value, group, lo=0, hi=100):
    return _num(key, label, value, lo, hi, group=group)


SCHEMA = {"label": "Mưa", "params": [
    _choice("look", "Kiểu mưa", "medium", [("drizzle", "Mưa phùn"), ("medium", "Mưa vừa"), ("heavy", "Mưa to"),
                                           ("storm", "Mưa bão")], presets=PRESETS, group="Kiểu"),
    _p("amount", "Mật độ", 45, "Hạt mưa"), _p("size", "Độ dày hạt", 35, "Hạt mưa"),
    _p("length", "Độ dài vệt", 50, "Hạt mưa"), _p("speed", "Tốc độ rơi", 60, "Hạt mưa"),
    _p("opacity", "Độ đậm", 45, "Hạt mưa"), _p("variety", "Hạt đậm nhạt khác nhau", 50, "Hạt mưa"),
    _p("edge", "Viền tối quanh hạt (thấy rõ hơn trên nền trời sáng)", 35, "Hạt mưa"),
    _p("wind", "Gió (nghiêng trái ← → phải)", 10, "Gió", -100, 100), _p("gust", "Gió giật từng cơn", 20, "Gió"),
    _p("waves", "Mưa từng đợt (lúc dày, lúc thưa)", 15, "Gió"),
    _p("depth", "Chiều sâu (xa mảnh, gần to)", 50, "Chiều sâu"), _p("blur", "Nhoè hạt ở gần", 35, "Chiều sâu"),
    _p("veil", "Màn mưa bay theo gió", 30, "Màn mưa"), _p("mist", "Hơi nước mờ", 0, "Màn mưa"),
    _p("ground", "Mặt đất ở đâu (tính từ mép trên xuống, %; 100: không thấy đất)", 100, "Bắn nước", 10, 100),
    _p("splash", "Nước bắn khi hạt chạm đất", 50, "Bắn nước"), _p("ripple", "Gợn nước trên mặt đất", 30, "Bắn nước"),
    _p("lightning", "Chớp sáng (0: không chớp)", 0, "Chớp"),
    _p("flashes", "Số lần chớp trong 30 giây", 2, "Chớp", 1, 10),
    _colour("color", "Màu", "#d4dce6", group="Màu")]}

PAD = 8  # spare columns either side of a layer: whatever falls outside the frame lands there
GRID = 2048  # samples of the periodic signals (gusts, waves) over one loop
BAND = 48  # rows laid over the picture at a time (they stay in the processor's cache)
RING_S = 0.6  # seconds a ring on the ground lasts


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _bumps(times, heights, width):
    """A periodic signal over one loop (GRID samples): smooth bumps at `times` (share of the loop), `width` wide."""
    tau = np.arange(GRID) / GRID
    k = 1.0 / (4 * math.pi ** 2 * width ** 2)
    out = np.zeros(GRID)
    for t0, a in zip(times, heights):
        out += a * np.exp(k * (np.cos(2 * math.pi * (tau - t0)) - 1))
    return out


def _at(table, tau):
    """A periodic signal (GRID samples over one loop) at phase(s) tau."""
    return np.interp(np.mod(tau, 1.0) * GRID, np.arange(GRID + 1), np.append(table, table[0]))


def _noise(rng, h, w, fy, fx):
    """Smooth noise that tiles, h x w, 0..1: features about h / fy tall and w / fx wide."""
    ky = np.fft.fftfreq(h)[:, None] * h
    kx = np.fft.rfftfreq(w)[None, :] * w
    r2 = (ky / fy) ** 2 + (kx / fx) ** 2
    amp = np.sqrt(r2) * np.exp(-r2)
    spec = amp * np.exp(2j * math.pi * rng.random(amp.shape))
    out = np.fft.irfft2(spec, (h, w))
    lo, hi = np.percentile(out, (1, 99))
    return np.clip((out - lo) / (hi - lo), 0, 1).astype(np.float32)


def _sample(tex, u, v):
    """`tex` (it tiles) at texel coordinates u (across) and v (down), bilinear, any shapes that broadcast."""
    h, w = tex.shape
    u, v = np.broadcast_arrays(u, v)
    u0, v0 = np.floor(u), np.floor(v)
    fu, fv = (u - u0).astype(np.float32), (v - v0).astype(np.float32)
    i0 = u0.astype(np.intp) % w
    i1 = (i0 + 1) % w
    j0 = (v0.astype(np.intp) % h) * w
    j1 = (j0 + w) % (h * w)
    flat = tex.ravel()
    top = flat[j0 + i0] * (1 - fu) + flat[j0 + i1] * fu
    bot = flat[j1 + i0] * (1 - fu) + flat[j1 + i1] * fu
    return top * (1 - fv) + bot * fv


def _up2(a, r0, r1, w):
    """Rows r0..r1-1 of `a` made twice as large (bilinear, pixel centres kept), the first w columns."""
    h2 = a.shape[0]
    s0, s1 = r0 // 2, (r1 + 1) // 2
    src = a[s0:s1]
    above = a[max(s0 - 1, 0):s1 - 1] if s0 > 0 else np.concatenate([a[:1], a[:s1 - 1]])
    below = a[s0 + 1:min(s1 + 1, h2)]
    if s1 + 1 > h2:
        below = np.concatenate([below, a[-1:]])
    t = np.empty((2 * (s1 - s0), a.shape[1]), np.float32)
    t[0::2] = src
    t[0::2] *= 0.75
    t[0::2] += 0.25 * above
    t[1::2] = src
    t[1::2] *= 0.75
    t[1::2] += 0.25 * below
    t = t[r0 - 2 * s0:r1 - 2 * s0]
    o = np.empty((t.shape[0], 2 * t.shape[1]), np.float32)
    o[:, 0::2] = t
    o[:, 0::2] *= 0.75
    o[:, 2::2] += 0.25 * t[:, :-1]
    o[:, 0] += 0.25 * t[:, 0]
    o[:, 1::2] = t
    o[:, 1::2] *= 0.75
    o[:, 1:-1:2] += 0.25 * t[:, 1:]
    o[:, -1] += 0.25 * t[:, -1]
    return o[:, :w]


def _tent(s):
    """How much of a tent of half-width 1 (and area 1) lies left of s."""
    s = np.clip(s, -1.0, 1.0)
    return np.where(s < 0, 0.5 * (1 + s) ** 2, 1 - 0.5 * (1 - s) ** 2)


def _over(canvas, r0, cover, colour, mask, under=None, colour2=None):
    """Lays rows r0.. of one colour (coverage `cover`), with a second colour's coverage `under` beneath it, over
    what is on `canvas` (one channel at a time: much quicker than letting numpy spread a colour over pixels)."""
    prgb = np.empty(cover.shape + (3,), np.float32)
    tmp = np.empty_like(cover)
    for k in range(3):
        np.multiply(cover, colour[k], out=tmp)
        if under is not None:
            tmp += under * colour2[k]
        prgb[..., k] = tmp
    canvas.over_rgba(0, r0, prgb, cover if under is None else cover + under, mask)


class Layer:
    """Coverage of one colour, `w` x `h`, with spare columns either side (flat, for adding many pieces at once)."""

    def __init__(self, w, h):
        self.w, self.h, self.stride = w, h, w + 2 * PAD
        self.flat = np.zeros(h * self.stride, np.float32)

    def view(self):
        return self.flat.reshape(self.h, self.stride)[:, PAD:PAD + self.w]

    def streaks(self, tx, ty, hx, hy, width, weight, cut, soft=False):
        """Streaks from tail (tx, ty) to head (hx, hy), pixels, falling (hy > ty): `width` px across, `weight` their
        coverage at the brightest, nothing below row `cut`; soft: out of focus, blurred `width` px every way (across,
        past both ends, into the ground). Row by row: each row gets the share of the streak in it."""
        dy = hy - ty
        length = np.hypot(hx - tx, dy)
        across = length / dy * width  # the streak's width along a row, at its head
        if soft:
            y0, y1 = ty - dy / length * width, np.minimum(hy + dy / length * width, cut + width)
        else:
            y0, y1 = ty, np.minimum(hy, cut)
        top, bot = np.maximum(y0, 0.0), np.minimum(y1, self.h)
        reach = across + width + 2
        keep = np.flatnonzero((bot > top) & (np.maximum(tx, hx) > -reach) & (np.minimum(tx, hx) < self.w + reach))
        if not len(keep):
            return
        tx, ty, hx, top, bot, cut = tx[keep], ty[keep], hx[keep], top[keep], bot[keep], cut[keep]
        dy, length, across, width, weight = dy[keep], length[keep], across[keep], width[keep], weight[keep]
        r0 = np.floor(top).astype(np.intp)
        n = np.ceil(bot).astype(np.intp) - r0
        idx = np.repeat(np.arange(len(n)), n)
        row = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n) + r0[idx]
        y_a = np.maximum(row, top[idx]).astype(np.float32)
        y_b = np.minimum(row + 1, bot[idx]).astype(np.float32)
        mid = 0.5 * (y_a + y_b)
        u = ((mid - ty[idx]) / dy[idx]).astype(np.float32)  # 0 at the tail .. 1 at the head
        x = (tx[idx] + u * (hx - tx)[idx]).astype(np.float32)
        uc = np.clip(u, 0.0, 1.0)
        span = (across[idx] * (0.35 + 0.65 * np.sqrt(uc))).astype(np.float32)  # thinner to the tail
        to_head = (1.0 - u) * length[idx].astype(np.float32)
        if soft:  # the even streak blurred along its length by the same tent as across: soft ends
            blur = width[idx].astype(np.float32)
            from_tail = u * length[idx].astype(np.float32)
            ends = np.maximum(_tent(from_tail / blur) + _tent(to_head / blur) - 1.0, 0.0)
            ends *= _tent((cut[idx] - mid) / blur)  # and fading into the ground, not cut by it
        else:
            ends = np.minimum(1.0, to_head / (1.5 + 0.6 * span) + 0.3)  # a rounded head, as round as it is wide
        w = weight[idx].astype(np.float32) * (y_b - y_a) * (0.1 + 0.9 * uc ** 1.6) * ends
        self._rows(row, x, span, w, soft)

    def _rows(self, row, x, span, w, soft):
        """Adds `w` times each pixel's share of a line `span` px wide centred on x in row `row`: an even band
        (sharp), or a tent of the same area, twice as wide (soft)."""
        half = span if soft else span / 2
        c0 = np.floor(x - half)
        base = row * self.stride + PAD
        cols = (2 * half + 2).astype(np.intp)
        for j in range(int(cols.max())):
            sel = np.flatnonzero(cols > j) if j >= 2 else slice(None)
            c = c0[sel] + j
            a, b, xs = c - x[sel], c + 1 - x[sel], half[sel]
            if soft:
                cover = xs * (_tent(b / xs) - _tent(a / xs))
            else:
                cover = np.clip(np.minimum(b, xs) - np.maximum(a, -xs), 0.0, 1.0)
            at = base[sel] + np.clip(c, -PAD, self.w + PAD - 1).astype(np.intp)
            np.add.at(self.flat, at, w[sel] * cover)

    def dots(self, x, y, w):
        """Points at (x, y) px, each adding `w` (its area times its coverage) to the pixels around it."""
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = (x - x0).astype(np.float32), (y - y0).astype(np.float32)
        col = np.clip(x0, -PAD, self.w + PAD - 2).astype(np.intp) + PAD
        row = y0.astype(np.intp)
        w = w.astype(np.float32)
        for dr, wy in ((0, 1 - fy), (1, fy)):
            r = row + dr
            ok = (r >= 0) & (r < self.h)
            if not ok.any():
                continue
            at = r[ok] * self.stride + col[ok]
            a = (w * wy)[ok]
            np.add.at(self.flat, at, a * (1 - fx[ok]))
            np.add.at(self.flat, at + 1, a * fx[ok])


class Rain(Falling):
    CAP = 16000
    VERSION = 2

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 12))
        z = u[:, 0] ** 1.5  # 0 far .. 1 near: more drops far away
        ds = p["depth"] / 100
        near = np.exp(ds * 3.0 * (z - 0.3))  # how near (perspective): far drops small and slow, near ones large
        size_k = near ** 0.75
        speed_k = near ** 0.7
        drop = u[:, 3]  # how big a drop is: big ones fall faster
        fine = math.sqrt((0.15 + p["size"] / 100) / 0.5)  # and so does rain of big drops (1 at size 35)
        sp, ln = p["speed"] / 100, p["length"] / 100
        self.mean_speed = (0.6 + 4.933 * sp - 1.333 * sp * sp) * fine  # frame heights a second (3.08 at 60)
        speed = self.mean_speed * speed_k * (0.8 + 0.4 * drop)  # frame heights a second
        self.angle = math.radians(30) * p["wind"] / 100
        angle = self.angle + np.radians(2) * (u[:, 7] - 0.5)
        self.shutter = (0.08 + 1.36 * ln - 0.34 * ln * ln) / self.fps  # seconds of motion in a streak (0.675 at 50)

        # gusts: bursts of wind on top of the steady one, leaning the rain further; rain comes in waves with them
        srng = np.random.default_rng([self.seed, 1103, 2])
        count = max(1, round(self.loop_s / 8))
        times = (np.arange(count) + 0.2 + 0.6 * srng.random(count)) / count
        heights = 0.55 + 0.45 * srng.random(count)
        side = 1.0 if p["wind"] >= 0 else -1.0
        bursts = _bumps(times, heights, min(0.3, 1.6 / self.loop_s))
        gust = side * 0.6 * p["gust"] / 100 * bursts / bursts.max()
        self.gusting = gust  # extra slant (horizontal speed over falling speed)
        self.gust_mean = float(gust.mean())
        self.carried = np.cumsum(gust - self.gust_mean) / GRID  # its integral less the steady part: periodic
        self.carried -= self.carried.mean()
        waves = _bumps(times + 0.02, heights, min(0.35, 2.6 / self.loop_s))
        waves = (waves - waves.min()) / max(float(waves.max() - waves.min()), 1e-9)
        self.density = 1 - 0.85 * p["waves"] / 100 * (1 - waves)  # share of the drops falling (by when they start)
        # small drops wander a little in the air instead of falling straight (whole cycles a loop)
        wrng = np.random.default_rng([self.seed, 1103, 5])
        uw = wrng.random((self.CAP, 2))
        wander = 0.012 * (1 - drop) * (1 - p["size"] / 100) ** 3 * near ** 0.5  # frame heights
        reach = float(np.abs(gust).max()) * 1.3  # how far a gust can carry a drop sideways while it is in view
        reach += float(wander.max())

        longest = self.shutter * float(speed.max()) * 1.3
        share = self.paths(u[:, 1], u[:, 2], speed, angle, longest + 0.02, longest + 0.04 + reach)
        wanted = p["amount"] / 100 * 2600 * self.aspect / (16 / 9) / float(self.density.mean())
        v = p["variety"] / 100
        glint = 0.3 + 1.2 * u[:, 5] ** 2  # how much light a drop catches
        opacity = p["opacity"] / 100 * 0.62 * near ** 0.55 * ((1 - v) * 0.8 + v * glint)
        width = _lerp(0.55, 2.0, p["size"] / 100) * size_k * (0.6 + 0.5 * drop + 0.3 * u[:, 4]) * px
        defocus = np.minimum(1.0, 1.3 * p["blur"] / 100 * _smooth(0.4, 1.0, z))
        g = p["ground"] / 100
        hit = np.where(g < 1, g + (1 - g) * 1.3 * z ** 0.8, 9.0)  # where on the ground a drop lands (frame heights)
        kept = self.choose(share, wanted, {
            "z": z, "width": width, "opacity": np.minimum(1.0, opacity * 1.6), "defocus": defocus, "hit": hit,
            "gate": u[:, 8], "su": u[:, 9], "sv": u[:, 10], "sw": u[:, 11], "wander": wander,
            "wm": self.cycles(_lerp(0.25, 0.8, uw[:, 0])), "wphi": uw[:, 1] * 2 * math.pi})
        self.__dict__.update(kept)
        self.n = len(self.z)
        self.blurred = self.defocus > 0.02
        self.wide = self.width * (1 + 0.8 * self.defocus) + 6.0 * self.defocus * px  # out of focus: spread out

        # splashes and rings where drops land: how strong, how long, how big (by how near the spot is)
        self.ground = g
        self.splash, self.ripple = p["splash"] / 100, p["ripple"] / 100
        close = np.clip((self.hit - g) / max(1 - g, 1e-6), 0, 1.3)  # 0 at the far edge of the ground
        self.spot = (0.15 + 0.85 * close) * px  # size of things on the ground there, px of 1080 per px
        self.pass_s = self.len / self.speed  # seconds between two landings of the same drop

        # sheets of rain blown along (two drifting patterns, so their shapes change), mist
        vrng = np.random.default_rng([self.seed, 1103, 3])
        self.sheet_a = _noise(vrng, 64, 128, 1.5, 9.0)  # 2 x 4 frame heights
        self.sheet_b = _noise(vrng, 64, 128, 1.2, 7.0)
        self.haze = _noise(vrng, 32, 32, 1.3, 1.3)  # 4 x 4 frame heights
        amount_k = min(1.0, p["amount"] / 25)
        self.veil, self.mist = 0.14 * p["veil"] / 100 * amount_k, 0.12 * p["mist"] / 100 * amount_k
        self.clumps = p["veil"] / 100
        drift = (0.06 + 0.3 * abs(p["wind"]) / 100) * self.loop_s  # frame heights a loop
        down = self.mean_speed * self.loop_s
        # how far each pattern moves in a loop, frame heights: whole tiles (4 wide, 2 tall), so it loops
        self.move_a = (side * max(1, round(drift / 4)) * 4, max(1, round(0.15 * down / 2)) * 2)
        self.move_b = (side * max(1, round(drift * 1.6 / 4)) * 4, max(1, round(0.1 * down / 2)) * 2)
        self.move_h = side * max(1, round(drift * 0.4 / 4)) * 4

        # lightning
        self.flash = p["lightning"] / 100
        lrng = np.random.default_rng([self.seed, 1103, 4])
        nf = max(1, round(p["flashes"] * self.loop_s / 30))
        self.flash_t = (np.arange(nf) + 0.1 + 0.8 * lrng.random(nf)) / nf * self.loop_s
        self.flash_parts = [((0.0, 1.0, 0.07), (0.08 + 0.06 * r1, 0.5 + 0.4 * r2, 0.06), (0.25 + 0.1 * r3, 0.3, 0.12))
                            for r1, r2, r3 in lrng.random((nf, 3))]
        self.flash_side = lrng.random(nf) * 2 - 1
        self.colour = _rgb(p["color"])
        self.edge = p["edge"] / 100
        self.rim_colour = self.colour * 0.08  # water bends the light away at a drop's edges: a dark rim
        self.flash_colour = np.array([0.92, 0.95, 1.0], np.float32)

    # --- what moves -------------------------------------------------------------------------------------------

    def flash_at(self, tau):
        """How bright the lightning is at tau (0..1), and which side it is on."""
        if self.flash <= 0:
            return 0.0, 0.0
        t = tau * self.loop_s
        best, side = 0.0, 0.0
        for t0, parts, s in zip(self.flash_t, self.flash_parts, self.flash_side):
            dt = (t - t0) % self.loop_s
            e = sum(a * math.exp(-(dt - d) / k) for d, a, k in parts if dt >= d)
            if e > best:
                best, side = e, s
        return min(1.0, best), side

    def sheets(self, tau, x, y):
        """How dense the sheets of rain are (0..1) at x, y (frame heights)."""
        slope = math.tan(self.angle) + float(_at(self.gusting, tau)) / math.cos(self.angle)
        push = self.mean_speed * self.loop_s * float(_at(self.carried, tau))
        across = x - (y - 0.5) * slope + push
        out = 0
        for tex, (mx, my) in ((self.sheet_a, self.move_a), (self.sheet_b, self.move_b)):
            h, w = tex.shape
            out = out + _sample(tex, (across - mx * tau) * (w / 4), (y - my * tau) * (h / 2))
        return _smooth(0.7, 1.5, out)

    def drops(self, t):
        """Every drop's head (x, y, px), the way it moves (unit vector), streak length (px), where it is on its
        path, and whether it shows."""
        tau = self.tau(t)
        s = np.mod(self.s0 + self.k * self.len * tau, self.len)
        d = s - self.extra
        x = self.xe + d * self.sin
        y = -self.my + d * self.cos
        since = np.maximum(d, 0) / (self.k * self.len)  # share of the loop since it came into view
        x = x + self.speed * self.loop_s * (_at(self.carried, tau) - _at(self.carried, tau - since) +
                                            self.gust_mean * since)  # how far the gusts carried it
        phase = 2 * math.pi * self.wm * tau + self.wphi
        x = x + self.wander * np.sin(phase)
        lean = self.sin + _at(self.gusting, tau)
        lean = lean + self.wander * 2 * math.pi * self.wm / self.loop_s * np.cos(phase) / self.speed
        norm = np.hypot(lean, self.cos)
        falls = _at(self.density, tau - s / (self.k * self.len)) > self.gate
        length = self.shutter * self.speed * norm * self.h
        return x * self.h, y * self.h, lean / norm, self.cos / norm, length, s, (d >= 0) & falls

    def in_view(self, hx, ux, length, show):
        """The drops whose streak (out of focus: and its blur) reaches into the frame."""
        tx = hx - ux * length
        pad = float(self.wide.max()) + 4 * self.px
        return np.flatnonzero(show & (np.maximum(hx, tx) > -pad) & (np.minimum(hx, tx) < self.w + pad))

    def draw(self, t, canvas, mask):
        tau = self.tau(t)
        flash, side = self.flash_at(tau)
        if flash > 0.002:
            # light from the sky: strong at the top, little left near the ground
            rows = 0.05 + 0.95 * (1 - np.linspace(0, 1, self.h, dtype=np.float32)) ** 2.5
            rows *= 0.45 * self.flash * flash
            cols = 1 + 0.35 * side * np.linspace(-1, 1, self.w, dtype=np.float32)
            for r0 in range(0, self.h, BAND):
                r1 = min(r0 + BAND, self.h)
                _over(canvas, r0, np.minimum(np.outer(rows[r0:r1], cols), 1.0), self.flash_colour, mask)
        if self.n == 0:
            return
        sharp = Layer(self.w, self.h)
        soft_size = (self.w + 1) // 2, (self.h + 1) // 2
        on_ground = self.ground < 1 and (self.splash > 0 or self.ripple > 0)
        need = self.blurred.any() or self.veil + self.mist > 0 or (on_ground and self.px >= 1.5)
        soft = Layer(*soft_size) if need else None
        hx, hy, ux, uy, length, s, show = self.drops(t)
        i = self.in_view(hx, ux, length, show)
        glow = self.opacity[i] * (1 + 1.5 * self.flash * flash)
        if self.clumps > 0:  # drops in a sheet of rain are denser: brighter
            dense = self.sheets(tau, hx[i] / self.h, hy[i] / self.h)
            glow = glow * (1 - 0.6 * self.clumps + 1.2 * self.clumps * dense)
        hx, hy, ux, uy, length = hx[i], hy[i], ux[i], uy[i], length[i]
        cut, width, b = self.hit[i] * self.h, self.width[i], self.defocus[i]
        tx, ty = hx - ux * length, hy - uy * length
        if soft is None:
            sharp.streaks(tx, ty, hx, hy, width, glow, cut)
        else:
            k = b < 0.98
            sharp.streaks(tx[k], ty[k], hx[k], hy[k], width[k], (glow * (1 - b))[k], cut[k])
            k = self.blurred[i]
            wide = self.wide[i]
            soft.streaks(tx[k] / 2, ty[k] / 2, hx[k] / 2, hy[k] / 2, wide[k] / 2, (glow * b * width / wide)[k],
                         cut[k] / 2, soft=True)
        shade = None
        if self.edge > 0:  # a soft dark rim either side of each streak, under it
            shade = Layer(*soft_size)
            rim = width + 2.4 * self.px
            shade.streaks(tx / 2, ty / 2, hx / 2, hy / 2, rim / 2, glow * self.edge * 0.55, cut / 2, soft=True)
        if on_ground:
            self._splashes(tau, sharp, soft, s)
        if soft is not None and self.veil + self.mist > 0:
            self._veil(tau, soft)
        self._lay(canvas, mask, sharp, soft, shade)

    def _lay(self, canvas, mask, sharp, soft, shade):
        """The layers over the picture (the dark rims under the rest), a band of rows at a time."""
        top = sharp.view()
        half = soft.view() if soft is not None else None
        dark = shade.view() if shade is not None else None
        for r0 in range(0, self.h, BAND):
            r1 = min(r0 + BAND, self.h)
            if half is None:
                cover = np.minimum(top[r0:r1], 1.0)
            else:
                cover = _up2(half, r0, r1, self.w)
                cover += top[r0:r1]
                np.minimum(cover, 1.0, out=cover)
            if dark is None:
                _over(canvas, r0, cover, self.colour, mask)
                continue
            rim = _up2(dark, r0, r1, self.w)
            np.minimum(rim, 1.0, out=rim)
            rim *= 1 - cover
            _over(canvas, r0, cover, self.colour, mask, rim, self.rim_colour)

    def _splashes(self, tau, sharp, soft, s):
        """Water thrown up where drops land, and rings spreading on the wet ground. Drawn where a pixel is about a
        pixel of a 1080-line frame (the half-size layer from 4K up), so they look alike at every size."""
        d_hit = (self.hit + self.my) / self.cos
        s_hit = self.extra + d_hit
        age = np.mod(s - s_hit, self.len) / self.speed  # seconds since the drop landed
        ok = (self.hit < 1.0 + 0.02) & (d_hit < self.len)
        ok &= age < np.minimum(RING_S, 0.9 * self.pass_s)  # (as long as the longest part lasts: the rings)
        if not ok.any():
            return
        i = np.flatnonzero(ok)
        a = age[i]
        tau_hit = tau - a / self.loop_s
        landed = _at(self.density, tau_hit - s_hit[i] / (self.k[i] * self.len[i])) > self.gate[i]
        i, a, tau_hit = i[landed], a[landed], tau_hit[landed]
        since = d_hit[i] / (self.k[i] * self.len[i])
        x = self.xe[i] + d_hit[i] * self.sin[i] + self.speed[i] * self.loop_s * (
            _at(self.carried, tau_hit) - _at(self.carried, tau_hit - since) + self.gust_mean * since)
        x = x + self.wander[i] * np.sin(2 * math.pi * self.wm[i] * tau_hit + self.wphi[i])
        layer, f = (soft, 0.5) if soft is not None and self.px >= 1.5 else (sharp, 1.0)
        x, y = x * self.h * f, self.hit[i] * self.h * f
        size = self.spot[i] * f  # this layer's px for a size 1 at the bottom of a 1080-line frame
        o = self.opacity[i]
        if self.splash > 0:
            self._crowns(layer, a, x, y, size, o, i)
        if self.ripple > 0:
            self._rings(layer, a, x, y, size, o, i)

    def _rings(self, layer, a, x, y, size, o, i):
        """Rings spreading on the wet ground, a faint one inside: thin, soft lines that catch the light on their far
        side (the near side of a ripple faces down, away from the sky)."""
        dur = np.minimum(RING_S, 0.9 * self.pass_s[i])
        k = np.flatnonzero(a < dur)
        if not len(k):
            return
        f = a[k] / dur[k]
        flat = 0.2 + 0.25 * np.clip((self.hit[i][k] - self.ground) / max(1 - self.ground, 1e-6), 0, 1)
        reach = size[k] * (0.7 + 0.6 * self.sw[i][k])
        line = 0.8 * size[k]  # how thick the line is, px
        xs, ys, ws = [], [], []
        for lag, strength in ((0.0, 1.0), (0.3, 0.25)):
            g = np.clip((f - lag) / (1 - lag), 0, 1)
            rx = (1.5 + 11.0 * np.sqrt(g)) * reach
            ry = rx * flat
            around = 2 * math.pi * np.sqrt(0.5 * (rx * rx + ry * ry))
            n = np.where(g > 0, np.maximum(24, np.ceil(around / 0.8)), 0).astype(np.intp)  # a point every 0.8 px
            ring = np.repeat(np.arange(len(n)), n)
            th = (np.arange(len(ring)) - np.repeat(np.cumsum(n) - n, n)) * (2 * math.pi) / n[ring]
            sin = np.sin(th)
            cover = 0.55 * self.ripple * o[k] * strength * (1 - g) ** 1.5  # at its brightest, on the far side
            xs.append(x[k][ring] + rx[ring] * np.cos(th))
            ys.append(y[k][ring] + ry[ring] * sin)
            ws.append((cover * line * around / np.maximum(n, 1))[ring] * (0.08 + 0.92 * np.clip(-sin, 0, 1)))
        layer.dots(np.concatenate(xs), np.concatenate(ys), np.concatenate(ws))

    def _crowns(self, layer, a, x, y, size, o, i):
        """Droplets thrown up and out where drops land, each the short line it draws while the shutter is open,
        and the drop breaking."""
        dur = np.minimum(0.08 + 0.06 * self.sw[i], 0.9 * self.pass_s[i])  # over before the drop lands again
        live = a < dur
        if not live.any():
            return
        a, x, y, size, o, dur = a[live], x[live], y[live], size[live], o[live], dur[live]
        su, sv = self.su[i][live], self.sv[i][live]
        fade = np.sqrt(1 - a / dur)
        fast = 250 * float(size.max())  # px a second, the fastest droplet
        m = int(np.clip(np.ceil(fast * self.shutter / 0.7) + 1, 2, 12))  # points along each line, 0.7 px apart
        for j in range(4):  # (how each droplet flies: its own mix of the drop's random draws)
            r1, r2 = (su * (2.618 * j + 1.3) + sv * (1.618 * j + 0.7)) % 1.0, (sv * (j + 3) + 0.37 * su) % 1.0
            ang = (r1 - 0.5) + 0.12 * (j - 1.5)  # from straight up, within 0.5 or so
            speed = (90 + 160 * r2) * size  # px a second
            rise, side = speed * np.cos(ang), speed * np.sin(ang)
            fall = 2 * rise / dur  # back on the ground at the end
            w = self.splash * o * 1.6 * size ** 2 * fade / m
            for back in np.linspace(0, 1, m):  # where it was while the shutter was open
                b = np.maximum(a - back * self.shutter, 0)
                layer.dots(x + side * b, y - np.maximum(rise * b - 0.5 * fall * b * b, 0), w)
        first = a < 0.045  # the drop breaking on the ground: a short bright flat spot
        if first.any():
            w = (self.splash * o * 1.4 * size ** 2)[first]
            for dx in (-0.6, 0.6):
                layer.dots(x[first] + dx * size[first], y[first] - 0.4 * size[first], w)

    def _veil(self, tau, layer):
        """Sheets of rain and a light mist, worked out at an eighth of the size, added to the half-size layer."""
        w8, h8 = (self.w + 7) // 8, (self.h + 7) // 8
        x = ((np.arange(w8) + 0.5) * (self.w / w8) / self.h)[None, :]  # frame heights
        y = ((np.arange(h8) + 0.5) * (self.h / h8) / self.h)[:, None]
        dens = float(_at(self.density, tau))
        a = np.zeros((h8, w8), np.float32)
        if self.veil > 0:
            a += self.veil * (0.6 + 0.4 * dens) * self.sheets(tau, x, y)
        if self.mist > 0:
            hh, hw = self.haze.shape
            haze = _sample(self.haze, (x - self.move_h * tau) * (hw / 4), y * (hh / 4))
            a += self.mist * (0.5 + haze) * (0.8 + 0.2 * dens)
        small = Image.fromarray(a, "F").resize((layer.w, layer.h), Image.Resampling.BILINEAR)
        layer.view()[:] += np.asarray(small)
