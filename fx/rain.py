"""Falling rain (see effects.py for how every effect loops).

A drop is drawn as the streak it leaves while the camera's shutter is open (about half a frame): an even band as wide
as the drop, round at both ends, row by row with exact coverage (a streak thinner than a pixel covers that share of
it). One perspective factor sets both how wide a drop is and how fast it crosses the screen, so a thin drop is a far
one and moves slowly; near ones are wide and fast, and the nearest out of focus (drawn on a half-size layer, which
softens them, and blurred past their ends too). Small drops fall a little slower, lean further in the wind and wander:
drizzle drifts. Each time a drop comes down again it falls in another place (lanes()), so no streak runs down the same
line twice a second. Gusts add to the wind now and then: a drop drifts with the air it falls through, so its streak
leans with the gust.

A drop is a tiny wide-angle lens: what it shows is the light around it, not what is behind it. Given the scene's light
(light=: effects.light_map of the video), a drop is that light: warm and glittering near lamps and lit windows, shown
most where it falls before something darker than the light around it (lit from beside or behind), a little darker than
a lit window or a bright sky behind it (against a flat sky it barely shows), and faint where nothing lights it. The
"lit" setting mixes in the rain's own colour (at the default, a little: rain in the dark stays faint). The sheets of
rain and the mist show the light around them, with almost none of their own colour: they glow round the lights and
leave the dark dark. A light brighter than white is white (each channel clipped as a camera does), so nothing laid
over a lamp or the sun darkens it; a drop that bright shows more of itself instead. It stays an effect drawn over
the picture (not a lens): its preview clip carries it all.

Around the drops: soft sheets of rain blown along, a light mist, flashes of lightning, and on the ground (when there is
one in the picture) the drops landing: far more of them than the streaks drawn, each a crown of droplets and rings
spreading on the wet ground, and spray over it in heavy rain.

All of it is a function of tau = t / P with whole numbers of cycles per loop (gusts, sheets, rain coming in waves,
lightning, each drop's lanes, each place on the ground's landings), so frame P is frame 0 again.
"""
import math

import numpy as np
from PIL import Image

from .effects import Falling, LUMA, _choice, _colour, _lerp, _num, _rgb, linear, srgb

PRESETS = {
    "drizzle": {"amount": 80, "size": 10, "length": 50, "speed": 10, "opacity": 30, "variety": 40, "wind": 6,
                "gust": 10, "waves": 10, "depth": 40, "blur": 25, "veil": 10, "mist": 60, "splash": 15, "ripple": 30,
                "lightning": 0},
    "medium": {"amount": 45, "size": 35, "length": 50, "speed": 60, "opacity": 45, "variety": 50, "wind": 10,
               "gust": 20, "waves": 15, "depth": 50, "blur": 35, "veil": 30, "mist": 0, "splash": 50, "ripple": 30,
               "lightning": 0},
    "heavy": {"amount": 75, "size": 50, "length": 55, "speed": 70, "opacity": 50, "variety": 55, "wind": 15,
              "gust": 35, "waves": 25, "depth": 60, "blur": 45, "veil": 45, "mist": 20, "splash": 70, "ripple": 45,
              "lightning": 0},
    "storm": {"amount": 90, "size": 58, "length": 60, "speed": 82, "opacity": 50, "variety": 60, "wind": 40,
              "gust": 75, "waves": 45, "depth": 65, "blur": 50, "veil": 65, "mist": 35, "splash": 85, "ripple": 55,
              "lightning": 55, "flashes": 3},
}
# the looks as they were before the rain was lit by the scene (VERSION 3): a clip that kept one gets it as it is now
OLD_PRESETS = {
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
    _p("ground", "Mặt đất bắt đầu cách mép trên (% chiều cao; 100 = không thấy đất)", 100, "Bắn nước", 10, 100),
    _p("splash", "Nước bắn khi hạt chạm đất", 50, "Bắn nước"), _p("ripple", "Gợn nước trên mặt đất", 30, "Bắn nước"),
    _p("lightning", "Chớp sáng (0: không chớp)", 0, "Chớp"),
    _p("flashes", "Số lần chớp trong 30 giây", 2, "Chớp", 1, 10),
    _p("lit", "Hạt sáng theo đèn trong cảnh (0: chỉ một màu)", 70, "Màu và ánh sáng"),
    _colour("color", "Màu", "#d4dce6", group="Màu và ánh sáng")]}


def upgrade(raw):
    """Settings saved before the rain was lit by the scene (no "lit"): a look kept as it was becomes that look as it
    is now; settings changed by hand stay as they were (drawn the new way)."""
    if not isinstance(raw, dict) or "lit" in raw:
        return raw
    look = raw.get("look", "medium")
    old = OLD_PRESETS.get(look) if isinstance(look, str) else None
    if old is None:
        return raw
    try:
        kept = all(float(raw.get(k, OLD_PRESETS["medium"].get(k, v))) == v for k, v in old.items())
    except (TypeError, ValueError, OverflowError):
        kept = False
    return dict(raw, **PRESETS[look]) if kept else raw


PAD = 8  # spare columns either side of a layer: whatever falls outside the frame lands there
GRID = 2048  # samples of the periodic signals (gusts, waves) over one loop
BAND = 48  # rows laid over the picture at a time (they stay in the processor's cache)
RING_S = 0.6  # seconds a ring on the ground lasts
IMPACTS = 15000  # drops landing a second on the ground in view at amount 100 (ground from 45% of the height down)
MOST_IMPACTS = 24000
LANE = 0.382  # about this share of the room a drop's lane moves on each time it comes down
STRENGTH = 1.4  # coverage of a drop at opacity 100, before its own light and nearness
DROP_GAIN = 1.0  # a lit drop: this many times the light around it
LIT_SIDE = 1.5  # at night, before something darker than the light around it, a drop shows that much more of it
# (in daylight 40% of that)
MIST_GAIN = 1.0  # lit sheets and mist: the light around them (they scatter it towards the camera)
NEUTRAL = linear(_rgb("#d4dce6"))  # the default colour tints a lit drop not at all (another one does)
BOOST = 3.0  # a lit drop brighter than white shows more (coverage), at most this many times more


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


def _over(canvas, r0, cover, colour, mask):
    """Lays rows r0.. of one colour (coverage `cover`) over what is on `canvas`."""
    canvas.over_rgba(0, r0, _tint(cover, colour, r0), cover, mask)


def _tint(a, colour, r0, lift=None, into=None):
    """Coverage `a` (rows r0.., all columns) as premultiplied colour (added to `into` if given): one colour (3,), or
    a map of colours (uint8 planes (3, h, w): 0..255; the light changes slowly, 8 bits are plenty and as quick as one
    colour), plus `lift` (3,) if given. One channel at a time: much quicker than letting numpy spread a colour."""
    rows, w = a.shape
    out = np.empty((rows, w, 3), np.float32) if into is None else into
    tmp = np.empty((rows, w), np.float32) if into is not None else None
    if colour.ndim == 1:
        c = colour if lift is None else np.minimum(colour + lift, 1.0)
        planes = [None] * 3
    else:
        planes = colour[:, r0:r0 + rows]
        if lift is None:
            a = a * (1 / 255.0)
        else:  # (a flash of lightning: a few frames)
            planes = [np.minimum(planes[k] * (1 / 255.0) + lift[k], 1.0) for k in range(3)]
    for k in range(3):
        by = c[k] if planes[k] is None else planes[k]
        if into is None:
            np.multiply(a, by, out=out[..., k])
        else:
            np.multiply(a, by, out=tmp)
            out[..., k] += tmp
    return out


class Layer:
    """Coverage of one colour, `w` x `h`, with spare columns either side (flat, for adding many pieces at once)."""

    def __init__(self, w, h):
        self.w, self.h, self.stride = w, h, w + 2 * PAD
        self.flat = np.zeros(h * self.stride, np.float32)

    def view(self):
        return self.flat.reshape(self.h, self.stride)[:, PAD:PAD + self.w]

    def streaks(self, tx, ty, hx, hy, width, weight, cut, soft=False):
        """Streaks from tail (tx, ty) to head (hx, hy), pixels, falling (hy > ty): even bands `width` px across, round
        at both ends, `weight` their coverage, nothing below row `cut`; soft: out of focus, blurred `width` px every way
        (across, past both ends, into the ground). Row by row: each row gets the share of the streak in it."""
        dy = hy - ty
        length = np.hypot(hx - tx, dy)
        across = length / dy * width  # the streak's width along a row
        reach = 0.5 * width + 0.5 if not soft else width  # how far past its ends it shows (a round end, or its blur)
        y0 = ty - dy / length * reach
        y1 = np.minimum(hy + dy / length * reach, cut + width) if soft else np.minimum(hy + dy / length * reach, cut)
        top, bot = np.maximum(y0, 0.0), np.minimum(y1, self.h)
        side = across + width + 2
        keep = np.flatnonzero((bot > top) & (np.maximum(tx, hx) > -side) & (np.minimum(tx, hx) < self.w + side))
        if not len(keep):
            return
        tx, ty, hx, top, bot, cut = tx[keep], ty[keep], hx[keep], top[keep], bot[keep], cut[keep]
        dy, length, across, width, weight = dy[keep], length[keep], across[keep], width[keep], weight[keep]
        reach = reach[keep]
        r0 = np.floor(top).astype(np.intp)
        n = np.ceil(bot).astype(np.intp) - r0
        idx = np.repeat(np.arange(len(n)), n)
        row = np.arange(len(idx)) - np.repeat(np.cumsum(n) - n, n) + r0[idx]
        y_a = np.maximum(row, top[idx]).astype(np.float32)
        y_b = np.minimum(row + 1, bot[idx]).astype(np.float32)
        mid = 0.5 * (y_a + y_b)
        u = ((mid - ty[idx]) / dy[idx]).astype(np.float32)  # 0 at the tail .. 1 at the head
        x = (tx[idx] + np.clip(u, 0.0, 1.0) * (hx - tx)[idx]).astype(np.float32)
        span = across[idx].astype(np.float32)
        long = length[idx].astype(np.float32)
        from_tail, to_head = u * long, (1.0 - u) * long
        if soft:  # the even streak blurred along its length by the same tent as across: soft ends
            blur = width[idx].astype(np.float32)
            ends = np.maximum(_tent(from_tail / blur) + _tent(to_head / blur) - 1.0, 0.0)
            ends *= _tent((cut[idx] - mid) / blur)  # and fading into the ground, not cut by it
        else:  # round ends: the coverage of a disc as wide as the drop, coming in and going out
            r2 = 2.0 * reach[idx].astype(np.float32)
            ends = np.clip(0.5 + from_tail / r2, 0.0, 1.0) * np.clip(0.5 + to_head / r2, 0.0, 1.0)
        w = weight[idx].astype(np.float32) * (y_b - y_a) * ends
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
    VERSION = 4
    LIT = True
    PREVIEW_W = 1280  # its preview clip's width (engine.fx_clip_frame): thin streaks need the pixels
    # and its quality: 42 keeps 94% of medium rain's streaks (34: 97%) in two thirds the size; fine drizzle needs 34
    # (88%; 42: 80%), and its clip is small anyway (set per rain in __init__)
    PREVIEW_CRF = 42
    upgrade = staticmethod(upgrade)

    def __init__(self, *a, light=None):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 12))
        z = u[:, 0]  # 0 far .. 1 near
        ds = p["depth"] / 100
        near = np.exp(ds * 3.0 * (z - 0.5))  # perspective: how wide a drop looks and how fast it crosses the screen
        drop = u[:, 3]  # how big a drop is: big ones fall a little faster and lean less in the wind
        size = p["size"] / 100
        fine = ((0.15 + size) / 0.5) ** 0.2  # and rain of big drops falls a little faster (1 at size 35)
        sp, ln = p["speed"] / 100, p["length"] / 100
        self.mean_speed = (0.4 + 3.2 * sp - 0.8 * sp * sp) * fine  # frame heights a second (2.03 at 60)
        speed = self.mean_speed * near * (0.85 + 0.3 * drop)  # frame heights a second
        self.angle = math.radians(30) * p["wind"] / 100
        angle = np.arctan(math.tan(self.angle) * (1.3 - 0.55 * drop)) + np.radians(2) * (u[:, 7] - 0.5)
        self.shutter = (0.08 + 0.74 * ln + 0.3 * ln * ln) / self.fps  # seconds of motion in a streak (0.525 at 50)

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
        wander = 0.012 * (1 - drop) * (1 - size) ** 3 * near ** 0.5  # frame heights
        reach = float(np.abs(gust).max()) * 1.3  # how far a gust can carry a drop sideways while it is in view
        reach += float(wander.max())

        # how wide and how strong: a drop thinner than THIN is drawn that wide, as faint as it then is (the light of
        # a far drop spread by the lens and the sensor: an even line, not a flicker between pixels)
        width = _lerp(0.8, 3.6, size) * near * (0.6 + 0.6 * drop + 0.3 * u[:, 4]) * px
        drawn = np.maximum(width, 0.8 * px)
        v = p["variety"] / 100
        glint = 0.3 + 1.2 * u[:, 5] ** 2  # how much light a drop catches
        opacity = p["opacity"] / 100 * STRENGTH * near ** 0.4 * ((1 - v) * 0.8 + v * glint) * (width / drawn)
        blur = p["blur"] / 100  # the nearest drops out of focus, more of them the more blur
        defocus = min(1.0, 2.0 * blur) * _smooth(_lerp(0.88, 0.55, blur), 1.0, z)
        wide = drawn * (1 + 0.8 * defocus) + 6.0 * defocus * px  # out of focus: spread out

        longest = self.shutter * float(speed.max()) * 1.3
        past = float(wide.max()) / self.h  # (how far a blurred drop reaches past its ends)
        my, mx = longest + 0.02 + past, longest + 0.04 + reach + past
        share = self.paths(u[:, 1], u[:, 2], speed, angle, my, mx)
        drift = (1 + 2 * my) * np.tan(angle)  # each pass in its own lane, anywhere the path can be (paths())
        lo = np.minimum(0.0, -drift) - mx
        room = np.maximum(self.aspect, self.aspect - drift) + mx - lo
        wanted = p["amount"] / 100 * 2600 * self.aspect / (16 / 9) / float(self.density.mean())
        g = p["ground"] / 100
        hit = np.where(g < 1, g + (1 - g) * 1.3 * z ** 0.8, 9.0)  # where on the ground a drop lands (frame heights)
        kept = self.choose(share, wanted, {
            "z": z, "width": drawn, "opacity": np.minimum(1.0, opacity), "defocus": defocus, "wide": wide,
            "hit": hit, "gate": u[:, 8], "wander": wander,
            "wm": self.cycles(_lerp(0.25, 0.8, uw[:, 0])), "wphi": uw[:, 1] * 2 * math.pi,
            "lane0": u[:, 1] + u[:, 6], "lo": lo, "room": room})
        self.__dict__.update(kept)
        self.n = len(self.z)
        self.blurred = self.defocus > 0.02
        # each pass in its own lane: k lanes evenly over the room, the next one m / k on (m and k with no common
        # factor: every lane once a loop, and as far on from the last pass to the first)
        k = self.k.astype(np.int64)
        m = np.maximum(1, np.rint(LANE * k)).astype(np.int64)
        while True:
            bad = np.gcd(m, k) != 1
            if not bad.any():
                break
            m[bad] += 1
        self.step = m / k

        # splashes and rings where drops land on the ground: far more of them than the streaks drawn (_impacts())
        self.ground = g
        self.splash, self.ripple = p["splash"] / 100, p["ripple"] / 100
        self.hits = self._impacts(p, g) if g < 1 and (self.splash > 0 or self.ripple > 0) else None

        # sheets of rain blown along (two drifting patterns, so their shapes change), mist, spray over the ground
        vrng = np.random.default_rng([self.seed, 1103, 3])
        self.sheet_a = _noise(vrng, 64, 128, 1.5, 9.0)  # 2 x 4 frame heights
        self.sheet_b = _noise(vrng, 64, 128, 1.2, 7.0)
        self.haze = _noise(vrng, 32, 32, 1.3, 1.3)  # 4 x 4 frame heights
        amount_k = min(1.0, p["amount"] / 25)
        self.veil, self.mist = 0.14 * p["veil"] / 100 * amount_k, 0.12 * p["mist"] / 100 * amount_k
        self.spray = 0.1 * self.splash * float(np.clip((p["amount"] - 50) / 40, 0, 1)) if g < 1 else 0.0
        self.clumps = p["veil"] / 100
        drift = (0.06 + 0.3 * abs(p["wind"]) / 100) * self.loop_s  # frame heights a loop
        down = self.mean_speed * self.loop_s
        gusted = down * abs(self.gust_mean)  # the gusts' steady share (`carried` leaves it out)
        # how far each pattern moves in a loop, frame heights: whole tiles (4 wide, 2 tall), so it loops
        self.move_a = (side * max(1, math.ceil((drift + gusted) / 4)) * 4, max(1, round(0.15 * down / 2)) * 2)
        self.move_b = (side * max(1, math.ceil((drift * 1.6 + gusted) / 4)) * 4, max(1, round(0.1 * down / 2)) * 2)
        self.move_h = side * max(1, round(drift * 0.4 / 4)) * 4
        # the veil lies only where the pattern is densest (its top quarter), so the air between sheets stays clear
        gx, gy = np.meshgrid(np.linspace(0, self.aspect, 64), np.linspace(0, 1, 36))
        probe = np.concatenate([self.pattern(tau, gx, gy).ravel() for tau in np.arange(12) / 12])
        self.sheet_lo, self.sheet_hi = (float(q) for q in np.percentile(probe, (75, 99.5)))

        # lightning
        self.flash = p["lightning"] / 100
        lrng = np.random.default_rng([self.seed, 1103, 4])
        nf = max(1, round(p["flashes"] * self.loop_s / 30))
        self.flash_t = (np.arange(nf) + 0.1 + 0.8 * lrng.random(nf)) / nf * self.loop_s
        self.flash_parts = [((0.0, 1.0, 0.07), (0.08 + 0.06 * r1, 0.5 + 0.4 * r2, 0.06), (0.25 + 0.1 * r3, 0.3, 0.12))
                            for r1, r2, r3 in lrng.random((nf, 3))]
        self.flash_side = lrng.random(nf) * 2 - 1
        self.flash_colour = np.array([0.92, 0.95, 1.0], np.float32)

        # colour: its own, or lit by the scene (the light map, made into colours on first use: _colours())
        self.colour = _rgb(p["color"])
        self.light = light
        self.lit = p["lit"] / 100 if light is not None else 0.0
        self.maps = None
        if p["size"] < 25:
            self.PREVIEW_CRF = 34
        self.edge = p["edge"] / 100 * (1 - self.lit) ** 2  # (lit: against a bright sky a drop is darker already)
        if self.edge < 0.04:  # (too faint to see: not drawn)
            self.edge = 0.0

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

    def pattern(self, tau, x, y):
        """The two drifting patterns the sheets of rain are made of, added (0..2), at x, y (frame heights)."""
        slope = math.tan(self.angle) + float(_at(self.gusting, tau)) / math.cos(self.angle)
        push = self.mean_speed * self.loop_s * float(_at(self.carried, tau))
        across = x - (y - 0.5) * slope - push  # (a gust carries the sheets with the wind, as it does the drops)
        out = 0
        for tex, (mx, my) in ((self.sheet_a, self.move_a), (self.sheet_b, self.move_b)):
            h, w = tex.shape
            out = out + _sample(tex, (across - mx * tau) * (w / 4), (y - my * tau) * (h / 2))
        return out

    def sheets(self, tau, x, y):
        """How dense the rain is in the sheets (0..1) at x, y (frame heights): the drops there are brighter."""
        return _smooth(0.7, 1.5, self.pattern(tau, x, y))

    def lanes(self, tau, k=slice(None)):
        """Where (x, frame heights) the paths of drops `k` start at tau: each time a drop comes down (pass j of its k a
        loop) its lane moves on, so it falls somewhere else; the same pass a loop later, the same lane."""
        x = self.s0[k] + self.k[k] * self.len[k] * tau
        j = np.mod(np.rint((x - np.mod(x, self.len[k])) / self.len[k]), self.k[k])  # (as drops() finds where it is)
        return self.lo[k] + np.mod(self.lane0[k] + self.step[k] * j, 1.0) * self.room[k]

    def drops(self, t):
        """Every drop's head (x, y, px), the way it moves (unit vector), streak length (px), where it is on its
        path, and whether it shows."""
        tau = self.tau(t)
        s = np.mod(self.s0 + self.k * self.len * tau, self.len)
        d = s - self.extra
        x = self.lanes(tau) + d * self.sin
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

    # --- drawing ----------------------------------------------------------------------------------------------

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
        on_ground = self.hits is not None
        soft = Layer(*soft_size) if self.blurred.any() or (on_ground and self.px >= 1.5) else None
        shade = Layer(*soft_size) if self.edge > 0 or (on_ground and self.ripple > 0) else None
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
        if self.edge > 0:  # a soft dark rim either side of each streak, under it
            rim = width + 1.2 * self.px
            shade.streaks(tx / 2, ty / 2, hx / 2, hy / 2, rim / 2, glow * self.edge * 0.45, cut / 2, soft=True)
        if on_ground:
            self._splashes(tau, sharp, soft, shade)
        haze = self._veil(tau) if self.veil + self.mist + self.spray > 0 else None
        self._lay(canvas, mask, sharp, soft, shade, haze, flash)

    def _colours(self):
        """The colour of the drops, and of the sheets and mist: their own (3,), or, lit by the scene, maps of what the
        light around them makes them (sRGB, uint8 planes (3, h, w)), and how much more of a drop shows where that light
        is brighter than white (a drop by a lamp flashes: more coverage; None if nowhere). Worked out at the light
        map's size (it changes slowly), each channel clipped at white as a camera does (a light far brighter than white
        is white, not darkened). Made once; release() lets them go."""
        if self.maps is None:
            if self.lit <= 0:
                self.maps = (self.colour, self.colour, None)
            else:
                own = linear(self.colour)
                tint = own / NEUTRAL  # (its colour tints the light it shows)
                e, here = self.light[..., :3] * tint, self.light[..., 3:] * tint
                lit = self.lit
                # a drop before something darker than the light around it shows more of that light (lit from
                # beside or behind: by a lamp, a window, the sky), most at night; it flashes (more of it shows) only
                # where the light itself is brighter than white
                keep = (1 - lit) ** 3 * min(1.0, (0.15 + self.p["size"] / 100) / 0.5)  # (small drops: less)
                drops = keep * own + ((1 - keep) * DROP_GAIN) * e
                more = np.minimum(drops, 1.0)
                boost = np.clip((drops @ LUMA) / np.maximum(more @ LUMA, 1e-6), 1.0, BOOST)
                night = float(_smooth(0.16, 0.08, float((here @ LUMA).mean())))
                drops += ((1 - keep) * DROP_GAIN * LIT_SIDE * (0.4 + 0.6 * night)) * np.maximum(e - here, 0.0)
                shown = np.minimum(drops, 1.0, out=more)
                del drops, here
                keep = (1 - lit) ** 4  # (mist: little light of its own)
                mists = keep * own + ((1 - keep) * MIST_GAIN) * e
                np.minimum(mists, 1.0, out=mists)
                planes = [self._big(srgb(c)) for c in (shown, mists)]
                more = None
                if boost.max() > 1.0 + 1 / 255:
                    more = self._big(((boost - 1.0) / (BOOST - 1.0))[..., None])[0]
                self.maps = (planes[0], planes[1], more)
        return self.maps

    def _big(self, c):
        """A small map (h, w, n) of 0..1 values as uint8 planes (n, h, w) of the layer's size (bilinear; made in 8
        bits from the start: no full-size floats)."""
        planes = np.empty((c.shape[2], self.h, self.w), np.uint8)
        small = np.clip(c * 255.0 + 0.5, 0, 255).astype(np.uint8)
        for k in range(c.shape[2]):
            planes[k] = np.asarray(Image.fromarray(np.ascontiguousarray(small[..., k]), "L").resize(
                (self.w, self.h), Image.Resampling.BILINEAR))
        return planes

    def release(self):
        """Lets the colour maps go while the clip is not showing (made again when it shows)."""
        self.maps = None

    def _lay(self, canvas, mask, sharp, soft, shade, haze, flash):
        """The layers over the picture (dark rims under the drops, the sheets and mist under both), a band of rows at a
        time; half-size layers only where they hold something."""
        top = sharp.view()
        half = soft.view() if soft is not None else None
        dark = shade.view() if shade is not None else None
        drops, mists, more = self._colours()
        lift = None
        if flash > 0.002 and self.lit > 0:  # lit by the scene: the lightning lights the drops too
            lift = (0.6 * self.lit * self.flash * flash) * self.flash_colour

        def rows(a, r0, r1, smooth=False):  # rows r0..r1-1 of a half-size layer, full size; None if empty there
            s0 = r0 // 2
            if a is None or not a[s0:(r1 + 1) // 2].any():
                return None
            if smooth:  # (sheets and mist change slowly: each of their pixels as 2 x 2 is plenty, and quicker)
                out = np.repeat(np.repeat(a[s0:(r1 + 1) // 2], 2, 0), 2, 1)[r0 - 2 * s0:r1 - 2 * s0, :self.w]
            else:
                out = _up2(a, r0, r1, self.w)
            return np.minimum(out, 1.0, out=out)

        for r0 in range(0, self.h, BAND):
            r1 = min(r0 + BAND, self.h)
            cover = rows(half, r0, r1)
            if cover is None:
                cover = np.minimum(top[r0:r1], 1.0)
            else:
                cover += top[r0:r1]
                np.minimum(cover, 1.0, out=cover)
            gain = None
            if more is not None and more[r0:r1].any():  # (light brighter than white: more of the drop shows)
                gain = more[r0:r1] * ((BOOST - 1.0) / 255.0)
                gain += 1.0
                cover *= gain
                np.minimum(cover, 1.0, out=cover)
            prgb = _tint(cover, drops, r0, lift)
            alpha = cover
            rim, mist = rows(dark, r0, r1), rows(haze, r0, r1, smooth=True)
            if rim is not None:  # (water bends the light away at a drop's edges: dark)
                rim *= 1.0 - cover
                alpha = cover + rim
            if mist is not None:
                mist *= 1.0 - alpha
                _tint(mist, mists, r0, lift, into=prgb)
                alpha = alpha + mist
            canvas.over_rgba(0, r0, prgb, alpha, mask)

    def _impacts(self, p, g):
        """Drops landing on the ground in view, apart from the streaks drawn (they are a few of the drops that fall):
        places that each see a landing about once a second, k whole times a loop, each time somewhere else (a
        low-discrepancy step on from the last), so frame P is frame 0 again. The heavier the rain, the more; the
        bigger its drops, the more and bigger the splashes."""
        big = min(1.0, (0.15 + p["size"] / 100) / 0.5)
        rate = IMPACTS * (p["amount"] / 100) ** 1.5 * big ** 1.5 * (1 - g) / 0.55 * self.aspect / (16 / 9)
        n = int(min(MOST_IMPACTS, round(rate)))
        if n == 0:
            return None
        u = np.random.default_rng([self.seed, 1103, 6]).random((n, 8))
        k = self.cycles(1.0 / (0.75 + 0.5 * u[:, 0]))  # landings a loop
        return {"k": k, "phase": u[:, 1], "u": u[:, 2], "v": u[:, 3], "gate": u[:, 4], "r": u[:, 5:8].T,
                "big": 0.55 + 0.45 * big}

    def _splashes(self, tau, sharp, soft, shade):
        """Water thrown up where drops land, and rings spreading on the wet ground. Drawn where a pixel is about a
        pixel of a 1080-line frame (the half-size layer from 4K up), so they look alike at every size."""
        hits, g = self.hits, self.ground
        q = tau * hits["k"] + hits["phase"]  # landings so far
        j = np.floor(q)
        period = self.loop_s / hits["k"]
        age = (q - j) * period  # seconds since the last one
        i = np.flatnonzero(age < np.minimum(RING_S, 0.9 * period))  # (as long as the longest part lasts: a ring)
        if not len(i):
            return
        j, a = np.mod(j[i], hits["k"][i]), age[i]

        def each(r, step):  # a new draw each landing (the same one a loop later)
            return np.mod(r + step * j, 1.0)
        landed = _at(self.density, tau - a / self.loop_s) > each(hits["gate"][i], 0.6180339887)
        i, j, a = i[landed], j[landed], a[landed]
        u = each(hits["u"][i], 0.7548776662)
        v = each(hits["v"][i], 0.5698402910)
        su, sv, sw = (each(r[i], step) for r, step in zip(hits["r"], (0.4142135624, 0.7320508076, 0.2360679775)))
        y = g + (1 - g) * v ** 1.4  # (more far off: more ground in each pixel there)
        close = (y - g) / max(1 - g, 1e-6)  # 0 at the far edge of the ground
        layer, f = (soft, 0.5) if soft is not None and self.px >= 1.5 else (sharp, 1.0)
        x = (-0.02 + u * (self.aspect + 0.04)) * self.h * f
        y = y * self.h * f
        size = (0.15 + 0.85 * close) * hits["big"] * self.px * f  # this layer's px for 1 at a 1080 frame's bottom
        o = 0.6 + 0.4 * sw  # (how much light each catches)
        dur = np.minimum(RING_S, 0.9 * period[i])
        if self.splash > 0:
            self._crowns(layer, a, x, y, size, o, su, sv, sw, period[i])
        if self.ripple > 0:
            self._rings(layer, shade, 0.5 / f, a, x, y, size, o, sw, close, dur)

    def _rings(self, layer, dark, f, a, x, y, size, o, sw, close, dur):
        """Rings spreading on the wet ground, a faint one inside: soft lines that catch the light on their far side
        (the near side of a ripple faces down, away from the sky: a little darker, on `dark`, f times this layer's
        size)."""
        k = np.flatnonzero(a < dur)
        if not len(k):
            return
        f_age = a[k] / dur[k]
        flat = 0.2 + 0.25 * close[k]
        reach = size[k] * (0.7 + 0.6 * sw[k])
        line = 1.8 * size[k]  # how thick the line is, px
        xs, ys, ws, ds = [], [], [], []
        for lag, strength in ((0.0, 1.0), (0.3, 0.35)):
            g = np.clip((f_age - lag) / (1 - lag), 0, 1)
            rx = (2.0 + 16.0 * np.sqrt(g)) * reach
            ry = rx * flat
            around = 2 * math.pi * np.sqrt(0.5 * (rx * rx + ry * ry))
            n = np.where(g > 0, np.maximum(16, np.ceil(around / 1.1)), 0).astype(np.intp)  # a point every 1.1 px
            ring = np.repeat(np.arange(len(n)), n)
            th = (np.arange(len(ring)) - np.repeat(np.cumsum(n) - n, n)) * (2 * math.pi) / n[ring]
            sin = np.sin(th)
            cover = 1.2 * self.ripple * o[k] * strength * (1 - g) ** 1.5  # at its brightest, on the far side
            each = (cover * line * around / np.maximum(n, 1))[ring]
            xs.append(x[k][ring] + rx[ring] * np.cos(th))
            ys.append(y[k][ring] + ry[ring] * sin)
            ws.append(each * (0.2 + 0.8 * np.clip(-sin, 0, 1)))
            ds.append(each * 0.3 * np.clip(sin, 0, 1))
        xs, ys = np.concatenate(xs), np.concatenate(ys)
        layer.dots(xs, ys, np.concatenate(ws))
        if dark is not None:
            dark.dots(xs * f, ys * f, np.concatenate(ds) * (f * f))

    def _crowns(self, layer, a, x, y, size, o, su, sv, sw, period):
        """Droplets thrown up and out where drops land (a crown of them, rising 8 to 25 px near the camera in a
        1080-line frame), each the short line it draws while the shutter is open, and the drop breaking."""
        dur = np.minimum(0.12 + 0.08 * sw, 0.9 * period)  # in the air (over before the next landing there)
        live = a < dur
        if not live.any():
            return
        a, x, y, size, o, dur, su, sv = a[live], x[live], y[live], size[live], o[live], dur[live], su[live], sv[live]
        fade = np.sqrt(1 - a / dur)
        fast = 700 * float(size.max())  # px a second, the fastest droplet
        m = int(np.clip(np.ceil(fast * self.shutter / 0.7) + 1, 2, 12))  # points along each line, 0.7 px apart
        for j in range(8):  # (how each droplet flies: its own mix of the landing's random draws)
            r1, r2 = (su * (2.618 * j + 1.3) + sv * (1.618 * j + 0.7)) % 1.0, (sv * (j + 3) + 0.37 * su) % 1.0
            ang = 1.2 * (r1 - 0.5) + 0.12 * (j - 3.5)  # from straight up, within 0.65 or so
            speed = (220 + 480 * r2) * size  # px a second
            rise, side = speed * np.cos(ang), speed * np.sin(ang)
            fall = 2 * rise / dur  # back on the ground at the end
            w = self.splash * o * 1.6 * size ** 2 * fade / m
            for back in np.linspace(0, 1, m):  # where it was while the shutter was open
                b = np.maximum(a - back * self.shutter, 0)
                layer.dots(x + side * b, y - np.maximum(rise * b - 0.5 * fall * b * b, 0), w)
        first = a < 0.045  # the drop breaking on the ground: a short bright flat spot
        if first.any():
            w = (self.splash * o * 1.8 * size ** 2)[first]
            for dx in (-0.7, 0.0, 0.7):
                layer.dots(x[first] + dx * size[first], y[first] - 0.4 * size[first], w)

    def _veil(self, tau):
        """Sheets of rain, a light mist, and spray over the ground in heavy rain: worked out at an eighth of the size,
        their coverage on the half-size grid."""
        w8, h8 = (self.w + 7) // 8, (self.h + 7) // 8
        x = ((np.arange(w8) + 0.5) * (self.w / w8) / self.h)[None, :]  # frame heights
        y = ((np.arange(h8) + 0.5) * (self.h / h8) / self.h)[:, None]
        dens = float(_at(self.density, tau))
        a = np.zeros((h8, w8), np.float32)
        if self.veil > 0:
            a += self.veil * (0.6 + 0.4 * dens) * _smooth(self.sheet_lo, self.sheet_hi, self.pattern(tau, x, y))
        hh, hw = self.haze.shape
        if self.mist > 0 or self.spray > 0:
            haze = _sample(self.haze, (x - self.move_h * tau) * (hw / 4), y * (hh / 4))
            if self.mist > 0:
                a += self.mist * (0.5 + haze) * (0.8 + 0.2 * dens)
            if self.spray > 0:  # drops break into spray: a low mist over the ground, thickest far off (along it)
                g = self.ground
                band = np.exp(-((y - g - 0.01) / 0.035) ** 2) + 0.3 * _smooth(g, g + 0.05, y)
                a += self.spray * band * (0.5 + haze) * (0.6 + 0.4 * dens)
        small = Image.fromarray(a, "F").resize(((self.w + 1) // 2, (self.h + 1) // 2), Image.Resampling.BILINEAR)
        return np.asarray(small)
