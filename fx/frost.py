"""Frost and snow on a window pane: ice ferns grown in from the edges, fine rime, the milky haze of frosted glass,
snow lying along the bottom edge, snowflakes that land on the glass, stay and melt, and glints on the ice (an effect
laid over the picture; see effects.py for how every effect loops).

Frost grows where the glass is coldest: from its edges, most in the corners and along the bottom. The ice ferns
(feathers of ice) start there, each a slightly wandering stem with side branches at sixty degrees (ice is
hexagonal), longest near the stem's root and shorter towards its tip, and small twigs on the longer branches. They
are made once (in frame heights, so a small preview clip and the video show the same ferns) and drawn as thin lines:
points along every line laid on the frame's pixels (each one's share of its line's ink), so a thin line is soft and
a thick crossing of many is solid white. Each point knows how far along its fern's growth it lies, and where the
frost creeps (out and back once a loop: "creep") only the ferns' parts grown so far show. Rime and haze lie on a band
round the edges whose inner border wanders (smooth noise) and moves with the creeping.

Snow lying on the bottom edge is a static band, lumpy along its top and piled up in the corners. Flakes land, each on
its own clock (a whole number of times per loop), at a new place every time it lands (passes: the k places of a slot
come round once a loop), stay a while and melt. Glints flash on the ice now and then. Sizes are in pixels of a
1080-line frame.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import sprites
from .effects import Effect, _choice, _colour, _lerp, _num, _rgb, blur

DEFAULTS = {"cover": 35, "corner": 50, "ferns": 60, "rime": 45, "veil": 30, "creep": 25, "sill": 35, "amount": 40,
            "flake_size": 40, "melt": 50, "sparkle": 40, "opacity": 90}
LOOKS = {
    "edges": {},
    "corners": {"cover": 45, "corner": 100, "ferns": 70, "rime": 65, "veil": 35, "sill": 25},
    "heavy": {"cover": 70, "corner": 60, "ferns": 85, "rime": 70, "veil": 45, "sill": 60, "amount": 55},
    "light": {"cover": 22, "corner": 40, "ferns": 35, "rime": 30, "veil": 15, "sill": 0, "amount": 30},
}
LOOKS = {k: {**DEFAULTS, **v} for k, v in LOOKS.items()}

SCHEMA = {"label": "Tuyết bám kính", "params": [
    _choice("look", "Kiểu", "edges", [("edges", "Băng bám quanh viền kính"), ("corners", "Băng dày ở các góc"),
                                      ("heavy", "Kính đóng băng dày"), ("light", "Sương giá mỏng")],
            presets=LOOKS, group="Kiểu"),
    _num("cover", "Băng lan vào giữa kính", DEFAULTS["cover"], group="Băng giá"),
    _num("corner", "Dày ở các góc", DEFAULTS["corner"], group="Băng giá"),
    _num("ferns", "Hoa băng (vân như lông chim)", DEFAULTS["ferns"], group="Băng giá"),
    _num("rime", "Sương muối li ti", DEFAULTS["rime"], group="Băng giá"),
    _num("veil", "Mờ đục như kính mờ", DEFAULTS["veil"], group="Băng giá"),
    _num("creep", "Băng lan ra, thu vào", DEFAULTS["creep"], group="Băng giá"),
    _num("sill", "Tuyết đọng mép dưới", DEFAULTS["sill"], group="Tuyết"),
    _num("amount", "Bông tuyết đậu lên kính", DEFAULTS["amount"], group="Tuyết"),
    _num("flake_size", "Cỡ bông tuyết", DEFAULTS["flake_size"], group="Tuyết"),
    _num("melt", "Tan nhanh", DEFAULTS["melt"], group="Tuyết"),
    _num("sparkle", "Lấp lánh", DEFAULTS["sparkle"], group="Màu"),
    _num("opacity", "Độ đậm", DEFAULTS["opacity"], group="Màu"),
    _colour("color", "Màu băng tuyết", "#eef5ff", group="Màu")]}

STEP = 0.0026  # a fern stem's step (frame heights)
SPACING = 0.4  # pixels between the points a line is drawn with
INK = 1.6  # how much a line of ink a pixel wide covers (1 - e^-INK)
FERNS = 900  # ferns with fixed random draws (how many grow follows the frame's edges and "ferns")
FLAKES = 160  # flake slots
PLACES = 64  # the places a slot lands at, at most (a slot lands k times a loop, each pass at its own place)
GLINTS = 300
_MIN3 = ImageFilter.MinFilter(3)  # (the growth that reaches a pixel spread as far as its line's ink)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _flake(key, fx, fy):
    """A star of ice: six arms with two pairs of twigs each and a small plate in the middle (key: radius px, turn),
    drawn four times as fine and scaled down."""
    r, turn = key
    ss = 4
    half = int(math.ceil(r + 1.5))
    side = 2 * half + 1
    im = Image.new("L", (side * ss, side * ss), 0)
    d = ImageDraw.Draw(im)
    cx, cy = (half + fx) * ss, (half + fy) * ss
    wide = max(1, round(max(0.55, 0.12 * r) * ss))
    rs = r * ss
    for k in range(6):
        th = turn + k * math.pi / 3
        c, s = math.cos(th), math.sin(th)
        d.line([(cx, cy), (cx + rs * c, cy + rs * s)], fill=255, width=wide)
        for at, length in ((0.42, 0.36), (0.7, 0.24)):
            bx, by = cx + at * rs * c, cy + at * rs * s
            for side_ in (-1, 1):
                t2 = th + side_ * math.pi / 3
                d.line([(bx, by), (bx + length * rs * math.cos(t2), by + length * rs * math.sin(t2))], fill=255,
                       width=max(1, wide * 2 // 3))
    plate = 0.16 * rs
    d.regular_polygon((cx, cy, max(plate, ss * 0.5)), 6, rotation=math.degrees(turn), fill=255)
    a = np.asarray(im.resize((side, side), Image.Resampling.BOX), np.float32) / 255.0
    return a, -half, -half


def _glint(length, fx, fy):
    """A glint: a tiny bright point and four thin rays."""
    x0, y0, xs, ys = sprites._grid(length + 1, length + 1, fx, fy)
    core = np.exp(-(xs * xs + ys * ys) / 0.8)
    across = np.exp(-ys * ys / 0.4) * np.clip(1 - np.abs(xs) / length, 0, 1) ** 2
    down = np.exp(-xs * xs / 0.4) * np.clip(1 - np.abs(ys) / length, 0, 1) ** 2
    return np.minimum(1.0, core + np.maximum(across, down)).astype(np.float32), x0, y0


class Frost(Effect):
    VERSION = 1

    PREVIEW = 1280  # its preview clip: as many pixels as 1280 x 720 (the ferns' fine lines blur at 640 wide)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        self.colour = _rgb(p["color"])
        self.opacity = p["opacity"] / 100
        self.reach = _lerp(0.05, 0.5, p["cover"] / 100)  # how far in the frost lies (frame heights)
        self.creep = p["creep"] / 100 * 0.35  # how far back its growth goes halfway through the loop
        self._segments()
        self._glints()
        self._flakes()
        self.field = self.frost = self.sill = None  # (made when a frame is drawn, freed by release())
        self.sprites = (sprites.SpriteCache(lambda key, fx, fy: _flake(key, fx, fy)),
                        sprites.SpriteCache(lambda key, fx, fy: _glint(key, fx, fy)))

    def _rng(self, k):
        return np.random.default_rng([self.seed, 1103, k])

    # --- the ferns -----------------------------------------------------------------------------------------------

    def _segments(self):
        """Every fern as straight segments (frame heights): start x, y, angle, length, the growth there (0..1 of its
        fern's), how much growth along it per frame height, width (1080-line px)."""
        p, a, R = self.p, self.aspect, self.reach
        u = self._rng(1).random((FERNS, 8))
        n = int(round(p["ferns"] / 100 * 26 * 2 * (a + 1) * _lerp(0.6, 1.4, p["cover"] / 100)))
        corner = p["corner"] / 100
        cols = {k: [] for k in ("x", "y", "th", "len", "grow", "per", "w")}

        def add(x, y, th, length, grow, per, w):
            for k, v in zip(cols, (x, y, th, length, grow, per, w)):
                cols[k].append(np.broadcast_to(np.asarray(v, np.float64), np.shape(length)).ravel())

        for i in range(min(n, FERNS)):
            ui = u[i]
            # its root: an edge (the bottom most), more of them in the corners
            edge = np.searchsorted(np.cumsum([0.34, 0.22, 0.22, 0.22]), ui[0] * 0.999, side="right")
            along = ui[1]
            if ui[2] < 0.9 * corner ** 1.5:
                along = 0.5 * ui[1] ** 2.5 if ui[3] < 0.5 else 1 - 0.5 * ui[1] ** 2.5
            inset = R * 0.25 * ui[4] ** 2
            x, y, th = ((along * a, 1 - inset, -math.pi / 2), (inset, along, 0.0), (a - inset, along, math.pi),
                        (along * a, inset, math.pi / 2))[edge]
            length = R * _lerp(0.3, 1.1, ui[6]) * (1.25 if edge == 0 else 1.0)
            full = length * 1.5  # (its growth: the path from its root over about its longest one)
            self._stem(self._rng(100 + i), add, x, y, th + (ui[5] - 0.5) * 2.6, length, 0.0, full,
                       _lerp(0.55, 1.0, ui[7]), 0)
        self.seg = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}

    def _stem(self, r, add, x, y, th, length, grow, full, ink, depth):
        """One stem of a fern from (x, y) at angle th: a slightly wandering line with barbs at about fifty degrees
        (a feather: short at its root and tip, longest a third of the way, more on one side now and then), twigs on
        the longer barbs, and a fork or two that grow on as stems of their own."""
        steps = max(2, int(length / STEP))
        turn = th + np.cumsum(r.normal(0, 0.06, steps)) + np.arange(steps) * r.normal(0, 0.005)
        sx = x + np.r_[0, np.cumsum(np.cos(turn[:-1]))] * STEP
        sy = y + np.r_[0, np.cumsum(np.sin(turn[:-1]))] * STEP
        s = np.arange(steps) * STEP
        q = s / length
        add(sx, sy, turn, np.full(steps, STEP), grow + s / full, 1 / full, ink * (1.3 - 0.5 * q) * 0.85 ** depth)
        # barbs
        k = np.repeat(np.arange(1, steps), 2)
        side = np.tile([-1.0, 1.0], steps - 1)
        weak = r.random() < 0.4  # (a one-sided feather)
        keep = r.random(len(k)) < np.where((side > 0) & weak, 0.25, 0.75)
        k, side = k[keep], side[keep]
        envelope = np.minimum(1, q[k] / 0.25) ** 0.5 * (1 - q[k]) ** 0.8
        blen = length * _lerp(0.15, 0.4, r.random()) * envelope * (0.55 + 0.9 * r.random(len(k)))
        bth = turn[k] + side * (math.radians(50) + r.normal(0, 0.12, len(k)))
        add(sx[k], sy[k], bth, blen, grow + s[k] / full, 1 / full, 0.8 * ink)
        # twigs on the longer barbs, every two steps, both sides
        m = np.maximum(0, (blen / (2 * STEP)).astype(int) - 1)
        j = np.repeat(np.arange(len(k)), m * 2)
        if len(j):
            at = np.concatenate([np.repeat(np.arange(1, mi + 1), 2) for mi in m if mi]) * 2 * STEP
            tside = np.tile([-1.0, 1.0], len(j) // 2)
            tl = (blen[j] - at) * _lerp(0.2, 0.45, r.random(len(j)))
            tth = bth[j] + tside * (math.radians(55) + r.normal(0, 0.12, len(j)))
            add(sx[k][j] + np.cos(bth[j]) * at, sy[k][j] + np.sin(bth[j]) * at, tth, tl,
                grow + (s[k][j] + at) / full, 1 / full, 0.55 * ink)
        # forks
        if depth < 2:
            for _ in range(min(3, r.poisson(1.3 * length / self.reach))):
                f = int(r.uniform(0.15, 0.7) * steps)
                rest = length - s[f]
                self._stem(r, add, sx[f], sy[f], turn[f] + r.choice((-1, 1)) * r.uniform(0.4, 0.8),
                           rest * r.uniform(0.5, 0.9), grow + s[f] / full, full, ink * 0.9, depth + 1)

    def _ink(self):
        """The ferns on the frame's pixels: how much ink each pixel holds, and the earliest growth (0..1) that
        reaches it (2 where none does)."""
        h, w, seg = self.h, self.w, self.seg
        ink = np.zeros(h * w, np.float64)
        first = np.full(h * w, 2.0, np.float32)
        if not len(seg["len"]):
            return ink.reshape(h, w).astype(np.float32), first.reshape(h, w)
        lpx = seg["len"] * h
        count = np.maximum(1, np.ceil(lpx / SPACING).astype(np.int64))
        for lo in range(0, len(count), 20000):  # (in batches: a few million points at most at a time)
            c = count[lo:lo + 20000]
            idx = np.repeat(np.arange(lo, lo + len(c)), c)
            off = np.arange(len(idx)) - np.repeat(np.cumsum(c) - c, c)
            t = (off + 0.5) / c[idx - lo]
            d = t * seg["len"][idx]
            x = (seg["x"][idx] + np.cos(seg["th"][idx]) * d) * h - 0.5
            y = (seg["y"][idx] + np.sin(seg["th"][idx]) * d) * h - 0.5
            grow = (seg["grow"][idx] + d * seg["per"][idx]).astype(np.float32)
            mass = lpx[idx] / c[idx - lo] * seg["w"][idx] * self.px
            ix, iy = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
            fx, fy = x - ix, y - iy
            for dx, dy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy),
                               (1, 1, fx * fy)):
                X, Y = ix + dx, iy + dy
                ok = (X >= 0) & (X < w) & (Y >= 0) & (Y < h)
                flat = Y[ok] * w + X[ok]
                ink += np.bincount(flat, weights=(mass * wt)[ok], minlength=h * w)
                np.minimum.at(first, flat, grow[ok])
        ink = blur(ink.reshape(h, w), 0.5 * max(1.0, self.px))  # (a little soft: ice lines are not drawn with a pen)
        first = np.asarray(Image.fromarray(first.reshape(h, w), "F").filter(_MIN3), np.float32)
        return ink, first

    # --- the band of frost round the edges ---------------------------------------------------------------------

    def _noise(self, k, cells, size=None):
        """Smooth noise 0..1, `cells` per frame height, the same at any frame size."""
        nx = int(math.ceil(cells * self.aspect)) + 3
        grid = self._rng(k).random((int(cells) + 3, nx)).astype(np.float32)
        im = Image.fromarray(grid, "F").resize(size or (self.w, self.h), Image.Resampling.BICUBIC,
                                                box=(1, 1, 1 + cells * self.aspect, 1 + cells))
        return np.asarray(im, np.float32)

    def _field(self):
        """How far in each pixel lies from the edges, as a share of the frost's reach (frost where it is below 1, its
        inner border wandering), the rime's grain (0..1: fine crystals in clusters and patches) and the patches."""
        p, h, w, a = self.p, self.h, self.w, self.aspect
        y = ((np.arange(h, dtype=np.float32) + 0.5) / h)[:, None]
        x = ((np.arange(w, dtype=np.float32) + 0.5) / h)[None, :]
        dx = np.minimum(x, a - x)
        dy = np.minimum(y, (1 - y) / 1.35)  # (thicker along the bottom)
        k = 0.02 + 0.16 * p["corner"] / 100  # (a soft minimum: the corners thicker)
        d = -k * np.logaddexp(-dx / k, -dy / k)
        wander = 0.5 * self._noise(2, 2.5) + 0.3 * self._noise(3, 7.0) + 0.2 * self._noise(4, 18.0) - 0.5
        corner = p["corner"] / 100
        mid = _smooth(np.sqrt(dx * dx + dy * dy) / 0.5)  # (0 in a corner, 1 half a frame height from them)
        field = d / self.reach + 0.8 * wander + 1.3 * _smooth((corner - 0.4) / 0.6) * mid
        patches = self._noise(5, 40.0)
        grain = 0.45 * self._noise(6, 540.0) + 0.25 * self._noise(7, 120.0) + 0.3 * patches  # (fine crystals, in
        # clusters, in patches)
        return field.astype(np.float32), grain, patches

    def _frost_at(self, tau):
        """The frost (premultiplied colour, cover) at phase tau: it creeps back and out again once a loop."""
        if self.field is None:
            self.field, self.grain, self.patches = self._field()
            self.ink, self.first = self._ink()
        p = self.p
        grown = 1 - self.creep * (0.5 - 0.5 * math.cos(2 * math.pi * tau))
        if self.frost is not None and self.frost[0] == grown:
            return self.frost[1], self.frost[2]
        near = _smooth((1.0 - self.field / grown) / 0.7)  # 1 well inside the frost, 0 past its border
        ferns = 1 - np.exp(-INK * self.ink * np.clip((grown - self.first) / 0.04, 0, 1))
        ferns *= p["ferns"] / 100
        rime = near ** 2 * _smooth((self.grain - 0.4) / 0.4) * (p["rime"] / 100 * 0.65)
        veil = near * (0.5 + self.patches) * (p["veil"] / 100 * 0.4)
        cover = 1 - (1 - ferns) * (1 - rime) * (1 - veil)
        cover *= self.opacity
        shade = 0.9 + 0.1 * self.grain  # (ice catches the light unevenly)
        rgb = cover[..., None] * (shade[..., None] * self.colour)
        self.frost = (grown, rgb.astype(np.float32), cover.astype(np.float32))
        return self.frost[1], self.frost[2]

    # --- snow on the bottom edge -----------------------------------------------------------------------------

    def _sill(self):
        """The snow lying along the bottom (premultiplied colour, cover, first row), or None."""
        p, h, w, a = self.p, self.h, self.w, self.aspect
        hb = p["sill"] / 100 * 0.11
        if hb <= 0:
            return None
        x = (np.arange(w, dtype=np.float32) + 0.5) / h
        g = self._rng(5)
        bumps = np.interp(x, np.linspace(0, a, 40), g.random(40)) * 0.6 + np.interp(
            x, np.linspace(0, a, 140), g.random(140)) * 0.4 - 0.5
        pile = p["corner"] / 100 * (np.exp(-(x / 0.13) ** 2) + np.exp(-((a - x) / 0.13) ** 2))
        top = 1 - hb * (1 + 0.5 * bumps + 0.9 * pile)
        y0 = max(0, int(math.floor(float(top.min()) * h)) - 3)
        y = ((np.arange(y0, h, dtype=np.float32) + 0.5) / h)[:, None]
        edge = 1.2 / h
        cover = _smooth((y - top[None, :] + edge) / (2 * edge))
        depth = np.clip((y - top[None, :]) / np.maximum(hb, 1e-6), 0, 1)
        grain = self._noise(6, 260.0, (w, h))[y0:]
        shade = (1.0 - 0.16 * _smooth(depth / 0.8)) * (0.95 + 0.07 * grain)
        lip = 1 + 0.06 * np.exp(-(depth / 0.06) ** 2)
        colour = self.colour * np.array([0.97, 0.985, 1.0], np.float32)  # (a little blue in its shade)
        cover = (cover * self.opacity).astype(np.float32)
        rgb = cover[..., None] * np.minimum(1.0, (shade * lip)[..., None] * colour)
        return rgb.astype(np.float32), cover, y0

    # --- flakes and glints -----------------------------------------------------------------------------------

    def _flakes(self):
        p = self.p
        n = int(round(p["amount"] / 100 * 70 * self.aspect / (16 / 9)))
        u = self._rng(7).random((FLAKES, 8))[:n]
        self.places = self._rng(8).random((FLAKES, PLACES, 3))[:n]
        life = _lerp(14.0, 5.0, p["melt"] / 100) * (0.7 + 0.6 * u[:, 0])  # seconds from landing to gone
        self.k_fl = np.minimum(self.cycles(1 / (life * 1.15)), PLACES)  # (and a little while gone)
        self.ph_fl = u[:, 1]
        self.r_fl = _lerp(4.0, 16.0, p["flake_size"] / 100) * (0.6 + 0.8 * u[:, 2] ** 1.5) * self.px
        self.hold = _lerp(0.45, 0.65, u[:, 3])  # (of its time: landed and whole; then it melts)

    def _glints(self):
        """Glints flash on the ferns (and on the snow on the bottom edge): places, how often, when."""
        p, seg = self.p, self.seg
        n = int(round(p["sparkle"] / 100 * 120 * self.aspect / (16 / 9)))
        u = self._rng(9).random((GLINTS, 6))[:n]
        self.g_on_sill = u[:, 0] < (0.25 if p["sill"] > 0 else 0.0)
        if len(seg["len"]):
            pick = (u[:, 1] * len(seg["len"])).astype(int)
            d = u[:, 2] * seg["len"][pick]
            gx = seg["x"][pick] + np.cos(seg["th"][pick]) * d
            gy = seg["y"][pick] + np.sin(seg["th"][pick]) * d
            self.g_grow = seg["grow"][pick] + d * seg["per"][pick]
        else:
            gx, gy, self.g_grow = u[:, 1] * self.aspect, u[:, 2], np.full(n, 2.0)
        hb = p["sill"] / 100 * 0.11
        self.gx = np.where(self.g_on_sill, u[:, 1] * self.aspect, gx)
        self.gy = np.where(self.g_on_sill, 1 - hb * 0.4 * u[:, 2], gy)
        self.g_grow = np.where(self.g_on_sill, 0.0, self.g_grow)
        self.gq = self.cycles(_lerp(0.08, 0.3, u[:, 3]))
        self.gph = u[:, 4] * 2 * math.pi
        self.glen = _lerp(2.0, 6.0, u[:, 5]) * self.px

    def release(self):
        """Frees what is as big as the frame (made again when a frame is drawn)."""
        self.field = self.grain = self.patches = self.ink = self.first = self.frost = None
        self.sill = None

    def draw(self, t, canvas, mask):
        tau = self.tau(t)
        rgb, cover = self._frost_at(tau)
        canvas.over_rgba(0, 0, rgb, cover, mask)
        sill = self.sill
        if sill is None and self.p["sill"] > 0:
            sill = self.sill = self._sill()
        if sill is not None:
            canvas.over_rgba(0, sill[2], sill[0], sill[1], mask)
        flake, glint = self.sprites
        # flakes: land, stay, melt (shrinking and fading), gone a little while; a new place at every landing
        if len(self.k_fl):
            c = self.k_fl * tau + self.ph_fl
            n = np.floor(c).astype(int) % self.k_fl.astype(int)
            age = c - np.floor(c)
            land = _smooth(age / 0.04)
            melt = _smooth((age - self.hold) / (0.86 - self.hold))
            alpha = land * (1 - melt) * 0.9 * self.opacity
            for i in np.flatnonzero(alpha > 0.01):
                px, py, turn = self.places[i, n[i]]
                r = self.r_fl[i] * (1 - 0.45 * melt[i])
                if r < 0.6:
                    continue
                x, y = px * self.aspect * self.h, py * self.h
                ix, iy = math.floor(x), math.floor(y)
                key = (sprites.bucket(r), round(turn * 12) % 12 * math.pi / 36)
                spot, sx, sy = flake.get(key, x - ix, y - iy)
                canvas.over(ix + sx, iy + sy, spot, self.colour, float(alpha[i]), mask)
        # glints: a brief flash now and then, only on ice grown so far
        if len(self.gx):
            grown = 1 - self.creep * (0.5 - 0.5 * math.cos(2 * math.pi * tau))
            level = np.maximum(0.0, np.sin(2 * math.pi * self.gq * tau + self.gph)) ** 16
            level *= self.g_grow < grown
            white = np.ones(3, np.float32)
            for i in np.flatnonzero(level > 0.02):
                x, y = self.gx[i] * self.h, self.gy[i] * self.h
                ix, iy = math.floor(x), math.floor(y)
                spot, sx, sy = glint.get(sprites.bucket(self.glen[i]), x - ix, y - iy)
                canvas.over(ix + sx, iy + sy, spot, white, float(level[i]) * self.opacity, mask)
