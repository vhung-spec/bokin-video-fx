"""Steam rising from a hot cup (a placed effect: Effect.BOX; see effects.py for how every effect loops).

Steam is drawn as dye carried by moving air. Points across the cup's surface let go of a little steam all the time;
the air carries it up (a warm draught, slow at the surface, that gathers it a little; eddies that grow with height
and drift up with it; a breeze that leans it and sways it now and then). What shows is the streaklines: the curves
through all the steam each point let go of over the last seconds, which is what real steam looks like: thin wisps
near the cup that sway, curl, fold and spread into soft veils with fine streaks in them, then fade away. The surface
steams more under a few wisps that wander across it and breathe, and in bands that sweep over it, so wisps come and
go. Each bit of steam keeps its amount: a wisp is denser where the air squeezes it, fainter where it is stretched.
It blurs more as it rises (it mixes with the air): drawn on a few layers, each blurred more, plus a faint haze.

The air moves on a clock of "ticks" (TICKS a second, a whole number per loop), and every bit of steam is followed
tick by tick from where it was let go; a frame lies between two ticks. The air, the points and how much steam they let
go of are functions of the tick modulo the loop (whole numbers of cycles), so frame P is frame 0 again, exactly.
Frames drawn one after the other carry the steam on a tick at a time (from the last tick drawn, when it is a few ticks
back: a video of few frames a second); a frame drawn on its own starts a whole lifetime of the steam earlier (the same
sums, so the same picture).

The box: the cup's rim lies along the middle of its bottom (half its width: "rim"), the steam rising to its top.
Inside it every length is in box heights, so the small preview clip (the box alone) and the video show the same steam.
"""
import math

import numpy as np
from PIL import Image

from .effects import Effect, _choice, _colour, _lerp, _num, _place, _rgb

TICKS = 20  # the air's clock, ticks a second
SOURCES = 56  # points on the surface that let go of steam
WISPS = 6  # at most, where the surface lets go of more
MODES = 9  # the eddies: waves of a stream function
BANDS = 4  # how blurred: steam is drawn in this many layers, each blurred more
GRID_MAX = 320  # the steam is drawn on a grid at most this many rows high, then laid over the box
HAZE = 0.25  # the share of the steam in a faint haze around the wisps
DENSITY = 0.024  # how much a bit of steam covers

DEFAULTS = {"amount": 45, "rim": 50, "reach": 80, "spread": 40, "speed": 35, "curl": 40, "breeze": 6, "gust": 25,
            "soft": 40, "opacity": 55}
LOOKS = {
    "light": {},
    "thick": {"amount": 90, "spread": 55, "speed": 40, "curl": 45, "soft": 55, "opacity": 75},
    "long": {"amount": 50, "reach": 100, "spread": 35, "speed": 25, "curl": 60, "gust": 30, "soft": 45},
    "breeze": {"amount": 50, "reach": 85, "spread": 50, "curl": 45, "breeze": 50, "gust": 30},
}
LOOKS = {k: {**DEFAULTS, **v} for k, v in LOOKS.items()}


def _p(key, label, value, group, lo=0, hi=100):
    return _num(key, label, value, lo, hi, group=group)


SCHEMA = {
    "label": "Khói cốc cà phê",
    # for the page: a mark where the cup's mouth goes, a flat ellipse centred on the middle of the box's bottom edge,
    # as wide as the setting "rim" (% of the box's width), and a tip to go with it
    "mark": {"width": "rim", "tip": "Đặt vạch ở đáy khung trắng ngang giữa miệng cốc, rồi kéo góc khung cho vạch "
                                    "dài bằng miệng cốc."},
    "params": [
    _choice("look", "Kiểu", "light",
            [("light", "Hơi nóng nhẹ"), ("thick", "Khói đậm"), ("long", "Khói lượn dài"),
             ("breeze", "Khói bay nghiêng theo gió")], presets=LOOKS, group="Kiểu"),
    _p("amount", "Lượng khói", DEFAULTS["amount"], "Khói"),
    _p("rim", "Độ rộng miệng cốc (so với khung trắng)", DEFAULTS["rim"], "Khói", 10, 100),
    _p("reach", "Khói bay cao (so với khung trắng)", DEFAULTS["reach"], "Khói", 20, 100),
    _p("spread", "Độ loe khi bay lên", DEFAULTS["spread"], "Khói"),
    _p("soft", "Độ mềm, nhoè", DEFAULTS["soft"], "Khói"),
    _p("speed", "Tốc độ bay lên", DEFAULTS["speed"], "Chuyển động"),
    _p("curl", "Uốn lượn", DEFAULTS["curl"], "Chuyển động"),
    _p("breeze", "Gió nghiêng (trái ← → phải)", DEFAULTS["breeze"], "Gió", -100, 100),
    _p("gust", "Gió lay qua lại", DEFAULTS["gust"], "Gió"),
    _p("opacity", "Độ đậm", DEFAULTS["opacity"], "Màu"),
    _colour("color", "Màu khói", "#f4f1ec", group="Màu"),
    *_place(45)]}


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _boxes(sigma):
    """Widths (odd) of three box blurs that make a Gaussian blur of `sigma` pixels."""
    wl = int(math.sqrt(4 * sigma * sigma + 1))
    wl -= 1 - wl % 2
    m = round((12 * sigma * sigma - 3 * wl * wl - 12 * wl - 9) / (-4 * wl - 4))
    return [wl if i < m else wl + 2 for i in range(3)]


def _box(a, w, axis):
    """A box blur of odd width w along an axis (0 beyond the edges)."""
    r = w // 2
    if r == 0:
        return a
    a = np.moveaxis(a, axis, 0)
    n = a.shape[0]
    cs = np.zeros((n + 2 * r + 1,) + a.shape[1:], np.float32)
    cs[r + 1:r + 1 + n] = a
    np.cumsum(cs, axis=0, out=cs)
    out = cs[2 * r + 1:] - cs[:n]
    out *= 1.0 / w
    return np.moveaxis(out, 0, axis)


def _blur(a, sigma):
    """A Gaussian blur of `a` (rows, columns, float32) by `sigma` pixels, zero beyond the edges."""
    if sigma < 0.3:
        return a
    if sigma < 1.5:  # a small kernel
        r = int(math.ceil(3 * sigma))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
        k /= k.sum()
        for axis in (0, 1):
            out = a * k[r]
            for i in range(1, min(r, a.shape[axis] - 1) + 1):
                if axis == 0:
                    out[i:] += k[r - i] * a[:-i]
                    out[:-i] += k[r + i] * a[i:]
                else:
                    out[:, i:] += k[r - i] * a[:, :-i]
                    out[:, :-i] += k[r + i] * a[:, i:]
            a = out
        return a
    if sigma > 2.5 and min(a.shape) >= 16:  # wide: at half size
        h, w = a.shape
        e = np.pad(a, ((0, h % 2), (0, w % 2))) if (h % 2 or w % 2) else a
        small = (e[0::2, 0::2] + e[1::2, 0::2] + e[0::2, 1::2] + e[1::2, 1::2]) * 0.25
        small = _blur(small, math.sqrt(max(sigma * sigma - 0.25, 0)) / 2)
        big = Image.fromarray(small, "F").resize((e.shape[1], e.shape[0]), Image.Resampling.BILINEAR)
        return np.array(big, np.float32)[:h, :w]
    for axis in (1, 0):
        for w in _boxes(sigma):
            a = _box(a, w, axis)
    return a


class Steam(Effect):
    BOX = (1, 1)
    VERSION = 2  # (preview clips of the starter are made again)

    def __init__(self, *a):
        super().__init__(*a)
        p, rng = self.p, self.rng
        u = rng.random((SOURCES, 9))
        wu = rng.random((WISPS, 10))
        m = rng.random((MODES, 4))
        g = rng.random(8)
        fb = rng.random((2, 3))
        more = rng.random((SOURCES, 9))  # (points for a wide rim: drawn last, so the others keep their draws)
        self.bw = self.BOX[0] / self.BOX[1]  # the box's width, in box heights
        self.ticks = max(1, int(round(TICKS * self.loop_s)))  # ticks per loop
        self.dt = self.loop_s / self.ticks  # seconds a tick
        amount = p["amount"] / 100
        self.speed = _lerp(0.16, 0.6, p["speed"] / 100)  # box heights a second
        self.reach = p["reach"] / 100
        life = 1.5 * self.reach / self.speed + 0.6  # seconds a bit of steam is followed
        self.L = int(math.ceil(life / self.dt)) + 2  # ticks
        # the points on the surface, anywhere across the rim (not evenly: no comb of strands); the back of the surface
        # is a little higher. A few let go of much more than the others: a wisp is a ribbon or two and faint threads.
        # A rim wider than half the box has more points, as close together, and its wisps are as wide
        wide = max(1.0, p["rim"] / 50)
        n = self.n = int(round(SOURCES * wide))
        u = np.concatenate([u, more])[:n]
        rim = p["rim"] / 100 * self.bw
        self.v = u[:, 0] * 2 - 1
        self.sx = self.v * rim / 2 * 0.92
        self.sy = u[:, 1] * 0.025 * np.sqrt(1 - self.v ** 2)
        self.lift = _lerp(0.88, 1.12, u[:, 4])  # some rise a little quicker
        self.bright = 1.85 * np.exp(-2.0 * u[:, 5])  # (0.8 on average)
        self.wander = rim * 0.03 * u[:, 6]  # each point wanders a little
        self.wm = self.cycles(_lerp(0.1, 0.3, u[:, 7])).astype(np.int64)
        self.wphi = u[:, 8] * 2 * math.pi
        # how much steam they let go of: bands that sweep across the surface, and a flicker of each point
        self.bm2 = self.cycles(_lerp(0.15, 0.5, fb[:, 0])).astype(np.int64)
        self.bk = np.where(fb[:, 1] < 0.5, -1, 1) * math.pi * _lerp(0.8, 2.5, fb[:, 1])
        self.bphi2 = fb[:, 2] * 2 * math.pi
        self.fm = self.cycles(_lerp(0.25, 0.7, u[:, 2])).astype(np.int64)
        self.fphi = u[:, 3] * 2 * math.pi
        # wisps: where the surface lets go of more steam; they wander across it and breathe
        nw = 0 if amount <= 0 else int(round(_lerp(2, WISPS, amount ** 1.3)))
        wu = wu[:nw]
        self.nw = nw
        self.wc = ((np.arange(nw) * 0.6180339887 + g[5]) % 1.0 * 2 - 1) * 0.65
        self.wa = _lerp(0.1, 0.3, wu[:, 0]) / wide
        self.cm = self.cycles(_lerp(0.02, 0.07, wu[:, 1])).astype(np.int64)
        self.cphi = wu[:, 2] * 2 * math.pi
        self.ws = _lerp(0.1, 0.22, wu[:, 3]) / wide  # width, in half rims
        self.e1 = self.cycles(_lerp(0.04, 0.12, wu[:, 4])).astype(np.int64)
        self.e2 = self.cycles(_lerp(0.12, 0.3, wu[:, 5])).astype(np.int64)
        self.ephi1, self.ephi2 = wu[:, 6] * 2 * math.pi, wu[:, 7] * 2 * math.pi
        self.gain = _lerp(0.7, 1.0, amount) / max(1, nw) ** 0.3 / SOURCES
        # as much steam along a wisp at any speed (quicker steam would spread the same over a longer path)
        self.gain *= self.speed / _lerp(0.16, 0.6, DEFAULTS["speed"] / 100)
        self.veil = 0.04 + 0.16 * amount ** 3  # the whole surface steams too, a lot when it is very hot
        self.gather = 1.5 * (1 - 0.6 * amount ** 2)  # a little steam gathers into a few wisps; a lot rises wide
        # eddies: a stream function of waves that drift up with the steam
        curl = _lerp(0.12, 1.0, p["curl"] / 100)  # (never quite still: air in a room always moves a little)
        lam = np.exp(_lerp(math.log(0.12), math.log(0.55), m[:, 0]))  # wavelengths, box heights
        ang = np.where(m[:, 1] < 0.5, -1, 1) * _lerp(0.45, 1.25, (2 * m[:, 1]) % 1.0)  # from straight up
        k = 2 * math.pi / lam
        kx, ky = k * np.sin(ang), k * np.abs(np.cos(ang))
        vel = self.speed * curl * 0.76 * (lam / 0.55) ** 0.4 / math.sqrt(MODES / 3)
        amp = vel / k
        self.kx, self.ky = kx.astype(np.float32), ky.astype(np.float32)
        self.amp = np.stack([amp, -amp * ky, -amp * kx], 1).astype(np.float32)  # psi, d psi / dy, d psi / dx
        drift = self.speed * _lerp(0.5, 1.0, m[:, 2])  # how fast the eddies go up
        self.om = self.cycles(drift * ky / (2 * math.pi)).astype(np.int64)  # cycles per loop
        self.mphi = m[:, 3] * 2 * math.pi
        # the breeze: a lean, and a sway now and then
        self.lean = p["breeze"] / 100 * 1.1 * self.speed
        self.sway = p["gust"] / 100 * 0.5 * self.speed
        self.bm = self.cycles(np.array([_lerp(0.04, 0.08, g[1]), _lerp(0.1, 0.2, g[2])])).astype(np.int64)
        self.bphi = g[3:5] * 2 * math.pi
        self.flare = _lerp(0.0, 1.8, p["spread"] / 100) * self.speed  # how fast it spreads sideways, per second
        # blur: from thin wisps at the cup to soft veils at the top, and a faint haze around them
        soft = p["soft"] / 100
        self.s0 = _lerp(0.0015, 0.008, soft)
        self.s1 = _lerp(0.012, 0.05, p["spread"] / 100) * _lerp(0.8, 1.4, soft)
        self.colour = _rgb(p["color"])
        self.opacity = p["opacity"] / 100
        self.states = {}

    # --- the air -----------------------------------------------------------------------------------------------
    def _phase(self, cycles, k):
        """2 pi cycles k / ticks, from whole numbers (so tick k + ticks gives exactly the same)."""
        return (np.asarray(cycles, np.int64) * k % self.ticks) * (2 * math.pi / self.ticks)

    def _breeze(self, k):
        ph = self._phase(self.bm, k) + self.bphi
        return self.lean + self.sway * (0.65 * math.sin(ph[0]) + 0.35 * math.sin(ph[1]))

    def _velocity(self, x, y, age, k):
        """The air's velocity (box heights a second) at x, y for steam `age` seconds old, at tick k."""
        reach = self.reach
        up = self.lift * self.speed * (0.35 + 0.65 * (1 - np.exp(-age / 0.45)))  # slow at the surface
        up = up * (1 - 0.15 * _smooth(0.2, 1.0, y / reach))
        top = 0.45 * reach
        q = np.clip(y / top, 0.0, 1.0)
        z = np.clip((y - top) / (reach - top), 0.0, 1.0)
        gy = q * q * (3 - 2 * q) * (1 - 0.5 * z * z)  # the eddies grow with height, then calm down
        dg = 6 * q * (1 - q) / top - z * ((y > top) & (y < reach)) / (reach - top)
        vx = self._breeze(k) * (0.04 + 0.96 * _smooth(0.0, 0.25, y))  # (the cup shelters the steam just above it)
        axis = self.lean * np.maximum(y - 0.12, 0.0) / self.speed  # where the leaning steam is, roughly
        vx = vx + self.flare * (x - axis) * _smooth(0.05, 0.5 * reach, y)  # it spreads as it rises
        vx = vx - self.gather * x * np.exp(-np.maximum(y, 0.0) / 0.06)  # the draught over the cup gathers it
        th = (x.astype(np.float32)[..., None] * self.kx + y.astype(np.float32)[..., None] * self.ky -
              (self._phase(self.om, k) + self.mphi).astype(np.float32))
        psi = np.cos(th) @ self.amp[:, 0]
        si = np.sin(th)
        return vx + gy * (si @ self.amp[:, 1]) + dg * psi, up - gy * (si @ self.amp[:, 2])

    def _step(self, state, k, rows=None):
        """The steam at tick k + 1 from the steam at tick k (rows: age in ticks; columns: points). Only the first
        `rows` rows are carried on (the others are not steam yet when starting from scratch)."""
        x, y = state
        r = self.L - 1 if rows is None else rows
        age = (np.arange(r, dtype=np.float64) * self.dt)[:, None]
        xs, ys = x[:r], y[:r]
        vx, vy = self._velocity(xs, ys, age, k)
        hx, hy = xs + 0.5 * self.dt * vx, ys + 0.5 * self.dt * vy
        vx, vy = self._velocity(hx, hy, age + 0.5 * self.dt, k)
        nx, ny = x.copy(), y.copy()
        nx[1:r + 1], ny[1:r + 1] = xs + self.dt * vx, ys + self.dt * vy
        nx[0], ny[0] = self._source(k + 1)
        return nx, ny

    def _source(self, k):
        """Where the points let go of steam at tick k."""
        return self.sx + self.wander * np.sin(self._phase(self.wm, k) + self.wphi), self.sy

    def _state(self, k):
        """The steam at tick k: x, y of every bit of it (rows: age in ticks, columns: points)."""
        k %= self.ticks
        if k in self.states:
            return self.states[k]
        back = [(k - j) % self.ticks for j in self.states]
        near = min((b for b in back if 0 < b <= 8), default=0)  # a tick a little earlier: carried on from it
        if near:
            for j in range(k - near, k - 1):
                self._state(j + 1)
            prev = (k - 1) % self.ticks
            state = self._step(self.states[prev], prev)
        else:  # from scratch, a whole lifetime back: whatever was there before has gone by tick k
            sx, sy = self._source(k)
            state = (np.repeat(sx[None], self.L, 0), np.repeat(sy[None], self.L, 0))
            for i in range(self.L):
                state = self._step(state, (k - self.L + i) % self.ticks, min(i, self.L - 1))
        if len(self.states) > 3:
            self.states.pop(next(iter(self.states)))
        self.states[k] = state
        return state

    def _emission(self, j):
        """How much steam each point lets go of at ticks j (one row each): the wisps over it, breathing."""
        j = np.asarray(j)[:, None]
        out = np.full((j.shape[0], self.n), self.veil)
        for w in range(self.nw):
            c = self.wc[w] + self.wa[w] * np.sin(self._phase(self.cm[w], j) + self.cphi[w])
            e = 0.5 + 0.5 * (0.6 * np.sin(self._phase(self.e1[w], j) + self.ephi1[w]) +
                             0.4 * np.sin(self._phase(self.e2[w], j) + self.ephi2[w]))
            out += (e * e) * np.exp(-0.5 * ((self.v - c) / self.ws[w]) ** 2)
        bands = sum(c * np.sin(self._phase(self.bm2[i], j) + self.bk[i] * self.v + self.bphi2[i])
                    for i, c in enumerate((0.45, 0.3)))
        flicker = 0.5 + 0.5 * (bands + 0.25 * np.sin(self._phase(self.fm, j) + self.fphi))
        flicker = 0.15 + 0.85 * flicker * flicker
        return out * flicker * self.bright * np.sqrt(np.clip(1 - self.v ** 2, 0, 1))

    # --- drawing -----------------------------------------------------------------------------------------------
    def draw(self, t, canvas, mask):
        if self.nw == 0 or self.opacity <= 0:
            return
        x0, y0, w, h = self.box()
        bx, by = int(round(x0)), int(round(y0))
        bw, bh = int(round(x0 + w)) - bx, int(round(y0 + h)) - by
        if bw < 2 or bh < 2 or bx >= self.w or by >= self.h or bx + bw <= 0 or by + bh <= 0:
            return
        cover = self.cover(t, h)
        # the part of the box in the frame, at the box's size
        c0, c1, r0, r1 = max(0, -bx), min(bw, self.w - bx), max(0, -by), min(bh, self.h - by)
        gh, gw = cover.shape
        if (gh, gw) != (bh, bw):
            kx, ky = gw / bw, gh / bh
            cover = np.asarray(Image.fromarray(cover, "F").resize(
                (c1 - c0, r1 - r0), Image.Resampling.BILINEAR, box=(c0 * kx, r0 * ky, c1 * kx, r1 * ky)))
        else:
            cover = cover[r0:r1, c0:c1]
        bx, by = bx + c0, by + r0
        rows = np.flatnonzero(cover.max(axis=1) > 1 / 2048)
        if rows.size == 0:
            return
        cols = np.flatnonzero(cover.max(axis=0) > 1 / 2048)
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        canvas.over(bx + c0, by + r0, cover[r0:r1, c0:c1], self.colour, 1.0, mask)

    def cover(self, t, h):
        """How much the steam covers, 0..1, on a grid over the box (rows from the top), for a box h pixels high."""
        gh = int(min(GRID_MAX, max(16, round(h))))
        gw = max(4, int(round(gh * self.bw)))
        if float(t).is_integer():  # (exactly, so frame P is frame 0)
            k, f = divmod(int(t) * self.ticks, self.P)
            f /= self.P
        else:
            k = math.floor(t * self.ticks / self.P)
            f = t * self.ticks / self.P - k
        xa, ya = self._state(k)
        xb, yb = self._state(k + 1)
        # the bit of steam let go at tick k - a: row a at tick k, row a + 1 at tick k + 1
        x = (1 - f) * xa[:-1] + f * xb[1:]
        y = (1 - f) * ya[:-1] + f * yb[1:]
        age = (np.arange(self.L - 1) + f)[:, None] * self.dt
        life = self.L * self.dt
        mass = self._emission(k - np.arange(self.L - 1)) * self.gain
        e = min(1.0, self.reach / 0.6)  # (steam that does not rise far shows sooner, or there would be none)
        mass *= _smooth(0.0, 0.8 * e, age) * _smooth(0.0, 0.12 * e, y) * (1 - _smooth(0.35 * self.reach, self.reach, y))
        mass *= 1 - _smooth(life * 0.75, life, age)
        sig = self.s0 + (self.s1 - self.s0) * np.clip(y / max(self.reach, 1e-3), 0, 1.3) ** 2
        grid = self._splat(x, y, mass, sig, gh, gw)
        r = math.log(self.s1 / self.s0) / (BANDS - 1)
        sigmas = [self.s0 * math.exp(r * b) for b in range(BANDS)] + [self.s1 * 2.5]
        out = grid.sum(axis=0) * HAZE  # the haze: a share of all of it, blurred the most
        grid *= 1 - HAZE
        for b in range(BANDS - 1, -1, -1):  # each layer blurred as much as it needs on top of the next one's blur
            out = _blur(out, math.sqrt(sigmas[b + 1] ** 2 - sigmas[b] ** 2) * gh)
            out += grid[b]
        out = _blur(out, sigmas[0] * gh)
        out *= gh * gh * DENSITY
        # fade near the box's sides and top: nothing is cut off at its edges
        cols = (np.arange(gw, dtype=np.float32) + 0.5) / gw
        rows = (np.arange(gh, dtype=np.float32) + 0.5) / gh
        out *= _smooth(0.0, 0.1, rows)[:, None] * (_smooth(0.0, 0.12, cols) * _smooth(0.0, 0.12, 1 - cols))[None, :]
        np.negative(out, out=out)
        np.exp(out, out=out)
        return ((1 - out) * self.opacity).astype(np.float32)

    def _splat(self, x, y, mass, sig, gh, gw):
        """The steam laid on BANDS grids by how blurred it is: each bit as a curve on to the next bit from the same
        point (Catmull-Rom), in steps shorter than the blur, each step with its share of the bit's steam."""
        gx = ((x + self.bw / 2) * gh - 0.5).ravel()
        gy = ((1 - y) * gh - 0.5).ravel()
        rows, n = x.shape
        m = 0.5 * (mass[:-1] + mass[1:])
        top = gy[:-n].reshape(rows - 1, n)
        idx = np.flatnonzero(((m > 1e-4) & (top > -gh * 0.2) & (top < gh * 1.2)).ravel())
        if idx.size == 0:
            return np.zeros((BANDS, gh, gw), np.float32)
        a, col = idx // n, idx % n
        p0, p1, p2, p3 = (np.clip(a + d, 0, rows - 1) * n + col for d in (-1, 0, 1, 2))
        coef = []
        for g in (gx, gy):  # the curve through p1 and p2: ((c3 u + c2) u + c1) u + p1
            g0, g1, g2, g3 = g[p0], g[p1], g[p2], g[p3]
            coef.append((-0.5 * g0 + 1.5 * g1 - 1.5 * g2 + 0.5 * g3, g0 - 2.5 * g1 + 2 * g2 - 0.5 * g3,
                         0.5 * (g2 - g0), g1))
        s = 0.5 * (sig[:-1] + sig[1:]).ravel()[idx]
        step = np.clip(0.7 * s * gh, 0.7, 3.0)  # (a step need not be shorter than the blur it gets)
        seg = np.hypot(gx[p2] - gx[p1], gy[p2] - gy[p1])
        cnt = np.clip(np.ceil(seg / step), 1, 64).astype(np.int64)
        rep = np.repeat(np.arange(idx.size), cnt)
        u = ((np.arange(rep.size) - np.repeat(np.cumsum(cnt) - cnt, cnt) + 0.5) / cnt[rep]).astype(np.float32)
        px, py = ((((c3[rep] * u + c2[rep]) * u + c1[rep]) * u + c0[rep]) for c3, c2, c1, c0 in coef)
        r = math.log(self.s1 / self.s0) / (BANDS - 1)
        band = np.clip(np.log(np.maximum(s, 1e-9) / self.s0) / r, 0, BANDS - 1)
        b0 = np.minimum(band.astype(np.int64), BANDS - 2)
        wgt = m.ravel()[idx] / cnt
        w1 = (wgt * (band - b0))[rep]
        w0 = wgt[rep] - w1
        ix, iy = np.floor(px).astype(np.int64), np.floor(py).astype(np.int64)
        fx, fy = (px - ix).astype(np.float32), (py - iy).astype(np.float32)
        # on a grid one pixel larger all round: what falls on its border is dropped, and what lies beyond it
        out = (ix < -1) | (ix >= gw) | (iy < -1) | (iy >= gh)
        if out.any():
            w0[out] = 0
            w1[out] = 0
            np.clip(ix, -1, gw - 1, out=ix)
            np.clip(iy, -1, gh - 1, out=iy)
        W, H = gw + 2, gh + 2
        flat = (b0[rep] * H + iy + 1) * W + ix + 1
        corners = ((0, (1 - fx) * (1 - fy)), (1, fx * (1 - fy)), (W, (1 - fx) * fy), (W + 1, fx * fy))
        index = np.concatenate([flat + (o + d) for o in (0, H * W) for d, _ in corners])
        weight = np.concatenate([wb * c for wb in (w0, w1) for _, c in corners])
        acc = np.bincount(index, weight, BANDS * H * W).astype(np.float32).reshape(BANDS, H, W)
        return acc[:, 1:-1, 1:-1]
