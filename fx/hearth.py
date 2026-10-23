"""Fire in a fireplace ("Lửa lò sưởi"): a wood fire burning on the logs of a fireplace in the video (a placed lens
effect, as the candle): flames that lick up, swirl and break off, white-yellow at their roots, orange and dark red at
their tips, as a fire filmed; embers glowing under them; their light flickering on the firebox; and, if the video
already has a fire that does not move, that fire dimmed under the new one ("Che lửa cũ").

The box (Effect.BOX) goes over the fireplace's opening. Its mark, a flat ellipse sitting on its bottom edge, goes on
the logs where the fire burns, as wide as the fire ("Lửa rộng"); the flames rise in the box above it.

The flames are a small fluid simulation (Stam's stable fluids): fuel fed at burning places on the bed that wax and
wane, burnt into heat (which rises) and a short-lived glow (the flame), the air swirling (vorticity confinement). It
runs on its own grid over the box (GRID, the box's shape), RATE steps a second of fire whatever the frame rate, so the
box alone (the preview clip) and the video show the same fire. A fire is made frame after frame, each from the one
before, as far as a frame asked for needs (and kept: fire()); it loops by a crossfade: over its last second the
frames before its first come in (their glow blended, then coloured), so frame P is frame 0 again.

Colour: a real fire's colours by brightness (measured from footage of a wood fire: dark red where faint, orange-red,
orange, yellow, white-yellow where it burns hottest), a glow round the flames as a camera sees it, laid with Screen.
"""
import math
import threading

import numpy as np
from PIL import Image

from . import logs
from .effects import Effect, _choice, _lerp, _num, _place, blur

TAU = 2 * math.pi
GRID = (192, 144)  # the simulation's cells over the box (BOX's shape), x then y
RATE = 48.0  # simulation steps a second of fire (more a frame where frames are longer: no step is longer)
WARM = 3.0  # seconds of fire before the loop (well alight: its mean glow measured over the last half)
JOIN = 1.0  # seconds of the crossfade that closes the loop
MARK = 0.09  # the bed's mark on the page: a flat ellipse this high for its width, on the box's bottom edge
CUT = 0.12  # glow this faint is not drawn (no lingering haze above the flames)
STORE = 3.0  # (the glow is kept as 8 bits: its square root, up to this)
LEVEL = 0.155  # a fire's mean glow at the default settings (a bigger fire throws more light on the room)
KEEP = 240e6  # bytes of fires kept for later (the newest)
HALO = 0.35  # the glow round the flames (a camera's), as bright as the flames blurred times this
HALO_R = 0.07  # (its blur: box heights)
LIGHT = np.array([1.0, 0.62, 0.3], np.float32)  # the fire's light on what is round it (its colour)
LIGHT_K = 1.0  # (how much brighter it makes what is near it, at "light" 100, a fire of the default size)
LUT_TOP = 2.2  # (the fire's colours looked up in a table, brightness 0 to this: brighter is as white)
DIM = 0.28  # the old fire put out ("cover" 100): its glow blurred, this bright
PACE = 1.2  # seconds of fire a second of video at "speed" 50 (a fire filmed: tongues come and go quickly)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
# logs ("Củi"): the flames' light on them (times their glow near each light, over REACH box heights), the embers'
# (times "embers"), the room's (times the picture's mean light round the box), their cracks' glow (times "embers"),
# and how much they keep the light from the floor right under them
FLAME_ON_LOGS, EMBERS_ON_LOGS, ROOM_ON_LOGS, CRACKS = 2.6, 0.8, 1.0, 1.1
REACH = 0.07
BEHIND = 0.7  # (the bright flames' share seen over a log behind the fire, fainter ones less; none over one in front)
FLOOR_SHADE = 0.7
# a real wood fire's colours by brightness (measured from footage: dark red where faint, orange-red, orange, yellow,
# white-yellow where it burns hottest)
PAL_I = np.array([0.0, 0.12, 0.3, 0.5, 0.7, 0.9, 1.1, 1.35, 1.8], np.float32)
PAL = np.array([[0, 0, 0], [52, 12, 3], [169, 58, 15], [244, 95, 14], [251, 145, 8], [253, 195, 14],
                [253, 228, 94], [254, 239, 173], [255, 250, 228]], np.float32) / 255

DEFAULTS = {"bed": 70, "height": 50, "amount": 50, "flicker": 50, "speed": 50, "bright": 50, "warmth": 50,
            "embers": 40, "light": 45, "cover": 0}
LOOKS = {
    "logs": {},
    "big": {"height": 75, "amount": 75, "flicker": 60, "bright": 60, "embers": 50, "light": 60},
    "low": {"height": 30, "amount": 35, "flicker": 40, "bright": 45, "embers": 55, "light": 35},
    "embers": {"height": 12, "amount": 20, "flicker": 35, "bright": 40, "embers": 85, "light": 30},
}

WITH_LOGS = {"wood": list(logs.WOODS)}  # (the logs' settings show once a wood is chosen)

SCHEMA = {
    "label": "Lửa lò sưởi",
    "note": "Lửa củi cháy trong lò sưởi có sẵn trong video. (1) Bấm Suốt video. (2) Kéo khung trắng tới lò sưởi: hình "
            "bầu dục vàng ở đáy khung đặt lên mặt củi, chỗ lửa bốc lên; kéo thanh Lửa rộng cho nó rộng bằng đống củi; "
            "kéo góc khung cho lửa cao vừa miệng lò. (3) Video có sẵn lửa đứng yên thì kéo Che lửa cũ lên: lửa cũ tối "
            "đi, lửa mới cháy lên trên. Khúc củi phía trước che chân lửa: tô khúc củi đó vào Tiền cảnh rồi bật Bay ra "
            "sau tiền cảnh cho clip này. Lò trống, chưa có củi: chọn Củi (sồi, vỏ nâu, bạch dương): app vẽ đống củi "
            "cháy sém đứng trên hình bầu dục vàng (đặt nó lên sàn lò), lửa bốc lên từ giữa các khúc củi, than hồng "
            "trong vết nứt; chỉnh Cách xếp, Khúc củi to, Cháy sém, Nhìn từ trên xuống cho hợp góc máy. Lửa không xoay "
            "3D: lửa luôn bốc thẳng lên. Mỗi lần đổi Ngọn lửa cao, Lửa dày, Bập bùng, Tốc độ (và cỡ, cách xếp củi), "
            "app tính lại lửa từ đầu: dừng hình lần đầu có thể chờ vài giây.",
    # for the page: where the fire burns, a flat ellipse sitting on the middle of the box's bottom edge, as wide as
    # "bed" (% of the box's width)
    "mark": {"width": "bed", "ratio": MARK, "sits": True, "label": "chỗ lửa cháy (mặt củi)",
             "tip": "Đặt hình bầu dục vàng lên mặt củi, chỗ lửa bốc lên (đã chọn Củi: lên sàn lò, chỗ đặt củi); kéo "
                    "thanh Lửa rộng cho nó rộng bằng đống củi, kéo góc khung cho lửa cao vừa lò sưởi."},
    "params": [
        _choice("look", "Kiểu lửa", "logs", [("logs", "Lửa củi (vừa)"), ("big", "Lửa to, cháy mạnh"),
                                             ("low", "Lửa nhỏ, dịu"), ("embers", "Sắp tàn, than hồng")],
                presets={k: {key: v for key, v in {**DEFAULTS, **look}.items() if key not in ("bed", "cover")}
                         for k, look in LOOKS.items()}, group="Kiểu"),
        _choice("wood", "Củi", "none", [("none", "Không thêm (lửa cháy trên củi có sẵn trong video)"),
                                         ("oak", "Củi sồi (vỏ xám, rãnh sâu)"), ("brown", "Củi vỏ nâu"),
                                         ("birch", "Củi bạch dương (vỏ trắng)")], group="Củi"),
        _choice("stack", "Cách xếp", "stack", [("stack", "Ba khúc: hai dưới, một trên"),
                                              ("heap", "Đống củi (năm khúc)"), ("two", "Hai khúc nằm cạnh nhau")],
                group="Củi", show=WITH_LOGS),
        _num("log_size", "Khúc củi to", 50, group="Củi", show=WITH_LOGS),
        _num("char", "Cháy sém (than đen, vết nứt)", 65, group="Củi", show=WITH_LOGS),
        _num("view", "Nhìn từ trên xuống (ngang ← → cao)", 50, group="Củi", show=WITH_LOGS),
        _num("bed", "Lửa rộng (hình bầu dục vàng)", DEFAULTS["bed"], 10, 100, group="Ngọn lửa"),
        _num("height", "Ngọn lửa cao", DEFAULTS["height"], group="Ngọn lửa"),
        _num("amount", "Lửa dày", DEFAULTS["amount"], group="Ngọn lửa"),
        _num("flicker", "Bập bùng", DEFAULTS["flicker"], group="Ngọn lửa"),
        _num("speed", "Tốc độ (chậm ← → nhanh)", DEFAULTS["speed"], group="Ngọn lửa"),
        _num("bright", "Độ sáng", DEFAULTS["bright"], group="Màu, ánh sáng"),
        _num("warmth", "Màu lửa (vàng ← → đỏ cam)", DEFAULTS["warmth"], group="Màu, ánh sáng"),
        _num("embers", "Than hồng dưới chân lửa", DEFAULTS["embers"], group="Màu, ánh sáng"),
        _num("light", "Ánh lửa hắt lên lò sưởi", DEFAULTS["light"], group="Màu, ánh sáng"),
        _num("cover", "Che lửa cũ trong video", DEFAULTS["cover"], group="Màu, ánh sáng"),
        *_place(60)]}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft(a, radius):
    """A float picture (h x w, or h x w x c) blurred (Gaussian, `radius` its sigma in pixels): on a copy a few times
    smaller (as wide a blur, a little smoother), then enlarged."""
    h, w = a.shape[:2]
    step = max(1, min(int(radius / 2), h, w))
    small = (max(1, round(w / step)), max(1, round(h / step)))
    planes = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[2])]
    out = []
    for plane in planes:
        p = Image.fromarray(np.ascontiguousarray(plane, np.float32), "F").resize(small, Image.Resampling.BOX)
        p = blur(np.asarray(p, np.float32), radius / step)
        out.append(np.asarray(Image.fromarray(p, "F").resize((w, h), Image.Resampling.BILINEAR), np.float32))
    return out[0] if a.ndim == 2 else np.stack(out, 2)


def palette(i, warmth=0.5):
    """A fire's colours (... x 3) for brightness i: the measured ones, a little yellower or redder."""
    x = np.asarray(i, np.float32) * (1.0 / (1.0 + 0.4 * (warmth - 0.5)))  # (redder: brighter glow for one colour)
    out = np.empty(x.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(x, PAL_I, PAL[:, c])
    return out


class _Fluid:
    """The fire's air on the grid (x right, y up, in cells): its velocity, heat, fuel and the flames' glow."""

    def __init__(self, k, seed):
        gw, gh = GRID
        self.k = k
        self.u, self.v, self.heat, self.fuel, self.glow, self.p = (np.zeros((gh, gw), np.float32) for _ in range(6))
        ys, xs = np.mgrid[0:gh, 0:gw]
        self.X, self.Y = xs.astype(np.float32), ys.astype(np.float32)
        # the burning places: along the bed (its ellipse), at different depths in it, each waxing and waning
        g = np.random.default_rng([seed, 7103])
        bed = k["bed"]
        n = max(2, int(round(k["places"] * bed)))
        u = np.linspace(-1, 1, n) * 0.9 + g.uniform(-0.05, 0.05, n)
        self.at = (u * bed + 1) / 2 * gw
        half = MARK * bed * gw / 2  # (the ellipse's half height, cells)
        self.up = 1.5 + half * (1 + 0.7 * np.sqrt(np.maximum(0.0, 1 - u * u)) * g.uniform(-1, 1, n))
        if "base" in k:  # (logs: burning from between them up to their tops)
            self.up += k["base"] * g.uniform(0.25, 1.0, n) * np.sqrt(np.maximum(0.0, 1 - (0.8 * u) ** 2))
        self.wide = g.uniform(0.035, 0.07, n) * gw
        self.hz = g.uniform(0.3, 1.2, n) * k["waver"]
        self.ph = g.random(n) * TAU
        self.strength = g.uniform(0.6, 1.0, n) * (1 - 0.35 * u * u)
        rows = int(math.ceil(self.up.max() + 4))
        self.rows = slice(0, rows)
        dy = self.Y[:rows, :1] - self.up  # (rows x places)
        self.shape = np.exp(-(dy / 2.0) ** 2).astype(np.float32)  # each place's feed down the rows
        self.across = np.exp(-((self.X[0][:, None] - self.at) / self.wide) ** 2).astype(np.float32)  # and across
        self.s = 0.0  # seconds of fire

    def step(self, dt):
        """One step of dt (1/RATE seconds each, at most 1)."""
        k = self.k
        a = self.strength * (1 - k["wax"] + k["wax"] * np.sin(TAU * self.hz * self.s + self.ph))
        feed = (self.shape * a.astype(np.float32)) @ self.across.T  # (rows x cells)
        rows = self.rows
        self.fuel[rows] += k["feed"] * dt * feed
        np.maximum(self.v[rows], k["jet"] * feed, out=self.v[rows])
        burn = self.fuel * (1 - (1 - k["burn"]) ** dt)
        self.fuel -= burn
        self.heat += burn
        self.glow += burn * k["glow"]
        self.v += k["lift"] * dt * self.heat
        self.heat *= k["cool"] ** dt
        self.glow *= k["fade"] ** dt
        self._swirl(k["swirl"], dt)
        np.clip(self.u, -6.0, 6.0, out=self.u)  # (never more than a few cells a step: stable)
        np.clip(self.v, -6.0, 8.0, out=self.v)
        self._project()
        self.u, self.v, self.heat, self.fuel, self.glow = self._advect(
            (self.u, self.v, self.heat, self.fuel, self.glow), dt, (3, 4))
        damp = 0.985 ** dt
        self.u *= damp
        self.v *= damp
        self.s += dt / RATE

    def _advect(self, fields, dt, sharp=()):
        """Every field carried by the air (semi-Lagrangian: each cell takes what was where its air came from); the
        fields numbered in `sharp` by MacCormack (the step undone, half its error put back, within what was there):
        the flames' edges stay crisp."""
        here = self._where(self.X - self.u * dt, self.Y - self.v * dt)
        there = self._where(self.X + self.u * dt, self.Y + self.v * dt) if sharp else None
        out = []
        for n, f in enumerate(fields):
            if n in sharp:
                moved, lo, hi = self._sample(f, here, True)
                back = self._sample(moved, there)
                back -= f
                back *= -0.5
                back += moved
                out.append(np.clip(back, lo, hi, out=back))
            else:
                out.append(self._sample(f, here))
        return out

    @staticmethod
    def _where(x, y):
        """Where points fall on the grid: the cell under each, and the weights of its four corners."""
        gw, gh = GRID
        np.clip(x, 0, gw - 1.001, out=x)
        np.clip(y, 0, gh - 1.001, out=y)
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0  # (float32 throughout: the fields stay float32)
        gx, gy = 1 - fx, 1 - fy
        return y0.astype(np.int32) * gw + x0.astype(np.int32), gx * gy, fx * gy, gx * fy, fx * fy

    @staticmethod
    def _sample(f, where, bounds=False):
        """A field at points (bilinear), and the least and most of the four cells round each (bounds)."""
        i, w00, w01, w10, w11 = where
        gw = GRID[0]
        flat = f.ravel()
        a, b, c, d = flat.take(i), flat.take(i + 1), flat.take(i + gw), flat.take(i + gw + 1)
        v = a * w00
        v += b * w01
        v += c * w10
        v += d * w11
        if not bounds:
            return v
        return v, np.minimum(np.minimum(a, b), np.minimum(c, d)), np.maximum(np.maximum(a, b), np.maximum(c, d))

    def _project(self, iters=20):
        """The air made to neither pile up nor thin out (Jacobi): walls at the sides and under it, open above."""
        u, v, p = self.u, self.v, self.p
        gw = GRID[0]
        # (on the rows inside, as one run of cells: their first and last cells come out wrong, and are then set as
        # the walls are; each sum a run of memory, not a strided block)
        a, b = gw + 1, u.size - gw - 1
        uf, vf, pf = u.ravel(), v.ravel(), p.ravel()
        div = np.subtract(uf[a + 1:b + 1], uf[a - 1:b - 1])
        div += vf[a + gw:b + gw]
        div -= vf[a - gw:b - gw]
        div *= 0.5
        p[:] = 0.0
        part = np.empty_like(div)
        for _ in range(iters):
            np.add(pf[a + 1:b + 1], pf[a - 1:b - 1], out=part)
            part += pf[a + gw:b + gw]
            part += pf[a - gw:b - gw]
            part -= div
            part *= 0.25
            pf[a:b] = part
            p[:, 0], p[:, -1], p[0] = p[:, 1], p[:, -2], p[1]
            p[-1] = 0.0
        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])

    def _swirl(self, eps, dt):
        """Vorticity confinement: the air's small swirls kept up (the grid would smooth them away)."""
        u, v = self.u, self.v
        w = np.zeros_like(u)
        w[1:-1, 1:-1] = 0.5 * (v[1:-1, 2:] - v[1:-1, :-2] - u[2:, 1:-1] + u[:-2, 1:-1])
        a = np.abs(w)
        gx, gy = np.zeros_like(u), np.zeros_like(u)
        gx[1:-1, 1:-1] = 0.5 * (a[1:-1, 2:] - a[1:-1, :-2])
        gy[1:-1, 1:-1] = 0.5 * (a[2:, 1:-1] - a[:-2, 1:-1])
        n = np.sqrt(gx * gx + gy * gy) + 1e-5
        u += eps * dt * (gy / n) * w
        v -= eps * dt * (gx / n) * w


class _Fire:
    """One loop of a simulated fire: its frames made one after another as far as they are asked for, kept (the
    glow's square root as 8 bits), with the second before the loop (for the crossfade that closes it)."""

    def __init__(self, k, seed, frames, fps):
        self.k, self.seed, self.P = k, seed, int(frames)
        per = RATE * k["pace"] / fps  # (steps a frame)
        self.steps = max(1, math.ceil(per - 1e-9))
        self.dt = per / self.steps
        self.join = int(min(round(JOIN * fps), self.P // 4))
        self.warm = int(math.ceil(WARM * fps / k["pace"]))  # (frames)
        self.mean = LEVEL  # (its mean glow, measured as it is lit)
        gw, gh = GRID
        self.kept = np.zeros((self.join + self.P, gh, gw), np.uint8)  # frames -join..P-1
        self.level = np.zeros(self.join + self.P, np.float32)
        self.made = 0
        self.fluid = None
        self.lock = threading.Lock()

    def _make(self, upto):
        """Frames made until index `upto` (frame upto - join) is."""
        if self.fluid is None and self.made <= upto:
            self.fluid = _Fluid(self.k, self.seed)
            seen = []
            for i in range(self.warm):
                for _ in range(self.steps):
                    self.fluid.step(self.dt)
                if 2 * i >= self.warm:
                    seen.append(float(np.maximum(self.fluid.glow, 0.0).mean()))
            mean = float(np.mean(seen)) if seen else 0.0
            self.mean = mean if math.isfinite(mean) and mean > 1e-4 else LEVEL
        while self.made <= upto:
            for _ in range(self.steps):
                self.fluid.step(self.dt)
            g = np.maximum(self.fluid.glow[::-1], 0.0)
            ok = np.isfinite(g)
            if not ok.all():  # (never: the speeds are bounded; but a broken frame stays dark, not white)
                g = np.where(ok, g, 0.0)
            self.kept[self.made] = np.sqrt(np.minimum(g, STORE) * (1 / STORE)) * 255.0 + 0.5
            self.level[self.made] = float(g.mean())
            self.made += 1
        if self.made == len(self.kept):
            self.fluid = None  # (all made: its fields are not needed any more)

    def _glow(self, i):
        g = self.kept[i].astype(np.float32) * (1 / 255.0)
        g *= g
        g *= STORE
        return g, float(self.level[i])

    def frame(self, t):
        """The flames' glow at frame t of the loop (float32, rows x columns of the grid, row 0 at the top) and the
        fire's mean glow then."""
        t = int(t) % self.P
        s = t - (self.P - self.join)
        with self.lock:
            self._make(self.join + t)
            g, level = self._glow(self.join + t)
            if s >= 0:  # its last second: more and more the frames before its first (frame P is frame 0 again)
                w = float(_smooth((s + 0.5) / self.join))
                g0, l0 = self._glow(s)
                g *= 1 - w
                g += w * g0
                level += w * (l0 - level)
        return g, level


_FIRES = {}  # (what a fire is made of, its seed, its loop) -> _Fire, the newest last
_FIRES_LOCK = threading.Lock()


def fire(k, seed, frames, fps):
    """The fire of these settings (kept: the newest up to KEEP bytes)."""
    key = (tuple(sorted(k.items())), int(seed), int(frames), round(float(fps), 6))
    with _FIRES_LOCK:
        made = _FIRES.pop(key, None) or _Fire(k, seed, frames, fps)
        _FIRES[key] = made
        size = sum(f.kept.nbytes for f in _FIRES.values())
        for old in list(_FIRES)[:-1]:
            if size <= KEEP:
                break
            size -= _FIRES.pop(old).kept.nbytes
        return made


class Hearth(Effect):
    BOX = (4.0, 3.0)
    blend = "screen"
    lens = True
    UNDER = True  # its preview clip: the picture under the box with the fire in it (the old fire dimmed: "cover")
    UNDER_FADE = 0.0  # (its change already fades out toward the box's edges; its logs stand on its bottom edge)
    TURNS = False  # a fire always rises straight up
    VERSION = 1

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        h, n, f = (p[key] / 100 for key in ("height", "amount", "flicker"))
        self.bed = p["bed"] / 100
        self.k = {  # (what the simulation is made of: changing any makes the fire again)
            "bed": round(self.bed, 3), "places": round(_lerp(7.0, 15.0, n), 3),
            "feed": round(0.55 * 2 ** ((n - 0.5) * 1.4), 4), "jet": round(_lerp(1.05, 1.85, h), 4),
            "lift": round(_lerp(0.19, 0.35, h), 4), "fade": round(_lerp(0.87, 0.915, h), 4),
            "burn": 0.2, "glow": 1.7, "cool": 0.93,
            "swirl": round(_lerp(1.7, 2.3, f), 3), "wax": round(_lerp(0.25, 0.55, f), 3),
            "waver": round(_lerp(0.6, 1.4, f), 3), "pace": round(PACE * 2 ** ((p["speed"] - 50) / 50), 4)}
        self.bright_k = _lerp(0.75, 1.45, p["bright"] / 100)
        self.warmth = p["warmth"] / 100
        self.embers_k, self.light_k, self.cover_k = p["embers"] / 100, p["light"] / 100, p["cover"] / 100
        g = np.random.default_rng([self.seed, 7103, 2])
        self.ember_m = self.cycles(np.array([0.21, 0.37, 0.59]) * self.k["pace"])  # (whole cycles a loop)
        self.ember_ph = g.random(3) * TAU
        self._fire = None
        self._embers = None
        self._floor = None
        self._gshown = None
        self._laid = None
        self._lut = palette(np.linspace(0.0, LUT_TOP, 1024, dtype=np.float32), self.warmth)
        self.pile = None
        self._windows = None
        if p.get("wood") in logs.WOODS:  # logs under the flames (box heights from the bed's middle)
            wide = self.BOX[0] / self.BOX[1]
            self.pile = logs.Pile(p["wood"], p["stack"], p["log_size"] / 100, p["char"] / 100, self.bed * wide,
                                  math.radians(5 + 0.3 * p["view"]), self.seed, half=wide / 2 * 0.97,
                                  floor=-MARK * self.bed * wide / 2 + 0.006)
            _, mid = self.pile.screen(self.pile.fire)  # (the fire among the logs: cells above the bed's middle)
            self.k["base"] = round(max(0.0, float(mid) * 1.6) * GRID[1], 2)

    def frames(self):
        """This clip's fire (made as far as it is asked for, kept for the next clip with the same settings)."""
        if self._fire is None:
            self._fire = fire(self.k, self.seed, self.P, self.fps)
        return self._fire

    def _rect(self, x0, y0, x1, y1):
        r = (max(0, math.floor(x0)), max(0, math.floor(y0)), min(self.w, math.ceil(x1)), min(self.h, math.ceil(y1)))
        return r if r[0] < r[2] and r[1] < r[3] else None

    def _ember_field(self):
        """Where the embers glow on the grid (patches over the bed's ellipse), and a phase for each place."""
        if self._embers is None:
            gw, gh = GRID
            g = np.random.default_rng([self.seed, 7103, 3])
            coarse = g.random((gh // 6 + 3, gw // 6 + 3)).astype(np.float32)
            patches = np.asarray(Image.fromarray(coarse, "F").resize((gw, gh), Image.Resampling.BICUBIC), np.float32)
            fine = g.random((gh, gw)).astype(np.float32)
            ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
            u = ((xs + 0.5) / gw * 2 - 1) / max(self.bed, 1e-3)  # (the ellipse: -1..1 across)
            half = MARK * self.bed * gw / 2  # (its half height, cells)
            v = ((gh - 0.5 - ys) - half) / max(half, 1.0)
            inside = 1 - (u * u + v * v)
            field = _smooth(inside / 0.5) * np.clip(patches * 1.8 - 0.5, 0, 1) * (0.7 + 0.3 * fine)
            self._embers = field.astype(np.float32), np.clip(patches, 0, 1).astype(np.float32)
        return self._embers

    def _floor_at(self, xs):
        """How high (box heights) the flames come out of the bed across the box (xs: 0..1): an uneven line in its
        ellipse, as from between logs, not a straight floor."""
        if self._floor is None:
            g = np.random.default_rng([self.seed, 7103, 4])
            self._floor = g.random(12)
        u = np.clip(xs, 0, 1) * (len(self._floor) - 1)
        i = np.minimum(u.astype(np.int32), len(self._floor) - 2)
        f = _smooth(u - i)
        bumps = self._floor[i] * (1 - f) + self._floor[i + 1] * f
        return (MARK * self.bed * 4 / 3) * (0.15 + 0.6 * bumps)

    @staticmethod
    def _edge(xs, ys):
        """Nothing cut at the box's edges: how much shows at points of the box (xs across, ys down: 0..1)."""
        side = _smooth((np.minimum(xs, 1 - xs) - 0.01) / 0.05)  # (none on the box's outer hundredth)
        return (side[None, :] * (_smooth((ys - 0.01) / 0.05) * _smooth((1 - ys - 0.01) / 0.025))[:, None]).astype(
            np.float32)

    def _shown(self, xs, ys):
        """How much of the flames shows at points of the box: none at its edges, none under the uneven line they
        come out at (_floor_at)."""
        return (self._edge(xs, ys) * _smooth((1 - ys[:, None] - self._floor_at(xs)[None, :]) / 0.025 + 0.5)).astype(
            np.float32)

    def _grid_shown(self):
        """_shown() on the grid's cells."""
        if self._gshown is None:
            gw, gh = GRID
            self._gshown = self._shown((np.arange(gw, dtype=np.float32) + 0.5) / gw,
                                       (np.arange(gh, dtype=np.float32) + 0.5) / gh)
        return self._gshown

    def _layout(self, r, x0, y0, w, h):
        """What stays the same from frame to frame where the box is (`r`: its part in the frame), kept while the
        box stays."""
        key = (r, x0, y0, w, h)
        if self._laid is not None and self._laid[0] == key:
            return self._laid[1]
        rx0, ry0, rx1, ry1 = r
        gw, gh = GRID
        ys = (np.arange(ry0, ry1, dtype=np.float32) + 0.5 - y0) / h  # (0..1 down the box)
        xs = (np.arange(rx0, rx1, dtype=np.float32) + 0.5 - x0) / w
        edge = self._edge(xs, ys)
        lay = {"size": (rx1 - rx0, ry1 - ry0), "edge": edge, "shown": self._shown(xs, ys),
               # (the part of the grid that shows: whole pixels, up to a pixel more than the box at its edges)
               "src": (max(0.0, (rx0 - x0) / w * gw), max(0.0, (ry0 - y0) / h * gh), min(gw, (rx1 - x0) / w * gw),
                       min(gh, (ry1 - y0) / h * gh))}
        if self.light_k > 0:  # where the fire's light falls round it
            spill = np.exp(-((xs - 0.5) / (0.3 + 0.35 * self.bed)) ** 2)[None, :] * np.exp(
                -((ys - (1 - MARK * self.bed * 4 / 3)) / 0.6) ** 2)[:, None]
            lay["spill"] = (spill * edge).astype(np.float32)
        if self.cover_k > 0:  # where an old fire is put out: over the bed and above it, not at the box's bottom edge
            lay["zone"] = (_smooth((self.bed * 1.25 - np.abs(xs * 2 - 1)) / 0.15)[None, :] * edge * _smooth(
                (1 - ys) / (MARK * self.bed * 4 / 3 + 0.02))[:, None]).astype(np.float32)
        if self.pile is not None:  # the logs: traced over their part of the box (in the frame)
            cx, cy = x0 + w / 2, y0 + h - MARK * self.bed * w / 2  # (the bed's middle)
            left, right, bottom, top = self.pile.extent()
            px0, px1 = max(rx0, math.floor(cx + left * h)), min(rx1, math.ceil(cx + right * h))
            py0, py1 = max(ry0, math.floor(cy - top * h)), min(ry1, math.ceil(cy - bottom * h))
            if px0 < px1 and py0 < py1:
                lay["pile"] = (px0 - rx0, py0 - ry0, px1 - rx0, py1 - ry0)
                lay["logs"] = self.pile.lay((np.arange(px0, px1) + 0.5 - cx) / h, (cy - np.arange(py0, py1) - 0.5) / h,
                                            1.0 / h)
        self._laid = (key, lay)
        return lay

    def _light_windows(self):
        """For each of the logs' flame lights: the part of the grid round it and its weights there (the flames'
        light on it is their glow there)."""
        if self._windows is None:
            gw, gh = GRID
            wide = self.BOX[0] / self.BOX[1]
            out = []
            for at, kind in self.pile.lights:
                if kind != "flame":
                    continue
                x, s = self.pile.screen(at)
                col = (0.5 + x / wide) * gw
                row = (1 - (s + MARK * self.bed * wide / 2)) * gh
                rad = REACH * gh
                c0, c1 = max(0, int(col - 2 * rad)), min(gw, int(col + 2 * rad) + 1)
                r0, r1 = max(0, int(row - 2 * rad)), min(gh, int(row + 2 * rad) + 1)
                yy, xx = np.mgrid[r0:r1, c0:c1].astype(np.float32)
                wt = np.exp(-((xx + 0.5 - col) ** 2 + (yy + 0.5 - row) ** 2) / (2 * rad * rad))
                out.append(((slice(r0, r1), slice(c0, c1)), wt / max(float(wt.sum()), 1e-9)))
            self._windows = out
        return self._windows

    def _logs(self, t, out, lay, glow, level):
        """The logs on `out` (the box's part in the frame, changed in place): their shadow on the floor, then them,
        lit by the room, the flames near each light (`glow`: the flames on the grid) and the embers; their cracks
        glowing; `level`: how bright the fire is now (1: as on average)."""
        x0, y0, x1, y1 = lay["pile"]
        laid = lay["logs"]
        part = out[y0:y1, x0:x1]
        alpha = laid["alpha"][..., None]
        part *= 1 - FLOOR_SHADE * laid["floor"][..., None]
        room = np.maximum(out[::4, ::4].reshape(-1, 3).mean(axis=0), 0.0) ** logs.GAMMA * ROOM_ON_LOGS
        bright = self.bright_k * level
        flames = [float((glow[at] * wt).sum()) * FLAME_ON_LOGS * self.bright_k for at, wt in self._light_windows()]
        tau = self.tau(t)
        breathe = sum(math.sin(TAU * m * tau + ph) for m, ph in zip(self.ember_m, self.ember_ph)) / 3
        ember = EMBERS_ON_LOGS * self.embers_k * (0.85 + 0.15 * breathe) * bright
        embers = [ember] * sum(kind == "ember" for _, kind in self.pile.lights)
        colour = self.pile.colour(laid, room, flames, embers, CRACKS * self.embers_k * bright, tau, self.ember_m)
        np.clip(colour, 0.0, 1.0, out=colour)
        colour **= 1 / logs.GAMMA
        part += alpha * (colour - part)

    def _halo(self, flame, src, size):
        """The glow round the flames (a camera's): the flames' colours on the grid, blurred on a grid 4 times
        coarser (a wide blur: the same), enlarged to `size` over the part `src` of the grid."""
        gw, gh = GRID
        small = palette(flame, self.warmth).reshape(gh // 4, 4, gw // 4, 4, 3).mean(axis=(1, 3))
        small = blur(small, HALO_R * gh / 4) * (HALO * 255.0)
        img = Image.fromarray((np.clip(small, 0, 255) + 0.5).astype(np.uint8)).resize(
            size, Image.Resampling.BILINEAR, box=tuple(v / 4 for v in src))
        return np.asarray(img, np.float32) * (1 / 255.0)

    def draw(self, t, canvas, mask):
        """Nothing laid over the picture: apply() changes it."""

    def apply(self, t, picture, mask, gain, kept=None):
        """The fire on `picture` (float32 RGB 0..1, changed in place) in its box, `gain` (0..1, a fade) times as
        strongly, where `mask` (0..1, the clip's area; None: everywhere) says."""
        if gain <= 0:
            return
        x0, y0, w, h = self.box()
        r = self._rect(x0, y0, x0 + w, y0 + h)
        if r is None:
            return
        rx0, ry0, rx1, ry1 = r
        lay = self._layout(r, x0, y0, w, h)
        src, size = lay["src"], lay["size"]
        orig = picture[ry0:ry1, rx0:rx1]
        out = orig if mask is None and gain >= 1 else orig.copy()
        fire = self.frames()
        glow, level = fire.frame(t)
        level = min(max(level / fire.mean, 0.5), 1.5)  # (how bright the fire is now: 1 as on average)
        throws = self.light_k * LIGHT_K * math.sqrt(fire.mean / LEVEL) * (1 + 2 * (level - 1))  # (its light now)

        def big(a):  # a grid field over the part of the box in the frame
            p = np.pad(a, 1, mode="edge")  # (softened a little first: no stairs along its edges)
            a = (p[:-2] + 2 * p[1:-1] + p[2:]) * 0.25
            a = (a[:, :-2] + 2 * a[:, 1:-1] + a[:, 2:]) * 0.25
            return np.array(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize(
                size, Image.Resampling.BICUBIC, box=src), np.float32)  # (a copy: changed in place)

        # the old fire put out: its bright places, warm or burnt out white, over the bed and above it, become a dim
        # glow of their own colours (blurred: no still flame's edges left under the new flames)
        if self.cover_k > 0:
            luma = out @ LUMA
            hot = _smooth((luma - 0.2) / 0.35) * np.maximum(_smooth((out[..., 0] - out[..., 2] - 0.08) / 0.25),
                                                            _smooth((luma - 0.6) / 0.25))
            hot = np.minimum(_soft(hot, max(1.0, 0.012 * h)) * 1.5, 1.0)
            put_out = (self.cover_k * hot * lay["zone"])[..., None]
            out += put_out * (_soft(out, max(1.0, 0.04 * h)) * DIM - out)
        # its light on the firebox round it: brighter as the fire is (multiplying what is there: lit, not hazy)
        if self.light_k > 0:
            out *= 1 + lay["spill"][..., None] * (throws * LIGHT)
        hide = None
        if "logs" in lay:  # the logs (opaque), the flames behind those in front of the fire, the embers behind all
            self._logs(t, out, lay, glow, level)
            px0, py0, px1, py1 = lay["pile"]
            hide = (slice(py0, py1), slice(px0, px1)), lay["logs"]["alpha"]
        # the flames and the embers under them (their brightness on the grid, then over the box: crisp edges)
        flame = big(glow)
        flame -= CUT
        np.maximum(flame, 0.0, out=flame)
        flame *= self.bright_k / (1 - CUT)
        flame *= lay["shown"]
        if hide is not None:  # (over a log in front of the fire none of them; over one behind it only the brighter
            # tongues: no haze over it)
            part = flame[hide[0]]
            hidden = 1 - BEHIND * _smooth((part - 0.5) / 0.7)
            part *= 1 - hide[1] * (hidden + (1 - hidden) * lay["logs"]["front"])
        gflame = np.maximum(glow - CUT, 0.0) * (self.bright_k / (1 - CUT)) * self._grid_shown()
        if self.embers_k > 0:
            field, phase = self._ember_field()
            tau = self.tau(t)
            breathe = sum(np.sin(TAU * m * tau + ph + 5 * phase) for m, ph in zip(self.ember_m, self.ember_ph)) / 3
            embers = field * ((0.8 + 0.2 * breathe) * (0.7 * self.embers_k * (0.9 + 0.1 * level)))
            seen = big(embers) * lay["edge"]
            if hide is not None:
                seen[hide[0]] *= 1 - hide[1]
            flame += seen
            gflame += embers
        fire = self._lut.take((np.minimum(flame, LUT_TOP) * ((len(self._lut) - 1) / LUT_TOP)).astype(np.int32),
                              axis=0)  # (its colours)
        halo = self._halo(gflame, src, size)
        halo *= lay["edge"][..., None]
        if hide is not None:  # (less of it over the logs: not a haze over them)
            halo[hide[0]] *= 1 - 0.8 * hide[1][..., None]
        # laid with Screen: 1 - (1 - picture) (1 - flames) (1 - their glow)
        np.subtract(1.0, fire, out=fire)
        np.subtract(1.0, halo, out=halo)
        fire *= halo
        np.subtract(1.0, out, out=out)
        out *= fire
        np.subtract(1.0, out, out=out)
        if out is not orig:
            k = gain if mask is None else gain * mask[ry0:ry1, rx0:rx1, None]
            orig += k * (out - orig)
