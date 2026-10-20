"""Twinkling string lights: strands of small bulbs wound round a Christmas tree, hung in swags, falling as a curtain
or scattered (a placed effect: Effect.BOX; drawn as light, laid with Screen; see effects.py for how every effect
loops).

The box goes over the tree (or the window, the eaves). The strands are laid out in it once: round a cone standing in
the box (a spiral seen a little from the side, as a camera at the tree's middle sees it: the turns in front curve
down below the eye and up above it, the bulbs at the back mostly hidden by the tree), in swags hanging between
points, as a curtain of strands, or scattered. Bulbs sit along each strand at even spacing, nudged a little (a real
tree's branches never hold them in line), each colour in turn along a strand as on a real one.

Each bulb is a small hot core (near white), a halo of its colour and the rays a camera lens makes of a bright point.
How bright it is follows the way of twinkling chosen, as a light controller does: each bulb on its own, groups
breathing in turn, waves chasing along the strands, two groups flashing in turn, or steady; now and then a bulb
flares (its rays grow). Every frequency is a whole number of cycles per loop, so frame P is frame 0 again.

Inside the box every length is in box heights, so the preview clip (the box alone) and the video show the same.
"""
import math

import numpy as np

from . import sprites
from .effects import Effect, _choice, _colour, _lerp, _num, _place, _rgb

# the colours of the bulbs in turn along a strand ("one": the colour setting)
PALETTES = {"warm": ["#ffb347"], "multi": ["#ff2a1a", "#22d84a", "#2f6bff", "#ffb000", "#ff3fb0"],
            "cool": ["#b8d4ff"], "gold": ["#ffb347", "#ff2a1a"], "one": None}
WHITE = {"warm": 0.7, "multi": 0.45, "cool": 0.75, "gold": 0.55, "one": 0.6}  # how white a bulb's hot core is
EYE = 0.45  # the camera's height on the tree (share of the box from the top)
AWAY = 2.6  # and how far it stands (box heights): a turn of the spiral 1 box height below the eye looks this flat
HALO = 0.55  # a halo's light at its middle (times the bulb's)
CORE = 1.8  # a core's (it burns white in the middle)
CHASE = 7.0  # bulbs from one chasing wave to the next

DEFAULTS = {"amount": 45, "density": 45, "width": 62, "jitter": 40, "back": 25, "size": 40, "glow": 50, "rays": 35,
            "brightness": 75, "speed": 40, "depth": 60}
LAYOUTS = {"tree": {"width": 62, "density": 45}, "swag": {"width": 100, "density": 30},
           "curtain": {"width": 100, "density": 45}, "scatter": {"width": 80, "density": 45}}


def _p(key, label, value, group, lo=0, hi=100):
    return _num(key, label, value, lo, hi, group=group)


SCHEMA = {
    "label": "Dây đèn lấp lánh",
    "note": "Kéo khung trắng trùm lên cây thông (hoặc cửa sổ, mái hiên), kéo góc khung cho vừa. Chọn cách giăng dây: "
            "quấn quanh cây, võng ngang, rèm thả dọc hay rải đều. Muốn đèn chỉ nằm trên cây: bấm Vẽ vùng của clip "
            "này và tô lên cây.",
    "params": [
        _choice("layout", "Cách giăng dây", "tree",
                [("tree", "Quấn quanh cây thông"), ("swag", "Dây võng ngang"), ("curtain", "Rèm đèn thả dọc"),
                 ("scatter", "Rải đều trong khung")], presets=LAYOUTS, group="Kiểu"),
        _choice("mode", "Kiểu nháy", "twinkle",
                [("twinkle", "Lấp lánh từng bóng"), ("breathe", "Sáng mờ dần lần lượt"), ("chase", "Chạy đuổi"),
                 ("flash", "Nhấp nháy hai nhóm"), ("steady", "Sáng đứng")], group="Kiểu"),
        _choice("palette", "Màu đèn", "warm",
                [("warm", "Vàng ấm"), ("multi", "Nhiều màu"), ("cool", "Trắng lạnh"), ("gold", "Vàng và đỏ"),
                 ("one", "Một màu (chọn ở dưới)")], group="Kiểu"),
        _p("amount", "Số bóng", DEFAULTS["amount"], "Dây đèn"),
        _p("density", "Dây quấn dày (số vòng, số hàng)", DEFAULTS["density"], "Dây đèn"),
        _p("width", "Độ rộng (so với khung trắng)", DEFAULTS["width"], "Dây đèn", 10, 100),
        _p("jitter", "Lộn xộn (không thẳng hàng)", DEFAULTS["jitter"], "Dây đèn"),
        _p("back", "Bóng phía sau cây lộ ra", DEFAULTS["back"], "Dây đèn"),
        _p("size", "Cỡ bóng", DEFAULTS["size"], "Bóng đèn"),
        _p("glow", "Quầng sáng", DEFAULTS["glow"], "Bóng đèn"),
        _p("rays", "Tia sáng lấp lánh", DEFAULTS["rays"], "Bóng đèn"),
        _p("brightness", "Độ sáng", DEFAULTS["brightness"], "Bóng đèn"),
        _p("speed", "Tốc độ nháy", DEFAULTS["speed"], "Nháy"),
        _p("depth", "Độ nháy (sáng tối rõ)", DEFAULTS["depth"], "Nháy"),
        _colour("color", "Màu (khi chọn Một màu)", "#ffb347", group="Màu"),
        *_place(60)]}


def _even(lines, n, starts):
    """n bulbs at even spacing along polylines (each (m, 3): x, y, z in box heights; the spacing measured in 3D), the
    first one `starts` (0..1 for each line) of a step from its start: (points (k, 3), strand number, place along its
    strand)."""
    lengths = [np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))] for line in lines]
    step = sum(c[-1] for c in lengths) / max(n, 1)
    pts, strand, along = [], [], []
    for i, (line, c) in enumerate(zip(lines, lengths)):
        s = np.arange(step * starts[i], c[-1], step)
        if not len(s):
            continue
        pts.append(np.stack([np.interp(s, c, line[:, k]) for k in range(3)], 1))
        strand.append(np.full(len(s), i))
        along.append(np.arange(len(s)))
    if not pts:
        return np.zeros((0, 3)), np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(pts), np.concatenate(strand), np.concatenate(along)


def _core(r, fx, fy):
    """A bulb's hot core: a disc `r` px, a soft edge, hotter in the middle; one smaller than a pixel lights only its
    share of it."""
    x0, y0, xs, ys = sprites._grid(r + 1.5, r + 1.5, fx, fy)
    d = np.sqrt(xs * xs + ys * ys)
    out = np.clip(r + 0.5 - d, 0.0, 1.0) * (0.75 + 0.25 * np.exp(-d * d / (2 * (0.5 * r + 0.3) ** 2)))
    area = math.pi * r * r
    total = float(out.sum())
    if total > area > 0:
        out *= area / total
    return out.astype(np.float32), x0, y0


def _halo(sigma, fx, fy):
    x0, y0, xs, ys = sprites._grid(2.6 * sigma + 1, 2.6 * sigma + 1, fx, fy)
    return np.exp(-(xs * xs + ys * ys) / (2 * sigma * sigma)).astype(np.float32), x0, y0


def _rays(length, fx, fy):
    """The four rays a lens makes of a bright point (across and up and down), thin, fading to their ends."""
    x0, y0, xs, ys = sprites._grid(length + 1, length + 1, fx, fy)
    across = np.exp(-ys * ys / 0.5) * np.clip(1 - np.abs(xs) / length, 0, 1) ** 2.5
    down = np.exp(-xs * xs / 0.5) * np.clip(1 - np.abs(ys) / length, 0, 1) ** 2.5
    return np.maximum(across, down).astype(np.float32), x0, y0


class Lights(Effect):
    blend = "screen"
    BOX = (1, 1)
    CAP = 600  # bulbs at most

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        self.bw = self.BOX[0] / self.BOX[1]  # the box's width, in box heights
        n = int(round(_lerp(20, 360, p["amount"] / 100)))
        dens, width, jitter = p["density"] / 100, p["width"] / 100 * self.bw, p["jitter"] / 100
        layout = p["layout"]
        g = np.random.default_rng([self.seed, 1103, 1])
        lines = self._lines(layout, dens, width, g)
        if layout == "scatter":
            k = np.random.default_rng([self.seed, 1103, 2]).random((n, 2))
            pts = np.stack([self.bw / 2 + (k[:, 0] - 0.5) * width, 0.03 + 0.94 * k[:, 1], np.zeros(n)], 1)
            strand, along = np.zeros(n, int), np.argsort(np.argsort(pts[:, 0] + pts[:, 1]))
            step = math.sqrt(width * 0.94 / max(n, 1))
        else:
            pts, strand, along = _even(lines, n, 0.1 + 0.8 * g.random(len(lines)))  # (curtain: not in rows)
            step = sum(np.linalg.norm(np.diff(line, axis=0), axis=1).sum() for line in lines) / max(n, 1)
        k = len(pts)
        u = np.random.default_rng([self.seed, 1103, 3]).random((self.CAP, 12))[:k]
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        if layout == "tree":  # seen from in front of the tree's middle: below the eye a turn in front hangs lower
            front = np.clip(0.5 + 2 * z / np.maximum(np.abs(self.radius(y)), 1e-6), 0, 1)
            y = y + (y - EYE) / AWAY * z
            vis = front + (1 - front) * p["back"] / 100 * (0.25 + 0.75 * u[:, 0])
        else:
            vis = np.ones(k)
        # nudged off the line (branches), a little each way
        spread = jitter * step * 0.55
        self.x = x + spread * (u[:, 1] - 0.5) * 2 * (0.4 + 0.6 * u[:, 2])
        self.y = y + spread * (u[:, 3] - 0.5) * 2 * (0.4 + 0.6 * u[:, 4])
        self.vis = vis * (1 - 0.35 * jitter * u[:, 5])  # (half hidden among the needles now and then)
        self.strand, self.along = strand, along
        # colours: in turn along each strand
        pal = PALETTES[p["palette"]] or [p["color"]]
        halo = np.array([_rgb(c) for c in pal], np.float32)
        which = (along + 2 * strand) % len(pal)
        white = WHITE[p["palette"]]
        self.c_halo = halo[which]
        self.c_core = self.c_halo * (1 - white) + white
        # sizes, box heights (a bulb at the back a little smaller)
        self.core = _lerp(0.0015, 0.009, p["size"] / 100) * (0.8 + 0.2 * np.minimum(vis, 1)) * (0.85 + 0.3 * u[:, 6])
        self.halo = self.core * _lerp(2.0, 7.0, p["glow"] / 100)
        self.bright = p["brightness"] / 100 * 1.3 * (0.8 + 0.2 * u[:, 7])
        self.rays_k = p["rays"] / 100
        self.depth = p["depth"] / 100
        self.mode = p["mode"]
        # twinkling, each a whole number of cycles per loop
        sp = _lerp(0.25, 2.5, p["speed"] / 100)
        self.q1 = self.cycles(_lerp(0.15, 0.5, u[:, 8]) * sp)
        self.q2 = self.cycles(_lerp(0.4, 1.2, u[:, 9]) * sp)
        self.f1, self.f2 = u[:, 10] * 2 * math.pi, u[:, 11] * 2 * math.pi
        g2 = np.random.default_rng([self.seed, 1103, 4]).random((self.CAP, 2))[:k]
        self.q3 = self.cycles(_lerp(0.04, 0.12, g2[:, 0]) * sp)  # a flare now and then
        self.f3 = g2[:, 1] * 2 * math.pi
        self.m_breathe = self.cycles(0.12 * sp)
        self.m_chase = self.cycles(0.6 * sp)
        self.m_flash = self.cycles(0.5 * sp)
        self.group = along % 3 if self.mode == "breathe" else along % 2
        self.cache = (sprites.SpriteCache(lambda key, fx, fy: _core(key, fx, fy)),
                      sprites.SpriteCache(lambda key, fx, fy: _halo(key, fx, fy)),
                      sprites.SpriteCache(lambda key, fx, fy: _rays(key, fx, fy)))

    def radius(self, y):
        """The tree's radius at height y (box heights from the top)."""
        top, bottom = 0.04, 0.97
        return self.cone_r * np.clip((y - top) / (bottom - top), 0, 1)

    def _lines(self, layout, dens, width, g):
        """The strands as polylines (x, y, z: box heights, y down, z towards the eye)."""
        cx = self.bw / 2
        self.cone_r = width / 2
        if layout == "tree":
            turns = _lerp(3, 14, dens)
            s = np.linspace(0, 1, 3000)
            y = _lerp(0.09, 0.96, s)
            r = self.radius(y)
            th = 2 * math.pi * turns * s + g.random() * 2 * math.pi
            return [np.stack([cx + r * np.sin(th), y, r * np.cos(th)], 1)]
        if layout == "swag":
            rows = int(round(_lerp(1, 5, dens)))
            out = []
            for i in range(rows):
                top = 0.05 + 0.7 * i / max(rows - 1, 1)
                swags = 3 + i % 2  # (every other row its swags shorter: hung between other points)
                s = np.linspace(0, swags, 1200)
                x = cx - width / 2 + s / swags * width
                y = top + 0.28 * width / swags * (1 - (2 * (s % 1) - 1) ** 2) * (0.9 + 0.2 * g.random())
                out.append(np.stack([x, y, np.zeros_like(x)], 1))
            return out
        if layout == "curtain":
            m = int(round(_lerp(5, 30, dens)))
            ends = 0.97 - 0.06 * g.random(m)
            return [np.stack([np.full(2, cx - width / 2 + width * (i + 0.5) / m), [0.03, ends[i]], np.zeros(2)], 1)
                    for i in range(m)]
        return []

    def levels(self, tau):
        """How bright each bulb is at phase tau (0..1 of its brightest), and how much it flares (0..1)."""
        d = self.depth
        two = 2 * math.pi
        if self.mode == "twinkle":
            w = 0.5 + 0.5 * (0.65 * np.sin(two * self.q1 * tau + self.f1) + 0.35 * np.sin(two * self.q2 * tau + self.f2))
            level = 1 - d * (1 - w ** 1.5)
        elif self.mode == "breathe":
            w = 0.5 - 0.5 * np.cos(two * (self.m_breathe * tau + self.group / 3))
            level = 1 - d * w
        elif self.mode == "chase":
            w = (0.5 + 0.5 * np.cos(two * (self.m_chase * tau - self.along / CHASE))) ** 2
            level = 1 - d * (1 - w)
        elif self.mode == "flash":
            sq = np.clip(np.sin(two * self.m_flash * tau + math.pi * self.group) / 0.35 + 0.5, 0, 1)
            level = 1 - d * (1 - sq * sq * (3 - 2 * sq))
        else:
            level = 1 - 0.12 * d * (0.5 + 0.5 * np.sin(two * self.q1 * tau + self.f1))
        flare = np.maximum(0.0, np.sin(two * self.q3 * tau + self.f3)) ** 24 if self.mode != "steady" else 0.0
        return level, flare * np.ones_like(level)

    def draw(self, t, canvas, mask):
        x0, y0, _, h = self.box()
        if h < 2:
            return
        level, flare = self.levels(self.tau(t))
        lit = level * self.vis * self.bright
        xs, ys = x0 + self.x * h, y0 + self.y * h
        core, halo, rays = self.cache
        reach = self.halo * h * 2.6 + 2
        for i in np.flatnonzero(lit > 0.01):
            if xs[i] + reach[i] < 0 or ys[i] + reach[i] < 0 or xs[i] - reach[i] > self.w or ys[i] - reach[i] > self.h:
                continue
            ix, iy = math.floor(xs[i]), math.floor(ys[i])
            fx, fy = xs[i] - ix, ys[i] - iy
            spot, sx, sy = halo.get(sprites.bucket(self.halo[i] * h), fx, fy)
            canvas.light(ix + sx, iy + sy, spot, self.c_halo[i] * (HALO * lit[i]), mask)
            spot, sx, sy = core.get(sprites.bucket(self.core[i] * h), fx, fy)
            canvas.light(ix + sx, iy + sy, spot, self.c_core[i] * (CORE * lit[i]), mask)
            ray = self.rays_k * lit[i] * (0.3 + 1.2 * flare[i])
            if ray > 0.05:
                length = self.core[i] * h * _lerp(3.0, 14.0, self.rays_k) * (0.6 + 0.8 * flare[i])
                if length >= 2:
                    spot, sx, sy = rays.get(sprites.bucket(length), fx, fy)
                    canvas.light(ix + sx, iy + sy, spot, self.c_core[i] * min(ray, 1.5), mask)
