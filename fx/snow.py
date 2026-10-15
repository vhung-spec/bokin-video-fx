"""Falling snow (see effects.py for how every effect loops).

Flakes at every depth: far ones tiny, faint, slow and a little bluer; nearer ones bigger and quicker; a few right at
the lens, out of focus (soft discs with a faint brighter rim). Mid-depth flakes may be six-armed ice crystals, drawn
by code, that turn and tumble. Every flake flutters (side sway at two whole-number frequencies per loop, slowing at
the ends of each swing), drifts in gentle swirls (a smooth flow field that changes with tau) and in gusts that sweep
across the frame, and some of them glint now and then.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import sprites
from .effects import Falling, _choice, _colour, _lerp, _num, _rgb

LOOKS = {  # the "Kiểu tuyết" presets: every setting but the colours ("medium" is the defaults, added below)
    "light": {"flake": "mixed", "amount": 22, "size": 35, "speed": 25, "sway": 45, "swirl": 25, "spin": 35,
              "wind": 5, "gust": 15, "depth": 60, "blur": 35, "near": 10, "twinkle": 15, "glow": 0, "opacity": 80},
    "heavy": {"flake": "mixed", "amount": 85, "size": 50, "speed": 45, "sway": 35, "swirl": 35, "spin": 40,
              "wind": 15, "gust": 30, "depth": 70, "blur": 50, "near": 35, "twinkle": 10, "glow": 0, "opacity": 90},
    "blizzard": {"flake": "soft", "amount": 100, "size": 45, "speed": 90, "sway": 30, "swirl": 70, "spin": 60,
                 "wind": 75, "gust": 80, "depth": 75, "blur": 60, "near": 45, "twinkle": 5, "glow": 0,
                 "opacity": 90},
    "sparkle": {"flake": "crystal", "amount": 35, "size": 50, "speed": 25, "sway": 40, "swirl": 25, "spin": 50,
                "wind": 5, "gust": 15, "depth": 60, "blur": 30, "near": 15, "twinkle": 70, "glow": 35,
                "opacity": 90},
    "dreamy": {"flake": "mixed", "amount": 30, "size": 75, "speed": 15, "sway": 60, "swirl": 40, "spin": 35,
               "wind": 5, "gust": 10, "depth": 80, "blur": 70, "near": 45, "twinkle": 20, "glow": 25,
               "opacity": 85},
}

SCHEMA = {"label": "Tuyết rơi", "params": [
    _choice("look", "Kiểu tuyết", "medium",
            [("light", "Tuyết nhẹ"), ("medium", "Tuyết vừa"), ("heavy", "Tuyết dày"), ("blizzard", "Bão tuyết"),
             ("sparkle", "Tuyết lấp lánh"), ("dreamy", "Bông to bay chậm")], presets=LOOKS, group="Kiểu"),
    _choice("flake", "Hình bông tuyết", "mixed",
            [("soft", "Tròn mềm"), ("crystal", "Tinh thể sáu cánh"), ("mixed", "Trộn cả hai")], group="Kiểu"),
    _num("amount", "Mật độ", 45, group="Bông tuyết"), _num("size", "Cỡ bông", 40, group="Bông tuyết"),
    _num("speed", "Tốc độ rơi", 35, group="Chuyển động"), _num("sway", "Đung đưa", 40, group="Chuyển động"),
    _num("swirl", "Xoáy nhẹ theo gió", 35, group="Chuyển động"),
    _num("spin", "Xoay, lật bông tinh thể", 40, group="Chuyển động"),
    _num("wind", "Gió (trái ← → phải)", 10, -100, 100, group="Gió"),
    _num("gust", "Gió giật từng cơn", 25, group="Gió"),
    _num("depth", "Chiều sâu (xa nhỏ, gần to)", 60, group="Chiều sâu"),
    _num("blur", "Nhoè bông ở gần", 40, group="Chiều sâu"),
    _num("near", "Bông to sát ống kính", 25, group="Chiều sâu"),
    _num("twinkle", "Lấp lánh", 15, group="Ánh sáng"), _num("glow", "Quầng sáng quanh bông", 0, group="Ánh sáng"),
    _num("opacity", "Độ đậm", 85, group="Màu"), _colour("color", "Màu", "#ffffff", group="Màu"),
    _colour("far_color", "Màu bông ở xa", "#d8e2ef", group="Màu")]}
LOOKS["medium"] = {p["key"]: p["value"] for p in SCHEMA["params"] if p["key"] in LOOKS["light"]}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _grid(hx, hy, fx, fy):
    """Pixel centres around a particle at sub-pixel offset (fx, fy) of its own pixel (see sprites.py)."""
    x0, y0 = -int(math.ceil(hx)), -int(math.ceil(hy))
    xs = np.arange(x0, -x0 + 1, dtype=np.float32) + 0.5 - fx
    ys = np.arange(y0, -y0 + 1, dtype=np.float32) + 0.5 - fy
    return x0, y0, xs[None, :], ys[:, None]


def flake(radius, soft, rim, dx, dy, fx, fy):
    """A round flake: a disc of `radius` px with edges `soft` px wide, a little brighter in the middle (or, with
    `rim` 0..1, towards its rim, as an out-of-focus flake looks), smeared over the (dx, dy) px it moves while a
    camera's shutter is open. Returns (coverage, x0, y0)."""
    soft = max(soft, 0.7)
    move = math.hypot(dx, dy)
    reach = radius + soft / 2 + 1
    x0, y0, xs, ys = _grid(reach + abs(dx) / 2, reach + abs(dy) / 2, fx, fy)
    if move > 0.5:
        ux, uy = dx / move, dy / move
        along = xs * ux + ys * uy
        across = xs * uy - ys * ux
        across *= across
        half = move / 2
        beyond = np.maximum(np.abs(along) - half, 0.0)
        d = np.sqrt(beyond * beyond + across)  # distance to the path of the centre
        # how long the disc covers each pixel while it moves: the streak fades at both ends
        chord = np.sqrt(np.maximum(radius * radius - across, 0.25))
        stay = np.minimum(along + chord, half) - np.maximum(along - chord, -half)
        stay *= 1 / move
    else:
        d = np.sqrt(xs * xs + ys * ys)
        stay = None
    out = np.clip((radius - d) * (1 / soft) + 0.5, 0.0, 1.0)
    out = out * out * (3 - 2 * out)
    if stay is not None:
        out *= np.clip(stay, 0.0, 1.0)
    if radius < 1.5:  # a flake smaller than a pixel only covers part of it
        total = float(out.sum())
        area = math.pi * radius * radius
        if total > area > 0:
            out *= area / total
    if radius >= 2.0 or rim > 0:
        q = np.minimum(d * (1 / radius), 1.0)
        if rim > 0:
            out *= 1.0 - (1.0 - rim) * 0.3 * q * q - rim * 0.45 * (1.0 - _smooth((q - 0.6) / 0.35))
        else:
            out *= 1.0 - 0.3 * q * q
    return out, x0, y0


def halo(spread, fx, fy):
    """The faint glow around a flake: a wide soft spot, `spread` px (its standard deviation), peak 1."""
    x0, y0, xs, ys = _grid(2.5 * spread, 2.5 * spread, fx, fy)
    return np.exp((xs * xs + ys * ys) * (-0.5 / (spread * spread))), x0, y0


def glint(length, fx, fy):
    """A glint of light on a flake: a bright point with four thin rays `length` px long."""
    x0, y0, xs, ys = _grid(length + 1.5, length + 1.5, fx, fy)
    w = 2 * 0.45 ** 2
    rays = np.maximum(np.exp(-ys * ys / w) * np.clip(1 - np.abs(xs) / length, 0, 1) ** 2.5,
                      np.exp(-xs * xs / w) * np.clip(1 - np.abs(ys) / length, 0, 1) ** 2.5)
    core = np.exp(-(xs * xs + ys * ys) / 2.0)
    return np.minimum(0.8 * rays + core, 1.0).astype(np.float32), x0, y0


def smear(cover, dx, dy):
    """`cover` smeared over the (dx, dy) px it moves while the shutter is open: copies along the way, at most a
    pixel apart (each one split between four pixels). Returns (coverage, x0, y0): where its top-left pixel now sits
    relative to the original's."""
    n = int(math.ceil(math.hypot(dx, dy))) + 1
    h, w = cover.shape
    px, py = int(math.ceil(abs(dx) / 2)) + 1, int(math.ceil(abs(dy) / 2)) + 1
    out = np.zeros((h + 2 * py, w + 2 * px), np.float32)
    for k in range(n):
        o = k / (n - 1) - 0.5
        sx, sy = px + o * dx, py + o * dy
        ix, iy = math.floor(sx), math.floor(sy)
        fx, fy = sx - ix, sy - iy
        out[iy:iy + h, ix:ix + w] += cover * ((1 - fx) * (1 - fy))
        out[iy:iy + h, ix + 1:ix + 1 + w] += cover * (fx * (1 - fy))
        out[iy + 1:iy + 1 + h, ix:ix + w] += cover * ((1 - fx) * fy)
        out[iy + 1:iy + 1 + h, ix + 1:ix + 1 + w] += cover * (fx * fy)
    out *= 1 / n
    return out, -px, -py


# --- ice crystals -------------------------------------------------------------------------------------------

CRYSTAL_KINDS = ("dendrite", "fern", "plate", "star", "sector", "dendrite", "fern", "plate")
CRYSTAL_SIZES = (12, 16, 22, 30, 42, 58, 80, 112, 156)  # px across; the arms reach 0.45 of it from the middle
_DESIGNS, _CRYSTALS = {}, {}


def crystal_design(shape):
    """The design of crystal `shape`, the same on all six arms: its branches (where, how long), plates, widths."""
    if shape not in _DESIGNS:
        rng = np.random.default_rng([shape, 6021])
        kind = CRYSTAL_KINDS[shape % len(CRYSTAL_KINDS)]
        pairs = {"dendrite": rng.integers(3, 5), "fern": rng.integers(5, 8), "plate": rng.integers(1, 3),
                 "star": 1, "sector": 0}[kind]
        reach = {"dendrite": rng.uniform(0.4, 0.55), "fern": rng.uniform(0.22, 0.32), "plate": rng.uniform(0.2, 0.3),
                 "star": rng.uniform(0.15, 0.22), "sector": 0.0}[kind]
        at = np.linspace(rng.uniform(0.25, 0.4), rng.uniform(0.72, 0.86), pairs) if pairs else []
        branches = [(float(f), float(reach * (1 - f) ** 0.7 * rng.uniform(0.85, 1.15))) for f in at]
        _DESIGNS[shape] = {
            "kind": kind, "branches": branches, "twigs": kind == "dendrite",
            "width": rng.uniform(0.045, 0.07) * (1.6 if kind == "sector" else 1.0),
            "plate": {"plate": rng.uniform(0.16, 0.24), "sector": rng.uniform(0.5, 0.62),
                      "star": rng.uniform(0.08, 0.12)}.get(kind, rng.uniform(0.05, 0.1)),
            "tip": rng.uniform(0.045, 0.065) if kind == "plate" else 0.0}
    return _DESIGNS[shape]


def _hexagon(cx, cy, r, turn):
    return [(cx + r * math.cos(turn + k * math.pi / 3), cy + r * math.sin(turn + k * math.pi / 3)) for k in range(6)]


def crystal(shape, size):
    """Crystal `shape` drawn `size` px across as coverage 0..1 (a Pillow "L" image): six arms with side branches,
    a translucent plate in the middle and a soft icy glow. No stroke is thinner than about a pixel."""
    key = (shape, size)
    if key in _CRYSTALS:
        return _CRYSTALS[key]
    design = crystal_design(shape)
    ss = 4  # supersampling
    big = size * ss
    c, arm = big / 2, 0.45 * big
    width = max(design["width"] * arm, 0.9 * ss)
    plate = Image.new("L", (big, big), 0)
    lines = Image.new("L", (big, big), 0)
    pd, ld = ImageDraw.Draw(plate), ImageDraw.Draw(lines)

    def bar(x0, y0, x1, y1, w0, w1):  # a tapered stroke
        ln = math.hypot(x1 - x0, y1 - y0) or 1.0
        nx, ny = -(y1 - y0) / ln, (x1 - x0) / ln
        ld.polygon([(x0 + nx * w0 / 2, y0 + ny * w0 / 2), (x1 + nx * w1 / 2, y1 + ny * w1 / 2),
                    (x1 - nx * w1 / 2, y1 - ny * w1 / 2), (x0 - nx * w0 / 2, y0 - ny * w0 / 2)], fill=255)

    up = -math.pi / 2
    pd.polygon(_hexagon(c, c, design["plate"] * arm, up), fill=90 if design["kind"] == "sector" else 150)
    thin = max(0.55 * width, 0.9 * ss)
    for k in range(6):
        a = up + k * math.pi / 3
        dx, dy = math.cos(a), math.sin(a)
        bar(c, c, c + dx * arm, c + dy * arm, width, max(0.5 * width, 0.9 * ss))
        for f, ln in design["branches"]:
            bx, by = c + dx * f * arm, c + dy * f * arm
            for side in (-1, 1):
                b = a + side * math.pi / 3
                ex, ey = bx + math.cos(b) * ln * arm, by + math.sin(b) * ln * arm
                bar(bx, by, ex, ey, thin, max(0.6 * thin, 0.9 * ss))
                if design["twigs"] and ln > 0.14:  # little twigs on long branches, parallel to the arm
                    tx, ty = bx + math.cos(b) * 0.55 * ln * arm, by + math.sin(b) * 0.55 * ln * arm
                    for s2 in (-1, 1):
                        t2 = b + s2 * math.pi / 3
                        bar(tx, ty, tx + math.cos(t2) * 0.3 * ln * arm, ty + math.sin(t2) * 0.3 * ln * arm,
                            max(0.8 * thin, 0.9 * ss), 0.9 * ss)
        if design["tip"]:
            pd.polygon(_hexagon(c + dx * arm * 0.94, c + dy * arm * 0.94, design["tip"] * arm, up), fill=170)
    cover = np.maximum(np.asarray(plate.resize((size, size), Image.Resampling.LANCZOS), np.float32),
                       np.asarray(lines.resize((size, size), Image.Resampling.LANCZOS), np.float32)) / 255.0
    glow = np.asarray(Image.fromarray((cover * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(max(0.8, 0.05 * size))), np.float32) / 255.0
    cover = np.clip(cover + 0.4 * glow, 0.0, 1.0)
    img = Image.fromarray((cover * 255 + 0.5).astype(np.uint8), "L")
    _CRYSTALS[key] = img
    return img


# --- gusts --------------------------------------------------------------------------------------------------

def _gust_curve(n=1024, sharp=4.0):
    """How far a gust has pushed the flakes over one gust cycle (0..2 pi), from -0.5 to 0.5: the wind blows in
    short bursts (exp(sharp * (cos - 1))) and a little less between them, so the push is periodic."""
    th = np.arange(n) * (2 * math.pi / n)
    push = np.exp(sharp * (np.cos(th) - 1))
    push -= push.mean()
    curve = np.cumsum(push)
    curve -= 0.5 * (curve.max() + curve.min())
    curve /= curve.max() - curve.min()
    return np.append(curve, curve[0])


_GUST = _gust_curve()


def gust_push(theta):
    return np.interp(np.mod(theta, 2 * math.pi) * (len(_GUST) - 1) / (2 * math.pi), np.arange(len(_GUST)), _GUST)


class Snow(Falling):
    CAP = 6000  # flakes with fixed random draws
    NEAR = 240  # flakes right at the lens (their own draws)
    VERSION = 2
    SHARP = 16  # how brief a glint is
    SHUTTER = 0.35  # a frame's share the camera's shutter is open: how much a moving flake smears
    CRISP = 0.4  # crystals smear less, so their arms stay clear

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 27))
        v = np.random.default_rng([self.seed, 1103, 2]).random((self.NEAR, 27))
        flow = np.random.default_rng([self.seed, 1103, 3])
        u = np.concatenate([u, v])
        near = np.arange(len(u)) >= self.CAP
        ds = p["depth"] / 100
        z = np.where(near, 1.0 + 0.5 * u[:, 0], u[:, 0] ** 1.5)  # 0 far .. 1 near; above 1: at the lens
        zc = np.minimum(z, 1.0)

        # size, focus and strength (pixels of a 1080-line frame)
        size_k = (1 - ds) + ds * (0.2 + 2.5 * zc ** 2)
        radius = _lerp(0.9, 4.5, p["size"] / 100) * size_k * (0.7 + 0.6 * u[:, 4])
        radius = np.where(near, _lerp(8, 26, p["size"] / 100) * (0.6 + 0.9 * u[:, 4]), radius)
        defocus = p["blur"] / 100 * np.clip((zc - 0.8) / 0.2, 0, 1) ** 1.5
        defocus = np.where(near, 0.45 + 0.55 * p["blur"] / 100, defocus)
        soft = np.maximum(0.75, 0.25 * radius) + defocus * radius * np.where(near, 0.7, 1.8)
        radius = radius * (1 + np.where(near, 0.4, 0.5) * defocus)
        opacity = p["opacity"] / 100 * _lerp(0.45, 1.0, zc ** 0.6) * (0.8 + 0.2 * u[:, 8]) * (1 - 0.55 * defocus)
        opacity = np.where(near, p["opacity"] / 100 * _lerp(0.2, 0.45, u[:, 8]) * (1 - 0.3 * defocus), opacity)
        rim = np.where(near, 0.6, 0.3) * np.minimum(1.0, 1.4 * defocus)
        glow = p["glow"] / 100
        mix = _smooth(zc ** 0.7)[:, None]
        colour = _rgb(p["far_color"]) * (1 - mix) + _rgb(p["color"]) * mix

        # ice crystals: mid-depth flakes big enough to show their arms
        style = p["flake"]
        tip = 2.2 * radius  # how far a crystal's arms reach
        spin = p["spin"] / 100

        # motion (frame heights)
        speed_k = (1 - ds) + ds * (0.3 + 1.5 * zc ** 1.5)
        speed = _lerp(0.08, 0.8, p["speed"] / 100) * speed_k * (0.85 + 0.3 * u[:, 3])
        speed = speed * np.where(near, _lerp(1.6, 2.4, u[:, 3]), 1.0)
        streak = self.SHUTTER * speed * self.h / self.fps  # pixels a flake smears over
        crystal_ = (~near) & (defocus < 0.25) & (tip * px >= 4.0) & (streak < 1.6 * tip * px) & (
            (style == "crystal") | ((style == "mixed") & (u[:, 13] < 0.5)))
        angle = math.radians(45) * p["wind"] / 100 + np.radians(5) * (u[:, 9] - 0.5)
        sway = p["sway"] / 100 * _lerp(0.008, 0.05, u[:, 5]) * (0.4 + 0.8 * z)
        self.swirl = p["swirl"] / 100 * 0.07
        swirl = self.swirl * (0.35 + 0.65 * z)
        self.gust_sign = 1.0 if p["wind"] >= 0 else -1.0
        push = p["gust"] / 100 * 0.32 * (0.3 + 0.7 * z)
        # how far a flake's picture reaches from its centre: wherever it may be (whatever its kind and glow) for
        # its path, so those settings leave every flake where it was; as it is, to skip the flakes out of sight
        halo_px = np.where(glow > 0, 2.5 * (1.2 * np.where(crystal_, tip, radius) + 1.5), 0.0)
        reach = np.maximum(np.maximum(np.where(crystal_, tip, radius) + soft, 2 * radius), halo_px) * px + 3
        most = np.maximum(tip + soft, 2.5 * (1.2 * tip + 1.5)) * px + 3
        margin_y = (most + streak) / self.h + 0.3 * sway + swirl + 0.01
        margin_x = most / self.h + 1.4 * sway + swirl + 0.5 * push + 0.02
        share = self.paths(u[:, 1], u[:, 2], speed, angle, margin_y, margin_x)

        # how many: about `wanted` on screen at a time (counted with the margins around it, where the flakes
        # also are), the first candidates of each kind; drawn far to near
        scale = self.aspect / (16 / 9)
        wider = [(1 + 2 * float(margin_x[k].mean()) / self.aspect) * (1 + 2 * float(margin_y[k].mean()))
                 for k in (~near, near)]
        wanted = p["amount"] / 100 * 1100 * scale * wider[0]
        wanted_near = p["near"] / 100 * math.sqrt(p["amount"] / 100) * 28 * scale * wider[1]
        pick = np.concatenate([self._first(share[:self.CAP], wanted),
                               self.CAP + self._first(share[self.CAP:], wanted_near)])
        order = pick[np.argsort(z[pick], kind="stable")]
        for name in ("cos", "sin", "k", "len", "extra", "s0", "xe", "speed", "my"):
            setattr(self, name, getattr(self, name)[order])
        u = u[order]
        self.n = len(order)
        self.z = z[order]
        self.radius, self.soft, self.rim = radius[order] * px, soft[order] * px, rim[order]
        self.opacity, self.colour, self.reach = opacity[order], colour[order], reach[order]
        self.near = near[order]
        self.sharp = defocus[order] < 0.3
        self.sway, self.swirl_k, self.push = sway[order], swirl[order], push[order]
        self.m1, self.m2 = self.cycles(_lerp(0.12, 0.35, u[:, 6])), self.cycles(_lerp(0.5, 1.0, u[:, 10]))
        self.phi1, self.phi2 = u[:, 7] * 2 * math.pi, u[:, 11] * 2 * math.pi
        self.a2 = _lerp(0.15, 0.4, u[:, 12])

        # the flow: three slow waves of a stream function; the flakes move along its curl (they swirl, never
        # bunch up), far flakes see it smaller, and each depth sees another part of it
        f = flow.random((3, 6))
        turn = f[:, 0] * 2 * math.pi
        wave = 2 * math.pi / _lerp(0.35, 0.9, f[:, 1])
        self.fk = np.stack([wave * np.cos(turn), wave * np.sin(turn)])  # (2, 3)
        self.fdir = np.stack([-np.sin(turn), np.cos(turn)])
        self.fm = np.rint(_lerp(0.03, 0.09, f[:, 2]) * self.loop_s)  # whole cycles per loop (0: a still pattern)
        self.fph = f[:, 3] * 2 * math.pi
        self.famp = np.array([1.0, 0.7, 0.5]) / 1.4
        self.fzoom = _lerp(1.7, 1.0, np.minimum(self.z, 1.0))
        self.fdepth = (self.z * 2.3)[:, None] * (f[:, 4] * 2 * math.pi)[None, :]
        self.gm = max(1.0, round(self.loop_s / 9))  # gusts per loop
        self.gph = float(f[0, 5]) * 2 * math.pi

        # crystals turn (whole turns per loop) and tumble (seen at a slant they look narrower)
        self.crystal = crystal_[order]
        self.shape = (u[:, 14] * 8).astype(int) % 8
        self.tip = tip[order] * px
        self.rot0 = u[:, 15] * 2 * math.pi
        self.turns = np.sign(u[:, 16] - 0.5) * np.rint(spin * _lerp(0.02, 0.15, u[:, 17]) * self.loop_s)
        self.wobble = spin * 0.5 * u[:, 18]
        self.mw, self.phw = self.cycles(_lerp(0.1, 0.3, u[:, 19])), u[:, 20] * 2 * math.pi
        self.tumble = spin * _lerp(0.15, 0.5, u[:, 21])
        self.mt, self.pht = self.cycles(_lerp(0.08, 0.25, u[:, 22])), u[:, 23] * 2 * math.pi

        # the glow: sharp flakes big enough to light the air around them
        self.halo = np.where(self.sharp & (self.radius >= 0.9 * px), 0.25 * glow, 0.0)
        self.spread = 1.2 * np.where(self.crystal, self.tip, self.radius) + 1.5 * px

        # glints: a share of the sharp flakes catch the light now and then
        tw = p["twinkle"] / 100
        self.glints = np.where((u[:, 24] < 0.08 + 0.42 * tw) & self.sharp & (tw > 0), tw, 0.0)
        self.q = self.cycles(_lerp(0.1, 0.35, u[:, 25]))
        self.qph = u[:, 26] * 2 * math.pi

        self.dots = sprites.SpriteCache(lambda key, fx, fy: flake(*key, 0.0, 0.0, fx, fy))
        self.halos = sprites.SpriteCache(halo)
        self.sparks = sprites.SpriteCache(lambda key, fx, fy: glint(key, fx, fy))

    @staticmethod
    def _first(share, wanted):
        """The first candidates that make about `wanted` on screen at a time."""
        n = int(np.searchsorted(np.cumsum(share), wanted)) + 1 if wanted > 0 else 0
        return np.arange(min(n, len(share)))

    def at(self, t, dt=0.0):
        """x, y (pixels) of every flake at frame t + dt, and whether it is on its visible stretch at frame t. Along
        its path a flake is followed from where it is at t (not started again at the top during dt)."""
        s = np.mod(self.s0 + self.k * self.len * self.tau(t), self.len)
        on = s >= self.extra
        d = s + self.k * self.len * (dt / self.P) - self.extra
        x, y = self.xe + d * self.sin, -self.my + d * self.cos
        tau = self.tau(t + dt)
        sway = 2 * math.pi * self.m1 * tau + self.phi1
        dx = self.sway * (np.sin(sway) + self.a2 * np.sin(2 * math.pi * self.m2 * tau + self.phi2))
        dy = 0.18 * self.sway * np.sin(2 * sway)  # a flake falls slower at the ends of its swing
        if self.swirl > 0:
            zx, zy = x * self.fzoom, y * self.fzoom
            phase = (zx[:, None] * self.fk[0] + zy[:, None] * self.fk[1] + 2 * math.pi * self.fm * tau + self.fph
                     + self.fdepth)
            c = np.cos(phase) * self.famp
            dx = dx + self.swirl_k * (c @ self.fdir[0])
            dy = dy + self.swirl_k * (c @ self.fdir[1])
        if self.push.any():  # a gust sweeps across the frame from the side the wind comes from
            theta = 2 * math.pi * self.gm * tau + self.gph - 2 * math.pi * 0.35 * self.gust_sign * x / self.aspect
            dx = dx + self.gust_sign * self.push * gust_push(theta)
        return (x + dx) * self.h, (y + dy) * self.h, on

    def draw(self, t, canvas, mask):
        if not self.n:
            return
        x, y, on = self.at(t)
        xb, yb, _ = self.at(t, -self.SHUTTER / 2)
        xa, ya, _ = self.at(t, self.SHUTTER / 2)
        mx, my = xa - xb, ya - yb  # pixels moved while the shutter is open
        tau = self.tau(t)
        spark = self.glints * np.maximum(np.sin(2 * math.pi * self.q * tau + self.qph), 0.0) ** self.SHARP
        opacity = np.minimum(1.0, self.opacity * (1 + 1.5 * spark))
        turn = self.rot0 + 2 * math.pi * self.turns * tau + self.wobble * np.sin(2 * math.pi * self.mw * tau + self.phw)
        squash = 1 - self.tumble * (0.5 - 0.5 * np.cos(2 * math.pi * self.mt * tau + self.pht))
        reach = self.reach + 0.5 * np.hypot(mx, my)
        seen = on & (x > -reach) & (x < self.w + reach) & (y > -reach) & (y < self.h + reach) & (opacity > 0.002)
        for i in np.flatnonzero(seen):
            if self.halo[i] > 0:  # the glow first, the flake over it
                ix, iy = math.floor(x[i]), math.floor(y[i])
                cover, x0, y0 = self.halos.get(sprites.bucket(self.spread[i], 1.1), x[i] - ix, y[i] - iy)
                canvas.over(ix + x0, iy + y0, cover, self.colour[i], self.halo[i] * opacity[i], mask)
            if self.crystal[i]:
                self._crystal(i, x[i], y[i], turn[i], squash[i], mx[i], my[i], opacity[i], canvas, mask)
            else:
                self._dot(i, x[i], y[i], mx[i], my[i], opacity[i], canvas, mask)
            if spark[i] > 0.02 and self.radius[i] >= 0.6 * self.px:
                ln = sprites.bucket(min(self.tip[i] * (1.8 + 2.0 * self.glints[i]) + 3 * self.px, 40 * self.px))
                ix, iy = math.floor(x[i]), math.floor(y[i])
                cover, x0, y0 = self.sparks.get(ln, x[i] - ix, y[i] - iy)
                canvas.over(ix + x0, iy + y0, cover, self.colour[i], min(1.0, 1.2 * self.opacity[i]) * spark[i],
                            mask)

    def _dot(self, i, x, y, mx, my, opacity, canvas, mask):
        ix, iy = math.floor(x), math.floor(y)
        r, s, rim = self.radius[i], self.soft[i], self.rim[i]
        if math.hypot(mx, my) < 0.6:
            cover, x0, y0 = self.dots.get((sprites.bucket(r), sprites.bucket(s, floor=0.7), round(rim, 2)),
                                          x - ix, y - iy)
        else:
            cover, x0, y0 = flake(r, s, rim, mx, my, x - ix, y - iy)
        canvas.over(ix + x0, iy + y0, cover, self.colour[i], opacity, mask)

    def _crystal(self, i, x, y, turn, squash, mx, my, opacity, canvas, mask):
        tip = self.tip[i]
        size = next((s for s in CRYSTAL_SIZES if 0.45 * s >= tip), CRYSTAL_SIZES[-1])
        scale = tip / (0.45 * size)
        half = 0.5 * size * scale + 1.5
        x0, y0 = math.floor(x - half), math.floor(y - half)
        w, h = math.ceil(x + half) - x0, math.ceil(y + half) - y0
        ca, sa = math.cos(turn), math.sin(turn)
        sx, sy = scale, scale * max(squash, 0.2)
        ox, oy = x0 - x, y0 - y
        data = (ca / sx, sa / sx, (ca * ox + sa * oy) / sx + size / 2,
                -sa / sy, ca / sy, (-sa * ox + ca * oy) / sy + size / 2)
        img = crystal(int(self.shape[i]), size).transform((w, h), Image.Transform.AFFINE, data,
                                                          resample=Image.Resampling.BILINEAR)
        cover = np.asarray(img, np.float32) * (1 / 255.0)
        mx, my = mx * self.CRISP, my * self.CRISP
        if math.hypot(mx, my) >= 0.7:
            cover, dx0, dy0 = smear(cover, mx, my)
            x0, y0 = x0 + dx0, y0 + dy0
        canvas.over(x0, y0, cover, self.colour[i], opacity, mask)
