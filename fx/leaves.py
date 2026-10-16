"""Falling leaves: maple leaves in red, yellow and green, big and small; ordinary leaves; cherry petals (see
effects.py for how every effect loops).

The leaves are drawn by code, once per design, as premultiplied RGBA pictures (front and back) at a few sizes. A
maple leaf is the union of its lobes: five main lobes with pointed tips and two small ones by the stem, each a
tapering blade along a slightly bent axis with fine teeth along its sides (some with a pointed side lobe), joined to
the palm with rounded sinuses (a smooth minimum of their distance fields). Veins run from the stem to every lobe
tip, with thin side veins out to the teeth. The colour varies across the leaf (crimson edges and orange near the
veins of a red leaf, golden yellow with green-tinged parts and a few brown freckles, green with yellow-green veins),
with a faint texture and shading from a leaf that is cupped, curled and a little sunk along its veins. The back is
duller and lighter, its veins standing out (a leaf turned over shows it mirrored).

Every leaf falls like a real one: most glide from side to side like a pendulum, falling fastest in the middle of a
swing and slowing (even lifting a little) at its ends, tilting into each swing; some tumble end over end, showing
their back half the time; some spiral down. Big leaves swing slower and wider. The wind leans their path, and gusts
come and go (never against the wind), pushing near leaves more than far ones. Far leaves are smaller, slower and a
little hazy; near ones bigger and a little out of focus; fast ones smear as a camera's shutter would.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .effects import Falling, _choice, _lerp, _num
from .snow import GUST_LEAD, gust_push

# --- the leaves' pictures -------------------------------------------------------------------------------------

MIPS = (288, 144, 72, 36, 18)  # px across a leaf's pictures: it is drawn at one of the first three (as large as
# the leaves will be seen), the smaller ones made from that
FAMILIES = ("red", "yellow", "green")
VARIANTS = {"maple": 10, "oval": 6, "petal": 8}  # designs of each kind of leaf, per colour
SS = 2  # supersampling of the veins and stems


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _smin(a, b, k):
    """A smooth minimum of two distance fields: their union with rounded inside corners (k: how round)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1 - h)


def _noise(rng, shape, cells):
    """Smooth value noise, -1..1 (about), `cells` blobs across the longer side."""
    h, w = shape
    n = max(2, int(cells))
    small = rng.random((max(2, round(n * h / max(h, w))) + 3, max(2, round(n * w / max(h, w))) + 3))
    img = Image.fromarray(small.astype(np.float32), "F")
    big = img.resize((w, h), Image.Resampling.BICUBIC, box=(1.5, 1.5, small.shape[1] - 1.5, small.shape[0] - 1.5))
    return (np.asarray(big, np.float32) - 0.5) * 2


class Lobe:
    """A tapering blade along an axis from (ox, oy) at `angle` (radians from straight up, positive to the right),
    `length` long and `width` at its widest (half-widths), the axis bent sideways by `bend` (at the tip, in lengths),
    with teeth along its sides (amp, period: leaf units)."""

    def __init__(self, ox, oy, angle, length, width, bend=0.0, amp=0.0, period=0.05, double=0.0, phase=(0.0, 0.0),
                 widest=0.35, tip=1.0, base=0.3, teeth_from=0.22, kind="lobe"):
        self.ox, self.oy, self.a, self.L, self.W = ox, oy, angle, length, width
        self.bend, self.amp, self.period, self.double, self.phase = bend, amp, period, double, phase
        self.tip, self.base, self.teeth_from, self.kind = tip, base, teeth_from, kind
        # the profile (1 - t)^tip (base + t)^b, widest at t = widest
        self.b = max(0.05, tip * (widest + base) / max(1e-6, 1 - widest))
        self.norm = (1 - widest) ** tip * (base + widest) ** self.b

    def axis(self, s):
        """Points (x, y) of the bent axis at distances s along it."""
        ca, sa = math.cos(self.a), math.sin(self.a)
        off = self.bend * self.L * (np.asarray(s) / self.L) ** 2
        return self.ox + s * sa + off * ca, self.oy + s * ca - off * sa

    def profile(self, t):
        """Half-width (in widths) and its slope (per length) at t = 0..1 along the lobe."""
        t = np.clip(t, 0.0, 1.0)
        one = np.maximum(1 - t, 1e-6)
        f = one ** self.tip * (self.base + t) ** self.b / self.norm
        slope = f * (self.b / (self.base + t) - self.tip / one)
        return f, slope

    def teeth(self, s, side):
        """How far the teeth stand out (leaf units) at s along the lobe, and their slope; each tooth leans to the
        tip: the margin goes out slowly, then back in steeply."""
        if self.amp <= 0:
            return 0.0, 0.0
        t = s / self.L
        env = _smoothstep(self.teeth_from, self.teeth_from + 0.18, t) * (1 - _smoothstep(0.9, 1.0, t))
        r = 0.78
        out, slope = 0.0, 0.0
        for share, freq, extra in ((1 - self.double, 1.0, 0.0), (self.double, 2.0, 0.37)):
            if share <= 0:
                continue
            u = np.mod(s * freq / self.period + np.where(side, self.phase[0], self.phase[1]) + extra, 1.0)
            up = u < r
            out = out + share * np.where(up, u / r, (1 - u) / (1 - r))
            slope = slope + share * np.where(up, 1 / r, -1 / (1 - r)) * freq / self.period
        return self.amp * env * (out - 0.45), self.amp * env * slope

    def field(self, x, y):
        """A distance-like field (negative inside, leaf units) at points x, y, and the position along the lobe
        (0 at its base, 1 at its tip)."""
        ca, sa = math.cos(self.a), math.sin(self.a)
        dx, dy = x - self.ox, y - self.oy
        s = dx * sa + dy * ca
        q = dx * ca - dy * sa
        q = q - self.bend * self.L * np.clip(s / self.L, 0, None) ** 2  # across from the bent axis
        t = s / self.L
        f, fs = self.profile(t)
        hw = self.W * f
        slope = self.W * fs / self.L
        d = (np.abs(q) - hw) / np.sqrt(1 + slope * slope)
        if self.amp > 0:  # the teeth move the margin only: they fade out within a few of their sizes of it
            at = np.abs(d) < 7 * self.amp
            ds = d[at]
            tw, ts = self.teeth(s[at], q[at] > 0)
            near = np.exp(-(ds / (2.5 * self.amp)) ** 2)
            d[at] = (ds - tw * near) / np.sqrt(1 + (ts * near) ** 2)
        return np.maximum(np.maximum(d, s - self.L), -s), t

    def bounds(self):
        pts = [self.axis(s) for s in np.linspace(0, self.L, 9)]
        reach = self.W * 1.1 + self.amp
        xs = [float(p[0]) for p in pts]
        ys = [float(p[1]) for p in pts]
        return min(xs) - reach, min(ys) - reach, max(xs) + reach, max(ys) + reach


# --- leaf designs: the shapes ---------------------------------------------------------------------------------

def maple_design(rng):
    """A maple leaf's lobes, palm, stem and how the leaf is cupped and curled. Three kinds blend: broad lobes with
    pointed side lobes (sugar maple), shallower lobes with doubly toothed edges (red maple), deeply cut narrow lobes
    with fine teeth (Japanese maple)."""
    kind = rng.choice(["broad", "toothed", "cut"], p=[0.38, 0.27, 0.35])
    lobes, k = [], 0.07
    sym = lambda v, j=0.06: v * rng.uniform(1 - j, 1 + j)  # noqa: E731 (a little asymmetry)
    if kind == "cut":  # Japanese maple: five long pointed lobes and two small ones by the stem, cut deep
        palm, k = rng.uniform(0.27, 0.33), rng.uniform(0.05, 0.065)
        spread = rng.uniform(0.62, 0.7)  # radians between lobes
        lengths = (1.0, rng.uniform(0.9, 0.98), rng.uniform(0.68, 0.8), rng.uniform(0.34, 0.46))
        width = rng.uniform(0.18, 0.22)
        amp, period = rng.uniform(0.01, 0.016), rng.uniform(0.038, 0.05)
        double = rng.uniform(0.3, 0.5)
        for i, ln in enumerate(lengths):
            for side in ((0,) if i == 0 else (-1, 1)):
                a = side * (i * spread * (1.05 if i == 3 else 1.0) + rng.normal(0, 0.03)) + (
                    rng.normal(0, 0.03) if i == 0 else 0)
                lobes.append(Lobe(0, 0, a, sym(ln), sym(width * (0.7 if i == 3 else 1.0)),
                                  bend=side * rng.uniform(0.0, 0.07) if i else rng.normal(0, 0.02),
                                  amp=amp, period=period, double=double, phase=tuple(rng.random(2)),
                                  widest=rng.uniform(0.42, 0.52), tip=rng.uniform(1.15, 1.4), base=0.2,
                                  teeth_from=0.3))
        stem = rng.uniform(0.35, 0.5)
    else:
        broad = kind == "broad"
        palm = rng.uniform(0.3, 0.36) if broad else rng.uniform(0.34, 0.42)
        k = rng.uniform(0.07, 0.1) if broad else rng.uniform(0.05, 0.07)
        up = rng.uniform(0.85, 1.0) if broad else rng.uniform(0.72, 0.85)  # angle of the upper side lobes
        low = up + rng.uniform(0.8, 0.95)
        lengths = (1.0, rng.uniform(0.9, 1.0), rng.uniform(0.52, 0.68), rng.uniform(0.14, 0.24))
        widths = (rng.uniform(0.25, 0.3), rng.uniform(0.23, 0.28), rng.uniform(0.17, 0.21), 0.09) if broad else \
            (rng.uniform(0.3, 0.36), rng.uniform(0.27, 0.32), rng.uniform(0.18, 0.22), 0.1)
        if broad:
            amp, period, double = rng.uniform(0.0, 0.008), rng.uniform(0.07, 0.1), 0.0
        else:
            amp, period, double = rng.uniform(0.018, 0.028), rng.uniform(0.055, 0.075), rng.uniform(0.35, 0.55)
        basal = rng.random() < 0.8
        angles = (0.0, up, low, low + rng.uniform(0.5, 0.65))
        for i, (ln, w, a0) in enumerate(zip(lengths, widths, angles)):
            if i == 3 and not basal:
                continue
            for side in ((0,) if i == 0 else (-1, 1)):
                a = side * (a0 + rng.normal(0, 0.035)) + (rng.normal(0, 0.03) if i == 0 else 0)
                L, W = sym(ln), sym(w)
                lobe = Lobe(0, 0, a, L, W, bend=(side * rng.uniform(-0.02, 0.06) if i else rng.normal(0, 0.02)),
                            amp=amp, period=period, double=double, phase=tuple(rng.random(2)),
                            widest=rng.uniform(0.3, 0.42), tip=rng.uniform(0.9, 1.2), base=0.35,
                            teeth_from=0.35 if broad else 0.25)
                lobes.append(lobe)
                # pointed side lobes on the main lobes (big ones on a sugar maple's, small on a red maple's)
                if i < 3:
                    pairs = (1 if i == 2 else rng.integers(1, 3)) if broad else int(i < 2 and rng.random() < 0.6)
                    for j in range(int(pairs)):
                        t0 = (rng.uniform(0.5, 0.62) if j == 0 else rng.uniform(0.3, 0.4)) - (0.06 if i == 2 else 0)
                        for s2 in (-1, 1):
                            if i > 0 and s2 == -side and rng.random() < 0.5:  # the inner side's are often missing
                                continue
                            bx, by = lobe.axis(t0 * L)
                            size = (0.3 if broad else 0.16) * L * (1 - 0.35 * j) * rng.uniform(0.8, 1.15)
                            dev = rng.uniform(0.75, 1.0)
                            lobes.append(Lobe(float(bx), float(by), lobe.a + s2 * dev, size + W * 0.6,
                                              size * rng.uniform(0.32, 0.42), amp=amp * 0.6, period=period,
                                              phase=tuple(rng.random(2)), widest=0.45, tip=1.1, base=0.5,
                                              teeth_from=0.5, kind="side"))
        stem = rng.uniform(0.4, 0.62)
    return {"kind": "maple", "lobes": lobes, "palm": palm, "k": k, "stem": stem,
            "stem_bend": rng.normal(0, 0.12), "stem_w": rng.uniform(0.022, 0.03),
            "cup": rng.uniform(-0.25, 0.45), "curl": rng.uniform(-0.3, 0.4), "fold": rng.uniform(0.02, 0.05),
            "wave": rng.uniform(0.01, 0.04), "pinnate": False}  # (fold: how deep the blade sinks along the veins)


def oval_design(rng):
    """An ordinary leaf: an oval blade with a pointed tip, a midrib with curved side veins, fine teeth or none."""
    shape = rng.choice(["ovate", "elliptic", "lance"], p=[0.4, 0.4, 0.2])
    width = {"ovate": rng.uniform(0.3, 0.36), "elliptic": rng.uniform(0.26, 0.32), "lance": rng.uniform(0.16, 0.2)}
    widest = {"ovate": 0.35, "elliptic": 0.45, "lance": 0.4}[shape]
    toothed = rng.random() < 0.6
    lobe = Lobe(0, 0, rng.normal(0, 0.03), 1.0, width[shape], bend=rng.normal(0, 0.05),
                amp=rng.uniform(0.008, 0.014) if toothed else 0.0, period=rng.uniform(0.035, 0.05),
                double=rng.uniform(0.0, 0.3), phase=tuple(rng.random(2)), widest=widest + rng.normal(0, 0.03),
                tip=rng.uniform(1.5, 1.9), base=0.06, teeth_from=0.12)
    return {"kind": "oval", "lobes": [lobe], "palm": 0.0, "k": 0.0, "stem": rng.uniform(0.12, 0.2),
            "stem_bend": rng.normal(0, 0.15), "stem_w": rng.uniform(0.018, 0.024),
            "cup": rng.uniform(-0.2, 0.5), "curl": rng.uniform(-0.3, 0.4), "fold": rng.uniform(0.03, 0.07),
            "wave": rng.uniform(0.01, 0.03), "pinnate": True}


def petal_design(rng):
    """A cherry petal: a rounded blade, narrow at its base, with a small notch at its tip."""
    return {"kind": "petal", "w": rng.uniform(0.34, 0.42), "notch": rng.uniform(0.06, 0.12),
            "tilt": rng.normal(0, 0.06), "cup": rng.uniform(0.2, 0.6), "curl": rng.uniform(-0.2, 0.3),
            "fold": 0.0, "wave": rng.uniform(0.01, 0.03)}


DESIGNS = {"maple": maple_design, "oval": oval_design, "petal": petal_design}


# --- leaf designs: drawn ----------------------------------------------------------------------------------------

def _petal_field(design, x, y):
    """A cherry petal (base at the origin, tip at y = 1): a distance-like field, and the position along it."""
    w, notch, tilt = design["w"], design["notch"], design["tilt"]
    x = x * math.cos(tilt) - y * math.sin(tilt)
    t = np.clip(y, 0.0, 1.0)
    # narrow at the base (the claw), widening evenly to about two thirds up, rounded at the tip
    top = 0.68
    hw = w * np.where(t < top, 0.1 + 0.9 * np.sin(0.5 * math.pi * t / top) ** 0.9,
                      np.sqrt(np.clip(1 - ((t - top) / (1 - top)) ** 2, 0, 1)))
    d = (np.abs(x) - hw) * 0.8
    d = np.maximum(d, np.maximum(-y, y - 1.0))
    # the notch: a small V cut into the middle of the tip
    v = (notch - (1.0 - y)) - np.abs(x) * 1.6
    return np.maximum(d, v), t


SOFT = 0.025  # how softly the lobes' positions along them blend where lobes meet (leaf units)


def _blade(design, x, y, k_side=0.4):
    """The blade's distance-like field (negative inside, leaf units) at x, y, and how far along its lobe a point is
    (0 at the palm .. 1 at a tip; blended smoothly where lobes meet)."""
    if design["kind"] == "petal":
        return _petal_field(design, x, y)
    palm, k = design["palm"], design["k"]
    if palm > 0:  # the palm, cut straight across at the bottom: the base of the leaf is the lobes by the stem
        d = np.maximum(np.hypot(x, y - 0.4 * palm) - palm, -(y + 0.25 * palm))
    else:
        d = np.full(x.shape, 10.0, np.float32)
    weight = np.exp(-np.clip(d, -0.5, 1.0) / SOFT)  # (the palm: along 0)
    along = np.zeros(x.shape)
    xs, ys = x[0, :], y[:, 0]
    for lobe in design["lobes"]:
        x0, y0, x1, y1 = lobe.bounds()
        grow = k + 0.2
        cols = np.flatnonzero((xs >= x0 - grow) & (xs <= x1 + grow))
        rows = np.flatnonzero((ys >= y0 - grow) & (ys <= y1 + grow))
        if not len(cols) or not len(rows):
            continue
        sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        dl, t = lobe.field(x[sl], y[sl])
        d[sl] = _smin(d[sl], dl, k * (k_side if lobe.kind == "side" else 1.0))
        w = np.exp(-np.clip(dl, -0.5, 1.0) / SOFT)
        weight[sl] += w
        along[sl] += w * np.clip(t, 0, 1)
    return d, along / weight


def _veins(design):
    """The veins as polylines of (x, y, width) points, leaf units: the main veins from the stem to the lobe tips
    (and the side lobes' tips), and the thin side veins from them out towards the margin."""
    main, side_veins = [], []
    rng = np.random.default_rng(design["seed"])
    if design["kind"] == "petal":
        for a in np.linspace(-0.55, 0.55, 9):
            a = a + rng.normal(0, 0.04)
            side_veins.append([(float(a * s * design["w"] * 1.25 * (1.2 - 0.2 * s)), float(s), 0.0045 * (1 - 0.5 * s))
                               for s in np.linspace(0.04, 0.82, 8)])
        return main, side_veins
    pinnate = design["pinnate"]
    for lobe in design["lobes"]:
        side = lobe.kind == "side"
        n = 10
        ss = np.linspace(0, lobe.L * (0.88 if side else 0.96), n)
        xs, ys = lobe.axis(ss)
        w0 = 0.008 if side else (0.02 if pinnate else 0.015) * min(1.0, lobe.L ** 0.5)
        main.append([(float(xs[j]), float(ys[j]), w0 * (1 - 0.7 * j / (n - 1))) for j in range(n)])
        if side:
            continue
        # side veins every `step` along the lobe (not in the palm), leaning to its tip, curving out to the margin
        step = 0.085 if pinnate else max(0.065, 0.13 * lobe.L)
        lean = 0.8 if pinnate else 0.95
        s0 = step * 0.9 if pinnate else max(step, 0.9 * design["palm"])
        while s0 < lobe.L * 0.86:
            f, _ = lobe.profile(s0 / lobe.L + 0.1)
            reach = lobe.W * f * rng.uniform(0.8, 0.95)
            for sgn in (-1, 1):
                at = s0 + rng.normal(0, 0.12) * step
                bx, by = lobe.axis(at)
                ang = lobe.a + sgn * (lean + rng.normal(0, 0.08))
                curve = (0.45 if pinnate else 0.25) * rng.uniform(0.7, 1.2)
                pts = []
                for j in range(7):
                    r = reach * j / 6 / math.sin(lean)
                    a = ang - sgn * curve * (j / 6) ** 2  # curving towards the tip
                    pts.append((float(bx + r * math.sin(a)), float(by + r * math.cos(a)), 0.006 * (1 - 0.6 * j / 6)))
                side_veins.append(pts)
            s0 += step * rng.uniform(0.85, 1.15)
    return main, side_veins


def _poly_stroke(draw, pts, scale, ox, oy, fill):
    """A stroke along `pts` (x, y, width in leaf units) on a Pillow image (y up in leaf units, down in the image)."""
    left, right = [], []
    for j, (x, y, w) in enumerate(pts):
        xa, ya = pts[max(j - 1, 0)][:2]
        xb, yb = pts[min(j + 1, len(pts) - 1)][:2]
        tx, ty = xb - xa, yb - ya
        ln = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / ln, tx / ln
        left.append(((x + nx * w / 2 - ox) * scale, (oy - (y + ny * w / 2)) * scale))
        right.append(((x - nx * w / 2 - ox) * scale, (oy - (y - ny * w / 2)) * scale))
    draw.polygon(left + right[::-1], fill=fill)
    for (x, y, w) in (pts[0], pts[-1]):
        r = w / 2 * scale
        cx, cy = (x - ox) * scale, (oy - y) * scale
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill)


def _stem_points(design):
    if design["kind"] == "petal":
        return []
    n, ln, bend, w = 8, design["stem"], design["stem_bend"], design["stem_w"]
    pts = []
    for j in range(n + 1):
        s = j / n
        pts.append((bend * ln * s * s, 0.02 - ln * s, w * (1.0 - 0.25 * s + 0.35 * max(0.0, s - 0.85) / 0.15)))
    return pts


def _bounds(design):
    if design["kind"] == "petal":
        return -design["w"] - 0.05, -0.05, design["w"] + 0.05, 1.05
    bx = [design["palm"], -design["palm"]]
    by = [design["palm"], -design["palm"]]
    for lobe in design["lobes"]:
        x0, y0, x1, y1 = lobe.bounds()
        bx += [x0, x1]
        by += [y0, y1]
    for x, y, w in _stem_points(design):
        bx += [x - w, x + w]
        by += [y - w, y + w]
    return min(bx), min(by), max(bx), max(by)


# --- leaf designs: coloured -------------------------------------------------------------------------------------

def _mix(a, b, f):
    """a (H, W, 3) towards colour b by f (H, W) or a number."""
    f = np.asarray(f, np.float32)
    if f.ndim:
        f = f[..., None]
    return a + (np.asarray(b, np.float32) - a) * f


def _blur(a, radius):
    """A Gaussian blur (standard deviation `radius` px) of a float array, edges repeated."""
    a = np.asarray(a, np.float32)
    if radius < 0.3:
        return a
    if radius > 4:  # a wide blur: worked out smaller, then made full size again
        f = int(radius / 2)
        h, w = a.shape
        small = Image.fromarray(a, "F").resize((math.ceil(w / f), math.ceil(h / f)), Image.Resampling.BOX)
        small = Image.fromarray(_blur(np.asarray(small, np.float32), radius / f), "F")
        return np.asarray(small.resize((w, h), Image.Resampling.BICUBIC, box=(0, 0, w / f, h / f)), np.float32)
    reach = int(math.ceil(3 * radius))
    k = np.exp(-0.5 * (np.arange(-reach, reach + 1) / radius) ** 2).astype(np.float32)
    k /= k.sum()
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (reach, reach)
        p = np.pad(a, pad, mode="edge")
        n = a.shape[axis]
        out = np.zeros_like(a)
        for i, w in enumerate(k):
            out += w * (p[i:i + n] if axis == 0 else p[:, i:i + n])
        a = out
    return a


def _ramp(stops, v):
    """A colour from `stops` [(position, (r, g, b)), ...] at v (0..1)."""
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if v <= p1:
            f = (v - p0) / (p1 - p0)
            return np.array(c0, np.float32) + (np.array(c1, np.float32) - np.array(c0, np.float32)) * f
    return np.array(stops[-1][1], np.float32)


RED = [(0.0, (0.48, 0.05, 0.07)), (0.35, (0.66, 0.06, 0.08)), (0.65, (0.80, 0.14, 0.07)), (1.0, (0.88, 0.32, 0.07))]
YELLOW = [(0.0, (0.93, 0.83, 0.26)), (0.5, (0.96, 0.73, 0.14)), (1.0, (0.92, 0.57, 0.09))]
GREEN = [(0.0, (0.17, 0.38, 0.09)), (0.5, (0.27, 0.49, 0.12)), (1.0, (0.42, 0.6, 0.15))]
PINK = [(0.0, (0.995, 0.93, 0.95)), (0.5, (0.99, 0.84, 0.89)), (1.0, (0.97, 0.72, 0.8))]


def _paint(design, family, rng, f):
    """Front and back colours (H, W, 3, 0..1) of a leaf from its fields `f`, and its stem's colour."""
    shape = f["alpha"].shape
    e, r, along, halo = f["inside"], f["r"], f["along"], f["halo"]
    main, side = f["main"], f["side"]
    tips = _smoothstep(0.45, 1.0, along)
    big, mid = f["big"], _noise(rng, shape, 11)
    grain = _blur(np.random.default_rng([design["seed"], 7]).normal(0.0, 1.0, shape).astype(np.float32),
                  0.6 * shape[0] / MIPS[0])  # (the same at any size of the picture)
    edge = np.exp(-e / 0.06)
    heart = np.maximum(halo, np.exp(-r / 0.3))  # near the main veins and the middle
    tone = rng.random()
    petal = design["kind"] == "petal"
    if petal:
        base = _ramp(PINK, tone)
        col = np.broadcast_to(base, shape + (3,)).astype(np.float32)
        col = _mix(col, (0.93, 0.5, 0.64), 0.55 * (1 - _smoothstep(0.0, 0.45, along)) ** 2)  # deep pink at the base
        col = _mix(col, (1.0, 0.97, 0.98), 0.5 * edge * _smoothstep(0.4, 1.0, along))  # paler at the edge
        veins = (0.92, 0.6, 0.72)
        stem = np.array((0.8, 0.55, 0.5), np.float32)
    elif family == "red":
        base = _ramp(RED, tone)
        col = np.broadcast_to(base, shape + (3,)).astype(np.float32)
        col = _mix(col, (0.92, 0.36, 0.08), np.clip(0.35 * big + 0.1 * mid, 0, 1) * rng.uniform(0.3, 0.8))  # orange
        col = _mix(col, base * 0.62 + np.array((0.12, 0.0, 0.02), np.float32),
                   np.clip(edge * rng.uniform(0.5, 0.9) + 0.35 * tips, 0, 1))  # crimson edges and tips
        col = _mix(col, (0.95, 0.6, 0.14), np.clip(heart * rng.uniform(0.35, 0.85) * (1 - edge), 0, 1))
        if rng.random() < 0.3:  # still turning: orange-yellow patches left in the middle
            col = _mix(col, (0.94, 0.55, 0.1), np.clip(0.9 * big + 0.25 * mid + 0.15, 0, 0.75) * (1 - tips) * (1 - edge))
        veins = (0.98, 0.72, 0.32)
        stem = np.array((0.62, 0.16, 0.08), np.float32)
    elif family == "yellow":
        base = _ramp(YELLOW, tone)
        col = np.broadcast_to(base, shape + (3,)).astype(np.float32)
        col = _mix(col, (0.62, 0.68, 0.17), np.clip(heart * rng.uniform(0.25, 0.7) + np.clip(big - 0.25, 0, 1)
                                                      * rng.uniform(0.2, 0.8), 0, 1))  # green-tinged parts
        col = _mix(col, (0.9, 0.48, 0.08), np.clip(edge * rng.uniform(0.0, 0.6) + 0.3 * tips * rng.random(), 0, 1))
        if rng.random() < 0.45:  # dry brown tips
            col = _mix(col, (0.5, 0.31, 0.11), np.clip(_smoothstep(0.8, 1.0, along) * edge * 1.5, 0, 0.85))
        elif rng.random() < 0.4:  # a blush of red at the tips
            col = _mix(col, (0.86, 0.3, 0.08), np.clip(tips * edge * 1.4 + 0.25 * tips, 0, 0.7))
        veins = (0.99, 0.93, 0.58)
        stem = np.array((0.72, 0.58, 0.2), np.float32)
    else:
        base = _ramp(GREEN, tone)
        col = np.broadcast_to(base, shape + (3,)).astype(np.float32)
        col = _mix(col, base * 1.25 + np.array((0.06, 0.05, 0.0), np.float32), np.clip(halo * 0.5, 0, 1))
        col = _mix(col, base * 0.8, np.clip(edge * 0.5, 0, 1))
        turning = rng.random()
        if turning < 0.3:  # early autumn: tips and edges turning yellow or red
            warm = (0.86, 0.66, 0.14) if turning < 0.2 else (0.72, 0.2, 0.07)
            col = _mix(col, warm, np.clip((0.7 * tips + 0.5 * edge) * rng.uniform(0.5, 1.0) + 0.3 * big, 0, 0.95))
        veins = (0.7, 0.8, 0.32)
        stem = np.array((0.48, 0.58, 0.22), np.float32)
    # texture: blotches and a fine grain
    col = col * (1 + (0.07 * mid + 0.03 * grain)[..., None] * (0.3 if petal else 1.0))
    if family == "yellow" and not petal:
        col = _freckles(col, f, rng, (0.42, 0.25, 0.09), int(rng.integers(3, 13)))
    elif family == "red" and not petal and rng.random() < 0.3:
        col = _freckles(col, f, rng, (0.3, 0.04, 0.05), int(rng.integers(2, 7)))
    front = _mix(col, veins, np.clip(0.2 * side if petal else 0.55 * main + 0.2 * side, 0, 1))
    # the back: duller and lighter, the veins standing out (a petal's: about the same as its front)
    grey = col.mean(axis=2, keepdims=True)
    if petal:
        back = _mix(col, (1.0, 0.95, 0.96), 0.12)
    else:
        back = np.clip((col + (grey - col) * 0.28) * 1.04 + 0.04, 0, 1)
    back = _mix(back, np.clip(np.array(veins) * 0.6 + 0.38, 0, 1),
                np.clip(0.5 * main + 0.25 * side + 0.1 * f["vein_wide"], 0, 1) * (0.4 if petal else 1.0))
    return np.clip(front, 0, 1), back, stem


def _freckles(col, f, rng, colour, n):
    """A few small spots, soft-edged (as many random draws at any size of the picture)."""
    h, w = f["alpha"].shape
    at = rng.random((64, 2))
    sizes = rng.uniform(0.006, 0.025, 64)
    strength = rng.uniform(0.5, 0.95, 64)
    rows, cols = (at[:, 0] * h).astype(int), (at[:, 1] * w).astype(int)
    for j in np.flatnonzero(f["inside"][rows, cols] > 0.05)[:n]:
        cy, cx = rows[j], cols[j]
        rad = max(sizes[j] * f["ppu"], 0.6)
        reach = int(math.ceil(3 * rad)) + 1
        y0, y1, x0, x1 = max(cy - reach, 0), min(cy + reach + 1, h), max(cx - reach, 0), min(cx + reach + 1, w)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / max(rad * rad, 1e-6)
        col[y0:y1, x0:x1] = _mix(col[y0:y1, x0:x1], colour, np.exp(-d2 * 1.5) * strength[j])
    return col


def _shade(design, f, back):
    """Light and shade (H, W) from how the leaf is cupped, curled and folded along its veins."""
    r, along = f["r"], f["along"]
    h = design["cup"] * 0.35 * r * r
    h = h + design["curl"] * 0.25 * _smoothstep(0.3, 1.0, along) ** 2
    h = h - design["fold"] * f["valley"]  # the blade a little sunk along the main veins
    h = h + design["wave"] * f["big"]
    if back:
        h = -h
    gy, gx = np.gradient(_blur(h, 0.012 * f["ppu"]), 1.0 / f["ppu"])
    gy = -gy  # rows run down, y up
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    light = np.array([-0.45, 0.55, 0.7])
    light /= np.linalg.norm(light)
    lam = n @ light
    shade = np.clip(1 + 0.75 * (lam - light[2]), 0.55, 1.3)
    if not back:  # a faint waxy sheen on the top
        half = light + np.array([0.0, 0.0, 1.0])
        half /= np.linalg.norm(half)
        shade = shade + 0.18 * np.clip(n @ half, 0, 1) ** 40
    return _blur(shade.astype(np.float32), 0.004 * f["ppu"])


class Sprite:
    """One leaf design in one colour: front and back as premultiplied RGBA pictures (Pillow "RGBa") at the MIPS
    sizes up to `size`, both the same way round (the leaf turning over mirrors the back); `extent`: the leaf's
    longest side and `reach`: how far it reaches from the picture's middle, in the largest picture (px); `rgb`: its
    mean colour."""

    def __init__(self, front, back, size, extent, reach, rgb):
        self.front, self.back, self.size = front, back, size
        self.extent, self.reach, self.rgb = extent, reach, rgb
        self.bytes = sum(img.size[0] * img.size[1] * 4 for img in front + back)

    def image(self, back, size):
        """The smallest of the ready pictures in which the leaf is still at least `size` px long."""
        mips = self.back if back else self.front
        for img in reversed(mips):
            if self.extent * img.size[0] / self.size >= size:
                return img
        return mips[0]


def _grid(n, cx, cy, ppu):
    xs = (cx + (np.arange(n) + 0.5 - n / 2) / ppu).astype(np.float32)
    ys = (cy - (np.arange(n) + 0.5 - n / 2) / ppu).astype(np.float32)
    return np.broadcast_to(xs[None, :], (n, n)), np.broadcast_to(ys[:, None], (n, n))


def make_sprite(kind, family, variant, size=MIPS[0]):
    """The pictures of leaf design `variant` of `kind` (maple, oval, petal) in colour `family`, the largest `size`
    px across (one of MIPS: the same leaf at any of them)."""
    rng = np.random.default_rng([("maple", "oval", "petal").index(kind), FAMILIES.index(family), variant, 4111])
    design = DESIGNS[kind](rng)
    design["seed"] = int(rng.integers(1 << 30))
    x0, y0, x1, y1 = _bounds(design)
    # where its weight is (the leaf turns about it): worked out roughly first
    n0 = 64
    ppu0 = (n0 - 4) / max(x1 - x0, y1 - y0)
    gx, gy = _grid(n0, (x0 + x1) / 2, (y0 + y1) / 2, ppu0)
    a0 = np.clip(0.5 - _blade(design, gx, gy)[0] * ppu0, 0, 1)
    total = float(a0.sum()) or 1.0
    cx, cy = float((a0 * gx).sum()) / total, float((a0 * gy).sum()) / total
    n = size
    half = max(cx - x0, x1 - cx, cy - y0, y1 - cy) + 0.02
    ppu = (n / 2 - 2) / half
    gx, gy = _grid(n, cx, cy, ppu)
    d, along = _blade(design, gx, gy)
    alpha = np.clip(0.5 - d * ppu, 0, 1).astype(np.float32)
    # the stem and the veins: drawn larger, then made smaller
    big = n * SS
    ox, oy = cx - n / 2 / ppu, cy + n / 2 / ppu  # the leaf point at the image's top-left corner
    pts = _stem_points(design)
    stem = np.zeros_like(alpha)
    if pts:
        img = Image.new("L", (big, big), 0)
        _poly_stroke(ImageDraw.Draw(img), pts, ppu * SS, ox, oy, 255)
        stem = np.asarray(img.resize((n, n), Image.Resampling.BOX), np.float32) / 255.0
    veins = []
    for lines in _veins(design):
        img = Image.new("L", (big, big), 0)
        draw = ImageDraw.Draw(img)
        for line in lines:
            # no stroke thinner than a pixel of the drawing: a fainter one instead, with as much ink
            thin = 1.0 / (ppu * SS)
            mean = sum(w for _, _, w in line) / len(line)
            _poly_stroke(draw, [(x, y, max(w, thin)) for x, y, w in line], ppu * SS, ox, oy,
                         int(255 * min(1.0, mean / thin)))
        veins.append(np.asarray(img.resize((n, n), Image.Resampling.BOX), np.float32) / 255.0 * alpha)
    main, side = veins
    inside = np.maximum(-d, 0.0).astype(np.float32)
    fields = {"alpha": alpha, "inside": inside, "r": np.hypot(gx, gy).astype(np.float32), "along": along,
              "valley": np.clip(_blur(main, 0.035 * ppu) * 2.5, 0, 1), "main": main, "side": side, "ppu": ppu,
              "halo": np.clip(_blur(main, 0.05 * ppu) * 3.0, 0, 1),
              "vein_wide": np.clip(_blur(np.maximum(main, side), 0.012 * ppu) * 1.5, 0, 1),
              "big": _noise(rng, alpha.shape, 3)}
    front, back, stem_rgb = _paint(design, family, rng, fields)
    front = front * _shade(design, fields, False)[..., None]
    back = back * _shade(design, fields, True)[..., None]
    rim = np.exp(-inside / 0.008)[..., None]  # the margin a shade darker
    front = front * (1 - 0.18 * rim)
    back = back * (1 - 0.08 * rim)
    # the stem, over the blade where they meet
    a = np.maximum(alpha, stem)
    over = (np.clip(stem - 0.6 * alpha, 0, 1) / np.maximum(a, 1e-6))[..., None]
    stem_col = stem_rgb * (0.85 + 0.15 * _noise(rng, alpha.shape, 6)[..., None])
    front = front * (1 - over) + stem_col * over
    back = back * (1 - over) + stem_col * 0.9 * over
    if design["kind"] == "petal":
        a = a * 0.93  # a petal lets a little light through
    ys, xs = np.nonzero(a > 0.02)
    extent = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    reach = float(np.hypot(xs + 0.5 - n / 2, ys + 0.5 - n / 2).max()) + 1.5
    rgb = (front * a[..., None]).reshape(-1, 3).sum(axis=0) / max(float(a.sum()), 1e-6)
    return Sprite(_mips(front, a), _mips(back, a), n, extent, reach, rgb)


def _mips(rgb, alpha):
    a = alpha[..., None]
    out = [Image.fromarray((np.dstack([np.clip(rgb, 0, 1) * a, a]) * 255 + 0.5).astype(np.uint8), "RGBa")]
    for size in MIPS:  # each from the one before: half its size
        if size < out[-1].size[0]:
            out.append(out[-1].resize((size, size), Image.Resampling.LANCZOS))
    return out


_SPRITES = {}
MAX_BYTES = 80 << 20  # the pictures kept take at most this much memory (every design at the largest size: ~50 MB)


def sprite(kind, family, variant, size=MIPS[0]):
    """make_sprite(), kept (petals are pink whatever the colour)."""
    key = (kind, "pink" if kind == "petal" else family, variant, size)
    item = _SPRITES.get(key)
    if item is None:
        item = make_sprite(kind, FAMILIES[0] if kind == "petal" else family, variant, size)
        while _SPRITES and sum(s.bytes for s in _SPRITES.values()) + item.bytes > MAX_BYTES:
            _SPRITES.pop(next(iter(_SPRITES)))  # (the oldest first)
        _SPRITES[key] = item
    return item


# --- the settings -----------------------------------------------------------------------------------------------

SIZE_PX = (10.0, 240.0)  # a leaf's longest side (px of a 1080-line frame) at 0 and 100 on the size sliders


def size_px(v):
    """A leaf's longest side (px of a 1080-line frame) at v (0..100) on a size slider (a log scale)."""
    return SIZE_PX[0] * (SIZE_PX[1] / SIZE_PX[0]) ** (v / 100)


def size_slider(px):
    """Where a leaf `px` long sits on a size slider."""
    return 100 * math.log(max(px, 1e-6) / SIZE_PX[0]) / math.log(SIZE_PX[1] / SIZE_PX[0])


LOOKS = {  # the "Kiểu lá rơi" presets: every setting but the kind of leaves and their colours (medium: the defaults)
    "few": {"amount": 12, "big": 72, "small": 35, "speed": 30, "sway": 60, "spin": 45, "wind": 8, "gust": 15,
            "depth": 55, "blur": 20, "haze": 20, "motion": 30, "opacity": 100},
    "many": {"amount": 75, "big": 70, "small": 20, "speed": 40, "sway": 50, "spin": 55, "wind": 20, "gust": 35,
             "depth": 70, "blur": 30, "haze": 35, "motion": 40, "opacity": 100},
    "windy": {"amount": 60, "big": 66, "small": 20, "speed": 55, "sway": 40, "spin": 80, "wind": 60, "gust": 75,
              "depth": 65, "blur": 35, "haze": 30, "motion": 65, "opacity": 100},
    "dreamy": {"amount": 22, "big": 88, "small": 42, "speed": 18, "sway": 70, "spin": 35, "wind": 5, "gust": 10,
               "depth": 75, "blur": 60, "haze": 25, "motion": 40, "opacity": 100},
}
SEASONS = {  # the "Màu theo mùa" presets: how many leaves are red, yellow and green
    "red": {"red": 80, "yellow": 15, "green": 5},
    "gold": {"red": 8, "yellow": 85, "green": 7},
    "redgold": {"red": 50, "yellow": 50, "green": 0},
    "mixed": {"red": 40, "yellow": 35, "green": 25},
    "early": {"red": 12, "yellow": 23, "green": 65},
    "summer": {"red": 0, "yellow": 0, "green": 100},
}

SCHEMA = {"label": "Lá rơi", "params": [
    _choice("look", "Kiểu lá rơi", "medium",
            [("few", "Lác đác vài lá"), ("medium", "Lá rơi vừa"), ("many", "Lá rơi dày"),
             ("windy", "Gió thu cuốn lá"), ("dreamy", "Lá to bay chậm")], presets=LOOKS, group="Kiểu"),
    _choice("shape", "Loại lá", "maple",
            [("maple", "Lá phong"), ("mixed", "Lá phong lẫn lá thường"), ("oval", "Lá thường (hình bầu dục)"),
             ("petal", "Cánh hoa anh đào")], group="Kiểu"),
    _choice("season", "Màu theo mùa", "mixed",
            [("red", "Thu đỏ rực"), ("gold", "Thu vàng"), ("redgold", "Thu đỏ vàng"), ("mixed", "Đỏ vàng xanh"),
             ("early", "Đầu thu (nhiều lá xanh)"), ("summer", "Lá xanh (mùa hè)")], presets=SEASONS,
            group="Màu lá (cánh hoa luôn màu hồng)"),
    _num("red", "Lá đỏ (nhiều hay ít)", 40, group="Màu lá (cánh hoa luôn màu hồng)"),
    _num("yellow", "Lá vàng (nhiều hay ít)", 35, group="Màu lá (cánh hoa luôn màu hồng)"),
    _num("green", "Lá xanh (nhiều hay ít)", 25, group="Màu lá (cánh hoa luôn màu hồng)"),
    _num("big", "Lá to nhất", 78, group="Cỡ lá"), _num("small", "Lá nhỏ nhất", 25, group="Cỡ lá"),
    _num("amount", "Số lá", 35, group="Số lá"), _num("opacity", "Độ đậm", 100, group="Số lá"),
    _num("speed", "Tốc độ rơi", 35, group="Chuyển động"),
    _num("sway", "Chao qua lại khi rơi", 55, group="Chuyển động"),
    _num("spin", "Xoay, lật mặt, lộn vòng", 50, group="Chuyển động"),
    _num("wind", "Gió (trái ← → phải)", 15, -100, 100, group="Gió"),
    _num("gust", "Gió giật từng cơn", 25, group="Gió"),
    _num("depth", "Chiều sâu (xa nhỏ, gần to)", 60, group="Chiều sâu"),
    _num("blur", "Nhoè lá ở gần", 20, group="Chiều sâu"),
    _num("haze", "Lá ở xa mờ nhạt", 30, group="Chiều sâu"),
    _num("motion", "Nhoè khi lá bay nhanh", 30, group="Chiều sâu")]}
LOOKS["medium"] = {p["key"]: p["value"] for p in SCHEMA["params"] if p["key"] in LOOKS["few"]}

OLD_STYLES = {"autumn": ("mixed", "redgold"), "green": ("oval", "summer"), "petal": ("petal", "mixed")}
OLD_FULL = 60  # leaves on screen at a time at the top of the first leaves' "Số lá"


def upgrade(raw):
    """The settings a clip of the first leaves effect saved (style, size; no kind of leaf), as this one's."""
    if not isinstance(raw, dict) or "shape" in raw or not ({"style", "size"} & set(raw)):
        return raw
    out = dict(raw)
    shape, season = OLD_STYLES.get(raw.get("style"), OLD_STYLES["autumn"])
    out.update(shape=shape, season=season, **SEASONS[season])
    try:
        size = min(max(float(raw.get("size", 45)), 0.0), 100.0)
    except (TypeError, ValueError, OverflowError):
        size = 45.0
    length = _lerp(22, 80, size / 100) if size == size else 51.0  # its leaves were about this long, 0.6 to 1.8 times
    out["big"], out["small"] = round(size_slider(1.7 * length)), round(size_slider(0.65 * length))
    try:
        out["amount"] = round(min(max(float(raw.get("amount", 30)), 0.0), 100.0) * OLD_FULL / Leaves.FULL)
    except (TypeError, ValueError, OverflowError):
        pass
    return out


# --- the effect -------------------------------------------------------------------------------------------------

GLIDE, TUMBLE, SPIRAL = 0, 1, 2  # how a leaf falls
PETAL = 0.6  # a cherry petal's size, as a share of a leaf's at the same place on the size sliders


def _rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class Leaves(Falling):
    CAP = 700  # leaves with fixed random draws; how many are used follows the "Số lá" slider
    VERSION = 2  # (1: the first leaves, simple shapes)
    FULL = 90  # leaves on screen at a time at the top of the "Số lá" slider (16:9)
    SHUTTER = 0.5  # the share of a frame the shutter is open at the top of the motion blur slider
    upgrade = staticmethod(upgrade)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        u = self.rng.random((self.CAP, 40))
        self.gph = float(self.rng.random()) * 2 * math.pi  # when the gusts come
        ds, spin, sway = p["depth"] / 100, p["spin"] / 100, p["sway"] / 100

        # depth and size: far leaves small, near ones big (the depth slider: how much), each its own size too
        z = u[:, 0] ** 1.25  # 0 far .. 1 near (more far ones: the space widens away from the camera)
        lo, hi = sorted((size_px(p["small"]), size_px(p["big"])))
        own = u[:, 2]
        where = np.clip(1.15 * (ds * (0.8 * z + 0.2 * own) + (1 - ds) * u[:, 1]) - 0.05, 0.0, 1.0)
        size = lo * (hi / lo) ** where  # px of a 1080-line frame: the leaf's longest side
        if p["shape"] == "petal":
            size = size * PETAL  # (a petal is much smaller than a leaf)
        bigness = (1 - ds) * u[:, 1] + ds * own  # 0..1: how big a leaf is for its depth (swings slower, wider)
        seen = size / 1080  # in frame heights

        # how it falls: a pendulum glide (most), tumbling end over end, or a spiral; whole cycles per loop
        mode = np.where(u[:, 3] < 0.3 * spin, TUMBLE, np.where(u[:, 3] < 0.3 * spin + 0.16, SPIRAL, GLIDE))
        glide, tumble, spiral = mode == GLIDE, mode == TUMBLE, mode == SPIRAL
        reach = sway * seen * _lerp(1.5, 4.0, u[:, 4]) * (0.75 + 0.5 * bigness)
        self.A = np.where(glide, reach, np.where(tumble, 0.35 * reach, 0.0))  # side to side
        self.As = np.where(spiral, 0.6 * reach, 0.0)  # round and round
        self.B = 0.6 * self.A  # it slows down (or rises a little) at the ends of a swing: a pendulum's arc
        self.m = self.cycles(_lerp(0.35, 0.8, u[:, 5]) / np.sqrt(0.75 + 0.5 * bigness))
        self.phi0 = u[:, 6] * 2 * math.pi
        self.R = np.where(glide, _lerp(0.3, 0.7, u[:, 7]) * (0.3 + 0.7 * sway), 0.0) * (0.5 + spin)  # tilt in a swing
        self.Rs = np.where(spiral, _lerp(0.15, 0.35, u[:, 7]), 0.0)
        self.j = np.where(spiral, np.where(u[:, 8] < 0.5, -1.0, 1.0), 0.0)  # a spiral turns it round once a circle
        self.turns = np.sign(u[:, 8] - 0.5) * np.rint(spin * np.where(tumble, _lerp(0.03, 0.2, u[:, 9]),
                                                                        _lerp(0.0, 0.07, u[:, 9])) * self.loop_s)
        self.turns = np.where(spiral, 0.0, self.turns)  # (whole turns a loop; gliding ones turn slowly, if at all)
        self.flips = np.where(tumble, np.sign(u[:, 10] - 0.5) * self.cycles(_lerp(0.35, 1.0, u[:, 11]) * (0.5 + spin)),
                              0.0)
        back = np.where(u[:, 12] < 0.25, math.pi, 0.0)  # (a quarter of them fall showing their back)
        self.beta0 = np.where(spiral, np.sign(u[:, 13] - 0.5) * _lerp(0.5, 1.0, u[:, 14]),
                              np.clip(np.tan((u[:, 13] - 0.5) * 2.2) * 0.45, -1.2, 1.2)) + back
        self.P_amp = np.where(glide, _lerp(0.2, 0.55, u[:, 14]) * sway, np.where(spiral, 0.25, 0.0))
        self.wm = self.cycles(_lerp(1.2, 2.6, u[:, 15]))  # a quick flutter on top
        self.wph = u[:, 16] * 2 * math.pi
        self.W = np.where(tumble, 0.0, 0.1 * spin * _lerp(0.4, 1.0, u[:, 17]))
        self.yaw0 = u[:, 18] * 2 * math.pi
        self.chi = np.clip(np.tan((u[:, 19] - 0.5) * 2.0) * 0.35, -1.2, 1.2)  # the axis it tilts about
        self.zs = np.where(spiral, 0.07, 0.0)  # nearer and further as it circles

        # the path: down, leaning with the wind (tumbling leaves glide off sideways), plus the gusts' steady drift
        speed_k = (1 - ds) + ds * (0.4 + 1.2 * z)
        speed = _lerp(0.04, 0.32, p["speed"] / 100) * speed_k * (0.85 + 0.3 * u[:, 20]) * (1.15 - 0.3 * bigness)
        angle = (math.radians(55) * p["wind"] / 100 + np.radians(6) * (u[:, 21] - 0.5)
                 + np.where(tumble, 0.3 * spin * np.sign(u[:, 22] - 0.5), 0.0))
        self.gust_sign = 1.0 if p["wind"] >= 0 else -1.0
        push = p["gust"] / 100 * 0.3 * (0.3 + 0.7 * z)
        self.gm = max(1.0, round(self.loop_s / 8))  # gusts per loop
        drift = self.gust_sign * push * GUST_LEAD * 2 * math.pi * self.gm / self.loop_s
        vx, vy = speed * np.sin(angle) + drift, speed * np.cos(angle)
        angle, speed = np.arctan2(vx, vy), np.hypot(vx, vy)

        # out of focus near the camera, hazy far away; the margins they may reach beyond the frame
        blur = p["blur"] / 100 * np.clip((z - 0.72) / 0.28, 0, 1) ** 1.5 * (0.5 + 0.045 * size)  # px (1080 lines)
        haze = p["haze"] / 100 * 0.6 * (1 - z) ** 1.5 * (0.4 + 0.6 * ds)
        out = 0.85 * seen * (1 + self.zs) + 3 * blur / 1080 + 0.01
        margin_y = out + self.B + 0.3 * push + 0.01
        margin_x = out + self.A + self.As + 0.5 * push + 0.02
        share = self.paths(u[:, 23], u[:, 24], speed, angle, margin_y, margin_x)

        # how many: about `wanted` in the frame at a time (counted with the margins, where leaves also are)
        wider = (1 + 2 * float(margin_x.mean()) / self.aspect) * (1 + 2 * float(margin_y.mean()))
        wanted = p["amount"] / 100 * self.FULL * self.aspect / (16 / 9) * wider
        n = min(int(np.searchsorted(np.cumsum(share), wanted)) + 1 if wanted > 0 else 0, self.CAP)
        order = np.argsort(z[:n], kind="stable")  # drawn far to near
        for name in ("cos", "sin", "k", "len", "extra", "s0", "xe", "speed", "my"):
            setattr(self, name, getattr(self, name)[order])
        for name in ("A", "As", "B", "m", "phi0", "R", "Rs", "j", "turns", "flips", "beta0", "P_amp", "wm", "wph",
                     "W", "yaw0", "chi", "zs"):
            setattr(self, name, getattr(self, name)[order])
        u = u[order]
        self.n = n
        self.z, self.size, self.mode = z[order], size[order], mode[order]
        self.blur, self.haze = blur[order] * self.px, haze[order]
        self.push = push[order]

        # which leaf: its kind, colour and design; a little brighter or darker, warmer or cooler than its design
        shape = p["shape"]
        self.kind = np.where(u[:, 26] < 0.65, "maple", "oval") if shape == "mixed" else np.full(n, shape)
        shares = np.array([p["red"], p["yellow"], p["green"]], float)
        shares = shares / shares.sum() if shares.sum() > 0 else np.full(3, 1 / 3)
        self.family = np.searchsorted(np.cumsum(shares)[:2], u[:, 27], side="right")
        self.variant = np.array([int(u[i, 28] * VARIANTS[str(k)]) for i, k in enumerate(self.kind)], int)
        self.tint = (_lerp(0.9, 1.08, u[:, 29])[:, None] * (1 + 0.05 * (u[:, 30:33] - 0.5))).astype(np.float32)
        self.opacity = p["opacity"] / 100 * np.where(self.kind == "petal", 0.95, 1.0) * (1 - 0.35 * self.haze)
        self.sheen = 0.25 * u[:, 33]
        # the leaves' pictures: drawn as large as the largest leaf will be seen (small in a small preview)
        most = float(self.size.max()) * self.px * 1.1 if n else 0.0
        self.master = next((s for s in MIPS[2::-1] if 0.7 * s >= most), MIPS[0])

    def at(self, t, dt=0.0):
        """x, y (pixels) of every leaf's middle at frame t + dt, whether it is on its visible stretch at frame t, and
        its swing's phase. Along its path a leaf is followed from where it is at t (not started again at the top)."""
        s = np.mod(self.s0 + self.k * self.len * self.tau(t), self.len)
        on = s >= self.extra
        d = s + self.k * self.len * (dt / self.P) - self.extra
        x, y = self.xe + d * self.sin, -self.my + d * self.cos
        tau = self.tau(t + dt)
        phi = 2 * math.pi * self.m * tau + self.phi0
        sn = np.sin(phi)
        x = x + self.A * sn + self.As * np.cos(phi)
        y = y - self.B * sn * sn
        if self.push.any():  # a gust sweeps across from the side the wind comes from, lifting the leaves a little
            theta = 2 * math.pi * self.gm * tau + self.gph - 2 * math.pi * 0.35 * self.gust_sign * x / self.aspect
            x = x + self.gust_sign * self.push * gust_push(theta)
            y = y - 0.25 * self.push * (np.exp(4 * (np.cos(theta) - 1)) - 0.25)
        return x * self.h, y * self.h, on, tau, phi

    def pose(self, tau, phi):
        """How every leaf is turned: tilt in the picture (roll), its own turn (yaw), how far it is turned towards
        or away from the camera (pitch: past a right angle its back shows) and a scale."""
        wob = np.sin(2 * math.pi * self.wm * tau + self.wph)
        roll = -self.R * np.sin(phi) + self.Rs * np.cos(phi) + self.W * wob
        yaw = self.yaw0 + self.j * phi + 2 * math.pi * self.turns * tau
        pitch = self.beta0 + self.P_amp * np.sin(phi) + 2 * math.pi * self.flips * tau + 1.5 * self.W * wob
        return roll, yaw, pitch, 1 + self.zs * np.sin(phi)

    def draw(self, t, canvas, mask):
        if not self.n:
            return
        x, y, on, tau, phi = self.at(t)
        shutter = self.SHUTTER * self.p["motion"] / 100
        if shutter > 0:
            xb, yb = self.at(t, -shutter / 2)[:2]
            xa, ya = self.at(t, shutter / 2)[:2]
            mx, my = xa - xb, ya - yb
        else:
            mx = my = np.zeros(self.n)
        roll, yaw, pitch, scale = self.pose(tau, phi)
        length = self.size * self.px * scale
        reach = 0.85 * length + 3 * self.blur + np.hypot(mx, my) + 2
        seen = on & (x > -reach) & (x < self.w + reach) & (y > -reach) & (y < self.h + reach)
        # the picture's turn and squash (seen at a slant): R(roll + chi) diag(1, squash) R(yaw - chi); turned over
        # (squash < 0), the leaf is seen mirrored
        c = np.cos(pitch)
        squash = np.copysign(np.maximum(np.abs(c), 0.05), c)
        ca, sa = np.cos(roll + self.chi), np.sin(roll + self.chi)
        cb, sb = np.cos(yaw - self.chi), np.sin(yaw - self.chi)
        m = np.stack([ca * cb - sa * squash * sb, -ca * sb - sa * squash * cb,
                      sa * cb + ca * squash * sb, -sa * sb + ca * squash * cb], axis=1)
        # light: a leaf turned edge-on catches less of it; the top glints now and then as it turns
        light = 0.8 + 0.2 * np.abs(c) + np.where(c > 0, self.sheen * np.maximum(np.cos(pitch - 0.45), 0) ** 24, 0)
        gain = self.tint * (light * self.opacity)[:, None]
        for i in np.flatnonzero(seen):
            self._leaf(i, x[i], y[i], m[i], c[i] < 0, abs(c[i]), length[i], gain[i], mx[i], my[i], canvas, mask)

    def _leaf(self, i, x, y, m, back, facing, length, gain, mx, my, canvas, mask):
        spr = sprite(str(self.kind[i]), FAMILIES[self.family[i]], int(self.variant[i]), self.master)
        img = spr.image(back, length * max(facing, 0.4))
        n = img.size[0]
        k = length / (spr.extent * n / spr.size)  # frame px per picture px
        a, b, c, d = m * k
        r = spr.reach * n / spr.size
        pad = 3 * float(self.blur[i]) + 2
        hw, hh = r * math.hypot(a, b) + pad, r * math.hypot(c, d) + pad
        x0, y0 = math.floor(x - hw), math.floor(y - hh)
        w, h = math.ceil(x + hw) - x0, math.ceil(y + hh) - y0
        det = a * d - b * c  # the frame -> picture map: the inverse
        ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
        ox, oy = x0 - x, y0 - y
        data = (ia, ib, ia * ox + ib * oy + n / 2, ic, id_, ic * ox + id_ * oy + n / 2)
        out = img.transform((w, h), Image.Transform.AFFINE, data, resample=Image.Resampling.BICUBIC)
        if self.blur[i] > 0.4:
            out = out.filter(ImageFilter.GaussianBlur(float(self.blur[i])))
        arr = np.asarray(out, np.float32) * (1 / 255.0)
        rgb = arr[..., :3]
        rgb *= gain
        arr[..., 3] *= self.opacity[i]
        hz = self.haze[i]
        if hz > 0.01:  # far away: paler, greyer
            grey = float(spr.rgb @ np.array([0.3, 0.55, 0.15]))
            target = (0.6 * grey + 0.4 * np.array([0.8, 0.82, 0.85])).astype(np.float32)
            rgb *= 1 - hz
            rgb += (hz * target) * arr[..., 3:4]
        if mx * mx + my * my > 1.0:
            arr, dx0, dy0 = smear(arr, mx, my)
            x0, y0 = x0 + dx0, y0 + dy0
        canvas.over_rgba(x0, y0, arr[..., :3], arr[..., 3], mask)


def smear(arr, dx, dy, most=24):
    """`arr` (h, w, channels) smeared over the (dx, dy) px it moves while the shutter is open: copies along the way,
    at most a pixel apart along the way it mostly moves (or `most` copies), each split between two pixels that way.
    Returns (array, x0, y0): where its top-left pixel now sits relative to the original's."""
    n = min(int(math.ceil(max(abs(dx), abs(dy)))) + 1, most)
    h, w = arr.shape[:2]
    px, py = int(math.ceil(abs(dx) / 2)) + 1, int(math.ceil(abs(dy) / 2)) + 1
    out = np.zeros((h + 2 * py, w + 2 * px) + arr.shape[2:], np.float32)
    down = abs(dy) >= abs(dx)
    for j in range(n):
        o = j / (n - 1) - 0.5
        sx, sy = px + o * dx, py + o * dy
        if down:
            iy, ix = math.floor(sy), round(sx)
            f = sy - iy
            out[iy:iy + h, ix:ix + w] += arr * (1 - f)
            out[iy + 1:iy + 1 + h, ix:ix + w] += arr * f
        else:
            ix, iy = math.floor(sx), round(sy)
            f = sx - ix
            out[iy:iy + h, ix:ix + w] += arr * (1 - f)
            out[iy:iy + h, ix + 1:ix + 1 + w] += arr * f
    out *= 1 / n
    return out, -px, -py
