"""The falling leaves' pictures (leaves.py makes them fall), drawn by code, once per design, as premultiplied RGBA
pictures at a few sizes: the front, the back, and the leaf lit from behind.

Nine shapes, each drawn from distance fields (negative inside the leaf):
- maple: five pointed lobes and two small ones by the stem, each a tapering blade along a slightly bent axis with
  teeth along its sides (sugar maple: broad, with pointed side lobes; red maple: shallower, doubly toothed), joined
  to the palm with rounded sinuses (a smooth minimum of their fields); veins from the stem to every lobe tip.
- japanese: a Japanese maple's seven long narrow lobes, cut deep, sharply and doubly toothed.
- ginkgo: a fan, its outer edge wavy and often notched in the middle, fine veins forking out from the stem.
- oak: an English oak's rounded lobes either side of the midrib, deep round sinuses, small ears at the base.
- almond: lá bàng (Terminalia catappa): big, widest near its rounded tip, narrowing to a small notched base, a thick
  midrib and side veins curving up to loop by the edge.
- birch: small, triangular-ovate with a long tip, doubly toothed, straight parallel side veins.
- willow: long and narrow, often a little curved, finely toothed, its back silvery.
- heart: a linden's heart-shaped leaf: rounded lobes either side of the stem, a short pointed tip, fine teeth.
- petal: a cherry blossom's petal, notched at its tip.

Each shape has its own natural colours for the red, yellow and green the sliders count (a ginkgo has no red: its
"red" is a deep golden orange; an oak's red is russet; a bàng leaf goes from wine red to orange, its veins darker;
a willow is yellow-green), varying across the leaf (tones that drift, deeper edges and tips, patches still green
by the veins, freckles, dry tips), with a faint texture. The shading comes from a leaf that is cupped, curled,
twisted, folded along its midrib and a little sunk along its veins, with a thin lighter rim (the edge lets light
through). The back is duller and lighter, its veins standing out (a leaf turned over shows it mirrored). Lit from
behind, a leaf glows: warmer and brighter, its veins and the fine net between them dark against the light.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

MIPS = (288, 144, 72, 36, 18)  # px across a leaf's pictures: it is drawn at one of the first three (as large as
# the leaf will be seen), the smaller ones made from that
FAMILIES = ("red", "yellow", "green")  # the colours the sliders count
PALETTES = FAMILIES + ("turning",)  # how a leaf is painted: "turning" is a green leaf going yellow or red at its
# edges (only when some leaves are already red or yellow: not in summer)
SHAPES = ("maple", "japanese", "ginkgo", "oak", "almond", "birch", "willow", "heart", "petal")
VARIANTS = {"maple": 8, "japanese": 6, "ginkgo": 6, "oak": 6, "almond": 6, "birch": 6, "willow": 6, "heart": 6,
            "petal": 8}  # designs of each shape, per colour (at most: fewer when many shapes fall together)
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


def _smax(a, b, k):
    """A smooth maximum: the intersection of two fields with rounded outside corners."""
    return -_smin(-a, -b, k)


def _noise(rng, shape, cells):
    """Smooth value noise, -1..1 (about), `cells` blobs across the longer side."""
    h, w = shape
    n = max(2, int(cells))
    small = rng.random((max(2, round(n * h / max(h, w))) + 3, max(2, round(n * w / max(h, w))) + 3))
    img = Image.fromarray(small.astype(np.float32), "F")
    big = img.resize((w, h), Image.Resampling.BICUBIC, box=(1.5, 1.5, small.shape[1] - 1.5, small.shape[0] - 1.5))
    return (np.asarray(big, np.float32) - 0.5) * 2


def _net(rng, shape, cells, width):
    """A net of thin wiggly lines (0..1) where smooth noise crosses zero: like the finest veins round the little
    islands of a leaf's blade (`width`: the lines' width, in noise units)."""
    n = _noise(rng, shape, cells)
    return np.exp(-(n / width) ** 2)


class Lobe:
    """A tapering blade along an axis from (ox, oy) at `angle` (radians from straight up, positive to the right),
    `length` long and `width` at its widest (half-widths), the axis bent sideways by `bend` (at the tip, in lengths),
    with teeth along its sides (amp, period: leaf units). kind "vein": a vein only (no blade of its own). The
    profile is (1 - t)^tip (base + t)^b: tip above 1 is a long drawn-out point, 1 a straight one, below 1 a rounded
    end; `base` sets how wide it starts."""

    def __init__(self, ox, oy, angle, length, width, bend=0.0, amp=0.0, period=0.05, double=0.0, phase=(0.0, 0.0),
                 widest=0.35, tip=1.0, base=0.3, teeth_from=0.22, kind="lobe", soft=None, round_base=0.0):
        self.ox, self.oy, self.a, self.L, self.W = ox, oy, angle, length, width
        self.round_base = round_base  # (how round the corners where it is cut across at its base)
        self.bend, self.amp, self.period, self.double, self.phase = bend, amp, period, double, phase
        self.tip, self.base, self.teeth_from, self.kind, self.soft = tip, base, teeth_from, kind, soft
        self.widest = widest
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
        slope = np.minimum(np.abs(self.W * fs / self.L), 6.0)
        d = (np.abs(q) - hw) / np.sqrt(1 + slope * slope)
        if self.tip < 1:  # a rounded end: past its widest, half an ellipse (the profile there is so steep at the
            # tip that every point across from it would seem to be on its edge)
            sw = self.widest * self.L
            past = s > sw
            if past.any():
                a, b = max(1e-6, self.L - sw), self.W
                qq, uu = np.abs(q[past]), s[past] - sw
                r = np.sqrt((qq / b) ** 2 + (uu / a) ** 2)
                g = np.sqrt((qq / (b * b)) ** 2 + (uu / (a * a)) ** 2) / np.maximum(r, 1e-9)
                d[past] = (r - 1) / np.maximum(g, 1e-9)
        if self.amp > 0:  # the teeth move the margin only: they fade out within a few of their sizes of it
            at = np.abs(d) < 7 * self.amp
            ds = d[at]
            tw, ts = self.teeth(s[at], q[at] > 0)
            near = np.exp(-(ds / (2.5 * self.amp)) ** 2)
            d[at] = (ds - tw * near) / np.sqrt(1 + (ts * near) ** 2)
        return np.maximum(_smax(d, -s, self.round_base), s - self.L), t

    def bounds(self):
        """(x0, y0, x1, y1): a box round the lobe (its outline, a little grown)."""
        ss = np.linspace(0, self.L, 17)
        xs, ys = self.axis(ss)
        hw = self.W * self.profile(ss / self.L)[0] * 1.05 + self.amp + 0.01
        ca, sa = math.cos(self.a), math.sin(self.a)
        px, py = np.concatenate([xs + hw * ca, xs - hw * ca]), np.concatenate([ys - hw * sa, ys + hw * sa])
        grow = 0.04 * self.L * (1 + abs(self.bend) * 4)  # (the bend turns the outline's sides a little)
        return float(px.min()) - grow, float(py.min()) - grow, float(px.max()) + grow, float(py.max()) + grow


# --- the shapes -------------------------------------------------------------------------------------------------

def _jit(rng):
    return lambda v, j=0.03: v * rng.uniform(1 - j, 1 + j)  # (one side a little unlike the other)


def _common(rng, shape, field, veins, stem, stem_w, cup=(-0.25, 0.45), curl=(-0.3, 0.4), fold=(0.02, 0.05),
            wave=(0.01, 0.04), vfold=(0.0, 0.0), twist=(-0.25, 0.25), **extra):
    """What every design has: how it is cupped, curled, twisted and folded, its stem, how its veins run."""
    return {"shape": shape, "field": field, "veins": veins, "stem": rng.uniform(*stem),
            "stem_bend": rng.normal(0, 0.12), "stem_w": rng.uniform(*stem_w), "cup": rng.uniform(*cup),
            "curl": rng.uniform(*curl), "fold": rng.uniform(*fold), "wave": rng.uniform(*wave),
            "vfold": rng.uniform(*vfold), "twist": rng.uniform(*twist), "lobes": [], "palm": 0.0, "k": 0.0, **extra}


def maple_design(rng):
    """A maple leaf with five lobes (and two small ones by the stem): broad lobes with pointed side lobes (sugar
    maple), or shallower lobes with doubly toothed edges (red maple). The two sides mirror each other, each a little
    different (a real leaf is never quite symmetric)."""
    broad = rng.random() < 0.55
    lobes, jit = [], _jit(rng)
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
        bend = rng.uniform(-0.02, 0.06)
        widest, tip = rng.uniform(0.3, 0.42), rng.uniform(0.9, 1.2)
        # pointed side lobes on the main lobes (big ones on a sugar maple's, small on a red maple's): planned once
        # for both of a pair, so the two sides match
        plan = []
        if i < 3:
            pairs = (1 if i == 2 else int(rng.integers(1, 3))) if broad else int(i < 2 and rng.random() < 0.6)
            for j in range(pairs):
                t0 = (rng.uniform(0.5, 0.62) if j == 0 else rng.uniform(0.3, 0.4)) - (0.06 if i == 2 else 0)
                size = (0.3 if broad else 0.16) * (1 - 0.35 * j) * rng.uniform(0.8, 1.15)
                inner = i == 0 or j == 0 or rng.random() < 0.5  # (the small inner ones are often missing)
                plan.append((t0, size, rng.uniform(0.75, 1.0), rng.uniform(0.32, 0.42), inner))
        for side in ((0,) if i == 0 else (-1, 1)):
            a = side * (a0 + rng.uniform(-0.02, 0.02)) + (rng.uniform(-0.012, 0.012) if i == 0 else 0)
            L, W = jit(ln, 0.02), jit(w)
            lobe = Lobe(0, 0, a, L, W, bend=(side * jit(bend, 0.2) if i else rng.normal(0, 0.01)),
                        amp=amp, period=period, double=double, phase=tuple(rng.random(2)),
                        widest=jit(widest), tip=jit(tip), base=0.35, teeth_from=0.35 if broad else 0.25)
            lobes.append(lobe)
            for t0, size, dev, thin, inner in plan:
                for s2 in (-1, 1):
                    if i > 0 and s2 == -side and not inner:  # (towards the middle lobe)
                        continue
                    bx, by = lobe.axis(jit(t0, 0.03) * L)
                    sz = jit(size, 0.06) * L
                    lobes.append(Lobe(float(bx), float(by), lobe.a + s2 * jit(dev, 0.03), sz + W * 0.6,
                                      sz * thin, amp=amp * 0.6, period=period, phase=tuple(rng.random(2)),
                                      widest=0.32, tip=1.35, base=0.45, teeth_from=0.5, kind="side"))
    return _common(rng, "maple", "lobes", {"style": "palmate"}, (0.4, 0.62), (0.022, 0.03), lobes=lobes, palm=palm,
                   k=k, curl_by="along")


def japanese_design(rng):
    """A Japanese maple's leaf: seven long narrow lobes, pointed, cut deep (about two thirds of the way in), sharply
    and doubly toothed, the lowest ones reaching out sideways."""
    lobes, jit = [], _jit(rng)
    palm, k = rng.uniform(0.13, 0.18), rng.uniform(0.025, 0.035)
    spread = rng.uniform(0.56, 0.62)  # radians between lobes
    lengths = (1.0, rng.uniform(0.9, 0.97), rng.uniform(0.72, 0.82), rng.uniform(0.42, 0.54))
    width = rng.uniform(0.105, 0.125)
    amp, period, double = rng.uniform(0.014, 0.02), rng.uniform(0.038, 0.05), rng.uniform(0.4, 0.6)
    widest, tip = rng.uniform(0.52, 0.6), rng.uniform(1.25, 1.5)  # (narrow where the lobes part)
    for i, ln in enumerate(lengths):
        bend = rng.uniform(0.0, 0.08)
        w = width * (0.85 if i == 3 else 1.0)
        for side in ((0,) if i == 0 else (-1, 1)):
            a = side * (i * spread + rng.uniform(-0.012, 0.012)) if i else rng.uniform(-0.01, 0.01)
            lobes.append(Lobe(0, 0, a, jit(ln, 0.02), jit(w, 0.02),
                              bend=side * jit(bend, 0.2) if i else rng.normal(0, 0.01),
                              amp=amp, period=period, double=double, phase=tuple(rng.random(2)),
                              widest=jit(widest, 0.02), tip=jit(tip, 0.02), base=0.1, teeth_from=0.36))
    return _common(rng, "japanese", "lobes", {"style": "palmate"}, (0.4, 0.55), (0.016, 0.022), lobes=lobes,
                   palm=palm, k=k, curl_by="along", cup=(-0.1, 0.5))


def ginkgo_design(rng):
    """A ginkgo leaf: a fan on a long stem, its outer edge wavy, often notched in the middle (some deeply, in two
    lobes), its sides drawn in toward the stem; fine veins fork out from the stem to the edge."""
    deep = rng.random()
    notch = rng.uniform(0.0, 0.06) if deep < 0.35 else rng.uniform(0.1, 0.24) if deep < 0.8 else \
        rng.uniform(0.3, 0.45)
    return _common(rng, "ginkgo", "ginkgo", {"style": "fan", "count": int(rng.integers(26, 38))}, (0.55, 0.85),
                   (0.016, 0.022), half=rng.uniform(0.78, 1.15), R=1.0, notch=notch,
                   notch_w=rng.uniform(0.05, 0.1) + 0.12 * notch, notch_at=rng.normal(0, 0.05),
                   waves=[(rng.uniform(0.006, 0.016), rng.uniform(9, 16), rng.uniform(0, 2 * math.pi)),
                          (rng.uniform(0.004, 0.012), rng.uniform(20, 34), rng.uniform(0, 2 * math.pi)),
                          (rng.uniform(0.01, 0.03), rng.uniform(1.5, 3.5), rng.uniform(0, 2 * math.pi))],
                   concave=rng.uniform(0.25, 0.45), round=rng.uniform(0.05, 0.09), curl_by="r",
                   cup=(0.1, 0.5), twist=(-0.15, 0.15), fold=(0.01, 0.02))


def oak_design(rng):
    """An English oak's leaf: rounded lobes either side of the midrib (not opposite each other), the biggest
    beyond the middle, deep rounded sinuses, a rounded end lobe and small ears at the base."""
    jit = _jit(rng)
    lobes = [Lobe(0, 0, rng.normal(0, 0.04), 1.0, rng.uniform(0.075, 0.095), bend=rng.normal(0, 0.05),
                  widest=0.7, tip=0.6, base=0.25)]
    mid = lobes[0]
    pairs = int(rng.integers(4, 6))
    for side in (-1, 1):
        offset = rng.uniform(0.35, 0.65) if side > 0 else 0.0  # (the two sides' lobes alternate)
        for j in range(pairs):
            s = 0.1 + 0.74 * (j + offset + rng.uniform(-0.08, 0.08)) / pairs
            if s > 0.86:
                continue
            env = 0.2 + 0.24 * math.sin(math.pi * min(1.0, s / 0.92) ** 0.8)  # (biggest beyond the middle)
            bx, by = mid.axis(s)
            ang = mid.a + side * (1.1 - 0.6 * s + rng.normal(0, 0.08))
            lobes.append(Lobe(float(bx), float(by), ang, jit(env, 0.12), jit(rng.uniform(0.06, 0.08), 0.1),
                              bend=side * rng.uniform(0.0, 0.1), widest=0.62, tip=0.6, base=0.4, kind="side",
                              soft=rng.uniform(0.025, 0.04)))
    for side in (-1, 1):  # the ears by the stem
        lobes.append(Lobe(0.0, 0.03, side * rng.uniform(2.2, 2.6), rng.uniform(0.07, 0.1), 0.045, widest=0.5,
                          tip=0.6, base=0.5, kind="side", soft=0.02))
    return _common(rng, "oak", "lobes", {"style": "pinnate", "step": None, "main_w": 0.022, "sub_w": 0.009},
                   (0.08, 0.14), (0.02, 0.026), lobes=lobes, k=0.06, curl_by="r", vfold=(0.03, 0.08),
                   wave=(0.02, 0.05))


def almond_design(rng):
    """A bàng leaf (Terminalia catappa): big and obovate, widest near its rounded end (with a short point), narrowing
    to a small notched base on a short thick stem; a thick midrib and side veins curving up to loop by the edge."""
    jit = _jit(rng)
    width = rng.uniform(0.26, 0.3)
    lobes = [Lobe(0, 0, rng.normal(0, 0.03), 1.0, width, bend=rng.normal(0, 0.04), widest=rng.uniform(0.7, 0.76),
                  tip=rng.uniform(0.38, 0.48), base=0.04, teeth_from=0.0)]
    mid = lobes[0]
    if rng.random() < 0.7:  # a short abrupt point at the rounded end
        tx, ty = mid.axis(0.93)
        lobes.append(Lobe(float(tx), float(ty), mid.a + rng.normal(0, 0.05), rng.uniform(0.08, 0.12), 0.045,
                          widest=0.15, tip=1.2, base=0.3, kind="side", soft=0.05))
    for side in (-1, 1):  # the small ears of the base
        lobes.append(Lobe(0.0, 0.04, side * rng.uniform(2.2, 2.5), jit(0.045, 0.15), 0.024, widest=0.5, tip=0.6,
                          base=0.5, kind="side", soft=0.012))
    return _common(rng, "almond", "lobes", {"style": "pinnate", "step": rng.uniform(0.085, 0.1), "lean": 1.0,
                                             "curve": 0.42, "reach": 0.9, "main_w": 0.03, "side_w": 0.009,
                                             "from": 0.08},
                   (0.06, 0.1), (0.032, 0.04), lobes=lobes, curl_by="r", vfold=(0.03, 0.07), fold=(0.04, 0.07),
                   wave=(0.02, 0.05), cup=(-0.15, 0.4))


def birch_design(rng):
    """A birch leaf: small, round-ovate with a short drawn-out point and a broad rounded base, doubly toothed;
    straight side veins running out to the teeth, all alike."""
    lobes = [Lobe(0, 0, rng.normal(0, 0.03), 1.0, rng.uniform(0.35, 0.41), bend=rng.normal(0, 0.03),
                  amp=rng.uniform(0.013, 0.018), period=rng.uniform(0.048, 0.06), double=rng.uniform(0.5, 0.7),
                  phase=tuple(rng.random(2)), widest=rng.uniform(0.3, 0.36), tip=rng.uniform(1.35, 1.6),
                  base=rng.uniform(0.14, 0.22), teeth_from=0.16, round_base=rng.uniform(0.05, 0.08))]
    return _common(rng, "birch", "lobes", {"style": "pinnate", "step": rng.uniform(0.1, 0.12), "lean": 0.72,
                                            "curve": 0.12, "reach": 1.0, "main_w": 0.016, "side_w": 0.0065,
                                            "from": 0.06},
                   (0.25, 0.36), (0.014, 0.018), lobes=lobes, curl_by="r", vfold=(0.02, 0.06))


def willow_design(rng):
    """A willow leaf: long and narrow, often a little curved, with a long point and fine teeth; a pale midrib and
    faint side veins; its back silvery."""
    lobes = [Lobe(0, 0, rng.normal(0, 0.03), 1.0, rng.uniform(0.07, 0.09), bend=rng.choice([-1, 1]) *
                  rng.uniform(0.02, 0.14), amp=rng.uniform(0.0035, 0.005), period=rng.uniform(0.022, 0.03),
                  phase=tuple(rng.random(2)), widest=rng.uniform(0.35, 0.45), tip=rng.uniform(1.4, 1.8), base=0.02,
                  teeth_from=0.08, round_base=0.02)]
    return _common(rng, "willow", "lobes", {"style": "pinnate", "step": rng.uniform(0.045, 0.055), "lean": 0.55,
                                             "curve": 0.35, "reach": 0.85, "main_w": 0.016, "side_w": 0.004,
                                             "from": 0.05},
                   (0.06, 0.1), (0.012, 0.016), lobes=lobes, curl_by="r", cup=(0.1, 0.5), curl=(-0.4, 0.4),
                   twist=(-0.5, 0.5), vfold=(0.02, 0.05))


def heart_design(rng):
    """A heart-shaped leaf, like a linden's: broad, with rounded lobes either side of a notch at the stem (one a
    little bigger: the leaf is lopsided), a short drawn-out point and fine teeth; veins fan out from the stem, then
    run along the midrib."""
    tilt = rng.normal(0, 0.05)
    drop = rng.uniform(0.13, 0.18)
    lobes = [Lobe(0.0, -drop, tilt, 1.0 + drop, rng.uniform(0.44, 0.5), bend=rng.normal(0, 0.04),
                  amp=rng.uniform(0.008, 0.012), period=rng.uniform(0.034, 0.044), double=rng.uniform(0.0, 0.3),
                  phase=tuple(rng.random(2)), widest=rng.uniform(0.3, 0.36), tip=rng.uniform(1.6, 2.0),
                  base=rng.uniform(0.32, 0.4), teeth_from=0.12, round_base=rng.uniform(0.1, 0.14))]
    for side in (-1, 1):  # the veins out to the lobes by the stem
        lobes.append(Lobe(0.0, 0.0, tilt + side * rng.uniform(1.7, 1.95), rng.uniform(0.3, 0.36), 0.1, kind="vein"))
    return _common(rng, "heart", "lobes", {"style": "pinnate", "step": rng.uniform(0.12, 0.14), "lean": 0.8,
                                            "curve": 0.3, "reach": 0.92, "main_w": 0.017, "side_w": 0.007,
                                            "from": 0.3, "sub_w": 0.011, "sub_side": True},
                   (0.3, 0.45), (0.014, 0.018), lobes=lobes, curl_by="r", vfold=(0.01, 0.04),
                   sinus=(rng.normal(0, 0.01), 0.02, rng.uniform(0.5, 0.75), 0.03))


def petal_design(rng):
    """A cherry petal: a rounded blade, narrow at its base, with a small notch at its tip."""
    return _common(rng, "petal", "petal", {"style": "petal"}, (0.0, 0.0), (0.0, 0.0), w=rng.uniform(0.34, 0.42),
                   notch=rng.uniform(0.06, 0.12), tilt=rng.normal(0, 0.06), cup=(0.2, 0.6), curl=(-0.2, 0.3),
                   fold=(0.0, 0.0), wave=(0.01, 0.03), twist=(-0.2, 0.2), curl_by="along")


DESIGNS = {"maple": maple_design, "japanese": japanese_design, "ginkgo": ginkgo_design, "oak": oak_design,
           "almond": almond_design, "birch": birch_design, "willow": willow_design, "heart": heart_design,
           "petal": petal_design}


# --- the shapes' fields -----------------------------------------------------------------------------------------

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


def _ginkgo_radius(g, th):
    """How far a ginkgo fan reaches from its stem at angle th (from straight up)."""
    u = th / g["half"]
    r = g["R"] * (1 - 0.06 * u * u)
    for amp, freq, ph in g["waves"]:
        r = r + amp * np.sin(freq * th + ph)
    return r - g["notch"] * np.clip(1 - np.abs(th - g["notch_at"]) / g["notch_w"], 0, 1) ** 1.7


def _ginkgo_field(g, x, y):
    r = np.hypot(x, y)
    th = np.arctan2(x, y)
    R = _ginkgo_radius(g, th)
    dR = (_ginkgo_radius(g, th + 1e-3) - _ginkgo_radius(g, th - 1e-3)) / 2e-3
    out = (r - R) / np.sqrt(1 + (dR / np.maximum(r, 0.05)) ** 2)  # (a distance: as steep at the notch as elsewhere)
    side = g["half"] * (1 - g["concave"] * np.clip(1 - r / g["R"], 0, 1) ** 2)  # the sides drawn in to the stem
    across = r * np.sin(np.clip(np.abs(th) - side, -0.5 * math.pi, 0.5 * math.pi))
    d = _smax(out, across, g["round"])
    return d, np.clip(r / np.maximum(R, 1e-3), 0.0, 1.0)


SOFT = 0.025  # how softly the lobes' positions along them blend where lobes meet (leaf units)


def _blade(design, x, y, k_side=0.4):
    """The blade's distance-like field (negative inside, leaf units) at x, y, and how far along its lobe a point is
    (0 at the palm .. 1 at a tip; blended smoothly where lobes meet)."""
    if design["field"] == "petal":
        return _petal_field(design, x, y)
    if design["field"] == "ginkgo":
        return _ginkgo_field(design, x, y)
    palm, k = design["palm"], design["k"]
    if palm > 0:  # the palm, cut straight across at the bottom: the base of the leaf is the lobes by the stem
        d = np.maximum(np.hypot(x, y - 0.4 * palm) - palm, -(y + 0.25 * palm))
    else:
        d = np.full(x.shape, 10.0, np.float32)
    weight = np.exp(-np.clip(d, -0.5, 1.0) / SOFT)  # (the palm: along 0)
    along = np.zeros(x.shape)
    xs, ys = x[0, :], y[:, 0]
    for lobe in design["lobes"]:
        if lobe.kind == "vein":
            continue
        x0, y0, x1, y1 = lobe.bounds()
        grow = k + 0.2
        cols = np.flatnonzero((xs >= x0 - grow) & (xs <= x1 + grow))
        rows = np.flatnonzero((ys >= y0 - grow) & (ys <= y1 + grow))
        if not len(cols) or not len(rows):
            continue
        sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        dl, t = lobe.field(x[sl], y[sl])
        soft = lobe.soft if lobe.soft is not None else k * (k_side if lobe.kind == "side" else 1.0)
        d[sl] = _smin(d[sl], dl, soft)
        w = np.exp(-np.clip(dl, -0.5, 1.0) / SOFT)
        weight[sl] += w
        along[sl] += w * np.clip(t, 0, 1)
    if design.get("sinus"):  # a notch cut up into the base, to the stem (a heart's)
        sx, sy, half, soft = design["sinus"]
        c, s = math.cos(half), math.sin(half)
        wedge = (np.abs(x - sx) * c - (sy - y) * s)  # (negative inside the wedge, which opens downwards)
        d = _smax(d, -wedge, soft)
    return d, along / weight


def _fan_veins(design, rng):
    """A ginkgo's veins: two leave the stem and fork again and again on their way out (each fork a little further
    out), so that 32 of them reach the edge, evenly spread."""
    half, depth = design["half"] * 0.95, 5
    count = 1 << depth
    radii = (0.0, 0.07, 0.19, 0.33, 0.5)
    forks = {(d, b): radii[d] * rng.uniform(0.85, 1.15) for d in range(1, depth) for b in range(1 << d)}
    lines = []
    for i in range(count):
        end_a = ((i + 0.5) / count * 2 - 1) * half + rng.normal(0, 0.008)
        pts = [(0.0, 0.0)]  # (radius, angle) of the forks on its way
        for d in range(1, depth):
            b = i >> (depth - d)
            pts.append((forks[(d, b)], ((b + 0.5) / (1 << d) * 2 - 1) * half))
        pts.append((float(_ginkgo_radius(design, np.array(end_a))) * rng.uniform(0.94, 0.98), end_a))
        line = []
        for (r0, a0), (r1, a1) in zip(pts, pts[1:]):
            for f in (0.0, 0.25, 0.5, 0.75):
                r, a = r0 + (r1 - r0) * f, a0 + (a1 - a0) * f
                side = design["half"] * (1 - design["concave"] * max(0.0, 1 - r) ** 2)
                a = float(np.clip(a, -side * 0.93, side * 0.93))
                line.append((r * math.sin(a), r * math.cos(a)))
        line.append((pts[-1][0] * math.sin(end_a), pts[-1][0] * math.cos(end_a)))
        lines.append([(x, y, 0.0045 * (1 - 0.3 * j / len(line))) for j, (x, y) in enumerate(line)])
    return [], lines


def _veins(design):
    """The veins as polylines of (x, y, width) points, leaf units: the main veins (to the lobe tips, or the midrib),
    and the thin side veins from them out towards the margin."""
    main, side_veins = [], []
    rng = np.random.default_rng(design["seed"])
    style = design["veins"]["style"]
    if style == "petal":
        for a in np.linspace(-0.55, 0.55, 9):
            a = a + rng.normal(0, 0.04)
            side_veins.append([(float(a * s * design["w"] * 1.25 * (1.2 - 0.2 * s)), float(s), 0.0045 * (1 - 0.5 * s))
                               for s in np.linspace(0.04, 0.82, 8)])
        return main, side_veins
    if style == "fan":
        return _fan_veins(design, rng)
    v = design["veins"]
    pinnate = style == "pinnate"
    for i, lobe in enumerate(design["lobes"]):
        sub = lobe.kind in ("side", "vein")
        if lobe.L < 0.12:  # (the ears, a short point: no veins of their own)
            continue
        n = 10
        ss = np.linspace(0, lobe.L * (0.88 if sub else 0.96), n)
        xs, ys = lobe.axis(ss)
        if pinnate:
            w0 = v["sub_w"] if sub and "sub_w" in v else v.get("side_w", 0.008) * 1.4 if sub else v["main_w"]
        else:
            w0 = 0.008 if sub else 0.015 * min(1.0, lobe.L ** 0.5)
        main.append([(float(xs[j]), float(ys[j]), w0 * (1 - 0.7 * j / (n - 1))) for j in range(n)])
        if (sub and not v.get("sub_side")) or (pinnate and i > 0 and not sub) or (pinnate and not v.get("step")):
            continue
        # side veins every `step` along the lobe (not in the palm), leaning to its tip, curving out to the margin
        if pinnate:
            step, lean, curve, reach0, width = v["step"], v["lean"], v["curve"], v["reach"], v["side_w"]
            s0 = v["from"] * lobe.L if not sub else 0.3 * lobe.L
        else:
            step, lean, curve, reach0, width = max(0.065, 0.13 * lobe.L), 0.95, 0.25, 0.88, 0.006
            s0 = max(step, 0.9 * design["palm"])
        while s0 < lobe.L * 0.86:
            f, _ = lobe.profile(s0 / lobe.L + 0.1)
            reach = lobe.W * f * reach0 * rng.uniform(0.92, 1.05)
            for sgn in ((-1, 1) if not sub else (1,) if lobe.a > 0 else (-1,)):  # (a lobe by the stem: outwards)
                at = s0 + rng.normal(0, 0.12) * step
                bx, by = lobe.axis(at)
                ang = lobe.a + sgn * (lean + rng.normal(0, 0.06))
                bend = curve * rng.uniform(0.7, 1.2)
                pts = []
                for j in range(7):
                    r = reach * j / 6 / math.sin(lean)
                    a = ang - sgn * bend * (j / 6) ** 2  # curving towards the tip
                    pts.append((float(bx + r * math.sin(a)), float(by + r * math.cos(a)), width * (1 - 0.6 * j / 6)))
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
    if design["shape"] == "petal":
        return []
    n, ln, bend, w = 8, design["stem"], design["stem_bend"], design["stem_w"]
    pts = []
    for j in range(n + 1):
        s = j / n
        pts.append((bend * ln * s * s, 0.02 - ln * s, w * (1.0 - 0.25 * s + 0.35 * max(0.0, s - 0.85) / 0.15)))
    return pts


def _bounds(design):
    if design["field"] == "petal":
        return -design["w"] - 0.05, -0.05, design["w"] + 0.05, 1.05
    if design["field"] == "ginkgo":
        reach = design["R"] * 1.05
        across = reach * math.sin(min(design["half"], 0.5 * math.pi)) + 0.05
        bx, by = [-across, across], [-0.05, reach + 0.05]
    else:
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


# --- the colours ------------------------------------------------------------------------------------------------

def _mix(a, b, f):
    """a (H, W, 3) towards colour b (a colour, or (H, W, 3)) by f (H, W) or a number."""
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
    """Colours from `stops` [(position, (r, g, b)), ...] at v (0..1: a number, or an array: (..., 3))."""
    pos = [p for p, _ in stops]
    v = np.asarray(v, np.float32)
    return np.stack([np.interp(v, pos, [c[i] for _, c in stops]) for i in range(3)], axis=-1).astype(np.float32)


# Each shape's colours: its red, yellow and green (ramps: each leaf takes a tone of them, drifting across it), the
# deeper colour of its edges and tips when red or yellow, its "turning" warm colours (green going ...), how its
# veins show on the front (lighter or darker), its stem's colour.
PAINT = {
    "maple": {"red": [(0.0, (0.40, 0.03, 0.08)), (0.3, (0.60, 0.04, 0.09)), (0.65, (0.76, 0.07, 0.07)),
                      (1.0, (0.86, 0.17, 0.06))],
              "yellow": [(0.0, (0.93, 0.83, 0.26)), (0.5, (0.96, 0.73, 0.14)), (1.0, (0.92, 0.57, 0.09))],
              "green": [(0.0, (0.17, 0.38, 0.09)), (0.5, (0.27, 0.49, 0.12)), (1.0, (0.42, 0.6, 0.15))],
              "deep": {"red": (0.32, 0.02, 0.06), "yellow": (0.88, 0.45, 0.07)},
              "warm": [((0.86, 0.66, 0.14), 0.7), ((0.74, 0.2, 0.07), 0.3)], "veins": 0.45, "dry": 0.45,
              "freckles": (3, 13)},
    "japanese": {"red": [(0.0, (0.5, 0.01, 0.06)), (0.35, (0.7, 0.03, 0.07)), (0.7, (0.86, 0.1, 0.06)),
                         (1.0, (0.93, 0.26, 0.05))],
                 "yellow": [(0.0, (0.97, 0.74, 0.14)), (0.5, (0.95, 0.6, 0.09)), (1.0, (0.92, 0.46, 0.07))],
                 "green": [(0.0, (0.27, 0.5, 0.11)), (0.5, (0.38, 0.6, 0.13)), (1.0, (0.52, 0.68, 0.17))],
                 "deep": {"red": (0.38, 0.01, 0.07), "yellow": (0.86, 0.3, 0.05)},
                 "warm": [((0.85, 0.22, 0.06), 0.6), ((0.93, 0.55, 0.1), 0.4)], "veins": 0.4, "dry": 0.25,
                 "freckles": (0, 4)},
    "ginkgo": {"red": [(0.0, (0.95, 0.62, 0.08)), (0.5, (0.92, 0.53, 0.06)), (1.0, (0.85, 0.44, 0.05))],
               "yellow": [(0.0, (0.99, 0.9, 0.36)), (0.5, (0.98, 0.82, 0.2)), (1.0, (0.96, 0.74, 0.12))],
               "green": [(0.0, (0.42, 0.58, 0.14)), (0.5, (0.53, 0.66, 0.18)), (1.0, (0.64, 0.73, 0.22))],
               "deep": {"red": (0.7, 0.33, 0.05), "yellow": (0.9, 0.62, 0.1)},
               "warm": [((0.93, 0.8, 0.22), 1.0)], "veins": -0.3, "dry": 0.3, "freckles": (0, 3)},
    "oak": {"red": [(0.0, (0.36, 0.12, 0.05)), (0.35, (0.5, 0.18, 0.06)), (0.75, (0.62, 0.27, 0.09)),
                    (1.0, (0.62, 0.36, 0.13))],
            "yellow": [(0.0, (0.84, 0.68, 0.28)), (0.5, (0.78, 0.58, 0.21)), (1.0, (0.68, 0.47, 0.17))],
            "green": [(0.0, (0.13, 0.27, 0.07)), (0.5, (0.2, 0.35, 0.09)), (1.0, (0.3, 0.43, 0.12))],
            "deep": {"red": (0.3, 0.13, 0.05), "yellow": (0.5, 0.32, 0.12)},
            "warm": [((0.66, 0.52, 0.18), 0.6), ((0.55, 0.28, 0.08), 0.4)], "veins": 0.22, "dry": 0.7,
            "freckles": (2, 9)},
    "almond": {"red": [(0.0, (0.42, 0.03, 0.05)), (0.3, (0.6, 0.06, 0.05)), (0.65, (0.78, 0.17, 0.05)),
                       (1.0, (0.88, 0.36, 0.07))],
               "yellow": [(0.0, (0.96, 0.7, 0.15)), (0.5, (0.93, 0.56, 0.1)), (1.0, (0.8, 0.43, 0.11))],
               "green": [(0.0, (0.1, 0.3, 0.07)), (0.5, (0.17, 0.4, 0.09)), (1.0, (0.26, 0.49, 0.12))],
               "deep": {"red": (0.33, 0.06, 0.04), "yellow": (0.6, 0.27, 0.07)},
               "warm": [((0.8, 0.2, 0.05), 0.65), ((0.92, 0.55, 0.1), 0.35)], "veins": -0.35, "dry": 0.5,
               "freckles": (1, 7)},
    "birch": {"red": [(0.0, (0.88, 0.55, 0.12)), (0.5, (0.82, 0.45, 0.1)), (1.0, (0.7, 0.36, 0.11))],
              "yellow": [(0.0, (0.98, 0.91, 0.34)), (0.5, (0.97, 0.83, 0.21)), (1.0, (0.94, 0.73, 0.14))],
              "green": [(0.0, (0.28, 0.48, 0.11)), (0.5, (0.38, 0.57, 0.14)), (1.0, (0.5, 0.65, 0.18))],
              "deep": {"red": (0.55, 0.26, 0.08), "yellow": (0.85, 0.6, 0.12)},
              "warm": [((0.92, 0.8, 0.22), 1.0)], "veins": 0.2, "dry": 0.35, "freckles": (2, 8)},
    "willow": {"red": [(0.0, (0.84, 0.58, 0.2)), (0.5, (0.79, 0.5, 0.17)), (1.0, (0.7, 0.43, 0.16))],
               "yellow": [(0.0, (0.8, 0.8, 0.3)), (0.5, (0.88, 0.81, 0.25)), (1.0, (0.94, 0.78, 0.2))],
               "green": [(0.0, (0.34, 0.47, 0.19)), (0.5, (0.44, 0.55, 0.21)), (1.0, (0.55, 0.63, 0.24))],
               "deep": {"red": (0.55, 0.32, 0.12), "yellow": (0.72, 0.62, 0.2)},
               "warm": [((0.86, 0.8, 0.26), 1.0)], "veins": 0.3, "dry": 0.3, "freckles": (0, 4)},
    "heart": {"red": [(0.0, (0.78, 0.38, 0.09)), (0.5, (0.69, 0.31, 0.08)), (1.0, (0.58, 0.25, 0.09))],
              "yellow": [(0.0, (0.97, 0.87, 0.32)), (0.5, (0.95, 0.79, 0.21)), (1.0, (0.9, 0.68, 0.14))],
              "green": [(0.0, (0.22, 0.43, 0.1)), (0.5, (0.3, 0.51, 0.13)), (1.0, (0.4, 0.58, 0.16))],
              "deep": {"red": (0.45, 0.2, 0.07), "yellow": (0.8, 0.5, 0.1)},
              "warm": [((0.9, 0.78, 0.22), 0.8), ((0.75, 0.38, 0.08), 0.2)], "veins": 0.25, "dry": 0.4,
              "freckles": (2, 9)},
}
PINK = [(0.0, (0.99, 0.84, 0.89)), (0.5, (0.98, 0.77, 0.85)), (1.0, (0.96, 0.68, 0.79))]


def _paint(design, family, rng, f):
    """Front and back colours (H, W, 3, 0..1) of a leaf from its fields `f`, its colour lit from behind, and its
    stem's colour."""
    shape = f["alpha"].shape
    e, along, halo = f["inside"], f["along"], f["halo"]
    main, side = f["main"], f["side"]
    tips = _smoothstep(0.45, 1.0, along)
    big, mid = f["big"], _noise(rng, shape, 11)
    grain = _blur(np.random.default_rng([design["seed"], 7]).normal(0.0, 1.0, shape).astype(np.float32),
                  0.6 * shape[0] / MIPS[0])  # (the same at any size of the picture)
    edge = np.exp(-e / 0.06)
    heart = np.maximum(halo, np.exp(-f["r"] / 0.3))  # near the main veins and the middle
    kind = design["shape"]
    tone = rng.random()
    drift = 0.22 * big + 0.07 * mid  # (a leaf's colour drifts across it: no flat fill)
    if kind == "petal":
        col = _ramp(PINK, np.clip(tone + 0.5 * drift, 0, 1))
        col = _mix(col, (0.93, 0.5, 0.64), 0.55 * (1 - _smoothstep(0.0, 0.45, along)) ** 2)  # deep pink at the base
        col = _mix(col, (1.0, 0.95, 0.97), 0.35 * edge * _smoothstep(0.4, 1.0, along))  # paler at the edge
        veins, vein_k = np.array((0.92, 0.6, 0.72), np.float32), 0.2
        stem = np.array((0.8, 0.55, 0.5), np.float32)
        col = col * (1 + (0.07 * mid + 0.03 * grain)[..., None] * 0.3)
        front = _mix(col, veins, np.clip(vein_k * side, 0, 1))
        return np.clip(front, 0, 1), col * 1.02, _lit(col, f, 0.25), stem
    pal = PAINT[kind]
    green = _ramp(pal["green"], np.clip(tone + drift, 0, 1))
    if family in ("red", "yellow"):
        base = _ramp(pal[family], np.clip(tone + drift, 0, 1))
        col = base
        bright = _ramp(pal[family], np.float32(min(1.0, tone + 0.35)))
        col = _mix(col, bright, np.clip(0.45 * big + 0.15 * mid, 0, 1) * rng.uniform(0.2, 0.6))  # brighter patches
        deep = pal["deep"][family]
        col = _mix(col, deep, np.clip(edge * rng.uniform(0.3, 0.75) + 0.3 * tips * rng.random(), 0, 0.9))
        if family == "yellow" and kind != "ginkgo":  # green-tinged parts left by the veins, in the middle
            tinge = _mix(green, (0.75, 0.72, 0.2), 0.4)
            col = _mix(col, tinge, np.clip(heart * rng.uniform(0.15, 0.55) + np.clip(big - 0.3, 0, 1)
                                           * rng.uniform(0.2, 0.7), 0, 0.8))
        elif family == "yellow":  # a ginkgo: a little green left by the stem only
            col = _mix(col, (0.72, 0.74, 0.22), np.clip((1 - _smoothstep(0.0, 0.35, along)) * rng.uniform(0, 0.6),
                                                        0, 1))
        if family == "red" and rng.random() < 0.35:  # a little orange-yellow left by the main veins
            col = _mix(col, _ramp(pal["yellow"], np.float32(0.7)),
                       np.clip(halo * rng.uniform(0.15, 0.4) * (1 - edge), 0, 1))
        if rng.random() < 0.15:  # still turning: a patch of green left in the middle
            col = _mix(col, _mix(green, base, 0.4), np.clip(0.9 * big + 0.2 * mid - 0.15, 0, 0.6) * (1 - tips)
                       * (1 - edge))
        if rng.random() < pal["dry"]:  # dry brown tips and edges
            col = _mix(col, (0.45, 0.28, 0.1), np.clip(_smoothstep(0.75, 1.0, along) * edge * 1.6, 0, 0.85)
                       * rng.uniform(0.5, 1.0))
    else:
        base = green
        col = _mix(green, green * 1.22 + np.array((0.05, 0.05, 0.0), np.float32), np.clip(halo * 0.5, 0, 1))
        col = _mix(col, green * 0.8, np.clip(edge * 0.5, 0, 1))
        if family == "turning":  # early autumn: tips and edges going warm, a yellow-green patch or two
            warms = pal["warm"]
            warm = warms[int(rng.choice(len(warms), p=[w for _, w in warms]))][0]
            col = _mix(col, (0.62, 0.66, 0.16), np.clip(0.6 * big + 0.15 * mid, 0, 0.6) * rng.uniform(0.3, 1.0))
            # (the veins stay green longest: the colour comes in between them)
            col = _mix(col, warm, np.clip((0.7 * tips + 0.5 * edge) * rng.uniform(0.5, 1.0) + 0.3 * big, 0, 0.95)
                       * (1 - 0.5 * halo))
    # texture: blotches and a fine grain; the finest veins round the blade's little islands
    col = col * (1 + (0.07 * mid + 0.03 * grain - 0.03 * f["net"])[..., None])
    lo, hi = pal["freckles"]
    if family in ("yellow", "red") and hi > 0:
        spot = (0.42, 0.25, 0.09) if family == "yellow" else (0.3, 0.06, 0.05)
        if kind == "almond":
            spot = (0.25, 0.1, 0.04)
        n = int(rng.integers(lo, hi + 1)) if family == "yellow" or rng.random() < 0.4 else 0
        col = _freckles(col, f, rng, spot, n)
    stem = np.clip(_ramp(pal[family if family in FAMILIES else "green"], np.array(0.85, np.float32)) * 0.72, 0, 1)
    # the veins: lighter on most leaves, darker on a bàng leaf turning red (its veins stay dark)
    k = pal["veins"]
    if k >= 0:
        veins = np.clip(col * 1.1 + np.array((0.2, 0.16, 0.08), np.float32), 0, 1)
    else:
        veins = col * (0.55 if family in ("red", "yellow") else 1.3)
        k = -k * (1.0 if family in ("red", "yellow") else 0.6)
    front = _mix(col, veins, np.clip(k * (main + 0.6 * side), 0, 1))
    # the back: duller and lighter, the veins standing out (a willow's: silvery; a ginkgo's: much like its front)
    grey = col.mean(axis=2, keepdims=True)
    if kind == "willow":
        back = _mix(col, (0.78, 0.82, 0.74), 0.5)
    elif kind == "ginkgo":
        back = np.clip(col * 1.03 + 0.02, 0, 1)
    else:
        back = np.clip((col + (grey - col) * 0.28) * 1.04 + 0.04, 0, 1)
    vein_back = np.clip(back * 0.75 + 0.3, 0, 1)
    back = _mix(back, vein_back, np.clip(0.5 * main + 0.25 * side + 0.1 * f["vein_wide"], 0, 1)
                * (0.4 if kind == "ginkgo" else 1.0))
    return np.clip(front, 0, 1), back, _lit(col, f, 1.0), stem


def _lit(col, f, veins):
    """A leaf lit from behind: the light through it, warmer and brighter, its veins (and the fine net between them)
    dark against it, the thin edge glowing."""
    lum = (col @ np.array([0.3, 0.55, 0.15], np.float32))[..., None]
    glow = np.clip(lum + (col - lum) * 1.3, 0, 1)  # (more saturated)
    glow = glow ** 0.72 * np.array((1.15, 1.03, 0.78), np.float32) + np.array((0.07, 0.04, 0.0), np.float32)
    dark = np.clip(veins * (0.38 * f["main"] + 0.28 * f["side"] + 0.1 * f["net"]), 0, 0.6)
    glow = glow * (1 - dark[..., None])
    glow = glow * (1 + 0.15 * np.exp(-f["inside"] / 0.01))[..., None]
    return np.clip(glow, 0, 1)


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


LIGHT = np.array([-0.45, 0.55, 0.7]) / np.linalg.norm([-0.45, 0.55, 0.7])  # (leaf units: y up)


def _shade(design, f, back):
    """Light and shade (H, W) from how the leaf is cupped, curled, twisted, folded along its midrib and sunk along
    its veins."""
    r, along = f["r"], f["along"]
    x, y = f["x"], f["y"]
    h = design["cup"] * 0.35 * r * r
    curl_at = along if design.get("curl_by") == "along" else np.clip(r / f["rmax"], 0, 1)
    h = h + design["curl"] * 0.25 * _smoothstep(0.3, 1.0, curl_at) ** 2
    h = h + design["twist"] * 0.25 * x * (y - 0.5)  # (a saddle: one corner up, the other down)
    h = h + design["vfold"] * np.sqrt(x * x + 0.0004)  # (the two halves raised a little either side of the midrib)
    h = h - design["fold"] * f["valley"]  # the blade a little sunk along the main veins
    h = h + design["wave"] * f["big"]
    if back:
        h = -h
    gy, gx = np.gradient(_blur(h, 0.012 * f["ppu"]), 1.0 / f["ppu"])
    gy = -gy  # rows run down, y up
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    n /= np.linalg.norm(n, axis=2, keepdims=True)
    lam = n @ LIGHT
    shade = np.clip(1 + 0.75 * (lam - LIGHT[2]), 0.55, 1.3)
    if not back:  # a faint waxy sheen on the top (a glossy bàng leaf: more)
        half = LIGHT + np.array([0.0, 0.0, 1.0])
        half /= np.linalg.norm(half)
        shine = 0.35 if design["shape"] == "almond" else 0.18
        shade = shade + shine * np.clip(n @ half, 0, 1) ** 40
    return _blur(shade.astype(np.float32), 0.004 * f["ppu"])


class Sprite:
    """One leaf design in one colour: front, back and lit from behind as premultiplied RGBA pictures (Pillow "RGBa")
    at the MIPS sizes up to `size`, all the same way round (the leaf turning over mirrors the back); `extent`: the
    leaf's longest side and `reach`: how far it reaches from the picture's middle, in the largest picture (px); `rgb`:
    its mean colour."""

    def __init__(self, front, back, lit, size, extent, reach, rgb):
        self.front, self.back, self.lit, self.size = front, back, lit, size
        self.extent, self.reach, self.rgb = extent, reach, rgb
        self.bytes = sum(img.size[0] * img.size[1] * 4 for img in front + back + lit)

    def level(self, size):
        """Which of the ready pictures: the smallest in which the leaf is still at least `size` px long."""
        for i in range(len(self.front) - 1, -1, -1):
            if self.extent * self.front[i].size[0] / self.size >= size:
                return i
        return 0

    def image(self, back, size, glow=0.0):
        """The picture of the leaf (its back if `back`) for a leaf `size` px long, lit from behind by `glow` (0..1)."""
        i = self.level(size)
        img = (self.back if back else self.front)[i]
        return Image.blend(img, self.lit[i], min(1.0, glow)) if glow > 0.01 else img


def _grid(n, cx, cy, ppu):
    xs = (cx + (np.arange(n) + 0.5 - n / 2) / ppu).astype(np.float32)
    ys = (cy - (np.arange(n) + 0.5 - n / 2) / ppu).astype(np.float32)
    return np.broadcast_to(xs[None, :], (n, n)), np.broadcast_to(ys[:, None], (n, n))


def design_of(kind, family, variant):
    """The design of leaf `variant` of shape `kind` in colour `family` (each its own: a red maple leaf is not a
    yellow one repainted), and the random draws that paint it."""
    rng = np.random.default_rng([SHAPES.index(kind), PALETTES.index(family), variant, 4111])
    design = DESIGNS[kind](rng)
    design["seed"] = int(rng.integers(1 << 30))
    return design, rng


def make_sprite(kind, family, variant, size=MIPS[0]):
    """The pictures of leaf design `variant` of shape `kind` in colour `family`, the largest `size` px across (one of
    MIPS: the same leaf at any of them). Its `extent` is its blade's longest side (its stem left out)."""
    design, rng = design_of(kind, family, variant)
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
    r = np.hypot(gx, gy).astype(np.float32)
    fields = {"alpha": alpha, "inside": inside, "r": r, "along": along, "x": gx, "y": gy,
              "rmax": float(r[alpha > 0.5].max()) if (alpha > 0.5).any() else 1.0,
              "valley": np.clip(_blur(main, 0.035 * ppu) * 2.5, 0, 1), "main": main, "side": side, "ppu": ppu,
              "halo": np.clip(_blur(main, 0.05 * ppu) * 3.0, 0, 1),
              "vein_wide": np.clip(_blur(np.maximum(main, side), 0.012 * ppu) * 1.5, 0, 1),
              "big": _noise(rng, alpha.shape, 3),
              "net": _net(np.random.default_rng([design["seed"], 9]), alpha.shape, 70, 0.05) * alpha}
    front, back, lit, stem_rgb = _paint(design, family, rng, fields)
    rim = np.exp(-inside / 0.008)[..., None]  # the thin edge: light comes through it
    if design["shape"] == "petal":  # a petal's shade: a deeper pink, never grey
        for side_, col in enumerate((front, back)):
            s = _shade(design, fields, bool(side_))
            s = np.clip(s * (alpha.sum() / max(float((s * alpha).sum()), 1e-6)), 0.88, 1.08)[..., None]  # (mean 1)
            col[:] = _mix(col, (0.85, 0.55, 0.65), np.clip((1 - s[..., 0]) / 0.12, 0, 1) * 0.5) * np.maximum(s, 1)
    else:
        front = front * _shade(design, fields, False)[..., None]
        back = back * _shade(design, fields, True)[..., None]
        front = front * (1 + 0.1 * rim)
        back = back * (1 + 0.06 * rim)
    # the stem, over the blade where they meet
    a = np.maximum(alpha, stem)
    over = (np.clip(stem - 0.6 * alpha, 0, 1) / np.maximum(a, 1e-6))[..., None]
    stem_col = stem_rgb * (0.85 + 0.15 * _noise(rng, alpha.shape, 6)[..., None])
    front = front * (1 - over) + stem_col * over
    back = back * (1 - over) + stem_col * 0.9 * over
    lit = lit * (1 - over) + stem_col * 0.5 * over
    if design["shape"] == "petal":
        a = a * 0.93  # a petal lets a little light through
    ys, xs = np.nonzero(alpha > 0.02)  # (the blade: its stem left out)
    extent = float(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1)
    ys, xs = np.nonzero(a > 0.02)
    reach = float(np.hypot(xs + 0.5 - n / 2, ys + 0.5 - n / 2).max()) + 1.5
    rgb = (front * a[..., None]).reshape(-1, 3).sum(axis=0) / max(float(a.sum()), 1e-6)
    return Sprite(_mips(front, a), _mips(back, a), _mips(lit, a), n, extent, reach, rgb)


def _mips(rgb, alpha):
    a = alpha[..., None]
    out = [Image.fromarray((np.dstack([np.clip(rgb, 0, 1) * a, a]) * 255 + 0.5).astype(np.uint8), "RGBa")]
    for size in MIPS:  # each from the one before: half its size
        if size < out[-1].size[0]:
            small = np.array(out[-1].resize((size, size), Image.Resampling.LANCZOS))
            np.minimum(small[..., :3], small[..., 3:], out=small[..., :3])  # (its ringing: never brighter than it
            out.append(Image.fromarray(small, "RGBa"))  # covers, no light rim)
    return out


_SPRITES = {}
MAX_BYTES = 96 << 20  # the pictures kept take at most this much memory


def sprite(kind, family, variant, size=MIPS[0]):
    """make_sprite(), kept (petals are pink whatever the colour); the least recently used go first when the
    pictures kept would take more than MAX_BYTES."""
    key = (kind, "pink" if kind == "petal" else family, variant, size)
    item = _SPRITES.pop(key, None)
    if item is None:
        item = make_sprite(kind, FAMILIES[0] if kind == "petal" else family, variant, size)
        while _SPRITES and sum(s.bytes for s in _SPRITES.values()) + item.bytes > MAX_BYTES:
            _SPRITES.pop(next(iter(_SPRITES)))
    _SPRITES[key] = item  # (the most recently used last)
    return item


def sprite_bytes(size):
    """The memory one leaf's pictures take, the largest `size` px across (front, back and lit, each with its smaller
    copies)."""
    return 3 * 4 * sum(s * s for s in MIPS if s <= size)
