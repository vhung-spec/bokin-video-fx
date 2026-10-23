"""Logs for a fireplace (the "Củi" settings of Lửa lò sưởi): a pile of round logs drawn by code as a camera sees
them, charred where the fire has reached them, lit by its flames, embers glowing in their cracks.

A log is a lumpy cylinder with rounded cut ends (Log.outside: < 0 inside it), in a small scene on the bed: x across
the box, y away from the camera, z up, lengths in box heights, the bed's middle on the floor at 0. The camera is far
off (straight rays) and looks down at the pile a little (`tilt`). A ray goes into the cylinder round each log, steps
along it to where it is first inside the log, and closes in on its surface; the nearest log's point is the one seen.
A pixel is one ray at its middle, or SS x SS rays where it is on a log's edge. What a log looks like at a point (bark,
or charred wood cracked in blocks round the log, ash; a cut end's rings; their relief tilting the way it faces) is
worked out there from the log's own random draws: the same seen at any size. That is done once for a place of the box
(Pile.lay), with how much light each of the fire's lights (fixed places in the pile and above it) gives each pixel:
facing it, not in another log's shadow (softly: the flames are big), less with distance; and how much the logs round
a point keep the room's light from it. Each frame (Pile.colour) only the lights' brightness changes (the flames near
each, the embers breathing) and the cracks' glow.
"""
import math

import numpy as np

TAU = 2 * math.pi
SS = 2  # rays a pixel each way on a log's edge (one ray inside a log or off it)
MARCH = 8  # steps along a ray through the cylinder round a log (then closing in on the log)
CLOSE = 6
GAMMA = 2.2  # (light is added in linear light, the picture is not)
SOFT = 0.45  # shadows' softness (a share of the shading log's radius each side of its edge)
TABLE = 256  # (a log's shape along it: looked up in a table this long)
BIGGEST = 0.1  # a log's radius at most (box heights: the flames need room above the pile)


def _lin(rgb):
    return (np.asarray(rgb, np.float32) / 255.0) ** GAMMA


# bark: its plates and the furrows between them (sRGB), the plates' size round a log (radii) and how many times
# longer they are along it, the furrows' width (a share), how deep the relief is (radii); birch: its colours only
WOODS = {  # (the kinds of wood)
    "oak": {"plate": (98, 90, 80), "furrow": (18, 15, 12), "width": 0.17, "length": 6.0, "gap": 0.04,
            "deep": 0.05},
    "brown": {"plate": (104, 70, 46), "furrow": (34, 21, 13), "width": 0.2, "length": 5.0, "gap": 0.02,
              "deep": 0.025},
    "birch": {"plate": (226, 221, 207), "furrow": (40, 37, 34)},
}
CHAR = _lin((22, 20, 19))  # charcoal (black patches)
SCORCH = _lin((46, 30, 20))  # (bark browned and darkened by the heat, next to where it has charred)
SHEEN = _lin((60, 54, 48))  # (charred wood coated with a little ash: dark grey-brown)
ASH = _lin((160, 156, 150))
CRACK = _lin((8, 7, 7))
EARLY, LATE = _lin((205, 162, 112)), _lin((150, 104, 64))  # a cut end's wood: spring and summer rings
HEART = _lin((158, 102, 60))  # (its heartwood, darker)
EMBER_I = np.array([0.0, 0.2, 0.45, 0.75, 1.1, 1.6], np.float32)  # embers' colours by how hot (sRGB)
EMBER = _lin([(0, 0, 0), (82, 12, 3), (190, 45, 8), (252, 105, 18), (255, 165, 50), (255, 222, 140)])
GLOW_CRACK, GLOW_SKIN = 2.4, 2.4  # how hot a burning log's cracks and the char between them glow (times "glow")
FLAME_LIGHT = np.array([1.0, 0.72, 0.45], np.float32)  # the flames' light on the logs (linear)
EMBER_LIGHT = np.array([1.0, 0.38, 0.1], np.float32)  # and the embers'


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _dsmooth(x):
    """The slope of _smooth at x."""
    x = np.clip(x, 0.0, 1.0)
    return 6.0 * x * (1.0 - x)


def _quintic(f):
    return f * f * f * (f * (f * 6.0 - 15.0) + 10.0)


def _smax(a, b, k):
    """The larger of a and b, the two blended where they are within k of each other (a rounded edge)."""
    e = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.maximum(a, b) + e * e * (k * 0.25)


def _noise(rng, a, q, size, extent, octaves=2, wrap=True):
    """Smooth value noise (0..1) at points a, q (radii): features about `size` radii, finer octaves after; round a
    log (`wrap`) q repeats every TAU (whole cells round it); `extent`: the largest a (and q when not wrapping). Its
    draws depend on these alone: a log looks the same made at any size."""
    total, weight = 0.0, 0.0
    for o in range(octaves):
        s = size / 2 ** o
        na = int(math.ceil(extent / s)) + 3
        nq = max(2, int(round(TAU / s))) if wrap else na
        table = rng.random((na, nq)).astype(np.float32)
        ga = a / s + 1.0
        gq = q * (nq / TAU) if wrap else q / s + 1.0
        ia, iq = np.floor(ga).astype(np.int32), np.floor(gq).astype(np.int32)
        fa, fq = _quintic(ga - ia), _quintic(gq - iq)
        ia = np.clip(ia, 0, na - 2)
        if wrap:
            q0, q1 = iq % nq, (iq + 1) % nq
        else:
            q0 = np.clip(iq, 0, nq - 2)
            q1 = q0 + 1
        v = (table[ia, q0] * (1 - fq) + table[ia, q1] * fq) * (1 - fa) + \
            (table[ia + 1, q0] * (1 - fq) + table[ia + 1, q1] * fq) * fa
        total = total + v * 0.5 ** o
        weight += 0.5 ** o
    return total / weight


def _noise_d(rng, a, q, size, extent, octaves=2):
    """_noise round a log, and its slopes along a and round q (per radius)."""
    total = da = dq = 0.0
    weight = 0.0
    for o in range(octaves):
        s = size / 2 ** o
        na = int(math.ceil(extent / s)) + 3
        nq = max(2, int(round(TAU / s)))
        table = rng.random((na, nq)).astype(np.float32)
        ga, gq = a / s + 1.0, q * (nq / TAU)
        ia, iq = np.floor(ga).astype(np.int32), np.floor(gq).astype(np.int32)
        ra, rq = ga - ia, gq - iq
        fa, fq = _quintic(ra), _quintic(rq)
        sa, sq = 30 * ra * ra * (ra - 1) ** 2, 30 * rq * rq * (rq - 1) ** 2  # (the fades' slopes)
        ia = np.clip(ia, 0, na - 2)
        q0, q1 = iq % nq, (iq + 1) % nq
        t00, t01, t10, t11 = table[ia, q0], table[ia, q1], table[ia + 1, q0], table[ia + 1, q1]
        lo, hi = t00 + (t01 - t00) * fq, t10 + (t11 - t10) * fq
        k = 0.5 ** o
        total = total + k * (lo + (hi - lo) * fa)
        da = da + k * (hi - lo) * sa / s
        dq = dq + k * ((t01 - t00) * (1 - fa) + (t11 - t10) * fa) * sq * (nq / TAU)
        weight += k
    return total / weight, da / weight, dq / weight


class Log:
    """One log: its axis from p0 to p1 (box heights), its radius r; a little thicker at one end, gently lumpy, its
    cut ends rounded; its looks drawn from its own seed."""

    def __init__(self, p0, p1, r, wood, rng):
        self.p0 = np.asarray(p0, np.float64)
        axis = np.asarray(p1, np.float64) - self.p0
        self.length = float(np.linalg.norm(axis))
        self.d = axis / self.length
        e1 = np.cross(self.d, (0.0, 0.0, 1.0))
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(self.d, (1.0, 0.0, 0.0))
        self.e1 = e1 / np.linalg.norm(e1)
        self.e2 = np.cross(self.d, self.e1)
        self.r = float(r)
        self.wood = wood
        # its shape along it (looked up: x 1 + taper + bulges, and how much it is oval or three-sided there)
        hs = np.linspace(-0.1, 1.1, TABLE) * self.length
        taper = rng.uniform(-0.12, 0.12) * (hs / self.length - 0.5)
        shape = [1 + taper]
        for amp in (0.02, 0.05, 0.05, 0.025, 0.025):
            k = rng.uniform(0.5, 2.0, 2) * TAU / self.length
            ph = rng.random(2) * TAU
            shape.append(amp * (0.6 * np.sin(k[0] * hs + ph[0]) + 0.4 * np.sin(k[1] * hs + ph[1])))
        self.shape = np.stack(shape, 1)  # (TABLE x 6: size, a bulge, oval (cos, sin), three-sided (cos, sin))
        self.shape[:, 0] += self.shape[:, 1]
        self.h0, self.dh = hs[0], hs[1] - hs[0]
        self.round = self.r * rng.uniform(0.28, 0.4)
        self.reach = self.r * float(np.abs(self.shape[:, 0]).max() + np.abs(self.shape[:, 2:]).sum(1).max()) * 1.02
        self.seed = int(rng.integers(1 << 31))

    def end(self):
        return self.p0 + self.d * self.length

    def local(self, p):
        """Points (... x 3) in the log's own frame: along its axis from p0, and the two ways across it."""
        w = p - self.p0
        return w @ self.d, w @ self.e1, w @ self.e2

    def outside(self, h, x1, x2):
        """How far outside the log points are (about: < 0 inside), given in its frame."""
        g = np.clip((h - self.h0) / self.dh, 0, TABLE - 1.001)
        i = g.astype(np.int32)
        f = (g - i)[..., None]
        t = self.shape[i] * (1 - f) + self.shape[i + 1] * f
        rho = np.sqrt(x1 * x1 + x2 * x2) + 1e-12
        c1, s1 = x1 / rho, x2 / rho
        c2, s2 = c1 * c1 - s1 * s1, 2 * c1 * s1
        c3, s3 = c2 * c1 - s2 * s1, s2 * c1 + c2 * s1
        radius = self.r * (t[..., 0] + t[..., 2] * c2 + t[..., 3] * s2 + t[..., 4] * c3 + t[..., 5] * s3)
        k = self.round
        return _smax(_smax(rho - radius, -h, k), h - self.length, k)

    def trace(self, o, f, before=None):
        """Where rays (origins o: N x 3, all one way f) first meet the log: the distance along each (inf: not, or
        not nearer than `before`, when given: what nearer logs hide is not traced)."""
        oh, o1, o2 = self.local(o)
        fh, f1, f2 = float(f @ self.d), float(f @ self.e1), float(f @ self.e2)
        a = f1 * f1 + f2 * f2
        out = np.full(len(o), np.inf)
        if a < 1e-12:
            return out
        b = o1 * f1 + o2 * f2
        c = o1 * o1 + o2 * o2 - self.reach ** 2
        disc = b * b - a * c
        hit = disc > 0
        root = np.sqrt(np.maximum(disc, 0.0))
        t0, t1 = (-b - root) / a, (-b + root) / a
        if abs(fh) > 1e-12:  # (and between its ends)
            s0, s1 = -oh / fh, (self.length - oh) / fh
            t0, t1 = np.maximum(t0, np.minimum(s0, s1)), np.minimum(t1, np.maximum(s0, s1))
        else:
            hit &= (oh >= 0) & (oh <= self.length)
        hit &= t1 > t0
        if before is not None:
            hit &= t0 < before
        idx = np.flatnonzero(hit)
        steps = np.arange(MARCH + 1) / MARCH
        for start in range(0, len(idx), 32768):
            i = idx[start:start + 32768]
            lo, span = t0[i], t1[i] - t0[i]
            ts = lo[:, None] + span[:, None] * steps[None, :]
            g = self.outside(oh[i, None] + fh * ts, o1[i, None] + f1 * ts, o2[i, None] + f2 * ts)
            inside = g < 0
            met = inside.any(axis=1)
            k = np.where(met, inside.argmax(axis=1), 1)
            rows = np.arange(len(i))
            ta, tb = ts[rows, k - 1], ts[rows, k]
            ga, gb = g[rows, k - 1], g[rows, k]
            for _ in range(CLOSE):  # (closing in: where the line through the two crosses, kept between them)
                tm = ta + (tb - ta) * np.clip(ga / np.maximum(ga - gb, 1e-12), 0.1, 0.9)
                gm = self.outside(oh[i] + fh * tm, o1[i] + f1 * tm, o2[i] + f2 * tm)
                ins = gm < 0
                tb, gb = np.where(ins, tm, tb), np.where(ins, gm, gb)
                ta, ga = np.where(ins, ta, tm), np.where(ins, ga, gm)
            out[i[met]] = tb[met]
        return out

    def normal(self, h, x1, x2):
        """Which way the surface faces at points on it (in the world): the slope of `outside` there."""
        e = self.r * 0.01
        gh = self.outside(h + e, x1, x2) - self.outside(h - e, x1, x2)
        g1 = self.outside(h, x1 + e, x2) - self.outside(h, x1 - e, x2)
        g2 = self.outside(h, x1, x2 + e) - self.outside(h, x1, x2 - e)
        n = gh[:, None] * self.d + g1[:, None] * self.e1 + g2[:, None] * self.e2
        return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    # --- what it looks like ---------------------------------------------------------------------------------------

    def bark(self, a, q, step):
        """Its bark at points a (along, radii), q (round, radians = radii of arc): colour (linear), relief slopes
        along a and round q (per radius).

        Furrows along the trunk that part and meet again (where a noise stretched along it is half way: the lines
        of a map's contours), plates between them, broken across here and there; birch: white, dark lenticels
        round the trunk, black scars."""
        w = WOODS[self.wood]
        extent = self.length / self.r + 1
        rng = np.random.default_rng([self.seed, 1])
        if self.wood == "birch":
            fibre = _noise(rng, a * 0.2, q, 0.04, extent, 2)
            dash = _noise(rng, a * 6.0, q * 0.4, 0.1, extent * 6.0, 1)
            spread = _noise(rng, a * 0.5, q, 0.3, extent, 1)
            lent = _smooth((dash - 0.9 + 0.12 * spread) / 0.03)  # (short dashes round the trunk)
            scar = _smooth((_noise(rng, a * 0.7, q, 0.6, extent, 3) - 0.7) / 0.04)
            peel = _smooth((_noise(rng, a, q, 0.4, extent, 2) - 0.72) / 0.06)
            ink = np.maximum(lent, scar * 0.95)
            plate = _lin(w["plate"]) * (0.86 + 0.24 * fibre[..., None])
            plate = plate * (1 - 0.5 * peel[..., None]) + _lin((190, 150, 120)) * (0.5 * peel[..., None])
            colour = plate * (1 - ink[..., None]) + _lin(w["furrow"]) * ink[..., None]
            flat = np.zeros_like(lent)  # (smooth bark)
            return colour, flat, flat
        stretch = 1.0 / w["length"]
        n, dna, dnq = _noise_d(rng, a * stretch, q, w["width"], extent * stretch, 3)
        ridge = np.abs(2 * n - 1)
        sign = np.sign(2 * n - 1)
        gap = w["gap"] * 4.0
        furrow = 1 - _smooth((ridge - 0.01) / gap)
        top = _smooth(ridge / (3.0 * gap))
        k = w["deep"] * 2.0 * (_dsmooth(ridge / (3.0 * gap)) / (3.0 * gap))
        sa = k * sign * dna * stretch
        sq = k * sign * dnq
        # broken across here and there
        brk = _noise(rng, a, q * 0.3, 0.45, extent, 1)
        across = (1 - _smooth(np.abs(2 * brk - 1) / 0.04)) * _smooth((ridge - gap) / gap) * 0.7
        fibre = _noise(rng, a * 0.15, q, 0.035, extent, 2)
        light = (0.62 + 0.55 * fibre) * (0.7 + 0.5 * top) * (0.85 + 0.3 * _noise(rng, a, q, 0.6, extent, 1))
        colour = _lin(w["plate"]) * light[..., None]
        dark = np.maximum(furrow, across)
        colour = colour * (1 - dark[..., None]) + _lin(w["furrow"]) * dark[..., None]
        return colour, sa, sq

    def char(self, a, q, step):
        """Its charred wood at points a, q: colour (linear), relief slopes (along, round), cracks (0..1: where embers
        glow), its block's random number, and noises for the ragged edge of the charred part and for its ash.

        As wood chars it shrinks across the grain: it cracks round the log every third of a radius or so (wavy,
        here and there not), and along it between those (each band its own places): blocks wider round the log
        than long; finer cracks wandering in them here and there. Dark grey where a little ash coats it, black in
        patches, streaked along the grain."""
        extent = self.length / self.r + 1
        rng = np.random.default_rng([self.seed, 2])
        spacing = 0.3
        nb = int(math.ceil(extent / spacing * 1.3)) + 6
        cuts = 14
        band = rng.random((nb, 3))
        gaps = rng.uniform(0.35, 1.0, (nb, cuts))
        where = np.cumsum(gaps, axis=1)
        where = where / where[:, -1:] * TAU  # (each band's cuts round the log, the last at TAU: the first again)
        where -= rng.random((nb, 1)) * TAU
        cell = rng.random((nb, cuts)).astype(np.float32)
        wave = _noise(rng, a, q, 0.55, extent, 2) - 0.5
        v = a / spacing * (0.85 + 0.3 * _noise(rng, a, q, 2.5, extent, 1)) + 0.9 * wave + 2.0
        b = np.floor(v)
        fv = v - b
        b = np.clip(b.astype(np.int32), 1, nb - 1)
        qd = q + 0.35 * (_noise(rng, a, q, 0.5, extent, 2) - 0.5)

        def split_of(k):  # (in band k: which block round the log, and how far from the cuts each side)
            pos = np.mod(qd[:, None] - where[k], TAU)
            lo = pos.min(axis=1)  # (from the cut before)
            i = pos.argmin(axis=1)
            hi = np.mod(where[k, (i + 1) % cuts] - where[k, i], TAU) - lo  # (to the cut after)
            hi = np.where(hi <= 0, hi + TAU, hi)
            return i, lo, hi
        s, lo, hi = split_of(b)
        s0 = split_of(b - 1)[0]
        up = np.where(cell[b, s] < 0.12, 9.0, (1 - fv) * spacing)  # (here and there no crack round it)
        down = np.where(cell[b - 1, s0] < 0.12, 9.0, fv * spacing)
        ea = np.minimum(up, down)
        eq = np.where(cell[b, s] > 0.85, 9.0, np.minimum(lo, hi) * 0.7)
        across = ea < eq
        edge = np.where(across, ea, eq)
        sa = np.where(across, np.where(down < up, 1.0, -1.0), 0.0)
        sq = np.where(across, 0.0, np.where(lo < hi, 1.0, -1.0) * 0.7)
        wide = (0.6 + 0.8 * band[b, 1]) * (0.55 + 0.9 * _noise(rng, a, q, 0.3, extent, 1))
        cw = np.maximum(0.022 * wide, 1.1 * step)
        x = (edge - 0.008 * wide) / cw
        crack = (1 - _smooth(x)) * np.minimum(1.0, 0.03 * wide / cw + 0.3)
        groove = 0.018 * _dsmooth(x) / cw
        # finer cracks wandering in the blocks, here and there
        fn, fda, fdq = _noise_d(rng, a, q * 0.8, 0.2, extent, 1)
        ridge = np.abs(2 * fn - 1)
        fw = max(0.03, 2.0 * step)
        craze = (1 - _smooth(ridge / fw)) * _smooth((_noise(rng, a, q, 0.6, extent, 1) - 0.45) / 0.15) * 0.75
        fibre, rda, rdq = _noise_d(rng, a * 0.2, q, 0.03, extent * 0.2, 2)  # (streaks along the grain)
        sa = groove * sa + 0.006 * 0.2 * rda  # (the grooves, the streaks; the finer cracks round it)
        sq = groove * sq + 0.006 * rdq + 0.004 * 0.8 * np.sign(2 * fn - 1) * fdq * craze
        tone = cell[b, s]
        black = rng.random((nb, cuts)).astype(np.float32)
        soot = _smooth((_noise(rng, a, q, 1.6, extent, 2) - 0.6 + 0.14 * (black[b, s] - 0.5)) / 0.02)  # (black
        # patches, their edges along the cracks)
        shade = (0.78 + 0.44 * tone) * (0.72 + 0.56 * fibre)
        grey = SHEEN * shade[..., None]
        colour = grey * (1 - soot[..., None]) + CHAR * (0.75 + 0.5 * fibre[..., None]) * soot[..., None]
        dark = np.maximum(crack, craze)
        colour = colour * (1 - dark[..., None]) + CRACK * dark[..., None]
        return (colour, sa, sq, np.maximum(crack, 0.6 * craze), tone, _noise(rng, a, q, 2.2, extent, 2),
                _noise(rng, a, q, 0.35, extent, 2))

    def face(self, u, v, burnt, step):
        """A cut end at points u, v (-1..1 across it, radii): colour (linear), relief (radii), cracks."""
        rng = np.random.default_rng([self.seed, 3])
        rho = np.sqrt(u * u + v * v)
        wob = _noise(rng, u + 1, v + 1, 0.5, 2.0, 3, wrap=False) - 0.5
        centre = rng.uniform(-0.15, 0.15, 2)  # (the pith is rarely in the middle)
        rr = np.sqrt((u - centre[0]) ** 2 + (v - centre[1]) ** 2) * (1 + 0.12 * wob)
        rings = rng.uniform(9, 16)
        frac = rr * rings
        frac -= np.floor(frac)
        late = _smooth((frac - 0.72) / 0.12) * (1 - _smooth((frac - 0.94) / 0.05))
        late = late * np.maximum(1 - 2.5 * step * rings, 0.2)  # (rings finer than the pixels: their mean)
        wood = EARLY * (1 - late[..., None]) + LATE * late[..., None]
        heart = 1 - _smooth((rr - 0.55) / 0.2)
        wood = wood * (1 - 0.35 * heart[..., None]) + HEART * (0.35 * heart[..., None])
        grain = _noise(rng, u + 1, v + 1, 0.15, 2.0, 2, wrap=False)
        wood = wood * (0.85 + 0.3 * grain[..., None])
        # radial checks: cracks from near the middle outward
        phi = np.arctan2(v - centre[1], u - centre[0])
        checks = np.zeros_like(rho)
        for _ in range(2 + int(rng.integers(0, 4))):
            at = rng.random() * TAU
            dphi = np.abs(np.angle(np.exp(1j * (phi - at - 0.25 * wob))))
            width = (0.012 + 0.025 * rr) / np.maximum(rr, 0.05) + step
            start = 0.15 + 0.55 * rng.random()
            checks = np.maximum(checks, (1 - _smooth(dphi / width)) * _smooth((rr - start) / 0.1))
        checks = checks * (0.4 + 0.6 * burnt)
        rim = WOODS[self.wood]["plate" if self.wood == "birch" else "furrow"]
        bark = _smooth((rho - 0.9) / 0.04)
        colour = wood * (1 - bark[..., None]) + _lin(rim) * 0.8 * bark[..., None]
        colour = colour * (1 - checks[..., None]) + CRACK * checks[..., None]
        # blackened from the rim in, as far as it has burnt
        black = _smooth((rho - (1.02 - 0.9 * burnt) + 0.15 * wob) / 0.18) * np.minimum(1.0, 0.3 + burnt)
        colour = colour * (1 - black[..., None]) + CHAR * (0.8 + 0.4 * grain[..., None]) * black[..., None]
        relief = 0.01 * grain - 0.02 * checks - 0.02 * bark
        return colour, relief, np.maximum(checks * black, checks * 0.3)


class Pile:
    """The logs of one fire: laid out once from its settings; traced for a place of the box (lay), lit each frame
    (colour)."""

    def __init__(self, wood, way, size, burnt, width, tilt, seed, half=None, floor=None):
        """`width`: the bed's (box heights); the pile kept within `half` box heights each side of the bed's middle
        and above `floor` (box heights up from it on the picture: the box's bottom edge), moved back if need be."""
        rng = np.random.default_rng([seed, 9091])
        self.width = float(width)
        self.r = min(0.085 * self.width * (0.6 + 0.8 * size), BIGGEST)  # (a log's radius)
        self.burnt = float(burnt)
        self.tilt = float(tilt)
        laid = [(np.array(p0, np.float64), np.array(p1, np.float64), r) for p0, p1, r in self._layout(way, rng)]
        if half is not None:  # (narrower if it would stick out of the box at its sides)
            most = max(max(abs(p0[0]), abs(p1[0])) + 1.15 * r for p0, p1, r in laid)
            if most > half:
                k = (half - 1.15 * self.r) / max(most - 1.15 * self.r, 1e-9)
                for p0, p1, _ in laid:
                    p0[0] *= k
                    p1[0] *= k
        if floor is not None:  # (further back if its front would be under the box's bottom edge)
            up = np.array([0.0, math.sin(self.tilt), math.cos(self.tilt)])  # (up the picture, in the scene)
            low = min(self._lowest(p0, p1, r, up) for p0, p1, r in laid)
            if low < floor and up[1] > 1e-3:
                for p0, p1, _ in laid:
                    p0[1] += (floor - low) / up[1]
                    p1[1] += (floor - low) / up[1]
        self.logs = [Log(p0, p1, r, wood, rng) for p0, p1, r in laid]
        self.top = max(max(lg.p0[2], lg.end()[2]) + lg.reach for lg in self.logs)
        ys = [y for lg in self.logs for y in (lg.p0[1], lg.end()[1])]
        self.fire = np.array([0.0, (min(ys) + max(ys)) / 2, 0.4 * self.top])  # (the fire's middle, among the logs)
        self.spread = 0.22 * self.width  # (its heart: a line across the pile this far each side of its middle)
        self.lights = self._lights()

    @staticmethod
    def _lowest(p0, p1, r, up):
        """How far up the picture (`up`, in the scene) a log's lowest point is: its axis from p0 to p1, its radius r
        (and a little more, for its lumps)."""
        d = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-9)
        return min(p0 @ up, p1 @ up) - 1.1 * r * math.sqrt(max(0.0, 1 - float(d @ up) ** 2))

    def _layout(self, way, rng):
        """(start, end, radius) of each log on the floor round the bed's middle (x across, y away, z up)."""
        r, w = self.r, self.width

        def j(s):
            return rng.uniform(-s, s)

        def lying(cx, cy, half, yaw, rad, z=None, pitch=0.0):
            dx, dy = math.cos(yaw) * half, math.sin(yaw) * half
            z = rad if z is None else z
            return (cx - dx, cy - dy, z - pitch * half), (cx + dx, cy + dy, z + pitch * half), rad

        back = lying(j(0.04) * w, 1.02 * r, w * (0.43 + j(0.03)), j(0.07), r * (1.0 + j(0.06)))
        front = lying(j(0.05) * w, -1.02 * r, w * (0.41 + j(0.03)), j(0.07), r * (0.97 + j(0.06)))
        if way == "two":
            return [back, front]
        lift = r + math.sqrt(4.0 - 1.02 ** 2) * r
        top = lying(j(0.06) * w, j(0.25) * r, w * (0.37 + j(0.03)), math.radians(12 + j(8)) * (1 if j(1) > 0 else -1),
                    r * (0.92 + j(0.06)), lift + 0.05 * r)
        if way == "stack":
            return [back, front, top]
        # a heap: two more, smaller, leaning across the stack at the sides
        side = 1 if j(1) > 0 else -1
        lean = lying(side * 0.3 * w, -2.2 * r, w * 0.22, math.radians(60 + j(10)) * side, r * 0.75, 1.6 * r, 0.5)
        small = lying(-side * 0.25 * w, 2.4 * r, w * 0.25, math.radians(-20 + j(10)) * side, r * 0.7, 2.1 * r, -0.3)
        return [small, back, front, top, lean]

    def _lights(self):
        """The fire's lights on the logs: (place, kind): "flame" above the pile and among the logs (as bright as the
        flames near each), "ember" on the floor in front of the pile (as bright as the embers)."""
        w, r, top, y = self.width, self.r, self.top, self.fire[1]
        xs = [x for lg in self.logs for x in (lg.p0[0], lg.end()[0])]
        reach = min(0.36 * w, 0.8 * max(abs(min(xs)), abs(max(xs))))
        out = [((x, y, top + 0.8 * r), "flame") for x in np.linspace(-1, 1, 4) * reach]
        out += [((x, y, 0.95 * r), "flame") for x in np.linspace(-0.8, 0.8, 3) * reach]
        front = min(min(lg.p0[1], lg.end()[1]) for lg in self.logs) - 0.9 * r
        out += [((x, front, 0.12 * r), "ember") for x in np.linspace(-0.7, 0.7, 2) * reach]
        return [(np.array(p, np.float64), kind) for p, kind in out]

    # --- where it is on the picture --------------------------------------------------------------------------------

    def screen(self, p):
        """Points (... x 3) on the picture: box heights right of and up from the bed's middle."""
        p = np.asarray(p, np.float64)
        return p[..., 0], p[..., 2] * math.cos(self.tilt) + p[..., 1] * math.sin(self.tilt)

    def extent(self):
        """(left, right, bottom, top) of the pile on the picture (box heights from the bed's middle, up), with room
        for its shadow on the floor round it."""
        corners = np.array([(dx, dy, dz) for dx in (-1, 1) for dy in (-1, 1) for dz in (-1, 1)], np.float64)
        pts = np.concatenate([p + corners * lg.reach for lg in self.logs for p in (lg.p0, lg.end())])
        x, s = self.screen(pts)
        m = 1.5 * self.r
        return x.min() - m, x.max() + m, s.min() - m * math.sin(self.tilt) - 0.5 * self.r, s.max() + 0.1 * self.r

    # --- traced once for a place of the box -----------------------------------------------------------------------

    def _rays(self, X, S):
        """Rays from the picture's points X (right) and S (up) box heights from the bed's middle: their starts, on
        the plane through it that faces the camera, and their one way (into the picture, a little down)."""
        sa, ca = math.sin(self.tilt), math.cos(self.tilt)
        return np.stack([X, S * sa, S * ca], -1), np.array([0.0, ca, -sa])

    def _trace(self, o, f):
        """Each log's distance along each ray (logs x rays; inf: missed, or hidden by a nearer log), the nearer logs
        first."""
        out = np.full((len(self.logs), len(o)), np.inf)
        best = np.full(len(o), np.inf)
        order = sorted(range(len(self.logs)), key=lambda k: min(self.logs[k].p0 @ f, self.logs[k].end() @ f))
        for k in order:
            out[k] = self.logs[k].trace(o, f, best)
            np.minimum(best, out[k], out=best)
        return out

    def lay(self, xs, ss, px):
        """What does not change from frame to frame over pixels whose middles are at xs (box heights right of the
        bed's middle, one per column) and ss (up from it, one per row, top row first), `px` box heights a pixel: how
        much a log covers each, its colour, how much of each light reaches it, the room's light it gets, its cracks'
        glow, whether it hides the flames (in front of the fire), and the logs' shadow on the floor."""
        h, w = len(ss), len(xs)
        X = np.broadcast_to(xs[None, :], (h, w)).reshape(-1)
        S = np.broadcast_to(ss[:, None], (h, w)).reshape(-1)
        o, f = self._rays(X, S)
        dist = self._trace(o, f)
        which = np.where(np.isfinite(dist.min(axis=0)), dist.argmin(axis=0), -1).reshape(h, w)
        # a log's edge (another log, or none, next to it): SS x SS rays there
        pad = np.pad(which, 1, mode="edge")
        edge = np.zeros((h, w), bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                edge |= pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] != which
        ei = np.flatnonzero(edge)
        sub = (np.arange(SS) + 0.5) / SS - 0.5
        ox = np.broadcast_to(X[ei, None, None] + sub[None, None, :] * px, (len(ei), SS, SS)).reshape(-1)
        oy = np.broadcast_to(S[ei, None, None] - sub[None, :, None] * px, (len(ei), SS, SS)).reshape(-1)
        eo, _ = self._rays(ox, oy)
        edist = self._trace(eo, f)
        inner = np.flatnonzero(~edge.reshape(-1) & (which.reshape(-1) >= 0))
        rays = np.concatenate([o[inner], eo])
        rdist = np.concatenate([dist[:, inner], edist], axis=1)
        pixel = np.concatenate([inner, np.repeat(ei, SS * SS)])
        weight = np.concatenate([np.ones(len(inner)), np.full(len(eo), 1.0 / (SS * SS))])
        near = rdist.argmin(axis=0)
        seen = np.isfinite(rdist.min(axis=0))
        n = len(rays)
        albedo = np.zeros((n, 3), np.float32)
        normal = np.zeros((n, 3))
        pos = np.zeros((n, 3))
        crack = np.zeros(n, np.float32)
        skin = np.zeros(n, np.float32)
        front = np.zeros(n, np.float32)
        for k, lg in enumerate(self.logs):
            i = np.flatnonzero(seen & (near == k))
            if not len(i):
                continue
            p = rays[i] + rdist[k, i, None] * f
            hh, x1, x2 = lg.local(p)
            nrm = lg.normal(hh, x1, x2)
            albedo[i], normal[i], crack[i], skin[i] = self._looks(lg, p, hh, x1, x2, nrm, px / lg.r)
            pos[i] = p
            front[i] = _smooth((self.fire[1] + 1.5 * lg.r - p[:, 1]) / (0.8 * lg.r) + 0.5)  # (none of the flames
            # over it unless it is well behind the fire's middle)
        wt = weight * seen
        size = h * w

        def pixels(a):  # the rays of each pixel to the pixel (their sum, by weight)
            if a.ndim == 1:
                return np.bincount(pixel, a * wt, size).reshape(h, w)
            return np.stack([np.bincount(pixel, a[:, c] * wt, size) for c in range(a.shape[1])], 1).reshape(h, w, -1)
        alpha = np.bincount(pixel, wt, size).reshape(h, w)
        some = np.maximum(alpha, 1e-9)
        out = {"alpha": alpha.astype(np.float32),
               "albedo": (pixels(albedo) / some[..., None]).astype(np.float32),
               "crack": (pixels(crack) / some).astype(np.float32),
               "skin": (pixels(skin) / some).astype(np.float32),
               "front": (pixels(front) / some).astype(np.float32)}
        nrm = pixels(normal)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-9)
        p = pixels(pos) / some[..., None]
        on = alpha > 0
        out["light"], out["room"] = self._shading(p[on], nrm[on], on)
        out["phase"] = self._phase(p[on], on)
        out["floor"] = self._floor(xs, ss)
        return out

    def _heat(self, lg, p, nrm, ragged):
        """How far the fire has burnt points on a log (about 0: not reached .. 1.5 and more: burning red): near the
        fire's heart (a line across the pile, among the logs: its middle part burns, its ends do not), facing it,
        between two logs (the fire burns in the gaps) and low down (on the embers), unevenly (`ragged`: a noise of
        the log)."""
        fire, r = self.fire, self.r
        q = np.stack([np.clip(p[:, 0], fire[0] - self.spread, fire[0] + self.spread), np.full(len(p), fire[1]),
                      np.full(len(p), fire[2])], 1)
        to = q - p
        d = np.linalg.norm(to, axis=1)
        facing = np.einsum("ij,ij->i", nrm, to) / np.maximum(d, 1e-9)
        core = np.exp(-(d / (2.4 * r)) ** 2)
        wide = np.exp(-(d / (3.5 * r)) ** 2)
        gap = np.full(len(p), np.inf)
        for other in self.logs:
            if other is not lg:
                gap = np.minimum(gap, np.linalg.norm(p - self._nearest_on(other, p), axis=1) - other.r)
        between = np.exp(-(np.maximum(gap, 0.0) / (0.7 * r)) ** 2)
        low = np.exp(-(np.maximum(p[:, 2], 0.0) / (0.45 * r)) ** 2)
        side = np.exp(-(np.maximum(np.abs(p[:, 0] - fire[0]) - self.spread, 0.0) / (1.3 * r)) ** 2)  # (its ends: cool)
        return (core * (0.7 + 0.3 * facing) + 0.7 * between * wide + 0.9 * low * wide) * (0.4 + 0.6 * side) + \
            0.4 * (ragged - 0.5)

    def _looks(self, lg, p, hh, x1, x2, nrm, step):
        """What a log looks like at points on it: colour (linear), the way it faces with its relief, and how much its
        cracks and the char between them glow (where it is burning). `step`: a pixel's size, radii.

        Where the fire has not reached it, its bark (browned next to where it has charred); where it has, charred
        wood, ash on its tops; where it is hottest (`_heat`), burning: its cracks and its char glowing."""
        a, q = hh / lg.r, np.mod(np.arctan2(x2, x1), TAU)
        along = nrm @ lg.d  # (how much of it is a cut end, by the way it faces: the rounded edge between)
        cap = _smooth((np.abs(along) - 0.55) / 0.3)
        coal, c_sa, c_sq, c_crack, tone, ragged, ash = lg.char(a, q, step)
        burnt = self.burnt
        heat = self._heat(lg, p, nrm, ragged)
        edge = heat - (0.95 - 0.75 * burnt) - 0.15 * (tone - 0.5)  # (charred block by block: its edge along cracks)
        charred = _smooth(edge / 0.05)
        scorched = _smooth((edge + 0.3) / 0.3) * (1 - charred)
        burning = _smooth((edge - 0.22) / 0.35) * charred
        colour = coal * charred[:, None]
        sa, sq = c_sa * charred, c_sq * charred
        some_bark = np.flatnonzero(charred < 0.999)
        if len(some_bark):
            bark, b_sa, b_sq = lg.bark(a[some_bark], q[some_bark], step)
            sc = scorched[some_bark, None]
            bark = bark * (1 - 0.55 * sc) + SCORCH * (0.55 * sc)
            keep = (1 - charred[some_bark])
            colour[some_bark] += bark * keep[:, None]
            sa[some_bark] += b_sa * keep
            sq[some_bark] += b_sq * keep
        ashy = (0.2 + 0.8 * ash) * _smooth((nrm[:, 2] + 0.1) / 0.9) * charred * (0.1 + 0.22 * burnt) * (1 - burning)
        colour = colour * (1 - ashy[:, None]) + ASH * (0.75 + 0.5 * ragged[:, None]) * ashy[:, None]
        spots = _smooth((0.75 - ash) / 0.5)  # (some patches hotter than others; none under ash)
        crack = c_crack * burning * (0.55 + 0.45 * spots)
        skin = burning * (0.3 + 0.7 * (1 - tone) ** 1.5) * (0.35 + 0.65 * spots) * (1 - 0.5 * ashy)
        # the relief: the surface tilted by its slopes along the log and round it
        tang = np.cross(lg.d, nrm)
        tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-9)
        bump = sa[:, None] * lg.d + sq[:, None] * tang
        # the cut ends (seen from outside each: u mirrored at the far end), burnt as far as the fire has reached them
        ci = np.flatnonzero(cap > 0.001)
        if len(ci):
            flip = np.where(along[ci] < 0, -1.0, 1.0)
            u, v = x1[ci] / lg.r * flip, x2[ci] / lg.r
            d = 0.03
            done = np.clip(edge[ci] / 0.4 + 0.5, 0.0, 1.0)
            wood, rel, f_crack = lg.face(u, v, done, step)
            ru, rv = lg.face(u + d, v, done, step)[1], lg.face(u, v + d, done, step)[1]
            c = cap[ci, None]
            colour[ci] = colour[ci] * (1 - c) + wood * c
            crack[ci] = crack[ci] * (1 - cap[ci]) + f_crack * cap[ci] * burning[ci]
            bump_end = ((ru - rel) / d * flip)[:, None] * lg.e1 + ((rv - rel) / d)[:, None] * lg.e2
            bump[ci] = bump[ci] * (1 - c) + bump_end * c
        bump -= np.einsum("ij,ij->i", bump, nrm)[:, None] * nrm  # (in the surface)
        nb = nrm - bump
        nb /= np.maximum(np.linalg.norm(nb, axis=1, keepdims=True), 1e-9)
        return colour.astype(np.float32), nb, crack.astype(np.float32), skin.astype(np.float32)

    def _shading(self, p, nrm, on):
        """For pixels on the logs (points p, facing nrm): how much of each light reaches each (lights x pixels:
        facing it, not in another log's shadow, less far off), and the room's light (less where logs round it hide
        the room)."""
        lights = np.zeros((len(self.lights),) + on.shape, np.float32)
        p32, n32 = p.astype(np.float32), nrm.astype(np.float32)
        start = p32 + n32 * np.float32(0.02 * self.r)
        own = [np.linalg.norm(p - self._nearest_on(lg, p), axis=1) < lg.reach * 1.05 for lg in self.logs]
        for k, (at, _) in enumerate(self.lights):
            to = at.astype(np.float32) - p32
            dist = np.sqrt(np.einsum("ij,ij->i", to, to))
            facing = np.maximum(np.einsum("ij,ij->i", n32, to) / np.maximum(dist, 1e-9), 0.0)
            seen = np.ones(len(p), np.float32)
            for lg, mine in zip(self.logs, own):
                seen *= self._unshaded(start, at, lg, mine)
            fall = 1.0 / (1.0 + (dist * np.float32(1 / (2.5 * self.r))) ** 2)
            lights[k][on] = facing * seen * fall
        room = np.ones(len(p))
        for lg in self.logs:  # (each log hides some of the room from what is near it: as a ball would, about)
            v = self._nearest_on(lg, p) - p
            dd = np.linalg.norm(v, axis=1)
            cos = np.clip(np.einsum("ij,ij->i", nrm, v) / np.maximum(dd, 1e-9), 0, 1)
            room *= 1 - np.clip(cos * (lg.r / np.maximum(dd, lg.r)) ** 2 * 1.3, 0, 0.9)
        room *= 0.55 + 0.45 * _smooth((nrm[:, 2] + 0.6) / 1.2)  # (facing down: the dark floor)
        full = np.zeros(on.shape, np.float32)
        full[on] = room
        return lights, full

    @staticmethod
    def _nearest_on(lg, p):
        """The nearest points to p on a log's axis."""
        t = np.clip((p - lg.p0) @ lg.d, 0, lg.length)
        return lg.p0 + t[:, None] * lg.d

    @staticmethod
    def _unshaded(p, at, lg, own):
        """How much of a light at `at` reaches points p (float32) past one log (0 in its shadow, softly at its edge):
        from how near the line from each point to the light passes the log's axis; none of it is kept from the log's
        own surface (`own`: whether it faces the light says that)."""
        a1 = at.astype(np.float32)[None, :] - p  # (the nearest points of two segments: p -> at, and the log's axis)
        b1 = (lg.d * lg.length).astype(np.float32)
        r = p - lg.p0.astype(np.float32)
        aa = np.einsum("ij,ij->i", a1, a1)
        bb = float(b1 @ b1)
        ab = a1 @ b1
        ar = np.einsum("ij,ij->i", a1, r)
        br = r @ b1
        den = aa * bb - ab * ab
        s = np.where(den > 1e-12, np.clip((ab * br - bb * ar) / np.maximum(den, 1e-12), 0, 1), 0.0)
        t = np.clip((ab * s + br) / bb, 0, 1)
        s = np.clip((ab * t - ar) / np.maximum(aa, 1e-12), 0, 1)
        g = r + a1 * s[:, None] - b1 * t[:, None]
        gap = np.sqrt(np.einsum("ij,ij->i", g, g))
        out = _smooth((gap - lg.r * (1 - SOFT)) * (1 / (2 * SOFT * lg.r)))
        out[own] = 1.0
        return out

    def _phase(self, p, on):
        """Each pixel's moment in the embers' breathing (0..1: near ones together, far ones apart)."""
        rng = np.random.default_rng([self.logs[0].seed, 4])
        k = 1.0 / (0.9 * self.r)
        out = np.zeros(on.shape, np.float32)
        out[on] = _noise(rng, p[:, 0] * k + 20.0, (p[:, 2] + 0.5 * p[:, 1]) * k + 20.0, 1.0, 60.0, 2, wrap=False)
        return out

    def _floor(self, xs, ss):
        """The logs' shadow on the floor (0..1: how much of the room's and the fire's light they keep from it) at
        the pixels' points on the floor."""
        x = np.broadcast_to(xs[None, :], (len(ss), len(xs))).reshape(-1)
        y = np.broadcast_to(ss[:, None] / math.sin(self.tilt), (len(ss), len(xs))).reshape(-1)
        p = np.stack([x, y, np.zeros_like(x)], -1)
        occ = np.zeros(len(p))
        for lg in self.logs:
            v = self._nearest_on(lg, p) - p
            dd = np.maximum(np.linalg.norm(v, axis=1), 1e-9)
            occ += np.clip(v[:, 2] / dd, 0, 1) * (lg.r / np.maximum(dd, lg.r * 0.9)) ** 2 * 1.4
        return np.clip(occ, 0, 1).reshape(len(ss), len(xs)).astype(np.float32)

    # --- each frame ------------------------------------------------------------------------------------------------

    def colour(self, laid, room, flames, embers, glow, tau, beats):
        """The logs now (linear RGB, h x w x 3, laid over the picture by laid["alpha"]) and the light their burning
        wood gives off (linear RGB, or None): lit by the room (`room`: its light, linear RGB), each light as bright as
        `flames` / `embers` say (one number each, in the order of self.lights of that kind), where they burn glowing
        `glow` bright (breathing at `beats`: whole cycles a loop, at loop phase tau)."""
        lights = laid["light"]
        fl = np.zeros(lights.shape[1:], np.float32)
        em = np.zeros(lights.shape[1:], np.float32)
        flames, embers = iter(flames), iter(embers)
        for k, (_, kind) in enumerate(self.lights):
            if kind == "flame":
                fl += lights[k] * np.float32(next(flames))
            else:
                em += lights[k] * np.float32(next(embers))
        light = laid["room"][..., None] * np.asarray(room, np.float32) + fl[..., None] * FLAME_LIGHT + \
            em[..., None] * EMBER_LIGHT
        out = laid["albedo"] * light
        if glow <= 0:
            return out, None
        # where it is burning: its cracks glowing orange, the char between them red, breathing
        ph = laid["phase"] * TAU
        breathe = sum(np.sin(TAU * m * tau + ph * (2 + i)) for i, m in enumerate(beats)) / max(1, len(beats))
        live = glow * (0.8 + 0.2 * breathe)
        heat = laid["crack"] * (GLOW_CRACK * live) + laid["skin"] * (GLOW_SKIN * live)
        given = np.empty_like(out)
        for c in range(3):
            given[..., c] = np.interp(heat, EMBER_I, EMBER[:, c])
        out += given
        return out, given
