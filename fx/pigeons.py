"""Pigeons ("Chim bồ câu"): rock doves on the ground or on a ledge, walking about, stopping to look round and, now and
then, bending down to peck at seeds (a placed effect: Effect.BOX; drawn over the picture, lit by the scene; see
effects.py for how every effect loops).

A pigeon is drawn by code, not from a picture. Its body is one smooth shape: ellipsoids (breast, body, belly, vent, the
folded wings over its flanks, their primaries along its sides turned in so their tips meet over the tail, the tail and
its spread end, the neck, the head) melted into each other as their distances are (a smooth union: Inigo Quilez's
polynomial smooth minimum), so it has no lumps where its parts meet, as feathers cover a real bird. Standing still, it
stands up tall, breast out, its tail down, as the rock dove of a photograph the parts were fitted to. A far camera
(straight rays) looks at it from a little above; each ray is marched to the smooth surface from where it meets the plain
union of the ellipsoids (a few steps: the smooth surface lies just outside it). Its small parts (beak, cere, legs, toes)
are ellipsoids met directly. Its feathers are painted by where on the bird a point is, each part's colour blended by the
same weights that melt the parts together (no seams), each part's patterns where they are on it in any pose: a rock
dove's slate head, its iridescent neck (green seen straight on, purple at a slant and round its foot), pale breast, pale
back and folded wings with two broad black bars across them (white-edged), the dark primaries behind, the black band at
the tail's end, darker flanks and belly, a small orange eye, the white cere, coral feet; or a checker, a dark one, a red
one, a pied one, a white dove; a faint mottling, every pattern blurred to the pixel so it does not shimmer. One ray a
pixel inside a bird DETAIL pixels long or more, ss x ss where its outline or a small part's crosses a pixel; softened a
little ("Độ mềm") as a photograph is. It is lit by the scene where its box stands (as steam is: light_at), and lies in
its own shadow: dark right under it, soft and wider away from the light.

Every bird keeps its own day over the loop: walking a while round a closed path of its own (whole laps a loop: back
where it started), stopping to look round or, now and then ("Mổ hạt"), to peck at the ground a few times, its beak
reaching it and its tail rising. As it walks, its planted foot stays put while the other steps forward, and its head
is held still and thrust forward a step at a time, as pigeons do.
"""
import math

import numpy as np

from . import steam
from .effects import Effect, _choice, _lerp, _num, _place

TAU = 2 * math.pi
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
LENGTH = 0.86  # a pigeon's length from beak to tail, in the units of its parts (its "size" is this many pixels)
CAP = {"ground": 30, "ledge": 14}  # birds at most, each way of being
SMALLEST, LARGEST = 0.03, 0.5  # (a bird's length at "Cỡ chim" 0 and 100: box heights; as close to the camera
# as in a photograph of pigeons on a pavement, at 100)
BLEND = 0.03  # how far the body's parts melt into each other (units)
DETAIL = 48  # a bird this long or more (px): one ray a pixel, ss x ss only where an edge crosses a pixel
SOFTEST = 0.0037  # (the blur at "Độ mềm" 100: a share of the box's height, 4 px for a box 1080 high)


def _rot(yaw=0.0, pitch=0.0, roll=0.0):
    """A rotation (radians): roll about x, then pitch about y (the nose up), then yaw about z."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cp, 0.0, -sp], [0.0, 1.0, 0.0], [sp, 0.0, cp]])
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _along(a, b):
    """A rotation whose z axis points from a to b (for a part stretched between two points)."""
    z = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    z /= max(float(np.linalg.norm(z)), 1e-9)
    x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 0.0, 1.0]) - z[2] * z
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z], 1)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _lin(rgb):
    return (np.asarray(rgb, np.float32) / 255.0) ** 2.2


def _fade(freq, fp):
    """How much of a pattern of `freq` radians a unit to keep where a pixel is `fp` units wide: all of it while its
    waves are more than 4 pixels long, none below about 2.6 (it would only shimmer)."""
    return float(_smooth((2.4 - freq * fp) / 0.8))


# --- its feathers ----------------------------------------------------------------------------------------------------

def _feathers(a, b, size, aspect, fp):
    """Rows of overlapping feathers `size` units long, the rows `aspect` times as wide apart, at points (a: along the
    bird toward its tail, b: across the rows; units), each one's rounded tip pointing back over the next: how light
    each point is (about 1: darker just past the tip before it, lighter toward its own), how near a tip's edge (0..1),
    a random number of its feather, and a mark near its tip (0..1: a checker's spot); faded out where the feathers
    are smaller than a few pixels (`fp`: a pixel's width, units). None where they would only shimmer."""
    k = float(np.clip((aspect * size / max(fp, 1e-9) - 2.0) / 3.0, 0.0, 1.0))
    if k <= 0:
        return None
    r = b / (aspect * size)
    row = np.floor(r)
    fr = r - row
    tip = 0.3 * (1 - (2 * fr - 1) ** 2)  # (each tip's edge: furthest back in the middle of its row)
    c = a / size + 0.5 * np.mod(row, 2) - tip
    col = np.floor(c)
    t = c - col  # (0: just past the tip before it, in its shade .. 1: at its own tip)
    width = max(0.04, 1.2 * fp / size)
    edge = 1 - _smooth(np.minimum(t, 1 - t) / width)
    rnd = np.sin(col * 12.9898 + row * 78.233) * 43758.5453
    rnd -= np.floor(rnd)
    light = 1 + k * (0.09 * t - 0.03 - 0.09 * (1 - _smooth(t / 0.3)) - 0.05 * (1 - _smooth(fr / 0.25)))
    mark = _smooth((t - 0.42) / 0.1) * _smooth((0.93 - t) / 0.05) * _smooth((0.72 - np.abs(2 * fr - 1)) / 0.15)
    return light, k * edge, rnd, k * mark


COATS = {  # sRGB: a rock dove (blue bar), a dark one, a red (ash-red) one, a white dove
    "grey": {"head": (96, 106, 140), "neck": (74, 82, 104), "green": (46, 146, 104), "purple": (124, 66, 144),
             "breast": (150, 148, 176), "back": (178, 184, 202), "flank": (142, 148, 170), "bar": (28, 28, 34),
             "prim": (52, 53, 60), "tail": (96, 100, 116), "band": (30, 30, 36), "belly": (148, 154, 176),
             "beak": (62, 60, 66), "cere": (206, 202, 196), "iris": (222, 112, 36), "feet": (214, 72, 76),
             "spot": (58, 60, 70), "sheen": 1.0},
    "dark": {"head": (52, 54, 60), "neck": (44, 48, 52), "green": (46, 104, 78), "purple": (96, 60, 104),
             "breast": (58, 56, 64), "back": (66, 68, 76), "flank": (58, 60, 68), "bar": (34, 34, 38),
             "prim": (40, 42, 46), "tail": (54, 56, 62), "band": (30, 30, 34), "belly": (64, 66, 74),
             "beak": (36, 34, 36), "cere": (212, 210, 204), "iris": (210, 110, 42), "feet": (188, 82, 86),
             "spot": (34, 34, 38), "sheen": 1.1},
    "brown": {"head": (150, 118, 104), "neck": (140, 104, 96), "green": (128, 118, 98), "purple": (156, 104, 120),
              "breast": (150, 112, 104), "back": (196, 172, 156), "flank": (164, 134, 120), "bar": (122, 74, 58),
              "prim": (150, 124, 112), "tail": (176, 152, 138), "band": (142, 102, 88), "belly": (170, 146, 134),
              "beak": (196, 160, 146), "cere": (232, 226, 218), "iris": (210, 110, 42), "feet": (198, 90, 92),
              "spot": (138, 92, 76), "sheen": 0.5},
    "white": {"head": (232, 232, 228), "neck": (230, 230, 226), "green": (226, 232, 224), "purple": (232, 226, 230),
              "breast": (234, 232, 228), "back": (238, 238, 234), "flank": (228, 228, 224), "bar": (234, 234, 230),
              "prim": (224, 224, 220), "tail": (234, 234, 230), "band": (230, 230, 226), "belly": (232, 232, 228),
              "beak": (206, 168, 156), "cere": (238, 232, 226), "iris": (64, 38, 28), "feet": (204, 100, 104),
              "spot": (232, 232, 228), "sheen": 0.0},
}
MIXED = (("grey", 0.42), ("checker", 0.18), ("dark", 0.09), ("brown", 0.07), ("pied", 0.14), ("white", 0.1))  # (a
# town's flock)
WHITE = _lin((236, 236, 232))


class Coat:
    """A bird's feathers: its coat's colours (linear light), dark spots on its wings (a checker) or white patches (a
    pied one), a little lighter or darker than the next bird, a faint mottling."""

    def __init__(self, name, rng):
        base = name if name in COATS else "grey"
        self.c = {k: (_lin(v) if isinstance(v, tuple) else v) for k, v in COATS[base].items()}
        self.checker = name == "checker"
        self.pied = name == "pied"
        self.tone = 0.92 + 0.16 * rng.random()
        self.waves = rng.normal(0.0, 1.0, (5, 3)) * 45.0  # (a faint mottling)
        self.phases = rng.random(5) * TAU
        self.patches = rng.normal(0.0, 1.0, (4, 3)) * 11.0  # (a pied bird's white patches)
        self.patch_phases = rng.random(4) * TAU
        self.patch_at = rng.uniform(-0.6, 0.3)  # (how much of it is white)

    def grain(self, p, fp):
        """A faint mottling at points p (about 0.93..1.07): its waves longer than a few pixels (`fp`: a pixel's width,
        units), so it does not shimmer on a small bird."""
        out = np.zeros(len(p), np.float32)
        for w, ph in zip(self.waves, self.phases):
            k = _fade(float(np.linalg.norm(w)), fp)
            if k > 0:
                out += k * np.sin(p @ w + ph).astype(np.float32)
        return 1.0 + 0.035 * out / 2.2

    def white(self, p):
        """How white a pied bird is at points p (0..1): a few big patches."""
        f = np.zeros(len(p), np.float32)
        for w, ph in zip(self.patches, self.patch_phases):
            f += np.sin(p @ w + ph).astype(np.float32)
        return _smooth((f / 2.0 - self.patch_at) / 0.25)


# --- its shape and poses ----------------------------------------------------------------------------------------------

HIP = np.array([-0.01, 0.0, 0.12])  # the body tips about this (pecking)
NECK = np.array([0.1, 0.0, 0.31])  # where the neck leaves the shoulders (standing)
HEAD = np.array([0.182, 0.0, 0.43])  # its head's middle (standing)
HEAD_R = np.array([0.059, 0.045, 0.052])
# the body as it walks: name, centre, radii, pitch (degrees, the nose up), and the folded wings' and their primaries'
# turn (degrees: their rear ends in, so the primaries' tips meet over the tail and it narrows smoothly to the tail seen
# from behind); the wings lie over the flanks, as a pigeon's cover its sides
BODY = (("breast", (0.08, 0.0, 0.262), (0.132, 0.106, 0.132), 16),
        ("body", (-0.055, 0.0, 0.245), (0.2, 0.104, 0.125), 5),
        ("belly", (-0.01, 0.0, 0.15), (0.14, 0.088, 0.075), 4),
        ("vent", (-0.2, 0.0, 0.188), (0.11, 0.058, 0.05), -9),
        ("wing", (-0.11, 0.062, 0.255), (0.2, 0.052, 0.085), 8, 9),
        ("wing", (-0.11, -0.062, 0.255), (0.2, 0.052, 0.085), 8, -9),
        ("tail", (-0.35, 0.0, 0.2), (0.18, 0.056, 0.022), -9),
        ("fan", (-0.45, 0.0, 0.184), (0.085, 0.064, 0.016), -9),
        ("prim", (-0.31, 0.032, 0.226), (0.165, 0.011, 0.026), -8, 11),
        ("prim", (-0.31, -0.032, 0.226), (0.165, 0.011, 0.026), -8, -11))
# the body standing tall, breast out, as a rock dove at rest (the same parts, in the same order: a pose goes from BODY
# to this as it stands up)
BODY_UP = (("breast", (0.073, 0.0, 0.287), (0.165, 0.104, 0.105), 68),  # (fitted to a photograph's outline)
           ("body", (-0.11, 0.0, 0.183), (0.199, 0.075, 0.064), 22),
           ("belly", (0.028, 0.0, 0.155), (0.156, 0.082, 0.065), 48),
           ("vent", (-0.205, 0.0, 0.104), (0.165, 0.05, 0.034), 32),
           ("wing", (-0.05, 0.06, 0.235), (0.23, 0.05, 0.1), 48, 9),
           ("wing", (-0.05, -0.06, 0.235), (0.23, 0.05, 0.1), 48, -9),
           ("tail", (-0.24, 0.0, 0.071), (0.157, 0.052, 0.018), 11),
           ("fan", (-0.36, 0.0, 0.047), (0.085, 0.06, 0.016), 11),
           ("prim", (-0.25, 0.05, 0.148), (0.17, 0.014, 0.04), 38, 6),
           ("prim", (-0.25, -0.05, 0.148), (0.17, 0.014, 0.04), 38, -6))
NECK_UP = np.array([0.034, 0.0, 0.332])  # (where its neck leaves its shoulders, standing tall)
TAIL_AT = np.array([-0.22, 0.0, 0.19])  # (the tail bends up and down about this)
TAILS = {"tail": 1.0, "fan": 1.0, "prim": 1.0, "vent": 0.5}  # (how much of the tail's bend each part takes: the
# wingtips lie on it)
TIGHT = {"prim": 0.008, "fan": 0.008}  # (the wingtips and the tail's spread end melt in less: thin, lying on it)
SMOOTH = len(BODY) + 2  # (the parts of the smooth body: BODY, the neck, the head)


class Pose:
    """How a bird stands now: how tall (0: as it walks .. 1: standing tall, breast out: BODY_UP), its body tipped
    forward (pitch, radians: < 0 tipped up), its tail bent up from it (radians), its head's middle (bird frame), the
    head's own pitch and turn, its feet (bird frame, each: x, y, lift)."""

    def __init__(self, pitch=0.0, head=None, head_pitch=0.17, head_yaw=0.0, feet=None, tail=0.0, tall=0.0):
        self.pitch, self.head_pitch, self.head_yaw, self.tail, self.tall = pitch, head_pitch, head_yaw, tail, tall
        self.head = HEAD.copy() if head is None else np.asarray(head, np.float64)
        self.feet = feet if feet is not None else ((0.0, 0.034, 0.0), (0.0, -0.034, 0.0))


def parts(pose):
    """A bird's parts in a pose: (name, centre, rotation (columns: its axes), radii), in the bird frame (x forward, y
    left, z up; its feet on the ground at 0): the SMOOTH parts of its body first, then its small ones."""
    tip = _rot(pitch=-pose.pitch)  # (the body tipped forward about the hip)
    k = min(max(float(pose.tall), 0.0), 1.0)
    out = []
    for (name, c, r, pitch, *turn), (_, cu, ru, pu, *tu) in zip(BODY, BODY_UP):
        c = np.array(c, np.float64) * (1 - k) + np.array(cu, np.float64) * k
        r = np.array(r, np.float64) * (1 - k) + np.array(ru, np.float64) * k
        yaw = math.radians(turn[0] + (tu[0] - turn[0]) * k) if turn else 0.0
        own = _rot(yaw=yaw, pitch=math.radians(pitch + (pu - pitch) * k))
        bend = TAILS.get(name, 0.0) * pose.tail
        if bend:  # (the tail bent up about its joint)
            up = _rot(pitch=-bend)
            c, own = TAIL_AT + up @ (c - TAIL_AT), up @ own
        out.append((name, HIP + tip @ (c - HIP), tip @ own, r))
    base = HIP + tip @ (NECK + (NECK_UP - NECK) * k - HIP)
    head = pose.head
    span = float(np.linalg.norm(head - base))
    out.append(("neck", base + (head - base) * 0.42, _along(base, head),
                np.array([0.072 + 0.002 * k, 0.064 + 0.008 * k, max(0.07, span / 2 + 0.02)])))
    turn = _rot(yaw=pose.head_yaw, pitch=pose.head_pitch)
    out.append(("head", head, turn, HEAD_R))
    out.append(("beak", head + turn @ np.array([0.066, 0.0, -0.012]), turn @ _rot(pitch=-0.35),
                np.array([0.031, 0.0105, 0.0115])))
    out.append(("cere", head + turn @ np.array([0.056, 0.0, 0.003]), turn @ _rot(pitch=-0.35),
                np.array([0.017, 0.012, 0.0105])))
    for side, foot in zip((1, -1), pose.feet):
        hip = np.array([0.0, side * 0.03, 0.11])
        f = np.array(foot, np.float64)
        knee = f + np.array([0.0, 0.0, 0.018])
        out.append(("leg", (hip + knee) / 2, _along(knee, hip),
                    np.array([0.012, 0.012, max(0.03, float(np.linalg.norm(hip - knee)) / 2 + 0.008)])))
        out.append(("toe", f + np.array([0.034, 0.0, 0.0075]), _rot(yaw=side * 0.16),
                    np.array([0.044, 0.0085, 0.0075])))
        out.append(("toe", f + np.array([0.024, side * 0.021, 0.0075]), _rot(yaw=side * 0.75),
                    np.array([0.038, 0.008, 0.0075])))
        out.append(("toe", f + np.array([-0.018, -side * 0.004, 0.0075]), _rot(yaw=side * 0.2),
                    np.array([0.026, 0.0075, 0.0075])))  # (the one pointing back)
    return out


# --- drawing a bird ---------------------------------------------------------------------------------------------------

class _Rays:
    """Rays of a far camera in a bird's frame: their origins (three arrays) and their one direction."""

    def __init__(self, ox, oy, oz, d):
        self.ox, self.oy, self.oz, self.d = ox, oy, oz, d

    def local(self, part):
        """In a part's own frame, scaled to its unit sphere: where each ray starts, and how far it goes a unit of its
        length."""
        _, c, rot, r = part
        ex, ey, ez = self.ox - c[0], self.oy - c[1], self.oz - c[2]
        return [(ex * rot[0, j] + ey * rot[1, j] + ez * rot[2, j]) / r[j] for j in range(3)], (self.d @ rot) / r

    def meet(self, part):
        """Where each ray enters and leaves an ellipsoid, nan where it misses."""
        return _enter(*self.local(part))


def _enter(q, dd):
    """Where rays (q, dd: from _Rays.local) enter and leave the unit sphere, nan where they miss."""
    a = float(dd @ dd)
    b = q[0] * dd[0] + q[1] * dd[1] + q[2] * dd[2]
    disc = b * b - a * (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] - 1.0)
    s = np.sqrt(np.maximum(disc, 0.0))
    t0, t1 = (-b - s) / a, (-b + s) / a
    miss = disc <= 0.0
    t0[miss] = np.nan
    t1[miss] = np.nan
    return t0, t1


def _distance(q, dd, r, t):
    """An ellipsoid's signed distance (Inigo Quilez's bound) along rays (q, dd: from _Rays.local) at distances t."""
    ax, ay, az = q[0] + t * dd[0], q[1] + t * dd[1], q[2] + t * dd[2]
    k0 = np.sqrt(ax * ax + ay * ay + az * az)
    bx, by, bz = ax / r[0], ay / r[1], az / r[2]
    return k0 * (k0 - 1.0) / np.maximum(np.sqrt(bx * bx + by * by + bz * bz), 1e-9)


NEAR = 0.45 * BLEND  # (the smooth surface lies outside the plain union of the parts: a quarter of BLEND where two
# meet, more where several do, as on the rump)


class _Body:
    """The smooth body seen along some rays: every part's frame along each ray (all parts at once: 3 x parts x rays,
    float32), where each ray meets the plain union of the parts, and where it passes within NEAR of one."""

    def __init__(self, rays, body):
        self.body = body
        centres = np.array([part[1] for part in body])
        rots = np.array([part[2] for part in body])
        radii = np.array([part[3] for part in body])
        o = np.stack([rays.ox, rays.oy, rays.oz])
        q = np.einsum("in,kij->jkn", o, rots) - np.einsum("ki,kij->jk", centres, rots)[:, :, None]
        q /= radii.T[:, :, None]
        dd = np.einsum("i,kij->jk", rays.d, rots) / radii.T
        self.q, self.dd, self.r = q.astype(np.float32), dd[:, :, None].astype(np.float32), radii.T[:, :, None]

        def enter(q, dd):  # (where rays enter and leave each unit sphere: parts x rays, nan where they miss)
            a = (dd * dd).sum(0)[:, None]
            b = (q * dd[:, :, None]).sum(0)
            disc = b * b - a * ((q * q).sum(0) - 1.0)
            root = np.sqrt(np.where(disc > 0, disc, np.nan))
            return (-b - root) / a, (-b + root) / a

        t0, _ = enter(q, dd)
        grown = radii.T / (radii.T + NEAR)  # (each part grown by NEAR)
        g0, g1 = enter(q * grown[:, :, None], dd * grown)
        with np.errstate(invalid="ignore"):
            self.union = np.where(np.isnan(t0), np.inf, t0).min(0)
            self.near = np.where(np.isnan(g0), np.inf, g0).min(0)
            self.far = np.where(np.isnan(g1), -np.inf, g1).max(0)
        self.blends = [TIGHT.get(part[0], BLEND) for part in body]

    def distance(self, idx, t, shares=False):
        """The smooth body's distance along rays `idx` at distances t (shares=True: also each part's share of it)."""
        a = self.q[:, :, idx] + t.astype(np.float32)[None, None, :] * self.dd  # (each part's unit sphere)
        k0 = np.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
        b = a / self.r
        each = k0 * (k0 - 1.0) / np.maximum(np.sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2]), 1e-9)
        d, out = each[0], [np.ones(len(idx))]
        for k in range(1, len(each)):
            blend = self.blends[k]
            h = np.clip(0.5 + 0.5 * (each[k] - d) / blend, 0.0, 1.0)
            d = each[k] + (d - each[k]) * h - blend * h * (1.0 - h)
            if shares:
                out = [s * h for s in out]
                out.append(1.0 - h)
        return (d, out) if shares else d

    def _refine(self, idx, lo, hi, dlo, dhi, steps):
        """The surface between lo (outside) and hi (inside) along rays idx: false position, kept off the ends."""
        for _ in range(steps):
            f = np.clip(dlo / np.maximum(dlo - dhi, 1e-12), 0.15, 0.85)
            tm = lo + (hi - lo) * f
            dm = self.distance(idx, tm)
            out = dm > 0
            lo, dlo = np.where(out, tm, lo), np.where(out, dm, dlo)
            hi, dhi = np.where(out, hi, tm), np.where(out, dhi, dm)
        return lo + (hi - lo) * np.clip(dlo / np.maximum(dlo - dhi, 1e-12), 0.0, 1.0)

    def march(self):
        """Where each ray meets the smooth body: its distance (inf where it does not)."""
        out = np.full(len(self.union), np.inf)
        # rays that meet the plain union: the smooth surface lies between NEAR before it and it
        idx = np.flatnonzero(np.isfinite(self.union))
        if len(idx):
            hi = self.union[idx]
            lo = hi - NEAR
            dlo, dhi = self.distance(idx, lo), self.distance(idx, hi)
            back = np.flatnonzero(dlo <= 0)  # (a ray grazing the body, a thin tail: it meets the smooth surface further
            # back, after it comes within NEAR of a part: the first change of sign from there)
            if len(back):
                k, b, cols = 6, idx[back], np.arange(len(back))
                ts = self.near[b][None, :] + (lo[back] - self.near[b])[None, :] * (np.arange(k + 1) / k)[:, None]
                ds = np.stack([self.distance(b, ts[j]) for j in range(k + 1)])
                j = np.maximum(np.argmax(ds <= 0, axis=0), 1)
                lo[back], dlo[back], hi[back], dhi[back] = ts[j - 1, cols], ds[j - 1, cols], ts[j, cols], ds[j, cols]
            out[idx] = self._refine(idx, lo, hi, np.maximum(dlo, 1e-9), np.minimum(dhi, 0.0), 4)
        # rays that only pass near a part: where the smooth surface bulges out between two parts, if anywhere
        idx = np.flatnonzero(~np.isfinite(self.union) & np.isfinite(self.near))
        if len(idx):
            k = 6
            ts = self.near[idx][None, :] + (self.far[idx] - self.near[idx])[None, :] * (np.arange(k + 1) / k)[:, None]
            ds = np.stack([self.distance(idx, ts[j]) for j in range(k + 1)])
            inside = ds < 0
            first = np.argmax(inside, axis=0)
            got = np.flatnonzero(inside.any(axis=0) & (first > 0))
            if len(got):
                j = first[got]
                lo, hi = ts[j - 1, got], ts[j, got]
                out[idx[got]] = self._refine(idx[got], lo, hi, ds[j - 1, got], ds[j, got], 3)
        return out


_RESTING = []
HOOD = 0.045  # how far the neck's iridescent feathers reach past it (units)


def _sheen(c, facing):
    """The neck's iridescent feathers seen at `facing` (|cos| of the angle to the camera): green straight on, purple at
    a slant."""
    f = facing[:, None]
    return np.clip(c["neck"] + c["sheen"] * (0.5 * (c["green"] - c["neck"]) * f ** 2
                                             + 0.5 * (c["purple"] - c["neck"]) * (1 - f) ** 1.5), 0.0, 1.0)


def _body_colour(q, nz, c, fp):
    """The colour of the body, the belly and the vent at points q on the bird as it walks (nz: how much each faces up
    there): its back pale, its flanks darker, its belly paler; faint soft feathers."""
    x, z = q[:, 0], q[:, 2]
    pale = _smooth((nz - 0.35) / 0.4) * _smooth((0.1 - x) / 0.06)
    out = c["flank"] + (c["back"] - c["flank"]) * pale[:, None]
    soft = _feathers(-x, (0.34 - z) + 0.5 * np.abs(q[:, 1]), 0.02, 0.8, fp)
    if soft is not None:
        out = out * (1 + 0.5 * (1 - pale) * (soft[0] - 1))[:, None]
    belly = _smooth((0.17 - z) / 0.05) * (1 - pale)
    return out + (c["belly"] - out) * belly[:, None]


WING = 0.23  # (the half length of the folded wing its patterns are laid out on: they lie as far from its front in
# any pose, however long it is)


def _wing_colour(a, b, up, reach, c, coat, fp):
    """A folded wing's colour at points of it (a: along it, +1 its front, -1 its rear end, `reach` units from its
    middle; up: -1 its lower edge .. +1 its top; b: across it, +1 its outer face): pale, rows of small feathers over
    its front, long ones behind them, two broad black bars across its outer face (their lower ends further forward)
    with pale tips behind them, its dark primaries behind."""
    count = len(a)
    a = 1.0 - (1.0 - a) * (reach / WING)
    along, high = a * WING, up * 0.08  # (units)
    out = np.broadcast_to(c["back"], (count, 3)).copy()
    rows_b = (0.08 - high) + 0.15 * (1 - np.abs(b))
    covert = _feathers(-along, rows_b, 0.03, 0.7, fp)
    front = _smooth((a + 0.1) / 0.15)
    if covert is not None:
        light, edge, rnd, mark = covert
        out = out * (1 + 0.6 * front * (light - 1))[:, None] + (0.02 * front * edge)[:, None] * c["back"]
        if coat.checker:  # (a dark mark near the tip of each covert: a checker)
            out = out + (c["spot"] - out) * (mark * (0.55 + 0.45 * rnd) * front)[:, None]
    elif coat.checker:  # (too small to see each feather: the checker's mean)
        out = out + (c["spot"] - out) * (0.3 * front)[:, None]
    long_ = _feathers(-(along - 0.12 * WING * up), rows_b, 0.09, 0.28, fp)  # (the secondaries, slanting as the bars)
    if long_ is not None:
        out = out * (1 + (1 - front) * (long_[0] - 1))[:, None]
    bars = np.zeros(count)
    edges = np.zeros(count)
    width = max(0.035, fp / WING)
    for a0 in (-0.13, -0.45):  # (across its rear half, their lower ends further forward)
        d = a - (a0 - 0.12 * up)
        bars = np.maximum(bars, _smooth((0.095 - np.abs(d)) / width))
        edges = np.maximum(edges, _smooth((0.055 - np.abs(d + 0.17)) / width))
    outer = _smooth((b + 0.3) / 0.4)  # (not round its inner face: no rings seen from behind)
    out = out + (np.minimum(c["back"] * 1.35, 1.0) - out) * (0.95 * edges * (1 - bars) * outer)[:, None]
    out = out + (c["bar"] - out) * (bars * outer)[:, None]
    tip = _smooth((-a - 0.62) / 0.16)  # (its dark primaries)
    return out + (c["prim"] - out) * tip[:, None]


def _part_colour(name, q, local, rad, nz, facing, hood, c, coat, fp):
    """One part's colour at points q on the bird as it walks (its own frame taken back there; `local`: the points in
    the part's own unit frame, its radii `rad` in this pose), facing up nz, facing the camera `facing`; `hood`: how
    near the neck (1 on it)."""
    if name == "wing":
        return _wing_colour(local[:, 0], local[:, 1] * np.sign(q[:, 1]), local[:, 2], rad[0], c, coat, fp)
    if name in ("body", "belly", "vent", "breast"):  # (iridescent, purple, round the neck's foot)
        out = _body_colour(q, nz, c, fp) * (0.95 if name == "vent" else 1.0) if name != "breast" else c["breast"]
        sheen = _sheen(c, facing)
        sheen = sheen + (c["purple"] - sheen) * (0.5 if c["sheen"] else 0.0)
        return out + (sheen - out) * hood[:, None]
    if name == "neck":  # (green, purple toward its foot)
        sheen = _sheen(c, facing)
        foot = _smooth((0.4 - local[:, 2]) / 0.5) * (0.75 if c["sheen"] else 0.0)
        return sheen + (c["purple"] - sheen) * foot[:, None]
    if name == "head":
        return np.broadcast_to(c["head"], (len(q), 3))
    x = q[:, 0]  # (the tail and the wingtips: long feathers)
    quills = _feathers(-x, np.abs(q[:, 1]) + 0.3 * (0.25 - q[:, 2]), 0.16, 0.13, fp)
    if name == "prim":  # (darkest along their top edge)
        tips = c["prim"] + (c["tail"] - c["prim"]) * (0.45 * _smooth((0.1 - local[:, 2]) / 0.9))[:, None]
        return tips * (quills[0] * (1 - 0.35 * quills[1]))[:, None] if quills is not None else tips
    tail = c["tail"] + (c["band"] - c["tail"]) * _smooth((-x - 0.455) / 0.02)[:, None]
    return tail * (quills[0] * (1 - 0.25 * quills[1]))[:, None] if quills is not None else tail


def _albedo(p, n, view, shares, body, coat, pose, fp):
    """The feathers' colour (linear light) at body points p facing n (the camera's way `view`): each part's own
    colour where the point is on that part as the bird walks (its patterns follow it in any pose), blended by the
    part's share of the smooth body there. `fp`: a pixel's width (units): the patterns blurred to it."""
    c = coat.c
    count = len(p)
    if not _RESTING:
        _RESTING.extend(parts(Pose())[:SMOOTH])
    facing = np.abs(n @ view)
    _, nc, nrot, nr = body[SMOOTH - 2]  # (the neck's own feathers reach a little past it: how far outside it)
    ln = ((p - nc) @ nrot) / nr
    k0 = np.linalg.norm(ln, axis=1)
    hood = _smooth((HOOD - k0 * (k0 - 1.0) / np.maximum(np.linalg.norm(ln / nr, axis=1), 1e-9)) / 0.035)
    out = np.zeros((count, 3), np.float32)
    whole = np.zeros((count, 3))  # (where each point is on the bird as it walks, all its parts' shares: the patches)
    w_head = np.zeros(count)
    sharp = [sk ** 4 for sk in shares]  # (each part's colour mostly its own, smoothly blended where they meet)
    total = np.maximum(sum(sharp), 1e-12)
    for (name, ck, rk, rad), (_, c0, r0, rad0), s in zip(body, _RESTING, [sk / total for sk in sharp]):
        idx = np.flatnonzero(s > 1e-4)
        if not len(idx):
            continue
        local = ((p[idx] - ck) @ rk) / rad
        qk = c0 + (local * rad0) @ r0.T
        nz = (n[idx] @ rk) @ r0.T[:, 2]  # (the way it faces, turned as the part is)
        out[idx] += s[idx, None] * _part_colour(name, qk, local, rad, nz, facing[idx], hood[idx], c, coat, fp)
        whole[idx] += s[idx, None] * qk
        if name == "head":
            w_head[idx] += s[idx]
    # its eyes: small, orange (a white dove's dark), a thin grey ring, a black pupil
    _, hc, hrot, hr = body[SMOOTH - 1]
    loc = ((p - hc) @ hrot) / hr
    edge = max(0.05, fp / 0.05)
    for side in (1.0, -1.0):
        d = np.sqrt((loc[:, 0] - 0.45) ** 2 + (loc[:, 2] - 0.22) ** 2 + 0.3 * (loc[:, 1] - side * 0.9) ** 2)
        on = (side * loc[:, 1] > 0.3) * w_head
        for size, colour, k in ((0.28, _lin((150, 132, 124)), 0.5), (0.2, c["iris"], 1.0),
                                (0.09, _lin((12, 10, 10)), 1.0)):
            m = (k * _smooth((size - d) / edge) * on)[:, None]
            out = out * (1 - m) + colour * m
    if coat.pied:
        white = coat.white(whole)[:, None]
        out = out * (1 - white) + WHITE * white
    return out * (coat.tone * coat.grain(whole, fp))[:, None]


SMALL = {"leg": "feet", "toe": "feet", "beak": "beak", "cere": "cere"}  # (each small part's colour)


def _trace(rays, body, small):
    """What each ray meets first: which (-1 nothing, 0 the body, 1 + k the k-th small part), its distance, and for a
    small part its own normal (not of unit length); the body as these rays see it."""
    seen = _Body(rays, body)
    best = seen.march()
    which = np.where(np.isfinite(best), 0, -1)
    own = np.zeros((len(best), 3))
    for k, part in enumerate(small):
        t0, _ = rays.meet(part)
        ok = ~np.isnan(t0) & (t0 < best)
        if ok.any():
            best[ok] = t0[ok]
            which[ok] = 1 + k
            _, c, rot, r = part
            at = np.stack([rays.ox[ok], rays.oy[ok], rays.oz[ok]], 1) + t0[ok, None] * rays.d
            own[ok] = ((((at - c) @ rot) / r) / r) @ rot.T
    return which, best, own, seen


def _shade(rays, which, best, own, seen, small, coat, pose, light, turn, fp):
    """The colour (linear light) of the rays that meet the bird: (their indices, colours)."""
    i = np.flatnonzero(which >= 0)
    rgb = np.zeros((len(i), 3), np.float32)
    if not len(i):
        return i, rgb
    p = np.stack([rays.ox[i], rays.oy[i], rays.oz[i]], 1) + best[i, None] * rays.d
    nrm = np.zeros((len(i), 3))
    alb = np.zeros((len(i), 3), np.float32)
    on = np.flatnonzero(which[i] == 0)
    if len(on):
        # the body's normal: its parts' own, weighted by their shares of the smooth distance there
        _, shares = seen.distance(i[on], best[i[on]], shares=True)
        pb = p[on]
        g = np.zeros((len(on), 3))
        for part, s in zip(seen.body, shares):
            _, c, rot, r = part
            gk = ((((pb - c) @ rot) / r) / r) @ rot.T
            g += s[:, None] * gk / np.maximum(np.linalg.norm(gk, axis=1, keepdims=True), 1e-12)
        nrm[on] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        alb[on] = _albedo(pb, nrm[on], -rays.d, shares, seen.body, coat, pose, fp)
    for k, part in enumerate(small):
        j = np.flatnonzero(which[i] == 1 + k)
        if len(j):
            g = own[i[j]]
            nrm[j] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
            alb[j] = coat.c[SMALL[part[0]]]
    key, way, ambient = light
    nw = nrm @ turn.T  # (the scene's frame)
    lam = np.clip((nw @ way + 0.35) / 1.35, 0.0, 1.0)  # (feathers: the light wraps round a little)
    sky = 0.55 + 0.45 * np.clip(0.5 + 0.5 * nw[:, 2], 0.0, 1.0)
    # feathers soft as velvet: a little more light where the surface is seen edge on
    rim = (1 - np.clip(np.abs(nrm @ rays.d), 0.0, 1.0)) ** 3
    rgb[:] = alb * (ambient * sky[:, None] + key * lam[:, None] + (0.35 * rim)[:, None] * (key + ambient))
    return i, rgb


def draw_bird(pose, coat, yaw, scale, tilt, light, ss):
    """A bird seen by a far camera tilted down by `tilt` (radians), heading `yaw` (0: to the right of the picture,
    pi/2: away), `scale` pixels a unit: its picture (premultiplied linear RGB, alpha) and where its feet are in it
    (x, y pixels). `light`: (key colour, key way, ambient colour) linear, the way in the scene (x right, y away, z up).
    A bird DETAIL pixels long or more: one ray a pixel, `ss` x `ss` where its outline or a small part's crosses a
    pixel; a smaller one: `ss` x `ss` everywhere."""
    allp = parts(pose)
    body, small = allp[:SMOOTH], allp[SMOOTH:]
    ct, st = math.cos(tilt), math.sin(tilt)
    fwd = np.array([0.0, ct, -st])
    up = np.array([0.0, st, ct])
    turn = _rot(yaw=yaw)
    fb = turn.T @ fwd
    right, down, back = turn.T @ np.array([1.0, 0.0, 0.0]), -(turn.T @ up), -3.0 * fb
    # where the bird lies on the picture (pixels from its feet: right, down): each part's middle +- its largest radius
    spans = []
    for _, c, _, r in allp:
        wv = turn @ c
        m = (float(r.max()) + BLEND) * scale + 1.5
        spans.append((wv[0] * scale - m, -(wv @ up) * scale - m, wv[0] * scale + m, -(wv @ up) * scale + m))
    spans = np.array(spans)
    x0, y0 = math.floor(spans[:, 0].min()), math.floor(spans[:, 1].min())
    w, h = max(1, math.ceil(spans[:, 2].max()) - x0), max(1, math.ceil(spans[:, 3].max()) - y0)
    fp = 1.0 / scale

    def rays(px, py):  # (through picture points px, py: pixels from its feet)
        u, v = px / scale, py / scale
        return _Rays(u * right[0] + v * down[0] + back[0], u * right[1] + v * down[1] + back[1],
                     u * right[2] + v * down[2] + back[2], fb)

    n = ss * ss
    sub = (np.arange(ss) + 0.5) / ss
    rgb = np.zeros((w * h, 3), np.float32)
    alpha = np.zeros(w * h, np.float32)
    if scale * LENGTH >= DETAIL:  # (a ray through each pixel's middle: its edges where what it meets changes)
        gx, gy = np.tile(np.arange(w) + 0.5 + x0, h), np.repeat(np.arange(h) + 0.5 + y0, w)
        could = np.zeros(w * h, bool)  # (only the pixels some part can be in)
        for sx0, sy0, sx1, sy1 in spans:
            could |= (gx >= sx0) & (gx < sx1) & (gy >= sy0) & (gy < sy1)
        cand = np.flatnonzero(could)
        r1 = rays(gx[cand], gy[cand])
        which_c, best, own, seen = _trace(r1, body, small)
        which = np.full(w * h, -1)
        which[cand] = which_c
        grid = which.reshape(h, w)
        edge = np.zeros((h, w), bool)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a0, a1 = (0, w - dx) if dx >= 0 else (-dx, w)
            diff = grid[:h - dy, a0:a1] != grid[dy:, a0 + dx:a1 + dx]
            edge[:h - dy, a0:a1] |= diff
            edge[dy:, a0 + dx:a1 + dx] |= diff
        flat = edge.reshape(-1)
        i, col = _shade(r1, np.where(flat[cand], -1, which_c), best, own, seen, small, coat, pose, light, turn, fp)
        rgb[cand[i]] = col
        alpha[cand[i]] = 1.0
        pixels = np.flatnonzero(flat)  # (made again below, ss x ss)
    else:
        pixels = np.arange(w * h)
    ey, ex = np.divmod(pixels, w)
    r2 = rays((ex[:, None] + x0 + np.tile(sub, ss)[None, :]).reshape(-1),
              (ey[:, None] + y0 + np.repeat(sub, ss)[None, :]).reshape(-1))
    which, best, own, seen = _trace(r2, body, small)
    i, col = _shade(r2, which, best, own, seen, small, coat, pose, light, turn, fp)
    at = pixels[i // n]  # (each edge ray's pixel: their mean, premultiplied)
    for ch in range(3):
        rgb[:, ch] += np.bincount(at, col[:, ch], w * h).astype(np.float32) / n
    alpha += np.bincount(at, None, w * h).astype(np.float32) / n
    return rgb.reshape(h, w, 3), alpha.reshape(h, w), (-x0, -y0)


def _blur(a, sigma):
    """A small Gaussian blur of an image (h x w, or h x w x 3), grown by 3 sigma each side: (blurred, how far)."""
    r = int(math.ceil(3 * sigma))
    if sigma < 0.08 or r < 1:
        return a, 0
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    a = np.pad(a, [(r, r), (r, r)] + [(0, 0)] * (a.ndim - 2))
    for axis in (0, 1):
        out = np.zeros_like(a)
        size = a.shape[axis]
        for j, kv in enumerate(k):
            lo, hi = max(0, r - j), min(size, size + r - j)
            dst = [slice(None)] * a.ndim
            src = [slice(None)] * a.ndim
            dst[axis], src[axis] = slice(lo, hi), slice(lo + j - r, hi + j - r)
            out[tuple(dst)] += kv * a[tuple(src)]
        a = out
    return a, r


def _inside(ix, iy, shape, x0, y0, w, h):
    """How much of each pixel of a picture laid at (ix, iy) is inside the box (its preview clip is the box alone: what
    lies outside it is cut off there), or None if none of it is."""
    xs = np.arange(ix, ix + shape[1], dtype=np.float64)
    ys = np.arange(iy, iy + shape[0], dtype=np.float64)
    across = np.clip(np.minimum(xs + 1, x0 + w) - np.maximum(xs, x0), 0.0, 1.0)
    down = np.clip(np.minimum(ys + 1, y0 + h) - np.maximum(ys, y0), 0.0, 1.0)
    if not across.any() or not down.any():
        return None
    return (down[:, None] * across[None, :]).astype(np.float32)


# --- a bird's day on the ground ---------------------------------------------------------------------------------------

STRIDE = 0.32  # (how far it goes while each foot steps once, units: steps of 0.16, about 6 cm)
WALK = 0.5  # (its walking speed, units a second, at "speed" 50: about three steps a second)
TURN = 0.1  # (on a ledge: the radius it turns round at each end, units)
PECK = np.array([0.31, 0.0, 0.099])  # (its head's middle as its beak touches the ground)
WALK_UP, TAIL_WALK = 0.12, 0.05  # (walking: how far its body is tipped up and its tail bent up, radians)
HEAD_UP = np.array([0.123, 0.0, 0.6])  # (its head's middle standing tall, its neck stretched)
DOWN, PECKING, UP = 0.28, 0.42, 0.32  # (seconds: bending down to peck, each peck, coming up again)


class Walker:
    """One bird on the ground over the loop (`loop` seconds): walking round a closed path of its own (whole laps a
    loop), stopping to look round or to peck at the ground. `busy`: 0 (mostly stopped) .. 1 (walking most of the
    time); `hungry`: 0 (never pecks) .. 1 (pecks at most stops, a few times); `ledge`: walking to and fro along a line
    (a narrow loop) rather than about; `room`: the farthest its path may go from its home (units): a long walk goes
    round more than once."""

    def __init__(self, rng, loop, busy, pace, ledge=False, room=None, hungry=0.4):
        self.loop = loop
        segs, at = [], 0.0
        walk = rng.random() < 0.6
        while at < loop * 1.02 or len(segs) < 2:
            pick = rng.random(2)
            if walk:
                kind, d, n = "walk", (1.2 + 2.3 * pick[0]) * (0.6 + 0.8 * busy), 0
            elif pick[0] < hungry:  # (a few pecks: more when hungrier)
                n = 1 + int(pick[1] * (1 + 2.5 * hungry))
                kind, d = "peck", DOWN + n * PECKING + UP
            else:
                kind, d, n = "stand", (0.8 + 2.2 * pick[1]) * (1.4 - 0.8 * busy), 0
            segs.append([kind, at, d, n])
            at += d
            walk = not walk
        if segs[0][0] == segs[-1][0] and len(segs) > 2:  # (the loop closes: the last runs on into the one before)
            segs[-2][2] += segs[-1][2]
            segs.pop()
        k = loop / sum(s[2] for s in segs)
        at = 0.0
        for s in segs:
            s[1], s[2] = at, s[2] * k
            at += s[2]
        self.segs = segs
        speed = WALK * pace * rng.uniform(0.85, 1.15)
        self.ramp = 0.25
        self.walked = []  # (distance walked before each segment)
        total = 0.0
        for kind, start, d, _ in segs:
            self.walked.append(total)
            if kind == "walk":
                total += speed * (d - min(self.ramp, d / 2))
        self.speed = speed
        laps = 1
        if room is not None:  # (the longest path that keeps within `room` of home: along a ledge, to and fro; about,
            # a round path)
            laps = max(1, math.ceil(total / (4.0 * room if ledge else 0.55 * TAU * room)))
        self.length = max(total / laps, 1e-6)  # (its path's length)
        # its path: a closed curve this long (a narrow one along a ledge), as far round as it has walked
        m = 512
        u = np.linspace(0, TAU, m, endpoint=False)
        if ledge:  # (to and fro along it, turning round over a step or two at each end: a long, narrow loop)
            r = min(TURN, self.length / TAU)
            run = (self.length - TAU * r) / 2
            at = u / TAU * self.length
            arc = np.clip(at - run, 0.0, math.pi * r) / r + np.clip(at - 2 * run - math.pi * r, 0.0, math.pi * r) / r
            way = np.where(at < run, at, np.where(at < run + math.pi * r, run, np.where(
                at < 2 * run + math.pi * r, 2 * run + math.pi * r - at, 0.0)))  # (how far along the straights)
            px, py = way - run / 2 + r * np.sin(arc), -r * np.cos(arc)
        else:
            px, py = np.zeros(m), np.zeros(m)
            for harmonic in (1, 2, 3):
                a = rng.normal(0, 1.0 / harmonic ** 1.5, 4)
                px += a[0] * np.cos(harmonic * u) + a[1] * np.sin(harmonic * u)
                py += a[2] * np.cos(harmonic * u) + a[3] * np.sin(harmonic * u)
        seg = np.hypot(np.diff(px, append=px[:1]), np.diff(py, append=py[:1]))
        k = self.length / max(float(seg.sum()), 1e-9)
        self.px, self.py = px * k, py * k
        self.cum = np.concatenate([[0.0], np.cumsum(seg * k)])
        self.looks = rng.random(8) * TAU
        self.flip = rng.random() < 0.5  # (which way round it goes)

    def _seg(self, t):
        for i in range(len(self.segs) - 1, -1, -1):
            if t >= self.segs[i][1]:
                return i
        return 0

    def walked_at(self, t):
        """How far it has walked at t (seconds into the loop): steady, starting and stopping over `ramp` seconds."""
        i = self._seg(t)
        kind, start, d, _ = self.segs[i]
        if kind != "walk":
            return self.walked[i] % self.length
        r = min(self.ramp, d / 2)
        x = t - start
        full = d - r  # (the distance this walk covers, in seconds at speed)
        if x < r:
            s = x * x / (2 * r)
        elif x > d - r:
            s = full - (d - x) ** 2 / (2 * r)
        else:
            s = x - r / 2
        return (self.walked[i] + self.speed * min(max(s, 0.0), full)) % self.length

    def where(self, t):
        """Where it is (units from its home, across and away) and which way it faces (radians)."""
        s = self.walked_at(t)
        if self.flip:
            s = (self.length - s) % self.length
        u = np.interp(s, self.cum, np.arange(len(self.cum))) % len(self.px)
        i = int(u)
        f = u - i
        j = (i + 1) % len(self.px)
        x = self.px[i] * (1 - f) + self.px[j] * f
        y = self.py[i] * (1 - f) + self.py[j] * f
        dx, dy = self.px[j] - self.px[i], self.py[j] - self.py[i]
        if self.flip:
            dx, dy = -dx, -dy
        return x, y, math.atan2(dy, dx)

    def reach(self, t):
        """How far down its head is at t (0 up .. 1 its beak on the ground): bending down, a few pecks, coming up."""
        i = self._seg(t)
        kind, start, d, n = self.segs[i]
        if kind != "peck":
            return 0.0
        x = (t - start) / d * (DOWN + n * PECKING + UP)  # (its own seconds)
        if x < DOWN:
            return 0.75 * float(_smooth(x / DOWN))
        if x > DOWN + n * PECKING:
            return 0.75 * float(_smooth((DOWN + n * PECKING + UP - x) / UP))
        ph = ((x - DOWN) / PECKING) % 1.0  # (a peck: quickly down, back up a little)
        dip = float(_smooth(ph / 0.3)) if ph < 0.3 else float(_smooth((1.0 - ph) / 0.55))
        return 0.75 + 0.25 * dip

    def pose(self, t):
        """Its pose at t: stepping and bobbing its head as it walks, pecking, looking round."""
        i = self._seg(t)
        kind, start, d, _ = self.segs[i]
        x = t - start
        ease = float(_smooth(min(x, d - x) / 0.2))  # (into and out of what it does)
        pose = Pose(pitch=-WALK_UP, tail=TAIL_WALK)
        if kind == "walk":
            s = self.walked_at(t)
            phase = (s / STRIDE) % 1.0
            # its head: held still where it is as the body goes on, then thrust forward (a bob a step: two a stride)
            step = (2 * phase) % 1.0
            thrust = float(_smooth(step / 0.3)) if step < 0.3 else 1.0
            bob = (thrust - step - 0.35) * STRIDE / 2 * 0.9 * ease
            pose.head = HEAD + np.array([bob, 0.0, -0.01 * ease * math.sin(TAU * step)])
            # its feet: one planted (going back under it as much as it goes on), the other lifted and going forward
            feet = []
            for side, off in ((1, 0.0), (-1, 0.5)):
                ph = (phase + off) % 1.0
                if ph < 0.5:
                    fx, lift = STRIDE * (0.25 - ph), 0.0
                else:
                    g = (ph - 0.5) / 0.5
                    fx, lift = STRIDE * (-0.25 + 0.5 * float(_smooth(g))), 0.028 * math.sin(math.pi * g)
                feet.append((fx * ease, side * 0.034, lift * ease))
            pose.feet = tuple(feet)
        elif kind == "peck":
            reach = self.reach(t)
            pose.pitch = -WALK_UP + (0.45 + WALK_UP) * reach  # (its body tips forward, its tail rises)
            pose.tail = TAIL_WALK * (1 - reach)
            pose.head = HEAD + (PECK - HEAD) * reach
            pose.head_pitch = 0.17 - 1.25 * reach
        else:  # standing: looking round now and then (a quick turn of the head, then still)
            k = int(x / 1.3)
            f = float(_smooth((x - k * 1.3) / 0.18))
            turn = 0.7 * (math.sin(self.looks[k % 8]) * f + math.sin(self.looks[(k - 1) % 8]) * (1 - f) * (k > 0))
            pose.head_yaw = turn * ease
            tall = float(_smooth(min(x, d - x) / 0.35))  # (standing up tall, breast out, then down again to go on)
            pose.tall = tall
            pose.pitch = -WALK_UP * (1 - tall)
            pose.tail = TAIL_WALK * (1 - tall)
            pose.head = HEAD + (HEAD_UP - HEAD) * tall
            pose.head_pitch = 0.17 - 0.07 * tall
        return pose


# --- the effect -------------------------------------------------------------------------------------------------------

DEFAULTS = {"mode": "ground", "coat": "mixed", "amount": 40, "size": 50, "depth": 50, "view": 40, "shadow": 60,
            "soft": 30, "busy": 50, "peck": 40, "speed": 50, "lit": 70}

SCHEMA = {
    "label": "Chim bồ câu",
    "note": "Chim bồ câu vẽ bằng app (không phải ảnh có sẵn): đi lại, gật gù, đứng ngó quanh, thỉnh thoảng cúi xuống "
            "mổ hạt. (1) Bấm Suốt video. (2) Kéo khung trắng tới chỗ chim đứng (sân, vỉa hè, đường đi), kéo góc cho "
            "khung phủ chỗ đó: chim chỉ ở trong khung. Đáy khung là gần, đỉnh khung là xa: chim ở xa nhỏ hơn. Trên gờ "
            "tường, mái nhà: chọn Kiểu Đậu trên gờ, kéo khung cho đáy khung nằm trên gờ. (3) Chỉnh Số chim, Cỡ chim, "
            "Màu lông; Nhìn từ trên xuống cho hợp góc máy; Độ mềm cho chim mềm như trong ảnh chụp; Mổ hạt cho chim cúi "
            "mổ ít hay nhiều. Chim sáng tối theo ánh sáng chỗ khung đứng trong video.",
    "params": [
        _choice("mode", "Kiểu", DEFAULTS["mode"], [("ground", "Đi lại, mổ hạt trên mặt đất"),
                                                   ("ledge", "Đậu trên gờ tường, mái nhà")], group="Kiểu"),
        _choice("coat", "Màu lông", DEFAULTS["coat"], [("mixed", "Nhiều màu như đàn chim phố"),
                                                       ("grey", "Xám xanh (bồ câu thường)"),
                                                       ("checker", "Xám có đốm"), ("pied", "Loang trắng"),
                                                       ("white", "Bồ câu trắng"), ("brown", "Nâu đỏ"),
                                                       ("dark", "Xám đen")], group="Kiểu"),
        _num("amount", "Số chim", DEFAULTS["amount"], group="Chim"),
        _num("size", "Cỡ chim", DEFAULTS["size"], group="Chim"),
        _num("depth", "Gần xa (chim xa nhỏ hơn)", DEFAULTS["depth"], group="Chim"),
        _num("view", "Nhìn từ trên xuống (ngang ← → cao)", DEFAULTS["view"], group="Chim"),
        _num("shadow", "Bóng dưới chân", DEFAULTS["shadow"], group="Chim"),
        _num("soft", "Độ mềm (nét ← → mềm như ảnh chụp)", DEFAULTS["soft"], group="Chim"),
        _num("busy", "Đi lại nhiều (đứng ← → đi)", DEFAULTS["busy"], group="Chuyển động"),
        _num("peck", "Mổ hạt (ít ← → nhiều)", DEFAULTS["peck"], group="Chuyển động"),
        _num("speed", "Tốc độ", DEFAULTS["speed"], group="Chuyển động"),
        _num("lit", "Sáng theo cảnh", DEFAULTS["lit"], group="Ánh sáng"),
        *_place(50)]}

DAYLIGHT = (np.array([1.15, 1.1, 1.04], np.float32), np.array([0.5, 0.54, 0.6], np.float32))  # key, ambient


def count(mode, amount):
    """How many birds an "amount" (0..100) makes."""
    return int(round(CAP[mode] * (amount / 100) ** 1.2))


class Pigeons(Effect):
    BOX = (16, 9)
    TURNS = False
    LIT = True
    PREVIEW = 1280
    VERSION = 4  # (2: one smooth body, pecking now and then; 3: rows of feathers, soft as velvet, sturdier legs;
    # 4: standing tall as a photograph's, its folded wings and their primaries, the neck's sheen round its foot)
    light_at = staticmethod(steam.light_at)
    rounded_light = staticmethod(steam.rounded_light)

    def __init__(self, *a, light=None):
        super().__init__(*a)
        p = self.p
        self.mode = p["mode"] if p["mode"] in CAP else "ground"
        rng = np.random.default_rng([self.seed, 4407])
        n = count(self.mode, p["amount"])
        coats = []
        for _ in range(CAP[self.mode]):
            if p["coat"] == "mixed":
                r, name = rng.random(), MIXED[-1][0]
                for nm, share in MIXED:
                    if r < share:
                        name = nm
                        break
                    r -= share
            else:
                name = p["coat"]
            coats.append(Coat(name, rng))
        self.coats = coats[:n]
        u = rng.random((CAP[self.mode], 8))[:n]
        self.u = u
        self.pace = 2.0 ** ((p["speed"] - 50) / 50)
        self.tilt = math.radians(_lerp(8, 50, p["view"] / 100))
        self.size = SMALLEST * (LARGEST / SMALLEST) ** (p["size"] / 100)  # (a bird's length, box heights, nearest)
        self.depth = p["depth"] / 100
        self.walkers, self.homes = [], []
        x0, y0, w, h = self.box()
        for i in range(n):
            scale = self._scale(u[i], h)
            room = (0.42 if self.mode == "ledge" else 0.16) * w / max(scale, 1e-9)  # (units)
            walker = Walker(np.random.default_rng([self.seed, 4407, i]), self.loop_s, p["busy"] / 100, self.pace,
                            self.mode == "ledge", room, p["peck"] / 100)
            self.walkers.append(walker)
            self.homes.append(self._home(walker, u[i], scale, w, h))
        self.light = self._light(light, p["lit"] / 100)  # (the scene's where the box stands, or daylight)

    def _light(self, light, k):
        key, ambient = DAYLIGHT
        side = -0.6
        if light is not None and k > 0:
            tone = np.asarray(light, np.float32)
            if tone.ndim != 1:
                tone = steam.light_at(tone, self.box(), self.w, self.h)
            key = key * (1 - k) + np.clip(tone[0:3], 0, 4) * 1.3 * k
            ambient = ambient * (1 - k) + np.clip(tone[3:6], 0, 4) * 0.95 * k
            side = side * (1 - k) + float(np.clip(tone[6], -1, 1)) * 0.8 * k
        way = np.array([side, -0.55, 0.85])
        return key.astype(np.float32), way / np.linalg.norm(way), ambient.astype(np.float32)

    # --- where each bird is now ---------------------------------------------------------------------------------------

    def _scale(self, u, h):
        """A bird's size: pixels a unit (the farther up the box, the smaller)."""
        depth = 0.0 if self.mode == "ledge" else u[1]  # (0 near: the box's bottom, 1 far: its top)
        return self.size * (1.0 - self.depth * 0.7 * depth) * h / LENGTH

    def _home(self, walker, u, scale, w, h):
        """Where a bird's path is (its middle: shares of the box across and down), all of it inside the box."""
        sy = scale * math.sin(self.tilt)
        lo_x, hi_x = float(walker.px.min()) * scale, float(walker.px.max()) * scale
        lo_y, hi_y = -float(walker.py.max()) * sy, -float(walker.py.min()) * sy
        reach = 0.6 * self.size * h  # (a bird's own size round its feet)
        left, right = reach - lo_x, w - reach - hi_x
        hx = left + (right - left) * u[0] if right > left else w / 2 - (lo_x + hi_x) / 2
        if self.mode == "ledge":  # (the near side of its path on the box's bottom edge, the far one a little up)
            return hx / w, 0.97 - hi_y / h
        top, bottom = 1.3 * reach - lo_y, h * 0.97 - hi_y  # (its head this high above its feet, standing tall)
        hy = bottom - (bottom - top) * u[1] * 0.85 if bottom > top else h / 2
        return hx / w, hy / h

    def _ground(self, t, x0, y0, w, h):
        """Each bird now: (feet x, y in the frame, its scale px a unit, heading, pose)."""
        out = []
        sec = (t % self.P) / self.fps
        for walker, u, home in zip(self.walkers, self.u, self.homes):
            scale = self._scale(u, h)
            dx, dy, heading = walker.where(sec)
            fx = x0 + home[0] * w + dx * scale
            fy = y0 + home[1] * h - dy * scale * math.sin(self.tilt)
            out.append((fx, fy, scale, heading, walker.pose(sec)))
        return out

    def draw(self, t, canvas, mask):
        box = self.box()
        x0, y0, w, h = box
        if h < 4 or not len(self.u):
            return
        birds = self._ground(t, x0, y0, w, h)
        order = sorted(range(len(birds)), key=lambda i: (birds[i][1], i))  # (far ones first: higher up)
        if self.p["shadow"] > 0:
            for i in order:
                self._shadow(canvas, birds[i], mask, box)
        for i in order:
            fx, fy, scale, heading, pose = birds[i]
            if scale * LENGTH < 3:
                continue
            ss = 3 if scale * LENGTH < 20 else 2
            rgb, alpha, (ax, ay) = draw_bird(pose, self.coats[i], heading, scale, self.tilt, self.light, ss)
            sigma = self.p["soft"] / 100 * SOFTEST * h  # (as soft as the video round it: a share of the box, so the
            # preview clip and the video are alike, and a big bird near the camera no softer than a small one)
            rgb, grow = _blur(rgb, sigma)
            alpha, _ = _blur(alpha, sigma)
            ix, iy = math.floor(fx) - ax - grow, math.floor(fy) - ay - grow
            inside = _inside(ix, iy, alpha.shape, *box)
            if inside is None:
                continue
            prgb = np.clip(rgb, 0.0, None) ** (1 / 2.2)  # (to the picture's light: premultiplied, about)
            a = np.clip(alpha, 0.0, 1.0)
            prgb = np.minimum(prgb, 1.0) * np.where(a > 0, a ** (1 - 1 / 2.2), 0.0)[..., None]
            canvas.over_rgba(ix, iy, (prgb * inside[..., None]).astype(np.float32), (a * inside).astype(np.float32),
                             mask)

    def _shadow(self, canvas, bird, mask, box):
        """Its shadow on the ground: dark right under its body, softer and wider away from the light."""
        fx, fy, scale, heading, pose = bird
        key, way, ambient = self.light
        sun = float(key @ LUMA) / max(float((key + ambient) @ LUMA), 1e-6)  # (sunlight: darker shadows)
        st = max(0.12, math.sin(self.tilt))
        along, across = abs(math.cos(heading)), abs(math.sin(heading))
        # (the body's footprint as the camera sees it: long across the picture when the bird faces sideways)
        rx0 = scale * (0.2 * along + 0.11 * across)
        ry0 = scale * st * (0.11 * along + 0.2 * across)
        shift = 0.18 * scale * np.array([-way[0], way[1] * st]) / max(way[2], 0.3)  # (away from the light)
        for ox, oy, grow, depth in ((0.0, 0.0, 1.0, 0.5 + 0.2 * sun), (shift[0], shift[1], 1.6, 0.25 + 0.35 * sun)):
            rx, ry = rx0 * grow, max(ry0 * grow, 1.0)
            if rx < 1:
                continue
            cx, cy = fx - 0.05 * scale * math.cos(heading) + ox, fy + oy
            sx0, sy0 = math.floor(cx - 2 * rx), math.floor(cy - 2 * ry)
            xs = np.arange(int(4 * rx) + 2) + sx0 + 0.5 - cx
            ys = np.arange(int(4 * ry) + 2) + sy0 + 0.5 - cy
            d = (xs[None, :] / rx) ** 2 + (ys[:, None] / ry) ** 2
            inside = _inside(sx0, sy0, d.shape, *box)
            if inside is None:
                continue
            cover = (np.exp(-2.4 * d) * depth * self.p["shadow"] / 100 * inside).astype(np.float32)
            canvas.over(sx0, sy0, cover, np.zeros(3, np.float32), 0.6, mask)
