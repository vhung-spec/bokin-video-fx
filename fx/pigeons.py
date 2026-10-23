"""Pigeons ("Chim bồ câu"): rock doves walking about on the ground or on a ledge, pecking, looking round, or flying
in a flock or past (a placed effect: Effect.BOX; drawn over the picture, lit by the scene; see effects.py for how
every effect loops).

A pigeon is drawn by code, not from a picture: ellipsoids (its body, breast, belly, neck, head, beak, cere, tail, the
tips of its folded wings, its legs and toes; its spread wings in flight) seen by a far camera (straight rays) from a
little above. Every frame each bird is traced in its own pose (where its head is, how its body tips, where its feet
are, how its wings beat); where a ray meets it its feathers are painted on: a rock dove's grey, its pale folded wings
with two black bars and dark tips, the green and purple sheen of its neck (as the feathers face the camera), the
orange eye, the white cere, the black band at the end of its tail, its red feet (or a white dove's, a brown one's, a
dark one's), with a fine feather texture. Where its soft parts meet their normals are blended: a smooth bird, not
balls stuck together. It is lit by the scene where its box stands (as steam is: light_at), and casts a soft shadow
under its feet.

Every bird keeps its own day over the loop: walking a while round a closed path of its own (once a loop: back where
it started), stopping to peck, standing and looking round; as it walks its head is held still and thrust forward each
step, as pigeons do, and its feet step as far as it goes. In the air: round a loop (whole laps a loop), its wings
beating (whole beats a loop) or held up in a V as it glides, banking into the turn; or across the box, k passes a
loop.
"""
import math

import numpy as np

from . import steam
from .effects import Effect, _choice, _lerp, _num, _place

TAU = 2 * math.pi
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
LENGTH = 0.86  # a pigeon's length from beak to tail, in the units of its parts (its "size" is this many pixels)
CAP = {"ground": 30, "ledge": 14, "flock": 40, "pass": 16}  # birds at most, each way of being
GLIDE = math.radians(28)  # wings held up in a V while gliding
REACH = 0.92  # how far a flying bird's wings and body reach from its middle, in its lengths


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


# --- its feathers ----------------------------------------------------------------------------------------------------

COATS = {  # sRGB: a rock dove (blue bar), a white dove, a brown (ash-red) one, a dark one
    "grey": {"body": (118, 126, 140), "wing": (160, 166, 176), "bar": (32, 32, 36), "prim": (68, 70, 78),
             "head": (100, 108, 124), "breast": (116, 108, 128), "neck": (74, 84, 92), "green": (64, 128, 94),
             "purple": (124, 72, 132), "tail": (104, 110, 124), "band": (32, 32, 36), "rump": (176, 180, 188),
             "under": (196, 200, 208), "beak": (48, 44, 46), "cere": (232, 230, 224), "iris": (222, 110, 30),
             "feet": (196, 84, 86), "sheen": 1.0},
    "white": {"body": (234, 234, 230), "wing": (240, 240, 236), "bar": (240, 240, 236), "prim": (226, 226, 222),
              "head": (238, 238, 234), "breast": (236, 234, 232), "neck": (232, 232, 228), "green": (232, 236, 228),
              "purple": (236, 230, 234), "tail": (236, 236, 232), "band": (236, 236, 232), "rump": (240, 240, 236),
              "under": (240, 240, 236), "beak": (210, 160, 150), "cere": (242, 236, 230), "iris": (40, 26, 22),
              "feet": (214, 112, 112), "sheen": 0.0},
    "brown": {"body": (138, 104, 90), "wing": (184, 156, 138), "bar": (116, 70, 54), "prim": (128, 100, 88),
              "head": (148, 116, 102), "breast": (146, 104, 96), "neck": (140, 102, 94), "green": (124, 118, 96),
              "purple": (156, 100, 120), "tail": (162, 134, 118), "band": (140, 96, 80), "rump": (196, 176, 162),
              "under": (206, 194, 184), "beak": (190, 154, 140), "cere": (234, 228, 220), "iris": (222, 110, 30),
              "feet": (200, 90, 90), "sheen": 0.5},
    "dark": {"body": (60, 62, 68), "wing": (76, 78, 86), "bar": (32, 32, 36), "prim": (42, 44, 48),
             "head": (54, 56, 62), "breast": (62, 60, 68), "neck": (46, 50, 54), "green": (48, 120, 82),
             "purple": (110, 60, 120), "tail": (58, 60, 66), "band": (28, 28, 32), "rump": (72, 74, 80),
             "under": (96, 98, 104), "beak": (38, 36, 38), "cere": (214, 212, 206), "iris": (222, 110, 30),
             "feet": (190, 80, 82), "sheen": 1.2},
}
MIXED = (("grey", 0.55), ("dark", 0.13), ("brown", 0.12), ("white", 0.2))  # a town's flock


class Coat:
    """A bird's feathers: its coat's colours (linear light), a checker on its wings or not."""

    def __init__(self, name, rng):
        self.c = {k: (_lin(v) if isinstance(v, tuple) else v) for k, v in COATS[name].items()}
        self.checker = name == "grey" and rng.random() < 0.3  # (a blue checker: dark spots on its wings)
        self.tone = 0.93 + 0.14 * rng.random()  # (each bird a little lighter or darker)
        self.waves = rng.normal(0.0, 1.0, (6, 3)) * 60.0  # (its fine feather texture)
        self.phases = rng.random(6) * TAU

    def grain(self, p, fp):
        """A fine texture of feathers at points p (about 0.9..1.1): its waves longer than a few pixels (`fp`: a
        pixel's width, units: on a small bird it would only shimmer)."""
        out = np.zeros(len(p), np.float32)
        for w, ph in zip(self.waves, self.phases):
            k = _fade(float(np.linalg.norm(w)), fp)
            if k > 0:
                out += k * np.sin(p @ w + ph).astype(np.float32)
        return 1.0 + 0.05 * out / 2.4


# --- its shape and poses ----------------------------------------------------------------------------------------------

HIP = np.array([-0.02, 0.0, 0.13])  # the body tips about this (pecking)
NECK = np.array([0.13, 0.0, 0.37])  # where the neck leaves the breast (standing)
HEAD = np.array([0.198, 0.0, 0.55])  # its head's middle (standing)
# the body standing: name, centre, radii, pitch (degrees, the nose up)
BODY = (("belly", (-0.03, 0.0, 0.17), (0.17, 0.1, 0.085), 12), ("body", (-0.045, 0.0, 0.27), (0.27, 0.122, 0.13), 21),
        ("breast", (0.1, 0.0, 0.31), (0.145, 0.116, 0.152), 18), ("vent", (-0.19, 0.0, 0.17), (0.11, 0.07, 0.055), -14),
        ("tail", (-0.34, 0.0, 0.17), (0.17, 0.058, 0.016), -20), ("tips", (-0.3, 0.0, 0.212), (0.15, 0.06, 0.012), -18))
SOFT = ("belly", "body", "breast", "vent", "neck", "head")  # (their normals blended where they meet)


class Pose:
    """How a bird stands now: its body tipped forward (pitch, radians), its head's middle (bird frame), the head's
    own pitch and turn, its feet (bird frame, each: x, y, lift), or flying (its wings' beat, how far they are spread,
    its tail spread)."""

    def __init__(self, pitch=0.0, head=None, head_pitch=0.17, head_yaw=0.0, feet=None, flying=False, beat=0.0,
                 roll=0.0):
        self.pitch, self.head_pitch, self.head_yaw = pitch, head_pitch, head_yaw
        self.head = HEAD.copy() if head is None else np.asarray(head, np.float64)
        self.feet = feet if feet is not None else ((0.0, 0.036, 0.0), (0.0, -0.036, 0.0))
        self.flying, self.beat, self.roll = flying, beat, roll


def parts(pose):
    """A bird's parts in a pose: (name, centre, rotation (columns: its axes), radii), in the bird frame (x forward, y
    left, z up; its feet on the ground at 0)."""
    if pose.flying:
        return _flying(pose)
    out = []
    tip = _rot(pitch=-pose.pitch)  # (the body tipped forward about the hip)
    for name, c, r, pitch in BODY:
        centre = HIP + tip @ (np.array(c) - HIP)
        out.append((name, centre, tip @ _rot(pitch=math.radians(pitch)), np.array(r)))
    base = HIP + tip @ (NECK - HIP)
    head = pose.head
    turn = _rot(yaw=pose.head_yaw, pitch=pose.head_pitch)
    span = float(np.linalg.norm(head - base))
    out.append(("neck", (base + head) / 2 - np.array([0.012, 0, 0]), _along(base, head),
                np.array([0.088, 0.076, max(0.09, span / 2 + 0.035)]) * (1.0 - 0.12 * max(0.0, span - 0.2))))
    out.append(("head", head, turn, np.array([0.075, 0.058, 0.064])))
    out.append(("beak", head + turn @ np.array([0.083, 0.0, -0.015]), turn @ _rot(pitch=-0.45),
                np.array([0.044, 0.012, 0.013])))
    out.append(("cere", head + turn @ np.array([0.057, 0.0, 0.0]), turn @ _rot(pitch=-0.45),
                np.array([0.02, 0.016, 0.012])))
    for side, foot in zip((1, -1), pose.feet):
        hip = np.array([0.0, side * 0.032, 0.11])
        f = np.array(foot, np.float64)
        knee = f + np.array([0.0, 0.0, 0.02])
        out.append(("leg", (hip + knee) / 2, _along(knee, hip),
                    np.array([0.016, 0.016, max(0.03, float(np.linalg.norm(hip - knee)) / 2 + 0.01)])))
        out.append(("toe", f + np.array([0.03, 0.0, 0.007]), _rot(yaw=side * 0.14), np.array([0.042, 0.011, 0.007])))
    return out


def _plate(name, p0, p1, chord, thick):
    """A flat part from p0 to p1 (a wing, a feather): its chord across, as level as it can be (its z axis up)."""
    y = (p1 - p0) / max(float(np.linalg.norm(p1 - p0)), 1e-9)
    x = np.array([1.0, 0.0, 0.0]) - y[0] * y
    if np.linalg.norm(x) < 1e-6:
        x = np.array([0.0, 0.0, 1.0]) - y[2] * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    if z[2] < 0:
        x, z = -x, -z
    return name, (p0 + p1) / 2, np.stack([x, y, z], 1), np.array([chord, float(np.linalg.norm(p1 - p0)) / 2 + 0.01,
                                                                   thick])


def _flying(pose):
    """A bird in the air: its body level, its neck short, its tail spread, its legs tucked, its wings out (beat:
    the angle of its arms above the level, radians; its hands bent further)."""
    body = np.array([0.0, 0.0, 0.3])
    level = _rot(pitch=-0.12)
    out = [("body", body + [-0.04, 0, 0], level, np.array([0.25, 0.105, 0.095])),
           ("breast", body + [0.09, 0, -0.005], level, np.array([0.13, 0.1, 0.1])),
           ("vent", body + [-0.17, 0, -0.03], level, np.array([0.1, 0.065, 0.05])),
           ("tail", body + [-0.33, 0, -0.015], _rot(pitch=0.06), np.array([0.16, 0.095, 0.013]))]
    head = body + np.array([0.27, 0.0, 0.07])
    base = body + np.array([0.15, 0.0, 0.04])
    out.append(("neck", (base + head) / 2, _along(base, head), np.array([0.08, 0.07, 0.09])))
    turn = _rot(pitch=0.1)
    out.append(("head", head, turn, np.array([0.075, 0.058, 0.064])))
    out.append(("beak", head + turn @ np.array([0.083, 0.0, -0.012]), turn @ _rot(pitch=-0.3),
                np.array([0.044, 0.012, 0.013])))
    out.append(("cere", head + turn @ np.array([0.057, 0.0, 0.002]), turn @ _rot(pitch=-0.3),
                np.array([0.02, 0.016, 0.012])))
    for side in (1, -1):  # (each wing nearly as long as the bird: its span about twice its length)
        shoulder = body + np.array([0.04, side * 0.06, 0.05])
        a = pose.beat
        arm = np.array([-0.08, side * math.cos(a), math.sin(a)])
        arm /= np.linalg.norm(arm)
        wrist = shoulder + arm * 0.3
        b = a * 1.3 - 0.08  # (the hand bends further at each end of the beat)
        hand = np.array([-0.3, side * math.cos(b), math.sin(b)])
        hand /= np.linalg.norm(hand)
        back = np.array([-1.0, 0.0, 0.0]) - (np.array([-1.0, 0.0, 0.0]) @ hand) * hand  # (back, in the hand's plane)
        back /= np.linalg.norm(back)
        root = shoulder - arm * 0.1 + np.array([-0.05, 0.0, 0.0])  # (its root along the body, broad)
        out.append(_plate("arm", root, wrist + arm * 0.02 + np.array([-0.04, 0.0, 0.0]), 0.17, 0.02))
        out.append(_plate("coverts", wrist - hand * 0.06 + np.array([-0.03, 0.0, 0.0]), wrist + hand * 0.16, 0.1,
                          0.014))
        for k, (length, swing) in enumerate(((0.4, 0.0), (0.39, 0.13), (0.36, 0.27), (0.31, 0.42), (0.26, 0.58))):
            way = hand * math.cos(swing) + back * math.sin(swing)  # (the primaries fanned from the wrist)
            out.append(_plate("hand" if k else "tip", wrist + back * 0.02, wrist + way * length, 0.032, 0.006))
    if pose.roll:
        bank = _rot(roll=pose.roll)
        out = [(n, body + bank @ (c - body), bank @ r, rad) for n, c, r, rad in out]
    return out


# --- drawing a bird ---------------------------------------------------------------------------------------------------

def _fade(freq, fp):
    """How much of a pattern of `freq` radians a unit to keep where a pixel is `fp` units wide: all of it while its
    waves are more than 4 pixels long, none below about 2.6 (it would only shimmer)."""
    return float(_smooth((2.4 - freq * fp) / 0.8))


def _albedo(name, part, p, n, view, coat, pose, fp):
    """The feathers' colour (linear light) at bird-frame points p of a part (its centre, rotation, radii), facing n,
    the camera's way `view`; `fp`: a pixel's width (units), its patterns blurred to it (one ray a pixel inside the
    bird: draw_bird)."""
    c = coat.c
    count = len(p)
    if name in ("leg", "toe"):
        return np.tile(c["feet"], (count, 1))
    if name == "beak":
        return np.tile(c["beak"], (count, 1))
    if name == "cere":
        return np.tile(c["cere"], (count, 1))
    centre, rot, radii = part
    local = ((p - centre) @ rot) / radii  # (on its unit sphere)
    if name == "tips":
        return np.tile(c["prim"], (count, 1))
    if name == "tail":
        band = _smooth((-local[:, 0] - 0.62) / 0.12)
        return c["tail"] * (1 - band[:, None]) + c["band"] * band[:, None]
    if name in ("arm", "coverts", "hand", "tip"):  # (spread wings: pale under, grey above, two bars on the arm,
        # the primaries dark)
        up = n @ rot[:, 2] > 0  # (its z axis points up from its top)
        top = np.tile(c["wing"], (count, 1))
        if name == "arm":
            bars = _smooth((0.12 - np.abs(np.abs(local[:, 0] + 0.15) - 0.32)) / max(0.05, fp / radii[0]))
            top = top * (1 - bars[:, None]) + c["bar"] * bars[:, None]
        elif name in ("hand", "tip"):
            dark = 0.5 + 0.5 * _smooth((local[:, 1] * np.sign(rot[1, 1]) + 0.2) / 0.6)
            top = top * (1 - dark[:, None]) + c["prim"] * dark[:, None]
        under = np.tile(c["under"], (count, 1))
        return np.where(up[:, None], top, under)
    if name == "head":
        out = np.tile(c["head"], (count, 1))
        edge = max(0.05, fp / 0.06)  # (in the head's own size: a small eye blurred, not a flickering dot)
        for side in (1.0, -1.0):  # its eyes: orange (a white dove's dark), a pale ring, a black pupil
            d = np.sqrt((local[:, 0] - 0.42) ** 2 + (local[:, 2] - 0.25) ** 2 + 0.4 * (local[:, 1] - side * 0.86) ** 2)
            on = (side * local[:, 1] > 0.3)[:, None]
            for size, colour in ((0.3, _lin((214, 208, 194))), (0.22, c["iris"]), (0.1, _lin((10, 8, 8)))):
                k = (_smooth((size - d) / edge) * on[:, 0])[:, None]
                out = out * (1 - k) + colour * k
        return out
    # the body: standing (its frame before it tipped forward): breast, iridescent neck, folded wings, pale rump
    q = p if pose.flying else HIP + (p - HIP) @ _rot(pitch=-pose.pitch)
    x, z = q[:, 0], q[:, 2]
    out = np.tile(c["body"], (count, 1))
    breast = _smooth((x - 0.04) / 0.08) * _smooth((0.42 - z) / 0.08)
    out = out * (1 - breast[:, None]) + c["breast"] * breast[:, None]
    if name == "neck" or not pose.flying:
        neck = np.ones(count) if name == "neck" else _smooth((z - 0.36) / 0.05) * _smooth((x - 0.02) / 0.06)
        facing = np.abs(n @ view)  # (green seen straight on, purple at a slant)
        sheen = c["neck"] + c["sheen"] * (c["green"] - c["neck"]) * facing[:, None] ** 2 + c["sheen"] * (
            c["purple"] - c["neck"]) * (1 - facing[:, None]) ** 2
        out = out * (1 - neck[:, None]) + np.clip(sheen, 0, 1) * neck[:, None]
    if not pose.flying and name in ("body", "belly", "vent", "breast"):
        u, v = (x + 0.1) / 0.25, (z - 0.305) / 0.09
        wing = _smooth((1 - (u * u + v * v)) / 0.15) * _smooth((n[:, 2] + 0.5) / 0.4)
        bars = np.zeros(count)
        for at in (-0.04, -0.12):
            s = x - at + 0.3 * (z - 0.3)
            bars = np.maximum(bars, _smooth((0.012 - np.abs(s)) / max(0.005, fp)))
        bars *= _smooth((v + 0.5) / 0.3) * _smooth((0.8 - v) / 0.2)
        prim = _smooth((-x - 0.2) / 0.06)
        cover = c["wing"] * (1 - prim[:, None]) + c["prim"] * prim[:, None]
        if coat.checker:  # (dark spots on the coverts)
            spot = _smooth(0.4 - np.abs(np.sin(x * 70) * np.sin(z * 70 + 1))) * (1 - prim) * (u > -0.6)
            spot *= 0.6 * _fade(99.0, fp)
            cover = cover * (1 - spot[:, None]) + c["bar"] * spot[:, None]
        cover = cover * (1 - bars[:, None]) + c["bar"] * bars[:, None]
        rows = 0.97 - 0.03 * _fade(187.0, fp) * np.cos(2 * (x * 90 + 25 * z))  # (rows of feathers)
        out = out * (1 - wing[:, None]) + (cover * rows[:, None]) * wing[:, None]
        rump = _smooth((-x - 0.2) / 0.04) * _smooth((n[:, 2] - 0.3) / 0.3) * (1 - wing)
        out = out * (1 - 0.6 * rump[:, None]) + c["rump"] * (0.6 * rump[:, None])
    return out


DETAIL = 48  # (a bird at least this long, px: one ray a pixel inside it, ss x ss only where an edge crosses a pixel)


def _meet(ps, spans, rx, ry, scale, turn, up, fwd):
    """Where rays through picture points (rx, ry: pixels from its feet, right and down; a far camera) first meet a
    bird's parts, each tried only where it lies on the picture: which part (-1: none), and for the rays that meet
    one (their indices): the points (bird frame) and the part's normals there (not of unit length)."""
    m = len(rx)
    o = (rx / scale)[:, None] * np.array([1.0, 0.0, 0.0]) - (ry / scale)[:, None] * up - 3.0 * fwd
    ob = o @ turn  # (into the bird's frame)
    fb = turn.T @ fwd
    best = np.full(m, np.inf)
    which = np.full(m, -1)
    own = np.zeros((m, 3))
    for k, (name, c, rot, r) in enumerate(ps):
        idx = np.flatnonzero((rx >= spans[k, 0]) & (rx < spans[k, 2]) & (ry >= spans[k, 1]) & (ry < spans[k, 3]))
        if not len(idx):
            continue
        q = ((ob[idx] - c) @ rot) / r
        d = (fb @ rot) / r
        a = float(d @ d)
        b = q @ d
        disc = b * b - a * ((q * q).sum(1) - 1.0)
        hit = disc > 0
        if not hit.any():
            continue
        t = (-b[hit] - np.sqrt(disc[hit])) / a
        at_idx = idx[hit]
        near = t < best[at_idx]
        if not near.any():
            continue
        j = at_idx[near]
        best[j] = t[near]
        which[j] = k
        own[j] = ((q[hit][near] + t[near, None] * d) / r) @ rot.T
    seen = np.flatnonzero(which >= 0)
    return which, seen, ob[seen] + best[seen, None] * fb, own[seen]


def _shade(ps, p, which, own, fb, turn, coat, pose, light, fp):
    """The colour (linear light) where rays meet a bird: points p (bird frame) on parts `which`, their normals."""
    nrm = own / np.maximum(np.linalg.norm(own, axis=1, keepdims=True), 1e-12)
    soft = np.isin(which, [k for k, part in enumerate(ps) if part[0] in SOFT])
    acc = np.zeros_like(nrm)
    for name, c, rot, r in ps:  # (where soft parts meet: their normals blended)
        if name not in SOFT:
            continue
        q = ((p - c) @ rot) / r
        ln = np.sqrt((q * q).sum(1))
        g = (q / r) @ rot.T
        g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
        acc += (np.exp(-np.maximum(ln - 1, 0.0) * 14) * (ln < 1.25))[:, None] * g
    blend = acc + 0.3 * nrm
    blend /= np.maximum(np.linalg.norm(blend, axis=1, keepdims=True), 1e-12)
    nrm = np.where(soft[:, None], blend, nrm)
    view = -fb
    alb = np.zeros((len(p), 3), np.float32)
    for k, (name, c, rot, r) in enumerate(ps):
        j = np.flatnonzero(which == k)
        if len(j):
            alb[j] = _albedo(name, (c, rot, r), p[j], nrm[j], view, coat, pose, fp)
    alb *= coat.tone * coat.grain(p, fp)[:, None]
    key, way, ambient = light
    nw = nrm @ turn.T  # (the scene's frame)
    lam = np.clip((nw @ way + 0.25) / 1.25, 0.0, 1.0)  # (feathers: a little light wraps round)
    sky = 0.45 + 0.55 * np.clip(0.5 + 0.5 * nw[:, 2], 0, 1)
    rim = np.clip(1 - np.abs(nw @ (turn @ view)), 0, 1) ** 4 * 0.25  # (a little light at its edges)
    return alb * (ambient * (sky + rim)[:, None] + key * lam[:, None])


def draw_bird(pose, coat, yaw, scale, tilt, light, ss):
    """A bird seen by a far camera tilted down by `tilt` (radians), heading `yaw` (0: to the right of the picture,
    pi/2: away), `scale` pixels a unit: its picture (premultiplied linear RGB, alpha) and where its feet are in it
    (x, y pixels). `light`: (key colour, key way, ambient colour) linear, the way in the scene (x right, y away, z up).
    A bird DETAIL pixels long or more: one ray a pixel, `ss` x `ss` where its edge or the edge of a part crosses a
    pixel; a smaller one: `ss` x `ss` everywhere."""
    ps = parts(pose)
    ct, st = math.cos(tilt), math.sin(tilt)
    fwd = np.array([0.0, ct, -st])
    up = np.array([0.0, st, ct])
    turn = _rot(yaw=yaw)
    fb = turn.T @ fwd
    # where each part lies on the picture (pixels from its feet: right, down): its middle +- its largest radius
    spans = []
    for _, c, _, r in ps:
        w = turn @ c
        m = float(r.max()) * scale + 1.5
        spans.append((w[0] * scale - m, -(w @ up) * scale - m, w[0] * scale + m, -(w @ up) * scale + m))
    spans = np.array(spans)
    x0, y0 = math.floor(spans[:, 0].min()), math.floor(spans[:, 1].min())
    w, h = max(1, math.ceil(spans[:, 2].max()) - x0), max(1, math.ceil(spans[:, 3].max()) - y0)
    fp = 1.0 / scale
    n = ss * ss
    sub = (np.arange(ss) + 0.5) / ss
    if scale * LENGTH >= DETAIL:  # (a ray through each pixel's middle: its edges where what it meets changes)
        which, seen, p, own = _meet(ps, spans, np.tile(np.arange(w) + 0.5 + x0, h),
                                    np.repeat(np.arange(h) + 0.5 + y0, w), scale, turn, up, fwd)
        grid = which.reshape(h, w)
        edge = np.zeros((h, w), bool)
        for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
            a0, a1 = (0, w - dx) if dx >= 0 else (-dx, w)
            b0, b1 = a0 + dx, a1 + dx
            diff = grid[:h - dy, a0:a1] != grid[dy:, b0:b1]
            edge[:h - dy, a0:a1] |= diff
            edge[dy:, b0:b1] |= diff
        flat = edge.reshape(-1)
        keep = ~flat[seen]
        inner, p, own, which = seen[keep], p[keep], own[keep], which[seen[keep]]
        pixels = np.flatnonzero(flat)
    else:
        inner = np.zeros(0, np.int64)
        p, own, which = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
        pixels = np.arange(w * h)
    # ss x ss rays through each pixel on an edge (or every pixel of a small bird)
    ey, ex = np.divmod(pixels, w)
    rx = (ex[:, None] + x0 + np.tile(sub, ss)[None, :]).reshape(-1)
    ry = (ey[:, None] + y0 + np.repeat(sub, ss)[None, :]).reshape(-1)
    sw, sseen, sp, sown = _meet(ps, spans, rx, ry, scale, turn, up, fwd)
    rgb_all = _shade(ps, np.concatenate([p, sp]), np.concatenate([which, sw[sseen]]), np.concatenate([own, sown]),
                     fb, turn, coat, pose, light, fp)
    rgb = np.zeros((w * h, 3), np.float32)
    alpha = np.zeros(w * h, np.float32)
    rgb[inner] = rgb_all[:len(inner)]
    alpha[inner] = 1.0
    at = pixels[sseen // n]  # (each edge ray's pixel: their mean, premultiplied)
    for ch in range(3):
        rgb[:, ch] += np.bincount(at, rgb_all[len(inner):, ch], w * h).astype(np.float32) / n
    alpha += np.bincount(at, None, w * h).astype(np.float32) / n
    return rgb.reshape(h, w, 3), alpha.reshape(h, w), (-x0, -y0)


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
TURN = 0.1  # (on a ledge: the radius it turns round at each end, units)
WALK = 0.5  # (its walking speed, units a second, at "speed" 50: about three steps a second)
PECK = np.array([0.32, 0.0, 0.123])  # (its head's middle as its beak touches the ground)


class Walker:
    """One bird on the ground over the loop (`loop` seconds): walking round a closed path of its own (once a loop),
    pecking, standing and looking round. `busy`: 0 (mostly pecking, standing) .. 1 (walking most of the time);
    `ledge`: walking to and fro along a line (a narrow loop) rather than about."""

    def __init__(self, rng, loop, busy, pace, ledge=False, room=None):
        """`room`: the farthest its path may go from its home (units): a long walk goes round more than once."""
        self.loop = loop
        segs, at = [], 0.0
        walk = rng.random() < 0.6
        while at < loop * 1.02 or len(segs) < 2:
            if walk:
                kind, d = "walk", rng.uniform(1.2, 3.5) * (0.6 + 0.8 * busy)
            else:
                kind = "peck" if rng.random() < 0.65 - 0.2 * busy else "stand"
                d = rng.uniform(1.0, 3.5) * (1.4 - 0.8 * busy)
            segs.append([kind, at, d])
            at += d
            walk = not walk
        if segs[0][0] == segs[-1][0] and len(segs) > 2:  # (the loop closes: the last runs on into the first)
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
        for kind, start, d in segs:
            self.walked.append(total)
            if kind == "walk":
                total += speed * (d - min(self.ramp, d / 2))
        self.speed = speed
        laps = 1
        if room is not None:  # (round a smaller path more than once if its walk is long: the longest path that keeps
            # within `room` of home: along a ledge, to and fro; about, a round path)
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
        self.pecks = rng.uniform(2.0, 3.0)  # (pecks a second)
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
        kind, start, d = self.segs[i]
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

    def pose(self, t):
        """Its pose at t: stepping and bobbing its head as it walks, pecking, looking round."""
        i = self._seg(t)
        kind, start, d = self.segs[i]
        x = t - start
        ease = float(_smooth(min(x, d - x) / 0.2))  # (into and out of what it does)
        pose = Pose()
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
                    fx, lift = STRIDE * (-0.25 + 0.5 * float(_smooth(g))), 0.03 * math.sin(math.pi * g)
                feet.append((fx * ease, side * 0.036, lift * ease))
            pose.feet = tuple(feet)
        elif kind == "peck":
            ph = (x * self.pecks) % 1.0
            down = float(_smooth(ph / 0.35)) if ph < 0.35 else float(_smooth((1.0 - ph) / 0.45))
            reach = ease * (0.55 + 0.45 * down)
            pose.pitch = 0.42 * reach
            pose.head = HEAD + (PECK - HEAD) * (reach * (0.65 + 0.35 * down))
            pose.head_pitch = 0.17 - 1.2 * reach
        else:  # standing: looking round now and then
            k = int(x / 1.3)
            turn = math.sin(self.looks[k % 8]) * 0.7
            pose.head_yaw = turn * ease
            pose.head = HEAD + np.array([0.0, 0.0, 0.01 * ease])
        return pose


# --- the effect -------------------------------------------------------------------------------------------------------

DEFAULTS = {"mode": "ground", "coat": "mixed", "amount": 40, "size": 60, "depth": 50, "busy": 50, "speed": 50,
            "view": 50, "shadow": 50, "lit": 70, "way": "both"}
ON_THE_GROUND = {"mode": ["ground", "ledge"]}

SCHEMA = {
    "label": "Chim bồ câu",
    "note": "Chim bồ câu vẽ bằng app (không phải ảnh có sẵn): đi lại, gật gù, mổ thức ăn, đứng ngó quanh, hoặc bay. "
            "(1) Bấm Suốt video. (2) Kéo khung trắng tới chỗ chim đứng (sân, vỉa hè, quảng trường) hoặc chỗ chim bay "
            "(bầu trời), kéo góc cho khung phủ chỗ đó: chim chỉ ở trong khung. Đáy khung là gần, đỉnh khung là xa: "
            "chim ở xa nhỏ hơn. Trên gờ tường, mái nhà: chọn Kiểu Đậu trên gờ, kéo khung cho đáy khung nằm trên gờ. "
            "(3) Chỉnh Số chim, Cỡ chim, Màu lông; Nhìn từ trên xuống cho hợp góc máy. Chim sáng tối theo ánh sáng "
            "chỗ khung đứng trong video.",
    "params": [
        _choice("mode", "Kiểu", DEFAULTS["mode"], [("ground", "Đi lại, mổ thức ăn trên sân"),
                                                   ("ledge", "Đậu trên gờ tường, mái nhà"),
                                                   ("flock", "Bay lượn thành đàn"), ("pass", "Bay ngang qua")],
                group="Kiểu"),
        _choice("coat", "Màu lông", DEFAULTS["coat"], [("mixed", "Nhiều màu như đàn chim phố"),
                                                       ("grey", "Xám xanh (bồ câu thường)"),
                                                       ("white", "Bồ câu trắng"), ("brown", "Nâu đỏ"),
                                                       ("dark", "Xám đen")], group="Kiểu"),
        _num("amount", "Số chim", DEFAULTS["amount"], group="Chim"),
        _num("size", "Cỡ chim", DEFAULTS["size"], group="Chim"),
        _num("depth", "Gần xa (chim xa nhỏ hơn)", DEFAULTS["depth"], group="Chim"),
        _num("view", "Nhìn từ trên xuống (ngang ← → cao)", DEFAULTS["view"], group="Chim", show=ON_THE_GROUND),
        _num("shadow", "Bóng dưới chân", DEFAULTS["shadow"], group="Chim", show=ON_THE_GROUND),
        _num("busy", "Đi lại nhiều (mổ, đứng ← → đi)", DEFAULTS["busy"], group="Chuyển động", show=ON_THE_GROUND),
        _num("speed", "Tốc độ", DEFAULTS["speed"], group="Chuyển động"),
        _choice("way", "Hướng bay", DEFAULTS["way"], [("both", "Cả hai chiều"), ("right", "Sang phải"),
                                                     ("left", "Sang trái")], group="Chuyển động",
                show={"mode": ["pass"]}),
        _num("lit", "Sáng theo cảnh", DEFAULTS["lit"], group="Ánh sáng"),
        *_place(50)]}

DAYLIGHT = (np.array([1.25, 1.2, 1.12], np.float32), np.array([0.5, 0.55, 0.62], np.float32))  # key, ambient


def count(mode, amount):
    """How many birds an "amount" (0..100) makes."""
    return int(round(CAP[mode] * (amount / 100) ** 1.2))


class Pigeons(Effect):
    BOX = (16, 9)
    TURNS = False
    LIT = True
    PREVIEW = 1280
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
                name = p["coat"] if p["coat"] in COATS else "grey"
            coats.append(Coat(name, rng))
        self.coats = coats[:n]
        u = rng.random((CAP[self.mode], 8))[:n]
        self.u = u
        self.pace = 2.0 ** ((p["speed"] - 50) / 50)
        self.tilt = math.radians(_lerp(8, 50, p["view"] / 100)) if self.mode in ("ground", "ledge") else \
            math.radians(6)
        self.size = _lerp(0.035, 0.16, (p["size"] / 100) ** 1.2)  # (a bird's length, box heights, nearest)
        self.depth = p["depth"] / 100
        self.walkers, self.homes = [], []
        if self.mode in ("ground", "ledge"):
            x0, y0, w, h = self.box()
            for i in range(n):
                scale = self._scale(u[i], h)
                room = (0.42 if self.mode == "ledge" else 0.16) * w / max(scale, 1e-9)  # (units)
                walker = Walker(np.random.default_rng([self.seed, 4407, i]), self.loop_s, p["busy"] / 100, self.pace,
                                self.mode == "ledge", room)
                self.walkers.append(walker)
                self.homes.append(self._home(walker, u[i], scale, w, h))
        # flying: the flock together (whole laps, or passes, a loop: the same for all), each bird's wings beating
        # (whole beats a loop) and gliding now and then
        g = rng.random(4)
        self.laps = float(self.cycles(_lerp(0.07, 0.1, g[0]) * self.pace))
        self.beats = self.cycles(_lerp(4.5, 6.0, u[:, 1])) if n else np.zeros(0)
        self.glides = self.cycles(_lerp(0.15, 0.3, u[:, 2])) if n else np.zeros(0)
        self.orbit = g[1:]
        # light: the scene's where the box stands, or daylight
        self.light = self._light(light, p["lit"] / 100)

    def _light(self, light, k):
        key, ambient = DAYLIGHT
        side = -0.6
        if light is not None and k > 0:
            tone = np.asarray(light, np.float32)
            if tone.ndim != 1:
                tone = steam.light_at(tone, self.box(), self.w, self.h)
            scene_key = np.clip(tone[0:3], 0, 4) * 1.4
            scene_amb = np.clip(tone[3:6], 0, 4) * 0.9
            key = key * (1 - k) + scene_key * k
            ambient = ambient * (1 - k) + scene_amb * k
            side = side * (1 - k) + float(np.clip(tone[6], -1, 1)) * 0.8 * k
        way = np.array([side, -0.55, 0.85])
        return key.astype(np.float32), way / np.linalg.norm(way), ambient.astype(np.float32)

    # --- where each bird is now ---------------------------------------------------------------------------------------

    def _scale(self, u, h):
        """A bird's size on the ground: pixels a unit (the farther up the box, the smaller)."""
        depth = 0.0 if self.mode == "ledge" else u[1]  # (0 near: the box's bottom, 1 far: its top)
        return self.size * (1.0 - self.depth * 0.55 * depth) * h / LENGTH

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
        depth = u[1]
        top, bottom = 1.15 * reach - lo_y, h * 0.97 - hi_y
        hy = bottom - (bottom - top) * depth * 0.85 if bottom > top else h / 2
        return hx / w, hy / h

    def _ground(self, t, x0, y0, w, h):
        """Each bird on the ground now: (feet x, y in the frame, its scale px a unit, heading, pose)."""
        out = []
        sec = (t % self.P) / self.fps
        for walker, u, home in zip(self.walkers, self.u, self.homes):
            scale = self._scale(u, h)
            dx, dy, heading = walker.where(sec)
            fx = x0 + home[0] * w + dx * scale
            fy = y0 + home[1] * h - dy * scale * math.sin(self.tilt)
            out.append((fx, fy, scale, heading, walker.pose(sec)))
        return out

    def _flying(self, t, x0, y0, w, h):
        """Each bird in the air now: (its middle x, y in the frame, scale, heading, pose), all of it inside the box (a
        flock's loop made small enough; birds passing come in and go out at its sides)."""
        out = []
        tau = self.tau(t)
        body = self.size * 0.8 * h  # (a bird's length, px)
        spread = 1.6 * math.sqrt(len(self.u)) * body  # (how far apart the birds of a flock are)
        reach = REACH * body * (1.0 + 0.25 * self.depth)  # (the nearest's wings, round its middle)
        rx0, ry0 = w * 0.33, h * 0.18
        if self.mode == "flock":  # (its loop and spread made smaller to fit the box: as far as they can go)
            kx = min(1.0, max(0.0, w / 2 - reach) / (rx0 + 0.6 * spread))
            ky = min(1.0, max(0.0, h / 2 - reach) / (ry0 * (1 + 0.6 * spread / rx0) + 0.25 * spread + 0.04 * h))
        else:  # (a band across the box: its middle where the seed puts it, all of it between the box's top and bottom)
            half = 0.5 * spread + 0.04 * h
            ky = min(1.0, max(0.0, h / 2 - reach) / half)
            mid = min(max(h * (0.35 + 0.3 * self.orbit[2]), reach + ky * half), h - reach - ky * half)
        for i, u in enumerate(self.u):
            beat = math.sin(TAU * self.beats[i] * tau + u[3] * TAU)
            glide = float(_smooth((math.sin(TAU * self.glides[i] * tau + u[4] * TAU) - 0.35) / 0.4))
            angle = (1 - glide) * (0.15 + 0.85 * beat) + glide * GLIDE
            near = 1.0 - self.depth * 0.5 * u[5]
            scale = self.size * 0.8 * near * h / LENGTH
            if self.mode == "flock":  # (round one loop together, each a few lengths ahead or behind, in or out)
                ahead = (u[0] - 0.5) * 2 * spread / (math.pi * (rx0 + ry0))  # (a share of the loop)
                a = TAU * (self.laps * tau + self.orbit[0] + ahead)
                out_ = (u[6] - 0.5) * 1.2 * spread
                rx, ry = rx0 + out_, ry0 + out_ * ry0 / rx0
                fx = x0 + w / 2 + kx * rx * math.cos(a)
                fy = y0 + h / 2 + ky * (ry * math.sin(a) - (u[7] - 0.5) * 0.5 * spread
                                        - 0.04 * h * math.sin(2 * a + self.orbit[1] * TAU))
                scale *= 1.0 + 0.25 * self.depth * math.sin(a)  # (nearer at the bottom of the loop)
                heading = math.atan2(-math.cos(a), -math.sin(a))  # (round a level circle seen from above)
                roll = -0.35 * (1 - glide * 0.5)
            else:
                go = 1 if self.p["way"] == "right" or (self.p["way"] == "both" and u[6] < 0.5) else -1
                k = self.laps * 3  # (passes a loop: a group together, each a few lengths ahead or behind)
                # (from out of the box at one side to out of it at the other: the box cuts it off at its sides)
                s = (k * tau + (0.5 if go < 0 else 0.0) + (u[0] - 0.5) * 2 * spread / (1.6 * w)) % 1.0
                way = w + 2 * reach + 0.6 * w
                fx = x0 - reach - 0.3 * w + way * (s if go > 0 else 1 - s)
                fy = y0 + mid + ky * ((u[7] - 0.5) * spread + 0.04 * h * math.sin(TAU * (2 * s + u[2])))
                heading = 0.0 if go > 0 else math.pi
                heading += 0.25 * (u[3] - 0.5)  # (a little toward or away)
                roll = 0.0
            out.append((fx, fy, scale, heading, Pose(flying=True, beat=angle, roll=roll)))
        return out

    def draw(self, t, canvas, mask):
        x0, y0, w, h = self.box()
        if h < 4 or not len(self.u):
            return
        on_ground = self.mode in ("ground", "ledge")
        birds = self._ground(t, x0, y0, w, h) if on_ground else self._flying(t, x0, y0, w, h)
        order = sorted(range(len(birds)), key=lambda i: (birds[i][1], i))  # (far ones first: higher up)
        if on_ground and self.p["shadow"] > 0:
            for i in order:
                self._shadow(canvas, birds[i], mask)
        for i in order:
            fx, fy, scale, heading, pose = birds[i]
            if scale * LENGTH < 3:
                continue
            ss = 3 if scale * LENGTH < 40 else 2
            rgb, alpha, (ax, ay) = draw_bird(pose, self.coats[i], heading, scale, self.tilt, self.light, ss)
            if not on_ground:  # (a bird in the air: where it is is its middle, 0.3 units above its feet)
                fy += 0.3 * scale * math.cos(self.tilt)
            ix, iy = math.floor(fx) - ax, math.floor(fy) - ay
            inside = _inside(ix, iy, alpha.shape, x0, y0, w, h)
            if inside is None:
                continue
            prgb = np.clip(rgb, 0.0, None) ** (1 / 2.2)  # (to the picture's light: premultiplied, about)
            a = np.clip(alpha, 0.0, 1.0)
            prgb = np.minimum(prgb, 1.0) * np.where(a > 0, a ** (1 - 1 / 2.2), 0.0)[..., None]
            canvas.over_rgba(ix, iy, (prgb * inside[..., None]).astype(np.float32), (a * inside).astype(np.float32),
                             mask)

    def _shadow(self, canvas, bird, mask):
        """A soft shadow on the ground under a bird's feet."""
        fx, fy, scale, heading, pose = bird
        rx = 0.2 * scale * (abs(math.cos(heading)) * 0.8 + 0.4)
        ry = 0.2 * scale * max(0.15, math.sin(self.tilt)) * (abs(math.sin(heading)) * 0.8 + 0.4)
        if rx < 1:
            return
        x0, y0 = math.floor(fx - 2 * rx), math.floor(fy - 2 * ry)
        xs = np.arange(int(4 * rx) + 2) + x0 + 0.5 - fx
        ys = np.arange(int(4 * ry) + 2) + y0 + 0.5 - fy - 0.2 * ry
        d = (xs[None, :] / rx) ** 2 + (ys[:, None] / ry) ** 2
        inside = _inside(x0, y0, d.shape, *self.box())
        if inside is None:
            return
        key, _, ambient = self.light
        sun = float(key @ LUMA) / max(float((key + ambient) @ LUMA), 1e-6)  # (sunlight: darker shadows)
        cover = (np.exp(-2.2 * d) * (0.35 + 0.45 * sun) * self.p["shadow"] / 100 * inside).astype(np.float32)
        canvas.over(x0, y0, cover, np.zeros(3, np.float32), 1.0, mask)
