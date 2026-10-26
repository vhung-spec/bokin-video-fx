"""Pigeons ("Chim bồ câu"): rock doves on the ground or on a ledge, walking about, stopping to look round and, now and
then, bending down to peck at seeds (a placed effect: Effect.BOX; drawn over the picture, lit by the scene; see
effects.py for how every effect loops).

A pigeon is drawn by code, not from a picture. Its body is one smooth shape: ellipsoids (breast, body, belly, vent, the
folded wings over its flanks, their primaries along its sides turned in so their tips meet over the tail, the tail and
its spread end, its feathered thighs, the neck, the head) melted into each other as their distances are (a smooth union:
Inigo Quilez's polynomial smooth minimum), so it has no lumps where its parts meet, as feathers cover a real bird.
Standing still, it stands up tall, breast out, its tail down, as the rock dove of a photograph the parts were fitted to.
A far camera (straight rays) looks at it from a little above; each ray is marched to the smooth surface from where it
meets the plain union of the ellipsoids (a few steps: the smooth surface lies just outside it). Its small parts (beak,
cere, legs, toes) are ellipsoids met directly. Its feathers are painted by where on the bird a point is, each part's
colour blended by the same weights that melt the parts together (no seams), each part's patterns where they are on it in
any pose: a rock dove's slate head, its iridescent neck (green seen straight on, purple at a slant, at its throat and
round its foot, spilling onto its breast; the head's slate at its top; small iridescent feathers, each greener or
purpler, their tips catching the light in fine streaks), pale breast, pale back and folded wings with two black bars
across them as the owner's photograph has them (curving forward toward the wing's lower edge, their rear edges scalloped
by the feathers' tips, each feather's pale tip behind them), the wing's lower edge tucked under the flank's feathers,
the dark primaries behind, the long feathers' pale fringes, the black band at the tail's end, darker flanks and belly,
its feathered thighs over the tops of its legs, a small orange eye with the light caught in it, the white cere, coral
feet with dark claws (the colours measured on the photograph); or a checker (a dark wedge on each covert, crisp on the
pale wing), a dark one (these two the owner's choice: the flock by default), a red (ash-red) one, fawn and copper, a
pied one (white on whole parts of it as a pied pigeon's: its head, bib, belly, flights, tail, a patch on its wing or its
rump, with clean edges, a few white flecks), a white dove; a faint mottling, every pattern blurred to the pixel so it
does not shimmer. Matte, as feathers are (only its neck and feet faintly glossy). One ray a pixel inside a bird DETAIL
pixels long or more, ss x ss where its outline or a small part's crosses a pixel; drawn where it is to a fraction of a
pixel (it glides); softened a little ("Độ mềm") as a photograph is. It is lit by the scene where its box stands (as
steam is: light_at), less where its body shuts the light out (the creases between its parts, under it: ambient occlusion
from the smooth distance), and lies in its own shadow: dark right under it, soft and wider away from the light.

Every bird keeps its own day over the loop: walking a while, nearly straight, then stopping to look round or, now and
then ("Mổ hạt"), to peck at the ground a few times, its beak reaching it and its tail rising; between its walks it turns
on the spot, a step at a time, this way, then that, as a pigeon does, never round and round (the owner, 2026-10-25:
"chim xoay tròn là không giống nhất": birds walking round small loops of their own, males spinning round courting,
looked least like real ones); at the loop's end back where it started, facing the same way. As it walks, its planted
foot stays put while the other steps forward, its body swaying a little over it, and its head is held still and thrust
forward a step at a time, as pigeons do; turning, each foot stays where it was put down while the body turns over it.
It speeds up and slows down smoothly, its feet together where it starts and stops, and stands up tall, bends down and
looks round without a jolt.

No bird walks through another or stands in another (the owner, 2026-10-24): each is placed where, and its day begun
when, its path meets the others' least (_spread); where two still meet, the one walking steps aside round the other,
smoothly, a standing one never sliding (_plan: their footprints, ellipses round their bodies, kept apart); a box too
small for so many birds that big holds only those it has room for. Along a ledge each bird walks to and fro on a stretch
of its own, turning round on the spot at its stops. The flock is laid out in a box REF high, so it is the same at every
size it is drawn at.
"""
import collections
import math
import threading

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


def _smoother(x):
    """Ken Perlin's smootherstep: its speed and its acceleration both 0 at either end."""
    x = np.clip(x, 0.0, 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


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
    mark = _smooth(((t - 0.1) * 1.1 - np.abs(2 * fr - 1)) / 0.12) * _smooth((0.92 - t) / 0.05)  # (a dark wedge
    # widening toward its tip, a pale edge round it)
    return light, k * edge, rnd, k * mark


COATS = {  # sRGB: a rock dove (blue bar), a dark one, a red (ash-red) one, a white dove
    "grey": {"head": (114, 121, 148), "neck": (44, 50, 60), "green": (30, 92, 80), "purple": (80, 60, 106),
             "breast": (128, 132, 156), "back": (176, 180, 196), "flank": (112, 117, 138), "bar": (30, 30, 36),
             "prim": (60, 62, 72), "tail": (98, 102, 116), "band": (42, 42, 48), "belly": (134, 139, 158),
             "beak": (58, 56, 62), "cere": (206, 202, 196), "iris": (214, 96, 34), "feet": (192, 86, 94),
             "spot": (48, 50, 60), "sheen": 1.0},
    "dark": {"head": (52, 54, 60), "neck": (44, 48, 52), "green": (46, 104, 78), "purple": (96, 60, 104),
             "breast": (58, 56, 64), "back": (66, 68, 76), "flank": (58, 60, 68), "bar": (34, 34, 38),
             "prim": (40, 42, 46), "tail": (54, 56, 62), "band": (30, 30, 34), "belly": (64, 66, 74),
             "beak": (36, 34, 36), "cere": (212, 210, 204), "iris": (210, 110, 42), "feet": (184, 82, 90),
             "spot": (34, 34, 38), "sheen": 1.1},
    "brown": {"head": (150, 120, 106), "neck": (140, 106, 96), "green": (132, 120, 98), "purple": (150, 108, 102),
              "breast": (160, 128, 114), "back": (196, 172, 156), "flank": (164, 134, 120), "bar": (122, 74, 58),
              "prim": (150, 124, 112), "tail": (176, 152, 138), "band": (142, 102, 88), "belly": (170, 146, 134),
              "beak": (196, 160, 146), "cere": (232, 226, 218), "iris": (210, 110, 42), "feet": (194, 90, 96),
              "spot": (138, 92, 76), "sheen": 0.5},
    "white": {"head": (232, 232, 228), "neck": (230, 230, 226), "green": (226, 232, 224), "purple": (232, 226, 230),
              "breast": (234, 232, 228), "back": (238, 238, 234), "flank": (228, 228, 224), "bar": (234, 234, 230),
              "prim": (224, 224, 220), "tail": (234, 234, 230), "band": (230, 230, 226), "belly": (232, 232, 228),
              "beak": (206, 168, 156), "cere": (238, 232, 226), "iris": (64, 38, 28), "feet": (204, 104, 110),
              "spot": (232, 232, 228), "sheen": 0.0},
}
MIXED = (("grey", 0.42), ("checker", 0.18), ("dark", 0.09), ("brown", 0.07), ("pied", 0.14), ("white", 0.1))  # (a
# town's flock)
PAIR = (("checker", 0.6), ("dark", 0.4))  # (the owner's flock, 2026-10-25: checkers and dark ones only, so many of
# each however few birds)
MIXES = {"mixed": MIXED, "checker_dark": PAIR}
GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0
WHITE = _lin((236, 236, 232))
WHITES = (("head", 0.55), ("flights", 0.45), ("tail", 0.3), ("bib", 0.35), ("belly", 0.45), ("shield", 0.3),
          ("rump", 0.25))  # (a pied pigeon's white: on which parts, how often)
FLECKS = 2.6  # (its white flecks: this much smaller than its patches' waves)


class Coat:
    """A bird's feathers: its coat's colours (linear light), dark spots on its wings (a checker) or white parts (a
    pied one: `whites`, WHITES), a little lighter or darker than the next bird, a faint mottling."""

    def __init__(self, name, rng):
        base = name if name in COATS else "grey"
        self.c = {k: (_lin(v) if isinstance(v, tuple) else v) for k, v in COATS[base].items()}
        self.checker = name == "checker"
        self.pied = name == "pied"
        self.tone = 0.92 + 0.16 * rng.random()
        self.waves = rng.normal(0.0, 1.0, (5, 3)) * 45.0  # (a faint mottling)
        self.phases = rng.random(5) * TAU
        self.patches = rng.normal(0.0, 1.0, (4, 3)) * 11.0  # (a pied bird's white: its flecks, its edges' waves)
        self.patch_phases = rng.random(4) * TAU
        self.patch_at = rng.uniform(-0.6, 0.3)  # (how much of it is flecked white)
        odds = 0.5 * (1.0 + np.array([math.erf(v) for v in self.patches.ravel() / (11.0 * math.sqrt(2.0))]))
        self.whites = {name for (name, share), u in zip(WHITES, odds) if u < share} or {"head"}  # (a pied bird's
        # white parts: from the numbers above, so a flock's birds are where they were)

    def grain(self, p, fp):
        """A faint mottling at points p (about 0.93..1.07): its waves longer than a few pixels (`fp`: a pixel's width,
        units), so it does not shimmer on a small bird."""
        out = np.zeros(len(p), np.float32)
        for w, ph in zip(self.waves, self.phases):
            k = _fade(float(np.linalg.norm(w)), fp)
            if k > 0:
                out += k * np.sin(p @ w + ph).astype(np.float32)
        return 1.0 + 0.035 * out / 2.2

    def white(self, p, fp):
        """A pied bird's white flecks at points p (0..1): small, here and there, as many as its patch_at says."""
        f = np.zeros(len(p), np.float32)
        for w, ph in zip(self.patches, self.patch_phases):
            f += np.sin(p @ (FLECKS * w) + ph).astype(np.float32)
        return _smooth((f / 4.0 - (0.72 + 0.15 * self.patch_at)) / max(0.05, 25.0 * fp))

    def white_on(self, name, q, local, fp):
        """How white a pied bird's part `name` is at points of it (q: where on the bird as it walks; local: in the
        part's own unit frame), its white regions (`whites`) with wavy edges, sharp to a pixel or two (`fp`: a
        pixel's width, units)."""
        w = self.whites
        wave = np.sin(q @ (3.5 * self.patches[0]) + self.patch_phases[1]) * 0.6 + np.sin(
            q @ (2.5 * self.patches[1]) + self.patch_phases[2]) * 0.4  # (-1..1: its edges wavy)
        edge = max(0.012, 2.0 * fp)

        def side(d):  # (white on the side of an edge where d > 0, units, the edge wavy)
            return _smooth((d + 0.02 * wave) / edge + 0.5)

        out = np.zeros(len(q))
        x, z = q[:, 0], q[:, 2]
        if name == "head" or name in ("prim",) and "flights" in w or name in ("tail", "fan") and "tail" in w:
            out[:] = 1.0 if (name == "head" and "head" in w) or name != "head" else 0.0
        elif name == "neck":
            lz, lx = local[:, 2], local[:, 0]
            if "head" in w:
                out = np.maximum(out, _smooth((lz - 0.35 + 0.2 * wave) / 0.15 + 0.5))
            if "bib" in w:
                out = np.maximum(out, _smooth((lx - 0.15 + 0.2 * wave) / 0.2 + 0.5) * _smooth((0.6 - lz) / 0.2))
        elif name in ("breast", "body"):
            if "bib" in w:
                out = np.maximum(out, side(x - 0.11))
            if "belly" in w:
                out = np.maximum(out, side(0.19 - z))
            if "rump" in w and name == "body":
                out = np.maximum(out, side(z - 0.29) * side(-0.12 - x))
        elif name in ("belly", "vent", "thigh"):
            out[:] = 1.0 if "belly" in w else 0.0
        elif name == "wing":
            a, up = local[:, 0], local[:, 2]
            if "shield" in w:
                out = np.maximum(out, _smooth((a + 0.15 + 0.25 * wave) / 0.12 + 0.5)
                                 * _smooth((up + 0.35 + 0.25 * wave) / 0.12 + 0.5))
            if "flights" in w:
                out = np.maximum(out, _smooth((-0.62 - a + 0.15 * wave) / 0.12 + 0.5))
        return out


# --- its shape and poses ----------------------------------------------------------------------------------------------

HIP = np.array([-0.01, 0.0, 0.12])  # the body tips about this (pecking)
NECK = np.array([0.1, 0.0, 0.31])  # where the neck leaves the shoulders (standing)
HEAD = np.array([0.182, 0.0, 0.43])  # its head's middle (standing)
HEAD_R = np.array([0.063, 0.044, 0.05])
# the body as it walks: name, centre, radii, pitch (degrees, the nose up), and the folded wings' and their primaries'
# turn (degrees: their rear ends in, so the primaries' tips meet over the tail and it narrows smoothly to the tail seen
# from behind); the wings lie over the flanks, as a pigeon's cover its sides
BODY = (("breast", (0.08, 0.0, 0.262), (0.132, 0.106, 0.132), 16),
        ("body", (-0.055, 0.0, 0.245), (0.2, 0.104, 0.125), 5),
        ("belly", (-0.01, 0.0, 0.15), (0.14, 0.088, 0.075), 4),
        ("vent", (-0.2, 0.0, 0.188), (0.11, 0.058, 0.05), -9),
        ("wing", (-0.11, 0.062, 0.255), (0.2, 0.052, 0.085), 8, 9),
        ("wing", (-0.11, -0.062, 0.255), (0.2, 0.052, 0.085), 8, -9),
        ("tail", (-0.335, 0.0, 0.198), (0.165, 0.056, 0.022), -9),
        ("fan", (-0.45, 0.0, 0.184), (0.085, 0.064, 0.011), -9),
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
           ("tail", (-0.226, 0.0, 0.075), (0.142, 0.052, 0.018), 17),
           ("fan", (-0.355, 0.0, 0.036), (0.085, 0.06, 0.011), 17),
           ("prim", (-0.25, 0.05, 0.148), (0.17, 0.014, 0.04), 38, 6),
           ("prim", (-0.25, -0.05, 0.148), (0.17, 0.014, 0.04), 38, -6))
NECK_UP = np.array([0.034, 0.0, 0.332])  # (where its neck leaves its shoulders, standing tall)
TAIL_AT = np.array([-0.22, 0.0, 0.19])  # (the tail bends up and down about this)
TAILS = {"tail": 1.0, "fan": 1.0, "prim": 1.0, "vent": 0.5}  # (how much of the tail's bend each part takes: the
# wingtips lie on it)
TIGHT = {"prim": 0.008, "fan": 0.008}  # (the wingtips and the tail's spread end melt in less: thin, lying on it)
SIT, FLUFF = 0.072, 0.07  # (sitting down: how far its body drops, its belly on the ground; how much fuller its
# feathers fluffed up)
PUFF = 0.42  # (courting: how much thicker its neck puffed up)
SPREAD = {"fan": 0.75, "tail": 0.35}  # (its tail spread: how much wider each part)
SMOOTH = len(BODY) + 4  # (the parts of the smooth body: BODY, its thighs, the neck, the head)
# its feet, as a rock dove's (units: a bird is LENGTH long, about 33 cm): three toes forward and one back, each a few
# joints long, thinner to its tip, ending in a curved claw: (its turn out from straight ahead, radians; its joints'
# lengths; its thickness at its base; its claw's length); its tarsus thicker at the top; scales across them all
TOES = ((-0.5, (0.024, 0.02), 0.0057, 0.013),  # (the inner toe)
        (0.07, (0.027, 0.023, 0.019), 0.0064, 0.016),  # (the middle one, the longest)
        (0.55, (0.022, 0.019, 0.015), 0.0055, 0.013),  # (the outer one)
        (math.pi + 0.25, (0.021, 0.017), 0.006, 0.014))  # (the hind toe, a little in)
TARSUS = (0.0085, 0.0071)  # (the tarsus' radius at its top, at the foot)
SCALES = 0.0072  # (the scales across its legs and toes: this far apart)
THIGH = (0.42, 0.016, 0.034)  # (its thighs: the belly's feathers down over the top of each leg, melted into it: how
# far down the leg each one's middle is, its radius, its half length)
HEEL = np.array([-0.004, 0.0, 0.0068])  # (where its tarsus meets its foot, from the foot's middle)
FINE = 9.0  # (a middle toe this many pixels long or more: its toes drawn joint by joint; shorter, one piece each)


class Pose:
    """How a bird stands now: how tall (0: as it walks .. 1: standing tall, breast out: BODY_UP), its body tipped
    forward (pitch, radians: < 0 tipped up), its tail bent up from it (radians), its head's middle (bird frame), the
    head's own pitch and turn, its feet (bird frame, each: x, y, lift); as it steps, its body shifted over the foot it
    stands on (sway, units to its left), up and down (bob, units) and rolled (roll, radians: > 0 its back to its
    right); how far it has sat down on the ground (sit 0..1: its feathers fluffed, its toes tucked under it: tuck),
    how puffed up its neck and breast are (puff 0..1, a male courting) and how spread its tail (spread 0..1)."""

    def __init__(self, pitch=0.0, head=None, head_pitch=0.17, head_yaw=0.0, feet=None, tail=0.0, tall=0.0):
        self.pitch, self.head_pitch, self.head_yaw, self.tail, self.tall = pitch, head_pitch, head_yaw, tail, tall
        self.head = HEAD.copy() if head is None else np.asarray(head, np.float64)
        self.feet = feet if feet is not None else ((0.0, 0.034, 0.0), (0.0, -0.034, 0.0))
        self.sway = self.bob = self.roll = 0.0
        self.sit = self.tuck = self.puff = self.spread = 0.0


def _heel(at):
    """Where a foot's tarsus meets it (the foot `at`: x, y, how high it is lifted and, turning on the spot, its own
    way: a planted foot stays as it was put down while the body turns over it)."""
    own = float(at[3]) if len(at) > 3 else 0.0
    c, s = math.cos(own), math.sin(own)
    return np.array([at[0] + c * HEEL[0] - s * HEEL[1], at[1] + s * HEEL[0] + c * HEEL[1], at[2] + HEEL[2]])


def _foot(side, hip, at, fine, tuck=0.0):
    """One foot's parts (the bird's frame; `side` 1 its left, -1 its right): its tarsus from under its belly's
    feathers (`hip`) down to the foot (`at`: the foot's middle on the ground, how high it is lifted and, turning on the
    spot, its own way), its four toes flat on the ground, or curled and drawn together as the foot is lifted (or
    `tuck`ed under a bird sitting down), each ending in a claw curving down to the ground; the toes joint by joint
    (`fine`) or one piece each."""
    f = np.asarray(at[:2], np.float64)
    lift = float(at[2])
    own = float(at[3]) if len(at) > 3 else 0.0
    hang = min(max(lift / 0.02 - float(tuck) * 1.7, 0.0), 1.0)  # (lifted: hanging down; tucked: folded up)
    fold = min(max(float(tuck), 0.0), 1.0)
    curl = max(hang, fold)
    foot = np.array([f[0], f[1], lift])
    heel = _heel(at)
    rot = _along(heel, hip)
    span = float(np.linalg.norm(hip - heel))
    out = []
    for k in range(3):  # (a rod thinner toward the foot: three long pieces, each well into the next, the lowest
        # ending at the foot)
        f0, f1 = k / 3, (k + 1) / 3
        rad = TARSUS[1] + (TARSUS[0] - TARSUS[1]) * (f0 + f1) / 2
        grow = 1.6 * rad if k else 0.8 * rad
        out.append(("leg", heel + (hip - heel) * (f0 + f1) / 2 + rot[:, 2] * (1.6 * rad - grow) / 2, rot,
                    np.array([rad, 0.92 * rad, span / 6 + (1.6 * rad + grow) / 2])))
    base = foot + np.array([0.003 * math.cos(own), 0.003 * math.sin(own), 0.0])  # (where the toes leave the foot)
    for turn, joints, thick, claw in TOES:
        yaw = own + (side * turn * (1.0 - 0.4 * curl) if turn < 3 else side * (math.pi + (turn - math.pi) * (1 - curl)))
        cy, sy = math.cos(yaw), math.sin(yaw)
        a = z = 0.0  # (along the toe from its base, and down)
        ang = 0.35 * hang - 0.6 * fold  # (lifted: its toes hang down, bending more at each joint; tucked under a
        # sitting bird: folded up into its feathers)
        r = thick
        if not fine:
            length = sum(joints)
            ca, sa = math.cos(ang + 0.2 * curl), math.sin(ang + 0.2 * curl)
            out.append(("toe", base + np.array([cy * ca * length / 2, sy * ca * length / 2,
                                                0.84 * r - sa * length / 2]),
                        _rot(yaw=yaw, pitch=-(ang + 0.2 * curl)), np.array([length / 2 + 0.3 * r, r, 0.84 * r])))
            a, z, r = ca * length, sa * length, r * 0.86 ** len(joints)
            ang += 0.45 * curl * len(joints)
        else:
            for length in joints:
                ca, sa = math.cos(ang), math.sin(ang)
                out.append(("phal", base + np.array([cy * (a + ca * length / 2), sy * (a + ca * length / 2),
                                                     0.84 * r - z - sa * length / 2]),
                            _rot(yaw=yaw, pitch=-ang), np.array([length / 2 + 0.9 * r, r, 0.84 * r])))
                a, z = a + ca * length, z + sa * length
                r *= 0.86
                ang += 0.45 * curl
        # its claw: from the toe's tip, curving down to the ground (or under, curled), its tip on it
        tip = base + np.array([cy * a, sy * a, 0.84 * r - z])
        nails = []
        for share, bend, thin in ((0.55, 0.12, 0.62), (0.5, 0.38, 0.42)):
            down = ang + bend
            length = claw * share
            way = np.array([cy * math.cos(down), sy * math.cos(down), -math.sin(down)])
            rot, rad = _rot(yaw=yaw, pitch=-down), np.array([length / 2 + 0.3 * r * thin, r * thin, r * thin * 0.85])
            nails.append(["claw", tip + way * length / 2, rot, rad])
            tip = tip + way * length * 0.9
        low = min(float(c[2]) - math.sqrt(float(((rot[2, :] * rad) ** 2).sum())) for _, c, rot, rad in nails)
        for nail in nails:  # (not into the ground)
            nail[1] = nail[1] + np.array([0.0, 0.0, max(0.0, -low)])
            out.append(tuple(nail))
    return out


def parts(pose, fine=True):
    """A bird's parts in a pose: (name, centre, rotation (columns: its axes), radii), in the bird frame (x forward, y
    left, z up; its feet on the ground at 0): the SMOOTH parts of its body first, then its small ones (its toes joint
    by joint if `fine`)."""
    tip = _rot(roll=pose.roll) @ _rot(pitch=-pose.pitch)  # (the body tipped forward about the hip, rolled)
    sit, puff, spread = (min(max(float(v), 0.0), 1.0) for v in (pose.sit, pose.puff, pose.spread))
    shift = HIP + np.array([0.0, pose.sway, pose.bob - SIT * sit])  # (and shifted as it steps, or sits down)
    k = min(max(float(pose.tall), 0.0), 1.0)
    out = []
    for (name, c, r, pitch, *turn), (_, cu, ru, pu, *tu) in zip(BODY, BODY_UP):
        c = np.array(c, np.float64) * (1 - k) + np.array(cu, np.float64) * k
        r = np.array(r, np.float64) * (1 - k) + np.array(ru, np.float64) * k
        r = r * (1.0 + FLUFF * sit)  # (sitting: its feathers fluffed up)
        if name == "breast":
            r = r * (1.0 + 0.1 * puff)
        elif name in SPREAD:  # (its tail spread)
            r = r * np.array([1.0, 1.0 + SPREAD[name] * spread, 1.0])
        yaw = math.radians(turn[0] + (tu[0] - turn[0]) * k) if turn else 0.0
        own = _rot(yaw=yaw, pitch=math.radians(pitch + (pu - pitch) * k))
        bend = TAILS.get(name, 0.0) * pose.tail
        if bend:  # (the tail bent up about its joint)
            up = _rot(pitch=-bend)
            c, own = TAIL_AT + up @ (c - TAIL_AT), up @ own
        out.append((name, shift + tip @ (c - HIP), tip @ own, r))
    hips, feet = [], []
    for side, foot in zip((1, -1), pose.feet):
        hips.append(shift + tip @ (np.array([0.0, side * 0.03, 0.11]) - HIP))
        if pose.tuck > 0:  # (sitting down: its feet drawn up under its belly's feathers)
            foot = (foot[0] - 0.01 * pose.tuck, foot[1], foot[2] + 0.034 * pose.tuck, *foot[3:])
        feet.append(foot)
    for hip, foot in zip(hips, feet):  # (its thighs, over the top of its legs)
        heel = _heel(foot)
        out.append(("thigh", hip + (heel - hip) * THIGH[0], _along(heel, hip),
                    np.array([THIGH[1], 0.88 * THIGH[1], THIGH[2]])))
    base = shift + tip @ (NECK + (NECK_UP - NECK) * k - HIP)
    head = pose.head
    span = float(np.linalg.norm(head - base))
    thick = 1.0 + 0.12 * sit + PUFF * puff  # (pulled in, sitting; puffed up, courting)
    out.append(("neck", base + (head - base) * 0.42, _along(base, head),
                np.array([(0.072 + 0.002 * k) * thick, (0.064 + 0.008 * k) * thick, max(0.07, span / 2 + 0.02)])))
    turn = _rot(yaw=pose.head_yaw, pitch=pose.head_pitch)
    out.append(("head", head, turn, HEAD_R))
    out.append(("beak", head + turn @ np.array([0.066, 0.0, -0.012]), turn @ _rot(pitch=-0.35),
                np.array([0.031, 0.0105, 0.0115])))
    out.append(("cere", head + turn @ np.array([0.056, 0.0, 0.003]), turn @ _rot(pitch=-0.35),
                np.array([0.015, 0.011, 0.0095])))
    for side, hip, foot in zip((1, -1), hips, feet):
        out.extend(_foot(side, hip, foot, fine, pose.tuck))
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
        self.q, self.dd = q.astype(np.float32), dd[:, :, None].astype(np.float32)
        self.ir = (1.0 / radii.T[:, :, None]).astype(np.float32)  # (each part's radii, inverted: a multiply each)
        self._idx = self._qi = None  # (the rays last asked about: the same again as the surface is closed in on)

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
        if idx is not self._idx:
            self._idx, self._qi = idx, self.q[:, :, idx]
        a = self._qi + t.astype(np.float32)[None, None, :] * self.dd  # (each part's unit sphere)
        return _union(_ellipsoids(a, self.ir), self.blends, shares)

    def _refine(self, idx, lo, hi, dlo, dhi, steps):
        """The surface between lo (outside) and hi (inside) along rays idx: false position, kept off the ends."""
        for _ in range(steps):
            f = np.minimum(np.maximum(dlo / np.maximum(dlo - dhi, 1e-12), 0.15), 0.85)
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


def _ellipsoids(a, ir):
    """Each part's signed distance (Inigo Quilez's bound for an ellipsoid) at points `a` in the parts' unit spheres
    (3 x parts x n, float32; ir: the parts' radii inverted, 3 x parts x 1)."""
    k0 = np.sqrt(np.einsum("ikn,ikn->kn", a, a))
    a *= ir
    k1 = np.sqrt(np.einsum("ikn,ikn->kn", a, a))
    np.maximum(k1, 1e-9, out=k1)
    d = k0 - 1.0  # (k0 (k0 - 1) / k1)
    d *= k0
    d /= k1
    return d


def _union(each, blends, shares=False):
    """The parts' distances (parts x n) melted into one smooth body, in order (polynomial smooth minimum, each part's
    own blend), and with shares=True each part's share of it."""
    d, out = each[0], [np.ones(each.shape[1], np.float32)]
    for k in range(1, len(each)):
        e, blend = each[k], blends[k]
        diff = d - e
        h = diff * (-0.5 / blend)
        h += 0.5
        np.minimum(np.maximum(h, 0.0, out=h), 1.0, out=h)  # (as e + (d - e) h - blend h (1 - h))
        d = h * blend
        d -= blend
        d += diff
        d *= h
        d += e
        if shares:
            out = [s * h for s in out]
            out.append(1.0 - h)
    return (d, out) if shares else d


def _sdf(p, body):
    """The smooth body's distance at points p (n x 3, the bird's frame), as _Body.distance has it along rays."""
    rots = np.array([part[2] for part in body])
    radii = np.array([part[3] for part in body])
    a = (np.einsum("ni,kij->jkn", p, rots) - np.einsum("ki,kij->jk", np.array([part[1] for part in body]), rots)
         [:, :, None]) / radii.T[:, :, None]
    return _union(_ellipsoids(a.astype(np.float32), (1.0 / radii.T[:, :, None]).astype(np.float32)),
                  [TIGHT.get(part[0], BLEND) for part in body])


SATIN = 0.0  # (the feathers' broad sheen: none, matte as feathers are; the owner, 2026-10-25: "bớt bóng")
RIM = 0.12  # (feathers soft as velvet: how much more light where the surface is seen edge on)
GLOSS = 0.2  # (the soft highlight of its iridescent neck, its legs and claws, where they mirror the light: faint)
SHUT = ((0.015, 1.0), (0.04, 0.7), (0.08, 0.4))  # (how far out along the normal its body is looked for, how much
# that counts: the creases where wing, breast and neck meet, under the belly and the tail, darker)


def _occlusion(p, n, body):
    """How open each point of the bird is to the light round it (1 open .. about 0.15 shut in): the usual signed
    distance ambient occlusion, its body's distance a few steps out along the normal short of the step."""
    shut = np.zeros(len(p))
    for h, w in SHUT:
        d = _sdf(p + n * h, body)
        shut += w * np.clip((h - d) / h, 0.0, 1.0)
    return 1.0 - 0.85 * shut / sum(w for _, w in SHUT)


_RESTING = []
NECK_FEATHER, NECK_ROW = 0.013, 0.007  # (the neck's small feathers, units long, and their rows this far apart: each
# catching the light its own way)
NAPE = (0.62, 0.25)  # (up its neck (its own frame: -1 its foot .. 1 its head's end), where the head's slate begins to
# come in over the sheen, and how far it takes: no line round its throat)
HOOD = 0.065  # how far the neck's iridescent feathers reach past it (units)


def _sheen(c, facing):
    """The neck's iridescent feathers seen at `facing` (|cos| of the angle to the camera): green straight on, purple at
    a slant."""
    f = facing[:, None]
    return np.clip(c["neck"] + c["sheen"] * (0.7 * (c["green"] - c["neck"]) * f ** 2
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


BARS = ((-0.3, 0.14, 0.4), (-0.6, 0.15, 1.3))  # (the wing's two bars, as on the owner's photograph: where each
# crosses its upper half (a), how wide, where along it its feathers' tips fall (a phase): the greater coverts' row,
# then the secondaries')
CURVE = 0.15  # (each bar curves forward over the wing's lower half: this much further forward (a) at its lower edge)
SLANT = 0.1  # (the long feathers behind the bars slant as they do, their lower ends further forward)
TUCKED = (-0.8, 0.25)  # (the wing's lower edge tucked under its flank's feathers: up from here down, over this)
TIPS = 4.0  # (the feathers in a row across the folded wing, each a rounded tip)


def _wing_colour(a, b, up, reach, c, coat, fp):
    """A folded wing's colour at points of it (a: along it, +1 its front, -1 its rear end, `reach` units from its
    middle; up: -1 its lower edge .. +1 its top; b: across it, +1 its outer face): pale grey (paler toward its top, a
    little darker toward its lower edge and its rear), rows of small feathers over its front, long ones behind them;
    across its outer face two black bars, each the dark band near the end of a row of feathers: curving across it
    (their lower ends further forward, tapering), their rear edges scalloped by the feathers' rounded tips, each
    feather's pale tip behind its band and the next row's bases in its shade; its dark primaries behind."""
    count = len(a)
    a = 1.0 - (1.0 - a) * (reach / WING)
    along, high = a * WING, up * 0.08  # (units)
    shade = 1.0 + 0.05 * up - 0.06 * _smooth((-a - 0.1) / 0.5)
    out = c["back"] * shade[:, None]
    rows_b = (0.08 - high) + 0.15 * (1 - np.abs(b))
    covert = _feathers(-along, rows_b, 0.03, 0.7, fp)
    front = _smooth((a + 0.1) / 0.15)
    if covert is not None:
        light, edge, rnd, mark = covert
        out = out * (1 + 0.6 * front * (light - 1))[:, None] + (0.02 * front * edge)[:, None] * c["back"]
        if coat.checker:  # (a dark mark near the tip of each covert, crisp on the pale wing: a checker, as the
            # owner chose it, 2026-10-25)
            out = out + (c["spot"] - out) * (mark * (0.75 + 0.25 * rnd) * front)[:, None]
    elif coat.checker:  # (too small to see each feather: the checker's mean)
        out = out + (c["spot"] - out) * (0.3 * front)[:, None]
    long_ = _feathers(-(along - SLANT * WING * up), rows_b, 0.09, 0.28, fp)  # (the secondaries, slanting as the
    # bars)
    if long_ is not None:
        out = out * (1 + (1 - front) * (long_[0] - 1))[:, None]
    width = max(0.03, fp / WING)  # (an edge: a pixel or more)
    see = _fade(TAU * TIPS / (2 * 0.08), fp)  # (the scallops, once each feather's tip is a few pixels across)
    bars, tips, shadow = np.zeros(count), np.zeros(count), np.zeros(count)
    for a0, w0, phase in BARS:
        d = a - (a0 + CURVE * np.maximum(-up, 0.0) ** 2)  # (in front of its middle line: > 0)
        f = np.mod((up + 1) * TIPS / 2 + phase, 1.0)
        tip = see * 0.035 * (1 - (2 * f - 1) ** 2)  # (each feather's rounded tip: furthest back in its middle)
        w = w0 * (0.75 + 0.45 * _smooth((0.6 - up) / 1.2)) * _smooth((up + 1.0) / 0.35)  # (broadest over the
        # wing's lower half, rounded off at its lower edge)
        lead, back = w / 2, -w / 2 - tip  # (its edges: the front one under the row before, the rear one scalloped)
        band = _smooth((lead - d) / width) * _smooth((d - back) / width)
        end = back - 0.06 - 0.5 * tip  # (the pale tip behind it, scalloped as its feathers' ends)
        bars = np.maximum(bars, band)
        tips = np.maximum(tips, _smooth((back - d) / width) * _smooth((d - end) / width))
        shadow = np.maximum(shadow, _smooth((end - d) / width) * _smooth((d - end + 0.05) / 0.05))
    outer = _smooth((b + 0.3) / 0.4)  # (not round its inner face: no rings seen from behind)
    pale = float(c["back"] @ LUMA)
    marked = float(np.clip(2.0 * (pale - float(c["bar"] @ LUMA)) / max(pale, 1e-6), 0.0, 1.0))  # (the pale tips and
    # their shade only as strong as the bars are: none on a white dove)
    out = out * (1 - 0.12 * marked * shadow * outer)[:, None]
    out = out + (np.minimum(c["back"] * 1.22, 1.0) - out) * (0.9 * marked * tips * (1 - bars) * outer)[:, None]
    out = out + (c["bar"] - out) * (bars * outer)[:, None]
    tip = _smooth((-a - 0.74) / 0.12)  # (its dark primaries)
    out = out + (c["prim"] - out) * tip[:, None]
    under = 0.85 * _smooth((TUCKED[0] - up) / TUCKED[1])  # (its lower edge under the flank's feathers: no edge drawn
    # round it, as if laid on)
    return out + (c["flank"] - out) * under[:, None]


def _part_colour(name, q, local, rad, nz, facing, hood, c, coat, fp):
    """One part's colour at points q on the bird as it walks (its own frame taken back there; `local`: the points in
    the part's own unit frame, its radii `rad` in this pose), facing up nz, facing the camera `facing`; `hood`: how
    near the neck (1 on it). With how glossy each point is (its iridescent feathers)."""
    if name == "wing":
        return _wing_colour(local[:, 0], local[:, 1] * np.sign(q[:, 1]), local[:, 2], rad[0], c, coat, fp), 0.0
    if name in ("body", "belly", "vent", "breast", "thigh"):  # (iridescent, purple, round the neck's foot)
        out = _body_colour(q, nz, c, fp) * (0.95 if name == "vent" else 1.0) if name != "breast" else c["breast"]
        sheen = _sheen(c, facing)
        sheen = sheen + (c["purple"] - sheen) * 0.7 * min(c["sheen"], 1.0)  # (as glossy as its coat is)
        return out + (sheen - out) * (hood * min(1.0, 0.4 + 0.6 * c["sheen"]))[:, None], hood
    if name == "neck":  # (green, purple toward its foot, its head's slate toward the head; small feathers, each a
        # little greener or purpler, some catching the light more than others)
        sheen = _sheen(c, facing)
        front = _smooth(local[:, 0] / 0.8)  # (its throat purpler than its nape)
        foot = _smooth((0.1 + 0.25 * front - local[:, 2]) / 0.6) * (0.85 if c["sheen"] else 0.0)
        out = sheen + (c["purple"] - sheen) * foot[:, None]
        top = _smooth((local[:, 2] - NAPE[0]) / NAPE[1])  # (its upper end the head's colour, the sheen coming in
        # below it)
        rows = 2 * max(1, round(math.pi * rad[0] / NECK_ROW))  # (an even number of rows round it: no seam)
        around = np.arctan2(local[:, 1], local[:, 0]) / TAU * rows * NECK_ROW
        small = _feathers(-local[:, 2] * rad[2], around, NECK_FEATHER, NECK_ROW / NECK_FEATHER, fp)
        if small is None:
            return out + (c["head"] - out) * top[:, None], 1.0 - top
        light, _, rnd, streak = small
        out = out * (1 + 2.0 * (light - 1) + 0.9 * streak * rnd)[:, None]  # (each feather's tip catching the light:
        # fine streaks, some brighter)
        if c["sheen"]:
            tint = np.where(rnd < 0.5, 0, 1)[:, None] * (c["purple"] - c["green"]) + c["green"]
            out = out + (tint - out) * (0.12 * c["sheen"] * np.abs(2 * rnd - 1))[:, None]
        k = float(np.clip((NECK_ROW / max(fp, 1e-9) - 2.0) / 3.0, 0.0, 1.0))  # (fading as the feathers do)
        return out + (c["head"] - out) * top[:, None], (1.0 + k * (0.3 + 1.1 * (rnd * light) ** 2 - 1.0)) * (1 - top)
    if name == "head":
        return np.broadcast_to(c["head"], (len(q), 3)), 0.0
    x = q[:, 0]  # (the tail and the wingtips: long feathers)
    quills = _feathers(-x, np.abs(q[:, 1]) + 0.3 * (0.25 - q[:, 2]), 0.16, QUILL_ROW / 0.16, fp)
    if name == "prim":  # (darkest along their top edge, their inner webs paler)
        tips = c["prim"] + (c["tail"] * 1.3 - c["prim"]) * (0.7 * _smooth((0.1 - local[:, 2]) / 0.9))[:, None]
        return (tips * (quills[0] * (1 - 0.35 * quills[1]) * _fringes(q, fp))[:, None] if quills is not None
                else tips), 0.0
    tail = c["tail"] + (c["band"] - c["tail"]) * _smooth((-x - 0.465) / 0.02)[:, None]  # (its dark band, the last
    # 2.5 cm or so)
    return (tail * (quills[0] * (1 - 0.25 * quills[1]) * _fringes(q, fp))[:, None] if quills is not None
            else tail), 0.0


QUILL_ROW = 0.0208  # (the long feathers of its tail and wingtips: this far apart, side by side)


def _fringes(q, fp):
    """The long feathers' pale edges, where each lies over the next (about 1 .. 1.25): faded out where they would be
    under a few pixels apart (`fp`: a pixel's width, units)."""
    k = _fade(TAU / QUILL_ROW, fp)
    if k <= 0:
        return 1.0
    r = (np.abs(q[:, 1]) + 0.3 * (0.25 - q[:, 2])) / QUILL_ROW
    fr = r - np.floor(r)
    return 1.0 + 0.25 * k * _smooth((fr - 0.78) / 0.12) * _smooth((1.0 - fr) / 0.06)


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
    white, small = np.zeros(count), np.zeros(count)  # (a pied bird's white, part by part; its small feathers)
    gloss = np.zeros(count)  # (the iridescent feathers: glossy)
    sharp = [sk ** 4 for sk in shares]  # (each part's colour mostly its own, smoothly blended where they meet)
    total = np.maximum(sum(sharp), 1e-12)
    for (name, ck, rk, rad), (_, c0, r0, rad0), s in zip(body, _RESTING, [sk / total for sk in sharp]):
        idx = np.flatnonzero(s > 1e-4)
        if not len(idx):
            continue
        local = ((p[idx] - ck) @ rk) / rad
        qk = c0 + (local * rad0) @ r0.T
        nz = (n[idx] @ rk) @ r0.T[:, 2]  # (the way it faces, turned as the part is)
        colour, shine = _part_colour(name, qk, local, rad, nz, facing[idx], hood[idx], c, coat, fp)
        out[idx] += s[idx, None] * colour
        gloss[idx] += s[idx] * shine
        whole[idx] += s[idx, None] * qk
        if coat.pied:
            white[idx] += s[idx] * coat.white_on(name, qk, local, fp)
            if name not in ("tail", "fan", "prim"):  # (flecks on its small feathers, not its long ones)
                small[idx] += s[idx]
        if name == "head":
            w_head[idx] += s[idx]
    # its eyes: small, orange (a white dove's dark), deeper red round the iris' rim, a thin pale ring of bare skin, a
    # black pupil
    _, hc, hrot, hr = body[SMOOTH - 1]
    loc = ((p - hc) @ hrot) / hr
    edge = max(0.04, fp / 0.05)
    for side in (1.0, -1.0):
        d = np.sqrt((loc[:, 0] - 0.45) ** 2 + (loc[:, 2] - 0.22) ** 2 + 0.3 * (loc[:, 1] - side * 0.9) ** 2)
        on = (side * loc[:, 1] > 0.3) * w_head
        for size, colour, k in ((0.22, _lin((168, 156, 150)), 0.6), (0.155, c["iris"] * 0.55, 1.0),
                                (0.125, c["iris"], 1.0), (0.065, _lin((12, 10, 10)), 1.0)):
            m = (k * _smooth((size - d) / edge) * on)[:, None]
            out = out * (1 - m) + colour * m
        glint = np.sqrt((loc[:, 0] - 0.49) ** 2 + (loc[:, 2] - 0.27) ** 2 + 0.3 * (loc[:, 1] - side * 0.9) ** 2)
        m = (0.85 * _smooth((0.035 - glint) / edge) * on)[:, None]  # (the light caught in it)
        out = out * (1 - m) + 0.9 * m
    if coat.pied:  # (white where its white parts are, with clean edges where they meet the rest, and a few flecks)
        white = np.maximum(_smooth((white - 0.5) / 0.3 + 0.5), coat.white(whole, fp) * small)[:, None]
        out = out * (1 - white) + WHITE * white
        gloss *= 1.0 - white[:, 0]
    return out * (coat.tone * coat.grain(whole, fp))[:, None], gloss * min(c["sheen"], 1.0)


SMALL = {"leg": "feet", "toe": "feet", "phal": "feet", "claw": "beak", "beak": "beak", "cere": "cere"}  # (each small
# part's colour)


SCALY = {"leg": 2, "phal": 0, "toe": 0}  # (its legs and toes, scaly: the axis each runs along)


def _scaly(part, p, nrm, alb, fp):
    """A leg's or a toe's colour at points p on it (facing nrm): rings of scales across it SCALES apart, each a little
    lighter or darker, their edges dark (faded out where they would be under a few pixels: `fp`, a pixel's width),
    its underside paler, its joints' creases dark; and a little gloss (the scales shine)."""
    name, c, rot, r = part
    local = (p - c) @ rot
    along = local[:, SCALY[name]]
    k = _fade(TAU / SCALES, fp)
    out = alb.copy()
    if k > 0:
        s = along / SCALES
        ring = np.floor(s)
        f = s - ring
        rnd = np.sin(ring * 12.9898 + float(c @ np.array([78.2, 39.4, 11.7]))) * 43758.5453
        rnd -= np.floor(rnd)
        edge = 1 - _smooth(np.minimum(f, 1 - f) / 0.24)
        out *= (1 - k * (0.2 * edge - 0.1 * (rnd - 0.5)))[:, None]
    if name != "leg":  # (a toe's joints: creases where one piece goes into the next; its pads paler under it)
        end = np.abs(local[:, 0]) / r[0]
        out *= (1 - 0.28 * _smooth((end - 0.72) / 0.2))[:, None]
        out *= (1 + 0.22 * _smooth((-nrm[:, 2] - 0.2) / 0.5))[:, None]
    return out, np.full(len(p), 0.25)


def _trace(rays, body, small):
    """What each ray meets first: which (-1 nothing, 0 the body, 1 + k the k-th small part), its distance, and for a
    small part its own normal (not of unit length); the body as these rays see it."""
    seen = _Body(rays, body)
    best = seen.march()
    which = np.where(np.isfinite(best), 0, -1)
    own = np.zeros((len(best), 3))
    starts = [k for k, part in enumerate(small) if k == 0 or part[0] in ("beak", "leg") and small[k - 1][0] != "leg"]
    for g0, g1 in zip(starts, starts[1:] + [len(small)]):  # (the beak, each foot: only the rays near it tried)
        group = small[g0:g1]
        lo = np.min([c - r.max() for _, c, _, r in group], axis=0)
        hi = np.max([c + r.max() for _, c, _, r in group], axis=0)
        mid, reach = (lo + hi) / 2, float(np.linalg.norm(hi - lo)) / 2
        ex, ey, ez = rays.ox - mid[0], rays.oy - mid[1], rays.oz - mid[2]
        along = ex * rays.d[0] + ey * rays.d[1] + ez * rays.d[2]
        near = np.flatnonzero(ex * ex + ey * ey + ez * ez - along * along < reach * reach)
        if not len(near):
            continue
        sub = _Rays(rays.ox[near], rays.oy[near], rays.oz[near], rays.d)
        for k, part in enumerate(group, g0):
            t0, _ = sub.meet(part)
            ok = ~np.isnan(t0) & (t0 < best[near])
            if ok.any():
                idx = near[ok]
                best[idx] = t0[ok]
                which[idx] = 1 + k
                _, c, rot, r = part
                at = np.stack([sub.ox[ok], sub.oy[ok], sub.oz[ok]], 1) + t0[ok, None] * rays.d
                own[idx] = ((((at - c) @ rot) / r) / r) @ rot.T
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
    gloss = np.zeros(len(i))
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
        alb[on], gloss[on] = _albedo(pb, nrm[on], -rays.d, shares, seen.body, coat, pose, fp)
    for k, part in enumerate(small):
        j = np.flatnonzero(which[i] == 1 + k)
        if len(j):
            g = own[i[j]]
            nrm[j] = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
            alb[j] = coat.c[SMALL[part[0]]]
            if part[0] in SCALY:
                alb[j], gloss[j] = _scaly(part, p[j], nrm[j], alb[j], fp)
            elif part[0] == "claw":
                gloss[j] = 0.5
    key, way, ambient = light
    nw = nrm @ turn.T  # (the scene's frame)
    lam = np.clip((nw @ way + 0.35) / 1.35, 0.0, 1.0)  # (feathers: the light wraps round a little)
    sky = 0.55 + 0.45 * np.clip(0.5 + 0.5 * nw[:, 2], 0.0, 1.0)
    # feathers soft as velvet: a little more light where the surface is seen edge on (RIM)
    rim = (1 - np.clip(np.abs(nrm @ rays.d), 0.0, 1.0)) ** 3
    open_ = _occlusion(p, nrm, seen.body)  # (the creases between its parts, under it: less light reaches them)
    rgb[:] = alb * (ambient * (sky * open_)[:, None] + key * (lam * (0.4 + 0.6 * open_))[:, None]
                    + (RIM * rim * open_)[:, None] * (key + ambient))
    half = way - rays.d @ turn.T
    half /= np.linalg.norm(half)
    facing = np.clip(nw @ half, 0.0, 1.0)
    if gloss.any():  # (the iridescent neck, its legs and claws: a faint, soft highlight of their own colour where
        # they mirror the light)
        shine = facing ** 16 * gloss * open_
        rgb += (GLOSS * shine)[:, None] * key * (0.25 + alb)
    if SATIN:  # (all its feathers satin: a broad sheen where they face between the light and the camera)
        rgb += (SATIN * facing ** 6 * open_ * (which[i] == 0))[:, None] * key * (0.5 + 0.5 * alb)
    return i, rgb


def draw_bird(pose, coat, yaw, scale, tilt, light, ss, at=(0.0, 0.0)):
    """A bird seen by a far camera tilted down by `tilt` (radians), heading `yaw` (0: to the right of the picture,
    pi/2: away), `scale` pixels a unit: its picture (premultiplied linear RGB, alpha) and the pixel its feet are in
    (x, y), `at` (x, y: 0..1) across that pixel (drawn where it is to a fraction of a pixel, so it glides). `light`:
    (key colour, key way, ambient colour) linear, the way in the scene (x right, y away, z up). A bird DETAIL pixels
    long or more: one ray a pixel, `ss` x `ss` where its outline or a small part's crosses a pixel; a smaller one: `ss`
    x `ss` everywhere."""
    allp = parts(pose, fine=scale * TOES[1][1][0] * len(TOES[1][1]) >= FINE)
    if scale * 0.01 < 1.0:  # (claws: once they would show)
        allp = [part for part in allp if part[0] != "claw"]
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

    def rays(px, py):  # (through picture points px, py: pixels from the corner of the pixel its feet are in)
        u, v = (px - at[0]) / scale, (py - at[1]) / scale
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
WALK = 0.33  # (its walking speed, units a second, at "speed" 50: about two steps a second; the owner, 2026-10-25:
# slower)
PECK = np.array([0.31, 0.0, 0.099])  # (its head's middle as its beak touches the ground)
WALK_UP, TAIL_WALK = 0.12, 0.05  # (walking: how far its body is tipped up and its tail bent up, radians)
HEAD_UP = np.array([0.123, 0.0, 0.6])  # (its head's middle standing tall, its neck stretched)
DOWN, PECKING, UP = 0.36, 0.45, 0.42  # (seconds: bending down to peck, each peck, coming up again)
RAMP = 0.5  # (seconds: starting to walk, stopping)
EASE = 0.3  # (seconds: into and out of walking, pecking, standing)
THRUST = 0.4  # (its head thrust forward over this share of each step, held still the rest)
SIT_TIME = 0.8  # (seconds: sitting down, getting up again)
STAMP = 0.3  # (seconds: courting, from puffing up to stepping on the spot, and from stopping to calming down)
BOWS = 1.1  # (courting: bows a second)
# Where it goes (the owner, 2026-10-25: "chim xoay tròn là không giống nhất"): walks nearly straight, turning between
# them on the spot a step at a time, as a pigeon does; never round and round.
ARC = 0.45  # (radians: the most a walk curves, end to end)
BEND = 1.0  # (units: the tightest it curves as it walks)
TURN_RATE, TURN_FAST = 1.7, 2.6  # (radians a second: turning on the spot; at the fastest, short of time)
TURN_STEP = 0.75  # (radians: the most it turns at a step)
TURN_LEAST = 0.45  # (seconds: the least a turn on the spot takes: a step, then its other foot put down beside it)
TURN_MOST = 3.5  # (radians: at most a little over half round at once)
TINY = 0.06  # (radians: a turn this small it makes as it walks off, not on the spot)
BACK = 0.15  # (how often, between walks, it turns round to go back the way it came)
ZIGZAG = 0.65  # (how often it turns the other way from its turn before: zigzagging, not going round its home)
LONGER = (0.8, 1.25)  # (how much shorter or longer a walk on the ground may be made, a little slower or faster, so that
# its path closes)
SPIN_WALK, SPIN_TIME = 1.5 * LENGTH, 8.0  # (units, seconds: never more than a little over half round with less
# than this walked, in less than this time; some to spare: the test is a bird's length, 6 seconds)
WANDERS = 24  # (ways tried for each bird on the ground: the one keeping best within its room, changed least)
ROOM = 2.0  # (units: how far its path may go from its home, when no room is given)
CALM = 1.2, 5.0  # (with walks no longer than this (units) and less, seconds at least from one walk to the next, up to
# this with no room at all: a bird with little room walks less often, pecking and looking round between)
STOP_LEAST = {"stand": 0.5, "rest": 2 * SIT_TIME + 0.8, "preen": 1.4, "court": 2.0}  # (seconds: what a stop keeps
# when a turn on the spot takes time from it; pecking, 0.85 of its pecks' own time)


def _walk_time(far, speed):
    """How long a walk `far` units long takes at `speed`, speeding up and slowing down over RAMP each (as _into)."""
    x = far / max(speed, 1e-9)
    return x + RAMP if x >= RAMP else 2.0 * x


def _chords(lengths, arcs, heads):
    """Where walks end, from where each starts (2 x n, units): each `lengths` long, setting off along `heads`,
    curving `arcs` radians end to end (an arc of a circle)."""
    k = np.asarray(lengths, np.float64) * np.sinc(np.asarray(arcs, np.float64) / TAU)
    mid = np.asarray(heads, np.float64) + np.asarray(arcs, np.float64) / 2
    return np.stack([k * np.cos(mid), k * np.sin(mid)])


def _closed(lengths, arcs, heads, rounds=24, turn=1.0):
    """The walks' headings changed as little as may be (and their lengths a little, within LONGER) for a path of them
    to close: their chords adding up to nothing (Gauss-Newton: each round the least change of the headings and of the
    lengths' logarithms closing it to first order, no heading more than half a radian a round; `turn`: how freely a
    heading changes, a length's logarithm 0.3). The headings, each walk's length factor, how far it still misses."""
    lengths, arcs = np.asarray(lengths, np.float64), np.asarray(arcs, np.float64)
    n = len(lengths)
    th, lf = np.array(heads, np.float64), np.zeros(n)
    lo, hi = math.log(LONGER[0]), math.log(LONGER[1])
    give = np.concatenate([np.full(n, turn), np.full(n, 0.3)])
    for _ in range(rounds):
        c = _chords(lengths * np.exp(lf), arcs, th)
        miss = c.sum(axis=1)
        if math.hypot(*miss) < 1e-9:
            break
        jac = np.concatenate([np.stack([-c[1], c[0]]), c], axis=1)
        step = -(give[:, None] * jac.T) @ np.linalg.solve((jac * give) @ jac.T + 1e-12 * np.eye(2), miss)
        step *= min(1.0, 0.5 / max(float(np.abs(step[:n]).max()), 1e-12))
        th += step[:n]
        lf = np.clip(lf + step[n:], lo, hi)
    miss = _chords(lengths * np.exp(lf), arcs, th).sum(axis=1)
    return th, np.exp(lf), math.hypot(*miss)


def _turns(heads, arcs):
    """How far it turns before each walk, from the way the walk before it ended (radians, -pi .. pi)."""
    return np.remainder(heads - np.roll(heads + arcs, 1) + math.pi, TAU) - math.pi


def _spins(turns, fars, gaps, arcs=None):
    """How far round it turns, all told, at most, in a while (less than SPIN_WALK walked, SPIN_TIME gone by between
    its turns: radians, from one way it faces to the farthest round from it, either way). `turns[j]` comes before
    walk j, `fars[j]` long, curving `arcs[j]`, `gaps[j]` seconds before the next turn."""
    n, most = len(turns), 0.0
    for j in range(n):
        lo = hi = at = walked = gone = 0.0
        for k in range(n):
            if k:
                q = (j + k - 1) % n
                walked, gone = walked + fars[q], gone + gaps[q]
                at += 0.0 if arcs is None else arcs[q]
                lo, hi = min(lo, at), max(hi, at)
                if walked >= SPIN_WALK or gone >= SPIN_TIME:
                    break
            at += turns[(j + k) % n]
            lo, hi = min(lo, at), max(hi, at)
        most = max(most, hi - lo)
    return most


def _unspun(turns, fars, gaps, search=False, alternate=True):
    """Its turns, each that may go either way round (one near half round: no more than TURN_MOST the other way)
    turning round one way, then back the other (`alternate`): never round and round. `search`: then each such turned
    the other way round too, kept where it turns less far round in a while (_spins)."""
    turns = np.array(turns, np.float64)
    either = np.flatnonzero(np.abs(turns) >= TAU - TURN_MOST)
    way = math.copysign(1.0, turns[either[0]]) if len(either) else 1.0
    for j in either if alternate else ():
        if math.copysign(1.0, turns[j]) != way:
            turns[j] -= TAU * math.copysign(1.0, turns[j])
        way = -way
    for _ in range(3 if search else 0):
        better = False
        for j in either:
            other = turns.copy()
            other[j] -= TAU * math.copysign(1.0, turns[j])
            if _spins(other, fars, gaps) < _spins(turns, fars, gaps) - 1e-9:
                turns, better = other, True
        if not better:
            break
    return turns


def _wander(rng, lengths, gaps, room, thin):
    """Which way each walk on the ground goes and how much it curves: from where the walk before it ended, a turn this
    way or that, now and then right round (BACK), drawn back toward its home the farther out it is (its room, longer
    along a way of its own the more `thin`), the walks then closed (_closed). Of WANDERS tries, the one keeping best
    within its room, changed least to close, its turns smallest. Headings, curves, length factors."""
    n = len(lengths)
    axis = TAU * rng.random()
    far, near = room * (1.0 + 1.5 * thin), room * (1.0 - 0.4 * thin)  # (along its own way, across it)
    ca, sa = math.cos(axis), math.sin(axis)
    best, least = None, math.inf
    for _ in range(WANDERS):
        arcs = np.clip(rng.normal(0.0, ARC / 2, n), -ARC, ARC)
        arcs = np.clip(arcs, -lengths / BEND, lengths / BEND)
        heads, p, way = np.zeros(n), np.zeros(2), rng.choice((-1.0, 1.0))
        heads[0] = TAU * rng.random()
        lately = []  # (its turns and walks lately, with less than SPIN_WALK walked since)
        for i in range(n):
            if i:
                end = heads[i - 1] + arcs[i - 1]
                way = -way if rng.random() < ZIGZAG else way  # (this way, then that, mostly: not round and round)
                if rng.random() < BACK:
                    head = end + way * rng.uniform(2.3, 3.1)
                else:
                    head = end + way * abs(float(np.clip(rng.normal(0.0, 0.9), -2.2, 2.2)))
                qa, qb = (p[0] * ca + p[1] * sa) / far, (p[1] * ca - p[0] * sa) / near
                left = float(lengths[i:].sum())  # (as far as it has still to walk)
                pull = min(max((math.hypot(qa, qb) - 0.3) / 0.7, (math.hypot(*p) / max(left, 1e-9) - 0.35) / 0.45,
                               0.0), 1.0)
                if pull > 0:  # (the farther out, and the less it has still to walk, the more it heads home)
                    head += pull * math.remainder(math.atan2(-p[1], -p[0]) - head, TAU)
                if thin > 0:  # (roaming: mostly along its own way, to and fro)
                    along = axis if math.cos(head - axis) >= 0 else axis + math.pi
                    head += 0.6 * thin * math.remainder(along - head, TAU)
                turn, net = math.remainder(head - end, TAU), sum(t for t, _, _ in lately)
                if abs(net + turn) > 0.95 * math.pi:  # (turned round lately: back the other way, or not as far)
                    other = turn - math.copysign(TAU, turn)
                    turn = other if abs(net + other) < abs(net + turn) and abs(other) <= TURN_MOST else float(
                        np.clip(net + turn, -0.95 * math.pi, 0.95 * math.pi) - net)
                heads[i] = end + turn
                lately.append((turn + arcs[i], lengths[i], gaps[i]))
            else:
                lately.append((arcs[0], lengths[0], gaps[0]))
            while lately and (sum(w for _, w, _ in lately) >= SPIN_WALK or sum(g for _, _, g in lately) >= SPIN_TIME):
                lately.pop(0)
            p += _chords(lengths[i:i + 1], arcs[i:i + 1], heads[i:i + 1])[:, 0]
        th, f, miss = _closed(lengths, arcs, heads)
        c = _chords(lengths * f, arcs, th)
        starts = np.concatenate([np.zeros((2, 1)), np.cumsum(c, axis=1)[:, :-1]], axis=1)
        pts = np.concatenate([starts, starts + _chords(lengths * f / 2, arcs / 2, th)], axis=1)
        u, v = pts[0] * ca + pts[1] * sa, pts[1] * ca - pts[0] * sa
        out = max(float(np.ptp(u)) / (2 * far), float(np.ptp(v)) / (2 * near))
        cost = (4.0 * max(out - 1.0, 0.0) + float(np.abs(th - heads).mean()) + 2.0 * float(np.abs(np.log(f)).mean())
                + 20.0 * miss + _spun(th, arcs, lengths * f, gaps))
        if cost < least:
            best, least = (th, arcs, f), cost
    # to and fro along a line of its own (a bird with little room, or little walking, can always do that)
    ways, f = _to_and_fro(lengths, min(far, near * 1.5))
    th, arcs = axis + np.where(ways > 0, 0.0, math.pi), np.zeros(n)
    if 0.6 + 2.0 * float(np.abs(np.log(f)).mean()) + _spun(th, arcs, lengths * f, gaps) < least:
        best = (th, arcs, f)
    return best


def _spun(heads, arcs, fars, gaps):
    """How much a way of walking turns round and round (a cost): its turns' size, whole turns round its home over
    the loop, and above all turning more than a little over half round in a while (_spins)."""
    turns = _unspun(_turns(heads, arcs), fars, gaps)
    rounds = abs(round(float(turns.sum() + arcs.sum()) / TAU))
    return (0.05 * float(np.abs(turns).mean()) + 0.6 * rounds
            + 20.0 * max(_spins(turns, fars, gaps) - 1.1 * math.pi, 0.0))


def _to_and_fro(lengths, room, ways=None):
    """To and fro along a line (a ledge's, or on the ground one of its own): which way each walk goes (1 or -1 along
    it) and each one's length factor, so that it ends where it began, within its stretch (room each side of its
    middle): of the ways tried (every one, or as many at random for many walks), the one closing with its walks
    changed least, within its stretch, turning round least; closed by making the walks one way a little longer, the
    others shorter (a little faster, slower). `ways` given: those."""
    n = len(lengths)
    if ways is None:
        if n <= 12:
            bits = (np.arange(2 ** (n - 1))[:, None] >> np.arange(n - 1)[None, :]) & 1
        else:
            bits = np.random.default_rng(n).integers(0, 2, (4096, n - 1))
        ways = np.concatenate([np.ones((len(bits), 1)), 1.0 - 2.0 * bits], axis=1)
    ways = np.atleast_2d(np.asarray(ways, np.float64))
    net = (ways * lengths).sum(axis=1)
    on = ways == np.where(net >= 0, 1.0, -1.0)[:, None]  # (the walks going the way it would end up)
    big, small = (lengths * on).sum(axis=1), (lengths * ~on).sum(axis=1)
    whole = np.maximum(big + small, 1e-12)
    f = np.where(on, (2 * small / whole)[:, None], (2 * big / whole)[:, None])
    at = np.cumsum(ways * lengths * f, axis=1)
    span = np.maximum(at.max(axis=1), 0.0) - np.minimum(at.min(axis=1), 0.0)
    with np.errstate(divide="ignore"):
        changed = np.abs(np.log(np.maximum(f, 1e-300))).max(axis=1)
    over = span / max(2 * room, 1e-9) - 1.0
    turned = (ways != np.roll(ways, 1, axis=1)).sum(axis=1)  # (how often it turns round)
    cost = changed + 100.0 * (over > 1e-3) + 3.0 * np.maximum(over, 0.0) + 0.03 * turned
    q = int(np.argmin(cost))
    return ways[q], f[q] / max(1.0, 1.0 + float(over[q]))  # (none within its stretch: all a little shorter)


class Walker:
    """One bird on the ground over the loop (`loop` seconds): walking a while, nearly straight, then stopping to look
    round, to peck at the ground, to sit down a while, to preen or (a male) to court, puffed up; turning on the spot
    between its walks, a step at a time, as pigeons do, never round and round (the owner, 2026-10-25); at the loop's
    end where it began, facing the same way. `busy`: 0 (mostly stopped) .. 1 (walking most of the time); `hungry`: 0
    (never pecks) .. 1 (pecks at most stops, a few times); `rest`: 0 .. 1, how often it sits down or preens; `court`:
    0 .. 1, how often it courts (none unless `male`); `ledge`: to and fro along a line, turning round on the spot at
    its stops; `room`: the farthest its path may go from its home (units): walks no longer than it allows; `thin`:
    0 .. 1, its path longer along a way of its own, to and fro (roaming further)."""

    def __init__(self, rng, loop, busy, pace, ledge=False, room=None, hungry=0.4, rest=0.0, court=0.0, male=False,
                 thin=0.0):
        self.loop = loop
        segs = self._day(rng, loop, busy, hungry, rest, court, male)
        speed = WALK * pace * rng.uniform(0.85, 1.15)
        self.speed, self.ramp, self.room = speed, RAMP, room
        reach = ROOM if room is None else room
        far = reach * (1.0 + 1.5 * thin)
        segs = self._fit(segs, speed, 1.6 * reach if ledge else 0.8 * far, 0.8 if ledge else 0.9)
        legs = [i for i, s in enumerate(segs) if s[0] == "walk"]
        lengths = np.array([speed * (segs[i][2] - min(RAMP, segs[i][2] / 2)) for i in legs])
        gaps = np.diff(np.r_[[segs[i][1] for i in legs], legs and segs[legs[0]][1] + loop])  # (seconds from each
        # walk's start to the next one's)
        for line in (ledge, True):  # (about; to and fro along a line of its own if that would still spin round)
            path = self._path(rng, segs, legs, lengths, gaps, speed, reach, thin, ledge, line)
            if path[-1] <= 1.2 * math.pi:
                break
        segs, legs, fars, heads, arcs, turns, _ = path
        self._lay(segs, legs, fars, heads, arcs, turns)
        self.looks = rng.random(8) * TAU
        self.shift = 0.0  # (seconds: its day begun this far into the loop, out of the other birds' way)
        self._samples = None

    def _path(self, rng, segs, legs, lengths, gaps, speed, reach, thin, ledge, line):
        """Where its walks go (`line`: to and fro along a line, a ledge's or, on the ground, one of its own), its
        turns on the spot put into its day; then its walks closed again, some shortened by the turns before them
        (by their lengths, its turns kept as they go). Its day, its walks' places in it, their lengths, headings,
        curves, the turns before them, and how far round it turns at most in a while (_spins)."""
        if line:
            ways, f = _to_and_fro(lengths, reach)
            heads, arcs = (0.0 if ledge else TAU * rng.random()) + np.where(ways > 0, 0.0, math.pi), np.zeros(len(legs))
        else:
            heads, arcs, f = _wander(rng, lengths, gaps, reach, thin)
        turns = _unspun(_turns(heads, arcs), lengths * f, gaps, search=True)
        heads, arcs, turns = self._absorb(heads, arcs, turns, lengths * f)
        segs, legs = self._on_the_spot(rng, segs, legs, turns)
        lengths = np.array([speed * (segs[i][2] - min(RAMP, segs[i][2] / 2)) for i in legs])
        if line:
            _, f = _to_and_fro(lengths, reach, ways)
        else:
            heads, f, _ = _closed(lengths, arcs, heads, turn=0.02)  # (its lengths, not its turns, if it can)
            turns = turns + np.remainder(_turns(heads, arcs) - turns + math.pi, TAU) - math.pi
        spots = {self._serves(segs, k): k for k, s in enumerate(segs) if s[0] == "turn"}
        when = [segs[spots[j]][1] if j in spots else segs[i][1] for j, i in enumerate(legs)]  # (each turn's time)
        gaps = np.remainder(np.roll(when, -1) - np.array(when), self.loop)
        turns = _unspun(turns, lengths * f, gaps, search=True, alternate=False)  # (its walks now as they are)
        spot = np.array([j in spots for j in range(len(legs))], bool)
        arcs = np.where(spot, arcs, arcs + turns)  # (no turn on the spot there: made as it walks off)
        heads = np.where(spot, heads, heads - turns)
        turns = np.where(spot, turns, 0.0)
        return segs, legs, lengths * f, heads, arcs, turns, _spins(turns, lengths * f, gaps, arcs)

    @staticmethod
    def _day(rng, loop, busy, hungry, rest, court, male):
        """What it does over the loop, by turns walking and stopping: [kind, start, seconds, n] (n: its pecks, which
        side it preens), the loop's length in all."""
        segs, at = [], 0.0
        walk = rng.random() < 0.6
        sits, preens, courts = 0.4 * rest, 0.3 * rest, (0.5 * court if male else 0.0)  # (of its stops)
        while at < loop * 1.02 or len(segs) < 2:
            pick = rng.random(2)
            if walk:
                kind, d, n = "walk", (1.2 + 2.3 * pick[0]) * (0.6 + 0.8 * busy), 0
            elif (r := rng.random()) < courts:  # (puffed up, bowing, stepping on the spot)
                kind, d, n = "court", 3.0 + 3.5 * pick[0], 0
            elif r < courts + sits:  # (sitting down a while)
                kind, d, n = "rest", (4.0 + 5.0 * pick[0]) * (1.3 - 0.6 * busy), 0
            elif r < courts + sits + preens:  # (preening a wing: its left or its right)
                kind, d, n = "preen", 2.5 + 3.0 * pick[0], (1, -1, 0)[min(int(pick[1] * 3), 2)]
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
        for s in segs:
            s[2] *= k
        return Walker._timed(segs)

    @staticmethod
    def _timed(segs):
        at = 0.0
        for s in segs:
            s[1] = at
            at += s[2]
        return segs

    @staticmethod
    def _fit(segs, speed, cap, balance):
        """Its walks fitted to its room: none longer than `cap` (units: the rest of its time it stands there), with
        little room fewer of them (CALM), a lone walk made two (there and back again), the longest no longer than
        `balance` of the others together (else its path could not close)."""
        segs = [list(s) for s in segs]

        def far(i):
            return speed * (segs[i][2] - min(RAMP, segs[i][2] / 2))

        def shorten(i, most):  # (walk i no longer than `most`: the time saved spent standing after it)
            spare = segs[i][2] - _walk_time(most, speed)
            if spare > 1e-9:
                segs[i][2] -= spare
                if segs[(i + 1) % len(segs)][0] == "stand":
                    segs[(i + 1) % len(segs)][2] += spare
                else:
                    segs.insert(i + 1, ["stand", 0.0, spare, 0])

        for i in [i for i, s in enumerate(segs) if s[0] == "walk"][::-1]:
            if far(i) > cap:
                shorten(i, cap)
        apart = CALM[1] * min(max((CALM[0] - cap) / (CALM[0] - 0.4), 0.0), 1.0)  # (little room: fewer walks)
        walks = [i for i, s in enumerate(segs) if s[0] == "walk"]
        last = -math.inf
        for i in walks:
            if segs[i][1] - last < apart and len([s for s in segs if s[0] == "walk"]) > 2:
                segs[i][0] = "stand"  # (standing there instead)
            else:
                last = segs[i][1]
        walks = [i for i, s in enumerate(segs) if s[0] == "walk"]
        if len(walks) == 1:  # (one walk: there and back again, a pause between)
            i = walks[0]
            d = segs[i][2]
            pause = min(max(0.8, 0.3 * d), 0.5 * d)
            segs[i][2] = (d - pause) / 2
            segs[i + 1:i + 1] = [["stand", 0.0, pause, 0], ["walk", 0.0, (d - pause) / 2, 0]]
        for _ in range(4):
            walks = [i for i, s in enumerate(segs) if s[0] == "walk"]
            fars = np.array([far(i) for i in walks])
            most = int(np.argmax(fars))
            rest = float(fars.sum() - fars[most])
            if fars[most] <= balance * rest + 1e-9:
                break
            shorten(walks[most], balance * rest)
        merged = []  # (standing on, one stand)
        for s in segs:
            if merged and s[0] == "stand" == merged[-1][0]:
                merged[-1][2] += s[2]
            else:
                merged.append(s)
        return Walker._timed(merged)

    @staticmethod
    def _absorb(heads, arcs, turns, fars):
        """A turn too small to turn on the spot for, made as it walks off: its walk curving that much more (if it may:
        no tighter than BEND)."""
        small = (np.abs(turns) <= TINY) & (np.abs(arcs + turns) <= fars / BEND)
        return heads - turns * small, arcs + turns * small, np.where(small, 0.0, turns)

    @staticmethod
    def _serves(segs, k):
        """The walk a turn on the spot (segment k) comes before: its number among the walks (round the loop)."""
        n = len(segs)
        walks = [i for i, s in enumerate(segs) if s[0] == "walk"]
        for step in range(1, n + 1):
            i = (k + step) % n
            if segs[i][0] == "walk":
                return walks.index(i)
        return 0

    def _on_the_spot(self, rng, segs, legs, turns):
        """A turn on the spot before each walk that needs one (a segment of its own: ["turn", start, seconds, steps]),
        as long as its turn takes (TURN_RATE; TURN_LEAST at the least), out of the stop between the walks with the
        most time to spare (as much as it can spare: STOP_LEAST), after it (or now and then before it); short of time,
        quicker (TURN_FAST), the rest out of the walk. The day, and its walks' places in it."""
        segs = [list(s) for s in segs]
        n = len(legs)

        def least(stop):  # (what a stop keeps of its time)
            kind, _, _, k = segs[stop]
            return 0.85 * (DOWN + k * PECKING + UP) if kind == "peck" else STOP_LEAST.get(kind, 0.5)

        for j in range(n):
            if turns[j] == 0.0:
                continue
            walks = [i for i, s in enumerate(segs) if s[0] == "walk"]
            w = walks[j]
            between = [(walks[j - 1] + k) % len(segs) for k in range(1, (w - walks[j - 1]) % len(segs) or len(segs))]
            stop = max(between, key=lambda i: segs[i][2] - least(i)) if between else (w - 1) % len(segs)  # (the stop
            # with most time to spare: the turn before it now and then (as it stops), else after it)
            after = rng.random() < 0.35
            d = segs[stop][2]
            most = abs(turns[j])  # (as far as it may turn: either way round, if it may go either way; and a little
            # more, for its walks closed again)
            most = (max(most, TAU - most) if most >= TAU - TURN_MOST else most) + 0.05
            want = max(TURN_LEAST, most / TURN_RATE)
            need = max(0.6 * TURN_LEAST, most / TURN_FAST)
            took = min(want, max(d - least(stop), 0.0))
            segs[stop][2] -= took
            if took < need:  # (short of time: out of the walk too, as much as 70% of it; then the stop, but a little)
                cut = min(need - took, 0.7 * segs[w][2])
                segs[w][2] -= cut
                more = min(need - took - cut, max(segs[stop][2] - 0.2, 0.0))
                segs[stop][2] -= more
                took += cut + more
            steps = max(1, int(math.ceil(abs(turns[j]) / TURN_STEP)))
            segs.insert(stop if after else stop + 1, ["turn", 0.0, took, steps])
        segs = self._timed(segs)
        return segs, [i for i, s in enumerate(segs) if s[0] == "walk"]

    def _lay(self, segs, legs, fars, heads, arcs, turns):
        """Its day laid out: each segment's heading (a + b * how far through it is: a walk's distance, a turn's
        steps), each walk's speed and stride, how far it has walked before each, and its path (a table by distance
        walked: the walks one after another, what it still misses at the end spread along it, so it closes)."""
        self.segs = segs
        n = len(segs)
        leg_of = {i: j for j, i in enumerate(legs)}
        self.speeds, self.strides, self.walked, self.heads = [0.0] * n, [STRIDE] * n, [0.0] * n, [None] * n
        first = legs[0] if legs else 0
        h = float(heads[0]) if len(legs) else 0.0
        for step in range(n):
            i = (first + step) % n
            kind, _, d, _ = segs[i]
            if kind == "walk":
                j = leg_of[i]
                self.heads[i] = (h, float(arcs[j]))
                h += float(arcs[j])
                self.speeds[i] = fars[j] / max(d - min(RAMP, d / 2), 1e-9)
                self.strides[i] = fars[j] / (math.ceil(fars[j] / (STRIDE / 2) - 1e-9) / 2) if fars[j] > 0 else STRIDE
            elif kind == "turn":
                turn = float(turns[self._serves(segs, i)])
                self.heads[i] = (h, turn)
                h += turn
            else:
                self.heads[i] = (h, 0.0)
        wraps = round((h - float(heads[0] if len(legs) else 0.0)) / TAU)  # (whole turns round over the loop)
        for i in range(first):  # (before its first walk: as it was at the end of the loop, less those turns)
            a, b = self.heads[i]
            self.heads[i] = (a - TAU * wraps, b)
        pts, cum, p, total = [np.zeros(2)], [0.0], np.zeros(2), 0.0
        for i, (kind, _, d, _) in enumerate(segs):
            self.walked[i] = total
            if kind != "walk" or fars[leg_of[i]] <= 1e-9:
                continue
            far = fars[leg_of[i]]
            q = np.linspace(0.0, far, max(2, int(math.ceil(far * 48))) + 1)[1:]
            c = _chords(q, self.heads[i][1] * q / far, np.full(len(q), self.heads[i][0]))
            pts.extend(p + c.T)
            cum.extend(total + q)
            p, total = p + c[:, -1], total + far
        pts, cum = np.array(pts), np.array(cum)
        if total > 1e-9:
            pts -= p[None, :] * (cum / total)[:, None]  # (closed: what it misses spread along its way)
        pts -= (pts.min(axis=0) + pts.max(axis=0)) / 2  # (its home the middle of its path)
        self.px, self.py, self.cum = pts[:, 0], pts[:, 1], cum
        self.length = max(total, 1e-6)  # (how far it walks over the loop)

    def _seg(self, t):
        for i in range(len(self.segs) - 1, -1, -1):
            if t >= self.segs[i][1]:
                return i
        return 0

    def _into(self, t):
        """How far into its walk it is at t (units; 0 if not walking): its speed rising smoothly from 0 over `ramp`
        seconds, steady, falling smoothly to 0 (a smoothstep: no jolt)."""
        i = self._seg(t)
        kind, start, d, _ = self.segs[i]
        if kind != "walk":
            return i, 0.0
        r = min(self.ramp, d / 2)
        x, full = t - start, d - r  # (the distance this walk covers, in seconds at speed)

        def ramp(x):  # (how far, speeding up from 0 over r seconds: the integral of a smoothstep)
            x = min(max(x, 0.0), r)
            return x ** 3 / r ** 2 - x ** 4 / (2 * r ** 3)

        s = ramp(x) if x < r else (full - ramp(d - x) if x > d - r else x - r / 2)
        return i, self.speeds[i] * min(max(s, 0.0), full)

    def pace_at(self, t):
        """How fast it walks at t: a share of its walking speed (0 standing, turning on the spot .. 1 striding
        along), rising and falling smoothly as it starts and stops (the speed _into goes at)."""
        return self._pace((t + self.shift) % self.loop)

    def _pace(self, t):
        kind, start, d, _ = self.segs[self._seg(t)]
        if kind != "walk":
            return 0.0
        r = min(self.ramp, d / 2)
        x = min(t - start, start + d - t)
        return float(_smooth(x / r)) if x < r else 1.0

    def walked_at(self, t):
        """How far round its path it has walked at t (seconds into the loop)."""
        i, s = self._into(t)
        return (self.walked[i] + s) % self.length

    def where(self, t):
        """Where it is (units from its home, across and away) and which way it faces (radians)."""
        return self._where((t + self.shift) % self.loop)

    def samples(self, k):
        """Where it is, which way it faces and how fast it walks (where, pace_at) at k moments evenly round the loop,
        from its start: four rows, made once for its day begun at the loop's start, turned round by its shift (a
        whole number of moments, as _spread gives it)."""
        if self._samples is None or self._samples.shape[1] != k:
            ts = np.arange(k) * (self.loop / k)
            self._samples = np.array([(*self._where(t), self._pace(t)) for t in ts]).T
        m = self.shift * k / self.loop
        if abs(m - round(m)) < 1e-6:
            return np.roll(self._samples, -int(round(m)) % k, axis=1)
        ts = np.arange(k) * (self.loop / k)
        return np.array([(*self.where(t), self.pace_at(t)) for t in ts]).T

    def _where(self, t):
        i, into = self._into(t)
        kind, start, d, _ = self.segs[i]
        s = (self.walked[i] + into) % self.length
        x, y = float(np.interp(s, self.cum, self.px)), float(np.interp(s, self.cum, self.py))
        a, b = self.heads[i]
        if kind == "walk":
            far = self.speeds[i] * (d - min(self.ramp, d / 2))
            return x, y, a + b * (into / far if far > 0 else 0.0)
        if kind == "turn":
            return x, y, a + b * self._turned(i, t - start)
        return x, y, a

    def _turned(self, i, x):
        """How far through a turn on the spot it is, x seconds into it (0 .. 1): its body turning with each step it
        takes (a little faster in the middle of a step), still as it brings its other foot beside the last."""
        _, _, d, m = self.segs[i]
        tau = d / (m + 1)
        j = min(int(x / tau), m)
        if j >= m:
            return 1.0
        u = min(max(x / tau - j, 0.0), 1.0)
        return (j + 0.5 * u + 0.5 * float(_smoother(u))) / m

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
        """Its pose at t: stepping and bobbing its head as it walks, turning on the spot, pecking, looking round,
        sitting, preening, courting."""
        t = (t + self.shift) % self.loop
        i, into = self._into(t)
        kind, start, d, n = self.segs[i]
        x = t - start
        ease = float(_smooth(min(x, d - x) / EASE))  # (into and out of what it does)
        pose = Pose(pitch=-WALK_UP, tail=TAIL_WALK)
        if kind == "walk":
            stride = self.strides[i]
            phase = (0.25 + into / stride) % 1.0  # (from its feet together, one about to step: a whole number of
            # steps, so its feet are together again as it stops)
            # its head: held still where it is as the body goes on, then thrust forward (a bob a step: two a stride)
            step = (2 * phase) % 1.0
            thrust = float(_smooth(step / THRUST)) if step < THRUST else 1.0
            bob = (thrust - step - (0.5 - THRUST / 2)) * stride / 2 * 0.9 * ease
            pose.head = HEAD + np.array([bob, 0.0, -0.008 * ease * math.sin(TAU * step)])
            # its feet: one planted (going back under it as much as it goes on), the other lifted and going forward
            feet = []
            for side, off in ((1, 0.0), (-1, 0.5)):
                ph = (phase + off) % 1.0
                if ph < 0.5:
                    fx, lift = stride * (0.25 - ph), 0.0
                else:
                    g = (ph - 0.5) / 0.5
                    fx, lift = stride * (-0.25 + 0.5 * float(_smooth(g))), 0.028 * math.sin(math.pi * g)
                feet.append((fx, side * 0.034, lift * ease))
            pose.feet = tuple(feet)
            # its body over the foot it stands on: a little to that side, rolled, highest as it passes over it
            pose.sway = 0.006 * ease * math.sin(TAU * phase)
            pose.roll = -0.05 * ease * math.sin(TAU * phase)
            pose.bob = -0.003 * ease * math.cos(2 * TAU * phase)
        elif kind == "turn":  # turning on the spot, a step at a time: each foot put down where the body will be turned
            # at the end of its step, staying there as the body turns over it; its other foot brought beside at last
            turn = self.heads[i][1]
            tau = d / (n + 1)
            j = min(int(x / tau), n)
            u = min(max(x / tau - j, 0.0), 1.0)
            body = turn * self._turned(i, x)
            lead = 1 if turn >= 0 else -1  # (the foot on the side it turns to steps first)
            feet = []
            for side in (1, -1):
                first = 0 if side == lead else 1

                def put(k):  # (where this foot was put down at the end of its step k: turned as far as the body)
                    return turn * min(k + 1, n) / n if k >= first else 0.0

                if j >= first and (j - first) % 2 == 0:  # (stepping now)
                    was = put(j - 2)
                    way, lift = was + (put(j) - was) * float(_smooth(u)), 0.02 * math.sin(math.pi * u)
                else:
                    way, lift = put(j - 1 if (j - 1 - first) % 2 == 0 else j - 2), 0.0
                rel = way - body  # (the foot's own way, in the bird's frame)
                feet.append((-side * 0.034 * math.sin(rel), side * 0.034 * math.cos(rel), lift, rel))
            pose.feet = tuple(feet)
            stand = -lead if j % 2 == 0 else lead  # (the foot it stands on)
            pose.sway = 0.004 * stand * math.sin(math.pi * u)
            pose.roll = -0.03 * stand * math.sin(math.pi * u)
            pose.head = HEAD + np.array([0.0, 0.0, -0.005 * math.sin(math.pi * u)])
            pose.head_yaw = lead * min(0.45, 0.4 * abs(turn)) * math.sin(math.pi * min(1.25 * x / d, 1.0))  # (its
            # head turned ahead of it)
        elif kind == "peck":
            reach = self.reach(t)
            pose.pitch = -WALK_UP + (0.45 + WALK_UP) * reach  # (its body tips forward, its tail rises)
            pose.tail = TAIL_WALK * (1 - reach)
            pose.head = HEAD + (PECK - HEAD) * reach
            pose.head_pitch = 0.17 - 1.25 * reach
        elif kind == "rest":  # sitting down on the ground, feathers fluffed, its head drawn in; looking round
            sit = float(_smoother(min(x, d - x) / SIT_TIME))
            pose.sit = pose.tuck = sit
            pose.pitch = -WALK_UP * (1 - sit)
            pose.tail = TAIL_WALK * (1 - sit)
            k = int(x / 2.6)
            f = float(_smoother((x - k * 2.6) / 0.6))
            turn = 0.5 * (math.sin(self.looks[k % 8]) * f + math.sin(self.looks[(k - 1) % 8]) * (1 - f) * (k > 0))
            pose.head_yaw = turn * sit
            pose.head = HEAD + np.array([-0.03, 0.0, -0.11]) * sit
        elif kind == "preen":  # its head turned back over a shoulder (or down to its breast), nibbling at its
            # feathers
            side = float(n)
            e = float(_smoother(min(x, d - x) / 0.45))
            nibble = np.array([0.006 * math.sin(TAU * 3.3 * x), 0.0, 0.009 * math.sin(TAU * 4.1 * x + 1.0)])
            if side:
                pose.head = HEAD + (np.array([-0.05, side * 0.035, 0.395]) - HEAD + nibble) * e
                pose.head_yaw = side * 2.6 * e
                pose.head_pitch = 0.17 - 0.45 * e
                pose.roll = side * 0.06 * e
            else:
                pose.head = HEAD + (np.array([0.205, 0.0, 0.215]) - HEAD + nibble) * e
                pose.head_pitch = 0.17 - 1.15 * e
        elif kind == "court":  # a male courting: puffed up, its tail spread and dragged, bowing, stepping on the spot
            # (never turning round: the owner, 2026-10-25)
            e = float(_smoother(min(x, d - x) / 0.5))
            pose.puff, pose.spread = e, e
            pose.tail = TAIL_WALK * (1 - e) - 0.32 * e  # (down to the ground, spread)
            bow = 0.5 * (1 - math.cos(TAU * BOWS * x)) * e
            pose.pitch = -WALK_UP * (1 - e) + 0.26 * bow
            pose.head = HEAD + np.array([0.0, 0.0, 0.025]) * e + np.array([0.05, 0.0, -0.1]) * bow
            pose.head_pitch = 0.17 - 0.3 * bow
            step = math.sin(TAU * 2.0 * x) * e * float(_smooth(min(x - STAMP, d - STAMP - x) / 0.3))
            pose.feet = ((0.0, 0.034, 0.012 * max(step, 0.0)), (0.0, -0.034, 0.012 * max(-step, 0.0)))
        else:  # standing: looking round now and then (a quick turn of the head, then still)
            k = int(x / 1.3)
            f = float(_smoother((x - k * 1.3) / 0.3))
            turn = 0.7 * (math.sin(self.looks[k % 8]) * f + math.sin(self.looks[(k - 1) % 8]) * (1 - f) * (k > 0))
            pose.head_yaw = turn * ease
            tall = float(_smoother(min(x, d - x) / 0.6))  # (standing up tall, breast out, then down again to go
            # on: its head first)
            pose.tall = tall
            pose.pitch = -WALK_UP * (1 - tall)
            pose.tail = TAIL_WALK * (1 - tall)
            pose.head = HEAD + (HEAD_UP - HEAD) * float(_smoother(min(x, d - x) / 0.45))
            pose.head_pitch = 0.17 - 0.07 * tall
        return pose


# --- keeping out of each other's way ----------------------------------------------------------------------------------

FOOT = (0.36, 0.13, -0.1)  # (a bird's footprint on the ground, an ellipse round its body: half its length and half
# its width (units), its middle this far ahead of its feet: behind them)
APART = 0.04  # (units kept between two birds' footprints)
PLAN_RATE = 5  # (the flock's plan: samples a second)
ROUNDS = 60  # (rounds of pushing the birds apart and smoothing their steps aside)
AGAIN = 10  # (every so many rounds, which may meet and how they meet again, as they now go)
SLACK = 1.0  # (units: two birds whose footprints come this close may meet within the next AGAIN rounds)
SIDESTEP = 0.8  # (two birds moving past each other mostly step aside, across their way past each other)
PASSING = 0.15  # (units a second: two birds moving past each other at least this fast step aside fully)
MEETING = 0.5  # (two birds meet while their footprints come within this (units) of each other: an encounter)
EASE_WALK, EASE_STOP = 12.0, 4000.0  # (how smoothly a bird steps aside: walking, over a few samples; standing still,
# not at all: it never slides)
HOLD = 0.02  # (how strongly each bird keeps to its own path)
LIMIT = 2.0  # (units: the most two birds push each other further than they overlap)
SPOTS = 32  # (places and times tried for each bird on the ground: the first where it never comes near the birds
# before it)
SPARE = 0.15  # (units: the room kept between two birds' paths when they are placed)
REF = 1080.0  # (px: the height of the box a flock is laid out in)
DROPS = 4  # (times over, at most, leaving out a bird that still walks into others and planning again)
CROWD = 0.15  # (units: placed, no bird overlaps another deeper than this: those the birds step round)
LEDGE = 1.1  # (units: the least stretch of a ledge a bird walks to and fro on, its tail turning round at the ends)
OVERLAP = 0.02  # (units: two birds overlapping more than this still walk into each other)
SWEEPS = 2  # (moving the birds that still meet others to better places: at most this many times over)
TIGHTER, SHRINK = 1, 0.6  # (then, crowded: giving those that still meet others smaller paths, this many times,
# each this much smaller)
_PLANS = collections.OrderedDict()  # (the last flocks' layouts, by everything they depend on: a flock's slow part)
_PLANS_LOCK = threading.Lock()


def _reach(s, ux, uz, nx, nz):
    """How far birds' footprints (ellipses s px a unit, along unit vectors (ux, uz) on the ground) reach along unit
    vectors (nx, nz)."""
    along = nx * ux + nz * uz
    return s * np.sqrt(FOOT[1] ** 2 + (FOOT[0] ** 2 - FOOT[1] ** 2) * along * along)


class _Smoother:
    """Smoothing a bird's steps aside over the loop: for each bird (rows), the offsets o minimising, round the loop,
    HOLD |o_k|^2 + |o_k - target_k|^2 + ease_k |o_(k+1) - o_k|^2 (ease_k between samples k and k+1): a cyclic
    tridiagonal system, factored once (the Thomas algorithm, Sherman-Morrison for its corners)."""

    def __init__(self, ease):
        n, k = ease.shape
        before = np.roll(ease, 1, axis=1)
        self.a, self.c = -before, -ease  # (each row's coefficient of the sample before it, after it)
        b = HOLD + 1.0 + before + ease
        self.gamma = -b[:, 0]
        self.beta, self.alpha = self.a[:, 0], self.c[:, -1]  # (the corners: row 0's last, the last row's first)
        b = b.copy()
        b[:, 0] -= self.gamma
        b[:, -1] -= self.alpha * self.beta / self.gamma
        self.cp, self.den = np.zeros((n, k)), np.zeros((n, k))
        self.den[:, 0] = b[:, 0]
        self.cp[:, 0] = self.c[:, 0] / b[:, 0]
        for j in range(1, k):
            self.den[:, j] = b[:, j] - self.a[:, j] * self.cp[:, j - 1]
            self.cp[:, j] = self.c[:, j] / self.den[:, j]
        u = np.zeros((n, k, 1))
        u[:, 0, 0], u[:, -1, 0] = self.gamma, self.alpha
        self.z = self._thomas(u)

    def _thomas(self, r):
        n, k = self.den.shape
        y = np.zeros_like(r)
        y[:, 0] = r[:, 0] / self.den[:, 0, None]
        for j in range(1, k):
            y[:, j] = (r[:, j] - self.a[:, j, None] * y[:, j - 1]) / self.den[:, j, None]
        for j in range(k - 2, -1, -1):
            y[:, j] -= self.cp[:, j, None] * y[:, j + 1]
        return y

    def solve(self, target):
        x = self._thomas(target)
        ratio = (self.beta / self.gamma)[:, None]
        fact = (x[:, 0] + ratio * x[:, -1]) / (1.0 + self.z[:, 0] + ratio * self.z[:, -1])
        return x - fact[:, None, :] * self.z


def _passing(d, w, near):
    """How two birds get past each other in each encounter (pairs x samples; d: from the first to the second, w: the
    second's velocity less the first's, on the ground; near: when they are near each other): a unit vector across
    their way past each other (perpendicular to w), toward the side the second passes the first on, the same side all
    through the encounter (a run of samples near each other, round the loop). Each steps aside along it, the second
    with it, the first against it: one walking at the other steps round it, one meeting the other they keep to their
    sides, one crossing the other's way slows down a little as the other hurries past. Zero where they are not near or
    hardly move past each other."""
    speed = np.hypot(w[..., 0], w[..., 1])
    px, pz = -w[..., 1] / np.maximum(speed, 1e-9), w[..., 0] / np.maximum(speed, 1e-9)
    across = d[..., 0] * px + d[..., 1] * pz  # (how far the second is to that side)
    out = np.zeros(d.shape)
    k = near.shape[1]
    for q in np.flatnonzero(near.any(axis=1)):
        row = near[q]
        starts = np.flatnonzero(row & ~np.roll(row, 1)) if not row.all() else np.array([0])
        for s0 in starts:
            run = [s0]
            while row[(run[-1] + 1) % k] and len(run) < k:
                run.append((run[-1] + 1) % k)
            run = np.array(run)
            side = -1.0 if across[q, run].mean() < 0 else 1.0
            out[q, run, 0], out[q, run, 1] = side * px[q, run], side * pz[q, run]
    return out


def _meetings(at, i, j, way, scales, gap, loop):
    """How pairs of birds (i, j) meet, going as they do (at: their footprints' middles, samples round the loop; way:
    which way each faces): in each encounter, which way each steps aside (_passing), and how much aside rather than
    straight apart (as fast as they move past each other)."""
    k = at.shape[1]
    d = at[j] - at[i]
    dist = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-9)
    need = (_reach(scales[i][:, None], way[i][..., 0], way[i][..., 1], d[..., 0] / dist, d[..., 1] / dist)
            + _reach(scales[j][:, None], way[j][..., 0], way[j][..., 1], d[..., 0] / dist, d[..., 1] / dist) + gap)
    near = need + MEETING * 0.5 * (scales[i] + scales[j])[:, None] > dist
    vel = (np.roll(at, -1, axis=1) - np.roll(at, 1, axis=1)) * (k / (2.0 * loop))  # (ground px a second)
    rel = vel[j] - vel[i]
    past = _passing(d, rel, near)
    aside = SIDESTEP * _smooth(np.hypot(rel[..., 0], rel[..., 1]) / (PASSING * 0.5 * (scales[i] + scales[j])[:, None]))
    return past, aside[..., None] * (np.abs(past).sum(-1) > 0)[..., None]  # (moving past each other: aside)


def _close(at, scales, pairs, pushing):
    """Which of the pairs (two rows of bird numbers) may meet over the next rounds: their footprints' middles
    (at: samples round the loop) come within reach of each other, SLACK units to spare, anywhere round the loop (by
    the boxes round their ways), or they still push each other apart (pushing: n x n x samples)."""
    a, b = pairs
    lo, hi = at.min(axis=1), at.max(axis=1)
    apart = np.maximum(lo[b] - hi[a], lo[a] - hi[b]).max(axis=1)
    near = (apart < (FOOT[0] + SLACK) * (scales[a] + scales[b])) | pushing[a, b].any(axis=1)
    return a[near], b[near]


def _plan(walkers, homes, scales, box, tilt, loop):
    """How far (ground pixels: across the box, toward the camera) each bird steps aside from its own path at each
    moment of the loop (PLAN_RATE samples a second, round the loop), so that no two birds walk through each other or
    stand in each other: wherever their footprints would overlap they are pushed apart (a walking bird mostly, aside
    from its way; one standing still hardly), within the box, and each bird's steps aside smoothed over the loop (a
    standing bird's not at all: it does not slide; a walking one's over a few steps). Where they still overlap, the
    push grows round after round (a method of multipliers: the smoothed steps aside then clear each other). None
    when no two come near; and how much each two still overlap (units, at worst: an n x n table)."""
    n = len(walkers)
    w, h = box
    st = math.sin(tilt)
    k = max(16, int(round(loop * PLAN_RATE)))
    feet, heading, go = np.zeros((n, k, 2)), np.zeros((n, k)), np.zeros((n, k))
    for q, (walker, home) in enumerate(zip(walkers, homes)):
        dx, dy, heading[q], go[q] = walker.samples(k)
        feet[q, :, 0] = home[0] * w + dx * scales[q]
        feet[q, :, 1] = (home[1] * h - dy * scales[q] * st) / st
    stride = np.array([walker.speed for walker in walkers])[:, None] * go * scales[:, None]  # (walking: px a second)

    def facing(off):  # (which way each faces, stepping aside as it walks: as _ground turns it)
        moved = (np.roll(off, -1, axis=1) - np.roll(off, 1, axis=1)) * (k / (2.0 * loop))
        ax, az = stride * np.cos(heading) + moved[..., 0], -stride * np.sin(heading) + moved[..., 1]
        turned = heading + go * (np.mod(np.arctan2(-az, ax) - heading + math.pi, TAU) - math.pi)
        return np.stack([np.cos(turned), -np.sin(turned)], -1)

    way = facing(np.zeros((n, k, 2)))
    centre = feet + FOOT[2] * scales[:, None, None] * way
    every = np.triu_indices(n, 1)
    if not (np.linalg.norm(centre[every[1]] - centre[every[0]], axis=-1).min(axis=1)
            < 3.0 * FOOT[0] * (scales[every[0]] + scales[every[1]])).any():
        return None, np.zeros((n, n))
    moving = 0.5 * (go + np.roll(go, -1, axis=1))
    smoother = _Smoother(EASE_WALK + EASE_STOP * (1.0 - moving) ** 2)
    mobile = 0.06 + go
    reach = 0.6 * scales.max() * LENGTH  # (kept inside the box: as its paths are)
    lo = np.array([reach, 1.3 * reach / st])
    hi = np.array([w - reach, 0.97 * h / st])
    off = np.zeros((n, k, 2))
    pushing = np.zeros((n, n, k))  # (how much further than their overlap each pair pushes, grown round after round
    # while they still overlap: the steps aside, smoothed, then clear each other)
    below, above = np.zeros((n, k, 2)), np.zeros((n, k, 2))  # (the same, keeping them inside the box)
    i = j = more = None
    for r in range(ROUNDS):
        way = facing(off)
        at = feet + off + FOOT[2] * scales[:, None, None] * way
        if r % AGAIN == 0:  # (which may meet, and how they meet now: each one's way past the other, as they go)
            if i is not None:
                pushing[i, j] = more
            i, j = _close(at, scales, every, pushing)
            share = mobile[i] / (mobile[i] + mobile[j])
            gap = APART * 0.5 * (scales[i] + scales[j])[:, None]
            most = LIMIT * 0.5 * (scales[i] + scales[j])[:, None]
            more = pushing[i, j]
            past, aside = _meetings(at, i, j, way, scales, gap, loop)
        d = at[j] - at[i]
        dist = np.hypot(d[..., 0], d[..., 1])
        flat = dist < 1e-9  # (one right on the other: apart across the first one's way)
        nx = np.where(flat, -way[i][..., 1], d[..., 0] / np.maximum(dist, 1e-9))
        nz = np.where(flat, way[i][..., 0], d[..., 1] / np.maximum(dist, 1e-9))
        need = (_reach(scales[i][:, None], way[i][..., 0], way[i][..., 1], nx, nz)
                + _reach(scales[j][:, None], way[j][..., 0], way[j][..., 1], nx, nz) + gap)
        over = need - dist
        low, high = lo - at, at - hi
        if ((not over.size or float(over.max()) < 1e-3 * float(scales.min())) and float(low.max()) < 1e-3
                and float(high.max()) < 1e-3):
            break
        push = np.maximum(low + below, 0.0) - np.maximum(high + above, 0.0)  # (back into the box)
        across = past - (past[..., 0] * nx + past[..., 1] * nz)[..., None] * np.stack([nx, nz], -1)
        apart = np.stack([nx, nz], -1) * (1.0 - aside) + across * aside  # (straight apart; moving past each other,
        # mostly aside, each to its side: never toward each other)
        apart /= np.maximum(np.hypot(apart[..., 0], apart[..., 1]), 1e-9)[..., None]
        force = np.maximum(over + more, 0.0)
        np.add.at(push, i, -apart * (force * share)[..., None])
        np.add.at(push, j, apart * (force * (1.0 - share))[..., None])
        more = np.minimum(np.maximum(more + over, 0.0), most)
        below, above = np.maximum(below + low, 0.0), np.maximum(above + high, 0.0)
        off = smoother.solve(off + push)
    i, j = every  # (what each two still overlap: units, at worst)
    way = facing(off)
    at = feet + off + FOOT[2] * scales[:, None, None] * way
    d = at[j] - at[i]
    dist = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-9)
    nx, nz = d[..., 0] / dist, d[..., 1] / dist
    need = (_reach(scales[i][:, None], way[i][..., 0], way[i][..., 1], nx, nz)
            + _reach(scales[j][:, None], way[j][..., 0], way[j][..., 1], nx, nz))
    left = np.zeros((n, n))
    left[i, j] = left[j, i] = np.maximum(need - dist, 0.0).max(axis=1) / (0.5 * (scales[i] + scales[j]))
    return off, left


def _along_loop(plan, i, x):
    """Bird i's step aside at x samples round the loop (Catmull-Rom between the plan's samples, round the loop), and
    how fast it changes (a sample)."""
    k = plan.shape[1]
    j = math.floor(x)
    f = x - j
    p0, p1, p2, p3 = (plan[i, (j + d) % k] for d in (-1, 0, 1, 2))
    a, b, c = 2 * p0 - 5 * p1 + 4 * p2 - p3, -p0 + 3 * p1 - 3 * p2 + p3, p2 - p0
    return 0.5 * (2 * p1 + c * f + a * f * f + b * f ** 3), 0.5 * (c + 2 * a * f + 3 * b * f * f)


# --- the effect -------------------------------------------------------------------------------------------------------

DEFAULTS = {"mode": "ground", "coat": "checker_dark", "birds": 4, "size": 82, "depth": 20, "view": 40, "shadow": 60,
            "soft": 30, "busy": 50, "peck": 40, "rest": 30, "court": 25, "speed": 50, "roam": 50, "differ": 60,
            "lit": 70}  # (a few birds filling their box, the box their place in the video: the owner, 2026-10-26, "vừa
# với khung để tôi có thể đặt vào các vị trí trong video", "chỉ cần 3-5 con chim bồ câu")

SCHEMA = {
    "label": "Chim bồ câu",
    "note": "Chim bồ câu vẽ bằng app (không phải ảnh có sẵn): đi lại, gật gù, đứng ngó quanh, thỉnh thoảng cúi xuống "
            "mổ hạt. Khung trắng là một nhóm chim: chim to vừa khung. (1) Bấm Suốt video. (2) Kéo khung trắng tới "
            "chỗ chim đứng (sân, vỉa hè, đường đi); kéo góc khung cho chim to nhỏ hợp với cảnh (to như vật bên cạnh): "
            "chim chỉ ở trong khung. Muốn chim ở nhiều chỗ: thêm hiệu ứng nữa, đặt khung ở chỗ khác. Đáy khung là gần, "
            "đỉnh khung là xa. Trên gờ tường, mái nhà: chọn Kiểu Đậu trên gờ, kéo khung cho đáy khung nằm trên gờ. "
            "(3) Chỉnh Số chim (3 đến 5 con là vừa; ít chim thì xem nhanh có sớm hơn), Cỡ chim (so với khung), Màu "
            "lông; Nhìn từ trên xuống cho hợp góc máy; Độ mềm cho chim mềm như trong ảnh chụp. (4) Chuyển động: "
            "Đi lại nhiều, Mổ hạt, Nằm nghỉ và rỉa lông, Chim trống gù (phồng cổ, xoè đuôi, cúi chào), Tốc độ, Đi "
            "quanh quẩn hay đi khắp khung; Mỗi con một kiểu càng lớn thì mỗi con càng khác: con hay đi, con "
            "hay mổ, con lười nằm nghỉ, con đi nhanh, con đi chậm. Muốn đàn chim làm khác đi: bấm Đổi kiểu ngẫu "
            "nhiên. Chim tự tránh nhau, không đi xuyên qua nhau; chim to mà nhiều quá thì khung không đủ chỗ: app "
            "bớt chim. Chim sáng tối theo ánh sáng chỗ khung đứng trong video.",
    "params": [
        _choice("mode", "Kiểu", DEFAULTS["mode"], [("ground", "Đi lại, mổ hạt trên mặt đất"),
                                                   ("ledge", "Đậu trên gờ tường, mái nhà")], group="Kiểu"),
        _choice("coat", "Màu lông", DEFAULTS["coat"], [("checker_dark", "Xám có đốm và xám đen"),
                                                       ("mixed", "Nhiều màu như đàn chim phố"),
                                                       ("grey", "Xám xanh (bồ câu thường)"),
                                                       ("checker", "Xám có đốm"), ("pied", "Loang trắng"),
                                                       ("white", "Bồ câu trắng"), ("brown", "Nâu đỏ"),
                                                       ("dark", "Xám đen")], group="Kiểu"),
        _num("birds", "Số chim", DEFAULTS["birds"], 1, 30, group="Chim"),
        _num("size", "Cỡ chim", DEFAULTS["size"], group="Chim"),
        _num("depth", "Gần xa (chim xa nhỏ hơn)", DEFAULTS["depth"], group="Chim"),
        _num("view", "Nhìn từ trên xuống (ngang ← → cao)", DEFAULTS["view"], group="Chim"),
        _num("shadow", "Bóng dưới chân", DEFAULTS["shadow"], group="Chim"),
        _num("soft", "Độ mềm (nét ← → mềm như ảnh chụp)", DEFAULTS["soft"], group="Chim"),
        _num("busy", "Đi lại nhiều (đứng ← → đi)", DEFAULTS["busy"], group="Chuyển động"),
        _num("peck", "Mổ hạt (ít ← → nhiều)", DEFAULTS["peck"], group="Chuyển động"),
        _num("rest", "Nằm nghỉ, rỉa lông (ít ← → nhiều)", DEFAULTS["rest"], group="Chuyển động"),
        _num("court", "Chim trống gù, xoè đuôi (ít ← → nhiều)", DEFAULTS["court"], group="Chuyển động"),
        _num("speed", "Tốc độ", DEFAULTS["speed"], group="Chuyển động"),
        _num("roam", "Đi quanh quẩn ← → đi khắp khung", DEFAULTS["roam"], group="Chuyển động"),
        _num("differ", "Mỗi con một kiểu (giống nhau ← → khác nhau)", DEFAULTS["differ"], group="Chuyển động"),
        _num("lit", "Sáng theo cảnh", DEFAULTS["lit"], group="Ánh sáng"),
        *_place(35)]}

DAYLIGHT = (np.array([1.15, 1.1, 1.04], np.float32), np.array([0.5, 0.54, 0.6], np.float32))  # key, ambient


def count(mode, amount):
    """How many birds an "amount" (0..100) made: "Số chim" before it was the number itself (1.0.87 and before)."""
    return int(round(CAP[mode] * (amount / 100) ** 1.2))


def upgrade(raw):
    """Settings saved by an older version: "Số chim" was an amount (0..100), now it is the number of birds."""
    if not isinstance(raw, dict) or "birds" in raw or "amount" not in raw:
        return raw
    try:
        amount = min(max(float(raw["amount"]), 0.0), 100.0)
    except (TypeError, ValueError, OverflowError):
        amount = 40.0
    amount = 40.0 if amount != amount else amount
    out = {k: v for k, v in raw.items() if k != "amount"}
    out["birds"] = max(1, count(raw.get("mode") if raw.get("mode") in CAP else "ground", amount))
    return out


class Pigeons(Effect):
    BOX = (16, 9)
    TURNS = False
    LIT = True
    PREVIEW = 1280
    VERSION = 11  # (2: one smooth body, pecking now and then; 3: rows of feathers, soft as velvet, sturdier legs;
    # 4: standing tall as a photograph's, its folded wings and their primaries, the neck's sheen round its foot;
    # 5: smooth paths, steps, glides; creases shaded, a glossy feathered neck, catchlights, claws, the photo's colours;
    # 6: four scaly toes a foot, joint by joint, curling as a foot lifts; sitting, preening, courting; each its ways;
    # 7: the photograph's curved, scalloped wing bars, its colours measured again, a streaked neck, a purple bib,
    # feathered thighs, pale fringes on the long feathers, a satin sheen, checker wedges, the tail lower standing;
    # 8: out of each other's way: placed where and when they seldom meet, stepping aside, those the box has no room
    # for left out, a stretch of its own each along a ledge; 9: walks nearly straight, turning on the spot between
    # them a step at a time, never round and round, courting without turning round; 10: a pied bird's white on whole
    # parts of it, clean-edged; its tail's end a thin fan, its band dark grey; softer checks; a red one coppery; slower;
    # matte: no satin sheen, a fainter rim, a fainter gloss; 11: the checker's crisp dark wedges again, as the owner
    # chose it)
    light_at = staticmethod(steam.light_at)
    rounded_light = staticmethod(steam.rounded_light)
    upgrade = staticmethod(upgrade)

    def __init__(self, *a, light=None):
        super().__init__(*a)
        p = self.p
        self.mode = p["mode"] if p["mode"] in CAP else "ground"
        rng = np.random.default_rng([self.seed, 4407])
        n = min(int(p["birds"]), CAP[self.mode])
        self.size = SMALLEST * (LARGEST / SMALLEST) ** (p["size"] / 100)  # (a bird's length, box heights, nearest)
        self.rw, self.rh = REF * self.BOX[0] / self.BOX[1], REF  # (the flock laid out in a box this size, whatever
        # size it is drawn at: the same birds in the same places doing the same, in the preview clip and the video)
        w, h = self.rw, self.rh
        if self.mode == "ledge":  # (a ledge holds only so many birds, each walking to and fro on its own stretch)
            n = min(n, max(1, int((w - 1.2 * self.size * h) / (LEDGE * self.size * h / LENGTH))))
        coats = []
        first = None
        for k in range(CAP[self.mode]):
            r, name = rng.random(), p["coat"]  # (drawn whatever the coat: choosing another moves no bird)
            if name == "checker_dark":  # (spread evenly over the flock: so many of each, however few birds)
                first = r if first is None else first
                r = (first + k * GOLDEN) % 1.0
            if name in MIXES:
                mix = MIXES[name]
                name = mix[-1][0]
                for nm, share in mix:
                    if r < share:
                        name = nm
                        break
                    r -= share
            coats.append(Coat(name, rng))
        self.coats = coats[:n]
        u = rng.random((CAP[self.mode], 8))[:n]
        self.u = u
        self.pace = 2.0 ** ((p["speed"] - 50) / 50)
        self.tilt = math.radians(_lerp(8, 50, p["view"] / 100))
        self.depth = p["depth"] / 100
        self.walkers, self.homes = [], []
        roam = p.get("roam", 50) / 100  # (round and round near home .. long narrow paths over the box)
        slots = self._slots(u, w, h) if self.mode == "ledge" else [None] * n
        self.thin = max(0.0, roam * 2 - 1)
        for i in range(n):
            scale = self._scale(u[i], h)
            if slots[i] is not None:  # (along a ledge: within its own stretch of it, its tail too)
                room = max(0.05, slots[i][1] / max(scale, 1e-9) - (FOOT[0] - FOOT[2]) - APART / 2)
            else:
                room = 0.16 * _lerp(0.35, 1.0, min(roam * 2, 1.0)) * w / max(scale, 1e-9)  # (units)
            self.walkers.append(self._walker(i, room))
        self.plan = None
        if self.mode == "ground" and n > 1:  # (on the ground: each where and when it seldom meets the others, out of
            # each other's way; too many for the box, those it has no room for left out)
            self._flock(w, h)
        else:
            self.homes = [self._home(walker, u[i], self._scale(u[i], h), w, h, slots[i] and slots[i][0])
                          for i, walker in enumerate(self.walkers)]
        self.light = self._light(light, p["lit"] / 100)  # (the scene's where the box stands, or daylight)

    def _walker(self, i, room):
        """Bird i's day (Walker), its paths within `room` (units) of its home."""
        busy, hungry, rest, court, pace, male = self.temper(i)
        return Walker(np.random.default_rng([self.seed, 4407, i]), self.loop_s, busy, pace, self.mode == "ledge",
                      room, hungry, rest, court, male, self.thin)

    def temper(self, i):
        """Bird i's own ways (each a little or, "Mỗi con một kiểu", very different from the flock's settings): how
        busy, how hungry, how often it rests or preens, how often it courts (a male: about half of them), how fast."""
        p = self.p
        k = p.get("differ", 0) / 100
        z = np.random.default_rng([self.seed, 4408, i]).normal(0.0, 1.0, 6)
        male = z[5] > 0

        def own(v, j):
            return float(np.clip(v + 0.5 * k * z[j], 0.0, 1.0))
        return (own(p["busy"] / 100, 0), own(p["peck"] / 100, 1), own(p.get("rest", 0) / 100, 2),
                own(p.get("court", 0) / 100, 3) if male else 0.0, self.pace * 2.0 ** (0.45 * k * float(z[4])), male)

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

    def _slots(self, u, w, h):
        """Along a ledge, each bird's own stretch of it, so that none walks through another: the ledge cut at random
        into as many stretches as birds, each at least LEDGE units long, given in the order of the birds' u[0]
        (where along it they would have been): each one's middle and half its width (px)."""
        n = len(u)
        if not n:
            return []
        reach = 0.6 * self.size * h
        least = LEDGE * self.size * h / LENGTH
        share = 0.6 + np.random.default_rng([self.seed, 4410]).random(n)
        widths = np.minimum(least, (w - 2 * reach) / n) + max(w - 2 * reach - n * least, 0.0) * share / share.sum()
        mids = reach + np.cumsum(widths) - widths / 2
        out = [None] * n
        for k, i in enumerate(np.argsort(u[:, 0], kind="stable")):
            out[i] = (float(mids[k]), float(widths[k] / 2))
        return out

    def _spread(self, w, h):
        """On the ground, each bird where and when its path meets the others' least (u[i, :2]: where across the box
        and how far away; its walker's shift: when in the loop its day begins): each in turn, of SPOTS places and
        times (its own first, then others at random), the first where it never comes near the birds before it, else
        the one where it overlaps them least; then those still meeting others moved again to wherever they meet them
        least (SWEEPS times over); then, crowded, those given smaller paths (SHRINK) and placed again; and too many
        for the box, the bird overlapping the others most left out, and the next, until none overlaps another deeper
        than CROWD. The birds kept (their numbers): they seldom need to step out of each other's way (_plan)."""
        n = len(self.walkers)
        k = max(16, int(round(self.loop_s * PLAN_RATE)))
        st = math.sin(self.tilt)
        spare = (SPARE + APART) / 2  # (each bird's share of the room kept between two)
        moves = [None] * n
        at, scales, ways, starts = np.zeros((n, k, 2)), np.zeros(n), np.zeros((n, k, 2)), np.zeros(n, int)

        def walk(i):  # (bird i's steps round the loop from its home, its day begun at the loop's start)
            moves[i] = self.walkers[i].samples(k)[:3]

        def put(i, tries):  # (bird i at each place and time tried: its footprint's middle and way round the loop)
            out, faces, sizes = np.zeros((len(tries), k, 2)), np.zeros((len(tries), k, 2)), np.zeros(len(tries))
            for q, (u0, u1, m) in enumerate(tries):
                c = np.array([u0, u1])
                scale = self._scale(c, h)
                home = self._home(self.walkers[i], c, scale, w, h)
                dx, dy, a = np.roll(moves[i], -int(m), axis=1)
                faces[q, :, 0], faces[q, :, 1] = np.cos(a), -np.sin(a)
                out[q, :, 0] = home[0] * w + dx * scale
                out[q, :, 1] = (home[1] * h - dy * scale * st) / st
                out[q] += FOOT[2] * scale * faces[q]
                sizes[q] = scale
            return out, faces, sizes

        def meets(places, faces, sizes, others, room=spare):  # (how much a bird overlaps the others at each
            # place: squared, summed round the loop; and at worst, units: only those whose ways come near compared)
            cost, depth = np.zeros(len(places)), np.zeros(len(places))
            if not len(others):
                return cost, depth
            lo, hi = at[others].min(axis=1), at[others].max(axis=1)
            apart = np.maximum(lo[None] - places.max(axis=1)[:, None], places.min(axis=1)[:, None] - hi[None])
            c, o = np.nonzero(apart.max(axis=2) < (FOOT[0] + room) * (sizes[:, None] + scales[others][None]))
            if not len(c):
                return cost, depth
            o = others[o]
            d = at[o] - places[c]
            dist = np.maximum(np.hypot(d[..., 0], d[..., 1]), 1e-9)
            nx, nz = d[..., 0] / dist, d[..., 1] / dist
            s0, s1 = sizes[c][:, None], scales[o][:, None]
            over = np.maximum(_reach(s0, faces[c][..., 0], faces[c][..., 1], nx, nz) + s0 * room
                              + _reach(s1, ways[o][..., 0], ways[o][..., 1], nx, nz) + s1 * room - dist, 0.0)
            np.maximum.at(depth, c, (over / (0.5 * (s0 + s1))).max(axis=1))
            return np.bincount(c, (over ** 2).sum(axis=1), len(places)), depth

        def place(i, others, sweep):  # (bird i where it meets the others least; at first, the first free place)
            rng = np.random.default_rng([self.seed, 4411, i, sweep])
            tries = np.column_stack([rng.random((SPOTS, 2)), rng.integers(0, k, SPOTS)])
            tries[0] = self.u[i, 0], self.u[i, 1], starts[i]
            places, faces, sizes = put(i, tries)
            c = meets(places, faces, sizes, others)[0]
            q = int(np.argmax(c == 0.0)) if sweep == 0 and (c == 0.0).any() else int(np.argmin(c))
            self.u[i, :2], starts[i] = tries[q, :2], int(tries[q, 2])
            at[i], ways[i], scales[i] = places[q], faces[q], sizes[q]

        def costs(alive, room=spare):  # (how much each bird overlaps the others alive, and at worst)
            out = np.zeros((2, n))
            for i in alive:
                out[:, i] = np.array(meets(at[i][None], ways[i][None], scales[i:i + 1], alive[alive != i], room))[:, 0]
            return out

        everyone = np.arange(n)
        for i in range(n):
            walk(i)
            place(i, everyone[:i], 0)
        for sweep in range(1, SWEEPS + TIGHTER + 1):
            cost = costs(everyone)[0]
            if not cost.any():
                break
            for i in np.flatnonzero(cost):
                if sweep > SWEEPS:  # (still meeting others: a smaller path, nearer its home)
                    self.walkers[i] = self._walker(i, self.walkers[i].room * SHRINK)
                    walk(i)
                place(i, everyone[everyone != i], sweep)
        for i in range(n):
            self.walkers[i].shift = starts[i] * self.loop_s / k
        alive = everyone
        while len(alive) > 1:  # (too many for the box: the bird overlapping the others most left out, and so on,
            # until none overlaps another deeper than CROWD: those the birds step round, out of each other's way)
            cost, depth = costs(alive, 0.0)
            if depth.max() <= CROWD:
                break
            alive = alive[alive != int(np.argmax(cost))]
        return alive

    def _keep(self, keep):
        """Only the birds `keep` (their numbers), the others left out."""
        self.u = self.u[keep]
        self.coats, self.walkers = [self.coats[i] for i in keep], [self.walkers[i] for i in keep]

    def _flock(self, w, h):
        """On the ground: each bird where and when it seldom meets the others (_spread), their steps aside out of
        each other's way where they still meet (_plan; a bird that still walks into others left out and the plan made
        again, at most DROPS times over), their homes. Kept for the next clip with the same birds (_PLANS): a big
        flock takes a while."""
        key = (self.seed, self.P, self.fps, round(w, 3), round(h, 3),
               tuple((k, self.p.get(k)) for k in ("mode", "birds", "size", "depth", "view", "busy", "peck", "rest",
                                                   "court", "speed", "roam", "differ")))
        with _PLANS_LOCK:
            made = _PLANS.pop(key, None) or self._lay_out(w, h)
            _PLANS[key] = made
            while len(_PLANS) > 4:
                _PLANS.popitem(last=False)
        places, shifts, rooms, kept, self.plan = made
        for i, walker in enumerate(self.walkers):
            if rooms[i] != walker.room:  # (given a smaller path)
                self.walkers[i] = self._walker(i, rooms[i])
            self.walkers[i].shift = shifts[i]
        self.u[:, :2] = places
        self._keep(kept)
        self.homes = [self._home(walker, u, self._scale(u, h), w, h) for walker, u in zip(self.walkers, self.u)]

    def _lay_out(self, w, h):
        """The flock on the ground (_flock): where each bird is (u[:, :2]), when its day begins, how far it walks,
        the birds kept (their numbers) and their steps aside."""
        alive = self._spread(w, h)
        places = self.u[:, :2].copy()
        shifts, rooms = [k.shift for k in self.walkers], [k.room for k in self.walkers]
        homes = [self._home(walker, u, self._scale(u, h), w, h) for walker, u in zip(self.walkers, self.u)]
        scales = np.array([self._scale(u, h) for u in self.u])
        kept = alive
        for attempt in range(DROPS + 1):
            plan, left = _plan([self.walkers[i] for i in kept], [homes[i] for i in kept], scales[kept], (w, h),
                               self.tilt, self.loop_s)
            if attempt == DROPS or left.max() <= OVERLAP:
                break
            kept = np.delete(kept, int(np.argmax(left.sum(axis=1))))  # (the one still overlapping others most)
        return places, shifts, rooms, kept, plan

    def _home(self, walker, u, scale, w, h, at=None):
        """Where a bird's path is (its middle: shares of the box across and down), all of it inside the box; along a
        ledge, in the middle of its own stretch of it (`at`, px across the box)."""
        sy = scale * math.sin(self.tilt)
        lo_x, hi_x = float(walker.px.min()) * scale, float(walker.px.max()) * scale
        lo_y, hi_y = -float(walker.py.max()) * sy, -float(walker.py.min()) * sy
        reach = 0.6 * self.size * h  # (a bird's own size round its feet)
        left, right = reach - lo_x, w - reach - hi_x
        if right <= left:
            hx = w / 2 - (lo_x + hi_x) / 2
        elif at is not None:
            hx = min(max(at - (lo_x + hi_x) / 2, left), right)
        else:
            hx = left + (right - left) * u[0]
        if self.mode == "ledge":  # (the near side of its path on the box's bottom edge, the far one a little up)
            return hx / w, 0.97 - hi_y / h
        top, bottom = 1.3 * reach - lo_y, h * 0.97 - hi_y  # (its head this high above its feet, standing tall)
        hy = bottom - (bottom - top) * u[1] * 0.85 if bottom > top else h / 2
        return hx / w, hy / h

    def _ground(self, t, x0, y0, w, h):
        """Each bird now: (feet x, y in the frame, its scale px a unit, heading, pose)."""
        out = []
        sec = (t % self.P) / self.fps
        st = math.sin(self.tilt)
        for i, (walker, u, home) in enumerate(zip(self.walkers, self.u, self.homes)):
            scale = self._scale(u, h)
            dx, dy, heading = walker.where(sec)
            fx = x0 + home[0] * w + dx * scale
            fy = y0 + home[1] * h - dy * scale * st
            if self.plan is not None:  # (stepped aside, out of the others' way: planned in a box rh high)
                rate, k = self.plan.shape[1] / self.loop_s, h / self.rh
                (ox, oz), (vx, vz) = _along_loop(self.plan, i, sec * rate)
                fx, fy = fx + ox * k, fy + oz * st * k
                go = walker.pace_at(sec)
                if go > 0:  # (walking: facing the way it goes)
                    v = walker.speed * go * scale
                    ax, az = v * math.cos(heading) + vx * rate * k, -v * math.sin(heading) + vz * rate * k
                    heading += go * math.remainder(math.atan2(-az, ax) - heading, TAU)
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
            rgb, alpha, (ax, ay) = draw_bird(pose, self.coats[i], heading, scale, self.tilt, self.light, ss,
                                             (fx - math.floor(fx), fy - math.floor(fy)))
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
