"""The walls inside a fireplace ("Tường lò" of Lửa lò sưởi), drawn by code for a fireplace whose inside the video does
not show well: a firebox seen through the opening as a camera a little above it sees it (one-point perspective: its
front, the opening, square on), its back wall narrower than the opening, its two side walls turned in toward it, its
floor, and above, when the camera is low, the dark throat. Bricks in their courses (firebrick, buff; or red brick; or a
herringbone of firebrick on the back wall), each brick its own shade, their joints of mortar a little sunk; soot over
their upper part, thickest where the smoke rises above the fire, in streaks; ash on the floor round the fire, bits of
charcoal in it; the corners darker and the back less lit by the room.

Worked out once for a place of the box (lay): what the walls are like at each pixel there (their colours, linear), how
much of the room's light reaches each, and how much of each of the fire's lights (by the way the surface faces it,
how far it is, the corners), so each frame only weighs those by the room's light (Hearth._room) and by how bright the
flames round each light are now (Hearth._walls): the back wall bright behind the flames, the corners dark.

Lengths are in box heights, the opening's bottom middle at 0 (x right, y into the fireplace, z up)."""
import math

import numpy as np

from .logs import _noise

WALLS = ("firebrick", "herring", "red")  # (the kinds of wall: firebrick in courses, firebrick in a herringbone, red)
# their bricks' colours and their mortar's (sRGB), a brick's length over its height
BRICKS = {"firebrick": ((176, 136, 96), (150, 138, 122), 2.0),
          "herring": ((178, 138, 98), (150, 138, 122), 2.0),
          "red": ((132, 62, 44), (126, 118, 108), 2.6)}
SOOT = (22, 20, 19)  # (soot: black, a little warm)
ASH = (150, 146, 140)  # (the floor's ash)
NARROW = 0.68  # (the back wall's width: a share of the opening's)
AWAY = 4.5  # (the camera's distance in front of the opening: opening heights)
MORTAR = 0.07  # (a joint's width: a share of a brick's height)
LIGHTS = 5  # (the fire's light comes from this many places across it)


def _lin(rgb):
    return (np.asarray(rgb, np.float32) / 255.0) ** 2.2


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _joint(edge, width, soft):
    """How much of a joint `width` wide there is at points `edge` from its middle (blurred over `soft`: a pixel)."""
    return 1 - _smooth((edge - width / 2) / np.maximum(soft, 1e-6) + 0.5)


def _courses(u, v, height, ratio, soft_u, soft_v):
    """Bricks in courses (each course half a brick along from the one under it) at points u (along), v (up): which
    brick each point is on (two integers), how much mortar there is there (0 brick .. 1 a joint's middle) and how far
    the point is from its brick's edge; `soft_u`, `soft_v`: how far a pixel goes along and up there (the joints' edges
    blurred as much)."""
    row = np.floor(v / height)
    along = u / (height * ratio) + 0.5 * np.mod(row, 2.0)
    col = np.floor(along)
    fu, fv = along - col, v / height - row
    width = MORTAR * height
    du, dv = np.minimum(fu, 1 - fu) * ratio * height, np.minimum(fv, 1 - fv) * height
    return col, row, np.maximum(_joint(du, width, soft_u), _joint(dv, width, soft_v)), np.minimum(du, dv) - width / 2


def _herringbone(u, v, height, soft):
    """Bricks (twice as long as they are high) in a herringbone at 45 degrees: which brick each point is on, how much
    mortar there is and how far from its brick's edge (as _courses). The tiling: a brick lying [m + 3k, m + 3k + 2] x
    [m - k, m - k + 1] and one standing [m + 3k + 2, m + 3k + 3] x [m - k - 1, m - k + 1] for every m, k (bricks
    `height` wide), turned 45."""
    c = math.sqrt(0.5)
    x, y = (u + v) * c / height, (v - u) * c / height
    b0, a0 = np.floor((x - y) / 4), np.floor((x + 3 * y) / 4)
    best = np.full(x.shape, np.inf)
    ident = np.zeros(x.shape + (2,))
    for dm in (-1, 0, 1, 2):
        for dk in (-1, 0, 1):
            m, k = a0 + dm, b0 + dk
            for lie, (x0, x1, y0, y1) in ((0, (m + 3 * k, m + 3 * k + 2, m - k, m - k + 1)),
                                         (1, (m + 3 * k + 2, m + 3 * k + 3, m - k - 1, m - k + 1))):
                inside = np.minimum(np.minimum(x - x0, x1 - x), np.minimum(y - y0, y1 - y))  # (to its nearest side)
                got = (inside >= 0) & (best == np.inf)
                best = np.where(got, inside, best)
                ident = np.where(got[..., None], np.stack([m * 2 + lie, k], -1), ident)
    best = np.where(np.isfinite(best), best, 0.0) * height
    return ident[..., 0], ident[..., 1], _joint(best, MORTAR * height, soft), best - MORTAR * height / 2


def _rnd(a, b, salt):
    """A random number (0..1) for each integer pair (a, b)."""
    r = np.sin(a * 12.9898 + b * 78.233 + salt * 37.719) * 43758.5453
    return r - np.floor(r)


class Firebox:
    """A firebox of `kind` walls behind an opening `wide` x `high` (box heights), `deep` as deep, its bricks `size`
    (0..1) big, `soot` (0..1) sooty, seen from `tilt` (radians: how far above the camera looks down at its floor); a
    fire burning on its floor where the picture shows it (`bed` wide, its middle `mid` up from the opening's bottom),
    its light coming from LIGHTS places across it `flames` above the floor."""

    def __init__(self, kind, wide, high, deep, size, soot, tilt, seed, bed=0.6, mid=0.1, flames=0.2):
        self.kind = kind if kind in BRICKS else "firebrick"
        self.W, self.H, self.D = float(wide), float(high), float(deep)
        self.Wb = self.W * NARROW
        self.slope = (self.W - self.Wb) / (2 * self.D)  # (a side wall's way in: across a unit deeper)
        self.dist = AWAY * self.H
        self.zc = (self.dist + self.D / 2) * math.tan(tilt)  # (the camera's height: looking down at the floor's middle)
        self.brick = self.H * (0.05 + 0.07 * float(size))  # (a brick's height: a real firebrick's, in an opening
        # 30 inches high, at 50)
        self.soot = float(soot)
        self.seed = int(seed)
        # (the fire on the floor where the picture shows the bed: that far in; across it, inside the side walls)
        y = float(np.clip(self.dist * mid / max(self.zc - mid, 1e-6), 0.08 * self.D, 0.8 * self.D))
        half = min(0.4 * bed * (self.dist + y) / self.dist, 0.9 * (self.W / 2 - self.slope * y))
        self.lights = np.array([(x, y, flames) for x in np.linspace(-half, half, LIGHTS)])

    def screen(self, p):
        """Points (... x 3) on the picture: right of the opening's middle and up from its bottom (box heights)."""
        p = np.asarray(p, np.float64)
        k = self.dist / (self.dist + p[..., 1])
        return p[..., 0] * k, self.zc + (p[..., 2] - self.zc) * k

    def surfaces(self, X, Z):
        """What the camera sees through the opening at picture points X (right of the opening's middle), Z (up from
        its bottom): which surface (0 the back, 1 the left wall, 2 the right one, 3 the floor, 4 the throat above),
        and where on it (x, y, z)."""
        dist, zc = self.dist, self.zc
        far = np.full(X.shape, np.inf)
        which = np.full(X.shape, 0)
        # each plane the ray from the camera through the point could meet first: the one nearest the opening
        k = dist / (dist + self.D)  # (the back wall)
        yb = np.where((np.abs(X / k) <= self.Wb / 2 + 1e-9), self.D, np.inf)
        cands = [(0, yb)]
        for side, sign in ((1, -1.0), (2, 1.0)):  # (the left wall at x < 0, the right at x > 0)
            xs = sign * X
            ys = dist * (self.W / 2 - xs) / np.maximum(xs + self.slope * dist, 1e-9)
            ys = np.where((xs + self.slope * dist > 1e-9) & (ys >= -1e-9) & (ys <= self.D + 1e-9), ys, np.inf)
            cands.append((side, ys))
        with np.errstate(divide="ignore", invalid="ignore"):
            kf = (zc - Z) / zc  # (the floor: z 0)
            yf = np.where((kf > 0) & (kf <= 1 + 1e-9), dist / kf - dist, np.inf)
            yf = np.where((np.abs(X / np.where(kf > 0, kf, 1)) <= self.W / 2 - self.slope * yf + 1e-9) &
                          (yf <= self.D + 1e-9), yf, np.inf)
            cands.append((3, yf))
            if zc < self.H:  # (the throat above: the camera is lower than the opening's top)
                kt = (zc - Z) / (zc - self.H)
                yt = np.where((kt > 0) & (kt <= 1 + 1e-9), dist / np.where(kt > 0, kt, 1) - dist, np.inf)
                cands.append((4, np.where((yt >= 0) & (yt <= self.D + 1e-9), yt, np.inf)))
        for k_, y in cands:
            nearer = y < far
            far = np.where(nearer, y, far)
            which = np.where(nearer, k_, which)
        far = np.where(np.isfinite(far), far, self.D)
        k = dist / (dist + far)
        return which, X / k, far, zc + (Z - zc) / k

    def lay(self, X, Z):
        """The walls at picture points X, Z (as surfaces: a grid, rows of Z, columns of X): their colours (linear:
        "albedo"), how much of the room's light reaches each ("room": less deeper in, on the side walls, in the
        corners), and how much of each of the fire's lights ("fire": LIGHTS x h x w; its way, its distance, the
        corners)."""
        which, x, y, z = self.surfaces(X, Z)
        salt = self.seed % 9973
        brick, joint, ratio = _lin(BRICKS[self.kind][0]), _lin(BRICKS[self.kind][1]), BRICKS[self.kind][2]
        side = (which == 1) | (which == 2)
        # where on its own wall each point is (along, up), how far a pixel goes there, and its bricks
        along = np.where((which == 0) | (which == 3), x, y * math.sqrt(1 + self.slope ** 2))
        up = np.where(which == 3, y, z)
        steps = [np.abs(np.gradient(a, axis=ax)) for a in (along, up) for ax in (0, 1)] if min(X.shape) > 1 else \
            [np.zeros(X.shape)] * 4
        soft_u, soft_v = np.hypot(steps[0], steps[1]), np.hypot(steps[2], steps[3])
        col, row, mortar, inner = _courses(along, up, self.brick * np.where(which == 3, 1.25, 1.0), ratio, soft_u,
                                           soft_v)
        if self.kind == "herring":  # (the back wall a herringbone)
            hc, hr, hm, hi = _herringbone(x, z, self.brick * 0.9, np.maximum(soft_u, soft_v))
            back = which == 0
            col, row, mortar, inner = (np.where(back, a, b) for a, b in ((hc, col), (hr, row), (hm, mortar),
                                                                         (hi, inner)))
        r1, r2, r3 = (_rnd(col + k, row - k, salt + 7 * k + which) for k in range(3))
        a, q = along / self.brick + 40, up / self.brick + 40  # (noise over each wall: bricks from 40 on)
        top = float(max(a.max(), q.max())) + 2
        grain = _noise(np.random.default_rng([self.seed, 5501]), a * 6, q * 6, 1.0, 6 * top, 3, wrap=False)
        speck = _noise(np.random.default_rng([self.seed, 5503]), a * 24, q * 24, 1.0, 24 * top, 1, wrap=False)
        # each brick its own shade (a few burnt darker), warmer or greyer, mottled; small dark spots (as fired
        # clay's); its face a little rounded off toward its edges
        tone = (0.66 + 0.6 * r1) * (0.86 + 0.28 * grain) * np.where(r3 < 0.1, 0.72, 1.0)
        tone *= 1 - 0.45 * _smooth((speck - 0.76) / 0.07)
        tone *= 1 - 0.22 * (1 - _smooth(inner / (0.12 * self.brick)))
        out = brick * tone[..., None] * (1 + 0.22 * (r2 - 0.5)[..., None] * np.array([1.0, 0.45, -0.2], np.float32))
        out = out * (1 - mortar[..., None]) + joint * (0.75 + 0.25 * grain)[..., None] * mortar[..., None]
        # soot: over the upper part, thickest over the fire (the smoke rising), in streaks down the walls, in patches
        h = np.clip(z / self.H, 0.0, 1.5)
        rise = np.exp(-(x / (0.12 * self.W + 0.3 * self.H * h)) ** 2) * _smooth((h - 0.15) / 0.5)
        g = np.random.default_rng([self.seed, 5502])
        sx, sz = x / self.H + 2, z / self.H + 2
        streak = _noise(g, sx * 18, sz * 2.5, 1.0, 18 * (float(max(sx.max(), sz.max())) + 2), 2, wrap=False)
        patch = _noise(g, sx * 6, sz * 6, 1.0, 6 * (float(max(sx.max(), sz.max())) + 2), 3, wrap=False)
        low = 0.55 - 0.35 * self.soot  # (a sootier firebox blackened further down)
        soot = self.soot * np.clip((0.7 * _smooth((h - low) / 0.55) + 0.5 * rise + 0.35 * (streak - 0.45) * h) *
                                   (0.6 + 0.8 * patch), 0.0, 1.0)
        soot = np.where(which == 3, 0.25 * self.soot, soot)
        out = out * (1 - 0.9 * soot[..., None]) + _lin(SOOT) * (0.9 * soot)[..., None]
        # the floor: ash round the fire, thickest under it, in drifts; its bricks showing further off; bits of
        # charcoal in the ash
        fire_x, fire_y = float(self.lights[-1][0]), float(self.lights[0][1])  # (the fire's half width, how far in)
        sy = y / self.H + 2
        drift = _noise(g, sx * 5, sy * 5, 1.0, 5 * (float(max(sx.max(), sy.max())) + 2), 3, wrap=False)
        bits = _noise(g, sx * 90, sy * 90, 1.0, 90 * (float(max(sx.max(), sy.max())) + 2), 1, wrap=False)
        under = np.exp(-(np.maximum(np.abs(x) - fire_x, 0.0) / (0.35 * self.W)) ** 2 -
                       ((y - fire_y) / (0.9 * self.D)) ** 2)
        ash = np.clip(0.25 + 0.9 * under + 0.6 * (drift - 0.5) + 0.2 * (grain - 0.5), 0.0, 1.0) * (which == 3)
        out = out * (1 - ash[..., None]) + _lin(ASH) * (0.8 + 0.3 * grain)[..., None] * ash[..., None]
        char = 0.85 * _smooth((bits - 0.77) / 0.07) * ash
        out = out * (1 - char[..., None]) + _lin(SOOT) * char[..., None]
        out *= (1 - 0.3 * mortar)[..., None]  # (the joints a little sunk: less light reaches them)
        out = np.where((which == 4)[..., None], _lin(SOOT) * 0.5, out)
        # the corners darker (light from the opening and the fire reaching less into them)
        edge = np.where(which == 0, np.minimum(self.Wb / 2 - np.abs(x), z),
                        np.where(side, np.minimum((self.D - y) * math.sqrt(1 + self.slope ** 2), z),
                                 np.minimum(self.W / 2 - self.slope * y - np.abs(x), self.D - y)))
        corner = 1 - 0.5 * np.exp(-np.maximum(edge, 0.0) / (0.06 * self.H))
        room = (1.0 - 0.5 * y / self.D) * np.where(side, 0.72, np.where(which == 3, 0.9, 1.0)) * corner
        room = np.where(which == 4, 0.04, room)
        # each of the fire's lights on each point: by how squarely the surface faces it, softly less far off
        n = np.zeros(X.shape + (3,))
        norm = math.sqrt(1 + self.slope ** 2)
        n[which == 0] = (0.0, -1.0, 0.0)
        n[which == 1] = (1 / norm, -self.slope / norm, 0.0)
        n[which == 2] = (-1 / norm, -self.slope / norm, 0.0)
        n[which == 3] = (0.0, 0.0, 1.0)
        n[which == 4] = (0.0, 0.0, -1.0)
        p = np.stack([x, y, z], -1)
        fire = np.zeros((len(self.lights),) + X.shape, np.float32)
        near = (0.18 * self.H) ** 2  # (the flames are big: no light comes from a point)
        for i, at in enumerate(self.lights):
            to = at - p
            d2 = np.einsum("...i,...i->...", to, to)
            facing = np.clip(np.einsum("...i,...i->...", n, to) / np.sqrt(np.maximum(d2, 1e-12)), 0.0, 1.0)
            fire[i] = facing * near / (d2 + near) * corner * (which != 4)
        return {"albedo": out.astype(np.float32), "room": room.astype(np.float32), "fire": fire}
