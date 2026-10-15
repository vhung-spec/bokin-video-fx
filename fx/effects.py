"""The effects: falling snow, leaves and rain, and glowing particles (fireflies, dust in the light).

Every particle moves as a function of the phase tau = t / P, where t is the frame number and P the number of
frames of the loop, and every frequency in it is a whole number of cycles per loop. So frame P is exactly frame 0
again: the effect loops with the video, with no crossfade.

Falling particles travel down a slanted line (the wind) and start again at the top after leaving the bottom, a
whole number of times k per loop. Their speed stays free: the path can be longer than the frame, the rest of it
spent out of sight above the top, so k passes of a path of length L fill exactly the loop's duration.

Lengths are in frame heights and pixel sizes are given for a 1080-line frame, so a small preview shows the same
scene as the full video. Random draws are fixed per particle (by the layer's seed), so moving a slider changes
what it should and leaves every particle where it was.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from . import overlay, sprites


def _num(key, label, value, lo=0, hi=100):
    return {"key": key, "label": label, "type": "range", "min": lo, "max": hi, "value": value}


def _choice(key, label, value, options, presets=None):
    return {"key": key, "label": label, "type": "choice", "value": value,
            "options": [{"value": v, "label": t} for v, t in options], "presets": presets or {}}


def _colour(key, label, value):
    return {"key": key, "label": label, "type": "color", "value": value}


SCHEMA = {
    "snow": {"label": "Tuyết rơi", "params": [
        _num("amount", "Mật độ", 45), _num("size", "Cỡ bông", 40), _num("speed", "Tốc độ rơi", 35),
        _num("wind", "Gió (trái ← → phải)", 10, -100, 100), _num("sway", "Đung đưa", 40),
        _num("depth", "Chiều sâu (xa nhỏ, gần to)", 60), _num("blur", "Nhoè bông ở gần", 40),
        _num("opacity", "Độ đậm", 85), _colour("color", "Màu", "#ffffff")]},
    "leaves": {"label": "Lá rơi", "params": [
        _choice("style", "Kiểu", "autumn", [("autumn", "Lá thu"), ("green", "Lá xanh"),
                                             ("petal", "Cánh hoa anh đào")]),
        _num("amount", "Số lá", 30), _num("size", "Cỡ lá", 45), _num("speed", "Tốc độ rơi", 35),
        _num("wind", "Gió (trái ← → phải)", 15, -100, 100), _num("spin", "Xoay, lật", 50),
        _num("sway", "Đung đưa", 50), _num("depth", "Chiều sâu (xa nhỏ, gần to)", 50),
        _num("blur", "Nhoè lá ở gần", 30), _num("opacity", "Độ đậm", 100)]},
    "rain": {"label": "Mưa", "params": [
        _num("amount", "Mật độ", 45), _num("speed", "Tốc độ rơi", 60), _num("length", "Độ dài vệt", 50),
        _num("size", "Độ dày hạt", 35), _num("wind", "Gió (nghiêng)", 10, -100, 100),
        _num("depth", "Chiều sâu", 50), _num("opacity", "Độ đậm", 45), _colour("color", "Màu", "#d4dce6")]},
    "glow": {"label": "Hạt sáng", "params": [
        _choice("style", "Kiểu", "firefly", [("firefly", "Đom đóm"), ("dust", "Bụi sáng lơ lửng")],
                presets={"firefly": {"color": "#f2ff8c"}, "dust": {"color": "#fff1d6"}}),
        _num("amount", "Số hạt", 35), _num("size", "Cỡ", 40), _num("brightness", "Độ sáng", 70),
        _num("twinkle", "Nhấp nháy", 60), _num("drift", "Bay lượn", 50),
        _num("depth", "Chiều sâu (xa nhỏ, gần to)", 50), _colour("color", "Màu", "#f2ff8c")]},
}


SCHEMA["file"] = overlay.SCHEMA  # a layer from an outside file (video or picture)


def defaults(kind):
    return {p["key"]: p["value"] for p in SCHEMA[kind]["params"]}


def clean_params(kind, raw):
    """The parameters of a `kind` layer from the page: every key present, numbers clamped, choices checked."""
    out = {}
    raw = raw if isinstance(raw, dict) else {}
    for p in SCHEMA[kind]["params"]:
        v = raw.get(p["key"], p["value"])
        if p["type"] == "range":
            step = p.get("step", 1)
            try:
                v = round(round(float(v) / step) * step, 3)
            except (TypeError, ValueError, OverflowError):
                v = p["value"]
            if v != v:  # NaN
                v = p["value"]
            v = max(p["min"], min(p["max"], v))
            if step == 1:
                v = int(v)
        elif p["type"] == "choice":
            v = v if v in {o["value"] for o in p["options"]} else p["value"]
        else:
            v = v.lower() if isinstance(v, str) and len(v) == 7 and v[0] == "#" and all(
                c in "0123456789abcdefABCDEF" for c in v[1:]) else p["value"]
        out[p["key"]] = v
    return out


def _rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def _lerp(a, b, t):
    return a + (b - a) * t


class Canvas:
    """One frame's particles before they are laid over the picture: premultiplied colour and coverage for
    'over' layers, or light for 'screen' layers, with the box they touched."""

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.rgb = np.zeros((h, w, 3), np.float32)
        self.a = np.zeros((h, w), np.float32)
        self.box = None

    def reset(self):
        if self.box:
            x0, y0, x1, y1 = self.box
            self.rgb[y0:y1, x0:x1] = 0
            self.a[y0:y1, x0:x1] = 0
        self.box = None

    def scale(self, gain):
        """What was drawn, `gain` (0..1) times as strong: a clip fading in or out."""
        if self.box:
            x0, y0, x1, y1 = self.box
            self.rgb[y0:y1, x0:x1] *= gain
            self.a[y0:y1, x0:x1] *= gain

    def _place(self, x0, y0, h, w):
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, self.w), min(y0 + h, self.h)
        if cx0 >= cx1 or cy0 >= cy1:
            return None
        if self.box is None:
            self.box = [cx0, cy0, cx1, cy1]
        else:
            b = self.box
            b[0], b[1], b[2], b[3] = min(b[0], cx0), min(b[1], cy0), max(b[2], cx1), max(b[3], cy1)
        return (slice(cy0, cy1), slice(cx0, cx1)), (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))

    def over(self, x0, y0, cover, colour, opacity, mask):
        """A one-colour particle (coverage `cover` 0..1) laid over what is already there."""
        at = self._place(x0, y0, *cover.shape)
        if at is None:
            return
        dst, src = at
        a = cover[src] * opacity
        if mask is not None:
            a = a * mask[dst]
        rgb, cov = self.rgb[dst], self.a[dst]
        inv = 1.0 - a
        rgb *= inv[..., None]
        rgb += a[..., None] * colour
        cov *= inv
        cov += a

    def over_rgba(self, x0, y0, prgb, alpha, mask):
        """A many-coloured particle (premultiplied colour, coverage) laid over what is already there."""
        at = self._place(x0, y0, *alpha.shape)
        if at is None:
            return
        dst, src = at
        a, c = alpha[src], prgb[src]
        if mask is not None:
            m = mask[dst]
            a, c = a * m, c * m[..., None]
        rgb, cov = self.rgb[dst], self.a[dst]
        inv = 1.0 - a
        rgb *= inv[..., None]
        rgb += c
        cov *= inv
        cov += a

    def light(self, x0, y0, glow, colour, mask):
        at = self._place(x0, y0, *glow.shape)
        if at is None:
            return
        dst, src = at
        g = glow[src] if mask is None else glow[src] * mask[dst]
        self.rgb[dst] += g[..., None] * colour

    def flush(self, picture, blend):
        """Lay what was drawn over `picture` (float32 0..1, changed in place) and start empty again."""
        if self.box is None:
            return
        x0, y0, x1, y1 = self.box
        out = picture[y0:y1, x0:x1]
        if blend == "screen":
            out[:] = 1.0 - (1.0 - out) * (1.0 - np.minimum(self.rgb[y0:y1, x0:x1], 1.0))
        else:
            out *= 1.0 - self.a[y0:y1, x0:x1, None]
            out += self.rgb[y0:y1, x0:x1]
        self.reset()


class Effect:
    blend = "over"
    CAP = 1000  # particles with fixed random draws; how many are used follows the "amount" slider

    def __init__(self, params, seed, frames, fps, w, h):
        self.p, self.seed = params, int(seed)
        self.P, self.fps, self.w, self.h = int(frames), float(fps), int(w), int(h)
        self.aspect = self.w / self.h
        self.loop_s = self.P / self.fps
        self.px = self.h / 1080.0  # sizes below are given in pixels of a 1080-line frame
        self.rng = np.random.default_rng([self.seed, 1103])

    def cycles(self, rate_hz):
        """The whole number of cycles per loop nearest to `rate_hz` cycles a second (at least 1)."""
        return np.maximum(1.0, np.rint(np.asarray(rate_hz) * self.loop_s))

    def tau(self, t):
        return float(t) / self.P

    def draw(self, t, canvas, mask):
        raise NotImplementedError


class Falling(Effect):
    """Particles falling along slanted straight paths, k passes per loop, plus side sway and gusts."""

    def paths(self, ux, us, speed, angle, margin_y, margin_x):
        """Sets the paths of the candidate particles; returns each one's share of the loop spent on screen."""
        cos, sin = np.cos(angle), np.sin(angle)
        lmin = (1 + 2 * margin_y) / cos
        travel = speed * self.loop_s  # frame heights per loop at the wished speed
        ratio = travel / lmin
        k = np.where(ratio < 1, 1.0, np.floor(ratio))
        length = np.where(ratio < 1, lmin, travel / k)
        drift = (1 + 2 * margin_y) * sin / cos
        lo = np.minimum(0.0, -drift) - margin_x
        hi = np.maximum(self.aspect, self.aspect - drift) + margin_x
        xe = lo + ux * (hi - lo)
        flat = np.abs(sin) < 1e-6
        safe = np.where(flat, 1.0, sin)
        da, db = (-margin_x - xe) / safe, (self.aspect + margin_x - xe) / safe
        seen = np.clip(np.maximum(da, db), 0, lmin) - np.clip(np.minimum(da, db), 0, lmin)
        seen = np.where(flat, np.where((xe >= -margin_x) & (xe <= self.aspect + margin_x), lmin, 0.0), seen)
        self.cos, self.sin, self.k, self.len, self.my = cos, sin, k, length, margin_y
        self.extra, self.s0, self.xe = length - lmin, us * length, xe
        self.speed = k * length / self.loop_s  # the actual speed, frame heights a second
        return np.maximum(seen, 0) / length

    def choose(self, share, wanted, arrays):
        """How many of the candidates make about `wanted` particles on screen at a time; keeps those,
        sorted far to near (the drawing order)."""
        n = int(np.searchsorted(np.cumsum(share), wanted)) + 1 if wanted > 0 else 0
        n = min(n, len(share))
        keep = np.argsort(arrays["z"][:n], kind="stable")
        for name in ("cos", "sin", "k", "len", "extra", "s0", "xe", "speed"):
            setattr(self, name, getattr(self, name)[:n][keep])
        return {name: a[:n][keep] for name, a in arrays.items()}

    def gusts(self, strength):
        j = max(1, round(self.loop_s / 6))
        phases = self.rng.random(3) * 2 * math.pi
        self.gust = (strength, [(j * (i + 1), (1.0, 0.5, 0.25)[i] / 1.75, phases[i]) for i in range(3)])

    def gust_at(self, tau):
        strength, parts = self.gust
        return strength * sum(c * math.sin(2 * math.pi * m * tau + ph) for m, c, ph in parts)

    def where(self, t):
        """x, y (frame heights) of every particle at frame t, and whether it is on its visible stretch."""
        tau = self.tau(t)
        s = np.mod(self.s0 + self.k * self.len * tau, self.len)
        d = s - self.extra
        return self.xe + d * self.sin, -self.my + d * self.cos, d >= 0


class Snow(Falling):
    CAP = 4000

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 10))
        z = u[:, 0] ** 1.6
        ds = p["depth"] / 100
        size_k = (1 - ds) + ds * (0.3 + 2.2 * z ** 1.5)
        speed_k = (1 - ds) + ds * (0.45 + 1.3 * z)
        radius = _lerp(0.9, 4.2, p["size"] / 100) * size_k * (0.75 + 0.5 * u[:, 4])
        defocus = p["blur"] / 100 * np.clip((z - 0.55) / 0.45, 0, 1) ** 1.5
        soft = np.maximum(0.75, 0.3 * radius) + defocus * radius * 2.5
        radius = radius * (1 + 0.8 * defocus)
        opacity = p["opacity"] / 100 * _lerp(0.5, 0.95, z) * (0.8 + 0.2 * u[:, 8]) * (1 - 0.5 * defocus)
        self.angle = math.radians(45) * p["wind"] / 100
        angle = self.angle + np.radians(4) * (u[:, 9] - 0.5)
        sway = p["sway"] / 100 * _lerp(0.004, 0.025, u[:, 5]) * (0.5 + 0.7 * z)
        speed = _lerp(0.12, 0.85, p["speed"] / 100) * speed_k * (0.85 + 0.3 * u[:, 3])
        reach_px = float((radius + soft).max()) * px + 2
        margin_y = reach_px / self.h + 0.25 * float(speed.max()) / self.fps
        share = self.paths(u[:, 1], u[:, 2], speed, angle, margin_y, margin_y + float(sway.max()) + 0.03)
        wanted = p["amount"] / 100 * 900 * self.aspect / (16 / 9)
        kept = self.choose(share, wanted, {"z": z, "radius": radius, "soft": soft, "opacity": opacity,
                                           "sway": sway, "m": self.cycles(_lerp(0.15, 0.5, u[:, 6])),
                                           "phi": u[:, 7] * 2 * math.pi})
        self.__dict__.update(kept)
        self.gusts(0.02 * abs(p["wind"]) / 100 + 0.012 * p["sway"] / 100)
        streak = 0.5 * self.speed * self.h / self.fps  # a 180-degree shutter: half a frame of motion
        self.keys = [(sprites.bucket(r * px), sprites.bucket(s * px), sprites.bucket(st, floor=0.5))
                     for r, s, st in zip(self.radius, self.soft, streak)]
        self.colour = _rgb(p["color"])
        self.cache = sprites.SpriteCache(lambda key, fx, fy: sprites.dot(key[0], key[1], key[2], self.angle, fx, fy))

    def draw(self, t, canvas, mask):
        x, y, on = self.where(t)
        tau = self.tau(t)
        x = x + self.sway * np.sin(2 * math.pi * self.m * tau + self.phi) + self.gust_at(tau) * (0.3 + 0.7 * self.z)
        xs, ys = x * self.h, y * self.h
        for i in np.flatnonzero(on & (xs > -60) & (xs < self.w + 60)):
            ix, iy = math.floor(xs[i]), math.floor(ys[i])
            cover, x0, y0 = self.cache.get(self.keys[i], xs[i] - ix, ys[i] - iy)
            canvas.over(ix + x0, iy + y0, cover, self.colour, self.opacity[i], mask)


_LEAVES = {}


def leaf_set(style, seed):
    key = (style, seed)
    if key not in _LEAVES:
        if len(_LEAVES) > 8:
            _LEAVES.clear()
        _LEAVES[key] = [sprites.Leaf(style, i, seed) for i in range(6)]
    return _LEAVES[key]


class Leaves(Falling):
    CAP = 400

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 16))
        z = u[:, 0] ** 1.2
        ds = p["depth"] / 100
        size_k = (1 - ds) + ds * (0.45 + 1.6 * z ** 1.3)
        speed_k = (1 - ds) + ds * (0.55 + 0.9 * z)
        length = _lerp(22, 80, p["size"] / 100) * size_k * (0.8 + 0.4 * u[:, 4]) * px  # pixels
        speed = _lerp(0.1, 0.42, p["speed"] / 100) * speed_k * (0.85 + 0.3 * u[:, 3])
        angle = math.radians(50) * p["wind"] / 100 + np.radians(8) * (u[:, 9] - 0.5)
        swing = p["sway"] / 100 * _lerp(0.015, 0.06, u[:, 5]) * (0.6 + 0.6 * z)
        spin = p["spin"] / 100
        turns = np.where(u[:, 10] < 0.6 * spin, np.sign(u[:, 11] - 0.5) * self.cycles(_lerp(0.1, 0.4, u[:, 12])), 0)
        margin = float(length.max()) / self.h + 0.02
        share = self.paths(u[:, 1], u[:, 2], speed, angle, margin, margin + float(swing.max()) + 0.03)
        wanted = p["amount"] / 100 * 60 * self.aspect / (16 / 9)
        defocus = p["blur"] / 100 * np.clip((z - 0.7) / 0.3, 0, 1)
        kept = self.choose(share, wanted, {
            "z": z, "length": length, "swing": swing, "m": self.cycles(_lerp(0.2, 0.5, u[:, 6])),
            "phi": u[:, 7] * 2 * math.pi, "tilt": (0.25 + 0.6 * spin) * _lerp(0.5, 1.0, u[:, 8]),
            "turns": turns, "rot0": u[:, 13] * 2 * math.pi,
            "flips": np.rint(self.loop_s * _lerp(0.0, 0.35, u[:, 14]) * spin), "chi": (u[:, 15] - 0.5) * 2.0,
            "kind": (u[:, 12] * 6).astype(int) % 6, "blur": defocus * 5 * px,
            "opacity": p["opacity"] / 100 * _lerp(0.85, 1.0, z)})
        self.__dict__.update(kept)
        self.gusts(0.025 * abs(p["wind"]) / 100 + 0.01 * p["sway"] / 100)
        self.leaves = leaf_set(p["style"], self.seed)

    def pose(self, t):
        x, y, on = self.where(t)
        tau = self.tau(t)
        phase = 2 * math.pi * self.m * tau + self.phi
        x = x + self.swing * np.sin(phase) + self.gust_at(tau) * (0.3 + 0.7 * self.z)
        y = y + 0.25 * self.swing * np.sin(2 * phase)
        turn = self.rot0 + 2 * math.pi * self.turns * tau + self.tilt * np.cos(phase)
        flip = np.cos(2 * math.pi * self.flips * tau + self.chi)
        return x * self.h, y * self.h, turn, flip, on

    def draw(self, t, canvas, mask):
        now = self.pose(t)
        before, after = self.pose(t - 0.25), self.pose(t + 0.25)  # a 180-degree shutter
        xs, on = now[0], now[4]
        moved = np.hypot(after[0] - before[0], after[1] - before[1]) + np.abs(after[2] - before[2]) * self.length
        for i in np.flatnonzero(on & (xs > -200) & (xs < self.w + 200)):
            poses = (before, now, after) if moved[i] > 1.5 else (now,)
            patch = self._patch(i, poses)
            if patch is not None:
                canvas.over_rgba(*patch, mask)

    def _patch(self, i, poses):
        parts = [self._sample(i, pose[0][i], pose[1][i], pose[2][i], pose[3][i]) for pose in poses]
        x0, y0 = min(p[0] for p in parts), min(p[1] for p in parts)
        x1 = max(p[0] + p[2].shape[1] for p in parts)
        y1 = max(p[1] + p[2].shape[0] for p in parts)
        acc = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
        for px0, py0, arr in parts:
            acc[py0 - y0:py0 - y0 + arr.shape[0], px0 - x0:px0 - x0 + arr.shape[1]] += arr
        acc *= self.opacity[i] / len(parts)
        return x0, y0, acc[..., :3], acc[..., 3]

    def _sample(self, i, x, y, turn, flip):
        box = self.length[i] / 0.9  # the leaf fills 90 % of its image
        img = self.leaves[self.kind[i]].image(flip < 0, box)
        size = img.size[0]
        sx, sy = max(abs(flip), 0.06) * box / size, box / size
        ca, sa = math.cos(turn), math.sin(turn)
        blur = float(self.blur[i])
        hw = (abs(ca) * sx + abs(sa) * sy) * size / 2 + 2 + 3 * blur
        hh = (abs(sa) * sx + abs(ca) * sy) * size / 2 + 2 + 3 * blur
        x0, y0 = math.floor(x - hw), math.floor(y - hh)
        w, h = math.ceil(x + hw) - x0, math.ceil(y + hh) - y0
        ox, oy = x0 - x, y0 - y
        data = (ca / sx, sa / sx, (ca * ox + sa * oy) / sx + size / 2,
                -sa / sy, ca / sy, (-sa * ox + ca * oy) / sy + size / 2)
        out = img.transform((w, h), Image.Transform.AFFINE, data, resample=Image.Resampling.BICUBIC)
        if blur > 0.3:
            out = out.filter(ImageFilter.GaussianBlur(blur))
        arr = np.asarray(out, np.float32) * (1 / 255.0)
        arr[..., :3] *= 0.78 + 0.22 * abs(flip)  # a leaf seen edge-on catches less light
        return x0, y0, arr


class Rain(Falling):
    CAP = 6000

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        u = self.rng.random((self.CAP, 8))
        z = u[:, 0]
        ds = p["depth"] / 100
        speed = _lerp(1.4, 4.2, p["speed"] / 100) * ((1 - ds) + ds * (0.55 + 0.9 * z)) * (0.9 + 0.2 * u[:, 3])
        self.angle = math.radians(30) * p["wind"] / 100
        angle = self.angle + np.radians(3) * (u[:, 7] - 0.5)
        width = _lerp(0.6, 1.8, p["size"] / 100) * ((1 - ds) + ds * (0.5 + z)) * px
        shutter = _lerp(0.25, 1.1, p["length"] / 100)
        longest = shutter * float(speed.max()) * 1.3 / self.fps
        share = self.paths(u[:, 1], u[:, 2], speed, angle, longest + 0.02, longest + 0.04)
        wanted = p["amount"] / 100 * 1400 * self.aspect / (16 / 9)
        opacity = np.minimum(1.0, p["opacity"] / 100 * _lerp(0.25, 0.85, z) * (0.75 + 0.5 * u[:, 5]))
        kept = self.choose(share, wanted, {"z": z, "width": width, "opacity": opacity})
        self.__dict__.update(kept)
        self.gusts(0.006 * abs(p["wind"]) / 100)
        length = shutter * self.speed * self.h / self.fps
        self.keys = [(sprites.bucket(ln, 1.1, 1.0), sprites.bucket(wd, 1.1, 0.3)) for ln, wd in zip(length, self.width)]
        self.colour = _rgb(p["color"])
        self.cache = sprites.SpriteCache(lambda key, fx, fy: sprites.streak(key[0], key[1], self.angle, fx, fy))

    def draw(self, t, canvas, mask):
        x, y, on = self.where(t)
        x = x + self.gust_at(self.tau(t)) * (0.3 + 0.7 * self.z)
        xs, ys = x * self.h, y * self.h
        for i in np.flatnonzero(on & (xs > -100) & (xs < self.w + 100)):
            ix, iy = math.floor(xs[i]), math.floor(ys[i])
            cover, x0, y0 = self.cache.get(self.keys[i], xs[i] - ix, ys[i] - iy)
            canvas.over(ix + x0, iy + y0, cover, self.colour, self.opacity[i], mask)


class Glow(Effect):
    """Points of light that wander and twinkle in place: fireflies, or dust catching the light."""
    blend = "screen"
    CAP = 600

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        firefly = p["style"] == "firefly"
        u = self.rng.random((self.CAP, 16))
        n = int(round(p["amount"] / 100 * (110 if firefly else 450) * self.aspect / (16 / 9)))
        u = u[:n]
        z = u[:, 0]
        ds = p["depth"] / 100
        size_k = (1 - ds) + ds * (0.4 + 1.6 * z ** 1.5)
        base = _lerp(1.6, 5.0, p["size"] / 100) if firefly else _lerp(0.5, 2.2, p["size"] / 100)
        self.radius = base * size_k * (0.8 + 0.4 * u[:, 1]) * px
        self.bright = p["brightness"] / 100 * _lerp(0.6, 1.0, z) * (1.8 if firefly else 0.9) / np.sqrt(size_k)
        self.cx = -0.05 + u[:, 2] * (self.aspect + 0.1)
        self.cy = -0.05 + u[:, 3] * 1.1
        reach = p["drift"] / 100 * (_lerp(0.01, 0.06, u[:, 4]) if firefly else _lerp(0.005, 0.03, u[:, 4]))
        self.ax = np.stack([reach * 0.7, reach * 0.3])
        self.ay = np.stack([reach * 0.5, reach * 0.3])
        self.mx = np.stack([self.cycles(_lerp(0.05, 0.15, u[:, 5])), self.cycles(_lerp(0.15, 0.4, u[:, 6]))])
        self.my = np.stack([self.cycles(_lerp(0.05, 0.15, u[:, 7])), self.cycles(_lerp(0.15, 0.4, u[:, 8]))])
        self.px_ = u[:, 9:11].T * 2 * math.pi
        self.py_ = u[:, 11:13].T * 2 * math.pi
        self.q = self.cycles(_lerp(0.12, 0.35, u[:, 13]) if firefly else _lerp(0.15, 0.6, u[:, 13]))
        self.qphi = u[:, 14] * 2 * math.pi
        self.firefly, self.twinkle = firefly, p["twinkle"] / 100
        self.sharp = _lerp(0.6, 3.0, self.twinkle)
        self.colour = _rgb(p["color"])
        self.keys = [sprites.bucket(r) for r in self.radius]
        self.cache = sprites.SpriteCache(lambda key, fx, fy: sprites.glow(key, fx, fy))

    def draw(self, t, canvas, mask):
        tau = self.tau(t)
        x = self.cx + (self.ax * np.sin(2 * math.pi * self.mx * tau + self.px_)).sum(axis=0)
        y = self.cy + (self.ay * np.sin(2 * math.pi * self.my * tau + self.py_)).sum(axis=0)
        wave = np.sin(2 * math.pi * self.q * tau + self.qphi)
        if self.firefly:
            pulse = np.maximum(wave, 0.0) ** self.sharp
            level = (1 - self.twinkle) * 0.8 + self.twinkle * pulse
        else:
            level = (1 - 0.8 * self.twinkle) + 0.8 * self.twinkle * ((0.5 + 0.5 * wave) ** 2)
        level = level * self.bright
        xs, ys = x * self.h, y * self.h
        for i in np.flatnonzero(level > 0.004):  # halos reach far: Canvas clips whatever falls outside
            ix, iy = math.floor(xs[i]), math.floor(ys[i])
            spot, x0, y0 = self.cache.get(self.keys[i], xs[i] - ix, ys[i] - iy)
            canvas.light(ix + x0, iy + y0, spot, self.colour * level[i], mask)


KINDS = {"snow": Snow, "leaves": Leaves, "rain": Rain, "glow": Glow}


def make(kind, params, seed, frames, fps, w, h):
    """The effect of one layer, for a loop of `frames` frames at `fps`, drawn at w x h."""
    return KINDS[kind](clean_params(kind, params), seed, frames, fps, w, h)
