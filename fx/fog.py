"""Drifting fog ("Sương mù bay"; the owner asked 2026-11-03: "tôi cần hiệu ứng sương mù bay"): banks of fog that drift
with the wind across the scene, billow and thin out, as a layer of fog over the picture (drawn over it: its preview
clip is exactly what the video shows; see effects.py for how every effect loops).

The fog is a veil in the fog's colour whose coverage is 1 - exp(-depth), the way fog hides what is behind it and
lights up in the light around it (lit by the scene: the light of the video around each place, as frost and rain are;
fog near a lamp glows). Its depth is the sum of three layers, far to near: far ones small, faint and slow, near ones
big and quicker (a camera looking into fog sees the near banks pass faster). Each layer is a smooth random pattern
on a tile that wraps around (snow.drifts: wider blobs than tall, as fog lies) plus finer wisps on the same tile, seen
as two copies: each drifts with the wind through a cycle of its own (seconds, "Cuộn, đổi hình": shorter cycles,
quicker changes; whole cycles a loop, so frame P is frame 0 at any wind) and is laid in and out in turn, half a cycle
apart, as varied all through (their sum over the root of their squared weights); every cycle shows another part of the
tile. So the banks drift and change shape as fog does, and nothing repeats within the loop; with no wind they only
billow in place. How patchy it is ("Khe hở") is where the pattern is cut: low, an even veil with thicker and thinner
parts; high, separate wisps with clear air between them.

It lies in a band of the frame (its top and bottom edges, each faded over "Mép mềm"; the top edge billows with the
fog, it is not a straight line), denser toward the bottom if asked (ground fog): the looks are ground fog, a band of
fog across a valley or a lake, a veil over the whole scene, thin wisps blowing past, thick morning fog. The clip's
painted area (and the foreground, "Bay ra sau tiền cảnh") limit it further, as for every effect.

Worked out on a grid of FINE rows a frame height whatever the frame's size (the preview clip, the video and 4K show
the same fog), then scaled up smoothly: fog has no sharp detail.
"""
import math

import numpy as np
from PIL import Image

from . import effects as _fx
from .effects import Effect, _choice, _colour, _lerp, _num, _rgb
from .snow import drifts

FINE = 270  # rows of the grid the fog is worked out on (a frame height)
LAYERS = (  # far to near: share of the depth, blob size and speed (times the settings'), tile (times the blob)
    (0.3, 0.55, 0.5, 2.0), (0.35, 1.0, 1.0, 3.0), (0.35, 1.7, 1.8, 4.5))
WISPS = 0.35  # the finer wisps' share of a layer's pattern (blobs a third the size)
TILE = 3.0  # a layer's tile is at least this wide (frame heights: wider than a 16:9 frame and a cycle's drift)
SPEED = (0.0, 0.12)  # how fast the fog drifts at "Gió" 0 and at 100, frame heights a second (the middle layer)
LIGHT_GAIN = 1.15  # fog lit by the scene: the light around it, this much
ROOM = 0.08  # and a share of its own colour (light scattered from all round: at night fog is dark but near lamps)

DEFAULTS = {"amount": 55, "top": 52, "bottom": 100, "soft": 30, "lower": 50, "wind": 25, "swirl": 45, "size": 55,
            "patchy": 45, "depth": 50, "lit": 70, "opacity": 90}
LOOKS = {  # the "Kiểu sương" presets ("ground" is the defaults)
    "ground": {},
    "valley": {"amount": 60, "top": 40, "bottom": 66, "soft": 18, "lower": 20, "size": 60, "wind": 25, "swirl": 35,
               "patchy": 40},
    "veil": {"amount": 45, "top": 0, "bottom": 100, "soft": 20, "lower": 25, "size": 70, "wind": 15, "swirl": 35,
             "patchy": 25},
    "wisps": {"amount": 35, "top": 12, "bottom": 100, "soft": 20, "lower": 10, "size": 40, "wind": 45, "swirl": 55,
              "patchy": 75},
    "morning": {"amount": 85, "top": 0, "bottom": 100, "soft": 25, "lower": 40, "size": 70, "wind": 10, "swirl": 30,
                "patchy": 20},
}

SCHEMA = {
    "label": "Sương mù bay",
    "note": "Sương mù trôi theo gió, cuộn lại rồi tan, phủ lên cảnh như sương thật (lấy ánh sáng của chính video: gần "
            "đèn thì sương sáng lên). (1) Bấm Suốt video. (2) Chọn Kiểu sương. (3) Kéo Mép trên, Mép dưới cho lớp "
            "sương nằm đúng chỗ (sát mặt đất, ngang sườn núi, trên mặt hồ…). Muốn sương chỉ ở một chỗ: Vẽ vùng rồi tô "
            "chỗ đó. Muốn sương ở sau nhà, cây phía trước: tô chúng vào Tiền cảnh rồi bật Bay ra sau tiền cảnh.",
    "params": [
        _choice("look", "Kiểu sương", "ground",
                [("ground", "Sương sát mặt đất"), ("valley", "Dải sương trôi ngang (núi, mặt hồ)"),
                 ("veil", "Sương phủ cả cảnh"), ("wisps", "Làn sương mỏng bay qua"),
                 ("morning", "Sương sớm dày đặc")],
                presets={k: {**DEFAULTS, **v} for k, v in LOOKS.items()}, group="Kiểu"),
        _num("amount", "Độ dày sương", DEFAULTS["amount"], group="Sương"),
        _num("opacity", "Đậm nhất (chỗ sương dày)", DEFAULTS["opacity"], group="Sương"),
        _num("patchy", "Khe hở (liền một màn ← → từng đám rời)", DEFAULTS["patchy"], group="Sương"),
        _num("size", "Cỡ đám sương", DEFAULTS["size"], group="Sương"),
        _num("top", "Mép trên lớp sương (0: đỉnh hình)", DEFAULTS["top"], group="Chỗ có sương"),
        _num("bottom", "Mép dưới lớp sương (100: đáy hình)", DEFAULTS["bottom"], group="Chỗ có sương"),
        _num("soft", "Mép mềm", DEFAULTS["soft"], group="Chỗ có sương"),
        _num("lower", "Dày dần xuống dưới", DEFAULTS["lower"], group="Chỗ có sương"),
        _num("wind", "Gió (trái ← → phải; xa giữa: bay nhanh)", DEFAULTS["wind"], -100, 100, group="Chuyển động"),
        _num("swirl", "Cuộn, đổi hình", DEFAULTS["swirl"], group="Chuyển động"),
        _num("depth", "Xa gần (đám gần to, bay nhanh hơn)", DEFAULTS["depth"], group="Chuyển động"),
        _num("lit", "Sáng theo cảnh", DEFAULTS["lit"], group="Màu"),
        _colour("color", "Màu sương", "#dde4ec", group="Màu")]}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def grid(field, u, v):
    """`field` (a tile that wraps around, rows along u) at every pair of u (1-D, tiles) and v (1-D): (len(v), len(u)),
    bilinear (the grid is a product of its rows and columns: one pass along v, then along u)."""
    n = field.shape[0]
    v = np.asarray(v, np.float64) * n
    j = np.floor(v)
    fv = (v - j).astype(np.float32)
    j0 = j.astype(np.int64) % n
    cols = field[:, j0] * (1 - fv) + field[:, (j0 + 1) % n] * fv  # (n, len(v))
    u = np.asarray(u, np.float64) * n
    i = np.floor(u)
    fu = (u - i).astype(np.float32)[:, None]
    i0 = i.astype(np.int64) % n
    return (cols[i0] * (1 - fu) + cols[(i0 + 1) % n] * fu).T


def _cut_mean(b):
    """E[max(0, N + b)] of a standard normal N: how much is left on average where the pattern is cut at -b."""
    return b * 0.5 * (1 + math.erf(b / math.sqrt(2))) + math.exp(-b * b / 2) / math.sqrt(2 * math.pi)


class Fog(Effect):
    VERSION = 1
    LIT = True

    def __init__(self, *a, light=None):
        super().__init__(*a)
        p = self.p
        rng = np.random.default_rng([self.seed, 1103, 7])
        self.light = light
        self.lit = p["lit"] / 100 if light is not None else 0.0
        self.colour = _rgb(p["color"])
        self.gh = FINE
        self.gw = max(2, int(math.ceil(FINE * self.aspect)))
        # the grid's points, frame heights
        self.gx = (np.arange(self.gw) + 0.5) / FINE
        self.gy = (np.arange(self.gh) + 0.5) / FINE
        wind = p["wind"] / 100
        way = 1.0 if wind >= 0 else -1.0
        speed = _lerp(*SPEED, abs(wind))  # frame heights a second, the middle layer
        depth = p["depth"] / 100
        size = _lerp(0.35, 1.3, p["size"] / 100)  # a blob's width, frame heights (the middle layer)
        swirl = p["swirl"] / 100
        self.layers = []
        # each layer is two copies of its pattern, each drifting with the wind through a cycle of its own and laid
        # in and out in turn (a triangle each, half a cycle apart): any pace loops (whole cycles a loop), and as one
        # copy gives way to the other the banks change shape. Each cycle shows another part of the tile, so nothing
        # repeats within the loop but the loop itself
        cycle = _lerp(14.0, 6.0, swirl)  # seconds
        self.k = max(1, round(self.loop_s / cycle))
        cycle = self.loop_s / self.k
        for share, big, fast, tile in LAYERS:
            big, fast = big ** depth, fast ** depth  # ("Xa gần" 0: all alike)
            blob = size * big
            L = max(TILE, tile * blob)  # its tile (frame heights): wider than the frame and the drift of a cycle
            blobs = max(1.0, L / blob)
            n = int(min(256, max(32, 2 ** math.ceil(math.log2(blobs * 10)))))
            self.layers.append({"share": share, "L": L, "drift": way * speed * fast * cycle / L,  # (tiles a cycle)
                                "base": drifts(rng, n, blobs * 0.6, blobs * 1.6),  # (wider than tall)
                                "fine": drifts(rng, n, blobs * 1.8, blobs * 4.5),
                                "lift": 0.04 * swirl * blob / L, "bob": float(rng.random()),
                                "phase": rng.random(3)})
        b = _lerp(1.4, -0.6, p["patchy"] / 100)  # (where the pattern is cut)
        self.cut, self.cut_mean = b, _cut_mean(b)
        self.amount = _lerp(0.0, 2.6, (p["amount"] / 100) ** 1.2)
        self.most = p["opacity"] / 100
        self.top, self.bottom = p["top"] / 100, max(p["top"] / 100, p["bottom"] / 100)
        self.soft = max(0.01, p["soft"] / 100 * 0.5)
        self.lower = p["lower"] / 100
        self._paint = None  # (the fog's colour at each pixel: made when first drawn)

    def depth_at(self, tau):
        """The fog's optical depth on the grid at loop phase tau."""
        total = np.zeros((self.gh, self.gw), np.float32)
        y = self.gy[:, None].astype(np.float32)
        for lay in self.layers:
            L, ph = lay["L"], lay["phase"]
            bob = lay["lift"] * math.sin(2 * math.pi * (tau + lay["bob"]))
            n, weights = 0.0, 0.0
            for j in (0, 1):
                q = self.k * tau + ph[2] + 0.5 * j
                f = q - math.floor(q)  # (how far through its cycle this copy is)
                w = 1.0 - abs(2.0 * f - 1.0)
                if w <= 0:
                    continue
                cyc = math.floor(q) % self.k  # (which cycle: another part of the tile each time)
                u = self.gx / L - lay["drift"] * (f - 0.5) + ph[0] + 0.618034 * cyc + 0.31 * j
                v = self.gy / L + ph[1] + 0.414214 * cyc + 0.53 * j + bob
                n = n + w * ((1 - WISPS) * grid(lay["base"], u, v) + WISPS * grid(lay["fine"], u, v))
                weights += w * w
            n = n / math.sqrt(weights)  # (as varied all through the crossfade: a standard deviation of about 1)
            dens = np.maximum(n + self.cut, 0.0) / self.cut_mean
            # where it lies: the band's edges, the top one billowing with the fog
            yy = y - 0.35 * self.soft * np.clip(n, -2.0, 2.0)
            band = _smooth((yy - (self.top - self.soft)) / self.soft) if self.top > 0 else 1.0
            if self.bottom < 1:
                band = band * (1 - _smooth((yy - self.bottom) / self.soft))
            ramp = np.clip((y - self.top) / max(self.bottom - self.top, 0.05), 0.0, 1.0)
            heavier = (1 - self.lower) + self.lower * 2 * ramp  # (denser toward the bottom: as much in all)
            total += lay["share"] * dens * band * heavier
        return total * self.amount

    def coverage(self, tau):
        """How much the fog hides at each grid point (0..1)."""
        return (self.most * (1.0 - np.exp(-self.depth_at(tau)))).astype(np.float32)

    def _colours(self):
        """The fog's colour at each pixel (sRGB 0..1): its own, or lit by the light around it in the scene."""
        own = _fx.linear(self.colour)
        if self.lit and self.light is not None:
            around = _fx.light(self.light)  # (linear: the light around each place, at the light map's size)
            lin = own * ((1 - self.lit) + self.lit * ROOM) + self.lit * LIGHT_GAIN * around * (0.6 + 0.4 * own)
        else:
            lin = np.broadcast_to(own, (2, 2, 3))
        rgb = _fx.srgb(lin)
        return np.stack([np.asarray(Image.fromarray(np.ascontiguousarray(rgb[..., c]), "F").resize(
            (self.w, self.h), Image.Resampling.BILINEAR), np.float32) for c in range(3)], 2).clip(0.0, 1.0)

    def release(self):
        self._paint = None

    def draw(self, t, canvas, mask):
        if self.amount <= 0 or self.most <= 0:
            return
        cover = self.coverage(self.tau(t))
        a = np.asarray(Image.fromarray(cover, "F").resize((self.w, self.h), Image.Resampling.BILINEAR,
                                                          box=(0, 0, self.w / self.h * FINE, FINE)), np.float32)
        a = np.clip(a, 0.0, 1.0)
        if self._paint is None:
            self._paint = self._colours()
        canvas.over_rgba(0, 0, self._paint * a[..., None], a, mask)
