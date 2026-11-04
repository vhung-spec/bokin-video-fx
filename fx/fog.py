"""Drifting fog ("Sương mù bay"; the owner asked 2026-11-03: "tôi cần hiệu ứng sương mù bay", then "đẹp hơn nhìn thật
hơn"): banks of fog that drift with the wind across the scene, curl, billow and thin out, as a layer of fog over the
picture (drawn over it: its preview clip is exactly what the video shows; see effects.py for how every effect loops).

The fog is a veil whose coverage is 1 - exp(-depth), the way fog hides what is behind it, in the light around it
(lit by the scene: the light of the video around each place, as frost and rain are; fog near a lamp glows). Its depth
is the sum of three layers, far to near: far ones smaller, flatter and slower, near ones big and quicker (a camera
looking into fog sees the near banks pass faster). Each layer's pattern is fractal, as fog is: soft banks with ever
finer wisps on them (fractal(): a random tile that wraps around, each octave half the size and "Chi tiết" times as
strong, wider than tall: fog lies in layers), its coordinates bent by a smooth flow of their own (domain warping:
tendrils, curls; "Cuộn, đổi hình" sets how much). The flow drifts with the fog and turns only a little in a loop:
moved faster, it would stir the finest wisps from one frame to the next (they would boil). Lit from above, a bank is a
little darker under fog above it (its underside, its inside: SHADE), so it has a body.

It drifts and still loops: each layer is seen as two copies, each drifting with the wind through a cycle of its own
(seconds, whole cycles a loop, so frame P is frame 0 at any wind) and laid in and out in turn, half a cycle apart
(sin² windows), as varied all through (their sum over the root of their squared weights); every cycle shows another
part of the tile. So the banks drift and change shape, and nothing repeats within the loop; with no wind they only
curl in place. How patchy it is ("Khe hở") is where the pattern is cut, softly (to a power above 1: wisps thin out
to nothing, no hard rim): low, an even veil with thicker and thinner parts; high, separate wisps with clear air
between. "Đều khắp khung" (the owner asked 2026-11-04: "nhạt hơn, đều hơn, full khắp màn hình") takes the swings of
the pattern away (to a tenth at 100: as much fog in all, it still drifts and curls), and the shade with them.

It lies in a band of the frame (its top and bottom edges, each faded over "Mép mềm"; the top edge billows with the
fog, it is not a straight line), denser toward the bottom if asked (ground fog): the looks are ground fog, a band of
fog across a valley or a lake, a veil over the whole scene, thin wisps blowing past, thick morning fog, a light even
haze over the whole frame. The clip's painted area (and the foreground, "Bay ra sau tiền cảnh") limit it further, as
for every effect.

Worked out on a grid of FINE rows a frame height whatever the frame's size (the preview clip, the video and 4K show
the same fog), then scaled up smoothly.
"""
import math

import numpy as np
from PIL import Image

from . import effects as _fx
from .effects import Effect, _choice, _colour, _lerp, _num, _rgb
from .snow import sample

FINE = 216  # rows of the grid the fog is worked out on (a frame height: 5 px of a 1080-line frame)
LAYERS = (  # far to near: share of the depth, blob size, speed (times the settings'), how much wider than tall
    (0.36, 0.6, 0.5, 2.6), (0.34, 1.0, 1.0, 2.0), (0.30, 1.6, 1.8, 1.5))
TILE = 3.2  # a layer's tile is at least this wide (frame heights: wider than a 16:9 frame and a cycle's drift)
OCTAVES = 4  # the finest wisps: an eighth of a bank (or as fine as the grid shows)
ROUGH = (0.34, 0.56)  # each octave's strength, times the one before: "Chi tiết" 0 .. 100
WARP = (0.04, 0.28)  # how far the flow bends the pattern ("Cuộn, đổi hình" 0 .. 100), blob sizes (more: smoke)
TURN = 0.01  # how far the flow moves round once a loop (tiles: a little; faster, the finest wisps would boil)
CYCLE = (18.0, 8.0)  # seconds a copy drifts before it gives way ("Cuộn, đổi hình" 0 .. 100)
SOFT_CUT = 1.6  # the pattern is cut as max(0, n + cut) to this power ("Khe hở": a wisp thins out to nothing, no
# hard rim, and clear air between the banks)
SHADE = 0.12  # how much darker a bank is under fog above it (lit from above; more, and in daylight the banks in
# front of the sky read as grey smoke, darker than the sky behind them: 0.32 in version 2)
SHADE_REACH = 0.06  # how far above (frame heights) the fog that shades a place is
EVEN = 0.9  # "Đều khắp khung" 100 takes this much of the pattern's swings away (the rest: it still drifts, curls)
SPEED = (0.0, 0.12)  # how fast the fog drifts at "Gió" 0 and at 100, frame heights a second (the middle layer)
LIGHT_GAIN = 1.15  # fog lit by the scene: the light around it, this much
ROOM = 0.08  # and a share of its own colour (light scattered from all round: at night fog is dark but near lamps)

DEFAULTS = {"amount": 55, "top": 52, "bottom": 100, "soft": 30, "lower": 50, "wind": 25, "swirl": 50, "size": 55,
            "patchy": 45, "even": 0, "detail": 60, "depth": 50, "lit": 70, "opacity": 90}
LOOKS = {  # the "Kiểu sương" presets ("ground" is the defaults)
    "ground": {},
    "valley": {"amount": 60, "top": 40, "bottom": 66, "soft": 18, "lower": 20, "size": 60, "wind": 25, "swirl": 40,
               "patchy": 40, "detail": 55},
    "veil": {"amount": 45, "top": 0, "bottom": 100, "soft": 20, "lower": 25, "size": 70, "wind": 15, "swirl": 40,
             "patchy": 25, "detail": 45},
    "wisps": {"amount": 30, "top": 12, "bottom": 100, "soft": 20, "lower": 10, "size": 40, "wind": 45, "swirl": 40,
              "patchy": 60, "detail": 50, "opacity": 75},
    "morning": {"amount": 85, "top": 0, "bottom": 100, "soft": 25, "lower": 40, "size": 70, "wind": 10, "swirl": 35,
                "patchy": 20, "detail": 40},
    "haze": {"amount": 40, "opacity": 60, "top": 0, "bottom": 100, "soft": 20, "lower": 0, "size": 75, "wind": 25,
             "swirl": 30, "patchy": 0, "even": 75, "detail": 35},
}

SCHEMA = {
    "label": "Sương mù bay",
    "note": "Sương mù trôi theo gió, cuộn lại rồi tan, phủ lên cảnh như sương thật (lấy ánh sáng của chính video: gần "
            "đèn thì sương sáng lên). (1) Bấm Suốt video. (2) Chọn Kiểu sương. (3) Kéo Mép trên, Mép dưới cho lớp "
            "sương nằm đúng chỗ (sát mặt đất, ngang sườn núi, trên mặt hồ…). Muốn sương nhạt, đều, phủ khắp khung: "
            "chọn kiểu Sương nhạt phủ đều khắp khung (nhạt hơn nữa: kéo Đậm nhất xuống). Muốn sương chỉ ở một chỗ: Vẽ "
            "vùng rồi tô chỗ đó. Muốn sương ở sau nhà, cây phía trước: tô chúng vào Tiền cảnh rồi bật Bay ra sau "
            "tiền cảnh.",
    "params": [
        _choice("look", "Kiểu sương", "ground",
                [("ground", "Sương sát mặt đất"), ("valley", "Dải sương trôi ngang (núi, mặt hồ)"),
                 ("veil", "Sương phủ cả cảnh"), ("wisps", "Làn sương mỏng bay qua"),
                 ("morning", "Sương sớm dày đặc"), ("haze", "Sương nhạt phủ đều khắp khung")],
                presets={k: {**DEFAULTS, **v} for k, v in LOOKS.items()}, group="Kiểu"),
        _num("amount", "Độ dày sương", DEFAULTS["amount"], group="Sương"),
        _num("opacity", "Đậm nhất (chỗ sương dày)", DEFAULTS["opacity"], group="Sương"),
        _num("patchy", "Khe hở (liền một màn ← → từng đám rời)", DEFAULTS["patchy"], group="Sương"),
        _num("even", "Đều khắp khung (từng đám ← → đều như một lớp)", DEFAULTS["even"], group="Sương"),
        _num("size", "Cỡ đám sương", DEFAULTS["size"], group="Sương"),
        _num("detail", "Chi tiết làn sương (mịn ← → nhiều làn nhỏ)", DEFAULTS["detail"], group="Sương"),
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


def fractal(rng, n, cu, cv, rough, top, pair=False):
    """A random pattern on an n x n tile that wraps around (rows along u), standard deviation 1: banks about 1/cu of
    the tile along u and 1/cv along v, and finer detail below them, each octave half the size and `rough` times as
    strong, down to `top` times smaller than a bank. pair: two such patterns, unlike each other (the real and the
    imaginary part of one spectrum)."""
    fu = np.fft.fftfreq(n) * n
    fv = np.fft.fftfreq(n) * n if pair else np.fft.rfftfreq(n) * n
    r = np.hypot(fu[:, None] / cu, fv[None, :] / cv)  # (in banks: 1 is a bank's size)
    r4 = r ** 4
    # each octave holds 4 times the waves of the one before: a wave's amplitude r^(log2(rough) - 1) makes an octave
    # `rough` times the one before; nothing much bigger than a bank, nothing finer than `top`
    amp = np.where(r > 0, np.maximum(r, 1e-9) ** (math.log2(rough) - 1.0), 0.0) * r4 / (1.0 + r4)
    amp *= np.exp(-(r / top) ** 2)
    spec = amp * np.exp(2j * math.pi * rng.random(amp.shape))
    if pair:
        field = np.fft.ifft2(spec)
        out = []
        for part in (field.real, field.imag):
            out.append((part / max(float(part.std()), 1e-12)).astype(np.float32))
        return out
    field = np.fft.irfft2(spec, s=(n, n))
    return (field / max(float(field.std()), 1e-12)).astype(np.float32)


def _soft_mean(b, k):
    """E[max(0, N + b) ** k] of a standard normal N: how much is left on average where the pattern is cut, softly, at
    -b."""
    z = np.linspace(-8.0, 8.0, 4001)
    p = np.exp(-z * z / 2) / math.sqrt(2 * math.pi)
    return float((np.maximum(z + b, 0.0) ** k * p).sum() * (z[1] - z[0]))


class Fog(Effect):
    VERSION = 3  # (2: fractal wisps bent by a flow, a body lit from above; 3: lighter shade, "Đều khắp khung")
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
        self.gx = ((np.arange(self.gw) + 0.5) / FINE)[None, :]
        self.gy = ((np.arange(self.gh) + 0.5) / FINE)[:, None]
        wind = p["wind"] / 100
        way = 1.0 if wind >= 0 else -1.0
        speed = _lerp(*SPEED, abs(wind))  # frame heights a second, the middle layer
        depth = p["depth"] / 100
        size = _lerp(0.35, 1.3, p["size"] / 100)  # a bank's width, frame heights (the middle layer)
        swirl = p["swirl"] / 100
        rough = _lerp(*ROUGH, p.get("detail", DEFAULTS["detail"]) / 100)
        cycle = _lerp(*CYCLE, swirl)  # seconds
        self.k = max(1, round(self.loop_s / cycle))
        cycle = self.loop_s / self.k
        self.layers = []
        for share, big, fast, flat in LAYERS:
            big, fast = big ** depth, fast ** depth  # ("Xa gần" 0: all alike)
            blob = size * big
            L = max(TILE, 3.0 * blob)  # its tile (frame heights)
            cu, cv = L / blob, L / blob * flat  # (banks a tile, along and across: wider than tall)
            finest = min(2.0 ** (OCTAVES - 1), L * FINE / 4 / cv)  # (nothing finer than the grid shows)
            n = int(min(1024, max(64, 2 ** math.ceil(math.log2(2.6 * finest * cv)))))
            self.layers.append({
                "share": share, "L": L, "drift": way * speed * fast * cycle / L,  # (tiles a cycle)
                "field": fractal(rng, n, cu, cv, rough, finest * 1.2),
                "flow": fractal(rng, 128, cu * 0.6, cv * 0.6, 0.45, 1.2, pair=True),  # (smooth: banks' size)
                "bend": _lerp(*WARP, swirl) * blob / L,  # (tiles)
                "lift": 0.04 * swirl * blob / L, "phase": rng.random(4)})
        b = _lerp(1.4, -0.6, p["patchy"] / 100)  # (where the pattern is cut)
        self.cut, self.cut_mean = b, _soft_mean(b, SOFT_CUT)
        self.even = EVEN * p.get("even", 0) / 100  # (how much of the pattern's swings is taken away)
        self.shade = SHADE * (1 - p.get("even", 0) / 100)  # (an even layer has no banks to shade one another)
        self.amount = _lerp(0.0, 2.6, (p["amount"] / 100) ** 1.2)
        self.most = p["opacity"] / 100
        self.top, self.bottom = p["top"] / 100, max(p["top"] / 100, p["bottom"] / 100)
        self.soft = max(0.01, p["soft"] / 100 * 0.5)
        self.lower = p["lower"] / 100
        self._paint = None  # (the fog's colour at each pixel: made when first drawn)

    def _pattern(self, lay, tau):
        """A layer's pattern on the grid at loop phase tau (about a standard deviation of 1)."""
        L, ph = lay["L"], lay["phase"]
        bob = lay["lift"] * math.sin(2 * math.pi * (tau + ph[3]))
        turn = 2 * math.pi * tau + 2 * math.pi * ph[2]
        n, weights = 0.0, 0.0
        for j in (0, 1):
            q = self.k * tau + ph[2] + 0.5 * j
            f = q - math.floor(q)  # (how far through its cycle this copy is)
            w = math.sin(math.pi * f) ** 2
            if w <= 1e-6:
                continue
            cyc = math.floor(q) % self.k  # (which cycle: another part of the tile each time)
            u = self.gx / L - lay["drift"] * (f - 0.5) + ph[0] + 0.618034 * cyc + 0.31 * j
            v = self.gy / L + ph[1] + 0.414214 * cyc + 0.53 * j + bob
            u, v = np.broadcast_arrays(u, v)
            # the flow that bends it (drifting with it; turning slowly, once a loop)
            fu, fv = u + TURN * math.cos(turn), v + TURN * math.sin(turn)
            du, dv = sample(lay["flow"][0], fu, fv), sample(lay["flow"][1], fu, fv)
            n = n + w * sample(lay["field"], u + lay["bend"] * du, v + lay["bend"] * dv)
            weights += w * w
        return n / math.sqrt(weights)

    def depth_at(self, tau):
        """The fog's optical depth on the grid at loop phase tau."""
        total = np.zeros((self.gh, self.gw), np.float32)
        y = self.gy.astype(np.float32)
        ramp = np.clip((y - self.top) / max(self.bottom - self.top, 0.05), 0.0, 1.0)
        heavier = (1 - self.lower) + self.lower * 2 * ramp  # (denser toward the bottom: as much in all)
        for lay in self.layers:
            n = self._pattern(lay, tau).astype(np.float32)
            dens = np.maximum(n + self.cut, 0.0) ** SOFT_CUT / self.cut_mean
            if self.even:
                dens = 1.0 + (dens - 1.0) * (1.0 - self.even)  # (as much fog in all, its swings smaller)
            # where it lies: the band's edges, the top one billowing with the fog
            yy = y - 0.35 * self.soft * np.clip(n, -2.0, 2.0)
            band = _smooth((yy - (self.top - self.soft)) / self.soft) if self.top > 0 else 1.0
            if self.bottom < 1:
                band = band * (1 - _smooth((yy - self.bottom) / self.soft))
            total += lay["share"] * dens * band * heavier
        return total * self.amount

    def coverage(self, tau):
        """How much the fog hides at each grid point (0..1)."""
        return (self.most * (1.0 - np.exp(-self.depth_at(tau)))).astype(np.float32)

    def _shade(self, depth):
        """How lit each grid point is (1 .. 1 - SHADE): less under fog above it (the light comes from above). Above
        the frame the fog goes on as in its top row (fog over the whole frame is not lighter along the top edge)."""
        if not self.shade:
            return np.ones(depth.shape, np.float32)
        reach = max(1, round(SHADE_REACH * FINE))
        c = np.cumsum(np.vstack([np.zeros((1, depth.shape[1]), np.float32), np.repeat(depth[:1], reach, axis=0),
                                 depth]), axis=0)
        rows = np.arange(depth.shape[0])
        above = (c[rows + reach] - c[rows]) / reach  # (the mean depth over the rows just above)
        return (1.0 - self.shade * (1.0 - np.exp(-1.2 * above))).astype(np.float32)

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

    def _up(self, a):
        """A grid's values at the frame's pixels, smoothly."""
        return np.asarray(Image.fromarray(a, "F").resize((self.w, self.h), Image.Resampling.BILINEAR,
                                                         box=(0, 0, self.w / self.h * FINE, FINE)), np.float32)

    def draw(self, t, canvas, mask):
        if self.amount <= 0 or self.most <= 0:
            return
        depth = self.depth_at(self.tau(t))
        cover = (self.most * (1.0 - np.exp(-depth))).astype(np.float32)
        a = np.clip(self._up(cover), 0.0, 1.0)
        lit = self._up(self._shade(depth) * cover)  # (the shade, times the coverage: premultiplied)
        if self._paint is None:
            self._paint = self._colours()
        canvas.over_rgba(0, 0, self._paint * np.clip(lit, 0.0, 1.0)[..., None], a, mask)
