"""Fire in a fireplace ("Lửa lò sưởi"): a wood fire burning on the logs of a fireplace in the video (a placed lens
effect, as the candle): flames that lick up, swirl and break off, white-yellow at their roots, orange and dark red at
their tips, as a fire filmed; embers glowing under them; their light on the firebox; and, if the video already has a
fire that does not move, that fire dimmed under the new one ("Che lửa cũ").

The firebox is lit by the flames themselves (_light): their glow now, spread as a fire's light falls off (brightest
next to them, flickering where they flare), lighting what the picture shows there as surfaces of its colours lit by
the room (the light the room gives the firebox: _room), so bricks are lit brick red and their joints stay dark, not
hazed over.

The box (Effect.BOX) goes over the fireplace's opening. Its mark, a flat ellipse sitting on its bottom edge, goes on
the logs where the fire burns, as wide as the fire ("Lửa rộng"); the flames rise in the box above it.

The flames are a small fluid simulation (Stam's stable fluids): fuel fed at burning places on the bed (on logs the app
draws: along the top of the pile, a little under its outline) that wax and wane, burnt into heat (which rises) and a
short-lived glow (the flame), the air swirling (vorticity confinement). It runs on its own grid over the box (GRID,
the box's shape), RATE steps a second of fire whatever the frame rate, so the box alone (the preview clip) and the
video show the same fire. A fire is made frame after frame, each from the one before, as far as a frame asked for
needs (and kept: fire()); it loops by a crossfade: over its last second the frames before its first come in (their
glow blended, then coloured), so frame P is frame 0 again.

Colour: a real fire's colours by brightness (measured from footage of a wood fire: dark red where faint, orange-red,
orange, yellow, white-yellow where it burns hottest), a glow round the flames as a camera sees it, laid with Screen.
"""
import hashlib
import json
import math
import os
import secrets
import threading

import numpy as np
from PIL import Image

from . import firebox, logs, paths
from .effects import Effect, _choice, _lerp, _num, _place, blur, kept_of

TAU = 2 * math.pi
GRID = (192, 144)  # the simulation's cells over the box (BOX's shape), x then y
RATE = 48.0  # simulation steps a second of fire (more a frame where frames are longer: no step is longer)
WARM = 3.0  # seconds of fire before the loop (well alight: its mean glow measured over the last half)
JOIN = 1.0  # seconds of the crossfade that closes the loop
MARK = 0.09  # the bed's mark on the page: a flat ellipse this high for its width, on the box's bottom edge
CUT = 0.2  # glow this faint is not drawn (no lingering haze above the flames)
PUNCH, PUNCH_K = 1.6, 1.25  # (the flames' brightness drawn: to this power, times this: crisp tongues)
KNEE = 1.5  # (and their brightest parts held back softly under this: yellow hearts, white only at their hottest)
STORE = 3.0  # (the glow is kept as 8 bits: its square root, up to this)
LEVEL = 0.155  # a middling fire's mean glow (a bigger fire throws more light on the room)
KEEP = 240e6  # bytes of fires kept for later (the newest)
ON_DISK = 6  # fires kept on disk (paths.cache_dir; the newest): one is made once, then read by every process that
# needs it (the pieces of a preview clip or an export, the exact frame) and by later clips with the same fire
FIRE_V = 1  # (raised when the simulation makes another fire: the fires kept on disk are not read)
HALO = 0.18  # the glow round the flames (a camera's), as bright as the flames blurred times this
HALO_R = 0.07  # (its blur: box heights)
# the flames' light on the firebox: their glow (as bright as they are drawn) spread as a fire's light falls off (a
# few blurs: their sigmas in box heights, each a share of it), LIGHT_ON times that at "light" 100 lighting what is
# there as white light of that much would (on average a wall as the real fireplace filmed in the owner's example)
SPREAD = ((0.035, 0.35), (0.09, 0.3), (0.2, 0.22), (0.45, 0.13))
LIGHT_ON = 2.5
FIRELIGHT = np.array([1.0, 0.5, 0.17], np.float32)  # (its colour, linear: a wood fire's, as on a wall it lights)
ALBEDO = 0.2  # (the firebox's colours on average: the share of the room's light its sooty surfaces give back)
ALBEDO_MIN, ALBEDO_MAX = 0.03, 0.6  # (black in the picture as dark soot; nothing whiter than this, no colour more)
ROOM_MIN = 0.0004  # (the least light the room gives the firebox: a black picture is not lit infinitely)
WALL_ALBEDO, INSIDE = 0.45, 0.12  # (the wall round a fireplace's opening: how much light it gives back; the share of
# the light on it that reaches inside the fireplace)
# the fire's light on the walls drawn ("Tường lò"): each of their lights as bright as the flames in a band of the grid
# round it (their glow times box heights), times this at "light" 100; and BOUNCE of that (on average) everywhere the
# room's light reaches (light from the walls it lights)
WALL_LIGHT, BOUNCE = 6.0, 0.25
LUT_TOP = 2.2  # (the fire's colours looked up in a table, brightness 0 to this: brighter is as white)
DIM = 0.28  # the old fire put out ("cover" 100): its glow blurred, this bright
PACE = 1.2  # seconds of fire a second of video at "speed" 50 (a fire filmed: tongues come and go quickly)
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
# logs ("Củi"): the flames' light on them (times their glow near each light, over REACH box heights), the embers'
# (times "embers"), the room's (times the light the room gives the firebox: _room), their cracks' glow (times
# "embers"), and how much they keep the light from the floor right under them
FLAME_ON_LOGS, EMBERS_ON_LOGS, ROOM_ON_LOGS, CRACKS = 4.5, 1.6, 1.0, 1.1
REACH = 0.07
BEHIND = 0.5  # (the bright flames' share seen over a log behind the fire, fainter ones less; none over one in front)
FLOOR_SHADE = 0.7
SPARKS = 1.4  # sparks flying up a second at "amount" 100 (fewer for a smaller fire)
SPARK_R = 0.0036  # (a spark's size: box heights)
EMBER_HALO, EMBER_HALO_R = 0.5, 0.012  # (the glow round the logs' burning wood: as bright as it blurred times this; box
# heights)
# a real wood fire's colours by brightness (measured from footage: dark red where faint, orange-red, orange, yellow,
# white-yellow where it burns hottest)
PAL_I = np.array([0.0, 0.12, 0.3, 0.5, 0.7, 0.9, 1.1, 1.35, 1.8], np.float32)
PAL = np.array([[0, 0, 0], [52, 12, 3], [169, 58, 15], [244, 95, 14], [251, 145, 8], [253, 195, 14],
                [253, 228, 94], [254, 239, 173], [255, 250, 228]], np.float32) / 255

DEFAULTS = {"bed": 70, "height": 52, "amount": 75, "flicker": 50, "speed": 50, "bright": 62, "warmth": 50,
            "embers": 45, "light": 50, "cover": 0}
LOOKS = {
    "logs": {},
    "big": {"height": 72, "amount": 95, "flicker": 60, "bright": 72, "embers": 55, "light": 65},
    "low": {"height": 32, "amount": 45, "flicker": 40, "bright": 50, "embers": 55, "light": 40},
    "embers": {"height": 12, "amount": 22, "flicker": 35, "bright": 45, "embers": 90, "light": 35},
}

WITH_LOGS = {"wood": list(logs.WOODS)}  # (the logs' settings show once a wood is chosen)
WITH_WALLS = {"walls": list(firebox.WALLS)}  # (the walls' once a wall is)
FRAMES = ("rect", "arch", "round")  # the fireplace's opening ("Khung lò"): square, a round arch, rounded top corners
WITH_FRAME = {"frame": list(FRAMES)}

SCHEMA = {
    "label": "Lửa lò sưởi",
    "note": "Lửa củi cháy trong lò sưởi có sẵn trong video. (1) Bấm Suốt video. (2) Kéo khung trắng tới lò sưởi: hình "
            "bầu dục vàng ở đáy khung đặt lên mặt củi, chỗ lửa bốc lên; kéo thanh Lửa rộng cho nó rộng bằng đống củi; "
            "kéo góc khung cho lửa cao vừa miệng lò. (3) Video có sẵn lửa đứng yên thì kéo Che lửa cũ lên: lửa cũ tối "
            "đi, lửa mới cháy lên trên. Khúc củi phía trước che chân lửa: tô khúc củi đó vào Tiền cảnh rồi bật Bay ra "
            "sau tiền cảnh cho clip này. Miệng lò trong ảnh hẹp, cao hay hình vòm: chọn Khung lò (chữ nhật, vòm tròn, "
            "bo hai góc trên), kéo ô cam hai bên và ở cạnh trên cho viền cam trùng miệng lò: lửa, ánh lửa và củi chỉ ở "
            "trong miệng lò, không tràn ra mặt lò. Lò trống, chưa có củi: chọn Củi (sồi, vỏ nâu, bạch dương): app vẽ "
            "đống củi (khúc tròn và củi chẻ, xếp chồng như thật) cháy sém đứng trên hình bầu dục vàng (đặt nó lên sàn "
            "lò), lửa bốc lên từ trên đống củi, than hồng trong vết nứt; chỉnh Cách xếp, Khúc củi to, Cháy sém, Nhìn "
            "từ trên xuống cho hợp góc máy. Bên trong lò tối, không rõ tường: chọn Tường lò (gạch chịu lửa, xếp xương "
            "cá, gạch đỏ): app vẽ tường sau, hai tường bên và sàn lò trong miệng lò, gạch sáng lên theo ánh lửa, muội "
            "khói đen phía trên; chỉnh Lò sâu, Viên gạch to, Ám khói đen, Nhìn từ trên xuống. Lò sưởi nhìn nghiêng "
            "(ở tường bên, máy quay thấp hay cao): kéo nút 3D cạnh khung (hoặc các thanh Xoay 3D) cho khung nằm "
            "nghiêng theo mặt lò sưởi; lửa vẫn bốc thẳng lên theo mặt lò. Mỗi lần đổi Ngọn lửa cao, Lửa dày, Bập "
            "bùng, Tốc độ (và cỡ, cách xếp củi), app tính lại lửa từ đầu: dừng hình lần đầu có thể chờ vài giây.",
    # for the page: where the fire burns, a flat ellipse sitting on the middle of the box's bottom edge, as wide as
    # "bed" (% of the box's width)
    "mark": {"width": "bed", "ratio": MARK, "sits": True, "label": "chỗ lửa cháy (mặt củi)",
             "tip": "Đặt hình bầu dục vàng lên mặt củi, chỗ lửa bốc lên (đã chọn Củi: lên sàn lò, chỗ đặt củi); kéo "
                    "thanh Lửa rộng cho nó rộng bằng đống củi, kéo góc khung cho lửa cao vừa lò sưởi."},
    # for the page: the fireplace's opening (an outline on the box's bottom edge, its width and height shares of the
    # box's, its top corners rounded as frame_radii says; its handles change them)
    "frame": {"shape": "frame", "width": "frame_w", "height": "frame_h", "curve": "curve", "label": "miệng lò",
              "tip": "Kéo ô cam hai bên và ở giữa cạnh trên cho viền cam trùng miệng lò sưởi trong ảnh: lửa, ánh lửa "
                     "và củi chỉ ở trong miệng lò."},
    "params": [
        _choice("look", "Kiểu lửa", "logs", [("logs", "Lửa củi (vừa)"), ("big", "Lửa to, cháy mạnh"),
                                             ("low", "Lửa nhỏ, dịu"), ("embers", "Sắp tàn, than hồng")],
                presets={k: {key: v for key, v in {**DEFAULTS, **look}.items() if key not in ("bed", "cover")}
                         for k, look in LOOKS.items()}, group="Kiểu"),
        _choice("frame", "Khung lò (miệng lò trong ảnh)", "none",
                [("none", "Không (lửa trong cả khung trắng)"), ("rect", "Chữ nhật"), ("arch", "Vòm tròn"),
                 ("round", "Bo hai góc trên")], group="Khung lò"),
        _num("frame_w", "Khung lò rộng (% khung trắng)", 80, 20, 100, group="Khung lò", show=WITH_FRAME),
        _num("frame_h", "Khung lò cao (% khung trắng)", 85, 20, 100, group="Khung lò", show=WITH_FRAME),
        _num("curve", "Độ cong (vòm, góc bo)", 60, group="Khung lò", show={"frame": ["arch", "round"]}),
        _choice("wood", "Củi", "none", [("none", "Không thêm (lửa cháy trên củi có sẵn trong video)"),
                                         ("oak", "Củi sồi (vỏ xám, rãnh sâu)"), ("brown", "Củi vỏ nâu"),
                                         ("birch", "Củi bạch dương (vỏ trắng)")], group="Củi"),
        _choice("stack", "Cách xếp", "stack", [("stack", "Ba khúc: hai dưới, một trên"),
                                              ("heap", "Đống củi (năm khúc)"), ("two", "Hai khúc nằm cạnh nhau")],
                group="Củi", show=WITH_LOGS),
        _num("log_size", "Khúc củi to", 50, group="Củi", show=WITH_LOGS),
        _num("char", "Cháy sém (than đen, vết nứt)", 65, group="Củi", show=WITH_LOGS),
        _choice("walls", "Tường lò", "none", [("none", "Giữ tường trong video"),
                                              ("firebrick", "Gạch chịu lửa (vàng nâu)"),
                                              ("herring", "Gạch chịu lửa xếp xương cá"), ("red", "Gạch đỏ")],
                group="Tường lò"),
        _num("deep", "Lò sâu", 50, group="Tường lò", show=WITH_WALLS),
        _num("brick", "Viên gạch to", 50, group="Tường lò", show=WITH_WALLS),
        _num("soot", "Ám khói đen", 55, group="Tường lò", show=WITH_WALLS),
        _num("view", "Nhìn từ trên xuống (ngang ← → cao)", 50, group="Góc máy", show=[WITH_LOGS, WITH_WALLS]),
        _num("bed", "Lửa rộng (hình bầu dục vàng)", DEFAULTS["bed"], 10, 100, group="Ngọn lửa"),
        _num("height", "Ngọn lửa cao", DEFAULTS["height"], group="Ngọn lửa"),
        _num("amount", "Lửa dày", DEFAULTS["amount"], group="Ngọn lửa"),
        _num("flicker", "Bập bùng", DEFAULTS["flicker"], group="Ngọn lửa"),
        _num("speed", "Tốc độ (chậm ← → nhanh)", DEFAULTS["speed"], group="Ngọn lửa"),
        _num("bright", "Độ sáng", DEFAULTS["bright"], group="Màu, ánh sáng"),
        _num("warmth", "Màu lửa (vàng ← → đỏ cam)", DEFAULTS["warmth"], group="Màu, ánh sáng"),
        _num("embers", "Than hồng dưới chân lửa", DEFAULTS["embers"], group="Màu, ánh sáng"),
        _num("light", "Ánh lửa hắt lên lò sưởi", DEFAULTS["light"], group="Màu, ánh sáng"),
        _num("cover", "Che lửa cũ trong video", DEFAULTS["cover"], group="Màu, ánh sáng"),
        *_place(60)]}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _soft(a, radius):
    """A float picture (h x w, or h x w x c) blurred (Gaussian, `radius` its sigma in pixels): on a copy a few times
    smaller (as wide a blur, a little smoother), then enlarged."""
    h, w = a.shape[:2]
    step = max(1, min(int(radius / 2), h, w))
    small = (max(1, round(w / step)), max(1, round(h / step)))
    planes = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[2])]
    out = []
    for plane in planes:
        p = Image.fromarray(np.ascontiguousarray(plane, np.float32), "F").resize(small, Image.Resampling.BOX)
        p = blur(np.asarray(p, np.float32), radius / step)
        out.append(np.asarray(Image.fromarray(p, "F").resize((w, h), Image.Resampling.BILINEAR), np.float32))
    return out[0] if a.ndim == 2 else np.stack(out, 2)


def frame_radii(shape, fw, fh, curve):
    """The opening's top corners for a frame fw x fh wide and high (any unit): the radii (rx across, ry down) of the
    quarter ellipses that round them, as CSS draws border radii: none (rect), a round arch over the whole width
    ("arch", as high as `curve` says, 0..100: up to a half circle), or rounded corners ("round")."""
    k = min(max(curve, 0.0), 100.0) / 100
    if shape == "arch":
        return fw / 2, min(fh, k * fw / 2)
    if shape == "round":
        r = k * 0.6 * min(fw / 2, fh)
        return r, r
    return 0.0, 0.0


def palette(i, warmth=0.5):
    """A fire's colours (... x 3) for brightness i: the measured ones, a little yellower or redder."""
    x = np.asarray(i, np.float32) * (1.0 / (1.0 + 0.4 * (warmth - 0.5)))  # (redder: brighter glow for one colour)
    out = np.empty(x.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(x, PAL_I, PAL[:, c])
    return out


class _Fluid:
    """The fire's air on the grid (x right, y up, in cells): its velocity, heat, fuel and the flames' glow."""

    def __init__(self, k, seed):
        gw, gh = GRID
        self.k = k
        self.u, self.v, self.heat, self.fuel, self.glow, self.p = (np.zeros((gh, gw), np.float32) for _ in range(6))
        ys, xs = np.mgrid[0:gh, 0:gw]
        self.X, self.Y = xs.astype(np.float32), ys.astype(np.float32)
        # the burning places: along the bed (its ellipse), at different depths in it, each waxing and waning; on
        # logs, along the top of the pile (a little under its outline: their roots hidden by the logs), more of them
        # and stronger in its middle
        g = np.random.default_rng([seed, 7103])
        bed = k["bed"]
        n = max(2, int(round(k["places"] * bed)))
        u = np.linspace(-1, 1, n) * 0.9 + g.uniform(-0.05, 0.05, n)
        self.at = (u * bed + 1) / 2 * gw
        half = MARK * bed * gw / 2  # (the ellipse's half height, cells)
        self.up = 1.5 + half * (1 + 0.7 * np.sqrt(np.maximum(0.0, 1 - u * u)) * g.uniform(-1, 1, n))
        if "crest" in k:
            crest = np.asarray(k["crest"], np.float32)
            cols = np.flatnonzero(crest > 0)
            lo, hi = float(cols.min()), float(cols.max())
            u = np.sign(u) * np.abs(u) ** k.get("gather", 1.4)  # (closer together in the middle)
            self.at = (lo + hi) / 2 + u * (hi - lo) / 2 * 0.92
            self.up = np.interp(self.at, np.arange(gw), np.where(crest > 0, crest, crest.max())) - \
                g.uniform(0.0, 3.0, n)
        self.wide = g.uniform(0.035, 0.07, n) * gw * k.get("narrow", 1.0)
        self.hz = g.uniform(0.3, 1.2, n) * k["waver"]
        self.ph = g.random(n) * TAU
        self.strength = g.uniform(0.6, 1.0, n) * (1 - (0.5 if "crest" in k else 0.35) * u * u)
        rows = int(math.ceil(self.up.max() + 4))
        self.rows = slice(0, rows)
        dy = self.Y[:rows, :1] - self.up  # (rows x places)
        self.shape = np.exp(-(dy / 2.0) ** 2).astype(np.float32)  # each place's feed down the rows
        self.across = np.exp(-((self.X[0][:, None] - self.at) / self.wide) ** 2).astype(np.float32)  # and across
        self.s = 0.0  # seconds of fire

    def step(self, dt):
        """One step of dt (1/RATE seconds each, at most 1)."""
        k = self.k
        a = self.strength * (1 - k["wax"] + k["wax"] * np.sin(TAU * self.hz * self.s + self.ph))
        feed = (self.shape * a.astype(np.float32)) @ self.across.T  # (rows x cells)
        rows = self.rows
        self.fuel[rows] += k["feed"] * dt * feed
        np.maximum(self.v[rows], k["jet"] * feed, out=self.v[rows])
        burn = self.fuel * (1 - (1 - k["burn"]) ** dt)
        self.fuel -= burn
        self.heat += burn
        self.glow += burn * k["glow"]
        self.v += k["lift"] * dt * self.heat
        self.heat *= k["cool"] ** dt
        self.glow *= k["fade"] ** dt
        self._swirl(k["swirl"], dt)
        np.clip(self.u, -6.0, 6.0, out=self.u)  # (never more than a few cells a step: stable)
        np.clip(self.v, -6.0, 8.0, out=self.v)
        self._project()
        self.u, self.v, self.heat, self.fuel, self.glow = self._advect(
            (self.u, self.v, self.heat, self.fuel, self.glow), dt, (3, 4))
        damp = 0.985 ** dt
        self.u *= damp
        self.v *= damp
        self.s += dt / RATE

    def _advect(self, fields, dt, sharp=()):
        """Every field carried by the air (semi-Lagrangian: each cell takes what was where its air came from); the
        fields numbered in `sharp` by MacCormack (the step undone, half its error put back, within what was there):
        the flames' edges stay crisp."""
        here = self._where(self.X - self.u * dt, self.Y - self.v * dt)
        there = self._where(self.X + self.u * dt, self.Y + self.v * dt) if sharp else None
        out = []
        for n, f in enumerate(fields):
            if n in sharp:
                moved, lo, hi = self._sample(f, here, True)
                back = self._sample(moved, there)
                back -= f
                back *= -0.5
                back += moved
                out.append(np.clip(back, lo, hi, out=back))
            else:
                out.append(self._sample(f, here))
        return out

    @staticmethod
    def _where(x, y):
        """Where points fall on the grid: the cell under each, and the weights of its four corners."""
        gw, gh = GRID
        np.clip(x, 0, gw - 1.001, out=x)
        np.clip(y, 0, gh - 1.001, out=y)
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0  # (float32 throughout: the fields stay float32)
        gx, gy = 1 - fx, 1 - fy
        return y0.astype(np.int32) * gw + x0.astype(np.int32), gx * gy, fx * gy, gx * fy, fx * fy

    @staticmethod
    def _sample(f, where, bounds=False):
        """A field at points (bilinear), and the least and most of the four cells round each (bounds)."""
        i, w00, w01, w10, w11 = where
        gw = GRID[0]
        flat = f.ravel()
        a, b, c, d = flat.take(i), flat.take(i + 1), flat.take(i + gw), flat.take(i + gw + 1)
        v = a * w00
        v += b * w01
        v += c * w10
        v += d * w11
        if not bounds:
            return v
        return v, np.minimum(np.minimum(a, b), np.minimum(c, d)), np.maximum(np.maximum(a, b), np.maximum(c, d))

    def _project(self, iters=20):
        """The air made to neither pile up nor thin out (Jacobi): walls at the sides and under it, open above."""
        u, v, p = self.u, self.v, self.p
        gw = GRID[0]
        # (on the rows inside, as one run of cells: their first and last cells come out wrong, and are then set as
        # the walls are; each sum a run of memory, not a strided block)
        a, b = gw + 1, u.size - gw - 1
        uf, vf, pf = u.ravel(), v.ravel(), p.ravel()
        div = np.subtract(uf[a + 1:b + 1], uf[a - 1:b - 1])
        div += vf[a + gw:b + gw]
        div -= vf[a - gw:b - gw]
        div *= 0.5
        p[:] = 0.0
        part = np.empty_like(div)
        for _ in range(iters):
            np.add(pf[a + 1:b + 1], pf[a - 1:b - 1], out=part)
            part += pf[a + gw:b + gw]
            part += pf[a - gw:b - gw]
            part -= div
            part *= 0.25
            pf[a:b] = part
            p[:, 0], p[:, -1], p[0] = p[:, 1], p[:, -2], p[1]
            p[-1] = 0.0
        u[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2])
        v[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1])

    def _swirl(self, eps, dt):
        """Vorticity confinement: the air's small swirls kept up (the grid would smooth them away)."""
        u, v = self.u, self.v
        w = np.zeros_like(u)
        w[1:-1, 1:-1] = 0.5 * (v[1:-1, 2:] - v[1:-1, :-2] - u[2:, 1:-1] + u[:-2, 1:-1])
        a = np.abs(w)
        gx, gy = np.zeros_like(u), np.zeros_like(u)
        gx[1:-1, 1:-1] = 0.5 * (a[1:-1, 2:] - a[1:-1, :-2])
        gy[1:-1, 1:-1] = 0.5 * (a[2:, 1:-1] - a[:-2, 1:-1])
        n = np.sqrt(gx * gx + gy * gy) + 1e-5
        u += eps * dt * (gy / n) * w
        v -= eps * dt * (gx / n) * w


class _Fire:
    """One loop of a simulated fire: its frames made one after another as far as they are asked for, kept (the
    glow's square root as 8 bits), with the second before the loop (for the crossfade that closes it)."""

    def __init__(self, k, seed, frames, fps):
        self.k, self.seed, self.P = k, seed, int(frames)
        per = RATE * k["pace"] / fps  # (steps a frame)
        self.steps = max(1, math.ceil(per - 1e-9))
        self.dt = per / self.steps
        self.join = int(min(round(JOIN * fps), self.P // 4))
        self.warm = int(math.ceil(WARM * fps / k["pace"]))  # (frames)
        self.mean = LEVEL  # (its mean glow, measured as it is lit)
        gw, gh = GRID
        self.kept = np.zeros((self.join + self.P, gh, gw), np.uint8)  # frames -join..P-1
        self.level = np.zeros(self.join + self.P, np.float32)
        self.made = 0
        self.fluid = None
        self.lock = threading.Lock()
        key = json.dumps([FIRE_V, sorted(k.items()), int(seed), self.P, round(float(fps), 6)])
        self.file = "fire-" + hashlib.sha1(key.encode()).hexdigest()[:16] + ".npz"
        self._read()

    def _read(self):
        """The whole fire, if made before and kept on disk."""
        path = os.path.join(paths.cache_dir(), self.file)
        try:
            with np.load(path) as kept:
                if kept["kept"].shape != self.kept.shape or kept["level"].shape != self.level.shape:
                    return
                self.kept[:], self.level[:], self.mean = kept["kept"], kept["level"], float(kept["mean"])
            self.made = len(self.kept)
            os.utime(path)  # (used now: the last kept)
        except (OSError, KeyError, ValueError, EOFError):  # (none yet, or not readable: made here)
            pass

    def _write(self):
        """The whole fire kept on disk (written whole under another name first: a half-written one is never read),
        and only the ON_DISK used last."""
        folder = paths.cache_dir()
        tmp = os.path.join(folder, f"{self.file}.{secrets.token_hex(3)}.part")
        try:
            with open(tmp, "wb") as f:
                np.savez_compressed(f, kept=self.kept, level=self.level, mean=np.float64(self.mean))
            os.replace(tmp, os.path.join(folder, self.file))
        except OSError:  # (a full disk, or another process reading it as it is replaced: only slower next time)
            try:
                os.remove(tmp)
            except OSError:
                pass
            return
        try:
            fires = sorted((e for e in os.scandir(folder) if e.name.startswith("fire-") and e.name.endswith(".npz")),
                           key=lambda e: e.stat().st_mtime, reverse=True)
            for old in fires[ON_DISK:]:
                os.remove(old.path)
        except OSError:  # (one another process is reading: removed later)
            pass

    def _make(self, upto):
        """Frames made until index `upto` (frame upto - join) is."""
        if self.fluid is None and self.made <= upto:
            self.fluid = _Fluid(self.k, self.seed)
            seen = []
            for i in range(self.warm):
                for _ in range(self.steps):
                    self.fluid.step(self.dt)
                if 2 * i >= self.warm:
                    seen.append(float(np.maximum(self.fluid.glow, 0.0).mean()))
            mean = float(np.mean(seen)) if seen else 0.0
            self.mean = mean if math.isfinite(mean) and mean > 1e-4 else LEVEL
        while self.made <= upto:
            for _ in range(self.steps):
                self.fluid.step(self.dt)
            g = np.maximum(self.fluid.glow[::-1], 0.0)
            ok = np.isfinite(g)
            if not ok.all():  # (never: the speeds are bounded; but a broken frame stays dark, not white)
                g = np.where(ok, g, 0.0)
            self.kept[self.made] = np.sqrt(np.minimum(g, STORE) * (1 / STORE)) * 255.0 + 0.5
            self.level[self.made] = float(g.mean())
            self.made += 1
        if self.made == len(self.kept) and self.fluid is not None:
            self.fluid = None  # (all made: its fields are not needed any more)
            self._write()

    def _glow(self, i):
        g = self.kept[i].astype(np.float32) * (1 / 255.0)
        g *= g
        g *= STORE
        return g, float(self.level[i])

    def frame(self, t):
        """The flames' glow at frame t of the loop (float32, rows x columns of the grid, row 0 at the top) and the
        fire's mean glow then."""
        t = int(t) % self.P
        s = t - (self.P - self.join)
        with self.lock:
            self._make(self.join + t)
            g, level = self._glow(self.join + t)
            if s >= 0:  # its last second: more and more the frames before its first (frame P is frame 0 again)
                w = float(_smooth((s + 0.5) / self.join))
                g0, l0 = self._glow(s)
                g *= 1 - w
                g += w * g0
                level += w * (l0 - level)
        return g, level


_FIRES = {}  # (what a fire is made of, its seed, its loop) -> _Fire, the newest last
_FIRES_LOCK = threading.Lock()


def fire(k, seed, frames, fps):
    """The fire of these settings (kept: the newest up to KEEP bytes)."""
    key = (tuple(sorted(k.items())), int(seed), int(frames), round(float(fps), 6))
    with _FIRES_LOCK:
        made = _FIRES.pop(key, None) or _Fire(k, seed, frames, fps)
        _FIRES[key] = made
        size = sum(f.kept.nbytes for f in _FIRES.values())
        for old in list(_FIRES)[:-1]:
            if size <= KEEP:
                break
            size -= _FIRES.pop(old).kept.nbytes
        return made


class Hearth(Effect):
    BOX = (4.0, 3.0)
    blend = "screen"
    lens = True
    UNDER = True  # its preview clip: the picture under the box with the fire in it (the old fire dimmed: "cover")
    UNDER_FADE = 0.0  # (its change already fades out toward the box's edges; its logs stand on its bottom edge)
    # turned in 3D as a file clip is, to sit in a fireplace seen from one side or from below (the owner, 2026-10-26:
    # "hiệu ứng lò sưởi cần thêm di chuyển 3D để phù hợp với các lò sưởi của tôi"): drawn upright on its sheet,
    # warped with it (sheet.Sheet), its change a layer over the frame (apply's kept)
    TURNS = True
    VERSION = 5  # (2: logs burnt where the fire reaches them, glowing red; fuller flames; 3: the firebox lit by the
    # flames, the logs on a bed of glowing coals, brighter flames; 4: the firebox's walls drawn, if chosen; 5: split
    # firewood come to rest on each other, flames from the top of the pile, crisper flames, patchier embers)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        h, n, f = (p[key] / 100 for key in ("height", "amount", "flicker"))
        self.bed = p["bed"] / 100
        self.k = {  # (what the simulation is made of: changing any makes the fire again)
            "bed": round(self.bed, 3), "places": round(_lerp(7.0, 15.0, n), 3),
            "feed": round(0.55 * 2 ** ((n - 0.5) * 1.4), 4), "jet": round(_lerp(1.05, 1.85, h), 4),
            "lift": round(_lerp(0.19, 0.35, h), 4), "fade": round(_lerp(0.87, 0.915, h), 4),
            "burn": 0.2, "glow": 1.7, "cool": 0.93,
            "swirl": round(_lerp(1.7, 2.3, f), 3), "wax": round(_lerp(0.25, 0.55, f), 3),
            "waver": round(_lerp(0.6, 1.4, f), 3), "pace": round(PACE * 2 ** ((p["speed"] - 50) / 50), 4)}
        self.bright_k = _lerp(0.75, 1.45, p["bright"] / 100)
        self.warmth = p["warmth"] / 100
        self.embers_k, self.light_k, self.cover_k = p["embers"] / 100, p["light"] / 100, p["cover"] / 100
        g = np.random.default_rng([self.seed, 7103, 2])
        self.ember_m = self.cycles(np.array([0.21, 0.37, 0.59]) * self.k["pace"])  # (whole cycles a loop)
        self.ember_ph = g.random(3) * TAU
        self._fire = None
        self._embers = None
        self._floor = None
        self._gshown = None
        self._laid = None
        self._lut = palette(np.linspace(0.0, LUT_TOP, 1024, dtype=np.float32), self.warmth)
        self.light_colour = FIRELIGHT ** (1 + 0.6 * (self.warmth - 0.5))  # (the light of yellower or redder flames)
        self.frame = p.get("frame", "none") if p.get("frame") in FRAMES else None  # (the fireplace's opening)
        self.frame_w, self.frame_h = p.get("frame_w", 80) / 100, p.get("frame_h", 85) / 100
        self.pile = None
        self._windows = None
        # sparks flying up from the fire: whole sparks a loop, each born at its own moment of it, rising, wavering,
        # fading out (the loop's last frame runs on into its first)
        g = np.random.default_rng([self.seed, 7103, 5])
        count = int(round(SPARKS * (0.25 + 0.75 * n) * self.loop_s))
        self.sparks = {"born": (np.arange(count) + g.random(count)) / max(count, 1), "x": g.uniform(-0.42, 0.42, count),
                       "life": g.uniform(0.6, 1.5, count), "rise": g.uniform(0.3, 0.7, count),
                       "wob": g.uniform(0.008, 0.03, count), "hz": g.uniform(1.5, 4.0, count),
                       "ph": g.random(count) * TAU, "drift": g.uniform(-0.06, 0.06, count),
                       "hot": g.uniform(0.5, 1.0, count)}
        self.walls = None
        self._wall_w = None
        if p["walls"] in firebox.WALLS:  # the firebox drawn: its back, its sides, its floor, inside the opening; the
            # fire on its floor where the bed is, its light from a little above that
            box = self.BOX[0] / self.BOX[1]
            high = self.frame_h if self.frame is not None else 1.0
            wide = box * (self.frame_w if self.frame is not None else 1.0)
            self.walls = firebox.Firebox(p["walls"], wide, high, high * _lerp(0.35, 0.95, p["deep"] / 100),
                                         p["brick"] / 100, p["soot"] / 100, math.radians(5 + 0.3 * p["view"]),
                                         self.seed, bed=self.bed * box, mid=MARK * self.bed * box / 2,
                                         flames=_lerp(0.1, 0.3, h))
        if p.get("wood") in logs.WOODS:  # logs under the flames (box heights from the bed's middle)
            wide = self.BOX[0] / self.BOX[1]
            half = wide / 2 * (0.97 if self.frame is None else 0.95 * self.frame_w)  # (in the box, in its opening)
            self.pile = logs.Pile(p["wood"], p["stack"], p["log_size"] / 100, p["char"] / 100, self.bed * wide,
                                  math.radians(5 + 0.3 * p["view"]), self.seed, half=half,
                                  floor=-MARK * self.bed * wide / 2 + 0.006)
            # (the top of the pile's outline over the grid's columns, as cells up from the box's bottom, a little
            # lower: the flames come up from there, from among the logs; 0 where no log is)
            gw, gh = GRID
            top = self.pile.crest(((np.arange(gw) + 0.5) / gw - 0.5) * wide)
            roots = MARK * self.bed * gw / 2 + (top - 0.3 * self.pile.r) * gh
            self.k["crest"] = tuple(round(float(v), 1) if np.isfinite(v) and v > 1 else 0.0 for v in roots)
            # (from the logs' tops: as many tongues, narrower, a little lower and less fed than from a bed of embers)
            self.k.update(feed=round(self.k["feed"] * 0.8, 4), jet=round(self.k["jet"] * 0.89, 4),
                          fade=round(self.k["fade"] ** 1.13, 4), narrow=0.7, gather=1.15)

    def frames(self):
        """This clip's fire (made as far as it is asked for, kept for the next clip with the same settings)."""
        if self._fire is None:
            self._fire = fire(self.k, self.seed, self.P, self.fps)
        return self._fire

    def _rect(self, x0, y0, x1, y1):
        r = (max(0, math.floor(x0)), max(0, math.floor(y0)), min(self.w, math.ceil(x1)), min(self.h, math.ceil(y1)))
        return r if r[0] < r[2] and r[1] < r[3] else None

    def _ember_field(self):
        """Where the embers glow on the grid (patches over the bed's ellipse), and a phase for each place."""
        if self._embers is None:
            gw, gh = GRID
            g = np.random.default_rng([self.seed, 7103, 3])
            coarse = g.random((gh // 6 + 3, gw // 6 + 3)).astype(np.float32)
            patches = np.asarray(Image.fromarray(coarse, "F").resize((gw, gh), Image.Resampling.BICUBIC), np.float32)
            fine = g.random((gh, gw)).astype(np.float32)
            ys, xs = np.mgrid[0:gh, 0:gw].astype(np.float32)
            u = ((xs + 0.5) / gw * 2 - 1) / max(self.bed, 1e-3)  # (the ellipse: -1..1 across)
            half = MARK * self.bed * gw / 2  # (its half height, cells)
            v = ((gh - 0.5 - ys) - half) / max(half, 1.0)
            inside = 1 - (u * u + v * v)
            field = _smooth(inside / 0.5) * np.clip(patches * 1.8 - 0.5, 0, 1) * (0.7 + 0.3 * fine)
            self._embers = field.astype(np.float32), np.clip(patches, 0, 1).astype(np.float32)
        return self._embers

    def _floor_at(self, xs):
        """How high (box heights) the flames come out of the bed across the box (xs: 0..1): an uneven line in its
        ellipse, as from between logs, not a straight floor."""
        if self._floor is None:
            g = np.random.default_rng([self.seed, 7103, 4])
            self._floor = g.random(12)
        u = np.clip(xs, 0, 1) * (len(self._floor) - 1)
        i = np.minimum(u.astype(np.int32), len(self._floor) - 2)
        f = _smooth(u - i)
        bumps = self._floor[i] * (1 - f) + self._floor[i + 1] * f
        return (MARK * self.bed * 4 / 3) * (0.15 + 0.6 * bumps)

    @staticmethod
    def _edge(xs, ys):
        """Nothing cut at the box's edges: how much shows at points of the box (xs across, ys down: 0..1)."""
        side = _smooth((np.minimum(xs, 1 - xs) - 0.01) / 0.05)  # (none on the box's outer hundredth)
        return (side[None, :] * (_smooth((ys - 0.01) / 0.05) * _smooth((1 - ys - 0.01) / 0.025))[:, None]).astype(
            np.float32)

    def _shown(self, xs, ys):
        """How much of the flames shows at points of the box: none at its edges, none under the uneven line they
        come out at (_floor_at)."""
        return (self._edge(xs, ys) * _smooth((1 - ys[:, None] - self._floor_at(xs)[None, :]) / 0.025 + 0.5)).astype(
            np.float32)

    def _grid_shown(self):
        """_shown() on the grid's cells."""
        if self._gshown is None:
            gw, gh = GRID
            self._gshown = self._shown((np.arange(gw, dtype=np.float32) + 0.5) / gw,
                                       (np.arange(gh, dtype=np.float32) + 0.5) / gh)
        return self._gshown

    def _layout(self, r, x0, y0, w, h):
        """What stays the same from frame to frame where the box is (`r`: its part in the frame), kept while the
        box stays."""
        key = (r, x0, y0, w, h)
        if self._laid is not None and self._laid[0] == key:
            return self._laid[1]
        rx0, ry0, rx1, ry1 = r
        gw, gh = GRID
        ys = (np.arange(ry0, ry1, dtype=np.float32) + 0.5 - y0) / h  # (0..1 down the box)
        xs = (np.arange(rx0, rx1, dtype=np.float32) + 0.5 - x0) / w
        edge = self._edge(xs, ys)
        lay = {"size": (rx1 - rx0, ry1 - ry0), "edge": edge, "shown": self._shown(xs, ys), "box_h": h,
               # (the part of the grid that shows: whole pixels, up to a pixel more than the box at its edges)
               "src": (max(0.0, (rx0 - x0) / w * gw), max(0.0, (ry0 - y0) / h * gh), min(gw, (rx1 - x0) / w * gw),
                       min(gh, (ry1 - y0) / h * gh))}
        # (the firebox: inside the box's edges, inside its opening if it has one; its walls drawn to the box's sides
        # and top if it has none)
        if self.frame is not None:
            lay["inside"] = edge * self._opening(np.arange(rx0, rx1) + 0.5 - x0, np.arange(ry0, ry1) + 0.5 - y0, w, h)
        elif self.walls is not None:
            lay["inside"] = self._opening(np.arange(rx0, rx1) + 0.5 - x0, np.arange(ry0, ry1) + 0.5 - y0, w, h, True)
        else:
            lay["inside"] = edge
        if self.walls is not None:  # (its walls drawn: as the room lights them; and the room round the opening)
            X = (np.arange(rx0, rx1, dtype=np.float64) + 0.5 - (x0 + w / 2)) / h
            Z = (y0 + h - np.arange(ry0, ry1, dtype=np.float64) - 0.5) / h
            lay["walls"] = self.walls.lay(*np.meshgrid(X, Z))
            if self.frame is not None:
                lay["round"] = edge * (1 - lay["inside"])
        if self.cover_k > 0:  # where an old fire is put out: over the bed and above it, not at the box's bottom edge
            lay["zone"] = (_smooth((self.bed * 1.25 - np.abs(xs * 2 - 1)) / 0.15)[None, :] * edge * _smooth(
                (1 - ys) / (MARK * self.bed * 4 / 3 + 0.02))[:, None]).astype(np.float32)
        if self.frame is not None:  # the fireplace's opening: all of it inside, nothing outside
            lay["frame"] = self._opening(np.arange(rx0, rx1) + 0.5 - x0, np.arange(ry0, ry1) + 0.5 - y0, w, h)
        if self.pile is not None:  # the logs: traced over their part of the box (in the frame)
            cx, cy = x0 + w / 2, y0 + h - MARK * self.bed * w / 2  # (the bed's middle)
            left, right, bottom, top = self.pile.extent()
            px0, px1 = max(rx0, math.floor(cx + left * h)), min(rx1, math.ceil(cx + right * h))
            py0, py1 = max(ry0, math.floor(cy - top * h)), min(ry1, math.ceil(cy - bottom * h))
            if px0 < px1 and py0 < py1:
                lay["pile"] = (px0 - rx0, py0 - ry0, px1 - rx0, py1 - ry0)
                lay["logs"] = self.pile.lay((np.arange(px0, px1) + 0.5 - cx) / h, (cy - np.arange(py0, py1) - 0.5) / h,
                                            1.0 / h)
        self._laid = (key, lay)
        return lay

    def _opening(self, xs, ys, w, h, whole=False):
        """How much of the fire shows at points of the box (xs across, ys down: pixels from its top left corner):
        inside the fireplace's opening ("Khung lò": on the box's bottom edge, frame_w of its width and frame_h of its
        height, its top corners rounded as frame_radii says; `whole`: the whole box), none outside it, softly at its
        sides and top (not along its bottom: the floor of the fireplace)."""
        fw, fh = (w, h) if whole else (self.frame_w * w, self.frame_h * h)
        rx, ry = (0.0, 0.0) if whole else frame_radii(self.frame, fw, fh, self.p.get("curve", 60))
        u = np.asarray(xs, np.float64) - (w - fw) / 2
        side = np.minimum(u, fw - u)[None, :]  # (from the nearer side, pixels)
        down = (np.asarray(ys, np.float64) - (h - fh))[:, None]  # (from the top)
        d = np.minimum(side, down)
        if rx > 0.5 and ry > 0.5:  # (the corners: quarter ellipses; the distance to one about, by its slope)
            a, b = (side - rx) / rx, (down - ry) / ry
            q = np.sqrt(a * a + b * b)
            slope = np.sqrt((a / rx) ** 2 + (b / ry) ** 2) / np.maximum(q, 1e-9)
            corner = (side < rx) & (down < ry)
            d = np.where(corner, np.minimum(d, (1 - q) / np.maximum(slope, 1e-9)), d)
        soft = max(1.0, 0.006 * h)
        return _smooth(d / (2 * soft) + 0.5).astype(np.float32)

    def _light_windows(self):
        """For each of the logs' flame lights: the part of the grid round it and its weights there (the flames'
        light on it is their glow there)."""
        if self._windows is None:
            gw, gh = GRID
            wide = self.BOX[0] / self.BOX[1]
            out = []
            for at, kind in self.pile.lights:
                if kind != "flame":
                    continue
                x, s = self.pile.screen(at)
                col = (0.5 + x / wide) * gw
                row = (1 - (s + MARK * self.bed * wide / 2)) * gh
                rad = REACH * gh
                c0, c1 = max(0, int(col - 2 * rad)), min(gw, int(col + 2 * rad) + 1)
                r0, r1 = max(0, int(row - 2 * rad)), min(gh, int(row + 2 * rad) + 1)
                yy, xx = np.mgrid[r0:r1, c0:c1].astype(np.float32)
                wt = np.exp(-((xx + 0.5 - col) ** 2 + (yy + 0.5 - row) ** 2) / (2 * rad * rad))
                out.append(((slice(r0, r1), slice(c0, c1)), wt / max(float(wt.sum()), 1e-9)))
            self._windows = out
        return self._windows

    def _logs(self, t, out, lay, glow, level, room, lit=None):
        """The logs on `out` (the box's part in the frame, changed in place): their shadow on the floor, their bed
        of coals and ash, then them, lit by the room (`room`: the light it gives the firebox, as _room says), the
        flames near each light (`glow`: the flames on the grid) and the embers; their cracks glowing; `level`: how
        bright the fire is now (1: as on average); `lit`: how bright the flames' light is over the box (as _light
        gives it; None: none)."""
        x0, y0, x1, y1 = lay["pile"]
        laid = lay["logs"]
        part = out[y0:y1, x0:x1]
        alpha = laid["alpha"][..., None]
        floor = laid["floor"][..., None]
        part *= 1 - FLOOR_SHADE * floor
        room = np.full(3, room * ROOM_ON_LOGS, np.float32)
        bright = self.bright_k * level
        flames = [float((glow[at] * wt).sum()) * FLAME_ON_LOGS * self.bright_k for at, wt in self._light_windows()]
        tau = self.tau(t)
        bed = laid["bed"][..., None]
        bed_given = None
        if bed.any():  # (the bed of coals and ash: lit by the room and the flames, less near the logs; its coals
            # glowing, breathing)
            light = room * (1 - FLOOR_SHADE * floor)
            if lit is not None:
                light = light + lit[y0:y1, x0:x1, None] * (1 - 0.5 * floor) * self.light_colour
            coals, bed_given = self.pile.bed_colour(laid, light, CRACKS * self.embers_k * bright, tau, self.ember_m)
            np.clip(coals, 0.0, 1.0, out=coals)
            coals **= 1 / logs.GAMMA
            part += bed * (coals - part)
        breathe = sum(math.sin(TAU * m * tau + ph) for m, ph in zip(self.ember_m, self.ember_ph)) / 3
        ember = EMBERS_ON_LOGS * self.embers_k * (0.85 + 0.15 * breathe) * bright
        embers = [ember] * sum(kind == "ember" for _, kind in self.pile.lights)
        colour, given = self.pile.colour(laid, room, flames, embers, CRACKS * self.embers_k * bright, tau,
                                         self.ember_m)
        np.clip(colour, 0.0, 1.0, out=colour)
        colour **= 1 / logs.GAMMA
        part += alpha * (colour - part)
        if bed_given is not None:  # (the coals' light: where the logs do not hide them)
            bed_given *= bed * (1 - alpha)
            given = bed_given if given is None else given * alpha + bed_given
        elif given is not None:
            given = given * alpha
        if given is not None:  # (the glow round the burning wood, a camera's: over the logs and past their edges)
            halo = _soft(given, max(1.0, EMBER_HALO_R * lay["box_h"]))
            np.clip(halo * EMBER_HALO, 0.0, 1.0, out=halo)
            halo **= 1 / logs.GAMMA
            part += halo * (1 - part)

    def _room(self, out, lay):
        """The light the room gives the firebox (linear, as white light; `out`: the box's part in the frame): what
        most of it shows is ALBEDO of it (its median brightness, in its opening if it has one)."""
        luma = out[::4, ::4] @ LUMA
        inside = lay["inside"][::4, ::4] > 0.5
        level = max(float(np.median(luma[inside])), 0.0) if inside.any() else 0.0
        return max(level ** 2.2, ROOM_MIN) / ALBEDO

    def _wall_windows(self):
        """For each of the walls' lights: its weights across the grid's columns (its light: the flames' in a band of
        columns round it, all the way up; their glow times box heights)."""
        if self._wall_w is None:
            gw, gh = GRID
            X, _ = self.walls.screen(self.walls.lights)
            cols = (0.5 + X / (self.BOX[0] / self.BOX[1])) * gw
            sigma = max(2.0, 0.6 * float(np.diff(cols).mean())) if len(cols) > 1 else 0.1 * gw
            wt = np.exp(-(((np.arange(gw) + 0.5)[None, :] - cols[:, None]) / sigma) ** 2 / 2)
            self._wall_w = (wt / wt.sum(axis=1, keepdims=True) / gh).astype(np.float32)
        return self._wall_w

    def _walls(self, out, lay, gflame, room):
        """The firebox's walls drawn on `out` (the box's part in the frame, changed in place) inside its opening, lit
        by the room (its light inside the fireplace as the picture's inside says, or, if that is darker, a share of
        what lights the wall round the opening) and by the fire: each of their lights as bright as the flames round
        it now (`gflame`: on the grid), and a little of it everywhere (from the walls it lights). Returns that room
        light, and how bright the fire's light is over `out` (linear, as white light; its colour: light_colour)."""
        if "round" in lay:
            near = lay["round"][::4, ::4] > 0.5
            if near.any():
                level = max(float(np.median((out[::4, ::4] @ LUMA)[near])), 0.0)
                room = max(room, level ** 2.2 / WALL_ALBEDO * INSIDE)
        walls = lay["walls"]
        power = self._wall_windows() @ gflame.sum(axis=0) * np.float32(WALL_LIGHT * self.light_k)
        lit = walls["room"] * np.float32(BOUNCE * float(power.mean()))
        for e, g in zip(power, walls["fire"]):
            lit += e * g
        light = lit[..., None] * self.light_colour
        light += walls["room"][..., None] * np.float32(room)
        light *= walls["albedo"]
        np.clip(light, 0.0, 1.0, out=light)
        np.power(light, np.float32(1 / 2.2), out=light)
        a = lay["inside"][..., None]
        out += a * (light - out)
        return room, lit

    def _light(self, out, lay, gflame, room):
        """The flames' light on the firebox round them (`out`: the box's part in the frame, changed in place): their
        glow on the grid (`gflame`) spread as a fire's light falls off (SPREAD), on what is there as on surfaces
        of its colours under the room's light (`room`: as _room says; the picture's colours over it: bricks lit
        brick red, their joints dark), so lit, not hazy. Returns how bright that light is over `out` (linear, as
        white light; its colour: light_colour)."""
        gw, gh = GRID
        given = gflame.reshape(gh // 4, 4, gw // 4, 4).mean(axis=(1, 3))
        spread = sum(share * blur(given, sigma * gh / 4) for sigma, share in SPREAD)
        spread *= LIGHT_ON * self.light_k
        lit = np.asarray(Image.fromarray(np.ascontiguousarray(spread, np.float32), "F").resize(
            lay["size"], Image.Resampling.BILINEAR, box=tuple(v / 4 for v in lay["src"])), np.float32)
        lit = lit * lay["edge"]
        lin = np.power(np.maximum(out, 0.0), np.float32(2.2))
        # (a surface no whiter or more coloured than ALBEDO_MAX, softly; none darker than soot)
        a = np.maximum(np.maximum(lin[..., 0], lin[..., 1]), lin[..., 2])
        a *= np.float32(1 / (room * ALBEDO_MAX))
        a *= a
        a *= a
        a += 1
        np.sqrt(a, out=a)
        np.sqrt(a, out=a)
        np.multiply(a, np.float32(room), out=a)
        albedo = lin / a[..., None]
        np.maximum(albedo, ALBEDO_MIN, out=albedo)
        albedo *= lit[..., None]
        albedo *= self.light_colour
        lin += albedo
        np.power(lin, np.float32(1 / 2.2), out=out)
        return lit

    def _halo(self, flame, src, size):
        """The glow round the flames (a camera's): the flames' colours on the grid, blurred on a grid 4 times
        coarser (a wide blur: the same), enlarged to `size` over the part `src` of the grid."""
        gw, gh = GRID
        small = palette(KNEE * np.tanh(flame / KNEE), self.warmth).reshape(gh // 4, 4, gw // 4, 4, 3).mean(axis=(1, 3))
        small = blur(small, HALO_R * gh / 4) * (HALO * 255.0)
        img = Image.fromarray((np.clip(small, 0, 255) + 0.5).astype(np.uint8)).resize(
            size, Image.Resampling.BILINEAR, box=tuple(v / 4 for v in src))
        return np.asarray(img, np.float32) * (1 / 255.0)

    def _sparks(self, t, at, lay, x0, y0, w, h):
        """The sparks flying up now over the box's part in the frame (`at`: its top left corner; colours, 0..1), or
        None: each a little streak (as a camera sees a spark), from the top of the logs (or the bed) up, wavering,
        fading out."""
        sp = self.sparks
        if not len(sp["born"]) or self.p["amount"] <= 0:
            return None
        sec = (t % self.P) / self.fps
        age = np.mod(sec - sp["born"] * self.loop_s, self.loop_s)
        live = np.flatnonzero(age < sp["life"])
        if not len(live):
            return None
        width, height = lay["size"]
        acc = np.zeros((height, width), np.float32)
        if self.pile is not None:
            start = max(0.0, 0.7 * self.pile.extent()[3]) + MARK * self.bed * w / h / 2
        else:
            start = MARK * self.bed * 4 / 3 * 0.6
        sigma = max(0.6, SPARK_R * h)
        rad = int(math.ceil(3 * sigma))
        offs = np.arange(-rad, rad + 1, dtype=np.float32)
        for i in live:
            a, life = float(age[i]), float(sp["life"][i])
            bright = float(sp["hot"][i]) * (1 - (a / life) ** 2) * min(1.0, a / 0.06)
            for k in range(3):  # (a streak: where it was over a share of the frame's time, to where it is)
                b = max(0.0, a - k * 0.18 / self.fps)
                up = start + float(sp["rise"][i]) * (b / life) ** 0.75
                across = float(sp["x"][i]) * self.bed + float(sp["drift"][i]) * b + float(sp["wob"][i]) * math.sin(
                    TAU * float(sp["hz"][i]) * b + float(sp["ph"][i]))
                cx, cy = x0 + w * (0.5 + across) - at[0], y0 + h * (1 - up) - at[1]  # (across: box widths)
                ix, iy = int(round(cx)), int(round(cy))
                if ix + rad < 0 or iy + rad < 0 or ix - rad >= width or iy - rad >= height:
                    continue
                gx = np.exp(-((offs + ix - cx) / sigma) ** 2 / 2)
                gy = np.exp(-((offs + iy - cy) / sigma) ** 2 / 2)
                c0, c1, r0, r1 = max(0, ix - rad), min(width, ix + rad + 1), max(0, iy - rad), min(height, iy + rad + 1)
                acc[r0:r1, c0:c1] += (bright / 3) * np.outer(gy[r0 - iy + rad:r1 - iy + rad],
                                                            gx[c0 - ix + rad:c1 - ix + rad])
        acc *= lay["edge"]
        return self._lut.take((np.minimum(acc * 2.2, LUT_TOP) * ((len(self._lut) - 1) / LUT_TOP)).astype(np.int32),
                              axis=0)

    def draw(self, t, canvas, mask):
        """Nothing laid over the picture: apply() changes it."""

    def apply(self, t, picture, mask, gain, kept=None):
        """The fire on `picture` (float32 RGB 0..1, changed in place) in its box, `gain` (0..1, a fade) times as
        strongly, where `mask` (0..1, the clip's area; None: everywhere) says. Turned (on a sheet): `kept` says how
        much of each pixel it keeps (the least that makes its change a layer laid over the picture: kept_of())."""
        if gain <= 0:
            return
        x0, y0, w, h = self.box()
        r = self._rect(x0, y0, x0 + w, y0 + h)
        if r is None:
            return
        rx0, ry0, rx1, ry1 = r
        if kept is not None:
            before = picture[ry0:ry1, rx0:rx1].copy()
            self._apply(t, picture, mask, gain, r, x0, y0, w, h)
            kept[ry0:ry1, rx0:rx1] = kept_of(before, picture[ry0:ry1, rx0:rx1])
            return
        self._apply(t, picture, mask, gain, r, x0, y0, w, h)

    def _apply(self, t, picture, mask, gain, r, x0, y0, w, h):
        rx0, ry0, rx1, ry1 = r
        lay = self._layout(r, x0, y0, w, h)
        src, size = lay["src"], lay["size"]
        orig = picture[ry0:ry1, rx0:rx1]
        opening = lay.get("frame")
        out = orig if mask is None and gain >= 1 and opening is None else orig.copy()
        fire = self.frames()
        glow, level = fire.frame(t)
        level = min(max(level / fire.mean, 0.5), 1.5)  # (how bright the fire is now: 1 as on average)

        def big(a):  # a grid field over the part of the box in the frame
            p = np.pad(a, 1, mode="edge")  # (softened a little first: no stairs along its edges)
            a = (p[:-2] + 2 * p[1:-1] + p[2:]) * 0.25
            a = (a[:, :-2] + 2 * a[:, 1:-1] + a[:, 2:]) * 0.25
            return np.array(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize(
                size, Image.Resampling.BICUBIC, box=src), np.float32)  # (a copy: changed in place)

        # the old fire put out: its bright places, warm or burnt out white, over the bed and above it, become a dim
        # glow of their own colours (blurred: no still flame's edges left under the new flames)
        if self.cover_k > 0:
            luma = out @ LUMA
            hot = _smooth((luma - 0.2) / 0.35) * np.maximum(_smooth((out[..., 0] - out[..., 2] - 0.08) / 0.25),
                                                            _smooth((luma - 0.6) / 0.25))
            hot = np.minimum(_soft(hot, max(1.0, 0.012 * h)) * 1.5, 1.0)
            put_out = (self.cover_k * hot * lay["zone"])[..., None]
            out += put_out * (_soft(out, max(1.0, 0.04 * h)) * DIM - out)
        # the flames and the embers under them on the grid, as bright as they are drawn
        gflame = np.maximum(glow - CUT, 0.0) * (self.bright_k / (1 - CUT)) * self._grid_shown()
        embers = None
        if self.embers_k > 0:
            field, phase = self._ember_field()
            tau = self.tau(t)
            breathe = sum(np.sin(TAU * m * tau + ph + 5 * phase) for m, ph in zip(self.ember_m, self.ember_ph)) / 3
            embers = field * ((0.8 + 0.2 * breathe) * (0.7 * self.embers_k * (0.9 + 0.1 * level)))
            gflame += embers
        # the room's light on the firebox; the flames' light on it round them: on its walls drawn (if chosen), or
        # on the surfaces the picture shows there (lit, their colours kept: not a haze)
        room = self._room(out, lay)
        if "walls" in lay:
            room, lit = self._walls(out, lay, gflame, room)
        else:
            lit = self._light(out, lay, gflame, room) if self.light_k > 0 else None
        hide = None
        if "logs" in lay:  # the logs (opaque) on their bed of coals, the flames behind those in front of the fire
            self._logs(t, out, lay, glow, level, room, lit)
            px0, py0, px1, py1 = lay["pile"]
            hide = (slice(py0, py1), slice(px0, px1)), lay["logs"]["alpha"]
        # the flames and the embers under them (their brightness on the grid, then over the box: crisp edges)
        flame = big(glow)
        flame -= CUT
        np.maximum(flame, 0.0, out=flame)
        flame *= self.bright_k / (1 - CUT)
        np.power(flame, PUNCH, out=flame)  # (their faint parts fainter, their hearts brighter: as a camera sees
        # a fire, bright in a lit firebox; not burnt out white)
        flame *= PUNCH_K / KNEE
        np.tanh(flame, out=flame)
        flame *= KNEE
        flame *= lay["shown"]
        if hide is not None:  # (over a log in front of the fire none of them; over one behind it only the brighter
            # tongues: no haze over it)
            part = flame[hide[0]]
            hidden = 1 - BEHIND * _smooth((part - 0.5) / 0.7)
            part *= 1 - hide[1] * (hidden + (1 - hidden) * lay["logs"]["front"])
        if embers is not None and hide is None:  # (with logs, their bed's coals glow instead)
            flame += big(embers) * lay["edge"]
        fire = self._lut.take((np.minimum(flame, LUT_TOP) * ((len(self._lut) - 1) / LUT_TOP)).astype(np.int32),
                              axis=0)  # (its colours)
        halo = self._halo(gflame, src, size)
        halo *= lay["edge"][..., None]
        if hide is not None:  # (less of it over the logs: not a haze over them)
            halo[hide[0]] *= 1 - 0.8 * hide[1][..., None]
        # laid with Screen: 1 - (1 - picture) (1 - flames) (1 - their glow) (1 - sparks)
        np.subtract(1.0, fire, out=fire)
        np.subtract(1.0, halo, out=halo)
        fire *= halo
        sparks = self._sparks(t, (rx0, ry0), lay, x0, y0, w, h)
        if sparks is not None:
            fire *= 1.0 - sparks
        np.subtract(1.0, out, out=out)
        out *= fire
        np.subtract(1.0, out, out=out)
        if out is not orig:
            k = gain if mask is None else gain * mask[ry0:ry1, rx0:rx1, None]
            if opening is not None:  # (only inside the fireplace's opening)
                k = k * opening[..., None]
            orig += k * (out - orig)
