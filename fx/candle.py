"""A candle flame (a placed lens effect): brings a flame of the video back to life, or draws a new one over it.

The box (Effect.BOX, box()) goes over a flame: the flame's base (the wick's tip) near its bottom middle (ANCHOR),
the flame above it, room for its halo all round. Two ways ("mode"):
- "alive": the flame already in the picture sways and stretches, its brightness breathes, now and then it flutters,
  and its light flickers on what is around it. Only the flame's own light moves: the picture under the flame is
  filled in from what is round it (a clean plate), what the flame adds on top of that is warped (anchored at its
  base, more toward its tip) and laid back on the clean plate. So what is behind or beside it (a frame, bricks, a
  window) stays still.
- "new": the old flame's light is taken away (it is filled in softly from what is round it, with the grain of what
  is round it; what is darker, the wick, a dark frame behind, stays), and a new flame is drawn: a faint blue base, a
  dim orange cone over the wick, a white-yellow core, a warm orange-yellow body, a soft glow; it moves the same way
  and lights its surroundings.

apply() changes the picture (the export and the exact frames); draw() is what the page's live preview lays over
the picture with Screen: the halo, and the new flame. Every size is relative to the box, so the box alone (the
preview clip) and the whole frame look the same; only the light on the surroundings reaches out of the box.

Like every effect it loops: every motion is a sum of sines with a whole number of cycles per loop (none faster than
a quarter of the frame rate, so none flickers frame by frame), and the flutters are bursts at fixed times of the
loop with periodic envelopes, so frame P is frame 0 again.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .effects import Effect, _choice, _lerp, _num, _place

DEFAULTS = {"size": 50, "brightness": 50, "flicker": 35, "sway": 25, "flutter": 30, "speed": 50, "halo": 40,
            "light": 45, "reach": 40, "warmth": 50}
LOOKS = {
    "still": {},
    "breeze": {"flicker": 60, "sway": 65, "flutter": 70, "speed": 60, "light": 50},
    "soft": {"size": 38, "brightness": 40, "flicker": 20, "sway": 15, "flutter": 10, "speed": 40, "halo": 30,
             "light": 30, "reach": 30, "warmth": 40},
    "warm": {"size": 62, "brightness": 70, "flicker": 40, "sway": 30, "halo": 60, "light": 60, "reach": 60,
             "warmth": 65},
}

TAU = 2 * math.pi
BASE = 0.78  # the flame's base (the wick's tip) from the box's top, box heights
FLAME = 0.40  # the flame's height (base to tip) at size 50, box heights
HALF_W = 0.165  # the flame's half width where widest, flame heights
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
FLUTTERS = 48  # flutter slots with fixed random draws
FINE = 200  # pixels a flame height, at most, the new flame is drawn at (then enlarged)
WORK = 40.0  # pixels a flame height, at most, the old flame is found and filled in at
FILL_STEPS = 40  # smoothing steps of the fill over the old flame (each over every cell twice)
FAINT = 1.5 / 255  # light the old flame adds that is too faint to move (the grain of what is behind it)

SCHEMA = {"label": "Lửa nến", "params": [
    _choice("look", "Kiểu", "still", [("still", "Nến trong phòng (lặng gió)"), ("breeze", "Nến gần cửa sổ (gió nhẹ)"),
                                     ("soft", "Lửa nhỏ dịu"), ("warm", "Lửa to, sáng ấm")],
            presets={k: {**DEFAULTS, **v} for k, v in LOOKS.items()}, group="Kiểu"),
    _choice("mode", "Cách sửa", "alive", [("alive", "Làm sống ngọn lửa có sẵn"), ("new", "Vẽ ngọn lửa mới")],
            group="Kiểu"),
    _num("size", "Cỡ ngọn lửa", 50, group="Ngọn lửa"),
    _num("brightness", "Độ sáng ngọn lửa", 50, group="Ngọn lửa"),
    _num("cover", "Che ngọn lửa cũ (khi vẽ ngọn lửa mới)", 100, group="Ngọn lửa"),
    _num("flicker", "Lung linh", 35, group="Lung linh"),
    _num("sway", "Lay động", 25, group="Lung linh"),
    _num("flutter", "Thỉnh thoảng chao nhanh", 30, group="Lung linh"),
    _num("speed", "Nhịp (chậm ← → nhanh)", 50, group="Lung linh"),
    _num("halo", "Quầng sáng", 40, group="Ánh sáng"),
    _num("light", "Ánh nến hắt lên xung quanh", 45, group="Ánh sáng"),
    _num("reach", "Ánh sáng lan xa", 40, group="Ánh sáng"),
    _num("warmth", "Màu lửa (vàng ← → cam)", 50, group="Màu"),
    *_place(13)],
    # where the flame's base and tip go in the box (fractions of its width and height), for marks on the page's box:
    # the flame of the video is as tall as the effect's flame at size 50 (FLAME), with room for its halo above
    "anchor": [0.5, BASE], "anchor_hint": "Đặt chân ngọn lửa (đầu bấc) vào vạch này", "anchor_label": "đầu bấc",
    "tip": [0.5, BASE - FLAME], "tip_hint": "Kéo góc khung cho đỉnh ngọn lửa trong video chạm vạch này",
    "tip_label": "đỉnh lửa",
    "note": "Sửa nến bị lỗi: (1) nếu lửa trong video giật, méo hay biến mất ở chỗ loop, bấm V1 Video gốc rồi Vẽ vùng "
            "đứng yên, tô lên ngọn nến (chọn khung hình có lửa đẹp); (2) kéo khung trắng cho vạch vàng dưới (đầu bấc) "
            "nằm ở chân ngọn lửa, rồi kéo góc khung (vạch dưới đứng yên) cho vạch vàng trên (đỉnh lửa) chạm đỉnh ngọn "
            "lửa: phía trên còn trống để có quầng sáng; (3) Cách sửa: Làm sống ngọn lửa có sẵn, hoặc Vẽ ngọn lửa mới nếu lửa cũ hỏng "
            "nặng. Bấm Suốt video để lửa sống cả video. Mỗi ngọn nến một clip Lửa nến."}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _ramp(x, a, b):
    """0 below a, 1 above b, smooth between."""
    return _smooth((x - a) / (b - a))


def _sample(img, x, y):
    """`img` (h x w x 3) at float positions x, y (2-D; pixel i's centre at i + 0.5), clamped to its edges."""
    h, w = img.shape[:2]
    if w < 2 or h < 2:
        img = np.pad(img, ((0, int(h < 2)), (0, int(w < 2)), (0, 0)), mode="edge")
        h, w = img.shape[:2]
    x = np.clip(x - 0.5, 0.0, w - 1.0)
    y = np.clip(y - 0.5, 0.0, h - 1.0)
    xi = np.minimum(x.astype(np.int32), w - 2)
    yi = np.minimum(y.astype(np.int32), h - 2)
    fx, fy = x - xi, y - yi
    gx, gy = 1 - fx, 1 - fy
    i = yi * w + xi
    out = np.empty(x.shape + (3,), np.float32)
    for c in range(3):  # (a channel at a time, from a flat copy: much faster than indexing h x w x 3)
        p = np.ascontiguousarray(img[..., c]).ravel()
        top = p.take(i) * gx  # (weights written out in full: a pixel read at its centre is exactly itself)
        top += p.take(i + 1) * fx
        bottom = p.take(i + w) * gx
        bottom += p.take(i + w + 1) * fx
        top *= gy
        bottom *= fy
        bottom += top
        out[..., c] = bottom
    return out


def _gauss(n, sigma):
    """An n x n matrix that blurs along one axis (unnormalised: used for ratios)."""
    i = np.arange(n, dtype=np.float32)
    return np.exp(-0.5 * ((i[:, None] - i[None, :]) / max(sigma, 0.3)) ** 2).astype(np.float32)


def _blur(a, sigma_y, sigma_x=None):
    """A 2-D array blurred (normalised Gaussian, nothing comes in from beyond its edges)."""
    gy, gx = _gauss(a.shape[0], sigma_y), _gauss(a.shape[1], sigma_y if sigma_x is None else sigma_x)
    return (gy / gy.sum(axis=1, keepdims=True)) @ a @ (gx / gx.sum(axis=1, keepdims=True)).T


def _fine(a):
    """What is finer than about two pixels in a 2-D array (the grain): it minus a small blur."""
    p = np.pad(a, 1, mode="edge")
    b = (p[:-2] + 2 * p[1:-1] + p[2:]) * 0.25
    b = (b[:, :-2] + 2 * b[:, 1:-1] + b[:, 2:]) * 0.25
    return a - b


def _resize(a, w, h):
    """A float array (h0 x w0, or with channels) resized to w x h (bilinear)."""
    if a.ndim == 3:
        return np.dstack([_resize(a[..., c], w, h) for c in range(a.shape[2])])
    if a.shape == (h, w):
        return a
    return np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize(
        (w, h), Image.Resampling.BILINEAR), np.float32)


def _grow(a, n):
    """A 2-D array's largest value within n cells."""
    if n < 1:
        return a
    return np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").filter(
        ImageFilter.MaxFilter(2 * n + 1)), np.float32)


def _connected(seed, a):
    """What of `a` (0..1) is joined to `seed` (a fuzzy flood fill: each cell keeps the best path's weakest step)."""
    out = np.minimum(seed, a)
    for _ in range(4 * sum(a.shape)):
        p = np.pad(out, 1)
        nxt = np.minimum(np.maximum(np.maximum(np.maximum(p[:-2, 1:-1], p[2:, 1:-1]),
                                               np.maximum(p[1:-1, :-2], p[1:-1, 2:])), out), a)
        if np.array_equal(nxt, out):
            break
        out = nxt
    return out


def _membrane(fill, hole, up, down, count, steps=None, omega=1.9):
    """`fill` (h x w x 3) over `hole` (h x w, bool) smoothed into a membrane held by what is round it (each cell the
    mean of its neighbours; `up`, `down` (h x 1): whether a row's upper, lower neighbours count): red-black
    over-relaxation."""
    rows, cols = np.flatnonzero(hole.any(axis=1)), np.flatnonzero(hole.any(axis=0))
    if not len(rows):
        return fill
    # (only round the hole: a rect one cell larger, whose edge cells hold it)
    r0, r1, c0, c1 = max(0, rows[0] - 1), min(hole.shape[0], rows[-1] + 2), max(0, cols[0] - 1), \
        min(hole.shape[1], cols[-1] + 2)
    out = fill.copy()
    part = _membrane_in(out[r0:r1, c0:c1], hole[r0:r1, c0:c1], up[r0:r1], down[r0:r1], count[r0:r1],
                        FILL_STEPS if steps is None else steps, omega)
    out[r0:r1, c0:c1] = part
    return out


def _membrane_in(fill, hole, up, down, count, steps, omega):
    h, w = hole.shape
    p = np.pad(fill, ((1, 1), (1, 1), (0, 0)), mode="edge")
    c = p[1:-1, 1:-1]
    up, down, count = up[..., None], down[..., None], count[..., None]
    even = np.add.outer(np.arange(h), np.arange(w)) % 2 == 0
    sets = [(hole & even)[..., None], (hole & ~even)[..., None]]
    for _ in range(steps):
        for at in sets:
            step = (p[:-2, 1:-1] * up + p[2:, 1:-1] * down + p[1:-1, :-2] + p[1:-1, 2:]) / count - c
            step *= omega
            step += c
            np.copyto(c, step, where=at)
            p[0], p[-1] = p[1], p[-2]  # (the edges: as the cells next to them)
            p[:, 0], p[:, -1] = p[:, 1], p[:, -2]
    return c.copy()


class _Waves:
    """A smooth wandering signal: sines at about `rates` Hz (whole cycles per loop), unit RMS."""

    def __init__(self, fx, rng, rates, amps):
        rates = np.asarray(rates, np.float64) * (0.85 + 0.3 * rng.random(len(rates)))
        self.m = fx.rate_cycles(rates * fx.speed_k)
        self.ph = rng.random(len(rates)) * TAU
        a = np.asarray(amps, np.float64)
        self.a = a / math.sqrt(float((a * a).sum()) / 2)

    def at(self, tau):
        return float(np.dot(self.a, np.sin(TAU * self.m * tau + self.ph)))


class Candle(Effect):
    BOX = (0.75, 1.0)
    ANCHOR = (0.5, BASE)
    blend = "screen"
    lens = True
    UNDER = True  # its preview clip: the flame there, alive (made from the picture under the box)
    VERSION = 3  # (1: a first sketch, a glow only; 2: the first flame)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        self.new = p["mode"] == "new"
        self.speed_k = 2.0 ** ((p["speed"] - 50) / 50)
        self.fk, self.sk = p["flicker"] / 100, p["sway"] / 100
        # the new flame's size; the old one is only stretched a little (more would warp it out of shape)
        self.size_k = _lerp(0.7, 1.3, p["size"] / 100) if self.new else _lerp(0.8, 1.2, p["size"] / 100)
        self.bright_k = _lerp(0.7, 1.3, p["brightness"] / 100)
        self.halo_k = p["halo"] / 100
        self.light_k = p["light"] / 100
        self.reach = _lerp(0.9, 4.5, p["reach"] / 100)  # light radius, flame heights
        self.cover_k = p["cover"] / 100
        warm = p["warmth"] / 100
        shift = 2 * warm - 1  # the old flame made warmer or cooler (50: as it is)
        self.tint = np.array([1.0, 1 - 0.1 * shift, 1 - 0.25 * shift], np.float32)
        self.c_core = _lerp(np.array([1.0, 0.98, 0.9]), np.array([1.0, 0.93, 0.74]), warm).astype(np.float32)
        self.c_body = _lerp(np.array([1.0, 0.86, 0.5]), np.array([1.0, 0.7, 0.3]), warm).astype(np.float32)
        self.c_edge = _lerp(np.array([1.0, 0.62, 0.24]), np.array([1.0, 0.44, 0.1]), warm).astype(np.float32)
        self.c_halo = _lerp(np.array([1.0, 0.74, 0.45]), np.array([1.0, 0.55, 0.24]), warm).astype(np.float32)
        self.c_light = _lerp(np.array([1.0, 0.86, 0.68]), np.array([1.0, 0.7, 0.45]), warm).astype(np.float32)
        self.c_blue = np.array([0.32, 0.5, 1.0], np.float32)
        rng = np.random.default_rng([self.seed, 1103, 1])
        self.waves = [_Waves(self, rng, (0.23, 0.41, 0.67, 1.1, 1.7), (1, 0.8, 0.6, 0.45, 0.3)),  # slow
                      _Waves(self, rng, (2.9, 4.3, 6.1, 8.3), (1, 0.8, 0.6, 0.4)),  # fast
                      _Waves(self, rng, (0.11, 0.23, 0.41, 0.7), (1, 0.7, 0.5, 0.35)),  # lean
                      _Waves(self, rng, (2.3, 3.7, 5.3), (1, 0.7, 0.5))]  # quick side jitter
        self.m_rip = float(self.rate_cycles(3.2 * self.speed_k))
        # flutters: short bursts of a quick flicker at fixed times of the loop
        rng = np.random.default_rng([self.seed, 1103, 2])
        first, u = rng.random(), rng.random((FLUTTERS, 6))
        fl = p["flutter"] / 100
        self.fl = 0.6 + 0.4 * fl
        n = min(FLUTTERS, int(0.25 * fl * self.loop_s + first)) if fl > 0 else 0
        sigma = np.maximum((0.12 + 0.14 * u[:n, 1]) / self.speed_k, 2.5 / self.fps)  # seconds
        self.f_at = (np.arange(n) + 0.15 + 0.7 * u[:n, 0]) / max(n, 1)
        self.f_kappa = (self.loop_s / (TAU * sigma)) ** 2
        self.f_m = self.rate_cycles((4.0 + 2.0 * u[:n, 2]) * self.speed_k)
        self.f_py, self.f_px = u[:n, 3] * TAU, u[:n, 4] * TAU
        self.f_amp = 0.6 + 0.4 * u[:n, 5]
        self._plates = None  # the last clean plate (the same picture under the flame gives the same plate)
        self._grain = None

    def rate_cycles(self, rate_hz):
        """Whole cycles per loop near `rate_hz`, none faster than a quarter of the frame rate (faster than that,
        frames show it as a jitter, not as a flame moving): the rates are eased below that, not cut."""
        cap = self.fps / 4
        m = self.cycles(cap * np.tanh(np.asarray(rate_hz, np.float64) / cap))
        return np.minimum(m, max(1.0, math.floor(cap * self.loop_s)))

    # --- the motion ------------------------------------------------------------------------------------------

    def _motion(self, t):
        """(stretch, brightness, sway of the tip in flame heights, ripple, ripple phase, light on the room) at
        frame t."""
        tau = self.tau(t)
        slow, fast, lean, jit = (w.at(tau) for w in self.waves)
        burst = fy = fx = 0.0
        if len(self.f_at):
            env = self.f_amp * np.exp(self.f_kappa * (np.cos(TAU * (tau - self.f_at)) - 1.0))
            burst = min(1.0, float(env.sum()))
            fy = float(np.dot(env, np.sin(TAU * self.f_m * tau + self.f_py)))
            fx = float(np.dot(env, np.sin(TAU * self.f_m * tau + self.f_px)))
        fk, sk, fl = self.fk, self.sk, self.fl
        # a flutter mostly pinches and stretches the flame (its tip), and sways it a little
        calm_s, calm_b = fk * (0.07 * slow + 0.025 * fast), fk * (0.11 * slow + 0.045 * fast)
        flut_s, flut_b = fl * (0.09 * fy - 0.03 * burst), fl * (0.07 * fy - 0.06 * burst)
        sway = sk * (0.09 * lean + 0.02 * jit) + fl * 0.03 * fx
        ripple = fk * 0.03 + fl * 0.05 * burst
        # the light it throws: more as it brightens and grows; a quick flutter hardly darkens the room
        room = fk * (0.15 * slow + 0.02 * fast) + 0.4 * (flut_b + 0.6 * flut_s)
        return (max(0.6, 1 + calm_s + flut_s), max(0.3, 1 + calm_b + flut_b), sway, ripple,
                TAU * float(self.m_rip * tau % 1.0), room)

    def _geom(self):
        """(x0, y0, w, h) of the box, the flame's base (cx, ay) and its height there (pixels)."""
        x0, y0, w, h = self.box()
        return x0, y0, w, h, x0 + w * self.ANCHOR[0], y0 + h * self.ANCHOR[1], FLAME * h

    def _rect(self, x0, y0, x1, y1):
        """Pixel rows and columns covering x0..x1, y0..y1, within the frame (or None)."""
        r = (max(0, math.floor(x0)), max(0, math.floor(y0)), min(self.w, math.ceil(x1)), min(self.h, math.ceil(y1)))
        return r if r[0] < r[2] and r[1] < r[3] else None

    def _centre(self, u, hf, sway, ripple, phase):
        """The flame's centre line: its sideways offset (pixels) at heights u (flame heights above its base)."""
        uc = np.clip(u, 0.0, 1.2)
        return hf * (sway * uc ** 1.7 + ripple * uc ** 1.5 * np.sin(TAU * 1.1 * uc - phase))

    # --- what it lays over the picture (light), and the new flame --------------------------------------------

    def _lights(self, mo, g, r):
        """The light laid over the picture in rect r (premultiplied RGB, Screen): the halo, and the new flame with
        its glow; and the new flame's cover (it hides what is behind it) or None."""
        stretch, bright, sway, ripple, phase = mo[:5]
        x0, y0, w, h, cx, ay, hf = g
        rx0, ry0, rx1, ry1 = r
        xs = np.arange(rx0, rx1, dtype=np.float32) + 0.5
        ys = np.arange(ry0, ry1, dtype=np.float32) + 0.5
        out = np.zeros((ry1 - ry0, rx1 - rx0, 3), np.float32)
        hs = hf * self.size_k * stretch  # the flame's height now
        level = self.bright_k * bright
        # the halo: a wide soft glow round the flame's middle, moving with it, nothing at the box's edges
        if self.halo_k > 0:
            hx, hy = cx + 0.4 * sway * hs, ay - 0.48 * hs
            gx = np.exp(-0.5 * ((xs - hx) / (0.125 * h)) ** 2) * _ramp(xs - x0, 0, 0.16 * w) * _ramp(
                x0 + w - xs, 0, 0.16 * w)
            gy = np.exp(-0.5 * ((ys - hy) / (0.165 * h)) ** 2) * _ramp(ys - y0, 0, 0.12 * h) * _ramp(
                y0 + h - ys, 0, 0.12 * h)
            amp = 0.42 * self.halo_k * level
            glow = np.outer(gy, gx).astype(np.float32)
            for c in range(3):
                out[..., c] += glow * (amp * self.c_halo[c])
        if not self.new:
            return out, None
        # the new flame, on the rows and columns it can reach
        fr = self._rect(cx - 0.6 * hf * self.size_k, ay - 1.5 * hs, cx + 0.6 * hf * self.size_k, ay + 0.3 * hs)
        cover = np.zeros(out.shape[:2], np.float32)
        if fr is None:
            return out, cover
        fx0, fy0, fx1, fy1 = max(fr[0], rx0), max(fr[1], ry0), min(fr[2], rx1), min(fr[3], ry1)
        if fx0 >= fx1 or fy0 >= fy1:
            return out, cover
        fw, fh = fx1 - fx0, fy1 - fy0
        q = max(1.0, hs / FINE)  # a big flame is drawn smaller, then enlarged (it is soft at that size)
        lw, lh = max(2, round(fw / q)), max(2, round(fh / q))
        if lw < fw and lh < fh:
            fxs = fx0 + (np.arange(lw, dtype=np.float32) + 0.5) * (fw / lw)
            fys = fy0 + (np.arange(lh, dtype=np.float32) + 0.5) * (fh / lh)
            rgb, a = self._flame(fxs, fys, cx, ay, hf, hs, mo, level)
            rgb, a = _resize(rgb, fw, fh), _resize(a, fw, fh)
        else:
            rgb, a = self._flame(xs[fx0 - rx0:fx1 - rx0], ys[fy0 - ry0:fy1 - ry0], cx, ay, hf, hs, mo, level)
        out[fy0 - ry0:fy1 - ry0, fx0 - rx0:fx1 - rx0] += rgb
        cover[fy0 - ry0:fy1 - ry0, fx0 - rx0:fx1 - rx0] = a
        return out, cover

    def _flame(self, xs, ys, cx, ay, hf, hs, mo, level):
        """The new flame over columns xs and rows ys (pixel centres): its light (premultiplied RGB, glow
        included) and its cover."""
        stretch, bright, sway, ripple, phase = mo[:5]
        u = (ay - ys) / hs  # height above the base, 0..1 (the tip)
        uc = np.clip(u, 0.0, 1.0)
        um = 0.5 / 1.6  # where it is widest
        prof = (uc / um) ** 0.65 * ((1 - uc) / (1 - um)) ** 1.1
        r = HALF_W * hf * self.size_k / math.sqrt(stretch) * prof * (
            1 + 0.6 * ripple * np.sin(TAU * 1.7 * uc - phase - 1.0))
        xc = cx + self._centre(u, hs, sway, ripple, phase)
        dx = xs[None, :] - xc[:, None]
        adx = np.abs(dx)
        # its edge: soft, softer up to its tip, which fades out (a flame has no sharp tip)
        soft = (max(0.7, 0.03 * hf * self.size_k) * (1 + 2 * uc))[:, None]
        ends = (_ramp(u, -0.02, 0.03) * (1 - _ramp(u, 0.8, 1.03)))[:, None]
        cov = np.clip(0.5 + (r[:, None] - adx) / soft, 0.0, 1.0) * ends
        dn = adx / np.maximum(r, 0.35)[:, None]
        # it burns yellow round a dim cone over the wick (it lets through some of the light round it) above a
        # faint blue cup at its bottom, seen most at its sides (looked at through more of it)
        start = 0.05 + 0.1 * dn * dn + 0.09 * (1 - _ramp(dn, 0.0, 0.42))
        lit = _ramp(u[:, None] - start, -0.04, 0.1)
        # colour: orange-yellow at the edge, golden body, white-yellow core low in the middle, orange to the tip
        col = self.c_body + (self.c_edge - self.c_body) * _ramp(dn, 0.5, 1.0)[..., None]
        core = (1 - _ramp(dn, 0.25, 0.7)) * (_ramp(u, 0.2, 0.34) * (1 - _ramp(u, 0.5, 0.85)))[:, None]
        col += (self.c_core - col) * core[..., None]
        tip = _ramp(u, 0.6, 1.0)[:, None]
        col += (self.c_edge - col) * (0.6 * tip)[..., None]
        col += (self.c_edge - col) * (0.7 * (1 - lit))[..., None]  # (the dim cone glows orange)
        low = _ramp(u, 0.02, 0.14)[:, None]  # (the dim cone's light: none in the blue cup)
        lum = (0.85 + 0.6 * core) * (1 - 0.3 * tip) * (0.35 * low + 0.65 * lit)
        lum *= cov * level
        rgb = col * lum[..., None]
        blue = cov * (1 - lit) * _ramp(dn, 0.5, 0.9) * (1 - _ramp(u, 0.04, 0.1))[:, None]
        rgb += blue[..., None] * (0.08 * level * self.c_blue)
        # a soft glow just round the flame (as a camera sees a bright flame), wider over its upper half
        dv = (np.maximum(-u, 0) + 1.6 * np.maximum(u - 1, 0)) * hs
        dh = np.maximum(adx - r[:, None], 0)
        sig = 0.06 * hf * self.size_k * (0.8 + 0.7 * _ramp(u, 0.3, 1.0))[:, None]
        bloom = np.exp(-(dh * dh + (dv * dv)[:, None]) / (2 * sig * sig)) * (
            0.22 * level * (0.4 + 0.6 * _ramp(u, 0.0, 0.25)))[:, None]
        rgb += bloom[..., None] * self.c_body
        a = cov * lit * 0.9  # (a flame hides little of what is behind it, and nothing through its blue base)
        return rgb.astype(np.float32), a.astype(np.float32)

    def draw(self, t, canvas, mask):
        """The live preview: the halo, and the new flame (Screen, inside the box)."""
        g = self._geom()
        r = self._rect(g[0], g[1], g[0] + g[2], g[1] + g[3])
        if r is None:
            return
        light, _ = self._lights(self._motion(t), g, r)
        for c in range(3):
            unit = np.zeros(3, np.float32)
            unit[c] = 1.0
            canvas.light(r[0], r[1], light[..., c], unit, mask)

    # --- the picture -------------------------------------------------------------------------------------------

    def _plate(self, src, r, g):
        """The old flame found in the box's flame area (`src`: rect r of the frame) and filled in from what is
        round it: (the area's rect in the frame, what _find gives) or None. The same picture there gives the same
        plate (the last one is kept: a still area under the flame is found once)."""
        x0, y0, w, h, cx, ay, hf = g
        zr = self._rect(cx - 0.92 * hf, ay - 1.95 * hf, cx + 0.92 * hf, ay + 0.15 * hf)
        if zr is None:
            return None
        zx0, zy0, zx1, zy1 = max(zr[0], r[0]), max(zr[1], r[1]), min(zr[2], r[2]), min(zr[3], r[3])
        if zx0 >= zx1 or zy0 >= zy1:
            return None
        zone_rect = (zx0, zy0, zx1, zy1)
        pic = src[zy0 - r[1]:zy1 - r[1], zx0 - r[0]:zx1 - r[0]]
        last = self._plates
        if last is not None and last[0] == zone_rect and np.array_equal(last[1], pic):
            found = last[2]
        else:
            found = self._find(pic, zone_rect, g)
            self._plates = (zone_rect, pic.copy(), found)
        return None if found is None else (zone_rect, found)

    def _find(self, pic, zr, g):
        """The old flame in `pic` (the frame's rect zr) and the picture without it. In "new" anything bright in the
        flame's area (a flame bent aside, a blob above it): (how much of each pixel to fill in, the fill); in
        "alive" the flame joined to its base only: (what stays, the flame's own light)."""
        x0, y0, w, h, cx, ay, hf = g
        zx0, zy0, zx1, zy1 = zr
        zw, zh = zx1 - zx0, zy1 - zy0
        k = min(1.0, WORK / hf)  # worked on at about WORK pixels a flame height
        lw, lh = max(2, round(zw * k)), max(2, round(zh * k))
        small = _resize(pic, lw, lh)
        lhf = hf * lw / zw
        xs = (np.arange(lw, dtype=np.float32) + 0.5) * zw / lw + zx0
        ys = (np.arange(lh, dtype=np.float32) + 0.5) * zh / lh + zy0
        u = (ay - ys) / hf
        uc = np.clip(u, 0.0, 1.5)[:, None]
        adx = np.abs(xs - cx)[None, :] / hf
        # where the old flame may be: the box's flame area, narrow at the base (not the wax by the wick), up to
        # near the box's top
        wz = (0.3 + 0.42 * _ramp(u, 0.0, 0.6))[:, None]
        zone = (1 - _ramp(adx, wz - 0.1, wz)) * (_ramp(u, -0.03, 0.04) * (1 - _ramp(u, 1.72, 1.85)))[:, None]
        # the flame: brighter than the light round it, or among the brightest there (not its glow, which the
        # fill keeps); near its middle at the base, further aside higher up (around the base, or around where
        # the light of each row is: a flame bent aside)
        lum = small @ LUMA
        local = _blur(lum, 0.22 * lhf)  # (what is much darker, a dark frame beside the flame, counts little)
        dim = 1 - _ramp(local - lum, 0.04, 0.1)
        local = _blur(dim * lum, 0.22 * lhf) / (_blur(dim, 0.22 * lhf) + 1e-6)
        above = (lum - local) * zone
        wgt = np.maximum(above, 0) ** 2
        gr = _gauss(lh, 0.15 * lhf)
        gr /= gr.sum(axis=1, keepdims=True)
        rw, rx = gr @ wgt.sum(axis=1), gr @ (wgt @ xs)  # (each row's weight and centre, smoothed over rows)
        e = 0.05 * float(rw.max()) + 1e-9
        xr = ((rx + e * cx) / (rw + e))[:, None]
        lo, hi = 0.28 + 0.35 * uc, 0.4 + 0.45 * uc
        near = np.maximum(1 - _ramp(adx, lo, hi), 1 - _ramp(np.abs(xs[None, :] - xr) / hf, lo, hi))
        above *= near
        most = max(float(above.max()), 1e-3)
        inside = lum[zone > 0.5]
        if inside.size < 4 or (most < 0.04 and self.new):
            return None
        bg, top = float(np.percentile(inside, 25)), float(inside.max())
        span = max(top - bg, 0.02)
        on = float(_ramp(np.float32(most), 0.04, 0.1))
        sure = _ramp(lum, bg + 0.55 * span, bg + 0.8 * span) * near  # (among the brightest)
        strong = np.maximum(_ramp(above, 0.2 * most, 0.45 * most) * _ramp(lum, bg + 0.3 * span, bg + 0.5 * span),
                            sure) * zone * on
        # where the box says the flame is
        ox, oy = (xs - cx) / (0.24 * hf), (ys - (ay - 0.52 * hf)) / (0.58 * hf)
        core = (1 - _ramp(np.hypot(ox[None, :], oy[:, None]), 0.8, 1.0)).astype(np.float32)
        if self.new:
            # all of it: with what dimmer joins it (a thin tip, a bent tip, a fat flame's outer part), what is
            # bright in the column over it (a blob come loose above it), and always where the box says it is
            weak = _ramp(above, 0.06 * most, 0.2 * most) * _ramp(lum, bg + 0.1 * span, bg + 0.25 * span)
            column = 1 - _ramp(adx, 0.07 + 0.04 * uc, 0.12 + 0.06 * uc)
            flame = _connected(core, np.maximum(weak * zone * on, strong))
            flame = np.maximum(np.maximum(flame, strong * column), core)
        else:  # the flame joined to its base only (bright things beside it stay where they are)
            weak = _ramp(above, 0.12 * most, 0.3 * most) * _ramp(lum, bg + 0.15 * span, bg + 0.3 * span)
            flame = _connected(core, np.maximum(weak * zone * on, strong))
        flame = _grow(flame, max(1, round((0.12 if self.new else 0.08) * lhf)))
        body = flame
        # and the glow over its tip, which sways the most (nothing below its base; not what is bright and apart
        # from the flame, a light behind it)
        up = flame.copy()
        for i in range(1, max(1, round((0.15 if self.new else 0.3) * lhf)) + 1):
            np.maximum(up[:-i], flame[i:], out=up[:-i])
        apart = _grow(strong * (1 - flame), 1)
        flame = np.maximum(up * (1 - apart), flame) * _ramp(u, -0.03, 0.01)[:, None]
        # the fill: a smooth membrane over the hole held by a small blur of what is right round it, started from
        # a wider blur of what is round it; what is much darker than what is round it (a joint between bricks, a
        # dark frame or rim beside the flame) counts little; and nothing from below the base (the wax, lit
        # bright by the old flame, and the wick)
        hole = flame > 0.5
        keep = (~hole).astype(np.float32)
        ny, nx = _gauss(lh, 0.08 * lhf), _gauss(lw, 0.08 * lhf)
        for _ in range(3):
            close = ny @ (keep * lum) @ nx.T / (ny @ keep @ nx.T + 1e-6)
            keep = (~hole) * (1 - _ramp(close - lum, 0.015, 0.04)).astype(np.float32)
        gy, gx = _gauss(lh, 0.22 * lhf), _gauss(lw, 0.22 * lhf)
        above_base = keep * (u >= 0.03)[:, None]
        wide, close = gy @ above_base @ gx.T + 1e-6, ny @ keep @ nx.T + 1e-6
        fill = np.dstack([np.where(hole, gy @ (above_base * small[..., c]) @ gx.T / wide,
                                   ny @ (keep * small[..., c]) @ nx.T / close) for c in range(3)])
        floor = np.pad(u < 0.03, 1, mode="edge")[:, None].astype(np.float32)  # (rows at or below the base)
        n_up, n_down = 1 - floor[:-2], 1 - floor[2:]
        count = n_up + n_down + 2
        fill = _membrane(fill.astype(np.float32), hole, n_up, n_down, count)
        # all of the hole filled, softly round it
        soft = np.clip(_blur(_grow(hole.astype(np.float32), 2), 0.04 * lhf) * 1.15, 0, 1)
        # (nothing at the rect's edges: it may be cut by them)
        soft *= np.outer(_ramp(np.minimum(np.arange(lh), np.arange(lh)[::-1]), 0.0, 2.0),
                         _ramp(np.minimum(np.arange(lw), np.arange(lw)[::-1]), 0.0, 2.0))
        m = _resize(soft.astype(np.float32), zw, zh)
        fill = _resize(fill.astype(np.float32), zw, zh)
        ring = _resize(((_blur(flame, 0.2 * lhf) > 0.05) & (flame < 0.02)).astype(np.float32), zw, zh) > 0.5
        # the grain of what is round the hole (a still grain, the same every frame)
        fine = _fine(pic @ LUMA)[ring]
        if fine.size > 16:
            grain = 1.4826 * float(np.median(np.abs(fine - np.median(fine))))
            fill += (grain * self._noise(zw, zh))[..., None]
        if self.new:
            return m, fill
        # what moves: the flame's own light (what it adds to the fill), as it is where it is strong, smoothed
        # where it is no stronger than the texture round it (so the texture stays where it is)
        lit = np.maximum(pic - fill, 0.0) * m[..., None]
        lit = np.maximum(lit - FAINT, 0.0)
        body = _grow(body, max(1, round(0.05 * lhf)))  # (with all of its soft edge)
        out = 1 - body  # (smoothed from round the flame only: none of its own light spreads out)
        spread = _blur(out, 0.08 * lhf) + 1e-6
        lit_s = _resize(np.dstack([_blur(out * c, 0.08 * lhf) / spread
                                   for c in np.moveaxis(_resize(lit, lw, lh), 2, 0)]), zw, zh)
        fine = np.maximum(pic @ LUMA - _resize(_blur(lum, 0.06 * lhf).astype(np.float32), zw, zh), 0)[ring]
        tex = max(0.004, float(np.sqrt(np.mean(fine * fine)))) if fine.size > 16 else 0.004  # (what is brighter
        # than the fill: the darker texture never moves)
        sharp = np.maximum(_ramp(lit_s @ LUMA, 1.5 * tex, 4 * tex), _resize(_blur(body, 0.03 * lhf), zw, zh))
        lit = lit_s + sharp[..., None] * (lit - lit_s)
        return pic - lit, lit

    def _noise(self, w, h):
        """A fixed white grain of w x h, as strong as one whose part finer than two pixels (_fine) has a spread
        of 1."""
        if self._grain is None or self._grain.shape != (h, w):
            n = np.random.default_rng([self.seed, 1103, 3]).standard_normal((h, w)).astype(np.float32)
            self._grain = n / max(float(_fine(n).std()), 1e-6)
        return self._grain

    def _cover(self, out, r, g):
        """The old flame filled in from what is round it (in place in `out`, rect r of the frame)."""
        if self.cover_k <= 0:
            return
        found = self._plate(out, r, g)
        if found is None:
            return
        (zx0, zy0, zx1, zy1), (m, fill) = found
        pic = out[zy0 - r[1]:zy1 - r[1], zx0 - r[0]:zx1 - r[0]]
        # a flame only adds light: what is darker than the fill (the wick, a dark line behind) stays
        pic += (m * self.cover_k)[..., None] * np.minimum(fill - pic, 0.0)

    def _warp(self, src, r, g, mo):
        """The old flame alive: its light (what it adds to the picture filled in under it) read through a warp
        anchored at its base that sways and stretches it, its brightness breathing, to be laid back on the
        picture filled in under it. What is darker than that fill (texture, edges behind and beside it) does not
        move. Returns (where, in rect r: the flame's light there now, and moved) or None."""
        stretch, bright, sway, ripple, phase = mo[:5]
        x0, y0, w, h, cx, ay, hf = g
        found = self._plate(src, r, g)
        if found is None:
            return None
        (px0, py0, px1, py1), (rest, lit) = found  # (what stays, and the flame's own light)
        s = max(0.76, self.size_k * stretch)  # (so the warp never folds over: see `fade`)
        wr = self._rect(cx - 0.95 * hf, ay - 1.9 * hf, cx + 0.95 * hf, ay + 0.1 * hf)  # (inside the box)
        if wr is None:
            return None
        wx0, wy0, wx1, wy1 = max(wr[0], px0), max(wr[1], py0), min(wr[2], px1), min(wr[3], py1)
        if wx0 >= wx1 or wy0 >= wy1:
            return None
        xs = np.arange(wx0, wx1, dtype=np.float32) + 0.5
        ys = np.arange(wy0, wy1, dtype=np.float32) + 0.5
        u = (ay - ys) / hf
        # all of it up to past the tip, then less up to the box's top (its slope there, under 3, times
        # 1 / s - 1 stays under 1)
        fade = np.where(u < 0, 0.0, 1 - _ramp(u, 1.25, 1.9)).astype(np.float32)
        gv = np.clip(u, 0.0, 1.25) * fade
        side = (1 - _ramp(np.abs(xs - cx) / hf, 0.32, 0.95)).astype(np.float32)
        shift = self._centre(u, hf * s, sway, ripple, phase) * fade
        thin = (math.sqrt(s) - 1) * np.clip(u / 0.3, 0.0, 1.0) * fade  # taller is thinner
        xsrc = xs[None, :] + ((xs - cx)[None, :] * thin[:, None] - shift[:, None]) * side[None, :]
        ysrc = ys[:, None] + ((1 - 1 / s) * hf) * gv[:, None] * side[None, :]  # (read from lower: taller)
        lay = _sample(lit, xsrc - px0, ysrc - py0)
        at = slice(wy0 - py0, wy1 - py0), slice(wx0 - px0, wx1 - px0)
        # brightness: the flame's light breathes (not the wax below it); what the camera saw white had light to
        # spare (the flame, wax right by it): it stays white as the light dims
        weight = _ramp(u, -0.03, 0.03)[:, None]
        if bright != 1 or self.bright_k != 1 or (self.tint != 1).any():
            k = self.bright_k * (1 + 0.7 * (bright - 1))
            for c in range(3):
                mc = k * self.tint[c] - 1
                if mc == 0:
                    continue
                gain = weight * mc
                if mc < 0:
                    gain = gain * (1 - _ramp(rest[at + (c,)] + lay[..., c], 0.8, 1.0))
                lay[..., c] *= 1 + gain
        moved = lit.copy()
        moved[at] = lay
        return (slice(py0 - r[1], py1 - r[1]), slice(px0 - r[0], px1 - r[0])), lit, moved

    def _room(self, mo):
        """How much more light (or less: below 0) the flame throws on what is round it now."""
        return 0.3 * math.tanh(2.5 * self.light_k * mo[5] / 0.3)

    def _light_on(self, out, r, g, mo, white=None):
        """The flame's light flickering on what is round it: brighter and darker in turn as the flame brightens and
        grows, more near the flame (where most of the light is the flame's)."""
        stretch, bright, sway, ripple, phase = mo[:5]
        x0, y0, w, h, cx, ay, hf = g
        m = self._room(mo)
        if abs(m) < 1e-6:
            return
        rad = self.reach * hf
        lx, ly = cx + 0.35 * sway * hf, ay - 0.5 * hf * self.size_k * stretch
        xs = np.arange(r[0], r[2], dtype=np.float32) + 0.5
        ys = np.arange(r[1], r[3], dtype=np.float32) + 0.5
        share = np.outer(np.exp(-0.5 * ((ys - ly) / rad) ** 2), np.exp(-0.5 * ((xs - lx) / rad) ** 2)).astype(
            np.float32)
        for c in range(3):
            v = out[..., c]
            k = share * (m * self.c_light[c])
            if m < 0:  # (what the camera saw white stays white; `white`: (where, what the picture is there))
                k *= np.clip((1 - v) * 5, 0, 1)
                if white is not None:
                    at, pic = white
                    k[at] = share[at] * (m * self.c_light[c]) * np.clip((1 - pic[..., c]) * 5, 0, 1)
            k += 1
            v *= k

    def apply(self, t, picture, mask, gain):
        """The whole effect on `picture` (float32 RGB 0..1, changed in place) where `mask` (0..1, None:
        everywhere) says, `gain` (0..1, a fade) times as strongly."""
        if gain <= 0:
            return
        g = self._geom()
        x0, y0, w, h, cx, ay, hf = g
        mo = self._motion(t)
        reach = 3.2 * self.reach * hf if self.light_k > 0 else 0  # (the light's last 1/200 is left out)
        r = self._rect(min(x0, cx - reach), min(y0, ay - 0.6 * hf - reach), max(x0 + w, cx + reach),
                       max(y0 + h, ay - 0.4 * hf + reach))
        if r is None:
            return
        orig = picture[r[1]:r[3], r[0]:r[2]]
        out = orig.copy()
        moved = None
        if self.new:
            self._cover(out, r, g)
        else:
            moved = self._warp(orig, r, g, mo)
        lit_room = self.light_k > 0 and abs(self._room(mo)) >= 1e-6
        if moved is not None:  # (the old flame's light moved; the light on the room is not on the flame itself)
            at, lit, lay = moved
            part = out[at]
            if lit_room:
                part -= lit
                self._light_on(out, r, g, mo, (at, part + lay))
                part += lay
            else:
                lay -= lit
                part += lay
        elif lit_room:
            self._light_on(out, r, g, mo)
        br = self._rect(x0, y0, x0 + w, y0 + h)
        if br is not None:
            light, cover = self._lights(mo, g, br)
            part = out[br[1] - r[1]:br[3] - r[1], br[0] - r[0]:br[2] - r[0]]
            if cover is not None:
                part *= (1 - cover)[..., None]
            np.minimum(light, 1.0, out=light)
            light *= 1 - part  # (Screen: no light, no change)
            part += light
        k = gain if mask is None else gain * mask[r[1]:r[3], r[0]:r[2], None]
        if mask is None and gain >= 1:
            orig[:] = out
        else:
            orig += k * (out - orig)
