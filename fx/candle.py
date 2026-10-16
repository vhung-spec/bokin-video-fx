"""A candle flame (a placed lens effect): brings a flame of the video back to life, or draws a new one over it.

The box (Effect.BOX, box()) goes over a flame: the flame's base (the wick's tip) near its bottom middle, the flame
above it, room for its halo all round. Two ways ("mode"):
- "alive": the flame already in the picture sways and stretches (a local warp anchored at its base, growing toward
  its tip), its brightness breathes, now and then it flutters, and its light flickers on what is around it.
- "new": the old flame is filled in softly from what is around it, and a new flame is drawn: a small blue base, a
  white-yellow core, a warm orange-yellow body, a soft halo; it moves the same way and lights its surroundings.

apply() changes the picture (the export and the exact frames); draw() is what the page's live preview lays over
the picture with Screen: the halo, and the new flame. Every size is relative to the box, so the box alone (the
preview clip) and the whole frame look the same; only the light on the surroundings reaches out of the box.

Like every effect it loops: every motion is a sum of sines with a whole number of cycles per loop, and the
flutters are bursts at fixed times of the loop with periodic envelopes, so frame P is frame 0 again.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .effects import Effect, _choice, _lerp, _num, _place

DEFAULTS = {"size": 50, "brightness": 50, "flicker": 35, "sway": 25, "flutter": 30, "speed": 50, "halo": 40,
            "light": 45, "reach": 40, "warmth": 50}
LOOKS = {
    "still": {},
    "breeze": {"flicker": 60, "sway": 65, "flutter": 70, "speed": 60, "light": 60},
    "soft": {"size": 38, "brightness": 40, "flicker": 20, "sway": 15, "flutter": 10, "speed": 40, "halo": 30,
             "light": 30, "reach": 30, "warmth": 40},
    "warm": {"size": 62, "brightness": 70, "flicker": 40, "sway": 30, "halo": 60, "light": 60, "reach": 60,
             "warmth": 65},
}

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
    *_place(13)]}

TAU = 2 * math.pi
BASE = 0.78  # the flame's base (the wick's tip) from the box's top, box heights
FLAME = 0.40  # the flame's height (base to tip) at size 50, box heights
HALF_W = 0.165  # the flame's half width where widest, flame heights
LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
FLUTTERS = 48  # flutter slots with fixed random draws
FINE = 200  # pixels a flame height, at most, the new flame is drawn at (then enlarged)
FILL_STEPS = 120  # smoothing steps of the fill over the old flame (at about 40 pixels a flame height)


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


def _expose(v, m, weight):
    """Channel values `v` (changed in place) with m * weight (m > -1) times more light. What the camera saw white
    had light to spare (the flame, wax right by it): it stays white as the light dims."""
    k = weight * m
    if m < 0:
        k *= 1 - _ramp(v, 0.8, 1.0)
    k += 1
    v *= k


def _gauss(n, sigma):
    """An n x n matrix that blurs along one axis (unnormalised: used for ratios)."""
    i = np.arange(n, dtype=np.float32)
    return np.exp(-0.5 * ((i[:, None] - i[None, :]) / max(sigma, 0.3)) ** 2).astype(np.float32)


def _resize(a, w, h):
    """A float array (h0 x w0, or with channels) resized to w x h (bilinear)."""
    if a.ndim == 3:
        return np.dstack([_resize(a[..., c], w, h) for c in range(a.shape[2])])
    if a.shape == (h, w):
        return a
    return np.asarray(Image.fromarray(np.ascontiguousarray(a, np.float32), "F").resize(
        (w, h), Image.Resampling.BILINEAR), np.float32)


class _Waves:
    """A smooth wandering signal: sines at about `rates` Hz (whole cycles per loop), unit RMS."""

    def __init__(self, fx, rng, rates, amps):
        rates = np.asarray(rates, np.float64) * (0.85 + 0.3 * rng.random(len(rates)))
        self.m = fx.cycles(rates * fx.speed_k)
        self.ph = rng.random(len(rates)) * TAU
        a = np.asarray(amps, np.float64)
        self.a = a / math.sqrt(float((a * a).sum()) / 2)

    def at(self, tau):
        return float(np.dot(self.a, np.sin(TAU * self.m * tau + self.ph)))


class Candle(Effect):
    BOX = (0.75, 1.0)
    blend = "screen"
    lens = True
    VERSION = 2  # (1: a first sketch, a glow only)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        self.new = p["mode"] == "new"
        self.speed_k = 2.0 ** ((p["speed"] - 50) / 50)
        self.fk, self.sk = p["flicker"] / 100, p["sway"] / 100
        # the new flame's size; the old one is only stretched a little (more would warp it out of shape)
        self.size_k = _lerp(0.7, 1.3, p["size"] / 100) if self.new else _lerp(0.82, 1.22, p["size"] / 100)
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
        self.m_rip = self.cycles(3.2 * self.speed_k)
        # flutters: short bursts of a quick flicker at fixed times of the loop
        rng = np.random.default_rng([self.seed, 1103, 2])
        first, u = rng.random(), rng.random((FLUTTERS, 6))
        fl = p["flutter"] / 100
        self.fl = 0.6 + 0.4 * fl
        n = min(FLUTTERS, int(0.25 * fl * self.loop_s + first)) if fl > 0 else 0
        sigma = (0.12 + 0.14 * u[:n, 1]) / self.speed_k  # seconds
        self.f_at = (np.arange(n) + 0.15 + 0.7 * u[:n, 0]) / max(n, 1)
        self.f_kappa = (self.loop_s / (TAU * sigma)) ** 2
        self.f_m = self.cycles((6.0 + 4.0 * u[:n, 2]) * self.speed_k)
        self.f_py, self.f_px = u[:n, 3] * TAU, u[:n, 4] * TAU
        self.f_amp = 0.6 + 0.4 * u[:n, 5]

    # --- the motion ------------------------------------------------------------------------------------------

    def _motion(self, t):
        """(stretch, brightness, sway of the tip in flame heights, ripple, ripple phase) at frame t."""
        tau = self.tau(t)
        slow, fast, lean, jit = (w.at(tau) for w in self.waves)
        burst = fy = fx = 0.0
        if len(self.f_at):
            env = self.f_amp * np.exp(self.f_kappa * (np.cos(TAU * (tau - self.f_at)) - 1.0))
            burst = min(1.0, float(env.sum()))
            fy = float(np.dot(env, np.sin(TAU * self.f_m * tau + self.f_py)))
            fx = float(np.dot(env, np.sin(TAU * self.f_m * tau + self.f_px)))
        fk, sk, fl = self.fk, self.sk, self.fl
        stretch = 1 + fk * (0.07 * slow + 0.025 * fast) + fl * (0.09 * fy - 0.04 * burst)
        bright = 1 + fk * (0.11 * slow + 0.045 * fast) + fl * (0.08 * fy - 0.1 * burst)
        sway = sk * (0.09 * lean + 0.02 * jit) + fl * 0.07 * fx
        ripple = fk * 0.03 + fl * 0.05 * burst
        return max(0.6, stretch), max(0.3, bright), sway, ripple, TAU * float(self.m_rip * tau % 1.0)

    def _geom(self):
        """(x0, y0, w, h) of the box, the flame's base (cx, ay) and its height there (pixels)."""
        x0, y0, w, h = self.box()
        return x0, y0, w, h, x0 + w / 2, y0 + BASE * h, FLAME * h

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
        stretch, bright, sway, ripple, phase = mo
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
        fr = self._rect(cx - 0.6 * hf * self.size_k, ay - 1.45 * hs, cx + 0.6 * hf * self.size_k, ay + 0.3 * hs)
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
        stretch, bright, sway, ripple, phase = mo
        u = (ay - ys) / hs  # height above the base, 0..1 (the tip)
        uc = np.clip(u, 0.0, 1.0)
        um = 0.5 / 1.6  # where it is widest
        prof = np.sqrt(uc / um) * ((1 - uc) / (1 - um)) ** 1.1
        r = HALF_W * hf * self.size_k / math.sqrt(stretch) * prof * (
            1 + 0.6 * ripple * np.sin(TAU * 1.7 * uc - phase - 1.0))
        xc = cx + self._centre(u, hs, sway, ripple, phase)
        dx = xs[None, :] - xc[:, None]
        adx = np.abs(dx)
        soft = max(0.7, 0.03 * hf * self.size_k)
        ends = (_ramp(u, -0.02, 0.03) * (1 - _ramp(u, 0.97, 1.0)))[:, None]
        cov = np.clip(0.5 + (r[:, None] - adx) / soft, 0.0, 1.0) * ends
        dn = adx / np.maximum(r, 0.5 * soft)[:, None]
        # it burns yellow round a dark cone over the wick and above a dim blue cup at its bottom, seen most at its
        # sides (looked at through more of it)
        start = 0.05 + 0.1 * dn * dn + 0.17 * (1 - _ramp(dn, 0.0, 0.42))
        lit = _ramp(u[:, None] - start, 0.0, 0.08)
        # colour: orange-yellow at the edge, golden body, white-yellow core low in the middle, orange to the tip
        col = self.c_body + (self.c_edge - self.c_body) * _ramp(dn, 0.5, 1.0)[..., None]
        core = (1 - _ramp(dn, 0.25, 0.7)) * (_ramp(u, 0.2, 0.34) * (1 - _ramp(u, 0.5, 0.85)))[:, None]
        col += (self.c_core - col) * core[..., None]
        tip = _ramp(u, 0.6, 1.0)[:, None]
        col += (self.c_edge - col) * (0.6 * tip)[..., None]
        lum = (0.85 + 0.6 * core) * (1 - 0.3 * tip)
        lum *= cov * lit * level
        rgb = col * lum[..., None]
        dark = cov * (1 - lit) * (1 - _ramp(dn, 0.3, 0.6))  # the dark cone: a faint orange
        blue = cov * (1 - lit) * _ramp(dn, 0.45, 0.9) * (1 - _ramp(u, 0.1, 0.22))[:, None]
        rgb += dark[..., None] * (0.25 * level * self.c_edge) + blue[..., None] * (0.2 * self.c_blue)
        # a soft glow just round the flame (as a camera sees a bright flame)
        dv = (np.maximum(-u, 0) + 1.6 * np.maximum(u - 1, 0)) * hs
        dh = np.maximum(adx - r[:, None], 0)
        sig = 0.07 * hf * self.size_k
        bloom = np.exp(-(dh * dh + (dv * dv)[:, None]) / (2 * sig * sig)) * (0.22 * level)
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

    def _warp(self, src, out, r, g, mo):
        """The old flame alive: `out` (rect r of the frame) read from `src` through a warp anchored at the
        flame's base that sways and stretches it, and its brightness breathing."""
        stretch, bright, sway, ripple, phase = mo
        x0, y0, w, h, cx, ay, hf = g
        s = max(0.76, self.size_k * stretch)  # (so the warp never folds over: see `fade`)
        wr = self._rect(cx - 0.95 * hf, ay - 1.9 * hf, cx + 0.95 * hf, ay + 0.1 * hf)  # (inside the box)
        if wr is None:
            return
        wx0, wy0, wx1, wy1 = max(wr[0], r[0]), max(wr[1], r[1]), min(wr[2], r[2]), min(wr[3], r[3])
        if wx0 >= wx1 or wy0 >= wy1:
            return
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
        m = int(0.6 * hf) + 2  # (no pixel moves further)
        sx0, sy0 = max(r[0], wx0 - m), max(r[1], wy0 - m)
        part = _sample(src[sy0 - r[1]:min(r[3], wy1 + m) - r[1], sx0 - r[0]:min(r[2], wx1 + m) - r[0]],
                       xsrc - sx0, ysrc - sy0)
        # brightness: the bright flame (not what is round it) breathes with the light
        lum = part @ LUMA
        bg = float(np.percentile(lum[::2, ::2], 30))
        top = float(lum.max())
        span = top - bg
        if span > 0.05:
            flame = _ramp(lum, bg + 0.3 * span, bg + 0.8 * span) * _ramp(np.float32(span), 0.08, 0.25)
            flame *= side[None, :] * _ramp(u, -0.03, 0.03)[:, None]  # (not the wax below it)
            rows, cols = np.flatnonzero(flame.any(axis=1)), np.flatnonzero(flame.any(axis=0))
            if len(rows):
                at = slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)
                for c in range(3):
                    _expose(part[at + (c,)], self.bright_k * (1 + 0.7 * (bright - 1)) * self.tint[c] - 1,
                            flame[at])
        out[wy0 - r[1]:wy1 - r[1], wx0 - r[0]:wx1 - r[0]] = part

    def _cover(self, out, r, g):
        """The old flame filled in from what is round it (in place in `out`, rect r of the frame)."""
        x0, y0, w, h, cx, ay, hf = g
        zr = self._rect(cx - 0.75 * hf, ay - 1.6 * hf, cx + 0.75 * hf, ay + 0.15 * hf)
        if zr is None or self.cover_k <= 0:
            return
        zx0, zy0, zx1, zy1 = max(zr[0], r[0]), max(zr[1], r[1]), min(zr[2], r[2]), min(zr[3], r[3])
        if zx0 >= zx1 or zy0 >= zy1:
            return
        zw, zh = zx1 - zx0, zy1 - zy0
        pic = out[zy0 - r[1]:zy1 - r[1], zx0 - r[0]:zx1 - r[0]]
        k = min(1.0, 40.0 / hf)  # worked on at about 40 pixels a flame height
        lw, lh = max(2, round(zw * k)), max(2, round(zh * k))
        small = _resize(pic, lw, lh)
        lhf = hf * lw / zw
        xs = (np.arange(lw, dtype=np.float32) + 0.5) * zw / lw + zx0
        ys = (np.arange(lh, dtype=np.float32) + 0.5) * zh / lh + zy0
        # where the old flame may be: an upright oval over the base, nothing below the base
        ox, oy = (xs - cx) / (0.5 * hf), (ys - (ay - 0.6 * hf)) / (0.82 * hf)
        zone = (1 - _ramp(np.hypot(ox[None, :], oy[:, None]), 0.75, 1.0)) * (1 - _ramp(ys, ay - 0.05 * hf,
                                                                                         ay + 0.08 * hf))[:, None]
        # the flame: brighter than the light round it, or among the brightest there (not its glow, which the
        # fill keeps), and always where the box says the flame is
        lum = small @ LUMA
        gy, gx = _gauss(lh, 0.22 * lhf), _gauss(lw, 0.22 * lhf)
        local = (gy / gy.sum(axis=1, keepdims=True)) @ lum @ (gx / gx.sum(axis=1, keepdims=True)).T
        near = 1 - _ramp(np.abs(xs - cx) / hf, 0.28, 0.4)  # (a dark thing beside it makes its glow stand out)
        above = (lum - local) * zone * near[None, :]
        most = float(above.max())
        inside = lum[zone > 0.5]
        if inside.size < 4 or most < 0.04:
            return
        bg, top = float(np.percentile(inside, 25)), float(inside.max())
        span = top - bg
        flame = np.maximum(_ramp(above, 0.2 * most, 0.45 * most) * _ramp(lum, bg + 0.25 * span, bg + 0.45 * span),
                           _ramp(lum, bg + 0.55 * span, bg + 0.8 * span))
        flame *= zone * float(_ramp(np.float32(most), 0.04, 0.1))
        ox, oy = (xs - cx) / (0.24 * hf), (ys - (ay - 0.52 * hf)) / (0.58 * hf)
        np.maximum(flame, 1 - _ramp(np.hypot(ox[None, :], oy[:, None]), 0.8, 1.0), out=flame)
        grow = max(1, round(0.06 * lhf))
        flame = np.asarray(Image.fromarray(flame.astype(np.float32), "F").filter(ImageFilter.MaxFilter(2 * grow + 1)),
                           np.float32)
        # the fill: a smooth membrane over the hole held by what is right round it (nothing from further out
        # bleeds in), started from a blur of what is round it
        keep = 1 - flame
        wsum = gy @ keep @ gx.T + 1e-6
        fill = np.dstack([gy @ (keep * small[..., c]) @ gx.T / wsum for c in range(3)])
        hole = (flame > 0.5)[..., None]
        fill = np.where(hole, fill, small)
        for _ in range(FILL_STEPS):
            p = np.pad(fill, ((1, 1), (1, 1), (0, 0)), mode="edge")
            fill = np.where(hole, (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) * 0.25, fill)
        sy, sx = _gauss(lh, 0.04 * lhf), _gauss(lw, 0.04 * lhf)
        soft = (sy / sy.sum(axis=1, keepdims=True)) @ flame @ (sx / sx.sum(axis=1, keepdims=True)).T
        soft = np.clip(soft * 1.15, 0, 1) * zone
        m = _resize(soft.astype(np.float32), zw, zh) * self.cover_k
        fill = _resize(fill.astype(np.float32), zw, zh)
        pic += m[..., None] * (fill - pic)

    def _light_on(self, out, r, g, mo):
        """The flame's light flickering on what is round it: brighter and darker in turn, more near the flame."""
        stretch, bright, sway, ripple, phase = mo
        x0, y0, w, h, cx, ay, hf = g
        m = self.light_k * (bright - 1)
        if abs(m) < 1e-6:
            return
        rad = self.reach * hf
        lx, ly = cx + 0.35 * sway * hf, ay - 0.5 * hf * self.size_k * stretch
        xs = np.arange(r[0], r[2], dtype=np.float32) + 0.5
        ys = np.arange(r[1], r[3], dtype=np.float32) + 0.5
        share = np.outer(np.exp(-0.5 * ((ys - ly) / rad) ** 2), np.exp(-0.5 * ((xs - lx) / rad) ** 2)).astype(
            np.float32)
        for c in range(3):
            _expose(out[..., c], m * self.c_light[c], share)

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
        if self.new:
            self._cover(out, r, g)
        else:
            self._warp(orig, out, r, g, mo)
        if self.light_k > 0:
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
