"""Water on a window pane: fogged glass that blurs and whitens what is behind it, beads of water that stay, bigger
drops that hold, then run down in fits and starts leaving a clear trail that slowly fogs over again, and new drops
where rain hits the glass.

A lens effect: apply() changes the picture under it (the renderer: export and exact frames). Fogged glass spreads
the light of the scene behind it (a blur weighted by how fogged it is, with a wide glow) under a thin milky veil;
through every drop the scene is seen upside down and smaller (a wide view squeezed into the drop), with a darker
rim and a small bright glint. draw() is only what is laid over the picture (the page's live preview): the veil with
its clear trails, the rims, glints and a hint of the upside-down sky in each drop.

Like every effect it loops (see effects.py): a running drop goes round a whole number of cycles per loop (on another
path each cycle, from a fixed table), its trail's age wraps with the cycle, and the beads it wiped grow back as a
function of the time since, modulo the loop. Sizes are pixels of a 1080-line frame.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .effects import Effect, _choice, _colour, _lerp, _num, _rgb

DEFAULTS = {"haze": 45, "blur": 50, "uneven": 40, "refog": 6, "beads": 50, "bead_size": 40, "drops": 35, "size": 45,
            "speed": 40, "wiggle": 40, "hits": 30, "refraction": 50, "highlight": 50, "rim": 50}
LOOKS = {
    "rain": {},
    "mist": {"haze": 30, "blur": 35, "uneven": 55, "refog": 9, "beads": 30, "bead_size": 25, "drops": 10,
             "size": 35, "speed": 30, "hits": 8, "highlight": 40, "rim": 40},
    "steamy": {"haze": 80, "blur": 75, "uneven": 50, "refog": 12, "beads": 80, "bead_size": 45, "drops": 25,
               "size": 50, "speed": 35, "hits": 12, "rim": 55},
    "storm": {"haze": 30, "blur": 45, "uneven": 35, "refog": 4, "beads": 55, "bead_size": 50, "drops": 75,
              "size": 60, "speed": 70, "wiggle": 55, "hits": 85, "highlight": 60},
    "clear": {"haze": 12, "blur": 80, "uneven": 20, "refog": 5, "beads": 65, "bead_size": 55, "drops": 40,
              "size": 55, "speed": 45, "hits": 45, "refraction": 60, "highlight": 60, "rim": 55},
}

SCHEMA = {"label": "Nước trên kính", "params": [
    _choice("look", "Kiểu", "rain", [("rain", "Mưa trên kính"), ("mist", "Hơi nước nhẹ"),
                                    ("steamy", "Kính đọng nhiều hơi nước"), ("storm", "Mưa to, nước chảy nhiều"),
                                    ("clear", "Kính trong, giọt mưa đọng")],
            presets={k: {**DEFAULTS, **v} for k, v in LOOKS.items()}, group="Kiểu"),
    _num("haze", "Hơi nước (kính mờ)", 45, group="Hơi nước"),
    _num("blur", "Nhoè cảnh sau kính", 50, group="Hơi nước"),
    _num("uneven", "Chỗ đậm chỗ nhạt", 40, group="Hơi nước"),
    _num("refog", "Vệt nước chảy mờ lại sau (giây)", 6, 1, 20, group="Hơi nước"),
    _num("beads", "Số giọt đọng", 50, group="Giọt đọng"),
    _num("bead_size", "Cỡ giọt đọng", 40, group="Giọt đọng"),
    _num("drops", "Số giọt chảy", 35, group="Giọt chảy"),
    _num("size", "Cỡ giọt chảy", 45, group="Giọt chảy"),
    _num("speed", "Tốc độ chảy", 40, group="Giọt chảy"),
    _num("wiggle", "Uốn lượn khi chảy", 40, group="Giọt chảy"),
    _num("hits", "Giọt mưa mới bám vào kính", 30, group="Mưa rơi vào kính"),
    _num("refraction", "Cảnh lộn ngược trong giọt", 50, group="Ánh sáng"),
    _num("highlight", "Ánh sáng lấp lánh trên giọt", 50, group="Ánh sáng"),
    _num("rim", "Viền tối quanh giọt", 50, group="Ánh sáng"),
    _num("opacity", "Độ đậm", 100, group="Màu"),
    _colour("color", "Màu hơi nước", "#dfe6ea", group="Màu")]}

# the planes of the drop buffer: cover, where the drop shows the scene from (offset), rim, glint, how much it
# clears the fog, the preview's hint of sky, how small (which level of the picture pyramid) the scene is seen
A, DX, DY, RIM, HL, CLR, SKY, LOD = range(8)
LEVELS = 6  # the picture, then halved five times
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)
_HALVES = (2, 3, 5, 8, 12, 17, 24, 34, 48, 68, 96)  # patch half-sizes the drops are drawn on


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _bilinear(img, x, y):
    """`img` (h x w x 3) at float positions x, y (pixel i's centre at i + 0.5), clamped to its edges."""
    h, w = img.shape[:2]
    if w < 2 or h < 2:
        return img.reshape(-1, 3)[np.zeros(len(x), np.int64)]
    x = np.clip(x - 0.5, 0.0, w - 1.001)
    y = np.clip(y - 0.5, 0.0, h - 1.001)
    xi, yi = x.astype(np.int32), y.astype(np.int32)
    fx, fy = (x - xi)[:, None], (y - yi)[:, None]
    flat = img.reshape(-1, 3)
    i = yi * w + xi
    top = flat[i] * (1 - fx) + flat[i + 1] * fx
    bottom = flat[i + w] * (1 - fx) + flat[i + w + 1] * fx
    return top * (1 - fy) + bottom * fy


class Glass(Effect):
    lens = True
    VERSION = 1
    RUNS = 64  # running-drop slots with fixed random draws
    HITS = 200  # rain-hit slots
    VARIANTS = 64  # the different paths of a slot, cycle after cycle
    KNOTS = 16  # the stop-and-go speed table of a path

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        self.opacity = p["opacity"] / 100
        self.colour = _rgb(p["color"])
        self.veil = 0.3  # the milky veil of fully fogged glass
        self.blur_k = p["blur"] / 100
        self.blur_r = 24.0 * px * self.blur_k  # gaussian radius of the scene behind fogged glass
        refr = p["refraction"] / 100
        self.lens_k = min(1.0, refr * 4)  # 0: the scene goes straight through a drop
        self.view = _lerp(30.0, 320.0, refr ** 1.5) * px  # how far around it a drop sees (pixels of the scene)
        self.hl_k = p["highlight"] / 100 * 1.5
        self.rim_k = p["rim"] / 100 * 0.6
        self.buf = np.zeros((8, self.h, self.w), np.float32)
        self.flat = self.buf.reshape(8, -1)
        self.dirty = None  # (entries, trail rectangles) of the last frame, cleared before the next
        self._box_of = (None, None)
        self.fog = self._fog_map()
        self._beads()
        self._runs()
        self._hits()
        self._wipes()

    def _rng(self, k):
        return np.random.default_rng([self.seed, 1103, k])

    # --- what stays: the fog and the beads -------------------------------------------------------------------

    def _noise(self, g, cells, size=None):
        """Smooth noise 0..1 with `cells` cells per frame height, the same at any frame size (made at `size`, the
        frame's by default)."""
        nx = int(math.ceil(cells * self.aspect)) + 3
        grid = g.random((int(cells) + 3, nx)).astype(np.float32)
        im = Image.fromarray(grid, "F").resize(size or (self.w, self.h), Image.Resampling.BICUBIC,
                                                box=(1, 1, 1 + cells * self.aspect, 1 + cells))
        return np.array(im, np.float32)

    def _fog_map(self):
        """How fogged the glass is at each pixel (0..1) before drops and trails clear it."""
        p = self.p
        g = self._rng(1)
        lo_h = min(self.h, 270)  # the smooth parts are made small, then enlarged once
        self.lo = lo = (max(1, round(lo_h * self.aspect)), lo_h)
        n = 0.55 * self._noise(g, 2.5, lo) + 0.3 * self._noise(g, 6, lo) + 0.15 * self._noise(g, 14, lo) - 0.5
        grain = self._noise(g, 400)  # the fine texture of condensation
        self.mottle = self._noise(g, 120).astype(np.float16)  # where a trail fogs over first
        self.clusters = 0.6 * self._noise(g, 4, lo) + 0.4 * self._noise(g, 11, lo) - 0.5  # where beads gather
        shape = np.clip(1 + p["uneven"] / 100 * 3.0 * n, 0.08, 1.9)
        fog = np.array(Image.fromarray(shape, "F").resize((self.w, self.h), Image.Resampling.BICUBIC), np.float32)
        fog *= np.linspace(0.9, 1.12, self.h, dtype=np.float32)[:, None] * (p["haze"] / 100)  # thicker low down
        grain *= 0.1
        grain += 0.95
        fog *= grain
        return np.clip(fog, 0.0, 1.0, out=fog)

    def _beads(self):
        """The beads that stay on the glass: big, middle and small ones, each size on its own grid of cells with at
        most one bead a cell, smaller ones only where they do not touch a bigger one (so beads never overlap); how
        many cells keep theirs follows the slider."""
        p, px, h = self.p, self.px, self.h
        amount, size = p["beads"] / 100, p["bead_size"] / 100
        levels = []
        for level, (cell, keep, r0, r1) in enumerate(((0.06, 0.5, 3.5, 11.0), (0.028, 0.45, 1.8, 5.0),
                                                      (0.014, 0.55, 0.8, 2.2))):
            g = self._rng(10 + level)
            ny, nx = int(1 / cell) + 1, int(self.aspect / cell) + 1
            u = g.random((ny, nx, 8)).astype(np.float32)
            cpx = cell * h
            r = np.minimum(_lerp(r0, r1, size) * px * (0.45 + 0.55 * u[..., 0] ** 1.6), 0.3 * cpx)
            m = r * 1.5 + 1
            gy, gx = np.mgrid[0:ny, 0:nx].astype(np.float32)
            cx = (gx * cpx + m + u[..., 1] * np.maximum(cpx - 2 * m, 0)).ravel()
            cy = (gy * cpx + m + u[..., 2] * np.maximum(cpx - 2 * m, 0)).ravel()
            r, m = r.ravel(), m.ravel()
            lw, lh = self.lo
            here = self.clusters[np.clip((cy * lh / h).astype(int), 0, lh - 1),
                                 np.clip((cx * lw / self.w).astype(int), 0, lw - 1)]
            kept = (cx < self.w + m) & (cy < h + m) & (
                u[..., 3].ravel() < amount * keep * np.clip(1 + 4.0 * here, 0.1, 1.8))
            for big in levels:  # not touching a bigger bead
                bc = big["cell"]
                sy, sx = (cy // bc).astype(int), (cx // bc).astype(int)
                bny, bnx = big["shape"]
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        j = np.clip(sy + dy, 0, bny - 1) * bnx + np.clip(sx + dx, 0, bnx - 1)
                        d = np.hypot(cx - big["cx"][j], cy - big["cy"][j])
                        kept &= ~(big["kept"][j] & (d < big["r"][j] * 1.4 + r * 1.3 + 1))
            e = 1 + 0.16 * (u[..., 4].ravel() - 0.5) * (r > 2.5 * px)
            levels.append(dict(cx=cx, cy=cy, r=r, e=e, rot=u[..., 5].ravel() * math.pi,
                               taper=0.04 + 0.12 * u[..., 6].ravel(), kept=kept, cell=cpx, shape=(ny, nx)))
        cols = {}
        for key in ("cx", "cy", "r", "e", "rot", "taper"):
            cols[key] = np.concatenate([lv[key][lv["kept"]] for lv in levels]).astype(np.float32)
        self.bead = cols
        n = len(cols["cx"])
        r, e = cols["r"], cols["e"]
        ent = self._raster(cols["cx"], cols["cy"], r * e, r / e, cols["taper"], cols["rot"], np.ones(n, np.float32),
                           0.25 + 0.6 / np.maximum(r, 0.5))
        # where a bead's ring reaches into a neighbour, the stronger entry stays
        order = np.lexsort((ent["f"][A] * 2 + ent["f"][CLR], ent["idx"]))
        idx = ent["idx"][order]
        sel = order[np.r_[idx[1:] != idx[:-1], True]] if len(idx) else order
        self.b_idx, self.b_id, self.b_f = ent["idx"][sel], ent["id"][sel], ent["f"][:, sel]

    # --- drops: one drawing routine for all ------------------------------------------------------------------

    def _raster(self, cx, cy, rx, ry, taper, rot, vis, ring):
        """Drops (centre, radii across and down, pixels; top narrowed by `taper`; turned by `rot`; `vis` 0..1; fog
        cleared out to `ring` times the radius beyond its edge) as sparse entries: flat pixel index, drop number and
        the buffer's planes there."""
        w, h = self.w, self.h
        reff = np.sqrt(rx * ry)
        reach = np.maximum(rx, ry) * (1 + ring) + 1.5
        which = np.minimum(np.searchsorted(_HALVES, reach), len(_HALVES) - 1)
        idx_l, id_l, f_l = [], [], []
        for k in np.unique(which):
            sel = np.flatnonzero(which == k)
            hs = _HALVES[k]
            g = np.arange(-hs, hs + 1)
            bx, by = np.floor(cx[sel]).astype(np.int64), np.floor(cy[sel]).astype(np.int64)
            ix = bx[:, None, None] + g[None, None, :]
            iy = by[:, None, None] + g[None, :, None]
            col = lambda a: a[sel][:, None, None]  # noqa: E731
            dx = (ix + 0.5 - col(cx)).astype(np.float32)
            dy = (iy + 0.5 - col(cy)).astype(np.float32)
            c, s = np.cos(col(rot)), np.sin(col(rot))
            u = (c * dx + s * dy) / col(rx)
            v = (-s * dx + c * dy) / col(ry)
            u = u / (1 - col(taper) * np.clip(-v, 0, 1))
            q = np.sqrt(u * u + v * v)
            rg, vi = col(ring), col(vis)
            clr = np.where(q <= 1, 1.0, 1 - _smooth((q - 1) / rg)) * vi
            keep = (clr > 0.002) & (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            if not keep.any():
                continue
            pick = lambda a: np.broadcast_to(a, keep.shape)[keep]  # noqa: E731
            q, u, v, dx, dy = pick(q), pick(u), pick(v), pick(dx), pick(dy)
            re, vi, rxs, rys = pick(col(reff)), pick(vi), pick(col(rx)), pick(col(ry))
            cov = np.clip((1 - q) * re + 0.5, 0, 1) * vi
            # the scene through the drop: upside down, squeezed more towards the edge
            m0 = np.clip(self.view / re, 1.0, 120.0)
            out = 1 + m0 * (1 + 0.5 * q * q)
            # the glint: a soft spot up and to the left, never sharper than a pixel; light gathered at the bottom
            sig = np.maximum(0.16 * re, 0.6)
            amp = np.minimum(1.0, (0.16 * re / 0.6) ** 2) ** 0.5
            gx_, gy_ = dx + 0.32 * rxs, dy + 0.4 * rys
            spot = amp * np.exp(-(gx_ * gx_ + gy_ * gy_) / (2 * sig * sig))
            caustic = 0.3 * amp * np.exp(-((q - 0.74) / 0.15) ** 2) * np.clip(v, 0, 1) ** 2
            f = np.empty((8, len(q)), np.float32)
            f[A] = cov
            f[DX], f[DY] = -dx * out * self.lens_k, -dy * out * self.lens_k
            f[RIM] = self.rim_k * np.clip(re / 4.0, 0.2, 1) * _smooth((q - 0.6) / 0.4) ** 1.3 * (
                0.75 + 0.3 * np.clip(v, -1, 1)) * cov
            f[HL] = np.minimum(1.0, self.hl_k * (spot + caustic)) * cov
            f[CLR] = pick(clr)
            f[SKY] = np.clip(v, -1, 1) * cov
            f[LOD] = np.clip(np.log2(1 + m0 * (1 + 1.5 * q * q)) * self.lens_k - 0.5, 0, LEVELS - 1.001)
            idx_l.append(pick(iy) * w + pick(ix))
            id_l.append(np.broadcast_to(sel[:, None, None], keep.shape)[keep])
            f_l.append(f)
        if not idx_l:
            return {"idx": np.zeros(0, np.int64), "id": np.zeros(0, np.int64), "f": np.zeros((8, 0), np.float32)}
        return {"idx": np.concatenate(idx_l), "id": np.concatenate(id_l), "f": np.concatenate(f_l, axis=1)}

    # --- running drops ---------------------------------------------------------------------------------------

    def _runs(self):
        """Slots of drops that grow, hold, run down in fits and starts and leave a trail; a slot goes round a whole
        number of cycles per loop, on another path (from its fixed table) each cycle."""
        p, px, h, w = self.p, self.px, self.h, self.w
        g = self._rng(20)
        us = g.random((self.RUNS, 2))
        uv = g.random((self.RUNS, self.VARIANTS, 18))
        ug = g.random((self.RUNS, self.VARIANTS, self.KNOTS))
        stops = g.random((self.RUNS, self.VARIANTS, self.KNOTS))
        ub = g.random((self.RUNS, self.VARIANTS, 10, 4))
        n = min(self.RUNS, int(round(p["drops"] / 100 * 36 * self.aspect / (16 / 9))))
        size, speed, wig = p["size"] / 100, p["speed"] / 100, p["wiggle"] / 100
        refog = float(p["refog"])
        want = _lerp(5.0, 11.0, us[:n, 0]) * _lerp(1.3, 0.75, speed)
        c = np.maximum(1, np.floor(self.loop_s / np.maximum(want, refog)))
        T = self.loop_s / c
        self.r_n, self.r_c, self.r_phi, self.r_T = n, c, us[:n, 1], T
        self.r_tau = np.minimum(refog, T)  # a trail is fogged over again before the slot's next-but-one cycle
        u = uv[:n]
        R = _lerp(4.0, 15.0, size) * (0.75 + 0.5 * u[..., 0]) * px
        Tc = T[:, None]
        # seconds: growing, holding, running; the rest of the cycle it rests, then shrinks away (all within it)
        t_fade = np.minimum(0.6, 0.12 * Tc)
        t_grow = np.minimum(_lerp(0.5, 1.1, u[..., 4]), 0.2 * Tc)
        room = 0.97 * Tc - t_grow - t_fade
        t_hold = np.minimum(_lerp(0.3, 2.5, u[..., 5]), 0.35 * room)
        v_avg = _lerp(30.0, 240.0, speed) * (0.7 + 0.6 * u[..., 6]) * px
        L = _lerp(0.12, 1.25, u[..., 3]) * h
        L = np.minimum(L, v_avg * (room - t_hold))
        t_run = L / v_avg
        self.r_R = R
        self.r_x0 = -R + u[..., 1] * (w + 2 * R)
        self.r_y0 = (-0.04 + 0.8 * u[..., 2] ** 1.4) * h
        self.r_L = L
        self.r_sg = t_grow / Tc
        self.r_sr = (t_grow + t_hold) / Tc
        self.r_se = self.r_sr + t_run / Tc
        self.r_sf = np.maximum(self.r_se, 1 - (t_fade + 0.015 * Tc) / Tc)
        amp = wig * _lerp(1.5, 7.0, u[..., 7]) * px * np.sqrt(R / (8 * px))
        self.r_wig = np.stack([amp, 0.3 * amp * u[..., 10]])
        self.r_lam = np.stack([_lerp(90, 260, u[..., 8]), _lerp(30, 70, u[..., 11])]) * px
        self.r_ph = np.stack([u[..., 9], u[..., 12]]) * 2 * math.pi
        self.r_slant = (u[..., 13] - 0.5) * 0.12 * wig
        self.r_wf = _lerp(0.32, 0.55, u[..., 14])
        self.r_wph = u[..., 15] * 2 * math.pi
        # stop and go: each of KNOTS stretches is a stop (crawling) or a dash
        wgt = np.where(stops[:n] < 0.38, 0.03, 0.2 + ug[:n] ** 0.5)
        wgt = 0.25 * np.roll(wgt, 1, -1) + 0.5 * wgt + 0.25 * np.roll(wgt, -1, -1)
        cum = np.concatenate([np.zeros(wgt.shape[:-1] + (1,)), np.cumsum(wgt, -1)], -1)
        self.r_g = cum / cum[..., -1:]
        self.knots = np.linspace(0, 1, self.KNOTS + 1)
        # tiny beads left in the trail: where along it, how far to the side, size, whether there
        b = ub[:n]
        self.r_tb = (b[..., 0], (b[..., 1] - 0.5) * 1.4,
                     _lerp(0.6, 1.6, b[..., 2] ** 1.5) * px * np.sqrt(R / (8 * px))[..., None],
                     b[..., 3] < np.clip(L / (70 * px), 0.5, 10)[..., None] / 10)

    def _pos(self, i, v, s):
        """Down the path: y of slots i on paths v at cycle phase s (arrays)."""
        L, sr, se = self.r_L[i, v], self.r_sr[i, v], self.r_se[i, v]
        uu = np.clip((s - sr) / np.maximum(se - sr, 1e-6), 0, 1)
        k = np.minimum((uu * self.KNOTS).astype(int), self.KNOTS - 1)
        g = self.r_g[i, v]
        g0 = np.take_along_axis(g, k[:, None], 1)[:, 0]
        g1 = np.take_along_axis(g, k[:, None] + 1, 1)[:, 0]
        return self.r_y0[i, v] + L * (g0 + (g1 - g0) * (uu * self.KNOTS - k))

    def _x(self, i, v, y):
        """Across: the path's x at height y (it wanders a little, the trail follows it)."""
        d = y - self.r_y0[i, v]
        x = self.r_x0[i, v] + self.r_slant[i, v] * d
        for k in range(2):
            a, lam, ph = self.r_wig[k][i, v], self.r_lam[k][i, v], self.r_ph[k][i, v]
            x = x + a * (np.sin(2 * math.pi * d / lam + ph) - np.sin(ph))
        return x

    def _phase_at(self, i, v, y, speed=False):
        """The cycle phase at which the drop of slot i on path v left height(s) y (and if `speed`, how fast it went
        there, 1 = its average)."""
        g = self.r_g[i, v]
        frac = np.clip((y - self.r_y0[i, v]) / max(self.r_L[i, v], 1e-6), 0, 1)
        uu = np.interp(frac, g, self.knots)
        at = self.r_sr[i, v] + uu * (self.r_se[i, v] - self.r_sr[i, v])
        if not speed:
            return at
        k = np.minimum((uu * self.KNOTS).astype(int), self.KNOTS - 1)
        return at, (g[k + 1] - g[k]) * self.KNOTS

    # --- rain hitting the glass ------------------------------------------------------------------------------

    def _hits(self):
        """Slots of new drops: each cycle one lands somewhere (with a few tiny splashes), stays, then goes."""
        p, px = self.p, self.px
        g = self._rng(30)
        us = g.random((self.HITS, 3))
        uv = g.random((self.HITS, self.VARIANTS, 8))
        usat = g.random((self.HITS, self.VARIANTS, 3, 4))
        T0 = _lerp(2.5, 6.0, us[:, 0])
        rate = p["hits"] / 100 * 20 * self.aspect / (16 / 9)  # new drops a second
        n = min(self.HITS, int(np.searchsorted(np.cumsum(1 / T0), rate)) if rate > 0 else 0)
        self.h_n = n
        self.h_c = self.cycles(1 / T0[:n])
        self.h_phi = us[:n, 1]
        u = uv[:n]
        size = (p["size"] + p["bead_size"]) / 200
        self.h_x, self.h_y = u[..., 0] * self.w, u[..., 1] * self.h
        self.h_R = _lerp(2.0, 8.0, size) * (0.45 + 0.55 * u[..., 2] ** 1.3) * px
        self.h_sa = 0.2 * u[..., 3]
        self.h_sf = self.h_sa + _lerp(0.55, 0.72, u[..., 4])
        s = usat[:n]
        ang, dist = s[..., 0] * 2 * math.pi, _lerp(1.6, 3.2, s[..., 1])
        R = self.h_R[..., None]
        self.h_sat = (np.cos(ang) * dist * R, np.sin(ang) * dist * R, _lerp(0.22, 0.42, s[..., 2]) * R,
                      s[..., 3] < 0.55)

    def _wipes(self):
        """When each bead is wiped by a running drop (loop phase), and how long it takes to come back."""
        bx, by, br = self.bead["cx"], self.bead["cy"], self.bead["r"]
        order = np.argsort(bx)
        sx = bx[order]
        ids, at, back = [], [], []
        for i in range(self.r_n):
            c = int(self.r_c[i])
            for n in range(c):
                v = n % self.VARIANTS
                y0, L, R = self.r_y0[i, v], self.r_L[i, v], self.r_R[i, v]
                reach = abs(self.r_slant[i, v]) * L + 2.8 * self.r_wig[0][i, v] + 3 * R
                cand = order[np.searchsorted(sx, self.r_x0[i, v] - reach):np.searchsorted(sx, self.r_x0[i, v] + reach)]
                cand = cand[(by[cand] > y0 - R) & (by[cand] < y0 + L + R)]
                if not len(cand):
                    continue
                yy = np.clip(by[cand], y0, y0 + L)
                xs = self._x(np.full(len(cand), i), np.full(len(cand), v), yy)
                hit = np.abs(bx[cand] - xs) < R * max(self.r_wf[i, v], 0.8) + br[cand]
                cand, yy = cand[hit], yy[hit]
                if len(cand):
                    ids.append(cand)
                    at.append(np.mod((n + self._phase_at(i, v, yy) - self.r_phi[i]) / c, 1.0))
                    back.append(np.full(len(cand), self.r_tau[i] / self.loop_s))
        self.w_id = np.zeros(0, np.int64)
        if ids:
            ids, at, back = np.concatenate(ids), np.concatenate(at), np.concatenate(back)
            order = np.argsort(ids, kind="stable")
            self.w_id, self.w_at, self.w_back = ids[order], at[order], back[order]
            self.w_start = np.flatnonzero(np.r_[True, self.w_id[1:] != self.w_id[:-1]])

    # --- one frame -------------------------------------------------------------------------------------------

    def _fields(self, t):
        """Fills the drop buffer for frame t."""
        tau = self.tau(t)
        flat, buf = self.flat, self.buf
        if self.dirty is not None:
            idx, rects = self.dirty
            flat[:, idx] = 0
            for y0, y1, x0, x1 in rects:
                buf[CLR, y0:y1, x0:x1] = 0
        # the beads: there unless a running drop wiped them a moment ago
        vis = np.ones(len(self.bead["cx"]), np.float32)
        if len(self.w_id):
            since = np.mod(tau - self.w_at, 1.0) / self.w_back
            back = _smooth((since - 0.45) / 0.55).astype(np.float32)
            vis[self.w_id[self.w_start]] = np.minimum.reduceat(back, self.w_start)
        f = self.b_f * vis[self.b_id]
        f[DX], f[DY], f[LOD] = self.b_f[DX], self.b_f[DY], self.b_f[LOD]
        flat[:, self.b_idx] = f
        rects = self._trails(tau)
        small, big = self._drops(tau)
        self.dirty = (np.concatenate([self._merge(small), self._merge(big)]), rects)

    def _merge(self, ent):
        """Lays sparse drop entries over the buffer."""
        idx, f = ent["idx"], ent["f"]
        if not len(idx):
            return idx
        old = self.flat[:, idx]
        a, oa = f[A], old[A]
        new = np.empty_like(old)
        new[A] = a + oa * (1 - a)
        take = a >= oa
        for k in (DX, DY, LOD):
            new[k] = np.where(take, f[k], old[k])
        for k in (RIM, HL, SKY):
            new[k] = old[k] * (1 - a) + f[k]
        new[CLR] = np.maximum(old[CLR], f[CLR])
        self.flat[:, idx] = new
        return idx

    def _slots(self, tau):
        """Every running slot's path in this cycle and the one before, and the phase in this cycle."""
        z = tau * self.r_c + self.r_phi
        cyc = np.floor(z)
        cur = np.mod(cyc, self.r_c).astype(int) % self.VARIANTS
        prev = np.mod(cyc - 1, self.r_c).astype(int) % self.VARIANTS
        return np.arange(self.r_n), cur, prev, z - cyc

    def _trails(self, tau):
        """Clears the fog along every trail still showing; returns the rectangles touched."""
        rects = []
        if not self.r_n:
            return rects
        i_all, cur, prev, s_all = self._slots(tau)
        clr = self.buf[CLR]
        for i in i_all:
            T, tau_r = self.r_T[i], self.r_tau[i]
            for v, s in ((cur[i], s_all[i]), (prev[i], s_all[i] + 1)):
                sr, se = self.r_sr[i, v], self.r_se[i, v]
                s_head, s_tail = min(s, se), max(sr, s - tau_r / T)
                if s <= sr or s_tail >= s_head:
                    continue
                ii, vv = np.array([i]), np.array([v])
                y_head = self._pos(ii, vv, np.array([s_head]))[0]
                y_tail = self._pos(ii, vv, np.array([s_tail]))[0]
                R = self.r_R[i, v]
                y0 = max(0, int(math.floor(y_tail - 0.3 * R)))
                y1 = min(self.h, int(math.ceil(y_head)) + 1)
                if y1 <= y0:
                    continue
                ys = np.arange(y0, y1) + 0.5
                yc = np.clip(ys, self.r_y0[i, v], None)
                xc = self._x(np.full(len(ys), i), np.full(len(ys), v), yc)
                at, rel = self._phase_at(i, v, yc, True)
                fade = np.clip(1 - (s - at) * T / tau_r, 0, 1)
                fade = np.where(ys < self.r_y0[i, v] - 0.3 * R, 0, fade)
                # narrower where the drop dashed, wider where it stopped, a little uneven; thinner as it fogs over
                ph = self.r_wph[i, v]
                hw = R * self.r_wf[i, v] * np.clip(1.3 - 0.3 * rel, 0.75, 1.3) * (
                    0.84 + 0.1 * np.sin(ys / (9 * self.px) + ph) + 0.06 * np.sin(ys / (3.1 * self.px) + 2 * ph))
                hw = hw * (0.45 + 0.55 * np.sqrt(fade))
                x0 = max(0, int(math.floor((xc - hw).min())) - 1)
                x1 = min(self.w, int(math.ceil((xc + hw).max())) + 2)
                if x1 <= x0:
                    continue
                xs = np.arange(x0, x1) + 0.5
                soft = 1.2 * self.px + 0.5
                prof = np.clip((hw[:, None] - np.abs(xs[None, :] - xc[:, None])) / soft + 0.5, 0, 1)
                # the fog comes back unevenly, the way condensation forms
                prof *= np.clip((fade[:, None] - 0.35 * self.mottle[y0:y1, x0:x1]) * 1.6, 0, 1)
                np.maximum(clr[y0:y1, x0:x1], prof.astype(np.float32), out=clr[y0:y1, x0:x1])
                rects.append((y0, y1, x0, x1))
        return rects

    def _drops(self, tau):
        """Entries of the drops that move or come and go: (small ones, running drops)."""
        cols = [[] for _ in range(6)]  # x, y, radius across, radius down, taper, vis

        def add(*a):
            for c, x in zip(cols, a):
                c.append(np.asarray(x, np.float32).ravel())

        if self.h_n:  # rain hitting the glass
            z = tau * self.h_c + self.h_phi
            cyc = np.floor(z)
            s = z - cyc
            v = np.mod(cyc, self.h_c).astype(int) % self.VARIANTS
            i = np.arange(self.h_n)
            sa, sf = self.h_sa[i, v], self.h_sf[i, v]
            age = (s - sa) * self.loop_s / self.h_c
            grow = 1 - np.exp(-np.maximum(age, 0) / 0.05)
            fade = 1 - _smooth((s - sf) / 0.08)
            k = np.flatnonzero((s > sa) & (s < sf + 0.08))
            if len(k):
                ik, vk, gk = i[k], v[k], grow[k]
                R = self.h_R[ik, vk] * gk
                add(self.h_x[ik, vk], self.h_y[ik, vk], R, R, np.full(len(k), 0.06), fade[k])
                ox, oy, rr, there = (a[ik, vk] for a in self.h_sat)
                there = there & (gk > 0.3)[:, None]
                sr = (rr * gk[:, None])[there]
                add((self.h_x[ik, vk][:, None] + ox)[there], (self.h_y[ik, vk][:, None] + oy)[there], sr, sr,
                    np.zeros(len(sr)), np.broadcast_to(fade[k][:, None], there.shape)[there])
        if self.r_n:  # tiny beads left in the trails
            i, cur, prev, s_all = self._slots(tau)
            for v, s in ((cur, s_all), (prev, s_all + 1)):
                f, side, rad, there = (a[i, v] for a in self.r_tb)
                y = self.r_y0[i, v][:, None] + f * self.r_L[i, v][:, None]
                uu = np.empty_like(f)
                for j in range(len(i)):
                    uu[j] = np.interp(f[j], self.r_g[i[j], v[j]], self.knots)
                s_at = self.r_sr[i, v][:, None] + uu * (self.r_se[i, v] - self.r_sr[i, v])[:, None]
                age = (s[:, None] - s_at) * self.r_T[i][:, None]
                tau_r = self.r_tau[i][:, None]
                life = np.where(age > 0, 1 - _smooth((age - 0.7 * tau_r) / (0.3 * tau_r)), 0) * _smooth(age / 0.15)
                ok = there & (life > 0.01) & (y < self.h + 4)
                if ok.any():
                    ii = np.broadcast_to(i[:, None], ok.shape)[ok]
                    vv = np.broadcast_to(v[:, None], ok.shape)[ok]
                    yy = y[ok]
                    xx = self._x(ii, vv, yy) + side[ok] * self.r_R[ii, vv] * self.r_wf[ii, vv]
                    add(xx, yy, rad[ok], rad[ok], np.full(len(xx), 0.05), life[ok])
        small = self._batch(cols, 0.3)
        cols = [[] for _ in range(6)]
        if self.r_n:  # the running drops
            i, v, _, s = self._slots(tau)
            y = self._pos(i, v, s)
            dt = 0.08  # seconds: the speed over a short while, so the shape does not jump
            speed = (self._pos(i, v, s + dt / self.r_T) - self._pos(i, v, s - dt / self.r_T)) / (2 * dt)
            R = self.r_R[i, v]
            grow = np.clip(s / self.r_sg[i, v], 0, 1)
            sf = self.r_sf[i, v]
            fade = 1 - _smooth((s - sf) / np.maximum(0.985 - sf, 1e-3))
            vis = _smooth(grow * 4) * fade
            # it grows as water gathers, is a little smaller once it has left water in its trail, and shrinks away
            # at the end of its cycle
            Rn = R * np.sqrt(0.08 + 0.92 * _smooth(grow)) * np.where(s >= self.r_se[i, v], 0.88, 1.0) * (
                0.45 + 0.55 * fade)
            el = 0.4 * np.clip(speed / (140 * self.px), 0, 1)
            a, b = Rn * (1 - 0.12 * el), Rn * (1 + el)
            x = self._x(i, v, np.maximum(y, self.r_y0[i, v]))
            ok = (vis > 0.01) & (y - 2 * b < self.h + 2)
            add(x[ok], (y + Rn - b)[ok], a[ok], b[ok], (0.12 + 0.6 * el)[ok], vis[ok])
        return small, self._batch(cols, 0.2)

    def _batch(self, cols, ring):
        if not cols[0]:
            return {"idx": np.zeros(0, np.int64), "id": np.zeros(0, np.int64), "f": np.zeros((8, 0), np.float32)}
        cx, cy, rx, ry, tp, vis = (np.concatenate(c) for c in cols)
        keep = (rx > 0.15) & (vis > 0.01) & (cx > -40) & (cx < self.w + 40) & (cy > -60) & (cy < self.h + 60)
        cx, cy, rx, ry, tp, vis = cx[keep], cy[keep], rx[keep], ry[keep], tp[keep], vis[keep]
        return self._raster(cx, cy, rx, ry, tp, np.zeros(len(cx), np.float32), vis,
                            ring + 0.6 / np.maximum(rx, 0.5))

    # --- laying it on the picture ----------------------------------------------------------------------------

    def _box(self, mask):
        """The part of the frame the effect may change (the area's bounding box), or None if nowhere."""
        if mask is None:
            return 0, 0, self.w, self.h
        if self._box_of[0] is mask:
            return self._box_of[1]
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        box = None if not len(rows) else (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
        self._box_of = (mask, box)
        return box

    def _veil(self, box, preview):
        """The fog in the box (0..1, drops and trails cleared), and the cover of the milky veil laid over it."""
        x0, y0, x1, y1 = box
        fog = self.fog[y0:y1, x0:x1] * (1 - self.buf[CLR, y0:y1, x0:x1])
        a = self.veil * fog
        if preview:  # no blur in the preview: a little more veil instead
            a += (1 - a) * (0.22 * (0.4 + 0.6 * self.blur_k)) * np.sqrt(np.minimum(1, fog * 1.6))
        return fog, a

    def _dots(self, box):
        """The drops in the box: row and column of every pixel they cover, and the buffer's planes there."""
        x0, y0, x1, y1 = box
        b = self.buf[:, y0:y1, x0:x1]
        yy, xx = np.nonzero(b[A] > 1 / 1024)
        return yy, xx, b[:, yy, xx]

    def draw(self, t, canvas, mask):
        """What it lays over the picture (the page's live preview): veil, clear trails, rims and glints."""
        box = self._box(mask)
        if box is None or self.opacity <= 0:
            return
        self._fields(t)
        _, a = self._veil(box, True)
        rgb = a[..., None] * self.colour
        yy, xx, f = self._dots(box)
        if len(yy):
            c, ca = rgb[yy, xx], a[yy, xx]
            sky = self.lens_k * f[SKY]  # the sky seen upside down: lighter low in the drop, a little darker high up
            for alpha, white in ((f[RIM], False), (0.1 * np.maximum(-sky, 0), False),
                                 (0.3 * np.maximum(sky, 0), True), (f[HL], True)):  # black or white, laid over
                c *= (1 - alpha)[:, None]
                if white:
                    c += alpha[:, None]
                ca = ca * (1 - alpha) + alpha
            rgb[yy, xx], a[yy, xx] = c, ca
        if self.opacity < 1:
            rgb *= self.opacity
            a *= self.opacity
        canvas.over_rgba(box[0], box[1], rgb, a, mask)

    def apply(self, t, picture, mask, gain):
        """The whole effect on `picture` (float32 RGB 0..1, changed in place) where `mask` (0..1, None: everywhere)
        says, `gain` (0..1, a fade) times as strongly."""
        k = gain * self.opacity
        box = self._box(mask)
        if k <= 0 or box is None:
            return
        x0, y0, x1, y1 = box
        bw, bh = x1 - x0, y1 - y0
        self._fields(t)
        pic = picture[y0:y1, x0:x1]
        # the picture and its halves: the scene squeezed into a drop is read from the level that fits
        levels = [pic]
        if bw >= 2 and bh >= 2:
            hw2, hh2 = bw // 2 * 2, bh // 2 * 2
            levels.append((pic[0:hh2:2, 0:hw2:2] + pic[1:hh2:2, 0:hw2:2] + pic[0:hh2:2, 1:hw2:2]
                           + pic[1:hh2:2, 1:hw2:2]) * 0.25)
        else:
            levels.append(pic.copy())
        ims = [None, Image.fromarray((np.clip(levels[1], 0, 1) * 255 + 0.5).astype(np.uint8))]
        while len(ims) < LEVELS:
            ims.append(ims[-1].reduce(2) if min(ims[-1].size) >= 2 else ims[-1])
        levels += [None] * (LEVELS - len(levels))
        wide = None
        if self.blur_r >= 0.5:  # fogged glass spreads the light: a blur, with a wide glow
            lv = int(np.clip(math.floor(math.log2(max(self.blur_r / 3, 2))), 1, LEVELS - 1))
            src = ims[lv]
            r = self.blur_r * src.size[0] / bw
            wide = Image.blend(src.filter(ImageFilter.GaussianBlur(r)), src.filter(ImageFilter.GaussianBlur(r * 2.5)),
                               0.4)
        yy, xx, f = self._dots(box)  # (row by row)
        base = pic[yy, xx]
        glass, veil_at = base.copy(), np.zeros(len(yy), np.float32)  # the fogged glass under each drop pixel
        # the glass where it is fogged: picture * q1 + blurred * q2 + colour * q3 (the veil), in bands of rows
        band = max(8, (1 << 18) // bw)
        starts = np.searchsorted(yy, np.arange(0, bh + band, band))
        for n, r0 in enumerate(range(0, bh, band)):
            r1 = min(bh, r0 + band)
            fog, veil = self._veil((x0, y0 + r0, x1, y0 + r1), False)
            kk = k if mask is None else k * mask[y0 + r0:y0 + r1, x0:x1]
            q3 = kk * veil
            p = pic[r0:r1]
            d0, d1 = starts[n], starts[n + 1]
            ry, rx = yy[d0:d1] - r0, xx[d0:d1]
            veil_at[d0:d1] = veil[ry, rx]
            if wide is not None:
                sy = wide.size[1] / bh
                blurred = np.asarray(wide.resize((bw, r1 - r0), Image.Resampling.BILINEAR,
                                                 box=(0, r0 * sy, wide.size[0], r1 * sy))).astype(np.float32)
                wgt = self.blur_k ** 0.7 * np.sqrt(np.minimum(1, fog * 1.8))
                g = glass[d0:d1]
                g += wgt[ry, rx, None] * (blurred[ry, rx] * (1 / 255.0) - g)
                q2 = kk * wgt * (1 - veil)
                blurred *= (q2 * (1 / 255.0))[..., None]
                p *= (1 - q2 - q3)[..., None]
                p += blurred
            else:
                p *= (1 - q3)[..., None]
            p += q3[..., None] * self.colour
        if not len(yy):
            return
        # the drops, pixel by pixel: the scene upside down and smaller, rim, glint, then the veil
        sx, sy = xx + 0.5 + f[DX], yy + 0.5 + f[DY]
        lod = f[LOD]
        lo = lod.astype(np.int32)
        seen = np.empty((len(yy), 3), np.float32)
        for lv in np.unique(lo):
            sel = np.flatnonzero(lo == lv)
            pair = []
            for k2 in (lv, lv + 1):
                if levels[k2] is None:
                    levels[k2] = np.asarray(ims[k2], np.float32) * (1 / 255.0)
                sc = levels[k2].shape[1] / bw
                pair.append(_bilinear(levels[k2], sx[sel] * sc, sy[sel] * sc))
            seen[sel] = pair[0] + (pair[1] - pair[0]) * (lod[sel] - lv)[:, None]
        out = glass + f[A][:, None] * (seen - glass)
        out *= (1 - f[RIM])[:, None]
        # glints are brighter where there is more light around (the deepest level: the light of a wide area)
        sc = ims[-1].size[0] / bw
        near = _bilinear(np.asarray(ims[-1], np.float32) * (1 / 255.0), (xx + 0.5) * sc, (yy + 0.5) * sc) @ _LUMA
        hl = np.minimum(1.0, f[HL] * (0.55 + 1.1 * near))[:, None]
        out *= 1 - hl
        out += hl
        out += veil_at[:, None] * (self.colour - out)
        kd = k if mask is None else k * mask[y0 + yy, x0 + xx][:, None]
        pic[yy, xx] = base + kd * (out - base)
