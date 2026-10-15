"""Water on a window pane: fogged glass that blurs and whitens what is behind it, beads of water that stay, bigger
drops that hold, then run down in fits and starts leaving a clear trail that slowly fogs over again, and new drops
where rain hits the glass.

A lens effect: apply() changes the picture under it (the renderer: export and exact frames). Fogged glass spreads
the light of the scene behind it (a blur whose radius grows with how fogged it is, with a wide glow) under a thin
milky veil; the scene may also be out of focus everywhere (the camera focused on the glass). Through every drop the
scene is seen sharp, upside down and smaller (a wide view squeezed into the drop), with a darker rim and a small
bright glint. draw() is only what is laid over the picture (the page's live preview): the veil with its clear
trails, the rims, glints and a hint of the upside-down sky in each drop.

Like every effect it loops (see effects.py). A running drop's slot goes round a whole number of cycles per loop and
its cycles are chained: a drop grows, holds, runs, rests, and the next cycle carries it on from where it stopped,
until the chain's last cycle runs it off the bottom, so no drop vanishes in place. Trails' ages wrap with the cycle,
beads wiped by a drop grow back as a function of the time since, modulo the loop. Sizes are pixels of a 1080-line
frame.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .effects import Effect, _choice, _colour, _lerp, _num, _rgb

DEFAULTS = {"haze": 45, "blur": 50, "defocus": 0, "uneven": 40, "refog": 6, "beads": 50, "bead_size": 40,
            "drops": 35, "size": 45, "speed": 40, "wiggle": 40, "hits": 30, "refraction": 50, "highlight": 50,
            "rim": 50}
LOOKS = {
    "rain": {},
    "mist": {"haze": 30, "blur": 35, "uneven": 55, "refog": 9, "beads": 30, "bead_size": 25, "drops": 10,
             "size": 35, "speed": 30, "hits": 8, "highlight": 40, "rim": 40},
    "steamy": {"haze": 80, "blur": 75, "uneven": 50, "refog": 12, "beads": 80, "bead_size": 45, "drops": 25,
               "size": 50, "speed": 35, "hits": 12, "rim": 55},
    "storm": {"haze": 30, "blur": 45, "defocus": 15, "uneven": 35, "refog": 4, "beads": 55, "bead_size": 50,
              "drops": 75, "size": 60, "speed": 70, "wiggle": 55, "hits": 85, "highlight": 60},
    "clear": {"haze": 8, "blur": 60, "defocus": 70, "uneven": 20, "refog": 5, "beads": 65, "bead_size": 55,
              "drops": 40, "size": 55, "speed": 45, "hits": 45, "refraction": 60, "highlight": 60, "rim": 55},
}

SCHEMA = {"label": "Nước trên kính", "params": [
    _choice("look", "Kiểu", "rain", [("rain", "Mưa trên kính"), ("mist", "Hơi nước nhẹ"),
                                    ("steamy", "Kính đọng nhiều hơi nước"), ("storm", "Mưa to, nước chảy nhiều"),
                                    ("clear", "Kính trong, cảnh ngoài nhoè, giọt mưa rõ nét")],
            presets={k: {**DEFAULTS, **v} for k, v in LOOKS.items()}, group="Kiểu"),
    _num("haze", "Hơi nước (kính mờ)", 45, group="Hơi nước"),
    _num("blur", "Nhoè cảnh sau kính", 50, group="Hơi nước"),
    _num("defocus", "Nhoè cả chỗ kính trong", 0, group="Hơi nước"),
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
_NONE = {"idx": np.zeros(0, np.int64), "id": np.zeros(0, np.int64), "f": np.zeros((8, 0), np.float32)}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smooth_(x):
    """_smooth, in place (float32 arrays)."""
    np.clip(x, 0.0, 1.0, out=x)
    sq = x * x
    x *= -2.0
    x += 3.0
    x *= sq
    return x


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


def _split(total, share, cap):
    """`total` shared out in proportion to `share`, no part above its `cap` (sum(cap) >= total)."""
    part = total * share / share.sum()
    for _ in range(len(share)):
        over = part > cap
        if not over.any():
            break
        extra = float((part - cap)[over].sum())
        part[over] = cap[over]
        free = part < cap
        if not free.any():
            break
        part[free] += extra * share[free] / share[free].sum()
    return part


class Glass(Effect):
    lens = True
    VERSION = 2  # (1: the first release drew only the haze)
    RUNS = 64  # running-drop slots with fixed random draws
    HITS = 200  # rain-hit slots
    VARIANTS = 64  # a slot's cycles in one loop, at most
    KNOTS = 16  # the stop-and-go speed table of a run

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        self.opacity = p["opacity"] / 100
        self.colour = _rgb(p["color"])
        self.veil = 0.3  # the milky veil of fully fogged glass
        self.blur_k = p["blur"] / 100
        self.blur_r = 24.0 * px * self.blur_k  # gaussian radius of the scene behind fogged glass
        self.defocus = p["defocus"] / 100 * min(1.0, 2 * self.blur_k)  # out of focus even where the glass is clear
        refr = p["refraction"] / 100
        self.lens_k = min(1.0, refr * 4)  # 0: the scene goes straight through a drop
        self.view = _lerp(30.0, 320.0, refr ** 1.5) * px  # how far around it a drop sees (pixels of the scene)
        self.hl_k = p["highlight"] / 100 * 1.5
        self.rim_k = p["rim"] / 100 * 0.6
        self.bbox = self.buf = None  # the drop buffer covers only the area's box (made on first use)
        self.dirty = None  # (entries, trail rectangles) of the last frame, cleared before the next
        self._box_of = (None, None)
        self.fog = self._fog_map()
        self._beads()
        self._runs()
        self._hits()
        self._wipes()
        self._takes()

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
        self.mottle = self._noise(g, 120)  # where a trail fogs over first
        self.mottle *= 0.35
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
                           0.25 + 0.6 / np.maximum(r, 0.5), box=(0, 0, self.w, self.h))
        # where a bead's ring reaches into a neighbour, the stronger entry stays
        order = np.lexsort((ent["f"][A] * 2 + ent["f"][CLR], ent["idx"]))
        idx = ent["idx"][order]
        sel = order[np.r_[idx[1:] != idx[:-1], True]] if len(idx) else order
        self.beads_all = (ent["idx"][sel], ent["id"][sel], ent["f"][:, sel])  # (whole frame)

    # --- drops: one drawing routine for all ------------------------------------------------------------------

    def _raster(self, cx, cy, rx, ry, taper, rot, vis, ring, opn=None, box=None):
        """Drops (centre, radii across and down, pixels; top narrowed by `taper`; turned by `rot`; `vis` 0..1; fog
        cleared out to `ring` times the radius beyond its edge; `opn` 0..1: its top opens into the trail above it)
        as sparse entries in `box` (the buffer's by default): flat pixel index there, drop number and the buffer's
        planes."""
        bx0, by0, bx1, by1 = box or self.bbox
        bw = bx1 - bx0
        reff = np.sqrt(rx * ry)
        reach = np.maximum(rx, ry) * (1 + ring) + 1.5
        which = np.minimum(np.searchsorted(_HALVES, reach), len(_HALVES) - 1)
        if opn is None:
            opn = np.zeros(len(cx), np.float32)
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
            keep = (clr > 0.002) & (ix >= bx0) & (ix < bx1) & (iy >= by0) & (iy < by1)
            if not keep.any():
                continue
            pick = lambda a: np.broadcast_to(a, keep.shape)[keep]  # noqa: E731
            q, u, v, dx, dy = pick(q), pick(u), pick(v), pick(dx), pick(dy)
            re, vi, rxs, rys, op = pick(col(reff)), pick(vi), pick(col(rx)), pick(col(ry)), pick(col(opn))
            cov = np.clip((1 - q) * re + 0.5, 0, 1) * vi
            # the scene through the drop: upside down, squeezed more towards the edge
            m0 = np.clip(self.view / re, 1.0, 120.0)
            out = 1 + m0 * (1 + 0.5 * q * q)
            # the glint: a soft spot up and to the left, never sharper than a pixel (a tiny bead is all glint, so it
            # sparkles rather than speckles); light gathered at the bottom
            sig = np.maximum(0.16 * re, 0.6)
            amp = np.clip(0.267 * re, 0.5, 1.0)
            gx_, gy_ = dx + 0.32 * rxs, dy + 0.4 * rys
            spot = amp * np.exp(-(gx_ * gx_ + gy_ * gy_) / (2 * sig * sig))
            caustic = 0.3 * amp * np.exp(-((q - 0.74) / 0.15) ** 2) * np.clip(v, 0, 1) ** 2
            # a running drop's top melts into its trail: no rim there, and the water thins out
            top = op * _smooth((-v - 0.1) / 0.6)
            f = np.empty((8, len(q)), np.float32)
            f[RIM] = self.rim_k * np.clip(re / 4.0, 0.2, 1) * np.minimum(1, re / 3.0) * _smooth(
                (q - 0.6) / 0.4) ** 1.3 * (0.75 + 0.3 * np.clip(v, -1, 1)) * cov * (1 - top)
            cov = cov * (1 - 0.7 * op * _smooth((-v - 0.35) / 0.6))
            f[A] = cov
            f[DX], f[DY] = -dx * out * self.lens_k, -dy * out * self.lens_k
            f[HL] = np.minimum(1.0, self.hl_k * (spot + caustic)) * cov
            f[CLR] = pick(clr)
            f[SKY] = np.clip(v, -1, 1) * cov
            f[LOD] = np.clip(np.log2(1 + m0 * (1 + 1.5 * q * q)) * self.lens_k - 0.5, 0, LEVELS - 1.001)
            idx_l.append((pick(iy) - by0) * bw + pick(ix) - bx0)
            id_l.append(np.broadcast_to(sel[:, None, None], keep.shape)[keep])
            f_l.append(f)
        if not idx_l:
            return _NONE
        return {"idx": np.concatenate(idx_l), "id": np.concatenate(id_l), "f": np.concatenate(f_l, axis=1)}

    # --- running drops ---------------------------------------------------------------------------------------

    def _runs(self):
        """Slots of drops that grow, hold, run down in fits and starts and leave a trail. A slot goes round a whole
        number of cycles per loop, in chains of one to three: a new drop grows from nothing; each cycle it holds,
        runs, rests; the next cycle carries it on from there; the chain's last cycle runs it off the bottom."""
        p, px, h, w = self.p, self.px, self.h, self.w
        K, V = self.KNOTS, self.VARIANTS
        g = self._rng(20)
        us = g.random((self.RUNS, 2))
        uv = g.random((self.RUNS, V, 18))
        ug = g.random((self.RUNS, V, K))
        stops = g.random((self.RUNS, V, K)) < 0.38
        ub = g.random((self.RUNS, V, 10, 4))
        jog = g.standard_normal((self.RUNS, V, K))
        self.r_rip = g.random((self.RUNS, 2, 64)).astype(np.float32)  # how uneven a trail's width is, down it
        n = min(self.RUNS, int(round(p["drops"] / 100 * 36 * self.aspect / (16 / 9))))
        size, speed, wig = p["size"] / 100, p["speed"] / 100, p["wiggle"] / 100
        refog = float(p["refog"])
        want = _lerp(5.0, 11.0, us[:n, 0]) * _lerp(1.3, 0.75, speed)
        c = np.clip(np.floor(self.loop_s / np.maximum(want, refog)), 1, V)
        T = self.loop_s / c
        self.r_n, self.r_c, self.r_phi, self.r_T = n, c, us[:n, 1], T
        self.r_tau = np.minimum(refog, T)  # a trail is fogged over again before the slot's next-but-one cycle
        u = uv[:n]
        # stop and go: each of KNOTS stretches is a stop (crawling) or a dash
        wgt = np.where(stops[:n], 0.03, 0.2 + ug[:n] ** 0.5)
        wgt = 0.25 * np.roll(wgt, 1, -1) + 0.5 * wgt + 0.25 * np.roll(wgt, -1, -1)
        cum = np.concatenate([np.zeros(wgt.shape[:-1] + (1,)), np.cumsum(wgt, -1)], -1)
        self.r_g = cum / cum[..., -1:]
        self.knots = np.linspace(0, 1, K + 1)
        self.mid = (self.knots[:-1] + self.knots[1:]) / 2
        v_avg = _lerp(30.0, 240.0, speed) * (0.7 + 0.6 * u[..., 6]) * px
        # a trail is a little narrower than its drop (much narrower only on nearly clear glass)
        self.r_wf = _lerp(0.62, 0.85, u[..., 14]) * _lerp(0.6, 1.0, min(1.0, p["haze"] / 40))
        # a gentle long sway, and a turn wherever the drop paused
        self.r_amp = wig * _lerp(1.0, 4.0, u[..., 7]) * px
        self.r_lam = _lerp(150.0, 320.0, u[..., 8]) * px
        self.r_ph = u[..., 9] * 2 * math.pi
        turn = 0.13 * wig
        shape = (n, V)
        R, y0, L = np.zeros(shape), np.zeros(shape), np.ones(shape)
        sg, sr, se = np.zeros(shape), np.zeros(shape), np.ones(shape)
        new, last = np.zeros(shape, bool), np.zeros(shape, bool)
        yk, xk = np.zeros(shape + (K + 1,)), np.zeros(shape + (K + 1,))
        for i in range(n):
            ci, Ti, m0 = int(c[i]), float(T[i]), 0
            while m0 < ci:
                uc = u[i, m0]
                Rc = _lerp(4.0, 15.0, size) * (0.75 + 0.5 * uc[0]) * px
                y = (-0.04 + 0.8 * uc[2] ** 1.4) * h
                js = np.arange(m0, ci)
                # seconds: growing (a new drop), holding, at least this much rest after a run that is not the last
                t_g = np.where(js == m0, np.minimum(_lerp(0.5, 1.1, u[i, js, 4]), 0.2 * Ti), 0.0)
                t_h = np.where(js == m0, np.minimum(_lerp(0.3, 2.5, u[i, js, 5]), 0.3 * Ti),
                               np.minimum(_lerp(0.2, 1.2, u[i, js, 5]), 0.2 * Ti))
                rest = np.full(len(js), min(0.8, 0.15 * Ti))
                # as many cycles as it takes to run off the bottom from there (sometimes one more), if there are
                k = 1 + int(np.searchsorted(np.cumsum(v_avg[i, js] * (0.97 * Ti - t_g - t_h - rest) * 0.8),
                                            h + 3 * Rc - y)) + int(uc[16] < 0.3)
                k = min(len(js), k)
                js, t_g, t_h, rest = js[:k], t_g[:k], t_h[:k], rest[:k]
                rest[-1] = 0.0
                room = np.maximum(0.97 * Ti - t_g - t_h - rest, 0.05 * Ti)
                cap = v_avg[i, js] * room
                y = max(y, h + 3 * Rc - 0.95 * cap.sum())
                parts = _split(h + 3 * Rc - y, _lerp(0.5, 1.5, u[i, js, 3]), cap)
                x = -Rc + uc[1] * (w + 2 * Rc)
                d = slant = (uc[13] - 0.5) * 0.12 * wig
                for a, j in enumerate(js):
                    R[i, j], y0[i, j], L[i, j] = Rc, y, parts[a]
                    new[i, j], last[i, j] = a == 0, a == k - 1
                    sg[i, j], sr[i, j] = t_g[a] / Ti, (t_g[a] + t_h[a]) / Ti
                    se[i, j] = sr[i, j] + min(parts[a] / v_avg[i, j], room[a]) / Ti
                    yk[i, j] = y + parts[a] * self.r_g[i, j]
                    dirs = np.empty(K)
                    for kk in range(K):  # the way it runs (across per down) turns after every pause
                        if kk == 0 or stops[i, j, kk - 1]:
                            d = float(np.clip(slant + 0.6 * (d - slant) + turn * jog[i, j, kk], -0.35, 0.35))
                        dirs[kk] = d
                    xk[i, j] = x + np.concatenate([[0.0], np.cumsum(dirs * np.diff(yk[i, j]))])
                    x = xk[i, j, -1] + self._sway(i, j, parts[a])
                    y += parts[a]
                m0 += k
        self.r_R, self.r_y0, self.r_L = R, y0, L
        self.r_sg, self.r_sr, self.r_se, self.r_new, self.r_last = sg, sr, se, new, last
        self.r_yk, self.r_xk = yk, xk
        self.r_reach = np.abs(xk - xk[..., :1]).max(-1) + 2 * self.r_amp
        # tiny beads left in the trail: where along it, how far to the side, size, whether there
        b = ub[:n]
        self.r_tb = (b[..., 0], (b[..., 1] - 0.5) * 1.4,
                     _lerp(0.6, 1.6, b[..., 2] ** 1.5) * px * np.sqrt(R / (8 * px))[..., None],
                     b[..., 3] < np.clip(L / (70 * px), 0.5, 10)[..., None] / 10)

    def _sway(self, i, v, d):
        """The gentle long sway of slot i's run v, `d` pixels below where it set off (0 there)."""
        a, lam, ph = self.r_amp[i, v], self.r_lam[i, v], self.r_ph[i, v]
        return a * (np.sin(2 * math.pi * d / lam + ph) - np.sin(ph))

    def _pos(self, i, v, s):
        """Down the path: y of slots i on runs v at cycle phase s (arrays)."""
        sr, se = self.r_sr[i, v], self.r_se[i, v]
        uu = np.clip((s - sr) / np.maximum(se - sr, 1e-6), 0, 1) * self.KNOTS
        k = np.minimum(uu.astype(int), self.KNOTS - 1)
        yk = self.r_yk[i, v]
        y0 = np.take_along_axis(yk, k[:, None], 1)[:, 0]
        y1 = np.take_along_axis(yk, k[:, None] + 1, 1)[:, 0]
        return y0 + (y1 - y0) * (uu - k)

    def _x(self, i, v, y):
        """Across: the path's x at height(s) y (arrays; the trail follows it)."""
        yk, xk = self.r_yk[i, v], self.r_xk[i, v]
        y = np.clip(y, yk[:, 0], yk[:, -1])
        k = np.minimum((yk[:, 1:-1] <= y[:, None]).sum(1), self.KNOTS - 1)[:, None]
        ya, yb = np.take_along_axis(yk, k, 1)[:, 0], np.take_along_axis(yk, k + 1, 1)[:, 0]
        xa, xb = np.take_along_axis(xk, k, 1)[:, 0], np.take_along_axis(xk, k + 1, 1)[:, 0]
        f = (y - ya) / np.maximum(yb - ya, 1e-6)
        return xa + (xb - xa) * f + self._sway(i, v, y - yk[:, 0])

    def _phase_at(self, i, v, y, speed=False):
        """The cycle phase at which the drop of slot i on run v left height(s) y (and if `speed`, how fast it went
        there, 1 = its average, changing smoothly)."""
        uu = np.interp(y, self.r_yk[i, v], self.knots)
        at = self.r_sr[i, v] + uu * (self.r_se[i, v] - self.r_sr[i, v])
        if not speed:
            return at
        return at, np.interp(uu, self.mid, np.diff(self.r_g[i, v]) * self.KNOTS)

    def _ripple(self, i, y):
        """How much wider or narrower slot i's trails are at height(s) y: smooth noise, the same each pass."""
        out = np.zeros_like(y)
        for k, (scale, amp) in enumerate(((44.0, 0.2), (17.0, 0.09))):
            z = y / (scale * self.px)
            z0 = np.floor(z)
            j = z0.astype(np.int64) % 64
            a, b = self.r_rip[i, k][j], self.r_rip[i, k][(j + 1) % 64]
            out += amp * (a + (b - a) * _smooth(z - z0) - 0.5)
        return out

    # --- rain hitting the glass ------------------------------------------------------------------------------

    def _hits(self):
        """Slots of new drops: each cycle one lands somewhere (with a few tiny splashes) and stays, until it slowly
        dries up near the end of the cycle or a running drop takes it in."""
        p, px = self.p, self.px
        g = self._rng(30)
        us = g.random((self.HITS, 3))
        uv = g.random((self.HITS, self.VARIANTS, 8))
        usat = g.random((self.HITS, self.VARIANTS, 3, 4))
        T0 = _lerp(6.0, 15.0, us[:, 0])
        rate = p["hits"] / 100 * 20 * self.aspect / (16 / 9)  # new drops a second
        n = min(self.HITS, int(np.searchsorted(np.cumsum(1 / T0), rate)) if rate > 0 else 0)
        self.h_n = n
        self.h_c = np.minimum(self.cycles(1 / T0[:n]), self.VARIANTS)
        self.h_T = T = self.loop_s / self.h_c
        self.h_phi = us[:n, 1]
        u = uv[:n]
        size = (p["size"] + p["bead_size"]) / 200
        self.h_x, self.h_y = u[..., 0] * self.w, u[..., 1] * self.h
        self.h_R = _lerp(2.0, 8.0, size) * (0.45 + 0.55 * u[..., 2] ** 1.3) * px
        self.h_sa = 0.1 * u[..., 3]
        self.h_ev = np.minimum(_lerp(1.6, 2.6, u[..., 4]), 0.35 * T[:, None]) / T[:, None]  # drying up (phase)
        self.h_end = 0.98 - self.h_ev  # when it starts to dry up
        self.h_take = np.full(self.h_x.shape, np.inf)  # when a running drop takes it in (set by _takes)
        s = usat[:n]
        ang, dist = s[..., 0] * 2 * math.pi, _lerp(1.6, 3.2, s[..., 1])
        R = self.h_R[..., None]
        self.h_sat = (np.cos(ang) * dist * R, np.sin(ang) * dist * R, _lerp(0.22, 0.42, s[..., 2]) * R,
                      s[..., 3] < 0.55)

    def _runs_over(self, px_, py_, pr, wide):
        """For every run that passes over points (px_, py_) of radius pr: (point indices, loop phase at which the
        drop's centre passes, slot). `wide`: how far to the side it reaches, times its radius."""
        order = np.argsort(px_)
        sx = px_[order]
        for i in range(self.r_n):
            c = int(self.r_c[i])
            for v in range(c):
                y0, L, R = self.r_y0[i, v], self.r_L[i, v], self.r_R[i, v]
                x0, reach = self.r_xk[i, v, 0], self.r_reach[i, v] + wide * R + pr.max(initial=0) + 1
                cand = order[np.searchsorted(sx, x0 - reach):np.searchsorted(sx, x0 + reach)]
                cand = cand[(py_[cand] > y0 - R - pr[cand]) & (py_[cand] < y0 + L + R + pr[cand])]
                if not len(cand):
                    continue
                yy = np.clip(py_[cand], y0, y0 + L)
                xs = self._x(np.full(len(cand), i), np.full(len(cand), v), yy)
                cand = cand[np.abs(px_[cand] - xs) < wide * R + pr[cand]]
                if len(cand):
                    yield cand, i, v

    def _wipes(self):
        """When each bead is wiped by a running drop (loop phase), and how long it takes to come back."""
        bx, by, br = self.bead["cx"], self.bead["cy"], self.bead["r"]
        ids, at, back = [], [], []
        for cand, i, v in self._runs_over(bx, by, br, 1.0):
            yy = np.clip(by[cand], self.r_y0[i, v], self.r_y0[i, v] + self.r_L[i, v])
            ids.append(cand)
            at.append(np.mod((v + self._phase_at(i, v, yy) - self.r_phi[i]) / self.r_c[i], 1.0))
            back.append(np.full(len(cand), self.r_tau[i] / self.loop_s))
        self.w_id = np.zeros(0, np.int64)
        if ids:
            ids, at, back = np.concatenate(ids), np.concatenate(at), np.concatenate(back)
            order = np.argsort(ids, kind="stable")
            self.w_id, self.w_at, self.w_back = ids[order], at[order], back[order]
            self.w_start = np.flatnonzero(np.r_[True, self.w_id[1:] != self.w_id[:-1]])

    def _takes(self):
        """When a running drop takes in each rain hit before it dries up: as soon as the two touch (or at once, if it
        lands on a drop that is holding or resting there)."""
        if not (self.h_n and self.r_n):
            return
        hi = np.repeat(np.arange(self.h_n), self.h_c.astype(int))
        hv = np.concatenate([np.arange(int(c)) for c in self.h_c])
        hx, hy, hr = self.h_x[hi, hv], self.h_y[hi, hv], self.h_R[hi, hv]
        sa, c_h = self.h_sa[hi, hv], self.h_c[hi]
        start = np.mod((hv + sa - self.h_phi[hi]) / c_h, 1.0)  # loop phase it lands
        life = (self.h_end[hi, hv] + self.h_ev[hi, hv] - sa) / c_h
        best = np.full(len(hi), np.inf)
        for cand, i, v in self._runs_over(hx, hy, hr, 0.9):
            y0, L, R, c = self.r_y0[i, v], self.r_L[i, v], self.r_R[i, v], self.r_c[i]
            near = R + hr[cand]
            here = (self.r_sg[i, v] * 0.5) if self.r_new[i, v] else 0.0
            # cycle phases the drop covers the spot: from when its edge reaches it until it has gone past
            ca = np.where(hy[cand] - near <= y0, here, self._phase_at(i, v, hy[cand] - near))
            cb = np.where((hy[cand] + near >= y0 + L) & ~self.r_last[i, v], 1.0,
                          self._phase_at(i, v, np.minimum(hy[cand] + near, y0 + L)))
            ca_l = np.mod((v + ca - self.r_phi[i]) / c, 1.0)
            d = np.mod(ca_l - start[cand], 1.0)
            delay = np.where(d < life[cand], d, np.inf)
            delay = np.where(np.mod(start[cand] - ca_l, 1.0) < (cb - ca) / c, 0.0, delay)  # lands on the drop
            best[cand] = np.minimum(best[cand], delay)
        self.h_take[hi, hv] = sa + best * c_h

    # --- one frame -------------------------------------------------------------------------------------------

    def _use(self, box):
        """The drop buffer for `box` (made again when the area's box changes)."""
        if box == self.bbox:
            return
        x0, y0, x1, y1 = self.bbox = box
        self.buf = np.zeros((8, y1 - y0, x1 - x0), np.float32)
        self.flat = self.buf.reshape(8, -1)
        self.dirty = None
        idx, ids, f = self.beads_all
        yy, xx = np.divmod(idx, self.w)
        inside = (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        self.b_idx, self.b_id, self.b_f = (yy[inside] - y0) * (x1 - x0) + xx[inside] - x0, ids[inside], f[:, inside]

    def _fields(self, t, box=None):
        """Fills the drop buffer for frame t, in `box` (the whole frame by default)."""
        self._use(box or (0, 0, self.w, self.h))
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
        """Every running slot's run in this cycle and the one before, and the phase in this cycle."""
        z = tau * self.r_c + self.r_phi
        cyc = np.floor(z)
        cur = np.mod(cyc, self.r_c).astype(int)
        prev = np.mod(cyc - 1, self.r_c).astype(int)
        return np.arange(self.r_n), cur, prev, z - cyc

    def _trails(self, tau):
        """Clears the fog along every trail still showing; returns the rectangles touched (in the buffer)."""
        rects = []
        if not self.r_n:
            return rects
        bx0, by0, bx1, by1 = self.bbox
        i_all, cur, prev, s_all = self._slots(tau)
        clr = self.buf[CLR]
        soft = 1.2 * self.px + 0.5
        for i in i_all:
            T, tau_r = self.r_T[i], self.r_tau[i]
            for v, s in ((cur[i], s_all[i]), (prev[i], s_all[i] + 1)):
                sr, se = self.r_sr[i, v], self.r_se[i, v]
                s_head, s_tail = min(s, se), max(sr, s - tau_r / T)
                if s <= sr or s_tail >= s_head:
                    continue
                ii, vv = np.array([i]), np.array([v])
                y_head = self._pos(ii, vv, np.array([s_head]))[0]
                R, top = self.r_R[i, v], self.r_y0[i, v]
                y_cap = top - R  # the top of the drop where it set off: the trail's end is round there
                y_from = y_cap if s_tail <= sr else self._pos(ii, vv, np.array([s_tail]))[0] - 1
                y0 = max(by0, int(math.floor(y_from)))
                y1 = min(by1, int(math.ceil(y_head)) + 1)
                if y1 <= y0:
                    continue
                ys = np.arange(y0, y1) + 0.5
                yc = np.clip(ys, top, None)
                xc = self._x(np.full(len(ys), i), np.full(len(ys), v), yc)
                at, rel = self._phase_at(i, v, yc, True)
                fade = np.clip(1 - (s - at) * T / tau_r, 0, 1)
                # as wide as most of the drop, wider where it stopped, a little uneven; thinner as it fogs over
                hw = np.minimum(R * self.r_wf[i, v] * np.clip(1.2 - 0.25 * rel, 0.8, 1.15) * (1 + self._ripple(i, ys)),
                                0.98 * R) * (0.45 + 0.55 * np.sqrt(fade))
                hw = np.minimum(hw, np.sqrt(np.maximum(hw * hw - np.maximum(y_cap + hw - ys, 0) ** 2, 0)))
                hw, fade, xc = hw.astype(np.float32), fade.astype(np.float32), xc.astype(np.float32)
                for r0 in range(0, y1 - y0, 48):  # in short runs of rows, each only as wide as it needs
                    r1 = min(y1 - y0, r0 + 48)
                    x0 = max(bx0, int(math.floor((xc[r0:r1] - hw[r0:r1]).min())) - 1)
                    x1 = min(bx1, int(math.ceil((xc[r0:r1] + hw[r0:r1]).max())) + 2)
                    if x1 <= x0:
                        continue
                    xs = np.arange(x0, x1, dtype=np.float32) + 0.5
                    prof = np.abs(xs[None, :] - xc[r0:r1, None])
                    np.subtract(hw[r0:r1, None], prof, out=prof)
                    prof *= 1 / soft
                    prof += 0.5
                    np.clip(prof, 0, 1, out=prof)
                    # the fog comes back unevenly, the way condensation forms
                    m = fade[r0:r1, None] - self.mottle[y0 + r0:y0 + r1, x0:x1]
                    m *= 1.6
                    prof *= np.clip(m, 0, 1, out=m)
                    ry0, ry1, rx0, rx1 = y0 + r0 - by0, y0 + r1 - by0, x0 - bx0, x1 - bx0
                    np.maximum(clr[ry0:ry1, rx0:rx1], prof, out=clr[ry0:ry1, rx0:rx1])
                    rects.append((ry0, ry1, rx0, rx1))
        return rects

    def _running(self, tau):
        """The running drops at tau: x, y of the centre, radii across and down, taper, vis, how open the top is."""
        i, v, prev, s = self._slots(tau)
        y = self._pos(i, v, s)
        T = self.r_T
        dt = 0.08  # seconds: the speed over a short while, so the shape does not jump
        speed = (self._pos(i, v, s + dt / T) - self._pos(i, v, s - dt / T)) / (2 * dt)
        R, sg, sr, se = self.r_R[i, v], self.r_sg[i, v], self.r_sr[i, v], self.r_se[i, v]
        new, last = self.r_new[i, v], self.r_last[i, v]
        grow = np.where(new, np.clip(s / np.maximum(sg, 1e-6), 0, 1), 1.0)
        vis = _smooth(grow * 4)
        # a new drop grows as water gathers; it is a little smaller once it has left water in its trail, and grows
        # back while it rests (to where the next cycle starts it)
        run = _smooth((s - sr) / np.maximum(se - sr, 1e-6))
        back = np.where(last, 0.0, _smooth((s - se) / np.maximum(1 - se, 1e-6)))
        Rn = R * np.sqrt(0.08 + 0.92 * _smooth(grow)) * (1 - 0.08 * run + 0.08 * back)
        el = 0.4 * np.clip(speed / (140 * self.px), 0, 1)
        a, b = Rn * (1 - 0.12 * el), Rn * (1 + el)
        x = self._x(i, v, y)
        # how much of a trail there is just above it (its top melts into that)
        opn = np.zeros(len(i))
        for j in i:
            o = 0.0
            if s[j] > sr[j]:
                yt = max(y[j] - 0.8 * R[j], self.r_y0[j, v[j]])
                o = (1 - (s[j] - self._phase_at(j, v[j], yt)) * T[j] / self.r_tau[j]) * _smooth(
                    (y[j] - self.r_y0[j, v[j]]) / R[j])
            if not new[j]:
                o = max(o, 1 - (s[j] + 1 - self.r_se[j, prev[j]]) * T[j] / self.r_tau[j])
            opn[j] = np.clip((o - 0.15) * 1.6, 0, 1)
        return x, y + Rn - b, a, b, 0.12 + 0.6 * el, vis, opn

    def _drops(self, tau):
        """Entries of the drops that move or come and go: (small ones, running drops)."""
        cols = [[] for _ in range(7)]  # x, y, radius across, radius down, taper, vis, open top

        def add(*a):
            for c, x in zip(cols, a + (0.0,) * (7 - len(a))):
                c.append(np.broadcast_to(np.asarray(x, np.float32), np.shape(a[0])).ravel())

        if self.h_n:  # rain hitting the glass
            z = tau * self.h_c + self.h_phi
            cyc = np.floor(z)
            s = z - cyc
            v = np.mod(cyc, self.h_c).astype(int)
            i = np.arange(self.h_n)
            sa, end, ev, take = self.h_sa[i, v], self.h_end[i, v], self.h_ev[i, v], self.h_take[i, v]
            T = self.h_T
            age = (s - sa) * T
            grow = 1 - np.exp(-np.maximum(age, 0) / 0.05)
            dry = 1 - _smooth((s - end) / ev)  # it dries up: smaller and smaller
            vis = np.where(take > sa + 1e-9, 1 - _smooth((s - take) * T / 0.06), 0.0)  # taken in: gone at once
            k = np.flatnonzero((s > sa) & (dry > 0) & (vis > 0))
            if len(k):
                ik, vk = i[k], v[k]
                gk = grow[k] * np.sqrt(dry[k])
                R = self.h_R[ik, vk] * gk
                add(self.h_x[ik, vk], self.h_y[ik, vk], R, R, np.full(len(k), 0.06), vis[k])
                ox, oy, rr, there = (a[ik, vk] for a in self.h_sat)
                there = there & (grow[k] > 0.3)[:, None]
                sr = (rr * gk[:, None])[there]
                add((self.h_x[ik, vk][:, None] + ox)[there], (self.h_y[ik, vk][:, None] + oy)[there], sr, sr,
                    np.zeros(len(sr)), np.broadcast_to(vis[k][:, None], there.shape)[there])
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
        cols = [[] for _ in range(7)]
        if self.r_n:  # the running drops
            x, cy, a, b, taper, vis, opn = self._running(tau)
            ok = (vis > 0.01) & (cy - b < self.h + 2)
            add(x[ok], cy[ok], a[ok], b[ok], taper[ok], vis[ok], opn[ok])
        return small, self._batch(cols, 0.2)

    def _batch(self, cols, ring):
        if not cols[0]:
            return _NONE
        cx, cy, rx, ry, tp, vis, opn = (np.concatenate(c) for c in cols)
        x0, y0, x1, y1 = self.bbox
        keep = (rx > 0.15) & (vis > 0.01) & (cx > x0 - 40) & (cx < x1 + 40) & (cy > y0 - 60) & (cy < y1 + 60)
        cx, cy, rx, ry, tp, vis, opn = (a[keep] for a in (cx, cy, rx, ry, tp, vis, opn))
        return self._raster(cx, cy, rx, ry, tp, np.zeros(len(cx), np.float32), vis, ring + 0.6 / np.maximum(rx, 0.5),
                            opn)

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

    def _veil(self, box):
        """The fog in `box` (0..1, drops and trails cleared), and the cover of the milky veil laid over it."""
        x0, y0, x1, y1 = box
        bx0, by0 = self.bbox[:2]
        fog = self.fog[y0:y1, x0:x1] * (1 - self.buf[CLR, y0 - by0:y1 - by0, x0 - bx0:x1 - bx0])
        return fog, self.veil * fog

    def _dots(self, planes=None):
        """The drops in the buffer: row and column of every pixel they cover, and the buffer's planes there (only
        `planes`, if given)."""
        yy, xx = np.nonzero(self.buf[A] > 1 / 1024)
        if planes is None:
            return yy, xx, self.buf[:, yy, xx]
        i = yy * self.buf.shape[2] + xx
        return yy, xx, [self.flat[k][i] for k in planes]

    def draw(self, t, canvas, mask):
        """What it lays over the picture (the page's live preview): veil, clear trails, rims and glints."""
        box = self._box(mask)
        if box is None or self.opacity <= 0:
            return
        self._fields(t, box)
        fog, a = self._veil(box)
        lum = a
        if self.blur_r >= 0.5:  # no blur here: a faint grey haze where the export blurs (contrast lost, hardly
            haze = _smooth_(fog * 6)  # brighter), the veil over it
            haze *= 0.12 * min(1.0, 2 * self.blur_k)
            if self.defocus > 0:
                np.maximum(haze, 0.12 * float(_smooth(self.defocus * 3)), out=haze)
            haze -= haze * a
            lum = haze * 0.3
            lum += a
            a += haze
        rgb = np.empty(lum.shape + (3,), np.float32)
        for c in range(3):
            np.multiply(lum, self.colour[c], out=rgb[..., c])
        yy, xx, (rim, hl, sky) = self._dots([RIM, HL, SKY])
        if len(yy):
            c, ca = rgb[yy, xx], a[yy, xx]
            sky = self.lens_k * sky  # the sky seen upside down: lighter low in the drop, a little darker high up
            for alpha, white in ((rim, False), (0.05 * np.maximum(-sky, 0), False),
                                 (0.12 * np.maximum(sky, 0), True), (hl, True)):  # black or white, laid over
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
        self._fields(t, box)
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
        yy, xx, f = self._dots()  # (row by row)
        # what the drops see: the scene upside down and smaller (read now, before the glass is fogged below)
        sx, sy = xx + 0.5 + f[DX], yy + 0.5 + f[DY]
        lod = f[LOD]
        lo_i = lod.astype(np.int32)
        seen = np.empty((len(yy), 3), np.float32)
        for lv in np.unique(lo_i):
            sel = np.flatnonzero(lo_i == lv)
            pair = []
            for k2 in (lv, lv + 1):
                if levels[k2] is None:
                    levels[k2] = np.asarray(ims[k2], np.float32) * (1 / 255.0)
                sc = levels[k2].shape[1] / bw
                pair.append(_bilinear(levels[k2], sx[sel] * sc, sy[sel] * sc))
            seen[sel] = pair[0] + (pair[1] - pair[0]) * (lod[sel] - lv)[:, None]
        lo = hi = None
        if self.blur_r >= 0.5 and (self.defocus > 0 or self.p["haze"] > 0):
            # fogged glass spreads the light: blurred a little where thinly fogged, more (with a wide glow) where
            # thick; never the sharp scene mixed with a blurred one, so no crisp edge is left under a halo
            def blurred(r, glow):
                lv = int(np.clip(math.floor(math.log2(max(r / 3, 2))), 1, LEVELS - 1))
                src = ims[lv]
                rr = r * src.size[0] / bw
                im = src.filter(ImageFilter.GaussianBlur(rr))
                return Image.blend(im, src.filter(ImageFilter.GaussianBlur(rr * 2.5)), glow) if glow else im

            lo, hi = blurred(0.35 * self.blur_r, 0), blurred(self.blur_r, 0.25)
        base = pic[yy, xx]
        glass, veil_at = base.copy(), np.zeros(len(yy), np.float32)  # the fogged glass under each drop pixel
        # the glass: picture * (1 - q2 - q3) + blurred * q2 + colour * q3 (the veil), in bands of rows
        band = max(8, (1 << 18) // bw)
        starts = np.searchsorted(yy, np.arange(0, bh + band, band))
        to = min(1.0, 2 * self.blur_k)
        for n, r0 in enumerate(range(0, bh, band)):
            r1 = min(bh, r0 + band)
            fog, veil = self._veil((x0, y0 + r0, x1, y0 + r1))
            kk = k if mask is None else k * mask[y0 + r0:y0 + r1, x0:x1]
            q3 = kk * veil
            p = pic[r0:r1]
            d0, d1 = starts[n], starts[n + 1]
            ry, rx = yy[d0:d1] - r0, xx[d0:d1]
            veil_at[d0:d1] = veil[ry, rx]
            if lo is not None:
                sy = lo.size[1] / bh
                b = np.asarray(lo.resize((bw, r1 - r0), Image.Resampling.BILINEAR,
                                         box=(0, r0 * sy, lo.size[0], r1 * sy)), np.float32)
                sy = hi.size[1] / bh
                b2 = np.asarray(hi.resize((bw, r1 - r0), Image.Resampling.BILINEAR,
                                          box=(0, r0 * sy, hi.size[0], r1 * sy)), np.float32)
                m = _smooth_(fog * 2)  # how far towards the wide blur
                wgt = _smooth_(fog * 6)  # how much blurred: all of it once the glass is even lightly fogged
                wgt *= to
                if self.defocus > 0:
                    np.maximum(m, self.defocus, out=m)
                    np.maximum(wgt, float(_smooth(self.defocus * 3)), out=wgt)
                g = glass[d0:d1]
                q2 = veil - 1
                q2 *= wgt
                q2 *= -kk  # kk * wgt * (1 - veil)
                keep = 1 - q2
                keep -= q3
                for c in range(3):  # (channel by channel: much faster than broadcasting over 3)
                    bc = b[..., c]
                    d = b2[..., c]
                    d -= bc
                    d *= m
                    bc += d  # (0..255)
                    g[:, c] += wgt[ry, rx] * (bc[ry, rx] * (1 / 255.0) - g[:, c])
                    pc = p[..., c]
                    pc *= keep
                    bc *= q2
                    bc *= 1 / 255.0
                    pc += bc
                    pc += q3 * self.colour[c]
            else:
                for c in range(3):
                    pc = p[..., c]
                    pc *= 1 - q3
                    pc += q3 * self.colour[c]
        if not len(yy):
            return
        # the drops, pixel by pixel: what they see, rim, glint, then the veil
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
