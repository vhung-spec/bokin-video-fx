"""Foliage rustling in a breeze (a lens effect): inside the clip's area (bushes, a tree's crown, hanging flower
baskets), each cluster of leaves sways a little on its own, smoothly, while the outline of the area stays still.

What a camera sees of leaves in a light breeze (measured in a reference video, 1080p: bushes by a fence, a tree's
crown): the outline of a bush does not move; inside it, each cluster of leaves sways by about 0.5 to 1.5 pixels,
smoothly, with periods of about 2 to 4 seconds and slower drifts; neighbouring clusters are nearly independent (and
not at all beyond about 30 pixels); the light on a patch changes by a few levels as its leaves tilt. Not the whole
bush swaying.

So the picture inside the area is warped by a smooth displacement field, sampled bilinearly. The field is a sum of
octaves of smooth random fields (value noise: normal draws on a lattice as fine as a cluster of leaves, interpolated
bicubically), each with its own frequencies, every one a whole number of cycles per loop:
    D(x, t) = sum_k a_k(x) cos(2 pi m_k t / P) + b_k(x) sin(2 pi m_k t / P)
with a_k, b_k independent smooth random fields: each cluster wanders on its own, smoothly, a sum of a few sways, and
frame P is frame 0 again. Clusters sway at 2 to 6 second periods, smaller sprays of leaves a little faster and less,
and everything drifts slowly.

- A breeze from one side (optional): gusts, a whole number per loop, sweep across the frame from where the wind comes
  from; each pushes every cluster the same way, each by its own share, and stirs the rustling. The push has no mean
  over the loop: on average the picture is where it was.
- Leaves catching the light (optional): a cluster swaying toward the light gets a little brighter (a few percent at
  most), and a gust shows the paler backs of the leaves.
- The leaves move all over the area painted, right up to a thin rim along its outline that stays still ("Giữ yên
  viền": at the default the outermost 2 pixels of a 1080-line frame, the motion full 10 pixels in), so a bush's
  silhouette against the sky does not swim. The area is taken as painted: the clip's "Mép mềm" does not narrow it
  (engine.area_feather), the rim is the effect's own. Without an area the effect changes nothing: it is meant for
  the leaves only, never the whole picture.

Lengths are pixels of a 1080-line frame and the lattices lie over the frame in those units, so the effect looks the
same at any size: the preview clip, made from the picture under the area at the preview's size (Effect.UNDER), moves
as the export does. Only the area is warped, in bands of rows as wide as it is there (the field is made on a coarse
grid over the area's box, then enlarged over each band).
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .effects import Effect, _choice, _num

DEFAULTS = {"amount": 40, "cluster": 40, "speed": 50, "breeze": 0, "direction": "right", "shimmer": 30, "edge": 40}
LOOKS = {
    "gentle": {},
    "medium": {"amount": 52, "cluster": 47, "speed": 56, "breeze": 35, "shimmer": 36},
    "strong": {"amount": 62, "cluster": 58, "speed": 64, "breeze": 70, "shimmer": 42},
}

SCHEMA = {"label": "Lá cây lay động", "params": [
    _choice("look", "Kiểu", "gentle", [("gentle", "Gió nhẹ (lá rung khẽ)"), ("medium", "Gió vừa"),
                                      ("strong", "Gió lớn (lá xào xạc)")],
            presets={k: {**DEFAULTS, **v} for k, v in LOOKS.items()}, group="Kiểu"),
    _num("amount", "Lay mạnh", 40, group="Lá lay"),
    _num("cluster", "Cỡ chùm lá (nhỏ ← → to)", 40, group="Lá lay"),
    _num("speed", "Nhịp lay (chậm ← → nhanh)", 50, group="Lá lay"),
    _num("breeze", "Gió thổi một hướng (từng cơn)", 0, group="Gió"),
    _choice("direction", "Hướng gió", "right", [("right", "Từ trái sang phải"), ("left", "Từ phải sang trái")],
            group="Gió"),
    _num("shimmer", "Lá lấp lánh (sáng tối theo nắng)", 30, group="Ánh sáng"),
    _num("edge", "Giữ yên viền (viền bụi cây, tán lá)", 40, group="Viền")],
    "note": "Làm lá cây lay động nhẹ như có gió (bụi cây, tán cây, giỏ hoa); chỉ lá trong vùng của clip động. "
            "(1) Bấm Suốt video. (2) Bấm Vẽ vùng (hoặc Vùng của clip đang chọn) rồi tô lên bụi cây, tán lá; "
            "không tô lên nhà, xe, bầu trời. Chưa tô vùng thì chưa có gì lay động. (3) Chọn Kiểu: gió nhẹ, vừa, lớn. "
            "Lá động khắp vùng đã tô, chỉ một viền mỏng sát mép đứng yên (Giữ yên viền: kéo lên nếu thấy viền bụi cây "
            "rung), nên hãy tô sát mép bụi cây. Hiệu ứng này không xoay 3D."}

TAU = 2 * math.pi
# the octaves of the rustling: (lattice spacing, in cluster sizes; share of the motion; frequencies, Hz at speed 50)
OCTAVES = ((1.0, 1.0, (0.17, 0.24, 0.33, 0.46)),  # clusters of leaves: periods of 2 to 6 seconds
           (0.5, 0.35, (0.55, 0.75, 1.0)),  # smaller sprays of leaves, quicker
           (1.4, 0.5, (0.05, 0.09)))  # slow drifts
NOISE_STD = 0.816  # the standard deviation of bicubic value noise from unit normal draws
SWAY_PX = 0.0104  # each axis's standard deviation of the rustling, pixels of a 1080-line frame, per point of "amount"
BREEZE_PX = 0.032  # the push of the strongest gust, per point of "breeze"
SHINE = 0.0007  # brightness swing (a share, one standard deviation), per point of "shimmer"
# patches of leaves catching the light together: (lattice spacing, in cluster sizes; share of the shimmer;
# frequencies, Hz at speed 50)
PATCHES = (5.0, 1.0, (0.07, 0.12, 0.19))
EDGE_PX = 0.15  # the rim along the area's outline (a blur radius), per point of "edge": still for a third of it,
# the motion full 1.6 radii in
GUST_SPEED = 0.7  # how fast a gust crosses the frame, frame heights a second
MARGIN = 4  # lattice nodes beyond each edge of the frame


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class Foliage(Effect):
    lens = True
    UNDER = True  # its preview clip: the picture under its area, rustling (made from the video and its still area)
    AREA = True  # it works inside the clip's area only (no area: nothing moves)
    TURNS = False  # it moves the picture's own leaves where they are: nothing to place or turn
    PREVIEW = 1280  # sub-pixel motion needs a sharp preview clip (at 640 a 1-pixel sway is a third of a pixel)
    PREVIEW_CRF = 30
    VERSION = 2  # 2: the motion all over the area, a thin still rim (it faded in over 50 pixels and more)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        rng = self.rng
        self.cluster = 4.5 * 10 ** (p["cluster"] / 100)  # cluster size, 1080-line pixels: 4.5 to 45 (11 at 40)
        self.sway = SWAY_PX * p["amount"] * self.px  # each axis's standard deviation, pixels of this frame
        self.push = BREEZE_PX * p["breeze"] * self.px
        self.stir = p["breeze"] / 100  # how much a gust stirs the rustling (doubles it at full breeze)
        self.shine = SHINE * p["shimmer"]
        self.edge = EDGE_PX * p["edge"] * self.px
        self.side = 1.0 if p["direction"] == "right" else -1.0
        light = rng.random() * TAU  # where the light comes from (in the picture's plane)
        self.light = (math.cos(light), math.sin(light))
        rate = 2 ** ((p["speed"] - 50) / 30)  # 0.31 to 3.2 times as fast
        self.octaves = []
        total = math.sqrt(sum(share * share for _, share, _ in OCTAVES))
        for o, (size, share, rates) in enumerate(OCTAVES):
            self.octaves.append(self._field(50 + o, self.cluster * size, rates, 2, share / total, rate))
        size, share, rates = PATCHES
        self.patches = self._field(59, self.cluster * size, rates, 1, share, rate)
        # how strongly each part answers a gust (a broad share that does not change)
        self.answer = self._field(60, self.cluster * 3.0, (), 1, 1.0, rate)
        g = np.random.default_rng([self.seed, 1103, 61])
        # gusts: a whole number per loop (one every 8 seconds or so), each a smooth swell
        n = max(1.0, round(self.loop_s / 8 * math.sqrt(rate)))
        self.gust_m = (n, 2 * n + 1)
        self.gust_ph = g.random(2) * TAU
        self.gust_mean = self._gust_mean()
        self._plan_of = (None, None)

    def _field(self, k, spacing, rates, axes, share, rate):
        """Smooth random fields that change with time: for each of `rates` (Hz at speed 50, a little jittered,
        `rate` times as fast, a whole number of cycles per loop) a cos and a sin lattice of normal draws `spacing`
        (1080-line pixels) apart, for each of `axes`; at any time their sum has a standard deviation of `share`. No
        rates: one lattice that does not change."""
        jitter = 1 + 0.12 * (self.rng.random(len(rates)) - 0.5)
        m = np.maximum(1.0, np.rint(np.array(rates) * jitter * rate * self.loop_s))
        ny = int(math.ceil(1080 / spacing)) + 2 * MARGIN + 1
        nx = int(math.ceil(1080 * self.aspect / spacing)) + 2 * MARGIN + 1
        g = np.random.default_rng([self.seed, 1103, k])
        lat = g.standard_normal((max(1, 2 * len(rates)), axes, ny, nx)).astype(np.float32)
        lat *= share / NOISE_STD / math.sqrt(max(1, len(rates)))
        return {"spacing": spacing, "m": m, "lat": lat, "off": g.random(2)}

    def _now(self, field, part, where, plan, tau):
        """A field's values on the plan's grid at loop phase tau: one array per axis."""
        phase = TAU * field["m"] * tau
        coef = np.concatenate([np.cos(phase), np.sin(phase)]).astype(np.float32)
        both = part[0] * coef[0]  # (axes, rows, cols): its lattices summed for now (a loop: faster than tensordot)
        for c, lat in zip(coef[1:], part[1:]):
            both += c * lat
        return [self._on_grid(a, where, plan) for a in both]

    # --- the motion ------------------------------------------------------------------------------------------

    def _gust(self, tau):
        """How strong the gust is (0..1) at loop phase(s) tau."""
        (m1, m2), (p1, p2) = self.gust_m, self.gust_ph
        swell = 0.5 - 0.5 * np.cos(TAU * m1 * tau + p1)
        return 0.75 * swell * swell + 0.25 * (0.5 - 0.5 * np.cos(TAU * m2 * tau + p2))

    def _gust_mean(self):
        tau = (np.arange(4096) + 0.5) / 4096
        return float(self._gust(tau).mean())

    def _lattice_part(self, field, plan):
        """The part of a lattice field over the plan's grid, and the box that maps it onto the grid (Pillow)."""
        s = field["spacing"] * self.px  # lattice spacing in this frame's pixels
        ox, oy = field["off"]
        x0, y0 = plan["box"][:2]
        g, gw, gh = plan["g"], plan["gw"], plan["gh"]
        u0, v0 = x0 / s + MARGIN + ox, y0 / s + MARGIN + oy
        u1, v1 = u0 + gw * g / s, v0 + gh * g / s
        ny, nx = field["lat"].shape[2:]
        cx0, cy0 = max(0, int(math.floor(u0)) - 3), max(0, int(math.floor(v0)) - 3)
        cx1, cy1 = min(nx, int(math.ceil(u1)) + 3), min(ny, int(math.ceil(v1)) + 3)
        part = np.ascontiguousarray(field["lat"][:, :, cy0:cy1, cx0:cx1])
        return part, (u0 - cx0, v0 - cy0, u1 - cx0, v1 - cy0)

    def _on_grid(self, lat, where, plan):
        im = Image.fromarray(np.ascontiguousarray(lat, np.float32), "F")
        return np.array(im.resize((plan["gw"], plan["gh"]), Image.Resampling.BICUBIC, box=where), np.float32)

    def _grid(self, t, plan):
        """The displacement (x, y, pixels of this frame) and the brightness swing on the plan's grid at frame t,
        before the area's fade."""
        tau = self.tau(t)
        gw, gh = plan["gw"], plan["gh"]
        dx, dy = np.zeros((gh, gw), np.float32), np.zeros((gh, gw), np.float32)
        for octave, (part, where) in zip(self.octaves, plan["octaves"]):
            ox, oy = self._now(octave, part, where, plan, tau)
            dx += ox
            dy += oy
        shine = None
        if self.shine > 0:  # a cluster swaying toward the light gets brighter, and patches of leaves together
            shine = dx * self.light[0] + dy * self.light[1]
            shine += self._now(self.patches, *plan["patches"], plan, tau)[0]
        gust = None
        if self.push > 0 or self.stir > 0:
            # the gust where each column is: it sweeps across from where the wind comes from
            gx = plan["x0"] + (np.arange(gw, dtype=np.float32) + 0.5) * plan["g"]
            ahead = gx / self.h if self.side > 0 else self.aspect - gx / self.h
            gust = self._gust(tau - ahead / (GUST_SPEED * self.loop_s)).astype(np.float32)[None, :]
            stir = 1 + self.stir * gust
            dx *= stir
            dy *= stir
        if shine is not None:
            shine *= self.shine
            if gust is not None:
                shine += (0.6 * self.shine * self.stir) * (gust - self.gust_mean)  # the paler backs of the leaves
        dx *= self.sway
        dy *= self.sway
        if self.push > 0:
            push = self.push * (gust - self.gust_mean) * plan["answer"]
            dx += self.side * push
            dy += 0.15 * push  # (pushed leaves droop a little; in a lull they rise back)
        return dx, dy, shine

    # --- where: the area ------------------------------------------------------------------------------------

    def _plan(self, mask):
        """What stays while the area is the same: the box to warp, how strongly each pixel of it moves (the area,
        faded toward its outline), the grid the field is made on, the lattices' parts over it."""
        if self._plan_of[0] is mask:
            return self._plan_of[1]
        plan = self._make_plan(mask)
        self._plan_of = (mask, plan)
        return plan

    def _make_plan(self, mask):
        rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
        if not len(rows):
            return None
        x0, y0, x1, y1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
        m = mask[y0:y1, x0:x1]
        if self.edge >= 0.3:
            # the area blurred: about 0.5 at its outline, 1 well inside it; the motion fades in from a third of a blur
            # radius inside the outline (the rim stays still), fully there 1.6 radii in (where the area meets the
            # frame's edge it goes on beyond it: a crown cut by the frame moves there)
            r = int(math.ceil(3 * self.edge)) + 1
            ex0, ey0, ex1, ey1 = max(0, x0 - r), max(0, y0 - r), min(self.w, x1 + r), min(self.h, y1 + r)
            m8 = Image.fromarray((np.clip(mask[ey0:ey1, ex0:ex1], 0, 1) * 255 + 0.5).astype(np.uint8))
            soft = np.asarray(m8.filter(ImageFilter.GaussianBlur(self.edge)), np.float32)[
                y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] * (1 / 255.0)
            weight = m * _smooth((soft - 0.63) / 0.32)
        else:
            weight = m.copy()
        weight = weight.astype(np.float32)
        rows, cols = np.flatnonzero(weight.max(axis=1) > 1e-4), np.flatnonzero(weight.max(axis=0) > 1e-4)
        if not len(rows):
            return None
        bx0, by0, bx1, by1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
        weight = weight[by0:by1, bx0:bx1]
        x0, y0, x1, y1 = x0 + bx0, y0 + by0, x0 + bx1, y0 + by1
        # the grid the field is made on: half the finest lattice's spacing (enlarged bilinearly, it stays smooth)
        g = max(1.0, min(o["spacing"] for o in self.octaves) * self.px / 2)
        bw, bh = x1 - x0, y1 - y0
        plan = {"box": (x0, y0, x1, y1), "g": g, "x0": float(x0),
                "gw": int(math.ceil(bw / g)) + 1, "gh": int(math.ceil(bh / g)) + 1}
        # bands of rows, each as wide as the area is there: only they are warped (two bushes far apart, or a crown
        # and a bush, do not make the whole box between them work)
        step = max(16, round(48 * self.px))
        plan["bands"] = bands = []
        for r0 in range(0, bh, step):
            r1 = min(bh, r0 + step)
            used = np.flatnonzero(weight[r0:r1].max(axis=0) > 0)
            if len(used):
                c0, c1 = int(used[0]), int(used[-1]) + 1
                bands.append((r0, r1, c0, c1, np.ascontiguousarray(weight[r0:r1, c0:c1])))
        plan["octaves"] = [self._lattice_part(o, plan) for o in self.octaves]
        if self.shine > 0:
            plan["patches"] = self._lattice_part(self.patches, plan)
        if self.push > 0:
            part, where = self._lattice_part(self.answer, plan)
            answer = self._on_grid(part[0, 0], where, plan)
            plan["answer"] = np.clip(0.65 + 0.35 * answer, 0.2, 1.3)
        return plan

    def release(self):
        """Frees what is kept for the area while the clip is not showing (made again, the same, when it shows)."""
        self._plan_of = (None, None)

    # --- the picture -------------------------------------------------------------------------------------------

    def draw(self, t, canvas, mask):
        """Nothing is laid over the picture: the leaves themselves move (apply())."""

    def apply(self, t, picture, mask, gain, kept=None):
        """The rustling on `picture` (float32 RGB 0..1, changed in place) inside `mask` (the clip's area, 0..1; None:
        no area, nothing moves), `gain` (0..1, a fade) times as strongly (the motion and the shimmer fade, not a
        crossfade with the still picture: no doubled leaves). Where the area is 0 every pixel stays exactly as it
        was. (It never turns: `kept` is not used.)"""
        if mask is None or gain <= 0 or (self.sway <= 0 and self.push <= 0 and self.shine <= 0):
            return
        plan = self._plan(mask)
        if plan is None:
            return
        x0, y0, x1, y1 = plan["box"]
        g = plan["g"]
        gdx, gdy, gshine = self._grid(t, plan)
        reach = int(math.ceil(max(float(np.abs(gdx).max()), float(np.abs(gdy).max())) * gain)) + 2
        # the picture around the box, as far as the motion reaches (the frame's edge repeated beyond it), a channel
        # at a time (flat copies: gathering from them is much faster than indexing h x w x 3; and they are read
        # before any band is written)
        sx0, sy0 = max(0, x0 - reach), max(0, y0 - reach)
        sx1, sy1 = min(self.w, x1 + reach), min(self.h, y1 + reach)
        sw, sh = sx1 - sx0, sy1 - sy0
        if sw < 2 or sh < 2:
            return
        src = [np.ascontiguousarray(picture[sy0:sy1, sx0:sx1, c]).ravel() for c in range(3)]
        gdx, gdy = Image.fromarray(gdx, "F"), Image.fromarray(gdy, "F")
        if gshine is not None:
            gshine = Image.fromarray(gshine, "F")
        for r0, r1, c0, c1, weight in plan["bands"]:
            bw, bh = c1 - c0, r1 - r0
            where = (c0 / g, r0 / g, c1 / g, r1 / g)
            k = weight * gain

            def enlarge(im):  # the grid enlarged over the band (bilinear), faded by the area
                out = np.asarray(im.resize((bw, bh), Image.Resampling.BILINEAR, box=where), np.float32)
                return out * k

            # where each pixel is read from (backward warp), bilinear; a pixel that does not move is read exactly
            px = np.arange(bw, dtype=np.float32)[None, :] + float(x0 + c0 - sx0) - enlarge(gdx)
            py = np.arange(bh, dtype=np.float32)[:, None] + float(y0 + r0 - sy0) - enlarge(gdy)
            np.clip(px, 0.0, sw - 1.0, out=px)
            np.clip(py, 0.0, sh - 1.0, out=py)
            xi = np.minimum(px.astype(np.int32), sw - 2)
            yi = np.minimum(py.astype(np.int32), sh - 2)
            px -= xi  # (now the fractions)
            py -= yi
            qx, qy = 1.0 - px, 1.0 - py
            i = yi
            i *= sw
            i += xi
            shine = None
            if gshine is not None:
                shine = enlarge(gshine)
                shine += 1.0
            out = picture[y0 + r0:y0 + r1, x0 + c0:x0 + c1]
            for c in range(3):
                p = src[c]
                top = p.take(i)
                top *= qx
                b = p.take(i + 1)
                b *= px
                top += b
                bottom = p.take(i + sw)
                bottom *= qx
                b = p.take(i + sw + 1)
                b *= px
                bottom += b
                top *= qy
                bottom *= py
                bottom += top
                if shine is not None:
                    bottom *= shine
                out[..., c] = bottom
