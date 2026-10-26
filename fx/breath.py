"""A sleeping animal breathing (a placed lens effect): its belly rises and falls.

The owner, 2026-10-26, with a photo of a golden retriever asleep on its bed by the fire: "tôi muốn thêm hiệu ứng con
chó đang ngủ, hiệu ứng này sẽ làm bụng của con chó nhấp nhô như đang thở, tôi sẽ kéo hiệu ứng này vào vùng bụng con
chó". The box (Effect.BOX, 4:3) goes over the dog, its bottom edge where the dog lies on its bed; inside it an orange
outline, the body ("thân chó": frame_w, frame_h, as a fireplace's opening is set: dragged on the page), stands on the
bottom edge in the middle: its sides on the shoulder and the hip, its top on the top of the body (a dog lying on its
side: its back). The room above the body, up to the box's top edge, is where the rise fades out.

Nothing is drawn: the picture's own pixels move up and back down, by a smooth field:
- up the body: it rises more and more from the bed up to its top; above that the rise fades out over a band (FADE of
  the body's height, no higher than the box's top edge): what is there, the room behind the back, is pressed
  together a little as the back rises into it, and let go as it settles; above the band nothing moves;
- across: the most in the middle of the body (the ribs, the belly), fading out to nothing at its sides (shoulder,
  hip): nothing beside the body moves (the head, a paw).
Every profile is smooth with no slope at its ends, so no line shows where the motion starts or stops; everything
beside the body, above the band and outside the box stays exactly as it was. The field changes slowly (its steepest
slope is well under 1: the warp never folds), so each pixel is read from where it came from, below it (a backward
warp). Only up and down: a column of the picture is read from itself (fast).

The breathing: a sleeping dog breathes in (INHALE of a breath: the belly rising smoothly), breathes out (EXHALE: by
itself, quickly at first, then slower as it settles), and rests a moment before the next breath. A whole number of
breaths per loop (the rate's nearest whole number), each a little longer or shorter, and the longer ones a little
deeper (fixed draws by the clip's seed, so frame P is frame 0 again); a breath starts and ends at rest with no slope,
so going from one breath to the next never jumps. At rest the picture is exactly as it was: the pictured dog is the
dog breathed out.

The moved picture is read with a cubic filter (Catmull-Rom: a pixel read at its own place is exactly itself), kept
within the two pixels nearest to where it reads (no ringing along an outline): a fur's fine detail stays sharp as it
moves; a linear read blurs it by turns, the most as the shift goes through half a pixel.

Every length is relative to the box, so the box alone (its preview clip, made from the picture under it:
Effect.UNDER) moves as the export does. It turns in 3D as a file clip does (a dog lying on a slope, seen from one
side): drawn upright on its sheet, its change a layer laid over the frame (apply's kept).
"""
import math

import numpy as np

from .effects import Effect, _choice, _num, _place, kept_of

DEFAULTS = {"amount": 50, "rate": 18, "vary": 40}
LOOKS = {"dog": {}, "small": {"amount": 38, "rate": 26}, "deep": {"amount": 70, "rate": 12, "vary": 50}}

SIDE = 0.35  # the motion fades in over this share of the body's width from each of its sides
RISE = 0.06  # how far the top of the body rises in an average breath at "Bụng phồng" 100, body heights (3 % at 50)
FADE = 0.3  # the band above the body the rise fades out over, body heights (at most up to the box's top edge)
STEEPEST = 0.6  # the steepest the rise may get down a column (under 1: the warp never folds)
INHALE, EXHALE = 0.38, 0.47  # shares of a breath: breathing in, breathing out (the rest of it: a pause)
LONGER, DEEPER = 0.12, 0.2  # how much breaths differ at "Mỗi hơi thở một khác" 100: length, depth (one deviation)
TINY = 0.01  # pixels: a motion of no more than this is not made (the picture stays exactly as it was)
BAND = 64  # rows read at a time

SCHEMA = {"label": "Chó ngủ thở", "params": [
    _choice("look", "Kiểu", "dog", [("dog", "Chó ngủ say"), ("small", "Chó nhỏ, mèo (thở nhanh, nhẹ)"),
                                    ("deep", "Thở sâu, chậm")],
            presets={k: {**DEFAULTS, **v} for k, v in LOOKS.items()}, group="Kiểu"),
    _num("amount", "Bụng phồng (ít ← → nhiều)", 50, group="Hơi thở"),
    _num("rate", "Nhịp thở (số hơi mỗi phút)", 18, 6, 40, group="Hơi thở"),
    _num("vary", "Mỗi hơi thở một khác", 40, group="Hơi thở"),
    _num("frame_w", "Thân chó rộng (% khung)", 75, 20, 100, group="Thân chó"),
    _num("frame_h", "Thân chó cao (% khung)", 70, 20, 80, group="Thân chó"),
    *_place(35)],
    # the body: an orange outline on the box's bottom edge, in its middle, its sides and top dragged on the page (as a
    # fireplace's opening is: "fixed", always a rectangle)
    "frame": {"fixed": "rect", "width": "frame_w", "height": "frame_h", "label": "thân chó",
              "tip": "Kéo ô cam hai bên và ở giữa cạnh trên cho viền cam ôm thân con chó: chỉ phần thân trong viền cam "
                     "phồng lên xẹp xuống.",
              "side_tip": "Kéo cho viền cam rộng bằng thân con chó (từ vai tới hông)",
              "top_tip": "Kéo cho cạnh trên viền cam chạm mép trên thân con chó (lưng)"},
    "note": "Làm bụng con chó (hay con mèo) đang ngủ phồng lên xẹp xuống như đang thở. (1) Bấm Suốt video. (2) Kéo "
            "khung trắng lên con chó, cạnh dưới khung ở chỗ con chó nằm trên nệm; kéo góc khung cho khung cao hơn lưng "
            "con chó một khoảng. (3) Kéo ô cam hai bên tới vai và hông, ô cam ở cạnh trên tới lưng: chỉ thân trong "
            "viền cam thở, đầu và chân ngoài viền cam đứng yên. (4) Để tường, nệm đứng yên hẳn: ở mục Vùng bấm Vẽ "
            "vùng, tô cả thân con chó (tô lấn ra ngoài mép lông một chút), bấm Xong: chỉ phần đã tô chuyển động, lưng "
            "nhô lên che lên tường. (5) Chọn Kiểu, hoặc chỉnh Bụng phồng và Nhịp thở. Con chó nằm nghiêng theo hướng "
            "khác: xoay khung (nút tròn) cho cạnh dưới khung nằm dọc chỗ nằm.",
    # (what the page says while the clip's area is painted, and under "Vùng": the area is what moves, Effect.MOVES)
    "area_tip": "Tô lên cả thân con chó, cả lưng, tô lấn ra ngoài mép lông một chút. Chỉ phần đã tô chuyển động; "
                "tường, nệm không tô thì đứng yên, lưng chó nhô lên che lên tường.",
    "area_note": "Muốn tường, nệm đứng yên hẳn: bấm Vẽ vùng, tô cả thân con chó rồi bấm Xong. Chưa tô: mọi thứ trong "
                 "viền cam chuyển động, tường ngay trên lưng hơi co giãn theo."}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _rect(x0, y0, x1, y1, w, h):
    """Pixel rows and columns covering x0..x1, y0..y1 within a w x h frame (or None)."""
    r = (max(0, math.floor(x0)), max(0, math.floor(y0)), min(w, math.ceil(x1)), min(h, math.ceil(y1)))
    return r if r[0] < r[2] and r[1] < r[3] else None


def breath(phi):
    """How far a breath has filled the belly (0..1) at phi (0..1 through the breath): in smoothly, out by itself
    (quickly at first, then slower), then a pause at rest; 0 at both ends, with no slope there."""
    if phi < INHALE:
        return float(_smooth(phi / INHALE))
    s = (phi - INHALE) / EXHALE
    if s >= 1:
        return 0.0
    return float(1.0 - _smooth(1.0 - (1.0 - s) ** 1.6))


def _weights(f):
    """The Catmull-Rom weights of the four taps round a read at fraction f (0..1) past the second."""
    f2 = f * f
    f3 = f2 * f
    return ((f2 - 0.5 * (f3 + f)), (1.5 * f3 - 2.5 * f2 + 1.0), (2.0 * f2 + 0.5 * f - 1.5 * f3), (0.5 * (f3 - f2)))


def _read(src, sw, sh, cols, py):
    """The columns `cols` (1-D, int32) of the picture `src` (a flat float32 copy of each channel, sh x sw) read down
    each column at rows py (2-D float32, row i at i; a row of it per row read) with a cubic (Catmull-Rom) filter,
    kept within the two nearest pixels (no ringing), its edge repeated beyond it: an array py.shape + (3,)."""
    yi = np.floor(py)
    w = _weights(py - yi)
    yi = yi.astype(np.int32)
    at = [np.clip(yi + k, 0, sh - 1) * sw + cols for k in (-1, 0, 1, 2)]
    out = np.empty(py.shape + (3,), np.float32)
    for c in range(3):
        p = src[c]
        near = p.take(at[1])
        far = p.take(at[2])
        lo, hi = np.minimum(near, far), np.maximum(near, far)
        near *= w[1]
        far *= w[2]
        near += far
        for k in (0, 3):
            v = p.take(at[k])
            v *= w[k]
            near += v
        np.clip(near, lo, hi, out=near)
        out[..., c] = near
    return out


class Breath(Effect):
    BOX = (4.0, 3.0)
    lens = True
    UNDER = True  # its preview clip: the picture under its box, breathing
    UNDER_FADE = 0.0  # (its motion already fades out to nothing at the box's edges)
    PREVIEW_CRF = 30  # (its preview clip is fur moved a few pixels, laid on the fur: finer than a flame's)
    # turned in 3D as a file clip is (a dog lying on a slope, seen from one side): drawn upright on its sheet, its
    # change a layer laid over the frame (apply's kept)
    TURNS = True
    # the clip's area is what moves (the owner, 2026-10-26, seeing the wall above the back move with it: "tôi muốn sẽ
    # tự tô phần bụng con chó hoặc có thể tự mask để không dính hiệu ứng vào tường"): painted on the dog, the wall
    # and the bed stay as they are, the back rises over the wall
    MOVES = True
    VERSION = 1

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        self.body = (p["frame_w"] / 100, p["frame_h"] / 100)  # (its width and height, shares of the box's)
        # the breaths of a loop: a whole number, each its own length (the longer, the deeper) and depth
        n = int(max(1, round(p["rate"] / 60.0 * self.loop_s)))
        g = np.random.default_rng([self.seed, 1103, 71])
        z = np.clip(g.standard_normal((2, n)), -2.0, 2.0)  # (drawn whatever "vary" is: it only scales them)
        vary = p["vary"] / 100
        lengths = np.exp(LONGER * vary * z[0])
        depths = np.maximum(0.4, 1.0 + DEEPER * vary * (0.6 * z[0] + 0.8 * z[1]))
        self.ends = np.cumsum(lengths) / lengths.sum()  # where each breath ends, shares of the loop
        self.ends[-1] = 1.0
        self.depths = depths / depths.mean()
        self.start = float(g.random())  # (where in a breath the loop starts: two dogs do not breathe together)
        # box heights the top of the body rises in an average breath; the deepest no steeper than STEEPEST over the
        # band above the body or the body itself (a smooth step: its steepest slope is 1.5 times its average)
        bh = self.body[1]
        band = min(1.0 - bh, FADE * bh)
        self.rise = min(RISE * p["amount"] / 100 * bh, STEEPEST / 1.5 * min(band, bh) / float(self.depths.max()))
        self._shape = (None, None)

    def level(self, t):
        """How far the belly has risen at frame t: 0 breathed out (as pictured), about 1 in an average breath in."""
        x = (self.tau(t % self.P) + self.start) % 1.0
        i = min(int(np.searchsorted(self.ends, x, side="right")), len(self.ends) - 1)
        first = float(self.ends[i - 1]) if i else 0.0
        return float(self.depths[i]) * breath((x - first) / (float(self.ends[i]) - first))

    def _profiles(self, r, x0, y0, w, h):
        """The motion's shape over rect r (it is separable): how much each row rises (0..1), and each column."""
        key = (r, x0, y0, w, h)
        if self._shape[0] != key:
            bw, bh = self.body
            top, left = 1.0 - bh, 0.5 - bw / 2  # (the body's top and left side, shares of the box)
            fade = min(top, FADE * bh)
            v = (np.arange(r[1], r[3], dtype=np.float64) + 0.5 - y0) / h
            u = (np.arange(r[0], r[2], dtype=np.float64) + 0.5 - x0) / w
            rows = np.where(v < top, _smooth((v - top + fade) / max(fade, 1e-9)), _smooth((1.0 - v) / bh))
            side = SIDE * bw
            cols = _smooth((u - left) / side) * _smooth((left + bw - u) / side)
            self._shape = (key, (rows.astype(np.float32), cols.astype(np.float32)))
        return self._shape[1]

    def release(self):
        self._shape = (None, None)

    def draw(self, t, canvas, mask):
        """Nothing is laid over the picture: the picture itself moves (apply())."""

    def apply(self, t, picture, mask, gain, kept=None):
        """The breathing on `picture` (float32 RGB 0..1, changed in place) in its box, `gain` (0..1, a fade) times as
        strongly: the motion fades, not a crossfade with the still picture (no doubled outline). `mask` (0..1; None:
        everything) is what moves (Effect.MOVES: the clip's area, the dog painted): a pixel shows what rose into it
        as much as that came from the area, the rest of it stays as it was (the wall behind the back, the bed: never
        stretched, only covered as the body rises over them). Turned (on a sheet): `kept` says how much of each pixel
        it keeps (the least that makes its change a layer laid over the picture: kept_of())."""
        if gain <= 0:
            return
        x0, y0, w, h = self.box()
        lift = self.level(t) * self.rise * h * gain  # pixels the top of the body rises now
        if lift <= TINY:
            return
        bw, bh = self.body
        left = x0 + (0.5 - bw / 2) * w
        above = y0 + (1.0 - bh - min(1.0 - bh, FADE * bh)) * h
        r = _rect(left, above, left + bw * w, y0 + h, self.w, self.h)  # (nothing beside the body, above its band)
        if r is None:
            return
        rx0, ry0, rx1, ry1 = r
        rows, cols = self._profiles(r, x0, y0, w, h)
        rows = rows * lift
        # the picture round the rect, as far down as the motion reaches (the frame's edge repeated beyond it), a
        # channel at a time (flat copies: gathering from them is much faster than indexing h x w x 3); what moves too
        sy0, sy1 = max(0, ry0 - 2), min(self.h, ry1 + int(math.ceil(lift)) + 3)
        sw, sh = rx1 - rx0, sy1 - sy0
        if sh < 2:
            return
        src = [np.ascontiguousarray(picture[sy0:sy1, rx0:rx1, c]).ravel() for c in range(3)]
        moves = None if mask is None else np.ascontiguousarray(mask[sy0:sy1, rx0:rx1], np.float32).ravel()
        at = np.arange(sw, dtype=np.int32)
        for b0 in range(ry0, ry1, BAND):
            b1 = min(ry1, b0 + BAND)
            up = rows[b0 - ry0:b1 - ry0, None] * cols[None, :]  # pixels each one rises
            if float(up.max()) <= TINY:
                continue
            up += np.arange(b0 - sy0, b1 - sy0, dtype=np.float32)[:, None]  # (each read from below it: it rose)
            band = picture[b0:b1, rx0:rx1]
            before = band.copy() if kept is not None or moves is not None else None
            band[:] = _read(src, sw, sh, at, up)
            if moves is not None:  # (as much as what rose here came from what moves, the rest as it was)
                yi = np.floor(up)
                f = up - yi
                yi = np.clip(yi.astype(np.int32), 0, sh - 1)
                share = moves.take(yi * sw + at) * (1 - f) + moves.take(np.minimum(yi + 1, sh - 1) * sw + at) * f
                band -= before
                band *= share[..., None]
                band += before
            if kept is not None:
                kept[b0:b1, rx0:rx1] = kept_of(before, band)
