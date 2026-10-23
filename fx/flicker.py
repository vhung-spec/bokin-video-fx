"""A painted area flickering: a carved pumpkin with a candle inside, a lantern, a fireplace, a window lit by candles, or
a lamp or a sign going off and on: blinking, breathing, a faulty neon tube, a sign lit bit by bit (a lens effect: the
picture's own pixels get darker and back).

Only what is painted flickers, and only its light changes: inside the clip's area every pixel is darkened by the same
share of its light (its colours, its shapes, everything it shows stay; outside the area nothing changes at all), and
at its brightest the area is exactly as in the video (nothing is made brighter: nothing burns out). The area fades in
from its outline ("Mép mềm": the effect's own, as the rustling leaves' rim; the clip's soft edge is not used). Without
an area it changes nothing. So its preview clip is a black veil, as see-through as the light is bright: nothing of the
picture is in it, and laid over the video at any size it shows exactly what the export does (VEIL).

"Độ nháy" is how much of the light goes at the darkest: a flame mostly bright as in the video, fluttering darker, now
and then a deeper dip (a draught); a fire faster; a lamp or a sign on as in the video and off (quite dark at "Độ nháy"
100): an even blink, a slow breath, a faulty tube (steady, then now and then a burst of quick flickers off, one in
three dark for a moment before it catches again), or a sign lit bit by bit along its length (a wipe, all of it on for
a while, two blinks, off; "Hướng sáng lần lượt"). Every wave, blink and burst comes a whole number of times per loop,
so frame P is frame 0 again. "Tốc độ" makes every one faster or slower.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .effects import Effect, _choice, _num

TAU = 2 * math.pi
# each way: a flame's waves (Hz at speed 50, share) and dips (Hz, how sharp, how deep); a lamp's or a sign's ("off":
# on as in the video, then off) how it goes off and how often (Hz at speed 50). And what choosing it sets
MODES = {
    "candle": {"waves": ((0.6, 1.0), (1.1, 0.8), (1.9, 0.6), (3.1, 0.45), (4.7, 0.3), (7.3, 0.2)),
               "dip": (0.17, 8, 1.4), "preset": {"amount": 50}},
    "fire": {"waves": ((1.3, 1.0), (2.3, 0.9), (3.7, 0.7), (5.9, 0.6), (8.3, 0.45), (11.0, 0.35)),
             "dip": (0.3, 4, 1.0), "preset": {"amount": 50}},
    "breathe": {"off": "breath", "hz": 0.25, "preset": {"amount": 50}},
    "pulse": {"off": "blink", "hz": 1.0, "preset": {"amount": 80}},
    "faulty": {"off": "faulty", "hz": 0.15, "preset": {"amount": 80}},
    "sweep": {"off": "sweep", "hz": 0.25, "preset": {"amount": 80}},
}
DEFAULTS = {"amount": 50, "speed": 50, "edge": 40}
SWITCH = 0.02  # seconds a lamp or a sign takes to go on or off
# a sign lit bit by bit: shares of its cycle (lit along its length, all of it on until HOLD, two blinks, off from
# DOWN), and how soft the edge between lit and not yet lit is (a share of its length)
WIPE, HOLD, DOWN, FRONT = 0.4, 0.7, 0.9, 0.12
WAYS = {"right": ("across", 1.0), "left": ("across", -1.0), "down": ("down", 1.0), "up": ("down", -1.0)}

SCHEMA = {"label": "Nhấp nháy vùng", "params": [
    _choice("mode", "Kiểu nháy", "candle",
            [("candle", "Ánh nến lung linh (bí ngô, đèn lồng)"), ("fire", "Lửa bập bùng (lò sưởi, đống lửa)"),
             ("breathe", "Sáng mờ dần (như thở)"), ("pulse", "Bật tắt đều (biển hiệu, đèn nháy)"),
             ("faulty", "Chập chờn (đèn, biển neon hỏng)"), ("sweep", "Sáng lần lượt (chữ biển hiệu)")],
            presets={k: m["preset"] for k, m in MODES.items()}, group="Kiểu"),
    dict(_choice("way", "Hướng sáng lần lượt", "right",
                 [("right", "Trái sang phải"), ("left", "Phải sang trái"), ("down", "Trên xuống dưới"),
                  ("up", "Dưới lên trên")], group="Kiểu"), show={"mode": ["sweep"]}),
    _num("amount", "Độ nháy (lúc tối, tối đi bao nhiêu)", DEFAULTS["amount"], group="Nháy"),
    _num("speed", "Tốc độ (chậm ← → nhanh)", DEFAULTS["speed"], group="Nháy"),
    _num("edge", "Mép mềm", DEFAULTS["edge"], group="Vùng")],
    "note": "Làm đúng chỗ bạn tô sáng tối nhấp nháy: quả bí ngô có nến bên trong, đèn lồng, lò sưởi, hay biển hiệu "
            "quảng cáo, đèn neon. Cả chỗ tô tối đi rồi sáng lại như trong video, đều nhau: màu và hình giữ nguyên, "
            "ngoài chỗ tô không đổi gì. (1) Bấm Suốt video. (2) Bấm Vẽ vùng rồi tô đúng chỗ cần nhấp nháy (quả bí "
            "ngô, chữ của biển hiệu; muốn ánh sáng trên mặt bàn, bức tường quanh nó nháy theo thì tô cả chỗ đó). Chưa "
            "tô vùng thì chưa có gì nhấp nháy. (3) Chọn Kiểu nháy, chỉnh Độ nháy (lúc tối, tối đi bao nhiêu) và Tốc "
            "độ. Biển hiệu: Bật tắt đều, Chập chờn hay Sáng lần lượt; Độ nháy 100 là lúc tắt tối hẳn.",
    # (what the page says while the area is painted, and when nothing is painted yet)
    "area_tip": "Chỗ tô nhấp nháy, mềm dần từ mép vào: chỉnh Mép mềm trong Bảng chỉnh hiệu ứng.",
    "area_empty": "Chưa tô vùng nên chưa có gì nhấp nháy. Bấm Vẽ vùng rồi tô lên chỗ cần nhấp nháy (quả bí ngô, "
                  "biển hiệu)."}


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class Flicker(Effect):
    lens = True
    UNDER = True  # its preview clip: over the picture under its area
    VEIL = True  # (and only a black veil: nothing of the picture is in it, see veil())
    AREA = True  # only inside the clip's area (no area: nothing)
    TURNS = False  # the picture's own light, where it is: nothing to place or turn
    PREVIEW = 960
    VERSION = 3  # (2: stronger, in linear light, only the lit places; 3: the whole area, as it is, darker and back)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        mode = MODES.get(p["mode"], MODES["candle"])
        self.mode, self.off = mode, mode.get("off")
        self.depth = p["amount"] / 100  # the share of the light that goes at the darkest
        self.rate = 2 ** ((p["speed"] - 50) / 35)  # 0.37 to 2.7 times as fast
        self.edge = p["edge"] / 100 * 30 * self.px  # the soft rim's blur radius, pixels of this frame
        g = self.rng
        self.waves = []
        for hz, share in mode.get("waves", ()):
            self.waves.append((self.cycles(hz * self.rate * (0.9 + 0.2 * g.random())), share, g.random() * TAU))
        self.dip = None
        if "dip" in mode:
            hz, sharp, deep = mode["dip"]
            self.dip = (self.cycles(hz * self.rate), self.cycles(hz * self.rate * 1.6), sharp, deep,
                        g.random(2) * TAU)
        g.random(2)  # (two draws, a flame's sway's once: the later ones as they were)
        if self.off:  # how often it blinks, breathes, flickers or is lit along (whole numbers a loop)
            self.m, self.ph = float(self.cycles(mode["hz"] * self.rate)), g.random() * TAU
        self.switch = SWITCH / self.loop_s  # (loop phases)
        if self.off == "faulty":
            self.outages = self._outages()
        self.axis, self.sign = WAYS.get(p["way"], WAYS["right"])
        self.norm = self._norm() if not self.off else (0.0, 1.0)
        self._plan_of = (None, None)

    # --- the light -----------------------------------------------------------------------------------------------

    def _raw(self, tau):
        """A flame's light at loop phase(s) tau, before it is scaled (about -1..1)."""
        tau = np.asarray(tau, np.float64)
        s = np.zeros_like(tau)
        for m, share, ph in self.waves:
            s += share * np.sin(TAU * m * tau + ph)
        if self.dip is not None:  # now and then a draught: the flame ducks, then flares back
            m1, m2, sharp, deep, (p1, p2) = self.dip
            gate = (0.5 + 0.5 * np.sin(TAU * m1 * tau + p1)) * (0.5 + 0.5 * np.sin(TAU * m2 * tau + p2))
            s -= deep * gate ** sharp * 4
        return s

    def _norm(self):
        """(mean, scale): a flame's light over the loop set to a mean of 0 and a swing of about +-1."""
        s = self._raw((np.arange(4096) + 0.5) / 4096)
        return float(s.mean()), max(2.0 * float(s.std()), 1e-6)

    def _outages(self):
        """A faulty lamp's moments off over the loop (starts, lengths: loop phases): now and then (a whole number of
        times a loop) a burst of two to five quick flickers off, and one burst in three ends dark for a moment before
        it catches again with one more flicker."""
        g = np.random.default_rng([self.seed, 4421])
        n = int(self.m)
        quick = 1 / math.sqrt(self.rate)  # (faster: quicker flickers)
        starts, lengths = [], []
        for i in range(n):
            t = (i + 0.1 + 0.7 * g.random()) / n * self.loop_s  # seconds
            for _ in range(int(g.integers(2, 6))):
                off = g.uniform(0.07, 0.16) * quick
                starts.append(t)
                lengths.append(off)
                t += off + g.uniform(0.06, 0.25) * quick
            if g.random() < 1 / 3:  # dark for a moment, then it catches again
                off = g.uniform(0.35, 0.9) * quick
                starts.append(t)
                lengths.append(off)
                t += off + g.uniform(0.08, 0.15) * quick
                starts.append(t)
                lengths.append(g.uniform(0.07, 0.12) * quick)
        return (np.array(starts) / self.loop_s % 1.0, np.minimum(np.array(lengths) / self.loop_s, 0.3))

    def _windows(self, ph, starts, lengths, edge):
        """How much of the way into any of the windows (starts, lengths, in phases of a cycle of 1) phase(s) ph are,
        0..1, each window's edges `edge` long (a window over the cycle's end goes on at its start)."""
        u = (np.asarray(ph, np.float64)[..., None] - starts) % 1.0
        u = np.where(u > 0.5, u - 1.0, u)  # (just before a start is before it)
        return (_smooth(u / edge) * _smooth((lengths - u) / edge)).max(axis=-1)

    def level(self, tau, pos=0.0):
        """The light at loop phase(s) tau: a flame's about -1 (darkest) .. 1 (brightest); a lamp's or a sign's 0 (on:
        as in the video) .. -1 (off). A sign lit bit by bit is lit where it is along its length: pos -1 (where it
        starts) .. 1 (where it ends), broadcast with tau."""
        if not self.off:
            mean, scale = self.norm
            return np.clip((self._raw(tau) - mean) / scale, -1.5, 1.5)
        tau = np.asarray(tau, np.float64)
        if self.off == "blink":  # on half the time, off the other half (each switch about SWITCH)
            k = 1 / (TAU * self.m * self.switch)
            return 0.5 * np.tanh(k * np.sin(TAU * self.m * tau + self.ph)) / math.tanh(k) - 0.5
        if self.off == "breath":
            return -(0.5 - 0.5 * np.cos(TAU * self.m * tau + self.ph))
        if self.off == "faulty":
            return -self._windows(tau, *self.outages, self.switch)
        # a sign lit bit by bit: the lit part grows along it, all of it stays on, blinks twice, and goes off
        ph = (tau * self.m + self.ph / TAU) % 1.0
        edge = self.switch * self.m  # (a switch, in phases of a cycle)
        front = -1 - FRONT + (2 + 2 * FRONT) * np.minimum(ph / WIPE, 1.0)
        lit = _smooth((front - np.asarray(pos, np.float32)) / FRONT + 0.5)
        blinks = self._windows(ph, np.array([HOLD, HOLD + (DOWN - HOLD) / 2, DOWN]),
                               np.array([(DOWN - HOLD) / 4, (DOWN - HOLD) / 4, 1 - DOWN]), edge)
        return lit * (1 - blinks) - 1.0

    def darkness(self, tau, pos=0.0):
        """How dark the light is at loop phase(s) tau, 0 (as in the video) .. 1 (off), before "Độ nháy": a flame's
        mostly near 0, fluttering, now and then a deeper dip (its light's darker half, squared); a lamp's or a sign's
        0 on, 1 off (a sign lit bit by bit: where it is along its length, pos as level())."""
        if not self.off:
            return ((1.0 - np.clip(self.level(tau), -1.0, 1.0)) / 2) ** 2
        return -self.level(tau, pos)

    # --- where: the area -------------------------------------------------------------------------------------

    def _plan(self, mask):
        if self._plan_of[0] is mask:
            return self._plan_of[1]
        plan = self._make_plan(mask)
        self._plan_of = (mask, plan)
        return plan

    def _make_plan(self, mask):
        """What stays while the area is the same: its box, how strongly each pixel of it follows the light (the
        area, faded in from its outline), and where each column lies across it and each row down it (-1 .. 1: a
        sign lit bit by bit)."""
        rows, cols = np.flatnonzero(mask.max(axis=1) > 0), np.flatnonzero(mask.max(axis=0) > 0)
        if not len(rows):
            return None
        x0, y0, x1, y1 = int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1
        m = np.clip(mask[y0:y1, x0:x1], 0, 1).astype(np.float32)
        if self.edge >= 0.3:
            r = int(math.ceil(3 * self.edge)) + 1
            ex0, ey0, ex1, ey1 = max(0, x0 - r), max(0, y0 - r), min(self.w, x1 + r), min(self.h, y1 + r)
            m8 = Image.fromarray((np.clip(mask[ey0:ey1, ex0:ex1], 0, 1) * 255 + 0.5).astype(np.uint8))
            soft = np.asarray(m8.filter(ImageFilter.GaussianBlur(self.edge)), np.float32)[
                y0 - ey0:y1 - ey0, x0 - ex0:x1 - ex0] * (1 / 255.0)
            m *= _smooth((soft - 0.5) / 0.45)  # (0 at the outline, all of it well inside: never beyond the area)

        def along(n):  # -1 .. 1 over n pixels
            return np.linspace(-1.0, 1.0, n, dtype=np.float32) if n > 1 else np.zeros(1, np.float32)
        return {"box": (x0, y0, x1, y1), "weight": m, "across": along(x1 - x0)[None, :],
                "down": along(y1 - y0)[:, None]}

    def release(self):
        self._plan_of = (None, None)

    # --- the picture -------------------------------------------------------------------------------------------

    def draw(self, t, canvas, mask):
        """Nothing is laid over the picture: its own light changes (apply())."""

    def veil(self, t, mask, gain=1.0):
        """How much of the light inside `mask` (the clip's area, 0..1) goes at frame t, `gain` (a fade) times as
        much: ((x0, y0, x1, y1): the area's box, float32 (rows, cols) 0..1 over it), or None when none goes. The same
        for every pixel but where the area fades in from its outline: its preview clip is this, a black veil."""
        if mask is None or gain <= 0 or self.depth <= 0:
            return None
        plan = self._plan(mask)
        if plan is None:
            return None
        tau = self.tau(t)
        if self.off == "sweep":
            dark = self.darkness(tau, self.sign * plan[self.axis]).astype(np.float32)
        else:
            dark = float(self.darkness(tau))
        return plan["box"], plan["weight"] * np.asarray(dark * self.depth * gain, np.float32)

    def apply(self, t, picture, mask, gain, kept=None):
        """The flicker on `picture` (float32 RGB 0..1, changed in place) inside `mask` (the clip's area, 0..1; None:
        no area, nothing changes), `gain` (0..1, a fade) times as strongly: darker by veil(). Where the area is 0
        every pixel stays exactly as it was. (It never turns: `kept` is not used.)"""
        v = self.veil(t, mask, gain)
        if v is None:
            return
        (x0, y0, x1, y1), dark = v
        picture[y0:y1, x0:x1] *= (1.0 - dark)[..., None]
