"""A painted area flickering as if lit by a flame inside it (a carved pumpkin, a lantern, a fireplace, a window lit by
candles), or a lamp or a sign going on and off: blinking, breathing, a faulty neon tube, a sign lit bit by bit (a
lens effect: the picture's own pixels get brighter and dimmer).

Inside the clip's area the light of the picture is multiplied by a level that changes with time, in linear light (a
light twice as bright makes a place twice as bright). Only what that light lights follows it: a place is the room's
own light plus the flame's (the lamp's, the sign's), and the darker a place is, the less of it is the flame's ("Chỗ
sáng nháy nhiều hơn": at its most, a place as dark as the room's light, ROOM, does not change at all), so the glowing
eyes and mouth of a jack-o'-lantern and the table it lights flicker, the letters of a sign and the glow round them go
off, and the dark night around them stays as it is (an area painted wider than the pumpkin or the sign shows nowhere).
A highlight burnt out to white held more light than white (KNEE: the light it stands for grows above it), so a dimmed
flame turns a burnt-out yellow opening orange, as a dimmer candle does, and a brighter one burns it whiter. Light
burning up is a little whiter still, burning down a little redder ("Ấm lên khi sáng": blue follows the light most,
red least; a lamp's or a sign's own colour: 0). A flame inside also sways: one side of the area a little brighter,
then the other. The area fades in from its outline ("Mép mềm": the effect's own, as the rustling leaves' rim; the
clip's soft edge is not used), and outside it nothing changes at all. Without an area it changes nothing.

A flame's light swings round the video's own (STRENGTH): a candle's quick, irregular flutter with now and then a
deeper dip (a draught), a fire's faster and larger one. A lamp's or a sign's is on as in the video and goes darker,
to almost off at "Độ nháy" 100 (DARK): an even blink, a slow breath, a faulty tube (steady, then now and then a burst
of quick flickers, one in three dark for a moment before it catches again), or a sign lit bit by bit along its length
(a wipe, all of it on for a while, two blinks, off; "Hướng sáng lần lượt"). Every wave, blink and burst comes a whole
number of times per loop, so frame P is frame 0 again. "Tốc độ" makes every one faster or slower.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .effects import LUMA, Effect, _choice, _num

TAU = 2 * math.pi
# each way: a flame's waves (Hz at speed 50, share), dips (Hz, how sharp, how deep), how much it sways; a lamp's or a
# sign's ("off": on as in the video, then darker) how it goes off, how often (Hz at speed 50), and whether it goes
# off ("dark": DARK; a breath only dims, as far as a flame's light swings). And what choosing it sets (a lamp's or a
# sign's own colour: no warmer when brighter; a sign goes quite dark)
MODES = {
    "candle": {"waves": ((0.6, 1.0), (1.1, 0.8), (1.9, 0.6), (3.1, 0.45), (4.7, 0.3), (7.3, 0.2)),
               "dip": (0.17, 8, 1.4), "sway": 0.35, "preset": {"warm": 50}},
    "fire": {"waves": ((1.3, 1.0), (2.3, 0.9), (3.7, 0.7), (5.9, 0.6), (8.3, 0.45), (11.0, 0.35)),
             "dip": (0.3, 4, 1.0), "sway": 0.5, "preset": {"warm": 50}},
    "breathe": {"off": "breath", "hz": 0.25, "preset": {"warm": 0}},
    "pulse": {"off": "blink", "hz": 1.0, "dark": True, "preset": {"warm": 0, "amount": 80}},
    "faulty": {"off": "faulty", "hz": 0.15, "dark": True, "preset": {"warm": 0, "amount": 80}},
    "sweep": {"off": "sweep", "hz": 0.25, "dark": True, "preset": {"warm": 0, "amount": 80}},
}
DEFAULTS = {"amount": 50, "speed": 50, "focus": 60, "warm": 50, "edge": 40}
STRENGTH = 2.0  # a flame: how far its light swings at "amount" 100 (its level, about -1..1, times this: stops of e)
DARK = 6.0  # a lamp or a sign going off: how dark at "amount" 100 (e^-6, 0.25% of its light; at 50: 5%)
ROOM = 0.17  # a flame's room light (luma, sRGB) at most, at "focus" 100: no darker place follows the flame's
KNEE = 0.8  # linear light: above it a channel stands for more light than it shows (burnt out at white: about 1.6x)
MOST = 0.996  # (the most a channel shows, before it is read as burnt out)
SWITCH = 0.02  # seconds a lamp or a sign takes to go on or off
RING = 0.03  # a lamp's or a sign's room light is seen round its area, about this share of the frame's height out
WALL = 0.5  # and is no brighter than this (sRGB) at "focus" 100
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
    _num("amount", "Độ nháy (sáng tối mạnh)", DEFAULTS["amount"], group="Nháy"),
    _num("speed", "Tốc độ (chậm ← → nhanh)", DEFAULTS["speed"], group="Nháy"),
    _num("focus", "Chỗ sáng nháy nhiều hơn", DEFAULTS["focus"], group="Ánh sáng"),
    _num("warm", "Ấm lên khi sáng, đỏ đi khi tối", DEFAULTS["warm"], group="Ánh sáng"),
    _num("edge", "Mép mềm", DEFAULTS["edge"], group="Vùng")],
    "note": "Làm một chỗ trong video sáng tối nhấp nháy: quả bí ngô có nến bên trong, đèn lồng, lò sưởi, hay biển "
            "hiệu quảng cáo, đèn neon. (1) Bấm Suốt video. (2) Bấm Vẽ vùng rồi tô lên chỗ cần nhấp nháy (quả bí ngô, "
            "chữ của biển hiệu; tô cả mặt bàn, bức tường quanh nó để ánh sáng trên đó cũng nháy theo). Chưa tô vùng "
            "thì chưa có gì nhấp nháy. (3) Chọn Kiểu nháy, chỉnh Độ nháy và Tốc độ. Chỗ sáng (mắt, miệng bí ngô, chữ "
            "biển hiệu, chỗ được đèn chiếu) nháy rõ, chỗ tối trong vùng gần như yên, nên tô rộng ra cũng được. Biển "
            "hiệu: Bật tắt đều, Chập chờn hay Sáng lần lượt; Độ nháy càng cao thì lúc tắt càng tối.",
    # (what the page says while the area is painted, and when nothing is painted yet)
    "area_tip": "Chỗ tô nhấp nháy, mềm dần từ mép vào: chỉnh Mép mềm trong Bảng chỉnh hiệu ứng.",
    "area_empty": "Chưa tô vùng nên chưa có gì nhấp nháy. Bấm Vẽ vùng rồi tô lên chỗ cần nhấp nháy (quả bí ngô, "
                  "biển hiệu)."}
# (the colour of the light: blue follows it most, red least)
WARM = np.array([-0.15, 0.0, 0.35], np.float32)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _expand(lin):
    """The light a channel stands for (linear, 0..1 shown): itself up to KNEE, more above it (burnt out: about 1.6x)."""
    out = lin.copy()
    over = lin > KNEE
    out[over] = KNEE - (1 - KNEE) * np.log((1 - np.minimum(lin[over], MOST)) / (1 - KNEE))
    return out


def _compress(light):
    """What a channel shows of that light: _expand() undone, whatever it is (above KNEE it nears 1, never past)."""
    out = light.copy()
    over = light > KNEE
    out[over] = 1 - (1 - KNEE) * np.exp((KNEE - light[over]) / (1 - KNEE))
    return out


class Flicker(Effect):
    lens = True
    UNDER = True  # its preview clip: the picture under its area, flickering
    AREA = True  # only inside the clip's area (no area: nothing)
    TURNS = False  # the picture's own light, where it is: nothing to place or turn
    PREVIEW = 960
    VERSION = 2  # (2: stronger, in linear light, only the lit places; lamps and signs go off)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        mode = MODES.get(p["mode"], MODES["candle"])
        self.mode, self.off = mode, mode.get("off")
        # the light's swing: a flame's round the video's own, a lamp's or a sign's down from it (stops of e)
        self.amp = p["amount"] / 100 * (DARK if mode.get("dark") else STRENGTH)
        self.rate = 2 ** ((p["speed"] - 50) / 35)  # 0.37 to 2.7 times as fast
        self.room = p["focus"] / 100 * ROOM  # (sRGB) the room's own light at most: no place this dark follows
        self.warm = WARM * (p["warm"] / 100)
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
        self.sway_k = mode.get("sway", 0.0)
        self.sway = [(self.cycles(hz * self.rate), g.random() * TAU) for hz in (0.31, 0.53)]
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
        """The light at loop phase(s) tau: a flame's about -1 (darkest) .. 1 (brightest) round the video's own; a
        lamp's or a sign's 0 (on: as in the video) .. -1 (off). A sign lit bit by bit is lit where it is along its
        length: pos -1 (where it starts) .. 1 (where it ends), broadcast with tau."""
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

    def _sway_now(self, tau):
        if not self.sway_k:
            return 0.0
        return self.sway_k * 0.5 * sum(math.sin(TAU * m * tau + ph) for m, ph in self.sway)

    # --- where: the area -------------------------------------------------------------------------------------

    def _plan(self, mask):
        if self._plan_of[0] is mask:
            return self._plan_of[1]
        plan = self._make_plan(mask)
        self._plan_of = (mask, plan)
        return plan

    def _make_plan(self, mask):
        """What stays while the area is the same: its box, how strongly each pixel of it follows the light (the
        area, faded in from its outline), and where each column lies across it and each row down it (-1 .. 1: the
        sway, a sign lit bit by bit)."""
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
                "down": along(y1 - y0)[:, None], "ring": self._ring(mask, (x0, y0, x1, y1)) if self.off else None}

    def _ring(self, mask, box):
        """Some of the places just round the area (rows, columns of the frame), where a lamp's or a sign's room light
        is seen (a band about RING of the frame's height wide); None if there are too few (the area fills the frame)."""
        x0, y0, x1, y1 = box
        r = max(2.0, RING * self.h)
        pad = int(math.ceil(2 * r)) + 1
        ex0, ey0, ex1, ey1 = max(0, x0 - pad), max(0, y0 - pad), min(self.w, x1 + pad), min(self.h, y1 + pad)
        m8 = Image.fromarray((np.clip(mask[ey0:ey1, ex0:ex1], 0, 1) * 255 + 0.5).astype(np.uint8))
        near = np.asarray(m8.filter(ImageFilter.GaussianBlur(r))) > 8
        ys, xs = np.nonzero(near & (np.asarray(m8) == 0))
        if len(ys) < 64:
            return None
        step = max(1, len(ys) // 4000)  # (some 4000 of them are enough)
        return ys[::step] + ey0, xs[::step] + ex0

    def release(self):
        self._plan_of = (None, None)

    # --- the picture -------------------------------------------------------------------------------------------

    def draw(self, t, canvas, mask):
        """Nothing is laid over the picture: its own light changes (apply())."""

    def apply(self, t, picture, mask, gain, kept=None):
        """The flicker on `picture` (float32 RGB 0..1, changed in place) inside `mask` (the clip's area, 0..1; None:
        no area, nothing changes), `gain` (0..1, a fade) times as strongly. Where the area is 0 every pixel stays
        exactly as it was. (It never turns: `kept` is not used.)"""
        if mask is None or gain <= 0 or self.amp <= 0:
            return
        plan = self._plan(mask)
        if plan is None:
            return
        tau = self.tau(t)
        x0, y0, x1, y1 = plan["box"]
        out = picture[y0:y1, x0:x1]
        # the light here and now: the flicker, one side brighter as the flame sways; a sign lit along its length
        if self.off == "sweep":
            light = self.level(tau, self.sign * plan[self.axis]).astype(np.float32)
        else:
            light = float(self.level(tau)) + self._sway_now(tau) * plan["across"]
        light = light * (self.amp * gain)
        # each place's light (linear: gamma 2; a burnt-out highlight held more) is the room's own light and the
        # flame's, which follows the flicker. A sign's room light: the wall round it, of the wall's own colour (off,
        # its letters and its glow are the wall again); a flame's: as bright as the room's, of the place's own colour
        # (a pumpkin lit by it stays orange as it dims); never brighter than the place itself
        weight = plan["weight"]
        lin = out * out
        lit = _expand(lin)
        if self.off and plan["ring"] is not None:
            flame = lit - np.minimum(lin, self._wall(picture, plan["ring"]))
        else:
            flame = lit - lin * np.minimum(self._room(lin, weight) / np.maximum(lin @ LUMA, 1e-6), 1.0)[..., None]
        on = weight > 0  # (every other pixel stays exactly as it was)
        for c in range(3):
            more = (np.exp(light * (1.0 + self.warm[c]), dtype=np.float32) - 1.0) * weight
            new = flame[..., c] * more
            new += lit[..., c]
            out[..., c] = np.where(on, np.sqrt(_compress(new)), out[..., c])

    def _wall(self, picture, ring):
        """A lamp's or a sign's room light (linear RGB): the wall round it (its darker places, a third of them darker:
        not its glow where the area was painted short of it; a little brighter at "Chỗ sáng nháy nhiều hơn" 60, so
        its bricks and all stay as they are; none at 0), at most WALL at 100 (a sign in daylight on a bright wall
        still goes dark: only the sign should be painted then)."""
        f = self.p["focus"] / 100
        around = np.percentile(np.square(picture[ring]), 30, axis=0)
        return np.minimum(around * 2.2 * f, (WALL * f) ** 2).astype(np.float32)

    def _room(self, lin, weight):
        """A flame's room light in the area (linear luma; a flame lights what is round it: not that): the area's dark
        places, darker than ROOM at "Chỗ sáng nháy nhiều hơn" 100 (none at 0), as bright as nine in ten of them are
        (the night round a pumpkin); none of them: ROOM itself."""
        most = self.room ** 2
        if most <= 0:
            return 0.0
        step = max(1, int(math.sqrt(weight.size / 20000)))  # (some 20000 places are enough)
        luma = lin[::step, ::step] @ LUMA
        dark = luma[(weight[::step, ::step] > 0) & (luma < most)]
        return float(np.percentile(dark, 90)) if dark.size >= 16 else most
