"""A painted area flickering as if lit by a flame inside it (a carved pumpkin, a lantern, a fireplace, a window lit by
candles), or blinking, breathing, or a faulty lamp (a lens effect: the picture's own pixels get brighter and dimmer).

Inside the clip's area the picture is multiplied by a light level that changes with time. The brighter a pixel is,
the more it follows the light ("Chỗ sáng nháy nhiều hơn": the glowing eyes and mouth of a jack-o'-lantern and the
table it lights flicker, its dark stem hardly), and light burning up is a little whiter, burning down a little
redder ("Ấm lên khi sáng": blue follows the light most, red least). A flame inside also sways: one side of the area
a little brighter, then the other. The area fades in from its outline ("Mép mềm": the effect's own, as the rustling
leaves' rim; the clip's soft edge is not used), and outside it nothing changes at all. Without an area it changes
nothing.

The light is a sum of waves, each a whole number of cycles per loop, so frame P is frame 0 again: a candle's
quick, irregular flutter with now and then a deeper dip (a draught), a fire's faster and larger one, an even blink,
a slow breath, or a lamp that is steady but now and then stutters off. "Tốc độ" makes every wave faster or slower.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from .effects import Effect, _choice, _num

TAU = 2 * math.pi
# each way: the waves (Hz at speed 50, share), dips (Hz, how sharp, how deep), how much the flame sways
MODES = {
    "candle": {"waves": ((0.6, 1.0), (1.1, 0.8), (1.9, 0.6), (3.1, 0.45), (4.7, 0.3), (7.3, 0.2)),
               "dip": (0.17, 8, 1.4), "sway": 0.35},
    "fire": {"waves": ((1.3, 1.0), (2.3, 0.9), (3.7, 0.7), (5.9, 0.6), (8.3, 0.45), (11.0, 0.35)),
             "dip": (0.3, 4, 1.0), "sway": 0.5},
    "pulse": {"blink": 1.0, "sway": 0.0},
    "breathe": {"waves": ((0.25, 1.0),), "sway": 0.0},
    "faulty": {"faulty": (0.12, 9.0), "sway": 0.0},
}
DEFAULTS = {"amount": 50, "speed": 50, "focus": 60, "warm": 50, "edge": 40}

SCHEMA = {"label": "Nhấp nháy vùng", "params": [
    _choice("mode", "Kiểu nháy", "candle",
            [("candle", "Ánh nến lung linh (bí ngô, đèn lồng)"), ("fire", "Lửa bập bùng (lò sưởi, đống lửa)"),
             ("pulse", "Nhấp nháy đều"), ("breathe", "Sáng mờ dần (như thở)"), ("faulty", "Đèn chập chờn")],
            group="Kiểu"),
    _num("amount", "Độ nháy (sáng tối mạnh)", DEFAULTS["amount"], group="Nháy"),
    _num("speed", "Tốc độ (chậm ← → nhanh)", DEFAULTS["speed"], group="Nháy"),
    _num("focus", "Chỗ sáng nháy nhiều hơn", DEFAULTS["focus"], group="Ánh sáng"),
    _num("warm", "Ấm lên khi sáng, đỏ đi khi tối", DEFAULTS["warm"], group="Ánh sáng"),
    _num("edge", "Mép mềm", DEFAULTS["edge"], group="Vùng")],
    "note": "Làm một chỗ trong video sáng tối nhấp nháy, như quả bí ngô có nến bên trong, đèn lồng, lò sưởi. "
            "(1) Bấm Suốt video. (2) Bấm Vẽ vùng rồi tô lên chỗ cần nhấp nháy (quả bí ngô; tô cả mặt bàn quanh nó "
            "nếu muốn ánh sáng trên bàn cũng chập chờn theo). Chưa tô vùng thì chưa có gì nhấp nháy. (3) Chọn Kiểu "
            "nháy, chỉnh Độ nháy và Tốc độ. Chỗ sáng (mắt, miệng bí ngô) nháy nhiều, chỗ tối gần như yên.",
    # (what the page says while the area is painted, and when nothing is painted yet)
    "area_tip": "Chỗ tô nhấp nháy, mềm dần từ mép vào: chỉnh Mép mềm trong Bảng chỉnh hiệu ứng.",
    "area_empty": "Chưa tô vùng nên chưa có gì nhấp nháy. Bấm Vẽ vùng rồi tô lên chỗ cần nhấp nháy (quả bí ngô)."}
# (the colour of the light: blue follows it most, red least)
WARM = np.array([-0.15, 0.0, 0.35], np.float32)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class Flicker(Effect):
    lens = True
    UNDER = True  # its preview clip: the picture under its area, flickering
    AREA = True  # only inside the clip's area (no area: nothing)
    TURNS = False  # the picture's own light, where it is: nothing to place or turn
    PREVIEW = 960
    VERSION = 1

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        mode = MODES.get(p["mode"], MODES["candle"])
        self.mode = mode
        self.amp = p["amount"] / 100 * 0.55  # the light's swing at full flicker (one way)
        self.rate = 2 ** ((p["speed"] - 50) / 35)  # 0.37 to 2.7 times as fast
        self.focus = p["focus"] / 100
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
        self.sway_k = mode["sway"]
        self.sway = [(self.cycles(hz * self.rate), g.random() * TAU) for hz in (0.31, 0.53)]
        self.blink = self.cycles(mode["blink"] * self.rate) if "blink" in mode else None
        self.faulty = None
        if "faulty" in mode:
            hz, fast = mode["faulty"]
            self.faulty = (self.cycles(hz * self.rate), self.cycles(fast * self.rate), g.random() * TAU)
        self.norm = self._norm()
        self._plan_of = (None, None)

    # --- the light -----------------------------------------------------------------------------------------------

    def _raw(self, tau):
        """The light's change at loop phase(s) tau, before it is scaled (about -1..1)."""
        tau = np.asarray(tau, np.float64)
        s = np.zeros_like(tau)
        for m, share, ph in self.waves:
            s += share * np.sin(TAU * m * tau + ph)
        if self.dip is not None:  # now and then a draught: the flame ducks, then flares back
            m1, m2, sharp, deep, (p1, p2) = self.dip
            gate = (0.5 + 0.5 * np.sin(TAU * m1 * tau + p1)) * (0.5 + 0.5 * np.sin(TAU * m2 * tau + p2))
            s -= deep * gate ** sharp * 4
        if self.blink is not None:  # on and off, the edges a little soft
            s += np.tanh(3 * np.sin(TAU * self.blink * tau)) / math.tanh(3)
        if self.faulty is not None:  # steady, then now and then it stutters off for a moment
            m, fast, ph = self.faulty
            drop = (0.5 + 0.5 * np.sin(TAU * m * tau + ph)) ** 24
            s -= 2 * np.minimum(1.0, 3 * drop) * (0.6 + 0.4 * np.sin(TAU * fast * tau))
        return s

    def _norm(self):
        """(mean, scale): the light's change over the loop set to a mean of 0 and a swing of about +-1."""
        tau = (np.arange(4096) + 0.5) / 4096
        s = self._raw(tau)
        if self.blink is not None or self.faulty is not None:
            lo, hi = float(s.min()), float(s.max())  # (on and off: from its darkest to its brightest)
            return (lo + hi) / 2, max((hi - lo) / 2, 1e-6)
        return float(s.mean()), max(2.0 * float(s.std()), 1e-6)

    def level(self, tau):
        """The light at loop phase(s) tau: about -1 (darkest) .. 1 (brightest)."""
        mean, scale = self.norm
        return np.clip((self._raw(tau) - mean) / scale, -1.5, 1.5)

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
        area, faded in from its outline), and where each column lies across it (-1 left .. 1 right: the sway)."""
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
        across = np.linspace(-1.0, 1.0, x1 - x0, dtype=np.float32)[None, :] if x1 - x0 > 1 else \
            np.zeros((1, 1), np.float32)
        return {"box": (x0, y0, x1, y1), "weight": m, "across": across}

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
        # the light here and now: the flicker, one side brighter as the flame sways
        light = (float(self.level(tau)) + self._sway_now(tau) * plan["across"]) * (self.amp * gain)
        # how much each pixel follows it: the area, and more where it is bright (lit by the light that flickers)
        luma = out @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        follow = plan["weight"] * ((1 - self.focus) + self.focus * _smooth((luma - 0.12) / 0.6))
        d = follow * light  # (rows, cols): the change of light at each pixel
        for c in range(3):
            factor = 1.0 + d * (1.0 + self.warm[c])
            np.maximum(factor, 0.0, out=factor)
            out[..., c] *= factor
        np.clip(out, 0.0, 1.0, out=out)
