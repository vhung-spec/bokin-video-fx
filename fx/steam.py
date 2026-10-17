"""Smoke and steam rising from one place: steam over a hot cup of coffee or tea, the smoke of an incense stick, steam
over a pot or a kettle (a placed effect: Effect.BOX; see effects.py for how every effect loops).

Smoke is drawn as dye carried by moving air. Points over the surface (the liquid in the cup or the pot, the glowing
tip of the incense) let go of a little smoke all the time, the air carries it up, and what shows are the streaklines:
the curves through all the smoke each point let go of over the last seconds, which is how real steam and smoke look.

The air is a warm draught: slow at the surface, quicker above it. Over a cup or a pot it sweeps the steam of the whole
surface into a few wisps where the surface steams most (they wander over it and come and go: no fixed strands), then
the wisps into a plume that narrows a little and spreads as it rises. Waves of air drift up with it: long ones sway
it (at rates that never quite repeat), short ones grow with height so that no two wisps move as one. Above a calm
stretch (a long one for incense: a smooth thin column first) small whirls are born that roll the wisps into curls and
hooks; each whirl is a point carried by the air like the smoke (so it stays with the plume), rising a little slower,
turning more slowly as it ages. A pot lets its steam go in puffs, each with a pair of whirls that rolls it up as it
rises. A breeze leans the plume and sways it now and then; near the box's side it rises instead (and evaporates).

Each bit of smoke keeps its amount: a wisp is denser where the air squeezes it and fainter where it is stretched. It
blurs with the square root of its age as it mixes with the air (incense threads stay sharp for long), and steam
evaporates: it thins and tears as it rises rather than turning into fog, so it is densest near the surface and never
piles up into a lump. Drawn on a few layers, each blurred twice as much as the one before, the blurred ones on coarser
grids (a pyramid: as quick as drawing them all fine).

Lit by the scene (light=: effects.light_map of the video, as rain): steam shows the light that falls on it, the
brightest light near where it rises (a lamp-lit cup, a window behind it), its wisps lighter on the side the light comes
from, plus the light behind it scattered forward (backlit steam glows); each channel clipped at white. Its own colour
then only tints that light ("lit" mixes in its own colour). A placed effect is lit by the light where its box stands
upright (light_at(): a few numbers), however it is turned, so its preview clip is lit exactly as the video.

The air moves on a clock of "ticks" (TICKS a second, a whole number per loop), and every bit of smoke (and every
whirl) is followed tick by tick from where it was let go; a frame lies between two ticks. The air, the points, the
whirls' births and how much smoke the points let go of are functions of the tick modulo the loop (whole numbers of
cycles), so frame P is frame 0 again, exactly. Frames drawn one after the other carry the smoke on a tick at a time;
a frame drawn on its own starts a whole lifetime of the smoke earlier (the same sums, so the same picture).

The box: the place the smoke rises from (a cup's mouth: an ellipse MARK as high as wide, "rim" wide) sits in the middle
of its bottom, the smoke rising to its top. Inside it every length is in box heights, so the small preview clip (the
box alone) and the video show the same smoke.
"""
import math

import numpy as np
from PIL import Image

from .effects import LUMA, Effect, _choice, _colour, _lerp, _num, _place, _rgb, linear, srgb

TICKS = 20  # the air's clock, ticks a second
MARK = 0.22  # the place the smoke rises from (the page's mark): an ellipse this flat (height / width), sitting on the
# box's bottom edge
GRID_MAX = 640  # rows: the smoke is drawn at its box's size, at most this fine (the preview clip's height)
BANDS = 7  # how blurred: smoke is drawn on this many layers at most, each blurred twice as much as the one before
POINTS = 320  # random draws kept for the points that let go of smoke (the seed fixes them)
PLUMES = 12  # and for the places over the surface that let go of more
WHIRLS = 8  # columns of whirls at most
MOST_SPIN = 7.0  # radians a second at most a whirl turns (a tick turns a bit of smoke less than 0.35 radians)
DENSITY = 0.03  # how much a bit of smoke covers
ALBEDO = 0.85  # lit steam: this much of the light that falls on it
FORWARD = 0.35  # and of the light behind it (droplets scatter forward: backlit steam glows)

# Each style (the sliders are its own: at the same settings each looks as it should):
#   speed     how fast it rises, box heights a second, at speed 0 and 100
#   points    how many points let go of it (at rim 50; a wider rim more, as close together)
#   plumes    how many places over the surface let go of more (wisps; a pot's puffs: "puffs"), at amount 0 and 100;
#   veil      and how much the rest of the surface lets go of
#   gather    how strongly the draught over the surface sweeps it into the wisps; neck: the wisps into a plume
#   calm      the smooth stretch over the surface (share of its reach); stir: how much the air moves right over it
#   wander    how far the points wander over the surface (share of half the rim)
#   sway, wave, eddy, fine: the waves of air (long ones and their wavelengths, middle ones, short ones)
#   whirls, spin, rc: how many columns of whirls, how fast they turn, how big their cores are (box heights)
#   s0, grow, widen: its blur (box heights) at the start, with the square root of age (at soft 0 and 100), and with
#             age after a second (incense: threads that spread into veils)
#   evap, fade, show: how soon it evaporates, from what share of its reach it fades, how long it takes to show
#   mass      how much a point lets go of; grid: rows it is drawn on at most; colour, rim: its own (style presets)
STYLES = {
    "cup": {"speed": (0.16, 0.6), "points": 64, "plumes": (3, 7), "veil": (0.04, 0.2), "gather": 0.45, "neck": 0.4,
            "calm": 0.05, "stir": 0.0, "wander": 0.18, "sway": 1.0, "wave": (0.35, 0.9), "eddy": 1.0, "fine": 1.0,
            "whirls": 4, "spin": 1.0, "rc": 0.045, "s0": (0.0009, 0.0035), "grow": (0.0035, 0.012), "widen": 0.0,
            "evap": 1.0, "fade": 0.4, "show": 0.25, "puffs": False, "mass": 0.065, "colour": "#f4f1ec", "rim": 50},
    "incense": {"speed": (0.12, 0.45), "points": 6, "plumes": (0, 0), "veil": (1.0, 1.0), "gather": 0.0, "neck": 0.0,
                "calm": 0.25, "stir": 0.0, "wander": 0.0, "sway": 2.2, "wave": (0.1, 0.2), "eddy": 0.8, "fine": 0.7,
                "whirls": 2, "spin": 1.6, "rc": 0.04, "s0": (0.0004, 0.0013), "grow": (0.0003, 0.0022),
                "widen": 0.0012, "evap": 0.2, "fade": 0.7, "show": 0.1, "puffs": False, "mass": 0.065, "grid": 1080,
                "colour": "#ccd3dd", "rim": 3},
    "pot": {"speed": (0.25, 0.95), "points": 110, "plumes": (4, 9), "veil": (0.4, 0.9), "gather": 0.0, "neck": 0.0,
            "calm": 0.0, "stir": 0.3, "wander": 0.4, "sway": 0.8, "wave": (0.35, 0.9), "eddy": 1.1, "fine": 1.2,
            "whirls": 8, "spin": 1.0, "rc": 0.06, "s0": (0.003, 0.007), "grow": (0.008, 0.02), "widen": 0.0,
            "evap": 1.8, "fade": 0.3, "show": 0.05, "puffs": True, "mass": 0.6, "colour": "#f2f2f0", "rim": 85},
}

DEFAULTS = {"amount": 45, "reach": 80, "spread": 40, "speed": 35, "curl": 40, "breeze": 6, "gust": 25, "soft": 40,
            "opacity": 55}
# the looks: how it moves and how much of it there is, for any style (not its place, rim or colour)
LOOKS = {
    "light": {},
    "thick": {"amount": 90, "spread": 55, "speed": 40, "curl": 45, "soft": 55, "opacity": 75},
    "long": {"amount": 50, "reach": 100, "spread": 35, "speed": 25, "curl": 60, "gust": 30, "soft": 45},
    "breeze": {"amount": 50, "reach": 85, "spread": 50, "curl": 45, "breeze": 50, "gust": 30},
}
LOOKS = {k: {**DEFAULTS, **v} for k, v in LOOKS.items()}
# a style: its own rim and colour, and the first look (the style itself sets how it moves: the sliders are its own)
STYLE_PRESETS = {k: {**LOOKS["light"], "look": "light", "rim": st["rim"], "color": st["colour"]} for k, st in
                 STYLES.items()}
LIT = 70  # how much it is lit by the scene, by default


def _p(key, label, value, group, lo=0, hi=100):
    return _num(key, label, value, lo, hi, group=group)


SCHEMA = {
    "label": "Khói",
    "note": "Chọn kiểu khói: hơi nóng bốc lên từ cốc cà phê, trà; khói nhang; hay hơi nước từ nồi, ấm. Kéo khung trắng "
            "trên hình đến chỗ khói bốc lên: hình bầu dục vàng nằm ở đáy khung, đặt nó trùng miệng cốc (mặt nồi), hoặc "
            "chấm vàng vào đầu que nhang. Kéo góc khung cho hình vàng rộng bằng miệng cốc. Khói chỉ bay trong khung. "
            "Khói tự sáng theo đèn trong cảnh (đèn bên cạnh, cửa sổ phía sau).",
    # for the page: a mark where the smoke rises from, a flat ellipse (`ratio`: height / width) sitting on the middle
    # of the box's bottom edge, inside the box, as wide as the setting "rim" (% of the box's width), and a tip
    "mark": {"width": "rim", "ratio": MARK, "sits": True, "label": "chỗ khói bốc lên",
             "tip": "Đặt hình bầu dục vàng trùng miệng cốc (mặt nồi), hoặc chấm vàng vào đầu que nhang, rồi kéo góc "
                    "khung cho hình vàng rộng bằng miệng cốc."},
    "params": [
        _choice("style", "Kiểu khói", "cup",
                [("cup", "Hơi nóng cốc cà phê, trà"), ("incense", "Khói nhang"), ("pot", "Hơi nước nồi, ấm")],
                presets=STYLE_PRESETS, group="Kiểu"),
        _choice("look", "Dáng khói", "light",
                [("light", "Nhẹ nhàng"), ("thick", "Đậm"), ("long", "Bay cao, lượn dài"),
                 ("breeze", "Bay nghiêng theo gió")], presets=LOOKS, group="Kiểu"),
        _p("amount", "Lượng khói", DEFAULTS["amount"], "Khói"),
        _p("rim", "Độ rộng chỗ khói bốc lên (so với khung trắng)", STYLE_PRESETS["cup"]["rim"], "Khói", 1, 100),
        _p("reach", "Khói bay cao (so với khung trắng)", DEFAULTS["reach"], "Khói", 20, 100),
        _p("spread", "Độ loe khi bay lên", DEFAULTS["spread"], "Khói"),
        _p("soft", "Độ mềm, nhoè", DEFAULTS["soft"], "Khói"),
        _p("speed", "Tốc độ bay lên", DEFAULTS["speed"], "Chuyển động"),
        _p("curl", "Uốn lượn, cuộn xoáy", DEFAULTS["curl"], "Chuyển động"),
        _p("breeze", "Gió nghiêng (trái ← → phải)", DEFAULTS["breeze"], "Gió", -100, 100),
        _p("gust", "Gió lay qua lại", DEFAULTS["gust"], "Gió"),
        _p("opacity", "Độ đậm", DEFAULTS["opacity"], "Màu và ánh sáng"),
        _p("lit", "Sáng theo đèn trong cảnh (0: chỉ một màu)", LIT, "Màu và ánh sáng"),
        _colour("color", "Màu khói", STYLES["cup"]["colour"], group="Màu và ánh sáng"),
        *_place(45)]}


def upgrade(raw):
    """Settings saved before the smoke had styles (no "style"): steam over a cup, its settings as they were (drawn the
    new way). (Its box is moved down so its mark stays on the cup: upgrade_in(), with the video's shape.)"""
    if not isinstance(raw, dict) or "style" in raw:
        return raw
    return dict(raw, style="cup")


def upgrade_in(raw, w, h):
    """Settings saved before the mark sat inside the box (no "style"), for a video w x h: the box moved down by half
    the mark's height, so the mark (once centred on the box's bottom edge) stays where it was on the cup."""
    if not isinstance(raw, dict) or "style" in raw:
        return raw
    try:
        rim, scale, y = (float(raw.get(k, d)) for k, d in (("rim", 50), ("scale", 45), ("y", 0)))
    except (TypeError, ValueError, OverflowError):
        return upgrade(raw)
    if not all(map(math.isfinite, (rim, scale, y))) or w <= 0 or h <= 0:
        return upgrade(raw)
    box = min(w, h) * scale / 100  # the box's width (BOX: square), pixels
    return dict(upgrade(raw), y=y + MARK / 2 * rim / 100 * box / h * 100)


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _blur_small(a, sigma):
    """A Gaussian blur of `a` (rows, columns, float32) by a small `sigma` (pixels, under about 2.5), zero beyond the
    edges."""
    if sigma < 0.3:
        return a
    r = int(math.ceil(2.25 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2).astype(np.float32)
    k /= k.sum()
    for axis in (0, 1):
        n = a.shape[axis]
        out = a * k[r]
        for i in range(1, min(r, n - 1) + 1):
            if axis == 0:
                out[i:] += k[r - i] * a[:-i]
                out[:-i] += k[r + i] * a[i:]
            else:
                out[:, i:] += k[r - i] * a[:, :-i]
                out[:, :-i] += k[r + i] * a[:, i:]
        a = out
    return a


def _double(a, shape):
    """`a` made `shape` (about twice as big), bilinear."""
    return np.array(Image.fromarray(a, "F").resize((shape[1], shape[0]), Image.Resampling.BILINEAR))


def light_at(light, rect, w, h):
    """What lights steam whose box stands at `rect` (x0, y0, bw, bh: pixels) in a w x h frame, from the scene's light
    (effects.light_map over that frame): 7 numbers, linear light. The light falling on it (3: the brightest light near
    where it rises: a lamp-lit cup, a lamp, a window behind it), the light behind it (3: around where it rises), and
    the side the light comes from (-1: from the left, 1: from the right; 0: from the front or all round)."""
    lh, lw = light.shape[:2]
    x0, y0, bw, bh = (float(v) for v in rect)

    def part(ax0, ay0, ax1, ay1, channels):
        c0 = int(np.clip(math.floor(ax0 / w * lw), 0, lw - 1))
        r0 = int(np.clip(math.floor(ay0 / h * lh), 0, lh - 1))
        c1 = int(np.clip(math.ceil(ax1 / w * lw), c0 + 1, lw))
        r1 = int(np.clip(math.ceil(ay1 / h * lh), r0 + 1, lh))
        return light[r0:r1, c0:c1, channels].reshape(-1, 3)

    here = slice(3, 6)
    near = part(x0 - 0.25 * bw, y0 - 0.25 * bh, x0 + 1.25 * bw, y0 + 1.25 * bh, here)  # the box, and around it
    lum = near @ LUMA
    top = lum >= np.percentile(lum, 95)
    key = near[top].mean(axis=0) if top.any() else near.mean(axis=0)
    back = part(x0 + 0.15 * bw, y0 + 0.1 * bh, x0 + 0.85 * bw, y0 + 0.9 * bh, slice(0, 3)).mean(axis=0)
    left = float(part(x0 - 0.5 * bw, y0 + 0.2 * bh, x0 + 0.4 * bw, y0 + 1.3 * bh, here).mean(axis=0) @ LUMA)
    right = float(part(x0 + 0.6 * bw, y0 + 0.2 * bh, x0 + 1.5 * bw, y0 + 1.3 * bh, here).mean(axis=0) @ LUMA)
    side = (right - left) / max(right + left, 1e-6)
    return np.array([*key, *back, side], np.float32)


class Steam(Effect):
    BOX = (1, 1)
    VERSION = 3  # (3: styles, lit by the scene: its preview clips are made again)
    LIT = True
    upgrade = staticmethod(upgrade)
    upgrade_in = staticmethod(upgrade_in)
    light_at = staticmethod(light_at)

    def __init__(self, *a, light=None):
        super().__init__(*a)
        p, rng = self.p, self.rng
        st = self.style = STYLES.get(p["style"], STYLES["cup"])
        self.kind = p["style"]
        # the random draws (fixed by the seed: a slider moves what it should, not the rest)
        u = rng.random((POINTS, 12))
        pu = rng.random((PLUMES, 12))
        m = rng.random((14, 5))
        g = rng.random(12)
        wu = rng.random((WHIRLS, 8))
        self.bw = self.BOX[0] / self.BOX[1]  # the box's width, in box heights
        self.ticks = max(1, int(round(TICKS * self.loop_s)))  # ticks per loop
        self.dt = self.loop_s / self.ticks  # seconds a tick
        amount = p["amount"] / 100
        curl = _lerp(0.15, 1.0, p["curl"] / 100)  # (never quite still: air in a room always moves a little)
        self.speed = V = _lerp(*st["speed"], p["speed"] / 100)  # box heights a second
        self.reach = reach = p["reach"] / 100
        life = 1.3 * reach / V + 0.5  # seconds a bit of smoke is followed
        self.L = int(math.ceil(life / self.dt)) + 2  # ticks
        self.life = self.L * self.dt
        # where it rises from: an ellipse on the box's bottom edge, rim wide (box heights), a little inside it (the
        # liquid's surface); a wider rim has more points, as close together
        rim = p["rim"] / 100 * self.bw
        self.a = rim / 2 * 0.9  # half its width
        self.e = MARK / 2 * rim  # half its height
        self.yc = self.e  # its middle, above the box's bottom
        wide = max(1.0, p["rim"] / 50) if st["plumes"][1] else 1.0
        n = self.n = min(POINTS, int(round(st["points"] * wide)))
        u = u[:n]
        r, th = np.sqrt(u[:, 0]), u[:, 1] * 2 * math.pi
        self.ux = (r * np.cos(th)).astype(np.float32)  # where on the surface (-1..1 across it)
        self.sy = (self.yc + 0.85 * self.e * r * np.sin(th)).astype(np.float32)  # the back of it is higher
        if self.kind == "incense":  # a point: the glowing tip (a hair apart, they part higher up)
            self.ux = ((u[:, 0] - 0.5) * 0.12).astype(np.float32)
            self.sy = np.full(n, self.yc, np.float32)
        self.lift = _lerp(0.9, 1.1, u[:, 2]).astype(np.float32)  # some rise a little quicker
        self.bright = (0.6 + 0.8 * u[:, 3]).astype(np.float32)
        self.fuzz = np.exp(_lerp(math.log(0.7), math.log(2.2), u[:, 11]) * (self.kind != "incense")).astype(
            np.float32)  # (some strands crisp, some soft)
        # each point wanders over the surface (two rates, whole cycles a loop)
        self.wan = (st["wander"] * u[:, 4]).astype(np.float32)
        self.wm = np.stack([self.cycles(_lerp(0.03, 0.09, u[:, 5])), self.cycles(_lerp(0.09, 0.22, u[:, 6]))]
                           ).astype(np.int64)
        self.wphi = u[:, 7:9].T * 2 * math.pi
        self.fm = self.cycles(_lerp(0.3, 0.9, u[:, 9])).astype(np.int64)  # a flicker of each
        self.fphi = u[:, 10] * 2 * math.pi
        # places where the surface lets go of more: they wander over it, breathe, come and go (a pot: puffs)
        lo, hi = st["plumes"]
        nw = self.nw = int(round(_lerp(lo, hi, amount))) if hi else 0
        pu = pu[:nw]
        self.pc = ((np.arange(nw) * 0.6180339887 + g[0]) % 1.0 * 2 - 1) * 0.6  # where, -1..1
        self.pa = _lerp(0.25, 0.55, pu[:, 0])  # how far they wander
        self.pm = np.stack([self.cycles(_lerp(0.015, 0.05, pu[:, 1])), self.cycles(_lerp(0.05, 0.12, pu[:, 2]))]
                           ).astype(np.int64)
        self.pphi = pu[:, 3:5].T * 2 * math.pi
        self.pw = _lerp(0.12, 0.24, pu[:, 5]) / max(1.0, wide ** 0.5)  # how wide, -1..1 units
        self.em = np.stack([self.cycles(_lerp(0.03, 0.08, pu[:, 6])), self.cycles(_lerp(0.1, 0.25, pu[:, 7]))]
                           ).astype(np.int64)
        self.ephi = pu[:, 8:10].T * 2 * math.pi
        if st["puffs"]:  # a pot: puffs, quick ones
            self.em = np.stack([self.cycles(_lerp(0.3, 0.6, pu[:, 6])), self.cycles(_lerp(0.6, 1.1, pu[:, 7]))]
                               ).astype(np.int64)
            self.pm = np.stack([self.cycles(_lerp(0.03, 0.08, pu[:, 1])), self.cycles(_lerp(0.08, 0.2, pu[:, 2]))]
                               ).astype(np.int64)
        self.veil = _lerp(*st["veil"], amount)
        # bands of more and less steam that sweep across the surface (a floor: it breathes, it never goes)
        self.bm = self.cycles(np.array([_lerp(0.1, 0.3, g[1]), _lerp(0.25, 0.5, g[2])])).astype(np.int64)
        self.bk = np.where(g[3:5] < 0.5, -1, 1) * math.pi * _lerp(0.6, 1.8, g[3:5])
        self.bphi = g[5:7] * 2 * math.pi
        # how much smoke a point lets go of a tick: a strand as dense at any speed (a quicker one spreads it over a
        # longer path), a little denser when softer (it spreads wider)
        self.gain = (st["mass"] * _lerp(0.55, 1.0, amount) * V * self.dt * math.sqrt(_lerp(*st["s0"], 0.4) * 500) *
                     st["points"] / n)
        self.gather = st["gather"] * (1 - 0.5 * amount ** 2)  # into the wisps: a lot of steam rises wider
        self.neck = st["neck"] * (1 - 0.6 * amount ** 2)
        # the waves of air: long ones that sway it, middle ones, short ones; drifting up with it
        groups = (("sway", 3, st["wave"], (0.0, 0.35), st["sway"] * 0.4),
                  ("eddy", 6, (0.11, 0.32), (0.45, 1.25), st["eddy"] * 0.62),
                  ("fine", 5, (0.035, 0.075), (0.3, 1.3), st["fine"] * 0.32))
        kx, ky, amp, om, phi = [], [], [], [], []
        i = 0
        for _, count, (l0, l1), (a0, a1), strength in groups:
            mm = m[i:i + count]
            i += count
            lam = np.exp(_lerp(math.log(l0), math.log(l1), mm[:, 0]))  # wavelengths, box heights
            ang = np.where(mm[:, 1] < 0.5, -1, 1) * _lerp(a0, a1, (2 * mm[:, 1]) % 1.0)  # from straight up
            k = 2 * math.pi / lam
            kx.append(k * np.sin(ang))
            ky.append(k * np.abs(np.cos(ang)))
            vel = V * curl * strength * (lam / 0.3) ** 0.3 / math.sqrt(count / 3)
            amp.append(vel / k)
            drift = V * _lerp(0.55, 1.0, mm[:, 2])  # how fast they go up
            om.append(self.cycles(drift * ky[-1] / (2 * math.pi) + 0.02 * mm[:, 4]))
            phi.append(mm[:, 3] * 2 * math.pi)
        kx, ky, amp = np.concatenate(kx), np.concatenate(ky), np.concatenate(amp)
        self.kx, self.ky = kx.astype(np.float32), ky.astype(np.float32)
        self.amp = np.stack([amp, -amp * ky, -amp * kx], 1).astype(np.float32)  # psi, d psi / dy, d psi / dx
        self.om = np.concatenate(om).astype(np.int64)
        self.mphi = np.concatenate(phi)
        ends = np.cumsum([count for _, count, _, _, _ in groups])
        self.groups = [slice(e - count, e) for e, (_, count, _, _, _) in zip(ends, groups)]
        self.calm = st["calm"] * reach  # (box heights) the smooth stretch over the surface
        # whirls: columns of points carried by the air, a few of them turning (born a whole number of times a loop)
        nwh = self.nwh = min(WHIRLS, st["whirls"])
        wu = wu[:nwh]
        self.rc = st["rc"] * _lerp(0.8, 1.25, wu[:, 0]) * (reach / 0.8) ** 0.5  # their cores, box heights
        spin = st["spin"] * curl * V / self.rc * 2.6
        self.spin = np.minimum(spin, MOST_SPIN).astype(np.float32)
        births = self.cycles(_lerp(0.35, 0.7, wu[:, 1]) * (V / 0.3) * (0.8 / max(reach, 0.2)))
        self.born = np.zeros((nwh, self.ticks), bool)  # which ticks a whirl is born at, in each column
        self.wsign = np.zeros((nwh, self.ticks), np.float32)  # and which way and how strongly it turns
        for c in range(nwh):
            nb = int(min(births[c], self.ticks))
            at = (np.floor((np.arange(nb) + wu[c, 2]) * self.ticks / nb).astype(np.int64)) % self.ticks
            crng = np.random.default_rng([self.seed, 2207, c])
            draws = crng.random((nb, 2))
            self.born[c, at] = True
            if self.kind == "incense":  # alternately on each side: they roll the column into a wave of curls
                sign = np.where(np.arange(nb) % 2 == 0, 1.0, -1.0) * (1 if c == 0 else -1)
            elif st["puffs"]:  # a pair under each puff, rolling it up (left one this way, right one that way)
                sign = np.full(nb, 1.0 if c % 2 == 0 else -1.0)
            else:
                sign = np.where(draws[:, 0] < 0.5, -1.0, 1.0)
            self.wsign[c, at] = sign * _lerp(0.6, 1.25, draws[:, 1])
        self.wside = np.where(np.arange(nwh) % 2 == 0, -1.0, 1.0).astype(np.float32)
        # the breeze: a lean, and a sway now and then (rates that do not keep in step)
        self.lean = p["breeze"] / 100 * 1.1 * V
        self.sway = p["gust"] / 100 * 0.5 * V
        self.gm = self.cycles(np.array([_lerp(0.035, 0.07, g[7]), _lerp(0.09, 0.16, g[8]), _lerp(0.2, 0.32, g[9])])
                              ).astype(np.int64)
        self.gphi = g[9:12] * 2 * math.pi
        self.flare = _lerp(0.0, 1.0, p["spread"] / 100) * V * {"cup": 1.0, "incense": 0.55, "pot": 1.1}[self.kind]
        # blur: thin at the surface, more with the square root of age
        soft = p["soft"] / 100
        self.s0 = _lerp(*st["s0"], soft)
        self.grow = _lerp(*st["grow"], soft) * _lerp(0.75, 1.3, p["spread"] / 100)
        self.evap = st["evap"]
        # colour: its own, or lit by the scene
        self.colour = _rgb(p["color"])
        self.opacity = p["opacity"] / 100
        if self.kind == "incense":  # (its preview clip: thin threads need a little more quality)
            self.PREVIEW_CRF = 28
        self.tone, self.side = None, 0.0
        if light is not None and p["lit"] > 0:
            light = np.asarray(light, np.float32)
            self.tone = light if light.ndim == 1 else light_at(light, self.box(), self.w, self.h)
        self.colours = self._colours()
        # backlit steam glows (droplets scatter forward): more of it shows before a bright light
        self.glow = 1.0
        if self.tone is not None:
            self.glow = 1 + 2.5 * p["lit"] / 100 * min(1.0, float(self.tone[3:6] @ LUMA) / 0.5)
        self.states, self.emitted = {}, {}

    # --- the air -----------------------------------------------------------------------------------------------
    def _phase(self, cycles, k):
        """2 pi cycles k / ticks, from whole numbers (so tick k + ticks gives exactly the same)."""
        return (np.asarray(cycles, np.int64) * k % self.ticks) * (2 * math.pi / self.ticks)

    def _breeze(self, k):
        ph = self._phase(self.gm, k) + self.gphi
        return self.lean + self.sway * (0.55 * math.sin(ph[0]) + 0.3 * math.sin(ph[1]) + 0.15 * math.sin(ph[2]))

    def _plume_state(self, j):
        """Where the places that let go of more are at ticks j (an array: one row each; -1..1 across the surface)
        and how much they let go of (0..1): they wander, breathe, come and go (a pot's puffs: quickly)."""
        j = np.asarray(j)[..., None]
        ph = self._phase(self.pm[:, None], j[None]) + self.pphi[:, None]
        c = np.clip(self.pc + self.pa * (0.65 * np.sin(ph[0]) + 0.35 * np.sin(ph[1])), -0.9, 0.9)
        eh = self._phase(self.em[:, None], j[None]) + self.ephi[:, None]
        e = 0.5 + 0.5 * (0.6 * np.sin(eh[0]) + 0.4 * np.sin(eh[1]))
        e = _smooth(0.2, 0.8, e) if self.style["puffs"] else 0.25 + 0.75 * _smooth(0.15, 0.7, e)
        return c, e

    def _plumes(self, k):
        """Where the places that let go of more are at tick k (box heights across), how wide, how much (0..1)."""
        if not self.nw:
            return None
        c, e = self._plume_state([k])
        return c[0] * self.a, self.pw * self.a, e[0]

    def _velocity(self, x, y, age, k, whirls, plumes, lift=1.0):
        """The air's velocity (box heights a second) at x, y for smoke `age` seconds old, at tick k (`lift`: how much
        quicker each point's smoke rises)."""
        V, reach = self.speed, self.reach
        up = lift * V * (0.3 + 0.7 * (1 - np.exp(-age / 0.4)))  # slow at the surface, quicker above it
        up = up * (1 - 0.2 * _smooth(0.3 * reach, reach, y))
        h = np.maximum(y - self.yc, 0.0)
        # a breeze: the surface shelters what is just above it; near the box's side it rises instead
        lean = self._breeze(k)
        if self.calm > 0.1 * reach:  # (a smooth column first: only a steady breeze bends it there)
            lean = self.lean + (lean - self.lean) * _smooth(0.5 * self.calm, self.calm, h)
        vx = lean * (0.05 + 0.95 * _smooth(0.0, 0.2, h)) * (1 - _smooth(0.22, 0.44, x * np.sign(lean)))
        axis = self.lean * np.maximum(y - 0.12, 0.0) / V  # where the leaning plume is, roughly
        vx = vx + self.flare * (x - axis) * _smooth(0.04, 0.45 * reach, h)  # it spreads as it rises
        if plumes is not None:  # the draught over the surface sweeps the steam into the wisps
            c, wdt, e = plumes
            near = np.exp(-h / (0.02 + 0.5 * self.e))
            d = x[..., None] - c
            pull = (e * self.gather) * d * np.exp(-0.5 * (d / (1.8 * wdt + 0.15 * self.a)) ** 2)
            vx = vx - pull.sum(axis=-1) * near
            mid = float((c * e).sum() / max(e.sum(), 1e-6)) * 0.5
            vx = vx - self.neck * (x - mid) * np.exp(-h / 0.12) * _smooth(0.0, 0.04, h)
        # the waves: long ones sway it from low down, the others grow above the calm stretch
        th = (x.astype(np.float32)[..., None] * self.kx + y.astype(np.float32)[..., None] * self.ky -
              (self._phase(self.om, k) + self.mphi).astype(np.float32))
        co, si = np.cos(th), np.sin(th)
        calm, stir = self.calm, self.style["stir"]  # (stir: how much of them there is right over the surface)
        for gi, idx in enumerate(self.groups):
            if gi == 0:
                q0, q1, base = calm, calm + 0.25 * reach, max(stir, 0.25 if calm < 0.1 * reach else 0.0)
            elif gi == 1:
                q0, q1, base = calm, calm + 0.3 * reach, stir
            else:
                q0, q1, base = calm * 1.1, calm + 0.22 * reach, stir
            q = np.clip((h - q0) / (q1 - q0), 0.0, 1.0)
            gy = base + (1 - base) * q * q * (3 - 2 * q)
            dg = (1 - base) * 6 * q * (1 - q) / (q1 - q0)
            if gi:  # (calm again high up, for the middle and short ones: it fades there)
                z = np.clip((h - 0.75 * reach) / (0.25 * reach), 0.0, 1.0)
                gy = gy * (1 - 0.4 * z * z)
            a = self.amp[idx]
            psi = co[..., idx] @ a[:, 0]
            sg = si[..., idx]
            vx = vx + gy * (sg @ a[:, 1]) + dg * psi
            up = up - gy * (sg @ a[:, 2])
        if whirls is not None:  # the whirls roll it
            wx, wy, wo, wr = whirls
            dx = x[..., None] - wx
            dy = y[..., None] - wy
            f = 1.0 / (1.0 + (dx * dx + dy * dy) / wr)
            f *= f
            f *= wo
            vx = vx - (f * dy).sum(axis=-1)
            up = up + (f * dx).sum(axis=-1)
        return vx, up

    def _whirls(self, wx, wy, k, rows):
        """The whirls turning at tick k: where (from the columns of whirl points: their first `rows` rows, the others
        not let go yet when starting from scratch), how fast, how big."""
        if not self.nwh:
            return None
        b = (k - np.arange(rows)) % self.ticks
        on = self.born[:, b]  # (columns, rows)
        if not on.any():
            return None
        c, a = np.nonzero(on)
        a = np.minimum(a, self.L - 1)
        x, y = wx[a, c], wy[a, c]
        age = a * self.dt
        h = np.maximum(y - self.yc, 0.0)
        # they turn above the calm stretch, ever slower as they age, and are gone before the top
        env = (_smooth(self.calm * 0.8, self.calm + 0.12 * self.reach, h) * np.exp(-age / (0.55 * self.life)) *
               (1 - _smooth(0.7 * self.reach, 1.0 * self.reach, h)) * _smooth(0.0, 0.3, age))
        wo = self.spin[c] * self.wsign[c, b[a]] * env
        keep = np.abs(wo) > 0.05
        if not keep.any():
            return None
        r = self.rc[c] * (1 + 0.8 * np.minimum(age / self.life, 1.0))
        return (x[keep].astype(np.float32), y[keep].astype(np.float32), wo[keep].astype(np.float32),
                (r[keep] ** 2).astype(np.float32))

    def _step(self, state, k, rows=None):
        """The smoke (and the whirl points) at tick k + 1 from tick k (rows: age in ticks; columns: points). Only the
        first `rows` rows of smoke are carried on (the others are not smoke yet when starting from scratch; the whirl
        points are always all there: _state())."""
        x, y, wx, wy = state
        r = self.L - 1 if rows is None else rows
        age = (np.arange(r, dtype=np.float32) * self.dt)[:, None]
        whirls = self._whirls(wx, wy, k, self.L)
        plumes = self._plumes(k)
        xs, ys = x[:r], y[:r]
        vx, vy = self._velocity(xs, ys, age, k, whirls, plumes, self.lift)
        hx, hy = xs + 0.5 * self.dt * vx, ys + 0.5 * self.dt * vy
        vx, vy = self._velocity(hx, hy, age + 0.5 * self.dt, k, whirls, plumes, self.lift)
        nx, ny = np.empty_like(x), np.empty_like(y)
        nx[1:r + 1], ny[1:r + 1] = xs + self.dt * vx, ys + self.dt * vy
        if r + 1 < self.L:
            nx[r + 1:], ny[r + 1:] = x[r + 1:], y[r + 1:]
        nx[0], ny[0] = self._source(k + 1)
        return (nx, ny) + self._step_whirls(wx, wy, k)

    def _step_whirls(self, wx, wy, k, rows=None):
        """The whirl points at tick k + 1 from tick k: carried by the air (no whirls, nothing gathering them), a
        little slower than the smoke (only the first `rows` rows: as _step())."""
        if not self.nwh:
            return wx, wy
        r = self.L - 1 if rows is None else rows
        age = (np.arange(r, dtype=np.float32) * self.dt)[:, None]
        ax, ay = self._velocity(wx[:r], wy[:r], age, k, None, None)
        nwx, nwy = np.empty_like(wx), np.empty_like(wy)
        nwx[1:r + 1], nwy[1:r + 1] = wx[:r] + self.dt * ax, wy[:r] + self.dt * 0.8 * ay
        if r + 1 < self.L:
            nwx[r + 1:], nwy[r + 1:] = wx[r + 1:], wy[r + 1:]
        nwx[0], nwy[0] = self._whirl_source(k + 1)
        return nwx, nwy

    def _source(self, k):
        """Where the points let go of smoke at tick k."""
        ph = self._phase(self.wm, k) + self.wphi
        wander = self.wan * (0.6 * np.sin(ph[0]) + 0.4 * np.sin(ph[1]))
        ux = np.clip(self.ux + wander, -1.0, 1.0)
        return (ux * self.a).astype(np.float32), self.sy

    def _whirl_source(self, k):
        """Where the columns of whirl points start at tick k: in the wisps, on either side of them."""
        if not self.nwh:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)
        plumes = self._plumes(k)
        if plumes is None:  # (a point: on either side of the column)
            x = self.wside * (0.012 + 0.2 * self.a)
        else:
            c, wdt, _ = plumes
            j = np.arange(self.nwh) // 2 % self.nw
            x = c[j] + self.wside * (0.7 * wdt[j] + 0.01)
        return x.astype(np.float32), np.full(self.nwh, self.yc + 0.3 * self.e, np.float32)

    def _state(self, k):
        """The smoke at tick k: x, y of every bit of it (rows: age in ticks, columns: points), and the whirl points."""
        k %= self.ticks
        if k in self.states:
            return self.states[k]
        back = [(k - j) % self.ticks for j in self.states]
        near = min((b for b in back if 0 < b <= 8), default=0)  # a tick a little earlier: carried on from it
        if near:
            for j in range(k - near, k - 1):
                self._state(j + 1)
            prev = (k - 1) % self.ticks
            state = self._step(self.states[prev], prev)
        else:  # from scratch, a whole lifetime back: whatever was there before has gone by tick k. The whirls of
            # that time first (a lifetime before it: they do not depend on the smoke), so the smoke meets the same
            # whirls as carried on
            L = self.L
            wx, wy = (np.repeat(v[None], L, 0) for v in self._whirl_source(k))
            for i in range(L if self.nwh else 0):
                wx, wy = self._step_whirls(wx, wy, (k - 2 * L + i) % self.ticks, min(i, L - 1))
            sx, sy = self._source(k)
            state = (np.repeat(sx[None], L, 0), np.repeat(sy[None], L, 0), wx, wy)
            for i in range(L):
                state = self._step(state, (k - L + i) % self.ticks, min(i, L - 1))
        if len(self.states) > 3:
            self.states.pop(next(iter(self.states)))
        self.states[k] = state
        return state

    def _emission(self, j):
        """How much smoke each point lets go of at ticks j (one row each): more under the places that let go of
        more, in bands that sweep over the surface, a flicker of each point (never all gone). Kept for the ticks
        asked for lately (a frame after the next asks for one tick more)."""
        j = np.asarray(j) % self.ticks
        missing = [int(v) for v in j if int(v) not in self.emitted]
        if missing:
            for v, row in zip(missing, self._emit(np.array(missing))):
                self.emitted[v] = row
            if len(self.emitted) > 4 * self.L:
                keep = set(int(v) for v in j)
                for v in [v for v in self.emitted if v not in keep]:
                    del self.emitted[v]
        return np.stack([self.emitted[int(v)] for v in j])

    def _emit(self, j):
        j = np.asarray(j)[:, None]
        x = self._source_x(j)
        out = np.full(x.shape, self.veil, np.float32)
        if self.nw:
            c, e = self._plume_state(j[:, 0])
            for i in range(self.nw):
                out += (1.2 * e[:, i:i + 1]) * np.exp(-0.5 * ((x / self.a - c[:, i:i + 1]) / self.pw[i]) ** 2)
        bands = sum(c * np.sin(self._phase(self.bm[i], j) + self.bk[i] * x / max(self.a, 1e-6) + self.bphi[i])
                    for i, c in enumerate((0.45, 0.3)))
        flicker = 0.5 + 0.5 * (bands + 0.25 * np.sin(self._phase(self.fm, j) + self.fphi))
        floor = 0.8 if self.kind == "incense" else 0.45  # (an incense stick burns evenly)
        flicker = floor + (1 - floor) * np.clip(flicker, 0.0, 1.0)
        edge = np.sqrt(np.clip(1 - (x / max(self.a, 1e-6)) ** 2, 0.05, 1)) if self.kind != "incense" else 1.0
        return (out * flicker * self.bright * edge).astype(np.float32)

    def _source_x(self, j):
        """Where the points are at ticks j (one row each), box heights across."""
        ph = self._phase(self.wm[:, None, :], j[None]) + self.wphi[:, None, :]
        wander = self.wan * (0.6 * np.sin(ph[0]) + 0.4 * np.sin(ph[1]))
        return (np.clip(self.ux + wander, -1.0, 1.0) * self.a).astype(np.float32)

    # --- colour ------------------------------------------------------------------------------------------------
    def _colours(self):
        """Its colour where its wisps face away from the light and where they face it (sRGB 0..1; the same unlit)."""
        own = self.colour
        if self.tone is None:
            return own, own
        lit = self.p["lit"] / 100
        t = self.tone
        key, back, side = t[:3], t[3:6], float(np.clip(t[6], -1, 1))
        own_lin = linear(own)
        tint = own_lin / max(float(own_lin @ LUMA), 1e-4)  # (its hue: the light gives its brightness)
        tint = 1 + (tint - 1) * 0.8
        shaded = 0.45 * min(1.0, abs(side) * 2.5)  # (a light from one side: the far side of a wisp is darker)
        out = []
        for face in (0.0, 1.0):
            c = (ALBEDO * key * (1 - shaded * (1 - face)) + FORWARD * back) * tint
            c = (1 - lit) * own_lin + lit * np.minimum(c, 1.0)
            out.append(srgb(c))
        self.side = float(np.sign(side)) if shaded > 0.02 else 0.0
        return tuple(out)

    # --- drawing -----------------------------------------------------------------------------------------------
    def draw(self, t, canvas, mask):
        if self.opacity <= 0 or self.p["amount"] <= 0:
            return
        x0, y0, w, h = self.box()
        bx, by = int(round(x0)), int(round(y0))
        bw, bh = int(round(x0 + w)) - bx, int(round(y0 + h)) - by
        if bw < 2 or bh < 2 or bx >= self.w or by >= self.h or bx + bw <= 0 or by + bh <= 0:
            return
        got = self.cover(t, h)
        if got is None:
            return
        cover, shade, gr, gc, (gh, gw) = got
        # the part of the box in the frame, at the box's size
        c0, c1, r0, r1 = max(0, -bx), min(bw, self.w - bx), max(0, -by), min(bh, self.h - by)
        if (gh, gw) == (bh, bw):  # (the grid is the box: only the part that holds smoke)
            c0, c1 = max(c0, gc), min(c1, gc + cover.shape[1])
            r0, r1 = max(r0, gr), min(r1, gr + cover.shape[0])
            if c0 >= c1 or r0 >= r1:
                return
            cut = (slice(r0 - gr, r1 - gr), slice(c0 - gc, c1 - gc))
            cover = cover[cut]
            shade = None if shade is None else shade[cut]
        else:  # a big box: the grid enlarged to it
            kx, ky = gw / bw, gh / bh

            def fit(a):
                if a is None:
                    return None
                full = np.zeros((gh, gw), np.float32)
                full[gr:gr + a.shape[0], gc:gc + a.shape[1]] = a
                return np.asarray(Image.fromarray(full, "F").resize(
                    (c1 - c0, r1 - r0), Image.Resampling.BILINEAR, box=(c0 * kx, r0 * ky, c1 * kx, r1 * ky)))
            cover, shade = fit(cover), fit(shade)
        bx, by = bx + c0, by + r0
        rows = np.flatnonzero(cover.max(axis=1) > 1 / 2048)
        if rows.size == 0:
            return
        cols = np.flatnonzero(cover.max(axis=0) > 1 / 2048)
        r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
        cover = cover[r0:r1, c0:c1]
        lo, hi = self.colours
        if shade is None:
            canvas.over(bx + c0, by + r0, cover, lo, 1.0, mask)
            return
        shade = shade[r0:r1, c0:c1]
        prgb = np.empty(cover.shape + (3,), np.float32)
        for c in range(3):
            np.multiply(shade, hi[c] - lo[c], out=prgb[..., c])
            prgb[..., c] += lo[c]
            prgb[..., c] *= cover
        canvas.over_rgba(bx + c0, by + r0, prgb, cover, mask)

    def cover(self, t, h):
        """How much the smoke covers, 0..1, on a grid over the box (rows from the top) for a box h pixels high: the
        part of it that holds smoke, how much each place there faces the light (0..1; None when it is all one
        colour), where that part is on the grid (row, column) and the grid's size; None if there is no smoke."""
        gh = int(min(self.style.get("grid", GRID_MAX), max(16, round(h))))
        gw = max(4, int(round(gh * self.bw)))
        if float(t).is_integer():  # (exactly, so frame P is frame 0)
            k, f = divmod(int(t) * self.ticks, self.P)
            f /= self.P
        else:
            k = math.floor(t * self.ticks / self.P)
            f = t * self.ticks / self.P - k
        xa, ya = self._state(k)[:2]
        xb, yb = self._state(k + 1)[:2]
        # the bit of smoke let go at tick k - a: row a at tick k, row a + 1 at tick k + 1
        x = (1 - f) * xa[:-1] + f * xb[1:]
        y = (1 - f) * ya[:-1] + f * yb[1:]
        age = ((np.arange(self.L - 1) + f) * self.dt)[:, None].astype(np.float32)
        mass = self._emission(k - np.arange(self.L - 1)) * self.gain
        reach, life = self.reach, self.life
        h_ = y - self.yc
        mass *= _smooth(0.0, self.style["show"], age)  # (steam shows as it cools, a little above the surface)
        # it evaporates: as it ages, and before its reach
        mass *= np.exp(-age * self.evap / (0.45 * life)) * (1 - _smooth(self.style["fade"] * reach, reach, h_))
        mass *= 1 - _smooth(life * 0.75, life, age)
        if self.lean:  # (blown sideways it evaporates before the box's side)
            lo = max(0.28, self.a + 0.04)  # (past the rim, on the side it is blown to)
            mass *= 1 - _smooth(lo, max(0.46, lo + 0.06), x * math.copysign(1.0, self.lean))
        sig = (self.s0 * self.fuzz + self.grow * np.sqrt(np.maximum(age - 0.05, 0.0)) +
               self.style["widen"] * np.maximum(age - 1.0, 0.0))
        got = self._splat(x, y, mass, sig, gh, gw)
        if got is None:
            return None
        dens, gr, gc = got
        dens *= DENSITY * self.glow
        # fade near the box's sides and top: nothing is cut off at its edges
        hh, ww = dens.shape
        cols = (np.arange(gc, gc + ww, dtype=np.float32) + 0.5) / gw
        rows = (np.arange(gr, gr + hh, dtype=np.float32) + 0.5) / gh
        dens *= _smooth(0.0, 0.1, rows)[:, None] * (_smooth(0.0, 0.1, cols) * _smooth(0.0, 0.1, 1 - cols))[None, :]
        shade = None
        if self.tone is not None and self.side:  # which way each place faces: up its slope, toward the light or not
            g = np.zeros_like(dens)
            np.subtract(dens[:, 2:], dens[:, :-2], out=g[:, 1:-1])
            shade = g * -self.side  # (the edge of a wisp rising toward the light is lit: its near side)
            np.abs(g, out=g)
            g += 0.6 * dens
            g += 1e-6
            shade /= g
            shade *= 0.5
            shade += 0.5
        cover = np.negative(dens, out=dens)
        np.exp(cover, out=cover)
        cover *= -self.opacity
        cover += self.opacity
        return cover, shade, gr, gc, (gh, gw)

    def _splat(self, x, y, mass, sig, gh, gw):
        """The smoke's density over a gh x gw grid (in box heights squared): each bit as a curve on to the next bit
        from the same point (Catmull-Rom), in steps shorter than its blur, laid on the layer of its blur (two,
        between them); the layers blurred, the wide ones on coarser grids (a pyramid), and summed. Only the part of
        the grid the smoke reaches is worked out: (density there, its first row, its first column); None if none."""
        gx = ((x + self.bw / 2) * gh - 0.5).ravel()
        gy = ((1 - y) * gh - 0.5).ravel()
        rows, n = x.shape
        m = 0.5 * (mass[:-1] + mass[1:])
        top = gy[:-n].reshape(rows - 1, n)
        idx = np.flatnonzero(((m > 1e-5) & (top > -gh * 0.2) & (top < gh * 1.2)).ravel())
        if idx.size == 0:
            return None
        a, col = idx // n, idx % n
        p0, p1, p2, p3 = (np.clip(a + d, 0, rows - 1) * n + col for d in (-1, 0, 1, 2))
        coef = []
        for g in (gx, gy):  # the curve through p1 and p2: ((c3 u + c2) u + c1) u + p1
            g0, g1, g2, g3 = g[p0], g[p1], g[p2], g[p3]
            coef.append((-0.5 * g0 + 1.5 * g1 - 1.5 * g2 + 0.5 * g3, g0 - 2.5 * g1 + 2 * g2 - 0.5 * g3,
                         0.5 * (g2 - g0), g1))
        s = (0.5 * (sig[:-1] + sig[1:])).ravel()[idx] * gh  # blur, pixels of the grid
        # the layers: band b blurred s0 2^b, drawn on a grid 2^level coarser (its blur there 1..2 of its pixels)
        s0 = max(self.s0 * gh, 0.35)
        nb = int(min(BANDS, max(2, math.ceil(math.log2(max(float(s.max()) / s0, 1.0))) + 1)))
        sb = s0 * 2.0 ** np.arange(nb)
        level = np.floor(np.log2(np.maximum(sb, 1.0))).astype(np.int64)
        band = np.clip(np.log2(np.maximum(s, 1e-6) / s0), 0, nb - 1)
        b0 = np.minimum(band.astype(np.int64), nb - 2)
        frac = (band - b0).astype(np.float32)
        step = np.clip(0.7 * s, 0.7, 3.0 * 2.0 ** level[b0])  # (a step need not be shorter than the blur it gets)
        seg = np.hypot(gx[p2] - gx[p1], gy[p2] - gy[p1])
        cnt = np.clip(np.ceil(seg / step), 1, 64).astype(np.int64)
        rep = np.repeat(np.arange(idx.size), cnt)
        u = ((np.arange(rep.size) - np.repeat(np.cumsum(cnt) - cnt, cnt) + 0.5) / cnt[rep]).astype(np.float32)
        px, py = ((((c3[rep] * u + c2[rep]) * u + c1[rep]) * u + c0[rep]) for c3, c2, c1, c0 in coef)
        # the part of the grid it reaches (its blur round it), on whole pixels of the coarsest layer (past the grid's
        # edge too: the blur there spills back, as across the grid)
        q = 2 ** int(level[-1])
        pad = 3.0 * float(s.max()) + 2 * q + 2
        r0 = max(0, math.floor((float(py.min()) - pad) / q) * q)
        c0 = max(0, math.floor((float(px.min()) - pad) / q) * q)
        r1 = min(math.ceil(gh / q) * q, math.ceil((float(py.max()) + pad) / q) * q)
        c1 = min(math.ceil(gw / q) * q, math.ceil((float(px.max()) + pad) / q) * q)
        if r0 >= min(r1, gh) or c0 >= min(c1, gw):
            return None
        hh, ww = r1 - r0, c1 - c0
        px -= c0
        py -= r0
        wgt = (m.ravel()[idx] / cnt).astype(np.float32)
        w1 = (wgt * frac)[rep]
        w0 = wgt[rep] - w1
        bb = b0[rep]
        # each layer's grid (a pixel larger all round: what falls on its border is dropped)
        sh_, sw_ = hh >> level, ww >> level
        start = np.concatenate([[0], np.cumsum((sh_ + 2) * (sw_ + 2))])
        index, weight = [], []
        for d, wb in ((0, w0), (1, w1)):
            b = bb + d
            k = (0.5 ** level[b]).astype(np.float32)
            qx, qy = (px + 0.5) * k - 0.5, (py + 0.5) * k - 0.5
            ix, iy = np.floor(qx).astype(np.int64), np.floor(qy).astype(np.int64)
            fx, fy = qx - ix, qy - iy
            sh, sw = sh_[b], sw_[b]
            out = (ix < -1) | (ix >= sw) | (iy < -1) | (iy >= sh)
            wb = np.where(out, 0.0, wb * k * k).astype(np.float32)  # (density: a coarse pixel is 4^level fine ones)
            ix, iy = np.clip(ix, -1, sw - 1), np.clip(iy, -1, sh - 1)
            W = sw + 2
            flat = start[b] + (iy + 1) * W + ix + 1
            for o, c in ((0, (1 - fx) * (1 - fy)), (1, fx * (1 - fy)), (W, (1 - fx) * fy), (W + 1, fx * fy)):
                index.append(flat + o)
                weight.append(wb * c)
        acc = np.bincount(np.concatenate(index), np.concatenate(weight), int(start[-1])).astype(np.float32)
        layers = [acc[start[b]:start[b + 1]].reshape(sh_[b] + 2, sw_[b] + 2)[1:-1, 1:-1] for b in range(nb)]
        # from the most blurred: blurred as much as it needs on top of the next one's blur, made finer, the next added.
        # Each band's blur (variance, fine pixels squared) is sb^2 in all: its own blurs, the blurs of the finer
        # bands after it, and what laying it on its grid (1/6 of a pixel of that grid, squared) and making that grid
        # finer (as much) add
        lay = 4.0 ** level / 6.0
        out = layers[nb - 1]
        for b in range(nb - 1, 0, -1):
            finer = lay[b] if level[b - 1] < level[b] else 0.0
            more = sb[b] ** 2 - sb[b - 1] ** 2 - lay[b] + lay[b - 1] - finer
            out = _blur_small(out, math.sqrt(max(more, 0.0)) / 2 ** level[b])
            if finer:
                out = _double(out, layers[b - 1].shape)
            out += layers[b - 1]
        # the last blur, on the first band's grid (made finer after it, to the grid's own size: that adds as much)
        finer = sum(4.0 ** lv / 6.0 for lv in range(1, int(level[0]) + 1))
        out = _blur_small(out, math.sqrt(max(sb[0] ** 2 - lay[0] - finer, 0.0)) / 2 ** level[0])
        for lv in range(int(level[0]) - 1, -1, -1):
            out = _double(out, (hh >> lv, ww >> lv))
        out *= float(gh * gh)
        return out[:gh - r0, :gw - c0], r0, c0
