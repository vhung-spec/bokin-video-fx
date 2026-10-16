"""The effects: what they share (the canvas they draw on, falling particles), glowing particles (fireflies, dust in
the light), and the list of every effect (snow, rain, leaves... each in its own module).

Every particle moves as a function of the phase tau = t / P, where t is the frame number and P the number of
frames of the loop, and every frequency in it is a whole number of cycles per loop. So frame P is exactly frame 0
again: the effect loops with the video, with no crossfade.

Falling particles travel down a slanted line (the wind) and start again at the top after leaving the bottom, a
whole number of times k per loop. Their speed stays free: the path can be longer than the frame, the rest of it
spent out of sight above the top, so k passes of a path of length L fill exactly the loop's duration.

Lengths are in frame heights and pixel sizes are given for a 1080-line frame, so a small preview shows the same
scene as the full video. Random draws are fixed per particle (by the layer's seed), so moving a slider changes
what it should and leaves every particle where it was.
"""
import math

import numpy as np

from . import overlay, sprites


# A parameter may also have a "group": the page shows a heading where the group changes (e.g. "Hạt", "Gió").
def _num(key, label, value, lo=0, hi=100, **extra):
    return {"key": key, "label": label, "type": "range", "min": lo, "max": hi, "value": value, **extra}


def _choice(key, label, value, options, presets=None, **extra):
    return {"key": key, "label": label, "type": "choice", "value": value,
            "options": [{"value": v, "label": t} for v, t in options], "presets": presets or {}, **extra}


def _colour(key, label, value, **extra):
    return {"key": key, "label": label, "type": "color", "value": value, **extra}


def _toggle(key, label, value=False, **extra):
    return {"key": key, "label": label, "type": "toggle", "value": value, **extra}


def _place(scale=20, x=0, y=0, group="Vị trí và cỡ"):
    """The settings that place a placed effect (Effect.BOX): as a file clip's, so the box on the page's picture
    moves and sizes it."""
    return [_num("scale", "Cỡ (%)", scale, 1, 300, step=0.1, group=group),
            _num("x", "Dịch ngang (%)", x, -100, 100, step=0.05, group=group),
            _num("y", "Dịch dọc (%)", y, -100, 100, step=0.05, group=group)]


SCHEMA = {
    "glow": {"label": "Hạt sáng", "params": [
        _choice("style", "Kiểu", "firefly", [("firefly", "Đom đóm"), ("dust", "Bụi sáng lơ lửng")],
                presets={"firefly": {"color": "#f2ff8c"}, "dust": {"color": "#fff1d6"}}),
        _num("amount", "Số hạt", 35), _num("size", "Cỡ", 40), _num("brightness", "Độ sáng", 70),
        _num("twinkle", "Nhấp nháy", 60), _num("drift", "Bay lượn", 50),
        _num("depth", "Chiều sâu (xa nhỏ, gần to)", 50), _colour("color", "Màu", "#f2ff8c")]},
}


SCHEMA["file"] = overlay.SCHEMA  # a layer from an outside file (video or picture)
# (snow, rain, leaves, water on glass, steam, candle: their own modules, added at the end)


def defaults(kind):
    return {p["key"]: p["value"] for p in SCHEMA[kind]["params"]}


def clean_params(kind, raw):
    """The parameters of a `kind` layer from the page: every key present, numbers clamped, choices checked."""
    out = {}
    raw = raw if isinstance(raw, dict) else {}
    upgrade = getattr(KINDS.get(kind), "upgrade", None)
    if upgrade:  # (settings saved by an older version of the effect)
        raw = upgrade(raw)
    for p in SCHEMA[kind]["params"]:
        v = raw.get(p["key"], p["value"])
        if p["type"] == "range":
            step = p.get("step", 1)
            try:
                v = round(round(float(v) / step) * step, 3)
            except (TypeError, ValueError, OverflowError):
                v = p["value"]
            if v != v:  # NaN
                v = p["value"]
            v = max(p["min"], min(p["max"], v))
            if step == 1:
                v = int(v)
        elif p["type"] == "choice":
            v = v if v in {o["value"] for o in p["options"]} else p["value"]
        elif p["type"] == "toggle":
            v = v is True or v in (1, "1", "true", "on")
        else:
            v = v.lower() if isinstance(v, str) and len(v) == 7 and v[0] == "#" and all(
                c in "0123456789abcdefABCDEF" for c in v[1:]) else p["value"]
        out[p["key"]] = v
    return out


def _rgb(hex_colour):
    return np.array([int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)], np.float32) / 255.0


def _lerp(a, b, t):
    return a + (b - a) * t


class Canvas:
    """One frame's particles before they are laid over the picture: premultiplied colour and coverage for
    'over' layers, or light for 'screen' layers, with the box they touched."""

    def __init__(self, w, h, block=None):
        """`block`: memory to draw in (float32, at least w * h * 4 of them, all 0; every user leaves it all 0 again,
        as reset() does), shared with other canvases used one after another (sheet.Scratch); None: its own."""
        self.w, self.h = w, h
        if block is None:
            self.rgb = np.zeros((h, w, 3), np.float32)
            self.a = np.zeros((h, w), np.float32)
        else:
            self.rgb = block[:h * w * 3].reshape(h, w, 3)
            self.a = block[h * w * 3:h * w * 4].reshape(h, w)
        self.box = None

    def reset(self):
        if self.box:
            x0, y0, x1, y1 = self.box
            self.rgb[y0:y1, x0:x1] = 0
            self.a[y0:y1, x0:x1] = 0
        self.box = None

    def scale(self, gain):
        """What was drawn, `gain` (0..1) times as strong: a clip fading in or out."""
        if self.box:
            x0, y0, x1, y1 = self.box
            self.rgb[y0:y1, x0:x1] *= gain
            self.a[y0:y1, x0:x1] *= gain

    def _place(self, x0, y0, h, w):
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + w, self.w), min(y0 + h, self.h)
        if cx0 >= cx1 or cy0 >= cy1:
            return None
        if self.box is None:
            self.box = [cx0, cy0, cx1, cy1]
        else:
            b = self.box
            b[0], b[1], b[2], b[3] = min(b[0], cx0), min(b[1], cy0), max(b[2], cx1), max(b[3], cy1)
        return (slice(cy0, cy1), slice(cx0, cx1)), (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))

    def over(self, x0, y0, cover, colour, opacity, mask):
        """A one-colour particle (coverage `cover` 0..1) laid over what is already there."""
        at = self._place(x0, y0, *cover.shape)
        if at is None:
            return
        dst, src = at
        a = cover[src] * opacity
        if mask is not None:
            a = a * mask[dst]
        rgb, cov = self.rgb[dst], self.a[dst]
        inv = 1.0 - a
        rgb *= inv[..., None]
        rgb += a[..., None] * colour
        cov *= inv
        cov += a

    def over_rgba(self, x0, y0, prgb, alpha, mask):
        """A many-coloured particle (premultiplied colour, coverage) laid over what is already there."""
        at = self._place(x0, y0, *alpha.shape)
        if at is None:
            return
        dst, src = at
        a, c = alpha[src], prgb[src]
        if mask is not None:
            m = mask[dst]
            a, c = a * m, c * m[..., None]
        rgb, cov = self.rgb[dst], self.a[dst]
        inv = 1.0 - a
        rgb *= inv[..., None]
        rgb += c
        cov *= inv
        cov += a

    def light(self, x0, y0, glow, colour, mask):
        at = self._place(x0, y0, *glow.shape)
        if at is None:
            return
        dst, src = at
        g = glow[src] if mask is None else glow[src] * mask[dst]
        self.rgb[dst] += g[..., None] * colour

    def flush(self, picture, blend):
        """Lay what was drawn over `picture` (float32 0..1, changed in place) and start empty again."""
        if self.box is None:
            return
        x0, y0, x1, y1 = self.box
        out = picture[y0:y1, x0:x1]
        if blend == "screen":
            out[:] = 1.0 - (1.0 - out) * (1.0 - np.minimum(self.rgb[y0:y1, x0:x1], 1.0))
        else:
            out *= 1.0 - self.a[y0:y1, x0:x1, None]
            out += self.rgb[y0:y1, x0:x1]
        self.reset()


class Effect:
    blend = "over"
    CAP = 1000  # particles with fixed random draws; how many are used follows the "amount" slider
    VERSION = 1  # raised when the picture an effect draws changes (its preview clips are made again)
    # lens: the effect also changes the picture under it (blur, refraction), in apply(); draw() is then only what
    # it lays over the picture, which is all the page's live preview shows. A lens that turns (TURNS) makes each
    # pixel kept * itself + what it adds (both 0..1), and says how much it keeps: apply(t, picture, mask, gain,
    # kept) fills `kept` (sheet.Sheet lays a turned lens on the frame with it)
    lens = False
    # placed: an effect at one place (a candle flame, steam over a cup), drawn inside its box (box()). BOX is the
    # box's shape (w, h); the settings x, y, scale (_place) place it as a file clip with fit "contain" is placed.
    # Inside the box every size is relative to the box, so the preview clip is the box alone (rendered with x, y 0
    # and scale 100 on a canvas of the box's shape) and the page lays it where the box is.
    BOX = None
    # under: a placed lens effect whose preview clip is made from the picture under its box (the video and its
    # still area there, with the effect applied), laid with Normal: the live preview then shows what it does
    UNDER = False
    # turns: placed and turned in 3D as a file clip is (sheet.Sheet): a full-frame effect on a sheet of the frame's
    # shape, a placed one in its box. A candle's flame does not turn (it always rises straight up).
    TURNS = True
    PREVIEW = None  # a full-frame effect's sharper preview clip: as many pixels as a 16:9 one this wide (None:
    # engine.FX_CLIP_WIDTH wide)

    def __init__(self, params, seed, frames, fps, w, h, aspect=None):
        self.p, self.seed = params, int(seed)
        self.P, self.fps, self.w, self.h = int(frames), float(fps), int(w), int(h)
        # (a sheet drawn at a size rounded to whole pixels gets its true shape: how many particles there are follows
        # it, and a count one off would move many of them; sheet.Sheet)
        self.aspect = float(aspect) if aspect else self.w / self.h
        self.loop_s = self.P / self.fps
        self.px = self.h / 1080.0  # sizes below are given in pixels of a 1080-line frame
        self.rng = np.random.default_rng([self.seed, 1103])

    def box(self):
        """(x0, y0, w, h): where a placed effect's box lies in the frame, in pixels (floats)."""
        return place_box(self.BOX, self.p, self.w, self.h)

    def cycles(self, rate_hz):
        """The whole number of cycles per loop nearest to `rate_hz` cycles a second (at least 1)."""
        return np.maximum(1.0, np.rint(np.asarray(rate_hz) * self.loop_s))

    def tau(self, t):
        return float(t) / self.P

    def draw(self, t, canvas, mask):
        raise NotImplementedError


class Falling(Effect):
    """Particles falling along slanted straight paths, k passes per loop, plus side sway and gusts."""

    def paths(self, ux, us, speed, angle, margin_y, margin_x):
        """Sets the paths of the candidate particles; returns each one's share of the loop spent on screen."""
        cos, sin = np.cos(angle), np.sin(angle)
        lmin = (1 + 2 * margin_y) / cos
        travel = speed * self.loop_s  # frame heights per loop at the wished speed
        ratio = travel / lmin
        k = np.where(ratio < 1, 1.0, np.floor(ratio))
        length = np.where(ratio < 1, lmin, travel / k)
        drift = (1 + 2 * margin_y) * sin / cos
        lo = np.minimum(0.0, -drift) - margin_x
        hi = np.maximum(self.aspect, self.aspect - drift) + margin_x
        xe = lo + ux * (hi - lo)
        flat = np.abs(sin) < 1e-6
        safe = np.where(flat, 1.0, sin)
        da, db = (-margin_x - xe) / safe, (self.aspect + margin_x - xe) / safe
        seen = np.clip(np.maximum(da, db), 0, lmin) - np.clip(np.minimum(da, db), 0, lmin)
        seen = np.where(flat, np.where((xe >= -margin_x) & (xe <= self.aspect + margin_x), lmin, 0.0), seen)
        self.cos, self.sin, self.k, self.len, self.my = cos, sin, k, length, margin_y
        self.extra, self.s0, self.xe = length - lmin, us * length, xe
        self.speed = k * length / self.loop_s  # the actual speed, frame heights a second
        return np.maximum(seen, 0) / length

    def choose(self, share, wanted, arrays):
        """How many of the candidates make about `wanted` particles on screen at a time; keeps those,
        sorted far to near (the drawing order)."""
        n = int(np.searchsorted(np.cumsum(share), wanted)) + 1 if wanted > 0 else 0
        n = min(n, len(share))
        keep = np.argsort(arrays["z"][:n], kind="stable")
        for name in ("cos", "sin", "k", "len", "extra", "s0", "xe", "speed"):
            setattr(self, name, getattr(self, name)[:n][keep])
        return {name: a[:n][keep] for name, a in arrays.items()}

    def gusts(self, strength):
        j = max(1, round(self.loop_s / 6))
        phases = self.rng.random(3) * 2 * math.pi
        self.gust = (strength, [(j * (i + 1), (1.0, 0.5, 0.25)[i] / 1.75, phases[i]) for i in range(3)])

    def gust_at(self, tau):
        strength, parts = self.gust
        return strength * sum(c * math.sin(2 * math.pi * m * tau + ph) for m, c, ph in parts)

    def where(self, t):
        """x, y (frame heights) of every particle at frame t, and whether it is on its visible stretch."""
        tau = self.tau(t)
        s = np.mod(self.s0 + self.k * self.len * tau, self.len)
        d = s - self.extra
        return self.xe + d * self.sin, -self.my + d * self.cos, d >= 0


class Glow(Effect):
    """Points of light that wander and twinkle in place: fireflies, or dust catching the light."""
    blend = "screen"
    CAP = 600

    def __init__(self, *a):
        super().__init__(*a)
        p, px = self.p, self.px
        firefly = p["style"] == "firefly"
        u = self.rng.random((self.CAP, 16))
        n = int(round(p["amount"] / 100 * (110 if firefly else 450) * self.aspect / (16 / 9)))
        u = u[:n]
        z = u[:, 0]
        ds = p["depth"] / 100
        size_k = (1 - ds) + ds * (0.4 + 1.6 * z ** 1.5)
        base = _lerp(1.6, 5.0, p["size"] / 100) if firefly else _lerp(0.5, 2.2, p["size"] / 100)
        self.radius = base * size_k * (0.8 + 0.4 * u[:, 1]) * px
        self.bright = p["brightness"] / 100 * _lerp(0.6, 1.0, z) * (1.8 if firefly else 0.9) / np.sqrt(size_k)
        self.cx = -0.05 + u[:, 2] * (self.aspect + 0.1)
        self.cy = -0.05 + u[:, 3] * 1.1
        reach = p["drift"] / 100 * (_lerp(0.01, 0.06, u[:, 4]) if firefly else _lerp(0.005, 0.03, u[:, 4]))
        self.ax = np.stack([reach * 0.7, reach * 0.3])
        self.ay = np.stack([reach * 0.5, reach * 0.3])
        self.mx = np.stack([self.cycles(_lerp(0.05, 0.15, u[:, 5])), self.cycles(_lerp(0.15, 0.4, u[:, 6]))])
        self.my = np.stack([self.cycles(_lerp(0.05, 0.15, u[:, 7])), self.cycles(_lerp(0.15, 0.4, u[:, 8]))])
        self.px_ = u[:, 9:11].T * 2 * math.pi
        self.py_ = u[:, 11:13].T * 2 * math.pi
        self.q = self.cycles(_lerp(0.12, 0.35, u[:, 13]) if firefly else _lerp(0.15, 0.6, u[:, 13]))
        self.qphi = u[:, 14] * 2 * math.pi
        self.firefly, self.twinkle = firefly, p["twinkle"] / 100
        self.sharp = _lerp(0.6, 3.0, self.twinkle)
        self.colour = _rgb(p["color"])
        self.keys = [sprites.bucket(r) for r in self.radius]
        self.cache = sprites.SpriteCache(lambda key, fx, fy: sprites.glow(key, fx, fy))

    def draw(self, t, canvas, mask):
        tau = self.tau(t)
        x = self.cx + (self.ax * np.sin(2 * math.pi * self.mx * tau + self.px_)).sum(axis=0)
        y = self.cy + (self.ay * np.sin(2 * math.pi * self.my * tau + self.py_)).sum(axis=0)
        wave = np.sin(2 * math.pi * self.q * tau + self.qphi)
        if self.firefly:
            pulse = np.maximum(wave, 0.0) ** self.sharp
            level = (1 - self.twinkle) * 0.8 + self.twinkle * pulse
        else:
            level = (1 - 0.8 * self.twinkle) + 0.8 * self.twinkle * ((0.5 + 0.5 * wave) ** 2)
        level = level * self.bright
        xs, ys = x * self.h, y * self.h
        for i in np.flatnonzero(level > 0.004):  # halos reach far: Canvas clips whatever falls outside
            ix, iy = math.floor(xs[i]), math.floor(ys[i])
            spot, x0, y0 = self.cache.get(self.keys[i], xs[i] - ix, ys[i] - iy)
            canvas.light(ix + x0, iy + y0, spot, self.colour * level[i], mask)


from . import candle, glass, leaves, rain, snow, steam  # (here: they build on the classes above)

SCHEMA = {"snow": snow.SCHEMA, "rain": rain.SCHEMA, "glass": glass.SCHEMA, "steam": steam.SCHEMA,
          "candle": candle.SCHEMA, "leaves": leaves.SCHEMA, **SCHEMA}
KINDS = {"snow": snow.Snow, "leaves": leaves.Leaves, "rain": rain.Rain, "glow": Glow, "glass": glass.Glass,
         "steam": steam.Steam, "candle": candle.Candle}

# Placed and turned like a file clip (Effect.TURNS): the same settings as a file clip's, under the same headings,
# after the effect's own. A full-frame effect gets a place and a size (at 100 and in the middle: the frame itself, as
# before); every effect that turns gets the turn in 3D (a placed effect's place and size are its box's: _place()).
SHEET_KEYS = ("x", "y", "scale") + overlay.ROTATIONS + ("persp",)  # its place and turn: not in its preview clip
_FILE = {p["key"]: p for p in overlay.SCHEMA["params"]}
_PLACE = [dict(_FILE[k], group="Vị trí và cỡ") for k in ("scale", "x", "y")]
_TURN = [dict(_FILE[k], group="Xoay 3D") for k in overlay.ROTATIONS + ("persp",)]
for _kind, _cls in KINDS.items():
    if _cls.TURNS:
        SCHEMA[_kind] = dict(SCHEMA[_kind], params=SCHEMA[_kind]["params"] + ([] if _cls.BOX else _PLACE) + _TURN)


def place_box(box, p, w, h):
    """(x0, y0, w, h) of a box of shape `box` placed by p's scale, x, y in a w x h frame (as a file clip, "contain")."""
    bw, bh = box
    k = min(w / bw, h / bh) * p["scale"] / 100
    return (w - bw * k) / 2 + p["x"] / 100 * w, (h - bh * k) / 2 + p["y"] / 100 * h, bw * k, bh * k


def box_of(kind, params, w, h):
    """Where a placed effect's box lies in a w x h frame, without making the effect."""
    return place_box(KINDS[kind].BOX, clean_params(kind, params), w, h)


def sheet_rect(kind, p, w, h):
    """(x, y, sw, sh): where the sheet an effect is drawn on lies in a w x h frame before it is turned (p: its
    cleaned settings): a placed effect's box, or the frame's own shape placed as a file clip of the frame's size
    with fit "contain" (at scale 100, x = y = 0: exactly the frame)."""
    return place_box(KINDS[kind].BOX or (w, h), p, w, h)


def sheet_params(kind, params):
    """The settings an effect is drawn with on its sheet (and its preview clip made with): its place and turn left
    out, so they never change what it draws; a placed effect fills its sheet (its box)."""
    out = {k: v for k, v in (params or {}).items() if k not in SHEET_KEYS}
    return dict(out, x=0, y=0, scale=100) if KINDS[kind].BOX else out


def on_sheet(kind, params):
    """Is a clip of this kind drawn on a sheet that is placed and turned (sheet.Sheet)? Only when it is placed or
    turned: unturned, a full-frame effect at scale 100 in the middle is drawn on the frame as it always was, and a
    placed effect draws itself in its box."""
    cls = KINDS[kind]
    if not cls.TURNS:
        return False
    p = clean_params(kind, params)
    return overlay.rotated(p) or (not cls.BOX and (p["scale"] != 100 or p["x"] != 0 or p["y"] != 0))


def quad(kind, params, w, h):
    """Where the corners of an effect's sheet are in a w x h frame (the page's quad(): the same numbers)."""
    p = clean_params(kind, params)
    rect = sheet_rect(kind, p, w, h)
    return overlay.corners(overlay.homography(p, rect, w, h)[0], rect[2], rect[3])


def make(kind, params, seed, frames, fps, w, h):
    """The effect of one layer, for a loop of `frames` frames at `fps`, drawn at w x h."""
    return KINDS[kind](clean_params(kind, params), seed, frames, fps, w, h)
