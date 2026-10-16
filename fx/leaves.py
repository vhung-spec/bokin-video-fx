"""Falling leaves: maple, Japanese maple, ginkgo, oak, bàng (Indian almond), birch, willow and heart-shaped leaves,
and cherry petals, in red, yellow and green, big and small (see effects.py for how every effect loops).

The leaves' pictures are drawn by code (leafshapes.py): each shape in its own natural colours, front, back and lit
from behind. The shapes are ticked one by one ("Kiểu dáng lá": several at once fall together, in equal shares;
none ticked: maple leaves); "Loại lá" ticks a well-known set at once and colours it (a bàng tree in winter, golden
ginkgo...). Each shape keeps its true size beside the others (a bàng leaf or a maple's is big, a birch's or a
willow's small), all between "Lá nhỏ nhất" and "Lá to nhất".

Every leaf falls like a real one: most glide from side to side like a pendulum, falling fastest in the middle of a
swing and slowing at its ends, tilting into each swing; some tumble end over end, showing their back half the time;
some spiral down. The swing keeps pace with the fall (a slow fall swings slowly and never rocks in place), big
leaves swing slower and wider. The wind leans their path, and gusts come and go (never against the wind), pushing
near leaves more than far ones. Sizes spread from "Lá nhỏ nhất" to "Lá to nhất": many small far away, a few big
near; far leaves are slower, softer and a little hazy, near ones a little out of focus; fast ones smear as a
camera's shutter would. The leaves come down in lanes spread evenly across the frame, and each time a leaf comes
down again (k times a loop) it is another leaf: a little aside, another design, another swing.

Light: a leaf is lit from the upper left, in front: brighter as it turns its face to the light, darker turned away,
its glossy top glinting now and then. The sun is also behind the leaves, a little to the right ("Nắng chiếu xuyên
lá"): a leaf facing it from in front of it glows (its picture lit from behind), warmer and brighter, its veins dark.
"""
import math

import numpy as np
from PIL import Image, ImageFilter

from . import leafshapes
from .effects import Falling, _choice, _lerp, _num, _toggle
from .leafshapes import _SPRITES, FAMILIES, MIPS, PALETTES, SHAPES, SHARE, VARIANTS, sprite, sprite_bytes  # noqa: F401
# (the pictures kept, and the colours the sliders count: as the tests and the page know them)
from .snow import GUST_LEAD, gust_push

# --- the settings -----------------------------------------------------------------------------------------------

SIZE_PX = (10.0, 240.0)  # a leaf's longest side (px of a 1080-line frame) at 0 and 100 on the size sliders


def size_px(v):
    """A leaf's longest side (px of a 1080-line frame) at v (0..100) on a size slider (a log scale)."""
    return SIZE_PX[0] * (SIZE_PX[1] / SIZE_PX[0]) ** (v / 100)


def size_slider(px):
    """Where a leaf `px` long sits on a size slider."""
    return 100 * math.log(max(px, 1e-6) / SIZE_PX[0]) / math.log(SIZE_PX[1] / SIZE_PX[0])


# how big each shape's leaves are beside the others (the longest side of the blade): the biggest ticked reaches "Lá
# to nhất", the others as much smaller as they really are (a petal: much smaller than a maple leaf)
REL = {"almond": 1.0, "maple": 0.88, "oak": 0.84, "japanese": 0.72, "heart": 0.68, "ginkgo": 0.64, "willow": 0.62,
       "birch": 0.52, "petal": 0.53}
SHAPE_LABELS = {"maple": "Lá phong 5 thuỳ", "japanese": "Lá phong Nhật (7 thuỳ mảnh)",
                "ginkgo": "Lá bạch quả (hình quạt)", "oak": "Lá sồi (thuỳ tròn)", "almond": "Lá bàng (to, đỏ cam)",
                "birch": "Lá bạch dương (nhỏ, răng cưa)", "willow": "Lá liễu (dài, mảnh)", "heart": "Lá hình tim",
                "petal": "Cánh hoa anh đào"}
DEFAULT_SHAPES = ("maple", "japanese")
SHAPES_GROUP = "Kiểu dáng lá (tích nhiều ô để trộn)"
COLOURS_GROUP = "Màu lá (cánh hoa luôn màu hồng)"

LOOKS = {  # the "Kiểu lá rơi" presets: how the leaves fall, how many, how big, how deep (medium: the defaults)
    "few": {"amount": 12, "big": 88, "small": 34, "speed": 30, "sway": 60, "spin": 45, "wind": 8, "gust": 15,
            "depth": 55, "blur": 20, "haze": 20, "motion": 30, "opacity": 100},
    "many": {"amount": 75, "big": 82, "small": 16, "speed": 40, "sway": 50, "spin": 55, "wind": 20, "gust": 35,
             "depth": 70, "blur": 30, "haze": 35, "motion": 40, "opacity": 100},
    "windy": {"amount": 60, "big": 82, "small": 18, "speed": 55, "sway": 40, "spin": 80, "wind": 60, "gust": 75,
              "depth": 65, "blur": 35, "haze": 30, "motion": 65, "opacity": 100},
    "dreamy": {"amount": 22, "big": 92, "small": 40, "speed": 18, "sway": 60, "spin": 35, "wind": 5, "gust": 10,
               "depth": 75, "blur": 60, "haze": 25, "motion": 40, "opacity": 100},
}
SEASONS = {  # the "Màu theo mùa" presets: how many leaves are red, yellow and green
    "red": {"red": 80, "yellow": 15, "green": 5},
    "gold": {"red": 8, "yellow": 85, "green": 7},
    "redgold": {"red": 50, "yellow": 50, "green": 0},
    "winter": {"red": 55, "yellow": 35, "green": 10},
    "mixed": {"red": 40, "yellow": 35, "green": 25},
    "early": {"red": 12, "yellow": 23, "green": 65},
    "summer": {"red": 0, "yellow": 0, "green": 100},
}
# the "Loại lá" presets: which shapes, in which season's colours (cherry petals: the colours left as they are)
SCENE_LIST = [("maple", "Rừng phong (phong 5 thuỳ và phong Nhật)", DEFAULT_SHAPES, "mixed"),
              ("japanese", "Phong Nhật đỏ rực", ("japanese",), "red"),
              ("almond", "Lá bàng mùa đông", ("almond",), "winter"),
              ("ginkgo", "Bạch quả vàng", ("ginkgo",), "gold"),
              ("oak", "Rừng sồi", ("oak",), "redgold"),
              ("birch", "Bạch dương vàng", ("birch",), "gold"),
              ("willow", "Lá liễu", ("willow",), "early"),
              ("heart", "Lá hình tim", ("heart",), "gold"),
              ("petal", "Hoa anh đào", ("petal",), None),
              ("forest", "Rừng thu nhiều loại lá", ("maple", "oak", "birch", "heart"), "mixed")]
SCENES = {name: {**{s: s in shapes for s in SHAPES}, **({"season": season, **SEASONS[season]} if season else {})}
          for name, _, shapes, season in SCENE_LIST}
CUSTOM = "custom"  # (the shapes ticked by hand: no preset)

SCHEMA = {"label": "Lá rơi", "params": [
    _choice("scene", "Loại lá (chọn nhanh)", "maple",
            [(name, label) for name, label, _, _ in SCENE_LIST] + [(CUSTOM, "Tự chọn (tích ô bên dưới)")],
            presets=SCENES, match=list(SHAPES), group="Kiểu"),
    _choice("look", "Kiểu lá rơi", "medium",
            [("few", "Lác đác vài lá"), ("medium", "Lá rơi vừa"), ("many", "Lá rơi dày"),
             ("windy", "Gió thu cuốn lá"), ("dreamy", "Lá to bay chậm")], presets=LOOKS, group="Kiểu"),
    *[_toggle(s, SHAPE_LABELS[s], s in DEFAULT_SHAPES, group=SHAPES_GROUP) for s in SHAPES],
    _choice("season", "Màu theo mùa", "mixed",
            [("red", "Thu đỏ rực"), ("gold", "Thu vàng"), ("redgold", "Thu đỏ vàng"),
             ("winter", "Đỏ cam, vàng, ít lá xanh"), ("mixed", "Đỏ vàng xanh"),
             ("early", "Đầu thu (nhiều lá xanh)"), ("summer", "Lá xanh (mùa hè)")], presets=SEASONS,
            group=COLOURS_GROUP),
    _num("red", "Lá đỏ (nhiều hay ít)", 40, group=COLOURS_GROUP),
    _num("yellow", "Lá vàng (nhiều hay ít)", 35, group=COLOURS_GROUP),
    _num("green", "Lá xanh (nhiều hay ít)", 25, group=COLOURS_GROUP),
    _num("glow", "Nắng chiếu xuyên lá (lá rực sáng)", 40, group="Ánh sáng"),
    _num("big", "Lá to nhất", 82, group="Cỡ lá"), _num("small", "Lá nhỏ nhất", 22, group="Cỡ lá"),
    _num("amount", "Số lá", 35, group="Số lá"), _num("opacity", "Độ đậm", 100, group="Số lá"),
    _num("speed", "Tốc độ rơi", 35, group="Chuyển động"),
    _num("sway", "Chao qua lại khi rơi", 55, group="Chuyển động"),
    _num("spin", "Xoay, lật mặt, lộn vòng", 50, group="Chuyển động"),
    _num("wind", "Gió (trái ← → phải)", 15, -100, 100, group="Gió"),
    _num("gust", "Gió giật từng cơn", 25, group="Gió"),
    _num("depth", "Chiều sâu (xa nhỏ, gần to)", 60, group="Chiều sâu"),
    _num("blur", "Nhoè lá ở gần", 20, group="Chiều sâu"),
    _num("haze", "Lá ở xa mờ nhạt", 30, group="Chiều sâu"),
    _num("motion", "Nhoè khi lá bay nhanh", 30, group="Chiều sâu")]}
LOOKS["medium"] = {p["key"]: p["value"] for p in SCHEMA["params"] if p["key"] in LOOKS["few"]}

OLD_STYLES = {"autumn": ("mixed", "redgold"), "green": ("oval", "summer"), "petal": ("petal", "mixed")}
OLD_FULL = 45  # leaves in the frame at a time at the top of the first leaves' "Số lá" (its 60 counted its margins)
# the kinds of leaf of the remake before the shapes ("Loại lá": maple leaves, both kinds of maple; mixed with
# ordinary oval leaves; ordinary leaves alone; petals), as shapes
OLD_SHAPES = {"maple": DEFAULT_SHAPES, "mixed": DEFAULT_SHAPES + ("birch",), "oval": ("birch",), "petal": ("petal",)}


def scene_of(p):
    """The "Loại lá" whose shapes are those ticked in `p` (custom when none is)."""
    return next((name for name, preset in SCENES.items() if all(p.get(s) == preset[s] for s in SHAPES)), CUSTOM)


def upgrade(raw):
    """Settings saved by an older leaves effect, as this one's: the first one's (style, size; no kind of leaf),
    and the remake's single kind of leaf ("shape"), as the shapes ticked."""
    if not isinstance(raw, dict) or any(s in raw for s in SHAPES):
        return raw
    out = dict(raw)
    if "shape" not in raw and {"style", "size"} & set(raw):  # the first leaves effect
        shape, season = OLD_STYLES.get(raw.get("style"), OLD_STYLES["autumn"])
        out.update(shape=shape, season=season, **SEASONS[season])
        try:
            size = min(max(float(raw.get("size", 45)), 0.0), 100.0)
        except (TypeError, ValueError, OverflowError):
            size = 45.0
        length = _lerp(22, 80, size / 100) if size == size else 51.0  # its leaves were about this long, 0.6 to 1.8
        out["big"], out["small"] = round(size_slider(1.8 * length)), round(size_slider(0.7 * length))
        try:
            out["amount"] = round(min(max(float(raw.get("amount", 30)), 0.0), 100.0) * OLD_FULL / Leaves.FULL)
        except (TypeError, ValueError, OverflowError):
            pass
    if "shape" in out:
        old = out.pop("shape")
        names = OLD_SHAPES.get(old, DEFAULT_SHAPES) if isinstance(old, str) else DEFAULT_SHAPES
        out.update({s: s in names for s in SHAPES})
        out["scene"] = scene_of(out)
    return out


# --- the effect -------------------------------------------------------------------------------------------------

GLIDE, TUMBLE, SPIRAL = 0, 1, 2  # how a leaf falls
SQUASHED = 0.75  # seen at more of a slant than this (its width / its length), a leaf is drawn in two steps
PASS = (0.6180339887, 0.7548776662)  # steps of two sequences that spread evenly (lanes; a lane's shift each pass)
DESIGNS_MOST = 48  # designs (shape x colour x design) one clip draws from at most: fewer each with many shapes
CLIP_SHARE = 0.4  # the share of the pictures' memory (leafshapes.MAX_BYTES) one clip's pictures take at most: two
# of the heaviest leaves clips (or three as usual) drawn together never make a picture again


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


FRONT_LIGHT = _unit([-0.35, -0.6, -0.72])  # towards the light in front (camera: x right, y down, z away from it)
BACK_LIGHT = _unit([0.3, -0.5, 0.8])  # towards the sun behind the leaves
HALF = _unit(FRONT_LIGHT + np.array([0.0, 0.0, -1.0]))  # (between the light and the camera: the glint)
GLOWING = 1.1  # how bright a leaf glowing in the sun behind it is (1: face on to the light in front)


def _uniform_again(s, a):
    """s = a U1 + (1 - a) U2 (U1, U2 independent, uniform 0..1) made uniform 0..1 again: its distribution function
    (a weighted mean of two draws would bunch in the middle)."""
    a, b = max(a, 1 - a), min(a, 1 - a)
    if b < 1e-6:
        return np.clip(s, 0.0, 1.0)
    return np.clip(np.where(s < b, s * s / (2 * a * b),
                            np.where(s <= a, (s - b / 2) / a, 1 - (1 - s) ** 2 / (2 * a * b))), 0.0, 1.0)


def ticked(p):
    """The shapes ticked in settings `p`, in their order (none ticked: maple leaves)."""
    return [s for s in SHAPES if p.get(s)] or ["maple"]


class Leaves(Falling):
    CAP = 700  # leaves with fixed random draws; how many are used follows the "Số lá" slider
    VERSION = 4  # (1: the first leaves, simple shapes; 2: the remake before its review; 3: the remake)
    FULL = 90  # leaves on screen at a time at the top of the "Số lá" slider (16:9)
    SHUTTER = 0.5  # the share of a frame the shutter is open at the top of the motion blur slider
    SWING_HZ = 0.6  # swings a second of a middling leaf at the default fall speed
    PREVIEW = 960  # its preview clip: as many pixels as 960 x 540 (small leaves blur at 640 wide)
    upgrade = staticmethod(upgrade)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        u = self.rng.random((self.CAP, 40))
        self.gph = float(self.rng.random()) * 2 * math.pi  # when the gusts come
        ds, spin, sway = p["depth"] / 100, p["spin"] / 100, p["sway"] / 100
        shapes = ticked(p)
        # which shape each leaf is (each in an equal share). Ticking another shape gives most leaves another
        # shape, and (the sizes and margins changing) other places too: a new mix of leaves
        shape = np.array(shapes)[np.minimum((u[:, 26] * len(shapes)).astype(int), len(shapes) - 1)]

        # size: from "Lá nhỏ nhất" to "Lá to nhất", many small and a few really big (as many more far away as the
        # space widens away from the camera); with depth, far leaves are the small ones (and slower, hazier), near
        # ones the big ones; `own`: how big a leaf really is (a big one swings slower and wider). Each shape as big
        # as it is beside the others: the biggest ticked reaches "Lá to nhất"
        own = u[:, 2]
        near = _uniform_again(ds * u[:, 0] + (1 - ds) * own, ds) ** 1.25  # 0 small (far) .. 1 big (near)
        lo, hi = sorted((size_px(p["small"]), size_px(p["big"])))
        top = max([REL[s] for s in shapes if s != "petal"] or [REL["maple"]])
        rel = np.array([REL[str(s)] for s in shape])
        tops = np.maximum(lo, hi * rel / top)
        size = lo * (tops / lo) ** near  # px of a 1080-line frame: the leaf's longest side
        seen = size / 1080  # in frame heights

        # the path: down, leaning with the wind (tumbling leaves glide off sideways), plus the gusts' steady drift
        mode = np.where(u[:, 3] < 0.3 * spin, TUMBLE, np.where(u[:, 3] < 0.5 * spin, SPIRAL, GLIDE))
        glide, tumble, spiral = mode == GLIDE, mode == TUMBLE, mode == SPIRAL
        fall = _lerp(0.04, 0.32, p["speed"] / 100)  # frame heights a second
        speed = fall * ((1 - ds) + ds * (0.4 + 1.2 * near)) * (0.85 + 0.3 * u[:, 20]) * (1.15 - 0.3 * own)
        angle = (math.radians(55) * p["wind"] / 100 + np.radians(6) * (u[:, 21] - 0.5)
                 + np.where(tumble, 0.3 * spin * np.sign(u[:, 22] - 0.5), 0.0))
        self.gust_sign = 1.0 if p["wind"] >= 0 else -1.0
        push = p["gust"] / 100 * 0.3 * (0.3 + 0.7 * near)
        self.gm = max(1.0, round(self.loop_s / 8))  # gusts per loop
        drift = self.gust_sign * push * GUST_LEAD * 2 * math.pi * self.gm / self.loop_s
        vx, vy = speed * np.sin(angle) + drift, speed * np.cos(angle)
        angle, speed = np.arctan2(vx, vy), np.hypot(vx, vy)

        # the swing: a pendulum glide (most), tumbling end over end, or a spiral; whole cycles per loop. It keeps
        # pace with the fall (a slower fall swings slower) and big leaves swing slower; never so wide or fast that a
        # leaf seems to rock in place: sideways at most 2-3 times as fast as it falls, rising (if at all) a little
        # at the ends of a swing
        rate = (self.SWING_HZ * (fall / _lerp(0.04, 0.32, 0.35)) ** 0.6 / np.sqrt(0.6 + 0.8 * (0.7 * own + 0.3 * near))
                * _lerp(0.88, 1.12, u[:, 5]))  # (big as it is, and a little as it looks: near)
        self.m = self.cycles(rate)
        w = 2 * math.pi * self.m / self.loop_s  # radians a second
        reach = sway * seen * _lerp(1.5, 4.0, u[:, 4]) * (0.75 + 0.5 * own)
        reach = np.minimum(reach, _lerp(2.0, 3.0, u[:, 35]) * vy / w)
        self.A = np.where(glide, reach, np.where(tumble, 0.35 * reach, 0.0))  # side to side
        self.As = np.where(spiral, 0.6 * reach, 0.0)  # round and round
        self.B = np.minimum(0.6 * self.A, _lerp(0.5, 1.05, u[:, 36]) * vy / w)  # higher at the ends: a pendulum's arc
        self.phi0 = u[:, 6] * 2 * math.pi
        self.R = np.where(glide, _lerp(0.3, 0.7, u[:, 7]) * (0.3 + 0.7 * sway), 0.0) * (0.5 + spin)  # tilt in a swing
        self.Rs = np.where(spiral, _lerp(0.15, 0.35, u[:, 7]), 0.0)
        self.j = np.where(spiral, np.where(u[:, 8] < 0.5, -1.0, 1.0), 0.0)  # a spiral turns it round once a circle
        self.turns = np.sign(u[:, 8] - 0.5) * np.rint(spin * np.where(tumble, _lerp(0.03, 0.2, u[:, 9]),
                                                                        _lerp(0.0, 0.07, u[:, 9])) * self.loop_s)
        self.turns = np.where(spiral, 0.0, self.turns)  # (whole turns a loop; gliding ones turn slowly, if at all)
        self.flips = np.where(tumble, np.sign(u[:, 10] - 0.5) * self.cycles(_lerp(0.35, 1.0, u[:, 11]) * (0.5 + spin)),
                              0.0)
        back = np.where(u[:, 12] < 0.25, math.pi, 0.0)  # (a quarter of them fall showing their back)
        self.beta0 = np.where(spiral, np.sign(u[:, 13] - 0.5) * _lerp(0.5, 1.0, u[:, 14]),
                              np.clip(np.tan((u[:, 13] - 0.5) * 2.2) * 0.45, -1.2, 1.2)) + back
        self.P_amp = np.where(glide, _lerp(0.2, 0.55, u[:, 14]) * sway, np.where(spiral, 0.25, 0.0))
        self.wm = self.cycles(_lerp(1.2, 2.6, u[:, 15]))  # a quick flutter on top
        self.wph = u[:, 16] * 2 * math.pi
        self.W = np.where(tumble, 0.0, 0.1 * spin * _lerp(0.4, 1.0, u[:, 17]))
        self.yaw0 = u[:, 18] * 2 * math.pi
        self.chi = np.clip(np.tan((u[:, 19] - 0.5) * 2.0) * 0.35, -1.2, 1.2)  # the axis it tilts about
        self.zs = np.where(spiral, 0.07, 0.0)  # nearer and further as it circles

        # out of focus near the camera, hazy far away; the margins they may reach beyond the frame
        blur = p["blur"] / 100 * np.clip((near - 0.72) / 0.28, 0, 1) ** 1.5 * (0.5 + 0.045 * size)  # px (1080 lines)
        haze = p["haze"] / 100 * 0.6 * (1 - near) ** 1.5 * (0.4 + 0.6 * ds)
        out = 0.85 * seen * (1 + self.zs) + 3 * blur / 1080 + 0.01
        margin_y = out + self.B + 0.3 * push + 0.01
        margin_x = out + self.A + self.As + 0.5 * push + 0.02
        lanes = np.mod(u[0, 23] + PASS[0] * np.arange(self.CAP), 1.0)  # (spread evenly, however many are used)
        share = self.paths(lanes, u[:, 24], speed, angle, margin_y, margin_x)

        # how many: about `wanted` in the frame at a time (counted with the margins, where leaves also are)
        wider = (1 + 2 * float(margin_x.mean()) / self.aspect) * (1 + 2 * float(margin_y.mean()))
        wanted = p["amount"] / 100 * self.FULL * self.aspect / (16 / 9) * wider
        n = min(int(np.searchsorted(np.cumsum(share), wanted)) + 1 if wanted > 0 else 0, self.CAP)
        # their lanes again, biggest first: the big leaves spread evenly among themselves too (no side of the frame
        # left with only small ones)
        rank = np.empty(n, int)
        rank[np.argsort(-size[:n], kind="stable")] = np.arange(n)
        self.lane = np.mod(u[0, 23] + PASS[0] * rank, 1.0)
        self.paths(self.lane, u[:n, 24], speed[:n], angle[:n], margin_y[:n], margin_x[:n])
        drift = (1 + 2 * margin_y[:n]) * self.sin / self.cos  # (the lanes' room, as paths() has it)
        self.lo = np.minimum(0.0, -drift) - margin_x[:n]
        self.room = np.maximum(self.aspect, self.aspect - drift) + margin_x[:n] - self.lo
        # each time a leaf comes down again (k times a loop) it is another leaf: moved across (about the room its
        # lane has: the biggest wander furthest, so no band of the frame misses them for a whole loop), swinging
        # another way, another design
        self.lane_shift = 0.8 / np.minimum(rank + 1, max(n, 1))
        order = np.argsort(near[:n], kind="stable")  # drawn far to near
        for name in ("cos", "sin", "k", "len", "extra", "s0", "xe", "speed", "my", "lane", "lo", "room",
                     "lane_shift"):
            setattr(self, name, getattr(self, name)[order])
        for name in ("A", "As", "B", "m", "phi0", "R", "Rs", "j", "turns", "flips", "beta0", "P_amp", "wm", "wph",
                     "W", "yaw0", "chi", "zs"):
            setattr(self, name, getattr(self, name)[order])
        u = u[order]
        self.n = n
        self.z, self.size, self.mode, self.own = near[order], size[order], mode[order], own[order]
        self.blur, self.haze = blur[order] * self.px, haze[order]
        self.push = push[order]
        self.pass0 = u[:, 25]

        # which leaf: its shape, colour and design; a little brighter or darker, warmer or cooler than its design.
        # Green leaves turn yellow or red at their edges only when some leaves have already turned (not in summer)
        self.kind = shape[:n][order]
        shares = np.array([p["red"], p["yellow"], p["green"]], float)
        shares = shares / shares.sum() if shares.sum() > 0 else np.full(3, 1 / 3)
        self.family = np.searchsorted(np.cumsum(shares)[:2], u[:, 27], side="right")
        turning = min(0.5, 1.2 * (shares[0] + shares[1]))  # (the share of green leaves turning)
        self.palette = np.where((self.family == 2) & (u[:, 34] < turning), 3, self.family)
        petal = self.kind == "petal"
        self.tint = (_lerp(0.9, 1.08, u[:, 29])[:, None] * (1 + 0.05 * (u[:, 30:33] - 0.5))).astype(np.float32)
        self.opacity = p["opacity"] / 100 * np.where(petal, 0.95, 1.0) * (1 - 0.35 * self.haze)
        self.sheen = np.where(self.kind == "almond", _lerp(0.25, 0.5, u[:, 33]), 0.25 * u[:, 33])
        self.sheen = np.where(petal, 0.05, self.sheen)
        # in the sun or in the shade: how much a leaf glows when the sun is behind it
        self.sun = p["glow"] / 100 * _lerp(0.35, 1.0, u[:, 37]) * np.where(petal, 0.6, 1.0)
        self.soft = np.clip(_lerp(0.72, 1.0, near[order] / 0.35), 0.72, 1.0)  # (far ones a little softer)
        # the leaves' pictures: each drawn as large as the leaf will be seen (small in a small preview: its shape's
        # blade takes SHARE of its picture at least), from as many designs as memory allows
        self.longest = self.size * self.px * (1 + self.zs)
        self.master = np.array([next((s for s in MIPS[2::-1] if SHARE[str(kind)] * s >= 1.1 * v), MIPS[0])
                                for kind, v in zip(self.kind, self.longest)], int)
        self.count = self._designs(u[:, 28])

    def _designs(self, draw):
        """How many designs of each shape the leaves come in (fewer when many shapes and colours fall together, so
        that all their pictures fit in a share of memory at once, room left for another leaves clip: none is made
        again and again), which each leaf starts with (`variant`), and how large each design's pictures are made
        (`sizes`: as large as the largest leaf showing it needs, the others drawn from them, so no design's pictures
        are made twice)."""
        pal = np.where(self.kind == "petal", 0, self.palette)
        combos = len(set(zip(self.kind.tolist(), pal.tolist()))) or 1
        most = max(2, DESIGNS_MOST // combos)
        while True:
            count = np.array([min(VARIANTS[str(k)], most) for k in self.kind], int)
            self.variant = (draw * count).astype(int)
            self.sizes = {}
            for i in range(self.n):
                for j in range(int(min(self.k[i], count[i]))):
                    key = (str(self.kind[i]), int(pal[i]), int((self.variant[i] + 3 * j) % count[i]))
                    self.sizes[key] = max(self.sizes.get(key, 0), int(self.master[i]))
            if most <= 1 or sum(map(sprite_bytes, self.sizes.values())) <= CLIP_SHARE * leafshapes.MAX_BYTES:
                return count
            most -= 1

    def passes(self, t):
        """Which time round (0 .. k - 1) every leaf is on at frame t (it comes down k times a loop)."""
        return np.mod(np.floor((self.s0 + self.k * self.len * self.tau(t)) / self.len), self.k)

    def design(self, t):
        """Which design every leaf is at frame t (another each time round)."""
        return (self.variant + 3 * self.passes(t).astype(int)) % self.count

    def at(self, t, dt=0.0):
        """x, y (pixels) of every leaf's middle at frame t + dt, whether it is on its visible stretch at frame t, and
        its swing's phase. Along its path a leaf is followed from where it is at t (not started again at the top)."""
        s = np.mod(self.s0 + self.k * self.len * self.tau(t), self.len)
        on = s >= self.extra
        d = s + self.k * self.len * (dt / self.P) - self.extra
        j = self.passes(t)
        lane = np.mod(self.lane + self.lane_shift * (np.mod(self.pass0 + PASS[1] * j, 1.0) - 0.5), 1.0)
        x = self.lo + lane * self.room + d * self.sin
        y = -self.my + d * self.cos
        tau = self.tau(t + dt)
        phi = 2 * math.pi * self.m * tau + self.phi0 + 2.4 * j
        sn = np.sin(phi)
        x = x + self.A * sn + self.As * np.cos(phi)
        y = y - self.B * sn * sn
        if self.push.any():  # a gust sweeps across from the side the wind comes from, lifting the leaves a little
            theta = 2 * math.pi * self.gm * tau + self.gph - 2 * math.pi * 0.35 * self.gust_sign * x / self.aspect
            x = x + self.gust_sign * self.push * gust_push(theta)
            y = y - 0.25 * self.push * (np.exp(4 * (np.cos(theta) - 1)) - 0.25)
        return x * self.h, y * self.h, on, tau, phi

    def pose(self, tau, phi):
        """How every leaf is turned: tilt in the picture (roll), its own turn (yaw), how far it is turned towards
        or away from the camera (pitch: past a right angle its back shows) and a scale."""
        wob = np.sin(2 * math.pi * self.wm * tau + self.wph)
        roll = -self.R * np.sin(phi) + self.Rs * np.cos(phi) + self.W * wob
        yaw = self.yaw0 + self.j * phi + 2 * math.pi * self.turns * tau
        pitch = self.beta0 + self.P_amp * np.sin(phi) + 2 * math.pi * self.flips * tau + 1.5 * self.W * wob
        return roll, yaw, pitch, 1 + self.zs * np.sin(phi)

    def light(self, roll, pitch):
        """How bright every leaf is in the light in front (1: face on), and how much it glows lit from behind
        (0..1), from which way its face points (tilted about an axis at roll + chi in the picture by pitch)."""
        a = roll + self.chi
        sp, cp = np.sin(pitch), np.cos(pitch)
        face = np.where(cp >= 0, 1.0, -1.0)  # (the side seen: its front, or its back)
        n = np.stack([-sp * np.sin(a), sp * np.cos(a), -cp]) * face  # (towards the camera)
        lit = 0.75 + 0.35 * np.maximum(n.T @ FRONT_LIGHT, 0.0)
        lit = lit + np.where(cp > 0, self.sheen * np.maximum(n.T @ HALF, 0.0) ** 30, 0.0)  # (the top glints)
        glow = self.sun * np.maximum(-(n.T @ BACK_LIGHT), 0.0) ** 1.5
        return lit, glow

    def draw(self, t, canvas, mask):
        if not self.n:
            return
        x, y, on, tau, phi = self.at(t)
        shutter = self.SHUTTER * self.p["motion"] / 100
        if shutter > 0:
            xb, yb = self.at(t, -shutter / 2)[:2]
            xa, ya = self.at(t, shutter / 2)[:2]
            mx, my = xa - xb, ya - yb
        else:
            mx = my = np.zeros(self.n)
        roll, yaw, pitch, scale = self.pose(tau, phi)
        length = self.size * self.px * scale
        reach = 0.85 * length + 3 * self.blur + np.hypot(mx, my) + 2
        seen = on & (x > -reach) & (x < self.w + reach) & (y > -reach) & (y < self.h + reach)
        # the picture's turn and squash (seen at a slant): R(roll + chi) diag(1, squash) R(yaw - chi); turned over
        # (squash < 0), the leaf is seen mirrored
        c = np.cos(pitch)
        squash = np.copysign(np.maximum(np.abs(c), 0.05), c)
        lit, glow = self.light(roll, pitch)
        glow = np.minimum(glow, 0.85)
        gain = self.tint * ((lit + (GLOWING - lit) * glow) * self.opacity)[:, None]
        design = self.design(t)
        for i in np.flatnonzero(seen):
            spr = self.picture(i, int(design[i]))
            out = place(spr, c[i] < 0, length[i], roll[i] + self.chi[i], yaw[i] - self.chi[i], squash[i],
                        x[i], y[i], 3 * float(self.blur[i]) + 2, float(glow[i]), float(self.soft[i]),
                        (float(mx[i]), float(my[i])))
            self._finish(i, spr, out, gain[i], canvas, mask)

    def picture(self, i, design):
        """The pictures of leaf i's design: drawn large enough for it (made larger when its blade turns out to be
        smaller in its picture than the leaf will be seen)."""
        kind, pal = str(self.kind[i]), PALETTES[self.palette[i]]
        size = self.sizes.get((kind, 0 if kind == "petal" else int(self.palette[i]), design), int(self.master[i]))
        spr = sprite(kind, pal, design, size)
        while spr.extent < self.longest[i] and spr.size < MIPS[0]:
            spr = sprite(kind, pal, design, spr.size * 2)
        return spr

    def _finish(self, i, spr, placed, gain, canvas, mask):
        out, x0, y0 = placed
        if self.blur[i] > 0.4:
            out = out.filter(ImageFilter.GaussianBlur(float(self.blur[i])))
        drawn = out.getbbox()  # (only what the leaf covers: its box's corners are often empty)
        if drawn is None:
            return
        if drawn != (0, 0) + out.size:
            out = out.crop(drawn)
            x0, y0 = x0 + drawn[0], y0 + drawn[1]
        arr = np.asarray(out, np.float32) * (1 / 255.0)
        rgb = arr[..., :3]
        rgb *= gain
        arr[..., 3] *= self.opacity[i]
        hz = self.haze[i]
        if hz > 0.01:  # far away: paler, greyer
            grey = float(spr.rgb @ np.array([0.3, 0.55, 0.15]))
            target = (0.6 * grey + 0.4 * np.array([0.8, 0.82, 0.85])).astype(np.float32)
            rgb *= 1 - hz
            rgb += (hz * target) * arr[..., 3:4]
        np.minimum(rgb, arr[..., 3:4], out=rgb)  # (never brighter than it covers: no light rims)
        canvas.over_rgba(x0, y0, arr[..., :3], arr[..., 3], mask)


def place(spr, back, length, roll, turn, squash, x, y, pad, glow=0.0, soft=1.0, smear=(0.0, 0.0)):
    """A leaf's picture (`spr`, its back if `back`, lit from behind by `glow`) `length` px long, turned by `turn`,
    squashed across by `squash` (seen at a slant; < 0: turned over, mirrored), turned by `roll` in the frame, its
    middle at (x, y), smeared over the (dx, dy) px of `smear` it moves while the shutter is open: a Pillow "RGBa"
    image and the frame px of its top-left corner.

    The picture is chosen by the leaf's full length (`soft` < 1: up to 1 / soft times smaller, a little soft, for a
    far leaf), so its long side is not enlarged more than that (no blur that comes and goes as it turns). Squashed a
    lot, it is turned first, made lower with a filter that averages (not point by point: a leaf seen edge-on stays a
    solid line, no shimmer), then turned to its place. Only the box round the leaf is drawn (its corners where the
    turns take them: a ginkgo's long stem or a willow's narrow blade leave much of its square picture empty)."""
    img = spr.image(back, length * soft, glow)
    n = img.size[0]
    f = n / spr.size
    k = length / (spr.extent * f)  # frame px per picture px
    corners = [(px * f, py * f) for px in spr.box[0::2] for py in spr.box[1::2]]  # (from the picture's middle)
    ca, sa, cb, sb = math.cos(roll), math.sin(roll), math.cos(turn), math.sin(turn)
    # the frame box: where the corners go, R(roll) diag(1, squash) R(turn) k
    a, b = k * (ca * cb - sa * squash * sb), k * (-ca * sb - sa * squash * cb)
    c, d = k * (sa * cb + ca * squash * sb), k * (-sa * sb + ca * squash * cb)
    us, vs = [a * px + b * py for px, py in corners], [c * px + d * py for px, py in corners]
    box = (min(us) - pad, min(vs) - pad, max(us) + pad, max(vs) + pad)
    if abs(squash) >= SQUASHED:  # in one step: frame offsets -> the picture, the inverse of [[a, b], [c, d]]
        det = a * d - b * c
        ia, ib, ic, id_ = d / det, -b / det, -c / det, a / det
        return _laid(img, (ia, ib, n / 2, ic, id_, n / 2), x, y, box, smear, Image.Resampling.BICUBIC)
    # 1. turned by `turn` and made `k` times the size (k R(turn)), in a box round the leaf, its middle at (ox, oy)
    qx = [k * (cb * px - sb * py) for px, py in corners]
    qy = [k * (sb * px + cb * py) for px, py in corners]
    ox, oy = 2 - math.floor(min(qx)), 2 - math.floor(min(qy))
    tw, th = ox + math.ceil(max(qx)) + 2, oy + math.ceil(max(qy)) + 2
    data = (cb / k, sb / k, n / 2 - (cb * ox + sb * oy) / k, -sb / k, cb / k, n / 2 - (cb * oy - sb * ox) / k)
    turned = img.transform((tw, th), Image.Transform.AFFINE, data, resample=Image.Resampling.BICUBIC)
    # 2. squashed up-down, averaging
    low = max(1, round(th * abs(squash)))
    turned = turned.resize((tw, low), Image.Resampling.LANCZOS)
    e = math.copysign(abs(squash) * th / low, squash)  # (what is left of the squash: about 1, and the mirroring)
    # 3. turned by `roll` into the frame: frame offset = R(roll) diag(1, e) (q - middle)
    return _laid(turned, (ca, sa, ox, -sa / e, ca / e, oy * low / th), x, y, box, smear, Image.Resampling.BICUBIC)


def _warp(img, src, size, out, resample):
    """`img` drawn into a picture of `size`: `out` (an affine map, as Pillow's) takes that picture's points to frame
    offsets (u, v) from the leaf's middle, `src` takes frame offsets to points of `img`."""
    p, q, r, s, t, w = src
    o0, o1, o2, o3, o4, o5 = out
    data = (p * o0 + q * o3, p * o1 + q * o4, p * o2 + q * o5 + r,
            s * o0 + t * o3, s * o1 + t * o4, s * o2 + t * o5 + w)
    return img.transform(size, Image.Transform.AFFINE, data, resample=resample)


def _laid(img, src, x, y, box, smear, resample):
    """`img` laid in the frame, its points at frame offsets (u, v) from (x, y) where `src` takes them in it, within
    `box` (u0, v0, u1, v1) round there: (picture, x0, y0). Smeared over (dx, dy) = `smear`: drawn first where the
    way it moves runs straight down (or across: each row moved aside by a fraction of a pixel, never whole pixels,
    which would leave a fine zipper along its edges), averaged along it (a box: a long smear costs no more than a
    short one), then put back."""
    u0, v0, u1, v1 = box
    dx, dy = smear
    if dx * dx + dy * dy <= 1.0:
        x0, y0 = math.floor(x + u0), math.floor(y + v0)
        size = (math.ceil(x + u1) - x0, math.ceil(y + v1) - y0)
        return _warp(img, src, size, (1, 0, x0 - x, 0, 1, y0 - y), resample), x0, y0
    down = abs(dy) >= abs(dx)
    k = dx / dy if down else dy / dx
    if down:  # sheared: (u - k v, v), the way straight down
        lo, hi = u0 - max(k * v0, k * v1) - 2, u1 - min(k * v0, k * v1) + 2
        sox, soy = math.floor(lo), math.floor(v0 - abs(dy) / 2 - 2)
        size = (math.ceil(hi) - sox, math.ceil(v1 + abs(dy) / 2 + 2) - soy)
        sheared = _warp(img, src, size, (1, k, sox + k * soy, 0, 1, soy), resample)
        sheared = sheared.filter(ImageFilter.BoxBlur((0, max(0.0, (abs(dy) - 1) / 2))))
    else:  # sheared: (u, v - k u), the way straight across
        lo, hi = v0 - max(k * u0, k * u1) - 2, v1 - min(k * u0, k * u1) + 2
        sox, soy = math.floor(u0 - abs(dx) / 2 - 2), math.floor(lo)
        size = (math.ceil(u1 + abs(dx) / 2 + 2) - sox, math.ceil(hi) - soy)
        sheared = _warp(img, src, size, (1, 0, sox, k, 1, soy + k * sox), resample)
        sheared = sheared.filter(ImageFilter.BoxBlur((max(0.0, (abs(dx) - 1) / 2), 0)))
    x0, y0 = math.floor(x + u0 - abs(dx) / 2 - 1), math.floor(y + v0 - abs(dy) / 2 - 1)
    fox, foy = x0 - x, y0 - y
    data = (1, -k, fox - k * foy - sox, 0, 1, foy - soy) if down else (1, 0, fox - sox, -k, 1, foy - k * fox - soy)
    size = (math.ceil(x + u1 + abs(dx) / 2 + 1) - x0, math.ceil(y + v1 + abs(dy) / 2 + 1) - y0)
    return sheared.transform(size, Image.Transform.AFFINE, data, resample=Image.Resampling.BILINEAR), x0, y0


def smear(img, dx, dy):
    """A picture (Pillow "RGBa") smeared evenly over the (dx, dy) px it moves while the shutter is open (as place()
    smears a leaf). Returns (image, x0, y0): where its top-left pixel now sits relative to the original's."""
    w, h = img.size
    return _laid(img, (1, 0, w / 2, 0, 1, h / 2), w / 2, h / 2, (-w / 2, -h / 2, w / 2, h / 2), (dx, dy),
                 Image.Resampling.BILINEAR)
