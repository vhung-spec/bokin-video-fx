"""Smoke from the chimneys in a picture ("Khói ống khói"): wood smoke rising from a cabin's, a farmhouse's, a whole
winter village's chimneys, each clicked on the picture, all of them in one clip, leaning in the same wind (a
full-frame effect drawn over the picture; see effects.py for how every effect loops).

The owner clicks each chimney's mouth on the picture (the page: a yellow ellipse there, its ends dragged to make it
as wide as the mouth), kept in the clip's settings as places in the frame ("stacks": the mouth's middle, x and y as
shares of the frame's width and height; its width as a share of the frame's width; how much smoke it gives, 0..1, 0.5
as the sliders say). Each chimney's smoke is the smoke's own (fx/steam.py, its chimney style: wood smoke drawn as dye
carried by moving air) in a box standing on the mouth as the smoke's box stands on a cup, as many mouth widths high
as "Khói bay cao" makes it: a chimney far away (a small mouth) has a small plume, a near one a big one. Each has its
puffs and whirls of its own (its own seed: no two chimneys smoke in step); the wind ("Gió nghiêng", "Gió lay qua lại")
is the same for all of them, as it is outside. Lit by the scene (light=: the scene's light map over the frame), each
by the light where it stands. The higher up the picture a chimney is, the farther away it is: its smoke is drawn
first, behind the nearer ones'.
"""
import math

import numpy as np

from . import steam
from .effects import Effect, _num, _points_param

STACKS = 64  # chimneys at most
TALL = 12.5  # a plume's box, mouth widths high, at "tall" 50 (as the smoke's chimney style: its mark 8% of its box)
SIDE = 2.0  # "tall" 0 .. 100: a plume half .. twice as high
MOUTH = 0.025  # a new chimney's mouth (the page), a share of the frame's width
MUCH = 0.5  # how much smoke a new chimney gives (0..1: as the sliders say at 0.5)

_STEAM = {d["key"]: d for d in steam.SCHEMA["params"]}
_PRESET = steam.STYLE_PRESETS["chimney"]  # (the sliders as the smoke's chimney style sets them)
SHARED = ("amount", "spread", "soft", "speed", "curl", "breeze", "gust", "opacity", "lit", "color")


def _like(key):
    """The smoke's own setting, at the value its chimney style sets."""
    return dict(_STEAM[key], value=_PRESET.get(key, _STEAM[key]["value"]))


SCHEMA = {
    "label": "Khói ống khói",
    "stacks_most": STACKS,
    "mouth": MOUTH, "much": MUCH,  # (the page: a new chimney)
    "plume": {"ratio": steam.MARK, "tall": TALL, "side": SIDE},  # (the page: its ellipse, the box its smoke rises in)
    "note": "Khói bốc lên từ các ống khói có sẵn trong hình (nhà gỗ, cả làng tuyết): mọi ống khói trong một clip, "
            "cùng một hướng gió. Bấm lên miệng từng ống khói trên hình: có một hình bầu dục vàng ở đó, kéo hai đầu nó "
            "cho rộng bằng miệng ống khói (ống khói xa thì nhỏ, khói của nó cũng nhỏ). Cách đặt ở khung ngay dưới. "
            "Bấm Suốt video để khói bay cả video.",
    "params": [
        _points_param("stacks", "Ống khói", 4, STACKS, group="Ống khói"),  # (put on the picture; Dán thiết lập: the
        # chimneys' places on their own, not with the smoke's look)
        _num("tall", "Khói bay cao (thấp ← → cao)", 50, group="Khói"),
        *(_like(k) for k in SHARED)]}


def stacks_of(points):
    """The chimneys in a list of points ([x, y, width, how much]): each one's mouth (x, y as shares of the frame's
    width and height), width (a share of the frame's width) and how much smoke it gives (0..1), its place in the
    list kept (its seed). One with no width: left out."""
    return [(i, x, y, wid, much) for i, (x, y, wid, much) in enumerate(points) if wid > 0]


def plume_box(x, y, wid, tall, w, h):
    """(x0, y0, side): the box (pixels, a square) a chimney's smoke rises in, in a w x h frame: its mark (the smoke's
    ellipse, MARK as high as wide) on the mouth, as wide, sitting on the box's bottom edge; as many mouth widths high
    as "tall" makes it."""
    mw = wid * w
    side = mw * TALL * SIDE ** ((tall - 50) / 50)
    return x * w - side / 2, y * h + steam.MARK / 2 * mw - side, side


class _Plume(steam.Steam):
    """One chimney's smoke: the smoke's own, its chimney style, in a box the chimney places (not x, y and scale)."""

    def __init__(self, rect, *a, light=None):
        self.rect = rect
        super().__init__(*a, light=light)

    def box(self):
        return self.rect


class Chimney(Effect):
    BOX = None  # (the whole frame: a chimney anywhere on the picture)
    TURNS = False  # (smoke rises straight up from the chimneys where they are)
    LIT = True
    VERSION = 1
    PREVIEW = 1280  # (small plumes over the whole picture: a sharper preview clip)

    def __init__(self, *a, light=None):
        super().__init__(*a)
        p = self.p
        from .effects import clean_params  # (defined after the effects are)
        base = clean_params("steam", {k: p[k] for k in SHARED})
        base.update(style="chimney", reach=100, x=0.0, y=0.0, scale=100.0)
        size = TALL * SIDE ** ((p["tall"] - 50) / 50)
        base["rim"] = 100.0 / size  # (its mark: the mouth, as wide)
        far_first = sorted(stacks_of(p["stacks"]), key=lambda s: s[2])
        self.plumes = []
        for i, x, y, wid, much in far_first:
            amount = min(100.0, p["amount"] * 2 * much)
            if amount <= 0:
                continue
            x0, y0, side = plume_box(x, y, wid, p["tall"], self.w, self.h)
            if x0 >= self.w or y0 >= self.h or x0 + side <= 0 or y0 + side <= 0 or side < 2:
                continue  # (its smoke nowhere in the frame)
            params = dict(base, amount=amount)
            seed = (self.seed * 1000003 + 7919 * (i + 1)) % (2 ** 31)
            self.plumes.append(_Plume((x0, y0, side, side), params, seed, self.P, self.fps, self.w, self.h,
                                      light=light))

    def draw(self, t, canvas, mask):
        for plume in self.plumes:
            plume.draw(t, canvas, mask)


def from_steam(raw, w, h):
    """The settings of smoke from a chimney made with the smoke (fx/steam.py, its "chimney" style, from 1.0.60: a
    box placed and perhaps turned in 3D, its mark on the chimney's mouth) as this effect's in a w x h video: one
    chimney where the mark was in the frame, as wide; its smoke as high (its box, mouth widths); its wind, amount and
    look."""
    from . import effects, overlay  # (they import this one)
    p = effects.clean_params("steam", dict(raw, style="cup"))  # (its place, its turn and its sliders as it had them)
    rim = p["rim"] / 100
    rect = effects.sheet_rect("steam", p, w, h)
    hm = overlay.homography(p, rect, w, h)[0]

    def at(u, v):  # a point of the box (pixels from its top left corner) in the frame
        q = hm @ (u, v, 1.0)
        return q[:2] / q[2]
    bw, bh = rect[2], rect[3]
    v = bh - steam.MARK / 2 * rim * bw  # (the mark's middle, its ends)
    mid, left, right = at(bw / 2, v), at(bw / 2 * (1 - rim), v), at(bw / 2 * (1 + rim), v)
    wid = float(np.hypot(*(right - left))) / w

    def share(v):
        return round(min(1.0, max(0.0, float(v))), 4)
    tall = 50 + 50 * math.log(1 / rim / TALL, SIDE) if rim > 0 else 50
    return dict({k: p[k] for k in SHARED}, stacks=[[share(mid[0] / w), share(mid[1] / h), share(wid), MUCH]],
                tall=int(round(min(100.0, max(0.0, tall)))))
