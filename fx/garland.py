"""String lights drawn by hand anywhere on the picture ("Dây đèn tự vẽ"): along the eaves and the roof, round a door
or a window, over a railing, up a tree (a full-frame effect drawn as light, laid with Screen; the bulbs, their
colours and their twinkling are the string lights' own: built on fx/lights.py; see effects.py for how every effect
loops).

The owner draws each strand on the picture (the page: a click is a point, a press and drag a stretch drawn by hand),
kept in the clip's settings as places in the frame ("strands": x, y as shares of its width and height, so the
preview clip and the video, any size of the same shape, show the same). Between two clicked points the wire hangs
down, each strand as much as its own "sag" (a level span the most: a quarter of its width in its middle at 1; an
upright one not at all); a stretch drawn by hand follows the hand. Bulbs sit along every strand at one spacing
("gap", as on a real string: a longer strand has more of them), the first and the last half a step in from its ends;
more than CAP in all and they are spread further apart. The page lays the guides of the strands and their bulbs with
the same rules (index.html: wirePath, wireBulbs).
"""
import math

import numpy as np

from . import lights
from .effects import _num, _points_param

START, HAND, CLICK = 1.0, 0.5, 0.0  # a point's kind (its third number): where a strand starts, drawn by hand, clicked
SEG = 24  # pieces of a span hanging between two clicked points
POINTS = 2400  # points of strands at most (x, y, kind, the strand's sag on its first point)
SPARSE, DENSE = 0.06, 0.008  # the bulbs' spacing at "gap" 0 and 100, frame heights (65 and 9 pixels at 1080 lines)
SIZE = (0.0015, 0.009)  # a bulb's core (frame heights) at "size" 0 and 100: as the string lights' in their box

CAP = 1200  # bulbs at most

_LIGHTS = {d["key"]: d for d in lights.SCHEMA["params"]}
SCHEMA = {
    "label": "Dây đèn tự vẽ",
    "bulbs_most": CAP,  # (the page lays the guides of the bulbs as the renderer does)
    "note": "Vẽ dây đèn ở bất cứ chỗ nào trên hình: dọc mái nhà, quanh cửa sổ, vắt qua lan can, cành cây. Cách vẽ ở "
            "khung ngay dưới. Bấm Suốt video để đèn sáng cả video.",
    "params": [
        _points_param("strands", "Dây đèn", 4, POINTS, group="Dây đèn"),
        _num("gap", "Số bóng (thưa ← → dày)", 50, group="Dây đèn"),
        dict(_LIGHTS["jitter"], value=10),
        _LIGHTS["bulb"], _LIGHTS["mode"], lights.palette_choice(),
        _LIGHTS["size"], _LIGHTS["glow"], _LIGHTS["rays"], _LIGHTS["brightness"],
        _LIGHTS["speed"], _LIGHTS["depth"], _LIGHTS["color"]]}


def gap_of(v):
    """The bulbs' spacing (frame heights) at the setting "gap" (0: sparse .. 100: dense)."""
    return math.exp(math.log(SPARSE) + (math.log(DENSE) - math.log(SPARSE)) * v / 100)


def strands_of(points):
    """The strands in a list of points ([x, y, kind, sag]: x, y shares of the frame's width and height; kind START
    where a strand begins, its sag the fourth number, 0..1; CLICK a clicked point; HAND a point of a stretch drawn by
    hand): for each, its points [(x, y, by hand)] and its sag. A point where the one before is: left out."""
    out = []
    for x, y, kind, sag in points:
        if kind >= 0.75 or not out:
            out.append(([], sag if kind >= 0.75 else 0.0))
        pts = out[-1][0]
        if not pts or (x, y) != pts[-1][:2]:
            pts.append((x, y, 0.25 < kind < 0.75))
    return out


def paths(points, aspect):
    """Each strand (two places at least) as a polyline (m, 2): x, y in frame heights from its top left corner
    (`aspect`: the frame's width in heights). Between two clicked points the wire hangs down in SEG pieces, a level
    span the most (sag 1: a quarter of its width in its middle), an upright one not at all; a stretch drawn by hand
    goes straight from point to point."""
    out = []
    t = np.arange(1, SEG + 1) / SEG
    for pts, sag in strands_of(points):
        if len(pts) < 2:
            continue
        xy = [(pts[0][0] * aspect, pts[0][1])]
        for (ax, ay, _), (bx, by, hand) in zip(pts[:-1], pts[1:]):
            ax, bx = ax * aspect, bx * aspect
            if hand or sag <= 0 or ax == bx:
                xy.append((bx, by))
            else:
                droop = sag * abs(bx - ax) * t * (1 - t)
                xy.extend(zip(ax + (bx - ax) * t, ay + (by - ay) * t + droop))
        out.append(np.array(xy, np.float64))
    return out


def bulbs_on(lines, gap, most):
    """Bulbs along polylines, `gap` apart on every strand (a strand gets round(its length / gap) of them, one at
    least), the first and the last half a step in from its ends; more than `most` in all and the spacing grows until
    they fit. (x, y, strand, along, step): their places, the strand of each, its number along it, and its strand's
    spacing."""
    cum = [np.r_[0.0, np.cumsum(np.hypot(*np.diff(line, axis=0).T))] for line in lines]
    lengths = np.array([c[-1] for c in cum], np.float64)

    def counts(g):
        return np.maximum(1, np.floor(lengths / g + 0.5)).astype(int)

    n = counts(gap)
    while n.sum() > most:
        gap *= max(1.02, n.sum() / most)
        n = counts(gap)
    if not len(lines):
        empty = np.zeros(0)
        return empty, empty, np.zeros(0, int), np.zeros(0, int), empty
    xs, ys, strand, along, step = [], [], [], [], []
    for i, (line, c, k) in enumerate(zip(lines, cum, n)):
        s = (np.arange(k) + 0.5) * c[-1] / k
        xs.append(np.interp(s, c, line[:, 0]))
        ys.append(np.interp(s, c, line[:, 1]))
        strand.append(np.full(k, i))
        along.append(np.arange(k))
        step.append(np.full(k, c[-1] / k))
    return tuple(np.concatenate(v) for v in (xs, ys, strand, along, step))


def from_lights(raw, w, h):
    """The settings of string lights drawn by hand with an older version (fx/lights.py, layout "draw": points kept in
    its box, x and y in box heights, the third 1 where a strand starts; one "sag" for all of them and "bulbs" bulbs
    in all) as this effect's in a w x h video: the strands where they were on the picture, each with that sag, the
    spacing that lays about as many bulbs on them, the bulbs about as big, and the rest of its look."""
    def num(key, value, lo, hi):
        try:
            v = float(raw.get(key, value))
        except (TypeError, ValueError, OverflowError):
            return value
        return min(hi, max(lo, v)) if math.isfinite(v) else value

    def share(v):
        return round(min(1.0, max(0.0, v)), 4)

    from .effects import place_box  # (defined after the effects are)
    place = {k: num(k, v, lo, hi) for k, v, lo, hi in (("scale", 60, 1, 300), ("x", 0, -100, 100),
                                                        ("y", 0, -100, 100))}
    bx, by, _, bh = place_box(lights.Lights.BOX, place, w, h)
    sag = num("sag", 35, 0, 100) / 100
    points = []
    for item in raw.get("strands") if isinstance(raw.get("strands"), list) else ():
        try:
            u, v, new = (float(q) for q in item)
        except (TypeError, ValueError, OverflowError):
            continue
        if not all(math.isfinite(q) for q in (u, v, new)):
            continue
        start = new >= 0.5 or not points
        points.append([share((bx + u * bh) / w), share((by + v * bh) / h), START if start else CLICK,
                       round(sag, 4) if start else 0.0])
    length = sum(float(np.hypot(*np.diff(line, axis=0).T).sum()) for line in paths(points, w / h))
    gap = length / num("bulbs", 60, 2, 600) if length > 0 else gap_of(50)
    dense = (math.log(gap) - math.log(SPARSE)) / (math.log(DENSE) - math.log(SPARSE)) * 100
    core = (SIZE[0] + (SIZE[1] - SIZE[0]) * num("size", 40, 0, 100) / 100) * bh / h  # (its box was the frame's
    # height times bh / h)
    keep = {k: raw[k] for k in ("bulb", "mode", "palette", "glow", "rays", "brightness", "speed", "depth", "color",
                                "jitter") if k in raw}
    return dict(keep, strands=points, gap=int(round(min(100.0, max(0.0, dense)))),
                size=int(round(min(100.0, max(0.0, (core - SIZE[0]) / (SIZE[1] - SIZE[0]) * 100)))))


class Garland(lights.Lights):
    BOX = None  # (the whole frame: drawn anywhere on the picture)
    TURNS = False  # (drawn on the picture itself, where its things are)
    CAP = CAP
    PREVIEW = 1280  # (small bulbs over the whole picture: a sharper preview clip)

    def _bulbs(self):
        lines = paths(self.p["strands"], self.aspect)
        x, y, strand, along, step = bulbs_on(lines, gap_of(self.p["gap"]), self.CAP)
        return np.stack([x, y, np.zeros_like(x)], 1), strand, along, step

    def _seen(self, pts, u):
        return pts[:, 0], pts[:, 1], np.ones(len(pts))

    def _frame(self):
        return 0.0, 0.0, float(self.h)
