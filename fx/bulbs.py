"""The lights already in a video under a string-lights clip's box ("Theo đèn có sẵn trong video", lights.py): small
bright spots, each brighter than what is round it (fairy lights on a tree, LEDs on a string), found in a few frames of
the video (the brightest of them: a light that blinks counts) inside the box, and inside the clip's area when one is
painted.

A spot is a local maximum of brightness well above its surroundings (the picture minus a blur of it about 6 px wide
at 1080 lines), not too faint, and at least a few pixels from a brighter one. Its colour is the colour round its
middle (a light's middle is often clipped white; its glow shows its colour). They are kept in the clip's settings as
places in the box (0..1), how bright (0..1) and their colour, so the effect needs nothing else to draw them.
"""
import numpy as np

from . import effects, engine, lights, video

FRAMES = 4  # frames of the video looked at (spread over the loop)
SIZE = 1080  # rows they are looked at, at most


class FindError(Exception):
    pass


def _frames(folder, project):
    info = project["video"]
    h = min(SIZE, info["h"])
    w = max(2, round(info["w"] * h / info["h"]))
    loop = info["frames"]
    step = max(1, loop // FRAMES)
    with video.Frames(engine.input_path(folder, project), info, (w, h), start=0, count=min(FRAMES, loop),
                      step=step) as frames:
        out = [f for f in frames]
    if not out:
        raise FindError("Không đọc được video của dự án.")
    return np.max(np.stack(out), axis=0).astype(np.float32) * (1 / 255.0)


def spots(rgb, rect, area=None, most=lights.SPOTS):
    """The lights in `rgb` (h, w, 3: 0..1) inside rect (x0, y0, x1, y1: pixels) and the area (h, w: 0..1, or None):
    [[x, y, how bright, r, g, b], ...] with x, y in the rect (0..1), the brightest first."""
    h, w = rgb.shape[:2]
    px = h / 1080
    x0, y0 = max(0, int(rect[0])), max(0, int(rect[1]))
    x1, y1 = min(w, int(np.ceil(rect[2]))), min(h, int(np.ceil(rect[3])))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    pad = int(np.ceil(12 * px)) + 3
    cx0, cy0, cx1, cy1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    crop = rgb[cy0:cy1, cx0:cx1]
    luma = crop @ effects.LUMA
    above = luma - effects.blur(luma, max(1.5, 6 * px))
    # a local maximum (of a 5 x 5 square) well above its surroundings, not too faint; on a flat top (a clipped
    # light) only its first pixel
    peak = np.ones(luma.shape, bool)
    padded = np.pad(luma, 2, mode="edge")
    for dy in range(5):
        for dx in range(5):
            if dx != 2 or dy != 2:
                other = padded[dy:dy + luma.shape[0], dx:dx + luma.shape[1]]
                peak &= luma > other if (dy, dx) < (2, 2) else luma >= other
    peak &= (above > 0.1) & (luma > 0.4)
    ys, xs = np.nonzero(peak)
    fx, fy = xs + cx0, ys + cy0
    inside = (fx >= x0) & (fx < x1) & (fy >= y0) & (fy < y1)
    if area is not None:
        inside &= area[np.clip(fy, 0, h - 1), np.clip(fx, 0, w - 1)] > 0.5
    ys, xs, fx, fy = ys[inside], xs[inside], fx[inside], fy[inside]
    strength = above[ys, xs] + 0.5 * luma[ys, xs]
    order = np.argsort(-strength, kind="stable")
    apart = max(3.0, 4.5 * px)  # (a light's own glow has no second light in it)
    taken, cells = [], {}
    for i in order:
        cell = (int(fx[i] // apart), int(fy[i] // apart))
        near = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in cells.get((cell[0] + dx, cell[1] + dy), ())]
        if any((fx[j] - fx[i]) ** 2 + (fy[j] - fy[i]) ** 2 < apart * apart for j in near):
            continue
        cells.setdefault(cell, []).append(i)
        taken.append(i)
        if len(taken) >= most:
            break
    if not taken:
        return []
    taken = np.array(taken)
    # its colour: round its middle (2 to 5 px out at 1080 lines), the most colourful and bright of those
    ring = [(dx, dy) for dx in range(-5, 6) for dy in range(-5, 6) if 2 <= (dx * dx + dy * dy) ** 0.5 <= 5.5]
    k = max(1.0, px)
    colours = np.zeros((len(taken), 3), np.float32)
    weight = np.zeros(len(taken), np.float32)
    for dx, dy in ring:
        sx = np.clip(xs[taken] + int(round(dx * k)), 0, crop.shape[1] - 1)
        sy = np.clip(ys[taken] + int(round(dy * k)), 0, crop.shape[0] - 1)
        c = crop[sy, sx]
        wt = (c.max(axis=1) - c.min(axis=1) + 0.05) * c.max(axis=1) + 1e-4
        colours += c * wt[:, None]
        weight += wt
    colours /= weight[:, None]
    colours /= np.maximum(colours.max(axis=1, keepdims=True), 1e-3)
    bright = np.clip(strength[taken] / max(float(strength[taken].max()), 1e-6), 0, 1)
    u = (fx[taken] + 0.5 - rect[0]) / (rect[2] - rect[0])
    v = (fy[taken] + 0.5 - rect[1]) / (rect[3] - rect[1])
    return [[round(float(a), 4) for a in row] for row in np.column_stack([u, v, bright, colours])]


def find(folder, project, layer):
    """The lights in the project's video under a string-lights clip's box (and in its area, if painted)."""
    info = project["video"]
    rgb = _frames(folder, project)
    h, w = rgb.shape[:2]
    x0, y0, bw, bh = effects.box_of("lights", layer.get("params"), w, h)
    area = engine.load_mask(folder, layer["id"], (w, h), 0, info["h"])
    if area is not None and float(area.max()) <= 0.5:
        area = None  # (nothing painted: the whole box)
    return spots(rgb, (x0, y0, x0 + bw, y0 + bh), area)
