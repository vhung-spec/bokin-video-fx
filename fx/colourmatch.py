""""Khớp màu với cảnh": colour settings that make an outside-file clip sit in the scene under it.

The scene is the project's video where the clip lies (the box around its picture on the frame, a little wider,
inside its own area when one is painted), over a few frames of the clip's time. The clip is its file where it
shows: opaque pixels when laid with Normal, bright ones with Screen and Add (black is nothing there), dark ones
with Multiply, over a few frames of the file. The clip's colour settings (overlay.colour_ops: the sliders the owner
moves) are then chosen so the clip's light, contrast, colourfulness and colour cast go most of the way to the
scene's: all the way looks like a filter laid over it. Only the colour settings change, and they stay editable.
Each setting is fitted on the clip's own pixels through colour_ops itself, so what is measured is what renders.
"""
import numpy as np
from PIL import ImageOps

from . import effects, engine, overlay, video

SIZE = 256  # pixels wide the frames are measured at
FRAMES = 6  # frames of the video and of the file measured, spread over them
SAMPLES = 60_000  # pixels of the clip the settings are fitted on (a fixed choice: the same answer each time)
STRENGTH = 0.8  # how far the clip goes toward the scene's numbers
MARGIN = 0.25  # the scene measured around the clip: its box grown by this much of its size on each side
LUMA = np.array(overlay.LUMA, np.float32)
TINT = next(d["value"] for d in overlay.SCHEMA["params"] if d["key"] == "tint")  # the colour laid over by default


class MatchError(Exception):
    """A clip whose colours cannot be measured (the message is for the owner)."""


def _size(w, h, width=SIZE):
    """A small even size of the same shape."""
    sw = max(2, min(width, w) // 2 * 2)
    return sw, max(2, round(h * sw / w / 2) * 2)


def _stats(rgb, weight):
    """(mean colour, mean light, spread of light, colourfulness) of weighted pixels (n, 3) in 0..1. The
    colourfulness is how far colours stray from the pixels' own cast (a picture all in shades of one tint has
    none): the cast is matched on its own, and an orange evening is not a reason to make a clip garish."""
    total = float(weight.sum())
    mean = (rgb * weight[:, None]).sum(0) / total
    lum = rgb @ LUMA
    mu = float((lum * weight).sum() / total)
    sd = float(np.sqrt(((lum - mu) ** 2 * weight).sum() / total))
    cast = mean / max(float(mean @ LUMA), 1e-3)
    chroma = float((np.sqrt(((rgb - lum[:, None] * cast) ** 2).sum(1)) * weight).sum() / total)
    return mean, mu, sd, chroma


def _scene_frames(folder, project, start, end):
    """A few frames of the project's video during frames start..end-1 of the output (the video loops)."""
    info = project["video"]
    loop = info["frames"]
    first, length = start % loop, end - start
    if length >= loop or first + length > loop:  # the whole loop (or it goes round): any of its frames
        first, length = 0, loop
    step = max(1, length // FRAMES)
    with video.Frames(engine.input_path(folder, project), info, _size(info["w"], info["h"]), start=first,
                      count=min(FRAMES, length), step=step) as frames:
        out = [f.astype(np.float32) * (1 / 255.0) for f in frames]
    if not out:
        raise MatchError("Không đọc được video của dự án.")
    return out


def scene(folder, project, layer, params, info):
    """The scene's pixels under the clip (n, 3) in 0..1 and their weights."""
    frames = _scene_frames(folder, project, layer["start"], max(layer["end"], layer["start"] + 1))
    h, w = frames[0].shape[:2]
    pts = np.array(overlay.quad(params, info, w, h), np.float64)
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    gx, gy = (x1 - x0) * MARGIN, (y1 - y0) * MARGIN
    x0, y0 = max(0, int(x0 - gx)), max(0, int(y0 - gy))
    x1, y1 = min(w, int(np.ceil(x1 + gx))), min(h, int(np.ceil(y1 + gy)))
    weight = np.zeros((h, w), np.float32)
    if x0 < x1 and y0 < y1:
        weight[y0:y1, x0:x1] = 1.0
    else:  # (laid outside the frame: the whole picture)
        weight[:] = 1.0
    area = engine.load_mask(folder, layer["id"], (w, h), 0, project["video"]["h"])
    if area is not None and float((weight * area).sum()) >= 16:  # inside its own area
        weight *= area
    rgb = np.concatenate([f.reshape(-1, 3) for f in frames])
    return rgb, np.tile(weight.reshape(-1), len(frames))


def _clip_frames(path, info):
    """A few frames of the clip's file (RGBA, 0..1)."""
    if info.get("image"):
        with overlay.open_picture(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGBA")
            im.thumbnail((SIZE, SIZE))
            return [np.asarray(im, np.float32) * (1 / 255.0)]
    alpha = bool(info.get("alpha"))
    frames = []
    with video.Frames(path, info, _size(info["w"], info["h"]), count=FRAMES,
                      step=max(1, info["frames"] // FRAMES), alpha=alpha) as reader:
        for f in reader:
            f = f.astype(np.float32) * (1 / 255.0)
            frames.append(f if alpha else np.concatenate([f, np.ones(f.shape[:2] + (1,), np.float32)], 2))
    if not frames:
        raise MatchError("Không đọc được file của clip.")
    return frames


def clip(path, info, blend):
    """The clip's pixels where it shows (n, 3) in 0..1 and their weights."""
    px = np.concatenate([f.reshape(-1, 4) for f in _clip_frames(path, info)])
    rgb, a = px[:, :3], px[:, 3]
    lum = rgb @ LUMA
    if blend in ("screen", "add"):  # light laid over: black adds nothing
        weight = a * np.where(lum > 0.06, lum, 0.0)
    elif blend == "multiply":  # shade laid over: white takes nothing away
        weight = a * np.where(lum < 0.94, 1.0 - lum, 0.0)
    else:
        weight = np.where(a > 0.5, a, 0.0)
    keep = weight > 0
    rgb, weight = rgb[keep], weight[keep]
    if weight.sum() < 16:
        raise MatchError("Không đo được màu của clip này: phần hiện ra gần như không có gì.")
    if len(weight) > SAMPLES:
        pick = np.random.default_rng(0).choice(len(weight), SAMPLES, replace=False)
        rgb, weight = rgb[pick], weight[pick]
    return rgb.astype(np.float32), weight.astype(np.float32)


def _apply(rgb, p):
    ops = overlay.colour_ops(p)
    if ops is None:
        return rgb
    curve, mat = ops
    out = np.interp(rgb, np.linspace(0.0, 1.0, overlay.TONE_POINTS), curve).astype(np.float32)
    return np.clip(out @ mat.T.astype(np.float32), 0.0, 1.0)


def _solve(f, lo, hi, target, steps=24):
    """The x in lo..hi where the increasing f(x) reaches target (an end when it never does)."""
    if f(lo) >= target:
        return lo
    if f(hi) <= target:
        return hi
    for _ in range(steps):
        mid = (lo + hi) / 2
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _toward(have, want):
    """STRENGTH of the way from `have` to `want`, on a ratio scale (both above 0)."""
    return have * (want / have) ** STRENGTH


def settings(rgb, weight, scene_rgb, scene_weight, blend):
    """The colour settings (overlay.SCHEMA keys and steps) for clip pixels to sit in the scene's."""
    s_mean, s_mu, s_sd, s_chroma = _stats(scene_rgb, scene_weight)
    c_mean, c_mu, c_sd, c_chroma = _stats(rgb, weight)
    s_mu, c_mu = max(s_mu, 0.02), max(c_mu, 0.02)
    p = {"bright": 0, "contrast": 0, "sat": 0, "hue": 0, "temp": 0, "tint_amt": 0, "tint": "#ffffff"}

    def stat(q, which):
        return _stats(_apply(rgb, {**p, **q}), weight)[which]

    def light_target():
        if blend in ("screen", "add"):  # light laid over shows against the scene: only ever darker
            return min(c_mu, _toward(c_mu, min(max(1.3 * s_mu, 0.2), 0.9)))
        return _toward(c_mu, s_mu)

    def fit_light():
        p["bright"] = round(100 * _solve(lambda b: stat({"bright": 100 * b}, 1), -1.0, 1.0, light_target()))

    if blend not in ("screen", "add"):  # harsher than the scene (a sharp clip on a soft picture): flatter. Never
        # harsher: a thing is less varied than the stretch of scene around it, and that is no reason to add any
        want = _toward(max(c_sd / c_mu, 1e-3), max(s_sd / s_mu, 1e-3))
        p["contrast"] = round(100 * _solve(lambda c: (lambda st: st[2] / max(st[1], 1e-3))(
            _stats(_apply(rgb, {**p, "contrast": 100 * c}), weight)), -0.3, 0.0, want))
    fit_light()
    if c_chroma / c_mu > 0.01:  # (a grey clip has no colour to turn up or down)
        cur = stat({}, 3) / max(stat({}, 1), 1e-3)
        want = _toward(max(cur, 1e-3), max(s_chroma / s_mu, 1e-3))
        p["sat"] = round(100 * _solve(lambda s: (lambda st: st[3] / max(st[1], 1e-3))(
            _stats(_apply(rgb, {**p, "sat": 100 * s}), weight)), -0.4, 0.4, want))
    # the colour cast: warmer or cooler, then a little of the scene's own colour laid over (both straight lines
    # after the tone curve, so the clip's mean colour after them is the matrix times its mean after the curve)
    cast = s_mean / max(float(s_mean @ LUMA), 1e-3)
    tint = np.clip(cast / max(float(cast.max()), 1e-3), 0.0, 1.0)
    p["tint"] = "#" + "".join(f"{round(v * 255):02x}" for v in tint)
    curve_mean = None
    base = {**p}

    def cast_of(q):
        nonlocal curve_mean
        ops = overlay.colour_ops({**base, **q})
        mat = np.eye(3) if ops is None else ops[1]
        if curve_mean is None:
            c = overlay.colour_ops({**base, "hue": 1e-6})[0]  # (the same curve, whatever the matrix)
            curve_mean = np.interp(rgb, np.linspace(0.0, 1.0, overlay.TONE_POINTS), c).T @ weight / weight.sum()
        m = mat @ curve_mean
        return m / max(float(m @ LUMA), 1e-3)

    have = cast_of({})
    want = have + STRENGTH * (cast - have)
    best = None
    for w in range(-60, 61, 5):
        for a in range(0, 41, 2):
            got = cast_of({"temp": w, "tint_amt": a})
            cost = float(((got - want) ** 2).sum()) + 1e-6 * (abs(w) + a)  # (the least change among equals)
            if best is None or cost < best[0]:
                best = (cost, w, a)
    p["temp"], p["tint_amt"] = best[1], best[2]
    fit_light()  # a colour laid over darkens a little: the light again, with it
    if not p["tint_amt"]:
        p["tint"] = TINT
    return p


def match(folder, project, layer):
    """The colour settings for a file clip of the project to sit in its scene."""
    if layer.get("kind") != "file":
        raise MatchError("Chỉ khớp màu được cho clip video / ảnh ngoài.")
    params = effects.clean_params("file", layer.get("params"))
    path, info = engine.clip_file(folder, {**layer, "params": {**params, "reverse": False}})
    if path is None:
        raise MatchError("Không tìm thấy file của clip này.")
    rgb, weight = clip(path, info, params["blend"])
    scene_rgb, scene_weight = scene(folder, project, layer, params, info)
    if scene_weight.sum() < 16:
        raise MatchError("Không đo được màu của cảnh ở chỗ clip này.")
    return settings(rgb, weight, scene_rgb, scene_weight, params["blend"])
