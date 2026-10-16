"""Clips made from outside files, as in After Effects: a video (an effect on black, or one with transparency)
or a picture (PNG, JPG), laid over the video with a blend mode, opacity, size and position.

A clip over the whole video loops with it (timeline()): frame t shows the file at start + fade + t/fps; over the
last `fade` seconds it dissolves into the file at start .. start + fade, which is exactly what frame 0 shows next,
so the seam is a soft dissolve instead of a jump. A shorter clip plays the file from `start` (plus what was trimmed
off its head) for as long as it shows (playing()); it fades in and out, so its ends need not join. A file shorter
than it needs is played over and over, each repeat dissolving into the next.
"""
import math
import os

import numpy as np

from PIL import Image, ImageChops, ImageOps

from . import video

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_TYPES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
MAX_IMAGE_PIXELS = 40_000_000
PICTURE_FORMATS = ("PNG", "JPEG", "WEBP", "BMP")  # IMAGE_TYPES; a phone's JPEG may open as MPO, which is fine

BLENDS = [("normal", "Bình thường (Normal)"), ("screen", "Screen (hiệu ứng trên nền đen)"),
          ("add", "Add (cộng sáng)"), ("multiply", "Multiply (làm tối)")]
SCHEMA = {"label": "Video / ảnh ngoài", "params": [
    {"key": "blend", "label": "Chế độ hoà trộn", "type": "choice", "value": "screen",
     "options": [{"value": v, "label": t} for v, t in BLENDS], "presets": {}},
    {"key": "opacity", "label": "Độ đậm", "type": "range", "min": 0, "max": 100, "value": 100},
    {"key": "fit", "label": "Khớp khung", "type": "choice", "value": "cover", "presets": {},
     "options": [{"value": "cover", "label": "Phủ kín khung"}, {"value": "contain", "label": "Vừa trong khung"}]},
    {"key": "scale", "label": "Cỡ (%)", "type": "range", "min": 10, "max": 300, "step": 0.1, "value": 100},
    {"key": "x", "label": "Dịch ngang (%)", "type": "range", "min": -100, "max": 100, "step": 0.05, "value": 0},
    {"key": "y", "label": "Dịch dọc (%)", "type": "range", "min": -100, "max": 100, "step": 0.05, "value": 0},
    {"key": "flip_h", "label": "Lật ngang", "type": "toggle", "value": False},
    {"key": "flip_v", "label": "Lật dọc", "type": "toggle", "value": False},
    # turned about its middle in 3D, as a CSS "perspective(d) rotateX rotateY rotateZ" does (homography(); the page
    # lays its picture with that transform). All three 0: laid exactly as before. X and Y stop short of edge-on.
    {"key": "rot_x", "label": "Nghiêng trước / sau (°)", "type": "range", "min": -80, "max": 80, "step": 0.1,
     "value": 0, "group": "Xoay 3D"},
    {"key": "rot_y", "label": "Xoay trái / phải (°)", "type": "range", "min": -80, "max": 80, "step": 0.1,
     "value": 0, "group": "Xoay 3D"},
    {"key": "rot_z", "label": "Xoay phẳng (°)", "type": "range", "min": -180, "max": 180, "step": 0.1, "value": 0,
     "group": "Xoay 3D"},
    {"key": "persp", "label": "Độ sâu phối cảnh", "type": "range", "min": 0, "max": 100, "value": 50,
     "group": "Xoay 3D"},
    {"key": "start", "label": "Bắt đầu từ giây thứ (của file)", "type": "range", "min": 0, "max": 600, "step": 0.1,
     "value": 0},
    {"key": "fade", "label": "Hoà mượt khi file lặp lại (giây)", "type": "range", "min": 0, "max": 3, "step": 0.1,
     "value": 1.0},
    {"key": "reverse", "label": "Phát ngược (tua ngược)", "type": "toggle", "value": False},
    # its colours (colour_ops; the page builds the same SVG filter)
    {"key": "bright", "label": "Độ sáng", "type": "range", "min": -100, "max": 100, "value": 0},
    {"key": "contrast", "label": "Độ tương phản", "type": "range", "min": -100, "max": 100, "value": 0},
    {"key": "sat", "label": "Độ đậm màu", "type": "range", "min": -100, "max": 100, "value": 0},
    {"key": "hue", "label": "Đổi màu (xoay vòng màu)", "type": "range", "min": -180, "max": 180, "value": 0},
    {"key": "temp", "label": "Lạnh ↔ ấm", "type": "range", "min": -100, "max": 100, "value": 0},
    {"key": "tint_amt", "label": "Phủ một màu", "type": "range", "min": 0, "max": 100, "value": 0},
    {"key": "tint", "label": "Màu phủ", "type": "color", "value": "#ffb066"},
]}

LUMA = (0.2126, 0.7152, 0.0722)
TONE_POINTS = 33  # the tone curve's points (an SVG feComponentTransfer table: straight lines between them)

ROTATIONS = ("rot_x", "rot_y", "rot_z")
PERSPECTIVE = 0.6  # the viewer's distance at "Độ sâu phối cảnh" 100, in frame diagonals (at 50: twice as far)
NEAR = 1.25  # the viewer stays at least this many times as far as the nearest corner: none at or behind the eye
PAD = 4  # pixels of its edge repeated around a turned picture, so its soft outline is not darkened


def placement(p, info, w, h):
    """(x, y, sw, sh): where a file clip's picture lies in a w x h frame before it is turned, in pixels (floats,
    as the page's placed(); the plain way below rounds them)."""
    k = (max if p["fit"] == "cover" else min)(w / info["w"], h / info["h"]) * p["scale"] / 100
    sw, sh = info["w"] * k, info["h"] * k
    return (w - sw) / 2 + p["x"] / 100 * w, (h - sh) / 2 + p["y"] / 100 * h, sw, sh


def rotated(p):
    """Is the clip turned (in the plane or in 3D)?"""
    return any(p.get(k) for k in ROTATIONS)


def homography(p, rect, w, h):
    """(H, d): H (3x3) takes a point (u, v, 1) of the placed picture (u from its left, 0..sw; v from its top, 0..sh;
    the file's own picture, before any flip) to the frame, in homogeneous coordinates. Exactly the page's CSS
    "perspective(d) rotateX(x) rotateY(y) rotateZ(z) scale(flips)" about the picture's middle (y down, z toward the
    viewer): turned over first, then turned about Z, Y, X, then seen from d pixels away (d None: flat, no
    perspective). d comes from "persp" and the frame's diagonal, and is moved back when a corner would come
    too near (NEAR)."""
    x, y, sw, sh = rect
    a, b, c = (math.radians(p.get(k, 0)) for k in ROTATIONS)
    rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    r = rx @ ry @ rz
    fx, fy = (-1.0 if p.get("flip_h") else 1.0), (-1.0 if p.get("flip_v") else 1.0)
    ex, ey = r[:, 0] * fx, r[:, 1] * fy  # (u, v) -> a point in space, about the middle
    turn = np.column_stack([ex, ey, -(ex * sw + ey * sh) / 2])
    d = None
    if p.get("persp", 0) > 0:
        near = max(turn[2] @ (u, v, 1.0) for u, v in ((0, 0), (sw, 0), (sw, sh), (0, sh)))
        d = max(math.hypot(w, h) * PERSPECTIVE * 100 / p["persp"], NEAR * near)
    q = 1 / d if d else 0.0
    cx, cy = x + sw / 2, y + sh / 2
    view = np.array([[1, 0, -cx * q], [0, 1, -cy * q], [0, 0, -q]])  # then (cx, cy, 1) added
    return view @ turn + np.array([[0, 0, cx], [0, 0, cy], [0, 0, 1.0]]), d


def corners(hm, sw, sh):
    """The picture's corners (top left, top right, bottom right, bottom left) through H, in frame pixels."""
    pts = hm @ np.array([[0, sw, sw, 0], [0, 0, sh, sh], [1.0, 1, 1, 1]])
    return [(float(pts[0, i] / pts[2, i]), float(pts[1, i] / pts[2, i])) for i in range(4)]


def quad(p, info, w, h):
    """Where the corners of a file clip's picture are in a w x h frame (the page's quad(): the same numbers)."""
    rect = placement(p, info, w, h)
    return corners(homography(p, rect, w, h)[0], rect[2], rect[3])


def _clip_polygon(pts, w, h):
    """The part of a convex polygon inside the frame [0, w] x [0, h] (Sutherland-Hodgman)."""
    for axis, lim, keep in ((0, 0, 1), (0, w, -1), (1, 0, 1), (1, h, -1)):
        out = []
        for i, a in enumerate(pts):
            b = pts[(i + 1) % len(pts)]
            ina, inb = (a[axis] - lim) * keep >= 0, (b[axis] - lim) * keep >= 0
            if ina:
                out.append(a)
            if ina != inb:
                f = (lim - a[axis]) / (b[axis] - a[axis])
                out.append((a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])))
        pts = out
        if not pts:
            break
    return pts


def _area(pts):
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))


def coverage(quad_pts, box):
    """How much of each pixel of `box` (x0, y0, x1, y1) the convex quad covers, as an "L" image: 0.5 plus the
    signed distance of the pixel's middle to the nearest edge (a soft, even outline about a pixel wide)."""
    x0, y0, x1, y1 = box
    xs = np.arange(x0, x1, dtype=np.float64) + 0.5
    ys = (np.arange(y0, y1, dtype=np.float64) + 0.5)[:, None]
    sign = 1.0 if _area(quad_pts) > 0 else -1.0
    inside = np.full((y1 - y0, x1 - x0), np.inf, np.float64)
    for (ax, ay), (bx, by) in zip(quad_pts, quad_pts[1:] + quad_pts[:1]):
        ex, ey = bx - ax, by - ay
        n = math.hypot(ex, ey)
        if n < 1e-9:
            continue
        np.minimum(inside, sign * (ex * (ys - ay) - ey * (xs - ax)) / n, out=inside)
    return Image.fromarray((np.clip(inside + 0.5, 0.0, 1.0) * 255 + 0.5).astype(np.uint8), "L")


def colour_ops(p):
    """(tone curve at TONE_POINTS points, 3x3 colour matrix) for a file clip's colour settings, or None when they
    change nothing. The tone curve keeps black black (an effect on black laid with Screen stays clean): contrast
    bends it into an S (or flattens it) through 0 and 1, brightness then multiplies. The matrix: saturation, then
    hue, warmth and a colour laid over, as SVG's feColorMatrix (the page's live preview uses the same numbers)."""
    b, c = p.get("bright", 0) / 100, p.get("contrast", 0) / 100
    s, h = 1 + p.get("sat", 0) / 100, math.radians(p.get("hue", 0))
    w, a = p.get("temp", 0) / 100, p.get("tint_amt", 0) / 100
    if not (b or c or s != 1 or h or w or a):
        return None
    x = np.linspace(0.0, 1.0, TONE_POINTS)
    curve = np.clip((x + c * (x * x * (3 - 2 * x) - x)) * 2.0 ** b, 0.0, 1.0)
    sat = np.array([[0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
                    [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
                    [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s]])
    co, si = math.cos(h), math.sin(h)
    hue = np.array([[0.213 + co * 0.787 - si * 0.213, 0.715 - co * 0.715 - si * 0.715, 0.072 - co * 0.072 + si * 0.928],
                    [0.213 - co * 0.213 + si * 0.143, 0.715 + co * 0.285 + si * 0.140, 0.072 - co * 0.072 - si * 0.283],
                    [0.213 - co * 0.213 - si * 0.787, 0.715 - co * 0.715 + si * 0.715, 0.072 + co * 0.928 + si * 0.072]])
    warm = np.diag([1 + 0.2 * w, 1 + 0.04 * w, 1 - 0.2 * w])
    t = np.array([int(p.get("tint", "#ffb066")[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    tint = (1 - a) * np.eye(3) + a * np.outer(t, LUMA)  # white becomes the colour, black stays black
    return curve, tint @ warm @ hue @ sat


def coloured(im, ops):
    """`im` (RGB or RGBA Pillow image) with colour_ops applied to its colour (not to its transparency)."""
    curve, mat = ops
    a = np.asarray(im, np.float32) * (1 / 255.0)
    rgb = np.interp(a[..., :3], np.linspace(0.0, 1.0, TONE_POINTS), curve).astype(np.float32)
    rgb = np.clip(rgb @ mat.T.astype(np.float32), 0.0, 1.0)
    out = (np.concatenate([rgb, a[..., 3:]], axis=2) if a.shape[2] == 4 else rgb) * 255.0 + 0.5
    return Image.fromarray(out.astype(np.uint8), im.mode)


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_TYPES


def is_video(name):
    return os.path.splitext(name)[1].lower() in VIDEO_TYPES


def open_picture(path):
    """An uploaded picture, opened only by Pillow's own decoders of picture formats (never EPS/Ghostscript). GIF,
    TIFF and AVIF too: older versions kept such files under a .png/.jpg name, and they must still open."""
    Image.init()
    return Image.open(path, formats=[f for f in PICTURE_FORMATS + ("GIF", "TIFF", "AVIF") if f in Image.OPEN])


def probe_file(path):
    """What a file to lay over the loop is: like video.probe, with image=True for a picture."""
    if is_image(path):
        try:
            with open_picture(path) as im:
                if im.width * im.height > MAX_IMAGE_PIXELS:
                    raise video.VideoError("Ảnh quá lớn.")
                im = ImageOps.exif_transpose(im)
                alpha = im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info)
                return {"image": True, "w": im.width, "h": im.height, "alpha": alpha, "frames": 1, "seconds": 0}
        except video.VideoError:
            raise
        except Exception as e:  # noqa: BLE001 - any broken or unknown picture
            raise video.VideoError("Không đọc được ảnh này.") from e
    info = video.probe(path, overlay=True)
    if info["codec"] == "gif":  # ffmpeg decodes every GIF with an alpha channel: is any pixel see-through?
        with video.Frames(path, info, alpha=True) as frames:
            info["alpha"] = any(int(f[..., 3].min()) < 255 for f in frames)
    info["image"] = False
    return info


def _plan(seconds, start, fade, need, fps):
    """How the file from `start` on fills `need` seconds: (start, step, join). Played once if it is long enough
    (step None); else over and over, each repeat `step` seconds after the last and dissolving into it over
    `join` seconds. step 0: a file of a single frame there, shown still."""
    start = min(max(0.0, start), max(0.0, seconds - 1.0 / fps))
    avail = seconds - start
    if avail >= need - 0.5 / fps:  # (a file's length is kept to the millisecond: a clip as long as it plays it once)
        return start, None, 0.0
    join = min(max(0.0, fade), avail / 3)
    return start, max(0.0, avail - join), join


def _played(e, start, step, join):
    """[(time in the file, weight)] at `e` seconds into the file played from `start` as _plan said."""
    if step is None:
        return [(start + e, 1.0)]
    if step <= 0:
        return [(start, 1.0)]
    j = math.floor(e / step + 1e-9)
    local = max(0.0, e - j * step)  # the 1e-9 above can leave -1e-16 here
    if j > 0 and join > 0 and local < join:
        w = local / join
        return [(start + local, w), (start + local + step, 1.0 - w)]
    return [(start + local, 1.0)]


def timeline(t, fps, frames, seconds, start, fade):
    """[(time in the file, weight), ...] that make frame t of a loop of `frames` frames at `fps`, from a file of
    `seconds` seconds. Weights add up to 1."""
    loop = frames / fps
    fade = max(0.0, min(fade, loop / 2))
    start, step, join = _plan(seconds, start, fade, loop + fade, fps)
    if step is not None and step <= 0:
        return [(start, 1.0)]
    s = t / fps
    if fade <= 0:
        parts = _played(s, start, step, join)
    elif s < loop - fade:
        parts = _played(fade + s, start, step, join)
    else:
        w = (s - (loop - fade)) / fade
        parts = [(ft, pw * (1 - w)) for ft, pw in _played(fade + s, start, step, join)] + \
                [(ft, pw * w) for ft, pw in _played(s - (loop - fade), start, step, join)]
    return [(ft, w) for ft, w in parts if w > 1e-4]


def playing(e, fps, need, seconds, start, fade):
    """[(time in the file, weight), ...] `e` seconds into a clip that plays the file from `start` for `need`
    seconds (a clip shorter than the whole video: it fades in and out, so its ends need not join). Before the
    clip's start or past its run (a crossfade with the clip before or after), the file goes round."""
    start, step, join = _plan(seconds, start, fade, need, fps)
    if step is None:
        avail = seconds - start
        if avail > 0 and not 0 <= e < avail:
            e %= avail
    elif step > 0 and e < 0:
        e %= step
    return [(ft, w) for ft, w in _played(max(0.0, e), start, step, join) if w > 1e-4]


class Cursors:
    """Frames of a video by number, read in order by a few ffmpeg readers (no file of decoded frames):
    a loop needs a run of frames from one place, and near the end another run from the start."""

    def __init__(self, path, info, size, alpha, src=None):
        self.path, self.info, self.size, self.alpha, self.src = path, info, size, alpha, src
        self.open = []  # [reader, iterator, next frame number, last (number, frame)]

    def get(self, i):
        i = max(0, min(int(i), self.info["frames"] - 1))
        for c in self.open:
            if c[3] and c[3][0] == i:
                return c[3][1]
        ahead = [c for c in self.open if c[2] <= i <= c[2] + 48]
        c = max(ahead, key=lambda c: c[2]) if ahead else None
        if c is None:
            if len(self.open) >= 4:
                self.open.pop(0)[0].close()
            # by frame number, as probe() counted them: seeking by time misses when streams start at different
            # times, frames are uneven (GIFs, phone videos) or the file's clock does not start at 0
            reader = video.Frames(self.path, self.info, self.size, alpha=self.alpha, start=i, src=self.src)
            c = [reader, iter(reader), i, None]
            self.open.append(c)
        while c[2] <= i:
            frame = next(c[1], None)
            if frame is None:  # the file ended early: keep showing its last frame
                c[2] = i + 1
                if c[3] is None:
                    raise video.VideoError("Không đọc được video của lớp này.")
                c[3] = (i, c[3][1])
                break
            c[3] = (c[2], frame)
            c[2] += 1
        self.open.remove(c)
        self.open.append(c)  # most recently used last
        return c[3][1]

    def close(self):
        for c in self.open:
            c[0].close()
        self.open = []


class Overlay:
    """One clip from a file, laid over w x h frames of a video `frames` long at `fps`.

    Works on 8-bit Pillow images (the blend modes and masks are Pillow's C routines: fast, and exact where a mask
    is fully on or off). `weight` (an "L" image of the whole frame, or None for "everywhere, fully") already
    includes the clip's area, its foreground and its opacity. `span` (a timeline.Span) is when the clip shows;
    None or a full span: the whole video, the file looping with it."""

    def __init__(self, path, info, params, frames, fps, w, h, weight=None, span=None):
        self.info, self.p = info, params
        self.frames, self.fps = int(frames), float(fps)
        self.span = None if span is None or span.full else span
        self.warp = None
        if rotated(params):  # turned: laid onto its quad (the plain way below stays exactly as it was)
            self._setup_turned(path, info, params, w, h, weight)
            return
        k = (max if params["fit"] == "cover" else min)(w / info["w"], h / info["h"]) * params["scale"] / 100
        sw, sh = max(2, round(info["w"] * k)), max(2, round(info["h"] * k))
        ox = round((w - sw) / 2 + params["x"] / 100 * w)
        oy = round((h - sh) / 2 + params["y"] / 100 * h)
        x0, y0, x1, y1 = max(ox, 0), max(oy, 0), min(ox + sw, w), min(oy + sh, h)
        self.empty = x0 >= x1 or y0 >= y1
        self.box, self.full = (x0, y0, x1, y1), (x0, y0, x1, y1) == (0, 0, w, h)
        self.crop = (x0 - ox, y0 - oy, x1 - ox, y1 - oy)  # the part of the placed file inside the frame
        self.whole = self.crop == (0, 0, sw, sh)
        self.weight = None if weight is None or self.empty else (weight if self.full else weight.crop(self.box))
        self.still, self.cursors = None, None
        self.colours = colour_ops(params)
        if self.empty:
            return
        kx, ky = sw / info["w"], sh / info["h"]
        self.flips = [m for m, on in ((Image.Transpose.FLIP_LEFT_RIGHT, params.get("flip_h")),
                                      (Image.Transpose.FLIP_TOP_BOTTOM, params.get("flip_v"))) if on]
        cx0, cy0, cx1, cy1 = self.crop
        if params.get("flip_h"):  # turned over, the part inside the frame comes from the other side of the file
            cx0, cx1 = sw - cx1, sw - cx0
        if params.get("flip_v"):
            cy0, cy1 = sh - cy1, sh - cy0
        self.crop = (cx0, cy0, cx1, cy1)
        if info.get("image"):  # only the part inside the frame is made (a thin strip at "cover" is huge whole)
            with open_picture(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGBA")
                self.still = self._turned(im.resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS, box=(
                    cx0 / kx, cy0 / ky, min(cx1 / kx, im.width), min(cy1 / ky, im.height))))
        else:  # ffmpeg cuts the source pixels under that part first, then scales only them
            # on even pixels: 4:2:0 colour (half size) stays in line with the picture
            sx0, sy0 = math.floor(cx0 / kx) // 2 * 2, math.floor(cy0 / ky) // 2 * 2
            sx1 = min(info["w"], -(-math.ceil(cx1 / kx) // 2) * 2)
            sy1 = min(info["h"], -(-math.ceil(cy1 / ky) // 2) * 2)
            px0, py0 = round(sx0 * kx), round(sy0 * ky)
            size = (round(sx1 * kx) - px0, round(sy1 * ky) - py0)
            self.crop = (cx0 - px0, cy0 - py0, cx1 - px0, cy1 - py0)
            self.whole = self.crop == (0, 0, *size)
            self.cursors = Cursors(path, info, size, bool(info.get("alpha")), (sx0, sy0, sx1, sy1))

    def _setup_turned(self, path, info, params, w, h, weight):
        """A turned clip: its picture is warped onto the quad homography() gives. Only the source pixels under the
        part of the quad inside the frame are read, scaled to about the placed size (so the warp is near 1:1),
        then laid by the inverse homography (bicubic) into the box around that part, see-through outside the
        quad with a soft outline (coverage())."""
        self.still, self.cursors, self.colours = None, None, colour_ops(params)
        rect = placement(params, info, w, h)
        sw, sh = rect[2], rect[3]
        hm, _ = homography(params, rect, w, h)
        pts = corners(hm, sw, sh)
        seen = _clip_polygon(pts, w, h) if abs(_area(pts)) >= 1.0 else []  # edge-on (a sliver): nothing
        if len(seen) >= 3 and abs(_area(seen)) >= 1e-3:
            x0, y0 = max(0, math.floor(min(q[0] for q in seen))), max(0, math.floor(min(q[1] for q in seen)))
            x1, y1 = min(w, math.ceil(max(q[0] for q in seen))), min(h, math.ceil(max(q[1] for q in seen)))
        else:
            x0 = y0 = x1 = y1 = 0
        self.empty = x0 >= x1 or y0 >= y1
        self.box, self.full = (x0, y0, x1, y1), (x0, y0, x1, y1) == (0, 0, w, h)
        self.weight = None if weight is None or self.empty else (weight if self.full else weight.crop(self.box))
        if self.empty:
            return
        inv = np.linalg.inv(hm)
        back = inv @ np.array([[q[0] for q in seen], [q[1] for q in seen], [1.0] * len(seen)])
        us, vs = back[0] / back[2], back[1] / back[2]
        m = 3.0  # placed pixels around what shows: the bicubic's reach
        u0, v0 = max(0.0, float(us.min()) - m), max(0.0, float(vs.min()) - m)
        u1, v1 = min(sw, float(us.max()) + m), min(sh, float(vs.max()) + m)
        if u1 - u0 < 1e-6 or v1 - v0 < 1e-6:
            self.empty = True
            return
        kx, ky = sw / info["w"], sh / info["h"]
        self.cover = coverage(pts, self.box)
        if info.get("image"):  # only the part that shows is made, at the placed size
            with open_picture(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGBA")
                kx, ky = sw / im.width, sh / im.height
                size = (max(1, round(u1 - u0)), max(1, round(v1 - v0)))
                part = im.resize(size, Image.Resampling.LANCZOS, box=(u0 / kx, v0 / ky, u1 / kx, v1 / ky))
            self._lay(hm, (u0, v0, (u1 - u0) / size[0], (v1 - v0) / size[1]))
            self.still = self._warp(part)
        else:  # ffmpeg cuts the source pixels under that part (on even pixels, as above), then scales only them
            sx0, sy0 = math.floor(u0 / kx) // 2 * 2, math.floor(v0 / ky) // 2 * 2
            sx1 = min(info["w"], -(-math.ceil(u1 / kx) // 2) * 2)
            sy1 = min(info["h"], -(-math.ceil(v1 / ky) // 2) * 2)
            size = (max(1, round((sx1 - sx0) * kx)), max(1, round((sy1 - sy0) * ky)))
            self._lay(hm, (sx0 * kx, sy0 * ky, (sx1 - sx0) * kx / size[0], (sy1 - sy0) * ky / size[1]))
            self.cursors = Cursors(path, info, size, bool(info.get("alpha")), (sx0, sy0, sx1, sy1))

    def _lay(self, hm, part):
        """The warp's coefficients: from a pixel of the box to the padded part read (whose pixel i lies at
        u = u0 + i * step in the placed picture), as Pillow's PERSPECTIVE wants them (output -> input)."""
        u0, v0, su, sv = part
        to_part = np.array([[1 / su, 0, PAD - u0 / su], [0, 1 / sv, PAD - v0 / sv], [0, 0, 1.0]])
        g = to_part @ np.linalg.inv(hm) @ np.array([[1, 0, self.box[0]], [0, 1, self.box[1]], [0, 0, 1.0]])
        g = g / g[2, 2]
        self.warp = tuple(float(v) for v in g.flatten()[:8])

    def _warp(self, im):
        """The part read (RGB or RGBA) laid onto the quad: an RGBA image of the box."""
        a = np.asarray(im)
        a = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)), mode="edge")
        size = (self.box[2] - self.box[0], self.box[3] - self.box[1])
        out = Image.fromarray(a).transform(size, Image.Transform.PERSPECTIVE, self.warp,
                                         Image.Resampling.BICUBIC)
        if out.mode == "RGBA":
            out.putalpha(ImageChops.multiply(out.getchannel("A"), self.cover))
        else:
            out.putalpha(self.cover)
        return out

    def picture(self, t):
        """The file's picture for frame t (RGB, or RGBA with transparency), cut to the part inside the frame."""
        if self.still is not None:
            return self.still
        fps_in = video.fps_value(self.info["fps"])
        out, total = None, 0.0
        sp = self.span
        if sp is None:
            parts = timeline(t, self.fps, self.frames, self.info["seconds"], self.p["start"], self.p["fade"])
        else:  # from the clip's own start (plus what was trimmed off its head), for as long as it shows
            parts = playing((t - sp.anchor) / self.fps + sp.skip, self.fps, sp.need,
                            self.info["seconds"], self.p["start"], self.p["fade"])
        for when, weight in parts:
            frame = Image.fromarray(self.cursors.get(int(when * fps_in + 1e-6)))
            total += weight
            out = frame if out is None else Image.blend(out, frame, weight / total)
        if self.warp is not None:  # turned: onto its quad
            return self._warp(out)
        return self._turned(out if self.whole else out.crop(self.crop))

    def _turned(self, im):
        for how in self.flips:
            im = im.transpose(how)
        return im

    def draw(self, t, img, gain=1.0):
        """Lay frame t of the file over `img` (an RGB image of the whole frame, changed in place), `gain` (0..1,
        a fade) times as strongly; returns it."""
        if self.empty or self.p["opacity"] <= 0 or gain <= 0:
            return img
        top = self.picture(t)
        if self.colours is not None:
            if self.still is None:
                top = coloured(top, self.colours)
            else:  # a picture: once
                self.still = top = coloured(top, self.colours)
                self.colours = None
        alpha = top.getchannel("A") if top.mode == "RGBA" else None
        rgb = top.convert("RGB") if alpha is not None else top
        region = img if self.full else img.crop(self.box)
        weight, blend = self.weight, self.p["blend"]
        if gain < 1:
            weight = (Image.new("L", region.size, round(255 * gain)) if weight is None
                      else weight.point(lambda v: round(v * gain)))
        if blend == "normal":
            mask = alpha if weight is None else (weight if alpha is None else ImageChops.multiply(alpha, weight))
            out = rgb if mask is None else Image.composite(rgb, region, mask)
        else:
            if blend == "multiply":
                if alpha is not None:  # see-through parts multiply by white: no change
                    rgb = Image.composite(rgb, Image.new("RGB", rgb.size, (255, 255, 255)), alpha)
                mixed = ImageChops.multiply(region, rgb)
            else:
                if alpha is not None:  # see-through parts add nothing
                    rgb = ImageChops.multiply(rgb, Image.merge("RGB", (alpha, alpha, alpha)))
                mixed = ImageChops.screen(region, rgb) if blend == "screen" else ImageChops.add(region, rgb)
            out = mixed if weight is None else Image.composite(mixed, region, weight)
        if self.full:
            return out
        img.paste(out, self.box[:2])
        return img

    def close(self):
        if self.cursors:
            self.cursors.close()
