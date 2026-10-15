"""Layers made from outside files, as in After Effects: a video (an effect on black, or one with transparency)
or a picture (PNG, JPG), laid over the loop with a blend mode, opacity, size and position.

A video of any length is made to loop with the project. Frame t of the loop shows the file at
start + fade + t/fps; over the last `fade` seconds of the loop it dissolves into the file at start .. start + fade,
which is exactly what frame 0 shows next, so the seam is a soft dissolve instead of a jump. A file shorter than
the loop plus the fade is first played over and over, each repeat dissolving into the next.
"""
import math
import os

import numpy as np
from PIL import Image, ImageOps

from . import video

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_TYPES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
MAX_IMAGE_PIXELS = 40_000_000

BLENDS = [("normal", "Bình thường (Normal)"), ("screen", "Screen (hiệu ứng trên nền đen)"),
          ("add", "Add (cộng sáng)"), ("multiply", "Multiply (làm tối)")]
SCHEMA = {"label": "Video / ảnh ngoài", "params": [
    {"key": "blend", "label": "Chế độ hoà trộn", "type": "choice", "value": "screen",
     "options": [{"value": v, "label": t} for v, t in BLENDS], "presets": {}},
    {"key": "opacity", "label": "Độ đậm", "type": "range", "min": 0, "max": 100, "value": 100},
    {"key": "fit", "label": "Khớp khung", "type": "choice", "value": "cover", "presets": {},
     "options": [{"value": "cover", "label": "Phủ kín khung"}, {"value": "contain", "label": "Vừa trong khung"}]},
    {"key": "scale", "label": "Cỡ (%)", "type": "range", "min": 10, "max": 300, "value": 100},
    {"key": "x", "label": "Dịch ngang (%)", "type": "range", "min": -100, "max": 100, "value": 0},
    {"key": "y", "label": "Dịch dọc (%)", "type": "range", "min": -100, "max": 100, "value": 0},
    {"key": "start", "label": "Bắt đầu từ giây", "type": "range", "min": 0, "max": 600, "step": 0.1, "value": 0},
    {"key": "fade", "label": "Hoà mượt chỗ nối loop (giây)", "type": "range", "min": 0, "max": 3, "step": 0.1,
     "value": 1.0},
]}


def is_image(name):
    return os.path.splitext(name)[1].lower() in IMAGE_TYPES


def is_video(name):
    return os.path.splitext(name)[1].lower() in VIDEO_TYPES


def probe_file(path):
    """What a file to lay over the loop is: like video.probe, with image=True for a picture."""
    if is_image(path):
        try:
            with Image.open(path) as im:
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


def timeline(t, fps, frames, seconds, start, fade):
    """[(time in the file, weight), ...] that make frame t of a loop of `frames` frames at `fps`, from a file of
    `seconds` seconds. Weights add up to 1."""
    loop = frames / fps
    start = min(max(0.0, start), max(0.0, seconds - 1.0 / fps))
    avail = seconds - start
    fade = max(0.0, min(fade, loop / 2))
    if avail >= loop + fade:
        step, join = None, 0.0
    else:
        join = min(fade, avail / 3)
        step = avail - join
        if step <= 0:
            return [(start, 1.0)]

    def played(e):  # the file from `start` on, played over and over
        if step is None:
            return [(start + e, 1.0)]
        j = math.floor(e / step + 1e-9)
        local = max(0.0, e - j * step)  # the 1e-9 above can leave -1e-16 here
        if j > 0 and join > 0 and local < join:
            w = local / join
            return [(start + local, w), (start + local + step, 1.0 - w)]
        return [(start + local, 1.0)]

    s = t / fps
    if fade <= 0:
        parts = played(s)
    elif s < loop - fade:
        parts = played(fade + s)
    else:
        w = (s - (loop - fade)) / fade
        parts = [(ft, pw * (1 - w)) for ft, pw in played(fade + s)] + \
                [(ft, pw * w) for ft, pw in played(s - (loop - fade))]
    return [(ft, w) for ft, w in parts if w > 1e-4]


class Cursors:
    """Frames of a video by number, read in order by a few ffmpeg readers (no file of decoded frames):
    a loop needs a run of frames from one place, and near the end another run from the start."""

    def __init__(self, path, info, size, alpha):
        self.path, self.info, self.size, self.alpha = path, info, size, alpha
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
            reader = video.Frames(self.path, self.info, self.size, alpha=self.alpha, start=i)
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


def _premultiplied(frame):
    f = frame.astype(np.float32) * (1 / 255.0)
    if f.shape[2] == 3:
        return np.dstack([f, np.ones(f.shape[:2], np.float32)])
    f[..., :3] *= f[..., 3:4]
    return f


class Overlay:
    """One layer from a file, drawn at w x h for a loop of `frames` frames at `fps`."""

    def __init__(self, path, info, params, frames, fps, w, h):
        self.info, self.p = info, params
        self.frames, self.fps = int(frames), float(fps)
        k = (max if params["fit"] == "cover" else min)(w / info["w"], h / info["h"]) * params["scale"] / 100
        self.sw, self.sh = max(2, round(info["w"] * k)), max(2, round(info["h"] * k))
        self.ox = round((w - self.sw) / 2 + params["x"] / 100 * w)
        self.oy = round((h - self.sh) / 2 + params["y"] / 100 * h)
        self.w, self.h = w, h
        self.still, self.cursors = None, None
        if info.get("image"):
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im).convert("RGBA").resize((self.sw, self.sh), Image.Resampling.LANCZOS)
                self.still = _premultiplied(np.asarray(im))
        else:
            self.cursors = Cursors(path, info, (self.sw, self.sh), bool(info.get("alpha")))

    def layer(self, t):
        """The file's picture for frame t (premultiplied RGBA float32, at its placed size)."""
        if self.still is not None:
            return self.still
        fps_in = video.fps_value(self.info["fps"])
        out = None
        for when, weight in timeline(t, self.fps, self.frames, self.info["seconds"], self.p["start"], self.p["fade"]):
            f = _premultiplied(self.cursors.get(int(when * fps_in + 1e-6)))
            if out is None:
                out = f * weight if weight != 1.0 else f
            else:
                out += f * weight
        return out

    def draw(self, t, picture, mask):
        """Lay frame t of the file over `picture` (float32 RGB 0..1, changed in place)."""
        x0, y0 = max(self.ox, 0), max(self.oy, 0)
        x1, y1 = min(self.ox + self.sw, self.w), min(self.oy + self.sh, self.h)
        if x0 >= x1 or y0 >= y1 or self.p["opacity"] <= 0:
            return
        src = self.layer(t)[y0 - self.oy:y1 - self.oy, x0 - self.ox:x1 - self.ox]
        k = self.p["opacity"] / 100
        weight = np.full((y1 - y0, x1 - x0), k, np.float32) if mask is None else mask[y0:y1, x0:x1] * k
        colour = src[..., :3] * weight[..., None]
        out = picture[y0:y1, x0:x1]
        blend = self.p["blend"]
        if blend == "screen":
            out[:] = 1.0 - (1.0 - out) * (1.0 - np.minimum(colour, 1.0))
        elif blend == "add":
            out += colour
            np.minimum(out, 1.0, out=out)
        elif blend == "multiply":
            a = (src[..., 3] * weight)[..., None]
            out *= 1.0 - a + colour
        else:
            out *= 1.0 - (src[..., 3] * weight)[..., None]
            out += colour

    def close(self):
        if self.cursors:
            self.cursors.close()
