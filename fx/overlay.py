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
    {"key": "start", "label": "Bắt đầu từ giây thứ (của file)", "type": "range", "min": 0, "max": 600, "step": 0.1,
     "value": 0},
    {"key": "fade", "label": "Hoà mượt khi file lặp lại (giây)", "type": "range", "min": 0, "max": 3, "step": 0.1,
     "value": 1.0},
    {"key": "reverse", "label": "Phát ngược (tua ngược)", "type": "toggle", "value": False},
]}


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
    if avail >= need:
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
