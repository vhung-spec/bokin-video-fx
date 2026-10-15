"""Puts the frames together and renders: the project's video, then its still areas, then its layers in order.

A video is made for a length (30 s or 1 min): a whole number of loops of the project's video, so its end joins
its start. Every layer repeats over that whole length (its period), not over the short loop: an outside effect
plays through instead of restarting every few seconds. The frames are rendered in pieces by several worker
processes at once, each piece encoded on its own, then joined by copying.

A project folder holds the video (input.*), optionally the original picture (still.png, at the video's size),
and the masks painted on the page (masks/<name>.png: white = selected). A mask is softened at render time by its
"feather" (pixels of the full-size video), so the painting stays sharp and the softness can change.

- Still areas ("freeze"): there the frame is replaced by the original picture, or by one chosen frame of the
  video, so nothing moves. Where the mask is fully white the pixels are exactly the picture's.
- Layers: effects (effects.py) and files (overlay.py). A layer's area mask limits where it shows (no mask: the
  whole frame); a layer set "behind" is hidden by the foreground mask ("fg"), as if it passed behind those things.
"""
import collections
import concurrent.futures
import multiprocessing
import os
import secrets
import shutil
import threading

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from . import effects, overlay, video

PREVIEW_HEIGHT = 480
LOOK_WIDTH = 1280


def mask_path(folder, name):
    return os.path.join(folder, "masks", name + ".png")


def mask_image(folder, name, size, feather, video_h):
    """A mask as an "L" image at `size`, softened by `feather` px of the full video; None if none."""
    path = mask_path(folder, name)
    if not os.path.exists(path):
        return None
    with Image.open(path) as im:
        im.load()
        m = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
    if m.size != tuple(size):
        m = m.resize(size, Image.Resampling.BILINEAR)
    radius = feather * size[1] / video_h
    if radius >= 0.3:
        m = m.filter(ImageFilter.GaussianBlur(radius))
    return m


def load_mask(folder, name, size, feather, video_h):
    """The same mask as float32 (h, w) in 0..1; None if none."""
    m = mask_image(folder, name, size, feather, video_h)
    return None if m is None else np.asarray(m, np.float32) * (1 / 255.0)


def input_path(folder, project):
    return os.path.join(folder, project["video"]["file"])


def still_picture(folder, project, size):
    """The picture the still areas show, as an RGB image at `size`."""
    info = project["video"]
    still = project.get("still") or {}
    path = os.path.join(folder, "still.png")
    if still.get("source") == "image" and os.path.exists(path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != tuple(size):
                im = im.resize(size, Image.Resampling.LANCZOS)
            arr = np.asarray(im)
    else:
        arr = video.read_frame(input_path(folder, project), info, still.get("frame", 0), size)
    return Image.fromarray(arr)


def output_frames(info, seconds):
    """Frames of a video of about `seconds`: a whole number of the source loop, so the end joins the start."""
    return max(1, round(seconds / info["seconds"])) * info["frames"]


class Scene:
    """A project's settings made ready to draw frames at `size` (w, h); layers repeat every `period` frames.

    The picture is an 8-bit RGB image; still areas and outside files are laid with Pillow's C routines, built-in
    effects on a float copy of the picture (only while such layers are drawn)."""

    def __init__(self, folder, project, size, period=None):
        info = project["video"]
        self.w, self.h = size
        self.frames, fps = period or info["frames"], video.fps_value(info["fps"])
        vh = info["h"]
        self.freeze = mask_image(folder, "freeze", size, project.get("freeze", {}).get("feather", 0), vh)
        self.freeze_box = self.freeze.getbbox() if self.freeze is not None else None
        if self.freeze_box:
            box = self.freeze_box
            self.still = still_picture(folder, project, size).crop(box)
            self.freeze = self.freeze.crop(box)
        fg = mask_image(folder, "fg", size, project.get("fg", {}).get("feather", 0), vh)
        if fg is not None and fg.getbbox() is None:
            fg = None
        self.layers = []
        try:
            for layer in project.get("layers", []):
                if not layer.get("on", True):
                    continue
                area = mask_image(folder, layer["id"], size, layer.get("feather", 0), vh)
                if layer.get("behind") and fg is not None:
                    hidden = ImageChops.invert(fg)
                    area = hidden if area is None else ImageChops.multiply(area, hidden)
                if area is not None and area.getbbox() is None:
                    continue  # nowhere to show
                if area is not None and area.getextrema() == (255, 255):
                    area = None
                if layer["kind"] == "file":
                    source = layer.get("file") or {}
                    path = os.path.join(folder, "files", source.get("stored", ""))
                    if not source.get("stored") or not os.path.exists(path):
                        continue
                    params = effects.clean_params("file", layer.get("params"))
                    opacity = params["opacity"] / 100
                    if opacity < 1:
                        area = (Image.new("L", size, round(255 * opacity)) if area is None
                                else area.point(lambda v: round(v * opacity)))
                    drawer = overlay.Overlay(path, source["info"], params, self.frames, fps, self.w, self.h, area)
                    self.layers.append((drawer, None))
                else:
                    drawer = effects.make(layer["kind"], layer.get("params"), layer.get("seed", 1), self.frames, fps,
                                          self.w, self.h)
                    self.layers.append((drawer, None if area is None else np.asarray(area, np.float32) * (1 / 255.0)))
        except Exception:
            self.close()
            raise
        self.canvas = effects.Canvas(self.w, self.h) if any(
            not isinstance(d, overlay.Overlay) for d, _ in self.layers) else None

    def image(self, t, base):
        """Frame t as an RGB image: `base` (uint8 RGB array, the video's frame at this size) with the still areas
        and the layers."""
        img = Image.fromarray(base)
        if self.freeze_box:
            box = self.freeze_box
            img.paste(Image.composite(self.still, img.crop(box), self.freeze), box[:2])
        i = 0
        while i < len(self.layers):
            drawer, _ = self.layers[i]
            if isinstance(drawer, overlay.Overlay):
                img = drawer.draw(t, img)
                i += 1
                continue
            picture = np.asarray(img, np.float32) * (1 / 255.0)  # a run of built-in effects
            mode = None
            while i < len(self.layers) and not isinstance(self.layers[i][0], overlay.Overlay):
                drawer, area = self.layers[i]
                if mode and drawer.blend != mode:
                    self.canvas.flush(picture, mode)
                mode = drawer.blend
                drawer.draw(t, self.canvas, area)
                i += 1
            self.canvas.flush(picture, mode)
            np.clip(picture, 0.0, 1.0, out=picture)
            img = Image.fromarray((picture * 255.0 + 0.5).astype(np.uint8))
        return img

    def frame(self, t, base):
        """Frame t as a uint8 RGB array."""
        return np.asarray(self.image(t, base))

    def close(self):
        for drawer, _ in self.layers:
            if isinstance(drawer, overlay.Overlay):
                drawer.close()


def preview_size(info, height=PREVIEW_HEIGHT):
    h = min(height, info["h"])
    h -= h % 2
    return max(2, round(info["w"] * h / info["h"] / 2) * 2), h


def base_frames(folder, project, size, start, end):
    """The project video's frames for output frames start..end-1, the video looping."""
    info = project["video"]
    loop = info["frames"]
    t = start
    while t < end:
        first, count = t % loop, min(loop - t % loop, end - t)
        got = 0
        with video.Frames(input_path(folder, project), info, size, start=first, count=count) as frames:
            for frame in frames:
                yield frame
                got += 1
        if got < count:
            raise video.VideoError("Không đọc được hết khung hình của video.")
        t += count


_done = _stop = None  # set in each worker process: frames done (shared counter), stop request (shared event)


def _worker_init(done, stop):
    global _done, _stop
    _done, _stop = done, stop


def render_part(folder, project, size, period, start, end, path, crf, preset, threads):
    """One piece (frames start..end-1) of a video `period` frames long, encoded to `path`. False if stopped."""
    scene = Scene(folder, project, size, period)
    enc = video.Encoder(path, size[0], size[1], project["video"]["fps"], crf, preset, threads, part=True)
    try:
        for t, base in zip(range(start, end), base_frames(folder, project, size, start, end)):
            if _stop is not None and _stop.is_set():
                enc.abort()
                return False
            enc.write(scene.image(t, base))
            if _done is not None:
                with _done.get_lock():
                    _done.value += 1
        enc.close()
        return True
    except BaseException:
        enc.abort()
        raise
    finally:
        scene.close()


def workers():
    """Worker processes for a render: half the logical CPUs (the encoders use the rest), 1 to 8."""
    return max(1, min(8, (os.cpu_count() or 2) // 2))


def render(folder, project, out_path, seconds=None, size=None, crf=16, preset="faster", progress=None,
           cancelled=None, processes=None):
    """Writes the project to `out_path`: about `seconds` long (one loop if None), full size unless `size`.
    Returns False if cancelled."""
    info = project["video"]
    size = tuple(size or (info["w"], info["h"]))
    period = output_frames(info, seconds) if seconds else info["frames"]
    count = max(1, min(processes or workers(), period // 24))  # pieces of at least a second
    bounds = [round(period * i / count) for i in range(count + 1)]
    work = os.path.join(folder, "render-" + secrets.token_hex(3))
    os.makedirs(work)
    parts = [os.path.join(work, f"part-{i:02d}.mp4") for i in range(count)]
    threads = 2 if count > 1 else 0
    try:
        if count == 1:
            done = [0]

            class Counter:  # the same interface as the shared multiprocessing counter
                def get_lock(self):
                    return threading.Lock()

                @property
                def value(self):
                    return done[0]

                @value.setter
                def value(self, v):
                    done[0] = v
                    if progress:
                        progress(v / period)

            class Stop:
                def is_set(self):
                    return bool(cancelled and cancelled())

            _worker_init(Counter(), Stop())
            try:
                ok = render_part(folder, project, size, period, 0, period, parts[0], crf, preset, threads)
            finally:
                _worker_init(None, None)
        else:
            ctx = multiprocessing.get_context("spawn")
            shared, stop = ctx.Value("i", 0), ctx.Event()
            with concurrent.futures.ProcessPoolExecutor(count, mp_context=ctx, initializer=_worker_init,
                                                        initargs=(shared, stop)) as pool:
                futures = [pool.submit(render_part, folder, project, size, period, bounds[i], bounds[i + 1],
                                       parts[i], crf, preset, threads) for i in range(count)]
                while True:
                    finished, pending = concurrent.futures.wait(futures, timeout=0.4)
                    if progress:
                        progress(min(1.0, shared.value / period))
                    if cancelled and cancelled():
                        stop.set()
                    failed = next((f for f in finished if f.exception() is not None), None)
                    if failed is not None:
                        stop.set()
                        raise failed.exception()
                    if not pending:
                        break
                ok = all(f.result() for f in futures)
        if not ok or (cancelled and cancelled()):
            return False
        video.join(parts, out_path)
        return True
    finally:
        shutil.rmtree(work, ignore_errors=True)


def layer_clip(folder, project, layer_id, out_base, size, cancelled=None):
    """A short clip of one built-in effect alone, for the page's live preview: one loop of the project video,
    transparent WebM for effects laid over the picture, H.264 on black for light (laid with Screen).
    Returns the file written."""
    info = project["video"]
    layer = next(lay for lay in project["layers"] if lay["id"] == layer_id)
    fx = effects.make(layer["kind"], layer.get("params"), layer.get("seed", 1), info["frames"],
                      video.fps_value(info["fps"]), size[0], size[1])
    alpha = fx.blend != "screen"
    path = out_base + (".webm" if alpha else ".mp4")
    writer = video.ClipWriter(path, size[0], size[1], info["fps"], alpha)
    canvas = effects.Canvas(size[0], size[1])
    try:
        for t in range(info["frames"]):
            if cancelled and cancelled():
                writer.close()
                video._remove(path)
                return None
            fx.draw(t, canvas, None)
            if alpha:
                a = canvas.a.copy()
                rgb = np.where(a[..., None] > 1e-6, canvas.rgb / np.maximum(a[..., None], 1e-6), 0.0)
                frame = np.dstack([np.clip(rgb, 0, 1), np.clip(a, 0, 1)])
            else:
                frame = np.clip(canvas.rgb, 0, 1)
            writer.write((frame * 255 + 0.5).astype(np.uint8))
            canvas.reset()
        writer.close()
        return path
    except BaseException:
        writer.close()
        raise


class Looks:
    """Single frames with every layer, for the page while settings are being changed: the last scenes and video
    frames are kept, so moving a slider or the time only redoes what changed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.scenes = collections.OrderedDict()
        self.bases = collections.OrderedDict()

    def frame(self, folder, project, t, width=LOOK_WIDTH, seconds=None):
        info = project["video"]
        w = min(width, info["w"])
        w -= w % 2
        size = (w, max(2, round(info["h"] * w / info["w"] / 2) * 2))
        period = output_frames(info, seconds) if seconds else info["frames"]
        t = max(0, min(int(t), period - 1))
        with self.lock:
            key = (folder, project.get("rev", 0), size, period)
            scene = self.scenes.pop(key, None) or Scene(folder, project, size, period)
            self.scenes[key] = scene
            while len(self.scenes) > 2:
                self.scenes.popitem(last=False)[1].close()
            bkey = (folder, project["video"]["file"], t % info["frames"], size)
            base = self.bases.pop(bkey, None)
            if base is None:
                base = video.read_frame(input_path(folder, project), info, t % info["frames"], size)
            self.bases[bkey] = base
            while len(self.bases) > 8:
                self.bases.popitem(last=False)
            return scene.frame(t, base)

    def release(self, folder):
        """Closes the cached scenes of `folder` (their ffmpeg readers hold layer files open, and Windows does not
        let an open file be deleted). The video frames stay cached."""
        with self.lock:
            for key in [k for k in self.scenes if k[0] == folder]:
                self.scenes.pop(key).close()

    def forget(self, folder):
        with self.lock:
            for key in [k for k in self.scenes if k[0] == folder]:
                self.scenes.pop(key).close()
            for key in [k for k in self.bases if k[0] == folder]:
                self.bases.pop(key)
