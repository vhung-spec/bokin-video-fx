"""Puts the frames together and renders: the project's video, then its still areas, then its layers in order.

A project folder holds the video (input.*), optionally the original picture (still.png, at the video's size),
and the masks painted on the page (masks/<name>.png: white = selected). A mask is softened at render time by its
"feather" (pixels of the full-size video), so the painting stays sharp and the softness can change.

- Still areas ("freeze"): there the frame is replaced by the original picture, or by one chosen frame of the
  video, so nothing moves. Where the mask is fully white the pixels are exactly the picture's.
- Layers: effects (effects.py) and files (overlay.py). A layer's area mask limits where it shows (no mask: the
  whole frame); a layer set "behind" is hidden by the foreground mask ("fg"), as if it passed behind those things.
"""
import collections
import os
import threading

import numpy as np
from PIL import Image, ImageFilter

from . import effects, overlay, video

PREVIEW_HEIGHT = 480
LOOK_WIDTH = 1280


def mask_path(folder, name):
    return os.path.join(folder, "masks", name + ".png")


def load_mask(folder, name, size, feather, video_h):
    """A mask as float32 (h, w) in 0..1 at `size`, softened by `feather` px of the full video; None if none."""
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
    return np.asarray(m, np.float32) * (1 / 255.0)


def input_path(folder, project):
    return os.path.join(folder, project["video"]["file"])


def still_picture(folder, project, size):
    """The picture the still areas show, as float32 RGB at `size`."""
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
    return arr.astype(np.float32) * (1 / 255.0)


class Scene:
    """A project's settings made ready to draw frames at `size` (w, h)."""

    def __init__(self, folder, project, size):
        info = project["video"]
        self.w, self.h = size
        self.frames, fps = info["frames"], video.fps_value(info["fps"])
        vh = info["h"]
        self.freeze = load_mask(folder, "freeze", size, project.get("freeze", {}).get("feather", 0), vh)
        if self.freeze is not None and self.freeze.max() <= 0:
            self.freeze = None
        if self.freeze is not None:
            ys, xs = np.nonzero(self.freeze > 0)
            self.box = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
            self.still = still_picture(folder, project, size)
        fg = load_mask(folder, "fg", size, project.get("fg", {}).get("feather", 0), vh)
        if fg is not None and fg.max() <= 0:
            fg = None
        self.layers = []
        try:
            for layer in project.get("layers", []):
                if not layer.get("on", True):
                    continue
                area = load_mask(folder, layer["id"], size, layer.get("feather", 0), vh)
                if layer.get("behind") and fg is not None:
                    area = (1.0 - fg) if area is None else area * (1.0 - fg)
                if area is not None and area.max() <= 0:
                    continue  # nowhere to show
                if area is not None and area.min() >= 1:
                    area = None
                if layer["kind"] == "file":
                    source = layer.get("file") or {}
                    path = os.path.join(folder, "files", source.get("stored", ""))
                    if not source.get("stored") or not os.path.exists(path):
                        continue
                    params = effects.clean_params("file", layer.get("params"))
                    drawer = overlay.Overlay(path, source["info"], params, self.frames, fps, self.w, self.h)
                else:
                    drawer = effects.make(layer["kind"], layer.get("params"), layer.get("seed", 1), self.frames, fps,
                                          self.w, self.h)
                self.layers.append((drawer, area))
        except Exception:
            self.close()
            raise
        self.canvas = effects.Canvas(self.w, self.h)

    def frame(self, t, base):
        """Frame t: `base` (uint8 RGB, the video's frame t at this size) with the still areas and the layers."""
        picture = base.astype(np.float32) * (1 / 255.0)
        if self.freeze is not None:
            y0, y1, x0, x1 = self.box
            part = picture[y0:y1, x0:x1]
            part += (self.still[y0:y1, x0:x1] - part) * self.freeze[y0:y1, x0:x1, None]
        mode = None
        for drawer, area in self.layers:
            if isinstance(drawer, overlay.Overlay):
                if mode:
                    self.canvas.flush(picture, mode)
                    mode = None
                drawer.draw(t, picture, area)
            else:
                if mode and drawer.blend != mode:
                    self.canvas.flush(picture, mode)
                mode = drawer.blend
                drawer.draw(t, self.canvas, area)
        if mode:
            self.canvas.flush(picture, mode)
        np.clip(picture, 0.0, 1.0, out=picture)
        return (picture * 255.0 + 0.5).astype(np.uint8)

    def close(self):
        for drawer, _ in self.layers:
            if isinstance(drawer, overlay.Overlay):
                drawer.close()


def preview_size(info, height=PREVIEW_HEIGHT):
    h = min(height, info["h"])
    h -= h % 2
    return max(2, round(info["w"] * h / info["h"] / 2) * 2), h


def render(folder, project, out_path, size=None, crf=16, preset="slow", progress=None, cancelled=None):
    """Writes one loop of the project to `out_path` (full size unless `size`). False if cancelled."""
    info = project["video"]
    size = tuple(size or (info["w"], info["h"]))
    scene = Scene(folder, project, size)
    enc = video.Encoder(out_path, size[0], size[1], info["fps"], crf, preset)
    try:
        done = 0
        with video.Frames(input_path(folder, project), info, size) as frames:
            for t, base in enumerate(frames):
                if t >= info["frames"]:
                    break
                if cancelled and cancelled():
                    enc.abort()
                    return False
                enc.write(scene.frame(t, base))
                done += 1
                if progress:
                    progress(done / info["frames"])
        if done == 0:
            raise video.VideoError("Không đọc được khung hình nào của video.")
        enc.close()
        return True
    except BaseException:
        enc.abort()
        raise
    finally:
        scene.close()


class Looks:
    """Single frames with every layer, for the page while settings are being changed: the last scenes and video
    frames are kept, so moving a slider or the time only redoes what changed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.scenes = collections.OrderedDict()
        self.bases = collections.OrderedDict()

    def frame(self, folder, project, t, width=LOOK_WIDTH):
        info = project["video"]
        w = min(width, info["w"])
        w -= w % 2
        size = (w, max(2, round(info["h"] * w / info["w"] / 2) * 2))
        t = max(0, min(int(t), info["frames"] - 1))
        with self.lock:
            key = (folder, project.get("rev", 0), size)
            scene = self.scenes.pop(key, None) or Scene(folder, project, size)
            self.scenes[key] = scene
            while len(self.scenes) > 2:
                self.scenes.popitem(last=False)[1].close()
            bkey = (folder, project["video"]["file"], t, size)
            base = self.bases.pop(bkey, None)
            if base is None:
                base = video.read_frame(input_path(folder, project), info, t, size)
            self.bases[bkey] = base
            while len(self.bases) > 8:
                self.bases.popitem(last=False)
            return scene.frame(t, base)

    def forget(self, folder):
        with self.lock:
            for key in [k for k in self.scenes if k[0] == folder]:
                self.scenes.pop(key).close()
            for key in [k for k in self.bases if k[0] == folder]:
                self.bases.pop(key)
