"""Puts the frames together and renders: the project's video, then its still areas, then its clips.

A project makes one video of its length (30 s or 1 min, timeline.py): a whole number of loops of its video, so
its end joins its start. The clips sit on tracks (drawn bottom track first) and show between their start and end,
fading in and out; a clip over the whole video repeats over that whole length, not over the short loop, so an
outside effect plays through instead of restarting every few seconds. The frames are rendered in pieces by
several worker processes at once, each piece encoded on its own, then joined by copying.

A project folder holds the video (input.*), optionally the original picture (still.png, at the video's size),
and the masks painted on the page (masks/<name>.png: white = selected). A mask is softened at render time by its
"feather" (pixels of the full-size video), so the painting stays sharp and the softness can change.

- Still areas ("freeze"): there the frame is replaced by the original picture, or by one chosen frame of the
  video, so nothing moves. Where the mask is fully white the pixels are exactly the picture's.
- Clips: built-in effects (effects.py) and outside files (overlay.py, kept in the library). A clip's area mask
  limits where it shows (no mask: the whole frame); a clip set "behind" is hidden by the foreground mask ("fg"),
  as if it passed behind those things. A built-in effect placed or turned in 3D is drawn on its own sheet, then
  laid as a file clip is (sheet.py).
"""
import collections
import concurrent.futures
import copy
import hashlib
import io
import json
import math
import multiprocessing
import os
import secrets
import shutil
import threading

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from . import effects, library, overlay, paths, sheet, timeline, video

LOOK_WIDTH = 1280
FX_CLIP_WIDTH = 640


def mask_path(folder, name):
    return os.path.join(folder, "masks", name + ".png")


INSIDE = [max(0, 2 * v - 255) for v in range(256)]  # a softened edge moved in: 0 from the painted line outwards


def mask_image(folder, name, size, feather, video_h, saved=None, inside=False):
    """A mask as an "L" image at `size`, softened by `feather` px of the full video; None if none. `saved`: the
    masks as read before a render was handed out (found_files). `inside`: the soft edge lies within what was
    painted (a clip's area: nothing of the clip shows outside its outline), else across the painted line."""
    if saved is not None:
        path = saved.get(name) and io.BytesIO(saved[name])
    else:
        path = mask_path(folder, name)
        path = path if os.path.exists(path) else None
    if not path:
        return None
    with Image.open(path) as im:
        im.load()
        m = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
    if m.size != tuple(size):
        m = m.resize(size, Image.Resampling.BILINEAR)
    radius = feather * size[1] / video_h
    if radius >= 0.3:
        m = m.filter(ImageFilter.GaussianBlur(radius))
        if inside:
            m = m.point(INSIDE)
    return m


def area_feather(layer):
    """The softness of a clip's area as laid: an effect that works only in its area (rustling leaves) takes it as
    painted and fades in from its outline itself."""
    cls = effects.KINDS.get(layer.get("kind"))
    return 0 if cls is not None and cls.AREA else layer.get("feather", 0)


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
    if "_still" in project:  # read before a render was handed out (found_files)
        path = project["_still"] and io.BytesIO(project["_still"])
    elif not os.path.exists(path):
        path = None
    if still.get("source") == "image" and path:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.size != tuple(size):
                im = im.resize(size, Image.Resampling.LANCZOS)
            arr = np.asarray(im)
    else:
        arr = video.read_frame(input_path(folder, project), info, still.get("frame", 0), size)
    return Image.fromarray(arr)


LIGHT_FRAMES = 24  # frames of the video its light map is made from (spread evenly over it)
_lights = {}  # (video file, its stamp) -> its light map


def scene_light(folder, project):
    """The light of the project's video (effects.light_map), for the effects lit by it (Effect.LIT): made from
    frames spread evenly over the whole loop (a lamp lit for a moment counts for that moment), once per video file;
    None if it cannot be read. A copy handed to worker processes carries it ("_light": found_files, fx_clip), so
    they read nothing."""
    if "_light" in project:
        return project["_light"]
    path, info = input_path(folder, project), project["video"]
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, st.st_mtime_ns, st.st_size)
    if key not in _lights:
        h = min(info["h"], 4 * effects.LIGHT_H)  # (light averaged in linear light from four times the map's size)
        size = (max(2, round(info["w"] * h / info["h"] / 2) * 2), h - h % 2)
        try:
            step = max(1, info["frames"] // LIGHT_FRAMES)
            with video.Frames(path, info, size, count=min(info["frames"], LIGHT_FRAMES), step=step) as frames:
                light = effects.light_map(frames)
        except (video.VideoError, OSError, ValueError):
            light = None
        if len(_lights) > 8:
            _lights.clear()
        _lights[key] = light
    return _lights[key]


output_frames = timeline.output_frames
project_frames = timeline.project_frames


def clip_order(project):
    """The clips to draw, bottom first: by track (lowest first), then by start. Switched-off clips and clips on
    hidden tracks are left out."""
    tracks = project.get("tracks") or []
    rank = {t["id"]: i for i, t in enumerate(tracks)}
    hidden = {t["id"] for t in tracks if not t.get("on", True)}
    clips = [c for c in project.get("layers", []) if c.get("on", True) and c.get("track") not in hidden]
    return sorted(clips, key=lambda c: (rank.get(c.get("track"), len(rank)), c.get("start", 0)))


def clip_file(folder, clip):
    """(path, info) of an outside-file clip's file, or (None, None) when it is missing."""
    source = clip.get("file") or {}
    if source.get("found"):  # looked up before the work was handed to worker processes
        path, info = source["found"]
    elif source.get("lib"):
        try:
            item = library.load(source["lib"])
        except library.LibraryError:
            return None, None
        if (clip.get("params") or {}).get("reverse") and not item["info"].get("image"):
            path, info = library.reversed_path(item)  # played backwards: its own copy, nothing until it is made
            if path is None:
                return None, None
        else:
            path, info = library.source_path(item), item["info"]
    elif source.get("stored"):  # made by an older version, kept in the project
        path, info = os.path.join(folder, "files", source["stored"]), source.get("info")
    else:
        return None, None
    return (path, info) if info and os.path.exists(path) else (None, None)


class Scene:
    """A project's settings made ready to draw frames at `size` (w, h) of a video `period` frames long (the
    project's length if None).

    The picture is an 8-bit RGB image; still areas and outside files are laid with Pillow's C routines, built-in
    effects on a float copy of the picture (only while such clips show)."""

    def __init__(self, folder, project, size, period=None):
        info = project["video"]
        self.w, self.h = size
        self.frames = period or project_frames(project)
        fps = video.fps_value(info["fps"])
        vh, saved = info["h"], project.get("_masks")
        self.freeze = mask_image(folder, "freeze", size, project.get("freeze", {}).get("feather", 0), vh, saved)
        self.freeze_box = self.freeze.getbbox() if self.freeze is not None else None
        if self.freeze_box:
            box = self.freeze_box
            self.still = still_picture(folder, project, size).crop(box)
            self.freeze = self.freeze.crop(box)
        fg = mask_image(folder, "fg", size, project.get("fg", {}).get("feather", 0), vh, saved)
        if fg is not None and fg.getbbox() is None:
            fg = None
        self.layers = []  # (drawer, area as float or None (built-in effects), timeline.Span)
        self.missing = []  # file clips left out, their file not there (yet: a played-backwards copy being made)
        ids = []
        scratch = sheet.Scratch()  # (the memory the effects placed or turned share)
        light = []  # (the video's light, read when a clip is lit by it)

        def lit(kind):
            if not effects.KINDS[kind].LIT:
                return None
            if not light:
                light.append(scene_light(folder, project))
            return light[0]

        try:
            shown = clip_order(project)
            pairs = timeline.mix_pairs(shown, self.frames, fps)
            mixes = timeline.mixes(shown, self.frames, fps)
            for layer in shown:
                span = timeline.Span(layer, self.frames, fps, mixes.get(layer["id"], (0.0, 0.0)))
                if span.empty:
                    continue
                area = mask_image(folder, layer["id"], size, area_feather(layer), vh, saved, inside=True)
                if layer.get("behind") and fg is not None:
                    hidden = ImageChops.invert(fg)
                    area = hidden if area is None else ImageChops.multiply(area, hidden)
                if area is not None and area.getbbox() is None:
                    continue  # nowhere to show
                if area is not None and area.getextrema() == (255, 255):
                    area = None
                if layer["kind"] == "file":
                    path, source = clip_file(folder, layer)
                    if path is None:
                        self.missing.append(layer)
                        continue
                    params = effects.clean_params("file", layer.get("params"))
                    opacity = params["opacity"] / 100
                    if opacity < 1:
                        area = (Image.new("L", size, round(255 * opacity)) if area is None
                                else area.point(lambda v: round(v * opacity)))
                    drawer = overlay.Overlay(path, source, params, self.frames, fps, self.w, self.h, area, span)
                    self.layers.append((drawer, None, span))
                    ids.append(layer["id"])
                elif effects.on_sheet(layer["kind"], layer.get("params")):  # placed or turned: laid as a file clip
                    drawer = sheet.Sheet(layer["kind"], layer.get("params"), layer.get("seed", 1), self.frames, fps,
                                         self.w, self.h, area, info["w"] / info["h"], scratch, lit(layer["kind"]))
                    self.layers.append((drawer, None, span))
                    ids.append(layer["id"])
                else:
                    drawer = effects.make(layer["kind"], layer.get("params"), layer.get("seed", 1), self.frames, fps,
                                          self.w, self.h, lit(layer["kind"]))
                    self.layers.append((drawer, None if area is None else np.asarray(area, np.float32) * (1 / 255.0),
                                        span))
                    ids.append(layer["id"])
        except Exception:
            self.close()
            raise
        # layer i and the next one crossfade (a Mix): on a track, by start, so drawn one right after the other
        pairs = {(a, b) for a, b, _ in pairs}
        self.mixed = [(ids[i], ids[i + 1]) in pairs for i in range(len(ids) - 1)] + [False]
        upright = any(not isinstance(d, overlay.Laid) for d, _, _ in self.layers)
        sheets = [d.scratch_size() for d, _, _ in self.layers if isinstance(d, sheet.Sheet)]
        if sheets:  # one canvas for them and the upright effects, as big as the biggest, made once
            scratch.floats("canvas", max([n for n, _ in sheets] + [self.w * self.h * 4 if upright else 0]))
            scratch.floats("lens", max(n for _, n in sheets))
        self.canvas = None
        if upright:
            self.canvas = scratch.canvas(self.w, self.h) if sheets else effects.Canvas(self.w, self.h)

    def image(self, t, base):
        """Frame t as an RGB image: `base` (uint8 RGB array, the video's frame at this size) with the still areas
        and the clips showing at t."""
        img = Image.fromarray(base)
        if self.freeze_box:
            box = self.freeze_box
            img.paste(Image.composite(self.still, img.crop(box), self.freeze), box[:2])
        layers, i = self.layers, 0
        for drawer, _, span in layers:  # clips not showing now give back their big buffers (made again later)
            if hasattr(drawer, "release") and span.weight(t) <= 0:
                drawer.release()
        while i < len(layers):
            both = self._mix_at(i, t)
            if both:  # a Mix: the picture with each clip alone, then the two crossfaded (no dip in the middle)
                img = self._crossfade(t, img, i, *both)
                i += 2
                continue
            if isinstance(layers[i][0], overlay.Laid):  # a file, or an effect placed or turned: on the 8-bit picture
                drawer, _, span = layers[i]
                img = drawer.draw(t, img, span.weight(t))
                i += 1
                continue
            j = i + 1  # a run of built-in effects, drawn on a float copy of the picture
            while j < len(layers) and not isinstance(layers[j][0], overlay.Laid) and not self._mix_at(j, t):
                j += 1
            run = [(d, area, span.weight(t)) for d, area, span in layers[i:j]]
            i = j
            if not any(gain > 0 for _, _, gain in run):
                continue
            picture = np.asarray(img, np.float32) * (1 / 255.0)
            mode = None
            for drawer, area, gain in run:
                if gain <= 0:
                    continue
                if drawer.lens:  # it changes the picture under it: what was drawn before goes on first
                    if mode:
                        self.canvas.flush(picture, mode)
                        mode = None
                    drawer.apply(t, picture, area, gain)
                    continue
                if mode and (drawer.blend != mode or gain < 1):
                    self.canvas.flush(picture, mode)
                mode = drawer.blend
                drawer.draw(t, self.canvas, area)
                if gain < 1:  # fading: this clip alone, weaker
                    self.canvas.scale(gain)
                    self.canvas.flush(picture, mode)
                    mode = None
            if mode:
                self.canvas.flush(picture, mode)
            np.clip(picture, 0.0, 1.0, out=picture)
            img = Image.fromarray((picture * 255.0 + 0.5).astype(np.uint8))
        return img

    def _mix_at(self, i, t):
        """(weight of layer i, of the next one) when the two crossfade at frame t (both showing), else None."""
        if not self.mixed[i]:
            return None
        wa, wb = self.layers[i][2].weight(t), self.layers[i + 1][2].weight(t)
        return (wa, wb) if wa > 0 and wb > 0 else None

    def _crossfade(self, t, img, i, wa, wb):
        """`img` with layers i and i+1 in a Mix: img + wa * (img with i alone - img) + wb * (img with i+1 alone - img).
        With the same blend the page gets the same (both clips in one isolated group, the later one plus-lighter)."""
        base = np.asarray(img, np.float32)
        out = base.copy()
        for k, w in ((i, wa), (i + 1, wb)):
            out += w * (np.asarray(self._alone(k, t, img), np.float32) - base)
        return Image.fromarray((np.clip(out, 0.0, 255.0) + 0.5).astype(np.uint8))

    def _alone(self, k, t, img):
        """A copy of `img` with layer k laid over it at full strength."""
        drawer, area, _ = self.layers[k]
        if isinstance(drawer, overlay.Laid):
            return drawer.draw(t, img.copy())
        picture = np.asarray(img, np.float32) * (1 / 255.0)
        if drawer.lens:
            drawer.apply(t, picture, area, 1.0)
        else:
            drawer.draw(t, self.canvas, area)
            self.canvas.flush(picture, drawer.blend)
        np.clip(picture, 0.0, 1.0, out=picture)
        return Image.fromarray((picture * 255.0 + 0.5).astype(np.uint8))

    def frame(self, t, base):
        """Frame t as a uint8 RGB array."""
        return np.asarray(self.image(t, base))

    def close(self):
        for drawer, _, _ in self.layers:
            if isinstance(drawer, overlay.Laid):
                drawer.close()


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
_here = threading.local()  # the same, when a single piece is made in this process (by this thread)


def _worker_init(done, stop):
    global _done, _stop
    _done, _stop = done, stop


def render_part(folder, project, size, period, crf, preset, threads, start, end, path):
    """One piece (frames start..end-1) of a video `period` frames long, encoded to `path`. False if stopped."""
    scene = Scene(folder, project, size, period)
    enc = video.Encoder(path, size[0], size[1], project["video"]["fps"], crf, preset, threads, part=True)
    try:
        for t, base in zip(range(start, end), base_frames(folder, project, size, start, end)):
            if _stopped():
                enc.abort()
                return False
            enc.write(scene.image(t, base))
            _count_frame()
        enc.close()
        return True
    except BaseException:
        enc.abort()
        raise
    finally:
        scene.close()


def workers(encoder_threads=True, pixels=0):
    """Worker processes for a render: half the logical CPUs when the encoders need the rest, else all but one;
    1 to 8, and at most 4 for frames bigger than 1440p (each worker holds a few full frames and each effect's own
    buffers: at 4K about 0.9 GB with five snow clips, upright or turned (turned ones share one canvas:
    sheet.Scratch), 1.4 GB at 200%, more with water on glass: 2 to 3 GB with five; rain lit by the scene keeps
    its colours as full-size 8-bit maps: about 60 MB more a rain at 4K, 115 MB turned at 140%)."""
    cpus = os.cpu_count() or 2
    most = 4 if pixels > 2560 * 1440 else 8
    return max(1, min(most, cpus // 2 if encoder_threads else cpus - 1))


def _in_pieces(work, args, total, parts, progress=None, cancelled=None):
    """Runs work(*args, start, end, path) for len(parts) pieces of frames 0..total-1 (in worker processes when
    more than one); True if every piece was finished, False if cancelled."""
    count = len(parts)
    bounds = [round(total * i / count) for i in range(count + 1)]
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
                    progress(v / total)

        class Stop:
            def is_set(self):
                return bool(cancelled and cancelled())

        _here.done, _here.stop = Counter(), Stop()
        try:
            return work(*args, 0, total, parts[0])
        finally:
            _here.done = _here.stop = None
    ctx = multiprocessing.get_context("spawn")
    shared, stop = ctx.Value("i", 0), ctx.Event()
    with concurrent.futures.ProcessPoolExecutor(count, mp_context=ctx, initializer=_worker_init,
                                                initargs=(shared, stop)) as pool:
        futures = [pool.submit(work, *args, bounds[i], bounds[i + 1], parts[i]) for i in range(count)]
        while True:
            finished, pending = concurrent.futures.wait(futures, timeout=0.4)
            if progress:
                progress(min(1.0, shared.value / total))
            if cancelled and cancelled():
                stop.set()
            failed = next((f for f in finished if f.exception() is not None), None)
            if failed is not None:
                stop.set()
                raise failed.exception()
            if not pending:
                break
        return all(f.result() for f in futures)


def _count_frame():
    done = getattr(_here, "done", None) or _done
    if done is not None:
        with done.get_lock():
            done.value += 1


def _stopped():
    stop = getattr(_here, "stop", None) or _stop
    return stop is not None and stop.is_set()


def found_files(folder, project):
    """A copy of the project in which every outside-file clip names its file, found now: worker processes then
    look nothing up (the library is found through the data folder, which they might see differently). It also
    holds the masks and the still picture as they are now: an area painted or a clip deleted meanwhile changes
    nothing in a render under way."""
    project = copy.deepcopy(project)
    for layer in project.get("layers", []):
        if layer["kind"] == "file":
            path, info = clip_file(folder, layer)
            layer["file"] = {"found": (path, info)} if path else {}
    names = ["freeze", "fg"] + [layer["id"] for layer in project.get("layers", [])]
    project["_masks"] = {name: _read(mask_path(folder, name)) for name in names}
    if any(effects.KINDS[lay["kind"]].LIT for lay in project.get("layers", []) if lay["kind"] in effects.KINDS):
        project["_light"] = scene_light(folder, project)
    image = (project.get("still") or {}).get("source") == "image"
    project["_still"] = _read(os.path.join(folder, "still.png")) if image else None
    return project


def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def render(folder, project, out_path, seconds=None, size=None, crf=16, preset="faster", progress=None,
           cancelled=None, processes=None):
    """Writes the project to `out_path`: about `seconds` long (the project's length if None), full size unless
    `size`. Returns False if cancelled."""
    project = found_files(folder, project)
    info = project["video"]
    size = tuple(size or (info["w"], info["h"]))
    period = output_frames(info, seconds) if seconds else project_frames(project)
    count = max(1, min(processes or workers(pixels=size[0] * size[1]), period // 24))  # pieces of at least a second
    work = os.path.join(folder, "render-" + secrets.token_hex(3))
    os.makedirs(work)
    parts = [os.path.join(work, f"part-{i:02d}.mp4") for i in range(count)]
    threads = 2 if count > 1 else 0
    try:
        ok = _in_pieces(render_part, (folder, project, size, period, crf, preset, threads), period, parts,
                        progress, cancelled)
        if not ok or (cancelled and cancelled()):
            return False
        video.join(parts, out_path)
        return True
    finally:
        shutil.rmtree(work, ignore_errors=True)


def fx_clip_size(info, width=FX_CLIP_WIDTH):
    w = min(width, info["w"])
    w -= w % 2
    return w, max(2, round(info["h"] * w / info["w"] / 2) * 2)


def sharper_size(info, width):
    """A sharper preview clip's size: as many pixels as a 16:9 one `width` wide (a tall video gains nothing from
    more: the page shows it small), never fewer than the usual clip's, never more than the video's."""
    usual = fx_clip_size(info)
    scale = min(1.0, math.sqrt(width * width * 9 / 16 / (info["w"] * info["h"])))
    w = int(info["w"] * scale)
    w -= w % 2
    return usual if w <= usual[0] else (w, max(2, round(info["h"] * w / info["w"] / 2) * 2))


def fx_clip_frame(info, kind, layer=None):
    """The size of a built-in effect's preview clip: the video's, small (FX_CLIP_WIDTH wide, or as many pixels as a
    16:9 clip the effect's own PREVIEW wide: small falling leaves and thin rain streaks blur at 640); for a placed effect, its box's shape
    (one made from the picture under it no bigger than the box is in the video)."""
    cls = effects.KINDS[kind]
    box = cls.BOX
    if not box:
        return sharper_size(info, cls.PREVIEW) if cls.PREVIEW else fx_clip_size(info)
    h = min(FX_CLIP_WIDTH, max(64, info["h"]))
    if cls.UNDER and layer is not None:
        p = effects.clean_params(kind, layer.get("params"))
        h = min(h, max(64, math.ceil(min(info["w"] / box[0], info["h"] / box[1]) * p["scale"] / 100 * box[1])))
    h -= h % 2
    return max(2, round(h * box[0] / box[1] / 2) * 2), h


_near = {}  # (mask file, its stamp, a part of it) -> what is painted there


def _painted_near(path, stamp, part):
    """A short hash of what a mask has painted in `part` (left, top, right, bottom: fractions of its size, a few of
    its pixels added round it); None if nothing is painted there. Kept for the file's stamp."""
    key = (path, stamp, tuple(round(v, 5) for v in part))
    if key not in _near:
        try:
            with Image.open(path) as im:
                im.load()
                m = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
        except (OSError, ValueError):
            return "unreadable"  # (as if painted: the clip is made again when it can be read)
        rect = (max(0, math.floor(part[0] * m.width) - 2), max(0, math.floor(part[1] * m.height) - 2),
                min(m.width, math.ceil(part[2] * m.width) + 2), min(m.height, math.ceil(part[3] * m.height) + 2))
        crop = m.crop(rect) if rect[0] < rect[2] and rect[1] < rect[3] else None
        if len(_near) > 256:
            _near.clear()
        _near[key] = (hashlib.sha1(repr(rect).encode() + crop.tobytes()).hexdigest()[:12]
                      if crop is not None and crop.getbbox() else None)
    return _near[key]


def _folder_of(project):
    return os.path.join(paths.projects_dir(), project.get("id") or "-")


AREA_REACH = 16  # how far beyond its area an effect made from the picture under its area reads it (1080-line px)


def _area_painted(folder, name):
    """(a short hash of what a clip's area mask has painted, its painted box as fractions of the mask's size), or
    None when the area leaves nothing to change: no area (the whole frame), nothing painted, or painted all over (as
    Scene reads it: the whole frame). Kept for the file's stamp."""
    path = mask_path(folder, name)
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (path, (st.st_mtime_ns, st.st_size), "area")
    if key not in _near:
        try:
            with Image.open(path) as im:
                im.load()
                m = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
        except (OSError, ValueError):
            return "unreadable", (0.0, 0.0, 1.0, 1.0)  # (as if painted: the clip is made again when it can be read)
        box = m.getbbox()
        if len(_near) > 256:
            _near.clear()
        _near[key] = None if box is None or m.getextrema() == (255, 255) else (
            hashlib.sha1(m.tobytes()).hexdigest()[:12],
            (box[0] / m.width, box[1] / m.height, box[2] / m.width, box[3] / m.height))
    return _near[key]


def _under_key(project, layer):
    """What the picture under a placed effect's box (or a full-frame one's area: what is painted, how soft) is made
    of: the video, and the still area where its soft edge can reach the box (its softness, what is painted there,
    its picture). A still area elsewhere (painted, changed or undone) does not make the clip again."""
    key = [project["video"].get("file")]
    folder = _folder_of(project)
    info = project["video"]
    w, h = info["w"], info["h"]
    if effects.KINDS[layer["kind"]].BOX:
        x0, y0, bw, bh = effects.box_of(layer["kind"], layer.get("params"), w, h)
    else:  # its own area (the page lays the clip's area and the foreground over it, as over any clip)
        area = _area_painted(folder, layer["id"])
        if area is None:
            return key + ["nowhere"]  # (it changes nothing: an empty clip)
        painted, (ax0, ay0, ax1, ay1) = area
        key += ["area", painted, float(area_feather(layer) or 0)]
        reach = AREA_REACH * h / 1080
        x0, y0, bw, bh = ax0 * w - reach, ay0 * h - reach, (ax1 - ax0) * w + 2 * reach, (ay1 - ay0) * h + 2 * reach
    path = mask_path(folder, "freeze")
    try:
        st = os.stat(path)
    except OSError:
        return key
    feather = float((project.get("freeze") or {}).get("feather", 0) or 0)
    grow = 3 * feather + 2  # (a Gaussian blur of radius r spreads about 2.5 r)
    painted = _painted_near(path, (st.st_mtime_ns, st.st_size),
                            ((x0 - grow) / w, (y0 - grow) / h, (x0 + bw + grow) / w, (y0 + bh + grow) / h))
    if not painted:
        return key
    still, picture = project.get("still") or {}, None
    if still.get("source") == "image":  # (as still_picture: a frame of the video when there is no picture)
        try:
            sp = os.stat(os.path.join(folder, "still.png"))
            picture = ["image", sp.st_mtime_ns, sp.st_size]
        except OSError:
            pass
    return key + [feather, painted, picture or ["frame", still.get("frame", 0)]]


def fx_clip_params(kind, params):
    """The settings its preview clip is made with: not its place or turn (the page lays the clip there, turned, so
    turning is instant and the clip's name stays), and a placed effect fills the clip (effects.sheet_params)."""
    if not effects.KINDS[kind].BOX and not any(k in effects.SHEET_KEYS for k in params or {}):
        return params  # (a clip saved before effects were placed: exactly as before)
    return effects.sheet_params(kind, params)


def fx_clip_name(project, layer):
    """The file name of a built-in effect's preview clip: a key of everything that changes its picture (not its
    place on the timeline: the effect is drawn over the whole length), so the two halves of a cut clip share it."""
    info = project["video"]
    cls = effects.KINDS[layer["kind"]]
    # (made from the picture under its box: the box's place too, 2: only its change kept, see change_rgba; lit by the
    # video's light: the video too, and a placed one the light where its box stands)
    p = effects.clean_params(layer["kind"], layer.get("params"))
    lit = cls.LIT and p.get("lit", 1) > 0
    at = []
    if lit and cls.BOX:  # (the light where its box stands, a little rounded: a nudge does not make it again)
        tone = effects.placed_light(layer["kind"], scene_light(_folder_of(project), project), p, info["w"], info["h"],
                                    rounded=True)
        at = [["at", [round(float(v), 6) for v in tone]]] if tone is not None else []
    key = json.dumps([layer["kind"], fx_clip_params(layer["kind"], layer.get("params")), layer.get("seed", 1),
                      project_frames(project), info["fps"],
                      fx_clip_frame(info, layer["kind"], layer)] + [cls.VERSION] * (cls.VERSION > 1)  # (1: as before)
                     + ([p, _under_key(project, layer), 2] if cls.UNDER else [])
                     + [["light", info.get("file")]] * lit + at, sort_keys=True)
    alpha = cls.blend != "screen" or cls.UNDER  # (made from the picture under its box: soft edges)
    return f"fx-{hashlib.sha1(key.encode()).hexdigest()[:12]}" + (".webm" if alpha else ".mp4")


def fx_clip(folder, project, layer_id, cancelled=None, progress=None, processes=None):
    """The preview clip of one built-in effect alone, for the page's live preview: the whole length of the video
    made, small, transparent WebM for effects laid over the picture, H.264 on black for light (laid with Screen).
    Rendered in pieces like a video. Returns the file name (in folder/clips), or None if cancelled."""
    layer = next(lay for lay in project["layers"] if lay["id"] == layer_id)
    name = fx_clip_name(project, layer)
    clips = os.path.join(folder, "clips")
    os.makedirs(clips, exist_ok=True)
    path = os.path.join(clips, name)
    if os.path.exists(path):
        return name
    total = project_frames(project)
    if effects.KINDS[layer["kind"]].LIT:  # (the video's light, read here: the workers read nothing)
        project = dict(project, _light=scene_light(folder, project))
    if effects.KINDS[layer["kind"]].AREA:  # (the masks read once, here: every piece is made with the same area)
        project = dict(project, _masks={n: _read(mask_path(folder, n)) for n in ("freeze", "fg", layer_id)})
    count = max(1, min(processes or workers(False), total // 48))
    work = os.path.join(folder, "render-" + secrets.token_hex(3))
    os.makedirs(work)
    ext = os.path.splitext(name)[1]
    parts = [os.path.join(work, f"part-{i:02d}{ext}") for i in range(count)]
    try:
        if not _in_pieces(fx_part, (project, layer, total, folder), total, parts, progress, cancelled):
            return None
        if cancelled and cancelled():
            return None
        tmp = os.path.join(work, "joined" + ext)
        video.join(parts, tmp, fast_start=ext == ".mp4", alpha=ext == ".webm")
        os.replace(tmp, path)
        return name
    finally:
        shutil.rmtree(work, ignore_errors=True)


def max5(a):
    """The largest value within 5 x 5 pixels of each pixel of a 2-D uint8 array, the edge repeated beyond it: the
    same as Pillow's MaxFilter(5), many times faster (a square's maximum is a row's maximum of column maxima)."""
    h, w = a.shape
    p = np.pad(a, ((0, 0), (2, 2)), mode="edge")
    rows = p[:, 0:w].copy()
    for i in range(1, 5):
        np.maximum(rows, p[:, i:i + w], out=rows)
    p = np.pad(rows, ((2, 2), (0, 0)), mode="edge")
    out = p[0:h].copy()
    for i in range(1, 5):
        np.maximum(out, p[i:i + h], out=out)
    return out


def change_rgba(before, after, fade):
    """An RGBA picture (uint8) that, laid with Normal blend over `before`, gives `after` (both float RGB in 0..1),
    its change faded out by `fade` (h, w, 1). Its alpha is no more than the change needs (softly spread a little),
    so wherever the picture is not changed, what the page lays under it (other clips) shows through."""
    d = (np.clip(after, 0.0, 1.0) - before) * fade
    # the least alpha that reaches before + d with a colour in 0..1: d / (1 - before) up, -d / before down
    need = np.where(d > 0, d / np.maximum(1.0 - before, 1e-4), -d / np.maximum(before, 1e-4))
    need = np.maximum(np.maximum(need[..., 0], need[..., 1]), need[..., 2])
    a = (np.clip(need, 0.0, 1.0) * 255.0 + 0.999).astype(np.uint8)
    # spread a little: a gentler alpha keeps the colours tame where it changes fast (the clip's colour is halved)
    a = np.maximum(a, np.asarray(Image.fromarray(max5(a)).filter(ImageFilter.GaussianBlur(2))))
    k = np.where(a > 0, 255.0 / np.maximum(a, 1), 0.0).astype(np.float32)[..., None]
    colour = np.clip(before + d * k, 0.0, 1.0)
    return np.concatenate([(colour * 255.0 + 0.5).astype(np.uint8), a[..., None]], 2)


def _bands_of(where, grow, step=32):
    """Rectangles (x0, y0, x1, y1), bands of `step` rows each as wide as `where` (a 2-D bool array) is within `grow`
    pixels of them, that together cover every pixel within `grow` of a true one."""
    h, w = where.shape
    cols = np.flatnonzero(where.any(axis=0))
    rows = where.any(axis=1)
    out = []
    for y0 in range(0, h, step):
        y1 = min(h, y0 + step)
        near = slice(max(0, y0 - grow), min(h, y1 + grow))
        if not rows[near].any():
            continue
        used = cols[where[near][:, cols].any(axis=0)]
        out.append((max(0, int(used[0]) - grow), y0, min(w, int(used[-1]) + 1 + grow), y1))
    return out


def _area_part(project, layer, total, folder, start, end, path):
    """fx_part of a full-frame effect made from the picture under the clip's area (Effect.UNDER without a box:
    rustling leaves): the video and its still area at the preview clip's size, the effect applied there inside the
    clip's own area, as Scene does; only its change is kept (change_rgba), so it sits on the video the page plays
    under it, and lets the clips of lower tracks show where nothing changes."""
    info = project["video"]
    kind = layer["kind"]
    size = cw, ch = fx_clip_frame(info, kind)
    area = mask_image(folder, layer["id"], size, area_feather(layer), info["h"], project.get("_masks"), inside=True)
    rects = []
    if area is not None and area.getextrema() != (255, 255):  # (as Scene: painted all over is the whole frame)
        area = np.asarray(area, np.float32) * (1 / 255.0)
        rects = _bands_of(area > 0, 8)  # (change_rgba spreads its alpha a little)
    fx = effects.make(kind, fx_clip_params(kind, layer.get("params")), layer.get("seed", 1), total,
                      video.fps_value(info["fps"]), cw, ch)
    # the video with its still area, no clips (nothing painted: an empty clip, the video not even read)
    under = Scene(folder, dict(project, layers=[]), size, total) if rects else None
    writer = video.ClipWriter(path, cw, ch, info["fps"], True, getattr(fx, "PREVIEW_CRF", None), bt709=True)
    out = np.zeros((ch, cw, 4), np.uint8)  # (nothing changes outside the area: there it stays see-through)
    bases = base_frames(folder, project, size, start, end) if rects else (None for _ in range(start, end))
    try:
        for t, base in zip(range(start, end), bases):
            if _stopped():
                writer.abort()
                return False
            if rects:
                picture = np.asarray(under.image(t, base), np.float32) * (1 / 255.0)
                # each band worked out with 8 rows round it (the alpha's spread reaches across bands), its own kept
                before = [picture[max(0, y0 - 8):y1 + 8, x0:x1].copy() for x0, y0, x1, y1 in rects]
                fx.apply(t, picture, area, 1.0)
                for (x0, y0, x1, y1), was in zip(rects, before):
                    a = max(0, y0 - 8)
                    change = change_rgba(was, picture[a:y1 + 8, x0:x1], 1.0)
                    out[y0:y1, x0:x1] = change[y0 - a:y1 - a]
            writer.write(out)
            _count_frame()
        writer.close()
        return True
    except BaseException:
        writer.abort()
        raise
    finally:
        if under is not None:
            under.close()


def _under_part(project, layer, total, folder, start, end, path):
    """fx_part of an effect made from the picture under its box (Effect.UNDER): the video and its still area there,
    the box alone applied to it; only its change is kept (change_rgba), fading out toward the box's edges, so it
    sits on the video the page plays under it and lets the clips of lower tracks show where nothing changes. A
    full-frame one: under its area (_area_part)."""
    if not effects.KINDS[layer["kind"]].BOX:
        return _area_part(project, layer, total, folder, start, end, path)
    info = project["video"]
    fps, kind, seed = video.fps_value(info["fps"]), layer["kind"], layer.get("seed", 1)
    w, h = info["w"], info["h"]
    cw, ch = fx_clip_frame(info, kind, layer)
    x0, y0, bw, bh = effects.make(kind, layer.get("params"), seed, total, fps, w, h).box()
    alone = effects.make(kind, fx_clip_params(kind, layer.get("params")), seed, total, fps, cw, ch)
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    edge = np.minimum(np.minimum(xx + 0.5, cw - 0.5 - xx), np.minimum(yy + 0.5, ch - 0.5 - yy))
    fade = np.clip(edge / max(2.0, 0.08 * min(cw, ch)), 0.0, 1.0)
    fade = (fade * fade * (3 - 2 * fade))[..., None]
    inside = 0 <= x0 and 0 <= y0 and x0 + bw <= w and y0 + bh <= h
    under = Scene(folder, dict(project, layers=[]), (w, h), total)  # the video with its still area, no clips
    writer = video.ClipWriter(path, cw, ch, info["fps"], True)
    try:
        for t, base in zip(range(start, end), base_frames(folder, project, (w, h), start, end)):
            if _stopped():
                writer.abort()
                return False
            img = under.image(t, base)
            if inside:
                img = img.resize((cw, ch), Image.Resampling.BICUBIC, box=(x0, y0, x0 + bw, y0 + bh))
            else:  # partly outside the frame (black there)
                img = img.crop((math.floor(x0), math.floor(y0), math.ceil(x0 + bw), math.ceil(y0 + bh))).resize(
                    (cw, ch), Image.Resampling.BICUBIC)
            picture = np.asarray(img, np.float32) * (1 / 255.0)
            before = picture.copy()
            alone.apply(t, picture, None, 1.0)
            writer.write(change_rgba(before, picture, fade))
            _count_frame()
        writer.close()
        return True
    except BaseException:
        writer.abort()
        raise
    finally:
        under.close()


def fx_part(project, layer, total, folder, start, end, path):
    """Frames start..end-1 of a built-in effect's preview clip, written to `path`. False if stopped."""
    if effects.KINDS[layer["kind"]].UNDER:
        return _under_part(project, layer, total, folder, start, end, path)
    info = project["video"]
    w, h = fx_clip_frame(info, layer["kind"])
    light, cls = project.get("_light"), effects.KINDS[layer["kind"]]
    if cls.BOX and cls.LIT:  # (lit where its box stands in the video, not in the clip's own frame)
        light = effects.placed_light(layer["kind"], light, layer.get("params"), info["w"], info["h"], rounded=True)
    fx = effects.make(layer["kind"], fx_clip_params(layer["kind"], layer.get("params")), layer.get("seed", 1), total,
                      video.fps_value(info["fps"]), w, h, light)
    alpha = fx.blend != "screen"
    writer = video.ClipWriter(path, w, h, info["fps"], alpha, getattr(fx, "PREVIEW_CRF", None))
    canvas = effects.Canvas(w, h)
    out = np.zeros((h, w, 4 if alpha else 3), np.uint8)
    tmp, one = np.zeros((h, w), np.float32), np.zeros((h, w), np.float32)
    try:
        for t in range(start, end):
            if _stopped():
                writer.abort()
                return False
            fx.draw(t, canvas, None)
            if alpha:  # straight (not premultiplied) colour and coverage (a channel at a time: no big temporaries)
                np.maximum(canvas.a, 1 / 255, out=tmp)
                np.divide(255.0, tmp, out=tmp)
                for c in range(3):
                    np.multiply(canvas.rgb[..., c], tmp, out=one)
                    np.minimum(one, 255.0, out=one)
                    one += 0.5
                    out[..., c] = one
                np.multiply(canvas.a, 255.0, out=tmp)
                tmp += 0.5
                out[..., 3] = tmp
            else:
                rgb = np.minimum(canvas.rgb, 1.0)
                rgb *= 255.0
                rgb += 0.5
                out[:] = rgb
            writer.write(out)
            canvas.reset()
            _count_frame()
        writer.close()
        return True
    except BaseException:
        writer.abort()
        raise


def make_base_proxy(folder, project):
    """The light copy of the project's video the page plays, named after the video (input-<x>: proxy-<x>), so a
    replaced video never shares its name; returns its file name."""
    info = project["video"]
    tail = os.path.splitext(info["file"])[0][len("input"):]
    return os.path.basename(video.make_proxy(input_path(folder, project), info, os.path.join(folder, "proxy" + tail),
                                             alpha=False, keyint=6))


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
        period = project_frames(project)
        t = max(0, min(int(t), period - 1))
        with self.lock:
            key = (folder, project.get("rev", 0), size, period)
            scene = self.scenes.pop(key, None)
            if scene is not None and any(clip_file(folder, c)[0] for c in scene.missing):
                scene.close()  # a file it had to leave out is there now
                scene = None
            scene = scene or Scene(folder, project, size, period)
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

    def release_all(self):
        """Closes every cached scene (before a library file is deleted: a scene may still be reading it)."""
        with self.lock:
            while self.scenes:
                self.scenes.popitem()[1].close()

    def forget(self, folder):
        with self.lock:
            for key in [k for k in self.scenes if k[0] == folder]:
                self.scenes.pop(key).close()
            for key in [k for k in self.bases if k[0] == folder]:
                self.bases.pop(key)
