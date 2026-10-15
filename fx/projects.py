"""Projects: one video with its still areas, masks and timeline, kept in a folder of the data folder.

projects/<id>/project.json   the settings (written whole, atomically)
projects/<id>/input.<ext>    the video, as uploaded
projects/<id>/proxy.mp4      a light copy of it for the page's live preview (made in the background)
projects/<id>/still.png      the original picture, fitted to the video's size (optional)
projects/<id>/masks/*.png    masks painted on the page: freeze, fg, and one per clip
projects/<id>/clips/*        preview clips of built-in effects, for the page's live preview
projects/<id>/files/*        files of clips made by older versions (new clips use the library)

The timeline: `length` (30 or 60 seconds), `tracks` (bottom first) and `layers`, the clips: each on a track,
from frame `start` to `end` of the video made, with fades (timeline.py). A clip from an outside file names its
library item ("file": {"lib": id}).

Everything that comes from the page is checked here: ids by pattern, numbers clamped, files read by our code
only (never a path from the page).
"""
import json
import os
import re
import secrets
import shutil
import threading
import time
import unicodedata

import numpy as np
from PIL import Image, ImageOps

from . import effects, library, overlay, paths, timeline, video

PROJECT_ID = re.compile(r"^[0-9a-f]{12}$")
LAYER_ID = re.compile(r"^[a-z0-9]{4,16}$")
TRACK_ID = LAYER_ID
MASK_NAME = re.compile(r"^(freeze|fg|[a-z0-9]{4,16})$")
INPUT_NAME = re.compile(r"^input(-[0-9a-f]{6})?\.[a-z0-9]{2,4}$")
PROXY_NAME = re.compile(r"^proxy(-[0-9a-f]{6})?\.(mp4|webm)$")
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
MAX_LAYERS = 64  # clips
MAX_TRACKS = 16  # an older version allowed 16 layers, and upgrade() makes each one a track
MAX_FRAMES = 3600  # two and a half minutes at 24 fps: this app is for short loops
MAX_FPS = 240  # the output is a whole number of loops: its frame count grows with the frame rate
MASK_WIDTH = 1920  # masks are painted at most this wide; they are scaled to the video when used
LOCK = threading.RLock()


class UserError(Exception):
    """Something the user can fix; the message is shown on the page as is."""


def folder(pid):
    if not isinstance(pid, str) or not PROJECT_ID.match(pid):
        raise UserError("Không tìm thấy dự án.")
    path = os.path.join(paths.projects_dir(), pid)
    if not os.path.isfile(os.path.join(path, "project.json")):
        raise UserError("Không tìm thấy dự án.")
    return path


def load(pid):
    with LOCK:
        with open(os.path.join(folder(pid), "project.json"), encoding="utf-8") as f:
            project = json.load(f)
        if project.get("proxy") and project.get("proxy_v") != video.PROXY_V:
            project["proxy"] = None  # made by an older version (timed or coloured unlike the renderer): made again
        return upgrade(project)  # (closed first: Windows cannot replace a file that is open)


def upgrade(project):
    """A project saved by an older version: one track per layer, each layer a clip over the whole video, and its
    outside files moved into the library. Saved at once, so this happens only once."""
    if "tracks" in project:
        return project
    project["length"] = project.get("length") or timeline.DEFAULT_LENGTH
    total = timeline.project_frames(project)
    project["tracks"] = []
    for layer in project.get("layers", []):
        tid = "t" + layer["id"][:15]
        project["tracks"].append({"id": tid, "on": True})
        layer.update(track=tid, start=0, end=total, fade_in=0.0, fade_out=0.0, skip=0.0)
        stored = (layer.get("file") or {}).get("stored")
        if stored:
            layer["file"] = _to_library(project["id"], layer["file"])
    if not project["tracks"]:
        project["tracks"] = [{"id": secrets.token_hex(3), "on": True} for _ in range(2)]
    project.pop("preview", None)
    project.setdefault("proxy", None)
    if os.path.isdir(os.path.join(paths.projects_dir(), project.get("id", ""))):
        save(project, bump=False)
    return project


def _to_library(pid, source):
    """An outside file kept in a project by an older version, copied into the library; the clip's new "file"."""
    src = os.path.join(paths.projects_dir(), pid, "files", source["stored"])
    name = source.get("name") or source["stored"]
    try:
        copy = os.path.join(paths.data_dir(), "tmp", secrets.token_hex(8) + os.path.splitext(src)[1])
        os.makedirs(os.path.dirname(copy), exist_ok=True)
        shutil.copyfile(src, copy)
        item = library.add(copy, os.path.splitext(name)[0] + os.path.splitext(src)[1])
        return {"lib": item["id"], "name": item["name"], "info": item["info"]}
    except (OSError, library.LibraryError, video.VideoError):
        return source  # it stays where it was: still rendered, but not in the live preview


def replace(src, dst):
    """os.replace that waits a moment when Windows still has `dst` open for reading (a render, a request)."""
    for _ in range(20):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            time.sleep(0.05)
    os.replace(src, dst)


def save(project, bump=True):
    """Writes the project. Its new rev is one above the highest seen on disk or in `project`, so two different
    saved states never share a rev (the look cache is keyed by it)."""
    with LOCK:
        path = os.path.join(paths.projects_dir(), project["id"], "project.json")
        if bump:
            try:
                with open(path, encoding="utf-8") as f:
                    on_disk = json.load(f).get("rev", 0)
            except (OSError, ValueError):
                on_disk = 0
            project["rev"] = max(on_disk, project.get("rev", 0)) + 1
        project["updated"] = time.time()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(project, f, ensure_ascii=False, indent=1)
        replace(tmp, path)
    return project


def clean_name(name, fallback="video"):
    """A project's name from a file's name."""
    return clean_title(os.path.splitext(os.path.basename(str(name or "")))[0], fallback)


def clean_title(text, fallback):
    """A project's name as typed: whole (no file name parts cut off), safe in a file name (the exports')."""
    stem = unicodedata.normalize("NFC", str(text or ""))  # Mac/iPhone names and "Unicode tổ hợp" typing keep marks apart
    stem = re.sub(r"[^\w\- ]+", " ", stem, flags=re.UNICODE)
    stem = re.sub(r"\s+", " ", stem).strip()[:60].strip()
    return stem or fallback


def mask_size(info):
    w = min(info["w"], MASK_WIDTH)
    return w, max(2, round(info["h"] * w / info["w"]))


def summary(project):
    return {"id": project["id"], "name": project["name"], "updated": project.get("updated", 0),
            "seconds": project["video"]["seconds"], "w": project["video"]["w"], "h": project["video"]["h"],
            "length": project.get("length") or timeline.DEFAULT_LENGTH}


def list_all():
    out = []
    root = paths.projects_dir()
    for pid in os.listdir(root):
        if PROJECT_ID.match(pid):
            try:
                out.append(summary(load(pid)))
            except (OSError, ValueError, KeyError, UserError):
                continue
    return sorted(out, key=lambda p: -p["updated"])


def _checked_video(upload, original_name):
    ext = os.path.splitext(original_name or "")[1].lower()
    if ext not in VIDEO_TYPES:
        raise UserError("Hãy chọn một file video (mp4, mov, mkv, webm, avi).")
    info = video.probe(upload)
    fps = video.fps_value(info["fps"])
    if fps > MAX_FPS:
        raise UserError(f"Video có {fps:g} khung hình mỗi giây, nhiều hơn app nhận được (tối đa {MAX_FPS}). "
                        "Hãy xuất lại video ở 24, 25, 30 hoặc 60 khung/giây.")
    if info["frames"] > MAX_FRAMES:
        raise UserError(f"Video dài {info['seconds']:.0f} giây. Bokin FX dành cho video loop ngắn "
                        f"(tối đa {MAX_FRAMES // 24} giây ở 24 khung/giây).")
    return ext, info


def create(upload, original_name):
    """A new project from an uploaded video (`upload` is a file we wrote; it is moved into the project)."""
    ext, info = _checked_video(upload, original_name)
    pid = secrets.token_hex(6)
    path = os.path.join(paths.projects_dir(), pid)
    os.makedirs(os.path.join(path, "masks"))
    shutil.move(upload, os.path.join(path, "input" + ext))
    info["file"] = "input" + ext
    now = time.time()
    project = {"id": pid, "name": clean_name(original_name), "source_name": os.path.basename(original_name)[:200],
               "created": now, "rev": 0, "video": info, "still": {"source": "frame", "frame": 0},
               "freeze": {"feather": 4}, "fg": {"feather": 2}, "length": timeline.DEFAULT_LENGTH,
               "tracks": [{"id": secrets.token_hex(3), "on": True} for _ in range(3)], "layers": [], "proxy": None}
    return save(project)


def replace_video(project, upload, original_name):
    """A new video for the project (V1), keeping its timeline, areas and clips: clip times stay the same in
    seconds, clips that ran to the end still do. The new file gets a new name (the old one may still be open for
    reading on Windows; it goes in a later sweep). Returns (project, notes for the page)."""
    ext, info = _checked_video(upload, original_name)
    with LOCK:
        fresh = load(project["id"])
        path = os.path.join(paths.projects_dir(), fresh["id"])
        old = fresh["video"]
        name = f"input-{secrets.token_hex(3)}{ext}"
        shutil.move(upload, os.path.join(path, name))
        info["file"] = name
        old_total, old_fps, new_fps = timeline.project_frames(fresh), video.fps_value(old["fps"]), video.fps_value(info["fps"])
        fresh["video"] = info
        new_total = timeline.project_frames(fresh)
        most = timeline.output_frames(info, max(timeline.LENGTHS))

        def at(frame):  # the same time, in frames of the new video
            return int(round(frame / old_fps * new_fps))
        notes = []
        moved, off = _retime(fresh, at, old_total, new_total, most, new_fps)
        if moved:
            notes.append(f"{moved} clip nằm sau cuối video mới đã được dời lên trước. Hãy kiểm tra lại timeline.")
        if off:
            notes.append(f"{off} clip không còn chỗ trên track của nó nên đã được tắt. Hãy kiểm tra lại timeline.")
        fresh["still"]["frame"] = min(at(fresh["still"].get("frame", 0)), info["frames"] - 1)
        if abs(new_total / new_fps - old_total / old_fps) > 0.5 or abs(info["seconds"] - old["seconds"]) > 0.05:
            notes.append(f"Video mới có loop {info['seconds']:g} giây (video cũ {old['seconds']:g} giây): video xuất ra "
                         f"dài {new_total / new_fps:g} giây.")
        if info["h"] != old["h"]:  # soft edges are in pixels of the video: as soft, compared with the picture
            k = info["h"] / old["h"]
            for item in [fresh.setdefault("freeze", {}), fresh.setdefault("fg", {})] + fresh["layers"]:
                item["feather"] = max(0, min(200, int(round((item.get("feather") or 0) * k))))
        if (info["w"], info["h"]) != (old["w"], old["h"]) and \
                abs(info["w"] / info["h"] - old["w"] / old["h"]) > 0.01:
            notes.append("Video mới có khung hình khác tỉ lệ: các vùng đã vẽ bị kéo giãn theo. Hãy kiểm tra lại.")
        still = os.path.join(path, "still.png")
        if os.path.exists(still):  # the original picture (also when a frame is used now: it may be chosen again)
            image = fresh["still"].get("source") == "image"
            source = os.path.join(path, STILL_SOURCE)
            if (info["w"], info["h"]) != (old["w"], old["h"]):  # fitted to the new size
                if not os.path.exists(source):  # kept since version 1.0.11 only: the old fitted copy instead
                    source = still
                    if image and (info["w"] > old["w"] or info["h"] > old["h"]):
                        notes.append("Ảnh gốc của vùng đứng yên đã được phóng to theo video mới nên kém nét hơn. "
                                     "Hãy chọn lại ảnh gốc.")
                try:
                    fresh["still"]["warning"] = _fit_still(path, source, info)
                except UserError:
                    if image:
                        raise
                    _remove(still)  # a picture not in use that cannot be read: it does not stop the new video
            elif image:
                with Image.open(still) as im:
                    fresh["still"]["warning"] = _still_warning(path, im.convert("RGB"), info)
        fresh["source_name"] = os.path.basename(original_name)[:200]
        fresh["proxy"] = None
        _remove(os.path.join(path, "thumb.jpg"))
        saved = save(fresh)
        sweep_files(saved)
        return saved, notes


def delete(pid):
    path = folder(pid)
    with LOCK:
        for _ in range(5):
            try:
                shutil.rmtree(path)
                return
            except OSError:
                time.sleep(0.4)  # Windows: a file still open for a moment
        shutil.rmtree(path, ignore_errors=True)


def _layer_from_page(raw, old, project, tracks):
    if not isinstance(raw, dict):
        raise UserError("Dữ liệu clip không hợp lệ.")
    lid, kind = raw.get("id"), raw.get("kind")
    if not isinstance(lid, str) or not LAYER_ID.match(lid) or lid in ("freeze", "fg"):
        raise UserError("Mã clip không hợp lệ.")
    if kind not in effects.SCHEMA:
        raise UserError("Loại clip không hợp lệ.")
    most = timeline.output_frames(project["video"], max(timeline.LENGTHS))
    start = _int(raw.get("start"), 0, most - 1, 0)
    track = raw.get("track") if raw.get("track") in tracks else tracks[0]
    name = re.sub(r"[\x00-\x1f]", "", str(raw.get("name") or effects.SCHEMA[kind]["label"]))[:60]
    layer = {"id": lid, "kind": kind, "name": name, "on": bool(raw.get("on", True)),
             "behind": bool(raw.get("behind", False)), "feather": _int(raw.get("feather"), 0, 200, 0),
             "seed": _int(raw.get("seed"), 0, 2 ** 31 - 1, 1), "params": effects.clean_params(kind, raw.get("params")),
             "track": track, "start": start, "end": _int(raw.get("end"), start + 1, most, most),
             "fade_in": _num(raw.get("fade_in"), 0, timeline.MAX_FADE, 0),
             "fade_out": _num(raw.get("fade_out"), 0, timeline.MAX_FADE, 0),
             "skip": _num(raw.get("skip"), 0, 36000, 0, None),  # exact: the two halves of a cut clip must meet
             "need": _num(raw.get("need"), 0, 36000, 0, None),  # the first half: as long a run as the whole clip
             "short": raw.get("short") is True,  # ends at the video's end only since it was made shorter
             "mix": _num(raw.get("mix"), 0, timeline.MAX_MIX, 0, None)}  # exact: the page rounds it to frames  # a crossfade with the clip just before it
    if kind == "file":
        before = old.get("file") if old and old.get("kind") == "file" else None
        lib = raw.get("lib")  # the page names library items by id only, never files
        if lib is None or (before and before.get("lib") == lib):
            if before:
                layer["file"] = before
        else:
            try:
                item = library.load(lib)
                layer["file"] = {"lib": item["id"], "name": item["name"], "info": item["info"]}
            except library.LibraryError as e:
                if not (isinstance(lib, str) and library.ITEM_ID.match(lib)):
                    raise UserError(str(e)) from e
                layer["file"] = {"lib": lib, "name": name, "info": None}  # deleted from the library since: draws nothing
    return layer


def _int(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError, OverflowError):
        return default


def _num(value, lo, hi, default, digits=3):
    try:
        v = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if v != v:  # NaN
        return default
    v = max(lo, min(hi, v))
    return v if digits is None else round(v, digits)


def _tracks_from_page(raw):
    if not isinstance(raw, list) or len(raw) > MAX_TRACKS:
        raise UserError(f"Tối đa {MAX_TRACKS} track.")
    tracks = []
    for r in raw:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str) or not TRACK_ID.match(r["id"]):
            raise UserError("Mã track không hợp lệ.")
        tracks.append({"id": r["id"], "on": bool(r.get("on", True))})
    if len({t["id"] for t in tracks}) != len(tracks):
        raise UserError("Hai track trùng mã.")
    return tracks or [{"id": secrets.token_hex(3), "on": True}]


def apply(project, data):
    """Takes the settings the page sent (timeline, still, softness, name) and returns the saved project."""
    if not isinstance(data, dict):
        raise UserError("Dữ liệu không hợp lệ.")
    with LOCK:
        project = load(project["id"])  # the newest copy: uploads and previews may have saved since it was read
        if "name" in data:
            project["name"] = clean_title(data["name"], project["name"])
        if "length" in data:
            if data["length"] not in timeline.LENGTHS:
                raise UserError("Độ dài video không hợp lệ.")
            project["length"] = data["length"]
        still = data.get("still")
        if isinstance(still, dict):
            source = "image" if still.get("source") == "image" and project.get("still", {}).get("image_name") else "frame"
            project["still"].update({"source": source,
                                     "frame": _int(still.get("frame"), 0, project["video"]["frames"] - 1, 0)})
        for name in ("freeze", "fg"):
            if isinstance(data.get(name), dict):
                project[name] = {"feather": _int(data[name].get("feather"), 0, 200, 0)}
        if "tracks" in data:
            project["tracks"] = _tracks_from_page(data["tracks"])
        tracks = [t["id"] for t in project["tracks"]]
        if "layers" in data:
            raw = data["layers"]
            if not isinstance(raw, list) or len(raw) > MAX_LAYERS:
                raise UserError(f"Tối đa {MAX_LAYERS} clip.")
            old = {lay["id"]: lay for lay in project["layers"]}
            layers = [_layer_from_page(r, old.get(r.get("id")) if isinstance(r, dict) else None, project, tracks)
                      for r in raw]
            if len({lay["id"] for lay in layers}) != len(layers):
                raise UserError("Hai clip trùng mã.")
            project["layers"] = layers
            for lid in set(old) - {lay["id"] for lay in layers}:
                _remove(os.path.join(paths.projects_dir(), project["id"], "masks", lid + ".png"))
        for layer in project["layers"]:  # clips of a removed track go to the first one
            if layer.get("track") not in tracks:
                layer["track"] = tracks[0]
        saved = save(project)
        sweep_files(saved)
        return saved


def copy_mask(project, src, dst):
    """A clip cut in two: the new half gets the same area as the first one."""
    if not (MASK_NAME.match(src or "") and LAYER_ID.match(dst or "")) or dst in ("freeze", "fg"):
        raise UserError("Không tìm thấy clip.")
    with LOCK:
        project = load(project["id"])
        if dst not in {lay["id"] for lay in project["layers"]}:
            raise UserError("Không tìm thấy clip.")
        masks = os.path.join(paths.projects_dir(), project["id"], "masks")
        if os.path.exists(os.path.join(masks, src + ".png")):
            shutil.copyfile(os.path.join(masks, src + ".png"), os.path.join(masks, dst + ".png"))
        return save(project)


def using_library(iid):
    """Names of the projects whose clips use library item `iid`."""
    names = []
    for summary_ in list_all():
        try:
            project = load(summary_["id"])
        except (OSError, ValueError, UserError):
            continue
        if any((lay.get("file") or {}).get("lib") == iid for lay in project["layers"]):
            names.append(project["name"])
    return names


def sweep_files(project, keep_recent=4, keep_seconds=600, keep_most=16):
    """Removes files no clip uses any more: files of older versions' clips, and preview clips of built-in
    effects made for other settings. Of those, the `keep_recent` newest stay, and any made in the last
    `keep_seconds` (up to `keep_most`): the page may still be playing one while the next is made, and a change
    is often undone. A file Windows still holds open stays until a later sweep.

    Done on the newest saved copy of the project, under the lock: a copy read a moment ago may not know of a
    video that has just replaced the old one, and would remove it."""
    with LOCK:
        _sweep(load(project["id"]), keep_recent, keep_seconds, keep_most)


def _sweep(project, keep_recent, keep_seconds, keep_most):
    from .engine import fx_clip_name
    base = os.path.join(paths.projects_dir(), project["id"])
    keep = {(lay.get("file") or {}).get("stored") for lay in project["layers"]}
    for name in _names(os.path.join(base, "files")):
        if name not in keep:
            _remove(os.path.join(base, "files", name))
    current = project["video"]["file"]
    copy = "proxy" + os.path.splitext(current)[0][len("input"):]  # its light copy (engine.make_base_proxy)
    for name in _names(base):  # videos replaced by another one, and their light copies
        if INPUT_NAME.match(name) and name != current:
            _remove(os.path.join(base, name))
        elif PROXY_NAME.match(name) and os.path.splitext(name)[0] != copy:
            _remove(os.path.join(base, name))
    used = {fx_clip_name(project, lay) for lay in project["layers"] if lay["kind"] != "file"}
    clips = os.path.join(base, "clips")
    spare = sorted((n for n in _names(clips) if n not in used and ".part" not in n),
                   key=lambda n: _mtime(os.path.join(clips, n)), reverse=True)
    now = time.time()
    for i, name in enumerate(spare):
        if i >= keep_most or (i >= keep_recent and now - _mtime(os.path.join(clips, name)) > keep_seconds):
            _remove(os.path.join(clips, name))


def _names(folder):
    try:
        return os.listdir(folder)
    except OSError:
        return []


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def save_mask(project, name, data):
    """A mask painted on the page (PNG bytes; empty: the mask is removed)."""
    if not MASK_NAME.match(name or ""):
        raise UserError("Không tìm thấy lớp của vùng này.")
    mask = None
    if data:
        import io
        try:
            with Image.open(io.BytesIO(data)) as im:
                if im.format != "PNG" or im.width * im.height > 40_000_000:
                    raise ValueError
                im.load()
                mask = im.getchannel("A") if "A" in im.getbands() else im.convert("L")
        except Exception as e:  # noqa: BLE001 - anything that is not a sane PNG
            raise UserError("Vùng vẽ gửi lên không hợp lệ.") from e
        size = mask_size(project["video"])
        if mask.size != size:
            mask = mask.resize(size, Image.Resampling.BILINEAR)
    with LOCK:
        project = load(project["id"])
        if name not in ("freeze", "fg") and name not in {lay["id"] for lay in project["layers"]}:
            raise UserError("Không tìm thấy lớp của vùng này.")
        path = os.path.join(paths.projects_dir(), project["id"], "masks", name + ".png")
        if mask is None:
            _remove(path)
        else:
            tmp = f"{path}.{secrets.token_hex(3)}.tmp.png"
            mask.save(tmp, optimize=False, compress_level=3)
            replace(tmp, path)
        return save(project)


def _retime(project, at, old_total, new_total, most, fps):
    """The clips' times in frames of a new video (`at`: an old frame to a new one): the same in seconds. A clip
    over the whole video still covers it; the others are not stretched (the user joins clips). Clips past the end
    of the longest video the new one can make slide back, keeping their length, without running into the clip
    before them on their track; one with no room left is turned off. Returns (clips moved, clips turned off)."""
    moved = off = 0
    room = max(1, int(round(2 * timeline.MIN_EDGE_FADE * fps)))  # the least a clip can be: its two fades
    for track in {c["track"] for c in project["layers"]}:
        clips = sorted((c for c in project["layers"] if c["track"] == track), key=lambda c: c["start"])
        before = 0  # where the previous clip on this track ends, in new frames
        for c in clips:
            s0, e0 = at(c["start"]), at(c["end"])
            length = max(1, e0 - s0)
            whole = c["start"] == 0 and c["end"] == old_total
            if whole:  # over the whole video
                s, e = 0, new_total
            else:
                s, e = s0, min(e0, most)
                if e0 > most:
                    s, e = max(0, most - length), most
            if s < before:
                s = before
            if e - s < min(room, length):
                c["on"] = False
                off += 1
                s, e = min(s0, most - 1), min(max(s0 + 1, e0), most)  # left where it was, turned off
            elif (s, e) != (s0, e0) and not whole:
                moved += 1
            c["start"], c["end"] = s, max(e, s + 1)
            before = max(before, c["end"]) if c.get("on", True) else before
    return moved, off


STILL_SOURCE = "still-source"  # the original picture as chosen, to fit again to another size


def _still_warning(path, fitted, info):
    """A warning when the fitted original picture does not look like the video's own frames."""
    first = video.read_frame(os.path.join(path, info["file"]), info, 0, (320, max(2, round(320 * info["h"] / info["w"]))))
    small = np.asarray(fitted.resize((first.shape[1], first.shape[0]), Image.Resampling.BOX), np.float32)
    diff = float(np.median(np.abs(small - first.astype(np.float32)).mean(axis=2)))
    if diff > 18:
        return ("Ảnh gốc có vẻ khác với video (lệch vị trí, khung hình hoặc màu). Kiểm tra lại, "
                "hoặc dùng một khung của video cho vùng đứng yên.")
    return None


def _fit_still(path, source, info):
    """still.png: the picture `source` fitted (cover, centred) to the video's size. Returns the warning, if any."""
    try:
        with overlay.open_picture(source) as im:
            if im.width * im.height > overlay.MAX_IMAGE_PIXELS:
                raise UserError("Ảnh quá lớn.")
            im = ImageOps.exif_transpose(im).convert("RGB")
            fitted = ImageOps.fit(im, (info["w"], info["h"]), Image.Resampling.LANCZOS)
    except UserError:
        raise
    except Exception as e:  # noqa: BLE001
        raise UserError("Không đọc được ảnh này. Hãy chọn ảnh PNG hoặc JPG.") from e
    warning = _still_warning(path, fitted, info)  # before anything is written: a failure leaves the old picture
    still = os.path.join(path, "still.png")
    tmp = still + ".tmp.png"
    fitted.save(tmp, compress_level=3)
    replace(tmp, still)
    return warning


def set_still_image(project, upload, original_name):
    """The original picture for the still areas: fitted (cover, centred) to the video's size, and kept as it was
    chosen (a new video of another size gets it fitted again). Warns when it does not look like the video's own
    frames."""
    path = os.path.join(paths.projects_dir(), project["id"])
    with LOCK:  # fitted to the project's video as it is now (it may have been replaced during the upload)
        fresh = load(project["id"])
        warning = _fit_still(path, upload, fresh["video"])
        _remove(os.path.join(path, STILL_SOURCE))
        shutil.move(upload, os.path.join(path, STILL_SOURCE))
        fresh["still"].update({"source": "image", "image_name": os.path.basename(original_name)[:200],
                               "warning": warning})
        return save(fresh)


def set_proxy(pid, name, source):
    """The light copy of the project's video `source` is ready (made in the background); kept only if that is
    still the project's video."""
    with LOCK:
        fresh = load(pid)
        if fresh["video"]["file"] != source:
            _remove(os.path.join(paths.projects_dir(), pid, name))
            return fresh
        fresh["proxy"], fresh["proxy_v"] = name, video.PROXY_V
        return save(fresh, bump=False)
