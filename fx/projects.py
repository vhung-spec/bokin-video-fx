"""Projects: one video with its still areas, masks and layers, kept in a folder of the data folder.

projects/<id>/project.json   the settings (written whole, atomically)
projects/<id>/input.<ext>    the video, as uploaded
projects/<id>/still.png      the original picture, fitted to the video's size (optional)
projects/<id>/masks/*.png    masks painted on the page: freeze, fg, and one per layer
projects/<id>/files/*        files of "file" layers (outside videos and pictures)
projects/<id>/preview-N.mp4  the latest small preview

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

from . import effects, overlay, paths, video

PROJECT_ID = re.compile(r"^[0-9a-f]{12}$")
LAYER_ID = re.compile(r"^[a-z0-9]{4,16}$")
MASK_NAME = re.compile(r"^(freeze|fg|[a-z0-9]{4,16})$")
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}
MAX_LAYERS = 16
MAX_FRAMES = 3600  # two and a half minutes at 24 fps: this app is for short loops
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
    with LOCK, open(os.path.join(folder(pid), "project.json"), encoding="utf-8") as f:
        return json.load(f)


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
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    stem = unicodedata.normalize("NFC", stem)  # Mac/iPhone names and "Unicode tổ hợp" typing keep marks apart
    stem = re.sub(r"[^\w\- ]+", " ", stem, flags=re.UNICODE)
    stem = re.sub(r"\s+", " ", stem).strip()[:60]
    return stem or fallback


def mask_size(info):
    w = min(info["w"], MASK_WIDTH)
    return w, max(2, round(info["h"] * w / info["w"]))


def summary(project):
    return {"id": project["id"], "name": project["name"], "updated": project.get("updated", 0),
            "seconds": project["video"]["seconds"], "w": project["video"]["w"], "h": project["video"]["h"]}


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


def create(upload, original_name):
    """A new project from an uploaded video (`upload` is a file we wrote; it is moved into the project)."""
    ext = os.path.splitext(original_name or "")[1].lower()
    if ext not in VIDEO_TYPES:
        raise UserError("Hãy chọn một file video (mp4, mov, mkv, webm, avi).")
    info = video.probe(upload)
    if info["frames"] > MAX_FRAMES:
        raise UserError(f"Video dài {info['seconds']:.0f} giây. Bokin FX dành cho video loop ngắn "
                        f"(tối đa {MAX_FRAMES // 24} giây ở 24 khung/giây).")
    pid = secrets.token_hex(6)
    path = os.path.join(paths.projects_dir(), pid)
    os.makedirs(os.path.join(path, "masks"))
    os.makedirs(os.path.join(path, "files"))
    shutil.move(upload, os.path.join(path, "input" + ext))
    info["file"] = "input" + ext
    now = time.time()
    project = {"id": pid, "name": clean_name(original_name), "source_name": os.path.basename(original_name)[:200],
               "created": now, "rev": 0, "video": info, "still": {"source": "frame", "frame": 0},
               "freeze": {"feather": 4}, "fg": {"feather": 2}, "layers": [], "preview": None}
    return save(project)


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


def _layer_from_page(raw, old):
    if not isinstance(raw, dict):
        raise UserError("Dữ liệu lớp không hợp lệ.")
    lid, kind = raw.get("id"), raw.get("kind")
    if not isinstance(lid, str) or not LAYER_ID.match(lid) or lid in ("freeze", "fg"):
        raise UserError("Mã lớp không hợp lệ.")
    if kind not in effects.SCHEMA:
        raise UserError("Loại lớp không hợp lệ.")
    name = re.sub(r"[\x00-\x1f]", "", str(raw.get("name") or effects.SCHEMA[kind]["label"]))[:40]
    layer = {"id": lid, "kind": kind, "name": name, "on": bool(raw.get("on", True)),
             "behind": bool(raw.get("behind", False)), "feather": _int(raw.get("feather"), 0, 200, 0),
             "seed": _int(raw.get("seed"), 0, 2 ** 31 - 1, 1), "params": effects.clean_params(kind, raw.get("params"))}
    if kind == "file" and old and old.get("kind") == "file" and old.get("file"):
        layer["file"] = old["file"]  # the page never names files: the stored one stays
    return layer


def _int(value, lo, hi, default):
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError, OverflowError):
        return default


def apply(project, data):
    """Takes the settings the page sent (layers, still, softness, name) and returns the saved project."""
    if not isinstance(data, dict):
        raise UserError("Dữ liệu không hợp lệ.")
    with LOCK:
        project = load(project["id"])  # the newest copy: uploads and previews may have saved since it was read
        if "name" in data:
            project["name"] = clean_name(data["name"], project["name"])
        still = data.get("still")
        if isinstance(still, dict):
            source = "image" if still.get("source") == "image" and project.get("still", {}).get("image_name") else "frame"
            project["still"].update({"source": source,
                                     "frame": _int(still.get("frame"), 0, project["video"]["frames"] - 1, 0)})
        for name in ("freeze", "fg"):
            if isinstance(data.get(name), dict):
                project[name] = {"feather": _int(data[name].get("feather"), 0, 200, 0)}
        if "layers" in data:
            raw = data["layers"]
            if not isinstance(raw, list) or len(raw) > MAX_LAYERS:
                raise UserError(f"Tối đa {MAX_LAYERS} lớp.")
            old = {lay["id"]: lay for lay in project["layers"]}
            layers = [_layer_from_page(r, old.get(r.get("id")) if isinstance(r, dict) else None) for r in raw]
            if len({lay["id"] for lay in layers}) != len(layers):
                raise UserError("Hai lớp trùng mã.")
            project["layers"] = layers
            for lid in set(old) - {lay["id"] for lay in layers}:
                _remove(os.path.join(paths.projects_dir(), project["id"], "masks", lid + ".png"))
        saved = save(project)
        sweep_files(saved)
        return saved


def sweep_files(project):
    """Removes layer files no layer uses any more. A file Windows still holds open (a render using it) stays
    until a later sweep."""
    path = os.path.join(paths.projects_dir(), project["id"], "files")
    keep = {(lay.get("file") or {}).get("stored") for lay in project["layers"]}
    try:
        names = os.listdir(path)
    except OSError:
        return
    for name in names:
        if name not in keep:
            _remove(os.path.join(path, name))


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


def set_still_image(project, upload, original_name):
    """The original picture for the still areas: fitted (cover, centred) to the video's size. Returns a
    warning when it does not look like the video's own frames."""
    info = project["video"]
    try:
        with Image.open(upload) as im:
            if im.width * im.height > overlay.MAX_IMAGE_PIXELS:
                raise UserError("Ảnh quá lớn.")
            im = ImageOps.exif_transpose(im).convert("RGB")
            fitted = ImageOps.fit(im, (info["w"], info["h"]), Image.Resampling.LANCZOS)
    except UserError:
        raise
    except Exception as e:  # noqa: BLE001
        raise UserError("Không đọc được ảnh này. Hãy chọn ảnh PNG hoặc JPG.") from e
    path = os.path.join(paths.projects_dir(), project["id"])
    fitted.save(os.path.join(path, "still.png"), compress_level=3)
    _remove(upload)
    first = video.read_frame(os.path.join(path, info["file"]), info, 0, (320, max(2, round(320 * info["h"] / info["w"]))))
    small = np.asarray(fitted.resize((first.shape[1], first.shape[0]), Image.Resampling.BOX), np.float32)
    diff = float(np.median(np.abs(small - first.astype(np.float32)).mean(axis=2)))
    warning = None
    if diff > 18:
        warning = ("Ảnh gốc có vẻ khác với video (lệch vị trí, khung hình hoặc màu). Kiểm tra lại, "
                   "hoặc dùng một khung của video cho vùng đứng yên.")
    with LOCK:
        fresh = load(project["id"])
        fresh["still"].update({"source": "image", "image_name": os.path.basename(original_name)[:200],
                               "warning": warning})
        return save(fresh)


def set_layer_file(project, lid, upload, original_name):
    """The outside file of a "file" layer: checked, kept in the project, its blend set to suit it."""
    if not any(lay["id"] == lid and lay["kind"] == "file" for lay in project["layers"]):
        raise UserError("Không tìm thấy lớp để gắn file.")
    ext = os.path.splitext(original_name or "")[1].lower()
    if ext not in overlay.IMAGE_TYPES | overlay.VIDEO_TYPES:
        raise UserError("Hãy chọn video (mp4, mov, webm…) hoặc ảnh (png, jpg).")
    src = _with_ext(upload, ext)
    try:
        info = overlay.probe_file(src)
        with LOCK:  # the newest copy, and the project still there (it may have been deleted meanwhile)
            fresh = load(project["id"])
            layer = next((lay for lay in fresh["layers"] if lay["id"] == lid and lay["kind"] == "file"), None)
            if layer is None:
                raise UserError("Không tìm thấy lớp để gắn file.")
            path = os.path.join(paths.projects_dir(), fresh["id"], "files")
            os.makedirs(path, exist_ok=True)
            stored = f"{lid}-{secrets.token_hex(3)}{ext}"
            shutil.move(src, os.path.join(path, stored))
            layer["file"] = {"stored": stored, "name": os.path.basename(original_name)[:200], "info": info}
            layer["params"]["blend"] = "normal" if info.get("alpha") else "screen"
            if not info.get("image"):
                layer["params"]["start"] = 0
            saved = save(fresh)
            sweep_files(saved)
            return saved
    finally:
        _remove(src)


def _with_ext(upload, ext):
    """The upload renamed to end in its type's extension (what tells pictures from videos here)."""
    if upload.lower().endswith(ext):
        return upload
    os.replace(upload, upload + ext)
    return upload + ext


def set_preview(project, name):
    with LOCK:
        fresh = load(project["id"])
        old = fresh.get("preview")
        fresh["preview"] = name
        save(fresh, bump=False)
    if old and old != name:
        _remove(os.path.join(paths.projects_dir(), project["id"], old))
    return fresh
