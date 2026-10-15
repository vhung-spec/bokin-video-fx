"""The effects library: outside videos and pictures (effects on black, with transparency, PNG overlays), kept once
in the data folder and used by the clips of any project.

library/<id>/item.json     what the file is (written whole, atomically)
library/<id>/source.<ext>  the file, as imported
library/<id>/thumb.jpg     a small picture for the library panel
library/<id>/proxy.*       a light copy the page plays in its live preview (made in the background for videos)

Imported files are untrusted: they are read only by our own code (Pillow, ffmpeg), the page names items by id
only, and an item a project still uses cannot be deleted.
"""
import json
import os
import re
import secrets
import shutil
import threading
import time

import numpy as np
from PIL import Image, ImageOps

from . import overlay, paths, video

ITEM_ID = re.compile(r"^[0-9a-f]{10}$")
TYPES = overlay.IMAGE_TYPES | overlay.VIDEO_TYPES
PROXY_WIDTH = 1280
THUMB_WIDTH = 320
LOCK = threading.RLock()


class LibraryError(Exception):
    """Something the user can fix; the message is shown on the page as is."""


def root():
    path = os.path.join(paths.data_dir(), "library")
    os.makedirs(path, exist_ok=True)
    return path


def folder(iid):
    if not isinstance(iid, str) or not ITEM_ID.match(iid):
        raise LibraryError("Không tìm thấy hiệu ứng này trong thư viện.")
    path = os.path.join(root(), iid)
    if not os.path.isfile(os.path.join(path, "item.json")):
        raise LibraryError("Không tìm thấy hiệu ứng này trong thư viện.")
    return path


def load(iid):
    with LOCK, open(os.path.join(folder(iid), "item.json"), encoding="utf-8") as f:
        return json.load(f)


def save(item):
    with LOCK:
        path = os.path.join(root(), item["id"], "item.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(item, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    return item


def source_path(item):
    return os.path.join(root(), item["id"], item["file"])


def proxy_path(item):
    return os.path.join(root(), item["id"], item["proxy"]) if item.get("proxy") else None


def list_all():
    out = []
    for iid in os.listdir(root()):
        if ITEM_ID.match(iid):
            try:
                out.append(public(load(iid)))
            except (OSError, ValueError, KeyError, LibraryError):
                continue
    return sorted(out, key=lambda i: -i["created"])


def public(item):
    return {k: item[k] for k in ("id", "name", "created", "info", "proxy", "size") if k in item}


def _clean_name(name):
    import unicodedata
    stem = os.path.splitext(os.path.basename(str(name or "")))[0]
    stem = unicodedata.normalize("NFC", stem)
    stem = re.sub(r"[\x00-\x1f<>:\"/\\|?*]+", " ", stem)
    return re.sub(r"\s+", " ", stem).strip()[:80] or "Hiệu ứng"


def add(upload, original_name):
    """A new library item from an uploaded file (`upload` is a file we wrote; it is moved into the library)."""
    ext = os.path.splitext(original_name or "")[1].lower()
    if ext not in TYPES:
        raise LibraryError("Hãy chọn video (mp4, mov, webm…) hoặc ảnh (png, jpg).")
    src = upload if upload.lower().endswith(ext) else upload + ext
    if src != upload:
        os.replace(upload, src)
    try:
        info = overlay.probe_file(src)
    except video.VideoError as e:
        _remove(src)
        raise LibraryError(str(e)) from e
    iid = secrets.token_hex(5)
    path = os.path.join(root(), iid)
    os.makedirs(path)
    try:
        stored = "source" + ext
        shutil.move(src, os.path.join(path, stored))
        item = {"id": iid, "name": _clean_name(original_name), "original": os.path.basename(original_name)[:200],
                "created": time.time(), "file": stored, "info": info, "proxy": None,
                "size": os.path.getsize(os.path.join(path, stored))}
        _thumb(item)
        if info.get("image"):
            _image_proxy(item)
        return save(item)
    except BaseException:
        shutil.rmtree(path, ignore_errors=True)
        raise


def _on_dark(rgba):
    """A picture with transparency shown over a dark checkerboard (as editors show see-through parts)."""
    h, w = rgba.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    board = np.where(((yy // 12) + (xx // 12)) % 2 == 0, 46, 34).astype(np.float32)[..., None]
    a = rgba[..., 3:4].astype(np.float32) / 255
    return (rgba[..., :3] * a + board * (1 - a) + 0.5).astype(np.uint8)


def _thumb(item):
    info = item["info"]
    w = min(THUMB_WIDTH, info["w"])
    size = (max(2, w), max(2, round(info["h"] * w / info["w"])))
    if info.get("image"):
        with Image.open(source_path(item)) as im:
            arr = np.asarray(ImageOps.exif_transpose(im).convert("RGBA").resize(size, Image.Resampling.LANCZOS))
    else:
        with video.Frames(source_path(item), info, size, start=info["frames"] // 2, count=1, alpha=True) as frames:
            arr = next(iter(frames), None)
        if arr is None:
            raise LibraryError("Không đọc được video này.")
    Image.fromarray(_on_dark(arr)).save(os.path.join(root(), item["id"], "thumb.jpg"), quality=84)


def _image_proxy(item):
    """A picture the page shows as is: PNG (keeps transparency), at most PROXY_WIDTH wide."""
    info = item["info"]
    w = min(PROXY_WIDTH, info["w"])
    with Image.open(source_path(item)) as im:
        im = ImageOps.exif_transpose(im).convert("RGBA")
        if im.width != w:
            im = im.resize((w, max(1, round(im.height * w / im.width))), Image.Resampling.LANCZOS)
        im.save(os.path.join(root(), item["id"], "proxy.png"), compress_level=3)
    item["proxy"] = "proxy.png"


def make_proxy(iid):
    """Makes the light copy of a library video (run in the background); returns the item."""
    item = load(iid)
    if item.get("proxy"):
        return item
    name = os.path.basename(video.make_proxy(source_path(item), item["info"],
                                             os.path.join(root(), iid, "proxy"), PROXY_WIDTH))
    with LOCK:
        try:
            fresh = load(iid)
        except LibraryError:  # deleted meanwhile
            return item
        fresh["proxy"] = name
        return save(fresh)


def delete(iid, users):
    """Removes an item; `users` are the names of the projects whose clips still use it (then it stays)."""
    path = folder(iid)
    if users:
        raise LibraryError("Hiệu ứng này đang được dùng trong dự án: " + ", ".join(users[:3])
                           + ("…" if len(users) > 3 else "") + ". Xoá các clip dùng nó trước.")
    with LOCK:
        for _ in range(5):
            try:
                shutil.rmtree(path)
                return
            except OSError:
                time.sleep(0.4)  # Windows: a file still open for a moment
        shutil.rmtree(path, ignore_errors=True)


def rename(iid, name):
    with LOCK:
        item = load(iid)
        item["name"] = _clean_name(name + ".x")
        return save(item)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass
