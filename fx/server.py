"""The app's local web server: the page (index.html) and its JSON API under /api/.

Only this computer's own page may use it: a request must name 127.0.0.1 or localhost as its host (so a web site
cannot reach it through DNS rebinding) and every POST must carry the header X-Bokin: 1, which another web site
cannot add without asking first (a CORS preflight, never allowed here). Nothing from the page is used as a path:
projects, library items, masks and files are found by ids and names checked against fixed patterns, and uploads
are written by this code into the data folder.
"""
import io
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

import bokin_license

from . import colourmatch, effects, engine, guide, jobs, keys, library, licensing, overlay, paths, projects, timeline, video

UPLOAD_LIMITS = {"video": 4 << 30, "replace": 4 << 30, "library": 4 << 30, "still": 200 << 20}
UPLOAD_TYPES = {"video": (projects.VIDEO_TYPES, "Hãy chọn một file video (mp4, mov, mkv, webm, avi)."),
                "replace": (projects.VIDEO_TYPES, "Hãy chọn một file video (mp4, mov, mkv, webm, avi)."),
                "still": (overlay.IMAGE_TYPES, "Hãy chọn ảnh PNG hoặc JPG."),
                "library": (library.TYPES, "Hãy chọn video (mp4, mov, webm…) hoặc ảnh (png, jpg).")}
MASK_LIMIT = 50 << 20
JSON_LIMIT = 1 << 20
OUT_NAME = re.compile(r"^[^\\/:*?\"<>|\x00-\x1f]{1,180}\.mp4$")
FX_NAME = re.compile(r"^fx-[0-9a-f]{12}\.(webm|mp4)$")
PROXY_NAME = re.compile(r"^proxy\.(mp4|webm|png)$")  # a library item's
REV_PROXY_NAME = re.compile(r"^proxy-rev\.(mp4|webm)$")  # its played-backwards light copy
MEDIA_TYPES = {".mp4": "video/mp4", ".webm": "video/webm", ".png": "image/png", ".jpg": "image/jpeg"}
ERRORS = (projects.UserError, video.VideoError, library.LibraryError, bokin_license.LicenseError,
          colourmatch.MatchError)
# What answers before the app is activated (licensing): the page and its activation screen, the installer's and the
# updater's questions. Everything else is refused.
FREE_GET = {"/", "/index.html", "/api/ping", "/api/license", "/api/state", "/api/guide/video", "/media/guide"}
FREE_POST = {"/api/license", "/api/update/check", "/api/update", "/api/quit"}


class App:
    """What the request handlers share."""

    def __init__(self, desktop=True):
        self.desktop = desktop
        self.jobs = jobs.Jobs()
        self.looks = engine.Looks()
        self.updater = None
        self.seen = time.time()
        self.quit = threading.Event()
        self.relaunch = False
        self.uploads = 0  # files being received: quitting now would cut them off
        self.counter = threading.Lock()

    def busy(self):
        return self.jobs.busy() or self.uploads > 0

    def uploading(self, step):
        with self.counter:
            self.uploads += step

    def state(self):
        up = self.updater.state if self.updater else {"current": paths.version(), "available": None,
                                                       "ready": False, "error": None, "checked": None}
        return {"app": paths.APP_ID, "version": paths.version(), "update": up, "jobs": self.jobs.public(),
                "projects": projects.list_all(), "library": library.list_all(), "out_dir": paths.out_dir(),
                "desktop": self.desktop}


def _jpeg(arr, quality=88):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def open_in_explorer(path, select=False):
    if os.name == "nt":
        if select:
            subprocess.Popen(["explorer", "/select,", path])
        else:
            os.startfile(path)  # noqa: S606 - our own output folder
    else:
        subprocess.Popen(["xdg-open", os.path.dirname(path) if select else path])


class Handler(BaseHTTPRequestHandler):
    server_version = "BokinFX"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    @property
    def app(self):
        return self.server.app

    def log_message(self, fmt, *args):  # quiet: errors are printed where they happen
        pass

    # --- answers -------------------------------------------------------------------------------------------

    def send(self, code, body, kind="application/json; charset=utf-8", headers=(), cache="no-store"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")  # never inside another page's frame (clickjacking)
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def fail(self, code, message):
        # close the connection: a body left unread must never be read as a second request
        self.send(code, {"error": message}, headers=(("Connection", "close"),))

    def refused(self):
        """Not activated (or the key ran out meanwhile): the page shows its activation screen."""
        self.send(403, {"error": "Bokin FX chưa được kích hoạt trên máy này.", "license": True},
                  headers=(("Connection", "close"),))

    def send_file(self, path, kind):
        """A file, with byte ranges (the video player asks for parts)."""
        try:
            size = os.path.getsize(path)
            f = open(path, "rb")
        except OSError:
            return self.fail(404, "Không tìm thấy file.")
        with f:
            start, end = 0, size - 1
            m = re.match(r"bytes=(\d*)-(\d*)$", self.headers.get("Range") or "")
            if m and (m.group(1) or m.group(2)):
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "frame-ancestors 'none'")
            self.end_headers()
            if self.command == "HEAD":
                return
            f.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
            except (ConnectionError, OSError):
                pass  # the player stopped reading

    # --- who may ask -----------------------------------------------------------------------------------------

    def allowed(self, post):
        host = (self.headers.get("Host") or "").strip().lower()
        name = host.rsplit(":", 1)[0] if not host.startswith("[") else host.split("]")[0] + "]"
        if name not in ("127.0.0.1", "localhost"):
            return False
        return not post or self.headers.get("X-Bokin") == "1"

    # --- requests --------------------------------------------------------------------------------------------

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self.allowed(False):
            return self.fail(403, "forbidden")
        url = urllib.parse.urlsplit(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        try:
            self.route_get(url.path, q)
        except ERRORS as e:
            self.fail(400, str(e))
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self.fail(500, "Có lỗi trong app. Xem nhật ký (app.log).")

    def route_get(self, path, q):
        if path in ("/", "/index.html"):
            with open(os.path.join(paths.resource_dir(), "index.html"), "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/ping":
            return self.send(200, {"app": paths.APP_ID, "version": paths.version(), "owner": paths.owner()})
        if path.startswith("/api/"):
            self.app.seen = time.time()
        if path not in FREE_GET and not licensing.licensed():
            return self.refused()
        if path == "/api/license":
            return self.send(200, licensing.view())
        if path == "/api/state":
            return self.send(200, self.app.state())
        if path == "/api/schema":
            return self.send(200, {"kinds": effects.SCHEMA, "lengths": list(timeline.LENGTHS),
                                   "min_edge_fade": timeline.MIN_EDGE_FADE, "max_fade": timeline.MAX_FADE,
                                   "max_mix": timeline.MAX_MIX,
                                   "blend": {k: c.blend for k, c in effects.KINDS.items()},
                                   "lens": {k: c.lens for k, c in effects.KINDS.items()},
                                   "lit": {k: True for k, c in effects.KINDS.items() if c.LIT},
                                   "placed": {k: list(c.BOX) for k, c in effects.KINDS.items() if c.BOX},
                                   "under": {k: True for k, c in effects.KINDS.items() if c.UNDER},
                                   # only inside the clip's painted area (no area: nothing)
                                   "area": {k: True for k, c in effects.KINDS.items() if c.AREA},
                                   # turned in 3D like a file clip; on a sheet of the frame's shape (not a box)
                                   "turns": {k: True for k, c in effects.KINDS.items() if c.TURNS},
                                   "sheet": {k: True for k, c in effects.KINDS.items() if c.TURNS and not c.BOX},
                                   "max_clips": projects.MAX_LAYERS, "max_tracks": projects.MAX_TRACKS})
        if path == "/api/keys":
            return self.send(200, keys.view())
        if path == "/api/guide":  # the owner's guide (Markdown), shown in the page's Phím tắt window
            with open(keys.guide_path(), "rb") as f:
                return self.send(200, f.read(), "text/markdown; charset=utf-8")
        if path == "/api/guide/video":  # the video guide's chapters (also before activation: how to activate)
            return self.send(200, guide.view())
        if path == "/media/guide":  # the video guide itself (HUONG-DAN.mp4, bundled)
            return self.send_file(guide.video_path(), "video/mp4")
        if path == "/api/project":
            return self.send(200, self.project_view(projects.load(q.get("id"))))
        if path == "/api/frame":  # a frame of the video at the masks' size, to paint on
            project = projects.load(q.get("id"))
            info = project["video"]
            arr = video.read_frame(engine.input_path(projects.folder(project["id"]), project), info,
                                   _int(q.get("t")), projects.mask_size(info))
            return self.send(200, _jpeg(arr, 90), "image/jpeg")
        if path == "/api/still":
            project = projects.load(q.get("id"))
            p = os.path.join(projects.folder(project["id"]), "still.png")
            with Image.open(p) as im:
                arr = np.asarray(im.convert("RGB").resize(projects.mask_size(project["video"])))
            return self.send(200, _jpeg(arr, 90), "image/jpeg")
        if path == "/api/thumb":
            project = projects.load(q.get("id"))
            folder = projects.folder(project["id"])
            thumb = os.path.join(folder, "thumb.jpg")
            if not os.path.exists(thumb):
                info = project["video"]
                arr = video.read_frame(engine.input_path(folder, project), info, 0,
                                       (320, max(2, round(320 * info["h"] / info["w"] / 2) * 2)))
                with open(thumb, "wb") as f:
                    f.write(_jpeg(arr, 80))
            return self.send_file(thumb, "image/jpeg")
        if path == "/api/mask":
            project = projects.load(q.get("id"))
            name = q.get("name") or ""
            if not projects.MASK_NAME.match(name):
                return self.fail(404, "no mask")
            p = engine.mask_path(projects.folder(project["id"]), name)
            if not os.path.exists(p):
                return self.fail(404, "no mask")
            return self.send_file(p, "image/png")
        if path == "/api/look":  # one frame with every layer, as the video will be
            project = projects.load(q.get("id"))
            arr = self.app.looks.frame(projects.folder(project["id"]), project, _int(q.get("t")),
                                       max(160, min(1920, _int(q.get("w"), engine.LOOK_WIDTH))))
            return self.send(200, _jpeg(arr, 88), "image/jpeg", cache="no-cache")  # the page shows the image it loaded
        if path == "/media/base":  # the project's video for the live preview: its light copy
            project = projects.load(q.get("id"))
            name = project.get("proxy") or ""
            if not projects.PROXY_NAME.match(name):
                return self.fail(404, "no proxy yet")
            return self.send_file(os.path.join(projects.folder(project["id"]), name), MEDIA_TYPES[os.path.splitext(name)[1]])
        if path == "/media/fx":  # a built-in effect's preview clip
            project = projects.load(q.get("id"))
            name = q.get("name") or ""
            if not FX_NAME.match(name):
                return self.fail(404, "no such clip")
            return self.send_file(os.path.join(projects.folder(project["id"]), "clips", name),
                                  MEDIA_TYPES[os.path.splitext(name)[1]])
        if path == "/media/lib":  # a library item's small picture or light copy
            item = library.load(q.get("id"))
            if q.get("what") == "thumb":
                return self.send_file(os.path.join(library.folder(item["id"]), "thumb.jpg"), "image/jpeg")
            if q.get("what") == "reverse":
                rev = library.reversed_copy(item)
                name = rev["proxy"] if rev else ""
                if not REV_PROXY_NAME.match(name):
                    return self.fail(404, "not made yet")
                return self.send_file(os.path.join(library.folder(item["id"]), name), MEDIA_TYPES[os.path.splitext(name)[1]])
            name = item.get("proxy") or ""
            if not PROXY_NAME.match(name):
                return self.fail(404, "no proxy yet")
            return self.send_file(library.proxy_path(item), MEDIA_TYPES[os.path.splitext(name)[1]])
        if path == "/media/out":
            name = q.get("name") or ""
            if not OUT_NAME.match(name) or name.startswith("."):
                return self.fail(404, "no such video")
            return self.send_file(os.path.join(paths.out_dir(), name), "video/mp4")
        return self.fail(404, "not found")

    def project_view(self, project):
        folder = projects.folder(project["id"])
        masks = sorted(n[:-4] for n in os.listdir(os.path.join(folder, "masks")) if n.endswith(".png"))
        w, h = projects.mask_size(project["video"])
        # every setting a clip's effect has now, as it is drawn (one saved by an older version lacks the newer ones)
        layers = [{**lay, "params": effects.clean_params(lay["kind"], lay.get("params"))}
                  if lay.get("kind") in effects.SCHEMA else lay for lay in project.get("layers", [])]
        return {**project, "layers": layers, "masks": masks, "mask_w": w, "mask_h": h,
                "frames_out": timeline.project_frames(project),
                "has_still_image": os.path.exists(os.path.join(folder, "still.png"))}

    def do_POST(self):
        if not self.allowed(True):
            return self.fail(403, "forbidden")
        url = urllib.parse.urlsplit(self.path)
        q = {k: v[-1] for k, v in urllib.parse.parse_qs(url.query).items()}
        self.app.seen = time.time()
        try:
            self.route_post(url.path, q)
        except ERRORS as e:
            self.close_connection = True  # the body may not have been read
            self.fail(400, str(e))
        except (ConnectionError, BrokenPipeError):
            pass
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            self.close_connection = True
            self.fail(500, "Có lỗi trong app. Xem nhật ký (app.log).")

    def length(self, limit):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if n < 0 or n > limit:
            self.close_connection = True
            raise projects.UserError("File quá lớn." if n > limit else "Yêu cầu không hợp lệ.")
        return n

    def body_json(self):
        n = self.length(JSON_LIMIT)
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError as e:
            raise projects.UserError("Dữ liệu không hợp lệ.") from e
        if not isinstance(data, dict):
            raise projects.UserError("Dữ liệu không hợp lệ.")
        return data

    def receive(self, limit):
        """The request body, written to a new file in the data folder; returns its path."""
        n = self.length(limit)
        if n == 0:
            raise projects.UserError("File rỗng.")
        folder = os.path.join(paths.data_dir(), "tmp")
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, secrets.token_hex(8) + ".part")
        try:
            with open(path, "wb") as f:
                left = n
                while left > 0:
                    chunk = self.rfile.read(min(1 << 20, left))
                    if not chunk:
                        raise ConnectionError("upload cut short")
                    f.write(chunk)
                    left -= len(chunk)
        except BaseException:
            _remove(path)
            raise
        return path

    def route_post(self, path, q):
        app = self.app
        if path not in FREE_POST and not licensing.licensed():
            return self.refused()
        if path == "/api/license":  # a key pasted on the activation screen (or a new one, to renew)
            return self.send(200, licensing.activate(self.body_json().get("key")))
        if path == "/api/project/copy":  # "Lưu thành bản khác"
            data = self.body_json()
            project = projects.copy(data.get("id"), data.get("name"))
            return self.send(200, {"id": project["id"], "name": project["name"]})
        if path == "/api/keys":  # the owner's keyboard shortcuts
            keys.save(self.body_json().get("keys"))
            return self.send(200, keys.view())
        if path == "/api/upload":
            kind = q.get("kind")
            if kind not in UPLOAD_LIMITS:
                return self.fail(400, "Yêu cầu không hợp lệ.")
            stem, ext = os.path.splitext(os.path.basename(q.get("name") or ""))
            ext = ext[:16]
            name = stem[:200 - len(ext)] + ext  # a long name keeps its extension
            if ext.lower() not in UPLOAD_TYPES[kind][0]:  # before receiving gigabytes
                raise projects.UserError(UPLOAD_TYPES[kind][1])
            project = projects.load(q.get("id")) if kind in ("still", "replace") else None
            notes = []
            if kind == "replace":
                mine = [j for j in app.jobs.public() if j["project"] == project["id"] and j["state"] in ("queued", "running")]
                if any(j["kind"] == "export" for j in mine):
                    raise projects.UserError("Dự án đang được xuất video. Chờ xong rồi hãy đổi video gốc.")
            app.uploading(+1)
            try:
                upload = self.receive(UPLOAD_LIMITS[kind])
                try:
                    if kind == "video":
                        project = projects.create(upload, name)
                        app.jobs.project_proxy(project["id"])
                    elif kind == "still":
                        project = projects.set_still_image(project, upload, name)
                    elif kind == "replace":
                        for j in app.jobs.public():  # what was being prepared was for the old video
                            if j["project"] == project["id"] and j["kind"] in ("proxy", "fxclip"):
                                app.jobs.cancel(j["id"])
                        app.looks.forget(projects.folder(project["id"]))
                        project, notes = projects.replace_video(project, upload, name)
                        app.jobs.project_proxy(project["id"])
                    else:
                        item = library.add(upload, name)
                        if not item["info"].get("image"):
                            app.jobs.library_proxy(item["id"])
                        return self.send(200, library.public(item))
                finally:
                    _remove(upload)
                    _remove(upload + os.path.splitext(name)[1].lower())
            finally:
                app.uploading(-1)
            return self.send(200, {**self.project_view(project), "notes": notes})
        if path == "/api/mask":
            project = projects.load(q.get("id"))
            data = self.rfile.read(self.length(MASK_LIMIT))
            project = projects.save_mask(project, q.get("name"), data)
            return self.send(200, {"rev": project["rev"]})
        data = self.body_json()
        if path == "/api/save":
            project = projects.load(data.get("id"))
            app.looks.release(projects.folder(project["id"]))  # so a removed file layer's file can be deleted
            project = projects.apply(project, data)
            return self.send(200, self.project_view(project))
        if path == "/api/export":
            if not licensing.licensed(fresh=True):  # (a key with an end date: checked again before the work)
                return self.refused()
            app.looks.release(projects.folder(data.get("id")))
            return self.send(200, app.jobs.export(data.get("id")))
        if path == "/api/proxy":  # the light copy of the project's video, if it is not there yet
            job = app.jobs.project_proxy(data.get("id"))
            return self.send(200, {"job": job and job["id"]})
        if path == "/api/fxclip":  # the preview clip of a built-in effect, made if needed
            project = projects.load(data.get("id"))
            layer = next((lay for lay in project["layers"] if lay["id"] == data.get("layer")), None)
            if layer is None or layer["kind"] == "file":
                raise projects.UserError("Không tìm thấy clip.")
            name = engine.fx_clip_name(project, layer)
            length = project.get("length") or timeline.DEFAULT_LENGTH  # the page drops a clip of another length
            at = None  # where a clip made from the picture under its box was made for (it shows only there)
            if effects.KINDS[layer["kind"]].UNDER and effects.KINDS[layer["kind"]].BOX:  # (under an area: anywhere)
                p = effects.clean_params(layer["kind"], layer.get("params"))
                at = [p["x"], p["y"], p["scale"]] + [p.get(k, 0) for k in effects.TURN_KEYS]  # (and its turn)
            if os.path.exists(os.path.join(projects.folder(project["id"]), "clips", name)):
                return self.send(200, {"name": name, "ready": True, "length": length, "at": at})
            wanted = {engine.fx_clip_name(project, lay) for lay in project["layers"] if lay["kind"] != "file"}
            job = app.jobs.fx_clip(project["id"], layer["id"], name, wanted)
            return self.send(200, {"name": name, "ready": False, "job": job["id"], "length": length, "at": at})
        if path == "/api/match":  # "Khớp màu với cảnh": colour settings for a file clip, from the saved project
            project = projects.load(data.get("id"))
            layer = next((lay for lay in project["layers"] if lay["id"] == data.get("layer")), None)
            if layer is None:
                raise projects.UserError("Không tìm thấy clip.")
            return self.send(200, {"params": colourmatch.match(projects.folder(project["id"]), project, layer)})
        if path == "/api/copymask":  # a clip cut in two: the second half gets the first one's area
            project = projects.copy_mask(projects.load(data.get("id")), data.get("src"), data.get("dst"))
            return self.send(200, {"rev": project["rev"], "masks": self.project_view(project)["masks"]})
        if path == "/api/library/delete":
            iid = data.get("id")
            library.folder(iid)
            users = projects.using_library(iid)
            if not users:
                app.looks.release_all()
            library.delete(iid, users)
            return self.send(200, {"ok": True})
        if path == "/api/library/rename":
            item = library.rename(data.get("id"), str(data.get("name") or ""))
            return self.send(200, library.public(item))
        if path == "/api/library/proxy":
            job = app.jobs.library_proxy(data.get("id"))
            return self.send(200, {"job": job and job["id"]})
        if path == "/api/library/reverse":  # a clip set to play backwards: the item's played-backwards copy
            job = app.jobs.library_reverse(data.get("id"))
            return self.send(200, {"job": job and job["id"]})
        if path == "/api/cancel":
            return self.send(200, {"ok": app.jobs.cancel(str(data.get("job")))})
        if path == "/api/delete":
            pid = data.get("id")
            mine = [j for j in app.jobs.public() if j["project"] == pid and j["state"] in ("queued", "running")]
            if any(j["kind"] == "export" for j in mine):
                raise projects.UserError("Dự án đang được xuất video. Chờ xong hoặc huỷ trước khi xoá.")
            for j in mine:  # preparing its live preview: no longer needed
                app.jobs.cancel(j["id"])
            app.looks.forget(projects.folder(pid))
            projects.delete(pid)
            return self.send(200, {"ok": True})
        if path == "/api/open":
            if not app.desktop:
                return self.fail(400, "Không mở được thư mục trên máy chủ.")
            name = data.get("name")
            if name:
                if not isinstance(name, str) or not OUT_NAME.match(name):
                    return self.fail(400, "Yêu cầu không hợp lệ.")
                open_in_explorer(os.path.join(paths.out_dir(), name), select=True)
            else:
                open_in_explorer(paths.out_dir())
            return self.send(200, {"ok": True})
        if path == "/api/update/check":  # look for a newer version now (the window opened, or the version was clicked)
            started = bool(app.updater and app.updater.check_soon())
            return self.send(200, {**app.state()["update"], "started": started})
        if path == "/api/update":
            up = app.updater
            if not up or not up.state.get("ready") or not up.installer or not os.path.exists(up.installer):
                return self.fail(400, "Chưa có bản cập nhật sẵn sàng.")
            if app.busy():
                return self.fail(400, "App đang làm video. Cập nhật sau khi xong.")
            app.relaunch = True
            app.quit.set()
            return self.send(200, {"ok": True})
        if path == "/api/quit":  # the installer asks before replacing the files
            if app.busy():
                return self.send(200, {"busy": True})
            app.relaunch = None  # quit without installing anything ourselves
            app.quit.set()
            return self.send(200, {"ok": True})
        return self.fail(404, "not found")


def _int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False  # Windows would let a second app take the same port

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return  # the page went away mid-answer: nothing to report
        super().handle_error(request, client_address)


def start(app, port):
    """Serve on 127.0.0.1:port in a background thread; OSError if the port is taken."""
    server = Server(("127.0.0.1", port), Handler)
    server.app = app
    threading.Thread(target=server.serve_forever, daemon=True, name="fx-http").start()
    return server


if __name__ == "__main__":  # pragma: no cover - for trying the server alone
    srv = start(App(desktop=True), int(sys.argv[1]) if len(sys.argv) > 1 else 7890)
    print("http://127.0.0.1:%d" % srv.server_address[1])
    threading.Event().wait()
