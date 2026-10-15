"""Work that takes a while, done in the background, in two queues so the page never waits behind an export:

- export: the project's video, its whole length (30 s or 1 min), into the output folder;
- prep:   what the page's live preview plays: light copies of project videos and library videos, and preview
          clips of built-in effects.
"""
import os
import queue
import secrets
import shutil
import threading
import time
import traceback

from . import engine, library, paths, projects, timeline, video


def length_tag(seconds):
    if seconds < 60:
        return f"{seconds:g}s"
    h, rest = divmod(int(round(seconds)), 3600)
    m = rest // 60
    return (f"{h}h" if h else "") + (f"{m}m" if m else "")


def free_name(folder, stem):
    path = os.path.join(folder, stem + ".mp4")
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem}-{n}.mp4")
        n += 1
    return path


class Jobs:
    def __init__(self):
        self.items = {}
        self.order = []
        self.lock = threading.Lock()
        self.queues = {"export": queue.Queue(), "prep": queue.Queue()}
        for kind, q in self.queues.items():
            threading.Thread(target=self._work, args=(q,), daemon=True, name=f"fx-{kind}").start()

    def _new(self, kind, pid, queue_name, **extra):
        job = {"id": secrets.token_hex(4), "kind": kind, "project": pid, "state": "queued", "progress": 0.0,
               "message": "", "created": time.time(), "started": None, "finished": None, "output": None,
               "cancel": False, **extra}
        with self.lock:
            self.items[job["id"]] = job
            self.order.append(job["id"])
            while len(self.order) > 60:
                old = self.order.pop(0)
                if self.items[old]["state"] in ("queued", "running"):
                    self.order.insert(0, old)
                    break
                self.items.pop(old, None)
        self.queues[queue_name].put(job)
        return job

    def _pending(self, test):
        with self.lock:
            return next((j for j in self.items.values()
                         if j["state"] in ("queued", "running") and not j["cancel"] and test(j)), None)

    def _failed_lately(self, test, seconds=600):
        """A job like this failed in the last `seconds`: it is not tried again on every poll of the page."""
        now = time.time()
        with self.lock:
            return next((j for j in self.items.values() if j["state"] == "error" and test(j)
                         and now - (j["finished"] or now) < seconds), None)

    def export(self, pid):
        project = projects.load(pid)
        return self._new("export", pid, "export", seconds=project.get("length") or timeline.DEFAULT_LENGTH)

    def project_proxy(self, pid):
        """The light copy of a project's video, unless it is there or being made."""
        project = projects.load(pid)
        if project.get("proxy") and os.path.exists(os.path.join(projects.folder(pid), project["proxy"])):
            return None
        source = project["video"]["file"]  # a job for a video since replaced is not this one's
        mine = lambda j: j["kind"] == "proxy" and j["project"] == pid and j.get("source") == source  # noqa: E731
        return self._pending(mine) or self._failed_lately(mine) or self._new("proxy", pid, "prep", source=source)

    def library_proxy(self, iid):
        item = library.load(iid)
        if item.get("proxy"):
            return None
        mine = lambda j: j["kind"] == "libproxy" and j["item"] == iid  # noqa: E731
        return self._pending(mine) or self._failed_lately(mine) or self._new("libproxy", None, "prep", item=iid)

    def library_reverse(self, iid):
        """The played-backwards copy of a library video, unless it is there or being made."""
        if library.reversed_copy(library.load(iid)):
            return None
        mine = lambda j: j["kind"] == "librev" and j["item"] == iid  # noqa: E731
        return self._pending(mine) or self._failed_lately(mine) or self._new("librev", None, "prep", item=iid)

    def fx_clip(self, pid, lid, name, wanted):
        """The preview clip `name` of built-in effect clip `lid`. Clips still waiting that no clip of the project
        wants any more (`wanted`: the names its clips need now) are dropped."""
        with self.lock:
            for j in self.items.values():
                if j["kind"] == "fxclip" and j["project"] == pid and j["state"] == "queued" and j["name"] not in wanted:
                    j["cancel"] = True
        return self._pending(lambda j: j["kind"] == "fxclip" and j["project"] == pid and j["name"] == name) or \
            self._new("fxclip", pid, "prep", layer=lid, name=name)

    def cancel(self, jid):
        with self.lock:
            job = self.items.get(jid)
            if job and job["state"] in ("queued", "running"):
                job["cancel"] = True
                return True
        return False

    def busy(self):
        """An export is waiting or running (quitting now would lose it)."""
        with self.lock:
            return any(j["state"] in ("queued", "running") and j["kind"] == "export" for j in self.items.values())

    def public(self):
        with self.lock:
            return [{k: v for k, v in self.items[j].items() if k != "cancel"} for j in reversed(self.order)]

    def _work(self, q):
        while True:
            job = q.get()
            if job["cancel"]:
                job["state"], job["finished"] = "cancelled", time.time()
                continue
            job["state"], job["started"] = "running", time.time()
            try:
                done = getattr(self, "_" + job["kind"])(job)
                job["state"] = "done" if done else "cancelled"
                job["progress"] = 1.0 if done else job["progress"]
            except (projects.UserError, video.VideoError, library.LibraryError) as e:
                job["state"], job["message"] = "error", str(e)
            except Exception as e:  # noqa: BLE001 - shown on the page, written to the log
                traceback.print_exc()
                job["state"] = "error"
                job["message"] = "Có lỗi khi làm video. Xem nhật ký (app.log) để biết thêm." if not isinstance(
                    e, OSError) else f"Lỗi đọc/ghi file: {e.strerror or e}"
            job["finished"] = time.time()
            if job["kind"] == "export" or job["state"] == "error":
                print(f"job {job['kind']} {job['id']} {job['state']} {job['message']}".rstrip(), flush=True)

    def _progress(self, job):
        def report(fraction):
            job["progress"] = max(0.0, min(1.0, fraction))
        return report

    def _export(self, job):
        if not self._reversed_copies(job):
            return False
        project = projects.load(job["project"])  # (as it is now: the page may have changed it meanwhile)
        folder = projects.folder(job["project"])
        info = project["video"]
        total = timeline.project_frames(project)
        out_dir = paths.out_dir()
        need = total * info["w"] * info["h"] * 0.2 + 200e6  # the pieces, then the joined video
        free = shutil.disk_usage(out_dir).free
        if free < need:
            raise projects.UserError(f"Ổ đĩa không đủ chỗ: cần khoảng {need / 1e9:.1f} GB, còn {free / 1e9:.1f} GB. "
                                     "Hãy dọn bớt ổ đĩa.")
        seconds = total / video.fps_value(info["fps"])
        final = free_name(out_dir, f"{project['name']}-fx-{length_tag(round(seconds))}")
        made = os.path.join(folder, f"export-{job['id']}.mp4")
        try:
            if not engine.render(folder, project, made, progress=self._progress(job), cancelled=lambda: job["cancel"]):
                return False
            shutil.move(made, final)
            job["output"] = os.path.basename(final)
            return True
        finally:
            try:
                os.remove(made)
            except OSError:
                pass

    def _proxy(self, job):
        pid = job["project"]
        project = projects.load(pid)
        if project["video"]["file"] != job.get("source", project["video"]["file"]):
            return False  # the video was replaced before this started
        try:
            name = engine.make_base_proxy(projects.folder(pid), project)
            kept = projects.set_proxy(pid, name, project["video"]["file"]).get("proxy") == name
        except Exception:
            if job["cancel"]:  # the project was deleted meanwhile
                return False
            raise
        if not kept:
            return False  # the video was replaced meanwhile: this copy was of the old one, and is gone
        job["output"] = name
        return True

    def _reversed_copies(self, job):
        """Before an export: the played-backwards copies its clips need (made once, kept in the library).
        False if cancelled."""
        project = projects.load(job["project"])
        for lay in engine.clip_order(project):
            lib = (lay.get("file") or {}).get("lib")
            if lay["kind"] != "file" or not lib or not (lay.get("params") or {}).get("reverse"):
                continue
            try:
                if library.reversed_copy(library.load(lib)):
                    continue
            except library.LibraryError:
                continue  # gone from the library: the clip draws nothing
            job["message"] = "Đang chuẩn bị bản phát ngược…"
            try:
                made = library.make_reverse(lib, cancelled=lambda: job["cancel"], progress=self._progress(job))
            except library.LibraryError:
                made = True
            except Exception as e:
                traceback.print_exc()
                raise projects.UserError(f"Không làm được bản phát ngược của clip \"{lay.get('name', '')}\". "
                                         "Hãy bỏ ô Phát ngược của clip đó rồi xuất lại.") from e
            if made is None and job["cancel"]:
                return False
            job["message"], job["progress"], job["started"] = "", 0.0, time.time()  # (the time left: rendering only)
        return True

    def _librev(self, job):
        return library.make_reverse(job["item"], cancelled=lambda: job["cancel"], progress=self._progress(job)) is not None

    def _libproxy(self, job):
        job["output"] = library.make_proxy(job["item"]).get("proxy")
        return True

    def _fxclip(self, job):
        pid = job["project"]
        try:
            project = projects.load(pid)
        except projects.UserError:
            return False  # deleted meanwhile
        # a clip that still wants this job's preview clip (the one it was asked for may have other settings now)
        lay = next((lay for lay in project["layers"]
                    if lay["kind"] != "file" and engine.fx_clip_name(project, lay) == job["name"]), None)
        if lay is None:
            return False
        folder = projects.folder(pid)
        name = engine.fx_clip(folder, project, lay["id"], cancelled=lambda: job["cancel"],
                              progress=self._progress(job))
        if name is None:
            return False
        job["output"] = name
        try:
            projects.sweep_files(project)  # clips made for older settings (on the newest copy of the project)
        except projects.UserError:
            pass
        return True
