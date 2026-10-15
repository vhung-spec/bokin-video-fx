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
            return next((j for j in self.items.values() if j["state"] in ("queued", "running") and test(j)), None)

    def export(self, pid):
        project = projects.load(pid)
        return self._new("export", pid, "export", seconds=project.get("length") or timeline.DEFAULT_LENGTH)

    def project_proxy(self, pid):
        """The light copy of a project's video, unless it is there or being made."""
        project = projects.load(pid)
        if project.get("proxy") and os.path.exists(os.path.join(projects.folder(pid), project["proxy"])):
            return None
        return self._pending(lambda j: j["kind"] == "proxy" and j["project"] == pid) or \
            self._new("proxy", pid, "prep")

    def library_proxy(self, iid):
        item = library.load(iid)
        if item.get("proxy"):
            return None
        return self._pending(lambda j: j["kind"] == "libproxy" and j["item"] == iid) or \
            self._new("libproxy", None, "prep", item=iid)

    def fx_clip(self, pid, lid, name):
        """The preview clip `name` of built-in effect clip `lid`; ones still waiting for it with older settings
        are dropped."""
        same = self._pending(lambda j: j["kind"] == "fxclip" and j["project"] == pid and j["name"] == name)
        if same:
            return same
        with self.lock:
            for j in self.items.values():
                if j["kind"] == "fxclip" and j["project"] == pid and j["layer"] == lid and j["state"] == "queued":
                    j["cancel"] = True
        return self._new("fxclip", pid, "prep", layer=lid, name=name)

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
        project = projects.load(job["project"])
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
        try:
            name = engine.make_base_proxy(projects.folder(pid), project)
            projects.set_proxy(pid, name)
        except Exception:
            if job["cancel"]:  # the project was deleted meanwhile
                return False
            raise
        job["output"] = name
        return True

    def _libproxy(self, job):
        job["output"] = library.make_proxy(job["item"]).get("proxy")
        return True

    def _fxclip(self, job):
        pid, lid = job["project"], job["layer"]
        try:
            project = projects.load(pid)
        except projects.UserError:
            return False  # deleted meanwhile
        if not any(lay["id"] == lid for lay in project["layers"]):
            return False
        folder = projects.folder(pid)
        name = engine.fx_clip(folder, project, lid, cancelled=lambda: job["cancel"], progress=self._progress(job))
        if name is None:
            return False
        job["output"] = name
        try:
            projects.sweep_files(projects.load(pid))  # clips made for older settings
        except projects.UserError:
            pass
        return True
