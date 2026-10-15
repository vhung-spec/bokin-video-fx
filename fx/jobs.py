"""Work that takes a while, done in the background: previews (small, quick) and exports (the full video, then
copied end to end to the length asked for). Previews and exports have a worker each, so a preview never waits
behind a long export."""
import os
import queue
import secrets
import shutil
import threading
import time
import traceback

from . import engine, paths, projects, video

LENGTHS = [0, 30, 60, 300, 600, 1800, 3600, 7200, 10800, 21600, 36000, 43200]  # seconds; 0 = one loop
MAX_SECONDS = 12 * 3600


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
        self.queues = {"preview": queue.Queue(), "export": queue.Queue()}
        for kind, q in self.queues.items():
            threading.Thread(target=self._work, args=(q,), daemon=True, name=f"fx-{kind}").start()

    def _new(self, kind, pid, **extra):
        job = {"id": secrets.token_hex(4), "kind": kind, "project": pid, "state": "queued", "progress": 0.0,
               "message": "", "created": time.time(), "started": None, "finished": None, "output": None,
               "cancel": False, **extra}
        with self.lock:
            self.items[job["id"]] = job
            self.order.append(job["id"])
            while len(self.order) > 40:
                old = self.order.pop(0)
                if self.items[old]["state"] in ("queued", "running"):
                    self.order.insert(0, old)
                    break
                self.items.pop(old, None)
        return job

    def preview(self, pid):
        """A new preview of the project; an older one still queued or running for it is dropped."""
        projects.folder(pid)
        with self.lock:
            for job in self.items.values():
                if job["kind"] == "preview" and job["project"] == pid and job["state"] in ("queued", "running"):
                    job["cancel"] = True
        job = self._new("preview", pid)
        self.queues["preview"].put(job)
        return job

    def export(self, pid, seconds):
        projects.folder(pid)
        try:
            seconds = max(0, min(MAX_SECONDS, int(seconds)))
        except (TypeError, ValueError):
            seconds = 0
        job = self._new("export", pid, seconds=seconds)
        self.queues["export"].put(job)
        return job

    def cancel(self, jid):
        with self.lock:
            job = self.items.get(jid)
            if job and job["state"] in ("queued", "running"):
                job["cancel"] = True
                return True
        return False

    def busy(self):
        with self.lock:
            return any(j["state"] in ("queued", "running") for j in self.items.values())

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
                done = (self._preview if job["kind"] == "preview" else self._export)(job)
                job["state"] = "done" if done else "cancelled"
                job["progress"] = 1.0 if done else job["progress"]
            except (projects.UserError, video.VideoError) as e:
                job["state"], job["message"] = "error", str(e)
            except Exception as e:  # noqa: BLE001 - shown on the page, written to the log
                traceback.print_exc()
                job["state"] = "error"
                job["message"] = "Có lỗi khi làm video. Xem nhật ký (app.log) để biết thêm." if not isinstance(
                    e, OSError) else f"Lỗi đọc/ghi file: {e.strerror or e}"
            job["finished"] = time.time()
            print(f"job {job['kind']} {job['id']} {job['state']} {job['message']}".rstrip(), flush=True)

    def _progress(self, job, lo, hi):
        def report(fraction):
            job["progress"] = lo + (hi - lo) * fraction
        return report

    def _preview(self, job):
        project = projects.load(job["project"])
        folder = projects.folder(job["project"])
        name = f"preview-{project.get('rev', 0)}-{secrets.token_hex(2)}.mp4"
        out = os.path.join(folder, name)
        ok = engine.render(folder, project, out, size=engine.preview_size(project["video"]), crf=24,
                           preset="veryfast", progress=self._progress(job, 0, 1), cancelled=lambda: job["cancel"])
        if ok:
            projects.set_preview(project, name)
            job["output"] = name
        return ok

    def _export(self, job):
        project = projects.load(job["project"])
        folder = projects.folder(job["project"])
        info = project["video"]
        loop_s = info["seconds"]
        times = max(1, round(job["seconds"] / loop_s)) if job["seconds"] else 1
        out_dir = paths.out_dir()
        stem = f"{project['name']}-fx" + (f"-{length_tag(times * loop_s)}" if times > 1 else "")
        final = free_name(out_dir, stem)
        loop = os.path.join(folder, f"export-{job['id']}.mp4")
        try:
            render_share = 1.0 if times == 1 else 0.6
            ok = engine.render(folder, project, loop, progress=self._progress(job, 0, render_share),
                               cancelled=lambda: job["cancel"])
            if not ok:
                return False
            if times == 1:
                shutil.move(loop, final)
            else:
                need = os.path.getsize(loop) * times * 1.02 + 200e6
                free = shutil.disk_usage(out_dir).free
                if free < need:
                    raise projects.UserError(f"Ổ đĩa không đủ chỗ: cần khoảng {need / 1e9:.1f} GB, "
                                             f"còn {free / 1e9:.1f} GB. Chọn độ dài ngắn hơn hoặc dọn bớt ổ đĩa.")
                ok = video.repeat(loop, final, times, seconds=times * loop_s,
                                  progress=self._progress(job, render_share, 1.0), cancelled=lambda: job["cancel"])
                if not ok:
                    return False
            job["output"] = os.path.basename(final)
            return True
        finally:
            try:
                os.remove(loop)
            except OSError:
                pass
