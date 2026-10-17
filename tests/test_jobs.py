import os
import queue
import time
import unittest

from tests import helpers
from fx import jobs, library, projects, timeline


def fake(j, **job):
    job = {"id": job.get("id", "j1"), "kind": "fxclip", "project": "p", "state": "queued", "progress": 0,
           "message": "", "created": time.time(), "started": None, "finished": None, "output": None,
           "cancel": False, **job}
    j.items[job["id"]] = job
    j.order.append(job["id"])
    return job


class Queues(unittest.TestCase):
    def test_a_cancelled_job_is_not_waited_for(self):
        j = jobs.Jobs()
        fake(j, id="a", name="fx-1", layer="l1", cancel=True)
        self.assertIsNone(j._pending(lambda x: x["name"] == "fx-1"))
        fake(j, id="b", name="fx-1", layer="l2")
        self.assertEqual(j._pending(lambda x: x["name"] == "fx-1")["id"], "b")

    def test_preview_clips_no_clip_wants_are_dropped(self):
        """A preview clip waiting or being made that no clip of the project wants any more (a lit steam clip moved
        again and again) is dropped, so the one wanted now does not wait behind it; another project's is kept."""
        j = jobs.Jobs()
        j.queues = {"export": queue.Queue(), "prep": queue.Queue()}  # nothing is run
        waiting = fake(j, id="w", name="fx-old1", layer="l1")
        running = fake(j, id="r", name="fx-old2", layer="l1", state="running")
        kept = fake(j, id="k", name="fx-keep", layer="l2", state="running")
        other = fake(j, id="o", name="fx-old3", layer="l1", project="q", state="running")
        job = j.fx_clip("p", "l1", "fx-new", {"fx-new", "fx-keep"})
        self.assertTrue(waiting["cancel"] and running["cancel"])
        self.assertFalse(kept["cancel"] or other["cancel"] or job["cancel"])

    def test_a_failed_light_copy_is_not_retried_at_once(self):
        j = jobs.Jobs()
        fake(j, id="c", kind="libproxy", item="0123456789", state="error", finished=time.time())
        self.assertEqual(j._failed_lately(lambda x: x["kind"] == "libproxy")["id"], "c")
        fake(j, id="d", kind="proxy", state="error", finished=time.time() - 3600)
        self.assertIsNone(j._failed_lately(lambda x: x["kind"] == "proxy"))


class ProjectVideo(unittest.TestCase):
    def test_jobs_for_a_replaced_video_are_not_this_ones(self):
        """After V1 is replaced, a light copy that failed for the old video does not stop one for the new video,
        and one made for the old video is never taken as the new one's."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "jobs.mp4"), 160, 90, 12)
        project = projects.create(src, "jobs.mp4")
        j = jobs.Jobs()
        j.queues = {"export": queue.Queue(), "prep": queue.Queue()}  # nothing is run
        fake(j, id="old", kind="proxy", project=project["id"], source="input-0a0b0c.mp4", state="error",
             finished=time.time())
        job = j.project_proxy(project["id"])
        self.assertNotEqual(job["id"], "old")
        self.assertEqual(job["source"], project["video"]["file"])
        self.assertEqual(j.project_proxy(project["id"])["id"], job["id"])  # asked again: the same one
        stale = {"project": project["id"], "source": "input-0a0b0c.mp4", "cancel": False}
        self.assertFalse(j._proxy(stale))
        self.assertIsNone(projects.load(project["id"])["proxy"])


    def test_a_preview_clip_is_made_for_the_settings_it_was_asked_for(self):
        """Two halves of a cut share a preview clip; one half is changed while the job waits: the job still makes
        the shared clip (the other half wants it), not the changed half's."""
        from fx import engine
        src = helpers.make_loop(os.path.join(helpers.ROOT, "jobsfx.mp4"), 160, 90, 12)
        project = projects.create(src, "jobsfx.mp4")
        tid = project["tracks"][0]["id"]
        project = projects.apply(project, {"layers": [
            {"id": "halfaa", "kind": "snow", "track": tid, "start": 0, "end": 100, "seed": 3},
            {"id": "halfbb", "kind": "snow", "track": tid, "start": 100, "end": 200, "seed": 3}]})
        shared = engine.fx_clip_name(project, project["layers"][0])
        j = jobs.Jobs()
        j.queues = {"export": queue.Queue(), "prep": queue.Queue()}  # nothing is run
        job = j.fx_clip(project["id"], "halfaa", shared, {shared})
        layers = [dict(project["layers"][0], params=dict(project["layers"][0]["params"], amount=11)),
                  project["layers"][1]]
        projects.apply(projects.load(project["id"]), {"layers": layers})
        self.assertTrue(j._fxclip(job))
        self.assertEqual(job["output"], shared)
        self.assertTrue(os.path.exists(os.path.join(projects.folder(project["id"]), "clips", shared)))


class Export(unittest.TestCase):
    def test_a_clip_played_backwards_has_its_copy_made_first(self):
        """An export with a clip set to "Phát ngược" whose copy is not made yet makes it, then the video."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "jobsrev.mp4"), 96, 54, 12)
        project = projects.create(src, "jobsrev.mp4")
        fxv = helpers.make_loop(os.path.join(helpers.ROOT, "jobsrevfx.mp4"), 64, 36, 24)
        item = library.add(fxv, "jobsrevfx.mp4")
        tid = project["tracks"][0]["id"]
        projects.apply(project, {"layers": [{"id": "filerv", "kind": "file", "lib": item["id"], "track": tid,
                                             "start": 0, "end": 48, "params": {"reverse": True}}]})
        self.assertIsNone(library.reversed_copy(library.load(item["id"])))
        j = jobs.Jobs()
        j.queues = {"export": queue.Queue(), "prep": queue.Queue()}  # nothing is run
        job = fake(j, id="ex", kind="export", project=project["id"])
        self.assertTrue(j._export(job))
        self.assertTrue(library.reversed_copy(library.load(item["id"])))
        self.assertEqual(job["message"], "")

    def test_copies_for_clips_ticked_meanwhile_and_none_past_the_end(self):
        """A clip set to play backwards while the export makes the first copies gets its copy too; a clip past
        the video's end (never drawn) gets none."""
        from unittest import mock
        src = helpers.make_loop(os.path.join(helpers.ROOT, "jobsrev2.mp4"), 96, 54, 12)
        project = projects.create(src, "jobsrev2.mp4")
        first = library.add(helpers.make_loop(os.path.join(helpers.ROOT, "jr1.mp4"), 64, 36, 24), "jr1.mp4")
        second = library.add(helpers.make_loop(os.path.join(helpers.ROOT, "jr2.mp4"), 64, 36, 24), "jr2.mp4")
        late = library.add(helpers.make_loop(os.path.join(helpers.ROOT, "jr3.mp4"), 64, 36, 24), "jr3.mp4")
        tid = project["tracks"][0]["id"]
        total = timeline.project_frames(project)
        clip = {"kind": "file", "track": tid, "params": {"reverse": True}}
        projects.apply(project, {"layers": [dict(clip, id="filej1", lib=first["id"], start=0, end=24),
                                            dict(clip, id="filej2", lib=second["id"], start=24, end=48,
                                                 params={}),
                                            dict(clip, id="filej3", lib=late["id"], start=total, end=total + 24)]})
        real = library.make_reverse

        def tick_the_second(iid, **kw):  # the page ticks "Phát ngược" on the second clip meanwhile
            p = projects.load(project["id"])
            layers = [dict(lay, params=dict(lay["params"], reverse=True)) if lay["id"] == "filej2" else lay
                      for lay in p["layers"]]
            projects.apply(p, {"layers": layers})
            return real(iid, **kw)
        j = jobs.Jobs()
        j.queues = {"export": queue.Queue(), "prep": queue.Queue()}  # nothing is run
        job = fake(j, id="ex2", kind="export", project=project["id"])
        with mock.patch.object(library, "make_reverse", side_effect=tick_the_second):
            self.assertTrue(j._export(job))
        self.assertTrue(library.reversed_copy(library.load(second["id"])))
        self.assertIsNone(library.reversed_copy(library.load(late["id"])))
        self.assertTrue(os.path.exists(os.path.join(os.environ["BOKIN_FX_OUT"], job["output"])))


if __name__ == "__main__":
    unittest.main()
