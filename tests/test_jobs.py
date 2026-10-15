import os
import queue
import time
import unittest

from tests import helpers
from fx import jobs, projects


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


if __name__ == "__main__":
    unittest.main()
