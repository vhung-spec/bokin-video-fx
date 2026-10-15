import time
import unittest

from tests import helpers  # noqa: F401 - temporary data folder
from fx import jobs


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


if __name__ == "__main__":
    unittest.main()
