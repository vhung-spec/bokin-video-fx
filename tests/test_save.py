import io
import os
import unittest

from PIL import Image

from tests import helpers
from fx import paths, projects


def png(w, h, value):
    buf = io.BytesIO()
    Image.new("L", (w, h), value).save(buf, "PNG")
    return buf.getvalue()


class SaveAsCopy(unittest.TestCase):
    def setUp(self):
        src = helpers.make_loop(os.path.join(helpers.ROOT, "save.mp4"), 160, 90, 12)
        self.project = projects.create(src, "Quán cà phê.mp4")
        self.project = projects.apply(self.project, {
            "layers": [{"id": "snow1", "kind": "snow", "name": "Tuyết", "start": 0, "end": 48}]})
        self.project = projects.save_mask(self.project, "freeze", png(160, 90, 255))
        self.project = projects.save_mask(self.project, "snow1", png(160, 90, 128))
        base = os.path.join(paths.projects_dir(), self.project["id"])
        os.makedirs(os.path.join(base, "clips", "render-abc123"))
        for name in ("clips/fx-0123456789ab.webm", "clips/render-abc123/part-00.webm", "proxy.part.mp4"):
            with open(os.path.join(base, name), "wb") as f:
                f.write(b"x" * 10)

    def test_copy(self):
        copy = projects.copy(self.project["id"], "Quán cà phê - bản 2")
        self.assertNotEqual(copy["id"], self.project["id"])
        self.assertEqual(copy["name"], "Quán cà phê - bản 2")
        again = projects.load(copy["id"])
        self.assertEqual([lay["id"] for lay in again["layers"]], ["snow1"])
        self.assertEqual(again["video"], self.project["video"])
        a = os.path.join(paths.projects_dir(), self.project["id"])
        b = os.path.join(paths.projects_dir(), copy["id"])
        for name in (again["video"]["file"], "masks/freeze.png", "masks/snow1.png", "clips/fx-0123456789ab.webm"):
            self.assertTrue(os.path.isfile(os.path.join(b, name)), name)
        # files being written are left behind: a render's pieces, a light copy not finished
        self.assertFalse(os.path.exists(os.path.join(b, "clips", "render-abc123")))
        self.assertFalse(os.path.exists(os.path.join(b, "proxy.part.mp4")))
        # the two go their own ways: a change to one's area (or settings) is not the other's
        projects.save_mask(again, "snow1", png(160, 90, 0))
        projects.apply(again, {"name": "Đổi", "layers": []})
        with Image.open(os.path.join(a, "masks", "snow1.png")) as im:
            self.assertEqual(im.getpixel((5, 5)), 128)
        old = projects.load(self.project["id"])
        self.assertEqual((old["name"], [lay["id"] for lay in old["layers"]]), ("Quán cà phê", ["snow1"]))
        # deleting the old project leaves the copy whole
        projects.delete(self.project["id"])
        self.assertTrue(os.path.isfile(os.path.join(b, again["video"]["file"])))
        self.assertEqual(projects.load(copy["id"])["id"], copy["id"])

    def test_bad(self):
        with self.assertRaises(projects.UserError):
            projects.copy("../../x", "a")
        with self.assertRaises(projects.UserError):
            projects.copy("0123456789ab", "a")  # no such project
        self.assertEqual(projects.copy(self.project["id"], "///")["name"], self.project["name"])  # (no name left)


if __name__ == "__main__":
    unittest.main()
