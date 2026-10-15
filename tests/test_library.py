import os
import unittest

from PIL import Image

from tests import helpers
from fx import library, projects, video


class Library(unittest.TestCase):
    def test_video_and_picture(self):
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "lib.mp4"), 160, 90, 30, "30")
        item = library.add(clip, "Tuyết trên nền đen.mp4")
        self.assertEqual(item["name"], "Tuyết trên nền đen")
        self.assertFalse(item["info"]["image"])
        self.assertIsNone(item["proxy"])
        folder = library.folder(item["id"])
        self.assertTrue(os.path.exists(os.path.join(folder, "thumb.jpg")))
        item = library.make_proxy(item["id"])
        self.assertEqual(item["proxy"], "proxy.mp4")
        info = video.probe(library.proxy_path(item))
        self.assertEqual((info["frames"], info["codec"]), (30, "h264"))
        self.assertIn(item["id"], [i["id"] for i in library.list_all()])

        png = os.path.join(helpers.ROOT, "logo.png")
        Image.new("RGBA", (300, 200), (255, 0, 0, 128)).save(png)
        pic = library.add(png, "logo.png")
        self.assertTrue(pic["info"]["image"] and pic["info"]["alpha"])
        self.assertEqual(pic["proxy"], "proxy.png")  # pictures are ready at once

        self.assertEqual(library.rename(pic["id"], "Logo công ty")["name"], "Logo công ty")
        library.delete(pic["id"], [])
        with self.assertRaises(library.LibraryError):
            library.load(pic["id"])

    def test_in_use_cannot_be_deleted(self):
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "used.mp4"), 160, 90, 12)
        item = library.add(clip, "used.mp4")
        src = helpers.make_loop(os.path.join(helpers.ROOT, "base.mp4"), 160, 90, 12)
        project = projects.create(src, "Cảnh.mp4")
        projects.apply(project, {"layers": [{"id": "fileuu", "kind": "file", "lib": item["id"]}]})
        users = projects.using_library(item["id"])
        self.assertEqual(users, ["Cảnh"])
        with self.assertRaises(library.LibraryError):
            library.delete(item["id"], users)
        projects.apply(projects.load(project["id"]), {"layers": []})
        library.delete(item["id"], projects.using_library(item["id"]))

    def test_bad_files_and_ids(self):
        bad = os.path.join(helpers.ROOT, "bad.mov")
        with open(bad, "wb") as f:
            f.write(b"not a video")
        with self.assertRaises(library.LibraryError):
            library.add(bad, "bad.mov")
        with open(bad, "wb") as f:
            f.write(b"x")
        with self.assertRaises(library.LibraryError):
            library.add(bad, "virus.exe")
        for iid in ("../x", "", None, "0123456789/"):
            with self.assertRaises(library.LibraryError):
                library.folder(iid)
        with self.assertRaises(projects.UserError):  # a clip naming an item that is not there
            src = helpers.make_loop(os.path.join(helpers.ROOT, "b2.mp4"), 160, 90, 12)
            projects.apply(projects.create(src, "b2.mp4"),
                           {"layers": [{"id": "filezz", "kind": "file", "lib": "0123456789"}]})


if __name__ == "__main__":
    unittest.main()
