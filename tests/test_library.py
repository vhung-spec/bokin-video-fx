import os
import unittest

from PIL import Image

from tests import helpers
from fx import engine, library, projects, video


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
        src = helpers.make_loop(os.path.join(helpers.ROOT, "b2.mp4"), 160, 90, 12)
        project = projects.create(src, "b2.mp4")
        with self.assertRaises(projects.UserError):  # not an item's id at all
            projects.apply(project, {"layers": [{"id": "filezz", "kind": "file", "lib": "../x"}]})
        # an item deleted since (the clip came back with undo): the clip stays, drawing nothing, and saves go on
        saved = projects.apply(project, {"layers": [{"id": "filezz", "kind": "file", "lib": "0123456789"}]})
        self.assertIsNone(saved["layers"][0]["file"]["info"])
        self.assertEqual(engine.clip_file(projects.folder(project["id"]), saved["layers"][0]), (None, None))

    def test_light_copies_of_an_older_version_are_made_again(self):
        """A light copy made before video.PROXY_V (timed or coloured unlike the renderer) counts as missing."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "oldcopy.mp4"), 160, 90, 12)
        item = library.add(src, "oldcopy.mp4")
        item.update(proxy="proxy.mp4")
        item.pop("proxy_v", None)
        library.save(item)
        self.assertIsNone(library.load(item["id"])["proxy"])
        item.update(proxy="proxy.mp4", proxy_v=video.PROXY_V)
        library.save(item)
        self.assertEqual(library.load(item["id"])["proxy"], "proxy.mp4")


if __name__ == "__main__":
    unittest.main()


class Backwards(unittest.TestCase):
    def test_a_video_played_backwards(self):
        """The played-backwards copy: its frame k is frame n-1-k of the file (opaque: H.264; see-through: VP9 with
        its transparency), with a light copy for the page; a clip set to "Phát ngược" uses it."""
        import numpy as np
        from fx import engine
        src = helpers.make_loop(os.path.join(helpers.ROOT, "fwd.mp4"), 160, 90, 12)
        item = library.make_reverse(library.add(src, "fwd.mp4")["id"])
        src = library.source_path(item)  # (the upload was moved into the library)
        path, info = library.reversed_path(item)
        n = item["info"]["frames"]
        self.assertEqual(info["frames"], n)
        for k in (0, 5, n - 1):
            a = video.read_frame(src, item["info"], n - 1 - k).astype(np.int16)
            b = video.read_frame(path, info, k).astype(np.int16)
            self.assertLess(float(np.abs(a - b).mean()), 2.0, f"frame {k}")
        self.assertTrue(library.public(item)["reverse"])
        clip = {"kind": "file", "file": {"lib": item["id"]}, "params": {"reverse": True}}
        self.assertEqual(engine.clip_file(helpers.ROOT, clip)[0], path)
        self.assertEqual(engine.clip_file(helpers.ROOT, dict(clip, params={}))[0], library.source_path(item))

    def test_in_pieces_and_asked_twice_at_once(self):
        """A file too big for one run is turned round piece by piece; a prep job and an export asking for the same
        copy at once get the one copy (made once)."""
        import threading
        import numpy as np
        src = helpers.make_loop(os.path.join(helpers.ROOT, "long.mp4"), 64, 36, 30)
        info = video.probe(src, overlay=True)
        out = video.reverse(src, info, os.path.join(helpers.ROOT, "pieces"), budget=64 * 36 * 3 * 7)  # 7 a piece
        back = video.probe(out, overlay=True)
        self.assertEqual(back["frames"], info["frames"])
        for k in (0, 6, 7, 13, 29):  # either side of where two pieces meet
            a = video.read_frame(src, info, 29 - k).astype(np.int16)
            self.assertLess(float(np.abs(a - video.read_frame(out, back, k)).mean()), 2.0, f"frame {k}")
        iid = library.add(src, "long.mp4")["id"]
        got = []
        runs = [threading.Thread(target=lambda: got.append(library.make_reverse(iid))) for _ in range(2)]
        for t in runs:
            t.start()
        for t in runs:
            t.join()
        self.assertEqual(len(got), 2)
        self.assertTrue(all(g and g["reverse"] == got[0]["reverse"] for g in got))
        self.assertTrue(os.path.exists(library.reversed_path(library.load(iid))[0]))
        self.assertFalse(os.path.exists(os.path.join(library.folder(iid), "reverse.parts")))

    def test_see_through_played_backwards(self):
        import subprocess
        import numpy as np
        path = os.path.join(helpers.ROOT, "moving.mov")
        graph = ("color=c=black@0.0:s=96x64:r=12:d=1,format=rgba[bg];color=c=orange:s=16x16:r=12:d=1,format=rgba[fg];"
                 "[bg][fg]overlay=x='t*60':y=24:format=auto,format=rgba")
        subprocess.run([video.ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i", graph, "-c:v", "prores_ks",
                        "-profile:v", "4444", "-pix_fmt", "yuva444p10le", path], check=True)
        item = library.make_reverse(library.add(path, "moving.mov")["id"])
        path = library.source_path(item)
        rpath, info = library.reversed_path(item)
        self.assertTrue(info["alpha"])
        n = item["info"]["frames"]
        with video.Frames(path, item["info"], alpha=True) as f:
            fwd = list(f)
        with video.Frames(rpath, info, alpha=True) as f:
            back = list(f)
        self.assertEqual(len(back), n)
        for k in (0, n // 2, n - 1):
            self.assertLess(float(np.abs(fwd[n - 1 - k][..., 3].astype(np.int16) - back[k][..., 3]).mean()), 3.0)
