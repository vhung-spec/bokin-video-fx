"""A project from a picture instead of a video (the owner asked 2026-11-04: "app có thể nhận cả dự án là ảnh"): the
picture is kept as a PNG of ours and read as a still video (video.still_info), so the rest of the app treats it as
any other video."""
import io
import os
import unittest

import numpy as np
from PIL import Image

from tests import helpers
from fx import engine, projects, timeline, video


def picture(name, w=322, h=181, mode="RGB", seed=1, smooth=False, **save):
    rng = np.random.default_rng(seed)
    arr = (rng.random((h, w, len(mode))) * 255).astype(np.uint8)
    if smooth:  # (a scene a video codec keeps: gradients, no noise)
        y, x = np.mgrid[0:h, 0:w] / max(w, h)
        arr = np.stack([120 + 100 * np.sin(5 * x + c) * np.cos(4 * y - c) for c in range(len(mode))], 2).astype(np.uint8)
    path = os.path.join(helpers.ROOT, name)
    Image.fromarray(arr, mode).save(path, **save)
    return path, arr


class Picture(unittest.TestCase):
    def make(self, name="Phòng khách.png", **kw):
        path, arr = picture(name, **kw)
        project = projects.create(path, name)
        self.assertFalse(os.path.exists(path))  # (moved in, as a PNG of ours)
        return project, projects.folder(project["id"]), arr

    def test_a_still_video(self):
        """A still video of IMAGE_SECONDS at IMAGE_FPS, even sides (an odd last row and column left out), each frame
        the picture itself, to the pixel."""
        project, folder, arr = self.make()
        info = project["video"]
        self.assertTrue(info["image"])
        self.assertEqual((info["w"], info["h"], info["fps"], info["frames"], info["seconds"]), (322, 180, "30", 900, 30.0))
        self.assertEqual(info["file"], "input.png")
        path = engine.input_path(folder, project)
        for t in (0, 450, 899):
            self.assertTrue(np.array_equal(video.read_frame(path, info, t), arr[:180]))
        with video.Frames(path, info, (160, 90), start=10, count=5) as frames:
            self.assertEqual([f.shape for f in frames], [(90, 160, 3)] * 5)
        with video.Frames(path, info, step=37) as frames:
            self.assertEqual(len(list(frames)), len(range(0, 900, 37)))
        with video.Frames(path, info) as frames:
            frame = next(iter(frames))
            self.assertFalse(frame.flags.writeable)  # (as a decoded frame: nobody paints on the picture kept)
        self.assertEqual((timeline.output_frames(info, 30), timeline.output_frames(info, 60)), (900, 1800))
        self.assertTrue(projects.summary(project)["image"])

    def test_pictures_as_they_come(self):
        """A JPEG with its camera's turn, a see-through PNG on black, a big picture made 4K on its longer side; a
        tiny picture or a broken file refused."""
        path, arr = picture("turned.jpg", 200, 120, quality=95, exif=_turned())
        project = projects.create(path, "turned.jpg")
        self.assertEqual((project["video"]["w"], project["video"]["h"]), (120, 200))  # (stood up, as shot)
        path, arr = picture("clear.png", 128, 128, mode="RGBA")
        project = projects.create(path, "clear.png")
        got = video.read_frame(engine.input_path(projects.folder(project["id"]), project), project["video"], 0)
        a = arr[..., 3:].astype(np.float32) / 255
        self.assertLess(float(np.abs(got - arr[..., :3] * a).max()), 1.01)  # (over black)
        path, _ = picture("big.webp", 4000, 2250, lossless=False, quality=80)
        project = projects.create(path, "big.webp")
        self.assertEqual((project["video"]["w"], project["video"]["h"]), (projects.PICTURE_SIDE, 2160))
        path, _ = picture("tiny.png", 40, 40)
        with self.assertRaises(projects.UserError):
            projects.create(path, "tiny.png")
        bad = os.path.join(helpers.ROOT, "bad.png")
        with open(bad, "wb") as f:
            f.write(b"not a picture at all")
        with self.assertRaises(projects.UserError):
            projects.create(bad, "bad.png")

    def test_light_copy_and_export(self):
        """The page's light copy is a still video as long as the loop; the export, with an effect, is the length
        asked, at the picture's size and IMAGE_FPS."""
        project, folder, arr = self.make("export.png", w=320, h=180, smooth=True)
        name = engine.make_base_proxy(folder, project)
        copy = video.probe(os.path.join(folder, name))
        self.assertEqual((copy["w"], copy["h"], copy["frames"], copy["fps"], copy["codec"]), (320, 180, 900, "30", "h264"))
        track = project["tracks"][0]["id"]
        project = projects.apply(projects.load(project["id"]), {"layers": [
            {"id": "snowaa", "kind": "snow", "track": track, "start": 0, "end": 900, "seed": 1,
             "params": {"amount": 100}}]})
        out = os.path.join(helpers.ROOT, "picture-out.mp4")
        engine.render(folder, project, out)
        got = video.probe(out)
        self.assertEqual((got["w"], got["h"], got["frames"], got["fps"], got["matrix"]), (320, 180, 900, "30", "bt709"))
        with video.Frames(out, got, count=1) as frames:
            first = next(iter(frames)).astype(np.float32)
        scene = engine.Scene(folder, project, (320, 180))
        self.assertLess(float(np.abs(first - scene.frame(0, arr)).mean()), 4)  # (the picture, snow over it)

    def test_replaced_both_ways(self):
        """A picture may take the place of a video on V1, and the other way round; the clips stay at their times."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "was-video.mp4"), 160, 90, 24)
        project = projects.create(src, "was-video.mp4")
        track = project["tracks"][0]["id"]
        project = projects.apply(project, {"layers": [
            {"id": "snowbb", "kind": "snow", "track": track, "start": 24, "end": 72, "seed": 1},
            {"id": "snowcc", "kind": "snow", "track": project["tracks"][1]["id"], "start": 0,
             "end": timeline.project_frames(project), "seed": 2}]})
        path, _ = picture("now-picture.jpg", 160, 90, quality=95)
        project, notes = projects.replace_video(projects.load(project["id"]), path, "now-picture.jpg")
        self.assertTrue(project["video"]["image"])
        self.assertTrue(any("ảnh đứng yên" in n for n in notes), notes)
        clips = {c["id"]: c for c in project["layers"]}
        self.assertEqual((clips["snowbb"]["start"], clips["snowbb"]["end"]), (30, 90))  # (1 s to 3 s, at 30 fps)
        self.assertEqual(clips["snowcc"]["end"], 900)  # (to the end, still)
        src = helpers.make_loop(os.path.join(helpers.ROOT, "video-again.mp4"), 160, 90, 24)
        project, notes = projects.replace_video(projects.load(project["id"]), src, "video-again.mp4")
        self.assertFalse(project["video"].get("image"))
        self.assertEqual(project["video"]["fps"], "24")


def _turned():
    """EXIF that says: turn the picture a quarter (orientation 6), as a phone held upright writes it."""
    exif = Image.Exif()
    exif[0x0112] = 6
    return exif.tobytes()


class Uploads(unittest.TestCase):
    def test_kinds(self):
        self.assertTrue({".png", ".jpg", ".jpeg", ".webp"} <= projects.BASE_TYPES)
        self.assertIn(".mp4", projects.BASE_TYPES)
        self.assertNotIn(".gif", projects.BASE_TYPES)  # (a GIF may move: not a still picture)
        self.assertNotIn(".exe", projects.BASE_TYPES)
        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, "PNG")
        named = os.path.join(helpers.ROOT, "png-as.exe")
        with open(named, "wb") as f:
            f.write(buf.getvalue())
        with self.assertRaises(projects.UserError):
            projects.create(named, "png-as.exe")


if __name__ == "__main__":
    unittest.main()
