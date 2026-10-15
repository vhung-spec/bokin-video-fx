import io
import os
import unittest

import numpy as np
from PIL import Image, ImageDraw

from tests import helpers
from fx import engine, projects, video

W, H, P = 320, 180, 24


def png(box, size):
    im = Image.new("L", size, 0)
    ImageDraw.Draw(im).rectangle(box, fill=255)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class Engine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        src = helpers.make_loop(os.path.join(helpers.ROOT, "engine.mp4"), W, H, P)
        cls.project = projects.create(src, "engine.mp4")
        cls.folder = projects.folder(cls.project["id"])
        cls.info = cls.project["video"]

    def base(self, t):
        return video.read_frame(engine.input_path(self.folder, self.project), self.info, t)

    def test_probe(self):
        self.assertEqual((self.info["w"], self.info["h"], self.info["frames"], self.info["fps"]), (W, H, P, "24"))

    def test_still_area_is_exact(self):
        project = projects.save_mask(projects.load(self.project["id"]), "freeze", png([0, 0, 99, H], (W, H)))
        scene = engine.Scene(self.folder, dict(project, layers=[], freeze={"feather": 0}), (W, H))
        out = scene.frame(10, self.base(10))
        still = self.base(0)
        self.assertTrue((out[:, :98] == still[:, :98]).all())
        self.assertTrue((out[:, 102:] == self.base(10)[:, 102:]).all())
        # the original picture instead of a frame
        Image.fromarray(np.full((H * 2, W * 2, 3), 200, np.uint8)).save(os.path.join(helpers.ROOT, "still.png"))
        project = projects.set_still_image(project, os.path.join(helpers.ROOT, "still.png"), "still.png")
        self.assertIsNotNone(project["still"]["warning"])  # a grey picture does not look like the video
        scene = engine.Scene(self.folder, dict(project, layers=[], freeze={"feather": 0}), (W, H))
        self.assertTrue((scene.frame(3, self.base(3))[:, :98] == 200).all())
        projects.save_mask(projects.load(self.project["id"]), "freeze", b"")

    def test_areas_and_foreground(self):
        project = projects.load(self.project["id"])
        project = projects.apply(project, {"layers": [
            {"id": "snowaa", "kind": "snow", "params": {"amount": 100, "size": 100}, "seed": 1},
            {"id": "snowbb", "kind": "snow", "behind": True, "params": {"amount": 100, "size": 100}, "seed": 2}]})
        project = projects.save_mask(project, "snowaa", png([0, 0, W // 2 - 1, H], (W, H)))
        project = projects.save_mask(project, "fg", png([0, 0, W, H], (W, H)))  # everything is in front
        scene = engine.Scene(self.folder, dict(project, fg={"feather": 0}), (W, H))
        self.assertEqual(len(scene.layers), 1)  # the second one is hidden everywhere
        changed = scene.frame(5, self.base(5)) != self.base(5)
        self.assertFalse(changed[:, W // 2 + 30:].any())  # 20 px feather around the area
        self.assertTrue(changed[:, :W // 2 - 30].any())
        projects.save_mask(projects.load(self.project["id"]), "fg", b"")
        projects.apply(projects.load(self.project["id"]), {"layers": []})
        self.assertFalse(os.path.exists(engine.mask_path(self.folder, "snowaa")))  # removed with its layer

    def test_render_and_repeat(self):
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "rainaa", "kind": "rain", "params": {}, "seed": 1},
            {"id": "glowaa", "kind": "glow", "params": {}, "seed": 1}]})
        out = os.path.join(helpers.ROOT, "render.mp4")
        self.assertTrue(engine.render(self.folder, project, out, preset="veryfast"))
        info = video.probe(out)
        self.assertEqual((info["frames"], info["matrix"]), (P, "bt709"))
        data = open(out, "rb").read()
        for trace in (b"x264", b"Lavf", b"Lavc"):
            self.assertNotIn(trace, data)
        longer = os.path.join(helpers.ROOT, "render-x4.mp4")
        self.assertTrue(video.repeat(out, longer, 4, seconds=4))
        self.assertEqual(video.probe(longer)["frames"], 4 * P)
        stop = {"n": 0}

        def cancelled():
            stop["n"] += 1
            return stop["n"] > 3
        self.assertFalse(engine.render(self.folder, project, os.path.join(helpers.ROOT, "x.mp4"), cancelled=cancelled))
        self.assertFalse(os.path.exists(os.path.join(helpers.ROOT, "x.mp4")))
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def test_outside_file_layer(self):
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "clip.mp4"), 160, 90, 40, "30")
        project = projects.apply(projects.load(self.project["id"]), {"layers": [{"id": "fileaa", "kind": "file"}]})
        project = projects.set_layer_file(project, "fileaa", clip, "effect on black.mp4")
        layer = project["layers"][0]
        self.assertEqual(layer["params"]["blend"], "screen")
        self.assertTrue(os.path.exists(os.path.join(self.folder, "files", layer["file"]["stored"])))
        scene = engine.Scene(self.folder, project, (W, H))
        try:
            frames = [scene.frame(t, self.base(t)) for t in range(P)]
        finally:
            scene.close()
        self.assertFalse(np.array_equal(frames[0], self.base(0)))
        with self.assertRaises(projects.UserError):
            projects.set_layer_file(project, "fileaa", clip, "virus.exe")
        projects.apply(projects.load(self.project["id"]), {"layers": []})
        self.assertEqual(os.listdir(os.path.join(self.folder, "files")), [])


class Validation(unittest.TestCase):
    def test_not_a_video(self):
        bad = os.path.join(helpers.ROOT, "bad.mp4")
        with open(bad, "wb") as f:
            f.write(b"not a video at all")
        with self.assertRaises(video.VideoError):
            projects.create(bad, "bad.mp4")
        with self.assertRaises(projects.UserError):
            projects.create(bad, "bad.exe")

    def test_ids_from_the_page(self):
        src = helpers.make_loop(os.path.join(helpers.ROOT, "ids.mp4"), 160, 90, 12)
        project = projects.create(src, "ids.mp4")
        for bad in ("../x", "fg", "A", "x" * 40):
            with self.assertRaises(projects.UserError):
                projects.apply(project, {"layers": [{"id": bad, "kind": "snow"}]})
        with self.assertRaises(projects.UserError):
            projects.apply(project, {"layers": [{"id": "abcd", "kind": "../../x"}]})
        with self.assertRaises(projects.UserError):
            projects.save_mask(project, "../../etc", b"x")
        with self.assertRaises(projects.UserError):
            projects.folder("../../../windows")


if __name__ == "__main__":
    unittest.main()
