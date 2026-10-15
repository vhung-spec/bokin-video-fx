import io
import os
import unittest

import numpy as np
from PIL import Image, ImageDraw

from tests import helpers
from fx import engine, library, projects, timeline, video

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

    def test_render(self):
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "rainaa", "kind": "rain", "params": {}, "seed": 1},
            {"id": "glowaa", "kind": "glow", "params": {}, "seed": 1}]})
        out = os.path.join(helpers.ROOT, "render.mp4")
        self.assertTrue(engine.render(self.folder, project, out, seconds=1, preset="veryfast"))
        info = video.probe(out)
        self.assertEqual((info["frames"], info["matrix"]), (P, "bt709"))
        data = open(out, "rb").read()
        for trace in (b"x264", b"Lavf", b"Lavc"):
            self.assertNotIn(trace, data)
        stop = {"n": 0}

        def cancelled():
            stop["n"] += 1
            return stop["n"] > 3
        self.assertFalse(engine.render(self.folder, project, os.path.join(helpers.ROOT, "x.mp4"), seconds=2,
                                       cancelled=cancelled, processes=1))
        self.assertFalse(os.path.exists(os.path.join(helpers.ROOT, "x.mp4")))
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def test_long_video_in_pieces(self):
        """A 4-loop video rendered by two processes: right length, pieces joined without a jump, end joins start,
        and the same picture as one process makes."""
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "snowpp", "kind": "snow", "params": {"amount": 80}, "seed": 3}]})
        one, two = os.path.join(helpers.ROOT, "long-1.mp4"), os.path.join(helpers.ROOT, "long-2.mp4")
        self.assertTrue(engine.render(self.folder, project, one, seconds=4, processes=1, preset="veryfast"))
        self.assertTrue(engine.render(self.folder, project, two, seconds=4, processes=2, preset="veryfast"))
        info = video.probe(two)
        self.assertEqual(info["frames"], 4 * P)
        with video.Frames(two, info) as frames:
            b = [f.astype(np.int16) for f in frames]
        with video.Frames(one, video.probe(one)) as frames:
            a = [f.astype(np.int16) for f in frames]
        self.assertLess(max(float(np.abs(x - y).mean()) for x, y in zip(a, b)), 1.5)
        steps = [float(np.abs(b[(i + 1) % len(b)] - b[i]).mean()) for i in range(len(b))]
        usual = float(np.median(steps))
        for at in (len(b) // 2 - 1, len(b) - 1):  # where the two pieces meet, and where the end meets the start
            self.assertLess(steps[at], usual * 1.6 + 0.5, f"jump at frame {at}: {steps[at]:.2f}, usual {usual:.2f}")
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def test_outside_file_clip(self):
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "clip.mp4"), 160, 90, 40, "30")
        item = library.add(clip, "effect on black.mp4")
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "fileaa", "kind": "file", "lib": item["id"], "params": {"blend": "screen"}}]})
        layer = project["layers"][0]
        self.assertEqual(layer["file"]["lib"], item["id"])
        scene = engine.Scene(self.folder, project, (W, H))
        try:
            frames = [scene.frame(t, self.base(t)) for t in range(P)]
        finally:
            scene.close()
        self.assertFalse(np.array_equal(frames[0], self.base(0)))
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def frames(self, project, times):
        scene = engine.Scene(self.folder, project, (W, H))
        try:
            return {t: scene.frame(t, self.base(t % P)).astype(np.int16) for t in times}
        finally:
            scene.close()

    def test_clips_show_in_their_time(self):
        """A clip changes nothing outside its time, fades in and out, and a clip touching the end of the video
        fades out there, so its last frame runs smoothly into its first."""
        total = timeline.project_frames(self.project)
        tracks = [t["id"] for t in self.project["tracks"]]
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "glowtt", "kind": "glow", "track": tracks[0], "start": 48, "end": 120, "fade_in": 1,
             "fade_out": 1, "params": {"amount": 100, "brightness": 100}, "seed": 4},
            {"id": "snowtt", "kind": "snow", "track": tracks[1], "start": total - 48, "end": total,
             "params": {"amount": 100, "size": 100}, "seed": 5}]})
        got = self.frames(project, [0, 47, 48, 60, 84, 119, 120, total - 49, total - 1])
        for t in (0, 47, 120, total - 49):
            self.assertTrue((got[t] == self.base(t % P)).all(), f"frame {t} changed")
        change = {t: float(np.abs(got[t] - self.base(t % P)).mean()) for t in got}
        self.assertLess(change[48], change[60])  # fading in
        self.assertLess(change[60], change[84] + 1e-6)
        self.assertLess(change[119], change[84])  # fading out
        self.assertLess(change[total - 1], 0.2 * max(change[84], 1e-3) + 0.05)  # nearly gone at the very end
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def test_cut_clip_is_the_same_picture(self):
        """A clip cut in two (the second half starting where the first stopped) gives the same frames."""
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "cut.mp4"), 160, 90, 20, "24")  # shorter than the clip
        item = library.add(clip, "cut.mp4")
        tid = self.project["tracks"][0]["id"]
        whole = [{"id": "fileww", "kind": "file", "lib": item["id"], "track": tid, "start": 10, "end": 90,
                  "params": {"blend": "screen", "fade": 0.3}},
                 {"id": "snowww", "kind": "snow", "track": tid, "start": 90, "end": 150, "seed": 2}]
        cut = [dict(whole[0], id="filew1", end=50), dict(whole[0], id="filew2", start=50, skip=40 / 24),
               dict(whole[1], id="snoww1", end=120), dict(whole[1], id="snoww2", start=120)]
        times = [10, 30, 49, 50, 51, 70, 89, 100, 119, 120, 121, 149]
        a = self.frames(projects.apply(projects.load(self.project["id"]), {"layers": whole}), times)
        b = self.frames(projects.apply(projects.load(self.project["id"]), {"layers": cut}), times)
        for t in times:
            self.assertTrue((a[t] == b[t]).all(), f"frame {t} differs")
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def test_cut_full_clip_keeps_its_picture(self):
        """A clip over the whole video cut in two: each half starts where the whole clip was in its file
        (skip = its seam fade, as the page sets it), so the frames are the same away from the video's ends."""
        clip = helpers.make_loop(os.path.join(helpers.ROOT, "full.mp4"), 160, 90, 40, "24")  # shorter: it repeats
        item = library.add(clip, "full.mp4")
        total = timeline.project_frames(self.project)
        tid = self.project["tracks"][0]["id"]
        whole = {"id": "fileff", "kind": "file", "lib": item["id"], "track": tid, "start": 0, "end": total,
                 "params": {"blend": "screen", "fade": 0.5}}
        cut = [dict(whole, id="filef1", end=360, skip=0.5), dict(whole, id="filef2", start=360, skip=0.5 + 360 / 24)]
        times = [12, 100, 300, 359, 360, 361, 500, 700, total - 13]
        a = self.frames(projects.apply(projects.load(self.project["id"]), {"layers": [whole]}), times)
        b = self.frames(projects.apply(projects.load(self.project["id"]), {"layers": cut}), times)
        for t in times:
            self.assertTrue((a[t] == b[t]).all(), f"frame {t} differs")
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def test_export_and_preview_clip_at_once(self):
        """An export and a preview clip made at the same time in this process (one piece each, as on a 2-CPU
        computer) keep their own progress and stop request."""
        import threading
        project = projects.apply(projects.load(self.project["id"]), {"layers": [
            {"id": "snowcc", "kind": "snow", "params": {"amount": 40}, "seed": 3}]})
        got = {}
        stop = {"n": 0}

        def cancelled():
            stop["n"] += 1
            return stop["n"] > 5

        def export():
            got["export"] = engine.render(self.folder, project, os.path.join(helpers.ROOT, "both.mp4"), seconds=2,
                                          processes=1, cancelled=cancelled, preset="veryfast")

        def preview():
            got["clip"] = engine.fx_clip(self.folder, project, "snowcc", processes=1)
        threads = [threading.Thread(target=export), threading.Thread(target=preview)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertFalse(got["export"])
        self.assertTrue(got["clip"])
        info = video.probe(os.path.join(self.folder, "clips", got["clip"]), overlay=True)
        self.assertEqual(info["frames"], timeline.project_frames(project))
        projects.apply(projects.load(self.project["id"]), {"layers": []})

    def test_hidden_track_and_switched_off_clip(self):
        tracks = self.project["tracks"]
        project = projects.apply(projects.load(self.project["id"]), {
            "tracks": [dict(tracks[0], on=False)] + tracks[1:],
            "layers": [{"id": "snowh1", "kind": "snow", "track": tracks[0]["id"], "params": {"amount": 100}},
                       {"id": "snowh2", "kind": "snow", "track": tracks[1]["id"], "on": False}]})
        self.assertEqual(engine.Scene(self.folder, project, (W, H)).layers, [])
        projects.apply(projects.load(self.project["id"]), {"tracks": tracks, "layers": []})

    def test_old_project_is_upgraded(self):
        import json
        src = helpers.make_loop(os.path.join(helpers.ROOT, "old.mp4"), 160, 90, 12)
        project = projects.create(src, "old.mp4")
        path = os.path.join(projects.folder(project["id"]), "project.json")
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for key in ("tracks", "length", "proxy"):
            raw.pop(key)
        files = os.path.join(projects.folder(project["id"]), "files")
        os.makedirs(files)
        helpers.make_loop(os.path.join(files, "fileo1-abc123.mp4"), 160, 90, 12)
        raw["layers"] = [{"id": "snowo1", "kind": "snow", "name": "Tuyết", "on": True, "behind": False,
                          "feather": 0, "seed": 1, "params": {}},
                         {"id": "glowo1", "kind": "glow", "name": "Sáng", "on": True, "behind": False,
                          "feather": 0, "seed": 1, "params": {}},
                         {"id": "fileo1", "kind": "file", "name": "Bụi", "on": True, "behind": False, "feather": 0,
                          "seed": 1, "params": {"blend": "screen"},
                          "file": {"stored": "fileo1-abc123.mp4", "name": "Bụi sáng.mp4",
                                   "info": video.probe(os.path.join(files, "fileo1-abc123.mp4"), overlay=True)}}]
        raw["preview"] = "preview-1-abcd.mp4"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(raw, f)
        old = projects.load(project["id"])
        self.assertEqual(old["length"], 30)
        self.assertEqual(len(old["tracks"]), 3)
        self.assertEqual([lay["track"] for lay in old["layers"]], [t["id"] for t in old["tracks"]])
        self.assertTrue(all(timeline.Span(lay, timeline.project_frames(old), 24).full for lay in old["layers"]))
        self.assertNotIn("preview", old)
        item = library.load(old["layers"][2]["file"]["lib"])  # the outside file went into the library
        self.assertEqual(item["name"], "Bụi sáng")
        self.assertEqual(projects.load(project["id"])["layers"][2]["file"]["lib"], item["id"])  # once only
        saved = projects.apply(old, {"name": "Cũ"})
        self.assertEqual(len(saved["tracks"]), 3)
        self.assertEqual(os.listdir(files), [])  # the project's own copy is no longer needed


class Concurrency(unittest.TestCase):
    def test_late_writers_keep_newer_changes(self):
        """A mask save or upload that started from an older copy must not undo what was saved meanwhile."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "race.mp4"), 160, 90, 12)
        project = projects.create(src, "race.mp4")
        stale = projects.load(project["id"])
        newer = projects.apply(projects.load(project["id"]), {"layers": [{"id": "snowcc", "kind": "snow"}], "name": "Mới"})
        after = projects.save_mask(stale, "freeze", png([0, 0, 40, 90], (160, 90)))
        self.assertEqual([lay["id"] for lay in after["layers"]], ["snowcc"])
        self.assertEqual(after["name"], "Mới")
        self.assertGreater(after["rev"], newer["rev"])
        stale_again = dict(stale)
        later = projects.apply(stale_again, {"fg": {"feather": 9}})
        self.assertEqual(later["layers"][0]["id"], "snowcc")
        self.assertGreater(later["rev"], after["rev"])  # never a rev seen before

    def test_names(self):
        import unicodedata
        self.assertEqual(projects.clean_name(unicodedata.normalize("NFD", "Cảnh đêm mưa.mov")), "Cảnh đêm mưa")
        self.assertEqual(projects.clean_name("../../x<>y.mp4"), "x y")


class Replace(unittest.TestCase):
    def test_new_video_keeps_the_timeline(self):
        """V1 replaced by a video of another frame rate and loop length: clips keep their times in seconds, a clip
        to the end still runs to the end, the areas stay, the light copy is made again for the new video."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "old24.mp4"), 160, 90, 24, "24")  # 1 s at 24 fps
        project = projects.create(src, "old24.mp4")
        folder = projects.folder(project["id"])
        tid = project["tracks"][0]["id"]
        project = projects.apply(project, {"still": {"source": "frame", "frame": 12}, "layers": [
            {"id": "snowrr", "kind": "snow", "track": tid, "start": 24, "end": 720},
            {"id": "glowrr", "kind": "glow", "track": project["tracks"][1]["id"], "start": 48, "end": 96}]})
        project = projects.save_mask(project, "freeze", png([0, 0, 40, 90], (160, 90)))
        projects.set_proxy(project["id"], engine.make_base_proxy(folder, project), project["video"]["file"])
        new = helpers.make_loop(os.path.join(helpers.ROOT, "new30.mp4"), 160, 90, 60, "30")  # 2 s at 30 fps
        project, notes = projects.replace_video(projects.load(project["id"]), new, "Cảnh mới.mov")
        self.assertEqual((project["video"]["fps"], project["video"]["frames"]), ("30", 60))
        self.assertEqual(timeline.project_frames(project), 900)
        clips = {c["id"]: (c["start"], c["end"]) for c in project["layers"]}
        self.assertEqual(clips, {"snowrr": (30, 900), "glowrr": (60, 120)})
        self.assertEqual(project["still"]["frame"], 15)
        self.assertIsNone(project["proxy"])
        self.assertTrue(project["video"]["file"].startswith("input-"))
        self.assertFalse(os.path.exists(os.path.join(folder, "input.mp4")))  # the old video and its copy are gone
        self.assertFalse(os.path.exists(os.path.join(folder, "proxy.mp4")))
        self.assertTrue(os.path.exists(engine.mask_path(folder, "freeze")))
        self.assertTrue(any("2 giây" in n for n in notes), notes)
        frame = engine.Scene(folder, project, (160, 90)).frame(450, video.read_frame(engine.input_path(folder, project),
                                                                                    project["video"], 30))
        self.assertEqual(frame.shape, (90, 160, 3))
        name = engine.make_base_proxy(folder, project)
        self.assertNotEqual(name, "proxy.mp4")
        self.assertEqual(projects.set_proxy(project["id"], name, "input.mp4")["proxy"], None)  # made for the old one
        self.assertEqual(projects.set_proxy(project["id"], name, project["video"]["file"])["proxy"], name)

    def test_clips_past_the_new_end_keep_their_length(self):
        """A shorter longest video: a clip beyond its end slides back whole (never a one-frame flash), up to the
        clip before it on its track; one with no room left is turned off. Clips to the end still run to it."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "two24.mp4"), 160, 90, 48, "24")  # 2 s: 1 min = 1440
        project = projects.create(src, "two24.mp4")
        t0, t1 = project["tracks"][0]["id"], project["tracks"][1]["id"]
        project = projects.apply(project, {"length": 60, "layers": [
            {"id": "endsnow", "kind": "snow", "track": t0, "start": 1368, "end": 1440},  # 57 s to the end
            {"id": "lategl", "kind": "glow", "track": t1, "start": 1272, "end": 1392},  # 53-58 s
            {"id": "lastgl", "kind": "glow", "track": t1, "start": 1392, "end": 1416}]})  # 58-59 s
        new = helpers.make_loop(os.path.join(helpers.ROOT, "eleven24.mp4"), 160, 90, 264, "24")  # 11 s: 1 min = 55 s
        project, notes = projects.replace_video(projects.load(project["id"]), new, "eleven.mp4")
        clips = {c["id"]: (c["start"], c["end"], c["on"]) for c in project["layers"]}
        self.assertEqual(timeline.project_frames(project), 1320)
        self.assertEqual(clips["endsnow"], (1248, 1320, True))  # 52-55 s: to the end, 3 s as before
        self.assertEqual(clips["lategl"], (1200, 1320, True))  # 50-55 s
        self.assertFalse(clips["lastgl"][2])  # no room after it: turned off, not squeezed into a frame
        self.assertTrue(any("dời lên trước" in n for n in notes), notes)
        self.assertTrue(any("đã được tắt" in n for n in notes), notes)

    def test_a_new_frame_rate_alone_is_no_news(self):
        src = helpers.make_loop(os.path.join(helpers.ROOT, "rate24.mp4"), 160, 90, 48, "24")
        project = projects.create(src, "rate24.mp4")
        new = helpers.make_loop(os.path.join(helpers.ROOT, "rate30.mp4"), 160, 90, 60, "30")
        project, notes = projects.replace_video(projects.load(project["id"]), new, "rate30.mp4")
        self.assertEqual(notes, [])

    def test_a_larger_video_gets_the_original_picture_and_softness(self):
        """The still picture is fitted again from the picture as chosen (not enlarged from the small copy), and soft
        edges keep their share of the picture."""
        src = helpers.make_loop(os.path.join(helpers.ROOT, "small.mp4"), 160, 90, 24, "24")
        project = projects.create(src, "small.mp4")
        project = projects.apply(project, {"freeze": {"feather": 20}, "layers": [
            {"id": "softly", "kind": "snow", "track": project["tracks"][0]["id"], "feather": 7}]})
        rng = np.random.default_rng(3)
        pic = os.path.join(helpers.ROOT, "pic-upload")
        Image.fromarray(rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)).save(pic, "PNG")
        project = projects.set_still_image(project, pic, "ảnh gốc.png")
        new = helpers.make_loop(os.path.join(helpers.ROOT, "large.mp4"), 320, 180, 24, "24")
        project, notes = projects.replace_video(projects.load(project["id"]), new, "large.mp4")
        folder = projects.folder(project["id"])
        with Image.open(os.path.join(folder, "still.png")) as im:
            still = np.asarray(im.convert("RGB"))
        with Image.open(os.path.join(folder, projects.STILL_SOURCE)) as im:
            self.assertTrue(np.array_equal(still, np.asarray(im.convert("RGB"))))  # the same size: every pixel
        self.assertEqual(project["freeze"]["feather"], 40)
        self.assertEqual(project["layers"][0]["feather"], 14)
        self.assertFalse(any("kém nét" in n for n in notes), notes)


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
