import json
import os
import unittest

from tests import helpers  # noqa: F401  (a temporary data folder)
from fx import keys, paths


class Keys(unittest.TestCase):
    def setUp(self):
        self.addCleanup(self.forget)
        self.forget()

    @staticmethod
    def forget():
        try:
            os.remove(os.path.join(paths.data_dir(), keys.FILE))
        except OSError:
            pass

    def test_defaults(self):
        ids = [a["id"] for a in keys.ACTIONS]
        self.assertEqual(len(ids), len(set(ids)))
        for a in keys.ACTIONS:
            self.assertIn(a["group"], keys.GROUPS)
            self.assertIn(a["where"], ("all", "edit", "paint"))
            self.assertTrue(a["keys"], a["id"])
            for k in a["keys"]:
                self.assertTrue(keys.usable(k), k)
        # no two actions that work at the same time share a key, so every default is in use
        self.assertEqual(keys.effective({}), {a["id"]: a["keys"] for a in keys.ACTIONS})
        self.assertEqual(keys.clean({a["id"]: a["keys"] for a in keys.ACTIONS}), {a["id"]: a["keys"] for a in keys.ACTIONS})

    def test_names(self):
        self.assertEqual(keys.name("Ctrl+Shift+KeyZ"), "Ctrl+Shift+Z")
        self.assertEqual(keys.name("Space"), "Phím cách")
        self.assertEqual(keys.name("Ctrl+Digit0"), "Ctrl+0")
        self.assertEqual(keys.name("Backslash"), "\\")

    def test_clean(self):
        self.assertEqual(keys.clean(None), {})
        self.assertEqual(keys.clean([1]), {})
        got = keys.clean({
            "nope": ["KeyQ"],  # no such action
            "razorTool": ["KeyD", "KeyD", "Ctrl+KeyW", "F5", "Shift+F12", "KeyÄ", "../x", 5, "Alt+F4", "Meta+KeyA"],
            "snap": "KeyS",  # not a list
            "export": ["KeyA", "KeyB", "KeyE", "KeyF", "KeyG"],  # at most MAX_KEYS
        })
        self.assertEqual(got, {"razorTool": ["KeyD"], "export": ["KeyA", "KeyB", "KeyE", "KeyF"]})
        # two actions that work at the same time: the first in the list keeps the key
        self.assertEqual(keys.clean({"undo": ["KeyQ"], "redo": ["KeyQ", "KeyW"]}), {"undo": ["KeyQ"], "redo": ["KeyW"]})
        # one only while painting, the other only when not: they may share it
        self.assertEqual(keys.clean({"razorTool": ["KeyQ"], "brushTool": ["KeyQ"]}),
                         {"razorTool": ["KeyQ"], "brushTool": ["KeyQ"]})

    def test_owner_keys_win(self):
        # V for the razor: the select tool (whose default it was) loses it; painting keeps its own B
        eff = keys.effective(keys.clean({"razorTool": ["KeyV"], "polyTool": ["KeyB"]}))
        self.assertEqual(eff["razorTool"], ["KeyV"])
        self.assertEqual(eff["selectTool"], [])
        self.assertEqual(eff["brushTool"], [])
        self.assertEqual(eff["polyTool"], ["KeyB"])
        self.assertEqual(eff["play"], ["KeyL"])  # (L was the polygon's too, while painting)
        # a key given to an action that works always is taken from every other
        eff = keys.effective({"export": ["KeyL"]})
        self.assertEqual((eff["play"], eff["polyTool"]), ([], []))

    def test_saved(self):
        self.assertEqual(keys.load(), {})
        kept = keys.save({"razorTool": ["KeyD"], "snap": ["KeyS"], "undo": []})
        self.assertEqual(kept, {"razorTool": ["KeyD"], "undo": []})  # (snap: as its default)
        self.assertEqual(keys.load(), kept)
        v = keys.view()
        self.assertEqual(v["custom"], kept)
        self.assertEqual(v["keys"]["razorTool"], ["KeyD"])
        self.assertEqual(v["keys"]["undo"], [])
        self.assertEqual(v["keys"]["snap"], ["KeyS"])
        with open(os.path.join(paths.data_dir(), keys.FILE), "w", encoding="utf-8") as f:
            f.write("{not json")
        self.assertEqual(keys.load(), {})
        with open(os.path.join(paths.data_dir(), keys.FILE), "w", encoding="utf-8") as f:
            json.dump(["KeyA"], f)
        self.assertEqual(keys.load(), {})

    def test_guide_has_the_current_defaults(self):
        with open(keys.guide_path(), encoding="utf-8") as f:
            text = f.read()
        self.assertIn(keys.markdown(), text, "HUONG-DAN.md: chạy `python -m fx.keys` để ghi lại phím mặc định")
        for a in keys.ACTIONS:
            self.assertIn(a["label"], text)


if __name__ == "__main__":
    unittest.main()
