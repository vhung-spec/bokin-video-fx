"""Shared by the tests: a temporary data folder, and small test videos."""
import atexit
import math
import os
import shutil
import tempfile

import numpy as np

ROOT = tempfile.mkdtemp(prefix="bokinfx-tests-")
os.environ["BOKIN_FX_DATA"] = os.path.join(ROOT, "data")
os.environ["BOKIN_FX_OUT"] = os.path.join(ROOT, "out")


@atexit.register
def cleanup():
    shutil.rmtree(ROOT, ignore_errors=True)


def make_loop(path, w=320, h=180, frames=24, fps="24"):
    """A loop: a gradient with a disc going round once per loop."""
    from fx import video
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.dstack([40 + 70 * yy / h, 60 + 50 * xx / w, 110 + 60 * yy / h])
    enc = video.Encoder(path, w, h, fps, crf=18, preset="veryfast")
    for t in range(frames):
        f = bg.copy()
        a = 2 * math.pi * t / frames
        cx, cy = w * 0.6 + w * 0.2 * math.cos(a), h * 0.5 + h * 0.2 * math.sin(a)
        f[(xx - cx) ** 2 + (yy - cy) ** 2 < (h * 0.08) ** 2] = (240, 210, 130)
        enc.write(f.astype(np.uint8))
    enc.close()
    return path
