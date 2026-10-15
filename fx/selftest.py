"""--selftest: the installed app makes a real video and checks it (CI runs it on Windows after every build).

It checks the bundled page and ffmpeg, then in a temporary data folder makes a 2-second 1080p loop, paints the
masks, adds every kind of layer (with an outside video that has transparency and a PNG) and checks:
every effect returns exactly to its first frame after one loop; the still areas are the picture pixel for pixel;
the exported loop and a 3x longer copy decode with the right number of frames, BT.709 tags, nothing that names
the tools, and a seam no bigger than any other frame change; the local server only answers its own page.
"""
import io
import json
import math
import os
import shutil
import socket
import subprocess
import tempfile
import time
import traceback
import urllib.error
import urllib.request

import numpy as np
from PIL import Image, ImageDraw

from . import effects, engine, paths, projects, video

W, H, FPS, P = 1920, 1080, 24, 48
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(("OK   " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)
    return ok


def make_loop(path):
    """A 2 s loop: a gradient with a bright disc going round once per loop."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.dstack([40 + 70 * yy / H, 60 + 50 * xx / W, 110 + 60 * yy / H])
    enc = video.Encoder(path, W, H, str(FPS), crf=18, preset="veryfast")
    for t in range(P):
        f = bg.copy()
        a = 2 * math.pi * t / P
        cx, cy = W * 0.6 + 300 * math.cos(a), H * 0.5 + 200 * math.sin(a)
        f[(xx - cx) ** 2 + (yy - cy) ** 2 < 80 ** 2] = (240, 210, 130)
        enc.write(f.astype(np.uint8))
    enc.close()


def make_alpha_mov(path, seconds=3, fps=30, size=(640, 360)):
    """An outside effect with transparency (QuickTime Animation, RGBA): a soft square drifting across."""
    w, h = size
    args = [video.ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgba",
            "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-", "-c:v", "qtrle", path]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, **video.NO_WINDOW)
    for i in range(seconds * fps):
        f = np.zeros((h, w, 4), np.uint8)
        x = int((i * 7) % (w - 60))
        f[100:160, x:x + 60] = (255, 255, 255, 128)
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("could not make the test overlay")


def png_bytes(draw, size, mode="L"):
    im = Image.new(mode, size, 0)
    draw(ImageDraw.Draw(im))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def periodic_checks():
    for kind in ("snow", "leaves", "rain", "glow"):
        for extra in ({}, {"wind": -80, "speed": 100}, {"style": "dust"} if kind == "glow" else {"wind": 60}):
            fx = effects.make(kind, extra, 5, P, FPS, 640, 360)
            canvas = effects.Canvas(640, 360)
            frames = []
            for t in (0, P):
                pic = np.full((360, 640, 3), 0.3, np.float32)
                fx.draw(t, canvas, None)
                canvas.flush(pic, fx.blend)
                frames.append(pic)
            diff = float(np.abs(frames[0] - frames[1]).max())
            if not check(f"{kind} {extra or ''} returns to frame 0 after one loop", diff < 1 / 512, f"max diff {diff:.6f}"):
                return


def server_checks():
    from . import server
    app = server.App(desktop=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = server.start(app, port)
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=10) as r:
            check("server: page", r.status == 200 and b"Bokin FX" in r.read())
        with urllib.request.urlopen(base + "/api/state", timeout=10) as r:
            check("server: state", json.loads(r.read())["app"] == paths.APP_ID)

        def status(req):
            try:
                with urllib.request.urlopen(req, timeout=10) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code
        check("server: POST without X-Bokin refused",
              status(urllib.request.Request(base + "/api/preview", data=b"{}", method="POST")) == 403)
        check("server: other host names refused",
              status(urllib.request.Request(base + "/api/state", headers={"Host": "example.com"})) == 403)
        check("server: no path tricks", status(base + "/media/out?name=..%2F..%2Fx.mp4") == 404)
    finally:
        srv.shutdown()


def run():
    started = time.time()
    print(f"Bokin FX {paths.version()} self-test", flush=True)
    old_env = {k: os.environ.get(k) for k in ("BOKIN_FX_DATA", "BOKIN_FX_OUT")}
    work = tempfile.mkdtemp(prefix="bokinfx-selftest-")
    try:
        with open(os.path.join(paths.resource_dir(), "index.html"), encoding="utf-8") as f:
            check("page bundled", "Bokin FX" in f.read())
        encoders = subprocess.run([video.ffmpeg_exe(), "-hide_banner", "-encoders"], capture_output=True,
                                  stdin=subprocess.DEVNULL, **video.NO_WINDOW).stdout.decode("utf-8", "replace")
        check("ffmpeg with libx264", "libx264" in encoders)

        os.environ["BOKIN_FX_DATA"] = os.path.join(work, "data")
        os.environ["BOKIN_FX_OUT"] = os.path.join(work, "out")
        periodic_checks()

        src = os.path.join(work, "loop.mp4")
        make_loop(src)
        project = projects.create(src, "selftest loop.mp4")
        info = project["video"]
        check("video read", info["frames"] == P and info["w"] == W and info["matrix"] == "bt709", str(info))
        pid = project["id"]
        folder = projects.folder(pid)
        mw, mh = projects.mask_size(info)
        projects.save_mask(project, "freeze", png_bytes(lambda d: d.rectangle([0, 0, mw // 4, mh], fill=255), (mw, mh)))
        projects.save_mask(project, "fg", png_bytes(lambda d: d.rectangle([0, int(mh * .85), mw, mh], fill=255), (mw, mh)))
        layers = [{"id": "snow01", "kind": "snow", "params": {"amount": 60}, "seed": 1},
                  {"id": "leaf01", "kind": "leaves", "behind": True, "params": {}, "seed": 2},
                  {"id": "rain01", "kind": "rain", "behind": True, "params": {"amount": 25}, "seed": 3},
                  {"id": "glow01", "kind": "glow", "params": {}, "seed": 4},
                  {"id": "dust01", "kind": "glow", "params": {"style": "dust"}, "seed": 5},
                  {"id": "file01", "kind": "file", "params": {}, "seed": 6},
                  {"id": "file02", "kind": "file", "params": {"scale": 30}, "seed": 7}]
        project = projects.apply(project, {"layers": layers})
        project = projects.save_mask(project, "snow01",
                                     png_bytes(lambda d: d.rectangle([mw // 3, 0, mw, mh // 2], fill=255), (mw, mh)))
        mov = os.path.join(work, "overlay.mov")
        make_alpha_mov(mov)
        project = projects.set_layer_file(project, "file01", mov, "overlay.mov")
        logo = os.path.join(work, "logo.png")
        im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse([20, 20, 180, 180], fill=(255, 200, 120, 200))
        im.save(logo)
        project = projects.set_layer_file(project, "file02", logo, "logo.png")
        file01 = next(lay for lay in project["layers"] if lay["id"] == "file01")
        check("outside video with transparency read", file01["file"]["info"]["alpha"] and
              file01["params"]["blend"] == "normal", str(file01["file"]["info"]))

        # still areas are the picture, pixel for pixel, where nothing is drawn over them
        bare = engine.Scene(folder, dict(project, layers=[]), (W, H))
        base = video.read_frame(engine.input_path(folder, project), info, 20)
        still = video.read_frame(engine.input_path(folder, project), info, 0)
        out = bare.frame(20, base)
        inner = W // 4 - 8
        check("still area exact", bool((out[:, :inner] == still[:, :inner]).all()))
        check("outside the still area untouched", bool((out[:, W // 4 + 8:] == base[:, W // 4 + 8:]).all()))

        loop = os.path.join(work, "export.mp4")
        t0 = time.time()
        ok = engine.render(folder, project, loop)
        took = time.time() - t0
        check("export rendered", ok, f"{took:.1f} s for {P} frames at {W}x{H} with {len(layers)} layers")
        got = video.probe(loop)
        check("export: size, frames, BT.709", (got["w"], got["h"], got["frames"], got["matrix"]) == (W, H, P, "bt709"),
              str(got))
        tags = video.run(["-i", loop, "-f", "null", "-", "-frames:v", "1"])
        check("export: colour tags", "bt709" in tags)
        data = open(loop, "rb").read()
        traces = [s for s in ("x264", "Lavf", "Lavc", "FFmpeg", "ffmpeg") if s.encode() in data]
        check("export: nothing names the tools", not traces, ", ".join(traces))
        small = (480, 270)
        with video.Frames(loop, got, small) as frames:
            seq = [f.astype(np.int16) for f in frames]
        steps = [float(np.abs(seq[(i + 1) % P] - seq[i]).mean()) for i in range(P)]
        seam, others = steps[-1], steps[:-1]
        check("export: seamless loop", seam <= max(others) * 1.25 + 0.5,
              f"seam {seam:.2f}, other frames {min(others):.2f}..{max(others):.2f}")

        longer = os.path.join(work, "longer.mp4")
        video.repeat(loop, longer, 3, seconds=3 * P / FPS)
        got3 = video.probe(longer)
        data3 = open(longer, "rb").read()
        check("3x longer copy", got3["frames"] == 3 * P and not any(s.encode() in data3 for s in ("Lavf", "x264")),
              str(got3["frames"]))

        look = engine.Looks().frame(folder, project, 10, 640)
        check("single-frame look", look.shape == (360, 640, 3))
        server_checks()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        check("self-test ran to the end", False)
    finally:
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(work, ignore_errors=True)
    passed = all(RESULTS) and bool(RESULTS)
    print(f"self-test {'passed' if passed else 'FAILED'} in {time.time() - started:.0f} s", flush=True)
    return 0 if passed else 1
