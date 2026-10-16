"""--selftest: the installed app makes a real video and checks it (CI runs it on Windows after every build).

It checks the bundled page and ffmpeg, then in a temporary data folder makes a 2-second 1080p loop, paints the
masks, puts every kind of clip on the timeline (built-in effects, an outside video with transparency and a PNG from
the library, clips that fade in and out, a clip cut in two) and checks: every effect returns exactly to its first
frame after one loop; the still areas are the picture pixel for pixel; the exported video, rendered in pieces by
worker processes, decodes with the right number of frames, BT.709 tags, nothing that names the tools, and a seam no
bigger than any other frame change; the live preview's light copies keep their transparency; the local server
only answers its own page.
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

from . import effects, engine, library, paths, projects, sheet, timeline, video

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
    yy, xx = np.mgrid[0:360, 0:640]
    scene = np.dstack([0.2 + 0.5 * (((xx // 16) + (yy // 16)) % 2), 0.3 + 0.4 * xx / 640, 0.25 + 0.5 * yy / 360])
    for kind in effects.KINDS:
        third = {"style": "dust"} if kind == "glow" else {"wind": 60}
        if kind == "leaves":  # every way a leaf falls, and both kinds of leaf
            third = {"wind": 60, "gust": 80, "spin": 100, "shape": "mixed"}
        for extra in ({}, {"wind": -80, "speed": 100}, third):
            fx = effects.make(kind, extra, 5, P, FPS, 640, 360)
            canvas = effects.Canvas(640, 360)
            frames = []
            for t in (0, P):
                pic = scene.astype(np.float32)  # (a lens effect changes the picture under it: something to see)
                if fx.lens:
                    fx.apply(t, pic, None, 1.0)
                else:
                    fx.draw(t, canvas, None)
                    canvas.flush(pic, fx.blend)
                frames.append(pic)
            diff = float(np.abs(frames[0] - frames[1]).max())
            if not check(f"{kind} {extra or ''} returns to frame 0 after one loop", diff < 1 / 512, f"max diff {diff:.6f}"):
                return
    picture = Image.fromarray((scene * 255 + 0.5).astype(np.uint8))
    for kind in ("snow", "glass", "steam"):  # placed and turned in 3D: drawn on a sheet, warped onto the frame
        fx = sheet.Sheet(kind, {"rot_x": 20, "rot_y": -30, "rot_z": 8, "persp": 60, "scale": 70}, 5, P, FPS, 640, 360)
        a, b = (np.asarray(fx.draw(t, picture.copy())).astype(int) for t in (0, P))
        diff = int(np.abs(a - b).max())
        if not check(f"{kind} turned in 3D returns to frame 0 after one loop", diff <= 1, f"max diff {diff}"):
            return


def server_checks():
    from . import server
    app = server.App(desktop=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = server.start(app, port)
    base = f"http://127.0.0.1:{port}"
    local = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with local.open(base + "/", timeout=10) as r:
            check("server: page", r.status == 200 and b"Bokin FX" in r.read())
        with local.open(base + "/api/state", timeout=10) as r:
            check("server: state", json.loads(r.read())["app"] == paths.APP_ID)

        def status(req):
            try:
                with local.open(req, timeout=10) as r:
                    return r.status
            except urllib.error.HTTPError as e:
                return e.code
        check("server: POST without X-Bokin refused",
              status(urllib.request.Request(base + "/api/export", data=b"{}", method="POST")) == 403)
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
        mov = os.path.join(work, "overlay.mov")
        make_alpha_mov(mov)
        effect = library.add(mov, "overlay.mov")
        logo = os.path.join(work, "logo.png")
        im = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
        ImageDraw.Draw(im).ellipse([20, 20, 180, 180], fill=(255, 200, 120, 200))
        im.save(logo)
        picture = library.add(logo, "logo.png")
        check("library: outside video with transparency read", effect["info"]["alpha"], str(effect["info"]))
        effect = library.make_proxy(effect["id"])
        proxy = video.probe(library.proxy_path(effect), overlay=True)
        check("library: light copy keeps the transparency", proxy["alpha"] and proxy["codec"] == "vp9", str(proxy))

        seconds = 4  # two loops: the clips run over both, in pieces made by two worker processes
        total = timeline.output_frames(info, seconds)
        tracks = [t["id"] for t in project["tracks"]]
        layers = [{"id": "snow01", "kind": "snow", "params": {"amount": 60}, "seed": 1, "track": tracks[0]},
                  {"id": "leaf01", "kind": "leaves", "behind": True, "params": {}, "seed": 2, "track": tracks[1],
                   "start": 10, "end": 60, "fade_in": 0.3, "fade_out": 0.3},
                  {"id": "leaf02", "kind": "leaves", "behind": True, "params": {}, "seed": 2, "track": tracks[1],
                   "start": 60, "end": total},  # the second half of a cut clip, ending with the video
                  {"id": "rain01", "kind": "rain", "behind": True, "params": {"amount": 25}, "seed": 3,
                   "track": tracks[2], "start": 0, "end": 30},
                  {"id": "glow01", "kind": "glow", "params": {}, "seed": 4, "track": tracks[2], "start": 30},
                  # the outside files turned in 3D (warped onto their quads)
                  {"id": "file01", "kind": "file", "lib": effect["id"], "seed": 6, "track": tracks[0] + "x",
                   "params": {"blend": "normal", "rot_y": 15, "rot_z": 5}},
                  {"id": "file02", "kind": "file", "lib": picture["id"], "seed": 7, "track": tracks[1] + "x",
                   "params": {"scale": 30, "blend": "normal", "rot_x": 25, "rot_y": -30, "rot_z": 12},
                   "start": 20, "end": 70, "fade_in": 0.5, "fade_out": 0.5},
                  # built-in effects placed and turned in 3D: snow and water on a window seen at an angle (the
                  # glass: a lens, only its change laid back), a cup's steam turned
                  {"id": "snow02", "kind": "snow", "seed": 8, "track": tracks[2] + "x", "behind": True,
                   "params": {"amount": 70, "scale": 55, "x": -18, "y": -8, "rot_x": 15, "rot_y": 35, "rot_z": -6,
                              "persp": 70}},
                  {"id": "glass01", "kind": "glass", "seed": 9, "track": tracks[0] + "y",
                   "params": {"scale": 45, "x": 22, "y": -12, "rot_x": 10, "rot_y": -35, "rot_z": 4, "persp": 70}},
                  {"id": "steam01", "kind": "steam", "seed": 10, "track": tracks[1] + "y",
                   "params": {"scale": 40, "x": -25, "y": 20, "rot_y": 25, "rot_z": 5}, "start": 30, "end": 80,
                   "fade_in": 0.3, "fade_out": 0.3}]
        extra = ["x", "y"]
        project = projects.apply(project, {"tracks": project["tracks"] + [{"id": tracks[i] + k} for k in extra
                                                                          for i in range(3) if (i, k) != (2, "y")],
                                           "layers": layers})
        project = projects.save_mask(project, "snow01",
                                     png_bytes(lambda d: d.rectangle([mw // 3, 0, mw, mh // 2], fill=255), (mw, mh)))
        check("timeline saved", [lay["track"] for lay in project["layers"]][5:7] == [tracks[0] + "x", tracks[1] + "x"])
        small = engine.Scene(folder, project, (64, 36))
        check("built-in effects turned in 3D: drawn on their sheets",
              [isinstance(d, sheet.Sheet) for d, _, _ in small.layers][-3:] == [True, True, True])
        small.close()

        # still areas are the picture, pixel for pixel, where nothing is drawn over them
        bare = engine.Scene(folder, dict(project, layers=[]), (W, H))
        base = video.read_frame(engine.input_path(folder, project), info, 20)
        still = video.read_frame(engine.input_path(folder, project), info, 0)
        out = bare.frame(20, base)
        inner = W // 4 - 8
        check("still area exact", bool((out[:, :inner] == still[:, :inner]).all()))
        check("outside the still area untouched", bool((out[:, W // 4 + 8:] == base[:, W // 4 + 8:]).all()))

        made = os.path.join(work, "export.mp4")
        t0 = time.time()
        ok = engine.render(folder, project, made, seconds=seconds, processes=2)
        took = time.time() - t0
        check("export rendered", ok, f"{took:.1f} s for {total} frames at {W}x{H} with {len(layers)} clips")
        got = video.probe(made)
        check("export: size, frames, BT.709", (got["w"], got["h"], got["frames"], got["matrix"]) == (W, H, total, "bt709"),
              str(got))
        tags = video.run(["-i", made, "-f", "null", "-", "-frames:v", "1"])
        check("export: colour tags", "bt709" in tags)
        data = open(made, "rb").read()
        traces = [s for s in ("x264", "Lavf", "Lavc", "FFmpeg", "ffmpeg") if s.encode() in data]
        check("export: nothing names the tools", not traces, ", ".join(traces))
        small = (480, 270)
        with video.Frames(made, got, small) as frames:
            seq = [f.astype(np.int16) for f in frames]
        steps = [float(np.abs(seq[(i + 1) % total] - seq[i]).mean()) for i in range(total)]
        seam, middle, others = steps[-1], steps[total // 2 - 1], steps[:-1]
        check("export: seamless loop", seam <= max(others) * 1.25 + 0.5,
              f"seam {seam:.2f}, other frames {min(others):.2f}..{max(others):.2f}")
        check("export: pieces joined without a jump", middle <= sorted(others)[len(others) * 9 // 10] * 1.5 + 0.5,
              f"{middle:.2f}")

        name = engine.fx_clip(folder, dict(project, length=30), "snow01", processes=2)
        clip = video.probe(os.path.join(folder, "clips", name), overlay=True)
        check("live preview: effect clip with transparency", clip["alpha"] and clip["frames"] ==
              timeline.output_frames(info, 30), str(clip))
        proxy_name = engine.make_base_proxy(folder, project)
        check("live preview: light copy of the video", video.probe(os.path.join(folder, proxy_name))["frames"] == P)

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
