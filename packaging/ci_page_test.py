"""The page in a real browser (CI: Microsoft Edge on Windows, the browser the app opens).

  python packaging/ci_page_test.py [screenshot folder]

Starts the app's server on test data, then works the page as a person would: opens a loop, imports an effect video
and a picture into the library, drags the effect onto the timeline, adds snow, plays, cuts a clip, moves and trims
clips, undoes, paints a still area and exports a 30-second video. It checks the timeline the page saved, that the
live preview plays the light copies (H.264 and transparent WebM), and that the page reports no script errors.
Screenshots of each step go to the folder given (default: page-test).

The browser: Edge when installed (Windows), else the Chromium of Playwright, or BOKIN_BROWSER (a path).
"""
import math
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# the render's worker processes run this file again: they must find the same folders
WORK = os.environ.setdefault("BOKIN_PAGE_TEST", tempfile.mkdtemp(prefix="bokinfx-page-"))
os.environ.setdefault("BOKIN_FX_DATA", os.path.join(WORK, "data"))
os.environ.setdefault("BOKIN_FX_OUT", os.path.join(WORK, "out"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from fx import server, video  # noqa: E402

SHOTS = sys.argv[1] if len(sys.argv) > 1 else "page-test"
FAILED = []


def check(name, ok, detail=""):
    print(("OK   " if ok else "FAIL ") + name + (f": {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(name)


def media():
    """A 2 s loop (960x540), an effect on black (3 s) and a see-through picture."""
    folder = os.path.join(WORK, "media")
    os.makedirs(folder)
    w, h, n = 960, 540, 48
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg = np.dstack([40 + 70 * yy / h, 60 + 50 * xx / w, 110 + 60 * yy / h])
    enc = video.Encoder(os.path.join(folder, "loop.mp4"), w, h, "24", crf=18, preset="veryfast")
    for t in range(n):
        f = bg.copy()
        a = 2 * math.pi * t / n
        cx, cy = w * 0.6 + 200 * math.cos(a), h * 0.5 + 120 * math.sin(a)
        f[(xx - cx) ** 2 + (yy - cy) ** 2 < 50 ** 2] = (240, 210, 130)
        enc.write(f.astype(np.uint8))
    enc.close()
    args = [video.ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x360",
            "-framerate", "24", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", os.path.join(folder, "dust.mp4")]
    proc = subprocess.Popen(args, stdin=subprocess.PIPE, **video.NO_WINDOW)
    pts = np.random.default_rng(1).random((60, 2))
    for i in range(72):
        f = np.zeros((360, 640, 3), np.uint8)
        for x, y in pts:
            px, py = int(x * 640), int(((y + i / 72) % 1) * 360)
            f[max(0, py - 3):py + 3, max(0, px - 3):px + 3] = 255
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    enc = video.Encoder(os.path.join(folder, "loop30.mp4"), w, h, "30", crf=18, preset="veryfast")
    for t in range(45):  # another loop, 1.5 s at 30 frames a second, to replace the first one
        f = bg[:, ::-1].copy()
        a = 2 * math.pi * t / 45
        f[(xx - w * 0.4 - 150 * math.cos(a)) ** 2 + (yy - h * 0.5) ** 2 < 60 ** 2] = (120, 220, 160)
        enc.write(f.astype(np.uint8))
    enc.close()
    im = Image.new("RGBA", (400, 200), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse([20, 20, 380, 180], fill=(255, 200, 0, 160))
    im.save(os.path.join(folder, "logo.png"))
    return folder


def upload(folder, name, as_name, kind):
    with open(os.path.join(folder, name), "rb") as f:
        return {"name": as_name, "mimeType": kind, "buffer": f.read()}


def launch(p):
    exe = os.environ.get("BOKIN_BROWSER")
    if exe:
        return p.chromium.launch(executable_path=exe), False
    if os.name == "nt":
        try:
            return p.chromium.launch(channel="msedge"), True
        except Exception as e:  # noqa: BLE001
            print(f"no Edge ({e}); Playwright's Chromium instead", flush=True)
    return p.chromium.launch(), False


def run(page, folder, edge):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "400" not in m.text else None)

    def shot(name):
        page.screenshot(path=os.path.join(SHOTS, name + ".png"))

    def clips():
        return page.evaluate("S.project.layers.map(c => ({id: c.id, kind: c.kind, track: c.track, start: c.start, "
                             "end: c.end, skip: c.skip}))")

    def lane_y(i):  # rows from the top: new track, V4, V3, V2, V1
        b = page.locator(".lane[data-row]").nth(i).bounding_box()
        return b["y"] + b["height"] / 2

    def x_of(sec):
        return page.locator("#tlLanes").bounding_box()["x"] + sec * page.evaluate("S.pps")

    def drag(x0, y0, x1, y1):
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, steps=4)
        page.mouse.move(x1, y1, steps=4)
        page.mouse.up()
        page.wait_for_timeout(300)

    page.wait_for_function("S.schema && S.state")
    shot("01-start")
    page.set_input_files("#fileVideo", upload(folder, "loop.mp4", "Quán cà phê.mp4", "video/mp4"))
    page.wait_for_function("S.project && S.project.id", timeout=60000)
    check("a loop opens as a project", page.evaluate("S.project.name") == "Quán cà phê")
    page.set_input_files("#fileLib", [upload(folder, "dust.mp4", "Bụi sáng nền đen.mp4", "video/mp4"),
                                      upload(folder, "logo.png", "Logo.png", "image/png")])
    page.wait_for_function("S.lib.length === 2", timeout=60000)
    page.wait_for_function("S.project.proxy && S.lib.every(i => i.proxy)", timeout=120000)
    check("library: both files, with light copies", True)

    # the effect video dropped on V2 at 6 s: it runs to the end of the video
    scroll = page.locator("#tlScroll").bounding_box()
    v2 = lane_y(3)
    page.locator(".tile[data-lib]").last.drag_to(page.locator("#tlScroll"), target_position={
        "x": x_of(6) - scroll["x"], "y": v2 - scroll["y"]})
    page.wait_for_timeout(400)
    c = clips()
    check("drop on the timeline", len(c) == 1 and abs(c[0]["start"] - 144) <= 2 and c[0]["end"] == 720, str(c))
    drag(x_of(c[0]["end"] / 24) - 2, v2, x_of(20), v2)  # trim its end to 20 s
    c = clips()
    check("trim the end", abs(c[0]["end"] - 480) <= 3, str(c[0]))
    drag(x_of(c[0]["start"] / 24 + 2), v2, x_of(c[0]["start"] / 24 + 6), v2)  # move it 4 s later
    c = clips()
    check("move", abs(c[0]["start"] - 240) <= 3, str(c[0]))
    start = c[0]["start"]
    drag(x_of(start / 24) + 2, v2, x_of(start / 24 + 2) + 2, v2)  # trim its head by 2 s
    c = clips()
    check("trim the head keeps the picture in place", c[0]["start"] > start and
          abs(c[0]["skip"] - (c[0]["start"] - start) / 24) < 1e-6, str(c[0]))

    page.click('.tab[data-bin="fx"]')
    page.dblclick('.tile[data-kind="snow"]')
    page.wait_for_function("(() => { const c = S.project.layers.find(l => l.kind === 'snow'); "
                           "return c && S.fx[c.id] && S.fx[c.id].ready; })()", timeout=180000)
    check("snow over the whole video, with its preview clip", True)
    shot("02-timeline")

    page.keyboard.press("Space")
    page.wait_for_timeout(2500)
    playing = page.evaluate("""(() => {
        const live = Object.values(S.els).filter(e => e.media.tagName === 'VIDEO' && e.wrap.style.display !== 'none');
        return {time: S.time, base: baseVid.readyState, baseTime: baseVid.currentTime, broken: !!S.baseBroken,
                layers: live.map(e => [e.media.readyState, e.media.currentTime, e.media.paused])};
    })()""")
    shot("03-playing")
    page.keyboard.press("Space")
    check("plays", playing["time"] > 1.5, str(playing))
    if edge:  # Edge plays H.264: the light copy of the loop and the effect video
        check("live preview: the loop plays", playing["base"] >= 2 and not playing["broken"] and playing["baseTime"] > 0,
              str(playing))
    check("live preview: effect clips play", any(rs >= 2 and t > 0 and not paused for rs, t, paused in playing["layers"]),
          str(playing["layers"]))
    page.wait_for_function("document.querySelector('#exact').style.display === 'block'", timeout=60000)
    check("paused: the exact frame from the renderer", True)
    shot("04-paused")

    page.evaluate("(() => { const c = S.project.layers.find(l => l.kind === 'snow'); select(c.id); seek(5); })()")
    page.keyboard.press("Control+k")
    page.wait_for_timeout(300)
    snow = sorted((x for x in clips() if x["kind"] == "snow"), key=lambda x: x["start"])
    check("cut at the playhead", len(snow) == 2 and snow[0]["end"] == snow[1]["start"] == 120, str(snow))
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    check("undo", len([x for x in clips() if x["kind"] == "snow"]) == 1)
    page.keyboard.press("Control+Shift+z")
    page.wait_for_timeout(200)
    check("redo", len([x for x in clips() if x["kind"] == "snow"]) == 2)

    page.click(".clip.base >> nth=0")
    page.click("#paintFreeze")
    box = page.locator("#screen").bounding_box()
    drag(box["x"] + 40, box["y"] + box["height"] - 40, box["x"] + 160, box["y"] + box["height"] - 60)
    shot("05-paint")
    page.keyboard.press("Escape")
    page.wait_for_function("S.project.masks.includes('freeze')", timeout=30000)
    check("still area painted and saved", True)

    page.evaluate("saveNow()")
    saved = page.evaluate("(async () => (await get('/api/project?id=' + S.project.id)).layers"
                          ".map(c => [c.id, c.start, c.end, c.track]))()")
    mine = page.evaluate("S.project.layers.map(c => [c.id, c.start, c.end, c.track])")
    check("the app has the page's timeline", saved == mine, f"{saved} / {mine}")

    page.click("#exportBtn")
    page.click("#mGo")
    page.wait_for_function("S.state && S.state.jobs.some(j => j.kind === 'export' && j.state !== 'queued' "
                           "&& j.state !== 'running')", timeout=600000)
    job = page.evaluate("S.state.jobs.find(j => j.kind === 'export')")
    check("export", job["state"] == "done" and job["output"].startswith("Quán cà phê-fx-30s"), str(job))
    if job["state"] == "done":
        info = video.probe(os.path.join(os.environ["BOKIN_FX_OUT"], job["output"]))
        check("exported video: 30 s", info["frames"] == 720, str(info))
    shot("06-exported")

    # V1 replaced by another loop (30 frames a second): the clips keep their times in seconds
    before = page.evaluate("S.project.layers.map(c => [c.id, c.start / 24, c.end / 24])")
    page.click(".clip.base >> nth=0")
    page.once("dialog", lambda d: d.accept())
    page.set_input_files("#fileReplace", upload(folder, "loop30.mp4", "Cảnh mới.mp4", "video/mp4"))
    page.wait_for_function("S.project && S.project.video.fps === '30'", timeout=60000)
    after = page.evaluate("S.project.layers.map(c => [c.id, c.start / 30, c.end / 30])")
    same = len(before) == len(after) and all(a[0] == b[0] and abs(a[1] - b[1]) < 0.05 and
                                             (abs(a[2] - b[2]) < 0.05 or abs(b[2] - 30) < 0.05)
                                             for a, b in zip(before, after))
    check("replace V1: clips keep their times", same, f"{before} -> {after}")
    page.wait_for_function("S.project.proxy && S.project.proxy !== 'proxy.mp4'", timeout=120000)
    page.wait_for_timeout(800)
    page.keyboard.press("Space")
    page.wait_for_timeout(1500)
    state = page.evaluate("({t: S.time, base: baseVid.readyState, cur: baseVid.currentTime, src: baseVid.src})")
    page.keyboard.press("Space")
    if edge:
        check("replace V1: the new video plays", state["base"] >= 2 and state["cur"] > 0, str(state))
    check("replace V1: its light copy", "proxy-" in state["src"], str(state))
    shot("07-replaced")
    check("no script errors on the page", not errors, "\n".join(errors[:10]))


def main():
    os.makedirs(SHOTS, exist_ok=True)
    folder = media()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = server.start(server.App(desktop=False), port)
    try:
        with sync_playwright() as p:
            browser, edge = launch(p)
            print(f"browser: {browser.browser_type.name} {browser.version} ({'Edge' if edge else 'Chromium'})", flush=True)
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.goto(f"http://127.0.0.1:{port}/")
            started = time.time()
            try:
                run(page, folder, edge)
            except Exception as e:  # noqa: BLE001
                page.screenshot(path=os.path.join(SHOTS, "failed.png"))
                check("the walk-through ran to the end", False, f"{type(e).__name__}: {e}")
            print(f"page test took {time.time() - started:.0f} s", flush=True)
            browser.close()
    finally:
        srv.shutdown()
        shutil.rmtree(WORK, ignore_errors=True)
    print("page test " + ("FAILED: " + ", ".join(FAILED) if FAILED else "passed"), flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
