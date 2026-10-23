"""The page in a real browser (CI: Microsoft Edge on Windows, the browser the app opens).

  python packaging/ci_page_test.py [screenshot folder]

Starts the app's server on test data, then works the page as a person would: opens a loop, imports an effect video
and a picture into the library, drags the effect onto the timeline, adds snow, plays, cuts a clip, moves and trims
clips, undoes, paints a still area, zooms and moves the picture (a clip outside the frame grabbed there) and exports a
30-second video. It checks the timeline the page saved, that the live preview plays the light copies (H.264 and
transparent WebM), the falling leaves' settings (each shape a checkbox, "Loại lá" following them), rustling leaves
painted into their area (their live preview against the exact frame), the Phím tắt window (a key changed, taken from
another action, refused; the guide), saving (Ctrl+S, "Lưu thành bản khác"), the activation screen first (a wrong key
refused, the right one opens the app), and that the page reports no script errors.
Screenshots of each step go to the folder given (default: page-test).

The browser: Edge when installed (Windows), else the Chromium of Playwright, or BOKIN_BROWSER (a path).
"""
import io
import json
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

from bokin_license import ed25519, keys as license_keys, store as license_store  # noqa: E402
from fx import licensing, server, timeline, video  # noqa: E402

# The app opens on its activation screen (fx/licensing.py): this test is its own key center, a signing key made
# now and trusted in place of the seller's in this process, with an empty key folder of its own.
SECRET = os.urandom(32)
license_keys.PUBLIC_KEY = ed25519.public_key(SECRET)
license_store.default_folder = lambda: os.path.join(WORK, "license")

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
    im = Image.new("RGB", (480, 320), (30, 30, 30))  # a test pattern to turn in 3D: coloured squares, a white rim
    d = ImageDraw.Draw(im)
    colours = [(230, 80, 60), (60, 200, 90), (70, 110, 240), (240, 220, 70), (220, 90, 220), (80, 220, 230)]
    for j in range(8):
        for i in range(12):
            d.rectangle([i * 40, j * 40, i * 40 + 39, j * 40 + 39], fill=colours[(i + 2 * j) % 6])
    d.rectangle([0, 0, 479, 319], outline=(255, 255, 255), width=6)
    d.ellipse([170, 90, 310, 230], fill=(250, 250, 250))
    im.save(os.path.join(folder, "pattern.png"))
    return folder


# the media element's own corners on the screen, in video pixels: a twin of it with the same style and a mark at
# each corner (a <video> or <img> holds no marks), so the browser itself says where its transform lays them
CORNERS = """(id) => {
    const e = S.els[id], m = e.media, twin = document.createElement('div');
    for (const k of ['left', 'top', 'width', 'height', 'transform']) twin.style[k] = m.style[k];
    twin.style.position = 'absolute';
    twin.innerHTML = ['left:0;top:0', 'left:100%;top:0', 'left:100%;top:100%', 'left:0;top:100%']
      .map(s => `<i style="position:absolute;${s};width:0;height:0"></i>`).join('');
    e.wrap.appendChild(twin);
    const sc = screenEl.getBoundingClientRect(), W = info().w, H = info().h;
    const out = [...twin.children].map(i => { const r = i.getBoundingClientRect();
        return [(r.left - sc.left) / sc.width * W, (r.top - sc.top) / sc.height * H]; });
    twin.remove();
    return out;
}"""


def clip_quad(page, cid, info_w, info_h, lib_info=None):
    """Where the renderer lays a clip's corners (a file's picture, or an effect's sheet), in video pixels."""
    from fx import effects as fx_effects, overlay as fx_overlay
    c = page.evaluate(f"(c => ({{kind: c.kind, params: c.params}}))(clipById('{cid}'))")
    if c["kind"] == "file":
        return fx_overlay.quad(fx_effects.clean_params("file", c["params"]), lib_info, info_w, info_h)
    return fx_effects.quad(c["kind"], c["params"], info_w, info_h)


def picture_parity(page, cid, info_w, info_h, lib_info, live, exact):
    """How far the live preview's picture of a clip is from the exact frame's, inside its quad (2 px in from the
    outline): mean difference of the colours, and of the brightness softened by 1.5 px (the exact frame is a
    JPEG, its colour at half size: hard colour edges differ a lot without saying anything about the geometry)."""
    from PIL import ImageFilter
    from fx import overlay as fx_overlay
    live = np.asarray(Image.open(io.BytesIO(live)).convert("RGB"), np.float32)
    exact = np.asarray(Image.open(io.BytesIO(exact)).convert("RGB"), np.float32)
    sh, sw = live.shape[:2]
    q = [(x * sw / info_w, y * sh / info_h) for x, y in clip_quad(page, cid, info_w, info_h, lib_info)]
    core = Image.fromarray(np.where(np.asarray(fx_overlay.coverage(q, (0, 0, sw, sh))) == 255, 255, 0).astype(np.uint8))
    inner = np.asarray(core.filter(ImageFilter.MinFilter(5))) > 0

    def luma(a):
        return np.asarray(Image.fromarray(np.clip(a @ np.array(fx_overlay.LUMA), 0, 255).astype(np.uint8))
                          .filter(ImageFilter.GaussianBlur(1.5)), np.float32)
    rgb = np.abs(live - exact).mean(axis=2)[inner]
    soft = np.abs(luma(live) - luma(exact))[inner]
    return {"rgb mean": round(float(rgb.mean()), 2), "rgb p99": round(float(np.percentile(rgb, 99)), 1),
            "rgb max": round(float(rgb.max()), 1), "soft luma mean": round(float(soft.mean()), 2),
            "soft luma max": round(float(soft.max()), 1), "pixels": int(inner.sum())}


BOX = ("(() => { const r = xform.getBoundingClientRect(), sc = screenEl.getBoundingClientRect(), "
       "kx = info().w / sc.width, ky = info().h / sc.height; return [(r.left - sc.left) * kx, "
       "(r.top - sc.top) * ky, (r.right - sc.left) * kx, (r.bottom - sc.top) * ky]; })()")  # the white box, video px


def live_and_exact(page, name):
    """Screenshots of the picture: the live preview alone, then the exact frame for the project as saved now
    (kept as <name>-live.png, <name>-exact.png)."""
    page.evaluate("(async () => { S.sel = null; renderControls(); showXform(); await saveNow(); })()")
    page.wait_for_timeout(500)
    page.evaluate("exact.style.visibility = 'hidden'")  # the live preview alone
    page.wait_for_timeout(400)
    live = page.locator("#screen").screenshot()
    page.evaluate("(() => { if (!S.exactOn) $('#tExact').click(); invalidateExact(); refreshExact(); })()")
    page.wait_for_function("exact.style.display === 'block' && exact.complete && exact.src.includes('rev=' + S.project.rev)",
                           timeout=60000)
    page.evaluate("exact.style.visibility = ''")
    page.wait_for_timeout(300)
    exact = page.locator("#screen").screenshot()
    for tag, png in (("live", live), ("exact", exact)):
        with open(os.path.join(SHOTS, f"{name}-{tag}.png"), "wb") as f:
            f.write(png)
    return live, exact


def as_close(upright, turned):
    """The live picture of a turned effect is about as close to the exact frame as upright (soft luma / rgb means,
    measured in Chromium: snow 0.43 / 0.93 upright, 0.47 / 1.12 turned; leaves 0.60 / 1.85 upright, 0.47 / 1.80
    turned. One leaf more on the sheet than in the preview clip moved all the smaller ones: 2.2 / 4.9, not close)."""
    return (turned["soft luma mean"] < max(1.5, 2 * upright["soft luma mean"] + 0.5) and
            turned["rgb mean"] < max(4.0, 2 * upright["rgb mean"] + 1.0))


def effects_turned(page, drag):
    """Built-in effects placed and turned in 3D as a file clip is: snow on a sheet of the frame's shape, with the
    white box around the frame, the round knob and "3D"; the live preview lays the same preview clip (its name does
    not change: turning is instant) on the renderer's corners, and its picture is the exact frame's (snow, leaves);
    light on black (H.264, Screen) and a transparent WebM turned in a Mix; "Đặt lại xoay", the tools column, undo;
    a corner grabbed outside the frame."""
    vw, vh = page.evaluate("[info().w, info().h]")
    sid = page.evaluate("S.project.layers.find(l => l.kind === 'snow').id")
    others = page.evaluate(f"S.project.layers.filter(c => c.id !== '{sid}' && c.on !== false).map(c => c.id)")
    kept = page.evaluate(f"S.project.layers.filter(c => c.id !== '{sid}').map(c => [c.id, JSON.stringify(c.params)])")

    page.evaluate(f"(() => {{ for (const id of {json.dumps(others)}) clipById(id).on = false; edited(); seek(2); "
                  f"select('{sid}'); }})()")  # the snow alone, at 2 s
    page.wait_for_function(f"S.fx['{sid}'] && S.fx['{sid}'].ready && S.els['{sid}'] && "
                           f"S.els['{sid}'].media.readyState >= 2", timeout=180000)
    page.wait_for_timeout(500)
    ui = page.evaluate("[$('#xform').style.display, $('#xformRot').style.display, $('#xform3d').style.display, "
                       "!!$('#ec input[data-num=\"rot_x\"]'), !!$('#rotReset'), "
                       "!!$('#ec input[type=range][data-param=\"scale\"]'), !!$('#tools [data-rotreset]')]")
    box = page.evaluate(BOX)
    ui.append("ô cửa nhìn chéo" in page.evaluate("$('#ec').textContent"))  # (the window example in its turn note)
    check("3D effects: snow has the white box around the frame, the round knob, \"3D\" and its place and turn "
          "settings", ui == ["block"] * 3 + [True] * 5 and all(abs(a - b) < 1.0 for a, b in zip(box, [0, 0, vw, vh])),
          str([ui, box]))
    name0 = page.evaluate(f"S.fx['{sid}'].shown")
    live, exact = live_and_exact(page, "05e-snow-upright")
    upright = picture_parity(page, sid, vw, vh, None, live, exact)

    def turn(cid, params):
        page.evaluate(f"(() => {{ const c = clipById('{cid}'); Object.assign(c.params, {json.dumps(params)}); "
                      "edited(); })()")

    worst = [0.0, 0.0, 0.0]
    for case in (dict(rot_x=-50, rot_y=35, rot_z=-120, persp=100, scale=120, x=-10, y=6),
                 dict(rot_x=15, rot_y=60, rot_z=0, persp=0, scale=45, x=20, y=-15),
                 dict(rot_x=25, rot_y=-30, rot_z=10, persp=60, scale=70, x=8, y=-5)):
        turn(sid, case)
        page.evaluate(f"select('{sid}')")
        want = clip_quad(page, sid, vw, vh)
        shown, js = page.evaluate(CORNERS, sid), page.evaluate(f"quad(clipById('{sid}'))")
        box = page.evaluate(BOX)
        around = [min(x for x, _ in want), min(y for _, y in want), max(x for x, _ in want), max(y for _, y in want)]
        worst = [max(worst[0], max(math.dist(a, b) for a, b in zip(shown, want))),
                 max(worst[1], max(math.dist(a, b) for a, b in zip(js, want))),
                 max(worst[2], max(abs(a - b) for a, b in zip(box, around)))]
    check("3D effects: the turned snow's preview clip lies on the renderer's corners (within half a pixel), its "
          "box with it", worst[0] < 0.5 and worst[1] < 1e-6 and worst[2] < 1.0,
          f"browser vs renderer {worst[0]:.3f} px, page's quad() vs renderer {worst[1]:.1e} px, box {worst[2]:.3f} px")
    page.evaluate("saveNow()")
    asked = page.evaluate(f"""(async () => {{ const r = await post('/api/fxclip', {{id: S.project.id, layer: '{sid}'}});
        return [r.name, r.ready, S.fx['{sid}'].shown]; }})()""")
    check("3D effects: turning keeps the preview clip (the same file, nothing made again)",
          asked == [name0, True, name0], str([name0, asked]))
    live, exact = live_and_exact(page, "05e-snow-turned")
    turned = picture_parity(page, sid, vw, vh, None, live, exact)
    check("3D effects: the live picture of the turned snow is the exact frame's (as close as upright)",
          as_close(upright, turned), str({"upright": upright, "turned": turned}))

    # the tools column, "Đặt lại xoay" and undo; the knob and "3D" turn it
    page.evaluate(f"select('{sid}')")
    page.wait_for_timeout(200)
    tools = page.evaluate("[$('#tools [data-rotreset]').disabled, [...$$('#tools .lab')].map(e => e.textContent).join(' | ')]")
    check("3D effects: the tools column shows the turn", tools[0] is False and "25°, -30°, 10°" in tools[1], str(tools))
    page.click("#rotReset")
    page.wait_for_timeout(200)
    flat = page.evaluate(f"(c => [c.params.rot_x, c.params.rot_y, c.params.rot_z, c.params.scale, "
                         f"S.els[c.id].media.style.transform])(clipById('{sid}'))")
    check("3D effects: \"Đặt lại xoay\" sets the snow straight (its place and size stay)", flat == [0, 0, 0, 70, ""],
          str(flat))
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    check("3D effects: undo turns it again", page.evaluate(f"turned(clipById('{sid}'))") is True)
    hb = page.locator("#xform3d").bounding_box()
    hx, hy = hb["x"] + hb["width"] / 2, hb["y"] + hb["height"] / 2
    drag(hx, hy, hx + 50, hy)
    tilt = page.evaluate(f"(p => [p.rot_x, p.rot_y, p.rot_z])(clipById('{sid}').params)")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    kb = page.locator("#xformRot").bounding_box()
    mid = page.evaluate(f"""(() => {{ const q = quad(clipById('{sid}')), sc = screenEl.getBoundingClientRect();
        const r = placed(clipById('{sid}')), c = [r.x + r.w / 2, r.y + r.h / 2];
        return [sc.left + c[0] / info().w * sc.width, sc.top + c[1] / info().h * sc.height]; }})()""")
    kx, ky = kb["x"] + kb["width"] / 2, kb["y"] + kb["height"] / 2
    page.mouse.move(kx, ky)
    page.mouse.down()
    for k in range(1, 7):  # round the middle, 30 degrees clockwise
        a = math.radians(30) * k / 6
        page.mouse.move(mid[0] + (kx - mid[0]) * math.cos(a) - (ky - mid[1]) * math.sin(a),
                        mid[1] + (kx - mid[0]) * math.sin(a) + (ky - mid[1]) * math.cos(a))
    page.mouse.up()
    page.wait_for_timeout(200)
    spun = page.evaluate(f"clipById('{sid}').params.rot_z")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    check("3D effects: \"3D\" tilts the snow, the round knob turns it in the plane",
          abs(tilt[1] - (-30 + 15)) < 1 and tilt[0] == 25 and abs(spun - 40) < 1.5 and
          page.evaluate(f"clipById('{sid}').params.rot_z") == 10, str([tilt, spun]))

    # the picture zoomed out: the snow's box is the frame, a corner grabbed on the area around it sizes it there
    turn(sid, dict(rot_x=0, rot_y=0, rot_z=0, scale=100, x=0, y=0))
    page.select_option("#zoomSel", "0.5")
    page.evaluate(f"select('{sid}')")
    page.wait_for_timeout(300)
    hb = page.locator("#xform i[data-h=se]").bounding_box()
    hx, hy = hb["x"] + hb["width"] / 2, hb["y"] + hb["height"] / 2
    drag(hx, hy, hx + 60, hy + 34)
    grown = page.evaluate(f"(r => [clipById('{sid}').params.scale, r.x, r.y])(placed(clipById('{sid}')))")
    check("3D effects: zoomed out, a corner of the snow's box grabbed outside the frame sizes it (the opposite corner "
          "stays)", grown[0] > 110 and abs(grown[1]) < 1.5 and abs(grown[2]) < 1.5, str(grown))
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    page.keyboard.press("Control+0")

    # the snow over the whole frame, selected: a drag in its box moves it, whatever lies under it (a small picture,
    # a file over the whole frame) or over it (light over the whole frame); a click (no drag) on the small picture
    # under it picks that picture, a click where only clips over the whole frame are keeps the snow
    fid, gid = page.evaluate(f"""(() => {{ const ti = S.project.tracks.findIndex(t => t.id === clipById('{sid}').track);
        const below = {{id: newId("t"), on: true}}; S.project.tracks.splice(ti, 0, below);
        addClip({{lib: S.lib.find(i => i.info.image).id}}, below.id, 0, total()); const f = clipById(S.sel);
        Object.assign(f.params, {{blend: 'normal', fit: 'contain', scale: 30}});
        const above = {{id: newId("t"), on: true}}; S.project.tracks.push(above);
        addClip({{kind: 'glow'}}, above.id, 0, total()); const g = clipById(S.sel);
        edited(); seek(2); select('{sid}'); return [f.id, g.id]; }})()""")
    page.wait_for_timeout(300)
    sc = page.evaluate("(r => [r.left + r.width / 2, r.top + r.height / 2])(screenEl.getBoundingClientRect())")
    xs = f"[S.sel, clipById('{sid}').params.x, clipById('{fid}').params.x, clipById('{gid}').params.x]"
    under = page.evaluate(f"[pickAt({{clientX: {sc[0]}, clientY: {sc[1]}}}) && pickAt({{clientX: {sc[0]}, "
                          f"clientY: {sc[1]}}}).id, coversFrame(clipById('{sid}')), coversFrame(clipById('{gid}'))]")
    drag(sc[0], sc[1], sc[0] + 120, sc[1] + 40)
    dragged = page.evaluate(xs)
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    page.mouse.click(sc[0], sc[1])
    page.wait_for_timeout(300)
    clicked = page.evaluate(xs)
    page.evaluate(f"(() => {{ Object.assign(clipById('{fid}').params, {{fit: 'cover', scale: 100}}); edited(); "
                  f"select('{sid}'); }})()")  # the picture now over the whole frame
    page.wait_for_timeout(200)
    drag(sc[0], sc[1], sc[0] + 120, sc[1] + 40)
    dragged2 = page.evaluate(xs)
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    page.mouse.click(sc[0], sc[1])
    page.wait_for_timeout(300)
    clicked2 = page.evaluate(xs)
    check("3D effects: dragging the box of the selected snow over the whole frame moves the snow (a small picture "
          "under it, a file over the whole frame under it, light over the whole frame over it); a click picks the "
          "small picture, not a clip over the whole frame",
          under == [fid, True, True] and dragged[0] == sid and dragged[1] > 5 and dragged[2:] == [0, 0] and
          clicked == [fid, 0, 0, 0] and dragged2[0] == sid and dragged2[1] > 5 and dragged2[2:] == [0, 0] and
          clicked2 == [sid, 0, 0, 0], str([under, dragged, clicked, dragged2, clicked2]))
    page.evaluate(f"""(() => {{ for (const id of ['{fid}', '{gid}']) {{ const t = clipById(id).track;
        deleteClip(clipById(id)); S.project.tracks = S.project.tracks.filter(x => x.id !== t); }}
        edited(); select('{sid}'); }})()""")
    page.wait_for_timeout(200)
    turn(sid, dict(rot_x=25, rot_y=-30, rot_z=10, persp=60, scale=70, x=8, y=-5))

    # leaves turned: the live picture is the exact frame's
    lv = page.evaluate(f"""(() => {{ const t = {{id: newId("t"), on: true}}; S.project.tracks.push(t);
        addClip({{kind: 'leaves'}}, t.id, 0, total()); const c = clipById(S.sel); clipById('{sid}').on = false;
        Object.assign(c.params, {{amount: 80}}); edited(); seek(2); return c.id; }})()""")
    page.wait_for_function(f"S.fx['{lv}'] && S.fx['{lv}'].ready && S.els['{lv}'] && "
                           f"S.els['{lv}'].media.readyState >= 2", timeout=300000)
    page.wait_for_timeout(500)
    live, exact = live_and_exact(page, "05f-leaves-upright")
    upright = picture_parity(page, lv, vw, vh, None, live, exact)
    turn(lv, dict(rot_x=20, rot_y=35, rot_z=-8, persp=70, scale=80, x=-6, y=4))
    live, exact = live_and_exact(page, "05f-leaves-turned")
    turned = picture_parity(page, lv, vw, vh, None, live, exact)
    check("3D effects: the live picture of turned leaves is the exact frame's (as close as upright)",
          as_close(upright, turned), str({"upright": upright, "turned": turned}))

    # the leaves' settings: "Loại lá" and "Kiểu lá rơi" under "Kiểu", every shape a checkbox under its own heading
    # (the two maples ticked); a "Loại lá" ticks its shapes and colours them, a shape ticked by hand turns it to "Tự
    # chọn", and back when the ticks are its own again
    page.evaluate(f"select('{lv}')")
    page.wait_for_timeout(200)
    shapes_head = "Kiểu dáng lá (tích nhiều ô để trộn)"
    heads = page.evaluate("[...document.querySelectorAll('#ec h4')].map(h => h.textContent)")
    boxes = page.evaluate("[...document.querySelectorAll('#ec input[type=checkbox][data-param]')]"
                          ".map(i => [i.dataset.param, i.checked])")
    state = f"""(() => {{ const c = clipById('{lv}'); return [c.params.scene,
        document.querySelector('#ec select[data-param=scene]').value, c.params.season,
        [...document.querySelectorAll('#ec input[type=checkbox][data-param]')].filter(i => i.checked)
          .map(i => i.dataset.param).join(' ')]; }})()"""
    page.select_option("#ec select[data-param=scene]", "ginkgo")
    page.wait_for_timeout(300)
    ginkgo = page.evaluate(state)
    box = page.evaluate("""(() => { const s = document.querySelector('#ec select[data-param=scene]');
        let p = s.parentElement;  // (the panel's own scroll box: the page itself stays put)
        while (p && !(p.scrollHeight > p.clientHeight && /(auto|scroll)/.test(getComputedStyle(p).overflowY)))
          p = p.parentElement;
        if (!p) return null;
        p.scrollTop += s.getBoundingClientRect().top - p.getBoundingClientRect().top - 60;
        const r = p.getBoundingClientRect(); return {x: r.left, y: r.top, width: r.width, height: r.height}; })()""")
    if box:
        page.screenshot(path=os.path.join(SHOTS, "05g-leaves-settings.png"), clip=box)
    page.click("#ec input[type=checkbox][data-param=oak]")
    page.wait_for_timeout(200)
    custom = page.evaluate(state)
    page.click("#ec input[type=checkbox][data-param=oak]")
    page.wait_for_timeout(200)
    again = page.evaluate(state)
    shapes = ["maple", "japanese", "ginkgo", "oak", "almond", "birch", "willow", "heart", "petal"]
    check("leaves: kinds of leaves at once and every shape a checkbox, under clear headings",
          "Kiểu" in heads and heads[heads.index("Kiểu"):][:4] == ["Kiểu", shapes_head, "Màu lá (cánh hoa luôn màu hồng)",
                                                                  "Ánh sáng"]
          and boxes == [[s, s in ("maple", "japanese")] for s in shapes]
          and ginkgo == ["ginkgo", "ginkgo", "gold", "ginkgo"] and custom == ["custom", "custom", "gold", "ginkgo oak"]
          and again == ["ginkgo", "ginkgo", "gold", "ginkgo"], str([heads, boxes, ginkgo, custom, again]))

    # light on black (H.264, laid with Screen) and a transparent WebM, turned, in a Mix: in their track's box, on
    # the renderer's corners
    for kind, blend in (("glow", "screen"), ("rain", "normal")):
        a, b = page.evaluate(f"""(() => {{ const t = {{id: newId("t"), on: true}}; S.project.tracks.push(t);
            addClip({{kind: '{kind}'}}, t.id, 0, 240); const a = clipById(S.sel);
            addClip({{kind: '{kind}'}}, t.id, 240, 480); const b = clipById(S.sel); b.seed = a.seed;
            Object.assign(a.params, {{rot_x: 20, rot_y: -30, rot_z: 6, scale: 75, x: -8}});
            Object.assign(b.params, {{rot_x: -15, rot_y: 25, rot_z: -10, scale: 85, x: 6}});
            edited(); setMix(b, 1); seek(10); return [a.id, b.id]; }})()""")
        page.wait_for_function(f"S.els['{a}'] && S.els['{b}']", timeout=300000)
        page.evaluate("applyTime()")
        page.wait_for_timeout(300)
        roles = page.evaluate(f"(() => {{ const b = clipById('{b}'), box = S.trackEls[b.track]; return [S.mixRole['{a}'], "
                              f"S.mixRole['{b}'], box.style.isolation, box.style.mixBlendMode, "
                              f"S.els['{a}'].media.parentNode.parentNode === box, "
                              f"S.els['{a}'].wrap.style.display, S.els['{b}'].wrap.style.display]; }})()")
        off = max(math.dist(p, q) for cid in (a, b)
                  for p, q in zip(page.evaluate(CORNERS, cid), clip_quad(page, cid, vw, vh)))
        check(f"3D effects: {kind} ({'H.264 on black, Screen' if kind == 'glow' else 'WebM'}) turned in a Mix: in its "
              "track's box, on the renderer's corners", roles == ["normal", "plus-lighter", "isolate", blend, True, "", ""]
              and off < 0.5, f"{roles}, {off:.3f} px")
        page.evaluate(f"(() => {{ const t = clipById('{a}').track; for (const id of ['{a}', '{b}']) deleteClip(clipById(id)); "
                      "S.project.tracks = S.project.tracks.filter(x => x.id !== t); edited(); })()")
    page.evaluate(f"""(() => {{ const t = clipById('{lv}').track; deleteClip(clipById('{lv}'));
        S.project.tracks = S.project.tracks.filter(x => x.id !== t);
        for (const id of {json.dumps(others + [sid])}) clipById(id).on = true; edited(); seek(3); }})()""")
    page.wait_for_timeout(300)
    now = dict(page.evaluate(f"S.project.layers.filter(c => c.id !== '{sid}').map(c => [c.id, JSON.stringify(c.params)])"))
    check("3D effects: every other clip is as it was", all(now.get(i) == p for i, p in kept),
          str([(i, p, now.get(i)) for i, p in kept if now.get(i) != p]))


def foliage(page, shot):
    """Rustling leaves: they move only inside the clip's painted area (none painted: the panel says so, nothing
    moves); its preview clip is the picture under that area with the leaves moving, made again when the area is
    painted, laid just over the video (under every track); once made, the live preview is what the export shows: the
    change it makes matches the exact frame's (the other clips switched off, the leaves' clip shown and hidden)."""
    page.click('.tab[data-bin="fx"]')
    page.evaluate("seek(2)")  # (the disc of the test loop at its right-most place, inside the area painted below)
    page.dblclick('.tile[data-kind="foliage"]')
    fid = page.evaluate("S.project.layers.find(l => l.kind === 'foliage').id")
    page.click("#fullBtn")
    page.select_option('#ec select[data-param="look"]', "strong")
    page.wait_for_timeout(300)
    ui = page.evaluate("""[$('#ec').textContent.includes('chưa tô vùng'), !!$('#ec .warn'), !!$('#clearArea'),
        $('#xform').style.display, !!$('#ec [data-num="rot_x"]'), !!$('#ec [data-param="x"]'),
        clipById(S.sel).params.breeze > 0]""")
    check("rustling leaves: no area yet: the panel says nothing moves until it is painted; no box, no turn",
          ui == [True, True, False, "none", False, False, True], str(ui))
    page.wait_for_function(f"S.fx['{fid}'] && S.fx['{fid}'].ready && S.fx['{fid}'].shown", timeout=300000)
    empty = page.evaluate(f"S.fx['{fid}'].shown")
    page.click("#paintArea")  # an outline round where the disc is at 2 s (in a real video: a bush)
    page.keyboard.press("l")
    box = page.locator("#screen").bounding_box()
    vw, vh = page.evaluate("[info().w, info().h]")
    outline = [(0.68, 0.3), (0.94, 0.31), (0.95, 0.72), (0.67, 0.7)]
    for fx_, fy_ in outline:
        page.mouse.click(box["x"] + box["width"] * fx_, box["y"] + box["height"] * fy_)
        page.wait_for_timeout(150)
    page.click("[data-done]")  # Xong: the outline is closed and kept
    page.evaluate("S.brush.tool = 'brush'")  # (the brush again, as the later steps expect)
    page.wait_for_function(f"S.project.masks.includes('{fid}') && S.fx['{fid}'].want !== {json.dumps(empty)}",
                           timeout=30000)
    painted = page.evaluate("[$('#ec').textContent.includes('vùng đã tô'), !!$('#ec .warn')]")
    page.wait_for_function(f"S.fx['{fid}'].ready && S.fx['{fid}'].shown === S.fx['{fid}'].want && S.els['{fid}'] && "
                           f"S.els['{fid}'].media.readyState >= 2", timeout=300000)
    laid = page.evaluate(f"""(() => {{ const c = clipById('{fid}'), e = S.els['{fid}'];
        return [e.wrap.parentNode === $('#layers').firstChild, e.media.getAttribute('src').includes('.webm'),
                e.wrap.style.mixBlendMode, areaLive(c), !!(S.schema.lens[c.kind] && !areaLive(c)),
                $('#ec').textContent.includes('gần đúng')]; }})()""")
    check("rustling leaves: painting its area makes its preview clip, laid just over the video (under every "
          "track); then the live preview counts as exact", painted == [True, False] and
          laid == [True, True, "normal", True, False, False], str([painted, laid]))
    # the change the leaves make, live and exact (every other clip off; the leaves' clip on, then off)
    others = page.evaluate(f"S.project.layers.filter(c => c.id !== '{fid}' && c.on !== false).map(c => c.id)")

    def switch(on):
        page.evaluate(f"""(() => {{ for (const c of S.project.layers) c.on = {json.dumps(on)}.includes(c.id);
            edited(); }})()""")
        page.wait_for_timeout(600)
    switch([fid])
    page.wait_for_function(f"S.els['{fid}'] && S.els['{fid}'].media.readyState >= 2", timeout=60000)
    on = live_and_exact(page, "05h-foliage-on")
    switch([])
    off = live_and_exact(page, "05h-foliage-off")
    switch(others + [fid])
    shots = [np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), np.float32) @ np.array([0.2126, 0.7152, 0.0722])
             for png in (*on, *off)]
    live_on, exact_on, live_off, exact_off = shots
    sh, sw = live_on.shape
    area = Image.new("L", (sw, sh), 0)
    ImageDraw.Draw(area).polygon([(x * sw, y * sh) for x, y in outline], fill=255)
    from PIL import ImageFilter
    inner = np.asarray(area.filter(ImageFilter.MinFilter(9))) > 0
    live, exact = (live_on - live_off)[inner], (exact_on - exact_off)[inner]
    corr = float(np.corrcoef(live, exact)[0, 1])
    amp = float(live.std() / max(exact.std(), 1e-6))
    near, far = (np.asarray(area.filter(ImageFilter.MaxFilter(k))) > 0 for k in (9, 61))
    outside = float(np.abs(live_on - live_off)[far & ~near].max())  # (round the area: nothing changes there)
    check("rustling leaves: the live preview moves the picture as the exact frame does (only inside the area)",
          exact.std() > 1.0 and corr > 0.7 and 0.6 < amp < 1.6 and outside < 3,
          f"corr {corr:.3f}, live/exact {amp:.2f}, exact change {exact.std():.2f}, outside {outside:.1f}")
    shot("05h-foliage")


def winter(page, shot, edge):
    """Frost on the pane and string lights on a tree: frost over the whole picture (a transparent WebM, Normal); the
    lights placed in a box (the white box on the picture, where the renderer puts it), light on black (H.264, Screen)
    laid in that box; each alone, the live picture is the exact frame's (the lights' in Edge: H.264)."""
    from fx import effects as fx_effects
    page.click('.tab[data-bin="fx"]')
    page.evaluate("seek(3)")
    ids = {}
    for kind in ("frost", "lights"):
        page.dblclick(f'.tile[data-kind="{kind}"]')
        ids[kind] = page.evaluate(f"S.project.layers.find(l => l.kind === '{kind}').id")
        page.click("#fullBtn")
    fid, lid = ids["frost"], ids["lights"]
    page.evaluate(f"select('{lid}')")
    page.wait_for_timeout(300)
    vw, vh = page.evaluate("[info().w, info().h]")
    ui = page.evaluate("""[$('#xform').style.display, !!$('#ec select[data-param="layout"]'),
        !!$('#ec select[data-param="mode"]'), !!$('#ec [data-num="rot_x"]')]""")
    box = page.evaluate(BOX)
    x0, y0, w, h = fx_effects.box_of("lights", page.evaluate(f"clipById('{lid}').params"), vw, vh)
    check("string lights: a box on the picture where the renderer puts them, their layout, twinkle and turn settings",
          ui == ["block", True, True, True] and all(abs(a - b) < 1.5 for a, b in zip(box, [x0, y0, x0 + w, y0 + h])),
          str([ui, box, [x0, y0, x0 + w, y0 + h]]))
    page.select_option('#ec select[data-param="palette"]', "multi")
    page.wait_for_timeout(300)
    others = page.evaluate(f"S.project.layers.filter(c => !['{fid}', '{lid}'].includes(c.id) && c.on !== false)"
                           ".map(c => c.id)")

    def switch(on):
        page.evaluate(f"""(() => {{ for (const c of S.project.layers) c.on = {json.dumps(on)}.includes(c.id);
            edited(); }})()""")
        page.wait_for_timeout(600)
    # (fine ice and tiny bulbs: a little apart in a compressed clip and in the JPEG of the exact frame, more than snow;
    # the bounds of the 3D check of a file clip)
    for kind, cid, clip, blend in (("frost", fid, ".webm", "normal"), ("lights", lid, ".mp4", "screen")):
        switch([cid])
        plays = edge or clip == ".webm"
        page.wait_for_function(f"S.fx['{cid}'] && S.fx['{cid}'].ready && S.fx['{cid}'].shown === S.fx['{cid}'].want "
                               f"&& S.els['{cid}']" + (f" && S.els['{cid}'].media.readyState >= 2" if plays else ""),
                               timeout=300000)
        page.evaluate("applyTime()")
        laid = page.evaluate(f"[S.els['{cid}'].media.getAttribute('src').includes('{clip}'), "
                             f"S.els['{cid}'].wrap.style.mixBlendMode || 'normal']")
        if not plays:
            check(f"{kind}: its preview clip ({clip}, {blend})", laid == [True, blend], str(laid))
            continue
        live, exact = live_and_exact(page, f"05i-{kind}")
        parity = picture_parity(page, cid, vw, vh, None, live, exact)
        check(f"{kind}: its preview clip ({clip}, {blend}), the live picture as the exact frame",
              laid == [True, blend] and parity["soft luma mean"] < 4 and parity["rgb mean"] < 7,
              str([laid, parity]))
    switch(others + [fid, lid])
    layouts = page.evaluate(f"""(() => {{ select('{lid}');
        return [...$('#ec select[data-param="layout"]').options].map(o => o.value); }})()""")
    check("string lights: round a tree, swags, a curtain or scattered (drawn by hand: Dây đèn tự vẽ)",
          layouts == ["tree", "swag", "curtain", "scatter"], str(layouts))
    # a painted area flickering (as the rustling leaves: nothing until its area is painted, no box, no turn)
    page.dblclick('.tile[data-kind="flicker"]')
    page.wait_for_timeout(400)
    warn = page.evaluate("""[$('#ec .warn') && $('#ec .warn').textContent, !!$('#ec select[data-param="mode"]'),
        $('#xform').style.display, !!$('#ec [data-num="rot_x"]')]""")
    check("Nhấp nháy vùng: the panel says to paint its area first; its ways of flickering; no box, no turn",
          bool(warn[0]) and "nhấp nháy" in warn[0] and warn[1:] == [True, "none", False], str(warn))
    # its ways: a flame's, a lamp's or a sign's; a sign lit bit by bit has a way to be lit (shown only for it), and a
    # sign's way sets its own colour and a deep dark
    ways = page.evaluate("""[[...$('#ec select[data-param="mode"]').options].map(o => o.value),
        !!$('#ec select[data-param="way"]')]""")
    page.select_option('#ec select[data-param="mode"]', "sweep")
    page.wait_for_timeout(200)
    sweep = page.evaluate("""(c => [!!$('#ec select[data-param="way"]'), c.params.warm, c.params.amount,
        [...$('#ec select[data-param="way"]').options].map(o => o.value)])(clipById(S.sel))""")
    page.select_option('#ec select[data-param="mode"]', "candle")
    page.wait_for_timeout(200)
    candle = page.evaluate("""(c => [!!$('#ec select[data-param="way"]'), c.params.warm])(clipById(S.sel))""")
    check("Nhấp nháy vùng: a flame's ways and a sign's; Sáng lần lượt shows which way it is lit, a sign's way its own "
          "colour and a deep dark", ways == [["candle", "fire", "breathe", "pulse", "faulty", "sweep"], False]
          and sweep == [True, 0, 80, ["right", "left", "down", "up"]] and candle == [False, 50],
          str([ways, sweep, candle]))
    page.evaluate("deleteClip(clipById(S.sel)); edited()")
    shot("05i-winter")


def garland(page, shot):
    """Dây đèn tự vẽ: a new clip starts drawing at once (no box, nothing to place or turn); clicks put a strand's points
    (the wire hangs between them) and a double-click on the last one ends it; a press and drag draws a stretch by hand
    (in one go: a strand of its own). The points are kept as places in the frame, and the renderer lays the bulbs where
    the page shows them, as many. A strand chosen by a click on its wire: its own sag; a point dragged; a wire pulled
    into a new point; the strand chosen deleted with its key; Backspace, Ctrl+Z and Ctrl+Y on the strand being drawn;
    Esc ends it, then the drawing; a click on the picture draws again; Xoá hết dây undone. Its preview clip: light on
    black over the whole frame."""
    from fx import effects as fx_effects
    page.click('.tab[data-bin="fx"]')
    page.dblclick('.tile[data-kind="garland"]')
    cid = page.evaluate("S.sel")
    page.wait_for_timeout(300)
    vw, vh = page.evaluate("[info().w, info().h]")
    state = page.evaluate("""(c => [c.kind, !!S.wire && S.wire.id === c.id, $('#xform').style.display, !!$('#wireDraw'),
        !!$('#ec [data-param="gap"]'), !!$('#ec [data-param="scale"]'), !!$('#ec [data-num="rot_x"]'),
        stage.classList.contains('wiring')])(clipById(S.sel))""")
    check("Dây đèn tự vẽ: drawing starts at once; no box, nothing to place or turn",
          state == ["garland", True, "none", True, True, False, False, True], str(state))
    box = page.locator("#screen").bounding_box()

    def at(fx_, fy_):
        return box["x"] + box["width"] * fx_, box["y"] + box["height"] * fy_

    def strands():
        return page.evaluate(f"clipById('{cid}').params.strands")

    first = [(0.15, 0.3), (0.45, 0.3), (0.75, 0.32)]
    for fx_, fy_ in first:
        page.mouse.click(*at(fx_, fy_))
        page.wait_for_timeout(120)
    page.mouse.dblclick(*at(*first[-1]))  # (on its last point: it ends the strand, no point more)
    page.wait_for_timeout(300)
    page.mouse.move(*at(0.2, 0.7))  # drawn by hand in one go
    page.mouse.down()
    for i in range(1, 21):
        page.mouse.move(*at(0.2 + 0.03 * i, 0.7 - 0.12 * math.sin(math.pi * i / 20)), steps=2)
    page.mouse.up()
    page.wait_for_timeout(300)
    got = strands()
    clicked, hand = got[:3], got[3:]
    ok_click = ([q[2] for q in clicked] == [1, 0, 0] and abs(clicked[0][3] - 0.35) < 1e-6 and
                all(abs(a - b) < 0.004 for q, w in zip(clicked, first) for a, b in zip(q[:2], w)))
    ok_hand = (len(hand) >= 4 and hand[0][2] == 1 and all(q[2] == 0.5 for q in hand[1:]) and
               abs(hand[0][0] - 0.2) < 0.01 and abs(hand[-1][0] - 0.8) < 0.01)
    params = page.evaluate(f"clipById('{cid}').params")
    fx = fx_effects.make("garland", dict(params, jitter=0), 3, 48, 24, vw, vh)
    shown = page.evaluate(f"wireGuide(clipById('{cid}')).bulbs")
    off = max((math.hypot(x - fx.x[i] * vh, y - fx.y[i] * vh) for i, (x, y, _) in enumerate(shown)), default=99) \
        if len(shown) == len(fx.x) else 99
    info = page.evaluate("$('#wireInfo').textContent")
    check("Dây đèn tự vẽ: clicks and a double-click make a strand, a drag one drawn by hand; the renderer's bulbs are "
          "where the page shows them, as many", ok_click and ok_hand and off < 0.6 and f"{len(shown)} bóng" in info,
          f"{got}, {len(shown)} / {len(fx.x)} bulbs, {off:.3f} px off, {info!r}")
    page.mouse.click(*at(0.4, 0.55))  # (a strand begun: drawn over the picture with the others)
    page.mouse.move(*at(0.6, 0.6))
    page.wait_for_timeout(300)
    shot("05j-garland-drawing")
    page.keyboard.press("Escape")  # (a single point: no strand)
    # the first strand chosen by a click on its wire (hanging between its points), then its own sag
    droop = 0.35 * 0.3 * vw / vh * 0.25
    page.mouse.click(*at(0.3, 0.3 + droop))
    page.wait_for_timeout(200)
    chosen = page.evaluate("[S.wire.sel, $('#ec').textContent.includes('dây đang chọn'), !!$('#wireDel') && !$('#wireDel').disabled]")
    page.evaluate("""(() => { const r = $('#wireSag'); r.value = 0; r.dispatchEvent(new Event('input', {bubbles: true}));
        r.dispatchEvent(new Event('change', {bubbles: true})); })()""")
    page.wait_for_timeout(200)
    sags = [q[3] for q in strands() if q[2] == 1]
    check("Dây đèn tự vẽ: a click on a wire chooses its strand; the sag slider is that strand's",
          chosen == [0, True, True] and sags[0] == 0 and abs(sags[1] - 0.35) < 1e-6, str([chosen, sags]))
    # a point dragged; the (now straight) wire pulled: a new point there
    page.mouse.move(*at(0.45, 0.3))
    page.mouse.down()
    page.mouse.move(*at(0.45, 0.2), steps=6)
    page.mouse.up()
    page.wait_for_timeout(200)
    moved = strands()[1][:2]
    page.mouse.move(*at(0.3, 0.25))
    page.mouse.down()
    page.mouse.move(*at(0.3, 0.42), steps=6)
    page.mouse.up()
    page.wait_for_timeout(200)
    got = strands()
    pulled = got[1][:3]
    check("Dây đèn tự vẽ: a point dragged where it is let go; a wire pulled bends at a new point",
          all(abs(a - b) < 0.004 for a, b in zip(moved, (0.45, 0.2))) and len(got) == len(clicked) + len(hand) + 1
          and all(abs(a - b) < 0.004 for a, b in zip(pulled, (0.3, 0.42, 0))), str([moved, pulled, len(got)]))
    n0 = len(got)
    page.keyboard.press("Delete")  # (the strand chosen, never the clip)
    page.wait_for_timeout(200)
    n1 = len(strands())
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    n2 = len(strands())
    # the strand being drawn: Backspace, Ctrl+Z, Ctrl+Y on its points; Esc ends it, then the drawing
    for fx_, fy_ in [(0.1, 0.9), (0.3, 0.92), (0.5, 0.9)]:
        page.mouse.click(*at(fx_, fy_))
        page.wait_for_timeout(100)
    page.keyboard.press("Backspace")
    page.keyboard.press("Control+z")
    a = page.evaluate("S.wire.cur.length")
    page.keyboard.press("Control+y")
    b = page.evaluate("S.wire.cur.length")
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    n3 = len(strands())
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    off_ = page.evaluate("[!!S.wire, stage.classList.contains('wiring'), S.sel]")
    check("Dây đèn tự vẽ: Delete removes the strand chosen (undone); Backspace, Ctrl+Z, Ctrl+Y on the strand being "
          "drawn; Esc ends it, then the drawing", [n1, n2, a, b, n3] == [len(hand), n0, 1, 2, n0 + 2]
          and off_ == [False, False, cid], str([n0, n1, n2, a, b, n3, off_]))
    page.mouse.click(*at(0.6, 0.1))  # the clip chosen: a click on the picture draws again
    page.wait_for_timeout(200)
    again = page.evaluate("[!!S.wire, S.wire && S.wire.cur.length]")
    page.keyboard.press("Escape")
    page.click("#wireClear")
    n4 = len(strands())
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    n5 = len(strands())
    page.click("#wireDraw")  # done drawing
    page.wait_for_timeout(200)
    done = page.evaluate("[!!S.wire, $('#polyCanvas').style.display]")
    check("Dây đèn tự vẽ: a click on the picture draws again; Xoá hết dây undone; done drawing",
          again == [True, 1] and [n4, n5] == [0, n0 + 2] and done == [False, "none"], str([again, n4, n5, done]))
    page.wait_for_function(f"S.fx['{cid}'] && S.fx['{cid}'].ready && S.els['{cid}']", timeout=300000)
    laid = page.evaluate(f"""[S.els['{cid}'].media.getAttribute('src').includes('.mp4'),
        S.els['{cid}'].wrap.style.mixBlendMode]""")
    check("Dây đèn tự vẽ: its preview clip, light on black (H.264, Screen)", laid == [True, "screen"], str(laid))
    shot("05j-garland")
    page.evaluate(f"(() => {{ select('{cid}'); deleteClip(clipById('{cid}')); edited(); }})()")


def chimneys(page, shot):
    """Khói ống khói: a new clip starts putting chimneys at once (no box, nothing to place or turn); a click puts one
    where it is (a yellow ellipse), as wide as a new one is, and chooses it; its end dragged makes it wider (the next
    one put as wide as the one chosen); its ellipse dragged moves it. The renderer's plumes stand in the boxes the page
    lays. A click chooses one: its own smoke; Delete removes the one chosen (undone); Esc ends; a click on the picture
    puts again; Xoá hết undone. Its preview clip: see-through smoke over the whole frame (WebM, Normal)."""
    from fx import chimney as fx_chimney, effects as fx_effects
    page.click('.tab[data-bin="fx"]')
    page.dblclick('.tile[data-kind="chimney"]')
    cid = page.evaluate("S.sel")
    page.wait_for_timeout(300)
    vw, vh = page.evaluate("[info().w, info().h]")
    state = page.evaluate("""(c => [c.kind, !!S.stack && S.stack.id === c.id, $('#xform').style.display,
        !!$('#stackPlace'), !!$('#ec [data-param="tall"]'), !!$('#ec [data-param="scale"]'),
        !!$('#ec [data-num="rot_x"]'), stage.classList.contains('stacking'),
        [...document.querySelectorAll('#ec h4')].map(h => h.textContent).filter(t => t.includes('khói'))])(clipById(S.sel))""")
    check("Khói ống khói: putting chimneys starts at once; no box, nothing to place or turn; no empty heading for the "
          "chimneys (put on the picture)", state == ["chimney", True, "none", True, True, False, False, True, []],
          str(state))
    box = page.locator("#screen").bounding_box()

    def at(fx_, fy_):
        return box["x"] + box["width"] * fx_, box["y"] + box["height"] * fy_

    def stacks():
        return page.evaluate(f"clipById('{cid}').params.stacks")

    def drag(a, b):
        page.mouse.move(*at(*a))
        page.mouse.down()
        page.mouse.move(*at(*b), steps=6)
        page.mouse.up()
        page.wait_for_timeout(200)

    def near(q, want, tol=0.004):
        return all(abs(a - b) < tol for a, b in zip(q, want))

    mouth, much = page.evaluate("[S.schema.kinds.chimney.mouth, S.schema.kinds.chimney.much]")
    for fx_, fy_ in [(0.3, 0.6), (0.7, 0.45)]:
        page.mouse.click(*at(fx_, fy_))
        page.wait_for_timeout(150)
    got, sel = stacks(), page.evaluate("S.stack.sel")
    check("Khói ống khói: a click puts a chimney where it is, as wide as a new one, and chooses it",
          len(got) == 2 and near(got[0], (0.3, 0.6, mouth, much)) and near(got[1], (0.7, 0.45, mouth, much))
          and sel == 1, str([got, sel]))
    drag((0.7 + mouth / 2, 0.45), (0.7 + mouth / 2 + 0.02, 0.45))  # its right end: wider (the left one mirrors it)
    wide = stacks()[1]
    page.mouse.click(*at(0.5, 0.3))  # a new one: as wide as the one chosen
    page.wait_for_timeout(200)
    third = stacks()[2]
    drag((0.3, 0.6), (0.35, 0.62))  # the first one's ellipse: moved
    got = stacks()
    check("Khói ống khói: an end dragged makes it wider, the next one put as wide; an ellipse dragged moves it",
          near(wide, (0.7, 0.45, mouth + 0.04)) and near(third, (0.5, 0.3, wide[2], much))
          and near(got[0], (0.35, 0.62, mouth)), str([wide, third, got[0]]))
    # the renderer's plumes, in the boxes the page lays
    params = page.evaluate(f"clipById('{cid}').params")
    fx = fx_effects.make("chimney", params, 3, 48, 24, vw, vh)
    shown = page.evaluate(f"(c => c.params.stacks.map(q => stackBox(c, q)))(clipById('{cid}'))")
    want = sorted((round(b[0], 2), round(b[1], 2), round(b[2], 2)) for b in (p_.box()[:3] for p_ in fx.plumes))
    have = sorted((round(b[0], 2), round(b[1], 2), round(b[2], 2)) for b in shown)
    mine = sorted(tuple(round(v, 2) for v in fx_chimney.plume_box(*q[:3], params["tall"], vw, vh)) for q in got)
    off = max((abs(a - b) for w_, h_ in zip(want, have) for a, b in zip(w_, h_)), default=99)
    check("Khói ống khói: the renderer's plumes stand in the boxes the page lays", len(want) == 3 and want == mine
          and off < 0.05, f"{want} / {have}, {off:.3f} px")
    page.mouse.move(*at(0.5, 0.5))
    shot("05k-chimneys-putting")
    # a click on one chooses it: its own smoke; Delete removes it (undone); Esc ends; a click on the picture again
    page.mouse.click(*at(0.35, 0.62))
    page.wait_for_timeout(200)
    chosen = page.evaluate("[S.stack.sel, !!$('#stackMuch'), !!$('#stackDel') && !$('#stackDel').disabled]")
    page.evaluate("""(() => { const r = $('#stackMuch'); r.value = 100; r.dispatchEvent(new Event('input', {bubbles: true}));
        r.dispatchEvent(new Event('change', {bubbles: true})); })()""")
    page.wait_for_timeout(200)
    own = [q[3] for q in stacks()]
    check("Khói ống khói: a click on a chimney chooses it; its own smoke slider is that chimney's",
          chosen == [0, True, True] and own == [1.0, much, much], str([chosen, own]))
    page.keyboard.press("Delete")  # (the chimney chosen, never the clip)
    page.wait_for_timeout(200)
    n1 = len(stacks())
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    n2 = len(stacks())
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    off_ = page.evaluate("[!!S.stack, stage.classList.contains('stacking'), S.sel]")
    page.mouse.click(*at(0.7, 0.45))  # the clip chosen: a click on the picture puts again (here: chooses that one)
    page.wait_for_timeout(200)
    again = page.evaluate("[!!S.stack, S.stack && S.stack.sel]")
    n3 = len(stacks())
    page.click("#stackClear")
    n4 = len(stacks())
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    n5 = len(stacks())
    page.click("#stackPlace")  # done
    page.wait_for_timeout(200)
    done = page.evaluate("[!!S.stack, $('#polyCanvas').style.display]")
    check("Khói ống khói: Delete removes the chimney chosen (undone); Esc ends; a click on the picture puts again; Xoá "
          "hết undone; done", [n1, n2, n3, n4, n5] == [2, 3, 3, 0, 3] and off_ == [False, False, cid]
          and again == [True, 1] and done == [False, "none"], str([n1, n2, off_, again, n3, n4, n5, done]))
    page.wait_for_function(f"S.fx['{cid}'] && S.fx['{cid}'].ready && S.els['{cid}']", timeout=300000)
    laid = page.evaluate(f"""[S.els['{cid}'].media.getAttribute('src').includes('.webm'),
        S.els['{cid}'].wrap.style.mixBlendMode || 'normal']""")
    check("Khói ống khói: its preview clip, see-through smoke over the whole frame (WebM, Normal)",
          laid == [True, "normal"], str(laid))
    shot("05k-chimneys")
    page.evaluate(f"(() => {{ select('{cid}'); deleteClip(clipById('{cid}')); edited(); }})()")


def shortcuts(page, shot):
    """The Phím tắt window: a key changed, taken from another action, refused, searched; the guide; all back."""
    def row(i):
        return f'.krow[data-act="{i}"]'

    def take(i, key):  # "+ Phím" on the row, then the key
        page.click(f"{row(i)} [data-kadd]")
        page.wait_for_selector(f"{row(i)}.wait")
        page.keyboard.press(key)
        page.wait_for_timeout(300)

    check("key names in the buttons' tips", "Phím cách" in page.get_attribute("#tPlay", "title") and
          "Ctrl+Z" in page.get_attribute("#undoBtn", "title"), page.get_attribute("#tPlay", "title"))
    page.click("#keysBtn")
    page.wait_for_selector("#kList .krow")
    n = page.evaluate("S.keymap.actions.length")
    check("the Phím tắt window lists every action", page.locator(".krow[data-act]").count() == n and n > 30, str(n))
    take("razorTool", "d")
    check("a new key for the razor", page.evaluate("S.keys.razorTool") == ["KeyC", "KeyD"] and
          "đã đổi" in page.inner_text(row("razorTool")), str(page.evaluate("S.keys.razorTool")))
    take("razorTool", "v")  # the select tool's: asked first
    check("a key in use: asked first", "Công cụ chọn" in page.inner_text(f"{row('razorTool')} .kmsg") and
          page.evaluate("S.keys.selectTool") == ["KeyV"])
    page.click(f"{row('razorTool')} [data-kmove]")
    page.wait_for_timeout(300)
    check("a key taken from another action", page.evaluate("S.keys.razorTool") == ["KeyC", "KeyD", "KeyV"] and
          page.evaluate("S.keys.selectTool") == [], str(page.evaluate("[S.keys.razorTool, S.keys.selectTool]")))
    page.click(f'{row("razorTool")} .kdel[data-kdel="KeyC"]')
    page.wait_for_timeout(300)
    check("a key taken away", page.evaluate("S.keys.razorTool") == ["KeyD", "KeyV"])
    take("export", "Tab")
    check("a key the app cannot use: said so", "không dùng được" in page.inner_text(f"{row('export')} .kmsg") and
          page.evaluate("S.keys.export") == ["Ctrl+KeyM"])
    page.click(f"{row('export')} [data-kok]")
    page.click(f"{row('split')} [data-kadd]")
    page.keyboard.press("Escape")  # changed one's mind: the window stays
    page.wait_for_timeout(200)
    check("Esc while waiting: nothing changed", page.evaluate("S.keyWait") is None and page.is_visible("#kd") and
          page.evaluate("S.keys.split") == ["Ctrl+KeyK"])
    page.fill("#kFind", "xuat video")  # (without accents too)
    page.wait_for_timeout(200)
    check("search the actions", page.locator(".krow[data-act]").count() == 1 and page.is_visible(row("export")))
    page.fill("#kFind", "")
    shot("05i-keys")
    page.click('[data-ktab="guide"]')
    page.wait_for_selector("#kGuide h3")
    text = page.inner_text("#kGuide")
    check("the guide in the window", page.locator("#kGuide h3").count() >= 5 and page.locator("#kGuide table").count() >= 5
          and "**" not in text and "|---" not in text and "<!--" not in text and
          page.locator('#kGuide [data-ktab="keys"]').count() == 1, text[:300])
    shot("05i-guide")
    page.keyboard.press("Escape")
    page.wait_for_timeout(200)
    check("Esc closes the window", not page.is_visible("#modal"))
    saved = page.evaluate("get('/api/keys')")
    check("the keys are kept by the app", saved["custom"].get("razorTool") == ["KeyD", "KeyV"] and
          saved["custom"].get("selectTool") == [], str(saved["custom"]))
    page.evaluate("S.tool = 'select'; renderTimeline()")
    page.keyboard.press("d")
    on_d = page.evaluate("S.tool")
    page.evaluate("S.tool = 'select'; renderTimeline()")
    page.keyboard.press("v")
    check("the new keys work at once", on_d == "razor" and page.evaluate("S.tool") == "razor", on_d)
    page.keyboard.press("Shift+Slash")  # the window's own key
    page.wait_for_selector("#kList .krow")
    page.click("#kResetAll")
    page.click("#kResetAll")  # (a second press)
    page.wait_for_timeout(300)
    check("all back to the defaults", page.evaluate("S.keymap.custom") == {} and page.evaluate("S.keys.selectTool") == ["KeyV"]
          and page.evaluate("S.keys.razorTool") == ["KeyC"])
    page.click("#kClose")
    page.keyboard.press("v")
    check("the default keys again", page.evaluate("S.tool") == "select")


def activation(page, shot):
    """No key on this computer: only the activation screen, with the machine code; a wrong key is refused with a
    reason; the right one opens the app. The footer says it is activated and shows the screen again on a click."""
    page.wait_for_selector("#lic", state="visible")
    machine = licensing.status(fresh=True)["machine"]
    check("activation screen: the machine code, nothing else started", page.inner_text("#licMachine") == machine and
          page.evaluate("!S.schema"), page.inner_text("#licMachine"))
    shot("00-activation")
    page.fill("#licKey", license_keys.make_key(SECRET, machine, ["loop"]))
    page.click("#licGo")
    page.wait_for_function("document.querySelector('#licErr').textContent.length > 0")
    check("activation screen: a key for another app refused, with the reason",
          "Bokin FX" in page.inner_text("#licErr"), page.inner_text("#licErr"))
    page.fill("#licKey", license_keys.make_key(SECRET, machine, ["fx"]))
    with page.expect_navigation():
        page.click("#licGo")
    page.wait_for_function("S.schema && S.state")
    check("activation: the app opens", not page.is_visible("#lic") and page.inner_text("#licChip") == "Đã kích hoạt")
    page.click("#licChip")
    check("the licence from the footer: the machine code, closable",
          page.is_visible("#lic") and page.inner_text("#licTitle") == "Bản quyền Bokin FX" and page.is_visible("#licClose"))
    page.click("#licClose")


def disc(png):
    """Where the loop's yellow disc is in a screenshot (x, y), or None."""
    import io
    a = np.asarray(Image.open(io.BytesIO(png)).convert("RGB")).astype(int)
    m = (a[..., 0] > 200) & (a[..., 1] > 170) & (a[..., 2] < 170) & (a[..., 2] > 90)
    ys, xs = np.nonzero(m)
    return (float(xs.mean()), float(ys.mean())) if len(xs) > 50 else None


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


def move_the_picture(page, fid, drag, shot):
    """The picture moves freely, as in Premiere: smaller than "Vừa khung", moved with the hand tool (H), the middle
    button, two fingers on a touchpad or Shift + wheel, never lost; a clip outside the frame is grabbed there by its
    box; Ctrl 0, \\ or a double-click around the picture put it back in the middle; painting lands under the mouse."""
    looks = []
    page.on("request", lambda r: looks.append(r.url) if "/api/look" in r.url else None)
    scr = "(() => { const r = screenEl.getBoundingClientRect(); return [r.left, r.top, r.width, r.height]; })()"
    zoom = ("[S.zoom, S.scale, $('#zoomSel').value, $('#zoomSel').selectedOptions[0].textContent, "
            "Math.round(100 * zoomOf(S.scale, fitSize())) + '%']")
    params = f"Object.assign({{}}, clipById('{fid}').params)"
    st = page.locator("#stage").bounding_box()
    sx, sy = st["x"] + st["width"] / 2, st["y"] + st["height"] / 2

    def moved(a, b, dx, dy):  # the picture moved by (dx, dy), at the same size
        return abs(b[0] - a[0] - dx) < 1 and abs(b[1] - a[1] - dy) < 1 and abs(b[2] - a[2]) < 0.01

    page.mouse.move(sx, sy)
    page.keyboard.press("Control+0")
    fit = page.evaluate(scr)
    page.keyboard.press("Control+Minus")
    page.keyboard.press("Control+Minus")
    z = page.evaluate(zoom)
    check("zoom out below Vừa khung (Ctrl -), the menu shows the size it is",
          z[1] < 0.7 and z[2] == "" and z[3] == z[4], str(z))
    at = (st["x"] + st["width"] * 0.3, st["y"] + st["height"] * 0.35)
    where = (f"(() => {{ const r = screenEl.getBoundingClientRect(); "
             f"return [({at[0]} - r.left) / r.width, ({at[1]} - r.top) / r.height, S.scale]; }})()")
    page.mouse.move(*at)
    p0 = page.evaluate(where)
    page.mouse.wheel(0, 100)  # a notch towards you: smaller, about the mouse
    page.wait_for_timeout(150)
    p1 = page.evaluate(where)
    check("the wheel zooms about the mouse: the point under it stays", p1[2] < p0[2] and
          abs(p1[0] - p0[0]) < 1e-3 and abs(p1[1] - p0[1]) < 1e-3, f"{p0} -> {p1}")
    page.select_option("#zoomSel", "0.1")
    z = page.evaluate(zoom)
    check("the menu's 10%", z[0] == "0.1" and z[2] == "0.1" and z[4] == "10%", str(z))

    # the hand tool in "Vừa khung": a drag that starts on the clip's box moves the picture, not the clip; the app is
    # not asked for a new exact frame (it moves with the picture)
    page.mouse.move(sx, sy)
    page.keyboard.press("Control+0")
    page.evaluate(f"select('{fid}')")
    page.wait_for_timeout(800)  # (the exact frame of this zoom asked for)
    before = page.evaluate(params)
    page.click("#handBtn")
    b0 = page.evaluate(scr)
    looks.clear()
    xb = page.locator("#xform").bounding_box()
    x0, y0 = xb["x"] + xb["width"] / 2, xb["y"] + xb["height"] / 2
    drag(x0, y0, x0 + 120, y0 + 70)
    page.wait_for_timeout(600)
    b1 = page.evaluate(scr)
    z = page.evaluate(zoom)
    check("the hand tool moves the whole picture from Vừa khung, not the clip", moved(b0, b1, 120, 70) and
          page.evaluate(params) == before and z[0] != "fit" and z[3] == z[4], f"{b0} -> {b1}, {z}")
    check("moving the picture asks nothing of the app (no new exact frame)", not looks, str(looks))
    page.keyboard.press("Escape")
    h1 = page.evaluate("[S.hand, S.sel]")
    page.keyboard.press("h")
    h2 = page.evaluate("S.hand && $('#handBtn').classList.contains('on')")
    page.keyboard.press("v")
    check("H takes the hand tool, Esc or V puts it down", h1 == [False, fid] and h2 is True and
          page.evaluate("S.hand") is False, str([h1, h2]))
    # H held down (the key repeats, as Windows sends it) while dragging: the hand for that drag only
    b0 = page.evaluate(scr)
    page.mouse.move(sx, sy)
    page.keyboard.down("h")
    for _ in range(4):
        page.keyboard.down("h")  # (repeat: true)
    held = page.evaluate("S.hand")
    drag(sx, sy, sx + 40, sy + 25)
    page.keyboard.up("h")
    b1 = page.evaluate(scr)
    let_go = page.evaluate("S.hand")
    check("H held while dragging: the hand stays through the key's repeats, and goes when H comes up",
          held is True and moved(b0, b1, 40, 25) and let_go is False and page.evaluate(params) == before,
          str([held, b0, b1, let_go]))

    # Space held: a drag moves the picture and plays nothing (the clip's box stays); Space alone plays and stops
    page.evaluate("document.activeElement && document.activeElement.blur()")
    b0 = page.evaluate(scr)
    page.keyboard.down("Space")
    page.keyboard.down("Space")  # (a repeat)
    drag(sx, sy, sx - 30, sy + 20)
    page.keyboard.up("Space")
    b1 = page.evaluate(scr)
    sp = page.evaluate("[S.playing, S.space, $('#xform').style.display, stage.classList.contains('grab')]")
    page.keyboard.press("Space")
    p1 = page.evaluate("S.playing")
    page.keyboard.press("Space")
    p2 = page.evaluate("S.playing")
    check("Space + drag moves the picture without playing; Space alone still plays and stops",
          moved(b0, b1, -30, 20) and sp == [False, False, "block", False] and p1 is True and p2 is False,
          str([b0, b1, sp, p1, p2]))

    # the keys + - (and Ctrl + -) with the mouse on the picture zoom about the mouse, as the wheel
    page.keyboard.press("Control+0")
    at = (st["x"] + st["width"] * 0.2, st["y"] + st["height"] * 0.25)
    under = (f"(() => {{ const r = screenEl.getBoundingClientRect(); "
             f"return [({at[0]} - r.left) / r.width, ({at[1]} - r.top) / r.height, S.scale]; }})()")
    page.mouse.move(*at)
    ks = [page.evaluate(under)]
    for key in ("Minus", "Control+Minus", "Equal", "Control+Equal"):
        page.keyboard.press(key)
        ks.append(page.evaluate(under))
    check("+ - and Ctrl + - with the mouse on the picture: the point under the mouse stays",
          ks[1][2] < ks[0][2] and ks[2][2] < ks[1][2] and ks[4][2] > ks[3][2] > ks[2][2] and
          all(abs(k[0] - ks[0][0]) < 1e-3 and abs(k[1] - ks[0][1]) < 1e-3 for k in ks), str(ks))

    # the middle button, two fingers on a touchpad (sideways too), Shift + wheel (sideways)
    page.mouse.move(sx, sy)
    b0 = page.evaluate(scr)
    page.mouse.down(button="middle")
    page.mouse.move(sx - 50, sy - 30, steps=4)
    page.mouse.up(button="middle")
    b1 = page.evaluate(scr)
    page.mouse.wheel(30, 20)
    page.wait_for_timeout(100)
    b2 = page.evaluate(scr)
    page.keyboard.down("Shift")
    page.mouse.wheel(0, 40)
    page.keyboard.up("Shift")
    page.wait_for_timeout(400)
    b3 = page.evaluate(scr)
    check("the middle button, two fingers and Shift + wheel move the picture", moved(b0, b1, -50, -30) and
          moved(b1, b2, -30, -20) and moved(b2, b3, -40, 0), str([b0, b1, b2, b3]))
    page.mouse.wheel(-6000, -6000)  # far away: a corner of the frame stays on the monitor
    page.wait_for_timeout(100)
    r = page.evaluate(scr)
    seen = (min(r[0] + r[2], st["x"] + st["width"]) - max(r[0], st["x"]),
            min(r[1] + r[3], st["y"] + st["height"]) - max(r[1], st["y"]))
    check("the picture is never lost", seen[0] >= 60 and seen[1] >= 60, f"{r} in {st}")
    page.mouse.dblclick(st["x"] + 30, st["y"] + 30)  # on the empty area around it
    page.wait_for_timeout(200)
    r = page.evaluate(scr)
    check("a double-click around the picture: Vừa khung, in the middle", page.evaluate("S.zoom") == "fit" and
          all(abs(a - b) < 0.5 for a, b in zip(r, fit)), f"{r} / {fit}")

    # a clip wholly outside the frame, not selected: nothing of it shows there, so a press there neither picks nor
    # moves it, and a double-click there is "Vừa khung"; picked on the timeline, its picture shows there (faintly)
    # and its box with it; a corner sizes it there and its box drags it back
    page.evaluate(f"(() => {{ const c = clipById('{fid}'); Object.assign(c.params, {{x: 90, y: 0, scale: 40}}); "
                  "S.sel = null; edited(); })()")
    page.select_option("#zoomSel", "0.25")
    page.wait_for_timeout(300)
    where_out = (f"(() => {{ const q = placed(clipById('{fid}')), sc = screenEl.getBoundingClientRect(); "
                 "return [sc.left + (q.x + q.w / 2) / info().w * sc.width, sc.top + (q.y + q.h / 2) / info().h "
                 "* sc.height, sc.right, sc.left + sc.width / 2, sc.top + sc.height / 2]; })()")
    out = page.evaluate(where_out)
    p0 = page.evaluate(params)
    drag(out[0], out[1], out[0] - 40, out[1] + 10)
    blind = page.evaluate(f"[S.sel, {params}]")
    page.mouse.dblclick(out[0], out[1])
    page.wait_for_timeout(200)
    check("outside the frame, a clip not selected (not shown there) is not picked by a press, nor moved; "
          "a double-click there is Vừa khung", out[0] > out[2] and blind == [None, p0] and
          page.evaluate("S.zoom") == "fit", str([out, blind, p0]))
    page.select_option("#zoomSel", "0.25")
    page.wait_for_timeout(300)
    out = page.evaluate(where_out)
    page.click(f".clip[data-clip='{fid}']")
    page.wait_for_timeout(300)
    sel = page.evaluate(f"[S.sel === '{fid}', $('#xform').style.display, "
                        f"S.els['{fid}'].wrap.classList.contains('spill'), getComputedStyle($('#frame')).display]")
    xb = page.locator("#xform").bounding_box()
    check("a clip outside the frame, picked on the timeline: its box shows where it lies",
          sel == [True, "block", True, "block"] and abs(xb["x"] + xb["width"] / 2 - out[0]) < 2 and
          abs(xb["y"] + xb["height"] / 2 - out[1]) < 2, str([out, sel, xb]))
    shot("05c-outside-the-frame")
    xb = page.locator("#xform").bounding_box()
    s0 = page.evaluate(params)["scale"]
    drag(xb["x"] + xb["width"] - 1, xb["y"] + xb["height"] - 1, xb["x"] + xb["width"] + 30, xb["y"] + xb["height"] + 17)
    s1 = page.evaluate(params)["scale"]
    xb = page.locator("#xform").bounding_box()
    drag(xb["x"] + xb["width"] / 2, xb["y"] + xb["height"] / 2, out[3], out[4])
    back = page.evaluate(params)
    check("outside the frame: a corner sizes it, its box drags it back", s1 > s0 * 1.2 and abs(back["x"]) < 3 and
          abs(back["y"]) < 3, f"{s0} -> {s1}, {back}")
    page.evaluate(f"(() => {{ Object.assign(clipById('{fid}').params, {{x: 0, y: 0, scale: 40}}); edited(); }})()")

    # the selected clip's picture shows faintly outside the frame (a picture: it shows in any browser), and the
    # picture inside the frame is just as it was
    lid = page.evaluate("""(() => { const t = {id: newId("t"), on: true}; S.project.tracks.push(t);
        addClip({lib: S.lib.find(i => i.info.image).id}, t.id, 0, total());
        const c = clipById(S.sel); Object.assign(c.params, {x: 60, y: 0, scale: 50}); edited(); return c.id; })()""")
    page.select_option("#zoomSel", "0.5")
    page.wait_for_function(f"S.els['{lid}'] && S.els['{lid}'].media.complete && "
                           f"S.els['{lid}'].wrap.style.display === ''", timeout=30000)
    page.wait_for_timeout(1000)
    pts = page.evaluate("""(() => { const sc = screenEl.getBoundingClientRect(), W = info().w, H = info().h;
        const at = (x, y) => [sc.left + x / W * sc.width, sc.top + y / H * sc.height];
        return [at(1.2 * W, 0.5 * H), at(1.2 * W, 0.87 * H)]; })()""")  # in the logo's yellow; under it, nothing

    def px(png, p):
        return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))[int(p[1]), int(p[0])].astype(int).tolist()

    on = page.screenshot()
    page.keyboard.press("Escape")  # no clip selected
    page.wait_for_timeout(200)
    off = page.screenshot()
    esc = page.evaluate(f"[S.sel, $('#xform').style.display, S.els['{lid}'].wrap.classList.contains('spill')]")
    yellow, empty, gone = px(on, pts[0]), px(on, pts[1]), px(off, pts[0])
    check("the selected clip's picture shows faintly outside the frame, and only while it is selected (Esc: gone, "
          "its box too)", yellow[0] > empty[0] + 20 and yellow[0] > yellow[2] + 20 and
          max(abs(a - b) for a, b in zip(gone, empty)) <= 3 and esc == [None, "none", False],
          str([yellow, empty, gone, esc]))
    page.evaluate(f"(() => {{ select('{lid}'); xform.style.display = 'none'; invalidateExact(); }})()")
    page.wait_for_timeout(300)
    inside = page.locator("#screen").screenshot()
    page.evaluate(f"S.els['{lid}'].wrap.classList.remove('spill')")
    page.wait_for_timeout(200)
    diff = np.abs(np.asarray(Image.open(io.BytesIO(inside)).convert("RGB"), np.int16) - np.asarray(
        Image.open(io.BytesIO(page.locator("#screen").screenshot())).convert("RGB"), np.int16)).max(axis=2)
    diff = diff[2:-2, 2:-2]  # (the screenshot's edge pixels lie partly outside the frame)
    # the same picture: at most the smoothing of a few edge pixels differs (the browser composites the layer anew)
    changed = float((diff > 8).mean())
    check("the picture inside the frame is the same with the clip's outside part shown",
          changed < 0.01 and float(diff.mean()) < 0.5, f"{100 * changed:.2f}% changed, mean {diff.mean():.2f}")
    page.evaluate(f"(() => {{ const c = clipById('{lid}'), t = c.track; deleteClip(c); "
                  "S.project.tracks = S.project.tracks.filter(x => x.id !== t); edited(); })()")

    page.mouse.wheel(40, 0)
    page.keyboard.press("Backslash")  # the mouse on the picture: Vừa khung
    r1 = page.evaluate(scr)
    page.mouse.wheel(-40, 25)
    page.keyboard.press("Control+0")
    r2 = page.evaluate(scr)
    check(r"\ and Ctrl 0: Vừa khung, in the middle", all(abs(a - b) < 0.5 for r in (r1, r2) for a, b in zip(r, fit)),
          str([r1, r2, fit]))
    tl = page.locator("#tlScroll").bounding_box()
    page.mouse.move(tl["x"] + tl["width"] / 2, tl["y"] + tl["height"] / 2)
    pps = page.evaluate("S.pps")
    page.keyboard.press("Equal")
    check("+ with the mouse on the timeline still zooms the timeline", page.evaluate("[S.pps > " + str(pps) +
          ", S.scale, S.zoom]") == [True, 1, "fit"])
    page.keyboard.press("Minus")

    # painting: the picture zoomed and moved, a dab of the brush lands under the mouse, with the brush circle
    page.mouse.move(sx, sy)
    page.select_option("#zoomSel", "2")
    page.keyboard.press("h")
    drag(sx, sy, sx - 90, sy + 40)
    page.keyboard.press("v")
    page.click(".clip.base >> nth=0")
    page.click("#paintFg")
    page.mouse.move(sx, sy)
    ring = page.evaluate("(() => { const r = $('#cursor').getBoundingClientRect(); "
                         "return [r.left + r.width / 2, r.top + r.height / 2, $('#cursor').style.display]; })()")
    page.mouse.down()
    page.mouse.up()
    page.wait_for_timeout(200)
    dab = page.evaluate(f"""(() => {{ const r = paintCanvas.getBoundingClientRect(), m = S.masks.fg, g = mctx(m);
        const x = ({sx} - r.left) / r.width * m.canvas.width, y = ({sy} - r.top) / r.height * m.canvas.height;
        const d = S.brush.size / 2;
        const a = (u, v) => g.getImageData(Math.round(u), Math.round(v), 1, 1).data[3];
        return [a(x, y), a(x - .8 * d, y), a(x + .8 * d, y), a(x, y - .8 * d), a(x, y + .8 * d),
                a(x - 1.3 * d, y), a(x + 1.3 * d, y), a(x, y - 1.3 * d), a(x, y + 1.3 * d)]; }})()""")
    check("the picture moved and zoomed: the brush lands under the mouse, its circle there too",
          all(v > 0 for v in dab[:5]) and not any(dab[5:]) and abs(ring[0] - sx) < 1.5 and abs(ring[1] - sy) < 1.5
          and ring[2] == "block", str([dab, ring, (sx, sy)]))
    shot("05d-paint-moved")
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    page.keyboard.press("Control+z")  # the painting undone
    page.wait_for_timeout(300)
    page.keyboard.press("Control+0")
    check("the painting undone", page.evaluate("S.masks.fg.empty") is True)
    page.evaluate(f"select('{fid}')")
    page.wait_for_timeout(300)


def run(page, folder, edge):
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "400" not in m.text else None)

    def shot(name):
        page.screenshot(path=os.path.join(SHOTS, name + ".png"))

    def clips():
        return page.evaluate("S.project.layers.map(c => ({id: c.id, kind: c.kind, track: c.track, start: c.start, "
                             "end: c.end, skip: c.skip}))")

    def lane_y(i):  # rows from the top: new track, V4, V3, V2, V1 (scrolled into view: many tracks overflow; in one
        # step, as the timeline is drawn anew now and then while preview clips are made)
        return page.evaluate(f"""(() => {{ const row = document.querySelectorAll('.lane[data-row]')[{i}];
            row.scrollIntoView({{block: 'nearest'}}); const r = row.getBoundingClientRect();
            return r.top + r.height / 2; }})()""")

    def x_of(sec):
        return page.locator("#tlLanes").bounding_box()["x"] + sec * page.evaluate("S.pps")

    def drag(x0, y0, x1, y1):
        page.mouse.move(x0, y0)
        page.mouse.down()
        page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, steps=4)
        page.mouse.move(x1, y1, steps=4)
        page.mouse.up()
        page.wait_for_timeout(300)

    activation(page, shot)
    shot("01-start")
    page.set_input_files("#fileVideo", upload(folder, "loop.mp4", "Quán cà phê.mp4", "video/mp4"))
    page.wait_for_function("S.project && S.project.id", timeout=60000)
    check("a loop opens as a project", page.evaluate("S.project.name") == "Quán cà phê")
    page.set_input_files("#fileLib", [upload(folder, "dust.mp4", "Bụi sáng nền đen.mp4", "video/mp4"),
                                      upload(folder, "logo.png", "Logo.png", "image/png")])
    page.wait_for_function("S.lib.length === 2", timeout=60000)
    page.wait_for_function("S.project.proxy && S.lib.every(i => i.proxy)", timeout=120000)
    check("library: both files, with light copies", True)

    # the effect video (3 s) dropped on V2 at 6 s: as long as the file, not stretched to the end
    scroll = page.locator("#tlScroll").bounding_box()
    v2 = lane_y(3)
    page.locator(".tile[data-lib]").last.drag_to(page.locator("#tlScroll"), target_position={
        "x": x_of(6) - scroll["x"], "y": v2 - scroll["y"]})
    page.wait_for_timeout(400)
    c = clips()
    check("drop on the timeline", len(c) == 1 and abs(c[0]["start"] - 144) <= 2 and
          c[0]["end"] - c[0]["start"] == 72, str(c))
    drag(x_of(c[0]["end"] / 24) - 2, v2, x_of(20), v2)  # pull its end out to 20 s
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
    page.evaluate("seek(10)")
    page.dblclick('.tile[data-kind="snow"]')
    snow = page.evaluate("S.project.layers.find(l => l.kind === 'snow')")
    check("a double-click adds 5 s at the playhead", snow["start"] == 240 and snow["end"] == 360,
          str([snow["start"], snow["end"]]))
    page.click("#fullBtn")  # "Suốt video": over the whole video, looping by itself
    glass = page.evaluate("[...document.querySelectorAll('.tile[data-kind]')].map(t => t.dataset.kind)")
    check("the built-in effects offered",
          glass == ["snow", "rain", "glass", "frost", "steam", "chimney", "candle", "leaves", "foliage", "flicker",
                    "lights", "garland"],  # (Hạt sáng no longer offered: an older project's clip of it still plays)
          str(glass))
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
    shown = page.evaluate("(() => { pause(); return {frame: curFrame() % loopFrames(), base: baseVid.currentTime}; })()")
    if edge:  # the pause keeps the frame the video stopped on
        check("pause keeps the frame on screen", int(shown["base"] * 24 + 0.01) == shown["frame"], str(shown))
    page.wait_for_function("baseFrame.style.display === 'none' || baseFrame.complete")  # (no H.264: a picture of it)
    first = page.locator("#screen").screenshot()
    live = disc(first)
    if edge:  # everything plays in Edge: a pause keeps the picture as it is (no other picture swapped in)
        page.wait_for_timeout(1200)
        later = page.locator("#screen").screenshot()
        change = float(np.abs(np.asarray(Image.open(io.BytesIO(first)).convert("L"), np.float32) -
                              np.asarray(Image.open(io.BytesIO(later)).convert("L"), np.float32)).mean())
        check("a pause keeps the picture still", change < 1.5 and
              page.evaluate("document.querySelector('#exact').style.display") != "block", f"mean change {change:.2f}")
    if not page.evaluate("S.exactOn"):
        page.click("#tExact")  # the exact frame, asked for
    page.wait_for_function("document.querySelector('#exact').style.display === 'block'", timeout=60000)
    page.wait_for_timeout(300)
    ready = disc(page.locator("#screen").screenshot())
    check("paused: the live preview shows the same frame as the renderer", live is not None and ready is not None and
          abs(live[0] - ready[0]) < 3 and abs(live[1] - ready[1]) < 3, f"disc at {live} live, {ready} rendered")
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
    shortcuts(page, shot)

    page.click(".clip.base >> nth=0")
    page.click("#paintFreeze")
    box = page.locator("#screen").bounding_box()
    drag(box["x"] + 40, box["y"] + box["height"] - 40, box["x"] + 160, box["y"] + box["height"] - 60)
    shot("05-paint")
    page.keyboard.press("Escape")
    page.wait_for_function("S.project.masks.includes('freeze')", timeout=30000)
    check("still area painted and saved", True)
    page.keyboard.press("Control+z")  # after Xong: the whole painting is one step back, and forward again
    page.wait_for_timeout(400)
    undone = page.evaluate("S.masks.freeze.empty")
    page.keyboard.press("Control+y")
    page.wait_for_function("S.project.masks.includes('freeze') && !S.masks.freeze.empty", timeout=30000)
    check("a painted area: undone and done again with Ctrl+Z / Ctrl+Y", undone is True)
    # outlining a shape: Ctrl+Z takes the last point back (the areas already drawn stay), Ctrl+Y puts it again,
    # Esc drops the unfinished shape, a second Esc ends the painting
    page.click(".clip.base >> nth=0")
    page.click("#paintFreeze")
    page.keyboard.press("l")  # the polygon tool
    box = page.locator("#screen").bounding_box()
    ver = page.evaluate("S.masks.freeze.ver")
    for fx_, fy_ in ((0.45, 0.2), (0.7, 0.3), (0.6, 0.55)):
        page.mouse.click(box["x"] + box["width"] * fx_, box["y"] + box["height"] * fy_)
        page.wait_for_timeout(300)
    n3 = page.evaluate("S.poly.length")
    page.keyboard.press("Control+z")
    n2 = page.evaluate("[S.poly.length, S.masks.freeze.ver === " + str(ver) + ", S.masks.freeze.empty]")
    page.keyboard.press("Control+y")
    n3b = page.evaluate("S.poly.length")
    page.keyboard.press("Escape")
    dropped = page.evaluate("[S.poly.length, !!S.paint]")
    page.keyboard.press("b")  # (the brush again, as the later steps expect)
    page.keyboard.press("Escape")
    out = page.evaluate("!S.paint")
    check("outlining: Ctrl+Z takes the last point back, Ctrl+Y puts it again, Esc drops the shape",
          n3 == 3 and n2 == [2, True, False] and n3b == 3 and dropped == [0, True] and out,
          str([n3, n2, n3b, dropped, out]))
    page.keyboard.press("Control+Equal")  # the picture's zoom, not the page's
    page.wait_for_timeout(200)
    zoomed = page.evaluate("[S.scale, window.devicePixelRatio]")
    page.keyboard.press("Control+0")
    page.wait_for_timeout(200)
    check("Ctrl + / Ctrl 0 zoom the picture", zoomed[0] > 1.2 and page.evaluate("S.zoom") == "fit", str(zoomed))

    # an effect's picture on the monitor: a double-click selects it and leaves it in place; a press and drag moves it
    fid = page.evaluate("S.project.layers.find(c => c.kind === 'file').id")
    params = f"Object.assign({{}}, clipById('{fid}').params)"
    page.evaluate("(() => { S.sel = null; renderControls(); showXform(); seek(16); })()")
    page.wait_for_timeout(300)
    before = page.evaluate(params)
    mid = page.evaluate(f"(() => {{ const q = placed(clipById('{fid}')), sc = screenEl.getBoundingClientRect(); "
                        "return [sc.left + (q.x + q.w / 2) / info().w * sc.width, sc.top + (q.y + q.h / 2) / info().h * sc.height]; })()")
    page.mouse.dblclick(*mid)
    page.wait_for_timeout(300)
    after = page.evaluate(params)
    check("a double-click on an effect's picture selects it", page.evaluate(f"S.sel === '{fid}'") and
          (after["x"], after["y"]) == (before["x"], before["y"]), f"{before} -> {after}")
    page.evaluate("(() => { S.sel = null; renderControls(); showXform(); })()")
    drag(mid[0], mid[1], mid[0] + 60, mid[1])
    after = page.evaluate(params)
    check("a press and drag on an effect's picture moves it", page.evaluate(f"S.sel === '{fid}'") and
          after["x"] > before["x"] + 2 and abs(after["y"] - before["y"]) < 0.5, f"{before} -> {after}")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    check("undo the move", page.evaluate(params)["x"] == before["x"])

    # a corner pulled out: the opposite corner stays where it was (as in Premiere)
    page.evaluate(f"(() => {{ const c = clipById('{fid}'); c.params.scale = 40; select(c.id); edited(); }})()")
    page.wait_for_timeout(300)
    xb = page.locator("#xform").bounding_box()
    page.mouse.move(xb["x"] + xb["width"] - 1, xb["y"] + xb["height"] - 1)  # the bottom-right corner
    page.mouse.down()
    page.mouse.move(xb["x"] + xb["width"] * 1.5, xb["y"] + xb["height"] * 1.5, steps=6)
    page.mouse.up()
    xb2 = page.locator("#xform").bounding_box()
    check("a corner sizes the clip, the opposite corner stays", abs(xb2["x"] - xb["x"]) < 2 and abs(xb2["y"] - xb["y"]) < 2
          and xb2["width"] > xb["width"] * 1.3, f"{xb} -> {xb2}")
    page.click('#ec input[data-param="flip_h"]')
    page.wait_for_timeout(200)
    check("turn it over", page.evaluate(f"clipById('{fid}').params.flip_h") is True and
          "scale(-1" in page.evaluate(f"S.els['{fid}'].media.style.transform"))
    page.keyboard.press("Control+z")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    move_the_picture(page, fid, drag, shot)

    # an outside file's colours: the live preview's filter has the renderer's numbers; "Đặt lại màu" takes it off
    from fx import effects as fx_effects, overlay as fx_overlay
    page.evaluate(f"(() => {{ Object.assign(clipById('{fid}').params, {{contrast: 40, sat: -30, hue: 60, temp: 50, "
                  "tint_amt: 20, bright: -10}); select(S.sel); edited(); })()")
    page.wait_for_timeout(300)
    js = page.evaluate(f"(() => {{ const o = colourOps(clipById('{fid}').params); return [o.curve, o.mat]; }})()")
    curve, mat = fx_overlay.colour_ops(fx_effects.clean_params("file", page.evaluate(f"clipById('{fid}').params")))
    check("colours: the live preview uses the renderer's numbers",
          np.abs(np.array(js[0]) - curve).max() < 1e-9 and np.abs(np.array(js[1]) - mat).max() < 1e-9 and
          "#cc-" in page.evaluate(f"S.els['{fid}'].media.style.filter"))  # (url("#cc-…"))
    page.click("#colourReset")
    page.wait_for_timeout(200)
    check("colours back as the file's", page.evaluate(f"S.els['{fid}'].media.style.filter") == "" and
          page.evaluate(f"clipById('{fid}').params.contrast") == 0)

    # played backwards: its copy is made in the background, then the live preview plays it (and so does the export)
    page.click('#ec input[data-param="reverse"]')
    page.wait_for_function(f"(() => {{ const it = libItem(clipById('{fid}').file.lib); return it && it.reverse; }})()",
                           timeout=180000)
    page.wait_for_timeout(800)
    src = page.evaluate(f"S.els['{fid}'] ? S.els['{fid}'].media.src : ''")
    check("play an effect video backwards", page.evaluate(f"clipById('{fid}').params.reverse") is True and
          "what=reverse" in src, src)
    page.click('#ec input[data-param="reverse"]')  # and forwards again: the preview plays the file itself
    page.wait_for_timeout(500)
    src = page.evaluate(f"S.els['{fid}'] ? S.els['{fid}'].media.src : ''")
    check("and forwards again", "what=proxy" in src, src)
    page.click('#ec input[data-param="reverse"]')
    page.wait_for_timeout(500)

    # an outside file turned in 3D (rain or light laid on a window seen at an angle): the live preview lays it with
    # CSS perspective / rotateX / rotateY / rotateZ on exactly the corners the renderer warps it to
    page.set_input_files("#fileLib", upload(folder, "pattern.png", "Ô kính.png", "image/png"))
    page.wait_for_function("S.lib.length === 3 && S.lib.every(i => i.proxy)", timeout=60000)
    rid = page.evaluate("""(() => {
        const t = {id: newId("t"), on: true}; S.project.tracks.push(t);  // the top track: nothing over it
        const it = S.lib.find(i => i.info.image && i.info.w === 480);
        addClip({lib: it.id}, t.id, 24, 168); const c = clipById(S.sel);
        Object.assign(c.params, {blend: 'normal', scale: 60}); edited(); seek(3); return c.id; })()""")
    page.wait_for_function(f"S.els['{rid}'] && S.els['{rid}'].media.complete", timeout=30000)
    vw, vh = page.evaluate("[info().w, info().h]")
    rinfo = page.evaluate(f"clipById('{rid}').file.info")
    ui = page.evaluate("[$('#xformRot').style.display, $('#xform3d').style.display, "
                       "!!$('#ec input[data-num=\"rot_x\"]'), !!$('#ec input[type=range][data-param=\"rot_z\"]'), !!$('#rotReset')]")
    check("3D: a file clip has the round knob, the 3D handle and the Xoay 3D settings", ui == ["block", "block", True, True, True],
          str(ui))

    def turn(params):
        page.evaluate(f"(() => {{ const c = clipById('{rid}'); Object.assign(c.params, {json.dumps(params)}); edited(); }})()")

    def screen_point(js_point):  # a point of the video (an expression giving [x, y]) on the screen
        return page.evaluate(f"(() => {{ const p = {js_point}, sc = screenEl.getBoundingClientRect(); "
                             "return [sc.left + p[0] / info().w * sc.width, sc.top + p[1] / info().h * sc.height]; })()")
    worst = [0.0, 0.0, 0.0]
    for case in (dict(rot_x=30, rot_y=-25, rot_z=10, persp=50),
                 dict(rot_x=-55, rot_y=40, rot_z=-120, persp=100, flip_h=True),
                 dict(rot_x=20, rot_y=65, rot_z=0, persp=0, x=20, y=-10, scale=140, flip_h=False)):
        turn(case)
        want = fx_overlay.quad(fx_effects.clean_params("file", page.evaluate(f"clipById('{rid}').params")), rinfo, vw, vh)
        shown, js = page.evaluate(CORNERS, rid), page.evaluate(f"quad(clipById('{rid}'))")
        box = page.evaluate("(() => { const r = xform.getBoundingClientRect(), sc = screenEl.getBoundingClientRect(), "
                            "kx = info().w / sc.width, ky = info().h / sc.height; return [(r.left - sc.left) * kx, "
                            "(r.top - sc.top) * ky, (r.right - sc.left) * kx, (r.bottom - sc.top) * ky]; })()")
        around = [min(x for x, _ in want), min(y for _, y in want), max(x for x, _ in want), max(y for _, y in want)]
        worst = [max(worst[0], max(math.dist(a, b) for a, b in zip(shown, want))),
                 max(worst[1], max(math.dist(a, b) for a, b in zip(js, want))),
                 max(worst[2], max(abs(a - b) for a, b in zip(box, around)))]
    check("3D: the live preview's corners are the renderer's (within half a pixel)", worst[0] < 0.5 and worst[1] < 1e-6,
          f"browser vs renderer {worst[0]:.3f} px, page's quad() vs renderer {worst[1]:.1e} px")
    check("3D: the box on the picture turns with the clip", worst[2] < 1.0, f"{worst[2]:.3f} px")

    turn(dict(rot_x=0, rot_y=0, rot_z=0, persp=50, x=0, y=0, scale=60))
    page.evaluate(f"select('{rid}')")
    kb = page.locator("#xformRot").bounding_box()
    mid = screen_point(f"(q => [q.x + q.w / 2, q.y + q.h / 2])(placed(clipById('{rid}')))")
    kx, ky = kb["x"] + kb["width"] / 2, kb["y"] + kb["height"] / 2
    page.mouse.move(kx, ky)
    page.mouse.down()
    for k in range(1, 7):  # round the middle, 30 degrees clockwise
        a = math.radians(30) * k / 6
        page.mouse.move(mid[0] + (kx - mid[0]) * math.cos(a) - (ky - mid[1]) * math.sin(a),
                        mid[1] + (kx - mid[0]) * math.sin(a) + (ky - mid[1]) * math.cos(a))
    page.mouse.up()
    page.wait_for_timeout(200)
    z = page.evaluate(f"clipById('{rid}').params.rot_z")
    check("3D: the round knob turns the clip in the plane", abs(z - 30) < 1.5 and
          "rotateZ(" in page.evaluate(f"S.els['{rid}'].media.style.transform"), str(z))
    hb = page.locator("#xform3d").bounding_box()
    hx, hy = hb["x"] + hb["width"] / 2, hb["y"] + hb["height"] / 2
    drag(hx, hy, hx + 100, hy)
    tilt = page.evaluate(f"(p => [p.rot_x, p.rot_y, p.rot_z])(clipById('{rid}').params)")
    check("3D: dragging \"3D\" sideways turns it left / right", abs(tilt[1] - 30) < 1 and tilt[0] == 0 and tilt[2] == z,
          str(tilt))
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    check("3D: undo takes the tilt back", page.evaluate(f"(p => [p.rot_y, p.rot_z])(clipById('{rid}').params)") == [0, z])

    # a double-click on the knob turns it back to 0 degrees in the plane; the clip stays where it is
    turn(dict(rot_z=30, x=10, y=5))
    page.evaluate(f"select('{rid}')")
    page.wait_for_timeout(200)
    kb = page.locator("#xformRot").bounding_box()
    page.mouse.dblclick(kb["x"] + kb["width"] / 2, kb["y"] + kb["height"] / 2)
    page.wait_for_timeout(300)
    got = page.evaluate(f"(p => [p.x, p.y, p.rot_z])(clipById('{rid}').params)")
    check("3D: a double-click on the round knob: back to 0°, the clip stays put", got == [10, 5, 0], str(got))

    # the name tag covers neither the knob (with its stem) nor "3D", small or turned
    hit = """(() => { const t = xformLab.getBoundingClientRect(), k = xformRot.getBoundingClientRect(), h = xform3d.getBoundingClientRect();
        const on = (p, q) => Math.min(p.right, q.right) > Math.max(p.left, q.left) && Math.min(p.bottom, q.bottom) > Math.max(p.top, q.top);
        return [on(t, k), on(t, h), on(k, h)]; })()"""
    covered = []
    for case in (dict(rot_z=0, x=0, y=0, scale=20), dict(scale=30), dict(rot_z=20, scale=60), dict(rot_z=180, scale=30)):
        turn(case)
        page.evaluate(f"select('{rid}')")
        covered.append(page.evaluate(hit))
    check("3D: the name tag hides neither the knob nor \"3D\"", all(c == [False, False, False] for c in covered), str(covered))
    # the name tag only while the box is dragged (at rest it would hide what the clip is being lined up with)
    rest = page.evaluate("xformLab.style.display")
    xb = page.locator("#xform").bounding_box()
    page.mouse.move(xb["x"] + xb["width"] * 0.3, xb["y"] + xb["height"] * 0.55)
    page.mouse.down()
    page.mouse.move(xb["x"] + xb["width"] * 0.3 + 30, xb["y"] + xb["height"] * 0.55, steps=4)
    during = page.evaluate("[xformLab.style.display, xformLab.textContent.length > 0]")
    page.mouse.up()
    page.wait_for_timeout(200)
    after = page.evaluate("xformLab.style.display")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    check("the clip's name tag shows only while its box is dragged", rest == "none" and during == ["block", True]
          and after == "none", str([rest, during, after]))

    # "3D" is not on the clip's middle: a click there selects the clip, a drag from the same point moves it
    turn(dict(rot_x=0, rot_y=0, rot_z=0, x=0, y=0, scale=60))
    page.evaluate("(() => { S.sel = null; renderControls(); showXform(); })()")
    mid = screen_point(f"(q => [q.x + q.w / 2, q.y + q.h / 2])(placed(clipById('{rid}')))")
    page.mouse.click(*mid)
    page.wait_for_timeout(300)
    picked = page.evaluate("S.sel")
    drag(mid[0], mid[1], mid[0] + 80, mid[1])
    moved = page.evaluate(f"(p => [p.x, p.rot_x, p.rot_y])(clipById('{rid}').params)")
    check("3D: dragging from the clip's middle moves it (no tilt)", picked == rid and moved[0] > 5 and moved[1:] == [0, 0],
          f"{picked}, {moved}")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)

    # "3D" stays on the picture while a small clip is tilted to edge-on, and after it
    turn(dict(scale=30))
    page.evaluate(f"select('{rid}')")
    hb = page.locator("#xform3d").bounding_box()
    hx, hy = hb["x"] + hb["width"] / 2, hb["y"] + hb["height"] / 2
    page.mouse.move(hx, hy)
    page.mouse.down()
    shown = []
    for k in range(1, 9):
        page.mouse.move(hx + 40 * k, hy)
        shown.append(page.evaluate("xform3d.style.display"))
    page.mouse.up()
    page.wait_for_timeout(300)
    after = page.evaluate(f"[clipById('{rid}').params.rot_y, xform3d.style.display]")
    check("3D: \"3D\" stays on the picture when tilted edge-on", set(shown) == {"block"} and after == [80, "block"],
          f"{shown}, {after}")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    turn(dict(scale=60))

    # turned, its picture is picked where it is now, not where it was upright
    turn(dict(rot_z=45))
    page.evaluate("(() => { S.sel = null; renderControls(); showXform(); })()")
    on = screen_point(f"(q => {{ const c = q.reduce((a, p) => [a[0] + p[0] / 4, a[1] + p[1] / 4], [0, 0]), "
                      "t = q.reduce((a, p) => p[1] < a[1] ? p : a); return [c[0] + 0.85 * (t[0] - c[0]), "
                      f"c[1] + 0.85 * (t[1] - c[1])]; }})(quad(clipById('{rid}')))")
    # near the upright box's corner: off the picture turned 45 degrees
    off = screen_point(f"(q => [q.x + 0.03 * q.w, q.y + 0.03 * q.h])(placed(clipById('{rid}')))")
    check("3D: (that point is off the turned picture)", page.evaluate(
        f"(() => {{ const r = placed(clipById('{rid}')); return !hitsClip(clipById('{rid}'), r.x + 0.03 * r.w, r.y + 0.03 * r.h); }})()"))
    page.mouse.click(*off)
    page.wait_for_timeout(200)
    missed = page.evaluate("S.sel")
    page.mouse.dblclick(*on)
    page.wait_for_timeout(300)
    # (the snow over the whole video may be picked there: an effect's sheet covers the frame; never the turned clip)
    check("3D: a click picks a turned clip by its turned picture", missed != rid and page.evaluate("S.sel") == rid and
          page.evaluate(f"[clipById('{rid}').params.x, clipById('{rid}').params.y]") == [0, 0], f"{missed}, {page.evaluate('S.sel')}")
    page.fill('#ec input[data-num="rot_x"]', "12.5")
    page.keyboard.press("Enter")
    page.wait_for_timeout(200)
    typed = page.evaluate(f"[clipById('{rid}').params.rot_x, +$('#ec input[type=range][data-param=\"rot_x\"]').value, "
                          f"S.els['{rid}'].media.style.transform]")
    check("3D: a number typed in the panel", typed[:2] == [12.5, 12.5] and "rotateX(12.5deg)" in typed[2], str(typed))
    kept = page.evaluate("document.activeElement === $('#ec input[data-num=\"rot_x\"]')")
    page.click('#ec input[data-num="rot_x"]')
    page.keyboard.press("Control+a")
    page.keyboard.type("40")  # key by key: the field's own undo would take back the "0" alone (a turn of 4 degrees)
    page.wait_for_timeout(100)
    typing = page.evaluate(f"clipById('{rid}').params.rot_x")
    page.keyboard.press("Control+z")  # in the field: the app's undo, not the field's own text undo
    page.wait_for_timeout(300)
    back = page.evaluate(f"[clipById('{rid}').params.rot_x, $('#ec input[data-num=\"rot_x\"]').value, "
                         "document.activeElement.tagName]")
    check("3D: Enter leaves the number field, Ctrl+Z in it is the app's undo", not kept and typing == 40 and
          back == [12.5, "12.5", "BODY"], f"{kept}, {typing}, {back}")
    page.evaluate("document.activeElement.blur()")

    # the picture itself: the live preview and the exact frame agree inside the turned clip
    turn(dict(rot_x=30, rot_y=-25, rot_z=10, persp=50))
    page.evaluate("(async () => { S.sel = null; renderControls(); showXform(); await saveNow(); })()")
    page.wait_for_timeout(500)
    page.evaluate("exact.style.visibility = 'hidden'")  # the live preview alone
    page.wait_for_timeout(300)
    live_png = page.locator("#screen").screenshot()
    page.evaluate("(() => { if (!S.exactOn) $('#tExact').click(); invalidateExact(); refreshExact(); })()")
    page.wait_for_function("exact.style.display === 'block' && exact.complete && exact.src.includes('rev=' + S.project.rev)",
                           timeout=60000)
    page.evaluate("exact.style.visibility = ''")
    page.wait_for_timeout(300)
    exact_png = page.locator("#screen").screenshot()
    with open(os.path.join(SHOTS, "05a-turned-live.png"), "wb") as f:
        f.write(live_png)
    with open(os.path.join(SHOTS, "05a-turned-exact.png"), "wb") as f:
        f.write(exact_png)
    parity = picture_parity(page, rid, vw, vh, rinfo, live_png, exact_png)
    check("3D: the live preview's picture is the exact frame's", parity["soft luma mean"] < 4 and parity["rgb mean"] < 14,
          str(parity))
    page.evaluate(f"select('{rid}')")
    page.click("#rotReset")
    page.wait_for_timeout(200)
    flat = page.evaluate(f"[clipById('{rid}').params.rot_x, clipById('{rid}').params.rot_y, clipById('{rid}').params.rot_z, "
                         f"S.els['{rid}'].media.style.transform]")
    check("3D: \"Đặt lại xoay\" sets it straight", flat == [0, 0, 0, ""], str(flat))
    page.focus('#ec input[type=range][data-param="rot_y"]')
    for _ in range(5):
        page.keyboard.press("ArrowRight")
    page.wait_for_timeout(200)
    tools = page.evaluate("[$('#tools [data-rotreset]').disabled, [...$$('#tools .lab')].map(e => e.textContent).join(' | ')]")
    check("3D: the tools column follows the panel's sliders", tools[0] is False and "0°, 0.5°, 0°" in tools[1], str(tools))
    page.keyboard.press("Control+z")  # the slider, then "Đặt lại xoay": turned again, for the export
    page.wait_for_timeout(200)
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    check("3D: undo turns it again", page.evaluate(f"turned(clipById('{rid}'))") is True)
    effects_turned(page, drag)
    shot("05g-effects-turned")

    # a Mix where two clips meet: the button on the join adds it; the page and the renderer weigh it alike
    ids = page.evaluate("""(() => {
        const t = {id: newId("t"), on: true}; S.project.tracks.push(t);
        addClip({kind: 'rain'}, t.id, 300, 400); const a = S.sel; addClip({kind: 'rain'}, t.id, 400, 500);
        return [a, S.sel]; })()""")
    page.wait_for_timeout(300)
    jb = page.locator(f'[data-join="{ids[1]}"]').bounding_box()  # a press on the join that moves: a trim, no Mix
    drag(jb["x"] + 3, jb["y"] + jb["height"] / 2, jb["x"] + 3 - 4 * page.evaluate("S.pps"), jb["y"] + jb["height"] / 2)
    trimmed = page.evaluate(f"[clipById('{ids[0]}').end, clipById('{ids[1]}').mix || 0]")
    check("a press and drag on a join trims the clip on that side", trimmed[0] < 400 and trimmed[1] == 0, str(trimmed))
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    page.click(f'[data-join="{ids[1]}"]')
    page.wait_for_timeout(200)
    shown = page.evaluate("clipOrder().map(c => Object.assign({}, c))")
    mixes = timeline.mixes(shown, 720, 24)
    want = {c["id"]: [timeline.Span(c, 720, 24, mixes.get(c["id"], (0.0, 0.0))).weight(t) for t in range(380, 420)]
            for c in shown if c["id"] in ids}
    got = page.evaluate(f"Object.fromEntries({ids}.map(id => [id, Array.from({{length: 40}}, (_, i) => weight(span(clipById(id)), 380 + i))]))")
    same = all(abs(got[i][k] - want[i][k]) < 1e-9 for i in ids for k in range(40))
    check("a Mix where two clips meet, weighed alike on the page and in the renderer",
          page.evaluate(f"clipById('{ids[1]}').mix") == 0.5 and same and
          all(abs(got[ids[0]][k] + got[ids[1]][k] - 1) < 1e-9 for k in range(14, 26)), str(mixes))
    roles = page.evaluate(f"(() => {{ mixGroups({{'{ids[0]}': 0.5, '{ids[1]}': 0.5}}); const b = clipById('{ids[1]}');"
                          f" return [S.mixRole['{ids[0]}'], S.mixRole['{ids[1]}'], S.trackEls[b.track].style.isolation]; }})()")
    check("in a Mix the two clips are laid as one picture", roles == ["normal", "plus-lighter", "isolate"], str(roles))
    page.evaluate(f"deleteClip(clipById('{ids[0]}'))")
    check("deleting one clip of a Mix takes the Mix away", page.evaluate(f"clipById('{ids[1]}').mix") == 0)
    page.evaluate(f"(() => {{ for (const id of {ids}) if (clipById(id)) deleteClip(clipById(id)); }})()")

    # the other half of a cut fades in again once the first half is deleted
    r = page.evaluate("""(() => {
        const t = {id: newId("t"), on: true}; S.project.tracks.push(t);
        addClip({kind: 'rain'}, t.id, 100, 300); const a = clipById(S.sel); splitClip(a, 200); const b = clipById(S.sel);
        const cut = [a.fade_out, b.fade_in]; deleteClip(a); const back = b.fade_in; deleteClip(b);
        S.project.tracks = S.project.tracks.filter(x => x !== t); edited(); return [cut, back]; })()""")
    check("deleting one half of a cut: the other half fades in again", r[0] == [0, 0] and r[1] > 0, str(r))
    # 60 s -> 30 s -> 60 s: a clip that only reached the end of the 30 s video keeps its length (saved with it)
    r = page.evaluate("""(async () => {
        const p = S.project, was = p.length; setLength(60);
        const t = {id: newId("t"), on: true}; p.tracks.push(t);
        addClip({kind: 'rain'}, t.id, 0, 720); const c = clipById(S.sel);
        setLength(30); await saveNow();
        const saved = (await get('/api/project?id=' + p.id)).layers.find(x => x.id === c.id).short;
        setLength(60); const kept = [c.start, c.end];
        deleteClip(c); p.tracks = p.tracks.filter(x => x !== t); setLength(was); return [saved, kept]; })()""")
    check("60 s -> 30 s -> 60 s keeps a clip that only reached the 30 s end", r == [True, [0, 720]], str(r))
    page.wait_for_timeout(200)
    # rain lit by the scene, moved or turned: its preview clip has the light of where it stood, so a pause shows the
    # exact frame (as for a lens)
    r = page.evaluate("""(() => {
        const t = {id: newId("t"), on: true}; S.project.tracks.push(t);
        addClip({kind: 'rain'}, t.id, 0, 240); const c = clipById(S.sel), pr = c.params, at = [litMoved(c)];
        pr.x = -20; at.push(litMoved(c)); pr.lit = 0; at.push(litMoved(c));
        pr.x = 0; pr.lit = 70; pr.rot_y = 20; at.push(litMoved(c));
        deleteClip(c); S.project.tracks = S.project.tracks.filter(x => x !== t); edited(); return at; })()""")
    check("lit rain moved or turned: the live preview is only close, a pause shows the exact frame",
          r == [False, True, False, True], str(r))

    # a cup's steam and a candle's flame sit in a box on the picture, with a yellow mark where the cup's mouth or the
    # wick goes. Steam's preview clip is its box alone, lit by the light where the box stands (moving the box makes a
    # new one, the old one shows meanwhile); the candle's is made from the picture under its box (moving the box makes
    # a new one, the old one hides). Both stay on for the export.
    page.click('.tab[data-bin="fx"]')
    page.evaluate("seek(2)")
    for kind, mark in (("steam", "xmark dot"), ("candle", "xmark")):
        page.dblclick(f'.tile[data-kind="{kind}"]')
        cid = page.evaluate(f"S.project.layers.find(l => l.kind === '{kind}').id")
        page.click("#fullBtn")
        page.wait_for_timeout(300)
        mk = page.evaluate("[$('#xform').style.display, $('#xformMark').style.display, $('#xformMark').className, "
                           "$('#xformTip').style.display, $('#xformMark').dataset.lab]")
        tip = "block" if kind == "candle" else "none"  # (the candle: the wick, and the flame's tip)
        check(f"{kind}: its box on the picture, with the yellow marks", mk[:4] == ["block", "block", mark, tip]
              and bool(mk[4]), str(mk))
        if kind == "steam":  # the place it rises from: an ellipse sitting inside the box on its bottom edge
            sits = page.evaluate("(() => { const m = $('#xformMark').getBoundingClientRect(), "
                                 "b = $('#xform').getBoundingClientRect(); return [m.bottom - b.bottom, "
                                 "m.height / m.width, b.bottom - m.top]; })()")
            check("steam: its mark sits inside the box on its bottom edge", abs(sits[0]) < 3 and 0.15 < sits[1] < 0.4
                  and sits[2] > 4, str(sits))
            styles = page.evaluate("[...$('#ec select[data-param=\"style\"]').options].map(o => o.value)")
            check("steam: its styles to choose from (smoke from a chimney: Khói ống khói)",
                  styles == ["cup", "incense", "pot"], str(styles))
            shot("05b-steam")
        spin = page.evaluate("[$('#xformRot').style.display, $('#xform3d').style.display, !!$('#ec [data-num=\"rot_x\"]')]")
        if kind == "steam":  # turned as a file clip, its box the sheet; a flame always rises straight up
            window = "ô cửa nhìn chéo" in page.evaluate("$('#ec').textContent")  # (an example for full-frame effects)
            check("steam: turned in 3D (the knob, the 3D handle, its settings; no window example in its note)",
                  spin == ["block", "block", True] and not window, str([spin, window]))
        else:
            check("candle: not turned in 3D (no knob, no 3D handle, no settings for it)", spin == ["none", "none", False]
                  and "không xoay 3D" in page.evaluate("$('#ec').textContent"), str(spin))
        page.wait_for_function(f"S.fx['{cid}'] && S.fx['{cid}'].ready && S.els['{cid}']", timeout=300000)
        if kind == "steam":  # turned, its preview clip (the box) lies on the renderer's corners
            vw, vh = page.evaluate("[info().w, info().h]")
            page.evaluate(f"(() => {{ Object.assign(clipById('{cid}').params, {{rot_x: 20, rot_y: 35, rot_z: 8}}); "
                          "edited(); })()")
            off = max(math.dist(p, q) for p, q in zip(page.evaluate(CORNERS, cid), clip_quad(page, cid, vw, vh)))
            check("steam: turned, its box on the renderer's corners", off < 0.5, f"{off:.3f} px")
            page.evaluate(f"(() => {{ Object.assign(clipById('{cid}').params, {{rot_x: 0, rot_y: 0, rot_z: 0}}); "
                          "edited(); })()")
        laid = f"""(() => {{ const c = clipById('{cid}'), e = S.els['{cid}'], q = placed(c);
            return [e.media.getAttribute('src'), e.wrap.style.mixBlendMode,
                    Math.abs(parseFloat(e.media.style.left) - 100 * q.x / info().w) < 0.01]; }})()"""
        el = page.evaluate(laid)
        check(f"{kind}: its preview clip, laid in the box", ".webm" in el[0] and el[1:] == ["normal", True], str(el))
        before = page.evaluate(f"[S.fx['{cid}'].shown, clipById('{cid}').params.x]")
        xb = page.locator("#xform").bounding_box()
        cx0, cy0 = xb["x"] + xb["width"] / 2, xb["y"] + xb["height"] / 2
        page.mouse.move(cx0, cy0)
        page.mouse.down()
        page.mouse.move(cx0 + 20, cy0, steps=4)
        mid = page.evaluate(f"S.els['{cid}'] ? S.els['{cid}'].wrap.style.display : 'gone'")
        # (steam: well over, into other light; a nudge keeps its clip, lit the same: test_steam)
        page.mouse.move(cx0 + (40 if kind == "candle" else 160), cy0, steps=8)
        page.mouse.up()
        page.wait_for_timeout(300)
        check(f"{kind}: while its box moves, its preview clip " +
              ("hides (made from the picture under the box: the video shows)" if kind == "candle" else "moves with it"),
              mid == ("none" if kind == "candle" else ""), mid)
        page.evaluate("saveNow()")
        page.wait_for_timeout(1500)
        try:  # (asked for once the app has the new place: both are made for where they stand)
            page.wait_for_function(f"S.fx['{cid}'].want !== {json.dumps(before[0])}", timeout=20000)
        except Exception:
            pass
        after = page.evaluate(f"[S.fx['{cid}'].want, clipById('{cid}').params.x]")
        made_again = after[0] != before[0]
        check(f"{kind}: dragging its box moves it, with a new preview clip" +
              (" (the steam is lit by the light where it stands)" if kind == "steam" else ""),
              after[1] > before[1] + 1 and made_again, f"{before} -> {after}")
        if kind == "steam":  # the old clip goes on showing, in the box, until the new one is made
            shown = page.evaluate(f"[S.fx['{cid}'].shown === {json.dumps(before[0])} || S.fx['{cid}'].ready, "
                                  f"S.els['{cid}'].wrap.style.display]")
            check("steam: its old preview clip shows in the box until the new one is made", shown == [True, ""],
                  str(shown))
        if kind == "candle":  # the old clip is not laid at the new place; the new one is, once made
            hidden = page.evaluate(f"S.fx['{cid}'].ready || S.els['{cid}'].wrap.style.display === 'none'")
            page.wait_for_function(f"S.fx['{cid}'].ready && S.fx['{cid}'].shown === S.fx['{cid}'].want", timeout=300000)
            page.evaluate("applyTime()")
            el = page.evaluate(laid + ".concat([S.els['" + cid + "'].wrap.style.display])")
            check("candle: its old preview clip hides until the new one is made, then that one shows in the box",
                  hidden and el[2] is True and el[3] == "", str([hidden] + el))
            under = page.evaluate(f"S.els['{cid}'].wrap.parentNode === $('#layers').firstChild")
            check("candle: laid just over the video and its still area (under every track)", under)
            wick = "(r => [r.left + r.width / 2, r.top])($('#xformMark').getBoundingClientRect())"
            w0 = page.evaluate(wick)
            xb = page.locator("#xform").bounding_box()
            drag(xb["x"] + 1, xb["y"] + 1, xb["x"] - 30, xb["y"] - 40)  # the top left corner, pulled out
            w1, xb1 = page.evaluate(wick), page.locator("#xform").bounding_box()
            check("candle: a corner sizes its box, the wick mark stays put", xb1["height"] > xb["height"] * 1.2 and
                  abs(w1[0] - w0[0]) < 1.5 and abs(w1[1] - w0[1]) < 1.5, f"{w0} -> {w1}, {xb['height']} -> {xb1['height']}")
            page.keyboard.press("Control+z")
            page.wait_for_timeout(200)
    shot("05b-steam-candle")
    foliage(page, shot)
    winter(page, shot, edge)
    garland(page, shot)
    chimneys(page, shot)

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
    match_and_copy(page, shot, lane_y, x_of)
    save_project(page, shot)
    paste_elsewhere(page, lane_y, x_of)
    check("no script errors on the page", not errors, "\n".join(errors[:10]))


def match_and_copy(page, shot, lane_y, x_of):
    """Khớp màu với cảnh: the app sets a file clip's colour sliders, Ctrl+Z takes them back. Copies of a clip: on
    another track at the same time ("Chép sang track…", with its area), at the playhead on the track under the mouse
    (Ctrl+C, Ctrl+V), and where a track is right-clicked."""
    keys = ["bright", "contrast", "sat", "hue", "temp", "tint_amt", "tint"]
    fid = page.evaluate("S.project.layers.find(l => l.kind === 'file').id")
    page.evaluate(f"select('{fid}')")
    before = page.evaluate(f"JSON.stringify(clipById('{fid}').params)")
    page.click("#colourMatch")
    page.wait_for_function("document.querySelector('#toast').textContent.includes('Đã khớp màu với cảnh')", timeout=60000)
    got = page.evaluate(f"clipById('{fid}').params")
    shown = page.evaluate("Object.fromEntries([...document.querySelectorAll('#ec input[data-param]')]"
                          ".map(i => [i.dataset.param, i.value]))")
    check("Khớp màu với cảnh: the colour sliders set, and shown", all(k in got for k in keys) and
          all(str(got[k]) == shown.get(k) for k in keys if k != "tint") and got["tint"] == shown.get("tint"),
          f"{got} / {shown}")
    shot("09-colour-match")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(300)
    check("Khớp màu với cảnh: undone in one step", page.evaluate(f"JSON.stringify(clipById('{fid}').params)") == before)

    # Dán thiết lập (Paste Attributes): this clip's colours and size onto another file clip, in one undo step; onto
    # a clip of another kind only what both have
    page.evaluate(f"(() => {{ const c = clipById('{fid}'); c.params.bright = 30; c.params.scale = 60; edited(); select(c.id); }})()")
    page.keyboard.press("Control+c")
    other = page.evaluate("(() => { addClip({lib: S.lib[0].id}, null, 0, 48); return S.sel; })()")
    page.keyboard.press("Control+Alt+v")
    page.wait_for_selector("#modalBox .pa")
    names = page.evaluate("[...document.querySelectorAll('#modalBox .groups label')].map(l => l.textContent.trim())")
    shot("09b-paste-attributes")
    page.click("#mGo")
    got = page.evaluate(f"(() => {{ const b = clipById('{other}').params; return [b.bright, b.scale, b.start]; }})()")
    check("Dán thiết lập: the colours and size of one clip onto another", got == [30, 60, 0] and
          names[:3] == ["Màu sắc", "Vị trí và cỡ", "Xoay 3D"], f"{got}, {names}")
    page.keyboard.press("Control+z")
    page.wait_for_timeout(200)
    got = page.evaluate(f"(() => {{ const b = clipById('{other}').params; return [b.bright, b.scale]; }})()")
    check("Dán thiết lập: undone in one step", got == [0, 100], str(got))
    page.evaluate("select(S.project.layers.find(l => l.kind === 'snow').id)")
    page.keyboard.press("Control+Alt+v")
    page.wait_for_selector("#modalBox .pa")
    names = page.evaluate("[...document.querySelectorAll('#modalBox .groups label')].map(l => l.textContent.trim())")
    check("Dán thiết lập onto another kind: only what both have",
          "Vị trí và cỡ" in names and "Xoay 3D" in names and "Màu sắc" not in names and "Mờ dần" in names, str(names))
    page.click("#mCancel")

    fps = page.evaluate("fps()")
    painted = page.evaluate("(S.project.layers.find(l => S.masks[l.id] && !S.masks[l.id].full) || {}).id || null")
    src = painted or page.evaluate("S.project.layers.find(l => l.kind === 'snow').id")
    page.evaluate(f"select('{src}')")
    tracks = page.evaluate("S.project.tracks.length")
    page.click("#copyToBtn")
    page.click("#menu >> text=Track mới")
    page.wait_for_timeout(200)
    got = page.evaluate(f"""(() => {{
        const a = clipById('{src}'), b = clipById(S.sel), ma = S.masks[a.id], mb = S.masks[b.id];
        const px = m => mctx(m).getImageData(0, 0, m.canvas.width, m.canvas.height).data;
        const same = (x, y) => x.length === y.length && x.every((v, i) => v === y[i]);
        return {{other: b.id !== a.id && b.track !== a.track, tracks: S.project.tracks.length, kind: b.kind === a.kind,
                 time: b.start === a.start && b.end === a.end, params: JSON.stringify(b.params) === JSON.stringify(a.params),
                 area: !ma || ma.full ? mb.full : !mb.full && same(px(ma), px(mb))}};
    }})()""")
    check("Chép sang track: the same clip at the same time on a new track, with its area" if painted else
          "Chép sang track: the same clip at the same time on a new track",
          got["other"] and got["kind"] and got["time"] and got["params"] and got["area"] and got["tracks"] == tracks + 1,
          str(got))

    page.evaluate(f"select('{src}')")
    page.keyboard.press("Control+c")
    page.evaluate("seek(3)")
    page.mouse.move(x_of(8), lane_y(0))  # the mouse on the row that makes a new track
    page.keyboard.press("Control+v")
    page.wait_for_timeout(200)
    got = page.evaluate(f"(() => {{ const b = clipById(S.sel); return {{id: b.id, kind: b.kind, start: b.start, "
                        f"tracks: S.project.tracks.length, name: b.name === clipById('{src}').name}}; }})()")
    want = 0 if page.evaluate(f"span(clipById('{src}'), false).full") else round(3 * fps)
    check("Ctrl+C, Ctrl+V: a copy at the playhead, on the track under the mouse", got["id"] != src and
          got["name"] and got["start"] == want and got["tracks"] == tracks + 2, f"{got}, want start {want}")

    page.mouse.click(x_of(1), lane_y(0), button="right")
    page.click("#menu >> text=vào đây")
    page.wait_for_timeout(200)
    got = page.evaluate("(() => { const b = clipById(S.sel); return {start: b.start, tracks: S.project.tracks.length}; })()")
    check("right-click on a track: Dán vào đây", got["tracks"] == tracks + 3 and
          (want == 0 or abs(got["start"] - fps) <= 0.3 * fps), str(got))
    shot("10-copies")


def paste_elsewhere(page, lane_y, x_of):
    """A clip copied in one project pastes into another (opened after)."""
    page.evaluate("select(S.project.layers[0].id)")
    page.keyboard.press("Control+c")
    other = page.evaluate("S.state.projects.find(p => p.id !== S.project.id).id")
    page.evaluate(f"openProject('{other}')")
    page.wait_for_function(f"S.project && S.project.id === '{other}' && S.project.masks", timeout=60000)
    n = page.evaluate("S.project.layers.length")
    page.evaluate("seek(2)")
    page.mouse.move(x_of(8), lane_y(0))
    page.keyboard.press("Control+v")
    page.wait_for_timeout(200)
    pasted = page.evaluate("S.sel")
    page.evaluate("saveNow()")
    page.wait_for_timeout(500)
    kept = page.evaluate(f"(async () => (await get('/api/project?id={other}')).layers.map(c => c.id))()")
    check("a clip copied in one project pastes into another", page.evaluate("S.project.layers.length") == n + 1 and
          pasted in kept, f"{pasted} in {kept}")
    page.keyboard.press("Control+z")  # (the project goes on as it was)
    page.wait_for_timeout(200)


def save_project(page, shot):
    """Lưu (Ctrl+S): the state by the button says saved; "Lưu thành bản khác": the work goes on in a copy, the old
    project stays as it was."""
    page.wait_for_function("document.querySelector('#saveState').textContent.startsWith('Đã lưu')", timeout=30000)
    page.evaluate("(() => { S.project.layers[0].name = 'Lưu thử 1'; edited(); })()")  # a change...
    page.keyboard.press("Control+s")  # ...saved at once
    page.wait_for_function("document.querySelector('#toast').textContent.includes('Đã lưu dự án')", timeout=30000)
    saved = page.evaluate("get('/api/project?id=' + S.project.id)")
    check("Ctrl+S saves the project", saved["layers"][0]["name"] == "Lưu thử 1" and
          page.inner_text("#saveState").startswith("Đã lưu lúc"), page.inner_text("#saveState"))
    old = page.evaluate("({id: S.project.id, name: S.project.name, layers: S.project.layers.length, "
                        "masks: S.project.masks.slice().sort()})")
    page.once("dialog", lambda d: d.accept(old["name"] + " - bản 2"))
    page.click("#saveMore")
    page.click("#menu >> text=Lưu thành bản khác")
    page.wait_for_function(f"S.project && S.project.id !== '{old['id']}' && S.project.masks", timeout=60000)
    now = page.evaluate("({id: S.project.id, name: S.project.name, layers: S.project.layers.length, "
                        "masks: S.project.masks.slice().sort()})")
    check("save as another project: the work goes on in the copy", now["name"] == old["name"] + " - bản 2" and
          now["layers"] == old["layers"] and now["masks"] == old["masks"], f"{old} -> {now}")
    page.evaluate("(() => { S.project.layers[0].name = 'Lưu thử 2'; edited(); })()")
    page.keyboard.press("Control+s")
    page.wait_for_function("document.querySelector('#toast').textContent.includes('Đã lưu dự án')", timeout=30000)
    kept = page.evaluate(f"get('/api/project?id={old['id']}')")
    names = page.evaluate("(S.state.projects || []).map(p => p.name)")
    check("the old project stays as it was", kept["layers"][0]["name"] == "Lưu thử 1" and
          old["name"] in names, str(names))
    page.click('.tab[data-bin="proj"]')
    page.wait_for_function(f"document.querySelectorAll('#binBody [data-open]').length >= 2", timeout=15000)
    shot("08-saved-copy")


def main():
    sys.stdout.reconfigure(errors="backslashreplace")  # a Windows console cannot print every Vietnamese letter
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
