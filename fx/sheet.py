"""A built-in effect placed and turned in 3D as a clip from a file is (overlay.py: homography(), the numbers the page's
CSS transform uses too).

A full-frame effect (snow, rain, leaves, light, water on glass) is drawn as always, on a flat sheet of the frame's
shape. The sheet is placed as a file clip of the frame's size with fit "contain" (x, y, scale: at 100 and in the
middle it is exactly the frame) and turned about its middle. A placed effect (steam) draws in its box, which is its
sheet. An effect neither moved, sized nor turned (a placed one: not turned) does not come here at all: it is drawn as
it always was, byte for byte (effects.on_sheet). The candle never turns: a flame rises straight up.

The sheet is drawn at the size the warp needs, as a file clip's part is read: never finer than a pixel of it to a
frame pixel where the sheet shows largest, never finer than the effect drawn at its placed size (so a small or far
sheet is cheap), and never more than SHEET_MOST frames of pixels. Then it is warped onto the frame (Pillow PERSPECTIVE,
bilinear: quick, and the effects are soft), see-through outside its quad with a soft outline (coverage()), and laid
as a file clip is (overlay.lay): with the effect's blend (Screen for light, Normal for the others), its fades, and
its area and foreground masks in frame space, after the warp. What would be cut at the sheet's edge is cut at the
turned edge. The turn does not move, so the effect loops as it does upright.

An effect lit by the scene (rain: Effect.LIT) gets the scene's light as it falls on the sheet (_lit()): a turned drop
still shows the light of the place on the frame it is drawn over.

A lens (water on glass) changes the picture under it. The picture under the sheet is brought onto the sheet (the inverse
warp; past the frame's edge, its edge repeated, as a blur at the frame's edge does) and the effect applied there. It
makes each pixel kept * itself + what it adds (the blurred scene, the veil, what a drop sees); a lens that turns says
how much it keeps (apply(kept=)). So its change is a layer over the picture: what it adds is premultiplied colour, what
it takes away (1 - kept) coverage. That layer is warped and laid on the frame's own pixels as a drawn effect is
(Normal): frame * kept + added. Where the glass fogs and blurs the scene, the frame's fine detail goes with it, as
upright (adding only the change, after - before, would bring back, crisp through the fog, the detail the trip onto the
sheet and back loses); where it changes nothing (kept 1, nothing added), and outside the sheet, the frame's pixels stay
exactly as they were; where the glass is clear the frame keeps its full detail (laying the effect's whole picture back
would resample, and soften, it). The live preview stays approximate for a lens, as upright: its preview clip is only
what it lays over the picture.

The sheets of a scene are drawn one after another, so they share their scratch memory (Scratch: one canvas, one
lens buffer, at the largest size asked for), and with it the scene's own canvas for upright effects.
"""
import math

import numpy as np
from PIL import Image, ImageChops

from . import effects, overlay

SHEET_MOST = 2.25  # a sheet holds at most this many frames of pixels (one made far bigger than the frame: coarser)
SMALLEST = 16  # rows of the smallest sheet drawn (one far away, a few pixels on the frame)
PERSPECTIVE, BILINEAR = Image.Transform.PERSPECTIVE, Image.Resampling.BILINEAR


class Scratch:
    """Memory the sheets of one scene share, as they are drawn one after another: "canvas" (all 0 between uses:
    effects.Canvas(block=)), "lens" (a lens's picture and what it keeps, filled before each use). Each block is as
    big as the largest asked for (made bigger, all 0, when a bigger one is asked for)."""

    def __init__(self):
        self.blocks = {}

    def floats(self, name, n):
        block = self.blocks.get(name)
        if block is None or block.size < n:
            self.blocks[name] = None  # (the smaller one let go first)
            block = self.blocks[name] = np.zeros(n, np.float32)
        return block

    def canvas(self, w, h):
        return effects.Canvas(w, h, self.floats("canvas", w * h * 4))


class Sheet(overlay.Laid):
    """One built-in effect clip, placed and turned, laid over w x h frames (draw(t, img, gain), as a file clip's).
    `weight`: its area and foreground (an "L" image of the whole frame, or None: everywhere). `aspect`: the video's
    true shape (w x h may be a smaller frame, rounded to whole pixels): the effect on the sheet is then the very scene
    of the video at full size, and of its preview clip. `scratch`: memory shared with the scene's other sheets
    (Scratch; None: its own)."""

    def __init__(self, kind, params, seed, frames, fps, w, h, weight=None, aspect=None, scratch=None, light=None):
        cls = effects.KINDS[kind]
        p = effects.clean_params(kind, params)
        self.kind, self.p, self.w, self.h = kind, p, int(w), int(h)
        self.lens, self.blend = cls.lens, cls.blend
        self.scratch = scratch or Scratch()
        self.fx = None
        rect = effects.sheet_rect(kind, p, w, h)
        sw, sh = rect[2], rect[3]
        hm, _ = overlay.homography(p, rect, w, h)
        pts, seen, box = overlay.outline(hm, sw, sh, w, h)
        self.empty = box[0] >= box[2] or box[1] >= box[3]
        self.box, self.full = box, box == (0, 0, w, h)
        self.weight = None if weight is None or self.empty else (weight if self.full else weight.crop(box))
        if self.empty:
            return
        us, vs, jac = overlay.seen_back(hm, seen)
        most = max(float(np.hypot(jac[:, 0, 0], jac[:, 1, 0]).max()), float(np.hypot(jac[:, 0, 1], jac[:, 1, 1]).max()))
        k = min(1.0, most, math.sqrt(SHEET_MOST * w * h / (sw * sh)))  # pixels of the sheet a placed pixel
        ch = max(SMALLEST, round(sh * k))
        cw = max(2, round(ch * sw / sh))
        self.size, self.hm = (cw, ch), hm
        self.step = (sw / cw, sh / ch)  # placed pixels a pixel of the sheet
        self.pad = overlay.pad_for(jac * self.step)
        self.cover = overlay.coverage(pts, box)
        # the part of the sheet that shows (two pixels round it: the bilinear's reach): only that is laid
        bu, bv = us / self.step[0], vs / self.step[1]
        self.part = (max(0, math.floor(bu.min()) - 2), max(0, math.floor(bv.min()) - 2),
                     min(cw, math.ceil(bu.max()) + 2), min(ch, math.ceil(bv.max()) + 2))
        shape = cls.BOX[0] / cls.BOX[1] if cls.BOX else (aspect or sw / sh)
        lit = {"light": None if light is None else self._lit(light, hm, sw, sh)} if cls.LIT else {}
        self.fx = cls(effects.clean_params(kind, effects.sheet_params(kind, p)), seed, frames, fps, cw, ch, shape,
                      **lit)
        if self.lens:
            self._under()

    def _lit(self, light, hm, sw, sh):
        """The scene's light map (over the frame) as it falls on the sheet: what lies on the frame where each point of
        the sheet lands (past the frame's edge: its edge), at the map's own size (it changes slowly)."""
        lh, lw = light.shape[:2]
        rows = lh
        cols = max(2, round(rows * sw / sh))
        u = (np.arange(cols) + 0.5) / cols * sw
        v = (np.arange(rows) + 0.5) / rows * sh
        uu, vv = np.meshgrid(u, v)
        q = np.tensordot(hm, np.stack([uu, vv, np.ones_like(uu)]), axes=1)
        z = np.where(q[2] > 1e-9, q[2], 1e-9)  # (a point behind the eye never shows)
        fx = np.clip(q[0] / z / self.w * lw - 0.5, 0, lw - 1)
        fy = np.clip(q[1] / z / self.h * lh - 0.5, 0, lh - 1)
        x0, y0 = np.minimum(np.floor(fx).astype(int), lw - 2), np.minimum(np.floor(fy).astype(int), lh - 2)
        ax, ay = (fx - x0)[..., None], (fy - y0)[..., None]
        top = light[y0, x0] * (1 - ax) + light[y0, x0 + 1] * ax
        bottom = light[y0 + 1, x0] * (1 - ax) + light[y0 + 1, x0 + 1] * ax
        return (top * (1 - ay) + bottom * ay).astype(np.float32)

    def _under(self):
        """A lens: the frame's pixels under the part of the sheet that shows, and the warps between them."""
        cw, ch = self.size
        su, sv = self.step
        x0, y0, x1, y1 = self.part
        self.mask = None  # the effect applied only there (the whole sheet: no mask)
        if self.part != (0, 0, cw, ch):
            self.mask = np.zeros((ch, cw), np.float32)
            self.mask[y0:y1, x0:x1] = 1.0
        # the frame under the part (its corners can lie off the frame: there the frame's edge is repeated)
        q = self.hm @ np.array([[x0 * su, x1 * su, x1 * su, x0 * su], [y0 * sv, y0 * sv, y1 * sv, y1 * sv], [1.0] * 4])
        fx, fy = q[0] / q[2], q[1] / q[2]
        far = max(self.w, self.h)  # (no further out than this: the effect never reads that far)
        fx0, fy0 = max(-far, math.floor(fx.min()) - 2), max(-far, math.floor(fy.min()) - 2)
        fx1, fy1 = min(self.w + far, math.ceil(fx.max()) + 2), min(self.h + far, math.ceil(fy.max()) + 2)
        self.crop = (min(max(0, fx0), self.w - 1), min(max(0, fy0), self.h - 1),
                     max(min(self.w, fx1), 1), max(min(self.h, fy1), 1))
        self.edges = (self.crop[1] - fy0, fy1 - self.crop[3], self.crop[0] - fx0, fx1 - self.crop[2])  # (t, b, l, r)
        # from a pixel of the part to the frame (padded): the inverse of the warp onto the frame
        g = (np.array([[1, 0, -fx0], [0, 1, -fy0], [0, 0, 1.0]]) @ self.hm @
             np.array([[su, 0, x0 * su], [0, sv, y0 * sv], [0, 0, 1.0]]))
        g = g / g[2, 2]
        self.back = tuple(float(v) for v in g.flatten()[:8])
        self.forward = overlay.warp_coefficients(self.hm, (x0 * su, y0 * sv, su, sv), self.pad, self.box)

    def draw(self, t, img, gain=1.0):
        """Frame t of the effect laid over `img` (an RGB image of the whole frame, changed in place), `gain` (0..1,
        a fade) times as strongly; returns it."""
        if self.empty or gain <= 0:
            return img
        return self._lens(t, img, gain) if self.lens else self._draw(t, img, gain)

    def scratch_size(self):
        """How many floats this sheet asks of its scratch memory: (canvas, lens)."""
        if self.empty:
            return 0, 0
        n = self.size[0] * self.size[1] * 4
        return (0, n) if self.lens else (n, 0)

    def _draw(self, t, img, gain):
        cw, ch = self.size
        canvas = self.scratch.canvas(cw, ch)
        self.fx.draw(t, canvas, None)
        if canvas.box is None:
            return img
        try:
            # what was drawn (a pixel round it: nothing there, or the sheet's edge, repeated round it by the warp),
            # where it shows
            bx0, by0, bx1, by1 = canvas.box
            px0, py0, px1, py1 = self.part
            x0, y0 = max(px0, bx0 - 1), max(py0, by0 - 1)
            x1, y1 = min(px1, bx1 + 1), min(py1, by1 + 1)
            region = self._landing(x0, y0, x1, y1) if x0 < x1 and y0 < y1 else None
            if region is None:
                return img
            su, sv = self.step
            coefficients = overlay.warp_coefficients(self.hm, (x0 * su, y0 * sv, su, sv), self.pad, region)
            size = (region[2] - region[0], region[3] - region[1])
            rgb = canvas.rgb[y0:y1, x0:x1]
            if self.blend == "screen":  # light: added to the picture with Screen
                light = np.empty(rgb.shape, np.uint8)
                light[:] = self._bytes(rgb)
                top = self._warp(light, "RGB", size, coefficients)
                top.putalpha(self._cover(region))
                weight = None if self.weight is None else self.weight.crop(self._in_box(region))
                return overlay.lay(img, top, region, region == (0, 0, self.w, self.h), weight, "screen", gain)
            part = np.empty(rgb.shape[:2] + (4,), np.uint8)  # colour and coverage
            part[..., :3] = self._bytes(rgb)
            part[..., 3] = self._bytes(canvas.a[y0:y1, x0:x1])
            return self._over(part, coefficients, region, img, gain)
        finally:
            canvas.reset()

    def _over(self, part, coefficients, region, img, gain):
        """A part of the sheet, premultiplied colour and coverage (8-bit: warped so, they blend rightly), warped onto
        `region` of the frame and laid over `img` with Normal, as its soft outline, its area and `gain` say."""
        size = (region[2] - region[0], region[3] - region[1])
        top = self._warp(part, "RGBa", size, coefficients).convert("RGBA")
        top.putalpha(ImageChops.multiply(top.getchannel("A"), self._cover(region)))
        weight = None if self.weight is None else self.weight.crop(self._in_box(region))
        return overlay.lay(img, top, region, region == (0, 0, self.w, self.h), weight, "normal", gain)

    def _landing(self, x0, y0, x1, y1):
        """The frame pixels a part of the sheet (pixels x0..x1, y0..y1) lands on, within the box that shows; None if
        none."""
        su, sv = self.step
        q = self.hm @ np.array([[x0 * su, x1 * su, x1 * su, x0 * su], [y0 * sv, y0 * sv, y1 * sv, y1 * sv], [1.0] * 4])
        fx, fy = q[0] / q[2], q[1] / q[2]
        bx0, by0, bx1, by1 = self.box
        rx0, ry0 = max(bx0, math.floor(fx.min()) - 2), max(by0, math.floor(fy.min()) - 2)
        rx1, ry1 = min(bx1, math.ceil(fx.max()) + 2), min(by1, math.ceil(fy.max()) + 2)
        return (rx0, ry0, rx1, ry1) if rx0 < rx1 and ry0 < ry1 else None

    def _in_box(self, region):
        bx, by = self.box[:2]
        return region[0] - bx, region[1] - by, region[2] - bx, region[3] - by

    def _cover(self, region):
        return self.cover if region == self.box else self.cover.crop(self._in_box(region))

    @staticmethod
    def _bytes(a):
        """0..1 floats as 0..255 (rounded, kept within)."""
        out = a * 255.0
        out += 0.5
        return np.clip(out, 0.0, 255.0, out=out)

    def _warp(self, a, mode, size, coefficients):
        """An 8-bit part of the sheet (`mode`: RGB, or RGBa: premultiplied) warped onto `size` pixels of the frame,
        its edge repeated round it (the soft outline never darkens: overlay.pad_for)."""
        p = self.pad
        a = np.ascontiguousarray(np.pad(a, ((p, p), (p, p), (0, 0)), mode="edge"))
        im = Image.frombuffer(mode, (a.shape[1], a.shape[0]), a, "raw", mode, 0, 1)
        return im.transform(size, PERSPECTIVE, coefficients, BILINEAR)

    def _lens(self, t, img, gain):
        cw, ch = self.size
        x0, y0, x1, y1 = self.part
        under = np.asarray(img.crop(self.crop))
        top, bottom, left, right = self.edges
        if top or bottom or left or right:
            under = np.pad(under, ((top, bottom), (left, right), (0, 0)), mode="edge")
        under = Image.fromarray(under).transform((x1 - x0, y1 - y0), PERSPECTIVE, self.back, BILINEAR)
        block = self.scratch.floats("lens", cw * ch * 4)  # (only the part is read and written)
        pic, kept = block[:cw * ch * 3].reshape(ch, cw, 3), block[cw * ch * 3:cw * ch * 4].reshape(ch, cw)
        before = np.asarray(under, np.float32)
        before *= 1 / 255.0
        pic[y0:y1, x0:x1] = before
        kept[y0:y1, x0:x1] = 1.0
        self.fx.apply(t, pic, self.mask, gain, kept)
        # its change is a layer over the picture: what it adds (after - kept * before) premultiplied colour, what it
        # takes away of the picture (1 - kept) coverage; where it changes nothing, both exactly 0
        after, k = pic[y0:y1, x0:x1], kept[y0:y1, x0:x1]
        part = np.empty(after.shape[:2] + (4,), np.uint8)
        for c in range(3):
            added = before[..., c] * k
            np.subtract(after[..., c], added, out=added)
            part[..., c] = self._bytes(added)
        part[..., 3] = self._bytes(1.0 - k)
        return self._over(part, self.forward, self.box, img, 1.0)

    def release(self):
        """Frees what grows with the frame while the clip is not showing (made again when it shows)."""
        if self.fx is not None and hasattr(self.fx, "release"):
            self.fx.release()

    def close(self):
        self.release()
