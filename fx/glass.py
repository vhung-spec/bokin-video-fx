"""Water on a window pane: fogged glass (a haze that also blurs what is behind it). A lens effect: apply() changes
the picture under it (the renderer: export and exact frames); draw() is only what it lays over the picture (the
page's live preview). See effects.py for how every effect loops."""
import numpy as np
from PIL import Image, ImageFilter

from .effects import Effect, _colour, _num, _rgb

SCHEMA = {"label": "Nước trên kính", "params": [
    _num("haze", "Hơi nước (kính mờ)", 45, group="Hơi nước"), _num("blur", "Nhoè cảnh sau kính", 50, group="Hơi nước"),
    _num("opacity", "Độ đậm", 100, group="Màu"), _colour("color", "Màu hơi nước", "#dfe6ea", group="Màu")]}


class Glass(Effect):
    lens = True
    moves = False  # (fogged glass alone: nothing moves)

    def __init__(self, *a):
        super().__init__(*a)
        p = self.p
        self.alpha = 0.55 * p["haze"] / 100 * p["opacity"] / 100
        self.colour = _rgb(p["color"])
        self.radius = 14 * self.px * p["blur"] / 100 * p["haze"] / 100

    def draw(self, t, canvas, mask):
        """What it lays over the picture: the haze."""
        if self.alpha <= 0:
            return
        cover = np.full((self.h, self.w), 1.0, np.float32)
        canvas.over(0, 0, cover, self.colour, self.alpha, mask)

    def apply(self, t, picture, mask, gain):
        """The whole effect on `picture` (float32 RGB 0..1, changed in place) where `mask` (0..1, None: everywhere)
        says, `gain` (0..1, a fade) times as strongly."""
        if gain <= 0:
            return
        out = picture.copy()
        if self.radius >= 0.3:
            im = Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8))
            out = np.asarray(im.filter(ImageFilter.GaussianBlur(self.radius)), np.float32) * (1 / 255.0)
        out = out * (1 - self.alpha) + self.colour * self.alpha
        k = gain if mask is None else gain * mask[..., None]
        picture += (out - picture) * k
