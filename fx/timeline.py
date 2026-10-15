"""The timeline's timing rules, shared by the renderer and the page (index.html mirrors them).

A project makes one video of a chosen length (30 s or 1 min): a whole number of loops of its video, so the end
joins the start. Its clips sit on tracks; each clip shows from frame `start` to frame `end` (not included) of
that video, fading in and out over `fade_in` / `fade_out` seconds.

A clip that covers the whole video ("full") loops with it: no fades, its motion repeats over the whole length.
A shorter clip always fades where it touches the start or the end of the video (at least MIN_EDGE_FADE), so
the last frame still runs smoothly into the first one when the video plays in a loop.
"""
import math

LENGTHS = (30, 60)  # seconds
DEFAULT_LENGTH = 30
MIN_EDGE_FADE = 0.5  # seconds
MAX_FADE = 10.0  # seconds


def output_frames(info, seconds):
    """Frames of a video of about `seconds`: a whole number of the source loop, so the end joins the start.
    Halves round up, as on the page (Python's round() would make a 12 s loop 24 s long at 30 s, the page 36 s)."""
    return max(1, math.floor(seconds / info["seconds"] + 0.5)) * info["frames"]


def project_frames(project):
    return output_frames(project["video"], project.get("length") or DEFAULT_LENGTH)


class Span:
    """When a clip shows in a video of `total` frames at `fps`: frames start..end-1, with its fades in frames."""

    def __init__(self, clip, total, fps):
        self.start = max(0, min(int(clip.get("start", 0)), total))
        self.end = max(self.start, min(int(clip.get("end", total)), total))
        self.full = self.start == 0 and self.end == total
        self.skip = max(0.0, float(clip.get("skip", 0) or 0))  # seconds of the clip's source already played
        # seconds of its file the clip's run needs: the first half of a cut keeps the whole clip's (same picture)
        self.need = max(self.skip + (self.end - self.start) / fps, max(0.0, float(clip.get("need", 0) or 0)))
        fin = fout = 0.0
        if not self.full:
            fin = max(0.0, min(MAX_FADE, float(clip.get("fade_in", 0) or 0))) * fps
            fout = max(0.0, min(MAX_FADE, float(clip.get("fade_out", 0) or 0))) * fps
            edge = MIN_EDGE_FADE * fps
            if self.start == 0:
                fin = max(fin, edge)
            if self.end == total:
                fout = max(fout, edge)
            length = self.end - self.start
            if fin + fout > length > 0:  # fades longer than the clip: both shortened alike
                k = length / (fin + fout)
                fin, fout = fin * k, fout * k
        self.fin, self.fout = fin, fout
        # the seam's own ramp, never shortened: a short clip at an end of the video still starts / ends near 0 there
        edge = MIN_EDGE_FADE * fps
        self.seam_in = edge if not self.full and self.start == 0 else 0.0
        self.seam_out = edge if not self.full and self.end == total else 0.0

    @property
    def empty(self):
        return self.end <= self.start

    def weight(self, t):
        """How much of the clip shows at frame t: 0 outside it, 1 inside, in between while fading."""
        if t < self.start or t >= self.end:
            return 0.0
        w = 1.0
        if self.fin > 0:
            w = min(w, (t - self.start + 0.5) / self.fin)
        if self.fout > 0:
            w = min(w, (self.end - t - 0.5) / self.fout)
        if self.seam_in:
            w = min(w, (t + 0.5) / self.seam_in)
        if self.seam_out:
            w = min(w, (self.end - t - 0.5) / self.seam_out)
        return max(0.0, min(1.0, w))
