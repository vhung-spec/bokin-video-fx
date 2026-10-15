"""The timeline's timing rules, shared by the renderer and the page (index.html mirrors them).

A project makes one video of a chosen length (30 s or 1 min): a whole number of loops of its video, so the end
joins the start. Its clips sit on tracks; each clip shows from frame `start` to frame `end` (not included) of
that video, fading in and out over `fade_in` / `fade_out` seconds.

A clip that covers the whole video ("full") loops with it: no fades, its motion repeats over the whole length.
A shorter clip always fades where it touches the start or the end of the video (at least MIN_EDGE_FADE), so
the last frame still runs smoothly into the first one when the video plays in a loop.

Two clips that meet on a track (one ends where the next starts) can crossfade ("Mix", the later clip's `mix`
seconds): over that time, centred on where they meet, the first one goes on and fades out while the second one
has already begun and fades in; their weights add up to 1.
"""
import math

LENGTHS = (30, 60)  # seconds
DEFAULT_LENGTH = 30
MIN_EDGE_FADE = 0.5  # seconds
MAX_FADE = 10.0  # seconds
MAX_MIX = 5.0  # seconds: the longest crossfade between two clips that meet


def output_frames(info, seconds):
    """Frames of a video of about `seconds`: a whole number of the source loop, so the end joins the start.
    Halves round up, as on the page (Python's round() would make a 12 s loop 24 s long at 30 s, the page 36 s)."""
    return max(1, math.floor(seconds / info["seconds"] + 0.5)) * info["frames"]


def project_frames(project):
    return output_frames(project["video"], project.get("length") or DEFAULT_LENGTH)


def mixes(clips, total, fps):
    """{clip id: (seconds of crossfade with the clip before, with the clip after)} among `clips` (those shown): a clip
    with a `mix` that starts where another one on its track ends, inside the video, crossfades with it, over at
    most MAX_MIX and the length of either clip. The page has the same rule (mixAt)."""
    out = {}
    for b in clips:
        m = float(b.get("mix", 0) or 0)
        if m <= 0 or int(b["start"]) >= total:
            continue
        a = next((c for c in clips if c is not b and c.get("track") == b.get("track")
                  and int(c["end"]) == int(b["start"])), None)
        if a is None:
            continue
        m = min(m, MAX_MIX, (int(a["end"]) - int(a["start"])) / fps, (int(b["end"]) - int(b["start"])) / fps)
        if m > 0:
            out[b["id"]] = (m, out.get(b["id"], (0.0, 0.0))[1])
            out[a["id"]] = (out.get(a["id"], (0.0, 0.0))[0], m)
    return out


def mix_frames(seconds, fps):
    """Half a crossfade, in frames (halves round up, as on the page)."""
    return max(1, math.floor(seconds * fps / 2 + 0.5)) if seconds > 0 else 0


class Span:
    """When a clip shows in a video of `total` frames at `fps`: frames start..end-1, with its fades in frames.
    `mix`: its crossfades with the clips before and after it (mixes()); then it shows from half a crossfade
    before its own start to half a crossfade past its own end. `anchor` is its own start (the time of its file)."""

    def __init__(self, clip, total, fps, mix=(0.0, 0.0)):
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
        # the seam's own ramp, never shortened: a short clip at an end of the video still starts / ends near 0 there
        edge = MIN_EDGE_FADE * fps
        self.seam_in = edge if not self.full and self.start == 0 else 0.0
        self.seam_out = edge if not self.full and self.end == total else 0.0
        self.anchor = self.start
        lead, tail = mix_frames(mix[0], fps), mix_frames(mix[1], fps)
        if lead:  # it fades in across where the clip before ends
            self.start -= lead
            fin = 2 * lead
        if tail:  # it goes on past its end, fading out while the next clip fades in
            self.end += tail
            fout = 2 * tail
        self.fin, self.fout = fin, fout

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
