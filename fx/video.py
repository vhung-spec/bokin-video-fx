"""Video in and out, through the ffmpeg bundled with imageio-ffmpeg.

probe() says what a video is; Frames reads its frames as RGB arrays; Encoder writes the delivery file (or one
piece of it, rendered in parallel) and join() puts the pieces together without encoding them again; make_proxy()
and ClipWriter make the light videos the page plays in its live preview.

Delivery format (the house standard): H.264 4:2:0, BT.709 limited range, the input's size and frame rate, no
audio, and nothing in the file that names the tools. The stream is encoded into a temporary file, then copied
into the delivered one without container or stream tags, and the SEI units in which x264 writes its version
and settings are dropped on the way.
"""
import os
import re
import shutil
import subprocess
import tempfile

import numpy as np

NO_WINDOW = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW: no console per call
CLEAN = ["-map_metadata", "-1", "-map_chapters", "-1", "-fflags", "+bitexact", "-flags:v", "+bitexact",
         "-metadata:s:v:0", "encoder="]
DROP_SEI = ["-bsf:v", "filter_units=remove_types=6"]
BT709 = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv"]
TO_BT709 = "scale=flags=accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range=tv,format=yuv420p"
# ffmpeg's colour space names -> the scale filter's matrix names
MATRICES = {"bt709": "bt709", "bt470bg": "bt470", "smpte170m": "smpte170m", "bt2020nc": "bt2020",
            "bt2020c": "bt2020", "smpte240m": "smpte240m", "fcc": "fcc"}
NTSC = {"23.98": "24000/1001", "23.976": "24000/1001", "29.97": "30000/1001", "47.95": "48000/1001",
        "59.94": "60000/1001", "119.88": "120000/1001"}
HDR = {"smpte2084", "arib-std-b67", "bt2020", "bt2020-10", "bt2020-12"}  # PQ, HLG and BT.2020 primaries
MAX_SIDE = 4096


class VideoError(Exception):
    """A file that cannot be used. The message is shown on the page as is."""


def ffmpeg_exe():
    exe = os.environ.get("BOKIN_FFMPEG")
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def _tail(text, lines=6):
    return "\n".join(text.strip().splitlines()[-lines:])


def run(args, timeout=None):
    """Run ffmpeg with `args` and return what it wrote to stderr; RuntimeError (with its last lines) if it fails."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-nostdin", *args], capture_output=True, timeout=timeout,
                       stdin=subprocess.DEVNULL, **NO_WINDOW)
    err = r.stderr.decode("utf-8", "replace")
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({r.returncode}): {_tail(err)}")
    return err


def parse_fps(text):
    if text in NTSC:
        return NTSC[text]
    value = float(text)
    return str(int(value)) if value == int(value) else f"{round(value * 1000)}/1000"


def fps_value(fps):
    num, _, den = str(fps).partition("/")
    return float(num) / float(den or 1)


ALPHA_FORMATS = re.compile(r"(yuva|rgba|bgra|argb|abgr|gbrap|ya\d|pal8)")


def probe(path, overlay=False):
    """{w, h, fps ('24' or '24000/1001'), frames, seconds, matrix, range, codec, alpha, decoder} of the first
    video stream. Frames are counted by decoding the video once, so the loop length is exact. An overlay (a file
    laid over the loop) may have any size and a single frame."""
    try:
        err = run(["-i", path, "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "null", "-"], timeout=600)
    except RuntimeError as e:
        raise VideoError("Không đọc được file này như một video.") from e
    except subprocess.TimeoutExpired as e:
        raise VideoError("Video quá dài hoặc quá nặng để đọc.") from e
    line = next((ln for ln in err.splitlines() if re.search(r"Stream #\d+:\d+.*: Video: ", ln)), None)
    if line is None:
        raise VideoError("File này không có hình.")
    desc = line.split(": Video: ", 1)[1]
    m = re.search(r",\s*([a-z0-9_]+)(?:\(([^)]*)\))?,\s*(\d+)x(\d+)", desc)
    if not m:
        raise VideoError("Không đọc được kích thước của video.")
    pix, colour, w, h = m.group(1), (m.group(2) or ""), int(m.group(3)), int(m.group(4))
    tags = [t.strip() for t in colour.split(",") if t.strip()]
    names = {n for t in tags for n in t.split("/")}  # "space/primaries/transfer", or one name for all three
    if names & HDR:
        raise VideoError("Video này là HDR / BT.2020 (thường do điện thoại quay ở chế độ HDR), app chưa xử lý được "
                         "màu của nó. Hãy xuất lại thành video thường (SDR, BT.709), hoặc tắt Video HDR khi quay.")
    colour_range = "pc" if "pc" in tags or pix.startswith("yuvj") else "tv"
    space = next((t.split("/")[0] for t in tags if t.split("/")[0] in MATRICES), None)
    matrix = MATRICES.get(space) or ("bt709" if h >= 600 else "smpte170m")  # untagged: the usual for its size
    rate = re.search(r",\s*([\d.]+) fps", desc) or re.search(r",\s*([\d.]+) tbr", desc)
    if not rate:
        raise VideoError("Không đọc được số khung hình mỗi giây của video.")
    rotation = re.search(r"rotation of (-?[\d.]+) degrees", err)
    if rotation and round(abs(float(rotation.group(1)))) % 180 == 90:
        w, h = h, w  # ffmpeg turns such a video upright when it decodes it
    counts = re.findall(r"frame=\s*(\d+)", err)
    frames = int(counts[-1]) if counts else 0
    if frames < (1 if overlay else 2):
        raise VideoError("Video không có đủ khung hình.")
    if max(w, h) > MAX_SIDE:
        raise VideoError(f"Video quá lớn ({w}×{h}). App nhận video tới {MAX_SIDE} điểm ảnh mỗi cạnh.")
    if (w % 2 or h % 2) and not overlay:
        raise VideoError(f"Kích thước video ({w}×{h}) phải là số chẵn.")
    fps = parse_fps(rate.group(1))
    codec = re.match(r"(\w+)", desc).group(1)
    # VP8/VP9 keep transparency in a side stream that only libvpx decodes
    vpx_alpha = codec in ("vp8", "vp9") and re.search(r"alpha_mode\s*:\s*1", err, re.I) is not None
    return {"w": w, "h": h, "fps": fps, "frames": frames, "seconds": round(frames / fps_value(fps), 3),
            "matrix": matrix, "range": colour_range, "codec": codec, "pix": pix,
            "alpha": bool(ALPHA_FORMATS.match(pix)) or vpx_alpha,
            "decoder": ("libvpx-vp9" if codec == "vp9" else "libvpx") if vpx_alpha else None}


def _from_yuv(info, size, src=None):
    w, h = size
    sw, sh = (src[2] - src[0], src[3] - src[1]) if src else (info["w"], info["h"])  # the part read (a crop)
    flags = "accurate_rnd+full_chroma_int" + ("" if (w, h) == (sw, sh) else "+area")
    return f"scale={w}:{h}:flags={flags}:in_color_matrix={info['matrix']}:in_range={info['range']}"


class Frames:
    """The frames of a video as RGB uint8 arrays (h, w, 3), or RGBA (h, w, 4) with `alpha`, scaled to `size`,
    from frame number `start` on (at most `count` of them), numbered in decoding order as probe() counts them.
    Use it in a with-block, so ffmpeg is stopped when reading ends early."""

    def __init__(self, path, info, size=None, start=0, count=None, alpha=False, src=None):
        self.w, self.h = size or (info["w"], info["h"])
        self.depth = 4 if alpha else 3
        vf = ([f"select=gte(n\\,{int(start)})"] if start else [])
        if src and tuple(src) != (0, 0, info["w"], info["h"]):  # only this part of the source (x0, y0, x1, y1)
            vf.append(f"crop={src[2] - src[0]}:{src[3] - src[1]}:{src[0]}:{src[1]}:exact=1")
        vf.append(_from_yuv(info, (self.w, self.h), src))
        args = [ffmpeg_exe(), "-v", "error", "-nostdin"]
        if alpha and info.get("decoder"):
            args += ["-c:v", info["decoder"]]
        args += ["-i", path, "-map", "0:v:0", "-vf", ",".join(vf), "-fps_mode", "passthrough", "-an", "-sn", "-dn"]
        if count:
            args += ["-frames:v", str(int(count))]
        args += ["-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24", "-"]
        self.err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=self.err,
                                     **NO_WINDOW)

    def __iter__(self):
        n = self.w * self.h * self.depth
        while True:
            buf = self.proc.stdout.read(n)
            if len(buf) < n:
                return
            yield np.frombuffer(buf, np.uint8).reshape(self.h, self.w, self.depth)

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.proc.stdout.close()
        self.err.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_frame(path, info, index, size=None):
    """One frame (RGB uint8) of a video."""
    index = max(0, min(int(index), info["frames"] - 1))
    with Frames(path, info, size, start=index, count=1) as frames:
        for frame in frames:
            return frame.copy()
    raise VideoError("Không đọc được khung hình này của video.")


def remux(src, dst, fast_start=True):
    """Copy the video stream of `src` into `dst` without any tags (no encoding)."""
    run(["-y", "-i", src, "-map", "0:v:0", "-c", "copy", *CLEAN]
        + (["-movflags", "+faststart"] if fast_start else []) + [dst])


class Encoder:
    """Writes RGB uint8 frames into the delivery file `path`. close() finishes it; abort() leaves nothing.
    A `part` (one piece of a video rendered in pieces) is written as is; join() makes the delivery file."""

    def __init__(self, path, w, h, fps, crf=16, preset="medium", threads=0, part=False, pix="yuv420p"):
        self.path, self.part = path, part
        self.tmp = path if part else path + ".enc.mp4"
        self.err = tempfile.TemporaryFile()
        args = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-", "-vf", TO_BT709.replace("yuv420p", pix),
                "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", pix, *BT709, *DROP_SEI, "-an"] \
            + (["-threads", str(threads)] if threads else []) + [self.tmp]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err,
                                     **NO_WINDOW)

    def write(self, frame):
        """A frame: a uint8 RGB array, or an RGB Pillow image."""
        data = frame.tobytes() if hasattr(frame, "getbands") else np.ascontiguousarray(frame, dtype=np.uint8).tobytes()
        try:
            self.proc.stdin.write(data)
        except (BrokenPipeError, OSError) as e:
            raise RuntimeError(f"ffmpeg stopped: {self._error()}") from e

    def _error(self):
        self.err.seek(0)
        return _tail(self.err.read().decode("utf-8", "replace"))

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait()
        try:
            if code != 0:
                raise RuntimeError(f"ffmpeg failed ({code}): {self._error()}")
            if not self.part:
                remux(self.tmp, self.path)
        finally:
            self.err.close()
            if not self.part:
                _remove(self.tmp)

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        self.err.close()
        _remove(self.tmp)
        _remove(self.path)


def _remove(path):
    try:
        os.remove(path)
    except OSError:
        pass


def join(parts, dst, fast_start=True, alpha=False):
    """`dst` = the pieces played one after another, copied (no encoding) into a clean delivery file. `alpha`:
    WebM pieces with transparency (the joined file must say so again, or players show it opaque)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for part in parts:
            f.write("file '" + os.path.abspath(part).replace("\\", "/").replace("'", "'\\''") + "'\n")
        listing = f.name
    try:
        run(["-y", "-f", "concat", "-safe", "0", "-i", listing, "-map", "0:v:0", "-c", "copy", *CLEAN]
            + (["-movflags", "+faststart"] if fast_start else [])
            + (["-metadata:s:v:0", "alpha_mode=1"] if alpha else []) + [dst])
    finally:
        _remove(listing)


class ClipWriter:
    """A small clip for the page's live preview: RGBA frames into WebM (VP9 keeps the transparency), or RGB
    frames into H.264."""

    def __init__(self, path, w, h, fps, alpha):
        self.err = tempfile.TemporaryFile()
        codec = (["-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-deadline", "realtime",
                  "-cpu-used", "8", "-b:v", "0", "-crf", "34", "-row-mt", "1"] if alpha else
                 ["-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-vf", TO_BT709])
        codec += ["-g", "24"]  # the page jumps anywhere in it
        args = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24",
                "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-", *codec, "-an", *CLEAN, path]
        self.path = path
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err,
                                     **NO_WINDOW)

    def write(self, frame):
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        except (BrokenPipeError, OSError) as e:
            self.err.seek(0)
            raise RuntimeError(f"ffmpeg stopped: {_tail(self.err.read().decode('utf-8', 'replace'))}") from e

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        code = self.proc.wait()
        self.err.seek(0)
        message = _tail(self.err.read().decode("utf-8", "replace"))
        self.err.close()
        if code != 0:
            _remove(self.path)
            raise RuntimeError(f"ffmpeg failed ({code}): {message}")

    def abort(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        self.err.close()
        _remove(self.path)


class _SeeThroughPiece:
    """RGBA frames into a VP9 WebM piece that keeps them well (BT.709, tagged, as probe() reads it back)."""

    def __init__(self, path, w, h, fps):
        self.path, self.err = path, tempfile.TemporaryFile()
        args = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                "-framerate", str(fps), "-i", "-", "-vf",
                "scale=flags=accurate_rnd+full_chroma_int:out_color_matrix=bt709:out_range=tv,format=yuva420p",
                "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0", "-b:v", "0", "-crf", "16",
                "-deadline", "good", "-cpu-used", "4", "-row-mt", "1", *BT709, "-an", *CLEAN, path]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err,
                                     **NO_WINDOW)

    write, close = ClipWriter.write, ClipWriter.close


def reverse(src, info, dst, cancelled=None, progress=None, budget=160_000_000):
    """The video played backwards, at its full size (a clip set to "Phát ngược"): `dst` with .mp4 (H.264), or
    .webm (VP9 keeping the transparency). The file is read once, in order, a run of frames at a time (as many as
    `budget` bytes hold); each run is written backwards as a piece, and the pieces are joined last one first.
    Returns the file written, or None if `cancelled`."""
    alpha = bool(info.get("alpha"))
    w, h, ext = info["w"], info["h"], ".webm" if alpha else ".mp4"
    run_of = max(1, int(budget // (w * h * (4 if alpha else 3))))
    out = os.path.splitext(dst)[0] + ext
    folder = os.path.splitext(dst)[0] + ".parts"
    shutil.rmtree(folder, ignore_errors=True)
    os.makedirs(folder)
    parts, frames_read = [], 0
    try:
        def backwards(frames):
            path = os.path.join(folder, f"{len(parts):05d}{ext}")
            writer = (_SeeThroughPiece(path, w, h, info["fps"]) if alpha else  # (4:2:0 needs even sides)
                      Encoder(path, w, h, info["fps"], crf=12, preset="veryfast", part=True,
                              pix="yuv444p" if w % 2 or h % 2 else "yuv420p"))
            for frame in reversed(frames):
                writer.write(frame)
            writer.close()
            parts.append(path)
        with Frames(src, info, alpha=alpha) as frames:
            run = []
            for frame in frames:
                if cancelled and cancelled():
                    return None
                run.append(frame)
                frames_read += 1
                if len(run) >= run_of:
                    backwards(run)
                    run = []
                if progress:
                    progress(0.9 * frames_read / max(1, info["frames"]))
            if run:
                backwards(run)
        if not parts:
            raise VideoError("Không đọc được khung hình nào của video.")
        tmp = os.path.splitext(dst)[0] + ".part" + ext
        join(list(reversed(parts)), tmp, alpha=alpha)
        os.replace(tmp, out)
        return out
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        _remove(os.path.splitext(dst)[0] + ".part" + ext)


def plays_in_browser(info):
    """Whether Edge/Chrome can play the file as it is (H.264 8-bit 4:2:0, or VP8/VP9 WebM)."""
    return (info["codec"] == "h264" and info.get("pix") in ("yuv420p", "yuvj420p")) or info["codec"] in ("vp8", "vp9")


PROXY_V = 2  # light copies with frame n at exactly n / fps and BT.709 colours; older ones are made again


def make_proxy(src, info, dst, width=1280, alpha=None, keyint=12):
    """A light copy of a video for the page's live preview: VP9 WebM when it has transparency, else H.264, with a
    key frame every `keyint` frames so the page can jump anywhere quickly. Returns the file written (dst with the
    right extension); it is written under another name first, so a half-made copy is never served."""
    w = min(width, info["w"])
    h = max(2, round(info["h"] * w / info["w"] / 2) * 2)
    w = max(2, w - w % 2)  # a 1-pixel-wide video: 2, not 0 (which ffmpeg reads as the input's width)
    alpha = bool(info.get("alpha")) if alpha is None else alpha
    out = os.path.splitext(dst)[0] + (".webm" if alpha else ".mp4")
    tmp = os.path.splitext(dst)[0] + ".part" + (".webm" if alpha else ".mp4")
    args = ["-y"] + (["-c:v", info["decoder"]] if alpha and info.get("decoder") else []) + ["-i", src, "-map", "0:v:0"]
    # frame n of the copy is shown at exactly n / fps, whatever the timing of the file: the page and the renderer
    # then mean the same frame by the same time (no frame doubled or dropped, uneven phone videos included)
    even = f"setpts=N/(({info['fps']})*TB)"
    timing = ["-fps_mode", "cfr", "-r", str(info["fps"])]  # every frame already sits on its slot: none doubled
    if alpha:
        args += ["-vf", f"scale={w}:{h}:flags=area,format=yuva420p,{even}", *timing, "-c:v", "libvpx-vp9",
                 "-auto-alt-ref", "0", "-deadline", "realtime", "-cpu-used", "8", "-b:v", "0", "-crf", "34",
                 "-row-mt", "1", "-g", str(keyint)]
    else:
        # SD (BT.601) and full-range data made BT.709 TV range, as tagged
        args += ["-vf", _from_yuv(info, (w, h)) + f":out_color_matrix=bt709:out_range=tv,format=yuv420p,{even}",
                 *timing, "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                 "-g", str(keyint), *BT709, "-movflags", "+faststart"]
    try:
        run(args + ["-an", *CLEAN, tmp], timeout=1800)
        os.replace(tmp, out)
    finally:
        _remove(tmp)
    return out
