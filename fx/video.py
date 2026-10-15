"""Video in and out, through the ffmpeg bundled with imageio-ffmpeg.

probe() says what a video is; Frames reads its frames as RGB arrays; Encoder writes the delivery file; repeat()
makes a long video from a loop by copying it end to end, without encoding it again.

Delivery format (the house standard): H.264 4:2:0, BT.709 limited range, the input's size and frame rate, no
audio, and nothing in the file that names the tools. The stream is encoded into a temporary file, then copied
into the delivered one without container or stream tags, and the SEI units in which x264 writes its version
and settings are dropped on the way.
"""
import os
import re
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
MAX_SIDE = 4096
FAST_START_LIMIT = 600  # seconds: longer files skip +faststart (it rewrites the whole file once more)


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
    vpx_alpha = codec in ("vp8", "vp9") and re.search(r"alpha_mode\s*:\s*1", err) is not None
    return {"w": w, "h": h, "fps": fps, "frames": frames, "seconds": round(frames / fps_value(fps), 3),
            "matrix": matrix, "range": colour_range, "codec": codec,
            "alpha": bool(ALPHA_FORMATS.match(pix)) or vpx_alpha,
            "decoder": ("libvpx-vp9" if codec == "vp9" else "libvpx") if vpx_alpha else None}


def _from_yuv(info, size):
    w, h = size
    flags = "accurate_rnd+full_chroma_int" + ("" if (w, h) == (info["w"], info["h"]) else "+area")
    return f"scale={w}:{h}:flags={flags}:in_color_matrix={info['matrix']}:in_range={info['range']}"


class Frames:
    """The frames of a video as RGB uint8 arrays (h, w, 3), or RGBA (h, w, 4) with `alpha`, scaled to `size`,
    from frame `start` on (at most `count` of them). `seek` (seconds) starts there instead, faster for a frame
    far into a file. Use it in a with-block, so ffmpeg is stopped when reading ends early."""

    def __init__(self, path, info, size=None, start=0, count=None, alpha=False, seek=None):
        self.w, self.h = size or (info["w"], info["h"])
        self.depth = 4 if alpha else 3
        vf = ([f"select=gte(n\\,{int(start)})"] if start and seek is None else []) + [_from_yuv(info, (self.w, self.h))]
        args = [ffmpeg_exe(), "-v", "error", "-nostdin"]
        if seek:
            args += ["-ss", f"{max(0.0, seek):.4f}"]
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
    """Writes RGB uint8 frames into the delivery file `path`. close() finishes it; abort() leaves nothing."""

    def __init__(self, path, w, h, fps, crf=16, preset="slow"):
        self.path, self.tmp = path, path + ".enc.mp4"
        self.err = tempfile.TemporaryFile()
        args = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-", "-vf", TO_BT709, "-c:v", "libx264",
                "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", *BT709, *DROP_SEI, "-an", self.tmp]
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.err,
                                     **NO_WINDOW)

    def write(self, frame):
        try:
            self.proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
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
            remux(self.tmp, self.path)
        finally:
            self.err.close()
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


def repeat(src, dst, times, seconds=None, progress=None, cancelled=None):
    """`dst` = the loop `src` played `times` times, by copying (no encoding, so no loss and no new seam).
    progress(fraction) is called while it runs; cancelled() -> True stops it and removes `dst`."""
    entry = "file '" + os.path.abspath(src).replace("\\", "/").replace("'", "'\\''") + "'\n"
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(entry * int(times))
        listing = f.name
    total = seconds or 0
    args = [ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "concat", "-safe", "0", "-i", listing,
            "-map", "0:v:0", "-c", "copy", *CLEAN] + (["-movflags", "+faststart"] if total <= FAST_START_LIMIT else []) \
        + ["-progress", "pipe:1", "-nostats", dst]
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err, **NO_WINDOW)
    try:
        for line in proc.stdout:
            if cancelled and cancelled():
                proc.kill()
                break
            key, _, value = line.decode("ascii", "replace").strip().partition("=")
            if key == "out_time_us" and total and progress and value.isdigit():
                progress(min(1.0, int(value) / 1e6 / total))
        code = proc.wait()
        if cancelled and cancelled():
            _remove(dst)
            return False
        if code != 0:
            err.seek(0)
            _remove(dst)
            raise RuntimeError(f"ffmpeg failed ({code}): {_tail(err.read().decode('utf-8', 'replace'))}")
        return True
    finally:
        proc.stdout.close()
        err.close()
        _remove(listing)
