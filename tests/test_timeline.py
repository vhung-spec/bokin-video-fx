import unittest

from tests import helpers  # noqa: F401 - temporary data folder
from fx import timeline

N, FPS = 720, 24  # 30 s


class Spans(unittest.TestCase):
    def test_full_clip_never_fades(self):
        span = timeline.Span({"start": 0, "end": N, "fade_in": 2, "fade_out": 2}, N, FPS)
        self.assertTrue(span.full)
        self.assertEqual([span.weight(t) for t in (0, 1, N // 2, N - 1)], [1.0, 1.0, 1.0, 1.0])

    def test_inside_and_outside(self):
        span = timeline.Span({"start": 100, "end": 200}, N, FPS)
        self.assertEqual((span.weight(99), span.weight(100), span.weight(199), span.weight(200)), (0, 1, 1, 0))

    def test_fades(self):
        span = timeline.Span({"start": 100, "end": 300, "fade_in": 1, "fade_out": 2}, N, FPS)
        weights = [span.weight(t) for t in range(100, 300)]
        self.assertTrue(all(b >= a for a, b in zip(weights[:24], weights[1:25])))  # up over a second
        self.assertEqual(span.weight(150), 1.0)
        self.assertTrue(all(b <= a for a, b in zip(weights[-48:], weights[-47:])))  # down over two
        self.assertLess(span.weight(100), 0.05)
        self.assertLess(span.weight(299), 0.05)

    def test_clip_at_the_ends_fades_there(self):
        """A clip touching the start or the end of a looping video fades there, so the seam stays smooth."""
        head = timeline.Span({"start": 0, "end": 200}, N, FPS)
        tail = timeline.Span({"start": 500, "end": N}, N, FPS)
        edge = timeline.MIN_EDGE_FADE * FPS
        self.assertAlmostEqual(head.fin, edge)
        self.assertAlmostEqual(tail.fout, edge)
        self.assertLess(head.weight(0), 0.1)
        self.assertLess(tail.weight(N - 1), 0.1)

    def test_a_short_clip_at_an_end_still_ramps_there(self):
        """A clip of a frame or two at the start or the end: its fades shrink to fit, but the seam's own ramp does
        not, so it never flashes at full strength where the video loops."""
        edge = timeline.MIN_EDGE_FADE * FPS
        last = timeline.Span({"start": N - 1, "end": N, "fade_in": 0.5, "fade_out": 0.5}, N, FPS)
        first = timeline.Span({"start": 0, "end": 1, "fade_in": 0.5, "fade_out": 0.5}, N, FPS)
        self.assertLessEqual(last.weight(N - 1), 0.5 / edge + 1e-9)
        self.assertLessEqual(first.weight(0), 0.5 / edge + 1e-9)
        tail = timeline.Span({"start": 500, "end": N}, N, FPS)  # a normal clip at the end: as before
        self.assertEqual(tail.weight(600), 1.0)

    def test_fades_longer_than_the_clip(self):
        span = timeline.Span({"start": 100, "end": 112, "fade_in": 5, "fade_out": 5}, N, FPS)
        self.assertAlmostEqual(span.fin + span.fout, 12)
        self.assertTrue(all(0 <= span.weight(t) <= 1 for t in range(90, 130)))

    def test_clamped_to_the_video(self):
        span = timeline.Span({"start": 600, "end": 1440}, N, FPS)  # made for 1 min, the video is 30 s
        self.assertEqual((span.start, span.end), (600, N))
        self.assertTrue(timeline.Span({"start": 800, "end": 900}, N, FPS).empty)

    def test_whole_loops(self):
        info = {"frames": 144, "seconds": 6.0}
        self.assertEqual(timeline.output_frames(info, 30), 5 * 144)
        self.assertEqual(timeline.output_frames(info, 60), 10 * 144)
        self.assertEqual(timeline.output_frames({"frames": 168, "seconds": 7.0}, 30), 4 * 168)  # 28 s
        # halves round up, as on the page: a 12 s loop makes 36 s, not 24 s
        self.assertEqual(timeline.output_frames({"frames": 288, "seconds": 12.0}, 30), 3 * 288)
        self.assertEqual(timeline.output_frames({"frames": 576, "seconds": 24.0}, 60), 3 * 576)


if __name__ == "__main__":
    unittest.main()
