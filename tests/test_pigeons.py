"""Pigeons (fx/pigeons.py): a placed effect of rock doves drawn by code, walking about on the ground or on a ledge,
looking round and now and then pecking at seeds, lit by the scene where their box stands."""
import math
import unittest
from unittest import mock

import numpy as np

from tests import helpers  # noqa: F401 - temporary data folder
from fx import effects  # (first: it loads every effect's module)
from fx import engine, overlay, pigeons

W, H = 480, 270
AT = {"scale": 90, "x": 0, "y": 0}
MODES = ("ground", "ledge")
DAY = {"lit": 0}  # (lit by daylight, not by a scene)


def make(params=None, frames=240, fps=24, w=W, h=H, seed=3, light=None):
    return effects.make("pigeons", {**AT, **(params or {})}, seed, frames, fps, w, h, light)


def drawn(fx, t):
    canvas = effects.Canvas(fx.w, fx.h)
    fx.draw(t, canvas, None)
    return canvas.rgb.copy(), canvas.a.copy()


def lowest(part):
    """The lowest point of a part (an ellipsoid: name, centre, rotation, radii), above the ground."""
    _, c, rot, r = part
    return float(c[2] - math.sqrt(float(((rot[2, :] * r) ** 2).sum())))


def walks(walker, least=2.5):
    """The walks of a walker's day long enough to be at full speed in their middle (start, length: seconds)."""
    return [(start, d) for kind, start, d, _ in walker.segs if kind == "walk" and d > least]


def light():
    key = np.array([1.1, 1.05, 1.0], np.float32)
    way = np.array([-0.5, -0.6, 0.8])
    return key, way / np.linalg.norm(way), np.array([0.45, 0.48, 0.55], np.float32)


class Loop(unittest.TestCase):
    def test_frame_p_is_frame_0(self):
        for mode in MODES:
            for frames, fps in ((240, 24), (150, 25)):
                fx = make({"mode": mode, "amount": 70, "peck": 100, "rest": 100, "court": 100}, frames, fps)
                a, b = drawn(fx, 0), drawn(fx, frames)
                self.assertLess(float(np.abs(a[0] - b[0]).max()), 1e-6, (mode, frames, fps))
                self.assertLess(float(np.abs(a[1] - b[1]).max()), 1e-6, (mode, frames, fps))

    def test_they_move(self):
        for mode in MODES:
            fx = make({"mode": mode, "amount": 70, "busy": 100})
            a = sum(np.abs(drawn(fx, 0)[1] - drawn(fx, t)[1]).max() for t in (6, 30, 90))
            self.assertGreater(float(a), 0.5, mode)

    def test_each_walker_comes_home(self):
        """A bird goes round a closed path of its own, as far as it walks over the loop (once round, or more often
        round a smaller path): at the loop's end where it started, facing the same way. Along a ledge its path goes
        no farther from home than its room, however long it walks, and turns round at its ends in a narrow half
        circle (TURN)."""
        for seed in range(6):
            for busy in (0.0, 0.5, 1.0):
                for loop, ledge, room in ((20.0, False, None), (20.0, False, 0.8), (20.0, True, 2.0),
                                          (60.0, True, 1.0)):
                    walker = pigeons.Walker(np.random.default_rng(seed), loop, busy, 1.0, ledge, room)
                    x0, y0, a0 = walker.where(0.0)
                    x1, y1, a1 = walker.where(loop - 1e-9)
                    self.assertLess(math.hypot(x1 - x0, y1 - y0), 1e-3, (seed, busy, ledge))
                    self.assertLess(abs(math.remainder(a1 - a0, math.tau)), 0.05, (seed, busy, ledge))
                    if ledge:
                        self.assertLessEqual(float(np.abs(walker.px).max()), room * 1.01, (seed, busy, loop))
                        self.assertLessEqual(float(np.abs(walker.py).max()), pigeons.TURN * 1.01, (seed, busy, loop))


class Walking(unittest.TestCase):
    """As a pigeon walks: its planted foot stays put on the ground, its head is held still and thrust forward a
    step at a time; it stops now and then to peck, its beak reaching the ground."""

    def test_a_planted_foot_stays_put(self):
        walker = pigeons.Walker(np.random.default_rng(4), 30.0, 0.8, 1.0)
        start, d = walks(walker)[0]
        for side in (0, 1):
            ground = []
            for t in np.linspace(start + 0.4, start + d - 0.4, 400):
                foot = walker.pose(t).feet[side]
                if foot[2] == 0.0:  # (planted: where it is on the ground, along the way the bird goes)
                    ground.append(walker.walked_at(t) + foot[0])
                else:
                    ground.append(None)
            runs, run = [], []
            for g in ground + [None]:
                if g is None:
                    if len(run) > 5:
                        runs.append(run)
                    run = []
                else:
                    run.append(g)
            self.assertGreater(len(runs), 2)
            for run in runs:
                self.assertLess(max(run) - min(run), 0.004, side)  # (units: a bird is 0.86 long)

    def test_its_feet_take_turns(self):
        walker = pigeons.Walker(np.random.default_rng(4), 30.0, 0.8, 1.0)
        start, d = walks(walker)[0]
        for t in np.linspace(start + 0.4, start + d - 0.4, 200):
            feet = walker.pose(t).feet
            self.assertTrue(feet[0][2] == 0.0 or feet[1][2] == 0.0, t)  # (always one on the ground)
            self.assertLess(max(abs(feet[0][0]), abs(feet[1][0])), pigeons.STRIDE / 4 + 1e-9)

    def test_its_head_held_still_then_thrust(self):
        """Its head stays nearly still on the ground's way most of the time, and goes forward fast a bob a step (two
        a stride)."""
        walker = pigeons.Walker(np.random.default_rng(4), 30.0, 0.8, 1.0)
        start, d = walks(walker)[0]
        ts = np.linspace(start + 0.5, start + d - 0.5, 2000)
        head = np.array([walker.walked_at(t) + walker.pose(t).head[0] for t in ts])
        body = np.array([walker.walked_at(t) for t in ts])
        speed, at = np.diff(head) / np.diff(ts), np.diff(body) / np.diff(ts)
        still = np.abs(speed) < 0.2 * at.mean()
        self.assertGreater(float(still.mean()), 0.55)
        thrusts = int(np.sum(np.diff(still.astype(int)) == -1))
        steps = (body[-1] - body[0]) / (pigeons.STRIDE / 2)
        self.assertLessEqual(abs(thrusts - steps), 2.0, (thrusts, steps))

    def test_pecks_now_and_then(self):
        """"Mổ hạt": never at 0; now and then by default (a few pecks a bird in half a minute, one to three at a
        time); often at 100. Each peck goes all the way down and back up a little."""
        def pecks(hungry):
            out = []
            for seed in range(30):
                walker = pigeons.Walker(np.random.default_rng(seed), 30.0, 0.5, 1.0, hungry=hungry)
                out += [n for kind, _, _, n in walker.segs if kind == "peck"]
            return out
        self.assertEqual(pecks(0.0), [])
        some = pecks(0.4)
        self.assertTrue(2 <= sum(some) / 30 <= 8, sum(some) / 30)
        self.assertTrue(all(1 <= n <= 3 for n in some), some)
        self.assertGreater(sum(pecks(1.0)), 2.5 * sum(some))
        walker = pigeons.Walker(np.random.default_rng(1), 30.0, 0.5, 1.0, hungry=1.0)
        kind, start, d, n = next(s for s in walker.segs if s[0] == "peck" and s[3] >= 2)
        reach = np.array([walker.reach(t) for t in np.linspace(start, start + d, 600)])
        self.assertEqual(reach[0], 0.0)
        self.assertLess(reach[-1], 1e-3)
        downs = np.sum((reach[1:] >= 0.99) & (reach[:-1] < 0.99))
        self.assertEqual(int(downs), n)  # (all the way down, once a peck)

    def test_no_jumps(self):
        """Nothing about a bird jumps from one moment to the next, all its day, about or along a ledge: its head and
        its body glide (no faster than its head goes as it thrusts it forward), it turns no faster than its path bends
        (never round on the spot), and its feet are together where each walk starts and stops."""
        fps = 240
        for ledge in (False, True):
            for seed in range(6):
                ways = {"rest": 0.7, "court": 0.9, "male": True} if seed % 2 else {}  # (sitting, preening,
                # courting too)
                walker = pigeons.Walker(np.random.default_rng(seed), 20.0, 0.7, 1.0, ledge, 1.2, hungry=0.6, **ways)
                ts = np.arange(0.0, 20.0, 1 / fps)
                head, breast, heading = [], [], []
                for t in ts:
                    x, y, a = walker.where(t)
                    allp = pigeons.parts(walker.pose(t))
                    turn = pigeons._rot(yaw=a)
                    head.append(np.array([x, y, 0.0]) + turn @ allp[pigeons.SMOOTH - 1][1])
                    breast.append(np.array([x, y, 0.0]) + turn @ allp[0][1])
                    heading.append(a)
                speed = [np.linalg.norm(np.diff(np.array(p), axis=0), axis=1) * fps for p in (head, breast)]
                self.assertLess(float(speed[0].max()), 2.5, (ledge, seed))  # (units a second: the bird is 0.86)
                self.assertLess(float(speed[1].max()), 1.0, (ledge, seed))
                spin = np.abs(np.angle(np.exp(1j * np.diff(heading)))) * fps
                bend = 1 / pigeons.TURN if ledge else 1 / pigeons.BEND
                courting = np.zeros(len(spin), bool)  # (turning round on the spot, courting: at most twice round)
                for kind, start, d, n in walker.segs:
                    if kind == "court":
                        courting |= (ts[1:] > start) & (ts[:-1] < start + d)
                        most = math.tau * abs(n) / (d - 2 * pigeons.SPIN_IN) * 1.875 * 1.05
                        self.assertLess(float(spin[(ts[1:] > start) & (ts[:-1] < start + d)].max()), most)
                if (~courting).any():
                    self.assertLess(float(spin[~courting].max()), bend * walker.speed * 1.15, (ledge, seed))
                for i, (kind, start, d, _) in enumerate(walker.segs):
                    if kind == "walk":
                        for t in (start + 1e-6, start + d - 1e-6):
                            feet = walker.pose(t).feet
                            self.assertLess(max(abs(feet[0][0]), abs(feet[1][0])), 0.004, (ledge, seed, t))
                            self.assertLess(max(feet[0][2], feet[1][2]), 1e-3, (ledge, seed, t))

    def test_paths_bend_gently(self):
        """A bird's path about is a smooth wobbly loop of its own: never sharper than BEND (a short path: a
        circle)."""
        for seed in range(40):
            walker = pigeons.Walker(np.random.default_rng(seed), 30.0, 0.5, 1.0, room=1.0 + seed % 3)
            x, y = walker.px, walker.py
            dx, dy = (np.roll(x, -1) - np.roll(x, 1)) / 2, (np.roll(y, -1) - np.roll(y, 1)) / 2
            ddx, ddy = np.roll(x, -1) - 2 * x + np.roll(x, 1), np.roll(y, -1) - 2 * y + np.roll(y, 1)
            bend = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
            most = max(1 / pigeons.BEND, math.tau / walker.length)
            self.assertLess(float(bend.max()), most * 1.05, seed)

    def test_its_beak_reaches_the_ground(self):
        for seed in range(4):
            walker = pigeons.Walker(np.random.default_rng(seed), 30.0, 0.2, 1.0, hungry=1.0)
            pecks = [(start, d) for kind, start, d, _ in walker.segs if kind == "peck"]
            self.assertTrue(pecks, seed)
            start, d = pecks[0]
            low = min(lowest([p for p in pigeons.parts(walker.pose(t)) if p[0] == "beak"][0])
                      for t in np.linspace(start, start + d, 600))
            self.assertLess(low, 0.01, seed)
            self.assertGreater(low, -0.01, seed)  # (not into the ground)

    def test_standing_on_its_feet(self):
        """Standing, its toes lie on the ground (four a foot: three forward, one back; joint by joint drawn big, one
        piece each drawn small), its claws reach down to it, nothing goes into it; its body is above its legs, its
        feathered thighs over their tops only."""
        for fine in (True, False):
            parts = pigeons.parts(pigeons.Pose(), fine)
            self.assertGreater(min(lowest(p) for p in parts), -1e-9, fine)
            toes = [p for p in parts if p[0] in ("toe", "phal")]
            self.assertTrue(all(abs(lowest(p)) < 1e-6 for p in toes), fine)
            if not fine:
                self.assertEqual(len(toes), 8)
                ahead = sorted(np.sign(p[2][0, 0]) for p in toes)  # (each toe's way: forward or back)
                self.assertEqual(ahead, [-1.0] * 2 + [1.0] * 6)
            claws = [p for p in parts if p[0] == "claw"]
            self.assertEqual(len(claws), 16)  # (each a base and a tip, curving down)
            self.assertLess(max(min(lowest(p) for p in claws[k:k + 2]) for k in range(0, 16, 2)), 1e-6, fine)
            body = [p for p in parts[:pigeons.SMOOTH] if p[0] != "thigh"]
            self.assertGreater(min(lowest(p) for p in body), 0.05)  # (its body above its legs)
            thighs = [p for p in parts[:pigeons.SMOOTH] if p[0] == "thigh"]
            self.assertEqual(len(thighs), 2)
            self.assertGreater(min(lowest(p) for p in thighs), 0.025)  # (its thighs over the tops of its legs only)

    def test_toes_curl_as_a_foot_lifts(self):
        """A lifted foot's toes are drawn together and hang down; on the ground they lie spread flat."""
        flat = [p for p in pigeons._foot(1, np.array([0.0, 0.03, 0.11]), (0.0, 0.03, 0.0), False) if p[0] == "toe"]
        up = [p for p in pigeons._foot(1, np.array([0.0, 0.03, 0.11]), (0.0, 0.03, 0.025), False) if p[0] == "toe"]
        spread = [float(np.ptp([math.atan2(p[2][1, 0], p[2][0, 0]) for p in toes[:3]])) for toes in (flat, up)]
        self.assertLess(spread[1], 0.75 * spread[0])
        self.assertTrue(all(p[2][2, 0] < -0.2 for p in up[:3]))  # (pointing down)
        self.assertTrue(all(abs(p[2][2, 0]) < 1e-9 for p in flat))


class Ways(unittest.TestCase):
    """Besides walking, looking round and pecking, a bird sits down a while, preens, or (a male) courts; each bird
    its own ways, as different from the next as "Mỗi con một kiểu" says."""

    def seg(self, kind, **ways):
        for seed in range(40):
            walker = pigeons.Walker(np.random.default_rng(seed), 40.0, 0.4, 1.0, **ways)
            for s in walker.segs:
                if s[0] == kind:
                    return walker, s
        self.fail(kind)

    def test_sits_down(self):
        """Sitting, its belly is on the ground, its feathers fluffed, its feet drawn up out of sight under it."""
        walker, (_, start, d, _) = self.seg("rest", rest=1.0)
        allp = pigeons.parts(walker.pose(start + d / 2))
        self.assertLess(min(lowest(p) for p in allp[:pigeons.SMOOTH]), 0.012)
        self.assertGreater(min(lowest(p) for p in allp if p[0] in ("phal", "toe")), 0.02)
        stood = pigeons.parts(walker.pose(start))
        self.assertGreater(min(lowest(p) for p in stood[:pigeons.SMOOTH] if p[0] != "thigh"), 0.05)  # (up again as
        # it starts)

    def test_preens(self):
        """Preening, it turns its head back over a shoulder onto its back, or bends it down to its breast."""
        for n in (1, -1, 0):
            walker, s = self.seg("preen", rest=1.0)
            s[3] = n
            pose = walker.pose(s[1] + s[2] / 2)
            if n:
                self.assertGreater(pose.head_yaw * n, 2.0)
                self.assertLess(pose.head[0], 0.0)  # (behind its neck)
            else:
                self.assertLess(pose.head[2], 0.25)
                self.assertLess(pose.head_pitch, -0.8)

    def test_courts(self):
        """A male courting puffs up his neck and breast, spreads his tail down to the ground, bows, and turns round
        on the spot once or twice, ending as he faced; a female never courts."""
        walker, (_, start, d, n) = self.seg("court", court=1.0, male=True)
        mid = pigeons.parts(walker.pose(start + d / 2))
        calm = pigeons.parts(pigeons.Pose())
        neck = [next(p for p in allp if p[0] == "neck")[3][0] for allp in (mid, calm)]
        fan = [next(p for p in allp if p[0] == "fan")[3][1] for allp in (mid, calm)]
        self.assertGreater(neck[0], 1.3 * neck[1])
        self.assertGreater(fan[0], 1.5 * fan[1])
        self.assertAlmostEqual(walker.spin(start + 1e-6), 0.0, places=6)
        self.assertAlmostEqual(walker.spin(start + d - 1e-6), math.tau * n, places=4)
        turned = [walker.spin(t) for t in np.linspace(start, start + d - 1e-6, 200)]
        self.assertTrue(np.all(np.diff(turned) * np.sign(n) >= -1e-9))
        bows = [walker.pose(t).head[2] for t in np.linspace(start + 0.5, start + d - 0.5, 200)]
        self.assertGreater(max(bows) - min(bows), 0.06)
        for seed in range(20):
            walker = pigeons.Walker(np.random.default_rng(seed), 40.0, 0.4, 1.0, court=1.0, male=False)
            self.assertNotIn("court", [s[0] for s in walker.segs])

    def test_each_its_own_ways(self):
        """"Mỗi con một kiểu" 0: every bird as the settings say; more: each its own (busier or lazier, hungrier,
        fast or slow, resting more or less); about half of them males."""
        same = make({"amount": 100, "differ": 0})
        ways = np.array([[same.temper(i)[j] for j in (0, 1, 2, 4)] for i in range(len(same.u))])
        self.assertLess(float(ways.std(axis=0).max()), 1e-9)
        apart = make({"amount": 100, "differ": 100})
        ways = np.array([apart.temper(i)[:5] for i in range(len(apart.u))])
        self.assertGreater(float(ways[:, 0].std()), 0.15)  # (busy)
        self.assertGreater(float(ways[:, 1].std()), 0.15)  # (hungry)
        self.assertGreater(float(ways[:, 4].max() / ways[:, 4].min()), 1.5)  # (fast and slow)
        males = sum(apart.temper(i)[5] for i in range(len(apart.u)))
        self.assertTrue(0.2 * len(apart.u) < males < 0.8 * len(apart.u))

    def test_roam(self):
        """"Đi quanh quẩn ← → đi khắp khung": their paths small round their homes, or wide over the box."""
        spans = []
        for roam in (0, 100):
            fx = make({"amount": 60, "busy": 100, "roam": roam}, frames=720)
            spans.append(np.mean([(float(np.ptp(w.px)) + float(np.ptp(w.py))) * fx._scale(u, fx.box()[3])
                                  for w, u in zip(fx.walkers, fx.u)]))
        self.assertGreater(spans[1], 1.5 * spans[0])


def overlaps(fx, step=2):
    """How deep, at worst over the loop, two birds' footprints (ellipses round their bodies on the ground) overlap:
    units (a bird is LENGTH long)."""
    x0, y0, w, h = fx.box()
    st = math.sin(fx.tilt)
    worst = 0.0
    for t in range(0, fx.P, step):
        birds = fx._ground(t, x0, y0, w, h)
        if len(birds) < 2:
            return 0.0
        feet = np.array([(b[0], b[1] / st) for b in birds])
        size = np.array([b[2] for b in birds])
        way = np.array([(math.cos(b[3]), -math.sin(b[3])) for b in birds])
        mid = feet + (pigeons.FOOT[2] * size)[:, None] * way
        i, j = np.triu_indices(len(birds), 1)
        d = mid[j] - mid[i]
        dist = np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)
        nx, nz = d[:, 0] / dist, d[:, 1] / dist
        need = (pigeons._reach(size[i], way[i, 0], way[i, 1], nx, nz)
                + pigeons._reach(size[j], way[j, 0], way[j, 1], nx, nz))
        worst = max(worst, float(((need - dist) / (0.5 * (size[i] + size[j]))).max()))
    return worst


class Line:
    """A stand-in for a Walker: to and fro along a straight line (or standing still), round a loop."""

    def __init__(self, x, y, way, length, loop, stand=False):
        self.x, self.y, self.way, self.length, self.loop, self.stand = x, y, way, length, loop, stand
        self.speed = 2 * length / loop

    def where(self, t):
        if self.stand:
            return self.x, self.y, self.way
        f = (t / self.loop) % 1.0
        back = f >= 0.5
        along = self.length * ((2 * f if not back else 2 - 2 * f) - 0.5)
        return (self.x + math.cos(self.way) * along, self.y + math.sin(self.way) * along,
                self.way + (math.pi if back else 0.0))

    def pace_at(self, t):
        return 0.0 if self.stand else 1.0

    def samples(self, k):
        return np.array([(*self.where(t), self.pace_at(t)) for t in np.arange(k) * (self.loop / k)]).T


class Apart(unittest.TestCase):
    """The owner (2026-10-24): "các con chim bồ câu bị dính vào nhau, đi xuyên qua nhau". Each bird is placed where
    and when its path meets the others' least (_spread), and steps aside out of their way where they still meet
    (_plan); too many for the box, the birds it has no room for are left out."""

    def test_none_walk_into_another(self):
        for params in ({}, {"amount": 100}, {"amount": 100, "size": 70}, {"amount": 70, "size": 60, "roam": 100},
                       {"amount": 100, "size": 40, "roam": 0}, {"amount": 100, "busy": 100, "speed": 100},
                       {"mode": "ledge", "amount": 100}, {"mode": "ledge", "amount": 60, "size": 70}):
            for seed in (3, 11):
                fx = make(params, 480, seed=seed)
                self.assertLess(overlaps(fx), 0.05, (params, seed))

    def test_they_would_without(self):
        """Left on their own paths, the same birds walk into each other (so the test above tests something)."""
        with mock.patch.object(pigeons.Pigeons, "_spread", lambda self, w, h: np.arange(len(self.walkers))), \
                mock.patch.object(pigeons, "_plan", lambda walkers, *a: (None, np.zeros((len(walkers),) * 2))):
            pigeons._PLANS.clear()
            worst = max(overlaps(make({"amount": 100}, 480, seed=seed)) for seed in (3, 11))
        pigeons._PLANS.clear()
        self.assertGreater(worst, 0.3)

    def test_most_keep_their_own_places(self):
        """Few birds: each where its own numbers put it unless it would meet another there; with room, every bird
        asked for (as many as "Số chim" says)."""
        for amount in (20, 40, 70, 100):
            fx = make({"amount": amount}, 480)
            self.assertEqual(len(fx.walkers), pigeons.count("ground", amount), amount)

    def test_too_many_for_the_box(self):
        """Big birds, many of them: the box holds only so many without them walking into each other; those left
        out, the rest apart. On a ledge, each bird a stretch of its own as long as LEDGE at least."""
        for mode in MODES:
            fx = make({"mode": mode, "amount": 100, "size": 100}, 480)
            self.assertLess(len(fx.walkers), pigeons.CAP[mode], mode)
            self.assertGreater(len(fx.walkers), 0, mode)
            self.assertLess(overlaps(fx), 0.05, mode)
        fx = make({"mode": "ledge", "amount": 100, "size": 60}, 480)
        x0, y0, w, h = fx.box()
        slots = sorted(fx._slots(fx.u, w, h))
        self.assertGreaterEqual(min(s[1] for s in slots) * 2, pigeons.LEDGE * fx.size * h / pigeons.LENGTH - 1e-6)
        self.assertLessEqual(slots[-1][0] + slots[-1][1], w - 0.6 * fx.size * h + 1e-6)

    def test_standing_still_it_does_not_slide(self):
        """Stepping aside only as it walks: while a bird stands (looks round, pecks, sits, preens, courts) its feet
        stay put."""
        for params in ({"amount": 100, "size": 70}, {"amount": 70, "size": 85, "busy": 100}):
            fx = make(params, 480)
            self.assertIsNotNone(fx.plan, params)
            x0, y0, w, h = fx.box()
            before = None
            for t in range(fx.P + 1):
                now = fx._ground(t, x0, y0, w, h)
                if before is not None:
                    for walker, a, b in zip(fx.walkers, before, now):
                        if walker.pace_at((t - 1) / fx.fps) == 0 and walker.pace_at(t / fx.fps) == 0:
                            self.assertLess(math.hypot(b[0] - a[0], b[1] - a[1]), 0.002 * b[2], (params, t))
                before = now

    def test_steps_aside(self):
        """One walking through another standing still, two walking at each other, two crossing: each steps aside (no
        overlap left), and a standing bird hardly moves."""
        tilt, loop = math.radians(30), 20.0
        for walkers in ([Line(0, 0, 0.0, 4.0, loop), Line(0.0, 0.05, 0.0, 0, loop, stand=True)],
                        [Line(0, 0, 0.0, 4.0, loop), Line(0, 0, math.pi, 4.0, loop)],
                        [Line(0, 0, 0.0, 4.0, loop), Line(0, 0, math.pi / 2, 4.0, loop)]):
            scales = np.full(2, 100.0)
            plan, left = pigeons._plan(walkers, [(0.5, 0.5)] * 2, scales, (2000.0, 1000.0), tilt, loop)
            self.assertIsNotNone(plan)
            self.assertLess(float(left.max()), 0.02)
            self.assertLess(float(np.abs(plan).max()), 100.0)  # (aside, not away: under a unit)
            if walkers[1].stand:
                self.assertLess(float(np.ptp(plan[1], axis=0).max()), 0.5)  # (px)

    def test_the_same_each_time(self):
        """The flock (places, days, steps aside, birds kept) is the same whenever it is made: each clip of an export
        draws the same birds."""
        params = {"amount": 100, "size": 70}
        pigeons._PLANS.clear()
        a = make(params, 480)
        pigeons._PLANS.clear()
        b = make(params, 480)
        self.assertTrue(np.array_equal(a.u, b.u))
        self.assertEqual([w.shift for w in a.walkers], [w.shift for w in b.walkers])
        self.assertTrue(np.array_equal(a.plan, b.plan))
        c = make(params, 480)  # (made again from the kept layout)
        self.assertTrue(np.array_equal(a.plan, c.plan) and np.array_equal(a.u, c.u))
        self.assertEqual([(w.shift, w.room) for w in a.walkers], [(w.shift, w.room) for w in c.walkers])
        self.assertEqual([b[:4] for b in a._ground(77, *a.box())], [b[:4] for b in c._ground(77, *c.box())])

    def test_the_same_at_every_size(self):
        """Laid out in a box of one size (REF high), the flock is the same however big it is drawn: the preview clip
        and the video show the same birds in the same places, doing the same."""
        params = {"amount": 100, "size": 70}
        a, b = make(params, 480), make(params, 480, w=1280, h=720)
        self.assertTrue(np.array_equal(a.u, b.u) and np.array_equal(a.plan, b.plan))
        for t in (0, 101, 333):
            (ax, ay, _, ah), (bx, by, _, bh) = a.box(), b.box()
            fa = np.array([(f[0] - ax, f[1] - ay, f[3]) for f in a._ground(t, *a.box())]) / [ah, ah, 1]
            fb = np.array([(f[0] - bx, f[1] - by, f[3]) for f in b._ground(t, *b.box())]) / [bh, bh, 1]
            self.assertLess(float(np.abs(fa - fb).max()), 1e-9, t)

    def test_its_day_shifted(self):
        """A bird's day begun later in the loop (shift): where it is and what it does at t are its own at t + shift,
        round the loop; its samples the same, turned round."""
        walker = pigeons.Walker(np.random.default_rng(5), 20.0, 0.6, 1.0, hungry=0.5)
        plain = walker.samples(40)
        walker.shift = 20.0 * 7 / 40
        for t in (0.0, 3.3, 15.0, 19.9):
            self.assertEqual(walker.where(t), walker._where((t + walker.shift) % 20.0))
            self.assertEqual(walker.pace_at(t), walker._pace((t + walker.shift) % 20.0))
        self.assertTrue(np.allclose(walker.samples(40), np.roll(plain, -7, axis=1)))


class Body(unittest.TestCase):
    """Its body is one smooth shape: its parts melted into each other where they meet."""

    def test_just_outside_its_parts(self):
        """Each ray meets the smooth body no later than the plain union of its parts, on its surface, within NEAR of
        a part; its parts' shares there make up all of it."""
        allp = pigeons.parts(pigeons.Pose(head_yaw=0.4))
        body = allp[:pigeons.SMOOTH]
        g = np.random.default_rng(1)
        d = np.array([0.3, 0.9, -0.3])
        d /= np.linalg.norm(d)
        o = g.uniform(-0.6, 0.6, (4000, 3)) * np.array([1.0, 0.5, 0.6]) + np.array([0.0, 0.0, 0.25]) - 2 * d
        rays = pigeons._Rays(o[:, 0], o[:, 1], o[:, 2], d)
        seen = pigeons._Body(rays, body)
        t = seen.march()
        both = np.isfinite(seen.union)
        self.assertGreater(int(both.sum()), 500)
        self.assertTrue(np.all(np.isfinite(t[both])))
        self.assertTrue(np.all(t[both] <= seen.union[both] + 1e-6))
        hit = np.flatnonzero(np.isfinite(t))
        d_hit, shares = seen.distance(hit, t[hit], shares=True)
        self.assertLess(float(np.abs(d_hit).max()), 2e-3)  # (on the surface)
        self.assertLess(float(np.abs(np.sum(shares, axis=0) - 1).max()), 1e-6)
        p = o[hit] + t[hit, None] * d
        nearest = np.min([pigeons._distance([((p - c) @ rot)[:, j] / r[j] for j in range(3)], np.zeros(3), r,
                                            np.zeros(len(p))) for _, c, rot, r in body], axis=0)
        self.assertLess(float(nearest.max()), pigeons.NEAR + 4e-3)  # (the distances are a bound, not exact)

    def test_no_seams(self):
        """Across its body (a white dove's: one colour; its feathers' rows left out) the picture changes smoothly where
        its parts meet: no seams, no creases."""
        with mock.patch.object(pigeons, "_feathers", lambda *a: None):
            self.seams()

    def seams(self):
        coat = pigeons.Coat("white", np.random.default_rng(0))
        pose = pigeons.Pose()
        allp = pigeons.parts(pose)
        body, small = allp[:pigeons.SMOOTH], allp[pigeons.SMOOTH:]
        tilt, scale = math.radians(18), 200.0
        fwd, up = np.array([0.0, math.cos(tilt), -math.sin(tilt)]), np.array([0.0, math.sin(tilt), math.cos(tilt)])
        worst = 0.0
        for v in np.linspace(0.2, 0.36, 12):  # (rows across the breast, the body, the wing and the neck's foot:
            # where they melt into each other, away from its outline)
            u = np.arange(-0.2, 0.12, 1 / scale)
            o = u[:, None] * np.array([1.0, 0.0, 0.0]) + v * up - 3.0 * fwd
            rays = pigeons._Rays(o[:, 0], o[:, 1], o[:, 2], fwd)
            which, best, own, seen = pigeons._trace(rays, body, small)
            i, rgb = pigeons._shade(rays, which, best, own, seen, small, coat, pose, light(), np.eye(3), 1 / scale)
            on = which[i] == 0
            lum = rgb[on] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
            near = np.diff(i[on]) == 1  # (neighbours along the row, both on the body)
            run = np.zeros_like(near)  # (and not at its outline, where the light grazes it: on the body each side)
            run[1:-1] = near[:-2] & near[1:-1] & near[2:]
            if run.any():
                worst = max(worst, float(np.abs(np.diff(lum))[run].max()))
        self.assertLess(worst, 0.07)

    def test_drawn_to_a_fraction_of_a_pixel(self):
        """A bird is drawn where it is to a fraction of a pixel: moved by part of a pixel, its picture moves as much
        (it glides, not jumping a whole pixel at a time)."""
        coat = pigeons.Coat("grey", np.random.default_rng(1))

        def middle(at):
            rgb, a, feet = pigeons.draw_bird(pigeons.Pose(), coat, 0.3, 120.0, 0.3, light(), 2, at)
            ys, xs = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            return np.array([(xs * a).sum(), (ys * a).sum()]) / a.sum() - np.array(feet)
        still = middle((0.0, 0.0))
        for at in ((0.25, 0.0), (0.5, 0.5), (0.0, 0.75)):
            moved = middle(at) - still
            self.assertLess(float(np.abs(moved - np.array(at)).max()), 0.1, (at, moved))

    def test_creases_shaded(self):
        """Less of the light round it reaches into the creases of its body (where its folded wing lies on its
        side, under its neck) than onto its open back (ambient occlusion from its smooth body)."""
        body = pigeons.parts(pigeons.Pose())[:pigeons.SMOOTH]
        down = np.array([0.0, 0.0, -1.0])
        top = pigeons._Rays(np.array([-0.06]), np.array([0.0]), np.array([1.0]), down)
        t = pigeons._Body(top, body).march()
        p = np.array([[-0.06, 0.0, 1.0 - t[0]]])
        self.assertGreater(float(pigeons._occlusion(p, -down[None, :], body)[0]), 0.95)  # (its back: open)
        across = np.array([0.0, -1.0, 0.0])  # (rays onto its left side, all down it)
        zs = np.linspace(0.12, 0.38, 60)
        xs = np.linspace(-0.25, 0.15, 40)
        gx, gz = np.meshgrid(xs, zs)
        rays = pigeons._Rays(gx.ravel(), np.full(gx.size, 1.0), gz.ravel(), across)
        seen = pigeons._Body(rays, body)
        t = seen.march()
        hit = np.flatnonzero(np.isfinite(t))
        p = np.stack([rays.ox[hit], rays.oy[hit], rays.oz[hit]], 1) + t[hit, None] * across
        n = np.array([pigeons._sdf(p + e, body) - pigeons._sdf(p - e, body) for e in np.eye(3) * 1e-4]).T
        n /= np.linalg.norm(n, axis=1, keepdims=True)
        self.assertLess(float(pigeons._occlusion(p, n, body).min()), 0.85)  # (somewhere down its side: a crease)

    def test_its_feathers(self):
        """Rows of feathers on its folded wing, seen on a big bird (each a little lighter toward its tip, the next
        one's base in its shade); on a small one, faded out (they would only shimmer)."""
        x = np.linspace(-0.2, 0.05, 400)
        b = np.full_like(x, 0.05)
        big = pigeons._feathers(-x, b, 0.03, 0.7, 0.0015)
        self.assertIsNotNone(big)
        light = big[0]
        self.assertGreater(float(light.max() - light.min()), 0.1)
        self.assertGreater(int((np.abs(np.diff(light)) > 0.05).sum()), 4)  # (a tip's edge every feather)
        self.assertIsNone(pigeons._feathers(-x, b, 0.03, 0.7, 0.02))  # (a pixel as big as a feather)

    def test_soft_as_a_photograph(self):
        """"Độ mềm": the birds' edges softer as it rises (the same amount of bird)."""
        def edges(soft):
            fx = make({"soft": soft, "amount": 30, "size": 70, **DAY}, w=960, h=540)
            a = drawn(fx, 12)[1]
            return float(((a > 0.02) & (a < 0.98)).sum()), float(a.sum())
        sharp, soft = edges(0), edges(100)
        self.assertGreater(soft[0], sharp[0] * 1.5)
        self.assertLess(abs(soft[1] - sharp[1]) / sharp[1], 0.08)

    def test_its_shadow(self):
        """"Bóng dưới chân": the ground darker round each bird's feet (only darker), none at 0."""
        pic = np.full((540, 960, 3), 0.6, np.float32)

        def frame(shadow):
            fx = make({"shadow": shadow, "amount": 30, "size": 65, **DAY}, w=960, h=540)
            canvas = effects.Canvas(960, 540)
            fx.draw(12, canvas, None)
            a = canvas.a.copy()  # (before it is laid: laying clears the canvas)
            out = pic.copy()
            canvas.flush(out, fx.blend)
            x0, y0, w, h = fx.box()
            return out, a, fx._ground(12, x0, y0, w, h)
        bare, a, birds = frame(0)
        self.assertLess(float(np.abs(bare - 0.6)[a < 1e-6].max()), 1e-6)  # (none: the ground as it was)
        dark, _, _ = frame(100)
        self.assertGreaterEqual(float((bare - dark).min()), -1e-6)  # (only darker)
        darker = [float((bare - dark)[int(b[1]) - 3:int(b[1]) + 4, int(b[0]) - 6:int(b[0]) + 7].max()) for b in birds]
        self.assertGreater(float(np.median(darker)), 0.1)


class Feathers(unittest.TestCase):
    """Its feathers' patterns, as on the owner's photograph of a rock dove."""

    @staticmethod
    def luma(rgb):
        return rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    def wing(self, coat, a, up, b=1.0):
        """The folded wing's colour at points of it (a: +1 its front .. -1 its rear; up: -1 its lower edge .. +1 its
        top; b: +1 its outer face), its rows of feathers left out."""
        a, up = np.broadcast_arrays(np.asarray(a, np.float64), np.asarray(up, np.float64))
        with mock.patch.object(pigeons, "_feathers", lambda *_: None):
            return pigeons._wing_colour(a.ravel(), np.full(a.size, b), up.ravel(), pigeons.WING, coat.c, coat, 1e-4)

    def test_wing_bars(self):
        """Two black bars across its outer face where the photograph has them, curving forward toward its lower edge,
        a pale tip behind each; none round its inner face; a white dove's wing plain."""
        grey = pigeons.Coat("grey", np.random.default_rng(0))
        dark = lambda a, up, b=1.0: float(self.luma(self.wing(grey, a, up, b))[0])  # noqa: E731
        plain = dark(0.1, 0.3)
        self.assertGreater(plain, 0.3)
        for a0, _, _ in pigeons.BARS:
            self.assertLess(dark(a0, 0.3), 0.1 * plain)  # (black)
            low = a0 + pigeons.CURVE * 0.7 ** 2  # (toward the lower edge: further forward)
            self.assertLess(dark(low, -0.7), 0.1 * plain)
            self.assertGreater(dark(low, 0.3), 0.6 * plain)
            behind = np.linspace(a0 - 0.2, a0 - 0.05, 16)  # (its feathers' pale tips behind it)
            self.assertGreater(float(self.luma(self.wing(grey, behind, 0.3)).max()), 1.05 * plain)
            self.assertGreater(dark(a0, 0.3, -1.0), 0.6 * plain)  # (not round its inner face)
        white = pigeons.Coat("white", np.random.default_rng(0))
        across = self.luma(self.wing(white, np.linspace(0.3, -0.7, 200), 0.3))
        self.assertLess(float(np.ptp(across)), 0.1 * float(across.mean()))

    def test_its_neck(self):
        """Its neck green seen straight on, its throat purpler than its nape, its top the head's slate (no line where
        the head begins); its small feathers streaked with light when big enough to see."""
        coat = pigeons.Coat("grey", np.random.default_rng(0))
        c = coat.c
        rad = np.array([0.072, 0.064, 0.16])

        def colour(x, z, facing=1.0, fp=1e-4, feathers=False):
            local = np.array([[x, (1 - x * x - z * z) ** 0.5 if x * x + z * z < 1 else 0.0, z]])
            q = np.zeros((1, 3))
            if feathers:
                return pigeons._part_colour("neck", q, local, rad, np.zeros(1), np.full(1, facing), np.zeros(1), c,
                                            coat, fp)[0][0]
            with mock.patch.object(pigeons, "_feathers", lambda *_: None):
                return pigeons._part_colour("neck", q, local, rad, np.zeros(1), np.full(1, facing), np.zeros(1), c,
                                            coat, fp)[0][0]
        mid = colour(0.0, 0.3)
        self.assertGreater(mid[1], 1.3 * mid[0])  # (green)
        self.assertGreater(mid[1], mid[2])
        purple = lambda rgb: float((rgb[0] + rgb[2]) / 2 - rgb[1])  # noqa: E731
        self.assertGreater(purple(colour(0.9, -0.1)), purple(colour(-0.9, -0.1)) + 0.005)  # (its throat purpler)
        self.assertLess(float(np.abs(colour(0.0, 0.98) - c["head"]).max()), 0.02)  # (its top: the head's colour)
        tops = np.array([colour(0.0, z) for z in np.linspace(0.4, 0.98, 40)])
        self.assertLess(float(np.abs(np.diff(self.luma(tops))).max()), 0.02)  # (no line)
        zs = np.linspace(-0.2, 0.3, 400)
        streaks = self.luma(np.array([colour(0.0, z, feathers=True) for z in zs]))
        self.assertGreater(float(np.ptp(streaks)), 0.25 * float(streaks.mean()))
        far = self.luma(np.array([colour(0.0, z, fp=0.01, feathers=True) for z in zs]))  # (a small bird: none)
        self.assertLess(float(np.ptp(far)), float(np.ptp(streaks)) / 2)

    def test_checker_wedges(self):
        """A checker's mark on each covert is a dark wedge widening toward its tip."""
        size, aspect = 0.03, 0.7
        frs = np.linspace(0.02, 0.98, 49)

        def mark(t):  # (across one feather of the first row, t of the way from its base to its tip)
            b = frs * aspect * size
            tip = 0.3 * (1 - (2 * frs - 1) ** 2)
            return pigeons._feathers((t + tip) * size, b, size, aspect, 1e-5)[3]
        widths = [float(mark(t).mean()) for t in (0.15, 0.4, 0.8)]
        self.assertLess(widths[0], widths[1])
        self.assertLess(widths[1], widths[2])
        self.assertGreater(float(mark(0.8)[24]), 0.9)  # (dark in the middle near its tip)
        self.assertLess(float(mark(0.97).max()), 0.1)  # (its very tip pale)


class Picture(unittest.TestCase):
    def test_only_in_its_box(self):
        """Nothing outside the box (its preview clip is the box alone): the birds walk in it."""
        for mode in MODES:
            for params in ({}, {"amount": 100, "size": 100, "depth": 100, "busy": 100, "speed": 100},
                           {"amount": 100, "size": 100, "x": 30, "y": -20, "scale": 40}):
                fx = make({"mode": mode, **params})
                x0, y0, w, h = fx.box()
                outside = np.ones((H, W), bool)  # (the pixels the box's edges cross are partly inside)
                outside[max(0, math.floor(y0)):math.ceil(y0 + h), max(0, math.floor(x0)):math.ceil(x0 + w)] = False
                for t in (0, 50, 133, 201):
                    self.assertEqual(float(drawn(fx, t)[1][outside].max(initial=0.0)), 0.0, (mode, params, t))

    def test_how_many(self):
        for mode in MODES:
            self.assertEqual(len(make({"mode": mode, "amount": 0}).u), 0)
            self.assertEqual(float(drawn(make({"mode": mode, "amount": 0}), 5)[1].max()), 0.0)
            counts = [len(make({"mode": mode, "amount": a, "size": 30}).u) for a in (10, 35, 80, 100)]
            self.assertEqual(counts, sorted(counts), mode)
            self.assertGreater(counts[0], 0, mode)
            self.assertEqual(counts[-1], pigeons.CAP[mode], mode)

    def test_their_coats(self):
        """A white dove is pale, a dark one dark, a brown one reddish, a rock dove grey with a green neck; a checker
        darker than a plain grey one, a pied one paler."""
        def colour(coat):
            fx = make({"coat": coat, "amount": 60, "size": 70, **DAY}, w=960, h=540)
            rgb, a = drawn(fx, 0)
            return rgb.reshape(-1, 3).sum(axis=0) / max(float(a.sum()), 1e-9)
        white, grey, dark, brown, checker, pied = (colour(c) for c in ("white", "grey", "dark", "brown", "checker",
                                                                        "pied"))
        luma = np.array([0.2126, 0.7152, 0.0722])
        self.assertGreater(white @ luma, grey @ luma)
        self.assertGreater(grey @ luma, dark @ luma)
        self.assertGreater(brown[0], brown[2] * 1.1)
        self.assertLess(abs(grey[0] - grey[2]), 0.12)  # (grey: not red, not blue)
        self.assertLess(checker @ luma, grey @ luma)
        self.assertGreater(pied @ luma, grey @ luma)
        coat = pigeons.Coat("grey", np.random.default_rng(0))
        self.assertGreater(coat.c["green"][1], coat.c["green"][0] * 1.3)  # (its sheen)

    def test_lit_by_the_scene(self):
        """In a dark scene the birds are darker, in a warm one warmer; with "Sáng theo cảnh" 0 the scene does not
        light them."""
        dark = np.array([0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.0], np.float32)
        warm = np.array([1.0, 0.6, 0.3, 0.5, 0.35, 0.2, 0.0], np.float32)

        def colour(params, light):
            fx = make({"mode": "ground", "amount": 60, "size": 65, **params}, w=960, h=540, light=light)
            rgb, a = drawn(fx, 30)
            return rgb.reshape(-1, 3).sum(axis=0) / max(float(a.sum()), 1e-9)
        day = colour(DAY, None)
        self.assertLess(colour({"lit": 100}, dark).sum(), day.sum() * 0.3)
        hot = colour({"lit": 100}, warm)
        self.assertGreater(hot[0] / hot[2], day[0] / day[2] * 1.3)
        self.assertLess(float(np.abs(colour(DAY, dark) - day).max()), 1e-6)

    def test_far_ones_smaller(self):
        """On the ground, the farther up the box (the smaller their feet's y) the smaller ("Gần xa"); with no depth,
        all alike."""
        for depth, smaller in ((100, True), (0, False)):
            fx = make({"amount": 100, "depth": depth})
            x0, y0, w, h = fx.box()
            birds = fx._ground(0, x0, y0, w, h)
            ys, scales = np.array([b[1] for b in birds]), np.array([b[2] for b in birds])
            if smaller:
                self.assertGreater(float(np.corrcoef(ys, scales)[0, 1]), 0.8)
                self.assertGreater(float(scales.max() / scales.min()), 1.8)
            else:
                self.assertLess(float(np.ptp(scales)), 1e-9)

    def test_a_ledge_is_a_line(self):
        """On a ledge they stand on one line near the box's bottom edge, walking to and fro along it."""
        fx = make({"mode": "ledge", "amount": 100})
        x0, y0, w, h = fx.box()
        for t in (0, 77, 160):
            feet = np.array([b[1] for b in fx._ground(t, x0, y0, w, h)])
            self.assertLess(float(np.ptp(feet)), 0.04 * h, t)
            self.assertGreater(float(feet.min()), y0 + 0.9 * h, t)


class Preview(unittest.TestCase):
    def test_preview_clip_is_the_box_alone(self):
        """The preview clip (the box alone, at the clip's size) is the box in the whole frame, at another size (as
        soft: the softening is a share of a bird's length)."""
        from PIL import Image
        info = {"w": 1920, "h": 1080, "fps": "24", "frames": 240}
        for mode in MODES:
            params = {**AT, "mode": mode, "amount": 50, **DAY}
            layer = {"kind": "pigeons", "params": params}
            cw, ch = engine.fx_clip_frame(info, "pigeons", layer)
            self.assertAlmostEqual(cw / ch, 16 / 9, places=1)
            whole = make(params, 240, 24, 1920, 1080)
            alone = effects.make("pigeons", engine.fx_clip_params("pigeons", params), 3, 240, 24, cw, ch)
            x0, y0, bw, bh = whole.box()
            for t in (0, 77):
                big, clip = drawn(whole, t)[1], drawn(alone, t)[1]
                part = np.asarray(Image.fromarray((big * 255 + 0.5).astype(np.uint8)).resize(
                    (cw, ch), Image.Resampling.BOX, box=(x0, y0, x0 + bw, y0 + bh)), np.float32) / 255
                self.assertLess(float(np.abs(part - clip).mean()), 0.004, (mode, t))
                self.assertGreater(float(clip.mean()), 0.003, (mode, t))

    def test_drawn_over_the_picture(self):
        cls = pigeons.Pigeons
        self.assertEqual((cls.blend, cls.lens, cls.TURNS, cls.BOX, cls.LIT), ("over", False, False, (16, 9), True))
        project = {"video": {"w": 1920, "h": 1080, "fps": "24", "frames": 144, "seconds": 6.0}, "length": 30}
        a = engine.fx_clip_name(project, {"kind": "pigeons", "params": DAY})
        self.assertTrue(a.endswith(".webm"))  # (with see-through, laid Normal)
        self.assertEqual(a, engine.fx_clip_name(project, {"kind": "pigeons", "params": {**DAY, "x": 12}}))  # (the box
        # alone)


class Settings(unittest.TestCase):
    def test_every_setting_has_a_heading_and_label(self):
        keys = [p["key"] for p in pigeons.SCHEMA["params"]]
        self.assertEqual(len(keys), len(set(keys)))
        for p in pigeons.SCHEMA["params"]:
            self.assertTrue(p.get("group") and p["label"], p["key"])
            self.assertIsNone(p.get("show"), p["key"])  # (every setting counts on the ground and on a ledge)
        self.assertFalse(set(overlay.ROTATIONS) & set(keys))
        self.assertTrue({"peck", "soft", "busy", "view", "shadow", "rest", "court", "roam", "differ"} <= set(keys))
        self.assertTrue(pigeons.SCHEMA["note"])
        self.assertEqual(effects.defaults("pigeons")["mode"], "ground")
        modes = [o["value"] for o in next(p for p in pigeons.SCHEMA["params"] if p["key"] == "mode")["options"]]
        self.assertEqual(sorted(modes), sorted(MODES))

    def test_birds_of_the_air_come_down(self):
        """A clip of 1.0.71 flying in a flock or past is a clip on the ground now (the owner only needs them there)."""
        for old in ({"mode": "flock", "way": "left"}, {"mode": "pass"}):
            p = effects.clean_params("pigeons", old)
            self.assertEqual(p["mode"], "ground")
            self.assertNotIn("way", p)
            self.assertGreater(len(make(old).u), 0)

    def test_coats_and_views(self):
        """Each coat a picture of its own; the view from above changes how they are seen."""
        a = drawn(make({"coat": "white", **DAY}), 10)[0]
        b = drawn(make({"coat": "dark", **DAY}), 10)[0]
        self.assertGreater(float(np.abs(a - b).max()), 0.3)
        low = drawn(make({"view": 0, **DAY}), 10)[1]
        high = drawn(make({"view": 100, **DAY}), 10)[1]
        self.assertGreater(float(np.abs(low - high).max()), 0.3)


if __name__ == "__main__":
    unittest.main()
