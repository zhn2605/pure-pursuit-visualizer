"""Self-check suite. Run before the showcase with:

    python src/test.py

Uses only the standard library plus numpy so it works on the booth laptop
with nothing extra installed.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from current import Current
from obstacles import (Course, Obstacle, body_corners, polygons_overlap,
                       scatter_course)
from purePursuit import PurePursuit
from safe_expr import (CONSTANTS, EXAMPLES, FUNCTIONS, ExpressionError,
                       compile_equation, evaluate)
from scoreboard import Scoreboard, clean_name
from simulation import FIXED_PARAMS, PARAM_LIMITS, clamp_params, simulate
from track import Track


def test_course():
    """A deliberately simple course owned by the tests.

    The showcase layout in obstacles.py is meant to be hand-edited, so no test
    may depend on which equations happen to solve it. This one has a single
    box straddling the centre line: y=0 must crash, y=10 must sail over.
    """
    return Course(
        name="Test",
        bounds=(0.0, 60.0, -16.0, 16.0),
        start_x=0.0,
        finish_x=58.0,
        obstacles=[Obstacle(30.0, 0.0, 6.0, 6.0, 0.0, name="centre-box")],
        default_equation="0",
        max_time=60.0,
    )


class TestSafeExpr(unittest.TestCase):
    def test_arithmetic_and_functions(self):
        x = np.linspace(0, 10, 5)
        np.testing.assert_allclose(evaluate("2*x + 1", x), 2 * x + 1)
        np.testing.assert_allclose(evaluate("sin(x)", x), np.sin(x))
        np.testing.assert_allclose(evaluate("sqrt(abs(x))", x), np.sqrt(np.abs(x)))

    def test_constant_broadcasts_to_input_shape(self):
        x = np.linspace(0, 1, 7)
        self.assertEqual(evaluate("5", x).shape, x.shape)

    def test_piecewise_via_comparison(self):
        x = np.array([0.0, 4.0, 6.0])
        np.testing.assert_allclose(evaluate("2*(x<5) - 2*(x>=5)", x), [2, 2, -2])

    def test_rejects_code_execution(self):
        attacks = [
            '__import__("os").system("id")',
            'open("/etc/passwd").read()',
            "(1).__class__.__bases__",
            "x.__class__",
            "[1, 2, 3]",
            "{'a': 1}",
            "lambda: 1",
            "x if x else 0",
        ]
        for source in attacks:
            with self.subTest(source=source):
                with self.assertRaises(ExpressionError):
                    evaluate(source, np.array([1.0]))

    def test_rejects_resource_exhaustion(self):
        with self.assertRaises(ExpressionError):
            evaluate("x**9999", np.array([2.0]))
        with self.assertRaises(ExpressionError):
            evaluate("sin(" * 40 + "x" + ")" * 40, np.array([1.0]))
        with self.assertRaises(ExpressionError):
            evaluate("1" + "+1" * 500, np.array([1.0]))

    def test_rejects_unknown_names(self):
        for source in ["y + 1", "foo(x)", "pi2"]:
            with self.subTest(source=source):
                with self.assertRaises(ExpressionError):
                    evaluate(source, np.array([1.0]))

    def test_empty_equation(self):
        with self.assertRaises(ExpressionError):
            evaluate("   ", np.array([1.0]))


class TestExamples(unittest.TestCase):
    """The preset buttons are generated from EXAMPLES, so a function without a
    worked example would silently vanish from the UI."""

    def test_every_function_and_constant_has_an_example(self):
        named = {name for name, _ in EXAMPLES}
        self.assertEqual(set(FUNCTIONS) - named, set(), "functions missing an example")
        self.assertEqual(set(CONSTANTS) - named, set(), "constants missing an example")

    def test_example_names_are_unique(self):
        names = [name for name, _ in EXAMPLES]
        self.assertEqual(len(names), len(set(names)))

    def test_every_example_parses_and_is_usable(self):
        xs = np.linspace(0, 60, 200)
        for name, equation in EXAMPLES:
            with self.subTest(name=name):
                ys = compile_equation(equation)(xs)
                self.assertEqual(ys.shape, xs.shape)
                finite = np.isfinite(ys)
                self.assertTrue(finite.any(), f"{name} produced nothing finite")
                if name != "flat":
                    span = np.ptp(ys[finite])
                    self.assertGreater(span, 0.5, f"{name} draws a flat line")

    def test_each_example_actually_uses_its_function(self):
        for name, equation in EXAMPLES:
            if name in ("flat",):
                continue
            with self.subTest(name=name):
                self.assertIn(name, equation,
                              f"the {name} example does not mention {name}")


class TestTrack(unittest.TestCase):
    def test_spline_preserves_a_straight_line(self):
        line = Track([(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]).spline(50)
        self.assertLess(float(np.abs(line.ys).max()), 1e-9)

    def test_spline_resamples_uniformly(self):
        track = Track.from_equation("8*cos(pi*(x-11.5)/10)", 0, 60, 60).spline(300)
        spacing = np.linalg.norm(np.diff(track.points, axis=0), axis=1)
        self.assertEqual(len(track), 300)
        self.assertLess(spacing.max() - spacing.min(), 1e-2)

    def test_spline_passes_through_control_points(self):
        pts = [(0, 0), (2, 3), (5, 1), (8, 4), (10, 0)]
        smooth = Track(pts).spline(600)
        for px, py in pts:
            distance = np.min(np.linalg.norm(smooth.points - np.array([px, py]), axis=1))
            self.assertLess(distance, 0.05, f"spline misses control point ({px}, {py})")

    def test_endpoints_and_heading(self):
        track = Track([(0, 0), (1, 1), (2, 2)])
        np.testing.assert_allclose(track.start, [0, 0])
        np.testing.assert_allclose(track.end, [2, 2])
        self.assertAlmostEqual(track.start_heading(), np.pi / 4)

    def test_with_endpoints_pins_start_and_end(self):
        pinned = Track([(1, 1), (2, 2)]).with_endpoints(start=(0, 0), end=(9, 9))
        np.testing.assert_allclose(pinned.start, [0, 0])
        np.testing.assert_allclose(pinned.end, [9, 9])

    def test_clamps_and_bridges_bad_values(self):
        track = Track.from_equation("100", 0, 10, 20, y_clamp=(-16, 16))
        self.assertLessEqual(float(track.ys.max()), 16.0)
        # tan() has poles; the track must still be finite everywhere.
        poles = Track.from_equation("tan(x)", 0, 10, 50, y_clamp=(-16, 16))
        self.assertTrue(np.all(np.isfinite(poles.points)))

    def test_arc_length(self):
        self.assertAlmostEqual(Track([(0, 0), (3, 4)]).arc_length(), 5.0)


class TestObstacle(unittest.TestCase):
    def setUp(self):
        self.rect = Obstacle.from_corner(0, 0, 10, 10)

    def test_segment_intersections(self):
        cases = [
            ((-5, 5), (5, 5), True),      # crosses the left edge
            ((-5, 5), (-1, 5), False),    # stops short
            ((5, 5), (6, 6), True),       # entirely inside
            ((-5, -5), (15, 15), True),   # diagonal straight through
            ((-5, 20), (15, 20), False),  # passes above
            ((5, -5), (5, 15), True),     # vertical through
            ((-1, -1), (-1, 11), False),  # vertical alongside
            ((-5, 0), (-5, 10), False),   # parallel and outside
        ]
        for p, q, expected in cases:
            with self.subTest(p=p, q=q):
                got = self.rect.intersects_segment(np.array(p, float), np.array(q, float))
                self.assertEqual(got, expected)

    def test_margin_inflates_the_box(self):
        p, q = np.array([-1.0, 5.0]), np.array([-0.5, 5.0])
        self.assertFalse(self.rect.intersects_segment(p, q))
        self.assertTrue(self.rect.intersects_segment(p, q, margin=1.0))

    def test_square_helper_and_validation(self):
        self.assertEqual(Obstacle.square(0, 0, 4).width, 4.0)
        with self.assertRaises(ValueError):
            Obstacle(0, 0, 0, 5)

    def test_course_collision_uses_the_real_footprint(self):
        course = test_course()          # one 6x6 box centred at (30, 0)
        width = 1.2
        # Nose inside the box: a hit.
        self.assertIsNotNone(
            course.collision(np.array([25.0, 0.0]), np.array([27.5, 0.0]), width))
        # Well clear above it: not a hit.
        self.assertIsNone(
            course.collision(np.array([28.0, 8.0]), np.array([30.5, 8.0]), width))

    def test_collision_boundary_matches_the_drawn_shape(self):
        """No crash without a visible overlap, and none missed once there is one.

        The box spans y -3..3; the car is 1.2 wide, so its edge is at
        centre +/- 0.6. Crossing must begin exactly at y = 3.6.
        """
        course = test_course()
        width = 1.2
        rear, front = np.array([29.0, 0.0]), np.array([31.5, 0.0])
        self.assertIsNotNone(course.collision(rear + [0, 3.55], front + [0, 3.55], width))
        self.assertIsNone(course.collision(rear + [0, 3.65], front + [0, 3.65], width))


class TestFootprint(unittest.TestCase):
    def test_footprint_is_a_rectangle_of_the_right_size(self):
        corners = body_corners(np.array([0.0, 0.0]), np.array([2.5, 0.0]), 1.2)
        self.assertEqual(corners.shape, (4, 2))
        sides = np.linalg.norm(np.diff(np.vstack([corners, corners[:1]]), axis=0), axis=1)
        np.testing.assert_allclose(sorted(sides), [1.2, 1.2, 2.5, 2.5], atol=1e-9)

    def test_footprint_follows_heading(self):
        # Pointing straight up: the rectangle is now tall, not wide.
        corners = body_corners(np.array([0.0, 0.0]), np.array([0.0, 2.5]), 1.2)
        np.testing.assert_allclose(corners[:, 0].min(), -0.6, atol=1e-9)
        np.testing.assert_allclose(corners[:, 1].max(), 2.5, atol=1e-9)

    def test_degenerate_footprint_does_not_explode(self):
        corners = body_corners(np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.2)
        self.assertTrue(np.all(np.isfinite(corners)))

    def test_sat_is_symmetric_and_exact(self):
        a = body_corners(np.array([0.0, 0.0]), np.array([2.5, 0.0]), 1.2)
        b = Obstacle(5.0, 0.0, 2.0, 2.0, 0.0).corners()
        self.assertFalse(polygons_overlap(a, b))
        self.assertFalse(polygons_overlap(b, a))
        c = Obstacle(3.0, 0.0, 2.0, 2.0, 0.0).corners()   # spans x 2..4
        self.assertTrue(polygons_overlap(a, c))
        self.assertTrue(polygons_overlap(c, a))

    def test_rotating_an_obstacle_changes_the_outcome(self):
        # A thin bar the car clears while it lies flat (y 2.5..3.5, above the
        # car's 1.2-wide body), but not once spun upright (y 0..6, x 1.5..2.5).
        car = body_corners(np.array([0.0, 0.0]), np.array([2.5, 0.0]), 1.2)
        flat = Obstacle(2.0, 3.0, 6.0, 1.0, 0.0)
        spun = Obstacle(2.0, 3.0, 6.0, 1.0, 90.0)
        self.assertFalse(flat.intersects_polygon(car))
        self.assertTrue(spun.intersects_polygon(car))


class TestWalls(unittest.TestCase):
    def setUp(self):
        self.course = test_course()
        self.width = 1.2

    def test_ceiling_and_floor_are_solid(self):
        top = self.course.ymax
        bottom = self.course.ymin
        # Footprint edge sits 0.6 from the axle line, so 0.5 clear still passes.
        self.assertIsNone(
            self.course.collision(np.array([5.0, top - 0.7]),
                                  np.array([7.5, top - 0.7]), self.width))
        hit = self.course.collision(np.array([5.0, top - 0.5]),
                                    np.array([7.5, top - 0.5]), self.width)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "top wall")

        hit = self.course.collision(np.array([5.0, bottom + 0.5]),
                                    np.array([7.5, bottom + 0.5]), self.width)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "bottom wall")

    def test_a_tilted_car_clips_the_ceiling_with_a_corner(self):
        # Level here is clear, but rotated the leading corner pokes through.
        y = self.course.ymax - 0.9
        level_rear, level_front = np.array([5.0, y]), np.array([7.5, y])
        self.assertIsNone(self.course.collision(level_rear, level_front, self.width))

        angle = np.radians(40)
        rear = np.array([5.0, y])
        front = rear + 2.5 * np.array([np.cos(angle), np.sin(angle)])
        hit = self.course.collision(rear, front, self.width)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.name, "top wall")

    def test_a_path_above_the_ceiling_never_starts(self):
        result, _, _ = simulate(self.course, "100", clamp_params({}), record=False)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.collided_with, "top wall")

    def test_climbing_into_the_ceiling_is_a_crash(self):
        # A steep ramp to the ceiling: the car overshoots the flattening path
        # and clips the wall. Pure pursuit recovers from a bad heading, so it
        # takes real momentum at the wall to hit it.
        result, _, _ = simulate(self.course, "min(3*x - 6, 15)",
                                clamp_params({}), record=False)
        self.assertEqual(result.status, "crashed")
        self.assertEqual(result.collided_with, "top wall")

    def test_diving_into_the_floor_is_a_crash(self):
        result, _, _ = simulate(self.course, "max(-4*x + 8, -15)",
                                clamp_params({"max_speed": 30}), record=False)
        self.assertEqual(result.status, "crashed")
        self.assertEqual(result.collided_with, "bottom wall")

    def test_a_heading_offset_alone_is_recovered_from(self):
        # The controller is closed-loop: a 90 degree error is steered out
        # rather than flying the car off the top of the arena.
        result, _, _ = simulate(self.course, "8",
                                clamp_params({"heading": 90}), record=False)
        self.assertEqual(result.status, "finished")

    def test_starting_through_a_wall_says_which_wall(self):
        # Nose pokes through the ceiling before the run begins.
        result, _, _ = simulate(self.course, "14",
                                clamp_params({"heading": 90}), record=False)
        self.assertEqual(result.status, "invalid")
        self.assertEqual(result.collided_with, "top wall")
        self.assertIn("top wall", result.message)


class TestParams(unittest.TestCase):
    def test_only_meaningful_knobs_are_exposed(self):
        self.assertEqual(set(PARAM_LIMITS), {"lookahead", "max_speed", "heading"})

    def test_removed_knobs_are_held_constant(self):
        self.assertEqual(set(FIXED_PARAMS),
                         {"velocity", "max_accel", "turn_sensitivity"})
        # Non-zero so curvature still costs speed and path shape still matters.
        self.assertGreater(FIXED_PARAMS["turn_sensitivity"], 0)

    def test_client_cannot_override_a_fixed_knob(self):
        params = clamp_params({"turn_sensitivity": 0.0, "max_accel": 60.0,
                               "velocity": 30.0})
        for key, value in FIXED_PARAMS.items():
            self.assertEqual(params[key], value, f"{key} was overridden")

    def test_simulate_still_receives_every_key_it_needs(self):
        params = clamp_params({})
        for key in ("lookahead", "velocity", "max_speed", "max_accel",
                    "heading", "turn_sensitivity"):
            self.assertIn(key, params)


class TestCar(unittest.TestCase):
    def test_front_axle_stays_consistent_with_pose(self):
        car = Current(velocity=3.0, dt=0.05)
        track = Track.from_equation("0", 0, 60, 50)
        car.align_car(track)
        pursuit = PurePursuit()
        for _ in range(50):
            car.update(pursuit, track)
            expected = car.position + car.wheelbase * np.array(
                [np.cos(car.theta), np.sin(car.theta)])
            np.testing.assert_allclose(car.front, expected, atol=1e-9)

    def test_does_not_mutate_caller_array(self):
        start = np.array([1.0, 2.0])
        car = Current(position=start, velocity=5.0)
        car.update_position()
        np.testing.assert_allclose(start, [1.0, 2.0])

    def test_steering_limit_is_in_radians(self):
        car = Current()
        self.assertAlmostEqual(car.max_steer, np.pi / 4)
        curvature_cap = np.tan(car.max_steer) / car.wheelbase
        self.assertAlmostEqual(curvature_cap, 1.0 / 2.5)

    def test_acceleration_is_clamped(self):
        car = Current(velocity=0.0, max_accel=2.0, dt=0.1)
        car.update_velocity(100.0)
        self.assertAlmostEqual(car.velocity, 0.2)


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.course = test_course()

    def test_good_line_finishes(self):
        result, _, frames = simulate(self.course, "10", clamp_params({}))
        self.assertEqual(result.status, "finished")
        self.assertGreater(result.time, 0)
        self.assertEqual(result.progress, 1.0)
        self.assertEqual(len(frames["t"]), len(frames["x"]))

    def test_line_through_a_box_crashes(self):
        result, _, _ = simulate(self.course, "0", clamp_params({}))
        self.assertEqual(result.status, "crashed")
        self.assertTrue(result.collided_with)
        self.assertLess(result.progress, 1.0)

    def test_faster_settings_give_a_better_time(self):
        slow, _, _ = simulate(self.course, "10", clamp_params({"max_speed": 8}))
        fast, _, _ = simulate(self.course, "10", clamp_params({"max_speed": 18}))
        self.assertEqual(slow.status, "finished")
        self.assertEqual(fast.status, "finished")
        self.assertLess(fast.time, slow.time)

    def test_run_is_deterministic(self):
        a, _, _ = simulate(self.course, "6*sin(x/9) + 10", clamp_params({}))
        b, _, _ = simulate(self.course, "6*sin(x/9) + 10", clamp_params({}))
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_params_are_clamped_not_rejected(self):
        params = clamp_params({"max_speed": 1e9, "lookahead": -50, "velocity": "abc"})
        self.assertEqual(params["max_speed"], 30.0)
        self.assertEqual(params["lookahead"], 0.5)
        self.assertEqual(params["velocity"], 2.0)

    def test_nan_params_fall_back_to_defaults(self):
        self.assertEqual(clamp_params({"max_accel": float("nan")})["max_accel"], 10.0)

    def test_never_exceeds_the_time_limit(self):
        result, _, _ = simulate(self.course, "0", clamp_params({}), record=False)
        self.assertLessEqual(result.time, self.course.max_time + 1e-6)


class TestScoreboard(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="pp-board-")
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.board = Scoreboard(self.directory)
        self.course = test_course()

    def _run(self, name, equation, **params):
        clamped = clamp_params(params)
        result, _, _ = simulate(self.course, equation, clamped, record=False)
        return self.board.record(name, result, equation, clamped), result

    def test_ranks_by_completion_then_time(self):
        self._run("Slow", "10", max_speed=8)
        fast, _ = self._run("Fast", "10", max_speed=20)
        crash, _ = self._run("Crasher", "0")

        board = self.board.ranked()
        # Finishers first by time, then unfinished runs by how far they got.
        self.assertEqual([row["name"] for row in board], ["Fast", "Slow", "Crasher"])
        self.assertEqual([row["rank"] for row in board], [1, 2, 3])
        self.assertEqual(self.board.rank_of(fast["id"]), 1)
        self.assertEqual(self.board.stats(),
                         {"attempts": 3, "finishes": 2, "players": 3})

    def test_failed_runs_are_saved_and_shown(self):
        entry, result = self._run("Crasher", "0")
        self.assertNotEqual(result.status, "finished")
        board = self.board.ranked()
        self.assertEqual(len(board), 1)
        self.assertEqual(board[0]["name"], "Crasher")
        # Reloading from disk must keep the failed run too.
        self.assertEqual(len(Scoreboard(self.directory).ranked()), 1)

    def test_finish_is_exactly_100_and_failures_have_no_time(self):
        self._run("Done", "10", max_speed=20)
        self._run("Nope", "0")
        board = {row["name"]: row for row in self.board.ranked()}
        self.assertEqual(board["Done"]["completion"], 100.0)
        self.assertIsNotNone(board["Done"]["time"])
        self.assertIsNone(board["Nope"]["time"], "unfinished runs must report no time")
        self.assertLess(board["Nope"]["completion"], 100.0)
        self.assertGreaterEqual(board["Nope"]["completion"], 0.0)

    def test_a_further_crash_outranks_a_shorter_one(self):
        near, _ = self._run("Near", "6*sin(x/40) + 9")
        short, _ = self._run("Short", "0")
        rows = {r["name"]: r for r in self.board.ranked()}
        if rows["Near"]["completion"] != rows["Short"]["completion"]:
            further = max(rows.values(), key=lambda r: r["completion"])["name"]
            self.assertEqual(self.board.ranked()[0]["name"], further)

    def test_legacy_entries_without_completion_still_rank(self):
        # Entries written before the completion field must not break the board.
        self.board._entries.append({
            "id": "legacy0000", "name": "Old", "status": "crashed",
            "time": None, "progress": 0.42, "equation": "0",
            "params": {}, "timestamp": 0,
        })
        row = [r for r in self.board.ranked() if r["name"] == "Old"][0]
        self.assertAlmostEqual(row["completion"], 42.0)

    def test_same_name_creates_separate_submissions(self):
        first, _ = self._run("Ada", "10", max_speed=10)
        second, _ = self._run("Ada", "10", max_speed=20)
        self.assertNotEqual(first["id"], second["id"])
        board = self.board.ranked()
        self.assertEqual(len(board), 2)
        self.assertEqual([row["name"] for row in board], ["Ada", "Ada"])
        self.assertEqual(self.board.stats()["players"], 1)

    def test_survives_a_restart(self):
        self._run("Ada", "10", max_speed=20)
        reloaded = Scoreboard(self.directory)
        self.assertEqual(len(reloaded.ranked()), 1)

    def test_recovers_from_a_corrupt_json_file(self):
        self._run("Ada", "10", max_speed=20)
        with open(os.path.join(self.directory, "scoreboard.json"), "w") as handle:
            handle.write("{ this is not json")
        recovered = Scoreboard(self.directory)
        self.assertEqual([row["name"] for row in recovered.ranked()], ["Ada"])

    def test_name_cleaning(self):
        self.assertEqual(clean_name("  Ada   Lovelace \x00"), "Ada Lovelace")
        self.assertEqual(len(clean_name("x" * 100)), 24)
        for bad in ["", "   ", "\x00"]:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    clean_name(bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
