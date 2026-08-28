"""Headless competition run: build a track, drive it, decide the outcome.

This is the module the web server calls. It deliberately contains no plotting
so a run costs milliseconds and can be replayed in the browser.
"""

from dataclasses import dataclass, asdict

import numpy as np

from current import Current
from obstacles import Course
from purePursuit import PurePursuit
from safe_expr import ExpressionError
from track import Track

PHYSICS_DT = 0.02
TRACK_SAMPLES = 500
EQUATION_SAMPLES = 160

# Tunable parameters exposed to players: (default, minimum, maximum).
#
# Only knobs that measurably change the outcome are exposed. Lookahead reshapes
# the driven line (~13 units of deviation across its range), heading changes
# both the line and whether a run survives, and max speed sets the clock.
PARAM_LIMITS = {
    "lookahead": (3.0, 0.5, 20.0),
    "max_speed": (12.0, 1.0, 30.0),
    "heading": (0.0, -90.0, 90.0),
}

# Held constant rather than exposed. The vehicle is a kinematic bicycle model:
# it integrates `theta += curvature * velocity * dt` while advancing
# `velocity * dt`, so heading change per unit *distance* is just curvature.
# With no slip or lateral-acceleration limit, speed cannot alter the geometry
# of the line -- these three moved the driven path by under 0.4 units across
# their entire ranges, so as knobs they only ever cost time.
#
# turn_sensitivity stays non-zero so curvature still costs speed: a tight,
# wiggly path is genuinely slower than a smooth one, which keeps path design
# meaningful. Exposing it just handed everyone the same optimum of 0.
FIXED_PARAMS = {
    "velocity": 2.0,
    "max_accel": 10.0,
    "turn_sensitivity": 3.0,
}

STATUS_MESSAGES = {
    "finished": "Finished!",
    "crashed": "Crashed into an obstacle.",
    "out_of_bounds": "Left the arena.",
    "timeout": "Ran out of time.",
    "invalid": "Could not start.",
}


def clamp_params(raw):
    """Coerce a raw request dict into safe numeric parameters.

    Out-of-range values are clamped rather than rejected so a slider or a
    typo never costs a player their turn at the booth.
    """
    params = dict(FIXED_PARAMS)
    for key, (default, low, high) in PARAM_LIMITS.items():
        value = raw.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = default
        if not np.isfinite(value):
            value = default
        params[key] = float(np.clip(value, low, high))
    return params


@dataclass
class RunResult:
    status: str
    message: str
    time: float
    max_x: float
    progress: float
    collided_with: str = ""

    def to_dict(self):
        return asdict(self)


def build_track(course, equation):
    """Sample ``equation`` across the arena and smooth it into a drivable path."""
    raw = Track.from_equation(
        equation,
        xmin=course.xmin,
        xmax=course.xmax,
        points=EQUATION_SAMPLES,
        y_clamp=course.y_clamp,
    )
    return raw.spline(samples=TRACK_SAMPLES)


def simulate(course, equation, params, record=True, dt=PHYSICS_DT):
    """Run one competition attempt.

    Returns ``(result, track, frames)``. ``frames`` is a dict of parallel
    arrays for playback, or None when ``record`` is False.
    """
    track = build_track(course, equation)

    car = Current(
        lookAheadDistance=params["lookahead"],
        velocity=params["velocity"],
        max_accel=params["max_accel"],
        turn_sensitivity=params["turn_sensitivity"],
        max_speed=params["max_speed"],
        min_speed=min(1.0, params["max_speed"]),
        dt=dt,
    )
    car.align_car(track, heading_offset=np.radians(params["heading"]))

    pursuit = PurePursuit()
    span = max(course.finish_x - car.position[0], 1e-9)

    frames = {k: [] for k in ("t", "x", "y", "fx", "fy", "lx", "ly", "v")} if record else None

    def snapshot(elapsed):
        if frames is None:
            return
        frames["t"].append(round(elapsed, 3))
        frames["x"].append(round(float(car.position[0]), 3))
        frames["y"].append(round(float(car.position[1]), 3))
        frames["fx"].append(round(float(car.front[0]), 3))
        frames["fy"].append(round(float(car.front[1]), 3))
        frames["lx"].append(round(float(car.lookAheadPosition[0]), 3))
        frames["ly"].append(round(float(car.lookAheadPosition[1]), 3))
        frames["v"].append(round(float(car.velocity), 3))

    def finish(status, elapsed, hit=""):
        max_x = max(float(car.front[0]), float(car.position[0]))
        progress = float(np.clip((max_x - track.start[0]) / span, 0.0, 1.0))
        return RunResult(
            status=status,
            message=STATUS_MESSAGES[status],
            time=round(elapsed, 3),
            max_x=round(max_x, 3),
            progress=round(progress, 4),
        collided_with=hit,
        )

    # Reject a start that is already illegal, rather than reporting a 0.00s
    # crash that looks like a bug to the player.
    hit = course.collision(car.position, car.front, car.width)
    if hit is not None:
        snapshot(0.0)
        result = finish("invalid", 0.0, hit.name)
        # Name what was hit: "an obstacle" is misleading when it is a wall.
        result.message = f"The car starts already touching the {hit.name}."
        return result, track, frames
    if not course.in_bounds(car.position, margin=0.5):
        snapshot(0.0)
        result = finish("invalid", 0.0)
        result.message = "Your path starts outside the arena."
        return result, track, frames

    elapsed = 0.0
    snapshot(elapsed)
    max_steps = int(course.max_time / dt) + 1

    for _ in range(max_steps):
        car.update(pursuit, track)
        elapsed += dt
        snapshot(elapsed)

        hit = course.collision(car.position, car.front, car.width)
        if hit is not None:
            return finish("crashed", elapsed, hit.name), track, frames

        if not course.in_bounds(car.front, margin=1.0) or \
           not course.in_bounds(car.position, margin=1.0):
            return finish("out_of_bounds", elapsed), track, frames

        if car.front[0] >= course.finish_x:
            return finish("finished", elapsed), track, frames

    return finish("timeout", elapsed), track, frames
