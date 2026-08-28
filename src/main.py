"""Desktop matplotlib visualiser.

Run the browser competition instead with:  python src/server.py
"""

import math
import numpy as np

from current import Current as Car
from purePursuit import PurePursuit
from history import History
from track import Track
from obstacles import scatter_course

# Set to None for a plain track with no obstacles or finish line.
course = scatter_course()

history = History(debug=False, course=course)

if course is not None:
    # Same equation parser and spline the web UI uses, so a line tested here
    # behaves identically on the leaderboard.
    history.generate_track_from_equation("8*cos(pi*(x-11.5)/10)",
                                         xmin=course.xmin, xmax=course.xmax)
else:
    track1 = lambda x: -((x - 5)**2) + 25           # Concave-down curve
    track2 = lambda x: math.sin(5*x)                # Basic sin wave
    track3 = lambda x: math.sin(x/5.0) * x/2.0      # Old track
    track4 = lambda x: 2 * (x < 5) - 2 * (x >= 5)   # Piecewise sharp-turn

    history.generate_track(xmin=0, xmax=50, points=100, function=track3)

car = Car(
    lookAheadDistance=3.0,
    velocity=2.0,
    max_accel=10.0,
    turn_sensitivity=3.0,
    max_speed=12.0,
    dt=0.05,
    )
pure_pursuit = PurePursuit()

car.align_car(history.track)

history.animate(car, pure_pursuit, interval=20, max_time=60.0)
