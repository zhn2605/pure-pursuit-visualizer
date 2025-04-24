import math
import numpy as np

from current import Current as Car
from purePursuit import PurePursuit
from history import History
from history import Track

# Initialize classes
history = History(debug=True)

# Create track as lambda functions
track1 = lambda x: -((x - 5)**2) + 25           # Concave-down curve
track2 = lambda x: math.sin(5*x)                # Basic sin wave
track3 = lambda x: math.sin(x/5.0) * x/2.0      # Old track
track4 = lambda x: 2 * (x < 5) - 2 * (x >= 5)   # Piecewise sharp-turnday, April 11th, 2025 

history.generate_track(xmin=0, xmax=50, points=100, function=track4)

car = Car(position=np.array([0.0, 0.0]), lookAheadDistance=1.0, velocity=1.0, max_accel=2.0)
pure_pursuit = PurePursuit()

history.animate(car, pure_pursuit)
