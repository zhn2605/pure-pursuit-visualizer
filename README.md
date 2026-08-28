# Pure Pursuit Visualizer

Simulates the pure pursuit path-tracking algorithm, with two front ends over a
shared physics core:

- **Competition mode** (`src/server.py`) — a localhost web app where players
  write a path equation, tune the controller, and race a fixed obstacle course
  for a spot on a persistent leaderboard.
- **Desktop mode** (`src/main.py`) — the original matplotlib animation, now
  able to render the obstacle course too.

![image](https://github.com/user-attachments/assets/4c2cb7ff-aab0-4666-9d2c-8bf1ab8584e0)

## Running the competition

```bash
python src/server.py            # then open http://127.0.0.1:8000
```

Stop it with **Ctrl-C** in the terminal it is running in. If it was started in
the background and you no longer have that terminal, the startup banner prints
its PID, or:

```bash
lsof -ti tcp:8000 | xargs kill     # stop whatever is serving port 8000
```

Starting a second server on a port already in use prints that same hint rather
than a bare traceback.

No dependencies beyond numpy — the server is pure standard library, so there
is nothing to install on the day. Useful flags:

```bash
python src/server.py --port 9000
python src/server.py --host 0.0.0.0   # expose to the local network
```

`--host 0.0.0.0` serves anyone who can reach the machine. It is fine on a
trusted network but the app has no authentication, so leave the default
`127.0.0.1` for a single kiosk laptop.

## How the game works

The arena is 60 x 32 units. Five walls alternate from the floor and ceiling and
run to the arena edge, so there is no way around the outside — the only route
is over, under, over, under, over.

A player supplies `y = f(x)`. That curve is sampled across the arena, smoothed
into a drivable path, and the car pure-pursues it from the left edge. The clock
stops when the front axle crosses the finish line at `x = 58`.

A run ends in one of five ways: `finished`, `crashed`, `out_of_bounds`,
`timeout`, or `invalid` (a path that starts inside an obstacle). **Every
attempt is saved and ranked**, finished or not. Repeat names are kept as
separate submissions by design.

### Collision

The car is a **2.5 x 1.2 rectangle** spanning rear axle to front axle, and
obstacles are rotated rectangles. Overlap is tested with the Separating Axis
Theorem: project both shapes onto each edge normal, and if any axis separates
them they are apart. This is exact for two convex polygons -- no margins, no
approximation -- and accounts for the rotation of *both* bodies.

The footprint drawn on screen is generated from the same rear/front/width
values the server collides, so **a crash always has a visible overlap**. An
earlier version modelled the car as a zero-width line segment and inflated each
obstacle by a hidden 0.4 margin to stand in for the car's width, which made
runs fail without appearing to touch anything.

The check runs once per 0.02 s physics step, with no sweep or prediction. At
the 30 u/s speed cap the car moves at most 0.6 units per step against obstacles
at least 2.2 units thick, so nothing tunnels through.

### Leaderboard order

Ranked by **completion** first, then **time**. Reaching the finish is exactly
100%; anything else is how far along the arena the car got. Unfinished runs
have no meaningful time and show `n/a`, so they sort after every finisher
rather than ahead of one.

### Tunable parameters

| Parameter | Range | Effect |
|---|---|---|
| Lookahead distance | 0.5 – 20 | How far ahead the car aims. Larger cuts corners harder and is faster, until it cuts straight into a wall. |
| Max speed | 1 – 30 | Speed ceiling on straights. |
| Max acceleration | 0.5 – 60 | How quickly the car reaches its target speed. |
| Start velocity | 0 – 30 | Speed at the start line. |
| Turn sensitivity | 0 – 20 | How much the car slows for curvature. `0` never lifts off. |
| Heading offset | -90° – 90° | Initial heading, relative to the path tangent. |

Out-of-range values are clamped rather than rejected, so a typo never costs
someone their turn.

### Equations

Whitelisted parser — user text is walked as an AST and never `eval`'d.
Available: `sin cos tan asin acos atan atan2 sinh cosh tanh exp log log10 sqrt
abs floor ceil round sign hypot mod min max clip where`, the constants `pi e
tau`, and comparisons (so `2*(x<5) - 2*(x>=5)` gives a piecewise path).

A fast line to beat: `5*cos(pi*(x-11.5)/10)` with lookahead 3, max speed 30,
turn sensitivity 0 finishes in about **2.6 s**. Nudging lookahead to 4 crashes
it — the quickest times sit right on the edge.

## Scoreboard

Lives in `data/` and survives restarts. Every attempt is stored, including
crashes. `scoreboard.json` is rewritten
atomically; `runs.jsonl` is an append-only log of every attempt written first,
and the board is rebuilt from it automatically if the JSON is ever lost or
corrupted. Delete both files to reset the competition. See `data/README.md`.

## Tests

```bash
python src/test.py          # backend: 34 tests, stdlib unittest, no pytest needed
node web/test_frontend.js   # frontend: playback timing
```

The Python suite covers the expression sandbox, spline, oriented-box collision
and rotation, vehicle model, game loop, and scoreboard persistence including
corruption recovery. The Node test covers playback frame indexing, which is
where a fractional array index once killed the animation silently.

## Layout

| File | Role |
|---|---|
| `src/purePursuit.py` | Lookahead search, curvature, heading error |
| `src/current.py` | Bicycle-model vehicle |
| `src/track.py` | Track geometry, natural cubic spline, arc-length resampling |
| `src/obstacles.py` | Rotated obstacles, car footprint, SAT collision, `Course` (**edit the layout here**) |
| `src/simulation.py` | Headless competition run loop |
| `src/safe_expr.py` | Sandboxed equation parser |
| `src/scoreboard.py` | Atomic JSON + append-only persistence |
| `src/server.py` | Stdlib HTTP server and JSON API |
| `src/history.py` | Matplotlib animation |
| `web/` | Canvas front end (matplotlib-style figure) |

## ToDo

- account for slipping / friction
- more courses, selectable from the UI
