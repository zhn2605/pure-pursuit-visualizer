"""Track geometry: control points, natural cubic spline smoothing, resampling.

A :class:`Track` is an ordered polyline the car tries to follow. It owns its
start and end points explicitly so the competition can pin a run to a fixed
start gate and finish marker regardless of the equation a player supplies.
"""

import numpy as np

from safe_expr import ExpressionError, compile_equation


def _natural_cubic_moments(t, y):
    """Second derivatives of the natural cubic spline through ``(t, y)``.

    Solves the standard tridiagonal system with the Thomas algorithm and the
    natural end conditions M[0] = M[n] = 0.
    """
    n = len(t) - 1
    if n < 2:
        return np.zeros(n + 1)

    h = np.diff(t)
    sub = h[:-1]                       # a_i, length n-1
    diag = 2.0 * (h[:-1] + h[1:])      # b_i
    sup = h[1:]                        # c_i
    rhs = 6.0 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])

    m = n - 1
    cp = np.zeros(m)
    dp = np.zeros(m)
    cp[0] = sup[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, m):
        denom = diag[i] - sub[i] * cp[i - 1]
        cp[i] = sup[i] / denom
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / denom

    moments = np.zeros(n + 1)
    moments[n - 1] = dp[-1]
    for i in range(m - 2, -1, -1):
        moments[i + 1] = dp[i] - cp[i] * moments[i + 2]
    return moments


def _spline_eval(t, y, moments, tq):
    """Evaluate the cubic spline defined by ``moments`` at query points ``tq``."""
    idx = np.clip(np.searchsorted(t, tq) - 1, 0, len(t) - 2)
    h = t[idx + 1] - t[idx]
    d = tq - t[idx]
    a = y[idx]
    b = (y[idx + 1] - y[idx]) / h - h * (2.0 * moments[idx] + moments[idx + 1]) / 6.0
    c = moments[idx] / 2.0
    e = (moments[idx + 1] - moments[idx]) / (6.0 * h)
    return a + b * d + c * d * d + e * d * d * d


class Track:
    """An ordered sequence of waypoints for the car to pursue."""

    def __init__(self, points=None, color="k"):
        self._points = []
        self._array = None
        self.color = color
        if points is not None:
            for point in points:
                self.add_point(point)

    # -- construction ----------------------------------------------------

    def add_point(self, point):
        self._points.append((float(point[0]), float(point[1])))
        self._array = None

    @classmethod
    def from_equation(cls, equation, xmin, xmax, points=100, y_clamp=None):
        """Build a track from ``y = f(x)`` sampled uniformly over ``[xmin, xmax]``.

        ``y_clamp`` is an optional ``(lo, hi)`` pair; values outside it are
        clamped rather than dropped so a wild equation visibly runs into the
        arena wall instead of silently leaving a gap in the path.
        """
        f = compile_equation(equation)
        xs = np.linspace(float(xmin), float(xmax), int(points))
        ys = f(xs)

        finite = np.isfinite(ys)
        if not np.any(finite):
            raise ExpressionError("Equation produces no usable points over this range.")
        if not np.all(finite):
            # Bridge singularities (e.g. tan) rather than rejecting the run.
            ys = np.interp(xs, xs[finite], ys[finite])
        if y_clamp is not None:
            ys = np.clip(ys, y_clamp[0], y_clamp[1])

        return cls(np.column_stack((xs, ys)))

    # -- access ----------------------------------------------------------

    @property
    def points(self):
        """``(N, 2)`` float array of waypoints. Cached until the track changes."""
        if self._array is None:
            self._array = np.asarray(self._points, dtype=float).reshape(-1, 2)
        return self._array

    @property
    def xs(self):
        return self.points[:, 0]

    @property
    def ys(self):
        return self.points[:, 1]

    @property
    def start(self):
        return self.points[0].copy()

    @property
    def end(self):
        return self.points[-1].copy()

    def __len__(self):
        return len(self._points)

    def start_heading(self):
        """Tangent angle at the first waypoint, for aligning the car."""
        pts = self.points
        if len(pts) < 2:
            return 0.0
        delta = pts[1] - pts[0]
        return float(np.arctan2(delta[1], delta[0]))

    def arc_length(self):
        """Total polyline length."""
        pts = self.points
        if len(pts) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def with_endpoints(self, start=None, end=None):
        """Return a copy pinned to the given start and/or end points.

        Duplicate-adjacent points are dropped so the spline parameterisation
        stays strictly increasing.
        """
        pts = list(map(tuple, self.points))
        if start is not None:
            pts.insert(0, (float(start[0]), float(start[1])))
        if end is not None:
            pts.append((float(end[0]), float(end[1])))
        return Track(pts, color=self.color)

    # -- smoothing -------------------------------------------------------

    def spline(self, samples=400):
        """Return a new Track: a natural cubic spline resampled by arc length.

        The spline is parameterised by cumulative chord length rather than by
        x, so it handles vertical segments and doublebacks that ``y = f(x)``
        sampling cannot. Output points are evenly spaced along the curve, which
        keeps pure pursuit's lookahead search well behaved.
        """
        pts = self.points
        if len(pts) < 3:
            return Track(pts, color=self.color)

        # Strictly increasing chord-length parameter; drop repeated points.
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        keep = np.concatenate(([True], seg > 1e-9))
        pts = pts[keep]
        if len(pts) < 3:
            return Track(pts, color=self.color)

        t = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
        mx = _natural_cubic_moments(t, pts[:, 0])
        my = _natural_cubic_moments(t, pts[:, 1])

        # Sample densely, then re-space uniformly along true arc length.
        dense_t = np.linspace(t[0], t[-1], max(samples * 8, 512))
        dense_x = _spline_eval(t, pts[:, 0], mx, dense_t)
        dense_y = _spline_eval(t, pts[:, 1], my, dense_t)

        dense = np.column_stack((dense_x, dense_y))
        seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(seg)))
        if cumulative[-1] <= 0:
            return Track(pts, color=self.color)

        targets = np.linspace(0.0, cumulative[-1], int(samples))
        out_x = np.interp(targets, cumulative, dense_x)
        out_y = np.interp(targets, cumulative, dense_y)
        return Track(np.column_stack((out_x, out_y)), color=self.color)

    def to_lists(self, decimals=3):
        """JSON-friendly ``{"xs": [...], "ys": [...]}`` for the browser."""
        pts = np.round(self.points, decimals)
        return {"xs": pts[:, 0].tolist(), "ys": pts[:, 1].tolist()}
