"""Rotated rectangular obstacles and the collision test used for game over."""

import numpy as np


def body_corners(rear, front, width):
    """The car's footprint: a rectangle spanning rear axle to front axle.

    This is the actual collision shape, and it is exactly what gets drawn, so
    a run never fails without something visibly overlapping.
    """
    rear = np.asarray(rear, dtype=float)
    front = np.asarray(front, dtype=float)
    axis = front - rear
    length = np.linalg.norm(axis)
    if length < 1e-12:
        axis = np.array([1.0, 0.0])
    else:
        axis = axis / length
    normal = np.array([-axis[1], axis[0]]) * (width / 2.0)
    return np.array([rear + normal, front + normal, front - normal, rear - normal])


def _edge_normals(corners):
    edges = np.diff(np.vstack([corners, corners[:1]]), axis=0)
    return np.column_stack([-edges[:, 1], edges[:, 0]])


def polygons_overlap(a, b):
    """Separating Axis Theorem for two convex polygons.

    If any axis separates the projections, the shapes are apart. For two
    rectangles this is exact -- no margins, no approximation.
    """
    for axes in (_edge_normals(a), _edge_normals(b)):
        for axis in axes:
            norm = np.hypot(axis[0], axis[1])
            if norm < 1e-12:
                continue
            axis = axis / norm
            pa = a @ axis
            pb = b @ axis
            if pa.max() < pb.min() - 1e-9 or pb.max() < pa.min() - 1e-9:
                return False
    return True


class Obstacle:
    """An oriented rectangle the car must not touch.

    Defined by its centre, size and rotation. Rotation means the collision
    test is an oriented-box test: the car segment is transformed into the
    rectangle's local frame, where the problem reduces to an axis-aligned one.
    """

    def __init__(self, cx, cy, width, height, angle=0.0, name=""):
        if width <= 0 or height <= 0:
            raise ValueError("Obstacle width and height must be positive.")
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = float(width)
        self.height = float(height)
        self.angle = float(angle)          # degrees, counter-clockwise
        self.name = name

    @classmethod
    def square(cls, cx, cy, size, angle=0.0, name=""):
        return cls(cx, cy, size, size, angle=angle, name=name)

    @classmethod
    def from_corner(cls, x, y, width, height, angle=0.0, name=""):
        """Build from a bottom-left corner instead of a centre."""
        return cls(x + width / 2.0, y + height / 2.0, width, height,
                   angle=angle, name=name)

    @property
    def radians(self):
        return np.radians(self.angle)

    def corners(self):
        """The four corners in world space, counter-clockwise."""
        hw, hh = self.width / 2.0, self.height / 2.0
        local = np.array([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])
        c, s = np.cos(self.radians), np.sin(self.radians)
        rotation = np.array([[c, -s], [s, c]])
        return local @ rotation.T + np.array([self.cx, self.cy])

    def to_local(self, point):
        """Map a world point into the rectangle's own axis-aligned frame."""
        c, s = np.cos(-self.radians), np.sin(-self.radians)
        dx = float(point[0]) - self.cx
        dy = float(point[1]) - self.cy
        return (c * dx - s * dy, s * dx + c * dy)

    def contains(self, point, margin=0.0):
        lx, ly = self.to_local(point)
        return (abs(lx) <= self.width / 2.0 + margin
                and abs(ly) <= self.height / 2.0 + margin)

    def intersects_segment(self, p, q, margin=0.0):
        """Does the segment ``p -> q`` touch this rectangle?

        Both endpoints are rotated into the rectangle's local frame, then
        clipped against its slabs (Liang-Barsky). The car is modelled as the
        segment from its rear axle to its front axle, so this is the whole
        collision test; ``margin`` inflates the box by the car's half-width.
        """
        px, py = self.to_local(p)
        qx, qy = self.to_local(q)
        half = (self.width / 2.0 + margin, self.height / 2.0 + margin)
        origin = (px, py)
        direction = (qx - px, qy - py)

        t_enter, t_exit = 0.0, 1.0
        for axis in (0, 1):
            lo, hi = -half[axis], half[axis]
            start = origin[axis]
            delta = direction[axis]
            if abs(delta) < 1e-12:
                # Parallel to this slab: either within it, or a guaranteed miss.
                if start < lo or start > hi:
                    return False
                continue
            t_a = (lo - start) / delta
            t_b = (hi - start) / delta
            if t_a > t_b:
                t_a, t_b = t_b, t_a
            t_enter = max(t_enter, t_a)
            t_exit = min(t_exit, t_b)
            if t_enter > t_exit:
                return False
        return True

    def intersects_polygon(self, corners):
        """Exact overlap test against any convex polygon, rotation included."""
        return polygons_overlap(np.asarray(corners, dtype=float), self.corners())

    def to_dict(self):
        return {
            "cx": self.cx, "cy": self.cy,
            "width": self.width, "height": self.height,
            "angle": self.angle, "name": self.name,
            "corners": [[round(float(x), 3), round(float(y), 3)]
                        for x, y in self.corners()],
        }

    def __repr__(self):
        return (f"Obstacle(cx={self.cx}, cy={self.cy}, width={self.width}, "
                f"height={self.height}, angle={self.angle}, name={self.name!r})")


class Wall:
    """The arena ceiling or floor. Solid: touching one ends the run."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"Wall({self.name!r})"


TOP_WALL = Wall("top wall")
BOTTOM_WALL = Wall("bottom wall")


class Course:
    """A fixed arena: bounds, obstacle layout, start edge and finish line.

    The ceiling and floor are solid walls, so leaving through the top or bottom
    is a crash rather than a quiet disqualification.
    """

    # Paths are clamped only well OUTSIDE the arena, purely to keep the spline
    # numerically sane for something like y = exp(x). Clamping to the inside
    # was an exploit: it quietly rewrote any absurd equation into a legal
    # path hugging the ceiling. Now a path that leaves the arena leads the car
    # into a wall, which is what it looks like it should do.
    PATH_OVERSHOOT = 5.0

    def __init__(self, name, bounds, start_x, finish_x, obstacles,
                 default_equation, max_time=60.0):
        self.name = name
        self.xmin, self.xmax, self.ymin, self.ymax = (float(v) for v in bounds)
        self.start_x = float(start_x)
        self.finish_x = float(finish_x)
        self.obstacles = list(obstacles)
        self.default_equation = default_equation
        self.max_time = float(max_time)

    @property
    def bounds(self):
        return (self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def y_clamp(self):
        """Range a generated path is clamped to -- outside the arena, not inside.

        A path may legally leave the arena; the car simply crashes into the
        wall when it gets there.
        """
        return (self.ymin - self.PATH_OVERSHOOT, self.ymax + self.PATH_OVERSHOOT)

    def collision(self, rear, front, car_width):
        """Return whatever the car's footprint hit -- obstacle or wall -- or None.

        Both bodies are oriented rectangles and the obstacle test is exact, so
        what the player sees touching is precisely what counts as a crash. The
        walls are checked against the same footprint, meaning a tilted car
        clips the ceiling with a corner exactly when it looks like it does.
        """
        corners = body_corners(rear, front, car_width)

        if corners[:, 1].max() > self.ymax:
            return TOP_WALL
        if corners[:, 1].min() < self.ymin:
            return BOTTOM_WALL

        for obstacle in self.obstacles:
            if obstacle.intersects_polygon(corners):
                return obstacle
        return None

    def in_bounds(self, point, margin=0.0):
        return (self.xmin - margin <= point[0] <= self.xmax + margin
                and self.ymin - margin <= point[1] <= self.ymax + margin)

    def to_dict(self):
        return {
            "name": self.name,
            "bounds": {"xmin": self.xmin, "xmax": self.xmax,
                       "ymin": self.ymin, "ymax": self.ymax},
            "start_x": self.start_x,
            "finish_x": self.finish_x,
            "obstacles": [o.to_dict() for o in self.obstacles],
            "default_equation": self.default_equation,
            "max_time": self.max_time,
        }


def scatter_course():
    """The showcase course: a scattered field of tilted boxes.

    ---------------------------------------------------------------------
    EDIT THE COURSE HERE. Each row is one obstacle:

        (centre x, centre y, width, height, angle in degrees)

    A row where width == height is a square. Angle is counter-clockwise and
    rotates about the centre, so changing it never moves the box. The arena
    is x 0..60, y -16..16; keep the first ~14 units of x clear so cars have
    a run-up. Add or delete rows freely -- nothing else needs changing, and
    both the browser and the matplotlib view pick it up on restart.
    ---------------------------------------------------------------------
    """
    layout = [
        (17.0,   6.5, 6.5, 2.6, -35),
        (18.0,  -6.0, 4.2, 4.2,  20),
        (25.0,  11.0, 6.0, 2.5,  15),
        (26.0,   0.5, 3.8, 3.8, -50),
        (28.0, -10.0, 6.5, 2.6,  40),
        (34.0,   6.0, 5.5, 2.5,  65),
        (36.0,  -3.5, 4.0, 4.0, -25),
        (41.0,  11.0, 6.0, 2.6, -20),
        (43.0,  -9.0, 5.5, 2.5,  30),
        (46.0,   5.0, 4.5, 4.5,  55),
        (51.0,  -3.5, 6.5, 2.6, -15),
        (52.0,   9.0, 4.0, 4.0,  35),
        # Pillars anchored to the ceiling and floor. Without something reaching
        # the walls there is an open lane along each one, and a straight line
        # like y = 15 sails over the whole course untouched.
        (22.0,  14.0, 3.2, 4.6,   8),
        (38.0,  14.0, 3.2, 4.6,  -8),
        (55.0,  14.0, 3.0, 4.6,  12),
        (21.0, -14.0, 3.2, 4.6, -10),
        (37.0, -14.0, 3.2, 4.6,   8),
        (54.0, -14.0, 3.0, 4.6, -12),
    ]
    boxes = [Obstacle(cx, cy, w, h, angle, name=f"box-{i + 1}")
             for i, (cx, cy, w, h, angle) in enumerate(layout)]

    return Course(
        name="Scatter",
        bounds=(0.0, 60.0, -16.0, 16.0),
        start_x=0.0,
        finish_x=58.0,
        obstacles=boxes,
        default_equation="6*sin(x/4)",
        max_time=60.0,
    )

