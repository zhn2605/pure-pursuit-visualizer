import numpy as np

from obstacles import body_corners

DEFAULT_DT = 0.1

# Width of the car's footprint, in world units. Length is the wheelbase, so the
# car is a 2.5 x 1.2 rectangle -- the exact shape used for collision and drawn.
CAR_WIDTH = 1.2


class Current:
    '''
    Represents a 'fake vehicle' for quick Pure Pursuit testing
    '''
    def __init__(self, position=np.array([0.0, 0.0]), lookAheadDistance=2.0, velocity=1.0,
                 heading=0.0, max_accel=1.0, turn_sensitivity=3.0,
                 max_speed=10.0, min_speed=1.0, max_steer_deg=45.0, dt=DEFAULT_DT,
                 width=CAR_WIDTH):
        # Current position of vehicle. Copied so a caller's array is never
        # mutated out from under them by update_position.
        self.position = np.array(position, dtype=float).copy()  # (x, y)
        self.wheelbase = 2.5
        self.width = float(width)

        # Look Ahead Distance
        self.lookAheadDistance = lookAheadDistance

        # Look Ahead position
        self.lookAheadPosition = self.position.copy()

        # Velocity of vehicle
        self.velocity = velocity

        # Orientation of vehicle
        self.theta = heading
        self.delta_theta = 0

        # Acceleration of vehicle
        self.acceleration = 0.0

        # Maximum acceleration
        self.max_accel = max_accel

        # Curvature
        self.curvature = 0.0

        # Speed limits
        self.max_speed = max_speed
        self.min_speed = min_speed

        # Settings
        # Stored in radians. The old code kept 45 here and fed it straight to
        # np.tan(), which computed tan(45 rad) and silently capped steering at
        # a meaningless value.
        self.max_steer = np.radians(max_steer_deg)
        self.turn_sens = turn_sensitivity

        # Integration step
        self.dt = dt

        # Front axle position, kept consistent with position/theta at all times.
        self.front = self._front_from_pose()

    def body_corners(self):
        """The car's rectangular footprint, rear axle to front axle."""
        return body_corners(self.position, self.front, self.width)

    def _front_from_pose(self):
        return self.position + self.wheelbase * np.array([
            np.cos(self.theta),
            np.sin(self.theta)
        ])

    def align_car(self, track, heading_offset=0.0):
        # Aligns car to the track, useful for beginning simulations
        self.position = np.array(track.start, dtype=float)
        self.theta = track.start_heading() + heading_offset
        self.front = self._front_from_pose()
        self.lookAheadPosition = self.position.copy()

    def calc_velocity(self):
        curvature_magnitude = abs(self.curvature)
        speed_factor = np.exp(-self.turn_sens * curvature_magnitude)

        target_speed = self.min_speed + (self.max_speed - self.min_speed) * speed_factor

        return target_speed

    def update_velocity(self, target_velocity):
        dv = target_velocity - self.velocity

        # Implement a clamp for maximum / minimum acceleration
        acceleration = np.clip(dv / self.dt, -self.max_accel, self.max_accel)

        self.acceleration = float(acceleration)
        self.velocity += acceleration * self.dt

    def update_position(self):
        # Separate into respective components
        dx = self.velocity * np.cos(self.theta) * self.dt
        dy = self.velocity * np.sin(self.theta) * self.dt

        # Update position, then derive the front axle from the new pose. The
        # old ordering left self.front one step stale, which skewed both the
        # lookahead search and any collision test built on the car body.
        self.position = self.position + np.array([dx, dy])
        self.front = self._front_from_pose()

    def update(self, pure_pursuit, track):
        self.lookAheadPosition = pure_pursuit.calc_lookahead_pos(self, track)

        # Use curvature for smoother turns
        self.curvature = pure_pursuit.calc_curvature(self, track)
        self.theta += self.curvature * self.velocity * self.dt

        target_velocity = self.calc_velocity()
        self.update_velocity(target_velocity)
        self.update_position()
