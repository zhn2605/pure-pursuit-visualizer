import numpy as np

class PurePursuit:
    def calc_lookahead_pos(self, car, track):
        # Vectorised over the track: the simulation runs thousands of steps per
        # submission, and a Python loop per step dominated the runtime.
        track_points = track.points

        # Find the index of the closest track point to the car
        distances = np.linalg.norm(track_points - car.front, axis=1)
        closest_index = int(np.argmin(distances))

        # Consider only future points (ensuring progression), and among those
        # take the nearest one at least lookAheadDistance away.
        forward = distances[closest_index + 1:]
        if forward.size == 0:
            return car.lookAheadPosition

        eligible = np.flatnonzero(forward >= car.lookAheadDistance)
        if eligible.size == 0:
            # Past the last point far enough out: aim at the end of the track
            # so the car still drives through the finish.
            return track_points[-1].copy()

        best = eligible[np.argmin(forward[eligible])]
        return track_points[closest_index + 1 + best].copy()

    def calc_distance(self, initial, desired):
        # pythagorean theroem
        return np.hypot(desired[0] - initial[0], desired[1] - initial[1])

    def calc_angle(self, car, track):
        direction = car.lookAheadPosition - car.front
        angle = np.arctan2(direction[1], direction[0])

        delta_theta = angle - car.theta
        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))

        return delta_theta

    def calc_curvature(self, car, track):
        # Vector from car to lookahead
        dx = car.lookAheadPosition[0] - car.position[0]
        dy = car.lookAheadPosition[1] - car.position[1]

        # VERY important step of localizing the lookahead distance
        local_x = dx * np.cos(car.theta) + dy * np.sin(car.theta)
        local_y = -dx * np.sin(car.theta) + dy * np.cos(car.theta)

        L = np.sqrt(local_x**2 + local_y**2)

        if abs(local_y) < 1e-6 or L < 1e-9:
            curvature = 0.0
        else:
            curvature = 2 * local_y / (L**2)

        curv_max = np.tan(car.max_steer) / car.wheelbase
        curvature = np.clip(curvature, -curv_max, curv_max)

        return curvature
