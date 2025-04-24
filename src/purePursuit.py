import numpy as np

class PurePursuit:
    def calc_lookahead_pos(self, car, track):
        min_dist = float('inf')
        lookAheadPoint = None

        # Convert track points into a NumPy array
        track_points = np.column_stack((track.xs, track.ys))

        # Find the index of the closest track point to the car
        distances = np.linalg.norm(track_points - car.front, axis=1)
        closest_index = np.argmin(distances)  # Get index of closest point

        # Iterate only over future points (ensuring progression)
        for i in range(closest_index + 1, len(track.xs)):  # Start after closest point
            point = np.array([track.xs[i], track.ys[i]])
            distance = np.linalg.norm(point - car.front)

            if distance >= car.lookAheadDistance and distance < min_dist:
                min_dist = distance
                lookAheadPoint = point

        return lookAheadPoint if lookAheadPoint is not None else car.lookAheadPosition

    def calc_distance(self, initial, desired):
        # pythagorean theroem
        return np.hypot(desired[0] - initial[0], desired[1] - initial[1])

    def calc_angle(self, car, track):
        direction = car.lookAheadPosition - car.position
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

        if abs(local_y) < 1e-6:
            curvature = 0
        else:
            curvature = 2 * local_y / (L**2)

        curv_max = np.tan(car.max_theta) / car.wheelbase
        curvature = np.clip(curvature, -curv_max, curv_max)

        return curvature

    # def calc_curvature(self, car, track):
    #     # Calculate radius of curvature
    #     L = car.lookAheadDistance
    #     desired_x = car.lookAheadPosition[0]

    #     r = (L**2)/(2 * abs(desired_x))
    #     curvature = 1/r

    #     return curvature