import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon

from track import Track  # Track now lives in track.py alongside the spline code


class History:
    def __init__(self, debug=True, course=None):
        self.track = Track()
        self.course = course
        self.position = []
        self.theta = []
        self.velocity = []
        self.time = []

        self.debug = debug

    def generate_track(self, xmin=0, xmax=10, points=100,
                       function=lambda x: -((x - 5)**2) + 25, spline=False):
        """Build the track from a Python callable, optionally spline-smoothed."""
        xs = np.linspace(xmin, xmax, points)
        track = Track([(x, function(x)) for x in xs])
        self.track = track.spline(samples=points * 4) if spline else track
        return self.track

    def generate_track_from_equation(self, equation, xmin=0, xmax=60, points=160, spline=True):
        """Build the track from a text equation, same parser the web UI uses."""
        track = Track.from_equation(equation, xmin=xmin, xmax=xmax, points=points)
        self.track = track.spline(samples=points * 3) if spline else track
        return self.track

    def append(self, time, curr):
        self.position.append(curr.position.copy())
        self.theta.append(curr.theta)
        self.velocity.append(curr.velocity)
        self.time.append(time)

    def _draw_course(self, ax):
        if self.course is None:
            return
        # Outlined in red, unfilled, drawn from the rotated corners so the
        # desktop view matches the browser view exactly.
        for obstacle in self.course.obstacles:
            ax.add_patch(Polygon(
                obstacle.corners(), closed=True,
                fill=False, edgecolor='r', linewidth=1.4, zorder=1,
            ))
        # Ceiling and floor are solid: draw them like the obstacles.
        ax.axhline(self.course.ymax, color='r', linewidth=2)
        ax.axhline(self.course.ymin, color='r', linewidth=2)
        ax.axvline(self.course.finish_x, color='k', linestyle='--',
                   linewidth=1.2, label='Finish')
        ax.set_aspect('equal', adjustable='box')

    def animate(self, car, pure_pursuit, interval=50, max_time=20.0):
        fig, ax = plt.subplots()

        if self.course is not None:
            ax.set_xlim(self.course.xmin - 1, self.course.xmax + 1)
            ax.set_ylim(self.course.ymin - 1, self.course.ymax + 1)
        else:
            ax.set_xlim(min(self.track.xs) - 2, max(self.track.xs) + 2)
            ax.set_ylim(min(self.track.ys) - 5, max(self.track.ys) + 5)

        self._draw_course(ax)
        ax.plot(self.track.xs, self.track.ys, marker='o', markersize=2,
                color=self.track.color, label="Track")

        # Store trajectory points
        trajectory_xs = []
        trajectory_ys = []
        trajectory_line, = ax.plot([], [], 'b-', label="Trajectory")

        car_rear, = ax.plot([], [], 'bo', markersize=8)
        car_front, = ax.plot([], [], 'bo', markersize=8)
        car_frame, = ax.plot([], [], 'k-', linewidth=2.25)
        car_body, = ax.plot([], [], 'b-', linewidth=1.4, label="Car")
        lookAheadDot, = ax.plot([], [], 'kx', markersize=10, label="Lookahead Point")
        lookahead_line, = ax.plot([], [], 'r--', linewidth=1.5, label="Lookahead")
        final_pos = self.track.end

        # The animation steps the car by car.dt, so the clock must agree with
        # the physics rather than with the wall-clock frame interval.
        current_time = 0.0

        def stop(reason):
            print(reason)
            ani.event_source.stop()

        def update(frame):
            nonlocal current_time

            car.update(pure_pursuit, self.track)
            self.append(current_time, car)

            trajectory_xs.append(car.position[0])
            trajectory_ys.append(car.position[1])

            car_rear.set_data([car.position[0]], [car.position[1]])
            car_front.set_data([car.front[0]], [car.front[1]])
            car_frame.set_data([car.position[0], car.front[0]],
                               [car.position[1], car.front[1]])
            # Closed footprint outline: exactly the shape collision uses.
            body = np.vstack([car.body_corners(), car.body_corners()[:1]])
            car_body.set_data(body[:, 0], body[:, 1])
            lookAheadDot.set_data([car.lookAheadPosition[0]], [car.lookAheadPosition[1]])
            lookahead_line.set_data([car.front[0], car.lookAheadPosition[0]],
                                    [car.front[1], car.lookAheadPosition[1]])
            trajectory_line.set_data(trajectory_xs, trajectory_ys)

            if self.debug:
                print(f"Time: {current_time:.2f}s, Car Position: {car.position}, "
                      f"Lookahead Position: {car.lookAheadPosition}")

            if self.course is not None:
                hit = self.course.collision(car.position, car.front, car.width)
                if hit is not None:
                    stop(f"GAME OVER - hit {hit.name} at t={current_time:.2f}s")
                elif car.front[0] >= self.course.finish_x:
                    stop(f"FINISHED in {current_time:.2f}s")
            elif np.linalg.norm(car.position - final_pos) < 0.5:
                stop("Car reached final point.")

            if current_time > max_time:
                stop("Time limit reached.")

            current_time += car.dt

            return (car_rear, car_front, car_frame, car_body, lookAheadDot,
                    trajectory_line, lookahead_line)

        ani = animation.FuncAnimation(fig, update, frames=None, interval=interval,
                                      blit=True, cache_frame_data=False)

        plt.legend(loc='upper right', fontsize=8)
        plt.grid(True)
        plt.show()
