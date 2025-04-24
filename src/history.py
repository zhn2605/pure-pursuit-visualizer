import numpy as np
import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

class Track:
    def __init__(self):
        self.xs = []  # x coordinates
        self.ys = []  # y coordinates
        self.color = 'k'

    def add_point(self, point):
        self.xs.append(point[0])
        self.ys.append(point[1])
    
    def show_track(self):
        plt.plot(self.points[:, 0], self.points[:, 1], color='k', marker='o', linestyle='-')
        plt.show()

    # Will work on this later
# def spline_track(self):
# if len(self.points) < 3
# print("Not enough points for spline")
# return
#
# x, y = self.points[:, 0], self.points[:, 1]
# cubic_spline = CubicSpline(x, y)
#

class History:
    def __init__(self, debug=True):
        self.track = Track()
        self.position = []
        self.theta = []
        self.velocity = []
        self.time = []

        self.debug = debug
    
    def display_track(self):
        self.track.show_track()
    
    def generate_track(self, xmin=0, xmax=10, points=100, function=lambda x: -((x - 5)**2) + 25):
        for x in np.linspace(xmin, xmax, points):
            point = np.array([x, function(x)])
            self.track.add_point(point)
    
    def append(self, time, curr):
        self.position.append(curr.position.copy())
        self.theta.append(curr.theta)
        self.velocity.append(curr.velocity)
        self.time.append(time)
    
    def animate(self, car, pure_pursuit, interval=50, max_time=20.0):        
        fig, ax = plt.subplots()
        ax.set_xlim(min(self.track.xs) - 2, max(self.track.xs) + 2)  # Set x-axis limits
        ax.set_ylim(min(self.track.ys) - 5, max(self.track.ys) + 5)  # Set y-axis limits
        ax.plot(self.track.xs, self.track.ys, marker='o', markersize=2, color=self.track.color, label="Track")  # Draw the static track
    
        # Store trajectory points
        trajectory_xs = []
        trajectory_ys = []
        trajectory_line, = ax.plot([], [], 'b-', label="Trajectory")  # Trajectory line
    
        car_rear, = ax.plot([], [], 'bo', markersize=8)
        car_front, = ax.plot([], [], 'bo', markersize=8)
        car_frame, = ax.plot([], [], 'k-', linewidth=2.25, label="Car")
        lookAheadDot, = ax.plot([], [], 'kx', markersize=10, label="Lookahead Point")
        lookahead_line, = ax.plot([], [], 'r--', linewidth=1.5, label="Lookahead")
        final_pos = np.array([self.track.xs[-1], self.track.ys[-1]])

        # Time limit
        current_time = 0.0
        dt = interval / 500.0

        def update(frame):
            nonlocal current_time

            # Update angle and position
            delta_theta = pure_pursuit.calc_angle(car, self.track)
            car.update(pure_pursuit, self.track)
            car.lookAheadPosition = pure_pursuit.calc_lookahead_pos(car, self.track)

            # Update and store trajectory
            trajectory_xs.append(car.position[0])
            trajectory_ys.append(car.position[1])

            # Update plot data
            car_rear.set_data([car.position[0]], [car.position[1]])
            car_front.set_data([car.front[0]], [car.front[1]])
            car_frame.set_data([car.position[0], car.front[0]], [car.position[1], car.front[1]])
            lookAheadDot.set_data([car.lookAheadPosition[0]], [car.lookAheadPosition[1]])
            lookahead_line.set_data([car.front[0], car.lookAheadPosition[0]], 
                                [car.front[1], car.lookAheadPosition[1]])
            trajectory_line.set_data(trajectory_xs, trajectory_ys)  # Update trajectory line
            
            # Debug flag
            if self.debug:
                print(f"Time: {current_time:.2f}s, Car Position: {car.position}, Lookahead Position: {car.lookAheadPosition}")

            if np.linalg.norm(car.position - final_pos) < 0.5:
                print("Car reached final point.")
                ani.event_source.stop()

            if current_time > max_time:
                print("Time limit reached.")
                ani.event_source.stop()
            
            current_time += dt

            return car_rear, car_front, car_frame, lookAheadDot, trajectory_line, lookahead_line

        ani = animation.FuncAnimation(fig, update, frames=100, interval=interval, blit=True)

        plt.legend()
        plt.grid(True)
        plt.show()