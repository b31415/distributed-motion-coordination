
from __future__ import print_function
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from helpers.circle_simulation_helpers import *

"""Plot Constants"""
SUP_TITLE = "Single Robot with Circular Goal" # Title of the whole plot
# Draw every nth frame to avoid needing a higher refresh rate
FRAME_SKIP = 3
SAVE = False
FILE_NAME = "0.8max, (4,4)c, polar"

class CircleSimulator(CircleSimulationHelper):
    def __init__(self, cartesian = False, **sim_args):
        """
        Args:
            cartesian (bool): Default = false
            r (float): Radius. Default = 1
            goal_velocity (float): Default = 0.8 x max_velocity
            goal_centre (np.ndarray): Default = [3, 4]
            goal_angular_velocity: Default = goal_velocity / r. How fast
                the robot should go around the circle in rad/s.
            max_velocity (float): Default = 1
            simulation_time (float): Default = 30
            dynamics_time_step (float): Default = 8e-3
        """
        super(CircleSimulator, self).__init__(**sim_args)
        self.cartesian = cartesian

    def step_simulation(self, curr_pos, curr_vel):
        """
        Calculates one time-step of the simulation.
        Args:
            curr_pos
            curr_velocity
        Returns:
            new_pos (np.ndarray): On-axis cartesian position
            new_vel (np.ndarray): On-axis cartesian velocity
            control_input (np.ndarray): Acceleration. The system has
                double integrator dynamics so the control law determines
                the acceleration.
        """
        get_control_input = self.get_cartesian_control_input \
            if self.cartesian else self.get_polar_control_input
        
        if not self.cartesian:
            # Although the entire trajectory can be calculated and
            # converted to cartesian coordinates at the end for greater
            # efficency when using run_simulation() for offline calculations,
            # cartesian coordinates are used for interfacing with
            # step_simulation() so that programs directly calling 
            # step_simulation() for real-time calculations can directly
            # pass cartesian coordinates and so that offline use
            # (i.e. run_simulation()) is consistent with this.
            curr_vel = self.cartesian_velocity_to_polar(curr_pos, curr_vel)
            curr_pos = cartesian_to_polar(np.array(curr_pos) - self.goal_centre)
            
        # Get the position and velocity for the current time step
        control_input = get_control_input(curr_pos, curr_vel)
        delta_pos = self.dynamics_time_step * curr_vel
        new_pos = curr_pos + delta_pos
        
        new_vel = curr_vel + self.dynamics_time_step*control_input

        #  Saturate the velocity if it exceeds the maximum
        if not self.cartesian:
            new_vel = self.polar_velocity_to_cartesian(new_pos, new_vel)
            new_pos = polar_to_cartesian(new_pos) + self.goal_centre
        
        new_vel = np.clip(new_vel, -self.max_velocity, self.max_velocity)
        # else:
        #     velocity_magnitude = norm(polar_to_cartesian(new_velocity))
        #     if velocity_magnitude > self.max_velocity:
        #         new_velocity /= velocity_magnitude
                
        return new_pos, new_vel, control_input

    def plot_trajectories(
        self,
        positions,
        circle_thickness = 1,
        trajectory_thickness = 1,
        grid_thickness = 0.15
    ):
        """
        Plot a trajectory in the (x, y) plane.
        Args:
            positions (np.ndarray)
            circle_thickness (float)
            trajectory_thickness (float)
            grid_thickness (float)
        Returns:
            (list)
        """
        plt.title("Robot Trajectory")
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        
        # Calculate bounds for the trajectory and goal to determine the plot limits
        rs = self.r * np.ones((1, 2))
        centre = np.expand_dims(self.goal_centre, 0)
        goal_min = centre - rs
        goal_max = centre + rs
        # Compute the max and min of the positions trajectory and the goal circle.
        # Round the limits outwards to the next multiple of 0.5
        x_min, y_min = np.floor(np.min(np.concatenate((positions, goal_min), 0), 0) * 2) / 2
        x_max, y_max = np.ceil(np.max(np.concatenate((positions, goal_max), 0), 0) * 2) / 2
        
        # Increase either the width or height of the plot so
        # that it has a square aspect ratio
        width = x_max - x_min
        height = y_max - y_min
        expanding_by = abs(height - width) / 2
        if width > height:
            y_min -= expanding_by
            y_max += expanding_by
        else:
            x_min -= expanding_by
            x_max += expanding_by
            
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])
        plt.grid(True, lw=grid_thickness)
        
        # Plot goal circle
        circle = self.get_goal_set()
        plt.plot(circle[:, 0], circle[:, 1], 'c--', lw=circle_thickness)
        
        # Create lines with no points to be used for the animation 
        pos_line = plt.plot([], [], 'g', lw=1)
        
        # Plot a complete, thin version of the trajectories to be animated
        plt.plot(positions[:, 0], positions[:, 1], 'g', lw=trajectory_thickness)
        plt.legend(["Goal", "Trajectory"], loc="lower right")
        
        return pos_line

    def setup_plot(self, times, positions, velocities):
        """Plot everything except for the animated trajecories."""
        plt.subplots(1, 2, figsize=(8.5, 5))
        plt.suptitle(SUP_TITLE)
        
        """Plot robot trajectory"""
        plt.subplot(121)
        pos_line = self.plot_trajectories(positions, 0.8, 0.2, 0.15)
        
        """Plot velocities"""
        plt.subplot(122)
        if self.cartesian:
            plt.title("Robot Velocities")
            plt.plot(times, len(times)*[self.goal_velocity], '--', lw=0.6)
            plt.plot(times, velocities, lw=0.8)
            plt.plot(times, norm(velocities, axis=1), lw=0.8)
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity (m/s)")
            plt.grid(True, lw=0.15)
            plt.legend(["$v_g$", "$v_x$", "$v_y$", r"$\Vert\mathbf{v}\Vert$"], loc="lower right")
        else:
            plt.title("Robot Velocities")
            plt.plot(times, len(times)*[self.goal_angular_velocity], '--', lw=0.6)
            plt.plot(times, velocities, lw=0.8)
            # plt.plot(times, norm(velocities, axis=1), lw=0.8)
            plt.xlabel("Time (s)")
            plt.ylabel("Velocity")
            plt.grid(True, lw=0.15)
            plt.legend([r"$\dot{\theta}_g$ (rad/s)", r"$\dot{r}$ (m/s)", r"$\dot{\theta}$ (rad/s)"], loc="lower right")
        plt.tight_layout()
        return pos_line

    x_data, y_data =[], []

    def init_animation(self):
        """
        Returns:
            (list)
        """
        plt.subplot(121)
        return line

    def animate_trajectories(self, time_step):
        """
        Handle one time step of the animation.
        Args:
            time_step (int)
        Returns:
            (list)
        """
        end = time_step + 1
        
        # Assign x and y coordinates to be plotted
        x = positions[:end, 0]
        y = positions[:end, 1]
        line[0].set_data(x, y)
        
        print("{}\r".format(time_step), end="")
        return line

    
if __name__ == "__main__":
    simulator = CircleSimulator(r = 2, use_integral_term = False)
    times, _, _, positions, velocities, desired_positions, desired_velocities = \
        simulator.trajectory_tracking((0, 0))

    #times, _, _, positions, velocities, control_signal = simulator.run_simulation((0, 0))
    
    # Plot and animate the simulation results
    line = simulator.setup_plot(times, positions, velocities)
    
    anim = animation.FuncAnimation(
        fig         = plt.gcf(),
        init_func   = simulator.init_animation,
        func        = simulator.animate_trajectories, 
        frames      = list(range(0, len(times), FRAME_SKIP)),
        interval    = 1,
        blit        = True
    )
    if SAVE:
        anim.save(FILE_NAME + ".mp4", fps=144, dpi=400)
        plt.savefig(FILE_NAME + ".pdf")
    plt.show()