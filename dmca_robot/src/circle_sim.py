"""
This file is for testing the conversion functions between
the coordinate system used for the robot simulation and the
coordinates for the robot.
"""
from math import sqrt, pi
import os
import matplotlib.pyplot as plt

from helpers.lab_helpers import *
from simulations.circle_simulation import CircleSimulator
from helpers.general_simulation_helpers import TIGHT_LEGEND_KWARGS

L = 0.5
max_control_vel = np.array((0.4, 2))
goal_centre = (3, 1.8)
r = 1
dynamics_time_step=0.01

simulator = CircleSimulator(
    goal_centre = goal_centre,
    r = r,
    goal_velocity=max_control_vel[0]/sqrt(2),
    simulation_time=30,
    dynamics_time_step=dynamics_time_step,
    use_integral_term=False
)

# Calculate the desired next position and velocity from the 
# current position in a similar manner to the real-time calculations
# with the physical robot.
expand_first = lambda value : np.expand_dims(value, 0)
 
simulation_time = simulator.simulation_time
n_iterations = int(simulation_time / simulator.dynamics_time_step)

times, _, control_velocities, positions, *_ = simulator.run_simulation(np.array((0, 0)))

# Plot
init_python2_plot()
plt.subplots(1, 3, figsize=(13, 4.5))
plt.subplot(131)
simulator.plot_trajectories(positions)
plt.legend(["Goal", r"$\tilde{\mathbf{x}}$"], **TIGHT_LEGEND_KWARGS)

GRID_THICKNESS = 0.15
def setup_timeseries():
    plt.xlabel("Time (m/s)")
    plt.grid(True, lw=GRID_THICKNESS)
    plt.xlim([0, times[-1]])

linear_velocities = control_velocities[:, 0]
angular_velocities = control_velocities[:, 1]
plt.subplot(132)
plt.title("Linear Velocity")
plt.ylabel("$v$ (m/s)")
plt.plot(times, linear_velocities)
setup_timeseries()

plt.subplot(133)
plt.title("Angular Velocity")
plt.ylabel("$\omega$ (rad/s)")
plt.plot(times, angular_velocities)
setup_timeseries()
clipped_velocities = np.clip(angular_velocities, a_min=-pi, a_max=pi)
plt.ylim([min(clipped_velocities), max(clipped_velocities)])

plt.tight_layout()

if os.path.exists("./dmca_robot"):
    os.chdir("./dmca_robot/src")
if not os.path.exists("./outputs/simulations/"):
    os.makedirs("./outputs/simulations/")

plt.savefig(
    "./outputs/simulations/desired_robot_trajectories_"
    "r{}_g{}-{}_maxVel{}-{}_dt{}.pdf".format(
        r, goal_centre[0], goal_centre[1], max_control_vel[0], max_control_vel[1], dynamics_time_step
    ))
plt.show()