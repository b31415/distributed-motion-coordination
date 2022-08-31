#!/usr/bin/env python
"""
Similar to circle_lab.py but used on each node in a multi-robot system.

# TODO: Get working with the second order consensus protocol
# TODO: Refactor so that this file works with a single robot as well as
#   multiple robots to replace circle_lab.py instead of duplicating it.
"""
from math import atan2, sqrt, cos, sin
import os
import sys
from time import time
from datetime import datetime
import importlib

import rospy
import numpy as np
import matplotlib.pyplot as plt
import argparse

from helpers.lab_helpers import init_python2_plot
from simulations.circle_simulation import CircleSimulator
from helpers.unicycle_robot import UnicycleRobot

def run_single_node(heading_offset, coordinate_offset, simulator, goal_args):
    simulation_module = __import__(simulator)
    print("Python version: {}".format(sys.version))
    init_python2_plot()

    """Robot setup"""
    max_control_velocity = np.array((0.5, 1)) # (m/s, rad/s)
    robot = UnicycleRobot(
        log_level = rospy.WARN,
        max_control_velocity = max_control_velocity,
        heading_offset = heading_offset,
        coordinate_offset = coordinate_offset
    )

    """Calculate desired goal relative to the initial state"""
    # TODO: Check initial heading is 0 and initial pos is (0, 0)
    
    #TODO: Refactor to use a Simulator() function in a file specified by a command-line argument
    # and that file should have a function for calculating the goal relative to the master.
    
    # Desired goal relative to the inital position.
    goal_centre = np.array((1, 1))

    """Simulator setup"""
    # The maximum linear velocity from the simulation is sqrt(xdot^2 + ydot^2).
    # Restrict xdot and ydot so that the maximum simulation velocity from
    # simulation doesn't exceed the maximum linear robot velocity.
    simulator = CircleSimulator(
        goal_centre = goal_centre,
        r = 1,
        dynamics_time_step = 1.0/robot.subscribe_freq,
        max_velocity = max_control_velocity[0]/sqrt(2)
    )
    
    # TODO: curr_pos and curr_vel should be a list of positions and velocities respectively
    #   for a multi-robot system.
    # TODO: Ideally simulator.step_simulation should accept *args, but that is not a high priority
    # TODO: Add delta_t (i.e. time_step) as an argument to step_simulation to make it more accurate?
    # Ignore the first argument of step_simulation so that get_next_state()
    # is in the form required by robot.publish_next_state()
    get_next_state = lambda curr_pos, curr_vel: simulator.step_simulation(curr_pos, curr_vel)[1:3]

    print("Finished setup")
    start_time = time()

    while not rospy.is_shutdown():
        robot.publish_next_state(get_next_state)

    total_time = time() - start_time
    print("Experiment finished in {:.3}s. Saving data...".format(total_time))

    # Save positions and velocities in case the data needs to be used later
    dir = "outputs/{}".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
    if os.path.exists('./dmca_robot'):
        os.chdir('./dmca_robot/src')
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    np.savetxt("{}/offaxis_pos.csv".format(dir), robot.offaxis_positions, delimiter=",")
    np.savetxt("{}/control_vel.csv".format(dir), robot.control_velocities, delimiter=",")
    
    goal_circle = simulator.get_goal_set()
    plt.subplots(2, 2, figsize=(6.5, 6))
    plt.subplot(221)
    simulator.plot_trajectories(robot.offaxis_positions)
    # plt.title("Off-axis Robot Positions")
    # plt.plot(robot.offaxis_positions[:, 0], robot.offaxis_positions[:, 1])
    # plt.plot(goal_circle[:, 0], goal_circle[:, 1])
    # plt.xlabel("$x$ (m)")
    # plt.ylabel("$y$ (m)")
    # plt.legend(["Trajectory", "Goal set"], loc='lower right')

    times = np.linspace(0, total_time, len(robot.control_velocities[:, 0]))
    plt.subplot(222)
    plt.title("Linear Velocity")
    plt.plot(times, robot.control_velocities[:, 0], lw=1)
    plt.xlabel("Time (s)")
    plt.ylabel("$v$ (m/s)")

    plt.subplot(223)
    plt.plot(times, robot.control_velocities[:, 1], lw=1)
    plt.title("Angular Velocity")
    plt.xlabel("Time (s)")
    plt.ylabel("$\omega$ (rad/s)")

    plt.subplot(224)
    plt.plot(times, robot.headings, lw=1)
    plt.title("Heading Angle")
    plt.xlabel("Time (s)")
    plt.ylabel("$\phi$ (rad)")

    plt.suptitle("Unicycle Robot Circular Control with $r = {}$ and "
        "c = ({:.2}, {:.2})".format(r, goal_centre[0], goal_centre[1]),
        y=1.05
    )
    plt.tight_layout()
    plt.savefig("{}/offaxis_robot_positions.png".format(dir), dpi=150)
    plt.savefig("{}/offaxis_robot_positions.pdf".format(dir))
    print("Saved figure")