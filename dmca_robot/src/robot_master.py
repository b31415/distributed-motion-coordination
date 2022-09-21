#!/usr/bin/env python
"""
Similar to circle_lab.py but used on each node in a multi-robot system.

# TODO: Get working with the second order consensus protocol
# TODO: Refactor so that this file works with a single robot as well as
#   multiple robots to replace circle_lab.py instead of duplicating it.
"""
import argparse
from math import atan2, sqrt, cos, sin
import os
import sys
from time import time
from datetime import datetime
import importlib

import rospy
import numpy as np
import matplotlib.pyplot as plt

from helpers.lab_helpers import init_python2_plot
from helpers.unicycle_robot import UnicycleRobot

class RosMaster:
    def __init__(self, args):
        """
        Save params
        """
        self.sim_args = dict(args.simulator_args)
        full_simulator_class = args.simulator_class
        
        # Convert simulator_class from a string to a class
        try:
            simulator_module, simulator_class = full_simulator_class.rsplit(".", 1)
            self.SimulatorClass = getattr(importlib.import_module(simulator_module), simulator_class)
        except Exception:
            print("Class '{}' not found".format(full_simulator_class))
    
    
    def run(self):
        """
        Args:
            full_simulator_class (str): In the form "<module>.<class>" relative to the
                ./dmca_robot/src directory. E.g. "simulations.circle_simulation.CircleSimulator".
            
        """
        
        # TODO: Check initial heading is 0 and initial pos is (0, 0)
        print("Python version: {}".format(sys.version))
        init_python2_plot()

        """Simulator setup"""
        # Desired goal relative to the inital position.
        goal_centre = np.array((1, 1))
        
        # The maximum linear velocity from the simulation is sqrt(xdot^2 + ydot^2).
        # Restrict xdot and ydot so that the maximum simulation velocity from
        # simulation doesn't exceed the maximum linear robot velocity.
        max_velocity = self.max_control_velocity[0]/sqrt(2)
        
        r = 1
        simulator = self.SimulatorClass(
            goal_centre = self.sim_args.goal_positions,
            r = r,
            dynamics_time_step = 1.0/self.robot.subscribe_freq,
            max_velocity = max_velocity
        )
        
        # TODO: curr_pos and curr_vel should be a list of positions and velocities respectively
        #   for a multi-robot system.
        # TODO: Add delta_t (i.e. time_step) as an argument to step_simulation to make it more accurate?
        get_next_state = lambda curr_pos, curr_vel: simulator.step_simulation(curr_pos, curr_vel)[:2]

        print("Finished setup")
        start_time = time()

        while not rospy.is_shutdown():
            print("State: " + str(get_next_state()))
            #TODO: Publish results to each robot
            #self.robot.publish_next_state(get_next_state)

        total_time = time() - start_time
        print("Experiment finished in {:.3}s. Saving data...".format(total_time))

        # Save positions and velocities in case the data needs to be used later
        dir = "outputs/{}".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
        if os.path.exists('./dmca_robot'):
            os.chdir('./dmca_robot/src')
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        np.savetxt("{}/offaxis_pos.csv".format(dir), self.robot.offaxis_positions, delimiter=",")
        np.savetxt("{}/control_vel.csv".format(dir), self.robot.control_velocities, delimiter=",")
        
        plt.subplots(2, 2, figsize=(6.5, 6))
        plt.subplot(221)
        simulator.plot_trajectories(self.robot.offaxis_positions)
        # plt.title("Off-axis Robot Positions")
        # plt.plot(robot.offaxis_positions[:, 0], self.robot.offaxis_positions[:, 1])
        # plt.plot(goal_circle[:, 0], goal_circle[:, 1])
        # plt.xlabel("$x$ (m)")
        # plt.ylabel("$y$ (m)")
        # plt.legend(["Trajectory", "Goal set"], loc='lower right')

        times = np.linspace(0, total_time, len(self.robot.control_velocities[:, 0]))
        plt.subplot(222)
        plt.title("Linear Velocity")
        plt.plot(times, self.robot.control_velocities[:, 0], lw=1)
        plt.xlabel("Time (s)")
        plt.ylabel("$v$ (m/s)")

        plt.subplot(223)
        plt.plot(times, self.robot.control_velocities[:, 1], lw=1)
        plt.title("Angular Velocity")
        plt.xlabel("Time (s)")
        plt.ylabel("$\omega$ (rad/s)")

        plt.subplot(224)
        plt.plot(times, self.robot.headings, lw=1)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_name")
    parser.add_argument("--goal_positions")
    parser.add_argument("--simulator_class")
    parser.add_argument("--node_names")
    args = parser.parse_args()
    print("Master args:\n" + args)

    master = RosMaster()
    master.run()
    