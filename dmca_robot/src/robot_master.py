#!/usr/bin/env python
"""
Similar to circle_lab.py but used on each node in a multi-robot system.

# TODO: Get working with the second order consensus protocol
# TODO: Refactor so that this file works with a single robot as well as
#   multiple robots to replace circle_lab.py instead of duplicating it.
"""
import argparse
from math import sqrt
import os
import sys
from time import time
from datetime import datetime
import importlib
from helpers.unicycle_robot import UnicycleRobot

import rospy
import numpy as np
import matplotlib.pyplot as plt

from helpers.lab_helpers import init_python2_plot

class RosMaster:
    def __init__(self, args):
        # Import simulator from config file
        try:
            # Get dict of args from the config file
            import_attr = lambda attr: getattr(importlib.import_module(args.config_file), attr)
            self.simulator = import_attr("simulator")
            headings = import_attr("headings")
            initial_positions = import_attr("initial_positions")
        except Exception:
            print("Config file {} or corresponding simulator not found".format(args.config_file))
    
        self.names = args.node_names
        self.n_robots = len(self.names)
        self.robots = []
        max_control_velocity = np.array((0.5, 1)) # (m/s, rad/s)
        
        rospy.init_node(args.master_name, anonymous=True, log_level=rospy.DEBUG)

        for id, name in enumerate(self.names):
            self.robots.append(
                UnicycleRobot(
                    name = name,
                    log_level = rospy.WARN,
                    max_control_velocity = max_control_velocity,
                    heading_offset = headings[id],
                    coordinate_offset = initial_positions[id],
                    init = False
                )
            )
        
    def run(self):
        """
        Args:
            full_simulator_class (str): In the form "<module>.<class>" relative to the
                ./dmca_robot/src directory. E.g. "simulations.circle_simulation.CircleSimulator".
            
        """
        
        # TODO: Check initial heading is 0 and initial pos is (0, 0)
        print("Python version: {}".format(sys.version))
        init_python2_plot()


        print("Finished setup")
        start_time = time()

        states = []
        samples_handled = 0

        while not rospy.is_shutdown():
            samples_recorded = min([len(robot.headings) for robot in self.robots])
            print("positions recorded: {}, positions handled: {}".format(samples_recorded, samples_handled))
            if samples_recorded > samples_handled:
                curr_pos = np.zeros((self.n_robots, 2))
                curr_vel = np.zeros((self.n_robots, 2))

                # Get the current state of each robot
                for i, robot in enumerate(self.robots):
                    _, pos, vel = robot.get_current_offaxis_state()
                    print("{} {}".format(robot, pos))
                    curr_pos[i] = pos
                    curr_pos[i] = vel

                # Calculate the next state
                new_state = self.simulator.step_simulation(curr_pos, curr_vel)
                new_pos, new_vel = new_state[:2]
                states.append(new_state)

                # Publish the new state to make the robots move
                for i, robot in enumerate(self.robots):
                    robot.publish_offaxis(new_pos[i], new_vel[i])

                samples_handled = samples_recorded

            self.robots[0].rate.sleep()

        total_time = time() - start_time
        print("Experiment finished in {:.3}s. Saving data...".format(total_time))

        # Save positions and velocities in case the data needs to be used later
        dir = "outputs/{}".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
        if os.path.exists('./dmca_robot'):
            os.chdir('./dmca_robot/src')
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        np.savetxt("{}/states.csv".format(dir), states, delimiter=",")

        times = np.linspace(0, total_time, len(self.robot.control_velocities[:, 0]))
        
        # Rearrange from a list of time-steps to a list of states
        states = zip(*states)
        self.simulator.setup_plot(times, *states)

        plt.tight_layout()
        plt.savefig("{}/robot.png".format(dir), dpi=150)
        plt.savefig("{}/robot.pdf".format(dir))
        print("Saved figure")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_name", default="donnie")
    #parser.add_argument("--node_names", default = ["donnie"])
    parser.add_argument("--node_names", default = ["donnie", "leonardo"])
    parser.add_argument("--config_file", default = "multi_robot_config")
    args = parser.parse_args()

    print("Master args: {}\n".format(args))

    master = RosMaster(args)
    master.run()
    