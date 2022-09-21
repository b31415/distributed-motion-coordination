#!/usr/bin/env python
"""
Similar to circle_lab.py but used on each node in a multi-robot system.

# TODO: Get working with the second order consensus protocol
# TODO: Refactor so that this file works with a single robot as well as
#   multiple robots to replace circle_lab.py instead of duplicating it.
"""
import sys
import rospy
import numpy as np
import argparse

from helpers.unicycle_robot import UnicycleRobot

class RosNode:
    def __init__(self, args):
        self.id = args.id
        self.name = args.name
        self.initial_position = np.array(args.initial_position)
        self.heading = args.heading

        """Robot setup"""
        self.max_control_velocity = np.array((0.5, 1)) # (m/s, rad/s)
        self.robot = UnicycleRobot(
            node_name = "RosAria" + self.name.capitalize(),
            log_level = rospy.WARN,
            max_control_velocity = self.max_control_velocity,
            heading_offset = self.heading,
            coordinate_offset = self.initial_position
        )

    def run(self):
        """
        Args:
            full_simulator_class (str): In the form "<module>.<class>" relative to the
                ./dmca_robot/src directory. E.g. "simulations.circle_simulation.CircleSimulator".
            
        """

        while not rospy.is_shutdown():
            pass
            #self.robot.publish_next_state(get_next_state)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int)
    parser.add_argument("--name")
    parser.add_argument("--initial_position")
    parser.add_argument("--heading", type=float)
    args = parser.parse_args()
    print(args.name.capitalize() + " args:\n" + str(args))

    robot = RosNode(args)
    #robot.run()
    