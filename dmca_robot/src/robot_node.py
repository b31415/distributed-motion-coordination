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

from helpers.lab_helpers import init_python2_plot
from helpers.unicycle_robot import UnicycleRobot
from robot_master import RosMaster

class RosNode:
    def __init__(self):
        self.name = rospy.get_param("~name")
        self.initial_position = rospy.get_param("~initial_position")
        self.heading = rospy.get_param("~heading")

        """Robot setup"""
        self.max_control_velocity = np.array((0.5, 1)) # (m/s, rad/s)
        self.robot = UnicycleRobot(
            node_name = self.name,
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
            self.robot.publish_next_state(get_next_state)

    
if __name__ == "__main__":
    robot = RosMaster() if rospy.get_param("~is_master") else RosNode()
    robot.run()
    