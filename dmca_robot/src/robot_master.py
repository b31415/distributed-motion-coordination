#!/usr/bin/env python
"""
Similar to circle_lab.py but used on each node in a multi-robot system.

# TODO: Get working with the second order consensus protocol
# TODO: Refactor so that this file works with a single robot as well as
#   multiple robots to replace circle_lab.py instead of duplicating it.
"""
import os
import sys
from time import time
from datetime import datetime
import importlib
import json

import rospy
import numpy as np
import matplotlib.pyplot as plt

from helpers.lab_helpers import init_python2_plot
from helpers.unicycle_robot import UnicycleRobot

class RosMaster:
    def __init__(self, config_module, master_name="donnie"):
        """
        Args:
            config_module (str): Name of the file (excluding the file extension)

        """
        # Import simulator from config file
        try:
            ConfigClass = getattr(importlib.import_module(config_module), "Config")
            config = ConfigClass()
        except Exception:
            print("Config file {} not found".format(config_module))
            exit()
    
        self.names = config.get_node_names()
        self.simulator = config.get_simulator()
        
        self.n_robots = len(self.names)
        self.robots = []
        max_control_velocity = np.array((0.2, 1)) # (m/s, rad/s)
        
        rospy.init_node(master_name, anonymous=True, log_level=rospy.DEBUG)

        for id, name in enumerate(self.names):
            self.robots.append(
                UnicycleRobot(
                    name = name,
                    log_level = rospy.INFO,
                    max_control_velocity = max_control_velocity,
                    heading_offset = config.get_headings()[id],
                    coordinate_offset = config.get_initial_positions()[id],
                    init = False
                )
            )
        
    def run(self):
        """
        Args:
            full_simulator_class (str): In the form "<module>.<class>" relative to the
                ./dmca_robot/src directory. E.g. "simulations.circle_simulation.CircleSimulator".
            
        """
        init_python2_plot()

        print("Finished setup")
        start_time = time()

        state_names = self.simulator.get_simulation_value_names()
        states = {name: None for name in state_names}
        
        samples_handled = 0
        curr_pos = np.zeros((self.n_robots, 2))
        curr_vel = np.zeros((self.n_robots, 2))

        while not rospy.is_shutdown():
            samples_recorded = min([len(robot.headings) for robot in self.robots])
            
            if samples_recorded > samples_handled:

                # Get the current state of each robot
                for i, robot in enumerate(self.robots):
                    _, pos, vel = robot.get_current_offaxis_state()
                    print("Current {} pos {}, vel {}".format(robot.name, pos, vel))
                    curr_pos[i] = pos
                    curr_vel[i] = vel

                # Calculate the next state
                print("Current total pos {}, vel {}, pos0 {}".format(curr_pos, curr_vel, curr_pos[0]))
                new_state = self.simulator.step_simulation(curr_pos, curr_vel)
                new_pos, new_vel = new_state[:2]
                print("Simulated donnie pos {}, vel {}".format(new_pos[0], new_vel[0]))
                # Save state
                states = self.append_state(states, new_state)

                # Publish the new state to make the robots move
                for i, robot in enumerate(self.robots):
                    robot.publish_offaxis(new_pos[i], new_vel[i])

                samples_handled = samples_recorded
            else:
                self.robots[0].rate.sleep()

        print("Robot {}'s positions: {}".format(self.robots[0].name, self.robots[0].offaxis_positions))
        total_time = time() - start_time
        print("Experiment finished in {:.3}s. Saving data...".format(total_time))

        # Save positions and velocities in case the data needs to be used later
        dir = "outputs/{}".format(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
        if os.path.exists('./dmca_robot'):
            os.chdir('./dmca_robot/src')
        if not os.path.exists(dir):
            os.makedirs(dir)
        
        
        times = np.linspace(0, total_time, states[state_names[0]].shape[0])
        
        # Get states.values() in the same order as state_names
        state_values = [states[name] for name in state_names]
        self.simulator.setup_plot(times, *state_values)

        plt.tight_layout()
        plt.savefig("{}/robot.png".format(dir), dpi=150)
        plt.savefig("{}/robot.pdf".format(dir))
        print("Saved figure")

        self.save_states(dir, states)

    def append_state(self, states, new_state):
        """
        Args:
            states (dict(np.ndarray: shape = [n_times, ...])):
                All previously processed states.
            new_state ((np.ndarray)): Tuple of states
                for the current time-step.
        Returns:
            (dict(np.ndarray: shape = [n_times, ...])): States
        """
        new_state = list(new_state)

        for i, state in enumerate(new_state):
            # Add time dimension
            new_state[i] = np.expand_dims(state, 0)
            
        keys = self.simulator.get_simulation_value_names()
        
        if states[keys[0]] is None:
            for i, key in enumerate(keys):
                states[key] = new_state[i]
        else:
            for i, key in enumerate(keys):
                states[key] = np.concatenate((states[key], new_state[i]), axis=0)
        
        return states

    def save_states(self, dir, states):
        """
        Convert ndarrays to lists so that they can
        be converted to JSON.

        Note: Mangles states
        """
        file_name = "{}/states.json".format(dir)
        print("Saving states to {}".format(file_name))

        for key, value in states.items():
            if type(value) == np.ndarray:
                states[key] = value.tolist()
        
        with open(file_name, "w") as file:
            json.dump(states, file, indent=4)
        
if __name__ == "__main__":
    config_module = "multi_robot_config"
    master = RosMaster(config_module)
    master.run()
    