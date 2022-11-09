from abc import ABCMeta, abstractmethod
import numpy as np

class ConfigBase():
    __metaclass__ = ABCMeta

    def __init__(self, node_names, headings = None, max_control_velocity = np.array([0.1, 1])):
        """
        Args:
            headings (list): Angles in radians
            max_control_velocity (np.ndarray: shape = (2,)): Maximum linear
                and angular velocities in m/s and rad/s, respectively. 
        """
        self.node_names = node_names
        self.headings = headings
        self.max_control_velocity = max_control_velocity

        self.mocap_offsets = [0 for _ in node_names]
        for i, name in enumerate(node_names):
            if name == "donnie":
                self.mocap_offsets[i] = 0.15
            if name == "leonardo":
                self.mocap_offsets[i] = 0.08
            if name == "mikey":
                self.mocap_offsets[i] = 0.08

    def get_node_names(self):
        return self.node_names

    def get_headings(self):
        return self.headings

    def get_mocap_offsets(self):
        return self.mocap_offsets

    def get_max_control_velocity(self):
        return self.max_control_velocity

    @abstractmethod
    def get_simulator(self, initial_headings=None, initial_positions=None):
        pass
