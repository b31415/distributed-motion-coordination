from abc import ABCMeta, abstractmethod

class ConfigBase():
    __metaclass__ = ABCMeta

    def __init__(self, node_names, headings = None, initial_positions = None):
        """
        Args:
            headings (list): Angles in radians
        """
        self.node_names = node_names
        self.headings = headings
        self.initial_positions = initial_positions

    def get_node_names(self):
        return self.node_names

    def get_headings(self):
        return self.headings

    def get_initial_positions(self):
        return self.initial_positions

    @abstractmethod
    def get_simulator(self):
        pass
