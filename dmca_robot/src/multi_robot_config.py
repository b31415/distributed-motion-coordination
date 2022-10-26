import numpy as np
from numpy import sqrt
from simulations.formation_and_goal_seeking import MultiRobotGoalSeeker
from config_base import ConfigBase

class Config(ConfigBase):
    def __init__(self):

        super(Config, self).__init__(
            node_names = ["mikey", "leonardo", "donnie"]
        )

        self.goal_positions = np.array([
            [-1, -2], [0, -1], [1, 0]
            #[-1.5, -1.8], [-0.9, -2.08 + sqrt(3)/2], [-0.5, -2]
        ])

    def get_simulator(self):
        return MultiRobotGoalSeeker(
            n_robots = len(self.get_node_names()),
            goal_positions = self.goal_positions,
            dynamics_time_step = 0.1
        )