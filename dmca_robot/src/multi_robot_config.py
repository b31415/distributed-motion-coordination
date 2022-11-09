import numpy as np
from simulations.formation_and_goal_seeking import MultiRobotGoalSeeker
from config_base import ConfigBase

class Config(ConfigBase):
    def __init__(self):

        super(Config, self).__init__(
            node_names = ["mikey", "leonardo", "donnie"]
        )

        self.goal_positions = np.array([
            #[-0.8, -2.1], [-0.2, -1.5], [0.3, -2.33]
            [0.25, 0.5], [-0.05, 1.75], [1, 1.5]
        ])

    def get_simulator(self, *args):
        return MultiRobotGoalSeeker(
            n_robots = len(self.get_node_names()),
            goal_positions = self.goal_positions,
            dynamics_time_step = 0.1
        )