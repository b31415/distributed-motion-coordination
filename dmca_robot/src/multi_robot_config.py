import numpy as np
from simulations.formation_and_goal_seeking import MultiRobotGoalSeeker
from config_base import ConfigBase

class Config(ConfigBase):
    def __init__(self):
        
        # super(Config, self).__init__(
        #     node_names = ["donnie", "leonardo"],
        #     headings = [0.0, 0.0],
        #     initial_positions = np.array([
        #         [0, 0], [0, -1]
        #     ])
        # )

        # self.goal_positions = np.array([
        #     [1, 1], [2, 1]
        # ])

        super(Config, self).__init__(
            node_names = ["donnie", "leonardo"],
            headings = [0.0, np.pi/2],
            initial_positions = np.array([
                [0, 0], [0, -2]
            ])
        )

        self.goal_positions = np.array([
            [5, 0], [1, 2]
        ])

    def get_simulator(self):
        return MultiRobotGoalSeeker(
            n_robots = len(self.get_node_names()),
            goal_positions = self.goal_positions,
            dynamics_time_step = 0.1
        )