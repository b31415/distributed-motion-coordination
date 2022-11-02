import numpy as np
from simulations.obstacle_avoidance import ObstacleAvoider
from config_base import ConfigBase

class Config(ConfigBase):
    def __init__(self):

        super(Config, self).__init__(
            node_names = ["mikey", "leonardo", "donnie"]
        )

        self.goal_centre = np.array([1, 1])
        self.goal_heading = 0
        self.obstacle_centres = np.array([[0, 0]])
        self.obstacle_radii = np.array([0.5])
        self.distance_between_wheels = 0.34
        self.wheel_radius = 0.0925

        # Use Mikey as the front of the formation
        self.heading_weights = np.array([1, 0, 0])

    def get_simulator(self, initial_headings, initial_positions):
        return ObstacleAvoider(
            n_robots = len(self.get_node_names()),
            initial_headings = initial_headings,
            initial_positions = initial_positions,
            goal_centre = self.goal_centre,
            goal_heading = self.goal_heading,
            obstacle_centres = self.obstacle_centres,
            obstacle_radii = self.obstacle_radii,
            max_control_velocity = self.max_control_velocity,
            distance_between_wheels = self.distance_between_wheels,
            wheel_radius = self.wheel_radius,
            heading_weights = self.heading_weights,
            dynamics_time_step = 0.1
        )