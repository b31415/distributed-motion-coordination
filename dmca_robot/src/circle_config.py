import numpy as np
from simulations.circle_simulation import CircleSimulator

class Config():
    def __init__(self):
        super(Config, self).__init__(
            node_names        = ["donnie"],
            headings          = [0],
            initial_positions = np.array([[0, 0]])
        )

        # Desired goal relative to the inital position.
        self.goal_centre = np.array((1, 1))

        # The maximum linear velocity from the simulation is sqrt(xdot^2 + ydot^2).
        # Restrict xdot and ydot so that the maximum simulation velocity from
        # simulation doesn't exceed the maximum linear robot velocity.
        #max_velocity = self.max_control_velocity[0]/sqrt(2)

        self.r = 1

    def get_simulator():
        return CircleSimulator(
            goal_centre = Config.goal_centre,
            r = Config.r,
            dynamics_time_step = 0.1
        )