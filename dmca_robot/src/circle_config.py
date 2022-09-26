import numpy as np
from simulations.circle_simulation import CircleSimulator

# Desired goal relative to the inital position.
goal_centre = np.array((1, 1))

# The maximum linear velocity from the simulation is sqrt(xdot^2 + ydot^2).
# Restrict xdot and ydot so that the maximum simulation velocity from
# simulation doesn't exceed the maximum linear robot velocity.
#max_velocity = self.max_control_velocity[0]/sqrt(2)

r = 1
simulator = CircleSimulator(
    goal_centre = goal_centre,
    r = r,
    dynamics_time_step = 1.0/10.0
)