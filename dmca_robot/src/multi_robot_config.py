import numpy as np
from simulations.formation_and_goal_seeking import MultiRobotGoalSeeker

# headings = [0.0]
# initial_positions = np.array([
#     [0, 0]
# ])

# simulator = MultiRobotGoalSeeker(
#     n_robots = 1,
#     goal_positions = np.array([(1, 1)])#, (2, 1)],
# )

headings = [0.0, 0.0]
initial_positions = np.array([
    [0, 0], [0, -1]
])

simulator = MultiRobotGoalSeeker(
    n_robots = 2,
    goal_positions = np.array([
        [1, 1], [2, 1]
    ])
)