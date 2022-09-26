#simulation_class = "simulations.circle_simulation.CircleSimulator"
from simulations.formation_and_goal_seeking import MultiRobotGoalSeeker

simulator = MultiRobotGoalSeeker(
    n_robots = 1,
    goal_positions = [(1, 1)]#, (2, 1)],
)