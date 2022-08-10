
TIGHT_LEGEND_KWARGS = {
    "framealpha" : 0.3,
    "handletextpad" : 0.4,
    "borderaxespad" : 0.3,
    "labelspacing" : 0.3
}

class GeneralSimulationHelper(object):
    def __init__(self,
        max_velocity = 1,
        simulation_time = 30,
        dynamics_time_step = 8e-3,
        goal_centre = None
    ):
        """
        Helper class containing attributes common to many
        simulations involving unicycle robots.
        """
        self.max_velocity = max_velocity
        self.simulation_time = simulation_time
        self.dynamics_time_step = dynamics_time_step
        if goal_centre is not None:
            self.goal_centre = goal_centre