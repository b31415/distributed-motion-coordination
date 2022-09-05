from abc import ABCMeta, abstractmethod

"""
TODO: Add run_simulation(), setup_plot(), the plot animation, 
    FRAME_SKIP, etc. to here.
"""
TIGHT_LEGEND_KWARGS = {
    "framealpha" : 0.3,
    "handletextpad" : 0.4,
    "borderaxespad" : 0.3,
    "labelspacing" : 0.3
}

class GeneralSimulationHelper(object):
    __metaclass__ = ABCMeta
    
    def __init__(self,
        max_velocity = 1,
        simulation_time = 30,
        dynamics_time_step = 8e-3,
        goal_centre = None
    ):
        """
        Args:
            simulation_time (float): How many seconds to 
        Helper class containing attributes common to many
        simulations involving unicycle robots.
        """
        self.max_velocity = max_velocity
        self.simulation_time = simulation_time
        self.dynamics_time_step = dynamics_time_step
        if goal_centre is not None:
            self.goal_centre = goal_centre
            
    @abstractmethod
    def step_simulation(self, positions, velocities):
        """
        Returns:
            The desired state for the next time-step
        """
        pass