from abc import ABCMeta, abstractmethod
import numpy as np
from helpers.lab_helpers import *

"""
TODO: Add setup_plot(), the plot animation, 
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
        goal_centre = None,
        L = 0.3
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
        self.L = L
        
        if goal_centre is not None:
            self.goal_centre = goal_centre
            
    @abstractmethod
    def step_simulation(self, positions, velocities):
        """
        Returns:
            (new_positions, new_velocities, ...): The desired
                state for the next time-step
        """
        pass
    
    def run_simulation(self, initial_positions):
        """
        Args:
            initial_positions (np.ndarray)
        """
        times = np.arange(0, self.simulation_time, self.dynamics_time_step)[:-1]
        n_times = len(times)
        
        curr_heading = 0
        curr_control_vel = np.zeros_like(initial_positions)
        
        for i in range(n_times):
            curr_vel = to_cartesian_velocity(curr_heading, curr_control_vel, self.L)
            
            new_pos, new_vel, *other_values = self.step_simulation(curr_pos, curr_vel)