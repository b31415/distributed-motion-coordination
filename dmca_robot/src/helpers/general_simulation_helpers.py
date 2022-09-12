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
        # TODO: Rename to goal
        if goal_centre is not None:
            self.goal_centre = goal_centre
            
    @abstractmethod
    def step_simulation(self, positions, velocities):
        """
        Notes:
            Implementations of this method should save or change
            the state of any class attributes so that the output
            will always be the same if step_simulation() is called
            multiple times with the same arguments.
        Returns:
            (new_positions, new_velocities, ...): The desired
                state for the next time-step
        """
        pass
    
    def run_simulation(self, initial_positions, max_control_velocity=None):
        """
        Args:
            initial_positions (np.ndarray)
            max_control_velocity (Optional[Iterable]: length = 2): In the
                form (v, omega). If None, the control velocity won't be
                saturated (clipping may still be applied before converting
                to a control velocity).
        Returns:
            (times, headings, control_velocities, *step_simulation_values) =
            (times, headings, control_velocities, positions, velocities, ...)
                where the ith element of step_simulation_values is an np.ndarray
                corresponding to the ith return value of step_simulation(). The first
                dimension of this array has length len(times) and the remaining
                dimensions are the same as the return value of step_simulation().
                For a given time-step, the first 5 return values effectively
                correspond to INPUTS to step_simulation() and the rest correspond
                to OUTPUTS since their initial values are unknown (i.e. current
                vs next state).
        """
        times = np.arange(0, self.simulation_time, self.dynamics_time_step)[:-1]
        n_times = len(times)
        
        # Initial values
        multi_robot = (len(initial_positions.shape) == 2)
        if multi_robot:
            n_robots = initial_positions.shape[0]
        curr_heading = np.zeros(n_robots) if multi_robot else 0
        curr_pos = initial_positions
        curr_control_vel = np.zeros_like(initial_positions)
        
        # Pre-allocate arrays
        # Note: zero control velocity results in zero cartesian velocity
        # Get the dimensions of the arrays from dummy return values of step_simulation()
        arrays = self.step_simulation(initial_positions, curr_control_vel)
        step_simulation_values = [np.zeros((n_times, *arrays[i].shape)) \
            for i in range(len(arrays))]
        headings = np.zeros((n_times, n_robots)) if multi_robot else np.zeros(n_times)
        control_velocities = np.zeros_like(step_simulation_values[1])
        return_values = (times, headings, control_velocities, *step_simulation_values)
        
        for curr_time in range(n_times):
            # Calculate the next state
            #BUG: Array of ndarrays instead of big ndarray
            curr_vel = to_cartesian_velocity(curr_heading, curr_control_vel, self.L)
            
            other_values = self.step_simulation(curr_pos, curr_vel)
            new_pos, new_vel = other_values[0:2]
            
            new_control_vel = to_control_velocity(curr_heading, new_vel, self.L)
            if max_control_velocity is not None:
                new_control_vel = np.clip(
                    new_control_vel,
                    a_min = -max_control_velocity,
                    a_max = max_control_velocity
                )
            new_heading = curr_heading + self.dynamics_time_step * new_control_vel[1]
            
            # Save the current values
            state = (curr_heading, curr_control_vel, curr_pos, curr_vel, *other_values[2:])
            for i, state_value in enumerate(state):
                return_values[i + 1][curr_time] = state_value
            
            # Update the current state
            curr_heading     = new_heading
            curr_control_vel = new_control_vel
            curr_pos         = new_pos
            curr_vel         = new_vel
             
        return return_values