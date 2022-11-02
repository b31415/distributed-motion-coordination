from abc import ABCMeta, abstractmethod
import numpy as np
from helpers.lab_helpers import *

"""

References:
    [1] TODO: WMR Control Via Dynamic Feedback Linearization
    
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
    def get_simulation_value_names(self):
        """
        Returns:
            (list(str)): Name of each of the elements
                returned from step_simulation()
        Example:
            ["Positions", "Velocities"]
        """
        pass

    @abstractmethod
    def step_simulation(self, positions, velocities, headings=None):
        """
        Notes:
            Implementations of this method should save or change
            the state of any class attributes so that the output
            will always be the same if step_simulation() is called
            multiple times with the same arguments.
        Returns:
            (new_positions, new_velocities, new_accelerations, ...): The desired
                state for the next time-step
        """
        pass
    
    def run_simulation(self, initial_positions, max_control_velocity=None):
        """
        Args:
            initial_positions (Iterable)
            max_control_velocity (Optional[Iterable]: length = 2): In the
                form (v, omega). If None, the control velocity won't be
                saturated (clipping may still be applied before converting
                to a control velocity).
        Returns:
            (times, headings, control_velocities, *step_simulation_values) =
            (times, headings, control_velocities, positions, velocities, accelerations, ...)
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
        initial_positions = np.array(initial_positions)
        multi_robot = (len(initial_positions.shape) == 2)
        if multi_robot:
            n_robots = initial_positions.shape[0]
        curr_heading = np.zeros(n_robots) if multi_robot else 0
        curr_pos = initial_positions
        curr_control_vel = np.zeros_like(initial_positions)
        
        # Pre-allocate arrays
        # Note: zero control velocity results in zero cartesian velocity
        # Get the dimensions of the arrays from dummy return values of step_simulation()
        arrays = self.step_simulation(initial_positions, curr_control_vel, curr_heading)
        step_simulation_values = [np.zeros((n_times,) + arrays[i].shape) \
            for i in range(len(arrays))]
        headings = np.zeros((n_times, n_robots)) if multi_robot else np.zeros(n_times)
        control_velocities = np.zeros_like(step_simulation_values[1])
        return_values = (times, headings, control_velocities) + tuple(step_simulation_values)
        
        for curr_time in range(n_times):
            # Calculate the next state
            curr_vel = to_cartesian_velocity(curr_heading, curr_control_vel, self.L)
            
            other_values = self.step_simulation(curr_pos, curr_vel, curr_heading)
            new_pos, new_vel = other_values[0:2]
            
            new_control_vel = to_control_velocity(curr_heading, new_vel, self.L)
            if max_control_velocity is not None:
                new_control_vel = np.clip(
                    new_control_vel,
                    a_min = -max_control_velocity,
                    a_max = max_control_velocity
                )
            new_heading = curr_heading + self.dynamics_time_step * new_control_vel[..., 1]
            
            # Save the current values
            state = (curr_heading, curr_control_vel, curr_pos, curr_vel) + other_values[2:]
            for i, state_value in enumerate(state):
                return_values[i + 1][curr_time] = state_value
            
            # Update the current state
            curr_heading     = new_heading
            curr_control_vel = new_control_vel
            curr_pos         = new_pos
            curr_vel         = new_vel
             
        return return_values
    
    def trajectory_tracking(self, initial_positions, initial_heading=0, max_control_velocity=None):
        # TODO: Refactor
        # TODO: Handle zero velocity     
        times = np.arange(0, self.simulation_time, self.dynamics_time_step)[:-1]
        n_times = len(times)
        desired_positions = np.zeros((n_times, 2))
        desired_positions[0] = initial_positions
        desired_velocities = np.zeros((n_times, 2))
        desired_velocities[0] = (0.1, 0)
        desired_accelerations = np.zeros((n_times, 2))
        
        # Calculate desired values
        for i in range(1, n_times):
            simulation_values = self.step_simulation(desired_positions[i - 1], desired_velocities[i - 1])[:2]
            desired_positions[i] = simulation_values[0]
            desired_velocities[i] = simulation_values[1]
            desired_accelerations[i] = simulation_values[2]
        
        n_times = desired_positions.shape[0]
        control_velocities = np.zeros((n_times, 2))
        positions = np.zeros((n_times, 2))
        positions[0] = desired_positions[0]
        velocities = np.zeros((n_times, 2))
        curr_heading = initial_heading
        headings = np.zeros(n_times)
        headings[0] = curr_heading
        
        kp = 1
        kd = 1
        v = 0
        
        for i in range(1, n_times):
            desired_pos = desired_positions[i]
            desired_vel = desired_velocities[i]
            desired_accel = desired_accelerations[i]
                
            curr_pos = positions[i - 1]
            curr_vel = velocities[i - 1]
            
            # Calculate control velocity according to equations (8) and (17) in [1]
            u = desired_accel + kp*(desired_pos - curr_pos) + kd*(desired_vel - curr_vel)
            
            linear_accel = u[0]*np.cos(curr_heading) + u[1]*np.sin(curr_heading)
            v += linear_accel * self.dynamics_time_step
            omega = (u[1]*np.cos(curr_heading) - u[0]*np.sin(curr_heading)) / v
            
            control_velocities[i] = (v, omega)
            new_heading = curr_heading + self.dynamics_time_step * omega
            headings[i] = new_heading
            velocities[i] = (v*np.cos(new_heading), v*np.sin(new_heading))
            positions[i] = curr_pos + self.dynamics_time_step * velocities[i]
            
        return times, headings, control_velocities, positions, velocities, desired_positions, desired_velocities
        