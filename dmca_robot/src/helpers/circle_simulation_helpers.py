"""
Constants and helper functions related to circular control simulation.
"""
from abc import ABCMeta
import numpy as np
from numpy.linalg import norm
from helpers.general_simulation_helpers import GeneralSimulationHelper

def cartesian_to_polar(cartesian):
    """
    Convert cartesian coordinates to polar coordinates.
    Return:
        (np.ndarray)
    """
    x, y = cartesian
    r = np.linalg.norm(cartesian)
    theta = np.arctan2(y, x)
    return np.array((r, theta))


def polar_to_cartesian(polar):
    """
    Args:
        polar (Iterable): A point in polar coordinates
    Return:
        (np.ndarray)
    """
    r, theta = polar
    return np.array((r*np.cos(theta), r*np.sin(theta)))

def expand_and_append(a, b):
    """
    Append b to a where a is an array of points in R^2.
    
    Args:
        a (np.ndarray)
        b (np.ndarray)
    """
    return np.array(np.append(a, np.expand_dims(b, axis=0), axis=0))

# Helpers that rely on a set of attributes
class CircleSimulationHelper(GeneralSimulationHelper):
    __metaclass__ = ABCMeta
    
    def __init__(self,
        r = 1,
        goal_velocity = None,
        goal_centre = np.array([3, 4]),
        goal_angular_velocity = None,
        use_integral_term = False, 
        **general_helper_kwargs
    ):
        super(CircleSimulationHelper, self).__init__(goal_centre=goal_centre, **general_helper_kwargs)
        self.r = r
        self.max_angular_velocity = 1 / self.r
        self.goal_velocity = goal_velocity if goal_velocity else 0.8 * self.max_velocity
        self.goal_angular_velocity = goal_angular_velocity if goal_angular_velocity \
            else self.goal_velocity / self.r
        self.cartesian = False # Use either the cartesian or polar control law
        self.eps = 1e-10          # For avoiding division by 0
        self.radii_errors = 0
        self.use_integral_term = use_integral_term

    def polar_velocity_to_cartesian(self, polar_pos, polar_vel):
        """
        Return:
            (np.ndarray)
        """
        # Unit vector in the direction of the radius of the circle
        r_hat = polar_to_cartesian(polar_pos)
        r_hat = r_hat / max(np.linalg.norm(r_hat), self.eps)
        # Direction of the tangent to the circle
        theta_hat = np.array((-r_hat[1], r_hat[0]))
        r_dot, theta_dot = polar_vel
        # Get the cartesian velocity by adding the radial and tangential
        # components in cartesian coordinates
        r = polar_pos[0]
        return r_dot * r_hat + r * theta_dot * theta_hat
    
    def cartesian_velocity_to_polar(self, cartesian_pos, cartesian_vel):
        """
        Args:
            cartesian_pos (np.ndarray): Current position relative to the
                origin as opposed to the centre of the polar coordinate system.
            caresian_vel (np.ndarray)
        Returns:
            (np.ndarray): Velocity in polar coordinates relative to self.goal_centre
        """
        # Get radial and tangential vectors
        relative_cartesian_pos = cartesian_pos - self.goal_centre
        r_hat = relative_cartesian_pos / max(np.linalg.norm(relative_cartesian_pos), self.eps)
        theta_hat = np.array((-r_hat[1], r_hat[0]))
        
        # Compute the scalar projection of cartesian_vel
        # onto the radial and tangential unit vectors
        r_dot = np.dot(cartesian_vel, r_hat)
        tangential_component = np.dot(cartesian_vel, theta_hat)
        
        # Convert tangential_component from m/s to rad/s
        r, _ = cartesian_to_polar(relative_cartesian_pos)
        theta_dot = tangential_component / max(r, self.eps)
        return np.array([r_dot, theta_dot])
        
    def get_cartesian_control_input(self, position, velocity):
        """
        Calculate the acceleration control input given a position 
        and velocity in cartesian coordinates (equation 2.7 in thesis A report).
        
        Args:
            position (np.ndarray): (x, y) coordinates for the current time step
            velocity (np.ndarray): (xdot, ydot) for the current time step
        Return:
            (np.ndarray)
        """
        x = position
        v = velocity
        xc = x - self.goal_centre
        a = (norm(xc) - self.r) / norm(xc)
        b = v * ((norm(v) - self.goal_velocity) / norm(v)) if norm(v) != 0 else 0
        # Project velocity onto radius vector to dampen in the radial direction
        proj = xc * np.dot(v, xc) / (norm(xc)*norm(v)) if (norm(xc)*norm(v)) != 0 else 0
        return -a*xc - b - proj
        
    def get_polar_control_input(self, polar_position, polar_velocity):
        """
        Calculate the acceleration control input given the current 
        position and velocity in polar coordinates.

        Args:
            polar_position (np.ndarray)
            polar_velocity (np.ndarray)
        Return:
            (np.ndarray)
        """    
        r, _ = polar_position
        r_dot, theta_dot = polar_velocity

        if self.use_integral_term: 
            self.radii_errors += (1/5)*self.dynamics_time_step*(r - self.r)

        return np.array([
            self.r - r - r_dot - self.radii_errors,
            self.goal_angular_velocity - theta_dot
        ])

    def get_goal_set(self):
        """
        Create Nx2 array of (x, y) coordinates for a circle 
        with radius R and centre GOAL_CENTRE.
        Return:
            ((np.ndarray, np.ndarray))
        """
        n_points = 100
        return np.array([self.goal_centre + self.r*np.array([np.cos(theta), np.sin(theta)]) \
            for theta in np.linspace(0, 2*np.pi, n_points)])
