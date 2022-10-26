"""
Constants and helper functions for controlling unicycle robots
that don't involve ROS libraries.

A summary of functions to use for the following velocity conversions*:
    (xdot_tilde, ydot_tilde) -> (v, omega): to_control_velocity()
    (v, omega) -> (xdot_tilde, ydot_tilde): to_cartesian_velocity()
    (xdot, ydot) -> (xdot_tilde, ydot_tilde, v, omega): to_offaxis()
    (v, omega) -> (xdot, ydot): to_onaxis()
    
Functions to use for the following position conversions*():
    x -> x_tilde: to_offaxis()
    x_tilde -> x: to_onaxis()

*See function definitions for actual arguments and return values.

References:
    [1] J. Cortes and M. Egerstedt, "Coordinated control of multi-robot
        systems: A survey", SICE Journal of Control, Measurement, and
        System Integration, vol. 10, no. 6, pp. 495-503, 2017.
"""
import sys
from math import cos, sin, atan2, pi
import numpy as np
from numbers import Number
import matplotlib.pyplot as plt

# Default distance to an arbitrary point
# off the robot's rotation axis
L = 0.5

def get_twist_coefficients(l = L):
    """
    Get the matrix used to convert off-axis velocities
    to control velocities.
    Args:
        l (float): Distance to an arbitrary point off the
            axis of the robot.
    """
    return np.array((
        (1,   0),
        (0, 1/l)
    ))

def init_python2_plot():
    """
    Set the file of the figure to something similar
    to the default for python 3.
    """
    # Only change the default style for python 2
    if sys.version[0] == '3':
        return
    plt.style.use('bmh')
    plt.rc('axes', titlesize=10, labelsize=10)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('legend', fontsize=8)

def rotation_matrix(theta):
    """
    Args:
        theta (float)
    Returns:
        (np.ndarray)
    """
    return np.array((
        (cos(theta), -sin(theta)),
        (sin(theta),  cos(theta))
    ))

def apply_over_times_(func, *args):
    """
    Applies a function over corresponding elements of lists of the same length
    or elements corresponding to a single time.
    
    Args:
        func (Callable): Function in the form func(*single_args)
            where each arg is the value of a variable at a single point in time.
        *args (tuple): Each arg has type float or is an iterable with its first
        dimension having length 1 if all of the args represent a value at a single
        point in time. Otherwise each arg is an iterable with the first dimensions
        having the same length.
        trajectory (Iterable): An np.ndarray of size <number of time steps> by 2 or
            an iterable of length 2.
    Returns:
        (np.ndarray): Shape (N, M) if N > 1, otherwise M where M
            is the shape of the return value of func.
    """
    if len(args) == 0:
        raise ValueError("Requires at least one argument")
    
    # Case1: Each argument contains only on element
    if np.array([isinstance(arg, Number) or \
        np.array(arg).shape[0] == 1 for arg in args]).any():
        
        # Convert any elements in the form (1, M) to (M,)
        args = [arg.squeeze() if isinstance(arg, np.ndarray) else arg for arg in args]
        return func(*args)

    args = [np.array(arg) for arg in args]
    n_times = args[0].shape[0]      
    
    # Case 2: Each argument is an iterable containing
    # values for several time-steps
    if np.array([arg.shape[0] == n_times for arg in args[1:]]).all():
        get_ith_args = lambda i : [arg[i, ...] for arg in args]
        return np.array([func(*get_ith_args(i)) for i in range(n_times)])
    else:
        raise ValueError(
            "The first dimension of each iterable representing the "
            "number of time-steps should have the same length."
        )
    
def to_control_velocity(headings, cartesian_velocities, l = L):
    """
    Convert one or more off-axis velocities of the form
    (xdot_tilde, ydot_tilde) to the form (v, omega).
    
    Args:
        headings (float | Iterable): Direction the robot is facing in
            radians at each time step. Has the same number of
            headings as the number of pairs in <cartesian_velocity>.
        cartesian_velocity (Iterable): An np.ndarray of size <number of time steps>
            by 2 or an iterable of length 2, representing the off-axis
            velocity in cartesian coordinates.
    Returns:
        (np.ndarray)
    """  
    get_control_input = lambda heading, velocity: \
        np.matmul(np.matmul(get_twist_coefficients(l), rotation_matrix(-heading)), velocity)
    
    return apply_over_times_(get_control_input, headings, cartesian_velocities)

def to_cartesian_velocity(headings, control_inputs, l = L):
    """
    Convert one or more off-axis velocities of the form (v, omega)
    to the form (xdot_tilde, ydot_tilde).
    
    Args:
        headings (float | Iterable[float]): Direction the robot is facing in radians.
        control_inputs (Iterable[float]): An np.ndarray of size <number of time steps> by 2 or
            an iterable of length 2, where each pair has the form (v, omega).
    Returns:
        (np.ndarray) Array of off-axis velocities in cartesian coordinates
            with the same shape as control_input.
    """
    get_cartesian_velocity = lambda heading, control_in: \
        np.matmul(rotation_matrix(heading), np.array((control_in[0], l*control_in[1])))
        
    return apply_over_times_(get_cartesian_velocity, headings, control_inputs)

def calculate_headings(
    onaxis_velocities = None,
    positions = None,
    initial_heading = None):
    """
    Determine the headings of the robot, between -pi and pi, for each time step.
    Whilst headings due to rotating on the spot cannot be calculated
    given the on-axis velocities or naxis-positions,
    this function gives the headings required to move with the specified
    velocity, assuming some desired non-zero angular velocity.
    
    Args:
        onaxis_velocities (np.ndarray): On-axis cartesian velocities
            with shape (2,) or (N, 2).
        positions (np.ndarray): Either on or off-axis positions with shape
            (N + 1, 2). If both this and onaxis_velocities are given, the
            velocities are used.
        initial_heading (Optional[float]): Initial direction of the robot
            in radians. If an initial heading is not specified and the first
            velocity is zero, a ZeroDivisionError will be raised.
    Returns:
        headings (np.ndarray | float): A float is return if onaxis_velocities
            has shape (2,). Otherwise, returns an array of length N.
        
    Examples:
    1. Calculate from ona
        >>> v = np.array([0, 0])
        >>> theta = 0.5
        >>> h = calculate_headings(onaxis_velocities=v, initial_heading=theta)
        
        Here h is array([0.5]) because the velocity was zero, so
        the heading did not change from the initial heading.
    """
    if onaxis_velocities is None and positions is not None:
        # The direction of the velocity is important and not the magnitude.
        # So there is no need to divide by the time step between
        # each position.
        onaxis_velocities = np.array(
            [positions[i, :] - positions[i - 1, :] for i in range(1, positions.shape[0])]
        )
        
    is_single_velocity = (len(onaxis_velocities.shape) == 1)
    if is_single_velocity:
        onaxis_velocities = np.expand_dims(onaxis_velocities, 0)
        
    if onaxis_velocities.shape[1] != 2:
        raise ValueError("Velocities should be in the form <number of time steps> by 2")
    
    # Preallocate headings array
    n_times = onaxis_velocities.shape[0]
    headings = np.zeros(n_times + 1)
    if initial_heading is not None:
        headings[0] = initial_heading
    elif (onaxis_velocities[0] == [0, 0]).all():
        raise ZeroDivisionError(
            "The heading is undefined if the initial velocity is "
            "zero and no initial heading is specified."
        )
    
    # Calculate headings
    for i in range(n_times):
        xdot, ydot = onaxis_velocities[i]
        
        if xdot == 0 and ydot == 0:
            # Don't make the robot rotate on the spot
            # when the linear velocity is zero.
            headings[i + 1] = headings[i]
        else:
            # The robot moves in the direction it is facing, so use
            # the velocity to get the heading.
            headings[i + 1] = atan2(ydot, xdot)
    
    # Chop of the initial heading so that headings[i] corresponds
    # to onaxis_velocities[i] instead of onaxis_velocities[i - 1]
    if is_single_velocity:
        return headings[1]
    else:
        return headings[1:]

def calculate_omega_(headings, time_intervals):
    """
    Calculate the time derivate of the heading angle.
    Args:
        headings (Iterable): Has length len(time_intevals) + 1
        time_intervals (Iterable): The ith time interval is defined as the
            time between the ith and i + 1st headings.
    Returns:
        (np.ndarray): Has the same length as time_intevals
    """
    if len(time_intervals) != len(headings) - 1:
        raise ValueError("headings should have one more element than time_intervals.")
    
    # 
    def angle_difference(theta1, theta0):
        """
        Compute the finite difference of the angle, taking into
        account if the angle crosses the negative x-axis.
        """
        angle_diffs = theta1 - theta0 + np.array([-2*pi, 0, 2*pi])
        return angle_diffs[np.argmin(np.abs(angle_diffs))]
    
    omegas = np.array([
        angle_difference(headings[t], headings[t - 1]) / delta_t \
            for t, delta_t in enumerate(time_intervals)
    ])
    
    return omegas
    
def to_offaxis(
    headings,
    positions,
    velocities,
    time_intervals,
    l = L
):
    """
    Convert one or more positions of the centre of the robot over time
    to positions of a point off the axis of the robot and calculate
    the off-axis cartesian velocity.
    
    Args:
        headings (Iterable): Directions the robot is facing in
            radians. headings SHOULD HAVE LENGTH N + 1 where
            headings[i + 1] corresponds to the same time-step as
            positions[i, :], containing an extra heading at the
            start representing an initial or previous heading.  
        positions (np.ndarray): On-axis cartesian positions with shape
            (2,) or (N, 2).
        velocities (np.ndarray): On-axis cartesian velocities with shape
            (2,) or (N, 2). Each pair has the form (xdot, ydot).
        time_intervals (float | Iterable): Time intervals between each heading
            measurement used to calculate the derivate of the headings.
        l (float): Distance to a point off the axis of the robot.
    Returns:
        offaxis_positions (np.ndarray): Off-axis positions in
            the form (x_tilde, y_tilde)
        offaxis_velocities (np.ndarray): Off-axis velocities
            in the form (xdot_tilde, ydot_tilde)
        control_velocities (np.ndarray): Velocities in the form (v, omega)
        
        Each of the return values has shape (N, 2) if N > 1, otherwise (2,)
    """
    # Convert args to numpy arrays
    headings = np.array(headings)
    
    if isinstance(time_intervals, Number):
        time_intervals = np.array([time_intervals])
    
    if positions.shape != velocities.shape:
        raise ValueError("Positions and velocities must have the same shape.")
    
    # Check each array has the required length
    if len(positions.shape) == 1:
        n_times = 1
    else:
        n_times = positions.shape[0]
    
    if len(headings) != (n_times + 1) or len(time_intervals) != n_times:
        raise ValueError(
            "Positions, velocities and time_intervals must "
            "contain N elements and headings must contain N + 1 "
            "elements for some N."
        )
    # Calculate positions
    offsets = l * np.array([(cos(phi), sin(phi)) for phi in headings[1:]])
    offaxis_positions = positions + offsets
    
    # Calculate v and omega, then use the second pair of equations 
    # on page 500 of [1] to calculate the off-axis cartesian velocities
    if len(velocities.shape) == 1:
        velocities = np.expand_dims(velocities, 0)
    v = np.linalg.norm(velocities, axis=1)
    omega = calculate_omega_(headings, time_intervals)
    v = np.expand_dims(v, 1)
    omega = np.expand_dims(omega, 1)
    control_velocities = np.concatenate((v, omega), 1)
    
    offaxis_velocities = to_cartesian_velocity(headings[1:], control_velocities)
    # # Calculate velocities and prepend initial velocity
    # offaxis_velocities = np.array([
    #     (offaxis_positions[t] - offaxis_positions[t - 1]) / DYNAMICS_TIME_STEP \
    #         for t in range(1, offaxis_positions.shape[0])
    # ])
    # offaxis_velocities = np.concatenate((np.zeros((1, 2)), offaxis_velocities))
    if len(positions.shape) == 1:
        return offaxis_positions.squeeze(), offaxis_velocities.squeeze(), control_velocities.squeeze()
    else:
        return offaxis_positions, offaxis_velocities, control_velocities

def to_onaxis(
    headings,
    offaxis_positions,
    offaxis_velocities = None,
    control_velocities = None,
    l = L
):
    """
    Converts off-axis positions and velocities to on-axis values.
    Preconditions:
        One of offaxis_velocities or control_velocities should be given.
        If both are given, the former will be ignored.
    Args:
        headings (float | np.ndarray)
        offaxis_positions (np.ndarray)
        offaxis_velocities (Optional[np.ndarray])
        control_velocities (Optional[np.ndarray])
        l (float): Distance to a point off the axis of the robot.
    Returns:
        onaxis_positions (np.ndarray)
        onaxis_velocities (np.ndarray)
    """
    n_dims = len(offaxis_positions.shape)
    if isinstance(headings, Number):
        headings = np.array([headings])

    if offaxis_velocities is None and control_velocities is None:
        raise Exception(
            "Require either offaxis_velocities or control_velocities "
            "to be given"
        )
    
    # Calculate positions
    angle_projections = np.array([(cos(phi), sin(phi)) for phi in headings])
    onaxis_positions = offaxis_positions - l * angle_projections
    
    # Calculate velocities
    if control_velocities is None:
        control_velocities = to_control_velocity(headings, offaxis_velocities, l)

    # Make control_velocities have shape (N, 2)
    if n_dims == 1:
        control_velocities = np.expand_dims(control_velocities, 0)   
    v = control_velocities[:, 0]
    
    onaxis_velocities = np.array([v[i] * angle_projections[i] for i in range(len(v))])

    # Make the output arrays have the same number of dimensions as the inputs
    if n_dims == 1:
        return onaxis_positions.squeeze(), onaxis_velocities.squeeze()
    else:
        return onaxis_positions, onaxis_positions
    
    