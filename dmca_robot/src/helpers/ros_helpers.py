"""Functions to convert between ROS data types and numpy arrays."""
from math import acos, atan2, pi

import numpy as np
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist
def velocity_control_to_twist(control_velocity):
    """
    Args:
        control_velocity (Iterable): Has the form (v, omega)
    """
    twist = Twist()
    twist.linear.x = control_velocity[0]
    twist.angular.z = control_velocity[1]
    return twist

def twist_to_velocity_control(twist):
    """
    Args:
        twist (Twist)
    Returns:
        control_velocity (np.ndarray): Has the form (v, omega)
    """
    v = twist.linear.x
    omega = twist.angular.z
    return np.array([v, omega])

def extract_twist_info(twist):
    """
    Extract velocity from twist
    Args:
        twist (Twist)
    Returns:
        control_velocity (np.ndarray): Has the form (v, omega)
    """
    v = twist.linear.x
    omega = twist.angular.z
    velocity = np.array([v, omega])
    return velocity

def extract_pose_info(pose):
    """
    Args:
        pose (Pose): Contains position and orientation info.
    Returns:
        heading (float): Heading angle in radians
        position (np.ndarray): Off-axis (x, y) coordinates
    """
    # Ignore the z-axis since the robot drives in a flat plane
    position = np.array((pose.position.x, pose.position.y))

    # Calculate the heading (yaw) angle from the quaternion orientation
    q = pose.orientation
    _, _, heading = euler_from_quaternion([q.x, q.y, q.z, q.w])
    return heading, position