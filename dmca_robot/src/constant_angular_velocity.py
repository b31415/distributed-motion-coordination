#!/usr/bin/env python2.7
from time import sleep
import rospy
import sys
from math import acos, pi
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import std_msgs.msg as msg
import numpy as np
from helpers.lab_helpers import to_cartesian_velocity, to_unicycle_control_input
import matplotlib.pyplot as plt

robot_positions = np.zeros((1, 2))
def twist_to_cartesian(twist):
    """
    Converts a velocity in twist format to cartesian coordinates
    Args:
        twist (Twist)
    Return:
        (np.ndarray): Array in the form [xdot, ydot]
    """
    v = twist.linear.x
    omega = twist.angular.z
    #phi = get_heading()
    phi = 0
    cartesian_velocity = to_cartesian_velocity([phi], (v, omega))
    return cartesian_velocity

def cartesian_to_twist(cartesian_velocity):
    """
    Args:
        cartesian_velocity (np.ndarray)
    Return:
        (Twist)
    """
    #phi = get_heading()
    phi = 0
    velocity_control_input = to_unicycle_control_input([phi], cartesian_velocity)
    return velocity_control_to_twist(velocity_control_input)

def velocity_control_to_twist(control_input):
    """
    Args:
        control_input (Iterable): Has the form (v, omega)
    """
    twist = Twist()
    twist.linear.x = control_input[0]
    twist.angular.z = control_input[1]
    return twist

def get_heading_and_position(pose):
    """
    Args:
        pose (Odometry)
    Returns:
        (float): Heading angle in radians
        (np.ndarray): (x, y) coordinates
    """
    pose = pose.pose.pose

    # Ignore the z-axis since the robot drives in a flat plane
    position = np.array((pose.position.x, pose.position.y))
    

    # Calculate the heading angle from the quaternion orientation
    # For rotation about the z-axis, 
    # qx = qy = 0 and heading = 2*arcsin(qz) = 2*arccos(w)
    heading = 2 * acos(pose.orientation.w)
    return heading, position

def velocity_subscriber(velocity_msg):
    """
    Args:
        velocity_msg (Twist)
    """
    velocity_cart = twist_to_cartesian(velocity_msg)
    print("Received xdot: {}, ydot: {}".format(velocity_cart[0], velocity_cart[1]))

def velocity_publisher(publisher):
    control_input = (0.4, 1.2)
    velocity_msg = velocity_control_to_twist(control_input)
    publisher.publish(velocity_msg)
    print("Published xdot: {}, ydot: {}".format(control_input[0], control_input[1]))

def pose_subscriber(pose_msg):
    global robot_positions
    heading, position = get_heading_and_position(pose_msg)
    print("Received heading: {}, x: {}, y: {}".format(180.0 * heading / pi, position[0], position[1]))
    robot_positions = np.append(robot_positions, np.expand_dims(position, 0), 0)

if __name__=="__main__":
    print("Python version: {}".format(sys.version))

    rospy.init_node('controller', anonymous=True)
    velocity_pub = rospy.Publisher('/RosAria/cmd_vel', Twist, queue_size=10)
    velocity_sub = rospy.Subscriber('/RosAria/cmd_vel', Twist, velocity_subscriber)
    pose_sub = rospy.Subscriber('/RosAria/pose', Odometry, pose_subscriber)
    rate = rospy.Rate(10)


    print("Finished setup")

    for i in range(100):
    # while not rospy.is_shutdown():
        velocity_publisher(velocity_pub)
        rate.sleep()
        # rospy.spin()

    sleep(10)
    # Chop off artificial initial value used to set the shape
    robot_positions = robot_positions[1:, :]
    print("Robot positions: {}".format(robot_positions))
    plt.title("Robot Positions")
    plt.plot(robot_positions[:, 0], robot_positions[:, 1])
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.savefig("./robot_positions.png", dpi=150)
    print("Saved figure")