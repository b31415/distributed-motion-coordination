from time import time
from math import pi
from helpers.lab_helpers import rotation_matrix

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import numpy as np

from helpers.lab_helpers import to_cartesian_velocity, to_control_velocity
from helpers.ros_helpers import extract_pose_info, velocity_control_to_twist

class UnicycleRobot:
    def __init__(self,
        name = 'Robot',
        subscribe_freq = 10,
        log_level = rospy.DEBUG,
        max_control_velocity = (0.1, 1),
        heading_offset = 0,
        coordinate_offset = (0, 0)
    ):
        """
        Notes:
            Each ros node should be configured with a unique name.
        
        Args:
            name (str): Name used for referring to the current robot.
                Required for multi-agent systems.
            subscribe_freq (float): How often to subscribe to the robot's
                pose in Hz. If None, the pose is not subscribed to.
            log_level (Constant): The minimum severity of messages to log.
                rospy.INFO or lower will print verbose info relating to the
                robot's state.
            max_control_velocity (tuple): Maximum value of v and omega respectively.
            heading_offset (float): Angle in radians.
                If the heading and coordinate offsets are zero,
                then the measurements are in the robot's local frame of reference.
                Non-zero values can be given to convert each state to another
                frame of reference (i.e. the frame of the master robot in a
                multi-robot system) each time the state is read and convert back
                immediately prior to publishing so that when getting the current
                state of the robot, it corresponds to the global/non-local frame.
            coordinate offset (tuple[float, float]): See heading_offset.
        """
        self.L = 1
        self.heading_offset = heading_offset
        self.coordinate_offset = np.array(coordinate_offset)

        # Prevents updating the control velocity more often than
        # then the pose is recorded to avoid inaccurate velocity publishes.
        self.msgs_processed = 0

        self.headings = np.array([])
        self.offaxis_positions = np.zeros((0, 2))
        self.control_velocities = np.zeros((0, 2))

        self.subscribe_freq = subscribe_freq
        self.max_control_velocity = max_control_velocity

        rospy.init_node(name, anonymous=False, log_level=log_level)
        self.velocity_pub = rospy.Publisher('{}/cmd_vel'.format(name), Twist, queue_size=10)
        if subscribe_freq:
            self.pose_sub = rospy.Subscriber('{}/pose'.format(name), Odometry, self.pose_subscriber_)
        
        self.rate = rospy.Rate(self.subscribe_freq)
        
        self.last_pose_sub_time_ = None
    
    def publish_next_state(self, get_next_state):
        """
        Calculates the desired next velocity and publishes it.

        Args:
            get_next_state (Callable): A function of the form:
                    get_next_state(
                        curr_pos: ndarray, curr_vel: npdarray
                    ) -> (next_pos: ndarray, next_vel: ndarray)
                which uses an acceleration control law to calculate the 
                next state, where each position and velocity is an off-axis value.
        Returns:
            new_offaxis_pos (np.ndarray)
            new_offaxis_vel (np.ndarray)
            new_control_vel (np.ndarray)
        """
        # Get the most recently measured state
        _, offaxis_pos, offaxis_vel = self.get_current_offaxis_state()

        # Get the desired state for the next time-step from the current state
        new_offaxis_pos, new_offaxis_vel = get_next_state(offaxis_pos, offaxis_vel)

        return self.publish_offaxis(new_offaxis_pos, new_offaxis_vel)

    def publish_offaxis(self, new_offaxis_pos, new_offaxis_vel):
        """
        Converts the desired offaxis values to (v, omega),
        then publishes the control velocity.
        """
        heading = self.headings[-1]
        new_control_vel = to_control_velocity(heading, new_offaxis_vel, self.L)
        self.publish_control_velocity(new_control_vel)

        rospy.logdebug_throttle(2, 
            "Desired next state:\n"
            "\tx: {}, y: {}\n"
            "\tv: {}, omega: {}".format(
                new_offaxis_pos[0], new_offaxis_pos[1],
                new_control_vel[0], new_control_vel[1]
            )
        )
        return new_offaxis_pos, new_offaxis_vel, new_control_vel

    def publish_control_velocity(self, velocity):
        """
        Args:
            velocity (Iterable): The desired velocity to publish
                to the robot in the form (v, omega)
        """
        if type(velocity) != np.ndarray:
            velocity = np.array(velocity)
        
        velocity = np.clip(
            velocity,
            a_min = -self.max_control_velocity,
            a_max = self.max_control_velocity
        )
        
        # Convert velocity to the robot's local coordinate system,
        # giving a vector with the same magnitude but different direction
        rotate = rotation_matrix(-self.heading_offset)
        velocity = np.matmul(rotate, velocity)
        
        velocity_msg = velocity_control_to_twist(velocity)
        self.velocity_pub.publish(velocity_msg)

    def pose_subscriber_(self, pose_msg):
        """Appends the heading, position and velocity to the respective arrays."""
        self.last_pose_sub_time_ = time()
        
        heading, position, velocity = extract_pose_info(pose_msg)
        
        # Convert from the robot's local coordinate system
        # to the global coordinate system
        rotate = rotation_matrix(self.heading_offset)
        position = np.matmul(rotate, position) + self.coordinate_offset
        velocity = np.matmul(rotate, velocity)
        heading += self.heading_offset
        
        rospy.loginfo_throttle(2, "Received robot state:\n"
            "\theading: {}\n"
            "\tx: {}, y: {}\n"
            "\tv: {}, omega: {}".format(
                180.0 * heading / pi,
                position[0], position[1],
                velocity[0], velocity[1]
            )
        )
        self.append_new_state(heading, position, velocity)

    def get_current_offaxis_state(self):
        """
        Returns:
            (float): The most recently measured heading angle.
            (np.ndarray): Most recent off-axis position.
            (np.ndarray): Most recent velocities in the form (xdot_tilde, ydot_tilde).
        """
        # Wait until the latest state has been measured
        while len(self.headings) <= self.msgs_processed:
            self.rate.sleep()
        
        rospy.logdebug_throttle(2,
            "Finished blocking to get the state"
            "({}/{} messages processed)".format(
                self.msgs_processed, len(self.headings)
            )
        )

        heading = self.headings[-1]
        offaxis_pos = self.offaxis_positions[-1]
        control_vel = self.control_velocities[-1]
        offaxis_vel = to_cartesian_velocity(heading, control_vel, self.L)
        return heading, offaxis_pos, offaxis_vel

    def append_new_state(self, heading, offaxis_position, control_velocity):
        """
        For simulation to simulate appending another pose sample.
        """
        # Make the arrays have shape (1, 2) in order to append to
        # arrays with shape (N, 2)
        offaxis_position = np.expand_dims(offaxis_position, 0)
        control_velocity = np.expand_dims(control_velocity, 0)

        self.headings = np.append(self.headings, heading)
        self.offaxis_positions = np.concatenate(
            (self.offaxis_positions, offaxis_position), axis=0
        )
        self.control_velocities = np.concatenate(
            (self.control_velocities, control_velocity), axis=0
        )

    def time_since_pose_subscription(self):
        """
        Get the number of seconds since the last time the
        pose subscriber callback ran.
        """
        return time() - self.last_pose_sub_time_
    