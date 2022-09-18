#!/usr/bin/env python
from __future__ import print_function

import os
from time import sleep
import argparse

from fabric import ThreadingGroup as Group
import patchwork.transfers

def connect_robots(robot_params, server_package_path, simulation_class):
    """
    Preconditions:
        Assumes the path '~/catkin_ws/src/dmca_robot' exists on each robot.
        Assumes bringup_pioneer is already running on each of the robots.
    Args:
        args: From parser.parse_args()
        robot_params (list[RobotParams])
    """
    names = [r.name for r in robot_params]
    # Iterate through the list of robots to find
    # the master's name, terminating early once found
    master = next((r.name for r in robot_params if r.is_master), None)
    ros_master_uri = "http://{}:11311".format(master)

    with Group(*names) as robots:
        
        # Attempt to ssh to each robot
        try:
            robots.run('ls >1 /dev/null')
        except Exception as e:
            failed_robots = [r.host for r in e.result.failed.keys()]
            print("Could not connect to: {}".format(failed_robots))

        print("Connected to all robots: {}".format(names))

        # Sync the files in <server_package_path> with each robot
        package_path = "~/catkin_ws/src/"
        
        if server_package_path is not None:
            package_name = server_package_path.rsplit('/')[-1]
            file_path = os.path.join(package_path, package_name, "src", file_name)
            print("Syncing files...\r", end="")
            for connection in robots:
                patchwork.transfers.rsync(
                    connection,
                    server_package_path,
                    package_path,
                    exclude=('outputs/'),
                    delete=True
                )
            print("Finished syncing files")
            
        else:
            file_path = os.path.join("~/catkin/src/dmca_robot", file_name)

        # TODO: Make this line work
        robots.run("export ROS_MASTER_URI={} && bringup_pioneer &".format(ros_master_uri), disown=True)

        # Build the package
        try:
            # Try build just the required package
            print("Attempting to build package")
            robots.run('cd ~/catkin_ws && catkin build dmca_robot')
        except Exception as e:
            # Build everything from scratch
            for connection in list(e.result.failed.keys()):
                print("Couldn't build package so rebuilding everything for {}".format(connection.host))
                connection.run('rm -r ~/catkin_ws/devel/ ~/catkin_ws/build/')
                connection.run('cd ~/catkin_ws && catkin build')

        # Make sure the file is executable
        robots.run('chmod +x {}'.format(file_path))

        print("Running {}".format(file_name))
        # try:
        for robot in robots:
            if robot.original_host == master:
                file_name = "robot_master.py"
            else:
                file_name = "robot_node.py"

            # Get the params for the current robot
            params = next(
                (r for r in robot_params if robot.original_host == r.name), None
            )
            
            # Run the robot
            robot.run(
                "cd ~/catkin_ws && "
                "source ~/catkin_ws/devel/setup.bash && "
                "rosrun dmca_robot {} _simulation_class:={} "
                "{} > ./src/output.log".format(
                    file_name, simulation_class, str(params)
                )
            )
        # except Exception as e:
        #     failed_robots = [r.host for r in e.result.failed.keys()]
        #     print("{} failed on robots: {}".format(file_name, failed_robots))
        #     print(e.result.failed)

        sleep(5)

class RobotParams:
    def __init__(self, name, position=(0, 0), heading=0, is_master=False):
        """
        Attributes:
            name (str): Host name of the robot.
            position ((float, float)): Position relative to the ros
                master's coordinate system where the x-axis is given
                by the master's heading.
            heading (float): Heading relative to the ros master.
            is_master (bool): Whether the robot is the ros master.
        """
        self.name = name
        self.position = position
        self.heading = heading
        self.is_master = is_master
    
    def __str__(self):
        params = ''
        for key, value in self.__dict__.items():
            params += '_{}:={} '.format(key, value)

        return params.strip()
           
if __name__ == "__main__":
    
    robot_params = [
        RobotParams('donnie', master=True),
        RobotParams('leonardo', position=(0, -1), heading=0) # To the right of Donnie
    ]
    
    # The path of the directory on the server (computer running this
    # script) that should be copied/synced to each robot. If None, directories
    # are left as is without syncing.
    server_package_path = None

    simulation_class = "simulations.formation_and_goal_seeking.MultiRobotGoalSeeker"
    
    connect_robots(robot_params, server_package_path, simulation_class)