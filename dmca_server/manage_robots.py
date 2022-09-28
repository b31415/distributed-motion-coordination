#!/usr/bin/env python
"""
Run from ./dmca_server/ directory
"""
from __future__ import print_function
from abc import ABCMeta

import os
from time import sleep

from fabric import ThreadingGroup as Group
import patchwork.transfers

def connect_robots(master_params, robot_params, server_package_path):
    """
    Preconditions:
        Assumes the path '~/catkin_ws/src/dmca_robot' exists on each robot.
        Assumes anon_bringup is already running on each of the robots.
    Args:
        robot_params (list[RobotParams])
    """
    names = master_params.node_names
    # Iterate through the list of robots to find
    # the master's name, terminating early once found
    master = master_params.master_name
    ros_master_uri = "http://{}:11311".format(master)
    catkin_dir = "~/catkin_ws"
    node_file = "robot_node.py"
    master_file = "robot_master.py"

    with Group(*names) as robots:
        
        # Attempt to ssh to each robot
        try:
            robots.run('ls >1 /dev/null')
        except Exception as e:
            failed_robots = [r.host for r in e.result.failed.keys()]
            print("Could not connect to: {}".format(failed_robots))

        print("Connected to all robots: {}".format(names))

        # Sync the files in <server_package_path> with each robot
        package_path = os.path.join(catkin_dir, "src/dmca_robot/")
        
        if server_package_path is not None:
            print("Syncing files...\r", end="")
            for connection in robots:
                patchwork.transfers.rsync(
                    connection,
                    server_package_path,
                    package_path,
                    exclude=('outputs/'),
                    delete=False
                )
            print("Finished syncing files")
            

        # TODO: Make this line work
        robots.run("export ROS_MASTER_URI={} && anon_bringup &".format(ros_master_uri), disown=True)

        # Build the package
        try:
            # Try build just the required package
            print("Attempting to build package")
            robots.run("cd {} && catkin build dmca_robot".format(catkin_dir))
        except Exception as e:
            # Build everything from scratch
            for connection in list(e.result.failed.keys()):
                print("Couldn't build package so rebuilding everything for {}".format(connection.host))
                connection.run('rm -r {} {}'.format(
                    os.path.join(catkin_dir, "devel"),
                    os.path.join(catkin_dir, "build")
                ))
                connection.run("cd {} && catkin build".format(catkin_dir))

        # Make sure the file is executable
        dir_path = os.path.join(catkin_dir, "src/dmca_robot/src/")
        for file in [node_file, master_file]:
            robots.run('chmod +x {}'.format(os.path.join(dir_path, file)))

        command = "cd {} && source {} && " \
            "rosrun dmca_robot".format(
            os.path.join(catkin_dir, "src/dmca_robot/src/"), 
            os.path.join(catkin_dir, "devel/setup.bash")
        )
        
        print("Running nodes:")
        # for id, robot in enumerate(robots):
        #     if robot.original_host == master:
        #         print("\tRunning master script on {}".format(master))
        #         robot.run("{} robot_master.py {} > output.log".format(command, str(master_params)))

        #     # Get the params for the current robot
        #     params = next(
        #         (r for r in robot_params if robot.original_host == r.name), None
        #     )
            
        #     # Run the robot
        #     print("\tRunning {}".format(robot.original_host))
        #     robot.run(
        #         "{} robot_node.py --id {} "
        #         "{} >> output.log".format(
        #             command, id, str(params)
        #         )
        #     )
        # except Exception as e:
        #     failed_robots = [r.host for r in e.result.failed.keys()]
        #     print("{} failed on robots: {}".format(file_name, failed_robots))
        #     print(e.result.failed)

        sleep(5)

class Params(object):
    __metaclass__ = ABCMeta
    
    def __str__(self):
        params = ''
        for key, value in self.__dict__.items():
            params += '--{} "{}" '.format(key, value)

        return params.strip()

class RobotParams(Params):
    def __init__(self, name, initial_position=(0, 0), heading=0):
        """
        Attributes:
            name (str): Host name of the robot.
            initial_position ((float, float)): Position relative to the ros
                master's coordinate system where the x-axis is given
                by the master's heading.
            heading (float): Heading relative to the ros master.
        """
        self.name = name
        self.initial_position = initial_position
        self.heading = heading
    
class MasterParams(Params):
    def __init__(self, config_file, master_name, node_names):
        """
        Args:
            config_file (str): Name of python file containing the simulation
                class without the file extension.
        """
        self.config_file = config_file
        self.master_name = master_name
        self.node_names = node_names

if __name__ == "__main__":
    """
        Lowercase robot initials for initial
        positions and uppercase for goals:
         -------
        |L      |
        |D      |
        |  d l  |
         -------
    """
    robot_params = [
        RobotParams('donnie')
    ]
    # robot_params = [
    #     RobotParams('donnie'),
    #     RobotParams('leonardo', initial_position=(0, -1), heading=0)
    # ]
    master_params = MasterParams(
        config_file = "multi_robot_config",
        master_name = "donnie",
        node_names = [r.name for r in robot_params]
    )
    
    # The path of the directory on the server (computer running this
    # script) that should be copied/synced to each robot. 
    # If None, directories are left as is without syncing.
    server_package_path = "../dmca_robot/"

    connect_robots(master_params, robot_params, server_package_path)