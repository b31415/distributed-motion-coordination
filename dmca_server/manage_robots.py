#!/usr/bin/env python
from __future__ import print_function

import os
from time import sleep
import argparse

from fabric import ThreadingGroup as Group
import patchwork.transfers

def connect_robots(args):
    """
    Preconditions:
        Assumes the path '~/catkin_ws/src/dmca_robot' exists on each robot.
        Assumes bringup_pioneer is already running on each of the robots.
    Args:
        args: From parser.parse_args()
    """
    names = args.robots
    file_name = args.file_name
    server_package_path = args.server_package_path
    ros_master_uri = "http://{}:11311".format(args.ros_master)

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
        robots.run(
            "cd ~/catkin_ws && "
            "source ~/catkin_ws/devel/setup.bash && "
            "rosrun dmca_robot {} > ./src/output.log".format(file_name))
        # except Exception as e:
        #     failed_robots = [r.host for r in e.result.failed.keys()]
        #     print("{} failed on robots: {}".format(file_name, failed_robots))
        #     print(e.result.failed)

        sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-name", required=True, help="The name of "
        "a python script in the ~/.../<package>/src directory.")
    parser.add_argument("--robots", nargs='+', default=['donnie', 'leonardo'],
        help="The names of the robots to use.")
    parser.add_argument("--ros-master", default="donnie")
    parser.add_argument("--server-package-path", default=None, help="The path of the directory on the "
        "server (computer running this script) that should be copied/synced to each robot. "
        "If None, directories are left as is without syncing.")
    args = parser.parse_args()
    connect_robots(args)