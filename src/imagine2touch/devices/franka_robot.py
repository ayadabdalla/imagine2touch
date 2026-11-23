"""
A Franka Emika Panda behind the Robot interface, through the ROS client of robot_io
(https://github.com/lukashermann/robot_io, its catkin package robot_io_ros) that the paper's scripts used.

On the machine that controls the arm, the server of that package talks to the robot:

    roslaunch robot_io_ros panda_server.launch

The server that ran for the paper differs from the public one.  `robot_io_server.patch`, beside this
file, holds the difference; apply it in a clone of robot_io at commit 14cc2ae:

    git apply PATH/TO/imagine2touch/src/imagine2touch/devices/robot_io_server.patch

It names the node `robot_io_ros_server` (the public launch file says `panda_ros_server`), stops the
server from reading the robot's state while a move is being commanded, reports a failed move as a
failed service call, and adds a blocking straight move and publishers of the joint states and the
wrench.  `server` is the name of the node; without the patch, pass `server="/panda_ros_server"`.
The state the server sends holds `tcp_pos`, `tcp_orn`, `joint_positions` and `force_torque`.  The
paper's setup also sent `joint_velocities`, which the public repository does not: without them the
arm counts as still when its joint positions stop changing.

Written from the scripts that recorded the paper's data (commit c2ea2c3 of this repository) and not run on hardware since:
check it against your setup before letting the robot move.
"""

import time

import numpy as np

from imagine2touch.geometry import Pose
from imagine2touch.interfaces import Robot


class FrankaRobot(Robot):
    """`sensor_in_tcp` is the pose of the sensing surface in the tool frame that robot_io reports."""

    def __init__(self, server="/robot_io_ros_server", sensor_in_tcp=None, still=1e-3):
        import rospy
        from robot_io_ros.robot_io_client import RobotClient      # from the catkin workspace that holds robot_io_ros
        if not rospy.core.is_initialized():            # some versions of the client start the node themselves
            rospy.init_node("imagine2touch", anonymous=True)
        self.client, self.still = RobotClient(server), still
        self.sensor_in_tcp = sensor_in_tcp or Pose()
        while self.client.get_state() is None:         # the first state takes a moment to arrive
            time.sleep(0.1)

    def _tcp(self, pose):
        return pose @ self.sensor_in_tcp.inverse()

    def _speed(self, pause=0.1):
        """The joint speeds [rad/s]: as the server reports them, or from two states `pause` apart."""
        state = self.client.get_state()
        if "joint_velocities" in state:
            return np.abs(state["joint_velocities"])
        before = np.asarray(state["joint_positions"], float)
        time.sleep(pause)
        return np.abs(np.asarray(self.client.get_state()["joint_positions"], float) - before) / pause

    def _wait(self):
        while not np.all(self._speed() < self.still):
            time.sleep(0.1)

    def pose(self):
        state = self.client.get_state()
        return Pose.from_quaternion(state["tcp_orn"], state["tcp_pos"]) @ self.sensor_in_tcp

    def joints(self):
        return np.asarray(self.client.get_state()["joint_positions"], float)

    def wrench(self):
        return np.asarray(self.client.get_state()["force_torque"], float)

    def move_to(self, pose, tolerance=0.004):
        tcp = self._tcp(pose)
        try:
            self.client.move_cart_pos_abs_ptp(tcp.t, tcp.quaternion)
        except Exception:                      # libfranka reports small misses as failures
            pass
        self._wait()
        return bool(np.linalg.norm(self.pose().t - pose.t) <= tolerance)

    def step_to(self, position, pose):
        tcp = self._tcp(Pose(pose.R, position))
        self.client.move_async_cart_pos_abs_lin(tcp.t, tcp.quaternion)
