"""The adapters of the real devices, against stand-ins for their libraries: no hardware and no ROS is touched."""

import sys
import types

import numpy as np
import pytest

from imagine2touch.devices.franka_robot import FrankaRobot
from imagine2touch.devices.reskin_sensor import ReSkinSensor
from imagine2touch.geometry import Pose
from imagine2touch.interfaces import Robot, TouchSensor
from imagine2touch.reskin_reading import ReSkinReading


class ArmClient:
    """What the robot_io client is to the adapter: an arm that comes to rest a few states after it was sent somewhere."""

    def __init__(self, server, velocities=False):
        self.server, self.velocities, self.asked, self.moving = server, velocities, 0, 0
        self.joints, self.position, self.orientation = np.zeros(7), np.array([0.4, 0.0, 0.3]), np.array([1.0, 0, 0, 0])

    def get_state(self):
        self.asked += 1
        if self.asked < 3:                              # the first state takes a moment to arrive
            return None
        speed = 0.1 if self.moving > 0 else 0.0
        self.moving -= 1
        self.joints = self.joints + 0.01 * speed
        state = dict(tcp_pos=self.position, tcp_orn=self.orientation, joint_positions=self.joints.copy(), force_torque=np.arange(6.0))
        return dict(state, joint_velocities=np.full(7, speed)) if self.velocities else state

    def move_cart_pos_abs_ptp(self, position, orientation):
        self.position, self.orientation, self.moving = np.asarray(position, float), np.asarray(orientation, float), 3

    def move_async_cart_pos_abs_lin(self, position, orientation):
        self.position, self.orientation = np.asarray(position, float), np.asarray(orientation, float)


@pytest.fixture
def ros(monkeypatch):
    """Stand-ins for rospy and robot_io_ros; returns the node names that were started."""
    started = []
    rospy = types.ModuleType("rospy")
    rospy.core = types.SimpleNamespace(is_initialized=lambda: bool(started))
    rospy.init_node = lambda name, **options: started.append(name)
    client = types.ModuleType("robot_io_ros.robot_io_client")
    client.RobotClient = ArmClient
    for name, module in (("rospy", rospy), ("robot_io_ros", types.ModuleType("robot_io_ros")), ("robot_io_ros.robot_io_client", client)):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    return started


@pytest.mark.parametrize("velocities", [False, True])
def test_franka_robot_moves_the_sensor_and_waits_for_the_arm(ros, velocities):
    robot = FrankaRobot(sensor_in_tcp=Pose(t=[0, 0, 0.02]))
    robot.client.velocities = velocities                # with or without joint velocities in the state
    assert isinstance(robot, Robot) and ros == ["imagine2touch"] and robot.client.server == "/robot_io_ros_server"
    assert robot.joints().shape == (7,) and robot.wrench().tolist() == list(range(6))
    target = Pose.from_normal([0, 0, -1], [0.5, 0.1, 0.2])
    assert robot.move_to(target) and robot.client.moving <= 0                   # it returned once the arm was still
    assert robot.pose().t == pytest.approx(target.t) and robot.pose().normal == pytest.approx(target.normal)
    assert robot.client.position == pytest.approx(target.t - 0.02 * target.normal)      # the tool frame is 2 cm behind the sensor
    robot.step_to(target.t + [0, 0, -0.01], target)
    assert robot.pose().t == pytest.approx(target.t + [0, 0, -0.01])


def test_franka_robot_starts_no_second_node(ros):
    ros.append("the client's own node")
    FrankaRobot()
    assert ros == ["the client's own node"]


def test_reskin_sensor_reads_the_mean_field(monkeypatch):
    class Process:
        def __init__(self, settings):
            self.settings, self.started = settings, False

        def start(self):
            self.started = True

        def get_data(self, num_samples):
            return [ReSkinReading.from_field(np.full(15, float(k))) for k in range(num_samples)]

    driver = types.ModuleType("reskin_sensor")
    driver.ReSkinProcess, driver.ReSkinSettings = Process, lambda **settings: settings
    monkeypatch.setitem(sys.modules, "reskin_sensor", driver)
    sensor = ReSkinSensor(port="/dev/ttyUSB0")
    assert isinstance(sensor, TouchSensor) and sensor.process.started
    assert sensor.process.settings["port"] == "/dev/ttyUSB0" and sensor.process.settings["num_mags"] == 5
    assert sensor.read(3) == pytest.approx(np.full(15, 1.0))                    # the mean of the readings 0, 1 and 2
