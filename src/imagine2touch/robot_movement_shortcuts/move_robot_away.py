# from robot_io.robot_interface.iiwa_interface  import IIWAInterface, TCP_RESKIN # Used to initialize robot
import numpy as np
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
from scipy.spatial.transform import Rotation
from src.imagine2touch.utils.utils import (
    euler_from_vector,
    threed_vector_to_homog,
    homog_vector_to_3d,
    WORLD_IN_ROBOT,
    add_ee,
    eulertoquat,
    WCAMERA_IN_TCP,
)
from robot_io.utils.utils import pos_orn_to_matrix


FIXED_ROBOT_ORN = [0, np.pi, 0]  # nice orientation of TCP
AWAY_POSE = [0.45, 0.1, 0.2]  # change to suit where you want the robot to move away


def test_exception(
    point=[-0.02785936, -0.02915366, 0.06036839],
    normal=[-0.49263922, 0.07681479, 0.86683683],
):
    # get orientation vector in world
    negative_normal = -1 * np.asarray(
        normal
    )  # for first sampled point, ignore other samples
    test_point_alignment = negative_normal / np.linalg.norm(
        negative_normal, 2
    )  # normalize the normal :D
    goal_orientation = euler_from_vector(0, test_point_alignment)

    test_point_homog = threed_vector_to_homog(point)
    test_point_homog_in_robot = WORLD_IN_ROBOT.dot(test_point_homog)
    test_point_in_robot = homog_vector_to_3d(test_point_homog_in_robot)
    test_point_in_robot_prepare = test_point_in_robot - 0.08 * test_point_alignment
    prep = add_ee(test_point_in_robot_prepare, goal_orientation)

    T_tcp_in_robot = pos_orn_to_matrix(prep, eulertoquat(goal_orientation))
    cam_view = np.array([-WCAMERA_IN_TCP[:, 3][0], -WCAMERA_IN_TCP[:, 3][1], 0, 1])
    cam_view_in_robot = T_tcp_in_robot.dot(cam_view)
    return cam_view_in_robot, goal_orientation


# utilities
def ping_robot_state_blocking():
    state = None
    while state is None:
        state = robot.get_state()
        print("waiting for state")
    print("got state")


# initialize robot
robot = RobotClient("/robot_io_ros_server")  # name of node in server launch file
ping_robot_state_blocking()

pos = robot.get_state()["tcp_pos"]
# robot.move_cart_pos_abs_ptp([pos[0],pos[1],pos[2]+0.02], robot.get_state()['tcp_orn']) #initial reflex
robot.move_cart_pos_abs_ptp(AWAY_POSE, eulertoquat(FIXED_ROBOT_ORN))
# robot.move_async_cart_pos_abs_lin(AWAY_POSE,eulertoquat(FIXED_ROBOT_ORN))
