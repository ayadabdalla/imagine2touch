# README: run in test_dummy_data directory
import sys
import time
import numpy as np
from src.imagine2touch.utils.data_utils import safety_three
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
from scipy.spatial.transform import Rotation
from robot_io.utils.utils import pos_orn_to_matrix, matrix_to_pos_orn, euler_to_quat
from src.imagine2touch.utils.utils import (
    WCAMERA_IN_TCP,
    euler_from_vector,
    WORLD_IN_ROBOT,
    eulertoquat,
    FIXED_ROBOT_ORN,
    inverse_transform,
    move_to_pt_with_v_safe,
)
import signal


robot = RobotClient("/robot_io_ros_server")  # name of node in server launch file
FIXED_ROBOT_ORN = FIXED_ROBOT_ORN  # nice orientation of TCP

# robot state#
start_pos = robot.get_state()["tcp_pos"]
start_rot = robot.get_state()["tcp_orn"]

# Transforms and offsets to test
T_tcp_in_robot = robot.get_tcp_pose()
T_robot_in_tcp = inverse_transform(T_tcp_in_robot)

# convert start_pos to homog
# start_pos = np.append(start_pos, 1)

# apply transform to start_pos
# rot_robot_in_tcp = extract_rotation(T_robot_in_tcp)
# T_tcp_in_world = inverse_transform(WORLD_IN_ROBOT).dot(T_tcp_in_robot)

test_point_normal = np.array([0, 0, -1, 1])
goal_orientation = euler_from_vector(0, test_point_normal)
# get z axis from transform Wcamera in tcp
# test_point = np.array(
#     [-WCAMERA_IN_TCP[:, 3][0], -WCAMERA_IN_TCP[:, 3][1], -WCAMERA_IN_TCP[:, 3][2], 1]
# )  # homog test position in tested frame
test_point = [0.05, 0.05, 0.025, 1]
# test_point = T_tcp_in_robot.dot(WCAMERA_IN_TCP.dot(test_point))
# test_point = T_tcp_in_robot.dot(test_point)
test_point = WORLD_IN_ROBOT.dot(test_point)  # test world in robot transform
# test_point = WORLD_IN_ROBOT.dot(inverse_transform(WORLD_IN_MARKER)).dot(test_point) #test world in world marker transform
# test_point = (inverse_transform(ROBOT_IN_MARKER)).dot(test_point) #test robot in robot marker transform
# test_point = utils.add_ee(test_point[:3],goal_orientation) #test sensor offset
# print(test_point)

# e_T_tcp_in_robot = pos_orn_to_matrix(test_point[:3],eulertoquat(goal_orientation))
# OFFSET_TCP=np.array([-WCAMERA_IN_TCP[:,3][0],
#                 -WCAMERA_IN_TCP[:,3][1],
#                 0,1])
# cam_view_in_robot=e_T_tcp_in_robot.dot(OFFSET_TCP) #test wcamera to tcp transform

# stopping signal handler
end = False


def cb_int(*args):
    global end
    end = True


signal.signal(signal.SIGINT, cb_int)

# #test_with_movement
robot.move_cart_pos_abs_ptp(test_point[:3], eulertoquat(FIXED_ROBOT_ORN))

# rotation = Rotation.from_quat(start_rot)

# # Define the default vector (e.g., forward direction)
# default_vector = np.array([0, 0, -1])

# # Rotate the default vector using the quaternion
# aligned_vector = rotation.apply(default_vector)
# # negate aligned vector
# aligned_vector = -aligned_vector

# robot.move_cart_pos_abs_ptp(
#     current_pos_in_robot_up, eulertoquat(FIXED_ROBOT_ORN)
# )
# stopping signal handler
end = False


def cb_int(*args):
    global end
    end = True


# make aligned vector unit vector
# aligned_vector = aligned_vector / np.linalg.norm(aligned_vector)
# current_pos_in_robot_away = start_pos - (0.04 * aligned_vector)
# signal.signal(signal.SIGINT, cb_int)
# restore = move_to_pt_with_v_safe(
#     robot,
#     goal_point=current_pos_in_robot_away,
#     goal_rot=start_rot,
#     p_i=0.0001,
#     end=end,
#     f_max=10,
#     T_max=20,
#     direct=True,
#     direction=aligned_vector,
#     contact=False,
#     goal_distance_tol=0.01,
# )
# print(restore)
if not safety_three(
    robot.get_state()["tcp_pos"][:3],
    eulertoquat(FIXED_ROBOT_ORN),
    [0.0085, 0.0085, 0],
    0.02,
):
    print("won't make a contact with an object")

time.sleep(1)
print(robot.get_tcp_pose())
# robot.move_async_cart_pos_abs_lin(test_point[:3],utils.eulertoquat(FIXED_ROBOT_ORN))
# utils.move_to_pt_with_v_safe(robot,goal_point=test_point[:3],
#                                 goal_rot=utils.eulertoquat(FIXED_ROBOT_ORN),
#                                 p_i=0.01, dt=0.1, end=end,f_z_max=-15, direct=True,
#                                 direction=test_point_normal)
