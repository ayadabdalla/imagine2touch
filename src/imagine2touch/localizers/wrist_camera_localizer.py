# Import standard libraries
import numpy as np
import pyrealsense2 as rs
import rospy
import hydra
from omegaconf import OmegaConf
import os

# Import repo modules
from robot_io.cams.realsense.realsense import Realsense
from robot_io.utils.utils import pos_orn_to_matrix
from robot_io_ros.src.imagine2touch.robot_io_ros.robot_io_client import RobotClient
from src.imagine2touch.localizers.aruco_cam_robot_world_localizer import (
    FrameUtil,
)  # used to get transforms in the experiment setting
from src.imagine2touch.utils.utils import (
    WORLD_IN_ROBOT,
    inverse_transform,
    FIXED_ROBOT_ORN,
    eulertoquat,
    HOME_POSE,
)
import time
import sys


# custom utils
def generate_step_orientation(roll, tilt_angle, step):
    ORN_array = []
    yaw = 0
    pitch = -tilt_angle + np.pi
    while yaw <= tilt_angle and pitch - np.pi <= 0:
        ORN_array.append([yaw, pitch, roll])
        yaw += step
        pitch += step

    yaw = tilt_angle
    pitch = 0 + np.pi
    while yaw >= 0 and pitch - np.pi <= tilt_angle:
        ORN_array.append([yaw, pitch, roll])
        yaw -= step
        pitch += step

    yaw = 0
    pitch = tilt_angle + np.pi
    while yaw >= -tilt_angle and pitch - np.pi >= 0:
        ORN_array.append([yaw, pitch, 0])
        yaw -= step
        pitch -= step

    # yaw = -tilt_angle
    # pitch = 0 + np.pi
    # while yaw <= 0 and pitch - np.pi >= -tilt_angle:
    #     ORN_array.append([yaw / 3, pitch, 0])
    #     yaw += step
    #     pitch -= step

    return ORN_array


def generate_step_pos(pos, r, initial_angle, orientation_steps, step, translation_step):
    pos_array = []
    for i in range(orientation_steps):
        pos_array.append(
            [
                pos[0] - r * np.cos(initial_angle),
                pos[1] + r * np.sin(initial_angle),
                pos[2],
                1,
            ]
        )
        pos_array.append(
            [
                pos[0] - r * np.cos(initial_angle),
                pos[1] + r * np.sin(initial_angle),
                pos[2] + translation_step,
                1,
            ]
        )
        pos_array.append(
            [
                pos[0] - r * np.cos(initial_angle),
                pos[1] + r * np.sin(initial_angle),
                pos[2] - translation_step,
                1,
            ]
        )
        pos_array.append(
            [
                pos[0] - r * np.cos(initial_angle),
                pos[1] + r * np.sin(initial_angle) + translation_step,
                pos[2],
                1,
            ]
        )
        pos_array.append(
            [
                pos[0] - r * np.cos(initial_angle),
                pos[1] + r * np.sin(initial_angle) - translation_step,
                pos[2],
                1,
            ]
        )
        pos_array.append(
            [
                pos[0] - r * np.cos(initial_angle) + translation_step,
                pos[1] + r * np.sin(initial_angle),
                pos[2],
                1,
            ]
        )
        pos_array.append(
            [
                pos[0] - r * np.cos(initial_angle) - translation_step,
                pos[1] + r * np.sin(initial_angle),
                pos[2],
                1,
            ]
        )
        initial_angle += step
    return pos_array


# constants
WORLD_IN_MARKER = pos_orn_to_matrix(
    [-0.299, 0.018, 0.04], [0, 0, 0.5 * np.pi]
)  # used to move wrist cam to the world marker, origin at pencil drawn co-ordinates
MARKER_IN_WORLD = inverse_transform(WORLD_IN_MARKER)
pose_in_marker = [0.05, 0.05, 0.2, 1]
pose_in_world = [0, 0, 0.2, 1]

if __name__ == "__main__":
    OmegaConf.register_new_resolver("path", lambda x: os.path.abspath(x))
    hydra.initialize("./conf", version_base=None)
    cfg = hydra.compose("wrist.yaml")

    # #initialize devices
    robot = RobotClient("/robot_io_ros_server")  # name of node in server launch file
    camera = Realsense(img_type="rgb_depth")

    start_pos = robot.get_state()["tcp_pos"]
    start_rot = robot.get_state()["tcp_orn"]

    # print(f'current pose {robot.get_state()["tcp_pos"][:3]}')
    i = 0
    transforms = []
    if not cfg.single:
        poses = generate_step_pos(
            pose_in_world,
            0.15,
            0,
            generate_step_orientation(0, np.pi / 4, 0.02),
            0.04,
            0.02,
        )
        orns = generate_step_orientation(0, np.pi / 4, 0.02)
        views = zip(poses, orns)
    else:
        views = zip([[0, 0, 0.1, 1]], [FIXED_ROBOT_ORN])
    for pose, orn in views:
        # marker_origin_in_world_pose = MARKER_IN_WORLD.dot(pose)
        # marker_origin_in_robot_pose = WORLD_IN_ROBOT.dot(marker_origin_in_world_pose)
        world_pose_in_robot = WORLD_IN_ROBOT.dot(pose)
        print(pose)
        print(orn)

        try:
            robot.move_cart_pos_abs_ptp(world_pose_in_robot[:3], eulertoquat(orn))
        except rospy.service.ServiceException as e:
            time.sleep(1)
            if (
                np.linalg.norm(
                    robot.get_state()["tcp_pos"][:3] - world_pose_in_robot[:3], 2
                )
                > 0.01
            ):
                print("failed to get view, pos error is: ")
                print(
                    np.linalg.norm(
                        robot.get_state()["tcp_pos"][:3] - world_pose_in_robot[:3], 2
                    )
                )
                continue
            else:
                print("got view within the following tolerance: ")
                print(
                    np.linalg.norm(
                        robot.get_state()["tcp_pos"][:3] - world_pose_in_robot[:3], 2
                    )
                )
                pass
        #

        # ###############################transforms calculator from scene###########################################
        frame_util = FrameUtil(
            robot, camera
        )  # here you initialized a camera program thread
        frame_util.start()
        rgb, dep = frame_util.camera.get_image()  # here you get the latest image
        # show current image
        # cv2.imshow("image",rgb[:, :, ::-1])
        # cv2.imshow("depth",dep)
        # cv2.waitKey(0)

        # ####################wait for transforms to be calculated###############################################
        while frame_util._T_camera_in_socket is None:
            print("transform is none")
            continue
        print("transform calculated")
        time.sleep(0.1)
        T_tcp_in_robot = robot.get_tcp_pose()
        T_wcamera_in_tcp = inverse_transform(T_tcp_in_robot).dot(
            WORLD_IN_ROBOT.dot(frame_util._T_camera_in_socket)
        )
        # #####################################################################################################

        # ########Save transform###############
        frame_util.stop()
        frame_util.wait()
        transforms.append(T_wcamera_in_tcp)
        print(f"transform {i} recorded")
        i = i + 1
        # #####################################
    transforms = np.mean(transforms, axis=0)
    np.save(
        f"{cfg.save_directory}/wcamera_tcp_transform", transforms, allow_pickle=True
    )
    print("saved_transform")
